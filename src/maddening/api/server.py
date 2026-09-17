"""
SimulationServer -- FastAPI + WebSocket server wrapping a GraphManager.

Provides REST endpoints for graph construction, validation, compilation,
state inspection, simulation stepping/running, parameter tuning, and
surrogate training, plus WebSocket endpoints that stream state snapshots
in real time (JSON or binary).

Usage
-----
    from maddening.api.server import SimulationServer
    from maddening.nodes import BallNode, TableNode

    server = SimulationServer(
        node_registry={"BallNode": BallNode, "TableNode": TableNode},
    )
    app = server.create_app()

    # Run with: uvicorn module:app
    # Or programmatically:
    #   import uvicorn
    #   uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel
except ImportError as _exc:
    raise ImportError(
        "The MADDENING API server requires 'fastapi' and 'pydantic'. "
        "Install them with:  pip install fastapi uvicorn"
    ) from _exc

_STATIC_DIR = Path(__file__).parent / "static"

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# JAX -> JSON helpers
# ------------------------------------------------------------------

def _jax_to_python(value: Any) -> Any:
    """Recursively convert JAX arrays to plain Python types for JSON."""
    if isinstance(value, dict):
        return {k: _jax_to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jax_to_python(v) for v in value]
    if hasattr(value, "shape"):
        if value.shape == ():
            return value.item()
        return value.tolist()
    return value


def _python_to_jax(value: Any) -> Any:
    """Convert plain Python numbers/lists back to JAX arrays."""
    if isinstance(value, dict):
        return {k: _python_to_jax(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return jnp.array(value, dtype=jnp.float32)
    if isinstance(value, (int, float)):
        return jnp.array(value, dtype=jnp.float32)
    return value


# ------------------------------------------------------------------
# Pydantic request/response models
# ------------------------------------------------------------------

class AddNodeRequest(BaseModel):
    type: str
    name: str
    timestep: float
    params: dict[str, Any] = {}


class AddEdgeRequest(BaseModel):
    source_node: str
    target_node: str
    source_field: str
    target_field: str


class RemoveEdgeRequest(BaseModel):
    source_node: str
    target_node: str
    source_field: str
    target_field: str


class SetNodeStateRequest(BaseModel):
    state: dict[str, Any]


class SetNodeParamsRequest(BaseModel):
    params: dict[str, Any]


class TrainSurrogateRequest(BaseModel):
    node_name: str
    n_data_steps: int = 500
    n_epochs: int = 100
    hidden_sizes: list[int] = [64, 64]
    batch_size: int = 64


# ------------------------------------------------------------------
# SimulationServer
# ------------------------------------------------------------------

def _non_finite_param(value: Any, path: str = "") -> Optional[str]:
    """The path of the first non-finite number inside *value*, else ``None``.

    A constructor constant is not validated by the node itself, and a NaN or
    an infinity in one is not merely a bad simulation: every endpoint that
    reports it serialises with ``allow_nan=False``, so the node's own 201
    response -- and every later ``GET /graph`` -- fails inside the encoder.
    Recurses into lists and dicts because a param may be an array.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            # A JSON integer literal is an unbounded Python int; one too big
            # for a float is as unusable as an infinity and must not raise
            # OverflowError here, which would be the 500 this check exists
            # to prevent.
            usable = math.isfinite(float(value))
        except (OverflowError, ValueError):
            usable = False
        return None if usable else (path or "value")
    if isinstance(value, dict):
        for key, item in value.items():
            found = _non_finite_param(item, f"{path}.{key}" if path else str(key))
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _non_finite_param(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _dry_run_node(node) -> None:
    """Abstractly trace one ``update`` of a freshly built node on its own
    initial state with zero boundary inputs: catches constants of the
    wrong type / shape before the node enters the graph."""
    import jax  # noqa: PLC0415

    state = node.initial_state()
    bi = {}
    try:
        for name, spec in (node.boundary_input_spec() or {}).items():
            bi[name] = jnp.zeros(tuple(getattr(spec, "shape", ()) or ()), jnp.float32)
    except Exception:  # noqa: BLE001 - descriptor is advisory
        bi = {}
    jax.eval_shape(lambda: node.update(state, bi, node.delta_t))


def _graph_structure_snapshot(gm) -> dict:
    """Shallow copies of every container a structural edit mutates.

    Nodes, edges and parameter arrays are shared with the live graph --
    ``add_node`` / ``remove_node`` / ``add_edge`` rebind the containers
    rather than mutating their contents, so copying one level is enough
    to put the graph back exactly as it was.  Restoring is what makes an
    endpoint that rebuilds a subgraph atomic: a failed rebuild leaves the
    caller with an error and the server with a graph that still compiles.
    """
    return {
        "nodes": dict(gm._nodes),
        "state": dict(gm._state),
        "edges": list(gm._edges),
        "external_inputs": list(gm._external_inputs),
        "param_spec_overrides": {k: dict(v) for k, v in gm._param_spec_overrides.items()},
        "params": {k: (dict(v) if isinstance(v, dict) else v)
                   for k, v in gm.params.items()},
        "dirty": gm._dirty,
    }


def _restore_graph_structure(gm, snapshot: dict) -> None:
    """Undo every structural edit made since :func:`_graph_structure_snapshot`."""
    gm._nodes = snapshot["nodes"]
    gm._state = snapshot["state"]
    gm._edges = snapshot["edges"]
    gm._external_inputs = snapshot["external_inputs"]
    gm._param_spec_overrides = snapshot["param_spec_overrides"]
    gm.params = snapshot["params"]
    gm._dirty = snapshot["dirty"]


class SimulationServer:
    """Wraps a ``GraphManager`` with a FastAPI HTTP + WebSocket interface.

    Parameters
    ----------
    node_registry : dict[str, type]
        Maps node type name strings to SimulationNode subclasses.
    graph_manager : GraphManager, optional
        An existing GraphManager to serve.  If ``None``, an empty one is
        created.
    frame_renderer : ServerFrameRendererBase, optional
        If provided, enables the ``/ws/render`` WebSocket endpoint that
        streams server-side rendered frames to thin browser clients.
        Any renderer implementing ``ServerFrameRendererBase`` works.
    """

    def __init__(
        self,
        node_registry: dict[str, type[SimulationNode]],
        graph_manager: Optional[GraphManager] = None,
        checkpoint_root: Optional[str] = None,
        frame_renderer: Optional[Any] = None,
    ) -> None:
        self.registry = dict(node_registry)
        self.gm = graph_manager if graph_manager is not None else GraphManager()
        # /checkpoint/{save,load} only touch files under this directory
        # (an unauthenticated client must not choose arbitrary server
        # paths).  Bind the server to localhost or put it behind auth.
        self.checkpoint_root = Path(checkpoint_root or Path.cwd() / "checkpoints").resolve()
        self.relay = StateRelay()
        self.runner: Optional[RealtimeRunner] = None
        self._runner_started = False
        self._relay_attached = False
        # Eagerly attach relay when a pre-built graph is provided
        if graph_manager is not None:
            try:
                self.relay.attach(self.gm)
                self._relay_attached = True
            except RuntimeError:
                pass
        # Surrogate training state
        self._surrogate_jobs: dict[str, dict] = {}
        self._original_nodes: dict[str, tuple] = {}  # name -> (node, edges, ext_inputs)
        self._active_surrogates: set[str] = set()
        # Binary encoder (lazily initialised)
        self._binary_encoder = None
        # Server-side frame renderer
        self._frame_renderer = frame_renderer
        # Cloud session (set externally or via /cloud/launch)
        self._cloud_session = None
        # Last JAX trace directory (set by /sim/profile/jax/stop) so
        # CloudSession teardown can pick it up.
        self._last_jax_trace_dir: Optional[str] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_relay_attached(self) -> None:
        if self._relay_attached:
            return
        try:
            self.relay.attach(self.gm)
            self._relay_attached = True
        except RuntimeError:
            pass

    def _ensure_runner(self) -> RealtimeRunner:
        if self.runner is None:
            self._ensure_relay_attached()
            self.runner = RealtimeRunner(self.gm, self.relay)
        return self.runner

    def _state_json(self) -> dict:
        return _jax_to_python(self.gm._state)

    def _node_state_json(self, name: str) -> dict:
        return _jax_to_python(self.gm.get_node_state(name))

    def _stop_runner(self) -> None:
        """Stop the runner if it's running. Safe to call multiple times."""
        if self.runner is not None and self._runner_started:
            self.runner.stop()
            self._runner_started = False
            self.runner = None

    def _reset_state(self) -> None:
        """Reset all nodes to their initial state (normalised, no retrace)."""
        self.gm.reset_state()
        # Reset relay counters
        with self.relay._lock:
            self.relay._step_count = 0
            self.relay._snapshot = None
        # Invalidate binary encoder (state shape may have changed)
        self._binary_encoder = None

    def _get_binary_encoder(self):
        """Lazily build a BinaryStateEncoder from the current state."""
        if self._binary_encoder is None:
            from maddening.api.binary_encoder import BinaryStateEncoder
            user_state = {
                k: v for k, v in self.gm._state.items() if k != "_meta"
            }
            self._binary_encoder = BinaryStateEncoder(user_state)
        return self._binary_encoder

    # ------------------------------------------------------------------
    # App factory
    # ------------------------------------------------------------------

    def create_app(self) -> FastAPI:
        """Build and return the FastAPI application."""
        app = FastAPI(
            title="MADDENING Simulation Server",
            description="HTTP/WebSocket API for the MADDENING simulation graph.",
            version="0.3.0",
        )

        # -- visualization endpoints -----------------------------------------

        @app.get("/viz/graph", tags=["viz"], response_class=HTMLResponse)
        def viz_graph():
            html_path = _STATIC_DIR / "graph.html"
            return HTMLResponse(content=html_path.read_text(), status_code=200)

        @app.get("/viz/app", tags=["viz"], response_class=HTMLResponse)
        def viz_app():
            """Serve the interactive simulation app."""
            html_path = _STATIC_DIR / "app.html"
            return HTMLResponse(content=html_path.read_text(), status_code=200)

        @app.get("/viz/render", tags=["viz"], response_class=HTMLResponse)
        def viz_render():
            """Serve the server-side rendered viewer."""
            html_path = _STATIC_DIR / "render.html"
            return HTMLResponse(content=html_path.read_text(), status_code=200)

        # -- graph structure endpoints ---------------------------------------

        @app.get("/graph", tags=["graph"])
        def get_graph():
            # Display, not persistence: a mapping without point references
            # is shown as far as it describes itself rather than refused.
            data = self.gm.to_dict(strict_mappings=False)
            data["active_surrogates"] = list(self._active_surrogates)
            return data

        @app.post("/graph/nodes", tags=["graph"], status_code=201)
        def add_node(req: AddNodeRequest):
            if req.type not in self.registry:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown node type '{req.type}'. Available: {list(self.registry.keys())}",
                )
            # A non-finite constant is refused here, exactly as
            # PUT /graph/state and PUT /graph/params refuse one: it survives
            # construction and the trace, and only blows up in the JSON
            # encoder -- after the node is already in the graph, which leaves
            # GET /graph answering 500 for the life of the process.
            bad = _non_finite_param(req.params)
            if bad is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"params.{bad}: value must be finite",
                )
            node_cls = self.registry[req.type]
            try:
                node = node_cls(name=req.name, timestep=req.timestep, **req.params)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            # Nodes do not validate their constants; a bad one only fails
            # inside the trace and would wedge every later /sim/step.
            # Trace one update on the node's own initial state (abstractly,
            # no compute) before it enters the graph.
            try:
                _dry_run_node(node)
            except Exception as exc:  # noqa: BLE001 - any trace failure is a 400
                raise HTTPException(
                    status_code=400,
                    detail=f"node '{req.name}' cannot run with these params: {exc}",
                )
            if req.name in self.gm._nodes:
                raise HTTPException(status_code=409, detail=f"Node '{req.name}' already exists in the graph.")
            try:
                self.gm.add_node(node)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "ok", "node": node.to_dict()}

        @app.delete("/graph/nodes/{name}", tags=["graph"])
        def remove_node(name: str):
            try:
                self.gm.remove_node(name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            return {"status": "ok"}

        @app.post("/graph/edges", tags=["graph"], status_code=201)
        def add_edge(req: AddEdgeRequest):
            for n in (req.source_node, req.target_node):
                if n not in self.gm._nodes:
                    raise HTTPException(status_code=404, detail=f"No node '{n}'.")
            src = self.gm._nodes[req.source_node].node
            fields = set(self.gm.get_node_state(req.source_node))
            try:
                fields |= set(src.boundary_flux_spec())
            except Exception:  # noqa: BLE001 - optional descriptor
                pass
            if req.source_field not in fields:
                raise HTTPException(
                    status_code=400,
                    detail=f"'{req.source_node}' has no field or flux '{req.source_field}'. "
                           f"Available: {sorted(fields)}",
                )
            try:
                self.gm.add_edge(
                    source=req.source_node, target=req.target_node,
                    source_field=req.source_field, target_field=req.target_field,
                )
            except (ValueError, KeyError) as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "ok"}

        @app.delete("/graph/edges", tags=["graph"])
        def remove_edge(req: RemoveEdgeRequest):
            self.gm.remove_edge(
                source=req.source_node, target=req.target_node,
                source_field=req.source_field, target_field=req.target_field,
            )
            return {"status": "ok"}

        @app.post("/graph/compile", tags=["graph"])
        def compile_graph():
            try:
                self.gm.compile()
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "ok", "schedule": self.gm.schedule}

        @app.post("/graph/validate", tags=["graph"])
        def validate_graph():
            issues = self.gm.validate()
            return {"issues": issues}

        # -- state endpoints -------------------------------------------------

        @app.get("/graph/state", tags=["state"])
        def get_state():
            return self._state_json()

        @app.get("/graph/state/{node_name}", tags=["state"])
        def get_node_state(node_name: str):
            try:
                return self._node_state_json(node_name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))

        @app.put("/graph/state/{node_name}", tags=["state"])
        def set_node_state(node_name: str, req: SetNodeStateRequest):
            """Replace a node's state.  Every field is required, coerced to
            the live leaf's dtype, and must match its shape and be finite;
            a 400 names the field and writes nothing."""
            if node_name not in self.gm._nodes:
                raise HTTPException(status_code=404, detail=f"No node '{node_name}'.")
            live = self.gm.get_node_state(node_name)
            if set(req.state) != set(live):
                raise HTTPException(
                    status_code=400,
                    detail=f"state fields must be exactly {sorted(live)}; got {sorted(req.state)}",
                )
            staged = {}
            for field, value in req.state.items():
                want = jnp.asarray(live[field])
                try:
                    arr = jnp.asarray(value, dtype=want.dtype)
                except (TypeError, ValueError) as exc:
                    raise HTTPException(status_code=400, detail=f"{field}: {exc}")
                if arr.shape != want.shape:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{field}: expected shape {want.shape}, got {arr.shape}",
                    )
                if jnp.issubdtype(arr.dtype, jnp.inexact) and not bool(jnp.all(jnp.isfinite(arr))):
                    raise HTTPException(status_code=400, detail=f"{field}: value must be finite")
                staged[field] = arr
            self.gm.set_node_state(node_name, staged)
            return {"status": "ok"}

        # -- parameter endpoints ---------------------------------------------

        @app.get("/graph/params/{node_name}", tags=["params"])
        def get_node_params(node_name: str):
            if node_name not in self.gm._nodes:
                raise HTTPException(status_code=404, detail=f"No node '{node_name}'.")
            node = self.gm._nodes[node_name].node
            # The live pytree wins over the constructor value: it is what
            # the step uses after a fit or a checkpoint restore.
            live = self.gm.params.get("nodes", {}).get(node_name) or {}
            return {**_jax_to_python(node.params), **_jax_to_python(live)}

        @app.put("/graph/params/{node_name}", tags=["params"])
        def set_node_params(node_name: str, req: SetNodeParamsRequest):
            """Update node parameters.

            Float parameters of nodes that accept injected params are
            written to ``gm.params`` and take effect on the next step
            without recompiling; anything else (structural values, nodes
            on the legacy contract) marks the graph dirty as before.
            """
            if node_name not in self.gm._nodes:
                raise HTTPException(status_code=404, detail=f"No node '{node_name}'.")
            node = self.gm._nodes[node_name].node
            live = self.gm.params.get("nodes", {}).get(node_name) or {}
            specs = self.gm.param_specs().get("nodes", {}).get(node_name, {})
            # Before the first compile ``gm.params`` is empty; validate a
            # params node against its own pytree so the same request gets
            # the same answer one compile() earlier or later.
            probe_only = False
            if not live and getattr(node, "accepts_params", lambda: False)():
                live = dict(node.params_pytree())
                probe_only = True
            # A key is addressable if it is a constructor param *or* a leaf
            # of the live pytree (surrogate weights, sharded wrappers whose
            # inner node owns the params).
            available = sorted(set(node.params) | set(live))
            # Validate everything before mutating anything: an
            # out-of-bounds slider value is a 400 here, not a NaN later.
            # Every check a live leaf needs (dtype coercion, shape,
            # finiteness, bounds) runs in this first loop; the second
            # loop only writes, so a 400 on the third key of a request
            # leaves the first two untouched too.
            staged: dict[str, Any] = {}
            for key, value in req.params.items():
                if key not in node.params and key not in live:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unknown param '{key}' for node '{node_name}'. "
                               f"Available: {available}",
                    )
                if isinstance(value, bool):
                    # only a genuinely boolean constructor param takes a bool;
                    # for a numeric leaf it would become 1.0 / 0.0 and drop
                    # out of the params pytree at the next recompile
                    if key in live or not isinstance(node.params.get(key), bool):
                        raise HTTPException(
                            status_code=400,
                            detail=f"{key}: expected a number, got a boolean",
                        )
                    continue
                if key not in live:
                    continue
                try:
                    new = jnp.asarray(value, dtype=live[key].dtype)
                except (TypeError, ValueError) as exc:
                    # A string for a float, ``null``, a ragged list, ...
                    raise HTTPException(status_code=400, detail=f"{key}: {exc}")
                if new.shape != live[key].shape:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{key}: expected shape {live[key].shape}, got {new.shape}",
                    )
                # NaN passes every bounds comparison; reject it here so it
                # is never written into gm.params (a recompile would bake
                # it into the step).
                if not bool(jnp.all(jnp.isfinite(new))):
                    raise HTTPException(
                        status_code=400, detail=f"{key}: value must be finite, got {value!r}",
                    )
                spec = specs.get(key)
                if spec is not None:
                    try:
                        spec.check(new, name=key)
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc))
                staged[key] = new
            for key, value in req.params.items():
                if key in staged:
                    # After a compile ``live`` *is* gm.params' leaf dict and
                    # this is the write; before one it is the throwaway probe
                    # copy, and this only keeps the echo below honest -- it
                    # used to report the pre-write value, contradicting the
                    # GET that follows it.
                    live[key] = staged[key]
                    if key in node.params:
                        # Store the constructor's Python type, never the
                        # raw JSON value: a JSON ``40`` for a float leaf
                        # would turn it into an ``int`` that
                        # ``params_pytree`` no longer exposes.
                        node.params[key] = np.asarray(staged[key]).tolist()
                else:
                    node.params[key] = value
                    self.gm._dirty = True
            shown = {**_jax_to_python(node.params), **_jax_to_python(live)}
            return {"status": "ok", "params": shown}

        # -- checkpoint endpoints -------------------------------------------

        def _checkpoint_path(name: str) -> Path:
            root = self.checkpoint_root
            target = (root / name).resolve()
            if root != target and root not in target.parents:
                raise HTTPException(
                    status_code=400,
                    detail=f"checkpoint path must stay under {root} (got {name!r})",
                )
            return target

        @app.post("/checkpoint/save", tags=["checkpoint"])
        def checkpoint_save(path: str = "checkpoint.npz"):
            target = _checkpoint_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                saved = self.gm.save_state(str(target))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"could not save checkpoint: {exc}")
            return {"status": "ok", "path": str(saved or target)}

        @app.post("/checkpoint/load", tags=["checkpoint"])
        def checkpoint_load(path: str = "checkpoint.npz"):
            target = _checkpoint_path(path)
            if not target.exists() and not target.with_suffix(target.suffix + ".npz").exists():
                raise HTTPException(status_code=404, detail=f"no checkpoint {path!r}")
            try:
                self.gm.load_state(str(target))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception:  # noqa: BLE001 - do not leak file/parse internals
                raise HTTPException(status_code=400, detail=f"could not load checkpoint {path!r}")
            return {"status": "ok", "state": self._state_json()}

        # -- simulation control endpoints -----------------------------------

        @app.post("/sim/step", tags=["sim"])
        def sim_step():
            try:
                self.gm.step()
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return self._state_json()

        @app.post("/sim/run", tags=["sim"])
        def sim_run(n_steps: int = 100):
            try:
                self.gm.run(n_steps)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return self._state_json()

        @app.post("/sim/start", tags=["sim"])
        def sim_start():
            runner = self._ensure_runner()
            if self._runner_started:
                raise HTTPException(status_code=409, detail="Runner is already started.")
            try:
                self._ensure_relay_attached()
                runner.start()
                self._runner_started = True
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "started"}

        @app.post("/sim/pause", tags=["sim"])
        def sim_pause():
            if self.runner is None or not self._runner_started:
                raise HTTPException(status_code=409, detail="Runner is not started.")
            self.runner.pause()
            return {"status": "paused"}

        @app.post("/sim/resume", tags=["sim"])
        def sim_resume():
            if self.runner is None or not self._runner_started:
                raise HTTPException(status_code=409, detail="Runner is not started.")
            self.runner.resume()
            return {"status": "resumed"}

        @app.post("/sim/stop", tags=["sim"])
        def sim_stop():
            if self.runner is None or not self._runner_started:
                raise HTTPException(status_code=409, detail="Runner is not started.")
            self._stop_runner()
            return {"status": "stopped"}

        @app.post("/sim/reset", tags=["sim"])
        def sim_reset():
            """Stop the runner and reset all nodes to initial state."""
            was_running = self._runner_started
            self._stop_runner()
            self._reset_state()
            self.gm._dirty = True
            return {"status": "ok", "was_running": was_running, "state": self._state_json()}

        @app.put("/sim/stride", tags=["sim"])
        def sim_set_stride(steps_per_frame: int = 1, relay_stride: int = 1):
            """Adjust physics-to-render rate decoupling.

            Parameters
            ----------
            steps_per_frame : int
                Physics steps batched per wall-clock frame in the runner.
            relay_stride : int
                Only capture every Nth step in the relay (reduces observer
                overhead for very fast physics).
            """
            if self.runner is not None:
                self.runner.steps_per_frame = steps_per_frame
            self.relay.stride = relay_stride
            return {
                "steps_per_frame": steps_per_frame,
                "relay_stride": relay_stride,
            }

        # -- surrogate endpoints --------------------------------------------

        @app.post("/surrogate/train", tags=["surrogate"])
        def surrogate_train(req: TrainSurrogateRequest):
            """Start training a surrogate in a background thread."""
            try:
                from maddening.surrogates.dataset import DatasetGenerator
                from maddening.surrogates.training.trainer import SurrogateTrainer
                from maddening.surrogates.architectures.mlp import MLPDirect
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="Surrogate training requires equinox+optax. "
                           "pip install maddening[surrogates]",
                )

            if req.node_name not in self.gm._nodes:
                raise HTTPException(status_code=404, detail=f"No node '{req.node_name}'.")

            job_id = str(uuid.uuid4())[:8]
            job = {
                "id": job_id,
                "node_name": req.node_name,
                "status": "running",
                "epoch": 0,
                "n_epochs": req.n_epochs,
                "train_loss": None,
                "val_loss": None,
                "result": None,
                "error": None,
            }
            self._surrogate_jobs[job_id] = job

            def _train_worker():
                try:
                    # Stop runner if going, generate data from varied ICs
                    self._stop_runner()
                    self._reset_state()
                    self.gm._dirty = True

                    # Generate diverse training data using sweep with
                    # varied initial conditions for the target node
                    target_node = self.gm._nodes[req.node_name].node
                    target_init = target_node.initial_state()

                    n_conditions = 16
                    steps_per_condition = min(200, req.n_data_steps)
                    key = jax.random.PRNGKey(42)

                    # Build batched initial states -- broadcast non-target
                    # nodes, vary target node's scalar fields
                    batched = {}
                    for name, spec in self.gm._nodes.items():
                        node_init = spec.node.initial_state()
                        batched[name] = {
                            k: jnp.broadcast_to(
                                v, (n_conditions,) + v.shape
                            )
                            for k, v in node_init.items()
                        }

                    # Vary target node's scalar fields
                    for field_name, val in target_init.items():
                        if val.shape == ():
                            key, subkey = jax.random.split(key)
                            center = float(val)
                            scale = max(abs(center), 1.0) * 2.0
                            varied = center + jax.random.uniform(
                                subkey, (n_conditions,),
                                minval=-scale, maxval=scale,
                            )
                            # Ensure non-negative for position-like fields
                            if "position" in field_name.lower():
                                varied = jnp.maximum(varied, 0.1)
                            batched[req.node_name][field_name] = varied

                    ds = DatasetGenerator.from_sweep(
                        self.gm, req.node_name,
                        n_steps=steps_per_condition,
                        initial_states_batch=batched,
                    )
                    # Reset state after data generation
                    self._reset_state()

                    arch = MLPDirect(hidden_sizes=tuple(req.hidden_sizes))
                    trainer = SurrogateTrainer(arch, ds)

                    def progress(epoch, metrics):
                        job["epoch"] = epoch
                        job["train_loss"] = float(metrics["train_loss"])
                        job["val_loss"] = float(metrics["val_loss"])

                    result = trainer.train(
                        n_epochs=req.n_epochs,
                        batch_size=req.batch_size,
                        rng_key=jax.random.PRNGKey(42),
                        callback=progress,
                    )
                    job["result"] = result
                    job["status"] = "done"
                except Exception as exc:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    logger.exception("Surrogate training failed")

            thread = threading.Thread(target=_train_worker, daemon=True)
            thread.start()
            return {"job_id": job_id, "status": "started"}

        @app.get("/surrogate/status/{job_id}", tags=["surrogate"])
        def surrogate_status(job_id: str):
            if job_id not in self._surrogate_jobs:
                raise HTTPException(status_code=404, detail=f"No job '{job_id}'.")
            job = self._surrogate_jobs[job_id]
            return {
                "job_id": job["id"],
                "node_name": job["node_name"],
                "status": job["status"],
                "epoch": job["epoch"],
                "n_epochs": job["n_epochs"],
                "train_loss": job["train_loss"],
                "val_loss": job["val_loss"],
                "error": job["error"],
            }

        @app.post("/surrogate/activate/{job_id}", tags=["surrogate"])
        def surrogate_activate(job_id: str):
            """Replace the physics node with the trained surrogate."""
            if job_id not in self._surrogate_jobs:
                raise HTTPException(status_code=404, detail=f"No job '{job_id}'.")
            job = self._surrogate_jobs[job_id]
            if job["status"] != "done":
                raise HTTPException(status_code=400, detail="Training not complete.")

            node_name = job["node_name"]
            result = job["result"]

            # Save original node info for deactivation
            if node_name not in self._original_nodes:
                orig_node = self.gm._nodes[node_name].node
                orig_edges = [e for e in self.gm._edges
                              if e.source_node == node_name or e.target_node == node_name]
                orig_ext = [ei for ei in self.gm._external_inputs
                            if ei.target_node == node_name]
                self._original_nodes[node_name] = (orig_node, orig_edges, orig_ext)

            # Use the ORIGINAL node's initial state for surrogate initial values
            orig_node = self._original_nodes[node_name][0]
            initial_values = {}
            for field_name, shape in result.state_spec.items():
                if shape == ():
                    initial_values[field_name] = 0.0
                else:
                    initial_values[field_name] = jnp.zeros(shape)
            orig_init = orig_node.initial_state()
            for k, v in orig_init.items():
                if k in initial_values:
                    initial_values[k] = v

            was_running = self._runner_started
            self._stop_runner()

            surrogate = result.to_node(
                name=node_name,
                timestep=orig_node.delta_t,
                initial_values=initial_values,
            )

            from maddening.surrogates.replace import replace_node
            replace_node(self.gm, node_name, surrogate)
            self.gm.compile()
            self._active_surrogates.add(node_name)
            self._reset_state()

            return {"status": "activated", "node": node_name}

        @app.post("/surrogate/deactivate/{node_name}", tags=["surrogate"])
        def surrogate_deactivate(node_name: str):
            """Restore the original physics node."""
            if node_name not in self._original_nodes:
                raise HTTPException(
                    status_code=400,
                    detail=f"No original node saved for '{node_name}'.",
                )

            was_running = self._runner_started
            self._stop_runner()

            orig_node, orig_edges, orig_ext = self._original_nodes[node_name]

            # The revert rebuilds a subgraph, so it is all-or-nothing: on
            # any failure the live graph goes back to the surrogate it had,
            # rather than being left with the original node, half its edges
            # and no compiled step.
            snapshot = _graph_structure_snapshot(self.gm)
            problems: list[str] = []
            try:
                try:
                    self.gm.remove_node(node_name)
                except KeyError:
                    pass

                self.gm.add_node(orig_node)
                for edge in orig_edges:
                    try:
                        # Every EdgeSpec field, not just the endpoints: a
                        # dropped ``mapping`` breaks the shapes, and a
                        # dropped ``additive`` silently overwrites a
                        # boundary input the graph used to add to.
                        self.gm.add_edge(
                            source=edge.source_node, target=edge.target_node,
                            source_field=edge.source_field,
                            target_field=edge.target_field,
                            transform=edge.transform,
                            additive=edge.additive,
                            source_units=edge.source_units,
                            target_units=edge.target_units,
                            mapping=edge.mapping,
                        )
                    except Exception as exc:  # noqa: BLE001 - reported below
                        problems.append(f"edge {edge.key}: {exc}")
                for ei in orig_ext:
                    try:
                        self.gm.add_external_input(
                            ei.target_node, ei.target_field, ei.shape, ei.dtype,
                        )
                    except Exception as exc:  # noqa: BLE001 - reported below
                        problems.append(
                            f"external input {ei.target_node}.{ei.target_field}: {exc}")
                if problems:
                    raise RuntimeError("; ".join(problems))
                self.gm.compile()
            except Exception as exc:
                _restore_graph_structure(self.gm, snapshot)
                try:
                    self.gm.compile()
                except Exception:  # pragma: no cover - the graph compiled a moment ago
                    logger.exception(
                        "Rolling back surrogate deactivation of %r left an "
                        "uncompilable graph", node_name)
                raise HTTPException(
                    status_code=500,
                    detail=f"Could not restore '{node_name}'; the surrogate is "
                           f"still active: {exc}",
                )

            self._active_surrogates.discard(node_name)
            self._reset_state()
            del self._original_nodes[node_name]

            return {"status": "deactivated", "node": node_name}

        # -- profile endpoints (v0.2 #9) -----------------------------------

        @app.post("/sim/profile", tags=["sim"])
        def sim_profile(n_steps: int = 50, n_warmup: int = 3):
            """Run a step-time profile and return a Perfetto-loadable JSON trace.

            POST with ``?n_steps=N`` to override (default 50, capped at
            1000 to keep request latency bounded).  The response is a
            Perfetto-format JSON trace; save the body as ``profile.json``
            and drag-and-drop into https://ui.perfetto.dev for an
            interactive flame-graph view of per-node + coupling
            overhead.
            """
            from maddening.core.simulation.profiler import (
                profile_graph, profile_report_to_perfetto,
            )
            n_steps = max(1, min(1000, int(n_steps)))
            n_warmup = max(0, min(50, int(n_warmup)))
            try:
                if self._runner_started:
                    raise HTTPException(
                        status_code=409,
                        detail="Cannot profile while the runner is started. "
                               "POST /sim/stop first.",
                    )
                report = profile_graph(self.gm, n_steps=n_steps, n_warmup=n_warmup)
            except HTTPException:
                raise
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return profile_report_to_perfetto(report)

        @app.post("/sim/profile/jax/start", tags=["sim"])
        def sim_profile_jax_start():
            """Begin a JAX-level XLA trace.

            All subsequent ``/sim/step`` and ``/sim/run`` calls (and
            any runner steps) are recorded.  POST
            ``/sim/profile/jax/stop`` to end the trace.  The trace
            directory is returned in the stop response and can be loaded
            via TensorBoard's "Trace Viewer" plugin (which uses a
            Perfetto frontend).
            """
            from maddening.core.simulation.profiler import (
                start_jax_trace, jax_trace_active,
            )
            if jax_trace_active():
                raise HTTPException(
                    status_code=409, detail="A JAX trace is already active.",
                )
            try:
                log_dir = start_jax_trace()
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "tracing", "log_dir": log_dir}

        @app.post("/sim/profile/jax/stop", tags=["sim"])
        def sim_profile_jax_stop():
            """End the active JAX trace and return the log directory."""
            from maddening.core.simulation.profiler import (
                stop_jax_trace, jax_trace_active,
            )
            if not jax_trace_active():
                raise HTTPException(
                    status_code=409, detail="No JAX trace is active.",
                )
            log_dir = stop_jax_trace()
            self._last_jax_trace_dir = log_dir
            return {"status": "stopped", "log_dir": log_dir}

        @app.get("/sim/profile/jax/status", tags=["sim"])
        def sim_profile_jax_status():
            from maddening.core.simulation.profiler import jax_trace_active
            return {
                "active": jax_trace_active(),
                "last_trace_dir": getattr(self, "_last_jax_trace_dir", None),
            }

        # -- cloud endpoints ------------------------------------------------

        @app.post("/cloud/launch", tags=["cloud"])
        def cloud_launch(config: dict[str, Any] = {}):
            """Launch a cloud GPU session."""
            if not _cloud_deps_available():
                raise HTTPException(
                    status_code=400,
                    detail="Cloud dependencies not installed. "
                           "pip install maddening[cloud]",
                )
            if self._cloud_session is not None:
                raise HTTPException(
                    status_code=409, detail="Cloud session already active.",
                )
            try:
                from maddening.cloud.session import CloudSession, CloudConfig
                self._cloud_session = CloudSession()
                cloud_config = CloudConfig.from_dict(config) if config else CloudConfig()
                info = self._cloud_session.launch(cloud_config)
                return {"status": "launching", "session_id": info.session_id}
            except Exception as exc:
                self._cloud_session = None
                raise HTTPException(status_code=500, detail=str(exc))

        @app.get("/cloud/status", tags=["cloud"])
        def cloud_status():
            """Get cloud session status."""
            if self._cloud_session is None:
                raise HTTPException(
                    status_code=501,
                    detail="No cloud session configured. "
                           "Use POST /cloud/launch to start one.",
                )
            info = self._cloud_session.info
            result = self._cloud_session.health_check()
            return {
                "stage": info.stage.value,
                "vm_ip": info.vm_ip,
                "session_id": info.session_id,
                "fully_ready": result.fully_ready,
                "error_stage": result.error_stage,
                "error_detail": result.error_detail,
            }

        @app.post("/cloud/teardown", tags=["cloud"])
        def cloud_teardown():
            """Tear down the cloud session.

            If a JAX trace was captured during this session (see
            ``/sim/profile/jax/start``), snapshot the trace directory
            into the response so the user can recover the .pb files
            after the VM is gone.  This implements the v0.2 #9
            "CloudSession teardown auto-snapshot" requirement.
            """
            if self._cloud_session is None:
                raise HTTPException(
                    status_code=501,
                    detail="No cloud session configured.",
                )
            trace_snapshot = None
            if self._last_jax_trace_dir is not None and os.path.isdir(
                self._last_jax_trace_dir
            ):
                from maddening.core.simulation.profiler import tar_trace_dir
                import base64
                tar_bytes = tar_trace_dir(self._last_jax_trace_dir)
                trace_snapshot = {
                    "source_dir": self._last_jax_trace_dir,
                    "size_bytes": len(tar_bytes),
                    "tar_gz_b64": base64.b64encode(tar_bytes).decode("ascii"),
                }
            try:
                self._cloud_session.teardown()
            finally:
                self._cloud_session = None
            return {
                "status": "torn_down",
                "jax_trace_snapshot": trace_snapshot,
            }

        def _cloud_deps_available() -> bool:
            """Check if cloud dependencies are installed."""
            import importlib.util
            return importlib.util.find_spec("sky") is not None

        # -- WebSocket endpoints --------------------------------------------

        @app.websocket("/ws/state")
        async def ws_state(websocket: WebSocket):
            """Stream state snapshots as JSON at ~30 Hz.

            Client may send JSON messages to configure the stream:

            * ``{"type": "subscribe", "fields": {"node": ["f1", "f2"]}}``
              — only include the listed node/field pairs in subsequent
              snapshots.  Send ``{"type": "subscribe", "fields": null}``
              to reset to full state.
            * ``{"type": "config", "fps": 15}`` — change poll rate.
            """
            await websocket.accept()
            logger.info("WebSocket client connected to /ws/state")
            self._ensure_relay_attached()

            sub_fields = [None]   # mutable: {node: [fields]} or None
            target_fps = [30.0]
            config_changed = asyncio.Event()

            async def _receive():
                import json as _json
                try:
                    while True:
                        raw = await websocket.receive_text()
                        try:
                            msg = _json.loads(raw)
                            if msg.get("type") == "subscribe":
                                sub_fields[0] = msg.get("fields")
                            elif msg.get("type") == "config":
                                if "fps" in msg:
                                    target_fps[0] = max(1, min(120, msg["fps"]))
                            config_changed.set()
                        except (ValueError, KeyError):
                            pass
                except (WebSocketDisconnect, Exception):
                    pass

            receiver = asyncio.create_task(_receive())

            last_sim_time = -1.0
            try:
                while True:
                    config_changed.clear()
                    sim_time, snapshot = self.relay.latest_snapshot()
                    if snapshot is not None and sim_time != last_sim_time:
                        last_sim_time = sim_time
                        state = snapshot
                        # Apply field subscription filter
                        filt = sub_fields[0]
                        if filt is not None:
                            state = {
                                node: {
                                    f: fields[f]
                                    for f in filt.get(node, [])
                                    if f in fields
                                }
                                for node, fields in state.items()
                                if node in filt
                            }
                        payload = {
                            "sim_time": sim_time,
                            "state": _jax_to_python(state),
                        }
                        await websocket.send_json(payload)
                    await asyncio.sleep(1.0 / target_fps[0])
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected from /ws/state")
            except Exception:
                logger.exception("WebSocket error on /ws/state")
            finally:
                receiver.cancel()

        @app.websocket("/ws/state/binary")
        async def ws_state_binary(websocket: WebSocket):
            """Stream state snapshots as binary at ~60 Hz.

            Protocol:
                1. Server sends JSON text frame with the binary schema.
                2. Server sends binary frames: ``[f64 sim_time][f32 values...]``.

            Client may send JSON messages:

            * ``{"type": "subscribe", "fields": {"node": ["f1", "f2"]},
              "compression": "zstd"}`` — rebuild encoder for listed
              fields only and (optionally) set a compression mode.
              ``compression`` ∈ ``{"none", "zstd", "zstd+xor"}``.  Server
              re-sends a new schema (with the updated ``compression``
              key) before resuming binary frames.  Send ``null`` fields
              to reset to full state; omit ``compression`` to keep the
              current mode.
            * ``{"type": "config", "fps": 30}`` — change poll rate.
            """
            await websocket.accept()
            logger.info("WebSocket client connected to /ws/state/binary")
            self._ensure_relay_attached()

            # Mutable state shared with receiver task
            target_fps = [60.0]
            current_encoder = [self._get_binary_encoder()]
            current_compression = ["none"]
            current_fields: list[dict | None] = [None]
            schema_dirty = asyncio.Event()

            await websocket.send_json(current_encoder[0].schema())

            async def _receive():
                import json as _json
                try:
                    while True:
                        raw = await websocket.receive_text()
                        try:
                            msg = _json.loads(raw)
                            if msg.get("type") == "subscribe":
                                from maddening.api.binary_encoder import (
                                    BinaryStateEncoder, VALID_COMPRESSIONS,
                                )
                                if "fields" in msg:
                                    current_fields[0] = msg["fields"]
                                if "compression" in msg:
                                    comp = msg["compression"]
                                    if comp in VALID_COMPRESSIONS:
                                        current_compression[0] = comp
                                user_state = {
                                    k: v for k, v in self.gm._state.items()
                                    if k != "_meta"
                                }
                                try:
                                    current_encoder[0] = BinaryStateEncoder(
                                        user_state,
                                        fields=current_fields[0],
                                        compression=current_compression[0],
                                    )
                                except ImportError:
                                    # zstandard not installed — fall back
                                    current_compression[0] = "none"
                                    current_encoder[0] = BinaryStateEncoder(
                                        user_state,
                                        fields=current_fields[0],
                                        compression="none",
                                    )
                                schema_dirty.set()
                            elif msg.get("type") == "config":
                                if "fps" in msg:
                                    target_fps[0] = max(1, min(120, msg["fps"]))
                        except (ValueError, KeyError):
                            pass
                except (WebSocketDisconnect, Exception):
                    pass

            receiver = asyncio.create_task(_receive())

            last_sim_time = -1.0
            try:
                while True:
                    # Re-send schema if subscription changed
                    if schema_dirty.is_set():
                        schema_dirty.clear()
                        await websocket.send_json(current_encoder[0].schema())

                    sim_time, snapshot = self.relay.latest_snapshot()
                    if snapshot is not None and sim_time != last_sim_time:
                        last_sim_time = sim_time
                        frame = current_encoder[0].encode(sim_time, snapshot)
                        await websocket.send_bytes(frame)
                    await asyncio.sleep(1.0 / target_fps[0])
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected from /ws/state/binary")
            except Exception:
                logger.exception("WebSocket error on /ws/state/binary")
            finally:
                receiver.cancel()

        @app.websocket("/ws/render")
        async def ws_render(websocket: WebSocket):
            """Stream server-side rendered frames as compressed images.

            Protocol:
                1. Server sends a JSON text frame with renderer config
                   (width, height, format, content_type).
                2. Server sends binary frames containing raw image bytes
                   (JPEG/WebP/PNG).
                3. Client may send JSON messages to adjust settings:
                   ``{"type": "config", "format": "webp", "quality": 80, "fps": 30}``
                   ``{"type": "reset"}`` to clear time-series buffers.

            Designed for thin browser clients that only display images.
            All rendering happens server-side -- suitable for deployment
            behind services like AWS AppStream.
            """
            if self._frame_renderer is None:
                await websocket.close(
                    code=1008,
                    reason="No frame renderer configured on this server.",
                )
                return

            await websocket.accept()
            logger.info("WebSocket client connected to /ws/render")
            self._ensure_relay_attached()

            renderer = self._frame_renderer
            target_fps = [30.0]  # mutable so the receiver task can update it

            # Send initial config
            await websocket.send_json({
                "type": "config",
                "width": renderer.width,
                "height": renderer.height,
                "format": renderer.fmt,
                "content_type": renderer.content_type,
            })

            config_changed = asyncio.Event()

            async def _receive_client_messages():
                """Background task: listen for client config messages."""
                import json as _json
                try:
                    while True:
                        raw = await websocket.receive_text()
                        try:
                            msg = _json.loads(raw)
                            if msg.get("type") == "config":
                                if "format" in msg:
                                    renderer.set_format(
                                        msg["format"], msg.get("quality"),
                                    )
                                if "fps" in msg:
                                    target_fps[0] = max(1, min(60, msg["fps"]))
                                config_changed.set()
                            elif msg.get("type") == "reset":
                                renderer.reset()
                        except (ValueError, KeyError):
                            pass
                except WebSocketDisconnect:
                    pass
                except Exception:
                    pass

            receiver = asyncio.create_task(_receive_client_messages())

            last_sim_time = -1.0
            try:
                while True:
                    # Re-send config if client changed settings
                    if config_changed.is_set():
                        config_changed.clear()
                        await websocket.send_json({
                            "type": "config",
                            "width": renderer.width,
                            "height": renderer.height,
                            "format": renderer.fmt,
                            "content_type": renderer.content_type,
                        })

                    sim_time, snapshot = self.relay.latest_snapshot()
                    if snapshot is not None and sim_time != last_sim_time:
                        last_sim_time = sim_time
                        loop = asyncio.get_event_loop()
                        frame = await loop.run_in_executor(
                            None, renderer.render, sim_time, snapshot,
                        )
                        await websocket.send_bytes(frame)

                    await asyncio.sleep(1.0 / target_fps[0])
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected from /ws/render")
            except Exception:
                logger.exception("WebSocket error on /ws/render")
            finally:
                receiver.cancel()

        return app
