"""
SimulationServer -- FastAPI + WebSocket server wrapping a GraphManager.

Provides REST endpoints for graph construction, validation, compilation,
state inspection, and simulation stepping/running, plus a WebSocket
endpoint that streams state snapshots in real time.

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
from pathlib import Path
from typing import Any, Optional

import jax.numpy as jnp

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
        # JAX or numpy array -- use .item() for scalars, .tolist() for arrays
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
    """Request body for POST /graph/nodes."""
    type: str
    name: str
    timestep: float
    params: dict[str, Any] = {}


class AddEdgeRequest(BaseModel):
    """Request body for POST /graph/edges."""
    source_node: str
    target_node: str
    source_field: str
    target_field: str


class RemoveEdgeRequest(BaseModel):
    """Request body for DELETE /graph/edges."""
    source_node: str
    target_node: str
    source_field: str
    target_field: str


class SetNodeStateRequest(BaseModel):
    """Request body for PUT /graph/state/{node_name}."""
    state: dict[str, Any]


# ------------------------------------------------------------------
# SimulationServer
# ------------------------------------------------------------------

class SimulationServer:
    """Wraps a ``GraphManager`` with a FastAPI HTTP + WebSocket interface.

    Parameters
    ----------
    node_registry : dict[str, type]
        Maps node type name strings (e.g. ``"BallNode"``) to the
        corresponding ``SimulationNode`` subclass.  Used by
        ``POST /graph/nodes`` to instantiate nodes by type.
    graph_manager : GraphManager, optional
        An existing ``GraphManager`` to serve.  If ``None``, an empty
        one is created.
    """

    def __init__(
        self,
        node_registry: dict[str, type[SimulationNode]],
        graph_manager: Optional[GraphManager] = None,
    ) -> None:
        self.registry = dict(node_registry)
        self.gm = graph_manager if graph_manager is not None else GraphManager()
        self.relay = StateRelay()
        self.runner: Optional[RealtimeRunner] = None
        self._runner_started = False
        self._relay_attached = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_relay_attached(self) -> None:
        """Lazily attach the relay once the graph has nodes.

        Safe to call multiple times -- only attaches once.
        """
        if self._relay_attached:
            return
        try:
            self.relay.attach(self.gm)
            self._relay_attached = True
        except RuntimeError:
            # No nodes yet -- attach will fail because timestep is unknown
            pass

    def _ensure_runner(self) -> RealtimeRunner:
        """Create a ``RealtimeRunner`` if one doesn't exist yet."""
        if self.runner is None:
            self._ensure_relay_attached()
            self.runner = RealtimeRunner(self.gm, self.relay)
        return self.runner

    def _state_json(self) -> dict:
        """Return the full graph state as plain-Python dicts."""
        return _jax_to_python(self.gm._state)

    def _node_state_json(self, name: str) -> dict:
        """Return a single node's state as plain Python."""
        return _jax_to_python(self.gm.get_node_state(name))

    # ------------------------------------------------------------------
    # App factory
    # ------------------------------------------------------------------

    def create_app(self) -> FastAPI:
        """Build and return the FastAPI application."""
        app = FastAPI(
            title="MADDENING Simulation Server",
            description="HTTP/WebSocket API for the MADDENING simulation graph.",
            version="0.1.0",
        )

        # -- visualization endpoint ----------------------------------------

        @app.get("/viz/graph", tags=["viz"], response_class=HTMLResponse)
        def viz_graph():
            """Serve the interactive graph visualization page."""
            html_path = _STATIC_DIR / "graph.html"
            return HTMLResponse(content=html_path.read_text(), status_code=200)

        # -- graph structure endpoints ------------------------------------

        @app.get("/graph", tags=["graph"])
        def get_graph():
            """Return the graph structure (nodes, edges, external inputs)."""
            return self.gm.to_dict()

        @app.post("/graph/nodes", tags=["graph"], status_code=201)
        def add_node(req: AddNodeRequest):
            """Add a node to the graph."""
            if req.type not in self.registry:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Unknown node type '{req.type}'. "
                        f"Available: {list(self.registry.keys())}"
                    ),
                )
            node_cls = self.registry[req.type]
            try:
                node = node_cls(name=req.name, timestep=req.timestep, **req.params)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            try:
                self.gm.add_node(node)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            return {"status": "ok", "node": node.to_dict()}

        @app.delete("/graph/nodes/{name}", tags=["graph"])
        def remove_node(name: str):
            """Remove a node (and its connected edges) from the graph."""
            try:
                self.gm.remove_node(name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            return {"status": "ok"}

        @app.post("/graph/edges", tags=["graph"], status_code=201)
        def add_edge(req: AddEdgeRequest):
            """Add a data-dependency edge between two nodes."""
            self.gm.add_edge(
                source=req.source_node,
                target=req.target_node,
                source_field=req.source_field,
                target_field=req.target_field,
            )
            return {"status": "ok"}

        @app.delete("/graph/edges", tags=["graph"])
        def remove_edge(req: RemoveEdgeRequest):
            """Remove a specific edge."""
            self.gm.remove_edge(
                source=req.source_node,
                target=req.target_node,
                source_field=req.source_field,
                target_field=req.target_field,
            )
            return {"status": "ok"}

        @app.post("/graph/compile", tags=["graph"])
        def compile_graph():
            """Compile the graph (topological sort + JIT)."""
            try:
                self.gm.compile()
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "ok", "schedule": self.gm.schedule}

        @app.post("/graph/validate", tags=["graph"])
        def validate_graph():
            """Validate the graph and return any issues."""
            issues = self.gm.validate()
            return {"issues": issues}

        # -- state endpoints ----------------------------------------------

        @app.get("/graph/state", tags=["state"])
        def get_state():
            """Return the current state of all nodes."""
            return self._state_json()

        @app.get("/graph/state/{node_name}", tags=["state"])
        def get_node_state(node_name: str):
            """Return the current state of a single node."""
            try:
                return self._node_state_json(node_name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))

        @app.put("/graph/state/{node_name}", tags=["state"])
        def set_node_state(node_name: str, req: SetNodeStateRequest):
            """Overwrite the state of a single node."""
            try:
                jax_state = _python_to_jax(req.state)
                self.gm.set_node_state(node_name, jax_state)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            return {"status": "ok"}

        # -- checkpoint endpoints -----------------------------------------

        @app.post("/checkpoint/save", tags=["checkpoint"])
        def checkpoint_save(path: str = "checkpoint.npz"):
            """Save the current simulation state to a file."""
            try:
                self.gm.save_state(path)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "ok", "path": path}

        @app.post("/checkpoint/load", tags=["checkpoint"])
        def checkpoint_load(path: str = "checkpoint.npz"):
            """Load simulation state from a file."""
            try:
                self.gm.load_state(path)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "ok", "state": self._state_json()}

        # -- simulation control endpoints ---------------------------------

        @app.post("/sim/step", tags=["sim"])
        def sim_step():
            """Advance the simulation by one timestep. Returns the new state."""
            try:
                self.gm.step()
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return self._state_json()

        @app.post("/sim/run", tags=["sim"])
        def sim_run(n_steps: int = 100):
            """Run *n_steps* simulation steps. Returns the final state."""
            try:
                self.gm.run(n_steps)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return self._state_json()

        @app.post("/sim/start", tags=["sim"])
        def sim_start():
            """Start the real-time simulation runner."""
            runner = self._ensure_runner()
            if self._runner_started:
                raise HTTPException(
                    status_code=409,
                    detail="Runner is already started. Stop it first.",
                )
            try:
                self._ensure_relay_attached()
                runner.start()
                self._runner_started = True
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "started"}

        @app.post("/sim/pause", tags=["sim"])
        def sim_pause():
            """Pause the real-time simulation runner."""
            if self.runner is None or not self._runner_started:
                raise HTTPException(
                    status_code=409, detail="Runner is not started."
                )
            self.runner.pause()
            return {"status": "paused"}

        @app.post("/sim/resume", tags=["sim"])
        def sim_resume():
            """Resume the real-time simulation runner."""
            if self.runner is None or not self._runner_started:
                raise HTTPException(
                    status_code=409, detail="Runner is not started."
                )
            self.runner.resume()
            return {"status": "resumed"}

        @app.post("/sim/stop", tags=["sim"])
        def sim_stop():
            """Stop the real-time simulation runner."""
            if self.runner is None or not self._runner_started:
                raise HTTPException(
                    status_code=409, detail="Runner is not started."
                )
            self.runner.stop()
            self._runner_started = False
            # Allow a fresh runner next time
            self.runner = None
            return {"status": "stopped"}

        # -- WebSocket endpoint -------------------------------------------

        @app.websocket("/ws/state")
        async def ws_state(websocket: WebSocket):
            """Stream state snapshots at ~30 Hz.

            The server polls the ``StateRelay`` and pushes JSON frames
            to the client.  No messages from the client are expected
            (but receiving them won't break anything).
            """
            await websocket.accept()
            logger.info("WebSocket client connected to /ws/state")

            # Make sure the relay is attached so it can receive snapshots
            self._ensure_relay_attached()

            last_sim_time = -1.0
            try:
                while True:
                    sim_time, snapshot = self.relay.latest_snapshot()
                    if snapshot is not None and sim_time != last_sim_time:
                        last_sim_time = sim_time
                        payload = {
                            "sim_time": sim_time,
                            "state": _jax_to_python(snapshot),
                        }
                        await websocket.send_json(payload)
                    # ~30 Hz polling
                    await asyncio.sleep(1.0 / 30.0)
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected from /ws/state")
            except Exception:
                logger.exception("WebSocket error on /ws/state")

        return app
