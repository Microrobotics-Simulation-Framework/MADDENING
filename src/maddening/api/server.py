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

    # Run with: uvicorn module:app --host 127.0.0.1
    # Or programmatically:
    #   import uvicorn
    #   uvicorn.run(app, host="127.0.0.1", port=8000)

Security
--------
A **loopback bind is unauthenticated**, exactly as it always was: bind
``127.0.0.1`` and nothing changes for local development.  **Any other
bind requires a bearer token on every route** except ``/healthz`` and
the static ``/viz/*`` pages -- see :mod:`maddening.api.auth` for where
the token comes from and how a client presents it.  Tell the server
which address it will be bound to::

    server = SimulationServer(registry, bind_host=host)
    server.auth.announce(port)          # logs a generated token once
    warn_if_publicly_bound(host, port)  # logs the remaining exposure
    uvicorn.run(server.create_app(), host=host, port=port)

There is still **no TLS**.  The token crosses the network in cleartext,
so a non-loopback bind belongs on a private network or behind a
TLS-terminating proxy; an SSH tunnel to a loopback-bound server remains
the best-supported way to reach this API from another machine.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
import uuid
import warnings
from pathlib import Path
from typing import Annotated, Any, Iterable, Optional
from urllib.parse import urlsplit

import jax
import jax.numpy as jnp
import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
    from pydantic import BaseModel, Field, field_validator
except ImportError as _exc:
    raise ImportError(
        "The MADDENING API server requires 'fastapi' and 'pydantic'. "
        "Install them with:  pip install fastapi uvicorn"
    ) from _exc

_STATIC_DIR = Path(__file__).parent / "static"

from maddening import __version__ as _maddening_version
from maddening.api.auth import (
    LOOPBACK_HOSTS,
    UNAUTHENTICATED_PATHS,
    WS_SUBPROTOCOL,
    APIAuth,
    bearer_from_headers,
    bearer_from_subprotocols,
    is_loopback,
)
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.graph_manager import (
    GraphManager,
    _declared_boundary_zeros,
    _hook_outputs,
    _leaf_values_equal,
    _node_with_params,
    _NodeSpec,
    _params_holders,
)
from maddening.core.node import SimulationNode, _method_accepts_params
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
# Request bounds
# ------------------------------------------------------------------
# The server has no authentication (see the module note above and
# ``warn_if_publicly_bound``), so every number a caller sends is a number
# an *anonymous* caller sends.  Each limit below caps the work or memory
# one request can name.  They are deliberately far above anything the
# shipped examples and UI ask for (200 steps, 2000 data steps, 500
# epochs) and are declared on the request models so the cap appears in
# ``/openapi.json`` and a violation is a 422 naming the field, not a
# server that stops answering.

#: Upper bound on ``POST /sim/run?n_steps=``.  A longer run belongs to
#: the background runner (``POST /sim/start``), which stays cancellable.
MAX_RUN_STEPS = 100_000

#: Upper bound on any integer constructor parameter in ``POST
#: /graph/nodes``.  A node turns such an integer into an array dimension,
#: so this is the memory one request can ask for.
MAX_NODE_PARAM_INT = 10_000_000

#: Upper bound on the number of scalars inside one node's parameters.
MAX_NODE_PARAM_ELEMENTS = 1_000_000

#: Upper bound on the elements of a new node's initial state, summed over
#: its fields.  Catches dimensions that multiply (each factor small, the
#: product not).  20e6 float32 elements is 80 MB.
MAX_NODE_STATE_ELEMENTS = 20_000_000

#: Bounds on ``POST /surrogate/train``.
MAX_SURROGATE_DATA_STEPS = 100_000
MAX_SURROGATE_EPOCHS = 10_000
MAX_SURROGATE_BATCH_SIZE = 65_536
MAX_SURROGATE_LAYERS = 16
MAX_SURROGATE_LAYER_WIDTH = 8192


def _oversized_param(value: Any, path: str = "") -> Optional[str]:
    """Why *value* is too big to accept as a node parameter, else ``None``.

    Integers are bounded because a node constructor turns one into an
    array dimension -- the auditor measured +433 MB of RSS from a single
    unauthenticated ``POST /graph/nodes``.  Element counts are bounded
    because a parameter may itself be a large array.  Floats are not
    bounded: a float is a physical constant, not a dimension, and any cap
    on one would be arbitrary.

    This check and :func:`_non_finite_param` partition the numbers rather
    than racing for them: an integer literal too large to be a ``float``
    at all (JSON allows one of any length) is *not* a dimension anyone
    could mean, it is the unusable constant ``_non_finite_param`` already
    owns, so it is left to that check and its 400 "value must be finite".
    What this function rejects is the plausible-but-too-big dimension,
    and it does so from the request model, as a 422.
    """
    total = 0
    stack: list[tuple[Any, str]] = [(value, path)]
    while stack:
        item, where = stack.pop()
        if isinstance(item, dict):
            stack.extend((v, f"{where}.{k}" if where else str(k))
                         for k, v in item.items())
            continue
        if isinstance(item, (list, tuple)):
            stack.extend((v, f"{where}[{i}]") for i, v in enumerate(item))
            continue
        total += 1
        if total > MAX_NODE_PARAM_ELEMENTS:
            return (f"params: at most {MAX_NODE_PARAM_ELEMENTS} values in total "
                    f"(this server is unauthenticated; see its README)")
        if isinstance(item, bool):
            continue
        if isinstance(item, int) and abs(item) > MAX_NODE_PARAM_INT:
            try:
                usable = math.isfinite(float(item))
            except (OverflowError, ValueError):
                usable = False
            if not usable:
                continue  # _non_finite_param's 400, not ours
            return (f"params.{where or 'value'}: integer magnitude must be at "
                    f"most {MAX_NODE_PARAM_INT} (a node turns one into an array "
                    f"dimension; this server is unauthenticated)")
    return None


# ------------------------------------------------------------------
# Pydantic request/response models
# ------------------------------------------------------------------

class AddNodeRequest(BaseModel):
    type: str
    name: str
    timestep: float
    params: dict[str, Any] = {}

    @field_validator("params")
    @classmethod
    def _params_within_bounds(cls, value: dict[str, Any]) -> dict[str, Any]:
        problem = _oversized_param(value)
        if problem is not None:
            raise ValueError(problem)
        return value


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

    # The same bound as ``POST /graph/nodes``: the route builds the node
    # with the new values to check them (its constructor, the state it
    # would build), so an integer here is an array dimension exactly as
    # it is there.  Without it ``PUT n_cells=3e7`` built a 30-million-cell
    # rod and its state (+235 MB) before the layout check refused it.
    @field_validator("params")
    @classmethod
    def _params_within_bounds(cls, value: dict[str, Any]) -> dict[str, Any]:
        problem = _oversized_param(value)
        if problem is not None:
            raise ValueError(problem)
        return value


class TrainSurrogateRequest(BaseModel):
    node_name: str
    n_data_steps: int = Field(500, ge=1, le=MAX_SURROGATE_DATA_STEPS)
    n_epochs: int = Field(100, ge=1, le=MAX_SURROGATE_EPOCHS)
    hidden_sizes: list[
        Annotated[int, Field(ge=1, le=MAX_SURROGATE_LAYER_WIDTH)]
    ] = Field([64, 64], min_length=1, max_length=MAX_SURROGATE_LAYERS)
    batch_size: int = Field(64, ge=1, le=MAX_SURROGATE_BATCH_SIZE)


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


def _same_param_value(old: Any, new: Any) -> bool:
    """Is a JSON *new* value for a structural parameter the value it has?

    Strict on purpose: ``5`` and ``5.0`` differ (the write would turn an
    integer constant into a float one), and anything that does not compare
    cleanly counts as changed.  "Changed" only means the write is checked.
    """
    if type(old) is not type(new):
        return False
    try:
        return bool(old == new)
    except Exception:  # noqa: BLE001 - an array-valued entry, say
        return False


def _state_elements(state: Any) -> int:
    """Total number of scalars across a node state's fields."""
    total = 0
    for leaf in jax.tree_util.tree_leaves(state):
        shape = getattr(leaf, "shape", None)
        if shape is None:
            total += 1
            continue
        size = 1
        for dim in shape:
            size *= int(dim)
        total += size
    return total


def _dry_run_node(node, state: Any = None) -> None:
    """Abstractly trace one ``update`` of a freshly built node on its own
    initial state with zero boundary inputs: catches constants of the
    wrong type / shape before the node enters the graph.

    *state* is the node's initial state when the caller has already built
    it (the size check does), so it is not allocated twice.
    """
    import jax  # noqa: PLC0415

    state = node.initial_state() if state is None else state
    bi = {}
    try:
        for name, spec in (node.boundary_input_spec() or {}).items():
            bi[name] = jnp.zeros(tuple(getattr(spec, "shape", ()) or ()), jnp.float32)
    except Exception:  # noqa: BLE001 - descriptor is advisory
        bi = {}
    jax.eval_shape(lambda: node.update(state, bi, node.delta_t))


# ------------------------------------------------------------------
# PUT /graph/params: what a write would do, asked before it is made
# ------------------------------------------------------------------

#: Raised when node code needs a value concretely under abstract
#: evaluation.  They mean "this cannot be told without building it", never
#: "the node refuses the value", so they never refuse a write.
_NEEDS_CONCRETE_VALUES = (
    jax.errors.ConcretizationTypeError,
    jax.errors.TracerArrayConversionError,
    jax.errors.TracerBoolConversionError,
    jax.errors.TracerIntegerConversionError,
    jax.errors.NonConcreteBooleanIndexError,
    jax.errors.UnexpectedTracerError,
)


def _probe_pair_with(node: Any, changes: dict[str, Any]) -> Optional[tuple]:
    """``(old, new)``: shallow copies of the node a write of every entry of
    *changes* into ``node.params`` lands on, reading the current params and
    the written ones -- the node, or the node a wrapper that cannot be
    copied wraps (``_param_probe_pair``'s rule, for several keys at once).
    ``None`` when no faithful copy can be made.  Nothing is constructed."""
    shared = getattr(node, "params", None)
    if not isinstance(shared, dict):
        return None
    for candidate in _params_holders(node):
        old = _node_with_params(candidate, dict(shared))
        new = _node_with_params(candidate, {**shared, **changes})
        if old is not None and new is not None:
            return old, new
    return None


def _abstract_initial_state(node: Any) -> tuple[str, Any]:
    """``node.initial_state()`` evaluated for shapes and dtypes only.

    ``("ok", shapes)``, ``("raises", exc)`` when the node's code raises
    under abstract evaluation for a reason that is not a concrete value it
    needed (a shape that does not broadcast, a validation error), or
    ``("unknown", exc)`` when it needed one.  JAX operations are staged,
    not run, so nothing of the state's size is allocated.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return "ok", jax.eval_shape(node.initial_state)
    except _NEEDS_CONCRETE_VALUES as exc:
        return "unknown", exc
    except Exception as exc:  # noqa: BLE001 - the node's own refusal
        return "raises", exc


def _abstract_layout(shapes: Any) -> dict[str, tuple]:
    """``{field path: (shape, dtype)}`` of an abstract state."""
    return {
        jax.tree_util.keystr(path): (tuple(leaf.shape), str(leaf.dtype))
        for path, leaf in jax.tree_util.tree_flatten_with_path(shapes)[0]
    }


def _state_write_reason_before_building(node: Any, changes: dict[str, Any]) -> Optional[str]:
    """Why the state the node would build with *changes* written into its
    params rules the write out, told without building anything: it holds
    more than :data:`MAX_NODE_STATE_ELEMENTS` elements, it has another
    layout than the running state, or ``initial_state()`` raises with it.
    ``None`` when none of these holds or it cannot be told this way.

    ``PUT /graph/params`` refuses every one of these outright, and the
    checks that establish it on concrete values construct the node and
    build its state *at the new size* first -- ``n_cells=3e7`` cost
    +235 MB and half a second before the 400, and a pipe's ``nx``
    multiplies into its whole volume.  This is asked first, on shallow
    copies of the node (:func:`_probe_pair_with`, no constructor call)
    evaluated abstractly (:func:`_abstract_initial_state`), so such a
    write is refused before anything of its size exists.  It refuses only
    on positive evidence, and only when the current params evaluate this
    way too; a node whose ``initial_state`` needs concrete values is left
    to the concrete checks.
    """
    pair = _probe_pair_with(node, changes)
    if pair is None:
        return None
    old_status, old_shapes = _abstract_initial_state(pair[0])
    if old_status != "ok":
        return None
    new_status, new_shapes = _abstract_initial_state(pair[1])
    if new_status == "raises":
        return (f"{type(node).__name__}.initial_state() raises with it "
                f"({type(new_shapes).__name__}: {new_shapes})")
    if new_status != "ok":
        return None
    n_elements = _state_elements(new_shapes)
    if n_elements > MAX_NODE_STATE_ELEMENTS:
        return (f"the node would hold {n_elements} state elements with it; at "
                f"most {MAX_NODE_STATE_ELEMENTS} are accepted over the API "
                "(this server is unauthenticated -- build a graph this size "
                "in-process)")
    was, now = _abstract_layout(old_shapes), _abstract_layout(new_shapes)
    if was == now:
        return None
    changed = [
        f"{field} {was.get(field, 'absent')} -> {now.get(field, 'absent')}"
        for field in sorted(set(was) | set(now))
        if was.get(field) != now.get(field)
    ]
    return (
        "it changes the layout of the state the node builds ((shape, "
        f"dtype) of {'; '.join(changed)}), and the running state keeps "
        "its layout until it is reset: the step recompiled for the new "
        "value would be traced against a state it was not written for"
    )


def _derived_attributes_that_differ(running: Any, rebuilt: Any) -> list[str]:
    """Attributes other than ``params`` that *rebuilt* holds differently
    from *running*: what the constructor derived from the params, compared.
    Only the rebuilt node's attributes count -- one the running node
    gained since it was constructed (a guard or a cache the graph set) is
    not something a constructor derives."""
    mine, theirs = vars(running), vars(rebuilt)
    out = []
    for name in sorted(set(theirs) - {"params"}):
        if name not in mine:
            out.append(name)
            continue
        a, b = mine[name], theirs[name]
        if a is b:
            continue
        try:
            if any(isinstance(v, (jax.Array, np.ndarray)) for v in (a, b)):
                same = _leaf_values_equal(a, b)
            else:
                same = bool(a == b)
        except Exception:  # noqa: BLE001 - not comparable: count it as changed
            same = False
        if not same:
            out.append(name)
    return out


def _what_the_node_computes(spec, probe: Any, descended: bool, state: Any,
                            leaves: Any) -> Optional[tuple]:
    """``(jaxpr text, jaxpr constants, initial state)`` of *probe*'s hooks,
    called the way the graph calls them (``update``, and the flux and
    interface-correction hooks where it has them) on *state* with a zero
    for every declared boundary input and *leaves* as the injected params,
    and of its ``initial_state()``.  ``None`` when it cannot be told."""
    probe_spec = _NodeSpec(
        node=probe, update_fn=probe.update, timestep=spec.timestep,
        accepts_params=(
            _method_accepts_params(probe, "update") and leaves is not None
            if descended else spec.accepts_params),
        flux_accepts_params=(
            _method_accepts_params(probe, "compute_boundary_fluxes")
            and leaves is not None
            if descended else spec.flux_accepts_params),
    )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bi = _declared_boundary_zeros(probe)
            closed = jax.make_jaxpr(
                lambda st, b, p: _hook_outputs(probe_spec, st, b, p),
            )(state, bi, leaves)
            initial = probe.initial_state()
    except Exception:  # noqa: BLE001 - cannot tell
        return None
    return str(closed.jaxpr), list(closed.consts), initial


def _computes_the_same(a: tuple, b: tuple) -> list[str]:
    """What differs between two :func:`_what_the_node_computes` results."""
    (text_a, consts_a, init_a), (text_b, consts_b, init_b) = a, b
    out = []
    if text_a != text_b:
        out.append("the step traces to a different computation")
    elif len(consts_a) != len(consts_b) or not all(
            _leaf_values_equal(x, y) for x, y in zip(consts_a, consts_b)):
        out.append("the step closes over different constants")
    if (jax.tree.structure(init_a) != jax.tree.structure(init_b)
            or not _leaf_values_equal(init_a, init_b)):
        out.append("initial_state() builds a different state")
    return out


def _saved_graph_write_reason(gm: GraphManager, owner: str, changes: dict[str, Any],
                              live_before: Optional[dict],
                              live_after: Optional[dict]) -> Optional[str]:
    """Why the graph a save after writing *changes* reloads would not run
    what the running graph runs with them, or ``None``.

    ``to_dict()`` saves each node as its class and its effective params
    (:meth:`~maddening.core.graph_manager.GraphManager.effective_node_params`),
    and ``from_dict()`` calls the class with them.  So a 200 from ``PUT
    /graph/params`` promises two things this asks of the whole request at
    once, with the params a save would carry -- every other key's live
    value included, which a fit may have moved:

    1. **The constructor takes them.**  Until 0.4.0 only a structural key
       was asked, one at a time and against the constructor's own values:
       a ``HeatNode`` took a ``thermal_diffusivity`` at Fourier number 0.6
       and an ``LBMPipeNode`` a ``rho_gas`` above its ``rho_liquid``, both
       live leaves, and neither saved graph loaded.
    2. **The node it builds computes what the running node computes.**
       A constructor may derive a branch or an array from a value
       (``LBMPipeNode``'s single-/multiphase switch from ``G != 0``); the
       running node keeps what it derived, so a write can be *used* -- the
       step reads the new ``G`` -- and still run another model than the
       reload: ``G=0`` on a multiphase pipe stayed multiphase with no
       interaction, while its save reloaded single-phase (0.645 apart in
       the tracer after ten steps).  Asked only when the rebuilt node
       holds some attribute differently from the running one, by tracing
       both nodes' hooks with the same state and params, and building
       both initial states.  A node that already differs from its own
       rebuild before the write (a constructor that draws random numbers,
       a value a fit moved that its constructor consumes) cannot be
       blamed on this write and is not refused here.

    Asked of the node, or of the node a wrapper wraps (whichever is
    rebuilt from its params, as for ``_constructor_write_reason``); a node
    that is not rebuilt from its params is not asked.
    """
    spec = gm._nodes[owner]
    node = spec.node
    shared = node.params
    try:
        effective = gm.effective_node_params(owner)
    except Exception:  # noqa: BLE001 - the graph cannot say what it would save
        return None
    for candidate in _params_holders(node):
        cls = type(candidate)

        def build(params, cls=cls, candidate=candidate):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return cls(name=candidate.name, timestep=candidate.delta_t, **params)

        try:
            rebuilt_old = build(effective)
        except Exception:  # noqa: BLE001 - not rebuilt from its params
            continue
        try:
            rebuilt_new = build({**effective, **changes})
        except Exception as exc:  # noqa: BLE001 - the constructor refuses it
            return (
                f"{cls.__name__}'s constructor refuses it ({type(exc).__name__}: "
                f"{exc}), so a graph saved with it would not load"
            )
        differ = _derived_attributes_that_differ(candidate, rebuilt_new)
        if not differ:
            return None
        probe_old = _node_with_params(candidate, dict(shared))
        probe_new = _node_with_params(candidate, {**shared, **changes})
        if probe_old is None or probe_new is None:
            return None
        descended = candidate is not node
        try:
            state = probe_old.initial_state() if descended else gm._state[owner]
        except Exception:  # noqa: BLE001 - cannot tell
            return None
        running = _what_the_node_computes(spec, probe_new, descended, state, live_after)
        reloaded = _what_the_node_computes(spec, rebuilt_new, descended, state, live_after)
        if running is None or reloaded is None:
            return None
        differences = _computes_the_same(running, reloaded)
        if not differences:
            return None
        before = _what_the_node_computes(spec, probe_old, descended, state, live_before)
        rebuilt_before = _what_the_node_computes(spec, rebuilt_old, descended, state,
                                                 live_before)
        if before is None or rebuilt_before is None \
                or _computes_the_same(before, rebuilt_before):
            return None
        return (
            f"a graph saved with it would reload a node that computes "
            f"something else: {cls.__name__} derives {differ} from its params "
            "when it is constructed, the running node keeps what it derived "
            "from the old value, and with the new value the running node and "
            f"the reloaded one differ: {'; '.join(differences)}"
        )
    return None


#: Methods a cross-origin page could use to change this server's state.
#:
#: ``GET`` and ``HEAD`` are left out deliberately: without an
#: ``Access-Control-Allow-Origin`` header -- which this API never sends
#: -- a cross-origin page cannot read the response, so a read is not an
#: exfiltration path, and refusing one would break embedding the viz
#: pages.  WebSockets are not on this list because they are not HTTP
#: methods; they are checked in :class:`_WebSocketAuthMiddleware`, and
#: they *are* a read path, because WebSocket is exempt from the
#: same-origin policy.
_STATE_CHANGING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


@stability(StabilityLevel.EVOLVING)
def origin_is_same_site(
    origin: Optional[str],
    host: Optional[str],
    allowed_origins: Iterable[str] = (),
) -> bool:
    """Whether a browser's ``Origin`` header names this server.

    The loopback threat model -- "the caller is anyone with a shell on
    this box" -- omits every web page the developer's browser loads.  A
    cross-origin *simple* request (``POST`` with
    ``Content-Type: text/plain``) needs no preflight and reaches the
    handler from any origin, and the peer backstop cannot help because
    the peer is 127.0.0.1 by construction.  A loopback-bound API has no
    legitimate cross-origin caller, so the answer is "same origin, or a
    named one, or no".

    Parameters
    ----------
    origin : str or None
        The ``Origin`` request header.  Absent means the caller is not a
        browser (curl, a script, a test client), which this rule does
        not constrain: ``True``.
    host : str or None
        The ``Host`` request header, which an attacker's page cannot
        choose.  The comparison ignores the scheme, because a
        TLS-terminating proxy makes the browser's ``https`` disagree
        with the server's own view of the connection while the authority
        still matches.
    allowed_origins : iterable of str, optional
        Origins an embedder serves its UI from, matched literally
        (case-insensitively, trailing slash ignored).

    Returns
    -------
    bool
        ``True`` when the request may proceed.  Anything that cannot be
        resolved to this host fails closed -- including ``null`` (a
        sandboxed iframe or a ``file://`` page) and an authority with
        userinfo, ``https://evil.example@host``, which reads as *host*
        to a careless parser and is not one.

    Examples
    --------
    >>> origin_is_same_site("http://127.0.0.1:8000", "127.0.0.1:8000")
    True
    >>> origin_is_same_site("https://evil.example", "127.0.0.1:8000")
    False
    >>> origin_is_same_site(None, "127.0.0.1:8000")
    True
    """
    if not origin:
        return True
    candidate = origin.strip()
    normalised = candidate.rstrip("/").lower()
    for allowed in allowed_origins:
        if normalised == allowed.strip().rstrip("/").lower():
            return True
    parts = urlsplit(candidate)
    if not parts.scheme or not parts.netloc:
        return False
    try:
        hostname, port = parts.hostname, parts.port
    except ValueError:
        return False
    if not hostname or parts.username is not None or parts.password is not None:
        return False
    authority = hostname if port is None else f"{hostname}:{port}"
    return authority == (host or "").strip().lower()


#: The 403 body.  Says what happened and what an embedder does about it.
_CROSS_ORIGIN_DETAIL = (
    "This request carried an Origin header that is not this server's own "
    "origin. A MADDENING server has no legitimate cross-origin caller: on a "
    "loopback bind the API is unauthenticated, so a page on any origin could "
    "otherwise reset the simulation or write a checkpoint from the "
    "developer's browser. If you are embedding the UI on another origin, "
    "pass allowed_origins to SimulationServer."
)


class _WebSocketAuthMiddleware:
    """Default-deny for WebSocket handshakes.

    ``@app.middleware("http")`` builds a Starlette ``BaseHTTPMiddleware``,
    which runs only for ``scope["type"] == "http"``.  Without this, every
    WebSocket handler had to remember to call
    :meth:`SimulationServer._authorise_ws` first, and a handler that
    forgot served an anonymous caller on a ``0.0.0.0`` bind -- measured:
    a ``@app.websocket`` route added with no authorisation passed the
    whole bearer-token suite and accepted the connection.

    This is pure ASGI rather than ``BaseHTTPMiddleware`` for exactly that
    reason.  It refuses before the route is reached, so the handlers'
    own ``_authorise_ws`` calls become the second of two independent
    checks rather than the only one.  A rejected handshake is closed with
    1008, the same code and the same behaviour a handler produces.

    Parameters
    ----------
    app : ASGI application
        The application to wrap.
    auth : maddening.api.auth.APIAuth
        The token and the rule for when it is demanded.
    """

    def __init__(self, app, auth: APIAuth, allowed_origins: Iterable[str] = ()) -> None:
        self.app = app
        self._auth = auth
        self._allowed_origins = frozenset(allowed_origins)

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "websocket":
            await self.app(scope, receive, send)
            return
        client = scope.get("client") or (None,)
        peer = client[0]
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in (scope.get("headers") or ())
        }
        # The origin check runs whether or not a token is demanded.  A
        # loopback bind demands none, and WebSocket is exempt from the
        # same-origin policy, so without this a page on any origin could
        # open /ws/state and *read* the simulation state -- a disclosure,
        # not the write-only exposure the HTTP half has.
        if not origin_is_same_site(
            headers.get("origin"), headers.get("host"), self._allowed_origins,
        ):
            logger.warning(
                "Refused WebSocket %s from origin %r: not this server's origin",
                scope.get("path", "?"), headers.get("origin"),
            )
            await self._refuse(receive, send)
            return
        if self._auth.required_for_peer(peer):
            offered = list(scope.get("subprotocols") or [])
            presented = (
                bearer_from_headers(headers)
                or bearer_from_subprotocols(offered)
            )
            if not self._auth.verify(presented):
                logger.warning(
                    "Refused WebSocket %s from %s: %s bearer token",
                    scope.get("path", "?"), peer or "?",
                    "invalid" if presented else "missing",
                )
                await self._refuse(receive, send)
                return
        await self.app(scope, receive, send)

    @staticmethod
    async def _refuse(receive, send) -> None:
        """Close an unaccepted handshake with 1008.

        The ``websocket.connect`` message has to be consumed first, or
        the server is answering nothing.  A client that hung up before
        sending it delivers ``websocket.disconnect`` instead, and
        answering *that* with a close is a protocol error -- so the
        refusal is simply already complete.
        """
        message = await receive()
        if message.get("type") == "websocket.disconnect":
            return
        await send({"type": "websocket.close", "code": 1008})


@stability(StabilityLevel.EVOLVING)
def warn_if_publicly_bound(host: str, port: int = 8000) -> bool:
    """Log a warning when *host* is not a loopback address.

    Such a bind now demands a bearer token (see
    :class:`maddening.api.auth.APIAuth`), so this is no longer the
    difference between private and open.  It is still the difference
    between a port only this machine can reach and a port on the
    network with **no TLS** in front of it: the token, and every
    simulation state the API returns, cross the wire in cleartext, and
    ``/healthz`` and the ``/viz/*`` pages answer without a credential.
    Call this immediately before handing the app to uvicorn so an
    operator sees what is exposed in the first screen of logs.

    Parameters
    ----------
    host : str
        Bind address about to be passed to the server.
    port : int, optional
        Bind port, for the log line only.

    Returns
    -------
    bool
        ``True`` when a warning was emitted, i.e. *host* is reachable
        from outside this machine.

    Notes
    -----
    This warns; it does not refuse.  Changing the bind default would
    break containerised deployments, where binding 127.0.0.1 makes the
    server unreachable even with a published port.  What refuses is the
    token check, which the same non-loopback bind switches on.
    """
    if is_loopback(host):
        return False
    logger.warning(
        "\n"
        "============================================================\n"
        "  MADDENING API is listening on %s:%s -- NOT loopback, so\n"
        "  every route needs 'Authorization: Bearer <token>'.\n"
        "  There is still NO TLS: the token and every state snapshot\n"
        "  cross the network in cleartext, and /healthz and the\n"
        "  /viz/* pages answer without a credential.\n"
        "  Bind MADDENING_HOST=127.0.0.1 and reach the server through\n"
        "  an SSH tunnel (ssh -L %s:127.0.0.1:%s <host>), or put a\n"
        "  TLS-terminating proxy in front of it and keep this port\n"
        "  closed in the firewall / security group.\n"
        "============================================================",
        host, port, port, port,
    )
    return True


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
    bind_host : str, optional
        The address this server will be bound to.  A loopback address
        leaves the API unauthenticated, as it has always been; anything
        else turns on the bearer token.  The app cannot see the socket
        uvicorn binds, so it has to be told.  ``None`` reads
        ``MADDENING_HOST`` and falls back to ``"127.0.0.1"``.
    api_token : str, optional
        An explicit bearer token, overriding ``MADDENING_API_TOKEN``.
    allowed_origins : iterable of str, optional
        Browser origins allowed to change state or open a WebSocket, for
        an embedder that serves its UI from another port.  The default
        -- none -- means same-origin only, which is the safe answer for
        every shipped configuration: see :func:`origin_is_same_site`.

    Attributes
    ----------
    auth : maddening.api.auth.APIAuth
        The token and the rule for when it is demanded.  Call
        ``auth.announce(port)`` before serving so a generated token
        reaches the log.

    Raises
    ------
    ValueError
        If ``MADDENING_API_TOKEN`` or *api_token* is set but blank.
    """

    def __init__(
        self,
        node_registry: dict[str, type[SimulationNode]],
        graph_manager: Optional[GraphManager] = None,
        checkpoint_root: Optional[str] = None,
        frame_renderer: Optional[Any] = None,
        bind_host: Optional[str] = None,
        api_token: Optional[str] = None,
        allowed_origins: Optional[Iterable[str]] = None,
    ) -> None:
        self.registry = dict(node_registry)
        self.auth = APIAuth(bind_host=bind_host, token=api_token)
        self.allowed_origins = frozenset(allowed_origins or ())
        self.gm = graph_manager if graph_manager is not None else GraphManager()
        # /checkpoint/{save,load} only touch files under this directory:
        # a client must not choose arbitrary server paths.  That holds
        # whether or not the bearer token is enforced -- on a loopback
        # bind the caller is anyone with a shell on this box.
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

    def _unauthorized_detail(self, peer: Optional[str]) -> str:
        """The 401 body: why a token is needed and where to find it."""
        if self.auth.enforced:
            return (
                "This server is bound to a non-loopback address, so every "
                "route requires 'Authorization: Bearer <token>'. The token "
                "is $MADDENING_API_TOKEN, or was logged once at start-up."
            )
        return (
            f"This server was configured for a loopback bind "
            f"({self.auth.bind_host!r}) but the request arrived from "
            f"{peer!r}. A request from off-host needs "
            f"'Authorization: Bearer <token>'. If the bind really is "
            f"public, pass bind_host to SimulationServer (or set "
            f"MADDENING_HOST) so the token is logged at start-up."
        )

    async def _authorise_ws(self, websocket: WebSocket) -> tuple[bool, Optional[str]]:
        """Authenticate a WebSocket handshake before accepting it.

        A browser cannot set ``Authorization`` on a WebSocket handshake,
        but it can offer subprotocols, so the token may arrive either as
        the header (non-browser clients) or inside a
        ``maddening.bearer.*`` subprotocol.  See
        :func:`maddening.api.auth.websocket_credentials`.

        Parameters
        ----------
        websocket : WebSocket
            The un-accepted connection.

        Returns
        -------
        tuple of (bool, str or None)
            ``(False, None)`` when the connection was refused -- it has
            already been closed with 1008 and the caller must return.
            Otherwise ``(True, subprotocol)``, where *subprotocol* is
            what ``accept()`` must echo: RFC 6455 requires the server to
            select one of the offered names, and a browser aborts the
            connection when it selects none.
        """
        offered = list(websocket.scope.get("subprotocols") or [])
        selected = WS_SUBPROTOCOL if WS_SUBPROTOCOL in offered else None
        peer = websocket.client.host if websocket.client else None
        if not self.auth.required_for_peer(peer):
            return True, selected
        presented = (
            bearer_from_headers(websocket.headers)
            or bearer_from_subprotocols(offered)
        )
        if self.auth.verify(presented):
            return True, selected
        logger.warning(
            "Refused WebSocket %s from %s: %s bearer token",
            websocket.url.path, peer or "?",
            "invalid" if presented else "missing",
        )
        await websocket.close(code=1008, reason="Invalid or missing bearer token")
        return False, None

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
        """Build and return the FastAPI application.

        Returns
        -------
        FastAPI
            The application.  When :attr:`auth` is enforced -- i.e. the
            bind address is not loopback -- every route but
            :data:`maddening.api.auth.UNAUTHENTICATED_PATHS` requires
            ``Authorization: Bearer <token>``, and ``/docs``, ``/redoc``
            and ``/openapi.json`` are not served at all.
        """
        # Swagger UI is a browser page that fetches /openapi.json with no
        # Authorization header, so it cannot work behind a bearer token.
        # A half-working docs page that 401s on its own schema is worse
        # than none: when the token is enforced these are switched off,
        # and the way to read them is an SSH tunnel to a loopback bind.
        interactive_docs = not self.auth.enforced
        app = FastAPI(
            title="MADDENING Simulation Server",
            description="HTTP/WebSocket API for the MADDENING simulation graph.",
            # The package version, not a separately maintained API
            # version: this was pinned at "0.3.0" and went stale.
            version=_maddening_version,
            docs_url="/docs" if interactive_docs else None,
            redoc_url="/redoc" if interactive_docs else None,
            openapi_url="/openapi.json" if interactive_docs else None,
        )

        # -- authentication ---------------------------------------------------
        # A middleware rather than a per-route ``Depends`` so that a route
        # added later is protected by default: forgetting the dependency
        # is exactly how this hole gets rebuilt.  It also covers the
        # FastAPI-generated docs routes, which take no dependencies.
        #
        # Two of them, because one cannot cover both: ``@app.middleware("http")``
        # is a BaseHTTPMiddleware and never runs for a WebSocket scope, so
        # the WebSocket half is a pure-ASGI middleware of its own.
        app.add_middleware(
            _WebSocketAuthMiddleware,
            auth=self.auth,
            allowed_origins=self.allowed_origins,
        )

        @app.middleware("http")
        async def _require_bearer_token(request, call_next):
            peer = request.client.host if request.client else None
            if (self.auth.required_for_peer(peer)
                    and request.url.path not in UNAUTHENTICATED_PATHS):
                if not self.auth.verify(bearer_from_headers(request.headers)):
                    logger.warning(
                        "Refused %s %s from %s: %s bearer token",
                        request.method, request.url.path, peer or "?",
                        "invalid" if request.headers.get("authorization")
                        else "missing",
                    )
                    return JSONResponse(
                        status_code=401,
                        content={"detail": self._unauthorized_detail(peer)},
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            if (request.method in _STATE_CHANGING_METHODS
                    and not origin_is_same_site(
                        request.headers.get("origin"),
                        request.headers.get("host"),
                        self.allowed_origins,
                    )):
                logger.warning(
                    "Refused %s %s from origin %r: not this server's origin",
                    request.method, request.url.path,
                    request.headers.get("origin"),
                )
                return JSONResponse(
                    status_code=403,
                    content={"detail": _CROSS_ORIGIN_DETAIL},
                )
            return await call_next(request)

        @app.get("/healthz", tags=["meta"], response_model=None)
        def healthz() -> dict[str, str]:
            """Liveness probe.  Served without a token, on purpose.

            A container health check has no credential, and this answer
            says nothing about the graph -- only that the process is up
            and which version it is.
            """
            return {"status": "ok", "version": _maddening_version}

        # -- visualization endpoints -----------------------------------------

        @app.get("/viz/auth.js", tags=["viz"], response_class=PlainTextResponse)
        def viz_auth_js() -> PlainTextResponse:
            """Serve the token helper the bundled pages load.

            Static and secret-free: it *finds* a token (a ``#token=``
            fragment, then ``?token=``, then ``sessionStorage``, then a
            prompt), it never contains one.  That is why it, and the
            pages that load it, are served without a credential --
            otherwise the page that asks for the token could not load.
            """
            js_path = _STATIC_DIR / "auth.js"
            return PlainTextResponse(
                content=js_path.read_text(),
                media_type="application/javascript",
            )

        @app.get("/viz/graph", tags=["viz"], response_class=HTMLResponse)
        def viz_graph() -> HTMLResponse:
            html_path = _STATIC_DIR / "graph.html"
            return HTMLResponse(content=html_path.read_text(), status_code=200)

        @app.get("/viz/app", tags=["viz"], response_class=HTMLResponse)
        def viz_app() -> HTMLResponse:
            """Serve the interactive simulation app."""
            html_path = _STATIC_DIR / "app.html"
            return HTMLResponse(content=html_path.read_text(), status_code=200)

        @app.get("/viz/render", tags=["viz"], response_class=HTMLResponse)
        def viz_render() -> HTMLResponse:
            """Serve the server-side rendered viewer."""
            html_path = _STATIC_DIR / "render.html"
            return HTMLResponse(content=html_path.read_text(), status_code=200)

        # -- graph structure endpoints ---------------------------------------

        @app.get("/graph", tags=["graph"], response_model=None)
        def get_graph() -> dict[str, Any]:
            # Display, not persistence: a mapping without point references
            # is shown as far as it describes itself rather than refused.
            data = self.gm.to_dict(strict_mappings=False)
            data["active_surrogates"] = list(self._active_surrogates)
            return data

        @app.post("/graph/nodes", tags=["graph"], status_code=201, response_model=None)
        def add_node(req: AddNodeRequest) -> dict[str, Any]:
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
            # AddNodeRequest bounds each integer the caller sends, which is
            # what keeps a single dimension from naming hundreds of MB.
            # Dimensions that multiply survive that bound, so the state the
            # node actually built is measured too -- before the node joins
            # the graph, so an oversized one is transient rather than
            # resident for the life of the process.
            try:
                initial_state = node.initial_state()
            except Exception as exc:  # noqa: BLE001 - a constant it cannot use
                raise HTTPException(
                    status_code=400,
                    detail=f"node '{req.name}' cannot build its initial state: {exc}",
                )
            n_elements = _state_elements(initial_state)
            if n_elements > MAX_NODE_STATE_ELEMENTS:
                del initial_state, node
                raise HTTPException(
                    status_code=400,
                    detail=(f"node '{req.name}' would hold {n_elements} state "
                            f"elements; at most {MAX_NODE_STATE_ELEMENTS} are "
                            f"accepted over the API (this server is "
                            f"unauthenticated -- build a graph this size "
                            f"in-process)"),
                )
            # Nodes do not validate their constants; a bad one only fails
            # inside the trace and would wedge every later /sim/step.
            # Trace one update on the node's own initial state (abstractly,
            # no compute) before it enters the graph.
            try:
                _dry_run_node(node, initial_state)
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

        @app.delete("/graph/nodes/{name}", tags=["graph"], response_model=None)
        def remove_node(name: str) -> dict[str, str]:
            try:
                self.gm.remove_node(name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))
            return {"status": "ok"}

        @app.post("/graph/edges", tags=["graph"], status_code=201, response_model=None)
        def add_edge(req: AddEdgeRequest) -> dict[str, str]:
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

        @app.delete("/graph/edges", tags=["graph"], response_model=None)
        def remove_edge(req: RemoveEdgeRequest) -> dict[str, str]:
            self.gm.remove_edge(
                source=req.source_node, target=req.target_node,
                source_field=req.source_field, target_field=req.target_field,
            )
            return {"status": "ok"}

        @app.post("/graph/compile", tags=["graph"], response_model=None)
        def compile_graph() -> dict[str, Any]:
            try:
                self.gm.compile()
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return {"status": "ok", "schedule": self.gm.schedule}

        @app.post("/graph/validate", tags=["graph"], response_model=None)
        def validate_graph() -> dict[str, Any]:
            issues = self.gm.validate()
            return {"issues": issues}

        # -- state endpoints -------------------------------------------------

        @app.get("/graph/state", tags=["state"], response_model=None)
        def get_state() -> dict[str, Any]:
            return self._state_json()

        @app.get("/graph/state/{node_name}", tags=["state"], response_model=None)
        def get_node_state(node_name: str) -> dict[str, Any]:
            try:
                return self._node_state_json(node_name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc))

        @app.put("/graph/state/{node_name}", tags=["state"], response_model=None)
        def set_node_state(node_name: str, req: SetNodeStateRequest) -> dict[str, str]:
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

        @app.get("/graph/params/{node_name}", tags=["params"], response_model=None)
        def get_node_params(node_name: str) -> dict[str, Any]:
            if node_name not in self.gm._nodes:
                raise HTTPException(status_code=404, detail=f"No node '{node_name}'.")
            node = self.gm._nodes[node_name].node
            # The live pytree wins over the constructor value: it is what
            # the step uses after a fit or a checkpoint restore.
            live = self.gm.params.get("nodes", {}).get(node_name) or {}
            return {**_jax_to_python(node.params), **_jax_to_python(live)}

        @app.put("/graph/params/{node_name}", tags=["params"], response_model=None)
        def set_node_params(node_name: str, req: SetNodeParamsRequest) -> dict[str, Any]:
            """Update node parameters.

            Float parameters of nodes that accept injected params are
            written to ``gm.params`` and take effect on the next step
            without recompiling; anything else (structural values, nodes
            on the legacy contract) marks the graph dirty as before.  An
            initial condition the node reads in ``initial_state()`` takes
            effect at the next ``POST /sim/reset``.

            A value the running node cannot use -- one it consumed when it
            was constructed (declared in ``static_data_deps``, or read by
            neither the step, its hooks nor ``initial_state()``) -- is a
            400 naming the parameter and why, and nothing in the request
            is written: rebuild the node to change it.  So is a value that
            changes the shape of the state the node builds (``n_cells``, a
            grid shape), sharded or not: the running state keeps its shape,
            and the step recompiled for the new value either failed on it
            or, sharded, stepped it on a grid it does not have.  And so is
            a structural value the node's constructor refuses: a graph saved
            with it could not be loaded.
            """
            if node_name not in self.gm._nodes:
                raise HTTPException(status_code=404, detail=f"No node '{node_name}'.")
            # Before anything else, and for every key, as POST /graph/nodes
            # does: a non-finite structural value used to skip the live-leaf
            # checks below, be written into node.params, and fail only in
            # the reply's JSON encoder -- a 500 after the write, and every
            # later GET /graph/params a 500 as well.
            bad = _non_finite_param(req.params)
            if bad is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"params.{bad}: value must be finite",
                )
            node = self.gm._nodes[node_name].node
            live = self.gm.params.get("nodes", {}).get(node_name) or {}
            specs = self.gm.param_specs().get("nodes", {}).get(node_name, {})
            # Before the first compile ``gm.params`` is empty; validate a
            # params node against its own pytree so the same request gets
            # the same answer one compile() earlier or later.
            probe_only = False
            # The graph's own answer, recorded by ``add_node`` from the one
            # params rule (``maddening.core.node._method_accepts_params``).
            # This used to read ``node.accepts_params`` and treat a node
            # object without that method as taking no params, which the
            # graph -- and so the ``live`` branch one compile later -- did
            # not.
            if not live and self.gm._nodes[node_name].accepts_params:
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
            # What the write would store in node.params, for every key that
            # changes; an unchanged key has nothing to refuse.
            ctor = node.params_pytree() if staged else {}
            changes: dict[str, Any] = {}
            for key, value in req.params.items():
                if key in staged:
                    ref = ctor.get(key, live.get(key))
                    if ref is not None and _leaf_values_equal(staged[key], ref):
                        continue
                    changes[key] = np.asarray(staged[key]).tolist()
                else:
                    if key in node.params and _same_param_value(node.params[key], value):
                        continue
                    changes[key] = value

            def refused(keys, reason: str, *, reported: bool = False) -> HTTPException:
                what = ("a new value for this parameter" if len(keys) == 1
                        else "these values together")
                also = ("  The write would be reported, and saved by to_dict() / "
                        "save_state(), as the value in force." if reported else "")
                return HTTPException(
                    status_code=400,
                    detail=(
                        f"{', '.join(keys)}: node '{node_name}' cannot take {what} "
                        f"while it runs: {reason}.{also}  Nothing was written; to "
                        "change it, rebuild the node (DELETE "
                        f"/graph/nodes/{node_name}, then POST /graph/nodes with "
                        "the new value)."
                    ),
                )

            # A state of another layout -- or over the API's state cap -- is
            # refused whatever else is true of the value, and the checks
            # below that establish it on concrete values build the node and
            # its state at the new size first.  Told here without building
            # anything, where the node's initial_state() can be evaluated
            # abstractly; one key at a time for the message, then together
            # (two dimensions that multiply).
            for key, node_value in changes.items():
                reason = _state_write_reason_before_building(node, {key: node_value})
                if reason is not None:
                    raise refused([key], reason)
            if len(changes) > 1:
                reason = _state_write_reason_before_building(node, changes)
                if reason is not None:
                    raise refused(list(changes), reason)
            # A value the node consumed when it was constructed (a wall
            # mask, an assembled operator, a copy of an initial condition)
            # is not rebuilt by writing node.params: the write would be
            # echoed, served by GET, written out by to_dict() /
            # save_state(), and ignored by every step.  Refused here, still
            # before anything is written, by the graph's own decision
            # (``GraphManager._unused_node_write_reason``), which is the
            # one that refuses the same leaf written into gm.params alone.
            for key, node_value in changes.items():
                # A structural value is checked against the node's own
                # constructor: the saved graph is rebuilt through it.  (Every
                # key, together and with a save's params, is checked below.)
                shape_reason = (
                    self.gm._constructor_write_reason(node_name, key, node_value)
                    if key not in staged else None
                ) or self.gm._state_shape_write_reason(node_name, key, node_value)
                if shape_reason is not None:
                    raise refused([key], shape_reason)
                reason = self.gm._unused_node_write_reason(node_name, key, node_value)
                if reason is not None:
                    raise refused([key], reason, reported=True)
            # And the graph a save after the write would reload must load,
            # and run what the running graph runs: the constructor asked
            # with every changed key at once and every other key's live
            # value (a live leaf used to skip it), and a node whose
            # constructor derives something from the value -- a branch, an
            # array -- asked whether the rebuilt node computes what the
            # running one does with it.
            saved = {k: v for k, v in changes.items() if k in node.params}
            if saved:
                accepts = self.gm._nodes[node_name].accepts_params
                reason = _saved_graph_write_reason(
                    self.gm, node_name, saved,
                    dict(live) if accepts else None,
                    {**live, **staged} if accepts else None,
                )
                if reason is not None:
                    raise refused(list(saved), reason)
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

        @app.post("/checkpoint/save", tags=["checkpoint"], response_model=None)
        def checkpoint_save(path: str = "checkpoint.npz") -> dict[str, str]:
            target = _checkpoint_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                saved = self.gm.save_state(str(target))
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=400, detail=f"could not save checkpoint: {exc}")
            return {"status": "ok", "path": str(saved or target)}

        @app.post("/checkpoint/load", tags=["checkpoint"], response_model=None)
        def checkpoint_load(path: str = "checkpoint.npz") -> dict[str, Any]:
            target = _checkpoint_path(path)
            if not target.exists() and not target.with_suffix(target.suffix + ".npz").exists():
                raise HTTPException(status_code=404, detail=f"no checkpoint {path!r}")
            from maddening.core.simulation.checkpoint import (  # noqa: PLC0415
                _restore_state_and_params,
                _state_and_params_snapshot,
            )
            try:
                # load_state compiles a dirty graph before it reads anything;
                # done first here so the undo below starts after it.
                if self.gm._dirty or self.gm._compiled_step is None:
                    self.gm.compile()
                undo = _state_and_params_snapshot(self.gm)
                self.gm.load_state(str(target))
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception:  # noqa: BLE001 - do not leak file/parse internals
                raise HTTPException(status_code=400, detail=f"could not load checkpoint {path!r}")
            # A checkpoint of a graph whose node was built with another value
            # of a parameter it consumes at construction carries that value
            # in gm.params.  The graph refuses such a leaf at the next step,
            # and this API has no reset_params: every later /sim/step would
            # be a 500 while GET /graph/params served the checkpoint's value.
            # Refused here instead, with the load undone.
            try:
                self.gm._refuse_baked_param_writes(self.gm.params, live=False)
            except ValueError as exc:
                _restore_state_and_params(self.gm, undo)
                raise HTTPException(
                    status_code=400,
                    detail=f"checkpoint {path!r} does not fit this graph, nothing "
                           f"was loaded: {exc}",
                )
            return {"status": "ok", "state": self._state_json()}

        # -- simulation control endpoints -----------------------------------

        @app.post("/sim/step", tags=["sim"], response_model=None)
        def sim_step() -> dict[str, Any]:
            try:
                self.gm.step()
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return self._state_json()

        @app.post("/sim/run", tags=["sim"], response_model=None)
        def sim_run(
            n_steps: int = Query(
                100, ge=0, le=MAX_RUN_STEPS,
                description="Steps to run synchronously.  Zero is a no-op "
                            "that returns the current state.  The upper "
                            "bound exists because the request holds a "
                            "worker for its whole duration and cannot be "
                            "cancelled; for a longer run use POST "
                            "/sim/start.",
            ),
        ) -> dict[str, Any]:
            try:
                self.gm.run(n_steps)
            except RuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            return self._state_json()

        @app.post("/sim/start", tags=["sim"], response_model=None)
        def sim_start() -> dict[str, str]:
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

        @app.post("/sim/pause", tags=["sim"], response_model=None)
        def sim_pause() -> dict[str, str]:
            if self.runner is None or not self._runner_started:
                raise HTTPException(status_code=409, detail="Runner is not started.")
            self.runner.pause()
            return {"status": "paused"}

        @app.post("/sim/resume", tags=["sim"], response_model=None)
        def sim_resume() -> dict[str, str]:
            if self.runner is None or not self._runner_started:
                raise HTTPException(status_code=409, detail="Runner is not started.")
            self.runner.resume()
            return {"status": "resumed"}

        @app.post("/sim/stop", tags=["sim"], response_model=None)
        def sim_stop() -> dict[str, str]:
            if self.runner is None or not self._runner_started:
                raise HTTPException(status_code=409, detail="Runner is not started.")
            self._stop_runner()
            return {"status": "stopped"}

        @app.post("/sim/reset", tags=["sim"], response_model=None)
        def sim_reset() -> dict[str, Any]:
            """Stop the runner and reset all nodes to initial state."""
            was_running = self._runner_started
            self._stop_runner()
            self._reset_state()
            self.gm._dirty = True
            return {"status": "ok", "was_running": was_running, "state": self._state_json()}

        @app.put("/sim/stride", tags=["sim"], response_model=None)
        def sim_set_stride(steps_per_frame: int = 1, relay_stride: int = 1) -> dict[str, int]:
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

        @app.post("/surrogate/train", tags=["surrogate"], response_model=None)
        def surrogate_train(req: TrainSurrogateRequest) -> dict[str, str]:
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

        @app.get("/surrogate/status/{job_id}", tags=["surrogate"], response_model=None)
        def surrogate_status(job_id: str) -> dict[str, Any]:
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

        @app.post("/surrogate/activate/{job_id}", tags=["surrogate"], response_model=None)
        def surrogate_activate(job_id: str) -> dict[str, str]:
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

        @app.post("/surrogate/deactivate/{node_name}", tags=["surrogate"], response_model=None)
        def surrogate_deactivate(node_name: str) -> dict[str, str]:
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
            # A mapped edge's weights live in ``gm.params["mappings"][key]``,
            # and its ParamSpec overrides under the same key; ``remove_node``
            # drops both, and the ``compile()`` below re-snapshots the
            # weights from the mapping object -- silently reverting a fit
            # made while the surrogate was active.  Read them off the *live*
            # graph, which is where such a fit landed.
            live_mappings = (self.gm.params or {}).get("mappings") or {}
            saved_mapping_params = {
                edge.key: dict(live_mappings[edge.key])
                for edge in orig_edges
                if edge.mapping is not None and edge.key in live_mappings
            }
            saved_edge_specs = {
                edge.key: dict(self.gm._param_spec_overrides[edge.key])
                for edge in orig_edges
                if edge.key in self.gm._param_spec_overrides
            }
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
                        # boundary input the graph used to add to.  Derived
                        # from the dataclass, so a field added to EdgeSpec
                        # cannot quietly stop being restored here.
                        self.gm.add_edge(**edge.add_edge_kwargs())
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
                restored_keys = {e.key for e in self.gm._edges}
                missing = sorted(set(saved_mapping_params) - restored_keys)
                if missing:
                    # ``add_edge`` recomputes ``ordinal``, and the key it
                    # builds from it names the weights' slot: put them back
                    # under a key no edge answers to and they are attached
                    # to nothing, or to the wrong edge.
                    raise RuntimeError(
                        f"restored edges do not carry the saved mapping "
                        f"key(s) {missing}"
                    )
                if saved_mapping_params:
                    self.gm.params.setdefault("mappings", {}).update(
                        saved_mapping_params)
                for edge_key, overrides in saved_edge_specs.items():
                    for param_key, spec in overrides.items():
                        self.gm.set_param_spec(edge_key, param_key, spec)
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

        @app.post("/sim/profile", tags=["sim"], response_model=None)
        def sim_profile(n_steps: int = 50, n_warmup: int = 3) -> dict[str, Any]:
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

        @app.post("/sim/profile/jax/start", tags=["sim"], response_model=None)
        def sim_profile_jax_start() -> dict[str, Any]:
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

        @app.post("/sim/profile/jax/stop", tags=["sim"], response_model=None)
        def sim_profile_jax_stop() -> dict[str, Any]:
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

        @app.get("/sim/profile/jax/status", tags=["sim"], response_model=None)
        def sim_profile_jax_status() -> dict[str, Any]:
            from maddening.core.simulation.profiler import jax_trace_active
            return {
                "active": jax_trace_active(),
                "last_trace_dir": getattr(self, "_last_jax_trace_dir", None),
            }

        # -- cloud endpoints ------------------------------------------------

        @app.post("/cloud/launch", tags=["cloud"], response_model=None)
        def cloud_launch(config: dict[str, Any] = {}) -> dict[str, Any]:
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

        @app.get("/cloud/status", tags=["cloud"], response_model=None)
        def cloud_status() -> dict[str, Any]:
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

        @app.post("/cloud/teardown", tags=["cloud"], response_model=None)
        def cloud_teardown() -> dict[str, Any]:
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
        async def ws_state(websocket: WebSocket) -> None:
            """Stream state snapshots as JSON at ~30 Hz.

            Client may send JSON messages to configure the stream:

            * ``{"type": "subscribe", "fields": {"node": ["f1", "f2"]}}``
              — only include the listed node/field pairs in subsequent
              snapshots.  Send ``{"type": "subscribe", "fields": null}``
              to reset to full state.
            * ``{"type": "config", "fps": 15}`` — change poll rate.

            Authentication is the same rule as every HTTP route; see
            :meth:`SimulationServer._authorise_ws`.
            """
            authorised, subprotocol = await self._authorise_ws(websocket)
            if not authorised:
                return
            await websocket.accept(subprotocol=subprotocol)
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
        async def ws_state_binary(websocket: WebSocket) -> None:
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

            Authentication is the same rule as every HTTP route; see
            :meth:`SimulationServer._authorise_ws`.
            """
            authorised, subprotocol = await self._authorise_ws(websocket)
            if not authorised:
                return
            await websocket.accept(subprotocol=subprotocol)
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
        async def ws_render(websocket: WebSocket) -> None:
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
            # Authenticate first: whether this deployment configured a
            # renderer is not something an anonymous caller gets to learn.
            authorised, subprotocol = await self._authorise_ws(websocket)
            if not authorised:
                return

            if self._frame_renderer is None:
                await websocket.close(
                    code=1008,
                    reason="No frame renderer configured on this server.",
                )
                return

            await websocket.accept(subprotocol=subprotocol)
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
