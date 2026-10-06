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
A **loopback bind is unauthenticated** for a direct connection from a
loopback address, exactly as it always was: bind ``127.0.0.1`` and nothing
changes for local development.  It answers only to the names this machine
is reached by (``localhost``, ``127.0.0.1``, ``[::1]``, and any
``allowed_hosts``) and refuses a state change from a foreign ``Origin``,
so a web page in the developer's browser -- DNS rebinding included --
cannot drive it.  A peer that is not a loopback IP literal, or a request
that carries ``X-Forwarded-For`` / ``Forwarded``, must present the token
(:mod:`maddening.api.auth`: under uvicorn's default ``proxy_headers`` the
peer is whatever a request arriving over loopback says it is; never
configure loopback as a trusted proxy).  **Any other
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
import atexit
import collections
import concurrent.futures
import contextlib
import functools
import inspect
import ipaddress
import json
import logging
import math
import os
import re
import signal
import threading
import time
import uuid
import weakref
from pathlib import Path
from typing import Annotated, Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit

import jax
import jax.numpy as jnp
import numpy as np

try:
    from fastapi import FastAPI, HTTPException, Query, WebSocket
    from fastapi.encoders import jsonable_encoder
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response
    from pydantic import BaseModel, Field, field_validator
    from starlette.exceptions import HTTPException as StarletteHTTPException
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
    _carries_forwarding_header,
    bearer_from_headers,
    bearer_from_subprotocols,
    is_loopback,
)
from maddening.core._quiet_warnings import quiet_warnings
from maddening.core._size_estimate import (
    AllocationEstimate,
    constructor_arguments,
    estimate_allocation,
    format_bytes,
    format_count,
)
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import register_route_stability, stability
from maddening.core.graph_manager import (
    EVENT_COMPILED,
    EVENT_NODE_ADDED,
    EVENT_NODE_REMOVED,
    EVENT_STEP,
    GraphManager,
    _BakedParamWrite,
)
from maddening.core._graph_specs import (
    _UNCARRIABLE_WHY,
    _node_name_refusal,
    _NodeSpec,
    _uncarriable_characters,
)
from maddening.core._param_probes import (
    _declared_boundary_zeros,
    _hook_outputs,
    _leaf_values_equal,
    _node_update_layout_drift,
    _node_with_params,
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


def _json_reply(value: Any) -> Any:
    """``_jax_to_python(value)`` with every non-finite float written as its
    quoted token -- ``"NaN"``, ``"Infinity"``, ``"-Infinity"`` -- for a
    reply body.

    The encoding of every MADDENING JSON surface
    (:mod:`maddening.serialization.json_codec`), and the one ``GET /graph``
    already serves (``GraphManager.to_dict`` encodes its tree).  The state
    replies used to hand Starlette bare floats, which it serialises with
    ``allow_nan=False``: a graph with a ``diagnostics=True`` coupling group
    (whose spectral ``_meta`` slots are seeded NaN until a solve fills
    them), or a simulation that diverged, made ``GET /graph/state`` a 500 --
    and ``POST /sim/reset``, ``POST /checkpoint/load`` and ``POST /sim/step``
    a 500 *after* their change had been applied.  A float a reply cannot
    carry is now written the way a config carries it, and
    :func:`~maddening.serialization.json_codec.loads` (or a check for the
    three strings) reads it back.
    """
    from maddening.serialization.json_codec import (  # noqa: PLC0415
        encode_non_finite,
    )
    return encode_non_finite(_jax_to_python(value))


_SURROGATES = re.compile(r"[\ud800-\udfff]")
#: What a reply carries where a string held a surrogate (U+FFFD).
_REPLACEMENT_CHARACTER = "\ufffd"
#: The body of a reply nothing else could be made of (see :class:`_Reply`).
_UNWRITABLE_REPLY = b'{"detail":"this reply could not be written as JSON"}'


def _reply_tree(value: Any) -> Any:
    """*value* as any JSON reply can carry it; never raises for a tree a
    request can cause.

    * A non-finite float is its quoted token, as in every reply
      (:func:`_json_reply`).
    * A string is kept as it is, **also when it spells one of those
      tokens**.  That is the difference from :func:`_json_reply`, which
      refuses such a string because a reader of *data* would decode it as
      a float: an error's ``detail`` echoes what the caller sent
      (``POST /sim/run?n_steps=NaN``) and is read by a person, so there the
      text is written back as the text it was.
    * A surrogate in a string, which UTF-8 cannot encode (a body may spell
      one, ``"\\ud800"``), is written as U+FFFD.
    * A key that is not a string is its ``str``.
    """
    if isinstance(value, str):
        return _SURROGATES.sub(_REPLACEMENT_CHARACTER, value)
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        from maddening.serialization.json_codec import (  # noqa: PLC0415
            INF_TOKEN,
            NAN_TOKEN,
            NEG_INF_TOKEN,
        )
        if math.isnan(value):
            return NAN_TOKEN
        return INF_TOKEN if value > 0 else NEG_INF_TOKEN
    if isinstance(value, dict):
        return {_reply_tree(key if isinstance(key, str) else str(key)): _reply_tree(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_reply_tree(item) for item in value]
    return value


class _Reply(JSONResponse):
    """The JSON reply of every route, refusal and exception handler of the
    app: a ``JSONResponse`` whose encoder cannot fail on what a request can
    put in a reply.

    Starlette's encoder raises for a non-finite float (``allow_nan=False``)
    and for a string that holds a surrogate (it cannot be UTF-8), and
    :func:`_json_reply` raises for a string that spells ``NaN``,
    ``Infinity`` or ``-Infinity``.  Each of those raised *inside* the code
    that was writing a refusal, so the refusal became a 500:
    ``POST /sim/run?n_steps=NaN`` -- what a browser sends for
    ``parseInt("")`` -- where ``n_steps=abc`` was a 422, and
    ``{"type": "\\ud800"}`` on ``POST /graph/nodes`` where ``"abc"`` was
    a 400.  A reply Starlette can encode is encoded by Starlette, byte for
    byte as before; one it cannot is written by :func:`_reply_tree`; and
    if that fails too (a tree nested past the interpreter's depth) the
    reply keeps its status and says only that it could not be written.
    """

    def render(self, content: Any) -> bytes:
        try:
            return super().render(content)
        except (ValueError, TypeError, RecursionError):  # UnicodeEncodeError is a ValueError
            pass
        try:
            return json.dumps(
                _reply_tree(content), ensure_ascii=False, allow_nan=False,
                separators=(",", ":"), default=str,
            ).encode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - a reply is written whatever it holds
            return _UNWRITABLE_REPLY


def _text_value_refusal(value: Any) -> Optional[str]:
    """Why a parameter's *value* holds text the graph's config could not
    carry, or ``None``.

    A text parameter's value is a JSON string in ``to_dict``, in the
    node's own 201 and in ``GET /graph/params``.  The text ``NaN``,
    ``Infinity`` or ``-Infinity`` there is refused by the config's encoder
    (it is how a non-finite float is written, and would be read back as
    one), and a surrogate by every file: a node of a registered class with
    a text parameter was added with ``"label": "NaN"``, its own reply was
    a 500, and ``GET /graph`` was one until the node was deleted.
    """
    from maddening.serialization.json_codec import (  # noqa: PLC0415
        NON_FINITE_TOKENS,
    )
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if item in NON_FINITE_TOKENS:
                return (f"the text {item!r} is how a config writes a non-finite "
                        "number, so a graph holding it could not be written as "
                        f"one (MADD-ANO-010); a different spelling ({item.lower()!r}, "
                        "say) is fine")
            if _SURROGATES.search(item):
                return (f"the text {item!r} holds a surrogate, which no file "
                        "can carry")
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return None


def _unrepresentable(value: Any, dtype: Any) -> Optional[str]:
    """Why a JSON *value* cannot be held by a leaf of float ``dtype``, or
    ``None``: a finite number the dtype overflows to an infinity (``1e39``
    into ``float32``), or a non-zero one it underflows to zero (``1e-50``),
    which loses the value as entirely.  One that rounds to a subnormal
    keeps its sign and magnitude and is held.

    Asked *before* ``jnp.asarray(value, dtype=...)``, because that cast is
    where NumPy says so -- a ``RuntimeWarning`` ("overflow encountered in
    cast"), a 500 wherever warnings are errors -- and the infinity it
    produces was then refused as "must be finite", a value the caller never
    sent.  The test of the FMU's own write paths
    (``maddening.fmi.sidecar._checked_value``), with its words.  A value
    that is not numeric at all is left to the cast, whose ``TypeError`` /
    ``ValueError`` is the 400 it always was.
    """
    dt = np.dtype(dtype)
    if not np.issubdtype(dt, np.floating):
        return None
    try:
        wide = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        narrow = wide.astype(dt)
    if bool(np.any(np.isfinite(wide) & ~np.isfinite(narrow))) \
            or bool(np.any((wide != 0) & (narrow == 0))):
        return f"value does not fit its type {dt}"
    return None


def _state_value_refusal(value: Any, dtype: Any) -> Optional[str]:
    """Why a JSON *value* (a number, or nested lists of them) cannot be
    written into a state field of *dtype*, or ``None``; asked of every
    element before anything is cast.

    The rule every other write surface applies (``PUT /graph/params``,
    ``POST /checkpoint/load`` through ``checkpoint._checked_cast``, the
    FMU): text, a boolean for a numeric field, ``null`` and anything that
    is not a number are refused; an integer field takes only integral
    values inside its dtype's range; a boolean field takes booleans, or 0
    and 1.  ``PUT /graph/state`` cast with ``jnp.asarray`` alone: ``"1.5"``,
    ``" 2 "`` and ``true`` were parsed as numbers, 0.5 and 1.7 were
    truncated to 0 and 1 in LBMNode's ``uint8`` wall mask, and 256, -1 or
    1e10 there were a 500 (``OverflowError``).  A float field's range is
    :func:`_unrepresentable`'s to say, and its finiteness the route's.
    """
    dt = np.dtype(dtype)
    stack = [value]
    while stack:
        v = stack.pop()
        if isinstance(v, (list, tuple)):
            stack.extend(v)
            continue
        if isinstance(v, bool):
            if dt.kind == "b":
                continue
            return "expected a number, got a boolean"
        if v is None:
            return "expected a number, got null"
        if isinstance(v, str):
            return "expected a number, got a string"
        if not isinstance(v, (int, float)):
            return f"expected a number, got {type(v).__name__}"
        if dt.kind == "b":
            if v not in (0, 1):
                return f"value {v!r} is not a boolean (true, false, 0 or 1)"
        elif dt.kind in "iu":
            if isinstance(v, float) and not v.is_integer():
                return f"value {v!r} is not an integer, and the field holds {dt}"
            info = np.iinfo(dt)
            if not info.min <= v <= info.max:
                return (f"value {v!r} is outside the range of {dt} "
                        f"[{info.min}, {info.max}]")
    return None


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
#: product not).  20e6 float32 elements is 80 MB.  Enforced *before* the
#: node is constructed for a class that can say what it would build (see
#: :data:`MAX_NODE_BUILD_BYTES`), and on the built state for any other.
MAX_NODE_STATE_ELEMENTS = 20_000_000

#: Upper bound on the memory constructing one node and building its initial
#: state may take, for a node class that can say so before it is built (the
#: built-in grid and basis nodes: ``HeatNode``, ``LBMNode``,
#: ``LBMPipeNode``, ``WaveletAdaptiveNode``).  Checked, with
#: :data:`MAX_NODE_STATE_ELEMENTS`, *before* any constructor call in
#: ``POST /graph/nodes`` and ``PUT /graph/params``: both used to build the
#: node first and refuse it afterwards, and one ``PUT
#: {"n_levels": 10000000}`` to a wavelet node grew the server to 57.7 GB
#: before the kernel killed it.  2 GiB is several times what the largest
#: lattice the state cap admits takes to build (about 0.3 GB) and admits
#: ``WaveletAdaptiveNode``'s validated range (a 4096-function basis --
#: 64^2, 16^3 -- assembles a dense operator in about 1 GiB).
MAX_NODE_BUILD_BYTES = 2 * 1024 ** 3

#: Upper bound on ``PUT /sim/stride?steps_per_frame=``: one frame of the
#: background runner is at most as many steps as one ``POST /sim/run``.
#: The runner checks its stop and pause flags between every step, so a
#: large value no longer delays ``POST /sim/stop``; the bound keeps one
#: frame's work, and the gap between relay snapshots, finite.
MAX_STEPS_PER_FRAME = MAX_RUN_STEPS

#: Upper bound on ``PUT /sim/stride?relay_stride=``: the relay keeps every
#: Nth step, and a stride past one run's length would never publish one.
MAX_RELAY_STRIDE = MAX_RUN_STEPS

#: How long ``POST /sim/stop`` (and every route that stops the runner
#: first) waits for the runner's thread to finish the step it is in.
#: Past it the route answers 503 and keeps the runner: the thread is
#: still alive, and reporting "stopped" -- or dropping the handle -- while
#: it steps is how a reset used to be overwritten.
_RUNNER_STOP_TIMEOUT = 10.0

#: Bounds on ``POST /surrogate/train``.
MAX_SURROGATE_DATA_STEPS = 100_000
MAX_SURROGATE_EPOCHS = 10_000
MAX_SURROGATE_BATCH_SIZE = 65_536
MAX_SURROGATE_LAYERS = 16
MAX_SURROGATE_LAYER_WIDTH = 8192

#: Upper bound on ``width ** 2 * depth`` of a ``POST /surrogate/train``
#: network, *width* its widest hidden layer and *depth* its number of
#: hidden layers: the hidden weights, which the two bounds above only
#: bound one at a time (8192 x 16 is a billion weights, 4 GB of them
#: before the optimiser copies them).  2**24 float32 weights are 64 MiB;
#: training holds about twelve copies (the weights, their gradient, Adam's
#: two moments and each step's new values), so about 0.75 GiB at the
#: bound, within :data:`MAX_SURROGATE_TRAIN_BYTES`.  It admits 4096 x 1,
#: 2048 x 4 and 1024 x 16, and the shipped UI's 64 x 2 by a wide margin.
MAX_SURROGATE_HIDDEN_WEIGHTS = 2 ** 24

#: Upper bound on the memory one ``POST /surrogate/train`` job may take,
#: estimated before its worker starts (:func:`_surrogate_training_bytes`):
#: the data sweep over the whole graph (16 initial conditions, up to 200
#: steps each, every node's history), the dataset made from it and the
#: copies training takes of it, and the network with its optimiser.  The
#: same bound as building one node, :data:`MAX_NODE_BUILD_BYTES`: one
#: request may name about that much, and no more.  One in-cap request
#: used to take 2.2 GB (a 25 000-cell rod) to 5.9 GB and an out-of-memory
#: error (an 8192 x 16 network).
MAX_SURROGATE_TRAIN_BYTES = MAX_NODE_BUILD_BYTES

#: How many finished ``POST /surrogate/train`` jobs (and their trained
#: networks) are kept for ``GET /surrogate/status`` and ``POST
#: /surrogate/activate``; the oldest finished job is dropped past it.  One
#: job runs at a time.  Every job used to be kept for the life of the
#: process, and any number ran at once.
MAX_SURROGATE_JOBS_KEPT = 8

#: Upper bound on the scalars of the whole graph's state, summed over
#: every node, after a ``POST /graph/nodes``: each node is held to
#: :data:`MAX_NODE_STATE_ELEMENTS`, and nothing bounded how many such
#: nodes a caller could add.  Five nodes at the per-node cap, 400 MB of
#: float32 state; a step holds the old and new state and a few temporaries
#: of it, about 1.6 GB at the bound -- the order of
#: :data:`MAX_NODE_BUILD_BYTES` again.  Checked before the node is built
#: for a class that can say what it would build, and on the built state
#: for any other.
MAX_GRAPH_STATE_ELEMENTS = 5 * MAX_NODE_STATE_ELEMENTS

#: Upper bound on a request body, enforced by a pure-ASGI middleware
#: before anything is parsed: a ``Content-Length`` above it is a 413
#: without reading the body, and a body sent without one is counted as it
#: arrives and refused the moment it passes it.  The largest body any
#: bounded field admits is :data:`MAX_NODE_PARAM_ELEMENTS` numbers in
#: ``POST /graph/nodes`` / ``PUT /graph/params``, at most 26 bytes each in
#: JSON (``-1.2345678901234567e-308, ``), about 26 MB; this is the next
#: power of two.  Parsing JSON takes 25-40 times the body, so about
#: 1.3 GB at the bound, again the order of :data:`MAX_NODE_BUILD_BYTES`.
#: Bodies were unbounded, and parsed whole before any cap was checked.
MAX_REQUEST_BODY_BYTES = 32 * 1024 ** 2

#: Upper bound on the WebSocket streams (``/ws/state``,
#: ``/ws/state/binary``, ``/ws/render``) open at once; past it a handshake
#: is closed with 1013 ("try again later").  Each stream encodes or renders
#: the state on its own, so the streams' share of the server grew with
#: every client.
MAX_STREAM_CONNECTIONS = 16

#: How long a request waits for the graph lock (see
#: :class:`SimulationServer`) before answering 503: another request, the
#: runner's step or a surrogate data sweep is using the graph.  A step
#: holds the lock for one step and ``POST /sim/run`` for a slice of about
#: :data:`_RUN_SLICE_SECONDS`, so this is reached only behind something
#: long -- a first compile, a large checkpoint -- and keeps a queue of
#: waiting requests from holding the server's worker threads.
_GRAPH_LOCK_TIMEOUT = 30.0

#: Threads of the small pool the runner routes ``POST /sim/start``,
#: ``/sim/stop``, ``/sim/pause`` and ``/sim/resume`` run their blocking work
#: on, apart from the worker pool every other route shares (``POST
#: /sim/reset`` has a pool of its own, :data:`_RESET_ROUTE_WORKERS`, since
#: it waits for the graph without holding the runner lock; ``PUT
#: /sim/stride`` waits for nothing and runs on the event loop).  They used
#: to run on the shared pool, so behind a queue of requests waiting for the
#: graph a stop waited for a free worker before its own deadline even
#: started (5.6 s behind 240 queued reads, at a 1 s lock timeout).  Each
#: takes the runner lock with a deadline counted from its arrival, and a
#: start holds that lock while it waits for the graph, so everything queued
#: here is bounded by the deadline of a request that arrived earlier: a
#: request is answered within about one :data:`_GRAPH_LOCK_TIMEOUT` of its
#: arrival however many are queued.
_RUNNER_ROUTE_WORKERS = 4

#: Threads of ``POST /sim/reset``'s own pool (see
#: :data:`_RUNNER_ROUTE_WORKERS`): a reset waits for the graph with a
#: deadline counted from its arrival, and on the runner routes' pool a few
#: of them would keep a stop waiting for a thread.
_RESET_ROUTE_WORKERS = 2

#: ``POST /sim/run`` steps in slices of about this many seconds, taking
#: the graph lock for each and checking between them whether the server
#: is shutting down; reads are served between slices.
_RUN_SLICE_SECONDS = 0.05

#: The most steps a JAX trace started by ``POST /sim/profile/jax/start``
#: records before it stops itself.  JAX's profiler holds a trace's events
#: in memory until the trace stops -- about 4.2 KB per step of a one-node
#: graph, more for a graph of more kernels -- and nothing stopped a trace
#: left running: a runner at 60 steps/s grew the server by about 22 GB a
#: day.  10 000 steps is about 40 MB for a small graph, and far more steps
#: than a trace viewer is useful for.
MAX_JAX_TRACE_STEPS = 10_000

#: The longest a JAX trace runs before it stops itself, in seconds: it is
#: stopped at the first step after this.
MAX_JAX_TRACE_SECONDS = 600.0

#: The stability of the routes whose level is set apart from the rest of
#: the API.  The surrogate-training routes and the state streams are
#: **experimental in 0.4.0**: they are to be hardened in 0.5.0, and may
#: change in any minor release until then.  None of them carried a level
#: before (``SimulationServer`` itself has none), so this is their first.
#: Registered in the stability report as ``maddening.api.server:<route>``;
#: an HTTP route here also carries ``x-maddening-stability`` in
#: ``/openapi.json``.
ROUTE_STABILITY: dict[str, StabilityLevel] = {
    "POST /surrogate/train": StabilityLevel.EXPERIMENTAL,
    "GET /surrogate/status/{job_id}": StabilityLevel.EXPERIMENTAL,
    "POST /surrogate/activate/{job_id}": StabilityLevel.EXPERIMENTAL,
    "POST /surrogate/deactivate/{node_name}": StabilityLevel.EXPERIMENTAL,
    "WS /ws/state": StabilityLevel.EXPERIMENTAL,
    "WS /ws/state/binary": StabilityLevel.EXPERIMENTAL,
    "WS /ws/render": StabilityLevel.EXPERIMENTAL,
}
for _route, _level in ROUTE_STABILITY.items():
    register_route_stability(__name__, _route, _level)
del _route, _level


def _route_openapi(method: str, path: str) -> Optional[dict]:
    """``openapi_extra`` for an HTTP route of :data:`ROUTE_STABILITY`."""
    level = ROUTE_STABILITY.get(f"{method} {path}")
    return None if level is None else {"x-maddening-stability": level.value}



def _oversized_param(value: Any, path: str = "", *, integers: bool = True) -> Optional[str]:
    """Why *value* is too big to accept as a node parameter, else ``None``.

    Integers are bounded because a node constructor turns one into an
    array dimension -- the auditor measured +433 MB of RSS from a single
    unauthenticated ``POST /graph/nodes``.  Element counts are bounded
    because a parameter may itself be a large array.  Floats are not
    bounded: a float is a physical constant, not a dimension, and any cap
    on one would be arbitrary.

    With ``integers=False`` only the element count is asked: the request
    models know no parameter's type, and a JSON number does not say
    whether it is an integer -- a browser's ``JSON.stringify(2e7)`` is
    ``20000000`` -- so the routes bound an integer only where its target
    is one (:func:`_oversized_new_node_param`, and ``PUT``'s coerced
    value), and an integral JSON number for a float parameter is a float.
    Both models used to bound every integer, so a stiffness of 2e7 N/m
    sent by the bundled ``app.html`` was a 422.

    This check and :func:`_non_finite_param` partition the numbers rather
    than racing for them: an integer literal too large to be a ``float``
    at all (JSON allows one of any length) is *not* a dimension anyone
    could mean, it is the unusable constant ``_non_finite_param`` already
    owns, so it is left to that check and its 400 "value must be finite".
    What this function rejects is the plausible-but-too-big dimension,
    and it does so from the request model, as a 422.
    """
    total = containers = 0
    stack: list[tuple[Any, str]] = [(value, path)]
    while stack:
        item, where = stack.pop()
        if isinstance(item, (dict, list, tuple)):
            # Lists and objects are counted too, against the same bound: a
            # body of eleven million empty lists holds no value at all, so
            # it passed this check and was refused for its shape only after
            # twenty seconds of converting it, inside the graph's lock.
            containers += 1
            if containers > MAX_NODE_PARAM_ELEMENTS:
                return (f"params: at most {MAX_NODE_PARAM_ELEMENTS} lists and objects "
                        f"in total (this server is unauthenticated; see its README)")
            if isinstance(item, dict):
                stack.extend((v, f"{where}.{k}" if where else str(k))
                             for k, v in item.items())
            else:
                stack.extend((v, f"{where}[{i}]") for i, v in enumerate(item))
            continue
        total += 1
        if total > MAX_NODE_PARAM_ELEMENTS:
            return (f"params: at most {MAX_NODE_PARAM_ELEMENTS} values in total "
                    f"(this server is unauthenticated; see its README)")
        if isinstance(item, bool) or not integers:
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


def _constructor_default(cls: Any, key: str) -> Any:
    """The default *cls*'s constructor gives parameter *key*, or ``None``
    when it gives none (a required parameter, one taken by ``**kwargs``) or
    the signature cannot be read -- read by the one helper that binds a
    request to a constructor (:func:`~maddening.core._size_estimate.constructor_arguments`)."""
    return (constructor_arguments(cls, {}) or {}).get(key)


def _holds_text(value: Any) -> bool:
    """Whether *value* is, or holds anywhere inside it, a string."""
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            return True
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return False


def _is_numeric_value(value: Any) -> bool:
    """A number that is not a boolean, or a non-empty list of them: the
    value of a numeric constructor parameter."""
    if isinstance(value, (list, tuple)):
        return bool(value) and all(_is_numeric_value(v) for v in value)
    return isinstance(value, (int, float, np.integer, np.floating)) \
        and not isinstance(value, (bool, np.bool_))


def _new_node_type_refusal(cls: Any, params: dict[str, Any]) -> Optional[str]:
    """Why ``POST /graph/nodes`` refuses a value for its parameter's type,
    or ``None``: a boolean or text for a parameter whose constructor
    default is a number.  ``PUT /graph/params`` refuses both for a numeric
    parameter, and every FMU door takes numbers only; the constructor took
    ``damping: true`` and the value dropped out of the params pytree."""
    for key, value in params.items():
        default = _constructor_default(cls, key)
        if not _is_numeric_value(default):
            continue
        if isinstance(value, bool) or (isinstance(value, (list, tuple))
                                       and any(isinstance(v, bool) for v in value)):
            return f"{key}: expected a number, got a boolean"
        if _holds_text(value):
            return f"{key}: expected a number, got a string"
    return None


def _oversized_new_node_param(cls: Any, params: dict[str, Any]) -> Optional[str]:
    """:func:`_oversized_param`'s integer bound on ``POST /graph/nodes``'s
    *params*, asked of each value as its parameter's type reads it
    (:func:`_coerced_to_param_type` against the constructor's default, the
    rule ``PUT /graph/params`` applies to a structural value): an integer
    for a float parameter is a float and is not bounded, an integral float
    for an integer parameter is an integer and is.  A parameter whose
    default says nothing (none, ``None``) is bounded as written."""
    for key, value in params.items():
        target, problem = _coerced_to_param_type(_constructor_default(cls, key), value)
        found = _oversized_param(value if problem is not None else target, str(key))
        if found is not None:
            return found
    return None


# ------------------------------------------------------------------
# Pydantic request/response models
# ------------------------------------------------------------------

class AddNodeRequest(BaseModel):
    type: str
    name: str
    # A finite number > 0: NaN or Infinity added the node and then answered
    # 500 (its reply could not be encoded), every step a 400 until it was
    # deleted; 0 or a negative value stepped it not at all, or backwards.
    timestep: float = Field(gt=0, allow_inf_nan=False)
    params: dict[str, Any] = {}

    @field_validator("params")
    @classmethod
    def _params_within_bounds(cls, value: dict[str, Any]) -> dict[str, Any]:
        # The element count only: an integer is bounded by the route, which
        # knows whether its parameter is an integer (_oversized_param).
        problem = _oversized_param(value, integers=False)
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
        # The element count only: an integer is bounded by the route, which
        # knows whether its parameter is an integer (_oversized_param).
        problem = _oversized_param(value, integers=False)
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

    # The width and depth bounds hold one number each; the hidden weights
    # grow with width squared times depth, so that is bounded too.
    @field_validator("hidden_sizes")
    @classmethod
    def _hidden_weights_within_bounds(cls, value: list[int]) -> list[int]:
        weights = max(value) ** 2 * len(value)
        if weights > MAX_SURROGATE_HIDDEN_WEIGHTS:
            raise ValueError(
                f"hidden_sizes: width**2 * depth is {weights} (widest layer "
                f"{max(value)}, {len(value)} layers); at most "
                f"{MAX_SURROGATE_HIDDEN_WEIGHTS} hidden weights are accepted "
                "(this server is unauthenticated -- train a network this size "
                "in-process)")
        return value


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

    Asked of the value :func:`_coerced_to_param_type` returns, so a JSON
    ``2.0`` for an integer ``2`` is the same value and nothing is written.
    Strict otherwise: values of different types differ, and anything that
    does not compare cleanly counts as changed.  "Changed" only means the
    write is checked.
    """
    if type(old) is not type(new):
        return False
    try:
        return bool(old == new)
    except Exception:  # noqa: BLE001 - an array-valued entry, say
        return False


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _coerced_to_param_type(old: Any, new: Any) -> tuple[Any, Optional[str]]:
    """``(value, problem)``: a JSON *new* value for a structural parameter
    whose current value is *old*, in *old*'s numeric type -- or ``problem``,
    why it cannot be.

    A JSON number does not say whether it is an integer (``4`` and ``4.0``
    are one number to most clients), and ``node.params`` is what
    ``params_pytree()``, ``to_dict()`` and every rebuild read.  Writing the
    raw value used to change the parameter's type: ``PUT`` HeatNode
    ``stencil_order: 4.0`` stored a float, which ``params_pytree()`` then
    exposed as a new *trainable* leaf of ``gm.params``, and an integer
    written for a structural float parameter stored an ``int``.  (A leaf of
    the params pytree is cast to its own dtype before this is asked.)  The
    rule:

    * an integer parameter takes an integer, or a float with no fractional
      part, stored as ``int`` (the convention the constructors' own count
      checks follow); any other float is refused;
    * a float parameter takes a float or an integer, stored as ``float``;
    * a list of integers (a grid shape) takes integers or integral floats,
      element by element, stored as ``int``; a list of floats takes
      numbers, stored as ``float``;
    * anything else is passed through unchanged for the checks after this
      one (a constructor refuses what it cannot use).

    Booleans are decided before this is asked.
    """
    if old is None or isinstance(old, bool) or isinstance(new, bool):
        return new, None
    if _is_integer(old):
        if _is_integer(new):
            return int(new), None
        if isinstance(new, float):
            if math.isfinite(new) and new.is_integer():
                return int(new), None
            return None, f"expected an integer (the parameter is {old!r}), got {new!r}"
        return new, None
    if isinstance(old, (float, np.floating)):
        if _is_integer(new):
            return float(new), None
        return new, None
    if isinstance(old, (list, tuple)) and isinstance(new, list) and old \
            and not any(isinstance(x, (list, tuple, dict)) for x in old):
        if all(_is_integer(x) for x in old):
            out = []
            for item in new:
                if isinstance(item, float) and not isinstance(item, bool):
                    if not (math.isfinite(item) and item.is_integer()):
                        return None, (f"expected integers (the parameter is {old!r}), "
                                      f"got {new!r}")
                    item = int(item)
                out.append(item)
            return out, None
        if all(isinstance(x, (float, np.floating)) for x in old):
            return [float(x) if _is_integer(x) else x for x in new], None
    return new, None


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
    """Refuse (``ValueError``) a freshly built node the graph could not
    step with its state's layout unchanged: one ``update`` is traced
    abstractly on the node's own initial state the way the graph calls it
    -- its params pytree injected where it takes one, with a zero for every
    declared boundary input and again with none -- and what it returns is compared with that
    state, leaf by leaf (names, shapes, kinds of dtype:
    :func:`~maddening.core._param_probes._state_layout_drift`).  Raises
    whatever the node's code raises while it is traced.

    The trace used to call ``update`` without the params pytree and to
    discard what it returned.  So a constant the step reads from the
    pytree (``HeartPumpNode`` ``venous_pressure: null``) was accepted and
    every later step was a 400; and a list where the node's state is a
    scalar (``BallNode`` ``initial_velocity: [0, 0]``) was accepted, the
    state changed shape at the first step (``GraphManager.step`` stores
    such a state: MADD-ANO-220), and the graph's own checkpoint no longer
    loaded after a reset.

    *state* is the node's initial state when the caller has already built
    it (the size check does), so it is not allocated twice.
    """
    state = node.initial_state() if state is None else state
    accepts = _method_accepts_params(node, "update")
    spec = _NodeSpec(node=node, update_fn=node.update, timestep=node.delta_t,
                     accepts_params=accepts)
    leaves = node.params_pytree() if accepts else None
    # With every declared input delivered (the node once its edges are
    # added), then with none, as the graph steps it until then: a constant
    # read only in the absence of an input is read by every step of the
    # node as it is added.  (Warnings are not silenced: the trace is the
    # node's first, as the graph's would be.)
    drift = _node_update_layout_drift(spec, state, leaves)
    if not drift:
        try:
            drift = _node_update_layout_drift(spec, state, leaves, {})
        except KeyError:
            # A node that needs an input it declares cannot step until its
            # edge is added, and says so at the step: not refused.
            drift = []
    if drift:
        raise ValueError(
            "one update changes the layout of its state (" + "; ".join(drift)
            + "), so its checkpoint would not load after a reset")


def _dry_run_refusal(node_cls: Any, name: str, timestep: float, params: dict[str, Any],
                     node: Any, state: Any) -> Optional[str]:
    """Why ``POST /graph/nodes`` refuses *node* -- built by *node_cls* from
    *params* -- for what :func:`_dry_run_node` finds, or ``None``.  The
    reason names the parameters it can be told from: those without which
    (each left to the class's default in turn) the node passes."""
    try:
        _dry_run_node(node, state)
    except Exception as exc:  # noqa: BLE001 - any trace failure is a refusal
        reason = str(exc)
    else:
        return None
    blamed = []
    for key in params:
        try:
            _dry_run_node(node_cls(name=name, timestep=timestep,
                                   **{k: v for k, v in params.items() if k != key}))
        except Exception:  # noqa: BLE001 - not this parameter alone
            continue
        blamed.append(key)
    where = (" (told from " + ", ".join(f"params.{k}" for k in blamed) + ")") if blamed else ""
    return f"cannot run with these params{where}: {reason}"


#: What a step raises because of the graph it is asked to run -- a node's
#: code failing on its params while the step is traced and compiled, a
#: structure ``compile()`` refuses -- rather than because the server is
#: broken.  ``POST /sim/step`` and ``POST /sim/run`` answer these 400 with
#: the message; they used to answer every one but ``RuntimeError`` with an
#: uncaught 500.  A failing ``/sim/step`` stores nothing; a ``/sim/run``
#: whose step raises at run time part-way through keeps the steps before
#: it, and its 400 says how many (``steps_run``).
_GRAPH_CONFIGURATION_ERRORS = (
    RuntimeError, ValueError, TypeError, KeyError, AttributeError, IndexError,
    ArithmeticError,
    # ``compile()`` reports every edge it cannot validate at once, as an
    # ExceptionGroup of ValueErrors: an edge between fields whose shapes do
    # not match, which ``POST /graph/edges`` accepts.  It was a 500.
    ExceptionGroup,
)


def _configuration_reason(exc: BaseException) -> str:
    """What a graph that cannot compile said, on one line: every member of
    the group ``compile()`` raises for its edges, or the error itself."""
    if isinstance(exc, ExceptionGroup):
        return f"{exc.message}: " + "; ".join(
            f"{type(e).__name__}: {e}" for e in exc.exceptions)
    return f"{type(exc).__name__}: {exc}"


def _cannot_step_detail(exc: BaseException) -> str:
    """The 400 body for a step that raised *exc*."""
    if isinstance(exc, RuntimeError):
        return str(exc)
    if isinstance(exc, ExceptionGroup):
        reasons = "; ".join(f"{type(e).__name__}: {e}" for e in exc.exceptions)
        return (f"the graph cannot step with its current configuration "
                f"({exc.message}: {reasons}); nothing was stepped")
    return (f"the graph cannot step with its current configuration "
            f"({type(exc).__name__}: {exc}); nothing was stepped")


def _overflowing_float(value: Any, path: str = "") -> Optional[str]:
    """The path of the first finite float inside *value* that the default
    floating dtype overflows to an infinity (``1e39`` under float32), else
    ``None``.

    For a structural parameter -- a value ``node.params`` stores as Python
    data (RigidBodyNode's ``constraints``), which the node turns into an
    array of the default dtype when it is traced.  A leaf of the params
    pytree is checked against its own dtype (:func:`_unrepresentable`);
    a structural value was not checked at all, so ``{"z": 1e39}`` was
    answered 200 and every later step produced infinities.
    """
    limit = float(np.finfo(jax.dtypes.canonicalize_dtype(jnp.float64)).max)
    stack: list[tuple[Any, str]] = [(value, path)]
    while stack:
        item, where = stack.pop()
        if isinstance(item, dict):
            stack.extend((v, f"{where}.{k}" if where else str(k)) for k, v in item.items())
        elif isinstance(item, (list, tuple)):
            stack.extend((v, f"{where}[{i}]") for i, v in enumerate(item))
        elif isinstance(item, float) and math.isfinite(item) and abs(item) > limit:
            return where or "value"
    return None


def _json_value_count(value: Any) -> tuple[int, int]:
    """``(numbers, lists)``: how many numbers a JSON *value* holds (nested
    lists flattened, a scalar is one) and how many lists, counted without
    converting it to an array."""
    total = lists = 0
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            lists += 1
            stack.extend(item)
        else:
            total += 1
    return total, lists


def _lists_in_shape(shape: tuple) -> int:
    """How many lists the JSON spelling of an array of *shape* holds: one
    for the array, one per row, and so on down to the last axis."""
    lists, rows = 0, 1
    for dim in shape:
        lists += rows
        rows *= int(dim)
    return lists


class _GraphLock:
    """The re-entrant lock around every use of a server's graph: granted
    first come, first served, and able to say whether the calling thread
    holds it.

    ``acquire`` / ``release`` as ``threading.RLock`` (the interface
    :class:`~maddening.viz.runner.RealtimeRunner` takes), plus
    :meth:`acquire_unless`, which the runner uses to wait for its turn
    while watching its stop event.  :meth:`held` is what lets a route
    refuse to wait for the runner's thread while it holds the lock that
    thread needs for its next step.

    First come, first served, because a ``threading.RLock`` is not: a
    runner behind its schedule releases the lock after a step and takes it
    straight back, and a request waiting for it could wait for many steps
    -- 36 s for twenty parameter writes on a two-core CI runner.  Here a
    thread that asks while others are waiting queues behind them, so a
    request waits for at most the step in flight.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._owner: Optional[int] = None
        self._depth = 0
        self._queue: collections.deque = collections.deque()

    def acquire(self, blocking: bool = True, timeout: float = -1,
                cancelled: Optional[Any] = None) -> bool:
        """Take the lock, after every thread that asked for it earlier.

        Parameters
        ----------
        blocking : bool
            ``False`` takes it only if it is free and nobody is waiting.
        timeout : float
            Seconds to wait; negative (the default) or ``None`` waits for
            as long as it takes.
        cancelled : callable, optional
            Asked between waits of at most 50 ms; when it returns true the
            wait is given up.  The thread keeps its place in the queue
            while it waits.

        Returns
        -------
        bool
            Whether the lock is now held.
        """
        me = threading.get_ident()
        with self._cond:
            if self._owner == me:
                self._depth += 1
                return True
            if self._owner is None and not self._queue:
                self._owner, self._depth = me, 1
                return True
            if not blocking:
                return False
            deadline = (None if timeout is None or timeout < 0
                        else time.monotonic() + timeout)
            ticket = object()
            self._queue.append(ticket)
            try:
                while self._owner is not None or self._queue[0] is not ticket:
                    if cancelled is not None and cancelled():
                        return False
                    wait = 0.05 if cancelled is not None else None
                    if deadline is not None:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return False
                        wait = remaining if wait is None else min(wait, remaining)
                    self._cond.wait(wait)
                self._queue.popleft()
                self._owner, self._depth = me, 1
                return True
            finally:
                if self._owner != me and ticket in self._queue:
                    # Gave up: the next in line may be at the head now.
                    self._queue.remove(ticket)
                    self._cond.notify_all()

    def acquire_unless(self, event: threading.Event) -> bool:
        """Wait for the lock unless *event* is set first; ``False`` (not
        held) when it is."""
        return self.acquire(cancelled=event.is_set)

    def release(self) -> None:
        with self._cond:
            if self._owner != threading.get_ident():
                raise RuntimeError("release of a graph lock this thread does not hold")
            self._depth -= 1
            if self._depth == 0:
                self._owner = None
                self._cond.notify_all()

    def held(self) -> bool:
        """Whether the calling thread holds the lock."""
        return self._owner == threading.get_ident()

    def __enter__(self) -> "_GraphLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.release()


def _refuse_dropped_geometry(gm: GraphManager, name: str, held: Any, incoming: Any) -> None:
    """A 409 if putting *incoming* in place of node *name* (*held*) would
    leave an edge's geometry-dependent mapping reading a field *incoming*
    does not hold with that shape and a float32 or float64 dtype
    (experimental).  Asked by the surrogate routes before they change
    anything, with the library's own rule (``replace_node``'s), so the
    refusal is the route's and says what is wrong: left to
    ``replace_node`` inside the transaction it was put back as well, but
    answered the generic 500."""
    from maddening.core._graph_specs import (  # noqa: PLC0415
        _refuse_unpreserved_geometry,
    )
    try:
        _refuse_unpreserved_geometry(gm._edges, name, held, incoming)
    except ValueError as exc:
        detail = str(exc).replace(" Nothing was changed.",
                                  " Nothing was changed; the graph is as it was.")
        raise HTTPException(status_code=409, detail=detail) from None


def _surrogate_training_bytes(gm: GraphManager, node_name: str,
                              req: "TrainSurrogateRequest") -> int:
    """The memory a ``POST /surrogate/train`` job for *req* would take,
    estimated from the graph's shapes before anything is built.

    The job runs a sweep of :data:`_SURROGATE_CONDITIONS` initial
    conditions over the whole graph for ``S = min(200, n_data_steps)``
    steps, keeping every node's history; pairs the target node's states
    with the next ones into a dataset; and trains a network on it.  So:

    * **the sweep's history**: ``C * S * E_graph`` values;
    * **the dataset** (states, next states and boundary inputs of the
      target node, each input at the size its edge delivers -- a mapped
      edge's ``n_target``, not its source field's):
      ``D = C * (S - 1) * (2 * E_node + E_inputs)`` values,
      held three times at the peak -- the dataset, the training/validation
      split and each epoch's shuffle (measured: a 25 000-cell rod at the
      defaults took 2.18 GiB, this says 2.08 GiB);
    * **the network**: ``P`` weights, the input and output layers sized by
      the node's state (``in * w + (d - 1) * w**2 + w * out``), about
      twelve copies of them while training (the weights, the gradient,
      Adam's two moments and each step's new values: measured 0.47 GiB
      above the fixed cost for a 2048 x 4 network), and a batch's
      activations, forward and backward;

    in the graph's widest float dtype.
    """
    def size(leaf: Any) -> int:
        n = 1
        for dim in getattr(leaf, "shape", ()):
            n *= int(dim)
        return n

    itemsize = 4
    e_graph = 0
    for name, fields in gm._state.items():
        if name == "_meta":
            continue
        for leaf in jax.tree_util.tree_leaves(fields):
            e_graph += size(leaf)
            dtype = getattr(leaf, "dtype", None)
            if dtype is not None and np.issubdtype(np.dtype(dtype), np.floating):
                itemsize = max(itemsize, np.dtype(dtype).itemsize)
    e_node = sum(size(leaf) for leaf in jax.tree_util.tree_leaves(gm._state[node_name]))
    e_inputs = 0
    for edge in gm._edges:
        if edge.target_node != node_name:
            continue
        source = gm._state.get(edge.source_node, {})
        delivered = size(source[edge.source_field]) if edge.source_field in source else 1
        if edge.mapping is not None and edge.source_field in source:
            # The dataset holds what the edge delivers: a mapping turns the
            # source field's first axis into its ``n_target`` entries.
            trailing = tuple(getattr(source[edge.source_field], "shape", ()))[1:]
            delivered = int(edge.mapping.n_target) * math.prod(int(d) for d in trailing)
        e_inputs += delivered
    for ext in gm._external_inputs:
        if ext.target_node == node_name:
            e_inputs += max(1, math.prod(int(d) for d in (ext.shape or ())))
    conditions = _SURROGATE_CONDITIONS
    steps = min(_SURROGATE_STEPS_PER_CONDITION, req.n_data_steps)
    history = conditions * steps * e_graph
    dataset = conditions * max(steps - 1, 1) * (2 * e_node + e_inputs)
    width, depth = max(req.hidden_sizes), len(req.hidden_sizes)
    n_in, n_out = e_node + e_inputs + 1, e_node
    weights = n_in * width + (depth - 1) * width ** 2 + width * n_out + depth * width + n_out
    activations = 3 * req.batch_size * (n_in + depth * width + n_out)
    return itemsize * (history + 3 * dataset + 12 * weights + activations)


#: The sweep ``POST /surrogate/train`` generates its data with: this many
#: initial conditions, of at most this many steps each.
_SURROGATE_CONDITIONS = 16
_SURROGATE_STEPS_PER_CONDITION = 200


class _TrainingCancelled(Exception):
    """Raised from a training job's progress callback when the server is
    shutting down: the job ends at the epoch it is in."""


class _OverBudget(Exception):
    """A training job's sweep, estimated again on the graph it would run
    over, is past :data:`MAX_SURROGATE_TRAIN_BYTES`: the job ends with
    status ``"error"`` before anything is swept."""


#: Servers whose surrogate jobs are cancelled and joined at interpreter
#: exit (see :meth:`SimulationServer._cancel_training_jobs`).
_LIVE_SERVERS: "weakref.WeakSet[SimulationServer]" = weakref.WeakSet()


@atexit.register
def _cancel_training_jobs_at_exit() -> None:
    """A training job's thread is a daemon, and one still inside XLA when
    the interpreter tore down its runtime aborted the process ("terminate
    called after throwing an instance of ...", a core dump).  At exit each
    job is told to stop and its thread is joined, so it is out of XLA
    first."""
    for server in list(_LIVE_SERVERS):
        server._cancel_training_jobs(timeout=30.0)


def _body_too_large_detail(limit: int) -> str:
    return (f"Request body is larger than {format_bytes(limit)}, the most this "
            "server accepts (MAX_REQUEST_BODY_BYTES): nothing was parsed or "
            "changed.  Send fewer values; a graph this size is built "
            "in-process.")


class _BodyTooLarge(StarletteHTTPException):
    """A request body passed :data:`MAX_REQUEST_BODY_BYTES` while it was
    being read.  An HTTP exception, so that FastAPI's body reader -- which
    turns any other exception into "There was an error parsing the body"
    (400) -- passes it on as the 413 it is."""

    def __init__(self, limit: int) -> None:
        super().__init__(status_code=413, detail=_body_too_large_detail(limit))


class _RequestBodyLimitMiddleware:
    """Refuse a request body over :data:`MAX_REQUEST_BODY_BYTES` with 413,
    before anything parses it.

    Pure ASGI, below only the authentication middlewares (which read no
    body): a ``Content-Length`` over the limit is refused without reading a
    byte of the body, and a body sent without one
    (chunked) is counted as the application reads it -- the read that
    takes it past the limit raises, and the 413 is sent in place of
    whatever the application would have answered.  Nothing bounded a body
    before: every element cap ran on the parsed request, and parsing a
    76 MB body of numbers took 2-3 GB.
    """

    def __init__(self, app) -> None:
        self.app = app

    @staticmethod
    async def _refuse(send, limit: int) -> None:
        body = json.dumps({"detail": _body_too_large_detail(limit)}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        limit = MAX_REQUEST_BODY_BYTES
        for key, value in scope.get("headers") or ():
            if key.lower() == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = None
                if declared is not None and declared > limit:
                    logger.warning("Refused %s %s: Content-Length %s over %s",
                                   scope.get("method"), scope.get("path"), declared, limit)
                    await self._refuse(send, limit)
                    return
        received = 0
        started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body") or b"")
                if received > limit:
                    raise _BodyTooLarge(limit)
            return message

        async def tracking_send(message):
            nonlocal started
            if message.get("type") == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except _BodyTooLarge:
            if started:
                raise
            logger.warning("Refused %s %s: body over %s bytes",
                           scope.get("method"), scope.get("path"), limit)
            await self._refuse(send, limit)


def _state_layout(state: dict) -> tuple:
    """``((node, field, shape), ...)`` of a user state, sorted: what a
    binary stream's schema depends on."""
    return tuple(
        (node, field, tuple(int(d) for d in getattr(value, "shape", ())))
        for node, fields in sorted(state.items()) if node != "_meta"
        for field, value in sorted(fields.items())
    )


def _encode_state_frame(sim_time: float, state: dict, fields: Optional[dict]) -> str:
    """One ``/ws/state`` frame, as the text Starlette's ``send_json`` would
    send: *state* filtered to *fields* (``{node: [field, ...]}``, ``None``
    for all of it), every non-finite float as its quoted token.  Run in a
    worker thread, never on the event loop: for a large state it takes
    tenths of a second, and every other request waited for it."""
    if fields is not None:
        state = {
            node: {f: values[f] for f in fields.get(node, []) if f in values}
            for node, values in state.items()
            if node in fields
        }
    return json.dumps({"sim_time": sim_time, "state": _json_reply(state)},
                      separators=(",", ":"), ensure_ascii=False)


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
    """``(old, new, descended)``: shallow copies of the node a write of
    every entry of *changes* into ``node.params`` lands on, reading the
    current params and the written ones -- the node, or the node a wrapper
    that cannot be copied wraps (``_param_probe_pair``'s rule, for several
    keys at once; ``descended`` says it is the wrapped one).  ``None`` when
    no faithful copy can be made.  Nothing is constructed."""
    shared = getattr(node, "params", None)
    if not isinstance(shared, dict):
        return None
    for candidate in _params_holders(node):
        old = _node_with_params(candidate, dict(shared))
        new = _node_with_params(candidate, {**shared, **changes})
        if old is not None and new is not None:
            return old, new, candidate is not node
    return None


def _allocation_refusal(estimate: AllocationEstimate) -> Optional[str]:
    """Why a node that would allocate *estimate* is refused over the API
    (:data:`MAX_NODE_STATE_ELEMENTS`, :data:`MAX_NODE_BUILD_BYTES`), or
    ``None``.  Phrased to follow "the node" / "node 'x'"."""
    if estimate.state_elements > MAX_NODE_STATE_ELEMENTS:
        return (f"would hold {format_count(estimate.state_elements)} state "
                f"elements; at most {MAX_NODE_STATE_ELEMENTS} are accepted over "
                "the API (this server is unauthenticated -- build a graph this "
                "size in-process)")
    if estimate.peak_bytes > MAX_NODE_BUILD_BYTES:
        return (f"would take about {format_bytes(estimate.peak_bytes)} to build; "
                f"at most {format_bytes(MAX_NODE_BUILD_BYTES)} is accepted over "
                "the API (this server is unauthenticated -- build a graph this "
                "size in-process)")
    return None


def _graph_budget_refusal(name: str, held: int, adding: int) -> Optional[str]:
    """Why adding node *name* with *adding* state elements to a graph that
    holds *held* is refused for :data:`MAX_GRAPH_STATE_ELEMENTS`, or
    ``None``."""
    if held + adding <= MAX_GRAPH_STATE_ELEMENTS:
        return None
    return (f"node '{name}' would bring the graph to {format_count(held + adding)} "
            f"state elements (it holds {held}, the node {format_count(adding)}); "
            f"at most {MAX_GRAPH_STATE_ELEMENTS} are accepted over the API in the "
            "whole graph (this server is unauthenticated -- build a graph this "
            "size in-process)")


def _allocation_write_reason(gm: GraphManager, owner: str,
                             changes: dict[str, Any]) -> Optional[tuple[list, str]]:
    """``(keys, reason)``: why a write of *changes* into node *owner*'s
    params is refused for the size of what the checks after this one would
    build, and the keys it is refused for; or ``None``.

    Those checks call the node's constructor with the new values -- one key
    at a time (:meth:`GraphManager._constructor_write_reason`, the mapping
    check), every key at once, and every key at once over the params a save
    would carry (:func:`_saved_graph_write_reason`) -- and build the state
    it makes.  Each of those calls is estimated here first, for every class
    a write lands on (the node, or the node a wrapper wraps), by the class's
    own ``_allocation_estimate``; nothing is constructed.  A class without
    an estimate is not checked here, and its built state is bounded by the
    checks that follow, as before.
    """
    node = gm._nodes[owner].node
    shared = getattr(node, "params", None)
    if not isinstance(shared, dict):
        return None
    ctor = {k: v for k, v in changes.items() if k in shared}
    if not ctor:
        return None
    candidates = [([key], {**shared, key: value}) for key, value in ctor.items()]
    candidates.append((list(ctor), {**shared, **ctor}))
    try:
        candidates.append((list(ctor), {**gm.effective_node_params(owner), **ctor}))
    except Exception:  # noqa: BLE001 - the graph cannot say what it would save
        pass
    for holder in _params_holders(node):
        cls = type(holder)
        for keys, params in candidates:
            estimate = estimate_allocation(cls, params)
            if estimate is None:
                continue
            refusal = _allocation_refusal(estimate)
            if refusal is not None:
                return keys, (f"with it the {cls.__name__} {refusal}; this is told "
                              "from the parameters, before anything of that size "
                              "is built")
    return None


def _trace_hooks(spec, probe: Any, descended: bool, state: Any, leaves: Any) -> None:
    """Trace *probe*'s hooks the way the graph calls them (``update``, and
    the flux and interface-correction hooks where it has them) on *state*
    with a zero for every declared boundary input and *leaves* as the
    injected params.  Raises whatever the node's code raises."""
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
    with quiet_warnings():
        bi = _declared_boundary_zeros(probe)
        jax.make_jaxpr(
            lambda st, b, p: _hook_outputs(probe_spec, st, b, p),
        )(state, bi, leaves)


def _step_trace_write_reason(gm: GraphManager, owner: str, changes: dict[str, Any],
                             leaves_before: Any, leaves_after: Any) -> Optional[str]:
    """Why the running node could not step with *changes* written into its
    params, or ``None``: its hooks trace with the current params and raise
    with the new ones (for a reason other than needing a concrete value).

    The other checks of ``PUT /graph/params`` ask whether a value is used,
    and a trace that raised counted as "cannot tell" -- so a value the step
    cannot run with was taken: ``RigidBodyNode`` ``constraints`` of
    ``null``, ``{"w": 0}``, ``{"y": {"finite": true}}`` or ``{"z": "high"}``
    each answered 200, after which every ``POST /sim/step`` was a 500.
    ``POST /graph/nodes`` dry-runs a new node's update for the same reason
    (:func:`_dry_run_node`).  Traced on shallow copies (the node is never
    written), abstractly: nothing is computed.  *leaves_before* and
    *leaves_after* are the injected params the step runs with before and
    after the write, or ``None`` for a node that takes none.
    """
    spec = gm._nodes[owner]
    pair = _probe_pair_with(spec.node, changes)
    if pair is None:
        return None
    old, new, descended = pair
    try:
        state = old.initial_state() if descended else gm._state[owner]
    except Exception:  # noqa: BLE001 - cannot tell
        return None
    try:
        _trace_hooks(spec, old, descended, state, leaves_before)
    except Exception:  # noqa: BLE001 - the running node is not traced this way
        return None
    try:
        _trace_hooks(spec, new, descended, state, leaves_after)
    except _NEEDS_CONCRETE_VALUES:
        return None
    except Exception as exc:  # noqa: BLE001 - the node's own failure
        return (f"{type(spec.node).__name__}'s step cannot run with it: its "
                f"hooks trace with the current value and raise with the new one "
                f"({type(exc).__name__}: {exc}), so every step after the write "
                "would fail")
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
        with quiet_warnings():
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
        with quiet_warnings():
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
            with quiet_warnings():
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


def _params_write_refusal(gm: GraphManager, owner: str, changes: dict[str, Any],
                          leaf_keys: Iterable[str], live_before: Optional[dict],
                          live_after: Optional[dict], *,
                          at_own_value: Iterable[str] = ()) -> Optional[tuple[list, str, bool]]:
    """``(keys, reason, reported)``: why the running graph cannot take
    *changes* -- ``{key: the value node.params would hold}`` -- into node
    *owner*'s params, or ``None``.

    The one decision ``PUT /graph/params`` and ``POST /checkpoint/load``
    share, so a value refused by one is refused by the other: a checkpoint
    used to restore parameter values ``PUT`` refuses -- outside a
    :class:`~maddening.core.params.ParamSpec`'s bounds (checked by the
    callers, with finiteness, before this), refused by the node's
    constructor at this graph's timestep -- answer 200, and leave a graph
    whose save did not reload and whose steps diverged.  *leaf_keys* are
    the keys that are leaves of the params pytree (a load writes nothing
    else); the rest are structural.  *live_before* / *live_after* are the
    node's injected params before and after the write.  ``reported`` says
    whether the write would be reported, and saved, as the value in force
    (:meth:`~maddening.core.graph_manager.GraphManager._unused_node_write_reason`).

    The keys in *at_own_value* hold the node's own value (the one it was
    built with) and are asked only in the combined checks -- the size of
    what the checks build, and the graph a save would reload, with every
    other key's live value: the node was built with the value, so the
    per-key checks have nothing to ask of it, but the graph's other live
    leaves (a fit, a load) may not take it.  A load that puts a calibrated
    leaf back to it and a ``PUT`` that writes it back both pass such a
    key here; each counts a key as changed when it differs from the live
    leaf, never from the node's own value.
    """
    leaf_keys = set(leaf_keys)
    spec = gm._nodes[owner]
    node = spec.node
    own = node.params
    # Before every check that builds the node with the new values (its
    # constructor, the state it makes): the size of what they would build,
    # for a class that can say so without building it.  They used to run
    # first, and ``n_levels=10000000`` on a wavelet node grew the server to
    # 57.7 GB before the kernel killed it.
    found = _allocation_write_reason(gm, owner, changes)
    if found is not None:
        return found[0], found[1], False
    skip = set(at_own_value)
    new = [k for k in changes if k not in skip]
    # A state of another layout -- or over the API's state cap -- is
    # refused whatever else is true of the value, and the checks below that
    # establish it on concrete values build the node and its state at the
    # new size first.  Told here without building anything, where the
    # node's initial_state() can be evaluated abstractly.  One key at a
    # time, like the checks it precedes: a key that changes the layout on
    # its own is refused by the first of them, so every node built below
    # has the running node's state layout.
    for key in new:
        reason = _state_write_reason_before_building(node, {key: changes[key]})
        if reason is not None:
            return [key], reason, False
    # A value the node consumed when it was constructed (a wall mask, an
    # assembled operator, a copy of an initial condition) is not rebuilt by
    # writing node.params: the write would be echoed, served by GET,
    # written out by to_dict() / save_state(), and ignored by every step.
    # Refused by the graph's own decision
    # (``GraphManager._unused_node_write_reason``), which is the one that
    # refuses the same leaf written into gm.params alone.
    for key in new:
        node_value = changes[key]
        # A structural value is checked against the node's own constructor:
        # the saved graph is rebuilt through it.  With the write's other
        # changes applied: a value valid only together with another one
        # (HeatNode ``stencil_order: 4`` with a smaller
        # ``thermal_diffusivity``) was refused when asked alone.  (Every
        # key, together and with a save's params, is checked below.)
        if key not in leaf_keys:
            others = {k: v for k, v in changes.items() if k != key and k in own}
            reason = gm._constructor_write_reason(owner, key, node_value, others=others,
                                                  live=live_before)
            if reason is not None:
                return [key, *others], reason, False
        shape_reason = gm._state_shape_write_reason(owner, key, node_value)
        if shape_reason is not None:
            return [key], shape_reason, False
        reason = gm._unused_node_write_reason(owner, key, node_value)
        if reason is not None:
            return [key], reason, True
        # A value the node's own step reads, but from which it also derived
        # the points an interface mapping was built from (a uniform
        # HeatNode's grid_x, from length): the mapping would keep the old
        # points' weights and a save would not load.  The graph's own
        # decision, the one that refuses the same value written into
        # gm.params alone at the next run.
        reason = gm._mapping_point_write_reason(owner, {key: node_value}, live=live_before)
        if reason is not None:
            return [key], reason, False
    # ... and of the whole write, every key at once (the keys at the node's
    # own value included), on the node a save would reload: a length that
    # the constructor refuses on its own and takes with the diffusivity
    # written beside it is asked of nothing above.
    if len(changes) > 1 or skip:
        reason = gm._mapping_point_write_reason(owner, changes, live=live_before)
        if reason is not None:
            return [k for k in changes if k in own], reason, False
    # The step must still run with the values: every check above asks
    # whether a value is *used*, and counted a trace that raised as "cannot
    # tell", so RigidBodyNode ``constraints: {"w": 0}`` answered 200 and
    # every later step was a 500.  Asked of the whole write at once.  A
    # leaf reaches the step as a traced input, so only a structural value
    # can make the trace raise.
    accepts = spec.accepts_params
    touched = [k for k in changes if k not in leaf_keys]
    if touched:
        reason = _step_trace_write_reason(
            gm, owner, changes,
            live_before if accepts else None, live_after if accepts else None)
        if reason is not None:
            return touched, reason, False
    # And the graph a save after the write would reload must load, and run
    # what the running graph runs: the constructor asked with every changed
    # key at once and every other key's live value (a live leaf used to
    # skip it), and a node whose constructor derives something from the
    # value -- a branch, an array -- asked whether the rebuilt node computes
    # what the running one does with it.
    saved = {k: v for k, v in changes.items() if k in own}
    if saved:
        reason = _saved_graph_write_reason(
            gm, owner, saved,
            live_before if accepts else None, live_after if accepts else None)
        if reason is not None:
            return list(saved), reason, False
    return None


def _leaf_value_refusal(key: str, value: Any, spec: Any) -> Optional[str]:
    """Why a params-pytree leaf cannot hold *value*: it is not finite, or
    *spec* (a :class:`~maddening.core.params.ParamSpec`, or ``None``)
    refuses it; ``None`` when it can.  ``PUT /graph/params`` refuses both
    before it writes; NaN passes every bounds comparison."""
    if not bool(np.all(np.isfinite(np.asarray(value)))):
        return f"{key}: value must be finite"
    if spec is not None:
        try:
            spec.check(value, name=key)
        except ValueError as exc:
            return str(exc)
    return None


def _new_node_bounds_refusal(node: Any, given: dict[str, Any]) -> Optional[str]:
    """Why ``POST /graph/nodes`` refuses the values *given* for a new
    *node*'s params-pytree leaves -- one the leaf's dtype cannot hold
    (:func:`_unrepresentable`), then :func:`_leaf_value_refusal` against the
    node's own :meth:`~maddening.core.node.SimulationNode.param_specs` -- or
    ``None``.  Asked of a node that takes injected params, as ``PUT
    /graph/params`` asks of a live leaf, and only of the keys the request
    gives: a class's own default is its own business."""
    if not _method_accepts_params(node, "update"):
        return None
    try:
        leaves = node.params_pytree()
        specs = node.param_specs()
    except Exception:  # noqa: BLE001 - the dry run below names what fails
        return None
    for key, value in given.items():
        if key in leaves:
            # A value the leaf's dtype flushes to zero or overflows, as PUT
            # refuses it ("does not fit its type"): 1e-50 was built as 0.0.
            problem = _unrepresentable(value, np.asarray(leaves[key]).dtype)
            if problem is not None:
                return f"{key}: {problem}, got {value!r}"
            reason = _leaf_value_refusal(key, leaves[key], specs.get(key))
            if reason is not None:
                return reason
    return None


def _installed_part(value: Any, *held: Any) -> Optional[np.ndarray]:
    """The elements of *value* a checkpoint load installs -- those equal to
    none of *held* (the leaf the graph holds now, the node's own value in
    its ``params``; ``None`` or another shape is skipped), NaN equal to
    NaN -- or ``None`` when there are none.  The whole value when every
    element is installed (a scalar's refusal then names the scalar), else
    those elements, flattened: a weight matrix whose zeros a fit left at
    zero is asked about the entries it moved, not about zeros a ``log``
    spec refuses and the graph was built with.

    The rule of :meth:`maddening.fmi.sidecar.FmuSidecar._check_restored_params`
    -- a value the parameter holds now, or held when the FMU was
    instantiated, is not a new value -- taken per element: a graph runs
    whatever its constructor was given, bounds being metadata to it, so a
    graph built outside its bounds reloads its own checkpoint, and goes
    back to its own value after a ``gm.params`` write moved the leaf.
    """
    v = np.asarray(value)
    new = np.ones(v.shape, dtype=bool)
    for other in held:
        if other is None:
            continue
        h = np.asarray(other)
        if h.shape != v.shape:
            continue
        try:
            same = np.asarray(v == h, dtype=bool)
            if v.dtype.kind in "fc" and h.dtype.kind in "fc":
                same = same | (np.isnan(v) & np.isnan(h))
        except (TypeError, ValueError):
            continue
        new &= ~same
    if not new.any():
        return None
    return v if new.all() else v[new]


def _install_param_leaves(gm: GraphManager, owner: str, leaves: dict[str, Any],
                          live: dict) -> None:
    """Write the params-pytree *leaves* of node *owner* where the graph
    holds them: into *live* (the node's leaf dict in ``gm.params`` -- or,
    before the first compile, the caller's copy of the node's own pytree)
    and, for a key the node's constructor took, into ``node.params``.

    The one install ``PUT /graph/params`` and ``POST /checkpoint/load``
    share.  The load used to write ``gm.params`` only, so a checkpoint
    saved before an initial condition was written stopped loading: the
    leaf then differed from the node's own value, only ``initial_state()``
    reads it, and the graph refuses such a leaf.  Written to both, the
    loaded value is the one ``POST /sim/reset`` builds the state from, as
    it is after a ``PUT``.
    """
    held = gm._nodes[owner].node.params
    for key, value in leaves.items():
        live[key] = value
        if key in held:
            # Store the constructor's Python type, never the raw JSON
            # value: a JSON ``40`` for a float leaf would turn it into an
            # ``int`` that ``params_pytree`` no longer exposes.
            held[key] = np.asarray(value).tolist()


def _leaves_a_load_changes(leaves: Any, live: Any) -> dict[str, Any]:
    """The leaves of a node's loaded params that differ from the ones the
    graph held (*live*): what a load is asked about, and what it installs
    (:func:`_install_param_leaves`)."""
    if not isinstance(leaves, dict) or not isinstance(live, dict):
        return {}
    return {k: v for k, v in leaves.items()
            if k in live and not _leaf_values_equal(v, live[k])}


def _shown_value(value: Any) -> str:
    """A parameter value for a refusal's text: the number, or the shape of
    an array of more than eight elements."""
    arr = np.asarray(value)
    if arr.size > 8:
        return f"an array of shape {arr.shape}"
    return np.array2string(arr, precision=7, separator=", ")


def _loaded_params_refusal(gm: GraphManager, loaded: dict) -> Optional[str]:
    """Why ``POST /checkpoint/load`` cannot take the parameter leaves
    *loaded* (the ``gm.params`` the load would leave) into the graph, or
    ``None``: what ``PUT /graph/params`` refuses of the values the load
    installs (:func:`_installed_part`) on each node -- a non-finite value,
    one outside its :class:`~maddening.core.params.ParamSpec`'s bounds --
    then :func:`_params_write_refusal` of the leaves it changes, and a
    non-finite or out-of-bounds interface-mapping weight it installs.
    Asked of the graph as it was before the load: the constructor and the
    save are asked with its live values.  ``GraphManager.load_state``
    asks none of these (``docs/user_guide/parameters.md``).
    """
    specs = gm.param_specs()
    live_tree = gm.params
    for owner, leaves in (loaded.get("nodes") or {}).items():
        spec = gm._nodes.get(owner)
        live = (live_tree.get("nodes") or {}).get(owner)
        if spec is None or not spec.accepts_params or not isinstance(live, dict) \
                or not isinstance(leaves, dict):
            continue
        staged = _leaves_a_load_changes(leaves, live)
        if not staged:
            continue
        own_specs = specs.get("nodes", {}).get(owner, {})
        ctor = spec.node.params_pytree()
        for key, value in staged.items():
            part = _installed_part(value, live[key], ctor.get(key))
            if part is None:
                continue
            reason = _leaf_value_refusal(key, part, own_specs.get(key))
            if reason is not None:
                return f"node {owner!r}, {reason}"
        changes = {k: np.asarray(v).tolist() for k, v in staged.items()}
        found = _params_write_refusal(
            gm, owner, changes, staged, dict(live), {**live, **staged},
            at_own_value=[k for k, v in staged.items()
                          if k in ctor and _leaf_values_equal(v, ctor[k])])
        if found is not None:
            keys, reason, _reported = found
            return f"node {owner!r}, {', '.join(keys)}: {reason}"
    for edge, leaves in (loaded.get("mappings") or {}).items():
        live = (live_tree.get("mappings") or {}).get(edge)
        if not isinstance(live, dict) or not isinstance(leaves, dict):
            continue
        edge_specs = specs.get("mappings", {}).get(edge, {})
        for key, value in leaves.items():
            if key not in live:
                continue
            part = _installed_part(value, live[key])
            if part is not None:
                reason = _leaf_value_refusal(key, part, edge_specs.get(key))
                if reason is not None:
                    return f"mapping {edge!r}, {reason}"
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
    try:
        parts = urlsplit(candidate)
    except ValueError:          # an unbalanced "[" in the authority
        return False
    if not parts.scheme or not parts.netloc:
        return False
    try:
        hostname, port = parts.hostname, parts.port
    except ValueError:
        return False
    if not hostname or parts.username is not None or parts.password is not None:
        return False
    # Compared as (host, port), both sides read the same way: ``urlsplit``
    # gives an IPv6 literal without its brackets, and the ``Host`` header
    # carries them, so a page served from ``http://[::1]:8000`` used to be
    # refused as cross-origin by its own server.
    return (hostname, port) == _host_and_port(host)


#: The 403 body.  Says what happened and what an embedder does about it.
_CROSS_ORIGIN_DETAIL = (
    "This request carried an Origin header that is not this server's own "
    "origin. A MADDENING server has no legitimate cross-origin caller: on a "
    "loopback bind the API is unauthenticated, so a page on any origin could "
    "otherwise reset the simulation or write a checkpoint from the "
    "developer's browser. If you are embedding the UI on another origin, "
    "pass allowed_origins to SimulationServer."
)


def _is_port(text: str) -> bool:
    """Whether *text* is a port as a ``Host`` header spells one: ASCII
    digits only.  ``str.isdigit`` is true of ``"\u00b2"`` (superscript two,
    a latin-1 byte a client can send) and ``int`` refuses it, so a Host
    ``localhost:\u00b2`` was a 500 on every route and on the WebSocket
    handshake."""
    return text.isascii() and text.isdigit()


def _host_and_port(host_header: Optional[str]) -> Optional[tuple[str, Optional[int]]]:
    """``(host, port)`` of a ``Host`` header as :func:`urllib.parse.urlsplit`
    reads an origin's authority -- lowercased, an IPv6 literal without its
    brackets, the port an ``int`` or ``None`` -- or ``None`` when it is not
    a valid ``host[:port]``.  Nothing else is normalised (a trailing dot is
    kept), so a comparison of the two is the authority comparison it always
    was, with the brackets read alike."""
    value = (host_header or "").strip().lower()
    if not value:
        return None
    if value.startswith("["):
        end = value.find("]")
        if end <= 1:
            return None
        name, rest = value[1:end], value[end + 1:]
        if not rest:
            return name, None
        if not (rest.startswith(":") and _is_port(rest[1:])):
            return None
        return name, int(rest[1:])
    if value.count(":") > 1:           # a bare IPv6 address is not a valid Host
        return None
    name, colon, port = value.partition(":")
    if not name or (colon and not _is_port(port)):
        return None
    return name, (int(port) if colon else None)


def _offered_subprotocols(scope: Mapping[str, Any]) -> list[str]:
    """The subprotocol names a WebSocket handshake offered.  ASGI specifies
    ``scope["subprotocols"]`` as a list of names; uvicorn 0.50.0 passes the
    ``Sec-WebSocket-Protocol`` header as one comma-joined entry, which left
    the browser's bearer carrier unread and ``maddening.v1`` unselected.
    Each entry is split on commas (a subprotocol name holds none)."""
    return [name.strip() for entry in (scope.get("subprotocols") or ())
            for name in str(entry).split(",") if name.strip()]


def _host_name(host_header: str) -> Optional[str]:
    """The host part of a ``Host`` header -- lowercased, without its port
    or a trailing dot, an IPv6 literal without its brackets -- or ``None``
    when the header is not a valid ``host[:port]``."""
    value = host_header.strip().lower()
    if not value:
        return None
    if value.startswith("["):
        end = value.find("]")
        rest = value[end + 1:] if end > 0 else None
        if rest is None or (rest and not (rest.startswith(":") and _is_port(rest[1:]))):
            return None
        return value[1:end]
    if value.count(":") > 1:           # a bare IPv6 address is not a valid Host
        return None
    name, colon, port = value.partition(":")
    if colon and not _is_port(port):
        return None
    return name.rstrip(".") or None


def _is_ip_address(peer: Optional[str]) -> bool:
    try:
        ipaddress.ip_address((peer or "").split("%", 1)[0])
    except ValueError:
        return False
    return True


def _rebinding_refusal(auth: APIAuth, host_header: Optional[str],
                       allowed_hosts: frozenset, *, authenticated: bool) -> Optional[str]:
    """Why a request that reached a loopback-bound API without a valid
    token is refused for its ``Host``, or ``None``.

    The ``Origin`` check compares the ``Origin`` with the ``Host``, and a
    DNS-rebinding page sends both naming its own domain: served from
    ``http://attacker.example:8000`` and then re-pointed at 127.0.0.1, it is
    same-origin in the browser's eyes, so it could reset the simulation,
    write checkpoints, read every reply and open ``/ws/state`` -- and
    ``POST /cloud/launch``.  It cannot choose the ``Host`` its browser
    sends, which names its own domain, so a loopback-bound API serves only
    the names this machine is reached by: ``localhost``, a ``127.0.0.0/8``
    or ``::1`` literal (any port), and the names in *allowed_hosts*.

    Asked of every request to a loopback bind that did not present a
    valid token (*authenticated*), whatever its peer: the token is what a
    rebound page does not hold.  It keys on the bind and the ``Host``
    header, never on the peer.  It used to be asked only of a peer that
    was an IP address, on the grounds that a browser reaches the API over
    TCP; but the peer is ``scope["client"]`` as the ASGI server reports
    it, which uvicorn's default ``proxy_headers`` rewrites from
    ``X-Forwarded-For`` for every request arriving over loopback, and a
    rebound page's ``fetch()`` may set that header.  ``X-Forwarded-For: x``
    made the peer the string ``"x"``, which skipped this check and the
    peer backstop both, and the page could step, read, add nodes and save
    checkpoints.  A non-loopback bind demands the token of every request
    and is not asked.  A request with no ``Host`` at all is not a
    browser's and is served.
    """
    if authenticated or auth.enforced or host_header is None:
        return None
    name = _host_name(host_header)
    if name is not None and (is_loopback(name) or name in allowed_hosts):
        return None
    return (
        f"This request named the host {host_header!r}, which is not a name of "
        "this loopback-bound server (localhost, 127.0.0.1, [::1]).  On a "
        "loopback bind the API is unauthenticated, and a request for another "
        "name is how a DNS-rebinding web page reaches it from the developer's "
        "browser.  To serve this server under another name (a reverse proxy, "
        "an /etc/hosts alias), pass allowed_hosts to SimulationServer."
    )


#: One label of a DNS host name: letters, digits, hyphens (not at either
#: end) and underscores (which ``/etc/hosts`` aliases and some proxies
#: use), 1 to 63 of them.
_HOST_LABEL = re.compile(r"(?!-)[a-z0-9_-]{1,63}(?<!-)")


def _is_host_name(name: str) -> bool:
    """Whether *name* (lowercased, without port, brackets or a trailing
    dot) is an IP literal or a DNS host name of at most 253 characters."""
    if _is_ip_address(name):
        return True
    return len(name) <= 253 and all(_HOST_LABEL.fullmatch(label)
                                    for label in name.split("."))


def _normalised_hosts(hosts: Iterable[str]) -> frozenset:
    """*hosts* as :func:`_host_name` reads a ``Host`` header (port and
    case ignored); an entry that is not a host name raises ``ValueError``.

    A host name is an IP literal (an IPv6 one in brackets) or DNS labels
    (:func:`_is_host_name`), with an optional port and nothing around it.
    Entries such as ``"a b"``, ``"*.example.com"``, ``"user@host"``,
    ``"evil.example/path"``, ``"host?x=1"`` or ``" host"`` used to be
    accepted: none can match a ``Host`` a browser sends, so the server was
    silently configured to serve a name it never would."""
    out = set()
    for host in hosts:
        text = str(host)
        name = _host_name(text) if text == text.strip() else None
        if name is None or not _is_host_name(name):
            raise ValueError(f"allowed_hosts entry {host!r} is not a host name "
                             "(an IP literal or DNS labels, optionally with a port)")
        out.add(name)
    return frozenset(out)


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

    def __init__(self, app, auth: APIAuth, allowed_origins: Iterable[str] = (),
                 allowed_hosts: frozenset = frozenset()) -> None:
        self.app = app
        self._auth = auth
        self._allowed_origins = frozenset(allowed_origins)
        self._allowed_hosts = frozenset(allowed_hosts)

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
        offered = _offered_subprotocols(scope)
        presented = bearer_from_headers(headers) or bearer_from_subprotocols(offered)
        authenticated = self._auth.verify(presented)
        # A DNS-rebinding page passes the origin check below (its Origin
        # and Host both name its own domain); its Host is what gives it
        # away.  Asked of every handshake without a valid token, whatever
        # its peer.  See _rebinding_refusal.
        if _rebinding_refusal(self._auth, headers.get("host"), self._allowed_hosts,
                              authenticated=authenticated) is not None:
            logger.warning(
                "Refused WebSocket %s for host %r: not a name of this "
                "loopback-bound server", scope.get("path", "?"), headers.get("host"),
            )
            await self._refuse(receive, send)
            return
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
        if not authenticated and self._auth._required_for_request(peer, headers):  # noqa: SLF001
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


# ------------------------------------------------------------------
# WebSocket streams: noticing a client that left
# ------------------------------------------------------------------

async def _receive_until_disconnect(websocket: WebSocket, on_message,
                                    disconnected: asyncio.Event) -> None:
    """Hand every JSON object a client sends to *on_message* until the
    client leaves, then set *disconnected*.

    The streams only learned that a client had gone when a send failed,
    and they send only when the simulation produced a new snapshot.  A
    client that left after the simulation stopped was never noticed, and
    neither was the close uvicorn sends on shutdown, so the handler slept
    on and SIGINT never completed.  Every way out of here -- the client's
    disconnect, a broken connection, cancellation -- sets *disconnected*,
    which ends the stream.  A message that is not a JSON object, or that
    *on_message* cannot apply, is ignored; it used to end this loop
    silently, and with it the stream's view of the client.
    """
    import json as _json  # noqa: PLC0415

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return
            raw = message.get("text")
            if raw is None:
                continue
            try:
                msg = _json.loads(raw)
                if isinstance(msg, dict):
                    on_message(msg)
            except Exception:  # noqa: BLE001 - a malformed message is ignored
                logger.debug("ignored WebSocket message %r", raw[:200], exc_info=True)
    except Exception:  # noqa: BLE001 - the connection broke: the client is gone
        return
    finally:
        disconnected.set()


async def _sleep_unless_disconnected(disconnected: asyncio.Event, seconds: float) -> None:
    """Sleep *seconds*, or until the client leaves, whichever is first."""
    try:
        await asyncio.wait_for(disconnected.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


#: Exception class names (anywhere in the MRO) that mean the peer closed
#: the WebSocket: Starlette's, the ``websockets`` library's, uvicorn's.
_DISCONNECT_EXCEPTION_NAMES = ("WebSocketDisconnect", "ConnectionClosed", "ClientDisconnected")


def _client_left(websocket: WebSocket, exc: BaseException,
                 disconnected: asyncio.Event) -> bool:
    """Whether a send that raised *exc* failed only because the client had
    left -- a normal end of the stream, not an error to log with a
    traceback.  uvicorn reports a send after the close as a
    ``RuntimeError`` ("Unexpected ASGI message 'websocket.send', after
    sending 'websocket.close'"), which every stream logged as an error on
    each ordinary disconnect."""
    from starlette.websockets import WebSocketState  # noqa: PLC0415

    if disconnected.is_set():
        return True
    if any(cls.__name__.startswith(_DISCONNECT_EXCEPTION_NAMES)
           for cls in type(exc).__mro__):
        return True
    if WebSocketState.DISCONNECTED in (getattr(websocket, "client_state", None),
                                       getattr(websocket, "application_state", None)):
        return True
    return isinstance(exc, RuntimeError) and "websocket.close" in str(exc)


#: The detail of a 500: what failed is in the server's log, with its
#: traceback, and the reply names no path, value or class of the server's.
#: The second is a write route's, whose transaction put the graph back.
_FAILED_UNEXPECTEDLY = "The request failed unexpectedly (the server's log says how)."
_UNEXPECTED_FAILURE_DETAIL = (
    f"{_FAILED_UNEXPECTEDLY} The graph was put back exactly as it was before "
    "the request.")


class _GraphTransaction:
    """What ``SimulationServer._graph_transaction`` records, holding the
    graph lock, and :meth:`rollback` puts back: the graph (everything
    ``GraphManager._transaction_snapshot`` reaches), the server's own record
    of the surrogates, whether the relay observes the graph, and the
    streams' clock and frame."""

    _RELAY_FIELDS = ("_snapshot", "_sim_time", "_step_count", "_timestep", "_elapsed")

    def __init__(self, server: "SimulationServer") -> None:
        self._server = server
        self._graph = server.gm._transaction_snapshot(
            also=(server._original_nodes, server._active_surrogates))
        self._relay_attached = server._relay_attached
        self._layout_generation = server._layout_generation
        relay = server.relay
        with relay._lock:
            self._relay_seq = relay._seq
            self._relay = {name: getattr(relay, name) for name in self._RELAY_FIELDS}
        self.rolled_back = False

    def rollback(self) -> None:
        """Put everything back as it was when the transaction began.  The
        streams are sent the restored frame, and the binary streams their
        schema again, only when the block had changed them."""
        server = self._server
        server.gm._transaction_restore(self._graph)
        server._relay_attached = self._relay_attached
        relay = server.relay
        with relay._lock:
            for name, value in self._relay.items():
                setattr(relay, name, value)
            if relay._seq != self._relay_seq:
                # A frame of the abandoned state may have been sent: a new
                # sequence number makes every stream send the restored one.
                relay._seq += 1
        if server._layout_generation != self._layout_generation:
            # The block added or removed a node, or compiled: the binary
            # streams were told the layout changed, and are told again.
            server._binary_encoder = None
            server._layout_generation += 1
        self.rolled_back = True


#: Held while :meth:`SimulationServer.create_app` builds an application, so
#: apps are built one at a time.  FastAPI builds each route's fields inside
#: ``warnings.catch_warnings()``, which saves the process's warnings
#: filters and puts them back on the way out and is not thread-safe: two
#: apps built at once in two threads put back each other's filters.  A
#: warning FastAPI silences was shown -- raised, where warnings are errors
#: -- and ``ignore::UserWarning`` could be left in the filters for good,
#: silencing every MADDENING warning in the process from then on
#: (MADD-ANO-191).
_CREATE_APP_LOCK = threading.RLock()


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
    allowed_hosts : iterable of str, optional
        Host names, besides ``localhost`` and the loopback addresses, that
        a loopback-bound server answers to (a reverse proxy's name, an
        ``/etc/hosts`` alias); ports are ignored.  On a loopback bind the
        API is unauthenticated, and a request whose ``Host`` names
        anything else is refused with 403: that is how a DNS-rebinding web
        page reaches it.  Asked of every request without a valid token,
        whatever its peer; not consulted on a non-loopback bind (the token
        decides there) nor for a request that presents the token.  A
        reverse proxy in front of a loopback bind needs its name here, and
        its forwarded requests (``X-Forwarded-For`` / ``Forwarded``) need
        the token.  Each entry must be a
        host name -- an IP literal (an IPv6 one in brackets) or DNS labels
        of letters, digits, hyphens and underscores, optionally with a
        port, nothing around it -- or the constructor raises
        ``ValueError``: an entry no ``Host`` header can match (``"a b"``,
        ``"*.example.com"``, ``"user@host"``) used to be taken.

    Attributes
    ----------
    auth : maddening.api.auth.APIAuth
        The token and the rule for when it is demanded.  Call
        ``auth.announce(port)`` before serving so a generated token
        reaches the log.

    Raises
    ------
    ValueError
        If ``MADDENING_API_TOKEN`` or *api_token* is set but blank, or an
        *allowed_hosts* entry is not a host name.

    Notes
    -----
    **Concurrency.**  FastAPI runs the routes on a thread pool, so
    requests arrive concurrently, and ``GraphManager.step`` reads the
    state, runs the compiled step (which releases the GIL) and stores the
    result: two steps in flight read the same state, and one result
    overwrote the other -- 200 concurrent ``POST /sim/step`` once took
    about 100 steps, every one answered 200.  One re-entrant lock now
    serialises every use of the graph: each route that reads or writes its
    state, params or structure takes it, the background runner takes it
    for each step, a surrogate job for its data sweep, and ``POST
    /sim/run`` for each slice of its run (writes are refused while a run
    is in progress, reads are served between slices).  It is never held
    while waiting for the runner's thread: a route that stops the runner
    does so first, without the lock, and then takes it; and the runner
    waits for it in short slices that watch its stop flag.
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
        allowed_hosts: Optional[Iterable[str]] = None,
    ) -> None:
        self.registry = dict(node_registry)
        self.auth = APIAuth(bind_host=bind_host, token=api_token)
        self.allowed_origins = frozenset(allowed_origins or ())
        self.allowed_hosts = _normalised_hosts(allowed_hosts or ())
        self.gm = graph_manager if graph_manager is not None else GraphManager()
        # /checkpoint/{save,load} only touch files under this directory:
        # a client must not choose arbitrary server paths.  That holds
        # whether or not the bearer token is enforced -- on a loopback
        # bind the caller is anyone with a shell on this box.
        self.checkpoint_root = Path(checkpoint_root or Path.cwd() / "checkpoints").resolve()
        self.relay = StateRelay()
        self.runner: Optional[RealtimeRunner] = None
        self._runner_started = False
        # ``PUT /sim/stride``'s value, kept for the runner a later
        # ``POST /sim/start`` creates (it used to be echoed and dropped).
        self._steps_per_frame = 1
        # A stop was asked for and timed out: the runner is kept until its
        # thread is seen to have exited (``POST /sim/stop`` again).
        self._runner_stop_pending = False
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
        # Every use of the graph is serialised by this lock (see Notes);
        # the runner's bookkeeping by the second, which is always taken
        # first when both are needed, and never while waiting for it.
        self._graph_lock = _GraphLock()
        self._runner_lock = threading.RLock()
        # Guards only the runner's steps-per-frame, the one value
        # ``PUT /sim/stride`` and a runner being created both touch: held
        # for a few assignments, never across a wait, so the stride answers
        # at once whatever the runner routes are waiting for.
        self._stride_lock = threading.Lock()
        # The runner routes' own threads (_RUNNER_ROUTE_WORKERS), and the
        # reset's (_RESET_ROUTE_WORKERS).
        self._runner_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_RUNNER_ROUTE_WORKERS, thread_name_prefix="maddening-runner-route")
        self._reset_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=_RESET_ROUTE_WORKERS, thread_name_prefix="maddening-reset-route")
        # A JAX trace this server started: its step and time budget
        # (MAX_JAX_TRACE_STEPS / _SECONDS), counted by _on_graph_event.
        self._trace_lock = threading.Lock()
        self._trace_steps = 0
        self._trace_started: Optional[float] = None
        self._trace_stopped_by: Optional[str] = None
        # The time budget's own clock: a timer started with the trace, so
        # the trace stops at its budget whether or not a step is recorded.
        self._trace_timer: Optional[threading.Timer] = None
        # A ``POST /sim/run`` is stepping the graph slice by slice.
        self._sync_run_active = False
        # Set when the server shuts down (the lifespan hook, a chained
        # SIGINT / SIGTERM, or request_shutdown()): an in-flight
        # ``POST /sim/run`` stops at its next slice, a training job at its
        # next epoch.
        self._shutdown = threading.Event()
        # Bumped whenever the graph's layout may have changed (a node
        # added or removed, a compile): the binary stream re-sends its
        # schema, and the cached encoder is dropped.
        self._layout_generation = 0
        self.gm.add_observer(self._on_graph_event)
        # The encoded ``/ws/state`` frame of the latest snapshot, shared by
        # every client that subscribes to the whole state.
        self._json_frame_cache: tuple[Optional[int], Optional[str]] = (None, None)
        self._stream_connections = 0
        self._surrogate_lock = threading.Lock()
        self._surrogate_threads: dict[str, threading.Thread] = {}
        _LIVE_SERVERS.add(self)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unauthorized_detail(self, peer: Optional[str],
                             headers: Optional[Any] = None) -> str:
        """The 401 body: why a token is needed and where to find it."""
        if self.auth.enforced:
            return (
                "This server is bound to a non-loopback address, so every "
                "route requires 'Authorization: Bearer <token>'. The token "
                "is $MADDENING_API_TOKEN, or was logged once at start-up."
            )
        if _carries_forwarding_header(headers):
            why = ("the request carried a forwarding header (X-Forwarded-For or "
                   "Forwarded), so it did not come straight from this machine")
        else:
            why = (f"the request arrived from {peer!r}, which is not a loopback "
                   "address (an in-process client such as Starlette's TestClient "
                   "reports a name: present server.auth.token, or construct it with "
                   "base_url='http://127.0.0.1' and client=('127.0.0.1', <port>))")
        return (
            f"This server was configured for a loopback bind "
            f"({self.auth.bind_host!r}), and a token is needed without a direct "
            f"connection from a loopback address: {why}. Present "
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
        offered = _offered_subprotocols(websocket.scope)
        selected = WS_SUBPROTOCOL if WS_SUBPROTOCOL in offered else None
        peer = websocket.client.host if websocket.client else None
        if not self.auth._required_for_request(peer, websocket.headers):  # noqa: SLF001
            return True, selected
        presented = (
            bearer_from_headers(websocket.headers)
            or bearer_from_subprotocols(offered)
        )
        if self.auth.verify(presented):
            return True, selected
        logger.warning(
            "Refused WebSocket %s from %s: %s bearer token",
            websocket.scope.get("path", "?"), peer or "?",
            "invalid" if presented else "missing",
        )
        await websocket.close(code=1008, reason="Invalid or missing bearer token")
        return False, None

    def _ensure_relay_attached(self) -> None:
        """Attach the relay to the graph once (it reads the graph's
        timestep): call it holding the graph lock."""
        if self._relay_attached:
            return
        try:
            self.relay.attach(self.gm)
            self._relay_attached = True
        except RuntimeError:
            pass

    def _ensure_relay_attached_locked(self) -> None:
        """:meth:`_ensure_relay_attached` under the graph lock, for a stream
        (run in a worker thread, never on the event loop)."""
        with self._graph_lock:
            self._ensure_relay_attached()

    def _ensure_runner(self) -> RealtimeRunner:
        # Asked on every call, not only when the runner is created: a
        # start that failed in its transaction leaves the runner it created
        # and puts the graph's observers back without the relay.
        self._ensure_relay_attached()
        if self.runner is None:
            with self._stride_lock:
                self.runner = RealtimeRunner(self.gm, self.relay,
                                             steps_per_frame=self._steps_per_frame,
                                             lock=self._graph_lock)
        return self.runner

    def _runner_stopping(self) -> bool:
        """Whether a runner thread is still alive after it was told to stop."""
        runner = self.runner
        return runner is not None and not self._runner_started and runner.is_alive

    def _runner_alive(self) -> bool:
        """Whether a runner thread is stepping the graph (running, paused,
        or still finishing after a stop)."""
        return self.runner is not None and self.runner.is_alive

    def _runner_running(self) -> bool:
        """Whether the runner is started and its thread is alive.  A
        runner whose thread died (a step raised) is not running, whatever
        ``_runner_started`` remembers: the routes used to report it
        started, paused and resumed long after it had gone."""
        return self._runner_started and self._runner_alive()

    def _reap_dead_runner(self) -> Optional[str]:
        """Forget a started runner whose thread has died; returns why it
        died (``None`` when it did not, or no reason was recorded).  Call
        it holding the runner lock."""
        runner = self.runner
        if runner is None or runner.is_alive or not self._runner_started:
            return None
        self.runner = None
        self._runner_started = False
        self._runner_stop_pending = False
        return runner.error or "its thread exited"

    def _refuse_while_runner_alive(self, action: str) -> None:
        """Refuse a write of the graph while something else steps it: a
        409 while the runner runs or a ``POST /sim/run`` is in progress, a
        503 while the runner is still stopping.  The runner stores each
        step's state over whatever is there, so a state written beside it
        used to be answered 200 and then lost -- a reset after a stop that
        had not finished came back with positions in the hundreds of
        thousands."""
        if self._runner_stopping():
            raise HTTPException(
                status_code=503,
                detail=(f"Cannot {action}: the runner was told to stop and its "
                        "thread is still finishing a step. Retry shortly."),
                headers={"Retry-After": "1"},
            )
        if self._runner_alive():
            raise HTTPException(
                status_code=409,
                detail=(f"Cannot {action} while the runner is started; "
                        "POST /sim/stop first."),
            )
        if self._sync_run_active:
            raise HTTPException(
                status_code=409,
                detail=(f"Cannot {action} while a POST /sim/run is in progress; "
                        "wait for it to return."),
            )

    @contextlib.contextmanager
    def _graph_access(self, action: str, *, write: bool = False,
                      deadline: Optional[float] = None):
        """Hold the graph lock for *action*: a 503 after
        :data:`_GRAPH_LOCK_TIMEOUT` seconds without it, or at *deadline*
        (``time.monotonic()``) for a request that has already waited for
        something else.

        With *write*, refused (:meth:`_refuse_while_runner_alive`) while
        the runner or a ``POST /sim/run`` steps the graph -- asked before
        waiting for the lock, whose holder may be the runner's thread in a
        step, and again once it is held, so that nothing can start
        stepping between the question and the write.
        """
        if write:
            self._refuse_while_runner_alive(action)
        timeout = (_GRAPH_LOCK_TIMEOUT if deadline is None
                   else max(0.0, deadline - time.monotonic()))
        if not self._graph_lock.acquire(timeout=timeout):
            raise HTTPException(
                status_code=503,
                detail=(f"Cannot {action}: the graph has been in use for "
                        f"{_GRAPH_LOCK_TIMEOUT:g} s (a long step or compile, a "
                        "large checkpoint, a surrogate data sweep). Nothing was "
                        "changed; retry shortly."),
                headers={"Retry-After": "1"},
            )
        try:
            if write:
                self._refuse_while_runner_alive(action)
            yield
        finally:
            self._graph_lock.release()

    @contextlib.contextmanager
    def _graph_transaction(self, action: str, *, write: bool = False,
                           deadline: Optional[float] = None):
        """:meth:`_graph_access` for a route that can change the graph, as
        one transaction: everything the graph holds is recorded once the
        lock is held, and put back if the block does not finish -- it raises
        a refusal (an ``HTTPException``), or anything else.  So a request
        that is refused, or that fails unexpectedly, leaves the graph
        exactly as it was, whatever the route had done by then; the
        unexpected failure is logged with its traceback and answered 500
        with :data:`_UNEXPECTED_FAILURE_DETAIL`, which names nothing of the
        server's.

        Recorded and put back (``GraphManager._transaction_snapshot``):
        every attribute of the graph and every container reachable from one
        -- state, params, nodes and their own ``params``, edges, coupling
        groups, external inputs, compile bookkeeping -- with this server's
        own record of the surrogates (``_original_nodes``,
        ``_active_surrogates``), whether the relay observes the graph, and
        the streams' clock and frame.  Arrays are shared, not copied.

        Not covered, and left to the route: a file under the checkpoint
        root, the runner's thread and its started / stopping flags, a
        surrogate job, a JAX trace, frames a stream has already sent, and
        observers of the graph already notified.

        Yields the transaction, whose ``rollback()`` a route calls before
        it *returns* a refusal instead of raising one.
        """
        with self._graph_access(action, write=write, deadline=deadline):
            transaction = _GraphTransaction(self)
            try:
                yield transaction
            except BaseException as exc:
                transaction.rollback()
                if isinstance(exc, HTTPException) or not isinstance(exc, Exception):
                    raise
                logger.exception("Could not %s: the request failed unexpectedly "
                                 "and the graph was put back as it was", action)
                raise HTTPException(
                    status_code=500, detail=_UNEXPECTED_FAILURE_DETAIL) from None

    @contextlib.contextmanager
    def _runner_control(self, action: str, deadline: float):
        """Hold the runner lock for *action*, waiting until *deadline* at
        most: a 503 then.  The routes that change the runner hold it; it
        used to be taken with no timeout while some of them waited for the
        graph lock inside it, so the k-th request behind a long holder of
        the graph lock answered after about k graph-lock timeouts."""
        if not self._runner_lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise HTTPException(
                status_code=503,
                detail=(f"Cannot {action}: another request has been starting, "
                        "stopping or resetting the runner for "
                        f"{_GRAPH_LOCK_TIMEOUT:g} s. Nothing was changed; retry "
                        "shortly."),
                headers={"Retry-After": "1"},
            )
        try:
            yield
        finally:
            self._runner_lock.release()

    def _after_stopping_the_runner(self, exc: HTTPException, action: str,
                                   was_running: bool) -> _Reply:
        """The reply of a route that stopped the runner and then could not
        do *action* (*exc*: the graph lock's 503, a 409): the runner stays
        stopped, which the reply says, with ``was_running``.  It used to
        answer "Nothing was changed" for a runner it had just stopped."""
        stopped = (" The runner was stopped first and stays stopped; POST "
                   "/sim/start starts it again." if was_running else "")
        detail = str(exc.detail).replace(" Nothing was changed;", " Nothing was "
                                         f"{action};") + stopped
        return _Reply(status_code=exc.status_code, headers=exc.headers,
                      content={"detail": detail, "was_running": was_running})

    def _publish_state(self, *, restart: bool = False) -> None:
        """Publish the graph's state to the streams, holding the graph
        lock: at step 0 and time 0 with *restart* (a reset), at the clock
        the relay has otherwise.  Routes that change the state without a
        step -- a reset, a state write, a node added or removed, a
        surrogate (de)activated -- used to publish nothing, so the streams
        served the state from before them until the next step."""
        self._ensure_relay_attached()
        if restart:
            self.relay.restore(self._user_state(), step_count=0, elapsed=0.0)
        else:
            self.relay.restore(self._user_state(), step_count=self.relay.step_count,
                               elapsed=self.relay.elapsed)

    @contextlib.contextmanager
    def _relay_detached(self):
        """Run a block without the relay observing the graph's steps (the
        profiler's): the streams neither publish nor count them."""
        callback = self.relay._on_event
        detached = callback in self.gm._observers
        if detached:
            self.gm._observers.remove(callback)
        try:
            yield
        finally:
            if detached:
                self.gm._observers.append(callback)

    def _state_json(self) -> dict:
        """The whole state, ``_meta`` included, as a reply body
        (:func:`_json_reply`: a non-finite float is its quoted token)."""
        return _json_reply(self.gm._state)

    def _node_state_json(self, name: str) -> dict:
        return _json_reply(self.gm.get_node_state(name))

    def _stop_runner(self) -> bool:
        """Stop the runner and wait for its thread; ``True`` once no runner
        thread is alive.  Safe to call multiple times.

        ``False`` when the thread is still alive after
        :data:`_RUNNER_STOP_TIMEOUT` seconds: the runner is kept, marked as
        stopping, and the caller must not report it stopped or write the
        state it steps.  This used to give up after two seconds and drop
        the handle while the thread kept stepping -- ``POST /sim/stop``
        answered "stopped", and a reset after it was overwritten.

        Never called holding the graph lock: the thread needs it for the
        step it may be waiting to take, so the wait would last the whole
        timeout and end in a 503 every time.  That is a programming error,
        and raises.
        """
        if self._graph_lock.held():
            raise RuntimeError(
                "SimulationServer._stop_runner called holding the graph lock: "
                "the runner's thread needs it to finish its step")
        runner = self.runner
        if runner is None:
            return True
        if self._runner_started or runner.is_alive:
            self._runner_started = False
            if not runner.stop(timeout=_RUNNER_STOP_TIMEOUT):
                self._runner_stop_pending = True
                return False
        self.runner = None
        self._runner_stop_pending = False
        return True

    def _stop_runner_or_refuse(self, action: str) -> None:
        """:meth:`_stop_runner`, as a 503 when the thread will not stop in
        time; nothing else is done.  The runner has been told to stop by
        then and stays stopped, which the 503 says: it used to say
        "Nothing was changed" of a runner it had just stopped."""
        if not self._stop_runner():
            raise HTTPException(
                status_code=503,
                detail=(f"Cannot {action}: the runner was told to stop, and its "
                        "thread is still finishing a step after "
                        f"{_RUNNER_STOP_TIMEOUT:g} s.  The runner stays stopped "
                        "(POST /sim/start starts it again once that step is done) "
                        "and nothing else was changed; retry shortly."),
                headers={"Retry-After": "1"},
            )

    @staticmethod
    def _stop_refused(exc: HTTPException, was_running: bool) -> _Reply:
        """The reply of a route whose :meth:`_stop_runner_or_refuse` refused:
        its 503, with ``was_running`` (whether a runner was running when
        the request told it to stop)."""
        return _Reply(status_code=exc.status_code, headers=exc.headers,
                      content={"detail": exc.detail, "was_running": was_running})

    def _reset_state(self) -> None:
        """Reset all nodes to their initial state (normalised, no retrace),
        and publish it to the streams at step 0."""
        self.gm.reset_state()
        # Reset relay counters, its clock included: the relay sums each
        # step's advance, so leaving that sum would restart the frames'
        # sim_time from where the run before the reset stopped.
        self.relay.reset()
        # Invalidate binary encoder (state shape may have changed)
        self._binary_encoder = None
        # And show the reset state: the streams used to send nothing until
        # the next step, so a client kept the state from before the reset.
        self._publish_state(restart=True)

    def _user_state(self) -> dict:
        """The state without ``_meta``, shallow-copied: call it holding the
        graph lock."""
        return {k: dict(v) for k, v in self.gm._state.items() if k != "_meta"}

    def _on_graph_event(self, event: str, data: Any) -> None:
        """Graph observer: drop the cached binary encoder, and tell the
        binary streams to re-send their schema, when a node is added or
        removed or the graph is compiled and the state's layout is no
        longer the one they encode.  The encoder used to be dropped only by
        a reset, so a node replaced over REST kept a schema the graph no
        longer had."""
        if event == EVENT_STEP:
            if self._trace_started is not None:
                self._count_traced_step()
            return
        if event not in (EVENT_NODE_ADDED, EVENT_NODE_REMOVED, EVENT_COMPILED):
            return
        self._binary_encoder = None
        self._layout_generation += 1

    def _count_traced_step(self) -> None:
        """One more step recorded by the JAX trace this server started:
        stop the trace past :data:`MAX_JAX_TRACE_STEPS` steps or
        :data:`MAX_JAX_TRACE_SECONDS` seconds."""
        with self._trace_lock:
            if self._trace_started is None:
                return
            self._trace_steps += 1
            if self._trace_steps >= MAX_JAX_TRACE_STEPS:
                self._stop_trace(f"its step budget, {MAX_JAX_TRACE_STEPS} steps")
            else:
                self._stop_trace_if_out_of_time()

    def _stop_trace_if_out_of_time(self) -> None:
        """Stop the trace if it has run :data:`MAX_JAX_TRACE_SECONDS`;
        call it holding the trace lock.  The elapsed time is read here, so
        a timer left over from an earlier trace -- which a stop cancels, but
        which may already be firing -- stops a later one only once that one
        is out of time too.

        The time used to be read only when a step was recorded, so a trace
        started on an idle simulation ran past its budget until the next
        step, or for ever.  It is now read by a timer set when the trace
        starts, and again by every status, stop and step, whichever comes
        first."""
        if self._trace_started is None:
            return
        budget = MAX_JAX_TRACE_SECONDS
        if time.monotonic() - self._trace_started >= budget:
            self._stop_trace(f"its time budget, {budget:g} s")

    def _trace_time_is_up(self) -> None:
        """The time budget's timer: stop the trace if it is out of time."""
        with self._trace_lock:
            self._stop_trace_if_out_of_time()

    def _stop_trace(self, by: str) -> Optional[str]:
        """Stop the JAX trace this server started, holding the trace lock;
        returns its directory.  *by* says why, for the status route."""
        from maddening.core.simulation.profiler import (  # noqa: PLC0415
            jax_trace_active,
            stop_jax_trace,
        )
        log_dir = None
        if jax_trace_active():
            try:
                log_dir = stop_jax_trace()
            except RuntimeError:          # stopped by someone else meanwhile
                log_dir = None
        if log_dir:
            self._last_jax_trace_dir = log_dir
        self._trace_started = None
        self._trace_stopped_by = by
        timer, self._trace_timer = self._trace_timer, None
        if timer is not None and timer is not threading.current_thread():
            timer.cancel()
        if by != "a request":
            logger.warning("The JAX trace stopped itself after %d steps (%s); its "
                           "trace is in %s", self._trace_steps, by, log_dir)
        return log_dir

    def _get_binary_encoder(self):
        """Lazily build a BinaryStateEncoder from the current state: call it
        holding the graph lock."""
        if self._binary_encoder is None:
            from maddening.api.binary_encoder import BinaryStateEncoder
            self._binary_encoder = BinaryStateEncoder(self._user_state())
        return self._binary_encoder

    def _binary_encoder_locked(self):
        """:meth:`_get_binary_encoder`, the layout generation it was built
        at and the state layout it encodes, under the graph lock (a stream
        runs it in a worker thread)."""
        with self._graph_lock:
            return (self._get_binary_encoder(), self._layout_generation,
                    _state_layout(self._user_state()))

    def _user_state_locked(self) -> dict:
        with self._graph_lock:
            return self._user_state()

    def _admit_stream(self) -> bool:
        """Count a new stream connection; ``False`` (not counted) past
        :data:`MAX_STREAM_CONNECTIONS`."""
        with self._surrogate_lock:
            if self._stream_connections >= MAX_STREAM_CONNECTIONS:
                return False
            self._stream_connections += 1
            return True

    def _release_stream(self) -> None:
        with self._surrogate_lock:
            self._stream_connections -= 1

    async def _refuse_stream(self, websocket: WebSocket, path: str,
                             subprotocol: Optional[str]) -> None:
        """Close a stream past :data:`MAX_STREAM_CONNECTIONS` with 1013.
        Accepted first: a handshake closed before it is accepted reaches a
        real client as HTTP 403 -- the answer to an Origin or token
        refusal -- with no close code at all (only Starlette's test client
        reports the code)."""
        logger.warning("Refused WebSocket %s: %d streams are open, the most this "
                       "server serves (MAX_STREAM_CONNECTIONS)", path,
                       MAX_STREAM_CONNECTIONS)
        await websocket.accept(subprotocol=subprotocol)
        await websocket.close(
            code=1013,
            reason=f"{MAX_STREAM_CONNECTIONS} streams are open already; try again later.",
        )

    async def _shared_state_frame(self, seq: int, sim_time: float, snapshot: dict,
                                  fields: Optional[dict]) -> str:
        """The ``/ws/state`` frame for one relay snapshot, encoded in a
        worker thread; a client of the whole state reuses the frame another
        client already encoded for the same snapshot."""
        if fields is None:
            cached_seq, cached = self._json_frame_cache
            if cached_seq == seq and cached is not None:
                return cached
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _encode_state_frame, sim_time,
                                          snapshot, fields)
        if fields is None:
            self._json_frame_cache = (seq, text)
        return text

    @stability(StabilityLevel.EVOLVING)
    def request_shutdown(self) -> None:
        """Tell the work that runs for a while to stop soon.

        An in-flight ``POST /sim/run`` stops at its next slice (a fraction
        of a second) and answers 503 with how many of its steps it ran;
        a surrogate training job stops at its next epoch.  Called by the
        app's lifespan shutdown and, when the app is served from the main
        thread, by a SIGINT / SIGTERM, ahead of the server's own handler.
        uvicorn waits for in-flight requests to finish before it runs the
        lifespan shutdown, which is why the signal is chained: a server
        embedded in another thread, or stopped by setting
        ``uvicorn.Server.should_exit``, calls this itself first.
        """
        self._shutdown.set()

    def _chain_shutdown_signals(self):
        """Have SIGINT and SIGTERM call :meth:`request_shutdown` before the
        handler already installed for them (uvicorn's), which they still
        call.  Only from the main thread, and only over a Python handler:
        a default or ignored signal is left alone.  Returns the function
        that puts the previous handlers back."""
        installed: list = []
        if threading.current_thread() is not threading.main_thread():
            return lambda: None
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)
            except (ValueError, OSError):
                continue
            if not callable(previous):
                continue

            def handler(signum, frame, _previous=previous):
                self.request_shutdown()
                _previous(signum, frame)

            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                continue
            installed.append((sig, handler, previous))

        def restore() -> None:
            for sig, handler, previous in installed:
                try:
                    if signal.getsignal(sig) is handler:
                        signal.signal(sig, previous)
                except (ValueError, OSError):
                    pass

        return restore

    def _cancel_training_jobs(self, timeout: float = 30.0) -> None:
        """Tell every running training job to stop and wait up to *timeout*
        seconds in all for their threads."""
        self._shutdown.set()
        deadline = time.monotonic() + timeout
        for thread in list(self._surrogate_threads.values()):
            thread.join(max(0.0, deadline - time.monotonic()))

    def _stop_for_shutdown(self) -> None:
        """The lifespan shutdown's work, in a worker thread: stop the
        runner and the training jobs."""
        with self._runner_lock:
            try:
                self._stop_runner()
            except Exception:  # noqa: BLE001 - shutting down regardless
                logger.exception("stopping the runner at shutdown")
        self._cancel_training_jobs(timeout=_RUNNER_STOP_TIMEOUT)

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

        Notes
        -----
        Safe to call from several threads at once: the apps are built one
        at a time (see :data:`_CREATE_APP_LOCK`).
        """
        with _CREATE_APP_LOCK:
            return self._build_app()

    def _build_app(self) -> FastAPI:
        """:meth:`create_app`'s application, built under :data:`_CREATE_APP_LOCK`."""
        # Swagger UI is a browser page that fetches /openapi.json with no
        # Authorization header, so it cannot work behind a bearer token.
        # A half-working docs page that 401s on its own schema is worse
        # than none: when the token is enforced these are switched off,
        # and the way to read them is an SSH tunnel to a loopback bind.
        interactive_docs = not self.auth.enforced

        @contextlib.asynccontextmanager
        async def lifespan(_app):
            # Startup: a SIGINT / SIGTERM tells the long-running work to stop
            # before uvicorn's handler sees it.  uvicorn waits for every
            # in-flight request before it runs the shutdown half below, so
            # a stop event set only there was never seen by the
            # ``POST /sim/run`` that SIGINT was waiting for.
            self._shutdown.clear()
            restore_signals = self._chain_shutdown_signals()
            try:
                yield
            finally:
                restore_signals()
                self.request_shutdown()
                await asyncio.get_running_loop().run_in_executor(
                    None, self._stop_for_shutdown)

        app = FastAPI(
            title="MADDENING Simulation Server",
            description="HTTP/WebSocket API for the MADDENING simulation graph.",
            # The package version, not a separately maintained API
            # version: this was pinned at "0.3.0" and went stale.
            version=_maddening_version,
            # Every route's reply is written by the one encoder that cannot
            # fail on it (_Reply), as every refusal and handler below is.
            default_response_class=_Reply,
            docs_url="/docs" if interactive_docs else None,
            redoc_url="/redoc" if interactive_docs else None,
            openapi_url="/openapi.json" if interactive_docs else None,
            lifespan=lifespan,
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
        #
        # The body limit is added first, so it is the innermost: the
        # authentication refuses an unauthenticated caller (401) before
        # anything is read, and the limit's 413 is raised straight into
        # FastAPI's body reader.  Outside a BaseHTTPMiddleware it would be
        # raised through that middleware's task group, which wraps it in an
        # ExceptionGroup that FastAPI answers as "There was an error
        # parsing the body" (400).  No middleware or route reads a body
        # before it either way.
        # A 422 echoes the value it refused, and an HTTPException's detail
        # often quotes one.  FastAPI's own handlers hand both to an encoder
        # that refuses NaN, the infinities and a surrogate: a request whose
        # refused field held one (``"timestep": NaN``) was a 500.  The
        # handler that replaced the first encoded with _json_reply, which
        # refuses the *text* ``NaN``: ``POST /sim/run?n_steps=NaN`` was a
        # 500.  Both are written by _Reply, which refuses nothing.
        @app.exception_handler(RequestValidationError)
        async def _request_validation_error(_request, exc):
            return _Reply(status_code=422,
                          content={"detail": jsonable_encoder(exc.errors())})

        @app.exception_handler(StarletteHTTPException)
        async def _http_exception(_request, exc):
            # FastAPI's handler, with this app's encoder: every
            # ``HTTPException(detail=...)`` a route raises, and Starlette's
            # own 404 and 405.
            headers = getattr(exc, "headers", None)
            if exc.status_code < 200 or exc.status_code in (204, 205, 304):
                return Response(status_code=exc.status_code, headers=headers)
            return _Reply(status_code=exc.status_code, headers=headers,
                          content={"detail": exc.detail})

        @app.exception_handler(Exception)
        async def _unexpected_failure(request, exc):
            # Anything a route did not expect, outside a write route's
            # transaction (which answers its own 500 after putting the
            # graph back): the same generic detail, where Starlette's
            # default is a plain-text body.  Starlette logs the traceback
            # and re-raises for the server after this reply is sent.
            logger.error("%s %s failed unexpectedly: %s", request.method,
                         request.scope.get("path", ""), type(exc).__name__)
            return _Reply(status_code=500, content={"detail": _FAILED_UNEXPECTEDLY})

        app.add_middleware(_RequestBodyLimitMiddleware)
        app.add_middleware(
            _WebSocketAuthMiddleware,
            auth=self.auth,
            allowed_origins=self.allowed_origins,
            allowed_hosts=self.allowed_hosts,
        )

        @app.middleware("http")
        async def _require_bearer_token(request, call_next):
            peer = request.client.host if request.client else None
            # The scope's path, not the request's URL: the URL is built from
            # the Host header, and a malformed one (``[::1]x``) raised
            # inside this middleware -- a 500 where the refusal is a 403.
            path = request.scope.get("path", "")
            authenticated = self.auth.verify(bearer_from_headers(request.headers))
            if (not authenticated
                    and path not in UNAUTHENTICATED_PATHS
                    and self.auth._required_for_request(peer, request.headers)):  # noqa: SLF001
                logger.warning(
                    "Refused %s %s from %s: %s bearer token",
                    request.method, path, peer or "?",
                    "invalid" if request.headers.get("authorization")
                    else "missing",
                )
                return _Reply(
                    status_code=401,
                    content={"detail": self._unauthorized_detail(peer, request.headers)},
                    headers={"WWW-Authenticate": "Bearer"},
                )
            # Every method, GET included: a rebound page is same-origin to
            # its browser, so it can read the replies it gets.  Asked of
            # every request without a valid token, whatever its peer (which
            # a proxy header can rewrite): see _rebinding_refusal.
            rebinding = _rebinding_refusal(self.auth, request.headers.get("host"),
                                           self.allowed_hosts, authenticated=authenticated)
            if rebinding is not None:
                logger.warning(
                    "Refused %s %s for host %r: not a name of this "
                    "loopback-bound server", request.method, path,
                    request.headers.get("host"),
                )
                return _Reply(status_code=403, content={"detail": rebinding})
            if (request.method in _STATE_CHANGING_METHODS
                    and not origin_is_same_site(
                        request.headers.get("origin"),
                        request.headers.get("host"),
                        self.allowed_origins,
                    )):
                logger.warning(
                    "Refused %s %s from origin %r: not this server's origin",
                    request.method, path,
                    request.headers.get("origin"),
                )
                return _Reply(
                    status_code=403,
                    content={"detail": _CROSS_ORIGIN_DETAIL},
                )
            return await call_next(request)

        @app.get("/healthz", tags=["meta"], response_model=None)
        async def healthz() -> dict[str, str]:
            """Liveness probe.  Served without a token, on purpose.

            A container health check has no credential, and this answer
            says nothing about the graph -- only that the process is up
            and which version it is.  Answered on the event loop, not the
            worker threads, so it answers while every worker is waiting
            for the graph.
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
            with self._graph_access("read the graph"):
                data = self.gm.to_dict(strict_mappings=False)
                data["active_surrogates"] = list(self._active_surrogates)
            return data

        @app.post("/graph/nodes", tags=["graph"], status_code=201, response_model=None)
        def add_node(req: AddNodeRequest) -> dict[str, Any]:
            """Add a node.  Refused (409) while the runner runs or a
            ``POST /sim/run`` is in progress.  Its size is bounded per node
            (:data:`MAX_NODE_STATE_ELEMENTS`, :data:`MAX_NODE_BUILD_BYTES`)
            and over the whole graph (:data:`MAX_GRAPH_STATE_ELEMENTS`),
            before it is built where its class can say what it would
            build; nodes are built one at a time (under the graph lock), so
            concurrent requests do not each hold a node's worth of memory
            at once."""
            # The name before anything is built or echoed: add_node's own
            # rule, asked here so a name it refuses costs no constructor,
            # and no reply quotes a name a reply cannot carry.
            refusal = _node_name_refusal(req.name)
            if refusal is not None:
                raise HTTPException(status_code=400, detail=refusal)
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
            # Text the config could not carry, for a parameter that takes
            # text: the node was added, and its own reply was the 500.
            for key, value in req.params.items():
                refusal = _text_value_refusal(value)
                if refusal is not None:
                    raise HTTPException(status_code=400,
                                        detail=f"params.{key}: {refusal}")
            # A finite number the node's float dtype cannot hold (1e39 in
            # float32) becomes an infinity once the node traces it.
            bad = _overflowing_float(req.params)
            if bad is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(f"params.{bad}: value does not fit its type "
                            f"{np.dtype(jax.dtypes.canonicalize_dtype(jnp.float64))}"),
                )
            node_cls = self.registry[req.type]
            # An integer is bounded where its parameter is an integer (an
            # array dimension): told from the constructor's default, so an
            # integral JSON number for a float parameter -- a browser's
            # 2e7 -- is a float, as PUT /graph/params stores it.
            oversized = _oversized_new_node_param(node_cls, req.params)
            if oversized is not None:
                raise HTTPException(status_code=422, detail=oversized)
            # A boolean or text for a numeric parameter, refused as PUT
            # refuses it (the constructor took both without a word).
            wrong_type = _new_node_type_refusal(node_cls, req.params)
            if wrong_type is not None:
                raise HTTPException(status_code=400,
                                    detail=f"node '{req.name}': {wrong_type}")
            # Before the constructor, for a class that can say what it would
            # build: the size checks below run on the built node and state,
            # so they used to refuse a D3Q19 lattice of 140^3 cells after
            # 1.1 GB had been allocated -- and a wavelet basis of
            # n_levels=10000000 was never refused at all, the constructor
            # running into the OOM killer first.
            estimate = estimate_allocation(node_cls, req.params)
            if estimate is not None:
                refusal = _allocation_refusal(estimate)
                if refusal is not None:
                    raise HTTPException(
                        status_code=400,
                        detail=(f"node '{req.name}' {refusal}; this is told from "
                                "its params, before anything of that size is built"),
                    )
            with self._graph_transaction("add a node", write=True):
                # The graph-wide budget: each node was held to the per-node
                # cap, and nothing bounded how many of them a caller added.
                held = sum(_state_elements(fields) for name, fields in self.gm._state.items()
                           if name != "_meta")
                if estimate is not None:
                    refusal = _graph_budget_refusal(req.name, held, estimate.state_elements)
                    if refusal is not None:
                        raise HTTPException(status_code=400, detail=refusal)
                try:
                    node = node_cls(name=req.name, timestep=req.timestep, **req.params)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                # The ParamSpec bounds PUT /graph/params holds a write to,
                # on the values this request gives: a damping of -5 below
                # its bound of 0 was added, which PUT then refused to
                # write, and a checkpoint of the graph carried it.
                refusal = _new_node_bounds_refusal(node, req.params)
                if refusal is not None:
                    raise HTTPException(status_code=400,
                                        detail=f"node '{req.name}': {refusal}")
                # AddNodeRequest bounds each integer the caller sends, which
                # is what keeps a single dimension from naming hundreds of
                # MB.  Dimensions that multiply survive that bound, so the
                # state the node actually built is measured too -- before
                # the node joins the graph, so an oversized one is transient
                # rather than resident for the life of the process.
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
                refusal = _graph_budget_refusal(req.name, held, n_elements)
                if refusal is not None:
                    del initial_state, node
                    raise HTTPException(status_code=400, detail=refusal)
                # Nodes do not validate their constants; a bad one only
                # fails inside the trace and would wedge every later
                # /sim/step.  Trace one update on the node's own initial
                # state (abstractly, no compute) before it enters the graph,
                # the way the graph calls it, and hold what it returns to
                # that state's layout.
                refusal = _dry_run_refusal(node_cls, req.name, req.timestep,
                                           req.params, node, initial_state)
                if refusal is not None:
                    del initial_state, node
                    raise HTTPException(status_code=400,
                                        detail=f"node '{req.name}' {refusal}")
                if req.name in self.gm._nodes:
                    raise HTTPException(status_code=409, detail=f"Node '{req.name}' already exists in the graph.")
                try:
                    self.gm.add_node(node)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                self._publish_state()
                # Through the reply encoder every other route uses: a
                # non-finite value in the node's dict is written as its
                # quoted token, never a 500 after the node was added.
                return _json_reply({"status": "ok", "node": node.to_dict()})

        @app.delete("/graph/nodes/{name}", tags=["graph"], response_model=None)
        def remove_node(name: str) -> dict[str, str]:
            """Remove a node, with its edges and external inputs.  A
            coupling group loses it as a member and keeps its options; a
            group left with fewer than two members is removed (no route
            adds one back); the reply's ``coupling_groups`` says which, when
            one changed.  A surrogate activated under the name is
            forgotten with it.  400 when a mapping on an edge between two
            other nodes was built from the node's points: remove that edge
            first.  Refused (409) while the runner runs or a
            ``POST /sim/run`` is in progress.  The streams are sent the
            graph without it (they used to go on serving the removed node,
            and the binary stream re-sent a schema that had it)."""
            with self._graph_transaction("remove a node", write=True):
                try:
                    # What remove_node() warns about, for the reply.
                    notes = self.gm._remove_node(name, replacing=False)
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail=str(exc))
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                # The server's own record of a surrogate under that name
                # goes with the node: a later deactivate used to add the
                # recorded original back, with its edges, into a graph the
                # node had been deleted from.  And the edges another
                # surrogate's record holds to or from the node go with it,
                # as the graph's own do: that deactivate used to fail on
                # them (a 500), for good.
                self._original_nodes.pop(name, None)
                self._active_surrogates.discard(name)
                for other, (orig, edges, ext) in list(self._original_nodes.items()):
                    kept = [e for e in edges
                            if name not in (e.source_node, e.target_node)]
                    if len(kept) != len(edges):
                        self._original_nodes[other] = (orig, kept, ext)
                self._publish_state()
            if notes:
                return {"status": "ok", "coupling_groups": "  ".join(notes)}
            return {"status": "ok"}

        @app.post("/graph/edges", tags=["graph"], status_code=201, response_model=None)
        def add_edge(req: AddEdgeRequest) -> dict[str, str]:
            """Add an edge.  Refused (409) while the runner runs or a
            ``POST /sim/run`` is in progress: an edge the graph cannot
            validate (shapes that do not match) used to be taken beside a
            running runner, whose next step raised and whose thread died
            while the routes went on reporting it started."""
            with self._graph_transaction("add an edge", write=True):
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
            """Remove every edge with these endpoints and fields; a 404 when
            the graph has none (it used to answer 200 and change nothing).
            Refused (409) while the runner runs or a ``POST /sim/run`` is in
            progress."""
            with self._graph_transaction("remove an edge", write=True):
                if not any(
                    e.source_node == req.source_node and e.target_node == req.target_node
                    and e.source_field == req.source_field
                    and e.target_field == req.target_field
                    for e in self.gm._edges
                ):
                    raise HTTPException(
                        status_code=404,
                        detail=(f"No edge {req.source_node}.{req.source_field} -> "
                                f"{req.target_node}.{req.target_field}."),
                    )
                self.gm.remove_edge(
                    source=req.source_node, target=req.target_node,
                    source_field=req.source_field, target_field=req.target_field,
                )
            return {"status": "ok"}

        @app.post("/graph/compile", tags=["graph"], response_model=None)
        def compile_graph() -> dict[str, Any]:
            """Compile the graph; a graph it cannot compile is a 400 naming
            why (an edge it cannot validate was a 500)."""
            with self._graph_transaction("compile the graph"):
                try:
                    self.gm.compile()
                except _GRAPH_CONFIGURATION_ERRORS as exc:
                    detail = _cannot_step_detail(exc)
                    if not isinstance(exc, RuntimeError):
                        detail = detail.replace("cannot step", "cannot compile").replace(
                            "nothing was stepped", "nothing was compiled")
                    raise HTTPException(status_code=400, detail=detail)
                return {"status": "ok", "schedule": self.gm.schedule}

        @app.post("/graph/validate", tags=["graph"], response_model=None)
        def validate_graph() -> dict[str, Any]:
            with self._graph_access("validate the graph"):
                issues = self.gm.validate()
            return {"issues": issues}

        # -- state endpoints -------------------------------------------------

        @app.get("/graph/state", tags=["state"], response_model=None)
        def get_state() -> dict[str, Any]:
            with self._graph_access("read the state"):
                return self._state_json()

        @app.get("/graph/state/{node_name}", tags=["state"], response_model=None)
        def get_node_state(node_name: str) -> dict[str, Any]:
            with self._graph_access("read the state"):
                try:
                    return self._node_state_json(node_name)
                except KeyError as exc:
                    raise HTTPException(status_code=404, detail=str(exc))

        @app.put("/graph/state/{node_name}", tags=["state"], response_model=None)
        def set_node_state(node_name: str, req: SetNodeStateRequest) -> dict[str, str]:
            """Replace a node's state.  Every field is required, coerced to
            the live leaf's dtype, and must match its shape and be finite;
            a 400 names the field and writes nothing.  Text, a boolean for
            a numeric field and ``null`` are refused, and an integer field
            takes only integral values inside its dtype's range
            (:func:`_state_value_refusal`).  Each field's value
            count is checked against the live field before anything is
            converted to an array."""
            with self._graph_transaction("write a node's state", write=True):
                if node_name not in self.gm._nodes:
                    raise HTTPException(status_code=404, detail=f"No node '{node_name}'.")
                live = self.gm.get_node_state(node_name)
                if set(req.state) != set(live):
                    raise HTTPException(
                        status_code=400,
                        detail=f"state fields must be exactly {sorted(live)}; got {sorted(req.state)}",
                    )
                # Counted on the JSON before any of it becomes an array: the
                # shape check used to run after the conversion, and a body of
                # 16 million numbers for a scalar field took 2 GB to refuse.
                for field, value in req.state.items():
                    want_size = int(np.prod(np.shape(live[field])))
                    got_size, got_lists = _json_value_count(value)
                    # The lists as well as the numbers: a million empty
                    # lists hold as many numbers as an empty field.
                    if got_size == want_size and got_lists != _lists_in_shape(
                            tuple(np.shape(live[field]))):
                        raise HTTPException(
                            status_code=400,
                            detail=(f"{field}: expected shape "
                                    f"{tuple(np.shape(live[field]))}, got a value "
                                    f"nested in {got_lists} list(s)"),
                        )
                    if got_size != want_size:
                        raise HTTPException(
                            status_code=400,
                            detail=(f"{field}: expected {want_size} value(s) (shape "
                                    f"{tuple(np.shape(live[field]))}), got {got_size}"),
                        )
                staged = {}
                for field, value in req.state.items():
                    want = jnp.asarray(live[field])
                    # Text, booleans, non-integral or out-of-range integers:
                    # refused as every other write surface refuses them.
                    problem = _state_value_refusal(value, want.dtype)
                    if problem is not None:
                        raise HTTPException(status_code=400, detail=f"{field}: {problem}")
                    # As for PUT /graph/params: refused before the cast warns.
                    problem = _unrepresentable(value, want.dtype)
                    if problem is not None:
                        raise HTTPException(status_code=400, detail=f"{field}: {problem}")
                    try:
                        arr = jnp.asarray(value, dtype=want.dtype)
                    except OverflowError:
                        # An integer past float64 for a float field (10**400).
                        raise HTTPException(
                            status_code=400,
                            detail=f"{field}: value does not fit its type {want.dtype}")
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
                # Published, so the streams show the written state (they
                # served the value from before the write until a step).
                self._publish_state()
            return {"status": "ok"}

        # -- parameter endpoints ---------------------------------------------

        @app.get("/graph/params/{node_name}", tags=["params"], response_model=None)
        def get_node_params(node_name: str) -> dict[str, Any]:
            with self._graph_access("read a node's params"):
                if node_name not in self.gm._nodes:
                    raise HTTPException(status_code=404, detail=f"No node '{node_name}'.")
                node = self.gm._nodes[node_name].node
                # The live pytree wins over the constructor value: it is what
                # the step uses after a fit or a checkpoint restore.
                live = self.gm.params.get("nodes", {}).get(node_name) or {}
                return _json_reply({**_jax_to_python(node.params), **_jax_to_python(live)})

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
            a request whose values the node's constructor refuses, asked
            with every changed key at once and the params a save would
            carry: a graph saved with them could not be loaded.  And so is
            one the running node would honour differently from the node a
            saved graph rebuilds -- a constructor that derives a branch or
            an array from the value (``LBMPipeNode``'s multiphase switch
            from ``G != 0``) leaves the running node computing with what it
            derived from the old one.  And so is a value the node's step
            cannot run with: its hooks are traced with the request's values
            on a copy of the node, and a trace that raises where the
            current values' trace does not is a 400 naming the error.

            A structural value is stored in the parameter's own numeric
            type: a float with no fractional part written for an integer
            parameter is stored as that integer (``stencil_order: 4.0`` is
            ``4``), any other float for one is a 400, and an integer for a
            float parameter is stored as a float.

            A non-finite number anywhere in the request is a 400 before
            anything else.  Integers and element counts are bounded as in
            ``POST /graph/nodes`` (a 422 from the request model, or from
            this route for an integral float), and a value that would take
            the node past :data:`MAX_NODE_STATE_ELEMENTS` or
            :data:`MAX_NODE_BUILD_BYTES`, or change its state's layout, is
            refused before any node or state is built with it -- told from
            the class's own size estimate where it has one, and wherever
            the node's ``initial_state()`` can be evaluated abstractly.

            Taken while the runner runs (the UI's sliders), between two of
            its steps: the graph lock makes the whole write one change as
            far as any step is concerned.
            """
            with self._graph_transaction("write a node's params"):
                return set_node_params_locked(node_name, req)

        def set_node_params_locked(node_name: str, req: SetNodeParamsRequest) -> dict[str, Any]:
            """``PUT /graph/params/{node_name}``, holding the graph lock."""
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
            # Every key is checked before any is written: an
            # out-of-bounds slider value is a 400 here, not a NaN later.
            # Every check a live leaf needs (dtype coercion, shape,
            # finiteness, bounds) runs in this first loop and the second
            # only writes.  That order is no longer what leaves the first
            # two keys unwritten when the third is refused -- the route's
            # transaction is, wherever the refusal comes from -- but it
            # keeps a refusal cheap and the checks in one place.
            staged: dict[str, Any] = {}
            # What a structural (non-leaf) key would store in node.params:
            # the JSON value in the parameter's own numeric type.
            structural: dict[str, Any] = {}
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
                    structural[key] = value
                    continue
                if _holds_text(value) and (key in live
                                           or _is_numeric_value(node.params.get(key))):
                    # NumPy parses "1.5" as 1.5, so a numeric string was
                    # stored as the number, where every FMU door refuses a
                    # string and the comment below says a string is a 400.
                    raise HTTPException(status_code=400,
                                        detail=f"{key}: expected a number, got a string")
                refusal = _text_value_refusal(value)
                if refusal is not None:
                    # As POST /graph/nodes refuses it: the config could not
                    # carry this text, and neither could this route's reply.
                    raise HTTPException(status_code=400, detail=f"{key}: {refusal}")
                if key not in live:
                    # Kept in the parameter's type: a float written for an
                    # integer used to be stored as a float (HeatNode
                    # ``stencil_order: 4.0`` became a trainable leaf of
                    # gm.params), an integer for a float as an int.
                    coerced, problem = _coerced_to_param_type(node.params.get(key), value)
                    if problem is not None:
                        raise HTTPException(status_code=400, detail=f"{key}: {problem}")
                    # An integral float is an integer from here on, so it is
                    # bounded as the request model bounds one sent as such.
                    oversized = _oversized_param(coerced, key)
                    if oversized is not None:
                        raise HTTPException(status_code=422, detail=oversized)
                    # A finite number inside a structural value that the
                    # node's float dtype overflows (RigidBodyNode
                    # ``constraints: {"z": 1e39}``) was taken, and every
                    # step after it produced infinities: refused here as a
                    # live leaf's is by _unrepresentable below.
                    overflow = _overflowing_float(coerced, key)
                    if overflow is not None:
                        raise HTTPException(
                            status_code=400,
                            detail=(f"{overflow}: value does not fit its type "
                                    f"{np.dtype(jax.dtypes.canonicalize_dtype(jnp.float64))}, "
                                    f"got {value!r}"),
                        )
                    structural[key] = coerced
                    continue
                # An integer leaf is an integer target: bounded as a
                # structural integer is.  A float leaf takes any integral
                # JSON number as the float it is.
                if jnp.issubdtype(live[key].dtype, jnp.integer):
                    oversized = _oversized_param(value, key)
                    if oversized is not None:
                        raise HTTPException(status_code=422, detail=oversized)
                # Before the cast, which overflows 1e39 into float32's inf
                # with a RuntimeWarning (a 500 under -W error) and so made
                # the refusal below say "must be finite" of a finite value.
                problem = _unrepresentable(value, live[key].dtype)
                if problem is not None:
                    raise HTTPException(status_code=400,
                                        detail=f"{key}: {problem}, got {value!r}")
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
            # changes; an unchanged key has nothing to refuse.  A leaf
            # changes when it differs from the value the graph runs with now
            # -- the live leaf -- not from the node's own: after a fit, or a
            # POST /checkpoint/load (which moves gm.params, not
            # node.params), the two differ, and a leaf written back to the
            # node's own value used to be dropped here.  The combined checks
            # below were then asked with its old live value: a HeatNode
            # rod's ``length`` and ``thermal_diffusivity`` went past its
            # Fourier limit together with a 200, ran to NaN, and saved a
            # graph that did not reload.  Such a key is asked as
            # ``at_own_value``, as a load asks it: the node was built with
            # it, so the per-key checks have nothing to ask, and the
            # combined ones take it with every other key's live value.
            ctor = node.params_pytree() if staged else {}
            changes: dict[str, Any] = {}
            at_own_value: list[str] = []
            for key, value in req.params.items():
                if key in staged:
                    if _leaf_values_equal(staged[key], live[key]):
                        continue
                    changes[key] = np.asarray(staged[key]).tolist()
                    own_leaf = ctor.get(key)
                    if own_leaf is not None and _leaf_values_equal(staged[key], own_leaf):
                        at_own_value.append(key)
                else:
                    value = structural[key]
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

            # Everything that builds or traces the node with the new values,
            # in the one decision a checkpoint load shares
            # (:func:`_params_write_refusal`): the size of what they would
            # build, the state's layout, a value the node consumed when it
            # was constructed, its constructor (each structural key with
            # the request's other changes), a mapping's points, the step's
            # trace, and the graph a save would reload.
            accepts = self.gm._nodes[node_name].accepts_params
            found = _params_write_refusal(
                self.gm, node_name, changes, staged,
                dict(live) if accepts else None,
                {**live, **staged} if accepts else None,
                at_own_value=at_own_value)
            if found is not None:
                keys, reason, reported = found
                raise refused(keys, reason, reported=reported)
            # After a compile ``live`` *is* gm.params' leaf dict and this is
            # the write; before one it is the throwaway probe copy, and this
            # only keeps the echo below honest -- it used to report the
            # pre-write value, contradicting the GET that follows it.
            _install_param_leaves(
                self.gm, node_name,
                {key: staged[key] for key in req.params if key in staged}, live)
            for key in req.params:
                if key not in staged:
                    node.params[key] = structural[key]
                    self.gm._dirty = True
            shown = _json_reply({**_jax_to_python(node.params), **_jax_to_python(live)})
            return {"status": "ok", "params": shown}

        # -- checkpoint endpoints -------------------------------------------

        def _checkpoint_path(name: str, *, loading: bool = False) -> Path:
            """The file ``/checkpoint/save`` writes (or ``/load`` reads) for
            *name*, resolved, after checking it is a file strictly inside
            the checkpoint root.

            The check is made on the file NumPy actually touches, not only
            on the name: ``numpy.savez`` appends ``.npz``, so a name that
            resolved to the root itself (``""``, ``"."``, ``"sub/.."``) used
            to write -- and load -- ``<root>.npz``, in the root's parent.
            A name with a NUL byte, or one the filesystem cannot represent,
            is a 400 rather than a 500.
            """
            root = self.checkpoint_root
            outside = HTTPException(
                status_code=400,
                detail=(f"checkpoint path must stay under the checkpoint root, naming "
                        f"a file inside it (got {name!r})"),
            )
            if not name or "\x00" in name:
                raise outside
            uncarriable = _uncarriable_characters(name)
            if uncarriable:
                # The rule of a node's name (GraphManager.add_node): the
                # name is written to the manifest, the reply and the log.
                raise HTTPException(
                    status_code=400,
                    detail=(f"checkpoint path {name!r} is invalid: it contains "
                            f"{', '.join(uncarriable)}, and {_UNCARRIABLE_WHY}."),
                )
            try:
                target = (root / name).resolve()
                if root == target or root not in target.parents:
                    raise outside
                if not (loading and target.is_file()) and target.suffix != ".npz":
                    target = target.with_suffix(target.suffix + ".npz")
                target = target.resolve()
            except HTTPException:
                raise
            except (OSError, ValueError, RuntimeError) as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"checkpoint path {name!r} is not usable: {_checkpoint_reason(exc)}",
                ) from None
            if root == target or root not in target.parents:
                raise outside
            return target

        def _name_too_long(target: Path, manifest: Optional[Path]) -> Optional[str]:
            """Why the filesystem under the checkpoint root cannot hold a
            name a save of *target* (and its *manifest*) would create --
            one longer than its ``NAME_MAX`` bytes -- or ``None``."""
            root = self.checkpoint_root
            try:
                limit = int(os.pathconf(root, "PC_NAME_MAX"))
            except (OSError, ValueError, AttributeError):
                limit = 255
            names = [(part, "a name") for part in target.relative_to(root).parts]
            if manifest is not None:
                names.append((manifest.name, "its manifest's name (the checkpoint's "
                                             "name and '.manifest.json')"))
            for part, what in names:
                try:
                    size = len(os.fsencode(part))
                except ValueError:          # a lone surrogate, say
                    return f"{what} cannot be represented by this filesystem"
                if size > limit:
                    return (f"{what} would be {size} bytes, and this filesystem takes "
                            f"at most {limit} in one name")
            return None

        def _checkpoint_reason(exc: BaseException) -> str:
            """Why a checkpoint route failed, for its 4xx detail, naming no
            absolute server path.  An ``OSError`` is told by its
            ``strerror`` ("Is a directory"), not by its text, which carries
            the absolute file name; any other message has the checkpoint
            root cut out of it, so a file inside it is named as the client
            named it and one outside it as ``<checkpoint root>/..``."""
            if isinstance(exc, OSError) and exc.strerror:
                return exc.strerror
            text = str(exc)
            root = str(self.checkpoint_root)
            return text.replace(root + os.sep, "").replace(root, "<checkpoint root>")

        def _manifest_path(target: Path) -> Optional[Path]:
            """The manifest beside checkpoint *target*
            (:func:`~maddening.core.simulation.checkpoint.write_manifest`'s
            name for it), or ``None`` when it would resolve outside the
            checkpoint root (a symlink)."""
            manifest = target.with_name(target.name + ".manifest.json")
            try:
                resolved = manifest.resolve()
            except (OSError, RuntimeError, ValueError):
                return None
            root = self.checkpoint_root
            return manifest if root in resolved.parents else None

        def _checkpoint_clock(target: Path) -> Optional[tuple[float, int]]:
            """``(sim_time, step_count)`` this server recorded when it saved
            *target*, or ``None``: no manifest, one that does not hash to
            the file (it was replaced since), or one written by something
            else."""
            from maddening.core.simulation.checkpoint import (  # noqa: PLC0415
                verify_manifest,
            )
            manifest_path = _manifest_path(target)
            try:
                if manifest_path is None or not manifest_path.is_file() \
                        or manifest_path.stat().st_size > 1 << 20:
                    return None
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                verify_manifest(target, manifest)
                clock = manifest["extra"]["server_clock"]
                sim_time, steps = float(clock["sim_time"]), int(clock["step_count"])
            except Exception:  # noqa: BLE001 - no clock to restore
                return None
            if not math.isfinite(sim_time) or sim_time < 0 or steps < 0:
                return None
            return sim_time, steps

        @app.post("/checkpoint/save", tags=["checkpoint"], response_model=None)
        def checkpoint_save(path: str = "checkpoint.npz") -> dict[str, Any]:
            """Save the graph's state and params to *path* under the
            checkpoint root, and beside it a manifest
            (``<path>.manifest.json``: its SHA-256 and the streams' clock,
            ``sim_time`` and step count), which ``POST /checkpoint/load``
            restores the clock from."""
            from maddening.core.simulation.checkpoint import (  # noqa: PLC0415
                write_manifest,
            )
            target = _checkpoint_path(path)
            manifest = _manifest_path(target)
            name = target.relative_to(self.checkpoint_root).as_posix()
            # Before anything is created: a name the filesystem cannot hold
            # -- the checkpoint's, a directory's, or the manifest's (the
            # checkpoint's name and ".manifest.json") -- used to be found
            # only when the file was written, after its directories had
            # been made, and the 400 that said "nothing was written" left
            # them behind.  The temporary names no longer carry the
            # checkpoint's name, which put a valid 183-byte name over the
            # limit.
            too_long = _name_too_long(target, manifest)
            if too_long is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"could not save checkpoint {name!r}, nothing was written: "
                           f"{too_long}")
            with self._graph_transaction("save a checkpoint"):
                self._ensure_relay_attached()
                clock = {"sim_time": self.relay.elapsed,
                         "step_count": self.relay.step_count}
                # Both files are written under temporary names first and
                # moved into place after both exist -- the checkpoint, then
                # its manifest -- so a save refused before the first move
                # writes nothing at all.  The manifest used to be written
                # beside the target before the checkpoint's move, so a save
                # refused at that move (its name a directory) answered
                # "nothing was written" having replaced the manifest of
                # that name.  A name that is not a file is refused before
                # anything is written; the one failure left after the first
                # move -- the manifest's own move -- is answered as what it
                # is: the checkpoint written, without its clock.
                uid = uuid.uuid4().hex
                partial = target.with_name(f".{uid}.partial.npz")
                manifest_partial = (None if manifest is None else
                                    manifest.with_name(f".{uid}.partial.manifest"))
                placed = False
                made: list[Path] = []     # directories this save created, deepest first
                try:
                    for existing in (target, manifest):
                        if existing is not None and (existing.exists() or existing.is_symlink()) \
                                and not existing.is_file():
                            raise IsADirectoryError(
                                21, f"{existing.name!r} exists and is not a file")
                    missing = target.parent
                    while missing != self.checkpoint_root and not missing.exists():
                        made.append(missing)
                        missing = missing.parent
                    target.parent.mkdir(parents=True, exist_ok=True)
                    self.gm.save_state(str(partial))
                    if manifest_partial is not None:
                        write_manifest(partial, extra={"server_clock": clock},
                                       manifest_path=manifest_partial)
                    os.replace(partial, target)
                    placed = True
                    if manifest_partial is not None:
                        os.replace(manifest_partial, manifest)
                except Exception as exc:  # noqa: BLE001
                    reason = _checkpoint_reason(exc)
                    if placed:
                        detail = (f"checkpoint {name!r} was written, but its manifest "
                                  f"could not be: {reason}; it loads, with sim_time "
                                  "counted from zero")
                    else:
                        detail = f"could not save checkpoint {name!r}, nothing was written: {reason}"
                    raise HTTPException(status_code=400, detail=detail) from None
                finally:
                    for leftover in (partial, manifest_partial):
                        if leftover is not None:
                            with contextlib.suppress(OSError):
                                leftover.unlink(missing_ok=True)
                    if not placed:
                        # "Nothing was written" includes the directories.
                        for directory in made:
                            with contextlib.suppress(OSError):
                                directory.rmdir()
            return {"status": "ok", "path": str(target), "sim_time": clock["sim_time"]}

        @app.post("/checkpoint/load", tags=["checkpoint"], response_model=None)
        def checkpoint_load(path: str = "checkpoint.npz") -> dict[str, Any]:
            """Load a checkpoint saved under the checkpoint root.  Refused
            (409) while the runner runs or a ``POST /sim/run`` is in
            progress: its next store would overwrite the load.

            A checkpoint that does not fit this graph is a 400 and nothing
            is loaded: other node or field names, a field or parameter of
            another shape, a value its dtype cannot hold, or a parameter
            value ``PUT /graph/params`` would refuse on the node as it
            stands -- non-finite, outside its ``ParamSpec`` bounds, one
            the node's constructor refuses with the graph's other values
            (so a save after the load would not reload), one the node
            consumed when it was constructed, or one that moves the points
            a mapped edge was built from; and text or a boolean for a
            numeric parameter.  Each is asked of what the load changes
            only, and finiteness and the bounds only of a value that is
            neither the leaf's now nor the node's own (per element, as the
            FMU's ``set_fmu_state`` asks), so a graph built outside its
            bounds reloads its own checkpoint.
            ``GraphManager.load_state`` itself refuses text and booleans,
            and asks none of the rest: a value outside a ``ParamSpec``'s
            bounds is one Python code may hold on purpose
            (``docs/user_guide/parameters.md``).

            The streams (``/ws/state``, ``/ws/state/binary``, ``/ws/render``)
            serve the loaded state at once, and their ``sim_time`` is the
            checkpoint's: the one this server recorded in the manifest when
            it saved it, or zero -- counted from the load -- for a file
            without one (``sim_time_from_checkpoint`` says which).  They
            used to serve the state from before the load until the next
            step, and to go on counting from the step before it.
            """
            with self._graph_transaction("load a checkpoint", write=True):
                target = _checkpoint_path(path, loading=True)
                try:
                    found = target.is_file()
                except OSError:
                    found = False
                if not found:
                    raise HTTPException(status_code=404, detail=f"no checkpoint {path!r}")
                from maddening.core.simulation.checkpoint import (  # noqa: PLC0415
                    CheckpointFormatError,
                    _restore_state_and_params,
                    _state_and_params_snapshot,
                )
                try:
                    # load_state compiles a dirty graph before it reads
                    # anything; done first here so ``before`` is the
                    # compiled graph's.
                    if self.gm._dirty or self.gm._compiled_step is None:
                        try:
                            self.gm.compile()
                        except _GRAPH_CONFIGURATION_ERRORS as exc:
                            # The graph's reason, as POST /sim/step gives
                            # it: this used to be answered "could not load
                            # checkpoint" of a file nothing was wrong with.
                            raise HTTPException(
                                status_code=400,
                                detail=(f"could not load checkpoint {path!r}: the graph "
                                        "cannot compile with its current configuration "
                                        f"({_configuration_reason(exc)}); nothing was "
                                        "loaded"),
                            ) from None
                    # Not an undo (a refusal is undone by the route's
                    # transaction, the compile above included): the values
                    # the checks below are asked against.
                    before = _state_and_params_snapshot(self.gm)
                    self.gm.load_state(str(target))
                except HTTPException:
                    raise
                except CheckpointFormatError:
                    # NumPy's own reasons -- how to load the file unsafely --
                    # are not the client's business.
                    raise HTTPException(
                        status_code=400,
                        detail=(f"could not load checkpoint {path!r}: it is not an .npz "
                                "archive of plain arrays saved by this API"),
                    ) from None
                except ValueError as exc:   # load_state's own: what does not fit
                    raise HTTPException(status_code=400,
                                        detail=_checkpoint_reason(exc)) from None
                except Exception:  # noqa: BLE001 - do not leak file/parse internals
                    raise HTTPException(status_code=400, detail=f"could not load checkpoint {path!r}")
                # The parameter values the load would leave are asked what
                # PUT /graph/params asks of a write -- finite, inside their
                # ParamSpec bounds, taken by the node's constructor with the
                # graph's other values, moving no mapped point, ... -- by the
                # same code (_loaded_params_refusal), of the graph as it was
                # before the load.  A checkpoint used to restore values PUT
                # refuses (a HeatNode past its Fourier limit at this graph's
                # timestep, a damping below its bound of 0), answer 200, and
                # leave a graph whose save did not reload and whose steps
                # diverged.
                loaded = _state_and_params_snapshot(self.gm)
                _restore_state_and_params(self.gm, before)
                try:
                    refusal = _loaded_params_refusal(self.gm, loaded[1])
                except Exception as exc:  # noqa: BLE001 - a check that cannot run refuses
                    refusal = (f"its parameters could not be checked "
                               f"({type(exc).__name__}: {_checkpoint_reason(exc)})")
                if refusal is not None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"checkpoint {path!r} does not fit this graph, nothing "
                               f"was loaded: {refusal}",
                    )
                _restore_state_and_params(self.gm, loaded)
                # Each leaf the load changed is installed as PUT installs a
                # written one -- in gm.params and in the node's own params
                # -- by PUT's own function.  The load used to move gm.params
                # only, so an initial condition it changed differed from
                # the node's own value, which the graph refuses of a leaf
                # only initial_state() reads: every checkpoint saved before
                # a PUT of an initial_* parameter was a 400, where PUT took
                # the checkpoint's value.
                held = self.gm.params.get("nodes") or {}
                for owner, leaves in (loaded[1].get("nodes") or {}).items():
                    changed = _leaves_a_load_changes(
                        leaves, (before[1].get("nodes") or {}).get(owner))
                    if changed and owner in self.gm._nodes \
                            and isinstance(held.get(owner), dict):
                        _install_param_leaves(self.gm, owner, changed, held[owner])
                # What is left for the graph to refuse is a leaf the load
                # did NOT change: one that already differed from its node's
                # own value in a way the step cannot read (written into
                # gm.params by Python code, on a graph not stepped since).
                # A load that answered 200 would leave a graph whose every
                # POST /sim/step is a 400 while GET /graph/params served
                # the leaf; the route's transaction undoes the load.
                try:
                    self.gm._refuse_baked_param_writes(self.gm.params, live=False)
                except _BakedParamWrite as exc:
                    own = ("" if exc.own is None
                           else f" and {_shown_value(exc.own)} on the node itself")
                    back = ("" if exc.own is None or np.size(exc.own) > 8 else (
                        f"  To load it, first write the node's own value back (PUT "
                        f"/graph/params/{exc.owner} with {exc.key}: "
                        f"{_shown_value(exc.own)}); the load is then asked what "
                        "that route asks of the checkpoint's value."))
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"checkpoint {path!r} does not fit this graph, nothing "
                            f"was loaded: node {exc.owner!r}, {exc.key}: the value is "
                            f"{_shown_value(exc.value)} in the checkpoint and in the "
                            f"graph's params{own}, and {exc.reason}.{back}"),
                    )
                clock = _checkpoint_clock(target)
                sim_time, steps = clock if clock is not None else (0.0, 0)
                self._ensure_relay_attached()
                self.relay.restore(self._user_state(), step_count=steps, elapsed=sim_time)
                return {"status": "ok", "state": self._state_json(),
                        "sim_time": sim_time, "sim_time_from_checkpoint": clock is not None}

        # -- simulation control endpoints -----------------------------------

        @app.post("/sim/step", tags=["sim"], response_model=None)
        def sim_step() -> dict[str, Any]:
            """Advance the graph one step.  A graph that cannot step is a
            400 naming why (nothing is stepped); a 409 while the runner is
            started, whose next step would overwrite this one's, or while a
            ``POST /sim/run`` is in progress.  Concurrent requests are
            stepped one after another: N of them take N steps."""
            with self._graph_transaction("step the graph", write=True):
                self._ensure_relay_attached()
                try:
                    self.gm.step()
                except _GRAPH_CONFIGURATION_ERRORS as exc:
                    raise HTTPException(status_code=400, detail=_cannot_step_detail(exc))
                return self._state_json()

        def _run_failed(exc: BaseException, steps_run: int, n_steps: int) -> _Reply:
            """The 400 of a ``POST /sim/run`` whose step raised after
            *steps_run* steps had been stored."""
            detail = _cannot_step_detail(exc)
            if not steps_run and "nothing was stepped" not in detail:
                detail += " (nothing was stepped)"     # a RuntimeError's own words
            if steps_run:
                detail = detail.replace(
                    "; nothing was stepped",
                    f"; the run stopped at step {steps_run + 1} of {n_steps}, and the "
                    f"graph is left after the {steps_run} step(s) it took")
                if detail == _cannot_step_detail(exc):     # a RuntimeError's own words
                    detail += (f" (the run stopped at step {steps_run + 1} of {n_steps}, "
                               f"and the graph is left after the {steps_run} step(s) "
                               "it took)")
            return _Reply(status_code=400, content={
                "detail": detail, "steps_run": steps_run, "n_steps": n_steps})

        def _run_failed_unexpectedly(steps_run: int, n_steps: int) -> _Reply:
            """The 500 of a ``POST /sim/run`` one of whose slices failed
            unexpectedly after *steps_run* steps had been stored by the
            slices before it: that slice was put back, and those steps were
            taken, which the reply says."""
            detail = _UNEXPECTED_FAILURE_DETAIL
            if steps_run:
                detail = (
                    f"{_FAILED_UNEXPECTEDLY[:-1]} after {steps_run} of the run's "
                    f"{n_steps} step(s). The graph is left after those "
                    f"{steps_run}, exactly as it was before the part of the run "
                    "that failed; GET /graph/state reads it.")
            return _Reply(status_code=500, content={
                "detail": detail, "steps_run": steps_run, "n_steps": n_steps})

        @app.post("/sim/run", tags=["sim"], response_model=None)
        def sim_run(
            n_steps: int = Query(
                100, ge=0, le=MAX_RUN_STEPS,
                description="Steps to run synchronously.  Zero is a no-op "
                            "that returns the current state.  The upper "
                            "bound exists because the request holds a "
                            "worker for its whole duration; for a longer "
                            "run use POST /sim/start.",
            ),
        ) -> Any:
            """Run *n_steps* steps and return the state.

            The run steps in slices of about :data:`_RUN_SLICE_SECONDS`,
            each holding the graph lock: reads are served between slices,
            and writes -- another step or run, a state, a reset, a load, a
            structural edit -- are refused (409) until it returns.  When
            the server is told to shut down (SIGINT, SIGTERM, the lifespan
            shutdown, :meth:`SimulationServer.request_shutdown`) the run
            stops at its next slice and answers **503** with
            ``{"status": "interrupted", "steps_run": k, "n_steps": n}``:
            the graph is left after the *k* steps it took.  A shutdown used
            to wait for the whole run, up to an hour.

            A step that raises answers 400 with ``steps_run``: the steps
            before it were taken and stored (a run-time check in the step,
            such as ``strict_convergence``, can raise part-way through), and
            the reply used to say "nothing was stepped" and not how many.

            Every graph-lock wait is bounded by :data:`_GRAPH_LOCK_TIMEOUT`,
            the final state's read and an ``n_steps=0`` compile included.
            One that runs out answers the same **503** ``interrupted`` body,
            ``steps_run`` saying how many steps were taken and stored: retry
            the whole run only when it is 0 (the reply then carries
            ``Retry-After``); otherwise the detail says how many remain.  A
            run that timed out between slices used to answer "Nothing was
            changed; retry shortly" after it had stepped the graph.
            """
            with self._graph_transaction("run the graph", write=True):
                self._ensure_relay_attached()
                self._sync_run_active = True
            done, chunk = 0, 1
            interrupted: Optional[str] = None
            try:
                while done < n_steps:
                    if self._shutdown.is_set():
                        interrupted = "The server is shutting down"
                        break
                    k = min(chunk, n_steps - done)
                    t0 = time.perf_counter()
                    taken: list = []
                    try:
                        # One transaction a slice, not one a run: the lock
                        # is released between slices, where a PUT
                        # /graph/params is taken, and putting the graph back
                        # as it was before the run would undo that write.
                        with self._graph_transaction("run the graph"):
                            try:
                                self.gm.run(k, callback=lambda i, _s: taken.append(i))
                            except _GRAPH_CONFIGURATION_ERRORS as exc:
                                # A step can raise at run time (an in-graph
                                # check, strict_convergence) after the steps
                                # before it were stored: say how many.
                                steps_run = done + len(taken)
                                return _run_failed(exc, steps_run, n_steps)
                    except HTTPException as exc:
                        if exc.status_code == 500:
                            return _run_failed_unexpectedly(done, n_steps)
                        if exc.status_code != 503:
                            raise
                        # The graph lock not had in time between two
                        # slices.  The 503 used to say "Nothing was
                        # changed; retry shortly" after the slices before
                        # it had stepped the graph, so a client retrying
                        # the whole run stepped it twice.
                        interrupted = (f"The graph has been in use for "
                                       f"{_GRAPH_LOCK_TIMEOUT:g} s by another request")
                        break
                    done += k
                    if time.perf_counter() - t0 < _RUN_SLICE_SECONDS / 2:
                        chunk = min(2 * chunk, MAX_RUN_STEPS)
                state = None
                if interrupted is None:
                    # The last access, with a timeout like every other: it
                    # used to wait for the graph lock with none, behind a
                    # long holder for as long as it held it.
                    try:
                        with self._graph_transaction("read the run's final state"):
                            if n_steps == 0:
                                try:
                                    self.gm.run(0)      # compiles a graph edited since
                                except _GRAPH_CONFIGURATION_ERRORS as exc:
                                    return _run_failed(exc, 0, 0)
                            state = self._state_json()
                    except HTTPException as exc:
                        if exc.status_code == 500:
                            return _run_failed_unexpectedly(done, n_steps)
                        if exc.status_code != 503:
                            raise
                        interrupted = (f"The graph has been in use for "
                                       f"{_GRAPH_LOCK_TIMEOUT:g} s by another request, "
                                       "so this run's final state could not be read "
                                       "(GET /graph/state reads it)")
            finally:
                self._sync_run_active = False
            if interrupted is not None:
                rest = (f"this run took {done} of its {n_steps} steps, and the graph is "
                        "left after them.")
                retry = None
                if self._shutdown.is_set():
                    pass
                elif not done:
                    rest = (f"this run took none of its {n_steps} steps.  Nothing was "
                            "changed; retry shortly.")
                    retry = {"Retry-After": "1"}
                elif done < n_steps:
                    rest += (f"  Do not repeat the whole run: POST /sim/run?n_steps="
                             f"{n_steps - done} takes the rest.")
                else:
                    rest += "  Do not repeat it: every step was taken."
                return _Reply(
                    status_code=503, headers=retry,
                    content={"status": "interrupted", "detail": f"{interrupted}: {rest}",
                             "steps_run": done, "n_steps": n_steps})
            return state

        async def on_runner_pool(fn, *, pool=None) -> Any:
            """Run a runner route's blocking *fn* on the runner routes' own
            threads (:data:`_RUNNER_ROUTE_WORKERS`, or *pool*), with the
            deadline its waits share counted from now -- the request's
            arrival on the event loop -- as its argument.  Its
            ``HTTPException`` propagates to the route as if raised there."""
            deadline = time.monotonic() + _GRAPH_LOCK_TIMEOUT
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                pool or self._runner_pool, functools.partial(fn, deadline))

        @app.post("/sim/start", tags=["sim"], response_model=None)
        async def sim_start() -> Any:
            """Start the runner.  Answered within about one graph-lock
            timeout of its arrival, a 503 past it: run on the runner routes'
            own threads, so it does not wait for a worker behind requests
            waiting for the graph.  A graph with no nodes is a 409: there is
            nothing to run."""
            return await on_runner_pool(sim_start_blocking)

        def sim_start_blocking(deadline: float) -> dict[str, str]:
            with self._runner_control("start the runner", deadline):
                if self._runner_stopping():
                    raise HTTPException(
                        status_code=503,
                        detail=("The previous run's thread is still finishing a step "
                                "after POST /sim/stop; two runners would step one "
                                "graph. Retry shortly."),
                        headers={"Retry-After": "1"},
                    )
                # A runner whose thread died is not started: it is replaced.
                self._reap_dead_runner()
                if self._runner_started:
                    raise HTTPException(status_code=409, detail="Runner is already started.")
                with self._graph_transaction("start the runner", write=True, deadline=deadline):
                    if not self.gm._nodes:
                        # The runner's thread read gm.timestep on its first
                        # frame and died ("No nodes registered.") after the
                        # route had answered "started".
                        raise HTTPException(
                            status_code=409,
                            detail=("The graph has no nodes, so there is nothing to "
                                    "run. Add a node (POST /graph/nodes), then start "
                                    "the runner."),
                        )
                    runner = self._ensure_runner()
                    try:
                        runner.start()
                        self._runner_started = True
                        self._runner_stop_pending = False
                    except _GRAPH_CONFIGURATION_ERRORS as exc:
                        raise HTTPException(status_code=400, detail=_cannot_step_detail(exc))
            return {"status": "started"}

        def _not_running_detail() -> Optional[str]:
            """The 409 for pause / resume when no runner is running, with
            why a started one stopped; ``None`` when it runs."""
            if self.runner is None or not self._runner_started:
                return "Runner is not started."
            if self.runner.is_alive:
                return None
            why = self._reap_dead_runner()
            return (f"Runner is not running: its thread stopped ({why}). "
                    "POST /sim/start starts a new one.")

        @app.post("/sim/pause", tags=["sim"], response_model=None)
        async def sim_pause() -> Any:
            """Pause the runner (on the runner routes' own threads)."""
            return await on_runner_pool(sim_pause_blocking)

        def sim_pause_blocking(deadline: float) -> dict[str, str]:
            with self._runner_control("pause the runner", deadline):
                detail = _not_running_detail()
                runner = self.runner
                if detail is not None or runner is None:
                    raise HTTPException(status_code=409,
                                        detail=detail or "Runner is not started.")
                runner.pause()
            return {"status": "paused"}

        @app.post("/sim/resume", tags=["sim"], response_model=None)
        async def sim_resume() -> Any:
            """Resume the runner (on the runner routes' own threads)."""
            return await on_runner_pool(sim_resume_blocking)

        def sim_resume_blocking(deadline: float) -> dict[str, str]:
            with self._runner_control("resume the runner", deadline):
                detail = _not_running_detail()
                runner = self.runner
                if detail is not None or runner is None:
                    raise HTTPException(status_code=409,
                                        detail=detail or "Runner is not started.")
                runner.resume()
            return {"status": "resumed"}

        @app.post("/sim/stop", tags=["sim"], response_model=None)
        async def sim_stop() -> Any:
            """Stop the runner and wait for its thread to exit.

            "stopped" means no runner thread is stepping the graph.  When
            the thread is still in a step after the wait, the answer is a
            503 and the runner is kept as stopping: retry, and the route
            waits for it again.  The 503 says the runner was told to stop
            and stays stopped, with ``was_running``.  A runner whose thread
            had already died is reported stopped, with ``error`` saying why
            it died.  Run on the runner routes' own threads, so it is not
            queued behind requests waiting for the graph.
            """
            return await on_runner_pool(sim_stop_blocking)

        def sim_stop_blocking(deadline: float) -> Any:
            with self._runner_control("stop the runner", deadline):
                if self.runner is None or not (
                        self._runner_started or self._runner_stop_pending):
                    raise HTTPException(status_code=409, detail="Runner is not started.")
                runner = self.runner
                died = self._runner_started and not runner.is_alive and runner.error
                was_running = self._runner_running()
                try:
                    self._stop_runner_or_refuse("report the runner stopped")
                except HTTPException as exc:
                    return self._stop_refused(exc, was_running)
            return {"status": "stopped", "error": died} if died else {"status": "stopped"}

        @app.post("/sim/reset", tags=["sim"], response_model=None)
        async def sim_reset() -> Any:
            """Stop the runner and reset all nodes to initial state.  A 503,
            with nothing reset, when the runner's thread will not stop in
            time (it would overwrite the reset); a 409 while a
            ``POST /sim/run`` is in progress.  ``was_running`` says whether
            a runner was running -- one whose thread had died was not.

            The runner is stopped first and the graph lock taken after,
            within about one graph-lock timeout of the request's arrival in
            all (it runs on threads of its own, not the shared workers); when the
            runner will not stop in time, or the graph cannot be had in
            time, the 503 (or a 409) says that the runner it stopped stays
            stopped, with ``was_running``.  The streams are sent the reset
            state at step 0.
            """
            return await on_runner_pool(sim_reset_blocking, pool=self._reset_pool)

        def sim_reset_blocking(deadline: float) -> Any:
            with self._runner_control("reset the graph", deadline):
                was_running = self._runner_running()
                try:
                    self._stop_runner_or_refuse("reset the graph")
                except HTTPException as exc:
                    return self._stop_refused(exc, was_running)
            try:
                with self._graph_transaction("reset the graph", write=True, deadline=deadline):
                    self._ensure_relay_attached()
                    self._reset_state()
                    self.gm._dirty = True
                    return {"status": "ok", "was_running": was_running,
                            "state": self._state_json()}
            except HTTPException as exc:
                return self._after_stopping_the_runner(exc, "reset", was_running)

        @app.put("/sim/stride", tags=["sim"], response_model=None)
        async def sim_set_stride(
            steps_per_frame: Optional[int] = Query(
                None, ge=1, le=MAX_STEPS_PER_FRAME,
                description="Physics steps batched per wall-clock frame in the "
                            "runner, at most MAX_STEPS_PER_FRAME (one POST "
                            "/sim/run's worth).  Kept for a runner started "
                            "later.  Left out: the current value is kept.",
            ),
            relay_stride: Optional[int] = Query(
                None, ge=1, le=MAX_RELAY_STRIDE,
                description="Capture only every Nth step in the relay, at "
                            "least 1 and at most MAX_RELAY_STRIDE.  Left out: "
                            "the current value is kept.",
            ),
        ) -> Any:
            """Adjust physics-to-render rate decoupling.

            Parameters
            ----------
            steps_per_frame : int
                Physics steps batched per wall-clock frame in the runner,
                ``1`` to :data:`MAX_STEPS_PER_FRAME`.  Applied to the
                running runner, and kept for the one a later ``POST
                /sim/start`` creates (it used to be echoed and dropped when
                no runner existed yet).
            relay_stride : int
                Only capture every Nth step in the relay (reduces observer
                overhead for very fast physics), ``1`` to
                :data:`MAX_RELAY_STRIDE`.  ``0`` used to be echoed as ``0``
                and applied as ``1``; it is a 422 now.

            A value left out of the query keeps its current value, so a
            call that changes one leaves the other as it was.  Until 0.4.0
            each defaulted to ``1``: a call naming only ``steps_per_frame``
            reset the relay's stride to ``1``, and the reverse the runner's
            steps per frame, without a word.

            Returns
            -------
            dict
                The values in force, as the runner and the relay hold them.

            Answered on the event loop: it waits for nothing (the stride
            lock is held for a few assignments, never across a wait), so it
            does not queue for a worker thread behind requests waiting for
            the graph.
            """
            # The stride lock alone, held for these assignments: the runner
            # lock is held by routes waiting for the graph, and the stride
            # waited behind them.
            with self._stride_lock:
                if steps_per_frame is not None:
                    self._steps_per_frame = steps_per_frame
                    runner = self.runner
                    if runner is not None:
                        runner.steps_per_frame = steps_per_frame
                if relay_stride is not None:
                    self.relay.stride = relay_stride
                return {
                    "steps_per_frame": self._steps_per_frame,
                    "relay_stride": self.relay.stride,
                }

        # -- surrogate endpoints --------------------------------------------

        @app.post("/surrogate/train", tags=["surrogate"], response_model=None,
                  openapi_extra=_route_openapi("POST", "/surrogate/train"))
        def surrogate_train(req: TrainSurrogateRequest) -> dict[str, Any]:
            """Start training a surrogate of one node in a background thread.
            **Experimental in 0.4.0** (to be hardened in 0.5.0).

            **One job runs at a time**: a 409 while another one does.  The
            last :data:`MAX_SURROGATE_JOBS_KEPT` finished jobs are kept, the
            oldest finished one dropped past it.  **The memory the job would
            take is estimated before it starts**
            (:func:`_surrogate_training_bytes`: the data sweep over the
            whole graph, the dataset and its training copies, the network
            and its optimiser) and refused with 400 over
            :data:`MAX_SURROGATE_TRAIN_BYTES`; the reply carries the
            estimate as ``estimated_bytes``.

            The data come from a batched sweep from varied initial
            conditions (``GraphManager.run_sweep``), which **leaves the live
            graph's state untouched** -- the job used to reset the live
            simulation twice without saying so -- and holds the graph lock
            while it runs (training does not).  A runner may keep running.
            When the server shuts down, the job stops at its next epoch and
            reads ``cancelled``.
            """
            try:
                from maddening.surrogates.architectures.mlp import MLPDirect
                from maddening.surrogates.dataset import DatasetGenerator
                from maddening.surrogates.training.trainer import SurrogateTrainer
            except ImportError:
                raise HTTPException(
                    status_code=400,
                    detail="Surrogate training requires equinox+optax. "
                           "pip install maddening[surrogates]",
                )
            if self._shutdown.is_set():
                raise HTTPException(status_code=503,
                                    detail="The server is shutting down; no job was started.")
            with self._graph_access("start a surrogate job"):
                if req.node_name not in self.gm._nodes:
                    raise HTTPException(status_code=404, detail=f"No node '{req.node_name}'.")
                need = _surrogate_training_bytes(self.gm, req.node_name, req)
            if need > MAX_SURROGATE_TRAIN_BYTES:
                raise HTTPException(
                    status_code=400,
                    detail=(f"this job would take about {format_bytes(need)} (the data "
                            f"sweep over the whole graph, its dataset and the network); "
                            f"at most {format_bytes(MAX_SURROGATE_TRAIN_BYTES)} is accepted "
                            "over the API (this server is unauthenticated -- train a "
                            "surrogate this size in-process).  Fewer n_data_steps, a "
                            "narrower network or a smaller graph take less; nothing "
                            "was started."),
                )

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
                "estimated_bytes": need,
            }

            def _train_worker():
                try:
                    # The data, under the graph lock: generated from batched
                    # initial conditions by ``run_sweep``, which writes
                    # nothing back to the graph.
                    with self._graph_lock:
                        if self._shutdown.is_set():
                            raise _TrainingCancelled()
                        if req.node_name not in self.gm._nodes:
                            raise RuntimeError(
                                f"node '{req.node_name}' was removed before the job started")
                        # Estimated again on the graph this sweep runs over:
                        # the request's estimate was taken under the lock,
                        # and a request queued behind it -- a node added --
                        # was served before this thread took it again.  A
                        # 2-scalar graph's 0.5 MiB admitted a sweep of 2.15
                        # GiB that way.
                        need_now = _surrogate_training_bytes(self.gm, req.node_name, req)
                        job["estimated_bytes"] = need_now
                        if need_now > MAX_SURROGATE_TRAIN_BYTES:
                            raise _OverBudget(
                                f"the graph changed before the job started: its sweep "
                                f"would now take about {format_bytes(need_now)}, over "
                                f"the {format_bytes(MAX_SURROGATE_TRAIN_BYTES)} accepted "
                                "over the API (MAX_SURROGATE_TRAIN_BYTES); nothing was "
                                "swept or trained")
                        target_init = self.gm._nodes[req.node_name].node.initial_state()
                        n_conditions = _SURROGATE_CONDITIONS
                        steps_per_condition = min(_SURROGATE_STEPS_PER_CONDITION,
                                                  req.n_data_steps)
                        key = jax.random.PRNGKey(42)
                        # Batched initial states: every node's broadcast, the
                        # target node's scalar fields varied.
                        batched = {}
                        for name, spec in self.gm._nodes.items():
                            node_init = spec.node.initial_state()
                            batched[name] = {
                                k: jnp.broadcast_to(v, (n_conditions,) + v.shape)
                                for k, v in node_init.items()
                            }
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

                    arch = MLPDirect(hidden_sizes=tuple(req.hidden_sizes))
                    trainer = SurrogateTrainer(arch, ds)

                    def progress(epoch, metrics):
                        job["epoch"] = epoch
                        job["train_loss"] = float(metrics["train_loss"])
                        job["val_loss"] = float(metrics["val_loss"])
                        if self._shutdown.is_set():
                            raise _TrainingCancelled()

                    result = trainer.train(
                        n_epochs=req.n_epochs,
                        batch_size=req.batch_size,
                        rng_key=jax.random.PRNGKey(42),
                        callback=progress,
                    )
                    job["result"] = result
                    job["status"] = "done"
                except _TrainingCancelled:
                    job["status"] = "cancelled"
                    job["error"] = "the server shut down before the job finished"
                except _OverBudget as exc:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    logger.warning("Surrogate job %s refused: %s", job_id, exc)
                except Exception as exc:
                    job["status"] = "error"
                    job["error"] = str(exc)
                    logger.exception("Surrogate training failed")
                finally:
                    with self._surrogate_lock:
                        self._surrogate_threads.pop(job_id, None)

            with self._surrogate_lock:
                running = [j["id"] for j in self._surrogate_jobs.values()
                           if j["status"] == "running"]
                if running:
                    raise HTTPException(
                        status_code=409,
                        detail=(f"Surrogate job '{running[0]}' is still running; one job "
                                f"runs at a time (GET /surrogate/status/{running[0]})."),
                    )
                finished = [k for k, j in self._surrogate_jobs.items()
                            if j["status"] != "running"]
                for old in finished[:max(0, len(finished) - (MAX_SURROGATE_JOBS_KEPT - 1))]:
                    del self._surrogate_jobs[old]
                self._surrogate_jobs[job_id] = job
                thread = threading.Thread(target=_train_worker, daemon=True,
                                          name=f"maddening-surrogate-{job_id}")
                self._surrogate_threads[job_id] = thread
                thread.start()
            return {"job_id": job_id, "status": "started", "estimated_bytes": need}

        @app.get("/surrogate/status/{job_id}", tags=["surrogate"], response_model=None,
                  openapi_extra=_route_openapi("GET", "/surrogate/status/{job_id}"))
        def surrogate_status(job_id: str) -> dict[str, Any]:
            """A training job's progress.  **Experimental in 0.4.0** (to be
            hardened in 0.5.0)."""
            job = self._surrogate_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"No job '{job_id}'.")
            return {
                "job_id": job["id"],
                "node_name": job["node_name"],
                "status": job["status"],
                "epoch": job["epoch"],
                "n_epochs": job["n_epochs"],
                "train_loss": job["train_loss"],
                "val_loss": job["val_loss"],
                "error": job["error"],
                "estimated_bytes": job.get("estimated_bytes"),
            }

        @app.post("/surrogate/activate/{job_id}", tags=["surrogate"], response_model=None,
                  openapi_extra=_route_openapi("POST", "/surrogate/activate/{job_id}"))
        def surrogate_activate(job_id: str) -> Any:
            """Replace the physics node with the trained surrogate
            (**experimental in 0.4.0**, to be hardened in 0.5.0; the runner
            is stopped first; a 409 while a ``POST /sim/run`` is in
            progress).  The graph is reset, and the streams are sent the
            reset state at step 0.  ``was_running`` says whether a runner
            was running; a refusal after it was stopped says it stays
            stopped."""
            job = self._surrogate_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"No job '{job_id}'.")
            if job["status"] != "done":
                raise HTTPException(status_code=400, detail="Training not complete.")

            node_name = job["node_name"]
            result = job["result"]

            deadline = time.monotonic() + _GRAPH_LOCK_TIMEOUT
            with self._runner_control("activate a surrogate", deadline):
                was_running = self._runner_running()
                self._stop_runner_or_refuse("activate a surrogate")
            try:
                return activate_locked(node_name, result, was_running, deadline)
            except HTTPException as exc:
                return self._after_stopping_the_runner(exc, "activated", was_running)

        def activate_locked(node_name: str, result: Any, was_running: bool,
                            deadline: float) -> dict[str, Any]:
            """``POST /surrogate/activate``, from the graph lock on."""
            with self._graph_transaction("activate a surrogate", write=True,
                                         deadline=deadline):
                if node_name not in self.gm._nodes:
                    raise HTTPException(status_code=404, detail=f"No node '{node_name}'.")
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

                surrogate = result.to_node(
                    name=node_name,
                    timestep=orig_node.delta_t,
                    initial_values=initial_values,
                )

                _refuse_dropped_geometry(
                    self.gm, node_name, self.gm._nodes[node_name].node, surrogate)
                from maddening.surrogates.replace import replace_node
                replace_node(self.gm, node_name, surrogate)
                self.gm.compile()
                self._active_surrogates.add(node_name)
                self._reset_state()

            return {"status": "activated", "node": node_name, "was_running": was_running}

        @app.post("/surrogate/deactivate/{node_name}", tags=["surrogate"], response_model=None,
                  openapi_extra=_route_openapi("POST", "/surrogate/deactivate/{node_name}"))
        def surrogate_deactivate(node_name: str) -> Any:
            """Restore the original physics node (**experimental in 0.4.0**,
            to be hardened in 0.5.0; the runner is stopped first).  The graph is reset, and the streams are sent the reset
            state at step 0; ``was_running`` as for activate.

            The node comes back with the edges it had when the surrogate
            was activated.  An edge added to or from it while the surrogate
            was active is removed with the surrogate and is not put back:
            ``dropped_edges`` lists each one (an empty list when there is
            none), to be added again with ``POST /graph/edges``."""
            if node_name not in self._original_nodes:
                raise HTTPException(
                    status_code=400,
                    detail=f"No original node saved for '{node_name}'.",
                )

            deadline = time.monotonic() + _GRAPH_LOCK_TIMEOUT
            with self._runner_control("deactivate a surrogate", deadline):
                was_running = self._runner_running()
                self._stop_runner_or_refuse("deactivate a surrogate")
            try:
                with self._graph_transaction("deactivate a surrogate", write=True,
                                             deadline=deadline):
                    return {**deactivate_locked(node_name), "was_running": was_running}
            except HTTPException as exc:
                return self._after_stopping_the_runner(exc, "deactivated", was_running)

        def deactivate_locked(node_name: str) -> dict[str, Any]:
            """``POST /surrogate/deactivate``, holding the graph lock."""
            orig_node, orig_edges, orig_ext = self._original_nodes[node_name]
            # The revert puts back the edges recorded when the surrogate
            # was activated: an edge the node was given since then is
            # removed with the surrogate and not added back.  The reply
            # says which (0.4.0's documented behaviour of an experimental
            # route; it used to drop them without a word).
            recorded = {(e.source_node, e.target_node, e.source_field, e.target_field)
                        for e in orig_edges}
            dropped_edges = [
                {"source_node": e.source_node, "target_node": e.target_node,
                 "source_field": e.source_field, "target_field": e.target_field}
                for e in self.gm._edges
                if node_name in (e.source_node, e.target_node)
                and (e.source_node, e.target_node, e.source_field, e.target_field)
                not in recorded]

            # The revert rebuilds a subgraph, and is all-or-nothing by the
            # route's transaction: anything raised here leaves the live
            # graph with the surrogate it had, compiled as it was, and is
            # answered the generic 500 (the reason is in the server's log).
            # The route used to keep its own copy of the graph's structure,
            # recompile after putting it back, and quote the error.
            #
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
            live = self.gm._nodes.get(node_name)
            if live is not None:
                _refuse_dropped_geometry(self.gm, node_name, live.node, orig_node)
            try:
                # As a replacement: the coupling group the surrogate is a
                # member of keeps the name, for the node added back below.
                self.gm._remove_node(node_name, replacing=True)
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

            self._active_surrogates.discard(node_name)
            self._reset_state()
            del self._original_nodes[node_name]

            return {"status": "deactivated", "node": node_name,
                    "dropped_edges": dropped_edges}

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

            The live simulation is left as it was: the graph is put back
            after the profile, and the streams neither show nor count the
            profiler's steps.
            """
            from maddening.core.simulation.profiler import (
                profile_graph, profile_report_to_perfetto,
            )
            n_steps = max(1, min(1000, int(n_steps)))
            n_warmup = max(0, min(50, int(n_warmup)))
            with self._graph_transaction("profile the graph", write=True) as transaction:
                # The profiler resets the graph to its initial state and
                # steps it: the route's transaction is rolled back when it
                # succeeds as well, so the live state, the params, the
                # compile bookkeeping (an edited graph is still waiting for
                # its compile) and the streams are as they were.  The route
                # used to leave the graph at the profiler's last step --
                # 0.09 s into a run that had been at 10 s -- while the
                # streams' clock went on from 10 s.
                try:
                    with self._relay_detached():
                        report = profile_graph(self.gm, n_steps=n_steps, n_warmup=n_warmup)
                except _GRAPH_CONFIGURATION_ERRORS as exc:
                    # Every error a graph that cannot step raises, as
                    # /sim/step and /graph/compile catch them: only a
                    # RuntimeError was, and an edge of mismatched shapes (an
                    # ExceptionGroup from the compile) was a 500.
                    raise HTTPException(status_code=400, detail=_cannot_step_detail(exc))
                transaction.rollback()
            return profile_report_to_perfetto(report)

        @app.post("/sim/profile/jax/start", tags=["sim"], response_model=None)
        def sim_profile_jax_start() -> dict[str, Any]:
            """Begin a JAX-level XLA trace.

            All subsequent ``/sim/step`` and ``/sim/run`` calls (and
            any runner steps) are recorded, for at most
            :data:`MAX_JAX_TRACE_STEPS` steps or
            :data:`MAX_JAX_TRACE_SECONDS` seconds: past either the trace
            stops itself, writes its files, and ``GET
            /sim/profile/jax/status`` says so.  JAX's profiler holds a
            trace's events in memory until it stops, and a trace left
            running grew the server without bound.  POST
            ``/sim/profile/jax/stop`` to end it sooner.  The trace
            directory is returned in the stop response and can be loaded
            via TensorBoard's "Trace Viewer" plugin (which uses a
            Perfetto frontend).
            """
            from maddening.core.simulation.profiler import (
                start_jax_trace, jax_trace_active,
            )
            with self._trace_lock:
                if jax_trace_active():
                    raise HTTPException(
                        status_code=409, detail="A JAX trace is already active.",
                    )
                try:
                    log_dir = start_jax_trace()
                except RuntimeError as exc:
                    raise HTTPException(status_code=400, detail=str(exc))
                self._trace_steps = 0
                self._trace_stopped_by = None
                self._trace_started = time.monotonic()
                timer = threading.Timer(MAX_JAX_TRACE_SECONDS, self._trace_time_is_up)
                timer.daemon = True
                self._trace_timer = timer
                timer.start()
            return {"status": "tracing", "log_dir": log_dir,
                    "max_steps": MAX_JAX_TRACE_STEPS, "max_seconds": MAX_JAX_TRACE_SECONDS}

        @app.post("/sim/profile/jax/stop", tags=["sim"], response_model=None)
        def sim_profile_jax_stop() -> dict[str, Any]:
            """End the active JAX trace and return the log directory.  A
            409 when none is active -- saying, for one that stopped itself
            at its budget, after how many steps and why, and that its
            directory is ``last_trace_dir`` in ``GET
            /sim/profile/jax/status`` (a 4xx detail names no server path)."""
            from maddening.core.simulation.profiler import jax_trace_active
            with self._trace_lock:
                self._stop_trace_if_out_of_time()
                if not jax_trace_active():
                    detail = "No JAX trace is active."
                    if self._trace_stopped_by not in (None, "a request"):
                        detail += (f" The last one stopped itself after "
                                   f"{self._trace_steps} steps ({self._trace_stopped_by}); "
                                   "its directory is last_trace_dir in GET "
                                   "/sim/profile/jax/status.")
                    raise HTTPException(status_code=409, detail=detail)
                log_dir = self._stop_trace("a request")
            return {"status": "stopped", "log_dir": log_dir, "steps": self._trace_steps}

        @app.get("/sim/profile/jax/status", tags=["sim"], response_model=None)
        def sim_profile_jax_status() -> dict[str, Any]:
            from maddening.core.simulation.profiler import jax_trace_active
            with self._trace_lock:
                self._stop_trace_if_out_of_time()
            return {
                "active": jax_trace_active(),
                "last_trace_dir": getattr(self, "_last_jax_trace_dir", None),
                "steps": self._trace_steps,
                "max_steps": MAX_JAX_TRACE_STEPS,
                "max_seconds": MAX_JAX_TRACE_SECONDS,
                "stopped_by": self._trace_stopped_by,
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
            """Stream state snapshots as JSON at ~30 Hz.  **Experimental in
            0.4.0** (to be hardened in 0.5.0).

            Client may send JSON messages to configure the stream:

            * ``{"type": "subscribe", "fields": {"node": ["f1", "f2"]}}``
              — only include the listed node/field pairs in subsequent
              snapshots.  Send ``{"type": "subscribe", "fields": null}``
              to reset to full state.
            * ``{"type": "config", "fps": 15}`` — change poll rate.

            Each frame is encoded in a worker thread, never on the event
            loop, and a frame of the whole state is encoded once and sent
            to every client of it.  At most :data:`MAX_STREAM_CONNECTIONS`
            streams are open at once (1013 past it).

            Authentication is the same rule as every HTTP route; see
            :meth:`SimulationServer._authorise_ws`.
            """
            authorised, subprotocol = await self._authorise_ws(websocket)
            if not authorised:
                return
            if not self._admit_stream():
                await self._refuse_stream(websocket, "/ws/state", subprotocol)
                return
            try:
                await websocket.accept(subprotocol=subprotocol)
                logger.info("WebSocket client connected to /ws/state")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._ensure_relay_attached_locked)

                sub_fields: list[Optional[dict]] = [None]   # {node: [fields]} or None
                target_fps = [30.0]
                disconnected = asyncio.Event()

                def _on_message(msg: dict) -> None:
                    if msg.get("type") == "subscribe":
                        fields = msg.get("fields")
                        if fields is None or isinstance(fields, dict):
                            sub_fields[0] = fields
                    elif msg.get("type") == "config":
                        if "fps" in msg:
                            target_fps[0] = max(1, min(120, msg["fps"]))

                receiver = asyncio.create_task(
                    _receive_until_disconnect(websocket, _on_message, disconnected))

                last = (None, None)   # (relay sequence, subscription) last sent
                try:
                    while not disconnected.is_set():
                        seq, sim_time, snapshot = self.relay.latest_frame()
                        filt = sub_fields[0]
                        if snapshot is not None and (seq, id(filt)) != last:
                            last = (seq, id(filt))
                            text = await self._shared_state_frame(seq, sim_time,
                                                                  snapshot, filt)
                            await websocket.send_text(text)
                        await _sleep_unless_disconnected(disconnected, 1.0 / target_fps[0])
                    logger.info("WebSocket client disconnected from /ws/state")
                except Exception as exc:  # noqa: BLE001 - classified below
                    if _client_left(websocket, exc, disconnected):
                        logger.info("WebSocket client disconnected from /ws/state")
                    else:
                        logger.exception("WebSocket error on /ws/state")
                finally:
                    receiver.cancel()
            finally:
                self._release_stream()

        @app.websocket("/ws/state/binary")
        async def ws_state_binary(websocket: WebSocket) -> None:
            """Stream state snapshots as binary at ~60 Hz.  **Experimental in
            0.4.0** (to be hardened in 0.5.0).

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

            Every frame is laid out by the last schema sent.  When the
            graph's layout changes -- a node added, removed or replaced, a
            compile -- the server sends a new schema before the next frame;
            the cached encoder used to outlive a replaced node, and frames
            in the old layout were resized to fit, dropping values or
            padding phantom zeros in a frame of the advertised length.

            Authentication is the same rule as every HTTP route; see
            :meth:`SimulationServer._authorise_ws`.
            """
            authorised, subprotocol = await self._authorise_ws(websocket)
            if not authorised:
                return
            if not self._admit_stream():
                await self._refuse_stream(websocket, "/ws/state/binary", subprotocol)
                return
            try:
                await websocket.accept(subprotocol=subprotocol)
                logger.info("WebSocket client connected to /ws/state/binary")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._ensure_relay_attached_locked)

                from maddening.api.binary_encoder import (  # noqa: PLC0415
                    BinaryStateEncoder, VALID_COMPRESSIONS,
                )

                # Mutable state shared with receiver task
                target_fps = [60.0]
                encoder, generation, layout = await loop.run_in_executor(
                    None, self._binary_encoder_locked)
                current_compression = ["none"]
                current_fields: list[dict | None] = [None]
                schema_dirty = asyncio.Event()
                disconnected = asyncio.Event()

                def build(state: dict):
                    try:
                        return BinaryStateEncoder(state, fields=current_fields[0],
                                                  compression=current_compression[0])
                    except ImportError:
                        # zstandard not installed — fall back
                        current_compression[0] = "none"
                        return BinaryStateEncoder(state, fields=current_fields[0],
                                                  compression="none")

                await websocket.send_json(encoder.schema())

                def _on_message(msg: dict) -> None:
                    if msg.get("type") == "subscribe":
                        if "fields" in msg and (msg["fields"] is None
                                                or isinstance(msg["fields"], dict)):
                            current_fields[0] = msg["fields"]
                        if "compression" in msg:
                            comp = msg["compression"]
                            if comp in VALID_COMPRESSIONS:
                                current_compression[0] = comp
                        schema_dirty.set()
                    elif msg.get("type") == "config":
                        if "fps" in msg:
                            target_fps[0] = max(1, min(120, msg["fps"]))

                receiver = asyncio.create_task(
                    _receive_until_disconnect(websocket, _on_message, disconnected))

                last_seq = None
                try:
                    while not disconnected.is_set():
                        seq, sim_time, snapshot = self.relay.latest_frame()
                        # A new schema when the client changed its
                        # subscription or the graph's layout changed: built
                        # from the snapshot about to be encoded, or from the
                        # graph when there is none yet.
                        if schema_dirty.is_set() or generation != self._layout_generation:
                            resubscribed = schema_dirty.is_set()
                            schema_dirty.clear()
                            generation = self._layout_generation
                            source = snapshot if snapshot is not None else \
                                await loop.run_in_executor(None, self._user_state_locked)
                            # A compile that left the layout as it was (the
                            # recompile after a reset or a parameter write)
                            # sends no schema; a subscription always does.
                            if resubscribed or _state_layout(source) != layout:
                                encoder, layout = build(source), _state_layout(source)
                                await websocket.send_json(encoder.schema())
                        if snapshot is not None and seq != last_seq:
                            last_seq = seq
                            # A snapshot of another layout than the schema's
                            # (a node added, removed or resized since): its
                            # own schema first.
                            if _state_layout(snapshot) != layout:
                                encoder, layout = build(snapshot), _state_layout(snapshot)
                                await websocket.send_json(encoder.schema())
                            try:
                                frame = await loop.run_in_executor(
                                    None, encoder.encode, sim_time, snapshot)
                            except (ValueError, KeyError):
                                # The snapshot is laid out otherwise than the
                                # schema the client holds: send its own.
                                encoder, layout = build(snapshot), _state_layout(snapshot)
                                await websocket.send_json(encoder.schema())
                                frame = await loop.run_in_executor(
                                    None, encoder.encode, sim_time, snapshot)
                            await websocket.send_bytes(frame)
                        await _sleep_unless_disconnected(disconnected, 1.0 / target_fps[0])
                    logger.info("WebSocket client disconnected from /ws/state/binary")
                except Exception as exc:  # noqa: BLE001 - classified below
                    if _client_left(websocket, exc, disconnected):
                        logger.info("WebSocket client disconnected from /ws/state/binary")
                    else:
                        logger.exception("WebSocket error on /ws/state/binary")
                finally:
                    receiver.cancel()
            finally:
                self._release_stream()

        @app.websocket("/ws/render")
        async def ws_render(websocket: WebSocket) -> None:
            """Stream server-side rendered frames as compressed images.
            **Experimental in 0.4.0** (to be hardened in 0.5.0).

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
            if not self._admit_stream():
                await self._refuse_stream(websocket, "/ws/render", subprotocol)
                return
            try:
                await websocket.accept(subprotocol=subprotocol)
                logger.info("WebSocket client connected to /ws/render")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._ensure_relay_attached_locked)

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
                disconnected = asyncio.Event()

                def _on_message(msg: dict) -> None:
                    if msg.get("type") == "config":
                        if "format" in msg:
                            renderer.set_format(msg["format"], msg.get("quality"))
                        if "fps" in msg:
                            target_fps[0] = max(1, min(60, msg["fps"]))
                        config_changed.set()
                    elif msg.get("type") == "reset":
                        renderer.reset()

                receiver = asyncio.create_task(
                    _receive_until_disconnect(websocket, _on_message, disconnected))

                last_seq = None
                try:
                    while not disconnected.is_set():
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

                        seq, sim_time, snapshot = self.relay.latest_frame()
                        if snapshot is not None and seq != last_seq:
                            last_seq = seq
                            frame = await loop.run_in_executor(
                                None, renderer.render, sim_time, snapshot,
                            )
                            await websocket.send_bytes(frame)

                        await _sleep_unless_disconnected(disconnected, 1.0 / target_fps[0])
                    logger.info("WebSocket client disconnected from /ws/render")
                except Exception as exc:  # noqa: BLE001 - classified below
                    # A client that closed its viewer is the ordinary end of a
                    # render stream; it used to log a traceback every time.
                    if _client_left(websocket, exc, disconnected):
                        logger.info("WebSocket client disconnected from /ws/render")
                    else:
                        logger.exception("WebSocket error on /ws/render")
                finally:
                    receiver.cancel()
            finally:
                self._release_stream()

        return app
