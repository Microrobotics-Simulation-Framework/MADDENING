"""Shared scaffolding for the two stateful property machines.

``test_stateful_api.py`` drives the REST server and ``test_stateful_bridge.py``
drives the FMU sidecar bridge; both need the same three things:

* a *small* node vocabulary -- two or three scalar nodes -- because every
  distinct graph structure costs a JAX trace, and a stateful machine builds
  hundreds of graphs per run;
* a value vocabulary that stays inside float32 and inside the nodes'
  declared :class:`~maddening.core.params.ParamSpec` bounds, so that a
  "valid" rule really is valid and a failure means the server is wrong;
* a NaN-aware structural comparison, because a state dict decoded from JSON
  compares ``nan != nan`` and both sides of a model check see the *same*
  NaN when the physics produces one.

The bridge's graph is compiled **once per process** and shared by every
example: a fresh :class:`~maddening.fmi.sidecar.FmuSidecar` built on the
same ``gm._compiled_step`` is an independent instance with an independent
state, so an example gets a clean bridge without paying for a compile.
"""

from __future__ import annotations

import math
import socket
import time
from typing import Any

import numpy as np
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.fmi import MODEL_IDENTIFIER, build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, recv_message, send_message
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

DT = 0.01
"""The one timestep every node in these machines runs at.

Deliberately fixed: a mixed-rate graph is a different scheduler path with
its own ``_meta`` counters, and varying it would multiply the number of
distinct compiled step functions without testing anything the REST or
bridge *surface* is responsible for.
"""

REGISTRY: dict[str, type] = {
    "BallNode": BallNode,
    "SpringDamperNode": SpringDamperNode,
    "TableNode": TableNode,
}

#: Per node type: the constructor parameters the machines may set, each with
#: the closed interval that is valid for *both* the constructor and the
#: node's declared ``ParamSpec`` bounds.  The upper ends keep the explicit
#: integrators stable at ``DT`` (``2*sqrt(mass/stiffness) >> DT``) so that a
#: legitimate sequence never produces a non-finite state.
VALID_PARAM_RANGE: dict[str, dict[str, tuple[float, float]]] = {
    "TableNode": {"position": (-5.0, 5.0)},
    "SpringDamperNode": {
        "stiffness": (0.5, 100.0),
        "damping": (0.0, 5.0),
        "mass": (0.5, 10.0),
        "rest_length": (0.0, 3.0),
        "initial_position": (-2.0, 2.0),
        "initial_velocity": (-2.0, 2.0),
    },
    "BallNode": {
        "initial_position": (0.0, 5.0),
        "initial_velocity": (-2.0, 2.0),
        "elasticity": (0.0, 1.0),
        "gravity": (-20.0, 0.0),
    },
}

#: Per node type: a parameter with a finite declared bound and a value
#: outside it.  ``PUT /graph/params`` must refuse each of these with a 400.
OUT_OF_BOUNDS: dict[str, tuple[str, float]] = {
    "TableNode": ("position", float("nan")),   # only bound TableNode has: finiteness
    "SpringDamperNode": ("stiffness", -1.0),   # ParamSpec bounds (0.0, None)
    "BallNode": ("elasticity", 2.0),           # ParamSpec bounds (0.0, 1.0)
}

#: Per node type: the fields ``GET /graph/state/<node>`` reports.
STATE_FIELDS: dict[str, tuple[str, ...]] = {
    "TableNode": ("position",),
    "SpringDamperNode": ("position", "velocity"),
    "BallNode": ("position", "velocity"),
}

#: Per node type: the boundary inputs an incoming edge may target.
BOUNDARY_INPUTS: dict[str, tuple[str, ...]] = {
    "TableNode": (),
    "SpringDamperNode": ("anchor_position",),
    "BallNode": ("table_position",),
}


def float32_in(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Finite float32-exact values in ``[lo, hi]``.

    ``width=32`` matters: every node state and parameter leaf is float32, so
    a float64 draw would be rounded on the way in and the model would have
    to re-implement the rounding to predict what comes back out.
    """
    return st.floats(min_value=lo, max_value=hi, allow_nan=False,
                     allow_infinity=False, allow_subnormal=False, width=32)


def constructor_params(type_name: str) -> st.SearchStrategy[dict[str, float]]:
    """A (possibly empty) subset of ``type_name``'s valid constructor params."""
    ranges = VALID_PARAM_RANGE[type_name]
    return st.dictionaries(
        keys=st.sampled_from(sorted(ranges)),
        values=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False,
                         allow_infinity=False, allow_subnormal=False, width=32),
        max_size=len(ranges),
    ).map(lambda d: {k: _clamp(v, *ranges[k]) for k, v in d.items()})


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(np.float32(min(max(value, lo), hi)))


def canonical(value: Any) -> Any:
    """``value`` with every NaN replaced by a sentinel, for ``==``.

    Both sides of a model comparison run the same float32 arithmetic, so an
    exact comparison is the right one -- except that ``nan != nan`` would
    turn an *agreement* about a non-finite state into a spurious failure.
    """
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return "<nan>"
    return value


def jsonify_state(state: dict) -> dict:
    """A GraphManager state dict in the shape ``GET /graph/state`` returns."""
    from maddening.api.server import _jax_to_python

    return canonical(_jax_to_python(state))


# ---------------------------------------------------------------------------
# FMU bridge scaffolding
# ---------------------------------------------------------------------------

_BRIDGE_GRAPH: tuple[GraphManager, Any] | None = None


def bridge_graph():
    """The (graph, model description) pair every bridge example shares.

    Compiled once per process.  Two scalar nodes, one external input and
    one parameterised node give twelve value references -- an independent
    variable, an input, three outputs and seven parameters -- which is the
    whole causality vocabulary the bridge distinguishes, at the price of a
    single trace.
    """
    global _BRIDGE_GRAPH
    if _BRIDGE_GRAPH is None:
        gm = GraphManager()
        gm.add_node(TableNode(name="table", timestep=DT))
        gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=30.0,
                                     damping=2.0, initial_position=0.5))
        gm.add_external_input("spring", "anchor_position")
        gm.compile()
        md = build_model_description(gm, model_name="Plant",
                                     model_identifier=MODEL_IDENTIFIER)
        _BRIDGE_GRAPH = (gm, md)
    return _BRIDGE_GRAPH


def new_bridge(**kwargs) -> tuple[Any, FmuTcpBridge]:
    """A fresh bridge over a fresh sidecar on the shared compiled step."""
    gm, md = bridge_graph()
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token,
        step_fn=gm._compiled_step,                                # noqa: SLF001
        initial_state={n: dict(f) for n, f in gm._state.items()},  # noqa: SLF001
        params=gm.params,
        param_specs=gm.param_specs(),
    ))
    return md, FmuTcpBridge(sidecar, md, master_dt=DT, **kwargs)


def vr_of(md, name: str) -> int:
    return next(v.value_reference for v in md.variables if v.name == name)


def connect(bridge: FmuTcpBridge, *, protocol: int | None = 2, binary: bool = True,
            retries: int = 60):
    """A client socket past its hello.

    The bridge serves one instance at a time and releases the lock only when
    the previous connection's EOF has been served, so a reconnect retries
    briefly rather than failing.
    """
    host, port = bridge.endpoint.split(":")
    if protocol is None:
        hello: dict[str, Any] = {"op": "hello"}
    else:
        hello = {"op": "hello", "protocol": protocol, "binary": binary}
    for _ in range(retries):
        conn = socket.create_connection((host, int(port)), timeout=30)
        send_message(conn, hello)
        reply = recv_message(conn)
        if reply is not None and reply.get("ok"):
            return conn, reply
        conn.close()
        assert reply is not None and "already serves" in reply.get("error", ""), reply
        time.sleep(0.02)
    raise AssertionError("bridge stayed busy")


def f64_bytes(values) -> bytes:
    return np.ascontiguousarray(values, dtype="<f8").tobytes()
