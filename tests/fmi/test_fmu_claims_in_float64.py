"""The FMU export of an x64 graph: values, time and refusals at float64.

``docs/validation/sysid_fmu_claims.yaml`` states the FMU's value claims for
"the variable's own type", and a graph built under ``jax_enable_x64``
carries float64 state and parameters, so its FMU variables are ``Float64``.
The rows' own tests export float32 graphs; these hold the same claims for

* an x64 spring graph (every variable float64): the variables' type and
  bounds, their lexical form, the advertised open bound, the bridge
  stepping as the graph does at the time it reports, the communication
  point tolerance, the value checks in ``Float64``, type-addressed access,
  ``set_params`` at float64, a subnormal below a zero ``min``, and float64
  values crossing the wire bit for bit;
* a float32 node in that x64 graph (``mixed_dtype``): its variables stay
  ``Float32`` and are still checked as float32 -- ``3.5e38`` and ``1e-50``
  refused, a ``Float64`` call refused, a float32 subnormal below a zero
  ``min`` refused -- and the bridge still steps as the graph does.

x64 is process-global, so every test turns it on for its own duration and
builds its graph inside.
"""

from __future__ import annotations

import contextlib
import socket
import time
from xml.etree import ElementTree

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.fmi import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, recv_message, send_message, values_of
from maddening.nodes.spring import SpringDamperNode

DT = 0.01
F64 = np.float64
SUB64 = float(np.nextafter(F64(0.0), F64(1.0)))


@contextlib.contextmanager
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


class _Gain32(SimulationNode):
    """A float32 node ``y <- g * u + b`` that keeps float32 under x64."""

    def __init__(self, name):
        super().__init__(name, DT, g=jnp.float32(2.0), b=jnp.float32(0.5))

    def initial_state(self):
        return {"y": jnp.float32(0.0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32, default=jnp.float32(0.0))}

    def param_specs(self):
        return {**super().param_specs(),
                "g": ParamSpec(bounds=(1.0, None), transform="log"),
                "b": ParamSpec(bounds=(0.0, None))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = jnp.asarray(boundary_inputs.get("u", jnp.float32(0.0)), jnp.float32)
        return {"y": (p["g"] * u + p["b"]).astype(jnp.float32)}


def _graph(*, mixed=False):
    """A spring with an external anchor (float64 under x64), and with *mixed*
    a float32 node reading an external input of its own."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=2.0, mass=1.0,
                                 rest_length=1.0, initial_position=0.5))
    gm.add_external_input("s", "anchor_position")
    if mixed:
        gm.add_node(_Gain32("q"))
        gm.add_external_input("q", "u")
    gm.compile()
    return gm


def _bridge(gm, md):
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
        input_resolver=gm._resolve_external_inputs))
    return FmuTcpBridge(sidecar, md, master_dt=gm.timestep), sidecar


def _vars(md):
    return {v.name: v for v in md.variables}


def _served(mixed=False):
    gm = _graph(mixed=mixed)
    md = build_model_description(gm, model_name="m")
    bridge, sidecar = _bridge(gm, md)
    return gm, md, bridge, sidecar


def _set(bridge, var, value, fmi_type=None):
    req = {"op": "set", "vr": [var.value_reference], "values": [value]}
    if fmi_type:
        req["type"] = fmi_type
    return bridge.handle(req)


def _get(bridge, var, fmi_type=None):
    req = {"op": "get", "vr": [var.value_reference]}
    if fmi_type:
        req["type"] = fmi_type
    return bridge.handle(req)


def _value(bridge, var):
    return float(values_of(_get(bridge, var))[0])


# ---------------------------------------------------------------------------
# The model description
# ---------------------------------------------------------------------------
def test_an_x64_graph_exports_float64_variables_with_float64_bounds():
    """FMU-004 under x64: outputs, inputs and parameters carry the graph's own
    dtype -- float64 -- and parameters their ``ParamSpec`` bounds."""
    with _x64():
        gm, md, _bridge_, _sc = _served()
        v = _vars(md)
        for name in ("s.position", "s.velocity", "s.anchor_position", "s.params.stiffness",
                     "s.params.damping"):
            assert v[name].dtype == "float64", (name, v[name].dtype)
        assert v["s.params.damping"].causality == "parameter"
        assert v["s.params.damping"].variability == "tunable"
        assert float(v["s.params.damping"].min) == 0.0
        assert 'Float64 name="s.position"' in md.to_xml()


def test_a_float32_node_in_an_x64_graph_keeps_float32_variables():
    """FMU-004 and FMU-005 with mixed dtypes: the float32 node's variables are
    ``Float32``, the spring's ``Float64``, and every start is a literal of its
    own type (the float32 start round-trips through float32, not float64)."""
    with _x64():
        gm, md, _bridge_, _sc = _served(mixed=True)
        v = _vars(md)
        assert v["q.y"].dtype == "float32" and v["q.params.g"].dtype == "float32"
        assert v["s.position"].dtype == "float64"
        xml = md.to_xml()
        assert 'Float32 name="q.y"' in xml and 'Float64 name="s.position"' in xml
        for name in ("q.params.g", "q.params.b"):
            start = v[name].start
            assert np.float32(float(start)) == np.float32(gm.params["nodes"]["q"][name[9:]])


def test_every_float64_start_and_bound_round_trips_through_its_lexical_form():
    """FMU-005 at float64: a start one float64 ulp off a float32 value is
    written so that reading it back gives the same float64, bit for bit."""
    with _x64():
        k = float(np.nextafter(F64(30.0), F64(np.inf)))       # not a float32
        gm = GraphManager()
        gm.add_node(SpringDamperNode("s", DT, stiffness=k, damping=0.1 + 0.2, mass=1.0))
        gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.1 + 0.2, None)))
        gm.compile()
        md = build_model_description(gm, model_name="m")
        root = ElementTree.fromstring(md.to_xml())
        attrs = {e.get("name"): e.attrib for e in root.iter("Float64")}
        assert F64(float(attrs["s.params.stiffness"]["start"])).tobytes() == F64(k).tobytes()
        assert F64(float(attrs["s.params.damping"]["start"])).tobytes() == F64(0.1 + 0.2).tobytes()
        assert F64(float(attrs["s.params.damping"]["min"])).tobytes() == F64(0.1 + 0.2).tobytes()


def test_an_open_bound_is_advertised_as_the_nearest_float64_inside_it():
    """FMU-016 at float64: a ``log`` bound at 0 is advertised as float64's
    smallest normal, and one at 1.0 as the next float64 above it."""
    with _x64():
        gm = _graph()
        v = _vars(build_model_description(gm, model_name="m"))
        assert float(v["s.params.stiffness"].min) == float(np.finfo(np.float64).tiny)
        gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(1.0, None), transform="log"))
        gm.compile()
        v = _vars(build_model_description(gm, model_name="m"))
        assert float(v["s.params.stiffness"].min) == float(np.nextafter(F64(1.0), F64(2.0)))


def test_a_float32_parameter_in_an_x64_graph_advertises_the_float32_envelope():
    """FMU-016 with mixed dtypes: the float32 leaf's open bound at 1.0 is the
    next float32 above it, which is what its sidecar check accepts."""
    with _x64():
        gm, md, bridge, _sc = _served(mixed=True)
        g = _vars(md)["q.params.g"]
        assert float(g.min) == float(np.nextafter(np.float32(1.0), np.float32(2.0)))
        assert _set(bridge, g, float(g.min))["ok"]
        assert not _set(bridge, g, 1.0)["ok"]


# ---------------------------------------------------------------------------
# Stepping and time
# ---------------------------------------------------------------------------
def _stepped_against_the_graph(mixed):
    gm, md, bridge, _sc = _served(mixed=mixed)
    v = _vars(md)
    assert bridge.handle({"op": "set", "vr": [v["s.params.stiffness"].value_reference,
                                              v["s.anchor_position"].value_reference],
                          "values": [45.0, 0.25]})["ok"]
    if mixed:
        assert _set(bridge, v["q.u"], 0.75)["ok"]
    for i in range(4):
        reply = bridge.handle({"op": "step", "t": i * 3 * DT, "dt": 3 * DT})
        assert reply["ok"] and reply["t"] == pytest.approx((i + 1) * 3 * DT, rel=1e-12)
    ref = _graph(mixed=mixed)
    p = jax.tree.map(lambda x: x, ref.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(45.0, p["nodes"]["s"]["stiffness"].dtype)
    ext = {"s": {"anchor_position": 0.25}}
    if mixed:
        ext["q"] = {"u": 0.75}
    for _ in range(12):
        ref.step(ext, params=p)
    want = float(np.asarray(ref.get_node_state("s")["position"], np.float64))
    got = _value(bridge, v["s.position"])
    assert got == want, (got, want)          # one compiled step, one arithmetic
    assert _value(bridge, v["time"]) == pytest.approx(12 * DT, rel=1e-12)
    if mixed:
        assert _value(bridge, v["q.y"]) == float(np.asarray(ref.get_node_state("q")["y"]))


def test_an_x64_graph_served_by_the_bridge_is_the_graph():
    """FMU-002 and FMU-020 under x64: a communication step of ``3 h`` runs
    three graph steps, and with a set parameter and a driven input the FMU
    is the graph at the time it reports -- to the bit, at float64."""
    with _x64():
        _stepped_against_the_graph(mixed=False)


def test_a_float32_node_in_an_x64_graph_served_by_the_bridge_is_the_graph():
    with _x64():
        _stepped_against_the_graph(mixed=True)


def test_communication_points_of_an_x64_graph_are_held_to_the_same_tolerance():
    """FMU-023 under x64: a point inside a millionth of a master step is
    adopted, one outside it refused with nothing advanced, and an importer's
    rounded points are accepted over a long run."""
    with _x64():
        gm, md, bridge, _sc = _served()
        v = _vars(md)
        assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
        nudged = DT + 0.5e-6 * DT
        assert bridge.handle({"op": "step", "t": nudged, "dt": DT}) == {"ok": True, "t": nudged + DT}
        before = _value(bridge, v["s.position"])
        refused = bridge.handle({"op": "step", "t": nudged + DT + 2e-6 * DT, "dt": DT})
        assert refused["ok"] is False and _value(bridge, v["s.position"]) == before
        t = nudged + DT
        for k in range(1, 400):
            point = t + k * DT - DT          # start + k*h, as FMPy computes it
            reply = bridge.handle({"op": "step", "t": point, "dt": DT})
            assert reply["ok"], (k, point, reply)


# ---------------------------------------------------------------------------
# Values, types and refusals
# ---------------------------------------------------------------------------
def test_a_float64_variable_is_checked_in_its_own_type():
    """FMU-024 under x64: a ``Float64`` input takes ``1e308`` (which a float32
    one refuses) and keeps a float64 subnormal's sign and magnitude; a
    non-finite value is refused with nothing written."""
    with _x64():
        gm, md, bridge, _sc = _served()
        anchor = _vars(md)["s.anchor_position"]
        assert _set(bridge, anchor, 1e308, "Float64")["ok"]
        assert _value(bridge, anchor) == 1e308
        assert _set(bridge, anchor, -SUB64, "Float64")["ok"]
        assert _value(bridge, anchor) == -SUB64
        refused = _set(bridge, anchor, float("inf"), "Float64")
        assert not refused["ok"] and _value(bridge, anchor) == -SUB64


def test_a_float64_variable_is_addressed_only_as_float64():
    """FMU-026 under x64: a ``Float32`` call on a ``Float64`` variable is refused,
    set and get, with nothing written."""
    with _x64():
        gm, md, bridge, _sc = _served()
        anchor = _vars(md)["s.anchor_position"]
        assert _set(bridge, anchor, 0.125, "Float64")["ok"]
        assert not _set(bridge, anchor, 0.5, "Float32")["ok"]
        assert not _get(bridge, anchor, "Float32")["ok"]
        assert _value(bridge, anchor) == 0.125


def test_a_float32_variable_of_an_x64_graph_is_still_checked_as_float32():
    """FMU-024 and FMU-026 with mixed dtypes: the float32 node's input refuses
    ``3.5e38`` and ``1e-50`` as float32 does without x64, keeps a float32
    subnormal, and is addressed only as ``Float32``."""
    with _x64():
        gm, md, bridge, _sc = _served(mixed=True)
        u = _vars(md)["q.u"]
        assert _set(bridge, u, 3.4e38, "Float32")["ok"]
        for bad in (3.5e38, 1e-50, -1e-50):
            reply = _set(bridge, u, bad, "Float32")
            assert not reply["ok"] and "does not fit its type float32" in reply["error"], reply
        assert _set(bridge, u, 1e-40, "Float32")["ok"] and 0.0 < _value(bridge, u) < 1.2e-38
        assert not _set(bridge, u, 0.5, "Float64")["ok"]
        assert 0.0 < _value(bridge, u) < 1.2e-38           # nothing written


def test_set_params_holds_a_float64_leaf_to_float64():
    """FMU-032 under x64: a float64 leaf takes a value only float64 holds and
    the next step uses it without a recompile; a non-finite one is refused."""
    with _x64():
        gm, md, bridge, sc = _served()
        sc.set_params({"s.params.stiffness": 1e300})
        assert float(sc.get_params()["s.params.stiffness"]) == 1e300
        with pytest.raises(ValueError):
            sc.set_params({"s.params.stiffness": float("nan")})
        sc.set_params({"s.params.stiffness": 45.0})
        state = sc.step({"s": {"anchor_position": 0.0}})
        ref = _graph()
        p = jax.tree.map(lambda x: x, ref.params)
        p["nodes"]["s"]["stiffness"] = jnp.asarray(45.0, jnp.float64)
        ref.step({"s": {"anchor_position": 0.0}}, params=p)
        assert float(state["s"]["position"]) == float(ref.get_node_state("s")["position"])


def test_set_params_holds_a_float32_leaf_of_an_x64_graph_to_float32():
    """FMU-032 with mixed dtypes: ``1e39`` does not fit the float32 leaf and is
    refused atomically, even in an x64 process; the largest float32 fits."""
    with _x64():
        gm, md, bridge, sc = _served(mixed=True)
        with pytest.raises(ValueError, match="q.params.b"):
            sc.set_params({"q.params.g": 3.0, "q.params.b": 1e39})
        assert float(sc.get_params()["q.params.g"]) == 2.0      # nothing written
        big = float(np.finfo(np.float32).max)
        sc.set_params({"q.params.b": big})
        assert float(sc.get_params()["q.params.b"]) == big


def test_a_float64_parameter_cannot_be_set_below_its_min_by_a_float64_subnormal():
    """FMU-039 under x64: ``damping`` advertises ``min="0.0"``; ``-5e-324`` is below it."""
    with _x64():
        gm, md, bridge, _sc = _served()
        damping = _vars(md)["s.params.damping"]
        reply = _set(bridge, damping, -SUB64, "Float64")
        assert not reply["ok"], reply
        assert _value(bridge, damping) >= 0.0


def test_a_float32_parameter_of_an_x64_graph_cannot_be_set_below_its_min_by_a_float32_subnormal():
    """FMU-039 with mixed dtypes: the float32 ``b`` (``min="0.0"``) refuses ``-1e-40``."""
    with _x64():
        gm, md, bridge, _sc = _served(mixed=True)
        b = _vars(md)["q.params.b"]
        reply = _set(bridge, b, -1e-40, "Float32")
        assert not reply["ok"], reply
        assert _value(bridge, b) >= 0.0


def _connect(bridge, retries=40):
    host, port = bridge.endpoint.split(":")
    for _ in range(retries):
        conn = socket.create_connection((host, int(port)), timeout=30)
        send_message(conn, {"op": "hello"})
        reply = recv_message(conn)
        if reply["ok"]:
            return conn
        conn.close()
        time.sleep(0.05)
    raise AssertionError("bridge stayed busy")


def test_an_x64_graphs_values_cross_the_wire_bit_for_bit():
    """FMU-043 under x64: "the wire carries every value as a float64" -- values
    only float64 holds, and the graph's own float64 state, cross a real
    socket and come back bit for bit."""
    with _x64():
        gm, md, bridge, _sc = _served()
        v = _vars(md)
        anchor, pos = v["s.anchor_position"], v["s.position"]
        values = [0.1, 1.0 / 3.0, SUB64, 1.7976931348623157e308, -2.0 ** -1074]
        with bridge:
            conn = _connect(bridge)
            with conn:
                for x in values:
                    send_message(conn, {"op": "set", "vr": [anchor.value_reference],
                                        "values": [x]})
                    assert recv_message(conn) == {"ok": True}
                    send_message(conn, {"op": "get", "vr": [anchor.value_reference]})
                    got = values_of(recv_message(conn))[0]
                    assert F64(got).tobytes() == F64(x).tobytes(), (x, got)
                send_message(conn, {"op": "set", "vr": [anchor.value_reference], "values": [0.1]})
                recv_message(conn)
                send_message(conn, {"op": "step", "t": 0.0, "dt": DT})
                assert recv_message(conn)["ok"]
                send_message(conn, {"op": "get", "vr": [pos.value_reference]})
                got = values_of(recv_message(conn))[0]
        ref = _graph()
        ref.step({"s": {"anchor_position": 0.1}})
        want = np.asarray(ref.get_node_state("s")["position"], np.float64)
        assert F64(got).tobytes() == want.tobytes(), (got, want)
