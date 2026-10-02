"""The FMI ``min`` / ``max`` attributes are the *settable* envelope: for a
strict (log / logit) bound the description advertises the next
representable float inside, for an inclusive one the bound itself.

Originally written from the independent audit of 2026-09-16 (round 1; report and
reproducers under ``benchmarks/results/audit1/``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.fmi.model_description import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

F32 = np.finfo(np.float32)


@pytest.fixture
def gm():
    g = GraphManager()
    # A table edge, so the ball's step reads ``elasticity``: the FMU exports
    # only parameters its step reads, and without a table ``elasticity`` is
    # a knob that does nothing.
    g.add_node(TableNode(name="table", timestep=1e-2))
    g.add_node(BallNode(name="ball", timestep=1e-2, initial_position=1.0, elasticity=0.7))
    g.add_node(SpringDamperNode(name="s", timestep=1e-2, stiffness=30.0, damping=2.0))
    g.add_edge("table", "ball", "position", "table_position")
    g.compile()
    return g


def _sidecar(gm, md):
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
    ))


def test_log_leaf_with_zero_bound_advertises_smallest_normal(gm):
    md = build_model_description(gm, model_name="m", include_evolving=True)
    var = next(v for v in md.variables if v.name == "s.params.stiffness")
    # not 0.0 (rejected as "below bound"), not nextafter(0) (a subnormal)
    assert var.min == float(F32.tiny) and var.min > 0.0
    assert var.max is None
    _sidecar(gm, md).set_params({"s.params.stiffness": var.min})    # accepted
    el = next(v for v in ET.fromstring(md.to_xml()).find("ModelVariables")
              if v.get("name") == "s.params.stiffness")
    assert float(el.get("min")) == var.min


def test_log_leaf_with_nonzero_bound_advertises_next_float(gm):
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(8.0, None), transform="log"))
    md = build_model_description(gm, model_name="m", include_evolving=True)
    var = next(v for v in md.variables if v.name == "s.params.stiffness")
    assert var.min == float(np.nextafter(np.float32(8.0), np.float32(np.inf)))
    assert var.min > 8.0
    sc = _sidecar(gm, md)
    sc.set_params({"s.params.stiffness": var.min})
    with pytest.raises(ValueError, match="below bound"):
        sc.set_params({"s.params.stiffness": 8.0})


def test_logit_leaf_advertises_both_open_bounds(gm):
    gm.set_param_spec("ball", "elasticity", ParamSpec(bounds=(0.0, 1.0), transform="logit"))
    md = build_model_description(gm, model_name="m")
    var = next(v for v in md.variables if v.name == "ball.params.elasticity")
    assert var.min == float(F32.tiny)
    assert var.max == float(np.nextafter(np.float32(1.0), np.float32(-np.inf)))
    assert 0.0 < var.min < var.max < 1.0
    sc = _sidecar(gm, md)
    sc.set_params({"ball.params.elasticity": var.min})
    sc.set_params({"ball.params.elasticity": var.max})
    with pytest.raises(ValueError, match="above bound"):
        sc.set_params({"ball.params.elasticity": 1.0})


def test_inclusive_identity_bounds_are_unchanged(gm):
    md = build_model_description(gm, model_name="m")
    var = next(v for v in md.variables if v.name == "ball.params.elasticity")
    assert (var.min, var.max) == (0.0, 1.0)
    damping = next(v for v in md.variables if v.name == "s.params.damping")
    assert (damping.min, damping.max) == (0.0, None)
    sc = _sidecar(gm, md)
    sc.set_params({"ball.params.elasticity": 0.0, "s.params.damping": 0.0})


# ---------------------------------------------------------------------------
# The bridge holds the advertised envelope however its sidecar was built
# ---------------------------------------------------------------------------
#
# ``SidecarConfig.param_specs`` defaults to ``None``, and a sidecar built
# without it enforced no bounds at all: the XML said ``elasticity`` in
# [0, 1], and a ``set`` of 1.5 -- or an FMU-state archive installing -3.0 --
# answered ok.  The bridge now adds a bounds-only spec, from its model
# description's ``min`` / ``max``, for every exported parameter its sidecar
# has no spec for.


def _bare_sidecar(gm, md, **kw):
    """A sidecar built the way the parameter docs used to show it: no specs."""
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, **kw))


def _vr(md, name):
    return next(v.value_reference for v in md.variables if v.name == name)


def _archive_with(bridge, key, value):
    import base64
    import io

    from maddening.fmi.tcp_bridge import state_of

    blob = state_of(bridge.handle({"op": "get_state"}))
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        members = {k: data[k] for k in data.files}
    members[key] = np.asarray(value, members[key].dtype)
    buf = io.BytesIO()
    np.savez(buf, **members)
    return {"op": "set_state", "state": base64.b64encode(buf.getvalue()).decode("ascii")}


def test_the_bridge_enforces_the_advertised_bounds_without_param_specs(gm):
    from maddening.fmi.tcp_bridge import FmuTcpBridge

    md = build_model_description(gm, model_name="m")
    sidecar = _bare_sidecar(gm, md)
    assert sidecar.param_specs is None
    bridge = FmuTcpBridge(sidecar, md, master_dt=1e-2)
    try:
        el = _vr(md, "ball.params.elasticity")
        reply = bridge.handle({"op": "set", "vr": [el], "values": [1.5]})
        assert reply == {"ok": False,
                         "error": "ValueError: ball.params.elasticity=1.5 above bound 1.0"}
        refused = bridge.handle(_archive_with(bridge, "p/nodes/ball/elasticity", -3.0))
        assert refused["ok"] is False and "below bound 0.0" in refused["error"], refused
        assert bridge.handle({"op": "get", "vr": [el]})["values"] == [pytest.approx(0.7)]
        # the envelope's own edges are settable: min / max are inclusive
        for edge in (0.0, 1.0):
            assert bridge.handle({"op": "set", "vr": [el], "values": [edge]}) == {"ok": True}
        # and the sidecar's in-process door is held to the same envelope
        with pytest.raises(ValueError, match="above bound 1.0"):
            sidecar.set_params({"ball.params.elasticity": 2.0})
        assert sidecar.param_specs["nodes"]["ball"]["elasticity"].bounds == (0.0, 1.0)
    finally:
        bridge.stop()


def test_a_parameter_the_description_bounds_nowhere_stays_unbounded(gm):
    """Only advertised bounds are added: a leaf with no ``min`` / ``max``
    takes any finite value, as before."""
    from maddening.fmi.tcp_bridge import FmuTcpBridge

    md = build_model_description(gm, model_name="m")
    var = next(v for v in md.variables if v.name == "ball.params.gravity")
    assert (var.min, var.max) == (None, None)
    bridge = FmuTcpBridge(_bare_sidecar(gm, md), md, master_dt=1e-2)
    try:
        assert bridge.handle({"op": "set", "vr": [var.value_reference],
                              "values": [-123.0]}) == {"ok": True}
    finally:
        bridge.stop()


def test_the_sidecars_own_specs_are_kept_and_not_written_into(gm):
    """A spec the sidecar was given is the graph's declaration (the open
    bound of a ``log`` leaf, here) and wins over the advertised one; and
    the caller's ``param_specs`` tree is never modified."""
    from maddening.fmi.tcp_bridge import FmuTcpBridge

    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(8.0, None), transform="log"))
    md = build_model_description(gm, model_name="m", include_evolving=True)
    specs = gm.param_specs()
    sidecar = _bare_sidecar(gm, md, param_specs=specs)
    before = {n: dict(v) for n, v in specs["nodes"].items()}
    bridge = FmuTcpBridge(sidecar, md, master_dt=1e-2)
    try:
        assert sidecar.param_specs["nodes"]["s"]["stiffness"].transform == "log"
        assert {n: dict(v) for n, v in specs["nodes"].items()} == before
        k = _vr(md, "s.params.stiffness")
        refused = bridge.handle({"op": "set", "vr": [k], "values": [8.0]})
        assert refused["ok"] is False and "below bound 8.0" in refused["error"]
    finally:
        bridge.stop()


def test_an_fmu_started_outside_its_advertised_bounds_restores_its_own_snapshot():
    """Neighbouring case of the added specs: the restore exemption (a value
    the FMU was instantiated with is not a new value) holds for them too,
    so an FMU exported with a parameter outside its bounds still restores
    the snapshot it handed out -- and a ``set`` is still held to them."""
    from maddening.fmi.tcp_bridge import FmuTcpBridge

    g = GraphManager()
    g.add_node(SpringDamperNode(name="s", timestep=1e-2, stiffness=30.0, rest_length=0.4))
    g.add_external_input("s", "anchor_position")
    g.compile()
    g.set_param_spec("s", "stiffness", ParamSpec(bounds=(50.0, None)))
    md = build_model_description(g, model_name="m", include_evolving=True)
    bridge = FmuTcpBridge(_bare_sidecar(g, md), md, master_dt=1e-2)
    try:
        assert bridge.handle({"op": "step", "t": 0.0, "dt": 2e-2})["ok"]
        snap = bridge.handle({"op": "get_state"})["state"]
        assert bridge.handle({"op": "step", "t": 2e-2, "dt": 1e-2})["ok"]
        assert bridge.handle({"op": "set_state", "state": snap})["ok"] is True
        k = _vr(md, "s.params.stiffness")
        refused = bridge.handle({"op": "set", "vr": [k], "values": [30.0]})
        assert refused["ok"] is False and "below bound 50.0" in refused["error"]
    finally:
        bridge.stop()
