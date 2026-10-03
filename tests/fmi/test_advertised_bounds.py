"""The FMI ``min`` / ``max`` attributes are the *settable* envelope: for a
strict (log / logit) bound the description advertises the next
representable float inside -- or, where the spec refuses that one, the
first float inside it the spec accepts -- for an inclusive one the bound
itself.

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


def _accepted(spec, value):
    try:
        spec.check(np.asarray(value, dtype=np.float32))
    except ValueError:
        return False
    return True


#: Open bounds where one float inside is not a value the spec takes: the band
#: ``TINY <= |b| < 2**-102``, where that float is a subnormal distance from the
#: bound and the step's arithmetic flushes the distance to zero; a negative
#: bound one and a half ``TINY`` below zero, whose first accepted value is a
#: subnormal that flushes to zero; and a ``logit`` edge whose neighbour's
#: coordinate rounds onto the bound.
_FIRST_ACCEPTED = [
    (ParamSpec(bounds=(float(F32.tiny), None), transform="log"), 0),
    (ParamSpec(bounds=(1e-35, None), transform="log"), 0),
    (ParamSpec(bounds=(1e-35, 1.0), transform="logit"), 0),
    (ParamSpec(bounds=(-1.0, -1e-35), transform="logit"), 1),
    (ParamSpec(bounds=(-1.5 * float(F32.tiny), None), transform="log"), 0),
    (ParamSpec(bounds=(-1.0, 1.0), transform="logit"), 1),
]


@pytest.mark.parametrize("spec, side", _FIRST_ACCEPTED,
                         ids=["log-at-TINY", "log-at-1e-35", "logit-lower-in-band",
                              "logit-upper-in-band", "log-below-zero", "logit-coordinate"])
def test_an_open_bound_advertises_the_first_value_its_spec_accepts(spec, side):
    """FMI's ``min`` / ``max`` are inclusive, so the advertised value must be one
    the spec accepts, and the float just outside it one the spec refuses: then
    a bridge whose sidecar has no specs, holding only the advertised envelope,
    takes exactly what the graph takes.  One float inside the bound was
    advertised, which in the band ``TINY <= |b| < 2**-102`` is a distance the
    step's arithmetic flushes to zero: ``ParamSpec.check`` refused it and such
    a bridge took it (the acceptance oracle's N2)."""
    from maddening.fmi.model_description import _advertised_bound

    m = np.float32(_advertised_bound(spec, side, "float32"))
    outward = np.nextafter(m, np.float32(-np.inf) if side == 0 else np.float32(np.inf))
    assert _accepted(spec, m), (spec, m)
    assert not _accepted(spec, outward), (spec, outward)


@pytest.mark.parametrize("bound", [8.0, 1.0, 2.0 ** -102, 1e-20])
def test_an_open_bound_whose_neighbour_is_accepted_still_advertises_the_neighbour(bound):
    """Outside the band the advertised value is unchanged: the next float inside
    the bound, which the spec accepts."""
    from maddening.fmi.model_description import _advertised_bound

    spec = ParamSpec(bounds=(bound, None), transform="log")
    assert _advertised_bound(spec, 0, "float32") == float(
        np.nextafter(np.float32(bound), np.float32(np.inf)))


def test_a_flushed_band_bound_is_held_by_the_graph_and_the_description_alike(gm):
    """Through the description: a ``log`` spec at ``TINY`` advertises ``2 * TINY``,
    which the sidecar built with the graph's specs accepts, and refuses the
    float below it -- the value the description used to advertise."""
    tiny = float(F32.tiny)
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(tiny, None), transform="log"))
    md = build_model_description(gm, model_name="m", include_evolving=True)
    var = next(v for v in md.variables if v.name == "s.params.stiffness")
    assert var.min == 2 * tiny
    sc = _sidecar(gm, md)
    sc.set_params({"s.params.stiffness": var.min})
    old = float(np.nextafter(np.float32(tiny), np.float32(np.inf)))
    with pytest.raises(ValueError, match="below bound"):
        sc.set_params({"s.params.stiffness": old})


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


# ---------------------------------------------------------------------------
# A log leaf with no lower bound is bounded below by 0 all the same
# ---------------------------------------------------------------------------
#
# ``ParamSpec(transform="log")`` with no lower bound is measured from 0
# (``lo = bounds[0] or 0; p > lo strictly``), and ``gm.check_params`` refuses
# ``mass = -1`` there.  The description advertised no ``min`` at all, so a
# bridge whose sidecar had no specs accepted ``mass = -1.0`` and ran it (the
# spring's position read -1.07 after 20 steps), and one with specs refused a
# value the XML declared settable (claims FMU-016, FMU-033, FMU-039).


def _unbounded_log_mass(gm):
    gm.set_param_spec("s", "mass", ParamSpec(transform="log"))
    return build_model_description(gm, model_name="m")


def test_a_log_leaf_without_a_lower_bound_advertises_the_smallest_normal(gm):
    md = _unbounded_log_mass(gm)
    var = next(v for v in md.variables if v.name == "s.params.mass")
    assert var.min == float(F32.tiny) and var.max is None
    el = next(v for v in ET.fromstring(md.to_xml()).find("ModelVariables")
              if v.get("name") == "s.params.mass")
    assert float(el.get("min")) == var.min
    # the same envelope as an explicit bound of 0, which ParamSpec says it is
    explicit = ParamSpec(bounds=(0.0, None), transform="log")
    from maddening.fmi.model_description import _advertised_bound
    for dtype in ("float32", "float64"):
        assert _advertised_bound(ParamSpec(transform="log"), 0, dtype) == \
            _advertised_bound(explicit, 0, dtype) == float(np.finfo(dtype).tiny)
    # an identity leaf without bounds is still unbounded
    assert _advertised_bound(ParamSpec(), 0, "float32") is None


@pytest.mark.parametrize("with_specs", [False, True], ids=["bare sidecar", "with specs"])
def test_no_bridge_takes_a_log_leaf_to_zero_or_below(gm, with_specs):
    """Whichever way the sidecar was built: ``-1``, ``0`` and a negative
    subnormal are refused with nothing written, and the advertised ``min``
    itself is accepted."""
    from maddening.fmi.tcp_bridge import FmuTcpBridge

    md = _unbounded_log_mass(gm)
    kw = {"param_specs": gm.param_specs()} if with_specs else {}
    bridge = FmuTcpBridge(_bare_sidecar(gm, md, **kw), md, master_dt=1e-2)
    try:
        mass = _vr(md, "s.params.mass")
        for bad in (-1.0, 0.0, -1e-40):
            reply = bridge.handle({"op": "set", "type": "Float32", "vr": [mass], "values": [bad]})
            assert reply["ok"] is False and "below bound" in reply["error"], (bad, reply)
        assert bridge.handle({"op": "get", "vr": [mass]})["values"] == [1.0]
        refused = bridge.handle(_archive_with(bridge, "p/nodes/s/mass", -1.0))
        assert refused["ok"] is False and "below bound" in refused["error"], refused
        floor = next(v.min for v in md.variables if v.name == "s.params.mass")
        assert bridge.handle({"op": "set", "type": "Float32", "vr": [mass],
                              "values": [floor]}) == {"ok": True}
    finally:
        bridge.stop()
