"""The bridge accepts exactly the parameter values its model description
advertises as settable: every value inside ``min`` / ``max``, none outside.

Two ways it did not (claims FMU-016, FMU-033, FMU-039):

* ``GraphManager.set_param_spec`` changes a spec without dirtying the graph,
  so a description built before the change advertised the old bounds while
  a sidecar built after it (``param_specs=gm.param_specs()``, as the guide
  builds it) enforced the new ones.  ``FmuSidecar._adopt_advertised_bounds``
  kept a spec the sidecar had (``setdefault``), so the bridge accepted
  ``damping = 0.5`` and ``7.0`` against an advertised ``[1, 5]``, or refused
  values inside an advertised ``[0, inf)``.  The bridge now refuses a
  description whose graph's specs changed since it was built, refuses a
  sidecar spec that enforces another envelope, and holds every write and
  restore to the advertised bounds alongside the sidecar's own spec.
* A float32 ``logit`` leaf under ``(-1, 1)`` advertised ``max =
  nextafter(1, 0)``, whose ``(p - lo) / (hi - lo)`` rounds to 1: a sidecar
  with specs -- and FMPy through the compiled wrapper -- refused the FMU's
  own advertised max.  An open bound is now advertised as the outermost
  value inside it that ``ParamSpec.check`` accepts.

Found by the B1 round-8 audit of the claims inventory (findings M3, L1).
"""

import base64
import dataclasses
import io

import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.fmi import MODEL_IDENTIFIER, FmuTcpBridge, build_model_description
from maddening.fmi.package import find_c_compiler
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import state_of
from maddening.nodes.spring import SpringDamperNode

F32 = np.float32
TINY = float(np.finfo(np.float32).tiny)


def _spring(**kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, damping=2.0, initial_position=0.5, **kw))
    gm.compile()
    return gm


def _describe(gm):
    return build_model_description(gm, model_name="Plant", model_identifier=MODEL_IDENTIFIER)


_FROM_THE_GRAPH = object()


def _sidecar(gm, md, specs=_FROM_THE_GRAPH):
    """The sidecar as the guide builds it: ``param_specs=gm.param_specs()``
    unless ``specs`` says otherwise (``None``: none)."""
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,     # noqa: SLF001
        initial_state=gm._state, params=gm.params,                          # noqa: SLF001
        param_specs=gm.param_specs() if specs is _FROM_THE_GRAPH else specs,
        fixed_params=md.fixed_parameters,
        input_resolver=gm._resolve_external_inputs))                        # noqa: SLF001


def _var(md, name):
    return next(v for v in md.variables if v.name == name)


def _set(bridge, var, value):
    return bridge.handle({"op": "set", "type": "Float32", "vr": [var.value_reference],
                          "values": [float(value)]})


def _archive_with(bridge, key, value):
    blob = state_of(bridge.handle({"op": "get_state"}))
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        members = {k: data[k] for k in data.files}
    members[key] = np.asarray(value, members[key].dtype)
    buf = io.BytesIO()
    np.savez(buf, **members)
    return {"op": "set_state", "state": base64.b64encode(buf.getvalue()).decode("ascii")}


# ---------------------------------------------------------------------------
# A spec changed between the description and the sidecar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("at_description, at_sidecar", [
    (ParamSpec(bounds=(1.0, 5.0)), ParamSpec(bounds=(0.0, None))),
    (ParamSpec(bounds=(0.0, None)), ParamSpec(bounds=(1.0, 5.0))),
    (ParamSpec(bounds=(0.0, 10.0)), ParamSpec(bounds=(0.0, 10.0), transform="logit")),
], ids=["widened", "narrowed", "opened"])
def test_a_spec_changed_after_the_description_is_refused_when_the_bridge_is_built(
        at_description, at_sidecar):
    """The audit's reproducer: the description, then ``set_param_spec``, then
    the sidecar from ``gm.param_specs()``.  The bridge used to start and
    enforce the sidecar's spec -- ``0.5`` and ``7.0`` accepted against an
    advertised ``[1, 5]``, or ``0.5`` refused inside an advertised ``[0,
    inf)``.  It refuses to start, naming the parameter and both envelopes."""
    gm = _spring()
    gm.set_param_spec("s", "damping", at_description)
    md = _describe(gm)
    gm.set_param_spec("s", "damping", at_sidecar)
    assert not gm._dirty                                              # noqa: SLF001
    with pytest.raises(ValueError, match=r"ParamSpec of 's\.params\.damping' has changed "
                                         r"since \(GraphManager\.set_param_spec\)"):
        FmuTcpBridge(_sidecar(gm, md), md, master_dt=gm.timestep)


def test_a_unit_changed_after_the_description_is_refused_when_the_bridge_is_built():
    """The envelope the stale check compares includes the unit the XML
    advertises: the same number in other units is another value."""
    gm = _spring()
    gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.0, None), units="N s/m"))
    md = _describe(gm)
    gm.set_param_spec("s", "damping", ParamSpec(bounds=(0.0, None), units="kN s/m"))
    with pytest.raises(ValueError, match="unit='N s/m'.*unit='kN s/m'"):
        FmuTcpBridge(_sidecar(gm, md), md, master_dt=gm.timestep)


def test_a_spec_change_that_leaves_the_advertised_envelope_alone_is_not_refused():
    """Freezing a parameter for a fit (``trainable=False``) or rewording its
    description changes nothing the XML advertises or the bridge enforces;
    and a spec changed and changed back is the description's again."""
    gm = _spring()
    md = _describe(gm)
    damping = _var(md, "s.params.damping")
    own = gm.get_node("s").param_specs()
    gm.set_param_spec("s", "damping", dataclasses.replace(
        own["damping"], trainable=False, description="viscous damping"))
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(1.0, None)))
    gm.set_param_spec("s", "stiffness", own["stiffness"])
    bridge = FmuTcpBridge(_sidecar(gm, md), md, master_dt=gm.timestep)
    try:
        assert _set(bridge, damping, 0.0) == {"ok": True}
        assert _set(bridge, damping, 7.0) == {"ok": True}
        refused = _set(bridge, damping, -1.0)
        assert refused["ok"] is False and "below bound 0.0" in refused["error"], refused
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# A sidecar spec that is not the description's
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("own", [
    ParamSpec(),
    ParamSpec(bounds=(0.0, 10.0)),
    ParamSpec(bounds=(1.0, None)),
    ParamSpec(bounds=(0.0, None), transform="log"),
], ids=["unbounded", "capped", "raised", "opened"])
def test_a_sidecar_spec_that_enforces_another_envelope_is_refused(own):
    """A hand-made ``param_specs`` tree (the graph unchanged, so no stale
    check applies) whose spec for an exported parameter enforces another
    envelope than ``min = 0.0`` -- wider, narrower, or open where the XML is
    inclusive (``log`` refuses the advertised ``0.0``)."""
    gm = _spring()
    md = _describe(gm)
    assert (_var(md, "s.params.damping").min, _var(md, "s.params.damping").max) == (0.0, None)
    specs = gm.param_specs()
    specs["nodes"]["s"]["damping"] = own
    with pytest.raises(ValueError, match=r"the sidecar's ParamSpec for 's\.params\.damping' "
                                         r"enforces .* but the model description advertises "
                                         r"min=0\.0, max=None"):
        FmuTcpBridge(_sidecar(gm, md, specs), md, master_dt=gm.timestep)


@pytest.mark.parametrize("own", [
    ParamSpec(bounds=(0.0, None)),
    ParamSpec(trainable=False, bounds=(0.0, float("inf")), description="other words"),
], ids=["equal", "same envelope"])
def test_a_sidecar_spec_with_the_advertised_envelope_is_accepted_and_kept(own):
    gm = _spring()
    md = _describe(gm)
    specs = gm.param_specs()
    specs["nodes"]["s"]["damping"] = own
    sidecar = _sidecar(gm, md, specs)
    bridge = FmuTcpBridge(sidecar, md, master_dt=gm.timestep)
    try:
        assert sidecar.param_specs["nodes"]["s"]["damping"] is own
        assert _set(bridge, _var(md, "s.params.damping"), -1.0)["ok"] is False
    finally:
        bridge.stop()


def test_the_advertised_bounds_hold_even_when_the_sidecars_own_spec_admits_more():
    """The intersection: a write and a restore are held to the sidecar's
    spec *and* to the advertised bounds.  Widen the sidecar's own spec after
    the bridge has checked it (``param_specs`` exposes the tree it reads),
    and a value the XML forbids is still refused, through every door."""
    gm = _spring()
    md = _describe(gm)
    sidecar = _sidecar(gm, md)
    bridge = FmuTcpBridge(sidecar, md, master_dt=gm.timestep)
    try:
        damping = _var(md, "s.params.damping")
        sidecar.param_specs["nodes"]["s"]["damping"] = ParamSpec()
        refused = _set(bridge, damping, -1.0)
        assert refused["ok"] is False and "below bound 0.0" in refused["error"], refused
        with pytest.raises(ValueError, match="below bound 0.0"):
            sidecar.set_params({"s.params.damping": -1.0})
        refused = bridge.handle(_archive_with(bridge, "p/nodes/s/damping", -1.0))
        assert refused["ok"] is False and "below bound 0.0" in refused["error"], refused
        assert bridge.handle({"op": "get", "vr": [damping.value_reference]})["values"] == [2.0]
        assert _set(bridge, damping, 0.0) == {"ok": True}
    finally:
        bridge.stop()


# ---------------------------------------------------------------------------
# An open bound is advertised as a value check accepts
# ---------------------------------------------------------------------------

_OPEN = [
    ParamSpec(bounds=(-1.0, 1.0), transform="logit"),
    ParamSpec(bounds=(-10.0, 10.0), transform="logit"),
    ParamSpec(bounds=(0.5, 2.0), transform="logit"),
    ParamSpec(bounds=(0.0, 1.0), transform="logit"),
    ParamSpec(bounds=(-1.0, -1e-35), transform="logit"),
    ParamSpec(bounds=(8.0, None), transform="log"),
    ParamSpec(bounds=(TINY, None), transform="log"),
    ParamSpec(bounds=(1e-35, None), transform="log"),
]
_OPEN_IDS = ["logit(-1,1)", "logit(-10,10)", "logit(0.5,2)", "logit(0,1)",
             "logit(-1,-1e-35)", "log(8)", "log(TINY)", "log(1e-35)"]


def _open_bridge(spec, with_specs):
    lo, hi = spec.bounds
    start = 1.0 if hi is None else 0.5 * (lo + hi)
    gm = _spring(rest_length=start)
    gm.set_param_spec("s", "rest_length", spec)
    md = _describe(gm)
    bridge = FmuTcpBridge(_sidecar(gm, md, _FROM_THE_GRAPH if with_specs else None), md,
                          master_dt=gm.timestep)
    return md, bridge


@pytest.mark.parametrize("with_specs", [True, False], ids=["specs", "no specs"])
@pytest.mark.parametrize("spec", _OPEN, ids=_OPEN_IDS)
def test_every_advertised_open_bound_is_settable_and_the_next_float_out_is_not(spec, with_specs):
    """Both documented sidecar configurations accept the advertised ``min``
    and ``max`` and refuse the next float32 outside each: the envelope an
    importer sees is the one enforced, to the last float.  ``logit(-1, 1)``
    and ``logit(-10, 10)`` used to advertise a ``max`` the specs refused;
    the ``log`` bounds in the subnormal-spacing band (``TINY``, ``1e-35``)
    and ``logit(-1, -1e-35)`` a bound the specs refused and a bare sidecar
    took."""
    md, bridge = _open_bridge(spec, with_specs)
    try:
        var = _var(md, "s.params.rest_length")
        for edge, outward in ((var.min, -np.inf), (var.max, np.inf)):
            if edge is None:
                continue
            assert _set(bridge, var, edge) == {"ok": True}, (edge, spec)
            beyond = float(np.nextafter(F32(edge), F32(outward)))
            refused = _set(bridge, var, beyond)
            assert refused["ok"] is False, (beyond, spec, refused)
    finally:
        bridge.stop()


@pytest.mark.parametrize("spec", _OPEN, ids=_OPEN_IDS)
def test_an_open_bound_is_advertised_as_the_outermost_value_its_check_accepts(spec):
    """The advertised bound is accepted by ``ParamSpec.check`` in the leaf's
    dtype, and the next float32 outward is not -- so a sidecar's own spec and
    the advertised bounds admit the same values.  Where the next float
    inside the bound is accepted, it is still what is advertised."""
    md, bridge = _open_bridge(spec, with_specs=True)
    bridge.stop()
    var = _var(md, "s.params.rest_length")
    for side, edge, outward in ((0, var.min, -np.inf), (1, var.max, np.inf)):
        bound = spec.bounds[side]
        if bound is None:
            continue
        spec.check(np.asarray(edge, np.float32))
        with pytest.raises(ValueError):
            spec.check(np.asarray(np.nextafter(F32(edge), F32(outward)), np.float32))
        neighbour = np.nextafter(F32(bound), F32(-outward))
        try:
            spec.check(np.asarray(neighbour, np.float32))
        except ValueError:
            continue
        if abs(neighbour) >= TINY:
            assert edge == float(neighbour), (spec, side)


@pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")
def test_the_wrapper_sets_a_logit_parameter_to_its_advertised_max(tmp_path):
    """End to end, as the audit's companion reproducer did: FMPy opens the
    packaged FMU and ``fmi3SetFloat32`` each bound its own
    ``modelDescription.xml`` advertises.  The max used to be ``fmi3Error``
    (``has no finite logit coordinate``)."""
    fmpy = pytest.importorskip("fmpy")
    from fmpy.fmi3 import FMU3Slave

    from maddening.fmi.package import build_fmu_binary, write_fmu

    md, bridge = _open_bridge(ParamSpec(bounds=(-1.0, 1.0), transform="logit"), with_specs=True)
    with bridge:
        fmu = write_fmu(md, tmp_path / "plant.fmu", binary=build_fmu_binary(tmp_path),
                        endpoint=bridge.endpoint)
        unz = fmpy.extract(str(fmu))
        desc = fmpy.read_model_description(unz)
        var = next(v for v in desc.modelVariables if v.name == "s.params.rest_length")
        inst = FMU3Slave(guid=desc.guid, unzipDirectory=unz,
                         modelIdentifier=MODEL_IDENTIFIER, instanceName="i")
        inst.instantiate()
        try:
            inst.enterInitializationMode(startTime=0.0)
            inst.exitInitializationMode()
            for edge in (float(var.min), float(var.max)):
                inst.setFloat32([var.valueReference], [edge])
                assert inst.getFloat32([var.valueReference])[0] == F32(edge)
            inst.terminate()
        finally:
            inst.freeInstance()
