"""An FMU never reports a parameter value it does not compute with.

The sidecar steps ``gm._compiled_step`` directly, so the graph's own refusal
of a ``gm.params`` leaf the step cannot read never ran for it:
``FmuSidecar.set_params({"p.params.pipe_radius": 0.5})`` was accepted,
``get_params`` reported 0.5, and every step kept the radius the wall mask was
built with -- against the sidecar's own promise that "an importer cannot
silently tune a constant the step never reads".  The bridge's ``set_state``
archive, which carries every leaf, was a second door, and the model
description exported every ``gm.params`` leaf as a *tunable* parameter,
``initial_*`` conditions included (setting one changed nothing).

Now the model description exports only the parameters the compiled step
reads and records the rest, with why, in ``fixed_parameters``; the bridge
applies that contract to the sidecar it serves; and every door into the
parameter tree -- ``set_params``, ``set_fmu_state``, the bridge's ``set``
and ``set_state``, the pickled RPC -- refuses a new value for a fixed one,
before anything is written.

Written from the confirmation audit of the 0.4.0 tree.
"""

from __future__ import annotations

import base64
import io
import os
import pickle
import xml.etree.ElementTree as ET

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import _STABILITY_REGISTRY
from maddening.core.graph_manager import GraphManager
from maddening.fmi.fmu_state import serialize_fmu_state
from maddening.fmi.model_description import build_model_description
from maddening.fmi.package import build_fmu_binary, find_c_compiler, write_fmu
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge
from maddening.nodes.adaptive.wavelet import WaveletAdaptiveNode
from maddening.nodes.lbm_pipe import LBMPipeNode
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr

PIPE = dict(nx=8, ny=10, nz=10, pipe_radius=0.8, propeller_x=2,
            propeller_strength=0.01)
RADIUS = float(np.float32(0.8))
NOT_TUNABLE = r"parameter 'p\.params\.pipe_radius' is not tunable"


@pytest.fixture
def promoted(monkeypatch):
    """``LBMPipeNode`` and ``WaveletAdaptiveNode`` are ``EXPERIMENTAL``, and
    the FMU exports no variable of a node at that level; promote them the
    way a later release might, so the liveness rule is what decides."""
    for cls in (LBMPipeNode, WaveletAdaptiveNode):
        monkeypatch.setitem(_STABILITY_REGISTRY, f"{cls.__module__}.{cls.__name__}",
                            StabilityLevel.STABLE)


def _pipe(**overrides):
    gm = GraphManager()
    gm.add_node(LBMPipeNode("p", 1.0, **{**PIPE, **overrides}))
    gm.compile()
    return gm


def _sidecar(gm, md, **kw):
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
        **kw,
    ))


def _run(sc, n=10):
    for _ in range(n):
        sc.step({})
    return np.asarray(sc.state["p"]["velocity"])


def _exported(md):
    return {v.name for v in md.variables if v.causality == "parameter"}


def _every_leaf(gm):
    return {f"{n}.params.{k}" for n, leaves in gm.params["nodes"].items() for k in leaves}


def _assert_exports_only_what_the_step_reads(gm, md):
    """The invariant: every exported parameter is read by the compiled step
    and built into no static; every other leaf is listed as fixed."""
    reads = gm._params_read_by_step()
    assert reads is not None
    for name in _exported(md):
        node, _, key = name.partition(".params.")
        assert (node, key) in reads, name
        deps = gm._nodes[node].node.static_data_deps() or {}
        assert not any(key in names for names in deps.values()), name
    assert set(md.fixed_parameters) == _every_leaf(gm) - _exported(md)
    tunable_in_xml = {el.get("name") for el in ET.fromstring(md.to_xml()).iter()
                      if el.get("causality") == "parameter"}
    assert tunable_in_xml == _exported(md)


# ---------------------------------------------------------------------------
# The model description advertises no knob that does nothing
# ---------------------------------------------------------------------------


def test_an_experimental_node_exports_no_parameter_and_lists_every_leaf_as_fixed():
    gm = _pipe()
    md = build_model_description(gm, model_name="pipe", include_evolving=True)
    assert not _exported(md)
    assert set(md.fixed_parameters) == _every_leaf(gm)
    assert "stability level" in md.fixed_parameters["p.params.pipe_radius"]


def test_a_pipe_exports_what_its_step_reads_and_not_its_geometry(promoted):
    gm = _pipe()
    md = build_model_description(gm, model_name="pipe")
    assert {"p.params.tau", "p.params.propeller_strength", "p.params.gravity"} <= _exported(md)
    # The geometry is declared: the masks are built from it (static_data_deps).
    for name in ("p.params.pipe_radius", "p.params.propeller_radius"):
        assert "static_data_deps" in md.fixed_parameters[name], name
    # The initial fill and velocity are read by initial_state() alone.
    for name in ("p.params.fill_fraction", "p.params.initial_velocity"):
        assert "no operation of the compiled step" in md.fixed_parameters[name], name
    _assert_exports_only_what_the_step_reads(gm, md)


def test_a_wavelet_node_does_not_export_the_mass_baked_into_its_operator(promoted):
    gm = GraphManager()
    gm.add_node(WaveletAdaptiveNode("w", 1.0, n_levels=3, mass=1.0, blindness_gate=False))
    gm.compile()
    md = build_model_description(gm, model_name="wavelet")
    assert "w.params.mass" not in _exported(md)
    assert "static_data_deps" in md.fixed_parameters["w.params.mass"]
    _assert_exports_only_what_the_step_reads(gm, md)


def test_a_stable_graph_does_not_export_its_initial_conditions():
    gm = _graph()
    md = build_model_description(gm, model_name="m")
    assert {"spring.params.initial_position", "ball.params.initial_velocity",
            "table.params.position"} <= set(md.fixed_parameters)
    assert "spring.params.stiffness" in _exported(md)
    _assert_exports_only_what_the_step_reads(gm, md)


# ---------------------------------------------------------------------------
# The sidecar
# ---------------------------------------------------------------------------


def test_the_sidecar_refuses_a_value_its_step_would_ignore(promoted):
    """The audit's probe: accepted, reported, and never used."""
    gm = _pipe()
    md = build_model_description(gm, model_name="pipe")
    sc = _sidecar(gm, md, fixed_params=md.fixed_parameters)
    with pytest.raises(ValueError, match=NOT_TUNABLE + r": LBMPipeNode bakes it into static_data"):
        sc.set_params({"p.params.pipe_radius": 0.5})
    # atomic: the tunable key of a refused request is not written either
    with pytest.raises(ValueError, match=NOT_TUNABLE):
        sc.set_params({"p.params.propeller_strength": 0.03, "p.params.pipe_radius": 0.5})
    assert float(sc.get_params()["p.params.pipe_radius"]) == RADIUS
    assert float(sc.get_params()["p.params.propeller_strength"]) == pytest.approx(0.01)
    np.testing.assert_array_equal(_run(sc), _run(_sidecar(_pipe(), md)))
    sc.set_params({"p.params.pipe_radius": RADIUS})        # its own value: fine


def test_a_tunable_parameter_still_takes_effect_on_the_next_step(promoted):
    gm = _pipe()
    md = build_model_description(gm, model_name="pipe")
    sc = _sidecar(gm, md, fixed_params=md.fixed_parameters)
    sc.set_params({"p.params.propeller_strength": 0.03})
    moved = _run(sc)
    built = _pipe(propeller_strength=0.03)
    np.testing.assert_allclose(moved, _run(_sidecar(built, md)), rtol=1e-6, atol=1e-9)
    assert not np.allclose(moved, _run(_sidecar(_pipe(), md)), rtol=1e-6, atol=1e-9)


def test_a_snapshot_cannot_install_a_value_the_step_cannot_read(promoted):
    gm = _pipe()
    md = build_model_description(gm, model_name="pipe")
    sc = _sidecar(gm, md, fixed_params=md.fixed_parameters)
    params = {s: {o: dict(l) for o, l in owners.items()} for s, owners in sc.params.items()}
    params["nodes"]["p"]["pipe_radius"] = jnp.asarray(0.5, jnp.float32)
    snap = serialize_fmu_state(state=sc.state, schema_token=md.instantiation_token,
                               params=params)
    sc.step({})                                   # so a restore would be visible
    state = {n: {f: np.asarray(v).copy() for f, v in fs.items()} for n, fs in sc.state.items()}
    with pytest.raises(ValueError, match=NOT_TUNABLE):
        sc.set_fmu_state(snap)
    for n, fs in state.items():
        for f, v in fs.items():
            np.testing.assert_array_equal(np.asarray(sc.state[n][f]), v)
    assert float(sc.get_params()["p.params.pipe_radius"]) == RADIUS


def test_the_pickled_rpc_answers_err_for_a_value_the_step_cannot_read(promoted):
    gm = _pipe()
    md = build_model_description(gm, model_name="pipe")
    sc = _sidecar(gm, md, fixed_params=md.fixed_parameters, allow_pickle_rpc=True)
    status, detail = pickle.loads(sc.handle(pickle.dumps(
        ("set_params", {"p.params.pipe_radius": 0.5}))))
    assert status == "err" and "is not tunable" in detail
    assert float(sc.get_params()["p.params.pipe_radius"]) == RADIUS


# ---------------------------------------------------------------------------
# The bridge: the model description is the contract, however the sidecar
# was configured
# ---------------------------------------------------------------------------


def test_a_bridge_applies_its_model_description_to_a_sidecar_built_without_it(promoted):
    gm = _pipe()
    md = build_model_description(gm, model_name="pipe")
    sc = _sidecar(gm, md)                          # no fixed_params
    bridge = FmuTcpBridge(sc, md, master_dt=1.0)
    try:
        assert set(sc.fixed_params) == set(md.fixed_parameters)
        with pytest.raises(ValueError, match=NOT_TUNABLE):
            sc.set_params({"p.params.pipe_radius": 0.5})
        # a parameter the description does not export at all is fixed too
        unexported = FmuTcpBridge(_sidecar(_pipe(), md),
                                  build_model_description(gm, model_name="pipe",
                                                          include_parameters=False),
                                  master_dt=1.0)
        try:
            with pytest.raises(ValueError, match="propeller_strength' is not tunable"):
                unexported._sidecar.set_params({"p.params.propeller_strength": 0.03})
        finally:
            unexported.stop()
    finally:
        bridge.stop()


def test_the_bridge_set_state_refuses_an_archive_that_changes_a_fixed_parameter(promoted):
    """``set`` cannot address a fixed parameter (it has no value reference);
    the archive can name it, so it is checked there, before anything is
    written, and answered with an error the C wrapper turns into
    ``fmi3Error``."""
    gm = _pipe()
    md = build_model_description(gm, model_name="pipe")
    bridge = FmuTcpBridge(_sidecar(gm, md), md, master_dt=1.0)
    try:
        assert bridge.handle({"op": "hello"})["ok"]
        blob = base64.b64decode(bridge.handle({"op": "get_state"})["state"])
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        members["p/nodes/p/pipe_radius"] = np.asarray(0.5, np.float32)
        buf = io.BytesIO()
        np.savez(buf, **members)
        assert bridge.handle({"op": "step", "dt": 1.0})["ok"]
        before = {n: {f: np.asarray(v).copy() for f, v in fs.items()}
                  for n, fs in bridge._sidecar.state.items()}
        reply = bridge.handle({"op": "set_state",
                               "state": base64.b64encode(buf.getvalue()).decode("ascii")})
        assert not reply["ok"]
        assert "'p.params.pipe_radius' is not tunable" in reply["error"]
        for n, fs in before.items():
            for f, v in fs.items():
                np.testing.assert_array_equal(np.asarray(bridge._sidecar.state[n][f]), v)
        assert float(bridge._sidecar.get_params()["p.params.pipe_radius"]) == RADIUS
        # a tunable one still goes through ``set`` and moves the next step
        vr = next(v.value_reference for v in md.variables
                  if v.name == "p.params.propeller_strength")
        assert bridge.handle({"op": "set", "vr": [vr], "values": [0.03]})["ok"]
        assert float(bridge._sidecar.get_params()["p.params.propeller_strength"]) \
            == pytest.approx(0.03)
    finally:
        bridge.stop()


def test_an_archive_cannot_change_an_initial_condition_of_a_stable_graph():
    gm = _graph()
    md, bridge = _bridge(gm)
    try:
        blob = base64.b64decode(bridge.handle({"op": "get_state"})["state"])
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        members["p/nodes/spring/initial_position"] = np.asarray(0.9, np.float32)
        buf = io.BytesIO()
        np.savez(buf, **members)
        reply = bridge.handle({"op": "set_state",
                               "state": base64.b64encode(buf.getvalue()).decode("ascii")})
        assert not reply["ok"]
        assert "'spring.params.initial_position' is not tunable" in reply["error"]
        # the unchanged archive still restores
        assert bridge.handle({"op": "set_state",
                              "state": base64.b64encode(blob).decode("ascii")})["ok"]
    finally:
        bridge.stop()


@pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")
def test_the_compiled_fmu_returns_fmi3_error_for_such_an_archive(tmp_path):
    """End to end through the C wrapper: ``fmi3SetFMUState`` of an archive
    changing an initial condition is ``fmi3Error``, and the instance keeps
    its state."""
    pytest.importorskip("fmpy")
    from fmpy import extract, read_model_description
    from fmpy.fmi1 import FMICallException
    from fmpy.fmi3 import FMU3Slave

    gm = _graph()
    md, bridge = _bridge(gm)
    so = build_fmu_binary(tmp_path)
    with bridge:
        fmu = write_fmu(md, tmp_path / "plant.fmu", binary=so, endpoint=bridge.endpoint)
        unz = extract(str(fmu))
        desc = read_model_description(unz)
        inst = FMU3Slave(guid=desc.guid, unzipDirectory=unz,
                         modelIdentifier=desc.coSimulation.modelIdentifier, instanceName="i")
        inst.instantiate(loggingOn=True)
        inst.enterInitializationMode(startTime=0.0)
        inst.exitInitializationMode()
        pos = _vr(md, "spring.position")
        inst.doStep(currentCommunicationPoint=0.0, communicationStepSize=DT)
        st = inst.getFMUState()
        raw = bytes(inst.serializeFMUState(st))
        encoded = not raw.startswith(b"PK")
        blob = base64.b64decode(raw) if encoded else raw
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        members["p/nodes/spring/initial_position"] = np.asarray(0.9, np.float32)
        buf = io.BytesIO()
        np.savez(buf, **members)
        forged = base64.b64encode(buf.getvalue()) if encoded else buf.getvalue()
        inst.doStep(currentCommunicationPoint=DT, communicationStepSize=DT)
        here = inst.getFloat64([pos])[0]
        bad = inst.deserializeFMUState(forged)
        with pytest.raises(FMICallException) as caught:
            inst.setFMUState(bad)
        assert caught.value.status == 3                    # fmi3Error
        assert inst.getFloat64([pos])[0] == here           # nothing restored
        inst.setFMUState(st)                               # the genuine one still works
        inst.freeFMUState(st)
        inst.freeFMUState(bad)
        inst.terminate()
        inst.freeInstance()
