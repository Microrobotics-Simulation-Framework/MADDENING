"""An FMU restores its own snapshots: a parameter started outside its bounds,
and a state field it started non-finite.

A graph runs whatever its constructor was given: ``ParamSpec.bounds`` are
metadata to it, so a graph can be exported with a parameter outside its
declared bounds (here ``stiffness = 30`` under a lower bound of 50).  Both
FMU restore paths -- the TCP bridge's ``set_state`` and
``FmuSidecar.set_fmu_state`` -- bounds-checked the *whole* restored tree, so
the FMU refused the snapshot it had handed out itself, while
``GraphManager.load_state`` restored the same graph's checkpoint.

The two paths now share one check (``FmuSidecar._check_restored_params``),
applied to the values a restore would *install*: a value the parameter
holds now, or held when the FMU was instantiated, is not a new value.
Every value a snapshot of this FMU can carry is one of those or one
``set_params`` accepted, so its own snapshots restore -- and a forged
out-of-bounds value is refused by both, with the same words.  State follows
the same rule: a non-finite value the FMU was instantiated with (a
``diagnostics=True`` group's NaN-seeded spectral ``_meta`` slots) restores,
and a field that became non-finite still does not.
"""

from __future__ import annotations

import io
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.fmi.fmu_state import deserialize_fmu_state, serialize_fmu_state
from maddening.fmi.model_description import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, state_of
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

STIFFNESS = "spring.params.stiffness"


@pytest.fixture(scope="module")
def model():
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=1e-2))
    gm.add_node(BallNode(name="ball", timestep=1e-2, initial_position=1.0, elasticity=0.7))
    gm.add_node(SpringDamperNode(name="spring", timestep=1e-2, stiffness=30.0, damping=2.0))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    gm.set_param_spec("spring", "stiffness", ParamSpec(bounds=(50.0, None)))
    return gm, build_model_description(gm, model_name="m")


def _sidecar(model) -> FmuSidecar:
    gm, md = model
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,  # noqa: SLF001
        initial_state={n: dict(f) for n, f in gm._state.items()},        # noqa: SLF001
        params={s: {o: dict(v) for o, v in owners.items()} for s, owners in gm.params.items()},
        param_specs=gm.param_specs(), fixed_params=md.fixed_parameters))


@pytest.fixture
def paths(model):
    """A sidecar driven in process, and a bridge over its own sidecar."""
    _, md = model
    bridge = FmuTcpBridge(_sidecar(model), md, master_dt=1e-2)
    try:
        yield _sidecar(model), bridge
    finally:
        bridge.stop()


def _stiffness(sc) -> float:
    return float(sc.get_params()[STIFFNESS])


def _ext(model):
    return model[0]._default_external_inputs()                     # noqa: SLF001


def _with_stiffness_snapshot(sc, value):
    state, params = deserialize_fmu_state(sc.get_fmu_state(),
                                          expected_schema_token=sc._config.schema_token,  # noqa: SLF001
                                          return_params=True)
    params["nodes"]["spring"]["stiffness"] = np.float32(value)
    return serialize_fmu_state(state=state, schema_token=sc._config.schema_token,  # noqa: SLF001
                               params=params)


def _with_stiffness_archive(blob, value):
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        members = {k: data[k] for k in data.files}
    members["p/nodes/spring/stiffness"] = np.float32(value)
    buf = io.BytesIO()
    np.savez(buf, **members)
    return buf.getvalue()


def test_the_fixture_starts_outside_the_declared_bound(model):
    gm, _ = model
    with pytest.raises(ValueError, match="below bound 50"):
        gm.check_params()


def test_the_instantiation_snapshot_restores_after_a_legal_set(model, paths):
    """Snapshot at 30, a set to 60 (inside the bound), a step, a restore:
    the snapshot is the FMU's own, so it puts 30 back on both paths."""
    sc, bridge = paths
    snap = sc.get_fmu_state()
    blob = state_of(bridge.handle({"op": "get_state"}))
    sc.set_params({STIFFNESS: 60.0})
    vr = next(v.value_reference for v in model[1].variables if v.name == STIFFNESS)
    assert bridge.handle({"op": "set", "vr": [vr], "values": [60.0]}) == {"ok": True}
    sc.step(_ext(model))
    assert bridge.handle({"op": "step", "t": 0.0, "dt": 1e-2})["ok"]
    sc.set_fmu_state(snap)
    assert bridge.handle({"op": "set_state", "state": blob}) == {"ok": True}
    assert _stiffness(sc) == 30.0
    assert float(bridge._sidecar.get_params()[STIFFNESS]) == 30.0  # noqa: SLF001


def test_set_params_still_holds_a_set_value_to_the_bounds(paths):
    """A set is a value the importer chose, so it is held to the bounds even
    when it is the value the FMU started with: only a restore puts back what
    the FMU held."""
    sc, _ = paths
    with pytest.raises(ValueError, match="below bound 50"):
        sc.set_params({STIFFNESS: 30.0})


@pytest.mark.parametrize("value, refusal", [
    (40.0, "below bound 50"),     # out of bounds, neither live nor instantiation value
    (70.0, None),                 # a new value inside the bound
    (30.0, None),                 # the instantiation (and live) value
])
def test_both_restore_paths_judge_a_forged_value_alike(paths, value, refusal):
    sc, bridge = paths
    blob = state_of(bridge.handle({"op": "get_state"}))
    reply = bridge.handle({"op": "set_state", "state": _with_stiffness_archive(blob, value)})
    snap = _with_stiffness_snapshot(sc, value)
    if refusal is None:
        assert reply == {"ok": True}, reply
        sc.set_fmu_state(snap)
        assert _stiffness(sc) == value
        return
    with pytest.raises(ValueError, match=refusal) as exc:
        sc.set_fmu_state(snap)
    assert reply == {"ok": False, "error": f"ValueError: {exc.value}"}, reply
    assert _stiffness(sc) == 30.0


def test_after_a_legal_set_the_instantiation_value_is_still_restorable(paths):
    """Live at 60 and instantiated at 30: a snapshot carrying 30 restores
    (the FMU held it), one carrying 40 does not (it never did)."""
    sc, _ = paths
    sc.set_params({STIFFNESS: 60.0})
    sc.set_fmu_state(_with_stiffness_snapshot(sc, 30.0))
    assert _stiffness(sc) == 30.0
    sc.set_params({STIFFNESS: 60.0})
    with pytest.raises(ValueError, match="below bound 50"):
        sc.set_fmu_state(_with_stiffness_snapshot(sc, 40.0))
    assert _stiffness(sc) == 60.0


# ---------------------------------------------------------------------------
# The same rule for state: a non-finite value the FMU started with restores
# ---------------------------------------------------------------------------

def _diagnostics_model():
    """A coupling group with ``diagnostics=True``: its spectral ``_meta``
    slots are seeded NaN at instantiation, until a solve fills them."""
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0, gravity=-3.0))
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=20.0, rest_length=0.5))
    gm.add_edge("ball", "spring", "position", "anchor_position")
    gm.add_edge("spring", "ball", "position", "table_position")
    gm.add_coupling_group(["ball", "spring"], diagnostics=True, max_iterations=3)
    gm.compile()
    return gm, build_model_description(gm, model_name="d")


@pytest.fixture(scope="module")
def diagnostics_model():
    return _diagnostics_model()


def test_a_snapshot_of_the_nan_seeded_diagnostics_restores_on_both_paths(diagnostics_model):
    """Found by the FMU differential property once it drew graphs starting
    outside their bounds.  Both restore paths refused the snapshot taken at
    instantiation ("_meta...rho_spectral: value must be finite") while
    ``GraphManager.load_state`` restored the graph's checkpoint."""
    gm, md = diagnostics_model
    seeds = {k: float(v) for k, v in gm._state["_meta"].items()}   # noqa: SLF001
    assert any(np.isnan(v) for v in seeds.values()), seeds
    sc = _sidecar(diagnostics_model)
    snap = sc.get_fmu_state()
    sc.step(gm._default_external_inputs())                          # noqa: SLF001
    sc.set_fmu_state(snap)
    for key, value in seeds.items():
        np.testing.assert_array_equal(np.asarray(sc.state["_meta"][key]), value)
    bridge = FmuTcpBridge(_sidecar(diagnostics_model), md, master_dt=1e-2)
    try:
        blob = state_of(bridge.handle({"op": "get_state"}))
        assert bridge.handle({"op": "step", "t": 0.0, "dt": 1e-2})["ok"]
        assert bridge.handle({"op": "set_state", "state": blob}) == {"ok": True}
    finally:
        bridge.stop()


def test_a_field_that_became_non_finite_still_does_not_restore(diagnostics_model):
    """The documented refusal of a diverged model's snapshot is kept: a NaN
    in a field that started finite is refused by both paths alike."""
    gm, md = diagnostics_model
    sc = _sidecar(diagnostics_model)
    state, params = deserialize_fmu_state(sc.get_fmu_state(),
                                          expected_schema_token=md.instantiation_token,
                                          return_params=True)
    state["spring"]["velocity"] = np.float32(np.nan)
    snap = serialize_fmu_state(state=state, schema_token=md.instantiation_token, params=params)
    with pytest.raises(ValueError, match=r"spring\.velocity: value must be finite") as exc:
        sc.set_fmu_state(snap)
    bridge = FmuTcpBridge(_sidecar(diagnostics_model), md, master_dt=1e-2)
    try:
        with np.load(io.BytesIO(state_of(bridge.handle({"op": "get_state"}))),
                     allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        members["s/spring/velocity"] = np.float32(np.nan)
        buf = io.BytesIO()
        np.savez(buf, **members)
        reply = bridge.handle({"op": "set_state", "state": buf.getvalue()})
    finally:
        bridge.stop()
    assert reply == {"ok": False, "error": f"ValueError: {exc.value}"}, reply
