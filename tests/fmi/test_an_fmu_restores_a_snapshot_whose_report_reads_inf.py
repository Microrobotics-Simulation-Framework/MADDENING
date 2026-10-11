"""An FMU restores its own snapshot when a coupling group's report holds a
non-finite reading, and still refuses one in anything else.

A ``diagnostics=True`` group writes what its last solve measured into
``_meta``, and ``inf`` is an ordinary reading there on a model whose fields
are all finite: ``gradient_relative_error_bound`` is ``inf`` where the bound
does not certify (a group stopped at its iteration cap on a curved map),
and the experimental ``geometry_plane_limit`` and ``geometry_plane_margin``
are ``inf`` on every step of a group none of whose positions the pass reads
from the iterate.  Both restore paths hold a snapshot to finite values
(a diverged model does not restore), so ``FmuSidecar.set_fmu_state`` and the
TCP bridge's ``set_state`` refused the snapshot the FMU had just handed out:
``FMU state _meta.coupling_a+b_gradient_relative_error_bound: value must be
finite``.  An importer could not roll such a model back, while
``GraphManager.load_state`` restored the graph's own checkpoint.

The report slots restore with the non-finite values they hold; a step
writes every one of them and reads none.  Everything else is refused as
before, by both paths in the same words: a node's field, a group's warm
starts beside its report, an integer report slot, a finite value its slot
cannot hold.

Two models.  The paths are held per push on the pair with
``diagnostics=False``, whose report has two floating slots and compiles in
a second: a non-finite reading is put into its report through a restore,
and the snapshot the FMU then hands out is rolled back to.  The pair with
``diagnostics=True``, where a step at the iteration cap writes the ``inf``
itself, compiles the spectral analysis and is slow-marked.
"""

from __future__ import annotations

import io
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling._group_layout import _GROUP_META_SUFFIXES
from maddening.core.coupling._reports import _REPORT_SLOT_SUFFIXES
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.fmi.fmu_state import deserialize_fmu_state, serialize_fmu_state
from maddening.fmi.model_description import build_model_description
from maddening.fmi.sidecar import (
    FmuSidecar,
    SidecarConfig,
    _is_report_slot,
    _restored_leaf,
)
from maddening.fmi.tcp_bridge import FmuTcpBridge, state_of

DT = 0.01
Z = jnp.zeros(1, jnp.float32)
BOUND = "coupling_a+b_gradient_relative_error_bound"
RESIDUAL = "coupling_a+b_residual"
NON_FINITE = {"nan": np.nan, "inf": np.inf, "minus_inf": -np.inf}
#: The slots of a group that are state: its warm starts.
STATE_SUFFIXES = ("V", "W", "pred_count", "pred_0", "pred_1", "pred_2")


class Quad(SimulationNode):
    """``x <- c + g * u**2``"""

    def __init__(self, name, timestep, c=0.0, g=1.0):
        super().__init__(name, timestep, c=jnp.asarray(c, jnp.float32),
                         g=jnp.asarray(g, jnp.float32))

    def initial_state(self):
        return {"x": Z}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32, default=Z)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs.get("u", Z)
        return {"x": p["c"] + p["g"] * u * u}

    def update_evaluations(self):
        return 1


class Ident(SimulationNode):
    """``x <- k * u``"""

    def __init__(self, name, timestep, k=1.0):
        super().__init__(name, timestep, k=jnp.asarray(k, jnp.float32))

    def initial_state(self):
        return {"x": Z}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32, default=Z)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["k"] * boundary_inputs.get("u", Z)}

    def update_evaluations(self):
        return 1


def _model(*, slope=0.9, cap=3, g=0.5, **group):
    """``a.x <- c + g u**2`` and ``b.x <- u`` in one group, with slope
    *slope* at the fixed point ``x* = slope / (2 g)``.  Stopped at a cap of
    three passes the group has not converged and its gradient bound reads
    ``inf``; every field is finite."""
    xs = slope / (2 * g)
    gm = GraphManager()
    gm.add_node(Quad("a", DT, c=xs - g * xs * xs, g=g))
    gm.add_node(Ident("b", DT))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=cap, tolerance=1e-6,
                          **{"diagnostics": True, **group})
    gm.compile()
    return gm, build_model_description(gm, model_name="P")


@pytest.fixture(scope="module")
def model():
    """The pair without the spectral analysis: ``residual`` and
    ``amplification`` are its floating report slots."""
    return _model(diagnostics=False)


@pytest.fixture(scope="module")
def capped():
    """The pair as reported: ``diagnostics=True``, stopped at three passes."""
    return _model()


@pytest.fixture(scope="module")
def warm_started():
    """The same pair with the slots that are state beside the report: an
    IQN-IMVJ group's ``_V`` / ``_W`` and a quadratic predictor's history."""
    return _model(acceleration="iqn-imvj", predictor="quadratic", diagnostics=False)


def _sidecar(model) -> FmuSidecar:
    gm, md = model
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,  # noqa: SLF001
        initial_state={n: dict(f) for n, f in gm._state.items()},        # noqa: SLF001
        params={s: {o: dict(v) for o, v in owners.items()} for s, owners in gm.params.items()},
        param_specs=gm.param_specs(), fixed_params=md.fixed_parameters,
        input_resolver=gm._resolve_external_inputs))                     # noqa: SLF001


def _bits(state) -> dict:
    """Every leaf of a state as ``(dtype, shape, bytes)``."""
    return {(node, field): (np.asarray(v).dtype.str, np.asarray(v).shape, np.asarray(v).tobytes())
            for node, fields in state.items() for field, v in fields.items()}


class Sidecar:
    """The restore path of ``FmuSidecar.set_fmu_state``."""

    def __init__(self, model):
        self.sc, self.token = _sidecar(model), model[1].instantiation_token

    def step(self):
        self.sc.step(None)

    def state(self):
        return self.sc.state

    def snapshot(self):
        return self.sc.get_fmu_state()

    def restore(self, snapshot):
        self.sc.set_fmu_state(snapshot)

    def forged(self, node, field, value):
        state, params = deserialize_fmu_state(
            self.snapshot(), expected_schema_token=self.token, return_params=True)
        state[node][field] = value
        return serialize_fmu_state(state=state, schema_token=self.token, params=params)

    def refusal(self, snapshot) -> str:
        with pytest.raises(ValueError) as refused:
            self.restore(snapshot)
        return f"ValueError: {refused.value}"

    def close(self):
        pass


class Bridge:
    """The restore path of the TCP bridge's ``set_state`` (its handler,
    called in process: no socket is opened)."""

    def __init__(self, model):
        self.dt = float(model[0].timestep)
        self.bridge = FmuTcpBridge(_sidecar(model), model[1], master_dt=self.dt)
        self.t = 0.0

    def step(self):
        reply = self.bridge.handle({"op": "step", "t": self.t, "dt": self.dt})
        assert reply["ok"] is True, reply
        self.t = reply.get("t", self.t + self.dt)

    def state(self):
        return self.bridge._sidecar.state                             # noqa: SLF001

    def snapshot(self):
        return state_of(self.bridge.handle({"op": "get_state"})), self.t

    def restore(self, snapshot):
        blob, t = snapshot
        reply = self.bridge.handle({"op": "set_state", "state": blob})
        assert reply["ok"] is True, reply
        self.t = t

    def forged(self, node, field, value):
        blob, t = self.snapshot()
        with np.load(io.BytesIO(blob), allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        assert f"s/{node}/{field}" in members
        members[f"s/{node}/{field}"] = np.asarray(value)
        buf = io.BytesIO()
        np.savez(buf, **members)
        return buf.getvalue(), t

    def refusal(self, snapshot) -> str:
        reply = self.bridge.handle({"op": "set_state", "state": snapshot[0]})
        assert reply["ok"] is False, reply
        return reply["error"]

    def close(self):
        self.bridge.stop()


PATHS = {"sidecar": Sidecar, "bridge": Bridge}


@pytest.fixture(params=sorted(PATHS))
def path(request, model):
    door = PATHS[request.param](model)
    try:
        yield door
    finally:
        door.close()


@pytest.fixture(params=sorted(PATHS))
def capped_path(request, capped):
    door = PATHS[request.param](capped)
    try:
        yield door
    finally:
        door.close()


@pytest.fixture
def both(model):
    doors = [Sidecar(model), Bridge(model)]
    try:
        yield doors
    finally:
        for door in doors:
            door.close()


# ----------------------------------------------------- which slots are reports


def test_a_groups_report_slots_are_told_from_its_warm_starts_by_their_ending():
    """Every slot a group can hold in ``_meta`` is a report slot or state,
    and the name's ending says which, whatever the group's key spells."""
    assert set(_GROUP_META_SUFFIXES) == set(_REPORT_SLOT_SUFFIXES) | set(STATE_SUFFIXES)
    assert not set(_REPORT_SLOT_SUFFIXES) & set(STATE_SUFFIXES)
    keys = ["a+b", "a_residual+b", "coupling_a+b_iterations", "x_V+y_W", "a+b_pred_0",
            "gradient_relative_error_bound+geometry_gap"]
    for key in keys:
        for suffix in _REPORT_SLOT_SUFFIXES:
            assert _is_report_slot("_meta", f"coupling_{key}_{suffix}"), (key, suffix)
        for suffix in STATE_SUFFIXES:
            assert not _is_report_slot("_meta", f"coupling_{key}_{suffix}"), (key, suffix)
    # a node's field of the same name, and what is no group's slot
    assert not _is_report_slot("a", "coupling_a+b_residual")
    assert not _is_report_slot("_meta", "residual")
    assert not _is_report_slot("_meta", "step_count")
    assert not _is_report_slot("_meta", "x_residual")


@pytest.mark.parametrize("suffix", _REPORT_SLOT_SUFFIXES)
def test_every_floating_report_slot_restores_a_non_finite_reading(suffix):
    """Each suffix of the table, the experimental ``geometry_*`` among
    them, on the leaf check both paths share."""
    for dtype in (np.float32, np.float64):
        live = np.asarray(0.25, dtype)
        for value in NON_FINITE.values():
            got = _restored_leaf(np.asarray(value, np.float64), live,
                                 what=f"FMU state _meta.coupling_a+b_{suffix}", report=True)
            assert got.dtype == live.dtype and got.shape == ()
            assert got.tobytes() == np.asarray(value, dtype).tobytes()
            with pytest.raises(ValueError, match="value must be finite"):
                _restored_leaf(np.asarray(value, np.float64), live,
                               what=f"FMU state _meta.coupling_a+b_{suffix}")


def test_a_report_slot_is_still_held_to_every_other_check():
    live = np.asarray(0.25, np.float32)
    kw = {"what": "FMU state _meta.coupling_a+b_residual", "report": True}
    assert _restored_leaf(np.float64(0.5), live, **kw).tobytes() == np.float32(0.5).tobytes()
    with pytest.raises(ValueError, match="does not fit its type float32"):
        _restored_leaf(np.float64(1e39), live, **kw)
    with pytest.raises(ValueError, match="does not fit its type float32"):
        _restored_leaf(np.asarray([np.inf, 1e39]), np.zeros(2, np.float32), **kw)
    with pytest.raises(ValueError, match=r"shape \(2,\) != \(\)"):
        _restored_leaf(np.asarray([np.inf, 1.0]), live, **kw)
    with pytest.raises(ValueError, match="value must be a number, got a string"):
        _restored_leaf(np.asarray("inf"), live, **kw)
    # an integer report slot (the pass count) takes no NaN and no inf
    for value in NON_FINITE.values():
        with pytest.raises(ValueError, match="value must be finite"):
            _restored_leaf(np.float64(value), np.asarray(3, np.int32),
                           what="FMU state _meta.coupling_a+b_iterations", report=True)
    # a vector slot keeps its finite entries checked and its others as they are
    mixed = _restored_leaf(np.asarray([np.inf, 0.5, np.nan]), np.zeros(3, np.float32), **kw)
    assert mixed.tobytes() == np.asarray([np.inf, 0.5, np.nan], np.float32).tobytes()


# ------------------------------------------------------------ the two paths


def _fields_are_finite(state) -> bool:
    return all(bool(np.all(np.isfinite(np.asarray(v))))
               for node, fields in state.items() if node != "_meta" for v in fields.values())


def _floating_report_slots(state) -> list:
    return sorted(field for field, value in state["_meta"].items()
                  if _is_report_slot("_meta", field) and np.asarray(value).dtype.kind == "f")


def _rolled_back_to_the_bit(door) -> None:
    """Snapshot, step, restore, step, from where *door* stands: the
    restored state is the snapshot's and the second step is the first, in
    every leaf, ``_meta`` included."""
    held = _bits(door.state())
    snapshot = door.snapshot()
    door.step()
    stepped = _bits(door.state())
    assert stepped != held
    door.restore(snapshot)
    assert _bits(door.state()) == held
    door.step()
    assert _bits(door.state()) == stepped


def _every_floating_report_slot_takes_a_non_finite_reading(door, suffixes) -> None:
    door.step()
    slots = _floating_report_slots(door.state())
    assert {slot.removeprefix("coupling_a+b_") for slot in slots} == suffixes
    for slot in slots:
        dtype = np.asarray(door.state()["_meta"][slot]).dtype
        for value in NON_FINITE.values():
            door.restore(door.forged("_meta", slot, np.asarray(value, dtype)))
            got = np.asarray(door.state()["_meta"][slot])
            assert got.dtype == dtype and got.tobytes() == np.asarray(value, dtype).tobytes()
            door.step()                                   # and the model steps on from it
            assert _fields_are_finite(door.state())


@pytest.mark.parametrize("reading", sorted(NON_FINITE))
def test_a_snapshot_whose_report_holds_a_non_finite_reading_rolls_back_to_the_bit(path, reading):
    """The reading is put into the report as a restore puts it, and the
    snapshot the FMU then hands out is its own: it restores, and the step
    after it is the step that was taken."""
    path.step()
    value = np.float32(NON_FINITE[reading])
    path.restore(path.forged("_meta", RESIDUAL, value))
    assert np.asarray(path.state()["_meta"][RESIDUAL]).tobytes() == value.tobytes()
    assert _fields_are_finite(path.state())
    _rolled_back_to_the_bit(path)


def test_a_non_finite_reading_forged_into_any_floating_report_slot_restores(path):
    _every_floating_report_slot_takes_a_non_finite_reading(path, {"residual", "amplification"})


# Per push: tests/fmi/test_an_fmu_restores_a_snapshot_whose_report_reads_inf.py::test_a_snapshot_whose_report_holds_a_non_finite_reading_rolls_back_to_the_bit
@pytest.mark.slow
def test_a_group_stopped_at_its_cap_writes_inf_and_its_snapshot_rolls_back_to_the_bit(capped_path):
    """The reported case: the step itself writes the ``inf``, on a model
    with no non-finite field."""
    capped_path.step()
    assert float(capped_path.state()["_meta"][BOUND]) == np.inf
    assert _fields_are_finite(capped_path.state())
    _rolled_back_to_the_bit(capped_path)


# Per push: tests/fmi/test_an_fmu_restores_a_snapshot_whose_report_reads_inf.py::test_a_snapshot_whose_report_holds_a_non_finite_reading_rolls_back_to_the_bit
@pytest.mark.slow
def test_a_rollback_at_every_step_of_a_capped_run_changes_nothing(capped_path, capped):
    """What an importer does that rejects steps: the run with a snapshot
    taken and restored after every step ends where the run without one
    does.  The report reads ``inf`` on the first steps and a number once
    the iterate is close enough for the bound to certify."""
    plain = Sidecar(capped)
    readings = []
    for _ in range(4):
        plain.step()
        capped_path.step()
        readings.append(float(capped_path.state()["_meta"][BOUND]))
        capped_path.restore(capped_path.snapshot())
    assert readings[0] == np.inf and np.isfinite(readings[-1]), readings
    assert _bits(capped_path.state()) == _bits(plain.state())


# Per push: tests/fmi/test_an_fmu_restores_a_snapshot_whose_report_reads_inf.py::test_a_non_finite_reading_forged_into_any_floating_report_slot_restores
@pytest.mark.slow
def test_every_floating_slot_of_a_diagnosed_groups_report_takes_a_non_finite_reading(capped_path):
    _every_floating_report_slot_takes_a_non_finite_reading(capped_path, {
        "residual", "amplification", "rho_spectral", "spectral_residual",
        "spectral_amplification", "gradient_relative_error_bound", "pass_evaluations"})


@pytest.fixture(scope="module")
def constant_positions():
    """A group under the experimental geometry diagnostics none of whose
    positions the pass reads from the iterate (both anchors on the target):
    every step writes ``inf`` into ``geometry_plane_limit`` and
    ``geometry_plane_margin``, since no position can reach a lattice plane."""
    from tests.property import plane_sides as ps

    cfg = ps.rounded(ps.drawn(ps.Cfg(anchors=("target", "target")), np.random.default_rng(0)))
    gm = ps.build(cfg)
    gm.compile()
    ps.load(cfg, gm)
    return gm, build_model_description(gm, model_name="G")


# Per push: tests/fmi/test_an_fmu_restores_a_snapshot_whose_report_reads_inf.py::test_every_floating_report_slot_restores_a_non_finite_reading
@pytest.mark.slow
@pytest.mark.parametrize("door", sorted(PATHS))
def test_the_geometry_slots_read_inf_on_an_ordinary_step_and_the_snapshot_rolls_back(
        constant_positions, door):
    """The experimental slots had the defect on every step, not only at an
    iteration cap: this group converges, and its snapshot was refused at
    ``_meta.coupling_F+P_geometry_plane_limit``."""
    path = PATHS[door](constant_positions)
    try:
        path.step()
        meta = path.state()["_meta"]
        assert float(meta["coupling_F+P_geometry_plane_limit"]) == np.inf
        assert float(meta["coupling_F+P_geometry_plane_margin"]) == np.inf
        assert np.isfinite(float(meta["coupling_F+P_geometry_gap"]))
        assert _fields_are_finite(path.state())
        _rolled_back_to_the_bit(path)
    finally:
        path.close()


def _refused_alike(doors, node, field, value) -> str:
    """The one refusal both paths give a snapshot with *value* forged into
    ``node.field``; nothing is written by either."""
    said = []
    for door in doors:
        before = _bits(door.state())
        said.append(door.refusal(door.forged(node, field, value)))
        assert _bits(door.state()) == before
    assert said[0] == said[1], said
    return said[0]


def test_a_non_finite_value_in_anything_that_is_not_a_report_slot_is_still_refused(both):
    """A field of a node and the ``_meta`` entries that are no group's
    report, on the model whose own snapshot now restores."""
    for door in both:
        door.step()
    state = both[0].state()
    leaves = [(node, field) for node, fields in state.items() for field, value in fields.items()
              if not _is_report_slot(node, field)]
    assert {("a", "x"), ("b", "x")} <= set(leaves)
    floating = [(n, f) for n, f in leaves if np.asarray(state[n][f]).dtype.kind == "f"]
    assert {("a", "x"), ("b", "x")} <= set(floating)
    for node, field in leaves:
        live = np.asarray(state[node][field])
        for value in NON_FINITE.values():
            said = _refused_alike(both, node, field, np.full(live.shape, value, np.float64))
            assert said == f"ValueError: FMU state {node}.{field}: value must be finite"


def test_an_integer_report_slot_takes_no_non_finite_value_on_either_path(both):
    for door in both:
        door.step()
    for value in NON_FINITE.values():
        said = _refused_alike(both, "_meta", "coupling_a+b_iterations", np.float64(value))
        assert said == "ValueError: FMU state _meta.coupling_a+b_iterations: value must be finite"


def test_a_groups_warm_starts_are_state_and_take_no_non_finite_value(warm_started):
    """``_V``, ``_W`` and the predictor's history sit beside the report in
    ``_meta`` and are what the next step continues from."""
    doors = [Sidecar(warm_started), Bridge(warm_started)]
    try:
        for door in doors:
            door.step()
        meta = doors[0].state()["_meta"]
        warm = sorted(field for field in meta
                      if field.startswith("coupling_") and not _is_report_slot("_meta", field))
        assert {w.removeprefix("coupling_a+b_") for w in warm} == set(STATE_SUFFIXES)
        for field in warm:
            live = np.asarray(meta[field])
            if live.dtype.kind != "f":
                continue
            for value in NON_FINITE.values():
                said = _refused_alike(doors, "_meta", field,
                                      np.full(live.shape, value, live.dtype))
                assert said == f"ValueError: FMU state _meta.{field}: value must be finite"
        # and its own snapshot restores, report slots and all
        for door in doors:
            door.restore(door.snapshot())
    finally:
        for door in doors:
            door.close()
