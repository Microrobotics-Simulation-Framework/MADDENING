"""``convergence_norm="interface"`` reads every internal edge as the step delivers it.

The step turns an edge into a boundary input by the edge rule
(``maddening.core.edge._delivered``): the interface mapping, with the
weights the step runs with, then the transform.  The interface norm, its
float floor and the report read "the values the group's internal edges
carry", and until this was fixed they applied the transform and left the
mapping out: on a mapped edge the criterion was taken on the source field,
a value the consuming node never sees.  A 9% change of what a selecting
mapping delivered read 0.0071 of the tolerance, and the group was reported
converged on it.

Every reading now goes through one generator
(``acceleration._interface_readings``).  Checked here, each against an
oracle that does not call it:

* the two norm functions on a mapped edge equal the same functions on a
  plain edge that carries the delivered value, for a mapping alone, a
  mapping then a transform, and a multi-component field;
* the weights are the ones the step ran with (``params["mappings"]``, which
  a caller may override for one step), not the mapping object's own;
* the dead band, the unevaluable-field rule and the float resolution are
  the delivered value's;
* in a compiled step, under both solvers and both sweeps, a one-pass group
  reports the change of what its edges delivered, in float64 closed form;
* the report's floor is the one the step measured with its own weights
  (the ``reading_floor`` slot), and only a group whose norm reads a mapped
  edge owns that slot.
"""

from __future__ import annotations

import math
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    coupling_residual_interface,
    residual_precision_floor,
    spectral_error_bound,
)
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core import coupling_domains as cd

F32 = jnp.float32
RTOL = 1e-2
EPS32 = float(np.finfo(np.float32).eps)


def _offset(v):
    """A unit conversion's offset, written for the value the mapping delivers."""
    return v + 273.15


def _last(v):
    """A selection of the mapped value: its last entry."""
    return v[-1:]


def _negate_twice(v):
    return -2.0 * v


#: The mapping delivers the field's second entry only.
SELECT = np.array([[0.0, 1.0]], np.float32)
#: Not square, not a selection: a field of 3 delivered as 2.
DENSE = np.array([[1.0, 2.0, 0.0], [0.5, 0.0, -3.0]], np.float32)

#: ``variant -> (source new, source old, H, transform)``.
VARIANTS = {
    "a selection": ([1000.0, 1.1], [1000.0, 1.0], SELECT, None),
    "a dense mapping": ([1.0, 2.0, 3.0], [1.5, 2.0, 2.0], DENSE, None),
    "a mapping then an offset": ([1.0, 2.0, 3.0], [1.5, 2.0, 2.0], DENSE, _offset),
    "a mapping then a selection of what it delivered": (
        [1.0, 2.0, 3.0], [1.5, 2.0, 2.0], DENSE, _last),
    "a mapping then a scale": ([1000.0, 1.1], [1000.0, 1.0], SELECT, _negate_twice),
    "a field of vectors": ([[1.0, -1.0], [2.0, 0.5], [3.0, 4.0]],
                           [[1.5, -1.0], [2.0, 0.25], [2.0, 4.5]], DENSE, None),
}


def _states(variant, weights=None):
    """``(new, old, mapped edge, plain edge)``: node ``a`` holds the source
    field and node ``d`` the value the edge delivers, computed here."""
    new, old, H, transform = VARIANTS[variant]
    H = H if weights is None else weights
    new, old = jnp.asarray(new, F32), jnp.asarray(old, F32)

    def delivered(v):
        v = jnp.asarray(H) @ v
        return v if transform is None else transform(v)

    s_new = {"a": {"x": new}, "d": {"x": delivered(new)}}
    s_old = {"a": {"x": old}, "d": {"x": delivered(old)}}
    mapped = EdgeSpec("a", "b", "x", "u", mapping=matrix_mapping(VARIANTS[variant][2]),
                      transform=transform)
    return s_new, s_old, mapped, EdgeSpec("d", "b", "x", "u")


def _closed_form(variant, weights=None, rtol=RTOL):
    """The interface norm of the delivered value in float64, as a
    ``pytest.approx`` that allows the float32 evaluation its rounding: a
    difference of two delivered values of magnitude ``ref`` is exact to
    ``eps * ref``, so to ``eps * ref / |difference|`` of itself."""
    new, old, H, transform = VARIANTS[variant]
    H = np.asarray(H if weights is None else weights, np.float64)
    t = (lambda v: v) if transform is None else transform
    a = np.asarray(t(H @ np.asarray(new, np.float64)))
    b = np.asarray(t(H @ np.asarray(old, np.float64)))
    ref = max(np.max(np.abs(a)), np.max(np.abs(b)))
    value = float(np.sqrt(np.mean((np.abs(a - b) / (rtol * ref)) ** 2)))
    conditioning = ref / float(np.max(np.abs(a - b)))
    return pytest.approx(value, rel=64 * EPS32 * conditioning)


# ---------------------------------------------------------------------------
# The two norm functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_norm_of_a_mapped_edge_is_the_norm_of_what_it_delivers(variant):
    """The mapped edge delivers exactly what edge ``d -> b`` carries, so the two
    norms are equal to the bit, and both are the float64 closed form.

    Before, the selection read 0.0071 where the delivered value moved by
    9.09 units of ``rtol``, and a transform written for the mapped shape was
    applied to the unmapped field.
    """
    s_new, s_old, mapped, plain = _states(variant)
    got = coupling_residual_interface(s_new, s_old, [mapped], atol=0.0, rtol=RTOL)
    want = coupling_residual_interface(s_new, s_old, [plain], atol=0.0, rtol=RTOL)
    assert float(got) == float(want), (variant, float(got), float(want))
    assert float(got) == _closed_form(variant)
    assert float(got) > 0.0


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_the_floor_of_a_mapped_edge_counts_what_it_delivers(variant):
    """``residual_precision_floor`` reads the same entries the norm does: on a
    mapped edge the delivered value's, as many as it has."""
    s_new, _s_old, mapped, plain = _states(variant)
    floors = [float(residual_precision_floor(s_new, ["a", "d"], "interface", 0.0, RTOL, [e]))
              for e in (mapped, plain)]
    assert floors[0] == floors[1] == pytest.approx(PRECISION_FLOOR_ULPS * EPS32 / RTOL)


def test_the_selection_no_longer_reads_on_the_source_fields_scale():
    """The audit's number: 0.1 of 1.1 against ``rtol = 1e-2`` is 9.09, not the
    0.0071 the source field's 1000 made of it."""
    s_new, s_old, mapped, _plain = _states("a selection")
    got = float(coupling_residual_interface(s_new, s_old, [mapped], atol=0.0, rtol=RTOL))
    assert got == pytest.approx(0.1 / (RTOL * 1.1), rel=1e-5)
    assert got > 1.0        # not converged: on the source's scale it read 0.0071


@pytest.mark.parametrize("variant", ["a selection", "a dense mapping",
                                     "a mapping then an offset"])
def test_the_weights_read_are_the_ones_passed_not_the_mapping_objects_own(variant):
    """``mappings`` is the step's ``params["mappings"]``: the norm and the floor
    read the edge with those weights, as the step applies it."""
    own = VARIANTS[variant][2]
    other = (3.0 * own[::-1, ::-1] + 0.25).astype(np.float32)
    s_new, s_old, mapped, plain = _states(variant, weights=other)
    weights = {mapped.key: {"H": jnp.asarray(other)}}
    got = float(coupling_residual_interface(s_new, s_old, [mapped], 0.0, RTOL, mappings=weights))
    want = float(coupling_residual_interface(s_new, s_old, [plain], 0.0, RTOL))
    assert got == want == _closed_form(variant, other)
    # The fixture premise: the mapping's own weights give another number.
    assert float(coupling_residual_interface(s_new, s_old, [mapped], 0.0, RTOL)) != got
    # An edge with no entry in ``mappings`` is read with its own weights.
    assert float(coupling_residual_interface(
        s_new, s_old, [mapped], 0.0, RTOL, mappings={})) == _closed_form(variant)


def test_the_dead_band_is_the_delivered_values():
    """``atol`` is compared with what the edge delivers, in its units.

    The source field is 1000 and the delivered entry 1.1: with ``atol = 10``
    the edge is inside the dead band and leaves the norm and the floor
    (before, the source's 1000 kept it in, measured against 1000).  A
    mapping that scales a small field up puts it back.
    """
    s_new, s_old, mapped, plain = _states("a selection")
    for edge in (mapped, plain):
        assert float(coupling_residual_interface(s_new, s_old, [edge], 10.0, RTOL)) == 0.0
        assert float(residual_precision_floor(
            s_new, ["a", "d"], "interface", 10.0, RTOL, [edge])) == 0.0
    up = {mapped.key: {"H": jnp.asarray(100.0 * SELECT)}}
    assert float(coupling_residual_interface(
        s_new, s_old, [mapped], 10.0, RTOL, mappings=up)) == pytest.approx(
            0.1 / (RTOL * 1.1), rel=1e-5)
    assert float(residual_precision_floor(
        s_new, ["a", "d"], "interface", 10.0, RTOL, [mapped], mappings=up)) > 0.0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_a_delivered_value_that_is_not_finite_fails_the_criterion(bad):
    """The unevaluable-field rule is applied to what the edge delivers.

    A non-finite weight, or a non-finite source entry -- even one the
    mapping weights with zero, since ``0 * inf`` is NaN and that NaN is
    what the target is handed -- reads ``inf``, never zero.
    """
    s_new, s_old, mapped, _plain = _states("a selection")
    weights = {mapped.key: {"H": jnp.asarray([[0.0, bad]], F32)}}
    assert math.isinf(float(coupling_residual_interface(
        s_new, s_old, [mapped], 0.0, RTOL, mappings=weights)))
    poisoned = {**s_new, "a": {"x": jnp.asarray([bad, 1.1], F32)}}
    assert math.isinf(float(coupling_residual_interface(poisoned, s_old, [mapped], 0.0, RTOL)))
    # The floor counts an unevaluable edge as read, as the norm does.
    assert float(residual_precision_floor(
        poisoned, ["a"], "interface", 0.0, RTOL, [mapped])) > 0.0


def test_an_integer_source_and_an_empty_delivery_carry_no_norm():
    """The edges the norm skips are skipped on a mapped edge too: a source
    field that is not floating, and a mapping that delivers no entries."""
    counter = {"a": {"x": jnp.asarray([3, 4], jnp.int32)}}
    moved = {"a": {"x": jnp.asarray([5, 9], jnp.int32)}}
    edge = EdgeSpec("a", "b", "x", "u", mapping=matrix_mapping(SELECT))
    assert float(coupling_residual_interface(moved, counter, [edge], 0.0, RTOL)) == 0.0
    assert float(residual_precision_floor(moved, ["a"], "interface", 0.0, RTOL, [edge])) == 0.0
    s_new, s_old, _mapped, _plain = _states("a selection")
    nothing = EdgeSpec("a", "b", "x", "u", mapping=matrix_mapping(np.zeros((0, 2), np.float32)))
    assert float(coupling_residual_interface(s_new, s_old, [nothing], 0.0, RTOL)) == 0.0
    assert float(residual_precision_floor(
        s_new, ["a"], "interface", 0.0, RTOL, [nothing])) == 0.0


def test_a_delivered_value_is_no_finer_than_the_field_it_was_computed_from():
    """The floor takes the coarser of the delivered dtype's eps and the source's.

    Under x64 a float64 mapping matrix applied to a float32 field delivers
    float64 numbers that carry float32 rounding: at float64's eps the floor
    read ``eps64 / eps32`` (1.9e-9) of what the source resolves.  A value
    delivered in its source's dtype, or a narrower one, keeps its own eps.
    """
    eps64 = float(np.finfo(np.float64).eps)
    eps16 = float(jnp.finfo(jnp.float16).eps)
    with cd.x64(True):
        field32 = {"a": {"x": jnp.asarray([1000.0, 1.1], jnp.float32)}}
        field64 = {"a": {"x": jnp.asarray([1000.0, 1.1], jnp.float64)}}
        wide = EdgeSpec("a", "b", "x", "u",
                        mapping=matrix_mapping(np.asarray(SELECT, np.float64)))
        same = EdgeSpec("a", "b", "x", "u", mapping=matrix_mapping(SELECT))
        narrow = EdgeSpec("a", "b", "x", "u", mapping=matrix_mapping(SELECT),
                          transform=lambda v: v.astype(jnp.float16))
        cast_up = EdgeSpec("a", "b", "x", "u", transform=lambda v: v.astype(jnp.float64))

        def floor(state, edge):
            return float(residual_precision_floor(state, ["a"], "interface", 0.0, 1.0, [edge]))

        assert jnp.asarray(wide.mapping.apply(field32["a"]["x"])).dtype == jnp.float64
        assert floor(field32, wide) == PRECISION_FLOOR_ULPS * EPS32
        assert floor(field32, cast_up) == PRECISION_FLOOR_ULPS * EPS32
        assert floor(field32, same) == PRECISION_FLOOR_ULPS * EPS32
        assert floor(field64, wide) == PRECISION_FLOOR_ULPS * eps64
        assert floor(field32, narrow) == PRECISION_FLOOR_ULPS * eps16


# ---------------------------------------------------------------------------
# In a compiled step
# ---------------------------------------------------------------------------


class _Lin(SimulationNode):
    """``u <- G @ inp + c`` with the constants baked in."""

    def __init__(self, name, G, c, u0):
        super().__init__(name, 1.0)
        self._G = np.asarray(G, np.float32)
        self._c = np.asarray(c, np.float32)
        self._u0 = np.asarray(u0, np.float32)

    def initial_state(self):
        return {"u": jnp.asarray(self._u0)}

    def boundary_input_spec(self):
        k = self._G.shape[1]
        return {"inp": BoundaryInputSpec(shape=(k,), dtype=F32, default=jnp.zeros(k, F32))}

    def update(self, state, boundary_inputs, dt):
        return {"u": jnp.asarray(self._G) @ boundary_inputs["inp"] + jnp.asarray(self._c)}

    def update_evaluations(self):
        return 1.0


KEY = "A+B"
#: ``A`` holds three entries and ``B`` two; ``A -> B`` through ``H_AB`` and
#: ``B -> A`` through ``H_BA`` then a scale by -2: a mapping alone, and a
#: mapping then a transform.
H_AB = np.array([[0.5, -1.0, 0.25], [0.0, 2.0, 1.0]], np.float32)
H_BA = np.array([[1.0, 0.5], [-0.25, 1.5]], np.float32)
G_A = np.array([[0.10, -0.05], [0.02, 0.08], [-0.06, 0.04]], np.float32)
G_B = np.array([[0.07, 0.03], [-0.04, 0.09]], np.float32)
C_A = np.array([1.0, -2.0, 0.5], np.float32)
C_B = np.array([0.75, 1.25], np.float32)
U_A0 = np.array([0.3, -0.7, 1.1], np.float32)
U_B0 = np.array([0.9, -0.4], np.float32)
#: Other weights for the same edges, handed to one step through ``params``.
H_AB_STEP = np.array([[2.0, 0.5, -1.0], [1.0, -0.5, 3.0]], np.float32)
H_BA_STEP = np.array([[-0.5, 2.0], [1.5, 0.25]], np.float32)


_GRAPHS: dict = {}


def _mapped_pair(**group) -> GraphManager:
    """The mapped pair under *group*, compiled once per module and handed
    back reset to its initial state."""
    key = ("mapped",) + tuple(sorted(group.items()))
    if key not in _GRAPHS:
        gm = GraphManager()
        gm.add_node(_Lin("A", G_A, C_A, U_A0))
        gm.add_node(_Lin("B", G_B, C_B, U_B0))
        gm.add_edge("A", "B", "u", "inp", mapping=matrix_mapping(H_AB))
        gm.add_edge("B", "A", "u", "inp", mapping=matrix_mapping(H_BA),
                    transform=_negate_twice)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # the deprecated "fori" says so
            gm.add_coupling_group(["A", "B"], **group)
            gm.compile()
        _GRAPHS[key] = gm
    _GRAPHS[key].reset_state()
    return _GRAPHS[key]


#: The group most tests step: four passes with the report's analysis on.
STANDARD = dict(max_iterations=4, convergence_norm="interface", rtol=1e-4, diagnostics=True)


def _with_weights(gm, h_ab, h_ba) -> dict:
    p = gm.params
    maps = {k: dict(v) for k, v in p["mappings"].items()}
    maps["A.u->B.inp"]["H"] = jnp.asarray(h_ab, F32)
    maps["B.u->A.inp"]["H"] = jnp.asarray(h_ba, F32)
    return {**p, "mappings": maps}


def _delivered_norm(new, old, h_ab, h_ba, rtol) -> float:
    """The interface norm between two states of the pair, in float64 NumPy."""
    terms = []
    for v_new, v_old in (
            (np.asarray(h_ab, np.float64) @ new["A"], np.asarray(h_ab, np.float64) @ old["A"]),
            (-2.0 * (np.asarray(h_ba, np.float64) @ new["B"]),
             -2.0 * (np.asarray(h_ba, np.float64) @ old["B"]))):
        ref = max(np.max(np.abs(v_new)), np.max(np.abs(v_old)))
        terms.append(np.abs(v_new - v_old) / (rtol * ref))
    return float(np.sqrt(np.mean(np.concatenate(terms) ** 2)))


def _source_norm(new, old, rtol) -> float:
    """What the norm read before: the source fields, the transform applied, no mapping."""
    terms = []
    for v_new, v_old in ((new["A"], old["A"]), (-2.0 * new["B"], -2.0 * old["B"])):
        ref = max(np.max(np.abs(v_new)), np.max(np.abs(v_old)))
        terms.append(np.abs(v_new - v_old) / (rtol * ref))
    return float(np.sqrt(np.mean(np.concatenate(terms) ** 2)))


def _fields(gm) -> dict:
    return {n: np.asarray(gm.get_node_state(n)["u"], np.float64) for n in ("A", "B")}


@pytest.mark.parametrize("mode", ["jacobi", "gauss-seidel"])
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_one_pass_reports_the_change_of_what_its_edges_delivered(solver, mode):
    """``max_iterations=1`` reports how far the one pass moved the norm's reading.

    The oracle is the float64 norm of the two delivered values between the
    start and the returned state, written out from ``H`` and the transform:
    it calls nothing of the library's.  With the weights replaced for the
    step through ``params``, the reading is taken with those.
    """
    rtol = 1e-3
    for h_ab, h_ba, override in ((H_AB, H_BA, False), (H_AB_STEP, H_BA_STEP, True)):
        gm = _mapped_pair(max_iterations=1, convergence_norm="interface", rtol=rtol,
                          solver=solver, diagnostics=True, iteration_mode=mode)
        before = _fields(gm)
        gm.step(params=_with_weights(gm, h_ab, h_ba) if override else None)
        after = _fields(gm)
        got = gm.coupling_diagnostics()[KEY]["residual"]
        want = _delivered_norm(after, before, h_ab, h_ba, rtol)
        assert got == pytest.approx(want, rel=256 * EPS32), (solver, mode, override)
        # The fixture premise: the source fields read otherwise, and so do
        # the mapping objects' own weights under the override.
        assert abs(_source_norm(after, before, rtol) - want) > 1e-2 * want
        if override:
            assert abs(_delivered_norm(after, before, H_AB, H_BA, rtol) - want) > 1e-2 * want


def test_a_group_iterates_until_what_its_mapped_edge_delivers_has_stopped_moving():
    """A converged verdict is about the delivered value, at the tolerance asked for.

    ``A`` holds ``[1000, s]`` and ``B`` reads ``s`` through a selecting
    mapping; ``s <- 0.9 s + 0.1`` round the loop, fixed point 1.  Measured
    on the source field, a change of ``s`` is divided by 1000: the group
    stopped after a few passes, far from its fixed point, and reported
    ``converged=True`` at ``rtol = 1e-3``.
    """
    gm = GraphManager()
    gm.add_node(_Lin("A", [[0.0], [0.9]], [1000.0, 0.1], [1000.0, 0.0]))
    gm.add_node(_Lin("B", [[1.0]], [0.0], [0.0]))
    gm.add_edge("A", "B", "u", "inp", mapping=matrix_mapping(SELECT))
    gm.add_edge("B", "A", "u", "inp")
    gm.add_coupling_group(["A", "B"], max_iterations=400, convergence_norm="interface",
                          rtol=1e-3, iteration_mode="jacobi")
    gm.compile()
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    s = float(gm.get_node_state("A")["u"][1])
    assert d["converged"], dict(d)
    assert abs(s - 1.0) < 2e-2, (s, dict(d))
    assert d["iterations"] > 20, dict(d)


# ---------------------------------------------------------------------------
# The report's floor: measured by the step, with the step's weights
# ---------------------------------------------------------------------------

SLOT = f"coupling_{KEY}_reading_floor"


def _banded_pair() -> GraphManager:
    """``A = [10, s]``, ``B = [t]`` with ``t`` inside the dead band; ``A -> B``
    through a selecting mapping, ``B -> A`` plain."""
    if "banded" not in _GRAPHS:
        gm = GraphManager()
        gm.add_node(_Lin("A", [[0.0], [50.0]], [10.0, 0.5], [10.0, 0.5]))
        gm.add_node(_Lin("B", [[0.001]], [0.0], [0.001]))
        gm.add_edge("A", "B", "u", "inp", mapping=matrix_mapping(SELECT))
        gm.add_edge("B", "A", "u", "inp")
        gm.add_coupling_group(["A", "B"], convergence_norm="interface", rtol=1e-4, atol=0.05,
                              max_iterations=4, diagnostics=True, iteration_mode="jacobi")
        gm.compile()
        _GRAPHS["banded"] = gm
    _GRAPHS["banded"].reset_state()
    return _GRAPHS["banded"]


def _floor_with(gm, mappings, evaluations=1.0) -> float:
    """The floor of the graph's state, by the function the report calls,
    with *mappings* as the weights."""
    group = gm._committed_coupling_groups[KEY]
    state = {n: gm._state[n] for n in ("A", "B")}
    return float(residual_precision_floor(
        state, ["A", "B"], "interface", group.atol, group.rtol, list(gm._edges),
        evaluations=evaluations, mappings=mappings))


def _bound_from_slots(gm, residual, floor) -> float:
    meta = gm._state["_meta"]
    return float(spectral_error_bound(
        residual, float(meta[f"coupling_{KEY}_rho_spectral"]),
        float(meta[f"coupling_{KEY}_spectral_residual"]),
        float(meta[f"coupling_{KEY}_spectral_amplification"]), floor=floor))


def test_the_reports_floor_is_taken_with_the_weights_the_step_ran_with():
    """A ``params`` override that moves a mapped edge into the dead band moves the floor.

    With the graph's own weights the mapped edge delivers ``s`` (about 0.5),
    above ``atol = 0.05``, and is the one edge the norm reads.  A step run
    with those weights scaled by 0.01 delivers 0.005: the edge is inside the
    dead band, the norm reads nothing, and the residual's floor is zero.
    The graph does not hold the override afterwards, so the step records
    the floor it measured and the report reads that.
    """
    gm = _banded_pair()
    gm.step()
    slot = float(gm._state["_meta"][SLOT])
    assert slot == _floor_with(gm, gm.params["mappings"]) == pytest.approx(
        PRECISION_FLOOR_ULPS * EPS32 / 1e-4)
    assert gm.coupling_diagnostics()[KEY]["spectral_error_bound"] > 0.0

    gm = _banded_pair()
    scaled = {"A.u->B.inp": {"H": jnp.asarray(0.01 * SELECT)}}
    gm.step(params={**gm.params, "mappings": scaled})
    slot = float(gm._state["_meta"][SLOT])
    assert slot == _floor_with(gm, scaled) == 0.0
    assert _floor_with(gm, gm.params["mappings"]) > 0.0        # the graph's own would say otherwise
    d = gm.coupling_diagnostics()[KEY]
    assert d["residual"] == 0.0 and d["converged"] and d["iterations"] == 1, dict(d)
    assert d["precision_limited"] is False, dict(d)
    want = _bound_from_slots(gm, d["residual"], 0.0)
    assert d["spectral_error_bound"] == want or (
        math.isnan(d["spectral_error_bound"]) and math.isnan(want)), (dict(d), want)


def test_the_recorded_floor_times_the_count_is_the_floor_the_report_adds():
    """The report multiplies the step's per-evaluation floor by the pass's
    evaluation count, as ``residual_precision_floor(evaluations=)`` does:
    the bound it reports is the documented formula on that floor, to the bit."""
    gm = _mapped_pair(**STANDARD)
    gm.step()
    d = gm.coupling_diagnostics()[KEY]
    evaluations = float(gm._state["_meta"][f"coupling_{KEY}_pass_evaluations"])
    floor = _floor_with(gm, gm.params["mappings"], evaluations)
    assert evaluations >= 2.0 and floor > 0.0      # Gauss-Seidel: a chain of two
    assert d["spectral_error_bound"] == _bound_from_slots(gm, d["residual"], floor), dict(d)


def _plain_pair(norm) -> GraphManager:
    """The same nodes joined without a mapping (``A -> B`` delivers two of
    ``A``'s three entries through a transform), a mapped edge to a reader
    outside the group."""
    gm = GraphManager()
    gm.add_node(_Lin("A", G_A, C_A, U_A0))
    gm.add_node(_Lin("B", G_B, C_B, U_B0))
    gm.add_node(_Lin("out", np.zeros((2, 2)), [0.0, 0.0], [0.0, 0.0]))
    gm.add_edge("A", "B", "u", "inp", transform=lambda v: v[:2])
    gm.add_edge("B", "A", "u", "inp")
    gm.add_edge("A", "out", "u", "inp", mapping=matrix_mapping(H_AB))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(["A", "B"], convergence_norm=norm, max_iterations=3)
        gm.compile()
    return gm


def test_only_a_group_whose_norm_reads_a_mapped_edge_records_its_floor():
    """The ``reading_floor`` slot belongs to the interface norm on a mapped
    internal edge; every other group's ``_meta`` is what it was.

    Seeded NaN ("not measured"), written by a step, back to NaN on
    ``reset_state()``, and restarted when the group is replaced.
    """
    for norm in ("l2", "mixed"):
        gm = _mapped_pair(max_iterations=3, convergence_norm=norm)
        assert SLOT not in gm._state["_meta"], norm
        gm.step()
        assert SLOT not in gm._state["_meta"], norm
    for norm in ("l2", "mixed", "interface"):
        gm = _plain_pair(norm)
        assert SLOT not in gm._state["_meta"], norm
    gm.step()       # the interface norm, a mapped edge that leaves the group
    assert SLOT not in gm._state["_meta"]

    gm = _mapped_pair(**STANDARD)
    assert math.isnan(float(gm._state["_meta"][SLOT]))
    assert KEY not in gm.coupling_diagnostics()
    gm.step()
    assert float(gm._state["_meta"][SLOT]) == pytest.approx(PRECISION_FLOOR_ULPS * EPS32 / 1e-4)
    gm.reset_state()
    assert math.isnan(float(gm._state["_meta"][SLOT]))


def test_a_replaced_group_does_not_inherit_the_recorded_floor():
    """A group replaced by another over the same nodes restarts its report
    slots, the recorded floor with them: it described a step the new group
    did not judge."""
    gm = GraphManager()
    gm.add_node(_Lin("A", G_A, C_A, U_A0))
    gm.add_node(_Lin("B", G_B, C_B, U_B0))
    gm.add_edge("A", "B", "u", "inp", mapping=matrix_mapping(H_AB))
    gm.add_edge("B", "A", "u", "inp", mapping=matrix_mapping(H_BA))
    gm.add_coupling_group(["A", "B"], max_iterations=1, convergence_norm="interface", rtol=1e-4)
    gm.compile()
    gm.step()
    assert math.isfinite(float(gm._state["_meta"][SLOT]))
    gm.remove_coupling_group(["A", "B"])
    gm.add_coupling_group(["A", "B"], max_iterations=1, convergence_norm="interface", rtol=1e-3)
    gm.compile()
    assert math.isnan(float(gm._state["_meta"][SLOT]))
    assert KEY not in gm.coupling_diagnostics()


def test_a_state_without_the_recorded_floor_is_read_with_the_graphs_own_weights():
    """A ``_meta`` whose slot was never written (a checkpoint from before the
    slot existed) still reports: the floor is taken from the returned state
    with ``gm.params``' weights, which is what a ``params=None`` step runs.

    The graph's own weights are edited to deliver the mapped edge inside the
    dead band (the mapping object still holds the matrix it was built
    with).  The step records a floor of zero; with the slot removed the
    report must still say so.  Read on the source field, or with the
    mapping object's weights, the edge is outside the band and the floor is
    not zero.
    """
    gm = _banded_pair()
    key = "A.u->B.inp"
    built_with = gm.params["mappings"][key]["H"]
    try:
        gm.params["mappings"][key]["H"] = jnp.asarray(0.01 * SELECT)
        gm.step()
        saved = gm._state
        assert float(saved["_meta"][SLOT]) == 0.0
        with_slot = dict(gm.coupling_diagnostics()[KEY])
        meta = dict(saved["_meta"])
        del meta[SLOT]
        gm._state = {**saved, "_meta": meta}
        without = dict(gm.coupling_diagnostics()[KEY])
        gm._state = saved
    finally:
        gm.params["mappings"][key]["H"] = built_with
    assert with_slot["precision_limited"] is False and with_slot["residual"] == 0.0, with_slot
    assert set(without) == set(with_slot)
    for name, value in with_slot.items():
        other = without[name]
        assert value == other or (isinstance(value, float) and math.isnan(value)
                                  and math.isnan(other)), (name, value, other)
    # The fixture premise: the mapping object's own weights keep the edge in the norm.
    assert _floor_with(gm, None) > 0.0


def _fresh_mapped_pair() -> GraphManager:
    """An uncached mapped pair without the report's analysis (cheap to trace)."""
    gm = GraphManager()
    gm.add_node(_Lin("A", G_A, C_A, U_A0))
    gm.add_node(_Lin("B", G_B, C_B, U_B0))
    gm.add_edge("A", "B", "u", "inp", mapping=matrix_mapping(H_AB))
    gm.add_edge("B", "A", "u", "inp", mapping=matrix_mapping(H_BA), transform=_negate_twice)
    gm.add_coupling_group(["A", "B"], max_iterations=4, convergence_norm="interface",
                          rtol=1e-4)
    gm.compile()
    return gm


def test_a_mapped_group_steps_on_one_trace_and_scans_as_it_steps():
    """The recorded floor is one more scalar in the scan carry, in the
    residual slot's dtype: the step is traced once whether or not the
    weights are passed for it, and ``run_scan`` of three steps leaves the
    state and every report slot ``step()`` three times leaves, to the bit."""
    stepped = _fresh_mapped_pair()
    stepped.step()
    stepped.step(params=_with_weights(stepped, H_AB_STEP * 0.1, H_BA_STEP * 0.1))
    stepped.step()
    assert stepped.trace_count == 1
    scanned = _fresh_mapped_pair()
    for gm in (stepped, scanned):
        gm.reset_state()
    for _ in range(3):
        stepped.step()
    scanned.run_scan(3)
    for name in ("A", "B"):
        a, b = (np.asarray(g._state[name]["u"]) for g in (stepped, scanned))
        assert a.tobytes() == b.tobytes(), (name, a, b)
    meta_a, meta_b = stepped._state["_meta"], scanned._state["_meta"]
    assert set(meta_a) == set(meta_b) and SLOT in meta_a
    for key in meta_a:
        a, b = np.asarray(meta_a[key]), np.asarray(meta_b[key])
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), (key, a, b)
    assert math.isfinite(float(meta_b[SLOT]))


def _bits_after_two_steps(gm) -> list:
    gm.step(params=_with_weights(gm, H_AB_STEP * 0.1, H_BA_STEP * 0.1))
    gm.step()
    meta = gm._state["_meta"]
    return ([np.asarray(gm._state[n]["u"]) for n in ("A", "B")]
            + [np.asarray(meta[f"coupling_{KEY}_{k}"])
               for k in ("iterations", "residual", "amplification", "reading_floor")])


def _assert_the_diagnostics_move_nothing(mode):
    off = _bits_after_two_steps(_mapped_pair(**{**STANDARD, "diagnostics": False,
                                                "iteration_mode": mode}))
    on = _bits_after_two_steps(_mapped_pair(**{**STANDARD, "iteration_mode": mode}))
    for a, b in zip(off, on):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), (mode, a, b)


def test_the_diagnostics_do_not_move_the_state_of_a_mapped_group():
    """The reading's spectral analysis reports on the step: the state, the
    loop's own slots and the recorded floor are the bits they are without it
    (Gauss-Seidel; a step with overridden weights, then one with the graph's)."""
    _assert_the_diagnostics_move_nothing("gauss-seidel")


# Per push: tests/core/test_coupling_interface_reading_is_what_the_edge_delivers.py::test_the_diagnostics_do_not_move_the_state_of_a_mapped_group
@pytest.mark.slow
def test_the_diagnostics_do_not_move_the_state_of_a_mapped_group_under_jacobi():
    """The same under the Jacobi sweep (another compiled pair)."""
    _assert_the_diagnostics_move_nothing("jacobi")


# Per push: tests/core/test_coupling_interface_reading_is_what_the_edge_delivers.py::test_the_recorded_floor_times_the_count_is_the_floor_the_report_adds
@pytest.mark.slow
def test_the_report_has_no_derivative_with_respect_to_the_mapping_weights():
    """The reading the report is taken on stops the gradient of the weights.

    A forward-mode derivative of the step with respect to a mapping matrix
    moves the state (the map reads the weights) and leaves every report
    slot's tangent at zero: the spectral analysis and the recorded floor
    describe the returned state, as they do with respect to the state
    itself and to the pass's other constants.
    """
    gm = _mapped_pair(**{**STANDARD, "iteration_mode": "jacobi"})
    p0, state0 = gm.params, gm._state

    def step(h_ab):
        maps = {**p0["mappings"], "A.u->B.inp": {"H": h_ab}}
        out = gm._raw_step_fn(state0, gm._default_external_inputs(), {**p0, "mappings": maps})
        meta = out["_meta"]
        return out["A"]["u"], {k: meta[f"coupling_{KEY}_{k}"] for k in (
            "rho_spectral", "spectral_residual", "spectral_amplification", "reading_floor")}

    h = jnp.asarray(H_AB)
    (_u, _slots), (du, dslots) = jax.jvp(step, (h,), (jnp.ones_like(h),))
    assert np.any(np.asarray(du) != 0.0)
    for name, tangent in dslots.items():
        assert float(tangent) == 0.0, (name, float(tangent))
