"""Under ``"l2"`` and ``"mixed"`` the float floor counts the rounding of
the weights a geometry-dependent mapping forms (experimental).

The ``multilinear_grid`` kernel forms the lattice coordinate
``(x - origin) / spacing`` in the positions' dtype, so a weight is
resolved to ``eps`` times that coordinate: float32 markers 2000 spacings
from their grid's first point get weights off by 1.35e-4 of a cell, the
same at every pass, and the group converges to the fixed point of a
slightly different gather.  Nothing counted that: a pair whose positions
are constants of the pass (the documented way to keep the usable flags)
reported ``spectral_usable`` and ``gradient_bound_usable`` beside a bound
at 0.19 (1000 spacings in) and 0.08 (2000) of the distance to the fixed
point of the positions as stored, 0.03 with members that declare one
evaluation, and ``compile()`` said nothing (MADD-ANO-261).

**The pair** (float32): ``G`` holds a field on a 1-D grid of
``2 * into + 1`` points, spacing 0.75, its first point ``into`` spacings
below the coordinates' zero; ``P`` holds four markers within four
spacings of zero.  ``G -> P`` is a ``multilinear_grid`` gather anchored
at ``P.pos`` (a target anchor: the positions are constants of the pass);
``P -> G`` is a plain edge.  ``P.x <- bP + aP * u`` and ``G.x <- bG +
aG * mean(u)``, so every grid entry answers the markers and the pooled
norm is not diluted by entries that never move.

**The reference** is the fixed point of the same pair with the gather
taken in exact arithmetic at the positions as stored
(``tests/core/multilinear_reference.py``: fractions, no code shared with
the library) and the constants as float32 holds them: one dense float64
solve.  Distances are in the group's own norm at the returned state, the
unit ``spectral_error_bound`` is reported in.

**The scatter pair** (the row guard, MADD-ANO-257): ``m`` markers in one
cell of a four-point grid, a conservative scatter anchored at the grid
node's own copy of the positions (a target anchor again), a uniform
field: one grid entry adds up ``m`` terms in order.
"""

from __future__ import annotations

import functools
import gc
import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _group_layout, _interface_plan, reason_codes
from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    _kernel_rounding,
    _kernel_rounding_eps,
    residual_precision_floor,
)
from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.coupling.group import CouplingGroup
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core import multilinear_reference as reference
from tests.core.coupling_reason_rules import assert_reason_rules
from tests.sparse_mapping_support import x64

KEY = "G+P"
M = 4
SPACING = 0.75
EPS32 = float(np.finfo(np.float32).eps)
ADVISORY = "cannot be resolved to this tolerance"
#: The audit's two tolerances: ``rtol`` under "mixed", ``tolerance`` under "l2".
TOLERANCE = {"mixed": 1e-5, "l2": 3e-5}
A_P, A_G = 0.5, 0.8


class Markers(SimulationNode):
    """``x <- b + a * u``; ``pos <- pos + dt * vel`` (a constant of the pass)."""

    def __init__(self, name, b, pos, pos_dtype="float32", vel=0.0):
        super().__init__(name, 1.0)
        self._b, self._pos = np.asarray(b, np.float64), np.asarray(pos, np.float64)
        self._pos_dtype, self._vel = jnp.dtype(pos_dtype), float(vel)

    def initial_state(self):
        return {"x": jnp.zeros(self._b.shape, jnp.float32),
                "pos": jnp.asarray(self._pos, self._pos_dtype)}

    def boundary_input_spec(self):
        n = self._b.shape[0]
        return {"u": BoundaryInputSpec(shape=(n,), dtype=jnp.float32,
                                       default=jnp.zeros(n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": jnp.asarray(self._b, jnp.float32)
                + jnp.float32(A_P) * boundary_inputs["u"],
                "pos": state["pos"] + jnp.asarray(dt * self._vel, self._pos_dtype)}


class Field(SimulationNode):
    """``x <- b + a * mean(u)``: every grid entry answers the markers."""

    def __init__(self, name, a, b, n_in):
        super().__init__(name, 1.0)
        self._a, self._b, self._n_in = np.asarray(a, np.float64), np.asarray(b, np.float64), n_in

    def initial_state(self):
        return {"x": jnp.zeros(self._b.shape, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n_in,), dtype=jnp.float32,
                                       default=jnp.zeros(self._n_in, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": jnp.asarray(self._b, jnp.float32)
                + jnp.asarray(self._a, jnp.float32) * jnp.mean(boundary_inputs["u"])}


def _declaring(cls):
    """*cls* with ``update_evaluations() == 1`` declared."""
    return type(cls.__name__ + "Declared", (cls,), {"update_evaluations": lambda self: 1})


def _knobs(norm, schedule, diagnostics=True, tolerance=None):
    tolerance = TOLERANCE[norm] if tolerance is None else tolerance
    knobs = dict(convergence_norm=norm, max_iterations=400, iteration_mode=schedule,
                 solver="ift", diagnostics=diagnostics)
    knobs["tolerance" if norm == "l2" else "rtol"] = tolerance
    return knobs


def _held(values):
    """*values* as float32 holds them, widened exactly."""
    return np.asarray(np.asarray(values, np.float32), np.float64)


def _distance(norm, tolerance, fields):
    """The group's norm of ``returned - reference`` over *fields*
    (``[(returned, reference), ...]``, every floating field of the group;
    a position field counts its entries and adds nothing)."""
    total, count = 0.0, 0
    for returned, exact in fields:
        returned = np.asarray(returned, np.float64).ravel()
        total += float(np.sum(((returned - np.asarray(exact, np.float64).ravel())
                               / np.max(np.abs(returned))) ** 2))
        count += returned.size
    return math.sqrt(total) if norm == "l2" else math.sqrt(total / count) / tolerance


def gather_pair(into, norm, schedule, *, declared=False, seed=0, pos_dtype="float32",
                spacing=SPACING, centred=True, diagnostics=True, vel=0.0, extra_cells=0,
                tolerance=None):
    """The pair of the module's docstring, its markers ``into`` spacings
    from the grid's first point; uncompiled, with what the reference needs."""
    rng = np.random.default_rng(seed)
    n_grid = 2 * into + 1 + extra_cells
    origin = -into * spacing if centred else 0.0
    cells, frac = np.arange(M), rng.uniform(0.15, 0.85, M)
    pos = (origin + (into + cells + frac) * spacing).reshape(M, 1)
    b_grid = np.where(np.arange(n_grid) % 2 == 0, 1.0, 3.0)   # varies across a cell by its own size
    b_markers = rng.uniform(0.5, 1.5, M)
    field, markers = (_declaring(Field), _declaring(Markers)) if declared else (Field, Markers)
    gm = GraphManager()
    gm.add_node(field("G", A_G * b_grid, b_grid, M))
    gm.add_node(markers("P", b_markers, pos, pos_dtype, vel))
    gm.add_edge("G", "P", "x", "u",
                mapping=multilinear_grid_mapping([origin], [spacing], [n_grid], n_points=M,
                                                 mode="consistent"),
                geometry=("target", "pos"))
    gm.add_edge("P", "G", "x", "u")
    gm.add_coupling_group(["G", "P"], **_knobs(norm, schedule, diagnostics, tolerance))
    return gm, dict(origin=origin, spacing=spacing, n_grid=n_grid, b_grid=b_grid,
                    b_markers=b_markers)


def _compiled(gm):
    """``compile()``'s warnings, as texts."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
    return [str(w.message) for w in caught]


def _report(gm):
    report = dict(gm.coupling_diagnostics()[KEY])
    assert_reason_rules(report, KEY)
    return report


def _gather_fixed_point(pos_held, origin, spacing, n_grid, b_grid, b_markers):
    """The pair's fixed point with the gather in exact arithmetic at
    *pos_held* and the constants as float32 holds them."""
    W = reference.dense_matrix(reference.Grid((origin,), (spacing,), (n_grid,)), pos_held)
    a_markers, a_grid = _held(A_P), _held(A_G * b_grid)
    b_markers, b_grid = _held(b_markers), _held(b_grid)
    # xG = bG + aG mean(xP);  xP = bP + aP (W bG + (W aG) mean(xP))
    K = np.eye(M) - a_markers * np.outer(W @ a_grid, np.full(M, 1.0 / M))
    markers = np.linalg.solve(K, b_markers + a_markers * (W @ b_grid))
    return b_grid + a_grid * np.mean(markers), markers


@functools.lru_cache(maxsize=None)
def solved(into, norm, schedule, declared=False, seed=0, pos_dtype="float32",
           spacing=SPACING, centred=True, diagnostics=True):
    """``(report, distance, advisories)`` of one cell (numbers only: no
    compiled graph is kept).  Members that declare are stepped three
    times, as the audit's were, so that they stall at their floor."""
    with x64(pos_dtype == "float64"):
        gm, built = gather_pair(into, norm, schedule, declared=declared, seed=seed,
                                pos_dtype=pos_dtype, spacing=spacing, centred=centred,
                                diagnostics=diagnostics)
        advisories = [text for text in _compiled(gm) if ADVISORY in text]
        for _ in range(3 if declared else 1):
            gm.step()
        report = _report(gm)
        state = {name: {field: np.asarray(value, np.float64)
                        for field, value in gm.get_node_state(name).items()}
                 for name in ("G", "P")}
    grid, markers = _gather_fixed_point(state["P"]["pos"], **built)
    distance = _distance(norm, TOLERANCE[norm], [
        (state["G"]["x"], grid), (state["P"]["x"], markers),
        (state["P"]["pos"], state["P"]["pos"])])
    return report, distance, tuple(advisories)


def _usable(report) -> bool:
    return bool(report["spectral_usable"] or report["gradient_bound_usable"])


# --------------------------------------------------------------- the count


def test_the_count_is_one_rounding_of_the_lattice_coordinate_at_the_positions_the_kernel_reads():
    """``_kernel_rounding`` gives, per geometry edge, ``eps`` of the
    positions' dtype times the largest lattice coordinate its kernel
    reads, and ``_kernel_rounding_eps`` that over ``PRECISION_FLOOR_ULPS``;
    a group with no geometry edge has none (nothing is built for it)."""
    gm, built = gather_pair(1000, "mixed", "gauss-seidel")
    _compiled(gm)
    state = {name: gm.get_node_state(name) for name in ("G", "P")}
    edges = list(gm._edges)
    lattice = (np.asarray(state["P"]["pos"], np.float64) - built["origin"]) / SPACING
    entries = _kernel_rounding(edges, state, pre_step=state)
    assert [(record.key, holder) for record, holder, _reach, _res in entries] == [
        ("G.x->P.u", ("P", "pos"))]
    reach, resolution = entries[0][2], float(entries[0][3])
    np.testing.assert_allclose(np.asarray(reach, np.float64), lattice, rtol=2e-7)
    assert resolution == pytest.approx(EPS32 * lattice.max(), rel=1e-6)
    assert float(_kernel_rounding_eps(edges, state, pre_step=state)) == pytest.approx(
        resolution / PRECISION_FLOOR_ULPS, rel=1e-6)
    # A target anchor reads the target's pre-step positions: without them
    # there is nothing to read, and the function says so.
    with pytest.raises(ValueError, match="only the step that solved the group holds"):
        _kernel_rounding(edges, state)
    # No geometry edge: no entry, no eps, the floor's Python-float path.
    plain = [EdgeSpec("G", "P", "x", "u", mapping=matrix_mapping(jnp.zeros((M, 2001)))),
             EdgeSpec("P", "G", "x", "u")]
    assert _kernel_rounding(plain, state) == []
    assert _kernel_rounding_eps(plain, state) is None
    assert _kernel_rounding_eps((), state) is None


@pytest.mark.parametrize("norm", ["mixed", "l2"])
def test_the_floor_takes_every_value_field_no_finer_than_that_and_a_position_at_its_own(norm):
    """``residual_precision_floor`` under "l2" and "mixed": every value
    entry at the larger of the group's eps and one rounding of the
    lattice coordinate over ``PRECISION_FLOOR_ULPS``; the position field
    at its own dtype's eps.  It only raises: within
    ``PRECISION_FLOOR_ULPS`` spacings the floor is the one it was."""
    floors = {}
    for into in (0, 300):
        gm, built = gather_pair(into, norm, "gauss-seidel", extra_cells=8)
        _compiled(gm)
        state = {name: gm.get_node_state(name) for name in ("G", "P")}
        state["G"] = {"x": jnp.asarray(built["b_grid"], jnp.float32)}
        state["P"] = {**state["P"], "x": jnp.asarray(built["b_markers"], jnp.float32)}
        reach = float(np.max((np.asarray(state["P"]["pos"], np.float64) - built["origin"])
                             / SPACING))
        value_eps = max(EPS32, EPS32 * reach / PRECISION_FLOOR_ULPS)
        n_values, n_positions = built["n_grid"] + M, M
        tolerance = TOLERANCE[norm]
        sum_sq = n_values * value_eps ** 2 + n_positions * EPS32 ** 2
        expected = PRECISION_FLOOR_ULPS * 2.0 * (
            math.sqrt(sum_sq) if norm == "l2"
            else math.sqrt(sum_sq / (n_values + n_positions)) / tolerance)
        floors[into] = float(residual_precision_floor(
            state, ["G", "P"], norm, 0.0, tolerance, list(gm._edges), evaluations=2.0,
            pre_step=state))
        assert floors[into] == pytest.approx(expected, rel=1e-5), (into, reach)
        if into == 0:
            # (reach < PRECISION_FLOOR_ULPS: the count the floor always was)
            assert reach < PRECISION_FLOOR_ULPS and value_eps == EPS32
    assert floors[300] > 70 * floors[0]


def test_a_group_with_a_target_anchored_geometry_edge_owns_the_floor_slot_under_every_norm():
    """Static: the step records the floor (``reading_floor``) of a group
    under "l2" or "mixed" exactly where a geometry edge of a kind with a
    length scale is anchored at its target, whose pre-step positions the
    returned state does not hold; a source anchor, a static mapping and
    a plain pair own no slot under those norms (their step is the one it
    was)."""
    gm, _built = gather_pair(3, "mixed", "gauss-seidel")
    gm.compile()
    nodes, state = gm._nodes, gm._state
    gather = multilinear_grid_mapping([-2.25], [SPACING], [7], n_points=M, mode="consistent")
    scatter = multilinear_grid_mapping([-2.25], [SPACING], [7], n_points=M,
                                       mode="conservative")
    cases = {
        "gather, target anchor": ([
            EdgeSpec("G", "P", "x", "u", mapping=gather, geometry=("target", "pos")),
            EdgeSpec("P", "G", "x", "u")], True),
        "scatter, source anchor": ([
            EdgeSpec("G", "P", "x", "u"),
            EdgeSpec("P", "G", "x", "u", mapping=scatter, geometry=("source", "pos"))], False),
        "static mapping": ([
            EdgeSpec("G", "P", "x", "u", mapping=matrix_mapping(jnp.zeros((M, 7)))),
            EdgeSpec("P", "G", "x", "u")], False),
        "plain": ([EdgeSpec("G", "P", "x", "u"), EdgeSpec("P", "G", "x", "u")], False),
    }
    for name, (edges, owns) in cases.items():
        plan = _interface_plan.interface_plan({"G", "P"}, edges, ["G", "P"], state, nodes)
        assert plan.kernel_rounds_beyond_the_state() is owns, name
        for norm in ("l2", "mixed"):
            group = CouplingGroup(nodes=["G", "P"], convergence_norm=norm, solver="ift")
            assert _group_layout._reads_mapping_weights(group, plan) is owns, (name, norm)
            assert _group_layout._floor_needs_the_step(group, edges) is owns, (name, norm)


# ---------------------------------------------- the report, far into a grid


def test_far_into_its_grid_an_ordinary_pair_is_at_its_floor_and_no_flag_stands():
    """2000 spacings in (the audit's cell: both flags set beside a bound
    at 0.08 of the distance, ``precision_limited=False``): the residual
    the loop accepts is under the floor, so the report says
    ``precision_limited=True`` and withdraws both flags with the code of
    a residual at its float floor.  The loop is not asked anything new:
    ``converged`` stands."""
    report, distance, advisories = solved(2000, "mixed", "gauss-seidel")
    assert report["converged"] and report["precision_limited"]
    assert not report["spectral_usable"] and not report["gradient_bound_usable"]
    assert report["reason_codes"]["spectral_usable"] == [reason_codes.AT_FLOAT_FLOOR]
    # One rounding of the lattice coordinate per evaluation, two a pass:
    # 2 * eps * 2003.x / rtol, pooled with four positions at their own eps.
    assert report["residual_precision_floor"] == pytest.approx(
        2 * EPS32 * 2003.5 / 1e-5, rel=2e-3)
    # The state is tolerances from the fixed point of the positions as
    # stored (the audit's finding), and the bound, as reported, covers it.
    assert distance > 2.0
    assert report["spectral_error_bound"] > distance
    assert len(advisories) == 1


@pytest.mark.parametrize("norm,into", [("l2", 2000), ("mixed", 3000)])
def test_far_into_its_grid_members_that_declare_keep_their_flags_on_a_bound_that_holds(
        norm, into):
    """Members that declare ``update_evaluations()`` keep their flags at
    the floor, where the floor is the bound: 2000 spacings in under "l2"
    it read 0.03 of the distance; 3000 in, the geometry self-check
    withheld the report, its probe being under the kernel's rounding (the
    step's own per-entry resolution takes the count too: with the count
    in the report alone this cell has no number).  With the count the
    flags stand on a bound over the distance."""
    report, distance, _advisories = solved(into, norm, "gauss-seidel", declared=True)
    assert report["converged"] and report["precision_limited"]
    assert report["spectral_usable"] and report["gradient_bound_usable"]
    assert "not_usable_reason" not in report
    assert report["spectral_error_bound"] > 4 * distance > 0


def test_three_spacings_into_its_grid_the_pair_keeps_both_flags():
    """The ordinary case is not charged: markers 3 to 7 spacings from the
    grid's first point at the same tolerance keep both flags, the report
    not at its floor, on a bound that holds (counted
    ``PRECISION_FLOOR_ULPS`` times per evaluation instead of once, this
    report read ``precision_limited=True`` and lost both)."""
    report, distance, advisories = solved(3, "mixed", "gauss-seidel")
    assert report["converged"] and not report["precision_limited"]
    assert report["spectral_usable"] and report["gradient_bound_usable"]
    assert "not_usable_reason" not in report
    assert report["spectral_error_bound"] > distance > 0
    # The floor is the fields' own count up to the factor reach / 4 < 2.
    base = PRECISION_FLOOR_ULPS * 2 * EPS32 / 1e-5
    assert base <= report["residual_precision_floor"] < 1.75 * base
    assert advisories == ()


def test_float64_positions_beside_float32_fields_keep_the_flags_far_into_the_grid():
    """The way out the advisory names: positions held in float64 are
    rounded at float64's eps, 2000 spacings in as anywhere, and the pair
    keeps both flags on a bound that holds, with no advisory."""
    report, distance, advisories = solved(2000, "mixed", "gauss-seidel", pos_dtype="float64")
    assert report["converged"] and not report["precision_limited"]
    assert report["spectral_usable"] and report["gradient_bound_usable"]
    assert report["spectral_error_bound"] > distance > 0
    assert report["residual_precision_floor"] == pytest.approx(
        PRECISION_FLOOR_ULPS * 2 * EPS32 / 1e-5, rel=2e-3)
    assert advisories == ()


# Per push: tests/core/test_the_float_floor_counts_a_geometry_kernels_rounding.py::test_far_into_its_grid_an_ordinary_pair_is_at_its_floor_and_no_flag_stands
@pytest.mark.slow
def test_a_power_of_two_spacing_from_zero_is_counted_like_any_other():
    """The rule's price: with the grid's first point at zero and a
    power-of-two spacing the kernel's quotient is exact and the bound
    held uncounted (3.1 to 4.3 times the distance), but the count reads
    positions, not whether a spacing rounds: 2000 spacings in the report
    is at its floor and its flags are withdrawn all the same."""
    report, distance, advisories = solved(2000, "mixed", "gauss-seidel", spacing=0.5,
                                          centred=False)
    assert report["converged"] and report["precision_limited"]
    assert not _usable(report)
    assert report["spectral_error_bound"] > distance
    assert len(advisories) == 1


def test_the_floor_and_the_verdict_are_the_same_with_diagnostics_off():
    """``diagnostics=False`` reports the same ``precision_limited``,
    ``converged`` and ``iterations``, and the same floor up to the
    measured count of evaluations (which diagnostics add and which is
    never below the structural one)."""
    on, _distance_on, _ = solved(2000, "mixed", "gauss-seidel")
    off, _distance_off, _ = solved(2000, "mixed", "gauss-seidel", diagnostics=False)
    for key in ("precision_limited", "converged", "iterations", "residual"):
        assert on[key] == off[key], key
    assert off["residual_precision_floor"] == pytest.approx(
        2 * EPS32 * 2003.5 / 1e-5, rel=2e-3)
    assert on["residual_precision_floor"] >= off["residual_precision_floor"] * (1 - 1e-6)
    assert on["residual_precision_floor"] == pytest.approx(
        off["residual_precision_floor"], rel=1e-3)


def test_the_floor_is_counted_at_the_positions_the_pass_read_not_the_ones_it_left():
    """A target anchor: the kernel reads the target's **pre-step**
    positions.  Markers 100 spacings in that the step moves 600 further
    report the floor of 100, which is what rounded the weights of the
    pass the report describes."""
    gm, built = gather_pair(100, "mixed", "gauss-seidel", diagnostics=False,
                            vel=600 * SPACING, extra_cells=1500)
    _compiled(gm)
    before = np.asarray(gm.get_node_state("P")["pos"], np.float64)
    gm.step()
    after = np.asarray(gm.get_node_state("P")["pos"], np.float64)
    report = _report(gm)

    def floor_at(positions):
        reach = float(np.max((positions - built["origin"]) / SPACING))
        value_eps = EPS32 * reach / PRECISION_FLOOR_ULPS
        n_values = built["n_grid"] + M
        return PRECISION_FLOOR_ULPS * 2 * math.sqrt(
            (n_values * value_eps ** 2 + M * EPS32 ** 2) / (n_values + M)) / 1e-5

    assert floor_at(after) > 6 * floor_at(before)
    assert report["residual_precision_floor"] == pytest.approx(floor_at(before), rel=1e-4)


def test_the_steps_own_resolution_is_counted_at_the_positions_the_pass_read_too():
    """The step's analysis takes the same count from the same positions
    as the floor it records.  Members that declare, 3000 spacings in,
    whose markers the step carries to within 14 spacings of the grid's
    first point: the pass read the positions 3000 in, so the bounds'
    per-entry resolution is that of 3000, the geometry self-check's probe
    stands clear of the kernel's rounding, and the flags stand on a
    reported bound.  (Sized by the positions the step leaves, the probe
    is under that rounding and the self-check withholds the report, as
    it did before the count: gap 0.8 against a limit of 0.25.)"""
    gm, built = gather_pair(3000, "mixed", "gauss-seidel", declared=True,
                            vel=-2990 * SPACING)
    assert len([t for t in _compiled(gm) if ADVISORY in t]) == 1
    gm.step()
    report = _report(gm)
    left = np.asarray(gm.get_node_state("P")["pos"], np.float64)
    assert float(np.max((left - built["origin"]) / SPACING)) < 14.0
    assert report["converged"] and report["precision_limited"]
    assert reason_codes.GEOMETRY_SELF_CHECK_FAILED not in report["reason_codes"]["spectral_usable"]
    assert report["spectral_usable"] and report["gradient_bound_usable"], report
    assert math.isfinite(report["spectral_error_bound"])
    assert report["residual_precision_floor"] == pytest.approx(
        2 * EPS32 * 3003.5 / 1e-5, rel=2e-3)


# ------------------------------------------------------------ the advisory


@pytest.mark.parametrize("norm", ["mixed", "l2"])
def test_compile_says_what_the_report_will_say_far_into_a_grid_and_nothing_near_it(norm):
    """``compile()`` warns under "l2" and "mixed" where one rounding of
    the lattice coordinate per evaluation reaches the tolerance (300
    spacings in at the audit's tolerances) and not 3 spacings in; the
    text names the positions, the edge, where they are and the ways out,
    and offers no change of coordinates (a lattice coordinate does not
    depend on the coordinates' zero)."""
    near, _ = gather_pair(3, norm, "gauss-seidel")
    assert [t for t in _compiled(near) if ADVISORY in t] == []
    far, _ = gather_pair(300, norm, "gauss-seidel")
    texts = [t for t in _compiled(far) if ADVISORY in t]
    assert len(texts) == 1, texts
    text = texts[0]
    for part in (f"convergence_norm={norm!r}", "float32 positions P.pos", "'G.x->P.u'",
                 "spacings from the grid's first point", "precision_limited=True",
                 "hold P.pos in float64", "No choice of the coordinates' origin",
                 "MADD-ANO-247"):
        assert part in text, (part, text)
    assert "use coordinates local to the grid" not in text


def test_the_advisory_is_only_where_the_kernels_rounding_is_what_sets_the_floor():
    """A tolerance float32 cannot resolve anywhere is not this advisory's
    to report: three spacings in, the fields' own rounding is the floor
    (``rtol=1e-8``), and the kernel adds nothing to it."""
    gm, _ = gather_pair(0, "mixed", "gauss-seidel", tolerance=1e-8, extra_cells=8)
    assert [t for t in _compiled(gm) if ADVISORY in t] == []


# ------------------------------------------------- the row of a scatter


class Spread(SimulationNode):
    """``x <- b + g * u``; holds the constant positions the scatter is
    anchored at (the target of ``P -> G``)."""

    def __init__(self, name, n, pos, gain):
        super().__init__(name, 1.0)
        self._n, self._pos, self._gain = n, np.asarray(pos, np.float64), gain

    def initial_state(self):
        return {"x": jnp.ones(self._n, jnp.float32),
                "pos": jnp.asarray(self._pos, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=jnp.float32,
                                       default=jnp.zeros(self._n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": jnp.float32(0.25) + jnp.float32(self._gain) * boundary_inputs["u"],
                "pos": state["pos"]}


class Points(SimulationNode):
    """``x <- 1 + u[cell] / m``: ``m`` values, each reading one grid entry."""

    def __init__(self, name, m, n):
        super().__init__(name, 1.0)
        self._m, self._n = m, n

    def initial_state(self):
        return {"x": jnp.ones(self._m, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=jnp.float32,
                                       default=jnp.zeros(self._n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        fed = boundary_inputs["u"][1] / jnp.float32(self._m)
        return {"x": jnp.float32(0.1) + fed * jnp.ones(self._m, jnp.float32)}


def scatter_pair(m, *, rtol, declared, gain=0.9):
    """*m* markers on one lattice point of a four-point grid (weights 1
    and 0: the scatter is an in-order sum of *m* equal terms into one
    entry), anchored at the grid node's own positions."""
    grid_cls, points_cls = (_declaring(Spread), _declaring(Points)) if declared else (
        Spread, Points)
    gm = GraphManager()
    gm.add_node(grid_cls("G", 4, np.ones((m, 1)), gain))
    gm.add_node(points_cls("P", m, 4))
    gm.add_edge("P", "G", "x", "u",
                mapping=multilinear_grid_mapping([0.0], [1.0], [4], n_points=m,
                                                 mode="conservative"),
                geometry=("target", "pos"))
    gm.add_edge("G", "P", "x", "u")
    gm.add_coupling_group(["G", "P"], convergence_norm="mixed", rtol=rtol, max_iterations=400,
                          iteration_mode="gauss-seidel", solver="ift", diagnostics=True)
    return gm


def _scatter_distance(gm, m, rtol, gain=0.9):
    """The distance of the returned state from the pair's exact fixed
    point, in the group's norm.  With every marker on lattice point 1:
    ``G.x[1] = 0.25 + g m p`` and ``p = 0.1 + G.x[1] / m`` (constants as
    float32 holds them), the other grid entries 0.25."""
    g, c, d = float(np.float32(gain)), 0.25, float(np.float32(0.1))
    top = (c + g * m * d) / (1.0 - g)
    exact_grid = np.array([c, top, c, c])
    exact_points = np.full(m, d + top / m)
    state = {n: {f: np.asarray(v, np.float64) for f, v in gm.get_node_state(n).items()}
             for n in ("G", "P")}
    return _distance("mixed", rtol, [(state["G"]["x"], exact_grid),
                                     (state["P"]["x"], exact_points),
                                     (state["G"]["pos"], state["G"]["pos"])])


def test_a_scatter_is_a_row_of_its_points_and_a_gather_of_its_cells_corners():
    """What ``compile()`` hands the row guard for a geometry-dependent
    mapping: the kind's ``geometry_longest_row()`` -- a scatter's number
    of points (how many share a grid entry is state: all can), a
    gather's ``2 ** d`` weights -- whichever side it is anchored at."""
    for d in (1, 2, 3):
        gather = multilinear_grid_mapping([0.0] * d, [1.0] * d, [4] * d, n_points=37,
                                          mode="consistent")
        scatter = multilinear_grid_mapping([0.0] * d, [1.0] * d, [4] * d, n_points=37,
                                           mode="conservative")
        assert gather.geometry_longest_row() == 2 ** d
        assert scatter.geometry_longest_row() == 37
        assert 2 ** d <= _group_layout.MAPPED_ROW_FLOOR_LIMIT
    for anchor in ("source", "target"):
        rows = _group_layout._mapped_rows([
            EdgeSpec("P", "G", "x", "u", mapping=scatter, geometry=(anchor, "pos")),
            EdgeSpec("G", "P", "x", "u", mapping=gather, geometry=(
                "target" if anchor == "source" else "source", "pos"))])
        assert [(key, row) for key, _what, row in rows] == [("P.x->G.u", 37), ("G.x->P.u", 8)]
        assert "its number of points" in rows[0][1]


# Per push: tests/core/test_the_float_floor_counts_a_geometry_kernels_rounding.py::test_behind_a_scatter_of_many_points_no_flag_stands_at_the_floor[300]
@pytest.mark.parametrize("m", [
    300, pytest.param(3000, marks=pytest.mark.slow), pytest.param(30000, marks=pytest.mark.slow)])
def test_behind_a_scatter_of_many_points_no_flag_stands_at_the_floor(m):
    """The audit's three sizes, members declaring one evaluation, stalled
    at the float32 floor (``rtol=1e-7``): both flags were set beside a
    bound at 0.40 (300 markers in one cell) and 0.048 (3000) of the
    distance.  The row guard now reads the edge at its number of points
    and withdraws both, with its reason and code, where the report shows
    a floor; no report has a flag."""
    gm = scatter_pair(m, rtol=1e-7, declared=True)
    _compiled(gm)
    for _ in range(3):
        gm.step()
    report = _report(gm)
    assert report["converged"]
    assert not report["spectral_usable"] and not report["gradient_bound_usable"]
    if math.isfinite(report["residual_precision_floor"]):
        assert reason_codes.LONG_MAPPED_ROW in report["reason_codes"]["spectral_usable"]
        assert f"adds up {m} entries" in report["not_usable_reason"]
        assert "its number of points" in report["not_usable_reason"]
    if m == 3000:
        # The premise: the in-order sum's rounding is what the state is
        # off by, and the bound as reported does not cover it.
        assert report["spectral_error_bound"] < _scatter_distance(gm, m, 1e-7)


def test_behind_a_scatter_of_ten_points_the_flags_stand_on_a_bound_that_holds():
    """A row within the limit is not charged: ten markers keep both flags
    at the floor, on a bound over the distance."""
    gm = scatter_pair(10, rtol=1e-7, declared=True)
    _compiled(gm)
    for _ in range(3):
        gm.step()
    report = _report(gm)
    assert report["converged"] and report["precision_limited"]
    assert report["spectral_usable"] and report["gradient_bound_usable"]
    assert "not_usable_reason" not in report
    assert report["spectral_error_bound"] >= _scatter_distance(gm, 10, 1e-7)


# Per push: tests/core/test_the_float_floor_counts_a_geometry_kernels_rounding.py::test_behind_a_scatter_of_ten_points_the_flags_stand_on_a_bound_that_holds
@pytest.mark.slow
def test_behind_a_long_scatter_a_residual_that_stands_clear_keeps_its_flags():
    """The guard's own rule is unchanged: it withdraws only where the
    residual is under the floor the row would give.  300 markers at
    ``rtol=1e-2`` (floor 9.5e-5 tolerances, times 300 is 0.029) with a
    residual above that keep both flags, the bound over the distance."""
    gm = scatter_pair(300, rtol=1e-2, declared=False, gain=0.5)
    _compiled(gm)
    gm.step()
    report = _report(gm)
    assert report["converged"]
    assert report["residual"] > 300 * report["residual_precision_floor"]
    assert report["spectral_usable"] and report["gradient_bound_usable"]
    assert report["spectral_error_bound"] >= _scatter_distance(gm, 300, 1e-2, gain=0.5)


# --------------------------------------------------------- the audit's cells


# Per push: tests/core/test_the_float_floor_counts_a_geometry_kernels_rounding.py::test_far_into_its_grid_an_ordinary_pair_is_at_its_floor_and_no_flag_stands
@pytest.mark.slow
@pytest.mark.parametrize("schedule", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("norm", ["mixed", "l2"])
@pytest.mark.parametrize("declared", [False, True])
def test_on_the_audits_cells_no_flag_stands_beside_a_bound_under_the_distance(
        norm, schedule, declared):
    """1000, 2000, 3000 and 4000 spacings in, four seeds of the markers'
    places: every report converges; no report has a usable flag beside a
    ``spectral_error_bound`` under the distance to the fixed point of
    the positions as stored; ordinary members are at their floor
    (``precision_limited=True``) from 1000 on and have no flag; and
    ``compile()`` warned of every one."""
    for into in (1000, 2000, 3000, 4000):
        for seed in range(4):
            report, distance, advisories = solved(into, norm, schedule, declared, seed)
            where = (norm, schedule, declared, into, seed)
            assert report["converged"], where
            assert len(advisories) == 1, where
            if _usable(report):
                assert report["spectral_error_bound"] > distance, (where, report, distance)
            if not declared:
                assert report["precision_limited"] and not _usable(report), where


# Per push: tests/core/test_the_float_floor_counts_a_geometry_kernels_rounding.py::test_three_spacings_into_its_grid_the_pair_keeps_both_flags
@pytest.mark.slow
@pytest.mark.parametrize("schedule", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("norm", ["mixed", "l2"])
def test_three_spacings_in_every_seed_keeps_both_flags_on_a_bound_that_holds(norm, schedule):
    """The ordinary case over the audit's seeds, both norms and both
    schedules, ordinary members: both flags, not at the floor, the bound
    over the distance."""
    for seed in range(4):
        report, distance, advisories = solved(3, norm, schedule, False, seed)
        where = (norm, schedule, seed)
        assert report["converged"] and not report["precision_limited"], where
        assert report["spectral_usable"] and report["gradient_bound_usable"], where
        assert report["spectral_error_bound"] > distance, where
        assert advisories == (), where


# Per push: tests/core/test_the_float_floor_counts_a_geometry_kernels_rounding.py::test_float64_positions_beside_float32_fields_keep_the_flags_far_into_the_grid
@pytest.mark.slow
@pytest.mark.parametrize("into", [1000, 4000])
def test_float64_positions_keep_the_flags_on_every_seed(into):
    """The audit's control: positions in float64, fields in float32, both
    flags on a bound over the distance (1.6 to 7 times, as measured
    uncounted)."""
    for seed in range(4):
        report, distance, advisories = solved(into, "mixed", "gauss-seidel", False, seed,
                                              "float64")
        assert report["converged"] and not report["precision_limited"], seed
        assert report["spectral_usable"] and report["gradient_bound_usable"], seed
        assert report["spectral_error_bound"] > distance, seed
        assert advisories == (), seed


@pytest.fixture(autouse=True, scope="module")
def _forget_the_cells_after_the_module():
    """The cells' numbers are cached per module (no compiled graph is)."""
    yield
    solved.cache_clear()


@pytest.fixture(autouse=True)
def _release_what_a_slow_test_compiled(request):
    """After each SLOW test only: every cell of a sweep compiles a graph
    of its own, and the slow lane runs a whole shard in one process."""
    yield
    if request.node.get_closest_marker("slow") is not None:
        gc.collect()
        jax.clear_caches()
        gc.collect()
