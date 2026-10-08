"""The coupling search on groups with a ``multilinear_grid`` geometry edge.

Experimental.  Under ``convergence_norm="l2"`` and ``"mixed"`` a
single-rate coupling group whose pass resolves ``multilinear_grid``
mappings reports its bounds like any other group (MAP-036).  A gather or a
scatter is nonlinear in the positions it reads, and where those positions
are a member's own state the pass depends on the iterate *through* them:
the numbers are held here to :mod:`tests.property.coupling_reference`,
whose ``jacfwd`` Jacobian has every geometry field in the iterate.
:mod:`tests.property.geometry_cells` holds the cells, the draws and the
scores.

**Per push:** the scores on the one cell of :data:`PER_PUSH_CELLS` at the
house floor of examples (a float32 pair under Jacobi and the mixed norm
whose gather reads ``P.pos`` before the step and whose scatter reads it
from the iterate: both places a pass reads a geometry from); the radius
to 1e-5 of the reference's on seeds where a product without the geometry
would read 12% and 14% off; the self-check (MAP-045) passing on every
draw with its margin, its arithmetic on a map whose tangent is half its
derivative, and the report's reading of its slot.  **Slow:** the hunt
over every cell of :data:`CELLS`, three seeds; the radius seeds under
Gauss-Seidel in float64, and a sweep whose Jacobian has no geometry
column; a member whose tangent is half its derivative, end to end; a
geometry held by a node outside the group.

**Across a lattice plane** (MAP-049).  The stencil is another polynomial
in the next lattice cell, and the spectrum is taken at the returned
iterate: where a position the pass reads from the iterate is within twice
the bound of a plane, the report withdraws ``spectral_usable`` and
``gradient_bound_usable`` and keeps the numbers.  **Per push:** plane
draws (``geometry_cells.PlaneCase``: a fixed point a drawn 1e-6 to 1e-2 of
a spacing from a lattice plane or a face of the hull) on the per-push pair
at a looser tolerance, where the iterate stops a plane away from its fixed
point: no flag with the fixed point across, on twelve pinned draws of
which several had the flag before the criterion, and on the house floor of
drawn ones; the slot against its definition; the report's reading of the
slot; a marker on the top face of the hull.  **Slow:** the plane hunt over
:data:`PLANE_CELLS` (every anchoring, both schedules, both norms, both
dtypes, one and two axes), and float32 fields at float64 positions.

**The self-check's limits** (MAP-045), each measured in the slow lane: a
missed term that moves a field by nine float resolutions reads 0.22,
under the tolerance; an honest Gauss-Seidel pass behind a gather that
cancels three digits and more is withheld in bands of the amplitude
(MADD-ANO-212, where the bound is wrong in the bands between); an honest
float32 pass that reads a position it built in the same sweep, a few
millionths of a spacing from a lattice plane, can be withheld
(MADD-ANO-246: the plane hunt counts them).

**Seeded faults** (``plans``-side mutant list; the table in
``tests/property/geometry_graphs.py`` names them): the geometry term
dropped from the product, the product taken with the geometry of another
iterate, a target-anchored geometry read from the iterate, and the
self-check's comparison disabled are each caught by a per-push test of
this module or of the geometry harness; so are the lattice-plane
criterion disabled, taken on another field, taken in lattice units, and
blind to the faces of the hull.
"""

from __future__ import annotations

import dataclasses
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from maddening.core.coupling import _bounds
from tests.property import coupled_graphs as cg
from tests.property import coupling_reference as cr
from tests.property import geometry_cells as gc
from tests.property.geometry_cells import Case, Cell
from tests.property.targeted_search import SLOW

#: The per-push cell.  One, because a cell's cost is the compile of the
#: graph under test with its diagnostics (seconds).
_FIRST = (
    Cell(("target", "source"), True, "float32", 1, 120),
)


#: The rows of ``gc.KNOBS`` the rotation below takes, FROZEN: the six rows
#: that table held when these cells were laid out.  The rotation is over
#: this tuple and never over the length of a table of another module, so a
#: row added there cannot move a cell here; a new configuration is a new
#: cell, appended (``test_coupling_search_cells_are_pinned.py`` holds every
#: cell's configuration per push).
ROTATED_KNOBS = (0, 1, 2, 3, 4, 5)


def _cells() -> tuple:
    """Every anchoring with a moving and a still geometry, both dtypes,
    the configurations, the caps, the lattice's dimension, the members'
    order and the lattice's origin rotated."""
    out = []
    anchorings = ((("source", "source"), True), (("source", "source"), False),
                  (("target", "target"), True), (("target", "source"), True),
                  (("source", "target"), True))
    k = 0
    for a, (anchors, moving) in enumerate(anchorings):
        for t, dtype in enumerate(("float32", "float64")):
            for v in range(2):
                out.append(Cell(anchors, moving, dtype,
                                ROTATED_KNOBS[(a + 3 * t + 2 * v) % len(ROTATED_KNOBS)],
                                (5, 120)[(a + t + v) % 2], d=1 + (k % 2), m=2 + (k % 3 == 2),
                                origin=(0.5, 40.0, -3.25)[k % 3],
                                order=(("F", "P"), ("P", "F"))[(k // 2) % 2]))
                k += 1
    return tuple(out)


CELLS = _FIRST + tuple(c for c in _cells() if c not in _FIRST)
PER_PUSH_CELLS = tuple(range(len(_FIRST)))
ALL_CELLS = tuple(range(len(CELLS)))
#: The slow hunt's blocks (each cell compiles a graph and a twin).
BLOCKS = tuple(ALL_CELLS[k::4] for k in range(4))
#: The least fraction of a hunt's examples with the flag set, and the
#: least fraction of the finite ones the reference found a fixed point
#: for and scored (every point in one lattice cell from the returned
#: iterate to the fixed point).
USABLE_FLOOR = 0.25
SCORED_FLOOR = 0.5

SEARCH = gc.Search(CELLS)


def _held(name: str, fractions: dict) -> None:
    assert fractions["scored"] >= SCORED_FLOOR * fractions["finite"] > 0, (name, fractions)
    assert fractions[gc.FLAG[name]] >= USABLE_FLOOR, (name, fractions)


@pytest.mark.parametrize("name", gc.SEARCHES + ("radius_strict",))
def test_a_reported_number_of_a_group_with_a_geometry_edge_holds_against_the_reference_per_push(
        name):
    report, fractions = SEARCH.run(name, PER_PUSH_CELLS)
    _held(name, fractions)
    if name == "bound":
        assert fractions["near"] > 0, fractions
    del report


#: Draws on the per-push cell where the geometry block of the Jacobian
#: carries a tenth of the radius or more: ``(case, least share)``, the
#: share being how far the radius of the Jacobian without its geometry
#: columns is from the true one (measured: 0.137 and 0.117).
RADIUS_SEEDS = {
    "the-geometry-lowers-the-radius": (Case(0, 8, 0.7, 0.2), 0.1),
    "the-geometry-raises-the-radius": (Case(0, 29, 0.05, 0.2), 0.1),
}


@pytest.mark.parametrize("seed", sorted(RADIUS_SEEDS))
def test_the_reported_radius_has_the_geometry_term_in_it(seed):
    """Eight float32 scalars and eight Krylov steps: ``rho_spectral`` is
    the radius of the pass's Jacobian at the returned iterate to rounding
    (CPL-087; measured within 1.1e-6 of it), on draws where a product that
    left the geometry out would be a tenth or more off."""
    case, least = RADIUS_SEEDS[seed]
    seen = SEARCH.observe(case)
    report = seen["report"]
    assert seen["scored"] and seen["spectral_usable"] and seen["gradient_usable"], seen
    true, without = report["rho_true"], report["rho_without_geometry"]
    assert abs(without - true) >= least * true, (
        f"premise: the geometry moves the radius by {abs(without - true) / true:.3g}")
    assert abs(report["rho_spectral"] - true) <= 1e-5 * true, report
    over = {name: seen[name] for name in gc.SEARCHES + ("radius_strict",)
            if seen[name] > gc.THRESHOLD[name]}
    assert not over, (over, report)


#: The Gauss-Seidel cells (slow).  The first: both edges read their
#: source's positions; ``F`` is updated first and reads ``P.pos`` from the
#: iterate, so the geometry is in the pass's Jacobian (the draws: 30% and
#: 45% of the radius).  The second: both edges read ``P.pos`` and ``P`` is
#: updated first, so the scatter reads the positions *this pass* has just
#: returned and the gather the pre-step ones: the sweep never reads a
#: position from the iterate, and the Jacobian has no geometry column.
GS_CELLS = (Cell(("source", "source"), True, "float64", 0, 5),
            Cell(("target", "source"), True, "float64", 0, 5, order=("P", "F")))
GS_SEARCH = gc.Search(GS_CELLS)
GS_RADIUS_SEEDS = {"the-geometry-raises-the-radius": (Case(0, 7, 0.3, 0.2), 0.25),
                   "the-geometry-lowers-the-radius": (Case(0, 8, 0.7, 0.2), 0.25)}


# Slow: a compile of a float64 group with its diagnostics, and of its twin.
# Per push: tests/property/test_coupling_geometry_search.py::test_the_reported_radius_has_the_geometry_term_in_it
@pytest.mark.slow
@pytest.mark.parametrize("seed", sorted(GS_RADIUS_SEEDS))
def test_the_reported_radius_has_the_geometry_term_in_it_under_gauss_seidel(seed):
    """The other schedule: a member updated before the holder of the
    positions it reads takes them from the iterate, and the geometry term
    is in the Gauss-Seidel pass's Jacobian.  ``rho_spectral`` is the
    float64 reference's radius to 1e-9 where the Jacobian without its
    position columns is a quarter or more off."""
    case, least = GS_RADIUS_SEEDS[seed]
    seen = GS_SEARCH.observe(case)
    report = seen["report"]
    assert seen["scored"] and seen["spectral_usable"] and seen["reason"] is None, seen
    true, without = report["rho_true"], report["rho_without_geometry"]
    assert abs(without - true) >= least * true, (without, true)
    assert abs(report["rho_spectral"] - true) <= 1e-9 * true, report
    over = {name: seen[name] for name in gc.SEARCHES + ("radius_strict",)
            if seen[name] > gc.THRESHOLD[name]}
    assert not over, (over, report)


# Slow: a compile of a float64 group with its diagnostics, and of its twin.
# Per push: tests/property/test_coupling_geometry_search.py::test_the_reported_radius_has_the_geometry_term_in_it
@pytest.mark.slow
def test_a_sweep_that_updates_the_holder_first_has_no_geometry_column():
    """Where Gauss-Seidel updates the holder of the positions before every
    member that reads them, no read of a position is from the iterate: the
    reference's Jacobian has exactly zero in every position column, a
    product without the geometry is the same product, and the report is
    the reference's all the same (the self-check moves the positions the
    gather reads from the pre-step state)."""
    cell = GS_CELLS[1]
    for seed in (7, 8):
        case = Case(1, seed, 0.7, 0.2)
        seen = GS_SEARCH.observe(case)
        report = seen["report"]
        # The only reason this report may carry is the gradient flag's own
        # (a plane inside the Kantorovich ball, MADD-ANO-247): the radius
        # and its flag are what is read here.
        assert seen["scored"] and seen["spectral_usable"], seen
        assert seen["reason"] is None or (
            "Newton-Kantorovich ball" in seen["reason"] and not seen["gradient_usable"]), seen
        _gm, twin, ref = GS_SEARCH._built(1)                # noqa: SLF001
        values = gc.values_of(case, cell)
        ref = gc.bound_reference(ref, twin, values)
        with gc.precision(True):
            _pre, state, _d, _meta = gc.run_once(_gm, values)
        J = ref.jacobian(ref.flat(state))
        assert not np.any(J[:, gc.geometry_columns(cell, ref)])
        assert report["rho_without_geometry"] == report["rho_true"]
        assert abs(report["rho_spectral"] - report["rho_true"]) <= 1e-9 * report["rho_true"]
        assert report["geometry_gap"] <= HONEST_GAP


def test_the_product_at_another_iterate_s_geometry_is_another_radius():
    """The premise of the fault "the geometry read at another time
    level": on a radius seed the Jacobian at the first pass's geometry
    (the positions the pass returns from the pre-step state) has a radius
    the reported one is not within 1e-6 of."""
    case, _least = RADIUS_SEEDS["the-geometry-lowers-the-radius"]
    cell = CELLS[case.cell]
    seen = SEARCH.observe(case)
    _gm, twin, ref = SEARCH._built(case.cell)             # noqa: SLF001
    values = gc.values_of(case, cell)
    ref = gc.bound_reference(ref, twin, values)
    with gc.precision(cell.dtype == "float64"):
        _pre, state, _d, _meta = gc.run_once(_gm, values)
    x = ref.flat(state)
    first = ref.apply(ref.flat({n: {f: values[n][f] for f in s} for n, s in state.items()}))
    mask = gc.geometry_columns(cell, ref)
    elsewhere = np.where(mask, first, x)
    other = cr.radius(ref.jacobian(elsewhere))
    true = seen["report"]["rho_true"]
    assert abs(other - true) > 1e-3 * true, (other, true)
    assert abs(seen["report"]["rho_spectral"] - other) > 1e-3 * other


# ---------------------------------------------------------------------------
# The self-check of the product along the geometry (MAP-045)
# ---------------------------------------------------------------------------

#: An honest pass's gap is the finite difference's own rounding over the
#: larger of what the geometry moves and 32 float resolutions of the
#: residual: at most about ``2 / 32`` by construction, in any dtype, and
#: in practice far less (the module's measurements are in MAP-045).  Held
#: here to a fifth of the tolerance; a tangent that is half the derivative
#: reads 0.5 and a product without the geometry 1.0.
HONEST_GAP = _bounds.GEOMETRY_GAP_TOLERANCE / 5.0


def _per_push_draws() -> list:
    drawn = []
    for name in gc.SEARCHES:
        SEARCH.run(name, PER_PUSH_CELLS)
    drawn.extend(SEARCH._seen)                            # noqa: SLF001
    return [c for c in drawn if c.cell in PER_PUSH_CELLS]


def test_the_self_check_passes_on_every_per_push_draw_with_a_margin():
    """No draw's report is withheld by the self-check.  (A draw far from
    its fixed point can lose ``spectral_usable`` to the lattice-plane rule
    of MAP-049 -- a cap reached with a bound that spans a lattice cell and
    no certified linearisation -- which is another reason, and keeps the
    numbers.)"""
    draws = _per_push_draws()
    assert len(draws) >= 20
    gaps = []
    for case in draws:
        seen = SEARCH.observe(case)
        assert seen["reason"] is None or "lattice plane" in seen["reason"], (
            case, seen["reason"])
        assert math.isfinite(seen["report"]["rho_spectral"]) or not seen["finite"], case
        if seen["finite"]:
            gaps.append(seen["report"]["geometry_gap"])
    assert len(gaps) >= 15 and max(gaps) <= HONEST_GAP, max(gaps)
    assert all(g >= 0 for g in gaps) and max(gaps) > 0, "the check compared nothing"


def _toy_gap(tangent_share: float, *, moved: float = 1.0, value=1.0):
    """The gap of the map ``F(x, c) = [x0 * c0 * moved, x1 + 100 c0]``
    whose first field's tangent in ``c`` is *tangent_share* of its
    derivative (the second field's is the derivative)."""
    import jax.numpy as jnp  # noqa: PLC0415

    def step_pure(x, c):
        seen = tangent_share * c + jax.lax.stop_gradient((1.0 - tangent_share) * c)
        return jnp.stack([x[0] * seen[0] * moved, x[1] + 100.0 * c[0]]), jnp.zeros(())

    x = jnp.asarray([value, 2.0], jnp.float32)
    c = jnp.asarray([0.5], jnp.float32)
    step = jnp.asarray([3.0e-4], jnp.float32)
    return float(_bounds._geometry_product_gap(                    # noqa: SLF001
        step_pure, x, (c,), [(jnp.zeros_like(x), [step])], jnp.ones(2, jnp.float32),
        jnp.full(2, 1.2e-7, jnp.float32), np.asarray([0, 1]), 2))


def test_the_gap_is_the_share_of_the_derivative_the_product_does_not_see():
    """The arithmetic of the self-check on a two-field map: an honest
    product reads the finite difference's rounding, one that sees half of
    the geometry's effect a half, one that sees none of it one; per
    field, so a field the product is right about does not dilute another;
    a field the geometry moves by less than the rounding it is allowed is
    not compared; a state it cannot be evaluated at is NaN."""
    assert _toy_gap(1.0) <= 0.01
    assert 0.45 <= _toy_gap(0.5) <= 0.55
    assert 0.9 <= _toy_gap(0.0) <= 1.0
    assert _toy_gap(0.0, moved=1e-6) <= HONEST_GAP          # under the rounding allowed
    assert math.isnan(_toy_gap(1.0, value=math.nan))
    assert 0.9 <= _toy_gap(0.0) <= 1.0 and _toy_gap(0.0) > _bounds.GEOMETRY_GAP_TOLERANCE
    # A term the product does not see at all, moving its field by nine of
    # the 32 resolutions the difference is allowed as rounding: the gap is
    # ``9 / (9 + 32)``, the reading of the weak-deposit fault below.  That
    # is the check's stated limit: a missed term this weak is under the
    # tolerance.
    weak = _toy_gap(0.0, moved=3.6e-3)
    assert 0.2 <= weak <= 0.24, weak


def test_the_report_withholds_the_bounds_where_the_step_s_gap_is_over_the_tolerance():
    """The report's side of the self-check, on the per-push graph with the
    slot the step wrote replaced: under the tolerance the bounds stand,
    above it and at NaN they are withheld with the reason, the gap in it.
    0.305 is the smallest reading of a pass whose every read of a geometry
    was under ``stop_gradient`` on the six-cell set of MAP-045.  A gap
    over the tolerance names both things it can be, a derivative that is
    not the value's and a finite difference that could not be formed (an
    honest pass beside a lattice plane has read 0.29); a gap that is not
    a number names neither."""
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    case, _least = RADIUS_SEEDS["the-geometry-lowers-the-radius"]
    cell = CELLS[case.cell]
    gm, _twin, _ref = SEARCH._built(case.cell)             # noqa: SLF001
    slot = f"coupling_{gc.KEY}_geometry_gap"
    keys = [e.key for e in gm._edges]                      # noqa: SLF001
    with gc.precision(cell.dtype == "float64"):
        _pre, _state, honest, meta = gc.run_once(gm, gc.values_of(case, cell))
        assert "not_usable_reason" not in honest and honest["spectral_usable"] is True
        kept = gm._state                                   # noqa: SLF001
        try:
            allowed = _bounds.GEOMETRY_GAP_TOLERANCE
            for gap, fails in ((0.9 * allowed, False), (1.1 * allowed, True), (0.305, True),
                               (1.0, True), (math.nan, True)):
                gm._state = {**kept, "_meta": {                    # noqa: SLF001
                    **kept["_meta"], slot: np.asarray(gap, meta["geometry_gap"].dtype)}}
                report = dict(gm.coupling_diagnostics()[gc.KEY])
                if not fails:
                    assert report == honest, (gap, report)
                    continue
                # A gap that is not a number is a check that compared
                # nothing: the reason does not blame a derivative.
                why = "self-check" if gap == gap else "self-check-unevaluated"
                gg.assert_not_diagnosed(report, keys, why)
                assert (f"relative gap {gap:.3g}, allowed {allowed:.3g}"
                        in report["not_usable_reason"])
                for reading in ("derivative is not that of its value",
                                "a finite difference that could not be formed",
                                "within the check's step of a plane of the lattice",
                                "cancels digits", "does not tell these apart"):
                    assert (reading in report["not_usable_reason"]) is (gap == gap), reading
                (row,) = list(gm.coupling_report())
                assert any(gg.WHY[why] in f for f in row["flags"])
        finally:
            gm._state = kept                               # noqa: SLF001
    assert dict(gm.coupling_diagnostics()[gc.KEY]) == honest


class HalfTangent(gc.GeoRelay):
    """A member whose tangent in its mapped input is half its derivative."""

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"]
        half = 0.5 * u + jax.lax.stop_gradient(0.5 * u)
        return super().update(state, {**boundary_inputs, "u": half}, dt, params=params)


# Slow: one more compile of a group with its diagnostics.
# Per push: tests/property/test_coupling_geometry_search.py::test_the_gap_is_the_share_of_the_derivative_the_product_does_not_see
# Per push: tests/property/test_coupling_geometry_search.py::test_the_report_withholds_the_bounds_where_the_step_s_gap_is_over_the_tolerance
@pytest.mark.slow
@pytest.mark.parametrize("anchors", [("source", "source"), ("target", "target")],
                         ids=["read from the iterate", "read from the pre-step state"])
def test_a_pass_whose_product_along_the_geometry_is_not_its_derivative_reports_no_bound(
        anchors):
    """The pass's value follows the positions and its tangent follows
    them half as far: the self-check reads a gap of a half, and the
    report withholds every bound with the reason, as for a geometry the
    diagnostics do not read.  Once with the positions read from the
    iterate (the check's first direction) and once with them read from
    the members' pre-step state, constants of the pass (its second)."""
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    cell = Cell(anchors, True, "float32", 0, 5)
    with gc.precision(False):
        gm = gc.build(cell, cell.knobs, cell.dtype, node=HalfTangent)
    values = gc.values_of(Case(0, 8, 0.7, 0.2), cell)
    with gc.precision(False):
        _pre, _state, d, meta = gc.run_once(gm, values)
    gap = float(meta["geometry_gap"])
    assert 0.4 <= gap <= 0.6, gap
    gg.assert_not_diagnosed(d, [e.key for e in gm._edges], why="self-check")   # noqa: SLF001
    assert f"{gap:.3g}" in d["not_usable_reason"]
    # The slots are the step's own: its spectrum was computed, and would
    # have been reported.
    assert math.isfinite(float(meta["rho_spectral"]))


def test_the_self_check_is_traced_only_with_diagnostics_on_a_geometry_group():
    """No slot, and nothing of the check in the step, for the same group
    without diagnostics; no slot for a group without a geometry edge
    (``scripts/compile_counts.py`` and the step-program digests hold the
    programs)."""
    cell = _FIRST[0]
    knobs = {**cell.knobs, "diagnostics": False}
    with gc.precision(False):
        gm = gc.build(cell, knobs, cell.dtype)
        values = gc.values_of(Case(0, 7, 0.05, 0.1), cell)
        _pre, _state, d, meta = gc.run_once(gm, values)
    assert "geometry_gap" not in meta
    assert "not_usable_reason" not in d and not d["spectral_usable"]
    assert d["ratio_usable"] in (True, False) and math.isfinite(d["residual"])


# ---------------------------------------------------------------------------
# Across a lattice plane (MAP-049, MADD-ANO-242)
# ---------------------------------------------------------------------------

#: The cell of the pinned case: one marker on a line of four lattice
#: points, the gather reading its position before the step and the scatter
#: from the iterate, Gauss-Seidel with the grid first, the plain norm at a
#: tolerance of 1e-4.
PINNED_CELL = Cell(("target", "source"), True, "float32", 0, 400, d=1, m=1, origin=0.0,
                   tolerance=1e-4)
PINNED_SEARCH = gc.Search((PINNED_CELL,))


def _pinned_values(*, drift: float) -> dict:
    """A grid ``x <- a x_pre + b deposit + s`` and a marker ``x <- c x_pre
    + d sampled``, ``pos <- pos_pre + drift + e sampled``, as
    :class:`~tests.property.geometry_cells.GeoRelay` holds them (its
    ``alpha`` is fixed, so the rest of ``a x_pre`` is in the bias).

    With ``drift = 0.42483514706375847`` the marker's fixed point is 2e-4
    of a spacing past the lattice plane at 1.0 and the pass contracts at
    0.28 in the cell before the plane and at 0.97 in the cell after it: an
    iterate that stops before the plane is a few bounds from a fixed point
    its linearisation knows nothing of.
    """
    a, b, c, d, e = (0.325056206536607, 0.9249118408408878, 0.17671563155427855,
                     0.8011272103796088, -0.6953020489474653)
    grid = np.asarray([1.436, 0.491, -0.452, -0.91])
    bias = np.asarray([-0.855, 0.377, -0.4, -0.428])
    marker = np.asarray([2.788])
    return {"F": {"x": grid, "g": b, "c": (a - gc.ALPHA) * grid + bias},
            "P": {"x": marker, "g": d, "c": (c - gc.ALPHA) * marker,
                  "pos": np.asarray([[0.7877]]), "drift": np.asarray([[drift]]),
                  "Q": np.asarray([[e]])}}


PINNED_ACROSS = 0.42483514706375847
PINNED_MID_CELL = 0.30


def _pinned(drift: float) -> dict:
    case = Case(0, 0, 0.0, 0.0)
    return gc.observe(PINNED_CELL, case, PINNED_SEARCH._built(0),      # noqa: SLF001
                      values=_pinned_values(drift=drift), across=True)


def test_the_flag_is_withdrawn_with_the_fixed_point_across_a_lattice_plane():
    """The audited case.  The solve converges (its criterion is met) with
    the marker before the plane at 1.0 and its fixed point after it.  The
    spectrum at the returned iterate is right about the cell the iterate
    is in (``rho_spectral`` is the reference's radius there) and the bound
    built on it is several times under the true distance.  Before the
    criterion ``spectral_usable`` was set on it; now a plane is within
    twice the bound, the step's Newton-Kantorovich check did not certify
    the linearisation across it, and the flag is withdrawn with the
    numbers kept and the reason.  The same graph with the fixed point in
    the middle of a cell keeps its flag and holds its bound."""
    seen = _pinned(PINNED_ACROSS)
    report = seen["report"]
    assert seen["referenced"] and seen["crossed"] and report["converged"], seen
    # The premise: the flag as it was, on a bound under half the distance.
    assert seen["usable_before"] and seen["plane_before"] > 1.0, seen
    assert report["distance"] > 2.0 * report["spectral_error_bound"] > 0.0
    assert report["spectral_error_bound"] > report["plane_limit"] >= 0.0
    assert not math.isfinite(report["gradient_relative_error_bound"])
    # The fix.
    assert seen["spectral_usable"] is False and seen["gradient_usable"] is False
    assert seen["plane"] == 0.0
    reason = seen["reason"]
    assert "lattice plane" in reason and "P.x->F.u" in reason and "F.x->P.u" in reason
    assert f"{report['spectral_error_bound']:.3g}" in reason
    assert "do not read a moving geometry" not in reason       # the numbers are reported
    assert math.isfinite(report["rho_spectral"]) and math.isfinite(report["residual"])

    control = _pinned(PINNED_MID_CELL)
    assert control["referenced"] and not control["crossed"] and control["scored"], control
    assert control["spectral_usable"] is True and control["reason"] is None, control
    assert control["report"]["plane_limit"] > control["report"]["spectral_error_bound"]
    assert control["near"] and 0.0 < control["bound"] <= gc.THRESHOLD["bound"], control


#: Plane draws on the per-push pair (its own tolerance): each face of the
#: hull from either side, an interior plane from either side, at three
#: distances, on either coordinate.
PINNED_PLANES = tuple(
    gc.PlaneCase(0, seed, loop, 0.15, which=which, plane=plane, offset=offset)
    for seed, loop, which, plane, offset in (
        (1, 0.3, 0, 0, 3e-4), (2, 0.8, 1, 0, -5e-3), (3, 0.3, 1, 3, -1e-5), (4, 0.8, 0, 3, 5e-3),
        (5, 0.3, 0, 1, -3e-4), (6, 0.8, 1, 1, 1e-5), (7, 0.3, 1, 2, 5e-3), (5, 0.8, 0, 2, -1e-5)))


def test_the_plane_limit_the_step_stores_is_its_definition():
    """``geometry_plane_limit`` against
    :func:`~tests.property.geometry_cells.plane_limit_reference`, which
    measures the distance to the nearest plane in the reference's norm at
    the returned state: the entries of ``P.pos`` (the positions the pass
    reads from the iterate), in the mixed norm's units, in lengths, the
    faces of the hull counted.  A point outside the hull is among the
    draws, and one whose nearest plane is a face."""
    outside = faces = 0
    origin, spacing, shape = CELLS[0].grid
    for case in PINNED_PLANES:
        seen = SEARCH.observe(case)
        assert seen["placed"] and seen["referenced"], (case, seen)
        report = seen["report"]
        want = report["plane_limit_reference"]
        assert math.isfinite(want) and want >= 0.0
        assert report["plane_limit"] == pytest.approx(
            want, rel=1e-3, abs=report["plane_limit_resolution"]), (case, report)
        # No flag on a bound the fixed point is outside twice of.
        assert seen["plane"] <= gc.THRESHOLD["plane"], (case, seen)
        assert seen["bound"] <= gc.THRESHOLD["bound"], (case, seen)
        top = origin[0] + (shape[0] - 1) * spacing[0]
        target = origin[0] + (case.plane % shape[0] + case.offset) * spacing[0]
        outside += not origin[0] <= target <= top
        faces += case.plane % shape[0] in (0, shape[0] - 1)
    assert outside >= 2 and faces >= 4


def test_no_flag_stands_on_a_fixed_point_beyond_twice_the_bound_on_drawn_planes_per_push():
    report, fractions = SEARCH.run("plane", PER_PUSH_CELLS, planes=True)
    assert fractions["referenced"] > 0.5, fractions
    del report


def test_no_gradient_flag_stands_with_the_fixed_point_across_a_plane_on_drawn_planes_per_push():
    """The gradient's flag on plane draws (MADD-ANO-247): its bound against
    the reference's error wherever it is set, and never set with the fixed
    point in another lattice cell than the returned iterate -- the flag
    stands only with no plane inside the Newton-Kantorovich ball around the
    iterate (``_bounds._kantorovich_ball_plane_margin``).  The draws that
    put the fixed point just past a plane the Newton point stops short of
    are the table's (``test_coupling_plane_sides.py``): these have it 1e-6
    of a spacing or more from the plane."""
    report, fractions = SEARCH.run("gradient", PER_PUSH_CELLS, planes=True)
    assert fractions["referenced"] > 0.5, fractions
    del report
    placed = [(case, seen) for case, seen in SEARCH._seen.items()         # noqa: SLF001
              if isinstance(case, gc.PlaneCase) and seen.get("placed") and seen["referenced"]]
    across = [case for case, seen in placed if seen["crossed"]]
    assert across and len(across) < len(placed), (len(across), len(placed))
    flagged = [case for case, seen in placed if seen["crossed"] and seen["gradient_usable"]]
    assert not flagged, flagged[:3]


def test_the_report_withdraws_the_flag_where_a_plane_is_in_reach_and_the_step_is_not_certified():
    """The report's side, on the per-push graph with the two slots the
    rule reads replaced: the flag stands with the bound at the limit, and
    over it where the gradient bound is finite (the step certified its
    linearisation across the Newton step); over the limit without that it
    is withdrawn, the numbers kept, with the reason; a limit that is not a
    number counts as a plane in reach, ``inf`` as none."""
    case, _least = RADIUS_SEEDS["the-geometry-lowers-the-radius"]
    cell = CELLS[case.cell]
    gm, _twin, _ref = SEARCH._built(case.cell)             # noqa: SLF001
    limit_slot = f"coupling_{gc.KEY}_geometry_plane_limit"
    gradient_slot = f"coupling_{gc.KEY}_gradient_relative_error_bound"
    with gc.precision(cell.dtype == "float64"):
        _pre, _state, honest, meta = gc.run_once(gm, gc.values_of(case, cell))
        assert "not_usable_reason" not in honest and honest["spectral_usable"] is True
        assert honest["gradient_bound_usable"] is True
        bound = honest["spectral_error_bound"]
        kept = gm._state                                   # noqa: SLF001
        dtype = meta["geometry_plane_limit"].dtype
        under = float(np.nextafter(np.asarray(bound, dtype), np.asarray(0, dtype)))

        def report_with(limit, gradient=None):
            slots = {limit_slot: np.asarray(limit, dtype)}
            if gradient is not None:
                slots[gradient_slot] = np.asarray(gradient, dtype)
            gm._state = {**kept, "_meta": {**kept["_meta"], **slots}}     # noqa: SLF001
            return dict(gm.coupling_diagnostics()[gc.KEY])

        try:
            for limit in (bound, math.inf, 2.0 * bound):
                assert report_with(limit) == honest, limit
                assert report_with(limit, math.inf)["spectral_usable"] is True, limit
            # A plane in reach, certified: the flag stands.
            for limit in (under, 0.0, math.nan):
                assert report_with(limit) == honest, limit
            # A plane in reach, not certified.
            for limit in (under, 0.0, math.nan):
                for gradient in (math.inf, math.nan):
                    report = report_with(limit, gradient)
                    assert report["spectral_usable"] is False
                    assert report["gradient_bound_usable"] is False
                    reason = report.pop("not_usable_reason")
                    assert "lattice plane" in reason and f"{bound:.3g}" in reason
                    assert all(e.key in reason for e in gm._edges)         # noqa: SLF001
                    rest = {k: v for k, v in honest.items()
                            if k not in ("spectral_usable", "gradient_bound_usable",
                                         "gradient_relative_error_bound")}
                    assert {k: report[k] for k in rest} == rest, report
                    (row,) = list(gm.coupling_report())
                    assert any("lattice plane" in f for f in row["flags"]), row["flags"]
            # The gradient's flag reads the margin of the Kantorovich ball,
            # whatever the limit and the bound read: over one it stands; at
            # one, under it, not a number or absent it is withdrawn alone,
            # with the numbers and the spectral flag kept (MADD-ANO-247).
            margin_slot = f"coupling_{gc.KEY}_geometry_plane_margin"
            assert float(kept["_meta"][margin_slot]) > 1.0

            def report_at(margin, limit=math.inf):
                slots = {limit_slot: np.asarray(limit, dtype)}
                meta_now = {**kept["_meta"], **slots}
                if margin is None:
                    del meta_now[margin_slot]
                else:
                    meta_now[margin_slot] = np.asarray(margin, dtype)
                gm._state = {**kept, "_meta": meta_now}                    # noqa: SLF001
                return dict(gm.coupling_diagnostics()[gc.KEY])

            above = float(np.nextafter(np.asarray(1.0, dtype), np.asarray(2.0, dtype)))
            for margin in (above, 26.0, math.inf):
                for limit in (math.inf, 0.0):
                    assert report_at(margin, limit) == honest, (margin, limit)
            for margin in (1.0, 0.5, 0.0, math.nan, None):
                for limit in (math.inf, 2.0 * bound, 0.0):
                    report = report_at(margin, limit)
                    assert report["gradient_bound_usable"] is False, (margin, limit)
                    assert report["spectral_usable"] is True, (margin, limit)
                    reason = report.pop("not_usable_reason")
                    assert "Newton-Kantorovich ball" in reason and "lattice plane" in reason
                    assert all(e.key in reason for e in gm._edges)         # noqa: SLF001
                    rest = {k: v for k, v in honest.items() if k != "gradient_bound_usable"}
                    assert {k: report[k] for k in rest} == rest, report
                    (row,) = list(gm.coupling_report())
                    assert any(f.startswith("gradient_bound_usable=False") and "ball" in f
                               for f in row["flags"]), row["flags"]
                    assert not any(f.startswith("spectral_usable=False") for f in row["flags"])
            # The self-check's failure is the whole report's; the plane's is not added to it.
            gm._state = {**kept, "_meta": {**kept["_meta"],                # noqa: SLF001
                                           limit_slot: np.asarray(0.0, dtype),
                                           gradient_slot: np.asarray(math.inf, dtype),
                                           f"coupling_{gc.KEY}_geometry_gap":
                                               np.asarray(1.0, dtype)}}
            assert "lattice plane" not in gm.coupling_diagnostics()[gc.KEY]["not_usable_reason"]
        finally:
            gm._state = kept                               # noqa: SLF001
    assert dict(gm.coupling_diagnostics()[gc.KEY]) == honest


def test_a_marker_on_the_top_face_of_the_hull_passes_the_self_check():
    """MADD-ANO-243.  The per-push pair with one marker starting exactly on
    the last lattice point of its axis (and one on the first): the gather
    reads those positions as constants of the pass, and the self-check
    moves them.  It stepped a point on the top face out of the hull, where
    the kernel clamps, and read a gap of 0.6 to 0.76 on an honest pass;
    stepping inwards it reads the finite difference's own error."""
    case, _least = RADIUS_SEEDS["the-geometry-lowers-the-radius"]
    cell = CELLS[case.cell]
    gm, _twin, _ref = SEARCH._built(case.cell)             # noqa: SLF001
    origin, spacing, shape = cell.grid
    top = origin[0] + (shape[0] - 1) * spacing[0]
    for first, second in ((top, origin[0] + 0.4 * spacing[0]), (origin[0], top)):
        values = gc.values_of(case, cell)
        values["P"]["pos"] = np.asarray([[first], [second]])
        values["P"]["drift"] = np.zeros((2, 1))
        with gc.precision(False):
            _pre, _state, d, meta = gc.run_once(gm, values)
        gap = float(meta["geometry_gap"])
        assert 0.0 < gap <= HONEST_GAP, (first, second, gap)
        assert "do not read a moving geometry" not in d.get("not_usable_reason", ""), d
        assert math.isfinite(d["rho_spectral"]) and math.isfinite(d["spectral_error_bound"])


# Slow: a compile of a Gauss-Seidel group with its diagnostics and of its twin.
# Per push: tests/core/test_grid_probe_step_and_plane_distance.py::test_a_step_beside_the_positions_derived_from_it_carries_neither_across_a_plane
@pytest.mark.slow
def test_a_marker_whose_in_pass_position_is_a_rounding_from_a_plane_passes_the_self_check():
    """MADD-ANO-243, the other way a step left the stencil's polynomial.
    Under Gauss-Seidel a member swept after the holder of the positions
    reads them as the pass has just built them, ``pos_pre + drift + ...``.
    The self-check moved the pre-step positions towards the middle of
    *their* cell; here that carried the in-pass position, 1e-6 of a
    spacing under a lattice plane, across it, and an honest pass read a
    gap of 0.54.  The plane hunt found it."""
    cell = PLANE_CELLS[2]
    assert cell.anchors == ("source", "source") and cell.order == ("F", "P")
    assert gc.KNOBS[cell.knob]["iteration_mode"] == "gauss-seidel"
    search = _PLANE_HUNTS.setdefault(2, gc.Search(PLANE_CELLS))
    case = gc.PlaneCase(2, 6, 0.02, 0.02, which=6, plane=6, offset=-1e-06)
    seen = search.observe(case)
    assert seen["placed"] and seen["referenced"], seen
    assert seen["report"]["geometry_gap"] <= HONEST_GAP, seen["report"]
    assert seen["reason"] is None or "lattice plane" in seen["reason"], seen["reason"]


class WidePositions(gc.GeoRelay):
    """Float32 values at float64 positions."""

    def initial_state(self):
        state = super().initial_state()
        if "pos" in state:
            state["pos"] = state["pos"].astype("float64")
        return state

    def update(self, state, boundary_inputs, dt, *, params=None):
        out = super().update(state, boundary_inputs, dt, params=params)
        if "pos" in out:
            out["pos"] = out["pos"].astype("float64")
        return out


# Slow: a compile of a group with its diagnostics under x64.
# Per push: tests/core/test_grid_probe_step_and_plane_distance.py::test_float32_fields_at_float64_positions_pass_the_self_check_at_the_pass_s_step
@pytest.mark.slow
@pytest.mark.parametrize("anchors", [("source", "source"), ("target", "target")],
                         ids=["read from the iterate", "read from the pre-step state"])
def test_float32_fields_at_float64_positions_pass_the_self_check(anchors):
    """MADD-ANO-244.  The self-check's step was ``sqrt(eps)`` of the
    spacing in the *positions'* dtype, 7e-9 of a spacing in float64, which
    a float32 field cannot resolve: an honest pass read a gap of 0.27 to
    1.0 and every bound was withheld.  The step is the pass's coarsest
    dtype's."""
    cell = Cell(anchors, True, "float32", 0, 120)
    with gc.precision(True):
        gm = gc.build(cell, cell.knobs, cell.dtype, node=WidePositions)
        for seed in (7, 8, 29):
            _pre, state, d, meta = gc.run_once(gm, gc.values_of(Case(0, seed, 0.3, 0.2), cell))
            assert {str(v.dtype) for v in state["P"].values()} == {"float32", "float64"}
            gap = float(meta["geometry_gap"])
            assert 0.0 < gap <= HONEST_GAP, (seed, gap)
            assert "do not read a moving geometry" not in d.get("not_usable_reason", ""), d
            assert math.isfinite(d["rho_spectral"])


class SameFieldCancellationInAGather(AssertionError):
    """MADD-ANO-212, through the reference kind: a usable bound under the
    true distance of a Gauss-Seidel group whose gather samples a field
    that changes sign across a lattice cell."""


#: The amplitudes of the sign-changing field the cases below are read at.
#: Measured under Gauss-Seidel in float32 at a tolerance of 1e-7 on 126
#: amplitudes from 10 to 1e5, the same digits on jaxlib 0.10.2, 0.11.0 and
#: 0.11.2 (the flag is set at every one before the self-check is asked):
#:
#: ===========  =============  ================  ==========================
#: amplitude    self-check gap bound / distance  the report
#: ===========  =============  ================  ==========================
#: 10 - 420     under 0.047    1.29 or more      stands, and the bound holds
#: 430 - 510    0.025          0.87              **stands, the bound too low**
#: 520 - 850    0.025 - 0.078  2.2               stands, and the bound holds
#: 860 - 1020   0.087          0.57              **stands, the bound too low**
#: 1030 - 2000  0.25 - 0.34    0.15 - 0.34       withheld by the self-check
#: 2200 - 2700  0.144          0.15 - 0.11       **stands, the bound too low**
#: 3000 - 1e5   0.34 - 0.75    0.28 - 0.006      withheld by the self-check
#: ===========  =============  ================  ==========================
#:
#: (A tolerance of 0.05 would withhold everything from 640 on and leave the
#: band at 430 to 510; it is not adopted, for the reason MAP-045 gives.)
SIGN_CHANGING_BOUNDED = (10.0, 100.0, 300.0)
SIGN_CHANGING_TOO_LOW = (470.0, 1000.0, 2500.0)
SIGN_CHANGING_WITHHELD = (1100.0, 1400.0, 3000.0)
_SIGN_CHANGING: dict = {}


def _sign_changing(mode: str, amplitude: float) -> dict:
    """A float32 grid field alternating ``+A, -A`` and two markers in the
    middle of a cell each: every sample is the difference of two numbers
    ``A / 2`` large, which the gather rounds at ``eps A / 2`` and the
    floor's count of evaluations does not see."""
    knob = {"gauss-seidel": 0, "jacobi": 3}[mode]
    assert gc.KNOBS[knob]["iteration_mode"] == mode and gc.KNOBS[knob]["convergence_norm"] == "l2"
    cell = Cell(("target", "source"), True, "float32", knob, 80, d=1, m=2, origin=0.0,
                tolerance=1e-7)
    if mode not in _SIGN_CHANGING:
        _SIGN_CHANGING[mode] = gc.built(cell)             # one compile a schedule
    grid = amplitude * np.asarray([1.0, -1.0, 1.0, -1.0])
    markers = np.asarray([0.5, 1.5])
    values = {"F": {"x": grid, "g": 0.3,
                    "c": (0.5 - gc.ALPHA) * grid + np.asarray([0.2, 0.1, 0.4, 0.3])},
              "P": {"x": markers, "g": 0.4, "c": (0.5 - gc.ALPHA) * markers,
                    "pos": np.asarray([[0.25], [0.75]]), "drift": np.zeros((2, 1)),
                    "Q": 0.0002 * np.eye(2)}}
    return gc.observe(cell, Case(0, 0, 0.0, 0.0), _SIGN_CHANGING[mode], values=values,
                      across=True)


# Slow: a compile of a group with its diagnostics and of its twin.
# Per push: tests/core/test_coupling_floor_gain_of_a_difference_within_one_field.py::test_a_gauss_seidel_group_reading_a_difference_within_one_field_is_bounded
@pytest.mark.slow
@pytest.mark.parametrize("amplitude", SIGN_CHANGING_TOO_LOW[:2])
def test_a_jacobi_group_whose_gather_samples_a_sign_changing_field_is_bounded(amplitude):
    """The control of the cases below: under Jacobi the read is of the
    iterate, the bound covers the stall and the self-check's finite
    difference is resolved (a gap of 4e-3 at most from 10 to 1e4)."""
    seen = _sign_changing("jacobi", amplitude)
    assert seen["referenced"] and not seen["crossed"] and seen["scored"], seen
    report = seen["report"]
    assert seen["spectral_usable"] and seen["reason"] is None, seen
    assert report["spectral_error_bound"] >= report["distance"] > 0.0, report
    assert 0.0 < report["geometry_gap"] <= HONEST_GAP, report


# Slow: a compile of a group with its diagnostics and of its twin.
# Per push: tests/core/test_coupling_floor_gain_of_a_difference_within_one_field.py::test_a_gauss_seidel_group_reading_a_difference_within_one_field_is_bounded
@pytest.mark.slow
@pytest.mark.parametrize("amplitude", SIGN_CHANGING_BOUNDED)
def test_a_gauss_seidel_group_whose_gather_cancels_a_few_digits_is_bounded(amplitude):
    """Below the first band: the flag is set, the self-check passes and
    the bound holds (5.3 times the distance at 100, 2.2 at 300)."""
    seen = _sign_changing("gauss-seidel", amplitude)
    report = seen["report"]
    assert seen["referenced"] and not seen["crossed"] and seen["scored"], seen
    assert seen["spectral_usable"] and seen["reason"] is None, seen
    assert report["spectral_error_bound"] >= report["distance"] > 0.0, report
    assert 0.0 < report["geometry_gap"] <= _bounds.GEOMETRY_GAP_TOLERANCE, report


# Slow: a compile of a group with its diagnostics and of its twin.
# Per push: tests/core/test_coupling_floor_gain_of_a_difference_within_one_field.py::test_a_gauss_seidel_group_reading_a_difference_within_one_field_is_bounded
@pytest.mark.slow
@pytest.mark.xfail(strict=True, raises=SameFieldCancellationInAGather, reason=(
    "MADD-ANO-212: the floor's gain of a same-pass read is measured along the source's "
    "own state, which a gather of a field that changes sign across a cell cancels, so a "
    "Gauss-Seidel group stalled behind such a gather reports a usable bound below the "
    "true distance; open, deferred to 0.5.0"))
@pytest.mark.parametrize("amplitude", SIGN_CHANGING_TOO_LOW)
def test_a_gauss_seidel_group_whose_gather_samples_a_sign_changing_field_is_bounded(amplitude):
    """Every ``multilinear_grid`` gather is a same-pass read that
    differences entries of one field wherever the sampled field changes
    sign across a cell (a velocity near a stagnation point, a signed
    distance near its zero level).  Measured at a tolerance of 1e-7, one
    amplitude in each band the scan found where the report is neither
    right nor withheld (the table above): the group stalls at
    ``residual=0.0`` and the bound reads 0.87 of the true distance at
    470, 0.57 at 1000 (3.0e-6 against 5.3e-6) and 0.15 at 2500, usable,
    with a self-check gap under the tolerance (0.025, 0.087, 0.144); the
    same digits on jaxlib 0.10.2, 0.11.0 and 0.11.2.  The search's own
    bound score is not what is asked here: it allows a bound the
    cancellation it measures inside an update, which is this defect's
    size."""
    seen = _sign_changing("gauss-seidel", amplitude)
    report = seen["report"]
    assert seen["referenced"] and not seen["crossed"] and seen["scored"], seen
    assert seen["spectral_usable"] and seen["reason"] is None, seen
    if report["spectral_error_bound"] < report["distance"]:
        raise SameFieldCancellationInAGather(
            f"the bound is {report['spectral_error_bound'] / report['distance']:.3g} of the "
            f"true distance, usable: {report}")


# Slow: a compile of a group with its diagnostics and of its twin.
# Per push: tests/property/test_coupling_geometry_search.py::test_the_report_withholds_the_bounds_where_the_step_s_gap_is_over_the_tolerance
@pytest.mark.slow
@pytest.mark.parametrize("amplitude", SIGN_CHANGING_WITHHELD)
def test_a_gauss_seidel_group_behind_a_strongly_cancelling_gather_reports_no_bound(amplitude):
    """Between and above those bands the self-check withholds the report
    of this honest pass.  The markers' field, of size one, reads the
    grid's same-pass update, of size ``A / 2``, through the gather, and
    the check's finite difference of it is the grid's rounding: a gap of
    0.34 at 1100, 0.30 at 1400 and 0.40 at 3000, where the bound it would
    have carried is 0.15, 0.34 and 0.11 of the true distance.  The reason
    does not say that a derivative is wrong: it names both things a gap
    can be."""
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    seen = _sign_changing("gauss-seidel", amplitude)
    report = seen["report"]
    assert seen["finite"] and report["converged"], seen
    assert report["geometry_gap"] > _bounds.GEOMETRY_GAP_TOLERANCE, report
    assert not seen["spectral_usable"] and not seen["gradient_usable"], seen
    assert math.isnan(report["rho_spectral"]) and math.isnan(report["spectral_error_bound"])
    assert gg.WHY["self-check"] in seen["reason"], seen["reason"]
    assert "a finite difference that could not be formed" in seen["reason"]
    assert "a gather of a field that changes sign across a lattice cell" in seen["reason"]


def _deposit_pair(deposit: float, pull: float) -> dict:
    """A grid ``x <- 0.5 x_pre + deposit * scattered + s`` and two markers
    ``x <- 0.5 x_pre + 0.4 sampled``, ``pos <- pos_pre + 0.005 + pull *
    sampled``, as :class:`~tests.property.geometry_cells.GeoRelay` holds
    them (its ``alpha`` is fixed, so the rest of ``0.5 x_pre`` is in the
    bias)."""
    grid = np.asarray([1.0, 1.4, 1.2, 2.0])
    markers = np.asarray([1.5, 2.5])
    return {"F": {"x": grid, "g": deposit,
                  "c": (0.5 - gc.ALPHA) * grid + np.asarray([0.2, 0.1, 0.4, 0.3])},
            "P": {"x": markers, "g": 0.4, "c": (0.5 - gc.ALPHA) * markers,
                  "pos": np.asarray([[0.3], [0.9]]), "drift": np.full((2, 1), 0.005),
                  "Q": pull * np.eye(2)}}


# Slow: two compiles of a Gauss-Seidel group with its diagnostics.
# Per push: tests/property/test_coupling_geometry_search.py::test_the_gap_is_the_share_of_the_derivative_the_product_does_not_see
# Per push: tests/property/test_coupling_geometry_search.py::test_the_report_withholds_the_bounds_where_the_step_s_gap_is_over_the_tolerance
@pytest.mark.slow
def test_a_product_that_misses_the_geometry_term_of_a_weak_deposit_is_at_the_check_s_limit(
        monkeypatch):
    """What the self-check sees of a seeded fault, and what it does not
    (MAP-045's stated limit, measured here).  Every source-anchored
    geometry is read under ``stop_gradient``, so the pass's value follows
    the positions the scatter reads from the iterate and its product does
    not.  With a strong deposit the gap is 0.88 and the report is
    withheld.  With a weak one (a gain of 0.02: the scatter's dependence
    on the positions moves the grid's field by nine float resolutions
    over the check's step, against the 32 the difference is allowed) the
    gap is 0.22, under the tolerance, and the report stands with a
    spectral radius of 0.0030 where the honest pass's is 0.0249.  The
    same digits on jaxlib 0.10.2, 0.11.0 and 0.11.2.  A tolerance under
    0.22 would catch it, and would withhold honest passes beside a
    lattice plane (the plane hunt below), which read up to 0.29."""
    from maddening.core import _graph_specs  # noqa: PLC0415
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    cell = Cell(("target", "source"), True, "float32", 0, 100, d=1, m=2, origin=0.0,
                tolerance=1e-5)
    assert gc.KNOBS[cell.knob]["iteration_mode"] == "gauss-seidel"
    strong, weak = _deposit_pair(0.9, 0.3), _deposit_pair(0.02, 0.5)
    with gc.precision(False):
        honest_gm = gc.build(cell, cell.knobs, cell.dtype)
        _pre, _state, honest, honest_meta = gc.run_once(honest_gm, weak)
    assert honest["spectral_usable"] is True and "not_usable_reason" not in honest, honest
    assert 0.0 < float(honest_meta["geometry_gap"]) <= HONEST_GAP

    real = _graph_specs._traceable_geometry                # noqa: SLF001

    def unseen(edge, field, value):
        out = real(edge, field, value)
        return jax.lax.stop_gradient(out) if edge.geometry[0] == "source" else out

    # ``_edge_geom`` reads the name in its own module at every call.
    monkeypatch.setattr(_graph_specs, "_traceable_geometry", unseen)
    with gc.precision(False):
        gm = gc.build(cell, cell.knobs, cell.dtype)
        keys = [e.key for e in gm._edges]                  # noqa: SLF001
        _pre, _state, d, meta = gc.run_once(gm, strong)
        # The patch reached the traced pass: a term this strong is missed whole.
        assert float(meta["geometry_gap"]) > 0.8, meta["geometry_gap"]
        gg.assert_not_diagnosed(d, keys, "self-check")
        _pre, _state, d, meta = gc.run_once(gm, weak)
    gap = float(meta["geometry_gap"])
    assert 0.2 <= gap <= 0.24, gap
    # The limit: under the tolerance, so the report stands, on a radius an
    # eighth of the honest pass's.
    assert gap < _bounds.GEOMETRY_GAP_TOLERANCE
    assert "not_usable_reason" not in d and d["spectral_usable"] is True, d
    assert d["rho_spectral"] < 0.2 * honest["rho_spectral"], (d, honest)


#: The plane hunt's cells: where the pass reads positions from the
#: iterate, each anchoring, schedule, norm, dtype and dimension, at a
#: tolerance that stops an iterate a plane away from its fixed point; and
#: the anchoring where every position is a constant of the pass, whose
#: flag no plane may withdraw.  Written out: no other table's length
#: decides a cell.
PLANE_CELLS = (
    Cell(("target", "source"), True, "float32", 1, 120, tolerance=1e-3),
    Cell(("target", "source"), True, "float64", 0, 400, d=1, m=1, origin=0.0, tolerance=1e-4),
    Cell(("source", "source"), True, "float32", 0, 120, tolerance=1e-4),
    Cell(("source", "source"), True, "float64", 3, 120, d=2, m=3, tolerance=1e-4),
    Cell(("target", "source"), True, "float64", 0, 120, order=("P", "F"), tolerance=1e-4),
    Cell(("source", "target"), True, "float64", 1, 120, origin=40.0, tolerance=1e-3),
    Cell(("source", "source"), True, "float32", 2, 120, d=2, tolerance=1e-3),
    Cell(("target", "target"), True, "float32", 0, 120, tolerance=1e-4),
)
PLANE_HUNT_SEEDS = (3000, 3001)
_PLANE_HUNTS: dict = {}
#: How near a lattice plane, in spacings, a fixed point is for an honest
#: pass to read a self-check gap over :data:`HONEST_GAP` (MADD-ANO-246).
#: Measured on three jaxlib versions: every such example of the hunt (28
#: of 589 on the two cells below on 0.11.0, up to 0.096; 0.26 and 0.29 on
#: one draw on 0.10.2 and 0.11.2) is within 1.7e-5, and about a thousand
#: draws further out than 2e-5 read 4.4e-3 at most.
BESIDE_A_PLANE = 1e-4


def _reads_a_position_built_in_the_same_sweep(cell: Cell) -> bool:
    """Whether a member of *cell*'s pass reads positions that another
    member built earlier in the same sweep: Gauss-Seidel, a
    source-anchored geometry, its holder swept before its reader.  The
    self-check's step of the positions in the iterate moves such a
    position through its holder's mapped input, by about the holder's
    pull times the step, in a direction the step does not choose."""
    if gc.KNOBS[cell.knob]["iteration_mode"] != "gauss-seidel":
        return False
    down, up = cell.anchors
    first = cell.order[0]
    return (down == "source" and first == "F") or (up == "source" and first == "P")


# Slow: two seeds of 60 plane draws a search on each of eight cells, each a
# compile of the graph with its diagnostics and of its twin.
# Per push: tests/property/test_coupling_geometry_search.py::test_the_flag_is_withdrawn_with_the_fixed_point_across_a_lattice_plane
# Per push: tests/property/test_coupling_geometry_search.py::test_the_plane_limit_the_step_stores_is_its_definition
@pytest.mark.slow
@pytest.mark.parametrize("index,seed", [(i, s) for i in range(len(PLANE_CELLS))
                                        for s in PLANE_HUNT_SEEDS])
def test_the_hunt_finds_no_flag_on_a_fixed_point_beyond_twice_the_bound_across_a_plane(
        index, seed):
    """Not shrunk.  Per cell and seed: the three scores of a plane draw;
    the slot against its definition on every example; and the counts the
    record quotes (examples with the fixed point across a plane, how many
    of those had the flag before the criterion and have it now, and the
    honest ones -- same cell -- whose flag the criterion took)."""
    search = _PLANE_HUNTS.setdefault(index, gc.Search(PLANE_CELLS))
    cell = PLANE_CELLS[index]
    for name in gc.PLANE_SEARCHES:
        profile = dataclasses.replace(SLOW, max_examples=60).seeded(seed + 10 * index,
                                                                    shrink=False)
        report, fractions = search.run(name, (index,), profile=profile, planes=True)
        print(f"{name}, cell {index} ({cell!r}), seed {seed}: worst {report.score:.4g}; "
              f"{fractions}")
    counts = dict(placed=0, across=0, across_before=0, across_violating_before=0,
                  across_now=0, same=0, same_before=0, same_withheld=0,
                  across_gradient_now=0, same_gradient_before=0, same_gradient_now=0)
    for case, seen in search._seen.items():               # noqa: SLF001
        if case.cell != index or not (seen.get("placed") and seen["referenced"]):
            continue
        counts["placed"] += 1
        report = seen["report"]
        if "plane_limit" in report and math.isfinite(report["plane_limit_reference"]):
            assert report["plane_limit"] == pytest.approx(
                report["plane_limit_reference"], rel=1e-3,
                abs=report["plane_limit_resolution"]), (case, report)
        kind = "across" if seen["crossed"] else "same"
        counts[kind] += 1
        counts[f"{kind}_before"] += seen["usable_before"]
        # The gradient's flag (MADD-ANO-247): it stood wherever the spectral
        # one did on a finite bound; it now needs no plane in the
        # Kantorovich ball.  Never across a plane, its bound held wherever
        # it is set, and the honest ones (same cell) it costs are counted.
        counts[f"{kind}_gradient_now"] += seen["gradient_usable"]
        counts["same_gradient_before"] += (not seen["crossed"] and seen["spectral_usable"]
                                           and math.isfinite(
                                               report["gradient_relative_error_bound"]))
        assert seen["gradient"] <= gc.THRESHOLD["gradient"], (case, seen)
        if seen["crossed"]:
            counts["across_violating_before"] += seen["plane_before"] > 1.0
            counts["across_now"] += seen["spectral_usable"]
        else:
            counts["same_withheld"] += seen["usable_before"] and not seen["spectral_usable"]
    print(f"cell {index}, seed {seed}: {counts}")
    assert counts["placed"] >= 30, counts
    assert counts["across_gradient_now"] == 0, counts
    # The self-check on these honest passes.  Away from a plane it never
    # fires and every gap has its margin.  Beside one (MADD-ANO-246, open)
    # a pass that reads a position it built in the same sweep can read a
    # gap of any size in float32: the check's finite difference carries
    # that position across the plane.  Held: it happens nowhere else, and
    # to a few in a hundred of the cell's examples (measured: 2.4% of
    # 1383 and 1.0% of 1446 over ten more seeds), under one in ten here.
    gaps = {case: seen["report"]["geometry_gap"]
            for case, seen in search._seen.items()        # noqa: SLF001
            if case.cell == index and seen.get("placed") and seen["finite"]
            and "geometry_gap" in seen["report"]}
    fired = [case for case, seen in search._seen.items()  # noqa: SLF001
             if case.cell == index and seen["reason"] is not None
             and "lattice plane" not in seen["reason"]]
    over = {case: gap for case, gap in gaps.items() if not gap <= HONEST_GAP}
    print(f"cell {index}, seed {seed}: the self-check fired on {len(fired)} of {len(gaps)} "
          f"examples; {len(over)} gaps over {HONEST_GAP:g}, the worst "
          f"{max(gaps.values()):.3g}")
    assert set(fired) <= set(over), fired
    if _reads_a_position_built_in_the_same_sweep(cell) and cell.dtype == "float32":
        assert all(abs(case.offset) <= BESIDE_A_PLANE for case in over), over
        assert len(over) <= 0.1 * len(gaps), (len(over), len(gaps))
    else:
        assert not over, over
    if not cell.iterate_reads:
        # Constants of the pass: nothing to cross, and no flag withdrawn.
        assert counts["across"] == 0 and counts["same_withheld"] == 0, counts
        assert all(seen["report"].get("plane_limit") == math.inf
                   for case, seen in search._seen.items()                 # noqa: SLF001
                   if case.cell == index and seen.get("placed") and seen["finite"])


# ---------------------------------------------------------------------------
# What the diagnostics still do not read (each narrowing of MAP-036)
# ---------------------------------------------------------------------------


def _narrowed(*, dt_p=gc.DT, norm="l2", outside=False, diagnostics=False, **knobs):
    """The float32 pair of the first cell, with the point side at *dt_p*,
    under *norm*; with *outside* the gather comes from a third node
    outside the group and the pair is joined by plain edges."""
    from maddening.core.coupling.mapping import matrix_mapping  # noqa: PLC0415

    cell = Cell(("source", "source"), True, "float32", 0, 5)
    origin, spacing, shape = cell.grid
    gm = gc.GraphManager()
    gm.add_node(gc.GeoRelay("F", gc.DT, n=cell.n_grid, points=(cell.m, 1)))
    gm.add_node(gc.GeoRelay("P", dt_p, n=cell.m, points=(cell.m, 1)))
    gather = gc.multilinear_grid_mapping(origin, spacing, shape, n_points=cell.m)
    if outside:
        gm.add_node(gc.GeoRelay("G", gc.DT, n=cell.n_grid, points=(cell.m, 1)))
        gm.add_edge("G", "P", "x", "u", mapping=gather, geometry=("source", "pos"),
                    additive=True)
        gm.add_edge("F", "P", "x", "u", additive=True, mapping=matrix_mapping(
            np.full((cell.m, cell.n_grid), 0.1, np.float32)))
        gm.add_edge("P", "F", "x", "u", mapping=matrix_mapping(
            np.full((cell.n_grid, cell.m), 0.1, np.float32)))
    else:
        gm.add_edge("F", "P", "x", "u", mapping=gather, geometry=("source", "pos"))
        gm.add_edge("P", "F", "x", "u", geometry=("source", "pos"),
                    mapping=gc.multilinear_grid_mapping(origin, spacing, shape, n_points=cell.m,
                                                        mode="conservative"))
    live = {"tolerance": 1e-6} if norm == "l2" else {"rtol": 1e-4}
    gm.add_coupling_group(["F", "P"], convergence_norm=norm, max_iterations=20,
                          diagnostics=diagnostics, **live, **knobs)
    import warnings  # noqa: PLC0415
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    values = gc.values_of(Case(0, 8, 0.3, 0.1), cell)
    values["G"] = values["F"]
    with gc.precision(False):
        gc.set_initial(gm, {n: v for n, v in values.items() if n in gm.node_names})
        gm.step(params=gc.params_for(gm, {n: v for n, v in values.items()
                                          if n in gm.node_names}))
    return gm, dict(gm.coupling_diagnostics()[gc.KEY])


def test_a_sub_cycled_group_with_a_geometry_edge_still_reports_no_bound():
    """The geometry of each sub-step is not followed: the narrowing of
    phase 1 stands for a sub-cycled group, with a reason that says so and
    names the member (the group's diagnostics are off here, which the
    reason does not depend on); the same pair at one rate has no reason."""
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    gm, report = _narrowed(dt_p=gc.DT / 2, subcycling=True)
    gg.assert_not_diagnosed(report, [e.key for e in gm._edges], "sub-cycled")   # noqa: SLF001
    assert "['P']" in report["not_usable_reason"]
    assert "geometry_gap" not in cg.group_meta(gm, gc.KEY)
    gm, report = _narrowed()
    assert "not_usable_reason" not in report and math.isfinite(report["error_estimate"])


def test_the_interface_norm_with_a_geometry_edge_from_outside_still_reports_no_bound():
    """An internal geometry edge under the interface norm is refused at
    compile (``geometry_graphs.assert_interface_norm_refused``); one that
    enters the group from outside compiles, and its report is withheld
    with the norm named."""
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    _gm, report = _narrowed(norm="interface", outside=True)
    gg.assert_not_diagnosed(report, ["G.x->P.u"], "norm")


# Slow: a compile of a group with its diagnostics.
# Per push: tests/property/test_coupling_geometry_search.py::test_the_self_check_passes_on_every_per_push_draw_with_a_margin
@pytest.mark.slow
def test_a_geometry_held_outside_the_group_is_checked_as_a_constant_of_the_pass():
    """A source-anchored edge into the group from a node outside it: the
    positions are a constant of the pass, the self-check moves them
    there, and the group reports."""
    gm, report = _narrowed(norm="mixed", outside=True, diagnostics=True)
    assert "not_usable_reason" not in report and report["spectral_usable"] is True, report
    gap = float(cg.group_meta(gm, gc.KEY)["geometry_gap"])
    assert 0 < gap <= HONEST_GAP, gap


# ---------------------------------------------------------------------------
# The hunt
# ---------------------------------------------------------------------------

#: The hunts' seeds.
HUNT_SEEDS = (2000, 2001, 2002)
_HUNTS: dict = {}


# Slow: three seeds of 60 random examples a search on each of four blocks
# of five or six cells, each cell a compile of the graph with its
# diagnostics and of its twin.
# Per push: tests/property/test_coupling_geometry_search.py::test_a_reported_number_of_a_group_with_a_geometry_edge_holds_against_the_reference_per_push
@pytest.mark.slow
@pytest.mark.parametrize("block,seed", [(b, s) for b in range(len(BLOCKS)) for s in HUNT_SEEDS])
def test_the_hunt_finds_no_number_on_the_wrong_side_of_a_group_with_a_geometry_edge(block, seed):
    """Not shrunk: the example that fails is reported as drawn."""
    search = _HUNTS.setdefault(block, gc.Search(CELLS))   # a block's graphs compile once
    worst_gap = {"float32": 0.0, "float64": 0.0}
    limit = {"float32": HONEST_GAP, "float64": HONEST_GAP}
    for name in gc.SEARCHES + ("radius_strict",):
        profile = dataclasses.replace(SLOW, max_examples=60).seeded(seed + 10 * block,
                                                                    shrink=False)
        report, fractions = search.run(name, BLOCKS[block], profile=profile)
        print(f"{name}, block {block}, seed {seed}: worst {report.score:.4g}; {fractions}")
        _held(name, fractions)
    fired = withheld = honest = before = 0
    for case, seen in search._seen.items():               # noqa: SLF001
        took = seen["usable_before"] and not seen["spectral_usable"]
        withheld += took
        # ... of which the search's own standard holds the bound: scored
        # (same lattice cells), near (h < 1) and under the threshold.
        honest += took and seen["scored"] and seen["near_before"]
        before += seen["usable_before"]
        fired += seen["reason"] is not None and "lattice plane" not in seen["reason"]
        if seen["finite"] and "geometry_gap" in seen["report"]:
            dtype = CELLS[case.cell].dtype
            gap = seen["report"]["geometry_gap"]
            worst_gap[dtype] = max(worst_gap[dtype], gap if math.isfinite(gap) else math.inf)
    print(f"block {block}, seed {seed}: {len(search._seen)} examples, self-check fired on "  # noqa: SLF001
          f"{fired}, worst gap {worst_gap}; the lattice-plane rule withdrew {withheld} of "
          f"{before} flags, {honest} of them on a bound the reference holds (near, same "
          f"cells)")
    assert fired == 0, f"the self-check fired on {fired} honest examples"
    for dtype, gap in worst_gap.items():
        assert gap <= limit[dtype], (dtype, gap)
