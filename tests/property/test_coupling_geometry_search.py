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
over every cell of :data:`CELLS`, three seeds; a member whose tangent is
half its derivative, end to end; a geometry held by a node outside the
group.

**Seeded faults** (``plans``-side mutant list; the table in
``tests/property/geometry_graphs.py`` names them): the geometry term
dropped from the product, the product taken with the geometry of another
iterate, a target-anchored geometry read from the iterate, and the
self-check's comparison disabled are each caught by a per-push test of
this module or of the geometry harness.
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
                out.append(Cell(anchors, moving, dtype, (a + 3 * t + 2 * v) % len(gc.KNOBS),
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
GS_RADIUS_SEEDS = {"the-geometry-raises-the-radius": (Case(0, 7, 0.05, 0.1), 0.25),
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
        assert seen["scored"] and seen["spectral_usable"] and seen["reason"] is None, seen
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
    draws = _per_push_draws()
    assert len(draws) >= 20
    gaps = []
    for case in draws:
        seen = SEARCH.observe(case)
        assert seen["reason"] is None, (case, seen["reason"])
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


def test_the_report_withholds_the_bounds_where_the_step_s_gap_is_over_the_tolerance():
    """The report's side of the self-check, on the per-push graph with the
    slot the step wrote replaced: at the tolerance the bounds stand, above
    it and at NaN they are withheld with the reason, the gap in it."""
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
            for gap, fails in ((_bounds.GEOMETRY_GAP_TOLERANCE, False), (0.26, True),
                               (1.0, True), (math.nan, True)):
                gm._state = {**kept, "_meta": {                    # noqa: SLF001
                    **kept["_meta"], slot: np.asarray(gap, meta["geometry_gap"].dtype)}}
                report = dict(gm.coupling_diagnostics()[gc.KEY])
                if not fails:
                    assert report == honest, (gap, report)
                    continue
                gg.assert_not_diagnosed(report, keys, "self-check")
                assert f"relative gap {gap:.3g}, allowed 0.25" in report["not_usable_reason"]
                (row,) = list(gm.coupling_report())
                assert any("disagrees with a finite difference" in f for f in row["flags"])
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
    fired = 0
    for case, seen in search._seen.items():               # noqa: SLF001
        fired += seen["reason"] is not None
        if seen["finite"] and "geometry_gap" in seen["report"]:
            dtype = CELLS[case.cell].dtype
            gap = seen["report"]["geometry_gap"]
            worst_gap[dtype] = max(worst_gap[dtype], gap if math.isfinite(gap) else math.inf)
    print(f"block {block}, seed {seed}: {len(search._seen)} examples, self-check fired on "  # noqa: SLF001
          f"{fired}, worst gap {worst_gap}")
    assert fired == 0, f"the self-check fired on {fired} honest examples"
    for dtype, gap in worst_gap.items():
        assert gap <= limit[dtype], (dtype, gap)
