"""Instruments for "the interface norm reads a mapped edge on its compact side".

``convergence_norm="interface"`` pools every entry it reads on the internal
edges of a coupling group into one RMS.  Read as *delivered*, a mapping
from a small field onto a large one (a scatter: 30 marker forces onto a
grid of ``N`` cells) puts ``N`` entries into the pool of which a few dozen
change, and the criterion is diluted: measured before the rule below
(``benchmarks/results/interface_norm_dilution``), the marker forces of a
converged group were 23 to 459 tolerances from their fixed point at ``N`` =
1e3 to 1e6, and 3.3 at every ``N`` when both readings are marker-sized.

**The decision** (maintainer, 2026-10-07): the interface norm reads a mapped
edge on its compact side.  A mapping whose target is larger than its source
is read at its source value, before the mapping; any other edge at the
value it delivers; a tie reads the delivered value.  The library does this
for every static mapping (``coupled_topologies.INTERFACE_SIDE`` is
``"compact"``; ``_interface_plan._norm_side`` is the one place), and this
module is what it is held to.  It was written, and its failing rows
pinned, while the constant read ``"delivered"``: every expectation follows
the constant, and the comparisons under the *other* rule are strict
expected failures, so each instrument is shown able to tell the two apart.

**The claim scored** (:mod:`tests.property.interface_side_graphs`):
``converged=True`` under the interface norm implies the distance to the
fixed point, in the compact readings, is at most ``K`` tolerances, ``K =
||D (I - A)^{-1} (I - L) D^{-1}||`` of the loop on its compact readings --
the identity CPL-088's bound rests on, with a converged residual at most
its threshold (CPL-048) -- which does not see ``N``.

What is here:

* the claim as a property on groups whose edges all gather (it holds on
  this tree), over 5 to 60 markers, 1e2 to 1e5 cells, dense and sparse
  mappings in either layout, both schedules, both dtypes, every stock
  acceleration, loop gains 0.2 to 0.9 of either sign;
* the claim on groups with a scatter edge at two sizes two decades apart
  (and at a million cells, slow).  Before the rule these rows were pinned
  as known-failing, strictly, on a growth of the excess with the large
  side (:func:`_dilution`, which still names that failure
  :class:`Diluted` should it come back);
* the reference of either rule against the library, residual, pass count
  and verdict: under the tree's rule it must agree everywhere, and under
  the other rule it must *disagree* wherever the two rules differ -- so the
  comparison is shown able to fail, and turns over with the constant;
* the same on relays through :class:`~tests.property.coupled_topologies.LinearModel`
  and its ``interface_side`` option: a tie, pairs of size ratio up to 100,
  and a hub whose one field a gather edge and a scatter edge both read.

**Seeded faults**, and the instrument that catches each, as measured with
``plans/tools/mutants.py`` in a scratch copy of ``src/`` on the tree that
implements the rule (the signal is the first test that failed; "core" is
``tests/core/test_the_interface_norm_reads_a_mapped_edge_on_its_compact_side.py``,
"pins" ``tests/core/test_interface_plan.py``, "search" the fifth score of
``test_coupling_targeted_search.py``).  The T rows are faults of a
delivered reading, first seeded while the library read every edge that way;
the R rows change the rule; the S rows make two of the places that describe
a group's edges disagree.

====  ======================================================  ========================================================
id    fault                                                   caught by (first failing test)
====  ======================================================  ========================================================
R1    delivered read where the source is prescribed (the      here: the scatter rows of ``..._within_K_tolerances_at_
      rule before 2026-10-07)                                 every_size`` (``Diluted``); search; core; pins
R2    the source read where delivered is prescribed: a tie    a tie: ``test_the_plain_loop_...[tie-40-40-...-
      read at its source; a gather read at its source         delivered]``; a gather: ``..._edges_all_gather_is_
                                                              within_K_tolerances``; search, core and pins for both
R3    the source value read, the delivered value's entry      here: the scatter rows (the residual is the diluted
      count pooled                                            one); search; core ``test_the_residual_reads_each_...``
R4    the side decided differently in the criterion, the      the floor: core only (``test_the_float_floor_takes_its_
      floor and the spectral analysis                         eps_on_the_side_that_is_read``: see T6).  The spectrum
                                                              (T5): ``test_an_edge_mapped_graph_s_diagnostics_are_
                                                              its_marker_side_twin_s`` (slow), and per push the
                                                              diagnosed twin of core.  A source-side reading that
                                                              is merely not counted as "the field itself" (the
                                                              report then takes a second spectrum of the same
                                                              numbers): pins only
R5    the side taken from the weights' shape (a sparse        ``..._edges_all_gather_...[gather-only-60-1000-sparse-
      layout's ``(rows, k)``)                                 jacobi-float64]``; search; core (a mapping class of
                                                              the caller's own); pins.  Every graph here is built
                                                              with zero weights and stepped with ``params``
R6    a source-side reading taken through the edge's          rows with ``offset``: ``test_the_plain_loop_...[two-
      transform                                               way-...-offset-compact]``; core; pins
R7    a field two edges read is read once (T3)                ``test_a_step_is_the_models_under_its_rule[side-hub-
                                                              ...-compact]``; pins
S1    ``compile()`` describes every edge as delivered while   core only: ``test_a_group_owns_the_floor_slot_where_an_
      the trace reads the compact side                        edge_is_read_through_its_mapping[scatter-only]`` (a
                                                              slot seeded and never written)
S2    the trace describes every edge as delivered             as R1, and core's fixture (a slot written that was
                                                              never seeded)
S3    a bare edge (the report's fallback floor, a direct      core ``test_the_residual_reads_each_...``; pins
      call of the norm) described as delivered                ``test_the_side_is_the_edges_own_wherever_it_is_
                                                              described``
T2    the pool counts the source field's entries where the    ``..._edges_all_gather_...[gather-only-5-100-matrix-
      delivered value is read                                 ...]``; search; core
T4    ``_delivered`` applies the transform before the         ``test_the_plain_loop_...[two-way-...-offset-
      mapping                                                 compact]``; core
T5    the spectral analysis reads the delivered values        see R4.  Before the rule nothing caught it (the
      (``_reading_parts`` in ``_coupled_block.py``) under a   "bound" score of the search and
      criterion on the compact ones                           ``test_interface_reading_twins.py`` let it through)
T6    the floor reads the source values of every edge         core only (the eps cell and the dead-band cell)
      (``residual_precision_floor``)
====  ======================================================  ========================================================

**What no instrument here sees.**  The float floor of the residual is
``PRECISION_FLOOR_ULPS * eps / rtol`` per entry the norm reads, whichever
entries those are, so a floor taken on the other side of a mapping is the
same number unless a reading is dead-banded on one side only or the two
sides differ in dtype: those two cells are held edge by edge in the core
file, and nothing in this module tells such a floor apart (T6, and the
floor half of R4, survive here).  IQN's ``InterfacePlan.iqn_fields``
chooses the state fields the accelerator works on, which the side rule
does not change; a wrong choice there moves pass counts, not a verdict's
truth, and is held only through the property under ``iqn-*``.  The
marker-side twin holds the spectrum to the criterion's side, and has
finite bounds to compare only on its settled rows (:data:`TWIN_SETTLED`).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import dataclasses
import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from tests.property import coupled_graphs as cg
from tests.property import coupled_topologies as ct
from tests.property import interface_side_graphs as sg
from tests.property.sysid_transform_grid import precision

#: The rule this tree's library implements.
RULE = ct.INTERFACE_SIDE


class Diluted(Exception):
    """A converged group with a scatter edge is further from its fixed point
    than ``K`` tolerances, by a factor that grows with the large side as the
    dilution of a delivered reading does (measured 0.63 to 1.20 of
    ``sqrt(large / small)`` before the rule).  Not an ``AssertionError``:
    the failure the compact side removes, named should it return."""


def _id(shape: sg.Shape) -> str:
    extra = (f"-{shape.acceleration}" if shape.acceleration != "none" else "") + (
        "-offset" if shape.offset else "")
    kind = "tie" if shape.n_large == shape.n_small else shape.kind
    return (f"{kind}-{shape.n_small}-{shape.n_large:g}-{shape.mapping}-"
            f"{shape.schedule}-{shape.dtype}{extra}")


def rules_differ(shape: sg.Shape) -> bool:
    """Does a compact-side rule read *shape* differently from a delivered one?"""
    return shape.kind != "gather-only" and shape.n_large > shape.n_small


# ---------------------------------------------------------------------------
# 1. The claim where it holds today: every edge gathers
# ---------------------------------------------------------------------------

GATHER_SHAPES = (
    sg.Shape("gather-only", 100, 5, "matrix", "gauss-seidel", "float32"),
    sg.Shape("gather-only", 1000, 60, "sparse", "jacobi", "float64"),
    sg.Shape("gather-only", 1000, 12, "matrix", "gauss-seidel", "float64", "fixed"),
    sg.Shape("gather-only", 100_000, 30, "sparse-transposed", "gauss-seidel", "float64",
             "aitken"),
)

draws = st.builds(sg.Draw, seed=st.integers(0, 2 ** 16), gain=st.floats(0.2, 0.9),
                  sign=st.sampled_from((1.0, -1.0)))


def _held(shape: sg.Shape, draw: sg.Draw) -> dict:
    seen = sg.run(shape, draw)
    assert seen["excess"] <= 1.0, (
        f"{_id(shape)}, {draw}: converged in {seen['iterations']} passes at "
        f"{seen['distance']:.3g} tolerances from the fixed point in the compact readings; "
        f"K = {seen['K']:.3g} allows {seen['K']:.3g} (largest entry error "
        f"{seen['marker_error']:.3g} tolerances)")
    return seen


@pytest.mark.parametrize("shape", GATHER_SHAPES, ids=_id)
@given(draw=draws)
def test_a_converged_group_whose_edges_all_gather_is_within_K_tolerances(shape, draw):
    _held(shape, draw)


#: IQN keeps a history of the whole interface and costs a tenth of a second
#: an example: three fixed draws per push, the property in the slow sweep.
IQN_SHAPES = (
    sg.Shape("gather-only", 300, 30, "sparse", "jacobi", "float32", "iqn-ils"),
    sg.Shape("gather-only", 400, 20, "sparse-transposed", "gauss-seidel", "float64",
             "iqn-imvj"),
)


@pytest.mark.parametrize("shape", GATHER_SHAPES + IQN_SHAPES, ids=_id)
def test_the_gather_only_shapes_converge_within_K_tolerances(shape):
    """The property above scores 0 where a group does not converge: it does."""
    seen = [_held(shape, sg.Draw(seed, gain)) for seed, gain in ((1, 0.2), (2, 0.6), (3, 0.9))]
    assert all(s["converged"] for s in seen), [s["report"] for s in seen]
    assert all(s["excess"] > 0 for s in seen)


SCATTER_REFERENCES = tuple(
    sg.Shape(kind, n, m, schedule=schedule, offset=offset)
    for kind in ("two-way", "scatter-only") for n, m in ((1000, 5), (100_000, 60))
    for schedule, offset in (("gauss-seidel", 0.0), ("jacobi", 8.0)))


@pytest.mark.parametrize("shape", SCATTER_REFERENCES, ids=_id)
@given(draw=draws)
def test_the_compact_rule_itself_keeps_the_claim_on_a_scatter(shape, draw):
    """No graph: the reference's own plain loop under the compact rule stops
    within ``K`` tolerances at every size.  ``K`` is the rule's constant, not
    a number fitted to gather-only measurements."""
    ref = sg.Reference(shape, draw)
    out = ref.plain_exit("compact")
    assert out["converged"]
    assert ref.distance(out["state"]) <= ref.K * (1.0 + 1e-9), (ref.distance(out["state"]), ref.K)


# ---------------------------------------------------------------------------
# 2. The claim on a scatter edge, at sizes two and three decades apart
# ---------------------------------------------------------------------------

#: The pinned draw: a loop gain at which one pass moves the error by less
#: than a factor of two under either schedule, so the pass a solve stops on
#: cannot hide a tenfold dilution.
PIN = sg.Draw(seed=1, gain=0.6)
SMALL, LARGE, HUGE = 1000, 100_000, 1_000_000
#: The least growth of the excess between two sizes that counts as the
#: dilution of a delivered reading, as a fraction of ``sqrt(large / small)``
#: (the entry count's growth: 10 between 1e3 and 1e5).  Measured before the
#: rule at the pinned draw: 0.63 to 1.20 of it over the pinned rows, the
#: same on jaxlib 0.10.2, 0.11.0 and 0.11.2; a third leaves most of a
#: factor of two.
GROWTH_FRACTION = 1.0 / 3.0

PINNED = (
    ("two-way", "matrix", "gauss-seidel", "float64", "none"),
    ("two-way", "sparse", "jacobi", "float32", "none"),
    ("scatter-only", "sparse", "gauss-seidel", "float32", "none"),
    ("scatter-only", "sparse-transposed", "jacobi", "float64", "fixed"),
)

def _dilution(row, small: int, large: int, draw: sg.Draw = PIN) -> None:
    """Hold the claim at *small* and at *large* cells.

    Three outcomes.  The claim holds at both sizes: returns.  The claim
    fails at *large* by a factor that grew from *small* as the entry
    count did: :class:`Diluted`, the failure of a delivered reading
    (while the library read every edge as delivered these rows were
    strict expected failures raising exactly that).  Anything else -- a
    solve that did not converge, a failure that does not grow with the
    size -- is an ``AssertionError``.
    """
    kind, mapping, schedule, dtype, acceleration = row
    seen = {n: sg.run(sg.Shape(kind, n, 30, mapping, schedule, dtype, acceleration), draw)
            for n in (small, large)}
    assert all(s["converged"] for s in seen.values()), {n: s["report"] for n, s in seen.items()}
    lo, hi = seen[small]["excess"], seen[large]["excess"]
    if hi <= 1.0:
        assert lo <= 1.0, f"{row}: the claim fails at {small} cells ({lo:.3g}) and not at {large}"
        return
    growth = hi / lo
    least = GROWTH_FRACTION * math.sqrt(large / small)
    assert growth >= least, (
        f"{row}: {hi:.3g} K tolerances from the fixed point at {large} cells but "
        f"{lo:.3g} at {small}: a factor {growth:.3g}, where the dilution of a scatter's "
        f"delivered value grows by at least {least:.3g}.  Another defect.")
    raise Diluted(f"{row}: {lo:.3g} K tolerances at {small} cells, {hi:.3g} at {large}")


@pytest.mark.parametrize("row", PINNED, ids=["-".join(r) for r in PINNED])
def test_a_converged_group_with_a_scatter_edge_is_within_K_tolerances_at_every_size(row):
    _dilution(row, SMALL, LARGE)


# Slow: a million cells (a dense mapping of that size is 240 MB a copy, so
# the sparse layouts only).
# Per push: tests/property/test_coupling_interface_side.py::test_a_converged_group_with_a_scatter_edge_is_within_K_tolerances_at_every_size
@pytest.mark.slow
@pytest.mark.parametrize("row", [r for r in PINNED if r[1] != "matrix"],
                         ids=["-".join(r) for r in PINNED if r[1] != "matrix"])
def test_a_converged_group_with_a_scatter_edge_is_within_K_tolerances_at_a_million_cells(row):
    _dilution(row, SMALL, HUGE)


def _sweep_shapes(kind: str, acceleration: str, schedule: str):
    """Eight shapes: every size with 5 and with 60 markers, the mapping
    layouts and the dtypes rotated over them.  IQN keeps a history of the
    whole interface, so it stops at 1e4 cells."""
    sizes = (100, 1000, 10_000) if acceleration.startswith("iqn") else (
        100, 1000, 10_000, 100_000)
    out = []
    for i, n in enumerate(sizes):
        for j, m in enumerate((5, 60)):
            k = 2 * i + j + len(acceleration) + len(schedule)
            out.append(sg.Shape(kind, n, m, sg.MAPPINGS[k % 3], schedule,
                                ("float64", "float32")[(i + j) % 2], acceleration))
    return out


SWEEP = [(kind, acceleration, schedule) for kind in sg.KINDS
         for acceleration in sg.ACCELERATIONS for schedule in ("gauss-seidel", "jacobi")]


# Slow: 30 configurations of eight compiled shapes, three draws each.
# Per push: tests/property/test_coupling_interface_side.py::test_a_converged_group_whose_edges_all_gather_is_within_K_tolerances
@pytest.mark.slow
@pytest.mark.parametrize("kind,acceleration,schedule", SWEEP,
                         ids=["-".join(row) for row in SWEEP])
def test_the_claim_over_every_configuration(kind, acceleration, schedule):
    """Every stock acceleration under both schedules, at every size: the
    claim is asserted on every kind, and the worst excess per size is
    printed (the measurement, kept current)."""
    worst: dict = {}
    for shape in _sweep_shapes(kind, acceleration, schedule):
        for seed, (gain, sign) in enumerate(((0.2, 1.0), (0.55, -1.0), (0.9, 1.0))):
            draw = sg.Draw(seed, gain, sign)
            seen = _held(shape, draw)
            assert seen["converged"], (_id(shape), draw, seen["report"])
            worst[shape.n_large] = max(worst.get(shape.n_large, 0.0), seen["excess"])
    print(f"{kind} {acceleration} {schedule}: worst distance over K tolerances by size "
          + ", ".join(f"{n:g}: {v:.3g}" for n, v in sorted(worst.items())))


# ---------------------------------------------------------------------------
# 3. The reference of each rule against the library
# ---------------------------------------------------------------------------

EXIT_SHAPES = (
    sg.Shape("gather-only", 2000, 20, "matrix", "gauss-seidel", "float64"),
    sg.Shape("gather-only", 2000, 20, "sparse-transposed", "jacobi", "float32"),
    sg.Shape("gather-only", 2000, 20, "sparse", "gauss-seidel", "float64", offset=8.0),
    sg.Shape("two-way", 2000, 20, "matrix", "gauss-seidel", "float64"),
    sg.Shape("two-way", 2000, 20, "sparse", "jacobi", "float32"),
    sg.Shape("two-way", 2000, 20, "sparse-transposed", "gauss-seidel", "float64", offset=8.0),
    sg.Shape("scatter-only", 2000, 20, "sparse", "gauss-seidel", "float32"),
    sg.Shape("scatter-only", 2000, 20, "sparse-transposed", "jacobi", "float64", offset=8.0),
    sg.Shape("scatter-only", 100, 60, "sparse-transposed", "jacobi", "float64"),
    # A tie: both mappings square, read as delivered under either rule.
    sg.Shape("two-way", 40, 40, "sparse", "gauss-seidel", "float64"),
    sg.Shape("two-way", 40, 40, "matrix", "jacobi", "float64", offset=8.0),
)
#: A predicted pass count is asserted where every estimate the reference's
#: loop compared was at least this factor from the threshold: five
#: thousand times a float32 residual's rounding.
EXIT_MARGIN = 1.05


def _exit_draw(shape: sg.Shape, rule: str) -> sg.Draw:
    """The first seed whose exit under *rule* is decided with margin (the
    reference's float64 arithmetic alone: the same seed on every platform)."""
    for seed in range(40):
        draw = sg.Draw(seed, 0.6)
        if sg.Reference(shape, draw).plain_exit(rule)["margin"] >= EXIT_MARGIN:
            return draw
    raise AssertionError(f"{_id(shape)}: no seed decides its exit with margin")


def _exit_rows():
    for shape in EXIT_SHAPES:
        for rule in ("delivered", "compact"):
            wrong = rule != RULE and rules_differ(shape)
            marks = [pytest.mark.xfail(strict=True, raises=AssertionError,
                                       reason=f"the library reads {RULE!r}, which is not "
                                              f"{rule!r} on this shape")] if wrong else []
            yield pytest.param(shape, rule, id=f"{_id(shape)}-{rule}", marks=marks)


def _assert_returned(shape, seen, want) -> None:
    """The state the step returned is *want*, to the rounding of its passes."""
    close = 1e-9 if shape.dtype == "float64" else 2e-4
    for name in ("p", "q"):
        gap = float(np.max(np.abs(seen["state"][name] - want[name])))
        assert gap <= close * float(np.max(np.abs(want[name]))), (
            f"{name} is {gap:.3e} from the accepted iterate with its unmeasured fields "
            f"one pass on (measured whole: {seen['whole']})")


@pytest.mark.parametrize("shape,rule", list(_exit_rows()))
def test_the_plain_loop_stops_where_the_reference_of_its_rule_does(shape, rule):
    """Residual, verdict and pass count of ``acceleration="none"`` against the
    reference under *rule*.  Under the tree's rule: everywhere.  Under the
    other: only where the two rules read the same thing (every edge a
    gather, or a tie); elsewhere the comparison must fail (strict)."""
    draw = _exit_draw(shape, rule)
    seen = sg.run(shape, draw)
    ref = seen["reference"]
    expected = ref.plain_exit(rule)
    # A float32 residual is a float32 sum of squares of float32 readings.
    tight = 1e-9 if shape.dtype == "float64" else 2e-3
    # The report is of the iterate the loop accepted: the reference's own
    # loop stops on it, with this residual.
    assert abs(seen["residual"] - expected["residual"]) <= tight * expected["residual"], (
        f"the step reports residual {seen['residual']!r}; the {rule} reading of the iterate "
        f"the reference's loop accepts gives {expected['residual']!r}")
    assert seen["converged"] == expected["converged"]
    assert seen["iterations"] == expected["iterations"], (
        f"the step took {seen['iterations']} passes; the plain loop under the {rule} "
        f"reading stops after {expected['iterations']} (margin {expected['margin']:.3g})")
    # ... and the state returned is that iterate with the fields the norm
    # does not measure whole one plain pass on (both, behind two mappings).
    _assert_returned(shape, seen, ref.returned(expected["state"], seen["whole"]))


TWIN_SHAPES = (
    sg.Shape("two-way", 2000, 20, "sparse", "gauss-seidel", "float64"),
    sg.Shape("two-way", 500, 6, "matrix", "jacobi", "float64"),
    sg.Shape("two-way", 2000, 20, "sparse-transposed", "gauss-seidel", "float32"),
)
#: The same with ``diagnostics=True`` (twelve seconds a compile: slow).  On
#: these three the loop closes on 12 or 40 numbers, which the report's
#: eight Krylov steps do not resolve: both graphs report
#: ``spectral_usable=False``, an infinite ``spectral_error_bound`` and no
#: gradient bound, so what is compared of the spectrum is the unsettled
#: ``rho_spectral`` alone.
TWIN_DIAGNOSED = tuple(dataclasses.replace(s, diagnostics=True) for s in TWIN_SHAPES)
#: Diagnosed rows whose spectrum settles (three or four markers: a loop on
#: six or eight numbers), so that ``spectral_usable`` holds in both graphs
#: and ``spectral_error_bound`` -- and, on the rows marked, the gradient
#: bound -- are finite numbers a spectrum taken on the wrong side moves.
#: ``(shape, the gradient bound is reported)``.  Measured on jaxlib 0.11.0:
#: every key agrees to 1e-7 or better in both dtypes, the gradient bound
#: included (the scatter's weights are parameters of the step in one graph
#: and constants of a node in the other; the bound's own norm is the raw
#: source fields, which the two graphs share).
#:
#: **Not a row: float32 with the ``sparse-transposed`` layout.**  There the
#: edge-mapped graph sums a scatter in another order than the twin's node
#: does, the two passes differ in the last bit, and where the Krylov space
#: is exhausted before eight steps (four markers or fewer under
#: Gauss-Seidel) the float32 bounds amplify that rounding: measured
#: ``spectral_error_bound`` 1.60 against 1.17 with three markers, and the
#: gradient bound 2.9e-4 against 4.4e-4 with four, on two programs of one
#: problem (jaxlib 0.11.0).  Both bound the distance; neither is the other
#: to 2%.  The same layout in float64 is a row.
TWIN_SETTLED = (
    (sg.Shape("two-way", 500, 3, "matrix", "jacobi", "float64", diagnostics=True), True),
    (sg.Shape("two-way", 2000, 3, "sparse", "gauss-seidel", "float64", diagnostics=True), True),
    (sg.Shape("two-way", 2000, 4, "sparse-transposed", "jacobi", "float64", diagnostics=True),
     False),
    (sg.Shape("two-way", 2000, 4, "sparse", "gauss-seidel", "float32", diagnostics=True), True),
    (sg.Shape("two-way", 500, 4, "matrix", "jacobi", "float32", diagnostics=True), False),
)
#: Every number of a report, to the rounding of two differently ordered
#: float evaluations of one problem (the spectral keys come from eight
#: Krylov steps of it).
TWIN_RTOL = {"float64": 1e-6, "float32": 2e-2}


def _twin_reports(shape: sg.Shape):
    draw = _exit_draw(shape, "compact")
    seen = sg.run(shape, draw)
    ref = seen["reference"]
    with precision(shape.dtype == "float64"):
        twin = sg.run(shape, draw, graph=sg.marker_side_twin(shape, ref))
    return ref, seen, twin


@pytest.mark.parametrize("shape", TWIN_SHAPES, ids=_id)
def test_the_marker_side_twin_stops_where_the_compact_reference_does(shape):
    """The twin is the compact rule built out of plain edges: valid today."""
    ref, _seen, twin = _twin_reports(shape)
    expected = ref.plain_exit("compact")
    tight = 1e-9 if shape.dtype == "float64" else 2e-3
    assert abs(twin["residual"] - expected["residual"]) <= tight * twin["residual"]
    assert (twin["iterations"], twin["converged"]) == (expected["iterations"], True)
    # ``p`` is read by a plain edge and kept; ``q``, behind the gather, is
    # one pass on.
    assert twin["whole"] == ("p",)
    _assert_returned(shape, twin, ref.returned(expected["state"], twin["whole"]))
    assert twin["excess"] <= 1.0


def _same_reports(shape: sg.Shape) -> tuple:
    """Every key of the two reports, to :data:`TWIN_RTOL`; ``(edge-mapped, twin)``."""
    _ref, seen, twin = _twin_reports(shape)
    a, b = seen["report"], twin["report"]
    assert sorted(a) == sorted(b)
    rtol = TWIN_RTOL[shape.dtype]
    differ = {}
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, (bool, np.bool_, str, type(None))) or isinstance(vb, (str, type(None))):
            same = va == vb
        else:
            va, vb = float(va), float(vb)
            # Two NaNs (a bound not made) and two equal infinities (a bound
            # on an unsettled spectrum) are the same report.
            same = (math.isnan(va) and math.isnan(vb)) or va == vb or abs(va - vb) <= rtol * max(
                abs(va), abs(vb))
        if not same:
            differ[key] = (va, vb)
    assert not differ, f"{_id(shape)}: (edge-mapped, twin) {differ}"
    return a, b


def test_the_two_rules_stop_on_different_passes_where_they_differ():
    """The comparison above can tell the rules apart by the pass count alone."""
    differ = [s for s in EXIT_SHAPES if rules_differ(s)]
    same = [s for s in EXIT_SHAPES if not rules_differ(s)]
    assert differ and same
    for shape in differ:
        ref = sg.Reference(shape, sg.Draw(0, 0.6))
        assert (ref.plain_exit("delivered")["iterations"]
                < ref.plain_exit("compact")["iterations"]), _id(shape)
    for shape in same:
        ref = sg.Reference(shape, sg.Draw(0, 0.6))
        a, b = ref.plain_exit("delivered"), ref.plain_exit("compact")
        assert (a["iterations"], a["residual"]) == (b["iterations"], b["residual"]), _id(shape)


def test_a_source_side_reading_does_not_carry_the_edges_transform():
    """The step applies the mapping and then the transform, so the value
    before the mapping has been through neither: an offset on a scatter
    edge is in the delivered reading and not in the compact one."""
    plain = sg.Reference(sg.Shape("two-way", 2000, 20), sg.Draw(0, 0.6))
    shifted = sg.Reference(sg.Shape("two-way", 2000, 20, offset=8.0), sg.Draw(0, 0.6))
    x = shifted.fixed_point
    np.testing.assert_array_equal(shifted.readings(x, "compact")[0], x["p"])
    np.testing.assert_allclose(shifted.readings(x, "delivered")[0],
                               plain.readings(x, "delivered")[0] + 8.0, rtol=0, atol=1e-12)
    # The gather edge of the pair is read as delivered under both rules.
    np.testing.assert_array_equal(shifted.readings(x, "compact")[1],
                                  shifted.readings(x, "delivered")[1])


def test_the_side_rule_is_larger_target_reads_source_and_a_tie_reads_delivered():
    assert sg.side_of(30, 1000) == "source"
    assert sg.side_of(1000, 30) == "delivered"
    assert sg.side_of(40, 40) == "delivered"
    assert sg.side_of(30, 1000, "delivered") == "delivered"
    assert sg.side_of(1000, 30, "source") == "source"


# ---------------------------------------------------------------------------
# 4. The exact model's per-side option
# ---------------------------------------------------------------------------

SIDE = ct.side_topologies()
CAP = 200
RTOL = 1e-4

#: ``(structure, mapping kind, dtype, schedule)``.
MODEL_ROWS = (
    ("side-4-4", "sparse-local", "float64", "jacobi"),
    ("side-4-4", "matrix", "float32", "gauss-seidel"),
    ("side-3-6", "matrix-local", "float64", "gauss-seidel"),
    ("side-2-12", "matrix", "float64", "gauss-seidel"),
    ("side-2-12-r", "sparse-local", "float32", "jacobi"),
    ("side-3-300", "sparse-local", "float64", "gauss-seidel"),
    ("side-hub", "matrix-local", "float64", "gauss-seidel"),
    ("side-hub", "sparse-local", "float32", "jacobi"),
)


def _knobs(schedule: str) -> dict:
    return cg.live_knobs(dict(acceleration="none", iteration_mode=schedule,
                              convergence_norm="interface", rtol=RTOL, solver="ift",
                              max_iterations=CAP))


def _topology_rules_differ(topo: ct.Topology) -> bool:
    return any(e.mapped and topo.node(e.dst).n > topo.node(e.src).n
               for e in (topo.edges[i] for i in topo.internal_edges(0)))


def _model_rows():
    for row in MODEL_ROWS:
        for rule in ("delivered", "compact"):
            wrong = rule != RULE and _topology_rules_differ(SIDE[row[0]])
            marks = [pytest.mark.xfail(strict=True, raises=AssertionError,
                                       reason=f"the library reads {RULE!r}, which is not "
                                              f"{rule!r} on this structure")] if wrong else []
            yield pytest.param(row, rule, id="-".join(row) + f"-{rule}", marks=marks)


@pytest.mark.parametrize("row,rule", list(_model_rows()))
def test_a_step_is_the_models_under_its_rule(row, rule):
    """One step of a relay structure against ``LinearModel(interface_side=rule)``:
    the reported residual is the model's reading of the returned state, the
    state is within what that residual allows, and the plain loop stopped on
    the pass, with the verdict, the model's restated loop does."""
    structure, kind, dtype, schedule = row
    topo, knobs = SIDE[structure], _knobs(schedule)
    cfgs = ct.group_cfgs_of([knobs])
    with precision(dtype == "float64"):
        built = ct.build(topo, knobs, dtype=dtype, mapping_kind=kind)
        for seed in range(40):
            values = ct.side_values(topo, np.random.default_rng(seed), 0.6, dtype=dtype,
                                    group_cfgs=cfgs, mapping_kind=kind)
            model = ct.LinearModel(topo, values, dtype=dtype, group_cfgs=cfgs,
                                   interface_side=rule)
            pre = {nd.name: {"x": np.asarray(values["nodes"][nd.name]["x0"])}
                   for nd in topo.nodes}
            expected = model.plain_exit(0, pre, pre, CAP)
            if expected["margin"] >= EXIT_MARGIN:
                break
        else:
            raise AssertionError(f"{row}: no seed decides its exit with margin")
        (step,) = ct.run(built, values, 1)
    report = step.reports[0]
    # The report is of the iterate the loop accepted, which the model's
    # restated loop gives; the state returned is that iterate with the
    # fields the norm does not measure whole one plain pass on.
    restated = expected["residual"]
    tight = 1e-9 if dtype == "float64" else 2e-3
    assert abs(float(report["residual"]) - restated) <= tight * restated, (
        f"the step reports residual {report['residual']!r}; the {rule} reading of the "
        f"iterate the model's loop accepts gives {restated!r}")
    close = 1e-9 if dtype == "float64" else 2e-4
    for name, want in expected["returned"].items():
        got = np.asarray(step.state[name]["x"], np.float64)
        assert np.max(np.abs(got - want)) <= close * max(float(np.max(np.abs(want))), 1.0), (
            row, name, got, want)
    model.check_step(step.pre, step.state, step.reports, thresholds=[1.0],
                     where="-".join(row))
    assert bool(report["converged"]) == expected["converged"]
    assert int(report["iterations"]) == expected["iterations"], (
        f"{report['iterations']} passes; the model's loop under {rule!r} stops after "
        f"{expected['iterations']} (margin {expected['margin']:.3g})")


def _fields(model: ct.LinearModel) -> list:
    return [B for B, _gamma in model.norm_fields(0)]


@pytest.mark.parametrize("structure", sorted({**ct.search_topologies(), **SIDE}))
def test_the_per_side_option_changes_only_edges_onto_a_larger_target(structure):
    """The default is the tree's rule; ``"compact"`` reads the source's
    field itself on a mapped edge onto a larger target and leaves every
    other reading -- a smaller target, a tie, an unmapped edge -- as the
    delivered rule has it."""
    topo = {**ct.search_topologies(), **SIDE}[structure]
    knobs = _knobs("gauss-seidel")
    cfgs = ct.group_cfgs_of([knobs])
    draw = ct.side_values if structure in SIDE else ct.draw_values
    values = draw(topo, np.random.default_rng(0), 0.5, dtype="float64", group_cfgs=cfgs)
    make = lambda **kw: ct.LinearModel(topo, values, dtype="float64",  # noqa: E731
                                       group_cfgs=cfgs, **kw)
    default, delivered, compact = make(), make(interface_side="delivered"), make(
        interface_side="compact")
    for a, b in zip(_fields(default), _fields(make(interface_side=RULE))):
        np.testing.assert_array_equal(a, b)
    members, off, _k = compact._group_layout(0)
    for i, B_d, B_c in zip(topo.internal_edges(0), _fields(delivered), _fields(compact)):
        e = topo.edges[i]
        n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
        larger = e.mapped and n_dst > n_src
        assert compact.interface_side_of(i) == ("source" if larger else "delivered")
        assert delivered.interface_side_of(i) == "delivered"
        if not larger:
            np.testing.assert_array_equal(B_c, B_d)
            continue
        # The source's own entries, once each, with no factor of a transform.
        assert B_c.shape[0] == n_src and B_d.shape[0] == n_dst
        np.testing.assert_array_equal(B_c[:, off[e.src]:off[e.src] + n_src], np.eye(n_src))
        assert np.count_nonzero(B_c) == n_src
    if not _topology_rules_differ(topo):
        # Where no target is larger the two references are one reference.
        pre = {nd.name: {"x": np.asarray(values["nodes"][nd.name]["x0"])} for nd in topo.nodes}
        a, b = delivered.plain_exit(0, pre, pre, CAP), compact.plain_exit(0, pre, pre, CAP)
        assert (a["iterations"], a["residual"], a["converged"]) == (
            b["iterations"], b["residual"], b["converged"])


def test_a_declared_side_overrides_the_sizes_edge_by_edge():
    topo = SIDE["side-2-12"]
    cfgs = ct.group_cfgs_of([_knobs("jacobi")])
    values = ct.side_values(topo, np.random.default_rng(0), 0.5, dtype="float64",
                            group_cfgs=cfgs)
    gather, scatter = topo.internal_edges(0)
    assert (topo.edges[gather].src, topo.edges[scatter].src) == ("g", "s")
    sizes = lambda side: [B.shape[0] for B in _fields(ct.LinearModel(  # noqa: E731
        topo, values, dtype="float64", group_cfgs=cfgs, interface_side=side))]
    assert sizes("delivered") == [2, 12]
    assert sizes("compact") == [2, 2]
    assert sizes({scatter: "delivered"}) == [2, 12]
    assert sizes({gather: "source"}) == [12, 2]
    assert sizes({}) == sizes("compact")


def test_a_hubs_field_is_read_once_per_edge_by_each_edges_own_rule():
    """``h`` (4) feeds a gather edge (to 2) and a scatter edge (to 8): the
    compact rule reads its field twice, as two delivered entries and as its
    own four, and the two leaves' fields by their own edges' rules."""
    topo = SIDE["side-hub"]
    cfgs = ct.group_cfgs_of([_knobs("jacobi")])
    values = ct.side_values(topo, np.random.default_rng(0), 0.5, dtype="float64",
                            group_cfgs=cfgs)
    model = ct.LinearModel(topo, values, dtype="float64", group_cfgs=cfgs,
                           interface_side="compact")
    _members, off, _k = model._group_layout(0)
    read = {}
    for i, B in zip(topo.internal_edges(0), _fields(model)):
        e = topo.edges[i]
        read[(e.src, e.dst)] = (model.interface_side_of(i), B.shape[0])
        cols = np.flatnonzero(np.abs(B).sum(axis=0))
        assert set(cols) <= set(range(off[e.src], off[e.src] + topo.node(e.src).n))
    assert read == {("h", "s"): ("delivered", 2), ("h", "g"): ("source", 4),
                    ("s", "h"): ("source", 2), ("g", "h"): ("delivered", 4)}


@pytest.mark.parametrize("shape", TWIN_SHAPES, ids=_id)
def test_an_edge_mapped_graph_reports_what_its_marker_side_twin_reports(shape):
    """Every key of ``coupling_diagnostics()`` without diagnostics: the
    passes, the residual, the estimate and the verdict."""
    _same_reports(shape)


# Slow: six graphs compiled with diagnostics.
# Per push: tests/property/test_coupling_interface_side.py::test_an_edge_mapped_graph_reports_what_its_marker_side_twin_reports
@pytest.mark.slow
@pytest.mark.parametrize("shape", TWIN_DIAGNOSED, ids=_id)
def test_an_edge_mapped_graph_s_diagnostics_are_its_marker_side_twin_s(shape):
    """With ``diagnostics=True``: the criterion, the float floor behind
    ``precision_limited`` and the spectral analysis each enumerate the
    edges for themselves, and the three must read one side -- the spectral
    radius, the error bound and the gradient bound of the two graphs are
    then the same numbers.  (On these rows the spectrum does not settle:
    see :data:`TWIN_DIAGNOSED`, and the settled rows below.)"""
    _same_reports(shape)


# Slow: ten graphs compiled with diagnostics.
# Per push: tests/core/test_the_interface_norm_reads_a_mapped_edge_on_its_compact_side.py::test_a_diagnosed_edge_mapped_pair_reports_every_number_of_its_marker_side_twin
@pytest.mark.slow
@pytest.mark.parametrize("shape,gradient", TWIN_SETTLED, ids=[_id(s) for s, _g in TWIN_SETTLED])
def test_a_settled_spectrum_and_its_bounds_are_the_marker_side_twin_s(shape, gradient):
    """The same comparison where it has numbers to compare: the spectrum
    settles in both graphs, ``spectral_error_bound`` is finite, and on the
    rows that report a gradient bound that is finite too.  A spectral
    analysis left on the delivered side of the scatter (the fault no
    instrument saw while the criterion read that side too) moves these."""
    mapped, twin = _same_reports(shape)
    for report in (mapped, twin):
        assert bool(report["converged"]) and bool(report["spectral_usable"]), report
        assert math.isfinite(float(report["spectral_error_bound"])), report
        assert bool(report["gradient_bound_usable"]) is gradient, report
        assert math.isfinite(float(report["gradient_relative_error_bound"])) is gradient, report
