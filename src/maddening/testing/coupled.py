"""The coupled level: the order of accuracy of a whole graph.

Verified nodes joined by verified edges are still not a verified model:
the coupled scheme has an order of its own, set by how the exchange is
advanced in time and by what the edges reproduce.  This module measures
it, with the machinery of :mod:`maddening.testing.mms` and nothing new
numerically: :func:`verify_graph_order` runs a refinement ladder against
a known solution, :func:`verify_graph_gci` runs one that compares the
levels with each other (no known solution needed).

Two things differ from the node level.

**A graph declares no order.**  The caller states the order expected of
the coupled scheme (``expected=``), and without it the result is a
``SKIP`` that says so.  What a partitioned coupling can reach: every
node receives its boundary inputs once per step, so the exchange is
first order in time whatever the nodes' own integrators do
(MADD-ANO-014), and an edge whose mapping reproduces polynomials of
degree ``p`` transfers a smooth field to order ``p + 1`` in the grid
spacing.

**The iteration error is not the discretisation error.**  A coupling
group iterates each step's exchange to a tolerance.  What is left of
that iteration is an error of the *solver*, and a refinement ladder
reads it as if it were an error of the *scheme*: a ladder run at a loose
tolerance mixes the two and reports an order that belongs to neither
(measured on two rods exchanging heat along their length, at
``tolerance=1e-4``: pairwise orders 1.01, 1.13, 0.51 where the scheme's
is 1.00).  Both functions therefore take a guard and refuse to report an
order as verified without it.  The guard asks, at every level, that the
iteration error is below ``iteration_factor`` of the discretisation
error being measured, by one of two routes:

* from ``coupling_diagnostics()`` (:func:`coupling_iteration_bound`),
  where every group reports a usable bound -- ``solver="ift"`` with
  ``diagnostics=True`` and ``spectral_usable`` true;
* otherwise by running the level again at a tighter coupling tolerance
  and comparing the measured quantity itself (``tightened_error_at`` /
  ``tightened_solution_at``).  This route assumes nothing about the
  graph: a custom edge, a geometry-dependent mapping, ``solver="fori"``
  and diagnostics switched off are all the same to it.

Where neither can be made the result says so, and is a ``SKIP``, not a
pass.

Requires ``jax`` and ``numpy``; imported from ``maddening.testing``,
which needs the ``[verify]`` extra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Sequence

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.testing.mms import (
    DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE,
    DEFAULT_ORDER_EXCESS,
    DEFAULT_ORDER_SHORTFALL,
    DEFAULT_STAGNATION_RTOL,
    InconclusiveStudyError,
    RefinementAxis,
    UndeclaredOrderError,
    check_gci,
    check_order,
    measure_gci,
    measure_order,
)
from maddening.testing.verification import VerificationResult

__all__ = [
    "DEFAULT_ITERATION_ERROR_FACTOR",
    "IterationErrorBound",
    "coupling_iteration_bound",
    "verify_graph_order",
    "assert_graph_order_verified",
    "verify_graph_gci",
    "assert_graph_gci_verified",
]

#: How far below the discretisation error the iteration error must be,
#: as a fraction of it.  An error known to within a fraction ``f`` at two
#: adjacent levels moves the order read from them by at most
#: ``log((1 + f) / (1 - f)) / log(r)``: 0.14 at this value for a
#: refinement ratio ``r`` of 2, inside the 0.25 the order band allows
#: (:data:`~maddening.testing.mms.DEFAULT_ORDER_SHORTFALL`).  At 0.1 it
#: is 0.29, outside it.
DEFAULT_ITERATION_ERROR_FACTOR = 0.05  # units: fraction of the discretisation error


@stability(StabilityLevel.EXPERIMENTAL)
@dataclass(frozen=True)
class IterationErrorBound:
    """What ``coupling_diagnostics()`` says of a run's iteration error.

    Attributes
    ----------
    usable : bool
        Whether :attr:`accumulated` is a number the guard can use: every
        coupling group of the graph reported a usable bound, or the
        graph has no coupling group (nothing iterates, the bound is 0).
    per_step : float
        The largest ``spectral_error_bound`` among the groups: the
        distance of the last step's returned state from that step's
        fixed point, in each group's own norm (each field over its own
        magnitude, so a relative figure).  NaN when not usable.
    steps : int
        The coupling solves the run made, as the caller stated them.
    accumulated : float
        ``steps * per_step``.  The diagnostics describe one step.  A
        fixed-point iteration stopped early from the previous step's
        state errs the same way at every step, and a step map that does
        not amplify carries each of those errors to the end of the run,
        so their sum is what the final state may hold.  NaN when not
        usable.
    groups : tuple of str
        The coupling groups read.
    reason : str
        Why the bound is not usable, group by group; empty when it is.
    """

    usable: bool
    per_step: float
    steps: int
    accumulated: float
    groups: tuple[str, ...]
    reason: str = ""


@stability(StabilityLevel.EXPERIMENTAL)
def coupling_iteration_bound(graph: Any, *, steps: int) -> IterationErrorBound:
    """The iteration error of a run, from the graph's coupling diagnostics.

    Reads ``graph.coupling_report()`` (``coupling_diagnostics()``, one
    row per group) after the run.  The bound is usable where every group
    reports ``spectral_usable=True``: ``solver="ift"`` with
    ``diagnostics=True``, at most seven independent scalars crossing the
    group's edges, a residual above its float floor (see "Reading
    ``coupling_diagnostics()``" in the coupling algorithm guide).  A
    group that reports no usable bound -- ``solver="fori"``, diagnostics
    off, a single pass, a geometry-dependent mapping the diagnostics do
    not read, a wide interface -- makes the whole bound unusable, with
    the group's own flags as the reason: run the level again at a
    tighter tolerance instead (``tightened_error_at``).

    Parameters
    ----------
    graph : GraphManager
        The graph, after the run whose error is being measured.
    steps : int
        How many coupling solves that run made (its number of steps, for
        a group that solves every step).  The report is of the last step
        alone, and the bound returned is ``steps`` times it; see
        :attr:`IterationErrorBound.accumulated`.  It assumes the last
        step's bound is no smaller than the earlier steps': where a run
        ends in a state far smoother than it passed through, use the
        re-run route.

    Returns
    -------
    IterationErrorBound
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError(f"steps must be a positive integer, got {steps!r}")
    rows = list(graph.coupling_report())
    groups = tuple(str(row["group"]) for row in rows)
    if not rows:
        return IterationErrorBound(True, 0.0, steps, 0.0, (), "")
    worst, problems = 0.0, []
    for row in rows:
        bound = row.get("spectral_error_bound")
        usable = row.get("spectral_usable") is True or (
            row.get("spectral_usable") is not None and bool(row.get("spectral_usable")))
        if usable and bound is not None and math.isfinite(float(bound)):
            worst = max(worst, float(bound))
            continue
        flags = "; ".join(str(flag) for flag in (row.get("flags") or ()))
        problems.append(
            f"group {row['group']} (solver={row.get('solver')!r}) reports no usable "
            f"spectral_error_bound" + (f": {flags}" if flags else ""))
    if problems:
        return IterationErrorBound(False, math.nan, steps, math.nan, groups,
                                   "; ".join(problems))
    return IterationErrorBound(True, worst, steps, steps * worst, groups, "")


@dataclass(frozen=True)
class _Guarded:
    """The guard's finding at one level."""

    level: Any
    route: str
    iteration: float | None
    discretisation: float
    reason: str = ""

    def ok(self, factor: float) -> bool | None:
        if self.iteration is None:
            return None
        return self.iteration <= factor * self.discretisation


def _from_bound(level: Any, found: Any, scale: float) -> tuple[str, float | None, str]:
    """``(route, iteration error, reason)`` from what ``iteration_bound_at``
    returned; *scale* turns a relative bound into the measured quantity's
    units."""
    if found is None:
        return "", None, "iteration_bound_at returned None"
    if isinstance(found, IterationErrorBound):
        if not found.usable:
            return "", None, found.reason
        if not found.groups:
            return "no coupling group", 0.0, ""
        return "diagnostics", found.accumulated * scale, ""
    value = float(found)
    if not math.isfinite(value) or value < 0:
        return "", None, f"iteration_bound_at returned {found!r}"
    return "stated bound", value * scale, ""


def _guard(
    levels: Sequence[Any],
    discretisation: Sequence[float],
    measured: Sequence[float],
    relative_scale: Sequence[float],
    iteration_bound_at: Callable[[Any], Any] | None,
    tightened_at: Callable[[Any], float] | None,
    iterated: bool,
) -> list[_Guarded]:
    out = []
    for level, disc, value, scale in zip(levels, discretisation, measured, relative_scale):
        if not iterated:
            out.append(_Guarded(level, "stated not iterated", 0.0, disc))
            continue
        route, iteration, reason = "", None, "no guard was given"
        if iteration_bound_at is not None:
            route, iteration, reason = _from_bound(level, iteration_bound_at(level), scale)
        if iteration is None and tightened_at is not None:
            again = float(tightened_at(level))
            route, iteration, reason = "re-run", abs(value - again), ""
        out.append(_Guarded(level, route or "none", iteration, disc, reason))
    return out


def _guard_table(found: Sequence[_Guarded], factor: float) -> str:
    lines = [
        f"  iteration-error guard (iteration error <= {factor:g} x discretisation error):",
        f"  {'level':>10}  {'route':<20}  {'iteration':>12}  {'discretis.':>12}  verdict",
    ]
    for g in found:
        ok = g.ok(factor)
        verdict = "not checked" if ok is None else ("ok" if ok else "TOO LARGE")
        iteration = "-" if g.iteration is None else f"{g.iteration:.6g}"
        lines.append(f"  {g.level!s:>10}  {g.route:<20}  {iteration:>12}  "
                     f"{g.discretisation:12.6g}  {verdict}")
    return "\n".join(lines)


def _too_large(name: str, found: Sequence[_Guarded], factor: float, table: str,
               n: int) -> VerificationResult:
    bad = [str(g.level) for g in found if g.ok(factor) is False]
    return VerificationResult(
        name, "FAIL", n_examples=n,
        detail=(
            f"the coupling's iteration error is not far below the discretisation error "
            f"being measured at level(s) {', '.join(bad)}, so the ladder mixes the "
            f"error of the coupling solver with the error of the scheme and no order "
            f"can be read from it.  This is an inconclusive study, not a wrong model: "
            f"tighten the coupling tolerance (or raise max_iterations) until the guard "
            f"holds at every level.\n{_guard_table(found, factor)}\n{table}"
        ),
    )


def _unguarded(name: str, found: Sequence[_Guarded], factor: float, summary: str,
               table: str, n: int) -> VerificationResult:
    missing = [g for g in found if g.ok(factor) is None]
    reasons = sorted({g.reason for g in missing if g.reason})
    return VerificationResult(
        name, "SKIP", n_examples=n,
        detail=(
            f"{summary}, but the coupling's iteration error could not be checked at "
            f"level(s) {', '.join(str(g.level) for g in missing)}"
            + (f" ({'; '.join(reasons)})" if reasons else "")
            + ", so the ladder may be reading the coupling solver's error as the "
              "scheme's and the result is not reported as verified.  Give the guard a "
              "route: iteration_bound_at= (coupling_iteration_bound of the graph each "
              "level ran, where its groups report a usable bound), the same quantity "
              "re-run at a tighter coupling tolerance (tightened_error_at= / "
              "tightened_solution_at=), or iterated=False for a graph with no coupling "
              f"group.\n{_guard_table(found, factor)}\n{table}"
        ),
    )


def _check_factor(iteration_factor: float) -> None:
    if not (isinstance(iteration_factor, (int, float)) and 0 < iteration_factor <= 1):
        raise ValueError(
            f"iteration_factor is the fraction of the discretisation error the "
            f"iteration error must stay below, in (0, 1]; got {iteration_factor!r}")


@stability(StabilityLevel.EXPERIMENTAL)
def verify_graph_order(
    *,
    axis: RefinementAxis,
    error_at: Callable[[Any], float],
    levels: Sequence[Any],
    expected: float | None = None,
    h_of: Callable[[Any], float] | None = None,
    iteration_bound_at: Callable[[Any], Any] | None = None,
    tightened_error_at: Callable[[Any], float] | None = None,
    iteration_factor: float = DEFAULT_ITERATION_ERROR_FACTOR,
    iterated: bool = True,
    shortfall: float = DEFAULT_ORDER_SHORTFALL,
    excess: float = DEFAULT_ORDER_EXCESS,
) -> VerificationResult:
    """Measure a coupled graph's order of convergence and judge it.

    The graph-level counterpart of
    :func:`~maddening.testing.mms.verify_node_order`, over the same
    :func:`~maddening.testing.mms.measure_order` and
    :func:`~maddening.testing.mms.check_order`.  **Experimental.**

    Parameters
    ----------
    axis : RefinementAxis
        Which axis ``error_at`` refines.  One at a time: refining the
        grid and the timestep together measures the smaller of the two
        orders.
    error_at : callable
        ``level -> error``: builds the graph at that level, runs it and
        returns one error against the known solution, a *relative* norm
        taken the same way at every level.  Build a fresh graph per
        call: a run leaves a graph at its final state.
    levels : sequence
        Refinement levels, coarsest first.
    expected : float, optional
        The order expected of the coupled scheme.  A graph declares
        none, so without it the result is a ``SKIP`` and nothing is run.
    h_of : callable, optional
        ``level -> h``; ``1 / level`` when omitted.
    iteration_bound_at : callable, optional
        ``level -> IterationErrorBound`` (or a number, or ``None``): the
        iteration error of the run ``error_at`` made at that level,
        normally ``coupling_iteration_bound(graph, steps=...)`` of the
        same graph.  A bound that is not usable falls through to
        ``tightened_error_at``.  The bound is relative, in the groups'
        own norm, and is compared with the relative error ``error_at``
        returns.
    tightened_error_at : callable, optional
        ``level -> error`` of the same level with the coupling tolerance
        tightened (two orders of magnitude, say).  The iteration error
        is then the difference of the two errors: a direct measurement,
        in the units of the study, that assumes nothing about the graph.
    iteration_factor : float
        The iteration error must be at most this fraction of the level's
        error.  See :data:`DEFAULT_ITERATION_ERROR_FACTOR`.
    iterated : bool
        ``False`` states that nothing in the graph iterates (it has no
        coupling group), so there is no iteration error and no guard is
        applied.  The lag of a staggered exchange is then part of the
        time discretisation the study measures.
    shortfall, excess : float
        The order band; see
        :data:`~maddening.testing.mms.DEFAULT_ORDER_SHORTFALL`.

    Returns
    -------
    VerificationResult
        ``SKIP`` when no ``expected`` order was given.  ``FAIL`` when
        the guard finds the iteration error too large at a level (an
        inconclusive study, said as such), or when the observed order is
        outside the band or the ladder does not converge.  ``SKIP`` when
        the order is inside the band but the guard could not be made at
        some level.  ``PASS`` otherwise; the detail carries the
        refinement table and the guard's table.
    """
    name = f"coupled_{axis.order_attribute}_order"
    _check_factor(iteration_factor)
    if expected is None:
        return VerificationResult(name, "SKIP", detail=(
            f"no expected {axis.order_attribute} order was given, and a graph declares "
            f"none: there is nothing to measure against.  Pass expected= (the order of "
            f"the coupled scheme: at most 1 in time for a partitioned exchange, and at "
            f"most p + 1 in space through a mapping that reproduces degree p)."))
    measurement = measure_order(error_at, levels, axis=axis, h_of=h_of)
    errors = measurement.errors
    found = _guard(measurement.levels, errors, errors, [1.0] * len(errors),
                   iteration_bound_at, tightened_error_at, iterated)
    result = check_order(measurement, float(expected), name=name,
                         shortfall=shortfall, excess=excess)
    table = measurement.table()
    n = len(errors)
    if any(g.ok(iteration_factor) is False for g in found):
        return _too_large(name, found, iteration_factor, table, n)
    if not result.passed:
        return VerificationResult(name, result.status, n_examples=n, detail=(
            f"{result.detail}\n{_guard_table(found, iteration_factor)}"))
    if any(g.ok(iteration_factor) is None for g in found):
        return _unguarded(
            name, found, iteration_factor,
            f"observed {axis.order_attribute} order {measurement.observed:.3f} is inside "
            f"the band around the expected {float(expected):g}", table, n)
    return VerificationResult(name, "PASS", n_examples=n, detail=(
        f"{result.detail}\n{_guard_table(found, iteration_factor)}"))


@stability(StabilityLevel.EXPERIMENTAL)
def assert_graph_order_verified(**kwargs: Any) -> None:
    """:func:`verify_graph_order` that raises rather than returning.

    Raises
    ------
    AssertionError
        The order is outside the band, the ladder did not converge, or
        the iteration error is too large for the ladder to mean
        anything.  The message carries both tables.
    UndeclaredOrderError
        No ``expected`` order was given.
    InconclusiveStudyError
        The order is inside the band but the iteration error could not
        be checked, so the study is not reported as verified.
    """
    result = verify_graph_order(**kwargs)
    if result.skipped:
        if kwargs.get("expected") is None:
            raise UndeclaredOrderError(result.detail)
        raise InconclusiveStudyError(result.detail)
    if not result.passed:
        raise AssertionError(f"the coupled graph failed the {result.name} check:\n"
                             f"{result.detail}")


@stability(StabilityLevel.EXPERIMENTAL)
def verify_graph_gci(
    *,
    axis: RefinementAxis,
    solution_at: Callable[[Any], float],
    levels: Sequence[Any],
    expected: float | None = None,
    max_gci: float | None = None,
    h_of: Callable[[Any], float] | None = None,
    iteration_bound_at: Callable[[Any], Any] | None = None,
    tightened_solution_at: Callable[[Any], float] | None = None,
    iteration_factor: float = DEFAULT_ITERATION_ERROR_FACTOR,
    iterated: bool = True,
    safety_factor: float | None = None,
    stagnation_rtol: float = DEFAULT_STAGNATION_RTOL,
    asymptotic_tolerance: float = DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE,
    shortfall: float = DEFAULT_ORDER_SHORTFALL,
    excess: float = DEFAULT_ORDER_EXCESS,
) -> VerificationResult:
    """Run a grid convergence study on a coupled graph and judge it.

    The graph-level counterpart of
    :func:`~maddening.testing.mms.verify_node_gci`, over the same
    :func:`~maddening.testing.mms.measure_gci` and
    :func:`~maddening.testing.mms.check_gci`: the levels are compared
    with each other, so no known solution is needed, and an error that
    is the same at every level (the time error of a spatial ladder run
    at one timestep) cancels.  **Experimental.**

    Parameters
    ----------
    axis : RefinementAxis
    solution_at : callable
        ``level -> phi``: builds the graph at that level, runs it and
        returns one scalar functional of the solution, computed the same
        way at every level.
    levels : sequence
        At least three, coarsest first.
    expected : float, optional
        The order expected of the coupled scheme; also the formal order
        of the asymptotic-range check.
    max_gci : float, optional
        The widest acceptable error band on the finest solution, as a
        fraction of it.
    h_of : callable, optional
        ``level -> h``; ``1 / level`` when omitted.
    iteration_bound_at : callable, optional
        As for :func:`verify_graph_order`.  The relative bound is taken
        as a fraction of the functional's own value.
    tightened_solution_at : callable, optional
        ``level -> phi`` of the same level at a tighter coupling
        tolerance; the iteration error is the difference of the two.
    iteration_factor : float
        The iteration error at a level must be at most this fraction of
        the smallest difference between that level's solution and a
        neighbouring level's: the differences are what the study reads.
    iterated : bool
        As for :func:`verify_graph_order`.
    safety_factor, stagnation_rtol, asymptotic_tolerance
        As for :func:`~maddening.testing.mms.richardson_study`.
    shortfall, excess : float
        The order band.

    Returns
    -------
    VerificationResult
        As :func:`~maddening.testing.mms.verify_node_gci` judges a
        study -- ``FAIL`` for a ladder that did not converge or is
        outside the asymptotic range, ``SKIP`` for one that converged
        with neither ``expected`` nor ``max_gci`` to be judged by -- and,
        as :func:`verify_graph_order`, ``FAIL`` when the guard finds the
        iteration error too large at a level and ``SKIP`` when it could
        not be made.
    """
    name = f"coupled_{axis.order_attribute}_grid_convergence"
    _check_factor(iteration_factor)
    levels = tuple(levels)
    study = measure_gci(
        solution_at, levels, axis=axis, h_of=h_of,
        formal_order=None if expected is None else float(expected),
        safety_factor=safety_factor, stagnation_rtol=stagnation_rtol,
        asymptotic_tolerance=asymptotic_tolerance,
    )
    values = tuple(float(v) for v in study.values)
    n = len(values)
    differences = []
    for i in range(n):
        near = [abs(values[i] - values[j]) for j in (i - 1, i + 1) if 0 <= j < n]
        differences.append(min(near))
    found = _guard(levels, differences, values, [abs(v) for v in values],
                   iteration_bound_at, tightened_solution_at, iterated)
    table = study.table()
    if any(g.ok(iteration_factor) is False for g in found):
        return _too_large(name, found, iteration_factor, table, n)
    if expected is None and max_gci is None:
        result = check_gci(study, name=name)
        if result.failed:
            return VerificationResult(name, result.status, n_examples=n, detail=(
                f"{result.detail}\n{_guard_table(found, iteration_factor)}"))
        return VerificationResult(name, "SKIP", n_examples=n, detail=(
            f"the ladder ran and converged (the table below is the measurement), but "
            f"no expected {axis.order_attribute} order and no max_gci was given, and a "
            f"graph declares no order: a convergent ladder on its own is not a "
            f"verified model.  Pass expected= or max_gci=.\n{table}"))
    result = check_gci(study, expected=expected, max_gci=max_gci, name=name,
                       shortfall=shortfall, excess=excess)
    if not result.passed:
        return VerificationResult(name, result.status, n_examples=n, detail=(
            f"{result.detail}\n{_guard_table(found, iteration_factor)}"))
    if any(g.ok(iteration_factor) is None for g in found):
        return _unguarded(
            name, found, iteration_factor,
            f"the study converged at observed {axis.order_attribute} order "
            f"{study.order:.3f}", table, n)
    return VerificationResult(name, "PASS", n_examples=n, detail=(
        f"{result.detail}\n{_guard_table(found, iteration_factor)}"))


@stability(StabilityLevel.EXPERIMENTAL)
def assert_graph_gci_verified(**kwargs: Any) -> None:
    """:func:`verify_graph_gci` that raises rather than returning.

    Raises
    ------
    AssertionError
        The ladder did not converge, the order or the band missed, or
        the iteration error is too large for the ladder to mean
        anything.
    InconclusiveStudyError
        The ladder converged but there was nothing to judge it against,
        or the iteration error could not be checked.
    """
    result = verify_graph_gci(**kwargs)
    if result.skipped:
        raise InconclusiveStudyError(result.detail)
    if not result.passed:
        raise AssertionError(f"the coupled graph failed the {result.name} check:\n"
                             f"{result.detail}")
