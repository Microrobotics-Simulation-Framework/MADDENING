"""The coupled level: the order of a graph, and the iteration-error guard.

The first half drives :func:`verify_graph_order` and
:func:`verify_graph_gci` with closed-form ladders, to pin what each
outcome of the guard does to the result.  The second half is the worked
example of the verification guide, on real graphs: two rods on the same
interval that exchange heat along their whole length,

    dTa/dt = alpha Ta'' + k (Tb - Ta),    dTb/dt = alpha Tb'' + k (Ta - Tb),

insulated at both ends.  With ``Ta(x, 0) = 1 + cos(pi x)`` and
``Tb(x, 0) = 0`` the sum and the difference of the two decouple, and

    Ta + Tb = 1 + cos(pi x) exp(-alpha pi^2 t),
    Ta - Tb = exp(-2 k t) + cos(pi x) exp(-(alpha pi^2 + 2 k) t).

Each rod is a ``HeatNode``; the exchange is four additive edges into the
rods' ``heat_source`` (each rod's own temperature times ``-k``, the
other's times ``k``).  On matching grids the cross edges are plain; on
non-matching grids they carry a shipped mapping, which is where the edge
level and the coupled level meet: the degree the mapping reproduces
bounds the order the coupled model can show.

Precision: the studies run under ``jax_enable_x64`` (see
``test_mms_order.py`` for why a ladder needs it); a ``HeatNode``'s state
is float32 until it is set, so each graph is given float64 fields.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib  # noqa: E402
import functools  # noqa: E402
import math  # noqa: E402
import warnings  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maddening.core.coupling.mapping import (  # noqa: E402
    nearest_neighbor_mapping,
    projection_1d_mapping,
    rbf_mapping,
)
from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.nodes.heat import HeatNode  # noqa: E402
from maddening.testing.coupled import (  # noqa: E402
    DEFAULT_ITERATION_ERROR_FACTOR,
    IterationErrorBound,
    assert_graph_gci_verified,
    assert_graph_order_verified,
    coupling_iteration_bound,
    verify_graph_gci,
    verify_graph_order,
)
from maddening.testing.mms import (  # noqa: E402
    InconclusiveStudyError,
    RefinementAxis,
    UndeclaredOrderError,
)

SPACE, TIME = RefinementAxis.SPACE, RefinementAxis.TIME


@contextlib.contextmanager
def _float64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def float64():
    with _float64():
        yield


# ---------------------------------------------------------------------------
# The wrappers on closed-form ladders
# ---------------------------------------------------------------------------

LEVELS = (10, 20, 40)


def _second_order(n):
    return 3.0 / n ** 2


def _usable(per_step, steps=100):
    return IterationErrorBound(True, per_step, steps, steps * per_step, ("a+b",))


_UNUSABLE = IterationErrorBound(False, math.nan, 100, math.nan, ("a+b",),
                                "group a+b (solver='fori') reports no usable "
                                "spectral_error_bound")
_NO_GROUP = IterationErrorBound(True, 0.0, 100, 0.0, ())


def test_without_an_expected_order_the_result_is_a_skip_and_nothing_is_run():
    def never(level):
        raise AssertionError("the ladder ran")

    result = verify_graph_order(axis=SPACE, error_at=never, levels=LEVELS)
    assert result.skipped and "a graph declares none" in result.detail
    assert result.name == "coupled_spatial_order"
    with pytest.raises(UndeclaredOrderError):
        assert_graph_order_verified(axis=SPACE, error_at=never, levels=LEVELS)


def test_an_order_inside_the_band_with_no_guard_is_a_skip_that_says_so():
    result = verify_graph_order(axis=SPACE, error_at=_second_order, levels=LEVELS,
                                expected=2.0)
    assert result.skipped and not result.failed
    assert "could not be checked" in result.detail and "no guard was given" in result.detail
    assert "not reported as verified" in result.detail
    with pytest.raises(InconclusiveStudyError, match="could not be checked"):
        assert_graph_order_verified(axis=SPACE, error_at=_second_order, levels=LEVELS,
                                    expected=2.0)


def test_a_graph_stated_not_to_iterate_needs_no_guard():
    result = verify_graph_order(axis=SPACE, error_at=_second_order, levels=LEVELS,
                                expected=2.0, iterated=False)
    assert result.status == "PASS" and "stated not iterated" in result.detail
    assert_graph_order_verified(axis=SPACE, error_at=_second_order, levels=LEVELS,
                                expected=2.0, iterated=False)


@pytest.mark.parametrize("bound, route", [
    (_usable(1e-9), "diagnostics"), (_NO_GROUP, "no coupling group"), (1e-7, "stated bound"),
])
def test_a_bound_far_below_the_error_passes_by_its_route(bound, route):
    result = verify_graph_order(axis=SPACE, error_at=_second_order, levels=LEVELS,
                                expected=2.0, iteration_bound_at=lambda level: bound)
    assert result.status == "PASS", result.detail
    assert route in result.detail and "TOO LARGE" not in result.detail


def test_a_bound_that_is_not_far_below_the_error_fails_as_inconclusive():
    """1e-6 a step over 100 steps is 1e-4: below the coarse error (0.03)
    by far, and above a twentieth of the fine one (1.9e-3)."""
    result = verify_graph_order(axis=SPACE, error_at=_second_order, levels=LEVELS,
                                expected=2.0, iteration_bound_at=lambda level: _usable(1e-6))
    assert result.failed
    assert "not far below the discretisation error" in result.detail
    assert "level(s) 40" in result.detail and "inconclusive study" in result.detail
    with pytest.raises(AssertionError, match="not far below"):
        assert_graph_order_verified(
            axis=SPACE, error_at=_second_order, levels=LEVELS, expected=2.0,
            iteration_bound_at=lambda level: _usable(1e-6))
    # The factor is the caller's to state.
    loose = verify_graph_order(
        axis=SPACE, error_at=_second_order, levels=LEVELS, expected=2.0,
        iteration_bound_at=lambda level: _usable(1e-6), iteration_factor=0.1)
    assert loose.status == "PASS"


def test_an_unusable_bound_falls_through_to_the_rerun_and_to_a_skip_without_one():
    kw = dict(axis=SPACE, error_at=_second_order, levels=LEVELS, expected=2.0,
              iteration_bound_at=lambda level: _UNUSABLE)
    alone = verify_graph_order(**kw)
    assert alone.skipped and "solver='fori'" in alone.detail
    rerun = verify_graph_order(tightened_error_at=lambda n: _second_order(n) * 1.001, **kw)
    assert rerun.status == "PASS" and "re-run" in rerun.detail
    polluted = verify_graph_order(
        tightened_error_at=lambda n: _second_order(n) - 2e-4, **kw)
    assert polluted.failed and "level(s) 40" in polluted.detail


def test_the_rerun_route_is_used_only_where_the_bound_is_missing():
    calls = []

    def tightened(level):
        calls.append(level)
        return _second_order(level)

    result = verify_graph_order(
        axis=SPACE, error_at=_second_order, levels=LEVELS, expected=2.0,
        iteration_bound_at=lambda level: _usable(1e-9) if level != 20 else None,
        tightened_error_at=tightened)
    assert result.status == "PASS" and calls == [20]


def test_a_wrong_order_fails_whatever_the_guard_could_do():
    for guard in ({}, {"iterated": False}):
        result = verify_graph_order(axis=SPACE, error_at=lambda n: 3.0 / n, levels=LEVELS,
                                    expected=2.0, **guard)
        assert result.failed and "below the declared 2" in result.detail
    with pytest.raises(ValueError, match="iteration_factor"):
        verify_graph_order(axis=SPACE, error_at=_second_order, levels=LEVELS, expected=2.0,
                           iteration_factor=0.0)


def _first_order_solution(n):
    return 2.0 + 0.5 / n


def test_a_grid_convergence_study_is_guarded_on_the_differences_it_reads():
    kw = dict(axis=TIME, solution_at=_first_order_solution, levels=LEVELS, expected=1.0)
    assert verify_graph_gci(iterated=False, **kw).status == "PASS"
    assert verify_graph_gci(**kw).skipped
    with pytest.raises(InconclusiveStudyError, match="could not be checked"):
        assert_graph_gci_verified(**kw)
    # The finest level's solution differs from its neighbour's by 0.0125:
    # an iteration error of 1e-3 is not a twentieth of that, 1e-4 is.
    near = verify_graph_gci(tightened_solution_at=lambda n: _first_order_solution(n) + 1e-3,
                            **kw)
    assert near.failed and "not far below" in near.detail
    far = verify_graph_gci(tightened_solution_at=lambda n: _first_order_solution(n) + 1e-4,
                           **kw)
    assert far.status == "PASS" and "re-run" in far.detail
    assert far.name == "coupled_temporal_grid_convergence"
    # A relative bound is a fraction of the functional's own value (2).
    assert verify_graph_gci(iteration_bound_at=lambda n: _usable(1e-7), **kw).passed
    assert verify_graph_gci(iteration_bound_at=lambda n: _usable(1e-5), **kw).failed
    with pytest.raises(AssertionError, match="not far below"):
        assert_graph_gci_verified(iteration_bound_at=lambda n: _usable(1e-5), **kw)


def test_a_convergent_ladder_with_nothing_to_judge_it_by_is_a_skip_and_a_stalled_one_fails():
    kw = dict(axis=TIME, levels=LEVELS, iterated=False)
    converged = verify_graph_gci(solution_at=_first_order_solution, **kw)
    assert converged.skipped and "a graph declares no order" in converged.detail
    banded = verify_graph_gci(solution_at=_first_order_solution, max_gci=0.05, **kw)
    assert banded.status == "PASS"
    stalled = verify_graph_gci(solution_at=lambda n: 2.0, expected=1.0, **kw)
    assert stalled.failed
    # Third order where first is expected: the study is outside the
    # asymptotic range of the order it was told, and says that.
    wrong = verify_graph_gci(solution_at=lambda n: 2.0 + 0.5 / n ** 3, expected=1.0, **kw)
    assert wrong.failed and "not in the asymptotic range" in wrong.detail


class _Reported:
    """A graph as :func:`coupling_iteration_bound` reads it."""

    def __init__(self, rows):
        self.rows = rows

    def coupling_report(self):
        return self.rows


def test_the_bound_is_the_last_steps_times_the_steps_and_the_worst_group_decides():
    assert coupling_iteration_bound(_Reported([]), steps=7) == IterationErrorBound(
        True, 0.0, 7, 0.0, ())
    rows = [
        {"group": "a+b", "solver": "ift", "spectral_error_bound": 2e-9,
         "spectral_usable": True, "flags": ()},
        {"group": "c+d", "solver": "ift", "spectral_error_bound": 5e-9,
         "spectral_usable": np.bool_(True), "flags": ()},
    ]
    found = coupling_iteration_bound(_Reported(rows), steps=40)
    assert found.usable and found.per_step == 5e-9 and found.accumulated == 40 * 5e-9
    assert found.groups == ("a+b", "c+d") and found.reason == ""
    rows[0].update(spectral_usable=False, flags=("precision_limited",))
    rows.append({"group": "e+f", "solver": "fori", "spectral_error_bound": None,
                 "spectral_usable": None, "flags": ("no report: solver='fori' records "
                                                    "diagnostics only with diagnostics=True",)})
    found = coupling_iteration_bound(_Reported(rows), steps=40)
    assert not found.usable and math.isnan(found.accumulated)
    assert "group a+b" in found.reason and "precision_limited" in found.reason
    assert "group e+f (solver='fori')" in found.reason and "group c+d" not in found.reason
    rows[1]["spectral_error_bound"] = math.inf
    assert "group c+d" in coupling_iteration_bound(_Reported(rows), steps=1).reason
    for steps in (0, 2.0, True):
        with pytest.raises(ValueError, match="steps must be a positive integer"):
            coupling_iteration_bound(_Reported([]), steps=steps)


def test_the_default_factor_keeps_the_order_inside_the_band():
    f = DEFAULT_ITERATION_ERROR_FACTOR
    assert math.log((1 + f) / (1 - f)) / math.log(2.0) < 0.25


# ---------------------------------------------------------------------------
# The worked example: two rods exchanging heat along their length
# ---------------------------------------------------------------------------

ALPHA, EXCHANGE, T_END = 0.1, 1.0, 0.5
FOURIER = 0.2  # units: dt * alpha / dx**2 on the finer rod, below its limit of 1/2


def _centres(n):
    return (np.arange(n) + 0.5) / n


def _boundaries(n):
    return np.linspace(0.0, 1.0, n + 1)


MAPPINGS = {
    # (reproduces polynomials of degree, builder from one rod's cells to the other's)
    "projection_1d": lambda n_from, n_to: projection_1d_mapping(
        _boundaries(n_from), _boundaries(n_to)),
    "rbf": lambda n_from, n_to: rbf_mapping(
        _centres(n_from), _centres(n_to), kernel="thin_plate_spline"),
    "nearest_neighbor": lambda n_from, n_to: nearest_neighbor_mapping(
        _centres(n_from), _centres(n_to)),
}


def _times(factor):
    def transform(value):
        return factor * value
    return transform


def run(n_a, n_b, steps, *, mapping=None, group=None):
    """The two rods after ``steps`` steps to ``T_END``: a fresh graph."""
    dt = T_END / steps
    gm = GraphManager()
    gm.add_node(HeatNode("a", dt, n_cells=n_a, thermal_diffusivity=ALPHA))
    gm.add_node(HeatNode("b", dt, n_cells=n_b, thermal_diffusivity=ALPHA))
    for rod in ("a", "b"):
        gm.add_edge(rod, rod, "temperature", "heat_source", transform=_times(-EXCHANGE),
                    additive=True)
    for source, target, n_from, n_to in (("a", "b", n_a, n_b), ("b", "a", n_b, n_a)):
        gm.add_edge(source, target, "temperature", "heat_source",
                    transform=_times(EXCHANGE), additive=True,
                    mapping=None if mapping is None else MAPPINGS[mapping](n_from, n_to))
    if group is not None:
        gm.add_coupling_group(["a", "b"], **group)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    gm.set_node_state("a", {"temperature": jnp.asarray(1.0 + np.cos(np.pi * _centres(n_a)))})
    gm.set_node_state("b", {"temperature": jnp.zeros(n_b, jnp.float64)})
    gm.run_scan(steps)
    return gm


def exact(x, rate=ALPHA * np.pi ** 2):
    """``(Ta, Tb)`` at ``T_END``; *rate* is ``alpha`` times the eigenvalue
    of ``cos(pi x)`` (the continuous operator's, unless given)."""
    total = 1.0 + np.cos(np.pi * x) * math.exp(-rate * T_END)
    difference = math.exp(-2 * EXCHANGE * T_END) + np.cos(np.pi * x) * math.exp(
        -(rate + 2 * EXCHANGE) * T_END)
    return 0.5 * (total + difference), 0.5 * (total - difference)


def relative_error(gm, n_a, n_b, rate=ALPHA * np.pi ** 2):
    t_a = np.asarray(gm.get_node_state("a")["temperature"])
    t_b = np.asarray(gm.get_node_state("b")["temperature"])
    assert t_a.dtype == np.float64 and t_b.dtype == np.float64
    e_a, e_b = exact(_centres(n_a), rate)[0], exact(_centres(n_b), rate)[1]
    return math.sqrt((np.mean((t_a - e_a) ** 2) + np.mean((t_b - e_b) ** 2))
                     / (np.mean(e_a ** 2) + np.mean(e_b ** 2)))


def _space_steps(n_fine):
    return math.ceil(T_END * ALPHA * n_fine ** 2 / FOURIER)


def space_error(n, *, ratio=1.0, mapping=None, group=None):
    """The error on grids of ``n`` and ``ratio * n`` cells with the
    timestep refined as ``dx**2``: the scheme is first order in time, so
    its time error falls at order 2 in ``dx`` and cannot limit a ladder
    that expects 2 (or 1)."""
    n_b = round(n * ratio)
    return relative_error(run(n, n_b, _space_steps(max(n, n_b)), mapping=mapping, group=group),
                          n, n_b)


def _group(tolerance, **more):
    return dict(max_iterations=100, tolerance=tolerance, **more)


N_TIME = 16
#: ``alpha`` times the eigenvalue of ``cos(pi x)`` under the rods' own
#: second-order stencil on ``N_TIME`` cells: the solution of the
#: equations *as discretised in space*, which leaves a time ladder on
#: that grid its time error alone.
RATE_ON_GRID = ALPHA * 4 * N_TIME ** 2 * math.sin(math.pi / (2 * N_TIME)) ** 2
TIME_LEVELS = (50, 100, 200, 400)


def time_error(steps, tolerance, **more):
    gm = run(N_TIME, N_TIME, steps, group=_group(tolerance, **more))
    return relative_error(gm, N_TIME, N_TIME, RATE_ON_GRID), gm


def test_matching_grids_on_plain_edges_converge_at_second_order_in_space(float64):
    result = verify_graph_order(
        axis=SPACE, levels=(8, 16, 32), expected=2.0,
        error_at=lambda n: space_error(n, group=_group(1e-10)),
        tightened_error_at=lambda n: space_error(n, group=_group(1e-12)))
    assert result.status == "PASS", result.detail
    assert "re-run" in result.detail


@pytest.mark.parametrize("mapping, expected", [("projection_1d", 2.0), ("nearest_neighbor", 1.0)])
def test_non_matching_grids_converge_at_the_order_the_mapping_allows(float64, mapping,
                                                                    expected):
    """Three cells of one rod to two of the other.  The cell-average
    projection keeps the scheme's second order; nearest neighbour, which
    reproduces constants and nothing more, brings the coupled model down
    to first -- and is caught claiming second."""
    kw = dict(axis=SPACE, levels=(8, 16, 32), iterated=False,
              error_at=lambda n: space_error(n, ratio=1.5, mapping=mapping))
    result = verify_graph_order(expected=expected, **kw)
    assert result.status == "PASS", result.detail
    if mapping == "nearest_neighbor":
        claimed = verify_graph_order(expected=2.0, **kw)
        assert claimed.failed and "below the declared 2" in claimed.detail


def test_the_diagnostics_route_reads_a_real_group_and_refuses_one_with_no_bound(float64):
    steps = 50
    gm = run(N_TIME, N_TIME, steps, group=_group(1e-8, solver="ift", diagnostics=True))
    found = coupling_iteration_bound(gm, steps=steps)
    assert found.usable and found.groups == ("a+b",), found
    assert 0.0 < found.per_step < 1e-6 and found.accumulated == steps * found.per_step
    assert found.accumulated < DEFAULT_ITERATION_ERROR_FACTOR * relative_error(
        gm, N_TIME, N_TIME, RATE_ON_GRID)
    plain = run(N_TIME, N_TIME, steps, group=_group(1e-8, solver="fori"))
    refused = coupling_iteration_bound(plain, steps=steps)
    assert not refused.usable and "solver='fori'" in refused.reason
    staggered = run(N_TIME, N_TIME, steps)
    assert coupling_iteration_bound(staggered, steps=steps) == IterationErrorBound(
        True, 0.0, steps, 0.0, ())


# Per push: tests/verification/test_coupled_order.py::test_non_matching_grids_converge_at_the_order_the_mapping_allows
@pytest.mark.slow
@pytest.mark.parametrize("mapping, ratio, expected", [
    ("rbf", 1.5, 2.0), ("projection_1d", 2.0, 2.0), ("rbf", 2.0, 2.0),
    ("nearest_neighbor", 2.0, 1.0),
])
def test_non_matching_grids_in_a_converged_group_keep_the_mappings_order(
        float64, mapping, ratio, expected):
    result = verify_graph_order(
        axis=SPACE, levels=(8, 16, 32, 64), expected=expected,
        error_at=lambda n: space_error(n, ratio=ratio, mapping=mapping, group=_group(1e-10)),
        tightened_error_at=lambda n: space_error(n, ratio=ratio, mapping=mapping,
                                                 group=_group(1e-12)))
    assert result.status == "PASS", result.detail


# Per push: tests/verification/test_coupled_order.py::test_the_diagnostics_route_reads_a_real_group_and_refuses_one_with_no_bound and tests/verification/test_coupled_order.py::test_a_bound_far_below_the_error_passes_by_its_route
@pytest.mark.slow
def test_the_exchange_is_first_order_in_time_and_the_diagnostics_guard_it(float64):
    """A partitioned exchange hands each rod its input once per step, so
    the coupled scheme is first order in time whatever converges inside
    the step (MADD-ANO-014)."""
    @functools.lru_cache(maxsize=None)
    def level(steps):
        return time_error(steps, 1e-8, solver="ift", diagnostics=True)

    result = verify_graph_order(
        axis=TIME, levels=TIME_LEVELS, expected=1.0,
        error_at=lambda steps: level(steps)[0],
        iteration_bound_at=lambda steps: coupling_iteration_bound(level(steps)[1],
                                                                  steps=steps))
    assert result.status == "PASS", result.detail
    assert "diagnostics" in result.detail and "re-run" not in result.detail


# Per push: tests/verification/test_coupled_order.py::test_a_bound_that_is_not_far_below_the_error_fails_as_inconclusive and tests/verification/test_coupled_order.py::test_an_unusable_bound_falls_through_to_the_rerun_and_to_a_skip_without_one
@pytest.mark.slow
def test_a_loose_tolerance_is_caught_by_both_routes_before_it_is_read_as_an_order(float64):
    """At ``tolerance=1e-4`` the group stops after one pass on the fine
    levels and two on the coarse ones, so the ladder mixes two schemes.
    The last step's bound is a few percent of the error at every level --
    it is the bound times the number of steps that is not small."""
    @functools.lru_cache(maxsize=None)
    def level(steps, tolerance=1e-4):
        return time_error(steps, tolerance, solver="ift", diagnostics=True)

    errors = [level(steps)[0] for steps in TIME_LEVELS]
    last_step = [coupling_iteration_bound(level(steps)[1], steps=steps).per_step
                 for steps in TIME_LEVELS]
    assert all(b < 0.1 * e for b, e in zip(last_step, errors)), (last_step, errors)
    by_diagnostics = verify_graph_order(
        axis=TIME, levels=TIME_LEVELS, expected=1.0,
        error_at=lambda steps: level(steps)[0],
        iteration_bound_at=lambda steps: coupling_iteration_bound(level(steps)[1],
                                                                  steps=steps))
    assert by_diagnostics.failed and "not far below" in by_diagnostics.detail
    by_rerun = verify_graph_order(
        axis=TIME, levels=TIME_LEVELS, expected=1.0,
        error_at=lambda steps: level(steps)[0],
        tightened_error_at=lambda steps: level(steps, 1e-8)[0])
    assert by_rerun.failed and "not far below" in by_rerun.detail
    assert "level(s)" in by_rerun.detail and "400" in by_rerun.detail.split("level(s)")[1][:40]


# Per push: tests/verification/test_coupled_order.py::test_a_grid_convergence_study_is_guarded_on_the_differences_it_reads
@pytest.mark.slow
def test_the_spatial_order_alone_from_levels_compared_with_each_other(float64):
    """One timestep at every level: the time error is the same at each
    and cancels in the differences, so the study reads the spatial order
    with no known solution and no refinement of the timestep."""
    steps = 320  # units: steps to T_END; a Fourier number of 0.16 on 32 cells

    def amplitude(n, tolerance=1e-10):
        gm = run(n, n, steps, group=_group(tolerance))
        t_a = np.asarray(gm.get_node_state("a")["temperature"])
        return float(2.0 * np.mean(t_a * np.cos(np.pi * _centres(n))))

    result = verify_graph_gci(
        axis=SPACE, levels=(8, 16, 32), expected=2.0, solution_at=amplitude,
        tightened_solution_at=lambda n: amplitude(n, 1e-12))
    assert result.status == "PASS", result.detail
