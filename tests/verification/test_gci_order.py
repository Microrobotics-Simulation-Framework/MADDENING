"""Grid convergence: observed order and error band without a reference.

``test_mms_order.py`` measures the order of convergence by manufacturing
an exact solution and feeding the node the source term that makes it
exact.  That needs somewhere to put the source, which is a real
restriction: :class:`~maddening.nodes.lbm_pipe.LBMPipeNode` declares no
boundary inputs at all, a generic source term is not even well defined
for a lattice Boltzmann collision operator, and a node a *user* writes
will usually have no forcing input either.  Extending
``SimulationNode`` with a manufactured-source hook was considered and
rejected; ``TODO.md``, "DECIDED AGAINST: a manufactured-source
convention on ``SimulationNode``", records why.

These tests cover the fallback that made rejecting it acceptable.  A
Richardson / Grid Convergence Index study compares three successively
refined solutions to **each other**: no exact field, no source term,
nothing required of the node but the ability to refine.  It yields the
observed order of convergence, the extrapolated limit, and an error
band on the finest solution [@Richardson1911; @Roache1994; @Celik2008].

What it cannot do, and this file is careful not to imply otherwise: GCI
never sees the true answer, so it cannot catch a scheme that converges
cleanly to the *wrong* limit.  MMS can.  Where a node has a natural
forcing input, MMS is the stronger test and GCI is the uncertainty
statement on top of it.

The file is organised as: validation against problems whose exact
answer and exact order are known; mutation tests for every way a study
can fail to reach a verdict; and the demonstration on ``LBMPipeNode``,
the node that motivated the mode.

Precision: the studies run under ``jax_enable_x64``.  A convergence
ratio is a quotient of differences of nearly equal numbers, and float32
runs out of signal well before a refinement ladder does — the same
reason ``test_mms_order.py`` gives.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib  # noqa: E402
import functools  # noqa: E402
import math  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from maddening.core.compliance.metadata import (  # noqa: E402
    DiscretizationOrder,
    NodeMeta,
)
from maddening.core.compliance.validation import (  # noqa: E402
    _BENCHMARK_REGISTRY,
    BenchmarkType,
    verification_benchmark,
)
from maddening.core.node import SimulationNode  # noqa: E402
from maddening.nodes.lbm_pipe import LBMPipeNode  # noqa: E402
from maddening.testing.mms import (  # noqa: E402
    CAUTIOUS_SAFETY_FACTOR,
    DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE,
    DEFAULT_SAFETY_FACTOR,
    DEFAULT_STAGNATION_RTOL,
    MAX_APPARENT_ORDER,
    MIN_APPARENT_ORDER,
    RECOMMENDED_MIN_REFINEMENT_RATIO,
    ConvergenceRegime,
    InconclusiveStudyError,
    RefinementAxis,
    apparent_order,
    assert_node_gci_verified,
    check_gci,
    measure_gci,
    richardson_study,
    verify_node_gci,
)

SPACE = RefinementAxis.SPACE


@contextlib.contextmanager
def _float64():
    """Run the body in double precision, restoring the global setting."""
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


def _study(values, h=(0.1, 0.05, 0.025), **kwargs):
    """A study built straight from numbers, for the mutation tests."""
    return richardson_study(values, h, axis=SPACE, **kwargs)


# ---------------------------------------------------------------------------
# Validation: problems whose exact answer and exact order are known
# ---------------------------------------------------------------------------

#: ``int_0^1 exp(x) dx``.  Chosen because both quadratures below have a
#: textbook order on it and neither is exact on it, so a recovered
#: order is a real measurement rather than a lucky cancellation.
_EXACT_INTEGRAL = math.e - 1.0


def _trapezoid(n_intervals):
    """Composite trapezoid rule: second order, by Euler-Maclaurin."""
    h = 1.0 / n_intervals
    xs = np.arange(n_intervals + 1) * h
    return float(h * (np.exp(xs).sum() - 0.5 * (math.e ** 0 + math.e)))


def _left_riemann(n_intervals):
    """Left Riemann sum: first order."""
    h = 1.0 / n_intervals
    return float(h * np.exp(np.arange(n_intervals) * h).sum())


_QUADRATURES = {"trapezoid": (_trapezoid, 2.0), "left-riemann": (_left_riemann, 1.0)}


@pytest.mark.parametrize("rule", sorted(_QUADRATURES))
def test_a_known_order_is_recovered_from_the_solutions_alone(rule):
    """The whole claim of the mode, on a case where the answer is known.

    Neither ladder is told the exact integral or the exact order.  The
    observed order has to come out of the three sums and nothing else.
    """
    quadrature, formal = _QUADRATURES[rule]
    study = measure_gci(quadrature, (10, 20, 40), axis=SPACE)

    assert study.regime is ConvergenceRegime.MONOTONE
    assert study.order == pytest.approx(formal, abs=0.05)


@pytest.mark.parametrize("rule", sorted(_QUADRATURES))
def test_the_gci_band_actually_contains_the_true_error(rule):
    """An error band that does not contain the error is worse than none.

    The band must cover the true error of the finest solution, and must
    not cover it by so much that it says nothing: for a clean power law
    the Grid Convergence Index is the safety factor times the true
    relative error, so the ratio pins the arithmetic as well as the
    inequality.
    """
    quadrature, formal = _QUADRATURES[rule]
    study = measure_gci(
        quadrature, (10, 20, 40), axis=SPACE, formal_order=formal,
    )
    finest = study.values[-1]
    true_relative_error = abs((finest - _EXACT_INTEGRAL) / finest)

    assert study.gci_fine >= true_relative_error
    assert study.gci_fine / true_relative_error == pytest.approx(
        DEFAULT_SAFETY_FACTOR, rel=0.05,
    )
    low, high = study.band
    assert low <= _EXACT_INTEGRAL <= high


def test_the_richardson_limit_is_far_closer_than_the_finest_solution():
    """Extrapolation has to buy something, or it is decoration.

    Two orders beyond the finest grid is the point of the ``h -> 0``
    estimate; anything less and the band could be read off the raw
    difference instead.
    """
    study = measure_gci(_trapezoid, (10, 20, 40), axis=SPACE)
    finest_error = abs(study.values[-1] - _EXACT_INTEGRAL)
    limit_error = abs(study.extrapolated - _EXACT_INTEGRAL)

    assert limit_error < finest_error / 100.0


def test_a_non_constant_refinement_ratio_is_solved_and_not_assumed():
    """The failure mode this implementation exists to avoid.

    With ``h`` refined by 1.7 and then by 40/17, the textbook
    ``log(eps_32 / eps_21) / log(r)`` is simply wrong — it reads 0.98 on
    a second-order rule, which would look like a perfectly plausible
    first-order scheme and would be believed.  Solving the implicit
    equation with Celik et al.'s ``q(p)`` term recovers 2.

    A GCI that quietly assumed a constant ratio would report the wrong
    order here with no sign that anything was amiss, which is why this
    test compares against the shortcut rather than only against 2.
    """
    study = measure_gci(_trapezoid, (10, 17, 40), axis=SPACE)
    ratios = study.refinement_ratios

    assert not study.constant_ratio
    assert study.order_method == "fixed-point"
    assert study.order == pytest.approx(2.0, abs=0.05)

    eps_coarse, eps_fine = study.differences
    shortcut = math.log(abs(eps_coarse / eps_fine)) / math.log(ratios[-1])
    assert abs(shortcut - 2.0) > 0.9, (
        f"the constant-ratio shortcut was supposed to be badly wrong here "
        f"and read {shortcut:.4f}; if it has become right, this test no "
        f"longer proves the implicit solve earns its place"
    )


def test_four_levels_settle_the_asymptotic_range_with_nothing_declared():
    """The check that needs no declared order: do independent triples agree?

    Three levels and no declared order cannot decide the asymptotic
    range at all, and the harness says ``None`` rather than ``True``.
    A fourth level gives a second, independent estimate of the order,
    and agreement between them is evidence.
    """
    three = measure_gci(_trapezoid, (10, 20, 40), axis=SPACE)
    four = measure_gci(_trapezoid, (10, 20, 40, 80), axis=SPACE)

    assert three.in_asymptotic_range is None
    assert three.safety_factor == CAUTIOUS_SAFETY_FACTOR
    assert four.in_asymptotic_range is True
    assert four.safety_factor == DEFAULT_SAFETY_FACTOR
    assert len(four.triplet_orders) == 2
    assert max(four.triplet_orders) - min(four.triplet_orders) < 0.05


# ---------------------------------------------------------------------------
# Mutation tests: every way a study can fail to reach a verdict
# ---------------------------------------------------------------------------


class TestTheGciGateCanFail:
    """A gate is worth what it fails on, not what it passes.

    Each case here is a triple the gate must refuse, and must refuse
    *by name* — a study that cannot reach a verdict has to be
    distinguishable from one that reached a bad one, and neither may
    look like a pass.
    """

    def test_a_diverging_ladder_is_refused(self):
        """Refining changed the answer by more than the refinement before."""
        study = _study([1.0, 1.2, 1.7])
        assert study.regime is ConvergenceRegime.DIVERGENT
        assert study.ratio > 1.0
        result = check_gci(study)
        assert result.failed
        assert "monotone divergence" in result.detail

    def test_a_non_monotone_ladder_is_refused_and_named(self):
        """Sign-alternating differences must not average into an order."""
        study = _study([1.10, 0.95, 1.02])
        assert study.regime is ConvergenceRegime.OSCILLATORY
        assert -1.0 < study.ratio < 0.0
        assert math.isnan(study.order)
        assert math.isnan(study.gci_fine)
        result = check_gci(study)
        assert result.failed
        assert "oscillatory convergence" in result.detail

    def test_an_oscillating_diverging_ladder_is_refused(self):
        study = _study([1.0, 1.2, 0.7])
        assert study.regime is ConvergenceRegime.OSCILLATORY_DIVERGENT
        assert study.ratio <= -1.0
        assert check_gci(study).failed

    def test_identical_solutions_are_refused_rather_than_divided_by(self):
        """The zero-difference triple: every formula here divides by it."""
        study = _study([1.234, 1.234, 1.234])
        assert study.regime is ConvergenceRegime.STAGNANT
        assert math.isnan(study.order)
        result = check_gci(study, max_gci=0.5)
        assert result.failed
        assert "did not respond to refinement" in result.detail

    def test_solutions_identical_to_round_off_are_refused_too(self):
        """Exact equality is the easy case; the noise floor is the real one."""
        base = 4.0
        study = _study([base, base + 1e-15, base + 2e-15])
        assert study.regime is ConvergenceRegime.STAGNANT
        assert check_gci(study).failed

    def test_a_difference_just_above_the_floor_is_not_called_stagnant(self):
        """The floor must not swallow a genuine study.

        Pins the boundary from the other side, so a later widening of
        :data:`DEFAULT_STAGNATION_RTOL` cannot silently start rejecting
        real ladders.
        """
        base = 4.0
        step = 100.0 * DEFAULT_STAGNATION_RTOL * base
        study = _study([base + 5 * step, base + step, base])
        assert study.regime is ConvergenceRegime.MONOTONE

    def test_a_non_finite_solution_is_refused(self):
        study = _study([1.0, 1.2, float("nan")])
        assert study.regime is ConvergenceRegime.INVALID
        result = check_gci(study)
        assert result.failed
        assert "non-finite" in result.detail

    @pytest.mark.parametrize("measured", [1.0, 1.5, 1.74])
    def test_an_order_short_of_the_declared_one_fails(self, measured):
        hs = (0.1, 0.05, 0.025)
        study = richardson_study(
            [1.0 + h**measured for h in hs], hs, axis=SPACE,
            asymptotic_tolerance=10.0,   # isolate the order gate
        )
        result = check_gci(study, expected=2.0)
        assert result.failed
        assert "below the declared" in result.detail

    @pytest.mark.parametrize("measured", [3.5, 6.0])
    def test_an_order_far_above_the_declared_one_fails(self, measured):
        hs = (0.1, 0.05, 0.025)
        study = richardson_study(
            [1.0 + h**measured for h in hs], hs, axis=SPACE,
            asymptotic_tolerance=10.0,
        )
        result = check_gci(study, expected=2.0)
        assert result.failed
        assert "above the declared" in result.detail

    def test_a_band_wider_than_asked_for_fails(self):
        hs = (0.1, 0.05, 0.025)
        study = richardson_study(
            [1.0 + 20.0 * h**2 for h in hs], hs, axis=SPACE, formal_order=2.0,
        )
        assert study.gci_fine > 0.001
        result = check_gci(study, max_gci=0.001)
        assert result.failed
        assert "Grid Convergence Index" in result.detail

    def test_a_study_outside_the_asymptotic_range_fails(self):
        """A band quoted outside the asymptotic range is not a band.

        The four values are contrived so the two triples disagree
        violently about the order — 1 over the coarse triple, 4 over
        the fine one — while every difference keeps the same sign.
        Monotone, and still meaningless.
        """
        study = richardson_study(
            [1.49, 1.17, 1.01, 1.00], (0.8, 0.4, 0.2, 0.1), axis=SPACE,
        )
        assert study.regime is ConvergenceRegime.MONOTONE
        assert study.in_asymptotic_range is False
        result = check_gci(study, max_gci=1.0)
        assert result.failed
        assert "not in the asymptotic range" in result.detail

    def test_refinement_ratios_that_determine_no_order_are_refused(self):
        """Ratios far enough apart, and the equation has no root at all.

        ``r_32 = 4`` against ``r_21 = 1.05``: the residual is negative
        across the whole admissible interval, so there is no observed
        order to report and the harness says exactly that instead of
        returning the endpoint it got closest at.
        """
        study = richardson_study(
            [12.0001, 11.0, 10.0], (4.2, 1.05, 1.0), axis=SPACE,
        )
        assert study.regime is ConvergenceRegime.MONOTONE
        assert math.isnan(study.order)
        assert study.order_method == "no-solution"
        result = check_gci(study)
        assert result.failed
        assert "could not be determined" in result.detail

    def test_two_levels_are_refused(self):
        """Two solutions give a difference but not a rate."""
        with pytest.raises(ValueError, match="at least three"):
            measure_gci(_trapezoid, (10, 20), axis=SPACE)

    def test_levels_given_finest_first_are_refused(self):
        """Reversed levels would flip the sign of every difference."""
        with pytest.raises(ValueError, match="coarsest first"):
            measure_gci(_trapezoid, (40, 20, 10), axis=SPACE)

    def test_a_values_and_h_length_mismatch_is_refused(self):
        with pytest.raises(ValueError, match="for 2 refinement levels"):
            richardson_study([1.0, 2.0, 3.0], (0.1, 0.05), axis=SPACE)


# ---------------------------------------------------------------------------
# Nodes: refusing to refine, and having nothing to be judged against
# ---------------------------------------------------------------------------


class _IgnoresRefinementNode(SimulationNode):
    """A node whose answer does not depend on the resolution asked for.

    The silent failure the mode has to catch.  A driver that forgets to
    pass the level through, a node that caches a grid built at
    construction, a resolution argument that is accepted and dropped —
    all of them look like this from outside, and all of them would make
    a naive study report a perfect, meaningless convergence.
    """

    def initial_state(self):
        return {"x": jnp.zeros(())}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"] + 1.0}


class _UndeclaredNode(_IgnoresRefinementNode):
    """Converges properly, but declares no order of accuracy."""


class _DeclaredSecondOrderNode(_IgnoresRefinementNode):
    meta = NodeMeta(
        description="test double that claims second-order accuracy in space",
        discretization_order=DiscretizationOrder(
            spatial=2.0, notes="declared for the harness's benefit only",
        ),
    )


def _clean_second_order(level):
    return 1.0 + 0.5 * (1.0 / level) ** 2


def _ignores_refinement(level):
    return 1.234


class TestANodeThatDoesNotRefine:
    """A node that silently fails to refine must not produce a green result.

    This is the one case that has to hold whatever else is true of the
    node — in particular, whether or not it declares an order.  A node
    with nothing declared would otherwise skip, and a skip counts as
    passed.
    """

    def test_it_fails_rather_than_passing(self):
        node = _DeclaredSecondOrderNode("frozen", 0.01)
        result = verify_node_gci(
            node, axis=SPACE, solution_at=_ignores_refinement,
            levels=(10, 20, 40),
        )
        assert result.status == "FAIL"
        assert "did not respond to refinement" in result.detail

    def test_it_fails_even_with_no_declared_order_to_hide_behind(self):
        """The study runs before the declaration is looked at, for this.

        ``verify_node_order`` skips an undeclared node *before* running
        its ladder.  If this one did the same, a node that both declares
        nothing and ignores refinement would skip — and a skip passes.
        """
        node = _UndeclaredNode("frozen", 0.01)
        result = verify_node_gci(
            node, axis=SPACE, solution_at=_ignores_refinement,
            levels=(10, 20, 40),
        )
        assert result.status == "FAIL"
        assert not result.passed

    def test_the_assert_form_raises_an_assertion_not_a_skip_signal(self):
        node = _UndeclaredNode("frozen", 0.01)
        with pytest.raises(AssertionError, match="did not respond to refinement"):
            assert_node_gci_verified(
                node, axis=SPACE, solution_at=_ignores_refinement,
                levels=(10, 20, 40),
            )


class TestNothingToJudgeAgainst:
    """The SKIP contract, matching ``verify_node_order``.

    A converging ladder on a node that declares nothing, with no band
    asked of it, is a measurement and not a verdict.  It reports
    ``SKIP`` explicitly rather than ``PASS``.
    """

    def test_a_node_declaring_nothing_is_skipped_not_passed(self):
        node = _UndeclaredNode("undeclared", 0.01)
        result = verify_node_gci(
            node, axis=SPACE, solution_at=_clean_second_order,
            levels=(10, 20, 40),
        )
        assert result.status == "SKIP"
        assert "declares no spatial order" in result.detail
        assert "max_gci" in result.detail

    def test_the_skip_still_carries_the_measurement(self):
        """Skipping the verdict must not throw away the study."""
        node = _UndeclaredNode("undeclared", 0.01)
        result = verify_node_gci(
            node, axis=SPACE, solution_at=_clean_second_order,
            levels=(10, 20, 40),
        )
        assert "observed order p = 2.000" in result.detail

    def test_the_assert_form_raises_rather_than_passing_silently(self):
        node = _UndeclaredNode("undeclared", 0.01)
        with pytest.raises(InconclusiveStudyError, match="declares no spatial order"):
            assert_node_gci_verified(
                node, axis=SPACE, solution_at=_clean_second_order,
                levels=(10, 20, 40),
            )

    def test_a_requested_band_is_enough_to_judge_an_undeclared_node(self):
        """The criterion a user-written node can use on day one."""
        node = _UndeclaredNode("undeclared", 0.01)
        result = verify_node_gci(
            node, axis=SPACE, solution_at=_clean_second_order,
            levels=(10, 20, 40), max_gci=0.01,
        )
        assert result.status == "PASS"

    def test_the_declared_order_is_taken_from_the_node(self):
        """Reading the claim off the node is what makes a wrong claim visible."""
        node = _DeclaredSecondOrderNode("declared", 0.01)
        result = verify_node_gci(
            node, axis=SPACE, solution_at=_clean_second_order,
            levels=(10, 20, 40),
        )
        assert result.status == "PASS"
        assert "against the declared 2" in result.detail

    def test_a_node_whose_declaration_is_wrong_fails(self):
        node = _DeclaredSecondOrderNode("declared", 0.01)
        result = verify_node_gci(
            node, axis=SPACE, solution_at=lambda n: 1.0 + 0.5 / n,
            levels=(10, 20, 40),
        )
        assert result.status == "FAIL"

    def test_an_explicit_expectation_overrides_the_declaration(self):
        node = _DeclaredSecondOrderNode("declared", 0.01)
        result = verify_node_gci(
            node, axis=SPACE, solution_at=lambda n: 1.0 + 0.5 / n,
            levels=(10, 20, 40), expected=1.0,
        )
        assert result.status == "PASS"


# ---------------------------------------------------------------------------
# The observed-order solve
# ---------------------------------------------------------------------------


class TestApparentOrder:
    def test_a_constant_ratio_uses_the_closed_form(self):
        phi = [1.0 + 0.5 * h**2 for h in (0.025, 0.05, 0.1)]
        got = apparent_order(phi[1] - phi[0], phi[2] - phi[1], 2.0, 2.0)
        assert got.method == "closed-form"
        assert got.value == pytest.approx(2.0, abs=1e-9)

    def test_the_bisection_rescues_ratios_the_fixed_point_diverges_on(self):
        """Celik et al.'s iteration is not unconditionally convergent.

        Its map has derivative ``q'(p) / ln r_21``, which exceeds 1 in
        magnitude when the two refinement ratios are far apart: at
        ``r_21 = 1.4`` against ``r_32 = 2.7`` the iteration walks off to
        an overflow within twenty steps.  The bracketed fallback is
        what makes the non-constant-ratio support real rather than
        nominal, and this pins that it is exercised.
        """
        h = [0.1 / (2.7 * 1.4), 0.1 / 2.7, 0.1]
        phi = [1.0 + 0.7 * x**3 for x in h]
        got = apparent_order(phi[1] - phi[0], phi[2] - phi[1], 1.4, 2.7)
        assert got.method == "bracketed"
        assert got.value == pytest.approx(3.0, abs=1e-6)

    def test_a_large_order_does_not_overflow(self):
        """``r**p`` overflows a float64 long before ``p`` reaches the cap."""
        h = [0.1 / (3.0 * 1.5), 0.1 / 3.0, 0.1]
        phi = [1.0 + 0.3 * x**8 for x in h]
        got = apparent_order(phi[1] - phi[0], phi[2] - phi[1], 1.5, 3.0)
        assert math.isfinite(got.value)
        assert got.value == pytest.approx(8.0, rel=1e-4)

    def test_the_search_interval_is_the_one_documented(self):
        assert (MIN_APPARENT_ORDER, MAX_APPARENT_ORDER) == (1e-3, 40.0)

    def test_a_zero_difference_raises_rather_than_returning_a_number(self):
        with pytest.raises(ValueError, match="identical"):
            apparent_order(0.0, 1.0, 2.0, 2.0)
        with pytest.raises(ValueError, match="identical"):
            apparent_order(1.0, 0.0, 2.0, 2.0)

    def test_a_ratio_at_or_below_one_is_refused(self):
        with pytest.raises(ValueError, match="must exceed 1"):
            apparent_order(0.1, 0.4, 1.0, 2.0)


class TestTheSafetyFactorConvention:
    """Which ``Fs`` is applied, and on what evidence.

    Roache (1994, 1998) ties the narrow 1.25 band to a *measured*
    order from three or more grids that agrees with the formal one; the
    ASME V&V 20 recipe (Celik et al. 2008) uses 1.25 throughout its
    three-grid procedure.  This module takes the stricter reading: 1.25
    only where the asymptotic range was actually demonstrated, and 3.0
    wherever it was not or could not be.
    """

    def test_the_documented_constants_are_the_ones_in_use(self):
        assert (DEFAULT_SAFETY_FACTOR, CAUTIOUS_SAFETY_FACTOR) == (1.25, 3.0)
        assert DEFAULT_ASYMPTOTIC_ORDER_TOLERANCE == 0.25
        assert RECOMMENDED_MIN_REFINEMENT_RATIO == 1.3

    def test_the_narrow_factor_needs_the_asymptotic_range_demonstrated(self):
        hs = (0.1, 0.05, 0.025)
        values = [1.0 + 0.5 * h**2 for h in hs]
        with_claim = richardson_study(values, hs, axis=SPACE, formal_order=2.0)
        without = richardson_study(values, hs, axis=SPACE)

        assert with_claim.in_asymptotic_range is True
        assert with_claim.safety_factor == DEFAULT_SAFETY_FACTOR
        assert without.in_asymptotic_range is None
        assert without.safety_factor == CAUTIOUS_SAFETY_FACTOR
        assert without.gci_fine > with_claim.gci_fine

    def test_an_order_far_from_the_claim_gets_the_cautious_factor(self):
        hs = (0.1, 0.05, 0.025)
        study = richardson_study(
            [1.0 + 0.5 * h for h in hs], hs, axis=SPACE, formal_order=2.0,
        )
        assert study.in_asymptotic_range is False
        assert study.safety_factor == CAUTIOUS_SAFETY_FACTOR

    def test_an_explicit_factor_is_honoured(self):
        hs = (0.1, 0.05, 0.025)
        study = richardson_study(
            [1.0 + 0.5 * h**2 for h in hs], hs, axis=SPACE, safety_factor=1.0,
        )
        assert study.safety_factor == 1.0


def test_the_roache_asymptotic_ratio_is_near_tautological_and_is_not_gated_on():
    """Pins the reason a different asymptotic check had to be used.

    Roache's published test is ``GCI_coarse / (r**p GCI_fine) ~ 1``.
    Substituting the definitions and the ``p`` solved from the same
    three solutions, it collapses algebraically to
    ``|phi_fine| / |phi_medium|`` for a constant refinement ratio — so
    it is within a per cent of 1 for any ladder whose solutions are
    close together, converging or not.  Quoting it as evidence of the
    asymptotic range would be quoting a tautology, so
    :attr:`GridConvergenceStudy.in_asymptotic_range` is decided on the
    apparent order instead, and this test is what stops the reported
    ratio being mistaken for the check.
    """
    hs = (0.1, 0.05, 0.025)
    for values in ([1.0 + 0.5 * h**2 for h in hs], [3.0, 2.2, 2.0]):
        study = richardson_study(values, hs, axis=SPACE)
        assert study.asymptotic_ratio == pytest.approx(
            abs(study.values[-1]) / abs(study.values[-2]), rel=1e-9,
        )


# ---------------------------------------------------------------------------
# LBMPipeNode: the node that motivated the mode
# ---------------------------------------------------------------------------

_PIPE_TAU = 0.8                       # nu = (tau - 1/2)/3 = 0.1
_PIPE_FORCE = 1e-6                    # lattice units; deep in the Stokes regime
_PIPE_RADIUS_FRACTION = 0.9

#: Hagen-Poiseuille: a paraboloid has ``u_max / u_mean = 2`` exactly in a
#: circular pipe, whatever the driving force, the viscosity or the
#: radius.  Used only as context for what the ladder is converging
#: towards -- never as a reference the study is scored against, because
#: a GCI study that had a reference would not be a GCI study.
_POISEUILLE_SHAPE_FACTOR = 2.0


def test_mms_cannot_reach_lbm_pipe_at_all():
    """The premise of this whole file, pinned so it cannot rot quietly.

    ``LBMPipeNode`` declares no boundary inputs whatsoever.  Its only
    forcing path is an undeclared scalar ``propeller_force`` applied on
    an actuator-disc mask fixed at construction, which cannot carry a
    spatially varying manufactured source.  If this ever stops being
    true the node becomes MMS-testable and the argument in ``TODO.md``
    for preferring GCI here needs revisiting -- so this fails loudly
    rather than leaving a stale justification in place.
    """
    node = LBMPipeNode("pipe", 1.0, nx=1, ny=8, nz=8, tau=_PIPE_TAU)
    assert node.boundary_input_spec() == {}


@functools.lru_cache(maxsize=None)
def _pipe_shape_factor(n_cells):
    """``u_max / u_mean`` of the steady pipe flow at an ``n x n`` cross-section.

    A dimensionless functional of the converged solution, which is what
    a GCI study needs: one scalar per refinement level, computed the
    same way at every level, with no reference solution anywhere in it.

    The setup is a fully developed pipe flow.  ``nx = 1`` with the
    propeller plane at ``x = 0`` and ``propeller_radius = 1.0`` turns
    the actuator disc into a uniform axial body force over the whole
    fluid cross-section -- the closest this node can come to a driven
    Poiseuille flow, and the only configuration in which its one
    forcing input is spatially uniform.  The pipe radius is a fixed
    *fraction* of the cross-section, so refining ``n`` refines the same
    physical geometry.

    Run to steady state: the slowest decay mode of a cylinder relaxes
    at ``5.78 nu / R^2``, so ``2 R^2 / nu`` steps is 11.6 e-foldings.
    Verified converged -- the functional moves by under ``2e-6``
    relative between this and twice as many steps, three orders of
    magnitude below the smallest difference the ladder resolves.

    Cached, because the two studies below share levels and each run is
    several thousand lattice steps.
    """
    node = LBMPipeNode(
        "pipe", 1.0, nx=1, ny=n_cells, nz=n_cells, tau=_PIPE_TAU,
        pipe_radius=_PIPE_RADIUS_FRACTION, propeller_x=0,
        propeller_radius=1.0, propeller_strength=_PIPE_FORCE,
        initial_velocity=0.0,
    )
    with _float64():
        state = {
            key: jnp.asarray(value, jnp.float64)
            for key, value in node.initial_state().items()
        }
        step = jax.jit(lambda s: node.update(s, {}, 1.0))
        viscosity = (_PIPE_TAU - 0.5) / 3.0
        radius = _PIPE_RADIUS_FRACTION * n_cells / 2.0
        steps = int(2.0 * radius * radius / viscosity) + 50
        state = jax.lax.fori_loop(0, steps, lambda _, s: step(s), state)
        u_x = np.asarray(jax.device_get(state["velocity"]), np.float64)[0, ..., 0]
        fluid = ~np.asarray(jax.device_get(node._wall_mask))[0]
    return float(u_x[fluid].max() / u_x[fluid].mean())


def _pipe_node(n_cells=16):
    return LBMPipeNode(
        "pipe", 1.0, nx=1, ny=n_cells, nz=n_cells, tau=_PIPE_TAU,
        pipe_radius=_PIPE_RADIUS_FRACTION, propeller_x=0,
        propeller_radius=1.0, propeller_strength=_PIPE_FORCE,
    )


@verification_benchmark(
    benchmark_id="MADD-VER-013",
    description=(
        "LBMPipeNode grid convergence by Richardson extrapolation: the "
        "dimensionless shape factor u_max/u_mean of the steady, uniformly "
        "forced pipe flow, refined over a 12/16/24 cross-section ladder.  "
        "No manufactured source and no reference solution, which is what "
        "lets it cover a node the MMS harness cannot reach at all."
    ),
    node_type="LBMPipeNode",
    benchmark_type=BenchmarkType.CONVERGENCE_STUDY,
    acceptance_criteria=(
        "Monotone convergence (R = 0.846 in (0, 1)) with a determinate "
        "observed order (measured: 1.49) and a Grid Convergence Index on "
        "the finest solution below 25% at Fs = 3.0 (measured: 10.7%).  "
        "Fs = 3.0 and not 1.25 because the node declares no order of "
        "accuracy, so the asymptotic range cannot be established from "
        "three levels; a fourth level shows it is not in it (MADD-VER-013 "
        "is a convergence statement, not an accuracy one)."
    ),
    references=(
        "Roache1994: Perspective: A Method for Uniform Reporting of Grid "
        "Refinement Studies",
        "Celik2008: Procedure for Estimation and Reporting of Uncertainty "
        "Due to Discretization in CFD Applications (ASME V&V 20)",
    ),
)
def test_the_pipe_flow_converges_under_refinement_where_mms_cannot_run():
    """``LBMPipeNode`` gets a convergence verdict and an error band.

    The headline of the mode: a node with no forcing input MMS can use,
    and no closed-form solution the harness is allowed to look at,
    still yields an observed order and a quantified uncertainty from
    three runs of itself.

    Read the band, not the order.  Three levels and an undeclared node
    cannot settle the asymptotic range, so the cautious safety factor
    applies and the companion test below shows a fourth level puts the
    study *outside* it.  What this test pins is that the ladder
    converges monotonically and the harness reaches a definite verdict
    on a node the other harness cannot touch.
    """
    study = measure_gci(_pipe_shape_factor, (12, 16, 24), axis=SPACE)

    assert study.regime is ConvergenceRegime.MONOTONE
    assert 0.0 < study.ratio < 1.0
    assert math.isfinite(study.order)
    assert study.in_asymptotic_range is None
    assert study.safety_factor == CAUTIOUS_SAFETY_FACTOR
    assert check_gci(study, max_gci=0.25).status == "PASS"

    # Converging on the paraboloid, from below, as the geometry says.
    assert all(v < _POISEUILLE_SHAPE_FACTOR for v in study.values)
    assert abs(study.values[-1] - _POISEUILLE_SHAPE_FACTOR) < abs(
        study.values[0] - _POISEUILLE_SHAPE_FACTOR
    )


def test_the_pipe_ladder_is_not_in_the_asymptotic_range_and_the_harness_says_so(
):
    """The finding, and the reason a GCI needs the asymptotic-range check.

    A fourth level gives a second, independent estimate of the order,
    and the two disagree by more than an order of accuracy: 1.49 over
    12/16/24 against 3.35 over 16/24/32.  The pipe wall is a circle
    staircased onto a Cartesian lattice with bounce-back, and the
    effective wall position jumps about as the resolution changes, so
    the differences do not follow a single power law.

    This matters because the band is *wrong* here, in the confident
    direction: the 16/24/32 triple quotes a Grid Convergence Index of
    1.2% around a solution 3.6% from the Hagen-Poiseuille answer.  A
    harness that reported that number without the asymptotic-range
    check would be understating its own error threefold -- which is why
    a study shown to be outside the range fails rather than reporting.
    """
    study = measure_gci(_pipe_shape_factor, (12, 16, 24, 32), axis=SPACE)

    assert study.regime is ConvergenceRegime.MONOTONE
    assert study.in_asymptotic_range is False
    assert "per-triple orders span" in study.asymptotic_detail
    spread = max(study.triplet_orders) - min(study.triplet_orders)
    assert spread > 1.0

    result = check_gci(study, max_gci=0.25)
    assert result.failed
    assert "not in the asymptotic range" in result.detail

    # The band from the finest triple alone, and what it misses.
    finest_triple = measure_gci(_pipe_shape_factor, (16, 24, 32), axis=SPACE)
    true_relative_error = abs(
        (finest_triple.values[-1] - _POISEUILLE_SHAPE_FACTOR)
        / finest_triple.values[-1]
    )
    assert finest_triple.gci_fine < true_relative_error, (
        "this test documents a band that is too narrow outside the "
        "asymptotic range; if the band now covers the error, the warning "
        "above needs rewriting rather than the assertion relaxing"
    )


def test_lbm_pipe_declares_no_order_so_the_study_skips_rather_than_passing():
    """An undeclared node with a converging ladder still is not verified.

    ``LBMPipeNode`` claims "2nd-order in space and time for low Ma" in
    prose, in ``meta.discretization``, but declares no
    ``DiscretizationOrder``, so there is nothing machine-readable to
    hold the measurement to.  The harness says so instead of reporting
    a pass -- and the skip carries the whole measurement, so nothing
    the study learned is thrown away.
    """
    node = _pipe_node()
    result = verify_node_gci(
        node, axis=SPACE, solution_at=_pipe_shape_factor, levels=(12, 16, 24),
    )
    assert result.status == "SKIP"
    assert "declares no spatial order" in result.detail
    assert "observed order p" in result.detail


class TestBenchmarkRegistration:
    def test_the_benchmark_is_registered(self):
        assert "MADD-VER-013" in _BENCHMARK_REGISTRY

    def test_the_benchmark_names_its_node_and_its_method(self):
        benchmark = _BENCHMARK_REGISTRY["MADD-VER-013"]
        assert benchmark.node_type == "LBMPipeNode"
        assert benchmark.benchmark_type is BenchmarkType.CONVERGENCE_STUDY
        assert benchmark.test_function.endswith(
            "test_the_pipe_flow_converges_under_refinement_where_mms_cannot_run"
        )
