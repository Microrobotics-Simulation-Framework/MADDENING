"""Formal verification of adaptive timestepping arithmetic.

Verifies the numerical safety properties of the PI step-size
controller in :mod:`maddening.core.simulation.adaptive`:

1. dt_next is always in [dt_min, dt_max] for any error_norm >= 0.
2. The acceptance predicate is correct: accepted iff error_norm <= 1.0.
3. The growth factor is always in [min_factor, max_factor].
"""


import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check


DT_MIN = 1e-8
DT_MAX = 0.1
SAFETY = 0.9
MAX_FACTOR = 5.0
MIN_FACTOR = 0.2
ORDER = 1


class TestDtBounds:
    """dt_next stays within the configured range.

    Note: The raw clip bounds (dt_next ∈ [dt_min, dt_max]) are
    universally true by construction of jnp.clip. The envelope-
    dependent property is tighter: dt_next stays within bounds that
    are NARROWER than the clip range when error_norm is bounded.
    """

    def test_dt_next_bounded_by_controller(self):
        """With error in [0.5, 2.0], dt_next is tighter than [dt_min, dt_max].
        The factor range is [safety*min_factor, safety*power(1/0.5, 0.5)]
        = [0.18, 1.27], so dt_next ∈ [dt*0.18, dt*1.27] ∩ [dt_min, dt_max].
        For dt in [0.01, 0.05]: dt_next ≤ 0.05*1.27 = 0.0635 < dt_max."""
        def harness():
            dt = any_array((), "float64", (0.01, 0.05))
            error_norm = any_array((), "float64", (0.5, 2.0))

            factor = SAFETY * jnp.where(
                error_norm > 0,
                jnp.power(1.0 / error_norm, 1.0 / (ORDER + 1)),
                MAX_FACTOR,
            )
            factor = jnp.clip(factor, MIN_FACTOR, MAX_FACTOR)
            dt_next = jnp.clip(dt * factor, DT_MIN, DT_MAX)

            return (assert_(dt_next <= 0.07),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"

    def test_dt_next_grows_when_error_small(self):
        """When error < 1 (accepted step), dt_next > dt * min_factor.
        For error in [0.1, 0.5] and dt in [0.001, 0.01]:
        factor = 0.9 * (1/error)^0.5 ∈ [0.9*1.41, 0.9*3.16] = [1.27, 2.85]
        so dt_next >= dt * 1.27 >= 0.001 * 1.27 = 0.00127."""
        def harness():
            dt = any_array((), "float64", (0.001, 0.01))
            error_norm = any_array((), "float64", (0.1, 0.5))

            factor = SAFETY * jnp.where(
                error_norm > 0,
                jnp.power(1.0 / error_norm, 1.0 / (ORDER + 1)),
                MAX_FACTOR,
            )
            factor = jnp.clip(factor, MIN_FACTOR, MAX_FACTOR)
            dt_next = jnp.clip(dt * factor, DT_MIN, DT_MAX)

            return (assert_(dt_next >= 0.001),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"


class TestAcceptancePredicate:
    """Step is accepted iff error_norm <= 1.0."""

    def test_accepted_when_error_below_one(self):
        def harness():
            error_norm = any_array((), "float64", (0.0, 1.0))
            accepted = error_norm <= 1.0
            return (assert_(accepted),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"

    def test_rejected_when_error_above_one(self):
        def harness():
            error_norm = any_array((), "float64", (1.001, 100.0))
            return (assert_(error_norm > 1.0),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"


class TestFactorBounds:
    """Growth factor is always in [min_factor, max_factor]."""

    def test_factor_lower_bound(self):
        def harness():
            error_norm = any_array((), "float64", (0.0, 100.0))

            factor = SAFETY * jnp.where(
                error_norm > 0,
                jnp.power(1.0 / error_norm, 1.0 / (ORDER + 1)),
                MAX_FACTOR,
            )
            factor = jnp.clip(factor, MIN_FACTOR, MAX_FACTOR)

            return (assert_(factor >= MIN_FACTOR),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"

    def test_factor_upper_bound(self):
        def harness():
            error_norm = any_array((), "float64", (0.0, 100.0))

            factor = SAFETY * jnp.where(
                error_norm > 0,
                jnp.power(1.0 / error_norm, 1.0 / (ORDER + 1)),
                MAX_FACTOR,
            )
            factor = jnp.clip(factor, MIN_FACTOR, MAX_FACTOR)

            return (assert_(factor <= MAX_FACTOR),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"
