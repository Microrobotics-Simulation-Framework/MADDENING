"""Formal verification of coupling convergence norms.

Verifies the numerical safety properties of the convergence norms
in :mod:`maddening.core.coupling.acceleration`:

1. L2 norm is non-negative for all inputs.
2. L2 norm is zero when inputs are identical (requires solver for
   correlation tracking).
3. Mixed norm: the denominator (atol + rtol * max(|a|, |b|)) is
   strictly positive when atol > 0 (scalar-level verification).
4. Mixed norm: the full RMS norm is non-negative.
"""


import jax.numpy as jnp

from stelling.harness import any_array, assert_
from stelling.preconditions import check

SOLVER_TIMEOUT = 15000


class TestL2NormNonNegativity:
    """L2 norm (sqrt of sum of squares) is always >= 0."""

    def test_l2_non_negative(self):
        def harness():
            a = any_array((8,), "float64", (-1e6, 1e6))
            b = any_array((8,), "float64", (-1e6, 1e6))
            diff = a - b
            norm = jnp.sqrt(jnp.sum(diff ** 2))
            return (assert_(norm >= 0.0),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"

    def test_l2_zero_when_identical(self):
        """Requires solver: interval arithmetic cannot track that a-a=0
        because it doesn't preserve variable correlation."""
        def harness():
            a = any_array((), "float64", (-1e6, 1e6))
            diff = a - a
            return (assert_(diff == 0.0),)

        v = check(harness, vacuity_mode="inputs-only",
                  solver_timeout_ms=SOLVER_TIMEOUT)
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"


class TestMixedNormDenominator:
    """Scale factor is strictly positive when atol > 0.

    Verified per-element (scalar declarations) because stelling's
    interval propagation handles scalar pointwise assertions; the
    jnp.all() reduction over arrays is not fully supported.
    """

    def test_scale_positive_with_positive_atol(self):
        def harness():
            a = any_array((), "float64", (-1e6, 1e6))
            b = any_array((), "float64", (-1e6, 1e6))
            atol = 1e-8
            rtol = 1e-6
            scale = atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))
            return (assert_(scale > 0.0),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"

    def test_scale_bounded_below_by_atol(self):
        """The scale is at least atol when max(|a|, |b|) is bounded away
        from zero. (At the exact boundary max=0, outward-rounded interval
        arithmetic straddles — the safety property 'scale > 0' above
        covers that case.)"""
        def harness():
            a = any_array((), "float64", (1.0, 1e6))
            b = any_array((), "float64", (1.0, 1e6))
            atol = 1e-8
            rtol = 1e-6
            scale = atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))
            return (assert_(scale >= atol),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"


class TestMixedNormNonNegativity:
    """The full mixed RMS norm is non-negative."""

    def test_mixed_norm_non_negative(self):
        def harness():
            a = any_array((4,), "float64", (-100.0, 100.0))
            b = any_array((4,), "float64", (-100.0, 100.0))
            atol = 1e-8
            rtol = 1e-6
            diff = jnp.abs(a - b)
            scale = atol + rtol * jnp.maximum(jnp.abs(a), jnp.abs(b))
            scaled = diff / scale
            sum_sq = jnp.sum(scaled ** 2)
            count = jnp.array(4, dtype=jnp.int32)
            norm = jnp.sqrt(sum_sq / jnp.maximum(count, 1))
            return (assert_(norm >= 0.0),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"
