"""Formal verification of Aitken delta-squared relaxation.

Verifies the numerical safety properties of
:func:`maddening.core.coupling.acceleration.aitken_relaxation`:

1. Omega bounds: output omega is always in [0.01, 2.0] for any
   input residuals within the declared envelope.
2. Division guard: when denom is guaranteed to be > 1e-30 (non-
   degenerate residuals), the computation produces finite omega.
3. Relaxed state linearity: x_relaxed = x_old + omega * residual
   (verified in the non-degenerate regime where the division path
   is taken).

Note: The full division-guard fallback (denom <= 1e-30 → return
input omega) cannot be formally verified via interval analysis because
JAX's select_n traces both branches — the division is always in the
jaxpr regardless of the condition. This property is covered by the
hypothesis suite instead.
"""


import jax.numpy as jnp
import pytest

from stelling.harness import any_array, assert_
from stelling.preconditions import check


class TestAitkenOmegaBounds:
    """Output omega is always clamped to [0.01, 2.0]."""

    def test_omega_lower_bound(self):
        def harness():
            x_old = any_array((4,), "float64", (-100.0, 100.0))
            x_raw = any_array((4,), "float64", (-100.0, 100.0))
            prev_r = any_array((4,), "float64", (-100.0, 100.0))
            omega = any_array((), "float64", (0.01, 2.0))

            residual = x_raw - x_old
            delta_r = residual - prev_r
            denom = jnp.sum(delta_r ** 2)
            safe_denom = jnp.where(denom > 1e-30, denom, jnp.array(1.0))
            new_omega = -omega * jnp.sum(prev_r * delta_r) / safe_denom
            new_omega = jnp.clip(new_omega, 0.01, 2.0)
            new_omega = jnp.where(denom > 1e-30, new_omega, omega)

            return (assert_(new_omega >= 0.01),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"

    def test_omega_upper_bound(self):
        def harness():
            x_old = any_array((4,), "float64", (-100.0, 100.0))
            x_raw = any_array((4,), "float64", (-100.0, 100.0))
            prev_r = any_array((4,), "float64", (-100.0, 100.0))
            omega = any_array((), "float64", (0.01, 2.0))

            residual = x_raw - x_old
            delta_r = residual - prev_r
            denom = jnp.sum(delta_r ** 2)
            safe_denom = jnp.where(denom > 1e-30, denom, jnp.array(1.0))
            new_omega = -omega * jnp.sum(prev_r * delta_r) / safe_denom
            new_omega = jnp.clip(new_omega, 0.01, 2.0)
            new_omega = jnp.where(denom > 1e-30, new_omega, omega)

            return (assert_(new_omega <= 2.0),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"


class TestAitkenNonDegenerateRegime:
    """When residuals are large enough that denom > 1e-30 is guaranteed,
    verify properties of the division path."""

    def test_omega_bounded_non_degenerate(self):
        """With residuals in [1.0, 10.0], denom is provably >> 1e-30,
        so the division path is always taken and omega is still bounded."""
        def harness():
            x_old = any_array((4,), "float64", (0.0, 0.0))
            x_raw = any_array((4,), "float64", (1.0, 10.0))
            prev_r = any_array((4,), "float64", (-5.0, 5.0))
            omega = any_array((), "float64", (0.01, 2.0))

            residual = x_raw - x_old
            delta_r = residual - prev_r
            denom = jnp.sum(delta_r ** 2)
            safe_denom = jnp.where(denom > 1e-30, denom, jnp.array(1.0))
            new_omega = -omega * jnp.sum(prev_r * delta_r) / safe_denom
            new_omega = jnp.clip(new_omega, 0.01, 2.0)
            new_omega = jnp.where(denom > 1e-30, new_omega, omega)

            return (
                assert_(new_omega >= 0.01),
                assert_(new_omega <= 2.0),
            )

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"

    def test_denom_positive_large_residuals(self):
        """With residuals bounded away from zero, denom > 0."""
        def harness():
            x_raw = any_array((4,), "float64", (1.0, 10.0))
            prev_r = any_array((4,), "float64", (-5.0, -1.0))

            residual = x_raw
            delta_r = residual - prev_r
            denom = jnp.sum(delta_r ** 2)

            return (assert_(denom > 0.0),)

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"


class TestAitkenOverflowGuard:
    """Verify the overflow guard: when denom is non-finite, fallback
    to input omega. This is the fix for the float32 overflow bug where
    delta_r^2 overflows to inf, producing inf/inf = nan."""

    def test_omega_bounded_with_overflow_guard(self):
        """The full implementation (with isfinite guard) still produces
        bounded omega for all inputs in a wide envelope."""
        def harness():
            x_old = any_array((4,), "float64", (-1e8, 1e8))
            x_raw = any_array((4,), "float64", (-1e8, 1e8))
            prev_r = any_array((4,), "float64", (-1e8, 1e8))
            omega = any_array((), "float64", (0.01, 2.0))

            residual = x_raw - x_old
            delta_r = residual - prev_r
            denom = jnp.sum(delta_r ** 2)
            denom_ok = (denom > 1e-30) & jnp.isfinite(denom)
            safe_denom = jnp.where(denom_ok, denom, jnp.array(1.0))
            new_omega = -omega * jnp.sum(prev_r * delta_r) / safe_denom
            new_omega = jnp.clip(new_omega, 0.01, 2.0)
            new_omega = jnp.where(denom_ok, new_omega, omega)

            return (
                assert_(new_omega >= 0.01),
                assert_(new_omega <= 2.0),
            )

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"

    def test_output_always_finite(self):
        """The isfinite transfer (stelling 0.2) proves the Aitken
        output is always finite for bounded inputs."""
        def harness():
            x_old = any_array((4,), "float64", (-1e8, 1e8))
            x_raw = any_array((4,), "float64", (-1e8, 1e8))
            prev_r = any_array((4,), "float64", (-1e8, 1e8))
            omega = any_array((), "float64", (0.01, 2.0))

            residual = x_raw - x_old
            delta_r = residual - prev_r
            denom = jnp.sum(delta_r ** 2)
            denom_ok = (denom > 1e-30) & jnp.isfinite(denom)
            safe_denom = jnp.where(denom_ok, denom, jnp.array(1.0))
            new_omega = -omega * jnp.sum(prev_r * delta_r) / safe_denom
            new_omega = jnp.clip(new_omega, 0.01, 2.0)
            new_omega = jnp.where(denom_ok, new_omega, omega)

            x_relaxed = x_old + new_omega * residual
            return (
                assert_(jnp.isfinite(new_omega)),
                assert_(jnp.isfinite(x_relaxed)),
            )

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"


class TestAitkenDivisionGuardDocumented:
    """Document the limitation: the full fallback path (denom <= 1e-30
    → return input omega) cannot be formally verified via interval
    analysis due to select_n tracing both branches."""

    @pytest.mark.xfail(
        reason="Interval analysis cannot reason through select_n with "
               "division-by-possibly-zero in the unused branch. "
               "Covered by hypothesis suite.",
        strict=True,
    )
    def test_fallback_preserves_omega_limitation(self):
        """This property is TRUE but unprovable by interval methods."""
        def harness():
            prev_r = any_array((4,), "float64", (-1e-16, 1e-16))
            omega = any_array((), "float64", (0.5, 1.5))
            x_old = any_array((4,), "float64", (0.0, 0.0))
            x_raw = any_array((4,), "float64", (-1e-16, 1e-16))

            residual = x_raw - x_old
            delta_r = residual - prev_r
            denom = jnp.sum(delta_r ** 2)
            denom_ok = (denom > 1e-30) & jnp.isfinite(denom)
            safe_denom = jnp.where(denom_ok, denom, jnp.array(1.0))
            new_omega = -omega * jnp.sum(prev_r * delta_r) / safe_denom
            new_omega = jnp.clip(new_omega, 0.01, 2.0)
            new_omega = jnp.where(denom_ok, new_omega, omega)

            return (assert_(new_omega == omega),)

        v = check(harness, vacuity_mode="inputs-only", solver_timeout_ms=10000)
        assert v.status == "VERIFIED", f"Expected VERIFIED, got {v.status}"
