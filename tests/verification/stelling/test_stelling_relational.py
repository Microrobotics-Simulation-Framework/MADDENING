"""Relational property verification (stelling 0.2 — solver-forwarded assumes).

These tests use assume(e1 < e2) to prove ordering/monotonicity
properties that are impossible in a non-relational interval domain
but trivial for the SMT solver.
"""

import jax.numpy as jnp

from stelling.harness import any_array, assert_, assume
from stelling.preconditions import check


SOLVER_TIMEOUT = 15000


class TestPIControllerMonotonicity:
    """The PI step-size controller assigns larger growth factors to
    smaller errors — the fundamental correctness property of adaptive
    timestepping."""

    def test_inverse_error_monotone(self):
        """1/e1 > 1/e2 when e1 < e2 (both positive)."""
        def harness():
            e1 = any_array((), "float64", (0.01, 5.0))
            e2 = any_array((), "float64", (0.01, 5.0))
            assume(e1 < e2)
            return (assert_(1.0 / e1 > 1.0 / e2),)

        v = check(harness, vacuity_mode="inputs-only",
                  solver_timeout_ms=SOLVER_TIMEOUT)
        assert v.status == "VERIFIED"

    def test_clipped_factor_monotone(self):
        """clip(0.9/e, 0.2, 5.0) is non-increasing in e."""
        def harness():
            e1 = any_array((), "float64", (0.01, 5.0))
            e2 = any_array((), "float64", (0.01, 5.0))
            assume(e1 < e2)
            f1 = jnp.clip(0.9 / e1, 0.2, 5.0)
            f2 = jnp.clip(0.9 / e2, 0.2, 5.0)
            return (assert_(f1 >= f2),)

        v = check(harness, vacuity_mode="inputs-only",
                  solver_timeout_ms=SOLVER_TIMEOUT)
        assert v.status == "VERIFIED"

    def test_full_pi_with_sqrt_monotone(self):
        """The full PI formula with sqrt: 0.9 * (1/e)^0.5 is
        non-increasing (rational pow emission)."""
        def harness():
            e1 = any_array((), "float64", (0.01, 5.0))
            e2 = any_array((), "float64", (0.01, 5.0))
            assume(e1 < e2)
            f1 = jnp.clip(0.9 * jnp.power(1.0 / e1, 0.5), 0.2, 5.0)
            f2 = jnp.clip(0.9 * jnp.power(1.0 / e2, 0.5), 0.2, 5.0)
            return (assert_(f1 >= f2),)

        v = check(harness, vacuity_mode="inputs-only",
                  solver_timeout_ms=SOLVER_TIMEOUT)
        assert v.status == "VERIFIED"


class TestNewtonConvergence:
    """Newton's method converges in one step for linear residuals."""

    def test_linear_residual_exact(self):
        """R(x) = x*(1+dt*k) - x_old is linear. Newton solves it in
        one step: x_new = x_old / (1+dt*k), residual = 0."""
        def harness():
            x = any_array((), "float64", (-10.0, 10.0))
            x_old = any_array((), "float64", (-10.0, 10.0))
            J = 2.0  # 1 + dt*k with dt=0.01, k=100
            r = x * J - x_old
            x_new = x - r / J
            r_new = x_new * J - x_old
            return (assert_(r_new == 0.0),)

        v = check(harness, vacuity_mode="inputs-only",
                  solver_timeout_ms=SOLVER_TIMEOUT)
        assert v.status == "VERIFIED"


class TestSpringDivisionSafety:
    """SpringDamperNode force/m is always finite when m > 0."""

    def test_accel_finite_for_bounded_params(self):
        """With m ∈ [0.001, 100], force/m is finite for any bounded
        state and parameters."""
        def harness():
            pos = any_array((), "float64", (-100.0, 100.0))
            vel = any_array((), "float64", (-100.0, 100.0))
            k = any_array((), "float64", (0.1, 1e6))
            c = any_array((), "float64", (0.0, 100.0))
            m = any_array((), "float64", (0.001, 100.0))
            rest = any_array((), "float64", (-10.0, 10.0))
            dt = any_array((), "float64", (1e-5, 0.1))

            force = -k * (pos - rest) - c * vel
            accel = force / m
            new_vel = vel + accel * dt
            new_pos = pos + new_vel * dt

            return (
                assert_(jnp.isfinite(accel)),
                assert_(jnp.isfinite(new_vel)),
                assert_(jnp.isfinite(new_pos)),
            )

        v = check(harness, vacuity_mode="inputs-only")
        assert v.status == "VERIFIED"


class TestSquareMonotonicity:
    """x^2 ordering for positive inputs (pow emission + relational)."""

    def test_square_preserves_order(self):
        """0 < e1 < e2 implies e1^2 < e2^2."""
        def harness():
            e1 = any_array((), "float64", (0.1, 10.0))
            e2 = any_array((), "float64", (0.1, 10.0))
            assume(e1 < e2)
            return (assert_(e1 ** 2 <= e2 ** 2),)

        v = check(harness, vacuity_mode="inputs-only",
                  solver_timeout_ms=SOLVER_TIMEOUT)
        assert v.status == "VERIFIED"
