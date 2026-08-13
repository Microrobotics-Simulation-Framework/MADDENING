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
