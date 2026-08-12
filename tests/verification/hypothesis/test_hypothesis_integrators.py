"""Property-based tests for time integrators.

Tests numerical properties of Euler, Heun, and RK4 from
:mod:`maddening.core.simulation.integrators`.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from maddening.core.simulation.integrators import euler_step, heun_step, rk4_step


def _constant_derivs(state, boundary_inputs):
    """Derivative function that returns constants (dx/dt = 1 for all fields)."""
    return {k: jnp.ones_like(v) for k, v in state.items()}


def _linear_derivs(state, boundary_inputs):
    """dx/dt = -x (exponential decay)."""
    return {k: -v for k, v in state.items()}


dt_st = st.floats(min_value=1e-6, max_value=0.1,
                  allow_nan=False, allow_infinity=False)


class TestZeroStepIdentity:
    """update(state, {}, dt=0) should return state unchanged.

    Note: XLA flushes subnormal float32 values to zero, so we exclude
    the subnormal range (|x| < 1.18e-38) from exact-equality checks.
    """

    @given(x=st.floats(min_value=-1e4, max_value=1e4,
                       allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_euler_zero_dt(self, x):
        assume(abs(x) > 1.18e-38 or x == 0.0)
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = euler_step(_constant_derivs, state, {}, 0.0)
        assert float(out["x"]) == float(state["x"])

    @given(x=st.floats(min_value=-1e4, max_value=1e4,
                       allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_heun_zero_dt(self, x):
        assume(abs(x) > 1.18e-38 or x == 0.0)
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = heun_step(_constant_derivs, state, {}, 0.0)
        assert float(out["x"]) == float(state["x"])

    @given(x=st.floats(min_value=-1e4, max_value=1e4,
                       allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_rk4_zero_dt(self, x):
        assume(abs(x) > 1.18e-38 or x == 0.0)
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = rk4_step(_constant_derivs, state, {}, 0.0)
        assert float(out["x"]) == float(state["x"])


class TestConstantDerivativeExact:
    """For dx/dt = c, all integrators should give x + dt*c exactly."""

    @given(x=st.floats(min_value=-1e3, max_value=1e3,
                       allow_nan=False, allow_infinity=False),
           dt=dt_st)
    @settings(max_examples=500)
    def test_euler_constant(self, x, dt):
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = euler_step(_constant_derivs, state, {}, dt)
        expected = x + dt
        assert abs(float(out["x"]) - expected) < 1e-4, (
            f"Euler: {float(out['x'])} != {expected}"
        )

    @given(x=st.floats(min_value=-1e3, max_value=1e3,
                       allow_nan=False, allow_infinity=False),
           dt=dt_st)
    @settings(max_examples=500)
    def test_heun_constant(self, x, dt):
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = heun_step(_constant_derivs, state, {}, dt)
        expected = x + dt
        assert abs(float(out["x"]) - expected) < 1e-4

    @given(x=st.floats(min_value=-1e3, max_value=1e3,
                       allow_nan=False, allow_infinity=False),
           dt=dt_st)
    @settings(max_examples=500)
    def test_rk4_constant(self, x, dt):
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = rk4_step(_constant_derivs, state, {}, dt)
        expected = x + dt
        assert abs(float(out["x"]) - expected) < 1e-4


class TestFiniteOutput:
    """Integrators should produce finite output for bounded inputs."""

    @given(x=st.floats(min_value=-1e4, max_value=1e4,
                       allow_nan=False, allow_infinity=False),
           dt=dt_st)
    @settings(max_examples=500)
    def test_euler_finite(self, x, dt):
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = euler_step(_linear_derivs, state, {}, dt)
        assert jnp.isfinite(out["x"]), f"Euler non-finite for x={x}, dt={dt}"

    @given(x=st.floats(min_value=-1e4, max_value=1e4,
                       allow_nan=False, allow_infinity=False),
           dt=dt_st)
    @settings(max_examples=500)
    def test_rk4_finite(self, x, dt):
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = rk4_step(_linear_derivs, state, {}, dt)
        assert jnp.isfinite(out["x"]), f"RK4 non-finite for x={x}, dt={dt}"


class TestOrderVerification:
    """Halving dt should reduce error by ~2^order for smooth problems.

    These tests require float64 precision to observe the expected
    convergence rates (float32 noise dominates at the error levels
    being compared).
    """

    @given(x=st.floats(min_value=0.1, max_value=10.0,
                       allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_euler_first_order(self, x):
        import jax
        if not jax.config.jax_enable_x64:
            pytest.skip("Requires float64 for order verification")
        state = {"x": jnp.array(x, dtype=jnp.float64)}
        dt = 0.01
        out_dt = euler_step(_linear_derivs, state, {}, dt)
        s_half = euler_step(_linear_derivs, state, {}, dt / 2)
        out_half = euler_step(_linear_derivs, s_half, {}, dt / 2)
        exact = x * np.exp(-dt)
        err_dt = abs(float(out_dt["x"]) - exact)
        err_half = abs(float(out_half["x"]) - exact)
        if err_half > 1e-12:
            ratio = err_dt / err_half
            assert ratio > 1.5, (
                f"Euler order violation: ratio={ratio:.2f} (expected ~2)"
            )

    @given(x=st.floats(min_value=0.1, max_value=10.0,
                       allow_nan=False, allow_infinity=False))
    @settings(max_examples=100)
    def test_rk4_fourth_order(self, x):
        import jax
        if not jax.config.jax_enable_x64:
            pytest.skip("Requires float64 for order verification")
        state = {"x": jnp.array(x, dtype=jnp.float64)}
        dt = 0.01
        out_dt = rk4_step(_linear_derivs, state, {}, dt)
        s1 = rk4_step(_linear_derivs, state, {}, dt / 2)
        out_half = rk4_step(_linear_derivs, s1, {}, dt / 2)
        exact = x * np.exp(-dt)
        err_dt = abs(float(out_dt["x"]) - exact)
        err_half = abs(float(out_half["x"]) - exact)
        if err_half > 1e-14:
            ratio = err_dt / err_half
            assert ratio > 8.0, (
                f"RK4 order violation: ratio={ratio:.2f} (expected ~16)"
            )
