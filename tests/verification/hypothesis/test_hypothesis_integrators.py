"""Property-based tests for time integrators.

Tests numerical properties of Euler, Heun, and RK4 from
:mod:`maddening.core.simulation.integrators`.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.simulation.integrators import euler_step, heun_step, rk4_step
from tests.conftest import EXAMPLES_CHEAP, EXAMPLES_STANDARD


def _constant_derivs(state, boundary_inputs):
    """Derivative function that returns constants (dx/dt = 1 for all fields)."""
    return {k: jnp.ones_like(v) for k, v in state.items()}


def _linear_derivs(state, boundary_inputs):
    """dx/dt = -x (exponential decay)."""
    return {k: -v for k, v in state.items()}


dt_st = st.floats(min_value=1e-6, max_value=0.1,
                  allow_nan=False, allow_infinity=False)

#: Smallest positive *normal* float32, ``2**-126``.  XLA flushes anything
#: below it to zero, so a state value in that range is not preserved by a
#: zero-length step and the exact-equality properties below do not hold there.
F32_MIN_NORMAL = 1.18e-38

#: States a zero-length step must return bit-identical: zero, or a magnitude
#: XLA will not flush.  Generated rather than assumed.  ``st.floats`` draws
#: float64 and deliberately favours the nasty end of the range -- subnormals,
#: 5e-324, 1e-308 -- so ``assume(abs(x) > F32_MIN_NORMAL or x == 0.0)`` over a
#: plain ``floats(-1e4, 1e4)`` threw away 23% of every draw in this class,
#: measured.  Excluding the range instead rejects 0.0%.
normal_f32_st = st.one_of(
    st.just(0.0),
    st.floats(min_value=F32_MIN_NORMAL, max_value=1e4, exclude_min=True,
              allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1e4, max_value=-F32_MIN_NORMAL, exclude_max=True,
              allow_nan=False, allow_infinity=False),
)


class TestZeroStepIdentity:
    """update(state, {}, dt=0) should return state unchanged.

    XLA flushes subnormal float32 values to zero, so the subnormal range
    (``0 < |x| < 1.18e-38``) is outside the property.  It is excluded by
    ``normal_f32_st`` rather than by ``assume``: these tests reject 0.0% of
    their draws, and the ``assert`` below says so out loud, so a future
    change to the strategy that lets subnormals back in fails here instead
    of quietly returning to discarding a quarter of the search.
    """

    @given(x=normal_f32_st)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_euler_zero_dt(self, x):
        assert abs(x) > F32_MIN_NORMAL or x == 0.0, (
            "normal_f32_st must not produce a value XLA flushes to zero"
        )
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = euler_step(_constant_derivs, state, {}, 0.0)
        assert float(out["x"]) == float(state["x"])

    @given(x=normal_f32_st)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_heun_zero_dt(self, x):
        assert abs(x) > F32_MIN_NORMAL or x == 0.0, (
            "normal_f32_st must not produce a value XLA flushes to zero"
        )
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = heun_step(_constant_derivs, state, {}, 0.0)
        assert float(out["x"]) == float(state["x"])

    @given(x=normal_f32_st)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_rk4_zero_dt(self, x):
        assert abs(x) > F32_MIN_NORMAL or x == 0.0, (
            "normal_f32_st must not produce a value XLA flushes to zero"
        )
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = rk4_step(_constant_derivs, state, {}, 0.0)
        assert float(out["x"]) == float(state["x"])


class TestConstantDerivativeExact:
    """For dx/dt = c, all integrators should give x + dt*c exactly."""

    @given(x=st.floats(min_value=-1e3, max_value=1e3,
                       allow_nan=False, allow_infinity=False),
           dt=dt_st)
    @settings(max_examples=EXAMPLES_CHEAP)
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
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_heun_constant(self, x, dt):
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = heun_step(_constant_derivs, state, {}, dt)
        expected = x + dt
        assert abs(float(out["x"]) - expected) < 1e-4

    @given(x=st.floats(min_value=-1e3, max_value=1e3,
                       allow_nan=False, allow_infinity=False),
           dt=dt_st)
    @settings(max_examples=EXAMPLES_CHEAP)
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
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_euler_finite(self, x, dt):
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = euler_step(_linear_derivs, state, {}, dt)
        assert jnp.isfinite(out["x"]), f"Euler non-finite for x={x}, dt={dt}"

    @given(x=st.floats(min_value=-1e4, max_value=1e4,
                       allow_nan=False, allow_infinity=False),
           dt=dt_st)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_rk4_finite(self, x, dt):
        state = {"x": jnp.array(x, dtype=jnp.float32)}
        out = rk4_step(_linear_derivs, state, {}, dt)
        assert jnp.isfinite(out["x"]), f"RK4 non-finite for x={x}, dt={dt}"


class TestOrderVerification:
    """Halving dt should reduce error by ~2^order for smooth problems.

    These tests need float64 precision to observe the expected
    convergence rates (float32 noise dominates at the error levels being
    compared), so they enable ``jax_enable_x64`` for their body (and
    restore it) instead of skipping when the process is float32, which
    is every run of the suite locally and in CI.
    """

    @given(x=st.floats(min_value=0.1, max_value=10.0,
                       allow_nan=False, allow_infinity=False))
    @settings(max_examples=EXAMPLES_STANDARD)
    def test_euler_first_order(self, x):
        import jax
        was = bool(jax.config.jax_enable_x64)
        jax.config.update("jax_enable_x64", True)
        try:
            self._euler_order_body(x)
        finally:
            jax.config.update("jax_enable_x64", was)

    def _euler_order_body(self, x):
        state = {"x": jnp.array(x, dtype=jnp.float64)}
        assert state["x"].dtype == jnp.float64
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
    @settings(max_examples=EXAMPLES_STANDARD)
    def test_rk4_fourth_order(self, x):
        import jax
        was = bool(jax.config.jax_enable_x64)
        jax.config.update("jax_enable_x64", True)
        try:
            self._rk4_order_body(x)
        finally:
            jax.config.update("jax_enable_x64", was)

    def _rk4_order_body(self, x):
        state = {"x": jnp.array(x, dtype=jnp.float64)}
        assert state["x"].dtype == jnp.float64
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
