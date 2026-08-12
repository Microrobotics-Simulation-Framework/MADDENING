"""Property-based tests for adaptive timestepping arithmetic.

Tests the PI step-size controller logic extracted from
:func:`maddening.core.simulation.adaptive.build_adaptive_step`.
"""


import numpy as np
import jax.numpy as jnp
from hypothesis import given, settings, assume
from hypothesis import strategies as st


DT_MIN = 1e-8
DT_MAX = 0.1
SAFETY = 0.9
MAX_FACTOR = 5.0
MIN_FACTOR = 0.2
ORDER = 1


def step_size_controller(dt, error_norm):
    """Extract of the PI controller logic from adaptive.py."""
    factor = SAFETY * jnp.where(
        error_norm > 0,
        jnp.power(1.0 / error_norm, 1.0 / (ORDER + 1)),
        MAX_FACTOR,
    )
    factor = jnp.clip(factor, MIN_FACTOR, MAX_FACTOR)
    dt_next = jnp.clip(dt * factor, DT_MIN, DT_MAX)
    accepted = error_norm <= 1.0
    return dt_next, factor, accepted


dt_st = st.floats(min_value=DT_MIN, max_value=DT_MAX,
                  allow_nan=False, allow_infinity=False)
error_st = st.floats(min_value=0.0, max_value=1e6,
                     allow_nan=False, allow_infinity=False)

DT_MIN_F32 = float(np.float32(DT_MIN))
DT_MAX_F32 = float(np.float32(DT_MAX))


class TestDtBoundsProperty:
    """dt_next is always in [dt_min, dt_max] for any valid inputs."""

    @given(dt=dt_st, error_norm=error_st)
    @settings(max_examples=1000)
    def test_dt_next_in_bounds(self, dt, error_norm):
        dt_j = jnp.asarray(dt)
        err_j = jnp.asarray(error_norm)
        dt_next, _, _ = step_size_controller(dt_j, err_j)
        val = float(dt_next)
        assert DT_MIN_F32 <= val <= DT_MAX_F32, (
            f"dt_next={val} not in [{DT_MIN_F32}, {DT_MAX_F32}] "
            f"(dt={dt}, error={error_norm})"
        )


class TestFactorBoundsProperty:
    """Growth factor is always in [min_factor, max_factor]."""

    @given(dt=dt_st, error_norm=error_st)
    @settings(max_examples=1000)
    def test_factor_in_bounds(self, dt, error_norm):
        dt_j = jnp.asarray(dt)
        err_j = jnp.asarray(error_norm)
        _, factor, _ = step_size_controller(dt_j, err_j)
        val = float(factor)
        min_f32 = float(np.float32(MIN_FACTOR))
        max_f32 = float(np.float32(MAX_FACTOR))
        assert min_f32 <= val <= max_f32, (
            f"factor={val} not in [{min_f32}, {max_f32}] "
            f"(dt={dt}, error={error_norm})"
        )


class TestAcceptanceMonotone:
    """If error_a < error_b and error_b is accepted, then error_a is
    also accepted. (Monotonicity of the acceptance predicate.)"""

    @given(dt=dt_st,
           error_a=st.floats(min_value=0.0, max_value=1.0,
                             allow_nan=False, allow_infinity=False),
           error_b=st.floats(min_value=0.0, max_value=1.0,
                             allow_nan=False, allow_infinity=False))
    @settings(max_examples=500)
    def test_lower_error_also_accepted(self, dt, error_a, error_b):
        assume(error_a <= error_b)
        dt_j = jnp.asarray(dt)
        _, _, accepted_a = step_size_controller(dt_j, jnp.asarray(error_a))
        _, _, accepted_b = step_size_controller(dt_j, jnp.asarray(error_b))
        if bool(accepted_b):
            assert bool(accepted_a), (
                f"error_a={error_a} rejected but error_b={error_b} accepted"
            )


class TestZeroErrorMaxGrowth:
    """When error_norm = 0, factor should be max_factor (maximum growth)."""

    @given(dt=dt_st)
    @settings(max_examples=100)
    def test_zero_error_max_factor(self, dt):
        dt_j = jnp.asarray(dt)
        _, factor, accepted = step_size_controller(dt_j, jnp.asarray(0.0))
        val = float(factor)
        assert val >= float(np.float32(MIN_FACTOR))
        assert val <= float(np.float32(MAX_FACTOR)) + 1e-7
        assert bool(accepted)
