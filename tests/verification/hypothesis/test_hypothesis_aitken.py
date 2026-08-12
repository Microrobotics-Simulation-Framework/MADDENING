"""Property-based tests for Aitken delta-squared relaxation.

Complements the stelling suite by sampling from a much richer space,
including the degenerate regime (denom <= 1e-30) that interval
analysis cannot verify due to JAX's select_n tracing both branches.
"""


import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from maddening.core.coupling.acceleration import aitken_relaxation


float_arrays = arrays(
    dtype=np.float32,
    shape=(8,),
    elements=st.floats(min_value=-1e6, max_value=1e6,
                       allow_nan=False, allow_infinity=False),
)

small_arrays = arrays(
    dtype=np.float32,
    shape=(8,),
    elements=st.floats(min_value=-1e-16, max_value=1e-16,
                       allow_nan=False, allow_infinity=False),
)

OMEGA_LO = float(np.float32(0.01))
OMEGA_HI = float(np.float32(2.0))

omega_st = st.floats(min_value=OMEGA_LO, max_value=OMEGA_HI,
                     allow_nan=False, allow_infinity=False)


class TestAitkenOmegaAlwaysBounded:
    """omega output is always in [0.01, 2.0] regardless of inputs."""

    @given(x_old=float_arrays, x_raw=float_arrays,
           prev_r=float_arrays, omega=omega_st)
    @settings(max_examples=500)
    def test_omega_in_range(self, x_old, x_raw, prev_r, omega):
        x_old_j = jnp.asarray(x_old)
        x_raw_j = jnp.asarray(x_raw)
        prev_r_j = jnp.asarray(prev_r)
        omega_j = jnp.asarray(omega)

        _, new_omega, _ = aitken_relaxation(
            x_old_j, x_raw_j, prev_r_j, omega_j
        )
        val = float(new_omega)
        assert OMEGA_LO <= val <= OMEGA_HI, f"omega={val} out of [{OMEGA_LO}, {OMEGA_HI}]"

    @given(x_old=float_arrays, x_raw=float_arrays,
           prev_r=float_arrays, omega=omega_st)
    @settings(max_examples=500)
    def test_omega_is_finite(self, x_old, x_raw, prev_r, omega):
        x_old_j = jnp.asarray(x_old)
        x_raw_j = jnp.asarray(x_raw)
        prev_r_j = jnp.asarray(prev_r)
        omega_j = jnp.asarray(omega)

        _, new_omega, _ = aitken_relaxation(
            x_old_j, x_raw_j, prev_r_j, omega_j
        )
        assert jnp.isfinite(new_omega), "omega is not finite"


class TestAitkenDivisionGuard:
    """When residuals are near-zero, omega falls back to input value."""

    @given(x_old=small_arrays, x_raw=small_arrays,
           prev_r=small_arrays, omega=omega_st)
    @settings(max_examples=500)
    def test_degenerate_returns_input_omega(self, x_old, x_raw, prev_r, omega):
        x_old_j = jnp.asarray(x_old)
        x_raw_j = jnp.asarray(x_raw)
        prev_r_j = jnp.asarray(prev_r)
        omega_j = jnp.asarray(omega)

        residual = x_raw_j - x_old_j
        delta_r = residual - prev_r_j
        denom = float(jnp.sum(delta_r ** 2))
        assume(denom <= 1e-30)

        _, new_omega, _ = aitken_relaxation(
            x_old_j, x_raw_j, prev_r_j, omega_j
        )
        assert float(new_omega) == float(omega_j), (
            f"Expected omega={float(omega_j)}, got {float(new_omega)} "
            f"(denom={denom})"
        )


class TestAitkenRelaxedState:
    """x_relaxed = x_old + new_omega * (x_raw - x_old)."""

    @given(x_old=float_arrays, x_raw=float_arrays,
           prev_r=float_arrays, omega=omega_st)
    @settings(max_examples=500)
    def test_affine_combination(self, x_old, x_raw, prev_r, omega):
        x_old_j = jnp.asarray(x_old)
        x_raw_j = jnp.asarray(x_raw)
        prev_r_j = jnp.asarray(prev_r)
        omega_j = jnp.asarray(omega)

        x_relaxed, new_omega, _ = aitken_relaxation(
            x_old_j, x_raw_j, prev_r_j, omega_j
        )
        expected = x_old_j + new_omega * (x_raw_j - x_old_j)
        assert jnp.allclose(x_relaxed, expected, atol=1e-5), (
            f"Relaxed state is not x_old + omega * residual"
        )


class TestAitkenOverflowRobustness:
    """Aitken never produces NaN/Inf regardless of input magnitude."""

    @given(
        scale=st.floats(min_value=1e10, max_value=1e38,
                        allow_nan=False, allow_infinity=False),
        omega=omega_st,
    )
    @settings(max_examples=200)
    def test_large_residuals_stay_finite(self, scale, omega):
        x_old = jnp.zeros(4, dtype=jnp.float32)
        x_raw = jnp.array([scale, 0, 0, 0], dtype=jnp.float32)
        prev_r = jnp.array([-scale, 0, 0, 0], dtype=jnp.float32)
        omega_j = jnp.asarray(np.float32(omega))

        x_relaxed, new_omega, _ = aitken_relaxation(
            x_old, x_raw, prev_r, omega_j
        )
        assert jnp.isfinite(new_omega), (
            f"NaN/Inf omega at scale={scale}"
        )
        assert jnp.all(jnp.isfinite(x_relaxed)), (
            f"NaN/Inf in relaxed state at scale={scale}"
        )
