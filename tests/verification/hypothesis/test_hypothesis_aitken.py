"""Property-based tests for Aitken delta-squared relaxation.

Samples from a rich input space, including the degenerate regime (a
residual that did not change, so the denominator is exactly zero) that
interval analysis cannot verify due to JAX's select_n tracing both
branches, and the same inputs in other units: the guard used to be an
absolute ``denom > 1e-30``, which froze omega for any group written in
small enough units (MEDIUM-2 of the 0.4.0 round-3 audit), and the
dot products are now taken on the vectors rescaled by an exact power of
two.
"""


import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from maddening.core.coupling.acceleration import aitken_relaxation
from tests.conftest import EXAMPLES_CHEAP


float_arrays = arrays(
    dtype=np.float32,
    shape=(8,),
    elements=st.floats(min_value=-1e6, max_value=1e6,
                       allow_nan=False, allow_infinity=False),
)

#: Multiples of 1/1024 up to 1024: exact in float32, and so is every
#: difference of two of them, which never comes near the subnormal range --
#: so the same arrays times any power of two from 2**-100 to 2**100 are the
#: same computation in other units, bit for bit, wherever the arithmetic is
#: units-invariant.
grid_arrays = arrays(
    dtype=np.float32,
    shape=(8,),
    elements=st.integers(-2 ** 20, 2 ** 20).map(lambda i: i / 1024.0),
)

OMEGA_LO = float(np.float32(0.01))
OMEGA_HI = float(np.float32(2.0))

omega_st = st.floats(min_value=OMEGA_LO, max_value=OMEGA_HI,
                     allow_nan=False, allow_infinity=False)


class TestAitkenOmegaAlwaysBounded:
    """omega output is always in [0.01, 2.0] regardless of inputs."""

    @given(x_old=float_arrays, x_raw=float_arrays,
           prev_r=float_arrays, omega=omega_st)
    @settings(max_examples=EXAMPLES_CHEAP)
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
    @settings(max_examples=EXAMPLES_CHEAP)
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
    """omega falls back to its input exactly where the formula is degenerate.

    Degenerate means the residual did not change between the two passes
    (``delta_r == 0``, a denominator of exactly zero; the zero first-pass
    sentinel is the other fallback).  Small *values* are not degenerate:
    the same residual sequence in units ``2**-100`` times smaller is the
    same Aitken step, and the guard on an absolute ``denom > 1e-30`` that
    froze it is gone.
    """

    @given(x_old=float_arrays, x_raw=float_arrays, omega=omega_st)
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_an_unchanged_residual_returns_input_omega(self, x_old, x_raw, omega):
        # The previous residual *is* this one: ``x_raw - x_old`` rounds the
        # same in NumPy and in XLA (one IEEE subtraction each).
        prev_r = (x_raw - x_old).astype(np.float32)
        _, new_omega, _ = aitken_relaxation(
            jnp.asarray(x_old), jnp.asarray(x_raw), jnp.asarray(prev_r), jnp.asarray(omega)
        )
        assert float(new_omega) == float(np.float32(omega)), (
            f"Expected omega={omega}, got {float(new_omega)} for an unchanged residual"
        )

    @given(x_old=grid_arrays, x_raw=grid_arrays, prev_r=grid_arrays,
           omega=omega_st, k=st.integers(-100, 100))
    @settings(max_examples=EXAMPLES_CHEAP)
    def test_omega_does_not_depend_on_the_units(self, x_old, x_raw, prev_r, omega, k):
        """The same inputs times ``2**k``: the same omega, the step times ``2**k``.

        Failed before the fix for any ``k`` that put ``sum(delta_r**2)``
        below 1e-30 (omega frozen at its input) or its squares outside
        float32 (flushed to zero, or overflowing to inf).
        """
        s = np.float32(2.0 ** k)
        x1, w1, _ = aitken_relaxation(jnp.asarray(x_old), jnp.asarray(x_raw),
                                      jnp.asarray(prev_r), jnp.asarray(omega))
        xs, ws, _ = aitken_relaxation(jnp.asarray(x_old * s), jnp.asarray(x_raw * s),
                                      jnp.asarray(prev_r * s), jnp.asarray(omega))
        assert float(ws) == float(w1), (k, float(w1), float(ws))
        np.testing.assert_array_equal(np.asarray(xs) / s, np.asarray(x1))


class TestAitkenRelaxedState:
    """x_relaxed = x_old + new_omega * (x_raw - x_old)."""

    @given(x_old=float_arrays, x_raw=float_arrays,
           prev_r=float_arrays, omega=omega_st)
    @settings(max_examples=EXAMPLES_CHEAP)
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
    @settings(max_examples=EXAMPLES_CHEAP)
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
