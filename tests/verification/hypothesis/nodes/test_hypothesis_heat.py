"""Property-based tests for the HeatNode.

Verifies conservation, stability, and boundary-condition properties.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from maddening.nodes.heat import HeatNode


class TestHeatConservation:
    """Total heat should be conserved with Neumann (zero-flux) BCs."""

    @given(
        n_cells=st.integers(min_value=10, max_value=50),
        diffusivity=st.floats(min_value=0.001, max_value=0.1,
                              allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_zero_flux_conserves_total_heat(self, n_cells, diffusivity):
        node = HeatNode(
            name="h", timestep=0.01, n_cells=n_cells,
            thermal_diffusivity=diffusivity,
            initial_temperature=0.0,
        )
        # CFL-safe dt: dt < dx^2 / (2 * alpha)
        dx = 1.0 / n_cells
        dt_safe = 0.4 * dx ** 2 / (2.0 * diffusivity)

        # Random initial temperature profile
        rng = np.random.default_rng(42)
        T_init = rng.uniform(10.0, 100.0, n_cells).astype(np.float32)
        state = {"temperature": jnp.asarray(T_init)}

        # Step without boundary inputs (Neumann BCs by default)
        out = node.update(state, {}, dt_safe)
        total_before = float(jnp.sum(state["temperature"]))
        total_after = float(jnp.sum(out["temperature"]))

        # Ghost-cell extrapolation at boundaries introduces O(dt*alpha/dx)
        # leakage per step. For a single step this is bounded but non-zero.
        rel_err = abs(total_after - total_before) / max(abs(total_before), 1e-10)
        assert rel_err < 0.05, (
            f"Heat not conserved: before={total_before}, after={total_after}, "
            f"rel_err={rel_err}"
        )


class TestHeatCFLInstability:
    """Document: explicit heat WILL go negative above CFL limit."""

    def test_above_cfl_produces_negative_temperature(self):
        """This is a KNOWN LIMITATION, not a bug. Explicit methods
        require dt <= dx^2 / (2*alpha) for stability."""
        node = HeatNode(
            name="h", timestep=0.01, n_cells=8,
            thermal_diffusivity=1.0,
        )
        # CFL limit: dx=0.125, alpha=1.0 → dt_cfl = 0.0078
        # Use dt well above CFL with a spike
        state = {"temperature": jnp.array(
            [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            dtype=jnp.float32,
        )}
        out = node.update(state, {}, 0.05)
        assert bool(jnp.any(out["temperature"] < 0)), (
            "Expected negative temperature above CFL (known limitation)"
        )


class TestHeatFiniteOutput:
    """Output should be finite for CFL-safe dt."""

    @given(
        n_cells=st.integers(min_value=5, max_value=30),
        diffusivity=st.floats(min_value=0.001, max_value=0.1,
                              allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None)
    def test_cfl_safe_produces_finite(self, n_cells, diffusivity):
        node = HeatNode(
            name="h", timestep=0.01, n_cells=n_cells,
            thermal_diffusivity=diffusivity,
        )
        dx = 1.0 / n_cells
        dt_safe = 0.3 * dx ** 2 / (2.0 * diffusivity)

        rng = np.random.default_rng(123)
        T_init = rng.uniform(0.0, 500.0, n_cells).astype(np.float32)
        state = {"temperature": jnp.asarray(T_init)}

        out = node.update(state, {}, dt_safe)
        assert jnp.all(jnp.isfinite(out["temperature"])), (
            f"Non-finite temperature with CFL-safe dt={dt_safe}"
        )
