"""Property-based tests for the SpringDamperNode.

Verifies energy dissipation, mass validation, and finite output.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from maddening.nodes.spring import SpringDamperNode
from maddening.testing.strategies import node_states, bounded_dt


class TestSpringMassValidation:
    """mass <= 0 should be rejected at construction time."""

    def test_zero_mass_rejected(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            SpringDamperNode(name="s", timestep=0.01, mass=0.0)

    def test_negative_mass_rejected(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            SpringDamperNode(name="s", timestep=0.01, mass=-1.0)


class TestSpringEnergyDissipation:
    """With damping > 0, total energy should not increase."""

    @given(
        pos=st.floats(min_value=-5.0, max_value=5.0,
                      allow_nan=False, allow_infinity=False),
        vel=st.floats(min_value=-10.0, max_value=10.0,
                      allow_nan=False, allow_infinity=False),
        dt=st.floats(min_value=1e-4, max_value=0.005,
                     allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=500)
    def test_energy_non_increasing(self, pos, vel, dt):
        k = 10.0
        m = 1.0
        c = 2.0
        rest = 1.0
        node = SpringDamperNode(
            name="s", timestep=0.01,
            stiffness=k, damping=c, mass=m, rest_length=rest,
        )
        state = {
            "position": jnp.array(pos, dtype=jnp.float32),
            "velocity": jnp.array(vel, dtype=jnp.float32),
        }

        def energy(s):
            x = float(s["position"])
            v = float(s["velocity"])
            return 0.5 * k * (x - rest) ** 2 + 0.5 * m * v ** 2

        e_before = energy(state)
        out = node.update(state, {}, dt)
        e_after = energy(out)

        # Semi-implicit Euler can overshoot slightly for large dt*omega
        # but with small dt and moderate damping, energy should decrease
        # Allow small tolerance for numerical overshoot
        assert e_after <= e_before + 1e-3 * abs(e_before) + 1e-6, (
            f"Energy increased: {e_before} → {e_after} "
            f"(pos={pos}, vel={vel}, dt={dt})"
        )


class TestSpringFiniteOutput:
    """Output should be finite for bounded inputs with stable dt."""

    @given(
        state=node_states(
            SpringDamperNode(name="s", timestep=0.01,
                             stiffness=100.0, damping=1.0),
            bounds={"position": (-10.0, 10.0),
                    "velocity": (-50.0, 50.0)},
        ),
        dt=bounded_dt(1e-5, 0.005),
    )
    @settings(max_examples=500)
    def test_always_finite(self, state, dt):
        node = SpringDamperNode(
            name="s", timestep=0.01, stiffness=100.0, damping=1.0,
        )
        out = node.update(state, {}, dt)
        for field, val in out.items():
            assert jnp.all(jnp.isfinite(val)), (
                f"Non-finite in '{field}' with dt={dt}"
            )
