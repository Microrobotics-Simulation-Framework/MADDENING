"""Property-based tests for built-in MADDENING nodes.

Uses the extensible strategies from maddening.testing.strategies
to verify universal properties of all built-in nodes.
"""


import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from maddening.testing.strategies import node_states, bounded_dt
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode


class TestBallNodeProperties:
    """Properties of the bouncing ball node."""

    def setup_method(self):
        self.node = BallNode(
            name="ball", timestep=0.01,
            initial_position=5.0, initial_velocity=0.0,
            elasticity=0.7,
        )

    @given(
        state=node_states(
            BallNode(name="b", timestep=0.01,
                     initial_position=5.0, initial_velocity=0.0,
                     elasticity=0.7),
            bounds={"position": (-100.0, 100.0),
                    "velocity": (-50.0, 50.0)},
        ),
        dt=bounded_dt(1e-4, 0.05),
    )
    @settings(max_examples=500)
    def test_update_returns_finite(self, state, dt):
        out = self.node.update(state, {"table_position": jnp.array(0.0)}, dt)
        for field, val in out.items():
            assert jnp.all(jnp.isfinite(val)), (
                f"BallNode.update produced non-finite in '{field}'"
            )

    @given(
        state=node_states(
            BallNode(name="b", timestep=0.01,
                     initial_position=5.0, initial_velocity=0.0,
                     elasticity=0.7),
            bounds={"position": (-100.0, 100.0),
                    "velocity": (-50.0, 50.0)},
        ),
        dt=bounded_dt(1e-4, 0.05),
    )
    @settings(max_examples=500)
    def test_update_preserves_structure(self, state, dt):
        out = self.node.update(state, {"table_position": jnp.array(0.0)}, dt)
        assert set(out.keys()) == set(state.keys()), (
            f"Key mismatch: {set(out.keys())} != {set(state.keys())}"
        )
        for field in state:
            assert out[field].shape == state[field].shape, (
                f"Shape mismatch in '{field}': "
                f"{out[field].shape} != {state[field].shape}"
            )

    @given(
        state=node_states(
            BallNode(name="b", timestep=0.01,
                     initial_position=5.0, initial_velocity=0.0,
                     elasticity=0.7),
            bounds={"position": (-100.0, 100.0),
                    "velocity": (-50.0, 50.0)},
        ),
        dt=bounded_dt(1e-4, 0.05),
    )
    @settings(max_examples=200)
    def test_update_deterministic(self, state, dt):
        bi = {"table_position": jnp.array(0.0)}
        out1 = self.node.update(state, bi, dt)
        out2 = self.node.update(state, bi, dt)
        for field in out1:
            assert jnp.array_equal(out1[field], out2[field]), (
                f"Non-deterministic in '{field}'"
            )


class TestSpringDamperNodeProperties:
    """Properties of the spring-damper node."""

    def setup_method(self):
        self.node = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=10.0, damping=1.0, rest_length=1.0,
        )

    @given(
        state=node_states(
            SpringDamperNode(name="s", timestep=0.01,
                             stiffness=10.0, damping=1.0, rest_length=1.0),
            bounds={"position": (-5.0, 5.0),
                    "velocity": (-20.0, 20.0)},
        ),
        dt=bounded_dt(1e-4, 0.05),
    )
    @settings(max_examples=500)
    def test_update_returns_finite(self, state, dt):
        out = self.node.update(state, {}, dt)
        for field, val in out.items():
            assert jnp.all(jnp.isfinite(val)), (
                f"SpringDamperNode.update produced non-finite in '{field}'"
            )

    @given(
        state=node_states(
            SpringDamperNode(name="s", timestep=0.01,
                             stiffness=10.0, damping=1.0, rest_length=1.0),
            bounds={"position": (-5.0, 5.0),
                    "velocity": (-20.0, 20.0)},
        ),
        dt=bounded_dt(1e-4, 0.05),
    )
    @settings(max_examples=500)
    def test_update_preserves_structure(self, state, dt):
        out = self.node.update(state, {}, dt)
        assert set(out.keys()) == set(state.keys())
        for field in state:
            assert out[field].shape == state[field].shape
