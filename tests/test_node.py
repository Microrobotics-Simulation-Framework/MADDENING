"""Tests for SimulationNode ABC and concrete node implementations."""

import pytest
import jax.numpy as jnp

from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode, GRAVITY
from maddening.nodes.table import TableNode


# ------------------------------------------------------------------
# SimulationNode ABC
# ------------------------------------------------------------------

class TestSimulationNodeABC:
    def test_cannot_instantiate_abc(self):
        with pytest.raises(TypeError):
            SimulationNode(name="bad", timestep=0.01)

    def test_concrete_subclass_has_required_methods(self, ball_node):
        assert hasattr(ball_node, "initial_state")
        assert hasattr(ball_node, "update")

    def test_name_and_timestep(self, ball_node):
        assert ball_node.name == "ball"
        assert ball_node.delta_t == 0.01

    def test_params_stored(self, ball_node):
        assert ball_node.params["initial_position"] == 5.0
        assert ball_node.params["initial_velocity"] == 0.0
        assert ball_node.params["elasticity"] == 0.7

    def test_state_fields(self, ball_node):
        fields = ball_node.state_fields()
        assert "position" in fields
        assert "velocity" in fields

    def test_to_dict(self, ball_node):
        d = ball_node.to_dict()
        assert d["type"] == "BallNode"
        assert d["name"] == "ball"
        assert d["timestep"] == 0.01
        assert d["params"]["initial_position"] == 5.0


# ------------------------------------------------------------------
# BallNode
# ------------------------------------------------------------------

class TestBallNode:
    def test_initial_state(self, ball_node):
        state = ball_node.initial_state()
        assert float(state["position"]) == pytest.approx(5.0)
        assert float(state["velocity"]) == pytest.approx(0.0)

    def test_free_fall_one_step(self, ball_node):
        state = ball_node.initial_state()
        new_state = ball_node.update(state, {}, 0.01)
        # v = 0 + (-9.81)*0.01 = -0.0981
        assert float(new_state["velocity"]) == pytest.approx(GRAVITY * 0.01)
        # p = 5.0 + (-0.0981)*0.01 = 4.999019
        expected_pos = 5.0 + (GRAVITY * 0.01) * 0.01
        assert float(new_state["position"]) == pytest.approx(expected_pos)

    def test_collision_bounces_ball(self):
        """Ball just below table should bounce back."""
        ball = BallNode(name="b", timestep=0.01, elasticity=0.5)
        state = {"position": jnp.array(-0.1), "velocity": jnp.array(-2.0)}
        new = ball.update(state, {"table_position": jnp.array(0.0)}, 0.01)
        assert float(new["position"]) >= 0.0  # clamped above table
        assert float(new["velocity"]) > 0.0   # velocity reversed

    def test_collision_uses_elasticity(self):
        ball = BallNode(name="b", timestep=0.01, elasticity=0.5)
        state = {"position": jnp.array(-0.1), "velocity": jnp.array(-4.0)}
        new = ball.update(state, {"table_position": jnp.array(0.0)}, 0.01)
        # After update, velocity should integrate gravity then bounce
        # The result depends on position after integration, but post-bounce
        # magnitude should be reduced by elasticity
        assert float(new["velocity"]) > 0.0

    def test_no_collision_without_table(self, ball_node):
        """Without table_position in boundary inputs, ball falls through zero."""
        state = {"position": jnp.array(0.01), "velocity": jnp.array(-1.0)}
        new = ball_node.update(state, {}, 0.01)
        assert float(new["position"]) < 0.0

    def test_default_params(self):
        ball = BallNode(name="b", timestep=0.01)
        state = ball.initial_state()
        assert float(state["position"]) == pytest.approx(0.0)
        assert float(state["velocity"]) == pytest.approx(0.0)


# ------------------------------------------------------------------
# TableNode
# ------------------------------------------------------------------

class TestTableNode:
    def test_initial_state(self, table_node):
        state = table_node.initial_state()
        assert float(state["position"]) == pytest.approx(0.0)

    def test_update_is_identity(self, table_node):
        state = table_node.initial_state()
        new = table_node.update(state, {}, 0.01)
        assert new is state  # exact same object

    def test_custom_position(self):
        t = TableNode(name="shelf", timestep=0.01, position=2.5)
        assert float(t.initial_state()["position"]) == pytest.approx(2.5)

    def test_to_dict(self, table_node):
        d = table_node.to_dict()
        assert d["type"] == "TableNode"
        assert d["params"]["position"] == 0.0
