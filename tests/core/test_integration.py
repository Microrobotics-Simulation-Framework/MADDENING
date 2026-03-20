"""Integration tests -- full workflows through the framework."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


# ------------------------------------------------------------------
# Full bouncing ball simulation
# ------------------------------------------------------------------

class TestBouncingBallIntegration:
    def test_ball_stays_above_table(self, bouncing_ball_graph):
        positions = []

        def record(i, state):
            positions.append(float(state["ball"]["position"]))

        bouncing_ball_graph.run(2000, callback=record)
        # Ball should never penetrate the table
        assert all(p >= -1e-6 for p in positions)

    def test_energy_dissipation(self, bouncing_ball_graph):
        """With elasticity < 1, the ball should lose energy over time."""
        positions = []

        def record(i, state):
            positions.append(float(state["ball"]["position"]))

        bouncing_ball_graph.run(1000, callback=record)
        # Max height should decrease over time
        first_half_max = max(positions[:500])
        second_half_max = max(positions[500:])
        assert second_half_max < first_half_max

    def test_ball_settles(self, bouncing_ball_graph):
        """Ball should settle near the table after enough steps."""
        bouncing_ball_graph.run(5000)
        state = bouncing_ball_graph.get_node_state("ball")
        # Position should be very close to table (0.0)
        assert float(state["position"]) < 0.5
        # Velocity should be very small
        assert abs(float(state["velocity"])) < 1.0


# ------------------------------------------------------------------
# JIT + Autodiff through graph step
# ------------------------------------------------------------------

class TestJITAndGrad:
    def test_jit_compiled_step_matches_uncompiled(self):
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_node(BallNode(name="b", timestep=0.01, initial_position=5.0))
        gm.add_edge("t", "b", "position", "table_position")
        gm.compile()

        # Get initial state and step
        state0 = dict(gm._state)
        ext = gm._default_external_inputs()
        state1 = gm._compiled_step(state0, ext)

        # Ball should have moved
        assert float(state1["b"]["position"]) < 5.0
        # Table should be unchanged
        assert float(state1["t"]["position"]) == 0.0

    def test_grad_through_compiled_step(self, bouncing_ball_graph):
        """Differentiate final position w.r.t. initial position."""
        ext = bouncing_ball_graph._default_external_inputs()

        def loss_fn(init_pos):
            state = {
                "table": {"position": jnp.array(0.0)},
                "ball": {"position": init_pos, "velocity": jnp.array(0.0)},
            }
            for _ in range(10):
                state = bouncing_ball_graph._compiled_step(state, ext)
            return state["ball"]["position"]

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(jnp.array(5.0))
        # Gradient should be finite and nonzero (position depends on init)
        assert jnp.isfinite(grad_val)
        assert float(grad_val) != 0.0

    def test_grad_through_multiple_steps(self, bouncing_ball_graph):
        """Gradient should still work with more steps."""
        ext = bouncing_ball_graph._default_external_inputs()

        def loss_fn(init_vel):
            state = {
                "table": {"position": jnp.array(0.0)},
                "ball": {"position": jnp.array(5.0), "velocity": init_vel},
            }
            for _ in range(5):
                state = bouncing_ball_graph._compiled_step(state, ext)
            return state["ball"]["position"]

        grad_fn = jax.grad(loss_fn)
        grad_val = grad_fn(jnp.array(0.0))
        assert jnp.isfinite(grad_val)


# ------------------------------------------------------------------
# Edge transform
# ------------------------------------------------------------------

class TestEdgeTransform:
    def test_edge_with_transform(self):
        """Edge transforms should be applied during graph step."""
        gm = GraphManager()
        gm.add_node(TableNode(name="t", timestep=0.01, position=5.0))
        gm.add_node(BallNode(name="b", timestep=0.01, initial_position=10.0))
        # Transform: negate the table position
        gm.add_edge("t", "b", "position", "table_position",
                     transform=lambda x: -x)
        gm.compile()

        # The ball should see table_position = -5.0 (negated)
        # so it shouldn't collide at position 10.0
        gm.step()
        state = gm.get_node_state("b")
        # Ball is at ~10 and table_position is -5, so no collision
        assert float(state["position"]) > 0.0


# ------------------------------------------------------------------
# Cycle handling (back-edge staggering)
# ------------------------------------------------------------------

class TestCycleStaggering:
    def test_mutual_dependency_runs(self):
        """Two nodes that depend on each other should still run (staggered)."""

        class EchoNode(SimulationNode):
            @property
            def requires_halo(self) -> bool:
                return False

            def initial_state(self):
                return {"val": jnp.array(1.0)}

            def update(self, state, boundary_inputs, dt):
                inp = boundary_inputs.get("other_val", jnp.array(0.0))
                return {"val": state["val"] + inp * dt}

        gm = GraphManager()
        gm.add_node(EchoNode(name="a", timestep=0.01))
        gm.add_node(EchoNode(name="b", timestep=0.01))
        gm.add_edge("a", "b", "val", "other_val")
        gm.add_edge("b", "a", "val", "other_val")

        # Should compile with a warning, not an error
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()

        # Should run without error
        gm.run(10)
        assert float(gm.get_node_state("a")["val"]) > 1.0
        assert float(gm.get_node_state("b")["val"]) > 1.0


# ------------------------------------------------------------------
# Multi-node graph
# ------------------------------------------------------------------

class TestMultiNodeGraph:
    def test_three_node_chain(self):
        """A -> B -> C chain should execute in order."""
        gm = GraphManager()
        gm.add_node(TableNode(name="ground", timestep=0.01, position=0.0))
        gm.add_node(TableNode(name="shelf", timestep=0.01, position=2.0))
        gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0))
        gm.add_edge("ground", "shelf", "position", "table_position")  # dummy edge for ordering
        gm.add_edge("shelf", "ball", "position", "table_position")    # ball bounces on shelf
        gm.compile()
        gm.run(500)
        state = gm.get_node_state("ball")
        # Ball should settle near the shelf (at 2.0)
        assert float(state["position"]) >= 1.5

    def test_disconnected_nodes_run(self):
        """Disconnected nodes should still update independently."""
        gm = GraphManager()
        gm.add_node(BallNode(name="b1", timestep=0.01, initial_position=5.0))
        gm.add_node(BallNode(name="b2", timestep=0.01, initial_position=10.0))

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()

        gm.run(10)
        # Both should have fallen (no table to bounce off)
        assert float(gm.get_node_state("b1")["position"]) < 5.0
        assert float(gm.get_node_state("b2")["position"]) < 10.0
