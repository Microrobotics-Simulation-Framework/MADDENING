"""Tests for GraphManager.run_sweep -- parameter sweeps via jax.vmap."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_ball_graph(initial_position=5.0, initial_velocity=0.0,
                     elasticity=0.8, table_position=0.0):
    """Build and compile a single bouncing-ball graph."""
    gm = GraphManager()
    ball = BallNode(
        name="ball", timestep=0.01,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
        elasticity=elasticity,
    )
    table = TableNode(name="table", timestep=0.01, position=table_position)
    gm.add_node(table)
    gm.add_node(ball)
    gm.add_edge(source="table", target="ball",
                source_field="position", target_field="table_position")
    gm.compile()
    return gm


def _make_ball_only_graph(initial_position=5.0, initial_velocity=0.0):
    """Build and compile a ball-only graph (no table, no collision)."""
    gm = GraphManager()
    ball = BallNode(
        name="ball", timestep=0.01,
        initial_position=initial_position,
        initial_velocity=initial_velocity,
    )
    gm.add_node(ball)
    gm.compile()
    return gm


def _batched_ball_only_states(positions):
    """Build batched initial states for ball-only graph."""
    batch = len(positions)
    return {
        "ball": {
            "position": jnp.array(positions, dtype=jnp.float32),
            "velocity": jnp.zeros(batch, dtype=jnp.float32),
        },
    }


def _batched_ball_table_states(positions, table_pos=0.0):
    """Build batched initial states for ball+table graph."""
    batch = len(positions)
    return {
        "table": {
            "position": jnp.full(batch, table_pos, dtype=jnp.float32),
        },
        "ball": {
            "position": jnp.array(positions, dtype=jnp.float32),
            "velocity": jnp.zeros(batch, dtype=jnp.float32),
        },
    }


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestRunSweep:
    """Tests for the ``run_sweep`` parameter-sweep feature."""

    def test_sweep_basic(self):
        """Sweep over 5 initial ball positions. Output shapes have batch dim."""
        gm = _make_ball_only_graph()
        positions = [1.0, 2.0, 3.0, 4.0, 5.0]
        batch_states = _batched_ball_only_states(positions)

        result = gm.run_sweep(n_steps=100, initial_states=batch_states)

        # result should be a dict with same structure, batch dim on each leaf
        assert result["ball"]["position"].shape == (5,)
        assert result["ball"]["velocity"].shape == (5,)

    def test_sweep_matches_serial(self):
        """Sweep results must match independent serial simulations exactly."""
        positions = [1.0, 3.0, 5.0, 7.0, 9.0]
        n_steps = 200

        # --- serial runs ---
        serial_positions = []
        serial_velocities = []
        for pos in positions:
            gm = _make_ball_only_graph(initial_position=pos)
            final = gm.run_scan(n_steps)
            serial_positions.append(float(final["ball"]["position"]))
            serial_velocities.append(float(final["ball"]["velocity"]))

        # --- batched sweep ---
        gm = _make_ball_only_graph()
        batch_states = _batched_ball_only_states(positions)
        result = gm.run_sweep(n_steps=n_steps, initial_states=batch_states)

        sweep_positions = result["ball"]["position"]
        sweep_velocities = result["ball"]["velocity"]

        for i, pos in enumerate(positions):
            assert jnp.allclose(sweep_positions[i], serial_positions[i], atol=1e-5), \
                f"Position mismatch for initial_position={pos}"
            assert jnp.allclose(sweep_velocities[i], serial_velocities[i], atol=1e-5), \
                f"Velocity mismatch for initial_position={pos}"

    def test_sweep_with_history(self):
        """return_history=True gives shapes (batch, n_steps, ...)."""
        gm = _make_ball_only_graph()
        batch_size = 4
        n_steps = 50
        positions = [float(i + 1) for i in range(batch_size)]
        batch_states = _batched_ball_only_states(positions)

        finals, histories = gm.run_sweep(
            n_steps=n_steps,
            initial_states=batch_states,
            return_history=True,
        )

        # Final states: (batch,) on each leaf
        assert finals["ball"]["position"].shape == (batch_size,)
        assert finals["ball"]["velocity"].shape == (batch_size,)

        # Histories: (batch, n_steps) on each leaf
        assert histories["ball"]["position"].shape == (batch_size, n_steps)
        assert histories["ball"]["velocity"].shape == (batch_size, n_steps)

        # The last history entry should match the final state
        jnp.allclose(histories["ball"]["position"][:, -1],
                      finals["ball"]["position"], atol=1e-6)
        jnp.allclose(histories["ball"]["velocity"][:, -1],
                      finals["ball"]["velocity"], atol=1e-6)

    def test_sweep_single_batch(self):
        """batch_size=1 should work without issues."""
        gm = _make_ball_only_graph()
        batch_states = _batched_ball_only_states([5.0])

        result = gm.run_sweep(n_steps=100, initial_states=batch_states)

        assert result["ball"]["position"].shape == (1,)
        assert result["ball"]["velocity"].shape == (1,)

        # Should match a single serial run
        gm2 = _make_ball_only_graph(initial_position=5.0)
        serial_final = gm2.run_scan(100)

        assert jnp.allclose(result["ball"]["position"][0],
                            serial_final["ball"]["position"], atol=1e-5)

    def test_sweep_multi_node(self):
        """Sweep with a multi-node graph (ball + table with edge)."""
        gm = _make_ball_graph()
        positions = [2.0, 4.0, 6.0]
        batch_states = _batched_ball_table_states(positions)

        result = gm.run_sweep(n_steps=200, initial_states=batch_states)

        # Both nodes should have batch dimension in output
        assert result["ball"]["position"].shape == (3,)
        assert result["ball"]["velocity"].shape == (3,)
        assert result["table"]["position"].shape == (3,)

        # Ball should never go below the table (position >= 0)
        # After 200 steps, all balls should still be >= 0
        assert jnp.all(result["ball"]["position"] >= -1e-5), \
            "Ball fell through the table in sweep"

        # Higher initial position => higher or equal final position (more energy)
        # Not necessarily monotone due to bouncing dynamics, but all should be >= 0.

    def test_sweep_gradient(self):
        """jax.grad through run_sweep: differentiate final position w.r.t. initial velocity."""
        gm = _make_ball_only_graph()
        n_steps = 50

        def loss(velocities):
            """Sum of final positions given different initial velocities."""
            batch_states = {
                "ball": {
                    "position": jnp.ones(velocities.shape, dtype=jnp.float32) * 5.0,
                    "velocity": velocities,
                },
            }
            result = gm.run_sweep(n_steps=n_steps, initial_states=batch_states)
            return jnp.sum(result["ball"]["position"])

        velocities = jnp.array([0.0, 1.0, -1.0], dtype=jnp.float32)
        grad_fn = jax.grad(loss)
        grads = grad_fn(velocities)

        # Gradient should exist and be finite
        assert grads.shape == (3,)
        assert jnp.all(jnp.isfinite(grads)), "Gradients should be finite"

        # With no collision (ball at height 5, only 50 steps at dt=0.01 =
        # 0.5s), positive initial velocity should lead to higher final
        # position, so gradient w.r.t. velocity should be positive.
        # For free-fall: position = p0 + v0*t + 0.5*g*t^2
        # d(position)/d(v0) = t = 0.5
        # All three should have the same gradient since there's no collision
        # in 0.5s from height 5.
        assert jnp.allclose(grads, grads[0], atol=1e-4), \
            "Gradients should be identical for different velocities (no collision)"

    def test_sweep_large_batch(self):
        """Batch of 50 runs -- verify correctness (and implicitly, performance)."""
        gm = _make_ball_only_graph()
        batch_size = 50
        n_steps = 100

        positions = [float(i * 0.5 + 1.0) for i in range(batch_size)]
        batch_states = _batched_ball_only_states(positions)

        result = gm.run_sweep(n_steps=n_steps, initial_states=batch_states)

        assert result["ball"]["position"].shape == (batch_size,)
        assert result["ball"]["velocity"].shape == (batch_size,)

        # All results should be finite
        assert jnp.all(jnp.isfinite(result["ball"]["position"]))
        assert jnp.all(jnp.isfinite(result["ball"]["velocity"]))

        # Verify a few spot checks against serial runs
        for idx in [0, 24, 49]:
            gm_serial = _make_ball_only_graph(initial_position=positions[idx])
            serial_final = gm_serial.run_scan(n_steps)
            assert jnp.allclose(result["ball"]["position"][idx],
                                serial_final["ball"]["position"], atol=1e-5), \
                f"Mismatch at batch index {idx}"

    def test_sweep_with_external_inputs(self):
        """External inputs should be applied identically to all batch elements."""
        # Build a graph with an external input
        gm = GraphManager()
        ball = BallNode(
            name="ball", timestep=0.01,
            initial_position=5.0, initial_velocity=0.0,
        )
        table = TableNode(name="table", timestep=0.01, position=0.0)
        gm.add_node(table)
        gm.add_node(ball)
        gm.add_edge(source="table", target="ball",
                    source_field="position", target_field="table_position")
        gm.add_external_input("ball", "force", shape=(), dtype=jnp.float32)
        gm.compile()

        batch_size = 3
        positions = [2.0, 4.0, 6.0]
        batch_states = _batched_ball_table_states(positions)

        # Provide external inputs (not batched -- same for all)
        ext = {"ball": {"force": jnp.array(0.0, dtype=jnp.float32)}}

        result = gm.run_sweep(
            n_steps=100,
            initial_states=batch_states,
            external_inputs=ext,
        )

        assert result["ball"]["position"].shape == (batch_size,)
        assert jnp.all(jnp.isfinite(result["ball"]["position"]))

        # Compare to serial: with zero external force, should match graph
        # without the external input
        gm_ref = _make_ball_graph()
        for i, pos in enumerate(positions):
            ref_states = {
                "table": {"position": jnp.array(0.0, dtype=jnp.float32)},
                "ball": {
                    "position": jnp.array(pos, dtype=jnp.float32),
                    "velocity": jnp.array(0.0, dtype=jnp.float32),
                },
            }
            gm_ref._state = ref_states
            ref_final = gm_ref.run_scan(100)
            assert jnp.allclose(result["ball"]["position"][i],
                                ref_final["ball"]["position"], atol=1e-5), \
                f"External input mismatch at batch index {i}"
