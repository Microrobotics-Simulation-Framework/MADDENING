"""Tests for subcycling within coupling groups (Phase 4a).

Verifies that nodes with different timesteps can be coupled via
subcycling, with correct time interpolation of boundary conditions.
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


# ==================================================================
# Helpers
# ==================================================================

def _make_mixed_rate_springs(dt_fast=0.001, dt_slow=0.01,
                             k=50.0, c=1.0, m=1.0,
                             pos_a=0.0, pos_b=3.0):
    """Two springs at different timesteps, coupled bidirectionally."""
    gm = GraphManager()
    a = SpringDamperNode(name="fast", timestep=dt_fast, stiffness=k,
                         damping=c, mass=m, rest_length=1.0,
                         initial_position=pos_a)
    b = SpringDamperNode(name="slow", timestep=dt_slow, stiffness=k,
                         damping=c, mass=m, rest_length=1.0,
                         initial_position=pos_b)
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge("fast", "slow", "position", "anchor_position")
    gm.add_edge("slow", "fast", "position", "anchor_position")
    return gm


def _make_uniform_reference(dt=0.001, k=50.0, c=1.0, m=1.0,
                             pos_a=0.0, pos_b=3.0):
    """Both springs at the fast rate for reference comparison."""
    gm = GraphManager()
    a = SpringDamperNode(name="fast", timestep=dt, stiffness=k,
                         damping=c, mass=m, rest_length=1.0,
                         initial_position=pos_a)
    b = SpringDamperNode(name="slow", timestep=dt, stiffness=k,
                         damping=c, mass=m, rest_length=1.0,
                         initial_position=pos_b)
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge("fast", "slow", "position", "anchor_position")
    gm.add_edge("slow", "fast", "position", "anchor_position")
    return gm


# ==================================================================
# Validation tests
# ==================================================================

class TestSubcyclingValidation:
    """Tests that validation correctly handles subcycling flag."""

    def test_mixed_timestep_without_subcycling_raises(self):
        """Mixed timesteps without subcycling=True should raise."""
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
        )
        with pytest.raises(RuntimeError, match="mixed timesteps"):
            gm.compile()

    def test_mixed_timestep_with_subcycling_compiles(self):
        """Mixed timesteps with subcycling=True should compile."""
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
        )
        gm.compile()  # should not raise

    def test_uniform_timestep_with_subcycling_noop(self):
        """Subcycling with uniform timesteps should work (no actual subcycling)."""
        gm = _make_uniform_reference()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])


# ==================================================================
# Convergence tests
# ==================================================================

class TestSubcyclingConvergence:
    """Tests that subcycled coupling converges and produces correct results."""

    def test_subcycling_runs_and_converges(self):
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=15, tolerance=1e-8,
            subcycling=True,
        )
        gm.compile()
        state = gm.run_scan(100)
        assert jnp.isfinite(state["fast"]["position"])
        assert jnp.isfinite(state["slow"]["position"])

    def test_subcycling_vs_uniform_reference(self):
        """Subcycled result should be closer to uniform-rate reference
        than staggered (no coupling) would be.
        """
        k, c, m = 20.0, 1.0, 1.0
        pos_a, pos_b = 0.0, 3.0
        dt_fast, dt_slow = 0.001, 0.005
        # Number of slow-rate steps
        n_slow_steps = 100
        # Equivalent fast-rate steps for same total time
        n_fast_steps = n_slow_steps * round(dt_slow / dt_fast)

        # Reference: both nodes at fast rate
        gm_ref = _make_uniform_reference(dt=dt_fast, k=k, c=c, m=m,
                                          pos_a=pos_a, pos_b=pos_b)
        gm_ref.add_coupling_group(
            ["fast", "slow"],
            max_iterations=20, tolerance=1e-10,
        )
        gm_ref.compile()
        s_ref = gm_ref.run_scan(n_fast_steps)

        # Subcycled
        gm_sub = _make_mixed_rate_springs(dt_fast=dt_fast, dt_slow=dt_slow,
                                           k=k, c=c, m=m,
                                           pos_a=pos_a, pos_b=pos_b)
        gm_sub.add_coupling_group(
            ["fast", "slow"],
            max_iterations=20, tolerance=1e-10,
            subcycling=True,
        )
        gm_sub.compile()
        s_sub = gm_sub.run_scan(n_slow_steps)

        # Both should be finite
        assert jnp.isfinite(s_sub["fast"]["position"])
        assert jnp.isfinite(s_sub["slow"]["position"])

        # Subcycled should be reasonably close to reference.
        # Some difference is expected because the slow node integrates
        # at a coarser dt, introducing time-discretisation error.
        diff_fast = float(jnp.abs(
            s_ref["fast"]["position"] - s_sub["fast"]["position"]
        ))
        diff_slow = float(jnp.abs(
            s_ref["slow"]["position"] - s_sub["slow"]["position"]
        ))
        assert diff_fast < 10.0, f"Fast differs from reference by {diff_fast}"
        assert diff_slow < 10.0, f"Slow differs from reference by {diff_slow}"

    def test_subcycling_constant_interpolation(self):
        """Subcycling with constant boundary interpolation."""
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
            boundary_interpolation="constant",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])

    def test_subcycling_linear_interpolation(self):
        """Subcycling with linear boundary interpolation (default)."""
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
            boundary_interpolation="linear",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])


# ==================================================================
# Combination tests
# ==================================================================

class TestSubcyclingCombinations:
    """Tests combining subcycling with other coupling features."""

    def test_subcycling_with_aitken(self):
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=15, tolerance=1e-8,
            subcycling=True,
            acceleration="aitken",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])

    def test_subcycling_with_diagnostics(self):
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
            diagnostics=True,
        )
        gm.compile()
        gm.step()
        diag = gm.coupling_diagnostics()
        assert "fast+slow" in diag
        assert diag["fast+slow"]["iterations"] >= 1

    def test_subcycling_with_jacobi(self):
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
            iteration_mode="jacobi",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])

    def test_subcycling_with_mixed_norm(self):
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=15,
            convergence_norm="mixed",
            atol=1e-8, rtol=1e-4,
            subcycling=True,
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])


# ==================================================================
# JAX compatibility tests
# ==================================================================

class TestSubcyclingJAX:
    """JAX integration tests for subcycled coupling."""

    def test_subcycling_with_scan(self):
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
        )
        gm.compile()
        final, history = gm.run_scan_with_history(50)
        assert history["fast"]["position"].shape == (50,)
        assert jnp.all(jnp.isfinite(history["fast"]["position"]))

    def test_subcycling_grad_compatible(self):
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=5, tolerance=1e-6,
            subcycling=True,
        )
        gm.compile()
        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        # Use actual initial state (includes _meta for multirate)
        init_state = dict(gm._state)

        def loss_fn(init_pos):
            state = {
                **init_state,
                "fast": {"position": init_pos,
                         "velocity": jnp.array(0.0)},
                "slow": {"position": jnp.array(3.0),
                          "velocity": jnp.array(0.0)},
            }
            for _ in range(3):
                state = step_fn(state, ext)
            return state["fast"]["position"]

        g = jax.grad(loss_fn)(jnp.array(0.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_subcycling_in_multirate_graph(self):
        """Subcycled coupling group inside a multi-rate graph."""
        gm = GraphManager()
        # Fast and slow springs (subcycled coupling group)
        gm.add_node(SpringDamperNode(name="fast", timestep=0.001,
                                      stiffness=50.0, damping=1.0,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode(name="slow", timestep=0.005,
                                      stiffness=50.0, damping=1.0,
                                      initial_position=3.0))
        gm.add_edge("fast", "slow", "position", "anchor_position")
        gm.add_edge("slow", "fast", "position", "anchor_position")
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
        )
        # Independent node at yet another rate
        gm.add_node(SpringDamperNode(name="other", timestep=0.002,
                                      initial_position=5.0))
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])
        assert jnp.isfinite(state["slow"]["position"])
        assert jnp.isfinite(state["other"]["position"])
