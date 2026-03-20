"""Tests for IQN-ILS quasi-Newton coupling acceleration (Phase 3).

Verifies that IQN-ILS converges, produces finite results, is
differentiable, and works with both scalar and array-valued state.
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.coupling.acceleration import iqn_ils_update
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


# ==================================================================
# Helpers
# ==================================================================

def _make_bidirectional_springs(dt=0.001, k=50.0, c=1.0, m=1.0,
                                rest=1.0, pos_a=0.0, pos_b=2.0):
    gm = GraphManager()
    a = SpringDamperNode(name="spring_a", timestep=dt, stiffness=k,
                         damping=c, mass=m, rest_length=rest,
                         initial_position=pos_a)
    b = SpringDamperNode(name="spring_b", timestep=dt, stiffness=k,
                         damping=c, mass=m, rest_length=rest,
                         initial_position=pos_b)
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
    return gm


def _make_coupled_heat_rods(dt=0.001, n_cells=10):
    gm = GraphManager()
    a = HeatNode(name="rod_a", timestep=dt, n_cells=n_cells,
                 thermal_diffusivity=0.01, length=1.0,
                 initial_temperature=100.0)
    b = HeatNode(name="rod_b", timestep=dt, n_cells=n_cells,
                 thermal_diffusivity=0.01, length=1.0,
                 initial_temperature=0.0)
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                transform=lambda T: T[-1])
    gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                transform=lambda T: T[0])
    return gm


# ==================================================================
# Unit tests for iqn_ils_update
# ==================================================================

class TestIQNILSUnit:
    """Unit tests for the IQN-ILS update function."""

    def test_iqn_ils_returns_finite(self):
        n_dof = 4
        max_cols = 5
        x_raw = jnp.array([1.1, 2.2, 3.3, 4.4])
        x_old = jnp.array([1.0, 2.0, 3.0, 4.0])
        prev_r = jnp.zeros(n_dof)
        prev_s = x_old
        V = jnp.zeros((n_dof, max_cols))
        W = jnp.zeros((n_dof, max_cols))
        n_cols = jnp.int32(0)
        omega = jnp.array(1.0)
        prev_ra = jnp.zeros(n_dof)

        result = iqn_ils_update(
            x_raw, x_old, prev_r, prev_s,
            V, W, n_cols, omega, prev_ra,
        )
        x_new = result[0]
        assert jnp.all(jnp.isfinite(x_new))

    def test_iqn_ils_first_iteration_uses_fallback(self):
        """On first iteration (n_cols=0), QN is skipped -> Aitken fallback."""
        n_dof = 2
        x_raw = jnp.array([2.0, 4.0])
        x_old = jnp.array([1.0, 2.0])
        result = iqn_ils_update(
            x_raw, x_old,
            jnp.zeros(n_dof), x_old,
            jnp.zeros((n_dof, 3)), jnp.zeros((n_dof, 3)),
            jnp.int32(0), jnp.array(1.0), jnp.zeros(n_dof),
        )
        x_new = result[0]
        assert jnp.all(jnp.isfinite(x_new))
        # n_cols should still be 0 on first call
        assert int(result[3]) == 0

    def test_iqn_ils_column_count_grows(self):
        """After a non-first iteration, n_cols should increase."""
        n_dof = 2
        max_cols = 5
        x_raw = jnp.array([1.1, 2.2])
        x_old = jnp.array([1.0, 2.0])
        prev_r = jnp.array([0.05, 0.1])
        prev_s = jnp.array([0.9, 1.8])
        result = iqn_ils_update(
            x_raw, x_old, prev_r, prev_s,
            jnp.zeros((n_dof, max_cols)),
            jnp.zeros((n_dof, max_cols)),
            jnp.int32(1), jnp.array(1.0), jnp.zeros(n_dof),
        )
        assert int(result[3]) == 2  # n_cols grew from 1 to 2


# ==================================================================
# Integration tests
# ==================================================================

class TestIQNILSIntegration:
    """Integration tests for IQN-ILS in coupled graphs."""

    def test_iqn_convergence(self):
        gm = _make_bidirectional_springs(dt=0.01, k=50.0, c=0.5,
                                         pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-10,
            acceleration="iqn-ils",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])

    def test_iqn_matches_converged_plain(self):
        """IQN-ILS should reach the same fixed point as plain iteration."""
        kwargs = dict(dt=0.001, k=10.0, c=1.0, pos_a=0.0, pos_b=3.0)
        n_steps = 50

        gm_plain = _make_bidirectional_springs(**kwargs)
        gm_plain.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-12,
        )
        gm_plain.compile()
        s_plain = gm_plain.run_scan(n_steps)

        gm_iqn = _make_bidirectional_springs(**kwargs)
        gm_iqn.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-12,
            acceleration="iqn-ils",
        )
        gm_iqn.compile()
        s_iqn = gm_iqn.run_scan(n_steps)

        diff = float(jnp.abs(
            s_plain["spring_a"]["position"]
            - s_iqn["spring_a"]["position"]
        ))
        assert diff < 1e-3

    def test_iqn_with_scan(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=8, tolerance=1e-8,
            acceleration="iqn-ils",
        )
        gm.compile()
        state = gm.run_scan(100)
        assert jnp.isfinite(state["spring_a"]["position"])

    def test_iqn_grad_compatible(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=5, tolerance=1e-6,
            acceleration="iqn-ils",
        )
        gm.compile()
        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        def loss_fn(init_pos):
            state = {
                "spring_a": {"position": init_pos,
                             "velocity": jnp.array(0.0)},
                "spring_b": {"position": jnp.array(3.0),
                             "velocity": jnp.array(0.0)},
            }
            for _ in range(5):
                state = step_fn(state, ext)
            return state["spring_a"]["position"]

        g = jax.grad(loss_fn)(jnp.array(0.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_iqn_with_heat_nodes(self):
        gm = _make_coupled_heat_rods()
        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="iqn-ils",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))
        assert jnp.all(jnp.isfinite(state["rod_b"]["temperature"]))

    def test_iqn_with_diagnostics(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="iqn-ils",
            diagnostics=True,
        )
        gm.compile()
        gm.step()
        diag = gm.coupling_diagnostics()
        assert "spring_a+spring_b" in diag
        assert diag["spring_a+spring_b"]["iterations"] >= 1

    def test_iqn_with_jacobi(self):
        """IQN-ILS combined with Jacobi iteration mode."""
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="iqn-ils",
            iteration_mode="jacobi",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["spring_a"]["position"])
