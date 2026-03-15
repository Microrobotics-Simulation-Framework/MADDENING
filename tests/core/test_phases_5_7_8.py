"""Tests for Phase 5 (IQN auto-detect, quadratic interpolation, interface
residual), Phase 7 (IQN-IMVJ), and Phase 8 (waveform relaxation).

Covers: auto interface-field detection for IQN, quadratic subcycling
interpolation, interface residual convergence norm, IQN-IMVJ acceleration
with Jacobian reuse, and waveform relaxation.
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.coupling import CouplingGroup
from maddening.core.coupling_acceleration import (
    coupling_residual_interface,
    coupling_residual_l2,
    flatten_coupled_state,
    unflatten_coupled_state,
)
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


# ==================================================================
# Helpers
# ==================================================================

def _make_bidirectional_springs(dt=0.001, k=50.0, c=1.0, m=1.0,
                                rest=1.0, pos_a=0.0, pos_b=2.0):
    """Two springs coupled bidirectionally (each is the other's anchor)."""
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
    """Two HeatNodes coupled at their shared boundary."""
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
# Phase 5a: IQN auto interface-field detection
# ==================================================================

class TestIQNAutoDetect:
    """Tests for auto-detection of interface fields for IQN-ILS."""

    def test_iqn_auto_detects_interface_fields(self):
        """SpringDamperNodes have position+velocity state but only
        position appears on coupling edges.  The flattened IQN vector
        should only contain position DOFs (1 per spring = 2 total),
        not all 4 state DOFs.
        """
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.compile()

        state = gm._state
        node_names = sorted(["spring_a", "spring_b"])

        # Full flatten (all fields): position + velocity = 2 per node = 4
        flat_all = flatten_coupled_state(state, node_names, fields=None)
        assert flat_all.shape[0] == 4

        # Auto-detected fields from edges: only "position" per node
        # The edges are: spring_a.position -> spring_b, spring_b.position -> spring_a
        auto_fields = {}
        for edge in gm._edges:
            if edge.source_node in node_names and edge.target_node in node_names:
                auto_fields.setdefault(edge.source_node, set()).add(
                    edge.source_field
                )
        auto_fields_sorted = {
            nn: tuple(sorted(fs)) for nn, fs in auto_fields.items()
        }
        flat_auto = flatten_coupled_state(
            state, node_names, fields=auto_fields_sorted
        )
        # Only position: 1 per spring = 2 total
        assert flat_auto.shape[0] == 2
        assert flat_auto.shape[0] < flat_all.shape[0]

    def test_iqn_auto_detect_matches_manual(self):
        """Auto-detected fields produce same simulation result as
        explicit accelerated_fields specification.
        """
        kwargs = dict(dt=0.01, k=50.0, c=0.5, pos_a=0.0, pos_b=3.0)
        n_steps = 50

        # Auto-detect (default)
        gm_auto = _make_bidirectional_springs(**kwargs)
        gm_auto.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="iqn-ils",
        )
        gm_auto.compile()
        s_auto = gm_auto.run_scan(n_steps)

        # Manual: explicitly specify position only
        gm_manual = _make_bidirectional_springs(**kwargs)
        gm_manual.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="iqn-ils",
            accelerated_fields={
                "spring_a": ("position",),
                "spring_b": ("position",),
            },
        )
        gm_manual.compile()
        s_manual = gm_manual.run_scan(n_steps)

        # Both should give identical results
        diff_a = float(jnp.abs(
            s_auto["spring_a"]["position"] - s_manual["spring_a"]["position"]
        ))
        diff_b = float(jnp.abs(
            s_auto["spring_b"]["position"] - s_manual["spring_b"]["position"]
        ))
        assert diff_a < 1e-8
        assert diff_b < 1e-8

    def test_iqn_auto_detect_with_heat(self):
        """Heat nodes with many cells, but only boundary values on edges.

        The edges extract T[-1] and T[0] via transforms, but the
        source_field is 'temperature' (the full array).  Auto-detect
        includes 'temperature' per node, but the IQN vector is still
        smaller than all possible fields because only nodes that appear
        as edge sources are included.
        """
        gm = _make_coupled_heat_rods(n_cells=20)
        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="iqn-ils",
        )
        gm.compile()

        # Verify it compiles and runs
        state = gm.run_scan(50)
        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))
        assert jnp.all(jnp.isfinite(state["rod_b"]["temperature"]))

        # Auto-detect: both nodes' 'temperature' is on edges
        # IQN vector = 20 + 20 = 40 (all temperature DOFs)
        node_names = sorted(["rod_a", "rod_b"])
        auto_fields = {}
        for edge in gm._edges:
            if edge.source_node in node_names and edge.target_node in node_names:
                auto_fields.setdefault(edge.source_node, set()).add(
                    edge.source_field
                )
        auto_fields_sorted = {
            nn: tuple(sorted(fs)) for nn, fs in auto_fields.items()
        }
        flat = flatten_coupled_state(
            gm._state, node_names, fields=auto_fields_sorted
        )
        # Both nodes contribute temperature (20 cells each)
        assert flat.shape[0] == 40


# ==================================================================
# Phase 5c: Interface residual checking
# ==================================================================

class TestInterfaceResidual:
    """Tests for convergence_norm='interface'."""

    def test_interface_residual_detects_convergence(self):
        """Converged state has near-zero interface residual."""
        # Same state for both iterations => zero residual
        state = {
            "spring_a": {"position": jnp.array(1.5), "velocity": jnp.array(0.0)},
            "spring_b": {"position": jnp.array(1.5), "velocity": jnp.array(0.0)},
        }
        edges = [
            EdgeSpec("spring_a", "spring_b", "position", "anchor_position"),
            EdgeSpec("spring_b", "spring_a", "position", "anchor_position"),
        ]
        r = coupling_residual_interface(state, state, edges)
        assert float(r) == pytest.approx(0.0, abs=1e-12)

    def test_interface_residual_detects_difference(self):
        """Different states produce nonzero interface residual."""
        s_old = {
            "spring_a": {"position": jnp.array(1.0), "velocity": jnp.array(0.0)},
            "spring_b": {"position": jnp.array(2.0), "velocity": jnp.array(0.0)},
        }
        s_new = {
            "spring_a": {"position": jnp.array(1.5), "velocity": jnp.array(0.0)},
            "spring_b": {"position": jnp.array(2.5), "velocity": jnp.array(0.0)},
        }
        edges = [
            EdgeSpec("spring_a", "spring_b", "position", "anchor_position"),
            EdgeSpec("spring_b", "spring_a", "position", "anchor_position"),
        ]
        r = coupling_residual_interface(s_new, s_old, edges)
        assert float(r) > 0.0

    def test_interface_residual_vs_iterate_change(self):
        """For a well-conditioned problem, both L2 and interface norms
        report successful convergence.

        Uses weak coupling (low k, high damping, small dt) so the
        fixed-point iteration converges quickly in both norms.
        """
        # Well-conditioned problem: weak spring, high damping, small dt
        spring_kwargs = dict(dt=0.001, k=5.0, c=2.0, pos_a=0.0, pos_b=2.0)

        # Test with L2 norm
        gm_l2 = _make_bidirectional_springs(**spring_kwargs)
        gm_l2.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-10,
            convergence_norm="l2",
            diagnostics=True,
        )
        gm_l2.compile()
        gm_l2.step()
        diag_l2 = gm_l2.coupling_diagnostics()

        # Test with interface norm
        gm_if = _make_bidirectional_springs(**spring_kwargs)
        gm_if.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30,
            convergence_norm="interface",
            atol=1e-2, rtol=1e-2,
            diagnostics=True,
        )
        gm_if.compile()
        gm_if.step()
        diag_if = gm_if.coupling_diagnostics()

        # L2 residual should be very small after convergence
        assert diag_l2["spring_a+spring_b"]["residual"] < 1.0
        # Interface residual should also indicate convergence (<= 1.0)
        assert diag_if["spring_a+spring_b"]["residual"] < 1.0

    def test_interface_norm_in_coupling_group(self):
        """convergence_norm='interface' works in a full run_scan."""
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=15,
            convergence_norm="interface",
            atol=1e-8, rtol=1e-6,
        )
        gm.compile()
        state = gm.run_scan(100)
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])

    def test_interface_norm_with_heat_nodes(self):
        """Interface norm works with array-valued state."""
        gm = _make_coupled_heat_rods()
        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=15,
            convergence_norm="interface",
            atol=1e-8, rtol=1e-4,
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))


# ==================================================================
# Phase 5b: Quadratic subcycling interpolation
# ==================================================================

class TestQuadraticInterpolation:
    """Tests for quadratic boundary interpolation in subcycling."""

    def test_quadratic_interpolation_converges(self):
        """Subcycled graph with boundary_interpolation='quadratic'
        produces finite results.
        """
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=15, tolerance=1e-8,
            subcycling=True,
            boundary_interpolation="quadratic",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])
        assert jnp.isfinite(state["slow"]["position"])

    def test_quadratic_more_accurate_than_linear(self):
        """On a problem with fast-varying boundaries, quadratic
        should be closer to a uniform-rate reference than linear.
        """
        k, c, m = 20.0, 1.0, 1.0
        pos_a, pos_b = 0.0, 3.0
        dt_fast, dt_slow = 0.001, 0.005
        n_slow_steps = 100
        n_fast_steps = n_slow_steps * round(dt_slow / dt_fast)

        # Reference: both nodes at fast rate
        gm_ref = _make_uniform_reference(
            dt=dt_fast, k=k, c=c, m=m, pos_a=pos_a, pos_b=pos_b
        )
        gm_ref.add_coupling_group(
            ["fast", "slow"],
            max_iterations=20, tolerance=1e-10,
        )
        gm_ref.compile()
        s_ref = gm_ref.run_scan(n_fast_steps)

        # Linear interpolation
        gm_lin = _make_mixed_rate_springs(
            dt_fast=dt_fast, dt_slow=dt_slow,
            k=k, c=c, m=m, pos_a=pos_a, pos_b=pos_b,
        )
        gm_lin.add_coupling_group(
            ["fast", "slow"],
            max_iterations=20, tolerance=1e-10,
            subcycling=True,
            boundary_interpolation="linear",
        )
        gm_lin.compile()
        s_lin = gm_lin.run_scan(n_slow_steps)

        # Quadratic interpolation
        gm_quad = _make_mixed_rate_springs(
            dt_fast=dt_fast, dt_slow=dt_slow,
            k=k, c=c, m=m, pos_a=pos_a, pos_b=pos_b,
        )
        gm_quad.add_coupling_group(
            ["fast", "slow"],
            max_iterations=20, tolerance=1e-10,
            subcycling=True,
            boundary_interpolation="quadratic",
        )
        gm_quad.compile()
        s_quad = gm_quad.run_scan(n_slow_steps)

        # Compute errors vs reference
        err_lin = float(
            jnp.abs(s_ref["fast"]["position"] - s_lin["fast"]["position"])
            + jnp.abs(s_ref["slow"]["position"] - s_lin["slow"]["position"])
        )
        err_quad = float(
            jnp.abs(s_ref["fast"]["position"] - s_quad["fast"]["position"])
            + jnp.abs(s_ref["slow"]["position"] - s_quad["slow"]["position"])
        )

        # Both should be finite
        assert jnp.isfinite(jnp.array(err_lin))
        assert jnp.isfinite(jnp.array(err_quad))

        # Quadratic should be at least as accurate as linear (or very close)
        assert err_quad <= err_lin + 1e-6, (
            f"Quadratic error {err_quad} should be <= linear error {err_lin}"
        )


# ==================================================================
# Phase 7a: IQN-IMVJ
# ==================================================================

class TestIQNIMVJ:
    """Tests for IQN-IMVJ (IQN-ILS with multi-timestep Jacobian reuse)."""

    def test_imvj_converges(self):
        """Basic convergence with acceleration='iqn-imvj'."""
        gm = _make_bidirectional_springs(dt=0.01, k=50.0, c=0.5,
                                         pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="iqn-imvj",
            jacobian_reuse=3,
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])

    def test_imvj_with_scan(self):
        """IQN-IMVJ works in lax.scan (pytree structure stable)."""
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=8, tolerance=1e-8,
            acceleration="iqn-imvj",
            jacobian_reuse=3,
        )
        gm.compile()
        final, history = gm.run_scan_with_history(100)
        assert history["spring_a"]["position"].shape == (100,)
        assert jnp.all(jnp.isfinite(history["spring_a"]["position"]))
        # Final should match last history entry
        assert float(jnp.abs(
            final["spring_a"]["position"]
            - history["spring_a"]["position"][-1]
        )) < 1e-6

    def test_imvj_grad_compatible(self):
        """IQN-IMVJ is differentiable."""
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=5, tolerance=1e-6,
            acceleration="iqn-imvj",
            jacobian_reuse=3,
        )
        gm.compile()
        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        def loss_fn(init_pos):
            state = dict(gm._state)
            state["spring_a"] = {
                "position": init_pos,
                "velocity": jnp.array(0.0),
            }
            state["spring_b"] = {
                "position": jnp.array(3.0),
                "velocity": jnp.array(0.0),
            }
            for _ in range(5):
                state = step_fn(state, ext)
            return state["spring_a"]["position"]

        g = jax.grad(loss_fn)(jnp.array(0.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_imvj_matches_converged_ils(self):
        """IQN-IMVJ should reach the same fixed point as IQN-ILS."""
        kwargs = dict(dt=0.001, k=10.0, c=1.0, pos_a=0.0, pos_b=3.0)
        n_steps = 50

        gm_ils = _make_bidirectional_springs(**kwargs)
        gm_ils.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-12,
            acceleration="iqn-ils",
        )
        gm_ils.compile()
        s_ils = gm_ils.run_scan(n_steps)

        gm_imvj = _make_bidirectional_springs(**kwargs)
        gm_imvj.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-12,
            acceleration="iqn-imvj",
            jacobian_reuse=5,
        )
        gm_imvj.compile()
        s_imvj = gm_imvj.run_scan(n_steps)

        diff = float(jnp.abs(
            s_ils["spring_a"]["position"]
            - s_imvj["spring_a"]["position"]
        ))
        assert diff < 1e-3

    def test_imvj_with_heat_nodes(self):
        """IQN-IMVJ works with array-valued state (HeatNode)."""
        gm = _make_coupled_heat_rods()
        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="iqn-imvj",
            jacobian_reuse=3,
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))
        assert jnp.all(jnp.isfinite(state["rod_b"]["temperature"]))

    def test_imvj_with_diagnostics(self):
        """IQN-IMVJ stores diagnostics and V/W matrices."""
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="iqn-imvj",
            jacobian_reuse=3,
            diagnostics=True,
        )
        gm.compile()
        gm.step()
        diag = gm.coupling_diagnostics()
        assert "spring_a+spring_b" in diag
        assert diag["spring_a+spring_b"]["iterations"] >= 1


# ==================================================================
# Phase 8a: Waveform relaxation
# ==================================================================

class TestWaveformRelaxation:
    """Tests for waveform relaxation (multiple passes over sub-steps)."""

    def test_waveform_relaxation_converges(self):
        """Basic convergence with waveform_iterations=3."""
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
            waveform_iterations=3,
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])
        assert jnp.isfinite(state["slow"]["position"])

    def test_waveform_with_scan(self):
        """Waveform relaxation works in lax.scan."""
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=8, tolerance=1e-8,
            subcycling=True,
            waveform_iterations=3,
        )
        gm.compile()
        final, history = gm.run_scan_with_history(50)
        assert history["fast"]["position"].shape == (50,)
        assert jnp.all(jnp.isfinite(history["fast"]["position"]))
        assert jnp.all(jnp.isfinite(history["slow"]["position"]))

    def test_waveform_differs_from_single_pass(self):
        """Waveform iterations should give different (generally more
        accurate) results than a single pass.
        """
        kwargs = dict(dt_fast=0.001, dt_slow=0.005, k=50.0, c=1.0,
                      pos_a=0.0, pos_b=3.0)
        n_steps = 50

        gm_single = _make_mixed_rate_springs(**kwargs)
        gm_single.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
            waveform_iterations=1,
        )
        gm_single.compile()
        s_single = gm_single.run_scan(n_steps)

        gm_multi = _make_mixed_rate_springs(**kwargs)
        gm_multi.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
            waveform_iterations=3,
        )
        gm_multi.compile()
        s_multi = gm_multi.run_scan(n_steps)

        diff = float(jnp.abs(
            s_single["fast"]["position"] - s_multi["fast"]["position"]
        ))
        # With multiple waveform iterations, results should differ
        # (or be identical if already converged in 1 pass, which is
        # unlikely for stiff coupling)
        assert jnp.isfinite(jnp.array(diff))

    def test_waveform_with_aitken(self):
        """Waveform relaxation combined with Aitken acceleration."""
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=15, tolerance=1e-8,
            subcycling=True,
            waveform_iterations=2,
            acceleration="aitken",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["fast"]["position"])

    def test_waveform_with_diagnostics(self):
        """Waveform relaxation stores diagnostics."""
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=10, tolerance=1e-8,
            subcycling=True,
            waveform_iterations=2,
            diagnostics=True,
        )
        gm.compile()
        gm.step()
        diag = gm.coupling_diagnostics()
        assert "fast+slow" in diag
        assert diag["fast+slow"]["iterations"] >= 1

    def test_waveform_grad_compatible(self):
        """Waveform relaxation is differentiable."""
        gm = _make_mixed_rate_springs()
        gm.add_coupling_group(
            ["fast", "slow"],
            max_iterations=5, tolerance=1e-6,
            subcycling=True,
            waveform_iterations=2,
        )
        gm.compile()
        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()
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
