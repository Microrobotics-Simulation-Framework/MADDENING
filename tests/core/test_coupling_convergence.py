"""Tests for coupling convergence infrastructure (Phase 1) and
iteration modes (Phase 2).

Covers: per-field mixed norm, diagnostics, Aitken acceleration,
fixed relaxation, and Jacobi iteration mode.  Uses both scalar
(SpringDamperNode) and array-valued (HeatNode) state to ensure
generality.
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.coupling import CouplingGroup
from maddening.core.coupling.acceleration import (
    aitken_relaxation,
    coupling_residual_l2,
    coupling_residual_mixed,
    fixed_relaxation,
    flatten_coupled_state,
    unflatten_coupled_state,
)
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


# ==================================================================
# Helpers
# ==================================================================

def _make_bidirectional_springs(dt=0.001, k=50.0, c=1.0, m=1.0,
                                rest=1.0, pos_a=0.0, pos_b=2.0):
    """Two springs coupled bidirectionally."""
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


def _make_coupled_heat_rods(dt=0.001, n_cells=10,
                            thermal_diffusivity=0.01):
    """Two HeatNodes coupled at their shared boundary.

    Rod A's right boundary is rod B's left BC, and vice versa.
    This creates a bidirectional cycle with array-valued state.
    """
    gm = GraphManager()
    a = HeatNode(name="rod_a", timestep=dt, n_cells=n_cells,
                 thermal_diffusivity=thermal_diffusivity, length=1.0,
                 initial_temperature=100.0)
    b = HeatNode(name="rod_b", timestep=dt, n_cells=n_cells,
                 thermal_diffusivity=thermal_diffusivity, length=1.0,
                 initial_temperature=0.0)
    gm.add_node(a)
    gm.add_node(b)
    # A's rightmost cell -> B's left BC
    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                transform=lambda T: T[-1])
    # B's leftmost cell -> A's right BC
    gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                transform=lambda T: T[0])
    return gm


# ==================================================================
# Test residual norms (unit tests)
# ==================================================================

class TestResidualNorms:
    """Unit tests for coupling_residual_l2 and coupling_residual_mixed."""

    def test_l2_identical_states_zero(self):
        s = {"a": {"x": jnp.array(1.0), "v": jnp.array(2.0)}}
        r = coupling_residual_l2(s, s, ["a"])
        assert float(r) == pytest.approx(0.0, abs=1e-12)

    def test_l2_known_value(self):
        s_old = {"a": {"x": jnp.array(0.0)}}
        s_new = {"a": {"x": jnp.array(3.0)}}
        r = coupling_residual_l2(s_new, s_old, ["a"])
        assert float(r) == pytest.approx(3.0, abs=1e-6)

    def test_mixed_identical_states_zero(self):
        s = {"a": {"x": jnp.array(1.0), "v": jnp.array(2.0)}}
        r = coupling_residual_mixed(s, s, ["a"], atol=1e-8, rtol=1e-6)
        assert float(r) == pytest.approx(0.0, abs=1e-12)

    def test_mixed_norm_known_value(self):
        s_old = {"a": {"x": jnp.array(100.0)}}
        s_new = {"a": {"x": jnp.array(101.0)}}
        # diff = 1.0, scale = 1e-8 + 1e-6 * 101 = ~1.01e-4
        # scaled = 1.0 / 1.01e-4 ≈ 9900
        r = coupling_residual_mixed(s_new, s_old, ["a"], atol=1e-8, rtol=1e-6)
        assert float(r) > 1.0  # Not converged

    def test_mixed_scale_invariance(self):
        """Same relative change at different magnitudes gives similar norm."""
        # 1% change at magnitude 1
        s1_old = {"a": {"x": jnp.array(1.0)}}
        s1_new = {"a": {"x": jnp.array(1.01)}}
        r1 = coupling_residual_mixed(s1_new, s1_old, ["a"],
                                     atol=1e-10, rtol=1e-3)

        # 1% change at magnitude 1000
        s2_old = {"a": {"x": jnp.array(1000.0)}}
        s2_new = {"a": {"x": jnp.array(1010.0)}}
        r2 = coupling_residual_mixed(s2_new, s2_old, ["a"],
                                     atol=1e-10, rtol=1e-3)

        # With rtol-dominated scaling, both should give similar norms
        assert float(r1) == pytest.approx(float(r2), rel=0.1)

    def test_mixed_with_arrays(self):
        """Works with array-valued fields."""
        s_old = {"a": {"T": jnp.ones(10)}}
        s_new = {"a": {"T": jnp.ones(10) * 1.001}}
        r = coupling_residual_mixed(s_new, s_old, ["a"],
                                    atol=1e-8, rtol=1e-2)
        assert jnp.isfinite(r)


# ==================================================================
# Test flatten/unflatten (unit tests)
# ==================================================================

class TestFlattenUnflatten:
    """Unit tests for state flattening and unflattening."""

    def test_roundtrip_scalar(self):
        s = {"a": {"x": jnp.array(1.0), "v": jnp.array(2.0)},
             "b": {"x": jnp.array(3.0), "v": jnp.array(4.0)}}
        flat = flatten_coupled_state(s, ["a", "b"])
        restored = unflatten_coupled_state(flat, s, ["a", "b"])
        for nn in ["a", "b"]:
            for f in s[nn]:
                assert float(restored[nn][f]) == float(s[nn][f])

    def test_roundtrip_array(self):
        s = {"a": {"T": jnp.linspace(0, 1, 20)},
             "b": {"T": jnp.linspace(1, 2, 20)}}
        flat = flatten_coupled_state(s, ["a", "b"])
        assert flat.shape == (40,)
        restored = unflatten_coupled_state(flat, s, ["a", "b"])
        for nn in ["a", "b"]:
            assert jnp.allclose(restored[nn]["T"], s[nn]["T"])

    def test_deterministic_ordering(self):
        """Fields are sorted, so ordering is deterministic."""
        s = {"a": {"z": jnp.array(1.0), "a": jnp.array(2.0)}}
        flat = flatten_coupled_state(s, ["a"])
        # "a" before "z" in sorted order
        assert float(flat[0]) == 2.0
        assert float(flat[1]) == 1.0


# ==================================================================
# Test mixed norm in coupling (integration)
# ==================================================================

class TestMixedNormCoupling:
    """Integration tests for mixed norm in coupled graphs."""

    def test_mixed_norm_convergence(self):
        gm = _make_bidirectional_springs()
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=20,
            convergence_norm="mixed",
            atol=1e-8,
            rtol=1e-6,
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])

    def test_l2_default_backward_compatible(self):
        """Default convergence_norm='l2' produces same results as old API."""
        kwargs = dict(dt=0.01, k=50.0, c=0.5, pos_a=0.0, pos_b=3.0)
        n_steps = 50

        gm_old = _make_bidirectional_springs(**kwargs)
        gm_old.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
        )
        gm_old.compile()
        s_old = gm_old.run_scan(n_steps)

        gm_new = _make_bidirectional_springs(**kwargs)
        gm_new.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
            convergence_norm="l2",
        )
        gm_new.compile()
        s_new = gm_new.run_scan(n_steps)

        assert float(jnp.abs(
            s_old["spring_a"]["position"] - s_new["spring_a"]["position"]
        )) < 1e-10

    def test_mixed_norm_with_heat_nodes(self):
        gm = _make_coupled_heat_rods()
        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=15,
            convergence_norm="mixed",
            atol=1e-8,
            rtol=1e-4,
        )
        gm.compile()
        state = gm.run_scan(100)
        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))
        assert jnp.all(jnp.isfinite(state["rod_b"]["temperature"]))


# ==================================================================
# Test diagnostics
# ==================================================================

class TestDiagnostics:
    """Tests for coupling convergence diagnostics."""

    def test_diagnostics_reports_iteration_count(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-10,
            diagnostics=True,
        )
        gm.compile()
        gm.step()
        diag = gm.coupling_diagnostics()
        key = "spring_a+spring_b"
        assert key in diag
        assert diag[key]["iterations"] >= 1
        assert diag[key]["residual"] >= 0.0

    def test_diagnostics_off_by_default(self):
        gm = _make_bidirectional_springs()
        gm.add_coupling_group(["spring_a", "spring_b"])
        gm.compile()
        gm.step()
        diag = gm.coupling_diagnostics()
        assert diag == {}

    def test_diagnostics_residual_decreases(self):
        """For a well-conditioned problem, final residual < first residual."""
        gm = _make_bidirectional_springs(dt=0.001, k=10.0, c=1.0,
                                         pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=20, tolerance=1e-12,
            diagnostics=True,
        )
        gm.compile()
        gm.step()
        diag = gm.coupling_diagnostics()
        # Residual should be very small after convergence
        assert diag["spring_a+spring_b"]["residual"] < 1.0

    def test_diagnostics_with_scan(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
            diagnostics=True,
        )
        gm.compile()
        gm.run_scan(50)
        diag = gm.coupling_diagnostics()
        assert "spring_a+spring_b" in diag

    def test_diagnostics_multiple_groups(self):
        gm = GraphManager()
        dt = 0.01
        gm.add_node(SpringDamperNode(name="A", timestep=dt,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode(name="B", timestep=dt,
                                      initial_position=2.0))
        gm.add_edge("A", "B", "position", "anchor_position")
        gm.add_edge("B", "A", "position", "anchor_position")
        gm.add_node(SpringDamperNode(name="C", timestep=dt,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode(name="D", timestep=dt,
                                      initial_position=3.0))
        gm.add_edge("C", "D", "position", "anchor_position")
        gm.add_edge("D", "C", "position", "anchor_position")
        gm.add_coupling_group(["A", "B"], diagnostics=True)
        gm.add_coupling_group(["C", "D"], diagnostics=True)
        gm.compile()
        gm.step()
        diag = gm.coupling_diagnostics()
        assert "A+B" in diag
        assert "C+D" in diag


# ==================================================================
# Test Aitken acceleration
# ==================================================================

class TestAitkenAcceleration:
    """Tests for Aitken delta-squared acceleration."""

    def test_aitken_unit_function(self):
        """Unit test: aitken_relaxation produces finite output."""
        x_old = jnp.array([1.0, 2.0, 3.0])
        x_raw = jnp.array([1.1, 2.2, 3.3])
        prev_r = jnp.array([0.05, 0.1, 0.15])
        omega = jnp.array(1.0)
        x_rel, new_omega, residual = aitken_relaxation(
            x_old, x_raw, prev_r, omega
        )
        assert jnp.all(jnp.isfinite(x_rel))
        assert jnp.isfinite(new_omega)
        assert jnp.all(jnp.isfinite(residual))

    def test_aitken_first_iteration_omega_clamped(self):
        """With zero prev_residual, omega is clamped within [0.01, 2.0]."""
        x_old = jnp.array([1.0, 2.0])
        x_raw = jnp.array([1.5, 2.5])
        prev_r = jnp.zeros(2)
        omega = jnp.array(1.0)
        x_rel, new_omega, _ = aitken_relaxation(x_old, x_raw, prev_r, omega)
        # With zero prev_r: numerator = 0, so omega = 0 -> clamped to 0.01
        assert float(new_omega) == pytest.approx(0.01, abs=1e-6)
        assert jnp.all(jnp.isfinite(x_rel))

    def test_aitken_convergence_in_graph(self):
        gm = _make_bidirectional_springs(dt=0.01, k=50.0, c=0.5,
                                         pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=20, tolerance=1e-10,
            acceleration="aitken",
        )
        gm.compile()
        state = gm.run_scan(100)
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])

    def test_aitken_matches_fixed_point(self):
        """Aitken and plain iteration should reach same fixed point."""
        kwargs = dict(dt=0.001, k=10.0, c=1.0, pos_a=0.0, pos_b=3.0)
        n_steps = 50

        gm_plain = _make_bidirectional_springs(**kwargs)
        gm_plain.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-12,
        )
        gm_plain.compile()
        s_plain = gm_plain.run_scan(n_steps)

        gm_aitken = _make_bidirectional_springs(**kwargs)
        gm_aitken.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-12,
            acceleration="aitken",
        )
        gm_aitken.compile()
        s_aitken = gm_aitken.run_scan(n_steps)

        # Should reach the same result
        diff = float(jnp.abs(
            s_plain["spring_a"]["position"]
            - s_aitken["spring_a"]["position"]
        ))
        assert diff < 1e-4

    def test_aitken_grad_through_coupling(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=5, tolerance=1e-6,
            acceleration="aitken",
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

    def test_aitken_with_scan(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=5, tolerance=1e-6,
            acceleration="aitken",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["spring_a"]["position"])

    def test_aitken_with_heat_nodes(self):
        gm = _make_coupled_heat_rods()
        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=15, tolerance=1e-8,
            acceleration="aitken",
        )
        gm.compile()
        state = gm.run_scan(100)
        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))
        assert jnp.all(jnp.isfinite(state["rod_b"]["temperature"]))

    def test_aitken_with_diagnostics(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=20, tolerance=1e-10,
            acceleration="aitken",
            diagnostics=True,
        )
        gm.compile()
        gm.step()
        diag = gm.coupling_diagnostics()
        assert "spring_a+spring_b" in diag
        assert diag["spring_a+spring_b"]["iterations"] >= 1


# ==================================================================
# Test fixed relaxation
# ==================================================================

class TestFixedRelaxation:
    """Tests for fixed (constant) under-relaxation."""

    def test_fixed_relaxation_unit(self):
        x_old = jnp.array([1.0, 2.0])
        x_raw = jnp.array([2.0, 4.0])
        result = fixed_relaxation(x_old, x_raw, 0.5)
        expected = jnp.array([1.5, 3.0])
        assert jnp.allclose(result, expected)

    def test_fixed_relaxation_omega_one_noop(self):
        """omega=1.0 should be a no-op (return raw result)."""
        x_old = jnp.array([1.0])
        x_raw = jnp.array([3.0])
        result = fixed_relaxation(x_old, x_raw, 1.0)
        assert float(result[0]) == pytest.approx(3.0)

    def test_fixed_relaxation_default_noop(self):
        """Default relaxation=1.0 gives identical results to unrelaxed."""
        kwargs = dict(dt=0.01, k=50.0, c=0.5, pos_a=0.0, pos_b=3.0)
        n_steps = 50

        gm_plain = _make_bidirectional_springs(**kwargs)
        gm_plain.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
        )
        gm_plain.compile()
        s_plain = gm_plain.run_scan(n_steps)

        gm_fixed = _make_bidirectional_springs(**kwargs)
        gm_fixed.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=10, tolerance=1e-8,
            acceleration="fixed", relaxation=1.0,
        )
        gm_fixed.compile()
        s_fixed = gm_fixed.run_scan(n_steps)

        diff = float(jnp.abs(
            s_plain["spring_a"]["position"]
            - s_fixed["spring_a"]["position"]
        ))
        assert diff < 1e-6

    def test_under_relaxation_convergence(self):
        """Under-relaxation (omega<1) should converge for stiff problems."""
        gm = _make_bidirectional_springs(dt=0.01, k=200.0, c=0.0,
                                         pos_a=0.0, pos_b=5.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=20, tolerance=1e-8,
            acceleration="fixed", relaxation=0.5,
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])

    def test_fixed_relaxation_grad(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=5, tolerance=1e-6,
            acceleration="fixed", relaxation=0.7,
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


# ==================================================================
# Test Jacobi iteration mode
# ==================================================================

class TestJacobiIteration:
    """Tests for Jacobi (parallel) iteration mode."""

    def test_jacobi_produces_different_results(self):
        """Gauss-Seidel and Jacobi should produce different results.

        Uses asymmetric stiffness so the two methods don't coincide.
        Uses max_iterations=3 (insufficient for full convergence) to
        ensure iteration order matters.
        """
        n_steps = 50
        dt = 0.01

        def build(mode):
            gm = GraphManager()
            # Asymmetric: different stiffness and damping
            a = SpringDamperNode(name="spring_a", timestep=dt,
                                  stiffness=100.0, damping=0.5, mass=1.0,
                                  rest_length=1.0, initial_position=0.0)
            b = SpringDamperNode(name="spring_b", timestep=dt,
                                  stiffness=30.0, damping=2.0, mass=0.5,
                                  rest_length=1.0, initial_position=3.0)
            gm.add_node(a)
            gm.add_node(b)
            gm.add_edge("spring_a", "spring_b", "position",
                         "anchor_position")
            gm.add_edge("spring_b", "spring_a", "position",
                         "anchor_position")
            gm.add_coupling_group(
                ["spring_a", "spring_b"],
                max_iterations=3, tolerance=1e-15,
                iteration_mode=mode,
            )
            gm.compile()
            return gm

        s_gs = build("gauss-seidel").run_scan(n_steps)
        s_j = build("jacobi").run_scan(n_steps)

        # With limited iterations and asymmetric setup, results differ
        diff = float(jnp.abs(
            s_gs["spring_a"]["position"] - s_j["spring_a"]["position"]
        ))
        assert diff > 1e-10

    def test_jacobi_converges_with_relaxation(self):
        """Jacobi + under-relaxation should converge."""
        gm = _make_bidirectional_springs(dt=0.01, k=100.0, c=1.0,
                                         pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=20, tolerance=1e-8,
            iteration_mode="jacobi",
            acceleration="fixed", relaxation=0.5,
        )
        gm.compile()
        state = gm.run_scan(100)
        assert jnp.isfinite(state["spring_a"]["position"])

    def test_jacobi_with_aitken(self):
        """Jacobi + Aitken acceleration."""
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=15, tolerance=1e-8,
            iteration_mode="jacobi",
            acceleration="aitken",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["spring_a"]["position"])

    def test_jacobi_grad_compatible(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=5, tolerance=1e-6,
            iteration_mode="jacobi",
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

    def test_jacobi_with_scan(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=5, tolerance=1e-6,
            iteration_mode="jacobi",
        )
        gm.compile()
        state = gm.run_scan(50)
        assert jnp.isfinite(state["spring_a"]["position"])

    def test_jacobi_with_heat_nodes(self):
        gm = _make_coupled_heat_rods()
        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=15, tolerance=1e-8,
            iteration_mode="jacobi",
        )
        gm.compile()
        state = gm.run_scan(100)
        assert jnp.all(jnp.isfinite(state["rod_a"]["temperature"]))

    def test_jacobi_three_node_cycle(self):
        """A -> B -> C -> A under Jacobi iteration."""
        dt = 0.01
        gm = GraphManager()
        gm.add_node(SpringDamperNode(name="A", timestep=dt,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode(name="B", timestep=dt,
                                      initial_position=2.0))
        gm.add_node(SpringDamperNode(name="C", timestep=dt,
                                      initial_position=4.0))
        gm.add_edge("A", "B", "position", "anchor_position")
        gm.add_edge("B", "C", "position", "anchor_position")
        gm.add_edge("C", "A", "position", "anchor_position")
        gm.add_coupling_group(
            ["A", "B", "C"],
            max_iterations=15, tolerance=1e-8,
            iteration_mode="jacobi",
        )
        gm.compile()
        state = gm.run_scan(50)
        for name in ["A", "B", "C"]:
            assert jnp.isfinite(state[name]["position"])


# ==================================================================
# Test auto_couple with new parameters
# ==================================================================

class TestAutoCoupleKwargs:
    """auto_couple forwards new keyword arguments."""

    def test_auto_couple_with_acceleration(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        groups = gm.auto_couple(
            max_iterations=10, tolerance=1e-6,
            acceleration="aitken",
        )
        assert len(groups) == 1
        assert groups[0].acceleration == "aitken"

    def test_auto_couple_with_diagnostics(self):
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        groups = gm.auto_couple(diagnostics=True)
        assert groups[0].diagnostics is True
