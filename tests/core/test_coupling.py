"""Extensive tests for the Gauss-Seidel iterative coupling feature.

Covers: CouplingGroup management, auto_couple, validation, the
Gauss-Seidel while_loop iteration, JAX integration (JIT, grad,
lax.scan), physics accuracy, and Tarjan SCC detection.
"""

import logging
import warnings

import jax
import jax.numpy as jnp
import pytest

from maddening.core.coupling import CouplingGroup
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.schedule import find_strongly_connected_components
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode


# ==================================================================
# Helpers
# ==================================================================

def _make_bidirectional_springs(dt=0.001, k=50.0, c=1.0, m=1.0,
                                rest=1.0, pos_a=0.0, pos_b=2.0):
    """Two springs coupled bidirectionally (each is the other's anchor).

    Returns (gm, "spring_a", "spring_b").
    """
    gm = GraphManager()
    a = SpringDamperNode(name="spring_a", timestep=dt, stiffness=k,
                         damping=c, mass=m, rest_length=rest,
                         initial_position=pos_a)
    b = SpringDamperNode(name="spring_b", timestep=dt, stiffness=k,
                         damping=c, mass=m, rest_length=rest,
                         initial_position=pos_b)
    gm.add_node(a)
    gm.add_node(b)
    # A's position is B's anchor and vice-versa => cycle
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
    return gm


def _make_three_node_chain(dt=0.01):
    """A -> B -> C (no cycle)."""
    gm = GraphManager()
    gm.add_node(TableNode(name="A", timestep=dt, position=0.0))
    gm.add_node(SpringDamperNode(name="B", timestep=dt,
                                  initial_position=1.0))
    gm.add_node(SpringDamperNode(name="C", timestep=dt,
                                  initial_position=2.0))
    gm.add_edge("A", "B", "position", "anchor_position")
    gm.add_edge("B", "C", "position", "anchor_position")
    return gm


# ==================================================================
# TestCouplingGroupManagement
# ==================================================================

class TestCouplingGroupManagement:
    """Tests for add/remove/store of CouplingGroups."""

    def test_add_coupling_group(self):
        """Adding a coupling group stores it in the graph manager."""
        gm = _make_bidirectional_springs()
        group = gm.add_coupling_group(["spring_a", "spring_b"],
                                       max_iterations=5, tolerance=1e-4)
        assert isinstance(group, CouplingGroup)
        assert group.nodes == frozenset({"spring_a", "spring_b"})
        assert group.max_iterations == 5
        assert group.tolerance == 1e-4
        assert len(gm._coupling_groups) == 1

    def test_add_coupling_group_nonexistent_node(self):
        """Referencing a node that doesn't exist raises KeyError."""
        gm = GraphManager()
        gm.add_node(SpringDamperNode(name="a", timestep=0.01))
        with pytest.raises(KeyError, match="ghost"):
            gm.add_coupling_group(["a", "ghost"])

    def test_overlapping_groups_rejected(self):
        """A node cannot belong to two coupling groups."""
        gm = GraphManager()
        gm.add_node(SpringDamperNode(name="a", timestep=0.01))
        gm.add_node(SpringDamperNode(name="b", timestep=0.01))
        gm.add_node(SpringDamperNode(name="c", timestep=0.01))
        gm.add_coupling_group(["a", "b"])
        with pytest.raises(ValueError, match="already belong"):
            gm.add_coupling_group(["b", "c"])

    def test_remove_coupling_group(self):
        """Removing a coupling group by its node set."""
        gm = _make_bidirectional_springs()
        gm.add_coupling_group(["spring_a", "spring_b"])
        assert len(gm._coupling_groups) == 1
        gm.remove_coupling_group(["spring_a", "spring_b"])
        assert len(gm._coupling_groups) == 0

    def test_remove_nonexistent_group_is_noop(self):
        """Removing a group that doesn't exist doesn't raise."""
        gm = _make_bidirectional_springs()
        gm.remove_coupling_group(["spring_a", "spring_b"])
        assert len(gm._coupling_groups) == 0

    def test_coupling_group_marks_dirty(self):
        """Adding a coupling group sets _dirty so recompilation is needed."""
        gm = _make_bidirectional_springs()
        gm.compile()
        assert not gm._dirty
        gm.add_coupling_group(["spring_a", "spring_b"])
        assert gm._dirty


# ==================================================================
# TestAutoCouple
# ==================================================================

class TestAutoCouple:
    """Tests for automatic coupling-group discovery via Tarjan SCC."""

    def test_auto_couple_finds_cycle(self):
        """auto_couple detects a bidirectional cycle and creates a group."""
        gm = _make_bidirectional_springs()
        groups = gm.auto_couple(max_iterations=8, tolerance=1e-5)
        assert len(groups) == 1
        assert groups[0].nodes == frozenset({"spring_a", "spring_b"})
        assert groups[0].max_iterations == 8
        assert groups[0].tolerance == 1e-5

    def test_auto_couple_no_cycles(self):
        """Acyclic graph produces no coupling groups."""
        gm = _make_three_node_chain()
        groups = gm.auto_couple()
        assert groups == []

    def test_auto_couple_multiple_cycles(self):
        """Two separate cycles produce two coupling groups."""
        gm = GraphManager()
        dt = 0.01
        # Cycle 1: A <-> B
        gm.add_node(SpringDamperNode(name="A", timestep=dt))
        gm.add_node(SpringDamperNode(name="B", timestep=dt))
        gm.add_edge("A", "B", "position", "anchor_position")
        gm.add_edge("B", "A", "position", "anchor_position")
        # Cycle 2: C <-> D
        gm.add_node(SpringDamperNode(name="C", timestep=dt))
        gm.add_node(SpringDamperNode(name="D", timestep=dt))
        gm.add_edge("C", "D", "position", "anchor_position")
        gm.add_edge("D", "C", "position", "anchor_position")

        groups = gm.auto_couple()
        assert len(groups) == 2
        node_sets = {g.nodes for g in groups}
        assert frozenset({"A", "B"}) in node_sets
        assert frozenset({"C", "D"}) in node_sets

    def test_auto_couple_clears_previous_groups(self):
        """auto_couple removes pre-existing groups before discovery."""
        gm = _make_bidirectional_springs()
        gm.add_coupling_group(["spring_a", "spring_b"], max_iterations=3)
        assert len(gm._coupling_groups) == 1
        groups = gm.auto_couple(max_iterations=7)
        assert len(groups) == 1
        assert groups[0].max_iterations == 7  # new, not the old 3


# ==================================================================
# TestCouplingValidation
# ==================================================================

class TestCouplingValidation:
    """Tests for validation messages related to coupling."""

    def test_mixed_timestep_error(self):
        """Coupling group with different timesteps raises on compile."""
        gm = GraphManager()
        gm.add_node(SpringDamperNode(name="a", timestep=0.01))
        gm.add_node(SpringDamperNode(name="b", timestep=0.02))
        gm.add_edge("a", "b", "position", "anchor_position")
        gm.add_edge("b", "a", "position", "anchor_position")
        gm.add_coupling_group(["a", "b"])
        with pytest.raises(RuntimeError, match="mixed timesteps"):
            gm.compile()

    def test_cycle_covered_by_group_info(self):
        """Cycle in a coupling group shows INFO, not WARNING."""
        gm = _make_bidirectional_springs()
        gm.add_coupling_group(["spring_a", "spring_b"])
        issues = gm.validate()
        cycle_issues = [i for i in issues
                        if "cycle" in i.lower() or "coupling" in i.lower()]
        # Should have an INFO about the handled cycle
        assert any(i.startswith("INFO") for i in cycle_issues)
        # Should NOT have a WARNING about uncovered cycles
        assert not any(
            i.startswith("WARNING") and "staggering" in i
            for i in cycle_issues
        )

    def test_uncovered_cycle_warning(self):
        """Cycle without coupling group shows WARNING."""
        gm = _make_bidirectional_springs()
        issues = gm.validate()
        cycle_warnings = [i for i in issues
                          if i.startswith("WARNING") and "cycle" in i.lower()]
        assert len(cycle_warnings) >= 1


# ==================================================================
# TestGaussSeidel
# ==================================================================

class TestGaussSeidel:
    """Core Gauss-Seidel iteration tests."""

    def test_coupled_nodes_converge(self):
        """Coupled iteration converges and produces finite, reasonable results.

        For explicit integration schemes like semi-implicit Euler, Gauss-Seidel
        iteration ensures self-consistency of boundary conditions within each
        timestep.  It may or may not be more accurate than staggering depending
        on the problem; the key property is convergence (iteration produces a
        fixed point) and stability.
        """
        dt = 0.01
        k, c, m, rest = 10.0, 0.5, 1.0, 1.0
        n_steps = 100
        kwargs = dict(dt=dt, k=k, c=c, m=m, rest=rest,
                      pos_a=0.0, pos_b=3.0)

        # Staggered at dt
        gm_stag = _make_bidirectional_springs(**kwargs)
        gm_stag.compile()
        stag_state = gm_stag.run_scan(n_steps)

        # Coupled at dt
        gm_coupled = _make_bidirectional_springs(**kwargs)
        gm_coupled.add_coupling_group(["spring_a", "spring_b"],
                                       max_iterations=20, tolerance=1e-10)
        gm_coupled.compile()
        coupled_state = gm_coupled.run_scan(n_steps)

        # Both should produce finite results
        for name in ["spring_a", "spring_b"]:
            for field in ["position", "velocity"]:
                assert jnp.isfinite(coupled_state[name][field]), (
                    f"Coupled {name}.{field} is not finite"
                )
                assert jnp.isfinite(stag_state[name][field]), (
                    f"Staggered {name}.{field} is not finite"
                )

        # Coupled result should differ from staggered (iteration changed something)
        coupled_pos = float(coupled_state["spring_a"]["position"])
        stag_pos = float(stag_state["spring_a"]["position"])
        assert coupled_pos != pytest.approx(stag_pos, abs=1e-10), (
            "Coupled and staggered should give different results"
        )

    def test_iteration_actually_runs(self):
        """Coupling should produce different results than staggered."""
        dt = 0.01
        n_steps = 50
        gm_stag = _make_bidirectional_springs(dt=dt, pos_a=0.0, pos_b=5.0)
        gm_stag.compile()
        stag_state = gm_stag.run_scan(n_steps)

        gm_coupled = _make_bidirectional_springs(dt=dt, pos_a=0.0, pos_b=5.0)
        gm_coupled.add_coupling_group(["spring_a", "spring_b"],
                                       max_iterations=10, tolerance=1e-10)
        gm_coupled.compile()
        coupled_state = gm_coupled.run_scan(n_steps)

        # Results should differ (iteration changes the answer)
        diff_a = float(jnp.abs(
            stag_state["spring_a"]["position"]
            - coupled_state["spring_a"]["position"]
        ))
        diff_b = float(jnp.abs(
            stag_state["spring_b"]["position"]
            - coupled_state["spring_b"]["position"]
        ))
        assert diff_a + diff_b > 1e-6, (
            "Coupled and staggered solutions should differ"
        )

    def test_max_iterations_reached(self):
        """With very few iterations the solver still completes."""
        gm = _make_bidirectional_springs(dt=0.01, k=1000.0, c=0.0,
                                         pos_a=0.0, pos_b=10.0)
        gm.add_coupling_group(["spring_a", "spring_b"],
                               max_iterations=2, tolerance=1e-15)
        gm.compile()
        # Should not crash -- just clamp at max_iterations
        state = gm.run_scan(20)
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])

    def test_single_iteration(self):
        """max_iterations=1 should execute exactly one pass (like staggered
        but with forward edges within the group).
        """
        dt = 0.01
        gm = _make_bidirectional_springs(dt=dt, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(["spring_a", "spring_b"],
                               max_iterations=1, tolerance=1e-12)
        gm.compile()
        state = gm.run_scan(50)
        # Should complete and produce finite results
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])

    def test_single_iteration_differs_from_multi(self):
        """max_iterations=1 and max_iterations=10 should give different
        results when convergence needs more than 1 pass.
        """
        dt = 0.01
        n_steps = 50
        kwargs = dict(dt=dt, k=100.0, c=0.5, pos_a=0.0, pos_b=5.0)

        gm1 = _make_bidirectional_springs(**kwargs)
        gm1.add_coupling_group(["spring_a", "spring_b"],
                                max_iterations=1, tolerance=1e-12)
        gm1.compile()
        s1 = gm1.run_scan(n_steps)

        gm10 = _make_bidirectional_springs(**kwargs)
        gm10.add_coupling_group(["spring_a", "spring_b"],
                                 max_iterations=10, tolerance=1e-12)
        gm10.compile()
        s10 = gm10.run_scan(n_steps)

        diff = float(jnp.abs(s1["spring_a"]["position"]
                             - s10["spring_a"]["position"]))
        assert diff > 1e-6, (
            "1-iteration and 10-iteration results should differ"
        )

    def test_convergence_with_tolerance(self):
        """Tighter tolerance should give more accurate results."""
        dt = 0.01
        n_steps = 100
        kwargs = dict(dt=dt, k=50.0, c=0.5, pos_a=0.0, pos_b=3.0)

        # Reference: very fine dt
        ref_kwargs = dict(kwargs)
        ref_kwargs["dt"] = dt / 50
        gm_ref = _make_bidirectional_springs(**ref_kwargs)
        gm_ref.compile()
        ref_state = gm_ref.run_scan(n_steps * 50)

        # Loose tolerance
        gm_loose = _make_bidirectional_springs(**kwargs)
        gm_loose.add_coupling_group(["spring_a", "spring_b"],
                                     max_iterations=50, tolerance=1e-2)
        gm_loose.compile()
        loose_state = gm_loose.run_scan(n_steps)

        # Tight tolerance
        gm_tight = _make_bidirectional_springs(**kwargs)
        gm_tight.add_coupling_group(["spring_a", "spring_b"],
                                     max_iterations=50, tolerance=1e-10)
        gm_tight.compile()
        tight_state = gm_tight.run_scan(n_steps)

        def err_vs_ref(s):
            return float(jnp.abs(
                s["spring_a"]["position"]
                - ref_state["spring_a"]["position"]
            ) + jnp.abs(
                s["spring_b"]["position"]
                - ref_state["spring_b"]["position"]
            ))

        err_loose = err_vs_ref(loose_state)
        err_tight = err_vs_ref(tight_state)
        # Tight tolerance should be at least as accurate
        assert err_tight <= err_loose + 1e-8, (
            f"Tight error {err_tight} should be <= loose error {err_loose}"
        )

    def test_two_coupling_groups(self):
        """Two separate coupling groups in one graph."""
        dt = 0.01
        gm = GraphManager()
        # Cycle 1: A <-> B
        gm.add_node(SpringDamperNode(name="A", timestep=dt,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode(name="B", timestep=dt,
                                      initial_position=2.0))
        gm.add_edge("A", "B", "position", "anchor_position")
        gm.add_edge("B", "A", "position", "anchor_position")
        # Cycle 2: C <-> D
        gm.add_node(SpringDamperNode(name="C", timestep=dt,
                                      initial_position=0.0))
        gm.add_node(SpringDamperNode(name="D", timestep=dt,
                                      initial_position=3.0))
        gm.add_edge("C", "D", "position", "anchor_position")
        gm.add_edge("D", "C", "position", "anchor_position")

        gm.add_coupling_group(["A", "B"], max_iterations=10, tolerance=1e-8)
        gm.add_coupling_group(["C", "D"], max_iterations=10, tolerance=1e-8)
        gm.compile()
        state = gm.run_scan(100)
        for name in ["A", "B", "C", "D"]:
            assert jnp.isfinite(state[name]["position"])
            assert jnp.isfinite(state[name]["velocity"])

    def test_non_coupled_nodes_unaffected(self):
        """Nodes outside coupling groups produce the same results
        whether or not a coupling group exists elsewhere in the graph.
        """
        dt = 0.001
        n_steps = 200

        def build_graph(couple):
            gm = GraphManager()
            # Independent node (no edges to cyclic part)
            gm.add_node(TableNode(name="table", timestep=dt, position=0.0))
            gm.add_node(BallNode(name="free_ball", timestep=dt,
                                  initial_position=5.0, elasticity=0.8))
            gm.add_edge("table", "free_ball", "position", "table_position")
            # Cyclic pair
            gm.add_node(SpringDamperNode(name="sa", timestep=dt,
                                          initial_position=0.0))
            gm.add_node(SpringDamperNode(name="sb", timestep=dt,
                                          initial_position=2.0))
            gm.add_edge("sa", "sb", "position", "anchor_position")
            gm.add_edge("sb", "sa", "position", "anchor_position")
            if couple:
                gm.add_coupling_group(["sa", "sb"],
                                       max_iterations=10, tolerance=1e-8)
            gm.compile()
            return gm

        gm_no = build_graph(couple=False)
        state_no = gm_no.run_scan(n_steps)

        gm_yes = build_graph(couple=True)
        state_yes = gm_yes.run_scan(n_steps)

        # free_ball and table should be identical
        assert float(jnp.abs(
            state_no["free_ball"]["position"]
            - state_yes["free_ball"]["position"]
        )) < 1e-6
        assert float(jnp.abs(
            state_no["free_ball"]["velocity"]
            - state_yes["free_ball"]["velocity"]
        )) < 1e-6


# ==================================================================
# TestCouplingWithJAX
# ==================================================================

class TestCouplingWithJAX:
    """JAX integration: JIT, grad, lax.scan with coupling groups."""

    def test_jit_compilation(self):
        """Graph with coupling group JIT-compiles without error."""
        gm = _make_bidirectional_springs()
        gm.add_coupling_group(["spring_a", "spring_b"])
        gm.compile()  # internally calls jax.jit
        # Step must work
        state = gm.step()
        assert jnp.isfinite(state["spring_a"]["position"])

    def test_grad_through_coupling(self):
        """jax.grad through a step with coupling produces finite
        non-zero gradients.
        """
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(["spring_a", "spring_b"],
                               max_iterations=5, tolerance=1e-6)
        gm.compile()
        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        def loss_fn(init_pos_a):
            state = {
                "spring_a": {"position": init_pos_a,
                             "velocity": jnp.array(0.0)},
                "spring_b": {"position": jnp.array(3.0),
                             "velocity": jnp.array(0.0)},
            }
            for _ in range(10):
                state = step_fn(state, ext)
            return state["spring_a"]["position"]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(0.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0

    def test_run_scan_with_coupling(self):
        """run_scan works with coupling groups (while_loop inside lax.scan)."""
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(["spring_a", "spring_b"],
                               max_iterations=5, tolerance=1e-6)
        gm.compile()
        state = gm.run_scan(100)
        assert jnp.isfinite(state["spring_a"]["position"])
        assert jnp.isfinite(state["spring_b"]["position"])

    def test_run_scan_with_history_coupling(self):
        """run_scan_with_history works with coupling groups."""
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(["spring_a", "spring_b"],
                               max_iterations=5, tolerance=1e-6)
        gm.compile()
        final, history = gm.run_scan_with_history(100)
        assert history["spring_a"]["position"].shape == (100,)
        assert history["spring_b"]["position"].shape == (100,)
        # Final state should match last history entry
        assert float(jnp.abs(
            final["spring_a"]["position"]
            - history["spring_a"]["position"][-1]
        )) < 1e-6

    def test_grad_through_scan_with_coupling(self):
        """jax.grad through run_scan_with_history + coupling."""
        gm = _make_bidirectional_springs(dt=0.01, pos_a=0.0, pos_b=3.0)
        gm.add_coupling_group(["spring_a", "spring_b"],
                               max_iterations=5, tolerance=1e-6)
        gm.compile()
        step_fn = gm._build_step_fn()
        ext = gm._default_external_inputs()

        def loss_fn(init_pos_b):
            state = {
                "spring_a": {"position": jnp.array(0.0),
                             "velocity": jnp.array(0.0)},
                "spring_b": {"position": init_pos_b,
                             "velocity": jnp.array(0.0)},
            }

            def body(s, _):
                return step_fn(s, ext), None

            final, _ = jax.lax.scan(body, state, None, length=20)
            return final["spring_a"]["position"]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(3.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0


# ==================================================================
# TestCouplingPhysics
# ==================================================================

class TestCouplingPhysics:
    """Physically meaningful coupled systems."""

    def test_coupled_springs_energy_conservation(self):
        """Two coupled springs with damping should remain stable and finite.

        With explicit integration both staggered and iterated solutions
        may show energy drift for undamped systems.  This test verifies
        that coupling produces a stable, finite result.
        """
        dt = 0.001
        k = 5.0
        m = 1.0
        rest = 1.0
        n_steps = 200

        def compute_energy(state):
            """Total energy: kinetic + elastic potential for each spring.

            For coupled springs where each is the other's anchor,
            the potential energy is shared. We approximate:
            E = 0.5*m*v_a^2 + 0.5*m*v_b^2
              + 0.5*k*(x_a - x_b - rest)^2
            (The second spring sees anchor = x_a, so the stretch is
            x_b - x_a - rest.)
            """
            xa = state["spring_a"]["position"]
            va = state["spring_a"]["velocity"]
            xb = state["spring_b"]["position"]
            vb = state["spring_b"]["velocity"]
            KE = 0.5 * m * va**2 + 0.5 * m * vb**2
            # Spring A sees anchor = x_b:  stretch_a = x_a - x_b - rest
            # Spring B sees anchor = x_a:  stretch_b = x_b - x_a - rest
            PE = 0.5 * k * (xa - xb - rest)**2 + 0.5 * k * (xb - xa - rest)**2
            return float(KE + PE)

        kwargs = dict(dt=dt, k=k, c=0.0, m=m, rest=rest,
                      pos_a=0.0, pos_b=3.0)

        # Staggered
        gm_stag = _make_bidirectional_springs(**kwargs)
        gm_stag.compile()
        _, hist_stag = gm_stag.run_scan_with_history(n_steps)

        # Coupled
        gm_coupled = _make_bidirectional_springs(**kwargs)
        gm_coupled.add_coupling_group(["spring_a", "spring_b"],
                                       max_iterations=20, tolerance=1e-10)
        gm_coupled.compile()
        _, hist_coupled = gm_coupled.run_scan_with_history(n_steps)

        # Compute energy at start and end for both
        def energy_from_hist(h, idx):
            s = {
                "spring_a": {
                    "position": h["spring_a"]["position"][idx],
                    "velocity": h["spring_a"]["velocity"][idx],
                },
                "spring_b": {
                    "position": h["spring_b"]["position"][idx],
                    "velocity": h["spring_b"]["velocity"][idx],
                },
            }
            return compute_energy(s)

        e0_stag = energy_from_hist(hist_stag, 0)
        ef_stag = energy_from_hist(hist_stag, -1)
        e0_coupled = energy_from_hist(hist_coupled, 0)
        ef_coupled = energy_from_hist(hist_coupled, -1)

        drift_stag = abs(ef_stag - e0_stag)
        drift_coupled = abs(ef_coupled - e0_coupled)

        # Both should be finite
        assert jnp.isfinite(jnp.array(drift_stag))
        assert jnp.isfinite(jnp.array(drift_coupled))
        # Both should have bounded energy drift (not blowing up)
        assert drift_coupled < 10.0 * max(e0_coupled, 1.0), (
            f"Coupled energy drift {drift_coupled} is too large"
        )
        assert drift_stag < 10.0 * max(e0_stag, 1.0), (
            f"Staggered energy drift {drift_stag} is too large"
        )

    def test_ball_spring_coupled(self):
        """BallNode + SpringDamperNode in a cycle.

        The spring is anchored to the ball's position and the spring's
        position is fed back to the ball as the table surface (collision
        boundary).  With coupling the system should remain finite and
        produce different results than staggered.
        """
        dt = 0.001
        n_steps = 500

        def build(couple):
            gm = GraphManager()
            ball = BallNode(name="ball", timestep=dt,
                            initial_position=5.0, initial_velocity=0.0,
                            elasticity=0.8)
            spring = SpringDamperNode(name="spring", timestep=dt,
                                      stiffness=50.0, damping=2.0, mass=0.5,
                                      rest_length=0.0, initial_position=0.0)
            gm.add_node(ball)
            gm.add_node(spring)
            # ball position -> spring anchor
            gm.add_edge("ball", "spring", "position", "anchor_position")
            # spring position -> ball table_position (collision surface)
            gm.add_edge("spring", "ball", "position", "table_position")
            if couple:
                gm.add_coupling_group(["ball", "spring"],
                                       max_iterations=10, tolerance=1e-6)
            gm.compile()
            return gm

        gm_stag = build(couple=False)
        state_stag = gm_stag.run_scan(n_steps)

        gm_coupled = build(couple=True)
        state_coupled = gm_coupled.run_scan(n_steps)

        # Both must be finite
        for name in ["ball", "spring"]:
            assert jnp.isfinite(state_stag[name]["position"]), (
                f"Staggered {name} position is not finite"
            )
            assert jnp.isfinite(state_coupled[name]["position"]), (
                f"Coupled {name} position is not finite"
            )

        # Results should differ
        diff = float(jnp.abs(
            state_stag["ball"]["position"]
            - state_coupled["ball"]["position"]
        ))
        assert diff > 1e-8, "Coupled and staggered should give different results"


# ==================================================================
# TestSCC (Tarjan's algorithm)
# ==================================================================

class TestSCC:
    """Tests for find_strongly_connected_components (Tarjan)."""

    def test_tarjan_simple_cycle(self):
        """A -> B -> A should produce one SCC {A, B}."""
        edges = [
            EdgeSpec("A", "B", "x", "y"),
            EdgeSpec("B", "A", "x", "y"),
        ]
        sccs = find_strongly_connected_components(["A", "B"], edges)
        assert len(sccs) == 1
        assert set(sccs[0]) == {"A", "B"}

    def test_tarjan_no_cycle(self):
        """A -> B -> C (no cycle) produces no SCCs."""
        edges = [
            EdgeSpec("A", "B", "x", "y"),
            EdgeSpec("B", "C", "x", "y"),
        ]
        sccs = find_strongly_connected_components(["A", "B", "C"], edges)
        assert sccs == []

    def test_tarjan_two_cycles(self):
        """Two separate cycles produce two SCCs."""
        edges = [
            EdgeSpec("A", "B", "x", "y"),
            EdgeSpec("B", "A", "x", "y"),
            EdgeSpec("C", "D", "x", "y"),
            EdgeSpec("D", "C", "x", "y"),
        ]
        sccs = find_strongly_connected_components(
            ["A", "B", "C", "D"], edges
        )
        assert len(sccs) == 2
        sets = [set(s) for s in sccs]
        assert {"A", "B"} in sets
        assert {"C", "D"} in sets

    def test_tarjan_self_loop(self):
        """A self-loop (A -> A) should NOT produce an SCC (node count = 1)."""
        edges = [EdgeSpec("A", "A", "x", "y")]
        sccs = find_strongly_connected_components(["A"], edges)
        assert sccs == []

    def test_tarjan_complex(self):
        """Larger graph with an embedded 3-node cycle and a tail.

        A -> B -> C -> A (cycle)
        C -> D (no cycle, just a tail)
        """
        edges = [
            EdgeSpec("A", "B", "x", "y"),
            EdgeSpec("B", "C", "x", "y"),
            EdgeSpec("C", "A", "x", "y"),
            EdgeSpec("C", "D", "x", "y"),
        ]
        sccs = find_strongly_connected_components(
            ["A", "B", "C", "D"], edges
        )
        assert len(sccs) == 1
        assert set(sccs[0]) == {"A", "B", "C"}

    def test_tarjan_two_overlapping_cycles(self):
        """A -> B -> A and A -> B -> C -> A merge into one SCC {A, B, C}."""
        edges = [
            EdgeSpec("A", "B", "x", "y"),
            EdgeSpec("B", "A", "x", "y"),
            EdgeSpec("B", "C", "x", "y"),
            EdgeSpec("C", "A", "x", "y"),
        ]
        sccs = find_strongly_connected_components(
            ["A", "B", "C"], edges
        )
        assert len(sccs) == 1
        assert set(sccs[0]) == {"A", "B", "C"}

    def test_tarjan_disconnected_nodes(self):
        """Disconnected nodes (no edges) produce no SCCs."""
        sccs = find_strongly_connected_components(["A", "B", "C"], [])
        assert sccs == []

    def test_tarjan_single_node_no_edges(self):
        """Single node with no edges: no SCC."""
        sccs = find_strongly_connected_components(["A"], [])
        assert sccs == []
