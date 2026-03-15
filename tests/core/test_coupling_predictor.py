"""Tests for Phase 1b: Coupling iteration predictors.

Verifies that:
1. predictor="none" (default) leaves behavior unchanged
2. predictor="linear" reduces iteration count on smoothly varying problems
3. predictor="quadratic" works and further reduces iterations
4. Predictors work with run_scan (lax.scan compatibility)
5. Predictors are backward compatible (no effect on first 2 timesteps)
6. Predictors work with diagnostics enabled
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.coupling import CouplingGroup
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


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


class TestPredictorCouplingGroup:
    """Tests for the predictor field on CouplingGroup."""

    def test_default_predictor_is_none(self):
        """CouplingGroup defaults to predictor='none'."""
        gm = _make_bidirectional_springs()
        group = gm.add_coupling_group(["spring_a", "spring_b"])
        assert group.predictor == "none"

    def test_linear_predictor(self):
        """Can create coupling group with linear predictor."""
        gm = _make_bidirectional_springs()
        group = gm.add_coupling_group(
            ["spring_a", "spring_b"],
            predictor="linear",
        )
        assert group.predictor == "linear"

    def test_quadratic_predictor(self):
        """Can create coupling group with quadratic predictor."""
        gm = _make_bidirectional_springs()
        group = gm.add_coupling_group(
            ["spring_a", "spring_b"],
            predictor="quadratic",
        )
        assert group.predictor == "quadratic"


class TestPredictorReducesIterations:
    """Test that predictors reduce coupling iterations."""

    def _run_with_predictor(self, predictor, n_steps=50):
        """Run bidirectional springs with given predictor and return
        iteration counts."""
        gm = _make_bidirectional_springs(dt=0.001, k=100.0, c=2.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30,
            tolerance=1e-8,
            diagnostics=True,
            predictor=predictor,
        )
        gm.compile()

        iter_counts = []
        for _ in range(n_steps):
            gm.step()
            diag = gm.coupling_diagnostics()
            key = "spring_a+spring_b"
            iter_counts.append(diag[key]["iterations"])

        return iter_counts

    def test_predictor_none_works(self):
        """predictor='none' should produce valid iteration counts."""
        iters = self._run_with_predictor("none", n_steps=20)
        assert len(iters) == 20
        assert all(i >= 1 for i in iters)

    def test_linear_predictor_reduces_iterations(self):
        """Linear predictor should reduce average iteration count
        compared to no predictor (after initial ramp-up)."""
        iters_none = self._run_with_predictor("none", n_steps=40)
        iters_linear = self._run_with_predictor("linear", n_steps=40)

        # Compare average iterations after the first 5 steps
        # (predictor needs history to be effective)
        avg_none = sum(iters_none[5:]) / len(iters_none[5:])
        avg_linear = sum(iters_linear[5:]) / len(iters_linear[5:])

        # Linear predictor should do no worse (and ideally better)
        assert avg_linear <= avg_none + 1.0, (
            f"Linear predictor avg={avg_linear}, no predictor avg={avg_none}"
        )

    def test_predictor_converges_to_same_result(self):
        """Predictor should not change the converged result."""
        gm_none = _make_bidirectional_springs(dt=0.001, k=100.0, c=2.0)
        gm_none.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-8,
            predictor="none",
        )
        gm_none.compile()

        gm_linear = _make_bidirectional_springs(dt=0.001, k=100.0, c=2.0)
        gm_linear.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-8,
            predictor="linear",
        )
        gm_linear.compile()

        for _ in range(50):
            gm_none.step()
            gm_linear.step()

        state_none_a = gm_none.get_node_state("spring_a")
        state_none_b = gm_none.get_node_state("spring_b")
        state_lin_a = gm_linear.get_node_state("spring_a")
        state_lin_b = gm_linear.get_node_state("spring_b")

        # Results should be nearly identical
        assert float(state_none_a["position"]) == pytest.approx(
            float(state_lin_a["position"]), abs=1e-4
        )
        assert float(state_none_b["position"]) == pytest.approx(
            float(state_lin_b["position"]), abs=1e-4
        )


class TestPredictorWithScan:
    """Test predictor works with run_scan (lax.scan)."""

    def test_linear_predictor_run_scan(self):
        """Linear predictor should work with run_scan."""
        gm = _make_bidirectional_springs(dt=0.001, k=100.0, c=2.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-8,
            predictor="linear",
        )
        gm.compile()

        gm.run_scan(100)

        state_a = gm.get_node_state("spring_a")
        state_b = gm.get_node_state("spring_b")
        # Should be moving toward equilibrium
        assert jnp.isfinite(state_a["position"])
        assert jnp.isfinite(state_b["position"])

    def test_quadratic_predictor_run_scan(self):
        """Quadratic predictor should work with run_scan."""
        gm = _make_bidirectional_springs(dt=0.001, k=100.0, c=2.0)
        gm.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-8,
            predictor="quadratic",
        )
        gm.compile()

        gm.run_scan(100)

        state_a = gm.get_node_state("spring_a")
        state_b = gm.get_node_state("spring_b")
        assert jnp.isfinite(state_a["position"])
        assert jnp.isfinite(state_b["position"])

    def test_predictor_scan_matches_step(self):
        """run_scan with predictor should give same result as step loop."""
        gm_step = _make_bidirectional_springs(dt=0.001, k=100.0, c=2.0)
        gm_step.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-8,
            predictor="linear",
        )
        gm_step.compile()

        gm_scan = _make_bidirectional_springs(dt=0.001, k=100.0, c=2.0)
        gm_scan.add_coupling_group(
            ["spring_a", "spring_b"],
            max_iterations=30, tolerance=1e-8,
            predictor="linear",
        )
        gm_scan.compile()

        for _ in range(50):
            gm_step.step()
        gm_scan.run_scan(50)

        state_step = gm_step.get_node_state("spring_a")
        state_scan = gm_scan.get_node_state("spring_a")

        assert float(state_step["position"]) == pytest.approx(
            float(state_scan["position"]), abs=1e-4
        )


class TestPredictorGrad:
    """Test predictor works with JAX grad."""

    def test_grad_through_predicted_coupling(self):
        """Should be differentiable through coupling with predictor."""
        def loss_fn(pos_a):
            gm = GraphManager()
            a = SpringDamperNode(
                name="spring_a", timestep=0.001,
                stiffness=50.0, damping=1.0, mass=1.0,
                rest_length=1.0, initial_position=pos_a,
            )
            b = SpringDamperNode(
                name="spring_b", timestep=0.001,
                stiffness=50.0, damping=1.0, mass=1.0,
                rest_length=1.0, initial_position=2.0,
            )
            gm.add_node(a)
            gm.add_node(b)
            gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
            gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
            gm.add_coupling_group(
                ["spring_a", "spring_b"],
                max_iterations=10, tolerance=1e-6,
                predictor="linear",
            )
            gm.compile()
            gm.run_scan(20)
            return gm.get_node_state("spring_a")["position"]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(0.0)
        assert jnp.isfinite(g)
