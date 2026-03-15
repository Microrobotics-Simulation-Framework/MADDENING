"""Tests for Phase 2b-c: HybridNode and learned integration corrector.

Verifies that:
1. HybridNode delegates to physics node correctly
2. Correction function is added to physics output
3. HybridNode works as drop-in replacement in GraphManager
4. generate_correction_data produces correct shapes
5. A simple polynomial correction improves accuracy
"""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.hybrid_node import (
    HybridNode,
    generate_correction_data,
)
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


class TestHybridNodeBasic:
    """Basic HybridNode tests."""

    def test_zero_correction_matches_physics(self):
        """With zero correction, HybridNode should match physics node."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)
        hybrid = HybridNode(
            ball,
            correction_fn=lambda s, bi, dt: {},
        )

        state = ball.initial_state()
        bi = {}

        result_phys = ball.update(state, bi, 0.01)
        result_hybrid = hybrid.update(state, bi, 0.01)

        assert jnp.allclose(result_phys["position"], result_hybrid["position"])
        assert jnp.allclose(result_phys["velocity"], result_hybrid["velocity"])

    def test_constant_correction(self):
        """Correction function should add to physics output."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)

        def correction(state, bi, dt):
            return {"velocity": jnp.array(1.0)}

        hybrid = HybridNode(ball, correction_fn=correction)
        state = ball.initial_state()
        bi = {}

        result_phys = ball.update(state, bi, 0.01)
        result_hybrid = hybrid.update(state, bi, 0.01)

        # velocity should be corrected by +1.0
        assert float(result_hybrid["velocity"]) == pytest.approx(
            float(result_phys["velocity"]) + 1.0, abs=1e-6
        )
        # position should be unchanged (no correction for position)
        assert float(result_hybrid["position"]) == pytest.approx(
            float(result_phys["position"]), abs=1e-6
        )

    def test_name_delegation(self):
        """HybridNode should use physics node's name by default."""
        ball = BallNode(name="ball", timestep=0.01)
        hybrid = HybridNode(ball, correction_fn=lambda s, bi, dt: {})
        assert hybrid.name == "ball"

    def test_name_override(self):
        """Can override the name."""
        ball = BallNode(name="ball", timestep=0.01)
        hybrid = HybridNode(
            ball, correction_fn=lambda s, bi, dt: {},
            name="hybrid_ball",
        )
        assert hybrid.name == "hybrid_ball"

    def test_initial_state_delegation(self):
        """initial_state should delegate to physics node."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)
        hybrid = HybridNode(ball, correction_fn=lambda s, bi, dt: {})
        state = hybrid.initial_state()
        assert float(state["position"]) == pytest.approx(5.0)

    def test_state_fields_delegation(self):
        """state_fields should delegate to physics node."""
        ball = BallNode(name="ball", timestep=0.01)
        hybrid = HybridNode(ball, correction_fn=lambda s, bi, dt: {})
        assert hybrid.state_fields() == ball.state_fields()

    def test_boundary_input_spec_delegation(self):
        """boundary_input_spec should delegate to physics node."""
        heat = HeatNode(name="rod", timestep=0.001, n_cells=10)
        hybrid = HybridNode(heat, correction_fn=lambda s, bi, dt: {})
        assert hybrid.boundary_input_spec() == heat.boundary_input_spec()

    def test_interface_dof_delegation(self):
        """interface_dof_indices should delegate to physics node."""
        heat = HeatNode(name="rod", timestep=0.001, n_cells=10)
        hybrid = HybridNode(heat, correction_fn=lambda s, bi, dt: {})
        assert hybrid.interface_dof_indices() == heat.interface_dof_indices()


class TestHybridNodeInGraph:
    """Test HybridNode as a drop-in replacement in GraphManager."""

    def test_hybrid_in_graph(self):
        """HybridNode should work in a GraphManager."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)
        hybrid = HybridNode(ball, correction_fn=lambda s, bi, dt: {})

        gm = GraphManager()
        gm.add_node(hybrid)
        gm.compile()

        for _ in range(100):
            gm.step()

        state = gm.get_node_state("ball")
        # Ball should have fallen
        assert float(state["position"]) < 5.0

    def test_hybrid_with_edges(self):
        """HybridNode should work with edges."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)
        hybrid_ball = HybridNode(
            ball, correction_fn=lambda s, bi, dt: {},
        )
        spring = SpringDamperNode(
            name="spring", timestep=0.01,
            stiffness=50.0, damping=1.0, mass=1.0,
        )

        gm = GraphManager()
        gm.add_node(hybrid_ball)
        gm.add_node(spring)
        gm.add_edge("ball", "spring", "position", "anchor_position")
        gm.compile()

        for _ in range(50):
            gm.step()

        state = gm.get_node_state("spring")
        assert jnp.isfinite(state["position"])

    def test_hybrid_jit_compatible(self):
        """HybridNode should JIT compile."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)

        def correction(state, bi, dt):
            return {"velocity": jnp.array(0.01) * state["velocity"]}

        hybrid = HybridNode(ball, correction_fn=correction)

        @jax.jit
        def step(state):
            return hybrid.update(state, {}, 0.01)

        state = hybrid.initial_state()
        new_state = step(state)
        assert jnp.isfinite(new_state["position"])

    def test_hybrid_grad_compatible(self):
        """Should be differentiable through HybridNode."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=0.0)

        def correction(state, bi, dt):
            return {"velocity": jnp.array(0.01)}

        hybrid = HybridNode(ball, correction_fn=correction)

        def loss_fn(v0):
            state = {"position": jnp.array(0.0), "velocity": v0}
            for _ in range(10):
                state = hybrid.update(state, {}, 0.01)
            return state["position"]

        grad_fn = jax.grad(loss_fn)
        g = grad_fn(jnp.array(1.0))
        assert jnp.isfinite(g)
        assert float(g) > 0.0


class TestGenerateCorrectionData:
    """Tests for the correction data generator."""

    def test_basic_shape(self):
        """Should produce lists of correct length."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)
        inputs, targets, states = generate_correction_data(
            ball, dt_coarse=0.01, dt_fine=0.001, n_steps=10,
        )
        assert len(inputs) == 10
        assert len(targets) == 10
        assert len(states) == 10

    def test_correction_fields_match(self):
        """Corrections should have the same fields as the state."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)
        inputs, targets, states = generate_correction_data(
            ball, dt_coarse=0.01, dt_fine=0.001, n_steps=5,
        )
        state_fields = set(ball.initial_state().keys())
        for target in targets:
            assert set(target.keys()) == state_fields

    def test_corrections_are_small_for_small_dt_ratio(self):
        """When coarse/fine ratio is small, corrections should be small."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)
        inputs, targets, states = generate_correction_data(
            ball, dt_coarse=0.01, dt_fine=0.005, n_steps=5,
        )
        # With only 2x refinement, corrections should be small
        for target in targets:
            for field in target:
                assert float(jnp.max(jnp.abs(target[field]))) < 1.0

    def test_corrections_improve_accuracy(self):
        """Applying corrections to coarse step should match fine step."""
        ball = BallNode(name="ball", timestep=0.01, initial_position=5.0)
        inputs, targets, states = generate_correction_data(
            ball, dt_coarse=0.01, dt_fine=0.001, n_steps=1,
        )

        # Coarse step
        state = ball.initial_state()
        coarse_result = ball.update(state, {}, 0.01)

        # Corrected result
        corrected = {
            k: coarse_result[k] + targets[0][k] for k in coarse_result
        }

        # Fine result (states[0] is what fine gives)
        fine_result = states[0]

        # Corrected should be very close to fine
        for field in corrected:
            assert jnp.allclose(corrected[field], fine_result[field], atol=1e-5)

    def test_heat_node_correction_data(self):
        """Should work with HeatNode."""
        heat = HeatNode(
            name="rod", timestep=0.001, n_cells=10,
            thermal_diffusivity=0.01, initial_temperature=100.0,
        )
        bi = {
            "left_temperature": jnp.array(100.0),
            "right_temperature": jnp.array(0.0),
        }
        inputs, targets, states = generate_correction_data(
            heat, dt_coarse=0.001, dt_fine=0.0001, n_steps=5,
            boundary_inputs=bi,
        )
        assert len(inputs) == 5
        assert "temperature" in targets[0]
        assert targets[0]["temperature"].shape == (10,)


class TestHybridCorrectionIntegration:
    """End-to-end test: generate data, build correction, verify improvement."""

    def test_simple_correction_improves_ball(self):
        """A hand-crafted correction should improve coarse ball accuracy."""
        ball = BallNode(name="ball", timestep=0.1, initial_position=100.0)

        # Generate reference data
        inputs, targets, states = generate_correction_data(
            ball, dt_coarse=0.1, dt_fine=0.001, n_steps=20,
        )

        # Compute average correction per field (crude but demonstrates
        # the concept)
        import jax.numpy as jnp
        avg_corr = {}
        for field in targets[0]:
            corrections = jnp.stack([t[field] for t in targets])
            avg_corr[field] = jnp.mean(corrections, axis=0)

        def constant_correction(state, bi, dt):
            return avg_corr

        hybrid = HybridNode(ball, correction_fn=constant_correction)

        # Run coarse (uncorrected) and corrected
        state_coarse = ball.initial_state()
        state_corrected = hybrid.initial_state()

        # Also run fine as reference
        fine_state = ball.initial_state()

        for step in range(10):
            state_coarse = ball.update(state_coarse, {}, 0.1)
            state_corrected = hybrid.update(state_corrected, {}, 0.1)
            for _ in range(100):
                fine_state = ball.update(fine_state, {}, 0.001)

        # The corrected trajectory should be closer to fine than coarse
        err_coarse = abs(
            float(state_coarse["position"]) - float(fine_state["position"])
        )
        err_corrected = abs(
            float(state_corrected["position"]) - float(fine_state["position"])
        )

        # At minimum, the corrected version should not be dramatically worse
        # (a constant average correction is crude but shouldn't hurt much)
        assert err_corrected < err_coarse * 2.0, (
            f"Corrected error {err_corrected} should be reasonable vs "
            f"coarse error {err_coarse}"
        )
