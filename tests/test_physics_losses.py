"""Tests for physics-informed loss functions."""

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.surrogates.architectures.mlp import MLPDirect
from maddening.surrogates.dataset import DatasetGenerator
from maddening.surrogates.physics_losses import (
    residual_loss,
    energy_conservation_loss,
    momentum_conservation_loss,
    smoothness_loss,
    composite_loss,
)
from maddening.surrogates.trainer import SurrogateTrainer


class TestResidualLoss:
    def test_zero_for_perfect_prediction(self):
        def update(state, boundary, dt):
            return {
                "position": state["position"] + state["velocity"] * dt,
                "velocity": state["velocity"] - 9.81 * dt,
            }

        loss_fn = residual_loss(update)
        state = {"position": jnp.array(5.0), "velocity": jnp.array(0.0)}
        boundary = {}
        dt = 0.01

        # Perfect prediction matches the update function
        pred = update(state, boundary, dt)
        loss = loss_fn(None, state, boundary, pred, dt)
        assert float(loss) < 1e-10

    def test_nonzero_for_wrong_prediction(self):
        def update(state, boundary, dt):
            return {"x": state["x"] + 1.0}

        loss_fn = residual_loss(update)
        state = {"x": jnp.array(0.0)}
        pred = {"x": jnp.array(999.0)}
        loss = loss_fn(None, state, {}, pred, 0.01)
        assert float(loss) > 0

    def test_with_trainer(self):
        """Residual loss integrates with SurrogateTrainer."""
        gm = GraphManager()
        ball = BallNode("ball", timestep=0.01, initial_position=10.0)
        gm.add_node(ball)
        gm.compile()
        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=50)

        arch = MLPDirect(hidden_sizes=(16,))
        trainer = SurrogateTrainer(
            arch, ds,
            physics_loss_fn=residual_loss(ball.update),
            physics_loss_weight=0.1,
        )
        result = trainer.train(n_epochs=3, batch_size=16, rng_key=jax.random.PRNGKey(0))
        assert len(result.train_losses) == 3


class TestEnergyConservationLoss:
    def test_conserved_energy_low_loss(self):
        KE = lambda s: 0.5 * s["velocity"] ** 2
        PE = lambda s: 9.81 * s["position"]
        loss_fn = energy_conservation_loss(KE, PE)

        state = {"position": jnp.array(5.0), "velocity": jnp.array(0.0)}
        # Prediction that conserves energy
        pred = {"position": jnp.array(4.0), "velocity": jnp.array(4.429)}  # ~sqrt(2*9.81*1)
        loss = loss_fn(None, state, {}, pred, 0.01)
        # Should be relatively small (not exactly 0 due to discrete approx)
        assert float(loss) < 1.0

    def test_energy_violating_high_loss(self):
        KE = lambda s: 0.5 * s["velocity"] ** 2
        PE = lambda s: 9.81 * s["position"]
        loss_fn = energy_conservation_loss(KE, PE)

        state = {"position": jnp.array(5.0), "velocity": jnp.array(0.0)}
        # Energy-violating prediction
        pred = {"position": jnp.array(10.0), "velocity": jnp.array(10.0)}
        loss = loss_fn(None, state, {}, pred, 0.01)
        assert float(loss) > 10.0


class TestMomentumConservationLoss:
    def test_conserved_momentum(self):
        mom_fn = lambda s: s["velocity"]
        loss_fn = momentum_conservation_loss(mom_fn)

        state = {"velocity": jnp.array(3.0)}
        pred = {"velocity": jnp.array(3.0)}  # same momentum
        loss = loss_fn(None, state, {}, pred, 0.01)
        assert float(loss) < 1e-10

    def test_with_force(self):
        mom_fn = lambda s: s["velocity"]
        force_fn = lambda s, b: jnp.array(-9.81)  # gravity
        loss_fn = momentum_conservation_loss(mom_fn, force_fn)

        state = {"velocity": jnp.array(0.0)}
        dt = 0.01
        # Correct: v_new = v_old + F*dt = 0 + (-9.81)*0.01 = -0.0981
        pred = {"velocity": jnp.array(-0.0981)}
        loss = loss_fn(None, state, {}, pred, dt)
        assert float(loss) < 1e-6


class TestSmoothnessLoss:
    def test_smooth_prediction(self):
        loss_fn = smoothness_loss()
        state = {"x": jnp.array(1.0)}
        pred = {"x": jnp.array(1.001)}  # small change
        loss = loss_fn(None, state, {}, pred, 0.01)
        assert float(loss) < 1.0  # small rate of change

    def test_large_jump_penalized(self):
        loss_fn = smoothness_loss()
        state = {"x": jnp.array(1.0)}
        pred = {"x": jnp.array(100.0)}  # huge jump
        loss = loss_fn(None, state, {}, pred, 0.01)
        assert float(loss) > 1e6  # large penalty


class TestCompositeLoss:
    def test_combines_losses(self):
        KE = lambda s: 0.5 * s["velocity"] ** 2
        PE = lambda s: 9.81 * s["position"]

        combined = composite_loss(
            (energy_conservation_loss(KE, PE), 0.5),
            (smoothness_loss(), 0.1),
        )

        state = {"position": jnp.array(5.0), "velocity": jnp.array(0.0)}
        pred = {"position": jnp.array(4.5), "velocity": jnp.array(3.0)}

        loss = combined(None, state, {}, pred, 0.01)
        assert float(loss) > 0

    def test_gradients_flow(self):
        """Composite loss is differentiable."""
        loss_fn = composite_loss(
            (smoothness_loss(), 1.0),
        )

        state = {"x": jnp.array(1.0)}
        pred = {"x": jnp.array(2.0)}

        # Just verify grad doesn't crash
        def wrapper(x):
            p = {"x": x}
            return loss_fn(None, state, {}, p, 0.01)

        grad = jax.grad(wrapper)(jnp.array(2.0))
        assert jnp.isfinite(grad)
