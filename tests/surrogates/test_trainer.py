"""Tests for SurrogateTrainer -- convergence on simple problems."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.surrogates.dataset import DatasetGenerator
from maddening.surrogates.training.trainer import SurrogateTrainer, TrainResult, mse_loss
from maddening.surrogates.architectures.mlp import MLPDirect, MLPDerivative


class TestTrainerBasic:
    def _make_dataset(self, n_steps=200):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()
        return DatasetGenerator.from_graph(gm, "ball", n_steps=n_steps)

    def test_training_reduces_loss(self):
        ds = self._make_dataset(n_steps=200)
        arch = MLPDirect(hidden_sizes=(32, 32))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(
            n_epochs=30, batch_size=32,
            rng_key=jax.random.PRNGKey(42),
        )

        assert isinstance(result, TrainResult)
        assert len(result.train_losses) == 30
        assert len(result.val_losses) == 30
        # Training loss should decrease
        assert result.train_losses[-1] < result.train_losses[0]

    def test_callback_is_called(self):
        ds = self._make_dataset(n_steps=100)
        arch = MLPDirect(hidden_sizes=(16,))
        trainer = SurrogateTrainer(arch, ds)

        epochs_seen = []
        def cb(epoch, metrics):
            epochs_seen.append(epoch)
            assert "train_loss" in metrics
            assert "val_loss" in metrics

        trainer.train(n_epochs=5, batch_size=32, rng_key=jax.random.PRNGKey(0),
                      callback=cb)
        assert epochs_seen == [0, 1, 2, 3, 4]

    def test_to_node(self):
        ds = self._make_dataset(n_steps=100)
        arch = MLPDirect(hidden_sizes=(16,))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(n_epochs=3, batch_size=32,
                               rng_key=jax.random.PRNGKey(0))

        node = result.to_node(
            name="ball", timestep=0.01,
            initial_values={"position": 10.0, "velocity": 0.0},
        )
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert jnp.isfinite(new_state["position"])
        assert jnp.isfinite(new_state["velocity"])


class TestDerivativeTrainer:
    def test_derivative_mode_training(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()
        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=200)

        arch = MLPDerivative(hidden_sizes=(32, 32))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(
            n_epochs=30, batch_size=32,
            rng_key=jax.random.PRNGKey(42),
        )

        # Training should converge somewhat
        assert result.train_losses[-1] < result.train_losses[0]


class TestMSELoss:
    def test_mse_zero(self):
        pred = {"x": jnp.array(1.0)}
        target = {"x": jnp.array(1.0)}
        assert float(mse_loss(pred, target)) == pytest.approx(0.0, abs=1e-7)

    def test_mse_nonzero(self):
        pred = {"x": jnp.array(0.0)}
        target = {"x": jnp.array(1.0)}
        assert float(mse_loss(pred, target)) == pytest.approx(1.0)

    def test_mse_multi_field(self):
        pred = {"a": jnp.array(0.0), "b": jnp.array(0.0)}
        target = {"a": jnp.array(1.0), "b": jnp.array(1.0)}
        # Two fields, each has error 1.0, total sum = 2.0, count = 2
        assert float(mse_loss(pred, target)) == pytest.approx(1.0)
