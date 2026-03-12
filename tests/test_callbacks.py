"""Tests for surrogate training callbacks."""

import os
import tempfile

import jax
import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.surrogates.callbacks import (
    TrainingCallback,
    EarlyStopping,
    ModelCheckpoint,
    LRSchedule,
)
from maddening.surrogates.dataset import DatasetGenerator
from maddening.surrogates.trainer import SurrogateTrainer
from maddening.surrogates.architectures.mlp import MLPDirect


@pytest.fixture
def ball_dataset():
    """Generate a small dataset from a free-falling ball."""
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
    gm.compile()
    return DatasetGenerator.from_graph(gm, "ball", n_steps=100)


class TestEarlyStopping:
    def test_stops_after_patience(self, ball_dataset):
        arch = MLPDirect(hidden_sizes=(16,))
        trainer = SurrogateTrainer(arch, ball_dataset)
        # Use a large min_delta to force early stop (loss decreases slowly)
        early = EarlyStopping(patience=3, min_delta=10.0)

        result = trainer.train(
            n_epochs=200, batch_size=32,
            rng_key=jax.random.PRNGKey(42),
            callbacks=[early],
        )

        # Should stop early since improvements < 10.0 per epoch after a while
        assert len(result.train_losses) < 200

    def test_restores_best_weights(self, ball_dataset):
        arch = MLPDirect(hidden_sizes=(16,))
        trainer = SurrogateTrainer(arch, ball_dataset)
        early = EarlyStopping(patience=5, restore_best=True)

        result = trainer.train(
            n_epochs=50, batch_size=32,
            rng_key=jax.random.PRNGKey(42),
            callbacks=[early],
        )

        # Best epoch should be tracked
        assert early.best_epoch >= 0

    def test_no_stop_if_improving(self, ball_dataset):
        arch = MLPDirect(hidden_sizes=(16,))
        trainer = SurrogateTrainer(arch, ball_dataset)
        early = EarlyStopping(patience=1000, min_delta=0.0)

        result = trainer.train(
            n_epochs=5, batch_size=32,
            rng_key=jax.random.PRNGKey(42),
            callbacks=[early],
        )

        # Should complete all epochs since patience is huge
        assert len(result.train_losses) == 5


class TestModelCheckpoint:
    def test_saves_checkpoint(self, ball_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.npz")
            arch = MLPDirect(hidden_sizes=(16,))
            trainer = SurrogateTrainer(arch, ball_dataset)
            ckpt = ModelCheckpoint(path=path, save_best_only=True)

            result = trainer.train(
                n_epochs=5, batch_size=32,
                rng_key=jax.random.PRNGKey(42),
                callbacks=[ckpt],
            )

            assert os.path.exists(path)

    def test_saves_per_epoch(self, ball_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model_e{epoch}.npz")
            arch = MLPDirect(hidden_sizes=(16,))
            trainer = SurrogateTrainer(arch, ball_dataset)
            ckpt = ModelCheckpoint(path=path, save_best_only=False)

            result = trainer.train(
                n_epochs=3, batch_size=32,
                rng_key=jax.random.PRNGKey(42),
                callbacks=[ckpt],
            )

            for e in range(3):
                ep = os.path.join(tmpdir, f"model_e{e}.npz")
                assert os.path.exists(ep)


class TestLRSchedule:
    def test_lr_multiplier_updates(self, ball_dataset):
        schedule = LRSchedule(lambda epoch: max(0.1, 1.0 - epoch * 0.1))
        arch = MLPDirect(hidden_sizes=(16,))
        trainer = SurrogateTrainer(arch, ball_dataset)

        result = trainer.train(
            n_epochs=5, batch_size=32,
            rng_key=jax.random.PRNGKey(42),
            callbacks=[schedule],
        )

        # LR should have decreased
        assert schedule.lr_multiplier < 1.0
        assert len(result.train_losses) == 5


class TestCallbackComposition:
    def test_early_stopping_with_checkpoint(self, ball_dataset):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "best.npz")
            arch = MLPDirect(hidden_sizes=(16,))
            trainer = SurrogateTrainer(arch, ball_dataset)

            result = trainer.train(
                n_epochs=100, batch_size=32,
                rng_key=jax.random.PRNGKey(42),
                callbacks=[
                    EarlyStopping(patience=3, min_delta=5.0),
                    ModelCheckpoint(path=path),
                ],
            )

            assert os.path.exists(path)
            assert len(result.train_losses) < 100

    def test_callback_with_simple_callback(self, ball_dataset):
        """Callbacks work alongside the simple callback= parameter."""
        epochs_seen = []
        arch = MLPDirect(hidden_sizes=(16,))
        trainer = SurrogateTrainer(arch, ball_dataset)

        result = trainer.train(
            n_epochs=5, batch_size=32,
            rng_key=jax.random.PRNGKey(42),
            callback=lambda e, m: epochs_seen.append(e),
            callbacks=[EarlyStopping(patience=100)],
        )

        assert len(epochs_seen) == 5
