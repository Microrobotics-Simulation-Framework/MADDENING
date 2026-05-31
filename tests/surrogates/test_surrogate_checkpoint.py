"""Tests for surrogate weight serialization/checkpointing."""

import os
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.surrogates.architectures.mlp import MLPDirect, MLPDerivative
from maddening.surrogates.weights.checkpoint import save_weights, load_weights, load_train_result
from maddening.surrogates.dataset import DatasetGenerator
from maddening.surrogates.training.trainer import SurrogateTrainer, TrainResult


@pytest.fixture
def trained_result():
    """Train a small MLP on ball data and return TrainResult."""
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
    gm.compile()
    ds = DatasetGenerator.from_graph(gm, "ball", n_steps=100)
    arch = MLPDirect(hidden_sizes=(16, 16))
    trainer = SurrogateTrainer(arch, ds)
    return trainer.train(
        n_epochs=5, batch_size=32,
        rng_key=jax.random.PRNGKey(42),
    )


class TestSaveLoadWeights:
    def test_save_creates_file(self, trained_result, tmp_path):
        path = str(tmp_path / "model.npz")
        save_weights(
            path, trained_result.weights,
            architecture=trained_result.architecture,
            state_spec=trained_result.state_spec,
            boundary_spec=trained_result.boundary_spec,
        )
        assert os.path.exists(path)

    def test_load_reconstructs_weights(self, trained_result, tmp_path):
        path = str(tmp_path / "model.npz")
        save_weights(
            path, trained_result.weights,
            architecture=trained_result.architecture,
            state_spec=trained_result.state_spec,
            boundary_spec=trained_result.boundary_spec,
        )

        arch2 = MLPDirect(hidden_sizes=(16, 16))
        loaded_weights, meta = load_weights(path, arch2)

        # Check metadata
        assert meta["architecture_type"] == "MLPDirect"
        assert meta["architecture_mode"] == "direct"
        assert "position" in meta["state_spec"]
        assert "velocity" in meta["state_spec"]

        # Check weight values match
        orig_leaves = jax.tree.leaves(trained_result.weights[0])
        loaded_leaves = jax.tree.leaves(loaded_weights[0])
        assert len(orig_leaves) == len(loaded_leaves)
        for orig, loaded in zip(orig_leaves, loaded_leaves):
            np.testing.assert_allclose(orig, loaded, rtol=1e-6)

    def test_load_produces_same_predictions(self, trained_result, tmp_path):
        path = str(tmp_path / "model.npz")
        save_weights(
            path, trained_result.weights,
            architecture=trained_result.architecture,
            state_spec=trained_result.state_spec,
            boundary_spec=trained_result.boundary_spec,
        )

        arch2 = MLPDirect(hidden_sizes=(16, 16))
        loaded_weights, _ = load_weights(path, arch2)

        # Compare predictions
        test_state = {"position": jnp.array(5.0), "velocity": jnp.array(-3.0)}
        test_boundary = {}

        pred_orig = trained_result.architecture.forward(
            trained_result.weights, test_state, test_boundary, 0.01,
        )
        pred_loaded = arch2.forward(
            loaded_weights, test_state, test_boundary, 0.01,
        )

        for k in pred_orig:
            np.testing.assert_allclose(
                pred_orig[k], pred_loaded[k], rtol=1e-5,
            )

    def test_metadata_preserved(self, trained_result, tmp_path):
        path = str(tmp_path / "model.npz")
        save_weights(
            path, trained_result.weights,
            architecture=trained_result.architecture,
            state_spec=trained_result.state_spec,
            boundary_spec=trained_result.boundary_spec,
            metadata={"epoch": 42, "val_loss": 0.001},
        )

        _, meta = load_weights(path, MLPDirect(hidden_sizes=(16, 16)))
        assert meta["extra"]["epoch"] == 42
        assert abs(meta["extra"]["val_loss"] - 0.001) < 1e-6

    def test_architecture_config_saved(self, trained_result, tmp_path):
        path = str(tmp_path / "model.npz")
        save_weights(
            path, trained_result.weights,
            architecture=trained_result.architecture,
            state_spec=trained_result.state_spec,
            boundary_spec=trained_result.boundary_spec,
        )

        _, meta = load_weights(path, MLPDirect(hidden_sizes=(16, 16)))
        assert meta["architecture_config"]["hidden_sizes"] == [16, 16]


class TestTrainResultSaveLoad:
    def test_train_result_save_method(self, trained_result, tmp_path):
        path = str(tmp_path / "result.npz")
        trained_result.save(path)
        assert os.path.exists(path)

    def test_train_result_load_method(self, trained_result, tmp_path):
        path = str(tmp_path / "result.npz")
        trained_result.save(path)

        loaded = TrainResult.load(path, MLPDirect(hidden_sizes=(16, 16)))
        assert loaded.state_spec == trained_result.state_spec
        assert loaded.boundary_spec == trained_result.boundary_spec

    def test_to_node_from_loaded(self, trained_result, tmp_path):
        path = str(tmp_path / "result.npz")
        trained_result.save(path)

        loaded = TrainResult.load(path, MLPDirect(hidden_sizes=(16, 16)))
        node = loaded.to_node(
            name="ball", timestep=0.01,
            initial_values={"position": 10.0, "velocity": 0.0},
        )
        assert node.name == "ball"

        # Run the node
        state = node.initial_state()
        new_state = node.update(state, {}, 0.01)
        assert "position" in new_state
        assert "velocity" in new_state


class TestDerivativeArchitecture:
    def test_save_load_derivative_mode(self, tmp_path):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()
        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=50)

        arch = MLPDerivative(hidden_sizes=(16,))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(n_epochs=3, batch_size=32, rng_key=jax.random.PRNGKey(0))

        path = str(tmp_path / "deriv.npz")
        result.save(path)

        loaded = TrainResult.load(path, MLPDerivative(hidden_sizes=(16,)))
        assert loaded.architecture.mode == "derivative"
