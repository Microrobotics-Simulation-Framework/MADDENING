"""End-to-end integration tests: physics graph -> data -> train -> replace -> run -> compare."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.surrogates.dataset import DatasetGenerator
from maddening.surrogates.training.trainer import SurrogateTrainer
from maddening.surrogates.replace import replace_node
from maddening.surrogates.validator import SurrogateValidator
from maddening.surrogates.architectures.mlp import MLPDirect


class TestEndToEnd:
    """Full pipeline: generate data from physics, train surrogate, replace, validate."""

    def test_full_pipeline(self):
        # 1. Build physics graph
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()

        # 2. Generate training data
        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=300)
        assert ds.states["position"].shape[0] == 299

        # 3. Train surrogate
        arch = MLPDirect(hidden_sizes=(64, 64))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(
            n_epochs=50, batch_size=64,
            rng_key=jax.random.PRNGKey(42),
        )

        # Should converge
        assert result.train_losses[-1] < result.train_losses[0] * 0.5

        # 4. Create surrogate node
        surrogate = result.to_node(
            name="ball", timestep=0.01,
            initial_values={"position": 10.0, "velocity": 0.0},
        )

        # 5. Replace in a fresh graph
        gm_surr = GraphManager()
        gm_surr.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm_surr.compile()
        replace_node(gm_surr, "ball", surrogate)
        gm_surr.compile()

        # 6. Run both
        gm_phys = GraphManager()
        gm_phys.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm_phys.compile()

        report = SurrogateValidator.compare_graphs(gm_phys, gm_surr, 50, "ball")

        # Should have reasonable accuracy for a short rollout
        assert report.per_field_mse["position"] < 10.0  # generous threshold
        assert report.per_timestep_errors is not None
        assert "position" in report.per_timestep_errors

    def test_surrogate_in_scan_with_history(self):
        """Surrogate node works with run_scan_with_history."""
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()

        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=100)
        arch = MLPDirect(hidden_sizes=(32, 32))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(n_epochs=10, batch_size=32,
                               rng_key=jax.random.PRNGKey(0))

        surrogate = result.to_node(
            name="ball", timestep=0.01,
            initial_values={"position": 10.0, "velocity": 0.0},
        )

        gm2 = GraphManager()
        gm2.add_node(surrogate)
        gm2.compile()

        final, history = gm2.run_scan_with_history(20)
        assert history["ball"]["position"].shape == (20,)
        assert history["ball"]["velocity"].shape == (20,)

    def test_surrogate_in_sweep(self):
        """Surrogate node works with run_sweep (vmap)."""
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()

        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=100)
        arch = MLPDirect(hidden_sizes=(32, 32))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(n_epochs=10, batch_size=32,
                               rng_key=jax.random.PRNGKey(0))

        surrogate = result.to_node(
            name="ball", timestep=0.01,
            initial_values={"position": 10.0, "velocity": 0.0},
        )

        gm2 = GraphManager()
        gm2.add_node(surrogate)
        gm2.compile()

        batch_init = {
            "ball": {
                "position": jnp.array([10.0, 8.0, 6.0]),
                "velocity": jnp.zeros(3),
            }
        }
        finals = gm2.run_sweep(10, batch_init)
        assert finals["ball"]["position"].shape == (3,)

    def test_validator_compare_nodes(self):
        """SurrogateValidator.compare_nodes produces a report."""
        gm = GraphManager()
        ball = BallNode("ball", timestep=0.01, initial_position=10.0)
        gm.add_node(ball)
        gm.compile()

        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=100)
        arch = MLPDirect(hidden_sizes=(32, 32))
        trainer = SurrogateTrainer(arch, ds)
        result = trainer.train(n_epochs=20, batch_size=32,
                               rng_key=jax.random.PRNGKey(0))

        surrogate = result.to_node(
            name="ball", timestep=0.01,
            initial_values={"position": 10.0, "velocity": 0.0},
        )

        # Use a small test split
        test_ds = DatasetGenerator.from_graph(gm, "ball", n_steps=30)
        report = SurrogateValidator.compare_nodes(ball, surrogate, test_ds)

        assert "position" in report.per_field_mse
        assert "velocity" in report.per_field_mse
        summary = report.summary()
        assert "position" in summary

    def test_validation_report_summary(self):
        from maddening.surrogates.validator import ValidationReport
        report = ValidationReport(
            node_name="test",
            per_field_mse={"x": 0.001},
            per_field_max_error={"x": 0.05},
            per_field_relative_error={"x": 0.01},
        )
        s = report.summary()
        assert "test" in s
        assert "x" in s
