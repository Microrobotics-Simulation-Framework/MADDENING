"""Tests for DatasetGenerator -- shape correctness and boundary_inputs alignment."""

import pytest
import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.surrogates.dataset import DatasetGenerator, SurrogateDataset


class TestFromGraph:
    def test_basic_shapes(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()

        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=50)

        assert isinstance(ds, SurrogateDataset)
        assert ds.node_name == "ball"
        assert ds.dt == 0.01
        assert ds.states["position"].shape == (49,)
        assert ds.states["velocity"].shape == (49,)
        assert ds.next_states["position"].shape == (49,)
        assert ds.next_states["velocity"].shape == (49,)

    def test_state_spec(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01))
        gm.compile()

        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=10)
        assert "position" in ds.state_spec
        assert "velocity" in ds.state_spec

    def test_boundary_inputs_from_edges(self):
        gm = GraphManager()
        gm.add_node(TableNode("table", timestep=0.01))
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()

        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=20)

        assert "table_position" in ds.boundary_inputs
        assert ds.boundary_inputs["table_position"].shape[0] == 19
        assert "table_position" in ds.boundary_spec

    def test_external_inputs_are_zeros(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01))
        gm.add_external_input("ball", "force", shape=())
        gm.compile()

        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=10)

        assert "force" in ds.boundary_inputs
        assert jnp.allclose(ds.boundary_inputs["force"], 0.0)

    def test_consecutive_state_pairs(self):
        """states[i] -> next_states[i] should be consecutive timesteps."""
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm.compile()

        ds = DatasetGenerator.from_graph(gm, "ball", n_steps=20)

        # Manually run to get history
        gm2 = GraphManager()
        gm2.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
        gm2.compile()
        _, hist = gm2.run_scan_with_history(20)

        # states[0] should be history step 0, next_states[0] should be step 1
        assert float(ds.states["position"][0]) == pytest.approx(
            float(hist["ball"]["position"][0])
        )
        assert float(ds.next_states["position"][0]) == pytest.approx(
            float(hist["ball"]["position"][1])
        )


class TestFromSweep:
    def test_basic_shapes(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=0.0))
        gm.compile()

        batch_size = 5
        n_steps = 20
        init_batch = {
            "ball": {
                "position": jnp.linspace(1.0, 10.0, batch_size),
                "velocity": jnp.zeros(batch_size),
            }
        }

        ds = DatasetGenerator.from_sweep(gm, "ball", n_steps, init_batch)

        expected_samples = batch_size * (n_steps - 1)
        assert ds.states["position"].shape == (expected_samples,)
        assert ds.next_states["position"].shape == (expected_samples,)

    def test_sweep_boundary_inputs(self):
        gm = GraphManager()
        gm.add_node(TableNode("table", timestep=0.01))
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()

        batch_size = 3
        n_steps = 10
        init_batch = {
            "ball": {
                "position": jnp.array([5.0, 6.0, 7.0]),
                "velocity": jnp.zeros(3),
            },
            "table": {
                "position": jnp.zeros(3),
                "velocity": jnp.zeros(3),
            },
        }

        ds = DatasetGenerator.from_sweep(gm, "ball", n_steps, init_batch)

        expected_samples = batch_size * (n_steps - 1)
        assert "table_position" in ds.boundary_inputs
        assert ds.boundary_inputs["table_position"].shape[0] == expected_samples
