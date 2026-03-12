"""Tests for serialization round-trips."""

import pytest
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


REGISTRY = {"BallNode": BallNode, "TableNode": TableNode}


class TestSerialization:
    def test_to_dict_structure(self, bouncing_ball_graph):
        d = bouncing_ball_graph.to_dict()
        assert "nodes" in d
        assert "edges" in d
        assert "external_inputs" in d
        assert len(d["nodes"]) == 2
        assert len(d["edges"]) == 1

    def test_round_trip_nodes(self, bouncing_ball_graph):
        config = bouncing_ball_graph.to_dict()
        gm2 = GraphManager.from_dict(config, REGISTRY)
        assert set(gm2.node_names) == {"ball", "table"}

    def test_round_trip_edges(self, bouncing_ball_graph):
        config = bouncing_ball_graph.to_dict()
        gm2 = GraphManager.from_dict(config, REGISTRY)
        assert len(gm2._edges) == 1
        assert gm2._edges[0].source_node == "table"
        assert gm2._edges[0].target_node == "ball"

    def test_round_trip_compile_and_step(self, bouncing_ball_graph):
        config = bouncing_ball_graph.to_dict()
        gm2 = GraphManager.from_dict(config, REGISTRY)
        gm2.compile()
        gm2.step()
        state = gm2.get_node_state("ball")
        assert float(state["position"]) < 5.0  # ball moved

    def test_round_trip_with_external_inputs(self):
        gm = GraphManager()
        gm.add_node(BallNode(name="b", timestep=0.01))
        gm.add_node(TableNode(name="t", timestep=0.01))
        gm.add_edge("t", "b", "position", "table_position")
        gm.add_external_input("b", "force", shape=(3,))
        config = gm.to_dict()

        gm2 = GraphManager.from_dict(config, REGISTRY)
        assert len(gm2._external_inputs) == 1
        assert gm2._external_inputs[0].target_node == "b"
        assert gm2._external_inputs[0].target_field == "force"
        assert gm2._external_inputs[0].shape == (3,)

    def test_round_trip_preserves_params(self, bouncing_ball_graph):
        config = bouncing_ball_graph.to_dict()
        gm2 = GraphManager.from_dict(config, REGISTRY)
        ball_state = gm2.get_node_state("ball")
        assert float(ball_state["position"]) == pytest.approx(5.0)

    def test_from_dict_missing_node_type_raises(self):
        config = {
            "nodes": [{"type": "UnknownNode", "name": "x", "timestep": 0.01}],
            "edges": [],
        }
        with pytest.raises(KeyError):
            GraphManager.from_dict(config, REGISTRY)

    def test_config_module_delegates(self, bouncing_ball_graph):
        """Test that maddening.serialization.config works."""
        from maddening.serialization.config import to_dict, from_dict
        config = to_dict(bouncing_ball_graph)
        gm2 = from_dict(config, REGISTRY)
        assert set(gm2.node_names) == {"ball", "table"}
