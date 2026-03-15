"""
Tests for USD graph serialization (Phase 7).

Tests save_graph_to_usd and load_graph_from_usd round-trip,
including edge transforms, coupling groups, and external inputs.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from pxr import Usd, Sdf, Vt

import maddening.usd  # noqa: F401 (schema registration)
from maddening.usd.serialization import (
    save_graph_to_usd,
    load_graph_from_usd,
    register_node_class,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.transforms import register_transform, get_transform_name
from maddening.nodes.heat import HeatNode
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode


@register_transform("test_negate_for_usd", "Negate for USD test")
def _negate_for_usd(x):
    return -x


class TestSaveGraphToUSD:
    """Test serialization of GraphManager to USD."""

    def test_save_simple_graph(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", 0.01, initial_position=5.0))
        gm.add_node(TableNode("table", 0.01, position=0.0))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        # Check root
        root = stage.GetPrimAtPath("/Simulation")
        assert root.IsValid()
        assert root.GetTypeName() == "MaddeningSimulationGraph"

        # Check nodes
        ball_prim = stage.GetPrimAtPath("/Simulation/nodes/ball")
        assert ball_prim.IsValid()
        assert "BallNode" in ball_prim.GetAttribute("maddening:nodeType").Get()

        table_prim = stage.GetPrimAtPath("/Simulation/nodes/table")
        assert table_prim.IsValid()

        # Check edge
        edge_prim = stage.GetPrimAtPath("/Simulation/edges/e0")
        assert edge_prim.IsValid()
        assert edge_prim.GetAttribute("maddening:sourceNode").Get() == "table"
        assert edge_prim.GetAttribute("maddening:targetNode").Get() == "ball"
        assert edge_prim.GetAttribute("maddening:sourceField").Get() == "position"
        assert edge_prim.GetAttribute("maddening:targetField").Get() == "table_position"

    def test_save_with_registered_transform(self):
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform="extract_last")
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        edge = stage.GetPrimAtPath("/Simulation/edges/e0")
        assert edge.GetAttribute("maddening:transformName").Get() == "extract_last"

    def test_save_unregistered_transform_raises(self):
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform=lambda T: T[-1])  # unregistered lambda
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        from maddening.core.transforms import UnregisteredTransformError
        with pytest.raises(UnregisteredTransformError):
            save_graph_to_usd(gm, stage)

    def test_save_coupling_group(self):
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                     transform="extract_last")
        gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                     transform="extract_first")
        gm.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=15,
            tolerance=1e-7,
            acceleration="aitken",
            diagnostics=True,
        )
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        cg = stage.GetPrimAtPath("/Simulation/coupling_groups/cg0")
        assert cg.IsValid()
        nodes = sorted(cg.GetAttribute("maddening:nodes").Get())
        assert nodes == ["rod_a", "rod_b"]
        assert cg.GetAttribute("maddening:maxIterations").Get() == 15
        assert abs(cg.GetAttribute("maddening:tolerance").Get() - 1e-7) < 1e-15
        assert cg.GetAttribute("maddening:acceleration").Get() == "aitken"
        assert cg.GetAttribute("maddening:diagnostics").Get() is True

    def test_save_external_input(self):
        gm = GraphManager()
        gm.add_node(BallNode("ball", 0.01, initial_position=5.0))
        gm.add_external_input("ball", "external_force", shape=(3,))
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        ext = stage.GetPrimAtPath("/Simulation/external_inputs/ext0")
        assert ext.IsValid()
        assert ext.GetAttribute("maddening:targetNode").Get() == "ball"
        assert ext.GetAttribute("maddening:targetField").Get() == "external_force"
        assert list(ext.GetAttribute("maddening:shape").Get()) == [3]

    def test_save_additive_edge(self):
        gm = GraphManager()
        gm.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm.add_edge("rod_a", "rod_b", "temperature", "heat_source",
                     transform="extract_last", additive=True)
        gm.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm, stage)

        edge = stage.GetPrimAtPath("/Simulation/edges/e0")
        assert edge.GetAttribute("maddening:additive").Get() is True


class TestLoadGraphFromUSD:
    """Test deserialization of GraphManager from USD."""

    def test_round_trip_simple(self):
        """Save and reload a simple graph, verify structure matches."""
        gm1 = GraphManager()
        gm1.add_node(BallNode("ball", 0.01, initial_position=5.0,
                               initial_velocity=0.0, elasticity=0.7))
        gm1.add_node(TableNode("table", 0.01, position=0.0))
        gm1.add_edge("table", "ball", "position", "table_position")
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)

        assert set(gm2._nodes.keys()) == {"ball", "table"}
        assert len(gm2._edges) == 1
        assert gm2._edges[0].source_node == "table"
        assert gm2._edges[0].target_node == "ball"

    def test_round_trip_heat_nodes(self):
        """Save and reload HeatNodes, verify params preserved."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod", 0.005, n_cells=20, length=2.0,
                               thermal_diffusivity=0.05))
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        rod = gm2._nodes["rod"].node
        assert isinstance(rod, HeatNode)
        assert rod.params["n_cells"] == 20
        assert rod.params["length"] == 2.0
        assert abs(rod.params["thermal_diffusivity"] - 0.05) < 1e-10

    def test_round_trip_with_transform(self):
        """Save and reload edges with registered transforms."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm1.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm1.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                      transform="extract_last")
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        assert len(gm2._edges) == 1
        edge = gm2._edges[0]
        assert edge.transform is not None
        # Verify the transform works
        arr = jnp.array([1.0, 2.0, 3.0])
        assert float(edge.transform(arr)) == 3.0

    def test_round_trip_coupling_group(self):
        """Save and reload coupling groups."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm1.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm1.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                      transform="extract_last")
        gm1.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                      transform="extract_first")
        gm1.add_coupling_group(
            ["rod_a", "rod_b"],
            max_iterations=25,
            tolerance=1e-9,
            acceleration="aitken",
            diagnostics=True,
            convergence_norm="mixed",
        )
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        assert len(gm2._coupling_groups) == 1
        cg = gm2._coupling_groups[0]
        assert cg.nodes == frozenset(["rod_a", "rod_b"])
        assert cg.max_iterations == 25
        assert abs(cg.tolerance - 1e-9) < 1e-15
        assert cg.acceleration == "aitken"
        assert cg.diagnostics is True
        assert cg.convergence_norm == "mixed"

    def test_round_trip_external_input(self):
        """Save and reload external inputs."""
        gm1 = GraphManager()
        gm1.add_node(BallNode("ball", 0.01, initial_position=5.0))
        gm1.add_external_input("ball", "force", shape=(3,))
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        assert len(gm2._external_inputs) == 1
        ext = gm2._external_inputs[0]
        assert ext.target_node == "ball"
        assert ext.target_field == "force"
        assert ext.shape == (3,)

    def test_round_trip_file_persistence(self, tmp_path):
        """Full round-trip through a .usda file."""
        filepath = str(tmp_path / "graph.usda")

        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod_a", 0.01, n_cells=8))
        gm1.add_node(HeatNode("rod_b", 0.01, n_cells=8))
        gm1.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                      transform="extract_last")
        gm1.compile()

        # Save to file
        stage = Usd.Stage.CreateNew(filepath)
        save_graph_to_usd(gm1, stage)
        stage.Save()

        # Load from file
        stage2 = Usd.Stage.Open(filepath)
        gm2 = load_graph_from_usd(stage2)

        assert set(gm2._nodes.keys()) == {"rod_a", "rod_b"}
        assert len(gm2._edges) == 1

        # Verify the loaded graph can compile and step
        gm2.compile()
        state = gm2.step()
        assert "rod_a" in state
        assert "rod_b" in state

    def test_round_trip_additive_edge(self):
        """Save and reload additive edges."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("rod_a", 0.01, n_cells=5))
        gm1.add_node(HeatNode("rod_b", 0.01, n_cells=5))
        gm1.add_edge("rod_a", "rod_b", "temperature", "heat_source",
                      transform="identity", additive=True)
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        gm2 = load_graph_from_usd(stage)
        assert gm2._edges[0].additive is True

    def test_load_nonexistent_path_raises(self):
        stage = Usd.Stage.CreateInMemory()
        with pytest.raises(ValueError, match="No prim"):
            load_graph_from_usd(stage, root_path="/NonExistent")

    def test_round_trip_multirate(self):
        """Save and reload a multi-rate graph."""
        gm1 = GraphManager()
        gm1.add_node(HeatNode("fast", 0.001, n_cells=5))
        gm1.add_node(HeatNode("slow", 0.01, n_cells=5))
        gm1.add_edge("fast", "slow", "temperature", "left_temperature",
                      transform="extract_last")
        gm1.compile()

        stage = Usd.Stage.CreateInMemory()
        save_graph_to_usd(gm1, stage)

        root = stage.GetPrimAtPath("/Simulation")
        assert root.GetAttribute("maddening:isMultirate").Get() is True

        gm2 = load_graph_from_usd(stage)
        fast_node = gm2._nodes["fast"].node
        slow_node = gm2._nodes["slow"].node
        assert abs(fast_node.delta_t - 0.001) < 1e-10
        assert abs(slow_node.delta_t - 0.01) < 1e-10
