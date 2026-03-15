"""
Tests for USDWriter (Phase 6 write path).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from pxr import Usd, Sdf, Vt

import maddening.usd  # noqa: F401 (schema registration)
from maddening.usd.writer import USDWriter
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


@pytest.fixture
def simple_graph():
    """A minimal graph with a ball and table."""
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=5.0, initial_velocity=0.0))
    gm.add_node(TableNode("table", 0.01, position=0.0))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


@pytest.fixture
def heat_graph():
    """A graph with a single HeatNode."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", 0.001, n_cells=10, length=1.0))
    gm.compile()
    return gm


class TestUSDWriterCreation:
    """Test USDWriter initialization."""

    def test_create_writer(self, simple_graph):
        stage = Usd.Stage.CreateInMemory()
        writer = USDWriter(stage, simple_graph)
        root = stage.GetPrimAtPath("/Simulation")
        assert root.IsValid()
        assert root.GetTypeName() == "MaddeningSimulationGraph"

    def test_custom_root_path(self, simple_graph):
        stage = Usd.Stage.CreateInMemory()
        writer = USDWriter(stage, simple_graph, root_path="/MyGraph")
        root = stage.GetPrimAtPath("/MyGraph")
        assert root.IsValid()


class TestWriteFrame:
    """Test writing simulation frames."""

    def test_write_single_frame(self, simple_graph):
        stage = Usd.Stage.CreateInMemory()
        writer = USDWriter(stage, simple_graph)
        # Use the internal state (initial state before any step)
        state = dict(simple_graph._state)
        writer.write_frame(state, 0.0)

        # Ball prim should exist with state attributes
        ball_prim = stage.GetPrimAtPath("/Simulation/nodes/ball")
        assert ball_prim.IsValid()
        assert ball_prim.GetTypeName() == "MaddeningNode"

        # Check time-sampled scalar attribute
        pos_attr = ball_prim.GetAttribute("state:position")
        assert pos_attr.IsValid()
        val = pos_attr.Get(0.0)
        assert isinstance(val, float)
        assert abs(val - 5.0) < 1e-5

    def test_write_multiple_frames(self, simple_graph):
        stage = Usd.Stage.CreateInMemory()
        writer = USDWriter(stage, simple_graph)

        for t in range(5):
            state = dict(simple_graph._state)
            writer.write_frame(state, float(t))
            simple_graph.step()

        ball_prim = stage.GetPrimAtPath("/Simulation/nodes/ball")
        pos_attr = ball_prim.GetAttribute("state:position")

        # Should have 5 time samples
        samples = pos_attr.GetTimeSamples()
        assert len(samples) == 5

        # First should be near 5.0, later ones should decrease (gravity)
        assert abs(pos_attr.Get(0.0) - 5.0) < 1e-5
        assert pos_attr.Get(4.0) < 5.0

    def test_write_array_field(self, heat_graph):
        stage = Usd.Stage.CreateInMemory()
        writer = USDWriter(stage, heat_graph)
        state = dict(heat_graph._state)
        writer.write_frame(state, 0.0)

        rod_prim = stage.GetPrimAtPath("/Simulation/nodes/rod")
        temp_attr = rod_prim.GetAttribute("state:temperature")
        assert temp_attr.IsValid()

        val = temp_attr.Get(0.0)
        assert len(val) == 10  # n_cells = 10

    def test_skip_meta_key(self, simple_graph):
        stage = Usd.Stage.CreateInMemory()
        writer = USDWriter(stage, simple_graph)
        state = dict(simple_graph._state)
        state["_meta"] = {"step_counter": jnp.array(0)}
        writer.write_frame(state, 0.0)

        # _meta should not create a prim
        meta_prim = stage.GetPrimAtPath("/Simulation/nodes/_meta")
        assert not meta_prim.IsValid()

    def test_node_metadata_written(self, simple_graph):
        stage = Usd.Stage.CreateInMemory()
        writer = USDWriter(stage, simple_graph)
        state = dict(simple_graph._state)
        writer.write_frame(state, 0.0)

        ball_prim = stage.GetPrimAtPath("/Simulation/nodes/ball")
        node_type = ball_prim.GetAttribute("maddening:nodeType").Get()
        assert "BallNode" in node_type

    def test_save_and_reload(self, heat_graph, tmp_path):
        filepath = str(tmp_path / "heat_sim.usda")

        # Write
        stage = Usd.Stage.CreateNew(filepath)
        writer = USDWriter(stage, heat_graph)
        state = dict(heat_graph._state)
        writer.write_frame(state, 0.0)
        heat_graph.step()
        state = dict(heat_graph._state)
        writer.write_frame(state, 1.0)
        stage.Save()

        # Reload
        stage2 = Usd.Stage.Open(filepath)
        rod = stage2.GetPrimAtPath("/Simulation/nodes/rod")
        temp = rod.GetAttribute("state:temperature")
        assert len(temp.GetTimeSamples()) == 2
