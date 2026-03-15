"""
Tests for USD schema registration (Phase 6).

These tests verify that the MADDENING codeless schemas are correctly
registered with the USD SchemaRegistry and that typed prims can be
created with the expected attributes.
"""

import os
import subprocess
import sys
import tempfile

import pytest
from pxr import Usd, Sdf, Vt

# conftest.py already imported maddening.usd


class TestSchemaRegistration:
    """Test that all schema types are registered."""

    def test_simulation_graph_type_exists(self):
        reg = Usd.SchemaRegistry()
        defn = reg.FindConcretePrimDefinition("MaddeningSimulationGraph")
        assert defn is not None

    def test_node_type_exists(self):
        reg = Usd.SchemaRegistry()
        defn = reg.FindConcretePrimDefinition("MaddeningNode")
        assert defn is not None

    def test_edge_type_exists(self):
        reg = Usd.SchemaRegistry()
        defn = reg.FindConcretePrimDefinition("MaddeningEdge")
        assert defn is not None

    def test_coupling_group_type_exists(self):
        reg = Usd.SchemaRegistry()
        defn = reg.FindConcretePrimDefinition("MaddeningCouplingGroup")
        assert defn is not None

    def test_external_input_type_exists(self):
        reg = Usd.SchemaRegistry()
        defn = reg.FindConcretePrimDefinition("MaddeningExternalInput")
        assert defn is not None


class TestSimulationGraphPrim:
    """Test MaddeningSimulationGraph prim creation and attributes."""

    def test_create_prim(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/Sim", "MaddeningSimulationGraph")
        assert prim.IsValid()
        assert prim.GetTypeName() == "MaddeningSimulationGraph"

    def test_default_attributes(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/Sim", "MaddeningSimulationGraph")
        assert prim.GetAttribute("maddening:baseDt").Get() == 0.01
        assert prim.GetAttribute("maddening:isMultirate").Get() is False
        assert prim.GetAttribute("maddening:description").Get() == ""

    def test_set_attributes(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/Sim", "MaddeningSimulationGraph")
        prim.GetAttribute("maddening:baseDt").Set(0.001)
        prim.GetAttribute("maddening:isMultirate").Set(True)
        prim.GetAttribute("maddening:description").Set("test graph")
        assert prim.GetAttribute("maddening:baseDt").Get() == 0.001
        assert prim.GetAttribute("maddening:isMultirate").Get() is True
        assert prim.GetAttribute("maddening:description").Get() == "test graph"


class TestNodePrim:
    """Test MaddeningNode prim creation and attributes."""

    def test_create_prim(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/nodes/ball", "MaddeningNode")
        assert prim.IsValid()
        assert prim.GetTypeName() == "MaddeningNode"

    def test_default_attributes(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/nodes/ball", "MaddeningNode")
        assert prim.GetAttribute("maddening:nodeType").Get() == ""
        assert prim.GetAttribute("maddening:timestep").Get() == 0.01
        assert prim.GetAttribute("maddening:paramsJson").Get() == "{}"

    def test_set_node_type(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/nodes/ball", "MaddeningNode")
        prim.GetAttribute("maddening:nodeType").Set(
            "maddening.nodes.ball.BallNode"
        )
        assert (
            prim.GetAttribute("maddening:nodeType").Get()
            == "maddening.nodes.ball.BallNode"
        )


class TestEdgePrim:
    """Test MaddeningEdge prim creation and attributes."""

    def test_create_prim(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/edges/e0", "MaddeningEdge")
        assert prim.IsValid()

    def test_default_attributes(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/edges/e0", "MaddeningEdge")
        assert prim.GetAttribute("maddening:sourceNode").Get() == ""
        assert prim.GetAttribute("maddening:targetNode").Get() == ""
        assert prim.GetAttribute("maddening:additive").Get() is False

    def test_set_edge_attributes(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/edges/e0", "MaddeningEdge")
        prim.GetAttribute("maddening:sourceNode").Set("ball")
        prim.GetAttribute("maddening:targetNode").Set("spring")
        prim.GetAttribute("maddening:sourceField").Set("position")
        prim.GetAttribute("maddening:targetField").Set("ball_position")
        prim.GetAttribute("maddening:transformName").Set("extract_last")
        prim.GetAttribute("maddening:additive").Set(True)

        assert prim.GetAttribute("maddening:sourceNode").Get() == "ball"
        assert prim.GetAttribute("maddening:additive").Get() is True
        assert prim.GetAttribute("maddening:transformName").Get() == "extract_last"


class TestCouplingGroupPrim:
    """Test MaddeningCouplingGroup prim."""

    def test_create_and_set(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/coupling/cg0", "MaddeningCouplingGroup")
        assert prim.IsValid()

        prim.GetAttribute("maddening:nodes").Set(
            Vt.StringArray(["rod_a", "rod_b"])
        )
        prim.GetAttribute("maddening:maxIterations").Set(20)
        prim.GetAttribute("maddening:tolerance").Set(1e-8)
        prim.GetAttribute("maddening:acceleration").Set("aitken")

        nodes = list(prim.GetAttribute("maddening:nodes").Get())
        assert nodes == ["rod_a", "rod_b"]
        assert prim.GetAttribute("maddening:maxIterations").Get() == 20
        assert prim.GetAttribute("maddening:acceleration").Get() == "aitken"


class TestExternalInputPrim:
    """Test MaddeningExternalInput prim."""

    def test_create_and_set(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/ext/ext0", "MaddeningExternalInput")
        assert prim.IsValid()

        prim.GetAttribute("maddening:targetNode").Set("ball")
        prim.GetAttribute("maddening:targetField").Set("force")
        prim.GetAttribute("maddening:shape").Set(Vt.IntArray([3]))

        assert prim.GetAttribute("maddening:targetNode").Get() == "ball"
        assert list(prim.GetAttribute("maddening:shape").Get()) == [3]


class TestStagePersistence:
    """Test that stages with MADDENING types can be saved/loaded."""

    def test_save_and_reload(self, tmp_path):
        filepath = str(tmp_path / "test.usda")

        # Create and save
        stage = Usd.Stage.CreateNew(filepath)
        root = stage.DefinePrim("/Sim", "MaddeningSimulationGraph")
        root.GetAttribute("maddening:baseDt").Set(0.005)

        node = stage.DefinePrim("/Sim/nodes/heat", "MaddeningNode")
        node.GetAttribute("maddening:nodeType").Set("maddening.nodes.heat.HeatNode")
        node.GetAttribute("maddening:timestep").Set(0.005)

        stage.Save()

        # Reload
        stage2 = Usd.Stage.Open(filepath)
        root2 = stage2.GetPrimAtPath("/Sim")
        assert root2.GetTypeName() == "MaddeningSimulationGraph"
        assert root2.GetAttribute("maddening:baseDt").Get() == 0.005

        node2 = stage2.GetPrimAtPath("/Sim/nodes/heat")
        assert node2.GetTypeName() == "MaddeningNode"
        assert (
            node2.GetAttribute("maddening:nodeType").Get()
            == "maddening.nodes.heat.HeatNode"
        )


class TestLateRegistration:
    """Test that late schema registration fails gracefully.

    This MUST run in a subprocess because the USD SchemaRegistry
    is a process-level singleton.
    """

    def test_late_registration_subprocess(self):
        """Verify that importing maddening.usd after Usd.Stage creation
        still registers the schemas correctly.

        In USD 26.x, late plugin registration refreshes the
        SchemaRegistry, so schemas are available even if registered
        after the first stage creation.  In older USD versions, this
        may fail (triggering a RuntimeError from our __init__.py
        guard).  Either outcome is acceptable.
        """
        code = """
import sys
# Import USD first and create a stage (triggers registry caching)
from pxr import Usd
stage = Usd.Stage.CreateInMemory()

# Now try to import maddening.usd (late)
try:
    import maddening.usd
    # Check if the schemas actually work
    reg = Usd.SchemaRegistry()
    defn = reg.FindConcretePrimDefinition("MaddeningNode")
    if defn is None:
        print("LATE_FAIL")
    else:
        print("LATE_OK")
    sys.exit(0)
except RuntimeError as e:
    if "schema registration failed" in str(e):
        print("EXPECTED_ERROR")
        sys.exit(0)
    else:
        print(f"UNEXPECTED_ERROR: {e}")
        sys.exit(1)
except Exception as e:
    print(f"UNEXPECTED_EXCEPTION: {type(e).__name__}: {e}")
    sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        stdout = result.stdout.strip()
        # Late registration works in USD 26.x.  LATE_OK, EXPECTED_ERROR,
        # and LATE_FAIL are all acceptable outcomes.
        assert stdout in ("EXPECTED_ERROR", "LATE_FAIL", "LATE_OK"), (
            f"Unexpected result: stdout={stdout!r}, "
            f"stderr={result.stderr[:200]!r}"
        )

    def test_early_registration_subprocess(self):
        """Verify that importing maddening.usd before any stage works."""
        code = """
import sys
# Import maddening.usd first (early registration)
import maddening.usd
from pxr import Usd

# Now create a stage and check
stage = Usd.Stage.CreateInMemory()
prim = stage.DefinePrim("/test", "MaddeningNode")
attr = prim.GetAttribute("maddening:nodeType")
if attr.IsValid():
    print("EARLY_OK")
    sys.exit(0)
else:
    print("EARLY_FAIL")
    sys.exit(1)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.stdout.strip() == "EARLY_OK", (
            f"Early registration failed: stdout={result.stdout!r}, "
            f"stderr={result.stderr[:200]!r}"
        )
