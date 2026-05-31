"""Tests for the FMI 3.0 ``modelDescription.xml`` generator."""

import os
import xml.etree.ElementTree as ET

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi.model_description import (
    FMIVariable,
    ModelDescription,
    build_model_description,
)
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


@pytest.fixture
def bouncing_ball_graph():
    gm = GraphManager()
    gm.add_node(BallNode(name="ball", timestep=1e-2,
                         initial_position=1.0, initial_velocity=0.0))
    gm.add_node(TableNode(name="table", timestep=1e-2))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


class TestFMIVariable:

    def test_valid_construction(self):
        v = FMIVariable(
            name="x", value_reference=1, dtype="float32",
            causality="output", variability="continuous",
        )
        assert v.name == "x"

    def test_invalid_causality_rejected(self):
        with pytest.raises(ValueError, match="causality"):
            FMIVariable(
                name="x", value_reference=1, dtype="float32",
                causality="bogus", variability="continuous",
            )

    def test_invalid_variability_rejected(self):
        with pytest.raises(ValueError, match="variability"):
            FMIVariable(
                name="x", value_reference=1, dtype="float32",
                causality="output", variability="bogus",
            )

    def test_invalid_dtype_rejected(self):
        with pytest.raises(ValueError, match="dtype"):
            FMIVariable(
                name="x", value_reference=1, dtype="complex128",
                causality="output", variability="continuous",
            )


class TestBuildModelDescription:

    def test_minimal_graph_produces_valid_xml(self, bouncing_ball_graph):
        md = build_model_description(
            bouncing_ball_graph, model_name="BouncingBall",
        )
        assert md.model_name == "BouncingBall"
        assert md.fmi_version == "3.0"
        xml = md.to_xml()
        root = ET.fromstring(xml)
        assert root.tag == "fmiModelDescription"
        assert root.attrib["fmiVersion"] == "3.0"
        assert root.attrib["modelName"] == "BouncingBall"
        # ModelVariables present.
        mv = root.find("ModelVariables")
        assert mv is not None

    def test_outputs_include_state_fields(self, bouncing_ball_graph):
        md = build_model_description(
            bouncing_ball_graph, model_name="BouncingBall",
        )
        # Both BallNode and TableNode are tagged @stability(STABLE), so
        # their state fields should appear as FMU outputs.
        output_names = {
            v.name for v in md.variables if v.causality == "output"
        }
        # Ball's state: position + velocity.  Table's state: position.
        assert "ball.position" in output_names
        assert "ball.velocity" in output_names
        assert "table.position" in output_names

    def test_instantiation_token_is_deterministic(self, bouncing_ball_graph):
        md1 = build_model_description(
            bouncing_ball_graph, model_name="X",
        )
        md2 = build_model_description(
            bouncing_ball_graph, model_name="X",
        )
        assert md1.instantiation_token == md2.instantiation_token

    def test_different_model_name_yields_different_token(
        self, bouncing_ball_graph,
    ):
        md1 = build_model_description(bouncing_ball_graph, model_name="X")
        md2 = build_model_description(bouncing_ball_graph, model_name="Y")
        assert md1.instantiation_token != md2.instantiation_token

    def test_selected_outputs_restrict_surface(self, bouncing_ball_graph):
        md = build_model_description(
            bouncing_ball_graph, model_name="BB",
            selected_outputs=[("ball", "position")],
        )
        output_names = {
            v.name for v in md.variables if v.causality == "output"
        }
        assert output_names == {"ball.position"}

    def test_default_step_size_from_graph(self, bouncing_ball_graph):
        md = build_model_description(
            bouncing_ball_graph, model_name="BB",
        )
        # Master timestep is 1e-2 on every node.
        assert md.default_step_size > 0

    def test_xml_is_well_formed(self, bouncing_ball_graph):
        md = build_model_description(
            bouncing_ball_graph, model_name="BB",
        )
        # Round-trip: serialize, parse, check the variables match.
        xml = md.to_xml()
        root = ET.fromstring(xml)
        mv = root.find("ModelVariables")
        n_in_xml = sum(1 for _ in mv)
        assert n_in_xml == len(md.variables)


class TestStabilityGating:

    def test_evolving_nodes_excluded_by_default(self, bouncing_ball_graph):
        # Force-import so its @stability decorator fires and the
        # surface gets registered.
        import maddening.api.binary_encoder  # noqa: F401
        from maddening.fmi.model_description import (
            _ensure_stable_only_or_opt_in,
        )
        # An EVOLVING surface is excluded by default.
        assert not _ensure_stable_only_or_opt_in(
            "maddening.api.binary_encoder.BinaryStateEncoder",
            include_evolving=False,
        )
        # ... but admitted when the caller opts in.
        assert _ensure_stable_only_or_opt_in(
            "maddening.api.binary_encoder.BinaryStateEncoder",
            include_evolving=True,
        )

    def test_stable_surfaces_always_included(self):
        from maddening.fmi.model_description import (
            _ensure_stable_only_or_opt_in,
        )
        assert _ensure_stable_only_or_opt_in(
            "maddening.core.node.SimulationNode", include_evolving=False,
        )


class TestStabilityTagging:

    def test_build_md_tagged_evolving(self):
        from maddening.core.compliance.metadata import StabilityLevel
        assert build_model_description._stability_level == \
            StabilityLevel.EVOLVING
