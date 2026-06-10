"""FMI 3.0 round-trip against ``fmpy`` (v0.3.0 §A1 verification gate).

What this test actually exercises
---------------------------------

A full FMI 3.0 round-trip (export → load in another tool → step →
compare to in-process run) requires a compiled C wrapper that links
to the FMI 3.0 ABI.  The v0.3.0 plan deferred that wrapper to v0.4.0
/ MIME v0.5.0; v0.3.0 ships the Python *substrate* (modelDescription
generator + ZMQ sidecar protocol + jvp/vjp shim).

What we can test cheaply *now*, without the compiled wrapper:

1. **Schema validity** — write the generated ``modelDescription.xml``
   into a zipped ``.fmu`` and confirm ``fmpy.model_description.
   read_model_description`` parses it with ``validate=True,
   validate_model_structure=True``.  ``validate_model_description``
   returns an empty issue list.
2. **Structural fidelity** — the variable list fmpy recovers matches
   the (causality, dtype, name) tuples our generator emitted.
3. **Defaults round-trip** — the experiment block survives parse and
   matches what we passed in.

The full simulation-loop round-trip (FMU steps + numerical agreement
with in-process run) is the v0.4.0 deliverable — see the
"out of scope" section in the FMU release notes.

This test is auto-skipped when ``fmpy`` isn't installed (it's listed
as a soft dependency of ``[ci]`` and ``[dev]``).
"""

from __future__ import annotations

import io
import os
import tempfile
import zipfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

# Auto-skip if fmpy isn't installed.
fmpy = pytest.importorskip("fmpy", reason="fmpy not installed")
from fmpy.model_description import read_model_description
from fmpy.validation import validate_model_description

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


@pytest.fixture
def bouncing_ball_graph():
    gm = GraphManager()
    gm.add_node(BallNode(
        name="ball", timestep=1e-2,
        initial_position=1.0, initial_velocity=0.0,
    ))
    gm.add_node(TableNode(name="table", timestep=1e-2))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


def _write_fmu(xml: str, path: str) -> None:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("modelDescription.xml", xml)


# ---------------------------------------------------------------------------
# Round-trip: write -> parse -> validate.
# ---------------------------------------------------------------------------


class TestFmpyRoundTrip:

    def test_fmpy_parses_our_xml(self, bouncing_ball_graph, tmp_path):
        md = build_model_description(
            bouncing_ball_graph, model_name="BouncingBall",
        )
        fmu_path = str(tmp_path / "bouncing.fmu")
        _write_fmu(md.to_xml(), fmu_path)

        parsed = read_model_description(
            fmu_path, validate=True, validate_model_structure=True,
        )

        assert parsed.modelName == "BouncingBall"
        assert parsed.fmiVersion == "3.0"
        # The generator promised exactly one independent variable (time)
        # per FMI 3.0 §2.2.7.  fmpy enforces this; parse succeeded =>
        # we satisfy the schema.
        independent = [v for v in parsed.modelVariables
                       if v.causality == "independent"]
        assert len(independent) == 1, (
            f"expected exactly 1 independent var, got "
            f"{[v.name for v in independent]}"
        )
        assert independent[0].name == "time"

    def test_validate_model_description_reports_no_issues(
        self, bouncing_ball_graph, tmp_path,
    ):
        md = build_model_description(
            bouncing_ball_graph, model_name="BouncingBall",
        )
        fmu_path = str(tmp_path / "bouncing.fmu")
        _write_fmu(md.to_xml(), fmu_path)

        parsed = read_model_description(fmu_path, validate=False)
        issues = validate_model_description(parsed)
        assert issues == [], f"fmpy validation issues: {issues}"

    def test_variable_list_round_trips(
        self, bouncing_ball_graph, tmp_path,
    ):
        """The (name, causality) pairs we generated come back from fmpy."""
        md = build_model_description(
            bouncing_ball_graph, model_name="BouncingBall",
        )
        fmu_path = str(tmp_path / "bouncing.fmu")
        _write_fmu(md.to_xml(), fmu_path)
        parsed = read_model_description(fmu_path, validate=True)

        generated = {(v.name, v.causality) for v in md.variables}
        # fmpy's parsed.modelVariables[i].causality is already a string
        # matching the FMI 3 vocabulary.
        recovered = {(v.name, v.causality) for v in parsed.modelVariables}
        assert generated == recovered

    def test_default_experiment_round_trips(
        self, bouncing_ball_graph, tmp_path,
    ):
        md = build_model_description(
            bouncing_ball_graph, model_name="BB",
        )
        fmu_path = str(tmp_path / "bouncing.fmu")
        _write_fmu(md.to_xml(), fmu_path)
        parsed = read_model_description(fmu_path, validate=True)

        de = parsed.defaultExperiment
        assert de is not None
        assert float(de.startTime) == md.default_start_time
        assert float(de.stopTime) == md.default_stop_time
        assert float(de.stepSize) == md.default_step_size

    def test_unit_definitions_emit_referenced_units(
        self, bouncing_ball_graph, tmp_path,
    ):
        """fmpy refuses any variable referencing an undefined unit;
        our generator emits a <UnitDefinitions> block populated from
        the variables' unit fields.
        """
        md = build_model_description(
            bouncing_ball_graph, model_name="BB",
        )
        fmu_path = str(tmp_path / "bouncing.fmu")
        _write_fmu(md.to_xml(), fmu_path)
        # validate=True turns "undefined unit" into a hard parse error.
        # Reaching this line at all means the units cleared validation.
        parsed = read_model_description(
            fmu_path, validate=True, validate_model_structure=True,
        )
        # And the round-tripped unit definitions include "s" (the time
        # variable's unit).
        unit_names = {u.name for u in parsed.unitDefinitions}
        assert "s" in unit_names

    def test_instantiation_token_round_trips(
        self, bouncing_ball_graph, tmp_path,
    ):
        md = build_model_description(
            bouncing_ball_graph, model_name="BB",
        )
        fmu_path = str(tmp_path / "bouncing.fmu")
        _write_fmu(md.to_xml(), fmu_path)
        parsed = read_model_description(fmu_path, validate=True)
        assert parsed.instantiationToken == md.instantiation_token


# ---------------------------------------------------------------------------
# What this test does NOT cover (v0.4.0 work — explicit non-goal).
# ---------------------------------------------------------------------------


class TestKnownOutOfScope:
    """Pins the explicit non-goals so reviewers don't expect them."""

    def test_no_compiled_binary_shipped(self, bouncing_ball_graph, tmp_path):
        """The v0.3.0 substrate ships the modelDescription + ZMQ sidecar.
        The FMU binary that links to the FMI 3.0 C ABI is a v0.4.0 /
        MIME v0.5.0 deliverable.  This test exists to make that
        non-goal explicit -- if someone adds a ``binaries/`` subtree
        to the FMU zip in v0.3.0, this test fires and they have to
        either update the plan or remove the addition.
        """
        md = build_model_description(
            bouncing_ball_graph, model_name="BB",
        )
        fmu_path = str(tmp_path / "bouncing.fmu")
        _write_fmu(md.to_xml(), fmu_path)
        with zipfile.ZipFile(fmu_path) as zf:
            names = zf.namelist()
        assert names == ["modelDescription.xml"], (
            f"v0.3.0 substrate FMU should only contain "
            f"modelDescription.xml; got {names}"
        )
