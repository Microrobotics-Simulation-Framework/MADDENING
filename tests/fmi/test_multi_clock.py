"""Multi-clock FMU export (v0.4.0 plan, "Other ideas").

``build_model_description(multi_clock=True)`` emits one FMI 3.0
``<Clock>`` per distinct node timestep and tags every exported output and
external input with the clock of its node.  Off by default, so the
single-clock v0.3.0 surface is byte-for-byte unchanged.
"""

import xml.etree.ElementTree as ET
import zipfile

import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode


@pytest.fixture
def multirate_graph():
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=1e-2))
    gm.add_node(BallNode(name="ball", timestep=1e-2, initial_position=1.0))
    gm.add_node(SpringDamperNode(name="spring", timestep=5e-2))     # 5x coarser
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


def _by_name(md):
    return {v.name: v for v in md.variables}


def test_default_export_has_no_clocks(multirate_graph):
    md = build_model_description(multirate_graph, model_name="m")
    assert md.clocks() == []
    assert all(not v.clocks and v.variability == "continuous"
               for v in md.variables if v.causality in ("output", "input"))
    assert "<Clock" not in md.to_xml()


def test_one_clock_per_distinct_timestep_and_outputs_tagged(multirate_graph):
    md = build_model_description(multirate_graph, model_name="m", multi_clock=True)
    clocks = md.clocks()
    assert [c.name for c in clocks] == ["clock_0", "clock_1"]
    assert [c.interval_decimal for c in clocks] == [1e-2, 5e-2]
    assert clocks[0].interval_decimal == md.default_step_size      # fastest == master
    v = _by_name(md)
    fast, slow = clocks[0].value_reference, clocks[1].value_reference
    assert v["ball.position"].clocks == (fast,) and v["table.position"].clocks == (fast,)
    assert v["spring.position"].clocks == (slow,)
    assert v["spring.position"].variability == "discrete"
    # the external input into the slow node is sampled at the slow rate
    ext = next(x for x in md.variables if x.causality == "input" and not x.is_clock)
    assert ext.clocks == (slow,) and ext.variability == "discrete"
    # parameters are not clocked
    assert all(not x.clocks for x in md.variables if x.causality == "parameter")
    assert "spring" in clocks[1].description and "ball" in clocks[0].description


def test_single_rate_graph_gets_one_clock():
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=1e-2))
    gm.add_node(BallNode(name="ball", timestep=1e-2, initial_position=1.0))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    md = build_model_description(gm, model_name="m", multi_clock=True)
    assert len(md.clocks()) == 1
    assert all(x.clocks == (md.clocks()[0].value_reference,)
               for x in md.variables if x.causality == "output")


def test_xml_clock_elements_and_attributes(multirate_graph):
    md = build_model_description(multirate_graph, model_name="m", multi_clock=True)
    root = ET.fromstring(md.to_xml())
    mv = root.find("ModelVariables")
    clocks = [e for e in mv if e.tag == "Clock"]
    assert len(clocks) == 2
    for e in clocks:
        assert e.get("intervalVariability") == "constant"
        assert e.get("causality") == "input" and e.get("variability") == "discrete"
        assert "start" not in e.attrib and "Dimension" not in [c.tag for c in e]
    spring_pos = next(e for e in mv if e.get("name") == "spring.position")
    slow_vr = next(e.get("valueReference") for e in clocks if e.get("intervalDecimal") == "0.05")
    assert spring_pos.get("clocks") == slow_vr
    # clocked outputs are still declared outputs / initial unknowns
    ms = root.find("ModelStructure")
    out_vrs = {e.get("valueReference") for e in ms if e.tag == "Output"}
    assert spring_pos.get("valueReference") in out_vrs


def test_multi_clock_changes_the_instantiation_token(multirate_graph):
    a = build_model_description(multirate_graph, model_name="m")
    b = build_model_description(multirate_graph, model_name="m", multi_clock=True)
    assert a.instantiation_token != b.instantiation_token
    assert b.instantiation_token == build_model_description(
        multirate_graph, model_name="m", multi_clock=True).instantiation_token


def test_selected_outputs_only_emit_needed_clocks(multirate_graph):
    md = build_model_description(multirate_graph, model_name="m", multi_clock=True,
                                 selected_outputs=[("ball", "position")])
    # every exported node still gets a clock (the input into spring needs its own)
    assert len(md.clocks()) == 2
    assert [v.name for v in md.variables if v.causality == "output"] == ["ball.position"]


def test_fmpy_validates_a_multi_clock_description(multirate_graph, tmp_path):
    fmpy = pytest.importorskip("fmpy")
    from fmpy.model_description import read_model_description
    from fmpy.validation import validate_model_description

    md = build_model_description(multirate_graph, model_name="MultiRate", multi_clock=True)
    fmu = str(tmp_path / "multirate.fmu")
    with zipfile.ZipFile(fmu, "w") as zf:
        zf.writestr("modelDescription.xml", md.to_xml())
    parsed = read_model_description(fmu, validate=True, validate_model_structure=True)
    assert validate_model_description(parsed) == []
    clocks = [v for v in parsed.modelVariables if v.type == "Clock"]
    assert len(clocks) == 2
    assert all(c.intervalVariability == "constant" for c in clocks)
    spring_pos = next(v for v in parsed.modelVariables if v.name == "spring.position")
    assert spring_pos.clocks and spring_pos.clocks[0].name == "clock_1"
    ext = next(v for v in parsed.modelVariables if v.name == "spring.anchor_position")
    assert ext.causality == "input" and ext.clocks[0].name == "clock_1"


def test_external_inputs_of_a_real_graph_are_fmu_inputs(multirate_graph):
    """A GraphManager keeps ``_external_inputs``; the builder used to look
    for a ``_external_input_specs`` dict that never existed, so real graphs
    exported no ``<input>`` variables at all."""
    md = build_model_description(multirate_graph, model_name="m")
    inputs = [v for v in md.variables if v.causality == "input"]
    assert [v.name for v in inputs] == ["spring.anchor_position"]
    assert inputs[0].unit == "m"                      # from boundary_input_spec
    assert inputs[0].description                      # ditto
    only = build_model_description(multirate_graph, model_name="m",
                                   selected_inputs=["nope"])
    assert not [v for v in only.variables if v.causality == "input"]


def test_default_step_size_is_the_fastest_node_timestep(multirate_graph):
    md = build_model_description(multirate_graph, model_name="m")
    assert md.default_step_size == 1e-2
