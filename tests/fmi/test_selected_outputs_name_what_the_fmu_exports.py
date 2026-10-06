"""``selected_outputs`` names outputs the FMU exports, or the export is refused.

``build_model_description(selected_outputs=[...])`` walked the graph's state
fields and kept those it found in the selection, so a pair that named
nothing -- a misspelt field, a node the graph does not have, a node the
stability filter keeps out -- exported nothing and said nothing:
``selected_outputs=[("spring", "positon")]`` gave an FMU with no outputs.
``selected_inputs`` has refused an unknown name since 0.4.0's fix of the
same silence; the outputs now do too, and an entry that is not a
``(node, field)`` pair of names is refused by name instead of being
unpacked letter by letter.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.fmi import build_model_description
from maddening.nodes import BallNode, HeatNode, SpringDamperNode, TableNode
from maddening.nodes.heart_pump import HeartPumpNode


@stability(StabilityLevel.EVOLVING)
class _Evolving(SimulationNode):
    """An EVOLVING node: in an FMU only with ``include_evolving=True``."""

    def __init__(self, name, timestep):
        super().__init__(name, timestep)

    def initial_state(self):
        return {"level": jnp.asarray(0.0, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"level": state["level"] + dt}


def _graph():
    gm = GraphManager()
    gm.add_node(TableNode("table", 0.01, position=0.25))
    gm.add_node(BallNode("ball", 0.01, initial_position=1.5))
    gm.add_node(SpringDamperNode("spring", 0.02, stiffness=30.0, initial_position=0.5))
    gm.add_node(HeatNode("rod.A", 0.01, n_cells=4, thermal_diffusivity=0.01))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_external_input("spring", "anchor_position")
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def gm():
    return _graph()


def _outputs(md):
    return [v.name for v in md.variables if v.causality == "output"]


def test_a_selection_exports_exactly_the_outputs_it_names(gm):
    md = build_model_description(gm, model_name="m", selected_outputs=[
        ("spring", "position"), ("ball", "velocity"), ("rod.A", "temperature")])
    assert sorted(_outputs(md)) == ["ball.velocity", "rod.A.temperature", "spring.position"]
    # a list entry is a pair too, an empty selection exports none, and None exports all
    md = build_model_description(gm, model_name="m", selected_outputs=[["spring", "velocity"]])
    assert _outputs(md) == ["spring.velocity"]
    assert _outputs(build_model_description(gm, model_name="m", selected_outputs=[])) == []
    everything = _outputs(build_model_description(gm, model_name="m"))
    assert {"spring.position", "spring.velocity", "ball.position", "table.position"} <= set(everything)
    # with clocks: the selection still decides, and is checked first
    md = build_model_description(gm, model_name="m", multi_clock=True,
                                 selected_outputs=[("spring", "position")])
    assert _outputs(md) == ["spring.position"]


@pytest.mark.parametrize("pair, said", [
    (("spring", "positon"), "node 'spring' has no state field 'positon' (its state fields: "
                            "['position', 'velocity'])"),
    (("spring", "anchor_position"), "node 'spring' has no state field 'anchor_position'"),
    (("spring", "params.stiffness"), "node 'spring' has no state field 'params.stiffness'"),
    (("sprng", "position"), "the graph has no node 'sprng' (its nodes: ['ball', 'rod.A', "
                            "'spring', 'table'])"),
    (("rod", "A.temperature"), "the graph has no node 'rod'"),
    (("spring", ""), "node 'spring' has no state field ''"),
    (("", "position"), "the graph has no node ''"),
])
def test_a_pair_that_names_no_state_field_is_refused_naming_it(gm, pair, said):
    """It exported nothing, in silence -- alone, and beside pairs that did
    name outputs (the FMU then lacked one output and nobody was told)."""
    for selection in ([pair], [("spring", "position"), pair, ("ball", "position")]):
        with warnings.catch_warnings():
            # Refused before anything is built or warned about: the input
            # this export leaves out would be warned about ("held at zero")
            # by an export that went ahead.
            warnings.simplefilter("error")
            with pytest.raises(ValueError) as caught:
                build_model_description(gm, model_name="m", selected_outputs=selection,
                                        selected_inputs=[])
        text = str(caught.value)
        assert "selected_outputs names output(s) this FMU cannot export" in text
        assert repr(pair) in text and said in text, text


def test_every_such_pair_is_named_at_once(gm):
    with pytest.raises(ValueError) as caught:
        build_model_description(gm, model_name="m", selected_outputs=[
            ("spring", "positon"), ("nobody", "x"), ("ball", "position")])
    text = str(caught.value)
    assert "('spring', 'positon')" in text and "('nobody', 'x')" in text
    assert "('ball', 'position')" not in text


@pytest.mark.parametrize("selection, entry", [
    ("spring.position", None),                       # the form selected_inputs takes
    (["spring.position"], "'spring.position'"),
    (["ab"], "'ab'"),                                # unpacked as ('a', 'b'): exported nothing
    ([("spring", "position", "x")], "('spring', 'position', 'x')"),
    ([("spring",)], "('spring',)"),
    ([("spring", 3)], "('spring', 3)"),
    ([("spring", None)], "('spring', None)"),
    ([{"spring": "position"}], "{'spring': 'position'}"),
    ([b"sp"], "b'sp'"),
])
def test_an_entry_that_is_not_a_pair_of_names_is_refused(gm, selection, entry):
    with pytest.raises(ValueError) as caught:
        build_model_description(gm, model_name="m", selected_outputs=selection)
    text = str(caught.value)
    assert "selected_outputs takes (node, field) pairs of names" in text
    if entry is None:
        assert "got the string 'spring.position'" in text
    else:
        assert f"got the entry {entry}" in text
    if selection == ["spring.position"]:
        assert "pass ('spring', 'position')" in text          # the dotted name, split for them


def test_an_output_of_a_node_this_fmu_does_not_export_is_refused_by_name():
    """The stability filter: an EVOLVING node's field without
    ``include_evolving``, and an EXPERIMENTAL node's with it or without,
    were dropped in silence like a misspelt one."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", 0.01))
    gm.add_node(_Evolving("ev", 0.01))
    gm.add_node(HeartPumpNode("pump", 0.01))
    gm.compile()
    pump_field = sorted(gm.get_node_state("pump"))[0]
    with pytest.raises(ValueError, match="is not a stability level this FMU exports") as caught:
        build_model_description(gm, model_name="m", selected_outputs=[("ev", "level")])
    assert "('ev', 'level')" in str(caught.value) and "_Evolving" in str(caught.value)
    md = build_model_description(gm, model_name="m", include_evolving=True,
                                 selected_outputs=[("ev", "level"), ("spring", "position")])
    assert sorted(_outputs(md)) == ["ev.level", "spring.position"]
    for include_evolving in (False, True):
        with pytest.raises(ValueError, match="HeartPumpNode"):
            build_model_description(gm, model_name="m", include_evolving=include_evolving,
                                    selected_outputs=[("pump", pump_field)])
    # a field the unexported node does not have is named as that
    with pytest.raises(ValueError, match="node 'pump' has no state field 'nothing'"):
        build_model_description(gm, model_name="m", selected_outputs=[("pump", "nothing")])


def test_the_same_mistake_in_selected_inputs_is_still_refused(gm):
    """The sibling option, pinned beside it: an unknown input name, and a
    string where a list of names belongs."""
    with pytest.raises(ValueError, match="selected_inputs names .*anchor_positon"):
        build_model_description(gm, model_name="m", selected_inputs=["spring.anchor_positon"])
    with pytest.raises(ValueError, match="selected_inputs must be an iterable of input names"):
        build_model_description(gm, model_name="m", selected_inputs="spring.anchor_position")
