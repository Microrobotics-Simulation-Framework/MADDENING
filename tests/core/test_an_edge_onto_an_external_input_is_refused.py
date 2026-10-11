"""A field that is the target of an edge and a declared external input: the
graph is refused, by every door, with one message that says what to do.

The step writes a node's external inputs into its ``boundary_inputs`` after
its edges, and an input the caller does not feed is zeros.  So with both on
one field the edge never reached the node: ``b`` below read 0.0 instead of
the 7.0 the edge carries (3.0 when the input was fed 3.0; never 7, and never
10 under an additive edge), ``validate()`` returned nothing and no entry
point warned (MADD-ANO-265).  ``compile()`` now raises one ``ValueError``
that names every such edge, ``validate()`` lists the same text as one
``ERROR`` line, and every entry point that compiles on demand raises it.

The rule is asked of the names alone, so the tests of the kinds of edge
build the graph and do not compile it; the controls beside them (the edge
alone, the input alone, the two on different fields) step.
"""

from __future__ import annotations

import ast
import json
import re

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.fmi import build_model_description
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

from tests.property import geometry_graphs as gg

DT = 0.01
HEAD = "an edge and a declared external input target the same field"


class Holds(SimulationNode):
    """Keeps the constant ``y`` it was built with."""

    def __init__(self, name, timestep, value=7.0):
        super().__init__(name, timestep, value=value)

    def initial_state(self):
        return {"y": jnp.asarray(self.params["value"], jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"y": state["y"]}


class Reads(SimulationNode):
    """Stores what it reads on ``p`` (and on ``q``), -99 where nothing arrives."""

    def __init__(self, name, timestep):
        super().__init__(name, timestep)

    def initial_state(self):
        return {"seen": jnp.asarray(-1.0, jnp.float32), "seen_q": jnp.asarray(-1.0, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"seen": jnp.asarray(boundary_inputs.get("p", -99.0), jnp.float32),
                "seen_q": jnp.asarray(boundary_inputs.get("q", -99.0), jnp.float32)}


class Port(SimulationNode):
    """``x <- u`` on ``n`` cells, and a kept field ``g`` of shape ``(2, 3)``."""

    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)

    def initial_state(self):
        n = int(self.params["n"])
        return {"x": jnp.arange(1, n + 1, dtype=jnp.float32),
                "g": jnp.ones((2, 3), jnp.float32)}

    def boundary_input_spec(self):
        n = int(self.params["n"])
        return {"u": BoundaryInputSpec(shape=(n,), dtype=jnp.float32,
                                       default=jnp.zeros(n, jnp.float32))}

    def update(self, state, boundary_inputs, dt):
        return {"x": boundary_inputs.get("u", jnp.zeros_like(state["x"])), "g": state["g"]}


REGISTRY = {"Holds": Holds, "Reads": Reads}


def _pair(*, edge=True, external=True, external_first=False, field="p", **edge_kwargs):
    """``a`` (holds 7) and ``b`` (reads ``p``): the edge ``a.y -> b.p``, the
    external input ``b.<field>``, or both, declared in either order."""
    gm = GraphManager()
    gm.add_node(Holds("a", DT))
    gm.add_node(Reads("b", DT))
    if external and external_first:
        gm.add_external_input("b", field)
    if edge:
        gm.add_edge("a", "b", "y", "p", **edge_kwargs)
    if external and not external_first:
        gm.add_external_input("b", field)
    return gm


def _refusal(gm) -> str:
    """The text ``compile()`` refuses *gm* with, which ``validate()`` must
    list as exactly one ``ERROR`` line."""
    with pytest.raises(ValueError) as caught:
        gm.compile()
    assert type(caught.value) is ValueError
    text = str(caught.value)
    assert text.startswith(HEAD)
    assert [i for i in gm.validate() if i.startswith("ERROR")] == [f"ERROR: {text}"]
    return text


def _seen(gm, field="seen") -> float:
    return float(gm.get_node_state("b")[field])


# ---------------------------------------------------------------------------
# The controls: what each of the two does alone, and on different fields
# ---------------------------------------------------------------------------


def test_the_edge_alone_delivers_and_the_external_input_alone_is_what_is_fed_or_zero():
    alone = _pair(external=False)
    assert [i for i in alone.validate() if i.startswith("ERROR")] == []
    alone.step()
    assert _seen(alone) == 7.0

    fed = _pair(edge=False)
    fed.step(external_inputs={"b": {"p": jnp.asarray(3.0, jnp.float32)}})
    assert _seen(fed) == 3.0
    unfed = _pair(edge=False)
    unfed.step()
    assert _seen(unfed) == 0.0


def test_an_edge_and_an_external_input_on_two_fields_of_one_node_both_arrive():
    gm = _pair(field="q")
    assert [i for i in gm.validate() if i.startswith("ERROR")] == []
    gm.step(external_inputs={"b": {"q": jnp.asarray(3.0, jnp.float32)}})
    assert (_seen(gm), _seen(gm, "seen_q")) == (7.0, 3.0)


def test_an_external_input_of_the_same_field_name_on_another_node_is_no_collision():
    gm = _pair(external=False)
    gm.add_node(Reads("c", DT))
    gm.add_external_input("c", "p")
    assert [i for i in gm.validate() if i.startswith("ERROR")] == []
    gm.step(external_inputs={"c": {"p": jnp.asarray(3.0, jnp.float32)}})
    assert _seen(gm) == 7.0
    assert float(gm.get_node_state("c")["seen"]) == 3.0


# ---------------------------------------------------------------------------
# The refusal and its message
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("external_first", [False, True], ids=["edge_first", "input_first"])
def test_compile_raises_one_value_error_and_validate_lists_it_in_either_order(external_first):
    text = _refusal(_pair(external_first=external_first))
    assert "edge a.y -> b.p and external input b.p" in text


def test_the_message_names_the_pair_and_both_fixes_and_only_suggests_how_it_came_about():
    text = _refusal(_pair())
    # The pair: the edge as source.field -> target.field, the input as target.field.
    assert "edge a.y -> b.p" in text and "external input b.p" in text
    # What the library does with the pair, and so why it is refused.
    assert "replaces whatever an edge delivers" in text
    assert "zeros when it is not fed" in text
    assert "would never reach the node" in text
    # How a graph may come to have both: offered, not asserted.
    assert "One way to get here" in text
    for stated_as_fact in ("you declared", "you added", "your builder", "because a builder"):
        assert stated_as_fact not in text
    # Both fixes, with the calls spelled for this pair.
    assert "To use the edge, do not declare the external input" in text
    assert "there is no remove_external_input" in text
    assert "add_external_input('b', 'p', ...)" in text
    assert "To use the external input, remove the edge: remove_edge('a', 'b', 'y', 'p')" in text
    # For someone who has just upgraded.
    assert "Before 0.4.0 such a graph compiled" in text
    assert "0.3.x did not fill an omitted input with zeros" in text
    # One paragraph: validate() lists it as one line.
    assert "\n" not in text


def test_the_remove_edge_call_the_message_spells_leaves_a_graph_that_reads_the_input():
    gm = _pair()
    text = _refusal(gm)
    (spelled,) = re.findall(r"remove_edge\(([^)]*)\)", text)
    gm.remove_edge(*ast.literal_eval(f"({spelled})"))
    assert [i for i in gm.validate() if i.startswith("ERROR")] == []
    gm.step(external_inputs={"b": {"p": jnp.asarray(3.0, jnp.float32)}})
    assert _seen(gm) == 3.0


def test_the_declaration_the_message_says_to_leave_out_is_the_colliding_one():
    text = _refusal(_pair())
    (node, field) = re.findall(r"add_external_input\('([^']*)', '([^']*)', \.\.\.\)", text)[0]
    # Left out, the graph is the edge alone, which the first control steps.
    assert (node, field) == ("b", "p")


# ---------------------------------------------------------------------------
# Every kind of edge
# ---------------------------------------------------------------------------


def _vector_pair(**edge_kwargs):
    gm = GraphManager()
    gm.add_node(Port("a", DT, n=3))
    gm.add_node(Port("b", DT, n=2))
    gm.add_edge("a", "b", "x", "u", **edge_kwargs)
    return gm, ("b", "u", (2,)), ["edge a.x -> b.u and external input b.u"]


def _in_a_group():
    gm = _pair(external=False)
    gm.add_edge("b", "a", "seen", "back")
    gm.add_coupling_group(["a", "b"], max_iterations=3)
    return gm, ("b", "p", ()), ["edge a.y -> b.p and external input b.p"]


def _self_edge():
    gm = GraphManager()
    gm.add_node(Reads("b", DT))
    gm.add_edge("b", "b", "seen", "p")
    return gm, ("b", "p", ()), ["edge b.seen -> b.p and external input b.p"]


def _two_additive_edges():
    gm = _pair(external=False, additive=True)
    gm.add_node(Holds("c", DT, value=5.0))
    gm.add_edge("c", "b", "y", "p", additive=True)
    return gm, ("b", "p", ()), ["edge a.y -> b.p and external input b.p",
                                "edge c.y -> b.p and external input b.p"]


def _into_a_wrapped_node(wrap):
    gm = GraphManager()
    gm.add_node(Holds("a", DT))
    gm.add_node(wrap(HeatNode("rod", DT, n_cells=8, initial_temperature=300.0)))
    gm.add_edge("a", "rod", "y", "left_temperature")
    return (gm, ("rod", "left_temperature", ()),
            ["edge a.y -> rod.left_temperature and external input rod.left_temperature"])


def _sharded(node):
    return ShardedStencilNode(node, create_device_mesh(shape=(1,)), {"devices": 0},
                              boundary="edge")


def _hybrid(node):
    return HybridNode(node, lambda state, boundary_inputs, dt: state)


KINDS = {
    "plain": lambda: (_pair(external=False), ("b", "p", ()),
                      ["edge a.y -> b.p and external input b.p"]),
    "additive": lambda: (_pair(external=False, additive=True), ("b", "p", ()),
                         ["edge a.y -> b.p and external input b.p"]),
    "two_additive_edges": _two_additive_edges,
    "transformed": lambda: (_pair(external=False, transform=lambda v: 2.0 * v), ("b", "p", ()),
                            ["edge a.y -> b.p and external input b.p"]),
    "with_units": lambda: (_pair(external=False, source_units="m", target_units="m"),
                           ("b", "p", ()), ["edge a.y -> b.p and external input b.p"]),
    "mapped": lambda: _vector_pair(mapping=matrix_mapping(np.ones((2, 3), np.float32))),
    "mapped_and_transformed": lambda: _vector_pair(
        mapping=matrix_mapping(np.ones((2, 3), np.float32)), transform=lambda v: 2.0 * v),
    "with_geometry": lambda: _vector_pair(mapping=gg.geom_matrix_mapping(2, 3),
                                          geometry=("source", "g")),
    "in_a_coupling_group": _in_a_group,
    "from_the_node_to_itself": _self_edge,
    "into_a_sharded_node": lambda: _into_a_wrapped_node(_sharded),
    "into_a_hybrid_node": lambda: _into_a_wrapped_node(_hybrid),
}


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_every_kind_of_edge_onto_an_external_input_is_refused(kind):
    gm, (node, field, shape), pairs = KINDS[kind]()
    before = [i for i in gm.validate() if i.startswith("ERROR")]
    assert not any(HEAD in i for i in before)
    gm.add_external_input(node, field, shape=shape)
    with pytest.raises(ValueError) as caught:
        gm.compile()
    text = str(caught.value)
    assert text.startswith(HEAD)
    for pair in pairs:
        assert pair in text
    # validate() lists it once, beside whatever it said of the graph before.
    after = [i for i in gm.validate() if i.startswith("ERROR")]
    assert after.count(f"ERROR: {text}") == 1
    assert sorted(after) == sorted(before + [f"ERROR: {text}"])


def test_several_pairs_give_one_error_that_names_them_all_in_the_order_of_the_edges():
    gm = GraphManager()
    for name in ("a", "c"):
        gm.add_node(Holds(name, DT))
    for name in ("b", "d", "e"):
        gm.add_node(Reads(name, DT))
    gm.add_edge("a", "d", "y", "q")                 # onto an external input
    gm.add_edge("a", "b", "y", "p", additive=True)  # onto an external input
    gm.add_edge("c", "b", "y", "p", additive=True)  # onto the same one
    gm.add_edge("c", "e", "y", "p")                 # onto no external input
    gm.add_external_input("b", "p")
    gm.add_external_input("d", "q")
    gm.add_external_input("e", "q")                 # no edge delivers to it
    text = _refusal(gm)
    named = re.findall(r"edge (\S+) -> (\S+) and external input (\S+?)[;.] ", text)
    assert named == [("a.y", "d.q", "d.q"), ("a.y", "b.p", "b.p"), ("c.y", "b.p", "b.p")]
    assert re.findall(r"remove_edge\(([^)]*)\)", text) == [
        "'a', 'd', 'y', 'q'", "'a', 'b', 'y', 'p'", "'c', 'b', 'y', 'p'"]
    # One declaration per input, though two edges deliver to b.p.
    assert re.findall(r"add_external_input\(([^)]*)\)", text) == [
        "'d', 'q', ...", "'b', 'p', ..."]
    assert "e.p" not in text and "e.q" not in text


def test_an_edge_added_twice_is_named_once():
    gm = _pair(external=False, additive=True)
    gm.add_edge("a", "b", "y", "p", additive=True)
    gm.add_external_input("b", "p")
    text = _refusal(gm)
    assert text.count("edge a.y -> b.p and external input b.p") == 1
    assert text.count("remove_edge('a', 'b', 'y', 'p')") == 1
    # ... and that one call removes both, which is why it is spelled once.
    gm.remove_edge("a", "b", "y", "p")
    assert gm.edges == []


# ---------------------------------------------------------------------------
# Every door
# ---------------------------------------------------------------------------

DOORS = {
    "step": lambda gm: gm.step(),
    "run": lambda gm: gm.run(2),
    "run_scan": lambda gm: gm.run_scan(2),
    "run_scan_with_history": lambda gm: gm.run_scan_with_history(2),
    "run_sweep": lambda gm: gm.run_sweep(
        2, {"a": {"y": jnp.ones(2, jnp.float32)},
            "b": {"seen": jnp.zeros(2, jnp.float32), "seen_q": jnp.zeros(2, jnp.float32)}}),
    "run_adaptive": lambda gm: gm.run_adaptive(2 * DT),
    "run_adaptive_scan": lambda gm: gm.run_adaptive_scan(2 * DT),
}


def test_the_doors_are_every_stepping_entry_point_of_the_graph():
    stepping = {name for name in vars(GraphManager)
                if name == "step" or (name.startswith("run") and not name.startswith("_"))}
    assert stepping == set(DOORS)


@pytest.mark.parametrize("door", sorted(DOORS))
def test_every_stepping_entry_point_refuses_with_the_text_compile_raises(door):
    expected = _refusal(_pair())
    gm = _pair()
    with pytest.raises(ValueError) as caught:
        DOORS[door](gm)
    assert str(caught.value) == expected
    # Nothing was stepped, and the next call is refused again.
    assert _seen(gm) == -1.0
    with pytest.raises(ValueError, match=HEAD):
        DOORS[door](gm)


@pytest.mark.parametrize("added_last", ["external_input", "edge"])
def test_a_compiled_graph_that_gains_the_other_one_is_refused_at_its_next_step(added_last):
    gm = _pair(edge=added_last != "edge", external=added_last != "external_input")
    gm.step()
    held = _seen(gm)
    if added_last == "edge":
        gm.add_edge("a", "b", "y", "p")
    else:
        gm.add_external_input("b", "p")
    expected = _refusal(_pair())
    for door in sorted(DOORS):
        with pytest.raises(ValueError) as caught:
            DOORS[door](gm)
        assert str(caught.value) == expected, door
    # The refused calls stepped nothing.
    assert _seen(gm) == held


def test_a_config_that_holds_both_loads_and_is_refused_when_it_is_compiled():
    config = json.loads(json.dumps(_pair().to_dict()))
    assert len(config["edges"]) == 1 and len(config["external_inputs"]) == 1
    loaded = GraphManager.from_dict(config, REGISTRY)
    expected = _refusal(_pair())
    assert _refusal(loaded) == expected
    with pytest.raises(ValueError) as caught:
        loaded.step()
    assert str(caught.value) == expected
    # The config with either one taken out is a graph that steps.
    for dropped, reads in (("external_inputs", 7.0), ("edges", 0.0)):
        kept = GraphManager.from_dict({**config, dropped: []}, REGISTRY)
        kept.step()
        assert _seen(kept) == reads


def _plant(*, edge=True, external=True):
    """Built-in nodes, which an FMU exports: a table under a spring's anchor."""
    gm = GraphManager()
    gm.add_node(TableNode("table", DT))
    gm.add_node(SpringDamperNode("spring", DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    if edge:
        gm.add_edge("table", "spring", "position", "anchor_position")
    if external:
        gm.add_external_input("spring", "anchor_position")
    return gm


def test_no_fmu_is_described_from_such_a_graph():
    # The export takes a compiled graph, unchanged since its compile(), and
    # compiles nothing itself: so a graph compile() refuses has no FMU.
    never_compiled = _plant()
    with pytest.raises(ValueError, match="the graph has not been compiled"):
        build_model_description(never_compiled, model_name="Plant")
    text = _refusal(never_compiled)
    assert ("edge table.position -> spring.anchor_position and external input "
            "spring.anchor_position") in text

    gained = _plant(external=False)
    gained.compile()
    gained.add_external_input("spring", "anchor_position")
    with pytest.raises(ValueError, match=r"changed since its last compile\(\)"):
        build_model_description(gained, model_name="Plant")
    # ... and the compile() that message asks for is the refusal.
    assert _refusal(gained) == text
    # With the edge removed, the same graph is exported with its input.
    gained.remove_edge("table", "spring", "position", "anchor_position")
    gained.compile()
    described = build_model_description(gained, model_name="Plant")
    assert [v.name for v in described.variables if v.causality == "input"] == [
        "spring.anchor_position"]
