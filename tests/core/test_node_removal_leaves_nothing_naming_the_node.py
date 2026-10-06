"""After a node is removed or replaced, nothing the graph holds names a node
it does not have.

``remove_node`` took the node's edges, external inputs, live parameters and
``ParamSpec`` overrides with it, and left a coupling group listing it
(MADD-ANO-214): ``validate``, ``compile`` and every step then failed
("coupling group references non-existent node"), and the graph's own
``to_dict()`` did not load.  ``DELETE /graph/nodes/{name}`` answered 200 to
it, and no route removes a group.

The rule, the one edges follow: what names the node goes with it.  A group
loses the member and keeps its options; a group of one is no group.  An
interface mapping on an edge *between two other nodes* that was built from
the node's points cannot lose the reference, so the removal is refused.
``replace_node`` puts a node of the same name back, and keeps everything.

The invariant every test ends on: the graph's config, through JSON, reloads
and steps as the graph does.
"""

from __future__ import annotations

import itertools
import json
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.core.coupling.mapping import nearest_neighbor_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.heat import HeatNode
from maddening.surrogates.replace import replace_node

#: ``remove_node`` says which coupling group changed; one test reads it.
pytestmark = pytest.mark.filterwarnings("ignore:Removing node:UserWarning")

REGISTRY = {"HeatNode": HeatNode}
NAMES = ("a", "b", "c")
OPTIONS = dict(max_iterations=5, tolerance=1e-5, solver="fori", acceleration="fixed",
               relaxation=0.75, iteration_mode="jacobi")


def _rod(name: str, level: float = 0.5) -> HeatNode:
    return HeatNode(name, 0.01, n_cells=6, length=1.0, thermal_diffusivity=0.05,
                    initial_temperature=np.linspace(level, level + 1.0, 6).tolist())


def _ring(members=NAMES, *, extra=(), **options) -> GraphManager:
    """Rods ``a``, ``b``, ``c`` (and *extra*), each heating the next of
    *members* around a ring, *members* in one coupling group; an external
    input into ``c``'s left boundary."""
    gm = GraphManager()
    for i, name in enumerate((*NAMES, *extra)):
        gm.add_node(_rod(name, 0.5 + 0.25 * i))
    for source, target in zip(members, (*members[1:], members[0])):
        gm.add_edge(source, target, "temperature", "heat_source")
    gm.add_external_input("c", "left_temperature", ())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # options one another make inert
        gm.add_coupling_group(list(members), **{**OPTIONS, **options})
    return gm


def _compiled(gm: GraphManager) -> GraphManager:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def _named_by(gm: GraphManager) -> set[str]:
    """Every node name the graph's structures hold."""
    names = {n for n in (*gm._state, *gm.params.get("nodes", {}))  # noqa: SLF001
             if not n.startswith("_")}
    for edge in gm._edges:  # noqa: SLF001
        names |= {edge.source_node, edge.target_node}
    names |= {ei.target_node for ei in gm._external_inputs}  # noqa: SLF001
    for group in gm._coupling_groups:  # noqa: SLF001
        names |= set(group.nodes) | set(group.accelerated_fields or {})
    names |= {key.split(".")[0] for key in gm._param_spec_overrides if "->" not in key}  # noqa: SLF001
    for edge in gm._edges:  # noqa: SLF001
        spec = getattr(edge.mapping, "spec", None)
        for ref in (getattr(spec, "points", None) or {}).values():
            if isinstance(ref, dict) and "node" in ref:
                names.add(ref["node"])
    return names


def _reloads_and_steps_as_the_graph(gm: GraphManager) -> GraphManager:
    assert _named_by(gm) <= set(gm._nodes), _named_by(gm) - set(gm._nodes)  # noqa: SLF001
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert not [issue for issue in gm.validate() if issue.startswith("ERROR")]
        config = json.loads(json.dumps(gm.to_dict()))
        fresh = GraphManager.from_dict(config, REGISTRY)
        fresh.compile()
        gm.compile()
        for name in gm.node_names:
            fresh.set_node_state(name, gm.get_node_state(name))
        inputs = {(ei.target_node, ei.target_field): jnp.float32(0.25)
                  for ei in gm._external_inputs}  # noqa: SLF001
        for graph in (gm, fresh):
            for _ in range(3):
                graph.step(external_inputs={
                    node: {field: value} for (node, field), value in inputs.items()} or None)
    assert [g.to_dict() for g in fresh._coupling_groups] == \
        [g.to_dict() for g in gm._coupling_groups]  # noqa: SLF001
    for name in gm.node_names:
        for field, value in gm.get_node_state(name).items():
            np.testing.assert_array_equal(np.asarray(value),
                                          np.asarray(fresh.get_node_state(name)[field]))
    return fresh


# ---------------------------------------------------------------------------
# remove_node and the coupling group
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gone", NAMES)
def test_a_group_of_three_loses_the_member_and_keeps_its_options(gone):
    gm = _compiled(_ring())
    gm.step(external_inputs={"c": {"left_temperature": jnp.float32(0.25)}})
    gm.remove_node(gone)
    (group,) = gm._coupling_groups  # noqa: SLF001
    assert group.nodes == frozenset(NAMES) - {gone}
    for option, value in OPTIONS.items():
        assert getattr(group, option) == value
    assert gm._dirty  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)


@pytest.mark.parametrize("gone", ["a", "b"])
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_group_left_with_one_member_goes_with_the_node(gone, solver):
    gm = _compiled(_ring(("a", "b"), solver=solver, iteration_mode="gauss-seidel",
                         acceleration="none", relaxation=1.0))
    gm.step(external_inputs={"c": {"left_temperature": jnp.float32(0.25)}})
    gm.remove_node(gone)
    assert gm._coupling_groups == []  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)


def test_a_group_the_node_is_not_in_is_left_as_it_was():
    gm = _compiled(_ring(("a", "b")))
    (before,) = gm._coupling_groups  # noqa: SLF001
    gm.remove_node("c")
    assert gm._coupling_groups == [before] and gm._coupling_groups[0] is before  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)


def test_only_the_group_that_names_the_node_changes():
    gm = _ring(("a", "b", "c"), extra=("d", "e"))
    gm.add_edge("d", "e", "temperature", "heat_source")
    gm.add_edge("e", "d", "temperature", "heat_source")
    other = gm.add_coupling_group(["d", "e"], max_iterations=3)
    gm.remove_node("b")
    assert [g.nodes for g in gm._coupling_groups] == [frozenset("ac"), frozenset("de")]  # noqa: SLF001
    assert gm._coupling_groups[1] is other  # noqa: SLF001
    gm.remove_node("d")
    assert [g.nodes for g in gm._coupling_groups] == [frozenset("ac")]  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)


@pytest.mark.parametrize("accelerated, gone, left", [
    ({"a": ("temperature",), "b": ("temperature",)}, "a", {"b": ("temperature",)}),
    ({"a": ("temperature",), "b": ("temperature",)}, "c", {"a": ("temperature",),
                                                           "b": ("temperature",)}),
    # the only entry was the removed node's: an empty selection is one the
    # group refuses to be built with, and ``None`` is the default it names
    ({"a": ("temperature",)}, "a", None),
])
def test_an_accelerated_fields_entry_goes_with_its_node(accelerated, gone, left):
    gm = _compiled(_ring(solver="fori", acceleration="iqn-ils", iteration_mode="gauss-seidel",
                         relaxation=1.0, accelerated_fields=accelerated))
    gm.remove_node(gone)
    (group,) = gm._coupling_groups  # noqa: SLF001
    assert group.accelerated_fields == left
    assert group.acceleration == "iqn-ils"
    _reloads_and_steps_as_the_graph(gm)


@pytest.mark.parametrize("members, said", [
    (NAMES, r"took it out of the coupling group of \['a', 'b', 'c'\], which keeps its "
            r"options over \['b', 'c'\]"),
    (("a", "b"), r"removed the coupling group of \['a', 'b'\] with it"),
])
def test_removing_a_member_warns_once_naming_the_group_and_nothing_else(members, said):
    """A node added back under the name used to be a member again, the
    group having gone on naming it; that it no longer is, is said.  And the
    smaller group is built from options that were validated, and warned
    about, when the group was made: none of that is said again."""
    gm = _ring(members)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.remove_node("a")
    (warning,) = caught
    assert warning.category is UserWarning and warning.filename == __file__
    assert "add_coupling_group" in str(warning.message)
    import re
    assert re.search(said, str(warning.message)), str(warning.message)


def test_removing_a_node_no_group_names_warns_about_nothing():
    gm = _ring(("a", "b"))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.remove_node("c")


def test_a_node_added_back_under_the_name_is_not_a_member_until_the_group_is_added():
    gm = _compiled(_ring())
    gm.remove_node("b")
    gm.add_node(_rod("b"))
    assert [g.nodes for g in gm._coupling_groups] == [frozenset("ac")]  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)
    gm.remove_coupling_group(["a", "c"])
    gm.add_edge("a", "b", "temperature", "heat_source")
    gm.add_edge("b", "c", "temperature", "heat_source")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(list(NAMES), **OPTIONS)
    _reloads_and_steps_as_the_graph(gm)


def test_removing_every_node_in_every_order_leaves_a_graph_that_reloads():
    """The invariant, exhaustively over a small graph: after each accepted
    removal, the graph reloads from its own save and steps as it does."""
    for order in itertools.permutations(NAMES):
        gm = _ring()
        gm.set_param_spec("b", "thermal_diffusivity", ParamSpec(bounds=(0.0, 1.0)))
        for name in order[:2]:
            gm.remove_node(name)
            _reloads_and_steps_as_the_graph(gm)


def test_an_unknown_node_is_a_key_error_and_changes_nothing():
    gm = _ring()
    (before,) = gm._coupling_groups  # noqa: SLF001
    with pytest.raises(KeyError, match="No node named 'ghost'"):
        gm.remove_node("ghost")
    assert gm._coupling_groups == [before] and len(gm._edges) == 3  # noqa: SLF001


# ---------------------------------------------------------------------------
# A mapping built from the points of a node that is neither of its ends
# ---------------------------------------------------------------------------

def _mapped_from_a_third_node() -> GraphManager:
    """``a`` mapped onto ``b``, the source points named as ``c``'s grid
    (which is ``a``'s: the rods are alike)."""
    gm = GraphManager()
    for name in NAMES:
        gm.add_node(_rod(name))
    grid = np.asarray(gm.get_node("c").static_data["grid_x"].value)
    gm.add_edge("a", "b", "temperature", "heat_source", mapping=nearest_neighbor_mapping(
        grid, grid, source_ref={"node": "c", "field": "grid_x"},
        target_ref={"node": "b", "field": "grid_x"}))
    gm.add_edge("c", "a", "temperature", "heat_source")
    return gm


def test_a_node_whose_points_a_remaining_mapping_was_built_from_is_not_removed():
    gm = _compiled(_mapped_from_a_third_node())
    edges, nodes = list(gm._edges), list(gm._nodes)  # noqa: SLF001
    with pytest.raises(ValueError, match=r"Cannot remove node 'c'.*a\.temperature->b\.heat_source"):
        gm.remove_node("c")
    assert list(gm._edges) == edges and list(gm._nodes) == nodes  # noqa: SLF001
    assert not gm._dirty  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)
    # ... and is, once the edge that holds the reference is gone.
    gm.remove_edge("a", "b", "temperature", "heat_source")
    gm.remove_node("c")
    _reloads_and_steps_as_the_graph(gm)


@pytest.mark.parametrize("gone", ["a", "b"])
def test_a_mapped_edge_goes_with_either_of_its_ends_whatever_its_points_name(gone):
    gm = _compiled(_mapped_from_a_third_node())
    gm.remove_node(gone)
    assert not gm.params.get("mappings")
    _reloads_and_steps_as_the_graph(gm)


# ---------------------------------------------------------------------------
# An edge whose mapping reads its geometry from one of its ends
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("gone", ["grid", "markers"])
def test_a_geometry_edge_goes_with_either_end_and_nothing_reads_the_removed_field(gone):
    """``geometry=(anchor, field)`` names an end of the edge itself, so the
    edge -- and the read of the field -- goes with the node."""
    from tests.core import geometry_surface_graphs as G

    gm = G.graph()
    gm.step()
    gm.remove_node(gone)
    assert gm._edges == [] and not gm.params.get("mappings")  # noqa: SLF001
    assert _named_by(gm) <= set(gm._nodes)  # noqa: SLF001
    assert not [issue for issue in gm.validate() if issue.startswith("ERROR")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
        gm.step()
        gm.to_dict()


@pytest.mark.parametrize("replaced", ["grid", "markers"])
def test_a_replaced_end_of_a_geometry_edge_keeps_the_edge_and_its_geometry(replaced):
    from tests.core import geometry_surface_graphs as G

    gm = G.graph()
    gm.step()
    before = [(e.key, e.geometry) for e in gm._edges]  # noqa: SLF001
    cls = type(gm.get_node(replaced))
    replace_node(gm, replaced, cls(replaced, G.DT))
    assert [(e.key, e.geometry) for e in gm._edges] == before  # noqa: SLF001
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
        gm.step()


# ---------------------------------------------------------------------------
# replace_node: a node of the same name comes back, and everything stays
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("replaced", NAMES)
def test_a_replaced_member_stays_in_its_group_with_every_option(replaced):
    gm = _compiled(_ring(accelerated_fields={"a": ("temperature",), "c": ("temperature",)},
                         solver="fori", acceleration="iqn-ils",
                         iteration_mode="gauss-seidel", relaxation=1.0))
    (before,) = gm._coupling_groups  # noqa: SLF001
    edges = [e.key for e in gm._edges]  # noqa: SLF001
    replace_node(gm, replaced, _rod(replaced, 2.0))
    assert gm._coupling_groups == [before] and gm._coupling_groups[0] is before  # noqa: SLF001
    assert sorted(e.key for e in gm._edges) == sorted(edges)  # noqa: SLF001
    assert [ei.target_node for ei in gm._external_inputs] == ["c"]  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)


def test_a_replaced_node_keeps_the_mappings_built_from_its_points():
    gm = _compiled(_mapped_from_a_third_node())
    replace_node(gm, "c", _rod("c"))
    assert len(gm._edges) == 2  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)


# ---------------------------------------------------------------------------
# A config edited by hand: from_dict refuses what names a node it lacks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("keep", ["coupling_groups", "edges", "external_inputs"])
def test_a_config_that_names_a_node_it_does_not_have_is_refused(keep):
    """The node deleted from a saved config by hand, and one structure left
    naming it."""
    config = json.loads(json.dumps(_ring().to_dict()))
    gone = "c"
    assert set(config) >= {"nodes", "edges", "external_inputs", "coupling_groups"}, sorted(config)
    nodes = config["nodes"]
    if isinstance(nodes, dict):
        nodes.pop(gone)
    else:
        config["nodes"] = [n for n in nodes if n.get("name") != gone]

    def names(entry) -> bool:
        return gone in json.dumps(entry)

    for section in ("coupling_groups", "edges", "external_inputs"):
        if section != keep:
            config[section] = [e for e in config[section] if not names(e)]
    assert any(names(e) for e in config[keep])
    # from_dict refuses the group; an edge or an external input is refused
    # by the compile ("references non-existent ...").
    with pytest.raises((KeyError, ValueError, RuntimeError), match="'c'"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            GraphManager.from_dict(config, REGISTRY).compile()


# ---------------------------------------------------------------------------
# DELETE /graph/nodes/{name}
# ---------------------------------------------------------------------------

def _serve(gm: GraphManager) -> tuple[SimulationServer, TestClient]:
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm)
    return server, TestClient(server.create_app(), raise_server_exceptions=False)


@pytest.mark.parametrize("members", [NAMES, ("a", "b")])
def test_the_route_removes_a_member_and_the_graph_still_steps_and_reloads(members):
    gm = _compiled(_ring(members))
    _server, client = _serve(gm)
    resp = client.delete("/graph/nodes/a")
    assert resp.status_code == 200, resp.text
    assert [sorted(g.nodes) for g in gm._coupling_groups] == \
        ([["b", "c"]] if len(members) == 3 else [])  # noqa: SLF001
    said = resp.json()["coupling_groups"]
    assert ("keeps its options over ['b', 'c']" if len(members) == 3
            else "removed the coupling group of ['a', 'b']") in said
    if len(members) == 2:
        # ... and a node no group names is removed with nothing to say.
        assert client.delete("/graph/nodes/c").json() == {"status": "ok"}
        return
    assert client.post("/graph/compile").status_code == 200
    assert client.post("/sim/step").status_code == 200
    _reloads_and_steps_as_the_graph(gm)


def test_the_route_refuses_a_node_a_remaining_mapping_was_built_from():
    gm = _compiled(_mapped_from_a_third_node())
    _server, client = _serve(gm)
    before = client.get("/graph/state").json()
    resp = client.delete("/graph/nodes/c")
    assert resp.status_code == 400, resp.text
    assert "Cannot remove node 'c'" in resp.json()["detail"]
    assert sorted(gm._nodes) == list(NAMES) and len(gm._edges) == 2  # noqa: SLF001
    assert client.get("/graph/state").json() == before
    assert client.post("/sim/step").status_code == 200


def test_the_route_forgets_a_surrogate_recorded_under_the_removed_name():
    """The server's record of a surrogate names a node too: a deactivate
    after the node was deleted used to add the recorded original back, with
    its edges."""
    gm = _compiled(_ring(("a", "b")))
    server, client = _serve(gm)
    original = gm.get_node("c")
    server._original_nodes["c"] = (original, [], list(gm._external_inputs))  # noqa: SLF001
    server._active_surrogates.add("c")  # noqa: SLF001
    assert client.delete("/graph/nodes/c").status_code == 200
    assert not server._original_nodes and not server._active_surrogates  # noqa: SLF001
    resp = client.post("/surrogate/deactivate/c")
    assert resp.status_code == 400, resp.text
    assert "c" not in gm._nodes  # noqa: SLF001


def test_the_route_drops_the_removed_node_from_the_edges_a_surrogate_record_holds():
    """... and the edges another surrogate's record holds to the removed
    node go with it, as the graph's own do: the deactivate then puts back
    the original and the edges that still have both ends."""
    gm = _compiled(_ring())
    server, client = _serve(gm)
    recorded = [e for e in gm._edges if "b" in (e.source_node, e.target_node)]  # noqa: SLF001
    assert len(recorded) == 2
    server._original_nodes["b"] = (gm.get_node("b"), recorded, [])  # noqa: SLF001
    server._active_surrogates.add("b")  # noqa: SLF001
    assert client.delete("/graph/nodes/a").status_code == 200
    kept = server._original_nodes["b"][1]  # noqa: SLF001
    assert [(e.source_node, e.target_node) for e in kept] == [("b", "c")]
    resp = client.post("/surrogate/deactivate/b")
    assert resp.status_code == 200, resp.text
    assert [sorted(g.nodes) for g in gm._coupling_groups] == [["b", "c"]]  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)


def test_a_deactivated_surrogate_is_still_a_member_of_its_group():
    gm = _compiled(_ring())
    server, client = _serve(gm)
    (before,) = gm._coupling_groups  # noqa: SLF001
    recorded = [e for e in gm._edges if "b" in (e.source_node, e.target_node)]  # noqa: SLF001
    server._original_nodes["b"] = (gm.get_node("b"), recorded, [])  # noqa: SLF001
    server._active_surrogates.add("b")  # noqa: SLF001
    resp = client.post("/surrogate/deactivate/b")
    assert resp.status_code == 200, resp.text
    assert gm._coupling_groups == [before]  # noqa: SLF001
    _reloads_and_steps_as_the_graph(gm)


@pytest.mark.xfail(strict=True, reason=(
    "POST /surrogate/deactivate (experimental) puts back the edges recorded when the "
    "surrogate was activated, not the edges the node has now: an edge added in process "
    "while the surrogate was active is dropped by the revert, with a 200"))
def test_a_deactivated_surrogate_keeps_the_edges_added_while_it_was_active():
    """The revert removes the surrogate -- and with it every edge it has --
    and adds the original with the edges of the server's record.  Nothing
    is left dangling (the graph reloads), but an edge is lost silently."""
    gm = _compiled(_ring(extra=("d",)))
    server, client = _serve(gm)
    recorded = [e for e in gm._edges if "b" in (e.source_node, e.target_node)]  # noqa: SLF001
    server._original_nodes["b"] = (gm.get_node("b"), recorded, [])  # noqa: SLF001
    server._active_surrogates.add("b")  # noqa: SLF001
    gm.add_edge("b", "d", "temperature", "heat_source")
    assert client.post("/surrogate/deactivate/b").status_code == 200
    _reloads_and_steps_as_the_graph(gm)
    assert ("b", "d") in [(e.source_node, e.target_node) for e in gm._edges]  # noqa: SLF001
