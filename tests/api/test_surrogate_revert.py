"""``POST /surrogate/deactivate`` puts the graph back exactly as it was.

The revert used to re-add each saved edge with its endpoints and transform
only, dropping ``mapping``, ``additive`` and the units: with matching shapes
it answered 200 having quietly turned an additive coupling into an
overwriting one, and with mapped shapes the unguarded ``compile()`` raised,
leaving the live graph mutated, mapping-less and uncompilable for the life of
the process (whole-tree audit W4).

The invariants here are that the restored ``EdgeSpec`` is field-for-field the
saved one, and that a revert that cannot succeed changes nothing.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.coupling.mapping import rbf_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.heat import HeatNode

REGISTRY = {"HeatNode": HeatNode}


def _mapped_graph(n_target=5, source_temperature=0.0):
    """``A.temperature -> B.heat_source``: mapped, additive, transformed,
    unit-tagged -- every optional ``EdgeSpec`` field carries a non-default
    value, so a revert that drops any of them is visible.

    ``source_temperature`` is the source node's *initial* state, so a
    revert's ``reset_state()`` reproduces it and a value measured before
    the revert is comparable with one measured after.  A test asserting
    on delivered values passes something non-zero; the mapping of an
    all-zero field is all-zero whatever the weights are.
    """
    gm = GraphManager()
    source = HeatNode("A", 0.01, n_cells=5, initial_temperature=source_temperature)
    target = HeatNode("B", 0.01, n_cells=n_target)
    gm.add_node(source)
    gm.add_node(target)
    gm.add_edge(
        source="A", target="B",
        source_field="temperature", target_field="heat_source",
        mapping=rbf_mapping(
            np.linspace(0, 1, 5).reshape(-1, 1),
            np.linspace(0, 1, n_target).reshape(-1, 1),
            kernel="gaussian", epsilon=2.0,
        ),
        transform="negate",
        additive=True, source_units="K", target_units="K/s",
    )
    gm.compile()
    return gm, target


def _server_with_saved_original(gm, target):
    """A server that believes ``B`` was replaced by a surrogate."""
    server = SimulationServer(REGISTRY, gm)
    saved_edges = [e for e in gm._edges
                   if "B" in (e.source_node, e.target_node)]
    server._original_nodes["B"] = (target, saved_edges, [])
    server._active_surrogates.add("B")
    return server, saved_edges


def _fields(edge):
    return (edge.source_node, edge.target_node, edge.source_field,
            edge.target_field, edge.transform, edge.additive,
            edge.source_units, edge.target_units, edge.mapping, edge.ordinal)


def test_surrogate_deactivate_restores_every_edge_attribute():
    gm, target = _mapped_graph()
    server, saved_edges = _server_with_saved_original(gm, target)
    client = TestClient(server.create_app(), raise_server_exceptions=False)

    response = client.post("/surrogate/deactivate/B")
    assert response.status_code == 200, response.text

    restored = [e for e in gm._edges if "B" in (e.source_node, e.target_node)]
    assert len(restored) == len(saved_edges)
    for before, after in zip(saved_edges, restored):
        assert _fields(after) == _fields(before)
    # The mapping's trainable weights are back in the live pytree too.
    assert list(gm.params["mappings"]) == [saved_edges[0].key]
    assert not gm._dirty


def test_a_failing_surrogate_deactivate_leaves_the_graph_compilable():
    """A revert that cannot be completed is a 500 that changed nothing.

    The saved edge names a source node that no longer exists, so the
    restore cannot succeed.  The graph must still be the one the server
    had -- compiled, steppable, and with the surrogate still registered so
    the caller can retry.
    """
    gm, target = _mapped_graph()
    server, saved_edges = _server_with_saved_original(gm, target)
    client = TestClient(server.create_app(), raise_server_exceptions=False)

    # The edge's source disappears between activation and revert.
    gm.remove_node("A")
    gm.compile()
    before_edges = list(gm._edges)
    before_nodes = sorted(gm._nodes)

    response = client.post("/surrogate/deactivate/B")
    assert response.status_code == 500
    assert "B" in response.json()["detail"]

    assert list(gm._edges) == before_edges
    assert sorted(gm._nodes) == before_nodes
    assert not gm._dirty
    gm.step()  # the live graph still runs
    # The revert can be retried once the graph is whole again.
    assert "B" in server._original_nodes
    assert "B" in server._active_surrogates


@pytest.mark.parametrize("n_target", [5, 4])
def test_surrogate_deactivate_keeps_the_graph_runnable_for_any_mapped_shape(n_target):
    """Shapes that only agree through the mapping revert cleanly too.

    With ``n_target=4`` the mapping is the only thing making the shapes
    match, so dropping it turned the revert into a 500 that broke the
    graph; with ``n_target=5`` it answered 200 and changed the physics.
    """
    gm, target = _mapped_graph(n_target)
    server, _ = _server_with_saved_original(gm, target)
    client = TestClient(server.create_app(), raise_server_exceptions=False)

    assert client.post("/surrogate/deactivate/B").status_code == 200
    edge = next(e for e in gm._edges if e.target_node == "B")
    assert edge.mapping is not None and edge.additive
    gm.step()


def test_surrogate_deactivate_keeps_a_fit_mapping_weight():
    """A fit made while the surrogate was active must survive the revert.

    ``remove_node`` drops ``gm.params["mappings"][key]`` and the revert's
    ``compile()`` re-snapshots it from the mapping object, so a weight
    ``sysid`` had moved silently went back to what the ``MappingSpec``
    rebuilds -- the same silent reversion ``replace_node`` had on the way
    in.  Asserted on the value the edge delivers, not on the pytree: the
    weights being present says nothing about which weights they are.
    """
    gm, target = _mapped_graph(source_temperature=np.arange(1.0, 6.0))
    server, saved_edges = _server_with_saved_original(gm, target)
    key = saved_edges[0].key

    # Stand in for a sysid fit: the live weights move away from the ones
    # the mapping object (and therefore the recipe) would rebuild.
    slot = gm.params["mappings"][key]
    slot["H"] = (slot["H"] * 3.0).astype(slot["H"].dtype)
    before = np.asarray(gm.resolve_boundary_inputs("B")["heat_source"])
    # The fit has to be visible at all, or the test proves nothing.
    recipe = -np.asarray(
        saved_edges[0].mapping.apply(gm._state["A"]["temperature"])
    )
    assert not np.allclose(before, recipe, rtol=1e-3)

    client = TestClient(server.create_app(), raise_server_exceptions=False)
    assert client.post("/surrogate/deactivate/B").status_code == 200

    after = np.asarray(gm.resolve_boundary_inputs("B")["heat_source"])
    np.testing.assert_allclose(after, before, rtol=1e-6)


def test_surrogate_deactivate_keeps_trainable_mapping_weights_trainable():
    """``set_param_spec`` on a mapped edge key is dropped by
    ``remove_node`` too; an optimiser would silently stop moving the
    weights it was opted in to move."""
    gm, target = _mapped_graph()
    server, saved_edges = _server_with_saved_original(gm, target)
    key = saved_edges[0].key
    gm.set_param_spec(key, "H", ParamSpec(trainable=True))
    assert gm.trainable_mask()["mappings"][key]["H"] is True

    client = TestClient(server.create_app(), raise_server_exceptions=False)
    assert client.post("/surrogate/deactivate/B").status_code == 200

    assert gm.trainable_mask()["mappings"][key]["H"] is True
