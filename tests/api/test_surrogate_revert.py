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
from maddening.nodes.heat import HeatNode

REGISTRY = {"HeatNode": HeatNode}


def _mapped_graph(n_target=5):
    """``A.temperature -> B.heat_source``: mapped, additive, transformed,
    unit-tagged -- every optional ``EdgeSpec`` field carries a non-default
    value, so a revert that drops any of them is visible."""
    gm = GraphManager()
    source = HeatNode("A", 0.01, n_cells=5)
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
