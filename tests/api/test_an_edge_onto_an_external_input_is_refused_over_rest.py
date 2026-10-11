"""Over REST, an edge onto a declared external input is the 400 of a graph
that cannot be compiled, carrying the graph's own message.

``POST /graph/edges`` adds an edge and compiles nothing, so it accepts one
whose target field is a declared external input (the graph is built in any
order, and nothing is wrong until it is compiled).  Every route that
compiles then answers 400 with the text ``GraphManager.compile()`` raises
(MADD-ANO-265), ``POST /graph/validate`` lists it, and removing the edge
(``DELETE /graph/edges``, the route for the ``remove_edge`` the message
spells) gives back a graph that steps.  Before the rule these routes
answered 200 and the node read zeros where the edge carried a value.

Requests go to the graph and simulation routes only.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

DT = 0.01
REGISTRY = {"TableNode": TableNode, "SpringDamperNode": SpringDamperNode}
EDGE = {"source_node": "table", "target_node": "spring",
        "source_field": "position", "target_field": "anchor_position"}

#: Every route this module sends a request to.
ROUTES = ("/graph/edges", "/graph/validate", "/graph/compile", "/sim/step", "/sim/run",
          "/graph/state")
assert not [r for r in ROUTES if r.startswith(("/cloud", "/surrogate", "/ws"))]


def _plant(*, edge: bool, external: bool = True) -> GraphManager:
    gm = GraphManager()
    gm.add_node(TableNode("table", DT, position=0.25))
    gm.add_node(SpringDamperNode("spring", DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    if edge:
        gm.add_edge(EDGE["source_node"], EDGE["target_node"],
                    EDGE["source_field"], EDGE["target_field"])
    if external:
        gm.add_external_input("spring", "anchor_position")
    return gm


def _client(gm, tmp_path) -> TestClient:
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm,
                              checkpoint_root=str(tmp_path))
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _refusal() -> str:
    with pytest.raises(ValueError) as caught:
        _plant(edge=True).compile()
    return str(caught.value)


def _post(client, route, **kwargs):
    assert route in ROUTES
    return client.post(route, **kwargs)


def test_an_edge_posted_onto_an_external_input_is_a_400_wherever_the_graph_is_compiled(tmp_path):
    text = _refusal()
    assert "edge table.position -> spring.anchor_position and external input " \
           "spring.anchor_position" in text
    gm = _plant(edge=False)
    gm.compile()
    client = _client(gm, tmp_path)
    assert _post(client, "/sim/step").status_code == 200
    held = client.get("/graph/state").json()

    # The edge is taken: nothing is compiled by the route that adds it.
    assert _post(client, "/graph/edges", json=EDGE).status_code == 201
    assert f"ERROR: {text}" in _post(client, "/graph/validate").json()["issues"]
    for route, kwargs in (("/graph/compile", {}), ("/sim/step", {}),
                          ("/sim/run", {"params": {"n_steps": 2}})):
        reply = _post(client, route, **kwargs)
        assert reply.status_code == 400, (route, reply.text)
        assert text in reply.json()["detail"], route
    # The refused requests stepped nothing.
    assert client.get("/graph/state").json() == held

    # DELETE /graph/edges is the remove_edge the message spells.
    assert "remove_edge('table', 'spring', 'position', 'anchor_position')" in text
    assert client.request("DELETE", "/graph/edges", json=EDGE).status_code == 200
    assert [i for i in _post(client, "/graph/validate").json()["issues"]
            if i.startswith("ERROR")] == []
    assert _post(client, "/graph/compile").status_code == 200
    assert _post(client, "/sim/step").status_code == 200
    assert _post(client, "/sim/run", params={"n_steps": 2}).status_code == 200


def test_a_server_given_a_graph_that_holds_both_answers_400_and_never_steps_it(tmp_path):
    text = _refusal()
    client = _client(_plant(edge=True), tmp_path)
    held = client.get("/graph/state").json()
    for route, kwargs in (("/graph/compile", {}), ("/sim/step", {}),
                          ("/sim/run", {"params": {"n_steps": 2}})):
        reply = _post(client, route, **kwargs)
        assert reply.status_code == 400, (route, reply.text)
        assert text in reply.json()["detail"], route
    assert client.get("/graph/state").json() == held


def test_the_same_edge_on_a_graph_without_the_external_input_steps(tmp_path):
    gm = _plant(edge=False, external=False)
    gm.compile()
    client = _client(gm, tmp_path)
    assert _post(client, "/graph/edges", json=EDGE).status_code == 201
    assert [i for i in _post(client, "/graph/validate").json()["issues"]
            if i.startswith("ERROR")] == []
    assert _post(client, "/graph/compile").status_code == 200
    assert _post(client, "/sim/step").status_code == 200
