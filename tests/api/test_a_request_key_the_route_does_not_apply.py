"""A request key a route does not apply: refused where dropping it builds
another graph than the body asked for.

``POST /graph/edges`` takes four keys.  ``GraphManager.add_edge`` takes more
(``transform``, ``additive``, units, a ``mapping``, a ``geometry``), and the
route passes none of them: a body carrying one was answered 201 with a plain
edge added, and the graph stepped differently from the one the body
described (MADD-ANO-253).  ``POST /graph/nodes`` has the same shape: its
``params`` is optional, so a misspelt ``params`` (or the constructor's
arguments written beside ``type``) was a 201 on a node built from its
defaults.  Both are now a 422 that names the key, and add nothing.

The other request models of the server drop a key they do not know, and
dropping it does not change what the known keys do; the last test here
holds them to exactly that, so that forbidding their extras is a decision
and not a side effect.  No request in this file goes to a path outside
``/graph`` and ``/sim/step``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api import server as server_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

REGISTRY = {"SpringDamperNode": SpringDamperNode}
DT = 0.01
EDGE = {"source_node": "a", "target_node": "b",
        "source_field": "position", "target_field": "anchor_position"}
NODE = {"type": "SpringDamperNode", "name": "c", "timestep": DT}

#: Every argument of ``GraphManager.add_edge`` the route does not pass, with
#: a value a client could send for it, and two keys that are nobody's.
UNAPPLIED_EDGE_KEYS = {
    "mapping": {"kind": "nearest_neighbor", "hyperparameters": {"mode": "consistent"}},
    "geometry": ["target", "position"],
    "transform": "lambda x: 2 * x",
    "additive": True,
    "source_units": "m",
    "target_units": "mm",
    "mappng": 1,
    "sourceField": "position",
}
#: Keys ``POST /graph/nodes`` does not apply: ``params`` misspelt, and a
#: constructor argument written beside ``type`` instead of inside ``params``.
UNAPPLIED_NODE_KEYS = {
    "parameters": {"stiffness": 500.0},
    "param": {"stiffness": 500.0},
    "stiffness": 500.0,
    "initial_state": {"position": 3.0},
    "dt": 0.5,
}


def _graph():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", DT, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(SpringDamperNode("b", DT, stiffness=10.0, damping=1.0, initial_position=0.5))
    gm.compile()
    return gm


def _client(gm):
    app = SimulationServer(node_registry=REGISTRY, graph_manager=gm).create_app()
    client = TestClient(app, raise_server_exceptions=False)

    def call(method, path, **kw):
        assert path.startswith(("/graph", "/sim/step")), path
        if method == "delete":
            # A DELETE with a body: the client's ``delete`` takes none.
            return client.request("DELETE", path, **kw)
        return getattr(client, method)(path, **kw)

    return call


def _refusal_names(reply, key):
    assert reply.status_code == 422, (key, reply.status_code, reply.json())
    named = [e for e in reply.json()["detail"] if list(e.get("loc", ()))[-1:] == [key]]
    assert named and named[0]["type"] == "extra_forbidden", (key, reply.json())


def _stepped(gm, call):
    assert call("post", "/sim/step").status_code == 200
    return {n: {f: np.asarray(v) for f, v in gm.get_node_state(n).items()} for n in ("a", "b")}


@pytest.mark.parametrize("key", sorted(UNAPPLIED_EDGE_KEYS))
def test_an_edge_request_with_a_key_the_route_does_not_apply_is_refused_and_adds_nothing(key):
    gm = _graph()
    call = _client(gm)
    reply = call("post", "/graph/edges", json={**EDGE, key: UNAPPLIED_EDGE_KEYS[key]})
    _refusal_names(reply, key)
    assert gm.edges == [] and call("get", "/graph").json()["edges"] == []
    # The graph steps as the one that was never given the edge.
    bare = _graph()
    after, want = _stepped(gm, call), _stepped(bare, _client(bare))
    for node in want:
        for field in want[node]:
            np.testing.assert_array_equal(after[node][field], want[node][field])


def test_an_edge_request_of_the_four_keys_is_taken_and_changes_the_step():
    gm = _graph()
    call = _client(gm)
    assert call("post", "/graph/edges", json=EDGE).status_code == 201
    assert len(gm.edges) == 1 and gm.edges[0].additive is False and gm.edges[0].transform is None
    bare = _graph()
    after, want = _stepped(gm, call), _stepped(bare, _client(bare))
    # The fixture can tell an edge from none: the refusals above are not
    # vacuous.
    assert not np.array_equal(after["b"]["position"], want["b"]["position"])


def test_every_key_the_library_s_door_takes_is_applied_or_refused_by_the_route():
    """The table above is ``add_edge``'s signature: an argument added to it
    is refused by the route until the route passes it."""
    import inspect

    taken = set(inspect.signature(GraphManager.add_edge).parameters) - {"self"}
    applied = {"source", "target", "source_field", "target_field"}
    assert applied <= taken
    assert taken - applied <= set(UNAPPLIED_EDGE_KEYS), sorted(taken - applied)
    assert set(server_module.AddEdgeRequest.model_fields) == {
        "source_node", "target_node", "source_field", "target_field"}


@pytest.mark.parametrize("key", sorted(UNAPPLIED_NODE_KEYS))
def test_a_node_request_with_a_key_the_route_does_not_apply_is_refused_and_adds_nothing(key):
    gm = _graph()
    call = _client(gm)
    reply = call("post", "/graph/nodes", json={**NODE, key: UNAPPLIED_NODE_KEYS[key]})
    _refusal_names(reply, key)
    assert "c" not in gm._nodes                                        # noqa: SLF001
    assert sorted(n["name"] for n in call("get", "/graph").json()["nodes"]) == ["a", "b"]
    # The same node with its parameters where the route reads them.
    taken = call("post", "/graph/nodes", json={**NODE, "params": {"stiffness": 500.0}})
    assert taken.status_code == 201
    assert float(call("get", "/graph/params/c").json()["stiffness"]) == 500.0


def test_the_other_request_models_drop_a_key_they_do_not_know_and_apply_the_rest():
    """The state of the sweep, not a promise: these models have no optional
    key a dropped one could have been, and what their known keys do is
    unchanged by the extra.  Forbidding extras there is a change of a
    released route's behaviour for the maintainer to decide."""
    gm = _graph()
    call = _client(gm)
    state = {f: np.asarray(v).tolist() for f, v in gm.get_node_state("a").items()}
    state["position"] = 2.5
    r = call("put", "/graph/state/a", json={"state": state, "unknown_key": 1})
    assert r.status_code == 200 and float(gm.get_node_state("a")["position"]) == 2.5
    r = call("put", "/graph/params/a", json={"params": {"stiffness": 44.0}, "unknown_key": 1})
    assert r.status_code == 200
    assert float(call("get", "/graph/params/a").json()["stiffness"]) == 44.0
    assert call("post", "/graph/edges", json=EDGE).status_code == 201
    r = call("delete", "/graph/edges", json={**EDGE, "unknown_key": 1})
    assert r.status_code == 200 and gm.edges == []
    # The surrogate route's model, at the model (no request to its route).
    model = server_module.TrainSurrogateRequest.model_validate(
        {"node_name": "a", "n_epoch": 3})
    assert model.n_epochs == 100 and not hasattr(model, "n_epoch")
    forbidding = {name for name in ("AddNodeRequest", "AddEdgeRequest", "RemoveEdgeRequest",
                                    "SetNodeStateRequest", "SetNodeParamsRequest",
                                    "TrainSurrogateRequest")
                  if getattr(server_module, name).model_config.get("extra") == "forbid"}
    assert forbidding == {"AddNodeRequest", "AddEdgeRequest"}
