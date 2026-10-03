"""A node the graph cannot run is refused when it is added, not after.

* ``POST /graph/nodes`` never checked the timestep.  NaN or Infinity added
  the node and answered 500 (its reply could not be encoded); every step
  was then a 400 until the node was deleted.  0 or a negative value was a
  201, and the node stepped not at all, or backwards.  The request model
  bounds the timestep (a 422, encoded without a 500 even when the refused
  value is non-finite), and ``GraphManager.add_node`` refuses a timestep
  that is not a finite number > 0 for every caller.
* A node named ``_meta`` -- the key the graph keeps its own coupling and
  multirate state under -- was taken; the next compile dropped its state,
  every step was a ``KeyError`` and a checkpoint save was refused.  The
  checkpoint's ``_params`` and ``_params_mappings`` prefixes collide the
  same way.  ``add_node`` refuses all three, as it refuses the other
  reserved names.
* ``POST /sim/start`` on a graph with no nodes answered "started", and the
  runner's thread died on its first frame.  It is a 409 now.
"""

from __future__ import annotations

import json
import math
import warnings

import pytest

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
from tests._loopback_client import LoopbackTestClient as TestClient

REGISTRY = {"BallNode": BallNode}
RESERVED = ["_meta", "_params", "_params_mappings"]


def _served(*, nodes=True):
    gm = GraphManager()
    if nodes:
        gm.add_node(BallNode("ball", 0.01, initial_position=5.0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
    server = SimulationServer(REGISTRY, graph_manager=gm)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False)


def _post_node(client, name, timestep_literal):
    body = ('{"type": "BallNode", "name": "%s", "timestep": %s, "params": {}}'
            % (name, timestep_literal))
    return client.post("/graph/nodes", content=body,
                       headers={"content-type": "application/json"})


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "0", "0.0", "-0.0",
                                     "-0.01", "-1e308"])
def test_a_timestep_that_is_not_a_finite_positive_number_is_refused(literal):
    gm, _server, client = _served()
    resp = _post_node(client, "new", literal)
    assert resp.status_code == 422, resp.text
    body = json.loads(resp.text)            # a reply, never a 500
    assert body["detail"][0]["loc"][-1] == "timestep"
    assert "new" not in gm._nodes
    assert client.post("/sim/step").status_code == 200


def test_a_finite_positive_timestep_is_taken():
    gm, _server, client = _served()
    resp = _post_node(client, "new", "0.005")
    assert resp.status_code == 201, resp.text
    assert resp.json()["node"]["timestep"] == 0.005
    assert gm._nodes["new"].timestep == 0.005


@pytest.mark.parametrize("timestep", [math.nan, math.inf, -math.inf, 0.0, -0.0, -0.01])
def test_add_node_refuses_a_timestep_that_is_not_a_finite_positive_number(timestep):
    gm, _server, _client = _served()
    names = list(gm._nodes)
    with pytest.raises(ValueError, match="timestep must be a finite number > 0"):
        gm.add_node(BallNode("new", timestep))
    assert list(gm._nodes) == names and "new" not in gm._state


@pytest.mark.parametrize("name", RESERVED)
def test_a_reserved_state_key_is_refused_as_a_node_name(name):
    gm, _server, client = _served()
    resp = client.post("/graph/nodes", json={"type": "BallNode", "name": name,
                                             "timestep": 0.01, "params": {}})
    assert resp.status_code == 400, resp.text
    assert "is invalid" in resp.json()["detail"] and "reserves" in resp.json()["detail"]
    assert name not in gm._nodes
    with pytest.raises(ValueError, match="is invalid"):
        gm.add_node(BallNode(name, 0.01))
    assert client.post("/graph/compile").status_code == 200
    assert client.post("/sim/step").status_code == 200
    assert client.post("/checkpoint/save?path=after.npz").status_code == 200


@pytest.mark.parametrize("name", ["meta", "__meta", "_Meta", "_meta_", "params", "_params_",
                                  "_mappings"])
def test_a_lookalike_of_a_reserved_key_is_taken(name):
    gm, _server, client = _served()
    resp = client.post("/graph/nodes", json={"type": "BallNode", "name": name,
                                             "timestep": 0.01, "params": {}})
    assert resp.status_code == 201, resp.text
    assert client.post("/sim/step").status_code == 200
    assert client.post("/checkpoint/save?path=lookalike.npz").status_code == 200


def test_starting_the_runner_on_a_graph_with_no_nodes_is_refused():
    gm, server, client = _served(nodes=False)
    resp = client.post("/sim/start")
    assert resp.status_code == 409, resp.text
    assert "no nodes" in resp.json()["detail"]
    assert server.runner is None or not server.runner.is_alive
    assert client.post("/sim/pause").json()["detail"] == "Runner is not started."
    # The control: once the graph has a node, the runner starts.
    assert _post_node(client, "ball", "0.01").status_code == 201
    try:
        assert client.post("/sim/start").status_code == 200
    finally:
        assert client.post("/sim/stop").status_code == 200
