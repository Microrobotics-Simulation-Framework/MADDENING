"""The graph-structure and state endpoints validate before they write.

* ``PUT /graph/state/{node}`` requires the exact field set, coerces to the
  live dtype, checks shape and finiteness, and writes nothing on a 400;
* ``PUT /graph/params/{node}`` refuses a JSON boolean for a numeric leaf;
* ``POST /graph/nodes`` traces one update before adding the node, so a
  bad constant cannot wedge every later ``/sim/step``;
* ``POST /graph/edges`` checks that the nodes and the source field exist;
* ``/checkpoint/{save,load}`` are confined to the server's checkpoint root.

Originally written from the independent audit of 2026-09-16 (round 4; report and
reproducers under ``benchmarks/results/audit4/``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

REGISTRY = {"SpringDamperNode": SpringDamperNode, "BallNode": BallNode, "TableNode": TableNode}
DT = 0.01


def _spring():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.compile()
    return gm


def _client(gm, **kw):
    return TestClient(SimulationServer(node_registry=REGISTRY, graph_manager=gm, **kw).create_app(),
                      raise_server_exceptions=False)


def test_put_state_validates_before_writing():
    gm = _spring()
    c = _client(gm)
    before = c.get("/graph/state/s").json()
    for body, needle in (
        ({"position": [1, 2, 3], "velocity": 0.0}, "shape"),
        ({"position": 1.0}, "fields"),
        ({"position": "x", "velocity": 0.0}, "position"),
        ({"position": 1.0, "velocity": 0.0, "extra": 1.0}, "fields"),
    ):
        r = c.put("/graph/state/s", json={"state": body})
        assert r.status_code == 400 and needle in r.json()["detail"], (body, r.json())
        assert c.get("/graph/state/s").json() == before
    # NaN must be sent as raw JSON (the test client's encoder refuses it)
    r = c.put("/graph/state/s", content=b'{"state": {"position": NaN, "velocity": 0.0}}',
              headers={"content-type": "application/json"})
    assert r.status_code == 400 and "finite" in r.json()["detail"]
    assert c.get("/graph/state/s").json() == before
    assert c.put("/graph/state/s", json={"state": {"position": 2.0, "velocity": 0.5}}).status_code == 200
    assert c.post("/sim/step").status_code == 200
    assert c.put("/graph/state/ghost", json={"state": {}}).status_code == 404


def test_put_params_boolean_for_numeric_leaf_is_400():
    gm = _spring()
    c = _client(gm)
    r = c.put("/graph/params/s", json={"params": {"stiffness": True}})
    assert r.status_code == 400 and "boolean" in r.json()["detail"]
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 30.0
    assert type(gm._nodes["s"].node.params["stiffness"]) is float
    assert c.post("/sim/step").status_code == 200
    assert "stiffness" in gm.params["nodes"]["s"]


def test_add_node_with_bad_params_is_400_and_not_added():
    gm = _spring()
    c = _client(gm)
    r = c.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "s2", "timestep": DT,
                                     "params": {"stiffness": "hot"}})
    assert r.status_code == 400 and "s2" in r.json()["detail"]
    assert "s2" not in gm._nodes
    assert c.post("/sim/step").status_code == 200
    assert c.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "s2", "timestep": DT,
                                        "params": {"stiffness": 12.0}}).status_code == 201
    assert c.post("/graph/nodes", json={"type": "SpringDamperNode", "name": "s2", "timestep": DT}
                  ).status_code == 409


def test_add_edge_checks_nodes_and_fields():
    gm = _spring()
    c = _client(gm)
    r = c.post("/graph/edges", json={"source_node": "ghost", "target_node": "s",
                                     "source_field": "position", "target_field": "anchor_position"})
    assert r.status_code == 404
    r = c.post("/graph/edges", json={"source_node": "s", "target_node": "s",
                                     "source_field": "nope", "target_field": "anchor_position"})
    assert r.status_code == 400 and "Available" in r.json()["detail"]
    assert gm.edges == []
    assert c.post("/sim/step").status_code == 200


def test_checkpoint_endpoints_stay_under_the_root(tmp_path):
    gm = _spring()
    c = _client(gm, checkpoint_root=tmp_path / "ck")
    r = c.post("/checkpoint/save?path=run1.npz")
    assert r.status_code == 200 and (tmp_path / "ck" / "run1.npz").exists()
    outside = tmp_path / "elsewhere.npz"
    for bad in (str(outside), "../elsewhere.npz", "/etc/passwd"):
        r = c.post(f"/checkpoint/save?path={bad}")
        assert r.status_code == 400 and "must stay under" in r.json()["detail"], bad
        assert not outside.exists()
        assert c.post(f"/checkpoint/load?path={bad}").status_code == 400
    assert c.post("/checkpoint/load?path=missing.npz").status_code == 404
    c.post("/sim/step")
    r = c.post("/checkpoint/load?path=run1.npz")
    assert r.status_code == 200 and r.json()["state"]["s"]["position"] == pytest.approx(1.0)
