"""Regression tests for the independent audit of 2026-09-16, round 4
(core + REST side; report under benchmarks/results/audit4/).

* REST ``PUT /graph/state``, ``POST /graph/nodes``, ``POST /graph/edges``
  validate before writing; ``/checkpoint/{save,load}`` stay under a root;
  a JSON boolean for a numeric param is a 400;
* ``load_state`` refuses a state field of the wrong shape and coerces
  dtype; a checkpoint without ``_meta`` keeps the compiled ``_meta``;
* a wrong-shape params leaf is refused by ``step(params=)`` and by
  ``gm.params[...] =``;
* node names that would corrupt key namespaces are refused.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
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


# ------------------------------------------------------------------ REST

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


# ------------------------------------------------------------ checkpoints

def test_load_state_refuses_wrong_shape_state_and_coerces_dtype(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    gm = _spring()
    path = save_state(gm, tmp_path / "ck")
    data = dict(np.load(path))
    bad = dict(data)
    bad["s/position"] = np.ones(3, np.float32)
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="shape"):
        load_state(_spring(), tmp_path / "bad.npz")
    odd = dict(data)
    odd["s/position"] = np.array(7, np.int64)
    np.savez(tmp_path / "odd.npz", **odd)
    fresh = _spring()
    load_state(fresh, tmp_path / "odd.npz")
    assert fresh._state["s"]["position"].dtype == jnp.float32
    fresh.step()
    fresh.step()
    assert fresh.trace_count == 1


def test_checkpoint_without_meta_into_multirate_graph_keeps_compiled_meta(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    def build(slow_dt):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("s", DT, initial_position=1.0))
        gm.add_node(SpringDamperNode("b", slow_dt, initial_position=0.5))
        gm.add_edge("s", "b", "position", "anchor_position")
        gm.compile()
        return gm

    single = build(DT)
    single.run(2)
    path = save_state(single, tmp_path / "ck")
    assert not any(k.startswith("_meta/") for k in np.load(path).files)
    multi = build(2 * DT)
    load_state(multi, path)
    multi.run(3)                                    # used to raise KeyError: '_meta'
    assert int(multi._state["_meta"]["step_count"]) == 3


# ------------------------------------------------------------- params / names

def test_wrong_shape_params_leaf_is_refused_everywhere():
    gm = _spring()
    with pytest.raises(ValueError, match="shape"):
        gm.step(params={"nodes": {"s": {"stiffness": jnp.ones(3, jnp.float32) * 30}}})
    gm.params["nodes"]["s"]["stiffness"] = jnp.ones(3, jnp.float32) * 30
    with pytest.raises(ValueError, match="shape"):
        gm.step()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(30.0, jnp.float32)
    gm.step()
    assert gm._state["s"]["position"].shape == ()


@pytest.mark.parametrize("name", ["a/b", "a#1", "a->b", ""])
def test_node_names_that_break_key_namespaces_are_refused(name):
    gm = GraphManager()
    with pytest.raises(ValueError, match="invalid"):
        gm.add_node(SpringDamperNode(name, DT))
