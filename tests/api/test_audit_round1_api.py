"""Audit round 1 (feat/graph-params-sysid): ``PUT /graph/params/{node}``.

Finding #7: a string for a live float produced a 500, and NaN passed the
bounds check (every comparison with NaN is False) and was WRITTEN into
``gm.params`` and ``node.params`` before the response failed.  The
endpoint now validates dtype coercion, shape, finiteness and bounds for
every key before mutating anything; any failure is a 400 naming the key
and nothing is written.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

REGISTRY = {"BallNode": BallNode, "TableNode": TableNode, "SpringDamperNode": SpringDamperNode}


@pytest.fixture
def loaded():
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
    gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0, elasticity=0.7))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm)
    return TestClient(server.create_app(), raise_server_exceptions=False), gm


def _snapshot(gm):
    node = gm._nodes["ball"].node
    return ({k: np.asarray(v).copy() for k, v in gm.params["nodes"]["ball"].items()},
            dict(node.params))


def _assert_untouched(gm, before):
    live, ctor = before
    for k, v in live.items():
        np.testing.assert_array_equal(np.asarray(gm.params["nodes"]["ball"][k]), v)
    assert gm._nodes["ball"].node.params == ctor
    assert not gm._dirty


def test_put_string_for_live_float_is_400_and_writes_nothing(loaded):
    client, gm = loaded
    before = _snapshot(gm)
    resp = client.put("/graph/params/ball", json={"params": {"elasticity": "fast"}})
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert "elasticity" in resp.json()["detail"]
    _assert_untouched(gm, before)


def test_put_null_for_live_float_is_400(loaded):
    client, gm = loaded
    before = _snapshot(gm)
    resp = client.put("/graph/params/ball", json={"params": {"elasticity": None}})
    assert resp.status_code == 400, (resp.status_code, resp.text)
    _assert_untouched(gm, before)


def test_put_list_for_scalar_live_float_is_400(loaded):
    client, gm = loaded
    before = _snapshot(gm)
    resp = client.put("/graph/params/ball", json={"params": {"elasticity": [0.1, 0.2]}})
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert "shape" in resp.json()["detail"]
    _assert_untouched(gm, before)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
def test_put_non_finite_is_rejected_and_not_written(loaded, literal):
    """Python's json module (and many JS clients) emit these literals."""
    client, gm = loaded
    before = _snapshot(gm)
    resp = client.put("/graph/params/ball",
                      content='{"params": {"gravity": %s}}' % literal,
                      headers={"content-type": "application/json"})
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert "gravity" in resp.json()["detail"] and "finite" in resp.json()["detail"]
    assert math.isfinite(float(gm.params["nodes"]["ball"]["gravity"]))
    assert math.isfinite(float(gm._nodes["ball"].node.params["gravity"]))
    _assert_untouched(gm, before)


def test_put_rejects_whole_request_when_a_later_key_is_invalid(loaded):
    """Validation is staged: a valid first key is not written when the
    second one fails."""
    client, gm = loaded
    before = _snapshot(gm)
    resp = client.put("/graph/params/ball",
                      json={"params": {"gravity": -1.0, "elasticity": "fast"}})
    assert resp.status_code == 400, (resp.status_code, resp.text)
    _assert_untouched(gm, before)
    assert float(gm.params["nodes"]["ball"]["gravity"]) != -1.0
    # and the same request without the bad key goes through
    ok = client.put("/graph/params/ball", json={"params": {"gravity": -1.0}})
    assert ok.status_code == 200, ok.text
    assert float(gm.params["nodes"]["ball"]["gravity"]) == -1.0
