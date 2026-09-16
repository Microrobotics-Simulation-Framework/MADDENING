"""Audit reproducers: REST params endpoint + sidecar edge cases."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode

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


def test_put_string_for_live_float_is_400_not_500(loaded):
    client, gm = loaded
    resp = client.put("/graph/params/ball", json={"params": {"elasticity": "fast"}})
    assert resp.status_code == 400, (resp.status_code, resp.text)


def test_put_list_for_scalar_live_float_is_400_not_500(loaded):
    client, gm = loaded
    resp = client.put("/graph/params/ball", json={"params": {"elasticity": [0.1, 0.2]}})
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert np.shape(gm.params["nodes"]["ball"]["elasticity"]) == ()


def test_put_nan_is_rejected(loaded):
    client, gm = loaded
    # Python's json module emits NaN by default; many JS clients do too.
    resp = client.put("/graph/params/ball", content='{"params": {"elasticity": NaN}}',
                      headers={"content-type": "application/json"})
    assert resp.status_code == 400, (resp.status_code, resp.text)
    assert not math.isnan(float(gm.params["nodes"]["ball"]["elasticity"]))


def test_put_live_param_survives_a_structural_put_on_another_node(loaded):
    """A PUT that dirties the graph (legacy node / structural key) recompiles
    on the next step and must not silently revert live params set earlier."""
    client, gm = loaded
    assert client.put("/graph/params/ball", json={"params": {"gravity": -1.0}}).status_code == 200
    # TableNode is on the legacy contract -> node.params + dirty
    r = client.put("/graph/params/table", json={"params": {"position": 0.5}})
    assert r.status_code == 200, r.text
    client.post("/sim/step")
    assert float(gm.params["nodes"]["ball"]["gravity"]) == -1.0


def test_put_live_param_then_reset_keeps_it(loaded):
    client, gm = loaded
    assert client.put("/graph/params/ball", json={"params": {"gravity": -1.0}}).status_code == 200
    r = client.post("/sim/reset")
    assert r.status_code in (200, 404), r.text
    assert float(gm.params["nodes"]["ball"]["gravity"]) == -1.0
