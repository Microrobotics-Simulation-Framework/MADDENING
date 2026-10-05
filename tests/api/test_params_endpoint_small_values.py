"""``PUT /graph/params/{node}`` at the small end of a float32.

A value the leaf's type flushes to zero (``1e-50``) is refused as one it
overflows is, in the same words, before the cast; a subnormal below a bound
of 0 (``-1e-40``) is refused by the bounds check, which compares it exactly
instead of through ``jnp`` (where XLA's CPU backend read it as 0).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode


def _client():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.compile()
    app = SimulationServer(node_registry={"SpringDamperNode": SpringDamperNode},
                           graph_manager=gm).create_app()
    return gm, TestClient(app, raise_server_exceptions=False)


def test_a_value_float32_flushes_to_zero_is_refused_before_the_cast():
    gm, client = _client()
    r = client.put("/graph/params/s", json={"params": {"rest_length": 1e-50}})
    assert r.status_code == 400 and "does not fit its type float32" in r.json()["detail"]
    assert float(gm.params["nodes"]["s"]["rest_length"]) == 1.0


def test_a_subnormal_below_a_zero_bound_is_refused():
    gm, client = _client()
    r = client.put("/graph/params/s", json={"params": {"damping": -1e-40}})
    assert r.status_code == 400 and "below bound 0.0" in r.json()["detail"]
    assert float(gm.params["nodes"]["s"]["damping"]) == 2.0
    assert client.put("/graph/params/s", json={"params": {"damping": 1e-40}}).status_code == 200
