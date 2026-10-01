"""Every reply that carries state is valid JSON when a value is not finite.

The state replies handed Starlette bare Python floats, and its
``JSONResponse`` serialises with ``allow_nan=False``.  So a graph with a
``diagnostics=True`` coupling group -- whose spectral ``_meta`` slots are
seeded NaN at compile and at every reset, until a solve fills them -- made
``GET /graph/state`` a 500, and ``POST /sim/reset`` and ``POST
/checkpoint/load`` a 500 *after* their change had been applied; a diverged
simulation did the same to every state route.  A non-finite float is now
written as the quoted token every MADDENING JSON surface uses
(``"NaN"``, ``"Infinity"``, ``"-Infinity"``,
:mod:`maddening.serialization.json_codec`), which is how ``GET /graph``
already wrote one.  A finite reply is unchanged.
"""

from __future__ import annotations

import json
import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer, _jax_to_python
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.serialization import json_codec

REGISTRY = {"BallNode": BallNode, "SpringDamperNode": SpringDamperNode}
GROUP = "coupling_ball+spring"


def _strict(text: str):
    """``text`` parsed as RFC 8259 JSON: a bare ``NaN`` / ``Infinity`` --
    which Python's ``json`` reads and a browser's ``JSON.parse`` does not --
    is an error."""
    def refuse(token):
        raise AssertionError(f"bare {token} in a reply: {text[:300]}")
    return json.loads(text, parse_constant=refuse)


def _diagnostics_pair() -> GraphManager:
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0, gravity=-3.0))
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=20.0, rest_length=0.5))
    gm.add_edge("ball", "spring", "position", "anchor_position")
    gm.add_edge("spring", "ball", "position", "table_position")
    gm.add_coupling_group(["ball", "spring"], diagnostics=True, max_iterations=3)
    gm.compile()
    return gm


def _diverged() -> GraphManager:
    """A stiffness the explicit integrator cannot hold at this timestep."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=1.0e6, mass=0.5, rest_length=0.0,
                                 initial_position=1.0))
    gm.add_external_input("s", "anchor_position")
    gm.compile()
    gm.run(60)
    return gm


def _client(gm, tmp_path) -> TestClient:
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm,
                              checkpoint_root=str(tmp_path))
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _same_as_graph(reply: dict, state: dict) -> None:
    """The reply, decoded by the codec, is the graph's state value for value."""
    decoded = json_codec.decode_non_finite(reply)
    for node, fields in state.items():
        for field, value in fields.items():
            np.testing.assert_array_equal(np.asarray(decoded[node][field], np.float64),
                                          np.asarray(value, np.float64))


def test_the_diagnostics_seeds_are_nan_so_the_fixture_expresses_the_defect():
    gm = _diagnostics_pair()
    seeds = {k: float(v) for k, v in gm._state["_meta"].items()   # noqa: SLF001
             if k.startswith(GROUP)}
    assert any(np.isnan(v) for v in seeds.values()), seeds


def test_a_reset_of_a_diagnostics_group_answers_200_with_its_seeds_as_tokens(tmp_path):
    gm = _diagnostics_pair()
    gm.run(2)
    client = _client(gm, tmp_path)
    resp = client.post("/sim/reset")
    assert resp.status_code == 200, resp.text
    body = _strict(resp.text)
    meta = body["state"]["_meta"]
    assert json_codec.NAN_TOKEN in meta.values(), meta
    _same_as_graph(body["state"], gm._state)                       # noqa: SLF001
    for url in ("/graph/state", "/graph/state/ball", "/graph/state/spring"):
        resp = client.get(url)
        assert resp.status_code == 200, (url, resp.text)
        _strict(resp.text)


def test_a_checkpoint_taken_at_the_seeds_loads_through_the_route(tmp_path):
    """Saved straight after a reset -- NaN seeds in the checkpoint -- and
    loaded after some steps: the load is applied and answered, not a 500
    after the state moved."""
    gm = _diagnostics_pair()
    client = _client(gm, tmp_path)
    assert client.post("/sim/reset").status_code == 200
    assert client.post("/checkpoint/save", params={"path": "seeds.npz"}).status_code == 200
    assert client.post("/sim/run", params={"n_steps": 3}).status_code == 200
    resp = client.post("/checkpoint/load", params={"path": "seeds.npz"})
    assert resp.status_code == 200, resp.text
    body = _strict(resp.text)
    assert json_codec.NAN_TOKEN in body["state"]["_meta"].values()
    _same_as_graph(body["state"], gm._state)                       # noqa: SLF001


@pytest.mark.parametrize("method, url", [
    ("get", "/graph/state"), ("get", "/graph/state/s"), ("post", "/sim/step"),
    ("post", "/sim/run?n_steps=2"),
])
def test_every_state_route_answers_strict_json_once_the_simulation_diverged(
        method, url, tmp_path):
    gm = _diverged()
    resp = getattr(_client(gm, tmp_path), method)(url)
    assert resp.status_code == 200, resp.text
    body = _strict(resp.text)
    values = body["s"] if "s" in body else body
    assert set(values.values()) & json_codec.NON_FINITE_TOKENS, values
    _same_as_graph({"s": values}, {"s": gm.get_node_state("s")})


def test_a_finite_state_reply_is_written_as_numbers(tmp_path):
    """Nothing changes for a healthy graph: numbers, not strings."""
    gm = _diagnostics_pair()
    gm.run(2)
    reply = _client(gm, tmp_path).get("/graph/state").json()
    assert reply == _jax_to_python(gm._state)                       # noqa: SLF001
    assert not any(isinstance(v, str) for fields in reply.values() for v in fields.values())


def test_a_non_finite_parameter_reads_back_as_a_token(tmp_path):
    """``GET /graph/params`` reads the live pytree, which an in-process write
    can leave non-finite: it is served as a token, as ``GET /graph`` serves
    the same value, rather than failing in the encoder."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0))
    gm.add_external_input("s", "anchor_position")
    gm.compile()
    gm.params["nodes"]["s"]["damping"] = jnp.float32(np.nan)
    resp = _client(gm, tmp_path).get("/graph/params/s")
    assert resp.status_code == 200, resp.text
    assert _strict(resp.text)["damping"] == json_codec.NAN_TOKEN


def test_the_json_state_stream_is_strict_json_once_the_simulation_diverged(tmp_path):
    """``/ws/state`` sends the relay's snapshot of the same state: a bare
    ``NaN`` there made every frame unreadable to a browser's ``JSON.parse``."""
    gm = _diverged()
    client = _client(gm, tmp_path)
    with client.websocket_connect("/ws/state") as ws:
        deadline = time.monotonic() + 10.0
        text = None
        while text is None and time.monotonic() < deadline:
            assert client.post("/sim/step").status_code == 200
            text = ws.receive_text()
    assert text is not None
    frame = _strict(text)
    assert set(frame["state"]["s"].values()) & json_codec.NON_FINITE_TOKENS, frame
