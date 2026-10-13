"""The memory a request can make the server take is bounded before it is
taken: the request body, the graph as a whole, and how many node builds
run at once.

* **Bodies** were unbounded and parsed whole before any cap was checked --
  every element cap ran on the parsed request, and a 76 MB body of numbers
  peaked at 2-3 GB to be refused.  A pure-ASGI middleware now answers 413
  for a body over ``MAX_REQUEST_BODY_BYTES``, by its ``Content-Length``
  before reading it, or as it streams in.  ``PUT /graph/state`` counts a
  field's values against the live field before converting anything.
* **The graph**: each node was held to ``MAX_NODE_STATE_ELEMENTS``, and
  nothing bounded how many such nodes a caller added -- eight 5e6-cell rods
  were all accepted.  ``MAX_GRAPH_STATE_ELEMENTS`` bounds the sum, before
  the node is built where its class can say what it would build.
* **Concurrency**: six concurrent half-cap node builds peaked at 2.4 times
  one.  Builds run under the graph lock, one at a time.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api import server as server_module
from maddening.api.server import MAX_NODE_PARAM_ELEMENTS, SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, HeatNode


def _client(registry=None):
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer(registry or {"BallNode": BallNode, "HeatNode": HeatNode},
                              graph_manager=gm)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

_BODY_ROUTES = [
    ("POST", "/graph/nodes",
     lambda n: {"type": "BallNode", "name": "b2", "timestep": 0.01,
                "params": {"initial_position": [0.5] * n}}),
    ("PUT", "/graph/params/ball", lambda n: {"params": {"elasticity": [0.5] * n}}),
    ("PUT", "/graph/state/ball", lambda n: {"state": {"position": [0.5] * n,
                                                      "velocity": 0.0}}),
    ("POST", "/graph/edges", lambda n: {"source_node": "x" * 4 * n, "target_node": "ball",
                                        "source_field": "a", "target_field": "b"}),
]


@pytest.mark.parametrize("method, url, make", _BODY_ROUTES)
def test_a_body_over_the_limit_is_a_413_before_it_is_parsed(monkeypatch, method, url, make):
    monkeypatch.setattr(server_module, "MAX_REQUEST_BODY_BYTES", 4096)
    gm, server, client = _client()
    parsed = []
    monkeypatch.setattr(server_module, "_non_finite_param",
                        lambda *a, **k: parsed.append(a) or None)
    before = (list(gm._nodes), list(gm._edges), dict(gm.get_node_state("ball")))
    body = json.dumps(make(2000)).encode()
    assert len(body) > 4096
    resp = client.request(method, url, content=body,
                          headers={"content-type": "application/json"})
    assert resp.status_code == 413, resp.text
    assert "MAX_REQUEST_BODY_BYTES" in resp.json()["detail"]
    assert parsed == []
    assert (list(gm._nodes), list(gm._edges), dict(gm.get_node_state("ball"))) == before


def test_a_streamed_body_without_a_length_is_refused_as_it_passes_the_limit(monkeypatch):
    monkeypatch.setattr(server_module, "MAX_REQUEST_BODY_BYTES", 4096)
    gm, server, client = _client()
    pulled = []

    def chunks():
        yield b'{"type": "BallNode", "name": "b2", "timestep": 0.01, "params": {"x": ['
        for _ in range(1000):
            pulled.append(1)
            yield b"0.5, " * 20
        yield b"0.5]}}"

    resp = client.post("/graph/nodes", content=chunks(),
                       headers={"content-type": "application/json"})
    assert resp.status_code == 413, resp.text
    assert "b2" not in gm._nodes


def test_a_body_under_the_limit_is_served_as_before(monkeypatch):
    monkeypatch.setattr(server_module, "MAX_REQUEST_BODY_BYTES", 4096)
    gm, server, client = _client()
    resp = client.post("/graph/nodes", json={"type": "BallNode", "name": "b2",
                                             "timestep": 0.01, "params": {}})
    assert resp.status_code == 201, resp.text


def test_the_body_limit_admits_every_body_the_element_caps_admit():
    """The largest bounded body -- ``MAX_NODE_PARAM_ELEMENTS`` numbers at
    full float64 precision -- fits under the limit."""
    worst = json.dumps({"type": "BallNode", "name": "b", "timestep": 0.01,
                        "params": {"x": [-1.2345678901234567e-308] * MAX_NODE_PARAM_ELEMENTS}})
    assert len(worst.encode()) < server_module.MAX_REQUEST_BODY_BYTES


def test_a_state_field_with_the_wrong_number_of_values_is_refused_before_conversion():
    gm, server, client = _client()
    resp = client.put("/graph/state/ball",
                      json={"state": {"position": [0.5] * 100_000, "velocity": 0.0}})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "position: expected 1 value(s) (shape ()), got 100000"
    # The right count still converts, and shape and dtype are still checked.
    assert client.put("/graph/state/ball",
                      json={"state": {"position": 0.25, "velocity": 0.0}}).status_code == 200
    resp = client.put("/graph/state/ball", json={"state": {"position": [0.25],
                                                          "velocity": 0.0}})
    assert resp.status_code == 400 and "expected shape ()" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# The graph as a whole
# ---------------------------------------------------------------------------

def test_a_node_past_the_graph_wide_budget_is_refused_before_it_is_built(monkeypatch):
    monkeypatch.setattr(server_module, "MAX_GRAPH_STATE_ELEMENTS", 100)
    gm, server, client = _client()            # the ball holds 2
    built = []
    original = HeatNode.__init__

    @functools.wraps(original)
    def counting(self, *args, **kwargs):
        built.append(kwargs.get("n_cells"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(HeatNode, "__init__", counting)
    assert client.post("/graph/nodes", json={"type": "HeatNode", "name": "r1",
                                             "timestep": 1e-4,
                                             "params": {"n_cells": 60}}).status_code == 201
    resp = client.post("/graph/nodes", json={"type": "HeatNode", "name": "r2",
                                             "timestep": 1e-4, "params": {"n_cells": 60}})
    assert resp.status_code == 400, resp.text
    assert "at most 100 are accepted over the API in the whole graph" in resp.json()["detail"]
    assert built == [60], "the refused node was built"
    assert "r2" not in gm._nodes


def test_a_node_without_an_estimate_is_held_to_the_budget_on_its_built_state(monkeypatch):
    monkeypatch.setattr(server_module, "MAX_GRAPH_STATE_ELEMENTS", 3)
    gm, server, client = _client()            # the ball holds 2; another needs 2 more
    resp = client.post("/graph/nodes", json={"type": "BallNode", "name": "b2",
                                             "timestep": 0.01, "params": {}})
    assert resp.status_code == 400, resp.text
    assert "whole graph" in resp.json()["detail"]
    assert list(gm._nodes) == ["ball"]


def test_the_default_budget_admits_the_per_node_cap_more_than_once():
    assert server_module.MAX_GRAPH_STATE_ELEMENTS >= 2 * server_module.MAX_NODE_STATE_ELEMENTS


class _SlowBuild(BallNode):
    """A node whose constructor takes 0.1 s and records how many are being
    built at once."""

    lock = threading.Lock()
    building = 0
    most = 0

    def __init__(self, name, timestep, **params):
        with _SlowBuild.lock:
            _SlowBuild.building += 1
            _SlowBuild.most = max(_SlowBuild.most, _SlowBuild.building)
        try:
            time.sleep(0.1)
            super().__init__(name, timestep, **params)
        finally:
            with _SlowBuild.lock:
                _SlowBuild.building -= 1


def test_concurrent_node_builds_run_one_at_a_time():
    _SlowBuild.most = 0
    gm, server, client = _client({"SlowBuild": _SlowBuild})
    app = client.app
    codes = []

    def add(i):
        codes.append(TestClient(app, raise_server_exceptions=False).post(
            "/graph/nodes", json={"type": "SlowBuild", "name": f"n{i}",
                                  "timestep": 0.01, "params": {}}).status_code)

    threads = [threading.Thread(target=add, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert sorted(codes) == [201] * 6
    assert _SlowBuild.most == 1, f"{_SlowBuild.most} nodes were built at once"


def test_a_declared_length_over_the_limit_is_refused_without_reading_the_body(monkeypatch):
    """The middleware alone, as uvicorn calls it: with a ``Content-Length``
    over the limit it answers 413 without asking for one byte of the body,
    and never calls the application."""
    import asyncio

    from maddening.api.server import _RequestBodyLimitMiddleware

    monkeypatch.setattr(server_module, "MAX_REQUEST_BODY_BYTES", 1000)
    received, sent, called = [], [], []

    async def app(scope, receive, send):
        called.append(scope)

    async def receive():
        received.append(1)
        return {"type": "http.request", "body": b"x" * 2000, "more_body": False}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/graph/nodes",
             "headers": [(b"content-length", b"2000"), (b"content-type", b"application/json")]}
    asyncio.run(_RequestBodyLimitMiddleware(app)(scope, receive, send))
    assert received == [] and called == []
    assert sent[0]["type"] == "http.response.start" and sent[0]["status"] == 413
    assert b"MAX_REQUEST_BODY_BYTES" in sent[1]["body"]
