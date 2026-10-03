"""A JAX trace started over REST stops itself at a step or time budget.

JAX's profiler holds a trace's events in memory until the trace stops,
and ``POST /sim/profile/jax/start`` set no bound: a trace left running
grew the server by about 4.2 KB per step (509 MB over 120 000 steps of a
one-node graph; a runner at 60 Hz, about 22 GB a day).  A trace now stops
itself after ``MAX_JAX_TRACE_STEPS`` steps or ``MAX_JAX_TRACE_SECONDS``
seconds -- writing its files as a requested stop does -- and
``GET /sim/profile/jax/status`` says how many steps it recorded and why it
stopped.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.testclient import TestClient

from maddening.api import server as server_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation import profiler
from maddening.nodes import BallNode


@pytest.fixture
def client():
    gm = GraphManager()
    gm.add_node(BallNode("c", timestep=1.0 / 64.0, initial_velocity=1.0, gravity=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    c = TestClient(server.create_app(), raise_server_exceptions=False)
    assert c.post("/sim/step").status_code == 200          # compiled before tracing
    yield c
    if profiler.jax_trace_active():
        profiler.stop_jax_trace()


def test_a_trace_stops_itself_at_its_step_budget(monkeypatch, client):
    monkeypatch.setattr(server_module, "MAX_JAX_TRACE_STEPS", 5)
    resp = client.post("/sim/profile/jax/start")
    assert resp.status_code == 200, resp.text
    assert resp.json()["max_steps"] == 5
    assert client.post("/sim/run", params={"n_steps": 12}).status_code == 200
    status = client.get("/sim/profile/jax/status").json()
    assert status["active"] is False
    assert status["steps"] == 5
    assert "step budget" in status["stopped_by"]
    assert status["last_trace_dir"] and Path(status["last_trace_dir"]).is_dir()
    assert any(Path(status["last_trace_dir"]).rglob("*")), "the trace wrote no files"
    stop = client.post("/sim/profile/jax/stop")
    assert stop.status_code == 409
    assert "stopped itself after 5 steps" in stop.json()["detail"]
    # The directory is the status route's to give (a 200); the 409's detail
    # names no server path and says where to look.
    assert status["last_trace_dir"] not in stop.json()["detail"]
    assert "last_trace_dir in GET /sim/profile/jax/status" in stop.json()["detail"]


def test_a_trace_stops_itself_at_its_time_budget(monkeypatch, client):
    """A step recorded past the time budget stops the trace -- with the budget's own
    timer held back, so that it is the step's check that does it (the timer is
    ``tests/api/test_routes_answer_what_they_did.py``'s)."""
    class Held:
        def __init__(self, *a, **k):
            self.daemon = True

        def start(self):
            pass

        def cancel(self):
            pass

    monkeypatch.setattr(server_module.threading, "Timer", Held)
    monkeypatch.setattr(server_module, "MAX_JAX_TRACE_SECONDS", 0.0)
    assert client.post("/sim/profile/jax/start").status_code == 200
    assert client.post("/sim/step").status_code == 200
    from maddening.core.simulation import profiler
    assert not profiler.jax_trace_active()
    status = client.get("/sim/profile/jax/status").json()
    assert status["active"] is False and "time budget" in status["stopped_by"]
    assert status["steps"] == 1


def test_a_trace_stopped_by_request_is_reported_as_before(client):
    assert client.post("/sim/profile/jax/start").status_code == 200
    assert client.post("/sim/run", params={"n_steps": 3}).status_code == 200
    assert client.get("/sim/profile/jax/status").json()["active"] is True
    stop = client.post("/sim/profile/jax/stop")
    assert stop.status_code == 200, stop.text
    assert stop.json()["steps"] == 3 and Path(stop.json()["log_dir"]).is_dir()
    status = client.get("/sim/profile/jax/status").json()
    assert status["active"] is False and status["stopped_by"] == "a request"
    # A new trace starts its count again.
    assert client.post("/sim/profile/jax/start").status_code == 200
    assert client.get("/sim/profile/jax/status").json()["steps"] == 0


def test_the_budgets_are_published_and_bounded():
    assert 0 < server_module.MAX_JAX_TRACE_STEPS <= 100_000
    assert 0 < server_module.MAX_JAX_TRACE_SECONDS <= 3600
