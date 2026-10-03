"""The runner routes answer while every worker thread waits for the graph.

The REST guide: the runner routes "answer within about one
``_GRAPH_LOCK_TIMEOUT`` of their arrival", and ``POST /sim/stop`` "is
answered at once even while a long request holds the graph".  Both held
only while a worker thread was free: the routes were synchronous, so they
ran on the worker pool every other route shares, and behind a queue of
requests waiting for the graph a stop waited for a free worker before its
own deadline started (5.6 s behind 240 queued reads at a 1 s timeout, on
a loopback uvicorn).  They are ``async`` now, with their blocking work on
a small pool of their own (``_RUNNER_ROUTE_WORKERS``) and their deadline
counted from their arrival on the event loop.

Here one client (one event loop, one worker pool of 40 threads) sends more
reads than there are workers while the test thread holds the graph, so
every worker is waiting for it; then each runner route arrives.
"""

from __future__ import annotations

import threading
import time
import warnings

import pytest
from fastapi.testclient import TestClient

from maddening.api import server as server_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode

TIMEOUT = 1.5
N_READS = 48            # more than anyio's 40 worker threads


@pytest.fixture
def saturated(monkeypatch):
    """A client whose every worker thread is waiting for the graph lock,
    which the test thread holds for the length of the test."""
    monkeypatch.setattr(server_module, "_GRAPH_LOCK_TIMEOUT", TIMEOUT)
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    with TestClient(server.create_app(), raise_server_exceptions=False) as client:
        server._graph_lock.acquire()
        statuses: list = []
        readers = [threading.Thread(target=lambda: statuses.append(
            client.get("/graph/state").status_code), daemon=True) for _ in range(N_READS)]
        try:
            for r in readers:
                r.start()
            end = time.monotonic() + 10
            while len(server._graph_lock._queue) < 40 and time.monotonic() < end:
                time.sleep(0.005)
            assert len(server._graph_lock._queue) >= 40, len(server._graph_lock._queue)
            yield server, client, statuses
        finally:
            server._graph_lock.release()
            for r in readers:
                r.join(30)


@pytest.mark.parametrize("method, path, want", [
    ("POST", "/sim/stop", 409),         # no runner: "Runner is not started."
    ("POST", "/sim/pause", 409),
    ("POST", "/sim/resume", 409),
    ("PUT", "/sim/stride?steps_per_frame=3", 200),
])
def test_a_runner_route_that_needs_no_graph_answers_at_once(saturated, method, path, want):
    _server, client, statuses = saturated
    t0 = time.monotonic()
    resp = client.request(method, path)
    elapsed = time.monotonic() - t0
    assert resp.status_code == want, resp.text
    assert elapsed < TIMEOUT / 2, elapsed
    assert len(statuses) < N_READS, "the reads had finished: the test proves nothing"


@pytest.mark.parametrize("path", ["/sim/start", "/sim/reset"])
def test_a_runner_route_that_waits_for_the_graph_answers_within_one_timeout(saturated, path):
    """A start or a reset needs the graph the test thread holds: a 503 at
    its own deadline, counted from its arrival -- not after a worker freed
    up first (about one more timeout)."""
    server, client, _statuses = saturated
    t0 = time.monotonic()
    resp = client.post(path)
    elapsed = time.monotonic() - t0
    assert resp.status_code == 503, resp.text
    assert elapsed < 1.6 * TIMEOUT, elapsed
    assert server.runner is None
