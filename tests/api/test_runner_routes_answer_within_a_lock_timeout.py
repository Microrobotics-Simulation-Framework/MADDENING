"""The runner routes answer within about one graph-lock timeout, and say
what they did when they cannot finish.

* ``POST /sim/reset`` (and the surrogate activate / deactivate routes)
  stop the runner before waiting for the graph lock.  When that wait
  timed out the 503 said "Nothing was changed" -- and the runner it had
  stopped stayed stopped.  The refusal now says the runner stays stopped,
  with ``was_running``.
* The runner lock was taken with no timeout, and start, reset and the
  surrogate routes held it while they waited for the graph lock, so the
  k-th request behind one long holder of the graph answered after about k
  timeouts -- ``PUT /sim/stride``, which needs no graph lock, among them.
  Each request now answers within about one timeout of its arrival, and
  the stride at once.
"""

from __future__ import annotations

import os
import threading
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api import server as server_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
from tests.api.rest_claims_support import collector_off

TIMEOUT = 0.25


@pytest.fixture
def collector_is_off():
    """The garbage collector off for a test that times its requests; named
    before ``served`` in the test's arguments, so it is off from before the
    server is built (``collector_off``)."""
    with collector_off():
        yield


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(server_module, "_GRAPH_LOCK_TIMEOUT", TIMEOUT)
    gm = GraphManager()
    gm.add_node(BallNode("c", timestep=1.0 / 64.0, initial_velocity=1.0, gravity=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    yield server, client
    if server._graph_lock.held():
        server._graph_lock.release()
    with server._runner_lock:
        server._stop_runner()


def _wait_for(predicate, timeout=10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def test_a_reset_that_cannot_have_the_graph_says_it_stopped_the_runner(served):
    server, client = served
    assert client.post("/sim/start").status_code == 200
    assert _wait_for(lambda: server.relay.step_count > 3)
    assert server._graph_lock.acquire()          # a long holder of the graph
    resp = client.post("/sim/reset")
    server._graph_lock.release()
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["was_running"] is True
    assert "Nothing was changed" not in body["detail"]
    assert "Nothing was reset" in body["detail"]
    assert "stays stopped" in body["detail"]
    assert server.runner is None or not server.runner.is_alive


def test_a_reset_without_a_runner_says_only_that_nothing_was_reset(served):
    server, client = served
    assert server._graph_lock.acquire()
    resp = client.post("/sim/reset")
    server._graph_lock.release()
    assert resp.status_code == 503, resp.text
    assert resp.json()["was_running"] is False
    assert "stays stopped" not in resp.json()["detail"]


def test_an_activation_that_cannot_have_the_graph_says_it_stopped_the_runner(served):
    """The refusal comes before the job's result is used, so a finished
    job's record is enough here."""
    server, client = served
    server._surrogate_jobs["j"] = {"id": "j", "node_name": "c", "status": "done",
                                   "result": None}
    assert client.post("/sim/start").status_code == 200
    assert _wait_for(lambda: server.relay.step_count > 3)
    assert server._graph_lock.acquire()
    resp = client.post("/surrogate/activate/j")
    server._graph_lock.release()
    assert resp.status_code == 503, resp.text
    assert resp.json()["was_running"] is True
    assert "stays stopped" in resp.json()["detail"]


#: The queue test's lock timeout: long enough that a request's own cost
#: (a TestClient request on a two-core CI runner takes up to ~0.2 s) is
#: small beside it, so "about one timeout" and "about two" stay apart.
QUEUE_TIMEOUT = 1.0


def test_each_request_behind_a_long_holder_answers_within_about_one_timeout(
        collector_is_off, served, monkeypatch):
    monkeypatch.setattr(server_module, "_GRAPH_LOCK_TIMEOUT", QUEUE_TIMEOUT)
    server, client = served
    app = client.app
    log: dict = {}

    def call(label, method, path, **kw):
        sent = time.monotonic()
        resp = getattr(TestClient(app, raise_server_exceptions=False), method)(path, **kw)
        log[label] = (time.monotonic() - sent, resp.status_code)

    assert server._graph_lock.acquire()
    try:
        threads = []
        for label, method, path, kw in (
                ("start", "post", "/sim/start", {}),
                ("reset 1", "post", "/sim/reset", {}),
                ("reset 2", "post", "/sim/reset", {}),
                ("reset 3", "post", "/sim/reset", {}),
                ("stride", "put", "/sim/stride", {"params": {"steps_per_frame": 2}}),
                ("stop", "post", "/sim/stop", {})):
            t = threading.Thread(target=call, args=(label, method, path), kwargs=kw)
            t.start()
            threads.append(t)
            time.sleep(0.02)
        for t in threads:
            t.join(30)
    finally:
        server._graph_lock.release()
    assert set(log) == {"start", "reset 1", "reset 2", "reset 3", "stride", "stop"}, log
    # Within one timeout of arrival, and a margin for the request itself:
    # behind a start waiting for the graph, a request that then waited a
    # whole timeout of its own answered after nearly two.
    for label, (waited, status) in log.items():
        assert waited < 1.5 * QUEUE_TIMEOUT, (label, waited, status, log)
    assert log["stride"][1] == 200 and log["stride"][0] < QUEUE_TIMEOUT / 2, log
    assert server._steps_per_frame == 2


def test_a_runner_route_waits_for_the_runner_lock_only_until_its_deadline(served):
    """Another request holds the runner lock (it is starting, stopping or
    resetting the runner): ``POST /sim/start`` answers 503 within about one
    timeout of its arrival, while the lock is still held, instead of
    waiting for it."""
    server, client = served
    assert server._runner_lock.acquire()
    answered: dict = {}
    thread = threading.Thread(target=lambda: answered.update(resp=client.post("/sim/start")))
    try:
        thread.start()
        thread.join(TIMEOUT * 40)
        held_reply = dict(answered)
    finally:
        server._runner_lock.release()
        thread.join(60)
    assert "resp" in held_reply, "the route waited for the runner lock past its deadline"
    resp = held_reply["resp"]
    assert resp.status_code == 503, resp.text
    assert "starting, stopping or resetting the runner" in resp.json()["detail"]
    assert server.runner is None or not server.runner.is_alive
