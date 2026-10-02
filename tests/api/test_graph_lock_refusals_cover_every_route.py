"""The graph lock's 409 and 503 hold for every route the REST guide names.

The REST guide ("Concurrency, limits and shutdown"): "While the runner
runs, or a ``/sim/run`` is in progress, the routes that write the state or
the structure -- ``/sim/step``, ``/sim/run``, ``PUT /graph/state``,
``/checkpoint/load``, adding or removing a node or an edge,
``/sim/profile`` -- answer 409.  ``PUT /graph/params`` still reaches a
running graph", and "a request waits at most ``_GRAPH_LOCK_TIMEOUT``
(30 s) for the lock and then answers 503".
``tests/api/test_concurrent_requests_are_serialised.py`` and
``tests/api/test_runner_pacing_and_status.py`` check some of those routes
against one of the two steppers; these check every listed route against
both (the reset and the runner start, which the route docstrings add, as
well), with nothing written by any refused request.

The lock-timeout rows are checked with ``_GRAPH_LOCK_TIMEOUT`` patched to
a fraction of a second.  Three rows the tree does not meet are strict
xfails (``REST-044``, ``REST-045`` and ``REST-046`` in
``docs/validation/rest_runpod_claims.yaml``): the
runner routes wait for the runner's own lock before the graph's, so a
request queued behind a ``POST /sim/start`` that is waiting for the graph
answers after more than one timeout, a ``POST /sim/stop`` behind it is not
answered at once, and a reset that stopped the runner and then could not
have the graph says "Nothing was changed".

Nothing here can reach a cloud provider
(:func:`tests.property.differential.no_cloud_launch`).
"""

from __future__ import annotations

import os
import threading
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.testclient import TestClient

from maddening.api import server as server_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
from tests.property.differential import no_cloud_launch

EDGE = {"source_node": "ball", "target_node": "ball", "source_field": "position",
        "target_field": "table_position"}

#: The writes the guide refuses while something steps the graph, as
#: ``(label, method, url, request kwargs)``.  Each request is valid, so a
#: 409 cannot be a validation failure in disguise.
WRITES = [
    ("step", "POST", "/sim/step", {}),
    ("run", "POST", "/sim/run", {"params": {"n_steps": 1}}),
    ("state", "PUT", "/graph/state/ball", {"json": {"state": {"position": 5.0,
                                                              "velocity": 0.0}}}),
    ("load", "POST", "/checkpoint/load", {"params": {"path": "c.npz"}}),
    ("add node", "POST", "/graph/nodes", {"json": {"type": "BallNode", "name": "b2",
                                                   "timestep": 0.01, "params": {}}}),
    ("remove node", "DELETE", "/graph/nodes/ball", {}),
    ("add edge", "POST", "/graph/edges", {"json": EDGE}),
    ("remove edge", "DELETE", "/graph/edges", {"json": EDGE}),
    ("profile", "POST", "/sim/profile", {"params": {"n_steps": 1}}),
]


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with no_cloud_launch():
        yield


def _served(tmp_path, dt: float = 0.01) -> tuple[SimulationServer, TestClient]:
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=dt, initial_position=100.0, gravity=-1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({"BallNode": BallNode}, graph_manager=gm,
                              checkpoint_root=str(tmp_path))
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    assert client.post("/checkpoint/save", params={"path": "c.npz"}).status_code == 200
    return server, client


def _snapshot(server: SimulationServer) -> tuple:
    gm = server.gm
    return (list(gm._nodes), list(gm._edges),
            {f: float(v) for f, v in gm.get_node_state("ball").items()})


def _wait_for(predicate, timeout: float = 20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _stop(server: SimulationServer) -> None:
    if server._graph_lock.held():
        server._graph_lock.release()
    with server._runner_lock:
        server._stop_runner()


# ---------------------------------------------------------------------------
# 409: every write, beside both steppers
# ---------------------------------------------------------------------------

def test_every_write_is_a_409_while_a_run_is_in_progress(tmp_path):
    """The run's first slice is held open, so every request below arrives
    while the run is in progress and holds the graph; the refusal is asked
    before the lock, so none of them waits."""
    server, client = _served(tmp_path)
    gm = server.gm
    entered, release = threading.Event(), threading.Event()
    real_run = gm.run

    def held_open(n, *args, **kwargs):
        entered.set()
        release.wait(30)
        return real_run(n, *args, **kwargs)

    gm.run = held_open
    reply: dict = {}
    run = threading.Thread(target=lambda: reply.setdefault(
        "run", TestClient(client.app).post("/sim/run", params={"n_steps": 5})))
    run.start()
    try:
        assert entered.wait(20)
        before = _snapshot(server)
        for label, method, url, kwargs in WRITES + [
                ("reset", "POST", "/sim/reset", {}), ("start", "POST", "/sim/start", {})]:
            t0 = time.monotonic()
            resp = client.request(method, url, **kwargs)
            assert resp.status_code == 409, (label, resp.status_code, resp.text)
            assert "POST /sim/run is in progress" in resp.json()["detail"], (label, resp.text)
            assert time.monotonic() - t0 < 5.0, label
        assert _snapshot(server) == before
        assert server.runner is None
    finally:
        release.set()
        run.join(60)
    assert reply["run"].status_code == 200, reply["run"].text
    assert server.relay.step_count == 5


def test_every_write_is_a_409_while_the_runner_runs(tmp_path):
    server, client = _served(tmp_path, dt=0.001)
    try:
        assert client.post("/sim/start").status_code == 200
        assert _wait_for(lambda: server.relay.step_count > 3)
        for label, method, url, kwargs in WRITES:
            resp = client.request(method, url, **kwargs)
            assert resp.status_code == 409, (label, resp.status_code, resp.text)
            assert "runner is started" in resp.json()["detail"], (label, resp.text)
        assert client.post("/sim/start").status_code == 409
        assert list(server.gm._nodes) == ["ball"] and server.gm._edges == []
        # ... and the parameter write the guide exempts reaches it.
        resp = client.put("/graph/params/ball", json={"params": {"gravity": -2.0}})
        assert resp.status_code == 200, resp.text
        assert server.runner.is_alive
    finally:
        _stop(server)


def test_a_params_write_reaches_the_graph_between_the_slices_of_a_run(tmp_path):
    """Until the write is answered each slice of the run takes 30 ms, more
    than half of ``_RUN_SLICE_SECONDS``, so the run stays at one step a
    slice and cannot finish first (1000 slices); the write is taken between
    two of them while the run goes on.  After it the slices are fast."""
    server, client = _served(tmp_path)
    gm = server.gm
    real_run = gm.run
    answered = threading.Event()

    def slow_until_answered(n, *args, **kwargs):
        if not answered.is_set():
            time.sleep(0.03)
        return real_run(n, *args, **kwargs)

    gm.run = slow_until_answered
    reply: dict = {}
    run = threading.Thread(target=lambda: reply.setdefault(
        "run", TestClient(client.app).post("/sim/run", params={"n_steps": 1000})))
    run.start()
    try:
        assert _wait_for(lambda: server.relay.step_count >= 2)
        resp = client.put("/graph/params/ball", json={"params": {"gravity": -3.0}})
        still_running = run.is_alive()
        answered.set()
        assert resp.status_code == 200, resp.text
        assert still_running, "the run ended before the write; the test proves nothing"
    finally:
        answered.set()
        run.join(60)
    assert reply["run"].status_code == 200
    assert server.relay.step_count == 1000
    assert client.get("/graph/params/ball").json()["gravity"] == -3.0


# ---------------------------------------------------------------------------
# 503: the lock timeout
# ---------------------------------------------------------------------------

def test_a_request_that_cannot_have_the_graph_in_time_is_a_503_that_writes_nothing(
        tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "_GRAPH_LOCK_TIMEOUT", 0.2)
    server, client = _served(tmp_path)
    before = _snapshot(server)
    assert server._graph_lock.acquire()
    try:
        for method, url, kwargs in (
                ("GET", "/graph/state", {}),
                ("PUT", "/graph/state/ball", {"json": {"state": {"position": 5.0,
                                                                 "velocity": 0.0}}}),
                ("PUT", "/graph/params/ball", {"json": {"params": {"gravity": -2.0}}}),
                ("POST", "/sim/step", {})):
            t0 = time.monotonic()
            resp = client.request(method, url, **kwargs)
            waited = time.monotonic() - t0
            assert resp.status_code == 503, (url, resp.status_code, resp.text)
            assert resp.headers.get("retry-after") == "1"
            assert "Nothing was changed" in resp.json()["detail"]
            assert 0.15 < waited < 2.0, (url, waited)
    finally:
        server._graph_lock.release()
    assert _snapshot(server) == before
    assert server.gm._nodes["ball"].node.params["gravity"] == -1.0


TIMEOUT = 0.25


def _queue_behind_a_start(server, client, calls) -> dict:
    """Hold the graph, send ``POST /sim/start`` (which waits for it), then
    each of *calls* 20 ms apart; ``{label: (seconds to answer, status,
    body)}``."""
    log: dict = {}

    def call(label, method, url):
        sent = time.monotonic()
        resp = TestClient(client.app, raise_server_exceptions=False).request(method, url)
        log[label] = (time.monotonic() - sent, resp.status_code, resp.text)

    assert server._graph_lock.acquire()
    threads = []
    try:
        for label, method, url in [("start", "POST", "/sim/start"), *calls]:
            t = threading.Thread(target=call, args=(label, method, url))
            t.start()
            threads.append(t)
            time.sleep(0.02)
        for t in threads:
            t.join(30)
    finally:
        server._graph_lock.release()
    return log


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="REST-044: a runner route behind a waiting /sim/start answers after "
                          "more than one lock timeout; pending fix")
def test_a_runner_route_behind_a_waiting_start_answers_within_one_lock_timeout(
        tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "_GRAPH_LOCK_TIMEOUT", TIMEOUT)
    server, client = _served(tmp_path)
    try:
        log = _queue_behind_a_start(server, client, [("reset 1", "POST", "/sim/reset"),
                                                     ("reset 2", "POST", "/sim/reset")])
    finally:
        _stop(server)
    assert set(log) == {"start", "reset 1", "reset 2"}, log
    for label, (waited, status, _body) in log.items():
        assert waited < 1.6 * TIMEOUT, (label, round(waited, 3), status, log)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="REST-045: POST /sim/stop waits for a /sim/start that is waiting for "
                          "the graph; pending fix")
def test_stop_is_answered_at_once_while_a_start_waits_for_the_graph(tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "_GRAPH_LOCK_TIMEOUT", 1.0)
    server, client = _served(tmp_path)
    try:
        log = _queue_behind_a_start(server, client, [("stop", "POST", "/sim/stop")])
    finally:
        _stop(server)
    waited, status, body = log["stop"]
    assert waited < 0.5, (round(waited, 3), status, body)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="REST-046: a reset that stopped the runner and then timed out on "
                          "the graph says nothing was changed; pending fix")
def test_a_reset_that_stopped_the_runner_does_not_say_nothing_was_changed(
        tmp_path, monkeypatch):
    monkeypatch.setattr(server_module, "_GRAPH_LOCK_TIMEOUT", TIMEOUT)
    server, client = _served(tmp_path, dt=0.001)
    try:
        assert client.post("/sim/start").status_code == 200
        assert _wait_for(lambda: server.relay.step_count > 3)
        assert server._graph_lock.acquire()
        try:
            resp = client.post("/sim/reset")
        finally:
            server._graph_lock.release()
        runner_kept = server.runner is not None and server.runner.is_alive
    finally:
        _stop(server)
    assert resp.status_code == 503, resp.text
    detail = resp.json()["detail"]
    assert runner_kept or "Nothing was changed" not in detail, detail
