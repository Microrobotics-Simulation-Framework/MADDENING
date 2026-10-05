"""The background runner keeps real time after a pause, and the routes say
what it is doing.

* ``RealtimeRunner`` paced ``sim_time - sim_start`` against ``wall -
  wall_start``, both fixed when its loop started, so time spent paused read
  as time the simulation had fallen behind: after ``/sim/resume`` it
  stepped flat out until it caught up -- 3.7 s of simulated time in the
  0.5 s after a 3 s pause of a real-time run.  It is paced from the resume
  now, and a stall longer than ``max_catch_up`` (a first compile, a wait
  for the graph lock) moves the schedule instead of bursting through it.
* Nothing watched the runner's thread: when a step raised and the thread
  died, ``/sim/start`` answered 409 "already started", ``/sim/pause`` and
  ``/sim/resume`` 200, and ``/sim/reset`` ``was_running: true``.
* The structural routes did not refuse while the runner ran -- which is
  how its thread died: an edge whose shapes do not match was taken beside
  it, and the next step's edge validation raised an ``ExceptionGroup``,
  which ``POST /sim/step`` answered with a 500.
"""

from __future__ import annotations

import os
import threading
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, RigidBodyNode, TableNode
from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner


def _wait_for(predicate, timeout=10.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _real_time_runner(**kwargs):
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=1e6, elasticity=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    relay = StateRelay()
    relay.attach(gm)
    return gm, RealtimeRunner(gm, relay, time_scale=1.0, **kwargs)


# ---------------------------------------------------------------------------
# Pacing
# ---------------------------------------------------------------------------

def test_a_resumed_runner_keeps_real_time_instead_of_catching_up_the_pause():
    # A catch-up limit longer than the pause: the resume itself must move
    # the schedule.
    gm, runner = _real_time_runner(max_catch_up=10.0)
    runner.start()
    try:
        assert _wait_for(lambda: runner.sim_time > 0.2)
        runner.pause()
        time.sleep(0.1)
        paused_at = runner.sim_time
        time.sleep(1.0)
        runner.resume()
        time.sleep(0.4)
        advanced = runner.sim_time - paused_at
    finally:
        assert runner.stop(timeout=5.0)
    # Real time is ~0.4 s; catching up the pause was ~1.4 s.
    assert advanced < 0.8, f"{advanced:.2f} s simulated in the 0.4 s after a 1 s pause"


def test_a_stall_longer_than_the_catch_up_limit_is_not_burst_through():
    gm, runner = _real_time_runner(max_catch_up=0.1)
    real_step = gm.step
    stalled = threading.Event()
    at_stall = []

    def step(*args, **kwargs):
        if runner.sim_time > 0.2 and not stalled.is_set():
            at_stall.append(runner.sim_time)
            stalled.set()
            time.sleep(1.0)              # a long compile, a wait for the lock
        return real_step(*args, **kwargs)

    gm.step = step
    runner.start()
    try:
        assert stalled.wait(10)
        time.sleep(1.4)                  # the stall, then 0.4 s of running
        advanced = runner.sim_time - at_stall[0]
    finally:
        assert runner.stop(timeout=5.0)
    # Real time after the stall is ~0.4 s (the stalled step's own 0.01 s
    # aside); bursting through it was ~1.4 s.
    assert advanced < 0.9, f"{advanced:.2f} s simulated over a 1 s stall and 0.4 s after"


def test_a_runner_behind_by_less_than_the_limit_still_catches_up():
    """The catch-up limit is a limit, not a removal: a runner slowed for a
    moment still ends up on its schedule."""
    gm, runner = _real_time_runner(max_catch_up=5.0)
    runner.start()
    try:
        assert _wait_for(lambda: runner.sim_time > 0.05)
        t0, s0 = time.perf_counter(), runner.sim_time
        time.sleep(0.6)
        advanced, wall = runner.sim_time - s0, time.perf_counter() - t0
    finally:
        assert runner.stop(timeout=5.0)
    assert 0.3 < advanced < wall + 0.3


# ---------------------------------------------------------------------------
# What the routes report about a runner whose thread died
# ---------------------------------------------------------------------------

def _server_whose_runner_dies():
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    real_step = gm.step
    calls = []

    def step(*args, **kwargs):
        if threading.current_thread().name.startswith("Thread"):   # the runner's
            calls.append(1)
            if len(calls) > 3:
                raise ValueError("seeded: this step cannot run")
        return real_step(*args, **kwargs)

    gm.step = step
    assert client.post("/sim/start").status_code == 200
    assert _wait_for(lambda: not server.runner.is_alive)
    return gm, server, client


def test_a_runner_whose_thread_died_records_why():
    gm, server, client = _server_whose_runner_dies()
    assert server.runner.error == "ValueError: seeded: this step cannot run"


@pytest.mark.parametrize("route", ["/sim/pause", "/sim/resume"])
def test_pause_and_resume_of_a_dead_runner_are_409s_saying_why(route):
    gm, server, client = _server_whose_runner_dies()
    resp = client.post(route)
    assert resp.status_code == 409, resp.text
    assert "thread stopped" in resp.json()["detail"]
    assert "seeded: this step cannot run" in resp.json()["detail"]


def test_a_dead_runner_is_not_already_started_and_a_new_one_starts():
    gm, server, client = _server_whose_runner_dies()
    dead = server.runner
    resp = client.post("/sim/start")
    assert resp.status_code == 200, resp.text
    assert server.runner is not dead
    gm.step = type(gm).step.__get__(gm)            # the step works again
    assert client.post("/sim/stop").status_code == 200


def test_a_reset_after_the_runner_died_says_it_was_not_running():
    gm, server, client = _server_whose_runner_dies()
    resp = client.post("/sim/reset")
    assert resp.status_code == 200, resp.text
    assert resp.json()["was_running"] is False


def test_stop_of_a_dead_runner_reports_it_stopped_and_why():
    gm, server, client = _server_whose_runner_dies()
    resp = client.post("/sim/stop")
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"status": "stopped",
                           "error": "ValueError: seeded: this step cannot run"}
    assert server.runner is None


# ---------------------------------------------------------------------------
# Structural edits beside the runner
# ---------------------------------------------------------------------------

def _three_nodes():
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01))
    gm.add_node(TableNode("table", timestep=0.01))
    gm.add_node(RigidBodyNode("rb", timestep=0.01))
    gm.add_edge("table", "ball", "position", "table_position")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({"BallNode": BallNode}, graph_manager=gm)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False)


EDGE = {"source_node": "table", "target_node": "ball",
        "source_field": "position", "target_field": "table_position"}


@pytest.mark.parametrize("method, url, kwargs", [
    ("DELETE", "/graph/nodes/table", {}),
    ("POST", "/graph/nodes", {"json": {"type": "BallNode", "name": "b2",
                                       "timestep": 0.01, "params": {}}}),
    ("POST", "/graph/edges", {"json": {"source_node": "ball", "target_node": "rb",
                                       "source_field": "position",
                                       "target_field": "force"}}),
    ("DELETE", "/graph/edges", {"json": EDGE}),
])
def test_a_structural_edit_while_the_runner_runs_is_a_409(method, url, kwargs):
    gm, server, client = _three_nodes()
    nodes, edges = list(gm._nodes), list(gm._edges)
    assert client.post("/sim/start").status_code == 200
    try:
        resp = client.request(method, url, **kwargs)
        assert resp.status_code == 409, resp.text
        assert "POST /sim/stop first" in resp.json()["detail"]
        assert list(gm._nodes) == nodes and list(gm._edges) == edges
        assert server.runner.is_alive
    finally:
        assert client.post("/sim/stop").status_code == 200
    # ... and with the runner stopped it goes through.
    assert client.request(method, url, **kwargs).status_code in (200, 201)


@pytest.mark.parametrize("method, url, kwargs", [
    ("POST", "/sim/step", {}),
    ("POST", "/sim/run", {"params": {"n_steps": 3}}),
    ("POST", "/graph/compile", {}),
    ("POST", "/sim/start", {}),
])
def test_an_edge_the_graph_cannot_validate_is_a_400_on_every_route_that_steps(
        method, url, kwargs):
    gm, server, client = _three_nodes()
    resp = client.post("/graph/edges", json={
        "source_node": "ball", "target_node": "rb",
        "source_field": "position", "target_field": "force"})
    assert resp.status_code == 201, resp.text
    resp = client.request(method, url, **kwargs)
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert "edge validation failed" in detail and "ball.position -> rb.force" in detail
    assert server.runner is None or not server.runner.is_alive
