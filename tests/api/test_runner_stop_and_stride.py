"""``POST /sim/stop`` means the runner has stopped, and ``PUT /sim/stride``
applies what it echoes.

The audited sequence: ``PUT /sim/stride?steps_per_frame=100000000``, then
``POST /sim/stop`` answered ``{"status": "stopped"}`` after two seconds
with the runner's thread still alive (it read its stop flag only between
frames, and a frame was 1e8 steps), the server dropped its handle, ``POST
/sim/reset`` returned position 1.0, and ``GET /graph/state`` then showed
the ball at -196186, -268144, -350556 as the orphaned thread stepped over
the reset.  ``PUT /sim/stride`` before any runner existed echoed a value
and dropped it, and ``relay_stride=0`` was echoed as 0 and applied as 1.
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
from maddening.api.server import (
    MAX_RELAY_STRIDE,
    MAX_STEPS_PER_FRAME,
    SimulationServer,
)
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
from maddening.viz.relay import StateRelay
from maddening.viz.runner import RealtimeRunner


def _make():
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.001, initial_position=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({"BallNode": BallNode}, graph_manager=gm)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False)


def _wait_for(predicate, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# PUT /sim/stride
# ---------------------------------------------------------------------------

def test_a_stride_set_before_the_runner_exists_is_the_runners_stride():
    gm, server, client = _make()
    resp = client.put("/sim/stride", params={"steps_per_frame": 7, "relay_stride": 3})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"steps_per_frame": 7, "relay_stride": 3}
    assert client.post("/sim/start").status_code == 200
    try:
        assert server.runner.steps_per_frame == 7
        assert server.relay.stride == 3
    finally:
        assert client.post("/sim/stop").status_code == 200
    # ... and the next runner, after a stop dropped this one, too.
    assert client.post("/sim/start").status_code == 200
    try:
        assert server.runner.steps_per_frame == 7
    finally:
        assert client.post("/sim/stop").status_code == 200


@pytest.mark.parametrize("params", [
    {"relay_stride": 0}, {"steps_per_frame": 0}, {"relay_stride": -1},
    {"steps_per_frame": MAX_STEPS_PER_FRAME + 1}, {"relay_stride": MAX_RELAY_STRIDE + 1},
    {"steps_per_frame": 100_000_000},
])
def test_a_stride_outside_its_bounds_is_a_422_and_changes_nothing(params):
    gm, server, client = _make()
    client.put("/sim/stride", params={"steps_per_frame": 5, "relay_stride": 2})
    resp = client.put("/sim/stride", params=params)
    assert resp.status_code == 422, resp.text
    assert (server._steps_per_frame, server.relay.stride) == (5, 2)


def test_the_largest_stride_is_accepted_and_echoed_as_applied():
    gm, server, client = _make()
    resp = client.put("/sim/stride", params={"steps_per_frame": MAX_STEPS_PER_FRAME,
                                             "relay_stride": MAX_RELAY_STRIDE})
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"steps_per_frame": MAX_STEPS_PER_FRAME,
                           "relay_stride": MAX_RELAY_STRIDE}


# ---------------------------------------------------------------------------
# POST /sim/stop is honest
# ---------------------------------------------------------------------------

def test_stop_mid_frame_stops_the_thread_and_a_reset_stays_reset(monkeypatch):
    """The audited sequence at the largest frame the API now accepts.  Each
    step is slowed to a fraction of a millisecond so that the frame lasts
    well past the two seconds the old ``stop()`` waited, on any machine:
    stop must not wait for the frame."""
    gm, server, client = _make()
    real_step = gm.step

    def slowed(*args, **kwargs):
        time.sleep(2e-4)
        return real_step(*args, **kwargs)

    monkeypatch.setattr(gm, "step", slowed)
    assert client.post("/sim/start").status_code == 200
    resp = client.put("/sim/stride", params={"steps_per_frame": MAX_STEPS_PER_FRAME})
    assert resp.status_code == 200, resp.text
    thread = server.runner._thread
    # the next frame is the long one; let it begin
    assert _wait_for(lambda: server.runner.sim_time > 0.01)
    t0 = time.perf_counter()
    resp = client.post("/sim/stop")
    elapsed = time.perf_counter() - t0
    assert resp.status_code == 200 and resp.json() == {"status": "stopped"}, resp.text
    assert not thread.is_alive(), "stop answered 'stopped' with the thread alive"
    assert elapsed < 5.0
    reset = client.post("/sim/reset")
    assert reset.status_code == 200, reset.text
    assert reset.json()["state"]["ball"]["position"] == 1.0
    time.sleep(0.3)
    assert client.get("/graph/state").json()["ball"]["position"] == 1.0


@pytest.fixture
def blocked_runner(monkeypatch):
    """A started runner whose thread is stuck inside a step until the test
    releases it, and a stop wait short enough to time out on it."""
    gm, server, client = _make()
    release, inside = threading.Event(), threading.Event()
    real_step = gm.step

    def slow_step(*args, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            inside.set()
            release.wait(30)
        return real_step(*args, **kwargs)

    monkeypatch.setattr(gm, "step", slow_step)
    monkeypatch.setattr(server_module, "_RUNNER_STOP_TIMEOUT", 0.05)
    assert client.post("/sim/start").status_code == 200
    assert inside.wait(10)
    yield gm, server, client, release
    release.set()
    server.runner and server.runner.stop(timeout=10)


def test_a_stop_that_times_out_is_a_503_and_keeps_the_runner(blocked_runner):
    gm, server, client, release = blocked_runner
    thread = server.runner._thread
    resp = client.post("/sim/stop")
    assert resp.status_code == 503, resp.text
    assert resp.headers.get("retry-after") == "1"
    assert server.runner is not None and thread.is_alive()
    # once the step finishes the thread sees the flag; a retry reports it
    release.set()
    assert _wait_for(lambda: not thread.is_alive())
    resp = client.post("/sim/stop")
    assert resp.status_code == 200 and resp.json() == {"status": "stopped"}, resp.text
    assert server.runner is None


def _live_position(gm) -> float:
    """The ball's position, read off the graph.  Not through ``GET
    /graph/state``: a read takes the graph lock, which the stuck step
    holds, so it waits for that step like every other use of the graph."""
    return float(gm._state["ball"]["position"])


def test_while_the_runner_is_still_stopping_state_writes_are_503s(blocked_runner):
    gm, server, client, release = blocked_runner
    assert client.post("/sim/stop").status_code == 503
    position = _live_position(gm)
    for method, url, kwargs in [
        ("POST", "/sim/reset", {}),
        ("POST", "/sim/start", {}),
        ("POST", "/sim/step", {}),
        ("POST", "/sim/run", {"params": {"n_steps": 2}}),
        ("PUT", "/graph/state/ball", {"json": {"state": {"position": 5.0, "velocity": 0.0}}}),
        ("POST", "/checkpoint/load", {"params": {"path": "x.npz"}}),
    ]:
        resp = client.request(method, url, **kwargs)
        assert resp.status_code == 503, (url, resp.text)
    assert _live_position(gm) == position
    release.set()
    assert _wait_for(lambda: not server.runner._thread.is_alive())
    reset = client.post("/sim/reset")
    assert reset.status_code == 200, reset.text
    assert reset.json()["state"]["ball"]["position"] == 1.0


def test_a_reset_whose_stop_times_out_resets_nothing(blocked_runner):
    """Nothing is reset -- but the runner was told to stop, and stays
    stopped: the 503 says so, with ``was_running``.  It used to say
    "Nothing was changed" of a runner it had just stopped."""
    gm, server, client, release = blocked_runner
    before = _live_position(gm)
    resp = client.post("/sim/reset")
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert "Nothing was changed" not in body["detail"]
    assert "the runner was told to stop" in body["detail"]
    assert "stays stopped" in body["detail"] and body["was_running"] is True
    assert resp.headers.get("retry-after") == "1"
    assert _live_position(gm) == before
    release.set()
    assert _wait_for(lambda: not server.runner._thread.is_alive())
    assert client.post("/sim/pause").status_code == 409       # it did stop
    assert _live_position(gm) == pytest.approx(before, abs=1e-3)


def test_a_stop_that_times_out_says_the_runner_stays_stopped(blocked_runner):
    gm, server, client, release = blocked_runner
    resp = client.post("/sim/stop")
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert "Nothing was changed" not in body["detail"]
    assert "stays stopped" in body["detail"] and body["was_running"] is True
    # A retry while it is still stopping: told to stop already, not running.
    again = client.post("/sim/stop")
    assert again.status_code == 503 and again.json()["was_running"] is False


def test_while_the_runner_runs_state_writes_are_409s_and_params_still_go_through():
    """A state written beside a running runner would be overwritten by its
    next store; a param write (the UI's sliders) is read by its next step."""
    gm, server, client = _make()
    assert client.post("/sim/start").status_code == 200
    try:
        for method, url, kwargs in [
            ("POST", "/sim/step", {}),
            ("POST", "/sim/run", {"params": {"n_steps": 2}}),
            ("PUT", "/graph/state/ball", {"json": {"state": {"position": 5.0,
                                                             "velocity": 0.0}}}),
            ("POST", "/checkpoint/load", {"params": {"path": "x.npz"}}),
        ]:
            resp = client.request(method, url, **kwargs)
            assert resp.status_code == 409, (url, resp.text)
            assert "POST /sim/stop first" in resp.json()["detail"]
        resp = client.put("/graph/params/ball", json={"params": {"elasticity": 0.5}})
        assert resp.status_code == 200, resp.text
    finally:
        assert client.post("/sim/stop").status_code == 200


# ---------------------------------------------------------------------------
# RealtimeRunner itself
# ---------------------------------------------------------------------------

def _runner(steps_per_frame=1):
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.001, initial_position=1.0))
    relay = StateRelay()
    relay.attach(gm)
    return gm, RealtimeRunner(gm, relay, steps_per_frame=steps_per_frame)


def test_the_runner_stops_within_a_step_of_a_long_frame():
    gm, runner = _runner(steps_per_frame=10_000_000)
    runner.start()
    assert _wait_for(lambda: runner.sim_time > 0.01)
    t0 = time.perf_counter()
    assert runner.stop(timeout=5.0) is True
    assert time.perf_counter() - t0 < 5.0
    assert not runner.is_alive


def test_the_runner_pauses_within_a_step_of_a_long_frame():
    gm, runner = _runner(steps_per_frame=10_000_000)
    runner.start()
    try:
        assert _wait_for(lambda: runner.sim_time > 0.01)
        runner.pause()
        time.sleep(0.2)
        paused_at = runner.sim_time
        time.sleep(0.3)
        assert runner.sim_time == paused_at
    finally:
        assert runner.stop(timeout=5.0)


def test_a_runner_whose_thread_is_alive_will_not_start_a_second_one():
    gm, runner = _runner()
    release = threading.Event()
    real_step = gm.step

    def stuck(*args, **kwargs):
        release.wait(30)
        return real_step(*args, **kwargs)

    gm.step = stuck
    runner.start()
    try:
        assert runner.stop(timeout=0.05) is False
        assert runner.is_alive
        with pytest.raises(RuntimeError, match="still alive"):
            runner.start()
    finally:
        release.set()
        assert runner.stop(timeout=10) is True


# ---------------------------------------------------------------------------
# DELETE /graph/edges
# ---------------------------------------------------------------------------

def test_deleting_an_edge_that_does_not_exist_is_a_404():
    gm, server, client = _make()
    resp = client.request("DELETE", "/graph/edges", json={
        "source_node": "nope", "target_node": "ball",
        "source_field": "x", "target_field": "y"})
    assert resp.status_code == 404, resp.text
    assert "No edge nope.x -> ball.y" in resp.json()["detail"]
    assert gm._dirty is False
