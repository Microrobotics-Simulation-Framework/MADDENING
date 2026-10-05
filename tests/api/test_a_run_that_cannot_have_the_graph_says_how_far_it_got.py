"""``POST /sim/run`` bounds every wait for the graph, and its 503 says how far it got.

A run steps in slices, taking the graph lock for each.  When a slice could
not have the lock within ``_GRAPH_LOCK_TIMEOUT`` the run answered the
generic 503 "Nothing was changed; retry shortly" -- after the slices
before it had stepped the graph (240 steps in the audit's reproducer), so
a client that retried stepped it twice.  It now answers the
``interrupted`` body the shutdown path gives, ``steps_run`` saying how
many steps were taken and the detail how many remain, with
``Retry-After`` only when nothing was taken.  The run's last access -- the
final state's read, and the compile of ``n_steps=0`` -- waited with no
timeout at all; it is bounded like every other and folded into the same
reply.

The timeout is patched to a fraction of a second; the long holder is the
test thread taking the lock, queued behind the run's slice (the lock is
first come, first served), so the order is deterministic.
"""

from __future__ import annotations

import threading
import time
import warnings

import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api import server as server_module
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import EVENT_STEP, GraphManager
from maddening.nodes import BallNode

TIMEOUT = 0.25


@pytest.fixture
def served(monkeypatch):
    monkeypatch.setattr(server_module, "_GRAPH_LOCK_TIMEOUT", TIMEOUT)
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=1e6, elasticity=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    steps = []
    gm.add_observer(lambda ev, _data: steps.append(1) if ev == EVENT_STEP else None)
    server = SimulationServer({}, graph_manager=gm)
    return gm, server, server.create_app(), steps


def _hold_after_queueing(server: SimulationServer, hold: float) -> threading.Thread:
    """A thread that queues for the graph lock -- behind its current holder
    -- and holds it for *hold* seconds; returns once it is queued."""
    lock = server._graph_lock

    def holder():
        lock.acquire()
        try:
            time.sleep(hold)
        finally:
            lock.release()

    thread = threading.Thread(target=holder, daemon=True)
    queued = len(lock._queue)
    thread.start()
    end = time.monotonic() + 10
    while len(lock._queue) <= queued and lock._owner is not None and time.monotonic() < end:
        time.sleep(0.001)
    return thread


def test_a_run_that_cannot_have_the_graph_between_slices_says_how_many_steps_it_took(served):
    gm, server, app, steps = served
    real = gm._compiled_step

    def slow(*args):
        time.sleep(0.002)
        return real(*args)

    gm._compiled_step = slow
    out: dict = {}
    run = threading.Thread(target=lambda: out.setdefault(
        "resp", TestClient(app).post("/sim/run", params={"n_steps": 100_000})))
    run.start()
    end = time.monotonic() + 30
    while len(steps) < 20 and time.monotonic() < end:
        time.sleep(0.005)
    assert len(steps) >= 20
    with server._graph_lock:            # waits for the slice in flight, then holds
        time.sleep(4 * TIMEOUT)
    run.join(60)
    resp = out["resp"]
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["status"] == "interrupted" and body["n_steps"] == 100_000
    assert body["steps_run"] == len(steps) == server.relay.step_count > 0, (body, len(steps))
    assert "Nothing was changed" not in body["detail"]
    assert f"POST /sim/run?n_steps={100_000 - len(steps)} takes the rest" in body["detail"]
    assert "Retry-After" not in resp.headers


def test_the_final_state_read_is_bounded_and_says_every_step_was_taken(served):
    gm, server, app, steps = served
    real_run = gm.run
    holders = []

    def run_then_queue_a_holder(n, *args, **kwargs):
        result = real_run(n, *args, **kwargs)
        holders.append(_hold_after_queueing(server, 6 * TIMEOUT))
        return result

    gm.run = run_then_queue_a_holder
    t0 = time.monotonic()
    resp = TestClient(app).post("/sim/run", params={"n_steps": 1})
    elapsed = time.monotonic() - t0
    for h in holders:
        h.join(30)
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert (body["status"], body["steps_run"], body["n_steps"]) == ("interrupted", 1, 1)
    assert "final state could not be read" in body["detail"]
    assert "Do not repeat it: every step was taken" in body["detail"]
    assert len(steps) == 1
    # Answered at about the timeout, not when the holder let go.
    assert elapsed < 4 * TIMEOUT, elapsed


def test_a_zero_step_run_behind_a_long_holder_is_a_503_that_took_nothing(served):
    gm, server, app, steps = served
    real_attach = server._ensure_relay_attached
    holders = []

    def attach_then_queue_a_holder():
        real_attach()
        if not holders:
            holders.append(_hold_after_queueing(server, 6 * TIMEOUT))

    server._ensure_relay_attached = attach_then_queue_a_holder
    t0 = time.monotonic()
    resp = TestClient(app).post("/sim/run", params={"n_steps": 0})
    elapsed = time.monotonic() - t0
    for h in holders:
        h.join(30)
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert (body["steps_run"], body["n_steps"]) == (0, 0)
    assert "Nothing was changed; retry shortly" in body["detail"]
    assert resp.headers.get("Retry-After") == "1"
    assert steps == [] and elapsed < 4 * TIMEOUT, elapsed


def test_a_run_with_the_graph_free_answers_its_state(served):
    """The control: no holder, every step taken, the state returned."""
    _gm, server, app, steps = served
    resp = TestClient(app).post("/sim/run", params={"n_steps": 7})
    assert resp.status_code == 200 and "ball" in resp.json()
    assert len(steps) == 7 == server.relay.step_count
