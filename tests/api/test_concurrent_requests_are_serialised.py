"""Concurrent requests use the served graph one at a time.

FastAPI runs the routes on a thread pool, and ``GraphManager.step`` reads
the state, runs the compiled step (which releases the GIL) and stores the
result.  Nothing serialised the routes, so two ``POST /sim/step`` in
flight read the same state and one result overwrote the other: the
auditor's 8 clients x 25 steps took 107 steps of 200 (98 of 200 lost on a
re-run), every request answered 200, and the streams' ``sim_time`` -- the
relay counted every notification -- said all 200 had happened.

The invariants: N concurrent steps are N steps, bit-identical to N serial
ones, and counted N times by the relay; and no write -- a state, params, a
checkpoint load -- ever lands while a step is in flight.  The compiled
step is wrapped to take 2 ms (a larger graph's step), which widens the
window a race needs without changing what the step computes.

And the lock must not deadlock with the runner: ``POST /sim/stop`` is
answered while another request holds the graph, a reset beside a running
runner completes, and stopping the runner while holding the lock its
thread needs is refused as the programming error it is.
"""

from __future__ import annotations

import os
import threading
import time
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode

N_THREADS, PER_THREAD = 8, 25
STEP_SECONDS = 0.002


def _graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=1000.0, elasticity=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


class _InFlight:
    """Counts graph operations in flight; ``overlaps`` is how many started
    while another was running."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.active = 0
        self.overlaps = 0
        self.steps = 0

    def wrap(self, fn, *, is_step: bool = False):
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.active:
                    self.overlaps += 1
                self.active += 1
            try:
                time.sleep(STEP_SECONDS)
                return fn(*args, **kwargs)
            finally:
                with self._lock:
                    self.active -= 1
                    if is_step:
                        self.steps += 1
        return wrapper


def _server(tmp_path=None):
    gm = _graph()
    server = SimulationServer({"BallNode": BallNode}, graph_manager=gm,
                              checkpoint_root=str(tmp_path) if tmp_path else None)
    flight = _InFlight()
    gm._compiled_step = flight.wrap(gm._compiled_step, is_step=True)
    return gm, server, flight


def _hammer(app, jobs) -> list:
    """Run each ``(method, url, kwargs)`` list of *jobs* on its own thread
    and client; returns every response's status code."""
    codes, errors = [], []

    def run(calls):
        try:
            with_client = TestClient(app, raise_server_exceptions=False)
            for method, url, kwargs in calls:
                codes.append((url, with_client.request(method, url, **kwargs).status_code))
        except Exception as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(calls,)) for calls in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    assert not any(t.is_alive() for t in threads), "a request never returned"
    return codes


def _serial(n: int) -> tuple[dict, float]:
    gm = _graph()
    elapsed = 0.0
    for _ in range(n):
        gm.step()
        elapsed += gm.timestep
    return {k: np.asarray(v) for k, v in gm.get_node_state("ball").items()}, elapsed


def test_concurrent_steps_are_all_taken_bit_identical_to_serial():
    gm, server, flight = _server()
    app = server.create_app()
    codes = _hammer(app, [[("POST", "/sim/step", {})] * PER_THREAD] * N_THREADS)
    n = N_THREADS * PER_THREAD
    assert [c for _, c in codes] == [200] * n
    assert flight.overlaps == 0, f"{flight.overlaps} steps ran beside another"
    want, elapsed = _serial(n)
    got = gm.get_node_state("ball")
    for field, value in want.items():
        assert np.array_equal(np.asarray(got[field]), value), (field, got[field], value)
    # The streams' clock counts exactly the steps taken.
    assert server.relay.step_count == n
    assert server.relay.elapsed == elapsed
    assert server.relay.latest_snapshot()[0] == elapsed


def test_a_state_write_never_lands_inside_a_step():
    gm, server, flight = _server()
    gm.set_node_state = flight.wrap(gm.set_node_state)
    app = server.create_app()
    put = ("PUT", "/graph/state/ball",
           {"json": {"state": {"position": 500.0, "velocity": 0.0}}})
    jobs = [[("POST", "/sim/step", {})] * 10] * 4 + [[put] * 10]
    codes = _hammer(app, jobs)
    assert all(c == 200 for _, c in codes), codes
    assert flight.overlaps == 0, f"{flight.overlaps} operations ran beside another"
    steps_answered = sum(1 for url, c in codes if url == "/sim/step" and c == 200)
    assert flight.steps == steps_answered == server.relay.step_count == 40


def test_a_params_write_never_lands_inside_a_step():
    """The UI's sliders write params while the graph steps: each write is
    one change as far as any step is concerned, never landing while a step
    is in flight.  Seen from inside the step: the two leaves a write sets
    together do not change while it runs."""
    gm, server, flight = _server()
    torn = []
    real = gm._compiled_step

    def watching(state, ext, params):
        leaves = gm.params["nodes"]["ball"]
        before = (float(leaves["elasticity"]), float(leaves["gravity"]))
        time.sleep(0.008)            # the slider writes are rarer than steps
        out = real(state, ext, params)
        after = (float(leaves["elasticity"]), float(leaves["gravity"]))
        if before != after:
            torn.append((before, after))
        return out

    gm._compiled_step = watching
    app = server.create_app()
    writes = [("PUT", "/graph/params/ball",
               {"json": {"params": {"elasticity": 0.5 + 0.01 * i, "gravity": -9.0 - i}}})
              for i in range(20)]
    codes = _hammer(app, [[("POST", "/sim/step", {})] * 10] * 4 + [writes])
    assert all(c == 200 for _, c in codes), codes
    assert torn == [], f"{len(torn)} params writes landed inside a step"
    assert server.relay.step_count == 40


def test_a_checkpoint_load_never_lands_inside_a_step(tmp_path):
    gm, server, flight = _server(tmp_path)
    app = server.create_app()
    client = TestClient(app, raise_server_exceptions=False)
    assert client.post("/checkpoint/save", params={"path": "c.npz"}).status_code == 200
    gm.load_state = flight.wrap(gm.load_state)
    load = ("POST", "/checkpoint/load", {"params": {"path": "c.npz"}})
    codes = _hammer(app, [[("POST", "/sim/step", {})] * 10] * 4 + [[load] * 5])
    assert all(c == 200 for _, c in codes), codes
    assert flight.overlaps == 0, f"{flight.overlaps} operations ran beside another"


def test_a_run_refuses_writes_and_serves_reads_until_it_returns():
    """``POST /sim/run`` steps slice by slice: a read between two slices is
    answered while it runs, a step beside it is a 409, and the run's
    result is the serial one."""
    gm, server, flight = _server()
    app = server.create_app()
    result = {}

    def run():
        result["r"] = TestClient(app).post("/sim/run", params={"n_steps": 300})

    runner = threading.Thread(target=run)
    runner.start()
    client = TestClient(app, raise_server_exceptions=False)
    assert _wait_for(lambda: flight.steps > 10)
    read = client.get("/graph/state/ball")
    step = client.post("/sim/step")
    still_running = runner.is_alive()
    runner.join(60)
    assert read.status_code == 200
    assert still_running, "the run finished before the read; the test proves nothing"
    assert step.status_code == 409 and "POST /sim/run" in step.json()["detail"], step.text
    assert result["r"].status_code == 200, result["r"].text
    want, _ = _serial(300)
    assert result["r"].json()["ball"]["position"] == float(want["position"])


def _wait_for(predicate, timeout=20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------------------
# The lock and the runner
# ---------------------------------------------------------------------------

def _runner_server():
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.001, initial_position=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({"BallNode": BallNode}, graph_manager=gm)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False)


def test_stop_is_answered_while_another_request_holds_the_graph(tmp_path):
    """The runner's thread waits for the graph lock in slices that watch its
    stop flag: a stop asked for while a long request holds the lock is
    answered at once, not after the request (or a 503 at the stop's own
    timeout)."""
    gm, server, client = _runner_server()
    server.checkpoint_root = tmp_path
    real_save = gm.save_state

    def slow_save(path):
        time.sleep(1.5)
        return real_save(path)

    gm.save_state = slow_save
    assert client.post("/sim/start").status_code == 200
    assert _wait_for(lambda: server.runner.sim_time > 0.005)
    saver = threading.Thread(
        target=lambda: client.post("/checkpoint/save", params={"path": "c.npz"}))
    saver.start()
    time.sleep(0.2)                      # the save holds the graph now
    t0 = time.perf_counter()
    resp = client.post("/sim/stop")
    elapsed = time.perf_counter() - t0
    saver.join(30)
    assert resp.status_code == 200 and resp.json() == {"status": "stopped"}, resp.text
    assert elapsed < 1.0, f"stop waited {elapsed:.2f} s for the request holding the graph"


def test_stopping_the_runner_while_holding_the_graph_lock_is_refused():
    gm, server, client = _runner_server()
    assert client.post("/sim/start").status_code == 200
    try:
        with server._graph_lock, pytest.raises(RuntimeError, match="graph lock"):
            server._stop_runner()
        assert server.runner.is_alive
    finally:
        assert client.post("/sim/stop").status_code == 200


def test_a_reset_beside_a_running_runner_completes():
    gm, server, client = _runner_server()
    assert client.post("/sim/start").status_code == 200
    assert _wait_for(lambda: server.runner.sim_time > 0.01)
    t0 = time.perf_counter()
    resp = client.post("/sim/reset")
    assert resp.status_code == 200, resp.text
    assert time.perf_counter() - t0 < 5.0
    assert resp.json()["was_running"] is True
    assert resp.json()["state"]["ball"]["position"] == 1.0
    assert server.runner is None


def test_reads_and_param_writes_beside_the_runner_complete_and_it_stops():
    gm, server, client = _runner_server()
    assert client.post("/sim/start").status_code == 200
    app = client.app
    jobs = [[("GET", "/graph/state", {})] * 20,
            [("PUT", "/graph/params/ball", {"json": {"params": {"elasticity": 0.5}}})] * 20,
            [("GET", "/graph", {})] * 10]
    codes = _hammer(app, jobs)
    assert all(c == 200 for _, c in codes), codes
    assert server.runner.is_alive
    resp = client.post("/sim/stop")
    assert resp.status_code == 200, resp.text


def test_the_runners_steps_never_see_a_params_write_land_inside_them():
    """The runner takes the graph lock for each step, so a slider write
    while it runs lands between two of its steps, never inside one."""
    gm, server, client = _runner_server()
    torn = []
    real = gm._compiled_step

    def watching(state, ext, params):
        leaves = gm.params["nodes"]["ball"]
        before = (float(leaves["elasticity"]), float(leaves["gravity"]))
        time.sleep(0.004)
        out = real(state, ext, params)
        after = (float(leaves["elasticity"]), float(leaves["gravity"]))
        if before != after:
            torn.append((before, after))
        return out

    gm._compiled_step = watching
    assert client.post("/sim/start").status_code == 200
    try:
        for i in range(20):
            resp = client.put("/graph/params/ball", json={
                "params": {"elasticity": 0.5 + 0.01 * i, "gravity": -9.0 - i}})
            assert resp.status_code == 200, resp.text
    finally:
        assert client.post("/sim/stop").status_code == 200
    assert torn == [], f"{len(torn)} params writes landed inside a runner step"


def test_a_request_waits_for_at_most_the_runners_step_in_flight():
    """A runner behind its schedule releases the graph after each step and
    asks for it again at once.  The lock is first come, first served, so a
    request waiting for it goes before the runner's next step: twenty
    parameter writes took 36 s on a two-core CI runner when it was not."""
    gm, server, client = _runner_server()
    real = gm._compiled_step

    def heavy(*args):
        time.sleep(0.03)                 # always behind a real-time schedule
        return real(*args)

    gm._compiled_step = heavy
    assert client.post("/sim/start").status_code == 200
    waits = []
    try:
        assert _wait_for(lambda: server.runner.sim_time > 0.002)
        for i in range(10):
            t0 = time.perf_counter()
            resp = client.put("/graph/params/ball", json={"params": {"elasticity": 0.5 + 0.01 * i}})
            waits.append(time.perf_counter() - t0)
            assert resp.status_code == 200, resp.text
    finally:
        assert client.post("/sim/stop").status_code == 200
    assert max(waits) < 1.0, f"a write waited {max(waits):.2f} s behind the runner"


def test_the_graph_lock_is_granted_in_the_order_it_was_asked_for():
    from maddening.api.server import _GraphLock

    lock = _GraphLock()
    order, ready = [], []
    lock.acquire()

    def ask(name):
        ready.append(name)
        with lock:
            order.append(name)

    threads = []
    for name in ("a", "b", "c"):
        t = threading.Thread(target=ask, args=(name,), daemon=True)
        t.start()
        threads.append(t)
        assert _wait_for(lambda n=name: n in ready and len(lock._queue) == len(threads))
    # A thread that gives up leaves the order of the rest intact.
    gave_up = []
    quitter = threading.Thread(target=lambda: gave_up.extend(
        [lock.acquire(timeout=0.05), lock.acquire(cancelled=lambda: True)]), daemon=True)
    quitter.start()
    quitter.join(5)
    assert gave_up == [False, False]
    assert len(lock._queue) == 3
    lock.release()
    for t in threads:
        t.join(5)
    assert order == ["a", "b", "c"]


def test_the_graph_lock_is_re_entrant_and_only_its_holder_releases_it():
    from maddening.api.server import _GraphLock

    lock = _GraphLock()
    with lock:
        with lock:
            assert lock.held()
        assert lock.held()
    assert not lock.held()
    with pytest.raises(RuntimeError, match="does not hold"):
        lock.release()
    lock.acquire()
    errors = []
    other = threading.Thread(target=lambda: errors.append(lock.acquire(blocking=False)),
                             daemon=True)
    other.start()
    other.join(5)
    assert errors == [False]
    lock.release()


def test_a_stop_that_arrives_as_the_runner_is_granted_the_lock_costs_no_step():
    """The runner reads its stop flag again once it holds the lock: a stop
    asked for while it waited -- here, at the very moment the lock is
    granted -- must not cost one more step over a state a reset is about to
    write."""
    from maddening.viz.relay import StateRelay
    from maddening.viz.runner import RealtimeRunner

    gm = _graph()
    relay = StateRelay()
    relay.attach(gm)

    class GrantedAsTheStopArrives:
        def acquire_unless(self, event):
            event.set()
            return True

        def release(self):
            pass

    runner = RealtimeRunner(gm, relay, lock=GrantedAsTheStopArrives())
    runner.start()
    runner._thread.join(10)
    assert not runner.is_alive
    assert relay.step_count == 0


def test_a_write_queued_before_a_run_began_is_refused_once_it_has_the_lock():
    """``_graph_access`` asks whether something steps the graph twice: before
    waiting for the lock, and again once it holds it.  The second answer
    is the one that counts: a state write that asked while nothing ran, and
    queued behind a ``POST /sim/run``, gets the lock between the run's start
    and its first slice -- and must be refused there, not written into the
    run.  (The lock is first come, first served, so the order is fixed.)"""
    gm, server, flight = _server()
    app = server.create_app()
    replies: dict = {}

    def call(key, method, url, **kw):
        replies[key] = TestClient(app, raise_server_exceptions=False).request(method, url, **kw)

    lock = server._graph_lock
    assert lock.acquire()                       # a long holder of the graph
    try:
        run = threading.Thread(target=call, args=("run", "POST", "/sim/run"),
                               kwargs={"params": {"n_steps": 20}})
        run.start()
        assert _wait_for(lambda: len(lock._queue) == 1)
        write = threading.Thread(target=call, args=("write", "PUT", "/graph/state/ball"),
                                 kwargs={"json": {"state": {"position": 5.0, "velocity": 0.0}}})
        write.start()
        assert _wait_for(lambda: len(lock._queue) == 2)
    finally:
        lock.release()
    run.join(60)
    write.join(60)
    assert replies["run"].status_code == 200, replies["run"].text
    assert replies["write"].status_code == 409, replies["write"].text
    assert "POST /sim/run is in progress" in replies["write"].json()["detail"]
    want, _ = _serial(20)
    assert replies["run"].json()["ball"]["position"] == float(want["position"])
