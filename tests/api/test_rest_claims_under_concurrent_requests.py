"""The REST inventory's claims under simultaneous requests on a real loopback server.

The ``concurrent`` domain of ``docs/validation/rest_runpod_claims.yaml``:
the requests meet on one server's socket, event loop and worker pool, as
they do in production, not each on its own in-process client.  Each test
here serves the app with uvicorn on ``127.0.0.1`` (an ephemeral port, in a
thread of this process so the test can read the graph it serves) and talks
to it over HTTP with ``httpx``:

* ``test_the_claim_holds_under_simultaneous_requests_on_a_loopback_server``
  runs ``COPIES`` copies of a row's check from
  ``tests/api/rest_claims_support.py`` at once, released together by a
  barrier, while another thread reads the state the whole time.  The
  checks are written for it: each compares only what its own requests
  could change, every copy writes the same values, and names it creates
  carry the copy's index.
* the tests after it state the claims whose whole point is concurrency --
  the graph lock's serialisation, its 409 and 503, the graph budget,
  runner starts and stops, checkpoint saves of one name, the worker pool
  -- once each, against several simultaneous requests.

A peer here is ``127.0.0.1`` (httpx sends the server's own
``127.0.0.1:<port>`` as Host).  Any other peer is presented the way a
reverse proxy on the same machine presents one: ``X-Forwarded-For``, which
uvicorn trusts from ``127.0.0.1`` (``forwarded_allow_ips``, set
explicitly) -- and which the server answers by asking the token, whatever
peer it names.

Nothing here can reach a cloud provider: no request goes to ``/cloud/*``,
and the module runs under :func:`tests.property.differential.no_cloud_launch`.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import os
import socket
import ssl
import threading
import time
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import httpx
import numpy as np
import pytest

from maddening.api import server as server_module
from tests.api import rest_claims_support as S
from tests.property.differential import no_cloud_launch

#: Copies of a check run at once.
COPIES = 4
#: An older uvicorn (before its sans-I/O WebSocket implementation split the
#: ``Sec-WebSocket-Protocol`` header into the list ASGI specifies) serves
#: WebSockets with the websockets library's legacy implementation, which
#: warns that it is deprecated on the server's threads: the library's
#: warnings, not the server's, let through here.
pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning:websockets.*"),
              pytest.mark.filterwarnings("ignore::DeprecationWarning:uvicorn.*")]


@pytest.fixture(scope="module", autouse=True)
def _offline():
    with no_cloud_launch():
        yield


# ---------------------------------------------------------------------------
# A real server on loopback
# ---------------------------------------------------------------------------

def _http_client(base: str, opened: list):
    def make(*, headers, peer):
        headers = dict(headers)
        if peer is not None and peer != S.LOOPBACK_PEER:
            # Any peer but a direct loopback connection is presented through
            # uvicorn's proxy headers, which trust 127.0.0.1 here; the
            # header itself makes the server ask the token, as the peer does.
            headers["X-Forwarded-For"] = peer[0]
        client = httpx.Client(base_url=base, headers=headers, timeout=60.0)
        opened.append(client)
        return client
    return make


def _run_copy(chk: S.Check, server, base: str, root: Path, index: int) -> None:
    """One copy of *chk*, its clients closed after it."""
    opened: list = []
    make = _http_client(base, opened)
    client = make(headers=dict(S.BEARER) if server.auth.enforced else {}, peer=None)
    try:
        chk.fn(S.Ctx("concurrent", server, None, client, root, copies=COPIES, index=index,
                     make_client=make))
    finally:
        for c in opened:
            c.close()


def simultaneously(jobs, *, timeout: float = 120.0) -> list:
    """Run every job in its own thread, released together; their results,
    or the first exception any raised."""
    barrier = threading.Barrier(len(jobs))
    results: list = [None] * len(jobs)
    errors: list = []

    def run(i, job):
        try:
            barrier.wait(30)
            results[i] = job()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(i, job)) for i, job in enumerate(jobs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
    assert not any(t.is_alive() for t in threads), "a request never returned"
    if errors:
        raise errors[0]
    return results


@contextlib.contextmanager
def reading(base: str, headers: dict):
    """A thread reading the state the whole time; its non-200 answers are
    failures of the claim being exercised beside it."""
    stop = threading.Event()
    bad: list = []

    def read():
        with httpx.Client(base_url=base, headers=headers, timeout=60.0) as c:
            while not stop.is_set():
                for url in ("/graph/state", "/healthz"):
                    resp = c.get(url)
                    if resp.status_code != 200:
                        bad.append((url, resp.status_code, resp.text[:200]))

    t = threading.Thread(target=read)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(30)
    assert not bad, bad[:3]


@contextlib.contextmanager
def collector_off():
    """Python's cyclic garbage collector switched off for the block.

    A test here that times a request shares its process with the server it
    asks, and a full (oldest-generation) collection stops every thread of
    that process -- the client, the server's event loop and its workers --
    for as long as it takes: 0.1 to 0.4 s in these tests on an idle core,
    longer on a busy one.  The collector runs once enough objects have been
    allocated, so a test that has just built and sent a hundred requests is
    where one falls, and one that falls inside a timed request is counted
    as the route's own time.  The two misses reproduced here (jax 0.10.2,
    one busy core: a stop at 0.54 s and at 0.60 s against 0.5 s) each had
    a collection of 0.46 s and of 0.42 s inside the request; CI's three,
    0.70 to 0.76 s, are read the same way.  What the routes are asked for
    -- not to wait for a worker thread, or for the graph -- has nothing to
    do with it.

    A collection another thread has already begun is not stopped, so a
    test enters this before it starts its server or any thread."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


# ---------------------------------------------------------------------------
# Every check, several copies at once
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row", S.rows_for("concurrent"))
def test_the_claim_holds_under_simultaneous_requests_on_a_loopback_server(row, tmp_path):
    """``COPIES`` copies of the row's check, released together by a barrier
    in as many threads, against one uvicorn server on loopback, while a
    reader thread reads the state concurrently the whole time."""
    chk = S.CHECKS[row]
    with S.loopback_server(chk, tmp_path) as (server, base):
        headers = dict(S.BEARER) if server.auth.enforced else {}
        with reading(base, headers):
            simultaneously([lambda i=i: _run_copy(chk, server, base, tmp_path, i)
                            for i in range(COPIES)])


# ---------------------------------------------------------------------------
# The claims whose point is concurrency
# ---------------------------------------------------------------------------

_ANY = S.Check(fn=None, rows=(), bind="any", contexts=frozenset(), server_kw={}, patch={},
               xfail={})


def _client(base, server):
    return httpx.Client(base_url=base, timeout=60.0,
                        headers=dict(S.BEARER) if server.auth.enforced else {})


def test_simultaneous_steps_are_all_taken_bit_identical_to_serial(tmp_path):
    """REST-040, REST-050: six clients each send five POST /sim/step at once
    through one server: thirty steps, the state bit-identical to thirty
    serial steps of a twin, every reply one step on."""
    with S.loopback_server(_ANY, tmp_path) as (server, base):
        def job():
            with _client(base, server) as c:
                return [c.post("/sim/step").status_code for _ in range(5)]

        codes = simultaneously([job] * 6)
        assert [c for cs in codes for c in cs] == [200] * 30
        twin = S.build_graph()
        for _ in range(30):
            twin.step()
        assert server.relay.step_count == 30
        for name in ("ball", "spring", "c"):
            for f, v in twin.get_node_state(name).items():
                assert np.array_equal(np.asarray(server.gm.get_node_state(name)[f]),
                                      np.asarray(v)), (name, f)


def test_simultaneous_writes_beside_a_run_in_progress_are_409s(tmp_path):
    """REST-041, REST-096, REST-051: while a /sim/run is in flight (its slices
    slowed), every write the guide lists, sent at once from its own thread,
    is a 409 that changes nothing, reads are served, and the run's result is
    the serial one."""
    from tests.api.test_graph_lock_refusals_cover_every_route import WRITES

    with S.loopback_server(_ANY, tmp_path) as (server, base):
        with _client(base, server) as c:
            assert c.post("/checkpoint/save", params={"path": "c.npz"}).status_code == 200
        gm = server.gm
        real_run, done = gm.run, threading.Event()

        def slow(n, *args, **kwargs):
            if not done.is_set():
                time.sleep(0.03)
            return real_run(n, *args, **kwargs)

        gm.run = slow
        out: dict = {}
        run = threading.Thread(target=lambda: out.setdefault(
            "r", _client(base, server).post("/sim/run", params={"n_steps": 400})))
        run.start()
        assert S.wait_for(lambda: server.relay.step_count >= 2)
        before = (S.structure(S.Ctx("x", server, None, None, tmp_path)),
                  float(np.asarray(gm.params["nodes"]["spring"]["stiffness"])))

        def write(method, url, kwargs):
            def job():
                with _client(base, server) as c:
                    return c.request(method, url, **kwargs)
            return job

        def read():
            with _client(base, server) as c:
                return c.get("/graph/state/ball")

        replies = simultaneously([write(m, u, k) for _, m, u, k in WRITES] + [read] * 3)
        alive = run.is_alive()
        done.set()
        run.join(60)
        assert alive, "the run ended before the writes; the test proves nothing"
        for (label, *_), resp in zip(WRITES, replies):
            assert resp.status_code == 409, (label, resp.status_code, resp.text)
            assert "POST /sim/run is in progress" in resp.json()["detail"], label
        assert all(r.status_code == 200 for r in replies[len(WRITES):])
        assert (S.structure(S.Ctx("x", server, None, None, tmp_path)),
                float(np.asarray(gm.params["nodes"]["spring"]["stiffness"]))) == before
        assert out["r"].status_code == 200, out["r"].text
        twin = S.build_graph()
        for _ in range(400):
            twin.step()
        for f, v in twin.get_node_state("ball").items():
            assert float(out["r"].json()["ball"][f]) == float(np.asarray(v)), f


def test_simultaneous_params_writes_and_reads_beside_the_runner(tmp_path):
    """REST-042, REST-047: the runner running, eight PUT /graph/params of the
    same value and eight reads at once: every one answered 200, each read
    within two seconds (it waits for the step in flight, not for the
    others), and the running node computes with the value."""
    with S.loopback_server(_ANY, tmp_path) as (server, base):
        with _client(base, server) as c:
            assert c.post("/sim/start").status_code == 200
        assert S.wait_for(lambda: server.relay.step_count > 2)

        def put():
            with _client(base, server) as c:
                return c.put("/graph/params/spring", json={"params": {"stiffness": 46.0}})

        def read():
            with _client(base, server) as c:
                t0 = time.monotonic()
                resp = c.get("/graph/state/ball")
                return resp, time.monotonic() - t0

        replies = simultaneously([put] * 8 + [read] * 8)
        try:
            assert all(r.status_code == 200 for r in replies[:8]), [r.text for r in replies[:8]]
            for resp, elapsed in replies[8:]:
                assert resp.status_code == 200 and elapsed < 2.0, (resp.status_code, elapsed)
            assert server.runner is not None and server.runner.is_alive
            assert float(np.asarray(server.gm.get_node("spring").params["stiffness"])) == 46.0
        finally:
            S.stop_runner(server)


def test_simultaneous_requests_that_cannot_have_the_graph_are_503s(tmp_path):
    """REST-043: the graph held throughout, eight requests at once -- reads,
    PUT /graph/state, PUT /graph/params, /sim/step -- each a 503 'Nothing was
    changed' with Retry-After within about the (patched) timeout, and
    nothing written."""
    chk = S.Check(fn=None, rows=(), bind="any", contexts=frozenset(), server_kw={},
                  patch={"_GRAPH_LOCK_TIMEOUT": 0.3}, xfail={})
    with S.loopback_server(chk, tmp_path) as (server, base):
        lock = server._graph_lock
        before = (float(np.asarray(server.gm.params["nodes"]["spring"]["stiffness"])),
                  {f: float(np.asarray(v)) for f, v in server.gm.get_node_state("ball").items()})
        assert lock.acquire(timeout=20)
        try:
            calls = [("GET", "/graph/state", {}),
                     ("PUT", "/graph/state/ball", {"json": {"state": {"position": 1.0,
                                                                      "velocity": 0.0}}}),
                     ("PUT", "/graph/params/spring", {"json": {"params": {"stiffness": 47.0}}}),
                     ("POST", "/sim/step", {})] * 2

            def job(method, url, kw):
                def run():
                    with _client(base, server) as c:
                        t0 = time.monotonic()
                        return c.request(method, url, **kw), time.monotonic() - t0
                return run

            replies = simultaneously([job(*call) for call in calls])
        finally:
            lock.release()
        for (method, url, _), (resp, elapsed) in zip(calls, replies):
            assert resp.status_code == 503, (method, url, resp.status_code, resp.text)
            assert "Nothing was changed" in resp.json()["detail"]
            assert resp.headers.get("retry-after") == "1"
            assert elapsed < 5.0, (method, url, elapsed)
        assert (float(np.asarray(server.gm.params["nodes"]["spring"]["stiffness"])),
                {f: float(np.asarray(v)) for f, v in server.gm.get_node_state("ball").items()}
                ) == before


def test_simultaneous_node_adds_cannot_take_the_graph_past_its_budget(tmp_path):
    """REST-035, REST-036: at a budget of what the graph holds plus four
    elements, six two-element springs are added at once: exactly two are
    added, the rest refused naming the whole graph, the graph holds no more
    than its budget, and no two constructors ever ran at once."""
    from maddening.nodes import SpringDamperNode

    with S.loopback_server(_ANY, tmp_path) as (server, base):
        held = sum(server_module._state_elements(f) for n, f in server.gm._state.items()
                   if n != "_meta")
        running, overlaps = [0], [0]
        guard = threading.Lock()
        real_init = SpringDamperNode.__init__

        def counting(self, *args, **kwargs):
            with guard:
                running[0] += 1
                overlaps[0] += running[0] > 1
            try:
                time.sleep(0.02)
                real_init(self, *args, **kwargs)
            finally:
                with guard:
                    running[0] -= 1

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(server_module, "MAX_GRAPH_STATE_ELEMENTS", held + 4)
            mp.setattr(SpringDamperNode, "__init__", counting)

            def add(i):
                def job():
                    with _client(base, server) as c:
                        return c.post("/graph/nodes", json={
                            "type": "SpringDamperNode", "name": f"s{i}", "timestep": 0.01,
                            "params": {}})
                return job

            replies = simultaneously([add(i) for i in range(6)])
        codes = sorted(r.status_code for r in replies)
        assert codes == [201, 201, 400, 400, 400, 400], [r.text for r in replies]
        for r in replies:
            if r.status_code == 400:
                assert "whole graph" in r.json()["detail"]
        now = sum(server_module._state_elements(f) for n, f in server.gm._state.items()
                  if n != "_meta")
        assert now <= held + 4
        assert overlaps[0] == 0, f"{overlaps[0]} constructors ran beside another"


def test_simultaneous_starts_start_one_runner(tmp_path):
    """REST-054: six POST /sim/start at once: exactly one 200, the rest 409
    "already started", and one runner thread steps the graph."""
    with S.loopback_server(_ANY, tmp_path) as (server, base):
        def start():
            with _client(base, server) as c:
                return c.post("/sim/start")

        try:
            replies = simultaneously([start] * 6)
            codes = sorted(r.status_code for r in replies)
            assert codes == [200, 409, 409, 409, 409, 409], [r.text for r in replies]
            # the realtime runner's own threads (not the runner routes' pool)
            runners = [t for t in threading.enumerate() if getattr(
                getattr(t, "_target", None), "__qualname__", "") == "RealtimeRunner._run"]
            assert server.runner is not None and server.runner.is_alive
            assert len(runners) == 1, runners
        finally:
            S.stop_runner(server)


def test_simultaneous_stops_leave_no_runner_thread(tmp_path):
    """REST-056: the runner running, six POST /sim/stop at once: each is a
    200 "stopped" or a 409 "not started", at least one stopped it, and no
    runner thread steps the graph afterwards."""
    with S.loopback_server(_ANY, tmp_path) as (server, base):
        with _client(base, server) as c:
            assert c.post("/sim/start").status_code == 200
        assert S.wait_for(lambda: server.relay.step_count > 1)
        runner = server.runner

        def stop():
            with _client(base, server) as c:
                return c.post("/sim/stop")

        replies = simultaneously([stop] * 6)
        assert all(r.status_code in (200, 409) for r in replies), [r.text for r in replies]
        assert any(r.status_code == 200 and r.json()["status"] == "stopped" for r in replies)
        assert not runner.is_alive
        count = server.relay.step_count
        time.sleep(0.1)
        assert server.relay.step_count == count


def test_simultaneous_writes_while_the_runner_is_still_stopping_are_503s(tmp_path):
    """REST-049: the runner told to stop while its thread is held in a step
    past the (patched) stop timeout: the stop is a 503 that keeps it as
    stopping, and six writes and starts sent at once are each a 503 --
    "two runners would step one graph" -- that change nothing."""
    chk = S.Check(fn=None, rows=(), bind="any", contexts=frozenset(), server_kw={},
                  patch={"_RUNNER_STOP_TIMEOUT": 0.2}, xfail={})
    with S.loopback_server(chk, tmp_path) as (server, base):
        gm = server.gm
        with _client(base, server) as c:
            assert c.post("/sim/start").status_code == 200
        assert S.wait_for(lambda: server.relay.step_count > 0)
        real, release, entered = gm._compiled_step, threading.Event(), threading.Event()

        def held(*args):
            entered.set()
            release.wait(30)
            return real(*args)

        gm._compiled_step = held
        try:
            assert entered.wait(20)
            with _client(base, server) as c:
                assert c.post("/sim/stop").status_code == 503
            before = {f: float(np.asarray(v)) for f, v in gm.get_node_state("ball").items()}

            def call(method, url, kw):
                def job():
                    with _client(base, server) as c:
                        return c.request(method, url, **kw)
                return job

            calls = [("PUT", "/graph/state/ball", {"json": {"state": {"position": 1.0,
                                                                      "velocity": 0.0}}}),
                     ("POST", "/sim/step", {}), ("POST", "/sim/start", {})] * 2
            replies = simultaneously([call(*c) for c in calls])
            for (method, url, _), resp in zip(calls, replies):
                assert resp.status_code == 503, (method, url, resp.status_code, resp.text)
            assert {f: float(np.asarray(v)) for f, v in gm.get_node_state("ball").items()} \
                == before
        finally:
            release.set()
            gm._compiled_step = real
        assert S.wait_for(lambda: server.runner is None or not server.runner.is_alive)


def test_a_trace_counts_exactly_its_step_budget_under_simultaneous_steps(tmp_path):
    """REST-103 under concurrency: a JAX trace with a (patched) budget of ten
    steps, then four clients sending five POST /sim/step each at once: the
    trace stops itself having counted exactly ten, not one more or less."""
    from maddening.core.simulation import profiler

    chk = S.Check(fn=None, rows=(), bind="any", contexts=frozenset(), server_kw={},
                  patch={"MAX_JAX_TRACE_STEPS": 10}, xfail={})
    with S.loopback_server(chk, tmp_path) as (server, base):
        with _client(base, server) as c:
            assert c.post("/sim/step").status_code == 200        # compiled before tracing
            assert c.post("/sim/profile/jax/start").status_code == 200
        try:
            def steps():
                with _client(base, server) as c:
                    return [c.post("/sim/step").status_code for _ in range(5)]

            codes = simultaneously([steps] * 4)
            assert [x for cs in codes for x in cs] == [200] * 20
            assert S.wait_for(lambda: not profiler.jax_trace_active(), 20)
            with _client(base, server) as c:
                status = c.get("/sim/profile/jax/status").json()
            assert status["active"] is False and status["steps"] == 10, status
            assert "step budget" in status["stopped_by"]
        finally:
            if profiler.jax_trace_active():
                profiler.stop_jax_trace()


def test_a_run_that_cannot_have_the_graph_among_simultaneous_requests_says_how_far_it_got(
        tmp_path):
    """REST-108 on a real server: a /sim/run in flight (slices slowed), the
    graph then held by the test thread while three reads wait for it at
    once: every read is the 503 'Nothing was changed' (it took nothing), and
    the run is the 503 'interrupted' with ``steps_run`` the steps the graph
    took -- the two kinds of 503 told apart, under load."""
    chk = S.Check(fn=None, rows=(), bind="any", contexts=frozenset(), server_kw={},
                  patch={"_GRAPH_LOCK_TIMEOUT": 0.3}, xfail={})
    with S.loopback_server(chk, tmp_path) as (server, base):
        gm = server.gm
        real_run, held = gm.run, threading.Event()

        def slow(n, *args, **kwargs):
            if not held.is_set():
                time.sleep(0.03)
            return real_run(n, *args, **kwargs)

        gm.run = slow
        out: dict = {}
        run = threading.Thread(target=lambda: out.setdefault(
            "r", _client(base, server).post("/sim/run", params={"n_steps": 100_000})))
        run.start()
        lock = server._graph_lock
        try:
            assert S.wait_for(lambda: server.relay.step_count >= 3)
            assert lock.acquire(timeout=20)
            held.set()

            def read():
                with _client(base, server) as c:
                    return c.get("/graph/state")

            reads = simultaneously([read] * 3)
            run.join(30)
        finally:
            if lock.held():
                lock.release()
            held.set()
            gm.run = real_run
        for resp in reads:
            assert resp.status_code == 503 and "Nothing was changed" in resp.text, resp.text
        body = out["r"].json()
        assert out["r"].status_code == 503 and body["status"] == "interrupted", body
        assert body["steps_run"] == server.relay.step_count > 0


def test_simultaneous_saves_of_one_name_leave_a_manifest_that_hashes_to_the_file(tmp_path):
    """REST-091, REST-101: eight POST /checkpoint/save of one name at once,
    between steps: whichever lands last, the file and the manifest beside it
    belong together (the manifest's SHA-256 is the file's), the file loads,
    and no temporary file is left."""
    with S.loopback_server(_ANY, tmp_path) as (server, base):
        def save():
            with _client(base, server) as c:
                c.post("/sim/step")
                return c.post("/checkpoint/save", params={"path": "one.npz"})

        replies = simultaneously([save] * 8)
        assert all(r.status_code == 200 for r in replies), [r.text for r in replies]
        S._manifest_ok(tmp_path, "one.npz")
        assert not [p for p in tmp_path.rglob("*") if "partial" in p.name or p.name.startswith(
            ".")], S.files(tmp_path)
        with _client(base, server) as c:
            assert c.post("/checkpoint/load", params={"path": "one.npz"}).status_code == 200


def test_healthz_answers_while_every_worker_waits_for_the_graph(tmp_path):
    """REST-018: the graph held, more reads queued than anyio has worker
    threads (40), each waiting for the graph in a worker; /healthz, answered
    on the event loop, still answers within two seconds.  No collection
    runs meanwhile (:func:`collector_off`)."""
    with collector_off(), S.loopback_server(_ANY, tmp_path) as (server, base):
        lock = server._graph_lock
        assert lock.acquire(timeout=20)
        queued = []
        try:
            def read():
                with _client(base, server) as c:
                    return c.get("/graph/state").status_code

            threads = [threading.Thread(target=lambda: queued.append(read())) for _ in range(48)]
            for t in threads:
                t.start()
            assert S.wait_for(lambda: len(lock._queue) >= 40, 20), len(lock._queue)
            with _client(base, server) as c:
                t0 = time.monotonic()
                resp = c.get("/healthz")
                elapsed = time.monotonic() - t0
        finally:
            lock.release()
            for t in threads:
                t.join(60)
        assert resp.status_code == 200 and elapsed < 2.0, (resp.status_code, elapsed)


#: The graph-lock timeout of the saturated-pool test, and how many reads it
#: queues behind a held graph: three times anyio's 40 workers, so a route
#: that needs a worker waits two waves of timed-out reads for one.
#:
#: The timeout is also how long the test has to arrange that: every read
#: must have arrived before the first of them gives up.  The 120 take 0.07
#: to 0.14 s to arrive on three idle cores and up to 0.54 s on three busy
#: ones (0.89 s with a collection among them, before ``collector_off``):
#: too near a 1 s timeout.  The route's bounds are counted in timeouts, so
#: a longer one leaves the claim as it was.
LOCK_TIMEOUT = 2.0
QUEUED = 120
#: Runner routes behind a saturated worker pool: (route, the row, the
#: runner running?, what it must answer, within how many lock timeouts).
_ROUTES = [
    ("POST", "/sim/stop", "REST-045", True, (200, 503), 0.5),
    ("PUT", "/sim/stride?steps_per_frame=2", "REST-105", True, (200,), 0.5),
    ("POST", "/sim/start", "REST-044", False, (503,), 1.6),
    ("POST", "/sim/reset", "REST-046", True, (503,), 1.6),
]


class _ServerClock:
    """When each request reached the app and when its answer began, read on
    the server: an ASGI wrapper around the app :func:`S.loopback_server`
    serves (``SimulationServer.create_app`` patched to return it wrapped).

    What a route does about a request starts when the request reaches the
    app.  A client's clock also counts the time before that -- the request
    waiting for the event loop and the CPU behind whatever else the test
    process does at that moment -- which no route can shorten."""

    def __init__(self) -> None:
        self.events: list = []        # (path, "arrived" | "answered", time); appended on the loop

    def wrap(self, app):
        async def timed(scope, receive, send):
            if scope["type"] != "http":
                return await app(scope, receive, send)
            path = scope["path"]
            self.events.append((path, "arrived", time.monotonic()))

            async def answering(message):
                if message["type"] == "http.response.start":
                    self.events.append((path, "answered", time.monotonic()))
                await send(message)
            return await app(scope, receive, answering)
        return timed

    def count(self, path: str, kind: str) -> int:
        return sum(1 for p, k, _ in list(self.events) if p == path and k == kind)

    def took(self, path: str):
        """Seconds from the last arrival of *path* to its answer; ``None``
        when it was not answered."""
        events = [(k, t) for p, k, t in list(self.events) if p == path]
        arrived = [t for k, t in events if k == "arrived"]
        answered = [t for k, t in events if k == "answered" and arrived and t >= arrived[-1]]
        return answered[0] - arrived[-1] if answered else None


@pytest.mark.parametrize("method, url, row, running, want, timeouts", [
    pytest.param(*r, id=r[1].split("?")[0].rsplit("/", 1)[1]) for r in _ROUTES])
def test_a_runner_route_answers_in_time_while_every_worker_waits_for_the_graph(
        tmp_path, monkeypatch, method, url, row, running, want, timeouts):
    """REST-044, REST-045, REST-046, REST-105 on a real server: the graph held
    by a long request and 120 reads queued for it -- three times anyio's 40
    workers, every one of them waiting -- then a runner route arrives: stop
    and stride answer at once (well inside one lock timeout), start and a
    reset that stopped the runner answer their 503 within about one timeout
    of their arrival, not after a worker freed up.

    Timed on the server, from the route's arrival to its answer
    (:class:`_ServerClock`), asked only once every read has arrived and
    none has been answered, and with the garbage collector off
    (:func:`collector_off`).  The reads used to be sent from threads that
    each built their own client first -- an SSL context each, up to 1.6 s
    for the 120 on three cores -- so the first reads timed out before the
    last were sent, and the route was timed by the client in the middle of
    it: behind clients still being built, 503s being sent and -- in both
    misses reproduced with the collector watched -- a full collection of
    what had just been allocated.  That was 0.70 to 0.76 s on a CI runner
    for a stride or a stop that answers in milliseconds.  A route that
    needed a worker would wait about two lock timeouts here: for the two
    waves of reads queued ahead of it to time out."""
    chk = S.Check(fn=None, rows=(), bind="any", contexts=frozenset(), server_kw={},
                  patch={"_GRAPH_LOCK_TIMEOUT": LOCK_TIMEOUT}, xfail={})
    clock = _ServerClock()
    build = server_module.SimulationServer.create_app
    monkeypatch.setattr(server_module.SimulationServer, "create_app",
                        lambda self: clock.wrap(build(self)))
    path = url.split("?")[0]
    with collector_off(), S.loopback_server(chk, tmp_path) as (server, base):
        if running:
            with _client(base, server) as c:
                assert c.post("/sim/start").status_code == 200
            assert S.wait_for(lambda: server.relay.step_count > 1)
        # Every read's client built before any read is sent, sharing one
        # SSL context (each httpx client otherwise loads its own).
        tls = ssl.create_default_context()
        readers = [httpx.Client(base_url=base, timeout=60.0, verify=tls,
                                headers=dict(S.BEARER) if server.auth.enforced else {})
                   for _ in range(QUEUED)]
        lock = server._graph_lock
        assert lock.acquire(timeout=20)
        threads = [threading.Thread(target=c.get, args=("/graph/state",)) for c in readers]
        try:
            for t in threads:
                t.start()
            # every read at the server and every worker waiting for the
            # graph (a running runner queues too) ...
            assert S.wait_for(lambda: clock.count("/graph/state", "arrived") >= QUEUED
                              and len(lock._queue) >= 40 + running, 20), \
                (clock.count("/graph/state", "arrived"), len(lock._queue))
            # ... and still waiting: none has timed out
            assert clock.count("/graph/state", "answered") == 0, \
                "a read was answered before the route was asked"
            with _client(base, server) as c:
                t0 = time.monotonic()
                try:
                    status = c.request(method, url, timeout=10.0).status_code
                except httpx.TimeoutException:
                    status = None
                seen_by_client = time.monotonic() - t0
            took = clock.took(path)
        finally:
            lock.release()
            for t in threads:
                t.join(60)
            for c in readers:
                c.close()
            S.stop_runner(server)
    assert status in want and took is not None and took < timeouts * LOCK_TIMEOUT, \
        (row, status, took, seen_by_client)


def test_simultaneous_websocket_handshakes_need_the_credential(tmp_path):
    """REST-003, REST-012, REST-024: on a server told it is bound to 0.0.0.0,
    simultaneous handshakes over a real socket: anonymous ones and one from a
    foreign Origin holding the token are refused (1008), the bearer header
    and the subprotocol pair are accepted, the latter selecting
    ``maddening.v1``."""
    from websockets.exceptions import InvalidStatus, ConnectionClosed
    from websockets.sync.client import connect

    from maddening.api.auth import websocket_credentials

    chk = S.Check(fn=None, rows=(), bind="public", contexts=frozenset(), server_kw={},
                  patch={}, xfail={})
    with S.loopback_server(chk, tmp_path) as (server, base):
        url = base.replace("http://", "ws://") + "/ws/state"

        def refused(**kw):
            def job():
                try:
                    with connect(url, open_timeout=10, **kw) as ws:
                        ws.recv(timeout=10)
                except ConnectionClosed as exc:
                    return exc.rcvd is not None and exc.rcvd.code == 1008
                except InvalidStatus as exc:
                    return exc.response.status_code in (401, 403)
                return False
            return job

        def accepted(**kw):
            def job():
                with connect(url, open_timeout=10, **kw) as ws:
                    return ws.subprotocol or "accepted"
            return job

        jobs = [refused(), refused(subprotocols=websocket_credentials("wrong-token")),
                refused(additional_headers={**S.BEARER, "Origin": S.FOREIGN_ORIGIN}),
                accepted(additional_headers=S.BEARER),
                accepted(subprotocols=websocket_credentials(S.TOKEN))] * 2
        results = simultaneously(jobs)
        for i, got in enumerate(results):
            kind = i % 5
            if kind < 3:
                assert got is True, (kind, got)
            elif kind == 3:
                assert got == "accepted", got
            else:
                assert got == "maddening.v1", got
