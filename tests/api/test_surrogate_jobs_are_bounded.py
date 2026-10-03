"""``POST /surrogate/train``: bounded memory, one job at a time, the live
simulation untouched.

* **Memory.**  Every request number was capped on its own, and one in-cap
  request took 2.2 GB (a 25 000-cell rod: 16 initial conditions x 200 steps
  of history, then copies of it) or 5.9 GB and an out-of-memory error (an
  8192 x 16 network).  The job's memory is estimated from the graph's
  shapes before the worker starts and refused over
  ``MAX_SURROGATE_TRAIN_BYTES``, and ``width**2 * depth`` is bounded by the
  request model.
* **Jobs.**  Any number ran at once, each kept for the life of the
  process; one runs now (409 otherwise) and ``MAX_SURROGATE_JOBS_KEPT``
  finished ones are kept.
* **The live graph.**  The worker reset the live simulation twice, which
  the reply did not mention; the data now come from ``run_sweep``, which
  leaves the graph's state alone.
* **Exit.**  A job's daemon thread still inside XLA at interpreter exit
  aborted the process (a core dump); jobs are told to stop at their next
  epoch, and joined, at exit and at the lifespan shutdown.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.testclient import TestClient

pytest.importorskip("equinox", reason="surrogate training needs the surrogates extra")
pytest.importorskip("optax", reason="surrogate training needs the surrogates extra")

from maddening.api import server as server_module  # noqa: E402
from maddening.api.server import SimulationServer, TrainSurrogateRequest  # noqa: E402
from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.nodes import BallNode, HeatNode  # noqa: E402

QUICK = {"node_name": "ball", "n_epochs": 2, "n_data_steps": 5, "hidden_sizes": [8],
         "batch_size": 16}


def _ball():
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    return gm, server, TestClient(server.create_app(), raise_server_exceptions=False)


def _no_job_may_start(monkeypatch) -> None:
    """Make a job that starts fail at once instead of sweeping: the tests of
    a refusal must not allocate what they check is refused, even when the
    refusal they check is broken."""
    from maddening.surrogates.dataset import DatasetGenerator

    def refuse(*args, **kwargs):
        raise AssertionError("a surrogate job started that should have been refused")

    monkeypatch.setattr(DatasetGenerator, "from_sweep", staticmethod(refuse))


def _wait_status(client, job_id, *, until=lambda s: s != "running", timeout=120.0) -> dict:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = client.get(f"/surrogate/status/{job_id}").json()
        if until(body["status"]):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} still {body['status']} after {timeout} s")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def _surrogate_threads() -> set:
    """The live threads the server starts for training jobs (it names them
    ``maddening-surrogate-<job id>``)."""
    return {t for t in threading.enumerate() if t.name.startswith("maddening-surrogate-")}


def test_a_job_over_the_memory_budget_is_refused_before_anything_is_built(monkeypatch):
    """The audited request: a 25 000-cell rod at the defaults.  Refused from
    the estimate, without a sweep, a dataset or a thread."""
    assert hasattr(server_module, "MAX_SURROGATE_TRAIN_BYTES"), "no memory budget"
    _no_job_may_start(monkeypatch)
    gm = GraphManager()
    gm.add_node(HeatNode("rod", timestep=1e-12, n_cells=25_000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    before = _surrogate_threads()
    resp = client.post("/surrogate/train", json={"node_name": "rod", "n_epochs": 1,
                                                 "hidden_sizes": [8], "batch_size": 64})
    assert resp.status_code == 400, resp.text
    assert "nothing was started" in resp.json()["detail"]
    assert server._surrogate_jobs == {} and server._surrogate_threads == {}
    # No job thread was started.  Counted by the server's own thread name,
    # not by threading.active_count(): the server's worker pool and JAX
    # start and stop threads of their own, which made that count flaky.
    assert _surrogate_threads() - before == set()


def test_the_estimate_is_the_boundary_of_the_refusal(monkeypatch):
    gm, server, client = _ball()
    need = server_module._surrogate_training_bytes(gm, "ball", TrainSurrogateRequest(**QUICK))
    monkeypatch.setattr(server_module, "MAX_SURROGATE_TRAIN_BYTES", need - 1)
    resp = client.post("/surrogate/train", json=QUICK)
    assert resp.status_code == 400, resp.text
    assert server._surrogate_jobs == {}
    monkeypatch.setattr(server_module, "MAX_SURROGATE_TRAIN_BYTES", need)
    resp = client.post("/surrogate/train", json=QUICK)
    assert resp.status_code == 200, resp.text
    assert resp.json()["estimated_bytes"] == need
    assert _wait_status(client, resp.json()["job_id"])["status"] == "done"


def test_the_estimate_grows_with_the_sweep_the_dataset_and_the_network():
    """The terms the estimate counts: the history of every node, the target
    node's dataset, and the network's weights (sized by the node's state)."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", timestep=1e-9, n_cells=1000))
    gm.add_node(HeatNode("other", timestep=1e-9, n_cells=3000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    est = server_module._surrogate_training_bytes
    base = TrainSurrogateRequest(node_name="rod", n_data_steps=100, hidden_sizes=[8],
                                 batch_size=1)
    small = est(gm, "rod", base)
    # 16 conditions x 100 steps x 4000 values of history, 3 x 16 x 99 x 2000
    # of dataset, 12 x (1001 * 8 + 8 * 1000 + 8 + 1000) weights: float32.
    assert small == 4 * (16 * 100 * 4000 + 3 * 16 * 99 * 2000
                         + 12 * (1001 * 8 + 8 * 1000 + 8 + 1000) + 3 * (1001 + 8 + 1000))
    assert est(gm, "rod", base.model_copy(update={"n_data_steps": 200})) > small
    assert est(gm, "rod", base.model_copy(update={"n_data_steps": 10**5})) == \
        est(gm, "rod", base.model_copy(update={"n_data_steps": 200}))
    assert est(gm, "rod", base.model_copy(update={"hidden_sizes": [512, 512]})) > small


def test_a_node_added_between_the_request_and_the_sweep_is_counted(monkeypatch):
    """The estimate a request is admitted on is taken under the graph lock,
    and the job's sweep takes the lock again in its own thread.  A
    ``POST /graph/nodes`` queued behind the request is served first (the
    lock is first come, first served, and the worker does not exist yet),
    so the sweep ran over a graph the estimate never saw: a 2-scalar
    graph's 0.5 MiB admitted a 2.15 GiB sweep.  The worker estimates again
    on the graph it sweeps, and ends the job ``error``, naming the budget,
    before sweeping anything."""
    gm, server, client = _ball()
    server.registry["HeatNode"] = HeatNode
    small = server_module._surrogate_training_bytes(gm, "ball", TrainSurrogateRequest(**QUICK))
    monkeypatch.setattr(server_module, "MAX_SURROGATE_TRAIN_BYTES", small * 4)
    _no_job_may_start(monkeypatch)
    replies: dict = {}

    def call(key, method, url, **kw):
        replies[key] = getattr(TestClient(client.app), method)(url, **kw)

    lock = server._graph_lock
    assert lock.acquire()                    # a long holder of the graph
    try:
        train = threading.Thread(target=call, args=("train", "post", "/surrogate/train"),
                                 kwargs={"json": QUICK})
        train.start()
        assert _until(lambda: len(lock._queue) == 1)
        add = threading.Thread(target=call, args=("add", "post", "/graph/nodes"), kwargs={
            "json": {"type": "HeatNode", "name": "rod", "timestep": 0.01,
                     "params": {"n_cells": 2000, "length": 2000.0}}})
        add.start()
        assert _until(lambda: len(lock._queue) == 2)
    finally:
        lock.release()
    train.join(60)
    add.join(60)
    assert replies["train"].status_code == 200, replies["train"].text
    assert replies["add"].status_code == 201, replies["add"].text
    assert replies["train"].json()["estimated_bytes"] == small
    status = _wait_status(client, replies["train"].json()["job_id"])
    assert status["status"] == "error", status
    assert "MAX_SURROGATE_TRAIN_BYTES" in status["error"], status["error"]
    assert "nothing was swept" in status["error"]
    assert status["estimated_bytes"] > small * 4


def _until(predicate, timeout=20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.002)
    return False


@pytest.mark.parametrize("hidden, ok", [
    ([4096], True), ([2048] * 4, True), ([1024] * 16, True), ([64, 64], True),
    ([4097], False), ([2048] * 5, False), ([8192] * 16, False), ([8, 8192], False),
])
def test_width_squared_times_depth_is_bounded(monkeypatch, hidden, ok):
    assert hasattr(server_module, "MAX_SURROGATE_HIDDEN_WEIGHTS"), "no joint bound"
    _no_job_may_start(monkeypatch)
    gm, server, client = _ball()
    if ok:
        TrainSurrogateRequest(node_name="ball", hidden_sizes=hidden)
        return
    resp = client.post("/surrogate/train", json={"node_name": "ball", "hidden_sizes": hidden})
    assert resp.status_code == 422, resp.text
    assert "width**2 * depth" in resp.text
    assert server._surrogate_jobs == {}


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def test_one_job_runs_at_a_time_and_a_cancelled_one_ends_at_its_next_epoch():
    gm, server, client = _ball()
    first = client.post("/surrogate/train", json={**QUICK, "n_epochs": 10_000})
    assert first.status_code == 200, first.text
    job = first.json()["job_id"]
    _wait_status(client, job, until=lambda s: True)
    assert _wait_status(client, job, until=lambda s: s == "running")["status"] == "running"
    second = client.post("/surrogate/train", json=QUICK)
    assert second.status_code == 409, second.text
    assert job in second.json()["detail"]
    t0 = time.perf_counter()
    server._cancel_training_jobs(timeout=60.0)
    assert time.perf_counter() - t0 < 60.0
    body = client.get(f"/surrogate/status/{job}").json()
    assert body["status"] == "cancelled", body
    assert server._surrogate_threads == {}


def test_finished_jobs_kept_are_bounded(monkeypatch):
    monkeypatch.setattr(server_module, "MAX_SURROGATE_JOBS_KEPT", 2)
    gm, server, client = _ball()
    ids = []
    for _ in range(3):
        resp = client.post("/surrogate/train", json=QUICK)
        assert resp.status_code == 200, resp.text
        ids.append(resp.json()["job_id"])
        assert _wait_status(client, ids[-1])["status"] == "done"
    assert list(server._surrogate_jobs) == ids[1:]
    assert client.get(f"/surrogate/status/{ids[0]}").status_code == 404


def test_training_leaves_the_live_simulation_where_it_was():
    gm, server, client = _ball()
    assert client.post("/sim/run", params={"n_steps": 50}).status_code == 200
    before = client.get("/graph/state").json()
    clock = server.relay.latest_snapshot()[0]
    resp = client.post("/surrogate/train", json=QUICK)
    assert resp.status_code == 200, resp.text
    assert _wait_status(client, resp.json()["job_id"])["status"] == "done"
    assert client.get("/graph/state").json() == before
    assert server.relay.latest_snapshot()[0] == clock


def test_a_running_runner_keeps_running_through_a_job():
    gm, server, client = _ball()
    assert client.post("/sim/start").status_code == 200
    try:
        resp = client.post("/surrogate/train", json=QUICK)
        assert resp.status_code == 200, resp.text
        assert _wait_status(client, resp.json()["job_id"])["status"] == "done"
        assert server.runner.is_alive
    finally:
        assert client.post("/sim/stop").status_code == 200


_EXIT_WITH_A_JOB_ALIVE = textwrap.dedent('''
    import sys, time, warnings
    warnings.simplefilter("ignore")
    from fastapi.testclient import TestClient
    from maddening.api.server import SimulationServer
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes import BallNode
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=10.0))
    gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    client = TestClient(server.create_app())
    job = client.post("/surrogate/train", json={"node_name": "ball", "n_epochs": 10000,
        "n_data_steps": 5, "hidden_sizes": [8], "batch_size": 16}).json()["job_id"]
    while client.get(f"/surrogate/status/{job}").json()["epoch"] < 2:
        time.sleep(0.05)
    print("exiting with the job running", flush=True)
''')


# Per push: tests/api/test_surrogate_jobs_are_bounded.py::test_one_job_runs_at_a_time_and_a_cancelled_one_ends_at_its_next_epoch
@pytest.mark.slow  # a fresh interpreter importing JAX and training: about 8 s
def test_the_interpreter_exits_cleanly_with_a_job_running(tmp_path):
    import maddening

    script = tmp_path / "exit_with_job.py"
    script.write_text(_EXIT_WITH_A_JOB_ALIVE)
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("RUNPOD_", "AWS_", "GOOGLE_", "GCLOUD_", "SKY",
                                        "LAMBDA_", "AZURE_"))}
    env.update(HOME=str(tmp_path), JAX_PLATFORMS="cpu",
               PYTHONPATH=str(Path(maddening.__file__).resolve().parents[1])
               + os.pathsep + env.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, str(script)], env=env, capture_output=True,
                          text=True, timeout=300)
    output = proc.stdout + proc.stderr
    assert "exiting with the job running" in output, output[-2000:]
    assert proc.returncode == 0, output[-2000:]
    assert "terminate called" not in output, output[-2000:]
