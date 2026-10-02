"""``POST /sim/run`` stops within a slice when the server shuts down.

A run holds a worker for up to ``MAX_RUN_STEPS`` steps, and uvicorn's
graceful shutdown waits for every in-flight request before it runs the
app's lifespan shutdown: SIGINT during a 100 000-step run of a 256 x 256
lattice was still waiting 60 s later (the run's own length was about an
hour).  The run now steps in slices and checks a stop event between them;
the event is set by SIGINT / SIGTERM (chained ahead of uvicorn's handler,
because the lifespan shutdown comes only after the request), by the
lifespan shutdown, and by ``SimulationServer.request_shutdown()``.  The
interrupted run answers 503 with how many of its steps it took, and the
graph is left after exactly those.
"""

from __future__ import annotations

import os
import signal
import socket
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

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode


def _slow_server():
    """A graph whose every step takes a millisecond: 100 000 steps would
    take minutes."""
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=1e6, elasticity=0.5))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    real = gm._compiled_step

    def slow(*args):
        time.sleep(0.001)
        return real(*args)

    gm._compiled_step = slow
    server = SimulationServer({}, graph_manager=gm)
    return gm, server, server.create_app()


def _start_long_run(app) -> tuple[threading.Thread, dict]:
    out: dict = {}

    def run():
        out["resp"] = TestClient(app, raise_server_exceptions=False).post(
            "/sim/run", params={"n_steps": 100_000})

    thread = threading.Thread(target=run)
    thread.start()
    return thread, out


def _wait_for(predicate, timeout=20.0) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def _assert_interrupted(gm, server, out) -> None:
    resp = out["resp"]
    assert resp.status_code == 503, resp.text
    body = resp.json()
    assert body["status"] == "interrupted" and body["n_steps"] == 100_000
    assert 0 < body["steps_run"] < 100_000
    # The graph is left after exactly the steps the reply says it took.
    assert server.relay.step_count == body["steps_run"]


def test_a_run_stops_at_its_next_slice_when_shutdown_is_requested():
    gm, server, app = _slow_server()
    thread, out = _start_long_run(app)
    assert _wait_for(lambda: server.relay.step_count > 20)
    t0 = time.perf_counter()
    server.request_shutdown()
    thread.join(30)
    assert not thread.is_alive()
    assert time.perf_counter() - t0 < 2.0
    _assert_interrupted(gm, server, out)


def test_sigint_reaches_an_in_flight_run_before_the_servers_own_handler(monkeypatch):
    """The chained handler, from the main thread as uvicorn installs it:
    SIGINT sets the stop event and then calls the handler that was there
    (uvicorn's ``handle_exit``), which still runs."""
    gm, server, app = _slow_server()
    seen = []
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, lambda signum, frame: seen.append(signum))
    try:
        restore = server._chain_shutdown_signals()
        thread, out = _start_long_run(app)
        assert _wait_for(lambda: server.relay.step_count > 20)
        signal.raise_signal(signal.SIGINT)
        thread.join(30)
        restore()
        assert signal.getsignal(signal.SIGINT) is not None
        assert seen == [signal.SIGINT], "the server's own handler was not called"
    finally:
        signal.signal(signal.SIGINT, previous)
    _assert_interrupted(gm, server, out)


def test_the_lifespan_shutdown_stops_the_runner_and_sets_the_stop_event():
    gm, server, app = _slow_server()
    with TestClient(app) as client:
        assert not server._shutdown.is_set()
        assert client.post("/sim/start").status_code == 200
        assert _wait_for(lambda: server.relay.step_count > 5)
    assert server._shutdown.is_set()
    assert server.runner is None
    # A second serve of the same app starts clear.
    with TestClient(app) as client:
        assert not server._shutdown.is_set()
        resp = client.post("/sim/run", params={"n_steps": 3})
        assert resp.status_code == 200, resp.text


def test_a_run_not_interrupted_returns_the_state_as_before():
    gm, server, app = _slow_server()
    resp = TestClient(app).post("/sim/run", params={"n_steps": 50})
    assert resp.status_code == 200
    assert set(resp.json()) == {"ball"}
    assert server.relay.step_count == 50


# ---------------------------------------------------------------------------
# On loopback, under uvicorn
# ---------------------------------------------------------------------------

_SERVER = textwrap.dedent('''
    import sys, time, types, warnings
    warnings.simplefilter("ignore")
    stub = types.ModuleType("maddening.cloud.session")
    def _refuse(*a, **k):
        raise RuntimeError("cloud stubbed out in this test")
    stub.CloudSession = stub.CloudConfig = _refuse
    sys.modules["maddening.cloud.session"] = stub
    import uvicorn
    from maddening.api.server import SimulationServer
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes import BallNode
    gm = GraphManager()
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=1e6, elasticity=0.5))
    gm.compile()
    real = gm._compiled_step
    def slow(*args):
        time.sleep(0.001)
        return real(*args)
    gm._compiled_step = slow
    server = SimulationServer({}, graph_manager=gm, bind_host="127.0.0.1")
    uvicorn.run(server.create_app(), host="127.0.0.1", port=int(sys.argv[1]),
                log_level="warning")
''')


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# Per push: tests/api/test_sim_run_stops_when_the_server_shuts_down.py::test_sigint_reaches_an_in_flight_run_before_the_servers_own_handler
@pytest.mark.slow  # starts a uvicorn server process: about 10 s
def test_sigint_to_a_loopback_server_mid_run_exits_within_seconds(tmp_path):
    import httpx

    script = tmp_path / "serve.py"
    script.write_text(_SERVER)
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith(("RUNPOD_", "AWS_", "GOOGLE_", "GCLOUD_", "SKY",
                                        "LAMBDA_", "AZURE_"))}
    import maddening

    # The tree this test imports, whichever it is.
    src = str(Path(maddening.__file__).resolve().parents[1])
    env.update(HOME=str(home), JAX_PLATFORMS="cpu",
               PYTHONPATH=src + os.pathsep + env.get("PYTHONPATH", ""))
    port = _free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([sys.executable, str(script), str(port)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            try:
                httpx.get(f"{base}/healthz", timeout=1)
                break
            except httpx.HTTPError:
                assert proc.poll() is None, proc.stdout.read().decode()[-2000:]
                time.sleep(0.2)
        out: dict = {}

        def run():
            try:
                out["resp"] = httpx.post(f"{base}/sim/run", params={"n_steps": 100_000},
                                         timeout=300)
            except httpx.HTTPError as exc:
                out["error"] = exc

        client = threading.Thread(target=run)
        client.start()
        time.sleep(2.0)                     # the run is in its slices now
        t0 = time.perf_counter()
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=30)
        exited_after = time.perf_counter() - t0
        client.join(30)
    finally:
        if proc.poll() is None:              # only the process this test started
            proc.kill()
            proc.wait(timeout=30)
    assert exited_after < 10.0, f"the server took {exited_after:.1f} s to exit"
    assert "resp" in out, out
    assert out["resp"].status_code == 503, out["resp"].text
    body = out["resp"].json()
    assert body["status"] == "interrupted" and 0 < body["steps_run"] < 100_000
