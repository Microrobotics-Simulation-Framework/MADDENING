"""One ``step`` request takes a bounded number of graph steps, and stops at
the first one after ``stop()``.

A request names its work as ``h / master_dt``, and the bridge used to run
whatever that came to: ``dt = 1e9 * master_dt`` held the connection's
worker -- and the FMU's only instance -- for as long as a billion graph
steps take, and ``stop()`` could only log that the worker was still
inside a request.  Now a request is refused, with nothing advanced, above
``max_steps_per_request`` (default ``MAX_STEPS_PER_REQUEST``), and a step
checks the stop flag before every graph step, so a long one ends promptly
after ``stop()`` and commits nothing.  The first graph step, which
compiles, still cannot be interrupted.
"""

from __future__ import annotations

import socket
import threading
import time

import pytest

from maddening.fmi import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import (
    MAX_STEPS_PER_REQUEST,
    FmuTcpBridge,
    recv_message,
    send_message,
)
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr

SLOW_STEP_S = 0.002


@pytest.fixture(scope="module")
def gm():
    return _graph()


def _clock_and_state(md, bridge):
    names = ("time", "spring.position", "ball.position")
    return bridge.handle({"op": "get", "vr": [_vr(md, n) for n in names]})["values"]


def test_a_request_for_more_graph_steps_than_the_cap_is_refused(gm):
    md = build_model_description(gm, model_name="Plant")
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params))
    bridge = FmuTcpBridge(sidecar, md, master_dt=DT, max_steps_per_request=5)
    try:
        assert bridge.handle({"op": "step", "t": 0.0, "dt": 5 * DT}) == {"ok": True,
                                                                        "t": pytest.approx(5 * DT)}
        before = _clock_and_state(md, bridge)
        reply = bridge.handle({"op": "step", "t": 5 * DT, "dt": 6 * DT})
        assert reply["ok"] is False and "more than the 5 one request may take" in reply["error"]
        assert _clock_and_state(md, bridge) == before                       # nothing advanced
    finally:
        bridge.stop()


def test_the_audits_billion_step_request_is_refused_at_once(gm):
    md, bridge = _bridge(gm)
    try:
        assert MAX_STEPS_PER_REQUEST == 100_000
        before = _clock_and_state(md, bridge)
        started = time.monotonic()
        reply = bridge.handle({"op": "step", "t": 0.0, "dt": 1e9 * DT})
        assert time.monotonic() - started < 1.0
        assert reply["ok"] is False and "more than the 100000" in reply["error"], reply
        reply = bridge.handle({"op": "step", "t": 0.0, "dt": (MAX_STEPS_PER_REQUEST + 1) * DT})
        assert reply["ok"] is False and "more than the 100000" in reply["error"], reply
        # absurd sizes too: the count is judged before it is rounded
        reply = bridge.handle({"op": "step", "t": 0.0, "dt": 1e300})
        assert reply["ok"] is False and "more than the 100000" in reply["error"], reply
        assert _clock_and_state(md, bridge) == before
    finally:
        bridge.stop()


def _slow_bridge(gm, **kw):
    """A bridge whose every graph step takes at least SLOW_STEP_S."""
    md = build_model_description(gm, model_name="Plant")

    def slow_step(state, external_inputs, params):
        time.sleep(SLOW_STEP_S)
        return gm._compiled_step(state, external_inputs, params)

    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=slow_step,
        initial_state=gm._state, params=gm.params))
    return md, FmuTcpBridge(sidecar, md, master_dt=DT, **kw)


def test_a_long_step_stops_at_the_first_graph_step_after_stop(gm):
    md, bridge = _slow_bridge(gm)
    n = 50_000                                          # about 100 s at SLOW_STEP_S
    before = _clock_and_state(md, bridge)
    out: dict = {}
    worker = threading.Thread(
        target=lambda: out.update(reply=bridge.handle({"op": "step", "t": 0.0, "dt": n * DT})),
        daemon=True)
    worker.start()
    time.sleep(0.3)
    bridge.stop()
    stopped = time.monotonic()
    worker.join(timeout=10)
    assert not worker.is_alive(), "the step ran on after stop()"
    assert time.monotonic() - stopped < 2.0
    reply = out["reply"]
    assert reply["ok"] is False and "has been stopped" in reply["error"], reply
    assert "not committed" in reply["error"] and f"of its {n} graph steps" in reply["error"]
    done = int(reply["error"].split("after ")[1].split(" of")[0])
    assert 0 < done < n
    assert _clock_and_state(md, bridge) == before                           # nothing committed


def test_stop_ends_a_connection_worker_that_is_inside_a_long_step(gm):
    """The same on the socket path: ``stop()`` used to return with the
    worker still stepping, holding the instance slot, and log a warning;
    now the worker leaves its step at the next graph step and exits."""
    md, bridge = _slow_bridge(gm)
    bridge.start()
    host, port = bridge.endpoint.split(":")
    with socket.create_connection((host, int(port)), timeout=10) as conn:
        send_message(conn, {"op": "hello"})
        assert recv_message(conn)["ok"]
        send_message(conn, {"op": "step", "t": 0.0, "dt": 50_000 * DT})
        time.sleep(0.3)
        with bridge._live_lock:                                             # noqa: SLF001
            workers = set(bridge._live_workers)                             # noqa: SLF001
        assert workers
        started = time.monotonic()
        bridge.stop()
        assert time.monotonic() - started < 5.0
        for worker in workers:
            worker.join(timeout=5)
            assert not worker.is_alive(), "the connection worker outlived its step"
