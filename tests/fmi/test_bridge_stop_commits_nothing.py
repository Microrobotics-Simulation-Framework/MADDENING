"""Once ``FmuTcpBridge.stop()`` has begun, no request changes the model.

``stop()`` shuts every live connection down and joins its threads, but a
worker that is *inside a request* cannot be interrupted -- a sidecar
step can take longer than the join, and the first one compiles the
graph.  Until 0.4.0 such a worker went on after ``stop()`` returned and
replaced the sidecar's state and the bridge's time, with nothing logged,
while it still held the instance slot.

The invariants:

* ``stop()`` is bounded by one budget over all its threads, whatever is
  still running;
* a thread that outlives that budget is named in a logged warning, with
  the op it is serving;
* a request still running when ``stop()`` begins commits nothing: the
  state, the parameters, the pending inputs and the time are exactly what
  they were when ``stop()`` began;
* after ``stop()``, an in-process ``handle()`` of any op that would change
  the model is refused with nothing written, and reads are still answered.
"""

import io
import logging
import os
import socket
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from maddening.fmi import tcp_bridge
from maddening.fmi.tcp_bridge import recv_message, send_message, state_of
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr

_GRAPH = None

_BUDGET = 0.3
"""The join budget these tests run ``stop()`` under (5 s shipped)."""


def _shared_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _graph()
    return _GRAPH


def _frozen(bridge):
    """Everything a request could change, copied."""
    sc = bridge._sidecar
    return (
        {n: {f: np.asarray(v).copy() for f, v in fs.items()} for n, fs in sc.state.items()},
        {k: np.asarray(v).copy() for k, v in sc.get_params().items()},
        {n: {f: np.asarray(v).copy() for f, v in fs.items()} for n, fs in bridge._inputs.items()},
        bridge._time,
    )


def _assert_same(a, b):
    for x, y in zip(a[:3], b[:3]):
        assert x.keys() == y.keys()
        for k in x:
            if isinstance(x[k], dict):
                assert x[k].keys() == y[k].keys()
                for f in x[k]:
                    np.testing.assert_array_equal(x[k][f], y[k][f])
            else:
                np.testing.assert_array_equal(x[k], y[k])
    assert a[3] == b[3]


def _with_blocking_step(bridge):
    """Make the sidecar's step wait for an event: ``(entered, release)``.

    The hook is the configured ``step_fn``, which is what every path into a
    step calls, so the request is genuinely inside the sidecar when
    ``stop()`` runs.
    """
    entered, release = threading.Event(), threading.Event()
    cfg = bridge._sidecar._config
    real = cfg.step_fn

    def blocking(state, ext, params=None):
        entered.set()
        assert release.wait(timeout=60.0), "the test never released the step"
        return real(state, ext, params) if params is not None else real(state, ext)

    bridge._sidecar._config = cfg.__class__(**{**cfg.__dict__, "step_fn": blocking})
    return entered, release


def test_a_step_still_running_when_stop_returns_commits_nothing(monkeypatch, caplog):
    monkeypatch.setattr(tcp_bridge, "_STOP_JOIN_TIMEOUT", _BUDGET, raising=False)
    md, bridge = _bridge(_shared_graph())
    entered, release = _with_blocking_step(bridge)
    bridge.start()
    host, port = bridge.endpoint.split(":")
    conn = socket.create_connection((host, int(port)), timeout=30)
    try:
        conn.settimeout(30)
        send_message(conn, {"op": "hello"})
        assert recv_message(conn)["ok"]
        # a pending input, so an input commit would show too
        send_message(conn, {"op": "set", "vr": [_vr(md, "spring.anchor_position")],
                            "values": [0.25]})
        assert recv_message(conn)["ok"]
        before = _frozen(bridge)
        send_message(conn, {"op": "step", "dt": DT})
        assert entered.wait(timeout=30.0), "the step never started"
        with bridge._live_lock:
            [worker] = [w for w in bridge._live_workers if w.is_alive()]

        caplog.set_level(logging.WARNING, logger=tcp_bridge.__name__)
        t0 = time.monotonic()
        bridge.stop()
        took = time.monotonic() - t0

        # bounded: one budget, not one per thread and not "until the step ends"
        assert took < _BUDGET + 2.0, took
        assert worker.is_alive(), "the step was blocked; the worker must still be in it"
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("still alive" in m and "maddening-fmu-conn" in m
                   and "serving op 'step'" in m for m in warnings), warnings

        release.set()                          # the step now finishes...
        worker.join(timeout=60.0)
        assert not worker.is_alive()
        _assert_same(_frozen(bridge), before)  # ...and commits nothing
        assert not bridge._busy.locked()
    finally:
        release.set()
        conn.close()


def test_stop_with_nothing_running_logs_nothing(caplog):
    caplog.set_level(logging.WARNING, logger=tcp_bridge.__name__)
    with _bridge(_shared_graph())[1] as bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=10) as conn:
            conn.settimeout(10)
            send_message(conn, {"op": "hello"})
            assert recv_message(conn)["ok"]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def _archive(bridge):
    return state_of(bridge.handle({"op": "get_state"}))


@pytest.mark.parametrize("request_of", [
    lambda md, bridge, blob: {"op": "step", "dt": DT},
    lambda md, bridge, blob: {"op": "set", "vr": [_vr(md, "spring.params.stiffness"),
                                                  _vr(md, "spring.anchor_position")],
                              "values": [55.0, 0.5]},
    lambda md, bridge, blob: {"op": "set_state", "state": blob},
    lambda md, bridge, blob: {"op": "reset"},
], ids=["step", "set", "set_state", "reset"])
def test_a_stopped_bridge_refuses_every_op_that_would_change_the_model(request_of):
    md, bridge = _bridge(_shared_graph())
    # A state that differs from the initial one, so reset and set_state
    # would both be visible, and an archive of the initial one to restore.
    blob = _archive(bridge)
    assert bridge.handle({"op": "set", "vr": [_vr(md, "spring.anchor_position")],
                          "values": [0.25]})["ok"]
    assert bridge.handle({"op": "step", "dt": 3 * DT})["ok"]
    bridge.stop()
    before = _frozen(bridge)
    reply = bridge.handle(request_of(md, bridge, blob))
    assert reply["ok"] is False
    assert "has been stopped" in reply["error"] and "not committed" in reply["error"], reply
    _assert_same(_frozen(bridge), before)
    # reads are still answered
    got = bridge.handle({"op": "get", "vr": [_vr(md, "time")]})
    assert got["ok"] and got["values"] == [pytest.approx(3 * DT)]
    assert bridge.handle({"op": "get_state"})["ok"]
    assert io.BytesIO(_archive(bridge)).read(4) == b"PK\x03\x04"
