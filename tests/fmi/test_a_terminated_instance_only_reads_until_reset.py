"""After ``terminate`` the instance is in FMI 3.0's Terminated state.

Terminated allows reading variables (``fmi3Get*``), the FMU-state functions
and ``fmi3Reset``.  The bridge answered ``terminate`` with ``ok`` and
changed nothing, so an importer's ``fmi3DoStep`` after ``fmi3Terminate``
advanced the model and answered ``fmi3OK``.  Now ``step``, ``set`` and
``initialize`` are refused there, with nothing written, until ``reset``
(or a new instance) starts it again.  A malformed request is still refused
for what is wrong with it.
"""

from __future__ import annotations

import socket

import pytest

from maddening.fmi.tcp_bridge import recv_message, send_message
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture
def served(gm):
    md, bridge = _bridge(gm)
    yield md, bridge
    bridge.stop()


def _values(md, bridge):
    names = ("time", "spring.position", "spring.anchor_position", "spring.params.stiffness")
    return bridge.handle({"op": "get", "vr": [_vr(md, n) for n in names]})["values"]


def test_step_set_and_initialize_are_refused_after_terminate(served):
    md, bridge = served
    assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
    before = _values(md, bridge)
    assert bridge.handle({"op": "terminate"}) == {"ok": True}
    assert bridge.handle({"op": "terminate"}) == {"ok": True}          # idempotent
    for request in ({"op": "step", "t": DT, "dt": DT},
                    {"op": "set", "vr": [_vr(md, "spring.anchor_position")], "values": [0.3]},
                    {"op": "set", "vr": [_vr(md, "spring.params.stiffness")], "values": [40.0]}):
        reply = bridge.handle(request)
        assert reply["ok"] is False and "has been terminated" in reply["error"], (request, reply)
    assert _values(md, bridge) == before                                # nothing written
    # initialize is refused already once the instance has stepped; before
    # any step it is refused for being terminated
    assert bridge.handle({"op": "reset"}) == {"ok": True}
    assert bridge.handle({"op": "terminate"}) == {"ok": True}
    reply = bridge.handle({"op": "initialize", "t": 0.5})
    assert reply["ok"] is False and "has been terminated" in reply["error"], reply


def test_reading_and_the_fmu_state_functions_stay_allowed(served):
    md, bridge = served
    assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
    snap = bridge.handle({"op": "get_state"})["state"]
    assert bridge.handle({"op": "step", "t": DT, "dt": DT})["ok"]
    assert bridge.handle({"op": "terminate"}) == {"ok": True}
    assert bridge.handle({"op": "get", "vr": [_vr(md, "time")]})["values"] == [2 * DT]
    assert bridge.handle({"op": "get_state"})["ok"]
    assert bridge.handle({"op": "set_state", "state": snap}) == {"ok": True, "t": DT}
    # a restore does not leave Terminated: only reset does
    assert "has been terminated" in bridge.handle({"op": "step", "t": DT, "dt": DT})["error"]


def test_a_malformed_request_is_refused_for_what_is_wrong_with_it(served):
    md, bridge = served
    assert bridge.handle({"op": "terminate"}) == {"ok": True}
    reply = bridge.handle({"op": "step", "t": 0.0, "dt": 1.5 * DT})
    assert reply["ok"] is False and "whole multiple" in reply["error"], reply
    reply = bridge.handle({"op": "set", "vr": [10_000], "values": [0.0]})
    assert reply["ok"] is False and "unknown value reference" in reply["error"], reply


def test_reset_and_a_new_instance_start_it_again(served):
    md, bridge = served
    assert bridge.handle({"op": "terminate"}) == {"ok": True}
    assert bridge.handle({"op": "reset"}) == {"ok": True}
    assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
    assert bridge.handle({"op": "terminate"}) == {"ok": True}
    bridge.start()
    host, port = bridge.endpoint.split(":")
    with socket.create_connection((host, int(port)), timeout=10) as conn:
        send_message(conn, {"op": "hello"})
        assert recv_message(conn)["ok"]
        send_message(conn, {"op": "step", "t": 0.0, "dt": DT})
        assert recv_message(conn)["ok"] is True          # a new instance is not terminated
