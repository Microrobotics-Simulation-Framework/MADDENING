"""An FMU-state archive carries every pending input, or does not restore.

The bridge's ``set_state`` refused an archive whose state fields or
parameters were not exactly the model's (the key-set rule), and one
without ``_time``.  The ``i/<node>/<field>`` members were optional:
``_decode_state`` started from zero inputs and filled in what was there,
so an archive missing one restored with that input silently back at zero
-- the partial snapshot the key-set rule exists to refuse.  The bridge's
own ``get_state`` always writes every input, so a complete archive is
unaffected.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest

from maddening.fmi.tcp_bridge import state_of, values_of
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture
def served(gm):
    md, bridge = _bridge(gm)
    yield md, bridge
    bridge.stop()


def _members(bridge):
    blob = state_of(bridge.handle({"op": "get_state"}))
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def _wire(members):
    buf = io.BytesIO()
    np.savez(buf, **members)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _anchor(md, bridge):
    return float(values_of(bridge.handle({"op": "get", "vr": [_vr(md, "spring.anchor_position")]}))[0])


def test_an_archive_missing_an_input_member_is_refused_and_restores_nothing(served):
    md, bridge = served
    anchor = _vr(md, "spring.anchor_position")
    assert bridge.handle({"op": "set", "vr": [anchor], "values": [0.4]}) == {"ok": True}
    members = _members(bridge)
    assert "i/spring/anchor_position" in members
    partial = {k: v for k, v in members.items() if k != "i/spring/anchor_position"}
    bridge.handle({"op": "step", "t": 0.0, "dt": DT})
    before = bridge.handle({"op": "get_state"})["state"]
    reply = bridge.handle({"op": "set_state", "state": _wire(partial)})
    assert reply["ok"] is False
    assert "FMU state inputs differ from the model: missing ['i/spring/anchor_position']" \
        in reply["error"], reply
    assert bridge.handle({"op": "get_state"})["state"] == before            # nothing restored
    assert _anchor(md, bridge) == pytest.approx(0.4)


def test_an_archive_with_an_extra_input_member_is_refused(served):
    md, bridge = served
    members = _members(bridge)
    members["i/spring/rest_length"] = np.asarray(0.0, np.float32)
    reply = bridge.handle({"op": "set_state", "state": _wire(members)})
    assert reply["ok"] is False and "extra ['i/spring/rest_length']" in reply["error"], reply


def test_a_complete_archive_restores_its_inputs(served):
    md, bridge = served
    anchor = _vr(md, "spring.anchor_position")
    bridge.handle({"op": "set", "vr": [anchor], "values": [0.4]})
    snapshot = bridge.handle({"op": "get_state"})["state"]
    bridge.handle({"op": "set", "vr": [anchor], "values": [-0.7]})
    assert bridge.handle({"op": "set_state", "state": snapshot})["ok"] is True
    assert _anchor(md, bridge) == pytest.approx(0.4)


def test_a_model_with_no_inputs_restores_an_archive_with_none(gm):
    """The rule is "exactly the model's inputs": none for a model with none."""
    with pytest.warns(UserWarning, match="held at zero"):
        md, bridge = _bridge(gm, selected_inputs=[])
    try:
        assert not [v for v in md.variables if v.causality == "input"]
        members = _members(bridge)
        assert not [k for k in members if k.startswith("i/")]
        assert bridge.handle({"op": "set_state", "state": _wire(members)})["ok"] is True
        members["i/spring/anchor_position"] = np.asarray(0.0, np.float32)
        reply = bridge.handle({"op": "set_state", "state": _wire(members)})
        assert reply["ok"] is False and "extra ['i/spring/anchor_position']" in reply["error"]
    finally:
        bridge.stop()


def test_a_clocked_fmu_archive_carries_no_member_for_its_clocks(gm):
    """Clocks are inputs in the description but hold no value: they are
    neither written into the archive nor expected back."""
    md, bridge = _bridge(gm, multi_clock=True)
    try:
        assert any(v.is_clock for v in md.variables)
        members = _members(bridge)
        assert sorted(k for k in members if k.startswith("i/")) == ["i/spring/anchor_position"]
        assert bridge.handle({"op": "set_state", "state": _wire(members)})["ok"] is True
        members["i/clock_0/"] = np.asarray(0.0)
        assert bridge.handle({"op": "set_state", "state": _wire(members)})["ok"] is False
    finally:
        bridge.stop()
