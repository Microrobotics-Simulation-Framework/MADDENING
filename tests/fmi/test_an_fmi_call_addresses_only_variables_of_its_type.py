"""A ``get`` or ``set`` that names an FMI type addresses only variables of
that type; a ``set`` names each variable once and never a Clock.

FMI 3.0 ("Getting and Setting Variable Values"): the variable's type in
``modelDescription.xml`` "determines the function
``fmi3Get/Set{VariableType}`` that must be used for accessing the
respective variable values".  The C wrapper carries every width as a
float64, so the bridge could not tell ``fmi3SetBoolean`` from
``fmi3SetFloat32``: a Boolean set of a Float32 parameter stored 1.0, and
``fmi3GetInt32`` of a Float32 output truncated 0.5 to 0 in C.  The wrapper
now sends the type of every call (``"type": "Float32"``), and the bridge
refuses a variable of another one with nothing read or written.  A request
without a type (a Python client of the JSON protocol) is not checked.

Two more things a ``set`` used to accept and then drop: a value reference
named twice (the last value won, the first vanished) and a numeric value
for a ``<Clock>`` (its ticks are implied by time; the value was discarded
with ``ok``).
"""

from __future__ import annotations

import numpy as np
import pytest

from maddening.fmi.tcp_bridge import values_of
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture
def served(gm):
    md, bridge = _bridge(gm)
    yield md, bridge
    bridge.stop()


@pytest.fixture
def clocked(gm):
    md, bridge = _bridge(gm, multi_clock=True)
    yield md, bridge
    bridge.stop()


def _dtype(md, name):
    return next(v.dtype for v in md.variables if v.name == name)


def test_a_set_of_another_type_is_refused_and_writes_nothing(served):
    md, bridge = served
    el, k = _vr(md, "ball.params.elasticity"), _vr(md, "spring.params.stiffness")
    assert _dtype(md, "ball.params.elasticity") == "float32"
    before = bridge.handle({"op": "get", "vr": [el, k]})["values"]
    for wrong in ("Boolean", "Float64", "Int32"):
        reply = bridge.handle({"op": "set", "type": wrong, "vr": [k, el], "values": [40.0, 1.0]})
        assert reply["ok"] is False, wrong
        assert f"is Float32, so fmi3Set{wrong} cannot address it" in reply["error"], reply
    assert bridge.handle({"op": "get", "vr": [el, k]})["values"] == before
    # its own type is accepted, and so is a request that names none
    assert bridge.handle({"op": "set", "type": "Float32", "vr": [el], "values": [0.5]}) == {"ok": True}
    assert bridge.handle({"op": "set", "vr": [el], "values": [0.25]}) == {"ok": True}
    assert bridge.handle({"op": "get", "vr": [el]})["values"] == [0.25]


def test_a_binary_set_carries_its_type_to_the_same_check(served):
    md, bridge = served
    el = _vr(md, "ball.params.elasticity")
    raw = np.asarray([1.0], "<f8").tobytes()
    reply = bridge.handle({"op": "set", "type": "Boolean", "vr": [el], "n": 1, "dtype": "f64",
                           "raw": raw})
    assert reply["ok"] is False and "fmi3SetBoolean cannot address it" in reply["error"], reply
    assert bridge.handle({"op": "set", "type": "Float32", "vr": [el], "n": 1, "dtype": "f64",
                          "raw": raw}) == {"ok": True}


def test_a_get_of_another_type_is_refused(served):
    md, bridge = served
    pos, t = _vr(md, "spring.position"), _vr(md, "time")
    for wrong in ("Int32", "Float64", "Boolean"):
        reply = bridge.handle({"op": "get", "type": wrong, "vr": [pos]})
        assert reply["ok"] is False and "is Float32" in reply["error"], (wrong, reply)
    # a mixed list is refused as a whole: each variable through its own getter
    reply = bridge.handle({"op": "get", "type": "Float64", "vr": [t, pos]})
    assert reply["ok"] is False and "'spring.position' is Float32" in reply["error"]
    assert bridge.handle({"op": "get", "type": "Float64", "vr": [t]}) == {"ok": True, "values": [0.0]}
    assert values_of(bridge.handle({"op": "get", "type": "Float32", "vr": [pos]}))[0] == 0.5


@pytest.mark.parametrize("bad", ["float32", "Double", "", 3, "Clocks", ["Float32"]])
def test_a_type_that_is_not_an_fmi_type_is_refused(served, bad):
    md, bridge = served
    pos = _vr(md, "spring.position")
    for op in ({"op": "get", "type": bad, "vr": [pos]},
               {"op": "set", "type": bad, "vr": [_vr(md, "spring.anchor_position")],
                "values": [0.1]}):
        reply = bridge.handle(op)
        assert reply["ok"] is False and "type must be one of" in reply["error"], (op, reply)


def test_a_value_reference_named_twice_in_one_set_is_refused(served):
    md, bridge = served
    k, anchor = _vr(md, "spring.params.stiffness"), _vr(md, "spring.anchor_position")
    before = bridge.handle({"op": "get", "vr": [k, anchor]})["values"]
    for vrs, values in (([k, k], [45.0, 46.0]), ([anchor, k, anchor], [0.1, 45.0, 0.2]),
                        ([k, k], [30.0, 30.0])):          # even with the same value twice
        reply = bridge.handle({"op": "set", "vr": vrs, "values": values})
        assert reply["ok"] is False and "is named more than once in one set" in reply["error"]
        assert bridge.handle({"op": "get", "vr": [k, anchor]})["values"] == before
    raw = np.asarray([45.0, 46.0], "<f8").tobytes()
    reply = bridge.handle({"op": "set", "vr": [k, k], "n": 2, "dtype": "f64", "raw": raw})
    assert reply["ok"] is False and "more than once" in reply["error"]
    # reading a variable twice is harmless, and still answered
    assert bridge.handle({"op": "get", "vr": [k, k]})["values"] == [before[0]] * 2


def test_a_clock_takes_no_numeric_set(clocked):
    md, bridge = clocked
    clock = next(v for v in md.variables if v.is_clock)
    anchor = _vr(md, "spring.anchor_position")
    for request in ({"op": "set", "vr": [clock.value_reference], "values": [7.5]},
                    {"op": "set", "vr": [anchor, clock.value_reference], "values": [0.3, 0.0]}):
        reply = bridge.handle(request)
        assert reply["ok"] is False and "is a Clock" in reply["error"], reply
    assert bridge.handle({"op": "get", "vr": [anchor]})["values"] == [0.0]   # nothing written
    reply = bridge.handle({"op": "set", "type": "Float64", "vr": [clock.value_reference],
                           "values": [0.0]})
    assert reply["ok"] is False and "is Clock, so fmi3SetFloat64" in reply["error"], reply
    # the other variables of a clocked FMU are set as before
    assert bridge.handle({"op": "set", "type": "Float32", "vr": [anchor], "values": [0.3]}) \
        == {"ok": True}
    assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
