"""A value reference on the bridge's wire is an integer, never coerced.

``set`` and ``get`` used to look each value reference up as ``int(vr)``,
so ``10.9``, ``10.4`` and ``"10"`` all addressed variable 10 -- a ``set``
wrote it and answered ``ok`` -- and ``true`` read variable 1 (``time``).
An importer that sent a malformed reference was told it had succeeded at
something it never asked for.  Now anything but an integer is the usual
error reply, with nothing read or written, over JSON and binary frames
alike; an ``int`` and a NumPy integer are still accepted.
"""

import os
import struct

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from tests.fmi.test_c_wrapper import _bridge, _graph, _vr

_NOT_INTEGERS = [10.9, 10.4, 10.0, "10", True, False, None, [10], {"vr": 10}]
_IDS = ["fraction_up", "fraction_down", "integral_float", "string", "true", "false",
        "null", "list", "object"]


@pytest.fixture(scope="module")
def bridge_and_md():
    gm = _graph()
    md, bridge = _bridge(gm)
    yield md, bridge
    bridge.stop()


def _stiffness(md, bridge):
    return bridge.handle({"op": "get", "vr": [_vr(md, "spring.params.stiffness")]})["values"]


@pytest.mark.parametrize("bad", _NOT_INTEGERS, ids=_IDS)
def test_set_refuses_a_value_reference_that_is_not_an_integer(bridge_and_md, bad):
    md, bridge = bridge_and_md
    before = _stiffness(md, bridge)
    reply = bridge.handle({"op": "set", "vr": [bad], "values": [55.0]})
    assert reply == {"ok": False,
                     "error": f"ValueError: value reference must be an integer, got {bad!r}"}
    assert _stiffness(md, bridge) == before
    # the binary form of the same request goes through the same check
    raw = struct.pack("<d", 55.0)
    reply = bridge.handle({"op": "set", "vr": [bad], "n": 1, "dtype": "f64", "raw": raw})
    assert reply["ok"] is False and "must be an integer" in reply["error"], reply
    assert _stiffness(md, bridge) == before


@pytest.mark.parametrize("bad", _NOT_INTEGERS, ids=_IDS)
def test_get_refuses_a_value_reference_that_is_not_an_integer(bridge_and_md, bad):
    md, bridge = bridge_and_md
    reply = bridge.handle({"op": "get", "vr": [_vr(md, "time"), bad]})
    assert reply == {"ok": False,
                     "error": f"ValueError: value reference must be an integer, got {bad!r}"}


def test_integer_value_references_are_still_served(bridge_and_md):
    md, bridge = bridge_and_md
    k = _vr(md, "spring.params.stiffness")
    for vr in (k, np.int64(k), np.int32(k)):
        assert bridge.handle({"op": "set", "vr": [vr], "values": [40.0]}) == {"ok": True}
        assert bridge.handle({"op": "get", "vr": [vr]}) == {"ok": True, "values": [40.0]}
    assert bridge.handle({"op": "set", "vr": [k], "values": [30.0]}) == {"ok": True}
    # an integer that names nothing is still "unknown", not "not an integer"
    reply = bridge.handle({"op": "get", "vr": [10_000]})
    assert reply["ok"] is False and "unknown value reference 10000" in reply["error"]
