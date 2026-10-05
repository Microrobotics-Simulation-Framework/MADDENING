"""An Int64 / UInt64 value a float64 cannot hold is refused, never rounded.

The wire carries every value as a float64, which holds every value of
every FMI type but two: an ``Int64`` or ``UInt64`` above 2**53 in
magnitude is a float64 only where it is a multiple of the spacing there.
Nothing guarded the others.  A ``get`` widened the model's int64 to
float64 (``np.asarray(state, dtype=np.float64)``), so ``fmi3GetInt64``
answered ``fmi3OK`` with a neighbouring integer -- 9007199254740992 for a
model holding 9007199254740993 -- and a JSON ``set`` of ``2**53 + 1`` was
converted before it was checked and stored as ``2**53``.

Now: a ``get`` of a variable holding such a value is refused, naming the
variable and the integer, and so is a ``set`` of one to an integer
variable; a value a float64 *is* crosses exactly, at any magnitude.  The
value check itself (``_checked_value``) no longer compares through float64
either.  The same values through the compiled wrapper, the sidecar and the
graph are the typed battery of ``tests/property/test_differential_fmu.py``.

x64 is process-global, so every test turns it on for its own duration.
"""

from __future__ import annotations

import contextlib
import json
import socket
import struct

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.fmi import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig, _checked_value
from maddening.fmi.tcp_bridge import FmuTcpBridge, recv_message, send_message

DT = 0.01
BIG = 2 ** 53


@contextlib.contextmanager
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


@stability(StabilityLevel.STABLE)
class _Counter(SimulationNode):
    """``ticks <- ticks + 1 + add`` in int64; ``seen <- stamp`` in uint64;
    ``row <- row + bump`` for an int64 array of three; ``level <- gauge`` in
    float64."""

    def __init__(self, name, start=0):
        super().__init__(name, DT)
        self._start = start

    def initial_state(self):
        return {"ticks": jnp.asarray(self._start, jnp.int64),
                "seen": jnp.asarray(0, jnp.uint64),
                "row": jnp.zeros(3, jnp.int64),
                "level": jnp.asarray(0.0, jnp.float64)}

    def boundary_input_spec(self):
        return {"add": BoundaryInputSpec(shape=(), dtype=np.int64, default=jnp.zeros((), jnp.int64)),
                "stamp": BoundaryInputSpec(shape=(), dtype=np.uint64,
                                           default=jnp.zeros((), jnp.uint64)),
                "bump": BoundaryInputSpec(shape=(3,), dtype=np.int64,
                                          default=jnp.zeros(3, jnp.int64)),
                "gauge": BoundaryInputSpec(shape=(), dtype=np.float64,
                                           default=jnp.zeros((), jnp.float64))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        zero = jnp.zeros((), jnp.int64)
        return {"ticks": state["ticks"] + 1 + jnp.asarray(boundary_inputs.get("add", zero), jnp.int64),
                "seen": jnp.asarray(boundary_inputs.get("stamp", 0), jnp.uint64),
                "row": state["row"] + jnp.asarray(boundary_inputs.get("bump", jnp.zeros(3)), jnp.int64),
                "level": jnp.asarray(boundary_inputs.get("gauge", 0.0), jnp.float64)}


@contextlib.contextmanager
def _served(start=0):
    with _x64():
        gm = GraphManager()
        gm.add_node(_Counter("ctr", start))
        gm.add_external_input("ctr", "add", dtype=jnp.int64)
        gm.add_external_input("ctr", "stamp", dtype=jnp.uint64)
        gm.add_external_input("ctr", "bump", shape=(3,), dtype=jnp.int64)
        gm.add_external_input("ctr", "gauge", dtype=jnp.float64)
        gm.compile()
        md = build_model_description(gm, model_name="Counter")
        sidecar = FmuSidecar(SidecarConfig(
            schema_token=md.instantiation_token, step_fn=gm._compiled_step,   # noqa: SLF001
            initial_state=gm._state, params=gm.params,                        # noqa: SLF001
            input_resolver=gm._resolve_external_inputs))                      # noqa: SLF001
        bridge = FmuTcpBridge(sidecar, md, master_dt=gm.timestep)
        try:
            yield bridge, {v.name: v for v in md.variables}
        finally:
            bridge.stop()


def _set(bridge, var, values, fmi_type=None):
    req = {"op": "set", "vr": [var.value_reference], "values": values}
    if fmi_type:
        req["type"] = fmi_type
    return bridge.handle(req)


def _get(bridge, var, fmi_type=None):
    req = {"op": "get", "vr": [var.value_reference]}
    if fmi_type:
        req["type"] = fmi_type
    return bridge.handle(req)


def _held(bridge, field) -> int:
    return int(np.asarray(bridge._sidecar.state["ctr"][field]))               # noqa: SLF001


# ------------------------------------------------------------------------ get

def test_a_get_of_an_int64_a_float64_cannot_hold_is_refused_not_rounded():
    """The audit's counter: from 2**53 the model holds 2**53 + 1 .. + 4.
    The odd ones are no float64 and were read as their even neighbours,
    ``ok``; they are refused, and the even ones read exactly."""
    with _served(start=BIG) as (bridge, v):
        ticks = v["ctr.ticks"]
        assert ticks.dtype == "int64"
        for k in range(1, 5):
            assert bridge.handle({"op": "step", "t": (k - 1) * DT, "dt": DT})["ok"]
            held = _held(bridge, "ticks")
            assert held == BIG + k
            for fmi_type in ("Int64", None):
                reply = _get(bridge, ticks, fmi_type)
                if k % 2:
                    assert reply["ok"] is False, (k, reply)
                    assert f"variable 'ctr.ticks' holds {held}" in reply["error"], reply
                    assert f"a get would answer {int(float(held))}" in reply["error"]
                    assert "Int64 above 2**53" in reply["error"] and "nothing was read" in reply["error"]
                else:
                    assert reply == {"ok": True, "values": [float(held)]}, (k, reply)
                    assert int(reply["values"][0]) == held
        # the refusal is of that variable alone: time still reads, and a get
        # naming both fails as a whole
        assert bridge.handle({"op": "step", "t": 4 * DT, "dt": DT})["ok"]      # 2**53 + 5
        assert _get(bridge, v["time"]) == {"ok": True, "values": [pytest.approx(5 * DT)]}
        both = bridge.handle({"op": "get", "vr": [v["time"].value_reference,
                                                  ticks.value_reference]})
        assert both["ok"] is False and "'ctr.ticks' holds" in both["error"]


@pytest.mark.parametrize("field, value", [
    ("stamp", 2 ** 53 + 1), ("stamp", 2 ** 64 - 1), ("stamp", 2 ** 63 + 1),
    ("add", 2 ** 63 - 1), ("add", -(2 ** 53) - 1), ("add", -(2 ** 63) + 1)])
def test_a_get_of_an_input_holding_such_a_value_is_refused_too(field, value):
    """Every causality goes through the same conversion.  The value reaches
    the input through the FMU-state archive, which carries the type's own
    bytes, and is then refused by ``get`` -- as an input and, a step later,
    as the uint64 output."""
    import base64
    import io

    from maddening.fmi.tcp_bridge import state_of

    with _served() as (bridge, v):
        with np.load(io.BytesIO(state_of(bridge.handle({"op": "get_state"}))),
                     allow_pickle=False) as data:
            members = {k: data[k] for k in data.files}
        dtype = np.uint64 if field == "stamp" else np.int64
        members[f"i/ctr/{field}"] = np.asarray(value, dtype)
        buf = io.BytesIO()
        np.savez(buf, **members)
        assert bridge.handle({"op": "set_state",
                              "state": base64.b64encode(buf.getvalue()).decode()})["ok"]
        assert int(np.asarray(bridge._inputs["ctr"][field])) == value           # noqa: SLF001
        reply = _get(bridge, v[f"ctr.{field}"])
        assert reply["ok"] is False and f"holds {value}" in reply["error"], reply
        if field == "stamp":
            assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
            assert _held(bridge, "seen") == value                               # exact in the model
            reply = _get(bridge, v["ctr.seen"], "UInt64")
            assert reply["ok"] is False and f"'ctr.seen' holds {value}" in reply["error"]
            assert "UInt64 above 2**53" in reply["error"]


@pytest.mark.parametrize("value", [2 ** 53, -(2 ** 53), 2 ** 53 + 2, 2 ** 62, -(2 ** 63),
                                   2 ** 63 - 1024, 2 ** 60 + 2 ** 20])
def test_an_int64_that_is_a_float64_crosses_exactly_at_any_magnitude(value):
    """Not a blanket limit at 2**53: a value the wire can carry is set, held
    and read back as itself, as a JSON integer and as a JSON float."""
    with _served() as (bridge, v):
        for spelled in (value, float(value)):
            assert _set(bridge, v["ctr.add"], [0], "Int64") == {"ok": True}
            assert _set(bridge, v["ctr.add"], [spelled], "Int64") == {"ok": True}, spelled
            assert int(np.asarray(bridge._inputs["ctr"]["add"])) == value        # noqa: SLF001
            reply = _get(bridge, v["ctr.add"], "Int64")
            assert reply["ok"] and int(reply["values"][0]) == value


def test_one_entry_of_an_array_variable_refuses_the_get_and_names_the_entry():
    with _served() as (bridge, v):
        assert _set(bridge, v["ctr.bump"], [1, 2 ** 60, -3]) == {"ok": True}
        assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
        assert _get(bridge, v["ctr.row"])["values"] == [1.0, float(2 ** 60), -3.0]
        assert _set(bridge, v["ctr.bump"], [0, 1, 0]) == {"ok": True}
        assert bridge.handle({"op": "step", "t": DT, "dt": DT})["ok"]          # row[1] = 2**60 + 1
        reply = _get(bridge, v["ctr.row"])
        assert reply["ok"] is False, reply
        assert f"'ctr.row' holds {2 ** 60 + 1} (entry 1)" in reply["error"], reply


# ------------------------------------------------------------------------ set

@pytest.mark.parametrize("field, fmi_type, value", [
    ("add", "Int64", 2 ** 53 + 1), ("add", "Int64", -(2 ** 53) - 1), ("add", "Int64", 2 ** 63 - 1),
    ("stamp", "UInt64", 2 ** 53 + 1), ("stamp", "UInt64", 2 ** 64 - 1)])
def test_a_json_set_of_an_integer_a_float64_cannot_hold_is_refused(field, fmi_type, value):
    """It used to be converted to float64 first and compared with itself:
    stored as the neighbouring integer, ``ok``."""
    with _served() as (bridge, v):
        var = v[f"ctr.{field}"]
        assert _set(bridge, var, [5], fmi_type) == {"ok": True}
        for request_type in (fmi_type, None):
            reply = _set(bridge, var, [value], request_type)
            assert reply["ok"] is False, reply
            assert f"variable {var.name!r}: the integer {value} cannot be carried exactly" in \
                reply["error"], reply
            assert f"it would be stored as {int(float(value))}" in reply["error"]
            assert int(np.asarray(bridge._inputs["ctr"][field])) == 5            # noqa: SLF001
        # an integer array from an in-process caller is held to the same rule
        reply = _set(bridge, var, np.asarray([value], np.uint64 if value > 0 else np.int64))
        assert reply["ok"] is False and "cannot be carried exactly" in reply["error"], reply
        assert int(np.asarray(bridge._inputs["ctr"][field])) == 5                # noqa: SLF001


def test_one_such_integer_in_an_array_set_refuses_the_whole_set():
    with _served() as (bridge, v):
        assert _set(bridge, v["ctr.bump"], [1, 2, 3]) == {"ok": True}
        reply = _set(bridge, v["ctr.bump"], [10, 20, 2 ** 53 + 1])
        assert reply["ok"] is False and f"the integer {2 ** 53 + 1}" in reply["error"], reply
        assert np.asarray(bridge._inputs["ctr"]["bump"]).tolist() == [1, 2, 3]   # noqa: SLF001
        # naming two variables: the one that cannot take it refuses both
        reply = bridge.handle({"op": "set", "vr": [v["ctr.gauge"].value_reference,
                                                   v["ctr.add"].value_reference],
                               "values": [0.5, 2 ** 53 + 1]})
        assert reply["ok"] is False and "'ctr.add'" in reply["error"], reply
        assert float(np.asarray(bridge._inputs["ctr"]["gauge"])) == 0.0          # noqa: SLF001


def test_a_float_variable_takes_the_nearest_float_of_an_integer_literal():
    """For a float variable an integer literal means what a decimal literal
    means, the nearest float: 2**53 + 1 for a Float64 is 2**53, as 0.1 is
    the float nearest a tenth.  Only an integer variable is another
    integer for it."""
    with _served() as (bridge, v):
        assert _set(bridge, v["ctr.gauge"], [2 ** 53 + 1], "Float64") == {"ok": True}
        assert float(np.asarray(bridge._inputs["ctr"]["gauge"])) == float(2 ** 53)  # noqa: SLF001
        assert _set(bridge, v["ctr.gauge"], [10 ** 23]) == {"ok": True}
        assert float(np.asarray(bridge._inputs["ctr"]["gauge"])) == 1e23            # noqa: SLF001


@pytest.mark.parametrize("field, value, why", [
    ("add", 2 ** 63, "does not fit its type int64"),
    ("add", 2.0 ** 63, "does not fit its type int64"),
    ("add", -(2.0 ** 63) - 2048, "does not fit its type int64"),
    ("add", 0.5, "does not fit its type int64"),
    ("stamp", 2 ** 64, "does not fit its type uint64"),
    ("stamp", 2.0 ** 64, "does not fit its type uint64"),
    ("stamp", -1, "does not fit its type uint64"),
    ("stamp", -1.0, "does not fit its type uint64"),
    ("stamp", 2 ** 64 + 1, "cannot be carried exactly")])
def test_a_value_outside_a_64_bit_integer_type_is_refused(field, value, why):
    """One past the type's maximum is a power of two, exact in a float64:
    2**63 for an Int64 and 2**64 for a UInt64.  The value check compared
    the cast with the value through float64, where the type's maximum
    rounds up to exactly that number -- so wherever an out-of-range cast
    saturates, the maximum was stored for it."""
    with _served() as (bridge, v):
        reply = _set(bridge, v[f"ctr.{field}"], [value])
        assert reply["ok"] is False and why in reply["error"], reply
        assert int(np.asarray(bridge._inputs["ctr"][field])) == 0                # noqa: SLF001


# ------------------------------------------------------------ the value check

@pytest.mark.parametrize("value, dtype", [
    (np.float64(2.0 ** 63), np.int64), (np.float64(2.0 ** 64), np.uint64),
    (np.float64(-1.0), np.uint64), (np.float64(2.0 ** 31), np.int32),
    (np.int64(-1), np.uint64), (np.uint64(2 ** 63), np.int64), (np.int64(2 ** 53 + 1), np.int32),
    (np.int8(-1), np.uint8), (np.uint32(4_000_000_000), np.int32), (np.float64(0.5), np.int64),
    (np.float32(3e9), np.int32)])
def test_the_value_check_refuses_what_an_integer_type_cannot_hold(value, dtype):
    with pytest.raises(ValueError, match="does not fit its type"):
        _checked_value(value, dtype, what="x")


@pytest.mark.parametrize("value, dtype", [
    (np.int64(2 ** 53 + 1), np.int64), (np.uint64(2 ** 53 + 1), np.int64),
    (np.int64(2 ** 63 - 1), np.int64), (np.uint64(2 ** 64 - 1), np.uint64),
    (np.int64(2 ** 63 - 1), np.uint64), (np.float64(-(2.0 ** 63)), np.int64),
    (np.float64(2.0 ** 63 - 1024), np.int64), (np.float64(2.0 ** 64 - 2048), np.uint64),
    (np.float64(2.0 ** 53 + 2), np.int64), (np.float64(-0.0), np.uint8),
    (np.float32(127.0), np.int8), (np.int64(255), np.uint8)])
def test_the_value_check_keeps_an_integer_the_type_holds_exactly(value, dtype):
    """An int64 above 2**53 into an int64 leaf is itself, not its float64
    neighbour: the door ``FmuSidecar.set_params`` and an FMU-state archive
    come through carries the type's own bytes."""
    got = _checked_value(value, dtype, what="x")
    assert got.dtype == np.dtype(dtype) and int(got) == int(value)


# ---------------------------------------------------- the JSON literal "-0"

def test_the_json_literal_minus_zero_is_stored_as_negative_zero():
    """C's ``%.17g`` writes negative zero as ``-0``, which ``json.loads``
    reads as the integer 0: a JSON client's set of -0.0 was stored as
    +0.0.  The literal is negative zero as a value -- and still the
    integer 0 anywhere else in a request."""
    with _served() as (bridge, v):
        gauge = v["ctr.gauge"]
        with bridge:
            host, port = bridge.endpoint.split(":")
            with socket.create_connection((host, int(port)), timeout=30) as conn:
                send_message(conn, {"op": "hello"})
                assert recv_message(conn)["ok"]

                def raw(text: str) -> dict:
                    body = text.encode("utf-8")
                    conn.sendall(struct.pack(">I", len(body)) + body)
                    return recv_message(conn)

                vr = gauge.value_reference
                assert raw('{"op":"set","vr":[%d],"values":[1.5]}' % vr) == {"ok": True}
                for spelling in ("-0", "-0.0", "-0e0"):
                    assert raw('{"op":"set","vr":[%d],"values":[1.5]}' % vr) == {"ok": True}
                    assert raw('{"op":"set","vr":[%d],"values":[%s]}' % (vr, spelling)) == {"ok": True}
                    held = np.asarray(bridge._inputs["ctr"]["gauge"])            # noqa: SLF001
                    assert held == 0.0 and np.signbit(held), spelling
                    # and it goes back out with its sign
                    reply = raw('{"op":"get","vr":[%d]}' % vr)
                    assert reply["ok"] and np.signbit(np.float64(reply["values"][0]))
                assert raw('{"op":"set","vr":[%d],"values":[0]}' % vr) == {"ok": True}
                assert not np.signbit(np.asarray(bridge._inputs["ctr"]["gauge"]))   # noqa: SLF001
                # an integer variable has one zero
                add = v["ctr.add"].value_reference
                assert raw('{"op":"set","vr":[%d],"values":[-0]}' % add) == {"ok": True}
                assert int(np.asarray(bridge._inputs["ctr"]["add"])) == 0        # noqa: SLF001
                # elsewhere -0 is the integer it always was: a step at t = -0
                assert raw('{"op":"step","t":-0,"dt":%s}' % json.dumps(DT)) == {"ok": True, "t": DT}
                assert raw('{"op":"get","vr":[-0]}')["ok"] is False             # no variable 0
