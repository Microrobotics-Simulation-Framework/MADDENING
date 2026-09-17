"""Property tests for the binary-frame codec and the bridge's handling of
arbitrary binary requests (protocol 2).

The importer is untrusted: every byte string that reaches the bridge over
a flagged frame must either decode to a consistent ``(header, raw)`` pair
or be refused with ``ValueError`` (never another exception), and the
connection loop must answer *every* such frame with an error reply and
keep serving.  The float64 path must be bitwise exact for every value the
type can hold, including NaN payloads, infinities, negative zero and
subnormals.
"""

import json
import socket
import struct
import threading

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st
from hypothesis.extra import numpy as hnp

from maddening.fmi.tcp_bridge import (
    decode_binary, encode_binary, recv_message, send_message, values_of,
)
from tests.fmi.test_c_wrapper import _bridge, _graph, _vr
from tests.conftest import EXAMPLES_CHEAP

_BINARY = 0x80000000
_HDR = struct.Struct(">I")

# ------------------------------------------------------------------ codec

f64_arrays = hnp.arrays(
    dtype=np.float64,
    shape=st.integers(min_value=0, max_value=4096),
    elements=st.floats(allow_nan=True, allow_infinity=True, allow_subnormal=True,
                       width=64),
)


@given(values=f64_arrays)
@settings(max_examples=EXAMPLES_CHEAP, deadline=None)
def test_float64_payload_round_trips_bitwise(values):
    """encode -> decode -> frombuffer is the identity on the bit pattern."""
    raw = np.ascontiguousarray(values, dtype="<f8").tobytes()
    payload = encode_binary({"ok": True, "n": int(values.size), "dtype": "f64"}, raw)
    header, raw_back = decode_binary(payload)
    assert header == {"ok": True, "n": int(values.size), "dtype": "f64"}
    assert raw_back == raw and len(raw_back) == 8 * values.size
    back = values_of({**header, "raw": raw_back})
    assert back.dtype == np.float64 and back.shape == values.shape
    assert np.array_equal(back.view(np.uint64), values.view(np.uint64))


# Headers nested deeper than CPython's JSON scanner can recurse: the
# scanner raises RecursionError, which the codec must turn into ValueError.
deep_headers = st.integers(min_value=1, max_value=50_000).flatmap(
    lambda d: st.sampled_from([b"[" * d, b'{"a":' * d, b"[" * d + b"]" * d])
).map(lambda text: _HDR.pack(len(text)) + text)


@given(payload=st.one_of(st.binary(min_size=0, max_size=2048), deep_headers))
@settings(max_examples=EXAMPLES_CHEAP, deadline=None)
def test_any_payload_decodes_consistently_or_raises_value_error(payload):
    """No byte string makes the decoder raise anything but ValueError
    (deep nesting included), and a successful decode is consistent with
    the bytes it came from."""
    try:
        header, raw = decode_binary(payload)
    except ValueError:
        return
    assert isinstance(header, dict) and isinstance(raw, bytes)
    (hlen,) = _HDR.unpack_from(payload)
    assert payload[_HDR.size + hlen:] == raw
    assert json.loads(payload[_HDR.size:_HDR.size + hlen]) == header


@given(header=st.dictionaries(st.text(max_size=8),
                              st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=False),
                                        st.text(max_size=16), st.lists(st.integers(), max_size=4)),
                              max_size=6),
       raw=st.binary(max_size=256))
@settings(max_examples=EXAMPLES_CHEAP, deadline=None)
def test_encode_then_decode_is_the_identity_for_any_json_header(header, raw):
    got_header, got_raw = decode_binary(encode_binary(header, raw))
    assert got_header == json.loads(json.dumps(header)) and got_raw == raw


# ------------------------------------------------- the bridge's connection loop

_shared = {}


def _bridge_and_vrs():
    """One bridge for the whole module (graph compile is the expensive part);
    built lazily so Hypothesis sees no function-scoped fixture."""
    if not _shared:
        gm = _graph()
        md, bridge = _bridge(gm)
        _shared["bridge"] = bridge
        _shared["anchor"] = _vr(md, "spring.anchor_position")
        _shared["stiffness"] = _vr(md, "spring.params.stiffness")
        _shared["position"] = _vr(md, "spring.position")
    return _shared["bridge"], _shared


def _serve_one(bridge):
    """A connected client socket whose peer is served by the real
    connection loop in a thread; hello negotiated for binary."""
    client, server = socket.socketpair()
    t = threading.Thread(target=bridge._serve_conn, args=(server,), daemon=True)
    t.start()
    send_message(client, {"op": "hello", "protocol": 2, "binary": True})
    reply = recv_message(client)
    assert reply["ok"] and reply["binary"] is True, reply
    return client, t


def _send_flagged(conn, payload):
    conn.sendall(_HDR.pack(_BINARY | len(payload)) + payload)


def _structured_requests():
    _, vrs = _bridge_and_vrs()
    known = [vrs["anchor"], vrs["stiffness"], vrs["position"]]
    vr_list = st.lists(st.one_of(st.sampled_from(known), st.integers(-5, 10 ** 6)), max_size=4)
    header = st.fixed_dictionaries(
        {},
        optional={
            "op": st.one_of(st.sampled_from(["set", "set_state", "get", "step", "hello", "reset", "nope"]),
                            st.integers(), st.none()),
            "n": st.one_of(st.integers(-3, 40), st.booleans(), st.text(max_size=3), st.floats(allow_nan=False)),
            "dtype": st.one_of(st.just("f64"), st.sampled_from(["f32", "i64", ""]), st.none()),
            "vr": st.one_of(vr_list, st.integers(), st.none()),
            "values": st.lists(st.floats(allow_nan=False), max_size=3),
            "t": st.floats(allow_nan=False), "dt": st.floats(allow_nan=False),
        },
    )
    raw = st.one_of(
        st.binary(max_size=64),
        st.integers(0, 8).map(lambda n: np.random.default_rng(n).standard_normal(n).astype("<f8").tobytes()),
    )
    return st.builds(encode_binary, header, raw)


payloads = st.one_of(st.binary(max_size=512), st.deferred(_structured_requests), deep_headers)


@pytest.fixture(scope="module")
def running_bridge():
    bridge, _ = _bridge_and_vrs()
    with bridge:
        yield bridge


@given(payload=payloads)
@settings(max_examples=EXAMPLES_CHEAP, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_every_binary_request_gets_a_reply_and_the_connection_survives(running_bridge, payload):
    """An arbitrary flagged frame after a binary hello yields exactly one
    reply (an error, or ``{"ok": true}`` for a request that happened to be
    valid), the same connection then serves a plain ``get`` correctly, and
    no exception escapes the loop (the serving thread ends only when the
    client hangs up)."""
    bridge, vrs = _bridge_and_vrs()
    client, thread = _serve_one(bridge)
    with client:
        client.settimeout(30)
        _send_flagged(client, payload)
        reply = recv_message(client)
        assert reply is not None and isinstance(reply, dict) and "ok" in reply
        if reply["ok"] is False:
            assert isinstance(reply["error"], str) and reply["error"]
        else:
            assert reply == {"ok": True}                   # a valid set / set_state
        send_message(client, {"op": "get", "vr": [vrs["stiffness"]]})
        got = recv_message(client)
        assert got["ok"] is True and values_of(got).shape == (1,) and np.isfinite(values_of(got)).all()
        assert thread.is_alive()
    thread.join(timeout=30)
    assert not thread.is_alive()
    bridge.handle({"op": "reset"})            # any accepted set must not leak into the next example


# --------------------------------------------------------- set: dtype narrowing

@given(value=st.floats(allow_nan=True, allow_infinity=True, allow_subnormal=True, width=64))
@settings(max_examples=EXAMPLES_CHEAP, deadline=None)
def test_set_of_any_float64_is_refused_or_reads_back_finite(value):
    """For any float64 the importer sends for a float32 input, either the
    set is refused (the input keeps its value) or the read-back is finite
    and within float32 rounding of what was sent: narrowing happens
    before the finiteness check, so 1e308 can no longer become inf."""
    bridge, vrs = _bridge_and_vrs()
    anchor = vrs["anchor"]
    bridge.handle({"op": "reset"})
    r = bridge.handle({"op": "set", "vr": [anchor], "values": [value]})
    (got,) = bridge.handle({"op": "get", "vr": [anchor]})["values"]
    assert np.isfinite(got)
    if r["ok"]:
        assert np.float32(value) == np.float32(got) and np.isfinite(np.float32(value))
    else:
        assert got == 0.0 and ("finite" in r["error"] or "does not fit" in r["error"]), r
