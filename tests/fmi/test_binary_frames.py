"""Protocol 2 of the FMU sidecar bridge: binary frames for bulk payloads.

* a client that says ``{"op": "hello", "protocol": 2, "binary": true}``
  gets ``get`` / ``get_state`` replies as binary frames (raw little-endian
  float64 / raw npz bytes) and may send ``set`` / ``set_state`` the same
  way; everything else on that connection stays JSON;
* every malformed binary frame is an error reply and the connection
  (and the bridge) survives it; a binary frame before such a hello is
  refused; a flagged length over the limit drops only that connection;
* a JSON-only client (``{"op": "hello"}``) sees exactly the protocol-1
  behaviour, and an unknown higher protocol is refused at hello;
* a 10^6-element output goes over the wire at 8 bytes per value and
  several times faster than as JSON text (slow lane).
"""

import base64
import io
import json
import socket
import struct
import time

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import MODEL_IDENTIFIER, build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import (
    PROTOCOL_VERSION, FmuTcpBridge, decode_binary, encode_binary, recv_message, recv_raw,
    send_binary, send_message, state_of, values_of,
)
from maddening.nodes.heat import HeatNode
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr

_BINARY = 0x80000000


def _connect(bridge, *, binary=True, protocol=2, retries=40):
    """A client socket past its hello.  The bridge releases its one-instance
    lock when the previous client's EOF is served, so retry briefly."""
    host, port = bridge.endpoint.split(":")
    hello = {"op": "hello", "protocol": protocol, "binary": binary} if protocol else {"op": "hello"}
    for _ in range(retries):
        conn = socket.create_connection((host, int(port)), timeout=30)
        send_message(conn, hello)
        reply = recv_message(conn)
        if reply["ok"]:
            return conn, reply
        conn.close()
        assert "already serves" in reply["error"], reply
        time.sleep(0.05)
    raise AssertionError("bridge stayed busy")


def _f64(values):
    return np.asarray(values, dtype="<f8").tobytes()


# ------------------------------------------------------------- negotiation

def test_hello_advertises_protocol_and_negotiates_binary_per_request():
    gm = _graph()
    md, bridge = _bridge(gm)
    old = bridge.handle({"op": "hello"})
    assert old["ok"] and old["token"] == md.instantiation_token
    assert old["protocol"] == PROTOCOL_VERSION == 2 and old["binary"] is False
    new = bridge.handle({"op": "hello", "protocol": 2, "binary": True})
    assert new["binary"] is True and new["protocol"] == 2
    # protocol 1 with a binary wish, or protocol 2 without one: JSON
    assert bridge.handle({"op": "hello", "protocol": 1, "binary": True})["binary"] is False
    assert bridge.handle({"op": "hello", "protocol": 2})["binary"] is False
    assert bridge.handle({"op": "hello", "protocol": 2, "binary": "yes"})["binary"] is False


def test_unknown_higher_protocol_is_refused_at_hello():
    gm = _graph()
    md, bridge = _bridge(gm)
    r = bridge.handle({"op": "hello", "protocol": 3, "binary": True})
    assert not r["ok"] and "protocol 3" in r["error"] and "highest is 2" in r["error"]
    for bad in (0, -1, "2", 2.0, True, None):
        r = bridge.handle({"op": "hello", "protocol": bad})
        assert not r["ok"] and "protocol" in r["error"], (bad, r)
    with bridge:
        conn, _ = _connect(bridge, protocol=None)
        with conn:
            send_message(conn, {"op": "hello", "protocol": 99})
            r = recv_message(conn)
            assert not r["ok"] and "not supported" in r["error"]
            # a refused hello leaves the connection on JSON and alive
            send_message(conn, {"op": "get", "vr": [_vr(md, "time")]})
            assert recv_message(conn) == {"ok": True, "values": [0.0]}


# ------------------------------------------------------------ round trips

def test_binary_set_get_round_trip_matches_the_json_path():
    gm = _graph()
    md, bridge = _bridge(gm)
    k, anchor, pos, t = (_vr(md, n) for n in
                         ("spring.params.stiffness", "spring.anchor_position", "spring.position", "time"))
    with bridge:
        conn, hello = _connect(bridge)
        with conn:
            assert hello["binary"] is True
            send_binary(conn, {"op": "set", "vr": [k, anchor], "n": 2, "dtype": "f64"}, _f64([45.0, 0.25]))
            assert recv_message(conn) == {"ok": True}
            for i in range(5):
                send_message(conn, {"op": "step", "t": i * 3 * DT, "dt": 3 * DT})
                assert recv_message(conn)["ok"]
            send_message(conn, {"op": "get", "vr": [pos, k, t]})
            flag, body = recv_raw(conn)
            assert flag is True
            header, raw = decode_binary(body)
            assert header == {"ok": True, "n": 3, "dtype": "f64"} and len(raw) == 24
            got = np.frombuffer(raw, "<f8")
    assert bridge.binary_frames_received == 1 and bridge.binary_frames_served == 1
    # the same script through the JSON handler gives the same numbers
    md2, ref = _bridge(_graph())
    ref.handle({"op": "set", "vr": [k, anchor], "values": [45.0, 0.25]})
    for i in range(5):
        ref.handle({"op": "step", "t": i * 3 * DT, "dt": 3 * DT})
    want = ref.handle({"op": "get", "vr": [pos, k, t]})["values"]
    np.testing.assert_array_equal(got, np.asarray(want))
    assert got[1] == 45.0 and got[2] == pytest.approx(15 * DT)


def test_binary_state_round_trip_carries_raw_npz():
    gm = _graph()
    md, bridge = _bridge(gm)
    pos = _vr(md, "spring.position")
    with bridge:
        conn, _ = _connect(bridge)
        with conn:
            send_message(conn, {"op": "step", "t": 0.0, "dt": DT})
            recv_message(conn)
            send_message(conn, {"op": "get_state"})
            flag, body = recv_raw(conn)
            header, blob = decode_binary(body)
            assert flag and header == {"ok": True, "n": len(blob)}
            assert blob[:2] == b"PK"                                  # a zip, not base64
            with np.load(io.BytesIO(blob), allow_pickle=False) as data:
                assert str(data["_token"]) == md.instantiation_token
            send_message(conn, {"op": "get", "vr": [pos]})
            before = values_of(recv_message(conn))
            send_message(conn, {"op": "step", "t": DT, "dt": 5 * DT})
            recv_message(conn)
            send_message(conn, {"op": "get", "vr": [pos]})
            assert values_of(recv_message(conn)) != before
            send_binary(conn, {"op": "set_state", "n": len(blob)}, blob)
            assert recv_message(conn) == {"ok": True}
            send_message(conn, {"op": "get", "vr": [pos]})
            np.testing.assert_array_equal(values_of(recv_message(conn)), before)
            # the blob is validated exactly like the base64 form
            with np.load(io.BytesIO(blob), allow_pickle=False) as data:
                arrays = {k: data[k] for k in data.files}
            arrays["_token"] = np.array("other")
            buf = io.BytesIO()
            np.savez(buf, **arrays)
            bad = buf.getvalue()
            send_binary(conn, {"op": "set_state", "n": len(bad)}, bad)
            r = recv_message(conn)
            assert not r["ok"] and "token" in r["error"]
            send_message(conn, {"op": "get", "vr": [pos]})
            np.testing.assert_array_equal(values_of(recv_message(conn)), before)


def test_json_and_binary_requests_mix_on_one_connection():
    gm = _graph()
    md, bridge = _bridge(gm)
    anchor, el, pos = _vr(md, "spring.anchor_position"), _vr(md, "ball.params.elasticity"), \
        _vr(md, "spring.position")
    with bridge:
        conn, _ = _connect(bridge)
        with conn:
            send_message(conn, {"op": "set", "vr": [anchor], "values": [0.1]})       # JSON set
            assert recv_message(conn) == {"ok": True}
            send_binary(conn, {"op": "set", "vr": [anchor], "n": 1, "dtype": "f64"}, _f64([0.2]))
            assert recv_message(conn) == {"ok": True}
            send_message(conn, {"op": "get", "vr": [anchor]})                        # reply is binary
            r = recv_message(conn)
            assert "raw" in r and values_of(r).tolist() == [pytest.approx(0.2)]
            # validation is the same on both forms: atomic set, bounds, read-only
            send_binary(conn, {"op": "set", "vr": [anchor, el], "n": 2, "dtype": "f64"}, _f64([0.9, 1.5]))
            r = recv_message(conn)
            assert not r["ok"] and "above bound" in r["error"]
            send_message(conn, {"op": "get", "vr": [anchor]})
            assert values_of(recv_message(conn)).tolist() == [pytest.approx(0.2)]
            send_binary(conn, {"op": "set", "vr": [pos], "n": 1, "dtype": "f64"}, _f64([3.0]))
            assert "read-only" in recv_message(conn)["error"]
            send_binary(conn, {"op": "set", "vr": [anchor], "n": 1, "dtype": "f64"}, _f64([float("nan")]))
            assert "finite" in recv_message(conn)["error"]
            send_binary(conn, {"op": "set", "vr": [9999], "n": 1, "dtype": "f64"}, _f64([1.0]))
            assert "unknown value reference" in recv_message(conn)["error"]
            send_binary(conn, {"op": "set", "vr": [anchor], "n": 2, "dtype": "f64"}, _f64([1.0, 2.0]))
            assert "trailing" in recv_message(conn)["error"]
            # step / reset / terminate replies stay JSON on a binary connection
            send_message(conn, {"op": "step", "t": 0.0, "dt": DT})
            flag, body = recv_raw(conn)
            assert not flag and json.loads(body)["ok"]
            send_message(conn, {"op": "reset"})
            assert recv_message(conn) == {"ok": True}
            send_message(conn, {"op": "get", "vr": [anchor]})
            assert values_of(recv_message(conn)).tolist() == [0.0]


# --------------------------------------------------------- malformed frames

def _send_flagged(conn, payload: bytes, length=None):
    conn.sendall(struct.pack(">I", _BINARY | (len(payload) if length is None else length)) + payload)


def test_malformed_binary_frames_get_error_replies_and_the_bridge_survives():
    gm = _graph()
    md, bridge = _bridge(gm)
    anchor = _vr(md, "spring.anchor_position")
    good_hdr = json.dumps({"op": "set", "vr": [anchor], "n": 1, "dtype": "f64"}).encode()
    cases = {
        "header_len past payload": struct.pack(">I", 500) + good_hdr,
        "no header field": b"\x00\x00",
        "header not json": struct.pack(">I", 3) + b"{{{" + _f64([1.0]),
        "header not an object": struct.pack(">I", 3) + b"[1]" + _f64([1.0]),
        "n too large": encode_binary({"op": "set", "vr": [anchor], "n": 2, "dtype": "f64"}, _f64([1.0])),
        "raw longer than n": encode_binary({"op": "set", "vr": [anchor], "n": 1, "dtype": "f64"}, _f64([1.0, 2.0])),
        "n negative": encode_binary({"op": "set", "vr": [anchor], "n": -1, "dtype": "f64"}, b""),
        "n not an int": encode_binary({"op": "set", "vr": [anchor], "n": "1", "dtype": "f64"}, _f64([1.0])),
        "n is a bool": encode_binary({"op": "set", "vr": [anchor], "n": True, "dtype": "f64"}, _f64([1.0])),
        "bad dtype": encode_binary({"op": "set", "vr": [anchor], "n": 1, "dtype": "f32"}, _f64([1.0])),
        "no dtype": encode_binary({"op": "set", "vr": [anchor], "n": 1}, _f64([1.0])),
        "vr not a list": encode_binary({"op": "set", "vr": anchor, "n": 1, "dtype": "f64"}, _f64([1.0])),
        "op without binary form": encode_binary({"op": "step", "n": 0, "t": 0.0, "dt": DT}, b""),
        "get as binary": encode_binary({"op": "get", "vr": [anchor], "n": 0}, b""),
        "state length mismatch": encode_binary({"op": "set_state", "n": 3}, b"PK\x03\x04"),
        "state not an archive": encode_binary({"op": "set_state", "n": 4}, b"PK\x03\x04"),
        "utf-8 broken": struct.pack(">I", 2) + b"\xff\xfe" + b"",
    }
    with bridge:
        conn, _ = _connect(bridge)
        with conn:
            for name, payload in cases.items():
                _send_flagged(conn, payload)
                r = recv_message(conn)
                assert r is not None and r["ok"] is False and "raw" not in r, name
                assert "malformed" in r["error"] or "Error" in r["error"], (name, r)
            # the connection is intact and the input untouched
            send_message(conn, {"op": "get", "vr": [anchor]})
            assert values_of(recv_message(conn)).tolist() == [0.0]
            send_binary(conn, {"op": "set", "vr": [anchor], "n": 1, "dtype": "f64"}, _f64([0.5]))
            assert recv_message(conn) == {"ok": True}
        # counted: frames that were well formed at the frame level (the two
        # whose *request* then failed validation) plus the good one
        assert bridge.binary_frames_received == 3
        # a flagged length over the 64 MiB limit drops that connection only
        conn, _ = _connect(bridge)
        with conn:
            conn.sendall(struct.pack(">I", _BINARY | (64 * 1024 * 1024 + 1)))
            assert conn.recv(16) == b""
        conn, _ = _connect(bridge)
        with conn:
            send_message(conn, {"op": "get", "vr": [anchor]})
            assert values_of(recv_message(conn)).tolist() == [0.5]


def test_binary_frame_before_a_binary_hello_is_refused():
    gm = _graph()
    md, bridge = _bridge(gm)
    anchor = _vr(md, "spring.anchor_position")
    payload = encode_binary({"op": "set", "vr": [anchor], "n": 1, "dtype": "f64"}, _f64([0.5]))
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=10) as conn:
            _send_flagged(conn, payload)                                   # before any hello
            r = recv_message(conn)
            assert not r["ok"] and "hello" in r["error"]
            send_message(conn, {"op": "hello"})                            # a JSON-only hello
            assert recv_message(conn)["binary"] is False
            _send_flagged(conn, payload)
            r = recv_message(conn)
            assert not r["ok"] and "binary" in r["error"]
            send_message(conn, {"op": "get", "vr": [anchor]})
            assert recv_message(conn) == {"ok": True, "values": [0.0]}     # JSON, untouched
            send_message(conn, {"op": "hello", "protocol": 2, "binary": True})   # upgrade mid-way
            assert recv_message(conn)["binary"] is True
            _send_flagged(conn, payload)
            assert recv_message(conn) == {"ok": True}
    assert bridge.binary_frames_received == 1


# ---------------------------------------------------------- old clients

def test_json_only_client_sees_protocol_one_behaviour_unchanged():
    gm = _graph()
    md, bridge = _bridge(gm)
    k, pos = _vr(md, "spring.params.stiffness"), _vr(md, "spring.position")
    with bridge:
        conn, hello = _connect(bridge, protocol=None)
        with conn:
            assert set(hello) >= {"ok", "token", "model", "master_dt"} and hello["binary"] is False
            send_message(conn, {"op": "set", "vr": [k], "values": [45.0]})
            assert recv_message(conn) == {"ok": True}
            send_message(conn, {"op": "step", "t": 0.0, "dt": DT})
            assert recv_message(conn)["t"] == pytest.approx(DT)
            send_message(conn, {"op": "get", "vr": [pos, k]})
            flag, body = recv_raw(conn)
            assert not flag
            r = json.loads(body)
            assert r["ok"] and isinstance(r["values"], list) and r["values"][1] == 45.0
            send_message(conn, {"op": "get_state"})
            flag, body = recv_raw(conn)
            assert not flag
            snap = json.loads(body)["state"]
            assert isinstance(snap, str) and base64.b64decode(snap)[:2] == b"PK"
            send_message(conn, {"op": "step", "t": DT, "dt": DT})
            recv_message(conn)
            send_message(conn, {"op": "set_state", "state": snap})
            assert recv_message(conn) == {"ok": True}
            send_message(conn, {"op": "get", "vr": [pos]})
            assert recv_message(conn)["values"] == r["values"][:1]
    assert bridge.binary_frames_served == 0 and bridge.binary_frames_received == 0


def test_socketless_handle_keeps_the_json_forms():
    gm = _graph()
    md, bridge = _bridge(gm)
    pos = _vr(md, "spring.position")
    got = bridge.handle({"op": "get", "vr": [pos]})
    assert got == {"ok": True, "values": [0.5]} and type(got["values"][0]) is float
    snap = bridge.handle({"op": "get_state"})["state"]
    assert isinstance(snap, str)
    assert bridge.handle({"op": "set_state", "state": snap}) == {"ok": True}
    assert bridge.handle({"op": "set_state", "state": base64.b64decode(snap)}) == {"ok": True}
    assert state_of({"state": snap}) == base64.b64decode(snap)
    assert values_of({"values": [1, 2]}).dtype == np.float64
    # values of any JSON shape that is not a flat list are refused, not mis-read
    anchor = _vr(md, "spring.anchor_position")
    for bad in (5, "abc", [[0.5]], None, [[0.5], [0.5]]):
        r = bridge.handle({"op": "set", "vr": [anchor], "values": bad})
        assert not r["ok"], (bad, r)
    assert bridge.handle({"op": "set", "vr": [anchor], "values": [0.5]})["ok"]


# ---------------------------------------------------------------- benchmark

def _million_cell_bridge(n=1_000_000):
    gm = GraphManager()
    # length=n keeps dx at 1, so the Fourier number dt*alpha/dx^2 is 5e-4:
    # HeatNode refuses an explicit step above 0.5, and a million cells on
    # the default unit rod (dx = 1e-6) would be 5e8.  Nothing here steps
    # the rod; only the size of its temperature field matters.
    gm.add_node(HeatNode("h", 1e-3, n_cells=n, length=float(n),
                         thermal_diffusivity=0.5, initial_temperature=100.0))
    gm.compile()
    md = build_model_description(gm, model_name="Heat", model_identifier=MODEL_IDENTIFIER)
    # a realistic field (not a uniform 100.0, which JSON would print in 6 bytes)
    state = {k: dict(v) for k, v in gm._state.items()}
    live = np.asarray(state["h"]["temperature"])
    state["h"]["temperature"] = jnp.asarray(
        100.0 + np.random.default_rng(0).standard_normal(live.shape), dtype=live.dtype)
    sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm._compiled_step,
                                  initial_state=state, params=gm.params))
    return md, FmuTcpBridge(sc, md, master_dt=1e-3)


def _time_get(conn, vr, reps=3):
    best, nbytes, values = float("inf"), 0, None
    for _ in range(reps):
        t0 = time.perf_counter()
        send_message(conn, {"op": "get", "vr": [vr]})
        flag, body = recv_raw(conn)
        if flag:
            values = np.frombuffer(decode_binary(body)[1], "<f8")
        else:
            values = np.asarray(json.loads(body)["values"])
        best = min(best, time.perf_counter() - t0)
        nbytes = len(body)
    return best, nbytes, values


@pytest.mark.slow
def test_million_element_get_binary_is_faster_than_json():
    n = 1_000_000
    md, bridge = _million_cell_bridge(n)
    temp = _vr(md, "h.temperature")
    with bridge:
        conn, _ = _connect(bridge, protocol=None)
        with conn:
            t_json, b_json, v_json = _time_get(conn, temp)
        conn, _ = _connect(bridge)
        with conn:
            t_bin, b_bin, v_bin = _time_get(conn, temp)
    print(f"\nget of {n} float64 over loopback: JSON {t_json * 1e3:.1f} ms ({b_json / n:.1f} B/value), "
          f"binary {t_bin * 1e3:.1f} ms ({b_bin / n:.3f} B/value), speedup {t_json / t_bin:.1f}x")
    assert v_bin.shape == (n,) and np.array_equal(v_bin, v_json)
    assert 8.0 <= b_bin / n < 8.001                              # 8 bytes per value + a tiny header
    assert b_json / n > 8                                        # JSON text is wider
    assert t_json / t_bin >= 3.0, (t_json, t_bin)


# --------------------------------------------------------- limits and robustness

def test_bridge_never_sends_a_frame_over_the_limit(monkeypatch):
    """A ``get`` / ``get_state`` whose reply would exceed the frame limit
    (binary or JSON) gets a JSON error reply instead, and the connection
    stays in sync.  The C wrapper refuses to read a longer frame and
    drops the connection, so the bridge must never produce one.  The
    limit is lowered so the plant graph's few values can exceed it."""
    import maddening.fmi.tcp_bridge as tb
    gm = _graph()
    md, bridge = _bridge(gm)
    pos = _vr(md, "spring.position")
    monkeypatch.setattr(tb, "_MAX_MESSAGE", 512)
    with bridge:
        for binary in (True, False):
            conn, _ = _connect(bridge, protocol=2 if binary else None)
            with conn:
                send_message(conn, {"op": "get", "vr": [pos] * 200})          # 1600 B raw, 800 B JSON
                flag, body = recv_raw(conn)                                     # <= 512 or it raises
                assert not flag
                r = json.loads(body)
                assert not r["ok"] and "exceeds the 512-byte frame limit" in r["error"], r
                send_message(conn, {"op": "get", "vr": [pos]})                 # in sync, as negotiated
                r = recv_message(conn)
                assert ("raw" in r) is binary and values_of(r).tolist() == [0.5]
                send_message(conn, {"op": "get_state"})                        # the npz is > 512 B too
                r = recv_message(conn)
                assert not r["ok"] and "frame limit" in r["error"], r
                send_message(conn, {"op": "get", "vr": [pos]})
                assert values_of(recv_message(conn)).tolist() == [0.5]
    assert bridge.binary_frames_served == 2                # the two small gets of the binary round


def test_deeply_nested_header_is_an_error_reply_not_a_disconnect():
    """CPython's JSON scanner raises ``RecursionError`` on deep nesting;
    that must surface as ``ValueError`` from the codec and as an error
    reply (connection kept) on both the binary and the JSON path."""
    gm = _graph()
    md, bridge = _bridge(gm)
    anchor = _vr(md, "spring.anchor_position")
    deep = [b"[" * 100_000, b'{"a":' * 50_000, b"[" * 50_000]
    for text in deep:
        with pytest.raises(ValueError, match="nested too deeply"):
            decode_binary(struct.pack(">I", len(text)) + text)
    with bridge:
        conn, _ = _connect(bridge)
        with conn:
            for text in deep:
                _send_flagged(conn, struct.pack(">I", len(text)) + text)
                r = recv_message(conn)
                assert not r["ok"] and "nested too deeply" in r["error"], r
            send_message(conn, {"op": "get", "vr": [anchor]})
            assert values_of(recv_message(conn)).tolist() == [0.0]
        conn, _ = _connect(bridge, protocol=None)
        with conn:
            for text in deep:
                conn.sendall(struct.pack(">I", len(text)) + text)              # a JSON frame
                r = recv_message(conn)
                assert not r["ok"] and "nested too deeply" in r["error"], r
            send_message(conn, {"op": "get", "vr": [anchor]})
            assert recv_message(conn) == {"ok": True, "values": [0.0]}
    # the public reader keeps the same contract
    a, b = socket.socketpair()
    with a, b:
        a.sendall(struct.pack(">I", len(deep[0])) + deep[0])
        with pytest.raises(ValueError, match="nested too deeply"):
            recv_message(b)


def test_client_hangup_mid_reply_ends_the_connection_quietly():
    """A ``BrokenPipeError`` from the reply send (the importer died
    between request and reply) ends the connection loop like an EOF: no
    exception escapes ``_serve_conn`` and the instance lock is released."""
    gm = _graph()
    md, bridge = _bridge(gm)
    pos = _vr(md, "spring.position")
    client, server = socket.socketpair()
    send_message(client, {"op": "hello", "protocol": 2, "binary": True})
    send_message(client, {"op": "get", "vr": [pos]})
    send_message(client, {"op": "get_state"})
    client.close()                                   # every reply send now hits EPIPE
    bridge._serve_conn(server)                       # in this thread: returns, must not raise
    assert bridge.requests_served >= 1
    assert bridge._busy.acquire(blocking=False)      # released in the loop's finally
    bridge._busy.release()
    assert server.fileno() == -1                     # the loop closed its end


def test_handle_accepts_the_dict_recv_message_returns_for_a_binary_frame():
    """``recv_message`` yields the header plus ``"raw"`` for a binary frame;
    ``handle`` takes that dict and validates it exactly as the socket loop
    does (length against ``n``, dtype, op with a binary form)."""
    gm = _graph()
    md, bridge = _bridge(gm)
    anchor = _vr(md, "spring.anchor_position")
    a, b = socket.socketpair()
    with a, b:
        send_binary(a, {"op": "set", "vr": [anchor], "n": 1, "dtype": "f64"}, _f64([0.25]))
        req = recv_message(b)
        assert req["raw"] == _f64([0.25])
        assert bridge.handle(req) == {"ok": True}
        assert bridge.handle({"op": "get", "vr": [anchor]})["values"] == [0.25]
        for bad, needle in (
            ({**req, "n": 2}, "announces 2 float64 but carries 8"),
            ({**req, "dtype": "f32"}, "unsupported binary dtype"),
            ({**req, "raw": "0.25"}, "must be bytes"),
            ({**req, "op": "get"}, "no binary request form"),
        ):
            r = bridge.handle(bad)
            assert not r["ok"] and "malformed request" in r["error"] and needle in r["error"], (bad, r)
        assert bridge.handle({"op": "get", "vr": [anchor]})["values"] == [0.25]
        snap = state_of(bridge.handle({"op": "get_state"}))
        bridge.handle({"op": "set", "vr": [anchor], "values": [0.75]})
        send_binary(a, {"op": "set_state", "n": len(snap)}, snap)
        assert bridge.handle(recv_message(b)) == {"ok": True}
        assert bridge.handle({"op": "get", "vr": [anchor]})["values"] == [0.25]
