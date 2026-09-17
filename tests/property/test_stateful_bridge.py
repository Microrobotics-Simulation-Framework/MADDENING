"""An arbitrary sequence of sidecar frames keeps the bridge and a model in step.

:class:`~maddening.fmi.tcp_bridge.FmuTcpBridge` is driven over a socket by a
C wrapper an FMI importer loads, so everything that arrives is untrusted and
every reply is consumed positionally -- there is no request id, and a bridge
that answered one frame with two replies (or none) would leave the importer
reading the *previous* answer and calling it a success.  The existing tests
drive hand-written sequences; this module drives arbitrary ones.

A :class:`~hypothesis.stateful.RuleBasedStateMachine` holds one real
connection for the length of an example and interleaves ``hello`` (protocol
1 and 2, with and without binary), ``set``, ``get``, ``step``, ``get_state``,
``set_state``, ``reset``, ``terminate``, refusals of each, and malformed
frames of both kinds.  The reference is a second bridge over a second
sidecar on the same compiled step, driven in process with
:meth:`~maddening.fmi.tcp_bridge.FmuTcpBridge.handle`.  After every rule:

* every value readable over the wire equals what the in-process model
  produces for the same sequence -- which is also what catches a desync,
  since a reply that belongs to the previous frame is a reply with the wrong
  numbers in it;
* a state saved with ``get_state`` and restored later with ``set_state``
  reads back *exactly* the values captured at save time;
* a malformed frame is an error reply that changes nothing and leaves the
  connection serving;
* JSON and binary encodings of the same request are interchangeable within
  one connection.
"""

from __future__ import annotations

import base64
import json
import struct

import numpy as np
import pytest
from hypothesis import settings, strategies as st
from hypothesis.stateful import (
    RuleBasedStateMachine,
    invariant,
    precondition,
    rule,
    run_state_machine_as_test,
)

from maddening.fmi.tcp_bridge import (
    PROTOCOL_VERSION,
    encode_binary,
    recv_message,
    send_binary,
    send_message,
    state_of,
    values_of,
)

from tests.property.stateful_model import (
    DT,
    connect,
    f64_bytes,
    new_bridge,
    vr_of,
)

_HEADER = struct.Struct(">I")
_BINARY_FLAG = 0x80000000

WRITABLE: dict[str, tuple[float, float]] = {
    # input
    "spring.anchor_position": (-3.0, 3.0),
    # parameters; the ranges respect the ParamSpec bounds the sidecar
    # enforces and keep the explicit integrator stable at DT
    "table.params.position": (-5.0, 5.0),
    "spring.params.stiffness": (0.5, 100.0),
    "spring.params.damping": (0.0, 5.0),
    "spring.params.mass": (0.5, 10.0),
    "spring.params.rest_length": (0.0, 3.0),
    "spring.params.initial_position": (-2.0, 2.0),
    "spring.params.initial_velocity": (-2.0, 2.0),
}

READ_ONLY = ("time", "table.position", "spring.position", "spring.velocity")

ALL_VARIABLES = tuple(WRITABLE) + READ_ONLY

#: Frames whose payload is not a JSON object, so the bridge must answer
#: "malformed request" without dispatching anything.
NON_OBJECT_PAYLOADS = (
    b"", b"null", b"[]", b"123", b'"a string"', b"{", b'{"op":',
    b"\xff\xfe\x00", b"[" * 200, b'{"op": "set",}',
)


def _is_json_object(payload: bytes) -> bool:
    try:
        return isinstance(json.loads(payload.decode("utf-8")), dict)
    except Exception:                                               # noqa: BLE001
        return False


class FmuBridgeMachine(RuleBasedStateMachine):
    """Drive the sidecar bridge over TCP against an in-process model."""

    def __init__(self) -> None:
        super().__init__()
        self.md, self.bridge = new_bridge()
        self.bridge.start()
        _, self.model = new_bridge()
        try:
            self.conn, hello = connect(self.bridge, protocol=2, binary=True)
        except BaseException:
            # __init__ raising means teardown never runs; the bridge owns a
            # listening socket and a thread, and one leak per example adds up.
            self.bridge.stop()
            self.model.stop()
            raise
        assert hello["protocol"] == PROTOCOL_VERSION and hello["binary"] is True
        self.model.handle({"op": "hello", "protocol": 2, "binary": True})
        self.binary = True
        self.time = 0.0
        self.vr = {name: vr_of(self.md, name) for name in ALL_VARIABLES}
        self.all_vrs = [self.vr[n] for n in ALL_VARIABLES]
        self.snapshots: list[tuple[bytes, bytes, list[float]]] = []

    # ------------------------------------------------------------------
    # plumbing
    # ------------------------------------------------------------------

    def _round_trip(self, message: dict) -> dict:
        """One JSON frame out, one decoded reply in."""
        send_message(self.conn, message)
        reply = recv_message(self.conn)
        assert reply is not None, "the bridge closed the connection"
        return reply

    def _round_trip_binary(self, header: dict, raw: bytes) -> dict:
        send_binary(self.conn, header, raw)
        reply = recv_message(self.conn)
        assert reply is not None, "the bridge closed the connection"
        return reply

    def _raw_frame(self, payload: bytes, *, binary: bool = False) -> dict:
        word = (_BINARY_FLAG | len(payload)) if binary else len(payload)
        self.conn.sendall(_HEADER.pack(word) + payload)
        reply = recv_message(self.conn)
        assert reply is not None, "the bridge closed the connection"
        return reply

    def _refused(self, reply: dict) -> None:
        assert reply.get("ok") is False, reply
        assert isinstance(reply.get("error"), str) and reply["error"], reply

    def _wire_values(self, vrs: list[int]) -> np.ndarray:
        reply = self._round_trip({"op": "get", "vr": list(vrs)})
        assert reply.get("ok") is True, reply
        return values_of(reply)

    def _model_values(self, vrs: list[int]) -> np.ndarray:
        reply = self.model.handle({"op": "get", "vr": list(vrs)})
        assert reply.get("ok") is True, reply
        return np.asarray(reply["values"], dtype=np.float64)

    def _writable_vrs(self, names) -> list[int]:
        return [self.vr[n] for n in names]

    # ------------------------------------------------------------------
    # negotiation
    # ------------------------------------------------------------------

    @rule(protocol=st.sampled_from((1, 2)), want_binary=st.booleans())
    def say_hello_again(self, protocol, want_binary):
        """A second hello re-negotiates the encoding mid-connection."""
        reply = self._round_trip({"op": "hello", "protocol": protocol,
                                  "binary": want_binary})
        assert reply["ok"] is True, reply
        assert reply["protocol"] == PROTOCOL_VERSION
        assert reply["token"] == self.md.instantiation_token
        assert reply["master_dt"] == pytest.approx(DT)
        expected = protocol >= 2 and want_binary
        assert reply["binary"] is expected, reply
        self.binary = expected

    @rule(protocol=st.sampled_from((0, -1, 3, 99, "2", 2.0, True, None)))
    def rejected_hello(self, protocol):
        """A protocol this bridge does not speak leaves the encoding alone."""
        was_binary = self.binary
        self._refused(self._round_trip({"op": "hello", "protocol": protocol}))
        assert self.binary == was_binary

    # ------------------------------------------------------------------
    # set / get
    # ------------------------------------------------------------------

    @rule(data=st.data(), encoding=st.sampled_from(("json", "binary")))
    def set_values(self, data, encoding):
        names = data.draw(st.lists(st.sampled_from(sorted(WRITABLE)), min_size=1,
                                   max_size=len(WRITABLE), unique=True))
        values = [
            data.draw(st.floats(min_value=WRITABLE[n][0], max_value=WRITABLE[n][1],
                                allow_nan=False, allow_infinity=False,
                                allow_subnormal=False, width=32))
            for n in names
        ]
        vrs = self._writable_vrs(names)
        if encoding == "binary" and self.binary:
            reply = self._round_trip_binary(
                {"op": "set", "vr": vrs, "n": len(values), "dtype": "f64"},
                f64_bytes(values))
        else:
            reply = self._round_trip({"op": "set", "vr": vrs, "values": values})
        assert reply == {"ok": True}, reply
        assert self.model.handle({"op": "set", "vr": vrs, "values": values}) == {"ok": True}

    @rule(data=st.data())
    def get_values(self, data):
        names = data.draw(st.lists(st.sampled_from(sorted(ALL_VARIABLES)),
                                   max_size=len(ALL_VARIABLES)))
        vrs = [self.vr[n] for n in names]
        np.testing.assert_array_equal(self._wire_values(vrs), self._model_values(vrs))

    @rule(kind=st.sampled_from(("unknown_vr", "trailing_values", "short_values",
                                "non_finite", "read_only", "out_of_bounds",
                                "vr_not_a_list")),
          data=st.data())
    def rejected_set(self, kind, data):
        anchor = self.vr["spring.anchor_position"]
        request: dict = {"op": "set", "vr": [anchor], "values": [0.0]}
        if kind == "unknown_vr":
            request["vr"] = [10_000]
        elif kind == "trailing_values":
            request["values"] = [0.0, 1.0]
        elif kind == "short_values":
            request["vr"] = [anchor, self.vr["spring.params.stiffness"]]
        elif kind == "non_finite":
            request["values"] = [data.draw(st.sampled_from(
                (float("nan"), float("inf"), float("-inf"))))]
        elif kind == "read_only":
            request["vr"] = [self.vr[data.draw(st.sampled_from(READ_ONLY))]]
        elif kind == "out_of_bounds":
            request["vr"] = [self.vr["spring.params.stiffness"]]
            request["values"] = [-1.0]
        else:
            request["vr"] = anchor
        self._refused(self._round_trip(request))

    # ------------------------------------------------------------------
    # stepping
    # ------------------------------------------------------------------

    @rule(n=st.integers(min_value=1, max_value=3))
    def step(self, n):
        request = {"op": "step", "t": self.time, "dt": n * DT}
        reply = self._round_trip(request)
        assert reply["ok"] is True, reply
        model_reply = self.model.handle(dict(request))
        assert model_reply["ok"] is True, model_reply
        assert reply["t"] == model_reply["t"]
        self.time = float(reply["t"])

    @rule(kind=st.sampled_from(("zero", "negative", "nan", "infinite",
                                "not_a_multiple", "missing_dt", "bad_t")))
    def rejected_step(self, kind):
        request: dict = {"op": "step", "t": self.time, "dt": DT}
        if kind == "zero":
            request["dt"] = 0.0
        elif kind == "negative":
            request["dt"] = -DT
        elif kind == "nan":
            request["dt"] = float("nan")
        elif kind == "infinite":
            request["dt"] = float("inf")
        elif kind == "not_a_multiple":
            request["dt"] = 1.5 * DT
        elif kind == "missing_dt":
            request.pop("dt")
        else:
            request["t"] = "not a time"
        self._refused(self._round_trip(request))

    # ------------------------------------------------------------------
    # FMU state
    # ------------------------------------------------------------------

    @rule()
    def save_state(self):
        reply = self._round_trip({"op": "get_state"})
        assert reply.get("ok") is True, reply
        blob = state_of(reply)
        model_reply = self.model.handle({"op": "get_state"})
        assert model_reply.get("ok") is True, model_reply
        model_blob = base64.b64decode(model_reply["state"])
        captured = self._wire_values(self.all_vrs)
        np.testing.assert_array_equal(captured, self._model_values(self.all_vrs))
        self.snapshots.append((blob, model_blob, captured.tolist()))

    @precondition(lambda self: bool(self.snapshots))
    @rule(data=st.data())
    def restore_state(self, data):
        index = data.draw(st.integers(min_value=0, max_value=len(self.snapshots) - 1))
        blob, model_blob, captured = self.snapshots[index]
        encoding = data.draw(st.sampled_from(("json", "binary")))
        if encoding == "binary" and self.binary:
            reply = self._round_trip_binary({"op": "set_state", "n": len(blob)}, blob)
        else:
            reply = self._round_trip({
                "op": "set_state", "state": base64.b64encode(blob).decode("ascii")})
        assert reply == {"ok": True}, reply
        assert self.model.handle({"op": "set_state", "state": model_blob}) == {"ok": True}
        np.testing.assert_array_equal(self._wire_values(self.all_vrs),
                                      np.asarray(captured, dtype=np.float64))
        self.time = float(self._wire_values([self.vr["time"]])[0])

    @rule(kind=st.sampled_from(("not_an_archive", "empty", "truncated_npz",
                                "not_base64")))
    def rejected_set_state(self, kind):
        if kind == "not_an_archive":
            blob = b"PK\x03\x04 definitely not an npz"
        elif kind == "empty":
            blob = b""
        elif kind == "truncated_npz":
            source = self.snapshots[0][0] if self.snapshots else b"PK\x03\x04"
            blob = source[: max(1, len(source) // 2)]
        else:
            self._refused(self._round_trip({"op": "set_state", "state": "!!!not base64"}))
            return
        self._refused(self._round_trip({
            "op": "set_state", "state": base64.b64encode(blob).decode("ascii")}))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    @rule()
    def reset(self):
        assert self._round_trip({"op": "reset"}) == {"ok": True}
        assert self.model.handle({"op": "reset"}) == {"ok": True}
        self.time = 0.0

    @rule()
    def terminate(self):
        """``terminate`` is an acknowledgement, not a close: the FMI importer
        may still call fmi3FreeInstance (and the C wrapper still reads)."""
        assert self._round_trip({"op": "terminate"}) == {"ok": True}
        assert self.model.handle({"op": "terminate"}) == {"ok": True}

    @rule(op=st.sampled_from(("frobnicate", "", "HELLO", "get_stat", "set_stateX")))
    def unknown_op(self, op):
        reply = self._round_trip({"op": op})
        self._refused(reply)
        assert "unknown op" in reply["error"]

    # ------------------------------------------------------------------
    # malformed frames
    # ------------------------------------------------------------------

    @rule(payload=st.one_of(st.sampled_from(NON_OBJECT_PAYLOADS),
                            st.binary(max_size=96).filter(
                                lambda b: not _is_json_object(b))))
    def malformed_json_frame(self, payload):
        reply = self._raw_frame(payload)
        self._refused(reply)
        assert reply["error"].startswith("malformed request"), reply

    @rule(kind=st.sampled_from(("short_header", "header_longer_than_payload",
                                "header_not_an_object", "set_length_mismatch",
                                "set_wrong_dtype", "set_state_length_mismatch",
                                "op_without_a_binary_form", "negative_n")))
    def malformed_binary_frame(self, kind):
        anchor = self.vr["spring.anchor_position"]
        if kind == "short_header":
            payload = b"\x00\x00"
        elif kind == "header_longer_than_payload":
            payload = _HEADER.pack(4096) + b'{"op":"set"}'
        elif kind == "header_not_an_object":
            payload = encode_binary_list()
        elif kind == "set_length_mismatch":
            payload = encode_binary({"op": "set", "vr": [anchor], "n": 4,
                                     "dtype": "f64"}, f64_bytes([1.0]))
        elif kind == "set_wrong_dtype":
            payload = encode_binary({"op": "set", "vr": [anchor], "n": 1,
                                     "dtype": "f32"}, f64_bytes([1.0]))
        elif kind == "set_state_length_mismatch":
            payload = encode_binary({"op": "set_state", "n": 99}, b"short")
        elif kind == "op_without_a_binary_form":
            payload = encode_binary({"op": "step", "n": 0}, b"")
        else:
            payload = encode_binary({"op": "set", "vr": [anchor], "n": -1,
                                     "dtype": "f64"}, b"")
        reply = self._raw_frame(payload, binary=True)
        self._refused(reply)
        assert reply["error"].startswith("malformed request"), reply

    # ------------------------------------------------------------------
    # invariants
    # ------------------------------------------------------------------

    @invariant()
    def wire_values_match_the_model(self):
        np.testing.assert_array_equal(self._wire_values(self.all_vrs),
                                      self._model_values(self.all_vrs))

    def teardown(self) -> None:
        try:
            self.conn.close()
        finally:
            self.bridge.stop()
            self.model.stop()


def encode_binary_list() -> bytes:
    """A binary payload whose header is valid JSON but not an object."""
    header = b"[1, 2, 3]"
    return _HEADER.pack(len(header)) + header + b""


def test_arbitrary_sidecar_sequences_keep_the_bridge_and_the_model_in_step():
    """Any frame sequence: the wire agrees with the model, malformed or not.

    ``stateful_step_count`` is 25 rather than Hypothesis's default 50.  A
    rule here is a socket round trip and (for ``step``) a cached jitted
    graph step -- roughly a millisecond, an order of magnitude cheaper than
    the REST machine's -- but every rule is followed by the invariant's
    twelve-value ``get``, so 50 steps would double the run for coverage a
    second example buys more cheaply.  ``max_examples`` is left to the
    profile, per the house rule.
    """
    run_state_machine_as_test(
        FmuBridgeMachine,
        settings=settings(stateful_step_count=25),
    )


# ---------------------------------------------------------------------------
# Pinned regression sequences
# ---------------------------------------------------------------------------

def test_binary_frame_before_a_binary_hello_is_refused_and_the_connection_lives():
    """A JSON-only connection refuses a flagged frame without desyncing."""
    md, bridge = new_bridge()
    with bridge:
        conn, hello = connect(bridge, protocol=1, binary=True)
        with conn:
            assert hello["binary"] is False
            send_binary(conn, {"op": "set", "vr": [vr_of(md, "spring.anchor_position")],
                               "n": 1, "dtype": "f64"}, f64_bytes([1.0]))
            reply = recv_message(conn)
            assert reply["ok"] is False and "hello" in reply["error"]
            # still in sync: the next JSON request gets its own answer
            send_message(conn, {"op": "get", "vr": [vr_of(md, "time")]})
            assert recv_message(conn) == {"ok": True, "values": [0.0]}


def test_state_saved_then_restored_returns_the_captured_values():
    """get_state -> step -> set_state puts every value back, time included."""
    md, bridge = new_bridge()
    vrs = [vr_of(md, n) for n in ("time", "spring.position", "spring.velocity",
                                  "spring.anchor_position", "spring.params.stiffness")]
    with bridge:
        conn, _ = connect(bridge)
        with conn:
            send_message(conn, {"op": "set", "vr": [vrs[3], vrs[4]],
                                "values": [0.75, 42.0]})
            assert recv_message(conn) == {"ok": True}
            send_message(conn, {"op": "step", "t": 0.0, "dt": 2 * DT})
            assert recv_message(conn)["ok"]
            send_message(conn, {"op": "get_state"})
            blob = state_of(recv_message(conn))
            send_message(conn, {"op": "get", "vr": vrs})
            captured = values_of(recv_message(conn))
            for i in range(5):
                send_message(conn, {"op": "step", "t": (2 + i) * DT, "dt": DT})
                assert recv_message(conn)["ok"]
            send_binary(conn, {"op": "set_state", "n": len(blob)}, blob)
            assert recv_message(conn) == {"ok": True}
            send_message(conn, {"op": "get", "vr": vrs})
            np.testing.assert_array_equal(values_of(recv_message(conn)), captured)


def test_a_step_with_an_unusable_communication_point_advances_nothing():
    """Found by ``rejected_step(kind='bad_t')``.

    ``t`` was parsed *after* the sub-step loop, so a request whose ``t`` is
    not a number ran the physics, failed on the conversion and answered
    ``ok: false`` -- leaving the sidecar advanced, ``_time`` behind it, and
    the importer's clock permanently out of step with the state it reads.
    """
    md, bridge = new_bridge()
    vrs = [vr_of(md, n) for n in ("time", "spring.position", "spring.velocity")]
    with bridge:
        conn, _ = connect(bridge)
        with conn:
            send_message(conn, {"op": "get", "vr": vrs})
            before = values_of(recv_message(conn))
            for bad_t in ("not a time", None, [0.0], {"t": 0.0}, float("nan"),
                          float("inf")):
                send_message(conn, {"op": "step", "t": bad_t, "dt": DT})
                reply = recv_message(conn)
                assert reply["ok"] is False, (bad_t, reply)
                send_message(conn, {"op": "get", "vr": vrs})
                np.testing.assert_array_equal(values_of(recv_message(conn)), before,
                                              err_msg=f"t={bad_t!r} moved the state")
            # a good step still works afterwards
            send_message(conn, {"op": "step", "t": 0.0, "dt": DT})
            assert recv_message(conn)["t"] == pytest.approx(DT)


def test_a_malformed_frame_between_two_good_ones_changes_nothing():
    """The frame that used to desync a client is an error reply in place."""
    md, bridge = new_bridge()
    time_vr = vr_of(md, "time")
    with bridge:
        conn, _ = connect(bridge)
        with conn:
            send_message(conn, {"op": "step", "t": 0.0, "dt": DT})
            assert recv_message(conn)["ok"]
            send_message(conn, {"op": "get", "vr": [time_vr]})
            before = values_of(recv_message(conn))
            for payload in NON_OBJECT_PAYLOADS:
                conn.sendall(_HEADER.pack(len(payload)) + payload)
                reply = recv_message(conn)
                assert reply["ok"] is False
                assert reply["error"].startswith("malformed request"), reply
            send_message(conn, {"op": "get", "vr": [time_vr]})
            np.testing.assert_array_equal(values_of(recv_message(conn)), before)
