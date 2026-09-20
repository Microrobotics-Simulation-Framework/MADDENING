"""The FMI JSON wire carries a non-finite value as valid JSON.

``MADD-ANO-006``: a ``get`` reply carrying a non-finite state value --
which a diverged model produces -- used to go out as the bare token
``Infinity`` or ``NaN``.  Python read it back and the C wrapper's
``strtod`` parsed it, so the shipped stack round-tripped it exactly; the
module docstring calls the protocol JSON, and no conforming parser
accepts those tokens.  The wire now writes the quoted token.

The C wrapper reads both spellings.  ``strtod`` is C99 and parses ``nan``,
``inf`` and ``infinity`` case-insensitively; the quote was the only thing
it could not step over, and ``parse_values`` now steps over it (pinned in
``tests/fmi/c/test_maddening_fmu.c``, and end to end against the compiled
binary below).

The binary frames of protocol 2 carry raw float64 and never had the
problem; that is still asserted here, because the JSON path must not be
the only one that works.
"""

import dataclasses
import json
import logging
import math
import os
import socket
import struct
import threading

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.fmi import build_model_description
from maddening.fmi.package import (
    MODEL_IDENTIFIER, build_fmu_binary, find_c_compiler, write_fmu,
)
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import (
    FmuTcpBridge, decode_binary, recv_message, recv_raw, send_message, values_of,
)
from maddening.serialization.json_codec import dumps as json_codec_dumps
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr

needs_cc = pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")


def _refuse(token):
    """A strict reader: RFC 8259 has no literal for these."""
    raise ValueError(f"non-standard JSON token {token!r}")


def _strict_loads(text):
    return json.loads(text, parse_constant=_refuse)


def _diverged_bridge(bridge_cls=FmuTcpBridge):
    """A bridge whose state holds ``inf`` and ``NaN``.

    Written straight into the sidecar rather than through ``set`` or
    ``set_state``: both of those refuse a non-finite value, by design.  A
    model that diverges on its own arrives here without asking anyone,
    which is the case this whole file is about.
    """
    gm = _graph()
    md = build_model_description(gm, model_name="Plant",
                                 model_identifier=MODEL_IDENTIFIER)
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
    ))
    bridge = bridge_cls(sidecar, md, master_dt=DT)
    state = {n: dict(f) for n, f in bridge._sidecar.state.items()}
    state["spring"]["position"] = jnp.asarray(np.inf, dtype=jnp.float32)
    state["spring"]["velocity"] = jnp.asarray(np.nan, dtype=jnp.float32)
    bridge._sidecar._state = state
    return md, bridge


def _get_reply(bridge, md, names, hello):
    """One ``get`` over a real connection; returns ``(is_binary, body)``."""
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=15) as sock:
            sock.settimeout(15)
            send_message(sock, hello)
            hello_reply = recv_raw(sock)[1]
            assert hello_reply
            send_message(sock, {"op": "get", "vr": [_vr(md, n) for n in names]})
            return recv_raw(sock)


# --------------------------------------------------------------- the wire

def test_a_non_finite_get_reply_is_strict_json():
    md, bridge = _diverged_bridge()
    is_binary, body = _get_reply(
        bridge, md, ("spring.position", "spring.velocity"), {"op": "hello"})

    assert not is_binary
    text = body.decode("utf-8")
    reply = _strict_loads(text)               # a conforming reader accepts it
    assert reply["ok"] is True
    assert reply["values"] == ["Infinity", "NaN"]

    values = values_of(reply)
    assert math.isinf(values[0]) and values[0] > 0
    assert math.isnan(values[1])


def test_the_binary_frames_carry_the_same_values_exactly():
    """Protocol 2 has no encoding to argue about, and still does not."""
    md, bridge = _diverged_bridge()
    is_binary, body = _get_reply(
        bridge, md, ("spring.position", "spring.velocity"),
        {"op": "hello", "protocol": 2, "binary": True})

    assert is_binary
    _header, raw = decode_binary(body)
    values = values_of({"raw": raw})
    assert math.isinf(values[0]) and values[0] > 0
    assert math.isnan(values[1])


def test_a_finite_get_reply_is_strict_json():
    """The change is confined to non-finite values, not to the wire."""
    md, bridge = _bridge(_graph())
    _is_binary, body = _get_reply(bridge, md, ("spring.position",), {"op": "hello"})
    assert _strict_loads(body.decode("utf-8"))["ok"] is True


def test_send_message_writes_a_non_finite_payload_as_strict_json():
    a, b = socket.socketpair()
    with a, b:
        send_message(a, {"op": "note", "values": [math.inf, -math.inf, math.nan, 1.0]})
        _is_binary, body = recv_raw(b)
    text = body.decode("utf-8")
    assert _strict_loads(text)["values"] == ["Infinity", "-Infinity", "NaN", 1.0]


def test_a_frame_from_a_pre_040_peer_still_decodes():
    """Backward compatibility, pinned on a literal frame with bare tokens.

    ``json.loads`` parses them itself, so nothing on the reading side has
    to recognise the old spelling -- but nothing may reject it either.
    """
    body = b'{"ok":true,"values":[NaN,Infinity,-Infinity,1.0]}'
    a, b = socket.socketpair()
    with a, b:
        a.sendall(struct.pack(">I", len(body)) + body)
        message = recv_message(b)

    assert message["ok"] is True
    values = values_of(message)
    assert math.isnan(values[0])
    assert values[1] == math.inf and values[2] == -math.inf and values[3] == 1.0


# ----------------------------------------------------------- the property

@given(st.lists(st.floats(allow_nan=True, allow_infinity=True),
                min_size=0, max_size=8))
@settings(deadline=None)
def test_any_values_array_survives_the_json_wire_exactly(values):
    """Every reply a diverged model can produce is valid JSON and exact.

    The empty list is in range on purpose: a ``get`` of no value
    references is legal and must not become ``null`` or an error.
    """
    a, b = socket.socketpair()
    with a, b:
        send_message(a, {"ok": True, "values": values})
        _is_binary, body = recv_raw(b)

    text = body.decode("utf-8")
    reply = _strict_loads(text)               # no bare token survived
    back = values_of(reply)

    assert len(back) == len(values)
    for sent, got in zip(values, back):
        if math.isnan(sent):
            assert math.isnan(got)
        else:
            assert got == sent


# ------------------------------------------------- through the real binary

class _JsonOnlyBridge(FmuTcpBridge):
    """A bridge that declines binary frames, so the wrapper stays on JSON.

    The C wrapper offers protocol 2 at hello and uses binary frames when
    the bridge agrees, which would route a diverged ``get`` around the
    encoding entirely.  Declining is documented protocol-1 behaviour --
    "a bridge whose hello reply lacks protocol is protocol 1" -- so this
    exercises the shipped wrapper on the path under test rather than a
    modified one.
    """

    def _dispatch(self, req: dict) -> dict:
        reply = super()._dispatch(req)
        if req.get("op") == "hello" and reply.get("ok"):
            reply = {**reply, "binary": False}
        return reply


@needs_cc
def test_the_compiled_wrapper_reads_a_diverged_get_over_json(tmp_path):
    """End to end: the shipped binary, the JSON path, a diverged model.

    Before the fix the bridge wrote bare tokens and ``strtod`` read them,
    so this passed for the wrong reason.  Quoting them without teaching
    ``parse_values`` about the quote would make it fail with "malformed
    number in reply" -- which is what makes it worth having.
    """
    pytest.importorskip("fmpy")
    from fmpy import extract, read_model_description
    from fmpy.fmi3 import FMU3Slave

    md, bridge = _diverged_bridge(_JsonOnlyBridge)
    so = build_fmu_binary(tmp_path)
    with bridge:
        fmu = write_fmu(md, tmp_path / "diverged.fmu", binary=so,
                        endpoint=bridge.endpoint)
        unz = extract(str(fmu))
        desc = read_model_description(unz)
        inst = FMU3Slave(guid=desc.guid, unzipDirectory=unz,
                         modelIdentifier=desc.coSimulation.modelIdentifier,
                         instanceName="i")
        inst.instantiate(loggingOn=True)
        try:
            inst.enterInitializationMode(startTime=0.0)
            inst.exitInitializationMode()
            vr = {v.name: v.valueReference for v in desc.modelVariables}
            got = inst.getFloat64([vr["spring.position"], vr["spring.velocity"]])
        finally:
            inst.terminate()
            inst.freeInstance()

    assert bridge.binary_frames_served == 0, "the JSON path was not exercised"
    assert math.isinf(got[0]) and got[0] > 0
    assert math.isnan(got[1])


# ------------------------------- the model name, and replies that cannot be written
#
# ``MADD-ANO-010`` was recorded saying that "no request or reply
# ``FmuTcpBridge`` itself builds can reach [the refusal], because the
# protocol has no free-text field".  The ``hello`` reply carries
# ``"model": self._md.model_name`` -- the caller's unvalidated string --
# so an FMU named exactly ``NaN``, ``Infinity`` or ``-Infinity`` made
# ``_send_reply`` raise on the *first* frame of every connection, and the
# ``ValueError`` ended the worker thread: the importer got EOF instead of
# the error reply the protocol promises, and the only record was a
# traceback on the bridge's stderr.  Both halves of the repair are pinned
# here -- the build-time refusal, and the handler that makes *any*
# unwritable reply an error reply rather than a dead connection.


def _hand_built_description(gm, model_name):
    """A description whose ``model_name`` bypasses the build-time refusal.

    ``ModelDescription`` is a public dataclass, so a caller can assemble
    one itself; the bridge must hold whatever it is given.
    """
    md = build_model_description(gm, model_name="Plant",
                                 model_identifier=MODEL_IDENTIFIER)
    return dataclasses.replace(md, model_name=model_name)


def _bridge_for(gm, md, bridge_cls=FmuTcpBridge):
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
    ))
    return bridge_cls(sidecar, md, master_dt=DT)


@pytest.fixture(scope="module")
def compiled_graph():
    return _graph()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_a_model_name_spelling_a_non_finite_token_is_refused_at_build_time(
        compiled_graph, token):
    """Refused where the caller still holds the name and can change it.

    By the time the name reaches the wire the caller is the FMU's C
    wrapper on the far end of a socket, which can do nothing about it.
    """
    with pytest.raises(ValueError, match="non-finite JSON token"):
        build_model_description(compiled_graph, model_name=token,
                                model_identifier=MODEL_IDENTIFIER)


@pytest.mark.parametrize("name", ["nan", "infinity", "-infinity", "NaN ", "+Infinity",
                                  "Plant", "NaNo", "Infinity2"])
def test_a_model_name_that_only_looks_like_a_token_is_accepted(compiled_graph, name):
    """The refusal is an exact match, exactly like the codec's own.

    Widening it to a case-insensitive or substring match would refuse
    names the wire carries perfectly well, and would not match the
    decoder it exists to protect.
    """
    md = build_model_description(compiled_graph, model_name=name,
                                 model_identifier=MODEL_IDENTIFIER)
    assert md.model_name == name
    assert _strict_loads(json_codec_dumps({"model": md.model_name}))["model"] == name


def test_a_reply_that_cannot_be_encoded_is_an_error_reply_not_a_dead_worker(
        compiled_graph, caplog):
    """The connection survives a reply the codec refuses to write.

    A hand-built description gets the token-spelled name past
    ``build_model_description``.  What must happen then: the promised
    ``{"ok": false, "error": ...}``, a connection that still serves the
    next request, no exception out of the worker thread, and a log
    record -- the failure is reported, never swallowed.
    """
    md = _hand_built_description(compiled_graph, "Infinity")
    bridge = _bridge_for(compiled_graph, md)

    deaths = []
    previous_hook = threading.excepthook
    threading.excepthook = deaths.append
    try:
        with caplog.at_level(logging.ERROR, logger="maddening.fmi.tcp_bridge"):
            with bridge:
                host, port = bridge.endpoint.rsplit(":", 1)
                with socket.create_connection((host, int(port)), timeout=15) as sock:
                    sock.settimeout(15)
                    send_message(sock, {"op": "hello", "protocol": 2, "binary": True})
                    got = recv_raw(sock)
                    assert got is not None, "the worker died instead of replying"
                    is_binary, body = got
                    assert is_binary is False
                    reply = _strict_loads(body.decode("utf-8"))
                    assert reply["ok"] is False
                    assert "could not encode its reply" in reply["error"]
                    assert "$.model" in reply["error"], \
                        "the reply must still name what could not be written"

                    # The connection is still in sync and still serving.
                    send_message(sock, {"op": "get_state"})
                    second = recv_raw(sock)
                    assert second is not None
                    # ... and still speaking JSON: a hello whose reply never
                    # reached the peer must not switch this end to binary
                    # frames the peer does not know were agreed.
                    assert second[0] is False
                    assert _strict_loads(second[1].decode("utf-8"))["ok"] is True
    finally:
        threading.excepthook = previous_hook

    assert deaths == [], f"the worker thread died: {deaths}"
    assert bridge.replies_unencodable == 1
    assert any("could not encode the reply" in r.getMessage() for r in caplog.records)


class _UnsendableBridge(FmuTcpBridge):
    """Every reply fails to be framed, whatever the request was."""

    def _send_reply(self, conn, reply, binary):
        raise TypeError("this reply cannot be framed")


def test_a_reply_failure_on_an_unparsed_first_frame_is_still_answered(compiled_graph):
    """The handler runs even when no request was ever parsed.

    The first frame here is not JSON, so the request never becomes a
    dict; the reply-failure path must not assume one exists.  Without
    the guard this raised ``NameError`` *inside* the handler and cost
    the worker thread all over again.
    """
    md = build_model_description(compiled_graph, model_name="Plant",
                                 model_identifier=MODEL_IDENTIFIER)
    bridge = _bridge_for(compiled_graph, md, _UnsendableBridge)

    deaths = []
    previous_hook = threading.excepthook
    threading.excepthook = deaths.append
    try:
        with bridge:
            host, port = bridge.endpoint.rsplit(":", 1)
            with socket.create_connection((host, int(port)), timeout=15) as sock:
                sock.settimeout(15)
                body = b"not json at all"
                sock.sendall(struct.pack(">I", len(body)) + body)
                got = recv_raw(sock)
                assert got is not None, "the worker died instead of replying"
                reply = _strict_loads(got[1].decode("utf-8"))
                assert reply["ok"] is False
                assert "could not encode its reply" in reply["error"]
    finally:
        threading.excepthook = previous_hook

    assert deaths == [], f"the worker thread died: {deaths}"


@pytest.mark.parametrize("exc", [
    ValueError("NaN"),                      # the collision itself, as a message
    ValueError("Infinity"),
    ValueError("ключ \u0001 " + "x" * 5000),  # non-ASCII, control, far over the cap
    ValueError(""),
])
def test_the_last_resort_error_reply_is_always_itself_sendable(exc):
    """The reply that reports an unsendable reply must never be unsendable.

    Prefixed (so it can never equal a token and hit the same refusal),
    printable ASCII (the C wrapper hands it to the importer's log
    callback as a C string) and capped far below the frame limit.
    """
    reply = FmuTcpBridge._unencodable_reply(exc)

    assert reply["ok"] is False
    text = reply["error"]
    assert text.startswith("the bridge could not encode its reply: ")
    assert text.isascii(), text
    assert text.isprintable(), text
    assert len(text) <= 440
    assert _strict_loads(json_codec_dumps(reply))["error"] == text


@needs_cc
def test_the_importer_is_told_which_string_broke_the_hello(tmp_path, compiled_graph):
    """End to end, through the shipped binary: the importer learns why.

    The C wrapper sends ``hello`` from inside
    ``fmi3InstantiateCoSimulation``, so before the repair the importer
    got a closed socket and a bare "cannot instantiate": every trace of
    *why* was a traceback on the bridge's stderr, in another process.
    The bridge's error reply now reaches the wrapper, which pulls it out
    of the frame and hands it to the importer's log callback.
    """
    pytest.importorskip("fmpy")
    from fmpy import extract, read_model_description
    from fmpy.fmi3 import FMU3Slave

    md = _hand_built_description(compiled_graph, "Infinity")
    bridge = _bridge_for(compiled_graph, md)
    so = build_fmu_binary(tmp_path)
    logged = []

    with bridge:
        fmu = write_fmu(md, tmp_path / "badname.fmu", binary=so,
                        endpoint=bridge.endpoint)
        unz = extract(str(fmu))
        desc = read_model_description(unz)
        inst = FMU3Slave(guid=desc.guid, unzipDirectory=unz,
                         modelIdentifier=desc.coSimulation.modelIdentifier,
                         instanceName="i")
        with pytest.raises(Exception):          # noqa: B017 - fmpy's own wording
            inst.instantiate(
                loggingOn=True,
                logMessage=lambda _env, _status, _cat, message: logged.append(
                    message.decode("utf-8", "replace")),
            )

    assert bridge.replies_unencodable == 1
    assert any("could not encode its reply" in m for m in logged), logged
    assert any("Infinity" in m for m in logged), logged
