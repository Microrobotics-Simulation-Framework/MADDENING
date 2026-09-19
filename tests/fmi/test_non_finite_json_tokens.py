"""The FMI JSON wire writes a non-finite value as a non-standard token.

Pins the bridge half of ``MADD-ANO-006``.  A ``get`` reply carrying a
non-finite state value — which a diverged model produces — goes out as
the bare token ``Infinity`` or ``NaN``.  Python reads it back and the C
wrapper parses it with ``strtod``, so the shipped stack round-trips it
exactly; a conforming JSON reader rejects the frame, and the module
docstring calls the protocol JSON.

The anomaly is **open**, and on this surface deliberately so: JSON has no
literal for NaN, and a quoted or tagged form would break the C wrapper,
which reads values with ``strtod``.  Choosing an encoding for a shipped
protocol is a maintainer's decision.  Until then this test states the
behaviour the registry documents, so that changing it fails here and the
registry entry is updated with it.

The binary frames of protocol 2 carry raw float64 and have no such
problem; that is the documented workaround, and it is asserted here too.
"""

import json
import math
import os
import socket

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.fmi.tcp_bridge import (
    recv_raw, send_message, values_of,
)
from tests.fmi.test_c_wrapper import _bridge, _graph, _vr


def _refuse(token):
    """A strict reader: RFC 8259 has no literal for these."""
    raise ValueError(f"non-standard JSON token {token!r}")


def _diverged_bridge():
    """A bridge whose state holds ``inf`` and ``NaN``.

    Written straight into the sidecar rather than through ``set`` or
    ``set_state``: both of those refuse a non-finite value, which is the
    point of the rest of this branch.  A model that diverges on its own
    arrives here without asking anyone.
    """
    md, bridge = _bridge(_graph())
    state = {n: dict(f) for n, f in bridge._sidecar.state.items()}
    state["spring"]["position"] = jnp.asarray(np.inf, dtype=jnp.float32)
    state["spring"]["velocity"] = jnp.asarray(np.nan, dtype=jnp.float32)
    bridge._sidecar._state = state
    return md, bridge


def test_a_non_finite_get_reply_is_not_standard_json():
    md, bridge = _diverged_bridge()
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=15) as sock:
            sock.settimeout(15)
            send_message(sock, {"op": "hello"})
            assert recv_raw(sock)[1]
            send_message(sock, {"op": "get", "vr": [_vr(md, "spring.position"),
                                                    _vr(md, "spring.velocity")]})
            is_binary, body = recv_raw(sock)

    assert not is_binary
    text = body.decode("utf-8")
    assert "Infinity" in text and "NaN" in text, (
        "MADD-ANO-006 says the JSON wire emits the bare tokens; if that "
        "changed, update the registry entry"
    )
    # Python reads its own frame back...
    assert json.loads(text)["ok"] is True
    # ...and a conforming reader does not.
    with pytest.raises(ValueError):
        json.loads(text, parse_constant=_refuse)


def test_the_binary_frames_carry_the_same_values_exactly():
    """The documented workaround: protocol 2 has no encoding to argue about."""
    md, bridge = _diverged_bridge()
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=15) as sock:
            sock.settimeout(15)
            send_message(sock, {"op": "hello", "protocol": 2, "binary": True})
            assert json.loads(recv_raw(sock)[1])["binary"] is True
            send_message(sock, {"op": "get", "vr": [_vr(md, "spring.position"),
                                                    _vr(md, "spring.velocity")]})
            is_binary, body = recv_raw(sock)

    assert is_binary
    from maddening.fmi.tcp_bridge import decode_binary
    header, raw = decode_binary(body)
    values = values_of({"raw": raw})
    assert math.isinf(values[0]) and values[0] > 0
    assert math.isnan(values[1])


def test_a_finite_get_reply_is_strict_json():
    """The anomaly is confined to non-finite values, not to the wire."""
    md, bridge = _bridge(_graph())
    with bridge:
        host, port = bridge.endpoint.split(":")
        with socket.create_connection((host, int(port)), timeout=15) as sock:
            sock.settimeout(15)
            send_message(sock, {"op": "hello"})
            assert recv_raw(sock)[1]
            send_message(sock, {"op": "get", "vr": [_vr(md, "spring.position")]})
            _, body = recv_raw(sock)

    assert json.loads(body.decode("utf-8"), parse_constant=_refuse)["ok"] is True
