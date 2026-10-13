"""An FMU-state archive may not install a value ``set`` would refuse.

``fmi3SetFMUState`` takes a blob the importer may have built itself (with
``fmi3DeserializeFMUState``, from bytes of its own), so the archive is a
second door into the same state and parameter tree that ``set`` writes.
Until 0.4.0 that door had no value checks at all: it validated the schema
token, the member names and the shapes, then wrote straight through.  The
same bridge that refused ``set mass=-1.0`` against declared bounds
``(0.1, 10.0)`` accepted an archive that set it, answered ``ok``, and one
``step`` later the whole state was NaN.

Everything here goes over a real loopback socket, the way an importer
reaches the bridge.

Written from the independent audit of 2026-09-19 (``params-io``;
reproducer ``r17_setstate_bypass.py`` under
``benchmarks/results/audit_040_final/params-io/``).
"""

import base64
import io
import math
import os
import socket

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from maddening.fmi.tcp_bridge import recv_message, send_message, state_of
from tests.conftest import EXAMPLES_COSTLY
from tests.fmi.test_c_wrapper import _bridge, _graph, _vr

_GRAPH = None


def _shared_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _graph()
    return _GRAPH


class _Client:
    """One importer: a started bridge and a connected socket."""

    def __init__(self):
        self.md, self.bridge = _bridge(_shared_graph())
        self.bridge.start()
        host, port = self.bridge.endpoint.split(":")
        self.sock = socket.create_connection((host, int(port)), timeout=10)
        self.sock.settimeout(10)
        assert self.call({"op": "hello"})["ok"]

    def call(self, request):
        send_message(self.sock, request)
        return recv_message(self.sock)

    def vr(self, name):
        return _vr(self.md, name)

    def archive(self):
        """The current state as the npz an importer would hand back."""
        return state_of(self.call({"op": "get_state"}))

    def close(self):
        try:
            self.sock.close()
        finally:
            self.bridge.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _rewrite(blob: bytes, member: str, value: np.ndarray) -> bytes:
    """``blob`` with one member replaced -- an importer's own archive."""
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        members = {k: data[k] for k in data.files}
    assert member in members, sorted(members)
    members[member] = value
    buf = io.BytesIO()
    np.savez(buf, **members)
    return buf.getvalue()


def _set_state(client, member, value):
    blob = _rewrite(client.archive(), member, value)
    return client.call({"op": "set_state",
                        "state": base64.b64encode(blob).decode("ascii")})


# ------------------------------------------------------------- unit tests

def test_an_archive_cannot_set_a_parameter_outside_its_declared_bounds():
    with _Client() as c:
        refused = c.call({"op": "set", "vr": [c.vr("ball.params.elasticity")],
                          "values": [1.5]})
        assert not refused["ok"] and "above bound" in refused["error"]

        reply = _set_state(c, "p/nodes/ball/elasticity",
                           np.array(1.5, dtype=np.float32))
        assert not reply["ok"], "the archive walked past the declared bounds"
        assert "elasticity" in reply["error"]
        # and the live value is untouched
        got = c.call({"op": "get", "vr": [c.vr("ball.params.elasticity")]})
        assert got["values"][0] == pytest.approx(0.7)


def test_an_archive_cannot_install_a_non_finite_state_field():
    with _Client() as c:
        reply = _set_state(c, "s/spring/velocity", np.array(np.nan, np.float32))
        assert not reply["ok"] and "spring.velocity" in reply["error"]
        assert "finite" in reply["error"]
        # the model still steps to finite numbers
        assert c.call({"op": "step", "t": 0.0, "dt": 1e-2})["ok"]
        pos = c.call({"op": "get", "vr": [c.vr("spring.position")]})["values"][0]
        assert math.isfinite(pos)


def test_an_archive_cannot_narrow_a_value_the_live_dtype_cannot_hold():
    """A float64 ``1e300`` in a float32 field is ``inf``, not a value."""
    with _Client() as c:
        refused = c.call({"op": "set", "vr": [c.vr("spring.params.stiffness")],
                          "values": [1e300]})
        assert not refused["ok"] and "does not fit" in refused["error"]

        reply = _set_state(c, "p/nodes/spring/stiffness",
                           np.array(1e300, dtype=np.float64))
        assert not reply["ok"] and "does not fit" in reply["error"]


def test_an_archive_cannot_install_a_non_finite_input():
    with _Client() as c:
        refused = c.call({"op": "set", "vr": [c.vr("spring.anchor_position")],
                          "values": [float("inf")]})
        assert not refused["ok"] and "finite" in refused["error"]

        reply = _set_state(c, "i/spring/anchor_position",
                           np.array(np.inf, dtype=np.float32))
        assert not reply["ok"] and "anchor_position" in reply["error"]


def test_a_refused_archive_commits_nothing_at_all():
    """One bad member must not leave the others half applied."""
    with _Client() as c:
        assert c.call({"op": "set", "vr": [c.vr("spring.params.stiffness")],
                       "values": [45.0]})["ok"]
        blob = _rewrite(c.archive(), "p/nodes/spring/stiffness",
                        np.array(60.0, dtype=np.float32))
        blob = _rewrite(blob, "s/spring/velocity", np.array(np.nan, np.float32))
        reply = c.call({"op": "set_state",
                        "state": base64.b64encode(blob).decode("ascii")})
        assert not reply["ok"]
        got = c.call({"op": "get", "vr": [c.vr("spring.params.stiffness")]})
        assert got["values"][0] == pytest.approx(45.0)


def test_a_snapshot_of_a_healthy_model_still_round_trips():
    with _Client() as c:
        assert c.call({"op": "step", "t": 0.0, "dt": 1e-2})["ok"]
        blob = c.archive()
        assert c.call({"op": "step", "t": 1e-2, "dt": 5e-2})["ok"]
        moved = c.call({"op": "get", "vr": [c.vr("spring.position")]})["values"][0]
        assert c.call({"op": "set_state",
                       "state": base64.b64encode(blob).decode("ascii")})["ok"]
        back = c.call({"op": "get", "vr": [c.vr("spring.position")]})["values"][0]
        assert back != pytest.approx(moved)


# --------------------------------------------------------------- property

@settings(max_examples=EXAMPLES_COSTLY, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(value=st.floats(allow_nan=True, allow_infinity=True, width=64))
def test_no_archive_can_install_a_parameter_value_that_set_would_refuse(value):
    """The two doors into the parameter tree admit exactly the same values.

    For any float at all: if ``set_state`` accepts it for a parameter,
    ``set`` accepts it too, and the value that lands is the same one.
    """
    member, name = "p/nodes/ball/elasticity", "ball.params.elasticity"
    with _Client() as c:
        vr = c.vr(name)
        by_set = c.call({"op": "set", "vr": [vr], "values": [value]})
        if by_set["ok"]:
            after_set = c.call({"op": "get", "vr": [vr]})["values"][0]
        assert c.call({"op": "reset"})["ok"]

        by_archive = _set_state(c, member, np.array(value, dtype=np.float64))
        if by_archive["ok"]:
            assert by_set["ok"], (
                f"the archive installed {value!r}, which set refused: "
                f"{by_set['error']}"
            )
            after_archive = c.call({"op": "get", "vr": [vr]})["values"][0]
            assert after_archive == pytest.approx(after_set, nan_ok=False)
