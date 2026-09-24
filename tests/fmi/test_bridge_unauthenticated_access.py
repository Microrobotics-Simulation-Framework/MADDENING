"""The FMU TCP bridge authenticates nobody (MADD-ANO-023).

``FmuTcpBridge`` binds loopback by default and tells its operator to keep
it there unless the network is trusted.  Nothing on the socket proves who
is calling.  The ``hello`` handshake returns the model's instantiation
token, which identifies the instance to the FMU; it is handed to anyone
who asks, and no op requires it (or ``hello`` at all) first.  So any
process that can reach the port can read the model, write its
parameters, advance it, and hold the one instance slot so that the real
importer is refused.

The "unrelated client" here is a separate interpreter started with
``-I -S`` (no site-packages, no ``PYTHONPATH``, so no MADDENING and no
shared secret of any kind), which knows only the host and port and
speaks the documented framing with the standard library.  The bridge
binds port 0, so the OS assigns a free ephemeral port and parallel CI
jobs cannot collide.

These pin the open limitation.  When the bridge learns to authenticate
a caller, the first two tests fail and the entry is updated with them.
"""

import inspect
import json
import socket
import subprocess
import sys

from maddening.fmi.tcp_bridge import FmuTcpBridge, recv_message, send_message
from tests.fmi.test_c_wrapper import DT, _bridge, _graph, _vr

#: A client that shares nothing with the bridge but the port number.
_STRANGER = r"""
import json, socket, struct, sys

host, port = sys.argv[1], int(sys.argv[2])
stiffness, position, dt = int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])

def read_exact(s, n):
    buf = b""
    while len(buf) < n:
        chunk = s.recv(n - len(buf))
        if not chunk:
            raise EOFError("bridge closed the connection")
        buf += chunk
    return buf

def call(s, request):
    body = json.dumps(request).encode()
    s.sendall(struct.pack(">I", len(body)) + body)
    (n,) = struct.unpack(">I", read_exact(s, 4))
    return json.loads(read_exact(s, n))

with socket.create_connection((host, port), timeout=30) as s:
    out = {
        # No hello, no token: the first frame is a read.
        "read": call(s, {"op": "get", "vr": [position, stiffness]}),
        "write": call(s, {"op": "set", "vr": [stiffness], "values": [77.0]}),
        "step": call(s, {"op": "step", "t": 0.0, "dt": dt}),
        "hello": call(s, {"op": "hello"}),
    }
print(json.dumps(out))
"""


def _run_stranger(host, port, *argv):
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", _STRANGER, host, str(port),
         *(str(a) for a in argv)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_an_unrelated_local_process_reads_writes_and_steps_the_model_with_no_secret():
    gm = _graph()
    md, bridge = _bridge(gm)
    stiffness, position = _vr(md, "spring.params.stiffness"), _vr(md, "spring.position")
    with bridge:
        host, port = bridge.endpoint.rsplit(":", 1)
        assert host == "127.0.0.1"
        out = _run_stranger(host, port, stiffness, position, DT)

        assert out["read"] == {"ok": True, "values": [0.5, 30.0]}
        assert out["write"] == {"ok": True}
        assert out["step"] == {"ok": True, "t": DT}
        # The token is not a credential: the bridge gives it to anyone.
        assert out["hello"]["ok"] and out["hello"]["token"] == md.instantiation_token

        # The importer connecting afterwards inherits what the stranger did.
        with socket.create_connection((host, int(port)), timeout=30) as importer:
            send_message(importer, {"op": "hello"})
            assert recv_message(importer)["ok"]
            send_message(importer, {"op": "get", "vr": [stiffness, _vr(md, "time")]})
            got = recv_message(importer)
            # The documented partial workaround: fmi3Reset (the bridge's
            # ``reset``) undoes what an earlier caller wrote.
            send_message(importer, {"op": "reset"})
            assert recv_message(importer)["ok"]
            send_message(importer, {"op": "get", "vr": [stiffness, _vr(md, "time")]})
            after_reset = recv_message(importer)
    assert got == {"ok": True, "values": [77.0, DT]}
    assert after_reset == {"ok": True, "values": [30.0, 0.0]}


def test_an_unrelated_client_holding_the_slot_locks_the_importer_out():
    """The single-instance lock is not access control.  Whoever speaks
    first holds the model; a later importer is refused, after the
    hand-over grace, for as long as the stranger keeps talking."""
    gm = _graph()
    md, bridge = _bridge(gm)
    with bridge:
        host, port = bridge.endpoint.rsplit(":", 1)
        with socket.create_connection((host, int(port)), timeout=30) as stranger:
            send_message(stranger, {"op": "get", "vr": [_vr(md, "time")]})
            assert recv_message(stranger)["ok"]
            with socket.create_connection((host, int(port)), timeout=30) as importer:
                send_message(importer, {"op": "hello"})
                r = recv_message(importer)
    assert not r["ok"] and "already serves an FMU instance" in r["error"]


def test_the_default_bind_is_loopback():
    """The mitigation the workaround relies on: loopback unless the
    operator passes another host, and the docstring says to keep it."""
    default = inspect.signature(FmuTcpBridge.__init__).parameters["host"].default
    assert default == "127.0.0.1"
    md, bridge = _bridge(_graph())
    try:
        assert bridge.endpoint.startswith("127.0.0.1:")
    finally:
        bridge.stop()
    import maddening.fmi.tcp_bridge as tcp_bridge
    doc = " ".join((tcp_bridge.__doc__ or "").split())
    assert "Bind the bridge to ``127.0.0.1`` unless the network is trusted" in doc
