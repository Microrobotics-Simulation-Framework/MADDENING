"""The ZeroMQ transports must not hand their contents to a stranger.

Every socket test here runs over a **real TCP socket** on an ephemeral
port, not ``inproc://``, because the thing under test is libzmq's CURVE
handshake and a ZAP handler running on its own thread.  An in-process
double would prove nothing about either.

The two tests the brief asks for by name are
:func:`test_an_unauthenticated_subscriber_cannot_read_the_state_stream`
and
:func:`test_the_coordinator_ignores_an_unauthenticated_registration`.
The rest pin the things that would let those two pass while the
transport was still open: a wrong token, a self-generated keypair, a
missing ZAP allowlist, and the defaults that decide whether any of it
is switched on at all.
"""

from __future__ import annotations

import json
import socket
import time

import pytest

zmq = pytest.importorskip("zmq", reason="ZMQ transport security needs pyzmq")

from maddening.cloud.multigpu.coordinator import Coordinator  # noqa: E402
from maddening.transport_auth import (  # noqa: E402
    TransportAuth,
    TransportAuthError,
    address_is_loopback,
    resolve_security,
)
from maddening.viz.network import (  # noqa: E402
    CommandPublisher,
    CommandReceiver,
    NetworkRelay,
    NetworkReceiver,
)

TOKEN = "the-shared-token"
OTHER_TOKEN = "not-the-shared-token"

#: How long to keep publishing while waiting for the authorised peer.
_SETTLE = 5.0

#: How long to keep publishing *after* that, so a peer that should be
#: blocked has had every chance to receive something.  A test that
#: asserted "nothing arrived" without this would pass on a race.
_GRACE = 0.75


def _free_port() -> int:
    """An ephemeral TCP port that is free right now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _drain(sock) -> int:
    """Count the frames waiting on *sock* without blocking."""
    seen = 0
    while True:
        try:
            sock.recv(zmq.NOBLOCK)
        except zmq.Again:
            return seen
        seen += 1


def _exchange(emit, subscribers: dict) -> dict:
    """Publish until the authorised peer receives, then keep going.

    Parameters
    ----------
    emit : callable
        Publishes one frame.
    subscribers : dict
        ``{name: socket}``.  ``"authorised"`` must be among them; it is
        the peer whose first frame starts the grace period.

    Returns
    -------
    dict
        ``{name: frames received}``.
    """
    counts = {name: 0 for name in subscribers}
    deadline = time.monotonic() + _SETTLE
    grace_until = None
    while time.monotonic() < deadline:
        emit()
        for name, sock in subscribers.items():
            counts[name] += _drain(sock)
        if grace_until is None and counts["authorised"]:
            grace_until = time.monotonic() + _GRACE
        if grace_until is not None and time.monotonic() >= grace_until:
            break
        time.sleep(0.02)
    for name, sock in subscribers.items():
        counts[name] += _drain(sock)
    return counts


def _sub(context, address: str, *, token: str | None, own_keypair: bool = False):
    """A SUB socket in one of the four postures an attacker can take."""
    sock = context.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    if token is not None:
        auth = TransportAuth(token=token)
        if own_keypair:
            # Knows the server's public key -- assume it leaked -- but
            # brings a keypair of its own that no allowlist admits.
            server_public, _ = auth.server_keypair()
            public, secret = zmq.curve_keypair()
            sock.curve_secretkey = secret
            sock.curve_publickey = public
            sock.curve_serverkey = server_public
        else:
            auth.secure_client(sock)
    sock.setsockopt_string(zmq.SUBSCRIBE, "")
    sock.connect(address)
    return sock


class _FakeGraphManager:
    """The observer contract ``NetworkRelay.attach`` relies on."""

    timestep = 0.01

    def __init__(self) -> None:
        self._observers = []

    def add_observer(self, callback) -> None:
        self._observers.append(callback)

    def emit(self, state) -> None:
        for callback in self._observers:
            callback("step", state)


STATE = {"robot": {"joint_angle": 1.25, "secret_position": 42.0}}


# ---------------------------------------------------------------------
# The state stream must not be readable without the token
# ---------------------------------------------------------------------

class TestStateStreamConfidentiality:
    """``NetworkRelay`` on a reachable address discloses nothing."""

    @pytest.fixture
    def relay_and_subs(self):
        port = _free_port()
        address = f"tcp://127.0.0.1:{port}"
        relay = NetworkRelay(address=address, secure=True, token=TOKEN)
        graph = _FakeGraphManager()
        relay.attach(graph)
        context = zmq.Context()
        subs = {
            "authorised": _sub(context, address, token=TOKEN),
            "no-credential": _sub(context, address, token=None),
            "wrong-token": _sub(context, address, token=OTHER_TOKEN),
            "own-keypair": _sub(context, address, token=TOKEN, own_keypair=True),
        }
        try:
            yield relay, graph, subs
        finally:
            for sock in subs.values():
                sock.close()
            context.term()
            relay.close()

    def test_an_unauthenticated_subscriber_cannot_read_the_state_stream(
        self, relay_and_subs,
    ):
        """A plain SUB socket -- what any attacker runs first -- gets nothing.

        This is the gate the brief asks for: it fails if the state
        stream is readable by a peer holding no credential.
        """
        relay, graph, subs = relay_and_subs
        counts = _exchange(lambda: graph.emit(STATE), subs)

        assert counts["authorised"] > 0, (
            "the authorised subscriber received nothing, so this test "
            "proves nothing about the others"
        )
        assert counts["no-credential"] == 0

    def test_a_subscriber_holding_the_wrong_token_cannot_read_the_state_stream(
        self, relay_and_subs,
    ):
        relay, graph, subs = relay_and_subs
        counts = _exchange(lambda: graph.emit(STATE), subs)

        assert counts["authorised"] > 0
        assert counts["wrong-token"] == 0

    def test_a_subscriber_with_its_own_keypair_cannot_read_the_state_stream(
        self, relay_and_subs,
    ):
        """The ZAP allowlist, not the secrecy of the server key, is the gate.

        CURVE on its own authenticates the *server* to the client and
        accepts any client key.  Measured before the allowlist existed,
        a peer that knew the server's public key and generated its own
        keypair received every frame.  This test is what fails if
        ``start_authenticator`` is dropped.
        """
        relay, graph, subs = relay_and_subs
        counts = _exchange(lambda: graph.emit(STATE), subs)

        assert counts["authorised"] > 0
        assert counts["own-keypair"] == 0

    def test_the_published_frames_are_not_the_plaintext_state(self):
        """A passive observer of the wire sees no field name from the state.

        The subscriber tests above prove libzmq refuses the handshake.
        This one proves the bytes on the wire are encrypted too, so a
        tap that never completes a handshake learns nothing either.
        """
        port = _free_port()
        address = f"tcp://127.0.0.1:{port}"
        relay = NetworkRelay(address=address, secure=True, token=TOKEN)
        graph = _FakeGraphManager()
        relay.attach(graph)
        try:
            tap = socket.create_connection(("127.0.0.1", port), timeout=2.0)
            try:
                tap.settimeout(0.5)
                deadline = time.monotonic() + 2.0
                captured = b""
                while time.monotonic() < deadline:
                    graph.emit(STATE)
                    try:
                        chunk = tap.recv(65536)
                    except (TimeoutError, socket.timeout):
                        continue
                    if not chunk:
                        break
                    captured += chunk
            finally:
                tap.close()
        finally:
            relay.close()

        assert b"secret_position" not in captured
        assert b"joint_angle" not in captured


class TestCommandChannelConfidentiality:
    """The command channel carries torques; it is not world-readable."""

    def test_an_unauthenticated_subscriber_cannot_read_the_command_stream(self):
        port = _free_port()
        address = f"tcp://127.0.0.1:{port}"
        publisher = CommandPublisher(address=address, secure=True, token=TOKEN)
        context = zmq.Context()
        subs = {
            "authorised": _sub(context, address, token=TOKEN),
            "no-credential": _sub(context, address, token=None),
        }
        command = {"robot": {"joint_torques": [0.1, -0.2, 0.0]}}
        try:
            counts = _exchange(lambda: publisher.send(command), subs)
        finally:
            for sock in subs.values():
                sock.close()
            context.term()
            publisher.close()

        assert counts["authorised"] > 0
        assert counts["no-credential"] == 0

    def test_the_paired_receiver_reads_the_command_with_the_same_token(self):
        """The happy path: two MADDENING objects, one shared token."""
        port = _free_port()
        address = f"tcp://127.0.0.1:{port}"
        publisher = CommandPublisher(address=address, secure=True, token=TOKEN)
        receiver = CommandReceiver(address=address, secure=True, token=TOKEN)
        receiver.start()
        command = {"robot": {"joint_torques": [0.1, -0.2, 0.0]}}
        try:
            deadline = time.monotonic() + _SETTLE
            while time.monotonic() < deadline:
                publisher.send(command)
                if receiver.latest_commands() is not None:
                    break
                time.sleep(0.02)
            assert receiver.latest_commands() == command
        finally:
            receiver.stop()
            publisher.close()


class TestSecuredRelayRoundTrip:
    """A secured relay and its matching receiver still work end to end."""

    def test_the_paired_receiver_reads_the_state_with_the_same_token(self):
        port = _free_port()
        address = f"tcp://127.0.0.1:{port}"
        relay = NetworkRelay(address=address, secure=True, token=TOKEN)
        graph = _FakeGraphManager()
        relay.attach(graph)
        receiver = NetworkReceiver(address=address, secure=True, token=TOKEN)
        receiver.start()
        try:
            deadline = time.monotonic() + _SETTLE
            while time.monotonic() < deadline:
                graph.emit(STATE)
                if receiver.latest_snapshot()[1] is not None:
                    break
                time.sleep(0.02)
            _, snapshot = receiver.latest_snapshot()
            assert snapshot is not None
            assert snapshot["robot"]["joint_angle"] == pytest.approx(1.25)
        finally:
            receiver.stop()
            relay.close()


# ---------------------------------------------------------------------
# The coordinator must not act on an unauthenticated message
# ---------------------------------------------------------------------

def _register(context, address: str, subgraph_id: str, *, token: str | None,
              peer_address: str) -> None:
    """Send one ``register`` frame to the coordinator's ROUTER."""
    sock = context.socket(zmq.DEALER)
    sock.setsockopt(zmq.LINGER, 0)
    if token is not None:
        TransportAuth(token=token).secure_client(sock)
    sock.connect(address)
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            sock.send_multipart([b"", json.dumps({
                "type": "register",
                "subgraph_id": subgraph_id,
                "address": peer_address,
                "zmq_ports": {"state": 5555},
            }).encode()])
            time.sleep(0.1)
    finally:
        sock.close()


class TestCoordinatorAuthentication:
    """The ROUTER acts only on registrations it can authenticate."""

    @pytest.fixture
    def coordinator(self):
        port = _free_port()
        coord = Coordinator(
            expected_workers=["flow", "structure"],
            edges=[{"source": "flow", "target": "structure",
                    "source_field": "v", "target_field": "v"}],
            port=port,
            bind_host="127.0.0.1",
            secure=True,
            token=TOKEN,
        )
        coord.start()
        try:
            yield coord, f"tcp://127.0.0.1:{port}"
        finally:
            coord.shutdown()
            time.sleep(1.2)  # let _run leave its 1s recv timeout

    def test_the_coordinator_ignores_an_unauthenticated_registration(
        self, coordinator,
    ):
        """The gate the brief asks for: no credential, no registration.

        An accepted ``register`` is not merely noise.  The coordinator
        stores the ``address`` it carries and later hands it to the
        *other* workers as the peer to subscribe to, so a registration
        an attacker controls redirects a victim's data plane at a
        publisher the attacker owns.
        """
        coord, address = coordinator
        context = zmq.Context()
        try:
            _register(context, address, "flow", token=None,
                      peer_address="attacker.example:5555")
        finally:
            context.term()

        assert coord.registered_workers == {}

    def test_the_coordinator_ignores_a_registration_holding_the_wrong_token(
        self, coordinator,
    ):
        coord, address = coordinator
        context = zmq.Context()
        try:
            _register(context, address, "flow", token=OTHER_TOKEN,
                      peer_address="attacker.example:5555")
        finally:
            context.term()

        assert coord.registered_workers == {}

    def test_the_coordinator_accepts_a_registration_holding_the_token(
        self, coordinator,
    ):
        """The happy path, so the two tests above cannot pass vacuously."""
        coord, address = coordinator
        context = zmq.Context()
        try:
            _register(context, address, "flow", token=TOKEN,
                      peer_address="10.0.0.2:5555")
        finally:
            context.term()

        assert "flow" in coord.registered_workers
        assert coord.registered_workers["flow"].address == "10.0.0.2:5555"


# ---------------------------------------------------------------------
# Local development must stay frictionless
# ---------------------------------------------------------------------

class TestLoopbackStaysFrictionless:
    """No token, no keys, no configuration for a developer on one box."""

    def test_the_shipped_defaults_bind_loopback(self, monkeypatch):
        monkeypatch.delenv("MADDENING_API_TOKEN", raising=False)
        relay = NetworkRelay()
        publisher = CommandPublisher()
        try:
            assert relay.secure is False
            assert publisher.secure is False
        finally:
            relay.close()
            publisher.close()

    def test_a_loopback_relay_and_receiver_round_trip_without_a_token(
        self, monkeypatch,
    ):
        """The whole local viz flow, with MADDENING_API_TOKEN unset."""
        monkeypatch.delenv("MADDENING_API_TOKEN", raising=False)
        port = _free_port()
        address = f"tcp://127.0.0.1:{port}"
        relay = NetworkRelay(address=address)
        graph = _FakeGraphManager()
        relay.attach(graph)
        receiver = NetworkReceiver(address=address)
        receiver.start()
        try:
            assert relay.secure is False
            deadline = time.monotonic() + _SETTLE
            while time.monotonic() < deadline:
                graph.emit(STATE)
                if receiver.latest_snapshot()[1] is not None:
                    break
                time.sleep(0.02)
            assert receiver.latest_snapshot()[1] is not None
        finally:
            receiver.stop()
            relay.close()

    def test_a_loopback_coordinator_needs_no_token(self, monkeypatch):
        monkeypatch.delenv("MADDENING_API_TOKEN", raising=False)
        coord = Coordinator(expected_workers=["a"], edges=[], port=_free_port())
        assert coord.secure is False
        assert coord.bind_address.startswith("tcp://127.0.0.1:")


# ---------------------------------------------------------------------
# A reachable address fails closed
# ---------------------------------------------------------------------

class TestReachableAddressesFailClosed:
    """Missing configuration must stop the socket, not downgrade it."""

    @pytest.mark.parametrize("address", [
        "tcp://*:5555",
        "tcp://0.0.0.0:5555",
        "tcp://10.0.0.4:5555",
        "tcp://[::]:5555",
    ])
    def test_a_reachable_address_is_not_loopback(self, address):
        assert address_is_loopback(address) is False
        assert resolve_security(address, None) is True

    @pytest.mark.parametrize("address", [
        "tcp://127.0.0.1:5555",
        "tcp://localhost:5555",
        "tcp://[::1]:5555",
        "ipc:///tmp/maddening.sock",
        "inproc://maddening",
    ])
    def test_a_local_address_is_loopback(self, address):
        assert address_is_loopback(address) is True
        assert resolve_security(address, None) is False

    @pytest.mark.parametrize("address", ["", "5555", "not-an-address"])
    def test_an_unparseable_address_fails_closed(self, address):
        """An address we cannot classify is treated as reachable."""
        assert address_is_loopback(address) is False

    def test_a_relay_on_a_reachable_address_refuses_without_a_token(
        self, monkeypatch,
    ):
        monkeypatch.delenv("MADDENING_API_TOKEN", raising=False)
        with pytest.raises(TransportAuthError, match="MADDENING_API_TOKEN"):
            NetworkRelay(address=f"tcp://0.0.0.0:{_free_port()}")

    def test_a_coordinator_on_a_reachable_address_refuses_without_a_token(
        self, monkeypatch,
    ):
        """The constructor refuses, not the background thread.

        ``Coordinator._run`` executes on its own thread, where an
        exception is logged and lost.  If the token check lived there, a
        coordinator that failed to encrypt would be indistinguishable
        from one that succeeded.
        """
        monkeypatch.delenv("MADDENING_API_TOKEN", raising=False)
        with pytest.raises(TransportAuthError, match="MADDENING_API_TOKEN"):
            Coordinator(expected_workers=["a"], edges=[],
                        port=_free_port(), bind_host="0.0.0.0")

    def test_encryption_cannot_be_switched_off_on_a_reachable_address(self):
        with pytest.raises(TransportAuthError, match="secure=False"):
            resolve_security("tcp://0.0.0.0:5555", False)

    def test_a_blank_token_is_a_configuration_error(self, monkeypatch):
        monkeypatch.setenv("MADDENING_API_TOKEN", "   ")
        with pytest.raises(TransportAuthError, match="blank"):
            TransportAuth()

    def test_a_wildcard_bind_turns_encryption_on_by_itself(self, monkeypatch):
        """End to end on a real wildcard bind, with nothing forced.

        The other socket tests pass ``secure=True`` so they can stay on
        loopback.  This one proves the *automatic* path -- the one a
        user actually hits -- reaches the same place.
        """
        monkeypatch.setenv("MADDENING_API_TOKEN", TOKEN)
        port = _free_port()
        relay = NetworkRelay(address=f"tcp://0.0.0.0:{port}")
        graph = _FakeGraphManager()
        relay.attach(graph)
        context = zmq.Context()
        subs = {
            "authorised": _sub(context, f"tcp://127.0.0.1:{port}", token=TOKEN),
            "no-credential": _sub(context, f"tcp://127.0.0.1:{port}", token=None),
        }
        try:
            assert relay.secure is True
            counts = _exchange(lambda: graph.emit(STATE), subs)
        finally:
            for sock in subs.values():
                sock.close()
            context.term()
            relay.close()

        assert counts["authorised"] > 0
        assert counts["no-credential"] == 0


# ---------------------------------------------------------------------
# Key derivation
# ---------------------------------------------------------------------

class TestKeyDerivation:
    """One token on both sides has to produce one agreed keypair."""

    def test_the_same_token_derives_the_same_keys(self):
        first = TransportAuth(token=TOKEN)
        second = TransportAuth(token=TOKEN)
        assert first.server_keypair() == second.server_keypair()
        assert first.client_keypair() == second.client_keypair()

    def test_a_different_token_derives_different_keys(self):
        assert (TransportAuth(token=TOKEN).server_keypair()
                != TransportAuth(token=OTHER_TOKEN).server_keypair())

    def test_the_server_and_client_keys_are_independent(self):
        auth = TransportAuth(token=TOKEN)
        server_public, server_secret = auth.server_keypair()
        client_public, client_secret = auth.client_keypair()
        assert len({server_public, server_secret,
                    client_public, client_secret}) == 4

    def test_the_keys_are_well_formed_z85(self):
        public, secret = TransportAuth(token=TOKEN).server_keypair()
        assert len(public) == 40 and len(secret) == 40
        assert zmq.curve_public(secret) == public

    def test_the_token_is_read_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("MADDENING_API_TOKEN", TOKEN)
        assert TransportAuth().server_keypair() == (
            TransportAuth(token=TOKEN).server_keypair())
