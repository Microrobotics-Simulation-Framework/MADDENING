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
    TOKEN_ENV,
    TRANSPORT_TOKEN_ENV,
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

    def test_the_state_is_readable_without_curve_and_unreadable_with_it(self):
        """Differential: same observer, same payload, encryption the only change.

        The "nothing leaked" half is not evidence on its own.  This
        started life as a raw ``socket.create_connection`` tap asserting
        no field name appeared in the captured bytes -- and it **passed
        with encryption switched off**, because a socket that never
        completes a ZMTP handshake receives no published frame from
        either relay.  Measured, then replaced.

        So the readable case is asserted first: if a plain subscriber
        cannot read even the unencrypted relay, the second assertion
        proves nothing and this test says so instead of passing.
        """
        field = "secret_position"

        def capture(secure: bool) -> str:
            port = _free_port()
            address = f"tcp://127.0.0.1:{port}"
            relay = NetworkRelay(address=address, secure=secure, token=TOKEN)
            graph = _FakeGraphManager()
            relay.attach(graph)
            context = zmq.Context()
            sub = _sub(context, address, token=None)  # no credential at all
            seen = b""
            try:
                deadline = time.monotonic() + 2.5
                while time.monotonic() < deadline:
                    graph.emit(STATE)
                    while True:
                        try:
                            seen += sub.recv(zmq.NOBLOCK)
                        except zmq.Again:
                            break
                    if seen:
                        break
                    time.sleep(0.02)
            finally:
                sub.close()
                context.term()
                relay.close()
            return seen.decode("utf-8", "replace")

        cleartext = capture(secure=False)
        encrypted = capture(secure=True)

        assert field in cleartext, (
            "a plain subscriber could not read the UNENCRYPTED relay, so "
            "the encrypted result below would prove nothing"
        )
        assert field not in encrypted


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
    # A DEALER whose peer refuses the handshake goes mute, and an unbounded
    # send on a mute socket never returns. Without this the "no credential"
    # case hangs the test run instead of failing it.
    sock.setsockopt(zmq.SNDTIMEO, 500)
    if token is not None:
        TransportAuth(token=token).secure_client(sock)
    sock.connect(address)
    try:
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            try:
                sock.send_multipart([b"", json.dumps({
                    "type": "register",
                    "subgraph_id": subgraph_id,
                    "address": peer_address,
                    "zmq_ports": {"state": 5555},
                }).encode()])
            except zmq.Again:
                pass  # mute socket: the peer refused us, which is the point
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


class TestWorkerClientFailsFastOnAMismatch:
    """A worker that cannot authenticate must fail, not hang."""

    def test_a_worker_without_curve_does_not_hang_on_a_curve_coordinator(self):
        """The asymmetric case the address rule cannot see.

        A coordinator bound to a non-loopback address has CURVE on.  A
        worker reaching it over a *loopback* address -- rank 0's own
        worker, or anything through an SSH tunnel -- sees loopback and
        turns CURVE off.  libzmq then puts the DEALER in mute state and
        an unbounded send never returns, so before this was bounded the
        worker hung forever and ``timeout`` meant nothing.
        """
        from maddening.cloud.multigpu.worker_client import WorkerClient

        port = _free_port()
        coord = Coordinator(expected_workers=["flow"], edges=[], port=port,
                            bind_host="127.0.0.1", secure=True, token=TOKEN)
        coord.start()
        client = WorkerClient(
            coordinator_addr=f"127.0.0.1:{port}",
            subgraph_id="flow",
            address="127.0.0.1:5555",
            secure=False,          # the mismatch
        )
        started = time.monotonic()
        try:
            with pytest.raises((ConnectionError, TimeoutError)):
                client.register_and_wait(timeout=3)
            # Measured before the teardown below, which sleeps.
            elapsed = time.monotonic() - started
        finally:
            coord.shutdown()
            time.sleep(1.2)
        # Not merely "it finished": it must honour the deadline it was
        # given.  The recv timeout is the loop's poll interval, so a recv
        # timeout longer than `timeout` silently overshoots it -- which is
        # what a 5s RCVTIMEO did to a 3s deadline before this was pinned.
        assert elapsed < 3 + 1.5, (
            f"register_and_wait(timeout=3) took {elapsed:.1f}s; the "
            f"deadline is only checked between recv calls, so the recv "
            f"timeout must be short compared to it"
        )


# ---------------------------------------------------------------------
# Local development must stay frictionless
# ---------------------------------------------------------------------

class TestLoopbackStaysFrictionless:
    """No token, no keys, no configuration for a developer on one box."""

    def test_the_shipped_defaults_bind_loopback(self, monkeypatch):
        monkeypatch.delenv("MADDENING_API_TOKEN", raising=False)
        monkeypatch.delenv("MADDENING_TRANSPORT_TOKEN", raising=False)
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
        monkeypatch.delenv("MADDENING_TRANSPORT_TOKEN", raising=False)
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
        monkeypatch.delenv("MADDENING_TRANSPORT_TOKEN", raising=False)
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
        monkeypatch.delenv("MADDENING_TRANSPORT_TOKEN", raising=False)
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
        monkeypatch.delenv("MADDENING_TRANSPORT_TOKEN", raising=False)
        with pytest.raises(TransportAuthError, match="MADDENING_API_TOKEN"):
            Coordinator(expected_workers=["a"], edges=[],
                        port=_free_port(), bind_host="0.0.0.0")

    def test_encryption_cannot_be_switched_off_on_a_reachable_address(self):
        with pytest.raises(TransportAuthError, match="secure=False"):
            resolve_security("tcp://0.0.0.0:5555", False)

    def test_a_blank_token_is_a_configuration_error(self, monkeypatch):
        monkeypatch.delenv("MADDENING_TRANSPORT_TOKEN", raising=False)
        monkeypatch.setenv("MADDENING_API_TOKEN", "   ")
        with pytest.raises(TransportAuthError, match="blank"):
            TransportAuth()

    def test_a_wildcard_bind_turns_encryption_on_by_itself(self, monkeypatch):
        """End to end on a real wildcard bind, with nothing forced.

        The other socket tests pass ``secure=True`` so they can stay on
        loopback.  This one proves the *automatic* path -- the one a
        user actually hits -- reaches the same place.
        """
        monkeypatch.delenv("MADDENING_TRANSPORT_TOKEN", raising=False)
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
        monkeypatch.delenv("MADDENING_TRANSPORT_TOKEN", raising=False)
        monkeypatch.setenv("MADDENING_API_TOKEN", TOKEN)
        assert TransportAuth().server_keypair() == (
            TransportAuth(token=TOKEN).server_keypair())

    def test_the_derivation_matches_a_pinned_vector(self):
        """A golden vector, so the derivation cannot drift between versions.

        Everything else here checks *self-consistency*, which is
        automatic when both ends run the same code and therefore cannot
        see a change to the personalisation string, the role literals,
        the separator or the hash.  Measured: changing ``_CURVE_PERSON``
        from ``b"maddening-curve"`` to ``b"maddening-CURVE"`` left all 36
        tests in this file green, while making 0.4.x and 0.5.x unable to
        talk to each other -- and the failure mode of that is the silent
        one in ``NetworkReceiver``, not an exception.

        These are the keys ``TransportAuth(token="the-shared-token")``
        must produce for ever.  If this fails, the derivation changed and
        that is a wire-compatibility break, not a test to update.
        """
        auth = TransportAuth(token="the-shared-token")

        assert auth.server_keypair() == (
            b"wG#*&<Nt]&7xpdy={R1&}&A#?Hoq3SygSb?=B=Ru",
            b"ULh/<s#hK!!wx<$/U9(0ek11I?)Ug:G:])]J]Q&h",
        )
        assert auth.client_keypair() == (
            b"i%y(@xeWQ{xK$.wl?oOZw)<VO-.8@>wSR3!0Zioc",
            b"9]+A6Y2q:E2w#NO{U6nI9iCJpqe/i:gqxs<V?G5H",
        )


# ---------------------------------------------------------------------
# The transport secret is separable from the HTTP bearer credential
# ---------------------------------------------------------------------

class TestTransportSecretIsSeparableFromTheApiToken:
    """The CURVE seed must not have to be the cleartext HTTP credential.

    There is no TLS in front of the HTTP API, so ``MADDENING_API_TOKEN``
    is visible in an ``Authorization`` header on every request.  While
    that token was also the CURVE seed, one sniffed request yielded both
    keypairs and the "encrypted" state stream was readable -- measured
    over a real socket, not inferred.  ``MADDENING_TRANSPORT_TOKEN``
    exists so the streams need not inherit that exposure;
    ``MADDENING_API_TOKEN`` remains the fallback so a single-variable
    deployment keeps working.
    """

    def test_the_transport_variable_is_preferred(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV, "the-http-credential")
        monkeypatch.setenv(TRANSPORT_TOKEN_ENV, "the-transport-secret")

        auth = TransportAuth()

        assert auth.token == "the-transport-secret"
        assert auth.token_env == TRANSPORT_TOKEN_ENV

    def test_the_api_token_is_the_documented_fallback(self, monkeypatch):
        """A single-variable setup keeps working, exactly as before."""
        monkeypatch.delenv(TRANSPORT_TOKEN_ENV, raising=False)
        monkeypatch.setenv(TOKEN_ENV, "the-http-credential")

        auth = TransportAuth()

        assert auth.token == "the-http-credential"
        assert auth.token_env == TOKEN_ENV

    def test_the_explicit_argument_still_wins_over_both(self, monkeypatch):
        monkeypatch.setenv(TOKEN_ENV, "the-http-credential")
        monkeypatch.setenv(TRANSPORT_TOKEN_ENV, "the-transport-secret")

        auth = TransportAuth(token="explicit")

        assert auth.token == "explicit"
        assert auth.token_env is None

    def test_a_blank_transport_token_does_not_fall_back(self, monkeypatch):
        """A variable that is set is the operator's answer.

        Falling through to a *different* secret because this one is
        blank would leave two ends deriving different keys, which fails
        as a handshake timeout and reads as a network problem.
        """
        monkeypatch.setenv(TOKEN_ENV, "the-http-credential")
        monkeypatch.setenv(TRANSPORT_TOKEN_ENV, "  ")

        with pytest.raises(TransportAuthError, match=TRANSPORT_TOKEN_ENV):
            TransportAuth()

    def test_with_both_set_the_http_credential_does_not_open_the_stream(
        self, monkeypatch,
    ):
        """The gate, over a real socket.

        An attacker who sniffed one ``Authorization`` header holds
        ``MADDENING_API_TOKEN``.  With the transport variable set that is
        no longer the CURVE seed, so the attacker derives the wrong
        keypair and reads nothing -- while the peer holding the transport
        secret reads the stream, which is what stops this passing
        vacuously.
        """
        monkeypatch.setenv(TOKEN_ENV, "the-http-credential")
        monkeypatch.setenv(TRANSPORT_TOKEN_ENV, "the-transport-secret")
        port = _free_port()
        address = f"tcp://127.0.0.1:{port}"
        relay = NetworkRelay(address=address, secure=True)
        graph = _FakeGraphManager()
        relay.attach(graph)
        context = zmq.Context()
        subs = {
            "authorised": _sub(context, address, token="the-transport-secret"),
            "holds-the-http-token": _sub(
                context, address, token="the-http-credential",
            ),
        }
        try:
            counts = _exchange(lambda: graph.emit(STATE), subs)
        finally:
            for sock in subs.values():
                sock.close()
            context.term()
            relay.close()

        assert counts["authorised"] > 0, (
            "the peer holding the transport secret read nothing, so this "
            "test proves nothing about the peer that holds only the HTTP "
            "credential"
        )
        assert counts["holds-the-http-token"] == 0

    def test_the_fallback_is_what_makes_a_sniffed_api_token_sufficient(
        self, monkeypatch,
    ):
        """The control run for the test above: remove the separation.

        With only ``MADDENING_API_TOKEN`` set, the sniffed HTTP
        credential *is* the CURVE seed and does open the stream.  That is
        the documented cost of the single-variable setup, and asserting
        it is what makes the test above a measurement rather than a
        restatement of the code.
        """
        monkeypatch.delenv(TRANSPORT_TOKEN_ENV, raising=False)
        monkeypatch.setenv(TOKEN_ENV, "the-http-credential")
        port = _free_port()
        address = f"tcp://127.0.0.1:{port}"
        relay = NetworkRelay(address=address, secure=True)
        graph = _FakeGraphManager()
        relay.attach(graph)
        context = zmq.Context()
        subs = {"authorised": _sub(context, address, token="the-http-credential")}
        try:
            counts = _exchange(lambda: graph.emit(STATE), subs)
        finally:
            for sock in subs.values():
                sock.close()
            context.term()
            relay.close()

        assert counts["authorised"] > 0
