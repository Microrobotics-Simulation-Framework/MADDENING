"""``FmuTcpBridge`` connection lifetime, over real loopback sockets.

Everything here is lifecycle: who holds the bridge's single FMU instance,
for how long, and what ``stop()`` leaves behind.  A mock socket would hide
all of it, so every test drives a real ``127.0.0.1`` connection.

The invariants:

* a peer that connects and says nothing never takes the instance slot, so
  a crashed importer, a dropped link or a port scan cannot lock out every
  later client (it could, for ever, before 0.4.0);
* a connection that goes silent is dropped, whether or not it ever spoke,
  so no connection leaks a thread and the count of live threads stays
  bounded whatever a peer does;
* ``stop()`` ends the workers, including one parked on a live connection.

Written from the independent audit of 2026-09-19 (``params-io``; report and
reproducers ``r5_bridge_wedge.py``, ``r5c_bare_connect.py``,
``r19_thread_leak.py`` under
``benchmarks/results/audit_040_final/params-io/``).
"""

import os
import socket
import struct
import threading
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from maddening.fmi import tcp_bridge
from maddening.fmi.tcp_bridge import recv_message, send_message
from tests.conftest import EXAMPLES_COSTLY
from tests.fmi.test_c_wrapper import _bridge, _graph


_GRAPH = None


def _shared_graph():
    """One compiled graph for the whole module.

    Nothing here steps the model -- these tests are about sockets -- and a
    fresh ``FmuSidecar`` is built per bridge, so compiling once keeps the
    property test's per-example cost down to the socket work it is
    actually measuring.
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = _graph()
    return _GRAPH


def _endpoint(bridge):
    host, port = bridge.endpoint.split(":")
    return host, int(port)


def _hello(bridge, timeout=15.0):
    """One honest client: connect, say hello, read the reply, disconnect."""
    with socket.create_connection(_endpoint(bridge), timeout=timeout) as sock:
        sock.settimeout(timeout)
        send_message(sock, {"op": "hello"})
        return recv_message(sock)


def _wait_until(predicate, timeout=20.0):
    """Poll ``predicate`` until it holds; return whether it did."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.05)
    return predicate()


def _conn_threads():
    return [t for t in threading.enumerate() if t.name == "maddening-fmu-conn"]


@pytest.fixture
def fast_timeouts(monkeypatch):
    """Shrink the handshake / idle budgets so a timeout can be observed.

    The shipped values are minutes: a test that waited them out would be a
    test nobody runs.  The code path is the same one, and the *frame*
    budget is left alone because none of these tests announce a frame they
    do not send.

    The idle budget keeps a couple of seconds of slack so that a test which
    checks the instance is held *before* it is released does not race a
    loaded machine; the handshake one has nothing to race.
    """
    monkeypatch.setattr(tcp_bridge, "_HANDSHAKE_TIMEOUT", 0.5)
    monkeypatch.setattr(tcp_bridge, "_IDLE_TIMEOUT", 2.0)


def test_a_connection_that_sends_nothing_never_holds_the_instance_slot():
    """The wedge: one silent peer used to own the bridge for ever."""
    with _bridge(_shared_graph())[1] as bridge:
        silent = socket.create_connection(_endpoint(bridge), timeout=5)
        try:
            # No grace period to wait out: the slot is claimed by a frame,
            # not by a connection, so an honest client is served at once.
            for _ in range(3):
                reply = _hello(bridge)
                assert reply["ok"], reply
            # ...and the slot is free again once each of them hangs up,
            # with the silent peer still connected throughout.
            assert _wait_until(lambda: not bridge._busy.locked())
        finally:
            silent.close()


def test_a_connection_that_announces_a_frame_it_never_sends_holds_nothing():
    """A four-byte length prefix and then silence: the audit's ``r5``."""
    with _bridge(_shared_graph())[1] as bridge:
        stalled = socket.create_connection(_endpoint(bridge), timeout=5)
        try:
            stalled.sendall(struct.pack(">I", 100))     # promises 100 bytes
            time.sleep(0.2)
            reply = _hello(bridge)
            assert reply["ok"], reply
        finally:
            stalled.close()


def test_a_silent_peer_is_dropped_and_leaks_no_thread(fast_timeouts):
    with _bridge(_shared_graph())[1] as bridge:
        before = len(_conn_threads())
        socks = [socket.create_connection(_endpoint(bridge), timeout=5)
                 for _ in range(8)]
        try:
            assert _wait_until(lambda: len(_conn_threads()) <= before), (
                "a connection that said nothing must not park a thread"
            )
        finally:
            for sock in socks:
                sock.close()


def test_the_number_of_connection_threads_is_capped():
    """Past the cap, a connection is closed on accept rather than served."""
    with _bridge(_shared_graph())[1] as bridge:
        socks = [socket.create_connection(_endpoint(bridge), timeout=5)
                 for _ in range(tcp_bridge._MAX_CONNECTIONS + 24)]
        try:
            assert _wait_until(lambda: bridge.connections_refused_over_cap > 0)
            assert len(_conn_threads()) <= tcp_bridge._MAX_CONNECTIONS
        finally:
            for sock in socks:
                sock.close()


def test_an_established_connection_that_goes_silent_is_dropped(fast_timeouts):
    """The idle budget applies after the handshake too, and frees the slot."""
    with _bridge(_shared_graph())[1] as bridge:
        holder = socket.create_connection(_endpoint(bridge), timeout=5)
        try:
            holder.settimeout(5)
            send_message(holder, {"op": "hello"})
            assert recv_message(holder)["ok"]
            assert bridge._busy.locked()
            assert _wait_until(lambda: not bridge._busy.locked())
            assert _hello(bridge)["ok"]
        finally:
            holder.close()


def test_stop_ends_a_worker_parked_on_a_live_connection():
    """``stop()`` used to return in 0.00 s with the worker still parked."""
    _, bridge = _bridge(_shared_graph())
    bridge.start()
    holder = socket.create_connection(_endpoint(bridge), timeout=5)
    try:
        holder.settimeout(5)
        send_message(holder, {"op": "hello"})
        assert recv_message(holder)["ok"]
        workers = list(bridge._live_workers)
        assert workers, "the worker should be serving the connection"
        bridge.stop()
        assert not [w for w in workers if w.is_alive()], (
            "stop() must not leave a connection worker behind"
        )
        assert not bridge._busy.locked()
    finally:
        holder.close()


# --------------------------------------------------------------- property

_SILENT_ACTIONS = st.sampled_from([
    "connect",            # connect and say nothing at all
    "prefix",             # announce a frame, send no body
    "half_prefix",        # two bytes of a four-byte length prefix
    "connect_close",      # connect and hang up immediately
])


@settings(max_examples=EXAMPLES_COSTLY, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(actions=st.lists(_SILENT_ACTIONS, min_size=1, max_size=10))
def test_no_sequence_of_silent_connects_can_lock_out_an_honest_client(actions):
    """The bridge survives any sequence of connects that say nothing useful.

    Each draw opens a handful of connections that send nothing, or nothing
    complete, and keeps them open; an honest ``hello`` must still be
    answered.  Before 0.4.0 the very first ``"connect"`` was enough to
    refuse every client until the process was restarted.
    """
    with _bridge(_shared_graph())[1] as bridge:
        socks = []
        try:
            for action in actions:
                sock = socket.create_connection(_endpoint(bridge), timeout=5)
                socks.append(sock)
                if action == "prefix":
                    sock.sendall(struct.pack(">I", 4096))
                elif action == "half_prefix":
                    sock.sendall(b"\x00\x00")
                elif action == "connect_close":
                    sock.close()
            reply = _hello(bridge)
            assert reply["ok"], reply
        finally:
            for sock in socks:
                try:
                    sock.close()
                except OSError:
                    pass
