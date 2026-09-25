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
* ``stop()`` ends the workers, including one parked on a live connection;
* a frame whose length has arrived must be finished within the frame
  budget, whether the rest of it dribbles in or stops arriving;
* a bridge starts once, and not after ``stop()``.

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


def test_a_stop_that_lands_before_the_serve_thread_runs_is_quiet(monkeypatch):
    """``stop()`` can close the listening socket before the serve thread's
    first line runs -- ``start()`` then ``stop()`` on a loaded machine; seen
    under four xdist workers, where the thread died with EBADF from
    ``settimeout`` on the closed socket.  The serve loop is held back here
    until ``stop()`` has closed the socket, which is that ordering made
    deterministic: the thread must exit on its own, with no exception for
    pytest to report.  (This used to stop first and start after; a bridge
    now refuses to start once stopped.)"""
    raised = []
    monkeypatch.setattr(threading, "excepthook", lambda args: raised.append(args))
    _, bridge = _bridge(_shared_graph())
    serve = bridge._serve

    def serve_after_the_socket_is_closed():
        assert _wait_until(lambda: bridge._server.fileno() == -1, timeout=10.0)
        serve()

    monkeypatch.setattr(bridge, "_serve", serve_after_the_socket_is_closed)
    bridge.start()
    bridge.stop()
    assert not bridge._thread.is_alive()
    assert not raised, [f"{a.exc_type.__name__}: {a.exc_value}" for a in raised]


def test_a_bridge_starts_once():
    """``start()`` twice ran two accept loops on one socket, and ``stop()``
    joined only the last; both loops served."""
    def serve_threads():
        return {t for t in threading.enumerate() if t.name == "maddening-fmu-bridge"}

    before = serve_threads()
    _, bridge = _bridge(_shared_graph())
    with bridge:
        first = bridge._thread
        with pytest.raises(RuntimeError, match="already started"):
            bridge.start()
        assert bridge._thread is first
        assert serve_threads() - before == {first}
        assert _hello(bridge)["ok"]
    assert not first.is_alive()


@pytest.mark.parametrize("started", [True, False], ids=["after_serving", "never_started"])
def test_a_stopped_bridge_refuses_to_start(started):
    """``start()`` after ``stop()`` returned the bridge as if it served, with
    a thread that exited at once on the closed socket: every connection
    was refused and nothing said why."""
    _, bridge = _bridge(_shared_graph())
    if started:
        bridge.start()
        assert _hello(bridge)["ok"]
    bridge.stop()
    bridge.stop()                                  # idempotent, and quiet
    with pytest.raises(RuntimeError, match="has been stopped.*build a new FmuTcpBridge"):
        bridge.start()
    with pytest.raises(RuntimeError, match="has been stopped"):
        with bridge:
            pass
    assert bridge._thread is None or not bridge._thread.is_alive()


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


def test_stop_ends_a_worker_parked_on_a_live_connection(monkeypatch):
    """``stop()`` used to return in 0.00 s with the worker still parked.

    The worker must be *parked* when ``stop()`` runs, or the test proves
    nothing: stopped the moment the hello reply arrives, the worker is
    usually still between its ``sendall`` and its loop's stop-flag check,
    and exits on the flag without needing the ``shutdown`` that ``stop()``
    exists to perform -- a ``stop()`` that skipped it survived this test
    about half the time.  So ``stop()`` waits until the worker has entered
    its read of the *next* frame, which it does only after that flag
    check; from there nothing but a shutdown (or the five-minute idle
    budget) ends the read."""
    reads = []
    reading_next_frame = threading.Event()
    real_recv_raw = tcp_bridge.recv_raw

    def counting_recv_raw(conn, **kw):
        # The test's own client reads through the same module function.
        if threading.current_thread().name == "maddening-fmu-conn":
            reads.append(threading.current_thread())
            if len(reads) >= 2:               # the frame after the hello
                reading_next_frame.set()
        return real_recv_raw(conn, **kw)

    monkeypatch.setattr(tcp_bridge, "recv_raw", counting_recv_raw)
    _, bridge = _bridge(_shared_graph())
    bridge.start()
    holder = socket.create_connection(_endpoint(bridge), timeout=5)
    try:
        holder.settimeout(5)
        send_message(holder, {"op": "hello"})
        assert recv_message(holder)["ok"]
        workers = list(bridge._live_workers)
        assert workers, "the worker should be serving the connection"
        assert reading_next_frame.wait(timeout=20.0), "the worker never read again"
        assert set(reads) == set(workers)
        bridge.stop()
        assert not [w for w in workers if w.is_alive()], (
            "stop() must not leave a connection worker behind"
        )
        assert not bridge._busy.locked()
    finally:
        holder.close()


# ---------------------------------------------------------- frame deadline

_FRAME_BUDGET = 0.5
"""The frame budget the deadline tests run under.  The idle and handshake
budgets are set far above it, so a slot released near this value was
released by the frame deadline and by nothing else."""


@pytest.fixture
def frame_budget_only(monkeypatch):
    monkeypatch.setattr(tcp_bridge, "_FRAME_TIMEOUT", _FRAME_BUDGET)
    monkeypatch.setattr(tcp_bridge, "_IDLE_TIMEOUT", 60.0)
    monkeypatch.setattr(tcp_bridge, "_HANDSHAKE_TIMEOUT", 60.0)


def _holder(bridge):
    """A connection that has said hello and so holds the instance slot."""
    sock = socket.create_connection(_endpoint(bridge), timeout=10)
    sock.settimeout(10)
    send_message(sock, {"op": "hello"})
    assert recv_message(sock)["ok"]
    assert bridge._busy.locked()
    return sock


def _released_after(bridge, while_waiting=lambda: None, limit=10.0):
    """Seconds until the instance slot is free, calling ``while_waiting``
    every 0.1 s meanwhile; ``inf`` if it is still held after ``limit``."""
    t0 = time.monotonic()
    while bridge._busy.locked():
        if time.monotonic() - t0 > limit:
            return float("inf")
        while_waiting()
        time.sleep(0.1)
    return time.monotonic() - t0


def test_a_peer_that_stalls_mid_frame_is_dropped_at_the_frame_deadline(frame_budget_only):
    """A frame announced, begun and abandoned.  The deadline was checked
    only *between* reads, so the read that was waiting for the rest of
    the frame ran on the socket's idle timeout, and the slot was held for
    the idle budget (five minutes shipped; 60 s here), not the frame's."""
    with _bridge(_shared_graph())[1] as bridge:
        holder = _holder(bridge)
        try:
            holder.sendall(struct.pack(">I", 100) + b"{")    # 1 of 100 bytes
            took = _released_after(bridge)
            assert 0.8 * _FRAME_BUDGET <= took < 10.0, took
            assert _hello(bridge)["ok"]
        finally:
            holder.close()


def test_a_peer_that_dribbles_a_frame_is_dropped_at_the_frame_deadline(frame_budget_only):
    """One byte every 0.1 s renews any per-read timeout for ever; only a
    deadline over the whole frame ends it."""
    with _bridge(_shared_graph())[1] as bridge:
        holder = _holder(bridge)
        try:
            holder.sendall(struct.pack(">I", 1000))

            def dribble():
                try:
                    holder.sendall(b" ")
                except OSError:
                    pass                          # the bridge has hung up

            took = _released_after(bridge, while_waiting=dribble)
            assert 0.8 * _FRAME_BUDGET <= took < 10.0, took
        finally:
            holder.close()


def test_a_frame_that_arrives_in_pieces_within_its_budget_is_served(frame_budget_only):
    """The deadline covers the frame, not each piece: a frame split across
    reads and finished inside the budget is served, and the connection
    then waits on the idle budget -- longer than the frame's -- for the
    next one."""
    with _bridge(_shared_graph())[1] as bridge:
        holder = _holder(bridge)
        try:
            body = b'{"op": "hello"}'
            holder.sendall(struct.pack(">I", len(body)) + body[:5])
            time.sleep(_FRAME_BUDGET / 5)
            holder.sendall(body[5:])
            assert recv_message(holder)["ok"]
            time.sleep(2 * _FRAME_BUDGET)             # idle between frames
            send_message(holder, {"op": "hello"})
            assert recv_message(holder)["ok"]
            assert bridge._busy.locked()
        finally:
            holder.close()


def test_a_frame_read_restores_the_socket_timeout():
    """The frame deadline lowers the socket's timeout for each read; the
    connection's own budget must be back in place afterwards, or the reply
    that follows is sent under whatever was left of the frame's."""
    a, b = socket.socketpair()
    with a, b:
        b.settimeout(7.0)
        body = b'{"op": "hello"}'
        a.sendall(struct.pack(">I", len(body)) + body)
        assert tcp_bridge.recv_raw(b, frame_timeout=30.0) == (False, body)
        assert b.gettimeout() == 7.0
        a.sendall(struct.pack(">I", 10) + b"12")               # stalls mid-frame
        with pytest.raises(socket.timeout, match="still incomplete after 2 bytes"):
            tcp_bridge.recv_raw(b, frame_timeout=0.2)
        assert b.gettimeout() == 7.0


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
