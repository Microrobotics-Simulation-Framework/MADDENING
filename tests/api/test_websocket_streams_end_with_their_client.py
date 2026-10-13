"""A state or render stream ends when its client leaves, so the server can
shut down; and an ordinary disconnect is not logged as an error.

``/ws/state``, ``/ws/state/binary`` and ``/ws/render`` learned that a client
had gone only when a send failed, and they send only when the simulation
has produced a new snapshot.  A client that left after ``POST /sim/stop``
was never noticed -- nor was the close uvicorn sends on shutdown -- so the
handler slept on and SIGINT never completed.  A client that left while the
simulation ran made the next send fail with uvicorn's ``RuntimeError``,
which every stream logged with a traceback as "WebSocket error".

The shutdown tests run a real uvicorn server in a child process and bound
the wait for it, so a regression fails here instead of hanging the run;
the child is killed by its own PID if it outlives the bound.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import socket
import subprocess
import sys
import time

import pytest

httpx = pytest.importorskip("httpx", reason="the shutdown tests drive a real server over HTTP")
pytest.importorskip("uvicorn", reason="the shutdown tests run the app under uvicorn")
ws_client = pytest.importorskip(
    "websockets.sync.client", reason="the shutdown tests open real WebSockets")

from maddening.api import server as server_module  # noqa: E402

STREAMS = ("/ws/state", "/ws/state/binary", "/ws/render")

#: How long shutdown may take once SIGINT is sent.  It measures about 0.3 s.
SHUTDOWN_BOUND_S = 15.0

_SERVER = r'''
import logging, sys, warnings
warnings.simplefilter("ignore")
# The streams log their ends at INFO; show them, so the test can read how
# each stream ended.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
import uvicorn
from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode


class TinyRenderer:
    """The interface /ws/render uses, without matplotlib."""
    width, height, fmt, content_type = 4, 4, "png", "image/png"

    def render(self, sim_time, snapshot):
        return b"frame"

    def set_format(self, fmt, quality=None):
        pass

    def reset(self):
        pass


gm = GraphManager()
gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
gm.compile()
app = SimulationServer({"BallNode": BallNode}, graph_manager=gm,
                       frame_renderer=TinyRenderer(), bind_host="127.0.0.1").create_app()
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]), log_level="info")
'''


class _Server:
    def __init__(self, tmp_path):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()
        script = tmp_path / "serve.py"
        script.write_text(_SERVER)
        self.log_path = tmp_path / "server.log"
        self._log = open(self.log_path, "w")
        self.proc = subprocess.Popen([sys.executable, str(script), str(self.port)],
                                     stdout=self._log, stderr=subprocess.STDOUT)
        self.base = f"http://127.0.0.1:{self.port}"
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                if httpx.get(self.base + "/healthz", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            if self.proc.poll() is not None:
                break
            time.sleep(0.1)
        self.close()
        raise AssertionError(f"the server did not start:\n{self.log_path.read_text()}")

    def post(self, path):
        resp = httpx.post(self.base + path, timeout=30)
        assert resp.status_code == 200, resp.text
        return resp

    def connect(self, path):
        # ``max_queue=None``: the client keeps reading frames it does not
        # consume, so its own close handshake is not held up behind them
        # (a bounded queue stalled it for the 10 s close timeout).
        return ws_client.connect(f"ws://127.0.0.1:{self.port}{path}", open_timeout=30,
                                 max_queue=None)

    def interrupt(self) -> float | None:
        """SIGINT, then the seconds shutdown took, or ``None`` if it outlived
        :data:`SHUTDOWN_BOUND_S` (the child is then killed)."""
        t0 = time.perf_counter()
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=SHUTDOWN_BOUND_S)
        except subprocess.TimeoutExpired:
            return None
        finally:
            self.close()
        return time.perf_counter() - t0

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self._log.close()

    def log(self) -> str:
        return self.log_path.read_text()


def test_streams_whose_clients_left_let_the_server_shut_down(tmp_path):
    """The render client leaves while the simulation runs (its next frame
    hits the closed socket: the traceback case), then the simulation stops
    and the two state clients leave (nothing is sent to them again: the
    hang).  SIGINT must then shut the server down, with no error logged."""
    server = _Server(tmp_path)
    try:
        server.post("/sim/start")
        # Context managers: websockets >= 17.1 deprecates a bare connect().
        with contextlib.ExitStack() as stack:
            clients = {path: stack.enter_context(server.connect(path)) for path in STREAMS}
            for client in clients.values():
                for _ in range(2):
                    client.recv(timeout=30)
            clients.pop("/ws/render").close()
            time.sleep(0.3)                     # a few frames' time, still running
            server.post("/sim/stop")
            time.sleep(0.3)                     # no snapshot is produced from here
            for client in clients.values():
                client.close()
        time.sleep(0.3)
        took = server.interrupt()
    finally:
        server.close()
    log = server.log()
    assert took is not None, f"shutdown outlived {SHUTDOWN_BOUND_S} s:\n{log[-3000:]}"
    assert "Traceback" not in log and "WebSocket error" not in log, log[-3000:]
    for path in STREAMS:
        assert f"WebSocket client disconnected from {path}" in log, log[-3000:]


def test_streams_with_their_clients_still_open_let_the_server_shut_down(tmp_path):
    """SIGINT with every client still connected and the simulation stopped:
    uvicorn closes the sockets, and each stream must notice that close."""
    server = _Server(tmp_path)
    try:
        server.post("/sim/start")
        with contextlib.ExitStack() as stack:
            clients = [stack.enter_context(server.connect(path)) for path in STREAMS]
            for client in clients:
                for _ in range(2):
                    client.recv(timeout=30)
            server.post("/sim/stop")
            time.sleep(0.3)
            took = server.interrupt()
    finally:
        server.close()
    assert took is not None, f"shutdown outlived {SHUTDOWN_BOUND_S} s:\n{server.log()[-3000:]}"


# ---------------------------------------------------------------------------
# The helpers, in process
# ---------------------------------------------------------------------------

class _FakeSocket:
    def __init__(self, messages, then=None):
        self._messages = list(messages)
        self._then = then
        self.client_state = self.application_state = None

    async def receive(self):
        if self._messages:
            return self._messages.pop(0)
        if self._then is not None:
            raise self._then
        await asyncio.sleep(3600)


def _run_receiver(sock):
    seen = []

    async def go():
        disconnected = asyncio.Event()
        await asyncio.wait_for(
            server_module._receive_until_disconnect(sock, seen.append, disconnected), 5)
        return disconnected.is_set()

    return asyncio.run(go()), seen


def test_the_receiver_applies_objects_ignores_the_rest_and_ends_on_disconnect():
    sock = _FakeSocket([
        {"type": "websocket.receive", "text": '{"type": "config", "fps": 10}'},
        {"type": "websocket.receive", "text": "[1, 2]"},            # not an object
        {"type": "websocket.receive", "text": "{not json"},
        {"type": "websocket.receive", "bytes": b"\x00\x01"},        # binary: ignored
        {"type": "websocket.receive", "text": '{"type": "reset"}'},
        {"type": "websocket.disconnect", "code": 1000},
        {"type": "websocket.receive", "text": '{"type": "after"}'},
    ])
    ended, seen = _run_receiver(sock)
    assert ended is True
    assert seen == [{"type": "config", "fps": 10}, {"type": "reset"}]


def test_a_message_the_stream_cannot_apply_does_not_end_the_receiver():
    applied = []

    def picky(msg):
        if msg.get("fps") == "fast":
            raise TypeError("fps must be a number")
        applied.append(msg)

    sock = _FakeSocket([
        {"type": "websocket.receive", "text": '{"fps": "fast"}'},
        {"type": "websocket.receive", "text": '{"fps": 5}'},
        {"type": "websocket.disconnect"},
    ])

    async def go():
        disconnected = asyncio.Event()
        await server_module._receive_until_disconnect(sock, picky, disconnected)
        return disconnected.is_set()

    assert asyncio.run(go()) is True
    assert applied == [{"fps": 5}]


@pytest.mark.parametrize("error", [RuntimeError("socket broke"), OSError("reset")])
def test_a_broken_connection_ends_the_receiver_and_marks_the_client_gone(error):
    ended, seen = _run_receiver(_FakeSocket([], then=error))
    assert ended is True and seen == []


def test_cancelling_the_receiver_marks_the_client_gone():
    async def go():
        disconnected = asyncio.Event()
        task = asyncio.create_task(server_module._receive_until_disconnect(
            _FakeSocket([]), lambda msg: None, disconnected))
        await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return disconnected.is_set()

    assert asyncio.run(go()) is True


def test_the_wait_between_frames_ends_when_the_client_leaves():
    async def go():
        disconnected = asyncio.Event()
        asyncio.get_running_loop().call_later(0.05, disconnected.set)
        t0 = time.perf_counter()
        await server_module._sleep_unless_disconnected(disconnected, 30.0)
        return time.perf_counter() - t0

    assert asyncio.run(go()) < 5.0


@pytest.mark.parametrize("exc, gone", [
    (RuntimeError("Unexpected ASGI message 'websocket.send', after sending "
                  "'websocket.close' or response already completed."), True),
    (type("ConnectionClosedOK", (Exception,), {})(), True),
    (type("ClientDisconnected", (OSError,), {})(), True),
    (RuntimeError("the renderer failed"), False),
    (ValueError("bad frame"), False),
])
def test_what_counts_as_the_client_having_left(exc, gone):
    assert server_module._client_left(_FakeSocket([]), exc, asyncio.Event()) is gone


def test_any_send_failure_after_the_client_left_is_its_leaving():
    disconnected = asyncio.Event()
    disconnected.set()
    assert server_module._client_left(_FakeSocket([]), ValueError("x"), disconnected)
