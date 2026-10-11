"""The cloud container's entry point streams its running simulation to a WebRTC viewer.

Each test runs ``python3 -m maddening.cloud.entrypoint`` the way the
container does -- a child process, with the environment the entry point
reads: the cloud config JSON (``MADDENING_CLOUD_CONFIG``, carrying the
stream's size and frame rate), the stream secret
(``MADDENING_STREAM_SECRET``), the bind host (``MADDENING_HOST``, the
image's ``0.0.0.0``) and the API token that bind demands.  A graph is
built and run through the API's REST graph and simulation routes on
loopback, and the GStreamer peer of ``test_webrtc_loopback`` connects to
the signaling port with a token derived from the secret and the session
id the entry point logs.

Before the fix these tests pin, the entry point built a
``SelkiesSession`` and never started it: nothing listened on the
signaling port, so no viewer ever received a frame.  Its server also had
no node types, so ``POST /graph/nodes`` was refused for every type.

The only routes requested are under ``/graph/`` and ``/sim/`` (see
:func:`_rest`); none under ``/cloud``, ``/surrogate`` or ``/ws``.

Like ``test_webrtc_loopback``, this module needs GStreamer, its WebRTC
plugins and PyGObject, so it skips outside the
``docker/Dockerfile.streaming-test`` container (where the
``test-streaming`` CI job runs it, with no network).  From the
repository root::

    docker build -f docker/Dockerfile.streaming-test -t maddening-streaming-test .
    docker run --rm --network none --cap-add NET_ADMIN \\
      --cpuset-cpus=0-3 --memory=8g \\
      -v "$PWD":/src:ro maddening-streaming-test \\
      python3 -m pytest tests/cloud/test_entrypoint_streams.py -q -p no:cacheprovider

Not covered: a provider's network (both ends are in one container), a
browser (the peer is GStreamer), and a GPU encoder (``x264enc``).
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional

import pytest

# The peer, and the skip (or, under MADDENING_REQUIRE_STREAMING_TESTS, the
# failure) when GStreamer is absent, come from the loopback module.
from tests.cloud import test_webrtc_loopback as loopback

import numpy as np

from maddening.cloud._auth import generate_session_token

SECRET = "entrypoint-stream-test-secret"
API_TOKEN = "entrypoint-stream-test-api-token"
#: The entry point's own defaults: the API's port, and the signaling port
#: docker/Dockerfile.cloud exposes.  Free inside a network-less container.
API_PORT = 8000
SIGNALING_PORT = 8443

#: The stream the cloud config asks for: small, so x264 and the peer's
#: decoder are cheap; both sides multiples of 16.
WIDTH, HEIGHT, FPS = 320, 240, 15
STREAM_CONFIG = {"width": WIDTH, "height": HEIGHT, "fps": FPS, "bitrate_kbps": 800,
                 "ice_servers": []}

#: The graph built through the REST routes: a damped spring released away
#: from its rest length, so its position swings while it runs.
SPRING = {"type": "SpringDamperNode", "name": "spring", "timestep": 0.01,
          "params": {"stiffness": 50.0, "damping": 0.5, "rest_length": 1.0,
                     "initial_position": 3.0}}

#: Timeouts.  Measured in the container on four cores: the entry point is
#: serving (JAX imported, the session started) within about 5 s; the answer,
#: ICE and the first frames then take as long as in test_webrtc_loopback.
STARTUP_TIMEOUT_S = 90.0
EXIT_TIMEOUT_S = 30.0
#: How long the viewer watches the running simulation.
WATCH_S = 3.0

#: A decoded frame "changed" where this share of its pixels moved by more
#: than ``PIXEL_STEP`` levels (0-255, any channel) against an earlier one.
PIXEL_STEP = 64
CHANGED_SHARE = 0.002

_STREAM_LINE = re.compile(r"WebRTC streaming started: (\d+)x(\d+) at (\d+) fps\. "
                          r"A viewer connects to ws://<this host>:(\d+)/signaling/([0-9a-f]+)")


class EntryPoint:
    """``python3 -m maddening.cloud.entrypoint`` in a child process, and its log."""

    def __init__(self, tmp_path) -> None:
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("MADDENING_", "RESUME_", "RUNPOD", "AWS_", "GOOGLE_"))}
        env.update({
            "HOME": str(tmp_path), "TMPDIR": str(tmp_path), "MPLCONFIGDIR": str(tmp_path),
            "JAX_PLATFORMS": "cpu", "PYTHONDONTWRITEBYTECODE": "1",
            "MADDENING_CLOUD_CONFIG": json.dumps({"stream_config": STREAM_CONFIG}),
            "MADDENING_STREAM_SECRET": SECRET,
            "MADDENING_HOST": "0.0.0.0",
            "MADDENING_PORT": str(API_PORT),
            "MADDENING_API_TOKEN": API_TOKEN,
        })
        self.lines: list[str] = []
        self.stream_line: Optional[re.Match] = None
        self._streaming = threading.Event()
        self.process = subprocess.Popen(
            [sys.executable, "-m", "maddening.cloud.entrypoint"], cwd=tmp_path, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        self._reader = threading.Thread(target=self._read, name="entrypoint-log", daemon=True)
        self._reader.start()
        self._wait_until_serving()

    def _read(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.lines.append(line.rstrip("\n"))
            match = _STREAM_LINE.search(line)
            if match:
                self.stream_line = match
                self._streaming.set()

    def log(self) -> str:
        return "\n".join(self.lines[-60:])

    def _wait_until_serving(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT_S
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(f"the entry point exited ({self.process.returncode}):\n"
                                     f"{self.log()}")
            with contextlib.suppress(OSError):
                if _rest("GET", "/graph/state")[0] == 200:
                    break
            time.sleep(0.25)
        else:
            raise AssertionError(f"the API did not answer within {STARTUP_TIMEOUT_S} s:\n"
                                 f"{self.log()}")
        if not self._streaming.wait(5.0):
            raise AssertionError(f"the entry point did not log a started stream:\n{self.log()}")

    @property
    def session_id(self) -> str:
        assert self.stream_line is not None
        return self.stream_line.group(5)

    def signaling_url(self, token: Optional[str]) -> str:
        assert self.stream_line is not None
        url = f"ws://127.0.0.1:{self.stream_line.group(4)}/signaling/{self.session_id}"
        return url if token is None else f"{url}?token={token}"

    def terminate(self) -> int:
        """SIGTERM; the exit status (raises when it does not exit in time)."""
        self.process.send_signal(signal.SIGTERM)
        status = self.process.wait(EXIT_TIMEOUT_S)
        self._reader.join(5)
        return status

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGTERM)
            try:
                self.process.wait(EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(EXIT_TIMEOUT_S)
        self._reader.join(5)
        if self.process.stdout is not None:
            self.process.stdout.close()


def _rest(method: str, path: str, body: Optional[dict] = None) -> tuple[int, Any]:
    """One request to the entry point's API on loopback: ``(status, JSON body)``."""
    assert path.startswith(("/graph/", "/sim/")), f"{path} is not a route this module may request"
    request = urllib.request.Request(
        f"http://127.0.0.1:{API_PORT}{path}", method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as reply:
            return reply.status, json.loads(reply.read() or b"null")
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"null")


@pytest.fixture
def entrypoint(tmp_path):
    process = EntryPoint(tmp_path)
    yield process
    process.close()


@pytest.fixture
def peers():
    opened: list = []
    yield opened
    for peer in opened:
        peer.close()


def _viewer(peers: list, url: str) -> "loopback.Peer":
    peer = loopback.Peer(url, keep_pixels=True)
    peers.append(peer)
    peer.connect()
    return peer


def _changed_share(a: np.ndarray, b: np.ndarray) -> float:
    """The share of pixels in which ``a`` and ``b`` differ by more than ``PIXEL_STEP``."""
    step = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return float((step > PIXEL_STEP).mean())


def _frames_after(peer, start: int, wanted: int, timeout: float) -> list[np.ndarray]:
    """``wanted`` frames decoded after the first ``start``, within ``timeout``."""
    deadline = time.monotonic() + timeout
    while len(peer.pixels) < start + wanted and time.monotonic() < deadline:
        time.sleep(0.05)
    frames = list(peer.pixels)[start:]
    assert len(frames) >= wanted, (
        f"{len(frames)} frames decoded of {wanted}; errors {peer.errors + peer.pipeline_errors()}")
    return frames


# ---------------------------------------------------------------------------
# What a viewer holding the secret receives
# ---------------------------------------------------------------------------


def test_a_viewer_with_the_secret_watches_the_simulation_run(entrypoint, peers):
    # The stream the config asked for, on the documented port.
    width, height, fps, port = (int(entrypoint.stream_line.group(i)) for i in range(1, 5))
    assert (width, height, fps, port) == (WIDTH, HEIGHT, FPS, SIGNALING_PORT)

    # The graph, built and started through the REST routes.
    status, body = _rest("POST", "/graph/nodes", SPRING)
    assert status == 201, body
    status, body = _rest("POST", "/sim/start")
    assert status == 200, body

    peer = _viewer(peers, entrypoint.signaling_url(
        generate_session_token(entrypoint.session_id, SECRET)))
    loopback._wait(peer.answer, loopback.ANSWER_TIMEOUT_S, "an SDP answer", peer)
    loopback._wait(peer.ice_connected, loopback.ICE_TIMEOUT_S, "ICE connected", peer)

    # Frames of the configured size, which change as the simulation runs.
    first = _frames_after(peer, 0, 1, loopback.FRAMES_TIMEOUT_S)[0]
    time.sleep(WATCH_S)
    running = _frames_after(peer, 1, int(FPS * WATCH_S / 2), loopback.FRAMES_TIMEOUT_S)
    assert {frame.shape for frame in running} == {(HEIGHT, WIDTH, 3)}, (
        {frame.shape for frame in running})
    # A white figure: the picture was drawn, not an empty buffer.
    assert first.mean() > 150, first.mean()
    shares = [_changed_share(first, frame) for frame in running]
    assert max(shares) > CHANGED_SHARE, (
        f"the frames did not change while the simulation ran: shares {shares[-5:]}")
    assert peer.h264_profile() == "constrained-baseline", peer.h264_profile()

    # Paused, the picture holds still: what changed was the simulation.
    status, body = _rest("POST", "/sim/pause")
    assert status == 200, body
    time.sleep(1.0)
    held = len(peer.pixels)
    paused = _frames_after(peer, held, int(FPS), loopback.FRAMES_TIMEOUT_S)
    paused_shares = [_changed_share(paused[0], frame) for frame in paused[1:]]
    assert max(paused_shares) <= CHANGED_SHARE, (
        f"the frames changed while the simulation was paused: shares {paused_shares}")


# ---------------------------------------------------------------------------
# Who may connect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("credential", ["none", "wrong", "other secret", "other session"])
def test_a_viewer_without_a_valid_token_is_closed_and_sent_nothing(entrypoint, peers,
                                                                     credential):
    token = {"none": None,
             "wrong": "f" * 64,
             "other secret": generate_session_token(entrypoint.session_id, "not-the-secret"),
             "other session": generate_session_token("ba9876543210", SECRET)}[credential]
    status, body = _rest("POST", "/graph/nodes", SPRING)
    assert status == 201, body
    status, body = _rest("POST", "/sim/start")
    assert status == 200, body

    peer = loopback.Peer(entrypoint.signaling_url(token), keep_pixels=True)
    peers.append(peer)
    peer.connect()  # the handshake completes; the server then closes
    loopback._wait(peer.closed, loopback.CLOSE_TIMEOUT_S,
                   "the server closing the connection", peer)
    assert peer.close_code == 1008
    assert peer.received == []
    time.sleep(1.0)
    assert peer.pixels == []


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


def test_sigterm_stops_the_stream_and_the_process_exits_cleanly(entrypoint, peers):
    status, body = _rest("POST", "/graph/nodes", SPRING)
    assert status == 201, body
    status, body = _rest("POST", "/sim/start")
    assert status == 200, body
    peer = _viewer(peers, entrypoint.signaling_url(
        generate_session_token(entrypoint.session_id, SECRET)))
    _frames_after(peer, 0, 1, loopback.FRAMES_TIMEOUT_S)

    status = entrypoint.terminate()

    assert status == 0, entrypoint.log()
    # The session closed the viewer's signaling socket, and nothing listens.
    loopback._wait(peer.closed, loopback.CLOSE_TIMEOUT_S, "the viewer's socket being closed",
                   peer)
    for port in (API_PORT, SIGNALING_PORT):
        with pytest.raises(OSError):
            socket.create_connection(("127.0.0.1", port), timeout=2).close()
    log = entrypoint.log()
    assert "Received SIGTERM" in log, log
    assert "Traceback" not in log, log
