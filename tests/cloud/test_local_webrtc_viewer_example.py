"""The local WebRTC viewer example streams frames a viewer decodes.

``maddening.examples.cloud.streaming.09_local_webrtc_viewer`` runs a
``SelkiesSession`` on 127.0.0.1 fed by a small simulation.  Two runs of
it, each in its own process:

* ``--self-check``: its own GStreamer peer decodes the stream and the
  example checks the frames are the ones the simulation pushed;
* the viewer mode, as a user runs it: the page it writes is read while
  it runs, the GStreamer peer of ``test_webrtc_loopback.py`` connects
  with the URL in that page (the browser's part), decodes frames of the
  stream's size, and Ctrl-C stops the example cleanly and removes the
  page, which holds the session token.

GStreamer, its WebRTC plugins and PyGObject are not pip-installable, so
this module skips where they are absent and runs in the container built
from ``docker/Dockerfile.streaming-test``, which the ``test-streaming``
job in ``.github/workflows/ci.yml`` runs (the header of
``test_webrtc_loopback.py`` says how to run it on a developer machine).
``MADDENING_REQUIRE_STREAMING_TESTS=1`` turns the skip into a failure.
"""

from __future__ import annotations

import html
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

_REQUIRED_ENV = "MADDENING_REQUIRE_STREAMING_TESTS"


def _require_or_skip(reason: str) -> None:
    if os.environ.get(_REQUIRED_ENV):
        pytest.fail(f"{_REQUIRED_ENV} is set but {reason}", pytrace=False)
    pytest.skip(reason, allow_module_level=True)


try:
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstWebRTC", "1.0")
    gi.require_version("GstSdp", "1.0")
    from gi.repository import Gst  # noqa: F401 - only whether it imports
except (ImportError, ValueError) as _exc:
    _require_or_skip(
        f"needs PyGObject with the GStreamer 1.0, GstSdp and GstWebRTC typelibs "
        f"({_exc}); run it in the docker/Dockerfile.streaming-test container")

# The browser's part: the peer that signals as webrtc_client.html does.
# Importing the module also checks the elements both ends need (it skips,
# or fails under the variable, when one is missing).
from tests.cloud import test_webrtc_loopback as loopback  # noqa: E402

EXAMPLE = "maddening.examples.cloud.streaming.09_local_webrtc_viewer"
#: The example's default frame size.
WIDTH, HEIGHT = 640, 360
#: The process's start-up (jax, the first compile, GStreamer) and the
#: self-check's run, measured in the container on three cores: 3 s in all.
RUN_TIMEOUT_S = 180.0
#: Decoded frames the viewer-mode peer must see.
FRAMES_WANTED = 10


def _env() -> dict[str, str]:
    env = dict(os.environ)
    env.update({"JAX_PLATFORMS": "cpu", "PYTHONUNBUFFERED": "1"})
    return env


def test_the_self_check_decodes_the_frames_the_simulation_pushed(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", EXAMPLE, "--self-check"], cwd=tmp_path, env=_env(),
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=RUN_TIMEOUT_S)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output[-4000:]
    assert "Self-check passed." in result.stdout, output[-4000:]
    assert re.search(r"every decoded frame is within [\d.]+ .* of a frame pushed",
                     result.stdout), result.stdout


def _lines(proc: subprocess.Popen) -> queue.Queue:
    """The process's output, line by line, read on a thread."""
    lines: queue.Queue = queue.Queue()

    def read() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    threading.Thread(target=read, daemon=True).start()
    return lines


def test_the_viewer_page_connects_and_ctrl_c_stops_the_stream_and_removes_it(tmp_path):
    proc = subprocess.Popen(
        [sys.executable, "-m", EXAMPLE, "--seconds", "300"], cwd=tmp_path, env=_env(),
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True)
    seen: list[str] = []
    peer = None
    try:
        lines = _lines(proc)
        deadline = time.monotonic() + RUN_TIMEOUT_S
        uri = None
        while uri is None and time.monotonic() < deadline:
            try:
                line = lines.get(timeout=1.0)
            except queue.Empty:
                continue
            assert line is not None, f"the example exited {proc.wait()}: {''.join(seen)}"
            seen.append(line)
            found = re.search(r"(file://\S+webrtc_viewer\.html)", line)
            uri = found.group(1) if found else None
        assert uri is not None, f"no viewer page was announced: {''.join(seen)}"
        page = Path(urllib.request.url2pathname(urllib.parse.urlsplit(uri).path))
        assert (page.stat().st_mode & 0o777) == 0o600, oct(page.stat().st_mode)
        text = page.read_text(encoding="utf-8")
        assert "iceServers: []" in text and "stun:" not in text
        value = re.search(r'id="signaling-url" type="text"\s+value="([^"]*)"', text)
        assert value, text[:2000]
        url = html.unescape(value.group(1))
        assert url.startswith("ws://127.0.0.1:") and "token=" in url, url

        peer = loopback.Peer(url)
        peer.connect()
        loopback._wait(peer.answer, loopback.ANSWER_TIMEOUT_S, "the answer", peer)
        loopback._wait(peer.ice_connected, loopback.ICE_TIMEOUT_S, "ICE", peer)
        frames_by = time.monotonic() + loopback.FRAMES_TIMEOUT_S
        while len(peer.frames) < FRAMES_WANTED and time.monotonic() < frames_by:
            time.sleep(0.1)
        frames = list(peer.frames)
        assert len(frames) >= FRAMES_WANTED, (
            f"{len(frames)} frames decoded; errors {peer.errors + peer.pipeline_errors()}")
        assert {(w, h) for w, h, _mean, _spread in frames} == {(WIDTH, HEIGHT)}
        # The picture is the simulation's: not one flat colour.
        assert max(spread for *_rest, spread in frames) > 5.0, frames[:3]
        peer.close()
        peer = None

        proc.send_signal(signal.SIGINT)                       # Ctrl-C
        code = proc.wait(timeout=60)
        while (line := lines.get(timeout=10)) is not None:
            seen.append(line)
        output = "".join(seen)
        assert code == 0, output[-4000:]
        assert "Stopped after" in output and "Traceback" not in output, output[-4000:]
        assert not page.exists() and not page.parent.exists(), page
    finally:
        if peer is not None:
            peer.close()
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=30)

