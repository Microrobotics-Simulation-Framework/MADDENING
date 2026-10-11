"""A WebRTC viewer connected to a ``SelkiesSession`` decodes the frames pushed into it.

Every test here runs a real ``SelkiesSession`` (GStreamer ``x264enc`` and
``webrtcbin``, the authenticated signaling WebSocket) and connects a real
WebRTC peer to it over the loopback interface.  The peer is a second
GStreamer ``webrtcbin`` that behaves as the browser client
``maddening/api/static/webrtc_client.html`` does: it adds a receive-only
video transceiver, creates the OFFER, sends it as
``{"type": "sdp", "sdp": {"type": "offer", "sdp": ...}}``, trickles its
ICE candidates as ``{"type": "ice", "candidate": {...}}``, applies the
answer and the server's candidates it receives in the same shapes, and
decodes the H.264 it is sent (``rtph264depay ! avdec_h264``).  Before
the fix this test module pins, the server never answered: a browser
authenticated and then waited for ever, with no video.

What is NOT covered here: a GPU encoder (``x264enc`` is software), a
browser (the peer is GStreamer, not Chrome or Firefox), and NAT
traversal (both ends are on 127.0.0.1; no STUN or TURN server is
contacted).

GStreamer, its WebRTC plugins and PyGObject are not pip-installable, so
the module skips where they are absent (the ordinary CI lanes and a
plain virtualenv) and runs in the container built from
``docker/Dockerfile.streaming-test``, which the ``test-streaming`` job in
``.github/workflows/ci.yml`` runs.  The same loop on a developer machine,
from the repository root::

    docker build -f docker/Dockerfile.streaming-test -t maddening-streaming-test .
    docker run --rm --network none --cpuset-cpus=0-3 --memory=8g \\
      -v "$PWD":/src:ro maddening-streaming-test \\
      python3 -m pytest tests/cloud/test_webrtc_loopback.py -q -p no:cacheprovider

``--network none`` leaves the container only its loopback interface, so
nothing here can reach a network.  ``MADDENING_REQUIRE_STREAMING_TESTS=1``
(set by the CI job) turns a missing dependency into a failure instead of
a skip, so the one job that can run these tests cannot pass by skipping
them.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
import urllib.parse
from typing import Any, Optional

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
    from gi.repository import Gst, GstSdp, GstWebRTC
except (ImportError, ValueError) as _exc:
    _require_or_skip(
        f"needs PyGObject with the GStreamer 1.0 and GstWebRTC typelibs ({_exc}); "
        "run it in the docker/Dockerfile.streaming-test container"
    )

try:
    from websockets.sync.client import connect as _ws_connect
except ImportError as _exc:  # websockets < 11 has no threading client
    _require_or_skip(f"needs websockets >= 11 for its threading client ({_exc})")

Gst.init(None)
_MISSING = [name for name in ("appsrc", "videoconvert", "x264enc", "rtph264pay", "webrtcbin",
                              "rtph264depay", "h264parse", "avdec_h264", "appsink")
            if Gst.ElementFactory.find(name) is None]
if _MISSING:
    _require_or_skip(f"GStreamer elements {_MISSING} are not installed")

import numpy as np  # noqa: E402

from maddening.cloud.streaming import StreamConfig  # noqa: E402
from maddening.cloud.selkies_session import SelkiesSession  # noqa: E402

SECRET = "loopback-test-secret"

#: Small frames keep x264 and the decoder cheap on two CI cores; both
#: sides are multiples of 16, so no macroblock is padded.
WIDTH, HEIGHT, FPS = 320, 240, 30

#: Distinct solid colours, pushed in order, each for ``FRAMES_PER_COLOUR``
#: frames.  Every pair differs by at least 128 in some channel, so a
#: decoded frame can only be "near" the colour that was pushed.
PALETTE = np.array([
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0),
], dtype=np.float64)
FRAMES_PER_COLOUR = 6

#: How far a decoded frame's mean colour may sit from the colour pushed
#: (per channel, on 0-255).  The path is RGBA -> I420 (BT.601, limited
#: range: about one level of rounding per channel each way) -> H.264 ->
#: I420 -> RGB; a solid frame has no detail for the encoder to lose, so
#: what is left is the colour-space round trip and the quantiser's DC
#: error.  The value is generous next to that and far below half the
#: palette's spacing (64), so it cannot accept the wrong colour.
COLOUR_TOLERANCE = 24.0
#: A solid frame decodes solid: the spread of one decoded frame's pixels
#: about their mean.  A frame that was not decoded from a keyframe (grey
#: or smeared blocks) fails this.
UNIFORMITY_TOLERANCE = 12.0

#: How many decoded frames the peer must see.
FRAMES_WANTED = 20

#: Timeouts, each for one step on a loopback connection.  Measured in
#: the container on four cores, the steps take: answer < 0.5 s, ICE
#: connected < 1 s, twenty frames < 2 s (see the pull request for the
#: three runs); each limit leaves several times that.
ANSWER_TIMEOUT_S = 10.0
ICE_TIMEOUT_S = 15.0
FRAMES_TIMEOUT_S = 30.0
#: How long a rejected client is watched for a message it must not get.
SILENCE_S = 3.0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _solid_rgba(rgb) -> bytes:
    frame = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    frame[..., :3] = np.asarray(rgb, dtype=np.uint8)
    frame[..., 3] = 255
    return frame.tobytes()


def _mid_of(sdp_text: str, mline: int) -> Optional[str]:
    """The ``a=mid`` of m-line ``mline`` in ``sdp_text``, if it has one."""
    index = -1
    for line in sdp_text.splitlines():
        if line.startswith("m="):
            index += 1
        elif index == mline and line.startswith("a=mid:"):
            return line[len("a=mid:"):].strip()
    return None


class Peer:
    """A WebRTC receiver that signals the way ``webrtc_client.html`` does.

    ``webrtcbin`` with one receive-only H.264 transceiver offering payload
    type ``payload_type`` (a browser's are dynamic and rarely the 96 a
    GStreamer payloader defaults to, so the default here is not 96
    either).  Decoded frames are reduced to ``(width, height, mean RGB,
    spread)`` and kept in :attr:`frames`.
    """

    def __init__(self, url: str, *, headers: Optional[dict] = None,
                 payload_type: int = 102) -> None:
        self.url = url
        self.headers = headers
        self.frames: list[tuple[int, int, np.ndarray, float]] = []
        self.received: list[dict] = []          # every message from the server
        self.sent_candidates: list[dict] = []   # the peer's own, as sent
        self.answer = threading.Event()
        self.ice_connected = threading.Event()
        self.closed = threading.Event()
        self.close_code: Optional[int] = None
        self.errors: list[str] = []
        self._offer_text = ""
        self._ws = None
        self._stack = contextlib.ExitStack()
        self._send_lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None

        self.pipe = Gst.Pipeline.new("peer")
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "peer-webrtc")
        self.webrtc.set_property("bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)
        self.pipe.add(self.webrtc)
        caps = Gst.Caps.from_string(
            "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,"
            f"payload={payload_type},packetization-mode=(string)1,"
            "profile-level-id=(string)42e01f")
        self.webrtc.emit("add-transceiver",
                         GstWebRTC.WebRTCRTPTransceiverDirection.RECVONLY, caps)
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("pad-added", self._on_pad_added)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_state)

    # -- signaling -------------------------------------------------------

    def connect(self) -> None:
        """Open the signaling socket, then offer (as the browser does on open)."""
        kwargs: dict[str, Any] = {"open_timeout": 5}
        if self.headers:
            kwargs["additional_headers"] = self.headers
        # As a context manager: websockets >= 14 deprecates a bare connect().
        self._ws = self._stack.enter_context(_ws_connect(self.url, **kwargs))
        self._reader = threading.Thread(target=self._read, name="peer-reader", daemon=True)
        self._reader.start()
        self.pipe.set_state(Gst.State.PLAYING)
        promise = Gst.Promise.new_with_change_func(self._on_offer_created, None)
        self.webrtc.emit("create-offer", None, promise)

    def send(self, message: dict) -> None:
        with self._send_lock:
            if self._ws is not None and not self.closed.is_set():
                self._ws.send(json.dumps(message))

    def _read(self) -> None:
        try:
            for raw in self._ws:
                message = json.loads(raw)
                self.received.append(message)
                if message.get("type") == "sdp":
                    self._on_answer(message["sdp"])
                elif message.get("type") == "ice":
                    cand = message["candidate"]
                    self.webrtc.emit("add-ice-candidate", cand["sdpMLineIndex"],
                                     cand["candidate"])
        except Exception as exc:  # noqa: BLE001 - recorded, the test reads it
            if not self.closed.is_set():
                self.errors.append(f"reader: {exc!r}")
        finally:
            self.close_code = getattr(self._ws, "close_code", None)
            self.closed.set()

    def _on_offer_created(self, promise, _user_data) -> None:
        reply = promise.get_reply()
        offer = reply.get_value("offer")
        self.webrtc.emit("set-local-description", offer, Gst.Promise.new())
        self._offer_text = offer.sdp.as_text()
        self.send({"type": "sdp", "sdp": {"type": "offer", "sdp": self._offer_text}})

    def _on_answer(self, description: dict) -> None:
        assert description["type"] == "answer", description
        ok, sdp = GstSdp.SDPMessage.new_from_text(description["sdp"])
        assert ok == GstSdp.SDPResult.OK
        answer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, sdp)
        self.webrtc.emit("set-remote-description", answer, Gst.Promise.new())
        self.answer.set()

    def _on_ice_candidate(self, _element, mline: int, candidate: str) -> None:
        # The browser's RTCIceCandidate.toJSON() shape.
        payload = {"candidate": candidate, "sdpMLineIndex": mline,
                   "sdpMid": _mid_of(self._offer_text, mline)}
        self.sent_candidates.append(payload)
        self.send({"type": "ice", "candidate": payload})

    # -- media -----------------------------------------------------------

    def _on_ice_state(self, element, _pspec) -> None:
        state = element.get_property("ice-connection-state")
        if state in (GstWebRTC.WebRTCICEConnectionState.CONNECTED,
                     GstWebRTC.WebRTCICEConnectionState.COMPLETED):
            self.ice_connected.set()

    def _on_pad_added(self, _element, pad) -> None:
        if pad.get_direction() != Gst.PadDirection.SRC:
            return
        decode = Gst.parse_bin_from_description(
            "queue ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=frames emit-signals=true sync=false",
            True)
        self.pipe.add(decode)
        decode.sync_state_with_parent()
        decode.get_by_name("frames").connect("new-sample", self._on_sample)
        pad.link(decode.get_static_pad("sink"))

    def _on_sample(self, sink) -> Gst.FlowReturn:
        sample = sink.emit("pull-sample")
        structure = sample.get_caps().get_structure(0)
        width = structure.get_value("width")
        height = structure.get_value("height")
        buffer = sample.get_buffer()
        ok, info = buffer.map(Gst.MapFlags.READ)
        if ok:
            try:
                pixels = np.frombuffer(info.data, dtype=np.uint8)[: width * height * 3]
                rgb = pixels.reshape(height, width, 3).astype(np.float64)
                mean = rgb.reshape(-1, 3).mean(axis=0)
                spread = float(np.abs(rgb - mean).mean())
                self.frames.append((width, height, mean, spread))
            finally:
                buffer.unmap(info)
        return Gst.FlowReturn.OK

    def pipeline_errors(self) -> list[str]:
        """Errors the peer's pipeline posted (no main loop runs here: polled)."""
        bus = self.pipe.get_bus()
        found = []
        while (message := bus.pop_filtered(Gst.MessageType.ERROR)) is not None:
            err, debug = message.parse_error()
            found.append(f"peer pipeline: {err.message} ({debug})")
        self.errors.extend(found)
        return found

    def close(self) -> None:
        self.closed.set()
        try:
            self._stack.close()
        except Exception:  # noqa: BLE001 - already closed by the server
            pass
        if self._reader is not None:
            self._reader.join(timeout=5)
        self.pipe.set_state(Gst.State.NULL)


def _wait(event: threading.Event, timeout: float, what: str, peer: Peer) -> float:
    start = time.monotonic()
    if not event.wait(timeout):
        raise AssertionError(
            f"{what} did not happen within {timeout} s; the peer received "
            f"{[m.get('type') for m in peer.received]} and recorded errors "
            f"{peer.errors + peer.pipeline_errors()}")
    return time.monotonic() - start


def _url(session: SelkiesSession, token: Optional[str] = None) -> str:
    """The signaling URL on 127.0.0.1, with ``?token=`` when given."""
    path = urllib.parse.urlsplit(session_info(session).signaling_url).path
    url = f"ws://127.0.0.1:{session_port(session)}{path}"
    return url if token is None else f"{url}?token={token}"


def session_info(session: SelkiesSession):
    return session._info  # noqa: SLF001 - the StreamInfo start() returned


def session_port(session: SelkiesSession) -> int:
    return session._signaling_port  # noqa: SLF001 - the port the test chose


@pytest.fixture
def session():
    """A started session on a free loopback port; stopped after the test."""
    s = SelkiesSession(secret=SECRET, signaling_port=_free_port(), bind_host="127.0.0.1")
    s.start(StreamConfig(width=WIDTH, height=HEIGHT, fps=FPS, bitrate_kbps=800,
                         ice_servers=[]))
    yield s
    s.stop()


def _push_until(session: SelkiesSession, done, timeout: float) -> int:
    """Push the palette, one colour per ``FRAMES_PER_COLOUR`` frames, at FPS."""
    pushed = 0
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        colour = PALETTE[(pushed // FRAMES_PER_COLOUR) % len(PALETTE)]
        session.update_framebuffer_cpu(_solid_rgba(colour), WIDTH, HEIGHT, "RGBA")
        pushed += 1
        time.sleep(1.0 / FPS)
    return pushed


def test_the_server_answers_the_offer():
    """The browser's offer gets an SDP answer back on the same socket."""
    s = SelkiesSession(secret=SECRET, signaling_port=_free_port(), bind_host="127.0.0.1")
    s.start(StreamConfig(width=WIDTH, height=HEIGHT, fps=FPS, bitrate_kbps=800,
                         ice_servers=[]))
    relayed: list = []
    s.set_input_handler(relayed.append)
    peer = Peer(_url(s, s.session_token))
    try:
        deadline = time.monotonic() + 5
        while True:  # the base starts its server on a thread and returns at once
            try:
                peer.connect()
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.1)
        _wait(peer.answer, ANSWER_TIMEOUT_S, "an SDP answer", peer)
    finally:
        peer.close()
        s.stop()
    assert not [m for m in relayed if m.get("type") in ("sdp", "ice")], relayed
