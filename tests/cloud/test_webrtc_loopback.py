"""A WebRTC viewer connected to a ``SelkiesSession`` decodes the frames pushed into it.

Every test here runs a real ``SelkiesSession`` (GStreamer ``x264enc`` and
``webrtcbin``, the authenticated signaling WebSocket) and connects a real
WebRTC peer to it inside one container.  The peer is a second GStreamer
``webrtcbin`` that behaves as the browser client
``maddening/api/static/webrtc_client.html`` does: it adds a receive-only
video transceiver, creates the OFFER, sends it as
``{"type": "sdp", "sdp": {"type": "offer", "sdp": ...}}``, trickles its
ICE candidates as ``{"type": "ice", "candidate": {...}}`` (the shape of
``RTCIceCandidate.toJSON()``), applies the answer and the server's
candidates it receives in the same shapes, and decodes the H.264 it is
sent (``rtph264depay ! avdec_h264``).  Its H.264 payload type is 102,
not the 96 a GStreamer payloader defaults to, as a browser's usually is.
A GStreamer peer rather than a pure-Python one (aiortc) keeps the test's
dependencies at what the session itself needs, and gives a peer that
decodes with a real H.264 decoder.

Before the fix these tests pin, the server never answered: a viewer
authenticated, sent its offer, and was sent nothing (and the offer was
handed to the input handler).

What is NOT covered here: a GPU encoder (``x264enc`` is software), a
browser (the peer is GStreamer, not Chrome or Firefox), and NAT
traversal (both ends share one address; no STUN or TURN server is
reached).

GStreamer, its WebRTC plugins and PyGObject are not pip-installable, so
this module skips where they are absent (the ordinary CI lanes, a plain
virtualenv) and runs in the container built from
``docker/Dockerfile.streaming-test``, which the ``test-streaming`` job in
``.github/workflows/ci.yml`` runs.  The same loop on a developer machine,
from the repository root::

    docker build -f docker/Dockerfile.streaming-test -t maddening-streaming-test .
    docker run --rm --network none --cap-add NET_ADMIN \\
      --cpuset-cpus=0-3 --memory=8g \\
      -v "$PWD":/src:ro maddening-streaming-test \\
      python3 -m pytest tests/cloud/test_webrtc_loopback.py -q -p no:cacheprovider

``--network none`` leaves the container no network at all.  libnice
(``webrtcbin``'s ICE agent) never gathers candidates on a loopback
interface, so the image's entry point adds a veth pair that stays inside
the container (192.0.2.1, a never-routed documentation address) for both
ends to meet on; that is what ``--cap-add NET_ADMIN`` is for.
``MADDENING_REQUIRE_STREAMING_TESTS=1`` (set by the CI job) turns a
missing dependency into a failure instead of a skip, so the one job that
can run these tests cannot pass by skipping them.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
from pathlib import Path
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
        f"needs PyGObject with the GStreamer 1.0, GstSdp and GstWebRTC typelibs "
        f"({_exc}); run it in the docker/Dockerfile.streaming-test container"
    )

try:
    from websockets.exceptions import ConnectionClosed
    from websockets.sync.client import connect as _ws_connect
except ImportError as _exc:  # websockets < 11 has no threading client
    _require_or_skip(f"needs websockets >= 11 for its threading client ({_exc})")

Gst.init(None)
_MISSING = [name for name in ("appsrc", "queue", "videoconvert", "x264enc", "rtph264pay",
                              "webrtcbin", "rtph264depay", "h264parse", "avdec_h264",
                              "appsink")
            if Gst.ElementFactory.find(name) is None]
if _MISSING:
    _require_or_skip(f"GStreamer elements {_MISSING} are not installed")

import numpy as np  # noqa: E402

from maddening.cloud._auth import generate_session_token  # noqa: E402
from maddening.cloud.selkies_session import SelkiesSession, _disable_upnp  # noqa: E402
from maddening.cloud.streaming import StreamConfig  # noqa: E402
from maddening.viz.backends.selkies_renderer import SelkiesRenderer  # noqa: E402
from maddening.viz.renderer import GraphInfo, Renderer  # noqa: E402
from maddening.warnings import PerformanceWarning  # noqa: E402

SECRET = "loopback-test-secret"

#: Small frames keep x264 and the decoder cheap on a CI runner's cores;
#: both sides are multiples of 16, so no macroblock is padded.
WIDTH, HEIGHT, FPS = 320, 240, 30
BITRATE_KBPS = 800

#: Solid colours, pushed in this order, each for ``FRAMES_PER_COLOUR``
#: frames.  Every pair differs by 255 in at least one channel.
PALETTE = np.array([
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (0, 255, 255), (255, 0, 255), (255, 255, 255), (0, 0, 0),
], dtype=np.float64)
FRAMES_PER_COLOUR = 6

#: How far a decoded frame's mean colour may sit from the colour pushed,
#: in the worst channel, on 0-255.  The path is RGBA -> I420 (limited
#: range, 4:2:0) -> H.264 at ``BITRATE_KBPS`` -> I420 -> RGB; a solid
#: frame has no detail to lose, so what is left is the colour-space round
#: trip and the quantiser's error on the flat block.  Measured in the
#: container: at most 6.  The limit is four times that and far below half
#: the palette's spacing (127.5), so no frame can match the wrong colour.
COLOUR_TOLERANCE = 24.0
#: A solid frame decodes solid: the mean absolute deviation of one
#: decoded frame's pixels from their mean (measured: 0.0).  A frame that
#: was not decoded from its references (grey or smeared blocks) fails it.
UNIFORMITY_TOLERANCE = 12.0

#: How many decoded frames the peer must see.
FRAMES_WANTED = 20

#: Timeouts for one step each.  Measured in the container on four cores
#: (GStreamer 1.24.2): the answer arrives about 0.2 s after the offer,
#: ICE connects about 0.1 s later, and twenty frames are decoded within
#: 1.5 s; the whole module takes 3 s, its slowest test 0.75 s.  Each
#: limit is many times its step, for a loaded CI runner.
ANSWER_TIMEOUT_S = 10.0
ICE_TIMEOUT_S = 15.0
FRAMES_TIMEOUT_S = 30.0
#: How long the viewer a newer one replaces has to be closed, and a
#: rejected client to be closed.
CLOSE_TIMEOUT_S = 10.0


def _solid_rgba(rgb) -> bytes:
    frame = np.empty((HEIGHT, WIDTH, 4), dtype=np.uint8)
    frame[..., :3] = np.asarray(rgb, dtype=np.uint8)
    frame[..., 3] = 255
    return frame.tobytes()


def _palette_colour(index: int) -> np.ndarray:
    """The colour of the ``index``-th frame pushed."""
    return PALETTE[(index // FRAMES_PER_COLOUR) % len(PALETTE)]


def _mid_of(sdp_text: str, mline: int) -> Optional[str]:
    """The ``a=mid`` of m-line ``mline`` in ``sdp_text``, if it has one."""
    index = -1
    for line in sdp_text.splitlines():
        if line.startswith("m="):
            index += 1
        elif index == mline and line.startswith("a=mid:"):
            return line[len("a=mid:"):].strip()
    return None


def _candidate_address(candidate: str) -> tuple[str, int, str]:
    """``(address, port, type)`` of an ICE candidate line (RFC 8839 §5.1)."""
    fields = candidate.split()
    return fields[4], int(fields[5]), fields[fields.index("typ") + 1]


class Peer:
    """A WebRTC receiver that signals the way ``webrtc_client.html`` does.

    ``webrtcbin`` with one receive-only H.264 transceiver.  Decoded frames
    are reduced to ``(width, height, mean RGB, spread)`` in
    :attr:`frames`, and kept whole (``height x width x 3`` RGB, ``uint8``)
    in :attr:`pixels` when ``keep_pixels`` is set; every message the
    server sent is in :attr:`received`.
    """

    def __init__(self, url: str, *, payload_type: int = 102,
                 extra_payload_types: tuple[int, ...] = (),
                 keep_pixels: bool = False) -> None:
        self.url = url
        self.frames: list[tuple[int, int, np.ndarray, float]] = []
        self.pixels: list[np.ndarray] = []
        self._keep_pixels = keep_pixels
        self.received: list[dict] = []
        self.sent_candidates: list[dict] = []
        self.answer = threading.Event()
        self.ice_connected = threading.Event()
        self.closed = threading.Event()
        self.close_code: Optional[int] = None
        self.errors: list[str] = []
        self._offer_text = ""
        self._ws: Any = None
        self._stack = contextlib.ExitStack()
        self._send_lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None
        self._decode: Any = None

        self.pipe = Gst.Pipeline.new("peer")
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "peer-webrtc")
        self.webrtc.set_property("bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)
        self.pipe.add(self.webrtc)
        # One H.264 format per payload type, in this order, in one m-line.
        caps = Gst.Caps.from_string("; ".join(
            "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,"
            f"payload={pt},packetization-mode=(string)1,"
            "profile-level-id=(string)42e01f"
            for pt in (payload_type, *extra_payload_types)))
        self.webrtc.emit("add-transceiver",
                         GstWebRTC.WebRTCRTPTransceiverDirection.RECVONLY, caps)
        # A browser maps no router port; nor does this peer, so an SSDP
        # search seen in the container can only be the session's.
        _disable_upnp(self.webrtc)
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("pad-added", self._on_pad_added)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_state)

    # -- signaling -------------------------------------------------------

    def connect(self) -> None:
        """Open the signaling socket, then offer (the browser does both on open)."""
        # As a context manager: websockets >= 14 deprecates a bare connect().
        self._ws = self._stack.enter_context(_ws_connect(self.url, open_timeout=5))
        self._reader = threading.Thread(target=self._read, name="peer-reader", daemon=True)
        self._reader.start()
        self.pipe.set_state(Gst.State.PLAYING)
        promise = Gst.Promise.new_with_change_func(self._on_offer_created, None)
        self.webrtc.emit("create-offer", None, promise)

    def send(self, message: dict) -> None:
        with self._send_lock:
            if self._ws is None or self.closed.is_set():
                return
            try:
                self._ws.send(json.dumps(message))
            except ConnectionClosed as exc:
                self.errors.append(f"send after close: {exc!r}")

    def _read(self) -> None:
        try:
            for raw in self._ws:
                message = json.loads(raw)
                self.received.append(message)
                if message.get("type") == "sdp":
                    self._on_answer(message["sdp"])
                elif message.get("type") == "ice":
                    candidate = message["candidate"]
                    self.webrtc.emit("add-ice-candidate", candidate["sdpMLineIndex"],
                                     candidate["candidate"])
        except Exception as exc:  # noqa: BLE001 - recorded; the tests read it
            self.errors.append(f"reader: {exc!r}")
        finally:
            self.close_code = self._ws.close_code
            self.closed.set()

    def _on_offer_created(self, promise, _user_data) -> None:
        reply = promise.get_reply()  # held: the offer belongs to it
        offer = reply.get_value("offer")
        self._offer_text = offer.sdp.as_text()
        self.webrtc.emit("set-local-description", offer, Gst.Promise.new())
        self.send({"type": "sdp", "sdp": {"type": "offer", "sdp": self._offer_text}})

    def _on_answer(self, description: dict) -> None:
        assert description["type"] == "answer", description
        ok, sdp = GstSdp.SDPMessage.new_from_text(description["sdp"])
        assert ok == GstSdp.SDPResult.OK
        answer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, sdp)
        self.webrtc.emit("set-remote-description", answer, Gst.Promise.new())
        self.answer.set()

    def _on_ice_candidate(self, _element, mline: int, candidate: str) -> None:
        payload = {"candidate": candidate, "sdpMLineIndex": mline,
                   "sdpMid": _mid_of(self._offer_text, mline)}
        self.sent_candidates.append(payload)
        self.send({"type": "ice", "candidate": payload})

    # -- media -----------------------------------------------------------

    def _on_ice_state(self, element, _pspec) -> None:
        if element.get_property("ice-connection-state") in (
                GstWebRTC.WebRTCICEConnectionState.CONNECTED,
                GstWebRTC.WebRTCICEConnectionState.COMPLETED):
            self.ice_connected.set()

    def _on_pad_added(self, _element, pad) -> None:
        if pad.get_direction() != Gst.PadDirection.SRC:
            return
        decode = Gst.parse_bin_from_description(
            "queue ! rtph264depay ! h264parse name=parse ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=RGB ! appsink name=frames emit-signals=true sync=false",
            True)
        self._decode = decode
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
                raw = np.frombuffer(info.data, dtype=np.uint8)[: width * height * 3]
                raw = raw.reshape(height, width, 3)
                rgb = raw.astype(np.float64)
                mean = rgb.reshape(-1, 3).mean(axis=0)
                if self._keep_pixels:
                    self.pixels.append(raw.copy())
                self.frames.append((width, height, mean, float(np.abs(rgb - mean).mean())))
            finally:
                buffer.unmap(info)
        return Gst.FlowReturn.OK

    def h264_profile(self) -> Optional[str]:
        """The H.264 profile of the stream received, as ``h264parse`` reads it."""
        if self._decode is None:
            return None
        caps = self._decode.get_by_name("parse").get_static_pad("src").get_current_caps()
        return None if caps is None else caps.get_structure(0).get_value("profile")

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
        with contextlib.suppress(Exception):  # already closed by the server
            self._stack.close()
        if self._reader is not None:
            self._reader.join(timeout=5)
        self.closed.set()
        self.pipe.set_state(Gst.State.NULL)


def _wait(event: threading.Event, timeout: float, what: str, peer: Peer) -> float:
    """Wait for ``event``; on a timeout, say what the peer saw."""
    start = time.monotonic()
    if not event.wait(timeout):
        raise AssertionError(
            f"{what} did not happen within {timeout} s; the peer received "
            f"{[m.get('type') for m in peer.received]} and recorded errors "
            f"{peer.errors + peer.pipeline_errors()}")
    return time.monotonic() - start


def _start(config: Optional[StreamConfig] = None) -> tuple[SelkiesSession, str]:
    """A started session on 127.0.0.1 and its signaling URL (no token)."""
    session = SelkiesSession(secret=SECRET, signaling_port=0, bind_host="127.0.0.1")
    info = session.start(config or StreamConfig(
        width=WIDTH, height=HEIGHT, fps=FPS, bitrate_kbps=BITRATE_KBPS, ice_servers=[]))
    parts = urllib.parse.urlsplit(info.signaling_url)
    return session, f"ws://127.0.0.1:{parts.port}{parts.path}"


@pytest.fixture
def started():
    session, url = _start()
    yield session, url
    session.stop()


@pytest.fixture
def peers():
    """Peers a test opens; closed after it, whatever happened."""
    opened: list[Peer] = []
    yield opened
    for peer in opened:
        peer.close()


def _peer(peers: list, url: str, token: Optional[str] = None) -> Peer:
    peer = Peer(url if token is None else f"{url}?token={token}")
    peers.append(peer)
    return peer


def _push_until(push, done, timeout: float) -> int:
    """Call ``push(i)`` for frame ``i = 0, 1, ...`` at ``FPS`` until ``done()``."""
    pushed = 0
    deadline = time.monotonic() + timeout
    while not done() and time.monotonic() < deadline:
        push(pushed)
        pushed += 1
        time.sleep(1.0 / FPS)
    return pushed


def _push_palette(session: SelkiesSession, *, as_array: bool = False):
    """Push palette frames as bytes, or as ``(HEIGHT, WIDTH, 4)`` uint8 arrays."""
    def push(index: int) -> None:
        frame = _solid_rgba(_palette_colour(index))
        if as_array:
            frame = np.frombuffer(frame, dtype=np.uint8).reshape(HEIGHT, WIDTH, 4)
        session.update_framebuffer_cpu(frame, WIDTH, HEIGHT, "RGBA")
    return push


def _assert_decoded_the_palette(peer: Peer) -> None:
    """At least ``FRAMES_WANTED`` frames, of the stream's size, showing what was pushed."""
    frames = list(peer.frames)
    assert len(frames) >= FRAMES_WANTED, (
        f"{len(frames)} frames decoded; errors {peer.errors + peer.pipeline_errors()}")
    nearest = []
    for i, (width, height, mean, spread) in enumerate(frames):
        assert (width, height) == (WIDTH, HEIGHT), f"frame {i} is {width}x{height}"
        assert spread <= UNIFORMITY_TOLERANCE, (
            f"frame {i} is not solid (mean deviation {spread:.1f}): not decoded cleanly")
        error = np.abs(PALETTE - mean).max(axis=1)
        k = int(error.argmin())
        assert error[k] <= COLOUR_TOLERANCE, (
            f"frame {i}: mean colour {mean.round(1)} is {error[k]:.1f} from the nearest "
            f"colour pushed, {PALETTE[k]}")
        nearest.append(k)
    # The content changes as it was pushed: runs of one colour, in palette
    # order (a step of two allows one colour lost to a dropped frame run).
    runs = [k for j, k in enumerate(nearest) if j == 0 or k != nearest[j - 1]]
    assert len(runs) >= 3, f"the decoded frames show only the colours {runs}"
    steps = [(b - a) % len(PALETTE) for a, b in zip(runs, runs[1:])]
    assert all(1 <= step <= 2 for step in steps), (
        f"the decoded colours {runs} are not in the order they were pushed")
    # The answer promises the viewer constrained baseline (42e01f), the one
    # profile every browser decodes; the stream must be what it promised.
    assert peer.h264_profile() == "constrained-baseline", peer.h264_profile()


def _server_remote_candidates(session: SelkiesSession) -> set[tuple[str, int, str]]:
    """``(address, port, type)`` of every remote candidate the server's ``webrtcbin`` holds."""
    webrtc = session._negotiator._webrtc  # noqa: SLF001 - the viewer's webrtcbin
    promise = Gst.Promise.new()
    webrtc.emit("get-stats", None, promise)
    promise.wait()
    stats = promise.get_reply()
    found = set()
    for i in range(stats.n_fields()):
        value = stats.get_value(stats.nth_field_name(i))
        if isinstance(value, Gst.Structure) and \
                value.get_value("type") == GstWebRTC.WebRTCStatsType.REMOTE_CANDIDATE:
            found.add((value.get_value("address"), value.get_value("port"),
                       value.get_value("candidate-type")))
    return found


# ---------------------------------------------------------------------------
# Negotiation
# ---------------------------------------------------------------------------


def test_the_server_answers_the_offer_with_the_payload_type_offered(started, peers):
    session, url = started
    relayed: list = []
    session.set_input_handler(relayed.append)
    peer = _peer(peers, url, session.session_token)
    peer.connect()
    _wait(peer.answer, ANSWER_TIMEOUT_S, "an SDP answer", peer)

    (answer,) = [m["sdp"] for m in peer.received if m.get("type") == "sdp"]
    assert answer["type"] == "answer"
    video = [line for line in answer["sdp"].splitlines() if line.startswith("m=video")]
    assert len(video) == 1 and video[0].split()[3:] == ["102"], video
    assert "a=sendonly" in answer["sdp"].splitlines()
    # The offer went to the negotiation, not to the input handler.
    assert relayed == []


def test_ice_candidates_go_both_ways_and_ice_connects(started, peers):
    session, url = started
    peer = _peer(peers, url, session.session_token)
    peer.connect()
    _wait(peer.ice_connected, ICE_TIMEOUT_S, "ICE connected", peer)

    # Server -> viewer: trickled after the answer, in the browser's shape.
    kinds = [m.get("type") for m in peer.received]
    assert kinds[0] == "sdp" and "ice" in kinds, kinds
    ice = [m["candidate"] for m in peer.received if m.get("type") == "ice"]
    for i, candidate in enumerate(ice):
        # An empty candidate is the end-of-candidates marker (the browser's
        # shape for it too); it can only come last.
        assert candidate["candidate"].startswith("candidate:") or (
            candidate["candidate"] == "" and i == len(ice) - 1), candidate
        assert candidate["sdpMLineIndex"] == 0
        assert candidate["sdpMid"] == _mid_of(peer._offer_text, 0)

    # Viewer -> server: the server added the viewer's candidates.  Had it
    # dropped them, ICE could still connect (the viewer's checks would show
    # it the viewer's address as a peer-reflexive candidate), so this reads
    # the server's own record: a host candidate at the address and port
    # the viewer sent.
    sent = {_candidate_address(c["candidate"]) for c in peer.sent_candidates
            if c["candidate"]}  # not the end-of-candidates marker
    assert sent, "the peer gathered no candidates"
    held = _server_remote_candidates(session)
    assert {c for c in sent if c[2] == "host"} & {c for c in held if c[2] == "host"}, (
        f"the server holds {held}, none of the viewer's host candidates {sent}")


def test_the_peer_decodes_the_frames_that_were_pushed(started, peers):
    session, url = started
    peer = _peer(peers, url, session.session_token)
    peer.connect()
    _push_until(_push_palette(session, as_array=True), lambda: len(peer.frames) >= FRAMES_WANTED,
                FRAMES_TIMEOUT_S)
    _assert_decoded_the_palette(peer)


def test_input_events_reach_the_input_handler_and_negotiation_never_does(started, peers):
    session, url = started
    relayed: list = []
    session.set_input_handler(relayed.append)
    peer = _peer(peers, url, session.session_token)
    peer.connect()
    _wait(peer.ice_connected, ICE_TIMEOUT_S, "ICE connected", peer)
    events = [{"type": "mousemove", "x": 10, "y": 20, "buttons": 1},
              {"type": "keydown", "key": "a"}]
    for event in events:
        peer.send(event)

    deadline = time.monotonic() + ANSWER_TIMEOUT_S
    while len(relayed) < len(events) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert peer.sent_candidates, "the peer sent no ICE candidate, so this proves nothing"
    assert relayed == events


# ---------------------------------------------------------------------------
# Who may connect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("credential", ["none", "wrong", "other session", "other secret"])
def test_an_unauthenticated_peer_is_closed_and_sent_nothing(started, peers, credential):
    session, url = started
    relayed: list = []
    session.set_input_handler(relayed.append)
    session_id = urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1]
    token = {"none": None,
             "wrong": "f" * 64,
             "other session": generate_session_token("ba9876543210", SECRET),
             "other secret": generate_session_token(session_id, "not-the-secret")}[credential]
    peer = _peer(peers, url, token)
    peer.connect()  # the handshake completes; the server then closes

    _wait(peer.closed, CLOSE_TIMEOUT_S, "the server closing the connection", peer)
    assert peer.close_code == 1008
    assert peer.received == []
    assert relayed == []
    assert session._negotiator._pipeline is None  # noqa: SLF001 - nothing was built


def test_a_second_viewer_replaces_the_first(started, peers):
    session, url = started
    first = _peer(peers, url, session.session_token)
    first.connect()
    _wait(first.answer, ANSWER_TIMEOUT_S, "the first viewer's answer", first)

    second = _peer(peers, url, session.session_token)
    second.connect()
    _wait(first.closed, CLOSE_TIMEOUT_S, "the first viewer being closed", first)
    assert first.close_code == 4000

    _push_until(_push_palette(session), lambda: len(second.frames) >= FRAMES_WANTED,
                FRAMES_TIMEOUT_S)
    _assert_decoded_the_palette(second)
    assert [m.get("type") for m in first.received].count("sdp") == 1


#: A session in a child process, fed the palette from before any viewer
#: connects (as a running simulation feeds it), until its stdin closes.
_FED_SESSION = """
import sys, threading, time
from tests.cloud import test_webrtc_loopback as m
session, url = m._start()
print(url, session.session_token, flush=True)
closed = threading.Event()
threading.Thread(target=lambda: (sys.stdin.read(), closed.set()), daemon=True).start()
index = 0
while not closed.is_set():
    session.update_framebuffer_cpu(m._solid_rgba(m._palette_colour(index)),
                                   m.WIDTH, m.HEIGHT, "RGBA")
    index += 1
    time.sleep(1.0 / m.FPS)
session.stop()
"""


def test_a_viewer_in_another_process_decodes_a_session_that_was_already_fed(peers):
    """The deployment's shape: the viewer is elsewhere, and frames flow before it offers.

    Every other test here runs the viewer in the session's own process
    and pushes frames only once it has connected; there the offer is
    applied within milliseconds, before the first frame is encoded.  A
    frame that reaches ``webrtcbin`` first made it answer with an
    inactive m-line (ICE connected, DTLS never did, no video), which is
    what a viewer of the cloud entry point got.
    """
    child = subprocess.Popen([sys.executable, "-c", _FED_SESSION],
                             cwd=Path(__file__).resolve().parents[2],
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None and child.stdin is not None
        url, token = child.stdout.readline().split()
        peer = _peer(peers, url, token)
        peer.connect()
        deadline = time.monotonic() + FRAMES_TIMEOUT_S
        while len(peer.frames) < FRAMES_WANTED and time.monotonic() < deadline:
            time.sleep(0.05)
        _assert_decoded_the_palette(peer)
        child.stdin.close()
        assert child.wait(CLOSE_TIMEOUT_S) == 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(CLOSE_TIMEOUT_S)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

_SESSION_THREADS = {"selkies-signaling", "selkies-glib"}


def test_stop_ends_the_glib_loop_the_pipeline_and_the_server(peers):
    session, url = _start()
    try:
        peer = _peer(peers, url, session.session_token)
        peer.connect()
        _wait(peer.ice_connected, ICE_TIMEOUT_S, "ICE connected", peer)
        pipeline = session._negotiator._pipeline  # noqa: SLF001 - checked after stop()
        assert pipeline is not None
        assert _SESSION_THREADS <= {t.name for t in threading.enumerate()}
    finally:
        session.stop()

    # Only the session's own Python threads are named here: GStreamer's
    # threads register as "Dummy-N" once they call into Python, and stay
    # listed after they end.
    assert not _SESSION_THREADS & {t.name for t in threading.enumerate() if t.is_alive()}
    assert pipeline.get_state(0)[1] == Gst.State.NULL
    _wait(peer.closed, CLOSE_TIMEOUT_S, "the viewer's socket being closed", peer)
    port = urllib.parse.urlsplit(url).port
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    assert not session.is_alive()
    session.stop()  # a second stop is harmless


def test_a_stopped_session_starts_again_and_answers(peers):
    session, url = _start()
    session.stop()
    info = session.start(StreamConfig(width=WIDTH, height=HEIGHT, fps=FPS,
                                      bitrate_kbps=BITRATE_KBPS, ice_servers=[]))
    try:
        parts = urllib.parse.urlsplit(info.signaling_url)
        peer = _peer(peers, f"ws://127.0.0.1:{parts.port}{parts.path}", session.session_token)
        peer.connect()
        _wait(peer.answer, ANSWER_TIMEOUT_S, "an SDP answer after a restart", peer)
    finally:
        session.stop()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_configured_ice_servers_reach_webrtcbin(peers):
    # 192.0.2.1 is the container's own test address: nothing listens on
    # 3478, so neither server is reached, and ICE connects on host
    # candidates regardless.
    session, url = _start(StreamConfig(
        width=WIDTH, height=HEIGHT, fps=FPS, bitrate_kbps=BITRATE_KBPS, ice_servers=[
            {"urls": "stun:192.0.2.1:3478"},
            {"urls": ["turn:192.0.2.1:3478?transport=udp"],
             "username": "user", "credential": "secret"},
        ]))
    try:
        peer = _peer(peers, url, session.session_token)
        peer.connect()
        _wait(peer.answer, ANSWER_TIMEOUT_S, "an SDP answer", peer)
        webrtc = session._negotiator._webrtc  # noqa: SLF001 - the viewer's webrtcbin
        assert webrtc.get_property("stun-server") == "stun://192.0.2.1:3478"
        # (How a credential that needs escaping is written into the URI is
        # tested on the conversion itself, in test_webrtc_signaling_messages.)
        assert webrtc.get_property("turn-server") == \
            "turn://user:secret@192.0.2.1:3478?transport=udp"
    finally:
        session.stop()


def test_a_frame_of_the_wrong_size_is_refused(started):
    session, _url = started
    with pytest.raises(ValueError, match="320x240"):
        session.update_framebuffer_cpu(b"\x00" * (160 * 120 * 4), 160, 120, "RGBA")
    with pytest.raises(ValueError, match="bytes"):
        session.update_framebuffer_cpu(b"\x00" * 10, WIDTH, HEIGHT, "RGBA")


def test_a_frame_is_measured_in_bytes_and_must_hold_unsigned_bytes(started):
    """``memoryview(pixels).nbytes`` against ``W*H*4``, and format ``B`` only.

    Before: ``len(pixels)`` was compared, so a ``float32`` array of
    ``W*H*4`` elements (four times the bytes) streamed garbage, and a
    correct ``(H, W, 4)`` ``uint8`` frame (``len`` = ``H``) was refused.
    """
    session, _url = started
    n = WIDTH * HEIGHT * 4
    # Accepted: every C-contiguous unsigned-byte buffer of exactly n bytes.
    for good in (bytes(n), bytearray(n), memoryview(bytes(n)),
                 np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8),
                 np.zeros(n, dtype=np.uint8)):
        session.update_framebuffer_cpu(good, WIDTH, HEIGHT, "RGBA")
    # Refused, naming what was given: wider items with the element count of a frame.
    with pytest.raises(ValueError, match=r"format 'f' with 4-byte items"):
        session.update_framebuffer_cpu(np.zeros(n, dtype=np.float32), WIDTH, HEIGHT, "RGBA")
    with pytest.raises(ValueError, match=r"format 'I' with 4-byte items"):
        session.update_framebuffer_cpu(memoryview(np.zeros(n, dtype=np.uint32)),
                                       WIDTH, HEIGHT, "RGBA")
    with pytest.raises(ValueError, match="holds 4 bytes"):
        session.update_framebuffer_cpu(np.zeros(1, dtype=np.uint32).view(np.uint8),
                                       WIDTH, HEIGHT, "RGBA")
    with pytest.raises(ValueError, match="C-contiguous"):
        session.update_framebuffer_cpu(np.zeros((HEIGHT, 2 * WIDTH, 4), dtype=np.uint8)[:, ::2],
                                       WIDTH, HEIGHT, "RGBA")
    with pytest.raises(ValueError, match="bytes-like"):
        session.update_framebuffer_cpu([0] * n, WIDTH, HEIGHT, "RGBA")


def test_start_on_a_started_session_raises_and_leaves_it_running(started, peers):
    session, url = started
    with pytest.raises(RuntimeError, match="already started"):
        session.start(StreamConfig(width=WIDTH, height=HEIGHT, fps=FPS,
                                   bitrate_kbps=BITRATE_KBPS, ice_servers=[]))
    assert session.is_alive()
    peer = _peer(peers, url, session.session_token)
    peer.connect()
    _wait(peer.answer, ANSWER_TIMEOUT_S, "the answer after a refused second start", peer)


def test_the_pipeline_caps_carry_the_configured_frame_rate():
    session = SelkiesSession(secret=SECRET, signaling_port=0, bind_host="127.0.0.1")
    session._config = StreamConfig(width=WIDTH, height=HEIGHT, fps=12,  # noqa: SLF001
                                   bitrate_kbps=BITRATE_KBPS, ice_servers=[])
    description = session._pipeline_description(102)  # noqa: SLF001
    assert "framerate=12/1" in description and "key-int-max=12 " in description, description


# ---------------------------------------------------------------------------
# Offers the server cannot answer, and messages it cannot read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("payload_type", [35, 63, 95])
def test_an_offer_with_h264_only_below_96_is_refused_with_a_message(started, peers, caplog,
                                                                     payload_type):
    """The payloader sends only 96-127; the viewer is told why and closed, never left waiting."""
    session, url = started
    peer = Peer(f"{url}?token={session.session_token}", payload_type=payload_type)
    peers.append(peer)
    with caplog.at_level("WARNING", logger="maddening.cloud.selkies_session"):
        peer.connect()
        _wait(peer.closed, CLOSE_TIMEOUT_S, "the close after a refusal", peer)
    assert f"a=rtpmap:{payload_type} H264" in peer._offer_text  # noqa: SLF001
    assert [m["type"] for m in peer.received] == ["error"], peer.received
    assert "96-127" in peer.received[0]["message"], peer.received
    assert peer.close_code == 4001, peer.close_code
    refusals = [r for r in caplog.records if "Refusing the viewer's offer" in r.getMessage()]
    assert len(refusals) == 1 and refusals[0].levelname == "WARNING", caplog.text
    assert not any(r.exc_info for r in caplog.records), "a refusal must not log a traceback"


@pytest.mark.parametrize("low", [35, 63, 95])
def test_an_offer_with_h264_below_96_beside_a_usable_one_is_answered_with_the_usable_one(
        started, peers, low):
    session, url = started
    peer = Peer(f"{url}?token={session.session_token}", payload_type=low,
                extra_payload_types=(102,))
    peers.append(peer)
    peer.connect()
    _wait(peer.answer, ANSWER_TIMEOUT_S, "the answer", peer)
    answer = next(m["sdp"]["sdp"] for m in peer.received if m["type"] == "sdp")
    video = next(line for line in answer.splitlines() if line.startswith("m=video"))
    assert video.split()[3:] == ["102"], video
    _push_until(_push_palette(session), lambda: len(peer.frames) >= FRAMES_WANTED,
                FRAMES_TIMEOUT_S)
    _assert_decoded_the_palette(peer)


def test_malformed_messages_and_a_raising_input_handler_do_not_end_the_stream(started, peers,
                                                                              caplog):
    session, url = started

    def handler(_message: dict) -> None:
        raise KeyError("the handler's own fault")

    session.set_input_handler(handler)
    peer = _peer(peers, url, session.session_token)
    peer.connect()
    push = _push_palette(session)
    _push_until(push, lambda: len(peer.frames) >= 5, FRAMES_TIMEOUT_S)
    with caplog.at_level("WARNING", logger="maddening.cloud.selkies_session"):
        peer.send({"type": ["sdp"]})
        peer._ws.send("[" * 200_000)  # noqa: SLF001 - deeper than json.loads goes
        peer.send({"type": "mouse", "x": 1})
        seen = len(peer.frames)
        _push_until(push, lambda: len(peer.frames) >= seen + FRAMES_WANTED, FRAMES_TIMEOUT_S)
    assert not peer.closed.is_set(), (peer.close_code, peer.errors)
    assert len(peer.frames) >= seen + FRAMES_WANTED, len(peer.frames)
    errors = [m for m in peer.received if m.get("type") == "error"]
    assert len(errors) == 2, peer.received
    assert "malformed signaling message" in caplog.text
    raised = [r for r in caplog.records if "input handler raised" in r.getMessage()]
    assert raised and raised[0].exc_info and raised[0].exc_info[0] is KeyError, caplog.text


# ---------------------------------------------------------------------------
# What the session sends on the network: no UPnP
# ---------------------------------------------------------------------------

_SSDP_GROUP, _SSDP_PORT = "239.255.255.250", 1900


def test_with_no_ice_servers_the_session_sends_no_ssdp_search(peers):
    """libnice's UPnP (on by default) multicasts SSDP M-SEARCH for a router; it is off.

    A listener joined to the SSDP group on both interfaces records every
    M-SEARCH while a viewer (its UPnP off, as every ``Peer``'s is)
    negotiates and watches.  The listener is shown to work by one M-SEARCH
    the test sends itself; only searches after the session starts count.
    Before the fix the session sent 12 within 4 s (WANIPConnection,
    WANPPPConnection).
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((_SSDP_GROUP, _SSDP_PORT))
    for interface in ("192.0.2.1", "127.0.0.1"):
        listener.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP,
                            socket.inet_aton(_SSDP_GROUP) + socket.inet_aton(interface))
    listener.settimeout(0.2)
    got: list[tuple[float, tuple[str, int], bytes]] = []
    stop = threading.Event()

    def listen() -> None:
        while not stop.is_set():
            with contextlib.suppress(socket.timeout):
                data, address = listener.recvfrom(4096)
                got.append((time.monotonic(), address, data.split(b"\r\n")[0]))

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    session = None
    try:
        probe.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton("192.0.2.1"))
        probe.sendto(b"M-SEARCH * HTTP/1.1\r\nST: test-probe\r\n\r\n", (_SSDP_GROUP, _SSDP_PORT))
        probe_port = probe.getsockname()[1]
        deadline = time.monotonic() + 5.0
        while probe_port not in [a[1] for _t, a, _l in got] and time.monotonic() < deadline:
            time.sleep(0.05)
        assert probe_port in [a[1] for _t, a, _l in got], f"the listener does not work: {got}"
        started_at = time.monotonic()
        session, url = _start()
        peer = _peer(peers, url, session.session_token)
        _disable_upnp(peer.webrtc)
        peer.connect()
        _wait(peer.ice_connected, ICE_TIMEOUT_S, "ICE connected", peer)
        _push_until(_push_palette(session), lambda: len(peer.frames) >= FRAMES_WANTED,
                    FRAMES_TIMEOUT_S)
        time.sleep(2.0)  # libnice searched at once, then every few hundred ms
    finally:
        stop.set()
        thread.join(5)
        listener.close()
        probe.close()
        if session is not None:
            session.stop()
    searches = [(a, line) for t, a, line in got
                if t >= started_at and a[1] != probe_port and b"M-SEARCH" in line]
    assert searches == [], f"SSDP searches sent with ice_servers=[]: {searches}"


def test_viewer_candidates_sent_before_the_offer_survive_the_offers_teardown(started):
    """``_offer`` tears down the previous pipeline; the viewer's early candidates are kept."""
    from maddening.cloud.selkies_session import _WebRTCNegotiator

    session, _url = started
    negotiator = _WebRTCNegotiator(session._pipeline_description,  # noqa: SLF001
                                   lambda appsrc, encoder: None)
    client = _RecordingClient()
    negotiator._connect(client)  # noqa: SLF001
    early = {"type": "ice", "candidate": {"candidate": "candidate:1 1 UDP 1 192.0.2.1 5000 typ host",
                                          "sdpMid": "video0"}}
    negotiator._remote_ice(client, early)  # noqa: SLF001
    try:
        negotiator._offer(client, {"type": "offer", "sdp": _OFFER})  # noqa: SLF001
        assert negotiator._webrtc is not None, client.closed  # noqa: SLF001
        assert negotiator._pending_remote == [early]  # noqa: SLF001
    finally:
        negotiator._teardown()  # noqa: SLF001


# ---------------------------------------------------------------------------
# Ordering inside the negotiator (direct calls, no peer)
# ---------------------------------------------------------------------------


class _RecordingClient:
    """Stands in for a signaling connection: records what would be sent."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.closed: list[tuple[int, str]] = []

    def send(self, message: dict) -> None:
        self.sent.append(message)

    def close(self, code: int, reason: str) -> None:
        self.closed.append((code, reason))


class _RecordingWebRTC:
    """Stands in for ``webrtcbin``: records the signals emitted on it."""

    def __init__(self) -> None:
        self.emitted: list[tuple] = []

    def emit(self, *args) -> None:
        self.emitted.append(args)


def _negotiator_with_a_viewer():
    from maddening.cloud.selkies_session import _WebRTCNegotiator

    negotiator = _WebRTCNegotiator(lambda payload_type: "", lambda appsrc, encoder: None)
    client, webrtc = _RecordingClient(), _RecordingWebRTC()
    negotiator._client, negotiator._webrtc = client, webrtc  # noqa: SLF001
    negotiator._mids = {"video0": 0}  # noqa: SLF001
    return negotiator, client, webrtc


def test_candidates_gathered_before_the_answer_is_sent_follow_it():
    """A browser rejects a candidate for a description it does not have yet.

    On a loopback run webrtcbin happens to gather after the answer is
    applied, so the end-to-end test cannot see this ordering broken; the
    negotiator's own steps are driven here in the order that would.
    """
    negotiator, client, webrtc = _negotiator_with_a_viewer()
    negotiator._local_ice(webrtc, 0, "candidate:1 1 UDP 1 192.0.2.1 5000 typ host")
    assert client.sent == []
    negotiator._answer_applied(webrtc, None, "v=0\r\n")
    assert [m["type"] for m in client.sent] == ["sdp", "ice"]
    assert client.sent[1]["candidate"] == {
        "candidate": "candidate:1 1 UDP 1 192.0.2.1 5000 typ host",
        "sdpMLineIndex": 0, "sdpMid": "video0"}


def test_viewer_candidates_that_arrive_before_the_offer_is_applied_are_applied_after_it():
    negotiator, client, webrtc = _negotiator_with_a_viewer()
    early = {"type": "ice", "candidate": {"candidate": "candidate:1 1 UDP 1 192.0.2.1 5000 typ host",
                                          "sdpMid": "video0"}}
    negotiator._remote_ice(client, early)
    assert webrtc.emitted == []
    negotiator._remote_description_set(webrtc, None)
    assert webrtc.emitted[0] == ("add-ice-candidate", 0,
                                 "candidate:1 1 UDP 1 192.0.2.1 5000 typ host")
    assert webrtc.emitted[1][0] == "create-answer"


#: An offer a browser could send: one receive-only H.264 m-line (PT 102).
_OFFER = "\r\n".join([
    "v=0", "o=- 1 0 IN IP4 0.0.0.0", "s=-", "t=0 0", "a=group:BUNDLE video0",
    "m=video 9 UDP/TLS/RTP/SAVPF 102", "c=IN IP4 0.0.0.0", "a=mid:video0", "a=recvonly",
    "a=rtcp-mux", "a=ice-ufrag:abcd", "a=ice-pwd:abcdefghijklmnopqrstuvwx",
    "a=fingerprint:sha-256 " + ":".join(["00"] * 32), "a=setup:actpass",
    "a=rtpmap:102 H264/90000", "a=fmtp:102 packetization-mode=1;profile-level-id=42e01f",
    ""])


class _NamedElements:
    """Stands in for the viewer's pipeline once the offer has been applied."""

    def get_by_name(self, name: str) -> str:
        return f"<{name}>"


def test_frames_are_let_into_the_pipeline_only_once_the_answer_is_applied():
    """A frame that reaches ``webrtcbin`` before the offer is applied spoils the answer.

    So the offer builds the pipeline without handing its ``appsrc`` over,
    and the applied answer hands it over.  (End to end, in another
    process: ``test_a_viewer_in_another_process_decodes_a_session_that_was_already_fed``.)
    """
    from maddening.cloud.selkies_session import _WebRTCNegotiator

    handed: list = []
    negotiator = _WebRTCNegotiator(
        lambda payload_type: ("appsrc name=src ! fakesink  identity name=encoder  "
                              "webrtcbin name=webrtc"),
        lambda appsrc, encoder: handed.append((appsrc, encoder)))
    client = _RecordingClient()
    negotiator._client = client  # noqa: SLF001 - the negotiator's GLib-thread state
    try:
        negotiator._offer(client, {"type": "offer", "sdp": _OFFER})  # noqa: SLF001
        assert negotiator._pipeline is not None  # noqa: SLF001 - the offer built it
        assert handed == [], "frames were let in before the offer was applied"
    finally:
        negotiator._teardown()  # noqa: SLF001
    assert handed == [(None, None)]

    negotiator, client, webrtc = _negotiator_with_a_viewer()
    handed.clear()
    negotiator._on_pipeline = lambda appsrc, encoder: handed.append((appsrc, encoder))  # noqa: SLF001
    negotiator._pipeline = _NamedElements()  # noqa: SLF001
    negotiator._answer_applied(webrtc, None, "v=0\r\n")  # noqa: SLF001
    assert [m["type"] for m in client.sent] == ["sdp"]
    assert handed == [("<src>", "<encoder>")]


# ---------------------------------------------------------------------------
# The viz backend, end to end
# ---------------------------------------------------------------------------


class _PaletteRenderer(Renderer):
    """Draws the palette: one solid colour per ``FRAMES_PER_COLOUR`` updates."""

    def __init__(self) -> None:
        self.updates = 0
        self.torn_down = False

    def setup(self, graph_info: GraphInfo) -> None:
        pass

    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        self.updates += 1

    def teardown(self) -> None:
        self.torn_down = True

    def read_framebuffer_cpu(self):
        return _solid_rgba(_palette_colour(self.updates - 1)), WIDTH, HEIGHT, "RGBA"


def test_selkies_renderer_streams_what_its_inner_renderer_draws(peers):
    inner = _PaletteRenderer()
    session = SelkiesSession(secret=SECRET, signaling_port=0, bind_host="127.0.0.1")
    renderer = SelkiesRenderer(inner, session, StreamConfig(
        width=WIDTH, height=HEIGHT, fps=FPS, bitrate_kbps=BITRATE_KBPS, ice_servers=[]))
    graph_info = GraphInfo(node_names=[], node_params={}, node_state_fields={},
                           edges=[], timestep=0.01)
    with pytest.warns(PerformanceWarning, match="CPU path"):
        renderer.setup(graph_info)
    try:
        parts = urllib.parse.urlsplit(renderer.stream_info.signaling_url)
        peer = _peer(peers, f"ws://127.0.0.1:{parts.port}{parts.path}", session.session_token)
        peer.connect()
        _push_until(lambda i: renderer.update(i / FPS, {}),
                    lambda: len(peer.frames) >= FRAMES_WANTED, FRAMES_TIMEOUT_S)
    finally:
        renderer.teardown()
    _assert_decoded_the_palette(peer)
    assert inner.torn_down and not session.is_alive()
    assert not _SESSION_THREADS & {t.name for t in threading.enumerate() if t.is_alive()}
