#!/usr/bin/env python3
"""Watch a running simulation in a browser on this machine, over WebRTC.

No cloud account and no GPU: the stream is encoded on the CPU with
``x264enc``.  The example starts a ``SelkiesSession`` whose signaling
socket listens on ``127.0.0.1`` only, steps a small simulation (a
``HeatNode`` rod whose left end is driven by a sine, drawn as a heat
strip with its temperature profile underneath) and pushes one frame per
display frame into the session.  It writes a viewer page -- a copy of
``maddening/api/static/webrtc_client.html`` with the signaling URL and
the session token filled in -- prints its ``file://`` address, and runs
until Ctrl-C or ``--seconds``.  Open the address in a browser on the
same machine; the page connects by itself.

Nothing leaves the machine.  The session is given no STUN or TURN server
(``ice_servers=[]``) and the page's copy none either, so neither end
contacts anything to find its addresses.  The media still needs an
address other than loopback: libnice, the ICE agent inside GStreamer's
``webrtcbin``, never gathers candidates on a loopback interface.  On a
machine with a network interface the two ends meet on that interface's
own address, and the kernel delivers those packets locally.  The viewer
page holds the session token, a credential for this run only, so it is
written readable by you alone, and removed when the example stops.

``--self-check`` runs the same session headless: instead of a browser,
a GStreamer ``webrtcbin`` peer (a copy of the one in the repository's
``tests/cloud/test_webrtc_loopback.py``) signals as the browser page
does, completes ICE, decodes the H.264 it is sent, and the example
checks that the decoded frames are the frames the simulation pushed.
That mode is what is tested (CI's streaming job runs it), and it needs
no display.

What is covered, and what is not
--------------------------------
* Covered: the signaling protocol of the shipped browser page, token
  authentication, negotiation, ICE between two ends on one machine, and
  frames that decode to the pictures pushed (``--self-check``).  A real
  browser was not run by this example's tests; a ``SelkiesSession``
  stream was watched live in a real browser on 2026-10-12.
* Not covered: **a network between the two ends.**  A viewer outside a
  cloud provider's network needs a path for the media, which WebRTC sends
  over UDP and a provider may not forward (RunPod forwards only TCP).
  The 2026-10-12 session used TURN over TCP: a TURN server (coturn)
  running beside the viewer, reached from the pod through the session's
  own SSH connection (``ssh -R`` for the TURN port, ``-L`` for the API
  and the signaling socket), and named to the session as
  ``StreamConfig(ice_servers=[{"urls": "turn:127.0.0.1:3478?transport=tcp",
  "username": ..., "credential": ...}])``.  The entry point bound
  ``127.0.0.1`` on the pod, so nothing was exposed to the internet.
* Not covered: **a GPU encoder.**  There is none: ``x264enc`` encodes in
  software on every host, a GPU host included.

Requirements: GStreamer 1.x with its WebRTC plugins and PyGObject, which
pip cannot install (on Ubuntu 24.04: ``apt install python3-gi
gir1.2-gst-plugins-bad-1.0 gstreamer1.0-plugins-base
gstreamer1.0-plugins-good gstreamer1.0-plugins-bad
gstreamer1.0-plugins-ugly gstreamer1.0-nice gstreamer1.0-libav``, with
the system Python), and ``websockets`` 11 or later.  Without them the
example says what is missing and exits; or run it in the image the
streaming tests use, which has all of them (``docker/Dockerfile.streaming-test``
in the repository; its header says how to build and run it).

Usage
-----
    python -m maddening.examples.cloud.streaming.09_local_webrtc_viewer
    python -m maddening.examples.cloud.streaming.09_local_webrtc_viewer --seconds 60
    python -m maddening.examples.cloud.streaming.09_local_webrtc_viewer --self-check
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import secrets
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

# jax before GStreamer: in Ubuntu 24.04's GStreamer 1.24 a process that
# loaded GStreamer's library and only then imported jax aborted inside
# jaxlib (see maddening.cloud.selkies_session).  These imports load jax.
import jax.numpy as jnp
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode

INSTALL_HINT = """\
This example needs GStreamer with its WebRTC plugins, PyGObject and
websockets >= 11.  GStreamer and PyGObject are not pip-installable; on
Ubuntu 24.04, with the system Python 3.12:

    sudo apt install python3-gi gir1.2-gst-plugins-bad-1.0 \\
        gstreamer1.0-plugins-base gstreamer1.0-plugins-good \\
        gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \\
        gstreamer1.0-nice gstreamer1.0-libav

Or run it in the image the streaming tests use, from a checkout of the
repository:

    docker build -f docker/Dockerfile.streaming-test -t maddening-streaming-test .
    docker run --rm --network none --cap-drop ALL --cap-add NET_ADMIN \\
        -v "$PWD":/src:ro maddening-streaming-test \\
        python3 -m maddening.examples.cloud.streaming.09_local_webrtc_viewer --self-check

(The image's entry point gives the network-less container an interface
for ICE; that is what NET_ADMIN is for.  Inside it only --self-check
makes sense: no browser can reach it.)"""

#: The encoder's elements, and the ones the self-check's peer decodes with.
SERVER_ELEMENTS = ("appsrc", "queue", "videoconvert", "x264enc", "rtph264pay", "webrtcbin")
PEER_ELEMENTS = ("rtph264depay", "h264parse", "avdec_h264", "appsink")

# -- the simulation ----------------------------------------------------------

N_CELLS = 160
DX = 1e-3                      # cell size [m]
ALPHA = 1e-4                   # thermal diffusivity [m^2/s]
DT = 0.2 * DX ** 2 / ALPHA     # Fourier number 0.2: inside the stencil's limit [s]
STEPS_PER_FRAME = 100          # 0.2 s of simulated time a frame
PERIOD_S = 18.0                # period of the left end's sine [s]: 3 s on screen at 30 fps


def build_rod() -> GraphManager:
    """A rod at 20, its right end held at 20, its left end driven."""
    gm = GraphManager()
    gm.add_node(HeatNode("rod", DT, n_cells=N_CELLS, length=N_CELLS * DX,
                         thermal_diffusivity=ALPHA, initial_temperature=20.0))
    gm.add_external_input("rod", "left_temperature")
    gm.add_external_input("rod", "right_temperature")
    gm.compile()
    return gm


#: A dark-to-bright colour ramp (black, purple, red, orange, pale yellow).
_RAMP = np.array([(0, 0, 4), (87, 16, 110), (188, 55, 84), (249, 142, 9),
                  (252, 255, 164)], dtype=np.float64)


def render(temperature: np.ndarray, width: int, height: int) -> np.ndarray:
    """An ``(height, width, 4)`` RGBA frame: the rod as a heat strip over
    its temperature profile (0 to 100 degrees)."""
    t = np.clip(np.interp(np.linspace(0, len(temperature) - 1, width),
                          np.arange(len(temperature)), temperature) / 100.0, 0.0, 1.0)
    pos = t * (len(_RAMP) - 1)
    lo = np.minimum(pos.astype(int), len(_RAMP) - 2)
    colour = _RAMP[lo] + (pos - lo)[:, None] * (_RAMP[lo + 1] - _RAMP[lo])
    frame = np.full((height, width, 4), 40, dtype=np.uint8)     # a dark grey ground
    frame[..., 3] = 255
    strip = height // 2
    frame[:strip, :, :3] = colour.astype(np.uint8)[None, :, :]
    plot = height - strip
    rows = strip + ((1.0 - t) * (plot - 3)).astype(int) + 1
    for dy in (-1, 0, 1):
        frame[rows + dy, np.arange(width), :3] = 255
    return frame


class Simulation:
    """Steps the rod and renders a frame, ``STEPS_PER_FRAME`` steps at a time."""

    def __init__(self, width: int, height: int) -> None:
        self.gm = build_rod()
        self.width, self.height = width, height
        self.time = 0.0

    def next_frame(self) -> np.ndarray:
        left = 50.0 + 50.0 * np.sin(2 * np.pi * self.time / PERIOD_S)
        ext = {"rod": {"left_temperature": jnp.float32(left),
                       "right_temperature": jnp.float32(20.0)}}
        state = self.gm.run_scan(STEPS_PER_FRAME, external_inputs=ext)
        self.time += STEPS_PER_FRAME * DT
        return render(np.asarray(state["rod"]["temperature"]), self.width, self.height)


# -- GStreamer, loaded only when it is there ----------------------------------

def load_gstreamer(elements: tuple[str, ...]) -> tuple[Optional[SimpleNamespace], str]:
    """``(gst, "")`` with ``Gst``, ``GstSdp`` and ``GstWebRTC``, or
    ``(None, why)`` when PyGObject, a typelib, an element or websockets
    is missing."""
    try:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstSdp", "1.0")
        gi.require_version("GstWebRTC", "1.0")
        from gi.repository import Gst, GstSdp, GstWebRTC
    except (ImportError, ValueError) as exc:
        return None, f"PyGObject with the GStreamer 1.0 typelibs is missing ({exc})"
    try:
        from websockets.sync.client import connect
    except ImportError as exc:
        return None, f"websockets >= 11 is missing ({exc})"
    if not Gst.is_initialized():
        Gst.init(None)
    missing = [name for name in elements if Gst.ElementFactory.find(name) is None]
    if missing:
        return None, f"the GStreamer elements {missing} are not installed"
    return SimpleNamespace(Gst=Gst, GstSdp=GstSdp, GstWebRTC=GstWebRTC, connect=connect), ""


# -- the viewer page ----------------------------------------------------------

def write_viewer_page(signaling_url: str, directory: Path) -> Path:
    """A copy of the shipped browser client, its URL filled in, connecting
    on load, with no STUN server (everything is on this machine)."""
    import maddening.api

    source = Path(maddening.api.__file__).parent / "static" / "webrtc_client.html"
    page = source.read_text(encoding="utf-8")
    edits = [
        ('placeholder="ws://host:port/signaling/session_id?token=..."',
         f'value="{html.escape(signaling_url)}"'),
        ("iceServers: [{ urls: 'stun:stun.l.google.com:19302' }]", "iceServers: []"),
        ("</script>", "window.addEventListener('load', connect);\n</script>"),
    ]
    for old, new in edits:
        if page.count(old) != 1:
            raise RuntimeError(f"{source} has changed: {old!r} is not in it once")
        page = page.replace(old, new)
    path = directory / "webrtc_viewer.html"
    path.touch(mode=0o600)
    path.write_text(page, encoding="utf-8")
    return path


# -- the self-check's peer ----------------------------------------------------

class Peer:
    """A WebRTC receiver that signals the way ``webrtc_client.html`` does.

    A ``webrtcbin`` with one receive-only H.264 transceiver: it OFFERS,
    trickles its ICE candidates as ``{"type": "ice", "candidate": ...}``,
    applies the answer and the server's candidates, and decodes what it
    is sent (``rtph264depay ! h264parse ! avdec_h264``).  Decoded frames
    are kept as ``(height, width, 3)`` RGB in :attr:`frames`.  A trimmed
    copy of the peer in ``tests/cloud/test_webrtc_loopback.py``.
    """

    def __init__(self, gst: SimpleNamespace, url: str) -> None:
        from maddening.cloud.selkies_session import _disable_upnp

        Gst, GstWebRTC = gst.Gst, gst.GstWebRTC
        self.gst, self.url = gst, url
        self.frames: list[np.ndarray] = []
        self.answer, self.ice_connected = threading.Event(), threading.Event()
        self.opened = self.answered = self.connected = 0.0     # time.monotonic() of each
        self.errors: list[str] = []
        self._offer_text = ""
        self._ws: Any = None
        self._stack = contextlib.ExitStack()
        self._send_lock = threading.Lock()
        self._reader: Optional[threading.Thread] = None
        self.pipe = Gst.Pipeline.new("peer")
        self.webrtc = Gst.ElementFactory.make("webrtcbin", "peer-webrtc")
        self.webrtc.set_property("bundle-policy", GstWebRTC.WebRTCBundlePolicy.MAX_BUNDLE)
        self.pipe.add(self.webrtc)
        # Payload type 102, as a browser's usually is (not GStreamer's 96).
        caps = Gst.Caps.from_string(
            "application/x-rtp,media=video,encoding-name=H264,clock-rate=90000,"
            "payload=102,packetization-mode=(string)1,profile-level-id=(string)42e01f")
        self.webrtc.emit("add-transceiver",
                         GstWebRTC.WebRTCRTPTransceiverDirection.RECVONLY, caps)
        _disable_upnp(self.webrtc)        # like a browser, map no router port
        self.webrtc.connect("on-ice-candidate", self._on_ice_candidate)
        self.webrtc.connect("pad-added", self._on_pad_added)
        self.webrtc.connect("notify::ice-connection-state", self._on_ice_state)

    def connect(self) -> None:
        """Open the signaling socket, then offer (the browser does both on open)."""
        Gst = self.gst.Gst
        self._ws = self._stack.enter_context(self.gst.connect(self.url, open_timeout=5))
        self._reader = threading.Thread(target=self._read, name="peer-reader", daemon=True)
        self.opened = time.monotonic()
        self._reader.start()
        self.pipe.set_state(Gst.State.PLAYING)
        promise = Gst.Promise.new_with_change_func(self._on_offer_created, None)
        self.webrtc.emit("create-offer", None, promise)

    def _send(self, message: dict) -> None:
        with self._send_lock:
            try:
                self._ws.send(json.dumps(message))
            except Exception as exc:  # noqa: BLE001 - recorded, reported on a timeout
                self.errors.append(f"send: {exc!r}")

    def _read(self) -> None:
        try:
            for raw in self._ws:
                message = json.loads(raw)
                if message.get("type") == "sdp":
                    self._on_answer(message["sdp"])
                elif message.get("type") == "ice":
                    candidate = message["candidate"]
                    self.webrtc.emit("add-ice-candidate", candidate["sdpMLineIndex"],
                                     candidate["candidate"])
                elif message.get("type") == "error":
                    self.errors.append(f"server: {message.get('message')}")
        except Exception as exc:  # noqa: BLE001 - recorded, reported on a timeout
            self.errors.append(f"reader: {exc!r}")

    def _on_offer_created(self, promise, _user_data) -> None:
        reply = promise.get_reply()
        offer = reply.get_value("offer")
        self._offer_text = offer.sdp.as_text()
        self.webrtc.emit("set-local-description", offer, self.gst.Gst.Promise.new())
        self._send({"type": "sdp", "sdp": {"type": "offer", "sdp": self._offer_text}})

    def _on_answer(self, description: dict) -> None:
        GstSdp, GstWebRTC = self.gst.GstSdp, self.gst.GstWebRTC
        ok, sdp = GstSdp.SDPMessage.new_from_text(description["sdp"])
        if description.get("type") != "answer" or ok != GstSdp.SDPResult.OK:
            self.errors.append(f"not an answer: {description.get('type')!r}")
            return
        answer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, sdp)
        self.webrtc.emit("set-remote-description", answer, self.gst.Gst.Promise.new())
        self.answered = time.monotonic()
        self.answer.set()

    def _on_ice_candidate(self, _element, mline: int, candidate: str) -> None:
        mid = None
        index = -1
        for line in self._offer_text.splitlines():
            if line.startswith("m="):
                index += 1
            elif index == mline and line.startswith("a=mid:"):
                mid = line[len("a=mid:"):].strip()
        self._send({"type": "ice", "candidate": {"candidate": candidate,
                                                 "sdpMLineIndex": mline, "sdpMid": mid}})

    def _on_ice_state(self, element, _pspec) -> None:
        state = self.gst.GstWebRTC.WebRTCICEConnectionState
        if element.get_property("ice-connection-state") in (state.CONNECTED, state.COMPLETED):
            if not self.ice_connected.is_set():
                self.connected = time.monotonic()
            self.ice_connected.set()

    def _on_pad_added(self, _element, pad) -> None:
        Gst = self.gst.Gst
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

    def _on_sample(self, sink):
        Gst = self.gst.Gst
        sample = sink.emit("pull-sample")
        structure = sample.get_caps().get_structure(0)
        width, height = structure.get_value("width"), structure.get_value("height")
        buffer = sample.get_buffer()
        ok, info = buffer.map(Gst.MapFlags.READ)
        if ok:
            try:
                raw = np.frombuffer(info.data, dtype=np.uint8)[: width * height * 3]
                self.frames.append(raw.reshape(height, width, 3).copy())
            finally:
                buffer.unmap(info)
        return Gst.FlowReturn.OK

    def close(self) -> None:
        with contextlib.suppress(Exception):      # the server may have closed it
            self._stack.close()
        if self._reader is not None:
            self._reader.join(timeout=5)
        self.pipe.set_state(self.gst.Gst.State.NULL)


# -- running it -----------------------------------------------------------------

#: The self-check's limits.  Measured in the streaming test image on three
#: cores (three runs): the answer 0.2 to 0.33 s after the signaling socket
#: opened, ICE connected about 0.06 s later, the frames below decoded
#: within 1.3 s, and the whole run about 3 s.  Each limit is many times that.
FRAMES_WANTED = 30
ANSWER_TIMEOUT_S, ICE_TIMEOUT_S, FRAMES_TIMEOUT_S = 10.0, 15.0, 30.0
#: How far a decoded frame may be from the frame pushed that it is nearest
#: to: the mean absolute difference over pixels and channels, on 0 to 255.
#: The path is RGBA -> I420 (4:2:0) -> H.264 -> I420 -> RGB, which smears
#: the profile line's edges a little.  Measured in the image: at most 2.9 at
#: the default size and bitrate (three runs); the limit is four times that,
#: and an all-black frame is about 50 from every frame pushed.
FRAME_TOLERANCE = 12.0


def push_frames(session, simulation: Simulation, fps: int, until) -> list[np.ndarray]:
    """Render and push frames at ``fps`` until ``until()``; returns them (RGB)."""
    pushed = []
    next_due = time.monotonic()
    while not until():
        frame = simulation.next_frame()
        session.update_framebuffer_cpu(frame, simulation.width, simulation.height, "RGBA")
        pushed.append(frame[..., :3])
        next_due += 1.0 / fps
        time.sleep(max(0.0, next_due - time.monotonic()))
    return pushed


def self_check(gst: SimpleNamespace, session, url: str, simulation: Simulation,
               fps: int) -> None:
    peer = Peer(gst, url)
    try:
        peer.connect()
        deadline = time.monotonic() + ANSWER_TIMEOUT_S + ICE_TIMEOUT_S + FRAMES_TIMEOUT_S
        start = time.monotonic()
        pushed = push_frames(session, simulation, fps,
                             lambda: len(peer.frames) >= FRAMES_WANTED
                             or time.monotonic() > deadline)
        took = time.monotonic() - start
        problems = f"; the peer recorded {peer.errors}" if peer.errors else ""
        assert peer.answer.is_set(), f"the server never answered the offer{problems}"
        assert peer.ice_connected.is_set(), f"ICE did not connect{problems}"
        frames = list(peer.frames)
        assert len(frames) >= FRAMES_WANTED, (
            f"{len(frames)} frames decoded in {took:.1f} s, {len(pushed)} pushed{problems}")
    finally:
        peer.close()

    print(f"  the answer came {peer.answered - peer.opened:.2f} s after the socket opened, "
          f"ICE connected at {peer.connected - peer.opened:.2f} s;")
    print(f"  {len(frames)} frames decoded in {took:.1f} s ({len(pushed)} pushed; frames "
          f"pushed before the answer are dropped)")
    sample = np.stack(pushed)[:, ::4, ::4].astype(np.float32)
    nearest, worst = [], 0.0
    for i, frame in enumerate(frames):
        assert frame.shape == (simulation.height, simulation.width, 3), (
            f"frame {i} is {frame.shape[1]}x{frame.shape[0]}, the stream "
            f"{simulation.width}x{simulation.height}")
        error = np.abs(sample - frame[::4, ::4].astype(np.float32)).mean(axis=(1, 2, 3))
        k = int(error.argmin())
        assert error[k] <= FRAME_TOLERANCE, (
            f"decoded frame {i} is {error[k]:.1f} from the nearest frame pushed "
            f"(limit {FRAME_TOLERANCE})")
        nearest.append(k)
        worst = max(worst, float(error[k]))
    black = float(sample.mean(axis=(1, 2, 3)).min())
    print(f"  every decoded frame is within {worst:.1f} (mean absolute difference, 0-255) "
          f"of a frame pushed; a black frame would be {black:.0f} away")
    assert black > 3 * FRAME_TOLERANCE, "the frames pushed are too dark to tell apart"
    distinct = len(set(nearest))
    print(f"  the decoded frames show {distinct} different frames pushed, in order: "
          f"pushed frame {nearest[0]} first, {nearest[-1]} last")
    assert distinct >= FRAMES_WANTED // 3 and nearest[-1] > nearest[0], nearest
    assert all(b >= a for a, b in zip(nearest, nearest[1:])), (
        f"the decoded frames are not in the order they were pushed: {nearest}")


def watch(session, url: str, simulation: Simulation, fps: int, seconds: float) -> None:
    directory = Path(tempfile.mkdtemp(prefix="maddening-webrtc-"))
    try:
        page = write_viewer_page(url, directory)
        print(f"  Open this page in a browser on this machine (it connects by itself):\n\n"
              f"      {page.as_uri()}\n")
        print("  It holds this run's session token: it is readable by you only, and is")
        print("  removed when the example stops.  Stop with Ctrl-C"
              + (f", or wait {seconds:g} s." if seconds else "."))
        end = time.monotonic() + seconds if seconds else float("inf")
        try:
            pushed = push_frames(session, simulation, fps, lambda: time.monotonic() > end)
        except KeyboardInterrupt:
            pushed = None
        print(f"\n  Stopped after {simulation.time:.2f} s of simulated time"
              + (f", {len(pushed)} frames pushed." if pushed is not None else "."))
    finally:
        shutil.rmtree(directory, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-check", action="store_true",
                        help="Connect a GStreamer peer instead of a browser and check the "
                             "frames it decodes (headless)")
    parser.add_argument("--seconds", type=float, default=0.0,
                        help="Stop after this many seconds (default: run until Ctrl-C)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default 640)")
    parser.add_argument("--height", type=int, default=360, help="Frame height (default 360)")
    parser.add_argument("--fps", type=int, default=30, help="Frames a second (default 30)")
    parser.add_argument("--bitrate-kbps", type=int, default=2000,
                        help="H.264 bitrate in kbit/s (default 2000)")
    parser.add_argument("--port", type=int, default=0,
                        help="Signaling port on 127.0.0.1 (default: one the system chooses)")
    args = parser.parse_args(argv)
    if args.width < 64 or args.height < 64 or args.width % 2 or args.height % 2:
        parser.error("--width and --height must be even, and at least 64")

    gst, why = load_gstreamer(SERVER_ELEMENTS + (PEER_ELEMENTS if args.self_check else ()))
    if gst is None:
        print(f"Cannot stream: {why}.\n\n{INSTALL_HINT}", file=sys.stderr)
        return 2

    from maddening.cloud.selkies_session import SelkiesSession
    from maddening.cloud.streaming import StreamConfig

    print(f"Simulation: a {N_CELLS}-cell HeatNode rod, {STEPS_PER_FRAME} steps a frame.")
    simulation = Simulation(args.width, args.height)
    simulation.next_frame()                                    # compile before streaming
    # A secret made for this run: the token derived from it is good for
    # this session only, and nobody else holds the secret.
    session = SelkiesSession(secret=secrets.token_hex(16), signaling_port=args.port,
                             bind_host="127.0.0.1")
    info = session.start(StreamConfig(width=args.width, height=args.height, fps=args.fps,
                                      bitrate_kbps=args.bitrate_kbps, ice_servers=[]))
    try:
        parts = urllib.parse.urlsplit(info.signaling_url)
        url = (f"ws://127.0.0.1:{parts.port}{parts.path}?"
               + urllib.parse.urlencode({"token": session.session_token}))
        print(f"Streaming {args.width}x{args.height} at {args.fps} fps, x264enc on the CPU; "
              f"signaling on 127.0.0.1:{parts.port}, no STUN or TURN server.")
        if args.self_check:
            self_check(gst, session, url, simulation, args.fps)
            print("\nSelf-check passed.")
        else:
            watch(session, url, simulation, args.fps, args.seconds)
    finally:
        session.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
