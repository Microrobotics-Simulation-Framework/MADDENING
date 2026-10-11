"""SelkiesSession — GStreamer/WebRTC streaming implementation.

Encodes the frames pushed into it with GStreamer (``appsrc ->
videoconvert -> x264enc -> rtph264pay -> webrtcbin``) and runs an
embedded, authenticated signaling WebSocket through which one viewer at
a time negotiates the WebRTC connection.  Requires PyGObject and system
GStreamer libraries (``webrtcbin`` from gst-plugins-bad, ``x264enc``
from gst-plugins-ugly, ``gstreamer1.0-nice``).

Signaling protocol
------------------
This is the protocol of the browser client shipped as
``maddening/api/static/webrtc_client.html``; every message is a JSON
object on the signaling socket:

* viewer -> server ``{"type": "sdp", "sdp": {"type": "offer", "sdp": "..."}}``:
  the viewer OFFERS (it adds a receive-only video transceiver); the
  server answers.
* server -> viewer ``{"type": "sdp", "sdp": {"type": "answer", "sdp": "..."}}``.
* both ways ``{"type": "ice", "candidate": {"candidate": "...",
  "sdpMLineIndex": 0, "sdpMid": "..."}}``: trickled ICE candidates (the
  shape of the browser's ``RTCIceCandidate.toJSON()``).
* viewer -> server, any other object (mouse, keyboard, gamepad events):
  handed to the handler installed with
  :meth:`SelkiesSession.set_input_handler`.  SDP and ICE messages never
  reach that handler.

The viewer must authenticate before any of this (see
:func:`_make_signaling_handler`).

What is tested, and what is not
-------------------------------
``tests/cloud/test_webrtc_loopback.py`` connects a GStreamer WebRTC peer
that signals as the browser client does, on the loopback interface of
a CPU-only container, and checks that it receives the answer, completes
ICE, decodes the frames that were pushed (size and colour), and that its
input events reach the input handler.  Not tested: a GPU encoder (the
encoder is ``x264enc``, in software), a real browser, and NAT traversal
(see the ``ice_servers`` note on :class:`SelkiesSession`).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import threading
import urllib.parse
import uuid
from typing import Any, Callable, Optional

from maddening.cloud._auth import generate_session_token, validate_session_token
from maddening.cloud.streaming import (
    GPUFramebuffer,
    StreamConfig,
    StreamInfo,
    StreamReconfigError,
    StreamStartError,
    StreamingSession,
)

logger = logging.getLogger(__name__)

# Lazy GStreamer import check
_HAS_GST = None


def _check_gstreamer() -> bool:
    global _HAS_GST
    if _HAS_GST is None:
        try:
            import gi
            gi.require_version("Gst", "1.0")
            gi.require_version("GstSdp", "1.0")
            gi.require_version("GstWebRTC", "1.0")
            from gi.repository import Gst
            _HAS_GST = True
        except (ImportError, ValueError):
            _HAS_GST = False
    return _HAS_GST


HAS_GSTREAMER = property(lambda self: _check_gstreamer())

#: WebSocket close code for a policy violation (RFC 6455 §7.4.1).
_CLOSE_POLICY_VIOLATION = 1008
#: WebSocket close code for a server-side failure (RFC 6455 §7.4.1).
_CLOSE_INTERNAL_ERROR = 1011
#: Close code sent to a viewer when a newer authenticated viewer takes
#: its place (4000-4999 are left to applications by RFC 6455 §7.4.2).
_CLOSE_REPLACED = 4000

#: Message types the negotiation consumes.  These never reach the input
#: handler, whether or not a negotiator is attached.
_NEGOTIATION_TYPES = frozenset({"sdp", "ice"})

#: Viewer ICE candidates held while the offer is being applied; more
#: than this from one viewer before its offer is applied are dropped.
_MAX_PENDING_CANDIDATES = 256

#: How long :meth:`SelkiesSession.start` waits for the signaling socket to
#: listen and for the GLib loop to run, and :meth:`SelkiesSession.stop`
#: for each thread to end, in seconds.
_THREAD_TIMEOUT_S = 10


def _header_value(headers: Any, name: str) -> str:
    """The single value of request header *name*, or ``""``.

    A header sent more than once is treated as absent rather than
    resolved to one of its values: a request that disagrees with itself
    about its own credential must not be authenticated.
    """
    if headers is None:
        return ""
    get_all = getattr(headers, "get_all", None)
    try:
        values = list(get_all(name)) if get_all is not None else None
        if values is None:
            value = headers.get(name)
            values = [] if value is None else [value]
    except Exception:  # noqa: BLE001 - a malformed header set is "no header"
        return ""
    return values[0] if len(values) == 1 else ""


def _client_token(request: Any, path: str = "") -> str:
    """The session token the signaling client presented, or ``""``.

    Two carriers are accepted, checked in this order:

    * ``Authorization: Bearer <token>`` -- preferred, because a header
      does not land in proxy access logs the way a query string does.
    * ``?token=<token>`` on the request target.

    Parameters
    ----------
    request : object or None
        The connection's request object (``websockets`` exposes
        ``.path`` and ``.headers``).  ``None`` falls back to *path*.
    path : str, optional
        Request target, used when *request* carries no ``path`` (older
        ``websockets`` releases pass it to the handler separately).

    Returns
    -------
    str
        The presented token, or ``""`` when the client presented none,
        presented more than one, or presented an unparsable target.
        The empty string never validates, so "absent" and "wrong" take
        the same rejection path.
    """
    header = _header_value(getattr(request, "headers", None), "Authorization")
    scheme, _, value = header.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()

    target = getattr(request, "path", None) or path or ""
    try:
        query = urllib.parse.urlsplit(target).query
        tokens = urllib.parse.parse_qs(query).get("token", [])
    except ValueError:
        return ""
    return tokens[0] if len(tokens) == 1 else ""


# ---------------------------------------------------------------------------
# SDP / ICE helpers (pure Python: no GStreamer needed to test them)
# ---------------------------------------------------------------------------


def _fmtp_params(value: str) -> dict[str, str]:
    """``a=fmtp`` parameters (``k=v;k=v``) as a dict with lower-case keys."""
    params = {}
    for item in value.split(";"):
        key, sep, val = item.strip().partition("=")
        if sep:
            params[key.strip().lower()] = val.strip()
    return params


def _h264_payload_type(sdp_text: str) -> Optional[tuple[int, int]]:
    """The H.264 format of an offer that the session's stream can be sent as.

    The session sends constrained-baseline H.264 packetised by
    ``rtph264pay``, which fragments large NAL units (FU-A) and therefore
    needs ``packetization-mode=1``.  A browser offers several H.264
    formats under dynamic payload types; the answer must use one of the
    offer's payload types, and the RTP packets must carry it.

    Parameters
    ----------
    sdp_text : str
        The offer.

    Returns
    -------
    tuple or None
        ``(m-line index, payload type)``: the first video m-line that
        offers H.264 with ``packetization-mode=1``, and in it a
        constrained-baseline ``profile-level-id`` before any other
        baseline one before any other profile, ties going to the
        offer's own order.  ``None`` when the offer has no such format.
    """
    sections: list[dict[str, Any]] = []
    for raw in sdp_text.splitlines():
        line = raw.strip()
        if line.startswith("m="):
            fields = line[2:].split()
            sections.append({"kind": fields[0] if fields else "",
                             "formats": fields[3:], "h264": [], "fmtp": {}})
        elif not sections:
            continue
        elif line.startswith("a=rtpmap:"):
            pt_text, _, encoding = line[len("a=rtpmap:"):].partition(" ")
            if pt_text.isdigit() and encoding.split("/")[0].strip().upper() == "H264":
                sections[-1]["h264"].append(int(pt_text))
        elif line.startswith("a=fmtp:"):
            pt_text, _, params = line[len("a=fmtp:"):].partition(" ")
            if pt_text.isdigit():
                sections[-1]["fmtp"][int(pt_text)] = _fmtp_params(params)
    for index, section in enumerate(sections):
        if section["kind"] != "video":
            continue
        ranked = []
        for pt in section["h264"]:
            if str(pt) not in section["formats"]:
                continue
            params = section["fmtp"].get(pt, {})
            if params.get("packetization-mode") != "1":
                continue
            ranked.append((_h264_profile_rank(params.get("profile-level-id", "")),
                           section["formats"].index(str(pt)), pt))
        if ranked:
            return index, min(ranked)[2]
    return None


def _h264_profile_rank(profile_level_id: str) -> int:
    """0 for constrained baseline, 1 for other baseline, 2 for anything else.

    RFC 6184 §8.1: the first byte is ``profile_idc`` (66 = baseline) and
    the second the constraint flags, of which ``constraint_set1``
    (0x40) marks constrained baseline.  An absent value defaults to
    baseline (``420010``).
    """
    text = profile_level_id.strip() or "420010"
    try:
        profile_idc, constraints = int(text[0:2], 16), int(text[2:4], 16)
    except ValueError:
        return 2
    if profile_idc != 0x42:
        return 2
    return 0 if constraints & 0x40 else 1


def _remote_candidate(
    message: dict[str, Any], mids: dict[str, int],
) -> Optional[tuple[int, str]]:
    """``(m-line index, candidate line)`` of a viewer's ``ice`` message.

    The m-line comes from ``sdpMLineIndex`` when it is an integer, else
    from ``sdpMid`` looked up in *mids* (the offer's ``a=mid`` values).
    ``None`` for a malformed message or an empty candidate (the
    end-of-candidates marker, which needs no action here).
    """
    candidate = message.get("candidate")
    if not isinstance(candidate, dict):
        return None
    line = candidate.get("candidate")
    if not isinstance(line, str) or not line.strip():
        return None
    index = candidate.get("sdpMLineIndex")
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        return index, line
    mid = candidate.get("sdpMid")
    if isinstance(mid, str) and mid in mids:
        return mids[mid], line
    return None


def _ice_server_settings(
    ice_servers: list[dict[str, Any]],
) -> tuple[Optional[str], list[str]]:
    """``webrtcbin``'s ``stun-server`` and TURN URIs from ``StreamConfig.ice_servers``.

    ``ice_servers`` takes the browser's ``RTCIceServer`` shape: dicts with
    ``urls`` (a string or a list of them, ``stun:host[:port]`` or
    ``turn:``/``turns:host[:port][?transport=udp|tcp]``) and, for TURN,
    ``username`` and ``credential``.  ``webrtcbin`` takes one STUN server
    (``stun://host:port``; the first one listed is used) and any number of
    TURN servers (``turn://user:password@host:port``, with the user and
    password percent-encoded: ``webrtcbin`` unescapes both before use).
    Entries it cannot express are skipped with a warning.
    """
    stun: Optional[str] = None
    turn: list[str] = []
    for server in ice_servers or []:
        if not isinstance(server, dict):
            logger.warning("Ignoring ICE server entry %r: not a dict", server)
            continue
        urls = server.get("urls", server.get("url", []))
        urls = [urls] if isinstance(urls, str) else list(urls or [])
        for url in urls:
            scheme, _, rest = str(url).partition(":")
            scheme = scheme.lower()
            host_port, _, query = rest.lstrip("/").partition("?")
            if not host_port:
                logger.warning("Ignoring ICE server URL %r: no host", url)
            elif scheme == "stun":
                if stun is None:
                    stun = f"stun://{host_port}"
                else:
                    logger.warning(
                        "Ignoring STUN server %r: webrtcbin takes one, and %r "
                        "was listed first", url, stun)
            elif scheme in ("turn", "turns"):
                user = urllib.parse.quote(str(server.get("username", "")), safe="")
                password = urllib.parse.quote(str(server.get("credential", "")), safe="")
                if not user or not password:
                    logger.warning(
                        "Ignoring TURN server %r: it needs a username and a "
                        "credential", url)
                    continue
                suffix = f"?{query}" if query else ""
                turn.append(f"{scheme}://{user}:{password}@{host_port}{suffix}")
            else:
                logger.warning("Ignoring ICE server URL %r: unknown scheme", url)
    return stun, turn


# ---------------------------------------------------------------------------
# Signaling
# ---------------------------------------------------------------------------


class _SignalingClient:
    """One authenticated signaling connection, usable from any thread.

    Outgoing messages go through a queue drained by :meth:`pump` on the
    connection's own event loop, so messages sent from GStreamer's
    threads keep their order on the wire.
    """

    def __init__(self, ws: Any, loop: asyncio.AbstractEventLoop) -> None:
        self.ws = ws
        self.loop = loop
        self.outbox: asyncio.Queue = asyncio.Queue()

    def send(self, message: dict[str, Any]) -> None:
        text = json.dumps(message)
        with contextlib.suppress(RuntimeError):  # the loop has closed
            self.loop.call_soon_threadsafe(self.outbox.put_nowait, text)

    def close(self, code: int, reason: str) -> None:
        def _close() -> None:
            asyncio.ensure_future(self.ws.close(code, reason))
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(_close)

    async def pump(self) -> None:
        try:
            while True:
                await self.ws.send(await self.outbox.get())
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the viewer went away
            logger.debug("Signaling send failed; the viewer disconnected", exc_info=True)


def _make_signaling_handler(
    session_id: str,
    secret: str,
    get_input_handler: Callable[[], Optional[Callable[[dict[str, Any]], None]]],
    negotiator: Optional["_WebRTCNegotiator"] = None,
):
    """Build the signaling WebSocket handler for one session.

    The returned coroutine authenticates the *client* before anything
    else: it reads the token the client presented (see
    :func:`_client_token`) and validates that token -- not the server's
    own -- against *session_id* and *secret* with
    :func:`maddening.cloud._auth.validate_session_token`, which compares
    in constant time.  An absent, malformed, wrong or
    wrong-session token is closed with 1008 before the first message is
    read, and before the negotiator hears of the connection, so an
    unauthenticated client is sent nothing.

    An authenticated client's ``sdp`` and ``ice`` messages go to the
    *negotiator* (and are dropped without one); every other JSON message
    goes to the input handler.

    Parameters
    ----------
    session_id : str
        The session the token must be bound to.
    secret : str
        Shared HMAC secret.
    get_input_handler : callable
        Called per message to fetch the current input handler, so a
        handler installed after ``start()`` is still used.
    negotiator : _WebRTCNegotiator, optional
        Told of each authenticated connection, its negotiation messages
        and its end.  ``None`` relays input events only.

    Returns
    -------
    callable
        ``async (ws) -> None``, suitable for ``websockets.serve``.
    """

    async def handler(ws) -> None:
        request = getattr(ws, "request", None)
        token = _client_token(request, getattr(ws, "path", "") or "")
        if not validate_session_token(session_id, token, secret):
            logger.warning(
                "Rejected signaling connection for session %s: %s",
                session_id,
                "no token presented" if not token else "invalid token",
            )
            await ws.close(_CLOSE_POLICY_VIOLATION, "Invalid token")
            return
        client: Optional[_SignalingClient] = None
        pump: Optional[asyncio.Future] = None
        if negotiator is not None:
            client = _SignalingClient(ws, asyncio.get_running_loop())
            pump = asyncio.ensure_future(client.pump())
            negotiator.client_connected(client)
        try:
            async for msg in ws:
                if not isinstance(msg, str):
                    continue
                try:
                    message = json.loads(msg)
                except ValueError:
                    continue
                if isinstance(message, dict) and message.get("type") in _NEGOTIATION_TYPES:
                    if negotiator is not None and client is not None:
                        negotiator.client_message(client, message)
                    continue
                input_handler = get_input_handler()
                if input_handler:
                    try:
                        input_handler(message)
                    except (ValueError, TypeError):
                        pass
        except Exception:  # noqa: BLE001 - a dropped client is not an error
            logger.debug("Signaling connection closed", exc_info=True)
        finally:
            if negotiator is not None and client is not None:
                negotiator.client_disconnected(client)
            if pump is not None:
                pump.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await pump

    return handler


# ---------------------------------------------------------------------------
# WebRTC negotiation (GStreamer side)
# ---------------------------------------------------------------------------


class _WebRTCNegotiator:
    """Answers one viewer's offer with a fresh pipeline per viewer.

    Everything that touches GStreamer runs on a GLib main loop in a
    thread of its own (``selkies-glib``), started and stopped with the
    session; the signaling thread and ``webrtcbin``'s own threads hand
    work to it with :meth:`_invoke`, so the negotiation state needs no
    lock.  The pipeline is built when the viewer's offer arrives (the
    answer's payload type comes from the offer) and torn down when the
    viewer leaves, so no frame is encoded, or kept, while nobody watches.

    One viewer at a time: a newly authenticated viewer replaces the
    current one, which is closed with code 4000 (``_CLOSE_REPLACED``).

    Parameters
    ----------
    describe_pipeline : callable
        ``(payload_type) -> str``, the ``gst-launch`` description of the
        pipeline to build for a viewer; it must name its ``appsrc``
        ``src``, its encoder ``encoder`` and its ``webrtcbin`` ``webrtc``.
    on_pipeline : callable
        Called on the GLib thread with ``(appsrc, encoder)`` when a
        viewer's pipeline starts and with ``(None, None)`` when it stops.
    stun_server : str or None
        ``webrtcbin``'s ``stun-server``.
    turn_servers : list of str
        URIs for ``webrtcbin``'s ``add-turn-server``.
    """

    def __init__(
        self,
        describe_pipeline: Callable[[int], str],
        on_pipeline: Callable[[Any, Any], None],
        stun_server: Optional[str] = None,
        turn_servers: Optional[list[str]] = None,
    ) -> None:
        from gi.repository import GLib

        self._describe_pipeline = describe_pipeline
        self._on_pipeline = on_pipeline
        self._stun_server = stun_server
        self._turn_servers = list(turn_servers or [])
        self._context = GLib.MainContext.new()
        self._loop = GLib.MainLoop.new(self._context, False)
        self._thread: Optional[threading.Thread] = None
        # GLib-thread state.
        self._client: Optional[_SignalingClient] = None
        self._pipeline: Any = None
        self._webrtc: Any = None
        self._mids: dict[str, int] = {}
        self._answer_sent = False
        self._remote_set = False
        self._pending_local: list[tuple[int, str]] = []
        self._pending_remote: list[dict[str, Any]] = []

    # -- lifecycle (any thread) ----------------------------------------

    def start(self) -> None:
        running = threading.Event()

        def _run() -> None:
            self._context.push_thread_default()
            try:
                self._invoke(running.set)
                self._loop.run()
            finally:
                self._teardown()
                self._context.pop_thread_default()

        self._thread = threading.Thread(target=_run, name="selkies-glib", daemon=True)
        self._thread.start()
        if not running.wait(_THREAD_TIMEOUT_S):
            raise StreamStartError("the GLib main loop did not start")

    def stop(self) -> None:
        if self._thread is None:
            return
        self._invoke(self._loop.quit)
        self._thread.join(_THREAD_TIMEOUT_S)
        if self._thread.is_alive():
            logger.warning("The selkies-glib thread did not stop within %s s",
                           _THREAD_TIMEOUT_S)
        self._thread = None

    def _invoke(self, fn: Callable[..., Any], *args: Any) -> None:
        """Run ``fn(*args)`` on the GLib thread, after what is already queued."""
        from gi.repository import GLib

        def _call(*_unused: Any) -> bool:
            try:
                fn(*args)
            except Exception:  # noqa: BLE001 - one failed step must not kill the loop
                logger.exception("WebRTC negotiation step %s failed",
                                 getattr(fn, "__name__", fn))
            return False

        self._context.invoke_full(GLib.PRIORITY_DEFAULT, _call)

    # -- signaling events (called on the signaling thread) -------------

    def client_connected(self, client: _SignalingClient) -> None:
        self._invoke(self._connect, client)

    def client_message(self, client: _SignalingClient, message: dict[str, Any]) -> None:
        if message.get("type") == "sdp":
            self._invoke(self._offer, client, message.get("sdp"))
        else:
            self._invoke(self._remote_ice, client, message)

    def client_disconnected(self, client: _SignalingClient) -> None:
        self._invoke(self._disconnect, client)

    # -- GLib thread ---------------------------------------------------

    def _connect(self, client: _SignalingClient) -> None:
        if self._client is not None and self._client is not client:
            logger.info("A new viewer replaces the current one")
            self._client.close(_CLOSE_REPLACED, "replaced by a newer viewer")
            self._teardown()
        self._client = client

    def _disconnect(self, client: _SignalingClient) -> None:
        if client is self._client:
            self._teardown()
            self._client = None

    def _offer(self, client: _SignalingClient, description: Any) -> None:
        from gi.repository import GLib, Gst, GstSdp, GstWebRTC

        if client is not self._client:
            return
        if not isinstance(description, dict) or description.get("type") != "offer" \
                or not isinstance(description.get("sdp"), str):
            logger.warning("Ignoring an sdp message that is not an offer: the "
                           "server answers, it never offers")
            return
        text = description["sdp"]
        chosen = _h264_payload_type(text)
        if chosen is None:
            logger.warning("The viewer's offer has no H.264 video format with "
                           "packetization-mode=1; no answer is sent")
            return
        ok, sdp = GstSdp.SDPMessage.new_from_text(text)
        if ok != GstSdp.SDPResult.OK:
            logger.warning("Ignoring an offer that does not parse as SDP")
            return
        mline, payload_type = chosen

        # A second offer on one connection starts over with a fresh pipeline.
        self._teardown()
        self._mids = {}
        for index in range(sdp.medias_len()):
            mid = sdp.get_media(index).get_attribute_val("mid")
            if mid is not None:
                self._mids[mid] = index

        pipeline = Gst.parse_launch(self._describe_pipeline(payload_type))
        webrtc = pipeline.get_by_name("webrtc")
        if self._stun_server:
            webrtc.set_property("stun-server", self._stun_server)
        if self._turn_servers:
            # The first through the property (readable back), the rest added.
            webrtc.set_property("turn-server", self._turn_servers[0])
            for uri in self._turn_servers[1:]:
                webrtc.emit("add-turn-server", uri)
        webrtc.connect("on-ice-candidate", self._on_local_candidate)
        webrtc.connect("notify::ice-connection-state", self._on_ice_state)
        pipeline.get_bus().add_watch(GLib.PRIORITY_DEFAULT, self._on_bus_message, pipeline)
        self._pipeline, self._webrtc = pipeline, webrtc
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            logger.error("The streaming pipeline would not start")
            self._fail(client)
            return
        self._on_pipeline(pipeline.get_by_name("src"), pipeline.get_by_name("encoder"))
        logger.info("Answering the viewer's offer (H.264 payload type %d on m-line %d)",
                    payload_type, mline)

        offer = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.OFFER, sdp)
        webrtc.emit("set-remote-description", offer,
                    self._promise(self._remote_description_set, webrtc))

    def _promise(self, then: Callable[..., None], webrtc: Any) -> Any:
        """A promise whose reply is handed to ``then(webrtc, reply)`` on the GLib thread."""
        from gi.repository import Gst

        def _replied(promise: Any, _user_data: Any) -> None:
            # The reply belongs to the promise, which webrtcbin may free once
            # this returns: hand a copy to the GLib thread.
            reply = promise.get_reply()
            self._invoke(then, webrtc, reply.copy() if reply is not None else None)

        return Gst.Promise.new_with_change_func(_replied, None)

    def _step_failed(self, webrtc: Any, reply: Any, step: str) -> bool:
        if webrtc is not self._webrtc:
            return True  # the viewer left, or re-offered, meanwhile
        if reply is not None and reply.has_field("error"):
            logger.error("WebRTC %s failed: %s", step, reply.get_value("error"))
            self._fail(self._client)
            return True
        return False

    def _remote_description_set(self, webrtc: Any, reply: Any) -> None:
        if self._step_failed(webrtc, reply, "set-remote-description"):
            return
        self._remote_set = True
        for message in self._pending_remote:
            self._apply_remote_ice(message)
        self._pending_remote = []
        webrtc.emit("create-answer", None, self._promise(self._answer_created, webrtc))

    def _answer_created(self, webrtc: Any, reply: Any) -> None:
        if self._step_failed(webrtc, reply, "create-answer"):
            return
        answer = reply.get_value("answer") if reply is not None else None
        if answer is None:
            logger.error("webrtcbin created no answer")
            self._fail(self._client)
            return
        # The text is read now: the description belongs to the promise's
        # reply, which is freed once this callback returns.
        text = answer.sdp.as_text()
        webrtc.emit("set-local-description", answer,
                    self._promise(lambda w, r: self._answer_applied(w, r, text), webrtc))

    def _answer_applied(self, webrtc: Any, reply: Any, answer_text: str) -> None:
        if self._step_failed(webrtc, reply, "set-local-description"):
            return
        client = self._client
        if client is None:
            return
        client.send({"type": "sdp", "sdp": {"type": "answer", "sdp": answer_text}})
        self._answer_sent = True
        # Candidates gathered before the answer went out follow it, so the
        # viewer never gets a candidate for a description it does not have.
        for mline, candidate in self._pending_local:
            self._send_local_ice(client, mline, candidate)
        self._pending_local = []

    def _on_local_candidate(self, webrtc: Any, mline: int, candidate: str) -> None:
        # webrtcbin's thread: hand over to the GLib thread.
        self._invoke(self._local_ice, webrtc, mline, candidate)

    def _local_ice(self, webrtc: Any, mline: int, candidate: str) -> None:
        if webrtc is not self._webrtc or self._client is None:
            return
        if not self._answer_sent:
            self._pending_local.append((mline, candidate))
        else:
            self._send_local_ice(self._client, mline, candidate)

    def _send_local_ice(self, client: _SignalingClient, mline: int, candidate: str) -> None:
        payload: dict[str, Any] = {"candidate": candidate, "sdpMLineIndex": mline}
        mid = next((m for m, i in self._mids.items() if i == mline), None)
        if mid is not None:
            payload["sdpMid"] = mid
        client.send({"type": "ice", "candidate": payload})

    def _remote_ice(self, client: _SignalingClient, message: dict[str, Any]) -> None:
        if client is not self._client:
            return
        if not self._remote_set:
            if len(self._pending_remote) < _MAX_PENDING_CANDIDATES:
                self._pending_remote.append(message)
        else:
            self._apply_remote_ice(message)

    def _apply_remote_ice(self, message: dict[str, Any]) -> None:
        parsed = _remote_candidate(message, self._mids)
        if parsed is None:
            logger.debug("Ignoring an ICE message without a usable candidate")
            return
        if self._webrtc is not None:
            self._webrtc.emit("add-ice-candidate", parsed[0], parsed[1])

    def _on_ice_state(self, webrtc: Any, _pspec: Any) -> None:
        logger.info("WebRTC ICE connection state: %s",
                    webrtc.get_property("ice-connection-state").value_nick)

    def _on_bus_message(self, _bus: Any, message: Any, pipeline: Any) -> bool:
        from gi.repository import Gst

        if message.type == Gst.MessageType.ERROR and pipeline is self._pipeline:
            err, debug = message.parse_error()
            logger.error("Streaming pipeline error: %s (%s)", err.message, debug)
            self._fail(self._client)
        return True

    def _fail(self, client: Optional[_SignalingClient]) -> None:
        """End this viewer's stream and tell the viewer (1011)."""
        self._teardown()
        if client is not None:
            client.close(_CLOSE_INTERNAL_ERROR, "streaming pipeline error")

    def _teardown(self) -> None:
        self._remote_set = self._answer_sent = False
        self._pending_local, self._pending_remote = [], []
        if self._pipeline is None:
            return
        from gi.repository import Gst

        pipeline = self._pipeline
        self._pipeline = self._webrtc = None
        self._on_pipeline(None, None)
        pipeline.get_bus().remove_watch()
        pipeline.set_state(Gst.State.NULL)


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------

#: Bytes per pixel of the raw formats a frame may be pushed in.
_BYTES_PER_PIXEL = {"RGBA": 4, "BGRA": 4, "RGBx": 4, "BGRx": 4, "ARGB": 4,
                    "ABGR": 4, "xRGB": 4, "xBGR": 4, "RGB": 3, "BGR": 3}


class SelkiesSession(StreamingSession):
    """GStreamer-based WebRTC streaming session.

    A viewer connects to the signaling WebSocket, authenticates, and
    sends its offer; the session answers it and trickles ICE candidates
    both ways (the protocol is in the module docstring).  Frames pushed
    with :meth:`update_framebuffer_cpu` are encoded (``x264enc``,
    constrained-baseline H.264) and sent to the connected viewer.

    **One viewer at a time.**  A newly authenticated viewer replaces the
    current one, whose signaling socket is closed with code 4000;
    reloading the viewer page therefore
    reconnects at once instead of waiting for the old connection to time
    out.  Only a client holding the session token can do this.

    The encoding pipeline exists only while a viewer is connected: frames
    pushed while nobody watches are dropped, not queued.

    Parameters
    ----------
    secret : str
        Shared secret for HMAC-SHA256 token authentication.  Every
        signaling client must present
        ``generate_session_token(session_id, secret)`` -- as
        ``Authorization: Bearer <token>`` or ``?token=<token>`` -- or the
        connection is closed with 1008.  Left empty, a random per-process
        secret is generated: the session then still runs, but no client
        can compute a token, so only an in-process caller holding
        :attr:`session_token` can connect.  Pass the secret you shared
        with the client.
    signaling_port : int
        Port for the embedded WebSocket signaling server.  ``0`` lets the
        operating system choose; :attr:`StreamInfo.signaling_url` carries
        the port it chose.
    bind_host : str, optional
        Address the signaling server binds.  The default stays
        ``"0.0.0.0"`` because this session normally runs inside a
        container, where a loopback bind is unreachable even with a
        published port -- but it was **hardcoded**, so the socket could
        not be made loopback-only even deliberately.  Pass
        ``"127.0.0.1"`` when the viewer reaches it through an SSH
        tunnel.  The signaling protocol is HMAC-authenticated either
        way and fails closed when ``MADDENING_STREAM_SECRET`` is unset,
        so this is exposure, not authentication.

    Notes
    -----
    **NAT traversal.**  ``StreamConfig.ice_servers`` (browser
    ``RTCIceServer`` shape) is applied to the server's ``webrtcbin``: the
    first ``stun:`` URL becomes its ``stun-server`` and every ``turn:`` /
    ``turns:`` URL with a ``username`` and ``credential`` is added as a
    TURN server.  The default lists a public STUN server, which the
    session then contacts when a viewer connects; pass
    ``ice_servers=[]`` to contact none.  The media travels over UDP
    (or TURN), which a cloud provider may not expose: a viewer outside
    the provider's network needs a reachable TURN server, or a provider
    that forwards UDP.  None of this is covered by the loopback test.
    """

    def __init__(
        self,
        secret: str = "",
        signaling_port: int = 8443,
        bind_host: str = "0.0.0.0",
    ) -> None:
        if not _check_gstreamer():
            raise ImportError(
                "SelkiesSession requires PyGObject and GStreamer. "
                "Install with: pip install PyGObject && "
                "apt install gstreamer1.0-plugins-base gstreamer1.0-plugins-good "
                "gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-nice"
            )

        self._secret = secret or uuid.uuid4().hex
        if not secret:
            logger.warning(
                "SelkiesSession was constructed without a shared secret; a "
                "random one was generated. Signaling clients cannot compute a "
                "token from a secret nobody holds, so every external "
                "connection will be rejected. Pass secret=... (the cloud "
                "entry point reads MADDENING_STREAM_SECRET) to let a client "
                "in.",
            )
        self._signaling_port = signaling_port
        self._bind_host = bind_host
        self._alive = False
        self._config: Optional[StreamConfig] = None
        self._info: Optional[StreamInfo] = None
        self._media_lock = threading.Lock()
        self._appsrc: Any = None
        self._encoder: Any = None
        self._negotiator: Optional[_WebRTCNegotiator] = None
        self._signaling_thread: Optional[threading.Thread] = None
        self._signaling_loop: Optional[asyncio.AbstractEventLoop] = None
        self._signaling_server: Any = None
        self._input_handler: Optional[Callable[[dict[str, Any]], None]] = None
        self._session_id = ""

    def start(self, config: StreamConfig) -> StreamInfo:
        import gi
        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if not Gst.is_initialized():
            Gst.init(None)

        self._config = config
        self._session_id = uuid.uuid4().hex[:12]

        try:
            self._check_pipeline(config)
            stun, turn = _ice_server_settings(list(config.ice_servers))
            self._negotiator = _WebRTCNegotiator(
                self._pipeline_description, self._set_media, stun, turn)
            self._negotiator.start()
            self._start_signaling_server()
        except Exception as exc:
            self.stop()
            raise StreamStartError(f"Failed to start GStreamer pipeline: {exc}") from exc

        self._info = StreamInfo(
            session_id=self._session_id,
            signaling_url=f"ws://0.0.0.0:{self._signaling_port}/signaling/{self._session_id}",
            stream_url=f"http://0.0.0.0:{self._signaling_port}/stream/{self._session_id}",
            ice_servers=list(config.ice_servers),
            control_endpoint=f"http://0.0.0.0:{self._signaling_port}/control/{self._session_id}",
        )
        self._alive = True
        return self._info

    @property
    def session_token(self) -> str:
        """The token a client must present to the signaling server.

        Empty before :meth:`start` assigns a session id.  A client that
        holds the shared secret can derive this itself with
        :func:`maddening.cloud._auth.generate_session_token` from the
        session id in :attr:`StreamInfo.signaling_url`; this property is
        the in-process route for a caller that owns the session and has
        to hand the token to a viewer (it is a credential -- do not log
        it).
        """
        if not self._session_id:
            return ""
        return generate_session_token(self._session_id, self._secret)

    def stop(self) -> None:
        """Close the signaling server and its viewer, then the pipeline and the GLib loop.

        Returns once both threads (``selkies-signaling``, ``selkies-glib``)
        have ended.
        """
        self._alive = False
        self._stop_signaling_server()
        if self._negotiator is not None:
            self._negotiator.stop()
            self._negotiator = None
        self._set_media(None, None)

    def is_alive(self) -> bool:
        return self._alive

    def update_framebuffer_cpu(
        self,
        pixels: bytes,
        width: int,
        height: int,
        pixel_format: str = "RGBA",
    ) -> None:
        """Push one raw frame; dropped while no viewer is connected.

        Raises
        ------
        ValueError
            When the frame's size, format or byte count does not match
            the stream's :class:`StreamConfig`: the encoder was set up for
            that geometry, and a frame of another one would stop the
            stream with an error nobody sees.
        """
        config = self._config
        if config is not None:
            expected = (config.width, config.height, config.pixel_format)
            if (width, height, pixel_format) != expected:
                raise ValueError(
                    f"frame is {width}x{height} {pixel_format}, but the stream was "
                    f"started for {config.width}x{config.height} {config.pixel_format}; "
                    "push frames of the configured size (a resolution change "
                    "needs a new session)")
            bpp = _BYTES_PER_PIXEL.get(pixel_format)
            if bpp is not None and len(pixels) != width * height * bpp:
                raise ValueError(
                    f"frame holds {len(pixels)} bytes; {width}x{height} "
                    f"{pixel_format} needs {width * height * bpp}")
        with self._media_lock:
            appsrc = self._appsrc
        if appsrc is None:
            return

        from gi.repository import Gst

        buf = Gst.Buffer.new_wrapped(bytes(pixels))
        appsrc.emit("push-buffer", buf)

    def update_framebuffer_gpu(self, buffer: GPUFramebuffer) -> None:
        # Try GstCudaMemory zero-copy, fallback to cudaMemcpyDtoH
        try:
            self._push_gpu_buffer(buffer)
        except Exception:
            # Fallback: copy to CPU and push
            logger.debug("GPU zero-copy failed, falling back to CPU copy")
            import ctypes
            size = buffer.height * buffer.stride_bytes
            host_buf = (ctypes.c_char * size)()
            try:
                import cupy
                cupy.cuda.runtime.memcpy(
                    ctypes.addressof(host_buf), buffer.cuda_ptr,
                    size, cupy.cuda.runtime.memcpyDeviceToHost,
                )
            except ImportError:
                logger.warning("Neither GstCudaMemory nor cupy available for GPU buffer")
                return
            self.update_framebuffer_cpu(
                bytes(host_buf), buffer.width, buffer.height,
                buffer.pixel_format,
            )

    def reconfigure(self, config: StreamConfig) -> None:
        if self._config is None:
            raise StreamReconfigError("Session not started")

        if (config.width != self._config.width
                or config.height != self._config.height):
            raise StreamReconfigError(
                f"Resolution change ({self._config.width}x{self._config.height}"
                f" -> {config.width}x{config.height}) requires session restart"
            )

        # Bitrate change via encoder property; the next viewer's pipeline
        # is built from the new config.
        with self._media_lock:
            encoder = self._encoder
        if encoder is not None and config.bitrate_kbps != self._config.bitrate_kbps:
            try:
                encoder.set_property("bitrate", config.bitrate_kbps)
            except Exception:
                logger.warning("Failed to set encoder bitrate")

        self._config = config

    def set_input_handler(
        self,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        self._input_handler = handler

    # -- Internal pipeline construction --------------------------------

    def _pipeline_description(self, payload_type: int) -> str:
        """The ``gst-launch`` description of one viewer's pipeline.

        ``do-timestamp`` stamps each frame with the time it was pushed
        (frames arrive when the simulation produces them, not at a fixed
        rate); the leaky queue keeps at most two raw frames, dropping the
        oldest, so a slow encoder never makes frames pile up; the I420
        caps stop ``videoconvert`` from choosing a 4:4:4 format the
        baseline profile cannot encode; ``key-int-max`` bounds the wait
        for a keyframe to about a second of frames; and the payloader
        carries the payload type the viewer's offer named.
        """
        config = self._config
        assert config is not None
        caps_str = (
            f"video/x-raw,format={config.pixel_format},"
            f"width={config.width},height={config.height},"
            f"framerate={config.fps}/1"
        )
        return (
            f"appsrc name=src is-live=true format=time do-timestamp=true "
            f"caps=\"{caps_str}\" ! "
            f"queue leaky=downstream max-size-buffers=2 max-size-bytes=0 "
            f"max-size-time=0 ! "
            f"videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc tune=zerolatency bitrate={config.bitrate_kbps} "
            f"speed-preset=ultrafast key-int-max={max(1, config.fps)} name=encoder ! "
            f"video/x-h264,profile=constrained-baseline ! "
            f"rtph264pay config-interval=-1 aggregate-mode=zero-latency "
            f"pt={payload_type} ! "
            f"application/x-rtp,media=video,encoding-name=H264,"
            f"payload={payload_type},clock-rate=90000 ! "
            f"webrtcbin name=webrtc bundle-policy=max-bundle"
        )

    def _check_pipeline(self, config: StreamConfig) -> None:
        """Build (and discard) a viewer's pipeline, so a missing element fails ``start()``."""
        from gi.repository import Gst

        pipeline = Gst.parse_launch(self._pipeline_description(96))
        pipeline.set_state(Gst.State.NULL)

    def _set_media(self, appsrc: Any, encoder: Any) -> None:
        with self._media_lock:
            self._appsrc, self._encoder = appsrc, encoder

    def _push_gpu_buffer(self, buffer: GPUFramebuffer) -> None:
        """Attempt GstCudaMemory zero-copy push."""
        raise NotImplementedError("GstCudaMemory zero-copy not yet implemented")

    def _start_signaling_server(self) -> None:
        """Start the WebSocket signaling server in a daemon thread.

        Returns once the socket listens; raises when it cannot (the port
        is taken, ``websockets`` is not installed), since a session no
        viewer can reach streams nothing.
        """
        handler = _make_signaling_handler(
            self._session_id, self._secret, lambda: self._input_handler,
            self._negotiator,
        )
        listening = threading.Event()
        failure: list[BaseException] = []

        async def _run_server():
            try:
                import websockets

                server = await websockets.serve(
                    handler, self._bind_host, self._signaling_port,
                )
            except Exception as exc:  # noqa: BLE001 - reported by start()
                failure.append(exc)
                listening.set()
                return
            self._signaling_server = server
            self._signaling_loop = asyncio.get_running_loop()
            self._signaling_port = server.sockets[0].getsockname()[1]
            logger.info(
                "WebRTC signaling server listening on %s:%s",
                self._bind_host, self._signaling_port,
            )
            listening.set()
            await server.wait_closed()

        def _thread_target():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(_run_server())
            except Exception:
                logger.debug("Signaling server stopped", exc_info=True)
            finally:
                loop.close()

        self._signaling_thread = threading.Thread(
            target=_thread_target, daemon=True,
            name="selkies-signaling",
        )
        self._signaling_thread.start()
        if not listening.wait(_THREAD_TIMEOUT_S):
            raise StreamStartError("the signaling server did not start listening")
        if failure:
            raise StreamStartError(f"the signaling server could not listen: {failure[0]}")

    def _stop_signaling_server(self) -> None:
        loop, server = self._signaling_loop, self._signaling_server
        if loop is not None and server is not None:
            with contextlib.suppress(RuntimeError):  # the loop already ended
                loop.call_soon_threadsafe(server.close)
        if self._signaling_thread is not None:
            self._signaling_thread.join(_THREAD_TIMEOUT_S)
            if self._signaling_thread.is_alive():
                logger.warning("The selkies-signaling thread did not stop within %s s",
                               _THREAD_TIMEOUT_S)
        self._signaling_thread = None
        self._signaling_loop = self._signaling_server = None
