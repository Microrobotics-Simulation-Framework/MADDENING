"""SelkiesSession — GStreamer/WebRTC streaming implementation.

Wraps a GStreamer pipeline (``appsrc -> videoconvert -> encoder ->
webrtcbin``) with an embedded signaling server.  Requires PyGObject
and system GStreamer libraries.
"""

from __future__ import annotations

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
            gi.require_version("GstWebRTC", "1.0")
            from gi.repository import Gst
            _HAS_GST = True
        except (ImportError, ValueError):
            _HAS_GST = False
    return _HAS_GST


HAS_GSTREAMER = property(lambda self: _check_gstreamer())

#: WebSocket close code for a policy violation (RFC 6455 §7.4.1).
_CLOSE_POLICY_VIOLATION = 1008


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


def _make_signaling_handler(
    session_id: str,
    secret: str,
    get_input_handler: Callable[[], Optional[Callable[[dict[str, Any]], None]]],
):
    """Build the signaling WebSocket handler for one session.

    The returned coroutine authenticates the *client* before relaying
    anything: it reads the token the client presented (see
    :func:`_client_token`) and validates that token -- not the server's
    own -- against *session_id* and *secret* with
    :func:`maddening.cloud._auth.validate_session_token`, which compares
    in constant time.  An absent, malformed, wrong or
    wrong-session token is closed with 1008 before the first message is
    read.

    Parameters
    ----------
    session_id : str
        The session the token must be bound to.
    secret : str
        Shared HMAC secret.
    get_input_handler : callable
        Called per message to fetch the current input handler, so a
        handler installed after ``start()`` is still used.

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
        try:
            async for msg in ws:
                # Relay SDP/ICE messages
                input_handler = get_input_handler()
                if input_handler and isinstance(msg, str):
                    try:
                        input_handler(json.loads(msg))
                    except (ValueError, TypeError):
                        pass
        except Exception:  # noqa: BLE001 - a dropped client is not an error
            logger.debug("Signaling connection closed", exc_info=True)

    return handler


class SelkiesSession(StreamingSession):
    """GStreamer-based WebRTC streaming session.

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
        Port for the embedded WebSocket signaling server.
    """

    def __init__(
        self,
        secret: str = "",
        signaling_port: int = 8443,
    ) -> None:
        if not _check_gstreamer():
            raise ImportError(
                "SelkiesSession requires PyGObject and GStreamer. "
                "Install with: pip install PyGObject && "
                "apt install gstreamer1.0-plugins-base gstreamer1.0-plugins-good "
                "gstreamer1.0-plugins-bad gstreamer1.0-nice"
            )

        self._secret = secret or uuid.uuid4().hex
        self._ephemeral_secret = not secret
        if self._ephemeral_secret:
            logger.warning(
                "SelkiesSession was constructed without a shared secret; a "
                "random one was generated. Signaling clients cannot compute a "
                "token from a secret nobody holds, so every external "
                "connection will be rejected. Pass secret=... (the cloud "
                "entry point reads MADDENING_STREAM_SECRET) to let a client "
                "in.",
            )
        self._signaling_port = signaling_port
        self._alive = False
        self._config: Optional[StreamConfig] = None
        self._info: Optional[StreamInfo] = None
        self._pipeline = None
        self._appsrc = None
        self._encoder = None
        self._signaling_thread: Optional[threading.Thread] = None
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
            self._build_pipeline(config)
            self._start_signaling_server()
        except Exception as exc:
            raise StreamStartError(f"Failed to start GStreamer pipeline: {exc}")

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
        if self._pipeline is not None:
            from gi.repository import Gst
            self._pipeline.set_state(Gst.State.NULL)
            self._pipeline = None
        self._alive = False

    def is_alive(self) -> bool:
        return self._alive

    def update_framebuffer_cpu(
        self,
        pixels: bytes,
        width: int,
        height: int,
        pixel_format: str = "RGBA",
    ) -> None:
        if self._appsrc is None:
            return

        from gi.repository import Gst

        buf = Gst.Buffer.new_wrapped(bytes(pixels))
        buf.pts = Gst.CLOCK_TIME_NONE
        buf.duration = Gst.CLOCK_TIME_NONE
        self._appsrc.emit("push-buffer", buf)

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

        # Bitrate change via encoder property
        if self._encoder is not None and config.bitrate_kbps != self._config.bitrate_kbps:
            try:
                self._encoder.set_property("bitrate", config.bitrate_kbps)
            except Exception:
                logger.warning("Failed to set encoder bitrate")

        self._config = config

    def set_input_handler(
        self,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        self._input_handler = handler

    # -- Internal pipeline construction --------------------------------

    def _build_pipeline(self, config: StreamConfig) -> None:
        from gi.repository import Gst

        caps_str = (
            f"video/x-raw,format={config.pixel_format},"
            f"width={config.width},height={config.height},"
            f"framerate={config.fps}/1"
        )

        pipeline_str = (
            f"appsrc name=src is-live=true format=time "
            f"caps=\"{caps_str}\" ! "
            f"videoconvert ! "
            f"x264enc tune=zerolatency bitrate={config.bitrate_kbps} "
            f"speed-preset=ultrafast name=encoder ! "
            f"rtph264pay ! "
            f"webrtcbin name=webrtc bundle-policy=max-bundle"
        )

        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsrc = self._pipeline.get_by_name("src")
        self._encoder = self._pipeline.get_by_name("encoder")

        self._pipeline.set_state(Gst.State.PLAYING)

    def _push_gpu_buffer(self, buffer: GPUFramebuffer) -> None:
        """Attempt GstCudaMemory zero-copy push."""
        raise NotImplementedError("GstCudaMemory zero-copy not yet implemented")

    def _start_signaling_server(self) -> None:
        """Start embedded WebSocket signaling server in a daemon thread."""
        import asyncio

        handler = _make_signaling_handler(
            self._session_id, self._secret, lambda: self._input_handler,
        )

        async def _run_server():
            try:
                import websockets
            except ImportError:
                logger.warning("websockets not installed; signaling server disabled")
                return

            server = await websockets.serve(handler, "0.0.0.0", self._signaling_port)
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
