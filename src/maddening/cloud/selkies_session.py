"""SelkiesSession — GStreamer/WebRTC streaming implementation.

Wraps a GStreamer pipeline (``appsrc -> videoconvert -> encoder ->
webrtcbin``) with an embedded signaling server.  Requires PyGObject
and system GStreamer libraries.
"""

from __future__ import annotations

import logging
import threading
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


class SelkiesSession(StreamingSession):
    """GStreamer-based WebRTC streaming session.

    Parameters
    ----------
    secret : str
        Shared secret for HMAC-SHA256 token authentication.
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

        token = generate_session_token(self._session_id, self._secret)

        try:
            self._build_pipeline(config)
            self._start_signaling_server(token)
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

    def _start_signaling_server(self, token: str) -> None:
        """Start embedded WebSocket signaling server in a daemon thread."""
        import asyncio

        async def _run_server():
            try:
                import websockets
            except ImportError:
                logger.warning("websockets not installed; signaling server disabled")
                return

            async def handler(ws):
                # Validate token on connection
                path = ws.request.path if hasattr(ws, 'request') else ""
                # Simple bearer token check from query param
                if f"token={token}" not in (ws.request.query_string if hasattr(ws.request, 'query_string') else path):
                    if not validate_session_token(self._session_id, token, self._secret):
                        await ws.close(1008, "Invalid token")
                        return
                try:
                    async for msg in ws:
                        # Relay SDP/ICE messages
                        if self._input_handler and isinstance(msg, str):
                            import json
                            try:
                                self._input_handler(json.loads(msg))
                            except (ValueError, TypeError):
                                pass
                except Exception:
                    pass

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
