"""Mock streaming session for testing (zero external dependencies).

``MockStreamSession`` implements the ``StreamingSession`` ABC using
in-memory frame storage.  It is suitable for unit tests and for
developing against the streaming API without GStreamer or WebRTC.
"""

from collections import deque
from typing import Any, Callable, Optional
import uuid

from maddening.cloud.streaming import (
    GPUFramebuffer,
    StreamConfig,
    StreamInfo,
    StreamReconfigError,
    StreamStartError,
    StreamingSession,
)


class MockStreamSession(StreamingSession):
    """In-memory mock of a streaming session.

    Parameters
    ----------
    max_frames : int
        Maximum number of CPU frames to retain in the ring buffer.
    fail_on_start : bool
        If True, ``start()`` raises ``StreamStartError``.
    """

    def __init__(
        self,
        max_frames: int = 100,
        fail_on_start: bool = False,
    ) -> None:
        self._max_frames = max_frames
        self._fail_on_start = fail_on_start
        self._alive = False
        self._config: Optional[StreamConfig] = None
        self._info: Optional[StreamInfo] = None
        self._frames: deque[bytes] = deque(maxlen=max_frames)
        self._gpu_frames: list[GPUFramebuffer] = []
        self._input_handler: Optional[Callable[[dict[str, Any]], None]] = None

    # -- StreamingSession interface ------------------------------------

    def start(self, config: StreamConfig) -> StreamInfo:
        if self._fail_on_start:
            raise StreamStartError("MockStreamSession configured to fail")
        self._config = config
        sid = uuid.uuid4().hex[:12]
        self._info = StreamInfo(
            session_id=sid,
            signaling_url=f"ws://localhost:0/signaling/{sid}",
            stream_url=f"http://localhost:0/stream/{sid}",
            ice_servers=list(config.ice_servers),
            control_endpoint=f"http://localhost:0/control/{sid}",
        )
        self._alive = True
        return self._info

    def stop(self) -> None:
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
        self._frames.append(pixels)

    def update_framebuffer_gpu(self, buffer: GPUFramebuffer) -> None:
        self._gpu_frames.append(buffer)

    def reconfigure(self, config: StreamConfig) -> None:
        if self._config is None:
            raise StreamReconfigError("Session not started")
        if (config.width != self._config.width
                or config.height != self._config.height):
            raise StreamReconfigError(
                f"Resolution change ({self._config.width}x{self._config.height}"
                f" -> {config.width}x{config.height}) requires session restart"
            )
        self._config = config

    def set_input_handler(
        self,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        self._input_handler = handler

    # -- Test helpers --------------------------------------------------

    @property
    def frame_count(self) -> int:
        """Number of CPU frames received."""
        return len(self._frames)

    @property
    def last_frame(self) -> Optional[bytes]:
        """Most recent CPU frame, or None."""
        return self._frames[-1] if self._frames else None

    @property
    def config(self) -> Optional[StreamConfig]:
        """Active stream configuration."""
        return self._config

    @property
    def info(self) -> Optional[StreamInfo]:
        """Stream info returned by ``start()``."""
        return self._info
