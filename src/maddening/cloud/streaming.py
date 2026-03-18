"""Streaming session ABCs and data types.

Defines the ``StreamingSession`` abstract base class and supporting
dataclasses for WebRTC-based viewport streaming.  All types are pure
Python (stdlib only) so downstream code can develop against the ABC
with zero external dependencies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


# ------------------------------------------------------------------
# Quality presets
# ------------------------------------------------------------------

class QualityPreset(Enum):
    """Predefined quality tiers for streaming."""

    PREVIEW = "preview"      # <30ms latency, lower bitrate
    STANDARD = "standard"    # <50ms latency, balanced
    CAPTURE = "capture"      # <100ms latency, max quality


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

@dataclass(frozen=True)
class StreamConfig:
    """Configuration for a streaming session."""

    width: int = 1280
    height: int = 720
    fps: int = 30
    bitrate_kbps: int = 4000
    codec: str = "h264"
    pixel_format: str = "RGBA"
    enable_audio: bool = False
    ice_servers: list[dict[str, Any]] = field(default_factory=lambda: [
        {"urls": ["stun:stun.l.google.com:19302"]},
    ])

    @classmethod
    def from_preset(cls, preset: QualityPreset, **overrides) -> "StreamConfig":
        """Create a StreamConfig from a quality preset."""
        defaults = {
            QualityPreset.PREVIEW: dict(
                width=854, height=480, fps=24,
                bitrate_kbps=1500, codec="h264",
            ),
            QualityPreset.STANDARD: dict(
                width=1280, height=720, fps=30,
                bitrate_kbps=4000, codec="h264",
            ),
            QualityPreset.CAPTURE: dict(
                width=1920, height=1080, fps=60,
                bitrate_kbps=12000, codec="h264",
            ),
        }
        params = {**defaults[preset], **overrides}
        return cls(**params)

    @classmethod
    def from_dict(cls, d: dict) -> "StreamConfig":
        """Reconstruct from a plain dict (e.g. JSON deserialization)."""
        return cls(**d)


# ------------------------------------------------------------------
# Stream info (returned after start)
# ------------------------------------------------------------------

@dataclass(frozen=True)
class StreamInfo:
    """Information about an active stream, returned by ``start()``."""

    session_id: str
    signaling_url: str
    stream_url: str
    ice_servers: list[dict[str, Any]]
    control_endpoint: str = ""


# ------------------------------------------------------------------
# GPU framebuffer descriptor
# ------------------------------------------------------------------

@dataclass(frozen=True)
class GPUFramebuffer:
    """Describes a GPU framebuffer for zero-copy streaming."""

    cuda_ptr: int
    width: int
    height: int
    stride_bytes: int
    pixel_format: str = "RGBA"
    cuda_stream: int = 0
    fence_value: int = 0


# ------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------

class StreamStartError(Exception):
    """Raised when a streaming session fails to start."""

    pass


class StreamReconfigError(Exception):
    """Raised when reconfiguration is not possible (e.g. resolution change)."""

    pass


# ------------------------------------------------------------------
# StreamingSession ABC
# ------------------------------------------------------------------

class StreamingSession(ABC):
    """Abstract base for WebRTC viewport streaming sessions."""

    @abstractmethod
    def start(self, config: StreamConfig) -> StreamInfo:
        """Start the streaming session and return connection info."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Stop the streaming session and release resources."""
        ...

    @abstractmethod
    def is_alive(self) -> bool:
        """Return True if the session is actively streaming."""
        ...

    @abstractmethod
    def update_framebuffer_cpu(
        self,
        pixels: bytes,
        width: int,
        height: int,
        pixel_format: str = "RGBA",
    ) -> None:
        """Push a CPU-side framebuffer to the stream."""
        ...

    @abstractmethod
    def update_framebuffer_gpu(self, buffer: GPUFramebuffer) -> None:
        """Push a GPU-resident framebuffer to the stream (zero-copy)."""
        ...

    @abstractmethod
    def reconfigure(self, config: StreamConfig) -> None:
        """Reconfigure the stream (bitrate, codec).

        Raises ``StreamReconfigError`` for changes that require a
        session restart (e.g. resolution).
        """
        ...

    def set_input_handler(
        self,
        handler: Callable[[dict[str, Any]], None],
    ) -> None:
        """Register a callback for client-side input events.

        The default implementation is a no-op.  Concrete subclasses
        override this to receive keyboard/mouse/gamepad events from the
        remote viewer.
        """
        pass
