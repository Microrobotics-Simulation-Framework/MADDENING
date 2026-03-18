"""SelkiesRenderer — Renderer wrapper that streams via WebRTC.

Wraps an inner ``Renderer`` and a ``StreamingSession`` via composition.
Each ``update()`` call renders locally, then pushes the framebuffer to
the streaming session for WebRTC delivery.
"""

from __future__ import annotations

import warnings
from typing import Optional

from maddening.cloud.streaming import (
    StreamConfig,
    StreamInfo,
    StreamingSession,
)
from maddening.viz.renderer import GraphInfo, Renderer
from maddening.warnings import PerformanceWarning


class SelkiesRenderer(Renderer):
    """Renderer that streams frames to a remote viewer via WebRTC.

    Parameters
    ----------
    inner : Renderer
        The actual rendering backend (e.g. MatplotlibRenderer).
    session : StreamingSession
        The streaming session to push frames through.
    config : StreamConfig, optional
        Stream configuration.  Defaults to ``StreamConfig()``.
    """

    def __init__(
        self,
        inner: Renderer,
        session: StreamingSession,
        config: Optional[StreamConfig] = None,
    ) -> None:
        self._inner = inner
        self._session = session
        self._config = config or StreamConfig()
        self._stream_info: Optional[StreamInfo] = None
        self._gpu_path = False

    # -- Renderer interface --------------------------------------------

    def setup(self, graph_info: GraphInfo) -> None:
        """Set up inner renderer and start the streaming session."""
        self._inner.setup(graph_info)

        # Detect GPU path
        self._gpu_path = hasattr(self._inner, "read_framebuffer_gpu")
        if not self._gpu_path:
            warnings.warn(
                "Inner renderer does not support GPU framebuffer reads; "
                "falling back to CPU path (extra GPU->CPU->GPU copy).",
                PerformanceWarning,
                stacklevel=2,
            )

        self._stream_info = self._session.start(self._config)

    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        """Render a frame and push it to the stream."""
        self._inner.update(sim_time, state)

        if self._gpu_path:
            gpu_buf = self._inner.read_framebuffer_gpu()  # type: ignore[attr-defined]
            self._session.update_framebuffer_gpu(gpu_buf)
        else:
            # CPU fallback: read pixels from inner renderer
            if hasattr(self._inner, "read_framebuffer_cpu"):
                pixels, w, h, fmt = self._inner.read_framebuffer_cpu()  # type: ignore[attr-defined]
            else:
                # Absolute fallback — generate a placeholder frame
                w, h = self._config.width, self._config.height
                pixels = b"\x00" * (w * h * 4)
                fmt = "RGBA"
            self._session.update_framebuffer_cpu(pixels, w, h, fmt)

    def teardown(self) -> None:
        """Stop the stream and tear down the inner renderer."""
        self._session.stop()
        self._inner.teardown()

    def requested_fields(self) -> Optional[dict[str, list[str]]]:
        """Delegate to inner renderer."""
        return self._inner.requested_fields()

    # -- Public properties ---------------------------------------------

    @property
    def url(self) -> Optional[str]:
        """Stream URL, or None if ``setup()`` hasn't been called."""
        if self._stream_info is not None:
            return self._stream_info.stream_url
        return None

    @property
    def stream_info(self) -> Optional[StreamInfo]:
        """Full StreamInfo, or None if ``setup()`` hasn't been called."""
        return self._stream_info
