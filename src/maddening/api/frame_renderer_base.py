"""
Abstract base class for server-side frame renderers.

Any renderer that can produce compressed image bytes from simulation
state can implement this interface and be passed to ``SimulationServer``
for WebSocket streaming via ``/ws/render``.

This is deliberately backend-agnostic -- implementations may use
matplotlib, PyVista/VTK, moderngl, or any other rendering library.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class ServerFrameRendererBase(ABC):
    """Abstract base for server-side frame renderers.

    Implementations render ``(sim_time, state) -> bytes`` where
    ``state`` is a nested dict ``{node_name: {field: value}}``.
    The server's ``/ws/render`` WebSocket endpoint calls ``render()``
    each tick and sends the bytes to the browser.
    """

    @property
    @abstractmethod
    def width(self) -> int:
        """Output image width in pixels."""

    @property
    @abstractmethod
    def height(self) -> int:
        """Output image height in pixels."""

    @property
    @abstractmethod
    def fmt(self) -> str:
        """Image format name (``"jpeg"``, ``"png"``, ``"webp"``)."""

    @property
    @abstractmethod
    def content_type(self) -> str:
        """MIME type for the current format (e.g. ``"image/jpeg"``)."""

    @abstractmethod
    def render(self, sim_time: float, state: dict) -> bytes:
        """Render the current state to compressed image bytes.

        Parameters
        ----------
        sim_time : float
            Current simulation time.
        state : dict
            Nested ``{node_name: {field: value}}`` state dict.

        Returns
        -------
        bytes
            Encoded image data (JPEG/WebP/PNG).
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear any accumulated state (e.g. time-series buffers)."""

    @abstractmethod
    def set_format(self, fmt: str, quality: Optional[int] = None) -> None:
        """Change the output image format and/or quality."""

    def close(self) -> None:
        """Release rendering resources.  Optional override."""
