"""
Server-side frame rendering for browser streaming.

Renders simulation state to compressed image bytes (JPEG/WebP) using
matplotlib's non-interactive Agg backend.  Designed for streaming over
WebSocket to thin browser clients -- the server does all the heavy
lifting, the client just displays images.

Performance strategy:
    - Uses **matplotlib blitting** to avoid full figure redraws.
    - Static elements (axes, grid, labels, ticks) are rendered once
      and cached as a background bitmap.
    - Each frame only redraws *animated* artists (patches, lines,
      heatmap images) on top of the cached background.
    - A full redraw is triggered periodically (every ``redraw_every``
      frames) to update axis tick labels as data scrolls.
    - Typical per-frame cost: ~15-25 ms (vs ~500 ms for full redraw).

Usage::

    renderer = ServerFrameRenderer(
        scene=SceneConfig(objects=[...], xlim=(-2, 2), ylim=(-1, 8)),
        timeseries=[
            TimeSeriesConfig(fields=[("ball", "position", "Ball Y")], window=500),
        ],
    )
    jpeg_bytes = renderer.render(sim_time, state)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Callable, Optional

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import numpy as np

from maddening.api.frame_renderer_base import ServerFrameRendererBase

try:
    from PIL import Image as _PILImage
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


# ------------------------------------------------------------------
# Configuration dataclasses
# ------------------------------------------------------------------

@dataclass
class SceneObject:
    """A visual object in the 2D scene.

    Parameters
    ----------
    node : str
        Node name in the state dict.
    y : str
        State field mapped to the object's y position.
    kind : str
        ``"circle"``, ``"surface"``, or ``"hline"``.
    x : float or str
        Fixed x coordinate, or a state field name.
    radius : float
        Circle radius (for ``"circle"``).
    color : str
        Face colour.
    edgecolor : str
        Edge colour (for ``"circle"``).
    depth : float
        Filled depth below surface line (for ``"surface"``).
    linecolor : str
        Line colour (for ``"surface"`` / ``"hline"``).
    linewidth : float
        Line width.
    label : str
        Legend label (empty = no legend entry).
    """
    node: str
    y: str
    kind: str = "circle"
    x: float | str = 0.0
    radius: float = 0.2
    color: str = "#4488DD"
    edgecolor: str = ""
    depth: float = 0.5
    linecolor: str = "black"
    linewidth: float = 1.5
    label: str = ""


@dataclass
class SceneConfig:
    """Configuration for the 2D scene panel."""
    objects: list[SceneObject | dict]
    xlim: tuple[float, float] = (-2, 2)
    ylim: tuple[float, float] = (-1, 8)
    title: str = ""
    aspect: str = "equal"
    bgcolor: str = "#f8f8f8"


@dataclass
class TimeSeriesConfig:
    """Configuration for a time-series panel.

    Parameters
    ----------
    fields : list of (node, field, label)
        State fields to plot.
    window : int
        Number of data points to keep visible (0 = all).
    title : str
        Subplot title.
    ylabel : str
        Y-axis label.
    """
    fields: list[tuple[str, str, str]]
    window: int = 500
    title: str = ""
    ylabel: str = ""


@dataclass
class HeatmapConfig:
    """Configuration for a heatmap panel (array fields like temperature).

    Parameters
    ----------
    node : str
        Node name.
    field : str
        State field containing the array.
    title : str
        Subplot title.
    vmin, vmax : float
        Colour scale range.
    cmap : str
        Matplotlib colourmap name.
    """
    node: str
    field: str
    title: str = ""
    vmin: float = 0.0
    vmax: float = 100.0
    cmap: str = "hot"


# ------------------------------------------------------------------
# ServerFrameRenderer
# ------------------------------------------------------------------

class ServerFrameRenderer(ServerFrameRendererBase):
    """Matplotlib-based server-side frame renderer.

    Uses matplotlib's Agg backend (no display needed) with **blitting**
    for high frame rates.  Only animated artists are redrawn each frame;
    a full figure redraw happens every ``redraw_every`` frames to update
    axis decorations.

    Parameters
    ----------
    scene : SceneConfig, optional
        2D scene panel configuration.
    timeseries : list of TimeSeriesConfig, optional
        Time-series panel configurations.
    heatmaps : list of HeatmapConfig, optional
        Heatmap panel configurations.
    width, height : int
        Output image dimensions in pixels.
    dpi : int
        Rendering DPI.
    fmt : str
        Image format: ``"jpeg"``, ``"webp"``, or ``"png"``.
    quality : int
        JPEG/WebP quality (1-100).
    redraw_every : int
        Full figure redraw interval (frames).  Between full redraws,
        only animated artists are blitted for speed.  Set to 1 to force
        full redraws every frame (slower but pixel-perfect axis labels).
    """

    def __init__(
        self,
        scene: Optional[SceneConfig] = None,
        timeseries: Optional[list[TimeSeriesConfig]] = None,
        heatmaps: Optional[list[HeatmapConfig]] = None,
        width: int = 1280,
        height: int = 720,
        dpi: int = 100,
        fmt: str = "jpeg",
        quality: int = 85,
        redraw_every: int = 30,
    ):
        self.scene_config = scene
        self.timeseries_configs = timeseries or []
        self.heatmap_configs = heatmaps or []
        self._width = width
        self._height = height
        self.dpi = dpi
        self._fmt = fmt
        self.quality = quality
        self.redraw_every = max(1, redraw_every)

        # Artists / buffers populated by _build_figure
        self._fig: Optional[Figure] = None
        self._canvas: Optional[FigureCanvasAgg] = None
        self._scene_ax = None
        self._scene_artists: list[tuple] = []
        self._scene_time_text = None
        self._ts_axes: list = []
        self._ts_lines: dict = {}
        self._time_buffers: dict = {}
        self._data_buffers: dict = {}
        self._hm_axes: list = []
        self._hm_images: list = []

        # Blitting state
        self._background = None   # cached full-figure background
        self._frame_count = 0
        self._all_animated: list = []  # (ax, artist) pairs

        self._build_figure()

    # ------------------------------------------------------------------
    # Properties (ServerFrameRendererBase interface)
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def fmt(self) -> str:
        return self._fmt

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _build_figure(self):
        fig_w = self._width / self.dpi
        fig_h = self._height / self.dpi

        self._fig = Figure(figsize=(fig_w, fig_h), dpi=self.dpi,
                           facecolor="white")
        self._canvas = FigureCanvasAgg(self._fig)

        has_scene = self.scene_config is not None
        n_ts = len(self.timeseries_configs)
        n_hm = len(self.heatmap_configs)
        n_right = n_ts + n_hm

        if has_scene and n_right > 0:
            gs = GridSpec(max(n_right, 1), 2, figure=self._fig,
                          width_ratios=[1, 1])
            self._scene_ax = self._fig.add_subplot(gs[:, 0])
            idx = 0
            self._ts_axes = []
            for _ in self.timeseries_configs:
                self._ts_axes.append(self._fig.add_subplot(gs[idx, 1]))
                idx += 1
            self._hm_axes = []
            for _ in self.heatmap_configs:
                self._hm_axes.append(self._fig.add_subplot(gs[idx, 1]))
                idx += 1
        elif has_scene:
            self._scene_ax = self._fig.add_subplot(111)
        elif n_right > 0:
            gs = GridSpec(n_right, 1, figure=self._fig)
            idx = 0
            self._ts_axes = []
            for _ in self.timeseries_configs:
                self._ts_axes.append(self._fig.add_subplot(gs[idx, 0]))
                idx += 1
            self._hm_axes = []
            for _ in self.heatmap_configs:
                self._hm_axes.append(self._fig.add_subplot(gs[idx, 0]))
                idx += 1
        else:
            self._scene_ax = self._fig.add_subplot(111)

        if has_scene:
            self._init_scene()
        self._init_timeseries()
        self._init_heatmaps()
        self._fig.tight_layout(pad=1.5)

        # Initial full draw to populate the background
        self._canvas.draw()
        self._background = self._canvas.copy_from_bbox(self._fig.bbox)
        self._frame_count = 0

    def _init_scene(self):
        ax = self._scene_ax
        cfg = self.scene_config
        ax.set_xlim(cfg.xlim)
        ax.set_ylim(cfg.ylim)
        ax.set_aspect(cfg.aspect)
        ax.set_facecolor(cfg.bgcolor)
        ax.set_xticks([])
        ax.set_ylabel("Height (m)")
        ax.grid(True, alpha=0.3)
        if cfg.title:
            ax.set_title(cfg.title, fontsize=11)

        self._scene_time_text = ax.text(
            0.02, 0.97, "t = 0.00 s", transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
        )
        self._all_animated.append((ax, self._scene_time_text))

        for obj in cfg.objects:
            if isinstance(obj, dict):
                obj = SceneObject(**obj)

            if obj.kind == "circle":
                ec = obj.edgecolor or obj.color
                artist = mpatches.Circle(
                    (float(obj.x) if not isinstance(obj.x, str) else 0.0,
                     0.0),
                    obj.radius,
                    facecolor=obj.color, edgecolor=ec,
                    linewidth=obj.linewidth, zorder=5,
                )
                ax.add_patch(artist)
                self._scene_artists.append((obj, artist))
                self._all_animated.append((ax, artist))

            elif obj.kind == "surface":
                w = cfg.xlim[1] - cfg.xlim[0]
                rect = mpatches.Rectangle(
                    (cfg.xlim[0], -obj.depth), w, obj.depth,
                    facecolor=obj.color, linewidth=0, zorder=1,
                )
                ax.add_patch(rect)
                line = ax.axhline(
                    0, color=obj.linecolor,
                    linewidth=obj.linewidth, zorder=2,
                )
                self._scene_artists.append((obj, (rect, line)))
                self._all_animated.append((ax, rect))
                self._all_animated.append((ax, line))

            elif obj.kind == "hline":
                line = ax.axhline(
                    0, color=obj.color,
                    linewidth=obj.linewidth, zorder=2,
                )
                self._scene_artists.append((obj, line))
                self._all_animated.append((ax, line))

    def _init_timeseries(self):
        self._ts_lines = {}
        for i, cfg in enumerate(self.timeseries_configs):
            ax = self._ts_axes[i]
            if cfg.title:
                ax.set_title(cfg.title, fontsize=10)
            if cfg.ylabel:
                ax.set_ylabel(cfg.ylabel, fontsize=9)
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.grid(True, alpha=0.3)

            for node, field_name, label in cfg.fields:
                key = f"{node}.{field_name}"
                line, = ax.plot([], [], label=label, linewidth=1.5)
                self._ts_lines[key] = (cfg, line, ax)
                self._time_buffers[key] = []
                self._data_buffers[key] = []
                self._all_animated.append((ax, line))

            if cfg.fields:
                ax.legend(loc="upper right", fontsize=8)

    def _init_heatmaps(self):
        self._hm_images = []
        for i, cfg in enumerate(self.heatmap_configs):
            ax = self._hm_axes[i]
            if cfg.title:
                ax.set_title(cfg.title, fontsize=10)
            im = ax.imshow(
                np.zeros((1, 20)),
                aspect="auto", cmap=cfg.cmap,
                vmin=cfg.vmin, vmax=cfg.vmax,
                interpolation="nearest",
            )
            ax.set_yticks([])
            ax.set_xlabel("Cell index", fontsize=9)
            self._hm_images.append((cfg, im, ax))
            self._all_animated.append((ax, im))

    # ------------------------------------------------------------------
    # State update (modifies artist properties, does NOT draw)
    # ------------------------------------------------------------------

    def _update_scene(self, sim_time: float, state: dict):
        if self._scene_time_text is not None:
            self._scene_time_text.set_text(f"t = {sim_time:.2f} s")

        for obj, artist in self._scene_artists:
            if isinstance(obj, dict):
                obj = SceneObject(**obj)

            node = obj.node
            if node not in state:
                continue
            node_state = state[node]
            y_field = obj.y
            if y_field not in node_state:
                continue
            y_val = float(node_state[y_field])

            if obj.kind == "circle":
                if isinstance(obj.x, str) and obj.x in node_state:
                    x_val = float(node_state[obj.x])
                else:
                    x_val = float(obj.x) if not isinstance(obj.x, str) else 0.0
                artist.set_center((x_val, y_val + obj.radius))

            elif obj.kind == "surface":
                rect, line = artist
                rect.set_y(y_val - obj.depth)
                line.set_ydata([y_val, y_val])

            elif obj.kind == "hline":
                artist.set_ydata([y_val, y_val])

    def _update_timeseries(self, sim_time: float, state: dict):
        for key, (cfg, line, ax) in self._ts_lines.items():
            node, field_name = key.split(".", 1)
            if node not in state or field_name not in state[node]:
                continue

            val = float(state[node][field_name])
            self._time_buffers[key].append(sim_time)
            self._data_buffers[key].append(val)

            if cfg.window > 0 and len(self._time_buffers[key]) > cfg.window:
                self._time_buffers[key] = self._time_buffers[key][-cfg.window:]
                self._data_buffers[key] = self._data_buffers[key][-cfg.window:]

            line.set_data(self._time_buffers[key], self._data_buffers[key])
            ax.relim()
            ax.autoscale_view()

    def _update_heatmaps(self, state: dict):
        for cfg, im, ax in self._hm_images:
            if cfg.node not in state or cfg.field not in state[cfg.node]:
                continue
            data = state[cfg.node][cfg.field]
            arr = np.asarray(data, dtype=np.float32).reshape(1, -1)
            im.set_data(arr)
            im.set_extent([0, arr.shape[1], 0, 1])

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    def _encode_buffer(self) -> bytes:
        """Encode the current canvas buffer to image bytes."""
        w, h = self._canvas.get_width_height()

        if _HAS_PIL and self._fmt in ("jpeg", "jpg", "webp"):
            # Fast path: grab RGBA buffer → convert to RGB → PIL encode
            rgba = np.frombuffer(
                self._canvas.buffer_rgba(), dtype=np.uint8,
            ).reshape(h, w, 4)
            rgb = np.ascontiguousarray(rgba[:, :, :3])
            img = _PILImage.frombytes("RGB", (w, h), rgb.tobytes())
            buf = io.BytesIO()
            pil_kw = {"quality": self.quality} if self.fmt != "png" else {}
            fmt_name = "JPEG" if self.fmt in ("jpeg", "jpg") else self.fmt.upper()
            img.save(buf, format=fmt_name, **pil_kw)
            buf.seek(0)
            return buf.getvalue()

        # Fallback: use fig.savefig (slower, but works without PIL for PNG)
        buf = io.BytesIO()
        save_kw: dict = {"format": self.fmt, "facecolor": "white"}
        if self.fmt in ("jpeg", "jpg"):
            save_kw["pil_kwargs"] = {"quality": self.quality}
        elif self.fmt == "webp":
            save_kw["pil_kwargs"] = {"quality": self.quality}
        self._fig.savefig(buf, **save_kw)
        buf.seek(0)
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(self, sim_time: float, state: dict) -> bytes:
        """Render current state to compressed image bytes.

        Uses blitting for speed: most frames only redraw animated
        artists on a cached background.  Every ``redraw_every`` frames
        a full canvas draw is performed to update axis decorations.

        Parameters
        ----------
        sim_time : float
            Current simulation time.
        state : dict
            Nested ``{node_name: {field: value}}`` state dict.

        Returns
        -------
        bytes
            JPEG, WebP, or PNG image data.
        """
        # Update artist data
        if self.scene_config is not None:
            self._update_scene(sim_time, state)
        self._update_timeseries(sim_time, state)
        self._update_heatmaps(state)

        self._frame_count += 1

        if self._frame_count % self.redraw_every == 0 or self._background is None:
            # Full redraw — updates axis ticks, labels, grid, etc.
            self._canvas.draw()
            self._background = self._canvas.copy_from_bbox(self._fig.bbox)
        else:
            # Fast blit — only redraw animated artists
            self._canvas.restore_region(self._background)
            for ax, artist in self._all_animated:
                ax.draw_artist(artist)
            self._canvas.blit(self._fig.bbox)

        return self._encode_buffer()

    def reset(self):
        """Clear time-series buffers (e.g. after simulation reset)."""
        for key in self._time_buffers:
            self._time_buffers[key] = []
            self._data_buffers[key] = []
        # Force a full redraw on next render
        self._frame_count = self.redraw_every - 1

    def resize(self, width: int, height: int):
        """Change the output resolution.  Rebuilds the figure."""
        self._width = width
        self._height = height
        self.close()
        self._scene_artists = []
        self._all_animated = []
        self._hm_images = []
        self._ts_lines = {}
        self._build_figure()

    def set_format(self, fmt: str, quality: Optional[int] = None):
        """Change the image format and/or quality."""
        self._fmt = fmt
        if quality is not None:
            self.quality = quality

    @property
    def content_type(self) -> str:
        """MIME type for the current format."""
        return {
            "jpeg": "image/jpeg",
            "jpg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(self.fmt, "image/jpeg")

    def close(self):
        """Release matplotlib resources."""
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._canvas = None
            self._background = None
