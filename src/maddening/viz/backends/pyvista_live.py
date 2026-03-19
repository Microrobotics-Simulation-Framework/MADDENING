"""
PyVista live renderer -- real-time 3D visualization via the Renderer ABC.

Renders curve tubes, line plots, and static meshes in an interactive
PyVista window that updates as the simulation runs.  Uses PyVista's
timer callback mechanism for smooth updates.

Usage::

    from maddening.viz.backends.pyvista_live import PyVistaLiveRenderer

    renderer = PyVistaLiveRenderer()
    renderer.add_curve_tube("parent", "temperature", centerline, radius=0.05)
    renderer.setup(graph_info)

    relay = StateRelay()
    relay.attach(gm)

    runner = RealtimeRunner(gm, relay, steps_per_frame=100)
    runner.start()

    renderer.run_live(relay)  # blocks until window closed

Requires: pyvista, scipy
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from maddening.viz._imports import _import_pyvista
from maddening.viz.renderer import Renderer, GraphInfo


# ------------------------------------------------------------------
# Config dataclasses
# ------------------------------------------------------------------

@dataclass
class CurveTubeConfig:
    """Configuration for a curve tube visualization."""
    node: str
    field: str
    centerline: np.ndarray  # (N, 3)
    radius: float = 0.02
    n_sides: int = 12
    cmap: str = "coolwarm"
    clim: Optional[tuple[float, float]] = None
    opacity: float = 1.0
    label: str = ""


@dataclass
class LinePlotConfig:
    """Configuration for a 1D line plot in 3D space."""
    node: str
    field: str
    positions: Optional[np.ndarray] = None  # (N,) x-coords
    y_offset: float = 0.0
    scale: float = 1.0
    color: str = "#2244AA"
    line_width: float = 3.0


# ------------------------------------------------------------------
# PyVistaLiveRenderer
# ------------------------------------------------------------------

class PyVistaLiveRenderer(Renderer):
    """Real-time 3D visualization backend using PyVista.

    Parameters
    ----------
    window_size : tuple of int
        Window dimensions (width, height).
    background : str
        Background colour.
    title : str
        Window title.
    """

    def __init__(
        self,
        window_size: tuple[int, int] = (1400, 800),
        background: str = "#f0f0f0",
        title: str = "MADDENING -- Live Simulation",
    ):
        self._window_size = window_size
        self._background = background
        self._title = title

        # Visualization configs (populated before setup)
        self._tube_configs: list[CurveTubeConfig] = []
        self._line_configs: list[LinePlotConfig] = []
        self._static_meshes: list[tuple] = []  # (mesh, color, opacity)

        # Runtime state (set during setup)
        self._plotter = None
        self._tube_meshes: list[dict] = []  # per-tube runtime data
        self._line_meshes: list[dict] = []  # per-line runtime data
        self._graph_info = None
        self._frame_count = 0

        # Live loop state
        self._relay = None
        self._runner = None
        self._paused = False

    # ------------------------------------------------------------------
    # Builder API (call before setup)
    # ------------------------------------------------------------------

    def add_curve_tube(
        self,
        node: str,
        field: str,
        centerline: np.ndarray,
        **kwargs,
    ) -> "PyVistaLiveRenderer":
        """Add a tube along a 3D centerline, colored by a scalar field."""
        self._tube_configs.append(CurveTubeConfig(
            node=node, field=field,
            centerline=np.asarray(centerline, dtype=np.float64),
            **kwargs,
        ))
        return self

    def add_line_plot(
        self,
        node: str,
        field: str,
        **kwargs,
    ) -> "PyVistaLiveRenderer":
        """Add a 1D scalar field rendered as a 3D line."""
        self._line_configs.append(LinePlotConfig(
            node=node, field=field, **kwargs,
        ))
        return self

    def add_static_mesh(
        self, mesh, color: str = "#BBBBBB", opacity: float = 0.3
    ) -> "PyVistaLiveRenderer":
        """Add a static mesh (vessel wall, bounding box, etc.)."""
        self._static_meshes.append((mesh, color, opacity))
        return self

    def requested_fields(self) -> Optional[dict[str, list[str]]]:
        """Declare which fields this renderer needs from the relay."""
        fields: dict[str, list[str]] = {}
        for cfg in self._tube_configs:
            fields.setdefault(cfg.node, []).append(cfg.field)
        for cfg in self._line_configs:
            fields.setdefault(cfg.node, []).append(cfg.field)
        return fields if fields else None

    # ------------------------------------------------------------------
    # Renderer ABC implementation
    # ------------------------------------------------------------------

    def setup(self, graph_info: GraphInfo) -> None:
        """Build the PyVista scene."""
        pv = _import_pyvista()
        from scipy.spatial import cKDTree

        self._graph_info = graph_info
        self._plotter = pv.Plotter(
            window_size=list(self._window_size),
            title=self._title,
        )
        self._plotter.set_background(self._background)

        # Static meshes
        for mesh, color, opacity in self._static_meshes:
            self._plotter.add_mesh(
                mesh, color=color, opacity=opacity, smooth_shading=True
            )

        # Curve tubes
        for cfg in self._tube_configs:
            n = len(cfg.centerline)
            cells = np.zeros(n + 1, dtype=np.int64)
            cells[0] = n
            cells[1:] = np.arange(n)
            line = pv.PolyData(cfg.centerline, lines=cells)
            tube = line.tube(radius=cfg.radius, n_sides=cfg.n_sides)

            # Build nearest-neighbor map from tube surface to centerline
            tree = cKDTree(cfg.centerline)
            _, idx = tree.query(tube.points)

            # Initialize with zeros
            tube["scalars"] = np.zeros(tube.n_points, dtype=np.float32)

            show_bar = (len(self._tube_meshes) == 0)  # only first tube
            self._plotter.add_mesh(
                tube,
                scalars="scalars",
                cmap=cfg.cmap,
                clim=cfg.clim or (0, 100),
                opacity=cfg.opacity,
                show_scalar_bar=show_bar,
                scalar_bar_args={
                    "title": cfg.label or f"{cfg.node}.{cfg.field}",
                    "color": "black",
                },
            )

            self._tube_meshes.append({
                "config": cfg,
                "tube": tube,
                "idx": idx,
                "n_center": n,
            })

        # Line plots
        for cfg in self._line_configs:
            n = 10  # will be resized on first update
            xs = np.linspace(0, 1, n, dtype=np.float32)
            ys = np.full(n, cfg.y_offset, dtype=np.float32)
            zs = np.zeros(n, dtype=np.float32)
            pts = np.column_stack([xs, ys, zs])
            cells = np.zeros(n + 1, dtype=np.int64)
            cells[0] = n
            cells[1:] = np.arange(n)
            line = pv.PolyData(pts, lines=cells)
            self._plotter.add_mesh(
                line, color=cfg.color, line_width=cfg.line_width,
            )
            self._line_meshes.append({
                "config": cfg,
                "line": line,
            })

        # Status text
        self._plotter.add_text(
            "t = 0.0000 s  [running]",
            position="upper_right",
            font_size=10,
            color="black",
            name="status_text",
        )

        self._plotter.add_axes()
        self._plotter.reset_camera()

    def update(self, sim_time: float, state: dict[str, dict]) -> None:
        """Update all visualizations with new state data."""
        if self._plotter is None:
            return

        # Update curve tubes
        for tdata in self._tube_meshes:
            cfg = tdata["config"]
            tube = tdata["tube"]
            idx = tdata["idx"]
            n_center = tdata["n_center"]

            if cfg.node not in state or cfg.field not in state[cfg.node]:
                continue

            scalars = np.asarray(state[cfg.node][cfg.field], dtype=np.float32)

            if len(scalars) != n_center:
                x_s = np.linspace(0, 1, len(scalars))
                x_c = np.linspace(0, 1, n_center)
                scalars = np.interp(x_c, x_s, scalars).astype(np.float32)

            tube["scalars"] = scalars[idx]

        # Update line plots
        for ldata in self._line_meshes:
            cfg = ldata["config"]
            line = ldata["line"]

            if cfg.node not in state or cfg.field not in state[cfg.node]:
                continue

            vals = np.asarray(state[cfg.node][cfg.field], dtype=np.float32)
            n = len(vals)

            if cfg.positions is not None:
                xs = np.asarray(cfg.positions, dtype=np.float32)
            else:
                xs = np.arange(n, dtype=np.float32)

            pts = np.column_stack([
                xs,
                np.full(n, cfg.y_offset, dtype=np.float32),
                vals * cfg.scale,
            ])

            if line.n_points != n:
                pv = _import_pyvista()
                cells = np.zeros(n + 1, dtype=np.int64)
                cells[0] = n
                cells[1:] = np.arange(n)
                ldata["line"] = pv.PolyData(pts, lines=cells)
            else:
                line.points = pts

        # Status text
        tag = "[paused]" if self._paused else "[running]"
        self._plotter.add_text(
            f"t = {sim_time:.4f} s  {tag}   frame {self._frame_count}",
            position="upper_right",
            font_size=10,
            color="black",
            name="status_text",
        )

        self._frame_count += 1
        self._plotter.render()

    def teardown(self) -> None:
        """Close the PyVista window."""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None

    # ------------------------------------------------------------------
    # Keyboard controls
    # ------------------------------------------------------------------

    def _toggle_pause(self):
        """Space: toggle pause/resume on the runner."""
        if self._runner is None:
            return
        self._paused = not self._paused
        if self._paused:
            self._runner.pause()
            print("  [PAUSED]")
        else:
            self._runner.resume()
            print("  [RESUMED]")

    def _speed_up(self):
        """Up: double the time scale."""
        if self._runner is None:
            return
        self._runner.time_scale = self._runner.time_scale * 2.0
        print(f"  Time scale: {self._runner.time_scale:.1f}x")

    def _speed_down(self):
        """Down: halve the time scale."""
        if self._runner is None:
            return
        self._runner.time_scale = max(0.1, self._runner.time_scale / 2.0)
        print(f"  Time scale: {self._runner.time_scale:.1f}x")

    # ------------------------------------------------------------------
    # Timer callback for live updates
    # ------------------------------------------------------------------

    def _on_timer(self, step):
        """Called by PyVista's timer at ~target_fps."""
        if self._relay is None:
            return
        sim_time, snapshot = self._relay.latest_snapshot()
        if snapshot is not None:
            self.update(sim_time, snapshot)

    # ------------------------------------------------------------------
    # Main-thread render loop
    # ------------------------------------------------------------------

    def run_live(
        self,
        relay,
        runner=None,
        target_fps: int = 30,
    ) -> None:
        """Open the interactive window and render live until closed.

        Uses PyVista's built-in timer callback mechanism for smooth
        updates.  The window stays open until the user closes it.

        Parameters
        ----------
        relay : StateRelay
            Thread-safe buffer providing ``latest_snapshot()``.
        runner : RealtimeRunner or None
            If provided, keyboard controls (Space=pause, Up/Down=speed)
            are connected to it.
        target_fps : int
            Target frames per second for rendering.
        """
        if self._plotter is None:
            raise RuntimeError("Call setup() before run_live()")

        self._relay = relay
        self._runner = runner

        # Keyboard bindings
        if runner is not None:
            self._plotter.add_key_event("space", self._toggle_pause)
            self._plotter.add_key_event("Up", self._speed_up)
            self._plotter.add_key_event("Down", self._speed_down)

        # Timer callback for polling relay + updating scene
        interval_ms = max(16, int(1000.0 / target_fps))
        self._plotter.add_timer_event(
            max_steps=10_000_000,  # effectively infinite
            duration=interval_ms,
            callback=self._on_timer,
        )

        print(f"\nLive rendering at ~{target_fps} fps")
        if runner is not None:
            print("Controls:")
            print("  Space       pause / resume simulation")
            print("  Up/Down     speed up / slow down")
            print("  Mouse       rotate, zoom, pan")
        print("Close the window to stop.\n")

        # This blocks until the user closes the window
        self._plotter.show()

        # Cleanup
        self._relay = None
        self._runner = None
