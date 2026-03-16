"""
Vessel flow 3D renderer -- server-side rendering of blood flow.

Renders a Y-bifurcation vessel colored by velocity magnitude or
pressure, producing JPEG bytes for WebSocket streaming to the browser.
Uses PyVista's offscreen rendering (no display required).

This renderer is designed for the ``vessel_flow_server`` example but
can be reused for any simulation where nodes have 1D scalar fields
mapped onto 3D centerline geometry.

Architecture note (HPC support)
-------------------------------
This renderer runs in the same process as the physics server.  For
HPC deployments where physics runs on a remote GPU node:

1. **Local rendering**: the server sends rendered frames over the
   network (current approach, via ``/ws/render``).  The client is a
   thin browser.  This works over any network.
2. **Remote rendering**: the server sends raw state data (via
   ``/ws/state`` or binary WS), and the client renders locally.
   This reduces server load but requires a capable client.
3. **Hybrid**: server renders at low fps for overview, client
   optionally renders locally at high fps.  The server's
   ``/ws/state`` and ``/ws/render`` endpoints coexist naturally.

The ``host="0.0.0.0"`` in the server allows connections from any
network interface, so the same server works for both localhost and
remote access without code changes.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from maddening.api.frame_renderer_base import ServerFrameRendererBase


@dataclass
class VesselTubeConfig:
    """Configuration for one vessel tube in the 3D rendering."""
    node: str
    field: str
    centerline: np.ndarray  # (N, 3)
    radius: float = 0.02
    n_sides: int = 12


@dataclass
class VesselRendererConfig:
    """Configuration for the vessel flow renderer.

    Parameters
    ----------
    tubes : list of VesselTubeConfig
        Vessel segments to render.
    cmap : str
        Colormap for the scalar field.
    clim : tuple of float or None
        Fixed color limits.  None = auto-range each frame.
    scalar_bar_title : str
        Title for the color bar.
    background : str
        Background color.
    camera_position : str
        Camera preset ("xy", "xz", "yz", "iso").
    camera_zoom : float
        Camera zoom factor.
    show_time : bool
        Show simulation time annotation.
    """
    tubes: list[VesselTubeConfig] = field(default_factory=list)
    cmap: str = "coolwarm"
    clim: Optional[tuple[float, float]] = None
    scalar_bar_title: str = "Velocity"
    background: str = "white"
    camera_position: str = "xy"
    camera_zoom: float = 1.5
    show_time: bool = True


class VesselFlowRenderer(ServerFrameRendererBase):
    """Server-side 3D renderer for vessel blood flow.

    Renders vessel tubes colored by a scalar field (velocity magnitude,
    pressure, etc.) using PyVista offscreen rendering.  Produces JPEG
    bytes suitable for WebSocket streaming.

    Parameters
    ----------
    config : VesselRendererConfig
        Rendering configuration.
    width : int
        Output image width.
    height : int
        Output image height.
    """

    def __init__(
        self,
        config: VesselRendererConfig,
        width: int = 960,
        height: int = 540,
    ):
        import pyvista as pv
        from scipy.spatial import cKDTree

        self._config = config
        self._width_val = width
        self._height_val = height
        self._fmt_val = "jpeg"
        self._quality = 80

        # Build tube geometry (only done once)
        self._tube_data: list[dict] = []
        for tc in config.tubes:
            n = len(tc.centerline)
            cells = np.zeros(n + 1, dtype=np.int64)
            cells[0] = n
            cells[1:] = np.arange(n)
            line = pv.PolyData(tc.centerline, lines=cells)
            tube = line.tube(radius=tc.radius, n_sides=tc.n_sides)

            tree = cKDTree(tc.centerline)
            _, idx = tree.query(tube.points)

            tube["scalars"] = np.zeros(tube.n_points, dtype=np.float32)

            self._tube_data.append({
                "config": tc,
                "tube": tube,
                "idx": idx,
                "n_center": n,
            })

        # Create offscreen plotter (persistent — reused each frame)
        self._plotter = pv.Plotter(
            off_screen=True,
            window_size=[width, height],
        )
        self._plotter.set_background(config.background)

        # Add tube meshes to plotter
        for i, td in enumerate(self._tube_data):
            show_bar = (i == 0)
            self._plotter.add_mesh(
                td["tube"],
                scalars="scalars",
                cmap=config.cmap,
                clim=config.clim or (0, 1),
                show_scalar_bar=show_bar,
                scalar_bar_args={
                    "title": config.scalar_bar_title,
                    "color": "black",
                } if show_bar else {},
                name=f"tube_{i}",
            )

        self._plotter.camera_position = config.camera_position
        self._plotter.camera.zoom(config.camera_zoom)

        self._time_actor = None

    @property
    def width(self) -> int:
        return self._width_val

    @property
    def height(self) -> int:
        return self._height_val

    @property
    def fmt(self) -> str:
        return self._fmt_val

    @property
    def content_type(self) -> str:
        return f"image/{self._fmt_val}"

    def render(self, sim_time: float, state: dict) -> bytes:
        """Render the current state to JPEG bytes."""
        # Update tube scalars
        for td in self._tube_data:
            tc = td["config"]
            idx = td["idx"]
            tube = td["tube"]
            n_center = td["n_center"]

            if tc.node not in state or tc.field not in state[tc.node]:
                continue

            raw = np.asarray(state[tc.node][tc.field], dtype=np.float32)

            # For 3D velocity fields, compute magnitude
            if raw.ndim > 1:
                # Velocity field: take magnitude at each grid cell,
                # then extract a 1D profile (e.g., along centerline)
                if raw.ndim == 4:  # (nx, ny, nz, 3)
                    mag = np.sqrt(np.sum(raw ** 2, axis=-1))
                    # Extract a 1D profile through the center
                    ny, nz = mag.shape[1], mag.shape[2]
                    raw = mag[:, ny // 2, nz // 2]
                elif raw.ndim == 2:  # (N, 3)
                    raw = np.sqrt(np.sum(raw ** 2, axis=-1))
                else:
                    raw = raw.ravel()

            # Interpolate to match centerline if needed
            if len(raw) != n_center:
                x_s = np.linspace(0, 1, len(raw))
                x_c = np.linspace(0, 1, n_center)
                raw = np.interp(x_c, x_s, raw).astype(np.float32)

            tube["scalars"] = raw[idx]

        # Update color limits if auto
        if self._config.clim is None:
            all_vals = []
            for td in self._tube_data:
                all_vals.append(td["tube"]["scalars"])
            if all_vals:
                flat = np.concatenate(all_vals)
                vmin, vmax = float(flat.min()), float(flat.max())
                if vmax - vmin < 1e-10:
                    vmax = vmin + 1.0
                for i in range(len(self._tube_data)):
                    self._plotter.update_scalar_bar_range(
                        [vmin, vmax], name=f"tube_{i}"
                    )

        # Time annotation
        if self._config.show_time:
            self._plotter.add_text(
                f"t = {sim_time:.3f}",
                position="upper_right",
                font_size=10,
                color="black",
                name="time_text",
            )

        # Render to image array
        self._plotter.render()
        img = self._plotter.screenshot(return_img=True)

        # Encode to JPEG
        from PIL import Image
        pil_img = Image.fromarray(img)
        buf = io.BytesIO()
        if self._fmt_val == "jpeg":
            pil_img.save(buf, format="JPEG", quality=self._quality)
        elif self._fmt_val == "webp":
            pil_img.save(buf, format="WEBP", quality=self._quality)
        else:
            pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def reset(self) -> None:
        """Clear accumulated state."""
        for td in self._tube_data:
            td["tube"]["scalars"] = np.zeros(
                td["tube"].n_points, dtype=np.float32
            )

    def set_format(self, fmt: str, quality: Optional[int] = None) -> None:
        """Change image format and/or quality."""
        if fmt in ("jpeg", "webp", "png"):
            self._fmt_val = fmt
        if quality is not None:
            self._quality = max(1, min(100, quality))

    def close(self) -> None:
        """Release the offscreen plotter."""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
