"""
3D server-side frame renderer using PyVista/VTK.

Renders 3D structured-grid data (LBM velocity/density fields, etc.)
to compressed image bytes for WebSocket streaming.  Uses VTK's offscreen
rendering -- no display required.

Usage::

    renderer = ServerFrameRenderer3D(
        config=View3DConfig(
            node="fluid",
            slices=[
                SliceConfig(node="fluid", field="velocity",
                            normal="x", origin_frac=0.5),
            ],
            pipe_wall=PipeWallConfig(radius_frac=0.9),
        ),
    )
    jpeg_bytes = renderer.render(sim_time, state)
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from maddening.viz._imports import _import_pyvista

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
class SliceConfig:
    """Cross-section through a 3D field, colored by scalar value.

    Parameters
    ----------
    node : str
        Node name in the state dict.
    field : str
        State field name (e.g. ``"velocity"``, ``"density"``).
    component : int
        For vector fields: ``-1`` for magnitude, ``0/1/2`` for x/y/z.
        Ignored for scalar fields.
    normal : str
        Slice plane normal axis: ``"x"``, ``"y"``, or ``"z"``.
    origin_frac : float
        Position along the normal axis as a fraction (0.0--1.0).
    cmap : str
        Colormap name.
    clim : tuple, optional
        Fixed color range ``(min, max)``.  Auto-ranged if ``None``.
    opacity : float
        Slice opacity (0--1).
    show_colorbar : bool
        Display a scalar bar for this slice.
    """
    node: str
    field: str
    component: int = -1
    normal: str = "x"
    origin_frac: float = 0.5
    cmap: str = "coolwarm"
    clim: tuple[float, float] | None = None
    opacity: float = 1.0
    show_colorbar: bool = True


@dataclass
class ArrowConfig:
    """Arrow glyphs showing a vector field on a cross-section.

    Parameters
    ----------
    node : str
        Node name.
    field : str
        Vector state field (must be 4-D with last dim = 3).
    normal : str
        Slice plane normal axis.
    origin_frac : float
        Position along the normal axis (0.0--1.0).
    scale : float
        Arrow length scale factor.
    stride : int
        Sub-sample stride (``2`` = every other point).
    cmap : str
        Colormap for arrow coloring (by velocity magnitude).
    clim : tuple, optional
        Fixed color range.
    """
    node: str
    field: str = "velocity"
    normal: str = "x"
    origin_frac: float = 0.5
    scale: float = 5.0
    stride: int = 2
    cmap: str = "coolwarm"
    clim: tuple[float, float] | None = None


@dataclass
class PipeWallConfig:
    """Translucent cylindrical pipe wall.

    Parameters
    ----------
    radius_frac : float
        Pipe radius as fraction of ``min(ny, nz) / 2``.
    color : str
        Wall colour.
    opacity : float
        Wall opacity (0--1).
    resolution : int
        Number of facets around the circumference.
    """
    radius_frac: float = 0.9
    color: str = "#AAAAAA"
    opacity: float = 0.12
    resolution: int = 60


@dataclass
class StreamlineConfig:
    """Streamlines through a vector field.

    Parameters
    ----------
    node : str
        Node name.
    field : str
        Vector state field.
    n_lines : int
        Approximate number of streamlines.
    source_radius_frac : float
        Seed-disc radius as fraction of ``min(ny, nz) / 2``.
    source_x_frac : float
        Seed-disc position along x (0.0--1.0).
    cmap : str
        Colormap for streamline coloring (by speed).
    tube_radius : float
        Tube radius for rendering.
    max_length : float
        Maximum streamline length.
    """
    node: str
    field: str = "velocity"
    n_lines: int = 25
    source_radius_frac: float = 0.6
    source_x_frac: float = 0.1
    cmap: str = "coolwarm"
    tube_radius: float = 0.08
    max_length: float = 200.0


@dataclass
class View3DConfig:
    """Top-level configuration for 3D server-side rendering.

    Parameters
    ----------
    node : str
        Primary simulation node name (used for grid discovery).
    grid_field : str
        Scalar state field used to determine grid dimensions.
    slices : list of SliceConfig
        Cross-section visualizations.
    arrows : list of ArrowConfig
        Arrow-glyph visualizations.
    pipe_wall : PipeWallConfig, optional
        Pipe wall cylinder.
    streamlines : list of StreamlineConfig
        Streamline visualizations.
    camera_position : str or tuple
        PyVista camera preset (``"xz"``, ``"xy"``, ``"yz"``, ``"iso"``)
        or ``(position, focal_point, view_up)`` tuple.
    camera_zoom : float
        Zoom factor (>1 zooms in).
    background : str
        Background colour.
    show_axes : bool
        Show orientation axes widget.
    show_time : bool
        Overlay simulation time text.
    """
    node: str = "fluid"
    grid_field: str = "density"
    slices: list[SliceConfig] = field(default_factory=list)
    arrows: list[ArrowConfig] = field(default_factory=list)
    pipe_wall: PipeWallConfig | None = None
    streamlines: list[StreamlineConfig] = field(default_factory=list)
    camera_position: str | tuple = "xz"
    camera_zoom: float = 1.0
    background: str = "white"
    show_axes: bool = True
    show_time: bool = True


# ------------------------------------------------------------------
# ServerFrameRenderer3D
# ------------------------------------------------------------------

class ServerFrameRenderer3D(ServerFrameRendererBase):
    """PyVista/VTK-based 3D server-side frame renderer.

    Renders structured 3D simulation data as cross-sections, arrow
    glyphs, streamlines, and pipe geometry.  Uses VTK's offscreen
    rendering -- no display required.

    Parameters
    ----------
    config : View3DConfig
        Visualization configuration.
    width, height : int
        Output image dimensions in pixels.
    fmt : str
        Image format: ``"jpeg"``, ``"webp"``, or ``"png"``.
    quality : int
        JPEG/WebP quality (1--100).
    """

    def __init__(
        self,
        config: View3DConfig,
        width: int = 1280,
        height: int = 720,
        fmt: str = "jpeg",
        quality: int = 85,
    ):
        self._config = config
        self._width = width
        self._height = height
        self._fmt = fmt
        self._quality = quality

        self._plotter = None
        self._grid_dims: tuple[int, int, int] | None = None
        self._initialized = False

        # Tracked meshes/actors for incremental updates
        self._slice_meshes: dict[str, object] = {}   # key -> pv.StructuredGrid
        self._slice_actors: dict[str, object] = {}    # key -> actor
        self._dynamic_actors: dict[str, object] = {}  # key -> actor (arrows, etc.)
        self._time_actor = None

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

    @property
    def content_type(self) -> str:
        return {
            "jpeg": "image/jpeg", "jpg": "image/jpeg",
            "png": "image/png", "webp": "image/webp",
        }.get(self._fmt, "image/jpeg")

    # ------------------------------------------------------------------
    # Plotter setup
    # ------------------------------------------------------------------

    def _ensure_plotter(self):
        if self._plotter is not None:
            return
        pv = _import_pyvista()
        pv.OFF_SCREEN = True
        self._plotter = pv.Plotter(
            off_screen=True,
            window_size=[self._width, self._height],
            lighting="three lights",
        )
        self._plotter.set_background(self._config.background)

    def _discover_grid(self, state: dict):
        """Infer grid dimensions from the first render's state."""
        cfg = self._config
        arr = np.asarray(state[cfg.node][cfg.grid_field])
        self._grid_dims = arr.shape[:3]  # (nx, ny, nz)

    def _build_static(self):
        """Add static elements (pipe wall, axes) to the plotter."""
        pv = _import_pyvista()
        cfg = self._config
        nx, ny, nz = self._grid_dims

        if cfg.pipe_wall is not None:
            pw = cfg.pipe_wall
            radius = pw.radius_frac * min(ny, nz) / 2.0
            cy, cz = (ny - 1) / 2.0, (nz - 1) / 2.0
            cyl = pv.Cylinder(
                center=((nx - 1) / 2.0, cy, cz),
                direction=(1, 0, 0),
                radius=radius,
                height=float(nx),
                resolution=pw.resolution,
                capping=True,
            )
            self._plotter.add_mesh(
                cyl, color=pw.color, opacity=pw.opacity,
                smooth_shading=True,
            )

        if cfg.show_axes:
            self._plotter.add_axes()

        self._plotter.camera_position = cfg.camera_position
        if cfg.camera_zoom != 1.0:
            self._plotter.camera.zoom(cfg.camera_zoom)

    # ------------------------------------------------------------------
    # Scalar extraction helpers
    # ------------------------------------------------------------------

    def _extract_scalar(self, state, node, field_name, component):
        """Extract a scalar field from state (handles vector magnitude)."""
        arr = np.asarray(state[node][field_name])
        if arr.ndim == 4:  # vector field (nx, ny, nz, 3)
            if component == -1:
                return np.linalg.norm(arr, axis=-1).astype(np.float32)
            return arr[..., component].astype(np.float32)
        return arr.astype(np.float32)

    def _slice_index(self, normal: str, origin_frac: float):
        """Convert normal axis + fraction to (axis_int, grid_index)."""
        axis = {"x": 0, "y": 1, "z": 2}[normal]
        dim = self._grid_dims[axis]
        idx = int(origin_frac * (dim - 1))
        return axis, max(0, min(idx, dim - 1))

    # ------------------------------------------------------------------
    # Cross-section slices
    # ------------------------------------------------------------------

    def _make_slice_grid(self, axis: int, idx: int):
        """Create a 2D StructuredGrid for a fixed slice plane."""
        pv = _import_pyvista()
        nx, ny, nz = self._grid_dims

        if axis == 0:  # x-normal
            a = np.arange(ny, dtype=np.float32)
            b = np.arange(nz, dtype=np.float32)
            aa, bb = np.meshgrid(a, b, indexing="ij")
            cc = np.full_like(aa, float(idx))
            return pv.StructuredGrid(cc, aa, bb)
        elif axis == 1:  # y-normal
            a = np.arange(nx, dtype=np.float32)
            b = np.arange(nz, dtype=np.float32)
            aa, bb = np.meshgrid(a, b, indexing="ij")
            cc = np.full_like(aa, float(idx))
            return pv.StructuredGrid(aa, cc, bb)
        else:  # z-normal
            a = np.arange(nx, dtype=np.float32)
            b = np.arange(ny, dtype=np.float32)
            aa, bb = np.meshgrid(a, b, indexing="ij")
            cc = np.full_like(aa, float(idx))
            return pv.StructuredGrid(aa, bb, cc)

    def _extract_slice_data(self, scalars, axis, idx):
        """Extract a 2D slice from a 3D scalar field."""
        if axis == 0:
            return scalars[idx, :, :]
        elif axis == 1:
            return scalars[:, idx, :]
        else:
            return scalars[:, :, idx]

    def _update_slices(self, state: dict):
        for i, sc in enumerate(self._config.slices):
            scalars = self._extract_scalar(state, sc.node, sc.field, sc.component)
            axis, idx = self._slice_index(sc.normal, sc.origin_frac)
            data = self._extract_slice_data(scalars, axis, idx)

            key = f"slice_{i}"
            if key in self._slice_meshes:
                # Update data in place -- VTK picks up the change
                mesh = self._slice_meshes[key]
                mesh.point_data["scalar"] = data.ravel(order="F")
            else:
                # First frame: create mesh and add to plotter
                mesh = self._make_slice_grid(axis, idx)
                mesh.point_data["scalar"] = data.ravel(order="F")
                self._slice_meshes[key] = mesh

                clim = sc.clim
                if clim is None:
                    dmin, dmax = float(data.min()), float(data.max())
                    if dmax - dmin < 1e-10:
                        dmax = dmin + 1.0
                    clim = (dmin, dmax)

                actor = self._plotter.add_mesh(
                    mesh, scalars="scalar", cmap=sc.cmap,
                    opacity=sc.opacity, clim=clim,
                    show_scalar_bar=sc.show_colorbar,
                    scalar_bar_args={"title": sc.field},
                )
                self._slice_actors[key] = actor

    # ------------------------------------------------------------------
    # Arrow glyphs
    # ------------------------------------------------------------------

    def _update_arrows(self, state: dict):
        pv = _import_pyvista()

        for i, ac in enumerate(self._config.arrows):
            vel = np.asarray(state[ac.node][ac.field], dtype=np.float32)
            axis, idx = self._slice_index(ac.normal, ac.origin_frac)
            nx, ny, nz = self._grid_dims

            # Extract 2D velocity slice and build coordinate arrays
            if axis == 0:
                vel_slice = vel[idx, :, :, :]
                a = np.arange(ny, dtype=np.float32)
                b = np.arange(nz, dtype=np.float32)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                cc = np.full_like(aa, float(idx))
                coords = (cc, aa, bb)
            elif axis == 1:
                vel_slice = vel[:, idx, :, :]
                a = np.arange(nx, dtype=np.float32)
                b = np.arange(nz, dtype=np.float32)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                cc = np.full_like(aa, float(idx))
                coords = (aa, cc, bb)
            else:
                vel_slice = vel[:, :, idx, :]
                a = np.arange(nx, dtype=np.float32)
                b = np.arange(ny, dtype=np.float32)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                cc = np.full_like(aa, float(idx))
                coords = (aa, bb, cc)

            # Sub-sample
            s = ac.stride
            cx = coords[0][::s, ::s]
            cy = coords[1][::s, ::s]
            cz = coords[2][::s, ::s]
            vel_sub = vel_slice[::s, ::s, :]

            points = np.column_stack([
                cx.ravel(), cy.ravel(), cz.ravel(),
            ])
            vectors = vel_sub.reshape(-1, 3)
            mag = np.linalg.norm(vectors, axis=1)

            # Filter out zero/near-zero vectors (wall cells)
            mask = mag > 1e-8
            if not np.any(mask):
                # Remove old actor if all zero
                key = f"arrows_{i}"
                if key in self._dynamic_actors:
                    self._plotter.remove_actor(self._dynamic_actors.pop(key))
                continue

            points = points[mask]
            vectors = vectors[mask]
            mag = mag[mask]

            pd = pv.PolyData(points)
            pd["vectors"] = vectors
            pd["magnitude"] = mag

            arrows = pd.glyph(
                orient="vectors", scale="magnitude",
                factor=ac.scale,
            )

            key = f"arrows_{i}"
            if key in self._dynamic_actors:
                self._plotter.remove_actor(self._dynamic_actors[key])

            clim = ac.clim
            if clim is None:
                clim = (0.0, float(mag.max()) if mag.max() > 0 else 1.0)

            actor = self._plotter.add_mesh(
                arrows, scalars="magnitude", cmap=ac.cmap,
                clim=clim, show_scalar_bar=False,
            )
            self._dynamic_actors[key] = actor

    # ------------------------------------------------------------------
    # Streamlines
    # ------------------------------------------------------------------

    def _update_streamlines(self, state: dict):
        pv = _import_pyvista()

        for i, sc in enumerate(self._config.streamlines):
            vel = np.asarray(state[sc.node][sc.field], dtype=np.float32)
            nx, ny, nz = self._grid_dims

            # Build ImageData volume with velocity vectors
            grid = pv.ImageData(dimensions=(nx, ny, nz))
            vel_vtk = np.stack(
                [vel[:, :, :, c].ravel(order="F") for c in range(3)],
                axis=1,
            )
            grid.point_data["velocity"] = vel_vtk

            # Seed disc for streamline origins
            cy, cz = (ny - 1) / 2.0, (nz - 1) / 2.0
            source_x = sc.source_x_frac * (nx - 1)
            source_radius = sc.source_radius_frac * min(ny, nz) / 2.0

            source = pv.Disc(
                center=(source_x, cy, cz),
                inner=0.0,
                outer=source_radius,
                normal=(1, 0, 0),
                r_res=max(1, int(sc.n_lines ** 0.5)),
                c_res=max(4, sc.n_lines),
            )

            try:
                streamlines = grid.streamlines_from_source(
                    source,
                    vectors="velocity",
                    max_length=sc.max_length,
                    max_steps=2000,
                    integration_direction="forward",
                )
            except Exception:
                continue

            if streamlines.n_points == 0:
                continue

            # Color by speed
            if "velocity" in streamlines.point_data:
                v = streamlines.point_data["velocity"]
                streamlines.point_data["speed"] = np.linalg.norm(v, axis=1)
                scalars = "speed"
            else:
                scalars = None

            key = f"streamlines_{i}"
            if key in self._dynamic_actors:
                self._plotter.remove_actor(self._dynamic_actors[key])

            tubes = streamlines.tube(radius=sc.tube_radius)
            actor = self._plotter.add_mesh(
                tubes, scalars=scalars, cmap=sc.cmap,
                show_scalar_bar=False,
            )
            self._dynamic_actors[key] = actor

    # ------------------------------------------------------------------
    # Time text overlay
    # ------------------------------------------------------------------

    def _update_time_text(self, sim_time: float):
        if not self._config.show_time:
            return
        if self._time_actor is not None:
            self._plotter.remove_actor(self._time_actor)
        self._time_actor = self._plotter.add_text(
            f"t = {sim_time:.4f} s",
            position="upper_left",
            font_size=12,
            color="black",
        )

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def _encode(self, img: np.ndarray) -> bytes:
        """Encode RGB numpy array to compressed image bytes."""
        if not _HAS_PIL:
            raise ImportError(
                "Pillow is required for image encoding. "
                "Install with: pip install Pillow"
            )
        pil_img = _PILImage.fromarray(img)
        buf = io.BytesIO()
        if self._fmt in ("jpeg", "jpg"):
            pil_img.save(buf, format="JPEG", quality=self._quality)
        elif self._fmt == "webp":
            pil_img.save(buf, format="WEBP", quality=self._quality)
        else:
            pil_img.save(buf, format="PNG")
        return buf.getvalue()

    # ------------------------------------------------------------------
    # Public API (ServerFrameRendererBase)
    # ------------------------------------------------------------------

    def render(self, sim_time: float, state: dict) -> bytes:
        """Render current 3D state to compressed image bytes.

        On the first call, grid dimensions are discovered from the state
        and static elements (pipe wall, axes) are added.  Subsequent
        calls update dynamic data and re-render.

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
        self._ensure_plotter()

        if not self._initialized:
            self._discover_grid(state)
            self._build_static()
            self._initialized = True

        self._update_slices(state)
        self._update_arrows(state)
        self._update_streamlines(state)
        self._update_time_text(sim_time)

        img = self._plotter.screenshot(return_img=True)
        return self._encode(img)

    def reset(self) -> None:
        """No accumulated state to clear for 3D renderer."""

    def set_format(self, fmt: str, quality: Optional[int] = None) -> None:
        """Change the output image format and/or quality."""
        self._fmt = fmt
        if quality is not None:
            self._quality = quality

    def close(self) -> None:
        """Release VTK/PyVista resources."""
        if self._plotter is not None:
            self._plotter.close()
            self._plotter = None
            self._initialized = False
            self._grid_dims = None
            self._slice_meshes.clear()
            self._slice_actors.clear()
            self._dynamic_actors.clear()
            self._time_actor = None
