"""
Post-simulation 3D history viewer.

Generic replay viewer for any simulation that produces a time-indexed
history dict.  Domain-specific geometry (pipe walls, propellers, etc.)
is built by the caller and passed in via :meth:`add_static_mesh` or
:meth:`add_rotating_mesh`.

Usage::

    from maddening.viz.history_viewer import HistoryViewer3D

    final_state, history = gm.run_scan_with_history(500)

    viewer = HistoryViewer3D(history, dt=0.01)
    viewer.add_volume_slice("fluid", "velocity", component=-1,
                            normal="x", origin_frac=0.5, cmap="coolwarm")
    viewer.add_isosurface("fluid", "tracer", threshold=0.5,
                          color="#4488CC", opacity=0.6)
    viewer.show()
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ------------------------------------------------------------------
# Configuration dataclasses
# ------------------------------------------------------------------

@dataclass
class _SliceDef:
    node: str
    field_name: str
    component: int
    normal: str
    origin_frac: float
    cmap: str
    clim: tuple[float, float] | None
    opacity: float
    show_colorbar: bool


@dataclass
class _ArrowDef:
    node: str
    field_name: str
    normal: str
    origin_frac: float
    stride: int
    scale: float
    cmap: str
    clim: tuple[float, float] | None


@dataclass
class _StreamlineDef:
    node: str
    field_name: str
    dims: tuple[int, int, int]
    n_lines: int
    source_center: tuple[float, float, float]
    source_radius: float
    source_normal: tuple[float, float, float]
    max_length: float
    tube_radius: float
    cmap: str
    interval: int


@dataclass
class _IsosurfaceDef:
    node: str
    field_name: str
    threshold: float
    color: str
    opacity: float
    smooth_n_iter: int


@dataclass
class _StaticMeshDef:
    mesh: object
    color: str
    opacity: float
    smooth_shading: bool


@dataclass
class _RotatingMeshDef:
    mesh: object
    axis: str
    speed: float  # degrees per frame
    center: tuple[float, float, float]
    color: str
    opacity: float


@dataclass
class _ParticleDef:
    node: str
    field_name: str
    start_pos: tuple[float, float, float]
    radius: float
    color: str
    opacity: float
    clamp_y: tuple[float, float] | None
    clamp_z: tuple[float, float] | None
    periodic_x: float | None  # wrap-around period (e.g. nx), or None


# ------------------------------------------------------------------
# HistoryViewer3D
# ------------------------------------------------------------------

class HistoryViewer3D:
    """Interactive post-simulation 3D viewer.

    Parameters
    ----------
    history : dict
        Nested ``{node_name: {field: array}}`` where each leaf has a
        leading time axis of length *n_frames*.  This is the *history*
        output of ``GraphManager.run_scan_with_history()``.
    dt : float
        Simulation timestep (for time display).
    window_size : tuple of int
        ``(width, height)`` of the viewer window.
    background : str
        Background colour.
    title : str
        Window title.
    playback_fps : int
        Target frames per second during playback.
    autoplay : bool
        If True, start playback automatically when the viewer opens.
    """

    def __init__(
        self,
        history: dict,
        dt: float = 1.0,
        window_size: tuple[int, int] = (1400, 800),
        background: str = "#f0f0f0",
        title: str = "MADDENING -- History Viewer",
        playback_fps: int = 30,
        autoplay: bool = True,
    ):
        self._history = history
        self._dt = dt
        self._window_size = window_size
        self._background = background
        self._title = title
        self._playback_fps = playback_fps
        self._autoplay = autoplay
        self._n_frames = self._find_n_frames()

        # Element definitions (populated by add_* methods)
        self._slices: list[_SliceDef] = []
        self._arrows: list[_ArrowDef] = []
        self._streamlines: list[_StreamlineDef] = []
        self._isosurfaces: list[_IsosurfaceDef] = []
        self._static_meshes: list[_StaticMeshDef] = []
        self._rotating_meshes: list[_RotatingMeshDef] = []
        self._particles: list[_ParticleDef] = []

        # Runtime state (set during show())
        self._plotter = None
        self._frame_idx = 0
        self._playing = False
        self._time_slider = None

        self._slice_meshes: dict[str, object] = {}
        self._arrow_actors: dict[str, object] = {}
        self._stream_actors: dict[str, object] = {}
        self._iso_actors: dict[str, object] = {}
        self._iso_grids: dict[str, object] = {}
        self._rot_meshes: list[object] = []
        self._rot_angles: list[float] = []
        self._particle_meshes: list[object] = []
        self._particle_trajectories: list[np.ndarray] = []
        self._last_stream_frame: int = -999
        self._playback_speed: int = 1  # frames to advance per tick

    def _find_n_frames(self) -> int:
        for node_state in self._history.values():
            for arr in node_state.values():
                a = np.asarray(arr)
                if a.ndim >= 1:
                    return a.shape[0]
        raise ValueError("History dict is empty or has no array values")

    # ------------------------------------------------------------------
    # Public builder API
    # ------------------------------------------------------------------

    def add_volume_slice(
        self,
        node: str,
        field: str,
        component: int = -1,
        normal: str = "x",
        origin_frac: float = 0.5,
        cmap: str = "coolwarm",
        clim: tuple[float, float] | None = None,
        opacity: float = 0.92,
        show_colorbar: bool = True,
    ):
        """Add a cross-section slice coloured by a scalar or vector field."""
        self._slices.append(_SliceDef(
            node, field, component, normal, origin_frac,
            cmap, clim, opacity, show_colorbar,
        ))

    def add_arrows(
        self,
        node: str,
        field: str = "velocity",
        normal: str = "x",
        origin_frac: float = 0.5,
        stride: int = 2,
        scale: float = 200.0,
        cmap: str = "coolwarm",
        clim: tuple[float, float] | None = None,
    ):
        """Add velocity arrow glyphs on a cross-section."""
        self._arrows.append(_ArrowDef(
            node, field, normal, origin_frac, stride, scale, cmap, clim,
        ))

    def add_streamlines(
        self,
        node: str,
        field: str = "velocity",
        dims: tuple[int, int, int] = (48, 24, 24),
        n_lines: int = 20,
        source_center: tuple[float, float, float] = (4.0, 12.0, 12.0),
        source_radius: float = 5.0,
        source_normal: tuple[float, float, float] = (1, 0, 0),
        max_length: float = 300.0,
        tube_radius: float = 0.06,
        cmap: str = "coolwarm",
        interval: int = 15,
    ):
        """Add streamlines (recomputed every *interval* frames during playback)."""
        self._streamlines.append(_StreamlineDef(
            node, field, dims, n_lines, source_center, source_radius,
            source_normal, max_length, tube_radius, cmap, interval,
        ))

    def add_isosurface(
        self,
        node: str,
        field: str,
        threshold: float = 0.5,
        color: str = "#4488CC",
        opacity: float = 0.6,
        smooth_n_iter: int = 30,
    ):
        """Add an isosurface of a scalar field (e.g. liquid surface)."""
        self._isosurfaces.append(_IsosurfaceDef(
            node, field, threshold, color, opacity, smooth_n_iter,
        ))

    def add_static_mesh(
        self,
        mesh,
        color: str = "#BBBBBB",
        opacity: float = 0.1,
        smooth_shading: bool = True,
    ):
        """Add a static mesh (pipe wall, bounding box, etc.)."""
        self._static_meshes.append(_StaticMeshDef(
            mesh, color, opacity, smooth_shading,
        ))

    def add_rotating_mesh(
        self,
        mesh,
        axis: str = "x",
        speed: float = 5.0,
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        color: str = "#DD5533",
        opacity: float = 0.85,
    ):
        """Add a mesh that rotates each frame (propeller, rotor, gear, etc.).

        Parameters
        ----------
        axis : str
            Rotation axis (``"x"``, ``"y"``, or ``"z"``).
        speed : float
            Degrees of rotation per frame.
        center : tuple
            Rotation pivot point.
        """
        self._rotating_meshes.append(_RotatingMeshDef(
            mesh, axis, speed, center, color, opacity,
        ))

    def add_particle(
        self,
        node: str,
        field: str = "velocity",
        start_pos: tuple[float, float, float] = (10.0, 12.0, 12.0),
        radius: float = 0.5,
        color: str = "#22AA44",
        opacity: float = 1.0,
        clamp_y: tuple[float, float] | None = None,
        clamp_z: tuple[float, float] | None = None,
        periodic_x: float | None = None,
    ):
        """Add a particle advected by a vector field.

        Parameters
        ----------
        clamp_y, clamp_z : tuple of float, optional
            ``(min, max)`` bounds to keep the particle inside a domain.
        periodic_x : float, optional
            If set, the x-coordinate wraps at this period.
        """
        self._particles.append(_ParticleDef(
            node, field, start_pos, radius, color, opacity,
            clamp_y, clamp_z, periodic_x,
        ))

    # ------------------------------------------------------------------
    # Internal helpers: field extraction
    # ------------------------------------------------------------------

    def _get_field(self, node: str, field: str, frame: int) -> np.ndarray:
        return np.asarray(self._history[node][field][frame])

    def _extract_scalar(self, node, field, component, frame):
        arr = self._get_field(node, field, frame)
        if arr.ndim == 4:  # vector (nx, ny, nz, 3)
            if component == -1:
                return np.linalg.norm(arr, axis=-1).astype(np.float32)
            return arr[..., component].astype(np.float32)
        return arr.astype(np.float32)

    def _slice_index(self, normal, origin_frac, dims):
        axis = {"x": 0, "y": 1, "z": 2}[normal]
        dim = dims[axis]
        return axis, max(0, min(int(origin_frac * (dim - 1)), dim - 1))

    def _slice_data(self, scalars, axis, idx):
        if axis == 0:
            return scalars[idx, :, :]
        elif axis == 1:
            return scalars[:, idx, :]
        return scalars[:, :, idx]

    # ------------------------------------------------------------------
    # Internal helpers: mesh creation
    # ------------------------------------------------------------------

    def _make_slice_grid(self, axis, idx, dims):
        import pyvista as pv
        nx, ny, nz = dims
        if axis == 0:
            a, b = np.arange(ny, dtype=np.float32), np.arange(nz, dtype=np.float32)
            aa, bb = np.meshgrid(a, b, indexing="ij")
            return pv.StructuredGrid(np.full_like(aa, float(idx)), aa, bb)
        elif axis == 1:
            a, b = np.arange(nx, dtype=np.float32), np.arange(nz, dtype=np.float32)
            aa, bb = np.meshgrid(a, b, indexing="ij")
            return pv.StructuredGrid(aa, np.full_like(aa, float(idx)), bb)
        else:
            a, b = np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32)
            aa, bb = np.meshgrid(a, b, indexing="ij")
            return pv.StructuredGrid(aa, bb, np.full_like(aa, float(idx)))

    def _compute_streamline_tubes(self, sdef: _StreamlineDef, frame: int):
        import pyvista as pv
        vel = self._get_field(sdef.node, sdef.field_name, frame).astype(np.float32)
        nx, ny, nz = sdef.dims

        grid = pv.ImageData(dimensions=(nx, ny, nz))
        vel_vtk = np.stack(
            [vel[:, :, :, c].ravel(order="F") for c in range(3)], axis=1,
        )
        grid.point_data["velocity"] = vel_vtk

        source = pv.Disc(
            center=sdef.source_center, inner=0.0, outer=sdef.source_radius,
            normal=sdef.source_normal,
            r_res=max(1, int(sdef.n_lines ** 0.5)),
            c_res=max(4, sdef.n_lines),
        )
        try:
            sl = grid.streamlines_from_source(
                source, vectors="velocity",
                max_length=sdef.max_length, max_steps=2000,
                integration_direction="forward",
            )
        except Exception:
            return None
        if sl.n_points == 0:
            return None
        if "velocity" in sl.point_data:
            sl.point_data["speed"] = np.linalg.norm(
                sl.point_data["velocity"], axis=1,
            )
        return sl.tube(radius=sdef.tube_radius)

    def _compute_isosurface(self, idef: _IsosurfaceDef, frame: int):
        import pyvista as pv
        scalar = self._get_field(idef.node, idef.field_name, frame).astype(np.float32)
        dims = scalar.shape
        key_grid = f"iso_grid_{id(idef)}"
        if key_grid not in self._iso_grids:
            self._iso_grids[key_grid] = pv.ImageData(dimensions=dims)
        grid = self._iso_grids[key_grid]
        grid.point_data["s"] = scalar.ravel(order="F")
        try:
            iso = grid.contour([idef.threshold], scalars="s")
        except Exception:
            return None
        if iso.n_points == 0:
            return None
        if idef.smooth_n_iter > 0:
            iso = iso.smooth(n_iter=idef.smooth_n_iter)
        return iso

    # ------------------------------------------------------------------
    # Particle trajectory pre-computation
    # ------------------------------------------------------------------

    def _interp_velocity(self, vel: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Trilinear interpolation of velocity at a continuous position."""
        nx, ny, nz = vel.shape[:3]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        x = np.clip(x, 0.0, nx - 1.001)
        y = np.clip(y, 0.0, ny - 1.001)
        z = np.clip(z, 0.0, nz - 1.001)
        ix, iy, iz = int(x), int(y), int(z)
        fx, fy, fz = x - ix, y - iy, z - iz
        ix1 = min(ix + 1, nx - 1)
        iy1 = min(iy + 1, ny - 1)
        iz1 = min(iz + 1, nz - 1)
        c000 = vel[ix, iy, iz]
        c100 = vel[ix1, iy, iz]
        c010 = vel[ix, iy1, iz]
        c110 = vel[ix1, iy1, iz]
        c001 = vel[ix, iy, iz1]
        c101 = vel[ix1, iy, iz1]
        c011 = vel[ix, iy1, iz1]
        c111 = vel[ix1, iy1, iz1]
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz

    def _precompute_particle_trajectory(self, pdef: _ParticleDef) -> np.ndarray:
        """Advect a particle through all frames using the velocity history."""
        trajectory = np.zeros((self._n_frames, 3), dtype=np.float64)
        pos = np.array(pdef.start_pos, dtype=np.float64)
        trajectory[0] = pos
        for f in range(self._n_frames - 1):
            vel = self._get_field(pdef.node, pdef.field_name, f).astype(np.float64)
            v = self._interp_velocity(vel, pos)
            pos = pos + v
            if pdef.periodic_x is not None:
                pos[0] = pos[0] % pdef.periodic_x
            if pdef.clamp_y is not None:
                pos[1] = np.clip(pos[1], pdef.clamp_y[0], pdef.clamp_y[1])
            if pdef.clamp_z is not None:
                pos[2] = np.clip(pos[2], pdef.clamp_z[0], pdef.clamp_z[1])
            trajectory[f + 1] = pos
        return trajectory

    # ------------------------------------------------------------------
    # Build scene (called once)
    # ------------------------------------------------------------------

    def _build_scene(self):
        import pyvista as pv
        p = self._plotter

        # --- static meshes ---
        for mdef in self._static_meshes:
            p.add_mesh(mdef.mesh, color=mdef.color, opacity=mdef.opacity,
                       smooth_shading=mdef.smooth_shading)

        # --- rotating meshes ---
        for rdef in self._rotating_meshes:
            p.add_mesh(rdef.mesh, color=rdef.color, opacity=rdef.opacity,
                       smooth_shading=True)
            self._rot_meshes.append(rdef.mesh)
            self._rot_angles.append(0.0)

        # --- slices (created with frame-0 data) ---
        for i, sdef in enumerate(self._slices):
            scalars = self._extract_scalar(
                sdef.node, sdef.field_name, sdef.component, 0,
            )
            dims = scalars.shape
            axis, idx = self._slice_index(sdef.normal, sdef.origin_frac, dims)
            data = self._slice_data(scalars, axis, idx)
            mesh = self._make_slice_grid(axis, idx, dims)
            mesh.point_data["s"] = data.ravel(order="F")
            self._slice_meshes[f"slice_{i}"] = mesh
            clim = sdef.clim
            if clim is None:
                clim = self._global_clim(sdef)
            p.add_mesh(
                mesh, scalars="s", cmap=sdef.cmap, clim=clim,
                opacity=sdef.opacity,
                show_scalar_bar=sdef.show_colorbar,
                scalar_bar_args={"title": sdef.field_name} if sdef.show_colorbar else {},
            )

        # --- particles (pre-compute trajectories) ---
        for pdef in self._particles:
            print(f"  Pre-computing particle trajectory from "
                  f"({pdef.start_pos[0]:.1f}, {pdef.start_pos[1]:.1f}, "
                  f"{pdef.start_pos[2]:.1f})...")
            traj = self._precompute_particle_trajectory(pdef)
            self._particle_trajectories.append(traj)
            sphere = pv.Sphere(radius=pdef.radius, center=traj[0])
            p.add_mesh(sphere, color=pdef.color, opacity=pdef.opacity,
                       smooth_shading=True)
            self._particle_meshes.append(sphere)

        p.add_axes()
        p.reset_camera()

    def _global_clim(self, sdef: _SliceDef):
        """Compute colour limits across sampled frames for stable colouring."""
        vmin, vmax = np.inf, -np.inf
        sample = np.linspace(0, self._n_frames - 1, min(20, self._n_frames), dtype=int)
        for f in sample:
            scalars = self._extract_scalar(
                sdef.node, sdef.field_name, sdef.component, int(f),
            )
            vmin = min(vmin, float(scalars.min()))
            vmax = max(vmax, float(scalars.max()))
        if vmax - vmin < 1e-10:
            vmax = vmin + 1.0
        return (vmin, vmax)

    # ------------------------------------------------------------------
    # Update frame
    # ------------------------------------------------------------------

    def _update_frame(self, frame: int, light: bool = False):
        """Update visuals to the given frame.

        Parameters
        ----------
        light : bool
            If True, skip expensive operations (streamlines, arrows) for
            smooth playback.  Slices, isosurfaces, particles, and rotating
            meshes always update.
        """
        frame = max(0, min(frame, self._n_frames - 1))
        self._frame_idx = frame

        # --- slices (cheap: data copy) ---
        for i, sdef in enumerate(self._slices):
            scalars = self._extract_scalar(
                sdef.node, sdef.field_name, sdef.component, frame,
            )
            dims = scalars.shape
            axis, idx = self._slice_index(sdef.normal, sdef.origin_frac, dims)
            data = self._slice_data(scalars, axis, idx)
            self._slice_meshes[f"slice_{i}"].point_data["s"] = data.ravel(order="F")

        # --- isosurfaces (moderate cost, but essential for liquid viz) ---
        self._update_isosurfaces(frame)

        # --- rotating meshes ---
        for j, rdef in enumerate(self._rotating_meshes):
            target_angle = frame * rdef.speed
            delta = target_angle - self._rot_angles[j]
            if abs(delta) > 0.01:
                rotate_fn = {
                    "x": self._rot_meshes[j].rotate_x,
                    "y": self._rot_meshes[j].rotate_y,
                    "z": self._rot_meshes[j].rotate_z,
                }[rdef.axis]
                rotate_fn(delta, point=rdef.center, inplace=True)
                self._rot_angles[j] = target_angle

        # --- particles ---
        for k, traj in enumerate(self._particle_trajectories):
            new_pos = traj[frame]
            old_center = np.array(self._particle_meshes[k].center)
            self._particle_meshes[k].translate(new_pos - old_center, inplace=True)

        # --- arrows (skip during playback) ---
        if not light:
            self._update_arrows(frame)

        # --- streamlines (skip during playback, or update at intervals) ---
        if not light:
            self._update_streamlines(frame)
        elif self._streamlines:
            interval = self._streamlines[0].interval
            if abs(frame - self._last_stream_frame) >= interval:
                self._update_streamlines(frame)

        # --- status text ---
        t = frame * self._dt
        if self._playing:
            tag = f"  [playing x{self._playback_speed}]"
        else:
            tag = "  [paused]"
        self._plotter.add_text(
            f"t = {t:.3f} s    frame {frame}/{self._n_frames - 1}{tag}",
            position="upper_left", font_size=10, color="black",
            name="status",
        )

        # Force PyVista to redraw (modifying mesh data alone doesn't trigger it)
        if self._plotter is not None:
            self._plotter.render()

    def _update_isosurfaces(self, frame):
        for i, idef in enumerate(self._isosurfaces):
            key = f"iso_{i}"
            if key in self._iso_actors:
                self._plotter.remove_actor(self._iso_actors.pop(key))
            iso = self._compute_isosurface(idef, frame)
            if iso is not None:
                self._iso_actors[key] = self._plotter.add_mesh(
                    iso, color=idef.color, opacity=idef.opacity,
                    smooth_shading=True,
                )

    def _update_arrows(self, frame):
        import pyvista as pv
        for i, adef in enumerate(self._arrows):
            key = f"arrows_{i}"
            if key in self._arrow_actors:
                self._plotter.remove_actor(self._arrow_actors.pop(key))
            vel = self._get_field(adef.node, adef.field_name, frame).astype(np.float32)
            dims = vel.shape[:3]
            axis, idx = self._slice_index(adef.normal, adef.origin_frac, dims)
            if axis == 0:
                vel_slice = vel[idx, :, :, :]
                d1, d2 = dims[1], dims[2]
                a, b = np.arange(d1, dtype=np.float32), np.arange(d2, dtype=np.float32)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                coords = (np.full_like(aa, float(idx)), aa, bb)
            elif axis == 1:
                vel_slice = vel[:, idx, :, :]
                d1, d2 = dims[0], dims[2]
                a, b = np.arange(d1, dtype=np.float32), np.arange(d2, dtype=np.float32)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                coords = (aa, np.full_like(aa, float(idx)), bb)
            else:
                vel_slice = vel[:, :, idx, :]
                d1, d2 = dims[0], dims[1]
                a, b = np.arange(d1, dtype=np.float32), np.arange(d2, dtype=np.float32)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                coords = (aa, bb, np.full_like(aa, float(idx)))
            s = adef.stride
            pts = np.column_stack([c[::s, ::s].ravel() for c in coords])
            vecs = vel_slice[::s, ::s, :].reshape(-1, 3)
            mag = np.linalg.norm(vecs, axis=1)
            mask = mag > 1e-8
            if np.any(mask):
                pd = pv.PolyData(pts[mask])
                pd["vectors"] = vecs[mask]
                pd["magnitude"] = mag[mask]
                glyphs = pd.glyph(orient="vectors", scale="magnitude",
                                  factor=adef.scale)
                clim = adef.clim or (0.0, max(float(mag.max()), 1e-6))
                self._arrow_actors[key] = self._plotter.add_mesh(
                    glyphs, scalars="magnitude", cmap=adef.cmap,
                    clim=clim, show_scalar_bar=False,
                )

    def _update_streamlines(self, frame):
        for i, sdef in enumerate(self._streamlines):
            key = f"stream_{i}"
            if key in self._stream_actors:
                self._plotter.remove_actor(self._stream_actors.pop(key))
            tubes = self._compute_streamline_tubes(sdef, frame)
            if tubes is not None:
                scalars = "speed" if "speed" in tubes.point_data else None
                self._stream_actors[key] = self._plotter.add_mesh(
                    tubes, scalars=scalars, cmap=sdef.cmap,
                    show_scalar_bar=False, opacity=0.7,
                )
        self._last_stream_frame = frame

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def _on_time_slider(self, value):
        frame = max(0, min(self._n_frames - 1, int(round(value))))
        if frame != self._frame_idx:
            self._update_frame(frame, light=False)

    def _play_tick(self, step):
        if not self._playing:
            return
        nxt = self._frame_idx + self._playback_speed
        if nxt >= self._n_frames:
            self._playing = False
            self._update_frame(self._n_frames - 1, light=False)
            return
        self._update_frame(nxt, light=True)
        if self._time_slider is not None:
            self._time_slider.GetRepresentation().SetValue(float(nxt))

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing:
            if self._frame_idx >= self._n_frames - 1:
                self._update_frame(0, light=False)
                if self._time_slider is not None:
                    self._time_slider.GetRepresentation().SetValue(0.0)
            print("Playing...")
        else:
            self._update_frame(self._frame_idx, light=False)
            print("Paused.")

    def _speed_up(self):
        self._playback_speed = min(self._playback_speed * 2, 64)
        print(f"Speed: x{self._playback_speed}")
        # Refresh status text
        self._update_frame(self._frame_idx, light=True)

    def _speed_down(self):
        self._playback_speed = max(self._playback_speed // 2, 1)
        print(f"Speed: x{self._playback_speed}")
        self._update_frame(self._frame_idx, light=True)

    def _step_forward(self):
        self._playing = False
        nxt = min(self._frame_idx + 1, self._n_frames - 1)
        self._update_frame(nxt, light=False)
        if self._time_slider is not None:
            self._time_slider.GetRepresentation().SetValue(float(nxt))

    def _step_backward(self):
        self._playing = False
        prev = max(self._frame_idx - 1, 0)
        self._update_frame(prev, light=False)
        if self._time_slider is not None:
            self._time_slider.GetRepresentation().SetValue(float(prev))

    def _go_to_start(self):
        self._playing = False
        self._update_frame(0, light=False)
        if self._time_slider is not None:
            self._time_slider.GetRepresentation().SetValue(0.0)

    def _go_to_end(self):
        self._playing = False
        self._update_frame(self._n_frames - 1, light=False)
        if self._time_slider is not None:
            self._time_slider.GetRepresentation().SetValue(
                float(self._n_frames - 1)
            )

    # ------------------------------------------------------------------
    # Show
    # ------------------------------------------------------------------

    def show(self):
        """Open the interactive viewer window."""
        import pyvista as pv

        self._plotter = pv.Plotter(
            window_size=list(self._window_size),
            title=self._title,
        )
        self._plotter.set_background(self._background)

        self._build_scene()
        self._update_frame(0, light=False)

        # --- time slider ---
        self._time_slider = self._plotter.add_slider_widget(
            self._on_time_slider,
            rng=[0, self._n_frames - 1],
            value=0,
            title="Frame",
            pointa=(0.25, 0.92), pointb=(0.75, 0.92),
            style="modern",
        )

        # --- keyboard ---
        self._plotter.add_key_event("space", self._toggle_play)
        self._plotter.add_key_event("Right", self._step_forward)
        self._plotter.add_key_event("Left", self._step_backward)
        self._plotter.add_key_event("Home", self._go_to_start)
        self._plotter.add_key_event("End", self._go_to_end)
        self._plotter.add_key_event("Up", self._speed_up)
        self._plotter.add_key_event("Down", self._speed_down)

        # --- playback timer ---
        interval = max(16, int(1000.0 / self._playback_fps))
        self._plotter.add_timer_event(
            max_steps=1_000_000,
            duration=interval,
            callback=self._play_tick,
        )

        print("\nControls:")
        print("  Space       play / pause")
        print("  Left/Right  step backward / forward")
        print("  Up/Down     speed up / slow down")
        print("  Home/End    jump to first / last frame")
        print("  Slider      scrub to any frame")
        print("  Mouse       rotate, zoom, pan")

        if self._autoplay:
            self._playing = True
            print("Playing...")

        self._plotter.show()
