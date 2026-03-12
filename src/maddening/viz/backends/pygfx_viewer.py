"""
GPU-accelerated 3D history viewer using pygfx/wgpu.

Drop-in replacement for :class:`~maddening.viz.history_viewer.HistoryViewer3D`
with game-engine-like rendering patterns:

* Continuous 60 fps render loop (no explicit ``render()`` needed)
* GPU-resident geometry buffers updated in place
* Rotation/translation via transforms (not vertex mutation)
* Instanced rendering for repeated geometry

Same builder API — swap the import and demo scripts work unchanged::

    from maddening.viz.backends.pygfx_viewer import GPUHistoryViewer

    viewer = GPUHistoryViewer(history, dt=0.01)
    viewer.add_isosurface("fluid", "tracer", threshold=0.5)
    viewer.show()

Requirements::

    pip install maddening[gpu-viz]   # pygfx + scikit-image
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Re-use the configuration dataclasses from the PyVista viewer.
# They are pure Python — no PyVista import triggered.
from maddening.viz.history_viewer import (
    _ArrowDef,
    _IsosurfaceDef,
    _ParticleDef,
    _RotatingMeshDef,
    _SliceDef,
    _StaticMeshDef,
    _StreamlineDef,
)


def _hex_to_rgba(hex_color: str, opacity: float = 1.0):
    """Convert ``"#RRGGBB"`` to ``(r, g, b, a)`` floats in [0, 1]."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16) / 255, int(h[2:4], 16) / 255, int(h[4:6], 16) / 255
    return (r, g, b, opacity)


def _check_pygfx():
    try:
        import pygfx  # noqa: F401
        import rendercanvas  # noqa: F401
    except ImportError:
        raise ImportError(
            "GPUHistoryViewer requires 'pygfx', 'rendercanvas', and 'glfw'. "
            "Install them with:  pip install maddening[gpu-viz]"
        )


# ------------------------------------------------------------------
# Mesh conversion utilities
# ------------------------------------------------------------------

def _pyvista_mesh_to_arrays(mesh):
    """Extract (positions, indices, normals) from a PyVista-like mesh.

    Returns ``(positions_f32, indices_i4, normals_f32)``.
    """
    tri = mesh.triangulate()
    positions = np.ascontiguousarray(tri.points, dtype=np.float32)

    # Parse PyVista face array: [n, i0, i1, i2, n, i0, ...] → (N, 3)
    faces_raw = np.asarray(tri.faces)
    if hasattr(tri, "regular_faces"):
        indices = np.ascontiguousarray(tri.regular_faces, dtype=np.int32)
    else:
        n_cells = tri.n_cells
        indices = faces_raw.reshape(n_cells, 4)[:, 1:4]
        indices = np.ascontiguousarray(indices, dtype=np.int32)

    # Normals
    normals = tri.point_normals
    if normals is None:
        tri.compute_normals(inplace=True)
        normals = tri.point_normals
    if normals is None:
        # Fallback: flat zero normals (pygfx can auto-compute)
        normals = np.zeros_like(positions)
    normals = np.ascontiguousarray(normals, dtype=np.float32)

    return positions, indices, normals


def _make_sphere(center, radius, resolution=16):
    """Create a UV-sphere as (positions, indices, normals) arrays."""
    lat_steps = resolution
    lon_steps = resolution * 2
    verts = []
    norms = []
    for i in range(lat_steps + 1):
        theta = math.pi * i / lat_steps
        for j in range(lon_steps + 1):
            phi = 2 * math.pi * j / lon_steps
            x = radius * math.sin(theta) * math.cos(phi)
            y = radius * math.sin(theta) * math.sin(phi)
            z = radius * math.cos(theta)
            verts.append((center[0] + x, center[1] + y, center[2] + z))
            norms.append((
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ))
    faces = []
    for i in range(lat_steps):
        for j in range(lon_steps):
            a = i * (lon_steps + 1) + j
            b = a + lon_steps + 1
            faces.append((a, b, a + 1))
            faces.append((a + 1, b, b + 1))
    return (
        np.array(verts, dtype=np.float32),
        np.array(faces, dtype=np.int32),
        np.array(norms, dtype=np.float32),
    )


# ------------------------------------------------------------------
# GPUHistoryViewer
# ------------------------------------------------------------------

class GPUHistoryViewer:
    """GPU-accelerated post-simulation 3D viewer.

    Same builder API as
    :class:`~maddening.viz.history_viewer.HistoryViewer3D` but renders
    via **pygfx** on **wgpu** (WebGPU) for game-engine-like performance.

    Parameters
    ----------
    history : dict
        ``{node_name: {field: array_with_leading_time_axis}}``.
    dt : float
        Simulation timestep.
    window_size : tuple of int
        ``(width, height)``.
    background : str
        Background colour (hex string).
    title : str
        Window title.
    playback_fps : int
        Target playback rate.
    autoplay : bool
        Start playing automatically.
    camera_up : tuple of float or None
        World-space "up" direction for the camera orbit controller.
        ``(0, 0, 1)`` = z-up (useful when gravity is along -z).
        ``None`` = pygfx default (y-up).
    """

    def __init__(
        self,
        history: dict,
        dt: float = 1.0,
        window_size: tuple[int, int] = (1400, 800),
        background: str = "#f0f0f0",
        title: str = "MADDENING -- GPU History Viewer",
        playback_fps: int = 30,
        autoplay: bool = True,
        camera_up: tuple[float, float, float] | None = None,
    ):
        _check_pygfx()
        self._history = history
        self._dt = dt
        self._window_size = window_size
        self._background = background
        self._title = title
        self._playback_fps = playback_fps
        self._autoplay = autoplay
        self._camera_up = camera_up
        self._n_frames = self._find_n_frames()

        # Element definitions
        self._slices: list[_SliceDef] = []
        self._arrows: list[_ArrowDef] = []
        self._streamlines: list[_StreamlineDef] = []
        self._isosurfaces: list[_IsosurfaceDef] = []
        self._static_meshes: list[_StaticMeshDef] = []
        self._rotating_meshes: list[_RotatingMeshDef] = []
        self._particles: list[_ParticleDef] = []

        # Runtime state (set during show())
        self._scene = None
        self._frame_idx = 0
        self._playing = False
        self._playback_speed = 1

        # Per-element runtime objects
        self._iso_world_objects: dict[str, object] = {}
        self._slice_world_objects: dict[str, object] = {}
        self._rot_world_objects: list[object] = []
        self._rot_base_meshes: list[object] = []  # original pyvista meshes
        self._particle_world_objects: list[object] = []
        self._particle_trajectories: list[np.ndarray] = []
        self._status_text = None

    def _find_n_frames(self) -> int:
        for node_state in self._history.values():
            for arr in node_state.values():
                a = np.asarray(arr)
                if a.ndim >= 1:
                    return a.shape[0]
        raise ValueError("History dict is empty or has no array values")

    # ------------------------------------------------------------------
    # Public builder API (mirrors HistoryViewer3D exactly)
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
        self._particles.append(_ParticleDef(
            node, field, start_pos, radius, color, opacity,
            clamp_y, clamp_z, periodic_x,
        ))

    # ------------------------------------------------------------------
    # Field extraction helpers
    # ------------------------------------------------------------------

    def _get_field(self, node: str, field: str, frame: int) -> np.ndarray:
        return np.asarray(self._history[node][field][frame])

    def _extract_scalar(self, node, field, component, frame):
        arr = self._get_field(node, field, frame)
        if arr.ndim == 4:
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
    # Particle trajectory (same as HistoryViewer3D)
    # ------------------------------------------------------------------

    def _interp_velocity(self, vel: np.ndarray, pos: np.ndarray) -> np.ndarray:
        nx, ny, nz = vel.shape[:3]
        x = np.clip(float(pos[0]), 0.0, nx - 1.001)
        y = np.clip(float(pos[1]), 0.0, ny - 1.001)
        z = np.clip(float(pos[2]), 0.0, nz - 1.001)
        ix, iy, iz = int(x), int(y), int(z)
        fx, fy, fz = x - ix, y - iy, z - iz
        ix1 = min(ix + 1, nx - 1)
        iy1 = min(iy + 1, ny - 1)
        iz1 = min(iz + 1, nz - 1)
        c000 = vel[ix, iy, iz]; c100 = vel[ix1, iy, iz]
        c010 = vel[ix, iy1, iz]; c110 = vel[ix1, iy1, iz]
        c001 = vel[ix, iy, iz1]; c101 = vel[ix1, iy, iz1]
        c011 = vel[ix, iy1, iz1]; c111 = vel[ix1, iy1, iz1]
        c00 = c000 * (1 - fx) + c100 * fx
        c01 = c001 * (1 - fx) + c101 * fx
        c10 = c010 * (1 - fx) + c110 * fx
        c11 = c011 * (1 - fx) + c111 * fx
        c0 = c00 * (1 - fy) + c10 * fy
        c1 = c01 * (1 - fy) + c11 * fy
        return c0 * (1 - fz) + c1 * fz

    def _precompute_particle_trajectory(self, pdef: _ParticleDef) -> np.ndarray:
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
    # Isosurface extraction (CPU marching cubes, vectorised smoothing)
    # ------------------------------------------------------------------

    @staticmethod
    def _smooth_mesh(verts, faces, n_iter, relax=0.1):
        """Laplacian smoothing via scipy sparse — vectorised, fast."""
        if n_iter <= 0 or len(verts) == 0:
            return verts
        from scipy.sparse import csr_matrix, diags, eye as speye
        n = len(verts)
        # Build sparse adjacency from faces
        edges = np.concatenate([
            faces[:, [0, 1]], faces[:, [1, 0]],
            faces[:, [1, 2]], faces[:, [2, 1]],
            faces[:, [0, 2]], faces[:, [2, 0]],
        ], axis=0)
        data = np.ones(len(edges), dtype=np.float32)
        A = csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n))
        # Normalise rows
        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1
        D_inv = diags(1.0 / deg)
        # Smoothing operator: S = (1-r)*I + r*D_inv*A
        S = (1 - relax) * speye(n, format="csr") + relax * (D_inv @ A)
        v = verts.astype(np.float64)
        for _ in range(n_iter):
            v = S @ v
        return v.astype(np.float32)

    @staticmethod
    def _compute_vertex_normals(verts, faces):
        """Vectorised vertex normal computation from faces."""
        normals = np.zeros_like(verts, dtype=np.float64)
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        face_n = np.cross(v1 - v0, v2 - v0)
        np.add.at(normals, faces[:, 0], face_n)
        np.add.at(normals, faces[:, 1], face_n)
        np.add.at(normals, faces[:, 2], face_n)
        mag = np.linalg.norm(normals, axis=1, keepdims=True)
        mag = np.maximum(mag, 1e-8)
        return (normals / mag).astype(np.float32)

    def _compute_isosurface_arrays(self, idef: _IsosurfaceDef, frame: int):
        """Return (positions, indices, normals) or None."""
        try:
            from skimage.measure import marching_cubes
        except ImportError:
            raise ImportError(
                "Isosurface extraction requires scikit-image. "
                "pip install scikit-image"
            )
        scalar = self._get_field(idef.node, idef.field_name, frame).astype(np.float32)
        if scalar.min() >= idef.threshold or scalar.max() <= idef.threshold:
            return None
        try:
            verts, faces, normals, _ = marching_cubes(
                scalar, level=idef.threshold,
            )
        except Exception:
            return None
        if len(verts) == 0:
            return None

        if idef.smooth_n_iter > 0:
            verts = self._smooth_mesh(verts, faces, idef.smooth_n_iter)
            normals = self._compute_vertex_normals(verts, faces)

        positions = np.ascontiguousarray(verts, dtype=np.float32)
        indices = np.ascontiguousarray(faces, dtype=np.int32)
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        return positions, indices, normals

    def _precompute_isosurfaces(self):
        """Pre-compute isosurface geometry for all frames and definitions."""
        if not self._isosurfaces:
            return
        self._iso_cache: list[list[tuple | None]] = []
        for i, idef in enumerate(self._isosurfaces):
            cache = []
            print(f"  Pre-computing isosurface '{idef.field_name}' "
                  f"({self._n_frames} frames)...", end="", flush=True)
            t0 = time.perf_counter()
            for f in range(self._n_frames):
                cache.append(self._compute_isosurface_arrays(idef, f))
            elapsed = time.perf_counter() - t0
            print(f" {elapsed:.1f}s")
            self._iso_cache.append(cache)

    def _precompute_slices(self):
        """Pre-compute volume slice colour arrays for all frames."""
        if not self._slices:
            return
        self._slice_color_cache: list[list[np.ndarray]] = []
        for i, sdef in enumerate(self._slices):
            clim = sdef.clim
            if clim is None:
                clim = self._global_clim(sdef)
            cache = []
            print(f"  Pre-computing slice '{sdef.field_name}' "
                  f"({self._n_frames} frames)...", end="", flush=True)
            t0 = time.perf_counter()
            for f in range(self._n_frames):
                scalars = self._extract_scalar(
                    sdef.node, sdef.field_name, sdef.component, f,
                )
                dims = scalars.shape
                axis, idx_val = self._slice_index(sdef.normal, sdef.origin_frac, dims)
                data = self._slice_data(scalars, axis, idx_val)
                colors = self._scalar_to_colors(data.ravel(), clim, sdef.cmap, sdef.opacity)
                cache.append(colors)
            elapsed = time.perf_counter() - t0
            print(f" {elapsed:.1f}s")
            self._slice_color_cache.append(cache)
        # Store clim for each slice
        self._slice_clims = []
        for sdef in self._slices:
            self._slice_clims.append(sdef.clim if sdef.clim else self._global_clim(sdef))

    # ------------------------------------------------------------------
    # pygfx mesh creation
    # ------------------------------------------------------------------

    def _create_gfx_mesh(self, positions, indices, normals, color, opacity,
                         side="front"):
        import pygfx as gfx
        geom = gfx.Geometry(
            positions=positions,
            normals=normals,
            indices=indices,
        )
        rgba = _hex_to_rgba(color, opacity)
        mat = gfx.MeshPhongMaterial(
            color=rgba[:3],
            opacity=rgba[3],
            side=side,
        )
        if opacity < 1.0:
            mat.transparent = True
        return gfx.Mesh(geom, mat)

    def _convert_external_mesh(self, mesh):
        """Convert a PyVista-like mesh to (positions, indices, normals)."""
        if hasattr(mesh, "points") and hasattr(mesh, "faces"):
            return _pyvista_mesh_to_arrays(mesh)
        if isinstance(mesh, tuple) and len(mesh) == 3:
            return mesh
        raise TypeError(
            f"Cannot convert {type(mesh).__name__} to GPU mesh. "
            "Pass a PyVista mesh or (positions, indices, normals) tuple."
        )

    # ------------------------------------------------------------------
    # Scene building
    # ------------------------------------------------------------------

    def _build_scene(self, scene):
        import pygfx as gfx

        # --- static meshes ---
        for mdef in self._static_meshes:
            pos, idx, nrm = self._convert_external_mesh(mdef.mesh)
            mesh = self._create_gfx_mesh(pos, idx, nrm, mdef.color, mdef.opacity)
            scene.add(mesh)

        # --- rotating meshes ---
        for rdef in self._rotating_meshes:
            pos, idx, nrm = self._convert_external_mesh(rdef.mesh)
            mesh = self._create_gfx_mesh(pos, idx, nrm, rdef.color, rdef.opacity)
            # Store center offset so we rotate around the pivot
            cx, cy, cz = rdef.center
            # pygfx rotations are around the object's local origin,
            # so we shift the mesh so the pivot is at (0,0,0), then
            # translate the world object to the pivot.
            pos_shifted = pos.copy()
            pos_shifted[:, 0] -= cx
            pos_shifted[:, 1] -= cy
            pos_shifted[:, 2] -= cz
            geom = gfx.Geometry(
                positions=np.ascontiguousarray(pos_shifted),
                normals=nrm,
                indices=idx,
            )
            rgba = _hex_to_rgba(rdef.color, rdef.opacity)
            mat = gfx.MeshPhongMaterial(color=rgba[:3], opacity=rgba[3])
            if rdef.opacity < 1.0:
                mat.transparent = True
            mesh = gfx.Mesh(geom, mat)
            mesh.local.position = (cx, cy, cz)
            scene.add(mesh)
            self._rot_world_objects.append(mesh)

        # --- isosurfaces (from pre-computed cache, frame 0) ---
        for i, idef in enumerate(self._isosurfaces):
            key = f"iso_{i}"
            result = self._iso_cache[i][0] if self._iso_cache else None
            if result is not None:
                pos, idx, nrm = result
                # Double-sided so the surface is visible from above and below
                mesh = self._create_gfx_mesh(
                    pos, idx, nrm, idef.color, idef.opacity, side="both",
                )
                scene.add(mesh)
                self._iso_world_objects[key] = mesh
            else:
                self._iso_world_objects[key] = None

        # --- volume slices (geometry once, colours from pre-computed cache) ---
        for i, sdef in enumerate(self._slices):
            key = f"slice_{i}"
            scalars = self._extract_scalar(sdef.node, sdef.field_name, sdef.component, 0)
            dims = scalars.shape
            axis, idx_val = self._slice_index(sdef.normal, sdef.origin_frac, dims)

            # Build a flat grid of points for the slice plane
            if axis == 0:
                d0, d1 = dims[1], dims[2]
                a = np.arange(d0, dtype=np.float32)
                b = np.arange(d1, dtype=np.float32)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                xx = np.full_like(aa, float(idx_val))
                positions = np.stack([xx, aa, bb], axis=-1).reshape(-1, 3)
            elif axis == 1:
                d0, d1 = dims[0], dims[2]
                a = np.arange(d0, dtype=np.float32)
                b = np.arange(d1, dtype=np.float32)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                yy = np.full_like(aa, float(idx_val))
                positions = np.stack([aa, yy, bb], axis=-1).reshape(-1, 3)
            else:
                d0, d1 = dims[0], dims[1]
                a = np.arange(d0, dtype=np.float32)
                b = np.arange(d1, dtype=np.float32)
                aa, bb = np.meshgrid(a, b, indexing="ij")
                zz = np.full_like(aa, float(idx_val))
                positions = np.stack([aa, bb, zz], axis=-1).reshape(-1, 3)

            # Build triangle indices for the grid (vectorised)
            r_idx = np.arange(d0 - 1)
            c_idx = np.arange(d1 - 1)
            rr, cc = np.meshgrid(r_idx, c_idx, indexing="ij")
            tl = (rr * d1 + cc).ravel()
            tr = tl + 1
            bl = tl + d1
            br = bl + 1
            indices = np.column_stack([
                np.column_stack([tl, bl, tr]),
                np.column_stack([tr, bl, br]),
            ]).reshape(-1, 3).astype(np.int32)

            # Flat normals (pointing along the slice axis)
            normals = np.zeros_like(positions)
            normals[:, axis] = 1.0

            colors = self._slice_color_cache[i][0]

            geom = gfx.Geometry(
                positions=np.ascontiguousarray(positions, dtype=np.float32),
                normals=np.ascontiguousarray(normals, dtype=np.float32),
                indices=indices,
                colors=colors,
            )
            mat = gfx.MeshPhongMaterial(color_mode="vertex")
            if sdef.opacity < 1.0:
                mat.transparent = True
                mat.opacity = sdef.opacity
            mesh = gfx.Mesh(geom, mat)
            scene.add(mesh)
            self._slice_world_objects[key] = mesh

        # --- particles (sphere at origin, positioned via local.position) ---
        # Trajectories are already pre-computed in show() before _build_scene
        for k, pdef in enumerate(self._particles):
            traj = self._particle_trajectories[k]
            pos, idx, nrm = _make_sphere((0, 0, 0), pdef.radius, resolution=12)
            mesh = self._create_gfx_mesh(pos, idx, nrm, pdef.color, pdef.opacity)
            mesh.local.position = (float(traj[0][0]), float(traj[0][1]), float(traj[0][2]))
            scene.add(mesh)
            self._particle_world_objects.append(mesh)

    def _global_clim(self, sdef: _SliceDef):
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

    def _scalar_to_colors(self, values, clim, cmap_name, opacity):
        """Map scalar values to RGBA vertex colours."""
        import pygfx as gfx

        lo, hi = clim
        t = np.clip((values - lo) / max(hi - lo, 1e-10), 0, 1)

        # Simple built-in colormaps
        if cmap_name in ("coolwarm", "RdYlBu_r"):
            # Blue → white → red
            r = np.where(t < 0.5, 0.2 + t * 1.6, 1.0)
            g = np.where(t < 0.5, 0.2 + t * 1.6, 1.0 - (t - 0.5) * 1.6)
            b = np.where(t < 0.5, 1.0, 1.0 - (t - 0.5) * 1.6)
        elif cmap_name == "viridis":
            r = 0.267 + t * 0.726
            g = 0.004 + t * 0.329 + (1 - t) * 0.329
            b = 0.329 + t * (-0.329) + (1 - t) * 0.152
        else:
            # Grayscale fallback
            r = g = b = t

        colors = np.stack([
            np.clip(r, 0, 1),
            np.clip(g, 0, 1),
            np.clip(b, 0, 1),
            np.full_like(t, opacity),
        ], axis=-1).astype(np.float32)
        return colors

    # ------------------------------------------------------------------
    # Frame update
    # ------------------------------------------------------------------

    def _update_frame(self, frame: int, light: bool = False):
        import pygfx as gfx

        frame = max(0, min(frame, self._n_frames - 1))
        self._frame_idx = frame

        # --- isosurfaces (from pre-computed cache) ---
        if hasattr(self, "_iso_cache"):
            for i, idef in enumerate(self._isosurfaces):
                key = f"iso_{i}"
                old = self._iso_world_objects.get(key)
                result = self._iso_cache[i][frame]
                if result is not None:
                    pos, idx, nrm = result
                    if old is not None:
                        old.geometry.positions = gfx.Buffer(pos)
                        old.geometry.normals = gfx.Buffer(nrm)
                        old.geometry.indices = gfx.Buffer(idx)
                        old.visible = True
                    else:
                        mesh = self._create_gfx_mesh(
                            pos, idx, nrm, idef.color, idef.opacity, side="both",
                        )
                        self._scene.add(mesh)
                        self._iso_world_objects[key] = mesh
                else:
                    if old is not None:
                        old.visible = False

        # --- volume slices (from pre-computed cache) ---
        if hasattr(self, "_slice_color_cache"):
            for i, sdef in enumerate(self._slices):
                key = f"slice_{i}"
                mesh_obj = self._slice_world_objects.get(key)
                if mesh_obj is None:
                    continue
                colors = self._slice_color_cache[i][frame]
                mesh_obj.geometry.colors = gfx.Buffer(colors)

        # --- rotating meshes ---
        for j, rdef in enumerate(self._rotating_meshes):
            target_angle_deg = frame * rdef.speed
            angle_rad = math.radians(target_angle_deg)
            mesh_obj = self._rot_world_objects[j]
            ax = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[rdef.axis]
            # Build quaternion from axis-angle
            half = angle_rad / 2
            s = math.sin(half)
            qx, qy, qz, qw = ax[0] * s, ax[1] * s, ax[2] * s, math.cos(half)
            mesh_obj.local.rotation = (qx, qy, qz, qw)

        # --- particles ---
        for k, traj in enumerate(self._particle_trajectories):
            new_pos = traj[frame]
            mesh_obj = self._particle_world_objects[k]
            mesh_obj.local.position = (float(new_pos[0]), float(new_pos[1]), float(new_pos[2]))

        # --- status text ---
        if self._status_text is not None:
            t = frame * self._dt
            if self._playing:
                tag = f" [playing x{self._playback_speed}]"
            else:
                tag = " [paused]"
            self._status_text.set_text(
                f"t = {t:.3f}s  frame {frame}/{self._n_frames - 1}{tag}"
            )

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def _toggle_play(self):
        self._playing = not self._playing
        if self._playing and self._frame_idx >= self._n_frames - 1:
            self._frame_idx = 0
            self._update_frame(0, light=False)

    def _step_forward(self):
        self._playing = False
        nxt = min(self._frame_idx + 1, self._n_frames - 1)
        self._update_frame(nxt, light=False)

    def _step_backward(self):
        self._playing = False
        prev = max(self._frame_idx - 1, 0)
        self._update_frame(prev, light=False)

    def _speed_up(self):
        self._playback_speed = min(self._playback_speed * 2, 64)
        print(f"Speed: x{self._playback_speed}")

    def _speed_down(self):
        self._playback_speed = max(self._playback_speed // 2, 1)
        print(f"Speed: x{self._playback_speed}")

    def _go_to_start(self):
        self._playing = False
        self._update_frame(0, light=False)

    def _go_to_end(self):
        self._playing = False
        self._update_frame(self._n_frames - 1, light=False)

    # ------------------------------------------------------------------
    # Show
    # ------------------------------------------------------------------

    def show(self):
        """Open the interactive viewer window."""
        import pygfx as gfx
        from rendercanvas.auto import RenderCanvas, loop

        # Pre-compute all heavy data BEFORE opening the window so
        # the GUI never blocks on per-frame computation.
        print("Pre-computing frame data...")
        t0 = time.perf_counter()
        self._precompute_isosurfaces()
        self._precompute_slices()
        # Particle trajectories
        self._particle_trajectories = []
        for pdef in self._particles:
            print(f"  Pre-computing particle trajectory from "
                  f"({pdef.start_pos[0]:.1f}, {pdef.start_pos[1]:.1f}, "
                  f"{pdef.start_pos[2]:.1f})...")
            self._particle_trajectories.append(
                self._precompute_particle_trajectory(pdef)
            )
        elapsed = time.perf_counter() - t0
        print(f"Pre-computation done in {elapsed:.1f}s")

        canvas = RenderCanvas(
            size=self._window_size,
            title=self._title,
        )
        renderer = gfx.renderers.WgpuRenderer(canvas)

        scene = gfx.Scene()
        self._scene = scene

        # Background
        bg = _hex_to_rgba(self._background)
        scene.add(gfx.Background.from_color(bg[:3]))

        # Lighting
        scene.add(gfx.AmbientLight(intensity=0.5))
        light = gfx.DirectionalLight(intensity=0.7)
        light.local.position = (50, 50, 50)
        scene.add(light)

        # Camera
        camera = gfx.PerspectiveCamera(70)

        # Build scene elements
        print("Building GPU scene...")
        self._build_scene(scene)

        # Position camera to see everything
        camera.show_object(scene)

        # Override camera orientation if camera_up was specified.
        if self._camera_up is not None:
            ux, uy, uz = self._camera_up
            bbox = scene.get_bounding_box()
            if bbox is not None:
                center = (bbox[0] + bbox[1]) / 2
                size = bbox[1] - bbox[0]
            else:
                center = np.zeros(3)
                size = np.array([50.0, 50.0, 50.0])

            # Compute the distance needed to frame the scene at the
            # camera's FOV (default 70°).  Use the two axes visible
            # in the viewport (perpendicular to the view direction).
            fov_rad = camera.fov * math.pi / 180
            half_tan = math.tan(fov_rad / 2)
            elev_deg = 25  # degrees above the horizontal plane
            elev = math.radians(elev_deg)

            if abs(uz) > 0.5:
                # z-up: camera looks roughly along +y, x is horizontal,
                # z is vertical in the viewport.
                horiz = float(size[0])  # x extent
                vert = float(size[2])   # z extent
                # Account for aspect ratio (wider windows need less distance)
                w, h = self._window_size
                aspect = w / max(h, 1)
                dist_h = horiz / (2 * half_tan * aspect)
                dist_v = vert / (2 * half_tan)
                dist = max(dist_h, dist_v) * 1.4
                camera.local.position = (
                    float(center[0]),
                    float(center[1]) - dist * math.cos(elev),
                    float(center[2]) + dist * math.sin(elev),
                )
            elif abs(uy) > 0.5:
                # y-up: camera looks roughly along +z
                horiz = float(size[0])
                vert = float(size[1])
                w, h = self._window_size
                aspect = w / max(h, 1)
                dist_h = horiz / (2 * half_tan * aspect)
                dist_v = vert / (2 * half_tan)
                dist = max(dist_h, dist_v) * 1.4
                camera.local.position = (
                    float(center[0]),
                    float(center[1]) + dist * math.sin(elev),
                    float(center[2]) - dist * math.cos(elev),
                )
            # Point camera at scene center
            camera.look_at(tuple(center))

        # Orbit controls
        controller = gfx.OrbitController(camera, register_events=renderer)

        # Status text overlay
        try:
            self._status_text = gfx.Text(
                text="",
                font_size=16,
                anchor="top-left",
                screen_space=True,
                material=gfx.TextMaterial(color="black"),
            )
            self._status_text.local.position = (10, 10, 0)
            scene.add(self._status_text)
        except Exception:
            self._status_text = None

        # Initial frame
        self._update_frame(0, light=False)

        # Playback state
        if self._autoplay:
            self._playing = True
        last_advance = time.perf_counter()
        frame_interval = 1.0 / self._playback_fps

        # Keyboard handler
        @renderer.add_event_handler("key_down")
        def on_key(event):
            nonlocal last_advance
            key = event.key
            if key == " ":
                self._toggle_play()
            elif key == "ArrowRight":
                self._step_forward()
            elif key == "ArrowLeft":
                self._step_backward()
            elif key == "ArrowUp":
                self._speed_up()
            elif key == "ArrowDown":
                self._speed_down()
            elif key == "Home":
                self._go_to_start()
            elif key == "End":
                self._go_to_end()

        # Animation loop
        def animate():
            nonlocal last_advance
            now = time.perf_counter()
            if self._playing:
                elapsed = now - last_advance
                if elapsed >= frame_interval / self._playback_speed:
                    last_advance = now
                    nxt = self._frame_idx + 1
                    if nxt >= self._n_frames:
                        self._playing = False
                        self._update_frame(self._n_frames - 1, light=False)
                    else:
                        self._update_frame(nxt, light=True)
            renderer.render(scene, camera)
            canvas.request_draw()

        print("\nControls:")
        print("  Space       play / pause")
        print("  Left/Right  step backward / forward")
        print("  Up/Down     speed up / slow down")
        print("  Home/End    jump to first / last frame")
        print("  Mouse       rotate, zoom, pan")
        if self._autoplay:
            print("Playing...")

        canvas.request_draw(animate)
        loop.run()
