#!/usr/bin/env python
"""
Interactive 3D LBM Pipe Simulation
===================================

Runs a Lattice Boltzmann fluid simulation in a cylindrical pipe with a
spinning propeller, rendered live in an interactive PyVista window.

The propeller visually rotates (speed proportional to thrust), and
cross-section slices + streamlines show the fluid flow developing in
real time.

Usage::

    python maddening/examples/lbm_pipe_interactive.py

Controls:
    Mouse drag     -- rotate camera
    Scroll         -- zoom
    Shift+drag     -- pan
    Sliders        -- propeller strength, slice position, steps/frame
    Space          -- pause / resume
    R              -- reset simulation
    Q              -- quit

Requirements::

    pip install maddening[viz3d]
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings
import numpy as np

try:
    import pyvista as pv
except ImportError:
    print("This example requires PyVista. Install with: pip install maddening[viz3d]")
    import sys; sys.exit(1)

from maddening.core.graph_manager import GraphManager
from maddening.nodes.lbm_pipe import LBMPipeNode


# ------------------------------------------------------------------
# Grid / simulation constants
# ------------------------------------------------------------------

NX, NY, NZ = 48, 24, 24
TAU = 0.8
PIPE_RADIUS = 0.9
PROPELLER_X = NX // 6
PROPELLER_RADIUS_FRAC = 0.8
INITIAL_PROP_STRENGTH = 0.0005
STEPS_PER_FRAME = 10
TIMER_DURATION_MS = 33          # ~30 fps
TIMER_MAX_STEPS = 1_000_000    # effectively infinite

# Derived constants
CY = (NY - 1) / 2.0
CZ = (NZ - 1) / 2.0
PROP_R = PROPELLER_RADIUS_FRAC * PIPE_RADIUS * min(NY, NZ) / 2.0
PIPE_R = PIPE_RADIUS * min(NY, NZ) / 2.0


# ------------------------------------------------------------------
# Graph builder
# ------------------------------------------------------------------

def build_graph(prop_strength: float = INITIAL_PROP_STRENGTH) -> GraphManager:
    gm = GraphManager()
    gm.add_node(LBMPipeNode(
        "fluid",
        timestep=0.01,
        nx=NX, ny=NY, nz=NZ,
        tau=TAU,
        pipe_radius=PIPE_RADIUS,
        propeller_x=PROPELLER_X,
        propeller_radius=PROPELLER_RADIUS_FRAC,
        propeller_strength=prop_strength,
        initial_velocity=0.0,
    ))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


# ------------------------------------------------------------------
# Field extraction helpers
# ------------------------------------------------------------------

def velocity_magnitude(state):
    return np.linalg.norm(
        np.asarray(state["fluid"]["velocity"]), axis=-1,
    ).astype(np.float32)


def velocity_x(state):
    return np.asarray(state["fluid"]["velocity"][:, :, :, 0], dtype=np.float32)


def velocity_vectors(state):
    return np.asarray(state["fluid"]["velocity"], dtype=np.float32)


# ------------------------------------------------------------------
# Mesh construction
# ------------------------------------------------------------------

def make_xslice_grid(ix: int):
    """y-z StructuredGrid at x = ix."""
    y = np.arange(NY, dtype=np.float32)
    z = np.arange(NZ, dtype=np.float32)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    xx = np.full_like(yy, float(ix))
    return pv.StructuredGrid(xx, yy, zz)


def make_yslice_grid(iy: int):
    """x-z StructuredGrid at y = iy."""
    x = np.arange(NX, dtype=np.float32)
    z = np.arange(NZ, dtype=np.float32)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    yy = np.full_like(xx, float(iy))
    return pv.StructuredGrid(xx, yy, zz)


def make_pipe_wall():
    return pv.Cylinder(
        center=((NX - 1) / 2.0, CY, CZ),
        direction=(1, 0, 0),
        radius=PIPE_R,
        height=float(NX),
        resolution=60,
        capping=True,
    )


def make_propeller_mesh(n_blades: int = 3):
    """3-blade propeller: hub + flat blades.

    The mesh is centred at the propeller x-position and can be rotated
    in-place each frame to animate spinning.
    """
    px = float(PROPELLER_X)
    hub = pv.Cylinder(
        center=(px, CY, CZ),
        direction=(1, 0, 0),
        radius=PROP_R * 0.12,
        height=0.6,
        resolution=16,
    )
    parts = [hub]

    for i in range(n_blades):
        angle = i * 360.0 / n_blades
        # Blade as a thin box extending outward from the hub
        blade = pv.Box(bounds=[
            px - 0.15, px + 0.15,                  # thin in x
            CY, CY + PROP_R * 0.92,                # radial extent
            CZ - PROP_R * 0.10, CZ + PROP_R * 0.10,  # blade width
        ])
        blade.rotate_x(angle, point=(px, CY, CZ), inplace=True)
        # Slight pitch so it looks like a real blade
        blade.rotate_y(12, point=(px, CY, CZ), inplace=True)
        parts.append(blade)

    mesh = parts[0]
    for p in parts[1:]:
        mesh = mesh.merge(p)
    return mesh


def build_arrows(vel_slice, ix, stride=2, scale=200.0):
    """Arrow glyphs from a y-z velocity slice at x = ix."""
    y = np.arange(NY, dtype=np.float32)
    z = np.arange(NZ, dtype=np.float32)
    yy, zz = np.meshgrid(y, z, indexing="ij")

    yy_s, zz_s = yy[::stride, ::stride], zz[::stride, ::stride]
    vel_s = vel_slice[::stride, ::stride, :]

    points = np.column_stack([
        np.full(yy_s.size, float(ix)), yy_s.ravel(), zz_s.ravel(),
    ])
    vectors = vel_s.reshape(-1, 3)
    mag = np.linalg.norm(vectors, axis=1)

    mask = mag > 1e-8
    if not np.any(mask):
        return None

    pd = pv.PolyData(points[mask])
    pd["vectors"] = vectors[mask]
    pd["magnitude"] = mag[mask]
    return pd.glyph(orient="vectors", scale="magnitude", factor=scale)


def build_streamlines(vel_field, n_lines=20, source_x_frac=0.08,
                      source_r_frac=0.6, max_length=300.0, tube_r=0.06):
    """Streamlines through the full 3D velocity field."""
    grid = pv.ImageData(dimensions=(NX, NY, NZ))
    vel_vtk = np.stack(
        [vel_field[:, :, :, c].ravel(order="F") for c in range(3)],
        axis=1,
    ).astype(np.float32)
    grid.point_data["velocity"] = vel_vtk

    source_x = source_x_frac * (NX - 1)
    source_r = source_r_frac * PIPE_R
    source = pv.Disc(
        center=(source_x, CY, CZ),
        inner=0.0, outer=source_r,
        normal=(1, 0, 0),
        r_res=max(1, int(n_lines ** 0.5)),
        c_res=max(4, n_lines),
    )

    try:
        sl = grid.streamlines_from_source(
            source, vectors="velocity",
            max_length=max_length,
            max_steps=2000,
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
    return sl.tube(radius=tube_r)


# ------------------------------------------------------------------
# Interactive viewer
# ------------------------------------------------------------------

class LBMPipeViewer:
    """Interactive 3D viewer with spinning propeller and live flow."""

    def __init__(self):
        # -- simulation state --
        self.gm = build_graph()
        self.state = self.gm._state
        self.sim_time = 0.0
        self.total_steps = 0
        self.prop_strength = INITIAL_PROP_STRENGTH
        self.steps_per_frame = STEPS_PER_FRAME
        self.paused = False

        # Cross-section positions
        self.xslice_idx = PROPELLER_X
        self.xslice2_idx = NX // 2
        self.yslice_idx = NY // 2

        # Colour range (auto-expands)
        self.vel_clim = [0.0, 0.005]

        # Propeller animation
        self._prop_angle = 0.0      # accumulated rotation (degrees)

        # Streamline update cadence
        self._frame_count = 0
        self._streamline_interval = 15  # update every N frames

        # -- plotter --
        self.plotter = pv.Plotter(
            window_size=[1400, 800],
            title="MADDENING -- LBM Pipe Flow",
        )
        self.plotter.set_background("#f0f0f0")

        # Pre-build meshes that get their data updated each frame
        self._xslice_mesh = make_xslice_grid(self.xslice_idx)
        self._xslice2_mesh = make_xslice_grid(self.xslice2_idx)
        self._yslice_mesh = make_yslice_grid(self.yslice_idx)
        self._prop_mesh = make_propeller_mesh()

        # Dynamic actors (removed / re-added each frame)
        self._arrow_actor = None
        self._stream_actor = None

        self._setup_scene()
        self._setup_widgets()
        self._setup_keys()

    # ----- scene --------------------------------------------------

    def _setup_scene(self):
        p = self.plotter

        # Pipe wall (translucent)
        p.add_mesh(make_pipe_wall(), color="#BBBBBB", opacity=0.08,
                    smooth_shading=True)

        # Spinning propeller
        p.add_mesh(self._prop_mesh, color="#DD5533", opacity=0.85,
                    smooth_shading=True)

        # Cross-section at movable position
        vel_mag = velocity_magnitude(self.state)
        self._xslice_mesh.point_data["s"] = (
            vel_mag[self.xslice_idx, :, :].ravel(order="F")
        )
        p.add_mesh(
            self._xslice_mesh, scalars="s",
            cmap="coolwarm", clim=self.vel_clim,
            opacity=0.92, show_scalar_bar=True,
            scalar_bar_args={
                "title": "|u|",
                "position_x": 0.84, "position_y": 0.05,
                "width": 0.10, "height": 0.35,
            },
        )

        # Cross-section at mid-pipe
        self._xslice2_mesh.point_data["s"] = (
            vel_mag[self.xslice2_idx, :, :].ravel(order="F")
        )
        p.add_mesh(self._xslice2_mesh, scalars="s",
                    cmap="coolwarm", clim=self.vel_clim,
                    opacity=0.92, show_scalar_bar=False)

        # Longitudinal (x-z) slice
        vel_x = velocity_x(self.state)
        self._yslice_mesh.point_data["s"] = (
            vel_x[:, self.yslice_idx, :].ravel(order="F")
        )
        p.add_mesh(self._yslice_mesh, scalars="s",
                    cmap="RdYlBu_r", clim=self.vel_clim,
                    opacity=0.80, show_scalar_bar=False)

        # Axes widget
        p.add_axes()

        # Camera: isometric-ish view from above-front
        cx = (NX - 1) / 2.0
        p.camera.focal_point = (cx, CY, CZ)
        p.camera.position = (cx - NX * 0.15, CY - NX * 1.2, CZ + NX * 0.7)
        p.camera.up = (0, 0, 1)

        # Status text
        p.add_text(self._status_text(), position="upper_left",
                    font_size=10, color="black", name="status")

    def _status_text(self):
        tag = "  [PAUSED]" if self.paused else ""
        return (
            f"t = {self.sim_time:.3f} s   "
            f"step {self.total_steps}   "
            f"F = {self.prop_strength:.5f}{tag}"
        )

    # ----- widgets ------------------------------------------------

    def _setup_widgets(self):
        p = self.plotter
        p.add_slider_widget(
            self._on_prop_strength,
            rng=[0.0, 0.003], value=self.prop_strength,
            title="Propeller strength",
            pointa=(0.02, 0.92), pointb=(0.35, 0.92),
            style="modern",
        )
        p.add_slider_widget(
            self._on_steps_per_frame,
            rng=[1, 50], value=self.steps_per_frame,
            title="Steps / frame",
            pointa=(0.02, 0.82), pointb=(0.35, 0.82),
            style="modern",
        )
        p.add_slider_widget(
            self._on_xslice_pos,
            rng=[0, NX - 1], value=self.xslice_idx,
            title="X-slice position",
            pointa=(0.02, 0.72), pointb=(0.35, 0.72),
            style="modern",
        )

    def _on_prop_strength(self, value):
        self.prop_strength = float(value)

    def _on_steps_per_frame(self, value):
        self.steps_per_frame = max(1, int(round(value)))

    def _on_xslice_pos(self, value):
        idx = max(0, min(NX - 1, int(round(value))))
        if idx != self.xslice_idx:
            self.xslice_idx = idx
            self._xslice_mesh.points = make_xslice_grid(idx).points

    # ----- keyboard -----------------------------------------------

    def _setup_keys(self):
        self.plotter.add_key_event("r", self._reset_sim)
        self.plotter.add_key_event("space", self._toggle_pause)

    def _reset_sim(self):
        self.gm = build_graph(self.prop_strength)
        self.state = self.gm._state
        self.sim_time = 0.0
        self.total_steps = 0
        self._prop_angle = 0.0
        self.vel_clim = [0.0, 0.005]
        print("Simulation reset.")

    def _toggle_pause(self):
        self.paused = not self.paused
        print("Paused." if self.paused else "Resumed.")

    # ----- animation tick -----------------------------------------

    def _tick(self, step: int):
        """Timer callback: advance simulation, spin propeller, update viz."""

        # --- spin the propeller (even when paused, for visual feedback) ---
        # Angular speed proportional to propeller strength
        rpm = self.prop_strength * 80_000   # tuned for visual effect
        delta_deg = rpm * (TIMER_DURATION_MS / 1000.0) * 6.0  # 1 rpm = 6 deg/s
        if delta_deg > 0:
            px = float(PROPELLER_X)
            self._prop_mesh.rotate_x(
                delta_deg, point=(px, CY, CZ), inplace=True,
            )
            self._prop_angle += delta_deg

        if self.paused:
            self.plotter.add_text(
                self._status_text(), position="upper_left",
                font_size=10, color="black", name="status",
            )
            return

        # --- step the simulation ---
        update_fn = self.gm._nodes["fluid"].update_fn
        bi = {"propeller_force": self.prop_strength}
        for _ in range(self.steps_per_frame):
            self.state["fluid"] = update_fn(self.state["fluid"], bi, 0.01)
            self.sim_time += 0.01
            self.total_steps += 1

        # --- extract fields ---
        vel_mag = velocity_magnitude(self.state)
        vel_x_field = velocity_x(self.state)
        vel_vecs = velocity_vectors(self.state)

        # auto-adjust colour range
        vmax = float(np.max(vel_mag))
        if vmax > self.vel_clim[1]:
            self.vel_clim[1] = vmax * 1.05
        elif vmax > 0 and vmax < self.vel_clim[1] * 0.5:
            self.vel_clim[1] = max(vmax * 1.3, 0.001)

        # --- update slice meshes (in-place, fast) ---
        self._xslice_mesh.point_data["s"] = (
            vel_mag[self.xslice_idx, :, :].ravel(order="F")
        )
        self._xslice2_mesh.point_data["s"] = (
            vel_mag[self.xslice2_idx, :, :].ravel(order="F")
        )
        self._yslice_mesh.point_data["s"] = (
            vel_x_field[:, self.yslice_idx, :].ravel(order="F")
        )

        # --- arrows at movable slice ---
        if self._arrow_actor is not None:
            self.plotter.remove_actor(self._arrow_actor)
            self._arrow_actor = None
        arrows = build_arrows(
            vel_vecs[self.xslice_idx, :, :, :],
            self.xslice_idx,
            stride=2, scale=200.0,
        )
        if arrows is not None:
            self._arrow_actor = self.plotter.add_mesh(
                arrows, scalars="magnitude", cmap="coolwarm",
                clim=self.vel_clim, show_scalar_bar=False,
            )

        # --- streamlines (expensive, update less often) ---
        self._frame_count += 1
        if self._frame_count % self._streamline_interval == 0:
            if self._stream_actor is not None:
                self.plotter.remove_actor(self._stream_actor)
                self._stream_actor = None
            tubes = build_streamlines(vel_vecs, n_lines=20, tube_r=0.06)
            if tubes is not None:
                scalars = "speed" if "speed" in tubes.point_data else None
                self._stream_actor = self.plotter.add_mesh(
                    tubes, scalars=scalars, cmap="coolwarm",
                    clim=self.vel_clim, show_scalar_bar=False,
                    opacity=0.7,
                )

        # --- status text ---
        self.plotter.add_text(
            self._status_text(), position="upper_left",
            font_size=10, color="black", name="status",
        )

    # ----- run ----------------------------------------------------

    def run(self):
        """Open the interactive window and start the simulation loop."""
        self.plotter.add_timer_event(
            max_steps=TIMER_MAX_STEPS,
            duration=TIMER_DURATION_MS,
            callback=self._tick,
        )
        self.plotter.show()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("=" * 62)
    print("  MADDENING -- Interactive LBM Pipe Flow (3D)")
    print("=" * 62)
    print(f"\n  Grid:       {NX} x {NY} x {NZ} = {NX*NY*NZ:,} cells")
    print(f"  Viscosity:  nu = {(TAU - 0.5) / 3:.4f} (tau = {TAU})")
    print(f"  Propeller:  x = {PROPELLER_X}, strength = {INITIAL_PROP_STRENGTH}")
    print()
    print("  Controls:")
    print("    Mouse drag   -- rotate")
    print("    Scroll        -- zoom")
    print("    Shift+drag   -- pan")
    print("    Space        -- pause / resume")
    print("    R            -- reset simulation")
    print("    Q            -- quit")
    print("    Sliders      -- propeller strength, steps/frame, slice pos")
    print()

    viewer = LBMPipeViewer()
    viewer.run()


if __name__ == "__main__":
    import sys
    if "--gpu" in sys.argv:
        os.environ["JAX_PLATFORMS"] = ""
        print("GPU mode: JAX auto-detecting backend")
    main()
