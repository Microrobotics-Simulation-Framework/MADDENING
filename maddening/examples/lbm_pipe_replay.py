#!/usr/bin/env python
"""
LBM Pipe -- Post-Simulation Replay Demo
========================================

Runs a 3D Lattice Boltzmann simulation of a partially-filled pipe with
gravity and a propeller, then opens an interactive 3D viewer.

The liquid surface is rendered as an isosurface of the passive scalar
tracer field.  Particles are advected by the flow to show fluid motion.

Controls:
    Space       -- play / pause
    Left/Right  -- step backward / forward
    Home/End    -- jump to first / last frame
    Slider      -- scrub to any frame
    Mouse       -- rotate, zoom, pan the 3D scene

Usage::

    python maddening/examples/lbm_pipe_replay.py

Requirements::

    pip install maddening[gpu-viz]   # pygfx/wgpu GPU renderer
    pip install pyvista              # for building pipe/propeller meshes
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

from maddening.core.graph_manager import GraphManager
from maddening.nodes.lbm_pipe import LBMPipeNode


# -- Helpers for building domain-specific meshes --

def make_pipe_wall(nx, ny, nz, pipe_radius):
    """Build a half-pipe mesh cut along the pipe axis.

    The near side (y < center_y) is removed so the fluid interior is
    visible when looking from the default camera position.
    """
    import pyvista as pv
    cy, cz = (ny - 1) / 2.0, (nz - 1) / 2.0
    radius = pipe_radius * min(ny, nz) / 2.0
    full = pv.Cylinder(
        center=((nx - 1) / 2.0, cy, cz),
        direction=(1, 0, 0), radius=radius,
        height=float(nx), resolution=60, capping=True,
    )
    # Clip away the near half (y < cy) to expose the interior
    clipped = full.clip(normal=(0, 1, 0), origin=(0, cy, 0))
    return clipped


def make_propeller(nx, ny, nz, prop_x, prop_radius_frac, pipe_radius,
                   n_blades=3):
    """Build a multi-blade propeller mesh."""
    import pyvista as pv
    cy, cz = (ny - 1) / 2.0, (nz - 1) / 2.0
    prop_r = prop_radius_frac * pipe_radius * min(ny, nz) / 2.0
    px = float(prop_x)

    hub = pv.Cylinder(center=(px, cy, cz), direction=(1, 0, 0),
                      radius=prop_r * 0.12, height=0.6, resolution=16)
    parts = [hub]
    for i in range(n_blades):
        angle = i * 360.0 / n_blades
        blade = pv.Box(bounds=[
            px - 0.15, px + 0.15,
            cy, cy + prop_r * 0.92,
            cz - prop_r * 0.10, cz + prop_r * 0.10,
        ])
        blade.rotate_x(angle, point=(px, cy, cz), inplace=True)
        blade.rotate_y(12, point=(px, cy, cz), inplace=True)
        parts.append(blade)
    mesh = parts[0]
    for p in parts[1:]:
        mesh = mesh.merge(p)
    return mesh


def main():
    print("=" * 60)
    print("  MADDENING LBM Pipe -- Post-Simulation Replay Demo")
    print("=" * 60)

    # --- Grid parameters ---
    # Shan-Chen multiphase LBM: surface tension, waves, and phase
    # separation emerge naturally from the pseudopotential interaction.
    # The tracer field is derived from density (liquid=1, gas=0).
    nx, ny, nz = 48, 24, 24
    tau = 0.8
    prop_strength = 0.01
    prop_x = nx // 6  # = 8
    pipe_radius = 0.9
    gravity = -0.0008
    fill_fraction = 0.75
    n_steps = 800

    # Shan-Chen multiphase parameters.
    # rho_liquid/rho_gas are set close to the EOS coexistence densities
    # for G=-4.5 so the surface level is stable from the start (minimal
    # transient settling).
    G = -4.5           # interaction strength (phase separation)
    rho_liquid = 1.5   # initial liquid density (~equilibrium for G=-4.5)
    rho_gas = 0.15     # initial gas density  (~equilibrium for G=-4.5)
    rho_0 = 1.0        # pseudopotential reference density

    print(f"\n  Grid:       {nx} x {ny} x {nz} = {nx*ny*nz:,} cells")
    print(f"  Viscosity:  nu = {(tau - 0.5) / 3:.4f} (tau = {tau})")
    print(f"  Propeller:  x={prop_x}, strength={prop_strength}")
    print(f"  Gravity:    {gravity} (z-axis)")
    print(f"  Fill:       {fill_fraction*100:.0f}%")
    print(f"  Shan-Chen:  G={G}, rho_l={rho_liquid}, rho_g={rho_gas}")
    print(f"  Steps:      {n_steps}")

    # --- Build graph ---
    print("\nBuilding LBM graph...")
    gm = GraphManager()
    gm.add_node(LBMPipeNode(
        "fluid",
        timestep=0.01,
        nx=nx, ny=ny, nz=nz,
        tau=tau,
        pipe_radius=pipe_radius,
        propeller_x=prop_x,
        propeller_radius=0.8,
        propeller_strength=prop_strength,
        initial_velocity=0.0,
        gravity=gravity,
        fill_fraction=fill_fraction,
        G=G,
        rho_liquid=rho_liquid,
        rho_gas=rho_gas,
        rho_0=rho_0,
    ))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()

    # --- Run simulation ---
    print(f"Running {n_steps} simulation steps...")
    import time
    t0 = time.perf_counter()
    final_state, history = gm.run_scan_with_history(n_steps)
    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s ({n_steps / elapsed:.0f} steps/s)")

    # --- Open replay viewer ---
    print("\nSetting up GPU 3D replay viewer...")

    from maddening.viz.backends.pygfx_viewer import GPUHistoryViewer

    playback_fps = 30
    # camera_up=(0,0,1) so z-axis (gravity direction) appears vertical
    viewer = GPUHistoryViewer(history, dt=0.01, playback_fps=playback_fps,
                              camera_up=(0, 0, 1))

    cy, cz = (ny - 1) / 2.0, (nz - 1) / 2.0

    # Liquid surface (isosurface of tracer at 0.5)
    # Minimal smoothing preserves the actual surface shape from the LBM
    viewer.add_isosurface(
        "fluid", "tracer", threshold=0.5,
        color="#2288DD", opacity=1.0, smooth_n_iter=5,
    )

    # Longitudinal velocity slice (visible through the open pipe)
    viewer.add_volume_slice(
        "fluid", "velocity", component=0,
        normal="y", origin_frac=0.5,
        cmap="RdYlBu_r", clim=(0.0, 0.06),
    )

    # Pipe wall — cut lengthwise so the interior is visible
    pipe_mesh = make_pipe_wall(nx, ny, nz, pipe_radius)
    viewer.add_static_mesh(pipe_mesh, color="#AAAAAA", opacity=0.25)

    # Propeller (rotating mesh)
    # speed = degrees per frame; aim for ~2 revolutions over the full replay
    prop_rpm = 120.0  # visual RPM
    prop_deg_per_frame = prop_rpm * 360.0 / (60.0 * playback_fps)
    prop_mesh = make_propeller(nx, ny, nz, prop_x, 0.8, pipe_radius)
    viewer.add_rotating_mesh(
        prop_mesh,
        axis="x",
        speed=prop_deg_per_frame,
        center=(float(prop_x), cy, cz),
        color="#DD5533",
    )

    # Particles advected by flow — placed deep inside the liquid region.
    # The liquid settles to the bottom of the pipe under gravity + Shan-Chen
    # phase separation, so particles must start at low z to stay submerged.
    pipe_r = pipe_radius * min(ny, nz) / 2.0
    pipe_bottom = cz - pipe_r  # lowest z inside pipe
    y_clamp = (cz - pipe_r + 1.0, cz + pipe_r - 1.0)
    # Clamp z so particles stay in the liquid (bottom portion of pipe)
    z_clamp = (pipe_bottom + 1.0, pipe_bottom + pipe_r * 0.7)
    viewer.add_particle(
        "fluid", "velocity",
        start_pos=(prop_x + 4, cy, pipe_bottom + 3),
        radius=0.5, color="#22AA44",
        clamp_y=y_clamp, clamp_z=z_clamp, periodic_x=float(nx),
    )
    viewer.add_particle(
        "fluid", "velocity",
        start_pos=(prop_x + 8, cy + 2, pipe_bottom + 4),
        radius=0.4, color="#2266DD",
        clamp_y=y_clamp, clamp_z=z_clamp, periodic_x=float(nx),
    )
    viewer.add_particle(
        "fluid", "velocity",
        start_pos=(prop_x + 12, cy - 2, pipe_bottom + 2),
        radius=0.4, color="#DD6622",
        clamp_y=y_clamp, clamp_z=z_clamp, periodic_x=float(nx),
    )

    viewer.show()


if __name__ == "__main__":
    main()
