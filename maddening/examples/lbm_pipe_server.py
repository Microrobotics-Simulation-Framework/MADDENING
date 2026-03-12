#!/usr/bin/env python
"""
LBM Pipe + 3D Server-Side Rendering Demo
=========================================

Runs a 3D Lattice Boltzmann fluid simulation in a cylindrical pipe with
a propeller, rendered server-side using PyVista/VTK.  Compressed JPEG
frames are streamed over WebSocket to a thin browser client.

This demonstrates MADDENING's ability to handle real 3D physics with
server-side rendering -- ideal for remote/cloud deployment where
bandwidth is limited and full 3D state would be too large to ship.

Usage::

    python maddening/examples/lbm_pipe_server.py

Then open http://localhost:8000/viz/render in a browser.

Requirements::

    pip install maddening[api,viz3d]
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings

from maddening.core.graph_manager import GraphManager
from maddening.nodes.lbm_pipe import LBMPipeNode


def build_lbm_graph(
    nx: int = 48,
    ny: int = 24,
    nz: int = 24,
    tau: float = 0.8,
    propeller_strength: float = 0.0005,
) -> GraphManager:
    """Build a graph with a single LBM pipe node."""
    gm = GraphManager()
    gm.add_node(LBMPipeNode(
        "fluid",
        timestep=0.01,
        nx=nx, ny=ny, nz=nz,
        tau=tau,
        pipe_radius=0.9,
        propeller_x=nx // 6,
        propeller_radius=0.8,
        propeller_strength=propeller_strength,
        initial_velocity=0.0,
    ))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()

    return gm


def main():
    print("=" * 60)
    print("  MADDENING LBM Pipe -- 3D Server-Side Rendering Demo")
    print("=" * 60)

    # --- Grid parameters ---
    nx, ny, nz = 48, 24, 24
    tau = 0.8
    prop_strength = 0.0005

    print(f"\n  Grid:       {nx} x {ny} x {nz} = {nx*ny*nz:,} cells")
    print(f"  Viscosity:  nu = {(tau - 0.5) / 3:.4f} (tau = {tau})")
    print(f"  Propeller:  strength = {prop_strength}")
    print(f"  State size: ~{nx*ny*nz * (19 + 1 + 3) * 4 / 1e6:.1f} MB "
          f"(f + density + velocity)")

    print("\nBuilding LBM graph...")
    gm = build_lbm_graph(nx, ny, nz, tau, prop_strength)

    # --- 3D frame renderer ---
    from maddening.api.frame_renderer_3d import (
        ServerFrameRenderer3D,
        View3DConfig,
        SliceConfig,
        ArrowConfig,
        PipeWallConfig,
    )

    renderer = ServerFrameRenderer3D(
        config=View3DConfig(
            node="fluid",
            grid_field="density",
            slices=[
                # Cross-section at propeller: velocity magnitude
                SliceConfig(
                    node="fluid", field="velocity", component=-1,
                    normal="x", origin_frac=1.0 / 6.0,
                    cmap="coolwarm", show_colorbar=True,
                    clim=(0.0, 0.02),
                ),
                # Cross-section at mid-pipe: velocity magnitude
                SliceConfig(
                    node="fluid", field="velocity", component=-1,
                    normal="x", origin_frac=0.5,
                    cmap="coolwarm", show_colorbar=False,
                    clim=(0.0, 0.02),
                ),
                # Longitudinal section: x-velocity
                SliceConfig(
                    node="fluid", field="velocity", component=0,
                    normal="y", origin_frac=0.5,
                    cmap="RdYlBu_r", show_colorbar=True,
                    clim=(0.0, 0.02),
                ),
            ],
            arrows=[
                # Velocity arrows at propeller
                ArrowConfig(
                    node="fluid", field="velocity",
                    normal="x", origin_frac=1.0 / 6.0,
                    scale=200.0, stride=2,
                    cmap="coolwarm", clim=(0.0, 0.02),
                ),
            ],
            pipe_wall=PipeWallConfig(
                radius_frac=0.9,
                color="#BBBBBB",
                opacity=0.10,
            ),
            camera_position="xz",
            camera_zoom=1.3,
            background="#f0f0f0",
            show_axes=True,
            show_time=True,
        ),
        width=1280,
        height=720,
        fmt="jpeg",
        quality=85,
    )

    # --- Create server ---
    from maddening.api.server import SimulationServer

    server = SimulationServer(
        node_registry={"LBMPipeNode": LBMPipeNode},
        graph_manager=gm,
        frame_renderer=renderer,
    )
    app = server.create_app()

    print(f"\n  3D Renderer:")
    print(f"    Resolution: {renderer.width}x{renderer.height}")
    print(f"    Format:     JPEG (quality {renderer._quality})")
    print(f"    Panels:     2 cross-sections + longitudinal + arrows + pipe")
    print()
    print("  Open in browser:")
    print("    http://localhost:8000/viz/render    (3D server-rendered viewer)")
    print("    http://localhost:8000/viz/graph     (interactive graph topology)")
    print()
    print("  The browser is a thin display client -- 3D rendering happens")
    print("  on the server using VTK offscreen.  Only JPEG frames are sent.\n")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
