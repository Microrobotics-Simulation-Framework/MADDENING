#!/usr/bin/env python
"""
Server-side rendering demo -- simulation rendered on the server,
streamed as compressed image frames to a thin browser client.

The server does **all** the rendering (matplotlib Agg backend) and sends
JPEG/WebP frames over WebSocket.  The browser client is an ultra-thin
display -- it just paints the received images onto a canvas.

This architecture is ideal for:
  - Remote/cloud deployment (e.g. behind AWS AppStream or similar)
  - Thin clients (mobile, low-power devices)
  - Keeping simulation + rendering co-located for minimal latency

Usage::

    python maddening/examples/launch_server_render.py

Then open http://localhost:8000/viz/render in a browser.

Controls:
  - Space: start/pause simulation
  - R: reset simulation
  - F: toggle fullscreen
  - Bottom bar: change format, quality, target FPS
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode


def build_demo_graph() -> GraphManager:
    """Build the demo physics graph: ball + table + spring + heat rod."""
    gm = GraphManager()

    gm.add_node(TableNode("table", timestep=0.01, position=0.0))
    gm.add_node(BallNode(
        "ball", timestep=0.01,
        initial_position=5.0, initial_velocity=0.0,
        elasticity=0.7, gravity=-9.81,
    ))
    gm.add_node(SpringDamperNode(
        "spring", timestep=0.01,
        stiffness=50.0, damping=2.0, mass=0.5,
        rest_length=1.5, initial_position=3.0, initial_velocity=0.0,
    ))
    gm.add_node(HeatNode(
        "heat_rod", timestep=0.01,
        n_cells=20, length=1.0,
        thermal_diffusivity=0.01, initial_temperature=20.0,
    ))

    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_edge("ball", "spring", "position", "anchor_position")
    gm.add_edge(
        "ball", "heat_rod", "velocity", "left_temperature",
        transform=lambda v: jnp.clip(jnp.abs(v) * 10.0, 0.0, 100.0),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()

    return gm


def main():
    print("=" * 60)
    print("  MADDENING Server-Side Rendering Demo")
    print("=" * 60)

    print("\nBuilding physics graph...")
    gm = build_demo_graph()
    print(f"  Nodes: {list(gm._nodes.keys())}")
    print(f"  Edges: {len(gm._edges)}")

    # --- Server-side frame renderer ---
    from maddening.api.frame_renderer import (
        ServerFrameRenderer,
        SceneConfig,
        SceneObject,
        TimeSeriesConfig,
        HeatmapConfig,
    )

    renderer = ServerFrameRenderer(
        scene=SceneConfig(
            title="Ball-Spring-Table",
            objects=[
                SceneObject(
                    node="table", y="position", kind="surface",
                    color="#8B7355", depth=0.5, linecolor="black",
                ),
                SceneObject(
                    node="ball", y="position", kind="circle",
                    x=0.0, radius=0.25, color="#DD4444", edgecolor="#991111",
                    label="Ball",
                ),
                SceneObject(
                    node="spring", y="position", kind="circle",
                    x=0.8, radius=0.15, color="#4488DD", edgecolor="#114499",
                    label="Spring mass",
                ),
            ],
            xlim=(-2, 2), ylim=(-1, 8),
        ),
        timeseries=[
            TimeSeriesConfig(
                fields=[
                    ("ball", "position", "Ball pos"),
                    ("spring", "position", "Spring pos"),
                ],
                window=500, title="Position", ylabel="m",
            ),
            TimeSeriesConfig(
                fields=[("ball", "velocity", "Ball vel")],
                window=500, title="Velocity", ylabel="m/s",
            ),
        ],
        heatmaps=[
            HeatmapConfig(
                node="heat_rod", field="temperature",
                title="Heat Rod Temperature", vmin=15, vmax=100, cmap="hot",
            ),
        ],
        width=1280, height=720, dpi=100,
        fmt="jpeg", quality=85,
    )

    # --- Create server ---
    from maddening.api.server import SimulationServer

    node_registry = {
        "BallNode": BallNode,
        "TableNode": TableNode,
        "SpringDamperNode": SpringDamperNode,
        "HeatNode": HeatNode,
    }

    server = SimulationServer(
        node_registry=node_registry,
        graph_manager=gm,
        frame_renderer=renderer,
    )
    app = server.create_app()

    print("\n  Server-side rendering pipeline:")
    print(f"    Resolution: {renderer.width}x{renderer.height}")
    print(f"    Format:     {renderer.fmt.upper()} (quality {renderer.quality})")
    print(f"    Panels:     scene + 2 time series + heatmap")
    print()
    print("  Open in browser:")
    print("    http://localhost:8000/viz/render    (server-rendered viewer)")
    print("    http://localhost:8000/viz/app       (client-rendered app)")
    print()
    print("  The browser is a thin display client -- all rendering happens")
    print("  on the server.  Suitable for remote/cloud deployment.\n")

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


if __name__ == "__main__":
    main()
