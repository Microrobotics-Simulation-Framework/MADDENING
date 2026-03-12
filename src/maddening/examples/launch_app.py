#!/usr/bin/env python
"""
Launch the MADDENING interactive demo app.

Sets up a ball-spring-table-heat system demonstrating multi-node graph
coupling, then serves the interactive web UI.

Usage:
    python maddening/examples/launch_app.py
    # Then open http://localhost:8000/viz/app
"""

import os
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
# Use CPU by default for the demo app (reliable, fast enough for interactive use)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings
import webbrowser
import threading

import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode
from maddening.api.server import SimulationServer


def build_demo_graph() -> GraphManager:
    """Build the demo physics graph: ball + table + spring + heat rod.

    Wiring:
        table.position -> ball.table_position (collision surface)
        ball.position  -> spring.anchor_position (spring follows ball)
        ball.velocity  -> heat_rod.left_temperature (impact heating)
    """
    gm = GraphManager()

    # Nodes
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

    # Edges: data coupling between nodes
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
    print("  MADDENING Interactive Demo")
    print("=" * 60)

    print("\nBuilding physics graph...")
    gm = build_demo_graph()
    print(f"  Nodes: {list(gm._nodes.keys())}")
    print(f"  Edges: {len(gm._edges)}")
    print(f"  Schedule: {gm.schedule}")

    node_registry = {
        "BallNode": BallNode,
        "TableNode": TableNode,
        "SpringDamperNode": SpringDamperNode,
        "HeatNode": HeatNode,
    }

    server = SimulationServer(node_registry=node_registry, graph_manager=gm)
    app = server.create_app()

    print("\nStarting server at http://localhost:8000")
    print("Open http://localhost:8000/viz/app in your browser")
    print("Press Ctrl+C to stop\n")

    # Open browser after a short delay
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open("http://localhost:8000/viz/app")

    threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
