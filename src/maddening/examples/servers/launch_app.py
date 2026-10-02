#!/usr/bin/env python
"""
Launch the MADDENING interactive demo app.

Sets up a ball-spring-table-heat system demonstrating multi-node graph
coupling, then serves the interactive web UI.

Usage:
    python -m maddening.examples.servers.launch_app
    # Opens http://localhost:8000/viz/app in your browser
    python -m maddening.examples.servers.launch_app --no-browser --port 0
    # Serve on any free port and just print the address

It binds 127.0.0.1 only; to reach it from another machine:
``ssh -L 8000:127.0.0.1:8000 <host>``.
"""

import argparse
import os
import socket
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
        ball.velocity  -> heat_rod.left_temperature (the ball's speed sets
                          the rod's left-end temperature: 10 C per m/s,
                          clipped to 0-100 C)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="MADDENING interactive demo app")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port on 127.0.0.1 (default: 8000; 0 = any free port)")
    parser.add_argument("--no-browser", action="store_true",
                        help="Do not open a web browser")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  MADDENING Interactive Demo")
    print("=" * 60)

    print("\nBuilding physics graph...")
    gm = build_demo_graph()
    print(f"  Nodes: {gm.node_names}")
    print(f"  Edges: {len(gm.edges)}")
    print(f"  Schedule: {gm.schedule}")

    node_registry = {
        "BallNode": BallNode,
        "TableNode": TableNode,
        "SpringDamperNode": SpringDamperNode,
        "HeatNode": HeatNode,
    }

    server = SimulationServer(node_registry=node_registry, graph_manager=gm)
    app = server.create_app()

    # Bound to loopback: a loopback bind is the one the API serves
    # without a bearer token.  Binding 0.0.0.0 here (as this used to)
    # published a graph-mutating API on the LAN.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args.port))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"

    print(f"\nServing on {url}", flush=True)
    print(f"Open {url}/viz/app in your browser")
    print("Press Ctrl+C to stop\n", flush=True)

    if not args.no_browser:
        # Open the browser after a short delay, once the server is up
        def open_browser():
            import time
            time.sleep(1.5)
            webbrowser.open(f"{url}/viz/app")

        threading.Thread(target=open_browser, daemon=True).start()

    import uvicorn
    try:
        uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])
    except KeyboardInterrupt:   # uvicorn re-raises Ctrl-C after shutting down
        pass


if __name__ == "__main__":
    main()
