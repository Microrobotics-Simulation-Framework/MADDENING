#!/usr/bin/env python
"""
Run a simulation and publish state over the network.

This script is meant to run on the HPC / compute node.  A separate
visualization client (``remote_viz_client.py``) connects and renders
the results.

Usage
-----
    # On the simulation machine:
    python maddening/examples/remote_sim_server.py

    # To allow remote connections, specify the bind address:
    python maddening/examples/remote_sim_server.py --bind tcp://*:5555

    # For SSH tunnel usage (most common in HPC):
    #   1. On your local machine:  ssh -L 5555:localhost:5555 user@hpc-node
    #   2. On the HPC node:        python maddening/examples/remote_sim_server.py
    #   3. On your local machine:  python maddening/examples/remote_viz_client.py
"""

import sys
import os
import argparse
import time

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.viz.network import NetworkRelay


def main():
    parser = argparse.ArgumentParser(description="MADDENING remote simulation server")
    parser.add_argument("--bind", default="tcp://*:5555", help="ZMQ bind address (default: tcp://*:5555)")
    parser.add_argument("--time-scale", type=float, default=1.0, help="Simulation speed multiplier (default: 1.0)")
    args = parser.parse_args()

    # -- Build simulation --
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
    gm.add_node(BallNode(
        name="ball", timestep=0.01,
        initial_position=5.0, elasticity=0.7,
    ))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()

    # -- Publish state over network --
    relay = NetworkRelay(args.bind)
    relay.attach(gm)

    print(f"Simulation publishing on {args.bind}")
    print(f"Time scale: {args.time_scale}x")
    print("Press Ctrl-C to stop.\n")

    # -- Run simulation with wall-clock pacing --
    dt = gm.timestep
    sim_time = 0.0
    wall_start = time.perf_counter()

    try:
        while True:
            gm.step()
            sim_time += dt

            # Pace to wall clock
            target_wall = wall_start + sim_time / args.time_scale
            now = time.perf_counter()
            sleep_time = target_wall - now
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print(f"\nStopped at sim_time={sim_time:.2f}s")
    finally:
        relay.close()


if __name__ == "__main__":
    main()
