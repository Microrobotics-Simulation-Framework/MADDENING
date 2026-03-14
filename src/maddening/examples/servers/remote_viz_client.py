#!/usr/bin/env python
"""
Connect to a remote simulation and visualize it.

This script is meant to run on your local workstation.  It connects to
a ``remote_sim_server.py`` running on an HPC node and renders the
state using any available backend.

Usage
-----
    # Terminal only (works over SSH, no GUI needed):
    python maddening/examples/remote_viz_client.py

    # With matplotlib scene:
    python maddening/examples/remote_viz_client.py --mode scene

    # Connect to a specific address:
    python maddening/examples/remote_viz_client.py --connect tcp://hpc-node:5555

    # SSH tunnel setup (run this first on your local machine):
    #   ssh -L 5555:localhost:5555 user@hpc-node
"""

import sys
import os
import argparse

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from maddening.viz.network import NetworkReceiver


def run_terminal(receiver):
    """Terminal-only visualization."""
    from maddening.viz.backends.terminal_renderer import TerminalRenderer
    from maddening.viz.renderer import GraphInfo

    # We don't have full GraphInfo from the remote side, so set up
    # the renderer manually with known fields.
    renderer = TerminalRenderer(receiver, config={
        "title": "MADDENING — Remote Monitor",
        "fields": {"ball": ["position", "velocity"], "table": ["position"]},
        "precision": 4,
    })
    # Minimal setup without GraphInfo
    renderer._tracked = [
        ("ball", "position"),
        ("ball", "velocity"),
        ("table", "position"),
    ]
    renderer._precision = 4

    print("Waiting for data from simulation server...")
    renderer.run_event_loop(interval_ms=50)


def run_scene(receiver):
    """Matplotlib scene + time-series visualization."""
    from maddening.viz.renderer import GraphInfo
    from maddening.viz.backends.matplotlib_renderer import (
        MatplotlibSceneRenderer,
        MatplotlibTimeSeriesRenderer,
        run_matplotlib,
    )

    # Scene renderer
    scene = MatplotlibSceneRenderer(receiver, scene_config={
        "title": "Bouncing Ball — Remote",
        "xlim": (-1, 1),
        "ylim": (-0.8, 6.0),
        "objects": [
            {
                "type": "surface",
                "node": "table",
                "y": "position",
                "depth": 0.5,
                "color": "#8B7355",
            },
            {
                "type": "circle",
                "node": "ball",
                "y": "position",
                "x": 0.0,
                "radius": 0.2,
                "color": "#DD4444",
                "edgecolor": "#991111",
            },
        ],
    })

    # Time-series renderer
    timeseries = MatplotlibTimeSeriesRenderer(receiver, plot_config={
        "title": "Bouncing Ball — Remote Time Series",
        "fields": {"ball": ["position", "velocity"]},
    })

    # GraphInfo stub — enough for setup()
    graph_info = GraphInfo(
        node_names=["table", "ball"],
        node_params={"table": {"position": 0.0}, "ball": {"initial_position": 5.0, "elasticity": 0.7}},
        node_state_fields={"table": ["position"], "ball": ["position", "velocity"]},
        edges=[],
        timestep=0.01,
    )

    scene.setup(graph_info)
    timeseries.setup(graph_info)

    print("Waiting for data from simulation server...")
    print("Close plot windows to stop.")
    run_matplotlib(scene, timeseries, interval_ms=33)


def main():
    parser = argparse.ArgumentParser(description="MADDENING remote visualization client")
    parser.add_argument("--connect", default="tcp://localhost:5555",
                        help="ZMQ connect address (default: tcp://localhost:5555)")
    parser.add_argument("--mode", choices=["terminal", "scene"], default="terminal",
                        help="Visualization mode (default: terminal)")
    args = parser.parse_args()

    receiver = NetworkReceiver(args.connect)
    receiver.start()

    try:
        if args.mode == "terminal":
            run_terminal(receiver)
        elif args.mode == "scene":
            run_scene(receiver)
    except KeyboardInterrupt:
        pass
    finally:
        receiver.stop()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
