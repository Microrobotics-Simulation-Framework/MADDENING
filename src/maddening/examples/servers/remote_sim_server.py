#!/usr/bin/env python
"""
Run a simulation and publish its state over ZeroMQ.

A separate visualization client (``remote_viz_client``) subscribes and
renders it.  The two can run on one machine or on two.

Usage
-----
    # Everything on this machine, in one command -- the client starts
    # this server for you on a free loopback port and stops it on exit:
    python -m maddening.examples.servers.remote_viz_client --local

    # Or run the two halves yourself, in two terminals:
    python -m maddening.examples.servers.remote_sim_server
    python -m maddening.examples.servers.remote_viz_client

    # Simulation on a remote / HPC node, viewer on your machine, over an
    # SSH tunnel (the server keeps its loopback bind, so no credential):
    #   1. On your local machine:  ssh -L 5555:localhost:5555 user@hpc-node
    #   2. On the HPC node:        python -m maddening.examples.servers.remote_sim_server
    #   3. On your local machine:  python -m maddening.examples.servers.remote_viz_client

    # To accept connections from other hosts without a tunnel, bind a
    # non-loopback address.  That turns on ZMQ CURVE encryption and needs
    # the same shared token on both sides (MADDENING_TRANSPORT_TOKEN,
    # falling back to the HTTP API's MADDENING_API_TOKEN):
    MADDENING_TRANSPORT_TOKEN=<secret> \\
      python -m maddening.examples.servers.remote_sim_server --bind 'tcp://*:5555'

The server runs until Ctrl-C (or SIGTERM), or for ``--duration`` seconds.
"""

import argparse
import signal
import time

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.viz.network import NetworkRelay


def _raise_keyboard_interrupt(signum, frame):
    """Turn SIGTERM into the same clean shutdown as Ctrl-C."""
    raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description="MADDENING remote simulation server")
    parser.add_argument(
        "--bind",
        default="tcp://127.0.0.1:5555",
        help=(
            "ZMQ bind address (default: tcp://127.0.0.1:5555). A loopback "
            "bind needs no credential and is what the local and SSH-tunnel "
            "recipes above expect. Any other address turns on CURVE "
            "encryption and requires MADDENING_TRANSPORT_TOKEN (or "
            "MADDENING_API_TOKEN) on both sides."
        ),
    )
    parser.add_argument("--time-scale", type=float, default=1.0,
                        help="Simulation speed multiplier (default: 1.0)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many wall-clock seconds "
                             "(default: run until Ctrl-C)")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _raise_keyboard_interrupt)

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

    # ``flush``: the local mode of remote_viz_client waits for this line.
    print(f"Simulation publishing on {args.bind}", flush=True)
    print(f"Time scale: {args.time_scale}x", flush=True)
    print("Press Ctrl-C to stop.\n", flush=True)

    # -- Run simulation with wall-clock pacing --
    dt = gm.timestep
    sim_time = 0.0
    wall_start = time.perf_counter()
    deadline = None if args.duration is None else wall_start + args.duration

    try:
        while deadline is None or time.perf_counter() < deadline:
            gm.step()
            sim_time += dt

            # Pace to wall clock
            target_wall = wall_start + sim_time / args.time_scale
            now = time.perf_counter()
            sleep_time = target_wall - now
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass
    finally:
        # A second Ctrl-C (a terminal signals the whole process group,
        # and a supervising client may send its own) must not cut the
        # shutdown short.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        relay.close()
        print(f"\nStopped at sim_time={sim_time:.2f}s", flush=True)


if __name__ == "__main__":
    main()
