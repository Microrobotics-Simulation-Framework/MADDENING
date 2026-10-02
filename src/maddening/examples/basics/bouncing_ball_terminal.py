#!/usr/bin/env python
"""
Bouncing ball with terminal-only visualization.

Displays live-updating simulation state in the terminal.  No GUI
required -- works over SSH, in tmux, etc.

Press Ctrl-C to stop, or pass ``--duration`` to stop after that many
seconds.

Usage
-----
    python -m maddening.examples.basics.bouncing_ball_terminal
    python -m maddening.examples.basics.bouncing_ball_terminal --duration 5
"""

import argparse
import time

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.viz import StateRelay, RealtimeRunner, GraphInfo
from maddening.viz.backends import TerminalRenderer


def main(argv=None):
    parser = argparse.ArgumentParser(description="Bouncing ball, terminal monitor")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after this many seconds (default: until Ctrl-C)")
    args = parser.parse_args(argv)

    # -- Build simulation graph --
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01, position=0.0))
    gm.add_node(BallNode(
        name="ball", timestep=0.01,
        initial_position=5.0, elasticity=0.7,
    ))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()

    # -- Viz pipeline --
    relay = StateRelay()
    relay.attach(gm)
    graph_info = GraphInfo.from_graph_manager(gm)

    terminal = TerminalRenderer(relay, config={
        "title": "Bouncing Ball — Terminal Monitor",
        "fields": {"ball": ["position", "velocity"], "table": ["position"]},
        "precision": 6,
        "refresh_hz": 20,
    })
    terminal.setup(graph_info)

    # -- Run --
    runner = RealtimeRunner(gm, relay, time_scale=1.0)
    runner.start()
    print("Simulation running. Press Ctrl-C to stop.\n")

    try:
        if args.duration is None:
            terminal.run_event_loop(interval_ms=50)
        else:
            terminal.start_background(interval_ms=50)
            time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop()
        terminal.teardown()
        print(f"\nStopped at sim_time={runner.sim_time:.2f}s")


if __name__ == "__main__":
    main()
