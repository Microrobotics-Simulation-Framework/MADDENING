#!/usr/bin/env python
"""
Bouncing ball with terminal-only visualization.

Displays live-updating simulation state in the terminal.  No GUI
required -- works over SSH, in tmux, etc.

Press Ctrl-C to stop.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/bouncing_ball_terminal.py
"""

import sys
import os

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.viz import StateRelay, RealtimeRunner, GraphInfo
from maddening.viz.backends import TerminalRenderer


def main():
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
        terminal.run_event_loop(interval_ms=50)
    finally:
        runner.stop()
        terminal.teardown()
        print(f"\nStopped at sim_time={runner.sim_time:.2f}s")


if __name__ == "__main__":
    main()
