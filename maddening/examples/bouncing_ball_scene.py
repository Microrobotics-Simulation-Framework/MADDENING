#!/usr/bin/env python
"""
Bouncing ball with 2D scene visualization.

Shows an animated ball bouncing on a table surface alongside
position / velocity time-series plots, all driven through the
Renderer ABC via ``MatplotlibSceneRenderer`` and
``MatplotlibTimeSeriesRenderer``.

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/bouncing_ball_scene.py
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
from maddening.viz.backends import (
    MatplotlibSceneRenderer,
    MatplotlibTimeSeriesRenderer,
    run_matplotlib,
)


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
    print(f"Graph: {gm}")

    # -- Viz pipeline --
    relay = StateRelay()
    relay.attach(gm)
    graph_info = GraphInfo.from_graph_manager(gm)

    # Scene renderer: ball + table
    scene = MatplotlibSceneRenderer(relay, scene_config={
        "title": "Bouncing Ball",
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
    scene.setup(graph_info)

    # Time-series renderer: position + velocity
    timeseries = MatplotlibTimeSeriesRenderer(relay, plot_config={
        "title": "Bouncing Ball — Time Series",
        "fields": {"ball": ["position", "velocity"]},
    })
    timeseries.setup(graph_info)

    # -- Run --
    runner = RealtimeRunner(gm, relay, time_scale=1.0)
    runner.start()
    print("Simulation running. Close plot windows to stop.")

    try:
        run_matplotlib(scene, timeseries, interval_ms=33)
    finally:
        runner.stop()
        scene.teardown()
        timeseries.teardown()
        print(f"Stopped at sim_time={runner.sim_time:.2f}s")


if __name__ == "__main__":
    main()
