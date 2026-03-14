#!/usr/bin/env python
"""
API server demo -- bouncing ball + spring graph served over HTTP/WebSocket.

Starts a FastAPI server with a pre-loaded simulation graph:
  - A ball at height 5, bouncing on a table at height 0
  - A spring-damper anchored to the ball

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    pip install fastapi uvicorn   # if not already installed
    python maddening/examples/api_server.py

Then open http://localhost:8000/docs for the interactive API docs.

Quick smoke test
----------------
    # Get graph structure
    curl http://localhost:8000/graph

    # Step the simulation once
    curl -X POST http://localhost:8000/sim/step

    # Run 100 steps
    curl -X POST 'http://localhost:8000/sim/run?n_steps=100'

    # Get state
    curl http://localhost:8000/graph/state

    # Start real-time runner, then stream via WebSocket
    curl -X POST http://localhost:8000/sim/start
    python -c "
import asyncio, websockets, json
async def listen():
    async with websockets.connect('ws://localhost:8000/ws/state') as ws:
        for _ in range(10):
            msg = json.loads(await ws.recv())
            print(f't={msg[\"sim_time\"]:.3f}  ball={msg[\"state\"][\"ball\"][\"position\"]:.4f}')
asyncio.run(listen())
"
"""

import sys
import os

# Ensure the project root is on the path.
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, SpringDamperNode, TableNode


def build_demo_graph() -> GraphManager:
    """Build a bouncing ball + spring demo graph."""
    gm = GraphManager()

    table = TableNode(name="table", timestep=0.01, position=0.0)
    ball = BallNode(
        name="ball",
        timestep=0.01,
        initial_position=5.0,
        initial_velocity=0.0,
        elasticity=0.7,
    )
    spring = SpringDamperNode(
        name="spring",
        timestep=0.01,
        stiffness=50.0,
        damping=2.0,
        mass=0.5,
        rest_length=1.0,
        initial_position=4.0,
        initial_velocity=0.0,
    )

    gm.add_node(table)
    gm.add_node(ball)
    gm.add_node(spring)

    # Wire: table.position -> ball.table_position
    gm.add_edge(
        source="table",
        target="ball",
        source_field="position",
        target_field="table_position",
    )
    # Wire: ball.position -> spring.anchor_position
    gm.add_edge(
        source="ball",
        target="spring",
        source_field="position",
        target_field="anchor_position",
    )

    gm.compile()
    return gm


def main() -> None:
    gm = build_demo_graph()
    print(f"Graph: {gm}")
    print(f"Schedule: {gm.schedule}")

    registry = {
        "BallNode": BallNode,
        "TableNode": TableNode,
        "SpringDamperNode": SpringDamperNode,
    }

    server = SimulationServer(node_registry=registry, graph_manager=gm)
    app = server.create_app()

    try:
        import uvicorn
    except ImportError:
        print(
            "uvicorn is required to run the server. "
            "Install it with:  pip install uvicorn"
        )
        sys.exit(1)

    print("\nStarting MADDENING API server on http://0.0.0.0:8000")
    print("Interactive docs at http://0.0.0.0:8000/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
