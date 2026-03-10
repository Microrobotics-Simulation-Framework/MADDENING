"""
Interactive graph visualization server.

Starts a multi-node simulation graph and serves a web-based interactive
visualization at http://localhost:8000/viz/graph.  The page shows the
graph topology (nodes, edges, data flow) and updates live state in
real-time via WebSocket.

Usage::

    python maddening/examples/interactive_graph_server.py

Then open http://localhost:8000/viz/graph in your browser.
"""

import sys
sys.path.insert(0, ".")

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode


def build_demo_graph():
    """Build a multi-node demonstration graph.

    Graph topology:
        table --position--> ball.table_position
        ball --position--> spring.anchor_position
        spring --position--> ball.spring_force (via force transform)

    This creates a ball bouncing on a table with a spring pulling it
    back toward a rest position.
    """
    gm = GraphManager()

    # Nodes
    gm.add_node(TableNode("table", timestep=0.01, position=0.0))
    gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0,
                          initial_velocity=0.0, elasticity=0.8, gravity=-9.81))
    gm.add_node(SpringDamperNode("spring", timestep=0.01,
                                  stiffness=2.0, damping=0.1, mass=1.0,
                                  rest_length=3.0,
                                  initial_position=5.0))

    # Edges: table drives ball, ball drives spring anchor, spring drives ball force
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_edge("ball", "spring", "position", "anchor_position")

    gm.compile()
    return gm


def main():
    try:
        import uvicorn
    except ImportError:
        print("This example requires uvicorn. Install with: pip install uvicorn")
        sys.exit(1)

    from maddening.api.server import SimulationServer

    gm = build_demo_graph()

    server = SimulationServer(
        node_registry={
            "BallNode": BallNode,
            "TableNode": TableNode,
            "SpringDamperNode": SpringDamperNode,
        },
        graph_manager=gm,
    )
    app = server.create_app()

    print("=" * 60)
    print("MADDENING Interactive Graph Server")
    print("=" * 60)
    print()
    print("  Graph visualization: http://localhost:8000/viz/graph")
    print("  API docs:            http://localhost:8000/docs")
    print()
    print("  Nodes: table, ball, spring")
    print("  Edges: table->ball (position), ball->spring (anchor)")
    print()
    print("  Use the web UI to:")
    print("    - View the graph topology")
    print("    - Click nodes to inspect state and parameters")
    print("    - Start/pause/stop real-time simulation")
    print("    - Step through the simulation manually")
    print()
    print("  Press Ctrl-C to stop.")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
