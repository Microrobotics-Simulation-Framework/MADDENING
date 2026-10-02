"""
Interactive graph visualization server.

Starts a multi-node simulation graph and serves a web-based interactive
visualization at http://localhost:8000/viz/graph.  The page shows the
graph topology (nodes, edges, data flow) and updates live state in
real-time via WebSocket.

Usage::

    python -m maddening.examples.servers.interactive_graph_server
    python -m maddening.examples.servers.interactive_graph_server --port 0   # any free port

Then open http://localhost:8000/viz/graph in your browser (the server
prints the address it is serving on).  It binds 127.0.0.1 only; to reach
it from another machine: ``ssh -L 8000:127.0.0.1:8000 <host>``.
"""

import argparse
import socket
import sys

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode


def build_demo_graph():
    """Build a multi-node demonstration graph.

    Graph topology:
        table --position--> ball.table_position
        ball --position--> spring.anchor_position

    A ball bouncing on a table, and a spring whose free end follows the
    ball (the spring does not act back on the ball: nothing feeds back,
    so there is no cycle).
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

    # Edges: table drives ball, ball drives spring anchor
    gm.add_edge("table", "ball", "position", "table_position")
    gm.add_edge("ball", "spring", "position", "anchor_position")

    gm.compile()
    return gm


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactive graph visualization server")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port on 127.0.0.1 (default: 8000; 0 = any free port)")
    args = parser.parse_args(argv)

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

    # Bound to loopback: a loopback bind is the one the API serves
    # without a bearer token.  Binding 0.0.0.0 here (as this used to)
    # published a graph-mutating API on the LAN.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args.port))
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"

    print("=" * 60)
    print("MADDENING Interactive Graph Server")
    print("=" * 60)
    print()
    print(f"  Serving on {url}", flush=True)
    print(f"  Graph visualization: {url}/viz/graph")
    print(f"  API docs:            {url}/docs")
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
    print("=" * 60, flush=True)

    try:
        uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])
    except KeyboardInterrupt:   # uvicorn re-raises Ctrl-C after shutting down
        pass


if __name__ == "__main__":
    main()
