#!/usr/bin/env python
"""
API server demo -- bouncing ball + spring graph served over HTTP/WebSocket.

Starts a FastAPI server with a pre-loaded simulation graph:
  - A ball at height 5, bouncing on a table at height 0
  - A spring-damper anchored to the ball

Usage
-----
    pip install "maddening[api]"   # fastapi, uvicorn, websockets
    python -m maddening.examples.servers.api_server
    python -m maddening.examples.servers.api_server --port 0   # any free port

Then open http://localhost:8000/docs for the interactive API docs (the
server prints the address it is serving on; ``--port 0`` lets the OS pick
a free one).  It binds 127.0.0.1 only; to reach it from another machine,
forward the port: ``ssh -L 8000:127.0.0.1:8000 <host>``.

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

import argparse
import socket
import sys

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


def bind_loopback(port: int) -> socket.socket:
    """Bind 127.0.0.1:*port* (0 = a free port the OS picks) for uvicorn."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", port))
    return sock


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="MADDENING API server demo")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port on 127.0.0.1 (default: 8000; 0 = any free port)")
    args = parser.parse_args(argv)

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

    # Bound to loopback: a loopback bind is the one the API serves
    # without a bearer token.  Binding 0.0.0.0 here (as this used to)
    # published a graph-mutating API on the LAN.  To reach it from
    # another machine: ssh -L 8000:127.0.0.1:8000 <host>.
    sock = bind_loopback(args.port)
    url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    print(f"\nServing on {url}", flush=True)
    print(f"Interactive docs at {url}/docs\n", flush=True)
    try:
        uvicorn.Server(uvicorn.Config(app, proxy_headers=False)).run(sockets=[sock])
    except KeyboardInterrupt:   # uvicorn re-raises Ctrl-C after shutting down
        pass


if __name__ == "__main__":
    main()
