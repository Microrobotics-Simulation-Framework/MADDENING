# maddening.api

FastAPI + WebSocket server wrapping a GraphManager.

## SimulationServer (`server.py`)

```python
from maddening.api.server import SimulationServer
from maddening.nodes import BallNode, TableNode

server = SimulationServer(
    node_registry={"BallNode": BallNode, "TableNode": TableNode},
    graph_manager=gm,  # optional, creates empty one if omitted
)
app = server.create_app()
```

Run with uvicorn:

```bash
uvicorn module:app --host 127.0.0.1 --port 8000
# Interactive docs at http://localhost:8000/docs
```

## Security: this API has no authentication

There is no login, no API key and no TLS.  Every route below is open to
anyone who can reach the port, including `DELETE /graph/nodes/{name}`,
`POST /checkpoint/save` and `POST /cloud/launch` — which provisions paid
GPU instances with the credentials stored on the host — while `/docs`
publishes the full route list.  So:

* **Bind it to `127.0.0.1`** (`--host 127.0.0.1`, or `MADDENING_HOST` for
  the cloud entry point) and reach it from elsewhere through an SSH
  tunnel: `ssh -L 8000:127.0.0.1:8000 user@host`.
* **If it must listen on a network**, put an authenticating,
  TLS-terminating reverse proxy in front of it and keep port 8000 closed
  in the firewall / cloud security group.
* Containers are the exception that proves the rule: a container bound to
  `127.0.0.1` is unreachable even with `-p`, so the image binds `0.0.0.0`
  and the *published port* is what you must keep private.  The server
  logs a warning at startup whenever it binds a non-loopback address
  (`maddening.api.server.warn_if_publicly_bound`).

Request sizes are bounded (`n_steps` ≤ 100000, node integer parameters ≤
10000000 and ≤ 20000000 state elements per node, bounded
surrogate-training arguments; see `/openapi.json`) so one request cannot
exhaust the host, but that is a backstop, not authentication.  Over-size
is a 422; a non-finite constructor constant remains the 400 it has always
been.

## REST Endpoints

### Graph Structure

| Method | Path | Description |
|--------|------|-------------|
| GET | `/graph` | Return graph structure (nodes, edges, external inputs) |
| POST | `/graph/nodes` | Add a node (`{type, name, timestep, params}`) |
| DELETE | `/graph/nodes/{name}` | Remove a node and its connected edges |
| POST | `/graph/edges` | Add an edge (`{source_node, target_node, source_field, target_field}`) |
| DELETE | `/graph/edges` | Remove an edge (same body as POST) |
| POST | `/graph/compile` | Compile the graph (topo-sort + JIT). Returns schedule |
| POST | `/graph/validate` | Validate the graph. Returns issues list |

### State

| Method | Path | Description |
|--------|------|-------------|
| GET | `/graph/state` | Get state of all nodes |
| GET | `/graph/state/{node_name}` | Get state of one node |
| PUT | `/graph/state/{node_name}` | Overwrite node state (`{state: {field: value}}`) |

### Simulation Control

| Method | Path | Description |
|--------|------|-------------|
| POST | `/sim/step` | Advance one timestep. Returns new state |
| POST | `/sim/run?n_steps=100` | Run N steps. Returns final state |
| POST | `/sim/start` | Start real-time runner (background thread) |
| POST | `/sim/pause` | Pause the runner |
| POST | `/sim/resume` | Resume the runner |
| POST | `/sim/stop` | Stop the runner |

### WebSocket

| Path | Description |
|------|-------------|
| `/ws/state` | Streams state snapshots at ~30 Hz as JSON `{sim_time, state}` |

## Example curl Commands

```bash
# Get graph structure
curl http://localhost:8000/graph

# Add a node
curl -X POST http://localhost:8000/graph/nodes \
  -H 'Content-Type: application/json' \
  -d '{"type":"BallNode","name":"ball","timestep":0.01,"params":{"initial_position":5.0}}'

# Add an edge
curl -X POST http://localhost:8000/graph/edges \
  -H 'Content-Type: application/json' \
  -d '{"source_node":"table","target_node":"ball","source_field":"position","target_field":"table_position"}'

# Compile
curl -X POST http://localhost:8000/graph/compile

# Step once
curl -X POST http://localhost:8000/sim/step

# Run 100 steps
curl -X POST 'http://localhost:8000/sim/run?n_steps=100'

# Get current state
curl http://localhost:8000/graph/state

# Start real-time runner + stream via WebSocket
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
```

## Dependencies

Requires `fastapi`, `pydantic`, and `uvicorn`:

```bash
pip install fastapi uvicorn
```
