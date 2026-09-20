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

## Security: a loopback bind is open, anything else needs a token

**Binding `127.0.0.1` changes nothing** — no login, no header, exactly as
before.  **Binding anything else requires a bearer token on every route**
except `/healthz` and the static `/viz/*` pages, because those routes
include `DELETE /graph/nodes/{name}`, `POST /checkpoint/save` and
`POST /cloud/launch`, which provisions paid GPU instances with the
credentials stored on the host.

The server has to be *told* the bind address — it cannot see the socket
uvicorn opens:

```python
server = SimulationServer(registry, bind_host=host)   # or $MADDENING_HOST
server.auth.announce(port)                            # logs a generated token once
uvicorn.run(server.create_app(), host=host, port=port)
```

A request that arrives from a routable IP is challenged even when the
server was told the bind is loopback.  That backstop is why forgetting
`bind_host` costs you a 401, not an open API.

### The token

* `MADDENING_API_TOKEN` if set.  **Set but blank raises** — that is what
  `MADDENING_API_TOKEN=$UNSET_VARIABLE` produces, and reading it as
  "authentication off" is the failure this exists to prevent.
* Otherwise a `secrets.token_urlsafe(32)` value generated at startup and
  logged once.  If nothing reads your log — a detached container, a
  batch job — set `MADDENING_API_TOKEN` yourself, or point
  `MADDENING_API_TOKEN_FILE` at a path on a mounted volume and the
  generated token is written there with mode `0600`.

### Presenting it

The token is a bearer credential with no binding to a request, a time or a
client, and there is no TLS, so every carrier below is only as private as
the path between the client and the port.  What each one buys is where the
token does *not* end up.

| Client | Carrier |
| --- | --- |
| HTTP | `Authorization: Bearer <token>`. No authenticated route accepts `?token=`, so a credential never reaches an access log through this API. |
| WebSocket, non-browser | `Authorization: Bearer <token>` on the handshake. |
| WebSocket, browser | Subprotocols `["maddening.bearer.<base64url(token)>", "maddening.v1"]`; the server selects `maddening.v1`. Browsers cannot set a header on a handshake. |
| Bundled UI | Open `/viz/app#token=<token>` — a **fragment**, which the browser never sends, so the token reaches no access log. The page removes it from the address bar immediately and keeps it in `sessionStorage`. `?token=` also works and is sometimes easier to paste, but a query string *does* reach the log. Without either, the page prompts on the first 401. |

`/docs`, `/redoc` and `/openapi.json` are **not served** when the token is
enforced: Swagger UI fetches its own schema with no `Authorization`
header, so it cannot work behind a bearer token.  Read them from a
loopback-bound instance over an SSH tunnel.

### What the token does not do

There is still **no TLS**.  The token and every state snapshot cross the
network in cleartext, so a non-loopback bind belongs on a private network
or behind a TLS-terminating proxy.  Containers are the awkward case: a
container bound to `127.0.0.1` is unreachable even with `-p`, so the image
binds `0.0.0.0` — publish the port to loopback
(`docker run -p 127.0.0.1:8000:8000`) and tunnel to it.  The server logs
what is exposed at startup
(`maddening.api.server.warn_if_publicly_bound`).

Request sizes are bounded (`n_steps` ≤ 100000, node integer parameters ≤
10000000 and ≤ 20000000 state elements per node, bounded
surrogate-training arguments; see `/openapi.json`) so one request cannot
exhaust the host, but that is a backstop, not authentication.  Over-size
is a 422; a non-finite constructor constant remains the 400 it has always
been.

## REST Endpoints

Every path below needs `Authorization: Bearer <token>` when the bind is
not loopback; `/healthz` and `/viz/*` never do.

### Meta

| Method | Path | Description |
|--------|------|-------------|
| GET | `/healthz` | Liveness probe: `{status, version}`. Never authenticated — a container probe holds no credential, and the answer says nothing about the graph |
| GET | `/viz/app`, `/viz/graph`, `/viz/render` | The bundled UIs. Never authenticated: they hold no secret, and a page that could not load could not ask for the token. Open `#token=<token>` to hand one to the page |
| GET | `/viz/auth.js` | The pages' token helper |

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

These talk to `localhost`, so they need no token.  Against a non-loopback
server add `-H "Authorization: Bearer $MADDENING_API_TOKEN"` to every
`curl`, and pass the same header to `websockets.connect` via
`additional_headers=`.

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
