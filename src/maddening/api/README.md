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
uvicorn module:app --host 127.0.0.1 --port 8000 --no-proxy-headers
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
`bind_host` costs you a 401, not an open API.  On a loopback bind only a
**direct connection from a loopback address** is served without the
token: a peer that is not a loopback IP literal (Starlette's in-process
`TestClient` reports `"testclient"`, a Unix socket reports none) and any
request carrying `X-Forwarded-For` or `Forwarded` must present it.  The
peer is `scope["client"]` as the ASGI server reports it, and under
uvicorn's `proxy_headers` (on by default, trusting `127.0.0.1`) that is a
trusted proxy's `X-Forwarded-For` value -- which any request arriving over
loopback, a DNS-rebinding page's included, can set.  **Never configure
loopback as a trusted proxy for a loopback-bound server**; the library's
own launch paths pass `proxy_headers=False`.  A reverse proxy in front of a
loopback bind needs its name in `allowed_hosts` and its clients need the
token.  In-process, present `server.auth.token`, or build the client as a
loopback one: `TestClient(app, base_url="http://127.0.0.1",
client=("127.0.0.1", 50000))`.

On a loopback bind a request without a valid token is also asked its
`Host`: only `localhost`, a `127.0.0.0/8` or `::1` literal and the names
in `allowed_hosts` are served (403 otherwise), whatever its peer.  A valid
token is served under any `Host`.

### The token

* `MADDENING_API_TOKEN` if set.  **Set but blank raises** — that is what
  `MADDENING_API_TOKEN=$UNSET_VARIABLE` produces, and reading it as
  "authentication off" is the failure this exists to prevent.  **So does a
  token with whitespace before or after it** (a trailing newline from a
  file, say): the whitespace around a header's value is not part of the
  token a client presents, so no client could ever present it, and every
  request was a 401.
* Otherwise a `secrets.token_urlsafe(32)` value generated at startup and
  logged once.  If nothing reads your log — a detached container, a
  batch job — set `MADDENING_API_TOKEN` yourself, or point
  `MADDENING_API_TOKEN_FILE` at a path on a mounted volume and the
  generated token is written there with mode `0600`: into a new file moved
  over the path, so a file already there -- of any mode, or held open by
  another reader -- never holds it.  `APIAuth(environ=...)` reads all three
  variables from the mapping it is given (the token file's used to be read
  from `os.environ`).

### Presenting it

The token is a bearer credential with no binding to a request, a time or a
client, and there is no TLS, so every carrier below is only as private as
the path between the client and the port.  What each one buys is where the
token does *not* end up.

| Client | Carrier |
| --- | --- |
| HTTP | `Authorization: Bearer <token>`, read by RFC 9110's grammar and no more generously: the scheme in any case (`bearer`, `BEARER`), one or more spaces, then the token, compared exactly. Spaces and tabs around the whole header value are not part of it (an HTTP server removes them); a tab or any other character between the scheme and the token, or inside the value after the token, is no match (401). No authenticated route accepts `?token=`, so a credential never reaches an access log through this API. |
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

Request sizes are bounded so one request cannot exhaust the host by
mistake, but that is a backstop, not authentication, and not a defence
against a token-holder who sets out to exhaust it (see the next section).
More than 10^6 values in one
request's params, `n_steps` over 100000 or an out-of-range training
argument is a 422 from the request model, published in `/openapi.json`.
The lists and objects in a request's params are counted against the same
10^6 (a body of empty lists holds no value at all).
An integer over 10^7 for an **integer** parameter (an array dimension) is a
422 from the route, which reads the parameter's type -- the running node's
value on `PUT /graph/params`, the constructor's default on `POST
/graph/nodes`, a parameter with neither being bounded as written -- so an
integral JSON number for a float parameter is the float it spells: a
browser's `JSON.stringify(2e7)` is `20000000`, and it used to be a 422.  A node whose state would exceed 2·10^7 elements, or
whose build would exceed 2 GiB, or that would take the graph past 10^8
state elements, is a 400 naming the size: those caps are server constants
(`MAX_NODE_STATE_ELEMENTS`, `MAX_NODE_BUILD_BYTES`,
`MAX_GRAPH_STATE_ELEMENTS`), not part of the schema.  A non-finite
constructor constant remains the 400 it has always been.

### Who the server is for, and what it defends against

In 0.4.0 the server is written for two clients: **a trusted client on
loopback** (your own scripts and the bundled UI on the machine that runs
the simulation), and **a token-holder on a network or a cloud pod**.
Whoever holds the token is trusted as the person at the keyboard is: the
routes build nodes, run steps and write files under the checkpoint root on
their say.

It defends against:

* **an unauthenticated network peer**: every route of a non-loopback bind
  but `/healthz` and `/viz/*` needs the token, an unknown bind address is
  treated as reachable, and a peer from a routable address is challenged
  even on a server told its bind is loopback;
* **a web page in the user's browser reaching a loopback server**: the
  `Host` rule above (DNS rebinding), a 403 for a state-changing request or
  a WebSocket handshake whose `Origin` is not the server's own or one in
  `allowed_origins`, and the token demanded of any request that carries
  `X-Forwarded-For` or `Forwarded`, so a forwarded header cannot pass a
  page off as a loopback peer;
* **oversized bodies**: the 413 before a body is parsed, and the element,
  step and size bounds above;
* **path escapes from the checkpoint root**: `/checkpoint/save` and
  `/checkpoint/load` touch files under it only, whether or not the token is
  enforced.

It does **not** defend against a token-holder who means harm.  The
refusals this guide lists are the ones that exist, and each holds as
stated; beyond them the server is not claimed to withstand:

* **crafted checkpoint archives** (a load checks each member's name, shape
  and dtype before reading it; it is not hardened against an archive built
  to cost memory or time in another way);
* **resource exhaustion**: many requests, long runs, a graph at the size
  caps, slow steps, training jobs; a request body at the size limit is
  parsed on the event loop before any bound but its length is asked, and
  no route answers meanwhile;
* **pathological names** for nodes, fields and files, beyond the
  characters the routes refuse;
* and it provides **no TLS** (above): put a non-loopback bind on a private
  network or behind a TLS-terminating proxy.

So do not hand the token to a party you would not let run code on the
host, and do not expose the port to the internet with nothing in front of
it.

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
| POST | `/graph/nodes` | Add a node (`{type, name, timestep, params}`). 400, and nothing added, for a node the graph could not step with its state as built: one update is traced as the graph would call it, and must return the fields, shapes and kinds of dtype of the node's initial state (a list given for a scalar constant fails this). The streams are sent the graph with it |
| DELETE | `/graph/nodes/{name}` | Remove a node and its connected edges. A coupling group loses it as a member and keeps its options; a group left with fewer than two members is removed (the reply's `coupling_groups` says which changed; a node added back under the name is not a member, and no route adds a group). 400 when a mapping on an edge between two other nodes was built from the node's points. The streams are sent the graph without it |
| POST | `/graph/edges` | Add an edge (`{source_node, target_node, source_field, target_field}`). The target field is not checked against what the target node declares: an edge to a field the node does not read is accepted and delivers nothing |
| DELETE | `/graph/edges` | Remove an edge (same body as POST) |
| POST | `/graph/compile` | Compile the graph (topo-sort + JIT). Returns schedule |
| POST | `/graph/validate` | Validate the graph. Returns issues list |

### State

| Method | Path | Description |
|--------|------|-------------|
| GET | `/graph/state` | Get state of all nodes, `_meta` included |
| GET | `/graph/state/{node_name}` | Get state of one node |
| PUT | `/graph/state/{node_name}` | Overwrite node state (`{state: {field: value}}`). A value the field's dtype cannot hold (`1e39` into float32; for an integer field, a non-integral value or one outside its range: `0.5` or `256` into uint8), a non-finite one, text, `null` or a boolean for a numeric field is a 400 naming the field, and nothing is written. The streams are sent the written state, at their clock |

A non-finite number in any reply -- a diverged state, a coupling
diagnostic not yet filled (`diagnostics=True` seeds its spectral `_meta`
slots with NaN), a parameter a fit left non-finite -- is written as the
quoted token `"NaN"`, `"Infinity"` or `"-Infinity"`, as `GET /graph`
writes one (`maddening.serialization.json_codec.loads` reads them back).
Until 0.4.0 the reply failed instead: a 500, after a reset or a load had
already been applied.

### Parameters

| Method | Path | Description |
|--------|------|-------------|
| GET | `/graph/params/{node_name}` | The node's parameters, live values (`gm.params`) over constructor ones |
| PUT | `/graph/params/{node_name}` | Update parameters (`{params: {key: value}}`). A leaf the step reads takes effect on the next step; a structural value the node reads when traced marks the graph for recompilation; an initial condition `initial_state()` reads takes effect at the next `POST /sim/reset`. A value the running node cannot use (declared in `static_data_deps`, or consumed when the node was constructed) is a 400 naming it, and nothing in the request is written: rebuild the node (`DELETE` then `POST /graph/nodes`) to change it.  So is a request the node's constructor refuses with the params a save would carry, one the node a reload builds would compute differently with (a branch the constructor chose from the value), one that moves the points an interface mapping was built from (a uniform `HeatNode`'s `length`, from which its `grid_x` is derived), any non-finite value and any value the leaf's dtype cannot hold; integers are bounded as in `POST /graph/nodes` |

### Checkpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/checkpoint/save?path=` | Save state and parameters under the checkpoint root, and beside the file a manifest (`<path>.manifest.json`: its SHA-256 and the streams' clock, `sim_time` and step count). Both are written under temporary names and moved into place after both exist, the checkpoint first and its manifest last: a save refused before that (400, "nothing was written") leaves any earlier file of either name as it was, and a name that is not a file is refused before anything is written. If only the manifest's move fails, the 400 says the checkpoint was written without it (it loads, with `sim_time` counted from zero). No 4xx detail names the server's absolute paths; the 200 reply's `path` is the file's absolute path on the server (as the JAX trace routes' `log_dir` is). Replies `{status, path, sim_time}` |
| POST | `/checkpoint/load?path=` | Restore them. A checkpoint that does not fit this graph is a 400, and nothing is loaded: other node or field names, a field or parameter of another shape, a value its dtype cannot hold, or a parameter value `PUT /graph/params` would refuse on the node as it stands -- non-finite, outside its `ParamSpec` bounds, refused by the node's constructor with the graph's other values (a save after the load would not reload), one a node consumed at construction, one that moves a mapped edge's points, text or a boolean; each asked of what the load changes only, and finiteness and the bounds only of a value that is neither the leaf's now nor the node's own, so a graph built outside its bounds reloads its own checkpoint.  `GraphManager.load_state` refuses text and booleans and asks none of the rest (Python may hold a value outside a `ParamSpec`'s bounds on purpose). The streams then serve the loaded state at the checkpoint's `sim_time` -- the one its manifest records, or zero, counted from the load, for a file without one or whose manifest does not hash to it (`sim_time_from_checkpoint` says which). 409 while the runner runs or a `/sim/run` is in progress |

### Simulation Control

| Method | Path | Description |
|--------|------|-------------|
| POST | `/sim/step` | Advance one timestep. Returns new state |
| POST | `/sim/run?n_steps=100` | Run N steps (at most 100000). Returns final state. While it runs, writes are a 409 and reads are served between its slices. When the server shuts down it stops within a slice and answers 503 `{status: "interrupted", steps_run, n_steps}`; the graph is left after `steps_run` steps. So does a run that cannot have the graph within `_GRAPH_LOCK_TIMEOUT` -- between slices, for its final state, or for the compile of `n_steps=0` -- with the detail saying how many steps remain; retry the whole run only when `steps_run` is 0 (that reply carries `Retry-After`). A step that raises (a run-time check in the step, `strict_convergence`) answers 400 `{detail, steps_run, n_steps}`: the steps before it were taken. A run that fails unexpectedly answers 500 with `steps_run` too: the slice that failed is put back, the steps of the slices before it stay |
| POST | `/sim/start` | Start real-time runner (background thread). A runner whose thread died (a step raised) is replaced |
| POST | `/sim/pause` | Pause the runner. 409 when it is not running, saying why a started one stopped |
| POST | `/sim/resume` | Resume the runner, paced from the resume |
| POST | `/sim/stop` | Stop the runner. A runner whose thread had died is reported stopped, with `error` |
| POST | `/sim/reset` | Stop the runner and reset every node; the streams are sent the reset state at step 0. `was_running` is whether a runner was running. When the graph cannot be had in time after the runner was stopped, the 503 says the runner stays stopped, with `was_running` |
| PUT | `/sim/stride?steps_per_frame=&relay_stride=` | The runner's steps per frame and the relay's stride; a value left out keeps its current value (it used to be reset to 1). Answered at once, whatever the other runner routes wait for |
| POST | `/sim/profile?n_steps=&n_warmup=` | A step-time profile (Perfetto JSON) of `n_steps` steps (1 to 1000, default 50) after `n_warmup` (0 to 50, default 3); a value outside its range is a 422, not clamped. The graph is put back exactly as it was after it -- state, parameters, and an edited graph still waiting for its compile -- and the streams neither show nor count its steps |
| POST | `/sim/profile/jax/start`, `/sim/profile/jax/stop` | A JAX trace of the steps between them, for at most `MAX_JAX_TRACE_STEPS` (10 000) steps or `MAX_JAX_TRACE_SECONDS` (600 s): past either it stops itself and writes its files. The time budget has its own timer, so an idle trace stops at it too |
| GET | `/sim/profile/jax/status` | Whether a trace runs, its steps and budgets, its directory, and what stopped the last one |

### Surrogates

**Experimental in 0.4.0.**  These routes, and the WebSocket streams below,
are to be hardened in 0.5.0 and may change in any minor release until
then (`ROUTE_STABILITY` in `server.py`; each HTTP one carries
`x-maddening-stability` in `/openapi.json`).

| Method | Path | Description |
|--------|------|-------------|
| POST | `/surrogate/train` | Train a surrogate of one node in a background job. One job runs at a time (409 otherwise). The memory the job would take -- the data sweep over the whole graph, its dataset, the network -- is estimated first and refused (400) over `MAX_SURROGATE_TRAIN_BYTES`, and estimated again on the graph the sweep runs over: a graph that grew in between (a node added) ends the job `error`, naming the budget, before anything is swept. `width**2 * depth` of the network is bounded (422). The data come from a batched sweep that leaves the live simulation where it was. Replies `{job_id, status, estimated_bytes}` |
| GET | `/surrogate/status/{job_id}` | The job's progress. The last `MAX_SURROGATE_JOBS_KEPT` (8) finished jobs are kept; a job stopped by a shutdown reads `cancelled` |
| POST | `/surrogate/activate/{job_id}` | Replace the node with the trained surrogate (the runner is stopped first, the graph is reset). The node's edges at that moment are recorded for the revert |
| POST | `/surrogate/deactivate/{node_name}` | Put the original node back (the runner is stopped first, the graph is reset), **with the edges recorded when the surrogate was activated**: an edge added to or from the node while the surrogate was active is removed with it and is not put back. The reply's `dropped_edges` lists each such edge (`[]` when there is none); add them again with `POST /graph/edges` |

### WebSocket

**Experimental in 0.4.0**, with `StateRelay`, the snapshot buffer they
read: to be hardened in 0.5.0.  `BinaryStateEncoder`, the binary frame
format, stays `evolving`.

| Path | Description |
|------|-------------|
| `/ws/state` | Streams state snapshots at ~30 Hz as JSON `{sim_time, state}` |
| `/ws/state/binary` | The same as binary frames, after a JSON schema; a new schema is sent before the first frame of another layout (a node added, removed or replaced) |
| `/ws/render` | Server-rendered frames, when a renderer is configured |

At most `MAX_STREAM_CONNECTIONS` (16) streams are open at once; past it a
handshake is accepted and closed with 1013 (closed before the accept, a
real client saw HTTP 403, the answer to an `Origin` or token refusal).
Frames are encoded off the event loop, and a frame of the whole state is
encoded once for every client of it.

Every route that changes the state without a step publishes it: a reset
and a surrogate swap at step 0 and `sim_time` 0, a state write and a node
added or removed at the streams' clock, a checkpoint load at the
checkpoint's.  A step of `GraphManager.run_adaptive` adds its own `dt` to
the clock.

## A request that is refused, or that fails, changes nothing

**A request that is refused, or that fails unexpectedly, leaves the graph
exactly as it was.**  Every route that can change the graph -- adding or
removing a node or an edge, a compile, a state or params write, a
checkpoint save or load, a step, a run's slice, a start, a reset, a
profile, a surrogate swap -- runs in one transaction: the graph is
recorded when the graph lock is taken, and put back if the route answers
a 4xx of its own or fails at any point.  What is put back: the state, the
parameters, the nodes and their own params, the edges, the coupling
groups, the external inputs, the compile bookkeeping (a refused request
on an edited graph leaves it waiting for its compile) and the streams'
clock.

**An unexpected failure answers 500 with a generic detail**: `The request
failed unexpectedly (the server's log says how).`, and for a write route
`The graph was put back exactly as it was before the request.`  The detail
names no path, value or class of the server's; the traceback is in the
server's log.  A 500 is a defect: please report it.

What the promise does not cover:

* **`POST /sim/run` is one transaction per slice**, not per run: the lock
  is released between slices, where reads are served and
  `PUT /graph/params` is accepted, so putting the graph back to before the
  run would undo another request's accepted write.  A run that stops
  part-way leaves the graph after the steps it reports (`steps_run`): an
  unexpected failure (500) puts back the slice that failed and keeps the
  slices before it; a step that raises (400) keeps every step before that
  one; an interruption (503) keeps the slices it completed.
* **What is outside the graph**: a file under the checkpoint root (a save
  keeps its own cleanup, and says when a checkpoint was written without
  its manifest); the runner's thread and whether it is started or stopped
  (a reset that stopped the runner says so); a surrogate training job;
  a JAX trace; frames a stream has already sent (the streams are sent the
  restored frame); and anything under `/cloud`.

## Concurrency, limits and shutdown

FastAPI runs the routes on a thread pool, so requests arrive at the same
time.  **One lock serialises every use of the graph**: each route that
reads or writes its state, parameters or structure, the runner's every
step, a surrogate job's data sweep, and `/sim/run` slice by slice.  N
concurrent `POST /sim/step` take N steps.  Until 0.4.0 they raced: two
steps read the same state and one result overwrote the other, every reply
200.  A request waits at most `_GRAPH_LOCK_TIMEOUT` (30 s) for the lock and
then answers 503; a read waits for the step in flight.

While the runner runs, or a `/sim/run` is in progress, the routes that
write the state or the structure -- `/sim/step`, `/sim/run`,
`PUT /graph/state`, `/checkpoint/load`, adding or removing a node or an
edge, `/sim/profile` -- answer 409.  `PUT /graph/params` still reaches a
running graph, between two steps.

Memory: a request body over `MAX_REQUEST_BODY_BYTES` (32 MiB) is a 413
before it is parsed; `PUT /graph/state` counts each field's values against
the live field before converting them; the whole graph's state is held to
`MAX_GRAPH_STATE_ELEMENTS` (10^8) as well as each node's to
`MAX_NODE_STATE_ELEMENTS`; and nodes are built one at a time.

The runner routes (`/sim/start`, `/sim/stop`, `/sim/pause`,
`/sim/resume`, `/sim/reset`, surrogate activate and deactivate) answer
within about one `_GRAPH_LOCK_TIMEOUT` of their arrival, a 503 past it: the
runner's own lock is taken with a deadline, and a reset or a surrogate swap
lets go of it before waiting for the graph.  Start, stop, pause, resume and
reset count that deadline from the request's arrival and run on small
thread pools of their own (`PUT /sim/stride` on the event loop), so they
answer while every worker thread of the shared pool is waiting for the
graph; they used to queue for a worker first (a stop took 5.6 s behind 240
queued reads at a 1 s timeout).  The surrogate routes are experimental and
still run on the shared pool.  A stop or reset whose runner thread will not
stop in time answers 503 saying the runner was told to stop and stays
stopped, with `was_running`.

Shutdown: the app's lifespan, and a SIGINT or SIGTERM when it is served
from the main thread, tell an in-flight `/sim/run` to stop at its next
slice and a training job at its next epoch, then stop the runner.  The
signal is chained ahead of uvicorn's own handler, because uvicorn waits
for in-flight requests before it runs the lifespan shutdown.  A server run
in another thread, or stopped by setting `uvicorn.Server.should_exit`,
calls `SimulationServer.request_shutdown()` first.

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
