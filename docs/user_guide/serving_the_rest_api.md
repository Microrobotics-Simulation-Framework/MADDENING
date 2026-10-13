# Serving the REST API on a network

`SimulationServer` puts a graph behind an HTTP and WebSocket API.  Bound
to `127.0.0.1` it needs no configuration: a direct connection from this
machine is served without a credential.  This page is for the other case,
a server another machine reaches over the network.  Such a server asks
every request for a bearer token.  The steps below choose the bind
address, set the token, confine {term}`checkpoints <Checkpoint>` to one directory, call the
server with the token, and test it in-process.

```{note}
**Prefer an SSH tunnel when you can.**  The API has no TLS: on a network
bind the token and every state snapshot cross the wire in cleartext.  A
server that keeps its loopback bind and is reached through
`ssh -L 8000:127.0.0.1:8000 user@host` needs no token at all, and nothing
it sends leaves the tunnel.  Put a network bind on a private network or
behind a TLS-terminating proxy.
```

Install the server's dependencies with `pip install "maddening[api]"`.

## How the server decides

The server applies one of two rules, chosen by the address you tell it
it is bound to:

- **A loopback bind** (`127.0.0.1`, `localhost`, `::1`) serves a direct
  connection from a loopback address without a token.  A request from any
  other peer, or one carrying `X-Forwarded-For` or `Forwarded`, must
  present the token, and a request without a valid token is refused (403)
  unless its `Host` names this machine or one of `allowed_hosts`.
- **Any other bind** (`0.0.0.0`, a LAN or public address) asks every
  request for `Authorization: Bearer <token>`, except `/healthz` and the
  bundled `/viz/*` pages, which hold no information about the graph.

The application cannot see the socket uvicorn opens, so you tell it:
`SimulationServer(..., bind_host=host)`, or set `MADDENING_HOST`.  Left
unset, it assumes `127.0.0.1`.  A server that was told loopback but is
reached from a routable address still asks that peer for the token, so
forgetting `bind_host` costs a 401, not an open API.

## The token

The token is `MADDENING_API_TOKEN` when that variable is set.  Otherwise
the server generates one at start-up (32 random bytes, 43 URL-safe
characters), and on a network bind `server.auth.announce(port)` logs it
once.  A token that
is set but blank is refused, as is one with whitespace around it or a
character outside printable ASCII: no client could present it.

```python
from maddening.api.auth import APIAuth

# MADDENING_API_TOKEN set (read from the mapping given; os.environ by default)
fake_token = "-".join(("example", "token", "not", "a", "secret"))
auth = APIAuth(bind_host="0.0.0.0", environ={"MADDENING_API_TOKEN": fake_token})
assert auth.enforced and not auth.generated and auth.token == fake_token

# Not set: generated at start-up
auth = APIAuth(bind_host="0.0.0.0", environ={})
assert auth.generated and len(auth.token) == 43

# Set but blank: a configuration error, not "authentication off"
try:
    APIAuth(bind_host="0.0.0.0", environ={"MADDENING_API_TOKEN": ""})
except ValueError as err:
    print(str(err).split(".")[0])
```

```text
MADDENING_API_TOKEN is set but blank
```

If nothing will read the server's log (a detached container, a batch
job), set `MADDENING_API_TOKEN` yourself, or set `MADDENING_API_TOKEN_FILE`
to a path: the generated token is written there, readable by its owner
only (mode `0600`), when it is announced.

<!-- snippet: continues -->
```python
import logging
import os
import stat

logging.getLogger("maddening.api.auth").setLevel(logging.ERROR)  # keep it out of this page
auth = APIAuth(bind_host="0.0.0.0", environ={"MADDENING_API_TOKEN_FILE": "api-token"})
assert auth.announce(8000)
with open("api-token") as fh:
    assert fh.read().strip() == auth.token
assert stat.S_IMODE(os.stat("api-token").st_mode) == 0o600
```

## A server script

This script serves a graph on every interface.  It tells the server its
bind address, keeps checkpoints under one directory, logs a generated
token once, and turns off uvicorn's proxy headers, so that no forwarded
header can stand in for the peer address:

<!-- snippet: no-run, reason: network: binds every interface and serves until stopped; serve_with_token runs the same steps on loopback -->
```python
# serve.py
import os

import uvicorn

from maddening.api.server import SimulationServer, warn_if_publicly_bound
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, TableNode

HOST = os.environ.get("MADDENING_HOST", "0.0.0.0")
PORT = 8000

gm = GraphManager()
gm.add_node(TableNode("table", 0.01, position=0.0))
gm.add_node(BallNode("ball", 0.01, initial_position=2.0))
gm.add_edge("table", "ball", source_field="position", target_field="table_position")
gm.compile()

server = SimulationServer(
    node_registry={"BallNode": BallNode, "TableNode": TableNode},
    graph_manager=gm,
    bind_host=HOST,                                # switches the token on
    checkpoint_root="/srv/maddening/checkpoints",  # every checkpoint path is under it
)
server.auth.announce(PORT)          # logs a generated token once; says nothing of a set one
warn_if_publicly_bound(HOST, PORT)  # logs what the bind exposes
uvicorn.run(server.create_app(), host=HOST, port=PORT, proxy_headers=False)
```

Start it with a token you chose, so that you know it without reading the
log:

```bash
export MADDENING_API_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python serve.py
```

`python -m maddening.examples.servers.serve_with_token` runs the same
steps end to end without publishing anything: uvicorn in its own process,
on a loopback port, with the server told `bind_host="0.0.0.0"`.  It shows a
request refused without the token and served with it, a checkpoint path
refused outside the root, and a generated token logged once.

## Calling it

Every route but `/healthz` and `/viz/*` takes the token in an
`Authorization` header:

```bash
curl http://sim-host:8000/healthz                       # no token needed
curl -H "Authorization: Bearer $MADDENING_API_TOKEN" http://sim-host:8000/graph
curl -X POST -H "Authorization: Bearer $MADDENING_API_TOKEN" \
     "http://sim-host:8000/sim/run?n_steps=100"
```

From Python, send the same header with any HTTP client.  The block below
runs the server in-process with FastAPI's `TestClient`, which needs no
port; against a real server, give the same `headers=` to `httpx.Client`
or `requests`:

<!-- snippet: requires: fastapi httpx -->
```python
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, TableNode

gm = GraphManager()
gm.add_node(TableNode("table", 0.01, position=0.0))
gm.add_node(BallNode("ball", 0.01, initial_position=2.0))
gm.add_edge("table", "ball", source_field="position", target_field="table_position")
gm.compile()

fake_token = "-".join(("example", "token", "not", "a", "secret"))
server = SimulationServer(node_registry={"BallNode": BallNode, "TableNode": TableNode},
                          graph_manager=gm, bind_host="0.0.0.0", api_token=fake_token,
                          checkpoint_root="checkpoints")
app = server.create_app()

anonymous = TestClient(app)
assert anonymous.get("/healthz").status_code == 200
reply = anonymous.get("/graph")
print(reply.status_code, reply.json()["detail"])

client = TestClient(app, headers={"Authorization": f"Bearer {fake_token}"})
assert client.get("/graph").status_code == 200
assert client.post("/sim/run?n_steps=10").status_code == 200
```

```text
401 This server is bound to a non-loopback address, so every route requires 'Authorization: Bearer <token>'. The token is $MADDENING_API_TOKEN, or was logged once at start-up.
```

`api_token=` overrides `MADDENING_API_TOKEN`; a deployment usually sets
the variable instead.  The other clients:

| Client | How it presents the token |
|---|---|
| WebSocket client that can set headers | `Authorization: Bearer <token>` on the handshake |
| Browser WebSocket | the subprotocols `maddening.api.auth.websocket_credentials(token)` returns, `["maddening.bearer.<base64url(token)>", "maddening.v1"]` |
| Bundled pages | open `http://sim-host:8000/viz/app#token=<token>`; the fragment never reaches the server's access log |

No API route accepts the token in a query string.  `/docs` and
`/openapi.json` are not served while the token is enforced; read them
from a loopback-bound instance.

## Checkpoints stay under `checkpoint_root`

`POST /checkpoint/save?path=...` and `/checkpoint/load` take a path
relative to `checkpoint_root`, and refuse one that leaves it, whether or
not the token is enforced.  The root defaults to `checkpoints` under the
working directory the server was started in; set it explicitly in a
deployment.

<!-- snippet: continues -->
```python
from pathlib import Path

assert client.post("/checkpoint/save?path=run1.npz").status_code == 200
assert (Path("checkpoints") / "run1.npz").is_file()

reply = client.post("/checkpoint/save?path=../run1.npz")
print(reply.status_code, reply.json()["detail"])
assert not Path("run1.npz").exists()
```

```text
400 checkpoint path must stay under the checkpoint root, naming a file inside it (got '../run1.npz')
```

## Testing a server in-process

An in-process `TestClient` is not a loopback connection: it reports its
peer as `"testclient"`.  So even a loopback-bound server asks it for the
token, and `TestClient(app)` alone gets a 401.  Give it the token, or
build it as a direct loopback client:

<!-- snippet: continues -->
```python
local = SimulationServer(node_registry={"BallNode": BallNode, "TableNode": TableNode},
                         checkpoint_root="checkpoints")         # loopback by default
local_app = local.create_app()
assert TestClient(local_app).get("/graph").status_code == 401

with_token = TestClient(local_app,
                        headers={"Authorization": f"Bearer {local.auth.token}"})
as_loopback = TestClient(local_app, base_url="http://127.0.0.1",
                         client=("127.0.0.1", 50000))
assert with_token.get("/graph").status_code == 200
assert as_loopback.get("/graph").status_code == 200
```

## On a cloud instance

A cloud job opens no API port by default (`JobConfig.ports` is `[]`), so
reach its API through a tunnel, `ssh -L 8000:127.0.0.1:8000 root@<vm-ip>
-p <ssh-port>`, and talk to `http://localhost:8000`.  To publish the port
instead, set `ports=[8000]` and `MADDENING_API_TOKEN` in the job's
environment.  The [quickstart](quickstart.md#deploy-to-cloud) says more.

## What the token does not do

- It does not encrypt anything; see the note at the top.
- It does not limit what its holder can do.  Whoever holds the token can
  build {term}`nodes <Node>`, run steps and write files under the checkpoint root, as
  someone at the keyboard could.  Do not hand it to a party you would not
  let run code on the host.
- Request sizes are bounded, but that is a backstop against mistakes, not
  a defence against a token holder who sets out to exhaust the host.

The API's own reference, `src/maddening/api/README.md` in the
[repository](https://github.com/Microrobotics-Simulation-Framework/MADDENING/blob/main/src/maddening/api/README.md),
lists every route, the limits, and what the server does and does not
defend against.  The ZeroMQ state and command streams have a token of
their own: see [ZeroMQ transports](zeromq_transports.md).
