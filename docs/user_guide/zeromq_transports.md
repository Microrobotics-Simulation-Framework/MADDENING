# ZeroMQ transports

MADDENING streams a running simulation over ZeroMQ, in both directions:

| Class | Socket | Default address | Carries |
|---|---|---|---|
| `NetworkRelay` | PUB, binds | `tcp://127.0.0.1:5555` | the simulation state, every step |
| `NetworkReceiver` | SUB, connects | `tcp://localhost:5555` | (the viewer's side of the state stream) |
| `CommandPublisher` | PUB, binds | `tcp://127.0.0.1:5556` | commands, which the simulation passes to `GraphManager.step(external_inputs=...)` |
| `CommandReceiver` | SUB, connects | `tcp://localhost:5556` | (the simulation's side of the command stream) |
| `Coordinator` (multi-job) | ROUTER, binds | `127.0.0.1:5580` | worker registrations |

They live in `maddening.viz.network` (the coordinator in
`maddening.cloud.multigpu.coordinator`) and need the `network` extra:
`pip install "maddening[network]"`.

The rule is the HTTP API's: **a loopback address is open and unencrypted;
any other address is encrypted and needs a shared secret.**  This page
shows the loopback default, the SSH tunnel that reaches a remote
simulation without any secret, and the encrypted network bind with
`MADDENING_TRANSPORT_TOKEN`.  The blocks run on loopback, each on a port
the operating system picks, so they can run beside anything else; your
own code would use the defaults above.

## On one machine: the default

Nothing to configure.  A relay attached to a graph publishes each step's
state; a receiver started on the same port gets the latest one:

<!-- snippet: requires: zmq -->
```python
import socket
import time

from maddening import GraphManager
from maddening.nodes import BallNode
from maddening.transport_auth import TransportAuth, TransportAuthError
from maddening.viz.network import (CommandPublisher, CommandReceiver,
                                   NetworkReceiver, NetworkRelay)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def stream(relay_options=None, receiver_options=None, seconds=5.0):
    """Publish a falling ball for up to `seconds`; return (frame, handshake_error)."""
    port = free_port()
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=5.0))
    gm.compile()
    relay = NetworkRelay(f"tcp://127.0.0.1:{port}", **(relay_options or {}))
    receiver = NetworkReceiver(f"tcp://127.0.0.1:{port}", **(receiver_options or {}))
    relay.attach(gm)
    receiver.start()
    frame, deadline = None, time.monotonic() + seconds
    while frame is None and receiver.handshake_error is None \
            and time.monotonic() < deadline:
        gm.step()
        time.sleep(0.01)
        frame = receiver.latest_snapshot()[1]
    error = receiver.handshake_error
    receiver.stop()
    relay.close()
    return frame, error


frame, error = stream()
print("state received:", frame)
assert frame is not None and error is None
```

```text
state received: {'ball': {'position': 4.99901..., 'velocity': -0.09809...}}
```

Commands travel the other way the same way.  A PUB socket drops what it
sends before a subscriber has joined, so a controller sends its latest
command repeatedly, not once:

<!-- snippet: continues -->
```python
port = free_port()
publisher = CommandPublisher(f"tcp://127.0.0.1:{port}")
receiver = CommandReceiver(f"tcp://127.0.0.1:{port}")
receiver.start()
deadline = time.monotonic() + 5.0
while receiver.latest_commands() is None and time.monotonic() < deadline:
    publisher.send({"ball": {"table_position": 0.5}})
    time.sleep(0.01)
commands = receiver.latest_commands()
receiver.stop()
publisher.close()
assert commands == {"ball": {"table_position": 0.5}}
assert not publisher.secure          # loopback: no encryption, no secret
```

`ipc://` and `inproc://` addresses never leave the machine either, and
are treated as loopback.

## A remote simulation, through an SSH tunnel

Keep the defaults on both machines and forward the port.  Both ends of
every socket are then loopback addresses, so no secret is needed, and the
traffic crosses the network inside SSH's encryption:

```bash
# on your machine: forward local 5555 to the simulation host's loopback 5555
ssh -L 5555:127.0.0.1:5555 user@sim-host
# on the simulation host, in that session
python -m maddening.examples.servers.remote_sim_server
# on your machine, in another terminal
python -m maddening.examples.servers.remote_viz_client
```

Forward 5556 as well (`-L 5556:127.0.0.1:5556`) when a controller on your
machine sends commands.  This is the recommended way to reach a remote
simulation.

## A network bind, with `MADDENING_TRANSPORT_TOKEN`

Giving a socket an address another machine can reach (`tcp://*:5555`,
`tcp://0.0.0.0:5555`, a LAN or public address) turns on ZMQ CURVE: the
stream is encrypted, and only a peer holding the same secret can connect.
Set the secret on both machines, then bind and connect:

```bash
# on both machines: the same value
export MADDENING_TRANSPORT_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
# simulation host
python -m maddening.examples.servers.remote_sim_server --bind 'tcp://*:5555'
# viewer
python -m maddening.examples.servers.remote_viz_client --connect tcp://sim-host:5555
```

There are no key files: both CURVE key pairs are derived from that one
secret, on each machine, and the publisher accepts only a client whose key
was derived from it.  Without a secret, a socket on such an address does
not open; it never falls back to cleartext.  Nor can encryption be
switched off there with `secure=False`:

<!-- snippet: continues -->
```python
for options in ({}, {"secure": False}):
    try:
        NetworkRelay(f"tcp://0.0.0.0:{free_port()}", **options)
    except TransportAuthError as err:
        print(str(err).split(". ")[0])
```

```text
This ZeroMQ endpoint is reachable from other hosts, so it is encrypted with ZMQ CURVE, and that needs a shared secret: set MADDENING_TRANSPORT_TOKEN to the same value on both sides
secure=False was passed for 'tcp://0.0.0.0:...', which is reachable from other hosts
```

(The ports in the output vary from run to run.)  Both raise before
anything is bound.  A secret can also be passed in code
as `token=`, and `secure=True` turns CURVE on for a loopback address too.
That shows on one machine what a network bind does: the same secret
connects, another secret does not.

<!-- snippet: continues -->
```python
secret = "-".join(("example", "transport", "secret"))

frame, error = stream({"secure": True, "token": secret},
                      {"secure": True, "token": secret})
assert frame is not None                       # same secret: the stream arrives

frame, error = stream({"secure": True, "token": secret},
                      {"secure": True, "token": "another-secret"})
assert frame is None and error is not None     # refused: nothing ever arrives
print(error.split(". ")[0])
```

```text
No ZeroMQ handshake could be completed with tcp://127.0.0.1:...: the peer refused it, so no data will ever arrive on this socket
```

## Why the transport token is not the API token

When `MADDENING_TRANSPORT_TOKEN` is unset, the transports fall back to
`MADDENING_API_TOKEN`, so a setup with one variable keeps working.  Do not
rely on it where an API port and a ZeroMQ port are both published on a
network.  The HTTP API has no TLS: its token travels in cleartext in the
`Authorization` header of every request.  Anyone who sees one such request
could derive both CURVE key pairs from it and read the "encrypted" state
and command streams.  The CURVE secret itself is never sent.  So use two
values:

```bash
export MADDENING_API_TOKEN="<one secret>"          # HTTP bearer token, sent with every request
export MADDENING_TRANSPORT_TOKEN="<another secret>"  # CURVE seed, never sent
```

An SSH tunnel to loopback ports needs neither variable.  Both ends must
read the secret from the same variable: a relay that reads
`MADDENING_TRANSPORT_TOKEN` and a receiver that falls back to
`MADDENING_API_TOKEN` derive different keys and will not connect.  A
`MADDENING_TRANSPORT_TOKEN` that is set but blank is an error, and does
not fall back:

<!-- snippet: continues -->
```python
both = {"MADDENING_TRANSPORT_TOKEN": "t-value", "MADDENING_API_TOKEN": "a-value"}
assert TransportAuth(environ=both).token_env == "MADDENING_TRANSPORT_TOKEN"
assert TransportAuth(environ={"MADDENING_API_TOKEN": "a-value"}).token_env \
    == "MADDENING_API_TOKEN"                   # the fallback
try:
    TransportAuth(environ={"MADDENING_TRANSPORT_TOKEN": "", "MADDENING_API_TOKEN": "a"})
except TransportAuthError as err:
    print(str(err).split(". ")[0])
```

```text
MADDENING_TRANSPORT_TOKEN is set but blank
```

The secret protects the stream from anyone without it.  It does not make
token holders distrust each other (any holder can derive both key pairs,
so an authorised viewer could also publish), and a recorded session can
be read later by someone who learns the secret.  Where either matters,
keep the loopback bind and use a tunnel.

## When nothing arrives: the address asymmetry

Each end decides whether to use CURVE from its own address.  A publisher
bound to `tcp://0.0.0.0:5555` encrypts; a receiver that reaches it as
`tcp://127.0.0.1:5555`, through a tunnel or a container's published port,
does not, by default.  The handshake then fails, and no frame ever
arrives: `latest_snapshot()` keeps returning `(0.0, None)`, as it would
for an idle simulation.  The receiver logs the cause once and keeps it in
`receiver.handshake_error` (read it after `start()`).  Pass `secure=True`
on the receiver, with the same secret, and it connects:

<!-- snippet: continues -->
```python
encrypting = {"secure": True, "token": secret}   # what a 0.0.0.0 bind does

frame, error = stream(encrypting, {})           # receiver on 127.0.0.1: CURVE off
assert frame is None and "This end has CURVE off" in error

frame, error = stream(encrypting, {"secure": True, "token": secret})
assert frame is not None and error is None
```

`CommandReceiver.handshake_error` reports the same for the command stream,
where a refused handshake is a dead actuation path that looks like a quiet
controller.  For a `Coordinator` bound to `0.0.0.0` that its own workers
reach over `127.0.0.1`, pass `secure=True` on the workers the same way.

## See also

- [Serving the REST API on a network](serving_the_rest_api.md), for the
  HTTP API's token.
- `python -m maddening.examples.servers.remote_viz_client --local`, a
  simulation and a viewer on one machine in one command.
