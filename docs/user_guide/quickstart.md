# Quickstart

This guide gets you from zero to a running simulation in under 5 minutes.

## Install

```bash
pip install maddening[viz]    # simulation engine + matplotlib plots
```

For GPU acceleration, use `maddening[cuda12,viz]` instead.

## Your First Simulation

MADDENING simulations are graphs. {term}`Nodes <Node>` simulate physics. {term}`Edges <Edge>` couple them.

```python
import jax.numpy as jnp
from maddening import GraphManager, SimulationNode


# 1. Define a node
class BounceNode(SimulationNode):
    """A ball bouncing under gravity, optionally pushed by a force."""

    def initial_state(self):
        return {
            "position": jnp.array(5.0),
            "velocity": jnp.array(0.0),
        }

    def update(self, state, boundary_inputs, dt):
        gravity = -9.81
        # `external_force` arrives over an edge when the ball is coupled
        # (see "Coupled Simulation" below); standalone it defaults to 0.
        external_force = boundary_inputs.get("external_force", 0.0)
        new_vel = state["velocity"] + (gravity + external_force) * dt
        new_pos = state["position"] + new_vel * dt
        # Bounce off the floor
        new_vel = jnp.where(new_pos < 0, jnp.abs(new_vel) * 0.8, new_vel)
        new_pos = jnp.maximum(new_pos, 0.0)
        return {"position": new_pos, "velocity": new_vel}


# 2. Build the graph
gm = GraphManager()
gm.add_node(BounceNode(name="ball", timestep=0.01))
gm.compile()

# 3. Run
final_state, history = gm.run_scan_with_history(n_steps=500)

# 4. Plot
import matplotlib.pyplot as plt
positions = history["ball"]["position"]
plt.plot(positions)
plt.xlabel("Step")
plt.ylabel("Height")
plt.title("Bouncing Ball")
plt.savefig("bounce.png")
print(f"Final height: {float(final_state['ball']['position']):.3f}")
```

`gm.compile()` is quiet on a clean single-node graph.  The multi-node
advisories (disconnected nodes, cycle detection) are described under
{ref}`quickstart:Coupled Simulation` below.

## Coupled Simulation

Connect nodes with {term}`edges <Edge>` to exchange data every timestep.
Here a spring reacts to the ball's height and pushes back on it — the
ball drives the spring, and the spring's force feeds into the ball:

```python
import jax.numpy as jnp
from maddening import GraphManager, SimulationNode


# A second node: a spring that pulls the ball toward a rest height.
class SpringNode(SimulationNode):
    """Restoring force toward a fixed rest height."""

    def __init__(self, stiffness=12.0, rest_height=2.0, **kwargs):
        super().__init__(**kwargs)
        self.stiffness = stiffness
        self.rest_height = rest_height

    def initial_state(self):
        return {"force": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        anchor = boundary_inputs.get("anchor_position", self.rest_height)
        return {"force": -self.stiffness * (anchor - self.rest_height)}


# BounceNode is the node defined in the first example above.
gm = GraphManager()
gm.add_node(BounceNode(name="ball", timestep=0.01))    # produces "position"
gm.add_node(SpringNode(name="spring", timestep=0.01))  # produces "force"

# Ball position feeds the spring; spring force feeds back into the ball.
gm.add_edge("ball", "spring", source_field="position",
            target_field="anchor_position")
gm.add_edge("spring", "ball", source_field="force",
            target_field="external_force")

gm.compile()
final_state, history = gm.run_scan_with_history(n_steps=1000)
print(f"Final ball height: {float(final_state['ball']['position']):.2f}")
```

```{note}
**Advisories on multi-node graphs.**

* A *disconnected* node (no edges or external inputs into the rest
  of the graph) emits a `UserWarning` from `gm.compile()`.
  Disconnected-but-intentional nodes run correctly — the warning is
  advisory, not an error.
* A *feedback loop* is logged at `INFO` level through
  ``logging.getLogger("maddening.core.graph_manager")`` (no
  warning).  Back-edges are staggered to the previous timestep, so
  intentional coupling loops also run correctly.

Both messages are designed to surface unintentional graph mistakes
without breaking deliberate constructions.
```

## Differentiable Everything

The entire {term}`graph step <Graph step>` is JIT-compiled and differentiable:

```python
import jax

# Gradient of final position w.r.t. initial velocity
def loss(initial_velocity):
    gm.set_node_state("ball", {"position": jnp.array(5.0),
                                "velocity": initial_velocity})
    state = gm.run_scan(n_steps=100)
    return state["ball"]["position"]

grad_fn = jax.grad(loss)
print(f"d(final_pos)/d(init_vel) = {grad_fn(jnp.array(0.0))}")

# The loss stepped the graph under a transform, so its state is not one
# you want to keep running from.  Say what it should be next:
gm.reset_state()
```

`run_scan` (like `step` and `run`) writes its result back into the
graph, and under `jax.grad` that result is made of JAX tracers.  The
gradient is correct either way, and the graph puts itself back to the
state it had before the transform — with a `RuntimeWarning` — the next
time you use it.  Setting the state you actually want, as above, is
cheaper than relying on that and clearer to read.

## Deploy to Cloud

Run your simulation on a cloud GPU:

```bash
pip install maddening[runpod]

# Set up credentials (one-time)
mkdir -p ~/.maddening
cp src/maddening/examples/cloud/cloud_credentials.example.yaml ~/.maddening/cloud_credentials.yaml
# Edit ~/.maddening/cloud_credentials.yaml with your RunPod API key
# You can also choose a different path, in which case use the `creds` argument in the CloudLauncher constructor.
```

```python
from maddening.cloud.launcher import CloudLauncher

launcher = CloudLauncher() 
# launcher = CloudLauncher(credentials_path='path/to/cloud_credentials.yaml')
info = launcher.validate("job_config.yaml")
print(f"Instance: {info['instance_type']}, ${info['hourly_cost']:.2f}/hr")

job = launcher.launch("job_config.yaml")
job.stream_logs()
print(f"VM IP: {job.vm_ip}")
job.teardown()
```

See `src/maddening/examples/cloud/` for complete examples and config templates.

```{warning}
**The simulation API requires a bearer token unless it is bound to
loopback, and it still has no TLS.**  Running locally on `127.0.0.1`
nothing changes.  The cloud image binds `0.0.0.0`, so every route there
— including `POST /cloud/launch` and `POST /cloud/teardown`, which
provision and destroy paid GPU instances using the credentials stored on
that host — needs `Authorization: Bearer <token>`.  Set
`MADDENING_API_TOKEN` in your job's `envs:`, or read the token the server
generates and logs once at startup.  A blank `MADDENING_API_TOKEN` is a
configuration error and the server refuses to start.

`JobConfig.ports` no longer defaults to `[8000]`, so the provider's
firewall stays shut unless you ask for the port.  Reach the API through
an SSH tunnel:

    ssh -L 8000:127.0.0.1:8000 root@<vm-ip> -p <ssh-port>
    # then talk to http://localhost:8000

Because there is no TLS, the token crosses the network in cleartext:
prefer the tunnel, or a TLS-terminating reverse proxy, to a public port.
`/docs` is not served when the token is enforced (Swagger UI cannot send
a bearer header when it fetches its own schema).  WebRTC signaling on
8443 authenticates separately, once you set `MADDENING_STREAM_SECRET` and
share it with the viewer.

**The ZeroMQ transports follow the same rule with the same token.**
`NetworkRelay` (state, 5555), `CommandPublisher` (commands, 5556) and the
multi-job `Coordinator` (5580) bind **loopback** by default, so local work
needs no configuration at all:

    relay = NetworkRelay()            # tcp://127.0.0.1:5555, no token
    receiver = NetworkReceiver()      # tcp://localhost:5555

To watch a remote simulation, tunnel to its loopback port — still no
token, because both ends are loopback:

    ssh -L 5555:127.0.0.1:5555 root@<vm-ip> -p <ssh-port>

To publish the port instead, bind a non-loopback address. That turns on
**ZMQ CURVE encryption**, and the socket refuses to open without
`MADDENING_API_TOKEN`:

    # both machines
    export MADDENING_API_TOKEN=<the same secret>
    relay = NetworkRelay("tcp://0.0.0.0:5555")
    receiver = NetworkReceiver("tcp://<vm-ip>:5555")

There are **no key files**: both CURVE keypairs are derived from that one
token. Unlike the API, ZMQ traffic *is* encrypted once it leaves
loopback. Port 5556 is an actuation path — its payload reaches
`GraphManager.step(external_inputs=...)` — so publish it deliberately.

One asymmetry to know: a client decides whether to use CURVE from the
address *it* connects to, but the server's posture comes from the address
*it* bound. If you reach a coordinator bound to `0.0.0.0` over
`127.0.0.1` (rank 0's own worker, or a tunnel), pass `secure=True`
explicitly.
```

## Next Steps

- **[Installation Guide](installation.md)** — all extras, GPU setup, cloud providers
- **[DESIGN.md](https://github.com/Microrobotics-Simulation-Framework/MADDENING/blob/main/DESIGN.md)** — architecture decisions, node authoring contract
- **`examples/`** — coupling, adaptive timestepping, surrogates, servers
