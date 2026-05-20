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

```{note}
`gm.compile()` prints advisory notices about graph wiring — a lone node
reports as "disconnected", and a feedback loop reports a detected cycle
(back-edges are staggered to the previous timestep).  They are advisory,
not errors: a standalone single-node graph and an intentional coupling
loop both run correctly.
```

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
```

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

## Next Steps

- **[Installation Guide](installation.md)** — all extras, GPU setup, cloud providers
- **[DESIGN.md](../../DESIGN.md)** — architecture decisions, node authoring contract
- **`examples/`** — coupling, adaptive timestepping, surrogates, servers
