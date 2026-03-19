# Quickstart

This guide gets you from zero to a running simulation in under 5 minutes.

## Install

```bash
pip install maddening[viz]    # simulation engine + matplotlib plots
```

For GPU acceleration, use `maddening[cuda12,viz]` instead.

## Your First Simulation

MADDENING simulations are graphs. Nodes simulate physics. Edges couple them.

```python
import jax.numpy as jnp
from maddening import GraphManager, SimulationNode


# 1. Define a node
class BounceNode(SimulationNode):
    """Ball bouncing under gravity."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def initial_state(self):
        return {
            "position": jnp.array(5.0),
            "velocity": jnp.array(0.0),
        }

    def update(self, state, boundary_inputs, dt):
        gravity = -9.81
        new_vel = state["velocity"] + gravity * dt
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

## Coupled Simulation

Connect nodes with edges to exchange data each timestep:

```python
from maddening import GraphManager, EdgeSpec

gm = GraphManager()
gm.add_node(ball_node)       # produces "position"
gm.add_node(spring_node)     # consumes "anchor_position", produces "force"

# Ball position feeds into spring, spring force feeds back
gm.add_edge("ball", "spring", source_field="position", target_field="anchor_position")
gm.add_edge("spring", "ball", source_field="force", target_field="external_force")

gm.compile()
gm.run(n_steps=1000)
```

## Differentiable Everything

The entire graph step is JIT-compiled and differentiable:

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
