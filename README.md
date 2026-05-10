# MADDENING

**M**odular **A**utomatic **D**ifferentiation and **D**ata-**E**nhanced **N**eural-network **IN**teracting **G**raph

📖 **Documentation: <https://microrobotica.org/maddening/>**
🧩 Part of the [Microrobotics Simulation Framework](https://microrobotica.org/) (MADDENING · MIME · MICROROBOTICA).

A JAX-based modular simulation framework for multi-physics. Designed as the computational backbone of [MIME](https://microrobotica.org/mime/) (MIcrorobotics Multiphysics Engine).

> **Regulatory disclaimer**: MADDENING is research software. It is not a medical device as defined by EU MDR (EU 2017/745) and is not intended for clinical use. When used in a regulated product, MADDENING is classified as SOUP (Software of Unknown Provenance) under IEC 62304. See `docs/regulatory/` for details.

## What It Does

MADDENING manages a **simulation graph** where each node simulates one aspect of a physical system (fluid dynamics, rigid body mechanics, heat transfer, etc.) and edges represent coupling between them. The entire graph step is JIT-compiled into a single XLA computation via JAX, making the simulation fully differentiable and GPU-accelerated.

**Core capabilities:**
- **Graph-based multi-physics** — nodes are independent JAX programs coupled by typed edges
- **Fully differentiable** — `jax.grad` through entire coupled simulations (verified through 1000-step rollouts)
- **Iterative coupling** — Gauss-Seidel and Jacobi with convergence acceleration (Aitken, IQN-ILS, IQN-IMVJ)
- **Multi-rate timestepping** — each node at its own timestep, GCD-based base rate
- **Adaptive timestepping** — Richardson extrapolation with PI controller
- **Neural surrogates** — train MLP/DeepONet/FNO surrogates from simulation data, hot-swap into the graph
- **Cloud deployment** — provision GPU VMs via SkyPilot, stream rendered viewports via WebRTC
- **Multi-job** — distribute subgraphs across VMs with ZMQ-based rendezvous coordination

**Package structure:**
```
maddening/
├── core/                 # Graph manager, node ABC, edge spec, schedule
│   ├── coupling/         # Iterative coupling, convergence, acceleration, spatial mapping
│   ├── simulation/       # Adaptive dt, checkpoint, integrators, calibration, profiler
│   └── compliance/       # Metadata, stability, anomaly tracking, audit, UQ
├── nodes/                # Built-in nodes (Ball, Table, Spring, Heat, LBM, HeartPump, RigidBody)
├── surrogates/           # Neural surrogate framework (architectures, training, validation)
├── cloud/                # Cloud orchestration (SkyPilot, providers, coordinator, streaming)
│   └── multigpu/         # Device mesh, partitioning, sharded nodes, coordinator
├── viz/                  # Visualization (renderer ABC, relay, runner, backends)
├── api/                  # FastAPI server (REST + WebSocket + server-side rendering)
└── usd/                  # OpenUSD integration (graph serialization, geometry)
```

## Installation

```bash
pip install maddening                    # CPU (base)
pip install maddening[cuda12]            # GPU with CUDA 12
pip install maddening[cuda12,viz]        # GPU + matplotlib plots
pip install maddening[server,cuda12]     # GPU simulation server
pip install maddening[runpod]            # Cloud deploy to RunPod
```

| Extra | What it adds |
|-------|-------------|
| `cuda12` / `tpu` | GPU / TPU acceleration |
| `viz` | Matplotlib renderers |
| `terminal` | Rich terminal renderer |
| `api` | FastAPI HTTP/WS server |
| `network` | ZeroMQ remote transport |
| `surrogates` | Neural surrogate training (equinox + optax) |
| `runpod` / `lambda` / `aws` / `gcp` | Cloud deploy via SkyPilot |
| `server` | Bundle: FastAPI + ZMQ + rich + matplotlib |
| `cloud` | All supported cloud providers |
| `all` | Everything |

Works with both pip and [uv](https://docs.astral.sh/uv/). See [docs/user_guide/installation.md](docs/user_guide/installation.md) for the full guide.

## Quick Start

```python
import jax.numpy as jnp
from maddening import GraphManager, SimulationNode

class BounceNode(SimulationNode):
    @property
    def requires_halo(self) -> bool:
        return False  # pointwise (no spatial neighbors)

    def initial_state(self):
        return {"position": jnp.array(5.0), "velocity": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        new_vel = state["velocity"] + -9.81 * dt
        new_pos = state["position"] + new_vel * dt
        new_vel = jnp.where(new_pos < 0, jnp.abs(new_vel) * 0.8, new_vel)
        new_pos = jnp.maximum(new_pos, 0.0)
        return {"position": new_pos, "velocity": new_vel}

gm = GraphManager()
gm.add_node(BounceNode(name="ball", timestep=0.01))
gm.compile()
final_state, history = gm.run_scan_with_history(n_steps=500)
```

See [docs/user_guide/quickstart.md](docs/user_guide/quickstart.md) for the full tutorial.

## Cloud Deployment

Pre-built Docker image with JAX CUDA, GStreamer, and all server dependencies:

```bash
pip install maddening[runpod]
# Set up ~/.maddening/cloud_credentials.yaml (see examples/cloud/config/)
```

```python
from maddening.cloud.launcher import CloudLauncher

launcher = CloudLauncher()
job = launcher.launch("job_config.yaml")  # provisions GPU VM
job.ssh_run("python3 my_simulation.py")   # run directly on GPU
job.teardown()
```

Docker image: `ghcr.io/microrobotics-simulation-framework/maddening-cloud:latest`

## For MIME Developers

MADDENING is designed to be extended by MIME. See:
- [docs/user_guide/installation.md](docs/user_guide/installation.md) — install options
- [docs/developer_guide/node_authoring.md](docs/developer_guide/node_authoring.md) — writing custom nodes
- [DESIGN.md](DESIGN.md) — architecture decisions
- [CLOUD_ROADMAP.md](CLOUD_ROADMAP.md) — cloud/multi-GPU/multi-job architecture
- [DOCUMENTATION_ARCHITECTURE.md](DOCUMENTATION_ARCHITECTURE.md) — regulatory compliance structure

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE).
