# maddening.examples

All examples assume you are in the project root with the venv activated:

```bash
cd /home/nick/MSF/MADDENING
source ../venvs/.maddening/bin/activate
```

## Directory Structure

```
examples/
├── basics/           # Getting started: single nodes, simple graphs
├── coupling/         # Multi-physics coupling: acceleration, subcycling, interpolation
├── advanced/         # Power features: adaptive dt, sweeps, surrogates, optimization
└── servers/          # Web servers, remote viz, LBM demos
```

## basics/

| Script | Description | GUI? |
|--------|-------------|------|
| `bouncing_ball.py` | Headless batch: JIT, serialization, `jax.grad`. Saves plot. | No |
| `bouncing_ball_scene.py` | 2D animated scene + time-series via matplotlib. | Yes |
| `bouncing_ball_terminal.py` | Live terminal monitor using `rich`. SSH-friendly. | No |
| `bouncing_ball_combined.py` | All three renderers at once. | Yes |
| `heat_diffusion_demo.py` | 1D heat equation with Dirichlet BCs toward steady state. | No |
| `rigid_body_demo.py` | 2D rigid body with forces and torques. | No |

```bash
python -m maddening.examples.basics.bouncing_ball
python -m maddening.examples.basics.heat_diffusion_demo
```

## coupling/

| Script | Description | GUI? |
|--------|-------------|------|
| `coupling_demo.py` | Staggered vs Gauss-Seidel vs auto_couple comparison. | No |
| `coupled_spring_ball.py` | Two-way coupled ball + spring with scan history. | No |
| `acceleration_comparison.py` | Plain vs Aitken vs fixed vs IQN-ILS iteration counts. | No |
| `jacobi_vs_gauss_seidel.py` | GS vs Jacobi on a 3-node cycle, with/without Aitken. | No |
| `subcycling_demo.py` | Mixed-timestep coupling: stiff (1kHz) + soft (100Hz) springs. | No |
| `spatial_interpolation_demo.py` | Interface mapping: NN, linear, RBF, conservative projection. | No |
| `convergence_diagnostics_demo.py` | Diagnostics, mixed norm, and insufficient-iteration detection. | No |

```bash
python -m maddening.examples.coupling.acceleration_comparison
python -m maddening.examples.coupling.jacobi_vs_gauss_seidel
python -m maddening.examples.coupling.subcycling_demo
python -m maddening.examples.coupling.spatial_interpolation_demo
python -m maddening.examples.coupling.convergence_diagnostics_demo
```

## advanced/

| Script | Description | GUI? |
|--------|-------------|------|
| `multirate_demo.py` | Multi-rate scheduling: 1kHz ball + 100Hz spring. | No |
| `adaptive_demo.py` | Adaptive timestepping with Richardson extrapolation. | No |
| `external_inputs_demo.py` | Time-varying external force injection. | No |
| `differentiable_optimization.py` | `jax.grad` through full graph for gradient descent. | No |
| `parameter_sweep_demo.py` | Batched parameter sweeps via `jax.vmap`. | No |
| `scan_performance.py` | Benchmarks `run()` vs `run_scan()` (~9x speedup). | No |
| `surrogate_demo.py` | Neural surrogate training and graph replacement. | No |

```bash
python -m maddening.examples.advanced.multirate_demo
python -m maddening.examples.advanced.differentiable_optimization
```

## servers/

| Script | Description | GUI? |
|--------|-------------|------|
| `api_server.py` | FastAPI server with interactive docs at `/docs`. | No |
| `interactive_graph_server.py` | Cytoscape.js graph visualization. | No |
| `launch_app.py` | Full interactive demo app (ball+spring+heat). | No |
| `launch_server_render.py` | Server-side matplotlib rendering. | No |
| `lbm_pipe_server.py` | LBM 3D pipe with PyVista rendering. | No |
| `lbm_pipe_interactive.py` | Interactive LBM pipe demo. | No |
| `lbm_pipe_replay.py` | LBM pipe replay viewer. | No |
| `remote_sim_server.py` | Headless sim publishing state over ZMQ. | No |
| `remote_viz_client.py` | Connects to remote sim server. | Depends |

```bash
python -m maddening.examples.servers.api_server
python -m maddening.examples.servers.launch_app
```

## SSH Tunnel for Remote Viz

```bash
# On your local machine:
ssh -L 5555:localhost:5555 user@hpc-node

# On the HPC node:
python -m maddening.examples.servers.remote_sim_server

# On your local machine (another terminal):
python -m maddening.examples.servers.remote_viz_client
```
