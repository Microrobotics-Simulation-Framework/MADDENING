# maddening.examples

All examples assume you are in the project root with the venv activated:

```bash
cd /home/nick/MSF/MADDENING
source ../venvs/.maddening/bin/activate
```

## Examples

| Script | Description | GUI needed? |
|--------|-------------|-------------|
| `bouncing_ball.py` | Headless batch demo: JIT-compiled sim, serialization round-trip, `jax.grad` through the step function. Saves a plot to `bouncing_ball_result.png`. | No |
| `bouncing_ball_scene.py` | 2D animated scene + time-series plots via matplotlib. | Yes |
| `bouncing_ball_terminal.py` | Live terminal monitor using `rich`. Works over SSH. Ctrl-C to stop. | No |
| `bouncing_ball_combined.py` | All three renderers simultaneously: matplotlib scene, time-series, and terminal monitor on a background thread. | Yes |
| `remote_sim_server.py` | Headless sim publishing state over ZMQ. Run on HPC/compute node. | No |
| `remote_viz_client.py` | Connects to a remote sim server and renders (terminal or matplotlib). Run on local workstation. | Depends on `--mode` |
| `api_server.py` | FastAPI server with a pre-loaded ball + spring graph. Interactive docs at `/docs`. | No |
| `coupled_spring_ball.py` | Two-way coupled ball + spring-damper system. Demonstrates multi-physics coupling with scan history and plots. | No |
| `differentiable_optimization.py` | Uses `jax.grad` through the full graph to optimize initial velocity via gradient descent. | No |
| `scan_performance.py` | Benchmarks `run()` vs `run_scan()` vs `run_scan_with_history()`. Shows ~9x speedup. | No |
| `multirate_demo.py` | Multi-rate scheduling: fast ball at 1kHz, slow spring at 100Hz. Verifies staircase update pattern. | No |
| `external_inputs_demo.py` | Time-varying external force injection. Compares free, constant, and sinusoidal forcing. | No |

## Running

```bash
# Headless batch
python maddening/examples/bouncing_ball.py

# Live matplotlib visualization
python maddening/examples/bouncing_ball_scene.py

# Terminal-only (SSH-friendly)
python maddening/examples/bouncing_ball_terminal.py

# All renderers at once
python maddening/examples/bouncing_ball_combined.py

# Remote visualization (two terminals)
python maddening/examples/remote_sim_server.py          # terminal 1
python maddening/examples/remote_viz_client.py --mode scene  # terminal 2

# API server
pip install fastapi uvicorn
python maddening/examples/api_server.py
# Then: curl http://localhost:8000/docs

# New feature demos
python maddening/examples/coupled_spring_ball.py
python maddening/examples/differentiable_optimization.py
python maddening/examples/scan_performance.py
python maddening/examples/multirate_demo.py
python maddening/examples/external_inputs_demo.py
```

## SSH Tunnel for Remote Viz

```bash
# On your local machine:
ssh -L 5555:localhost:5555 user@hpc-node

# On the HPC node:
python maddening/examples/remote_sim_server.py

# On your local machine (another terminal):
python maddening/examples/remote_viz_client.py
```
