# maddening.examples

Every example runs as a module against an installed `maddening`, from any
working directory.  Scripts that save a plot, a USD file or a profile write
it into the current working directory.

```bash
pip install maddening            # or, from a checkout: pip install -e .
python -m maddening.examples.basics.bouncing_ball
```

Examples that take a size argument (`--steps`, `--epochs`, `--frames`, ...)
keep a default meant for a person watching; the test suite runs them small.
Optional dependencies are named per row: `[api]` is `pip install
"maddening[api]"`, and so on.

## Run it locally first

The client/server examples all run on one machine with no setup:

```bash
# Simulation server + viewer, one command: starts the server as a
# subprocess on a free loopback port and stops it when you exit
python -m maddening.examples.servers.remote_viz_client --local

# Any HTTP example server on a free port (it prints the address)
python -m maddening.examples.servers.api_server --port 0

# The cloud examples' server and rendezvous checks, without a cloud account
python -m maddening.examples.cloud.server.04_server_test --local
python -m maddening.examples.cloud.server.05_websocket_test --local
python -m maddening.examples.cloud.multijob.08_two_vm_test --local
```

The servers bind 127.0.0.1 only.  To reach one on another machine (an HPC
node, a cloud VM), forward its port over SSH instead of binding a public
address: `ssh -L 8000:127.0.0.1:8000 user@host` for HTTP, or
`ssh -L 5555:localhost:5555 user@hpc-node` for the ZMQ simulation stream
(then run `remote_viz_client` without `--local`).

## basics/

| Script | What it shows | Run |
|---|---|---|
| `bouncing_ball.py` | Graph build, JIT, `to_dict`/`from_dict` round trip, `jax.grad` through steps; saves a plot | `python -m maddening.examples.basics.bouncing_ball` |
| `heat_diffusion_demo.py` | 1-D heat equation with Dirichlet ends, checked against the steady profile | `... basics.heat_diffusion_demo` |
| `rigid_body_demo.py` | `RigidBodyNode` held in a plane by DOF constraints, checked against the analytic projectile and spin | `... basics.rigid_body_demo` |
| `bouncing_ball_terminal.py` | Live terminal table (`rich`); works over SSH | `... basics.bouncing_ball_terminal [--duration 5]` |
| `bouncing_ball_scene.py` | Animated matplotlib scene + time series (needs a display) | `... basics.bouncing_ball_scene` |
| `bouncing_ball_combined.py` | Three renderers on one relay (needs a display) | `... basics.bouncing_ball_combined` |

## coupling/

| Script | What it shows | Run |
|---|---|---|
| `coupling_demo.py` | Staggered vs Gauss-Seidel vs `auto_couple` on two masses joined by a spring; staggering's lag shifts the centre of mass | `... coupling.coupling_demo` |
| `coupled_spring_ball.py` | A one-way chain table -> ball -> spring with `run_scan_with_history`; saves a plot | `... coupling.coupled_spring_ball` |
| `acceleration_comparison.py` | Plain vs Aitken vs under-relaxation vs IQN-ILS, weak and strong coupling (IQN-ILS converges where plain diverges) | `... coupling.acceleration_comparison [--steps N]` |
| `convergence_diagnostics_demo.py` | `coupling_diagnostics()`: iterations, `converged`, starved budgets, the three norms, the spectral keys | `... coupling.convergence_diagnostics_demo [--steps N]` |
| `jacobi_vs_gauss_seidel.py` | Iteration modes on a 3-node cycle: same fixed point, different paths when under-converged | `... coupling.jacobi_vs_gauss_seidel [--steps N]` |
| `subcycling_demo.py` | Mixed-timestep coupling group vs a coarse and a fine reference | `... coupling.subcycling_demo` |
| `spatial_interpolation_demo.py` | Interface maps (nearest, linear, RBF, conservative) and two coupled rods at different resolutions | `... coupling.spatial_interpolation_demo [--heat-steps N]` |
| `interface_mapping_demo.py` | `add_edge(mapping=)` between an 8- and a 24-cell rod: weights in `params["mappings"]`, the patch test, a `to_dict` / `from_dict` round trip that steps identically, and the refusal of a write that would move the mapped points (MADD-ANO-063) | `... coupling.interface_mapping_demo [--steps N]` |
| `flux_coupling_demo.py` | Heat-rod value coupling, flux conservation, additive inputs, IQN-IMVJ, interface norm, sub-cycling options | `... coupling.flux_coupling_demo [--sections 1,3]` |
| `vessel_bifurcation.py` | Y-junction of three heat rods built from and written back to USD `[usd]` | `... coupling.vessel_bifurcation [--steps N] [--viz]` |
| `vessel_bifurcation_live.py` | The same, live in a PyVista window with a heat pulse `[usd, viz3d]`, needs a display | `... coupling.vessel_bifurcation_live` |
| `vessel_flow_helpers.py` | Helper: builds the heart-pump + LBM graph `servers/vessel_flow_server.py` serves | (imported) |

## advanced/

| Script | What it shows | Run |
|---|---|---|
| `multirate_demo.py` | Nodes at 1 kHz and 100 Hz in one graph; saves a plot | `... advanced.multirate_demo` |
| `adaptive_demo.py` | `run_adaptive` (step doubling + PI controller) at three tolerances | `... advanced.adaptive_demo` |
| `external_inputs_demo.py` | Time-varying external force through `add_external_input`; saves a plot | `... advanced.external_inputs_demo` |
| `differentiable_optimization.py` | Gradient descent on an initial velocity with `jax.grad` through a scanned step; saves a plot | `... advanced.differentiable_optimization` |
| `parameter_sweep_demo.py` | 20 initial heights in one `run_sweep` (`jax.vmap`); saves a plot | `... advanced.parameter_sweep_demo` |
| `scan_performance.py` | `run()` vs `run_scan()` vs `run_scan_with_history()` timings (compile included); saves a plot | `... advanced.scan_performance [--steps N]` |
| `surrogate_demo.py` | Train an MLP surrogate and swap it into the graph `[surrogates]` | `... advanced.surrogate_demo [--epochs N]` |
| `profile_lbm_step.py` | `profile_graph` on a two-rod graph, saved as Perfetto JSON | `... advanced.profile_lbm_step [--n-steps N]` |
| `profiling_demo.py` | The whole `ProfileReport` on a coupled pair: measured coupling overhead and cost per iteration, bottleneck, compile counts (checked identical on a fresh graph), a `trace=True` summary, Perfetto JSON; all output in a temporary directory | `... advanced.profiling_demo [--n-cells N] [--out-dir DIR]` |
| `sysid_demo.py` | Calibration: `fim` finds the spring's scale degeneracy, `fit` holds it (`excited_rank`, `hold_declined`), `fit_lm` recovers k and c under a `ParamSpec` freeze and a mask, Cramér–Rao bounds, `params_table()` before and after | `... advanced.sysid_demo [--samples N] [--n-iter N]` |
| `checkpoint_resume_demo.py` | `save_state` / `load_state`: a coupled run resumed in a fresh graph is bitwise identical, `_meta` warm starts and calibrated params included; a cold restart is not | `... advanced.checkpoint_resume_demo [--warmup N] [--steps N]` |
| `live_stage_bouncing_ball_demo.py` | `LiveStage` writing a time-sampled USD stage `[usd]` | `... advanced.live_stage_bouncing_ball_demo [--steps N]` |

## servers/

| Script | What it shows | Run |
|---|---|---|
| `remote_viz_client.py` + `remote_sim_server.py` | A simulation streaming state over ZMQ to a viewer (terminal, plain text or matplotlib) `[network]` | `... servers.remote_viz_client --local` |
| `api_server.py` | The REST/WebSocket API (`SimulationServer`), docs at `/docs` `[api]` | `... servers.api_server [--port 0]` |
| `interactive_graph_server.py` | Graph topology page at `/viz/graph` `[api]` | `... servers.interactive_graph_server [--port 0]` |
| `launch_app.py` | Interactive demo app at `/viz/app`; opens a browser `[api]` | `... servers.launch_app [--port 0] [--no-browser]` |
| `launch_server_render.py` | Server-side matplotlib frames streamed to `/viz/render` `[api, viz]` | `... servers.launch_server_render [--port 0]` |
| `lbm_pipe_server.py` | 3-D LBM pipe rendered server-side with VTK `[api, viz3d]` | `... servers.lbm_pipe_server [--port 0] [--grid 24 12 12]` |
| `vessel_flow_server.py` | Heart pump + LBM vessel with live parameter control `[api]` | `... servers.vessel_flow_server [--port 0] [--grid 32 16 16]` |
| `lbm_pipe_interactive.py` | 3-D LBM pipe in an interactive PyVista window `[viz3d]` | `... servers.lbm_pipe_interactive [--frames 20 --screenshot pipe.png]` |
| `lbm_pipe_replay.py` | Simulate, then replay in a GPU viewer (pygfx), needs a display | `... servers.lbm_pipe_replay` |

`remote_sim_server.py` binds `tcp://127.0.0.1:5555`.  A non-loopback
`--bind` turns on ZMQ CURVE encryption and needs the same token on both
sides (`MADDENING_TRANSPORT_TOKEN`, falling back to `MADDENING_API_TOKEN`).

## cloud/

Scripts that provision billable VMs on RunPod, Lambda, AWS or GCP; see
[`cloud/README.md`](cloud/README.md).  Three of them have a `--local` mode
(above) that runs their payload on this machine, and
`cloud/streaming/08_subscribe_lbm_velocity.py` runs entirely locally.
