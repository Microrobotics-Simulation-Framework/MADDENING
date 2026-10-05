# MADDENING SOUP Package Document

The version and date of record for this document are the ones in
[§1 Software Identification](#1-software-identification) below, which are
generated from `pyproject.toml`.  A second copy in a header is a second thing
to forget on a release; there is deliberately no longer one here.

## 1. Software Identification

<!-- BEGIN GENERATED: software-identification -- scripts/generate_soup_tables.py; do not edit by hand -->
| Field | Value |
|---|---|
| Name | MADDENING |
| Full Name | Modular Automatic Differentiation and Data Enhanced Neural-network INteracting Graph |
| Version | 0.4.0.dev0 |
| Release Date | unreleased (development build) |
| Licence | LGPL-3.0-or-later |
| Source Repository | https://github.com/Microrobotics-Simulation-Framework/MADDENING |
| Python Version | >=3.12 permitted; verified on 3.12 (the CI matrix) |
| JAX Version | jax>=0.10,<0.13 permitted; verified at 0.10.2, 0.11.2 (the versions CI installs) |
| Base Dependencies | jax>=0.10,<0.13, jaxlib>=0.10,<0.13, lineax>=0.0.7, numpy>=1.24, pyyaml>=6.0 |
| Build System | hatchling |
| Install | `pip install maddening` |
<!-- END GENERATED: software-identification -->

Optional extras (GPU, server, visualization, USD, …) are listed in
`pyproject.toml` under `[project.optional-dependencies]`; only the base
dependencies above are installed by `pip install maddening`.  FMU export
(`maddening.fmi`) needs no extra.  The resolved dependency tree of the base
install, and of the extras listed in §6, is in the SBOMs described there.

## 2. Functional Description

### Core Capabilities

- **Graph-based multi-physics simulation**: Compose, couple, and run physics simulations as directed graphs of nodes
- **Functional state pattern**: Pure functions, no shared mutable state, explicit data flow
- **{term}`JIT compilation`**: Full simulation step compiled to {term}`XLA` via {term}`JAX`
- **Automatic differentiation**: End-to-end differentiable simulation graphs
- **Neural surrogates**: Train and deploy neural network replacements for physics nodes
- **Adaptive timestepping**: {term}`Richardson extrapolation` with {term}`PI controller`
- **Parameter sweeps**: Batched simulation via `jax.vmap`
- **{term}`Coupling`**: {term}`Gauss-Seidel` iterative coupling via an early-exit `jax.lax.while_loop` with implicit-function-theorem differentiation

### Capabilities NOT Provided

- Clinical decision support
- Patient data processing
- Medical device functionality
- Real-time safety monitoring (HealthCheckNode provides infrastructure only; configuration and response logic are downstream responsibilities)
- Input validation or sanitization (assumes trusted inputs)

### Infrastructure Nodes

- **HealthCheckNode** (`maddening.nodes.health_check`): Base health monitor for execution-layer fault detection (NaN/Inf trapping, physical boundary checks). Downstream libraries instantiate and configure this node. Part of the MADDENING {term}`SOUP` dependency, not a separate SOUP item.

## 3. Known Anomalies

`known_anomalies.yaml` (same directory) is the registry of record.  The table
below is generated from it — it is a reading aid, not a second source, and a
stale copy fails CI rather than shipping.

<!-- BEGIN GENERATED: known-anomalies -- scripts/generate_soup_tables.py; do not edit by hand -->
| ID | Title | Severity | Safety Relevance | Status | Affected Versions |
|---|---|---|---|---|---|
| MADD-ANO-001 | LBM GPU segfault on CUDA 12.2 + jaxlib 0.5.1 | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-002 | HeatNode CFL stability not enforced at runtime | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-003 | AdaptiveNode frozen-set gradient omits a first-order term at active-set switches | `major` | `context_dependent` | `open` | >=0.4.0.dev0 |
| MADD-ANO-004 | The sharded wrappers ignored a REST parameter write; PUT /graph/params answered 200 without changing the physics | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-005 | A coupling group's converged flag was a residual test, not a bound on the distance to the fixed point | `major` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-006 | Non-finite numbers are written as non-standard JSON tokens | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-007 | HeatNode applies its Dirichlet boundary data at the first cell centre, not at the rod ends it documents | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-008 | HeatNode's 4th-order stencil converges at 1st order and is less accurate than the 2nd-order one | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-009 | HeatNode's documented CFL limit is the 2nd-order stencil's; the 4th-order stencil diverges below it | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-010 | A string spelling a non-finite token is refused by the JSON serialisers | `minor` | `context_dependent` | `open` | >=0.4.0.dev0 |
| MADD-ANO-015 | The ZeroMQ transports bound every interface with no authentication or encryption | `critical` | `safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-011 | BallNode's declared discretisation is forward Euler; the implementation is semi-implicit Euler | `minor` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-012 | HeartPumpNode samples its cardiac inflow at the end of the step, disagreeing with its own derivatives() | `minor` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-013 | Several nodes pin a float64 quantity to float32: HeartPumpNode's backpressure, and the rigid-body and HeatNode parameter and initial_state casts | `minor` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-014 | Every explicit integrator converges at 1st order when a time-varying boundary input is supplied once per step, including the one named 4th-order | `major` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-016 | cloud/_skypilot.py was written against a SkyPilot API older than the supported floor | `major` | `not_safety_relevant` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-017 | Under jax_enable_x64 a scan-shaped path refuses a float32 carry: implicit_euler_step in every release, and GraphManager's scans on a freshly compiled graph from 0.4.0 | `minor` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-018 | A calibrated params value reaches update() and cannot reach derivatives(), implicit_residual() or integrate_node() | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-019 | A coupling iteration that diverged to a non-finite state was reported residual=0.0, converged=True | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-020 | LBMNode's Zou-He pressure boundary imposed rho_p + S_K instead of the prescribed density | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-021 | A node that branches on its own state inside update() returns the gradient of the branch it took; BallNode's gradient through a bounce omits the contact time and is exactly zero in the drop height | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-022 | A mapped edge whose point reference names a grid derived from a trainable parameter keeps the constructor's geometry when that parameter is calibrated | `major` | `context_dependent` | `open` | >=0.4.0.dev0 |
| MADD-ANO-023 | The FMU TCP bridge authenticates no caller: any process that can reach its port can read, write, step and hold the model | `major` | `context_dependent` | `open` | >=0.4.0.dev0 |
| MADD-ANO-024 | PUT /graph/params accepted a value a node consumes at construction: reported and saved, never used by the running graph | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-025 | A sharded LBMNode computed a different model from the unsharded node: pressure faces imposed at every seam of a sharded face axis, and an edge-filled global halo by default | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-026 | With waveform_iterations > 1, coupling_diagnostics() reported only the last sweep's pass count: an earlier sweep stopped at max_iterations read as iterations=1 | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-027 | Sub-cycled coupling relaxes no waveform: waveform_iterations > 1 re-solves the same fixed point, and the sub-step interpolation runs between iterates, not across the step | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-028 | A sharded stencil node on a mesh axis of one device ran with periodic global halos whatever `boundary` said | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-029 | The default `"edge"` halo fill copied a halo two or more cells wide from the shard's first cells instead of repeating its edge cell | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-030 | A sharded HeatNode ignored its boundary temperature inputs: left_temperature and right_temperature never reached update_padded's rod ends | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-031 | A HeatNode at stencil_order=4 with no boundary input is not insulated: heat crosses a rod end that has no temperature given | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-032 | After a parameter write and compile(), the sharded wrappers kept computing with what they had built from the old value: a legacy node's constant, and ShardedStencilNode's halo width | `major` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.2.0 |
| MADD-ANO-033 | ShardedStencilNode stepped with dt rounded to float32 under jax_enable_x64 | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-034 | LBMNode's wall_mask was dropped by every save/reload: the reloaded graph ran with no walls | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-035 | ShardedUnstructuredNode reads a global-order state as partition layout on a balanced partition not in global order | `major` | `context_dependent` | `open` | >=0.3.0 |
| MADD-ANO-036 | A config round trip dropped a node's sharding with no word: the reloaded graph ran unsharded | `minor` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.2.0 |
| MADD-ANO-037 | ShardedUnstructuredNode accepted a Cartesian stencil node and stepped it wrong, every cell, with no error | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-038 | ShardedStencilNode edge-filled a sharded static's global halos under a periodic wrapper | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.1, <0.4.0 |
| MADD-ANO-039 | ShardedUnstructuredNode dropped the cells a node had past the layout's count, and gave a node no way to leave padding out of an integral | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-040 | The sharded wrappers placed a domain integral carried in the state like a grid field: an integral-emitting node could not run in a graph | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.2.1, <0.4.0 |
| MADD-ANO-041 | halo_exchange ignored a per-axis boundary key naming no exchanged mesh axis, and that axis took the edge fill | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-042 | ShardedStencilNode accepted an empty axis_map and sharded nothing | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-043 | run_adaptive / run_adaptive_scan advanced a sub-cycled node by its rate divider times dt on every adaptive step | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-044 | On a multi-rate graph a coupling group's diagnostics, predictor history and IQN-IMVJ warm start came from solves the step discarded | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-045 | solver="fori" with acceleration="iqn-imvj" carried zero secant columns to the next step, so jacobian_reuse did nothing | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-046 | A sub-cycled node whose timestep does not divide the group's macro timestep covered round(macro / node_dt) * node_dt per macro step, and nothing refused it | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-047 | PUT /graph/params accepted a write that flipped a branch the node fixed at construction: the running graph kept the old branch, a saved graph reloaded the new one | `major` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-048 | PUT /graph/params accepted values the node's own constructor refuses, so a graph saved after the write could not be loaded | `major` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-049 | A non-finite value written through PUT /graph/params was stored, then the reply failed with a 500, and every later GET /graph/params for the node failed too | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-050 | Two HeatNode rods coupled end to end are unstable above Fourier number 3/8 (stencil_order=2) or 0.226 (stencil_order=4) once the exchange converges, below the limit each rod's constructor accepts | `major` | `context_dependent` | `open` | >=0.4.0.dev0 |
| MADD-ANO-051 | The HTTP API served every route, POST /cloud/launch included, to any caller with no credential, and the shipped container bound it to 0.0.0.0 | `critical` | `safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-052 | POST /checkpoint/save and /checkpoint/load took any server path from an unauthenticated caller: a file write anywhere the server could write, and a file-existence oracle | `critical` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-053 | The WebRTC signaling server validated its own token, not the client's, so it admitted and relayed every client | `critical` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-054 | The FMU TCP bridge unpickled the state blob an importer hands to fmi3SetFMUState: remote code execution for anyone who could reach its port | `critical` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-055 | deserialize_fmu_state and FmuSidecar.handle unpickled the bytes they were given | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-056 | ShardedStencilNode read the block size it hands update_padded in shard_info off a domain integral carried in the state, so a sharded HeatNode carrying a vector or per-shard energy integral never closed its right rod end | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.1, <0.4.0 |
| MADD-ANO-057 | A sharded HeatNode or LBMNode applied, on every shard, a heat_source or body_force of a shape the unsharded node refuses | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-058 | The LBM constructors accepted a non-finite viscosity or relaxation time, and LBMPipeNode a propeller disc outside the grid, each running a different model with no error | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-059 | Accelerating a coupling group that holds an integer, boolean or PRNG-key state leaf failed at the first step | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-060 | A group-internal flux edge read by the interface norm, or by a sub-cycled member's linear boundary interpolation, failed with a bare KeyError | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-061 | run_adaptive advanced its clock by dt_min when it accepted an attempt at dt_min, while the state advanced by the attempted step | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-062 | HeatNode accepted a non-positive length or timestep, a negative diffusivity and grid_points out of order, and answered wrongly without a word | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-063 | A new value of a parameter from which a node derives the points of a mapped edge was taken by PUT /graph/params and gm.params, run with the old points' mapping weights, and saved as a config that does not load | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-064 | ShardedUnstructuredNode read a per-cell input the length of a shard's slab as every shard's own slab, where the unsharded node refuses it | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-065 | ShardedUnstructuredNode.gather_global reshaped a domain integral listed in state_fields() as a per-cell field: a scalar raised, and values one per shard came back reordered | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-066 | A ShardedStencilNode nested in another stacked a per-shard domain integral's initial value twice, so step() changed the state's shape | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-067 | ShardedStencilNode summed a domain integral over a mesh axis its axis_map leaves unused, counting every block once per device along it | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.1, <0.4.0 |
| MADD-ANO-068 | XLA miscompiles a ShardedStencilNode step inside a loop (run_scan, a coupling group's iteration) when a sharded static is replicated over a mesh axis and read in the halo beside a window at the shard's offset | `major` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.4.0.dev0 |
| MADD-ANO-069 | hold_undetermined held every direction a fit's gradients had not spanned, so a short or fast-converging fit returned parameters above the loss it had reached | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-070 | solver="ift" held a coupling group's non-floating fields at their first-pass values: a flag reading the iterate came back stale, and one read across an edge froze the map | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-071 | A coupling group in which a flux producer reads a flux failed with a bare KeyError under iteration_mode="jacobi" | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-072 | After jax.grad through run_scan, set_node_state wrote into the traced state and the next entry point discarded the write; reset_state raised on a predictor group | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-073 | Under acceleration="fixed" with relaxation below 1/(1+sqrt(rho)), a coupling group stopping on its first loop pass understated its distance about 2x and reported converged | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-074 | The adaptive steppers' error norm read integer and boolean state leaves: a bool raised, a uint32 wrapped and a counter moved the step sequence | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-075 | Opening a USD stage imported the Python module the stage named for a node, so load_graph_from_usd ran a stage author's code | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-076 | The HTTP API checked no Origin, so a web page in the user's browser could drive a loopback-bound server and read its state stream | `critical` | `safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-077 | Swapping a surrogate in with replace_node, or out with POST /surrogate/deactivate, re-added each edge without its additive flag and units: an additive coupling became an overwriting one, silently | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-078 | build_model_description of a real graph declared no FMU inputs and a default step of 1e-3, whatever the graph's inputs and timestep | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-079 | compile() restarted a multi-rate graph's sub-step counter, so a recompile in the middle of a run re-phased the sub-steps each slower node fired on | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-080 | The FMU bridge split a variable's name at its first dot to find the node, so every input of a node whose name holds a '.' was filed under a node the graph does not have | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-081 | After a sharded static was rewritten in place, compile() traced the new step against the wrapper's cached copy of the old buffer | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-082 | The FMU bridge installed an importer's FMU-state archive without the value checks set applies: a parameter outside its declared bounds and non-finite state were accepted | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-083 | A second FMU instance on the same bridge started from the previous instance's final state, parameters, inputs and time | `critical` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-084 | An external input the FMU did not export, or that an in-process caller omitted, was not zero-filled: the node took its own "input missing" branch | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-085 | The FMU advertised min / max for its parameters but enforced them only if the sidecar had been given param_specs: a set or an archive outside them was accepted | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-086 | A REST request could make the server allocate without bound: before 0.4.0 POST /graph/nodes built any size, and 0.4.0's size checks ran only after the node was built | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-087 | PUT /graph/params answered 200 to a value the running node could not step with, and every POST /sim/step after it was a 500 | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-088 | POST /sim/stop answered "stopped" while the runner's thread kept stepping, and a reset after it was overwritten | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-089 | A checkpoint path that resolved to the checkpoint root itself wrote, and loaded, <root>.npz in the root's parent | `major` | `not_safety_relevant` | `resolved` (in 0.4.0) | none |
| MADD-ANO-090 | PUT /sim/stride echoed values it did not apply, and DELETE /graph/edges answered 200 for an edge the graph did not have | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-091 | A state or render WebSocket stream never noticed a client that left after the simulation stopped, so the server could not shut down; an ordinary disconnect was logged as an error | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-092 | POST /sim/run and POST /surrogate/train took any size: one request could hold a worker indefinitely or name more memory than the host has | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-093 | A node.params write made after compile was dropped by every later compile: compile() kept the stale gm.params leaf over the node's new value | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-094 | Under Gauss-Seidel the coupling bound's float floor counted the worst node's evaluations, not the chain a pass composes: a stalled 32-relay ring read its spectral bound at 0.51x the true distance with spectral_usable=True | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-095 | After run_adaptive or run_adaptive_scan the coupling report described only the second kept half step: a first half step stopped at max_iterations was hidden, and converged read True where strict_convergence refused the step | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-096 | GraphManager.timestep was the GCD of the nodes' own timesteps, shorter than a step on a graph with a sub-cycling coupling group, so the live runner's clock and pacing and the REST and ZMQ relays' frame times ran slow | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-097 | USD baseDt and the FMU default step read a _base_dt attribute that nothing set: USDWriter wrote 0.01 for every graph, and save_graph_to_usd gave the smallest node timestep rather than the graph's step | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-098 | Two SpringDamperNodes anchored on each other in a converged coupling group gain momentum without bound when stiffness*dt > damping, far inside the single node's stability limit | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-099 | GraphManager.step did not zero-fill the inputs a partial external_inputs left out, and dropped an undeclared node or field name in silence: the node took its own "input missing" branch | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-100 | The FMU bridge took any master_dt: a doStep ran h / master_dt graph steps and reported t + h, so a 0.01 s graph served with master_dt=0.005 labelled a state at 0.10 s with the time 0.05 s | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-101 | A number for a Boolean FMU variable was stored by its truthiness: set gate.open = 0.5, 2.0 or -3.0 read back as true, and an FMU-state archive holding 0.25 in a Boolean field restored as true | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-102 | The FMU's C wrapper carried every width as a double and the bridge could not tell the calls apart: fmi3SetBoolean wrote 1.0 into a Float32 parameter and fmi3GetInt32 truncated a Float32 output, both fmi3OK | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-103 | An FMU-state archive missing its pending-input members restored with those inputs silently at zero | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-104 | A trainable parameter one step carried past the edge of its coordinate range stayed there for the rest of the fit, and fit_lm reported converged=True away from the optimum | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-105 | A node.params write with no recompile reached whichever program was traced next: gm.step kept its cached trace while run_scan at a new length, and later the sysid losses, ran the written value | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-106 | Concurrent REST requests stepped the graph at once and lost steps silently: 200 concurrent POST /sim/step took about 60, every one answered 200, and the streams counted all 200 | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-107 | POST /surrogate/train reset the live simulation to its initial state, and its reply did not say so | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-108 | After POST /checkpoint/load the state streams served the state from before the load, at the old step's time, until the next step | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-109 | /ws/state/binary kept the frame layout of a node that had been replaced over REST: values cut off, or phantom zeros, in frames of the advertised length | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-110 | A runner whose thread had died was reported running: /sim/pause and /sim/resume answered 200, /sim/start 409 "already started", and /sim/reset was_running true | `major` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-111 | The real-time runner burst through the time it had been paused: about 1.5 s simulated in the 0.4 s after a 1 s pause of a real-time run | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-112 | /ws/state encoded each frame on the event loop, once per client, for any number of clients: a large state stalled every other request | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-113 | The IFT adjoint (and in 0.4.0's development the tangent) of a coupled step was exactly zero, reported successful, whenever its right-hand side was below about 1e-8 | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-114 | The L2 coupling norm summed squared absolute changes: below about 1e-19 a pass's change squared to zero, and the group read residual=0.0 after one pass at 90% from its fixed point | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-115 | Below about 1e-33 the accelerators' steps and the report's spectral rate and gradient bound were formed from flushed differences: converged=True up to 7.4x the threshold from the fixed point, a gradient bound of 0.0 with gradient_bound_usable=True | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-116 | solver="fori" with Aitken or IQN acceleration on a bfloat16 or float16 coupling group raised a carry TypeError at trace time | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-117 | profile_graph's per-node timing called each node with no boundary inputs, so a node that reads an input raised KeyError | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-118 | run_adaptive accepted a rejected attempt larger than dt_min whenever shrinking it would reach dt_min, where run_adaptive_scan retried at dt_min | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-119 | An FMU-state archive installed new interface-mapping weights, which are not FMI variables: the FMU then computed a coupling its model description does not describe | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-120 | A node downstream of a cycle that was added to the graph before the cycle's nodes was scheduled ahead of them and read their previous-step output: an edge on no cycle, staggered silently | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-121 | fit_lm's Marquardt floor, scaled to the mean curvature, crushed the step of a parameter measured in small units, and fit_lm reported converged=True away from the truth | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-122 | A pending node.params write reached the entry points that run the graph but not the code that reads gm.params: a jitted or differentiated sysid loss, the FMU export, save_state and GET /graph/params used the old value | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-123 | sharded_cg, sharded_gmres and ift_linear_solve defaulted to an absolute atol=1e-8: a right-hand side near 1e-9 came back wrong (relative error 0.4-1.0) with converged=True | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-124 | sharded_gmres's loop backend returned NaN with converged=False whenever restart exceeded the system size, the default 50 against any system of fewer than 50 unknowns | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-125 | The multi-rate base timestep's GCD stopped on an absolute 1e-9: nodes at 1e-9 and 2e-9 got a base step of 2e-9 and their clocks drifted apart | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-126 | fit and fit_multiple_shooting applied Adam's eps=1e-8 in the loss's own units: a loss near 1e-12 left its parameters where they started | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-127 | The adaptive error norm guarded its division with max(scale, 1e-300): in float64 an entry below about 1e-297 read its error up to 1e297 times too small | `major` | `not_safety_relevant` | `resolved` (in 0.4.0) | none |
| MADD-ANO-128 | The adaptive steppers' default atol=1e-6 and dt_min=1e-8 are absolute, in state units and seconds: a state near 1e-9 is accepted at any step size | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-129 | The deprecated calibrate and tune_coupling_params default to absolute thresholds: a loss already below 1e-6 is converged=True before a single step | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-130 | The anomaly-registry validator raised TypeError on a list-valued severity, safety_relevance or resolution_status instead of reporting the entry | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-131 | POST /surrogate/train's memory budget was checked on the graph at the request, and the job swept the graph as it was later: a node added in between bypassed it | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-132 | POST /sim/profile reset the live simulation and left it at the profiler's last step, and the streams counted the profiler's steps | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-133 | A reset, a state write or a node removed over REST was not shown by the streams: they served the state from before it until the next step | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-134 | A JAX trace started over REST had no bound: left running, it grew the server's memory by kilobytes per step until stopped | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-135 | The fitters' identifiability guard asked its questions in the optimiser's coordinates, so a parameter's units decided what it held: a damping the data determined, in units 1e-5 or 1e6, was held at its start and fit_lm's loss rose from 1e-11 to 13.9 or 0.1, silently | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-136 | ParamSpec.check compared through jnp, which flushes a subnormal to zero on CPU: a float32 -1e-40 passed a (0, None) bound in check_params, the FMU's set and set_state and REST's PUT, and read back below the advertised min | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-137 | The FMU's value check refused a value its type overflows but stored one it underflows: a float32 set to 1e-50 read back as 0.0, the value and its sign lost | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-138 | load_state cast a checkpoint value to the graph's dtype with no check: a float64 1e39 loaded into a float32 field as inf, and -1e-50 as -0.0 | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-139 | PUT /sim/stride reset a value left out of the query to 1: a call naming only steps_per_frame reset the relay's stride, and the reverse | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-140 | A generated API token written to MADDENING_API_TOKEN_FILE kept the mode of a file already at that path: a placeholder of mode 0644 left the token readable by every local user | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-141 | POST /checkpoint/save refused at its final move answered "nothing was written" having replaced the manifest of that name, and checkpoint refusals named the server's absolute paths | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-142 | gradient_relative_error_bound read up to 5.2x below the true gradient error with gradient_bound_usable=True: the resolvent applied to each secant was the Krylov-restricted Arnoldi factor, and Newton-Kantorovich used it too | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-143 | For an array-valued constant the gradient bound covered one \|c\|-weighted random direction, not each entry: the gradient in a small entry read 23.6x above the bound, usable | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-144 | An outside node added between two members of a coupling group that is part of a larger feedback loop read the group one step late | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-145 | CouplingGroup accepted waveform_iterations <= 0, which ran no sweep and froze a sub-cycling group, and other out-of-range counts and thresholds | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-146 | spectral_error_bound read 0.94x the true distance in the norm it documents (the returned state's weights) on a group still growing toward its fixed point, spectral_usable=True | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-147 | An FMU built over a pending structural node.params write ran the old model, while every graph entry point ran the new one | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-148 | A log parameter with no lower bound was advertised with no FMI min, and a bridge whose sidecar had no specs accepted and ran mass = -1 | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-149 | The FMU bridge's time tolerance had an absolute floor, and a biased importer's reported time left the simulated time behind without bound | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-150 | Under jax_enable_x64, and with float32 constants in an x64 graph, the identifiability guard missed an exact degeneracy: full excited_rank, nothing held, the fitted scale 1-2% off its start | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-151 | windowed_loss restarted every coupling warm start cold at each window: with a predictor or IQN-IMVJ the loss at the generating parameters was not zero, and a fit started there walked 6.4% away | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-152 | fim's float32 rank verdict moved with noise_std once F = J^T J left the normal range: a determined direction dropped out, every crb inf, no warning | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-153 | The fitters read wrongly typed arguments instead of refusing them: fit(tol=True) reported converged at an unfitted start, and a mask leaf "False" fitted its parameter | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-154 | The IFT tangent and adjoint of a coupling group read exactly 0.0, reported successful, when the tangent or cotangent had a NaN or infinite entry | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-155 | sharded_cg returned zeros with converged=True for a right-hand side with a NaN or infinite entry, and ift_linear_solve returned zeros with no error | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-156 | A flux edge that crosses a coupling group's boundary -- from an outside producer into a member, or from a member to an outside reader -- fails to trace with a bare KeyError | `minor` | `not_safety_relevant` | `open` | >=0.1.0 |
| MADD-ANO-157 | A flux edge that an ungrouped cycle reads from the previous step fails to trace with a bare KeyError, so whether the graph runs depends on the order its nodes were added | `minor` | `not_safety_relevant` | `open` | >=0.1.0 |
| MADD-ANO-158 | A typed PRNG key in a coupling group member fails to trace under solver='ift' with acceleration='aitken' or 'fixed' | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-159 | A coupling group whose members are joined only through an outside node, on no cycle, reads that node one step late with no warning | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-160 | jax.grad through run_adaptive_scan was 0.0 or NaN where an attempt's step-doubling estimates agreed exactly, as for an accelerated coupling group whose iterate starts at its fixed point | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-161 | jax.grad through a 16-bit coupling group under solver="ift" raises NotImplementedError | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-162 | strict_convergence on a coupling group with a sharded member aborts the process | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-163 | POST /checkpoint/load restored parameter values PUT /graph/params refuses -- past a node's constructor limit, outside a ParamSpec's bounds, non-finite, a boolean, a numeric string -- answered 200, and left a graph whose save did not reload and whose steps diverged | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-164 | A parameter write to a HybridNode was answered and then lost: the hybrid held a copy of its physics node's params, so PUT /graph/params echoed the new value and the step kept the old one | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-165 | POST /sim/run that could not have the graph part-way through answered 503 'Nothing was changed; retry shortly' after it had stepped, so a client that retried stepped the graph twice; its final read waited with no timeout | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-166 | POST /sim/profile answered 500 for a graph that could not compile with any error but a RuntimeError | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-167 | A ParamSpec changed between the FMU description and its sidecar was enforced in place of the advertised min / max: the bridge accepted values its XML forbids, or refused values it declares settable | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-168 | At a large ratio of time to master step the FMU bridge's ulp slacks exceeded a master step: a doStep a whole step ahead was adopted, and a clock running 40% fast was never refused | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-169 | One in-limit get frame drove the FMU bridge past 6 GB and held the instance for about 45 s before its reply was refused | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-170 | The Aitken two-pass guard can add two passes to Aitken's own exit, where the guide says at most one | `minor` | `not_safety_relevant` | `open` | >=0.4.0.dev0 |
| MADD-ANO-171 | save_state cannot write a typed PRNG key leaf | `minor` | `not_safety_relevant` | `open` | >=0.1.0 |
| MADD-ANO-172 | reset_state() inside a differentiated loss raises UnexpectedTracerError on a predictor group after an earlier transform | `minor` | `not_safety_relevant` | `open` | >=0.4.0.dev0 |
| MADD-ANO-173 | run_adaptive never returns once its step-doubling error norm is NaN | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-174 | fit_lm reported converged=True at a wrong point, or at its unmoved start, for a float32 residual or parameter far from unit scale; fit returned its start | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-175 | node.params writes that never reached gm.params: every write into a replaced mapping, and an in-place element write to a list or array | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-176 | fim and fim_core reported rank 0 silently when F = J^T J flushed to exactly zero from a Jacobian that was not | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-177 | A logit ParamSpec whose width overflows the leaf's dtype was accepted, and constrain mapped the midpoint to the upper edge and far coordinates to NaN | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-186 | A bfloat16 or float16 coupling group's norms and floor were computed in its own dtype: residual 0.0 above 65,504 active entries, inf on a finite state at the default rtol | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-187 | spectral_error_bound under convergence_norm='interface' with a transform on an internal edge read 0.0014-0.098x the true distance, spectral_usable=True | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-188 | gradient_relative_error_bound read 0.65-0.986x the true gradient error, gradient_bound_usable=True, on a group stopped far from its fixed point | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-189 | The adaptive steppers' error norm was computed in a bfloat16 or float16 state's own dtype: run_adaptive did not return on a float16 state above 65,504 elements, run_adaptive_scan refused any 16-bit state | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-190 | run_adaptive_scan raises a scan-carry TypeError on a float32, bfloat16 or float16 state under jax_enable_x64 | `minor` | `not_safety_relevant` | `open` | >=0.1.0 |

*182 anomalies registered.  32 have a defect reachable in this version — every entry whose `resolution_status` is not `resolved` or `duplicate`, which is 23 `open` plus 9 `partially_resolved` whose residual risk is still live.  The Affected Versions column is a PEP 440 specifier set read against this document's version; `none` marks a defect introduced and fixed within one development cycle, which no release carried.  The convention, and the gate that holds every range to it, are in the header of `known_anomalies.yaml`.  Rationale, workaround, affected components and verification evidence for each: `known_anomalies.yaml`.*
<!-- END GENERATED: known-anomalies -->

## 4. Verification Evidence

See [`framework_verification.md`](framework_verification.md) for the registered
verification benchmarks and the shape of the test suite.  The
machine-readable registry is `maddening.compliance.get_benchmark_registry()`.

## 5. IEC 62304 Lifecycle Activities

See `docs/regulatory/iec62304_mapping.md` for the full lifecycle mapping.

## 6. Configuration Management

- Version control: Git (GitHub)
- Release tags: semantic versioning (`vX.Y.Z`)
- {term}`SBOM`: CycloneDX 1.6 JSON, one per covered install, in
  `docs/validation/sbom/` (below)
- CI: GitHub Actions

### Software Bill of Materials

Each SBOM is the environment that a clean install of MADDENING's wheel
resolved to.  `scripts/generate_sbom.py` builds the wheel from the tree,
installs it into a new, isolated virtual environment, and runs
`cyclonedx-py` against that environment from a separate tool environment, so
the tool is not in the SBOM.  MADDENING is the root component, with its
version, purl and licence.  Every other installed distribution is a
component with its name, version, `pkg:pypi` purl and the licence its
metadata declares, and its own `Requires-Dist` metadata, verbatim
(`maddening:sbom:requires-dist` properties, with a count).  The dependency
graph is included, and the root's edges are exactly the direct dependencies
`pyproject.toml` declares for that install.

| File | Install | What it covers |
|---|---|---|
| `maddening-0.4.0.dev0-core.cdx.json` | `pip install maddening` | The base dependencies: what every user gets, and the SOUP items of §1 |
| `maddening-0.4.0.dev0-server.cdx.json` | `pip install maddening[server]` | The network-facing bundle: the HTTP/WebSocket API, the ZeroMQ transports, terminal (`rich`) and matplotlib rendering, zstd frames.  A superset of the `api`, `network`, `viz` and `compression` extras, and of `terminal` except `termaid`, which draws the graph diagram on an operator's terminal (below) |
| `maddening-0.4.0.dev0-surrogates.cdx.json` | `pip install maddening[surrogates]` | Neural surrogate training (`optax`).  A trained surrogate replaces a physics node, so this code is in the computed result |
| `maddening-0.4.0.dev0-usd.cdx.json` | `pip install maddening[usd]` | OpenUSD stage read and write (`usd-core`, a binary wheel that bundles OpenUSD's C++ libraries) |

**Why these installs, and not the others.**  These are the installs whose
code runs in a deployed simulation: the base install, the network surfaces
a deployment exposes, the surrogate path that feeds computed results, and
the geometry import path.  The rest are left out on purpose:

- `cuda12` and `tpu` install hardware runtimes, and GPU is not a verified
  configuration: CI runs on CPU (MADD-ANO-001).  On 2026-09-25 `cuda12`
  resolved fifteen more wheels on Linux: JAX's two CUDA plugin wheels and
  thirteen NVIDIA CUDA libraries.  A GPU deployment records its own SBOM.
  The cloud Docker image installs `jax[cuda12]` with `.[server]`, and
  `python scripts/generate_sbom.py --extra cuda12+server --output-dir <dir>`
  records that combination.
- `all`, `dev` and `ci` describe a developer's workstation, not a deployment.
  On 2026-09-25 `all` resolved 176 packages, including SkyPilot's cloud SDKs,
  PyGObject and the display stack.  That is the shape of the whole-machine
  `sbom.json` which sat at the repository root from 2026-03 until 0.4.0
  removed it.  It listed one machine's packages, not MADDENING's dependency
  set.
- The cloud extras (`runpod`, `lambda`, `aws`, `gcp`, `cloud`, `cloud-all`)
  are launch tooling on the operator's machine, not code in the simulation's
  process.  `viz3d`, `gpu-viz` and `streaming` are display-side
  visualisation, and so is the `terminal` extra's `termaid`, which draws
  `GraphManager.print_graph_diagram` and computes nothing.  `verify` and
  `sbom` are tooling, and `ift` is empty.

**What an SBOM records, and what it does not.**  Each file records the
Python version and platform it was resolved on (the PEP 508 marker
variables and the wheel platform tag, as `maddening:sbom:*` properties) and
the resolution cutoff (`maddening:sbom:exclude-newer`): only distributions
uploaded before it were considered.  All three decide the transitive
versions.  jaxlib, numpy and scipy ship per-platform wheels, and a later
date resolves newer releases inside the declared ranges.  An SBOM here
records one resolution, on Python 3.12 and `linux-x86_64`, as of the
recorded date.  It is not a lock file: `pip install maddening` resolves
afresh, and it is not the environment any CI run installed (see the warning
in `framework_verification.md`).  A deployment records its own environment
by running `generate_sbom.py` for its own install, or `cyclonedx-py` on its
own environment.

**Consistency.**  `scripts/check_sbom.py` runs in CI through
`tests/compliance/test_sbom_check.py`, with no network.  It fails if a
covered install has no SBOM at the version `pyproject.toml` declares, or if
the directory holds any other SBOM.  It fails if a direct dependency
`pyproject.toml` declares for an install is missing from that install's
SBOM, or is at a version outside its declared range.  It fails if a SOUP
item §1 lists is missing, or is at a version outside the `pyproject.toml`
range.  It also fails if a component has no purl, or a purl that disagrees
with it, or no licence or a blank one; if a component is reached by no path
from the root in the dependency graph, so that no install brings it in; if
the recorded Python is one `requires-python` refuses; and if a file does not
record its resolution cutoff.  It follows the install's requirements from
the declared direct dependencies through each component's recorded
`Requires-Dist`, with markers evaluated in the recorded environment, and
fails if a requirement the install turns on names a package the SBOM lacks
or a version the requirement refuses, if the graph and the requirements
disagree, or if a component is required by nothing the install turns on.
It fails if the four files disagree about the cutoff, the index, the
resolver, the platform or any marker variable: they are one resolution, and
are regenerated together.  And it fails if a file was changed after
generation without being re-sealed: the `serialNumber` is derived from the
content.  Each failure names the discrepancy.

These checks prove that the files are consistent with `pyproject.toml`, with
§1 and with themselves.  They do not prove that the content is what a
resolver produced: the sealing function is public, so an edit can be
re-sealed, and one that breaks none of the rules above (a package moved to
another version that its range and its dependants all admit) passes.  Only
regeneration proves the content.  `python scripts/generate_sbom.py
--exclude-newer <the recorded cutoff> --output-dir <dir>` resolves each
install again, and its output must be byte-identical to the committed file.

**Determinism.**  Components and the dependency graph are sorted, keys are
written sorted, `metadata.timestamp` is the resolution cutoff (or
`SOURCE_DATE_EPOCH` when set), and the `serialNumber` is a UUIDv5 of the
content.  Regenerating with the same cutoff on the same platform therefore
writes the same bytes.  To compare SBOMs generated on different days,
`python scripts/check_sbom.py --normalise <file>` prints one without the
three date-dependent fields (`serialNumber`, `metadata.timestamp`, the
cutoff).

**At release, the SBOMs are regenerated from the tagged build.**  The files
above are generated from this tree at the version `pyproject.toml` declares,
which their names carry.  They are not the release SBOM.  The release step,
done by the maintainer:

1. On the release commit, after the version bump and before the tag,
   change the version in the file names in the table above.
2. Run `python scripts/generate_sbom.py` (it needs `uv` and network
   access).  It builds the wheel from that tree, resolves each install
   afresh, writes `maddening-<version>-<install>.cdx.json` for the new
   version, removes the previous version's files, and runs the check.
   Until both steps are done the check fails on the version bump, which
   makes them hard to skip.
3. Commit the SBOMs with the release commit, tag that commit, and attach
   the four `.cdx.json` files to the GitHub release as assets.

## 7. Anomaly Management Policy

See CONTRIBUTING.md for the three-phase anomaly lifecycle and three-tier release gate model.

## 8. Dependencies

The base dependencies — what `pip install maddening` pulls in — are listed in
[§1 Software Identification](#1-software-identification), generated from
`pyproject.toml`.  The optional extras are declared in `pyproject.toml`.
The transitive tree, with the version and declared licence of every
package, is in the CycloneDX {term}`SBOM` of each covered install (§6): the
core SBOM for the base install, and one each for the `server`, `surrogates`
and `usd` extras.
