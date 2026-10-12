# Changelog

All notable changes to MADDENING will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Additional sections per release: **Verification**, **Security**, and **Known Anomalies**.

## [Unreleased]

The release notes, [`docs/release_notes/v0.4.0.md`](docs/release_notes/v0.4.0.md),
give each item with its guide and say what to do. How the release was built and
checked, with the defects found and fixed before it, is in
[`docs/validation/development_record_v0.4.0.md`](docs/validation/development_record_v0.4.0.md).

### Added

- **`gm.matmul_precision`** (`EXPERIMENTAL`): set it to `None` to trace a graph's programs at JAX's own `jax_default_matmul_precision` instead of `"highest"` (reduced-precision tensor-core speed for MLP or FNO surrogate nodes); `sysid.fim` and `sysid.fim_core` take a `matmul_precision=` keyword (`EXPERIMENTAL`), and the fitters follow the graph's setting.
- **Node constants are parameters**: `gm.params` holds every node's constants as one tree of arrays that the compiled step takes as an argument, so `jax.grad` reaches them and a change does not recompile; the run methods take `params=` (`STABLE`). All eleven shipped node classes and both sharded wrappers take part; `gm.nodes_without_params()` lists your own nodes that do not.
- **`ParamSpec`** (`EVOLVING`): trainable, bounds and transform per parameter, with `gm.set_param_spec`, `gm.param_specs`, `gm.trainable_mask`, `gm.unconstrain`, `gm.constrain` and `gm.check_params`.
- **`maddening.sysid`** (`EVOLVING`): `fit`, `fit_lm`, `fit_multiple_shooting`, `windowed_loss`, `observations_from_history`, and `fim` (Fisher information, rank, Cramér-Rao bounds); the fitters hold the directions the data cannot determine. `fim_core` / `FIMCore`, with no host round trip, are `EXPERIMENTAL`.
- **Parameters follow the graph**: configs, USD stages and checkpoints store values and `ParamSpec` overrides; `PUT /graph/params/{node}` writes any parameter of a running graph after checking it; an exported FMU exposes each parameter as an FMI parameter.
- **Interface mappings on edges** (`EVOLVING`): `add_edge(..., mapping=)` with `rbf_mapping`, `nearest_neighbor_mapping`, `projection_1d_mapping` and `matrix_mapping`, weights in `gm.params["mappings"]`, saved as a `MappingSpec`.
- **Sparse interface mappings** (`EXPERIMENTAL`): `StaticSparseMapping` and the kinds `sparse_nearest_neighbor`, `sparse_projection_1d` and `sparse_matrix`.
- **Geometry-dependent mappings** (`EXPERIMENTAL`): `add_edge(..., mapping=m, geometry=(anchor, field))` and the kind `multilinear_grid`, between a uniform grid and points held in a node's state. A single-rate group whose geometry mappings are all `multilinear_grid` reports its bounds under `"l2"` and `"mixed"`; see the guide for the limits.
- **`@register_mapping`** (`EXPERIMENTAL`): a mapping kind of your own, saved and rebuilt like the built-in ones; a file can only name a kind already registered.
- **A graph exports as a working FMI 3.0 FMU** (`EVOLVING`): `build_fmu_binary` and `write_fmu` build a C co-simulation wrapper over a local TCP bridge; large arrays cross as binary frames; `build_model_description(multi_clock=True)` declares one clock per node timestep.
- **Coupling bounds and a convergence report**: `coupling_diagnostics()` reports per group an error estimate, `spectral_error_bound` with `rho_spectral`, and `gradient_relative_error_bound`, each with a flag saying whether to rely on it; `strict_convergence=True` raises on an unconverged step (`CouplingGroup`, `EVOLVING`).
- **Reason codes** (`EXPERIMENTAL`): `reason_codes`, `not_usable_reason` and `residual_precision_floor` in each coupling report say why a flag is `False`.
- **Coupling groups survive save and load**: configs and USD stages carry each group with all its options.
- **`AdaptiveNode`** (`EVOLVING`), a base class for solvers on a changing active set with a gradient exact while the set holds, and **`WaveletAdaptiveNode`** (`EXPERIMENTAL`), the first concrete one.
- **Per-neighbour halo exchange**: `ShardedUnstructuredNode(..., exchange="ppermute")` and `exchange_traffic(layout)` (`EVOLVING`).
- **Differentiable sharded solvers**: `sharded_cg` and `sharded_gmres` take `differentiable=True`; `jacobi_preconditioner` and `block_jacobi_preconditioner` (`EVOLVING`).
- **The sharded wrappers take part in parameters**, split grid-shaped boundary inputs across devices, and honour `SimulationNode.domain_integral_axes()`.
- **The HTTP API authenticates** off loopback (`APIAuth`, `EVOLVING`), checks the request's origin, confines checkpoint paths, and has a `/healthz` route; a failed REST write leaves the graph as it was.
- **ZeroMQ transports are private by default**: loopback binds, CURVE off loopback (`TransportAuth`, `EVOLVING`).
- **Read-only inspection** (`EXPERIMENTAL`): `gm.print_graph()`, `to_mermaid()`, `to_dot()`, state, parameter, memory and coupling tables, and `gm.print_graph_diagram()` (the `terminal` extra).
- **A version report**: `python -m maddening info` and `maddening.show_versions()` (`EXPERIMENTAL`).
- **A persistent compilation cache** (`compile_cache.enable`, `MADDENING_COMPILATION_CACHE_DIR`, `warm_cache`) and a profiler that measures coupling overhead per group (`EVOLVING`).
- **`gm.reset_state()` and `gm.get_node()`**.
- **`maddening.testing`** (`EXPERIMENTAL`): `verify_node`, order of accuracy by manufactured solutions or grid refinement (`mms`), `verify_mapping`, `verify_graph_order`.
- **`SimulationNode.static_data_deps()`** (`STABLE`): a static rebuilt when a parameter it derives from changes.
- **`py.typed`**: MADDENING's annotations reach your type checker.
- **Eight new or extended examples**, and a `--local` mode for the remote-simulation examples.
- **CycloneDX SBOMs** for the base install and the `server`, `surrogates` and `usd` extras in `docs/validation/sbom/`.
- **Registries of other projects validate**: `python -m maddening.compliance check-anomalies` accepts one `<project>_version` field (`--version-field NAME`); `ResolutionStatus.PARTIALLY_RESOLVED`.
- **Guides**: geometry-dependent mappings, moving a node's own sampling onto an edge, a node that draws random numbers, profiling, typing, verification, and algorithm pages for seven existing nodes and the explicit integrators.

### Changed

- **Graphs trace their float32 matrix products at full precision by default** (MADD-ANO-291): the compiled step, every scan, sweep and adaptive program, the implicit-function gradient, the sysid fits and the Fisher information run under `jax.default_matmul_precision("highest")`. On NVIDIA GPUs results in the affected paths (interface mappings on multi-component fields, the coupling bounds, IQN, FNO surrogates, `J.T @ J`) move by up to about 1e-4 relative; on CPU nothing changes, bit for bit.
- **`LBMNode` and `LBMPipeNode` compute their lattice products at `Precision.HIGHEST`** (MADD-ANO-290): on an NVIDIA GPU at JAX's default matmul precision they ran at reduced precision, and LBM gradients were off by up to 1.6e-3 relative. GPU results change by up to that much; CPU results are bit-identical. On 0.3.1, set `JAX_DEFAULT_MATMUL_PRECISION=highest`.
- **Python `>=3.12` and `jax` / `jaxlib` `>=0.10,<0.13`** (0.3.1: `>=3.10`, `>=0.4,<0.6`). The ranges do not overlap: migrate JAX first. `lineax` and `scipy` are base dependencies; the `[ift]` extra is empty; the `terminal` extra adds `termaid`.
- **Coupling groups default to `solver="ift"`, and `converged=True` means within tolerance of the fixed point** (MADD-ANO-005, partly resolved; MADD-ANO-114). Tolerances are relative; `atol` is a dead band defaulting to `0.0`, and `compile()` warns for a group that sets it. Expect more passes; recorded Aitken and IQN trajectories move.
- **`run_sweep` and `run_scan` may differ by about one ulp**: constants are no longer compiled in.
- **`HeatNode`** applies its boundary data at the rod ends, refuses an unstable timestep and bad geometry, and reports the flux at the end (MADD-ANO-007, 008, 009, 062); `boundary_flux_spec()` declares `K*m/s`.
- **`gm.timestep` is the time one `step()` advances** (MADD-ANO-096).
- **Multi-rate graphs keep every clock**: no usable common step, or a sub-cycled timestep that does not divide its group's largest, no longer compiles (MADD-ANO-227, 046); a slow node's flux is that of the state it holds (MADD-ANO-235).
- **A step that changes the shape of the state raises `ValueError`** (MADD-ANO-220).
- **An edge and an external input on one field no longer compile** (MADD-ANO-265); `add_external_input` refuses a second, different declaration of a field.
- **`add_node` refuses** a timestep that is not finite and positive, the names `_meta`, `_params` and `_params_mappings`, and control characters (MADD-ANO-181, 182, 208); `remove_node` takes the node out of its coupling group (MADD-ANO-214).
- **`copy.copy(gm)` raises `TypeError`**: use `copy.deepcopy`.
- **`to_dict()` returns a copy** (MADD-ANO-205) and warns of a constructor argument it does not save (MADD-ANO-277, partly resolved).
- **`add_coupling_group` refuses out-of-range options** (MADD-ANO-145), and `compile()` refuses a group-internal flux edge the loop reads from the state (MADD-ANO-060, 071).
- **Under `convergence_norm="interface"`** a group returns members computed from readings that met the criterion (MADD-ANO-240, 241).
- **`load_graph_from_usd` builds only built-in and registered node classes** (MADD-ANO-075): pass `node_registry=`.
- **REST is strict**: `POST /graph/nodes` refuses five input classes with 400 and an oversized dimension with 422; numbers given as text or booleans and unknown keys are refused (MADD-ANO-228, 253).
- **The sharded wrappers refuse settings that gave wrong answers** (MADD-ANO-025, 030, 037, 039, 042, 057, 064), and `LBMPipeNode` and the LBM constructors refuse out-of-grid propellers and non-finite constants (MADD-ANO-058).
- **`sharded_cg` / `sharded_gmres`**: the default call answers where lineax's CG breaks down (MADD-ANO-264, partly resolved); `converged` and `residual_norm` describe the returned value on every route (MADD-ANO-263, 269, 281); the default `atol` is relative (MADD-ANO-123) and a concrete negative or NaN tolerance raises (MADD-ANO-271); differentiating needs `differentiable=True` (MADD-ANO-266).
- **`build_model_description`** refuses a `selected_outputs` pair with no state field and a step that is not a whole number of graph steps (MADD-ANO-201, 219).
- **`FMIVariable` is keyword-only; `LBMNode.meta.algorithm_id` is `MADD-NODE-011`; `validate_anomaly_registry()` requires `resolution_status`.**

### Deprecated

- `calibrate` and `tune_coupling_params`: use `sysid.fit`; removed in 0.5.0.
- `solver="fori"`: removed in the next minor release.
- `maddening.core.simulation.checkpoint.download_and_load_state`: use `maddening.cloud.download_and_load_state`.

### Removed

- The external formal-verification suite and its dependency; the `[verify]` extra installs only `hypothesis`. No public API was removed.

### Fixed

One line per defect that a released version had, with its registry entry and the releases that had it. Fixes that change behaviour are under "Changed" and "Security".

- MADD-ANO-105: A node.params write with no recompile reached whichever program was traced next: gm.step kept its cached trace while run_scan at a new length, and later the sysid losses, ran the written value (0.1.0 to 0.3.1).
- MADD-ANO-278: An assignment to node.params on a sharded wrapper or a HybridNode replaced the wrapper's mapping alone: the step ran the old value while node.params read the new one back (0.1.0 to 0.3.1).
- MADD-ANO-026: With waveform_iterations > 1, coupling_diagnostics() reported only the last sweep's pass count: an earlier sweep stopped at max_iterations read as iterations=1 (0.1.0 to 0.3.1).
- MADD-ANO-045: solver="fori" with acceleration="iqn-imvj" carried zero secant columns to the next step, so jacobian_reuse did nothing (0.1.0 to 0.3.1).
- MADD-ANO-059: Accelerating a coupling group that holds an integer, boolean or PRNG-key state leaf failed at the first step (0.1.0 to 0.3.1).
- MADD-ANO-113: The IFT adjoint (and in 0.4.0's development the tangent) of a coupled step was exactly zero, reported successful, whenever its right-hand side was below about 1e-8 (0.3.0 and 0.3.1).
- MADD-ANO-116: solver="fori" with Aitken or IQN acceleration on a bfloat16 or float16 coupling group raised a carry TypeError at trace time (0.1.0 to 0.3.1).
- MADD-ANO-144: An outside node added between two members of a coupling group that is part of a larger feedback loop read the group one step late (0.1.0 to 0.3.1).
- MADD-ANO-158: A typed PRNG key in a coupling group member fails to trace under solver='ift' with acceleration='aitken' or 'fixed' (0.1.0 to 0.3.1).
- MADD-ANO-159: A coupling group whose members are joined only through an outside node, on no cycle, reads that node one step late with no warning (0.1.0 to 0.3.1).
- MADD-ANO-161: jax.grad through a 16-bit coupling group under solver="ift" raises NotImplementedError (0.3.0 and 0.3.1).
- MADD-ANO-186: A bfloat16 or float16 coupling group's norms and floor were computed in its own dtype: residual 0.0 above 65,504 active entries, inf on a finite state at the default rtol (0.1.0 to 0.3.1).
- MADD-ANO-193: The conservation diagnostic and the surrogate dataset generator rebuilt boundary inputs by a rule of their own, not the step's (0.1.0 to 0.3.1).
- MADD-ANO-233: A coupling group under IQN whose iterate left float range before its cap never returned from step() (0.3.0 and 0.3.1).
- MADD-ANO-283: ift_linear_solve with solver="cg" or "gmres" returned a value that was not the solution, with no error, where the floor of lineax's stopping test was zero, underflowed, or was negative or NaN (0.3.1).
- `validate()`'s line for a loop that a coupling group iterates ended `(Gauss-Seidel)` for a `"jacobi"` group too; it now ends `(Jacobi)` there (0.1.0 to 0.3.1).
- MADD-ANO-192: Interface-mapping factories returned an operator for coordinates their formula does not cover (0.1.0 to 0.3.1).
- MADD-ANO-043: run_adaptive / run_adaptive_scan advanced a sub-cycled node by its rate divider times dt on every adaptive step (0.1.0 to 0.3.1).
- MADD-ANO-044: On a multi-rate graph a coupling group's diagnostics, predictor history and IQN-IMVJ warm start came from solves the step discarded (0.1.0 to 0.3.1).
- MADD-ANO-061: run_adaptive advanced its clock by dt_min when it accepted an attempt at dt_min, while the state advanced by the attempted step (0.1.0 to 0.3.1).
- MADD-ANO-074: The adaptive steppers' error norm read integer and boolean state leaves: a bool raised, a uint32 wrapped and a counter moved the step sequence (0.1.0 to 0.3.1).
- MADD-ANO-079: compile() restarted a multi-rate graph's sub-step counter, so a recompile in the middle of a run re-phased the sub-steps each slower node fired on (0.1.0 to 0.3.1).
- MADD-ANO-095: After run_adaptive or run_adaptive_scan the coupling report described only the second kept half step: a first half step stopped at max_iterations was hidden, and converged read True where strict_convergence refused the step (0.1.0 to 0.3.1).
- MADD-ANO-118: run_adaptive accepted a rejected attempt larger than dt_min whenever shrinking it would reach dt_min, where run_adaptive_scan retried at dt_min (0.1.0 to 0.3.1).
- MADD-ANO-125: The multi-rate base timestep's GCD stopped on an absolute 1e-9: nodes at 1e-9 and 2e-9 got a base step of 2e-9 and their clocks drifted apart (0.1.0 to 0.3.1).
- MADD-ANO-160: jax.grad through run_adaptive_scan was 0.0 or NaN where an attempt's step-doubling estimates agreed exactly, as for an accelerated coupling group whose iterate starts at its fixed point (0.1.0 to 0.3.1).
- MADD-ANO-189: The adaptive steppers' error norm was computed in a bfloat16 or float16 state's own dtype: run_adaptive did not return on a float16 state above 65,504 elements, run_adaptive_scan refused any 16-bit state (0.1.0 to 0.3.1).
- MADD-ANO-099: GraphManager.step did not zero-fill the inputs a partial external_inputs left out, and dropped an undeclared node or field name in silence: the node took its own "input missing" branch (0.1.0 to 0.3.1).
- MADD-ANO-120: A node downstream of a cycle that was added to the graph before the cycle's nodes was scheduled ahead of them and read their previous-step output: an edge on no cycle, staggered silently (0.1.0 to 0.3.1).
- MADD-ANO-250: A state field written with another shape after compile() was broadcast into the node that reads it: the shape compile() holds an edge to was asked once (0.1.0 to 0.3.1).
- MADD-ANO-001: LBM GPU segfault on CUDA 12.2 + jaxlib 0.5.1 (0.1.0 to 0.3.1).
- MADD-ANO-020: LBMNode's Zou-He pressure boundary imposed rho_p + S_K instead of the prescribed density (0.1.0 to 0.3.1).
- MADD-ANO-034: LBMNode's wall_mask was dropped by every save/reload: the reloaded graph ran with no walls (0.1.0 to 0.3.1).
- MADD-ANO-004: The sharded wrappers ignored a REST parameter write; PUT /graph/params answered 200 without changing the physics (0.1.0 to 0.3.1).
- MADD-ANO-028: A sharded stencil node on a mesh axis of one device ran with periodic global halos whatever `boundary` said (0.2.0 to 0.3.1).
- MADD-ANO-029: The default `"edge"` halo fill copied a halo two or more cells wide from the shard's first cells instead of repeating its edge cell (0.2.0 to 0.3.1).
- MADD-ANO-033: ShardedStencilNode stepped with dt rounded to float32 under jax_enable_x64 (0.2.0 to 0.3.1).
- MADD-ANO-038: ShardedStencilNode edge-filled a sharded static's global halos under a periodic wrapper (0.2.1 to 0.3.1).
- MADD-ANO-040: The sharded wrappers placed a domain integral carried in the state like a grid field: an integral-emitting node could not run in a graph (0.2.1 to 0.3.1).
- MADD-ANO-041: halo_exchange ignored a per-axis boundary key naming no exchanged mesh axis, and that axis took the edge fill (0.2.0 to 0.3.1).
- MADD-ANO-056: ShardedStencilNode read the block size it hands update_padded in shard_info off a domain integral carried in the state, so a sharded HeatNode carrying a vector or per-shard energy integral never closed its right rod end (0.2.1 to 0.3.1).
- MADD-ANO-065: ShardedUnstructuredNode.gather_global reshaped a domain integral listed in state_fields() as a per-cell field: a scalar raised, and values one per shard came back reordered (0.3.0 and 0.3.1).
- MADD-ANO-067: ShardedStencilNode summed a domain integral over a mesh axis its axis_map leaves unused, counting every block once per device along it (0.2.1 to 0.3.1).
- MADD-ANO-124: sharded_gmres's loop backend returned NaN with converged=False whenever restart exceeded the system size, the default 50 against any system of fewer than 50 unknowns (0.3.0 and 0.3.1).
- MADD-ANO-155: sharded_cg returned zeros with converged=True for a right-hand side with a NaN or infinite entry, and ift_linear_solve returned zeros with no error (0.3.0 and 0.3.1).
- MADD-ANO-269: sharded_cg's default call and backend='lineax' answered an operator that returns NaN or infinity with zeros, converged=True and iters=0 (0.3.0 and 0.3.1).
- MADD-ANO-272: At the top of the dtype's range sharded_cg and sharded_gmres vouch for a value that is not the solution: zeros from the CG loop until 0.4.0 for a right-hand side whose norm overflows, and in 0.4.0 infinities on every route where the solution itself overflows (0.3.0 and 0.3.1).
- MADD-ANO-281: With a floor of zero for lineax's entrywise test (an explicit atol=0.0, or rtol=0) sharded_cg and sharded_gmres read converged=True on their lineax routes after a step or a cycle, at a residual of 0.4 ||b|| (0.3.0 and 0.3.1).
- MADD-ANO-006: Non-finite numbers are written as non-standard JSON tokens (0.1.0 to 0.3.1).
- MADD-ANO-097: USD baseDt and the FMU default step read a _base_dt attribute that nothing set: USDWriter wrote 0.01 for every graph, and save_graph_to_usd gave the smallest node timestep rather than the graph's step (0.1.0 to 0.3.1).
- MADD-ANO-138: load_state cast a checkpoint value to the graph's dtype with no check: a float64 1e39 loaded into a float32 field as inf, and -1e-50 as -0.0 (0.1.0 to 0.3.1).
- MADD-ANO-251: A node's delta_t written after add_node was neither followed nor reported: the graph kept its timestep and to_dict() wrote the new one (0.1.0 to 0.3.1).
- MADD-ANO-078: build_model_description of a real graph declared no FMU inputs and a default step of 1e-3, whatever the graph's inputs and timestep (0.3.0 and 0.3.1).
- MADD-ANO-273: ModelDescription.to_xml wrote the unit attribute on integer and Boolean variables: a modelDescription.xml that FMI 3.0's schema refuses, with nothing said at export (0.3.0 and 0.3.1).
- MADD-ANO-024: PUT /graph/params accepted a value a node consumes at construction: reported and saved, never used by the running graph (0.1.0 to 0.3.1).
- MADD-ANO-049: A non-finite value written through PUT /graph/params was stored, then the reply failed with a 500, and every later GET /graph/params for the node failed too (0.1.0 to 0.3.1).
- MADD-ANO-087: PUT /graph/params answered 200 to a value the running node could not step with, and every POST /sim/step after it was a 500 (0.1.0 to 0.3.1).
- MADD-ANO-088: POST /sim/stop answered "stopped" while the runner's thread kept stepping, and a reset after it was overwritten (0.1.0 to 0.3.1).
- MADD-ANO-090: PUT /sim/stride echoed values it did not apply, and DELETE /graph/edges answered 200 for an edge the graph did not have (0.1.0 to 0.3.1).
- MADD-ANO-091: A state or render WebSocket stream never noticed a client that left after the simulation stopped, so the server could not shut down; an ordinary disconnect was logged as an error (0.1.0 to 0.3.1).
- MADD-ANO-106: Concurrent REST requests stepped the graph at once and lost steps silently: 200 concurrent POST /sim/step took about 60, every one answered 200, and the streams counted all 200 (0.1.0 to 0.3.1).
- MADD-ANO-107: POST /surrogate/train reset the live simulation to its initial state, and its reply did not say so (0.1.0 to 0.3.1).
- MADD-ANO-108: After POST /checkpoint/load the state streams served the state from before the load, at the old step's time, until the next step (0.1.0 to 0.3.1).
- MADD-ANO-109: /ws/state/binary kept the frame layout of a node that had been replaced over REST: values cut off, or phantom zeros, in frames of the advertised length (0.1.0 to 0.3.1).
- MADD-ANO-110: A runner whose thread had died was reported running: /sim/pause and /sim/resume answered 200, /sim/start 409 "already started", and /sim/reset was_running true (0.1.0 to 0.3.1).
- MADD-ANO-111: The real-time runner burst through the time it had been paused: about 1.5 s simulated in the 0.4 s after a 1 s pause of a real-time run (0.1.0 to 0.3.1).
- MADD-ANO-112: /ws/state encoded each frame on the event loop, once per client, for any number of clients: a large state stalled every other request (0.1.0 to 0.3.1).
- MADD-ANO-132: POST /sim/profile reset the live simulation and left it at the profiler's last step, and the streams counted the profiler's steps (0.2.0 to 0.3.1).
- MADD-ANO-133: A reset, a state write or a node removed over REST was not shown by the streams: they served the state from before it until the next step (0.1.0 to 0.3.1).
- MADD-ANO-139: PUT /sim/stride reset a value left out of the query to 1: a call naming only steps_per_frame reset the relay's stride, and the reverse (0.1.0 to 0.3.1).
- MADD-ANO-164: A parameter write to a HybridNode was answered and then lost: the hybrid held a copy of its physics node's params, so PUT /graph/params echoed the new value and the step kept the old one (0.1.0 to 0.3.1).
- MADD-ANO-166: POST /sim/profile answered 500 for a graph that could not compile with any error but a RuntimeError (0.2.0 to 0.3.1).
- MADD-ANO-180: PUT /graph/state wrote every field as float32 whatever its dtype, and parsed text and booleans as numbers (0.1.0 to 0.3.1).
- MADD-ANO-183: POST /sim/start on a graph with no nodes answered 'started' and its runner thread died on its first frame (0.1.0 to 0.3.1).
- MADD-ANO-184: A request refused by its model for a NaN or infinite value answered 500 (0.1.0 to 0.3.1).
- MADD-ANO-191: Apps built at once in several threads left the process's warnings filters changed: a silenced warning was raised, or every UserWarning was ignored from then on (0.1.0 to 0.3.1).
- MADD-ANO-209: A refusal that echoed a surrogate from the request answered 500 (0.1.0 to 0.3.1).
- MADD-ANO-286: WebRTC streaming never negotiated: SelkiesSession sent a viewer no answer and no ICE, so no video ever reached a viewer (0.1.0 to 0.3.1).
- MADD-ANO-287: The cloud container streamed nothing: its entry point never started the WebRTC session, and its API could add no node to build a graph (0.1.0 to 0.3.1).
- MADD-ANO-077: Swapping a surrogate in with replace_node, or out with POST /surrogate/deactivate, re-added each edge without its additive flag and units: an additive coupling became an overwriting one, silently (0.1.0 to 0.3.1).
- MADD-ANO-117: profile_graph's per-node timing called each node with no boundary inputs, so a node that reads an input raised KeyError (0.1.0 to 0.3.1).
- MADD-ANO-130: The anomaly-registry validator raised TypeError on a list-valued severity, safety_relevance or resolution_status instead of reporting the entry (0.1.0 to 0.3.1).

### Verification

- How 0.4.0 was verified (the claims inventories, the property, differential and metamorphic tests, the compliance gates and the CI lanes) is in the development record, with every audit round.

### Security

- **The HTTP API requires a bearer token unless bound to loopback** (MADD-ANO-051): set `MADDENING_API_TOKEN` and `SimulationServer(bind_host=)`; `JobConfig.ports` defaults to `[]`.
- **The API checks the request's origin** (MADD-ANO-076), and a loopback-bound server asks the token of a non-loopback peer and of a forwarded request.
- **Checkpoint paths are confined to `checkpoint_root`** (MADD-ANO-052).
- **ZeroMQ sockets bind to loopback and demand CURVE off it** (MADD-ANO-015): set `MADDENING_TRANSPORT_TOKEN` on both ends.
- **The WebRTC signaling server checks the client's token** (MADD-ANO-053).
- **The FMU bridge no longer unpickles**: FMU-state blobs are `npz` (MADD-ANO-055).
- MADD-ANO-086: A REST request could make the server allocate without bound: before 0.4.0 POST /graph/nodes built any size, and 0.4.0's size checks ran only after the node was built (0.1.0 to 0.3.1).
- MADD-ANO-092: POST /sim/run and POST /surrogate/train took any size: one request could hold a worker indefinitely or name more memory than the host has (0.1.0 to 0.3.1).
- MADD-ANO-134: A JAX trace started over REST had no bound: left running, it grew the server's memory by kilobytes per step until stopped (0.2.0 to 0.3.1).
- MADD-ANO-185: A checkpoint load decompressed every archive member before it checked any name or shape: a small archive cost as much memory as its members declared (0.1.0 to 0.3.1).
- MADD-ANO-274: A 422 sent the refused value back whole: a request body the server refused was encoded and returned at its full size (0.1.0 to 0.3.1).
- MADD-ANO-288: The cloud entry point logged MADDENING_CLOUD_CONFIG verbatim at INFO, so a TURN username and credential in stream_config.ice_servers were written to the container's log (0.1.0 to 0.3.1).

### Known Anomalies

- 54 registry entries are reachable in 0.4.0: 43 open and 11 partly resolved. The release notes' "Known anomalies" section lists each with what to do; the registry is `docs/validation/known_anomalies.yaml`.

## [0.3.1] - 2026-06-22

An **experimental-pilot** point release: it ships one small, self-contained,
additive primitive — `ift_linear_solve` — early, for a downstream project
building on MADDENING that needs a `pip install`-able differentiable linear
solve now rather than waiting for the 0.4/M3 `AdaptiveNode` milestone.

```{note}
**Experimental pilot.**  `ift_linear_solve` is tagged
`@stability(EXPERIMENTAL)` in v0.3.1 — validated but not frozen.  It is
promoted to `@stability(STABLE)` when the `AdaptiveNode` framework lands in
0.4 (STACK_V1 §M3).  Pin against it only for short-lived / pilot work.
```

### Added

- **`maddening.core.solver_utils.ift_linear_solve`** (`@stability(EXPERIMENTAL)`)
  — a thin wrapper over `lineax.linear_solve`: any node solving a linear system
  in `update()` gains a clean differentiable path (lineax's native autodiff
  propagates the linear-solve adjoint, so no MADDENING-level `custom_vjp`).
  Backends `'gmres'` (default, restart clamped to `min(N, 50)`), `'cg'` (SPD),
  `'dense'`.  Optional `preconditioner` kwarg passes through to lineax with the
  array portion `stop_gradient`'d.  Verified against `BCOO`-backed operators.
  Purely additive; no change to any existing surface.

### Dependencies

- New **`[ift]` extra** (`lineax>=0.0.7`).  `lineax` is lazy-imported, so the
  base install is unchanged; `ift_linear_solve` callers install
  `maddening[ift]`.  (`lineax` was already a `[dev]`/`[ci]` dependency for the
  in-tree coupling-solver path; this promotes it to a user-facing extra, an item
  previously scheduled for 0.4.)

## [0.3.0] - 2026-06-10

v0.3.0 is the M2 "redesigns" milestone (STACK_V1 §3).  See
`docs/release_notes/v0.3.0.md` for the narrative summary,
`docs/developer_guide/stability_report.md` for the up-to-date
`@stability` audit table.

### Added

- **`maddening.fmi`** subpackage — FMI 3.0 substrate (§A1).
  `ModelDescription`/`build_model_description()` emit FMI 3.0
  `modelDescription.xml` from a compiled `GraphManager` +
  the `@stability` registry; `get_directional_derivative()` wraps
  `jax.jvp` / `jax.vjp` behind a `fmi3GetDirectionalDerivative`-
  shaped API; `serialize_fmu_state`/`deserialize_fmu_state` round-
  trip graph state through a schema-token-validated handle;
  `FmuSidecar` is a Python reference implementation of the ZMQ
  sidecar protocol the FMU's C wrapper (v0.4.0 deliverable) will
  marshal into.  All tagged `@stability(EVOLVING)`.
- **`maddening.cloud.multigpu.iterative_solver`** — `sharded_cg` /
  `sharded_gmres` (§A5).  Wrap user-supplied sharded matvecs;
  lineax-backed default with a hand-rolled `lax.while_loop` /
  `lax.fori_loop` fallback for when lineax misbehaves with
  `shard_map`.  Both tagged `@stability(STABLE)` — these are
  the surface MIME's v0.5.0 FVM PISO pressure correction calls.
- **`maddening.cloud.multigpu.sharded_unstructured.ShardedUnstructuredNode`**
  + **`maddening.cloud.multigpu.halo_unstructured`** (§A6).
  Graph-partitioned sharded execution; sparse halo exchange via
  `lax.all_to_all`; new `StaticArray(replication="partition")`
  variant with `partition_assignment` plumbing.  Toy 16-cell test
  + 1024-cell smoke + cross-cutting `sharded_cg` + Poisson test +
  `_MockFVMFluidNode` contract-stress-test (the v0.4.0 commitment
  gate).  Tagged `@stability(STABLE)`.
- **`maddening.usd.live_stage.LiveStage`** — generic per-timestep
  USD writer pulled out of MIME (§A3).  Domain-neutral stage
  creation, dynamic-prim registry, batched `Sdf.ChangeBlock`
  update loop, materials / dome lights / ground planes.
  `make_translate_updater` / `make_translate_orient_updater`
  cover the common cases without subclassing.  New non-MIME
  `live_stage_bouncing_ball_demo` example exports a time-sampled
  `.usda` runnable in MICROROBOTICA.  Tagged
  `@stability(EVOLVING)`.
- **IFT coupling-solver redesign merged** (§A4).  `solver="ift"` on
  `CouplingGroup`, matrix-free lineax GMRES backward, `acceleration`
  values including `aitken` and `iqn-imvj`, per-step IFT × IQN-IMVJ,
  embedded coupling groups, Literal-typed field validation.
- **`@stability` decorator + registry** (§A2).  `StabilityLevel`
  gains `EVOLVING` and `INTERNAL` (plus existing `STABLE`,
  `EXPERIMENTAL`, `PROVISIONAL`, `DEPRECATED`).  First-wave audit
  applied to the v0.3.0 plan's named surfaces.  Auto-generated
  `docs/developer_guide/stability_report.md` is now part of the
  docs build.
- **Choice-criteria developer-guide page** —
  `docs/developer_guide/sharding_topology.md` covers
  structured-vs-unstructured choice + partition-assignment handoff
  pattern + performance trade-offs + v0.4.0 commitment.

### Changed (breaking)

- **`SimulationNode.requires_halo`** (property + compat shim)
  removed (§B1).  Subclasses overriding `requires_halo` instead of
  `halo_width()` raise `MigrationError` at class-definition time
  (was `FutureWarning` in v0.2).
- **`ShardedNode`** (deprecated alias for `ShardedPointwiseNode`)
  removed (§B2).  Use `ShardedPointwiseNode` for pointwise
  sharding or `ShardedStencilNode` for stencil sharding.
- **Bare arrays in `static_data`** now raise `MigrationError`
  immediately (§B3); the v0.2.1 `FutureWarning`-coerce path is
  gone.  Wrap explicitly in `StaticArray(value=..., replication=...)`.
- **`EdgeValidationWarning` / `ShapeMismatchWarning` /
  `DtypeMismatchWarning`** deprecated aliases removed from
  `maddening.warnings` (§B4).  `UnitMismatchWarning` now roots at
  `UserWarning` directly.
- **`maddening.surrogates.{checkpoint,trainer,callbacks,physics_losses}`**
  legacy top-level paths removed (§B5).  The source files
  physically moved to `surrogates/weights/checkpoint.py` and
  `surrogates/training/{trainer,callbacks,physics_losses}.py`.
  Importing from the legacy paths raises `ModuleNotFoundError`.

### Stability audit

- 29 public surfaces tagged via `@stability` (13 stable, 7
  evolving, 9 experimental).  See
  `docs/developer_guide/stability_report.md` for the current
  table.  The v0.3.0 §A6 contract is `@stability(STABLE)`-ready
  per the hard v0.4.0 commitment ("sharded FVM in MIME v0.5.0").

### Dependencies

- Added `httpx2>=2.0` to the `[ci]` and `[dev]` extras (§C5).
  Starlette's testclient auto-detects httpx2 and uses it
  preferentially, closing the v0.2.1
  `StarletteDeprecationWarning`-ignore loop.  The corresponding
  filterwarning is removed from `pyproject.toml`.

## [0.2.1] - 2026-05-30

A patch release that closes the three v0.2 deferred items
(`V0.2_PROGRESS.md` "Deferred" block): sharded `StaticArray` runtime
slicing, the pre-announced edge-validation warning→error flip, and
the `compile()` advisory-noise cleanup.  The first item unblocks
MIME's multi-GPU `IBLBMFluidNode` sharding (load-bearing for the
de Boer step-out replication, MIME M1).

```{warning}
**Semver carve-out.**  v0.2.1 includes one breaking change under
strict semver — the edge-validation warning→error flip described
below.  This was pre-announced in v0.2.0 release notes and held
on the deprecation calendar; we ship the flip as a PATCH because
(a) the change was published in advance, (b) the migration path
is documented in
[`docs/developer_guide/edge_validation_migration.md`](docs/developer_guide/edge_validation_migration.md),
and (c) the deprecated ``*Warning`` aliases stay importable
through v0.2.x.  If your CI pins ``maddening<0.3``, expect this
change; the aliases are removed in v0.3.
```

### Added
- `domain_integral_fields()` method on `SimulationNode`: declares
  output keys that should be `lax.psum`-reduced across the device
  mesh after `update_padded` (e.g. drag force, drag torque on a
  sharded immersed-boundary node).  Default returns an empty set —
  pure additive, no behavioural change for existing nodes.
- `static_padded` and `shard_info` keyword-only optional parameters
  on `SimulationNode.update_padded`.  The wrapper passes the
  per-device + halo-padded slab of each sharded `StaticArray` via
  `static_padded`, and per-axis `(global_offset, local_extent)` via
  `shard_info` (the offset is a traced JAX scalar, usable in
  `dynamic_slice` but not in Python integer slicing).
- Runtime slicing for `StaticArray(replication="shard", shard_axis=K)`
  under `ShardedStencilNode`.  v0.2.0 stored `shard_axis` as
  metadata only; v0.2.1 actually materialises the per-device slice
  via `jax.device_put` + `NamedSharding`, halo-exchanges it with
  `boundary="edge"`, and delivers it as `static_padded` to
  `update_padded`.  Acceptance test in
  `tests/cloud/multigpu/test_sharded_static_data.py`.
- `EdgeValidationError`, `ShapeMismatchError(EdgeValidationError)`,
  `DtypeMismatchError(EdgeValidationError)` in
  `maddening.warnings` — the new error path for the validation flip.
- `BaseExceptionGroup` / `ExceptionGroup` re-export in
  `maddening.warnings` (builtin on 3.11+; `exceptiongroup` backport
  on 3.10).

### Changed (breaking — see semver carve-out above)
- `GraphManager.compile()` raises `ExceptionGroup("edge validation failed", [...])`
  on shape/dtype mismatches that previously emitted
  `ShapeMismatchWarning` / `DtypeMismatchWarning`.  All mismatches
  detected in a single `compile()` are aggregated into one group so
  callers can see every problem at once.  Catch `EdgeValidationError`
  (or the subclasses) via `except*` on 3.11+, or via explicit
  isinstance iteration on 3.10.  `UnitMismatchWarning` is
  **unchanged** — units are advisory by contract.

### Deprecated (kept as aliases for one release cycle, removed in v0.3)
- `ShapeMismatchWarning`, `DtypeMismatchWarning`, `EdgeValidationWarning`
  classes remain importable from `maddening.warnings` so downstream
  `pytest.warns(...)` references still resolve; nothing in MADDENING
  emits them in v0.2.1.  The v0.3 plan's compat-hygiene bucket
  removes these aliases.

### Fixed
- Single-node graphs (the quickstart shape) no longer emit a
  `"node 'X' is disconnected"` `UserWarning` from
  `GraphManager.compile()`.  The disconnected advisory now requires
  `len(node_names) > 1`.
- The uncovered "cycle detected" advisory moved from
  `warnings.warn(UserWarning)` to `logging.getLogger(__name__).info(...)`
  + an `INFO:`-prefixed entry in `validate()`'s issue list.  Cycles
  are handled correctly via back-edge staggering — surfacing them
  through the warning system was noise for downstream
  `filterwarnings=["error"]` configs.

### Dependencies
- Added `"exceptiongroup; python_version < '3.11'"` to base
  dependencies.  Provides the `BaseExceptionGroup` / `ExceptionGroup`
  builtins on Python 3.10.

### Verification
- Full MADDENING test suite: 1680 passed, 3 skipped (1 deselected
  via `-m "not slow"`).  Slow-marked tests deferred to a longer
  pre-release pass.
- Sharded `StaticArray` acceptance: 4-device CPU virtual-device mesh
  bit-compat with the single-device baseline (atol=0 on state, atol=1e-5
  on the `lax.psum` integral), 50-step multi-step convergence,
  construction-time validation (`shard_axis` must match the wrapper's
  spatial axes; nodes with sharded statics must accept `static_padded`
  on `update_padded`), `shard_info` delivery.
- Edge-validation flip: 15/15 `tests/core/test_edge_validation.py`
  green; aggregation test confirms shape + dtype errors raise in one
  `ExceptionGroup` alongside a `UnitMismatchWarning`.

## [0.2.0] - 2026-05-20

See [`docs/release_notes/v0.2.md`](docs/release_notes/v0.2.md) for the
narrative release notes; the itemized changes follow.

### Added
- Coupling convergence infrastructure: per-field mixed atol/rtol norm (`convergence_norm="mixed"`), convergence diagnostics (`diagnostics=True`), Aitken delta-squared acceleration (`acceleration="aitken"`), fixed under-relaxation (`acceleration="fixed"`), and Jacobi iteration mode (`iteration_mode="jacobi"`)
- `coupling_acceleration` module with standalone JAX-traceable residual norms, state flatten/unflatten, and acceleration functions
- `GraphManager.coupling_diagnostics()` method for retrieving iteration counts and final residuals
- IQN-ILS quasi-Newton coupling acceleration (`acceleration="iqn-ils"`) with Aitken fallback, pre-allocated matrices for fori_loop compatibility, and automatic column management
- Subcycling within coupling groups (`subcycling=True`) for mixed-timestep coupling with linear/constant boundary interpolation
- Spatial interpolation map factories in `interface_mapping` module: `nearest_neighbor_1d`, `linear_interpolation_1d`, `rbf_interpolation` (4 kernels), `conservative_projection_1d`
- `auto_couple()` and `add_coupling_group()` accept `**kwargs` forwarded to `CouplingGroup`
- Coupling examples: acceleration comparison, Jacobi vs Gauss-Seidel, subcycling, spatial interpolation, convergence diagnostics
- IQN-ILS/IMVJ auto interface-field detection: `flatten_coupled_state` accepts `fields` parameter to accelerate only coupling-edge fields, reducing V/W matrix size for nodes with many internal DOFs
- IQN-IMVJ multi-timestep Jacobian reuse (`acceleration="iqn-imvj"`, `jacobian_reuse=N`): warm-starts V/W from previous timestep for faster convergence
- Interface residual convergence norm (`convergence_norm="interface"`): checks coupling-edge values between iterations instead of full state change
- Quadratic subcycling boundary interpolation (`boundary_interpolation="quadratic"`): Lagrange interpolation through three successive iteration values
- Waveform relaxation for subcycled groups (`waveform_iterations=N`): repeats coupling block to improve boundary data quality
- Flux-based coupling: `SimulationNode.compute_boundary_fluxes()` exposes derived quantities (heat flux, spring force) consumable via edges; `SimulationNode.boundary_input_spec()` declares expected inputs with `BoundaryInputSpec` descriptors
- `EdgeSpec.additive` flag: edges with `additive=True` accumulate values instead of overwriting, enabling multi-source force/flux coupling
- `coupling_helpers` module: `add_value_coupling`, `add_flux_coupling`, `add_dirichlet_neumann_pair`, `add_symmetric_value_coupling`, `add_robin_coupling`, `check_conservation`
- `BoundaryInputSpec` dataclass and `boundary_input_spec()` on HeatNode, SpringDamperNode, BallNode, RigidBody2DNode
- `compute_boundary_fluxes()` on HeatNode (left/right heat flux) and SpringDamperNode (spring force)
- Flux coupling demo and node authoring guide sections on flux coupling patterns
- `TransformRegistry` with `@register_transform` decorator for named, serializable edge transforms; built-in transforms (`extract_first`, `extract_last`, `negate`, `scale`, `identity`); `GraphManager.add_edge` accepts string transform names
- `scripts/check_transforms.py` CI validation script for string transform references
- Gradient health audit: verified `jax.grad` finite through 1000-step coupled rollouts for springs, heat rods, and multi-physics systems
- Parameter recovery baseline: gradient-based recovery of spring stiffness and damping from trajectory data (inline physics, proving differentiability concept)
- Interface DOF awareness: `interface_dof_indices()` and `compute_interface_correction()` on SimulationNode; coupling system re-applies interface values after node update, fixing the DD coupling "cold lock" where HeatNode's Dirichlet BC enforcement prevented heat transfer
- Coupling iteration predictors (`predictor="linear"` or `"quadratic"` on CouplingGroup): extrapolates initial guess from previous timesteps' converged states, reducing iteration count
- `tune_coupling_params()` utility for grid-search optimization of coupling parameters (tolerance, max_iterations, acceleration)
- `HybridNode` wrapper: composes a physics node with an additive correction function; `generate_correction_data()` for training integration error correctors
- `derivatives()` method on SimulationNode with implementations on BallNode, SpringDamperNode, HeatNode; `integrators` module with `euler_step`, `heun_step`, `rk4_step` and convenience `integrate_node()`
- `calibrate()` utility for gradient-based parameter recovery from reference trajectories using `jax.grad`
- Implicit node support: `implicit_residual()` on SimulationNode with fixed-count Newton iteration via `jax.lax.fori_loop`; implemented for SpringDamperNode and HeatNode; unconditionally stable for stiff problems
- OpenUSD integration: codeless schemas (`MaddeningSimulationGraph`, `MaddeningNode`, `MaddeningEdge`, `MaddeningCouplingGroup`, `MaddeningExternalInput`), `USDWriter` for time-sampled state output, `save_graph_to_usd()` / `load_graph_from_usd()` for full graph serialization, late-registration guard with RuntimeError
- HeatNode 4th-order FD stencil (`stencil_order=4`), non-uniform grid support (`grid_points` parameter)
- 2D spatial interpolation: `nearest_neighbor_2d()`, `rbf_interpolation_2d()` in interface_mapping
- USD geometry reader: `load_grid_from_usd()`, `create_vessel_phantom()` (Y-shaped bifurcating vessel)
- `geometry_source` attribute on SimulationNode for USD-initialized nodes
- Vessel bifurcation coupling example: three HeatNodes initialized from USD geometry, coupled at Y-junction
- `HistoryViewer3D.add_curve_tube()`: render 3D centerline tubes colored by scalar fields (vessels, pipes, rods)
- `HistoryViewer3D.add_line_plot()`: render 1D fields as 3D line plots (temperature profiles, wave solutions)
- `viewer_from_usd()`, `viewer_from_usd_with_geometry()`, `render_usd_frame()`: bridge USD results data to the general-purpose HistoryViewer3D for interactive replay and screenshots
- USD tests skip gracefully when `usd-core` is not installed (CI compatibility for Python 3.10/3.11)
- `LBMNode`: general 3D Lattice Boltzmann on boolean mask domains with D3Q19/D2Q9 lattices, Zou-He pressure BCs, Guo forcing, runtime clot injection via `wall_mask_update`
- `lbm_geometry.voxelize_vessel()`: analytical Y-bifurcation voxelizer parametric by vessel geometry
- `RigidBodyNode`: full 6DOF rigid body (quaternion orientation, diagonal inertia, DOF constraints). `RigidBody2DNode` deprecated with thin wrapper.
- `HeartPumpNode`: 2-element Windkessel model with pulsatile cardiac output, configurable heart rate / stroke volume / resistance / compliance, bidirectional pressure coupling
- `PyVistaLiveRenderer`: real-time 3D visualization backend with timer callbacks, pause/resume/speed keyboard controls
- Vessel bifurcation live example: real-time simulation + USD recording + PyVista visualization + heat pulse injection demo
- Vessel flow server: FastAPI server with HeartPump+LBM coupling, REST endpoints for heart rate / resistance / clot injection, WebSocket live vitals streaming, browser UI with pressure waveform chart

- Cloud module (`maddening.cloud`): `StreamingSession` ABC and `StreamConfig`/`StreamInfo`/`QualityPreset`/`GPUFramebuffer` data types for WebRTC viewport streaming; `MockStreamSession` for zero-dep testing; HMAC-SHA256 session token auth; `SelkiesSession` GStreamer/WebRTC implementation (requires PyGObject)
- `CloudSession` state machine with SkyPilot VM orchestration, typed health probes (`HealthProbeError` with stage attribution), `CloudReadyResult` with per-stage pass/fail, `MockCloudSession` for testing; preemption detection with configurable policy (CHECKPOINT/FAILOVER/ABORT)
- `SelkiesRenderer(Renderer)`: wraps inner renderer + `StreamingSession`, auto-detects GPU/CPU framebuffer path, emits `PerformanceWarning` on CPU fallback
- Multi-GPU Jacobi coupling: `create_device_mesh()`, `assign_nodes_to_devices()` with coupling co-location, `build_sharded_jacobi_pass()` for distributed node updates; `GraphManager.enable_multigpu()` method
- Cloud container: `docker/Dockerfile.cloud` (CUDA + GStreamer + MADDENING), `entrypoint.py` with JSON config deserialization
- Cloud API endpoints on `SimulationServer`: `POST /cloud/launch`, `GET /cloud/status`, `POST /cloud/teardown` (unconditionally registered, returns 501 if unconfigured)
- `CloudLauncher`: user-facing cloud job orchestration with `CloudJob` handle, `JobConfig` YAML loading, `CostPolicy` cost guards, credential context manager with cleanup, and `CloudJob.from_cluster_name()` reconnect
- `CloudProvider` ABC with `RunPodProvider` and `LambdaLabsProvider` (stub); per-provider credential file management with write/delete lifecycle
- Cloud examples consolidated under `src/maddening/examples/cloud/`: `01_validate.py` (dry-run), `02_runpod_launch.py` (real launch), config templates
- Restructured package extras: per-provider cloud (`runpod`, `lambda`, `aws`, `gcp`), hardware acceleration (`cuda12`, `tpu`), task bundles (`server`, `client`), combo (`cloud`, `cloud-all`)
- Consistent import guards across all optional dependencies: missing extras now raise `ImportError` with the exact `pip install maddening[extra]` command
- User guide: `docs/user_guide/installation.md` (full install reference), `docs/user_guide/quickstart.md` (5-minute intro)
- `CostPolicy.spot_fallback`: when spot instances are unavailable, auto-retry on-demand (subject to same cost guards); configurable via job config YAML
- `retry_until_up` on all SkyPilot launches to handle transient SSH/provisioning failures
- Concise error message for spot unavailability (truncates verbose per-region table); other errors preserved in full
- Multi-GPU Phase 1: `enable_multigpu()` wired into `_build_step_fn()` — Jacobi coupling uses `jax.device_put` for per-node device placement; correctness validated (single step, 100 steps, `lax.scan` all match non-sharded)
- Multi-job architecture: `Coordinator` (ZMQ ROUTER-based rendezvous with registration, topology broadcast, heartbeat monitoring), `CloudGroup` (provision rank-0 first, inject `COORDINATOR_ADDR` into workers, `teardown_all` / `teardown_one` with `ISOLATE` mode), `SubgraphSpec` + `GroupConfig`
- Cloud examples organized into subdirectories: `config/`, `launch/`, `server/`, `streaming/`, `multigpu/`, `multijob/`
- `requires_halo` abstract property on `SimulationNode` — every node must declare whether it needs halo exchange for sharding
- `ShardedNode` wrapper for data-parallel distribution of pointwise nodes across device meshes; rejects stencil nodes automatically
- `WorkerClient` for multi-job rendezvous: `register_and_wait()`, heartbeat, shutdown/peer_dead callbacks
- Validated 2-VM multi-job rendezvous on RunPod (coordinator + worker across VMs via ZMQ)
- Real multi-GPU benchmark on 2xRTX4090: correctness validated, JIT fusion behaviour documented
- Core reorganized into `core/coupling/`, `core/simulation/`, `core/compliance/` subpackages (backward compatible via `core/__init__.py` re-exports)
- Docker image `ghcr.io/microrobotics-simulation-framework/maddening-cloud:latest` — pre-built with JAX CUDA, GStreamer, ZMQ, FastAPI; set as default `container_image` in `JobConfig`
- CycloneDX SBOM generation (`sbom.json`) for IEC 62304 SOUP compliance
- 3-D pencil/slab halo decomposition for stencil nodes: `SimulationNode.halo_width(axis) -> dict[int, int]` per-axis halo contract (supersedes the boolean `requires_halo`), an `update_padded` entry point, and a `halo_exchange` primitive built on `shard_map`/`ppermute`
- `ShardedStencilNode` for halo-resident stencil distribution across device meshes; `ShardedNode` renamed to `ShardedPointwiseNode` (old name kept as a deprecated alias)
- `LBMNode` D3Q19/D2Q9 halo-aware streaming (`_stream_padded`); sharded LBM verified mass-conserving on 2×4 and 4×4 pencil meshes
- Static-data channel: optional `SimulationNode.static_data` property for non-evolving per-node arrays (meshes, wall masks, lookup tables), baked into JIT-compiled HLO as constants instead of threaded through every step; `StaticArray` typed wrapper carries a `replication` / `shard_axis` policy; `static_data_hash()` drift check triggers a recompile when a node's static_data shape changes (e.g. after `replace_node`)
- Compile-time edge validation: `GraphManager.compile()` walks every edge and surfaces `ShapeMismatchWarning`, `DtypeMismatchWarning`, and `UnitMismatchWarning` (all subclasses of `EdgeValidationWarning`); an edge `transform=` suppresses the check
- `BinaryStateEncoder` field subscriptions (`fields={node: [field, ...]}`) and payload compression (`compression="zstd"` or `"zstd+xor"`); the `/ws/state/binary` subscribe message and ZMQ `NetworkRelay` accept the same `fields=` / `compression=` parameters
- `AWSProvider` and `GCPProvider` join `RunPodProvider` and `LambdaLabsProvider` (promoted out of stub status) under the shared `CloudProvider` ABC
- Spot-preemption resilience: `make_preempt_snapshot_hook()` snapshots GraphManager state on reclaim; `resume_from_url()` restores it on the replacement VM via `RESUME_FROM_URL`; each snapshot ships a sidecar manifest with `schema_version`, SHA-256, and size, verified on load
- Checkpoint schema versioning: `CHECKPOINT_SCHEMA_VERSION`, a `MIGRATIONS` registry, and `CheckpointVersionError` carrying structured drift fields
- `GraphManager.validate_sharding()` returns a list of `ShardingIssue` records for sharding-spec inconsistencies (does not raise — callers filter by severity and raise)
- Profiler endpoints: `POST /sim/profile` returns a Perfetto-loadable trace; `POST /sim/profile/jax/start|stop` wrap `jax.profiler` XLA capture; `POST /cloud/teardown` snapshots the last trace directory into its response
- `maddening.warnings.MigrationError` — the v0.3 hard-removal raise path for deprecated APIs, paired with the new `FutureWarning`-class advisories
- Surrogates subpackage scaffolding: `surrogates/primitives/`, `surrogates/weights/`, `surrogates/training/`, and `surrogates/replace/` re-export the v0.1 leaf modules ahead of the v0.2.x decoder-zoo pull-over
- `compression` optional dependency (`zstandard>=0.22`), also rolled into the `server`, `ci`, and `all` extras

### Fixed
- Subcycling dividers were inverted: fast nodes now correctly take multiple sub-steps while slow nodes take one step
- Coupling diagnostics were lost in multi-rate graphs when step counter overwrote `_meta`
- `maddening.compliance` namespace with schema types, anomaly validator, and CLI
- `NodeMeta` dataclass with `hazard_hints`, `validated_regimes`, `implementation_map` fields
- `StabilityLevel` and `UQReadiness` enums
- `@verification_benchmark` decorator and `ValidationBenchmark` registry
- `@stability` decorator (identity decorator; functional machinery in Phase 4)
- `HealthCheckNode` for execution-layer fault detection
- `NodeMeta` attached to all existing nodes (BallNode, TableNode, SpringDamperNode, RigidBody2DNode, HeatNode, LBMPipeNode)
- `AuditLogger` with `NullSink` and `JSONFileSink`
- `SimulationProvenance` for reproducibility tracking
- `UncertaintySpec` and `UncertainParameter` for UQ interface
- Regulatory documentation: `intended_use.md`, `downstream_integration.md`, `iec62304_mapping.md`, `eu_mdr_guidelines.md`, `mdcg_2019_11.md`
- `known_anomalies.yaml` with MADD-ANO-001 and MADD-ANO-002
- `soup_package.md` (skeleton)
- `SECURITY.md`, `CONTRIBUTING.md`, `CITATION.cff`
- Algorithm guide template and HeatNode algorithm guide
- `scripts/check_anomalies.py`, `scripts/check_impl_mapping.py`, and `scripts/check_citations.py`
- GitHub issue template for anomalies
- Developer guide: `docs/developer_guide/` with `node_authoring.md`, `documentation_standards.md`, `testing_standards.md`
- Bibliography citation system: Pandoc-style `[@Key]` syntax with CI validation
- Claude skill `.claude/skills/commit-and-push/` for commit/push compliance checklist
- Migrated to `src/` layout with hatchling build backend
- Reorganized tests into subdirectories: `core/`, `nodes/`, `surrogates/`, `api/`, `viz/`, `compliance/`, `verification/`

### Changed
- Build backend: setuptools → hatchling
- Package layout: flat → src/
- `pyproject.toml` URLs updated to Microrobotics-Simulation-Framework org
- `SimulationNode.__init_subclass__` now emits a `FutureWarning` (was `DeprecationWarning`) when a subclass overrides the legacy `requires_halo` instead of `halo_width`

### Deprecated
- `SimulationNode.requires_halo` — superseded by `halo_width()`; a default-implemented compat shim remains until v0.3 and emits a `FutureWarning` when a subclass overrides it
- `ShardedNode` — renamed to `ShardedPointwiseNode`; the old name remains as a deprecated alias until v0.3

### Verification
- 512+ existing tests pass after restructure
- New compliance test suite validates all Phase 0-4 artifacts
- 1613 tests pass on the default CPU lane (`pytest`); the slow lane is opt-in (`pytest -m 'slow or not slow'`)

### Security
- Cloud snapshot manifests are SHA-256 verified on load; tampering or schema-version drift raises `CheckpointIntegrityError` / `CheckpointVersionError`
- WebRTC streaming sessions authenticate with HMAC-SHA256 tokens

### Known Anomalies
- MADD-ANO-001: LBM GPU segfault on CUDA 12.2 + jaxlib 0.5.1 (open, context_dependent)
- MADD-ANO-002: HeatNode CFL stability not enforced at runtime (open, context_dependent)

## [0.1.0] - 2025-03-01

### Added
- Initial release: modular simulation framework with functional state pattern
- Core: GraphManager, SimulationNode ABC, EdgeSpec, scheduling, coupling, adaptive timestepping, parameter sweeps, checkpoint/restore
- Nodes: BallNode, TableNode, SpringDamperNode, RigidBody2DNode, HeatNode, LBMPipeNode
- Surrogate framework: SurrogateArchitecture ABC, SurrogateNode, SurrogateTrainer, DatasetGenerator, architectures (MLP, DeepONet, SDeepONet, FNO)
- Visualization: matplotlib, terminal, PyVista, pygfx backends; ZMQ network transport
- API: FastAPI server with REST, WebSocket (JSON + binary), server-side rendering
- 545+ tests
