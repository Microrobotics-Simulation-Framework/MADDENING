# Changelog

All notable changes to MADDENING will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Additional sections per release: **Verification**, **Security**, and **Known Anomalies**.

## [Unreleased]

### Changed

- **Resume-from-URL transport moved to `maddening.cloud.resume`.**
  `download_and_load_state` (with its `file://` / `http(s)://` / fsspec
  fetch helpers) now lives in the cloud package, where deployment concerns
  such as storage backends and credentials belong; the core
  `maddening.core.simulation.checkpoint` module keeps only the local
  save/load/manifest functions and imports neither `maddening.cloud` nor
  `fsspec`.  `maddening.cloud.download_and_load_state` is exported lazily
  like the other cloud names.  The old
  `maddening.core.simulation.checkpoint.download_and_load_state` import still
  works as a forwarding alias that emits a `DeprecationWarning`; the alias is
  removed in 1.0.  Behaviour, signature and errors are unchanged.
- **Coupling groups now default to the early-exit solver** (`CouplingGroup.solver="ift"`).
  The fixed-point iteration is a `jax.lax.while_loop` that exits as soon as the
  group's convergence norm meets its threshold, for every `acceleration`,
  `iteration_mode`, `convergence_norm` and `diagnostics` setting.  The legacy
  unrolled `fori_loop` ran `max_iterations` passes regardless of convergence
  (measured: step cost linear in `max_iterations`, 80–90 % dead iterations at
  typical convergence).  IQN-IMVJ cross-timestep Jacobian reuse now runs inside
  the while_loop with the same shift-and-insert column convention as before.
- **IFT derivative rule is a `jax.custom_jvp`** (was `custom_vjp`).  JAX derives
  reverse mode by transposing the linear tangent rule through lineax, so one
  definition serves `jax.jvp` / `jacfwd` (the FMI `FORWARD` directional
  derivative — previously a `TypeError` through coupled steps), `jax.grad` /
  `jacrev`, and `jax.hessian`.
- `iqn_ils_update` takes a keyword-only `have_prev` flag saying whether the
  previous-iterate arguments are real; loop bodies pass `i > first`.

### Added
- **`AdaptiveNode` base class** (`maddening.nodes.adaptive`, `@stability(STABLE)`,
  `MADD-NODE-009`): the frozen-active-set adjoint pattern for adaptive solvers.
  A subclass supplies `compute_active_set` (any fixed-shape `jnp` selection
  rule) and `solve_frozen` (the masked solve, through `ift_linear_solve`); the
  base class wires them into a JAX-traceable `update` over a padded `(c, mask)`
  state — adaptivity changes which mask entries are true, never an array
  shape, so the step runs under `jit` / `lax.scan` unchanged — commits the
  selection under `stop_gradient`, and zeroes coefficients off the mask.
  `jax.grad` through the node is the exact frozen-set adjoint on every region
  where the active set is constant (verified against finite differences and
  dense sub-block solves to 1e-6).  Physical parameters live in the graph
  parameter pytree with the subclass's `ParamSpec`s, so `fit` / `fim` reach
  them.  Palais-trap diagnostics from the design spike: `blindness_ratio`,
  `is_trapped_at`, `symmetry_break` (anisotropic step along the full-basis
  gradient, trainable leaves only), a cold-start gate in `initial_state`
  (`AdaptiveNodeBlindnessError`) and `cold_start()` with one automatic escape;
  constants `blindness_threshold = 0.7`, `blindness_break_delta = 0.05`,
  `D_threshold = 5` as documented class attributes.  Algorithm guide
  (`docs/algorithm_guide/nodes/adaptive_node.md`), authoring guide
  (`docs/developer_guide/adaptive_node.md`), benchmark `MADD-VER-004`
  (Green's-function reference for `-u'' + u = f`).  `ift_linear_solve` is
  promoted from `EXPERIMENTAL` to `STABLE` with its signature unchanged, as
  its v0.3.1 docstring promised.  The wavelet subclass stays post-1.0.
- **Interface mappings are serialisable** (`MappingSpec`, the "13(a)" half of
  the deferred mapping-serialisation item).  Every mapping factory
  (`rbf_mapping`, `nearest_neighbor_mapping`, `projection_1d_mapping`,
  `matrix_mapping`) attaches a `MappingSpec` — kind, hyper-parameters and
  *references* to its point sets, never the weights — which
  `GraphManager.to_dict` / `from_dict`, the config helpers and
  `save_graph_to_usd` / `load_graph_from_usd` (attribute
  `maddening:mappingSpecJson`) now carry instead of refusing mapped edges.
  Point references are `{"node": name, "field": key}` (a node's
  `static_data` or array-valued parameter), `{"asset": "<file>.npy|.npz"}`
  (relative to the config / stage directory, `base_dir=`; no absolute
  paths or `..`) or an inline list for at most 64 points; the factories
  take `source_ref=` / `target_ref=` (`matrix_mapping(asset=)` — an
  explicit matrix is never inlined).  On load the mapping is rebuilt by
  the same factory (weights bitwise equal) and registered in
  `params["mappings"]` as `add_edge(mapping=)` does — which now also
  accepts a spec directly; a checkpoint loaded afterwards keeps its
  (possibly trained) weights.  A mapping without a complete spec is
  refused by the writers with a message naming the argument to pass
  (`to_dict(strict_mappings=False)` for display; the REST `GET /graph`
  uses it).  The FMI exporter is unchanged (mapping weights never reach
  the FMU).  Guide: algorithm_guide/coupling/interface_mapping.md,
  "Serialisation".
- **Multi-GPU hardware-session tooling** (local preparation for the
  human-supervised 4xA100 session; nothing here launches a pod).
  `benchmarks/multigpu/run_pod.py` is the pod-side runner: `--goal
  exchange` times `all_to_all` vs `ppermute` unstructured halo exchanges
  at 1e5-1e6 cells (warmup + repeats, min/median, bytes from
  `exchange_traffic()`, bit-identity), `--goal forward` runs a
  `ShardedUnstructuredNode` at 1e6 cells on a real mesh (`--mesh
  edges.npz`) or a synthetic one against the unsharded node, `--goal
  gradient` reports sharded-vs-unsharded gradient parity through a
  rollout and the preconditioned `sharded_cg`; each goal writes one JSON
  under `--out`, `--summarise DIR` prints the ranking table and the
  ppermute-vs-all_to_all recommendation (only real-GPU points at >= 1e5
  cells decide), and `--dry-run` proves the whole script on CPU virtual
  devices (`tests/cloud/multigpu/test_run_pod_dry_run.py`, slow lane).
  `benchmarks/multigpu/README.md` is the session runbook (launch by
  hand with `CloudLauncher`/SkyPilot, copy-back, stop the pod).  The
  `tests/cloud/multigpu` conftest now forces virtual host devices only
  when no accelerator is about to be used (`JAX_PLATFORMS` naming
  cuda/gpu/rocm/tpu, or unset with a CUDA jaxlib plugin + `nvidia-smi`
  GPU + visible devices, leaves `XLA_FLAGS` alone; `MADDENING_VIRTUAL_DEVICES=N`
  overrides either way; `tests/cloud/multigpu/test_conftest_device_policy.py`),
  so on a GPU pod with `JAX_PLATFORMS=cuda` the multi-device tests run on
  the GPUs instead of 16 virtual CPUs.  The cloud examples' install
  command pinned `jax[cuda12]>=0.4,<0.6` (uninstallable next to this
  package); they now use the `pyproject` range `>=0.10,<0.13`.
- **FMU sidecar protocol 2: binary frames for bulk payloads.**  Bit 31 of
  the 4-byte length prefix marks a binary frame (`[u32 BE header_len]
  [header JSON][raw bytes]`); after a `{"op":"hello","protocol":2,
  "binary":true}` the bridge answers `get` / `get_state` with raw
  little-endian float64 / raw `npz` bytes and accepts binary `set` /
  `set_state`, so values no longer go through `%.17g` / `strtod` and
  state blobs no longer through base64.  The C wrapper negotiates it and
  falls back to JSON against a bridge whose hello lacks `protocol`; a
  JSON-only client sees the protocol-1 behaviour unchanged (the hello
  reply merely gains `protocol` and `binary`), and an unknown higher
  protocol is refused at hello.  All validation (vr, bounds, read-only,
  state token/shape/size, malformed frames as error replies) applies to
  both forms; the C side checks the header count against the raw length
  and the caller's array before any copy and refuses flagged lengths over
  the 64 MiB limit before allocating.  Measured: a 10^6-element `get`
  2.3 s as JSON vs 27 ms binary over loopback (8 bytes per value).
  `FmuTcpBridge.binary_frames_served` / `binary_frames_received` count
  the traffic; `tcp_bridge` gains `recv_raw`, `send_binary`,
  `encode_binary`, `decode_binary`, `values_of`, `state_of`,
  `PROTOCOL_VERSION`.  C unit tests, the sanitizer fuzz harness (now
  also binary-flagged replies), `tests/fmi/test_binary_frames.py` and the
  Hypothesis properties in `tests/fmi/test_binary_frames_properties.py`
  (bitwise float64 round trip incl. NaN payloads / infinities / negative
  zero / subnormals; any byte string decodes consistently or raises
  `ValueError`; any flagged frame after a binary hello gets exactly one
  reply and the connection keeps serving) cover it; user guide: "Wire
  protocol" in `fmu_export.md`.
- **Static type checking (phase 1, non-blocking).**  `pyrightconfig.json`
  (basic mode, `src/maddening` only, optional extras' imports downgraded to
  warnings), `pyright` in the `ci`/`dev` extras, a `typecheck` CI job that
  runs with `continue-on-error` and writes the error count to the step
  summary, and `scripts/typing_baseline.py`, which summarises a pyright run
  per rule and per file.  The measured baseline and the phase-2 plan (annotate
  the `STABLE` surface after the 0.4.0 API freeze, ship `py.typed`, make the
  check blocking on that surface) are in `docs/developer_guide/typing.md`.
  No source annotations changed.
- **Per-neighbour unstructured halo exchange** (v0.4.0 plan hard gate,
  the hardware-independent part).  `exchange_unstructured(...,
  method="ppermute")` and `ShardedUnstructuredNode(..., exchange="ppermute")`
  send one `lax.ppermute` per *communicating cyclic shift*, each sized to
  that shift's largest message, instead of one dense
  `(n_devices, n_ghost_max)` `all_to_all` payload to every shard; results
  are bit-identical (random partitions, gradients, a 10^5-cell ring in
  the slow lane).  `exchange_traffic(layout)` reports cells moved per
  shard for both transports and the useful count, so the transport can
  be chosen from the partition before any GPU time is spent (a ring on 4
  shards: 2 cells vs 8).  The default stays `all_to_all`; the NCCL
  timings that decide the default need a real multi-GPU host.
- **FMU C wrapper, TCP/JSON bridge and packaging** (v0.4.0 plan hard
  gate).  `src/maddening/fmi/c/maddening_fmu.c` implements the FMI 3.0
  co-simulation entry points (instantiate / initialise / `DoStep` /
  `Get`/`Set` for every numeric type / FMU state get, set, serialize /
  reset / terminate; model exchange and scheduled execution refuse) with
  nothing but libc: every call is forwarded as a length-prefixed JSON
  message over TCP to `maddening.fmi.tcp_bridge.FmuTcpBridge`, which maps
  value references onto the sidecar's inputs, outputs and parameters and
  runs the master-step loop.  `maddening.fmi.package.build_fmu_binary`
  compiles the wrapper against the vendored (BSD-2) FMI 3.0 headers and
  `write_fmu` packages `modelDescription.xml`, the binary and
  `resources/endpoint.txt` into a `.fmu`; `build_model_description(...,
  model_identifier=)` emits the `<CoSimulation>` element.  Verified end
  to end: FMPy `simulate_fmu` drives the compiled FMU through the bridge
  and reproduces `gm.run_scan` with a set parameter and a driven input.
  ZMQ is not required; a ZMQ transport can carry the same payloads later.
- **Multi-clock FMU export.**  `build_model_description(multi_clock=True)`
  emits one FMI 3.0 `<Clock>` (`intervalVariability="constant"`,
  `intervalDecimal=<dt>`) per distinct node timestep among the exported
  nodes and tags every exported output and external input with its node's
  clock (`clocks=` attribute, `variability="discrete"`; clocked outputs
  are not initial unknowns), so an importer knows a node on a coarser rate
  only changes on its ticks.  Clocks are `clock_<k>` in order of
  increasing interval; the fastest equals the default step size.  Off by
  default: the single-clock surface is unchanged.  Validated with FMPy
  (`validate=True, validate_model_structure=True`).

- **Graph parameter pytree** — the compiled step is now
  `step_fn(state, external_inputs, params)`.  `GraphManager.params`
  (`{"nodes": {name: node.params_pytree()}, "mappings": {}}`) is snapshotted
  at compile time and passed on every `step` / `run` / `run_scan` /
  `run_scan_with_history` / `run_sweep` / adaptive run (each accepts an
  optional `params=` keyword).  Node constants are therefore traced inputs
  rather than closure constants baked into the jit: `jax.grad` /
  `jax.jvp` / `jacfwd` reach them — including through coupling groups,
  where `closure_convert` hoists them into the IFT rule — and a changed
  value takes effect without recompiling.  A node opts in by declaring
  `update(..., *, params=None)` and reading its constants from `params`
  (`SpringDamperNode`, `BallNode`, `HeatNode`, `RigidBodyNode` migrated;
  then `RigidBody2DNode` — mass, inertia, gravity; `HeartPumpNode` — the
  six Windkessel constants including `systole_fraction` on a logit spec;
  `TableNode` and `HealthCheckNode` — on the contract with no dynamics
  constant to inject, `checks` and the surface height stay structural;
  `LBMNode` — `viscosity`, so `tau` is a traced constant; `LBMPipeNode` —
  `tau`, `tau_tracer`, `propeller_strength`, `gravity` and the Shan-Chen
  constants `G`, `rho_0`, `rho_wall`, `rho_liquid`, `rho_gas`, the latter
  trainable only when the node was built multiphase since `G != 0`
  selects the branch; `SurrogateNode` — every floating leaf of the
  network weights as a flat `"weights<path>"` entry, rebuilt into the
  weights pytree inside `update`, so `jax.grad` of a trajectory loss
  reaches the surrogate weights and fine-tuned weights need no recompile);
  nodes on the 3-argument contract keep working unchanged.
  `SimulationNode.params_pytree()` defaults to the float-valued entries of
  `self.params`; structural values (`n_cells`, shapes, ...) stay on the
  recompile path.  Checkpoints save and restore `params`; the REST
  `PUT /graph/params/{node}` updates `gm.params` in place for such nodes
  instead of forcing a recompile.  `TestParameterRecovery` now recovers
  `k, c` through the real graph (single spring and a coupled group), with
  the float32 gradient matching a float64 finite difference to 2.4e-6
  relative over 100 steps.  One consequence: expressions like
  `dt * gravity` are no longer constant-folded, so XLA may contract them
  into an FMA in one compiled shape and not another; `run_sweep` and
  individual `run_scan` results can now differ by ~1 ulp (they were
  bit-identical before), and the vmap-consistency test allows a few ulps.
- **`ParamSpec`** (`maddening.core.params`): per-parameter `trainable`,
  `bounds` and `transform` (`"log"` for strictly positive constants,
  `"logit"` for intervals, `None` = clip).  Nodes declare specs for their
  constants in `SimulationNode.param_specs()` (`initial_*` entries default
  to `trainable=False`; the four migrated nodes declare bounds/transforms);
  `GraphManager.set_param_spec(node, key, spec)` overrides per graph.
  `gm.trainable_mask()`, `gm.unconstrain()` / `gm.constrain(u)` and
  `gm.check_params()` are the pytree maps an optimiser needs; the
  transforms clamp to the representable float32 interior so a saturated
  step (`exp(89)`, `sigmoid(17)`) stays finite and invertible.
- `maddening.sysid.fit(gm, loss_fn, ...)`: Adam in the unconstrained
  coordinates under the trainable mask, returning physical params inside
  their bounds with frozen leaves bit-identical; raises on a non-finite
  gradient.  `fim(..., mask=)` restricts the Fisher matrix to the leaves a
  mask (e.g. `gm.trainable_mask()`) selects.  `TestParameterRecovery` now
  fits through `sysid.fit` with `mass` frozen by spec.
- Passing a `params` pytree that names a node whose `update` takes no
  `params`, an unknown node, or an unknown parameter key is now a
  `ValueError` at trace time (previously silently ignored — and a gradient
  with respect to it silently zero).  `gm.nodes_without_params()` lists
  the nodes whose constants are baked; `compile()` logs them.
- `verify_node` / `assert_node_verified` battery gains `params_consistent`
  (injected `params_pytree()` reproduces the baked-constant step to float32
  round-off), `params_gradient_finite` (finite `d(outputs)/d(params)`) and
  `params_effective` (every trainable leaf has a non-zero gradient on at
  least one sample — catches a constant still read from `self.params`);
  all `SKIP` — a new `VerificationResult` status that counts as passed —
  on nodes whose `update` takes no `params`.  `SimulationNode.accepts_params()`
  exposes that probe.  `tests/verification/test_builtin_nodes_verified.py`
  runs the full battery on every built-in node in CI and asserts the
  migrated ones do not skip the params checks.
- Hypothesis property suites for the params pytree
  (`test_hypothesis_params.py`: baked ≡ traced step, `step` ≡ `run_scan` ≡
  `run_sweep` within ulps, float32 params gradient vs float64 finite
  differences, jvp/vjp adjoint identity through an IFT-coupled group, no
  dirty/recompile/mutation on a modified pytree, checkpoint round trip) and
  for `maddening.sysid` (`test_hypothesis_sysid.py`: `windowed_loss` zero
  at truth and non-negative elsewhere over random tilings, single window ≡
  direct trajectory loss, unconverged masking, FIM symmetric PSD with
  orthonormal eigenvectors, injected null directions recovered).
- User guide page `docs/user_guide/parameters.md`.
- **Interface mappings on edges** (`maddening.core.coupling.mapping`):
  `add_edge(..., mapping=)` takes a `Mapping` (`apply` / `apply_T` /
  `params_pytree`), applied before the scalar `transform`; its weights
  are snapshotted into `gm.params["mappings"]["<src>.<field>-><tgt>.<field>"]`
  and passed as a traced input on every step, so `jax.grad` reaches them
  (also through an IFT coupling group) and a replaced matrix needs no
  recompile.  `StaticLinearMapping` with factories `rbf_mapping`
  (polynomial augmentation on by default, solve instead of `inv`,
  kernel-relative ridge, `mode="consistent"|"conservative"` where
  conservative is the transpose of the reverse consistent map and
  preserves totals exactly), `nearest_neighbor_mapping`,
  `projection_1d_mapping`, `matrix_mapping`.  Matrices are assembled and
  solved in float64 on the host at construction.  Gates: patch test
  (constants and linear fields reproduced across random non-conforming
  point sets, float32 round-off and 1e-8 in float64) and conservation test
  for every kernel; a mapped edge reproduces the closure-transform result;
  gradient with respect to the weights is finite and non-zero.  Mapping
  weights are `trainable=False` by default (opt in with
  `set_param_spec(edge.key, "H", ParamSpec())`); `add_edge` checks
  `n_source` / `n_target` against the field and declared boundary shape;
  `gm.edges`, `gm.resolve_boundary_inputs(node)`; `to_dict` records
  `mapping.describe()` (never the weights) and `from_dict` /
  `save_graph_to_usd` refuse mapped edges until `MappingSpec` lands with
  the USD read path.  `rbf_interpolation` (closure API) now shares the
  same matrix construction, so its multiquadric constant test went from
  `atol=0.1` to round-off.  Guide: `docs/algorithm_guide/coupling/interface_mapping.md`.
- **Profiler rewrite** (`maddening.core.simulation.profiler`): coupling
  overhead is now *measured* (the graph is recompiled with every group
  capped at one iteration and timed; the difference is the cost of the
  extra iterations, reported per iteration) instead of inferred from
  isolated node timings; per-group iteration statistics over the run
  (mean / min / max against `max_iterations`, fraction of steps at the
  cap, fraction converged) replace the last-step count that assumed
  `max_iterations`; `dispatch_floor_ms` (a jitted identity on the state
  pytree), median / p95 step time, device name; `trace=True` records a
  short `jax.profiler` trace and attributes device kernel time to the
  graph's `jax.named_scope` labels (`node:<name>`, `coupling:residual`,
  `coupling:accelerate`, `coupling:interface_override`, `edge:mapping`),
  with the device-busy fraction and kernels per step that tell a
  launch-bound step from a compute-bound one.  Recommendations use the
  new numbers (unconverged-at-cap, launch-bound, dispatch-bound).
- `benchmarks/bench_coupling.py`: coupled-step benchmark (coupling group
  vs staggered baseline, iterations used vs cap, measured per-iteration
  cost, optional trace attribution, PERF-1 acceptance) for the MIME AR4
  experiment graph (`--graph mime-ar4 --experiment DIR`), a two-spring
  pair and a heat chain; JSON output under `benchmarks/results/`.
- **Persistent compilation cache** (`maddening.core.simulation.compile_cache`,
  PERF-2): `enable(cache_dir)` points JAX's persistent cache at a
  directory with thresholds that cache sub-second compiles; `compile()`
  honours `MADDENING_COMPILATION_CACHE_DIR`; `warm_cache(gm_factory,
  n_steps=, scan_steps=)` compiles a graph's step and scan ahead of a run.
  A cross-process cache hit is tested.  Developer guide:
  `docs/developer_guide/profiling.md`.
- `maddening.sysid` follow-ups: **multiple shooting** — `windowed_loss(...,
  window_states=, continuity_weight=)` restarts each window from a free
  state and ties consecutive windows with a continuity penalty,
  `init_window_states` seeds them from the observations, and
  `fit_multiple_shooting` optimises params and window states jointly
  (noisy window starts no longer seed every window with measurement
  error); **noise model** — `fim(..., noise_std=)` (scalar or per-leaf σ)
  weights the residual so `crb` is in the parameters' own units;
  **Levenberg–Marquardt** — `fit_lm(gm, residual_fn, ...)` uses the same
  `jacfwd` sensitivities as `fim` in unconstrained coordinates under the
  trainable mask and recovers a spring's (k, c) from 2× perturbations in
  a handful of iterations where Adam needs hundreds; **progress events** —
  `fit` / `fit_lm` / `fit_multiple_shooting` notify the graph's observers
  with a `"fit_progress"` event (`EVENT_FIT_PROGRESS`: method, iteration,
  loss, params) every `notify_every` iterations, so the REST relay and
  live stage can show a calibration as it runs.
- **v0.4.0 plan items (sharded solvers, coupling, stability):**
  `sharded_cg` / `sharded_gmres` gain `differentiable=True`, which routes
  the solve through `lax.custom_linear_solve` so `jax.grad` and `jax.jvp`
  through the result — and the IFT adjoint of a coupling group whose node
  solves with them — are exact linear-solve adjoints using the same
  backend *and the same preconditioner* in the adjoint solve (`iters` is
  then -1; off by default to keep the STABLE result contract).
  `jacobi_preconditioner(diag)` and `block_jacobi_preconditioner(blocks)`
  are the first users of the `preconditioner=` hook; `backend="lineax"`
  now refuses a preconditioner instead of silently dropping it.  Gradient
  parity through the preconditioned solve is tested against the dense
  reference (reverse and forward mode, RHS and operator coefficients) and
  on a 4-device CPU-virtual mesh.  C1: a multi-physics IQN-IMVJ test
  (heat rod ⊗ spring, different operators per sub-domain) through the
  IFT while_loop with cross-timestep warm start, fori parity and a
  finite-difference gradient check.  Second stability wave: the
  unstructured partition layout / halo exchange / partition + gather
  helpers, `SidecarConfig` and `FMUState` are tagged `EVOLVING`.
  C4: `SimulationNode.domain_integral_axes()` — a domain integral can be
  reduced over a subset of mesh axes (one leading axis per unreduced
  axis) or not at all (per-shard values stacked), on both sharded
  wrappers; a body-surface integral living on some shards no longer
  needs a full-mesh `psum`.
- The IFT Krylov adjoint raises an actionable `ImportError` naming
  `pip install maddening[ift]` (and the `linear_solver='dense'` fallback)
  when lineax is missing; `tests/core/test_solver_ift_no_lineax.py`.
- Hypothesis property over random graphs of built-in nodes: the compiled
  step must trace exactly once across steps and after `set_node_state`
  (`test_hypothesis_retrace.py`).
- **Graph params on the sharded path.**  `ShardedStencilNode` and
  `ShardedUnstructuredNode` now take part in the graph parameter contract
  when the wrapped node's `update_padded` accepts `params`: the wrapper
  reports `accepts_params()` / `params_pytree()` / `param_specs()` from the
  inner node, and the node's entry of `gm.params` is replicated to every
  shard and handed to `update_padded(..., params=)` (the params signature
  is part of the shard_map cache key).  `LBMNode.update_padded` reads
  `viscosity` from the injected params, so a sharded LBM graph is
  calibratable and matches the unsharded graph for the same params;
  previously the sharded path silently ignored `gm.params`.
- `GraphManager.reset_state()`: reset every node to `initial_state()` and
  zero the `_meta` counters / coupling diagnostics / IQN warm-start
  caches with the same weak-type normalisation `compile()` applies, so a
  reset never retraces the jitted step.  The profiler, the REST server's
  reset and the example servers use it instead of assigning
  `initial_state()` into `_state`.
- REST `PUT /graph/params/{node}` can address any leaf of the node's live
  pytree, not only constructor params (surrogate weights, sharded wrappers
  whose inner node owns the params), with shape and dtype checks.
- REST `PUT /graph/params/{node}` validates values against the node's
  `ParamSpec` bounds before writing anything (400 with the offending leaf).
- `maddening.testing.strategies.node_states` samples bool / integer state
  fields with their own dtype, so the `structure` check covers monitor-style
  nodes (`HealthCheckNode`) instead of being skipped.
- **Params persistence and FMI.**  `gm.to_dict()` / `from_dict` and USD
  (`save_graph_to_usd` / `load_graph_from_usd`) store each node's
  *effective* params (`gm.effective_node_params`: constructor args with
  the live `gm.params` values written over them) and the graph's
  `ParamSpec` overrides (`param_specs` key; `maddening:paramSpecOverridesJson`
  on the node prim), so a calibrated graph reloads calibrated with the
  same trainable mask.  `build_model_description` exposes every
  `gm.params` leaf as an FMI `parameter` / `tunable` variable
  `<node>.params.<key>` with `ParamSpec` description, units and bounds
  (XML `min` / `max`; `include_parameters=False` to opt out).
  `SidecarConfig(params=..., param_specs=gm.param_specs())`
  makes the sidecar call the compiled step's 3-argument contract and
  serve `get_params` / `set_params` (also as wire requests); a set value
  takes effect on the next step without recompiling, unknown names,
  wrong shapes or out-of-bounds values are errors (the call is atomic),
  and FMU state snapshots carry the parameters
  (`serialize_fmu_state(params=)`, `deserialize_fmu_state(return_params=True)`;
  legacy snapshots still load).
- `maddening.sysid`: `windowed_loss` (teacher-forced windowed trajectory
  loss with optional masking of windows where a coupling group exited
  unconverged) and `fim` (Fisher information `JᵀJ` from `jacfwd`
  sensitivities, relative scaling, eigen-decomposition, Cramér–Rao bounds)
  — the identifiability check correctly isolates the (k, c, m) common-scale
  direction on a spring observed through position only.
- Coupling iteration count and residual are now always written to `_meta`
  under the default solver (not only with `diagnostics=True`), so
  `coupling_diagnostics()` and scan histories always carry the converged
  flag.
- `CouplingGroup.strict_convergence`: raise (jit-safe, via `equinox.error_if`)
  when a group exits at `max_iterations` unconverged, since the IFT gradient is
  then invalid.  Off by default.
- `GraphManager.coupling_diagnostics()` reports `"converged"` per group.
- `maddening.testing.verification` is a Hypothesis battery (`verify_node`,
  `assert_node_verified`): finite outputs, preserved structure, determinism,
  jit/eager agreement, finite gradients, plus opt-in `output_bounds`,
  `energy_fn` and custom `invariants`; failures return the shrunk
  counterexample.

### Deprecated

- `CouplingGroup.solver="fori"` emits `DeprecationWarning`; removed in the next
  minor release.

### Removed

- The stelling formal-verification suite, CI job and `stelling` dependency.
  The `[verify]` extra now only pulls `hypothesis`.

### Verification
- **C-level tests for the FMU wrapper** (`tests/fmi/test_c_unit.py`,
  `tests/fmi/c/`): a unit-test binary that includes the wrapper source
  (framing, JSON number parsing, endpoint discovery, every FMI entry
  point against a fake sidecar on a socketpair and a loopback listener),
  built plain and with `-fsanitize=address,undefined`; a deterministic
  fuzz harness for the reply surface (3 seeds x 3000 iterations in the
  fast lane, 60k in the slow lane, also exported as a libFuzzer target);
  the unit and fuzz binaries under valgrind memcheck; a short
  coverage-guided libFuzzer campaign with clang; FMPy driving an ASan
  build in a subprocess against a normal and a hostile bridge; FMPy's
  low-level FMI 3 API with two instances (params before initialisation,
  FMU state get/set/serialize, reset, terminate); `validate_fmu` on the
  packaged FMU; and a warning-free `-std=c11 -pedantic` build.  CI
  installs valgrind and clang so all of it runs there; each part
  self-skips where its tool is missing.
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

### Security
- **REST checkpoint endpoints are confined to a directory.**
  `/checkpoint/save` and `/checkpoint/load` took an arbitrary server-side
  path from an unauthenticated client (arbitrary file write, file-existence
  oracle).  Paths are now relative to `SimulationServer(checkpoint_root=)`
  (default `./checkpoints`) and must resolve under it; load errors no
  longer echo parser internals.  The API still has no authentication:
  bind it to localhost or put it behind a proxy that authenticates.
- **FMU bridge no longer unpickles importer bytes** (independent audit
  round 3, CRITICAL).  `FmuTcpBridge` `set_state` used `pickle.loads` on
  the base64 payload an importer hands to `fmi3SetFMUState`, i.e. remote
  code execution for anyone able to reach the bridge port.  The FMU-state
  blob is now an arrays-only `npz` (`allow_pickle=False`) carrying the
  schema token, time, inputs, states and params; on `set_state` the
  token, key set and every shape are validated before anything is
  written.  The pickle-based `FmuSidecar.handle` wire protocol is for
  trusted in-process / Python clients only and is documented as such.

### Fixed
- **Independent audit, round 4** (residue across rounds 1-3; report under
  `benchmarks/results/audit4/`, regression tests in
  `tests/core/test_checkpoint_and_params_shape_guards.py` and `tests/fmi/test_bridge_inputs_and_robustness.py`).
  FMU bridge: an input the importer never set is now the advertised zero
  start value (like `gm.step()`), also after `reset` and `set_state`, so a
  HeatNode FMU no longer runs adiabatic until its first `fmi3Set*`; a
  multi-sub-step `step` that fails leaves state and time untouched; a
  malformed request gets an error reply instead of a dropped connection;
  an FMU-state member larger than the live leaf is refused before it is
  decompressed and a non-finite time is refused; the C wrapper refuses a
  non-base64 state blob before it can break the request framing and
  accepts `[::1]:port` endpoints.  REST: `PUT /graph/state` validates
  field set, dtype, shape and finiteness before writing; a JSON boolean
  for a numeric param is a 400 (it used to drop the leaf at the next
  recompile); `POST /graph/nodes` traces one update abstractly before
  adding the node (a bad constant used to wedge every later step);
  `POST /graph/edges` checks that the nodes and the source field exist.
  Core: `load_state` refuses a state field of the wrong shape and coerces
  dtype; a checkpoint without `_meta` keeps the freshly compiled `_meta`
  instead of leaving a multirate graph to raise `KeyError`; a wrong-shape
  params leaf is refused by `step(params=)` and by `gm.params[...] =`
  (it used to broadcast the node's state permanently); `add_node` refuses
  names containing `/`, `#` or `->`; a sharded `HeatNode` accepts a
  halo-padded per-cell `heat_source`.  `fsspec` joins the `ci`/`dev`
  extras so the cloud-URL checkpoint tests run in CI instead of skipping.
- **FMU wrapper survives a sidecar that goes away.**  A `send()` to a
  closed peer raised SIGPIPE and killed the importer's whole process;
  sends now use `MSG_NOSIGNAL` (`SO_NOSIGPIPE` on macOS) and the call
  returns `fmi3Error` (unit-tested against a closed socketpair).  Found
  when the fuzz harness went thread-free: it preloads the fake reply
  into the socket instead of spawning a thread per iteration, which is
  what made the libFuzzer campaign reach 7.5 GB RSS and get OOM-killed on
  the CI runner.  The campaign is now built with UBSan only (the ASan build
  reported ~8 GB RSS at `INITED` with 25 MB of live heap on the runner and
  in long test sessions, a host-accounting effect, not a leak) and guarded
  by a per-allocation `-malloc_limit_mb` instead of an RSS limit; memory
  safety of the same harness stays covered by the ASan seeded runs and
  valgrind.  The
  cross-process persistent-cache test proves a hit by the cache
  directory gaining no entries rather than by a wall-clock ratio.
- **Independent audit, round 3** (FMU bridge / wrapper / exchange;
  regression tests in `tests/fmi/test_bridge_security_and_stepping.py`).  A
  communication step that is not a whole multiple of the master timestep
  is refused instead of silently snapping the physics while reporting
  `t + h` (the FMU now advertises a fixed communication step); a `set`
  request is atomic across parameters and inputs (a rejected parameter
  no longer leaves an already-applied input behind) and refuses
  non-finite inputs; a second FMU instance on one bridge gets a clear
  error instead of blocking; `fmi3DoStep` initialises its output flags
  on the error path and `fmi3EnterEventMode` refuses, matching
  `hasEventMode="false"`; `exchange_unstructured(method="all_to_all")`
  no longer raises when no shard needs a ghost cell (one device,
  edge-disjoint shards).  Also: the two Hypothesis integrator-order
  tests run under `jax.experimental.enable_x64()` instead of skipping.
- **FMU C wrapper, found by its own fuzz/sanitizer tests.**  A failed
  instantiation after the `hello` exchange (token mismatch) leaked the
  reply buffer; a failed or partial reply receive left a freshly grown
  buffer unterminated, so a later parse could read past it (ASan
  heap-buffer-overflow); `parse_values` / `GetFMUState` now refuse a
  missing reply; a `file://` resource path with nothing after the prefix
  is no longer indexed at `[-1]`; the endpoint file's trailing newline is
  stripped; POSIX feature macros make the source build under
  `-std=c11 -pedantic`.  Also: `gm.trace_count` replaces the jit cache
  size as the retrace probe in tests (the cache count reads 0 on JAX
  0.10, which broke CI).
- **Independent audit, round 2** (12 findings, all fixed; report under
  `benchmarks/results/audit2/`, regression tests in
  `tests/core/test_params_persistence_edge_cases.py`).  `compute_interface_correction`
  joined the params contract (a calibrated diffusivity now also corrects
  the coupled interface cells; `HeatNode`, `HybridNode`), and `HybridNode`
  forwards `params` to its physics node.  `PUT /graph/params/{node}`
  stores the constructor's Python type (a JSON `40` for a float leaf used
  to turn it into an `int` the pytree no longer exposed), validates a
  request before the first compile exactly as after it, and `GET` returns
  the live view.  `load_state` compiles a fresh graph *before* restoring,
  so the multirate step counter and coupling history survive a
  load-before-compile; checkpoints now carry `params["mappings"]`, and a
  params leaf of the wrong shape is refused.  `remove_edge` drops
  ordinal-key overrides; Python-scalar leaves in `gm.params` are coerced
  to the leaf dtype (no retrace, kept on recompile, also under x64); the
  coupling residual `_meta` seed takes the group's floating dtype
  (float64 graphs no longer fail `run_scan` after compile); the logit
  clamp uses `nextafter` limits so a few-ulp-wide interval stays strictly
  inside.
- **FMU export of a real graph had no inputs and a wrong step size.**
  `build_model_description` looked for a `_external_input_specs` dict a
  `GraphManager` never had, so external inputs were silently omitted, and
  read a `_master_timestep` attribute that does not exist, so the default
  experiment step was always 1e-3.  Inputs now come from the graph's
  external-input list as `<node>.<field>` (description / unit from the
  target's `boundary_input_spec`), and the step size is the graph's base
  timestep (fastest node).
- **Flux edges honour the params pytree** (audit round 1, HIGH).
  `compute_boundary_fluxes` gained the same keyword-only `params` as
  `update`; the graph passes the node's `gm.params` entry on every flux
  evaluation (Gauss-Seidel, Jacobi, uncoupled and IFT paths), so a
  calibrated stiffness / diffusivity changes the force or heat flux an
  edge *delivers* and the gradient through a flux consumer is correct.
  `SpringDamperNode`, `HeatNode`, `HeartPumpNode`, `LBMNode` and
  `HybridNode` migrated; `verify_node`'s params checks now cover fluxes
  and fail a producer that takes `params` in `update` only.
- **`compile()` keeps calibrated params** (audit round 1, HIGH).  Every
  recompile used to overwrite `gm.params` with the constructor snapshot,
  so adding an edge / external input, replacing a node, a REST write on a
  legacy node or the profiler's one-iteration variant silently discarded
  a fit.  Live leaves whose node/key/shape/dtype still exist are carried
  over (`_merge_live_params`; a dropped leaf warns), `gm.reset_params()`
  is the explicit way back, and `load_state` on a not-yet-compiled graph
  compiles first instead of dropping the checkpoint's params.
- **Non-float leaves in coupled graphs are bit-exact** (audit round 1,
  HIGH).  The float32 images that carry integer / boolean leaves through
  the IFT closure were exact only below 2**24: `uint32` / `int32` values
  above that were corrupted and typed PRNG keys crashed at trace time.
  Images are now 16-bit limbs (`float_image` / `from_float_image` in
  `coupling.acceleration`), keys travel as their uint32 data, and the
  coupling norms and the predictor act on floating fields only (a counter
  or flag keeps its first-pass value instead of being extrapolated).
- **A partial params pytree** through `gm.step` / `run` / `run_scan` is
  completed from the *live* `gm.params` (a missing node or key used to
  fall back to the constructor constant); the raw compiled step refuses
  an incomplete pytree with a message pointing at the wrappers.
- **Two mapped edges on one field pair** (two additive contributions)
  used to share a single `params["mappings"]` slot, the first silently
  using the second's weights; each now gets its own slot via
  `EdgeSpec.ordinal` (`"a.v->b.inp"`, `"a.v->b.inp#1"`).
- `set_param_spec` overrides no longer outlive `remove_node` /
  `remove_edge` (a stale one broke `to_dict` -> `from_dict`); edge
  `transform` (registered name), `additive` and units now survive the
  config round trip.
- **`ParamSpec` edge cases** (audit round 1).  `constrain` with
  `transform="log"` and `lo != 0` could land exactly *on* the open bound
  (`8.0 + exp(-15)` is `8.0` in float32), so `check_params` rejected the
  fit's own output and `unconstrain` returned `-inf`; the clamp is now
  relative to the bound (a few ulps of `lo` / `hi`), for `logit` too.
  `ParamSpec.check` / `check_params` reject NaN and `±inf` (NaN used to
  pass every bounds comparison).  `ParamSpec.from_dict({"bounds": null})`
  no longer crashes.  A bounded identity leaf keeps its dtype through
  `constrain` (integer weights were promoted to float32).
- **`PUT /graph/params/{node}`** validates dtype coercion, shape,
  finiteness and bounds for every key *before* writing anything: a string
  or `null` for a live float is a 400 naming the key (was a 500), and
  `NaN` / `Infinity` are refused (NaN used to be written into
  `gm.params` and `node.params` before the response failed).
- **`sysid.fim` / `fit_lm` with `noise_std` as a pytree** (documented,
  per-leaf sigma matching the residual structure) crashed because
  `jnp.ndim(dict) == 0` took the scalar branch; the scalar branch is now
  keyed on real numbers and 0-d arrays only.
- **FMI `min` / `max` for open bounds.**  A `log` / `logit` leaf's bound
  is strict, but the FMI attributes are inclusive, so the sidecar refused
  the very value the XML advertised.  The model description now advertises
  the nearest representable float32 inside the interval (the smallest
  normal for a zero bound — its subnormal neighbour is flushed to zero on
  XLA:CPU), so every advertised bound is settable.  Inclusive bounds are
  unchanged.
- **Sharded boundary-input heuristic and `HeatNode` params.**
  `ShardedStencilNode` classified an `(n,)` input on an `n x n` grid
  sharded on axis 0 as grid-shaped (only the sharded axis's extent was
  compared), sharded and halo-padded it and broke the inner
  `update_padded`; the input must now match the full leading grid shape.
  `HeatNode.update_padded` takes `params=` like `update`, so a
  `ShardedStencilNode(HeatNode)` reports `accepts_params()` and is
  calibratable / differentiable like the unsharded node.

- **Grid-shaped boundary inputs on sharded nodes.**  `ShardedStencilNode`
  replicated every boundary input, so a per-cell field (an LBM
  `body_force` map, a `wall_mask_update`) reached each shard at its global
  shape and broke `update_padded` (or silently mismatched); a sharded LBM
  could only take a uniform force vector.  Grid-shaped inputs are now
  sharded and halo-padded like state, on both the stencil and the
  unstructured path (partition-layout inputs; global-order ones are
  refused with a pointer to `partition_value`).  `verify_node` now passes
  its full battery on a `ShardedStencilNode`, which is how this was found.

- **Every run compiled the step three times.**  Leaves seeded as
  `jnp.array(0.0)` (nodes' initial states and MADDENING's own coupling
  residual in `_meta`) are weak-typed; after one step they come back
  strongly typed, so the jitted step retraced on the second step and
  again on the third for leaves that only change later.  `compile()` and
  `set_node_state()` now normalise weak types (`_strong_typed`), and the
  `_meta` residual is seeded as float32.  Measured on the MIME AR4 graph
  on an RTX A2000: three compiles (0.92 + 0.85 + 0.83 s) became one, and
  the experiment driver's "steady-state" figure — which had absorbed two
  of them — went from 9.5–33 ms/step to 1.7 ms/step.  Results are
  bit-identical (only the trace signature changed).
- `GraphManager._default_external_inputs()` allocated fresh `jnp.zeros`
  per declared input on every call (~1.5 ms/step on GPU for a graph with
  external inputs stepped without explicit inputs); the zero arrays are
  now allocated once per compile and shared (outer dicts stay fresh).
- Differentiating through `lax.scan` over a coupled step failed when a
  node in the coupling group had an integer / boolean state leaf
  (`UnexpectedTracerError` in reverse mode, a missing constant handler
  in forward mode): `closure_convert` hoisted the leaf as an integer
  constant of the IFT `custom_jvp`, which JAX cannot linearise under a
  scan.  The solver now captures float32 images of non-float leaves
  before `closure_convert` and restores their dtype inside, and keeps
  such fields out of the fixed-point vector altogether (they are
  recomputed from the pre-step state on every pass).  Found by a
  blind-spot test; reproduced outside MADDENING first.
- `unflatten_coupled_state` returned every field as float32, so an
  integer / boolean leaf of a node inside a coupling group came back as
  float after each step (semantic drift, and a retrace of the jitted
  step); it now restores each field's dtype.
- Gauss-Seidel coupling with a *flux* edge whose consumer is scheduled
  before its producer raised `KeyError` on the first pass; fluxes are now
  seeded from the previous iterate (two sweeps, producers may depend on
  each other) and overwritten as producers update.  IQN acceleration
  derived its interface fields from edge source fields, so a flux edge
  (not a state field) raised `KeyError` at compile; a flux source now maps
  to the producer's state fields.
- `LBMPipeNode` (multiphase): `_shan_chen_force` used `np.exp` on
  `rho_wall / rho_0`, which are traced now that the graph injects params;
  five multiphase graph tests failed with a tracer-conversion error.  Now
  `jnp.exp`.
- `LBMPipeNode` (multiphase): the EDM velocity clamp took
  `sqrt(sum(u**2))`, whose gradient is NaN on a cell with exactly zero
  shifted velocity (a uniform lattice, found by the `gradient_finite`
  battery).  The sum is now floored at 1e-20 inside the sqrt; the forward
  is unchanged wherever `|u| > 1e-10`, which the existing clamp assumed.
- `HeatNode`: `length` was in `params_pytree()` but the Laplacian read it
  from `self.params`, so an injected value was ignored and its gradient
  identically zero.  `update` now passes the injected `length` to the
  stencil (the new `params_effective` check fails on exactly this).
- **IQN-ILS never activated without Jacobian reuse**: the first-iteration test
  was `n_cols == 0` and reset `n_cols` to 0, so the secant basis never grew and
  the method silently ran as Aitken.  With `jacobian_reuse > 0` it escaped only
  by admitting a bogus first column.
- **IQN safeguard vetoed valid steps on stiff problems**: the quasi-Newton
  correction was rejected above 10× the residual, but a correct Newton step is
  ≈ residual / (1 − ρ) (50× at ρ = 0.98).  Bound relaxed to a blow-up guard.
- **IQN secant basis**: `W` is now built from differences of the raw operator
  outputs (Degroote 2009) rather than of the inputs; 2 vs 5 iterations on the
  ρ = 0.98 test contraction.
- **IQN gradient NaN with `jacobian_reuse > 0`**: `jnp.linalg.lstsq`'s SVD
  derivative is NaN on the repeated zero singular values of the masked secant
  matrix; replaced by `jnp.linalg.pinv` (rank-deficiency-safe `custom_jvp`,
  same solution).
- IFT path ignored `convergence_norm` (hard-coded L2 against `tolerance`) and,
  for IQN modes, iterated with non-interface fields frozen at their first-pass
  values, giving the residual a floor and running to the cap.
- Fori-path diagnostics recorded the residual one iteration stale.
- IFT linear solve reported a spurious GMRES "iterative breakdown" on long
  runs: lineax's tolerance is elementwise, so exact-zero entries of a
  cotangent had to reach `atol=1e-8` absolute while float32 round-off from
  the large entries is ~1e-5.  The solve now goes through
  `jax.lax.custom_linear_solve` with `atol` scaled to the largest rhs entry
  and `rtol` no tighter than ~100 ulp of the dtype.
- `maddening.testing.strategies` rejected float32 bounds that are not exactly
  representable (e.g. `0.1`); bounds now round inward to the sampling dtype.

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
