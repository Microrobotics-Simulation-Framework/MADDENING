# Changelog

All notable changes to MADDENING will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Additional sections per release: **Verification**, **Security**, and **Known Anomalies**.

## [Unreleased]

See [`docs/release_notes/v0.4.0.md`](docs/release_notes/v0.4.0.md) for the
narrative release notes — measurements, design rationale and migration
guidance; the itemized changes follow.

### Added
- **`SimulationNode.static_data_deps()`** declares which parameters a
  `static_data` array was derived from; `compile()` now refuses a graph whose
  static derives from a *trainable* parameter — freeze it or stop deriving it
- **`maddening.sysid` contract properties** (`tests/property/test_sysid_contract.py`)
  over generated graphs; `windowed_loss` now rejects `sample_every <= 0` and a
  window wider than the data instead of returning a meaningless loss
- **Property tests for the sharded surface** (`tests/cloud/multigpu/`):
  wrapper-contract, sharded-equals-unsharded, halo-exchange and round-trip
  invariants over generated meshes; two audit findings pinned as strict xfails
- **Measured guidance for choosing coupling options**: eight graph fixtures and
  a full option sweep behind `docs/developer_guide/coupling_algorithm_guide.md`;
  `profile_graph` gains `n_stat_steps` to pin the coupling-statistics window
- **Coupling groups are serialisable**: `to_dict` / `from_dict` carry a
  `coupling_groups` key with all 19 `CouplingGroup` fields, and the USD stage
  carries the same set, so a reloaded graph solves the way the saved one did
- **Graph parameter pytree**: the compiled step is `step_fn(state,
  external_inputs, params)`, so node constants are traced inputs that
  `jax.grad` reaches and that change without a recompile.  Opt a node in with
  `update(..., *, params=None)`; `gm.nodes_without_params()` lists those still
  baking constants
- Migrated to the params contract: `SpringDamperNode`, `BallNode`, `HeatNode`,
  `RigidBodyNode`, `RigidBody2DNode`, `HeartPumpNode`, `TableNode`,
  `HealthCheckNode`, `LBMNode`, `LBMPipeNode`, `SurrogateNode` (network
  weights), flux producers via `compute_boundary_fluxes(..., params=)`, and
  `ShardedStencilNode` / `ShardedUnstructuredNode` (which previously ignored
  `gm.params` entirely)
- **`ParamSpec`** (`maddening.core.params`): per-parameter `trainable` /
  `bounds` / `transform`, declared in `SimulationNode.param_specs()` or
  `gm.set_param_spec`, with `gm.trainable_mask()`, `gm.unconstrain()` /
  `gm.constrain()` and `gm.check_params()` as the optimiser-facing maps
- **`maddening.sysid`**: `fit`, `fit_lm`, `fit_multiple_shooting`,
  `windowed_loss` and `fim` (`mask=`, `noise_std=`, Cramér-Rao bounds); all
  fitters emit `EVENT_FIT_PROGRESS`.  Guide: `docs/user_guide/parameters.md`
- **Interface mappings on edges**: `add_edge(..., mapping=)` with
  `rbf_mapping`, `nearest_neighbor_mapping`, `projection_1d_mapping`,
  `matrix_mapping`; weights live in `gm.params["mappings"]`, are reachable by
  `jax.grad`, and are `trainable=False` until you opt in
- **Interface mappings are serialisable** (`MappingSpec`): config, USD and the
  REST view carry mapped edges instead of refusing them.  Pass `source_ref=` /
  `target_ref=` to the factories; an incomplete spec is refused with the
  argument named
- Params survive persistence and reach FMI: `to_dict`/`from_dict`, USD and
  checkpoints store effective params and `ParamSpec` overrides;
  `build_model_description` exposes each leaf as an FMI `parameter`/`tunable`;
  `SidecarConfig(params=, param_specs=)` serves `get_params` / `set_params`
- REST `PUT /graph/params/{node}` addresses any leaf of the live pytree,
  updates in place without a recompile, and validates dtype, shape, finiteness
  and `ParamSpec` bounds before writing
- **FMU C wrapper, TCP bridge and packaging**: a graph now builds a real FMI
  3.0 co-simulation `.fmu` (`build_fmu_binary`, `write_fmu`), driven end to
  end by FMPy
- **FMU sidecar protocol 2 — binary frames** for bulk
  `get`/`set`/`get_state`/`set_state`, negotiated at `hello`; JSON-only
  clients are unaffected.  See "Wire protocol" in
  `docs/user_guide/fmu_export.md`
- **Multi-clock FMU export**: `build_model_description(multi_clock=True)`
  emits one `<Clock>` per distinct node timestep and tags outputs and inputs
  with theirs.  Off by default
- **`AdaptiveNode`** (`maddening.nodes.adaptive`, `MADD-NODE-009`):
  frozen-active-set adjoint pattern for adaptive solvers, with
  `gradient_capture_ratio`, `check_gradient_capture`, `mask_safe` and
  `set_adaptive_diagnostics`.  Read `MADD-ANO-003` before optimising through
  one.  Guides: `algorithm_guide/nodes/adaptive_node.md`,
  `developer_guide/adaptive_node.md`; benchmark `MADD-VER-004`
- **Per-neighbour unstructured halo exchange**: `exchange_unstructured(...,
  method="ppermute")` and `ShardedUnstructuredNode(..., exchange="ppermute")`,
  bit-identical to the `all_to_all` default; `exchange_traffic(layout)`
  reports what each transport would move so you can choose before using a GPU
- Multi-GPU session tooling: `benchmarks/multigpu/run_pod.py` (`--goal
  exchange|forward|gradient`, `--summarise`, `--dry-run`) and
  `benchmarks/multigpu/README.md`.  Nothing here launches a pod
- `sharded_cg` / `sharded_gmres` take `differentiable=True` for exact
  linear-solve adjoints through the same backend and preconditioner, plus
  `jacobi_preconditioner` / `block_jacobi_preconditioner`
- `SimulationNode.domain_integral_axes()`: reduce a domain integral over a
  subset of mesh axes, or not at all, so a partial-surface integral needs no
  full-mesh `psum`
- Second `@stability` wave: the unstructured partition layout, halo exchange
  and partition/gather helpers, `SidecarConfig` and `FMUState` are `EVOLVING`
- **Persistent compilation cache**
  (`maddening.core.simulation.compile_cache`): `enable(cache_dir)`,
  `MADDENING_COMPILATION_CACHE_DIR`, and `warm_cache()` to compile step and
  scan ahead of a run
- Profiler rewrite (coupling overhead measured not inferred, per-group
  iteration statistics, `trace=True` kernel attribution;
  `docs/developer_guide/profiling.md`) and `benchmarks/bench_coupling.py`
- `GraphManager.reset_state()` (reset without retracing the jitted step) and
  `GraphManager.get_node()`
- The USD stage stores a node's own name (`maddening:nodeName`) and an edge's
  declared units, so node names USD cannot spell survive a round trip
- Coupling diagnostics: iteration count and residual always in `_meta`,
  `coupling_diagnostics()` reports `"converged"` per group, and
  `CouplingGroup.strict_convergence` raises on an unconverged exit (off by
  default — the IFT gradient is invalid there)
- The IFT Krylov adjoint needs `lineax`, which is a base dependency as of
  this release (it was an optional extra when this entry was first written)
- Node verification: `verify_node` gains `params_consistent` /
  `params_gradient_finite` / `params_effective` and a `SKIP` status
  (`SimulationNode.accepts_params()` exposes the probe);
  `maddening.testing.verification` is a Hypothesis battery over outputs,
  structure, determinism, jit/eager agreement and gradients;
  `strategies.node_states` samples bool and integer fields
- Property-test coverage for round trips, the REST and FMU-bridge surfaces
  (stateful machines), the params pytree, `sysid`, retracing and binary frames
- **Static type checking, phase 1 (non-blocking)**: `pyrightconfig.json`,
  `pyright` in the `ci`/`dev` extras, a `continue-on-error` `typecheck` job
  and `scripts/typing_baseline.py`.  No source annotations changed; baseline
  and phase-2 plan in `docs/developer_guide/typing.md`

### Changed
- **`converged=True` means "within `tolerance` of the fixed point"**, not "the
  last step was small": the threshold is tested against `residual / (1 - rho)`
  and every norm is now relative, so expect more iterations and retune `atol`
- **Extended-precision point sets are refused, never silently narrowed**: a
  `MappingSpec` reference of `float128` / `np.longdouble` (unwritable as JSON,
  unstable to hash) raises; pass `np.asarray(points, dtype=np.float64)` instead
- **`solver="ift"` returns the iterate whose residual met the criterion**, as
  `fori` always has, so `converged=True` names the state you were handed and
  both solvers return it; every converged group's answer moves by one residual
- **Recorded `aitken` and `iqn-*` trajectories move**: Aitken's first pass of a
  timestep relaxes with the `omega` it was seeded with, not the clip floor 0.01
- **`coupling_diagnostics()["residual"]` describes the state the step returned**
  and no longer depends on `solver`; a group that arrives on its last pass now
  reports `converged=True` instead of raising under `strict_convergence`
- **The interactive path stops redoing host work**: sharded wrappers place
  their static arrays on device once, not per `update`, and `run_scan` and its
  siblings compile once per `compile()`, not per call (`gm.scan_trace_count`)
- **`lineax` is a base dependency**, not the `[ift]` extra: a coupling group
  at its default settings could not be differentiated on a base install.
  `pip install maddening` is enough; the now-empty `[ift]` extra still resolves
- **Version is now `0.4.0.dev0`** (was `0.3.1`) so a development build is
  distinguishable from the last release.  `maddening.__version__` prefers
  installed distribution metadata, so an editable install predating this
  keeps reporting the old version until reinstalled.
- **Coupling groups default to `solver="ift"`**, a `while_loop` that exits on
  convergence instead of always running `max_iterations`.  Set `solver="fori"`
  to keep the old behaviour
- **The IFT derivative rule is a `jax.custom_jvp`** (was `custom_vjp`), so
  `jax.jvp` / `jacfwd` and the FMI `FORWARD` directional derivative now work
  through coupled steps
- Node constants are no longer constant-folded, so `run_sweep` and an
  individual `run_scan` can differ by ~1 ulp where they were bit-identical.
  Loosen any bitwise assertion between the two
- **Resume-from-URL transport moved to `maddening.cloud.resume`**
  (`maddening.cloud.download_and_load_state`); the old path forwards and
  warns.  Behaviour, signature and errors are unchanged
- `AdaptiveNode` and `AdaptiveNodeBlindnessError` are `@stability(EVOLVING)`,
  not `STABLE`, and `ift_linear_solve` returns to `EXPERIMENTAL`; the 0.4.0
  freeze picks the final levels
- `AdaptiveNode.blindness_ratio` / `blindness_threshold` are
  `gradient_capture_ratio` / `gradient_capture_threshold`, and the cold-start
  check warns rather than raising except under `on_blind="raise"` or a
  confirmed Palais trap
- The `AdaptiveNode` gradient is documented as exact within an active-set
  region and first-order wrong across a switch; the previous "Clarke
  subgradient" claim was false (`MADD-ANO-003`)
- `AdaptiveNode.n_max` is structural and no longer in `params`, while
  `blindness_gate`, `on_blind` and the diagnostic constants are; a subclass
  whose basis size must round trip declares its own integer parameter.
  `compute_active_set` and `solve_frozen` are `@abstractmethod`, and the
  constructor validates `n_max`, the diagnostic constants and `dtype`
- `iqn_ils_update` takes a keyword-only `have_prev` flag saying whether the
  previous-iterate arguments are real
- Hypothesis is configured once in the root `tests/conftest.py`: `dev`/`ci`
  profiles selected by `MADDENING_HYPOTHESIS_PROFILE`, three named depth
  tiers, a persisted example database, and a `max_examples` house rule in
  `docs/developer_guide/testing_standards.md`

### Deprecated
- `maddening.core.simulation.calibration.calibrate` and
  `tune_coupling_params` warn and are removed in 0.5.0; use
  `maddening.sysid.fit`, which has `ParamSpec` bounds and a trainable mask
- `CouplingGroup.solver="fori"` emits `DeprecationWarning`; removed in the
  next minor release
- `maddening.core.simulation.checkpoint.download_and_load_state` warns and is
  removed in 1.0; use `maddening.cloud.download_and_load_state`
- `AdaptiveNode.blindness_ratio` / `blindness_threshold` warn; use
  `gradient_capture_ratio` / `gradient_capture_threshold`

### Removed
- The stelling formal-verification suite, CI job and `stelling` dependency.
  The `[verify]` extra now only pulls `hypothesis`.

### Fixed
- **A recompile no longer re-phases a multi-rate graph or restarts a coupling
  warm start**: `_meta` is preserved like node state and `params`, so a mid-run
  edit changes no trajectory; `reset_state()` is still the way to zero it
- **Every `CouplingGroup` knob its configuration ignores now warns** — `relaxation`,
  `jacobian_reuse`, `accelerated_fields`, `waveform_iterations`,
  `boundary_interpolation`, `linear_solver`, `strict_convergence` — at your call line
- **A `CouplingGroup` tolerance its norm never reads now warns** instead of
  turning silently: `tolerance` under `convergence_norm="mixed"`/`"interface"`,
  and `atol`/`rtol` under `"l2"`.  Set the knob the message names instead
- **`fim` reports an unidentifiable parameter as `+inf`, not a tight bound**:
  `crb` was `diag(pinv(F))`, which is small in the null space; the new
  `FIMReport.rank` counts the directions the data resolves (`rank_rtol=`)
- **A fit returns the leaves it did not fit, bit for bit**: `fit`, `fit_lm` and
  `fit_multiple_shooting` copy every leaf outside the mask from the starting
  pytree, so comparing before and after says exactly what a calibration touched
- **A wrapper now reports the `static_data` of the node it wraps**, so the
  `static_data` drift check finally fires through `ShardedStencilNode`,
  `HybridNode` and friends instead of hashing to `0` forever
- **`fit`/`fit_lm`/`fit_multiple_shooting` refuse a `mask` that names a leaf its
  `ParamSpec` freezes**: it was optimised unclipped in physical coordinates and
  could leave its bounds — make the parameter trainable in the spec instead
- **`strict_convergence` no longer ignores a diverged group that overflowed**:
  a NaN residual raises like any other failure instead of passing silently,
  which is what `coupling_diagnostics()` already reported for it
- **`gm.compile()` drops every node's materialised statics**, including one
  inside a wrapped node: `invalidate_static_cache` is now a `SimulationNode`
  method that forwards inwards, so a static rewritten in place is not baked in
- **Examples no longer save plots into the installed package** (they broke on
  a read-only install): output goes to the working directory, usage lines use
  `python -m maddening.examples...`, and a smoke test pins both
- **A coupling group no longer reports convergence it has not reached**:
  `acceleration="aitken"` needs the threshold met on two consecutive passes
  (a lone dip is not arrival), `max_iterations=1` reports its real residual
- Sharded pointwise nodes honour parameter writes again (`PUT /graph/params`)
- `POST /surrogate/deactivate` restores every edge field, or changes nothing
- A `.` in a node name no longer misroutes that node's FMU inputs and outputs
- **FMU export of a real graph had no inputs and a wrong step size**: inputs
  now come from the graph's external-input list as `<node>.<field>`, and the
  step is the graph's base timestep
- **FMU bridge and C wrapper robustness**: a non-whole-multiple communication
  step, an unusable step time, an oversized or truncated reply, an over-nested
  request, a second instance and a vanished peer no longer desynchronise,
  wedge or kill the importer; `set` is atomic, values are narrowed before the
  finiteness check, an unset input reads as zero, and memory-safety defects
  (leaked and unterminated buffers, a missing reply parsed as data) are fixed
  along with the fuzz harness that had missed them
- **FMI `min`/`max` for open bounds** advertised a value the sidecar then
  refused; the nearest representable float32 inside the interval is advertised
  instead
- **`compile()` no longer discards a calibration**: live params leaves are
  carried across a recompile (`gm.reset_params()` to undo), a partial pytree
  is completed from the live values, and `load_state` compiles before
  restoring
- **Params pytree correctness**: flux edges honour `params`, `ParamSpec`
  clamps stay strictly inside open bounds and reject NaN/inf, leaf dtypes and
  Python scalar types are preserved, wrong-shape leaves are refused rather
  than broadcast, and `fim` / `fit_lm` accept a per-leaf `noise_std` pytree
- **Interface-mapping serialisation hardening**: edge-key `param_specs` apply
  after the edges exist, stale point references are refused by content hash,
  asset loading is bounded and confined to `base_dir`, rebuild failures are
  `MappingRebuildError`, and two mapped edges on one field pair get separate
  slots via `EdgeSpec.ordinal`
- **Resume-from-URL hardening**: the manifest URL is derived from the URL
  *path*, so pass `manifest_url=` / `RESUME_MANIFEST_URL` for a presigned
  object; fetches time out (`MADDENING_RESUME_TIMEOUT`), stream to disk, and
  clean up their temporary directory
- **`AdaptiveNode` survives a config / USD round trip** (`blindness_gate` was
  dropped, `n_max` replayed as a duplicate keyword); the gradient-capture
  diagnostic evaluates the parameters in use, is memoised, is disabled by
  `MADDENING_ADAPTIVE_DIAGNOSTICS=0`, and the `jnp.where` tangent trap is
  warned about with `mask_safe` as the remedy
- **Coupling with non-float state leaves**: integer, boolean and PRNG-key
  leaves survive the IFT closure exactly, keep their dtype through
  `unflatten_coupled_state`, and no longer break reverse-mode differentiation
  through `lax.scan`
- **IQN acceleration**, four defects: IQN-ILS silently ran as Aitken without
  Jacobian reuse; the safeguard vetoed valid steps on stiff problems; the
  secant basis is now built from operator-output differences (Degroote 2009);
  the gradient is no longer NaN with `jacobian_reuse > 0`
- **Coupling diagnostics and solver settings**: the IFT path honours
  `convergence_norm` and no longer freezes non-interface fields in IQN modes,
  fori diagnostics are no longer one iteration stale, the IFT linear solve no
  longer reports a spurious GMRES breakdown, and Gauss-Seidel no longer raises
  `KeyError` for a flux edge whose consumer precedes its producer or for a
  flux source under IQN
- **Every run compiled the step three times** because weak-typed leaves
  restrengthened after the first step; `compile()` and `set_node_state()`
  normalise weak types (results bit-identical), and
  `_default_external_inputs()` no longer reallocates zeros every step
- **Sharded boundary inputs**: grid-shaped inputs are sharded and halo-padded
  instead of replicated, the shape heuristic compares the full leading grid
  shape, and `HeatNode.update_padded` takes `params=`
- Node fixes: `LBMPipeNode` multiphase used `np.exp` on a tracer and had a NaN
  gradient in the EDM velocity clamp; `HeatNode` ignored an injected `length`
- **REST API**: a non-finite constructor constant is a 400 naming the
  parameter instead of a 500 that leaves `GET /graph` broken for the process's
  life; `PUT /graph/params/{node}` echoes what it wrote and validates first;
  `PUT /graph/state` and `POST /graph/edges` validate their inputs; `add_node`
  refuses names containing `/`, `#` or `->`
- **Multi-GPU session runner**: the `exchange` goal timed an unsharded input
  and charged both transports the same constant, and `recommend()` accepted
  rows no real GPU produced; inputs are pre-sharded outside the timed region
  and a recommendation needs a real accelerator run.  Stale `jax[cuda12]` /
  Python 3.10 pins in `docker/Dockerfile.cloud` and the cloud examples are
  corrected
- **`scripts/typing_baseline.py` can no longer make a broken pyright run look
  like a result**: infrastructure failures exit 2, a wrong interpreter or an
  empty analysis is detected, and `pyright` is pinned to `1.1.414` in `ci`
- `scripts/generate_stability_report.py` imports every `@stability`-tagged
  module from a list a compliance test checks against the source, so none can
  silently drop out of the release gate's report
- Property-test flakes removed: the rollout-parity and interface-mapping
  adjoint assertions scale their tolerance with the magnitudes compared and
  pin the failing cases.  No library code changed.
  `maddening.testing.strategies` no longer rejects float32 bounds that are not
  exactly representable

### Verification
- **The coupled adjoint-identity property stops scaling by a cancelling
  inner product**: it divides by the norm of the terms contracted, not by
  the value they produce, which removes a latent float32 flake
- **Mapping-spec resolver, against generated input** (11 properties): a
  mutated spec is refused naming the edge or loads exactly the recipe on
  disk, and no generated asset path escapes the config directory
- **C-level tests for the FMU wrapper** (`tests/fmi/test_c_unit.py`,
  `tests/fmi/c/`): unit binary plain and under ASan/UBSan, a self-checking
  deterministic fuzz harness, valgrind, a libFuzzer campaign, FMPy against
  normal and hostile bridges, `validate_fmu`, and a `-std=c11 -pedantic`
  build.  CI installs valgrind and clang; each part self-skips if its tool is
  missing
- Full MADDENING test suite: 1680 passed, 3 skipped (1 deselected via `-m "not
  slow"`).  Slow-marked tests deferred to a longer pre-release pass
- Sharded `StaticArray` acceptance on a 4-device virtual mesh: bit-compatible
  with the single-device baseline, 50-step convergence, construction-time
  validation, `shard_info` delivery
- Edge-validation flip: 15/15 `tests/core/test_edge_validation.py` green, with
  shape and dtype errors raised in one `ExceptionGroup`
- Differentiable sharded solves and the C1 multi-physics IQN-IMVJ case match
  their dense and `fori` references in both differentiation modes

### Security
- **Mapping-spec assets are opened once** (`O_NOFOLLOW`, `fstat`): the size cap
  and the data now come from the descriptor that was checked, closing a
  time-of-check/time-of-use window for a writer in the config directory
- **FMU bridge no longer unpickles importer bytes** (CRITICAL): the FMU-state
  blob is an arrays-only `npz` validated before use — regenerate any stored
  blob.  `FmuSidecar.handle` stays pickle-based and trusted-clients-only
- **FMU sidecar `set_state` zip bomb** via an archive member without a `.npy`
  suffix: the archive directory is checked before `np.load` runs, with
  per-member and total declared-size caps
- **REST checkpoint endpoints are confined to a directory**: paths are
  relative to `SimulationServer(checkpoint_root=)` (default `./checkpoints`).
  The API still has no authentication — bind it to localhost or authenticate
  in front of it

### Known Anomalies
- MADD-ANO-005: `converged=True` is a residual test, not a bound on the
  distance to the fixed point -- calibrate it by re-solving at a 100x tighter
  tolerance (minor, open, context_dependent)
- MADD-ANO-003: AdaptiveNode frozen-set gradient omits a first-order term at
  active-set switches -- the frozen-set objective jumps where two candidates
  swap rank, so no Clarke subgradient exists there and the integral of the
  returned gradient misses the sum of the jumps crossed (measured: 33 % of the
  objective change over a 0.1-wide theta window at n_max=256, k=16; ~1e-8 at
  k=64) (open, context_dependent)

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
