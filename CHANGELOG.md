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
- **CycloneDX SBOMs** in `docs/validation/sbom/` for the base install and the `server`, `surrogates` and `usd` extras, from a
  clean install of the wheel (`scripts/generate_sbom.py`), checked offline against `pyproject.toml` and the SOUP package
  (`scripts/check_sbom.py`, in CI).  Regenerated from the release commit before each tag; see `soup_package.md` §6
- Multi-GPU session runner covers the sharding checklist: `run_pod.py --goal checklist` (`indivisible`, `halo`, `coupled`,
  `stencil`, `hybrid`), each against its unsharded, NumPy or float64 reference; every goal records pass/fail `checks`
  (schema 4), stops the session on a failure; `--summarise` re-derives every verdict and closes an item only on >= 4 real GPUs
- **`coupling_diagnostics()` gains `gradient_relative_error_bound`** (and `gradient_bound_usable`): the
  IFT gradient's relative error at an early exit, under `solver="ift"`, `diagnostics=True`.  About the
  gradient, not the solve -- it reads 0.0 on an affine group whose state is far off; read `spectral_error_bound` for that
- **`coupling_diagnostics()` gains a bound, `spectral_error_bound`** (with `rho_spectral`,
  `spectral_usable`): eight Arnoldi steps on `dF/dx` under `solver="ift"`, `diagnostics=True`; 8.05x
  *over* the true distance where `error_estimate` is 122x under.  Reported, not applied; existing keys unchanged
- **`WaveletAdaptiveNode`** (`maddening.nodes.adaptive`, `EXPERIMENTAL`): the first
  concrete `AdaptiveNode` — an interpolating-wavelet solver for `(-Δ + m) u = f`
  in 1/2/3-D with a CDD active set; declares and measures spatial order 2 by MMS
- **`fim(scale="nominal", specs=gm.param_specs())`: a column scale from the
  bounds width, not the value** — a parameter at `0.0` is judged on the data;
  columns with no finite width stay value-scaled and `FIMReport.value_scaled` names them
- **`sysid.fim_core` / `FIMCore`: the Fisher information with no host round
  trip** — jittable, zero device syncs, device-array verdicts for a control
  loop.  `fim` itself is fast only with `reuse_trace=True`, for a pure residual:
  ~1 ms warm against ~0.2 s per default call, which re-traces (see Fixed)
- **Docstring examples are executed in CI** (`scripts/check_doctests.py`):
  every `>>>` in `src/maddening` now runs, and the gate fails if the
  collection shrinks — an example that stops working is a failing build
- **The 0.4.0 stability freeze round** (`docs/developer_guide/api_freeze_proposal.md`)
  plus a guard that a `stable` signature cannot change unannounced
  (`scripts/check_stable_signatures.py`): 34 surfaces tagged, none promoted; `stability_report.md` is the full list
- **`py.typed`: MADDENING's annotations now reach your type checker.** Under
  PEP 561 they were all ignored downstream; nothing to do but upgrade. The
  pyright job is blocking too — `core`/`nodes`/`fmi`/`cloud`/… at zero errors
- **GCI / Richardson mode in `maddening.testing.mms`** for nodes MMS cannot
  reach: `assert_node_gci_verified(node, solution_at=..., levels=...)` needs
  no source term and no reference — three refinements give an error band
- **Draw-rejection audit** (`scripts/audit_property_rejection.py`): measures what
  fraction of each property test's Hypothesis draws `assume`/`.filter` throws
  away, and fails CI over the gate — run it before narrowing a strategy
- **`fim` says when `rank` was decided at the float32 noise floor**: a
  `PrecisionLimitWarning` naming the eigenvalue ratio, the cutoff and the
  `jax_enable_x64` re-run that settles it.  Quiet on well-conditioned problems
- **A performance regression gate on compile counts, not the clock**:
  `profile_graph` reports retraces, jaxpr primitives and lowered HLO ops;
  `python scripts/compile_counts.py --check` gates them and CI runs it
- **`ResolutionStatus.PARTIALLY_RESOLVED`** — MADDENING's own
  `known_anomalies.yaml` has used `partially_resolved` since MADD-ANO-005 was
  written; the enum could not represent the registry this project ships
- **`SimulationNode.static_data_deps()`** declares which parameters a
  `static_data` array was derived from; `compile()` now refuses a graph whose
  static derives from a *trainable* parameter — freeze it or stop deriving it
- **`maddening.sysid` contract properties** (`tests/property/test_sysid_contract.py`)
  over generated graphs; `windowed_loss` now rejects `sample_every <= 0`, a window
  wider than the data and observation leaves of unequal length instead of returning a meaningless loss
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
  `SidecarConfig(params=, param_specs=)` serves `get_params` / `set_params`; it and `set_fmu_state`
  refuse a non-finite value, one the leaf's dtype cannot hold or one out of bounds, as the bridge does
- REST `PUT /graph/params/{node}` addresses any leaf of the live pytree,
  updates in place without a recompile, and validates dtype, shape, finiteness
  and `ParamSpec` bounds before writing
- **FMU C wrapper, TCP bridge and packaging**: a graph now builds a real FMI
  3.0 co-simulation `.fmu` (`build_fmu_binary`, `write_fmu`), driven end to
  end by FMPy.  A bridge starts once: `start()` again, or after `stop()`, raises `RuntimeError`
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

### Changed
- **`compile()` refuses a group-internal flux edge the coupling loop reads from the state** -- under `convergence_norm="interface"`, and into a sub-cycled member under `boundary_interpolation="linear"` / `"quadratic"` (a bare `KeyError` inside the step since 0.1.0, MADD-ANO-060) -- and an IQN `accelerated_fields` naming no floating field; a non-floating field it names is dropped.
  Action: use the `"mixed"` / `"l2"` norm or `"constant"` interpolation the message names; name a floating field.
- **`scripts/report_test_durations.py` warns when an allowlisted test passes the 20 s hard line** (the allowlist has no ceiling), and a `[cold-ci]` request whose head commit cannot be read now runs cold with a warning instead of silently warm.
  Action: none; if a kept test's warning appears, re-check its allowlist reason. `docs/developer_guide/testing_standards.md` now lists the framework properties that only the slow lane checks.
- **`scripts/report_test_durations.py` labels each shard's compilation cache from the hits its report records**: `warm` only when a restored cache served most lookups, else `restored but unused (cold)`, with every shard's hit rate; it warns when a `--cache-mode off` run records cache lookups in more than one file, and no longer offers a skipped or failed allowlisted test as removable.
  Action: none; compare times by a shard's label, not by whether it restored a cache. Pull requests that add or remove a slow mark, remove a test or edit the allowlist now run CI cold.
- **`compile()` refuses a sub-cycled node whose timestep does not divide its group's largest** (to 1e-9 relative), which covered `round(macro/node_dt) * node_dt` per macro step and drifted silently (MADD-ANO-046). Action: give it a dividing timestep; the error names the two nearest.
- **`windowed_loss(mask_unconverged=True)` refuses a coupling group with no convergence slot** (`solver="fori"` with `diagnostics=False`), which the mask silently never masked. Action: set `diagnostics=True` on the group, or use `solver="ift"`.
- **New sharding refusals of silently-wrong input**: `ShardedUnstructuredNode` refuses a node with a non-empty `halo_width()` and one whose per-cell state is not one row per layout cell, and `partition_value` a value of the wrong length (MADD-ANO-037, -039);
  `ShardedStencilNode` refuses an empty `axis_map` (MADD-ANO-042); `halo_exchange` refuses a `boundary` dict key naming no exchanged mesh axis (MADD-ANO-041).
  Action: shard a stencil node with `ShardedStencilNode` (only a pointwise node goes to the unstructured wrapper for an indivisible grid); build the layout from the node's cells; map a mesh axis; fix the misspelt axis name.
- **`GraphManager.from_dict` warns when it rebuilds a sharded node unsharded**: a config carries no device mesh, so the node comes back as the node it wraps; the `UserWarning` names the wrapper, the settings the config recorded and the `replace_node` call that wraps it again (MADD-ANO-036, silent since 0.2.0).
  Action: re-wrap after loading when the run must be sharded; filter the warning when an unsharded reload is what you want.
- **New refusals where a sharded wrapper or `PUT /graph/params` accepted silently-wrong input**: the route refuses a value that changes a node's state shape (`n_cells`), a structural value the node's constructor refuses, `LBMPipeNode`'s geometry and a write no probe copy can decide; `ShardedUnstructuredNode` refuses a per-cell input on a full partition not in global order, and a state not in partition layout;
  `ShardedStencilNode` refuses an outer `boundary` other than a wrapped `ShardedStencilNode`'s, and a state its node no longer builds; `ShardedPointwiseNode` refuses a node with no state field on the shard axis.
  Action: rebuild a node to change such a value; renumber cells with `np.argsort(partition_assignment, kind="stable")`; pass the inner wrapper's `boundary`; shard an axis the state has, or leave the node unwrapped.
- **`ShardedStencilNode` refuses `boundary="zero"` and `"periodic"` for a `HeatNode`**: the rod now builds its own end ghosts, so the fill would be ignored.
  Action: drop the argument and hold an end at 0 with `left_temperature=0.0` / `right_temperature=0.0` (in a graph, `gm.add_external_input(...)`), exactly as unsharded.
- **`ShardedStencilNode` refuses a per-axis `boundary` dict**: it was accepted but never worked in the wrapper (it zero-filled the halos of axes
  `axis_map` leaves unsharded). Action: pass one mode as a string; per-axis modes remain available on `halo_exchange` itself.
- **Coupling group keys are also refused when one plus `_total` spells another's** (a new `_meta` slot, `<key>_total_iterations`): groups keyed
  `a+b+c` and `a+b+c_total` now fail at `add_coupling_group`, with or without sub-cycling. Rename a node so the keys differ.
- **`profile_graph` reports `coupling_overhead_ms` signed, alongside a new
  `coupling_overhead_se_ms`**: it was clamped at zero, which biased it upward
  and printed `0.00 ms` for an overhead the run could not resolve
- **Python floor is now `>=3.12`** (0.3.1 shipped `>=3.10`): jax 0.11 requires
  3.12 and there is no 0.12, so a 3.11 floor made half of `jax>=0.10,<0.13`
  uninstallable. CI now runs 3.12 against both jax 0.10.2 and 0.11.2
- **Docs**: the SOUP identification table gives JAX's *verified* version as well
  as its permitted range; `DESIGN.md`, `DatasetGenerator.from_graph` and
  `SurrogateValidator.compare_graphs` say they advance the graph they are given
- **Renderer and trainer config dicts are PEP 589 `TypedDict`s**: a misspelled
  key or a wrong value type is now a type error, and a scene object whose `"x"`
  names a state field no longer kills `setup()` with a `ConversionError`
- **The six `step` / `run_*` entry points document that they advance the
  graph's own state**: a second call continues from the first's final state,
  so measure from a fresh `GraphManager`; `run_sweep`, which does not, says so
- **`fit_lm` and `fit_multiple_shooting` hold the undetermined directions too**,
  as `fit` already did, so all three fill `FitResult.excited_rank`; the spring's
  `(k, c, m)` scale drifted 0.43-4.8% before. `hold_undetermined=False` opts out
- **`fit` holds the directions the data cannot determine at the values it was
  given**, so a degenerate combination no longer lands wherever `n_iter` and
  `lr` leave it; `FitResult.excited_rank`, or `hold_undetermined=False`
- **`soup_package.md` §3 counts *reachable* defects, not `open` tickets**: it
  counted only `open` and left every `partially_resolved` entry out; one with a
  live residual risk now counts, by the version-range gate's own predicate
- **`fim`'s rank cutoff now sees the residual length**: `rank_rtol` defaults to
  `max(n, sqrt(m)) * eps`, not `n * eps`.  `rank` falls and `crb` goes `+inf`
  for some long-residual (`m > n**2`) fits; pass `rank_rtol=n * eps` to opt out
- **`coupling_diagnostics()` renames `bound_valid` to `ratio_usable` and
  `gradient_error_bound` to `gradient_error_estimate`** — the flag reports one
  of the four conditions the estimate rests on, not that it is a bound
- **`FitResult` is keyword-only**, the guard `FIMReport` got this release:
  no field has been inserted into it yet, and inserting one would silently
  swap `converged` and `n_iter` for any positional caller
- **Breaking:** `FMIVariable` is keyword-only (0.4.0 inserted `node` / `field`
  between `unit` and `shape`, so a positional call silently bound the wrong
  fields) and `load_graph_from_usd` gained `node_registry=` / `allow_import=`
- **`FIMReport` is keyword-only**: `rank` was inserted mid-dataclass this
  release, so positional construction silently reassigned every field after it;
  build it with keywords (every in-tree caller already did)
- **`AdaptiveNode` tightens its subclass contract and loosens its gate**:
  `compute_active_set` must return a non-empty **boolean** mask; `is_trapped_at`
  is now `frozen_gradient_vanishes_at`; `on_blind="warn"` no longer ever raises
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
  under either `solver`, agreeing between them only to its float32 noise floor
  (see Fixed); a group that arrives on its last pass now
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
  check warns rather than raising unless `on_blind="raise"`.  Like `is_trapped_at`
  and the `bound_valid` / `gradient_error_bound` keys above, these are renames
  within the 0.4.0 cycle: no release carried the old names, which still warn
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

### Removed
- The stelling formal-verification suite, CI job and `stelling` dependency.
  The `[verify]` extra now only pulls `hypothesis`.

### Fixed
- **`ParamSpec.from_dict` refuses a non-boolean `trainable` or a non-numeric bound** (`"false"` read as trainable, `true` as 1.0): a saved
  graph carrying one fails to load, a USD stage warns and skips it.  **`run_pod.py`'s stencil, hybrid and coupled goals now fail on a
  broken stencil wrapper** (four seeded faults; schema 5, pencil mesh, D2Q9).  Action: write JSON booleans; re-run schema-4 dry runs.
- **Coupling, second audit**: every acceleration acts on floating fields only (an integer, boolean or PRNG-key leaf raised a `TypeError` under `aitken` / `fixed` / IQN since 0.1.0, and `solver="fori"` rounded it through float32 during 0.4.0 development; MADD-ANO-059); `strict_convergence` under `run_adaptive*` raises only about a solve the stepper keeps; `windowed_loss(mask_unconverged=True)` drops a diverged window from the gradient too (it made the gradient NaN);
  `coupling_diagnostics()` judges the last step under the group that took it, not a replacement; after `jax.grad` of `run_scan` the graph is put back whatever its first node (`step()` raised `UnexpectedTracerError` on a coupled graph with lowercase node names); `run_adaptive` advances its clock by the step a `dt_min` accept keeps, not `dt_min` (MADD-ANO-061).
  Action: re-run `solver="fori"` results from accelerated groups holding integer state, masked sysid fits that saw a NaN gradient, and `run_adaptive` runs that warned "hit dt_min".
- **Sharded stencil wrapper and LBM nodes, round-3 audit** (MADD-ANO-056, 057, 058): `shard_info`'s block size comes from the grid, not a domain integral in the state (a sharded `HeatNode` carrying a vector energy integral left its right end open, 70.9 K off); a nested `ShardedStencilNode` keeps the node's parameters; a replicated `heat_source`/`body_force` the unsharded node refuses is refused sharded (it was applied on every shard);
  `LBMNode`/`LBMPipeNode` refuse a non-finite viscosity or `tau`, a `propeller_x` outside the grid and a disc covering no cell (each ran with no collision or no propeller).  Action: re-run sharded results whose node carries an integral in its state; give pipes of 10 planes or fewer an explicit `propeller_x` (the default is 10).
- **Compliance gates, second mutation audit**: `check_anomalies` requires `residual_risk` on a `partially_resolved` entry, and `_RETIRED_ANOMALY_IDS` is an `{id: reason}` dict that cannot retire an entry last committed as reachable (read from git; CI's compliance job now fetches full history); `check_impl_mapping` refuses a row traced to a class;
  `check_sbom` refuses an orphan component, a recorded Python outside `requires-python` and a missing licence; `check_citations` reads in-text `@Key`; `check_stable_signatures` calls a parameter a recorded `*args`/`**kwargs` used to catch breaking.
  Action: give a retirement its reason; begin a mapping row that means a class with ``Class `Name` ``; write a decorator in prose as code.
- **`PUT /graph/params` answers 200 only for a write a saved graph reproduces** (MADD-ANO-047, 048, 049): the constructor is asked with every changed key and the live values a save carries (a live leaf skipped it: a `HeatNode` past its Fourier limit, `rho_gas > rho_liquid`); a write the running node would honour unlike its rebuild is refused
  (`LBMPipeNode`'s `G` crossing 0; `HeatNode` now fixes its grid at construction, so `grid_points` on a uniform rod is refused); a non-finite value is a 400 before any write (it was stored, then a 500 on every GET); params carry POST's 422 bounds, and a
  state over the cap or of another layout is refused before anything that size is built.  Action: rebuild a node to change such a value, and check that configs saved after a REST write still load.
- **`profile_graph(measure_coupling=True)`** no longer repeats the caller's compile-time warnings (a disconnected node, an inert knob) from its variant and restore recompiles. Action: none.
- **Coupling runtime, audit of the frozen tree**: `run_adaptive*` sub-steps a sub-cycled node at `dt * node_dt / macro_dt` (it advanced `divider * dt`, MADD-ANO-043); on a multi-rate graph a group's diagnostics, predictor history and IQN-IMVJ warm start come only from the solves the step keeps, `strict_convergence` checks only those, and the group no longer solves on the base steps that discarded the result (MADD-ANO-044);
  `solver="fori"` + `iqn-imvj` carries the latching pass's secant columns, not zeros (MADD-ANO-045); `converged` is one verdict, in the residual's dtype, in the report, the profiler, sysid and strict; the profiler samples a multi-rate group on its firing steps only and stops re-warning about its one-iteration variant.
  Action: re-run `run_adaptive*` results with a sub-cycled group, and multi-rate results with a coupling group using `predictor` or `iqn-imvj`.
- **Sharded wrappers, round-2 audit**: `ShardedUnstructuredNode` no longer steps a Cartesian stencil node on the partition layout (a `HeatNode` rod ran 43 K off; MADD-ANO-037, since 0.3.0) nor drops a node's cells past the layout's count, and `shard_info["n_local"]` gives each shard its own cell count so an integral can leave the padding out (MADD-ANO-039, since 0.3.0);
  both wrappers place a domain integral carried in the state as the step returns it, so an integral-emitting node runs in a graph (MADD-ANO-040, since 0.2.1); `ShardedStencilNode` fills a sharded static's global halos periodically under `boundary="periodic"` (MADD-ANO-038, since 0.2.1).
  Action: re-run periodic sharded results whose node reads a static in its halo; mask a domain integral on an uneven partition with `jnp.arange(n_local_max) < shard_info["n_local"]`.
- **FMU bridge and the multi-GPU session verdict**: `FmuTcpBridge.stop()` returns within five seconds and logs any worker still inside a request, which then commits nothing; `set`/`get` refuse a non-integer value reference (`10.9`, `"10"` and `true` addressed variables 10 and 1); `FmuSidecar.set_fmu_state` refuses a snapshot whose nodes, fields or parameters differ from the model, as `set_state` does, and `set_state` keeps a node without state fields.
  `run_pod.py --summarise` closes a checklist item only from files on the current schema, from one commit, with `n_devices` within the devices they saw, holding every case the runner runs and exactly the checks it derives from their results under `LIMITS`; other files read `INVALID`.
  Action: send integer value references; restore snapshots that carry the model's parameters; re-run session goals whose files come from another commit or runner version.
- **`LBMNode(wall_mask=...)` keeps its walls through a save/reload**: the mask was not in `params`, so `from_dict` and a USD stage rebuilt the node with no walls and the reloaded graph ran an open domain, with no error (MADD-ANO-034, since 0.1.0).  The mask is recorded as nested lists of bool, and `POST /graph/nodes` takes it;
  `AdaptiveNode(dtype=...)` is recorded too (a reload came back at the canonical float).  Every built-in node's constructor arguments are now checked through a JSON and a USD reload.
  Action: re-run results from a reloaded graph that holds a walled `LBMNode`.
- **The coupling guide is re-measured on the release tree** (`benchmarks/results/coupling_sweep*_cpu.json`): its "fixed point that barely moves" row now
  starts from `gauss-seidel`/`none`, not Jacobi at ω = 0.8; the heat benchmark fixtures run at half their Fourier number (the rod-end fix doubled
  their gain; slab pairs are unstable above Fo = 3/8); `boundary_interpolation` modes can differ by O(tolerance), not "bit-identical"
- **Sharded wrappers, audit of the frozen tree**: a params write followed by `compile()` reaches the sharded step, for a legacy-contract node's constant and for a halo width that follows a parameter (`HeatNode.stencil_order`) (MADD-ANO-032, since 0.2.0); `ShardedStencilNode` keeps `dt` at the graph's precision under x64 (MADD-ANO-033, since 0.2.0); `HybridNode` and `ShardedUnstructuredNode` forward `update_evaluations()`;
  `ShardedUnstructuredNode` validates `domain_integral_axes` names.  The routes MADD-ANO-024's first fix left open (a part-full pipe's `pipe_radius`, `n_cells`, `stencil_order=3`, wrapped and unprobeable nodes) are closed.
  Action: re-run sharded results whose structural parameters were written after the wrapper was built, and sharded float64 stencil runs under x64.
- **CI test-time report** (`scripts/report_test_durations.py`): a cache read is no longer subtracted twice and is shown apart from compile; a share is capped at 100%; a test that started a process is listed as "work in a subprocess
  (not measured here)", not "slow even with a warm cache"; a report holding only collection errors exits 2 instead of passing.
  Action: none; a slow test's split in the lane summary now adds up, and a subprocess test is no longer sent for a code change a warm cache would make unneeded.
- **Compliance gates catch the defects a mutation audit slipped past them**: `check_doctests` pins every file's example count (`EXAMPLES_PER_FILE`) and it and `check_impl_mapping` sit at the counts; `check_anomalies` refuses a `pytest.skip()`/`xfail()` in a cited test, an aliased mark, `skipif(True)`,
  a broken first-party import and evidence under `tests/viz`, and counts imperative conditional skips; `check_citations` reads digit-led, line-wrapped and `@comment`-hidden keys; `check_heat_stability` judges `grid_points=None`;
  `generate_soup_tables --check` refuses a duplicated block. Action: a new docstring example goes into `EXAMPLES_PER_FILE` in `scripts/check_doctests.py`.
- **Sharded stencils at the edges of the global grid**: a size-1 mesh axis got periodic halos whatever `boundary` said (a one-device `HeatNode` ran as a ring, 0.40 off); `"edge"` wider than one cell put `r0, r1` before `r0`;
  a sharded `HeatNode` ignored `left_temperature`/`right_temperature` (0.87 off with both ends at 0) and now closes its rod ends exactly as unsharded; `ShardedStencilNode` refuses an unknown `boundary` at construction.
  Action: re-run sharded results taken on one device or a size-1 mesh axis, and every sharded `HeatNode` result with end temperatures or `stencil_order=4`.
- **`coupling_diagnostics()` counts every `waveform_iterations` sweep** (since 0.1.0, MADD-ANO-026): `iterations` is the largest sweep's, so `iterations >= max_iterations`
  is exact again (an earlier sweep at the cap read `iterations=1`); new `total_iterations` is the sum. State unchanged; a one-sweep group reports as before.
  Action: a cap check on `iterations` needs no change; read `total_iterations` for the work done.
- **Coupling bounds, confirmation audit:** a field below `tiny/eps` (~1e-31 in float32) no longer reads converged on a flushed change; the float floor counts the evaluations a pass rounds like (sub-cycling automatically,
  internal loops via the new `SimulationNode.update_evaluations()`; an undeclared node's group gets `spectral_usable=False` at the floor); float16-beside-float32 gradient floors, top-of-range spectral weights,
  `PYTHONHASHSEED`-dependent `_meta` seeds and `reset_state` of a key ending `_spectral` are fixed. Action: a node that sub-steps inside `update` should return its sub-step count from `update_evaluations()`.
- **`AdaptiveNode.update` refuses an injected `params` key it does not have**, for every subclass: `{"thetta": 0.9}` was merged,
  never read, and returned the constructor answer; fix the key the error names.  MADD-ANO-004 now also records `ShardedStencilNode`
  (0.2.0-0.3.1) and `ShardedUnstructuredNode` (0.3.0-0.3.1) ignoring a REST parameter write: on those releases set it on the inner node.
- **`PUT /graph/params` refuses a value the running node cannot use** (400, nothing written; since 0.1.0 it was saved and ignored, e.g. `LBMPipeNode.pipe_radius`): rebuild
  the node. `/checkpoint/load` refuses such a checkpoint; FMUs export only parameters the step reads and refuse the rest (pass `SidecarConfig(fixed_params=md.fixed_parameters)`).
  `params_effective` fails a constant split with an `__init__` copy; a `transform="log"` spec with no lower bound refuses values `<= 0`.
- **`LBMNode`'s algorithm ID is `MADD-NODE-011`** (it shared `MADD-NODE-007` with `RigidBodyNode`, which keeps it): update anything keyed on the old ID.
  The gates now refuse a duplicate node ID, a guide ID that differs from its `NodeMeta`, a `<FIX` that is not the entry's `resolution_version` or not a real release,
  a duplicated / skipped / uncollected `verification:` test, and a rod built through a local `HeatNode` subclass; the doctest and mapping floors sit at the current counts.
- **A sharded `LBMNode` is the unsharded node or refuses**: a pressure face on a sharded axis raises at the first step; `ShardedStencilNode` refuses a `boundary` other than a node's declared `halo_boundary()` (LBM: `"periodic"`), its default `"edge"` included; `inlet_face == outlet_face` is refused.
  `WaveletAdaptiveNode(frozen_solver="cg")` is bounded by `kappa * rtol` and a non-converging eager solve says why; inline points refuse complex values (any width), and without a `"dtype"` ints beyond 2**53 and inexact `Decimal`s.
  Action: wrap `LBMNode` with `boundary="periodic"` and shard an axis with no pressure face; use `frozen_solver="gather"`; add an explicit `"dtype"` (e.g. `"int64"`) to inline points.
- **Coupling bound keys no longer under-read**: `spectral_error_bound` adds the residual's float floor (new `precision_limited`); the gradient bound is `inf` where Newton-Kantorovich fails.
  `converged` still reads `True` on a stalled float32 iterate: read those keys with `diagnostics=True`. A NaN no edge reads, or an underflowing scale, no longer reads converged;
  a group with no step yet has no report; colliding group keys (node names containing `+`) are refused.
- **An inline point reference with no `"dtype"` refuses `np.longdouble` values** instead of rounding them to float64
  without a word; the error says no reference form keeps extended precision. Add `"dtype": "float64"` to accept the
  rounding, or convert first (`np.asarray(points, dtype=np.float64).tolist()`); float64 payloads are unaffected
- **Params contract audit:** one "takes `params`" rule for every probe (`update_padded(**kwargs)` calibratable under `ShardedUnstructuredNode`; duck-typed nodes verified);
  `params_effective` probes each path and vector element by value; a changed `gm.params` leaf the step cannot read (`initial_*`, `static_data_deps`) is a `ValueError`, never serialised; int-spelled
  declared constants kept; sharded wrappers forward flux/interface hooks; `run_adaptive*` resolve flux edges. Action: rebuild the node rather than edit such a leaf; fix nodes `params_effective` now fails.
- **`LBMNode`'s Zou-He pressure faces impose the pressure they are given** (MADD-ANO-020, every release): the face carried
  `p/cs2 + S_K` (+15%), a pressure-driven channel 0.58-0.80 of the imposed drop; pressure-driven results change, re-run them.
  `outlet_pressure_avg` reads the runtime wall mask; `LBMPipeNode` gains non-trainable `initial_rho_liquid`/`initial_rho_gas`.
- **`fim` no longer reuses a compiled trace across calls unless `reuse_trace=True`** (pure residuals only): the cache
  froze what the residual read, e.g. `gm.params` (a CRB of 0.41 for a true 44.7).  Specs resolve by one walk (namedtuple
  levels by field name; `check_bounds` refuses unreadable specs); integer leaves are named, not zeroed; NaN bounds refused
- **`WaveletAdaptiveNode` refuses a `mass` its dtype cannot carry** (float32 `mass=1e-6` read J 2.8x off, silently) and its
  active set no longer depends on rounding: eager, jit and graph agree, as do a float and its array spelling. Periodic
  source periodised; `n_levels=6.9` and unknown keys refused. Build in float64 or raise `mass` if refused.
- **Compliance gates no longer pass an empty or unreadable scope**: `check_transforms` / `check_stable_signatures`
  fail when nothing is verified; `check_doctests` floors executed examples and fails a `+SKIP`. Without the extras,
  run `check_transforms.py --allow-missing-optional`; drop a STABLE surface with `--update --accept-removal`.
- **Every anomaly's `affected_versions` is checked against the registry's own version** (PEP 440; convention in the
  registry header): ANO-016 is open-ended again, ANO-006 closes at 0.4.0, ANO-004 starts at 0.1.0, ANO-018/019 read `none`.
  A `verification:` `path::Class::method` must now name a method of that class; fix any loose node id in your own registry
- **A coupling iteration that diverged to NaN/inf was reported `residual=0.0, converged=True`** by both
  solvers and all three norms (MADD-ANO-019 resolved); a non-finite field now fails the criterion (`residual=inf`,
  `converged=False`) and `strict_convergence=True` raises naming the non-finite state. No action needed.
- **`HeatNode.compute_interface_correction(params=)` reads the injected `length`** (a calibrated length
  left coupled interface cells 11 K off, silently); `params_effective` uses a seeded projection (a conserving
  node passes; old-signature subclasses FAIL under `assert_node_verified`); `**kwargs` overrides take `params`
- **`WaveletAdaptiveNode`: the default budget `k` now holds the whole CDD seed** (every
  level-0 function, not `n_coarse**dim`): 16 default configurations silently truncated the
  gathered solve (`dJ/dθ` 5x off at `dim=2, n_levels=2`); `k` below the seed is now refused
- **`fim(scale="nominal")` refuses a `specs` that does not mirror `params`** (misspelt key,
  `to_dict()` entry, wrong nesting, non-dict) by key path instead of silently reporting `"relative"`;
  a spec keyed for a list/tuple leaf now reaches `fim`, `trainable_mask`, `unconstrain` and `check_bounds`
- **A calibrated `params` now reaches `derivatives()`, `implicit_residual()`,
  `integrate_node(..., params=)` and `implicit_euler_step(..., params=)`** (MADD-ANO-018
  resolved); a non-empty `params` for an override without the keyword is a `ValueError`, not a silent drop
- **Docs said `linear_solver="dense"` was the fallback when the GMRES adjoint
  struggles**: it needs `2*N^2*itemsize` and on a grid-coupled group is refused
  outright (523 GB at ~3.6e5 DOF). The option is unchanged; the advice is not
- **`fit_lm` no longer runs a whole rollout it throws away**: the residual was
  evaluated for a pytree structure `jax.eval_shape` can trace, once per call
  even with `noise_std=None`, and at `n_iter=0` it was the only call made
- **Three more gates that could not fail**: the pyright tiers now cover
  `maddening/__init__.py`, the SOUP drift classifier no longer calls a lost evidence
  row a safe regenerate, and `check_transforms` counts only confirmed references
- **`cloud/_skypilot.py` ported to SkyPilot's client-server API** (`MADD-ANO-016`):
  every call in it was written against the pre-0.7 API and could not work on the
  `>=0.11` floor, and a failed teardown or preemption check is now loud
- **Audit fixes on the serialisation surfaces** (`MADD-ANO-010`): an FMU or node named
  `NaN`/`Infinity` is refused where you set the name, an unencodable FMI reply is an error
  reply rather than a dead worker thread, and `dumps_encoded()` writes `to_dict()` output
- **`HeatNode` reports its boundary flux at the rod end**, not a cell inside:
  measured order 1.005 -> 1.999 (and 3.993 for `stencil_order=4`), and the units
  are `K*m/s`, not `W/m^2`. Flux-coupled results move by up to 10%
- **The compliance gates can now fail on the defects they exist to catch**: the
  anomaly and benchmark ID sets are pinned, four gates that reported success
  having verified nothing now fail closed, and two stopped inflating their counts
- **`rk4`/`heun` are 1st order with a time-varying boundary input** (`MADD-ANO-014`):
  every stage holds the one value you pass. The arithmetic is unchanged; carry time
  as a state field with derivative 1 to get 4.00 back — see the new solvers guide
- **`HeatNode` imposes its Dirichlet data at the rod ends, not half a cell in**,
  and its 4th-order ghosts sit where the stencil reads them: measured spatial
  order 1.001 -> 2.000 and 0.954 -> 3.957. `T[0]` no longer equals the datum
- **Non-finite numbers are written as valid JSON** (config, USD `paramsJson`, FMI
  wire): `NaN` / `Infinity` / `-Infinity` are quoted, and both spellings load
- **An FMU instance may reconnect at once**: the bridge no longer refuses the slot
- **A failed `compile()` leaves the graph exactly as it was**: schedule, rate
  dividers, external-input zeros and `params` commit after the last refusal.
  `auto_couple` keeps groups it cannot replace; `run_adaptive` checks after it
- **`coupling_diagnostics()["residual"]` has a float32 noise floor**, documented:
  a converged group's residual is a cancellation, so `solver="ift"` and `"fori"`
  can report `0.0` and `1e-05` for one state.  The state and verdict are exact
- **A compliance gate no longer calls an optional subpackage's symbol stale**:
  a `maddening.usd.*` reference in an environment without the `usd` extra is
  reported as *not checked*, with the extra named, instead of failing the run
- **A coupling group no longer reports `converged=True` with a small field far
  from its fixed point** — `atol` removes a field from the norm, so it defaults to
  `0.0` and is live under every norm — and `iterations` at the cap agrees by solver
- **A recompile no longer re-phases a multi-rate graph or restarts a coupling warm
  start** — only graphs with a rate divider > 1 or a warm start were ever affected;
  `set_param_spec` and `external_inputs` are now checked as strictly as `params`
- **Swapping a surrogate in or out no longer resets an edge's `additive`, units,
  `mapping` or fitted mapping weights**: an additive input read 3.0 before a swap and
  1.0 after.  Re-check results crossing `replace_node` / `POST /surrogate/deactivate`
- **`jax.grad` no longer crashes on a stiff coupling group**: a failed GMRES
  adjoint re-solves directly at small DOF, or names `linear_solver="dense"`
- **`error_estimate` accounts for `relaxation`** — and is an estimate, not a bound
- **Degenerate sysid inputs are refused, not reported**: a non-finite Fisher matrix,
  a σ that is not positive, a mask keyed unlike `params`, and `lr`/`eps`/`lam_up`
  values that invert their meaning now raise; `params_pytree` keeps float64 under x64
- **The four pre-0.4.0 `scripts/check_*.py` compliance gates now fail on the defects they
  exist to catch** — zero-reference transform scan, MRO-resolved mappings, a
  `%`-commented bib entry, an unchecked `resolution_status`.  Re-run them
- **A failed graph mutation is now a no-op**: `add_node` builds the state before
  it registers the node, so an `initial_state()` that raises no longer wedges
  the graph with a ghost `step()` dies on; same for `reset_state`/`remove_node`
- **A failed resume now leaves the graph untouched** and logs `RESUME FAILED`, not
  "starting fresh"; sharded wrappers honour `shard_axes`, pass `params` to a
  `**kwargs` node (whose gradient was silently zero) and refuse an indivisible grid
- **Do not pair `convergence_norm="interface"` with the auto-detected
  `accelerated_fields`**: both are the edge fields, so an accelerator fixes
  exactly what the criterion measures — use a norm that sees the whole state
- **Every `CouplingGroup` knob its configuration ignores now warns** — `relaxation`,
  `jacobian_reuse`, `accelerated_fields`, `waveform_iterations`,
  `boundary_interpolation`, `linear_solver`, `strict_convergence` — at your call line
- **A `CouplingGroup` tolerance its norm never reads now warns** instead of
  turning silently: `tolerance` under `convergence_norm="mixed"`/`"interface"`,
  and `rtol` under `"l2"`.  Set the knob the message names instead
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
- **MADD-VER-002's acceptance band no longer admits the defect it measured**:
  `[0.7, 2.5]` -> `[1.7, 2.3]` around the theoretical 2.0 (measured 1.900); it
  cited MADD-ANO-002 for a boundary defect. MADD-VER-001: 5% -> 1e-4
- **Four more nodes measured rather than skipped** (MADD-VER-009..012): `spring`,
  `ball`, `rigid_body_2d` and `heart_pump` declare a temporal order and meet it
  (1.029/1.002/1.000/1.000 against 1.0); 12 mutations confirm the ladders can fail
- **`LBMPipeNode` has a convergence verdict for the first time**
  (MADD-VER-013): the ladder converges, but a fourth level shows it is *not*
  in the asymptotic range, so no error band is quoted for it
- **Order of accuracy is measured, not asserted** (`maddening.testing.mms`):
  declare `NodeMeta(discretization_order=...)` and the harness refines a
  manufactured solution and fails the node on a shortfall (MADD-VER-005..008)
- **The committed stability report is compared with a fresh generation** in
  CI: it had rotted to 42 of 85 surfaces, hiding every deprecation.  Four
  surfaces that warn deprecated now carry the `DEPRECATED` tag
- **The coupled adjoint-identity property stops scaling by a cancelling
  inner product**: it divides by the norm of the terms contracted, not by
  the value they produce, which removes a latent float32 flake
- **Mapping-spec resolver, against generated input** (10 properties): a
  mutated spec is refused naming the edge or loads exactly the recipe on
  disk, and no generated asset path escapes the config directory
- **C-level tests for the FMU wrapper** (`tests/fmi/test_c_unit.py`,
  `tests/fmi/c/`): unit binary plain and under ASan/UBSan, a self-checking
  deterministic fuzz harness, valgrind, a libFuzzer campaign, FMPy against
  normal and hostile bridges, `validate_fmu`, and a `-std=c11 -pedantic`
  build.  CI installs valgrind and clang; each part self-skips if its tool is
  missing
- Full MADDENING test suite at `1ad3fa2`: **6386 tests collected** locally, 6198
  under `-m "not slow"` (188 deselected); CI's run at the same commit passed 5986
  (jax 0.10.2) / 5985 (jax 0.11.2) — see `docs/release_notes/v0.4.0.md`.  v0.2.1's own
  Verification block, which an edit during this cycle had moved here, is back
  under [0.2.1] as released
- Differentiable sharded solves and the C1 multi-physics IQN-IMVJ case match
  their dense and `fori` references in both differentiation modes

### Security
- **Cloud launches reach ready again, and cross-origin browser requests are
  refused**: set `MADDENING_TRANSPORT_TOKEN` so the ZeroMQ CURVE key is not the
  cleartext API bearer token; pass `allowed_origins=` to embed the UI elsewhere
- **The ZeroMQ transports bind loopback and encrypt any other bind** (CRITICAL, MADD-ANO-015):
  5555/5556/5580 published state, commands and worker rendezvous to anyone who
  could reach them; set `MADDENING_TRANSPORT_TOKEN` on both sides to stream off-box
- **The API requires a bearer token unless it is bound to loopback** (CRITICAL, MADD-ANO-051):
  set `MADDENING_API_TOKEN` or read the one logged at start-up; `JobConfig.ports`
  no longer defaults to `[8000]`, so a cloud launch stops opening the API port
- **FMI/USD hardening** (three HIGH): a silent TCP peer no longer wedges the FMU
  bridge, `set_state` is value-checked exactly as `set` is, and loading a USD
  stage no longer imports the class it names — pass `node_registry=` to allow one
- **Cloud surface**: the signaling WebSocket validated its own token, not the
  client's (CRITICAL, MADD-ANO-053; set `MADDENING_STREAM_SECRET`), and the unauthenticated
  API now caps `n_steps`, node dimensions and training args, warning on 0.0.0.0
- **Mapping-spec assets are opened once** (`O_NOFOLLOW`, `fstat`): the size cap
  and the data now come from the descriptor that was checked, closing a
  time-of-check/time-of-use window for a writer in the config directory
- **FMU bridge no longer unpickles importer bytes** (CRITICAL, MADD-ANO-054): the FMU-state
  blob is an arrays-only `npz` validated before use — regenerate any stored
  blob.  `FmuSidecar.handle` stays pickle-based and trusted-clients-only (MADD-ANO-055)
- **FMU sidecar `set_state` zip bomb** via an archive member without a `.npy`
  suffix: the archive directory is checked before `np.load` runs, with
  per-member and total declared-size caps
- **REST checkpoint endpoints are confined to a directory** (MADD-ANO-052): paths are
  relative to `SimulationServer(checkpoint_root=)` (default `./checkpoints`).
  Since this release the API *does* authenticate on a non-loopback bind
  (bearer token, see the Security entry above); loopback is unchanged

### Known Anomalies
- **MADD-ANO-059, 060, 061 (new, resolved in this release)**: accelerating a coupling group holding an integer, boolean or PRNG-key leaf raised a `TypeError`; a group-internal flux edge read by the interface norm or a sub-cycled member's linear interpolation raised a bare `KeyError`; `run_adaptive` advanced its clock by `dt_min` on a `dt_min` accept whose state covered more (all since 0.1.0; see `### Fixed`, `### Changed`).
  **MADD-ANO-027** now says each extra waveform sweep applies at least one more pass, moving a converged state by about one residual
- **MADD-ANO-051, 052, 053 (new, resolved in this release; all since 0.1.0)**: the HTTP API served every route, `/cloud/launch` included, with no credential while the container bound `0.0.0.0`; the checkpoint routes took any server path; the signaling server admitted every client (see `### Security`).
  **MADD-ANO-054 (new, never released)**: the FMU bridge unpickled the importer's state blob, remote code execution; **MADD-ANO-055 (new, resolved)**: `deserialize_fmu_state` and `FmuSidecar.handle` unpickled their input (since 0.3.0).
  The registry now holds every defect a release carried and every critical or major one found in the cycle, shipped or not (CONTRIBUTING.md)
- **MADD-ANO-047, 048, 049 (new, resolved in this release)**: `PUT /graph/params` accepted a write flipping a branch the node fixed at construction, values its constructor refuses, and a non-finite value (since 0.1.0; see `### Fixed`).
  **MADD-ANO-050 (new, open)**: two `HeatNode` rods coupled end to end by a converged exchange are unstable above Fo = 3/8, not the 1/2 each accepts; keep Fo < 3/8 on such pairs
- **MADD-ANO-046 (new, resolved in this release)**: a sub-cycled node whose timestep did not divide the macro timestep drifted by a fixed fraction of every step (since 0.1.0; see `### Changed`)
- **MADD-ANO-043, 044, 045 (new, resolved in this release)**: `run_adaptive*` advanced a sub-cycled node `divider * dt` per step; a multi-rate group's diagnostics, predictor and IQN-IMVJ warm start came from discarded solves;
  `solver="fori"` + `iqn-imvj` carried zero secant columns, so `jacobian_reuse` did nothing (all since 0.1.0; see `### Fixed`)
- **MADD-ANO-037 to 042 (new, resolved in this release)**: `ShardedUnstructuredNode` stepped a Cartesian stencil node wrong, dropped cells past its layout's count and gave no way to leave padding out of an integral (all since 0.3.0); both sharded wrappers placed a domain integral in the state like a grid field (since 0.2.1);
  a sharded static was edge-filled under a periodic wrapper (since 0.2.1); `halo_exchange` ignored an unknown `boundary` key and `ShardedStencilNode` accepted an empty `axis_map` (since 0.2.0).  See `### Fixed` and `### Changed`
- **MADD-ANO-034, 036 (new, resolved in this release)**: a walled `LBMNode` reloaded with no walls (since 0.1.0; see `### Fixed`); a config round trip dropped a node's sharding with no word (since 0.2.0; now a warning, see `### Changed`).
  **MADD-ANO-035 (new, open)**: on a balanced partition not in global order, `ShardedUnstructuredNode` reads a global-order state written with `set_node_state` as partition layout, so each cell steps from another cell's value (since 0.3.0).
  Convert the state with `partition_value` first, or renumber the cells with `np.argsort(partition_assignment, kind="stable")`; an explicit layout on the state write is planned for 0.5.0
- **MADD-ANO-032, 033 (new, resolved in this release)**: the sharded wrappers kept a compiled step across `compile()`, so a legacy node's params write after a step never reached the sharded physics; `ShardedStencilNode` stepped with a float32 `dt` under x64 (both since 0.2.0; see `### Fixed`).
  **MADD-ANO-024** now records the routes its first fix left open, all closed, and **MADD-ANO-004**'s workaround says it held only before the first step
- **MADD-ANO-028, 029, 030 (new, resolved in this release)**: periodic global halos on a size-1 mesh axis; a wide `"edge"` fill that copied the shard's first cells;
  a sharded `HeatNode` that ignored its end temperatures (all since 0.2.0; see `### Fixed`).  **MADD-ANO-031 (new, open)**: at `stencil_order=4` an end with no
  boundary input is not insulated (order 2 is); give it a temperature or use `stencil_order=2`
- **MADD-ANO-026 (new, resolved)**: with `waveform_iterations > 1`, `iterations` was the last sweep's count, hiding an earlier sweep at the cap (since 0.1.0).
  **MADD-ANO-027 (new, open)**: `waveform_iterations > 1` restarts the same solve rather than relaxing a waveform, and sub-step interpolation runs between
  iterates, so a converged step is the same in every mode (since 0.1.0). The option is now marked experimental; use `waveform_iterations=1`
- **MADD-ANO-024, 025 (new, resolved in this release)**: `PUT /graph/params` accepted and saved a value a node consumes at
  construction (since 0.1.0); a sharded `LBMNode` imposed its pressure faces at every seam and, by default, filled its global
  halos unlike its periodic streaming (since 0.2.0). Both are refusals now (see `### Fixed`)
- **MADD-ANO-021, 022, 023 (open)**: a gradient through a state-triggered branch omits the event time (BallNode's bounce:
  exactly 0 in the drop height); a mapped edge on a grid derived from a trainable parameter keeps its constructor
  geometry when that parameter is calibrated; the FMU TCP bridge authenticates no caller. Workarounds in the registry
- **MADD-ANO-020 is resolved in this release**: `LBMNode`'s pressure BC imposed the wrong face density since 0.1.0 (see `### Fixed`)
- **MADD-ANO-019 is resolved in this release**: a diverged coupling state can no longer read as converged (see `### Fixed`)
- **MADD-ANO-017** now also names `implicit_euler_step` (a float32 state under x64 is refused, in every
  release) and the `update`/`integrate_node` dtype divergence; `affected_versions` widens to `>=0.1.0`.
  It also records the legacy `solver="fori"` Aitken carry, which fails under x64 exactly as every other
  configuration does and adds no failure of its own
- **MADD-ANO-018 is resolved in this release**: `params` reaches every solver path (see `### Fixed`)
- **MADD-ANO-018**: a parameter calibrated through `gm.params` or `fit` reached
  `update()` and could not reach `derivatives()`, `implicit_residual()` or
  `integrate_node()` -- they took no `params`, so they ran constructor values
- **MADD-ANO-017**: `jax_enable_x64` does not reach `GraphManager`'s scan paths --
  float64 params against a float32 state seed, so `run_scan*`/`run_sweep` raise
  `TypeError` (open, minor); node-level `update()` unaffected, workarounds in the registry
- **MADD-ANO-016**: `cloud/_skypilot.py` was written against a SkyPilot older than
  the supported floor, so every `CloudSession` launch, teardown and preemption check
  was broken -- partially resolved in 0.4.0; end-to-end behaviour still unverified
- **MADD-ANO-015**: the ZeroMQ transports bound every interface unauthenticated
  and in cleartext from 0.1.0, and `launch_vm` published them whatever the job
  config said -- resolved in 0.4.0 (critical, safety_relevant)
- MADD-ANO-011/012/013 (BallNode, HeartPumpNode, the rigid bodies; all open):
  both ODE nodes name forward Euler and implement something else, and a float64
  quantity is pinned to float32 in several places -- HeartPumpNode's
  `backpressure`, and the `inertia`, `gravity` and `initial_state` casts in
  `RigidBodyNode`, `RigidBody2DNode` and `HeatNode` -- read the scheme from the
  algorithm guide, not from `discretization`
- **MADD-ANO-010**: a *string* that spells `NaN` / `Infinity` / `-Infinity` is now
  refused by `to_dict`, the USD JSON attributes and the FMI wire, because it would
  read back as that float -- spell such a value differently (minor, context_dependent)
- **MADD-ANO-001 (LBM GPU segfault) is resolved**: it needed jaxlib 0.5.1, which
  0.1.0-0.3.1 permitted and 0.4.0's floor does not; re-verified on GPU at jaxlib
  0.11.2 / CUDA 12.9, `LBMPipeNode` GPU vs CPU agreeing to 2.4e-07
- **MADD-ANO-007/008/009 (HeatNode) are resolved in this release**, all found
  by the MMS harness and all fixed before release: Dirichlet data was applied at
  the first cell centre, not the documented rod end (order 1, not 2);
  `stencil_order=4` converged at order 1 and was less accurate than the default;
  and the documented Fourier bound of 1/2 was the 3-point stencil's, with the
  4th-order stencil diverging inside it.  ANO-008's recorded diagnosis was
  corrected on re-derivation: the ghosts sit at -dx/2 and -3dx/2, and the oracle
  restores 3.76/3.90/3.95, not 3.78/5.02/4.79.  ANO-007 also said
  `compute_boundary_fluxes` was "not touched"; it is fixed too, the reported
  left flux moving from 10.6% error at n=10 to 0.13% and from order 1.005 to
  2.005
- Every anomaly whose defect is still reachable now records an open-ended
  `affected_versions`; ANO-005 no longer claims 0.4.0 is clean, and ANO-002's
  workaround names `thermal_diffusivity`, not the `alpha=` `HeatNode` never had
- MADD-ANO-005: before 0.4.0 `converged=True` was a residual test, not a bound
  on the distance to the fixed point.  0.4.0 applies the threshold to an error
  *estimate* instead; read `coupling_diagnostics()['ratio_usable']` to see
  whether the contraction ratio was usable, and treat `False` as the old
  behaviour.  The estimate is not a bound and can understate by 122x on a
  hidden slow mode (minor, partially_resolved in 0.4.0, context_dependent)
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
