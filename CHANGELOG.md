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
- **Geometry-dependent interface mappings (experimental): `add_edge(..., mapping=m, geometry=(anchor, field))` and the kind `multilinear_grid`** (`maddening.core.coupling.grid_mapping`): a mapping that reads a moving geometry -- a state field of the edge's own source or target node, such as marker positions -- at the time level the edge's value has, in the plain step, coupling groups, sub-cycled members and their gradients; `multilinear_grid_mapping` gathers from and scatters to a uniform grid of one to three axes. Carried by `to_dict` / `from_dict`, USD, checkpoints and FMUs; `register_mapping(..., needs_geometry=True)` registers your own kind.
  Limits in 0.4.0: a coupling group that resolves such a mapping reports its solve and no bound (`not_usable_reason` says why), `convergence_norm="interface"`, the adaptive steppers and sharded ends are refused for it, and the geometry must be held by the edge's source or target. Graphs without a geometry edge compile to the same programs as before. Guide: "Geometry-dependent mappings" (user guide and interface-mapping guide); claims MAP-026 to MAP-044.
- **Sparse interface mappings (experimental): `StaticSparseMapping` and three kinds, `sparse_nearest_neighbor`, `sparse_projection_1d` and `sparse_matrix`** (`maddening.core.coupling.sparse_mapping`): an integer index kept on the mapping and one weight per slot in `gm.params["mappings"]`, for interfaces whose dense matrix does not fit -- a million points a side build in seconds. The first two hold the dense kind's matrix bit for bit, nearest-neighbour ties included, and results agree with the dense mapping to the rounding of a row sum.
  The conservative nearest neighbour has two forms: a padded gather (the default; one result per compiled program, measured on CPU and GPU; refused past a byte cap with an error naming the other form) and `transpose="scatter"` (compact; measured not reproducible run to run on a GPU). A checkpoint records each sparse pattern's digest beside its weights, and `load_state` and `POST /checkpoint/load` refuse weights saved for another pattern.
  `scipy>=1.14` is now a declared base dependency (the k-d tree; jax already installed it, so no install changes). Interface mappings have their own claims inventory, `docs/validation/mapping_claims.yaml` (`MAP-NNN`; the registry's rows SYS-133 to SYS-143 moved there as MAP-001 to MAP-011). Guide: "Sparse mappings".
- **`register_mapping` (experimental): a mapping kind of your own, serialisable like the built-in ones.** The four interface-mapping kinds are now entries of one registry (`maddening.core.coupling.mapping_registry`); a factory decorated with `@register_mapping(kind, arrays=, hyperparameters=, references=)` survives `to_dict` / `from_dict` and USD, with its hyper-parameters and every reference limit checked before it runs and whatever it raises reported with its edge.
  A config, a stage or a checkpoint can only *name* a kind: nothing is imported because of what a file says, and a kind this process has not registered is a `MappingRebuildError` listing the ones it has. The built-in kinds write and rebuild exactly what they did.
  A mapping class other than `StaticLinearMapping` must return, from `params_pytree()`, a plain dict from identifiers to finite floating-point JAX arrays, the same on every call; `add_edge` now refuses anything else (MADD-ANO-196, never shipped: such a table was kept by reference and unchecked). Guide: "Registering your own mapping kind".
- **The coupling inventory's narrowed domain cells lifted** (`tests/core/coupling_domains.py`: one claim run over a memoryless pair in float64, mixed, bfloat16, float16, vmap, multi-rate, sub-cycled, predictor, restart, `run_adaptive` and sharded domains; the topology harness's invariances in eight): 188 cells in 24 rows, 350 still narrowed.
  It found MADD-ANO-170 to 173 (open, deferred to 0.5.0): the Aitken guard can cost two passes over its own exit (CPL-062 `failing`), `save_state` refuses a typed PRNG key, `reset_state()` inside a differentiated loss raises on a predictor group, and `run_adaptive` hangs on a NaN error norm.
- **A server-domain matrix in the REST and run_pod claims inventory** (`domain_set: server`): for loopback and non-loopback binds, the token demanded or not, simultaneous requests on a real loopback server, the runner, an in-flight `/sim/run`, oversized and hostile input, SIGTERM mid-request, wrapper nodes and a restored graph -- and run_pod's dry-run, relabelled and mixed-commit records -- each row names the test that exercises its claim there, `narrowed` or `n/a` (`testing_standards.md`, "The server set").
  Every row is `verified`: the four defects it reached (a `HybridNode` write lost, a `/sim/run` 503 saying nothing changed after steps, runner routes behind a saturated worker pool, a checkpoint load restoring values a write refuses) are the round-7 REST fixes' own, and its cells pass against them; sixteen rows' conditions are narrowed, and REST-098 (no TLS) is now pinned by a test -- read a row's conditions before relying on it outside the domains it names.
- **Terminal diagram of the graph** (experimental): `GraphManager.print_graph_diagram(theme=, direction=, use_ascii=, file=)` draws the structure `to_mermaid` exports as boxes and arrows (a frame per coupling group, dotted flux edges and external inputs), coloured by a termaid theme under `rich`.
  The `terminal` extra gains `termaid>=0.9,<1` (MIT, pure Python); `pip install "maddening[terminal]"` to use it. Read-only, like the rest of the inspection API.
- **A domain matrix in the coupling and sysid+FMU claims inventories** (`domain_set: numeric`, a `domains:` mapping per row): for float32, float64, mixed dtypes, 16-bit, jit, gradients, vmap, multi-rate, sub-cycling, predictors and warm starts, `run_adaptive`, checkpoint restarts and sharding, each row names the test that exercises its claim there, `narrowed` (its conditions now exclude the domain) or `n/a`; `test_claims_inventories.py` checks it.
  It found four failing coupling domains (MADD-ANO-160 to 162, and a bfloat16 `rho_spectral` that is not float32-exact) and two minor sysid ones under x64 (SYS-024, SYS-071): eleven rows are `failing`, and many conditions are narrowed -- read a row's conditions before relying on it outside float32 (`testing_standards.md`, "The domain matrix").
- **Coupling interaction and topology harness** (`tests/property/test_differential_coupling_{interactions,topologies}.py`): a strength-3 covering array of the group's knobs, and drawn graph shapes -- several groups, outside nodes on a group's loop, flux, mapped and additive edges -- held to a monolithic float64 reference of the step.
  It found MADD-ANO-156 to 159 (open, since 0.1.0): a flux edge across a group's boundary or staggered in an ungrouped cycle fails to trace, a typed PRNG key fails under `ift` with Aitken or fixed relaxation, and a group joined off any cycle reads a node a step late, silently; workarounds in the release notes.
- **Domain oracles** (`tests/property/test_differential_{precision,replay,param_acceptance,fmu}.py`, `test_metamorphic_fmu_stability_filter.py`): float32 against float64 (a worker under x64),
  replay from recorded state, every door that writes or restores a parameter (checkpoint loads, `POST /graph/nodes`, wrapper nodes), the FMU four ways across `node.params` writes, and the
  stability filter; known disagreements are strict xfails naming their finding (`testing_standards.md`, "Domains the claims cover").
- **REST and run_pod claims inventory** (`docs/validation/rest_runpod_claims.yaml`, prefixes `REST`, `RPD`): every documented claim about the HTTP API and the multi-GPU session runner with its conditions, oracle and a test that can fail -- 143 rows, 132 verified, 5 failing (strict xfails), 5 ambiguous, 1 untested; the surrogate and streaming endpoints are out of scope (experimental).
  Read the failing and ambiguous rows before relying on those claims; a new REST or `run_pod.py` claim gets a row in the same change (`testing_standards.md`).
- **System identification and FMU claims inventory** (`docs/validation/sysid_fmu_claims.yaml`, `SYS-NNN` / `FMU-NNN`): every documented `sysid`, `ParamSpec` and FMU-export claim with its conditions, oracle and a test that can fail -- 147 rows, 139 verified, 4 failing (strict xfails), 4 ambiguous.
  `tests/compliance/test_coupling_claims.py` becomes `test_claims_inventories.py`, which checks every `docs/validation/*_claims.yaml` (each declares the id `prefixes` it owns).
- **Coupling claims inventory** (`docs/validation/coupling_claims.yaml`): every documented coupling claim with its conditions, oracle and a test that can fail -- 130 rows, 114 verified, 6 failing (strict xfails), 10 ambiguous -- checked by `tests/compliance/test_coupling_claims.py`.
  Read the failing and ambiguous rows before relying on those claims; a new coupling claim gets a row in the same change (`testing_standards.md`).
- **Eight new or extended examples**, each asserting what it prints: `advanced.profiling_demo` (the full `ProfileReport`), `sysid_demo`,
  `checkpoint_resume_demo`, `sharding_demo` (4 emulated CPU devices), `fmu_export_demo`, `coupling.interface_mapping_demo`,
  `servers.rest_params_demo` (in-process), and `print_coupling_report()` / `strict_convergence` in `convergence_diagnostics_demo`
- **One-command local runs for the remote-simulation examples**: `python -m maddening.examples.servers.remote_viz_client --local` starts the
  simulation server on a free loopback port, streams to the viewer and stops the server on exit (Ctrl-C and errors included); the HTTP example
  servers take `--port 0`, and `cloud/server/04`, `05` and `cloud/multijob/08` gain a `--local` mode that needs no cloud account.
- **Read-only graph inspection** (experimental): `gm.print_graph()` / `format_graph()`, `to_mermaid()` / `to_dot()`, and tables
  `state_summary()`, `params_table()`, `coupling_report()` (caveats flagged), `memory_estimate()` with `print_*` forms; none writes or
  compiles anything.  `maddening.show_versions()` / `python -m maddening info` for bug reports; `print()` of `FitResult` / `FIMReport`
- **The slow-only rule is checked** (`tests/compliance/test_slow_only_rule.py`): a slow-marked framework test names a cheaper
  test the default lane runs (`# Per push: <node id>`) or is in the slow-only table; six framework properties that had neither
  now have one, and each slow hypothesis property a sibling on both JAX lanes.  A slow mark you add needs one (testing_standards)
- **Differential tests for coupling and numerics** (`tests/property/test_differential_*.py`): two paths that must agree, over
  generated graphs of synthetic nodes -- fori/ift, diagnostics on/off, the exact fixed point, multi-rate, sub-cycling, adaptive,
  `vmap`/`jit`, non-float leaves; each disagreement found is a strict xfail (`testing_standards.md`, "Differential tests")
- **Every Python code block in the docs runs in CI** (`tests/compliance/test_docs_snippets.py`), sandboxed, or carries a
  `<!-- snippet: no-run, reason: ... -->` marker whose imports and keywords are still checked; see the documentation
  standards.  The README quick start, which overrode the removed `requires_halo`, now runs
- **CI guard mutation test** (`tests/compliance/test_guard_mutations.py`, slow lane): seeded faults in the workflows, budget
  script, shard split, conftest, `pyproject.toml`, timing plugin, cache pruner and allowlist must each fail a guard test
  (83 of 87; 4 listed equivalents); anchors checked per push.  New guard: add its mutant (*Guard mutations*, testing_standards)
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
- **What an `"interface"` group returns for a floating field no internal edge reads** (released behaviour, corrected: MADD-ANO-235): the field as one pass computes it at the returned state, where it used to be the iterate's. `iterations`, `residual` and every field an edge reads are unchanged; a group whose every field an internal edge reads is unchanged to the bit. The recomputed field is the next step's starting value, so a run stepped at an unconverged cap follows another trajectory.
- **A step that changes the layout of the state raises `ValueError`; it used to be stored** (released behaviour, corrected: `MADD-ANO-220`): `GraphManager.step`, `run`, `run_adaptive` and `FmuSidecar.step` refuse a stepped state whose leaf has another shape or kind of dtype than the state it replaces, or whose fields differ, naming the node and the leaf; nothing is stored. A node's `update` must return the layout `initial_state()` built: build a state that grew at its first step at its final shape.
- **`add_coupling_group` refuses a boolean option that is not a boolean, and `rtol=0` under the norms that divide by it**: `diagnostics`, `subcycling` and `strict_convergence` raise `TypeError` for anything but `True`/`False` (`diagnostics="off"` turned the diagnostics on); `rtol=0` under `convergence_norm="mixed"` or `"interface"` raises `ValueError` (the residual was `0/0`, the step ran to its cap and the report raised `ZeroDivisionError`). Action: pass a bool; use a positive `rtol`.
- **A multi-rate graph whose schedule would not keep a node's clock no longer compiles** (MADD-ANO-227, in every release to 0.3.1): timesteps more than about 1e9 apart (`1.0` and `1e-10` got rate dividers of 1 and 0 and ran on different clocks, silently), or with no common step the float GCD finds, are a `ValueError` at `compile()` naming the two nodes, an error from `validate()`, and a 400 from `POST /graph/nodes` and `DELETE /graph/nodes/{name}`. The dividers of every graph that is kept are unchanged. Give the nodes timesteps that are whole multiples of one step.
- **REST numbers are strict, and counts are held to their declared range** (MADD-ANO-228): `"timestep": true` (a node at 1.0 s), `"0.5"` and `" 0.25 "` are a 422, as are a boolean or text for any number of a request model; an integer query parameter is decimal digits only; `POST /sim/profile` answers 422 for `n_steps` outside 1..1000 or `n_warmup` outside 0..50 where it clamped them without saying so. Send JSON numbers, and counts inside the documented range.
- **`maddening.__version__` is the tree's own version, not the installed distribution's** (a source tree on `PYTHONPATH` beside another installed version called itself that version, in `GET /healthz` and every record), and **an `Authorization` header is read by RFC 9110's grammar**: the scheme in any case, one or more spaces, the token exactly; a tab or other whitespace beside the token no longer matches.
- **An external input runs in its declared dtype at every entry point** (`add_external_input(..., dtype=)`, default `float32` also under `jax_enable_x64`): `step`, `run`, the scans, sweeps, adaptive runs and the raw compiled step now cast a supplied value to it, as the FMU export, the zeros of an omitted input and the saved configuration already took it. An x64 graph with a default-declared input used to run `0.1` as a float64 where its FMU ran `float32(0.1)` (MADD-ANO-218); it now runs `float32(0.1)` and says so once per input with a `UserWarning` whenever the cast changes a supplied value. Declare `dtype=jnp.float64` to keep float64 inputs; nothing changes without x64, or for values the declared dtype holds exactly.
- **`build_model_description(default_step_size=)` takes a finite, positive Python or NumPy number that is a whole number of graph steps to within four ulps, and refuses anything else by name** (a bridge given a hand-built description likewise). A NumPy scalar used to be written as `stepSize="np.float64(0.05)"`, which importers reject (MADD-ANO-219, shipped in 0.3.x); a step such as `float(np.float32(0.05))` on a 0.01 s graph was accepted and refused at the fourteenth `fmi3DoStep`. Pass `n * gm.timestep`.
- **`ParamSpec`: `constrain` lands inside the bounds `check` enforces where a bound is a subnormal number** (it returned `-0.0` for `(None, -1.4e-45)`), a `logit` spec with a subnormal bound or an interval under four smallest normals of the leaf's dtype is refused by name, and `fit` / `fit_lm`'s curvature test answers "no curvature test" with its warning, not `LinAlgError`, when a Jacobian column or Hessian product overflows over its scale.
- **`fit_lm` ends `converged=False`, with a `RuntimeWarning`, where its run stops beside a jump of the residual; and it warns when `excited_rank` reads the full count on a fit whose `JᵀJ` does not**: on a residual that is not differentiable in the trained parameters (a node with a contact, an event or a valve: `BallNode`, `HeartPumpNode`) the floor rule reported `converged=True` at the edge of a smooth piece, at losses of 0.02 to 3.7 against 1e-13; it now tells a jump from the rounding floor by comparing each small rejected step with its mirror image. Smooth fits are unchanged. The condition is stated in the `fit_lm`, `fit`, `fim` and `FitResult.converged` docstrings and in the parameter guide; `excited_rank` is defined as the count minus the directions the guard found undetermined.
  Action: on a model with events, read `converged` and the warning, and compare `best_loss` across starts; declare `transform="log"` on an identity parameter that shares a scale (`HeartPumpNode.stroke_volume`).
- **CI: the slow lane runs six shards per JAX version, split by measured time** (`MADDENING_TEST_SHARD=i/6:weighted`, `tests/slow_lane_weights.json`, `scripts/slow_lane_weights.py`): by hash, one shard of four held nearly half
  of the lane and reached its 175-minute timeout. Every test still runs once; the per-push lane's split is unchanged.
- **`GraphManager.remove_node` (and `DELETE /graph/nodes/{name}`) takes the node out of its coupling group** (`MADD-ANO-214`): the group keeps its options over the remaining members and is removed when one would remain; it used to go on naming the node, and every compile and step failed until a node of that name was added. A `UserWarning` (the route's `coupling_groups` reply) names each group that changes: after re-adding a member under its name, call `add_coupling_group` again. A node whose points a mapping on an edge between two other nodes was built from is refused (`ValueError`, 400): remove that edge first. `replace_node` keeps the membership.
- **`maddening.core.graph_manager` holds `GraphManager`; its private module-level machinery moved to private modules**: the coupled block, the fixed-point loop, the IFT solve, the spectral and gradient bounds, the convergence reports, the adaptive scan, the parameter probes and the bookkeeping structs are now in `maddening.core.coupling._coupled_block`, `._fixed_point`, `._ift`, `._bounds`, `._reports`, `._group_layout` and `maddening.core._adaptive_scan`, `._param_probes`, `._graph_specs`, moved verbatim (the lowered step programs are unchanged on every tested jax). Every public name (`GraphManager`, `ExternalInputSpec`, `ShardingIssue`, the `EVENT_*` names, `GRADIENT_PROBE_ENTRY_LIMIT`) imports from `maddening.core.graph_manager` as before.
  Action: none for the public API. Code that imported or patched an underscore-prefixed name through `maddening.core.graph_manager` now gets an `ImportError` or `AttributeError`; those names were never part of the API.
- **The REST guide says who the server is for, and its transaction is now the routes' only undo**: `src/maddening/api/README.md` states the audience (a trusted client on loopback, or a token-holder), what the server defends against and what it does not (a token-holder's crafted archives, resource exhaustion, pathological names; TLS), and the promise that a refused or failed request leaves the graph as it was, with its carve-outs; the REST claims rows are narrowed to it.
  The experimental `POST /surrogate/deactivate`'s 500 is the generic one (it quoted the graph's error), and `POST /sim/profile` puts the whole graph back, so an edited graph is still waiting for its compile after a profile.
  Action: none; read a failed surrogate revert's reason in the server's log.
- **A REST write that fails, at any point, leaves the graph exactly as it was**: every route that can change the graph (nodes, edges, compile, `PUT /graph/state`, `PUT /graph/params`, checkpoint save and load, `/sim/step`, each slice of `/sim/run`, `/sim/start`, `/sim/reset`, `/sim/profile`, surrogate activate and deactivate) runs in one transaction that records the graph under the lock and puts it back when the route refuses or fails unexpectedly; each route used to validate first or undo for itself, and a failure after its first write left what was written.
  An unexpected failure is now a JSON 500 with a generic detail (the traceback is in the server's log), where it was a plain-text 500; a refused `POST /sim/step` or `/checkpoint/load` of an edited graph no longer leaves it compiled; a `POST /sim/run` that fails unexpectedly after some slices answers 500 with `steps_run`.
  Action: none; not covered, and unchanged: files under the checkpoint root, the runner's thread, surrogate jobs, frames a stream already sent, `/cloud`.
- **Names and text a graph could not save again are refused where they are introduced, and every REST reply is written by one encoder that cannot fail** (MADD-ANO-208, 209, 210): `GraphManager.add_node` (so `from_dict`, a USD stage and `POST /graph/nodes`) refuses a name holding U+0000 to U+001F (tab, line feed and carriage return excepted), a surrogate or U+FFFE/U+FFFF (a NUL-named node's checkpoint saved and did not load; an FMU's model description could not be read back); `add_edge` and `add_external_input` refuse a field named `NaN`, `Infinity` or `-Infinity` (`GET /graph` was a 500 while the edge stood), or holding `#` (two mapped edges on one field pair shared one slot of weights) or those characters; `POST /graph/nodes` and `PUT /graph/params` refuse a text value spelling a token or holding a surrogate, and `/checkpoint/save` and `/load` a name with those characters.
  A refusal that echoed the text `NaN`, `Infinity` or `-Infinity` (`POST /sim/run?n_steps=NaN`, what a browser sends for an empty count) or a surrogate was a 500 (never released for the text; the surrogate since 0.1.0): every route, 422 and `HTTPException` now replies through one encoder.
  Action: rename a node, field or checkpoint whose name holds such a character; an edge to a target field its node does not declare is still taken.
- **The FMU export refuses what it used to round, skip or forget** (MADD-ANO-197, 198, 200, 201): `build_model_description(selected_outputs=...)` raises `ValueError` for a pair naming no exportable state field (it returned a description without that output); a `get` of an `Int64` / `UInt64` variable holding a value a float64 cannot hold is an error (`fmi3Error`), not a neighbouring integer; an FMU-state archive carries the drift reference, and one without it is refused by name; `load_state` raises for an integer of the other signedness.
  Action: fix the misspelt pair; take FMU states again after upgrading a development build.  The time bound is restated with the float64 clock's own rounding (three ulps of the time beyond a millionth plus a tenth of a step).
- **`to_dict()` returns a copy, and a hook naming `params` where no keyword reaches it is refused when its node is added** (MADD-ANO-205, since 0.1.0): a config shared its containers with the graph, so `cfg = gm.to_dict(); cfg["nodes"][0]["params"]["k"] = 5.0` also changed `gm` (from its next recompile; on 0.4.0 builds a three-argument node's next step) and `yaml.safe_dump` refused a three-argument node's config; and a hook naming `params` positional-only or as `*params` is refused by `add_node` and by the wrappers (`ValueError` naming the node, the hook and the fix), where its first trace raised `TypeError`: a clearer and earlier error for a node that never ran. Action: edit configs freely; write through `node.params` to change a node; declare `*, params=None`.
- **Assigning `node.params` stores a counting mapping; `SpringDamperNode` seeds its state in its constants' dtype** (MADD-ANO-175, MADD-ANO-017): writes made through `node.params` after `node.params = {...}` now reach `gm.params` (they were lost), so a reference to the assigned dict is no longer `node.params`; under x64 a spring with Python-float constants seeds float64, so its graph scans (it raised; 0.3.x ran it in float32).
  Action: write through `node.params` after assigning it; give a spring float32 constants to keep a float32 state under x64.
- **Interface-mapping factories refuse coordinates they cannot map, and every reader of an edge applies the step's rule** (MADD-ANO-192, 193, since 0.1.0): `projection_1d_mapping` / `conservative_projection_1d` returned a zero or wrong matrix for boundaries that are not strictly increasing (descending: all zeros; `[0, 2, 1, 3]`: rows summing to 4/3), nearest neighbour selected a NaN point for every target, `linear_interpolation_1d` took unsorted coordinates, and the RBF factories and `matrix_mapping` took non-finite input.  Each is now a `ValueError` naming the argument and the first offending index (`MappingRebuildError` from a config or a USD stage); nothing is sorted for you, and accepted input gives the same operator bit for bit.
  `check_conservation` and `DatasetGenerator` rebuilt boundary inputs without the interface mapping, let an additive edge overwrite the edges before it, and (dataset) applied a transform across the time axis; both, and `resolve_boundary_inputs`, now resolve through the step's own edge rule.  The diagnostic also resolves an edge that reads a flux output, reads `gm.params`, and raises for a flux the node does not report (it read 0.0), a node missing from the state, or two nodes that read each other's flux; `POST /surrogate/train` sizes a mapped input as delivered.
  Action: pass boundaries (with the field) in increasing order and finite points; re-run a conservation check, or regenerate a surrogate dataset, that involved additive edges, mapped edges or a transform that indexes its field.  Still open: the dataset pairs a forward edge's input with the step before (MADD-ANO-194, since 0.1.0), and `convergence_norm="interface"` reads a mapped edge before its mapping (MADD-ANO-195, never released).
- **Two released behaviours corrected** (MADD-ANO-160, 159, since 0.1.0): `jax.grad` through `run_adaptive_scan` read NaN or exactly 0.0 wherever an attempt's step-doubling estimates agreed exactly (a coupled group at its fixed point, a memoryless or resting node) and now reads the derivative; `compile()` warns, naming the edge, when a coupling group whose members are joined only through outside nodes reads one of them a step late.
  Action: re-run a calibration or sensitivity study that differentiated `run_adaptive_scan`; a group the new warning names should take the joining nodes in, or be split.
- **`GraphManager.add_node` refuses a timestep that is not a finite number > 0 and the names `_meta`, `_params`, `_params_mappings`; a loopback-bound API asks the token of a non-loopback peer and of a forwarded request** (MADD-ANO-179, 181, 182): each was accepted, then ran silently wrong or broke later.
  `run_pod.py` exits 2 when it refuses a run and 5 when a goal raises (both were 1, "a check failed"); `--summarise`'s statuses are unchanged.
  Action: give every node a finite timestep > 0; in-process REST clients present `server.auth.token`; read a goal's exit status by the runbook's table.
- **A bfloat16 or float16 state is measured in float32 by the coupling norms and the adaptive error norm; coupling round-6 fixes** (MADD-ANO-186 to 189): a 16-bit group's norms, floor and residual are measured, summed and held in float32 -- above 65,504 active entries a float16 group's "mixed"/"interface" residual read 0.0 (every release; on this cycle's builds it then stopped 4% from its fixed point, `converged=True`), and at the default `rtol` `inf` on a finite state -- and `strict_convergence` names a non-finite state from the state, not the estimate; the adaptive steppers' step-doubling error norm had the same defect (every release: `run_adaptive` did not return on a float16 state above 65,504 elements, and `run_adaptive_scan` raised a scan-carry `TypeError` on any 16-bit state).
  Under `convergence_norm="interface"` with an edge transform `spectral_error_bound` is taken on the transformed reading (it read 0.0014-0.098x the true distance, usable); the gradient bound's Kantorovich check adds the affine-covariant constant as an operator and carries the Newton step's miss (0.65-0.986x, usable, far from the fixed point); `rho_spectral` and the L2 `tolerance` docstrings are restated.
  Action: none; a 16-bit group's `residual` and its `_meta` slots are now float32 and it may take a different (correct) number of passes, a 16-bit state steps adaptively as its float32 twin does, and the gradient bound costs `11 + 4k + 5 n_p` JVPs and can withdraw itself where it was usable at `h` near 1/2.
- **REST refusals agree with `PUT /graph/params`, and the runner routes stay off the shared workers**: `POST /checkpoint/load` and `POST /graph/nodes` refuse a parameter value `PUT` refuses (outside its `ParamSpec` bounds, a boolean or text for a number; on a load also non-finite or refused by the constructor), `GraphManager.load_state` refuses text and booleans for a number, and `PUT` refuses a numeric string (it stored `"1.5"` as 1.5); an integer is bounded at 10^7 only for an integer parameter (an integral JSON number for a float one is a float); a token with surrounding whitespace and an `allowed_hosts` entry that is not a host name raise at construction.
  Start, stop, pause, resume and reset run on pools of their own with deadlines from arrival (`PUT /sim/stride` on the event loop); `run_pod.py` takes no option abbreviations and writes schema 7 (`requested_cells`; a synthetic grid holds at least the requested cells).
  Action: keep parameters inside their bounds; strip the token; spell `run_pod.py` options out; re-run a session recorded at schema 6.
- **A Krylov solve answers a right-hand side with a NaN or infinite entry with NaN, not zeros** (MADD-ANO-154, 155): the IFT tangent and adjoint of a coupling group read exactly 0.0, reported successful, for a NaN or `+-inf` tangent or cotangent (never released; dense and fori read NaN),
  and `sharded_cg` (since 0.3.0, mostly with `converged=True`) and `ift_linear_solve`'s CG (since 0.3.1) returned zeros, as did every GMRES path on this cycle's relative tolerance.  Every Krylov path (`_ift_linear_solve`, `ift_linear_solve`, both `sharded_*` backends, tangents and cotangents included) now answers NaN in every entry; a sharded result reads `converged=False` and a NaN `residual_norm`.
  Action: none for a finite right-hand side (its path is unchanged); a solve or derivative that read 0.0 from a non-finite input now reads NaN -- check the input with `jnp.isfinite`.
- **The surrogate-training routes and the state streams are experimental in 0.4.0** (`/surrogate/train`, `/surrogate/status`, `/surrogate/activate`, `/surrogate/deactivate`; `/ws/state`, `/ws/state/binary`, `/ws/render`; `StateRelay`): they are to be hardened in 0.5.0 and may change in any minor release until then.
  They carried no level before; the routes are listed in the stability report (`register_route_stability`) and tagged `x-maddening-stability` in `/openapi.json`. Action: pin the version if you build on them.
- **No absolute constant hides in a relative computation** (MADD-ANO-123 to 130; gate `scripts/check_numeric_constants.py`): `sharded_cg`/`sharded_gmres`/`ift_linear_solve` default `atol=None` (relative; a right-hand side near 1e-9 came back 40-100% wrong with `converged=True`), and the multi-rate GCD, Adam's `eps` and the adaptive error norm no longer depend on units.
  Coupled fields below `finfo.tiny/eps` warn once (`UnderflowRangeWarning`); the power-of-two frames are one helper (bit-identical, 125 configs x 3 jaxlibs).  `run_adaptive`'s `atol`/`dt_min` stay absolute (MADD-ANO-128).
  Action: pass `atol` only as a noise floor in `b`'s units; re-run small-unit solves, `fit` runs and nanosecond multi-rate graphs; give adaptive steppers `atol` in your state's units.
- **`compile()` warns about two `SpringDamperNode`s anchored on each other in a coupling group whose converged step grows** (MADD-ANO-098, still open; a `UserWarning` naming both nodes and the growth factor `g`, never a refusal, judged from the live `gm.params`). Action: use a smaller timestep, or keep `k*dt <= c <= (2*m - k*dt**2)/dt`.
- **`LBMPipeNode` refuses a `propeller_radius` above 1** (MADD-ANO-058): the disc reached past the pipe wall and pushed on wall cells (64 of 140 disc cells at 1.5 on a 12x12 cross-section; mean `u_x` 1.9% off), with no error.
  Action: pass a radius in `(0, 1]` (1 is the whole cross-section); a config saved with a larger one no longer reloads.
- **`run_scan` and the other loop entry points refuse a `ShardedStencilNode` step XLA miscompiles inside a loop** (MADD-ANO-068, now `partially_resolved`): a node reading a sharded static in its halo beside a `shard_info`-offset window into another array, the static copied along a mesh axis of 2+ devices,
  raises `RuntimeError` from `run_scan`, `run_scan_with_history`, `run_sweep`, `run_adaptive_scan` and `sysid.windowed_loss` (gradients included), and from every entry point in a coupling group; `step()` / `run()` keep working otherwise.
  Action: use a mesh without the axis the static is copied along, or `step()` / `run()`; check a loop you write yourself around such a node against `step()`.
- **A domain integral is neither summed nor stacked over a mesh axis a `ShardedStencilNode`'s `axis_map` leaves unused** (MADD-ANO-067):
  releases summed it there, counting every block once per device along it (2x on a `(2, 2)` mesh), and a per-shard integral there now has
  one leading axis per mesh axis that splits the grid.  Action: re-run sharded totals from such a mesh; drop the extra axis when reading per-shard values
- **`maddening.sysid.fit` and `fit_multiple_shooting` return the lowest-loss iterate they evaluated, not the last**: Adam's ~`lr`-sized step could end a run above where it started (one fit went from loss `2.7e-8` to `1.9e-4`, unreported). `FitResult.best_iteration` and `best_loss` say which iterate was returned; a run whose loss never rose is bit-identical to before, and `fit_lm` already returned its lowest iterate.
  Action: nothing for a fit that converged; where you relied on the last iterate, read `best_iteration` (it equals `len(losses)` when the last update's result was the lowest).
- **`HeatNode` refuses more of what it used to run wrongly, and `compile()` warns about an unstable coupled pair**: a rod on `grid_points` is held to `dt*alpha/min(h_L*h_R) <= 1/2` (the Fourier check skipped it; MADD-ANO-002), and a non-positive `length` or `timestep`, a negative `thermal_diffusivity`, a non-finite one of these, or `grid_points` not strictly increasing raise `ValueError` (MADD-ANO-062).
  Two uniform rods coupled end to end through `extract_first`/`extract_last` in a coupling group, past their pair limit (3/8 at `stencil_order=2`, 0.226 at 4), get a `UserWarning` naming both rods and MADD-ANO-050; it is a warning, not a refusal.
  Action: give a refused rod a stable timestep or meaningful constants; for a warned pair use a smaller timestep or exchange the data without a coupling group, and check a pair the warning cannot recognise yourself.
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
- **A coupling group under `convergence_norm="interface"` no longer returns a field no internal edge reads from a pass before the readings it converged on** (MADD-ANO-235; 0.1.0 to 0.3.1): a one-way pair under `iteration_mode="jacobi"` returned its target computed from the pre-step source with `converged=True`, `iterations=1`, residual 0; a chain its last member a pass behind; a weakly coupled pair such a field up to 384 tolerances off (25 under Gauss-Seidel); the sweep's ten `iqn-*` / interface rows `velocity` up to 2.25 off. Use `"mixed"` on an older release. Still open: a field read only through a mapping or a transform (MADD-ANO-236).
- **`fit_lm` no longer reports `converged=True` beside a small jump of the residual read from a long rejected step** (MADD-ANO-216's residual case, never released): the one-sided test divided the jump by the linearisation's change over the whole step, so a late bounce 22 float spacings from the iterate read 341 and 456 against a threshold of `2**10` on the stock ball. A rejected step reading above `2**5` is now read again across the one float spacing where the residual departs most (a few residual evaluations, at the end of the run only); the run ends `converged=False` with the existing warning. Action: none.
- **A coupling group under `acceleration="iqn-ils"` or `"iqn-imvj"` returns from `step()` when its iterate leaves float range** (MADD-ANO-233; 0.3.0 and 0.3.1 under `solver="ift"` with `"iqn-imvj"`): the secant least-squares handed LAPACK's SVD a matrix holding an `inf`, and the step never returned. It now runs to `max_iterations` and reports `converged=False`, `residual=inf`, as the other accelerations do; steps of a finite run are unchanged, bit for bit. Action: none; on 0.3.x use `solver="fori"` or `"aitken"`.
- **`gradient_relative_error_bound` covers the constants one pass resolves** (MADD-ANO-234, never released): with a constant in it whose tangent at the returned iterate is rounding (the centre or curve of a nonlinearity evaluated on its centre) it read 1.09 for a relative error of 1.134, usable. Such a constant, and a gain whose whole value moves the pass by less than the residual's float floor, is no longer in the bound; the docstring says which constants are. Action: none.
- **`GraphManager.step`, `run` and `run_adaptive` no longer store a step that reshapes a state leaf** (`MADD-ANO-220`, in 0.1.0 to 0.3.1): a list given for a scalar constant (`BallNode(initial_velocity=[1.0, 2.0])`), or an external input of another shape at a later step, broadcast a scalar leaf, and the state's checkpoint then did not load after `reset_state()`. The step is refused by name (compared once per trace, host-side: no compiled program changes), as `run_scan` always did; the FMU sidecar and bridge refuse it too.
- **`coupling_diagnostics()` describes the step that ran when the state is written afterwards** (MADD-ANO-231, never released): the float floor under `spectral_error_bound`, `precision_limited` and `spectral_usable` was measured on the live state, so `set_node_state` after a step moved that step's report (a bound of 0.0, flag set, with every member written to zero). The graph now keeps what the step left. A checkpoint saved after a member was written says so (an optional archive member), and the graph that loads it reports that group's bound as NaN and its `*_usable` flags False with a `not_usable_reason` until it steps; `_reports` is now a reserved node name. Action: none.
- **`coupling_diagnostics()`: three numbers that read wrong with their flag set, and one found beside them** (MADD-ANO-222, 225, 226, 229, never released): `gradient_relative_error_bound` at the float floor took the change along one stand-in direction (5.1e-6 for 3.0e-5); `rho_spectral` read settled past eight interface scalars (0.273 for 0.219), from sampled rounding on a float32 group far from normal (0.250 for 0.206), and lost a loop below a field's rounding on a sub-cycled group (1e-12 for 1.25e-4).
  The gradient bound's undirected distance now takes an operator norm; a Krylov space still growing at the cap is never settled (`spectral_usable=False` above seven interface scalars, eight where they are the whole state); a certificate over every perturbation of the measured size replaces the samples; a sub-cycled boundary value is differentiated as `(1 - alpha) a + alpha b` (values unchanged, gradients of sub-cycled groups move at rounding level).
  Action: none; expect `spectral_usable=False` on more float32 and 16-bit groups whose Jacobian is far from normal (usable fraction on the search's draws 0.979 to 0.976).
- **`POST /checkpoint/load` takes a checkpoint saved before an initial condition was written** (never released): after `PUT /graph/params {initial_*: v}` (or `TableNode.position`) every earlier checkpoint was a 400 "does not fit this graph", for six of the sixteen numeric parameters of the stock nodes. The load now installs each leaf it changes as `PUT` does (in the node too, so the next reset builds the state from it), answers as `PUT` of the same value on every parameter of every built-in node, and its 400 names the checkpoint's value and the route that fixes it.
- **`run_pod.py --summarise` reads a goal file with no cells or no checks INVALID (exit 3) for every goal** (never released): an emptied `forward.json`, `gradient.json` or `exchange.json` read "no checks" and the summary exited 0.
- **A constructor's refusal at `POST /graph/nodes` names the parameter** it can be told from, the value sent and the class's default (`n_cells: 8.0` was "'float' object cannot be interpreted as an integer").
- **An FMU parameter whose `ParamSpec` accepts no value advertises an empty `min` / `max`** (never released; the spec-level refusals are 0.4.0's own): a `logit` spec with a subnormal float32 bound, or a `log` / `logit` bound or width the leaf's dtype does not hold, is refused by `check_params`, the sidecar and REST for every value, while `build_model_description` advertised the neighbours of its bounds (`min = 1.18e-38`, `max` just under `1.42e-14` for `logit` on `(1.4e-45, 1.42e-14)`), so a bridge over a sidecar built without `param_specs` took every value between them. The variable now advertises `min = tiny > max = -tiny`, and the bridge refuses to serve the description, with or without specs. Action: none; narrow the bounds as the refusal says.
- **`POST /graph/nodes` accepts only a node the graph can step with its state as built** (the REST door of `MADD-ANO-220`; in process the step itself refuses such a node, see the entry above): one update is traced as the graph calls it (params pytree, with and without boundary inputs) and held to `initial_state()`'s fields, shapes and kinds of dtype; `venous_pressure: null` and a list for a scalar constant are a 400 naming the parameter. Also: lists and objects count against the 10^6 params bound; `POST /checkpoint/load` names why a graph cannot compile; `POST /surrogate/deactivate` (experimental) lists the edges it drops (`dropped_edges`).
- **`run_pod.py` refuses an option value no goal can use with exit 2, before the backend loads** (never released): `--cells 0`, `--warmup -1`, `--repeats 0`, `--steps 0`, a `--mesh` that cannot be read. They used to pass, or exit 5 as a crashed goal.
- **`rho_spectral` and `spectral_usable` under `diagnostics=True`** (MADD-ANO-221, 223, 224, never released): the spectral estimate took a Krylov direction below 1e-5 of a product for rounding in every dtype, took the compressed radius by repeated squaring, and called a spectrum settled on the Arnoldi residual alone; `rho_spectral` read 0.379 for 0.5 (float64), 0.0512 for 0.0500 (float32) and 0.941 for 0.735 (twelve non-normal scalars), and `spectral_error_bound` 0.61x the distance, each with `spectral_usable=True`.
  The breakdown test is now eight units of the products' own rounding, the radius is `eigvals`, and one more Jacobian-vector product (nine per group per step, was eight) checks the estimate against itself: where it moves the radius by more than 5% of `1 - rho_spectral` the flag is `False`.
  Action: none. The three the targeted search found beside these (MADD-ANO-222, 225 and 226: a loop below a float32 field's rounding, a radius read settled past eight interface scalars, the gradient bound at the float floor) are resolved by the `coupling_diagnostics()` entry above.
- **A params write is held to the points a mapped edge was built from whatever a fit or a checkpoint load did first** (`MADD-ANO-215`): `PUT /graph/params`, `POST /checkpoint/load` and a `gm.params` write refuse a `length` under a mapping when another leaf's live value differs from the one the node was built with, or is written in the same request; a structural value the live leaves allow is no longer refused for the built ones.
- **`convergence_norm="interface"` reads every internal edge as the step delivers it** (MADD-ANO-195, 211 and 213, never released): the norm, its float floor, the spectral bound's reading and the report applied an edge's transform and left its interface mapping out, so a mapped edge was judged on its source field -- a 9% change of the delivered value read 0.0071 of the tolerance, and `spectral_error_bound` 0.064-0.32x the distance in what the edge delivers, usable.
  One function now gives all four the step's value (the mapping with the step's weights, then the transform); the step records such a group's floor (`reading_floor`), and a delivered value wider than its source field is resolved at the source's `eps` (a stalled float32 pair behind edges that cast to float64 read 6.7e-7x, usable).  A field that several internal edges read (a star's hub) was counted once per edge by the norm and once by the bound's analysis, which read 0.26-0.73x the true distance, usable; such a group's report is now analysed on the same reading.
  Action: none; a group without a mapped internal edge keeps its states, reports and compile counts bit for bit, except the floor of an interface edge that widens its dtype under x64 and the spectral keys of an interface group with a field read by several internal edges (with `diagnostics=True`; 8 more Jacobian-vector products per step).  MADD-ANO-212 (a Gauss-Seidel read of a difference within one field: bound 0.054x, usable) is open for 0.5.0.
- **FMU export, round-9 audit fixes** (MADD-ANO-197 to 201; 201 since 0.3.0, the others never released): an FMU-state restore resumes the bridge's drift count instead of starting it again, so a master that saves and restores between steps is held to the time bound; an `Int64` / `UInt64` a float64 cannot hold is refused on `get` and `set` instead of rounded (`fmi3GetInt64` answered `fmi3OK` with a neighbour), and integer checks no longer compare through float64; negative zero keeps its sign on the wrapper's JSON path;
  `fmi3GetFMUState` reuses the state it is handed (it leaked one whole state per call) and `fmi3FreeInstance` frees its live states; `load_state` refuses an integer of the other signedness (`-1` for a `uint32` field loaded as 4294967295); `selected_outputs` refuses a pair that exports nothing; the instantiation token covers a variable's unit.
  Action: re-package an FMU built by an earlier development build (every token changes); free an FMU state before its instance; carry a 64-bit integer above 2^53 through the FMU-state functions.  FMU-059 to FMU-062 are new.
- **sysid round-9 fixes: bounded transforms in the fitters** (MADD-ANO-104 corrected, MADD-ANO-202, 203, never released): `fit_lm` no longer ends on the bound of a `log`/`logit` parameter that an early step carried to the edge of its range (20 of 60 starts under x64 on the parameter guide's spring) -- a step is read on the transform's tangent where that moves the value less, and the range a fit steps in ends `sqrt(eps)` of the bounds inside each bound; the fitters make one evaluation of `theta -> params`, so `best_loss` is the loss of exactly the arrays returned and a leaf a fit does not move is never run through its transform (a damping left out by `mask=` at 2.0 under a wide `logit` was run at 2.0266, the stiffness beside it fitted to 30.06 for 30);
  a fit that ends on its transform's edge, an Adam fit started on one and a fitted value its transform cannot resolve to `sqrt(eps)` of itself are each named in a warning; `node.params = {**node.params, "k": v}` writes `k` and no longer reverts the node's other calibrated constants.
  Action: declare a parameter that belongs on its bound with `transform=None`; read the new warnings; a `fit_lm` run with `log`/`logit` parameters takes another path to the same answer.  SYS-144 to SYS-147 are new.
- **Warnings in a process with several threads** (MADD-ANO-204, never released): the library's sixteen warning probes (parameter writes, `PUT /graph/params`, the sharded wrappers, the profiler) used `warnings.catch_warnings()`, which silenced every thread while one probed and, for two that overlapped, could leave every warning ignored for good; a probe now silences its own thread only and saves no filters. Action: none; other code's `catch_warnings()` in threads is still unsafe (see "Warnings when several threads run graphs" in the parameters guide).
- **Checkpoints and bfloat16** (MADD-ANO-207, never released; MADD-ANO-206, open): a checkpoint value loaded into a bfloat16 leaf is held to the cast check every other dtype gets (a float32 `1e-44` loaded as `0.0`, `3.4028235e38` as `inf`). A bfloat16 leaf still cannot be restored from its own graph's checkpoint (`load_state` raises, every release); hold such a state in float16 or float32 until 0.5.0.
- **sysid round-8 fixes** (MADD-ANO-174 to 177, never released): `fit_lm`'s loss and solves are framed by powers of two, so a float32 residual or parameter from `1e-30` to `1e30` no longer reports `converged=True` at a wrong point or its start, and `fit` lifts a flushed gradient; `fim`/`fim_core` flag an `F` flushed to zero; in-place and replaced-mapping `node.params` writes reach `gm.params`;
  a `log`/`logit` spec the leaf's dtype cannot hold is refused; float32 leaves in an x64 graph no longer creep; `fit_lm`'s floor verdict no longer follows units or an on-bound gradient's rounding; counts read integers in every spelling.
  Action: none; an ordinary `fit_lm` run moves in its last bits (the equilibrated solve's pivots).  SYS-131 and SYS-132 are new.
- **Apps built at once in several threads** (MADD-ANO-191, since 0.1.0): `SimulationServer.create_app()` builds one app at a time. FastAPI builds each route inside `warnings.catch_warnings()`, which is not thread-safe, so two builds at once could raise a warning FastAPI silences or leave `ignore::UserWarning` in the process's filters for good, dropping every MADDENING warning after it.
  Action: none on 0.4.0; before it, build the apps one after another.  An app built while another thread is inside a `catch_warnings` block of its own is not covered (the entry's residual risk).
- **The known coupling, sysid and FMU findings** (MADD-ANO-158 to 162, CPL-087, SYS-024, SYS-071): `jax.grad` / `jax.jvp` through a bfloat16 or float16 group under `solver="ift"` works (the adjoint solve runs in float32); a 16-bit group's spectral slots are float32; `strict_convergence` on a step spanning several devices raises instead of aborting the process; a typed PRNG key steps under `"ift"` with Aitken or fixed relaxation;
  the precision warning names float64 leaves when x64 is already on; `best_loss` and `losses` are the loss of exactly the parameters a fit returns (an untouched `log` leaf was evaluated at its round trip, an ulp away).
  Action: none; a 16-bit group's gradient is to its dtype's resolution.  MADD-ANO-156 and 157 (flux edges across a group's boundary or staggered by an ungrouped cycle) stay open for 0.5.0.
- **REST round-8 audit fixes** (MADD-ANO-178, 180 to 185): a params write back to a node's own value is asked the combined checks (178); `PUT /graph/state` refuses text, booleans and integers its field cannot hold (180, since 0.1.0); zero, negative or non-finite timesteps (181), a node named `_meta` (182), `/sim/start` of an empty graph (183) are refused; a 422 holding NaN is no 500 (184); a checkpoint load checks member headers before reading (185).
  Also: a non-ASCII `Host` port is a 403, not a 500; a token outside printable ASCII is refused at start-up; a comma-joined `Sec-WebSocket-Protocol` carries the token; `run_pod.py --summarise` reads any goal file's shape as INVALID and counts a file with no checks.
  Action: write state in its field's dtype; choose a printable ASCII token; re-run a summary that a mistyped file used to stop.
- **FMU export, round-8 audit fixes** (MADD-ANO-167 to 169, never released): the bridge refuses to start when a `set_param_spec` since the description (or a sidecar spec) would enforce another `min` / `max` than the XML advertises, and holds every write to the advertised bounds too; a start, step end or restored time whose 16 ulps pass a tenth of the master step is refused, and no time slack exceeds a tenth of a step;
  a `get` past what one reply frame carries is refused before anything is read (one 64 MiB frame took 6 GB), a repeated reference is read once; an open `log` / `logit` bound is advertised as the outermost value `ParamSpec.check` accepts (a float32 `logit(-1, 1)` max was refused).
  Action: build the description, sidecar and bridge after the last `set_param_spec`; start an FMU at a time its master step resolves; compute communication points as `start + k * h`; re-package an FMU with a float32 `logit` leaf (its token may change).
- **REST round-7 audit fixes** (MADD-ANO-163 to 166): `POST /checkpoint/load` no longer restores values `PUT /graph/params` refuses (163); a write to a `HybridNode`'s params reaches its step (164, lost since 0.1.0); a `/sim/run` that cannot have the graph says how many steps it took, and its final read is bounded (165); `/sim/profile` answers 400, not 500, for a graph that cannot step (166, since 0.2.0); the stop/reset 503 says the runner stays stopped; long checkpoint names save; `APIAuth(environ=)` reads the token file's path.
  `run_pod.py`: the "1e5-cell" exchange row decides (it measured 99 856 cells and never did), `--summarise` lists every row that does not decide and re-derives the speedup, a goal-named file of another goal reads INVALID, and `--dry` is refused (it used the GPUs).  Action: re-apply any `PUT` to a `HybridNode` made before 0.4.0.
- **sysid round-7 fixes** (MADD-ANO-150 to 153, never released): the identifiability guard holds an exact degeneracy under x64 and with float32 leaves in an x64 graph; `windowed_loss` replays coupling predictor and IQN-IMVJ warm starts across windows (zero loss at the truth); `fim` warns when its rank cutoff is below float32's normal range;
  bool and non-number hyper-parameters and non-bool mask leaves are refused; `ParamSpec.check` refuses a `log`/`logit` value without a finite coordinate; the nominal width guard tests the width, not its square; `fit_lm` reaches its float64 floor.
  Action: pass real numbers and bool mask leaves; a fit with a predictor or IQN-IMVJ group, or under x64, may return a different (correct) point.  SYS-071 is verified; SYS-127 to 130 are new.
- **FMU export, round-7 audit fixes** (MADD-ANO-147 to 149, never released): `build_model_description`, `FmuSidecar` and `FmuTcpBridge` refuse a graph changed since its `compile()` (an FMU built over a pending structural `node.params` write ran the old model); a `log` parameter without a lower bound advertises the smallest normal as `min`; the bridge's time tolerance is a millionth of the master step at any step, and the reported time stays within it of the simulated time;
  an input of a node the FMU does not export is held at zero, not exported; the C wrapper holds FMI 3.0's co-simulation state machine (no `fmi3DoStep` before initialization, no set once terminated, no `fmi3EnterStepMode`, configuration mode or `fmi3SetTime`).  FMU-012 and -017's wording is fixed.
  Action: `compile()` before exporting; initialize an FMU instance before reading or stepping it; re-package an FMU whose graph has a `log` parameter without a lower bound (its token changed).
- **Coupling round-5 audit fixes** (MADD-ANO-142 to 146; new, resolved): the gradient bound applies the exact resolvent to each secant (it read 5.2x below the true error, usable), probes each entry of an array constant, and certifies Kantorovich with the full resolvent norm; the spectral bound is in the returned state's weights;
  a group is one block in the schedule (an outside node between its members read it a step late, since 0.1.0) and compile() warns when a group is part of a larger loop; CouplingGroup refuses out-of-range counts and thresholds (waveform_iterations=0 froze a sub-cycling group, since 0.1.0).
  Action: re-read diagnostics=True bounds; expect a UserWarning for a group inside a larger feedback loop; fix any out-of-range CouplingGroup knob.
- **Checkpoints, the token file and the stride say what they did** (MADD-ANO-138 to 141): `load_state` refuses a value its dtype cannot hold (a float64 `1e39` loaded as `inf` since 0.1.0); `PUT /sim/stride` keeps an omitted value (it reset it to 1, since 0.1.0); the token file is a new `0600` file; a refused save writes nothing and no checkpoint 4xx names a server path;
  a JAX trace's time budget has its own timer; `run_pod.py --summarise` reads an unparseable goal file INVALID (exit 3).  Action: load a checkpoint into a graph of the precision that wrote it; send both stride values only if you relied on the reset.
- **The identifiability guard decides the same in any units** (MADD-ANO-135, never released): its tests, hold and tolerance measure an identity parameter relative to its size, as `fim(scale="relative")` does;
  **bounds checks compare exactly** (MADD-ANO-136): a float32 `-1e-40` no longer passes a `(0, None)` bound through XLA's subnormal flush; **a value its type flushes to 0 is refused** (MADD-ANO-137) like one it overflows.
  Action: none; a `fit_lm` with an identity parameter in units far from 1 may return a different (correct) point.  The sysid/FMU inventory's SYS-063, -088, -109, FMU-024 and -039 are verified (144 of 147 rows).
- **REST round-6 audit fixes** (MADD-ANO-131 to 134; 096 completed): a surrogate job re-checks its memory budget on the graph it sweeps; `POST /sim/profile` restores the live state and keeps the streams out; a reset, a state write, a node edit and a surrogate swap publish to the streams (the relay adds `run_adaptive`'s `dt`); a JAX trace stops itself at 10 000 steps or 600 s;
  `/sim/run` reports `steps_run` when a step raises; runner routes answer within one lock timeout and say when they left the runner stopped; checkpoint saves are atomic; past the stream cap a client gets 1013, not 403; `run_pod.py` records only a real commit and `--summarise` survives older files.
  Action: none; read `GET /sim/profile/jax/status` if a long trace ends early.
- **`fit_lm` no longer depends on the parameters' units, and a shrunken step cannot read as converged** (MADD-ANO-121, never released): the Marquardt floor is per column, and `converged` also needs the undamped Gauss-Newton step to be stationary;
  **every reader of `gm.params` sees a pending `node.params` write** (MADD-ANO-122, never released): one sync point, the `gm.params` getter, serves readers and runners alike, and the later write wins.  Action: none.
- **A node reading a cycle runs after it in the same step, whatever order it was added in** (MADD-ANO-120, since 0.1.0: added before a coupling group's members, it read their previous-step output); the interface norm sums its edges in the group's sweep order, not insertion order; `solver="fori"` docs: forward mode works.
  Eight coupling claims now state their true conditions (`docs/validation/coupling_claims.yaml`).  Action: rerun a 0.3.x graph whose `gm.schedule` lists a node ahead of a cycle it reads; its results change.
- **FMU export: schema-valid starts, a locale-proof wrapper, the FMI state machine** (MADD-ANO-119, never released): Boolean/integer `start`/`min`/`max` in their type's form and discrete; the C wrapper writes and reads numbers in the C locale; an archive cannot install mapping weights;
  the token covers starts and bounds; terminate refuses step/set/initialize until reset; `fmi3GetClock`/`SetClock` are `fmi3Error`; `FmuTcpBridge(idle_timeout=None)` waits for ever; a node named `x.params.y` has settable parameters.
  Action: rebuild FMUs (their tokens change) and re-package them with their bridge's description.
- **Coupling round-4 audit fixes** (MADD-ANO-113 to 118, new, resolved; 094 extended): the IFT tangent/adjoint is scale-free (it was exactly 0 for an rhs below ~1e-8, since 0.3.0); the float floor weights every read by its measured gain (a squaring ring read its bounds at 0.02-0.2x the truth, flags set) and the report reads the step's count, not the live graph;
  accelerators and the report's residuals, secant and tangents no longer flush below ~1e-31 (converged=True at 7x the threshold; a usable gradient bound of 0.0); fori+Aitken/IQN on 16-bit groups traces; per-node profiling feeds declared inputs; both adaptive steppers share one acceptance rule (`adaptive.step_decision`).
  Action: recompute IFT gradients of small-unit groups or near-minimum losses; re-read `diagnostics=True` bounds; `run_adaptive` now retries a failed attempt at `dt_min` instead of accepting it.
- **The REST server uses its graph one request at a time, and bounds what a request can take** (MADD-ANO-106 to 112; 086 and 092 completed): concurrent `/sim/step`s no longer lose steps, `/surrogate/train` leaves the live simulation alone (one job, memory estimated first), streams follow a checkpoint load and a replaced node, `/sim/run` stops on shutdown (503),
  a dead runner is reported, structural edits beside the runner are 409, bodies over 32 MiB are 413, the whole graph holds at most 1e8 state elements; `run_pod.py --keep-going` records a goal that raises, and `--summarise` exits 4 when no file records a commit.
  Action: expect 409 for writes during a `/sim/run` or beside the runner, and read `/checkpoint/load`'s `sim_time`; a 503 from `/sim/run` names the steps it took.
- **Fitters keep each parameter where `constrain` is not a clip or a clamp** (MADD-ANO-104, never released: a coordinate past one stayed there and `fit_lm` said converged); `fit_lm`'s floor is relative to the residual; `windowed_loss(start_step=)` (experimental) says where a multi-rate record began;
  `node.params` writes are counted, so every entry point recompiles for one and `load_state` supersedes them (MADD-ANO-105, since 0.1.0: `gm.step` and a new `run_scan` length ran different models).  Action: after writing `node.params` on a 0.3.x graph, `compile()` before running it.
- **The FMU bridge and C wrapper refuse what they used to coerce or drop** (MADD-ANO-100 to 103, never released): a `master_dt` other than `md.graph_timestep`, a Boolean other than 0/1, an `fmi3Get`/`Set` of another type than the variable's, a repeated value reference, a numeric set of a Clock,
  an archive missing an input; step size and communication point share one tolerance; a step takes at most `max_steps_per_request` (100000) graph steps and stops at `stop()`; `SetFMUState` checks the whole frame; the wrapper waits at most `MADDENING_FMU_TIMEOUT` (600 s).
  Action: pass `master_dt=gm.timestep`; read and write each FMU variable with the function of its type (`getFloat32` for a Float32).
- **`gm.timestep` is the step a graph takes** (MADD-ANO-096/097, since 0.1.0): a sub-cycling group counts at its largest member timestep, so the runner's and relays' clocks, USD `baseDt` and the FMU default step (all now from it) no longer run slow; `SpringDamperNode` states its coupled-pair limit `c >= k*dt` (MADD-ANO-098, open);
  MADD-ANO-099 registers the partial `external_inputs` v0.1.0-v0.3.1 did not zero-fill.  Action: on a sub-cycled graph, recompute any step count taken as `duration / gm.timestep`; keep `damping >= stiffness*dt` on coupled spring pairs.
- **`run_pod.py`'s wrapper goals see a fault confined to one spatial axis on four devices** (schema 6): each axis is split over all four on a
  1-D mesh of its own, beside a 1 x 4 two-axis mesh and the 2 x 2 pencil, on non-square grids whose inputs differ block to block (three seeded
  faults closed all six items).  `--summarise` names each item's commit, and exits 4 with `MIXED COMMITS` across commits.  Action: re-run dry runs.
- **Coupling round-3 audit fixes** (MADD-ANO-094/095, new, resolved): a Gauss-Seidel pass's float floor counts its longest chain of same-pass reads (a stalled 32-relay ring's spectral and gradient bounds read 0.5x the truth, flags set); after `run_adaptive*` the report covers both kept half steps;
  Aitken and IQN are units-invariant (exact power-of-two rescaling); `strict_convergence`'s threshold is pinned; 16-bit groups count passes in int32 and run `diagnostics=True`; x64 float32 groups step with every accelerator; the profiler's per-pass cost divides by `total_iterations` less one per sweep.
  Action: none for states (bit-identical); re-read `spectral_error_bound`/`precision_limited` of Gauss-Seidel groups and post-adaptive reports.
- **A `node.params` write after compile reaches the step at the next `compile()`** (MADD-ANO-093, never released: it was dropped); `fit_lm` reports `converged` at the float floor (its proposal is tested accepted or not; `step_tol` is relative, default 16 ulps); `windowed_loss` refuses a negative/NaN `continuity_weight` and a non-bool `mask_unconverged`;
  a clipped leaf on its bound has its full derivative into the range; a refused mask names the missing empty container.  Action: change a constant mid-run through `gm.params`; pass `step_tol` as a relative change.
- **REST, round-4 audit** (MADD-ANO-086 to 092): `POST /graph/nodes`, `PUT /graph/params`, `from_dict` and USD loading check the grid/basis nodes' size estimates before building (one `PUT n_levels=10000000` grew the server to 57.7 GB); `POST /sim/stop`
  never answers "stopped" while the runner's thread steps (503; state writes 409 while it runs); `PUT /graph/params` refuses values the step cannot trace and keeps a parameter's numeric type; the WebSocket
  streams end with their client (SIGINT hung); `PUT /sim/stride` is bounded (422); a missing edge's DELETE is a 404; root-equal checkpoint paths are a 400.  Action: send integers for integer params; stop the runner before writing state.
- **FMU export, round-4 audit** (MADD-ANO-083 to 085, never released): a new instance starts at the description's start values (it inherited the last one's state, parameters and time); inputs `selected_inputs` leaves out are held at zero (`held_inputs`); `SidecarConfig.input_resolver` steps as `GraphManager.step`; advertised min/max hold without `param_specs`.
  `dt`/`t`/values must be numbers; `doStep` must start at the FMU's time, which `fmi3EnterInitializationMode` sets; the wrapper refuses a reply longer than `nValues` and an empty token, drops a half-sent frame and sets `TCP_NODELAY` (every call took >=40 ms).
  Action: pass `input_resolver=gm._resolve_external_inputs`; build the sidecar and the description from the same parameters; open one connection per instance.
- **The shipped examples run, and print only what they measure** (49 examples; each now runs in CI or is excluded with a reason, see `examples/README.md`): two used the deprecated `RigidBody2DNode`, one raced its own server; the coupled-spring demos had no rest state yet reported "settled" (rest lengths now `+L`/`-L`);
  `vessel_flow_server`'s parameter endpoints reported success without changing the run (they write `gm.params` now); others printed claims their numbers contradicted, deleted the results they announced, or ignored `--gpu`.  Action: none, unless you copied an example -- re-copy it.
- **Coupling harness findings** (MADD-ANO-070 to 074, new, resolved): a group's integer/boolean fields are those the pass gives at the returned state, on both solvers (`ift` returned first-pass flags and froze edge-carried ones); predictor + mixed norm with such a field, Jacobi with a flux-reading producer, `reset_state`/`set_node_state` after `jax.grad`, under-relaxed `fixed` stopping on its first pass (~2x short) and the adaptive norm on integer leaves all fixed.
  `boundary_interpolation`'s "bit-identical under Jacobi" is round-off; a near-1 rate's noise-rejected ratio falls back to the raw test (MADD-ANO-005).  Action: re-run `run_adaptive*` results from graphs holding integer leaves.
- **`hold_undetermined` no longer returns a fit's parameters above the loss it reached** (MADD-ANO-069, never released): a direction is held only if the
  run's gradients missed it *and* the loss has no curvature there, and a hold that would raise the loss beyond rounding is refused (`FitResult.hold_declined`,
  `RuntimeWarning`).  `fit_lm` on a well-posed bowl went 0.0 -> 0.22.  Action: re-run guarded fits from earlier 0.4.0 builds, or compare to `best_loss`.
- **Sharding, from the differential harness**: `ShardedUnstructuredNode` refuses a per-cell input neither in partition layout nor broadcastable
  (a slab-length one was read by every shard as its slab; MADD-ANO-064); `gather_global` passes an integral listed in `state_fields()` through
  (MADD-ANO-065); a nested stencil wrapper starts a per-shard integral stacked once (MADD-ANO-066); zero-ghost reverse scans no longer segfault jaxlib 0.11.2
- **State and IO, from the differential harness** (MADD-ANO-063, 049): a write that moves the points a mapped edge was built from (a uniform `HeatNode`'s `length` under a mapping on its `grid_x`) is refused -- `PUT /graph/params` 400, `gm.params` at the next run, `to_dict` and `save_state`, `POST /checkpoint/load` undone, an FMU parameter fixed -- where it ran on the old mapping weights and saved a config that did not load;
  state replies, `GET /graph/params` and `/ws/state` write a non-finite float as its `json_codec` token (a `diagnostics=True` group's NaN seeds made `POST /sim/reset`, `GET /graph/state` and `POST /checkpoint/load` a 500 after applying); `PUT /graph/params` and `/graph/state` refuse a float32-overflowing value before the cast (a 500 under `-W error`); the FMU bridge and sidecar restore their own snapshot when a parameter started outside its bounds or a state field started non-finite (a diagnostics group's NaN `_meta` seeds).
  Action: read `"NaN"` / `"Infinity"` / `"-Infinity"` in a state reply (`json_codec.loads` decodes them); to change such a geometry parameter, rebuild the node and its mapped edge.
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
  start** (MADD-ANO-079, since 0.1.0) — only graphs with a rate divider > 1 or a warm start were ever affected;
  `set_param_spec` and `external_inputs` are now checked as strictly as `params`
- **Swapping a surrogate in or out no longer resets an edge's `additive`, units,
  `mapping` or fitted mapping weights** (MADD-ANO-077, since 0.1.0): an additive input read 3.0 before a swap and
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
  method that forwards inwards, so a static rewritten in place is not baked in (MADD-ANO-081, never released)
- **Examples no longer save plots into the installed package** (they broke on
  a read-only install): output goes to the working directory, usage lines use
  `python -m maddening.examples...`, and a smoke test pins both
- **A coupling group no longer reports convergence it has not reached**:
  `acceleration="aitken"` needs the threshold met on two consecutive passes
  (a lone dip is not arrival), `max_iterations=1` reports its real residual
- Sharded pointwise nodes honour parameter writes again (`PUT /graph/params`)
- `POST /surrogate/deactivate` restores every edge field, or changes nothing (MADD-ANO-077)
- A `.` in a node name no longer misroutes that node's FMU inputs and outputs (MADD-ANO-080, never released)
- **FMU export of a real graph had no inputs and a wrong step size** (MADD-ANO-078, since 0.3.0): inputs
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
- **Tests for verdicts taken over a span of steps**: a `mask_unconverged` window whose early steps hit the cap and whose last step converged, the profiler's fractions on such a window, and the batched `strict_convergence` firing gate; the slow normal-contraction property asserts the bound where `spectral_usable` and holds the flag's share of its draws to a floor (it asserted the flag on every draw, which no claim promises). `docs/user_guide/inspection.md` states what `diagnostics=True` costs (a GPU measurement in `benchmarks/results/gpu_eigvals_probe/`; MADD-ANO-232 records a not-usable value that differs in kind between backends).
- **The step-program gate lowers each graph from emptied jax caches** (`scripts/capture_step_programs.py`, `tests/core/test_step_program_digests.py`): eleven gate graphs failed in a slow-lane process with no program changed. The lowered text also says how many identical copies of its own helpers (`isinf`, `frexp`, `_where`) jax wrote out, which depends on what its bounded trace caches still hold. `graph_digests` now calls `jax.clear_caches()` before it builds a graph; the capture of 396eb59a is unchanged and still holds for all 24 graphs on jax 0.10.2, 0.11.0 and 0.11.2.
- **REST write-sequence oracle: a run that fails part-way is held to a replay of the steps it counted** (`tests/property/injected_failures.py`): the rule "an empty graph's clock stays" was wrong for a graph emptied over the API, whose steps the streams go on counting, and failed the slow hundred-request machine; the server was consistent. A per-push example now runs an emptied and a never-filled graph with a later slice failing
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
- Full MADDENING test suite on 2026-10-02: **8634 tests collected** locally, 8169
  in CI's default lane (`-m "not slow"`, `--ignore=tests/viz`; 432 slow deselected).
  These counts are regenerated at the tag, with CI's pass/skip figures from the tag's
  run — see `docs/release_notes/v0.4.0.md`.  v0.2.1's own
  Verification block, which an edit during this cycle had moved here, is back
  under [0.2.1] as released
- Differentiable sharded solves and the C1 multi-physics IQN-IMVJ case match
  their dense and `fori` references in both differentiation modes

### Security
- **`X-Forwarded-For` no longer gets a request past a loopback-bound API's `Host` check or its peer backstop** (CRITICAL, MADD-ANO-179, never released): uvicorn's default proxy headers let any request over loopback name its own peer, so a rebound page sending `X-Forwarded-For: x` drove every route;
  a peer that is not a loopback IP literal and any request carrying `X-Forwarded-For` / `Forwarded` now need the token, the `Host` check asks every request without one, and the library's launch paths pass `proxy_headers=False`.
  Action: never configure loopback as a trusted proxy; an in-process `TestClient` presents `server.auth.token` (or uses `base_url="http://127.0.0.1", client=("127.0.0.1", 50000)`); a reverse proxy to a loopback bind presents the token.
- **A loopback-bound API answers only to loopback host names** (CRITICAL, MADD-ANO-076, now resolved): a DNS-rebinding page's `Host` and
  `Origin` agree, so it passed the Origin check and could drive every route, `/cloud/launch` included; any other `Host` is now a 403.
  Action: pass `SimulationServer(allowed_hosts=)` to serve a loopback-bound server under a proxy's name or an `/etc/hosts` alias
- **Cloud launches reach ready again, and cross-origin browser requests are
  refused** (CRITICAL, MADD-ANO-076, since 0.1.0; partially resolved, a DNS-rebinding page still passes): set `MADDENING_TRANSPORT_TOKEN` so the ZeroMQ CURVE key is not the
  cleartext API bearer token; pass `allowed_origins=` to embed the UI elsewhere
- **The ZeroMQ transports bind loopback and encrypt any other bind** (CRITICAL, MADD-ANO-015):
  5555/5556/5580 published state, commands and worker rendezvous to anyone who
  could reach them; set `MADDENING_TRANSPORT_TOKEN` on both sides to stream off-box
- **The API requires a bearer token unless it is bound to loopback** (CRITICAL, MADD-ANO-051):
  set `MADDENING_API_TOKEN` or read the one logged at start-up; `JobConfig.ports`
  no longer defaults to `[8000]`, so a cloud launch stops opening the API port
- **FMI/USD hardening** (three HIGH): a silent TCP peer no longer wedges the FMU
  bridge, `set_state` is value-checked exactly as `set` is (MADD-ANO-082), and loading a USD
  stage no longer imports the class it names (MADD-ANO-075, since 0.1.0) — pass `node_registry=` to allow one
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
- **MADD-ANO-216 (new, never released, resolved)**: `fit_lm` reported `converged=True` beside a jump of a non-differentiable residual. **MADD-ANO-217 (new, open)**: `fit_lm` from a stiffness started 18 to 80 times too high can stop short, unconverged, on the end of a wide damping range; restart from the returned point. **MADD-ANO-021 (open)** gains its manifestation in system identification (what `fit_lm`, `fit` and `fim` say on a residual with jumps).
- **MADD-ANO-195 (never released) is now resolved**: the interface norm, its floor, its bound and the report read a mapped edge as the step delivers it.  **MADD-ANO-211 (new, never released, resolved)**: the float floor of an interface edge that delivers a wider dtype than its source field was taken at the wider dtype's `eps` (see `### Fixed`).  **MADD-ANO-213 (new, never released, resolved)**: under the interface norm a field read by several internal edges was counted once per edge by the residual and once by the spectral analysis (see `### Fixed`).  **MADD-ANO-212 (new, never released, open)**: the floor's gain of a same-pass read is measured along the source's own state, which a difference of entries of one field cancels -- a Gauss-Seidel group stalled behind a mapping row `[1, -1]` reports `spectral_error_bound` at 0.054x to 2.6e-4x the true distance, `spectral_usable=True`; use Jacobi there, or have the producer hold the difference as its own field.
- **MADD-ANO-178 to 185 (new, resolved in this release)**: the REST round-8 audit's defects (see `### Security` and `### Fixed`): 178 (params write back) and 179 (forwarded peer, critical) never released; 180 to 185 carried from v0.1.0 to v0.3.1.
- **MADD-ANO-163 to 166 (new, resolved in this release)**: a checkpoint load restored parameter values `PUT /graph/params` refuses (163, never released); a `HybridNode` parameter write was lost (164, since 0.1.0); `/sim/run`'s 503 said nothing changed after it had stepped (165, never released); `/sim/profile` was a 500 for a graph that cannot step (166, since 0.2.0).  See `### Fixed`.
- **MADD-ANO-075 to 082 (new)**: defects this release fixes that had no entry, five of them carried by releases — a USD stage imported the module it named (075, since 0.1.0); the API checked no `Origin` (076, critical, since 0.1.0, `partially_resolved`: a DNS-rebinding page still drives a loopback bind, so run the API only while you use it); a surrogate swap dropped edges' `additive` and units (077, since 0.1.0); FMI export of a real graph had no inputs and a 1e-3 step (078, since 0.3.0); a recompile re-phased a multi-rate graph (079, since 0.1.0) — and three majors never released (080, 081, 082; see `### Fixed`, `### Security`).
  **MADD-ANO-048** is `partially_resolved`: a `gm.params` write or a fit still saves a value the constructor refuses, and the saved graph then does not load; bound such a parameter inside the constructor's range.  **MADD-ANO-058** now also covers `propeller_radius > 1`.
- **MADD-ANO-069 (new, never released)**: `hold_undetermined` held every direction a fit's gradients had not spanned, so a short or fast-converging fit came back above the loss it had reached (0.4.0 development builds only; see `### Fixed`)
- **MADD-ANO-064 to 067 (new, resolved in this release)**: the four sharding defects above (064, 065 and 067 since 0.2.1 or 0.3.0; 066 never released).
  **MADD-ANO-068 (new, `partially_resolved`)**: XLA (jaxlib 0.10.2 to 0.11.2) miscompiles a `ShardedStencilNode` step inside `run_scan` for a node reading a sharded
  static replicated over a mesh axis in its halo beside a window at its `shard_info` offset; the loop entry points now refuse such a node (see `### Changed`), and a loop you write yourself is not asked
- **MADD-ANO-063 (new, never released)**: a write moving the points a mapped edge was built from ran on the old mapping weights and saved a config that did not load (see `### Fixed`); **MADD-ANO-049** is now resolved (an in-process non-finite parameter is served as a token, not a 500), and **MADD-ANO-022** narrowed: a write is refused, a fit through the mapped edge still uses the constructor's geometry
- **Severities defined; fourteen relabelled; four entries partially resolved**: `AnomalySeverity` now defines each level, and a silent wrong result is never `minor`, so MADD-ANO-005, 009, 025, 027, 029, 031, 038, 041, 048, 055, 057, 058 and 061 move to `major`, and 052 (a default-exposed route) to `critical`.
  MADD-ANO-032, 036, 047 and 048 are `partially_resolved`, not `resolved`: their routes outside a graph or the REST route are still live (see each `residual_risk`); the release notes' Known anomalies section now names every reachable entry, and a test keeps it so.
  Sharded nodes: iterate `update_padded`'s `shard_info` over its `int` keys only; `"n_local"` (unstructured wrapper) is the one string key.
- **MADD-ANO-062 (new, resolved in this release)**: `HeatNode` accepted a non-positive `length` or `timestep`, a negative `thermal_diffusivity` and `grid_points` out of order, and answered wrongly without a word (since 0.1.0; see `### Changed`)
- **MADD-ANO-059, 060, 061 (new, resolved in this release)**: accelerating a coupling group holding an integer, boolean or PRNG-key leaf raised a `TypeError`; a group-internal flux edge read by the interface norm or a sub-cycled member's linear interpolation raised a bare `KeyError`; `run_adaptive` advanced its clock by `dt_min` on a `dt_min` accept whose state covered more (all since 0.1.0; see `### Fixed`, `### Changed`).
  **MADD-ANO-027** now says each extra waveform sweep applies at least one more pass, moving a converged state by about one residual
- **MADD-ANO-051, 052, 053 (new, resolved in this release; all since 0.1.0)**: the HTTP API served every route, `/cloud/launch` included, with no credential while the container bound `0.0.0.0`; the checkpoint routes took any server path; the signaling server admitted every client (see `### Security`).
  **MADD-ANO-054 (new, never released)**: the FMU bridge unpickled the importer's state blob, remote code execution; **MADD-ANO-055 (new, resolved)**: `deserialize_fmu_state` and `FmuSidecar.handle` unpickled their input (since 0.3.0).
  The registry now holds every defect a release carried and every critical or major one found in the cycle, shipped or not (CONTRIBUTING.md)
- **MADD-ANO-047, 048, 049 (new; 049 resolved in this release, 047 and 048 `partially_resolved`)**: `PUT /graph/params` accepted a write flipping a branch the node fixed at construction, values its constructor refuses, and a non-finite value (since 0.1.0; see `### Fixed`).
  **MADD-ANO-050 (new, open)**: two `HeatNode` rods coupled end to end by a converged exchange are unstable above Fo = 3/8 at `stencil_order=2` and 0.226 at 4, not the 1/2 or 5/16 each accepts; keep Fo below those on such pairs (`compile()` warns; see `### Changed`)
- **MADD-ANO-046 (new, resolved in this release)**: a sub-cycled node whose timestep did not divide the macro timestep drifted by a fixed fraction of every step (since 0.1.0; see `### Changed`)
- **MADD-ANO-043, 044, 045 (new, resolved in this release)**: `run_adaptive*` advanced a sub-cycled node `divider * dt` per step; a multi-rate group's diagnostics, predictor and IQN-IMVJ warm start came from discarded solves;
  `solver="fori"` + `iqn-imvj` carried zero secant columns, so `jacobian_reuse` did nothing (all since 0.1.0; see `### Fixed`)
- **MADD-ANO-037 to 042 (new, resolved in this release)**: `ShardedUnstructuredNode` stepped a Cartesian stencil node wrong, dropped cells past its layout's count and gave no way to leave padding out of an integral (all since 0.3.0); both sharded wrappers placed a domain integral in the state like a grid field (since 0.2.1);
  a sharded static was edge-filled under a periodic wrapper (since 0.2.1); `halo_exchange` ignored an unknown `boundary` key and `ShardedStencilNode` accepted an empty `axis_map` (since 0.2.0).  See `### Fixed` and `### Changed`
- **MADD-ANO-034 (new, resolved in this release), 036 (new, `partially_resolved`)**: a walled `LBMNode` reloaded with no walls (since 0.1.0; see `### Fixed`); a config round trip dropped a node's sharding with no word (since 0.2.0; now a warning, see `### Changed`).
  **MADD-ANO-035 (new, open)**: on a balanced partition not in global order, `ShardedUnstructuredNode` reads a global-order state written with `set_node_state` as partition layout, so each cell steps from another cell's value (since 0.3.0).
  Convert the state with `partition_value` first, or renumber the cells with `np.argsort(partition_assignment, kind="stable")`; an explicit layout on the state write is planned for 0.5.0
- **MADD-ANO-033 (new, resolved in this release), 032 (new, `partially_resolved`)**: `ShardedStencilNode` stepped with a float32 `dt` under x64; the sharded wrappers kept a compiled step across `compile()`, so a legacy node's params write after a step never reached the sharded physics (a wrapper's own `update()` still does until `invalidate_static_cache()`; both since 0.2.0; see `### Fixed`).
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
  hidden slow mode (major, partially_resolved in 0.4.0, context_dependent)
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
