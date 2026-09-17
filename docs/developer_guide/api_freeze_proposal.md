# 1.0 API freeze — proposal

**Status: proposal. Nothing here is decided.** It recommends a stability level
for every tagged surface, lists the surfaces that are public in practice and
carry no tag, and writes up the six questions the freeze has to answer. A
follow-up branch applies whatever the repository owner agrees to; this branch
deliberately changed no `@stability` tag.

Written against the tree at the head of `release/0.4.0` plus this branch:
**79 tagged surfaces — 13 stable, 55 evolving, 11 experimental**
(`python scripts/generate_stability_report.py`).

## How to read this

### What a level costs

Promoting a surface to `stable` says its signature will not change
incompatibly before the next major version. The mechanics are in
[the deprecation policy](deprecation_policy.md); the price is:

- a breaking change needs two minor releases of `DeprecationWarning` and can
  only land in a major release;
- **lowering** the level again is itself a breaking change, so a promotion is
  hard to take back;
- a `stable` *class* promises every public method and property an instance
  answers to, inherited ones included — `docs/developer_guide/stable_api.json`
  records 223 members behind today's 13 tagged surfaces.

One clarification the freeze should adopt explicitly, because the level
docstrings are ambiguous about it: **between 0.4.0 and 1.0.0 a `stable` tag is
announced intent, not yet the contract.** The snapshot guard makes every
change to such a surface visible and deliberate in the meantime. At 1.0.0 the
snapshot *becomes* the contract. Without that reading, the
[PEP 589 typing phase](typing.md), which replaces bare `dict` annotations with
`TypedDict`s on exactly these surfaces, would be a breaking change on the day
it lands — and the agreed branch order puts it *after* this freeze.

### The evidence behind each row

Per surface, gathered mechanically:

- **tests** — number of files under `tests/` naming the surface;
- **ex** — number of files under `src/maddening/examples/` naming it;
- **down** — files in the MIME and MICROROBOTICA checkouts that name it *and*
  import `maddening`. A short name (`fit`, `enable`, `constrain`) is a noisy
  grep; treat those two columns as an order of magnitude, not a count;
- **churn** — commits touching the defining file since 2026-08-01, i.e. the
  0.4.0 cycle;
- **dict** — parameters annotated as a bare `dict`, the
  [TypedDict targets](typing.md);
- **conf** — how confident I am in the recommendation: `high` / `med` / `low`.
  Every `low` is a place to overrule me.

### The criteria I applied

A surface is proposed `stable` when all of these hold: it is covered by tests
that exercise its contract rather than its implementation; it did not change
in the 0.4.0 cycle, or changed only additively; it names no third-party type
in its signature that we do not control; and downstream (examples, MIME,
MICROROBOTICA) already depends on it, so the promise is being made in practice
whether or not it is written down.

A surface stays `evolving` when it is new in 0.4.0 (the mapping layer, the
params layer, sysid, the FMU bridge, `AdaptiveNode`), or when one of the open
questions below can still move it.

A surface stays `experimental` when its *physics* or *algorithm*, not just its
signature, is still being decided.

---

## Summary

| | today | proposed |
|---|---|---|
| stable | 13 | **22** |
| evolving | 55 | 42 |
| experimental | 11 | 10 |
| internal (first use of the level) | 0 | 5 |
| **tagged total** | **79** | **79** |

Nine promotions to `stable`: `ParamSpec`, the four `core.params` free
functions, the two preconditioner factories, `get_directional_derivative` and
`BinaryStateEncoder`. Five demotions to `internal` (three adaptive/mapping
plumbing helpers plus the two diagnostics toggles), four to `experimental`
(`AdaptiveNode` and its error type, `rbf_matrix`, `fit_multiple_shooting`),
and five promotions out of `experimental` to `evolving` (`RigidBodyNode`,
`RigidBody2DNode`, `HeartPumpNode`, `SurrogateNode`, `SurrogateArchitecture`).

Separately, about **30 untagged surfaces** are named below as needing a
decision; **9 of them are proposed `stable`**, six because a `stable`
signature already names them. Counting those, the proposed frozen surface is
**31**, not 22.

Confidence across the 79 rows: 46 high, 30 medium, 3 low. The three low ones
are `ShardedUnstructuredNode`, `core.params.check_bounds` and
`get_directional_derivative`; each says in its row what would settle it.

This is short of the 40–60 the 0.4.0 plan guessed, and the churn column says
why: **71 of the 79 surfaces were touched during this cycle, 48 of them three
times or more.** Only eight are unchanged since the spring. The mapping layer, the
params layer, sysid, the whole FMI package and `AdaptiveNode` were written or
rewritten in the last three weeks, several of them after an audit found a bug
in them. Freezing a signature that is three weeks old buys a promise we would
rather not have made.

---

## core — graph, node, edge, static data

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `core.graph_manager.GraphManager` | stable | **stable** | high | 117 | 39 | 226 | 18 | 0 | The framework. Nothing else can be frozen if this is not. But see the three caveats below — it is also the highest-churn file behind any tagged surface. |
| `core.node.SimulationNode` | stable | **stable** | high | 41 | 2 | 83 | 5 | 0 | The extension point: 83 downstream files name it, most of them subclassing. Its `update(state: dict, boundary_inputs: dict, dt, *, params=None) -> dict` is the single most important TypedDict target. |
| `core.edge.EdgeSpec` | stable | **stable** | high | 13 | 1 | 38 | 3 | 0 | Value object, ten fields, additive-only history. `mapping: Optional[Any]` should be narrowed to the `Mapping` protocol first (see below). |
| `core.static_data.StaticArray` | stable | **stable** | high | 9 | 0 | 19 | 0 | 0 | Unchanged since 2026-05-31, has its own migration guide, 19 MIME files. The easiest row in the table. |

Three things to settle **before** the `GraphManager` promise bites:

1. `to_dict` / `from_dict` have no coupling-group field (open question 3
   below). Adding one is additive, so it is not blocked by the freeze — but it
   is much cheaper to add now than to explain later why a `stable`
   round trip drops a coupling group.
2. `gm.params` is an instance attribute assigned in `__init__`, so the
   signature guard cannot see it, and it is as public as any method
   (`gm.params["nodes"][name][key]` appears throughout the parameters guide).
   Either make it a property, so it is recorded, or state in the freeze that
   its *shape* is frozen by the parameters guide rather than by the snapshot.
3. `validate_sharding()` returns `list['ShardingIssue']`, `add_coupling_group`
   takes a `CouplingGroup`, `uncertainty_spec()` returns
   `Optional['UncertaintySpec']` — none of those three types carries a tag.
   A `stable` method whose type is unfrozen is only half a promise. See
   [Untagged surfaces](#untagged-but-public-in-practice).

## core.params

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `core.params.ParamSpec` | evolving | **stable** | med | 24 | 0 | 0 | 4 | 0 | Frozen dataclass, five scalar fields, documented in `user_guide/parameters.md`, round-trips through config, USD and FMI. The shape is as settled as anything added this cycle gets. |
| `core.params.constrain` | evolving | **stable** | med | 4 | 0 | 0 | 4 | 2 | Already promised: `GraphManager.constrain` is in the `stable` snapshot and delegates here. Leaving the free function `evolving` while the method is `stable` is a distinction no caller can act on. |
| `core.params.unconstrain` | evolving | **stable** | med | 5 | 0 | 0 | 4 | 2 | Same argument as `constrain`. |
| `core.params.trainable_mask` | evolving | **stable** | med | 8 | 0 | 0 | 4 | 2 | Same argument; `GraphManager.trainable_mask` is in the snapshot. |
| `core.params.check_bounds` | evolving | **stable** | low | 1 | 0 | 0 | 4 | 2 | Same argument via `GraphManager.check_params`, but one test file is thin evidence for a promise. Overrule me to `evolving` if the coupling to the method is judged too indirect. |

All four take `(params: dict, specs: dict)`. That is eight of the 235 bare
`dict`s the typing phase wants to replace — and exactly why the pre-1.0
reading above matters: the same `dict`s are already inside the `stable`
`GraphManager` snapshot, so bare `dict` cannot be a promotion blocker here
without being a demotion argument there.

## core.coupling — interface mapping (all new in 0.4.0)

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `coupling.mapping.rbf_mapping` | evolving | **evolving** | high | 7 | 0 | 0 | 4 | 0 | Written this cycle, last touched 2026-09-17, no downstream user yet. |
| `coupling.mapping.nearest_neighbor_mapping` | evolving | **evolving** | high | 4 | 0 | 0 | 4 | 0 | As above. |
| `coupling.mapping.projection_1d_mapping` | evolving | **evolving** | high | 5 | 0 | 0 | 4 | 0 | As above. |
| `coupling.mapping.matrix_mapping` | evolving | **evolving** | high | 8 | 0 | 0 | 4 | 0 | Its `asset=` argument exists only because `MappingSpec` cannot serialise a matrix inline; that design is one release old. |
| `coupling.mapping.StaticLinearMapping` | evolving | **evolving** | high | 2 | 0 | 0 | 4 | 1 | `meta: dict = None` is both a bare dict and a mutable-looking default annotated non-optionally. Fix before any promotion. |
| `coupling.mapping.rbf_matrix` | evolving | **experimental** | med | 1 | 0 | 0 | 4 | 0 | A kernel-matrix builder shared with `rbf_interpolation`. One test file, no doc, no caller outside the package: a building block that got a tag because it sits next to ones that needed it. |
| `coupling.mapping_spec.MappingSpec` | evolving | **evolving** | high | 6 | 0 | 0 | 3 | 2 | The serialisation format itself. `hyperparameters: dict` and `points: dict` are its whole content and both are untyped. |
| `coupling.mapping_spec.build_mapping` | evolving | **evolving** | med | 0 | 0 | 0 | 3 | 0 | No test file names it; it is covered only through `from_dict`. Either give it direct tests or make it `internal` — it is the kind of surface that is public by accident. |
| `coupling.mapping_spec.make_point_resolver` | evolving | **internal** | med | 2 | 0 | 0 | 3 | 0 | Plumbing between `from_dict`/USD and the point references. `GraphManager.point_resolver()` is the public way in, and it is already in the `stable` snapshot. |
| `coupling.mapping_spec.point_array_digest` | evolving | **internal** | med | 3 | 0 | 0 | 3 | 0 | A content hash used to detect that referenced points moved. Nothing outside the package should call it. |

## core.simulation — compile cache and profiler

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `simulation.compile_cache.enable` | evolving | **evolving** | med | 3 | 0 | 7 | 1 | 0 | Downstream already calls it, and the signature is two arguments — but it is one release old, and `min_compile_time_secs` is a tuning knob whose default may want to move. One more release. |
| `simulation.compile_cache.warm_cache` | evolving | **evolving** | high | 1 | 0 | 0 | 1 | 1 | One test file; `external_inputs: Optional[dict]` untyped. |
| `simulation.compile_cache.enable_from_env` | evolving | **internal** | high | 1 | 0 | 0 | 1 | 0 | Called by `compile()` to honour the env var. No caller should ever call it directly; the env var is the surface. |
| `simulation.profiler.profile_graph` | evolving | **evolving** | high | 3 | 2 | 8 | 2 | 1 | Rewritten this cycle. Returns `ProfileReport`, which carries no tag — tag that before promoting this. |
| `simulation.profiler.TraceSummary` | evolving | **evolving** | high | 1 | 0 | 0 | 2 | 0 | A report dataclass whose fields are the surface; the trace-attribution work that fills them is explicitly still coarse (`TODO.md` performance item 5). |

## core.solver_utils

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `core.solver_utils.ift_linear_solve` | experimental | **experimental** | high | 4 | 0 | 1 | 3 | 0 | Blocked on open question 5: the signature hides `restart`/`max_steps`/`stagnation_iters` and lets `equinox._errors._EquinoxRuntimeError` out of a MADDENING function. Cannot be frozen with a third-party private error type in its `Raises` section. |

## nodes

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `nodes.ball.BallNode` | stable | **stable** | high | 65 | 22 | 93 | 2 | 0 | Four float constructor arguments beyond `name`/`timestep`; the quickstart node. |
| `nodes.spring.SpringDamperNode` | stable | **stable** | high | 69 | 17 | 62 | 4 | 0 | Same shape, the most-tested node in the tree. |
| `nodes.table.TableNode` | stable | **stable** | high | 42 | 20 | 73 | 1 | 0 | Three arguments. Nothing to get wrong. |
| `nodes.heat.HeatNode` | stable | **stable** | high | 50 | 10 | 50 | 5 | 0 | **Fix two annotations first**: `geometry_source: str = None` should be `Optional[str]`, and `grid_points` is unannotated. Neither is a runtime change, both will trip the guard, so do it in the same commit as the freeze rather than after it. |
| `nodes.rigid_body.RigidBodyNode` | experimental | **evolving** | med | 5 | 0 | 55 | 2 | 1 | 55 downstream files use a node whose tag says "may break in any minor release". That mismatch is the single worst tag in the tree. Not `stable`, because `constraints: dict \| None` and `inertia: tuple \| list` are both unsettled shapes, and five test files is thin for 55 dependants. |
| `nodes.rigid_body_2d.RigidBody2DNode` | experimental | **evolving** | med | 6 | 2 | 13 | 1 | 0 | Nine scalar-or-tuple arguments, no dicts, used by examples. `stable` is defensible; I did not want to jump two levels on six test files. Flagging it as a place to overrule me upward. |
| `nodes.heart_pump.HeartPumpNode` | experimental | **evolving** | med | 5 | 1 | 9 | 2 | 0 | Seven float arguments, two-element Windkessel, physics settled. `experimental` understates it. |
| `nodes.health_check.HealthCheckNode` | experimental | **experimental** | high | 3 | 0 | 7 | 1 | 1 | `checks: dict \| None` is an open-ended diagnostic contract. |
| `nodes.lbm.LBMNode` | experimental | **experimental** | high | 9 | 2 | 11 | 3 | 0 | The params migration found and fixed real bugs in the LBM nodes this cycle; the lattice/boundary arguments are research surface. |
| `nodes.lbm_pipe.LBMPipeNode` | experimental | **experimental** | high | 4 | 3 | 14 | 1 | 0 | Fifteen constructor arguments including `propeller_*`; a demo geometry, not an API. |
| `nodes.adaptive.base.AdaptiveNode` | evolving | **experimental** | med | 9 | 0 | 1 | 4 | 0 | Two of the six open questions are about its own signature (hooks, dtype), and `MADD-ANO-003` is open against its gradient. **This demotion is free today and breaking after 0.4.0 ships**, because no release has yet carried the `evolving` tag. If it must stay `evolving`, questions 2 and 6 have to be answered in 0.4.0. |
| `nodes.adaptive.base.AdaptiveNodeBlindnessError` | evolving | **experimental** | med | 5 | 0 | 0 | 4 | 0 | Follows the node; an exception type is only as stable as what raises it. |
| `nodes.adaptive.base.set_adaptive_diagnostics` | evolving | **internal** | med | 1 | 0 | 0 | 4 | 0 | A process-global diagnostics toggle, one test file, no downstream user. Global mutable switches are not the kind of thing to promise for a major version. |
| `nodes.adaptive.base.adaptive_diagnostics_enabled` | evolving | **internal** | med | 1 | 0 | 0 | 4 | 0 | Reader for the same toggle. |

## cloud.multigpu

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `multigpu.sharded_node.ShardedStencilNode` | stable | **stable** | high | 16 | 0 | 8 | 6 | 0 | Sixteen test files, a topology guide, a halo-width migration guide. The churn is bug fixes behind a fixed signature. |
| `multigpu.sharded_node.ShardedPointwiseNode` | stable | **stable** | high | 3 | 0 | 2 | 6 | 0 | Three arguments; already survived one rename cycle (`ShardedNode` → this, deprecated through v0.3). |
| `multigpu.sharded_unstructured.ShardedUnstructuredNode` | stable | **stable** | **low** | 6 | 0 | 3 | 5 | 0 | The signature is fine; `exchange: str = "all_to_all"` is not. The NCCL session (`verify/multigpu-nccl-session`) exists to decide whether `"ppermute"` becomes the default, and **changing a default is breaking**. Either run that session before the freeze, or freeze this with the default explicitly called out as still open. |
| `multigpu.iterative_solver.sharded_cg` | stable | **stable** | med | 5 | 0 | 2 | 2 | 0 | Eleven arguments, three of them (`preconditioner`, `backend`, `differentiable`) added this cycle — additively, so no promise was broken. Returns `SharedSolveResult`, **untagged**: tag it before freezing this. |
| `multigpu.iterative_solver.sharded_gmres` | stable | **stable** | med | 2 | 0 | 2 | 2 | 0 | As above, plus `restart: int = 50` is a performance default that the NCCL session could argue with. Two test files for a `stable` surface is the thinnest coverage in the frozen set. |
| `multigpu.iterative_solver.jacobi_preconditioner` | evolving | **stable** | med | 1 | 0 | 0 | 2 | 0 | `(diag: jax.Array) -> Callable[[jax.Array], jax.Array]`. There is no version of this function with a different signature. One test file, but the signature carries no risk. |
| `multigpu.iterative_solver.block_jacobi_preconditioner` | evolving | **stable** | med | 1 | 0 | 0 | 2 | 0 | Same argument, `blocks` instead of `diag`. |
| `multigpu.halo_unstructured.build_unstructured_partition` | evolving | **evolving** | high | 7 | 0 | 2 | 3 | 0 | Keyword-only and well tested, but it is the entry point to the layout that the NCCL session may reshape. |
| `multigpu.halo_unstructured.exchange_unstructured` | evolving | **evolving** | high | 3 | 0 | 3 | 3 | 0 | `method: str = "all_to_all"` — the same default question as the node. |
| `multigpu.halo_unstructured.partition_value` | evolving | **evolving** | med | 4 | 0 | 3 | 3 | 0 | Stable-looking, but it travels with the layout type. |
| `multigpu.halo_unstructured.gather_value` | evolving | **evolving** | med | 2 | 0 | 3 | 3 | 0 | As above. |
| `multigpu.halo_unstructured.UnstructuredPartitionLayout` | evolving | **evolving** | high | 2 | 0 | 3 | 3 | 0 | An eleven-field dataclass of numpy internals. Freezing it freezes the partitioning implementation, not just its interface. |
| `multigpu.halo_unstructured.exchange_traffic` | evolving | **evolving** | high | 1 | 0 | 0 | 3 | 0 | A measurement helper written to support the NCCL comparison; it may not outlive it. |

## cloud

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `cloud.providers.CloudProvider` | evolving | **evolving** | high | 2 | 0 | 6 | 0 | 0 | Untouched since May and used downstream, but it is a `Protocol`: its promise is its method set, and provider support is still growing (`cloud-all` extra). |
| `cloud.resume.download_and_load_state` | evolving | **evolving** | med | 3 | 0 | 3 | 2 | 0 | Has a user guide and a live deprecated alias pointing at it — but it *grew* `manifest_url` and `timeout` this cycle, and the deprecated alias at the old path still does not accept either. Promote at 0.5.0 once the alias and the target agree. |

## api

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `api.binary_encoder.BinaryStateEncoder` | evolving | **stable** | med | 4 | 1 | 9 | 0 | 0 | Untouched since 2026-05-31, nine downstream files, already typed (`dict[str, dict[str, Any]]`). The caveat is that the *signature* is not really the promise — the frame layout is — and the frame layout has no version field that a reader can check. Promote the signature; treat the wire format as a separate decision. |

The REST server itself (`maddening.api.server`) carries no tag at all, and its
response schema is the subject of open question 1. See
[Untagged surfaces](#untagged-but-public-in-practice).

## fmi

Everything in this package was written or rewritten during 0.4.0 — the C
wrapper, the TCP bridge, the binary frames, the multi-clock export — and two
audit rounds fixed a critical unpickling hole and several protocol bugs in it.
Nothing here should be frozen this cycle.

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `fmi.directional_derivatives.get_directional_derivative` | evolving | **stable** | low | 3 | 0 | 3 | 0 | 0 | The one exception: unchanged since May, FMI-3.0-defined semantics, three downstream files. Conditional on tagging `DirectionalDerivativeKind`, which its `kind=` argument names and which has no tag. If that is not done, keep it `evolving`. |
| `fmi.model_description.build_model_description` | evolving | **evolving** | high | 10 | 0 | 2 | 4 | 0 | Grew `include_parameters` and `multi_clock` this cycle; returns the untagged `ModelDescription`. |
| `fmi.sidecar.FmuSidecar` | evolving | **evolving** | high | 8 | 0 | 1 | 4 | 0 | Its `handle` path is documented as in-process only (pickle); that constraint belongs in the contract before the contract is frozen. |
| `fmi.sidecar.SidecarConfig` | evolving | **evolving** | high | 8 | 0 | 1 | 4 | 2 | `params`/`param_specs` are bare dicts and were added this cycle. |
| `fmi.tcp_bridge.FmuTcpBridge` | evolving | **evolving** | high | 6 | 0 | 0 | 6 | 0 | Six commits this cycle, protocol version only just introduced. |
| `fmi.fmu_state.FMUState` | evolving | **evolving** | high | 2 | 0 | 3 | 2 | 0 | An opaque payload whose encoding the binary-frames branch just changed. |
| `fmi.fmu_state.serialize_fmu_state` | evolving | **evolving** | high | 3 | 0 | 3 | 2 | 1 | As above; `params: Optional[dict]`. |
| `fmi.fmu_state.deserialize_fmu_state` | evolving | **evolving** | high | 2 | 0 | 3 | 2 | 0 | As above. Its `return_params: bool` flag changes the return *type*, which is a shape no frozen signature should have — fix before promotion. |
| `fmi.package.write_fmu` | evolving | **evolving** | high | 2 | 0 | 0 | 1 | 0 | Brand new. |
| `fmi.package.build_fmu_binary` | evolving | **evolving** | high | 1 | 0 | 0 | 1 | 0 | Brand new, and its `cc`/`extra_flags` arguments are a toolchain escape hatch that will want to grow. |

## sysid

Added in 0.4.0, hardened in the audits, no release has shipped it. Bare
`dict`s everywhere — 16 across nine surfaces — because the params pytree, the
observation map and the mask are all untyped dicts. Freezing them before the
TypedDicts exist would freeze the untyped versions.

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `sysid.fit` | evolving | **evolving** | high | 12 | 0 | 14 | 4 | 2 | The most-used of the nine and the best tested, but one release old. Revisit at 0.5.0. |
| `sysid.fit_lm` | evolving | **evolving** | high | 3 | 0 | 0 | 4 | 2 | Added in the overnight run; the λ-schedule arguments are tuning surface. |
| `sysid.fim` | evolving | **evolving** | high | 5 | 0 | 0 | 4 | 2 | `noise_std` was added after the first version shipped in this same cycle. |
| `sysid.windowed_loss` | evolving | **evolving** | high | 2 | 0 | 0 | 4 | 3 | Grew `window_states`/`continuity_weight` mid-cycle. |
| `sysid.fit_multiple_shooting` | evolving | **experimental** | med | 1 | 0 | 0 | 4 | 5 | One test file, five bare dicts, fifteen-plus arguments, written in a single overnight session. `evolving` overstates what we know. |
| `sysid.init_window_states` | evolving | **evolving** | med | 2 | 0 | 0 | 4 | 1 | Helper for the above; should follow whatever it gets. |
| `sysid.observations_from_history` | evolving | **evolving** | high | 3 | 0 | 0 | 4 | 0 | The only one of the nine with a fully typed signature. A promotion candidate for 0.5.0. |
| `sysid.FitResult` | evolving | **evolving** | med | 0 | 0 | 0 | 4 | 1 | No test file names it; it is checked through its fields. Its shape is its contract, and `params: dict` is one of them. |
| `sysid.FIMReport` | evolving | **evolving** | med | 0 | 0 | 0 | 4 | 0 | As above — six fields, no direct test. |

## surrogates

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `surrogates.node.SurrogateNode` | experimental | **evolving** | med | 8 | 0 | 21 | 1 | 1 | Exported from the **top-level** `maddening.__all__` while tagged "may break in any minor release". Whatever the right level is, it is not that one. `initial_values: dict` keeps it below `stable`. |
| `surrogates.architecture.SurrogateArchitecture` | experimental | **evolving** | med | 4 | 0 | 21 | 0 | 0 | Also in the top-level `__all__`. It is an ABC, so the snapshot records `(*args, **kwargs)` and the real promise is its abstract-method set — worth writing down explicitly in whatever level it lands at. |
| `surrogates.training.trainer.SurrogateTrainer` | experimental | **experimental** | high | 8 | 1 | 20 | 0 | 0 | Training loop, optimiser and loss plumbing; it will change when optax/equinox do. |
| `surrogates.dataset.DatasetGenerator` | experimental | **experimental** | high | 8 | 1 | 15 | 0 | 0 | As above. |

## usd

| Surface | now | propose | conf | tests | ex | down | churn | dict | Why |
|---|---|---|---|---|---|---|---|---|---|
| `usd.live_stage.LiveStage` | evolving | **evolving** | high | 1 | 1 | 2 | 0 | 0 | Unchanged since May, but one test file, and the USD package's *other* entry points (`save_graph_to_usd`, `load_graph_from_usd`, `USDWriter`) carry no tag at all — it makes no sense to settle the live-stage writer while the file writer is untagged. Do the package together. |

---

## Untagged but public in practice

An export with no tag has no promise, but callers cannot see that. These were
found mechanically: a name in some `__all__` that also appears in a user
guide, in `src/maddening/examples/`, or in a `stable` signature.

The tree has 209 distinct `__all__` names across 31 modules, of which 151
carry no `@stability` tag. About fifty of those are `viz` and `cloud` plumbing that
should probably be `internal`; the ones below actually need a decision.

### Named by a `stable` signature (highest priority)

A `stable` method that takes or returns an untagged type is only half frozen.

| Surface | Where it leaks in | Propose |
|---|---|---|
| `core.SharedSolveResult` | return type of `sharded_cg` **and** `sharded_gmres`, both `stable` | **stable** |
| `core.CouplingGroup` | argument to `GraphManager.add_coupling_group`, return of `auto_couple`; also in the **top-level `__all__`** | **stable** |
| `core.BoundaryInputSpec` | return of `SimulationNode.boundary_input_spec` | **stable** |
| `core.BoundaryFluxSpec` | return of `SimulationNode.boundary_flux_spec` | **stable** |
| `compliance.UncertaintySpec` | return of `SimulationNode.uncertainty_spec` | **evolving** |
| `core.ShardingIssue` | return of `GraphManager.validate_sharding` | **evolving** |

### Public by use, no tag

| Surface | Evidence | Propose |
|---|---|---|
| `serialization.to_dict` / `from_dict` | the config format; documented; `GraphManager.to_dict`/`from_dict` already in the `stable` snapshot | **stable** (both) |
| `core.save_state` / `load_state` | documented in a user guide; `GraphManager.save_state`/`load_state` already in the `stable` snapshot | **stable** (both) |
| `usd.save_graph_to_usd` / `load_graph_from_usd` / `USDWriter` / `register_node_class` | documented; the subject of open question 4 | **evolving** |
| `maddening.HistoryLogger` | top-level `__all__` | **evolving** |
| `maddening.AdaptiveConfig` | top-level `__all__`; the adaptive-timestep controller | **evolving** |
| `maddening.CloudSession` / `CloudConfig` | top-level `__all__`; `CloudSession` documented | **evolving** |
| `core.coupling.mapping.Mapping` | the protocol `EdgeSpec.mapping` accepts; `EdgeSpec` is `stable` and annotates it `Optional[Any]` | **evolving**, and narrow the `EdgeSpec` annotation to it |
| `core.transforms.register_transform` / `resolve_transform` | used by examples; enforced by `scripts/check_transforms.py`; required for USD round trips | **evolving** |
| `fmi.DirectionalDerivativeKind` | the `kind=` argument of `get_directional_derivative`, which this proposal wants `stable` | **stable** |
| `fmi.ModelDescription` / `FMIVariable` | returned by `build_model_description`, consumed by `write_fmu` | **evolving** |
| `core.simulation.profiler.ProfileReport` | returned by the tagged `profile_graph`; not in any `__all__`, but it is what the function hands back | **evolving** |
| `core.detect_cycles` / `topological_sort` / `find_strongly_connected_components` | in `core.__all__`; graph-algorithm helpers | **internal** unless someone downstream uses them |
| `maddening.api.server` (`create_app`, the REST routes, the response schemas) | the entire HTTP surface; MICROROBOTICA drives it; open question 1 is about its schema | **evolving**, and the *schema* needs a versioning story of its own |

The REST API is the biggest gap. Thirty-odd routes, a documented wire format,
a downstream IDE driving it, and not one `@stability` tag anywhere in
`maddening/api/server.py`. A tag on `create_app` would be cosmetic; what the
freeze actually needs to decide is whether the *response schemas* are frozen,
and question 1 below is the first test of that.

---

## The open questions

Six decisions the freeze has to make. Each was raised by this cycle's audits
or property-test branches and is recorded in `TODO.md` under "Decisions the
freeze must make". For each: what happens today, the options, what I would do,
and what it costs to change the decision after 1.0.

### 1. What the REST API returns when a simulation diverges

**Today.** Once any state leaf is `inf` or `NaN` — reachable with nothing but
valid calls, e.g. a stiffness the explicit integrator cannot hold at the
configured timestep — `GET /graph/state`, `GET /graph/state/{node}`,
`POST /sim/step` and `POST /sim/run` all raise **inside the response
encoder**: `_jax_to_python` emits Python floats and Starlette's `JSONResponse`
serialises with `allow_nan=False`. The caller sees a bare 500 with no
indication of which node diverged, and only `POST /sim/reset` recovers the
server. Pinned as a strict xfail:
`tests/property/test_stateful_api.py::test_state_endpoints_stay_below_500_when_the_simulation_diverges`.

**Options.**

| Option | Consequence |
|---|---|
| (a) **409 naming the diverged node**, body `{"detail": ..., "diverged": ["spring.position"]}` | Clear and actionable. Breaks any client that treats non-2xx as fatal, and makes "read the state" impossible — a debugging client cannot see *what* the values became. |
| (b) **200 with a flag and JSON `null` for non-finite leaves**: `{"state": {...}, "finite": false, "diverged": [...]}` | The state stays readable and the client learns something is wrong. `null` is lossy: it cannot distinguish `+inf`, `-inf` and `NaN`. Every client must learn to handle `null` where it expected a number. |
| (c) **200 with the JSON-5-style strings** `"NaN"`, `"Infinity"`, `"-Infinity"` | Lossless and the encoder change is one line (`allow_nan=True` produces bare `NaN` tokens, which are invalid JSON — so strings, not tokens). Clients get a string where they expected a number, which is the same typing problem as (b) with more information. |
| (d) Leave it | A 500 is indistinguishable from a server bug, and the server stays unreadable until reset. Not defensible in something MICROROBOTICA drives. |

**Recommendation: (b), with (c)'s information folded in** — `null` in the
`state` object for the wire-safe value, plus a `"non_finite"` map
`{"spring.position": "inf"}` naming each offending leaf and what it was. That
keeps the state document numeric-or-null (one rule for clients) and loses
nothing. Endpoints that *advance* the simulation additionally set
`"diverged": true` so a polling client can stop without diffing state.

**Cost of changing it after 1.0.** High and asymmetric. Going from 500 to a
2xx body is additive-ish (nobody was consuming a 500). Going the other way, or
changing the shape of the flag once clients branch on it, is a breaking wire
change with no deprecation mechanism — HTTP has no `DeprecationWarning`. If
the schema is not versioned, add the version now: this question is the reason
to do it.

### 2. `AdaptiveNode`'s dtype under `jax_enable_x64`

**Today.** `AdaptiveNode` resolves the canonical float in `__init__`, so its
state dtype follows `jax_enable_x64`. Every other node hard-codes `float32` in
`initial_state()`. Under x64, an edge out of an adaptive node's `float64` `c`
promotes the downstream node's `float32` state and `run_scan`'s carry types
stop matching: *"scan body function carry input and carry output must have
equal types"*. Pinned as a strict xfail:
`tests/property/test_adaptive_invariants.py::test_an_adaptive_node_under_x64_cannot_drive_a_float32_node`.

**Options.**

| Option | Consequence |
|---|---|
| (a) **Nodes stop pinning float32** — every `initial_state()` uses the canonical float | Principled: x64 means x64. Touches every built-in node and every downstream node (83 files subclass `SimulationNode`), silently doubles memory and changes every x64 user's numerics. The largest blast radius of the three. |
| (b) **`AdaptiveNode` stops following the flag** and pins float32 like everything else | Smallest change, one node, no downstream effect. Costs the adaptive node the one thing x64 is for: the frozen-set solve is exactly where extra precision pays, and `MADD-ANO-003` is a gradient-accuracy anomaly. |
| (c) **The graph coerces each update output back to its carry dtype** | Fixes the symptom everywhere, including mixed-precision graphs nobody has built yet. Adds a cast on every node output on every step (cheap, but it is in the hot loop), and silently down-casting a node's deliberate float64 is its own trap. |

**Recommendation: (b) now, (c) as the 1.x design.** Pin `AdaptiveNode` to
float32 for 0.4.0 — it is one node, no release has shipped its behaviour, and
it removes a strict xfail — and add an explicit, opt-in `dtype=` argument to
`AdaptiveNode.__init__` (keyword-only, default `jnp.float32`) so the precision
is available to whoever needs it without a global flag deciding it. Treat (c)
as the real fix and schedule it with the mixed-precision work, where a carry
dtype becomes a declared property of a node rather than an accident.

**Cost after 1.0.** (b) → (a) later is a numerics change to every node: a
major bump, and a painful one. (b) → (c) later is additive if the coercion
only fires where types already mismatch (today: an error). Choosing (b) keeps
both doors open; choosing (a) closes them.

### 3. `GraphManager.to_dict` has no coupling-group field

**Today.** The USD writer serialises coupling groups
(`/Simulation/coupling_groups/cg0`, read back via `gm.add_coupling_group`);
`GraphManager.to_dict` writes `nodes`, `param_specs`, `edges` and
`external_inputs`, and nothing else. A graph with a coupling group therefore
survives a USD round trip and **silently loses the group** through a config
round trip. It is a format gap, not a bug: nothing raises.

**Options.**

| Option | Consequence |
|---|---|
| (a) **Add `coupling_groups` to `to_dict`/`from_dict`** mirroring the USD fields | Additive on write (new key), additive on read (`config.get("coupling_groups", [])`). Old configs keep loading; new configs fail on an old reader with a `KeyError`-free silent drop — the same failure we have today, so no worse. |
| (b) **Warn on `to_dict` when a group would be dropped**, like the mapping-weights warning already does | Cheap and honest, and does not settle the format. Leaves the round trip broken. |
| (c) **Refuse** (`ValueError`) when a group is present and `strict=True` | Consistent with `strict_mappings`, but turns a working call into an error for anyone relying on the partial config today. |
| (d) Defer to 0.5.0 | The config format is the thing the freeze is supposed to settle. Deferring means 1.0 ships a config format that cannot express a first-class graph feature. |

**Recommendation: (a), in 0.4.0, plus (b) as the belt** — write the field, read
it back, and warn when `strict_mappings=False` is used to write a graph whose
group cannot be reconstructed. It is additive, the USD writer already defines
the field set to copy, and the round-trip property tests in
`tests/property/test_round_trips.py` will cover it the moment it exists.

**Cost after 1.0.** Adding the key later is still additive, so this is the
cheapest of the six to defer. The real cost of deferring is reputational: a
`stable` `to_dict` that drops data is a bug report waiting to be filed against
the frozen surface.

### 4. `save_graph_to_usd` does not warn about drifted mapping weights

**Today.** `GraphManager.to_dict` calls `_warn_about_unsaved_mapping_weights()`
and emits a `UserWarning` naming the edge when the live mapping weights differ
from what the `MappingSpec` rebuilds, pointing the caller at `save_state()`.
`save_graph_to_usd` calls `check_mapping_serialisable` for the same edges —
so it validates the spec — and then writes `mapping.describe()` **without the
warning**. Identical data loss, one path silent.

**Options.**

| Option | Consequence |
|---|---|
| (a) **Call the same warning from the USD writer** | Three lines. Makes two paths consistent. A caller who currently saves trained mapping weights to USD in a `-W error` test suite would start failing — correctly, because they *are* losing data. |
| (b) **Move the warning into `check_mapping_serialisable`** so every serialisation path gets it for free | Better: one place, and the next writer (FMU, a future exporter) inherits it. `check_mapping_serialisable` is currently a pure validator, so this gives it a side effect. |
| (c) Leave it | Two paths, two behaviours, no reason. |

**Recommendation: (b).** Put the drift check where the serialisability check
already is, and have both `to_dict` and `save_graph_to_usd` get it from one
call. Rename it if the side effect makes the name wrong
(`check_mapping_serialisable` → keep, and document that it warns). This is a
bug fix, not a policy decision — the only reason it is on this list is that it
changes the observable behaviour of a path the freeze is about to settle.

**Cost after 1.0.** Low. Adding a warning is not breaking under the policy
(it is not a result change), so this can land any time. It should still land
in 0.4.0 because the USD path is one of the two ways a calibrated graph leaves
the process.

### 5. `ift_linear_solve`: expose the solver knobs, own the error type

**Today.** The wrapper hard-codes `restart = min(N, 50)` and
`max_steps = max(4 * restart, 100)`, and leaves `stagnation_iters` at the
lineax default. A non-convergent solve surfaces
`equinox._errors._EquinoxRuntimeError` eagerly and `jax.errors.JaxRuntimeError`
under `jit`/`scan` — a *private* third-party type reachable from a MADDENING
function — and the remedy lineax prints ("try increasing `stagnation_iters` or
`restart`") is not reachable through the signature. The function is
`EXPERIMENTAL` precisely because of this.

**Options.**

| Option | Consequence |
|---|---|
| (a) **Add `restart` / `max_steps` / `stagnation_iters` as keyword-only with today's defaults** | Purely additive; the clamp stays the default, so no existing call changes behaviour. Exposes three lineax concepts in a MADDENING signature, which ties the signature to lineax's vocabulary. |
| (b) **Wrap the failure in a MADDENING error** (`IftSolveError`, carrying the residual, the iteration count and the suggested knob) | Not additive: anyone catching `JaxRuntimeError` today stops catching it. Under `jit` the error arrives through JAX's own machinery, so the wrapping has to happen at both the eager and the traced boundary — non-trivial, and easy to get wrong for a `scan`-internal failure. |
| (c) **Both**, then promote to `evolving` | The only combination that makes the `Raises` section writable: the docstring can name a type we own and a knob the caller can turn. |
| (d) Leave it `experimental` and revisit at 0.5.0 | Free. Leaves `AdaptiveNode` — which is built on this function — unable to rise above `experimental` either, since its failure mode is this one. |

**Recommendation: (c), and treat (a) as the part that must land in 0.4.0.**
The three keywords are additive and cost a day; do them now. The error type is
the harder half: scope it as "eager path wrapped in 0.4.0, traced path in
0.5.0", document the `jit` gap honestly in the `Raises` section, and keep the
function `experimental` until the traced path is covered. That is the
recommendation *against* freezing it this cycle — which is what the table
above says.

**Cost after 1.0.** (a) stays additive forever, so it is cheap either way.
(b) is a major-bump change once the function is `stable`: an exception type in
a documented `Raises` section is part of the contract. That asymmetry is the
argument for keeping it `experimental` now rather than freezing a signature
that leaks a private `equinox` class.

### 6. Should the `AdaptiveNode` hooks take `boundary_inputs` and `dt`?

**Today.** `AdaptiveNode.update` does `del boundary_inputs, dt`, and the three
hooks — `compute_active_set`, `solve_frozen`, `objective` — take only
`(state, mask, params)` / `(state, params)`. An `AdaptiveNode` can therefore
only ever be an edge **source**. To consume an edge's input, or to be
time-dependent at all, a subclass must override `update` and re-implement the
`stop_gradient` wrap, both shape validations and `_solve_and_pack`. The same
limitation makes an adaptive node inert to its partner's interface values
inside a coupling group.

**Options.**

| Option | Consequence |
|---|---|
| (a) **Add `boundary_inputs` and `dt` as keyword-only with defaults** to the three hooks | Existing overrides keep binding (Python matches by name, and a subclass that omits them simply never sees them — but only if the base calls them by keyword). Three hook signatures change, which is exactly the kind of change the freeze exists to stop happening later. |
| (b) **Pass a single `context` object** carrying `boundary_inputs`, `dt` and whatever comes next | One signature change, ever. Adds a concept the rest of the node API does not have — `SimulationNode.update` takes them positionally — so the adaptive node would be the odd one out. |
| (c) **Leave the hooks alone**; document that an `AdaptiveNode` is an edge source only | Honest, free, and permanent: an adaptive node can never be a coupling partner, which is most of the point of having one in a multi-physics graph. |
| (d) Do (a) **and** give `update` the `boundary_inputs` it currently deletes | Completes the node contract. Biggest change, and the one that makes the base class actually usable at the receiving end of an edge. |

**Recommendation: (d), before 0.4.0 ships, or keep the node `experimental`.**
The hooks are three weeks old and have two subclasses, both test fixtures
(`tests/nodes/adaptive/_toys.py`). Changing them costs nothing today. The
alternative — freezing a base class that structurally cannot sit at the
receiving end of an edge — is the kind of decision that gets discovered by the
first person who tries, in a release where fixing it needs a deprecation
cycle on an abstract method. Note that adding an argument to an
`@abstractmethod` is listed as breaking in the deprecation policy for a
reason: every downstream subclass stops binding.

If (d) is too much for 0.4.0, the fallback is not (c) — it is demoting
`AdaptiveNode` to `experimental` (which the table above recommends anyway) and
doing (d) in 0.5.0 while the level still allows it.

---

## What this proposal does not decide

- **Whether the REST response schema is versioned at all.** Question 1 needs
  it, and nothing in the tree provides it. That is a design task, not a tag.
- **The `ppermute` default.** `ShardedUnstructuredNode(exchange=)` and
  `exchange_unstructured(method=)` both default to `"all_to_all"`, and the
  NCCL session exists to decide whether that is right. Changing a default
  after the freeze is breaking; the session is the blocker, and it needs
  hardware and explicit consent.
- **The `viz` and `cloud` packages.** Fifty-odd untagged `__all__` names
  between them, no user-guide coverage, several used by examples. They need a sweep of
  their own; my guess is that most should be `internal`, but that is a guess.
- **Whether `dict` → `TypedDict` is breaking.** The pre-1.0 reading at the top
  of this document says no, *before* 1.0. After 1.0 it narrows a parameter
  type, which the policy calls breaking. Either every `stable` surface gets
  its TypedDicts before 1.0, or 1.0 ships bare `dict`s forever. The branch
  order (freeze, then typing phase 2) assumes the former; someone should
  confirm that assumption holds for all 21 proposed surfaces.
