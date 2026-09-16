# MADDENING audit round 2 — branch `feat/graph-params-sysid` (PR #9), working tree as of 2026-09-16

Scratch dir: `/home/nick/.claude/jobs/d22809a4/tmp/audit2/`
Reproducers (run from the repo root: `JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.venv/bin/python -m pytest <file> -q -p no:cacheprovider`):
- `test_batch1.py` — interface correction vs params, checkpoint `_meta` / mapping weights, half-precision and zero-size leaves in a coupled group
- `test_batch2.py` — REST entry points, `to_dict`/`from_dict`, `remove_edge`/`remove_node` cleanup, retrace with Python-float leaves
- `test_batch3.py` — REST JSON-int minimised, `from_dict` after `remove_edge`, Hypothesis partial-params agreement, ParamSpec strict interior, subcycled IFT with wide ints, checkpoint param shape mismatch
- `test_x64_probe.py` — run with `JAX_ENABLE_X64=1`: float64 states in an IFT group, int64/uint64 leaves, grad through scan

## Findings (ranked by severity)

### 1. HIGH — `PUT /graph/params/{node}` with a JSON integer for a live float leaf turns the constructor param into a Python `int`; the key then vanishes from `params_pytree()` and the next trace / `check_params` / recompile break
File: `src/maddening/api/server.py:418-421` (live path writes `node.params[key] = value` with the *raw* JSON value, while `live[key]` gets the float32-coerced array); interacts with `src/maddening/core/node.py:198-200` (`params_pytree` skips `int`).
Invariant broken: "a live PUT takes effect on the next step without recompiling" and "`gm.params` and `node.params` describe the same leaf set". A slider at an integer position sends `40`, not `40.0`.
Reproducer: `test_batch3.py::test_put_json_int_for_live_float_then_first_step` — PUT `{"stiffness": 40}` → 200, `type(node.params["stiffness"]) is int`, then `gm.step()` raises
`ValueError: params['nodes']['s'] has unknown key(s) ['stiffness']; SpringDamperNode.params_pytree() exposes ['damping', 'initial_position', 'initial_velocity', 'mass', 'rest_length']` (graph_manager.py:1795 via `_resolve_params`).
`::test_put_json_int_for_live_float_after_a_step_breaks_check_and_recompile` — with the step already traced the step keeps working (cache hit) but `gm.check_params()` raises the same error, and a recompile drops the live `stiffness` leaf with the "no longer fits" RuntimeWarning (the node then runs with the *int* 40 baked in; the leaf is no longer calibratable).
The same corruption reaches `to_dict()`/USD (`effective_node_params` starts from `node.params`), so a saved config re-creates the node with an `int` constant.
Fix: in the write loop store a value of the constructor's Python type, e.g. `node.params[key] = np.asarray(staged[key]).tolist()` (or `float(value)` for scalars), never the raw JSON value; optionally make `params_pytree` promote a Python `int` for a key that was float.

### 2. HIGH — `compute_interface_correction` still reads `self.params`; coupled HeatNode interface cells are corrected with the constructor diffusivity (round-1 #1 incomplete)
Files: `src/maddening/nodes/heat.py:514-560` (`alpha = self.params["thermal_diffusivity"]`, `n = self.params["n_cells"]`); call sites `src/maddening/core/graph_manager.py:611-671` (`_apply_interface_overrides`) invoked at :916 (subcycling), :958 (Gauss-Seidel), :1005 (Jacobi) with no params; `src/maddening/core/simulation/hybrid_node.py:100` forwards without params.
Invariant broken: "an explicit params pytree changes node constants for this step". In any coupling group where a HeatNode's `left_temperature`/`right_temperature` comes from an edge, the interface DOF override recomputes `T[0]`/`T[-1]` with the constructor alpha; interior cells use the injected alpha, so the wrong interface value diffuses inward on later steps.
Reproducer: `test_batch1.py::test_interface_correction_uses_injected_diffusivity[ift|fori]` — rods built with alpha=0.01, `gm.params` alpha=1.0, one step; interior cells match the alpha=1.0 reference, interface cell `rod_a.T[-1]` = 99.9001 (the alpha=0.01 value) vs 90.90909 expected. Both solvers.
Not measured: the correction's dependence on alpha is invisible to `jax.grad` by construction (an FD gradient test at rtol 5e-2 passed because the state path dominates over 3 steps).
Fix: give `compute_interface_correction(pre_state, boundary_inputs, dt, *, params=None)` the same contract; add a `_NodeSpec.correction_accepts_params` probe and pass `_np(nn)` from the three `_apply_interface_overrides` call sites (and `HybridNode`).

### 3. MEDIUM — `load_state` before the first compile restores state, then the compile it triggers resets `_meta` (multirate `step_count` → 0; predictor/IQN/diagnostic meta re-seeded)
Files: `src/maddening/core/simulation/checkpoint.py:196-218` (state and `_meta` applied, *then* `graph_manager.compile()`), `src/maddening/core/graph_manager.py:2566-2568` (`self._state[_META_KEY] = {"step_count": 0}` unconditionally for multirate), :2572-2650 (coupling meta re-seeded).
Invariant broken: "compile-then-load and load-then-compile are the same operation" (the round-1 fix made load-before-compile keep params but not meta).
Reproducer: `test_batch1.py::test_load_state_before_compile_keeps_multirate_step_count` — save at step_count=4, fresh graph, `load_state` before compile, `run(3)` → `step_count == 3`, expected 7 (`AssertionError: {'step_count': Array(3, dtype=int32)}`); the multirate schedule is shifted relative to the compile-first graph.
Also: the compile is only triggered when the checkpoint has params (`if param_keys and ...`), so a graph without params nodes loaded before compile loses its `_meta` at the next `step()` by the same mechanism (pre-existing on main; not reproduced separately).
Fix: compile at the *top* of `load_state` whenever `_dirty or _compiled_step is None` (regardless of `param_keys`), then apply state/meta/params.

### 4. MEDIUM — checkpoints do not carry `params["mappings"]`; calibrated interface-mapping weights are lost on resume
File: `src/maddening/core/simulation/checkpoint.py:104-107` (only `params["nodes"]` is written), :156-159 / :220-224 (only nodes read back).
Invariant broken: "a calibrated graph restores with the values it was calibrated to" (docstring) — mapping weights are trainable leaves (`trainable_mask`, `sysid.fit`) but are not checkpointed.
Reproducer: `test_batch1.py::test_checkpoint_keeps_calibrated_mapping_weights` — `H` set to 3I, save, fresh graph, load → `H == I`.
Fix: write `_params_mappings/<edge.key>/<leaf>` (edge keys contain `->`, `.`, `#` but no `/` unless node names do) and restore into `gm.params["mappings"]` with the same shape/dtype check as nodes.

### 5. MEDIUM — `PUT /graph/params/{node}` on a not-yet-compiled graph bypasses every live check (bounds, finiteness, dtype, shape)
File: `src/maddening/api/server.py:369-370` (`live = self.gm.params.get("nodes", {}).get(node_name) or {}` is empty before the first compile, so every key takes the structural path at :418-421).
Invariant broken: entry-point agreement — the same request is a 400 on a compiled graph and a 200 on the same graph one `compile()` earlier.
Reproducer: `test_batch2.py::test_put_out_of_bounds_before_first_compile_is_400` — `{"stiffness": -5.0}` → 400 when compiled, 200 when not; after compile `gm.check_params()` raises (`stiffness=-5.0 below bound 0.0`). A `SimulationServer` built on a freshly constructed graph (the normal case before the runner starts) is in exactly this state.
Fix: when the node `accepts_params()` and the graph is not compiled, validate against `node.params_pytree()` / `node.param_specs()` in the same loop, or compile first.

### 6. MEDIUM — `GET /graph/params/{node}` returns constructor values, not the live pytree; disagrees with the `PUT` response and with `sysid.fit`
File: `src/maddening/api/server.py:351-356` (`return _jax_to_python(node.params)`), vs :422 (`PUT` returns `{**node.params, **live}`).
Invariant broken: what the API reads back is what the step uses. After `gm.params[...] = 300` (what a fit or a checkpoint restore does) the step uses 300 and GET reports 30.
Reproducer: `test_batch2.py::test_get_params_reflects_live_values` — `assert 30.0 == 300.0`. (`::test_get_params_agrees_with_put_response` passes only because the PUT path also writes `node.params`.)
Fix: return the same merged view as PUT.

### 7. MEDIUM — under `jax_enable_x64` with float64 node states, an IFT/diagnostics group seeds the `_meta` residual as float32 but the step writes float64: `run_scan` straight after compile raises; `step` retraces once
Files: `src/maddening/core/graph_manager.py:2591` (`meta[f"coupling_{key}_residual"] = jnp.array(0.0, dtype=jnp.float32)`); the residual is computed in the state dtype (`acceleration.py:60-66`); `sysid.windowed_loss` seeds meta with `jnp.zeros_like(meta0)` (`sysid.py:172-179`) and inherits the same dtype.
Reproducer (`JAX_ENABLE_X64=1`): `test_x64_probe.py::test_float64_state_ift_step_does_not_retrace_and_run_scan_works` — cache size 2 after two steps (retrace on the dtype flip). `run_scan` directly after compile on the same graph (first version of the probe): `TypeError: scan body function carry input and carry output must have equal types ... state['_meta']['coupling_a+b_residual'] has type float32[] but the corresponding output carry component has type float64[]`. Float32 states are unaffected (inline probe).
Fix: seed the residual with the dtype of the group's flattened float fields, or cast the residual to the seed dtype where it is written.

### 8. LOW — `remove_edge` leaves ParamSpec overrides for ordinal keys (`...#1`) behind; `to_dict`/`from_dict` round trip then fails (round-1 #11 incomplete)
File: `src/maddening/core/graph_manager.py:2170` (`self._param_spec_overrides.pop(edge.key, None)` — only the ordinal-0 key; the mapping slots at :2171-2173 are cleaned by base key).
Reproducer: `test_batch3.py::test_from_dict_after_remove_edge_with_ordinal_override` → `KeyError: "unknown node 'a.v->b.inp#1'"` from `from_dict` → `set_param_spec` (graph_manager.py:1893). `test_batch2.py::test_stale_ordinal_spec_override_after_remove_edge` shows the stale entry in `param_spec_overrides()`.
Fix: pop every override key with `key.split("#")[0] == edge.key`.

### 9. LOW — `load_state` restores a params leaf with a different shape/dtype unchecked; the graph then runs with the wrong shape
File: `src/maddening/core/simulation/checkpoint.py:220-224` (only key membership is checked).
Reproducer: `test_batch3.py::test_load_state_rejects_param_shape_mismatch` — a `(3,)` `stiffness` is accepted (`DID NOT RAISE`); inline: the scalar spring's `position` becomes `[1. 1. 1.]` of shape `(3,)` after one step, silently.
Fix: apply the same shape/dtype check as `_merge_live_params` and raise `ValueError` (state fields already get this at :176-184).

### 10. LOW — under `jax_enable_x64`, a Python-float live leaf (`gm.params["nodes"]["s"]["stiffness"] = 300.0`) is dropped at the next recompile and reverts to the constructor value
File: `src/maddening/core/graph_manager.py:1720-1722` (`_merge_live_params`: `jnp.asarray(300.0)` is float64 under x64, the snapshot leaf is float32 → "shape/dtype changed" → dropped with a RuntimeWarning).
Reproducer: inline probe — after a dirty recompile `stiffness == 30.0`, warning `compile() dropped live gm.params leaves ... ["nodes['s']['stiffness'] (shape/dtype changed)"]`. Without x64 the same assignment is kept.
Fix: when shapes match, `v = jnp.asarray(value, dtype=base.dtype)` instead of comparing dtypes; only drop on shape mismatch.

### 11. LOW (performance) — params leaves given as Python floats retrace the jitted step once per weak/strong-type flip
File: `src/maddening/core/graph_manager.py:174-190` (`_strong_typed` is applied to the state at :2666 but not to `params` in `_params_or_default` / `_merge_live_params`).
Reproducer: `test_batch2.py::test_python_float_params_leaf_does_not_retrace_the_step` (partial tree with a Python float → cache size 1 → 2) and `::test_user_assigned_python_float_in_gm_params_does_not_retrace` (assignment to `gm.params` → 1 → 2).
Fix: `_strong_typed` the completed tree in `_params_or_default` and the merged tree in `_merge_live_params`.

### 12. LOW — `ParamSpec.to_constrained` (logit) can still land exactly on a bound for a 2-ulp-wide interval where an interior float exists
File: `src/maddening/core/params.py:148-152` (`m = min(4*eps*max(...), 0.25*(hi-lo))`; `hi - m` rounds back to `hi`).
Reproducer: `test_batch3.py::test_logit_constrain_strict_interior_signed_bounds` — Hypothesis: `lo=262144.0, width=0.0625, u=2.0` → `check` raises `param=262144.0625 above bound 262144.0625`; `262144.03125` is representable and strictly inside. (The other falsifying example, `lo=524288.0, width=0.0625`, has no interior float32 at all — the case the code comment already concedes.)
Fix: compute the clip limits as `nextafter(lo, +inf)` / `nextafter(hi, -inf)` in the leaf dtype (max'd with the relative margin), so they are representable by construction.

## Round-1 fixes re-checked
1. Flux params: complete for `compute_boundary_fluxes` (three coupled call sites + non-coupled path + `HybridNode` forward; `verify_node` FAILs a flux producer without params). Incomplete for `compute_interface_correction` (finding 2). `derivatives`/`implicit_residual` are not called by the graph or integrators (only `_apply_interface_overrides` is) — no graph-level effect.
2. Recompile keeps live params: complete for add_edge/add_external_input/dirty recompiles; remove+re-add uses the new constructor value (`test_batch2.py::test_readd_node_after_remove_uses_new_constructor_value`). `load_state` before compile keeps params but now clobbers `_meta` (finding 3); x64 Python-float leaves dropped (finding 10).
3. Non-float leaves in coupled groups: complete. uint32 `0xDEADBEEF` and int32 `2**30+12345` exact through a subcycled IFT group with linear and constant interpolation (`test_batch3.py::test_wide_int_leaves_survive_subcycled_ift_group[*]`); int64/uint64 (`2**63-5`) exact for ift and fori under x64, grad through a 3-step scan finite (`test_x64_probe.py`); float16/bfloat16 leaves keep dtype and value under ift and fori (`test_batch1.py::test_half_precision_leaf_in_group_keeps_dtype_and_value[*]`); zero-size leaf in the group OK (`::test_zero_size_leaf_in_group[*]`).
4. Ordinal mapping slots: complete for `params["mappings"]`; `remove_node` also drops ordinal slots and overrides (`test_batch2.py::test_remove_node_drops_ordinal_mapping_slots_and_overrides`); `remove_edge` misses ordinal overrides (finding 8).
5. Log/logit clamp: log holds for lo in [-1e6, 1e6], u in [-120, 120], 200 examples (`test_batch3.py::test_log_constrain_strict_interior_signed_bounds`); logit fails for ~2-ulp intervals (finding 12).
6. Partial params: complete — `run_scan(partial)`, `step(partial)`, the raw compiled step with the completed tree, and `trainable_mask(partial)` agree for random subsets/values with a modified live leaf, 25 examples (`test_batch3.py::test_partial_params_agree_across_entry_points`).
7. REST validate-then-write: complete for string/null/list/NaN; new holes for JSON ints (finding 1) and pre-compile (finding 5); GET stale (finding 6).
8. Sharded grid-shape heuristic: not re-run (needs 4 host devices; round-1 regression tests exist). The residual ambiguity the new docstring concedes (every state field carries component dims) remains.
9. `fim`/`fit_lm` noise_std pytree: read only — `_inverse_noise_std` keys the scalar branch on `numbers.Real` / 0-d array; a list would fall to `tree.map` and fail, acceptable.
10. NaN in `check`: read only — rejects non-finite inexact values; integer leaves correctly skip the check.
11. Stale overrides: `remove_node` complete; `remove_edge` incomplete (finding 8).
12. FMI advertised open bounds: not re-run (round-1 tests exist). See suspicious item on float64 dtype.
13/14. `from_dict(bounds=null)`, int identity clip dtype: read only, consistent.
15. Profiler one-iteration variant: not re-run (round-1 regression test exists).

## Probed and held
- Interior cells of a coupled HeatNode pair follow injected `thermal_diffusivity` for ift and fori (only the interface override does not) — `test_batch1.py::test_interface_correction_uses_injected_diffusivity` (first assertion).
- IFT gradient wrt diffusivity at an interface cell over 3 steps matches central FD at rtol 5e-2 — `test_batch1.py::test_gradient_wrt_diffusivity_at_interface_cell_matches_fd`.
- Predictor history restored by `load_state` before compile does not change a fori(4)+linear-predictor trajectory within 1e-6 — `test_batch1.py::test_load_state_before_compile_keeps_predictor_history` (passes; the re-seeded history comes from the loaded state, so only multirate `step_count` is observable).
- Half-precision and zero-size state leaves in a coupled group (ift/fori) — `test_batch1.py` (4 + 2 cases).
- Wide int32/uint32 leaves through subcycled IFT (linear/constant) — `test_batch3.py::test_wide_int_leaves_survive_subcycled_ift_group`.
- int64/uint64 leaves with float32 states under x64, ift and fori, exact; grad through scan finite — `test_x64_probe.py` (3 cases).
- Float32-state IFT group under x64: no retrace, `run_scan` fine (inline probe).
- Partial params agree across `run_scan` / `step` / raw compiled step / `trainable_mask` (Hypothesis, 25 examples) — `test_batch3.py::test_partial_params_agree_across_entry_points`.
- Log-transform clamp strictly inside and re-invertible for signed lo up to 1e6 (Hypothesis, 200 examples) — `test_batch3.py::test_log_constrain_strict_interior_signed_bounds`.
- `remove_node` then `add_node` with a different constructor value: compile uses the new value, no stale live leaf — `test_batch2.py::test_readd_node_after_remove_uses_new_constructor_value`.
- `remove_node` drops ordinal mapping slots and their overrides — `test_batch2.py::test_remove_node_drops_ordinal_mapping_slots_and_overrides`.
- Unregistered lambda transform: `to_dict`/`from_dict` is either loud or faithful (no silent drop) — `test_batch2.py::test_unregistered_lambda_transform_round_trip_is_loud_or_faithful`.
- `from_dict` refuses mapped edges with an explicit message (graph_manager.py:3975-3982) — documented limitation, read only.
- REST PUT response and GET agree when the PUT path itself wrote `node.params` — `test_batch2.py::test_get_params_agrees_with_put_response`.

## Suspicious but unconfirmed
- `HybridNode.update` takes no `params`, so `HybridNode(SpringDamperNode(...))` is on the legacy contract: `nodes_without_params() == ['s']`, no `gm.params["nodes"]` entry, `verify_node` skips the params checks, yet `flux_accepts_params` is True (inline probe). Not a crash; the docstring calls it a "drop-in replacement" and only a `logger.info` says the constants are baked. Either forward `params` to `physics_node.update` (with `accepts_params`/`params_pytree`/`param_specs` delegation) or document.
- `_advertised_bound` (model_description.py:287-318) computes `nextafter` in the *declared* dtype; a leaf declared `float64` with `jax_enable_x64` off is coerced to float32 by the sidecar, where `nextafter(lo, inf)` in float64 rounds back to `lo`, so the advertised `min` would again be rejected. Not exercised (no float64 leaves without x64 in the built-ins).
- `ShardedStencilNode` / `ShardedUnstructuredNode` have no `compute_boundary_fluxes` override, so a sharded HeatNode cannot feed a flux edge (pre-existing, not this branch).
- `add_edge`'s ordinal is derived from `e.key.split("#")[0]`; a node name containing `#` would miscount. Not exercised.
- `sysid.windowed_loss` requires a complete pytree (the completeness error names `gm.step`/`run_scan` for partial trees) while `fit`/`fit_lm`/`fim` complete it via `_params_or_default`; a user-written `residual_fn` calling `windowed_loss(gm, ..., params=partial)` inside `fim` therefore raises. Consistent with the error text; noting the asymmetry only.
