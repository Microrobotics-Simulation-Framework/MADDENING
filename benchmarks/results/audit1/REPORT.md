# MADDENING audit — branch `feat/graph-params-sysid` (PR #9)

Scratch dir: `/home/nick/.claude/jobs/d22809a4/tmp/audit1/`
Reproducers (run from repo root with the shared venv, `JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.venv/bin/python -m pytest <file> -q -p no:cacheprovider`):
- `test_repro_core.py` — params / checkpoint / flux edges / mappings / coupling dtype
- `test_repro_api.py` — `PUT /graph/params/{node}`
- `test_repro_props.py` — Hypothesis property tests (ParamSpec maps, graph structure, run_scan, reset_state, fit, mappings)
- `test_repro_sharded.py` — sharded boundary-input heuristic (sets XLA_FLAGS for 4 devices itself)
- `test_repro_misc.py` — sysid on multirate graphs, IFT gradient vs finite differences, adaptive + params (all pass)

## Findings (ranked by severity)

### 1. HIGH — compute_boundary_fluxes reads self.params; flux edges ignore the graph parameter pytree (value and gradient)
Files: src/maddening/nodes/spring.py:196-204, src/maddening/nodes/heat.py:574-586; call sites graph_manager.py:86-93 (_node_update passes params to update only), :2846 (_resolve_and_update_node), :923-944 (one_pass_gs), :950-956 (one_pass_jacobi) — all call compute_boundary_fluxes(state, bi, dt) with no params.
Invariant broken: "an explicit params pytree changes node constants for this step" / "jax.grad reaches every trainable leaf". A SpringDamperNode with stiffness set through gm.params (or sysid.fit, FMI set_params, PUT /graph/params) integrates itself with the new k, but the spring_force delivered over a flux edge is still computed with the constructor k and c. Same for HeatNode.left_heat_flux/right_heat_flux and thermal_diffusivity.
Reproducer: test_repro_core.py::test_spring_force_flux_ignores_graph_params — delivered force/dt = -23.0999991 (== constructor-k value), not the injected-k value. ::test_gradient_of_flux_consumer_wrt_stiffness_is_only_the_state_path — d sink.x/dk = +0.00023 (only the O(dt^2) state path) vs -0.00997 expected. ::test_heat_flux_ignores_graph_params — delivered -179.35 (alpha=0.1) vs -1793.5 (alpha=1.0 injected).
Root cause: the params contract was added to update() only; compute_boundary_fluxes / derivatives / implicit_residual still read self.params. verify_node's params_effective only exercises update.
Fix: give compute_boundary_fluxes (and derivatives, implicit_residual) an optional params= keyword mirroring update, and pass _np(nn) / node_params.nodes.get(name) at every call site (_resolve_and_update_node, both one_pass_*, _run_substeps). Add a verify_node check that fluxes respond to injected params.

### 2. HIGH — every compile() overwrites gm.params with the constructor snapshot; calibrated values are silently lost on any recompile, and load_state before the first compile drops checkpointed params
Files: graph_manager.py:2536 (self.params = self._snapshot_params() unconditionally); checkpoint.py:211-216 (restores only into keys already present in gm.params["nodes"]).
Invariant broken: "checkpoints store gm.params so a calibrated graph resumes calibrated"; "a set value takes effect on the next step". add_edge, add_external_input, remove_*, add_coupling_group, replace_node/_check_static_data_dirty, a REST PUT on a legacy node, or the profiler's one-iteration variant all recompile and reset every live leaf (including surrogate weights leaves, which have no constructor backing) with no warning.
Reproducers: test_repro_core.py::test_load_state_before_compile_drops_params (build; load_state; run(1) -> assert 30.0 == 77.0), ::test_recompile_after_calibration_discards_live_params (edit gm.params, add_external_input, step -> 30.0 == 77.0), ::test_static_data_dirty_recompile_discards_live_params. All existing tests compile before load_state, which is why this is uncovered.
Fix: in compile(), merge instead of replace: start from _snapshot_params() and overlay any existing self.params leaf whose node/key/shape/dtype still exist (warn on dropped ones). In load_state, compile first if not compiled (or stash the loaded params and apply after the merge).

### 3. HIGH — integer/uint32 state leaves in (or alongside) a coupling group pass through a float32 image; values >= 2^24 are silently corrupted, typed PRNG keys crash
Files: graph_manager.py:745-757 (initial_node_states casts every non-float leaf of group nodes to float32), :1160-1165 (template_img in _run_ift_forward does it for every node in the state, including nodes outside the group).
Reproducer: test_repro_core.py::test_uint32_and_large_int_leaves_survive_a_coupled_step[kw0|kw1] (IFT and fori): ACTUAL [3735928576, 305419904] vs DESIRED [0xDEADBEEF=3735928559, 0x12345678=305419896]. ::test_typed_prng_key_leaf_in_coupled_group and ::test_typed_prng_key_leaf_outside_group_with_ift_group (key-holding node NOT in the group): ValueError: Cannot convert_element_type from key<fry> to float32 at graph_manager.py:747 / :1161.
Root cause: the workaround for closure-converted integer constants is applied to all non-float leaves of all nodes; the "|value| < 2**24" precondition in the comment is not enforced.
Fix: restrict the image trick to group nodes; raise a clear trace-time error for uint32/int64/typed-key leaves; add a runtime error_if for |value| >= 2^24; longer term, carry non-float leaves through the custom_jvp as their own dtype.

### 4. MEDIUM — two mapped edges with the same key share one params["mappings"] slot; the first edge silently uses the second's weights
Files: graph_manager.py:1663-1676 (_snapshot_params keyed on edge.key), :1876-1922 (add_edge never rejects duplicates), edge.py:31-35 (key ignores mapping/additive/transform).
Reproducer: test_repro_core.py::test_duplicate_mapped_edges_collide_in_params_mappings — two additive edges a.v->b.inp with H1=I, H2=2I: len(gm.params["mappings"]) == 1 (expected 2); step delivers 2*(H2 v) instead of H1 v + H2 v. remove_edge also removes both.
Fix: reject a second mapped edge with an existing key in add_edge (or add an ordinal to key); check in validate().

### 5. MEDIUM — constrain with transform="log" and lo != 0 lands exactly ON the strict bound; check_params rejects the fit's own output; unconstrain returns -inf
File: params.py:120-131 (lo + clip(exp(u), fi.tiny, fi.max) — the clamp only works for lo == 0).
Reproducer: test_repro_props.py::test_log_constrain_is_strictly_inside_and_re_unconstrainable (Hypothesis falsifying example u=-15.0, lo=8.0) and ::test_log_constrain_with_nonzero_lower_bound_lands_on_bound_and_breaks_fit_continuation: p = 8.0 + exp(-15) -> 8.0 in float32; check(): "param=8.0 below bound 8.0"; unconstrain(p) = -inf. Consequence: fit() can return params check_params rejects (warm restart raises) and can evaluate the loss at the singular value p == lo.
Fix: clamp relative to lo: lo + max(exp(u), eps*max(1,|lo|)) with eps = finfo.eps (or nextafter(lo, inf) - lo); same treatment for logit with large |lo|, |hi|.

### 6. MEDIUM — a partial params pytree silently reverts missing nodes/keys to the CONSTRUCTOR constants, not to gm.params
File: graph_manager.py:86-93 (_node_update: node_params None -> 3-arg call -> node reads self.params); nodes merge {**self.params, **params} so a missing key falls back to the constructor value. _validate_params (:1680-1737) checks unknown keys but not missing ones.
Reproducer: test_repro_core.py::test_partial_params_pytree_uses_constructor_constants_not_gm_params and ::test_partial_node_entry_keyerror_or_fallback: with gm.params stiffness 300, step(params={"nodes": {}}) and step(params={"nodes": {"s": {"damping": 2.0}}}) both give velocity -0.3 (k=30) instead of -3.0.
Fix: require a complete pytree in _validate_params (static, free) or fill missing entries from params_snapshot.

### 7. MEDIUM — PUT /graph/params/{node} writes NaN into gm.params AND node.params then returns 500; a string for a live float returns 500
File: server.py:388 (spec.check(jnp.asarray(value)) raises TypeError for non-numeric -> unhandled 500), :392-404 (NaN passes check because NaN comparisons are False, is written, then response serialisation fails).
Reproducers: test_repro_api.py::test_put_nan_is_rejected — raw body {"params": {"elasticity": NaN}} -> 500 and afterwards gm.params[...]["elasticity"] == nan and node.params["elasticity"] == nan (the "validate everything before mutating anything" contract is violated; a recompile bakes NaN in). ::test_put_string_for_live_float_is_400_not_500 -> 500.
Fix: in the validation loop wrap jnp.asarray(value, dtype=live[key].dtype) in the same try/except the write loop uses; reject non-finite values with 400; make ParamSpec.check reject NaN (#10).

### 8. MEDIUM — ShardedStencilNode._grid_shaped_boundary_inputs misclassifies an (n,) input on an n x n grid
File: sharded_node.py:601-630. A per-column profile of shape (n,) for a square (n, n) grid sharded on axis 0 matches the sharded-axis extent by coincidence, is sharded and halo-padded, and the inner update_padded then fails.
Reproducer: test_repro_sharded.py::test_square_grid_profile_input_is_misclassified_as_grid_shaped — returns {'profile'}; sharded update raises TypeError: add got incompatible shapes for broadcasting: (4, 8), (1, 4) while the unsharded node runs. The rectangular variant passes.
Fix: require input.ndim >= state ndim (or exact leading-shape match); prefer an explicit BoundaryInputSpec flag over the heuristic.

### 9. LOW — sysid.fim / fit_lm noise_std as a pytree (documented) crashes
File: sysid.py:371 and :601: jnp.ndim(noise_std) == 0 is True for a dict (deprecation warning, returns 0) so the scalar branch runs jnp.asarray(dict).
Reproducer: test_repro_core.py::test_fim_noise_std_pytree_dict -> TypeError: float() argument must be a string or a real number, not 'dict'.
Fix: only take the scalar branch for numbers and 0-d arrays.

### 10. LOW — ParamSpec.check / check_params accept NaN
File: params.py:140-150. Reproducer: test_repro_core.py::test_check_params_rejects_nan -> DID NOT RAISE. Fix: reject non-finite values.

### 11. LOW — stale set_param_spec override survives remove_node; to_dict/from_dict round trip fails
Files: graph_manager.py:2006-2020 (remove_node), :1852-1858, :3777-3809. Reproducer: test_repro_core.py::test_stale_spec_override_after_remove_node_breaks_round_trip -> KeyError: "unknown node 't'" from from_dict. Fix: drop overrides in remove_node/remove_edge or filter param_spec_overrides() to existing nodes/edges.

### 12. LOW — FMI min attribute is inclusive but the sidecar rejects value == min for log/logit leaves
Files: model_description.py:477-478, sidecar.py:196 + params.py:140-150. Reproducer: test_repro_core.py::test_fmi_min_attribute_is_not_settable_for_log_leaves -> ValueError: s.params.stiffness=0.0 below bound 0.0. Fix: advertise min = nextafter(lo, +inf) for strict bounds or document the open interval.

### 13. LOW — ParamSpec.from_dict({"bounds": null}) crashes
File: params.py:95-101. Reproducer: test_repro_core.py::test_param_spec_from_dict_bounds_null -> TypeError: cannot unpack non-iterable NoneType object. Fix: `lo, hi = d.get("bounds") or (None, None)`.

### 14. LOW — constrain changes the dtype of an integer-typed bounded identity leaf
File: params.py:132-135 (jnp.clip with Python-float bounds promotes int32 -> float32). Reproducer: test_repro_props.py::test_identity_bounded_int_leaf_keeps_dtype (float32 != int32). Only reachable via integer matrix_mapping weights made trainable. Fix: .astype(u.dtype).

### 15. LOW (profiler) — the one-iteration measurement runs with constructor params
File: profiler.py:240-275: _one_iteration_variant calls gm.compile() (resets gm.params, see #2) and restores gm.params only in finally, so the timed one-iteration steps use constructor constants, not the live ones the real step uses. Timing-only effect; not reproduced numerically.

## Probed and held
- constrain(unconstrain(p)) == p for logit leaves in-bounds and log leaves with p - lo >= 1e-3 (Hypothesis, 40 examples each) — test_repro_props.py::test_logit_round_trip_inside_bounds, ::test_log_round_trip_inside_bounds.
- logit constrain(u) strictly inside (lo, hi) and re-unconstrainable for |u| <= 1e4 — ::test_logit_constrain_is_strictly_inside.
- trainable_mask / unconstrain / constrain share tree structure with params for random graphs mixing params nodes, a legacy 3-arg node, a params node with no float leaf, and a mapped edge; dtypes/shapes preserved; legacy nodes absent, empty nodes present as {} — ::test_mask_unconstrain_constrain_share_structure_and_round_trip.
- run_scan(n, params) == n x step(params=...) including a mapped edge and modified params — ::test_run_scan_equals_repeated_step_with_params.
- reset_state() then 3 steps bit-identical (state and _meta) to a fresh compile for default IFT, IQN-IMVJ with jacobian_reuse, linear predictor, Aitken+diagnostics — ::test_reset_state_then_step_matches_fresh.
- fit never moves masked-out leaves and its result passes check_params (lo=0 log leaves) — ::test_fit_never_moves_masked_leaf.
- Consistent RBF mappings reproduce constants and conservative ones preserve the sum for all four kernels on random 1-D point sets — ::test_consistent_rbf_mapping_reproduces_constants; projection_1d_mapping with a zero-width target cell is finite — ::test_projection_1d_mapping_degenerate_target_cell.
- fim(mask=...) accepts numpy-bool masks — ::test_fim_mask_with_numpy_bools_and_arrays.
- windowed_loss is exactly 0 at the truth on a MULTIRATE graph, with and without sample_every=2 — test_repro_misc.py::test_windowed_loss_zero_at_truth_on_multirate_graph, ::test_windowed_loss_with_sample_every_on_multirate_graph.
- IFT gradient through a 20-step lax.scan over a coupled group matches central finite differences (rel 2e-2) for acceleration none/aitken/iqn-ils and iteration_mode="jacobi" — ::test_ift_gradient_matches_finite_differences_over_scan.
- run_adaptive uses gm.params — ::test_run_adaptive_uses_params.
- REST: list for a scalar live key -> 400 with nothing written; a live PUT survives a later structural PUT on a legacy node and survives /sim/reset — test_repro_api.py::test_put_list_for_scalar_live_float_is_400_not_500, ::test_put_live_param_survives_a_structural_put_on_another_node, ::test_put_live_param_then_reset_keeps_it.
- Sharded vs unsharded agree for a (5,) profile input on an 8x5 grid — test_repro_sharded.py::test_rectangular_grid_profile_input_is_fine.

## Suspicious but unconfirmed
- Predictor path (graph_manager.py:1522-1547) flattens all fields of group nodes including integer leaves and extrapolates them (2*x_n - x_{n-1}); not tested.
- gm.params shared-dict mutation from the REST thread while RealtimeRunner steps: value replacement is atomic under the GIL; a PUT that added a key would break tree_flatten. Not possible today.
- EdgeSpec.to_dict() includes transform/additive/units but GraphManager.from_dict (:3831-3847) only re-adds the four name fields, so additive=True and transform are lost on round trip. Likely pre-existing; not checked on main.
- HeatNode.update_padded (heat.py:371-430) still reads self.params["thermal_diffusivity"]; ShardedStencilNode(HeatNode) therefore does not accept params — consistent with the contract but a silent loss of calibratability vs the unsharded node (LBM was migrated, heat was not).
- fit_lm: `if not accepted or step_norm < step_tol` relies on short-circuit evaluation; step_norm is unbound when not accepted.
- ParamSpec.to_dict() emits float('inf') bounds; json.dumps writes non-standard Infinity. Not exercised.
