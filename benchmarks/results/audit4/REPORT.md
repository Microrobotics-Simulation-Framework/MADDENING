# MADDENING audit round 4 — branch `feat/graph-params-sysid` (PR #9), working tree == HEAD 86dafe1

Scratch dir: `/home/nick/.claude/jobs/d22809a4/tmp/audit4/`.  Run reproducers from the repo root with
`JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../.venv/bin/python -m pytest <file> -q -p no:cacheprovider`:
- `test_b1.py` — FMU bridge unset inputs, params shape at step, checkpoint `_meta`/node-name edge cases, bridge state round trips, compressed-npz DoS probe
- `test_b2_rest.py` — REST state / node / edge / checkpoint / sim endpoints (validate-then-write discipline)
- `test_b3_num.py` — gradients vs finite differences (interface correction with params, mapping weights), sysid fit through a flux edge, checkpoint URL+manifest round trip on a multirate+IFT+predictor+mapped graph, HybridNode, to_dict/from_dict edge attributes, run_adaptive
- `test_b4_sharded.py` — ShardedStencilNode(HeatNode) params contract and grid-shaped boundary input (sets 4 host devices itself)
- `test_b5_cwrapper.py` — the compiled C wrapper driven through ctypes against the real bridge and against hostile servers (`cbuild/maddening_fmu_plain.so`; an ASan build is in `cbuild/` too)
- inline probes quoted below were run as `python - <<EOF` snippets; each is reproduced in the finding text.

## Findings (ranked by severity)

### 1. MEDIUM — FMU bridge: an input the importer has not set is *absent* from the step, not the advertised `start="0.0"`; a HeatNode FMU runs adiabatic until the first `fmi3Set*` and after every `reset` / `set_state`
File: `src/maddening/fmi/tcp_bridge.py:236` (`self._sidecar.step(self._inputs)`, `_inputs` starts as `{}`), `:262-267` (`reset` sets `_inputs = {}`), `:365-371` (`_get` reports 0.0 for an unset input), `src/maddening/fmi/model_description.py:598-605` (`start="0.0"`, comment "the graph's own default for an unset external input is zero").
Invariant: the FMU behaves like the graph (`gm.step()` fills every declared external input with zeros, `graph_manager.py:3225-3242`) and like its own model description.
Reproducer: `test_b1.py::test_bridge_unset_input_is_not_the_advertised_zero` — HeatNode with `left_temperature` as external input, 5 bridge steps with nothing set: FMU `temperature == [100]*8` (T_left defaulted to `T[0]`, `heat.py:463`) vs graph `[0, 78.81, 97.94, 99.91, ...]`; meanwhile `get` of the input returns `0.0`. Same after `reset` and after `set_state` from a snapshot taken before any `set`.
Root cause: the bridge never materialises the zero defaults the graph applies; nodes whose `boundary_inputs.get(name, <non-zero default>)` (HeatNode, HeartPump `backpressure`, BallNode `table_position=None`) therefore diverge; SpringDamper happens to default to 0 so the existing tests pass.
Fix: seed `_inputs` with `np.zeros(var.shape, var.dtype)` for every `causality="input"` variable at construction, in `reset`, and for inputs missing from a snapshot (or pass `gm._default_external_inputs()` merged with `_inputs`).

### 2. MEDIUM — `PUT /graph/state/{node}` validates nothing: a wrong shape is written and silently reshapes the node (200), a missing/string field wedges the server (every later `/sim/step` is 500), unknown fields are accepted
File: `src/maddening/api/server.py:341-348` (`_python_to_jax(req.state)` -> `gm.set_node_state`, no check), `src/maddening/core/graph_manager.py:3967-3970` (`set_node_state` only checks the node name).
Invariant: the same validate-then-write discipline the params endpoint got in rounds 1-2.
Reproducers (`test_b2_rest.py`): `::test_put_state_wrong_shape` — `{"position":[1,2,3],"velocity":0}` -> 200, next `/sim/step` -> 200 with `s.position == [1.0, 1.997, 2.994]` (spring now vector-valued); `::test_put_state_missing_field` — `{"position":1.0}` -> 200, then `/sim/step` -> 500 and stays 500; `::test_put_state_string` -> 200 then 500; `::test_put_state_unknown_field` -> 200 (extra key kept in `_state`); `::test_put_state_nan_then_run_scan` — NaN is accepted into the state (the test fails on JSON serialisation of the response; the write happened).
Fix: compare the field set with `gm.get_node_state(name)`, coerce each value to the live leaf's dtype, require identical shape and finiteness, and only then `set_node_state`; return 400 naming the field.

### 3. MEDIUM — `PUT /graph/params/{node}` with JSON `true`/`false` for a float leaf drops the leaf from `gm.params` at the recompile; the node then runs with `stiffness=True (=1.0)`, `GET` and `to_dict` report `True` (round-2 #1 incomplete)
File: `src/maddening/api/server.py:395` (`if key not in live or isinstance(value, bool): continue` routes a bool to the structural path), `:418-420` (`node.params[key] = value; gm._dirty = True`), `src/maddening/core/node.py:199` (`params_pytree` skips bools).
Reproducer (inline): `PUT {"params": {"stiffness": true}}` -> 200 with `"stiffness": 30.0` in the response body; `node.params["stiffness"] is True`; `/sim/step` -> recompile emits `RuntimeWarning: compile() dropped live gm.params leaves ... ["nodes['s']['stiffness']"]`, velocity after one step `0.005` (== k=1.0; k=30 gives 0.15); `gm.params["nodes"]["s"]` has no `stiffness`; `GET /graph/params/s` -> `{"stiffness": true, ...}`; `gm.to_dict()` serialises `"stiffness": true`; `gm.check_params()` passes.
Fix: a bool for a key that is a live float leaf (or a float constructor param) is a 400; only route bools to the structural path for keys whose constructor value is a bool.

### 4. MEDIUM — `POST /graph/nodes` with an invalid constructor param returns 201, adds the node, and every subsequent `/sim/step` is 500 (the graph is wedged until the node is deleted)
File: `src/maddening/api/server.py` add_node handler (`node_cls(**params)` succeeds because nodes do not validate their constants; the failure surfaces only inside the trace).
Reproducer: `test_b2_rest.py::test_add_node_bad_params` — `{"type":"SpringDamperNode","name":"s2","timestep":0.01,"params":{"stiffness":"hot"}}` -> 201, `/sim/step` -> 500 (`TypeError: bad operand type for unary -: 'str'` from the trace; inline run with `raise_server_exceptions=True`).
Fix: after constructing the node, run `node.params_pytree()` / a dry `verify_node`-style trace (or at least type-check against `param_specs` and the existing constructor value types) before `add_node`; on failure 400 and do not add.

### 5. MEDIUM (security; almost certainly pre-existing on main) — `/checkpoint/save` and `/checkpoint/load` take an arbitrary server-side path from an unauthenticated client
File: `src/maddening/api/server.py:426-441`; there is no authentication anywhere in `server.py` (grep for auth/token: none).
Reproducer: `test_b2_rest.py::test_checkpoint_save_arbitrary_path` — `POST /checkpoint/save?path=<tmp>/anywhere/evil` -> 200 and `<tmp>/anywhere/evil.npz` exists (arbitrary file write with attacker-influenced content: node names/fields and `_meta` keys are in the archive); `::test_checkpoint_load_arbitrary_path` — `POST /checkpoint/load?path=/etc/passwd` -> 400 whose detail is numpy's error text (file-existence / type oracle). Combined with #2/#4 the whole graph is mutable by anyone who can reach the port.
Fix: restrict checkpoint paths to a configured directory (`Path(root, name).resolve()` must stay under `root`), and document that the API must sit behind auth / bind to localhost.

### 6. LOW-MEDIUM — a corrupt `fmi3SetFMUState` blob kills the FMU instance for good: the C wrapper embeds the importer's bytes unescaped, the bridge drops the connection on the resulting JSON error
Files: `src/maddening/fmi/c/maddening_fmu.c:522-529` (`sprintf(in->req, "{\"op\":\"set_state\",\"state\":\"%s\"}", st->blob)` — `st->blob` comes from `fmi3DeserializeFMUState`, i.e. importer-supplied opaque bytes), `src/maddening/fmi/tcp_bridge.py:170-176` (`except (OSError, ValueError): break` — `json.JSONDecodeError` is a `ValueError`, so a malformed message closes the socket instead of answering `{"ok": false}`).
Reproducer: `test_b5_cwrapper.py::test_set_fmu_state_with_corrupt_bytes_does_not_kill_the_instance` — after `fmi3DeserializeFMUState(b'abc"def\\x')` + `fmi3SetFMUState` (-> fmi3Error, log "recv failed"), `fmi3DoStep` returns fmi3Error forever ("send failed"); inline socket probe: sending `{"op":"set_state","state":"abc"def"}` gets EOF, whereas `[1,2,3]` gets a proper error reply. FMI importers may legitimately hand back bytes from disk; the standard expects fmi3Error, not a dead instance.
Fix: C side — reject a blob that is not `[A-Za-z0-9+/=]*` before building the request (or JSON-escape it); bridge side — catch `json.JSONDecodeError`/`UnicodeDecodeError` in `_serve_conn` and reply with an error, break only on framing (`_MAX_MESSAGE`) and socket errors.

### 7. LOW-MEDIUM — `load_state` accepts a state field of the wrong shape or dtype (round-2 #9 fixed params only; the "state fields already get this" claim was wrong)
File: `src/maddening/core/simulation/checkpoint.py:196-215` (only node names and field *names* are compared; `jnp.array(arr)` is written verbatim).
Reproducer (inline): save a spring checkpoint, overwrite `s/position` with `np.ones(3)` -> `load_state` succeeds and the next step returns `position.shape == (3,)`; overwrite with `np.array(7, np.int64)` -> loads as int32, the step retraces (`trace_count == 2` after two steps) and the first step runs from an integer leaf.
Fix: the same shape check as `_restore` for params, plus `jnp.asarray(arr, dtype=live.dtype)`.

### 8. LOW — `_validate_params` checks keys but not shapes: a wrong-shape leaf via `gm.params[...] =` or `step(params=)` silently broadcasts the node's *state* to that shape, permanently
File: `src/maddening/core/graph_manager.py:1816-1900` (`_validate_params`), `:1752-1766` (`_coerce_params_leaves` only fixes non-arrays / weak types).
Reproducer: `test_b1.py::test_step_params_with_wrong_shape_is_rejected` and `::test_assigning_wrong_shape_into_gm_params_is_rejected` — `stiffness = ones(3)*30` -> `step` succeeds, `state['s'] == {position: (3,), velocity: (3,)}` and stays (3,) on later default steps (`trace_count` grows). The checkpoint (round 2 #9), REST and FMI entry points reject this; the Python API does not.
Fix: compare `jnp.shape(leaf)` against compile-time shapes (next to `self._params_dtypes`) in `_validate_params` (static, free) and in `_params_or_default` for `gm.params` itself.

### 9. LOW — a checkpoint without `_meta` loaded into a multirate graph (same nodes, different timesteps) makes the next step raise `KeyError: '_meta'`
File: `src/maddening/core/simulation/checkpoint.py:210-216` (`raw_state.pop(_META_KEY)` when the file has no meta), `src/maddening/core/graph_manager.py:3167` (`full_state[_META_KEY]["step_count"]`).
Reproducer: `test_b1.py::test_checkpoint_meta_missing_for_multirate_graph` — save from `(s: DT, b: DT)`, load into `(s: DT, b: 2*DT)`, `run(3)` -> `KeyError: '_meta'`. The round-2 fix compiles first (so `_meta` exists) and then the pop removes it again.
Fix: when the checkpoint has no `_meta`, keep the freshly compiled one (zeroed) instead of popping; or refuse with a clear `ValueError`.

### 10. LOW — bridge `step` with `n > 1` sub-steps is not atomic: an exception in sub-step k leaves the state advanced by k-1 sub-steps while `_time` (and the importer's clock) stay put
File: `src/maddening/fmi/tcp_bridge.py:235-238`.
Reproducer (inline, step_fn raising on its 3rd call): `step dt=5*DT` -> `{"ok": false}`, afterwards `get [time, position] == [0.0, 0.50447]` (position moved two sub-steps from 0.5). The C wrapper reports `lastSuccessfulTime = in->time` (unchanged) so the importer believes nothing happened.
Fix: run the sub-steps on a local copy of the sidecar state and commit state + time together (or report `t + (k-1)*dt` as the last successful time).

### 11. LOW — `ShardedStencilNode` delivers grid-shaped boundary inputs halo-padded, but `HeatNode.update_padded` expects `heat_source` at the unpadded local shape
Files: `src/maddening/cloud/multigpu/sharded_node.py:410-418` (`_pad_like_state(v) if k in grid_bi`), `src/maddening/nodes/heat.py:431-436` (`broadcast_to(source, (n_local,))`). `LBMNode.update_padded` (`lbm.py:815-828`) handles both forms; HeatNode does not.
Reproducer: `test_b4_sharded.py::test_sharded_heat_grid_shaped_source_matches_unsharded` — `ValueError: Incompatible types for broadcasting: input type=float32[6]{V:devices} and requested type=float32[4]`. Loud, so low; but the "drop-in sharded HeatNode" cannot take a per-cell source at all.
Fix: in `HeatNode.update_padded`, accept a source of padded length (`source[halo:-halo]`) as LBM does, or have the wrapper not pad inputs for nodes that do not declare it.

### 12. LOW — `POST /graph/edges` with a non-existent node -> 201, and every later step is 400 until the edge is deleted
`test_b2_rest.py::test_add_edge_bad` — `add_edge` does not resolve node names; `compile()` then fails ("edge references non-existent source node"). Validate in the handler (or in `add_edge`) and return 404/400 without mutating.

## Earlier fixes re-checked
- R1 #1 / R2 #2 (flux and interface-correction params): complete. d(rod_a.T[-1] after ONE step)/d(alpha) by `jax.grad` matches central FD to 7e-4 for `solver="ift"` and `"fori"` (`test_b3_num.py::test_grad_wrt_alpha_at_interface_cell_one_step_matches_fd`), i.e. the correction now sees the injected alpha and is differentiable. `HybridNode` forwards params (`::test_hybrid_node_params_contract`; the round-2 suspicion is resolved: `nodes_without_params() == []`).
- R1 #2 (recompile keeps live params): complete for the Python API; the REST bool path still drops a leaf (finding 3).
- R1 #3 (integer leaves in groups), R1 #4 (ordinal mapping slots), R1 #5/R2 #12 (log/logit clamps), R1 #9/#10/#13/#14, R3 all_to_all, R3 second instance: not re-run (existing regression tests; no new angle found in the time budget).
- R1 #6 / R2 #6 partial params: complete for keys/nodes; shapes are not validated (finding 8).
- R1 #7 / R2 #1 / R2 #5 / R2 #6 REST params: complete for string / null / list / NaN / JSON int / pre-compile / GET; incomplete for JSON booleans (finding 3). `mass: [1.0]` -> 400 "expected shape (), got (1,)" (inline).
- R1 #8 sharded heuristic: params contract through the wrapper holds — injected `thermal_diffusivity` equals a sharded node built with that constructor value on a non-uniform field (`test_b4_sharded.py::test_sharded_heat_accepts_params_contract`); grid-shaped `heat_source` fails loudly (finding 11).
- R1 #11 / R2 #8 stale overrides: not re-run.
- R2 #3 (load before compile keeps `_meta`) and R2 #4 (mapping weights in checkpoints): complete together — multirate + IFT + linear predictor + mapped edge, `save_state_with_manifest` -> `download_and_load_state("file://...")` into a fresh graph, 4 further steps within rtol 1e-6 in every node field and every `_meta` leaf (`test_b3_num.py::test_checkpoint_url_round_trip_multirate_coupled_with_mappings`). The no-`_meta` cross-graph case fails (finding 9).
- R2 #9 (checkpoint param shape check): complete for params; state fields still unchecked (finding 7).
- R2 #7 / #10 / #11 (x64 residual dtype, x64 Python-float leaves, weak-type retrace): read only; `compile()` now seeds the residual in the group's float dtype and `_merge_live_params` coerces to the base dtype — consistent.
- R3 pickle RCE: the TCP bridge is pickle-free and the round-3 tests hold. `FmuSidecar.handle` (`sidecar.py:238-283`) and `fmu_state.deserialize_fmu_state` still `pickle.loads` caller bytes — documented as in-process only, nothing in `src/` serves them on a socket (grep), so (b).
- R3 whole-multiple step: complete; the partial-advance desync on an exception is finding 10.
- R3 atomic `set`: complete through the compiled wrapper (`fmi3SetFloat32` + `fmi3GetFMUState`/`SetFMUState`/`Reset` round trip via ctypes, `test_b5_cwrapper.py::test_get_state_set_state_round_trip_and_reset_via_c`).
- SIGPIPE-safe sends (86dafe1): the wrapper survives a server that closes mid-body (`test_b5_cwrapper.py::test_wrapper_against_server_that_closes_mid_body_then_recovers`: fmi3Error, later calls fmi3Error, no signal) and a 0xFFFFFFFF length prefix on `hello` (`::test_wrapper_against_server_replying_huge_length`: instantiate returns NULL, "recv body failed").

## Probed and held
- `jax.grad` wrt `params["mappings"][key]["H"]` through 3 steps of a `matrix_mapping` edge matches central FD (36.0 vs 35.9999); `trainable_mask` flips with `set_param_spec(edge.key, "H", ParamSpec())`; `sysid.fit` on the mapping weights lowers the loss by >1e3 (H itself is not identifiable from this data, by construction) — `test_b3_num.py::test_grad_wrt_mapping_weights_matches_fd_and_fit_recovers_H`.
- `sysid.fit` + `windowed_loss` on a graph where the observed node is fed only by a **flux edge** (`spring_force`) recovers `stiffness = 45.0000` from a start at 20 — `::test_fit_through_flux_edge_recovers_stiffness`.
- Interface-correction gradient with params (see re-check above), both solvers.
- Checkpoint URL + manifest round trip on a multirate + IFT + predictor + mapped graph (see re-check above).
- `to_dict`/`from_dict` keep `transform` (registered name), `additive`, `source_units`/`target_units`, two additive edges into one field, an external input shape and a calibrated live leaf (`gain = 2.5`); trajectories identical — `::test_to_dict_from_dict_keeps_every_edge_attribute`.
- A partial `params` with a wrong-shape mapping weight raises at `step` — `::test_partial_params_with_wrong_mapping_shape`.
- `run_adaptive` uses `gm.params` (runs; no numeric check beyond finiteness) — `::test_run_adaptive_trace_count_and_params`.
- Bridge: many small steps == one large step (10xDT vs 1x10DT, rel 1e-6), `reset` restores params and inputs, `set_state` after `reset` restores position, params and pending inputs — `test_b1.py::test_bridge_set_state_after_reset_and_many_small_vs_one_large`.
- Bridge: a compressed npz whose *unexpected* key holds 400 MB is rejected before decompression (peak 1.4 MB) — `test_b1.py::test_bridge_compressed_npz_is_loaded_before_shape_check`. (A bomb under an *expected* key is decompressed before the shape check — see suspicious.)
- Bridge: malformed-JSON-but-valid-value (`[1,2,3]`) gets `{"ok": false}`, not a disconnect (inline).
- ShardedStencilNode(HeatNode) params contract (see re-check).
- REST: `/sim/run?n_steps=-5` is a no-op 200; `?n_steps=abc` -> 422; `mass: [1.0]` -> 400.
- Checkpoint: a node name containing `/` is refused loudly on load ("missing from checkpoint: ['a/b']; extra: ['a']") — `test_b1.py::test_checkpoint_node_name_with_slash`; (b) undocumented limitation (node names are not validated at `add_node`).

## Suspicious but unconfirmed
- `gm.trace_count` counts only the jitted `_compiled_step`; `run_scan`, `run_scan_with_history`, `run_sweep`, `run_adaptive` and `sysid.windowed_loss` call `_build_step_fn()` afresh and trace (and XLA-compile the whole scan) on **every** call without touching the counter — `trace_count == 0` after two `run_scan` calls and after `run_adaptive` (`test_b1.py::test_trace_count_under_run_scan`, `test_b3_num.py::test_run_adaptive_trace_count_and_params`). The docstring ("0 before the first step; more than 1 ... means retraced") is misleading for those entry points, and the per-call recompilation of `run_scan` is a pre-existing performance cost.
- Bridge `_decode_state`: a compressed npz with a huge array under an *expected* key (`s/<node>/<field>`) is fully decompressed (`arr = data[k]`) before the shape check — a 64 MB message can expand to gigabytes. Check `data.zip.getinfo(k).file_size` / the array header before loading, or cap the archive's uncompressed size.
- Bridge `_decode_state` accepts any `_time`, NaN included; `get time` then returns NaN, which the C wrapper's `strtod` happily parses.
- `read_endpoint` in the C wrapper splits on the last `:` and does not strip brackets, so `[::1]:5555` cannot resolve (IPv6 literal endpoints unsupported).
- `FmuSidecar.set_fmu_state` (pickle path) trusts `fmu_state.schema_token` carried inside the same caller-supplied object — the token check is not a defence if that path is ever exposed on a socket.
- `EdgeSpec.to_dict` emits `mapping` descriptions but `from_dict` refuses them (documented); `add_edge` ordinal derivation for node names containing `#` remains untested (round-2 note).
- `GraphManager.add_node` does not validate node names (`/`, `#`, `.`, `->` each break the flat checkpoint keys, edge keys, FMI variable names or `_META_KEY` parsing somewhere).
