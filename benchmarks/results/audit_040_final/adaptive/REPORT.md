# Audit: `src/maddening/nodes/adaptive/` + `nodes/heat.py` (c51cd6a)

Auditor `adaptive`. Worktree `/home/nick/MSF/msf/MADDENING-wt/audit/adaptive` (detached at `origin/release/0.4.0`, `c51cd6a`). All commands run with `JAX_PLATFORMS=cpu` and `PYTHONPATH=<wt>/src`.

> Transcribed by the orchestrator: a harness guard refused the auditor's own
> `Write` to this path. The 12 reproducer scripts and their `.out` files in this
> directory were written by the auditor itself.

## Summary

The core of the subsystem is sound, and I could not break it. `update()` is genuinely JAX-traceable: eager / `jit` / `vmap` / `scan` agree bit-for-bit, including under `vmap` with four *different* active sets across the batch, and the frozen adjoint matches central finite differences to 5e-10 relative for all three `ift_linear_solve` backends on two toys. Shape stability is real — the mask is a value, not a shape, so nothing recompiles and the active set cannot outgrow its allocation. MADD-ANO-003 (the gradient's blindness at an active-set switch) is honestly and unusually well documented; I reproduced it and it matches the registry to the digit.

What I found is concentrated at the **edges**: one graph-level interaction that leaves `GraphManager` unrecoverably corrupted (HIGH), and a cluster of things the base class does not enforce or diagnoses wrongly, which matter because the class is about to acquire a second subclass. `heat.py`'s +31 lines are correct.

## Findings

### HIGH — `add_node` at the documented Palais trap leaves a ghost node that wedges the graph and crashes `step()` with a bare `KeyError`

**What breaks:** `AdaptiveNode.initial_state()` raises `AdaptiveNodeBlindnessError` by design at a trap — the developer guide lists it under *Failure modes* with a recovery procedure, and `tests/nodes/adaptive/test_graph_integration.py::test_add_node_at_an_established_trap_fails_loudly` asserts it. `GraphManager.add_node` registers the node *before* calling `initial_state()`, so the raise leaves `_nodes["trap"]` populated and `_state["trap"]` absent. The name is then permanently taken: re-adding raises `ValueError`, `remove_node` raises `KeyError`, and `compile()` accepts the graph. The failure surfaces much later, inside the compiled step.

**Evidence** (`repro_addnode_not_atomic.py`):
```
add_node raised AdaptiveNodeBlindnessError (documented, recoverable)

gm.node_names            : ['trap']
gm._nodes (private)      : ['trap']
gm._state (private)      : []

recovery attempt 1 -- perturb theta and re-add under the same name:
   ValueError Node 'trap' already exists in the graph.
recovery attempt 2 -- remove_node first:
   KeyError 'trap'
what the half-added node does to the graph:
   compile(): ok
   reset_state(): ok
```
and with one healthy node alongside the ghost (`repro_ghost_node.py`):
```
node_names           : ['trap', 'good']
state keys           : ['good']
params['nodes'] keys : ['trap', 'good']
UserWarning: WARNING: node 'trap' is disconnected (no edges or external inputs)
...
  File ".../graph_manager.py", line 3724, in _resolve_and_update_node
    spec, new_state[node_name], boundary_inputs, spec.timestep,
KeyError: 'trap'
```
`gm.params["nodes"]` contains `trap`, so `sysid.fit` would optimise a parameter of a node that can never run, and `compile()`'s warning points at the wrong problem.

**Why it happens:** `src/maddening/core/graph_manager.py:2584-2585`
```python
self._nodes[node.name] = spec
self._state[node.name] = node.initial_state()   # <- raises here
```
No try/except, no rollback. `AdaptiveNode` is, as far as I can tell, the only node in the tree whose `initial_state()` raises on a supported configuration, so this release is what makes a pre-existing non-atomicity reachable.

**What would make this a non-issue:** (a) if the error were unrecoverable by contract — it is not; the dev guide tells the user to perturb and retry, and `cold_start()` exists to produce the retry point; (b) if `remove_node` cleaned up — it raises `KeyError` off the missing `_state` entry; (c) if `compile()` rejected a stateless node — it compiles, and only `step()` fails. I checked all three.

**Suggested fix:** make `add_node` atomic — build the state first, then commit both dicts, or `del self._nodes[node.name]` on any exception. Risk near zero: a failed `add_node` becomes a no-op. Belongs to the `graph_manager.py` owner; reported here as an interface mismatch, not re-audited internally.

---

### MEDIUM — `is_trapped_at()` asserts a Palais symmetry trap for *any* near-zero frozen gradient; an empty active set is the case the wavelet port will actually hit

**What breaks:** `is_trapped_at` is the diagnostic the whole design leans on to separate "your budget is too small" from "no selection rule can help you". Its docstring calls it "the one diagnostic that can *establish* a symmetry trap" and the exception says "confirms a Palais fixed point of the problem's symmetry". It establishes neither. It fires whenever `|g_frozen|` is small relative to its own rate of change — true at an empty active set and at any stationary point, symmetric or not.

Textbook adaptive-wavelet selection (thresholding the coefficients — `@VasilyevPaolucci1996`, which the algorithm guide names as "the intended first concrete subclass") selects the **empty** set at cold start, because `c` is all zeros there. The base class accepts it, solves it, returns `c = 0`, and diagnoses it as a symmetry trap.

**Evidence** (`repro_empty_active_set.py`):
```
AdaptiveNodeBlindnessError raised by initial_state():
CoefficientThresholdNode 'wave': gradient-capture ratio 0.000 is below the
threshold 0.700 ... is_trapped_at() confirms a Palais fixed point of the
problem's symmetry: the frozen-set gradient has no component in the escape
direction, and no selection rule can supply one.  Remedies: cold_start() ...
--- ground truth about the point ---
n_active           = 0
gradient_capture_ratio = 0.0
is_trapped_at      = True
dJ/dtheta (frozen) = 0.0
the problem has no symmetry at all: the operator is diag(1..16)*theta
```
The operator is `diag(1..16)*theta`, the source is `1`: there is no group acting on it, so no `Fix(G)`, and every remedy the message names (`cold_start()`, `symmetry_break()`, "perturb the parameters") is guaranteed useless — the set stays empty at every `theta`. Combined with the HIGH finding, this also wedges the graph.

Same false positive at an ordinary interior optimum (`repro_is_trapped_false_positive.py`): at `theta* = 0.343973370458` where `dJ_frozen/dtheta = 2.6e-14` on the sine toy (`n=256, k=64, sensor_x=1/3`; the only reflection fixed point of that toy is `theta=0.5`), `is_trapped_at` returns `True`. The algorithm guide recommends running it "between optimiser steps" above `D_threshold` parameters — i.e. it will fire on success.

**Why it happens:** `base.py:946-947`
```python
proxy = _tree_norm(g0) / (rate + 1e-30)
return bool(proxy < 1e-2)
```
Nothing in the expression involves the mask's contents, a group action, or a fixed-point set. That is a necessary condition for a Palais trap, never a sufficient one. The empty mask reaches it because neither `update` (`base.py:791-796`) nor `_cold_start_state` (`base.py:1032-1038`) checks `mask.any()`.

**What would make this a non-issue:** if an empty set were rejected earlier — it is not; the dev guide mentions "the mask is empty" only as a possible cause of NaN, and with an identity-off-mask operator it does not even NaN, it returns zeros. If the ratio's sentinel caught it — it returns `0.0`, not the `1.0` sentinel, because `|g_full|` is healthy. If `on_blind` could downgrade it — the trap branch raises under `"warn"` too (`base.py:731-747`).

**Suggested fix:** two things. (1) Reject an empty mask in `update`/`_cold_start_state` with a message naming `is_cold_start` and the `prev`-based hysteresis idiom — two lines, and it turns the most likely wavelet-port bug into a good error message. (2) Rename `is_trapped_at` to what it measures, or make the wording conditional, and stop the error text asserting symmetry it has not established.

---

### MEDIUM — the documented "build the basis in `__init__`, keep it on `self`" pattern silently zeroes the gradient, and the framework's own guard cannot see it

**What breaks:** `docs/developer_guide/adaptive_node.md` §*Hooks* says in its worked skeleton: `# build basis arrays once; keep them on self (static data), not in state`. Both shipped toys do that. If any such array derives from a **trainable** parameter, `jax.grad` returns `0.0` and nothing objects — `compile()` included. The framework *has* a guard, extended in this very release and used by the `heat.py` diff (`static_data_deps`), but it only inspects arrays published through `static_data`, and `AdaptiveNode` publishes none.

**Evidence** — same array, same dependency, two ways of holding it (`repro_baked_trainable_param.py`):
```
objective reads the BAKED array (the documented AdaptiveNode pattern):
   jax.grad =  0.000000000000e+00    central FD(h=1e-6) =  0.000000000000e+00
objective recomputes from the traced parameter:
   jax.grad =  6.927405773986e-02    central FD(h=1e-6) =  6.927405774015e-02

compile() verdict on the baked node:
   compile() ACCEPTED the graph (no static_data_deps violation seen)

same array, but published through static_data and declared:
   compile() REFUSED: ValueError node 'baked' declares
   static_data['phi_sensor'] as derived from parameter 'sensor_x', which is
   trainable. ... the gradient with respect to 'sensor_x' would silently omit ...
```
The true derivative is `6.93e-2`; the taught pattern returns `0.0`. Note the finite difference is *also* `0.0`, so the usual oracle does not catch it either.

**Why it happens:** `SimulationNode.static_data_deps` (`core/node.py:484`) is keyed off `static_data`; a bare instance attribute is invisible to it. `AdaptiveNode` neither overrides `static_data` nor mentions it; the algorithm guide's *Parameters* table and the dev guide's *Failure modes* never name it.

**What would make this a non-issue:** if the shipped toys were themselves wrong — they are not. I checked every baked array: `PoissonSineTopKNode` bakes `_phi_sensor` from `sensor_x`, which is `ParamSpec(trainable=False)` (legal), and `_x`/`_phi`/`_lambdas` from `n`, a structural `int` that never reaches `params_pytree()` (legal); `MaskedDenseNode` bakes from `seed`, also structural. So this is a live trap for the next author, not a wrong number in shipped code — which is why MEDIUM rather than HIGH despite the silently-zero gradient.

**Suggested fix:** have `AdaptiveNode` publish the subclass's precomputed basis through `static_data` (so the existing compile-time guard applies), or at minimum add the `static_data`/`static_data_deps` rule to the dev guide's *Hooks* section and stop the skeleton comment calling a bare attribute "static data". The first has a real API cost at the freeze; the doc fix is free.

---

### MEDIUM — `compute_active_set`'s shape is checked, its dtype is not: returning scores or an `argsort` silently turns the node into a full-basis solver that every diagnostic certifies as healthy

**What breaks:** `base.py:791` does `jnp.asarray(mask, dtype=bool)` and `base.py:792` checks only shape. A float score array of shape `(n_max,)` — the single most likely thing an author writes before remembering the comparison — becomes `!= 0`, i.e. all-true. The node silently stops being adaptive, pays `n_max` per solve, and `gradient_capture_ratio` reports `1.0000` (perfect), because the frozen set *is* the full set.

**Evidence** (`repro_mask_dtype_not_enforced.py`):
```
PoissonSineTopKNode    n_active=  8/64  ratio=0.5647  trapped=False  J= 0.0127174247
ReturnsScores          n_active= 64/64  ratio=1.0000  trapped=False  J= 0.0125625291
ReturnsPermutation     n_active= 63/64  ratio=1.0000  trapped=False  J= 0.0125625291
```
No error, no warning. `ReturnsPermutation` (an `argsort` the author forgot to slice and scatter) is worse: 63/64 active looks like a plausible mask if you print its popcount.

**What would make this a non-issue:** if the diagnostics caught it — they actively confirm the opposite. If `verify_node` caught it — it does not; the node is numerically correct, just not adaptive. No result is wrong; the feature silently disappears and the cost silently becomes `O(n_max)` per step, which for the wavelet node is the whole point of the subsystem.

**Suggested fix:** require `jnp.issubdtype(mask.dtype, jnp.bool_)` in `update` and `_cold_start_state`, alongside the emptiness check above. Risk: a subclass returning a 0/1 integer mask would break — that is the intent, and it is caught on the first call.

---

### LOW — MADD-VER-004's "top-K sensor error strictly decreasing over K ∈ {4,8,16,32}" holds at the one point it is asserted at and fails at 9 of 48 nearby ones

**What breaks:** the benchmark's `acceptance_criteria` and the algorithm guide's *Validated Physical Regimes* row state monotone convergence in the budget as a property of the method. It is asserted only at `theta=0.42, sigma=0.04, x_s=1/3`. Sweeping `theta ∈ {0.20…0.70}`, `x_s ∈ {0.25, 1/3, 0.5, 0.75}` and both selection rules, monotonicity fails in 9/48 — including `theta=0.42, x_s=0.5`, one grid point from the asserted one.

**Evidence** (`repro_convergence_sweep.py`, excerpt):
```
 theta    x_s sel | K=4          K=8          K=16         K=32         monotone?
  0.42  0.500   b | 9.697e-05    1.470e-05    4.229e-05    2.049e-07     NO
  0.30  0.500   b | 4.166e-04    5.207e-04    1.646e-05    9.471e-09     NO
  0.70  0.250   b | 1.221e-04    2.514e-04    5.744e-06    7.060e-09     NO
non-monotone in 9/48 configurations
```
**Why:** top-K on a non-nested, non-local basis — the K=8 set is not a superset of the K=4 set. The *trend* is robust (K=32 error ≤ 2.1e-7 in all 48 runs); only "strictly decreasing" does not generalise.

**Suggested fix:** reword to what is true and useful ("sensor error < 1e-4 at K=32 and ≥ two orders below the K=4 error") and note in the guide that the decrease is not monotone in K for a non-nested selection. The benchmark would still pass.

---

### LOW — `docs/developer_guide/stability_report.md` is stale: none of the release's new public surfaces are in it

Its own header says it is "Refreshed at release time as part of the v0.3.0+ verification gate". It has not been.
```
adaptive entries the registry now has:
    maddening.core.solver_utils.ift_linear_solve   *** MISSING FROM COMMITTED REPORT ***
    maddening.nodes.adaptive.base.AdaptiveNode   *** MISSING ***
    maddening.nodes.adaptive.base.AdaptiveNodeBlindnessError   *** MISSING ***
    maddening.nodes.adaptive.base.adaptive_diagnostics_enabled   *** MISSING ***
    maddening.nodes.adaptive.base.set_adaptive_diagnostics   *** MISSING ***
```
`scripts/generate_stability_report.py` already lists `maddening.nodes.adaptive` in `STABILITY_MODULES`, and `tests/compliance/test_stability.py` checks the script *reaches* every tagged module — but nothing checks the committed Markdown matches what the script would emit. One-command fix; add the equality check to CI.

---

### LOW — `mask_safe` broadcasts on the last axis, so a matrix operand is masked column-wise with no error

`mask_safe` is the base class's answer to the one gradient trap it cannot repair, documented as taking "`x` : Operand to sanitise" with no rank constraint. On an `(n_max, n_max)` operator it masks columns, not rows (`base.py:600-601`, a plain `jnp.where`):
```
   A = [[0 1 2 3] [4 5 6 7] [8 9 10 11] [12 13 14 15]]
   mask_safe(mask, A) = [[0 1 2 1] [4 1 6 1] [8 1 10 1] [12 1 14 1]]
```
Every documented use is 1-D and both toys use it that way, so this is a trap, not a live bug. Either assert `ndim == 1` or document the broadcasting and show the `mask[:, None]` form.

---

### LOW (interface, `core/node.py` — flag for that auditor) — `params_pytree()` hard-codes float32, so under `jax_enable_x64` an AdaptiveNode solves in float64 but its parameters, gradients and diagnostics are float32

```
jax_enable_x64          : True
node.dtype (cold buffer): float64
   params_pytree['sigma'] = Array(0.04, dtype=float32)
state c dtype           : float64
cold-start solve used sigma = 0.04 (float64 python float)
the diagnostic evaluates at sigma = 0.03999999910593033 (float32-rounded)
gm.params['nodes']['a'] dtypes: {'theta': 'float32', 'sigma': 'float32', ...}
gradient dtype returned to the user: {'theta': 'float32', ...}
```
Cause: `core/node.py:342`, `out[key] = jnp.asarray(value, dtype=jnp.float32)`, unconditional. Two consequences specific to this surface: the adaptive suite's `conftest.py` enables x64 "because the gradient checks ... need float64", yet the gradient the framework hands back through `gm.params` is float32; and `check_gradient_capture()` with `params=None` evaluates at the float32-rounded pytree while `initial_state()`'s own cold-start solve uses exact float64 constructor values, so the ratio is not measured at the point the state was built at. 2e-8 relative — harmless except near a stationary point (see suspicions). Fix would be to resolve the dtype from `jnp.zeros(()).dtype` as `AdaptiveNode.__init__` already does; that touches every node, so not an adaptive-node decision.

## Unverified suspicions

- **`gradient_capture_ratio`'s negligible-gradient sentinel is far too tight.** `base.py:898` returns the `1.0` sentinel only when `|g_full| < 1e-12 * (1 + |theta|)`. Near a stationary point both gradients are small but well above that and the ratio is noise over noise: at the stationary point in `repro_is_trapped_false_positive.py` I got 0.393 from one call path and 1.259 from another for the same physical parameters. No wrong *decision* follows from this alone (the trap branch also needs `is_trapped_at`), so not a finding.
- **`_fingerprint` (base.py:1180) omits array shape** — it digests `(key, dtype.str, tobytes())`, so two leaves with the same flat bytes and different shapes collide in `_capture_cache`/`_trapped_cache`. I could not construct a reachable case: a leaf's shape is fixed for the life of an instance and the caches are per-instance.
- **The cold-start diagnostic is on by default and costs a full-basis solve.** For the first intended concrete subclass (adaptive wavelet collocation) the full-basis solve is by construction the thing that does not fit, and with `blindness_gate=True` every `initial_state()` — `add_node`, `reset_state`, profiler, REST API, FMI model description, hypothesis strategies — performs one. Documented under *Cost* and `blindness_gate=False` is a one-word opt-out, so not a defect; but the default is backwards for the subsystem's own use case and the freeze is the moment to revisit it.

## What I checked and found sound

- **Traceability** (`repro_traceability.py`): eager vs `jit` bit-identical; `scan` vs Python loop to 1.7e-18; `vmap` over parameters matches the per-element loop exactly. `_warn_on_non_finite_off_mask` correctly detects tracers and skips (its `except` covers `TracerArrayConversionError`, `ConcretizationTypeError`, `TypeError`, `ValueError`), so no host callback reaches the traced path. No measurable eager overhead from it (780 ms vs 807 ms per `n=512` update — wall clock on a shared machine, do not quote as a timing).
- **Shape stability under a genuinely varying active set** (`repro_mask_safe_and_vmap.py` §b/§c): four batch elements with four *different* active sets under one `vmap`; masks, coefficients and gradients all match the per-element loop to exactly 0.0. The active set cannot outgrow its allocation (a mask over a fixed buffer, not a list), and a change between two `jit` calls does not retrace.
- **Gradients** (`repro_gradient_fd.py`). Central differences in float64; step size chosen as the standard `h ~ (3·eps_mach·|J| / |J'''|)^(1/3) ~ 1e-5` trade-off between `h²` truncation and `eps/h` round-off, confirmed by a sweep showing the expected V-shaped minimum at `h = 1e-5`–`1e-6`. Every step checked for an active-set change across `[θ-h, θ+h]` and reported. Results: `MaskedDenseNode` (dense non-diagonal SPD, real Krylov adjoint) agrees to 5.9e-10 (`gmres`), 3.9e-10 (`cg`), 4.9e-10 (`dense`); `PoissonSineTopKNode` `k=32, solver=cg` to 4.5e-11. The frozen-basis IFT adjoint is correct and lineax's native autodiff is doing the right thing through `ift_linear_solve` for all three backends.
- **The active-set freeze is real.** I could not leak a tangent through the selection; `stop_gradient` is applied on all three paths (`update:791`, `_cold_start_state:1032`, `_objective_gradient:1100`), and the boolean cast kills any soft score before `stop_gradient` even runs.
- **MADD-ANO-003, independently reproduced.** 11 switches of the top-16 set over `theta ∈ [0.40, 0.44]`; at a switch the FD is a valid oracle only once `h` is small enough that the set stops changing, exactly as documented. Registry entry, guide and module docstring agree with the code and with each other.
- **No drift or ratcheting in `c`.** `update` re-solves from the parameters every step and `_solve_and_pack` fully replaces `c` and zeroes it off the mask, so repeated refine/coarsen cannot accumulate error in the coefficients. (The one place ratcheting could enter is `extra_initial_state` fields — see the contract section.)
- **Documentation and compliance.** `docs/algorithm_guide/nodes/adaptive_node.md` exists with a complete Implementation Mapping table, `@stability(EVOLVING)` on all four public symbols, and `NodeMeta` with `algorithm_id MADD-NODE-009`. All four compliance scripts pass (`check_anomalies`; `check_impl_mapping` — 16 mappings verified; `check_citations` — 13 citations, two pre-existing unrelated uncited-bib warnings for `Kruger2017`/`ShanChen1993`; `check_transforms`). The algorithm guide is included via the `algorithm_guide/nodes/*` glob in `docs/index.md`; the developer guide is listed explicitly. The only documentation defect is the stale stability report above.
- **`heat.py` (+31 lines).** The new `static_data_deps` is correct. I checked every read of `_grid_x`/`_grid_x_array` (lines 356, 621): both are behind `if self._is_nonuniform`, so the uniform branch's `{}` is right — the uniform Laplacian recomputes `dx = length / n_cells` from the traced `length`, which stays differentiable. The non-uniform branch declares `grid_points`, which is `ParamSpec(trainable=False)` — a legal dependency. The docstring's reasoning matches the code.
- **Round trips, graph integration, USD** — covered by the existing suite and green; no gap worth a separate reproducer.

## The author-facing contract (for the wavelet port)

Not findings — what a second subclass author will get wrong that the base class does not stop, ranked by likelihood.

1. **An empty active set at cold start** is the default outcome of coefficient-magnitude thresholding and is diagnosed as a symmetry trap (MEDIUM above). `is_cold_start=True` exists to let you special-case it; nothing says you must, nothing checks that you did.
2. **A non-boolean mask** silently disables adaptivity while the diagnostics report perfect health (MEDIUM above).
3. **A basis array derived from a trainable parameter in `__init__`** silently zeroes that parameter's gradient, and `compile()` does not object (MEDIUM above).
4. **`update` ignores `dt` and `boundary_inputs`**, and the three hooks never receive them, so an `AdaptiveNode` is an edge *source* only and cannot time-step. A subclass needing either must override `update` wholesale and call the private `_solve_and_pack`. Listed as open question #2 in the dev guide, but the freeze is the last cheap moment, and a time-dependent wavelet collocation node (the cited `@VasilyevPaolucci1996` method) needs `dt`.
5. **`extra_initial_state()` fields are carried through `update` unmasked.** The base class zeroes `c` and nothing else. A per-mode auxiliary array in the extra state keeps stale values for modes just deactivated — the one place ratcheting could enter a design that is otherwise drift-free.
6. **Hooks receive `self.params` overlaid with the injected pytree**, so `params` contains the non-numeric diagnostic settings (`on_blind` as `str`, `blindness_gate` as `bool`) alongside the physics. Harmless until someone writes `jax.tree.map` over it.
7. **`compute_full_basis_gradient` assumes an all-`True` mask is a valid, meaningful solve.** Documented as assumption 2, but load-bearing for *every* diagnostic, and a selection rule expressed as a fixed-size gather rather than a mask will not satisfy it.

## Reproducers on disk (each with a `.out`)

| File | Shows |
|---|---|
| `repro_addnode_not_atomic.py`, `repro_ghost_node.py` | HIGH: the wedged graph and its `KeyError` |
| `repro_empty_active_set.py`, `repro_is_trapped_false_positive.py` | MEDIUM: `is_trapped_at` false positives |
| `repro_baked_trainable_param.py` | MEDIUM: silently-zero gradient vs the `static_data` guard |
| `repro_mask_dtype_not_enforced.py` | MEDIUM: non-boolean mask |
| `repro_convergence_sweep.py` | LOW: non-monotone convergence in K |
| `repro_mask_safe_and_vmap.py` | LOW: `mask_safe` broadcasting; plus the clean vmap result |
| `repro_gradient_fd.py` | sound: FD study, three solvers, active set checked per step |
| `repro_traceability.py`, `repro_degenerate_masks.py` | sound: jit/vmap/scan; degenerate-mask survey |
| `repro_params_float32_under_x64.py` | LOW: float32 params under x64 |

## Test commands run

```
cd /home/nick/MSF/msf/MADDENING-wt/audit/adaptive
PYTHONPATH=$PWD/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/nick/MSF/msf/.venv/bin/python -m pytest tests/nodes/adaptive/ -q -p no:cacheprovider -rs
  -> 101 passed in 63.21s   (0 skipped)

... -m pytest tests/nodes/test_heat.py tests/usd/test_usd_adaptive_node.py -q -p no:cacheprovider -rs
  -> 16 passed in 34.52s

scripts/check_anomalies.py  scripts/check_impl_mapping.py
scripts/check_citations.py  scripts/check_transforms.py    -> all OK
```
Full suite not run. Nothing modified outside this directory. No fixes applied.
