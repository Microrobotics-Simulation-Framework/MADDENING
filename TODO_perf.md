# MADDENING — Performance TODOs

Items here are perf opportunities surfaced by downstream consumers
(MIME, MICROROBOTICA). Each one is concrete, profiled, and has a
sketch of the win.

---

## TODO-PERF-1 — Fuse Gauss-Seidel coupling-group iterations into one XLA kernel

**Surfaced by**: MIME's AR4 + helical-UMR drive experiment
(`MIME/scripts/run_ar4_helical_drive.py`,
`MIME/experiments/ar4_helical_drive`). Tracked alongside MIME's
actuation-decomposition push (MIME `ddbece8`).

**Symptom**: The new-chain dejongh graph (Motor + PermanentMagnetNode
+ RobotArmNode + UMR + drag) takes ~76 ms / step on a 2060 with the
coupling group disabled, ~95 ms / step with it enabled. ~50% of that
budget goes to per-iteration GPU launch overhead inside the
Gauss-Seidel inner loop on the `body ↔ ext_magnet ↔ magnet` cycle —
each iteration launches CRBA + RNEA + cuSolver + dipole `jacrev` +
RigidBody integration as separate kernels.

For the user's iterative-visualisation workflow, sub-30 ms/step would
turn the runner into something approaching real-time on this rig.

**Where the cost goes**:

```
gm.step() compiled function (jit'd ✓)
   └── coupling group while_loop
         ├── iter k=0
         │     ├── arm.update            ← CRBA + RNEA + cuSolver call
         │     ├── motor.update          ← small ODE integrate
         │     ├── ext_magnet.update     ← jacrev(B_dipole)
         │     ├── magnet.update         ← T = m × B, F = (∇B) · m
         │     └── body.update           ← RigidBody integrate
         ├── iter k=1 (same kernels relaunched)
         …
         └── iter k=N (up to 20)
```

Each of those node updates inside the coupling group already lives
inside the outer jit, so they trace into one big XLA HLO graph.
Empirically though, XLA still emits separate CUDA launches per node
update inside the loop — `JAX_LOG_COMPILES=1` shows ~30 separate
"Finished tracing" events on the first call after the persistent
cache warms, and the steady-state cost is dominated by GPU dispatch
latency rather than actual compute.

**Status (2026-09-15)** — diagnosis revised, main fix landed:

- Measured on CPU (2 × 400k-cell HeatNode group, converges at iteration 2):
  fori step cost was linear in `max_iterations` (≈2.4 ms/iter) regardless of
  convergence — 80 % dead iterations at N=10, 90 % at N=20.  There was no
  per-node `jit` boundary, no host sync, and one XLA compile per step, so the
  launch-overhead hypothesis below was *not* the cause; the "~30 tracing
  events" were node-level compiles.
- **Landed**: `CouplingGroup.solver="ift"` (early-exit `while_loop` + IFT
  derivative) is the default; step cost is flat in `max_iterations`.
- Of the ≈2.4 ms/iter, ≈0.8 ms was physics and ≈1.5 ms bookkeeping.  Still
  open: `_apply_interface_overrides` does a full-array `.at[idx].set` per node
  per iteration (≈0.9 ms/node/iter) although the correction is
  iteration-invariant (hoist it out of the loop); the residual is over the
  full state (the `"interface"` norm is cheaper).
- Acceptance below still needs measuring on the AR4 graph on GPU.

**Status (2026-09-16)** — measured on the AR4 + helical-UMR graph
(RTX A2000, `benchmarks/bench_coupling.py --graph mime-ar4`, results in
`benchmarks/results/ar4_before.json`):

- Warm coupled step **1.41 ms**, staggered baseline **0.58 ms**; dispatch
  floor 0.17 ms; the group (5 nodes, cap 6) runs 5 body iterations on
  every step and never meets `tolerance=1e-6` (residual 1.9e-6 at the
  cap; it converges to 6e-8 at 6 body iterations, i.e. cap 8).  Measured
  coupling overhead 0.77 ms = 0.19 ms per extra iteration.  Device busy
  ~40 % of the wall step with ~416 kernels/step under CUDA graphs: the
  step is launch-bound, not compute-bound.  **Acceptance (<= 30 ms/step)
  passes by 20x.**
- The 33 ms/step the MIME driver reported was not the step: the
  jitted step was compiled three times per run (weak-typed seed leaves
  flipping to strong after the first steps — see CHANGELOG *Fixed*), and
  two of those compiles fell inside the driver's "steady-state" timing
  window.  With the seed state normalised the driver reports
  **1.67 ms/step** wall, results unchanged.
- The `_apply_interface_overrides` hoist idea is retired: the correction
  depends on each iteration's boundary inputs (not hoistable), only nodes
  with interface DOFs pay it, and on a GPU heat chain (4 x 64 cells) the
  whole coupling cost is 0.04 ms/step.
- Remaining lever for this graph is on MIME's side.  Measured (same
  graph, same 20-step start, 100 steps, RTX A2000; `|dpos|` is the max
  body-position difference against MIME's current cap-6 result, position
  scale 7e-4 m):

  | group config                      | ms/step | iters | converged | dpos    |
  |-----------------------------------|--------:|------:|----------:|--------:|
  | cap 6, L2 1e-6 (MIME today)       |  1.66   | 5.0   |   8 %     | —       |
  | cap 8                             |  2.03   | 6.6   |  74 %     | 3.5e-10 |
  | cap 12, aitken                    |  2.20   | 8.9   |  71 %     | 3.5e-10 |
  | cap 12, iqn-ils                   |  4.65   | 7.5   |  87 %     | 1.4e-9  |
  | cap 12, interface norm rtol 1e-4  |  1.60   | 3.1   | 100 %     | 1.2e-8  |

  Recommendation for MIME: `convergence_norm="interface", rtol=1e-4`
  (with a cap of 12 as a guard): the group converges on every step in
  ~3 iterations, is the fastest variant, and differs from today's
  truncated result by 1.6e-5 relative.  Aitken does not help on this
  fixed point; IQN-ILS costs more per iteration than it saves at 5 nodes.

**Sketch** (original):

1. **Audit the coupling-group code path** (`maddening.core.coupling.group`)
   to confirm whether it uses `lax.while_loop` (preferred — single
   compiled body) or a Python-side fixed-point loop unrolled into the
   trace. If it's already `while_loop`, look at the loop-body HLO for
   redundant copies / contractions that could be hoisted.
2. **Check whether each node's `update` is being inlined** into the
   outer trace or kept as a separate `jit` boundary. The latter would
   force a kernel launch per node per iteration. If so, removing the
   inner `@jit` decorators from node updates (relying on the outer
   `gm._compiled_step` to capture them) should let XLA fuse them.
3. **Consider folding the Gauss-Seidel residual check into the loop
   condition** — if the residual computation lives outside the
   `while_loop`, it'll force a host-device sync per iteration.
4. **Benchmark**: on the AR4 + helical-UMR graph (run via
   `MIME/scripts/run_ar4_helical_drive.py --no-coupling-group` for a
   baseline, then add `use_coupling_group=True`), aim for the
   coupling-group cost to be ≤ 50% over the no-coupling-group path
   (today it's ~25% slower with full 20-iteration cap). Stretch goal:
   make it ≤ 10% over baseline so the high-fidelity option is
   default-acceptable for visualisation.

**Acceptance** when this lands: AR4 + helical-UMR graph runs at
≤ 30 ms/step on a 2060 with `use_coupling_group=True`, on warm cache.

**Linked anomalies** (none open today; may need a `MADD-ANO-*` if
the redesign changes coupling-group convergence in a user-visible
way).

---

## TODO-PERF-2 — Persistent JAX compile-cache warmup tool

**Status (2026-09-16) — done.** `maddening.core.simulation.compile_cache`:
`enable(cache_dir)` (defaults: `MADDENING_COMPILATION_CACHE_DIR`, then
`~/.cache/maddening/xla`; sets the persistent-cache thresholds so
sub-second MADDENING compiles are cached), `enable_from_env()` (called by
`GraphManager.compile()`, so exporting the env var is enough), and
`warm_cache(gm_factory, n_steps=, scan_steps=)`.  Cross-process cache
hit is tested (`tests/core/test_compile_cache.py`, slow lane).

`one_pass_jacobi` `device_put` inside the traced body (TODO.md perf item
6): measured on one RTX A2000 with a 4-rod jacobi heat chain, 0.203 vs
0.187 ms/step with placement off/on, results bit-identical — a no-op on
a single device.  Left in place: removing it would change multi-device
semantics this rig cannot verify.

**Surfaced by**: same. The persistent cache lives at
`~/.cache/jax_compilation_cache` (set in `MIME/tests/conftest.py` and
`MIME/scripts/run_ar4_helical_drive.py`). First run on a clean cache
pays the full XLA compile (~50 s for the AR4 graph on a 2060);
subsequent runs hit the cache and start in ~14 s.

**Sketch**: a `maddening.scripts.warm_cache` module that takes a
`GraphManager` factory and runs one step on a representative input
to populate the cache. CI could ship the warm cache as an artefact
so first-developer-after-merge doesn't pay the cost.
