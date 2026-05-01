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

**Sketch**:

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

**Surfaced by**: same. The persistent cache lives at
`~/.cache/jax_compilation_cache` (set in `MIME/tests/conftest.py` and
`MIME/scripts/run_ar4_helical_drive.py`). First run on a clean cache
pays the full XLA compile (~50 s for the AR4 graph on a 2060);
subsequent runs hit the cache and start in ~14 s.

**Sketch**: a `maddening.scripts.warm_cache` module that takes a
`GraphManager` factory and runs one step on a representative input
to populate the cache. CI could ship the warm cache as an artefact
so first-developer-after-merge doesn't pay the cost.
