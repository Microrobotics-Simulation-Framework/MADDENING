# M18 — 64³ scale gate + memory footprint

**The number you asked for: the matrix-free path runs 64³ in ~2.1 GB above
baseline — comfortably within the A2000's 8 GB.** Dense assembly at 64³ would
need ~2.2 TB and is impossible on any single device.

## Memory footprint (matrix-free, fully assembled + CDD + solve, CPU fp64)

| grid | N = side³ | dense operator (one (N,N) fp64) | matrix-free peak (above baseline) |
|---|---|---|---|
| 16³ | 4,096 | 0.13 GB (× ~4 live ≈ 0.5 GB) | trivial |
| 32³ | 32,768 | 8.6 GB (× ~4 ≈ 34 GB — **dead on A2000**) | ~0.3 GB |
| 64³ | 262,144 | **0.55 TB (× ~4 ≈ 2.2 TB)** | **~2.1 GB** |

The 64³ run (`_matrixfree_scale_solve`, side=64, N=262 144):

- Peak RSS: baseline 146 MB → after setup 1279 → after CDD 2071 → after solve
  2275 MB. **Peak above baseline ≈ 2.1 GB.**
- Timings (CPU, fp64): setup 6.5 s, full CDD selection 72 s (40 outer × masked
  CG), one frozen solve 12 s.
- CDD filled the budget K=N/16=16 384 and reported converged=True.
- **Ran to completion.** Dense assembly would allocate a 262144² array first.

## Why it fits: memory is O(N), not O(N²)

Everything on the matrix-free path is O(N): the state/CG vectors (262 144 fp64 =
2 MB each, a handful live), the `norms`/`D` vectors (2 MB), and the transform's
lifting temporaries (O(N)). `column_norms_fast` and `wave_diagonal_fast` each
`vmap` synthesis over `1 + n_levels·n_subband` block representatives (36 at
64³) — the transient `(36, N)` array is ~75 MB. Nothing scales with N².

## Caveats (honest)

- **This is CPU RSS, not GPU memory.** I could not measure A2000 device memory in
  this environment (CPU-only run). The O(N) scaling and the ~2.1 GB CPU working
  set say it fits 8 GB with wide headroom, but the first real GPU run should
  confirm the actual device allocation (JAX preallocation / XLA buffers differ).
- **fp64 on the A2000 is slow** (~1/64 the fp32 rate). The 72 s CDD / 12 s solve
  are CPU numbers; the A2000 will be memory-comfortable but compute-bound in fp64.
  A production run may want fp32 with an fp64 refinement, or to accept the fp64
  latency — a perf question, not a memory one.
- The 64³ demonstration is **constant-coefficient** (uses `wave_diagonal_fast`,
  whose block-invariance holds only for constant `A_phys`). The variable-
  coefficient app-1 path (M19) lags `D` at a reference `a₀` — the diagonal is
  then built once from the reference, so the same O(log N) diagonal applies.

## Bottom line

The scale gate is met: **64³ (app-1's target order) is reachable on the A2000 in
memory**, which was impossible with dense assembly. What remains for app 1 is not
scale but the θ→A wiring (M19), the RHS/sensor (M20/M21), and — for χ ≥ ~10² — the
R1 contrast-robust preconditioner (D5), since matrix-free CG iteration count still
scales with contrast.
