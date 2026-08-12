# Wavelet solver — architectural performance investigation (decision document)

**Status: investigation complete, no production code written.** All three
performance levers were measured on the real matrix-free engine (RTX A2000 8GB,
float64). Prototype scripts in this directory. Every number below was produced or
reproduced here; measurement methodology was checked against this codebase's known
pitfalls (async-dispatch timing, change-of-basis identities, roundoff-dominated FD).

## TL;DR

| Lever | Verdict | Decisive measurement |
|---|---|---|
| Mixed precision (fp32/fp64) | **not worth it** | fp64/fp32 matvec = **1.9×** at 64³ (bandwidth-bound), and it forfeits the 1e-14 adjoint |
| R2 adaptive APPLY | **not worth it** | restricted-synthesis **dead** (100% grid coverage); sparse-assembly = **2.6 GB** BCOO at 64³ (3.6× memory regression) for ≤1.4× (Amdahl-capped) |
| R1 contrast-robust multigrid | **the win, with a fork** | full operator **15.4×** at χ-contrast 100 in 3D; masked (production) **3.9–9.4× fewer iters** |

R1 is the only lever worth building. It lines up with first principles: R1 attacks
the **iteration count** (the dominant factor and the source of the contrast cliff),
R2 the **per-matvec cost** (Amdahl-capped at ~1.4×), precision the **per-op
constant** (bandwidth-capped at ~1.9×).

---

## Lever 1 — Mixed precision: NOT WORTH IT

`bench_precision.py`, fp64 vs fp32 matvec, `block_until_ready`, A2000:

| grid | fp64 | fp32 | ratio |
|---|---|---|---|
| 64³ (N=262144) | 1.46 ms | 0.77 ms | **1.90×** |
| 64³ (2nd config) | 2.54 ms | 2.00 ms | 1.27× |
| 32³ (N=32768) | 1.89 ms | 0.38 ms | 5.02× |

The transform is **memory-bandwidth-bound**, so fp32 buys ~1.9× at the target
scale — not the theoretical 64× ALU ratio. Against that small win: fp32 CG
typically stalls near 1e-6 relative residual, and — decisively — the whole system
rests on a **1e-14 jit-vs-eager, ~5e-10 grad-vs-FD adjoint** that is a float64
result. A ~1.9× matvec is not worth risking that. (Could revisit later as an
opportunistic fp32-forward / fp64-adjoint split; low priority.)

## Lever 2 — R2 adaptive APPLY: NOT WORTH IT

Goal was to make the matvec cost scale with |Λ| instead of N. Two sub-approaches,
both fail on evidence.

**(b) restricted synthesis — DEAD.** `measure_coverage_amdahl.py`, real
CDD-selected Λ at 64³. Fraction of grid cells touched by synthesis of a
Λ-supported vector:

| set | \|set\| | cells touched |
|---|---|---|
| full Λ (coarse included, as CDD returns) | 16384 | **100.00%** |
| Λ minus the coarse level | 16320 | 99.98% |
| coarse level only | 64 | 100% |
| **a single coarse DOF** | 1 | **95.39%** |
| finest level of Λ only | 10965 | 4.18% |

The wrong-sign-safety mechanism mandates the coarse level, and coarse wavelets have
near-global support (one coarse DOF alone touches 95% of the domain). You cannot
make synthesis cheaper by restricting to touched cells — they are all touched. If
you *only* had fine wavelets it would win (4.18%), but you never do.

**(a) direct sparse assembly — feasible but a bad trade.** `A_wave` is exactly
block-Toeplitz (`measure_coverage_amdahl.py` part C: two columns of a structural
block differ by `0.000e+00`), so it could be assembled directly from O(log N)
representative columns. But the 3D density is high (`measure_nnz64.py`):

| grid | nnz/row @1e-12 | BCOO @1e-12 |
|---|---|---|
| 16³ | 262 | 0.02 GB |
| 32³ | 461 | 0.24 GB |
| **64³** | **619** | **2.60 GB** |

The sparse `A_wave` at 64³ is **2.6 GB** — 3.6× the *entire* current 717 MB
matrix-free solve, regressing the memory win we just achieved. And the Amdahl
ceiling caps the payoff: the jitted solve spends only ~30% of wall-clock in the
matvec (a free matvec → ≤1.4×). Trading 2.6 GB for ≤1.4× is a bad deal.
(1D nnz/row is small and contrast-independent — ~10→33 over N=16→256 — but 1D was
never the concern.)

## Lever 3 — R1 contrast-robust multigrid: THE WIN

A matrix-free geometric multigrid V-cycle (`mg.py`, linear prolongation,
rediscretised coarse operator with coefficient coarsening) used as an operator-mode
preconditioner via the pullback `M⁻¹ = Wn⁻¹ · MG · Wn⁻ᵀ`. An operator-dependent
Dendy/BoxMG variant (`stencil_mg.py`, exact Galerkin RAP) was also built.

**Conditioning — exact κ (gold standard, generalized eigenvalue), 2D N=1024:**

| contrast | unprec | hybrid-Jacobi | **mg (geometric)** | mg (op-dep) |
|---|---|---|---|---|
| 1 | 4.1e3 | 37.7 | **1.28** | 1.88 |
| 1e2 | 3.3e5 | 2814 | **12.4** | 23.0 |
| 1e4 | 3.1e7 | 8.4e5 | **522** | 784 |

227× better conditioning than hybrid at contrast 100; holds at 10⁴. Two surprises:
geometric MG **beats** the op-dependent variant on these fields, and while κ is not
perfectly contrast-*independent* (grows ~contrast^0.8), the **iteration count is
effectively flat** (below).

**Net wall-clock — full operator, matrix-free, jitted incl. setup (`r1_net_speedup.py`,
`r1_masked_and_setup.py` Gate A), 32³ smooth:**

| contrast | hybrid iters / wall | mg iters / wall | **net wall-clock** |
|---|---|---|---|
| 1 | 92 / 23.6 ms | 9 / 7.8 ms | **3.0×** |
| 1e2 | 591 / 151.9 ms | 11 / 9.7 ms | **15.7×** |
| 1e3 | 1965 / 505.2 ms | 12 / 10.4 ms | **48.4×** |

MG iterations are 9→12 across three decades of contrast — effectively
**contrast-independent**. The V-cycle + pullback costs only **~1.5 matvec-equivalents**
(the worry that it would be expensive is refuted). Per-solve hierarchy rebuild
(counted, Gate A) is negligible. Correctness cross-check (`agree`, both solve the
same system) ~1e-10.

**Pullback verified (the D3-trap-critical piece):** `wn_inv = Wn⁻¹` to 2.8e-17,
round-trip and adjoint-identity ~1e-15. Correct because `Wn⁻¹ = diag(norms)·analysis`
(not the transpose).

---

## THE FORK (why the full-operator 15.4× does not directly reach production)

Production does not solve the full operator — it solves the **CDD-masked active-set
system**. Gate B (`r1_masked_and_setup.py`) measured the masked path:

| grid / contrast | masked-hybrid iters | masked-MG iters | iter ratio |
|---|---|---|---|
| 32³ smooth, χ=1 | 38 | 33 | 1.2× |
| 32³ smooth, χ=1e2 | 164 | 42 | **3.9×** |
| 32³ smooth, χ=1e3 | 489 | 52 | **9.4×** |
| 64² jump, χ=1e2 | 282 | 44 | 6.4× |
| 64² jump, χ=1e3 | 741 | 88 | 8.4× |

Masking **degrades** the preconditioner (42 iters masked vs 11 full): `(M⁻¹)_ΛΛ ≠
(A_ΛΛ)⁻¹`, so the full-grid V-cycle restricted to Λ is a good-but-imperfect
preconditioner for the active block. Netting the ~1.5-matvec V-cycle cost, the
masked-path wall-clock win is estimated at **~1.6× (χ=1e2) to ~3.8× (χ=1e3)**.
(A direct jitted masked wall-clock measurement was attempted but the
MG-build-inside-jit nested in the CG `while_loop` hit a pathological compile time —
itself an implementation risk for the setup-inside-trace path, see Option A.)

**The deeper finding.** Combine two measured facts:
1. Masking degrades the MG preconditioner (Gate B: 42 vs 11 iters).
2. The matrix-free matvec costs O(N log N) **regardless of |Λ|** (R2 coverage +
   the fact that the transform touches all N). The masked matvec is *not cheaper*
   than the full matvec.

Therefore the CDD active-set masking gives **no per-iteration cost saving** in the
matrix-free regime, and it **hurts** a good preconditioner. The masking's solve-cost
rationale is inherited from the dense `gather_solve` era (O(K³), where small K
genuinely paid) and is **largely obsolete** once a contrast-robust preconditioner
exists. This is the most important thing the investigation found.

### Option A — scoped drop-in (keep the mask)
`MultigridPreconditioner` implementing the M5 Protocol in operator mode (identity
coordinate maps, `inner_precond` = the V-cycle pullback, indicator = `|M⁻¹r|`),
dropped into the existing masked solve. Add `wn_inv`/`wn_inv_T` to `matrixfree.py`,
a `multigrid.py` module (productionise `mg.py`), and a `preconditioner="multigrid"`
path in `WaveletEllipticNode._build_operator` (built in-trace from the current `a`).
- **Win:** ~1.6–3.8× net (χ 100→1000), 6–8× on jumps, and it **removes the contrast
  cliff** that forced the χ ≤ 10² scope cap.
- **Preserves:** the whole architecture, CDD adaptivity, and the frozen-active-set
  adjoint exactly. Adjoint-safe by construction (preconditioner is `stop_gradient`'d).
- **Risk:** the MG-build-inside-`jit` compile hang seen here must be resolved (build
  the hierarchy with static-shape structure, not a re-traced Python loop). Days of
  careful work + validation (adjoint 1e-14, MMS, masked-iteration gate).

### Option B — architectural (drop the mask for the solve)
Solve the **full** MG-preconditioned system; the active set is no longer used for
the solve.
- **Win:** the full **15.4×**, and a **simpler adjoint** (no frozen-set approximation
  — the full solve is smoothly differentiable in θ).
- **Cost:** makes CDD / blindness / frozen-active-set machinery largely vestigial;
  the solution becomes dense (N coefficients, ~2 MB at 64³ — cheap); and it questions
  the "AdaptiveNode" design premise. Adaptivity would then have to justify itself on
  the **solution representation** (sparse fields / adjoint locality), not solve cost.
- **Open questions before committing:** does the full solve reproduce the intended
  (adaptive) solution to required accuracy? What, concretely, does adaptivity still
  buy the target applications? These need their own investigation.

Both options preserve the 1e-14 adjoint. The choice is scope: A is "add a
preconditioner"; B is "the adaptive design's solve-cost rationale is obsolete —
reconsider it."

## Recommendation

Ship **Option A** when ready (concrete contrast-cliff removal, low risk, in-scope),
**and** open a separate, careful investigation of **Option B** before adopting it —
because if B holds up, it is both faster (15.4× vs ~1.6×) and *simpler* than the
current design, which is rare and worth verifying rather than assuming.
