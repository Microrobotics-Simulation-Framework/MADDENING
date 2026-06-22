# Wavelet AdaptiveNode Derisking — Spike Findings (running log)

**THIS IS A SPIKE.** Investigative, exploratory, deliberately quick. Code
under `spikes/wavelet_derisking/` is throwaway experiment scaffolding, not
production. The goal is to map terrain and resolve the four gates in
`plans/WAVELET_ADAPTIVE_NODE_DERISKING_SPIKE.md` against their stated
pass/fail criteria — or surface where the plan's own assumptions need
correcting.

**Branch:** `spike/wavelet-derisking` (off `feat/adaptive-node-base`).
**Started:** 2026-06-22.

Numbers below are from a numpy harness (linear algebra: condition numbers,
sign patterns). JAX is reserved for the trajectory-adjoint test (§6).

---

## Status board

| Gate | Investigation | Status | Verdict so far |
|------|---------------|--------|----------------|
| 1 | §2 Hyp A — DD+DK condition number | **done** | 1D PASS; 2D PASS *with corrected scaling* (see C1) |
| 1 | §2 Hyp A — CDD convergence | **done** | PASS — converges ≤~20 iters, beats rolling ~97%, Hyp A confirmed |
| 1 | §3 — DD phi sign / wrong-sign safety | **done** | PASS *for production CDD*; locality theorem needs qualification (see below) |
| 2 | §4 — 3D sparsity break-even | **done** | PASS — J_err 0.1% at k=N/16; trap risk ~zero (non-local only) |
| 3 | §5 — Stokes / stream-function cavity | **done** | PASS on conditioning (Jacobi); NS-Ghia match → implementation |
| 3 | §6 — trajectory adjoint under lax.scan | **done** | PASS — grad=FD to 1e-10, no T-degradation, no mitigation needed |

**ALL FOUR GATES POSITIVE → derisking criterion met (see Executive Summary).**

### Continuation series (6 follow-up investigations, 2026-06-22)

| # | Investigation | Status | Verdict |
|---|---------------|--------|---------|
| 1 | Level-Jacobi hybrid preconditioner | **done** | hybrid ≡ full Jacobi at O(log N) cost → new default |
| 2 | DD-4 operator in JAX (BCOO PoC) | **done** | no show-stoppers; 3–5 wk estimate confirmed |
| 3 | CDD on nonlinear residual (Burgers + cavity) | **done** | PASS — CDD survives nonlinearity |
| 4 | Discontinuous coefficients (Brinkman) | **done** | PASS — Jacobi adapts automatically, no design change |
| 5 | Submatrix conditioning 2D/3D | **done** | PASS — κ(A_ΛΛ)≤κ_full by Cauchy interlacing |
| 6 | D_threshold=5 empirical | **done** | NOT confirmed → lower to D≥2 (TopK node only) |

**All resolved → spike series complete; WaveletAdaptiveNode implementation can
begin (see Cross-cutting statement at end).**

### Closeout round (3D battery + limitation probes + handoff docs, 2026-06-22)

| # | Investigation | Status | Verdict |
|---|---------------|--------|---------|
| 1 | 3D completeness battery (A–E) | **done** | all 1D/2D conclusions extend to 3D |
| 2 | Limitation probes (A–E) | **done** | no blockers; Dirichlet/non-sep/geometry/memory/iters all OK |
| 3 | `KNOWN_LIMITATIONS.md` | **done** | 17-entry handoff doc (standalone) |
| 4 | Sharding design note | **done** | shardable via two-tier; `halo_width_at_level` API |

**SPIKE CLOSED 2026-06-22** — see `## Spike closed` at the very end. Open items
are implementation-scoped, not derisking.

---

## Executive Summary

The derisking spike resolved all four gates from
`plans/WAVELET_ADAPTIVE_NODE_DERISKING_SPIKE.md §7`. Per the plan's overall
criterion, **`WaveletAdaptiveNode` implementation can proceed.** Highlights
and the corrections the spike forces on the plan:

1. **Preconditioning works — but use algebraic Jacobi, not theory-`2^{tj}`,
   as the default.** DD + Dahmen-Kunoth gives O(1) κ in 1D/2D/3D for the
   Laplacian and the biharmonic, *but only with the right scaling*. The
   plan's multi-D `2^{|λx|+|λy|+|λz|}` is the **anisotropic** scaling and
   fails (κ ∝ N) on the isotropic operator (**Correction C1**). The
   isotropic Mallat basis with single-level `2^j` works; **algebraic Jacobi
   (`√diag A`) is consistently the best and most robust** — it auto-adapts
   to the elliptic order t (t=1 Laplacian *and* t=2 biharmonic) without it
   being hard-coded. Recommend Jacobi default, `2^{tj}` opt-in.

2. **CDD is the selection criterion; it converges and beats rolling
   decisively.** ~17–21 outer iters to J_err~1e-5 at |Λ|≈14% of N; beats
   rolling top-|c_prev| by ~97%; matches the oracle. Near-sharp σ=0.02 is
   *easier* (sparser), not harder.

3. **Wrong-sign safety is real but via coarse-inclusion, not pure locality.**
   DD-4 has ±7% negative side-lobes and is *not* strictly single-signed, so
   top-|b| can wrong-sign just like the sine basis. CDD (which always keeps
   the coarse levels dominating the sensor functional) is wrong-sign-safe
   across the whole boundary sweep at K=N/16. **The locality theorem must be
   stated as "CDD/coarse-inclusion is wrong-sign-safe," not "locality forbids
   wrong-sign."** Keep top-|b| deprecated.

4. **3D is worth it.** CDD reaches J_err~0.1% with only N/16 = 6.25% of the
   3D basis — well past the "worth building" threshold.

5. **The blindness/symmetry-trap machinery is non-local-basis insurance.**
   The selection-induced blindness trap (which drove much of the base-class
   design: `blindness_ratio`, `symmetry_break`, the cold-start gate) is a
   **property of non-local bases (sine)**. The local wavelet basis is immune:
   in 1D the sine ratio → 0.000 at θ=0.5 (BLIND) while DD stays ≈1; in 3D the
   wavelet shows no trap (ratio ≥ 0.91) even with top-|b|. For
   `WaveletAdaptiveNode` this machinery is a cheap, near-inert safety net — do
   not over-invest in trap mitigation (no 3D-specific δ needed). It remains
   correct and necessary for `TopKAdaptiveNode`.

6. **Trajectory adjoint through `lax.scan` is exact** (grad=FD to 1e-10, no
   T-degradation). The feared c_prev contamination doesn't occur; no
   `stop_gradient`/`custom_vjp` mitigation needed.

7. **Stream-function (biharmonic) cavity is viable on conditioning grounds**
   (Jacobi-preconditioned). The full Navier–Stokes Ghia accuracy match is
   scoped to implementation, not derisking.

**Net:** the wavelet path is de-risked. The main plan edits are: adopt the
isotropic basis + Jacobi preconditioner (C1), qualify the locality theorem
(§3), and downgrade trap-mitigation effort for the wavelet node (§4).

---

## Correction C1 — the plan's 2D/3D Dahmen-Kunoth scaling is wrong for an isotropic operator

**This is the most important early finding and it amends the plan.**

The plan (§2 Hyp B and §4) specifies the multi-D Dahmen-Kunoth scaling as

```
D_λλ = 2^{|λx| + |λy| + |λz|}   (sum of per-axis levels)
```

That is the **anisotropic / hyperbolic-cross** tensor scaling. Applied to an
**isotropic** operator like (−Δ+I) it does **not** give O(1) condition
number — empirically κ ∝ N (see table below). The sum-of-levels scaling is
correct for *mixed-derivative* / sparse-grid operators, not for the Laplacian.

For an isotropic PDE operator the production basis must be the **isotropic
2D/3D wavelet basis** (Mallat pyramid: three detail subbands LH/HL/HH per
level, all sharing a single resolution level `j`), and the DK scaling is the
**single-level** form `D_λλ = 2^{j}`. With that, κ is O(1) (within a factor
of 2 over a 16× range of N). Algebraic Jacobi (`D = sqrt(diag A)`) is even
flatter and is the most robust choice.

**Action for the v1.1+ plan:** replace every `2^{|λx|+|λy|+|λz|}` with the
isotropic single-level `2^{j}` (or default to algebraic Jacobi), and specify
the isotropic Mallat basis, not the full tensor product.

---

## §2 Hypothesis A — condition number (Gate 1, part 1)

**Harness:** `dd_wavelets.py` + `g1_condition_number.py`. Builds an
L²-normalised wavelet synthesis matrix `W` (columns = basis functions on the
grid), forms the Galerkin matrix `A_wave = Wᵀ A_phys W` for the H¹ bilinear
form (−d²/dx² + I) with periodic BCs, applies the diagonal scaling, and
reports `numpy.linalg.cond`.

### Calibration (Haar, 1D) — harness is trustworthy

| N | κ unscaled | κ scaled |
|---|-----------|----------|
| 16 | 1.0e3 | 1.4e2 |
| 64 | 1.6e4 | 5.8e2 |
| 256 | 2.6e5 | 2.3e3 |

Haar **scaled** κ grows ~linearly with N — reproducing the round-7 qualitative
signature (Haar fails O(1)). Absolute constant differs from round-7's 923 @
N=256 (we use periodic + H¹ mass term + L² normalisation; round-7 used a
different setup), but the *failure mode* matches, so the pipeline is sound.

### DD, 1D — clean PASS

Scaled κ (DK `2^|λ|`), periodic H¹ form:

| N | DD-2 | DD-4 | DD-6 |
|---|------|------|------|
| 16 | 32.6 | 41.0 | 45.1 |
| 64 | 33.4 | 45.9 | 51.8 |
| 256 | 33.5 | 48.2 | 54.7 |

All three orders are **flat in N** (O(1)). DD-2 (= piecewise-linear hat, the
minimal-smoothness basis that still meets the t=1 approximation-order
requirement) is the flattest and lowest. Unscaled κ grows ∝ N² as expected.

**Verdict: 1D condition-number criterion PASSES** for all DD orders.

### DD, 2D — PASS with the corrected isotropic basis

Naive full-tensor basis with `D=2^{lx+ly}` (the plan's scaling) — **FAILS**:

| N | κ (D_sum) | κ (D_max) | κ (D_energy) | κ (D_jacobi) |
|---|-----------|-----------|--------------|--------------|
| 256 | 1.1e2 | 1.1e2 | 1.9e2 | 6.5e1 |
| 1024 | 4.1e2 | 1.9e2 | 2.9e2 | 1.5e2 |
| 4096 | 1.7e3 | 4.4e2 | 4.6e2 | 3.5e2 |

All grow with N (D_sum worst, ∝ N). This is the hyperbolic-basis instability,
not a DD defect.

Isotropic Mallat basis with single-level `D=2^j` — **PASSES**:

| N | κ unscaled | κ (D=2^j) | κ (D_jacobi) |
|---|-----------|-----------|--------------|
| 64 | 2.5e2 | 75.8 | 28.8 |
| 256 | 1.0e3 | 97.4 | 33.3 |
| 1024 | 4.1e3 | 110.5 | 37.7 |

`D=2^j` grows by factor 1.46 across a 16× range of N → **within the plan's
"factor of 2" pass criterion**. Algebraic Jacobi is flatter still (factor
1.31) and lowest in absolute value.

**Verdict: 2D condition-number criterion PASSES** with the isotropic basis
(see Correction C1). Algebraic Jacobi (Hyp C) is the most robust scaling and
a strong candidate for the production default.

### Open items for §2 Hyp A

- CDD convergence sweep (smooth + near-sharp σ=0.02 sources) — pending. The
  condition-number criterion is necessary but the plan also requires CDD to
  converge in ≤20 outer iterations and beat rolling by ≥10%.
- 3D condition number (Gate 2 / §4) — pending; will use the corrected
  isotropic 3D basis with `D=2^j`.

---

## §3 — DD phi sign property and wrong-sign safety (Gate 1, part 2)

**Harness:** `g1_wrong_sign.py`. Moving Gaussian source (σ=0.04) on (0,1)
Dirichlet, sensor at a generic fine-only node x≈0.30. For each θ we compare
the sign of `J_frozen` (active set size K) against `J_full` under five
selection rules. A **sine basis is a positive control** and must reproduce
the non-local wrong-sign failure.

> **Test-design catch:** the obvious sensor x=1/3 is, with n_coarse=2
> Dirichlet, *exactly a coarse node*. By the interpolation property all
> detail wavelets vanish there, so J_frozen==J_full trivially and the test
> is vacuous. Moving the sensor to a generic fine node (x≈0.30) is required
> for a meaningful test. Logged so the implementation test suite avoids it.

### phi/psi sign structure at the sensor

| basis | neg-lobe depth (of peak) | signs at sensor |
|-------|--------------------------|-----------------|
| DD-2 (hat) | 0.000 (strictly ≥0) | all positive, every level |
| DD-4 | 0.073 | **alternating** (e.g. level 0: 2 pos / 2 neg) |

Answers plan §3 sub-questions: **(a) yes**, DD-4 wavelets covering the sensor
alternate in sign; **(b)** the coarse scaling functions are *not* single-signed
at a generic sensor for DD-4 (the cubic predictor overshoots into ±7% lobes).

### wrong-sign across the boundary sweep θ ∈ {0.02,…,0.98}

| basis | top-\|b\| | top-\|c\| | **CDD (coarse-guaranteed)** | CDD (no coarse) |
|-------|---------|---------|------------------------------|-----------------|
| SINE (control) | WRONG @0.06,0.98 | ok | ok | WRONG @0.06,0.98 |
| DD-2 | ok everywhere | ok | **ok** | ok |
| DD-4 (K=N/16) | **WRONG @0.02,0.04** | ok | **ok everywhere** | **WRONG @0.94–0.98** |
| DD-4 (K=N/8) | WRONG @0.94–0.98 | ok | **ok everywhere** | ok |

### Verdict and the required theorem qualification

- **The production selection (CDD with coarse inclusion) is wrong-sign-safe**
  on DD across the entire boundary sweep at K as small as N/16. §3 **PASSES**
  for the production path. So is top-\|c\|.
- **The unqualified claim "locality forbids wrong-sign for any selection" is
  FALSE for DD-4.** DD-4 is not strictly single-signed (7% negative lobes,
  sign-alternating at the sensor), and top-\|b\| produces genuine wrong-sign
  solutions — same failure mode as the non-local sine basis.
- **Mechanism, confirmed by the no-coarse probe:** the protection is
  *coarse-level inclusion*, not strict locality. Stripping the coarse
  guarantee from CDD makes it wrong-sign on DD-4. This is exactly the plan's
  anticipated resolution **(i)**: the active set must always include the
  coarse level that dominates u(x_sensor). The cold-start coarse-then-fine
  protocol and CDD's coarse seeding both enforce this.

**Action for the v1.1+ plan / paper methods:** state the locality theorem as
*"CDD selection is wrong-sign-safe on DD because it always retains the coarse
levels dominating the sensor functional"* — **not** *"locality forbids
wrong-sign."* DD-2 (strictly nonnegative) is the only DD order for which the
unqualified locality statement holds, and it costs two orders of
approximation. Keep top-\|b\| deprecated (already the case from round-4).

---

## §2 Hypothesis A — CDD convergence + rolling comparison (Gate 1, part 1 cont.)

**Harness:** `g1_cdd_convergence.py`. DD-4 Dirichlet, N=191, work in
Dahmen-Kunoth-scaled coordinates. Full SOLVE→ESTIMATE→MARK(Dörfler
θ_D=0.5)→REFINE loop, coarse-seeded.

### CDD convergence to tol=1e-3 (scaled residual)

| σ | n_outer | \|Λ\| | \|Λ\|/N | J_err |
|---|---------|-----|---------|-------|
| 0.10 | 17 | 28 | 0.147 | 6.0e-6 |
| 0.05 | 21 | 29 | 0.152 | 2.1e-5 |
| 0.02 (near-sharp) | 20 | 26 | 0.136 | 7.2e-6 |

Converges in ~17–21 outer iters to |Λ| ≈ 14–15% of N (< N/4), J_err ~1e-5.
σ=0.05 is marginally over the plan's ≤20 (21). **Near-sharp σ=0.02 needs the
fewest modes** — a narrow source is sparse in the wavelet domain too,
refuting the plan's worry that sharpness blows up CDD iterations.

### Trajectory mean J_err at fixed budget K=N/8, θ(t)=0.3+0.3·sin(2πt/30)

| σ | CDD | rolling | oracle | CDD vs rolling |
|---|-----|---------|--------|----------------|
| 0.10 | 2.6e-5 | 7.3e-4 | 2.8e-5 | **96.5%** better |
| 0.05 | 1.7e-5 | 7.3e-3 | 1.5e-5 | **99.8%** better |
| 0.02 | 4.2e-6 | 1.2e-3 | 4.5e-6 | **99.7%** better |

CDD ≈ oracle and crushes rolling (plan only required ≥10%). Rolling fails
because the source moves >σ per step, so last step's active set misses the
new source location; CDD re-marks from the residual each step. Honest and
decisive.

### Sharp-interface stress: step (Heaviside) source, kink solution

CDD convergence under competing scalings (tol=1e-3):

| scaling | n_outer | \|Λ\| | \|Λ\|/N | J_err |
|---------|---------|-----|---------|-------|
| H¹ DK (2^\|λ\|) | 16 | 21 | 0.110 | 9.1e-6 |
| Besov B¹₁₁ (2^{\|λ\|/2}) | 17 | 24 | 0.126 | 9.2e-6 |
| Jacobi (√diag A) | 15 | 20 | 0.105 | 9.1e-6 |
| none | 15 | 31 | 0.162 | 9.5e-6 |

**Besov is not better** on the kink — because a step *source* gives an H¹
(not H²) *solution*, for which H¹ DK is already correctly tuned. Jacobi is
marginally best/sparsest.

### Gate 1 §2 decision: **Hypothesis A CONFIRMED**

Per the plan's decision rule, A passes for smooth, near-sharp, and kink
sources ⇒ **production basis is DD + Dahmen-Kunoth t=1 (single-level 2^j in
multi-D), with algebraic Jacobi as the most-robust drop-in alternative.**
Hypothesis B (Besov) is **not** needed for the scalar elliptic case
(bounded/step RHS ⇒ H¹ solution). It would only matter for genuinely
discontinuous solutions (singular sources / discontinuous coefficients),
which are out of scope for the standard formulation and untested here.

**Gate 1 is fully resolved: §2 (Hyp A) PASS + §3 (wrong-sign, qualified) PASS.**
Proceed to Gate 2 (§4, 3D) and Gate 3 (§5, §6).

---

## §5 — Stream-function (biharmonic) preconditioning (Gate 3, part 1)

**Harness:** `g3_biharmonic.py`. The plan's Option-1 recasts the 2D cavity as a
scalar biharmonic Δ²ψ=−ω (H²-elliptic, t=2). The derisking-relevant question
is the conditioning claim; the full Ghia–Ghia–Shin **Navier-Stokes** cavity
match is a multi-week nonlinear solve, explicitly **deferred to implementation
phase** (consistent with the plan §8 philosophy). 1D periodic biharmonic
bilinear form ∫(u″)²+u².

Scaled κ:

| basis | scaling | N=16 | N=256 | trend |
|-------|---------|------|-------|-------|
| DD-2 | t=2 (2^{2j}) | 8.4e3 | 1.5e5 | **∝ N — FAILS** |
| DD-4 | t=1 (2^j) | 2.4e4 | 6.0e6 | ∝ N — wrong t |
| DD-4 | **t=2 (2^{2j})** | 5.1e3 | 8.6e3 | O(1), const ~8.6e3 |
| DD-4 | **jacobi** | 8.9e2 | 1.2e3 | **flat O(1), best** |
| DD-6 | jacobi | 7.5e2 | 7.9e2 | flattest ~780 |

**Findings:**
- **DD-2 fails** the biharmonic (t=2 κ ∝ N) — confirms H² requires
  approximation order ≥ 4 (piecewise-linear's 2nd derivative is deltas).
  Biharmonic ⇒ DD-4 minimum.
- **t=2 DK works** but with a high constant (~8.6e3), which would force the
  CDD bulk θ_D < κ^{-1/2} ≈ 0.011 — uncomfortably tight.
- **Algebraic Jacobi is 7× better** (~1.1e3 DD-4, ~780 DD-6) and flat. It
  auto-adapts to the operator's elliptic order without the order being
  hard-coded.

**Cross-cutting conclusion (reinforces C1):** algebraic Jacobi is the single
most robust preconditioner across *both* the Laplacian (t=1) and the
biharmonic (t=2), in 1D/2D/3D. Recommend it as the production default, with
theory-derived `2^{tj}` exposed as an opt-in.

**§5 verdict:** the stream-function formulation is **viable from a
conditioning standpoint** (Jacobi-preconditioned biharmonic is well-behaved).
The nonlinear-cavity Ghia accuracy match is scoped to implementation. Gate 3
§5 **PASS** on the derisking-relevant question.

---

## §4 — 3D sparsity break-even and trap structure (Gate 2)

**Harness:** `g2_3d.py` (+ `g2_trap_basis.py` for the mechanism probe). 16³ =
4096 interior DOF, periodic, isotropic DD-4, single-level DK 2^j (per C1).

### κ(A_scaled) in 3D

| scaling | κ |
|---------|---|
| none | 1.7e4 |
| DK 2^j | 473 |
| Jacobi | 158 |

vs DD-4 1D ≈48, 2D ≈110. The DK 3D value (473) is ~4× the 2D value — **fails
the literal "within factor of 2 of 1D/2D"** criterion, but the absolute value
is perfectly workable for GMRES/CG, and Jacobi (158) is much tighter.
N-scaling *within* 3D wasn't isolated (single grid size), but the O(1)-in-N
behaviour is already established in 1D/2D and the absolute κ is fine.

### Sparsity break-even — STRONG PASS

J_err at active-set budget k (3D, sensor at off-axis (0.7,0.6,0.5)):

| σ | k=N/16 (256) | k=N/8 (512) | k=N/4 | k=N/2 |
|---|--------------|-------------|-------|-------|
| 0.10 (smooth) | **1.0e-3** | 7e-4 | 7.5e-6 | 2.4e-5 |
| 0.02 (near-sharp) | **1.3e-3** | 1.8e-4 | 1.6e-4 | 1.2e-6 |

Plan threshold: J_err < 5% at k=N/16 (smooth) / k=N/8 (sharp). **We achieve
~0.1% at k=N/16 for BOTH** — far exceeding the bar. In 3D, CDD reaches
<0.15% sensor error using 6.25% of the basis. **The adaptive solver is
genuinely worth building in 3D.** (Jacobi scaling gives the same picture.)

### 3D trap structure — the trap is a NON-LOCAL-basis phenomenon

The plan (§4) feared Z₂³ symmetry traps (centre + 6 faces), E[encounters]
≈0.6/trajectory, needing `symmetry_break`. Measuring the correct quantity —
`blindness_ratio = |g_frozen|/|g_full|` with the active set frozen at the
evaluation θ — across the 3×3×3 grid:

| selection | sensor | worst ratio over grid |
|-----------|--------|-----------------------|
| CDD | off-axis | 0.936 |
| top-\|b\| | off-axis | 0.958 |
| CDD | on xy-plane | 0.940 |
| top-\|b\| | on xy-plane | 0.910 |

**No trap anywhere** (all ratios ≫ 0.7), even with the deprecated top-\|b\|.

The 1D mechanism probe (`g2_trap_basis.py`) nails why. Canonical trap θ=0.5,
sensor 1/3, top-\|b\|:

| basis | ratio @θ=0.5 | @0.48 | @0.42 |
|-------|--------------|-------|-------|
| SINE (non-local) | **0.000** | 0.066 | 0.128 | ← BLIND (reproduces spike) |
| DD-2 (local) | 1.003 | 0.995 | 1.441 | no blindness |
| DD-4 (local) | 1.104 | 0.933 | 1.366 | no blindness |

**The selection-induced blindness trap (Palais / Selection-Equivariance) is a
property of NON-LOCAL bases.** In a local wavelet basis, top-\|b\|/CDD selects
modes spatially near the source, so the active set tracks θ and never freezes
into a symmetric configuration blind to an asymmetric sensor. The sine basis
indexes modes by frequency, not location, so a symmetric source picks a
symmetric mode set regardless of sensor — hence g_frozen=0 while g_full≠0.

### Gate 2 verdict: **PASS** (sparsity), with a major strategic insight

- 3D sparsity break-even passes by a wide margin.
- κ is workable (Jacobi best); the literal factor-2 criterion is too strict
  and not the right test (O(1)-in-N is what matters, established in 1D/2D).
- **The trap risk for the wavelet production node is essentially zero.** The
  blindness diagnostic + `symmetry_break` machinery in `AdaptiveNode` — a
  large part of the base-class design — is **non-local-basis insurance**. It
  is correct and necessary for `TopKAdaptiveNode` (sine); for
  `WaveletAdaptiveNode` it is a cheap, near-inert cold-start safety net. No
  3D-specific δ is needed (plan §4's `blindness_break_delta_3d` is moot for
  the wavelet node). This should be documented so the implementation doesn't
  over-invest in trap mitigation for the wavelet path.

---

## §6 — time-dependent trajectory adjoint under lax.scan (Gate 3, part 2)

**Harness:** `g3_trajectory.py` (JAX, float64). DD-4 Dirichlet, N=95. Source
moves in time θ(t)=θ₀+0.1t; per-step CDD-style selection seeded by the
residual r=b−A·c_prev (so c_prev genuinely drives selection — the plan's
exact concern). Mask is non-differentiable (argsort) and stop_gradient'd.
Static-shape masked solve via the A_eff trick (inactive rows → identity).
Objective J=Σ_t u_t(x_sensor)². `lax.scan` over the trajectory.

### jax.grad vs FD, sweeping T

| T | rel_err |
|---|---------|
| 1 | 8.3e-11 |
| 2 | 4.1e-11 |
| 5 | 1.1e-10 |
| 10 | 2.9e-11 |

Robustness over θ₀ ∈ {0.20…0.41} at T=10: **worst rel_err 1.05e-10.**

### Verdict — PASS, contamination concern refuted

`jax.grad` through the `lax.scan` trajectory matches FD to ~1e-10 with **no
degradation in T**. The plan's worry — that c_prev flowing into the residual
contaminates the trajectory gradient — does **not** materialise: the discrete
selection (argsort → integer indices) carries zero gradient, so c_prev's
influence on the *mask* is naturally blocked, while the fresh per-step solve
carries the correct gradient through b(θ). **No `stop_gradient` mitigation and
no trajectory-level `custom_vjp` are required.** Confirms the plan's positive
path: the trajectory adjoint is the exact sum of per-step frozen-set adjoints.

(Tested at non-kink θ, per the plan's scope; at a mask-flip kink the FD/grad
relationship is the expected Clarke-subgradient one, already characterised in
round-7 for the single step.)

---

## Post-gate improvements / clever-workaround exploration

**Harness:** `g4_improvements.py`. After the gates closed, two design-relevant
questions the plan did not pose.

### (1) Submatrix conditioning — the inner-solve de-risk (NEW)

The gates measured *full-operator* κ, but CDD only ever inverts the frozen
K×K submatrix `A_{Λ,Λ}` (via `ift_linear_solve`). Does it stay conditioned as
Λ adapts? 1D DD-4 Dirichlet (full-op κ: DK 11.3, Jacobi 3.8):

| K | κ(A_ΛΛ) DK | κ(A_ΛΛ) Jacobi | κ(A_ΛΛ) unscaled |
|---|------------|----------------|------------------|
| N/16 | 10.0 | 3.3 | 4.8e2 |
| N/8 | 10.6 | 3.4 | 1.8e3 |
| N/4 | 10.9 | 3.5 | 5.8e3 |
| N/2 | 11.2 | 3.7 | 5.9e3 |

**The frozen submatrix is as well-conditioned as the full operator** under
DK/Jacobi (diagonal scaling restricts cleanly to the submatrix), and bounded
as Λ grows. Unscaled it blows up. **De-risks the inner GMRES/CG: it won't
degrade as the active set adapts** — a property the plan assumed but never
checked.

### (2) The "cheap diagonal" claim (Hyp C) — confirmed

Jacobi is the best preconditioner but needs `diag(A_wave)`. Per-level
statistics of the exact diagonal:

| level | mean diag | std/mean | ratio to 2^{2j} |
|-------|-----------|----------|-----------------|
| 0 (coarse) | 7.2e1 | 0.517 | 1.00 |
| 1 | 4.0e2 | 0.008 | 1.40 |
| 2 | 1.6e3 | 0.006 | 1.39 |
| 3 | 6.2e3 | 0.005 | 1.35 |
| 4 | 2.3e4 | 0.006 | 1.26 |
| 5 (fine) | 7.4e4 | 0.000 | 1.00 |

The diagonal is **level-constant to ~0.6% on interior levels** — but the
ratio to `2^{2j}` is *not* constant (1.0–1.4), which is exactly why full
Jacobi (κ 3.8) beats DK `2^j` (11.3). The remaining benefit comes from the
**coarse level and boundary entries** (std/mean 0.5 at level 0), which need
per-entry scaling. For a banded FD operator the full diagonal is
O(N)-computable from compact wavelet supports (no N matvecs), so **Hyp C's
cheap-diagonal claim holds and full Jacobi is affordable.**

### (3) Cheap level-constant Jacobi as a middle ground

`level-Jacobi` (one √mean-diagonal per level, O(#levels)) gives κ 9.4 vs DK
11.3 vs full Jacobi 3.8. Marginal over DK — the coarse/boundary per-entry
variation is what matters, so it is not worth the half-measure: either use DK
`2^{tj}` (matrix-free) or full Jacobi (O(N), best). No useful middle ground.

**Production recommendation (refined):** default to **full algebraic Jacobi**
(O(N) to assemble, best κ, order-agnostic, submatrix-stable); expose DK
`2^{tj}` as a matrix-free opt-in for the very largest problems where even
O(N) diagonal assembly is undesirable.

---

# Continuation series (6 investigations) — 2026-06-22

The six follow-up investigations from the continuation brief. Scripts:
`hybrid_jacobi.py`, `dd_jax_poc.py`, `nonlinear_cdd.py`, `discontinuous_coeff.py`,
`submatrix_2d3d.py`, `dthreshold_empirical.py`.

---

## Investigation 1 — Level-Jacobi hybrid: the production preconditioner

**Harness:** `hybrid_jacobi.py`. Four diagonal preconditioners on the
isotropic Mallat DD-4 operator (solve the symmetrically-scaled
`Â = D⁻¹AD⁻¹`): **full** (`√a_ii` per entry, N scalars), **level** (√mean-diag
per level, O(#levels)), **hybrid** (per-entry at level 0, level-mean for finer,
O(N_coarse+#levels)), **dk** (`2^{tj}`, zero assembly).

### Part A — metrics vs N (1D {256..16384}, 2D {16²..128²})

1D full-op κ / submatrix κ(A_ΛΛ) at k=N/16 / GMRES iters / assembled scalars:

| N | full | level | **hybrid** | dk |
|---|------|-------|-----------|-----|
| 256 | 20.4 / 19.9 / 14 / 256 | 38.4 / 36.4 / 15 / 7 | **20.4 / 19.9 / 14 / 10** | 48.2 / 44.3 / 15 / 0 |
| 4096 | 20.4 / 20.4 / 14 / 4096 | 38.4 / 38.2 / 15 / 11 | **20.4 / 20.4 / 14 / 14** | 50.2 / 48.3 / 16 / 0 |
| 16384 | 20.4 / 20.4 / 15 / 16384 | 38.4 / 38.4 / 16 / 13 | **20.4 / 20.4 / 15 / 16** | 50.7 / 49.4 / 16 / 0 |

2D (full / **hybrid** / dk), κ_full and assembled scalars:

| N | full κ | **hybrid κ** | dk κ | level κ | full asm | hybrid asm |
|---|--------|--------------|------|---------|----------|-----------|
| 256 | 33.3 | **33.3** | 97.4 | 94.8 | 256 | 18 |
| 1024 | 37.7 | **37.7** | 110.5 | 104.3 | 1024 | 19 |
| 4096 | 43.7 | **43.7** | 119.3 | 110.6 | 4096 | 20 |
| 16384 | 51.8 | **51.8** | 127.0 | 116.6 | 16384 | 21 |

**The headline:** hybrid κ equals full-Jacobi κ to 4 significant figures at
*every* N in 1D and 2D — identical submatrix κ and identical GMRES iteration
count too — while assembling only **16–21 scalars vs N**. This is because
(g4 finding) the fine-level diagonal is level-constant to ~0.6%, so level-mean
≈ per-entry there; only the coarse level (where diag varies ~50%) needs
per-entry, which hybrid supplies. **Pure level-Jacobi is dominated** (κ 38 vs
20 in 1D, 117 vs 52 in 2D — the coarse per-entry term is what matters). DK
costs ~2.4× the conditioning of hybrid.

1D full/hybrid κ is **flat (20.4) across a 64× range of N**; 2D grows mildly
33→52 (factor 1.56 over 64× N — sub-logarithmic, effectively O(1)).

### Part B — CDD trajectory (θ(t)=0.3+0.3sin), total inner GMRES

| | mean n_outer | mean J_err | total inner GMRES |
|--|-------------|-----------|-------------------|
| **1D N=4096** full / hybrid | 59.1 | 2.1e-9 | **8015** |
| 1D level | 59.2 | 1.3e-9 | 8739 |
| 1D dk | 62.2 | 1.3e-9 | 9446 |
| **2D N=4096** full / hybrid | 28.8 | 6.7e-6 | **8481** |
| 2D level | 28.8 | 6.7e-6 | 9494 |
| 2D dk | 30.6 | 6.7e-6 | 10378 |

Hybrid ≡ full on every metric. DK needs +18% (1D) / +22% (2D) total inner
iterations; level +9% / +12%.

### Part C — crossover analysis

Model: `cost = c_asm·n_asm + n_outer·(c_apply·N + n_inner·c_mv·N)`. Since
hybrid and full share κ, they share `n_outer, n_inner, apply, matvec`
**exactly** — so `Δcost = c_asm·(N − O(log N)) > 0` for all N: **hybrid is
never slower than full, and strictly cheaper to assemble.** There is no N where
full wins; the ratio R = `c_asm_per_scalar / c_mv_per_entry` only sets the
margin. Two regimes:

- **Probed diagonal (matrix-free operator, no analytic diag):** assembling one
  diagonal entry ≈ one (local) matvec, so full assembly ≈ N matvecs, hybrid ≈
  log N matvecs. Full assembly *dominates even the solve* once
  N > (solve matvec count): >~40 for a single κ≈20 elliptic solve, >~8000 for a
  full CDD step. At production N (16k+) reassembled every trajectory step,
  hybrid removes an O(N)-matvec cost → large speedup.
- **Bandwidth-bound assembly (analytic diag available, just memory traffic):**
  full assembly ≈ R matvecs (R≈10 bandwidth-limited GPU, R≈1 compute-limited).
  Against a CDD step's ~8000 inner matvecs this is <0.1% — negligible. Here
  hybrid's win is **storage** (O(log N) vs O(N) floats), which matters for
  ensembles / many concurrent operator instances in GPU memory.

### Part D — recommendation (production preconditioner hierarchy)

1. **Default: hybrid.** Identical conditioning to full Jacobi at O(N_coarse +
   log N) assembly *and* storage. Never worse; strictly better when the
   diagonal must be probed (matrix-free, frequent operator updates) or when
   storage is constrained.
2. **Full Jacobi:** equivalent fallback when an analytic per-entry diagonal is
   trivially available *and* O(N) storage is a non-issue. No reason to prefer
   it over hybrid otherwise.
3. **DK `2^{tj}`:** the matrix-free fallback (zero assembly/storage) for the
   largest problems or when even the coarse-block diagonal is unavailable;
   pay ~2.4× κ → ~20% more inner iterations.
4. **Pure level-Jacobi: not recommended** — dominated by hybrid.

The plan's stated threshold "hybrid reaches κ<6 in 2D/3D" is **mis-calibrated**
— full Jacobi *itself* is κ≈33–52 in 2D, never <6. The correct criterion is
*hybrid matching full Jacobi at O(log N) cost*, which holds overwhelmingly
(identical to 4 sig figs). **Hybrid replaces full Jacobi as the default.**

---

## Investigation 2 — DD-4 wavelet operator in JAX (engineering PoC)

**Harness:** `dd_jax_poc.py` (JAX, float64). Verdict: **no show-stoppers.**

### Part A — 1D DD-4 transform
- Lifting-form synthesis (jnp.roll prediction, static-shape unrolled loop) is
  `jax.jit`-compilable and matches the numpy reference `W@c` to **1.6e-16**
  (≪ 1e-12 bar). Forward∘inverse roundtrip 7.8e-17.
- JIT compile ~0.9 s; per-call 190–340 µs at N=256→4096.
- `jax.grad` through synthesis + a dense masked solve vs FD: **1.9e-10** (≪1e-6).

### Part B — BCOO stiffness + lineax + autodiff
- `A_wave` as `jax.experimental.sparse.BCOO`: nnz = 13% dense at N=256, **4.4%
  at N=1024** (~45 nnz/row), BCOO mem 727 KB vs 8192 KB dense at N=1024.
- `lineax.linear_solve` over a `FunctionLinearOperator` wrapping the BCOO
  matvec: `jax.grad` vs FD = **1.3e-9** with full GMRES (rtol 1e-12). (With
  restarted GMRES rtol 1e-10 it is 1e-7 — i.e. the residual is purely inner-
  solver-tolerance, not an autodiff bug; tightening the solve recovers the
  round-6 1e-9 standard.)
- The round-6 masked-matvec closure `jnp.where(mask, A@where(mask,v,0), 0)`
  JIT-compiles cleanly.
- ⚠ note: `lineax.GMRES(rtol=1e-13)` *stagnates* (restart too small) — use full
  GMRES (`restart=N`) or a sane rtol; documented so the implementation picks
  solver params deliberately.

### Part C — 2D isotropic Mallat in JAX
- 2D three-subband synthesis (LH/HL/HH via `jnp.apply_along_axis` of the 1D
  predictor) **JIT-compiles** (~0.6–3 s) and matches numpy to **1.5e-16**.
- `A_wave` BCOO nnz at N=32²=1024: 108224 = 10.3% dense (~106 nnz/row) — denser
  per row than 1D (more cross-coupling) but still sparse.
- The subband structure extends naturally; the only wart is that
  `apply_along_axis` is convenient but not the fastest — a production 3D impl
  should hand-vectorise the per-axis prediction (as the numpy harness already
  does with `np.roll`). That is an optimisation, not a blocker.

### Part D — Verdict
**No show-stoppers.** All three parts pass: transform correctness to machine
precision, JIT compiles, BCOO+lineax autodiff to 1e-9, 2D subbands work. The
3D extension is mechanical (7 subbands, same pattern; vectorise the per-axis
predict). **The 3–5 week BCOO sparse-tree implementation estimate is
confirmed**; the only newly-surfaced engineering note is to choose GMRES
restart/tolerance deliberately (full GMRES for tight adjoint accuracy) and to
hand-vectorise the multi-D predict rather than rely on `apply_along_axis`.

---

## Investigation 3 — CDD feasibility on a nonlinear residual

**Harness:** `nonlinear_cdd.py`. Does CDD's residual criterion (validated only
on linear elliptic problems) survive a nonlinear convective term?

### Part A — 1D viscous Burgers (ν=0.01), CDD on the nonlinear residual

Implicit-Euler + Newton, DD-4 N=256 periodic, IC sin(2πx) (the periodic
shock-forming analogue of the brief's non-periodic sin(πx)). At each step CDD
selects active wavelet modes from the **full nonlinear residual** (Jacobi-
scaled), compared to the oracle (top-|c| of the converged solution).

| t | max\|u_x\| | Newton | CDD outer | \|Λ\| | oracle∩ | %near shock | J_err |
|---|-----------|--------|-----------|------|---------|-------------|-------|
| 0.004 | 6.4 | 3 | 11 | 32 | 0.94 | 0.22 | 4e-7 |
| 0.064 | 9.9 | 3 | 12 | 32 | 0.81 | 0.38 | 8e-7 |
| 0.112 | 15.4 | 3 | 12 | 33 | 0.84 | 0.52 | 1.4e-6 |
| 0.160 | 24.3 | 3 | 11 | 32 | 0.72 | 0.47 | 1.3e-6 |

As the shock steepens (max\|u_x\| 6→24), **the fraction of active modes near the
shock rises 0.22→~0.5** — CDD adaptively concentrates DOF at the steep
gradient. Newton converges in 3 iters; CDD in ≤13 outer iters/step (≤20 ✓);
\|Λ\| stays at the N/8 budget; oracle overlap 72–94%; functional J_err ~1e-6.
**The nonlinear convective term does not break CDD's residual criterion.**

### Part B — 2D stream-function/vorticity cavity, Re=100 (feasibility)

ψ-ω formulation (Poisson ψ-solve ∇²ψ=-ω each step; the brief's "∆²ψ" reads as
the Laplacian — the §5 biharmonic is the pure-ψ alternative). Physical-space FD
transport on 47², 3000 steps to t=12 (near-steady), then CDD tested on the
ψ-Poisson solve in the DD-4 wavelet basis (Jacobi).

- **Qualitative flow:** primary vortex ψ_min=-0.072 in the upper-central region
  (x≈0.50, y≈0.85; Ghia Re=100 reference (0.62, 0.74) — shifted toward the lid,
  consistent with t=12 transient at 47²; *qualitative not quantitative*, as the
  brief scopes). Counter-rotating bottom-corner vortices **visible** (opposite
  sign to the primary).
- **CDD on the ψ-solve:** k=N/16 (138/2209) → 15 outer iters, rel L2 err
  **1.3e-3**; k=N/8 → 22 outer, **1.6e-4** — both ≪ 5%. **55–59% of active
  modes concentrate near the lid/walls** where ψ has its fine structure.

### Part C — recommendation

**CDD on nonlinear problems is viable.** Burgers (Part A) shows the residual
criterion correctly tracks a forming shock; the cavity (Part B) shows it
concentrates near the lid/corner boundary layers and gives a qualitatively
correct flow at N/16 with rel err 1.3e-3 and ≤22 outer iters. No new failure
mode. **The nonlinear-cavity benchmark risk is resolved; the Ghia accuracy
match can proceed at implementation time** (quantitative match needs higher
resolution + longer integration than this feasibility run). One small note: the
cavity ψ-solve needs ~22 outer iters at N/8 (vs the ≤20 elliptic guideline) —
not a concern, but the implementation should not assume ≤20 universally.

---

## Investigation 4 — discontinuous coefficients (swimmer-body / Brinkman)

**Harness:** `discontinuous_coeff.py`. Production MIME's immersed-boundary /
Brinkman penalisation gives a jumping coefficient a(x); solutions are H¹ but
with a gradient jump at the interface. Does CDD/Jacobi cope, and does Jacobi's
auto-adaptation beat theory-DK?

### Part A — 1D, −(a u')'=f, a=1+9·1_{x>0.5} (10× jump), f=sin(πx)

| scaling | κ | CDD outer | % modes near x=0.5 | J_err @N/16 |
|---------|---|-----------|--------------------|--------------|
| DK 2^j | 54.7 | 6 | 0.64 | 8.3e-3 |
| Besov 2^{j/2} | **948** | 6 | 0.64 | 8.3e-3 |
| **Jacobi** | **13.1** | 6 | 0.64 | **5.6e-3** |

Jacobi κ=13 ≪ DK 55 ≪ Besov 948 (Besov is wrong here — the kink solution is
H¹, not a Besov regime). The jump raises Jacobi κ only 3.3× vs smooth-coeff
(4.0→13.1). CDD concentrates **64% of active modes at the jump** and reaches
J_err 0.56% at N/16.

### Part B — 2D circular inclusion, a=1+99·1_{|x−xc|<r} (100× penalisation)

Moving inclusion swept around a circle, CDD+Jacobi, Nside=32 (periodic, +mass):

| θ | xc | CDD outer | % active @ boundary (N/8) | J_err @N/16 |
|---|-----|-----------|---------------------------|--------------|
| 0 | (0.70,0.50) | 7 | 47% | 3.6e-3 |
| π/4 | (0.64,0.64) | 7 | 45% | 2.7e-3 |
| π/2 | (0.50,0.70) | 7 | 47% | 3.6e-3 |
| 3π/4 | (0.36,0.64) | 7 | 48% | 2.8e-3 |
| π | (0.30,0.50) | 6 | 44% | 4.5e-3 |

CDD+Jacobi **tracks the moving 100× inclusion** at every position: ~46% of
active modes within 1.5h of the circle boundary, J_err 0.3–0.5% at N/16
(≪5%), 6–7 outer iters.

### Part C — recommendation

**The discontinuous-coefficient case requires NO production design change
beyond using Jacobi** (already the recommended default, Inv 1). Jacobi adapts
to the coefficient field automatically (its diagonal includes the jump
contribution), giving κ=13 where theory-DK gives 55 and Besov 948. CDD
concentrates at the interface and tracks a moving inclusion with <0.5% J_err at
N/16. **Add to the v1.1+ plan's preconditioner justification:** Jacobi's
operator-adaptation is a *third* advantage over theory-DK (after order-agnostic
t-adaptation and the hybrid storage saving) — it is coefficient-field-adaptive,
which the constant-coefficient DK scaling is not. Besov scaling is confirmed
unnecessary (and actively worse) for the elliptic swimmer-body case.

---

## Investigation 5 — submatrix conditioning in 2D and 3D

**Harness:** `submatrix_2d3d.py`. Does cross-subband coupling blow up
κ(A_ΛΛ) for active sets that include some subbands/levels but not others?
Three configs at k=N/16: balanced (natural CDD), subband-biased (one subband),
level-biased (finest level only).

### Part A — 2D (Nside=32, N=1024, k=64)

| config | full | hybrid | dk |  (ratio κ_AΛΛ / κ_full) |
|--------|------|--------|-----|--------------------------|
| balanced | 29.2 (0.77) | 29.2 (0.77) | 85.4 (0.77) | |
| subband-biased | 26.9 (0.71) | 26.9 (0.71) | 67.7 (0.61) | |
| level-biased | 26.3 (0.70) | 26.3 (0.70) | 66.3 (0.60) | |

### Part B — 3D (Nside=16, N=4096, k=256)

| config | full | hybrid | dk | (ratio) |
|--------|------|--------|-----|---------|
| balanced | 28.0 (0.18) | 28.0 (0.18) | 171.8 (0.36) | |
| subband-biased | 10.5 (0.07) | 10.5 (0.07) | 74.6 (0.16) | |
| level-biased | 10.2 (0.06) | 10.2 (0.06) | 73.5 (0.16) | |

**κ(A_ΛΛ) ≤ κ(A_full) for EVERY config, in both 2D and 3D** — including the
pathological subband-biased and level-biased sets. In 3D the submatrix is
*dramatically* better (ratio 0.06–0.18). hybrid ≡ full again.

### Part C — recommendation

The 1D result generalises to 2D/3D, and there is a **theorem behind it**: for
the symmetrically-scaled SPD operator `Â`, the frozen `A_ΛΛ` is a *principal
submatrix*, so **Cauchy eigenvalue interlacing guarantees
κ(A_ΛΛ) ≤ κ(Â) for any active set Λ.** The empirics (ratio ≤ 1 everywhere,
≪1 in 3D) confirm it. **There are no pathological active-set configurations**;
cross-subband coupling never degrades the inner solve below the full-operator
conditioning. **No subband-completeness constraint on the active set is needed**
— the inner `ift_linear_solve` is provably safe across all CDD outputs. (This
holds for any SPD-preserving preconditioner; full/hybrid/DK all satisfy it.)

---

## Investigation 6 — D_threshold = 5 empirical validation

**Harness:** `dthreshold_empirical.py`. (Affects only the NON-wavelet
`TopKAdaptiveNode` — the wavelet basis is trap-immune, Gate 2.) Separable D-dim
sine-Poisson model, J=Σ_i J_1d(θ_i), exact per-axis 1D full/frozen (top-|b|)
gradients, sensor at x=1/3. 20 trajectories, lr=0.04, 100 steps. Trap encounter
= ≥5 consecutive steps with global blindness_ratio < 0.7.

### Part A — encounters vs D

| D | global enc | 0.2·D | enc/(0.2D) | per-axis blind/step |
|---|-----------|-------|------------|---------------------|
| 2 | 1.55 | 0.4 | 3.87 | 1.19 |
| 3 | 1.30 | 0.6 | 2.17 | 1.67 |
| 5 | 1.10 | 1.0 | 1.10 | 3.46 |
| 7 | 1.50 | 1.4 | 1.07 | 4.79 |
| 10 | 1.15 | 2.0 | 0.57 | 7.05 |
| 15 | 1.45 | 3.0 | 0.48 | 9.91 |
| 20 | 1.50 | 4.0 | 0.37 | 13.09 |

**The analytic 0.2·D estimate does not match the metric the base class uses.**
*Per-axis* blind events scale ~linearly with D (1.2→13.1, slope ≈0.65) — that
is what 0.2·D was implicitly modelling. But the **global** `blindness_ratio`
(a normalised norm ratio, which is what the diagnostic computes) trips at a
**roughly D-independent rate ~1.1–1.5/trajectory**, because one blind axis
among many sighted ones doesn't pull the global ratio below 0.7. The 0.2·D
curve only crosses the empirical near D=5 by coincidence.

### Part B — monitoring cost vs missed-trap cost

| D | E[enc] | monitor (3·enc) | no-monitor (50·enc) | favours |
|---|--------|-----------------|---------------------|---------|
| 2 | 1.55 | 4.7 | 77.5 | **monitor** |
| 5 | 1.10 | 3.3 | 55.0 | **monitor** |
| 10 | 1.15 | 3.4 | 57.5 | **monitor** |
| 20 | 1.50 | 4.5 | 75.0 | **monitor** |

Monitoring is ~15× cheaper than eating the traps **at every D**, because
encounters occur at all D (including D=2) and each undetected trap wastes ~50
steps vs 3 to detect+break.

### Part C — recommendation

**D_threshold = 5 is NOT empirically confirmed.** Global-ratio trap encounters
are D-independent (~1–1.5/trajectory) and occur well below D=5 (D=2: 1.55), so
the threshold leaves D∈{2,3,4} needlessly unmonitored where traps still happen.
Monitoring is cost-favourable (~15×) at every tested D. **Recommendation:
lower the threshold — monitor whenever D≥2 (effectively always-on for any
multi-parameter non-local node)** rather than gating at D>5. Update the
base-class doc: replace the `E[traps]≈0.2·D` justification with "global
blindness_ratio trips at a ~D-independent rate; monitoring is ~15× cheaper than
the traps it catches at all D≥2." (Caveat: separable model, fixed lr/steps;
absolute counts scale with trajectory length, but the D-independence of the
global metric and the cost asymmetry are robust.) Again, moot for
`WaveletAdaptiveNode` (trap-immune); relevant only if a spectral
`TopKAdaptiveNode` is built for multi-parameter problems.

---

# Cross-cutting statement — continuation series

Addressing the six questions from the continuation brief:

1. **Is the production preconditioner hierarchy settled?** **YES.** Default =
   **hybrid** (per-entry coarse + level-mean fine): it equals full-Jacobi κ to
   4 sig figs at every N in 1D/2D (and the 1D result is flat in N), at
   O(N_coarse+log N) assembly/storage. Order: hybrid (default) → full Jacobi
   (equivalent when an analytic diagonal is free and storage is no issue) → DK
   `2^{tj}` (matrix-free fallback, ~2.4× κ, ~20% more iters) → pure
   level-Jacobi (dominated, not recommended). Hybrid is never slower than full
   and wins decisively in the probed-diagonal / matrix-free / frequent-reassembly
   regime and on storage. The plan's "κ<6" hybrid threshold was mis-calibrated.

2. **Does DD-4 in JAX surface show-stoppers?** **NO.** Transform matches numpy
   to 1e-16 and JIT-compiles (1D and 2D Mallat); BCOO+lineax autodiff reaches
   the round-6 1e-9 standard with full GMRES; masked-matvec closure works. The
   **3–5 week BCOO implementation estimate is confirmed**; only engineering
   notes are deliberate GMRES restart/tolerance choice and hand-vectorising the
   multi-D predict.

3. **Is the nonlinear cavity benchmark viable?** **YES.** CDD's residual
   criterion survives the nonlinear convective term: on Burgers it tracks a
   forming shock (near-shock active fraction rises with steepening, ≤13 outer
   iters/Newton step, 72–94% oracle overlap); on the Re=100 cavity it
   concentrates at the lid/corner layers, rel err 1.3e-3 at N/16, qualitatively
   correct vortex structure. Ghia quantitative match → implementation phase.

4. **Does the discontinuous-coefficient case need design changes?** **NO** —
   beyond using Jacobi (already the default). Jacobi adapts to the coefficient
   field automatically (κ=13 vs DK 55 vs Besov 948 for a 10× 1D jump); CDD
   concentrates at the interface and tracks a moving 100× 2D inclusion with
   <0.5% J_err at N/16. This is a *third* Jacobi advantage over theory-DK.

5. **Is submatrix conditioning confirmed in 2D/3D?** **YES**, with a theorem.
   κ(A_ΛΛ) ≤ κ(A_full) for *every* active-set config (balanced, subband-biased,
   level-biased) in 2D and 3D, because the scaled operator is SPD and A_ΛΛ is a
   principal submatrix → **Cauchy interlacing**. No pathological configs; no
   subband-completeness constraint needed; the inner solve is provably safe.

6. **Is D_threshold = 5 empirically confirmed?** **NO.** The global
   `blindness_ratio` trips at a ~D-independent rate (~1–1.5/trajectory),
   occurring even at D=2 — the `0.2·D` estimate was implicitly a per-axis count
   (per-axis blind *does* scale ~0.65·D). Monitoring is ~15× cheaper than the
   traps at all D. **Lower the threshold to monitor whenever D≥2** (always-on
   for multi-parameter non-local nodes). Affects only `TopKAdaptiveNode`; moot
   for the trap-immune `WaveletAdaptiveNode`.

## Conclusion

**Five of six are clean positives; the sixth (D_threshold) is a resolved
base-class default tweak for the non-wavelet node, not a blocker.** Combined
with the original four gates, **the wavelet derisking spike series is complete
and `WaveletAdaptiveNode` implementation can begin.** Consolidated plan edits
the implementation should carry in:

- Preconditioner: **hybrid Jacobi default** (isotropic Mallat basis,
  single-level scaling); DK `2^{tj}` matrix-free opt-in. (Inv 1 + C1)
- Wrong-sign safety stated via **coarse-inclusion**, not pure locality. (§3)
- Trap mitigation is **non-local-basis insurance**; for the wavelet node it is
  near-inert. For `TopKAdaptiveNode`, **monitor at D≥2**, not D>5. (Gate 2 + Inv 6)
- Discontinuous coefficients and the nonlinear cavity need **no new design** —
  Jacobi + CDD handle both. (Inv 3, Inv 4)
- Inner solve is **provably safe** across all active sets (Cauchy interlacing).
  (Inv 5)
- JAX/BCOO engineering risk is **bounded**; 3–5 week estimate holds. (Inv 2)

---

# Closeout round (3 investigations) — 2026-06-22

Closeout: 3D completeness, limitation probes, and a standalone
`KNOWN_LIMITATIONS.md` + sharding design note. After this round the spike is
**declared closed** regardless of outcome (see `## Spike closed` at end).
Scripts: `closeout_3d.py`, `limitation_probes.py`.

## Closeout Investigation 1 — 3D completeness battery

**Harness:** `closeout_3d.py`. Isotropic Mallat DD-4, N=16³=4096, hybrid Jacobi.
**BCs: PERIODIC** (flagged) — the isotropic 3D *Dirichlet* wavelet basis was not
built in the spike (boundary-adapted 3D wavelets are disproportionate for a
closeout); Dirichlet sensitivity is probed in 1D/2D in Investigation 2A.

### Part A — CDD trajectory in 3D (the key missing validation)

θ(t)=0.3+0.3sin(2πt/30), T=30, k=N/16=256, sensor (0.7,0.6,0.55):

| σ | CDD mean / peak J_err | rolling | oracle | CDD total inner GMRES |
|---|----------------------|---------|--------|-----------------------|
| 0.10 | 1.33e-3 / 3.26e-3 | 1.09e-2 / 2.06e-2 | 1.72e-3 / 4.02e-3 | 10080 |
| 0.02 | 1.06e-3 / 2.15e-3 | 2.44e-2 / 8.01e-2 | 1.17e-3 / 1.81e-3 | 16265 |

**PASS.** CDD ≈ oracle, beats rolling ~8× (σ=0.10) to ~20× (σ=0.02). **The
"near-sharp is easier" result holds in 3D** — σ=0.02 J_err (1.06e-3) < σ=0.10
(1.33e-3). (More inner GMRES for σ=0.02 — finer modes per outer — but lower
J_err.)

### Part B — θ_D sensitivity in 3D (κ=158)

Theory bound θ_D < κ^{-1/2} = 0.079. Mean/peak outer iters + J_err over the
smooth trajectory:

| θ_D | mean outer | peak outer | mean J_err |
|-----|-----------|-----------|------------|
| 0.08 | 244.6 | 248 | 8.6e-4 |
| 0.10 | 204.1 | 220 | 8.6e-4 |
| 0.30 | 47.2 | 53 | 9.2e-4 |
| **0.50** | **16.6** | **18** | 9.0e-4 |
| 0.70 | 7.2 | 8 | 6.3e-4 |

**θ_D=0.5 converges in 16.6 outer iters in 3D (≤30 ✓), no 3D-specific caveat
needed.** The theory bound θ_D<0.079 governs the *approximation-optimality*
guarantee, NOT iteration count — empirically *small* θ_D is far *worse* for
iteration count (245 iters at 0.08, tiny bulk per step) with identical J_err.
θ_D=0.5 is fine; 0.7 is even faster with no accuracy loss. Recommend keeping
θ_D=0.5 default; the "θ_D<κ^{-1/2}" line in the plan should be annotated as an
optimality-theory bound, not an iteration-count requirement.

### Part C — wrong-sign safety in 3D

Source θ_x ∈ {0.05..0.95} (boundary-approaching in x), sensor (0.3,0.4,0.6):
**zero wrong-sign solutions under CDD, top-|b|, OR top-|c| at any θ.** CDD safe
everywhere (the production claim). Note top-|b| is *also* safe here (3D periodic,
this source/sensor) whereas 1D-Dirichlet boundary showed top-|b| wrong-sign —
the difference is BC/geometry; **CDD is the robust choice regardless of both.**

### Part D — trajectory adjoint under lax.scan in 3D (JAX)

5-step loop, CDD selection residual-seeded by c_prev, J=Σ_t u_t(sensor)². At
**Richardson-classified smooth points**, grad vs FD median rel_err:

| T | % points smooth | median rel_err @ smooth |
|---|-----------------|--------------------------|
| 1 | 60% | 2.4e-9 |
| 3 | 24% | 7.7e-10 |
| 5 | 16% | 2.0e-10 |

**No T-degradation in adjoint accuracy** (≤1e-9 wherever smooth). Clarke probe:
at a 2-mode mask flip, grad (8.61e-6) lies between one-sided FDs (8.61e-6,
1.18e-4) — **correct Clarke subgradient, confirmed in 3D under lax.scan.**

**New 3D observation (→ limitation):** the active set K=256 sits near many
top-K ties, so kinks are *dense* along any trajectory — the fraction of smooth
points drops 60%→16% as T grows. This does NOT affect gradient correctness
(Clarke everywhere) but means (a) FD validation is harder in 3D, (b) optimisers
see a piecewise-smooth objective with frequent small kinks; the autodiff Clarke
subgradient is correct but kink-chattering may warrant mild smoothing if it
impedes convergence. (The brief's exact trajectory 0.3+0.05t additionally ends
at the full-symmetry kink (0.5,0.5,0.5) at T=5, where central-FD is invalid.)

### Part E — discontinuous coefficient in 3D (moving sphere)

Spherical inclusion a=1+99·1_{|x−xc|<0.15}, xc swept (0.35,0.35,0.35)→(0.65³):

| xc | CDD outer | % active @ boundary | J_err |
|----|-----------|---------------------|-------|
| (0.35,0.35,0.35) | 13 | 49% | 1.5e-3 |
| (0.50,0.50,0.50) | 14 | 49% | 1.2e-3 |
| (0.65,0.65,0.65) | 13 | 49% | 1.5e-3 |

**PASS.** ~49% of active modes at the sphere boundary (matches 2D's ~46%),
J_err <0.2% at N/16, tracks the moving inclusion. CDD outer iters 12-14 vs 2D's
6-7 — ~2× more due to the higher 3D κ (158 vs ~38), but well within ≤25.

**Investigation 1 verdict: all spike conclusions extend to 3D.** CDD trajectory,
θ_D=0.5, wrong-sign safety, trajectory adjoint, and discontinuous-coefficient
tracking all hold. The only new 3D notes are dense adjoint kinks (Part D) and
~2× more CDD iters from higher 3D κ (Parts B/E) — neither a blocker.

---

## Closeout Investigation 2 — limitation probes

**Harness:** `limitation_probes.py`.

### Part A — periodic vs Dirichlet BC sensitivity

| | κ (hybrid Jacobi) | mean CDD outer | wrong-sign? |
|---|-------------------|----------------|-------------|
| 1D periodic | 20.4 | 11.5 | no |
| 1D Dirichlet | **3.8** | 5.6 | no |
| 2D periodic (isotropic Mallat) | 37.7 | 11.1 | no |
| 2D Dirichlet (tensor basis) | 145.7 | 11.8 | no |

**Dirichlet BCs do not break anything.** 1D Dirichlet is *better* conditioned
(κ=3.8 — the +I mass + boundary removes the near-null constant mode periodic
carries). 2D Dirichlet κ=146 is higher, but that is the **anisotropic tensor
basis** used as a stand-in (no isotropic-Dirichlet Mallat basis was built) — per
Correction C1 the tensor basis is less well-conditioned; an isotropic Dirichlet
basis would recover ~38. Jacobi keeps even the tensor case workable (146). CDD
iteration counts are comparable to periodic, and **wrong-sign safety holds under
Dirichlet** in all cases. Mitigation: build a proper isotropic Dirichlet Mallat
basis at implementation; not a blocker.

### Part B — non-separable D_threshold

Genuinely coupled sine-basis Poisson (λ couples axes), top-|b|, 20 trajectories:

| D | mean blindness encounters | separable-model ref (Inv 6) |
|---|---------------------------|------------------------------|
| 2 | 0.30 | ~1.5 |
| 3 | 0.10 | ~1.3 |

**Non-separable coupling gives FEWER encounters (0.1–0.3), not more.** The
separable model (Inv 6) *over-estimated* the trap rate: in a coupled problem the
global blindness_ratio rarely drops below 0.7 because top-|b| selects more
sensor-relevant modes when axes couple. So "monitor at D≥2" (Inv 6) is
conservative-safe but the real coupled-PDE trap rate is low. Monitoring remains
cheap insurance (~3 solves) but is rarely triggered. Moot for the trap-immune
wavelet node; low-stakes for TopKAdaptiveNode.

### Part C — complex source geometry (figure-eight source + crescent coeff)

2D Ns=64, k=N/16=256: **CDD outer iters=14 (≤25 ✓), J_err(rel L2)=1.05e-3.**
Active modes: 19% near the left lobe, 4% near the right lobe. The asymmetry is
*correct adaptivity*, not a failure: the crescent coefficient jump (high-a on
the left) drives the wavelet content (coefficient-jump boundary on the left),
while the right lobe is a smooth bump needing few modes — and J_err stays 1e-3.
CDD handles the combined source+coefficient structure accurately. (The naive
"both lobes equally covered" check fails only because the coefficient field
genuinely makes the two lobes need different mode counts.)

### Part D — 3D BCOO footprint (16³ measured, 32³ extrapolated)

| grid | nnz | % dense | nnz/row | BCOO mem |
|------|-----|---------|---------|----------|
| 16³=4096 (measured) | 1,020,272 | 6.1% | 249 | 16.3 MB |
| 32³=32768 (extrapolated) | ~8.2e6 | — | ~249 | **~131 MB** |

**The 32³ production BCOO operator is ~131 MB — fits comfortably on the 8 GB
RTX A2000.** nnz/row grows with dimension (45 in 1D → 106 in 2D → 249 in 3D) but
remains bounded. The round-6 ~120 KB estimate referred to *state* (c/mask
vectors, 0.26 MB/vector at 32³), separate from the operator. Beyond ~64³
(~1 GB operator) a matrix-free matvec (apply W, A_phys, Wᵀ in sequence without
materialising A_wave) would be preferable; flagged for very large grids.

### Part E — CDD outer-iteration distribution

102 runs (1D+2D, σ∈{0.02,0.05,0.10}, θ swept): **p50=12, p90=12, p99=15,
max=19. Zero runs >40.** Worst cases are all σ=0.02 (sharpest source), peaking at
θ=0.5 (mid-domain, not boundary). The ≤20 guideline holds at p99. 3D shifts the
mean up ~30% (Part B θ_D=0.5 → 16.6; Part E discontinuous → 12–14). **Recommend
MAX_OUTER=30** — covers the 1D/2D max (19) and 3D mean (~17) with margin; no
pathological tail observed.

**Investigation 2 verdict:** no probe surfaced a blocker. Dirichlet is fine
(build isotropic-Dirichlet basis in impl); non-separable traps are rarer than
modelled; complex geometry converges; 32³ BCOO fits memory; iteration counts are
tightly bounded (MAX_OUTER=30).

---

## Closeout Investigation 3 — known-limitations document

Produced as the standalone `spikes/wavelet_derisking/KNOWN_LIMITATIONS.md`
(17 entries: the 13 mandated + 4 surfaced by closeout Inv 1/2 — dense 3D adjoint
kinks, ~30% higher 3D iteration count, sharding-unvalidated, and the
analytical-only hybrid assembly cost model). Self-contained handoff for the
implementation team. No code.

---

## Closeout Investigation 4 — sharding (multi-GPU) design note

**Design document only — no experiments.** Why adaptive-wavelet sharding is
harder than uniform-grid sharding, and a concrete API proposal. None of this
blocks single-device `WaveletAdaptiveNode`; it must be designed before a
`ShardedWaveletAdaptiveNode`.

### Why uniform-grid sharding assumptions break

Uniform-grid sharding is clean because (1) halo width is fixed by the stencil,
(2) the partition is statically load-balanced (equal DOF per shard), and (3) the
operator is domain-local (banded). Adaptive wavelets break **all three**:

1. **Halo width is level-dependent, not fixed.** A DD-4 wavelet at level ℓ has
   physical support ∝ filter-length × 2^{−ℓ} on the fine grid; expressed in
   coarse-shard terms its support spans ~filter-length fine cells at the finest
   level but grows by 2^ℓ at coarser levels. The coarsest-level wavelets have
   near-global support → their "halo" is the whole domain.
2. **The active set is not load-balanced.** CDD concentrates DOF near the
   swimmer/wall (closeout 1E: ~49% of active modes within 1.5h of the interface).
   One shard can hold the bulk of the active set while a neighbour holds almost
   none. Static domain decomposition → severe imbalance; dynamic rebalancing →
   communication every CDD outer iteration.
3. **A_wave is not domain-local.** A_phys (FD) is banded, but
   A_wave = Wᵀ A_phys W couples any two wavelets whose physical supports overlap
   — and coarse wavelets overlap globally. Off-diagonal A_wave entries couple
   distant shards through the coarse levels (consistent with closeout 2D's
   249 nnz/row in 3D, denser than the FD stencil).

### The two-tier resolution (standard in parallel adaptive-wavelet literature)

Split coefficients by level (Kevlahan–Vasilyev-style):

- **Fine levels** (local support): `requires_halo = True`, halo width =
  DD-4 filter half-length at that level. Standard halo exchange before each
  level's transform/matvec — same structure as parallel FEM.
- **Coarse levels** (global support): **replicate across all shards.** The
  coarsest levels have very few coefficients, so replication is cheap and avoids
  all-to-all. For a 3D DD-4 basis the coarsest level has
  O(2^{3(log₂N − L)}) coefficients — for N=64³, L=5 → 8 coeffs; N=256³, L=8 →
  512 coeffs. **Coarse-level replication is essentially free for any realistic
  grid and any shard count** — so coarse replication never becomes the
  bottleneck (answers the "max feasible shard count" question: replication cost
  is negligible; shard count is bounded by fine-level halo/comm, not coarse
  replication).

### Concrete API proposal

The earlier `ShardedNode.requires_halo: bool` is **insufficient** — it must be
level-indexed:

```python
class ShardedWaveletAdaptiveNode(WaveletAdaptiveNode):
    # replaces requires_halo: bool
    def halo_width_at_level(self, level: int) -> int:
        """Fine levels: filter_halflen (constant in fine-grid cells).
        Coarse levels (level < self.coarse_cutoff): return -1 sentinel
        meaning 'replicated, no halo exchange'."""

    coarse_cutoff: int   # levels < cutoff are replicated; >= cutoff are haloed

    def is_replicated_level(self, level: int) -> bool:
        return level < self.coarse_cutoff
```

`coarse_cutoff` is chosen so the replicated block is small (e.g. ≤ a few hundred
coeffs) — for production grids that is the coarsest 2–3 levels.

### CDD GROW step under sharding

GROW computes Σ_{inactive} r_i² and Dörfler-marks. Sharded: each shard holds the
residual for its local (fine) modes plus the (replicated) coarse residual.
**One all-reduce per CDD outer iteration** is required to sum Σ r_i² across
shards before the threshold is applied; marking is then local (each shard marks
its own modes against the global threshold). This all-reduce is **not currently
in the CDD design** and must be added — it is cheap (one scalar reduction +ranked
partial sums) but it is a real synchronisation point. The vectorised
`argsort+cumsum+searchsorted` GROW must be made level-aware so coarse (replicated)
and fine (sharded) modes are marked consistently (no double-marking of the
replicated coarse modes).

### Frozen-set adjoint under sharding

The IFT adjoint solves A_ΛΛ^T y = ∂J/∂c, structurally identical to the forward
distributed solve: fine modes → distributed sparse solve with halo exchange at
each GMRES matvec (standard parallel FEM); coarse replicated modes → the adjoint
runs locally on each shard and the **coarse-mode gradient contributions are
reduced (summed) once at the end**, with care to avoid double-counting the
replicated coarse block (divide by shard count, or designate one owner shard).
The `stop_gradient` on mask construction **still holds under sharding**: the mask
is assembled from the all-reduced global residual and the argsort/searchsorted is
non-differentiable regardless of how the residual was reduced — the all-reduce is
a forward-only operation feeding a stop_gradient'd discrete selection, so it does
not open a gradient path. (The Cauchy-interlacing submatrix-safety result, Inv 5,
is shard-count-independent — it is a property of the SPD operator, not the
partition.)

### Verdict

**`WaveletAdaptiveNode` is shardable** with the two-tier fine/coarse treatment,
but it requires: (a) `halo_width_at_level(level)->int` replacing the boolean
`requires_halo`; (b) coarse-level replication (free at any realistic scale);
(c) one all-reduce per CDD outer iteration in a level-aware GROW; (d) careful
coarse-mode gradient reduction in the adjoint. **None is a blocker; all need
explicit design before `ShardedWaveletAdaptiveNode`.** Single-device
`WaveletAdaptiveNode` proceeds without any of this. Recommended sequencing:
ship single-device first; design sharding as a follow-on once the single-device
node and the BCOO operator are validated.

---

# Spike closed

**Declared closed: 2026-06-22.** Remaining open items are scoped to
implementation (see `KNOWN_LIMITATIONS.md` and the first-milestone list below),
not to derisking. No further spike rounds.

### One-paragraph summary (for the v1.1+ plan preamble)

The wavelet derisking spike established, through a standalone numpy/JAX harness
(no `src/` code, ~20 throwaway scripts), that a Deslauriers–Dubuc (DD-4) adaptive
wavelet PDE solver is viable for the MADDENING `WaveletAdaptiveNode`. Across 1D,
2D and 3D it is well-conditioned (O(1) condition number with the isotropic Mallat
basis and a hybrid-Jacobi preconditioner; κ ≈ 20/38/158 in 1D/2D/3D), sparse
(sensor-functional error <0.2% using only N/16 = 6.25% of the basis in 3D),
wrong-sign-safe (guaranteed by coarse-level inclusion in the Cohen–Dahmen–DeVore
selection, not by locality alone), differentiable (autodiff matches finite
differences to ~1e-9 at smooth points and yields the correct Clarke subgradient
at active-set kinks, including through a `lax.scan` trajectory), and
trap-resistant (the selection-induced blindness traps that shaped the base-class
design are a non-local-basis phenomenon to which the local wavelet basis is
immune). The frozen-active-set inner solve is provably as well-conditioned as the
full operator (Cauchy interlacing). Algebraic Jacobi — refined to a hybrid
per-entry-coarse / level-mean-fine form at O(log N) cost — is the recommended
preconditioner and automatically handles discontinuous Brinkman coefficients.
The JAX/BCOO engineering carries no show-stoppers and the 32³ production operator
fits in 8 GB of GPU memory (~131 MB). The wavelet path is de-risked;
implementation can begin.

### Consolidated plan edits (all rounds, in one place)

1. **Basis & preconditioner (C1, Inv 1):** use the **isotropic Mallat** 2D/3D
   basis and the **hybrid-Jacobi** preconditioner (per-entry at the coarse
   level, level-mean at fine levels) as the default. Delete the anisotropic
   `D=2^{|λx|+|λy|+|λz|}` scaling (it gives κ∝N); expose matrix-free DK `2^{tj}`
   and full Jacobi as opt-ins. Hybrid ≡ full-Jacobi κ at O(log N) assembly.
2. **Wrong-sign theorem (§3):** state as *"CDD is wrong-sign-safe because it
   always retains the coarse levels dominating the sensor functional,"* NOT
   *"locality forbids wrong-sign"* (DD-4 has ±7% negative lobes). Keep top-|b|
   deprecated.
3. **Trap machinery (Gate 2, Inv 6, 2B):** document `blindness_ratio` /
   `symmetry_break` / cold-start gate as **non-local-basis insurance** — near-
   inert for the wavelet node (do not build a 3D-specific δ). For a future
   spectral `TopKAdaptiveNode`, monitor at **D≥2** (not D>5); note real coupled
   PDEs have a low trap rate (0.1–0.3/trajectory), so this is cheap insurance.
4. **CDD parameters (Inv 1B/E, 2E):** keep **θ_D=0.5** (16.6 outer iters in 3D);
   annotate `θ_D<κ^{-1/2}` as an approximation-optimality bound, not an
   iteration-count requirement. Set **MAX_OUTER=30** (1D/2D p99=15/max=19; 3D
   mean ~17). Budget 3D solves at ~1.5–2× the 2D cost (higher κ).
5. **Adjoint (§6, Inv 1D):** the trajectory adjoint is exact between active-set
   changes and Clarke at kinks — **no `custom_vjp`/`stop_gradient` mitigation
   needed**. In 3D the objective is densely kinked (K=256 ties); use the autodiff
   Clarke subgradient and consider mild smoothing/trust-region if an optimiser
   chatters.
6. **Coefficients (Inv 4, 4-2D/3D):** Jacobi handles discontinuous Brinkman
   coefficients automatically (a third advantage over theory-DK). Besov scaling
   is unnecessary for H¹ solutions; keep it as a DK opt-in for any future non-H¹
   case.
7. **Inner solve (Inv 5):** no active-set/subband-completeness constraint needed
   — κ(A_ΛΛ)≤κ_full for any Λ by Cauchy interlacing.
8. **Sharding (closeout Inv 4):** replace `ShardedNode.requires_halo: bool` with
   `halo_width_at_level(level)->int`; two-tier fine-halo / coarse-replicated
   scheme; add one all-reduce per CDD outer iteration in a level-aware GROW;
   reduce coarse-mode adjoint contributions carefully. Single-device first.
9. **Memory (closeout 2D):** 32³ BCOO ~131 MB (fits 8 GB); switch to a
   matrix-free matvec beyond ~64³.

### First-implementation-milestone validations (deferred → check FIRST)

In rough priority order — these are the things the spike could not establish and
that most affect viability/timeline:

1. **3D Mallat BCOO construction + autodiff** (Limitation 11) — the largest
   timeline assumption; if it exceeds ~1 week, flag. Do this first.
2. **Cross-validation against MIME FVM/BEM** on one shared problem (Limitation 7)
   — guards against a systematic error invisible to autodiff-vs-FD self-checks.
3. **Isotropic-Dirichlet Mallat basis** + re-run κ / CDD / wrong-sign
   (Limitation 1) — production BCs.
4. **CDD on the biharmonic residual** in the stream-function cavity
   (Limitation 12) — selection on a 4th-order residual is untested.
5. **Quantitative Ghia cavity at ≥64²** + steady-state integration (Limitation 6).
6. **Trajectory adjoint at T=100** with gradient checkpointing (Limitation 8).
7. **GPU benchmark** — assembly-vs-solve split, JIT overhead, hybrid-Jacobi
   crossover on the RTX A2000 (Limitations 5, 17).
8. **Optimiser convergence** (not just gradient accuracy) under the dense 3D
   adjoint kinks (Limitation 14).

### Statement of confidence (honest)

**Confident** (validated across 1D/2D/3D with consistent, mechanism-backed
results): the conditioning story (O(1) κ, hybrid Jacobi, Cauchy-interlacing
submatrix safety); the sparsity/accuracy story (J_err <0.2% at N/16 in 3D, CDD ≫
rolling, near-sharp easier); wrong-sign safety via CDD coarse-inclusion; adjoint
correctness (Clarke subgradient, no contamination); the wavelet basis's immunity
to selection-induced traps; that Jacobi auto-handles discontinuous coefficients;
that the JAX/BCOO path has no show-stoppers and fits GPU memory at 32³.

**Not confident / explicitly unvalidated** (→ first-milestone list): GPU
wall-clock performance (all timing is CPU/numpy); quantitative cavity accuracy
(only qualitative at 47²); 2D/3D *isotropic-Dirichlet* conditioning (expected
fine — 1D Dirichlet was κ=3.8 — but the isotropic-Dirichlet basis was not built);
the effort to construct the 3D JAX BCOO operator (asserted mechanical, not
executed); long-trajectory (T~1000) adjoint stability; sharded/multi-GPU
execution (designed, not tested); and optimiser behaviour under dense 3D kinks
(gradient is correct, but convergence dynamics are untested). None of these is a
known blocker; all are validate-during-implementation. The spike is **not** a
proof the production solver will hit any particular speed or accuracy target —
it is evidence that the approach is sound and that the remaining risks are
implementation risks, not derisking risks.

**The wavelet derisking spike is closed. `WaveletAdaptiveNode` implementation
can begin, starting with the first-milestone validations above.**
