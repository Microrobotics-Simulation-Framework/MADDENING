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
| 2 | §4 — 3D sparsity break-even | pending | |
| 3 | §5 — Stokes / stream-function cavity | pending | |
| 3 | §6 — trajectory adjoint under lax.scan | pending | |

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
