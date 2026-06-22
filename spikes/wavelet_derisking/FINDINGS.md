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
| 1 | §2 Hyp A — CDD convergence | pending | |
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
