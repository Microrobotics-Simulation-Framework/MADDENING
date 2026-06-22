# WaveletAdaptiveNode — Known Limitations (handoff to implementation)

**Status:** handoff document from the wavelet derisking spike (closed
2026-06-22). Self-contained — readable without the spike findings memo.

This is the honest, complete list of what the derisking spike did *not* fully
establish, written for the implementation team. The spike concluded that
`WaveletAdaptiveNode` is buildable and the wavelet path is de-risked; this
document records the caveats so they are validated (not rediscovered) during
implementation.

**Context in one paragraph.** The spike validated, via a numpy/JAX harness
(no `src/` code), that a Deslauriers–Dubuc (DD-4) adaptive wavelet PDE solver
with Cohen–Dahmen–DeVore (CDD) residual-driven mode selection, a hybrid-Jacobi
diagonal preconditioner, and a frozen-active-set implicit-function-theorem
adjoint is well-conditioned (O(1) κ in 1D/2D/3D), sparse (J_err <0.2% at N/16
in 3D), wrong-sign-safe (via coarse-level inclusion in CDD), differentiable
(autodiff = finite-difference to ~1e-9, Clarke subgradient at kinks), and
trap-resistant (the selection-induced blindness traps are a non-local-basis
phenomenon; the local wavelet basis is immune). Everything below is a boundary
of that validation.

Risk legend: **H** = could invalidate the approach; **M** = could change a
design choice or a quantitative prediction; **L** = cosmetic / well-understood
approximation.

---

## 1. Periodic BC assumption throughout the harness
- **What:** Every numpy condition-number / CDD / wrong-sign experiment used
  periodic boundary conditions. Production (MIME vessel walls, no-slip) is
  Dirichlet, which introduces boundary wavelets with modified support and a
  coarse-level inhomogeneity.
- **Evidence:** All of §2/§3/§4 and the continuation series used periodic BCs;
  closeout Investigation 2A added a 1D/2D Dirichlet probe.
- **Risk: M.** 2A showed 1D Dirichlet is actually *better* conditioned (κ=3.8)
  and wrong-sign safety holds; but 2D Dirichlet was tested only via the
  *anisotropic tensor* basis (κ=146, vs 38 isotropic-periodic) because no
  isotropic-Dirichlet Mallat basis was built. The true isotropic-Dirichlet κ is
  expected ~38 but unconfirmed.
- **Mitigation:** Build a boundary-adapted isotropic Dirichlet Mallat basis as
  an early implementation step and re-run the κ / CDD / wrong-sign checks; Jacobi
  keeps even the worst tested case (146) workable, so this is validate-not-fear.

## 2. Simplified source and inclusion geometries
- **What:** All sources were Gaussian bumps; all coefficient inclusions were
  circles/spheres or steps. The production swimmer is a helix with a complex
  Brinkman coefficient curve.
- **Evidence:** §2–§6, Inv 3/4; closeout 2C added a figure-eight source +
  crescent coefficient.
- **Risk: L** for the framework (CDD is geometry-agnostic — 2C converged in 14
  iters at J_err 1e-3 on a multi-lobe/crescent problem), **M** for specific
  convergence-rate predictions on the real geometry.
- **Mitigation:** Validate CDD iteration counts against the actual swimmer
  geometry during implementation; expect the qualitative behaviour to hold.

## 3. blindness_threshold = 0.7 calibrated in 1D/2D
- **What:** The 0.7 partial-blindness threshold was calibrated on 1D/2D
  sine-basis problems.
- **Evidence:** round-3/4/6 (sine basis); Gate 2 showed the wavelet basis is
  trap-immune so the threshold never fires for it.
- **Risk: L** for `WaveletAdaptiveNode` (trap-immune), **M** for a future
  spectral `TopKAdaptiveNode` at D≥2.
- **Mitigation:** Recalibrate during any TopKAdaptiveNode development; for the
  wavelet node the diagnostic is a near-inert cold-start safety net.

## 4. symmetry_break δ = 0.05 calibrated in 1D/2D
- **What:** The anisotropic perturbation magnitude was calibrated in 1D/2D, not
  3D, for non-wavelet bases.
- **Evidence:** round-7 (δ_1D=0.03, δ_2D=0.001); Gate 2 (wavelet trap-immune).
- **Risk: L** for the wavelet node, **M** for TopKAdaptiveNode in 3D.
- **Mitigation:** Re-run the symmetry_break calibration as part of any
  TopKAdaptiveNode development.

## 5. No GPU benchmarks
- **What:** All timing/throughput data is CPU/numpy (κ, iteration counts). GPU
  memory bandwidth, JIT compile overhead, and kernel efficiency at scale are
  unmeasured.
- **Evidence:** Inv 2 PoC reported CPU JIT compile times only; closeout 2D gave
  BCOO memory (131 MB at 32³) but no GPU timing.
- **Risk: M** for performance predictions, **L** for correctness.
- **Mitigation:** Benchmark on the target GPU (RTX A2000, 8 GB) early; the
  hybrid-Jacobi assembly cost (Inv 1) is the one quantity most sensitive to GPU
  memory-bandwidth vs compute balance.

## 6. Quantitative Ghia cavity match deferred
- **What:** The driven-cavity benchmark was validated only qualitatively at 47²
  (primary vortex + corner vortices visible); vortex centre at (0.50, 0.85) vs
  Ghia Re=100 reference (0.62, 0.74).
- **Evidence:** §5 + Inv 3B.
- **Risk: M** — a quantitative match needs ≥64² and longer time integration.
- **Mitigation:** Plan for ≥64² and steady-state integration in the benchmark
  milestone; the formulation (stream-function, Jacobi-preconditioned biharmonic)
  is validated on conditioning grounds.

## 7. No cross-validation against an independent code
- **What:** Every accuracy metric is a self-consistency check (autodiff vs FD,
  CDD vs oracle, CDD vs full solve). No comparison against BEM/FVM/COMSOL on an
  identical problem.
- **Evidence:** all rounds.
- **Risk: M** — a systematic error shared by the gradient and the FD reference
  (e.g. a wrong operator assembly) would be invisible to self-consistency.
- **Mitigation:** First implementation milestone must include one validation
  case against the existing MIME FVM or BEM node on a problem both can solve
  (e.g. Stokes flow past a sphere, or a Poisson problem with known solution).

## 8. lax.scan trajectory adjoint validated only at T≤10 (1D), T≤5 (3D)
- **What:** The no-T-degradation result holds to T≤10 (1D, §6) and T≤5 (3D,
  closeout 1D). Production trajectories may be T=100–1000.
- **Evidence:** §6 (1D, 1e-10 at T≤10); closeout 1D (3D, ~1e-9 at smooth points
  T≤5).
- **Risk: M** — floating-point accumulation or memory (reverse-mode stores all
  T steps) at T=1000 is untested.
- **Mitigation:** Validate at T=100 early; use `jax.checkpoint` (gradient
  checkpointing) on the scan if memory-bound; expect accuracy to hold (the
  per-step adjoint is exact, errors do not obviously compound).

## 9. θ_D = 0.5 iteration count: validated in 3D, theory bound clarified
- **What:** Gate 2 validated J_err at k=N/16 but not the outer iteration count;
  the optimality theory bound θ_D < κ^{-1/2} = 0.079 (3D) appears to forbid 0.5.
- **Evidence:** closeout 1B — θ_D=0.5 converges in **16.6 outer iters** in 3D
  (≤30); small θ_D is *worse* for iteration count (245 at θ_D=0.08) with
  identical J_err.
- **Risk: L** — the theory bound governs the *approximation-optimality rate*
  guarantee, not practical iteration count; θ_D=0.5 converges fine.
- **Mitigation:** Keep θ_D=0.5 (or 0.7, even faster); annotate the plan's
  "θ_D<κ^{-1/2}" line as an optimality bound, not an iteration requirement.

## 10. Separable model used for D_threshold; coupled model now adds evidence
- **What:** Inv 6 used a separable J=ΣJ_1d to estimate trap encounter rate.
- **Evidence:** Inv 6 (separable, ~1–1.5 encounters/trajectory, D-independent);
  closeout 2B (genuinely coupled, **0.1–0.3** encounters — *rarer*).
- **Risk: L** (downgraded from M) — the coupled PDE has *fewer* traps than the
  separable model, so the conservative "monitor at D≥2" is safe and low-stakes.
- **Mitigation:** Re-validate during TopKAdaptiveNode development; moot for the
  trap-immune wavelet node.

## 11. 3D BCOO stiffness matrix not constructed in JAX (only 1D/2D)
- **What:** Inv 2 PoC built and autodiff-validated the *2D* Mallat BCOO operator.
  The 3D operator (7 subbands/level vs 3 in 2D) was validated as a *numpy* dense
  operator (closeout Inv 1) but not as a JIT-compiled JAX BCOO.
- **Evidence:** Inv 2C (2D BCOO autodiff); closeout 2D (3D nnz measured from the
  numpy operator: 1.02M nnz, 249 nnz/row, 16.3 MB at 16³).
- **Risk: M** for the 3–5 week timeline (assumes the 3D BCOO extension is
  mechanical).
- **Mitigation:** **First implementation milestone is 3D Mallat BCOO
  construction + autodiff validation.** If it exceeds ~1 week, flag immediately —
  it is the single largest timeline assumption.

## 12. CDD on the biharmonic residual not tested (only the Laplacian residual)
- **What:** §5 validated Jacobi *conditioning* of the biharmonic operator; CDD
  *selection* was always tested on Laplacian (2nd-order) residuals.
- **Evidence:** §5 (biharmonic κ); Inv 3 (CDD on Laplacian/nonlinear residuals).
- **Risk: M** — biharmonic residuals (4th-order, t=2) have different spatial
  structure (steeper coefficient decay) than Laplacian residuals; CDD's
  marking may need a different bulk parameter.
- **Mitigation:** Validate CDD on the stream-function (biharmonic) residual early
  in the driven-cavity benchmark milestone.

## 13. Besov regime genuinely untested
- **What:** Inv 4 tested bounded discontinuous (Brinkman) coefficients giving H¹
  solutions and confirmed Besov scaling is unnecessary. Genuinely singular
  forcings (point/delta forces) or non-H¹ solutions were not tested.
- **Evidence:** Inv 4 (Brinkman, H¹); §2 Hyp B (Besov, recorded not implemented).
- **Risk: L** for current MIME applications (Brinkman penalisation → H¹).
- **Mitigation:** Documented; the Besov `2^{|λ|(1−d/2)}` scaling is available as a
  DK opt-in if a future application produces non-H¹ solutions.

---

## Additional limitations surfaced by closeout Investigations 1–2

## 14. 3D adjoint objective is densely kinked (K=256 top-K ties)
- **What:** In 3D with k=N/16=256 active modes, the active-set selection boundary
  is crossed frequently along any θ-trajectory — the fraction of "smooth" points
  drops from 60% (T=1) to 16% (T=5).
- **Evidence:** closeout 1D — grad is exact (≤1e-9) at smooth points and the
  correct Clarke subgradient at kinks, but kinks are dense.
- **Risk: M** for optimiser behaviour — gradient *correctness* is unaffected
  (Clarke everywhere), but a gradient-descent optimiser sees a piecewise-smooth
  objective with frequent small kinks, which can cause chattering.
- **Mitigation:** Use the autodiff Clarke subgradient (correct); if kink
  chattering impedes convergence, apply mild objective smoothing or a
  proximal/trust-region step. Validate optimiser convergence (not just gradient
  accuracy) on a real inverse problem early.

## 15. 3D CDD iteration count ~30% higher than 1D/2D (higher κ)
- **What:** 3D hybrid-Jacobi κ≈158 (vs ~38 2D, ~20 1D) yields ~30% more CDD
  outer iterations (16.6 mean at θ_D=0.5; 12–14 for discontinuous) and ~2× the
  per-step inner GMRES of 2D.
- **Evidence:** closeout 1B, 1E; 2E (1D/2D distribution p99=15, max=19).
- **Risk: L** — still well within MAX_OUTER=30; just a throughput factor.
- **Mitigation:** Set MAX_OUTER=30; budget 3D solves at ~1.5–2× the 2D cost.

## 16. Sharding (multi-GPU) design unvalidated — see SHARDING design note
- **What:** No sharded execution was tested. Adaptive wavelets break the three
  assumptions that make uniform-grid sharding clean (fixed halo, static
  load-balance, domain-local operator).
- **Evidence:** closeout Investigation 4 (design note in FINDINGS.md).
- **Risk: M** for the *sharded* production path (single-device is unaffected).
- **Mitigation:** The design note proposes a two-tier (fine-halo /
  coarse-replicated) scheme and a `halo_width_at_level(level)->int` API; this
  needs an explicit design pass before `ShardedWaveletAdaptiveNode`, but
  single-device `WaveletAdaptiveNode` can proceed without it.

## 17. Hybrid-Jacobi assembly cost model is analytical, not measured
- **What:** The claim that hybrid Jacobi (O(log N) assembly) beats full Jacobi
  (O(N)) in the matrix-free/probed-diagonal regime rests on a flop/bandwidth cost
  model, not a measured wall-clock crossover.
- **Evidence:** Inv 1C (crossover analysis); the conditioning equivalence
  (hybrid ≡ full κ) *is* measured.
- **Risk: L** — hybrid is never *worse* than full (identical κ, fewer scalars);
  the only question is *how much* better, which depends on GPU specifics.
- **Mitigation:** Measure the assembly-vs-solve split on the target GPU; default
  to hybrid regardless (it dominates or ties full Jacobi).
