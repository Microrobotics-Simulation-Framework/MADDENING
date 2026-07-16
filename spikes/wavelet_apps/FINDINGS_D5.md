# D5 — PCG iteration counts at χ ≤ 10³

**Verdict: FAIL at the stated bar (≲100 iters at χ=10³). STOP-and-report per protocol.**
But the failure is **scoped to application 1's high-contrast regime** — it does *not*
halt application 2 or the shared matrix-free infrastructure. Details below.

## Result (`d5_cg_iters.py`, 2D 32², CG to rel-resid 1e-8 on hybrid-Jacobi Ah)

| contrast | κ(Ah) | CG iters | unprec iters |
|---|---|---|---|
| 1e0 | 3.77e1 | 39 | 186 |
| 1e1 | 2.38e2 | 93 | 452 |
| 1e2 | 2.04e3 | 225 | 1026 |
| 1e3 | 2.01e4 | **393** | 1787 |

Iterations track ~√κ (sub-√κ, actually), exactly as measurement 1's κ ∝ contrast
predicts. This is the full-operator count; the masked active-set system is bounded
above by it (Cauchy interlacing), so 393 is a conservative ceiling, not an underestimate.

## What this does and does not mean

**It confirms the plan's central bet, precisely.** "Go matrix-free silently imports
need a contrast-robust preconditioner — application 1 only." At χ=10³, hybrid-Jacobi
CG needs ~400 iterations per solve. Multiplied through CDD outer iterations and
gradient steps, that is not affordable. Diagonal scaling cannot fix it (measurement 1:
it buys a constant factor, not a better scaling law), so **R1 (BPX/AMG) moves from
gated research to a near-term prerequisite for application 1 at χ ≥ ~10².**

**It is a performance finding, not a correctness one.** Matrix-free CG *converges* at
every contrast — it is just slow at high contrast. Matrix-free remains the only route
to scale (the hardware note is untouched). M13–M18 are unaffected as *infrastructure*;
what changes is that M16/M19 cannot claim app-1 performance at χ=10³ without R1.

**Application 2 is clear.** Its diffusion contrast is ~10 (D varies by ~one order in
tissue). At contrast 10, CG needs **93 iterations** — affordable. The app-2 path
(M1–M12) is fully derisked and does not depend on R1. (Note: the plan's √κ≈30 estimate
for app 2 was optimistic; measured 93 at contrast 10, but still fine. At contrast 100
it is 225 — a soft ceiling app 2 is unlikely to hit.)

## Consequence — proposed plan change (needs user decision)

The STOP is real: proceeding as if app-1 matrix-free is cheap would be wrong. Options
to surface:

1. **Reorder:** ship app 2 first (M1–M12, fully cleared), build shared matrix-free
   infrastructure (M13–M18), and gate app 1 (M19–M23) behind an R1 milestone inserted
   before M19. R1 (a BPX additive multilevel preconditioner) drops into M5's
   `inner_precond` slot — the seam was designed for exactly this.
2. **Cap app 1 at χ ≤ ~10²** near-term (225 iters — tolerable), deferring χ=10³–10⁵ to
   R1. Covers implants/surgical steel but not bulk magnetite.
3. **Promote R1 now** as an explicit milestone before any app-1 work.

All three keep app 2 and M1–M18 moving. The derisk did its job: this is exactly the
kind of assumption that should surface before weeks of app-1 work, not after.
