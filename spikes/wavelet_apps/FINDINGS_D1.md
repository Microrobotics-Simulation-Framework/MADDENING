# D1 — Drug BC: does periodic + zero-padding approximate no-flux?

**Verdict: NO, if the drug problem genuinely needs no-flux.** But the result reframes
the question: **padding models an *open/absorbing* boundary, not a *reflecting*
(no-flux) one.** Which is physically correct is a modeling decision for the user —
that decision, not a numerical fix, is what determines whether M9 is a no-op.

## Result (`d1_drug_bc.py`, 1D reaction-diffusion, no-flux reference vs padded)

| λ/L | pad | max rel err in ROI |
|---|---|---|
| 0.10 | 0.25 | 2.5e-2 |
| 0.10 | 0.50 | 2.5e-2 |
| 0.10 | 1.00 | 6.5e-3 |
| 0.25 | 0.50 | 1.4e-1 |
| 0.50 | 0.50 | 4.1e-1 |

Padding matches no-flux only when λ/L ≤ 0.1 **and** the domain is doubled (pad=1.0),
and even then at ~0.7%. In the realistic λ/L ~ 0.25 regime it is off by 13–15%
regardless of pad width.

## Why — this is physics, not a resolution artefact

Zero-padding forces `c → 0` far from the source: with enough pad it reproduces the
**free-space / open-domain** solution (drug diffuses away to background). No-flux
(`∂c/∂n = 0`) is a **reflecting wall**: drug cannot escape, so concentration *piles up*
near the boundary. These are different boundary *types*, and the difference propagates
into the ROI unless the solution has already decayed to ~0 before reaching the wall
(λ ≪ distance-to-wall). No amount of padding converts an absorbing boundary into a
reflecting one — padding converges to the open-domain answer, which is the wrong limit
if the wall is sealed.

## The decision this surfaces (for the user)

**What does the ROI boundary physically represent in the microrobot delivery problem?**

- **Sealed / impermeable wall** (organ capsule, vessel wall the drug cannot cross):
  no-flux is correct, padding is wrong by 2–15%, and **M9 is real work** — a
  variable-coefficient non-periodic assembly path that does not exist today
  (`operator.py:268-271` raises `NotImplementedError`; `physical_varcoeff` hardcodes
  `% side`). Comparable in size to the matrix-free workstream; it ends app 2's status
  as the cheap near-term deliverable.
- **Open boundary** (ROI is a computational window inside a larger tissue; drug
  diffuses out into surrounding tissue toward background): padding is *correct*, and
  **M9 is a no-op** as originally hoped. App 2 stays cheap.

The measurement cannot decide this — it is a question about the target scenario. It is
exactly the kind of assumption the derisk exists to surface before M9 is built.

## Note

The λ/L ≤ 0.1, pad=1.0 corner (0.65%) is a genuine third option: if the physical
decay length is short relative to the ROI, a sealed wall is *also* well-approximated by
padding, because the drug never reaches the wall. So even under the no-flux
interpretation, M9 could stay a no-op *if* the pharmacokinetics give λ/L ≲ 0.1. The
user's tissue/drug parameters (D, k) settle this.
