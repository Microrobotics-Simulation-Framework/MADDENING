# Wavelet AdaptiveNode derisking spike

**THIS IS A SPIKE.** Throwaway, investigative experiment scaffolding for
`plans/WAVELET_ADAPTIVE_NODE_DERISKING_SPIKE.md`. Not production code, not
wired into the package, no test-suite integration. Read `FINDINGS.md` for the
results and the executive summary.

**Outcome:** all four derisking gates PASS → `WaveletAdaptiveNode`
implementation can proceed. See `FINDINGS.md` for the plan corrections
(isotropic basis + Jacobi preconditioner; qualified locality theorem;
trap machinery is non-local-basis insurance).

## Files

| file | what |
|------|------|
| `dd_wavelets.py` | Deslauriers-Dubuc interpolating wavelets (1D, isotropic 2D/3D), Haar calibration baseline, physical operators. numpy. |
| `g1_condition_number.py` | §2 Hyp A — DK condition number, 1D/2D, Haar-calibrated. |
| `g1_wrong_sign.py` | §3 — DD phi-sign / wrong-sign safety vs a sine control. |
| `g1_cdd_convergence.py` | §2 Hyp A — CDD convergence, rolling comparison, step-source stress. |
| `g2_3d.py` | §4 — 3D sparsity break-even + trap structure (caches the heavy build to `/tmp/g2_3d_env.npz`). |
| `g2_trap_basis.py` | §4 mechanism probe — blindness is a non-local-basis phenomenon (sine traps, DD does not). |
| `g3_biharmonic.py` | §5 — stream-function biharmonic preconditioning (t=2). |
| `g3_trajectory.py` | §6 — trajectory adjoint through `lax.scan` (JAX, float64). |
| `g4_improvements.py` | post-gate — submatrix conditioning + cheap-diagonal verification. |
| `hybrid_jacobi.py` | Inv 1 — hybrid vs full/level/DK preconditioners (Parts A–C). `argv`: A1/A2/B1/B2/all, `--big` for N=16384. |
| `dd_jax_poc.py` | Inv 2 — DD-4 in JAX: transform, BCOO+lineax autodiff, 2D Mallat (JAX, float64). |
| `nonlinear_cdd.py` | Inv 3 — CDD on nonlinear residual: Burgers (A) + stream-function cavity (B). |
| `discontinuous_coeff.py` | Inv 4 — discontinuous-coefficient (Brinkman) 1D jump (A) + 2D moving inclusion (B). |
| `submatrix_2d3d.py` | Inv 5 — submatrix conditioning across active-set configs in 2D/3D. |
| `dthreshold_empirical.py` | Inv 6 — D_threshold=5 empirical trap-rate validation. |

The 6-investigation continuation series (2026-06-22) is complete; all resolved.
See `FINDINGS.md` "Continuation series" + "Cross-cutting statement".

## Running

```bash
PY=/home/nick/MSF/msf/.venv/bin/python
cd spikes/wavelet_derisking
$PY g1_condition_number.py
$PY g1_wrong_sign.py
$PY g1_cdd_convergence.py
$PY g3_biharmonic.py
$PY g3_trajectory.py
$PY g2_3d.py            # ~3 min first run (builds + caches 4096^3 op)
$PY g2_3d.py --traps-only   # fast after cache
$PY g2_trap_basis.py
$PY g4_improvements.py
```
