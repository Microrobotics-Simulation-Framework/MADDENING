# D2 — dJ/da in 2D/3D on the dense path

**Verdict: PASS.** `jax.grad` w.r.t. the coefficient field `a_grid` matches FD in
2D and 3D. App 1's forward-model gradient is sound above 1D. M19–M20 may proceed.

## Result (`d2_djda_nd.py`, fp64, central FD e=1e-4)

| dim | side | N | max relerr | mean relerr |
|---|---|---|---|---|
| 2 | 16 | 256 | 4.7e-7 | 1.9e-7 |
| 2 | 32 | 1024 | 1.6e-5 | 3.9e-6 |
| 3 | 8 | 512 | 5.8e-7 | 1.5e-7 |

Objective: `J(a) = φ(x_sensor)` for a fixed Gaussian source, solving
`(-∇·(a∇) + m)u = f` in the L²-normalised wavelet basis, preconditioner frozen at
`a₀=1` (gradient-irrelevant at convergence, per the existing 1D test pattern).

## Note — the first run reported a false FAIL at 32²

Initial pass (central FD `e=1e-6`, the 1D test's step) reported 32² at 2.3e-3,
tripping the 1e-3 gate. Diagnosis (`d2_diag.py`): the failing probe voxels all have
`|dJ/da| ~ 1e-11`, three orders below the ~1e-9 typical sensitivity. At that
magnitude a 1e-6 central step is **roundoff-dominated** — it subtracts two `J`
values equal to ~13 significant figures. Against a **4th-order** FD stencil the
worst error is **2.0e-5**; widening the central step to `e=1e-4` gives the clean
table above.

So this was the FD *reference* being inaccurate at low-sensitivity voxels, not the
analytic gradient. The gradient is correct; the harness step was too small.

## Consequence for the milestones

- Use `e=1e-4` (or a 4th-order stencil) for grad-vs-FD tests in 2D/3D. `e=1e-6` is
  fine in 1D (higher per-voxel sensitivity) but under-resolved in 2D/3D.
- This is the *dense-path* gradient. M17 must independently re-validate the
  **matrix-free** adjoint under `jit` — a different code path (lineax, not
  `jnp.linalg.solve`), where the D3 transpose trap could reintroduce a silent error.
