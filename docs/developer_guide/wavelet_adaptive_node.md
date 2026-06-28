# WaveletAdaptiveNode developer guide

`WaveletAdaptiveNode` is the production adaptive-wavelet PDE solver built on the
`AdaptiveNode` framework (see `adaptive_node.md` for the base class). It solves

```
(-Δ + m) u(x) = f(x; θ)          [or -∇·(a(x)∇u) + m u = f]
```

on an **isotropic Mallat Deslauriers–Dubuc (DD-4) wavelet basis**, selecting an
adaptive active set per step with **Cohen–Dahmen–DeVore (CDD)** residual marking
and a **hybrid-Jacobi** preconditioner, and exposing exact gradients through the
frozen active-set solve.

It is the implementation outcome of the wavelet derisking spike
(`spikes/wavelet_derisking/`, closed 2026-06-22); the design corrections it
carries in are summarised below and detailed in that spike's `FINDINGS.md` and
`KNOWN_LIMITATIONS.md`.

## When to use it

Use `WaveletAdaptiveNode` when you need a **differentiable** elliptic solve that
**adapts** resolution to localised features (a moving source, a swimmer body, a
boundary layer) and you want gradients w.r.t. continuous parameters (source
position, coefficient field) through the adaptation. It is *not* a replacement
for an FFT/spectral solver on a uniform constant-coefficient grid (FFT
diagonalises the operator and is unbeatable there) — its value is adaptivity,
variable coefficients, complex geometry, and differentiability.

## Quick start

```python
from maddening.nodes.adaptive import WaveletAdaptiveNode

node = WaveletAdaptiveNode(dim=2, n_levels=4, n_coarse=2, theta_init=0.42)
state = node.initial_state()          # cold-start (coarse-then-fine, blindness gate)
state = node.update(state, {}, 1.0)   # one CDD select + frozen solve
J = node._sensor(state)               # sensor functional u(x_sensor)
```

`dim` ∈ {1, 2, 3}; grid side = `n_coarse · 2**n_levels` (periodic) or the
boundary-adapted Dirichlet size; `N_max = side**dim`.

## Key constructor parameters

| param | default | meaning |
|-------|---------|---------|
| `dim` | 1 | spatial dimension (1/2/3) |
| `n_levels`, `n_coarse` | 6, 2 | refinement levels / coarse points per axis |
| `order` | 4 | DD order (DD-4 is the validated production default) |
| `K` | `N_max//16` | CDD active-set budget (the spike's sparsity target) |
| `sigma`, `sensor`, `theta_init` | — | Gaussian source width, sensor location, initial θ |
| `mass` | 1.0 | zeroth-order term |
| `preconditioner` | `"hybrid"` | `"hybrid"`/`"full"`/`"level"`/`"dk"` |
| `boundary` | `"periodic"` | `"periodic"` or `"dirichlet"` |

## Design choices carried from the spike (do not undo)

1. **Isotropic Mallat basis + hybrid-Jacobi default.** The anisotropic
   `2^{|λx|+|λy|+|λz|}` Dahmen-Kunoth scaling fails (κ ∝ N) on an isotropic
   operator; the isotropic single-level basis with hybrid Jacobi gives O(1) κ
   (spike Correction C1). DK `2^{tj}` and full Jacobi are opt-ins.
2. **Wrong-sign safety via coarse-inclusion, not pure locality.** DD-4 has ±7%
   negative side-lobes, so top-|b| selection can produce wrong-sign solutions.
   CDD always retains the coarse levels that dominate the sensor functional and
   is wrong-sign-safe (spike §3). Top-|b| is deprecated.
3. **The blindness / `symmetry_break` machinery is near-inert here.** The
   selection-induced Palais traps are a *non-local-basis* phenomenon; the local
   wavelet basis is trap-immune (spike Gate 2). The inherited cold-start gate is
   a cheap safety net, not an active mechanism — do not add wavelet-specific
   trap mitigation or a dimension-specific δ.
4. **CDD parameters:** `θ_D = 0.5`, `MAX_OUTER = 30`. The theory bound
   `θ_D < κ^{-1/2}` governs approximation optimality, *not* iteration count —
   small θ_D is worse for iterations (spike Inv 1B).
5. **No `custom_vjp` / `stop_gradient` mitigation on the adjoint.** Autodiff is
   exact between active-set changes and the correct Clarke subgradient at kinks
   (spike §6); confirmed through `lax.scan` to T=100 with no degradation.
6. **Jacobi handles discontinuous (Brinkman) coefficients automatically;** Besov
   scaling is unnecessary for H¹ solutions (a DK opt-in if ever needed).

## How the pieces fit (`maddening.nodes.adaptive.wavelets`)

- `transform.py` — matrix-free DD lifting transforms (1D + isotropic Mallat
  2D/3D), JIT-compilable, static-shape.
- `operator.py` — Galerkin operators: `assemble_wave_operator` (BCOO + dense),
  `physical_laplacian` / `_dirichlet` / `physical_varcoeff` /
  `physical_biharmonic`; `gather_solve` (the O(K) frozen solve); variable-
  coefficient differentiability via `bcoo_with_traced_data` (static indices,
  traced data — `jax.grad` w.r.t. `a(x)` flows through assembly).
- `precond.py` — `diagonal_scaling` (hybrid/full/level/dk).
- `cdd.py` — CDD selection (Python-unrolled outer loop, vectorised Dörfler
  GROW, static-shape mask, capped at K).
- `dirichlet.py` — boundary-adapted DD basis for homogeneous Dirichlet BCs
  (dense; multi-D via tensor product).

## Performance notes

- The frozen solve uses **gather-to-K**: the K active DOFs are gathered into a
  fixed buffer and solved directly (O(K³)), realising the adaptivity speedup —
  `node.update` is ~45 ms (2D 64²) / ~57 ms (3D 16³) on an RTX A2000.
- Everything is JIT-safe with **no recompilation** across same-shape calls
  (audited). The masked operator closes over a pre-assembled constant BCOO — do
  **not** call `BCOO.fromdense` on a traced array (fails under `jit`).
- Known perf headroom (tracked): early-exit CDD (`while_loop`) would shave the
  fixed 30-iteration cost; a matrix-free Dirichlet transform would speed up the
  dense-Wn Dirichlet path; matrix-free matvec is preferred beyond ~64³.

## Limitations

See `spikes/wavelet_derisking/KNOWN_LIMITATIONS.md` for the full list. Headline
items: variable-coefficient Dirichlet assembly is not yet supported (periodic
only); the Dirichlet basis is dense (matrix-free is future); multi-GPU sharding
is designed but not implemented (single-device only); the node is
`@stability(EXPERIMENTAL)` pending validation against the full production
swimmer geometry.

## Validation

`tests/adaptive/test_wavelet_*` (engine, node 1D/2D/3D, Dirichlet, biharmonic,
trajectory) and `tests/verification/test_wavelet_*` (MMS + MIME cross-code in
1D/2D/3D, lid-driven cavity vs Ghia). The cavity and trajectory tests are in the
`slow` lane.
