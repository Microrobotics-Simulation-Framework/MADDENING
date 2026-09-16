---
bibliography: ../../bibliography.bib
---

# AdaptiveNode

**Module**: `maddening.nodes.adaptive`
**Stability**: stable
**Algorithm ID**: `MADD-NODE-009`
**Version**: 1.0.0

## Summary

`AdaptiveNode` is the base class for solvers that represent their solution in a
padded basis of `n_max` candidates but solve only on a data-dependent **active
set**. It implements the *frozen-active-set adjoint*: the selection step is
committed as a non-differentiable forward operation and the gradient flows
through the frozen-basis linear solve by the implicit function theorem
[@Blondel2022]. The pattern is basis-agnostic — the wavelet, sparse-grid or
hierarchical-basis subclass supplies the selection rule and the solve
[@CohenDahmenDeVore2001; @VasilyevPaolucci1996] — and comes with a diagnostic
for the one failure mode selection cannot see, the Palais fixed point
[@Palais1979].

## Governing Equations

A discretised problem in a basis $\{\phi_i\}_{i=1}^{n_{\max}}$ with parameters
$\theta$:

$$
A(\theta)\, c = b(\theta), \qquad J(\theta) = s(c),
$$

restricted to an active set $M \subseteq \{1, \dots, n_{\max}\}$ chosen by a
selection map $\mathcal{M}(\theta, c_{\text{prev}})$:

$$
A_M(\theta)\, c_M = b_M(\theta), \qquad c_i = 0 \;\; (i \notin M),
\qquad J_{\text{frozen}}(\theta; M) = s\!\left(A_M^{-1} b_M\right).
$$

With $M$ frozen, the adjoint of the objective is the standard linear-solve
sensitivity

$$
\frac{dJ_{\text{frozen}}}{d\theta}
 = -\lambda^{\top}\!\left(\frac{\partial A_M}{\partial \theta} c_M - \frac{\partial b_M}{\partial \theta}\right),
\qquad A_M^{\top} \lambda = \nabla_{c_M} s,
$$

which is exact on every open region of $\theta$ where $M$ is constant and a
one-sided (Clarke) subgradient at the kinks where $M$ changes.

**Blindness.** Let $G$ be a symmetry group of $(A, b, s)$ with fixed-point set
$\mathrm{Fix}(G)$. If the selection scores modes by a $G$-invariant functional
of $(A, b)$, then at $\theta_* \in \mathrm{Fix}(G)$ the frozen gradient lies in
$T_{\theta_*}\mathrm{Fix}(G)$ (Palais' principle of symmetric criticality
applied to $J_{\text{frozen}}$): no selection rule can supply the transverse
component. The base class measures

$$
\rho(\theta) = \frac{\|\nabla_\theta J_{\text{frozen}}\|}{\|\nabla_\theta J_{\text{full}}\|},
$$

with $J_{\text{full}}$ the objective on the full basis ($M = $ everything), and
escapes with the anisotropic step
$\theta \leftarrow \theta + \delta\, \nabla_\theta J_{\text{full}} / \|\nabla_\theta J_{\text{full}}\|$.

## Discretization

- **Padded buffer.** `c` and the boolean `mask` both have shape `(n_max,)`;
  adaptivity changes which entries of `mask` are true, never an array shape,
  so `update` traces once and runs under `jax.jit` / `jax.lax.scan`.
- **Selection** (`compute_active_set`): any `jnp` rule — top-K, threshold,
  hysteresis on the previous mask — evaluated under `jax.lax.stop_gradient`.
- **Frozen solve** (`solve_frozen`): the masked system is solved as a full-size
  system that is the identity on inactive rows, through
  `ift_linear_solve` (CG / GMRES / dense); lineax's native autodiff gives the
  adjoint above, no MADDENING-level `custom_vjp`.
- **Consistency**: the base class zeroes `c` off the mask and stores the mask
  the coefficients were solved on.
- **Diagnostics** are host-side (Python floats), evaluated at cold start
  (`initial_state`, `cold_start`) or between optimiser steps, never inside the
  traced step.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| Selection $M = \mathcal{M}(\theta, c_{\text{prev}})$ under `stop_gradient` | `maddening.nodes.adaptive.base.AdaptiveNode.update` | Calls the subclass `compute_active_set` and wraps the boolean mask in `jax.lax.stop_gradient` |
| Frozen solve $A_M c_M = b_M$ | `maddening.nodes.adaptive.base.AdaptiveNode.solve_frozen` | Subclass hook; abstract in the base class |
| Adjoint $A_M^{\top}\lambda = \nabla_{c_M} s$ | `maddening.core.solver_utils.ift_linear_solve` | lineax `linear_solve` autodiff; GMRES restart clamped to `min(N, 50)` |
| $c_i = 0$ for $i \notin M$ | `maddening.nodes.adaptive.base.AdaptiveNode.update` | `jnp.where(mask, c, 0)` after every solve (also at cold start) |
| Cold-start state at the constructor parameters | `maddening.nodes.adaptive.base.AdaptiveNode.initial_state` | Selection with `is_cold_start=True`, solve, blindness gate |
| $\nabla_\theta J_{\text{full}}$ | `maddening.nodes.adaptive.base.AdaptiveNode.compute_full_basis_gradient` | Default: `jax.grad` of `objective` through `solve_frozen` with an all-true mask |
| Blindness ratio $\rho$ | `maddening.nodes.adaptive.base.AdaptiveNode.blindness_ratio` | Sentinel `1.0` when $\|\nabla J_{\text{full}}\|$ is negligible |
| Binary trap check | `maddening.nodes.adaptive.base.AdaptiveNode.is_trapped_at` | Re-thresholded finite difference along the escape direction |
| Escape step $\theta + \delta\, g_{\text{full}}/\|g_{\text{full}}\|$ | `maddening.nodes.adaptive.base.AdaptiveNode.symmetry_break` | Trainable leaves only (`ParamSpec.trainable`) |
| Gated cold start with one escape attempt | `maddening.nodes.adaptive.base.AdaptiveNode.cold_start` | Raises `AdaptiveNodeBlindnessError` on a persistent trap |

## Assumptions and Simplifications

1. The active set is locally constant in $\theta$; the frozen gradient is exact
   on each such region and a subgradient across a change of active set.
2. `solve_frozen` with every entry of the mask true is the full-basis solve
   (used by the default full-basis gradient).
3. The objective used by the diagnostics is a scalar function of the solved
   coefficients.
4. The masked operator is non-singular on the active set (the framework does
   not check this).
5. The physical parameters live in the graph parameter pytree, not in the
   state; the state carries only coefficients, the mask and any extra fields
   the subclass declares.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| `n_max` | 16 – 256 | Toy problems in the test suite (1-D sine basis, dense SPD system) |
| Active fraction `K / n_max` | 0.016 – 1.0 | Top-K budgets 4 – 256 of 256 modes; the padded buffer costs `n_max` regardless |
| Trainable parameters | 1 | The blindness constants were calibrated on 1-D and 2-D parameter spaces; above `D_threshold = 5` run `is_trapped_at` between optimiser steps |
| `blindness_threshold` | 0.7 | Spike round 6; states measured at 0.86 (good), 0.17 (partial), 0.0 (trap) |
| `blindness_break_delta` | 0.05 | Spike round 7; escapes the 1-D trap (minimum 0.03) and the 2-D traps tested |

## Known Limitations and Failure Modes

1. **Kinks.** A gradient step across an active-set change is a subgradient,
   not the derivative; optimisers that assume smoothness (line searches,
   quasi-Newton) can stall or oscillate there. Hysteresis in the subclass's
   selection rule (add above $\varepsilon_{\text{add}}$, remove below
   $\varepsilon_{\text{remove}} < \varepsilon_{\text{add}}$, using `prev`)
   reduces chattering; it does not remove the kinks.
2. **Palais traps.** At a fixed point of the problem's symmetry `jax.grad`
   returns a plausible gradient that is exactly zero in the escape direction;
   nothing in the forward pass signals this. The cold-start gate catches it
   at construction; routine monitoring is the caller's policy.
3. **Non-local bases can produce wrong-sign solutions** when the selection is
   by source magnitude near a boundary (spike round 4, sine basis with
   top-$|b|$): the active modes' values at the sensor alternate in sign.
   Selecting by solution magnitude ($|b_k/\lambda_k|$) or using a local basis
   avoids it; the base class does not choose for the subclass.
4. **Cost.** `blindness_ratio` and `symmetry_break` need a full-basis gradient,
   the expensive solve adaptivity exists to avoid; the padded buffer costs
   `n_max` memory and FLOPs per step regardless of how many entries are active.
5. **Abstract.** `AdaptiveNode` cannot be instantiated usefully on its own;
   `compute_active_set`, `solve_frozen` and (for the diagnostics) `objective`
   must be supplied.

## Stability Conditions

The frozen solve is a linear solve; its stability is that of the subclass's
operator on the active set (condition number of $A_M$, solver tolerances
passed to `ift_linear_solve`). There is no time-stepping stability limit in the
base class: `update` re-solves from the parameters and the previous mask.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `c` | `(n_max,)` | problem-defined | Basis coefficients; zero off the active set |
| `mask` | `(n_max,)` | — | Boolean active set the current `c` was solved on |

Subclasses may add fields through `extra_initial_state()`.

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `n_max` | int | required | — | Size of the padded basis buffer (structural) |
| `blindness_threshold` | float | 0.7 | — | Ratio below which a state is blind (class attribute; constructor override) |
| `blindness_break_delta` | float | 0.05 | parameter units | `symmetry_break` step size (class attribute; constructor override) |
| `D_threshold` | int | 5 | — | Trainable-parameter count above which runtime trap monitoring is recommended |
| `blindness_gate` | bool | True | — | Run the blindness check in `initial_state` |
| `dtype` | dtype | canonical float | — | dtype of `c` |
| subclass `**params` | float / int | — | problem-defined | Physical constants; floats become leaves of the graph parameter pytree with the subclass's `ParamSpec`s |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| — | — | — | The base class declares none; `update` ignores `boundary_inputs` and `dt`. Subclasses that couple to other nodes declare their own `boundary_input_spec` |

## References

- [@Palais1979] Palais, R. S. (1979). *The principle of symmetric criticality*. Communications in Mathematical Physics, 69(1), 19–30. — Why a frozen-set gradient at a symmetric point has no transverse component; the mechanism the blindness diagnostics detect.
- [@CohenDahmenDeVore2001] Cohen, A., Dahmen, W., DeVore, R. (2001). *Adaptive wavelet methods for elliptic operator equations: convergence rates*. Mathematics of Computation, 70(233), 27–75. — The active-set (bulk-chasing) framing of adaptive basis solvers the `compute_active_set` / `solve_frozen` split follows.
- [@VasilyevPaolucci1996] Vasilyev, O. V., Paolucci, S. (1996). *A dynamically adaptive multilevel wavelet collocation method for solving partial differential equations in a finite domain*. Journal of Computational Physics, 125(2), 498–512. — Threshold-driven adaptive wavelet collocation, the intended first concrete subclass.
- [@Blondel2022] Blondel, M. et al. (2022). *Efficient and modular implicit differentiation*. Advances in Neural Information Processing Systems 35. — Implicit-function-theorem adjoints through solver calls, as used by `ift_linear_solve`.
- [@LeVeque2007] LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM. — Green's function of the 1-D two-point boundary-value problem used as the reference solution in MADD-VER-004.

## Verification Evidence

- Benchmark: `MADD-VER-004` — frozen-active-set solve of $-u'' + u = f$ on
  $(0, 1)$ vs the exact Green's-function solution: full active set reproduces
  it (L2 relative error $< 10^{-4}$, sensor error $< 10^{-6}$); top-K sensor
  error decreases monotonically over $K \in \{4, 8, 16, 32\}$.
- Test files: `tests/nodes/adaptive/test_verification.py` (benchmark and the
  `verify_node` battery on two concrete subclasses),
  `tests/nodes/adaptive/test_frozen_solve_gradients.py` (`jax.grad` vs central
  finite differences to $10^{-6}$ and vs dense closed-form references),
  `tests/nodes/adaptive/test_blindness_diagnostics.py` (spike-measured
  blindness ratios 0.86 / 0.17 / 0.0 at $\theta = 0.42 / 0.48 / 0.5$, trap
  detection and escape), `tests/nodes/adaptive/test_traceability.py` (jit,
  scan, vmap, no gradient leak through the selection),
  `tests/nodes/adaptive/test_graph_integration.py` (GraphManager, params
  pytree, checkpoint round trip).
- Design evidence: `plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md`
  (seven spike rounds behind the constants above).

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-09-16 | Initial implementation (frozen-active-set adjoint, blindness diagnostics) |
