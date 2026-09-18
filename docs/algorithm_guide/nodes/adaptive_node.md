---
bibliography: ../../bibliography.bib
---

# AdaptiveNode

**Module**: `maddening.nodes.adaptive`
**Stability**: evolving
**Algorithm ID**: `MADD-NODE-009`
**Version**: 1.1.0

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

which is exact on every open region of $\theta$ where $M$ is constant.

**What happens at a region boundary.** The returned gradient ignores the
dependence of $M$ on $\theta$. Where two candidates swap rank, their
coefficients $b_k/\lambda_k$ and their sensor weights $\phi_k(x_s)$ differ, so
$J_{\text{frozen}}$ **jumps**:

$$
\lim_{\varepsilon \downarrow 0}
\big[ J_{\text{frozen}}(\theta_s + \varepsilon) - J_{\text{frozen}}(\theta_s - \varepsilon) \big]
\;\ne\; 0 .
$$

A discontinuous function is not locally Lipschitz at the discontinuity, so
**no Clarke subgradient exists there** and the returned value is not one: it is
the one-sided derivative of the branch the forward pass selected. The
consequence is *first order*, not second: over an interval the integral of the
returned gradient misses exactly the jumps it crossed,

$$
J(b) - J(a) \;=\; \int_a^b \frac{dJ_{\text{frozen}}}{d\theta}\, d\theta
\;+\; \sum_{\theta_s \in (a,b)} \big[ J(\theta_s^+) - J(\theta_s^-) \big].
$$

Measured on the 1-D sine toy at $n_{\max} = 256$, top-$|b|$, $K = 16$: over
$\theta \in [0.40, 0.50]$ the integral of the returned gradient is
$-1.5808 \times 10^{-3}$ against a true change of $-2.3598 \times 10^{-3}$ — a
**33 % shortfall**, equal to the sum of the 27 jumps crossed. The omitted term
decays with the active-set budget, because the coefficient that swaps rank
shrinks: on the same problem the jump contribution falls from $2.5 \times
10^{-1}$ of $|J|$ at $K = 8$ to $\sim 10^{-8}$ at $K = 64$ (the spike measured
$4 \times 10^{-8}$). Registered as anomaly `MADD-ANO-003`; asserted by
`tests/nodes/adaptive/test_active_set_switch.py`.

**Budget adequacy and blindness.** Let $G$ be a symmetry group of $(A, b, s)$ with fixed-point set
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

$\rho$ is **not** a symmetry test. It measures how much of the full-basis
gradient the frozen set reproduces, and it is driven mostly by the active-set
budget: at a fixed, entirely non-symmetric $\theta = 0.42$ on the 1-D toy it
measures 0.16 / 0.57 / 0.85 / 1.00 at $K = 4 / 8 / 16 / 32$, identically at
every $n_{\max}$. A low $\rho$ therefore means *either* too small a budget
*or* a trap. `frozen_gradient_vanishes_at` — which perturbs along the escape
direction, re-selects, and asks whether the frozen gradient is negligible
against its own rate of change — separates them in one direction only: a
`False` rules a trap out, a `True` is equally consistent with an ordinary
stationary point of the frozen objective. The cold-start check warns for the
budget case and raises when the frozen gradient has also collapsed.

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
| Gradient-capture ratio $\rho$ | `maddening.nodes.adaptive.base.AdaptiveNode.gradient_capture_ratio` | Active set re-selected at the evaluated $\theta$; sentinel `1.0` when $\|\nabla J_{\text{full}}\|$ is negligible. `blindness_ratio` is a deprecated alias |
| Cold-start policy (warn / raise / ignore) | `maddening.nodes.adaptive.base.AdaptiveNode.check_gradient_capture` | Warns on a low ratio; raises only when `frozen_gradient_vanishes_at` is also true or `on_blind="raise"` |
| Double-`where` guard for a masked operand | `maddening.nodes.adaptive.base.AdaptiveNode.mask_safe` | Sanitises the *input* of an operation that is singular off the active set |
| Vanishing-frozen-gradient check | `maddening.nodes.adaptive.base.AdaptiveNode.frozen_gradient_vanishes_at` | Re-thresholded finite difference along the escape direction. Necessary for a Palais trap, not sufficient: `False` rules one out, `True` also fires at an ordinary stationary point. `is_trapped_at` is a deprecated alias |
| Escape step $\theta + \delta\, g_{\text{full}}/\|g_{\text{full}}\|$ | `maddening.nodes.adaptive.base.AdaptiveNode.symmetry_break` | Trainable leaves only (`ParamSpec.trainable`) |
| Gated cold start with one escape attempt | `maddening.nodes.adaptive.base.AdaptiveNode.cold_start` | Raises `AdaptiveNodeBlindnessError` on a persistent trap |

## Assumptions and Simplifications

1. The active set is locally constant in $\theta$; the returned gradient is
   exact on each such region and ignores the set's dependence on $\theta$.
   Across a change of active set the objective jumps (see *Governing
   Equations*), so the gradient is neither the derivative nor a subgradient
   there and first-order methods accumulate the crossed jumps as bias.
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
| Active fraction `K / n_max` | 0.016 – 1.0 | Top-K budgets 4 – 256 of 256 modes, all constructible with the default cold-start policy. Below `K / n_max` ≈ 0.1 the gradient-capture ratio falls under the 0.7 threshold and construction *warns* (it is not rejected); the missing first-order term is then percent-level — see the row below |
| Jump contribution to $dJ/d\theta$ | $2.5\times10^{-1}$ (K=8) → $\sim10^{-8}$ (K=64) | Fraction of $\|J\|$ omitted by the returned gradient per unit $\theta$, 1-D sine toy over $[0.40, 0.42]$. Treat the frozen gradient as trustworthy only in the large-budget end of this range |
| Trainable parameters | 1 | The diagnostic constants were calibrated on 1-D and 2-D parameter spaces; above `D_threshold = 5` run `frozen_gradient_vanishes_at` between optimiser steps, reading a `False` as "not a trap" rather than a `True` as "trap" |
| `gradient_capture_threshold` | 0.7 | Spike round 6; states measured at 0.86 (good), 0.17 (partial), 0.0 (trap) |
| `blindness_break_delta` | 0.05 | Spike round 7; escapes the 1-D trap (minimum 0.03) and the 2-D traps tested |

## Known Limitations and Failure Modes

1. **Jump discontinuities at active-set switches** (`MADD-ANO-003`). A step
   across an active-set change misses a jump in the objective: the returned
   gradient is neither the derivative nor a subgradient there, and the
   objective/gradient pair a line search or quasi-Newton method sees is
   inconsistent. First-order methods accumulate the sum of the crossed jumps
   as systematic bias — 33 % of the objective change over a 0.1-wide window at
   $K = 16$ (see *Governing Equations*), $\sim 10^{-8}$ at $K = 64$.
   Hysteresis in the subclass's selection rule (add above
   $\varepsilon_{\text{add}}$, remove below
   $\varepsilon_{\text{remove}} < \varepsilon_{\text{add}}$, using `prev`)
   reduces chattering; it does not remove the jumps. A large active-set budget
   does shrink them.
2. **Palais traps.** At a fixed point of the problem's symmetry `jax.grad`
   returns a plausible gradient that is exactly zero in the escape direction;
   nothing in the forward pass signals this. The cold-start check catches it
   at the parameters it is handed — the constructor's, unless you call
   `check_gradient_capture(gm.params["nodes"][name])` after seeding a graph —
   and routine monitoring is the caller's policy.
3. **A low gradient-capture ratio is usually a budget, not a trap.** The ratio
   cannot distinguish them, and neither diagnostic *establishes* a trap:
   `frozen_gradient_vanishes_at` returning `False` rules one out, but a `True`
   also fires at any stationary point of the frozen objective — including the
   optimum a successful fit converges to, measured at `theta = 0.343973` on
   the 1-D toy, whose only reflection fixed point is `theta = 0.5`. Establish
   the symmetry from the operator, source and objective. `cold_start()` /
   `symmetry_break()` help only in the genuine trap case (at a budget-limited
   point the audited case went 0.565 → 0.060).
4. **Masked operands poison the gradient, not the value.** `solve_frozen` that
   evaluates a singular expression on inactive entries returns a clean forward
   pass and a `NaN` gradient; the base class cannot repair it. Use
   `AdaptiveNode.mask_safe` on the operand.
5. **Non-local bases can produce wrong-sign solutions** when the selection is
   by source magnitude near a boundary (spike round 4, sine basis with
   top-$|b|$): the active modes' values at the sensor alternate in sign.
   Selecting by solution magnitude ($|b_k/\lambda_k|$) or using a local basis
   avoids it; the base class does not choose for the subclass.
6. **Cost.** `gradient_capture_ratio` and `symmetry_break` need a full-basis gradient,
   the expensive solve adaptivity exists to avoid; the padded buffer costs
   `n_max` memory and FLOPs per step regardless of how many entries are active.
7. **Abstract.** `AdaptiveNode` cannot be instantiated at all without
   `compute_active_set` and `solve_frozen` (both `@abstractmethod`);
   `objective` stays optional and is needed only by the diagnostics.

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
| `n_max` | int | required | — | Size of the padded basis buffer. **Structural and not stored in `self.params`**: a subclass that wants its basis size to survive serialisation declares its own integer parameter for it |
| `gradient_capture_threshold` | float | 0.7 | — | Ratio below which the active set is judged not to reproduce the full-basis gradient (class attribute; constructor override). `blindness_threshold` is a deprecated alias |
| `blindness_break_delta` | float | 0.05 | parameter units | `symmetry_break` step size (class attribute; constructor override) |
| `D_threshold` | int | 5 | — | Trainable-parameter count above which runtime trap monitoring is recommended |
| `blindness_gate` | bool | True | — | Run the cold-start diagnostic in `initial_state` (recorded in `self.params`, so it survives a round trip) |
| `on_blind` | str | `"warn"` | — | `"warn"` / `"raise"` / `"ignore"` policy for a low ratio (recorded in `self.params`) |
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
  finite differences to $10^{-6}$ — each with the active set asserted not to
  change within the step — and vs dense closed-form references),
  `tests/nodes/adaptive/test_blindness_diagnostics.py` (spike-measured
  blindness ratios 0.86 / 0.17 / 0.0 at $\theta = 0.42 / 0.48 / 0.5$, trap
  detection and escape), `tests/nodes/adaptive/test_traceability.py` (jit,
  scan, vmap, no gradient leak through the selection),
  `tests/nodes/adaptive/test_graph_integration.py` (GraphManager, params
  pytree, checkpoint round trip),
  `tests/nodes/adaptive/test_active_set_switch.py` (the jump at a switch, the
  1/h divergence of finite differences across one, the exact in-region
  gradient against the selected branch, and the integral/jump reconstruction
  quoted above),
  `tests/nodes/adaptive/test_masked_gradient_traps.py` (the `jnp.where`
  gradient trap and `mask_safe`),
  `tests/nodes/adaptive/test_round_trip.py` and
  `tests/usd/test_usd_adaptive_node.py` (config and USD reconstruction).
- Design evidence: `plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md`
  (seven spike rounds behind the constants above).

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.1.0 | 2026-09-17 | Audit follow-up: jump-discontinuity framing replaces the Clarke/kink claim (`MADD-ANO-003`); `blindness_ratio` renamed `gradient_capture_ratio` (budget adequacy, not symmetry) and the cold-start check warns rather than rejecting unless a trap is established; `n_max` out of `self.params` and the diagnostic settings in it, so a node survives a config / USD round trip; stability lowered to `evolving` pending the 0.4.0 API freeze |
| 1.0.0 | 2026-09-16 | Initial implementation (frozen-active-set adjoint, blindness diagnostics) |
