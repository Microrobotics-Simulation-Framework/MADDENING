---
bibliography: ../../bibliography.bib
---

# WaveletAdaptiveNode

**Module**: `maddening.nodes.adaptive.wavelet`
**Stability**: experimental
**Algorithm ID**: `MADD-NODE-010`
**Version**: 1.0.1

## Summary

`WaveletAdaptiveNode` is the first concrete
[`AdaptiveNode`](adaptive_node.md) subclass with a real basis. It solves the
steady elliptic problem $(-\Delta + m)\,u = f$ on the unit cube in one, two or
three dimensions, periodic or with homogeneous Dirichlet walls, in an
interpolating (Deslauriers–Dubuc) wavelet basis [@DeslauriersDubuc1989],
selects the active set by Cohen–Dahmen–DeVore bulk chasing
[@CohenDahmenDeVore2001; @Doerfler1996] up to a budget $k$, solves on that
set through a gathered dense block, and returns the sensor reading
$J = u(x_s)$ with the frozen-active-set adjoint the base class provides:
plain reverse mode through the gathered solve, or the implicit-function
rule of [@Blondel2022] on the masked-CG path.

## Governing Equations

$$
(-\Delta + m)\, u(x) = f(x;\, \theta, \sigma), \qquad x \in [0, 1)^d,\; d \in \{1, 2, 3\},
$$

with $m > 0$ and the isotropic Gaussian source

$$
f(x;\, \theta, \sigma) = \exp\!\left(-\frac{(x_1 - \theta)^2 + \sum_{i \ge 2} (x_i - \tfrac12)^2}{\sigma^2}\right),
$$

periodic on every axis or $u = 0$ on every wall. The objective is the point
reading $J(\theta, \sigma) = u(x_s)$ at the grid point nearest the sensor
$x_s$. In the basis $u = \sum_j c_j \psi_j$ this is

$$
A\, c = b(\theta, \sigma), \qquad A = W_n^{\top} A_{\text{phys}} W_n, \qquad
b = h^{d}\, W_n^{\top} f, \qquad J = W_n[s, :]\, c,
$$

and, restricted to an active set $M$ with $|M| \le k$ chosen by

$$
M = \mathrm{CDD}\big(D^{-1} A D^{-1},\; D^{-1} b;\; M_0 = \text{coarse level},\; k\big),
$$

the frozen problem is $A_M c_M = b_M$, $c_j = 0$ for $j \notin M$, with the
adjoint of the base class ([`adaptive_node.md`](adaptive_node.md) *Governing
Equations*): exact within a region of constant $M$, blind to the region's
boundary (`MADD-ANO-003`).

## Discretization

- **Physical operator.** Second-order central differences for $-\Delta$ with
  lumped mass $h^d$ on a uniform dyadic grid of $n_c\,2^{J}$ points per axis
  (periodic) or $(n_c + 1)\,2^{J} - 1$ interior points (Dirichlet), $h = 1/n$
  or $1/(n+1)$.
- **Basis.** The Deslauriers–Dubuc interpolating wavelet basis of order 4 on
  the isotropic Mallat multiresolution (one level per refinement, $2^d - 1$
  detail subbands per level), built by a lifting scheme (predict the midpoint
  from the four nearest coarse samples, add the detail), matrix-free in JAX.
  The Dirichlet basis pins the wall values to zero and shrinks the stencil to
  linear at the walls; it is built dense at construction and the multi-D
  Dirichlet basis is the tensor product of the 1-D one. Columns are
  $L^2$-normalised ($W_n$).
- **Exact change of basis.** $A = W_n^{\top} A_{\text{phys}} W_n$ is assembled
  once, in NumPy, *checked* for symmetry and only then symmetrised: the
  correct product measures $\max|A - A^{\top}| / \max|A| \approx 10^{-16}$
  (15 configurations, 1-D to 3-D, both boundaries), a one-sided
  $[-1, 2, -1]/h$ stencil measures $1.07$, and `SYMMETRY_TOL = 10^{-12}`
  refuses anything above it. Symmetrising unconditionally would have
  turned that first-order stencil into a consistent second-order one and
  passed the order gate against the defect (measured 2.04). Because it is
  the finite-difference
  operator in another basis, the **full-basis** solve reproduces the
  finite-difference solution to round-off and the **order of accuracy is the
  stencil's: 2 in $h$**, for both boundary types. Declared as
  `DiscretizationOrder(spatial=2.0, temporal=None)` and measured by MMS
  (`MADD-VER-014`).
- **Preconditioning.** Symmetric diagonal scaling $\hat A = D^{-1} A D^{-1}$
  with $D$ the hybrid-Jacobi scaling (per-entry $\sqrt{A_{jj}}$ on the coarse
  level, the level mean of it on finer levels), which matches full Jacobi to
  four figures. $\kappa(\hat A) \approx 20$ (1-D, 256 points), $\approx 38$
  (2-D, $32^2$), $\approx 3.8$ (1-D Dirichlet). The Dahmen–Kunoth
  $2^{t\,\text{level}}$ scaling [@DahmenKunoth1992] is available and weaker
  here.
- **Selection.** CDD: start from the coarse level, estimate the residual
  $\hat r = \hat b - \hat A \hat c$, mark the smallest set of inactive
  functions carrying $\theta_D^2 = 0.25$ of the squared residual (Dörfler
  marking), capped at the room left under $k$, re-solve, repeat until
  $|M| \ge k$ or 30 iterations, as a `lax.while_loop`. The seed is *every
  level-0 function* -- the coarse block plus the first detail band,
  $(2 n_c)^d$ periodic or $(2 n_c + 1)^d$ Dirichlet -- and $k$ is validated
  to hold it. The selection is a function of the parameters alone (never of
  the previous state), so it never empties, never exceeds $k$, and does not
  chatter at fixed parameters. For $k$ above about $n_{\max}/2$ the
  iteration bound is the exit, not the budget: on 128 points $k = 64$ and
  $k = 96$ both stop at $|M| = 54$ (sensor-reading error $3 \times 10^{-11}$;
  200 iterations reach 64), because each Dörfler step marks a fixed fraction
  of the *remaining* residual. The mask is still a valid active set;
  `selection_diagnostics()` reports `outer_iterations` and `budget_reached`.
- **Frozen solve.** The $k$ active functions gathered into a dense
  $k \times k$ block and solved directly (`frozen_solver="gather"`,
  $O(k^3)$), or the masked full-size operator (identity off the mask) handed
  to `ift_linear_solve` with CG (`frozen_solver="cg"`). On the 128-point
  basis at $k = 8$ the two agree to $3 \times 10^{-17}$ in $\max|\Delta c|$;
  the property claimed and pinned is agreement to the CG tolerance,
  $10^{-10}$. Differentiation differs between them: the gathered path is
  plain reverse mode through `jnp.linalg.solve`; only the CG path uses the
  implicit-function rule of `ift_linear_solve` [@Blondel2022]. A mask with
  more than $k$ functions cannot be solved in the gathered block: a concrete
  one is refused with a message, a traced one poisons the block with NaN
  rather than silently dropping the excess.
- **Full-basis gradient.** A dense solve on $A$, overriding the base default:
  the gathered solve holds exactly $k$ functions and an all-true mask would be
  silently truncated.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| Basis synthesis $u = W_n c$ | `maddening.nodes.adaptive.wavelets.transform.synthesis`, `maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.field` | Lifting scheme, matrix-free; `field` applies the normalised dense $W_n$ |
| Analysis $c = W^{-1} u$ | `maddening.nodes.adaptive.wavelets.transform.analysis` | Exact inverse of the lifting scheme |
| Dirichlet basis | `maddening.nodes.adaptive.wavelets.dirichlet.synthesis_matrix_dirichlet` | Dense, walls pinned to zero, tensor product in multi-D |
| Operator assembly $A = W_n^{\top} A_{\text{phys}} W_n$ | `maddening.nodes.adaptive.wavelets.operator.assemble_operator` | NumPy, once per node; returns the dense $A$, a sparse copy, $W_n$, levels |
| Diagonal scaling $D$ | `maddening.nodes.adaptive.wavelets.precond.diagonal_scaling` | Hybrid Jacobi by default; Jacobi, level-mean and Dahmen–Kunoth selectable |
| Source $f(x;\theta,\sigma)$ | `maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.source_field` | Reads `theta` and `sigma` from the injected `params`; the override point for a manufactured source |
| Right-hand side $b = h^d W_n^{\top} f$ | `maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode._rhs` | Recomputed on every call, so the tangent flows |
| Active set $M$ by CDD | `maddening.nodes.adaptive.wavelets.cdd.cdd_select`, `maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.compute_active_set` | `while_loop` over solve / estimate / mark / refine; input under `stop_gradient`; the coarse level is the seed |
| Frozen solve $A_M c_M = b_M$ (gathered) | `maddening.nodes.adaptive.wavelets.operator.gather_solve`, `maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.solve_frozen` | Dense $k \times k$ block; JAX primitive `jnp.linalg.solve`; requires $\lvert M \rvert \le k$ -- an oversized concrete mask is refused, a traced one is NaN-poisoned |
| Selection diagnostics (iterations, budget reached) | `maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.selection_diagnostics`, `maddening.nodes.adaptive.wavelets.cdd.cdd_select_with_iterations` | Host-side; `outer_iterations`, `max_outer`, `active`, `k`, `budget_reached` |
| Frozen solve (masked CG) | `maddening.nodes.adaptive.wavelets.operator.make_masked_operator`, `maddening.core.solver_utils.ift_linear_solve` | Identity off the mask; `solver="cg"` |
| $c_j = 0$ for $j \notin M$ | `maddening.nodes.adaptive.base.AdaptiveNode.update` | Inherited: the base class zeroes off the mask after every solve |
| Sensor functional $J = W_n[s,:]\, c$ | `maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.objective` | Nearest grid point to `sensor` |
| Full-basis gradient $\nabla J_{\text{full}}$ | `maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.compute_full_basis_gradient` | Dense solve on $A$; overrides the base default |
| Baked constants and their provenance | `maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode.static_data_deps` | `scaling` from `mass`, `sensor_row` from `sensor`; `compile()` refuses to train either |

## Assumptions and Simplifications

1. Unit cube, uniform dyadic grid, constant coefficient $m > 0$: the operator
   is assembled once from the structural settings and `mass`, so `mass` is
   `ParamSpec(trainable=False)`. Variable coefficients are not supported.
2. The source is the isotropic Gaussian above unless `source_field` is
   overridden; `theta` and `sigma` are the trainable leaves.
3. $\text{seed} \le k \le n_{\max}$, validated at construction, where the seed
   is every level-0 function -- the coarse block plus the first detail band,
   $(2 n_c)^d$ periodic or $(2 n_c + 1)^d$ Dirichlet, counted from the
   assembled basis -- so the CDD seed fits in the gathered buffer and the
   set never exceeds it. The default $k = \min(n_{\max}, \max(\text{seed}, 8,
   n_{\max}/16))$ always satisfies the bound. $k = n_{\max}$ turns
   adaptivity off.
4. The base-class assumptions: the returned gradient is exact within a
   region of constant $M$ and ignores the set's dependence on $\theta$
   (`MADD-ANO-003`); the objective is a scalar function of $c$.
5. The sensor is read at the nearest grid point, not interpolated.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| Grid, order of accuracy | 16 – 256 points (1-D), $8^2$ – $32^2$ (2-D) | MMS order 2.000 on the 1-D periodic ladder 16/32/64/128/256; 2.000 on the Dirichlet ladder 23 – 191 and 2.022 on the 2-D periodic ladder $8^2$ – $32^2$ (`tests/verification/test_wavelet_mms_order.py`). No 3-D order ladder |
| Grid, adaptive solve and frozen gradient | up to 256 (1-D), $64^2$ (2-D), $16^3$ (3-D); Dirichlet $23^2$ and $7^3$ | At every size: the seed fits the default budget, the gathered solve equals the masked dense solve to $10^{-12}$, `jax.grad` matches central differences with the set held fixed to $10^{-6}$, capture ratio in $(0.9, 1.1)$. $64^2$ and $16^3$ (4096 functions, budget 256) are `@slow` tests, 10 – 15 s each on a loaded 24-core box (`test_the_largest_sizes_the_metadata_claims_construct_select_solve_and_differentiate`) |
| Default budget vs seed | all default `(boundary, dim, n_levels <= 2, n_coarse <= 3)` up to 529 functions | The seed (level 0) is inside `k` and every active coefficient is solved -- the 13 cheapest of the 16 default configurations on which it used to be dropped (`test_the_default_budget_holds_the_whole_level_zero_seed_so_no_coefficient_is_dropped`) |
| Budget $k$ | $n_{\max}/16$ (default) – $n_{\max}$ | At $k = n_{\max}/16$ on 128 points the sensor reading is within $6 \times 10^{-3}$ of the full-basis one over $\theta \in \{0.04, 0.30, 0.42, 0.50, 0.92\}$ and never changes sign (`MADD-VER-015`). The error is **not** monotone in $k$ and no rate in $k$ is claimed |
| Gradient-capture ratio | 0.99 – 1.01 | $\theta = 0.42$, 1-D 128 points, 2-D $16^2$, 3-D $8^3$, Dirichlet 95 points: the local basis reproduces the full-basis gradient at the default budget, so the cold-start check passes silently |
| `jax.grad` vs central differences | $1.5 \times 10^{-9}$ (θ), $4 \times 10^{-11}$ (σ) relative | Active set asserted unchanged across the step; through `jit`, `scan` and the compiled graph step |
| Order of the basis | 4 | 2 and 6 construct and round-trip; the operator's order stays 2 regardless |

## Known Limitations and Failure Modes

1. **Jump discontinuities at active-set switches** (`MADD-ANO-003`),
   inherited. At the default budget the ratio is ≈ 1, so the omitted
   first-order term is small, but a step across a switch still misses a jump.
2. **The adaptive solution is a truncation.** At a small $k$ the sensor
   reading can be off by more than the $10^{-2}$ measured at $n_{\max}/16$;
   the error is not monotone in $k$.
3. **`mass` and `sensor` are baked.** Changing either on a built node leaves
   the operator and the sensor row stale; rebuild the node. `compile()`
   refuses a graph that unfreezes them.
4. **Steady only.** `update` ignores `dt` and `boundary_inputs` because the
   base class's hooks do not receive them, so the node is an edge *source*
   and cannot time-step or consume an edge; a time-dependent wavelet node
   needs the freeze question in the developer guide answered.
5. **Dense operator.** $A$ is an $n_{\max} \times n_{\max}$ constant of the
   step; sizes beyond the validated range need a matrix-free assembly.
6. **Cost per update** is $O(30\,k^3 + 30\,\mathrm{nnz})$ for the selection
   plus one gathered solve, whether or not the budget is reached early.
7. **Singular at $m = 0$** (periodic): the constructor refuses a
   non-positive `mass`.
8. **The iteration bound, not the budget, ends the selection for
   $k \gtrsim n_{\max}/2$.** Measured on 128 points: $k = 64$ and $k = 96$
   both stop at $|M| = 54$ after 30 iterations (200 reach 64); the sensor
   error there is $3 \times 10^{-11}$, so the objective is unaffected, but
   `budget_reached` is `False`. The bound is deliberately not raised -- each
   iteration is a $k \times k$ solve on every update; if the budget matters,
   `selection_diagnostics()` says whether it was met.
9. **A mask larger than $k$ is not a valid input to the gathered solve.**
   The node never produces one (the seed is validated, the marking is
   capped); one supplied from outside is refused eagerly and NaN-poisoned
   under `jit`. `frozen_solver="cg"` accepts any mask.

## Stability Conditions

The frozen solve is a linear solve on a symmetric positive-definite block;
its conditioning is that of $\hat A_M$, bounded by $\kappa(\hat A)$ above.
There is no time-stepping and no stability limit.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `c` | `(n_max,)` | problem-defined | Wavelet coefficients of $u$ (physical, $u = W_n c$); zero off the active set |
| `mask` | `(n_max,)` | — | Boolean active set the current `c` was solved on |

No extra state: nothing can carry a stale per-mode value across a change of
active set.

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `dim` | int | 1 | — | Spatial dimension (structural) |
| `n_levels`, `n_coarse` | int | 6, 2 | — | Refinements and coarse points per axis (structural); together with `boundary` and `dim` they fix `n_max` |
| `order` | int | 4 | — | Interpolating order, one of 2, 4, 6 (structural) |
| `k` | int | `min(n_max, max(seed, 8, n_max // 16))` | — | Active-set budget (structural); `seed` is the level-0 count, $(2 n_c)^d$ periodic / $(2 n_c + 1)^d$ Dirichlet; `k < seed` is refused |
| `theta` | float | 0.42 | — | Source centre on axis 0; **trainable**, bounds $(0, 1)$, logit |
| `sigma` | float | 0.10 | — | Source width; **trainable**, positive, log |
| `mass` | float | 1.0 | — | $m$; `trainable=False`, baked into the operator |
| `sensor` | tuple of float | `(0.30,)`, `(0.30, 0.40)`, `(0.30, 0.40, 0.60)` | — | Sensor location, snapped to the nearest grid point -- on the circle for a periodic axis (`1.0` is point 0), among the interior points for a Dirichlet axis; `trainable=False`, baked into the sensor row |
| `preconditioner` | str | `"hybrid"` | — | `"hybrid"`, `"full"`, `"level"`, `"dk"` |
| `boundary` | str | `"periodic"` | — | `"periodic"` or `"dirichlet"` |
| `frozen_solver` | str | `"gather"` | — | `"gather"` or `"cg"` |
| base-class settings | — | — | — | `blindness_gate`, `on_blind`, `dtype`, the diagnostic constants |

Every entry is stored in `self.params`, so `cls(name=..., timestep=...,
**node.params)` rebuilds the node; `n_max` is derived and not stored.

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| — | — | — | None. The node is an edge source only (see *Known Limitations* 4) |

## References

- [@DeslauriersDubuc1989] Deslauriers, G., Dubuc, S. (1989). *Symmetric iterative interpolation processes*. Constructive Approximation, 5(1), 49–68. — The interpolating subdivision the basis is built from.
- [@CohenDahmenDeVore2001] Cohen, A., Dahmen, W., DeVore, R. (2001). *Adaptive wavelet methods for elliptic operator equations: convergence rates*. Mathematics of Computation, 70(233), 27–75. — Bulk chasing on the residual; the selection rule.
- [@Doerfler1996] Dörfler, W. (1996). *A convergent adaptive algorithm for Poisson's equation*. SIAM Journal on Numerical Analysis, 33(3), 1106–1124. — The bulk marking criterion inside the selection.
- [@DahmenKunoth1992] Dahmen, W., Kunoth, A. (1992). *Multilevel preconditioning*. Numerische Mathematik, 63(1), 315–344. — The level-based diagonal scaling (`preconditioner="dk"`) and why a diagonal scaling suffices.
- [@Blondel2022] Blondel, M. et al. (2022). *Efficient and modular implicit differentiation*. NeurIPS 35. — The implicit-function rule used by `ift_linear_solve` on the masked-CG path (`frozen_solver="cg"`); the default gathered path is plain reverse mode through `jnp.linalg.solve` and gives the same adjoint.
- [@Roache2002] Roache, P. J. (2002). *Code verification by the method of manufactured solutions*. Journal of Fluids Engineering, 124(1), 4–10. — The order study in `MADD-VER-014`.

## Verification Evidence

- `MADD-VER-014` (manufactured solution): observed spatial order over the
  finest pair of the 16/32/64/128/256 periodic 1-D ladder within
  $[-0.25, +1.0]$ of the declared 2 (measured 2.000); the Dirichlet
  (23 – 191) ladder measures 2.000 and the 2-D ($8^2$ – $32^2$) one 2.022 in
  the same test module, and a mis-specified source is shown to fail the gate.
- `MADD-VER-015` (regression against the full-basis solve): adaptive sensor
  reading within $10^{-2}$ of the full one and of the same sign across the
  source sweep at $k = n_{\max}/16$; full budget equals the dense solve to
  $10^{-10}$.
- Test files: `tests/nodes/adaptive/test_wavelet_node.py` (the seven-item
  author-facing contract, traceability, gradients with the active set held
  fixed, round trips, graph integration, the explicit-float32 dtype
  measurement, the seed-fits-the-budget pins in every dimension and both
  boundaries, refusal of an oversized set, construction inside a trace,
  the selection diagnostics, the sensor at the periodic seam, $64^2$ and
  $16^3$), `tests/nodes/adaptive/test_wavelet_engine.py` (transforms,
  assembly and its symmetry check, conditioning, both frozen solves, the
  NaN poison on an oversized mask, CDD and its iteration count),
  `tests/verification/test_wavelet_mms_order.py` (order studies and the
  two benchmarks).

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-09-21 | Ported onto the 0.4.0 `AdaptiveNode` API: parameters in the graph pytree, `{c, mask}` state, CDD as a `while_loop`, gathered frozen solve, dense full-basis gradient, declared and measured order 2 |
| 1.0.1 | 2026-09-22 | Merged-tree audit: `k` sized and validated against the real CDD seed (level 0, not the coarse block) -- 16 default configurations were silently truncating the gathered solve; an oversized set is refused / NaN-poisoned; the assembly checks symmetry before symmetrising; construction is legal inside a trace; `selection_diagnostics()`; a periodic sensor at 1.0 snaps to point 0 |
