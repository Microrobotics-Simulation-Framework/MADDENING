# AdaptiveNode developer guide

`AdaptiveNode` is the basis-agnostic framework primitive for adaptive
PDE solvers in MADDENING.  It provides the **frozen-active-set adjoint
infrastructure**, a **blindness diagnostic** that detects when a
state has fallen into a Palais fixed point, and an **anisotropic
symmetry-break protocol** that escapes such traps in one perturbation.

This guide is intended for **subclass authors**: you have an adaptive
PDE solver — wavelet, sparse grid, hierarchical RBF, hat-function,
or anything else — and you want to plug it into MADDENING with
adjoint correctness, trap robustness, and the SimulationNode
contract preserved.

> The framework's design rationale is documented in the seven-round
> spike series at
> `plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md`.  This guide is
> the public-facing summary; the spike memo is the source of
> truth for **why** each piece exists.

## When to subclass `AdaptiveNode`

The framework is right for your problem if **all** of the following
hold:

1. Your PDE solve has a natural **basis representation**
   ``A(θ) c(θ) = b(θ)`` where ``A``, ``c``, ``b`` are basis-
   coefficient quantities.
2. You want **gradients** ``dJ / dθ`` for an objective ``J`` that
   depends on the basis coefficients (sensor reading, integral
   quantity, control objective).
3. You want the solve to use only a **subset** of the basis — an
   active set ``Λ ⊆ {1, …, N_max}`` — and the choice of ``Λ`` adapts
   with ``θ`` (or with previous-step state).
4. You're willing to provide a way to compute the **full-basis**
   gradient ``∇_θ J_full(state)``.  This is the price of trap
   detection; subclass authors who genuinely cannot provide it can
   override `blindness_ratio` and `symmetry_break` directly with
   problem-specific approximations.

If your PDE solve is uniform-grid or has a static active set,
`AdaptiveNode` is overkill — use a plain `SimulationNode`.

## The Selection-Equivariance Theorem

Before reading the API, understand the theorem that pins down which
trap mitigations are valid.  Stated informally:

> Let ``G`` be the symmetry group of the problem
> (operator + source + boundary conditions).  At any state ``θ_*``
> on the fixed-point set ``Fix(G)``, if your selection criterion
> scores modes by a ``G``-invariant functional of ``(A, b)`` —
> top-|b|, top-|c|, top-|residual|, anything — then the frozen-set
> adjoint at ``θ_*`` has **zero component transverse to ``Fix(G)``**.
> The optimizer is **selection-blind** to the direction it needs
> to move to escape.

The corollary the framework relies on: **no selection criterion can
escape the trap**.  The only valid escape is an **anisotropic
perturbation** of ``θ`` transverse to ``Fix(G)``.  The framework
implements this via :meth:`AdaptiveNode.symmetry_break`, which uses
the **full-basis** gradient direction (which by Palais 1979 lies in
``T_θ Fix(G)`` at the trap, hence transverse perturbation is in
``+g_full``).

Chen-Ziyin (2023) proves the same conclusion for ML loss landscapes:
**isotropic SGD noise cannot escape Type-II saddles**.  The Palais
trap is a Type-II saddle.

Full theorem statement and proof sketch: `maddening.nodes.adaptive.base`
module docstring.

## The API surface

### Configuration constants

Override either as a class-level class-attr on your subclass or as a
constructor kwarg:

| Constant | Default | Meaning |
|---|---|---|
| `blindness_threshold` | `0.7` | States with `blindness_ratio < threshold` trigger `symmetry_break` at cold start.  Round-6 Investigation 1 minimised expected cost at 0.7. |
| `blindness_break_delta` | `0.05` | Perturbation magnitude.  Round-7 Investigation 2: 1D min 0.03, 2D min 0.001 — default 0.05 covers both with margin. |
| `D_threshold` | `5` | Dimensionality above which runtime monitoring (not just cold-start) is recommended.  Round-5 Investigation 3 analytical estimate. |

### Subclass hooks (you implement these)

- :meth:`compute_active_set(state, *, prev, is_cold_start)` —
  return the boolean mask of shape ``(N_max,)``.
- :meth:`solve_frozen(state, mask)` — perform the masked solve.
  Typical pattern: build the masked operator, call
  :func:`maddening.core.solver_utils.ift_linear_solve`, assemble the
  returned coefficient vector into the state dict.
- :meth:`compute_full_basis_gradient(state)` — return
  ``∇_θ J_full(state)``.  Used by the blindness diagnostic.
- :meth:`_initial_state_impl()` — your "raw" initial state, BEFORE
  the cold-start gate runs.
- :meth:`_get_theta(state)`, :meth:`_set_theta(state, theta_new)` —
  accessors over the theta field in state.

You also typically override `_sensor(state)` to return the scalar
sensor functional your objective depends on.

### Base-class machinery (you get these for free)

- :meth:`initial_state()` — runs `_initial_state_impl`, applies the
  cold-start blindness gate, perturbs once if needed, raises
  :exc:`AdaptiveNodeBlindnessError` on persistent trap.
- :meth:`blindness_ratio(state)` — the full diagnostic.
- :meth:`is_trapped_at(state)` — cheap binary check (re-thresholded
  FD).  Reliable for exact-trap detection; unreliable as a
  continuous estimator.
- :meth:`symmetry_break(state, delta)` — anisotropic perturbation.
- :meth:`update(state, boundary_inputs, dt)` — default flow is one
  `compute_active_set` call + one `solve_frozen` call.  Override if
  your subclass needs more elaborate dynamics (e.g., a CDD outer
  loop).

## Worked examples

Two toy subclasses ship with the framework as worked examples.
Read them in order:

1. **:class:`maddening.nodes.adaptive.TopKAdaptiveNode`** — the simpler
   of the two.  Sine eigenbasis (non-local), top-K selection on either
   ``|b|`` or ``|c|``.  Reproduces the round-4 wrong-sign failure
   mode at boundary-θ with `selection_quantity='b'`, and avoids it
   with `selection_quantity='c'` (the default).  Use this as the
   pattern for a subclass that has a diagonal-operator basis.

2. **:class:`maddening.nodes.adaptive.HierarchicalHatAdaptiveNode`** —
   local dyadic hat basis with the level-0 root always included.
   Galerkin projection of the FD Dirichlet Helmholtz operator.
   Demonstrates the round-4 locality theorem (no wrong-sign failure
   at any boundary-θ tested) and the round-6
   BCOO + lineax compatibility path (the masked operator is
   represented as a `jax.experimental.sparse.BCOO` matrix and
   passed to :func:`ift_linear_solve`).  Use this as the pattern for
   a subclass with a non-diagonal basis where you want sparse storage.

## Common failure modes

### `AdaptiveNodeBlindnessError` at construction

```
AdaptiveNodeBlindnessError: State has blindness ratio 0.000 below
threshold 0.700 even after one symmetry_break perturbation of
delta=0.05.  The optimizer appears to be at a persistent Palais
fixed point; perturb your initial theta and retry.
```

Cause: your `theta_init` (or whatever drives your subclass's
`_initial_state_impl`) is sitting on **and is structurally bound to**
``Fix(G)`` — one perturbation of ``blindness_break_delta`` did not
move it off.  This is typically a problem-symmetry issue: try
randomizing `theta_init` by a small amount, or increasing
`blindness_break_delta` in the constructor.

### `jax.grad` returns wildly wrong values

Most likely cause: your selection in `compute_active_set` is
not wrapped in :func:`jax.lax.stop_gradient`.  The mask construction
is supposed to be a non-differentiable selection; failing to mark it
as such can leak phantom gradients through the boolean mask.  Even
when the underlying operations (`top_k`, `>=`) are naturally
non-differentiable to JAX, **wrap the mask in `stop_gradient`** for
documentation and to guard against future mask-construction
changes that might use a differentiable surrogate.

### Solve returns NaN / Inf

Most likely cause: the masked operator passed to `ift_linear_solve`
is singular or near-singular.  Confirm:

- The active mask has at least one True entry.
- The basis is linearly independent on the active set.
- For `solver='cg'`, the matrix is actually SPD on the active
  subblock.
- The off-diagonal blocks of A on the (active, inactive) pair are
  zero (so the active solve is decoupled from the inactive zero
  coefficients).

## References

- Palais, "The Principle of Symmetric Criticality,"
  *Comm. Math. Phys.* 69 (1979), 19–30.  The theorem that pins
  down what gradient blindness IS.
- Chen, Ziyin et al. (2023), arXiv:2303.13093 — "Type-II saddles"
  immune to isotropic SGD noise; anisotropic perturbation required.
- Cohen, Dahmen, DeVore, "Adaptive wavelet methods for elliptic
  operator equations: Convergence rates," *Math. Comp.* 70 (2001),
  27–75.  The residual bulk-chasing criterion that
  WaveletAdaptiveNode (post-1.0) will use.
- `plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md` — the
  seven-round design spike.
