# Authoring an AdaptiveNode

`maddening.nodes.adaptive.AdaptiveNode` is the base class for solvers that
keep a padded basis of `n_max` candidates and solve only on an **active set**
chosen from the data — wavelet thresholding, top-K by coefficient, residual
bulk-chasing, hierarchical refinement. It gives a subclass three things:

- a JAX-traceable `update` (fixed-size buffer plus boolean mask, so the
  active set can change every step without a retrace);
- the **frozen-active-set adjoint**: `jax.grad` through the node is the exact
  derivative of the objective with the active set held fixed, routed through
  `maddening.core.solver_utils.ift_linear_solve`;
- the **blindness diagnostics** that catch the one thing selection cannot
  see — a Palais fixed point of the problem's symmetry, where the frozen
  gradient is exactly zero in the direction an optimiser needs.

This guide is for subclass authors. The method and its evidence are in the
[algorithm guide](../algorithm_guide/nodes/adaptive_node.md) and in
`plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md`.

## When to use it

All of the following should hold:

1. The solve has a basis form $A(\theta)\, c = b(\theta)$ (or is reducible to
   a sequence of such solves).
2. You want $dJ/d\theta$ for an objective $J$ of the coefficients, and
   $\theta$ is something the graph parameter pytree should carry (a source
   position, a material constant — the things `fit` / `fim` identify).
3. The active set changes with $\theta$ or with the previous solution.

A static active set or a uniform grid does not need this: write a plain
`SimulationNode` and call `ift_linear_solve` directly.

## The contract

### State

| Field | Shape | Meaning |
|---|---|---|
| `c` | `(n_max,)` | coefficients; the base class zeroes them off the mask after every solve |
| `mask` | `(n_max,)` bool | the active set `c` was solved on |

Add fields with `extra_initial_state()`; they are carried through `update`
and can be written by `solve_frozen` (return them alongside `"c"`).

Physical parameters are **not** state. Pass them to `super().__init__` as
keyword arguments: floats become leaves of `params_pytree()`, ints and
strings stay structural, and `param_specs()` declares bounds, transforms and
trainability exactly as for any other node
(see the [parameters guide](../user_guide/parameters.md)).

### Hooks

```python
from maddening.core.params import ParamSpec
from maddening.core.solver_utils import ift_linear_solve
from maddening.nodes.adaptive import AdaptiveNode


class MyAdaptiveNode(AdaptiveNode):
    def __init__(self, name, timestep, *, theta, k, **kw):
        super().__init__(name, timestep, n_max=256, theta=theta, k=int(k), **kw)
        # build basis arrays once; keep them on self (static data), not in state

    def param_specs(self):
        return {**super().param_specs(),
                "theta": ParamSpec(bounds=(0.0, 1.0), transform="logit")}

    def compute_active_set(self, state, params, *, prev=None, is_cold_start=False):
        score = jnp.abs(self.rhs(params))          # any jnp rule with a fixed shape
        return score >= jnp.sort(score)[-self.params["k"]]

    def solve_frozen(self, state, mask, params):
        A, b = self.matrix(params), jnp.where(mask, self.rhs(params), 0.0)
        op = lambda v: jnp.where(mask, A @ jnp.where(mask, v, 0.0), v)   # identity off the mask
        return {"c": ift_linear_solve(op, b, solver="cg")}

    def objective(self, state, params):
        return self.sensor @ state["c"]
```

- **`compute_active_set(state, params, *, prev, is_cold_start)`** returns a
  boolean `(n_max,)` mask. It must be traceable with a fixed output shape.
  The base class wraps the result in `jax.lax.stop_gradient`; keep
  differentiable surrogates (soft thresholds, softmax scores) off this path
  anyway — a tangent leaking through the selection is the one silent failure
  the framework cannot detect. `prev` is the previous mask (`None` at cold
  start) for rolling or hysteresis rules; `is_cold_start` is true on the call
  from `initial_state`.
- **`solve_frozen(state, mask, params)`** is the differentiable half. Build
  the masked operator as a full-size operator that is the identity on
  inactive rows (so the buffer keeps its shape and the inactive block is
  trivially well-conditioned), mask the right-hand side, and call
  `ift_linear_solve`. Return `{"c": ...}` plus any extra state fields. The
  base class zeroes `c` off the mask.
- **`objective(state, params)`** is the scalar the diagnostics differentiate.
  `update` never calls it; leave it out (and construct with
  `blindness_gate=False`) if you do not need the diagnostics.
- **`compute_full_basis_gradient(state, params)`** defaults to
  `jax.grad(objective ∘ solve_frozen)` with an all-true mask; override if you
  have a cheaper full-basis solve.

`params` in the hooks is always the merged dict `{**self.params, **injected}`:
read the physical constants from it, never from `self.params`, or the graph's
injected (traced, differentiable) values are ignored.

### What the base class does

- `update(state, boundary_inputs, dt, *, params=None)`: select on the
  previous mask, freeze, solve, zero off-mask coefficients, store the mask.
  Ignores `boundary_inputs` and `dt`.
- `initial_state()`: cold start at the constructor parameters
  (`is_cold_start=True`), then — with `blindness_gate=True`, the default —
  raises `AdaptiveNodeBlindnessError` if `blindness_ratio` is below
  `blindness_threshold`. `gm.add_node` therefore fails loudly at a trap.
- `cold_start(params=None) -> (state, params)`: the same, but applies one
  `symmetry_break` before giving up and returns the (possibly perturbed)
  parameter pytree to assign into `gm.params["nodes"][name]`.

## Diagnostics

| Method | Cost | Use |
|---|---|---|
| `blindness_ratio(state, params=None) -> float` | 2 gradients, one full-basis | The full diagnostic. ~1 trustworthy, ~0 trapped, `1.0` sentinel when the full gradient itself vanishes |
| `is_trapped_at(state, params=None, *, eps=1e-3) -> bool` | 2 frozen gradients + 1 full | Cheap binary check between optimiser steps; reliable for exact traps, not for partial blindness |
| `symmetry_break(state, params=None, *, delta=None) -> params` | 1 full gradient | Step `delta` (default `blindness_break_delta`) along the unit full-basis gradient; trainable leaves only |

All three are host-side (they return Python scalars or concrete pytrees) and
are never called inside the traced step.

The constants — `blindness_threshold = 0.7`, `blindness_break_delta = 0.05`,
`D_threshold = 5` — are class attributes with constructor overrides. They are
deliberately not parameter leaves: they steer diagnostics, they are not
physics a fit could identify. `D_threshold` is advisory: above that many
trainable parameters, run `is_trapped_at` between optimiser steps rather than
relying on the cold-start gate alone.

## Why the trap exists, in one paragraph

If the operator, source and objective share a symmetry $G$ and the selection
scores modes by a $G$-invariant functional of $(A, b)$, then at a symmetric
$\theta_*$ the active set is $G$-stable and, by Palais' principle of symmetric
criticality, the frozen gradient lies in the tangent space of the symmetric
manifold. Every selection rule is blind in the same direction, so mitigation
has to act on $\theta$, and isotropically random noise does not do it — the
step must be anisotropic. `symmetry_break` uses the full-basis gradient
direction, which does have the transverse component. In the test toy
(Gaussian source on $(0,1)$, sine basis, top-$|b|$), $\theta = 0.5$ is such a
point: every selected mode has $db_k/d\theta = 0$ there, `jax.grad` returns
exactly zero, and one `symmetry_break` moves to $\theta = 0.45$ where the
ratio is above 0.7.

## Failure modes

- **`AdaptiveNodeBlindnessError` from `add_node` / `initial_state`.** The
  constructor parameters sit at a trap. Use `cold_start()` and seed
  `gm.params` with the returned pytree, or perturb the parameters yourself.
- **`jax.grad` disagrees with finite differences.** If the disagreement is
  at isolated parameter values, you are at a kink (active-set change): both
  answers are valid one-sided quantities. If it is everywhere, a tangent is
  leaking through `compute_active_set` (a soft score) or `solve_frozen` reads
  a constant from `self.params` instead of `params`.
- **NaN / Inf from the solve.** The masked operator is singular on the active
  set, the mask is empty, or `solver="cg"` was used on a non-SPD operator.
  The identity-off-mask construction keeps the inactive block harmless; the
  active block is the subclass's responsibility.
- **`verify_node` reports `params_effective` failing.** `update` did not read
  the parameter from the injected `params` dict.

## Worked example

`tests/nodes/adaptive/_toys.py` contains two complete subclasses used by the
test suite: `PoissonSineTopKNode` (the spike's 1-D problem, diagonal operator,
the known trap at $\theta = 0.5$) and `MaskedDenseNode` (a dense SPD operator,
so the Krylov adjoint is checked against `jnp.linalg.solve` on the active
sub-block). They are test fixtures, not public API; read them as the pattern.

## Out of scope in 0.4

The wavelet subclass (Deslauriers–Dubuc filters, Dahmen–Kunoth scaling,
Cohen–Dahmen–DeVore bulk-chasing, the sparse-tree representation) and the
multigrid preconditioner are post-1.0 research items; nothing in the base
class presumes them.
