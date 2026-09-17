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
- the **cold-start diagnostics**: how much of the full-basis gradient your
  active set reproduces (`gradient_capture_ratio`), and whether the parameters
  sit at a Palais fixed point of the problem's symmetry, where the frozen
  gradient is exactly zero in the direction an optimiser needs
  (`is_trapped_at`).

> **Stability.** `AdaptiveNode`, `AdaptiveNodeBlindnessError` and
> `ift_linear_solve` are **not** `STABLE`. The node surfaces are
> `@stability(EVOLVING)` and `ift_linear_solve` is back at its pre-0.4
> `EXPERIMENTAL`. Nothing has shipped, so lowering the promise costs nothing
> and raising it later would not be free; the 0.4.0 API freeze picks the final
> level, informed by the open questions at the end of this guide.

> **What the gradient is.** Exact *within* an active-set region, and blind to
> the set's dependence on the parameters. The frozen-set objective **jumps**
> across a region boundary, so it is not locally Lipschitz there and no Clarke
> subgradient exists: a step that crosses a switch carries a first-order
> error. Measured on the 1-D sine toy at `n_max = 256`, `k = 16`, the integral
> of the returned gradient over `theta` in `[0.40, 0.50]` is `-1.5808e-3`
> against a true change of `-2.3598e-3` — a 33 % shortfall equal to the sum of
> the 27 jumps crossed. It is negligible with a large basis budget (the spike
> measured `4e-8` at `k = 64`). Anomaly `MADD-ANO-003`.

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

`n_max` is **structural and is not stored in `self.params`**. Every
serialisation path rebuilds a node with `cls(name=..., timestep=...,
**node.params)`, so a parameter named `n_max` would arrive twice —
`TypeError: got multiple values for keyword argument 'n_max'`. If your
subclass's basis size must survive a config / USD round trip, declare it as
your own integer parameter and forward it:

```python
def __init__(self, name, timestep, *, n=256, **kw):
    super().__init__(name, timestep, n_max=n, n=int(n), **kw)
```

The diagnostic settings (`blindness_gate`, `on_blind`,
`gradient_capture_threshold`, `blindness_break_delta`, `D_threshold`) *are*
recorded in `self.params` so they round-trip, and `params_pytree()` excludes
them so a fit never sees them as leaves.

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

    # If any expression inside solve_frozen is singular on the inactive
    # entries, sanitise its *input* -- see "Masked operands" below:
    #   d = self.mask_safe(mask, diagonal, fill=1.0)

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
  `compute_active_set` and `solve_frozen` are `@abstractmethod`: a subclass
  that forgets one fails at construction with a `TypeError`, not at trace
  time.
- **`objective(state, params)`** is the scalar the diagnostics differentiate.
  `update` never calls it; it is deliberately *not* abstract — leave it out
  (and construct with `blindness_gate=False`) if you do not need the
  diagnostics.
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
  `check_gradient_capture()` at those parameters.
- `check_gradient_capture(params=None, *, state=None, on_blind=None)`: measure
  the ratio and apply the policy. A low ratio **warns** (naming the measured
  value, the threshold and the remedies) and raises only when `is_trapped_at`
  confirms a Palais fixed point — the one cause the diagnostics can establish
  — or when the node was built with `on_blind="raise"`. `gm.add_node`
  therefore still fails loudly at a trap, and no longer rejects a small
  active-set budget, which is the point of an adaptive solver.
  **It evaluates the parameters you hand it**: `initial_state` sees the
  constructor's, so after seeding a graph call
  `node.check_gradient_capture(gm.params["nodes"][name])` — that is the point
  `sysid.fit` optimises.
- `cold_start(params=None) -> (state, params)`: the same, but applies one
  `symmetry_break` before giving up and returns the (possibly perturbed)
  parameter pytree to assign into `gm.params["nodes"][name]`. It is the remedy
  for a **trap**, not for a small budget.

## Diagnostics

| Method | Cost | Use |
|---|---|---|
| `gradient_capture_ratio(state, params=None) -> float` | 2 gradients, one full-basis | How much of the full-basis gradient the frozen set reproduces. ~1 trustworthy, ~0 either a trap **or** too small a budget, `1.0` sentinel when the full gradient itself vanishes. The active set is re-selected at `params` (never read from a stale `state["mask"]`) |
| `is_trapped_at(state, params=None, *, eps=1e-3) -> bool` | 2 frozen gradients + 1 full | The only check that can *establish* a symmetry trap; reliable for exact traps, not for partial blindness |
| `symmetry_break(state, params=None, *, delta=None) -> params` | 1 full gradient | Step `delta` (default `blindness_break_delta`) along the unit full-basis gradient; trainable leaves only |
| `check_gradient_capture(params=None, ...) -> float or None` | as above, memoised | The policy wrapper the cold start uses; call it yourself at `gm.params["nodes"][name]` |

`blindness_ratio()` is a deprecated alias of `gradient_capture_ratio()` and
warns. All of them are host-side (they return Python scalars or concrete
pytrees) and are never called inside the traced step.

**A low ratio is usually a budget, not a trap.** At a fixed, entirely
non-symmetric point on the 1-D sine toy the ratio is a function of the budget
alone — 0.16 / 0.57 / 0.85 / 1.00 at `k = 4 / 8 / 16 / 32`, at every `n_max`.
Remedies, in order: raise the budget; lower `gradient_capture_threshold=`;
`on_blind="ignore"` or `blindness_gate=False`. `cold_start()` and
`symmetry_break()` are *not* remedies here — they move along the full-basis
gradient, which at a budget-limited point lowered the audited ratio from 0.565
to 0.060.

The constants — `gradient_capture_threshold = 0.7` (deprecated alias
`blindness_threshold`), `blindness_break_delta = 0.05`, `D_threshold = 5` —
are class attributes with constructor overrides. They are not parameter
leaves: they steer diagnostics, they are not physics a fit could identify.
`D_threshold` is advisory: above that many trainable parameters, run
`is_trapped_at` between optimiser steps rather than relying on the cold start
alone.

**Cost.** The diagnostic costs two gradient evaluations, one of them
full-basis — measured 8–10x the cost of an unguarded `initial_state()` — and
`initial_state()` is called from `GraphManager.add_node` / `reset_state`, the
profiler, the REST API, the sharded-node paths, the FMI model description and
the hypothesis strategies. It is memoised per instance and parameter point, so
those paths pay once; it is skipped entirely when the gate is off, when
`on_blind="ignore"`, or when the node has no trainable parameter leaf. To turn
it off process-wide, set `MADDENING_ADAPTIVE_DIAGNOSTICS=0` in the environment
or call `maddening.nodes.adaptive.set_adaptive_diagnostics(False)` (it returns
the previous setting, so you can restore it).

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
ratio is above 0.7. That is the case — and the only case — the cold-start
check still refuses to construct through.

## Failure modes

- **`AdaptiveNodeBlindnessError` from `add_node` / `initial_state`.**
  `is_trapped_at` confirmed a Palais fixed point at the constructor
  parameters (or you asked for `on_blind="raise"`). Use `cold_start()` and
  seed `gm.params` with the returned pytree, or perturb the parameters
  yourself.
- **A `UserWarning` about the gradient-capture ratio.** Not a trap: your
  active-set budget does not reproduce the full-basis gradient at these
  parameters. See *Diagnostics* above for the remedies, in order.
- **`jax.grad` disagrees with finite differences.** If the disagreement is at
  isolated parameter values, the finite difference crossed an active-set
  switch and is not a valid oracle there: the objective **jumps**, so the
  difference quotient diverges as `1/h` while the returned gradient stays
  finite. Verify by re-selecting the mask at both ends of the step; if it
  changed, shrink the step or compare against the branch the forward pass
  selected (`_solve_and_pack` with the mask held fixed). If the disagreement
  is everywhere, a tangent is leaking through `compute_active_set` (a soft
  score) or `solve_frozen` reads a constant from `self.params` instead of
  `params`.
- **Masked operands: a clean value and a `NaN` gradient.** `jnp.where(mask, c,
  0)` protects the *value*, never the *tangent*. If `solve_frozen` evaluates
  an expression that is singular on the inactive entries — `jnp.sqrt` of
  something negative there, a division by a diagonal that was zeroed off-mask
  — the forward pass is exactly right and `jax.grad` returns `NaN`. **The base
  class cannot repair this**: by the time it masks your output, the tangent of
  your expression is already poisoned. Sanitise the *input* of the unsafe
  operation with `self.mask_safe(mask, operand, fill=1.0)` (the double-`where`
  idiom):

  ```python
  d = self.mask_safe(mask, diagonal, fill=1.0)      # never 0 off the mask
  c = ift_linear_solve(lambda v: d * v, rhs, solver="cg")
  ```

  The base class warns when it can see the damage — when you leave the masking
  to it and the raw coefficients are non-finite — but a subclass that masks
  its own output hides the value and keeps the `NaN` gradient.
- **NaN / Inf from the solve itself.** The masked operator is singular on the
  active set, the mask is empty, or `solver="cg"` was used on a non-SPD
  operator. The identity-off-mask construction keeps the inactive block
  harmless; the active block is the subclass's responsibility.
- **`verify_node` reports `params_effective` failing.** `update` did not read
  the parameter from the injected `params` dict.

## Worked example

`tests/nodes/adaptive/_toys.py` contains two complete subclasses used by the
test suite: `PoissonSineTopKNode` (the spike's 1-D problem, diagonal operator,
the known trap at $\theta = 0.5$) and `MaskedDenseNode` (a dense SPD operator,
so the Krylov adjoint is checked against `jnp.linalg.solve` on the active
sub-block). They are test fixtures, not public API; read them as the pattern.

## Open questions for the 0.4.0 API freeze

Both are deliberately left open: changing them now would churn signatures the
freeze is about to settle anyway, and neither is a correctness bug.

1. **Should `ift_linear_solve` expose `restart` / `max_steps` /
   `stagnation_iters`, and own its error type?** Today the wrapper hard-codes
   `restart = min(N, 50)` and `max_steps = max(4 * restart, 100)` and leaves
   `stagnation_iters` at the lineax default. A non-convergent solve surfaces
   `equinox._errors._EquinoxRuntimeError` eagerly and `jax.errors.JaxRuntimeError`
   under `jit` / `scan` — third-party types reachable from a MADDENING
   function — and the remedy lineax prints ("try increasing `stagnation_iters`
   or `restart`") is not reachable through the signature. Adding the keywords
   is additive; wrapping the failure in a MADDENING-owned error type is not.
   The function is back at `EXPERIMENTAL` until this is decided.
2. **Should the hooks take `boundary_inputs` and `dt`?** `update` does
   `del boundary_inputs, dt`, and `compute_active_set` / `solve_frozen` /
   `objective` take only `(state, mask, params)`. An `AdaptiveNode` can
   therefore only ever be an edge *source*: to consume an edge's input, or to
   be time-dependent, a subclass must override `update` and re-implement the
   `stop_gradient` wrap, the two shape validations and `_solve_and_pack`. The
   same limitation makes an `AdaptiveNode` inert to its partner's interface
   values inside a coupling group. Adding the arguments keyword-only with
   defaults would keep existing overrides binding, but it changes three hook
   signatures, so it belongs to the freeze.

## Out of scope in 0.4

The wavelet subclass (Deslauriers–Dubuc filters, Dahmen–Kunoth scaling,
Cohen–Dahmen–DeVore bulk-chasing, the sparse-tree representation) and the
multigrid preconditioner are post-1.0 research items; nothing in the base
class presumes them.
