# WaveletAdaptiveNode: the worked AdaptiveNode subclass

`maddening.nodes.adaptive.WaveletAdaptiveNode` is the first
[`AdaptiveNode`](adaptive_node.md) subclass with a real basis, and it is
written to be read as the worked answer to the base class's author-facing
contract. This page is about *using* it and about what porting it onto the
0.4.0 API taught us; the method is in the
[algorithm guide](../algorithm_guide/nodes/wavelet_adaptive_node.md).

> **Stability.** `EXPERIMENTAL`, one level below the `EVOLVING` base class.
> The wavelet engine under `maddening.nodes.adaptive.wavelets` is importable
> and tagged `EXPERIMENTAL` function by function; the node is the surface
> the freeze round covers.

## Quick start

```python
import jax
from maddening.nodes.adaptive import WaveletAdaptiveNode

node = WaveletAdaptiveNode("wavelet", 1.0, n_levels=6, theta=0.42)
state = node.initial_state()            # cold start + the gradient-capture check
state = node.update(state, {}, 1.0)     # one CDD selection + frozen solve
u = node.field(state)                   # u on the grid, shape (n_max,)
J = node.objective(state, {})           # u at the sensor

# theta and sigma are graph-parameter leaves; the gradient is the frozen-set adjoint
dJ = jax.grad(lambda th: node.objective(
    node.update(state, {}, 1.0, params={"theta": th}), {}))(0.42)
```

`dim` is 1, 2 or 3; the grid has `n_coarse * 2**n_levels` points per axis
(periodic) or `(n_coarse + 1) * 2**n_levels - 1` interior points
(`boundary="dirichlet"`); `n_max` is that to the power `dim`, and the default
budget is `k = min(n_max, max(seed, 8, n_max // 16))`, where `seed` is the
CDD seed -- every level-0 function, the coarse block *plus the first detail
band*, `(2 * n_coarse) ** dim` periodic or `(2 * n_coarse + 1) ** dim`
Dirichlet. A `k` below the seed is refused with both numbers in the message.
`k = n_max` turns adaptivity off.

Inside a graph the node is an edge **source**: it declares no boundary
inputs and `update` reads neither `dt` nor `boundary_inputs`, because the
problem is steady. Edges out of `c` work as for any node; to expose the
field rather than the coefficients, map through `node.field` in the consumer
or add a transform.

## The seven contract items, answered

The 0.4.0 adaptive-node audit listed seven things a second subclass author
gets wrong that the base class does not stop. How the wavelet node handles
each, and the test that pins it (`tests/nodes/adaptive/test_wavelet_node.py`
unless stated):

| # | Item | How the wavelet node handles it | Pinned by |
|---|------|----------------------------------|-----------|
| 1 | Empty active set at cold start | CDD is seeded with level 0 (the coarse block plus the first detail band, which the budget is validated to hold) and only grows; the selection reads the *parameters*, never `c`, so the all-zero cold-start coefficients cannot empty it. `is_cold_start` is accepted and ignored | `test_the_cold_start_set_is_never_empty_because_the_coarse_level_seeds_it`, `test_the_selection_is_non_empty_and_within_budget_across_the_source_range`, `test_the_default_budget_holds_the_whole_level_zero_seed_so_no_coefficient_is_dropped` |
| 2 | Non-boolean mask | The mask is built from boolean operations on a boolean seed (`mask \| add`); the `k == n_max` branch returns `jnp.ones(..., bool)` | `test_the_hook_itself_returns_a_boolean_mask_not_only_the_validated_state`, and `test_cdd_keeps_the_coarse_level_returns_a_bool_mask_and_never_exceeds_the_budget` in `test_wavelet_engine.py` |
| 3 | Basis array baked from a trainable parameter | `theta` and `sigma` enter only through the right-hand side, recomputed from `params` on every call. `mass` (operator) and `sensor` (sensor row) *are* baked: both are `ParamSpec(trainable=False)` and published through `static_data` with `static_data_deps`, so `compile()` refuses a graph that unfreezes them | `test_gradients_with_respect_to_the_trainable_leaves_match_finite_differences`, `test_mass_and_sensor_are_frozen_and_their_statics_are_declared`, `test_compile_refuses_a_graph_that_unfreezes_a_baked_parameter`, `test_the_full_basis_gradient_is_not_bitwise_zero_for_any_trainable_leaf` |
| 4 | `update` ignores `dt` / `boundary_inputs` | Correct for a steady elliptic solve: two different `dt` and an unexpected boundary input give bitwise the same state, and the node declares no boundary inputs. The consequence — edge source only — is documented, and a time-dependent wavelet node would need the base class's open freeze question answered | `test_update_is_a_steady_solve_that_reads_neither_dt_nor_boundary_inputs` |
| 5 | `extra_initial_state()` carried unmasked | The state is exactly `{c, mask}`; `extra_initial_state()` is `{}`. Nothing can carry a stale per-mode value across an active-set change | `test_the_state_is_exactly_c_and_mask_so_nothing_can_carry_a_stale_value` |
| 6 | Hooks receive non-numeric settings in `params` | The hooks index the named leaves (`params["theta"]`, `params["sigma"]`) and never `tree.map` over the dict; the merged dict with its strings and bools and the numeric-only dict give identical results. The jitted kernels take the *arrays*, never the dict | `test_the_hooks_read_named_leaves_and_tolerate_the_merged_dict_with_its_strings` |
| 7 | `compute_full_basis_gradient` assumes an all-true mask is a valid solve | It is not, for the gathered solve: the buffer holds exactly `k` functions. A larger *concrete* mask is refused by name and a traced one is NaN-poisoned (it used to be silently truncated to a plausible wrong answer -- the mechanism behind the audit's 5x gradient error). The node overrides the hook with a dense solve on `A`; the test shows the base default is refused for `frozen_solver="gather"` and agrees for `"cg"`, so the override cannot be removed as redundant | `test_the_full_basis_gradient_override_matches_a_dense_finite_difference`, `test_the_base_default_full_basis_gradient_is_refused_for_the_gathered_solve_which_is_why_it_is_overridden`, `test_an_active_set_larger_than_the_budget_is_refused_eagerly_and_poisoned_under_jit`, and `test_gather_solve_poisons_a_mask_larger_than_its_buffer_instead_of_truncating` in `test_wavelet_engine.py` |

## Declared order, measured

The full-basis solve is the second-order finite-difference solution in a
different basis, so the node declares `DiscretizationOrder(spatial=2.0)` and
`tests/verification/test_wavelet_mms_order.py` measures it with
`maddening.testing.mms`: a manufactured field, its source derived by autodiff,
the level refined at `k = n_max` so only the discretisation error remains.
Periodic 1-D (16 – 256 points) measures 2.000; the Dirichlet basis
(23 – 191 interior points) measures 2.000 and 2-D ($8^2$ – $32^2$) 2.022
(`MADD-VER-014`). The manufactured solutions are checked for the properties
that let them see a broken scheme — a non-vanishing fourth derivative, no
symmetry about the centre, and for Dirichlet a non-zero second derivative at
*both* walls (a profile flat at the walls let a second-order closure measure
4.08 on another node this release) — and a source with the wrong sign on the
mass term is shown to fail the gate. Adaptive truncation is a separate claim
(`MADD-VER-015`): within 1e-2 of the full solve at `k = n_max / 16`, same
sign across the source sweep, no rate in `k`.

## What the merged-tree audit changed (1.0.1)

The pre-release audit found that the CDD seed is *every level-0 function*
-- the coarse block plus the first detail band -- while `k` was validated
against `n_coarse ** dim` and the default `k` sized from that. On 16 default
configurations the seed exceeded `k`, `gather_solve` kept the first `k`
active functions and dropped the rest, and nothing objected: at `dim=2,
n_levels=2` the mask had 16 functions in a buffer of 8, `c` was 4% off the
masked solve and `dJ/dtheta` was five times too small, while the capture
ratio read 0.97 because it measured through the same truncated solve. Now:

- the seed is counted from the level labels it is built from, `k >= seed`
  is validated (message names both numbers and the remedy), the default is
  `max(seed, 8, n_max // 16)`, and the count is re-checked against the
  assembled seed;
- an oversized concrete mask is refused on every eager path
  (`_refuse_oversized_mask`), and `gather_solve` poisons an oversized block
  with NaN under a trace -- the truncation cannot happen silently;
- `assemble_operator` measures `max|A - A^T| / max|A|` *before* symmetrising
  and refuses above `SYMMETRY_TOL = 1e-12` (correct assembly ~1e-16, a
  one-sided stencil 1.07), so the symmetrisation can no longer turn a
  first-order stencil into a consistent second-order one that passes the
  order gate;
- a periodic sensor at `1.0` snaps to grid point 0, its periodic image,
  rather than to `(side - 1) / side`.

## Construction inside a trace

Every constant of the node is built on the host from static settings
(`assemble_operator` runs under `jax.ensure_compile_time_eval`; `levels` and
`diagonal` are NumPy arrays), so a function that builds a fresh node -- or a
fresh `GraphManager` holding one -- per call traces under `jax.jit`. This is
what a `residual_fn` for `sysid.fim` that constructs its graph inside the
call needs; it used to fail in the assembly with a
`TracerArrayConversionError` whose traceback pointed at `sysid.py`. Build
with `blindness_gate=False` inside a trace: the gate's diagnostics return
host floats and cannot run traced. Pinned by
`test_the_node_can_be_constructed_inside_a_jit_trace` and
`test_a_fresh_graph_holding_the_node_can_be_built_inside_the_fim_trace`.

## Observing the selection: `selection_diagnostics`

```python
node.selection_diagnostics()
# {'active': 8, 'k': 8, 'outer_iterations': 4, 'max_outer': 30,
#  'budget_reached': True, 'resolved': False}
WaveletAdaptiveNode("w", 1.0, n_levels=6, k=64, blindness_gate=False).selection_diagnostics()
# float64: {'active': 54, 'k': 64, 'outer_iterations': 30, 'max_outer': 30,
#           'budget_reached': False, 'resolved': False}
```

The CDD loop has three exits: the budget, the 30-iteration bound, and a
step that finds nothing above the rounding floor to mark (`resolved`). For
`k` above about `n_max / 2` the bound is the one taken in float64, and in
float32 the floor usually comes first: each Dörfler step
marks a fixed fraction of the *remaining* residual, so the steps shrink once
the source is resolved. Measured on 128 points at the default source:
`k = 64` and `k = 96` both stop at `|mask| = 54` (200 iterations reach 64;
the sensor-reading error at 54 is 3e-11, so the objective does not care).
The bound is deliberately not raised -- every iteration is a `k x k` solve
on every update. If reaching the budget matters, read `budget_reached`; the
engine-level `cdd_select_with_iterations` returns the same count.

## What the phase-3 audit changed (1.1.0)

- **Conditioning is checked.** A small positive `mass` used to give a
  finite, wrong reading in the default float32 (128 points: `mass=1e-6`
  read `J = 6.7e5` against an FFT reference of `1.77e5`, `mass=1e-8`
  `-2.9e16`), silently, or with a warning blaming the active-set budget at
  `k = n_max`. The constructor now bounds the solve's relative error by
  `condition_number * eps(dtype) + physical_condition_number * eps(float64)`
  (`node.solve_error_bound()`) and refuses above `CONDITION_LIMIT = 1e-3`,
  saying which term is too large, whether float64 would carry it, and the
  smallest mass that would. In float32 that is a periodic mass below about
  `2e-3`; in float64 about `1e-8` to `6e-8`, where the float64 assembly of
  the Galerkin product becomes the limit. The full-basis gradient now
  solves the preconditioned system, so at `k = n_max` the capture ratio is
  1 where it used to drift in float32.
- **The selection no longer depends on rounding.** The source is centred
  on every axis but the first, so residuals of mirror-image functions tie
  to rounding and the last bits decided which one survived the cap: eager
  and compiled evaluations, and a Python float and its float32-array
  spelling, selected different sets at the same parameters, and a 1e-6
  sweep of `theta` switched set on 15 of 39 steps (a finite difference
  across one read -4.57 against `grad` -0.0101). The marking step now
  never marks a residual at or below `rounding_floor` and breaks ties
  within it by basis index, and parameter leaves are cast to the node's
  dtype before the source reads them. The diagnostics therefore describe
  the solve the graph runs. The set still changes at genuine crossings --
  isolated parameter values -- and a gradient step across one still
  misses a jump.
- **The periodic source is periodised**, so a source near the seam no
  longer loses the part that should wrap round (the same problem shifted
  across the seam read 28% low, with `dJ/dtheta` of the wrong sign).
- **Structural counts are validated, not truncated** (`n_levels=6.9` used
  to build 64 points), and `update` / `selection_diagnostics` refuse an
  unknown parameter key (`{"thetta": 0.9}` used to return the
  constructor-theta answer).

## Using a different source: `source_field`

Override `source_field(params)` to solve for another forcing. It must read
every parameter it depends on from `params` (never `self.params`), return
the forcing sampled on `grid_coordinates()` flattened row-major, and stay
traceable. The leaves it receives are already cast to the node's dtype. On
a periodic domain it should be periodic itself: the default one sums the
Gaussian's periodic images. The MMS study is exactly this: a subclass returning the
manufactured source, with `blindness_gate=False` because its source ignores
`theta` and `sigma` and the base class would (correctly) warn that their
full-basis gradient is bitwise zero.

## Performance notes

- **Compile time was the whole cost.** The 30-step CDD unroll the node was
  first written with cost 7–10 s of XLA compile in *every* outer `jit`
  (nested `jit` is inlined, so `update`, `grad` and the graph step each paid
  it again); a bare `vmap` for the synthesis matrix paid ~60 eager primitive
  compiles per size; and the base class calls the hooks eagerly, where one
  gathered solve cost 4.9 s and one frozen gradient 6.3 s on first use. Now:
  CDD is a `lax.while_loop`, assembly is NumPy, and the hook bodies are
  module-level jitted kernels taking the constant arrays as arguments, so
  every instance of one size shares the cache. Measured on the shared dev
  box: ~10 s for the first node of a new size (all first-use compiles),
  ~1 s for the next instance, `update` jit ≈ 1.2 s, `grad` jit ≈ 1.4 s.
- Each `update` runs up to 30 CDD iterations, each with one `k × k` solve
  and one sparse matvec, then one gathered solve. At the default budget the
  loop stops on the budget in a few iterations; for `k` above about
  `n_max / 2` it runs all 30 (see *Observing the selection*).
- The dense `n_max × n_max` operator is a constant of the step. MMS order
  is measured to 256 / $32^2$; construction, the adaptive solve and the
  frozen gradient are tested at $64^2$ and $16^3$ (4096 functions, `@slow`,
  10 – 15 s each on a loaded box, ~7 s of it the assembly).

## What porting it onto the 0.4.0 API found

These are findings about `AdaptiveNode`, not about the wavelet node; the
freeze is the moment they are cheap to act on.

1. **Overriding `compute_full_basis_gradient` means reimplementing private
   plumbing.** The hook must return a dict over `params_pytree()` leaves with
   non-trainable ones zeroed, which the subclass can only produce by calling
   `self._pytree` and `self._trainable`. A public helper taking a "solve on
   this mask" callable would let a subclass supply just the solve.
2. **Tuple parameters are trainable leaves.** `params_pytree()` turns a tuple
   of floats into an array leaf, so a structural location like `sensor` must
   be declared `trainable=False` *and* declared in `static_data_deps`, or it
   is a silently dead gradient. Nothing in the base class says so.
3. **The hooks are called eagerly by the diagnostics**, and an eager JAX
   function is compiled primitive by primitive. Any subclass with a
   non-trivial solve will find its cold start dominated by compile time until
   it jits its own hook bodies; the base class could jit the objective
   gradient it takes.
4. **`gather_solve`-style fixed-buffer solves are incompatible with the base
   default full-basis gradient** (item 7 above): the contract "an all-true
   mask is a valid solve" should be stated as a requirement of the *default*
   hook, with the override as the documented alternative.
5. **`dt` / `boundary_inputs` never reach the hooks.** Fine for this steady
   node; a time-dependent wavelet collocation node cannot be written without
   overriding `update` wholesale. Open question 2 of the base-class guide,
   now with a concrete customer.
6. **dtype policy.**  The strict xfail in
   `tests/property/test_adaptive_invariants.py` attributes the `run_scan`
   failure under `jax_enable_x64` to `AdaptiveNode` being the only node whose
   state dtype follows the flag.  Measured while porting: build the wavelet
   node with `dtype=jnp.float32` explicitly (so `c` *is* float32) and the
   graph still fails, on `carry['ball']` — and a plain `BallNode` or
   `SpringDamperNode` alone fails the same way on this tree, because
   `params_pytree()` injects float64 leaves under x64 and the node's arithmetic
   promotes its float32 state (`step()` hides it by storing the promoted
   state).  So of the three candidate policies, "`AdaptiveNode` stops following
   the flag" does not fix the graph on its own; "the graph coerces each update
   output back to its carry dtype" is the one that covers every node.  Pinned
   by `test_under_x64_the_mixed_graph_fails_on_the_downstream_carry_not_on_the_wavelet_one`.
   For the wavelet node itself float32 is a fine working precision: forward
   field 1.1e-6 and `dJ/dtheta` 1.8e-6 relative to float64 (jaxlib 0.11.0).
   Note the consequence for any float32-pinned subclass: it must cast every
   leaf it reads, which the wavelet node does in `_rhs`.

## Out of scope for this port

- The lid-driven-cavity fluid benchmark and the biharmonic (stream-function)
  operator: fluid numerics, not the node port, and the cavity run takes
  minutes rather than seconds.
- Variable coefficients $-\nabla\cdot(a\nabla u)$: the differentiable
  assembly exists in the source branch but no node surface exposes it.
- A matrix-free assembly for grids beyond the validated range.
