"""Cohen-Dahmen-DeVore residual-driven active-set selection.

The selection rule of
:class:`~maddening.nodes.adaptive.wavelet.WaveletAdaptiveNode`: the
SOLVE -> ESTIMATE -> MARK -> REFINE loop of adaptive wavelet methods
[CohenDahmenDeVore2001]_, grown from the coarse level by Doerfler bulk
marking [Doerfler1996]_ until an active-set budget ``K`` is reached.

Two properties the node's contract with
:class:`~maddening.nodes.adaptive.base.AdaptiveNode` rests on:

* **The coarse level is always active.**  The set starts from
  ``coarse_mask`` and only ever grows, so it is never empty -- at a cold
  start or anywhere else -- and the coarse functions that carry the bulk
  of any sensor reading are never dropped.  That inclusion, not the
  locality of the basis, is what keeps the adaptive sensor reading from
  changing sign relative to the full solve.
* **The set never exceeds** ``K``.  Each marking step is capped at the
  room left under the budget, which is what lets the node solve on a
  gathered ``K x K`` block with no risk of silently truncating the set.

JIT shape: the outer loop is a ``lax.while_loop`` bounded by
:data:`MAX_OUTER`, so the body is compiled once and the loop exits as
soon as the budget is reached.  A ``while_loop`` cannot be
reverse-differentiated, which is fine here *because* nothing may be
differentiated through a selection: the caller passes a right-hand
side under ``stop_gradient`` (the node does), and the base class wraps
the returned mask in ``stop_gradient`` again.  The mark step is
vectorised (``argsort`` + ``cumsum`` + first-crossing), so the whole
selection is a static graph over fixed-shape arrays and the mask it
returns is a boolean ``(N,)`` array.

.. [CohenDahmenDeVore2001] Cohen, A., Dahmen, W., DeVore, R. (2001).
   Adaptive wavelet methods for elliptic operator equations: convergence
   rates.  *Mathematics of Computation* 70(233), 27-75.
.. [Doerfler1996] Doerfler, W. (1996).  A convergent adaptive algorithm
   for Poisson's equation.  *SIAM Journal on Numerical Analysis* 33(3),
   1106-1124.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability

__all__ = ["cdd_select", "MAX_OUTER", "THETA_D"]

#: Bound on the outer iterations.  The 1-D and 2-D problems in the test
#: suite reach their budget in at most 15; 3-D in about 17.
MAX_OUTER: int = 30

#: Doerfler bulk parameter: each marking step takes the smallest set of
#: inactive functions carrying ``THETA_D**2`` of the squared residual.
THETA_D: float = 0.5


def _doerfler_grow(mask: jax.Array, resid: jax.Array, theta_d: float,
                   cap: int) -> jax.Array:
    """``mask`` enlarged by the smallest Doerfler bulk of inactive functions.

    Static-shape: sort the inactive residuals, take the prefix whose
    cumulative squared mass first reaches ``theta_d**2`` of the total,
    truncate that prefix to the room left under ``cap``, and scatter it
    back.  Functions with an exactly zero residual are never marked.
    """
    n = resid.shape[0]
    r = jnp.where(mask, 0.0, jnp.abs(resid))       # only inactive functions can be marked
    order = jnp.argsort(-r)                         # descending |r|
    csum = jnp.cumsum(r[order] ** 2)
    total = csum[-1] + 1e-30
    below = csum < (theta_d ** 2) * total
    first_cross = jnp.argmin(below.astype(jnp.int32))   # first False
    take_sorted = below.at[first_cross].set(True)
    n_room = cap - jnp.sum(mask)
    rank = jnp.cumsum(take_sorted.astype(jnp.int32))
    take_sorted = take_sorted & (rank <= n_room)
    add = jnp.zeros(n, dtype=bool).at[order].set(take_sorted)
    add = add & (r > 0)
    return mask | add


@stability(StabilityLevel.EXPERIMENTAL)
def cdd_select(
    apply_operator: Callable[[jax.Array], jax.Array],
    solve_masked: Callable[[jax.Array, jax.Array], jax.Array],
    b: jax.Array,
    coarse_mask: jax.Array,
    K: int,
    *,
    theta_d: float = THETA_D,
    max_outer: int = MAX_OUTER,
) -> tuple[jax.Array, jax.Array]:
    """Grow an active set from the coarse level to the budget ``K``.

    Works in whatever coordinates the caller supplies; the node passes
    the symmetrically preconditioned operator and right-hand side.

    Parameters
    ----------
    apply_operator : callable
        ``v -> A v`` on the full basis, for the residual estimate.
    solve_masked : callable
        ``(mask, b) -> c``, the frozen solve on ``mask``.  It is called
        ``max_outer + 1`` times, so it should be the cheap gathered
        solve rather than a full-size iterative one.
    b : jax.Array
        Right-hand side, shape ``(N,)``.  Pass it under
        ``jax.lax.stop_gradient``: the loop is not reverse-differentiable
        and a selection must not carry a tangent anyway.
    coarse_mask : jax.Array
        Boolean ``(N,)`` seed; must be non-empty and have at most ``K``
        entries set (the caller validates this once, at construction).
    K : int
        Active-set budget.  Growth stops once ``|mask| >= K``; the cap in
        the marking step guarantees ``|mask| <= K`` throughout.
    theta_d, max_outer
        Doerfler bulk and the iteration bound.

    Returns
    -------
    (mask, c)
        The boolean active set of shape ``(N,)`` and the solution on it
        in the caller's coordinates.  The mask is a plain array: the
        node's base class wraps it in ``stop_gradient``.
    """
    mask0 = jnp.asarray(coarse_mask, dtype=bool)
    c0 = solve_masked(mask0, b)

    def keep_going(carry):
        i, mask, _c = carry
        return (i < max_outer) & (jnp.sum(mask) < K)

    def grow(carry):
        i, mask, c = carry
        resid = b - apply_operator(c)
        mask = _doerfler_grow(mask, resid, theta_d, K)
        return i + 1, mask, solve_masked(mask, b)

    _, mask, c = jax.lax.while_loop(keep_going, grow, (jnp.int32(0), mask0, c0))
    return mask, c
