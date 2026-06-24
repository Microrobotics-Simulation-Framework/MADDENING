"""Cohen-Dahmen-DeVore (CDD) residual-driven active-set selection.

Port of the spike's CDD loop (``hybrid_jacobi.py::cdd_idx``,
``nonlinear_cdd.py``), rewritten to be JIT-compilable with **static shapes**.

JIT pattern (round-6 Investigation A, carried through per the plan): the outer
SOLVE -> ESTIMATE -> MARK(Doerfler) -> REFINE loop is a **Python ``for``-loop
unrolled at trace time** (``MAX_OUTER`` iterations), NOT ``lax.fori_loop`` /
``lax.while_loop`` -- the unroll produces a static graph that short-circuits
correctly (via ``jnp.where`` on a ``converged`` flag) and is faster at
production N; ``lax.fori_loop`` compiles but does not short-circuit correctly.
The MARK step uses the vectorised ``argsort`` + ``cumsum`` + ``searchsorted``
pattern (round-6 Investigation 3C) -- no per-index Python loop inside JIT.

The returned mask is a fixed-length ``(N,)`` boolean (static shape).  The active
set always contains the coarse level (``coarse_mask``) -- the wrong-sign-safety
mechanism (FINDINGS §3: safety comes from coarse-inclusion, not pure locality).
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax
import jax.numpy as jnp

__all__ = ["cdd_select", "MAX_OUTER", "THETA_D"]

MAX_OUTER: int = 30          # FINDINGS Inv 1E/2E: 1D/2D p99=15, 3D mean ~17
THETA_D: float = 0.5         # Doerfler bulk; FINDINGS Inv 1B confirms in 3D


def _doerfler_grow(mask: jax.Array, resid: jax.Array, theta_D: float) -> jax.Array:
    """Return ``mask`` enlarged by the smallest Doerfler bulk of inactive DOFs.

    Marks the smallest set of currently-inactive indices whose summed squared
    residual reaches ``theta_D**2`` of the total -- via sort + cumsum + the
    first-crossing index (static-shape boolean, no dynamic slice).
    """
    N = resid.shape[0]
    r = jnp.where(mask, 0.0, jnp.abs(resid))     # only inactive can be marked
    order = jnp.argsort(-r)                       # indices by descending |r|
    r2 = r[order] ** 2
    csum = jnp.cumsum(r2)
    total = csum[-1] + 1e-30
    below = csum < (theta_D ** 2) * total         # positions strictly below bulk
    # include the first position that crosses the threshold
    first_cross = jnp.argmin(below.astype(jnp.int32))  # first False
    take_sorted = below.at[first_cross].set(True)
    add = jnp.zeros(N, dtype=bool).at[order].set(take_sorted)
    # never mark where the residual is exactly zero (e.g. zero-source DOFs)
    add = add & (r > 0)
    return mask | add


def cdd_select(
    apply_operator: Callable[[jax.Array], jax.Array],
    solve_masked: Callable[[jax.Array, jax.Array], jax.Array],
    b: jax.Array,
    coarse_mask: jax.Array,
    K: int,
    *,
    theta_D: float = THETA_D,
    max_outer: int = MAX_OUTER,
) -> Tuple[jax.Array, jax.Array]:
    """Run CDD to an active-set budget ``K``; return ``(mask, c)``.

    Works in whatever coordinates the caller supplies (the node passes the
    symmetrically-scaled operator and RHS).

    Parameters
    ----------
    apply_operator : ``v -> Â v`` (full operator matvec; for the ESTIMATE step).
    solve_masked : ``(mask, rhs) -> c`` -- the frozen inner solve on ``mask``
        (the node wraps ``ift_linear_solve`` + ``make_masked_operator``).
    b : right-hand side (scaled).
    coarse_mask : boolean ``(N,)`` of always-included coarse DOFs.
    K : active-set budget; growth stops once ``|mask| >= K``.
    """
    mask = coarse_mask
    c = solve_masked(mask, b)
    for _ in range(max_outer):
        converged = jnp.sum(mask) >= K
        resid = b - apply_operator(c)
        grown = _doerfler_grow(mask, resid, theta_D)
        new_mask = jnp.where(converged, mask, grown)
        new_c = solve_masked(new_mask, b)
        # short-circuit: once converged, freeze mask and solution
        mask = jnp.where(converged, mask, new_mask)
        c = jnp.where(converged, c, new_c)
    return mask, c
