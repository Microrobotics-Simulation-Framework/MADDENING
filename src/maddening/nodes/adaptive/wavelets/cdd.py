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

# Iteration ceiling for the outer CDD loop.  The derisking spike measured
# 1D/2D p99=15, 3D mean ~17 on CONSTANT-coefficient problems and set 30.  That
# is unsafe at high coefficient contrast: the active set needs ~3x more Doerfler
# rounds to grow (spikes/wavelet_apps/FINDINGS_D5 — χ=1e5 needs ~90), and at 30
# the loop returned a worse-than-zero solution silently.  With M1's lax.while_loop
# early exit, a larger ceiling is free — healthy low-contrast solves still stop in
# ~15 iterations; only hard cases use the headroom, and trace time is O(1) in it.
MAX_OUTER: int = 200
THETA_D: float = 0.5         # Doerfler bulk; FINDINGS Inv 1B confirms in 3D


def _doerfler_grow(mask: jax.Array, resid: jax.Array, theta_D: float,
                   cap: int) -> jax.Array:
    """Return ``mask`` enlarged by the smallest Doerfler bulk of inactive DOFs.

    Marks the smallest set of currently-inactive indices whose summed squared
    residual reaches ``theta_D**2`` of the total -- via sort + cumsum + the
    first-crossing index (static-shape boolean, no dynamic slice).  The number
    added is capped so ``|mask| <= cap`` (the gather-solve buffer size), so the
    active set never exceeds the budget.
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
    # cap the number added so |mask| never exceeds `cap`
    n_room = cap - jnp.sum(mask)
    rank = jnp.cumsum(take_sorted.astype(jnp.int32))   # 1..count over taken
    take_sorted = take_sorted & (rank <= n_room)
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
    rtol: float = 1e-6,
    indicator: Callable[[jax.Array], jax.Array] | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Run CDD to an active-set budget ``K``; return ``(mask, c, converged)``.

    Works in whatever coordinates the caller supplies (the node passes the
    symmetrically-scaled operator and RHS).

    The outer SOLVE→ESTIMATE→MARK→REFINE loop is a :func:`jax.lax.while_loop`
    with a **real early exit** (round-6 rejected ``lax.fori_loop`` for failing to
    short-circuit; ``while_loop`` short-circuits natively, and the loop is safe
    because the caller ``stop_gradient``s the returned mask — no gradient flows
    through it).  It terminates on the first of: relative residual below
    ``rtol``, active set reaching the budget ``K``, or ``max_outer`` iterations.
    Cost is therefore the *actual* iteration count, not a fixed ``max_outer`` —
    and trace time is O(1) in ``max_outer`` (the old Python unroll paid both in
    full; see ``spikes/wavelet_apps`` and the module history).

    Parameters
    ----------
    apply_operator : ``v -> Â v`` (full operator matvec; for the ESTIMATE step).
    solve_masked : ``(mask, rhs) -> c`` -- the frozen inner solve on ``mask``
        (the node wraps ``ift_linear_solve`` + ``make_masked_operator``).
    b : right-hand side (scaled).
    coarse_mask : boolean ``(N,)`` of always-included coarse DOFs.
    K : active-set budget; growth stops once ``|mask| >= K``.
    rtol : relative-residual tolerance for the early exit.
    indicator : optional ``r_scaled -> per-DOF marking score`` (non-negative).
        The error indicator Doerfler marks on.  Defaults to ``|r|`` -- the
        correct choice under diagonal scaling, where the basis is (near-)Riesz
        stable.  A preconditioner that changes coordinates supplies its own
        (e.g. ``|M⁻¹ r|`` for an operator preconditioner); see
        :class:`maddening.nodes.adaptive.wavelets.preconditioners`.

    Returns
    -------
    mask, c : the frozen active set and its coefficients.
    converged : bool scalar -- ``True`` iff the loop stopped on the residual
        criterion (``rel < rtol``) **or** on the active-set budget ``K``.
        Reaching ``K`` is a *controlled sparse approximation*, the normal
        adaptive outcome (the active set is deliberately truncated), so it is
        healthy.  ``converged`` is ``False`` only when ``max_outer`` was
        exhausted *before* either -- i.e. the Doerfler marking was still
        growing the active set when it ran out of iterations.  That is the
        genuine failure the flag exists to surface: the returned ``c`` can be
        far from the best ``K``-term solution, at high contrast worse than
        zero (see ``spikes/wavelet_apps/FINDINGS_D5`` and the M1/M3 history).

        **Limitation (important).** The flag detects iteration starvation, not
        *inadequate budget at high contrast*.  If ``K`` is too small for a
        high-contrast coefficient field, the budget fills (flag ``True``) and
        the scaled residual can even be small, yet the solution error is large:
        at κ ~ contrast the restricted solve is ill-conditioned, so a small
        residual does not bound the error (measured: χ=1e5 at K=N/16 gives
        residual ~2e-3 but solution error ~1.5, worse than zero, with the flag
        reading ``True``).  No cheap local signal separates that from a healthy
        truncation.  The remedies are adequate budget (``K`` scaling with
        contrast) or a contrast-robust preconditioner (roadmap R1); this is the
        reason app-1 near-term scope caps at χ ≤ 10².
    """
    b_norm = jnp.linalg.norm(b) + 1e-30

    def _rel(c):
        return jnp.linalg.norm(b - apply_operator(c)) / b_norm

    mask0 = coarse_mask
    c0 = solve_masked(mask0, b)
    resid0 = b - apply_operator(c0)
    state0 = (jnp.int32(0), mask0, c0, resid0, _rel(c0))

    def cond(state):
        it, mask, _c, _resid, rel = state
        budget_left = jnp.sum(mask) < K
        return (it < max_outer) & (rel >= rtol) & budget_left

    def body(state):
        it, mask, _c, resid, _rel_prev = state
        score = resid if indicator is None else indicator(resid)
        grown = _doerfler_grow(mask, score, theta_D, K)
        new_c = solve_masked(grown, b)
        new_resid = b - apply_operator(new_c)
        new_rel = jnp.linalg.norm(new_resid) / b_norm
        return (it + 1, grown, new_c, new_resid, new_rel)

    _it, mask, c, _resid, rel_final = jax.lax.while_loop(cond, body, state0)
    # Healthy stop = residual tolerance reached OR budget K filled (a controlled
    # sparse approximation). Only max_outer-exhaustion-before-either is a failure.
    converged = (rel_final < rtol) | (jnp.sum(mask) >= K)
    return mask, c, converged
