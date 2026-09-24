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

The loop has three exits.  The budget; the iteration bound; and the
rounding floor.  Each marking step adds the smallest Doerfler bulk of
the *remaining* residual, so once the source is resolved the steps
shrink and a large budget is approached slowly: on the 128-point
periodic basis at the node's default source in float64, ``K = 8`` is
reached in a handful of iterations, but ``K = 64`` and ``K = 96`` both
stop at ``|mask| = 54`` when :data:`MAX_OUTER` is hit (200 iterations
reach 64; the sensor-reading error at 54 is 3e-11).  For ``K`` above
about half the basis the bound is therefore the branch taken in
float64.  In float32 the residual reaches round-off sooner, and a step
that finds nothing above :func:`rounding_floor` to mark ends the loop
early instead of spending the remaining iterations marking noise.  The
mask is valid at every exit -- the budget is a ceiling, not a target --
and :func:`cdd_select_with_iterations` returns the count so a caller
can see which exit was taken.

Determinism.  A problem with a mirror symmetry -- the node's source is
centred on every axis but the first -- produces residuals equal up to
rounding, and a plain ``argsort`` let the last bits of the input decide
which member of a tied pair was marked.  Differently compiled
evaluations of the same parameters (eager, jitted, a graph step) round
differently, so they selected different sets.  The marking step now
reads no difference below the rounding floor: magnitudes within it of
the cutoff are tied and taken in ascending basis index order
(:func:`_doerfler_grow`).  The selection can still change -- it is a
discrete function of continuous data -- but only where a magnitude gap
crosses the floor's width or the Doerfler bulk falls on a cumulative
sum: isolated parameter values rather than every rounding perturbation.

JIT shape: the outer loop is a ``lax.while_loop`` bounded by
:data:`MAX_OUTER`, so the body is compiled once and the loop exits as
soon as the budget is reached or the bound is.  A ``while_loop`` cannot be
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

__all__ = [
    "cdd_select", "cdd_select_with_iterations", "rounding_floor",
    "MAX_OUTER", "NOISE_FACTOR", "THETA_D",
]

#: Bound on the outer iterations.  The 1-D and 2-D problems in the test
#: suite reach their default budget in at most 15; 3-D in about 17.  A
#: budget above about half the basis is not reached within the bound (see
#: the module docstring); the bound is deliberately not raised for that,
#: since every iteration costs a ``K x K`` solve on every update.
MAX_OUTER: int = 30

#: Doerfler bulk parameter: each marking step takes the smallest set of
#: inactive functions carrying ``THETA_D**2`` of the squared residual.
THETA_D: float = 0.5

#: Margin of :func:`rounding_floor` over the residual's rounding error.
#: Measured on six configurations (1-D 128 points and order 6 on 48, 2-D
#: 8^2 and 16^2; mass 1 and 5e-3; float32 and float64; jaxlib 0.11.0): the
#: difference between two evaluations of one inactive residual magnitude
#: (JAX against a NumPy float64 solve of the same stored operator), and
#: the spread of a mirror-symmetric pair while the set is symmetric, were
#: at most ``1.05 eps (max|b| + max|c|)``.  ``16`` is the smallest power of
#: two with a tenfold margin over that.  The floor also ends refinement
#: once nothing above it is left; that cost nothing measurable: the float32
#: sensor reading against the full-basis one moved by at most 1.6e-7 at 16
#: (3e-6 even at 256), and float64 selections were unchanged.
NOISE_FACTOR: float = 16.0


@stability(StabilityLevel.EXPERIMENTAL)
def rounding_floor(b: jax.Array, c: jax.Array, factor: float = NOISE_FACTOR) -> jax.Array:
    """The residual magnitude below which a marking decision would read round-off.

    ``factor * eps * (max|b| + max|c|)``, with ``eps`` the machine epsilon
    of ``b``'s dtype.  In coordinates where the operator's diagonal is
    ``O(1)`` -- the symmetrically preconditioned ones the node works in --
    the rounding error of one residual ``b - A c`` is a small multiple of
    ``eps * (|b| + |A| |c|)``, so this scalar bounds it with a margin of
    ``factor``.  The ``max|c|`` term is what makes the floor grow with the
    conditioning: an ill-conditioned solve returns large coefficients and
    their products cancel in the residual.  It is both the level below
    which a function is never marked and the width within which two
    residual magnitudes count as tied (:func:`cdd_select_with_iterations`).
    """
    eps = jnp.finfo(b.dtype).eps
    return factor * eps * (jnp.max(jnp.abs(b)) + jnp.max(jnp.abs(c)))


def _doerfler_grow(mask: jax.Array, resid: jax.Array, theta_d: float,
                   cap: int, tol: jax.Array) -> jax.Array:
    """``mask`` enlarged by the smallest Doerfler bulk of inactive functions.

    Static-shape.  ``tol`` is :func:`rounding_floor` at the current
    iterate; the step reads no difference smaller than it:

    1. inactive residual magnitudes at or below ``tol`` are set to zero --
       they are indistinguishable from round-off and are never marked;
    2. the count ``n_take`` is the Doerfler count (the shortest
       descending-magnitude prefix carrying ``theta_d**2`` of the squared
       residual) capped at the room left under ``cap``;
    3. every function whose magnitude exceeds the cutoff ``r_cut`` (the
       ``n_take``-th largest) by more than ``tol`` is taken, and the rest
       of ``n_take`` is filled from the band ``|r - r_cut| <= tol`` in
       **ascending basis index order**.

    Step 3 is the tie-break.  A problem with a mirror symmetry produces
    residuals that are equal up to rounding, so a plain ``argsort`` let
    the last bits of the input decide which member of a tied pair
    survived the cap -- and the eager, jitted and compiled-graph
    evaluations, or a Python float and its float32-array spelling of the
    same parameter, round differently.  Here a tie within ``tol`` is
    broken by a fixed order, so the selected set can change only where a
    magnitude difference crosses ``tol`` or the cumulative bulk crosses
    ``theta_d**2`` of the total: isolated parameter values, not every
    step.
    """
    r = jnp.where(mask, 0.0, jnp.abs(resid))        # only inactive functions can be marked
    r = jnp.where(r > tol, r, 0.0)                  # below the floor is round-off
    rs = -jnp.sort(-r)                              # descending magnitudes (values only)
    csum = jnp.cumsum(rs ** 2)
    n_bulk = jnp.sum(csum < (theta_d ** 2) * csum[-1]) + 1
    n_room = jnp.maximum(cap - jnp.sum(mask), 0)
    n_take = jnp.minimum(jnp.minimum(n_bulk, jnp.sum(r > 0)), n_room)
    r_cut = rs[jnp.maximum(n_take - 1, 0)]
    above = r > r_cut + tol
    band = (r > 0) & (r >= r_cut - tol) & ~above
    band_rank = jnp.cumsum(band.astype(jnp.int32))  # ascending index order
    add = above | (band & (band_rank <= n_take - jnp.sum(above)))
    return mask | (add & (n_take > 0))


@stability(StabilityLevel.EXPERIMENTAL)
def cdd_select_with_iterations(
    apply_operator: Callable[[jax.Array], jax.Array],
    solve_masked: Callable[[jax.Array, jax.Array], jax.Array],
    b: jax.Array,
    coarse_mask: jax.Array,
    K: int,
    *,
    theta_d: float = THETA_D,
    max_outer: int = MAX_OUTER,
    noise_factor: float = NOISE_FACTOR,
) -> tuple[jax.Array, jax.Array, jax.Array]:
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
        With more, the loop never runs and the seed comes back unchanged
        for the caller's frozen solve to refuse.
    K : int
        Active-set budget.  Growth stops once ``|mask| >= K``; the cap in
        the marking step guarantees ``|mask| <= K`` throughout.
    theta_d, max_outer
        Doerfler bulk and the iteration bound.
    noise_factor : float
        Margin of the rounding floor, :func:`rounding_floor`.  The floor
        assumes an operator with an ``O(1)`` diagonal, which the
        symmetric diagonal preconditioning the node applies provides.

    Returns
    -------
    (mask, c, n_outer)
        The boolean active set of shape ``(N,)``, the solution on it in
        the caller's coordinates, and the number of outer iterations run
        (``int32`` scalar).  It equals ``max_outer`` when the bound ended
        the loop; below that with ``|mask| < K``, a step found nothing
        above the rounding floor to mark and the loop stopped there.  The
        mask is a plain array: the node's base class wraps it in
        ``stop_gradient``.
    """
    mask0 = jnp.asarray(coarse_mask, dtype=bool)
    c0 = solve_masked(mask0, b)

    def keep_going(carry):
        i, mask, _c, grew = carry
        return (i < max_outer) & (jnp.sum(mask) < K) & grew

    def grow(carry):
        i, mask, c, _grew = carry
        resid = b - apply_operator(c)
        new = _doerfler_grow(mask, resid, theta_d, K, rounding_floor(b, c, noise_factor))
        grew = jnp.any(new != mask)
        return i + 1, new, solve_masked(new, b), grew

    n_outer, mask, c, _ = jax.lax.while_loop(
        keep_going, grow, (jnp.int32(0), mask0, c0, jnp.bool_(True)),
    )
    return mask, c, n_outer


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
    noise_factor: float = NOISE_FACTOR,
) -> tuple[jax.Array, jax.Array]:
    """:func:`cdd_select_with_iterations` without the iteration count: ``(mask, c)``."""
    mask, c, _ = cdd_select_with_iterations(
        apply_operator, solve_masked, b, coarse_mask, K,
        theta_d=theta_d, max_outer=max_outer, noise_factor=noise_factor,
    )
    return mask, c
