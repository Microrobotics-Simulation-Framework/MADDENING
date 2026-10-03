"""
Adaptive timestepping via Richardson extrapolation.

Uses step-doubling: compares a full step at ``dt`` with two half-steps
at ``dt/2`` to estimate the local truncation error without modifying
individual node update functions.

A PI step-size controller adjusts ``dt`` each step to keep the error
within user-specified absolute and relative tolerances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class AdaptiveConfig:
    """Configuration for adaptive timestepping.

    Parameters
    ----------
    dt_initial : float
        Starting timestep.
    atol : float
        Absolute error tolerance.
    rtol : float
        Relative error tolerance.
    dt_min : float
        Minimum allowed timestep.
    dt_max : float
        Maximum allowed timestep.
    safety : float
        Safety factor for step-size controller (< 1).
    max_factor : float
        Maximum factor by which dt can grow per step.
    min_factor : float
        Minimum factor by which dt can shrink per step.
    order : int
        Order of the base method (Euler = 1).  Used by the PI controller.
    """
    dt_initial: float = 0.01
    atol: float = 1e-6
    rtol: float = 1e-3
    dt_min: float = 1e-8
    dt_max: float = 0.1
    safety: float = 0.9
    max_factor: float = 5.0
    min_factor: float = 0.2
    order: int = 1


def step_decision(error_norm, dt, dt_min, dt_max, *, safety, order, min_factor,
                  max_factor, xp=np):
    """``(accepted, forced, dt_next, factor)``: the one acceptance rule of the adaptive steppers.

    ``run_adaptive`` (host loop, ``xp=numpy``) and ``run_adaptive_scan``
    (traced, ``xp=jax.numpy``) and :func:`build_adaptive_step` all call
    this, so they accept the same attempts and choose the same next
    timestep.

    * An attempt is **accepted** when its step-doubling error is within
      tolerance (``error_norm <= 1``) or when it was already made at
      ``dt_min`` (``dt <= dt_min``), where the controller cannot shrink
      further; ``forced`` says the second, which ``run_adaptive`` warns
      about.
    * The next timestep is the PI controller's ``factor``,
      ``safety * error_norm**(-1/(order+1))`` clipped to
      ``[min_factor, max_factor]`` (``max_factor`` at zero error), times
      ``dt``, clipped to ``[dt_min, dt_max]`` -- after an accepted attempt
      and a rejected one alike.

    Until 0.4.0's round-4 audit the two steppers had their own copies, and
    ``run_adaptive``'s accepted a *rejected* attempt larger than ``dt_min``
    whenever shrinking it would reach ``dt_min``: on an explicit-Euler decay
    held at ``dt_min=0.005`` it took a failed first step of 0.008 and
    overshot ``t_end`` to 0.103, where ``run_adaptive_scan`` retried at
    ``dt_min`` and ended at 0.1 (final states 16% apart).
    """
    within = error_norm <= 1.0
    at_floor = dt <= dt_min
    accepted = xp.logical_or(within, at_floor)
    forced = xp.logical_and(at_floor, xp.logical_not(within))
    safe = xp.maximum(error_norm, 1e-10)
    factor = xp.clip(safety * xp.power(1.0 / safe, 1.0 / (order + 1)), min_factor, max_factor)
    dt_next = xp.clip(dt * factor, dt_min, dt_max)
    return accepted, forced, dt_next, factor


def _is_inexact_leaf(leaf) -> bool:
    """A floating (or complex) array: something a truncation error lives in."""
    return bool(jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.inexact))


def _tree_error_norm(state_fine, state_coarse, atol, rtol):
    """Compute the mixed absolute/relative error norm.

    Uses the formula:
        err_i = |fine_i - coarse_i| / (atol + rtol * max(|fine_i|, |coarse_i|))
    Returns the RMS norm over every element of every floating leaf.

    **Floating leaves only.**  The norm measures the local truncation
    error step doubling exposes, which lives in the fields an integrator
    advances.  An integer, unsigned, boolean or PRNG-key leaf (a counter,
    a tag, a flag) carries no truncation error, and reading one gave a
    wrong step sequence: a ``bool`` cannot be subtracted at all
    (``TypeError``); a ``uint32``'s ``fine - coarse`` wraps modulo
    ``2**32``, so an unread tag read as an error of order one and the
    controller rejected nearly every step (199,981 rejections in 199,991
    attempts on a two-node pair); and an ``int32`` counter, which is
    ``k + 2`` after the two half steps and ``k + 1`` after the full step
    *by construction*, added a term and an element to the RMS and moved
    the accepted steps (77 became 65 on the same pair).  Such leaves are
    skipped -- they add neither a term nor an element -- so a graph steps
    exactly as it would without them, and an all-floating state is
    measured exactly as before.  A floating field that is the same in
    both estimates still counts as an element: that is the RMS convention
    adaptive ODE solvers use.
    """
    sum_sq = jnp.array(0.0)
    count = jnp.array(0, dtype=jnp.int32)

    def _accumulate(fine, coarse):
        nonlocal sum_sq, count
        if not _is_inexact_leaf(fine):
            return
        diff = jnp.abs(fine - coarse)
        scale = atol + rtol * jnp.maximum(jnp.abs(fine), jnp.abs(coarse))
        # A zero scale (``atol=0`` on an entry at zero in both estimates)
        # contributes nothing; every other entry is divided by its own
        # scale.  The guard used to be ``max(scale, 1e-300)``, an absolute
        # floor in the state's units: in float64 an entry below about
        # ``1e-297`` (``1e-300 / rtol``) was divided by ``1e-300`` instead of
        # its scale, its error read up to ``rtol * |x| / 1e-300`` times too
        # small, and steps that should have been rejected were accepted.
        live = scale > 0
        scaled = jnp.where(live, diff / jnp.where(live, scale, 1.0), 0.0)
        sum_sq += jnp.sum(scaled ** 2)
        count += scaled.size

    jax.tree.map(_accumulate, state_fine, state_coarse)
    mean_sq = sum_sq / jnp.maximum(count, 1)
    # The square root's derivative is infinite at zero, so two estimates
    # that agree exactly (a memoryless or steady state, where the full step
    # and the two half steps land on one value) gave ``inf`` there, and
    # ``run_adaptive_scan``'s backward pass multiplied it by the zero the
    # controller's ``max(error_norm, 1e-10)`` sends back: NaN in every
    # gradient through the scan.  The double ``where`` keeps the value and
    # gives the zero-error case the zero derivative of ``max``'s flat side.
    nonzero = mean_sq > 0
    return jnp.where(nonzero, jnp.sqrt(jnp.where(nonzero, mean_sq, 1.0)), 0.0)


def build_adaptive_step(
    raw_step_fn: Callable,
    config: AdaptiveConfig,
    node_names: list[str],
) -> Callable:
    """Build an adaptive step function using Richardson extrapolation.

    The returned function has signature::

        (state, dt, external_inputs) -> (new_state, dt_next, error, accepted)

    The caller decides whether to accept or retry.

    Parameters
    ----------
    raw_step_fn : callable
        The unjitted graph step function with signature
        ``(full_state, external_inputs) -> full_state``.
        This function uses the *node's own timestep* internally.
        For adaptive stepping we need a dt-parameterised version,
        so we use a wrapper that scales boundary computations.
    config : AdaptiveConfig
        Adaptive stepping parameters.
    node_names : list of str
        Node names (used for error norm computation).
    """
    atol = config.atol
    rtol = config.rtol
    safety = config.safety
    max_factor = config.max_factor
    min_factor = config.min_factor
    order = config.order

    def adaptive_step(state, dt, external_inputs, dt_step_fn):
        """Take one adaptive step.

        Parameters
        ----------
        state : dict
            Full state dict.
        dt : jax array (scalar)
            Current timestep.
        external_inputs : dict
            External inputs.
        dt_step_fn : callable
            ``(state, ext, dt) -> new_state`` parameterised by dt.

        Returns
        -------
        new_state, dt_next, error_norm, accepted
        """
        # Full step at dt
        state_full = dt_step_fn(state, external_inputs, dt)

        # Two half-steps at dt/2
        half_dt = dt / 2.0
        state_half = dt_step_fn(state, external_inputs, half_dt)
        state_half = dt_step_fn(state_half, external_inputs, half_dt)

        # Error estimate: difference between the two approaches
        # Only compare user state (not _meta)
        user_full = {k: v for k, v in state_full.items() if k != "_meta"}
        user_half = {k: v for k, v in state_half.items() if k != "_meta"}
        error_norm = _tree_error_norm(user_half, user_full, atol, rtol)

        # The one acceptance rule and PI controller of the adaptive
        # steppers (``step_decision``): accepted within tolerance or at
        # ``dt_min``.
        accepted, _forced, dt_next, _factor = step_decision(
            error_norm, dt, config.dt_min, config.dt_max, safety=safety,
            order=order, min_factor=min_factor, max_factor=max_factor, xp=jnp)

        # Use the more accurate result (half-step) when accepted
        new_state = state_half

        return new_state, dt_next, error_norm, accepted

    return adaptive_step
