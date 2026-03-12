"""
Physics-informed loss functions for surrogate training.

These composable loss functions encode physical constraints -- energy
conservation, equation-of-motion residuals, PDE residuals -- that can
be combined with the standard data-fitting loss in
``SurrogateTrainer``.

Usage
-----
Pass as ``physics_loss_fn`` and ``physics_loss_weight`` to
``SurrogateTrainer``::

    from maddening.surrogates.physics_losses import residual_loss

    trainer = SurrogateTrainer(
        arch, dataset,
        physics_loss_fn=residual_loss(ball_node.update),
        physics_loss_weight=0.1,
    )
"""

from typing import Callable

import jax
import jax.numpy as jnp


# ------------------------------------------------------------------
# Physics loss function signature
# ------------------------------------------------------------------
# All physics loss functions follow this signature:
#
#   physics_loss_fn(weights, state, boundary_inputs, pred, dt) -> scalar
#
# where `weights` is the surrogate's (arrays, static) tuple,
# `state` and `boundary_inputs` are the inputs to the forward pass,
# `pred` is the surrogate's predicted next state, and `dt` is the
# timestep.
# ------------------------------------------------------------------


def residual_loss(update_fn: Callable) -> Callable:
    """Create a loss that penalizes deviation from a reference update function.

    The surrogate's prediction is compared against the output of
    ``update_fn(state, boundary_inputs, dt)`` -- typically the original
    physics node's ``update`` method.  This is the simplest form of
    physics-informed loss: the surrogate is trained to match the physics
    model exactly.

    Parameters
    ----------
    update_fn : callable
        A physics node's ``update(state, boundary_inputs, dt) -> state``
        method.

    Returns
    -------
    callable
        Physics loss function with the standard signature.
    """
    def loss_fn(weights, state, boundary_inputs, pred, dt):
        target = update_fn(state, boundary_inputs, dt)
        total = jnp.float32(0.0)
        count = 0
        for k in pred:
            if k in target:
                diff = pred[k] - target[k]
                total = total + jnp.sum(diff ** 2)
                count += diff.size
        return total / max(count, 1)
    return loss_fn


def energy_conservation_loss(
    kinetic_fn: Callable,
    potential_fn: Callable,
) -> Callable:
    """Penalize changes in total energy across a timestep.

    Suitable for conservative systems (no dissipation).  For dissipative
    systems, use with a low weight or combine with a dissipation term.

    Parameters
    ----------
    kinetic_fn : callable
        ``(state) -> scalar`` computing kinetic energy from state.
    potential_fn : callable
        ``(state) -> scalar`` computing potential energy from state.

    Returns
    -------
    callable
        Physics loss function.

    Example
    -------
    ::

        # For a ball under gravity
        KE = lambda s: 0.5 * s["velocity"] ** 2
        PE = lambda s: 9.81 * s["position"]
        loss = energy_conservation_loss(KE, PE)
    """
    def loss_fn(weights, state, boundary_inputs, pred, dt):
        e_before = kinetic_fn(state) + potential_fn(state)
        e_after = kinetic_fn(pred) + potential_fn(pred)
        return (e_after - e_before) ** 2
    return loss_fn


def momentum_conservation_loss(
    momentum_fn: Callable,
    force_fn: Callable = None,
) -> Callable:
    """Penalize violations of momentum conservation (or Newton's second law).

    If ``force_fn`` is ``None``, enforces strict momentum conservation
    (dp/dt = 0).  If provided, enforces dp = F * dt.

    Parameters
    ----------
    momentum_fn : callable
        ``(state) -> scalar_or_array`` computing momentum.
    force_fn : callable, optional
        ``(state, boundary_inputs) -> scalar_or_array`` computing net force.

    Returns
    -------
    callable
        Physics loss function.
    """
    def loss_fn(weights, state, boundary_inputs, pred, dt):
        p_before = momentum_fn(state)
        p_after = momentum_fn(pred)
        if force_fn is not None:
            expected_dp = force_fn(state, boundary_inputs) * dt
            residual = (p_after - p_before) - expected_dp
        else:
            residual = p_after - p_before
        return jnp.sum(residual ** 2)
    return loss_fn


def smoothness_loss() -> Callable:
    """Penalize large state changes relative to dt (Lipschitz regularization).

    Encourages the surrogate to produce smooth trajectories by penalizing
    the magnitude of d(state)/dt.

    Returns
    -------
    callable
        Physics loss function.
    """
    def loss_fn(weights, state, boundary_inputs, pred, dt):
        total = jnp.float32(0.0)
        count = 0
        for k in pred:
            if k in state:
                rate = (pred[k] - state[k]) / jnp.maximum(dt, 1e-10)
                total = total + jnp.sum(rate ** 2)
                count += rate.size
        return total / max(count, 1)
    return loss_fn


def composite_loss(*loss_fns_and_weights: tuple[Callable, float]) -> Callable:
    """Combine multiple physics losses with individual weights.

    Parameters
    ----------
    *loss_fns_and_weights : tuple[callable, float]
        Pairs of ``(loss_fn, weight)``.

    Returns
    -------
    callable
        Combined physics loss function.

    Example
    -------
    ::

        loss = composite_loss(
            (energy_conservation_loss(KE, PE), 0.5),
            (smoothness_loss(), 0.1),
        )
    """
    def loss_fn(weights, state, boundary_inputs, pred, dt):
        total = jnp.float32(0.0)
        for fn, w in loss_fns_and_weights:
            total = total + w * fn(weights, state, boundary_inputs, pred, dt)
        return total
    return loss_fn
