"""
Implicit time integration via fixed-count Newton iteration.

Provides ``implicit_euler_step`` which solves the backward Euler
equation using Newton's method with a fixed iteration count
(via ``jax.lax.fori_loop`` for JIT/grad/scan compatibility).

For nodes that implement ``implicit_residual()``, this provides
unconditional stability for stiff ODEs.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp


def implicit_euler_step(
    residual_fn: Callable,
    state_old: dict,
    boundary_inputs: dict,
    dt: float,
    n_newton: int = 5,
    initial_guess: dict | None = None,
) -> dict:
    """Solve the backward Euler equation using fixed-count Newton.

    Solves::

        x_new = x_old + dt * f(x_new, boundary_inputs)

    equivalently::

        R(x_new) = x_new - x_old - dt * f(x_new, ...) = 0

    using Newton's method with a fixed number of iterations.

    The Jacobian is computed via ``jax.jacfwd`` (forward-mode AD).
    For small systems this is efficient; for large systems consider
    using GMRES or matrix-free approaches.

    Parameters
    ----------
    residual_fn : callable
        ``(state_new, state_old, boundary_inputs, dt) -> {field: residual}``
        The node's ``implicit_residual`` method.
    state_old : dict
        State at the beginning of the timestep.
    boundary_inputs : dict
        Boundary inputs for this step.
    dt : float
        Timestep.
    n_newton : int
        Number of Newton iterations (fixed for JIT compatibility).
    initial_guess : dict or None
        Initial guess for x_new.  If None, uses ``state_old``
        (first-order predictor from explicit Euler could be better).

    Returns
    -------
    dict
        The solved state x_new.
    """
    if initial_guess is None:
        x = {k: v.copy() for k, v in state_old.items()}
    else:
        x = {k: v.copy() for k, v in initial_guess.items()}

    fields = sorted(x.keys())

    def _flatten(d):
        return jnp.concatenate([jnp.ravel(d[f]) for f in fields])

    def _unflatten(flat):
        result = {}
        offset = 0
        for f in fields:
            shape = x[f].shape
            size = 1
            for s in shape:
                size *= s
            result[f] = flat[offset:offset + size].reshape(shape)
            offset += size
        return result

    def residual_flat(x_flat):
        x_dict = _unflatten(x_flat)
        res = residual_fn(x_dict, state_old, boundary_inputs, dt)
        return jnp.concatenate([jnp.ravel(res[f]) for f in fields])

    x_flat = _flatten(x)

    def newton_step(i, x_f):
        r = residual_flat(x_f)
        J = jax.jacfwd(residual_flat)(x_f)
        # Solve J * dx = -r using regularized linear solve
        n = x_f.shape[0]
        J_reg = J + 1e-10 * jnp.eye(n)
        dx = jnp.linalg.solve(J_reg, -r)
        return x_f + dx

    x_flat = jax.lax.fori_loop(0, n_newton, newton_step, x_flat)
    return _unflatten(x_flat)
