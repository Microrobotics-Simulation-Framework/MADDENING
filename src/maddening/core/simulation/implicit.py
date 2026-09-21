"""
Implicit time integration via fixed-count Newton iteration.

Provides ``implicit_euler_step`` which solves the backward Euler
equation using Newton's method with a fixed iteration count
(via ``jax.lax.fori_loop`` for JIT/grad/scan compatibility).

For nodes that implement ``implicit_residual()``, this provides
unconditional stability for stiff ODEs.

Graph parameters
----------------
``implicit_euler_step(node.implicit_residual, ..., params=p)`` forwards
``p`` to the residual as ``params=p``, so a value calibrated through
``gm.params`` or ``maddening.sysid.fit`` reaches the implicit solve the
way it reaches ``update`` (the ``{**self.params, **params}`` rule).  The
solver stays a function of a *callable*: it never looks at a node, and a
residual that is not a node method -- a closure, a ``functools.partial``
-- is forwarded the keyword the same way.  A non-empty ``params`` for a
callable that has no ``params`` keyword is a ``ValueError`` naming it,
never a silent solve with the constructor's constants: that divergence
was ``MADD-ANO-018``, resolved in 0.4.0.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

import jax
import jax.numpy as jnp

from maddening.core.node import _method_with_params, _params_empty


def _residual_with_params(residual_fn: Callable, params: Any) -> Callable:
    """``residual_fn`` with ``params`` bound, or a refusal.

    A bound node method goes through the node's own probe
    (``accepts_params(method=...)``), so the refusal names the class and
    the method; any other callable is inspected directly.  Empty
    ``params`` returns the callable untouched, which is what keeps a
    4-argument residual working.
    """
    if _params_empty(params):
        return residual_fn
    owner = getattr(residual_fn, "__self__", None)
    name = getattr(residual_fn, "__name__", None)
    if owner is not None and name and hasattr(owner, "accepts_params"):
        return _method_with_params(owner, name, params)
    try:
        takes_params = "params" in inspect.signature(residual_fn).parameters
    except (TypeError, ValueError):
        takes_params = False
    if not takes_params:
        label = getattr(residual_fn, "__qualname__", None) or repr(residual_fn)
        raise ValueError(
            f"params given, but residual_fn {label} takes no 'params' "
            "keyword: the solve would run the constants it closed over "
            "while update() used the calibrated ones (MADD-ANO-018).  "
            "Accept params=None and read constants from it, close over the "
            "calibrated values yourself, or pass no params."
        )
    return functools.partial(residual_fn, params=params)


def implicit_euler_step(
    residual_fn: Callable,
    state_old: dict,
    boundary_inputs: dict,
    dt: float,
    n_newton: int = 5,
    initial_guess: dict | None = None,
    *,
    params=None,
) -> tuple[dict, jnp.ndarray]:
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
    params : dict, optional
        The node's entry of the graph parameter pytree.  Forwarded to
        ``residual_fn(..., params=params)`` on every Newton evaluation,
        so a node's ``implicit_residual`` reads the calibrated constants
        by the same ``{**self.params, **params}`` rule as ``update``.
        ``None`` or ``{}`` (the default) calls ``residual_fn`` with the
        four positional arguments only, which keeps a residual declared
        without the keyword working.

    Returns
    -------
    state : dict
        The solved state x_new.
    residual_norm : jnp.ndarray
        L2 norm of the final residual.  The caller should check this
        against a tolerance and reject the step if Newton did not
        converge (e.g., shrink dt in adaptive timestepping).

    Raises
    ------
    ValueError
        A non-empty ``params`` for a ``residual_fn`` that takes no
        ``params`` keyword -- for a node method, the message names the
        class and the method.  Solving with the constructor's constants
        while ``update`` used the calibrated ones was ``MADD-ANO-018``
        (resolved in 0.4.0); the refusal is the replacement for that
        silence, so it is never downgraded to a fallback.

    Notes
    -----
    ``params`` is applied by binding it onto ``residual_fn`` before the
    Newton loop (``functools.partial(residual_fn, params=params)``), so
    the loop body and the ``jacfwd`` Jacobian see a plain 4-argument
    callable and a traced ``params`` differentiates through the solve
    like any other closed-over array.  A caller can do the same binding
    by hand and pass no ``params``; the two are equivalent.
    """
    residual = _residual_with_params(residual_fn, params)
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
        res = residual(x_dict, state_old, boundary_inputs, dt)
        return jnp.concatenate([jnp.ravel(res[f]) for f in fields])

    x_flat = _flatten(x)

    def newton_step(i, x_f):
        r = residual_flat(x_f)
        J = jax.jacfwd(residual_flat)(x_f)
        n = x_f.shape[0]
        J_reg = J + 1e-10 * jnp.eye(n)
        dx = jnp.linalg.solve(J_reg, -r)
        return x_f + dx

    x_flat = jax.lax.fori_loop(0, n_newton, newton_step, x_flat)

    final_residual = residual_flat(x_flat)
    residual_norm = jnp.sqrt(jnp.sum(final_residual ** 2))

    return _unflatten(x_flat), residual_norm
