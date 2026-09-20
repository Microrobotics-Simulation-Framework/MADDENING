"""
Pluggable explicit ODE integrators for nodes that implement ``derivatives()``.

All integrators are pure JAX functions suitable for JIT, grad, and scan.
They take a derivatives function, current state, boundary inputs, and dt,
and return the new state.

Integrators
-----------
- ``euler_step``: Forward Euler -- 1st order
- ``heun_step``: Heun's method (explicit trapezoid) -- 2nd order
- ``rk4_step``: Classical Runge-Kutta -- 4th order

What the order claim is a claim about
-------------------------------------
None of these functions is given a time.  Each stage calls
``derivatives_fn(stage_state, boundary_inputs)`` with the *same*
``boundary_inputs`` object the caller passed for the whole step, so the
problem they integrate to the stated order is

.. math::  \\frac{dx}{dt} = f(x, u)

with ``u`` fixed over the step.  That is the autonomous problem, and for
it the orders above are exact.

**If ``u`` genuinely varies with time and the caller re-evaluates it once
per step, every method here degrades to 1st order**, because a
zero-order hold on the input contributes an O(dt^2) error per step
whatever the stage arithmetic does with it.  Measured on
``dx/dt = -x + u(t)``, ``u`` sinusoidal, over a 10/20/40/80/160 ladder in
float64 (``tests/verification/test_integrator_order.py``):

===========  ====================  =================================
method       ``u`` frozen per step ``u`` read at the stage time
===========  ====================  =================================
``euler``    1.02                  1.02
``heun``     1.02                  2.00
``rk4``      1.02                  4.00
===========  ====================  =================================

In the frozen column ``rk4`` is not merely 1st order: on that problem it
is also ~1.5x *less* accurate than ``euler`` at the same ``dt``, because
the extra stages refine a term that is no longer the dominant error.
Picking a higher-order method without giving it stage-time inputs is
therefore not a conservative choice.  This is recorded as
``MADD-ANO-014``.

Driving a time-varying input at full order
------------------------------------------
Make time part of the state, with ``dt/dt = 1``.  Every method here
builds its stage states as ``state + a_ij * dt * k_j``, so a state field
whose derivative is 1 arrives at each stage holding exactly the Butcher
node ``t + c_i * dt`` -- ``(0,)`` for Euler, ``(0, 1)`` for Heun,
``(0, 1/2, 1/2, 1)`` for RK4.  This is the textbook autonomisation of a
non-autonomous ODE, it is exact rather than an interpolation, and it
needs nothing from this module::

    def forced(state, _unused):
        t = state["time"]
        rest = {k: v for k, v in state.items() if k != "time"}
        derivs = node.derivatives(rest, {"u": u_of(t)})
        return {**derivs, "time": jnp.asarray(1.0)}

    state = {**node.initial_state(), "time": jnp.asarray(0.0)}
    for _ in range(n_steps):
        state = rk4_step(forced, state, {}, dt)

``u_of`` must be a function of time alone.  When the input is not -- when
it is data exchanged with another node during the step -- no single-node
integrator can reach stage times for it, and the order is set by the
coupling scheme instead; see ``docs/algorithm_guide/coupling/``.

See ``docs/algorithm_guide/solvers/explicit_integrators.md`` for the
tableaux, the order conditions, and the measured evidence.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp



def euler_step(
    derivatives_fn: Callable,
    state: dict,
    boundary_inputs: dict,
    dt: float,
) -> dict:
    """Forward Euler integration: x_{n+1} = x_n + dt * f(x_n).

    1st order.  Being a one-stage method it evaluates ``derivatives_fn``
    only at the start of the step, so unlike :func:`heun_step` and
    :func:`rk4_step` it loses nothing to a per-step input: 1st order is
    all it ever claimed.

    Parameters
    ----------
    derivatives_fn : callable
        ``(state, boundary_inputs) -> {field: d_field/dt}``
    state : dict
        Current state.
    boundary_inputs : dict
        Boundary inputs for this step.
    dt : float
        Timestep.

    Returns
    -------
    dict
        New state after one Euler step.  Contains exactly the fields
        ``derivatives_fn`` returned a derivative for; a state field it
        omits is dropped, not carried.
    """
    derivs = derivatives_fn(state, boundary_inputs)
    return {k: state[k] + dt * derivs[k] for k in derivs}


def heun_step(
    derivatives_fn: Callable,
    state: dict,
    boundary_inputs: dict,
    dt: float,
) -> dict:
    """Heun's method (explicit trapezoid): 2nd order.

    Two stages, at ``t`` and ``t + dt``.

    Parameters
    ----------
    derivatives_fn : callable
        ``(state, boundary_inputs) -> {field: d_field/dt}``
    state : dict
        Current state.
    boundary_inputs : dict
        Boundary inputs, passed unchanged to both stages.
    dt : float
        Timestep.

    Returns
    -------
    dict
        New state after one Heun step.

    Notes
    -----
    2nd order for the autonomous problem.
    The stages all use the ``boundary_inputs`` passed in, so this order
    holds for ``dx/dt = f(x, u)`` with ``u`` fixed over the step.  A
    caller who re-evaluates a time-varying ``u`` once per step gets 1st
    order from this function whatever its nominal order
    (``MADD-ANO-014``).  To keep the full order, carry time in the state
    with derivative 1 so each stage reads its own ``t + c_i*dt``; the
    module docstring has the recipe.
    """
    k1 = derivatives_fn(state, boundary_inputs)
    s_tilde = {k: state[k] + dt * k1[k] for k in k1}
    k2 = derivatives_fn(s_tilde, boundary_inputs)
    return {k: state[k] + 0.5 * dt * (k1[k] + k2[k]) for k in k1}


def rk4_step(
    derivatives_fn: Callable,
    state: dict,
    boundary_inputs: dict,
    dt: float,
) -> dict:
    """Classical 4th-order Runge-Kutta.

    Four stages, at Butcher nodes ``c = (0, 1/2, 1/2, 1)``.

    Parameters
    ----------
    derivatives_fn : callable
        ``(state, boundary_inputs) -> {field: d_field/dt}``
    state : dict
        Current state.
    boundary_inputs : dict
        Boundary inputs, passed unchanged to all four stages.
    dt : float
        Timestep.

    Returns
    -------
    dict
        New state after one RK4 step.

    Notes
    -----
    4th order for the autonomous problem.
    The stages all use the ``boundary_inputs`` passed in, so this order
    holds for ``dx/dt = f(x, u)`` with ``u`` fixed over the step.  A
    caller who re-evaluates a time-varying ``u`` once per step gets 1st
    order from this function whatever its nominal order
    (``MADD-ANO-014``).  To keep the full order, carry time in the state
    with derivative 1 so each stage reads its own ``t + c_i*dt``; the
    module docstring has the recipe.
    """
    k1 = derivatives_fn(state, boundary_inputs)
    s2 = {k: state[k] + 0.5 * dt * k1[k] for k in k1}
    k2 = derivatives_fn(s2, boundary_inputs)
    s3 = {k: state[k] + 0.5 * dt * k2[k] for k in k1}
    k3 = derivatives_fn(s3, boundary_inputs)
    s4 = {k: state[k] + dt * k3[k] for k in k1}
    k4 = derivatives_fn(s4, boundary_inputs)
    return {
        k: state[k] + (dt / 6.0) * (k1[k] + 2.0 * k2[k] + 2.0 * k3[k] + k4[k])
        for k in k1
    }


# ------------------------------------------------------------------
# Node-level integration
# ------------------------------------------------------------------

def integrate_node(
    node,
    state: dict,
    boundary_inputs: dict,
    dt: float,
    method: str = "rk4",
) -> dict:
    """Integrate a single node using its ``derivatives()`` method.

    Parameters
    ----------
    node : SimulationNode
        A node that implements ``derivatives()``.
    state : dict
        Current node state.
    boundary_inputs : dict
        Boundary inputs for this step, passed unchanged to every stage.
    dt : float
        Timestep.
    method : str
        Integration method: ``"euler"``, ``"heun"``, or ``"rk4"``.

    Returns
    -------
    dict
        New state after integration.

    Notes
    -----
    This convenience wrapper hands ``node.derivatives`` straight to the
    chosen stepper, so it can only offer the autonomous-problem order:
    with a time-varying ``boundary_inputs`` re-evaluated once per step,
    ``method="rk4"`` converges at 1st order (``MADD-ANO-014``).  There is
    nowhere in this signature to put a stage time, deliberately -- the
    caller who needs one composes the time-augmented derivatives function
    from the module docstring and calls :func:`rk4_step` directly.
    """
    integrators = {
        "euler": euler_step,
        "heun": heun_step,
        "rk4": rk4_step,
    }
    if method not in integrators:
        raise ValueError(
            f"Unknown integration method '{method}'. "
            f"Choose from: {list(integrators.keys())}"
        )
    return integrators[method](node.derivatives, state, boundary_inputs, dt)
