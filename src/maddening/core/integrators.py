"""
Pluggable ODE integrators for nodes that implement ``derivatives()``.

All integrators are pure JAX functions suitable for JIT, grad, and scan.
They take a derivatives function, current state, boundary inputs, and dt,
and return the new state.

Integrators
-----------
- ``euler_step``: Forward Euler (1st order) -- baseline
- ``rk4_step``: Classical Runge-Kutta (4th order) -- high accuracy
- ``heun_step``: Heun's method (2nd order) -- good balance

Graph-Level Integration
-----------------------
- ``make_rk4_graph_step``: Wraps a graph's derivative function in RK4
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
        New state after one Euler step.
    """
    derivs = derivatives_fn(state, boundary_inputs)
    return {k: state[k] + dt * derivs[k] for k in derivs}


def heun_step(
    derivatives_fn: Callable,
    state: dict,
    boundary_inputs: dict,
    dt: float,
) -> dict:
    """Heun's method (2nd order): predictor-corrector.

    Parameters
    ----------
    derivatives_fn : callable
        ``(state, boundary_inputs) -> {field: d_field/dt}``
    state : dict
        Current state.
    boundary_inputs : dict
        Boundary inputs (held constant across stages).
    dt : float
        Timestep.

    Returns
    -------
    dict
        New state after one Heun step.
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
    """Classical 4th-order Runge-Kutta integration.

    Parameters
    ----------
    derivatives_fn : callable
        ``(state, boundary_inputs) -> {field: d_field/dt}``
    state : dict
        Current state.
    boundary_inputs : dict
        Boundary inputs (held constant across stages).
    dt : float
        Timestep.

    Returns
    -------
    dict
        New state after one RK4 step.
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
# Graph-level integration
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
        Boundary inputs for this step.
    dt : float
        Timestep.
    method : str
        Integration method: ``"euler"``, ``"heun"``, or ``"rk4"``.

    Returns
    -------
    dict
        New state after integration.
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
