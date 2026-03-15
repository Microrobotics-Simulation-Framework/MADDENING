"""
Calibration utilities for MADDENING simulations.

Provides:

- ``tune_coupling_params``: black-box grid search over coupling
  parameters (tolerance, relaxation, acceleration method).

- ``calibrate``: differentiable parameter recovery using inline
  physics with JAX-traced parameters.  Uses ``jax.grad`` to
  minimise a loss function over node/physics parameters.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import jax.numpy as jnp


@dataclass
class TuneResult:
    """Result of coupling parameter tuning.

    Attributes
    ----------
    best_params : dict
        The coupling parameter dict that minimises total iteration
        count while staying within the accuracy threshold.
    best_total_iters : int
        Total coupling iterations across all timesteps for the best
        configuration.
    best_max_error : float
        Maximum deviation from the reference trajectory.
    all_trials : list[dict]
        Results for every tried configuration, each with keys
        ``"params"``, ``"total_iters"``, ``"max_error"``, ``"passed"``.
    """
    best_params: dict
    best_total_iters: int
    best_max_error: float
    all_trials: list[dict]


def _make_param_grid(param_grid: dict[str, list]) -> list[dict]:
    """Expand a parameter grid into a list of configurations."""
    import itertools
    keys = sorted(param_grid.keys())
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    return [dict(zip(keys, vals)) for vals in combos]


def tune_coupling_params(
    build_graph_fn: Callable[..., Any],
    param_grid: dict[str, list],
    n_steps: int = 100,
    reference_params: Optional[dict] = None,
    accuracy_threshold: float = 1e-3,
    state_metric: Optional[Callable] = None,
    external_inputs: Optional[dict] = None,
) -> TuneResult:
    """Try different coupling parameters and return the best.

    Builds the graph multiple times with different coupling
    parameters, runs each for *n_steps*, and compares accuracy
    against a tight-tolerance reference.

    Parameters
    ----------
    build_graph_fn : callable
        ``(**coupling_kwargs) -> GraphManager``
        A function that builds and returns a configured GraphManager.
        The coupling group must have ``diagnostics=True`` for
        iteration counting.  The function receives coupling keyword
        arguments (tolerance, max_iterations, etc.) from the grid.
    param_grid : dict[str, list]
        Grid of coupling parameters to search.  Example::

            {
                "tolerance": [1e-4, 1e-6, 1e-8],
                "max_iterations": [5, 10, 20],
                "acceleration": ["none", "aitken"],
            }
    n_steps : int
        Number of simulation steps to run per trial.
    reference_params : dict or None
        Coupling parameters for the reference run.  If None, uses
        the tightest tolerance in the grid with the most iterations.
    accuracy_threshold : float
        Maximum acceptable deviation from the reference trajectory.
        Trials exceeding this are rejected.
    state_metric : callable or None
        ``(state_dict) -> scalar``
        Extracts a scalar metric from the state for comparison.
        If None, uses the sum of all node state fields.
    external_inputs : dict or None
        External inputs passed to each run.

    Returns
    -------
    TuneResult
        The best configuration and all trial results.
    """
    # Default metric: sum of all state values
    if state_metric is None:
        def state_metric(state):
            total = 0.0
            for nn in state:
                if nn.startswith("_"):
                    continue
                for field in state[nn]:
                    total = total + float(jnp.sum(state[nn][field]))
            return total

    # Generate parameter combinations
    configs = _make_param_grid(param_grid)

    # Reference run: tight tolerance, many iterations
    if reference_params is None:
        reference_params = {}
        if "tolerance" in param_grid:
            reference_params["tolerance"] = min(param_grid["tolerance"])
        if "max_iterations" in param_grid:
            reference_params["max_iterations"] = max(
                param_grid["max_iterations"]
            )
        if "acceleration" in param_grid:
            # Prefer aitken or iqn-ils for reference
            for acc in ["iqn-ils", "aitken", "none"]:
                if acc in param_grid["acceleration"]:
                    reference_params["acceleration"] = acc
                    break

    # Run reference
    gm_ref = build_graph_fn(**reference_params)
    gm_ref.compile()
    ref_trajectory = []
    for _ in range(n_steps):
        state = gm_ref.step(external_inputs=external_inputs)
        ref_trajectory.append(state_metric(state))

    # Test each configuration
    all_trials = []
    best = None

    for params in configs:
        try:
            gm = build_graph_fn(**params)
            gm.compile()

            trajectory = []
            total_iters = 0
            for step_i in range(n_steps):
                state = gm.step(external_inputs=external_inputs)
                trajectory.append(state_metric(state))
                diag = gm.coupling_diagnostics()
                for group_key, group_diag in diag.items():
                    total_iters += group_diag["iterations"]

            # Compute max error
            max_error = max(
                abs(t - r)
                for t, r in zip(trajectory, ref_trajectory)
            )
            passed = max_error <= accuracy_threshold

            trial = {
                "params": params,
                "total_iters": total_iters,
                "max_error": max_error,
                "passed": passed,
            }
            all_trials.append(trial)

            if passed and (best is None or total_iters < best["total_iters"]):
                best = trial

        except Exception as e:
            all_trials.append({
                "params": params,
                "total_iters": float("inf"),
                "max_error": float("inf"),
                "passed": False,
                "error": str(e),
            })

    if best is None:
        # No passing configuration found -- return the one with
        # lowest error
        best = min(all_trials, key=lambda t: t.get("max_error", float("inf")))

    return TuneResult(
        best_params=best["params"],
        best_total_iters=best["total_iters"],
        best_max_error=best["max_error"],
        all_trials=all_trials,
    )


# ------------------------------------------------------------------
# Differentiable calibration
# ------------------------------------------------------------------

@dataclass
class CalibrateResult:
    """Result of differentiable parameter calibration.

    Attributes
    ----------
    params : dict
        The calibrated parameter values.
    loss_history : list[float]
        Loss value at each optimisation step.
    converged : bool
        Whether the loss dropped below the tolerance.
    """
    params: dict
    loss_history: list[float]
    converged: bool


def calibrate(
    forward_fn: Callable,
    initial_params: dict[str, jnp.ndarray],
    reference_trajectory: Any,
    loss_fn: Optional[Callable] = None,
    n_iters: int = 200,
    learning_rate: float = 0.01,
    tolerance: float = 1e-6,
    verbose: bool = False,
) -> CalibrateResult:
    """Differentiable parameter recovery using JAX gradient descent.

    Optimises physics parameters by minimising a loss function that
    measures deviation from reference data.  The ``forward_fn`` must
    be JAX-traceable and accept the parameter dict as input.

    Parameters
    ----------
    forward_fn : callable
        ``(params_dict) -> predicted``
        A JAX-traceable function that takes a parameter dict (with
        JAX arrays as values) and returns a prediction.
    initial_params : dict[str, jnp.ndarray]
        Starting values for the parameters to optimise.
        Values must be JAX arrays (not Python floats).
    reference_trajectory : any
        The target data to match.  Passed to ``loss_fn``.
    loss_fn : callable or None
        ``(predicted, reference) -> scalar``
        Loss function.  Defaults to MSE over all leaf arrays.
    n_iters : int
        Maximum number of gradient descent steps.
    learning_rate : float
        Step size for gradient descent.
    tolerance : float
        Stop when loss drops below this value.
    verbose : bool
        If True, print loss every 50 iterations.

    Returns
    -------
    CalibrateResult
        Calibrated parameters, loss history, and convergence flag.

    Example
    -------
    >>> import jax.numpy as jnp
    >>> def forward(params):
    ...     g = params["gravity"]
    ...     return 0.5 * g * 1.0**2  # x(t=1) for free fall
    >>> result = calibrate(
    ...     forward_fn=forward,
    ...     initial_params={"gravity": jnp.array(-5.0)},
    ...     reference_trajectory=jnp.array(-4.905),  # true g=-9.81
    ... )
    """
    import jax

    if loss_fn is None:
        def loss_fn(predicted, reference):
            leaves_pred = jax.tree.leaves(predicted)
            leaves_ref = jax.tree.leaves(reference)
            total = jnp.array(0.0)
            for p, r in zip(leaves_pred, leaves_ref):
                total = total + jnp.mean((p - r) ** 2)
            return total

    params = {k: jnp.asarray(v) for k, v in initial_params.items()}
    loss_history = []

    def objective(p):
        predicted = forward_fn(p)
        return loss_fn(predicted, reference_trajectory)

    grad_fn = jax.grad(objective)

    for i in range(n_iters):
        loss_val = float(objective(params))
        loss_history.append(loss_val)

        if verbose and i % 50 == 0:
            print(f"  calibrate step {i}: loss = {loss_val:.6e}")

        if loss_val < tolerance:
            return CalibrateResult(
                params=params,
                loss_history=loss_history,
                converged=True,
            )

        grads = grad_fn(params)
        params = {
            k: params[k] - learning_rate * grads[k]
            for k in params
        }

    return CalibrateResult(
        params=params,
        loss_history=loss_history,
        converged=loss_history[-1] < tolerance if loss_history else False,
    )
