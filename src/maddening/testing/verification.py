"""Formal verification harness for SimulationNode subclasses.

Provides reusable property checks that use stelling to prove numerical
properties of a node's ``update()`` function over declared input
envelopes.

Usage::

    from maddening.testing.verification import verify_node, node_no_overflow

    node = MyPhysicsNode(name="test", timestep=0.01)
    bounds = {"field_a": (-100.0, 100.0), "field_b": (0.0, 1000.0)}
    results = verify_node(node, bounds)
    # results is a dict mapping check names to stelling verdicts

Each check function takes a node + bounds → verdict dict. Users can
also call individual checks directly for finer control.

Requires ``stelling >= 0.1``. Install via::

    pip install maddening[verify]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax.numpy as jnp


def _import_stelling():
    try:
        from stelling.harness import any_array, assert_
        from stelling.preconditions import check
        return any_array, assert_, check
    except ImportError as e:
        raise ImportError(
            "stelling is required for formal verification. "
            "Install it with: pip install maddening[verify]"
        ) from e


@dataclass(frozen=True)
class VerificationResult:
    """Result of a single verification check."""
    name: str
    status: str
    detail: str = ""


def node_shape_stability(
    node,
    bounds: dict[str, tuple[float, float]],
    dt_range: tuple[float, float] = (1e-4, 0.1),
    *,
    solver_timeout_ms: int | None = None,
) -> VerificationResult:
    """Verify that update() preserves state structure.

    Checks that the output pytree has the same keys as the input.
    This is verified by asserting that each output field sum is finite
    (a proxy for "the field exists and has the right shape").

    Parameters
    ----------
    node : SimulationNode
        The node to verify.
    bounds : dict
        Mapping from state field names to (lo, hi) bounds for the
        declared input envelope.
    dt_range : tuple
        (min_dt, max_dt) range for the timestep declaration.
    solver_timeout_ms : int, optional
        Solver timeout for SMT escalation.
    """
    any_array, assert_, check = _import_stelling()

    initial = node.initial_state()

    def harness():
        state = {}
        for field, arr in initial.items():
            if field in bounds:
                lo, hi = bounds[field]
            else:
                lo, hi = -1e6, 1e6
            state[field] = any_array(arr.shape, "float64", (lo, hi))

        dt = any_array((), "float64", dt_range)
        out = node.update(state, {}, dt)

        assertions = []
        for field in initial:
            assertions.append(assert_(jnp.isfinite(jnp.sum(out[field]))))
        return tuple(assertions)

    kwargs = {"vacuity_mode": "inputs-only"}
    if solver_timeout_ms:
        kwargs["solver_timeout_ms"] = solver_timeout_ms

    v = check(harness, **kwargs)
    return VerificationResult(
        name="shape_stability",
        status=v.status,
        detail=v.render() if hasattr(v, "render") else "",
    )


def node_boundedness(
    node,
    bounds: dict[str, tuple[float, float]],
    output_bounds: dict[str, tuple[float, float]],
    dt_range: tuple[float, float] = (1e-4, 0.01),
    *,
    solver_timeout_ms: int | None = None,
) -> VerificationResult:
    """Verify that output state stays within declared bounds.

    For each field in ``output_bounds``, asserts that every element
    of the output is within [lo, hi] after one step.

    Parameters
    ----------
    node : SimulationNode
        The node to verify.
    bounds : dict
        Input state envelope.
    output_bounds : dict
        Expected output bounds per field.
    dt_range : tuple
        Timestep envelope.
    solver_timeout_ms : int, optional
        Solver timeout.
    """
    any_array, assert_, check = _import_stelling()

    initial = node.initial_state()

    def harness():
        state = {}
        for field, arr in initial.items():
            if field in bounds:
                lo, hi = bounds[field]
            else:
                lo, hi = -1e6, 1e6
            state[field] = any_array(arr.shape, "float64", (lo, hi))

        dt = any_array((), "float64", dt_range)
        out = node.update(state, {}, dt)

        assertions = []
        for field, (lo, hi) in output_bounds.items():
            if field in out:
                assertions.append(assert_(out[field] >= lo))
                assertions.append(assert_(out[field] <= hi))
        return tuple(assertions)

    kwargs = {"vacuity_mode": "inputs-only"}
    if solver_timeout_ms:
        kwargs["solver_timeout_ms"] = solver_timeout_ms

    v = check(harness, **kwargs)
    return VerificationResult(
        name="boundedness",
        status=v.status,
        detail=v.render() if hasattr(v, "render") else "",
    )


def node_no_overflow(
    node,
    bounds: dict[str, tuple[float, float]],
    dt_range: tuple[float, float] = (1e-4, 0.01),
    *,
    solver_timeout_ms: int | None = None,
) -> VerificationResult:
    """Verify that update() produces no overflow for bounded inputs.

    Asserts that every element of the output state is within
    representable bounds (proxy for finiteness until stelling gains
    an isfinite transfer).

    Parameters
    ----------
    node : SimulationNode
        The node to verify.
    bounds : dict
        Input state envelope.
    dt_range : tuple
        Timestep envelope.
    solver_timeout_ms : int, optional
        Solver timeout.
    """
    any_array, assert_, check = _import_stelling()

    initial = node.initial_state()

    def harness():
        state = {}
        for field, arr in initial.items():
            if field in bounds:
                lo, hi = bounds[field]
            else:
                lo, hi = -1e6, 1e6
            state[field] = any_array(arr.shape, "float64", (lo, hi))

        dt = any_array((), "float64", dt_range)
        out = node.update(state, {}, dt)

        # Elementwise assertions — stelling checks each element of the
        # array independently (no jnp.all needed; assert_ on an array
        # is already elementwise in stelling).
        assertions = []
        for field in out:
            assertions.append(assert_(out[field] > -1e30))
            assertions.append(assert_(out[field] < 1e30))
        return tuple(assertions)

    kwargs = {"vacuity_mode": "inputs-only"}
    if solver_timeout_ms:
        kwargs["solver_timeout_ms"] = solver_timeout_ms

    v = check(harness, **kwargs)
    return VerificationResult(
        name="no_overflow",
        status=v.status,
        detail=v.render() if hasattr(v, "render") else "",
    )


def node_energy_monotone(
    node,
    bounds: dict[str, tuple[float, float]],
    energy_fn: Callable[[dict], Any],
    dt_range: tuple[float, float] = (1e-4, 0.01),
    *,
    solver_timeout_ms: int | None = None,
) -> VerificationResult:
    """Verify that energy is non-increasing (dissipative system).

    Parameters
    ----------
    node : SimulationNode
        The node to verify.
    bounds : dict
        Input state envelope.
    energy_fn : callable
        Function from state dict to scalar energy.
    dt_range : tuple
        Timestep envelope.
    solver_timeout_ms : int, optional
        Solver timeout.
    """
    any_array, assert_, check = _import_stelling()

    initial = node.initial_state()

    def harness():
        state = {}
        for field, arr in initial.items():
            if field in bounds:
                lo, hi = bounds[field]
            else:
                lo, hi = -1e6, 1e6
            state[field] = any_array(arr.shape, "float64", (lo, hi))

        dt = any_array((), "float64", dt_range)
        e_before = energy_fn(state)
        out = node.update(state, {}, dt)
        e_after = energy_fn(out)
        return (assert_(e_after <= e_before),)

    kwargs = {"vacuity_mode": "inputs-only"}
    if solver_timeout_ms:
        kwargs["solver_timeout_ms"] = solver_timeout_ms

    v = check(harness, **kwargs)
    return VerificationResult(
        name="energy_monotone",
        status=v.status,
        detail=v.render() if hasattr(v, "render") else "",
    )


def verify_node(
    node,
    bounds: dict[str, tuple[float, float]],
    *,
    checks: list[str] | None = None,
    dt_range: tuple[float, float] = (1e-4, 0.01),
    solver_timeout_ms: int | None = None,
) -> dict[str, VerificationResult]:
    """Run all applicable verification checks on a node.

    Parameters
    ----------
    node : SimulationNode
        The node to verify.
    bounds : dict
        Input state envelope: field name → (lo, hi).
    checks : list of str, optional
        Subset of checks to run. Default: all applicable.
        Valid names: "shape_stability", "no_overflow".
    dt_range : tuple
        Timestep envelope.
    solver_timeout_ms : int, optional
        Solver timeout for SMT escalation.

    Returns
    -------
    dict[str, VerificationResult]
        Mapping from check name to result.
    """
    available = {
        "shape_stability": node_shape_stability,
        "no_overflow": node_no_overflow,
    }

    if checks is None:
        checks = list(available.keys())

    results = {}
    for name in checks:
        if name in available:
            results[name] = available[name](
                node, bounds, dt_range,
                solver_timeout_ms=solver_timeout_ms,
            )
    return results
