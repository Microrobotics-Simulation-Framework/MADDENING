"""Hypothesis strategies for property-based testing of SimulationNodes.

Provides reusable strategies that generate valid inputs based on a
node's declared interface (initial_state, boundary_input_spec).

Usage::

    from maddening.testing.strategies import node_states, bounded_dt
    from hypothesis import given

    node = MyPhysicsNode(name="test", timestep=0.01)

    @given(state=node_states(node, bounds={"pos": (-10, 10)}),
           dt=bounded_dt())
    def test_update_finite(state, dt):
        out = node.update(state, {}, dt)
        for v in out.values():
            assert jnp.all(jnp.isfinite(v))

Requires ``hypothesis >= 6.165``. Install via::

    pip install maddening[verify]
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays
except ImportError as e:
    raise ImportError(
        "hypothesis is required for property-based testing. "
        "Install it with: pip install maddening[verify]"
    ) from e

import jax.numpy as jnp


def node_states(
    node,
    bounds: dict[str, tuple[float, float]] | None = None,
    *,
    dtype: np.dtype | None = None,
) -> st.SearchStrategy:
    """Strategy that generates valid state dicts for a node.

    Reads the node's ``initial_state()`` to determine field names and
    shapes, then generates arrays within the declared bounds.

    Parameters
    ----------
    node : SimulationNode
        The node whose state structure to match.
    bounds : dict, optional
        Per-field (lo, hi) bounds. Fields not in bounds default to
        (-1e4, 1e4).
    dtype : numpy dtype
        Array dtype for generated states.
    """
    if bounds is None:
        bounds = {}

    initial = node.initial_state()

    field_strategies = {}
    for field, arr in initial.items():
        shape = tuple(int(d) for d in arr.shape)
        lo, hi = bounds.get(field, (-1e4, 1e4))
        field_dtype = dtype if dtype is not None else np.float32
        width = 32 if field_dtype == np.float32 else 64
        field_strategies[field] = arrays(
            dtype=field_dtype,
            shape=shape,
            elements=st.floats(
                min_value=lo, max_value=hi,
                allow_nan=False, allow_infinity=False,
                width=width,
            ),
        ).map(jnp.asarray)

    @st.composite
    def strategy(draw):
        return {k: draw(v) for k, v in field_strategies.items()}

    return strategy()


def bounded_dt(
    min_value: float = 1e-6,
    max_value: float = 0.1,
) -> st.SearchStrategy:
    """Strategy for realistic timestep values.

    Parameters
    ----------
    min_value : float
        Minimum timestep.
    max_value : float
        Maximum timestep.
    """
    return st.floats(
        min_value=min_value, max_value=max_value,
        allow_nan=False, allow_infinity=False,
    )


def boundary_inputs_for(
    node,
    bounds: dict[str, tuple[float, float]] | None = None,
    *,
    dtype: np.dtype = np.float64,
) -> st.SearchStrategy:
    """Strategy that generates boundary inputs matching a node's spec.

    Reads the node's ``boundary_input_spec()`` (if it exists) and
    generates a matching dict. Falls back to empty dict if the node
    has no spec.

    Parameters
    ----------
    node : SimulationNode
        The node whose boundary inputs to generate.
    bounds : dict, optional
        Per-input-name (lo, hi) bounds.
    dtype : numpy dtype
        Array dtype.
    """
    if bounds is None:
        bounds = {}

    if not hasattr(node, "boundary_input_spec"):
        return st.just({})

    spec = node.boundary_input_spec()
    if not spec:
        return st.just({})

    input_strategies = {}
    for name, input_spec in spec.items():
        shape = tuple(int(d) for d in input_spec.shape) if hasattr(input_spec, "shape") else ()
        lo, hi = bounds.get(name, (-1e4, 1e4))
        input_strategies[name] = arrays(
            dtype=dtype,
            shape=shape,
            elements=st.floats(
                min_value=lo, max_value=hi,
                allow_nan=False, allow_infinity=False,
            ),
        ).map(jnp.asarray)

    @st.composite
    def strategy(draw):
        return {k: draw(v) for k, v in input_strategies.items()}

    return strategy()
