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
import numpy.typing as npt

try:
    from hypothesis import strategies as st
    from hypothesis.extra.numpy import arrays
except ImportError as e:
    raise ImportError(
        "hypothesis is required for property-based testing. "
        "Install it with: pip install maddening[verify]"
    ) from e

import jax.numpy as jnp
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


def _representable(lo: float, hi: float, dtype: np.dtype) -> tuple[float, float]:
    """Round ``(lo, hi)`` inward to values exactly representable in ``dtype``.

    Hypothesis refuses ``st.floats(min_value=0.1, width=32)`` because
    ``0.1`` is not a float32; rounding inward keeps samples inside the
    envelope the caller declared.
    """
    lo_r, hi_r = dtype.type(lo), dtype.type(hi)
    if lo_r < lo:
        lo_r = np.nextafter(lo_r, dtype.type(np.inf))
    if hi_r > hi:
        hi_r = np.nextafter(hi_r, dtype.type(-np.inf))
    if lo_r > hi_r:
        raise ValueError(f"bounds ({lo}, {hi}) contain no {dtype} value")
    return float(lo_r), float(hi_r)


@stability(StabilityLevel.EXPERIMENTAL)
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
        (-1e4, 1e4) (integer fields: the dtype's full range).
    dtype : numpy dtype
        Array dtype for generated *floating* states.  Fields whose
        ``initial_state`` is bool or integer keep that dtype and are
        sampled as booleans / integers, so ``structure`` can check dtype
        preservation on monitor-style nodes.
    """
    if bounds is None:
        bounds = {}

    initial = node.initial_state()

    field_strategies = {}
    for field, arr in initial.items():
        shape = tuple(int(d) for d in arr.shape)
        # `Any`: the branches below select on it with `np.issubdtype`,
        # which the checker cannot use to narrow a `dtype[...]` union.
        init_dtype: Any = np.dtype(getattr(arr, "dtype", np.float32))
        if init_dtype == np.bool_:
            # Flags (a health monitor's status bits): sample both values.
            field_strategies[field] = arrays(
                dtype=np.bool_, shape=shape, elements=st.booleans(),
            ).map(jnp.asarray)
            continue
        if np.issubdtype(init_dtype, np.integer):
            # Counters / indices keep their integer dtype; ``bounds`` (if
            # given) are rounded inward, else the dtype's own range.
            info = np.iinfo(init_dtype)
            lo_b, hi_b = bounds.get(field, (info.min, info.max))
            lo_i = int(max(np.ceil(lo_b), info.min))
            hi_i = int(min(np.floor(hi_b), info.max))
            field_strategies[field] = arrays(
                dtype=init_dtype, shape=shape,
                elements=st.integers(min_value=lo_i, max_value=hi_i),
            ).map(jnp.asarray)
            continue
        field_dtype = np.dtype(dtype if dtype is not None else np.float32)
        lo, hi = _representable(*bounds.get(field, (-1e4, 1e4)), field_dtype)
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


@stability(StabilityLevel.EXPERIMENTAL)
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


@stability(StabilityLevel.EXPERIMENTAL)
def boundary_inputs_for(
    node,
    bounds: dict[str, tuple[float, float]] | None = None,
    *,
    dtype: npt.DTypeLike = np.float64,
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

    dtype = np.dtype(dtype)
    width = 32 if dtype == np.float32 else 64
    input_strategies = {}
    for name, input_spec in spec.items():
        shape = tuple(int(d) for d in input_spec.shape) if hasattr(input_spec, "shape") else ()
        lo, hi = _representable(*bounds.get(name, (-1e4, 1e4)), dtype)
        input_strategies[name] = arrays(
            dtype=dtype,
            shape=shape,
            elements=st.floats(
                min_value=lo, max_value=hi,
                allow_nan=False, allow_infinity=False,
                width=width,
            ),
        ).map(jnp.asarray)

    @st.composite
    def strategy(draw):
        return {k: draw(v) for k, v in input_strategies.items()}

    return strategy()
