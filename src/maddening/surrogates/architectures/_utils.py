"""Shared utilities for surrogate architectures."""

import math
from collections.abc import Mapping
from typing import Any, Sequence

import jax
import jax.numpy as jnp

try:
    import equinox as eqx
except ImportError:
    eqx = None


def check_equinox() -> None:
    if eqx is None:
        raise ImportError(
            "This architecture requires equinox. "
            "Install with: pip install maddening[surrogates]"
        )


def compute_sizes(
    state_spec: Mapping[str, tuple[int, ...]],
    boundary_spec: Mapping[str, tuple[int, ...]],
) -> tuple[int, int]:
    """Compute input and output sizes from specs."""
    input_size = sum(math.prod(s) if s else 1 for s in state_spec.values())
    input_size += sum(math.prod(s) if s else 1 for s in boundary_spec.values())
    input_size += 1  # dt
    output_size = sum(math.prod(s) if s else 1 for s in state_spec.values())
    return input_size, output_size


# shape: state/boundary_inputs are {field: Array} -- TypedDict candidate (phase 3)
def flatten_inputs(
    state: Mapping[str, Any],
    boundary_inputs: Mapping[str, Any],
    dt: float,
    state_spec: Mapping[str, tuple[int, ...]],
    boundary_spec: Mapping[str, tuple[int, ...]],
) -> jax.Array:
    """Flatten state + boundary_inputs + dt into a single vector."""
    parts = []
    for field in sorted(state_spec.keys()):
        parts.append(jnp.ravel(state[field]))
    for field in sorted(boundary_spec.keys()):
        if field in boundary_inputs:
            parts.append(jnp.ravel(boundary_inputs[field]))
        else:
            size = math.prod(boundary_spec[field]) if boundary_spec[field] else 1
            parts.append(jnp.zeros(size))
    parts.append(jnp.atleast_1d(jnp.asarray(dt, dtype=jnp.float32)))
    return jnp.concatenate(parts)


# shape: state_spec is {field: shape}, result is {field: Array} -- TypedDict candidates (phase 3)
def unflatten_output(
    output_vec: jax.Array,
    state_spec: Mapping[str, tuple[int, ...]],
) -> dict[str, jax.Array]:
    """Reshape a flat output vector back into a state dict."""
    result = {}
    offset = 0
    for field in sorted(state_spec.keys()):
        shape = state_spec[field]
        size = math.prod(shape) if shape else 1
        val = output_vec[offset:offset + size]
        result[field] = val.reshape(shape) if shape else val.squeeze()
        offset += size
    return result


# shape: state is {field: Array}, result is {field: shape} -- TypedDict candidates (phase 3)
def get_state_spec(state: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Infer state_spec from a state dict."""
    return {k: state[k].shape for k in sorted(state.keys())}


# shape: boundary_inputs is {field: Array}, result is {field: shape} -- TypedDict candidates (phase 3)
def get_boundary_spec(
    boundary_inputs: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    """Infer boundary_spec from a boundary_inputs dict."""
    if not boundary_inputs:
        return {}
    return {k: boundary_inputs[k].shape for k in sorted(boundary_inputs.keys())}
