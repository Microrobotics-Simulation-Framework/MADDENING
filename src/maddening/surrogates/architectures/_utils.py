"""Shared utilities for surrogate architectures."""

import math
from typing import Sequence

import jax
import jax.numpy as jnp

from maddening.surrogates.types import MutableStateDict, SpecDict, StateDict

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
    state_spec: SpecDict,
    boundary_spec: SpecDict,
) -> tuple[int, int]:
    """Compute input and output sizes from specs."""
    input_size = sum(math.prod(s) if s else 1 for s in state_spec.values())
    input_size += sum(math.prod(s) if s else 1 for s in boundary_spec.values())
    input_size += 1  # dt
    output_size = sum(math.prod(s) if s else 1 for s in state_spec.values())
    return input_size, output_size


# Dynamic keys (the node's own fields): an alias, not a TypedDict.
def flatten_inputs(
    state: StateDict,
    boundary_inputs: StateDict,
    dt: float,
    state_spec: SpecDict,
    boundary_spec: SpecDict,
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


# Dynamic keys (the node's own fields): an alias, not a TypedDict.
def unflatten_output(
    output_vec: jax.Array,
    state_spec: SpecDict,
) -> MutableStateDict:
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


# Dynamic keys (the node's own fields): an alias, not a TypedDict.
def get_state_spec(state: StateDict) -> dict[str, tuple[int, ...]]:
    """Infer state_spec from a state dict."""
    return {k: state[k].shape for k in sorted(state.keys())}


# Dynamic keys (the node's own fields): an alias, not a TypedDict.
def get_boundary_spec(
    boundary_inputs: StateDict,
) -> dict[str, tuple[int, ...]]:
    """Infer boundary_spec from a boundary_inputs dict."""
    if not boundary_inputs:
        return {}
    return {k: boundary_inputs[k].shape for k in sorted(boundary_inputs.keys())}
