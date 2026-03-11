"""
MLP-based surrogate architectures (direct and derivative modes).

Uses Equinox for the MLP implementation but exposes weights as a
plain JAX pytree for framework-independence at the SurrogateNode level.
"""

import math
from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from maddening.surrogates.architecture import SurrogateArchitecture

try:
    import equinox as eqx
except ImportError:
    eqx = None


def _check_equinox():
    if eqx is None:
        raise ImportError(
            "MLP architectures require equinox. "
            "Install with: pip install maddening[surrogates]"
        )


def _compute_sizes(state_spec, boundary_spec):
    """Compute input and output sizes from specs."""
    input_size = sum(math.prod(s) if s else 1 for s in state_spec.values())
    input_size += sum(math.prod(s) if s else 1 for s in boundary_spec.values())
    input_size += 1  # dt
    output_size = sum(math.prod(s) if s else 1 for s in state_spec.values())
    return input_size, output_size


def _flatten_inputs(state, boundary_inputs, dt, state_spec, boundary_spec):
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


def _unflatten_output(output_vec, state_spec):
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


class MLPDirect(SurrogateArchitecture):
    """MLP that directly predicts the next state.

    Parameters
    ----------
    hidden_sizes : sequence of int
        Hidden layer sizes (default ``[64, 64]``).
    activation : callable
        Activation function (default ``jax.nn.relu``).
    """

    mode = "direct"

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = jax.nn.relu,
    ):
        _check_equinox()
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation

    def init_params(self, rng_key, state_spec, boundary_spec):
        input_size, output_size = _compute_sizes(state_spec, boundary_spec)
        sizes = (input_size,) + self.hidden_sizes + (output_size,)

        # Build Equinox MLP and extract weight pytree
        mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=self.hidden_sizes[0] if len(set(self.hidden_sizes)) == 1 else self.hidden_sizes[0],
            depth=len(self.hidden_sizes),
            activation=self.activation,
            key=rng_key,
        )
        return eqx.partition(mlp, eqx.is_array)

    def forward(self, params, state, boundary_inputs, dt):
        arrays, static = params
        mlp = eqx.combine(arrays, static)
        x = _flatten_inputs(
            state, boundary_inputs, dt,
            # Use sorted keys to match init ordering
            {k: state[k].shape for k in sorted(state.keys())},
            {k: boundary_inputs[k].shape for k in sorted(boundary_inputs.keys())} if boundary_inputs else {},
        )
        output = mlp(x)
        return _unflatten_output(
            output,
            {k: state[k].shape for k in sorted(state.keys())},
        )


class MLPDerivative(SurrogateArchitecture):
    """MLP that predicts d(state)/dt (derivative mode).

    SurrogateNode integrates the output using the configured integrator
    (default Euler, optionally RK4).

    Parameters
    ----------
    hidden_sizes : sequence of int
        Hidden layer sizes (default ``[64, 64]``).
    activation : callable
        Activation function (default ``jax.nn.relu``).
    """

    mode = "derivative"

    def __init__(
        self,
        hidden_sizes: Sequence[int] = (64, 64),
        activation: Callable = jax.nn.relu,
    ):
        _check_equinox()
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation

    def init_params(self, rng_key, state_spec, boundary_spec):
        input_size, output_size = _compute_sizes(state_spec, boundary_spec)

        mlp = eqx.nn.MLP(
            in_size=input_size,
            out_size=output_size,
            width_size=self.hidden_sizes[0] if len(set(self.hidden_sizes)) == 1 else self.hidden_sizes[0],
            depth=len(self.hidden_sizes),
            activation=self.activation,
            key=rng_key,
        )
        return eqx.partition(mlp, eqx.is_array)

    def forward(self, params, state, boundary_inputs, dt):
        arrays, static = params
        mlp = eqx.combine(arrays, static)
        x = _flatten_inputs(
            state, boundary_inputs, dt,
            {k: state[k].shape for k in sorted(state.keys())},
            {k: boundary_inputs[k].shape for k in sorted(boundary_inputs.keys())} if boundary_inputs else {},
        )
        output = mlp(x)
        return _unflatten_output(
            output,
            {k: state[k].shape for k in sorted(state.keys())},
        )
