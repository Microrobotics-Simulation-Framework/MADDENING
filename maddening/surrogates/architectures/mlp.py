"""
MLP-based surrogate architectures (direct and derivative modes).

Uses Equinox for the MLP implementation but exposes weights as a
plain JAX pytree for framework-independence at the SurrogateNode level.
"""

from typing import Callable, Sequence

import jax
import jax.numpy as jnp

from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.architectures._utils import (
    check_equinox,
    compute_sizes,
    flatten_inputs,
    get_boundary_spec,
    get_state_spec,
    unflatten_output,
)

try:
    import equinox as eqx
except ImportError:
    eqx = None


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
        check_equinox()
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation

    def init_params(self, rng_key, state_spec, boundary_spec):
        input_size, output_size = compute_sizes(state_spec, boundary_spec)

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
        x = flatten_inputs(
            state, boundary_inputs, dt,
            get_state_spec(state),
            get_boundary_spec(boundary_inputs),
        )
        output = mlp(x)
        return unflatten_output(output, get_state_spec(state))


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
        check_equinox()
        self.hidden_sizes = tuple(hidden_sizes)
        self.activation = activation

    def init_params(self, rng_key, state_spec, boundary_spec):
        input_size, output_size = compute_sizes(state_spec, boundary_spec)

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
        x = flatten_inputs(
            state, boundary_inputs, dt,
            get_state_spec(state),
            get_boundary_spec(boundary_inputs),
        )
        output = mlp(x)
        return unflatten_output(output, get_state_spec(state))
