"""
DeepONet-based surrogate architectures.

Implements vanilla DeepONet (branch-trunk) and S-DeepONet (sequential
branch with GRU for time-dependent history).

For MADDENING's lumped-parameter nodes (scalar state fields), the trunk
degenerates to a learned basis matrix — no spatial query points needed.
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


# ------------------------------------------------------------------
# Equinox modules for branch-trunk
# ------------------------------------------------------------------

class _BranchTrunkNet(eqx.Module if eqx is not None else object):
    """Combined branch-trunk network for DeepONet.

    Branch: MLP that maps flattened input → latent of size n_basis.
    Trunk:  Learnable basis matrix of shape (n_basis, output_size).
    Output: branch_out @ trunk_basis
    """
    branch: object  # eqx.nn.MLP
    trunk_basis: jnp.ndarray  # (n_basis, output_size)
    output_bias: jnp.ndarray  # (output_size,)

    def __call__(self, x):
        b = self.branch(x)  # (n_basis,)
        return b @ self.trunk_basis + self.output_bias  # (output_size,)


def _build_branch_trunk(
    rng_key, input_size, output_size, n_basis, branch_hidden, activation,
):
    """Build and partition a BranchTrunkNet."""
    k1, k2, k3 = jax.random.split(rng_key, 3)
    branch = eqx.nn.MLP(
        in_size=input_size,
        out_size=n_basis,
        width_size=branch_hidden[0] if len(set(branch_hidden)) == 1 else branch_hidden[0],
        depth=len(branch_hidden),
        activation=activation,
        key=k1,
    )
    # Xavier init for trunk basis
    scale = jnp.sqrt(2.0 / (n_basis + output_size))
    trunk_basis = jax.random.normal(k2, (n_basis, output_size)) * scale
    output_bias = jnp.zeros(output_size)

    net = _BranchTrunkNet(
        branch=branch,
        trunk_basis=trunk_basis,
        output_bias=output_bias,
    )
    return eqx.partition(net, eqx.is_array)


# ------------------------------------------------------------------
# DeepONet Direct
# ------------------------------------------------------------------

class DeepONetDirect(SurrogateArchitecture):
    """DeepONet that directly predicts the next state.

    Parameters
    ----------
    n_basis : int
        Number of basis functions (latent dimension). Default 64.
    branch_hidden : sequence of int
        Branch net hidden layer sizes. Default (64, 64).
    activation : callable
        Activation function. Default jax.nn.tanh.
    """

    mode = "direct"

    def __init__(
        self,
        n_basis: int = 64,
        branch_hidden: Sequence[int] = (64, 64),
        activation: Callable = jax.nn.tanh,
    ):
        check_equinox()
        self.n_basis = n_basis
        self.branch_hidden = tuple(branch_hidden)
        self.activation = activation

    def init_params(self, rng_key, state_spec, boundary_spec):
        input_size, output_size = compute_sizes(state_spec, boundary_spec)
        return _build_branch_trunk(
            rng_key, input_size, output_size,
            self.n_basis, self.branch_hidden, self.activation,
        )

    def forward(self, params, state, boundary_inputs, dt):
        arrays, static = params
        net = eqx.combine(arrays, static)
        x = flatten_inputs(
            state, boundary_inputs, dt,
            get_state_spec(state),
            get_boundary_spec(boundary_inputs),
        )
        output = net(x)
        return unflatten_output(output, get_state_spec(state))


class DeepONetDerivative(SurrogateArchitecture):
    """DeepONet that predicts d(state)/dt (derivative mode).

    Parameters
    ----------
    n_basis : int
        Number of basis functions. Default 64.
    branch_hidden : sequence of int
        Branch net hidden layer sizes. Default (64, 64).
    activation : callable
        Activation function. Default jax.nn.tanh.
    """

    mode = "derivative"

    def __init__(
        self,
        n_basis: int = 64,
        branch_hidden: Sequence[int] = (64, 64),
        activation: Callable = jax.nn.tanh,
    ):
        check_equinox()
        self.n_basis = n_basis
        self.branch_hidden = tuple(branch_hidden)
        self.activation = activation

    def init_params(self, rng_key, state_spec, boundary_spec):
        input_size, output_size = compute_sizes(state_spec, boundary_spec)
        return _build_branch_trunk(
            rng_key, input_size, output_size,
            self.n_basis, self.branch_hidden, self.activation,
        )

    def forward(self, params, state, boundary_inputs, dt):
        arrays, static = params
        net = eqx.combine(arrays, static)
        x = flatten_inputs(
            state, boundary_inputs, dt,
            get_state_spec(state),
            get_boundary_spec(boundary_inputs),
        )
        output = net(x)
        return unflatten_output(output, get_state_spec(state))


# ------------------------------------------------------------------
# S-DeepONet (Sequential) with GRU branch
# ------------------------------------------------------------------

class _GRUBranchTrunkNet(eqx.Module if eqx is not None else object):
    """S-DeepONet: branch uses a GRU cell for sequential/history input.

    The GRU processes a sequence of past inputs and produces a latent
    vector that is combined with the trunk basis.
    """
    gru_cell: object       # eqx.nn.GRUCell
    branch_proj: object    # eqx.nn.MLP (post-GRU projection to n_basis)
    trunk_basis: jnp.ndarray  # (n_basis, output_size)
    output_bias: jnp.ndarray  # (output_size,)

    def __call__(self, x, hidden):
        """Forward pass.

        Parameters
        ----------
        x : array, shape (input_size,)
            Current input (flattened state + boundary + dt).
        hidden : array, shape (gru_hidden_size,)
            GRU hidden state from previous step.

        Returns
        -------
        output : array, shape (output_size,)
        new_hidden : array, shape (gru_hidden_size,)
        """
        new_hidden = self.gru_cell(x, hidden)
        b = self.branch_proj(new_hidden)  # (n_basis,)
        output = b @ self.trunk_basis + self.output_bias  # (output_size,)
        return output, new_hidden


class SDeepONetDirect(SurrogateArchitecture):
    """Sequential DeepONet (S-DeepONet) with GRU branch, direct mode.

    The GRU maintains a hidden state across timesteps, allowing the
    model to learn from temporal history.  The hidden state must be
    included in the node's state dict for it to persist across steps.

    Parameters
    ----------
    n_basis : int
        Number of basis functions. Default 32.
    gru_hidden_size : int
        GRU hidden state size. Default 32.
    proj_hidden : sequence of int
        Post-GRU projection hidden sizes. Default (32,).
    activation : callable
        Activation for projection MLP. Default jax.nn.tanh.
    """

    mode = "direct"

    def __init__(
        self,
        n_basis: int = 32,
        gru_hidden_size: int = 32,
        proj_hidden: Sequence[int] = (32,),
        activation: Callable = jax.nn.tanh,
    ):
        check_equinox()
        self.n_basis = n_basis
        self.gru_hidden_size = gru_hidden_size
        self.proj_hidden = tuple(proj_hidden)
        self.activation = activation

    def init_params(self, rng_key, state_spec, boundary_spec):
        # Exclude _gru_hidden from physical state for size computation
        phys_spec = {k: v for k, v in state_spec.items() if k != "_gru_hidden"}
        input_size, output_size = compute_sizes(phys_spec, boundary_spec)
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        gru_cell = eqx.nn.GRUCell(input_size, self.gru_hidden_size, key=k1)
        branch_proj = eqx.nn.MLP(
            in_size=self.gru_hidden_size,
            out_size=self.n_basis,
            width_size=self.proj_hidden[0] if self.proj_hidden else self.n_basis,
            depth=len(self.proj_hidden),
            activation=self.activation,
            key=k2,
        )
        scale = jnp.sqrt(2.0 / (self.n_basis + output_size))
        trunk_basis = jax.random.normal(k3, (self.n_basis, output_size)) * scale
        output_bias = jnp.zeros(output_size)

        net = _GRUBranchTrunkNet(
            gru_cell=gru_cell,
            branch_proj=branch_proj,
            trunk_basis=trunk_basis,
            output_bias=output_bias,
        )
        return eqx.partition(net, eqx.is_array)

    def hidden_size(self) -> int:
        """Return the GRU hidden state size (needed for initial state)."""
        return self.gru_hidden_size

    def forward(self, params, state, boundary_inputs, dt):
        """Forward pass with GRU hidden state.

        The state dict must contain a ``"_gru_hidden"`` field holding
        the GRU hidden state array of shape ``(gru_hidden_size,)``.
        This field is updated each step.
        """
        arrays, static = params
        net = eqx.combine(arrays, static)

        hidden = state["_gru_hidden"]

        # Build input from non-hidden state fields
        phys_state = {k: v for k, v in state.items() if k != "_gru_hidden"}
        phys_spec = get_state_spec(phys_state)
        x = flatten_inputs(
            phys_state, boundary_inputs, dt,
            phys_spec, get_boundary_spec(boundary_inputs),
        )

        output_vec, new_hidden = net(x, hidden)
        result = unflatten_output(output_vec, phys_spec)
        result["_gru_hidden"] = new_hidden
        return result


class SDeepONetDerivative(SurrogateArchitecture):
    """Sequential DeepONet with GRU branch, derivative mode.

    Same as SDeepONetDirect but returns d(state)/dt instead of new state.

    Parameters
    ----------
    n_basis : int
        Number of basis functions. Default 32.
    gru_hidden_size : int
        GRU hidden state size. Default 32.
    proj_hidden : sequence of int
        Post-GRU projection hidden sizes. Default (32,).
    activation : callable
        Activation for projection MLP. Default jax.nn.tanh.
    """

    mode = "derivative"

    def __init__(
        self,
        n_basis: int = 32,
        gru_hidden_size: int = 32,
        proj_hidden: Sequence[int] = (32,),
        activation: Callable = jax.nn.tanh,
    ):
        check_equinox()
        self.n_basis = n_basis
        self.gru_hidden_size = gru_hidden_size
        self.proj_hidden = tuple(proj_hidden)
        self.activation = activation

    def init_params(self, rng_key, state_spec, boundary_spec):
        # Exclude _gru_hidden from physical state for size computation
        phys_spec = {k: v for k, v in state_spec.items() if k != "_gru_hidden"}
        input_size, output_size = compute_sizes(phys_spec, boundary_spec)
        k1, k2, k3, k4 = jax.random.split(rng_key, 4)

        gru_cell = eqx.nn.GRUCell(input_size, self.gru_hidden_size, key=k1)
        branch_proj = eqx.nn.MLP(
            in_size=self.gru_hidden_size,
            out_size=self.n_basis,
            width_size=self.proj_hidden[0] if self.proj_hidden else self.n_basis,
            depth=len(self.proj_hidden),
            activation=self.activation,
            key=k2,
        )
        scale = jnp.sqrt(2.0 / (self.n_basis + output_size))
        trunk_basis = jax.random.normal(k3, (self.n_basis, output_size)) * scale
        output_bias = jnp.zeros(output_size)

        net = _GRUBranchTrunkNet(
            gru_cell=gru_cell,
            branch_proj=branch_proj,
            trunk_basis=trunk_basis,
            output_bias=output_bias,
        )
        return eqx.partition(net, eqx.is_array)

    def hidden_size(self) -> int:
        return self.gru_hidden_size

    def forward(self, params, state, boundary_inputs, dt):
        arrays, static = params
        net = eqx.combine(arrays, static)

        hidden = state["_gru_hidden"]
        phys_state = {k: v for k, v in state.items() if k != "_gru_hidden"}
        phys_spec = get_state_spec(phys_state)
        x = flatten_inputs(
            phys_state, boundary_inputs, dt,
            phys_spec, get_boundary_spec(boundary_inputs),
        )

        output_vec, new_hidden = net(x, hidden)
        result = unflatten_output(output_vec, phys_spec)
        # For derivative mode, _gru_hidden derivative is zero (it's updated directly)
        result["_gru_hidden"] = new_hidden
        return result
