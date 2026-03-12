"""
Fourier Neural Operator (FNO) surrogate architectures.

Supports 1D, 2D, and 3D spatial fields via a unified implementation.
The spectral convolution operates in Fourier space: FFT → multiply
low-frequency modes by learnable weights → IFFT.

For MADDENING nodes with mixed scalar + spatial state fields, scalar
fields pass through a small MLP bypass while the spatial field goes
through the FNO layers.
"""

import math
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp

from maddening.surrogates.architecture import SurrogateArchitecture
from maddening.surrogates.architectures._utils import check_equinox

try:
    import equinox as eqx
except ImportError:
    eqx = None


# ------------------------------------------------------------------
# Spectral convolution layers
# ------------------------------------------------------------------

class _SpectralConv(eqx.Module if eqx is not None else object):
    """Spectral convolution in Fourier space (supports 1D/2D/3D).

    Truncates to n_modes low-frequency components, applies learnable
    complex weights, then transforms back.
    """
    weights_real: jnp.ndarray  # real part of complex weights
    weights_imag: jnp.ndarray  # imaginary part
    n_modes: tuple             # modes to keep per spatial dim
    ndim: int                  # spatial dimensionality (1, 2, or 3)

    def __call__(self, x):
        """Apply spectral convolution.

        Parameters
        ----------
        x : array, shape (..., *spatial_dims, channels)
            Input feature map. Last axis is channels, preceding axes
            are spatial dimensions.

        Returns
        -------
        array, same shape as x
        """
        spatial_shape = x.shape[:-1]
        n_channels = x.shape[-1]

        # Move channels to front for easier FFT over spatial dims
        # x: (*spatial, C) -> (C, *spatial)
        axes_perm = (len(spatial_shape),) + tuple(range(len(spatial_shape)))
        x_t = jnp.transpose(x, axes_perm)

        # Forward FFT over spatial dimensions
        fft_axes = tuple(range(1, self.ndim + 1))
        x_ft = jnp.fft.rfftn(x_t, axes=fft_axes)

        # Build index slices for the low-frequency modes
        slices = [slice(None)]  # channels dim
        for i, m in enumerate(self.n_modes):
            if i < self.ndim - 1:
                slices.append(slice(0, m))
            else:
                # Last dim uses rfft so max modes = spatial_size//2 + 1
                slices.append(slice(0, m))

        x_ft_trunc = x_ft[tuple(slices)]

        # Complex weight multiplication: (C_out, C_in, *modes)
        weights = self.weights_real + 1j * self.weights_imag
        # Einstein notation: contract over input channels
        if self.ndim == 1:
            out_ft = jnp.einsum("im,jm->jm", x_ft_trunc, weights)
        elif self.ndim == 2:
            out_ft = jnp.einsum("imn,jmn->jmn", x_ft_trunc, weights)
        else:  # 3D
            out_ft = jnp.einsum("imnp,jmnp->jmnp", x_ft_trunc, weights)

        # Zero-pad back to full frequency shape
        out_full = jnp.zeros_like(x_ft)
        out_full = out_full.at[tuple(slices)].set(out_ft)

        # Inverse FFT
        x_out = jnp.fft.irfftn(out_full, s=spatial_shape, axes=fft_axes)

        # (C, *spatial) -> (*spatial, C)
        inv_perm = tuple(range(1, self.ndim + 1)) + (0,)
        return jnp.transpose(x_out, inv_perm)


class _FNOBlock(eqx.Module if eqx is not None else object):
    """Single FNO layer: spectral conv + pointwise linear + activation."""
    spectral: _SpectralConv
    pointwise_w: jnp.ndarray   # (channels, channels)
    pointwise_b: jnp.ndarray   # (channels,)

    def __call__(self, x, activation):
        # Spectral path
        s = self.spectral(x)
        # Pointwise path (linear transform at each spatial location)
        p = x @ self.pointwise_w + self.pointwise_b
        return activation(s + p)


class _FNONet(eqx.Module if eqx is not None else object):
    """Full FNO: lifting → N blocks → projection."""
    lifting_w: jnp.ndarray      # (in_channels, hidden_channels)
    lifting_b: jnp.ndarray      # (hidden_channels,)
    blocks: list                 # list of _FNOBlock
    proj_w: jnp.ndarray         # (hidden_channels, out_channels)
    proj_b: jnp.ndarray         # (out_channels,)

    def __call__(self, x, activation):
        """Forward pass.

        Parameters
        ----------
        x : array, shape (*spatial_dims, in_channels)
        activation : callable

        Returns
        -------
        array, shape (*spatial_dims, out_channels)
        """
        # Lifting
        h = x @ self.lifting_w + self.lifting_b
        # FNO blocks
        for block in self.blocks:
            h = block(h, activation)
        # Projection
        return h @ self.proj_w + self.proj_b


def _build_fno_net(
    rng_key, in_channels, out_channels, hidden_channels,
    n_modes, n_layers, ndim,
):
    """Build and partition an FNO network."""
    keys = jax.random.split(rng_key, 2 + 3 * n_layers)
    idx = 0

    # Lifting layer
    scale = jnp.sqrt(2.0 / (in_channels + hidden_channels))
    lifting_w = jax.random.normal(keys[idx], (in_channels, hidden_channels)) * scale
    lifting_b = jnp.zeros(hidden_channels)
    idx += 1

    # FNO blocks
    blocks = []
    for _ in range(n_layers):
        # Spectral conv weights: (channels, *n_modes)
        w_shape = (hidden_channels,) + tuple(n_modes)
        scale_w = 1.0 / (hidden_channels * math.prod(n_modes))
        wr = jax.random.normal(keys[idx], w_shape) * scale_w
        wi = jax.random.normal(keys[idx + 1], w_shape) * scale_w
        idx += 2

        spectral = _SpectralConv(
            weights_real=wr,
            weights_imag=wi,
            n_modes=tuple(n_modes),
            ndim=ndim,
        )

        # Pointwise linear
        pw_scale = jnp.sqrt(2.0 / (hidden_channels + hidden_channels))
        pw = jax.random.normal(keys[idx], (hidden_channels, hidden_channels)) * pw_scale
        pb = jnp.zeros(hidden_channels)
        idx += 1

        blocks.append(_FNOBlock(spectral=spectral, pointwise_w=pw, pointwise_b=pb))

    # Projection layer
    scale_p = jnp.sqrt(2.0 / (hidden_channels + out_channels))
    proj_w = jax.random.normal(keys[0], (hidden_channels, out_channels)) * scale_p
    proj_b = jnp.zeros(out_channels)

    net = _FNONet(
        lifting_w=lifting_w,
        lifting_b=lifting_b,
        blocks=blocks,
        proj_w=proj_w,
        proj_b=proj_b,
    )
    return eqx.partition(net, eqx.is_array)


# ------------------------------------------------------------------
# FNO Architecture classes
# ------------------------------------------------------------------

class FNODirect(SurrogateArchitecture):
    """Fourier Neural Operator, direct mode (predicts next state).

    Operates on a designated spatial state field. Remaining scalar
    fields pass through a small MLP bypass.

    Parameters
    ----------
    spatial_field : str
        Name of the state field containing the spatial array
        (e.g. ``"temperature"``).
    n_modes : tuple of int
        Number of Fourier modes per spatial dimension.
        Length determines dimensionality (1D, 2D, or 3D).
    hidden_channels : int
        Feature width of FNO layers. Default 16.
    n_layers : int
        Number of FNO blocks. Default 2.
    activation : callable
        Activation function. Default jax.nn.gelu.
    scalar_hidden : sequence of int
        Hidden sizes for the scalar bypass MLP. Default (16,).
        Only used if there are non-spatial state fields.
    """

    mode = "direct"

    def __init__(
        self,
        spatial_field: str,
        n_modes: tuple,
        hidden_channels: int = 16,
        n_layers: int = 2,
        activation: Callable = jax.nn.gelu,
        scalar_hidden: Sequence[int] = (16,),
    ):
        check_equinox()
        self.spatial_field = spatial_field
        self.n_modes = tuple(n_modes)
        self.ndim = len(n_modes)
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.activation = activation
        self.scalar_hidden = tuple(scalar_hidden)

    def init_params(self, rng_key, state_spec, boundary_spec):
        k1, k2 = jax.random.split(rng_key)

        # FNO params — input is spatial field as 1-channel
        fno_params = _build_fno_net(
            k1, in_channels=1, out_channels=1,
            hidden_channels=self.hidden_channels,
            n_modes=self.n_modes,
            n_layers=self.n_layers,
            ndim=self.ndim,
        )

        # Scalar bypass MLP (if there are non-spatial fields)
        scalar_fields = {k: v for k, v in state_spec.items() if k != self.spatial_field}
        scalar_size = sum(math.prod(s) if s else 1 for s in scalar_fields.values())
        boundary_size = sum(math.prod(s) if s else 1 for s in boundary_spec.values())
        scalar_input = scalar_size + boundary_size + 1  # +1 for dt

        scalar_mlp_params = None
        if scalar_size > 0:
            mlp = eqx.nn.MLP(
                in_size=scalar_input,
                out_size=scalar_size,
                width_size=self.scalar_hidden[0] if self.scalar_hidden else 16,
                depth=len(self.scalar_hidden),
                activation=self.activation,
                key=k2,
            )
            scalar_mlp_params = eqx.partition(mlp, eqx.is_array)

        return (fno_params, scalar_mlp_params)

    def forward(self, params, state, boundary_inputs, dt):
        (fno_arrays, fno_static), scalar_mlp_params = params

        fno_net = eqx.combine(fno_arrays, fno_static)
        activation = self.activation

        # Process spatial field through FNO
        spatial = state[self.spatial_field]
        # Add channel dim: (*spatial_dims,) -> (*spatial_dims, 1)
        spatial_in = spatial[..., None]
        spatial_out = fno_net(spatial_in, activation)
        spatial_out = spatial_out[..., 0]  # remove channel dim

        result = {self.spatial_field: spatial_out}

        # Process scalar fields through bypass MLP
        scalar_fields = {k: v for k, v in sorted(state.items()) if k != self.spatial_field}
        if scalar_fields and scalar_mlp_params is not None:
            s_arrays, s_static = scalar_mlp_params
            scalar_mlp = eqx.combine(s_arrays, s_static)

            parts = []
            for k in sorted(scalar_fields.keys()):
                parts.append(jnp.ravel(scalar_fields[k]))
            for k in sorted(boundary_inputs.keys()):
                parts.append(jnp.ravel(boundary_inputs[k]))
            parts.append(jnp.atleast_1d(jnp.asarray(dt, dtype=jnp.float32)))
            scalar_in = jnp.concatenate(parts)

            scalar_out = scalar_mlp(scalar_in)

            # Unflatten scalar output
            offset = 0
            for k in sorted(scalar_fields.keys()):
                shape = scalar_fields[k].shape
                size = math.prod(shape) if shape else 1
                val = scalar_out[offset:offset + size]
                result[k] = val.reshape(shape) if shape else val.squeeze()
                offset += size

        return result


class FNODerivative(SurrogateArchitecture):
    """Fourier Neural Operator, derivative mode (predicts d(state)/dt).

    Same as FNODirect but returns time derivatives.

    Parameters
    ----------
    spatial_field : str
        Name of the spatial state field.
    n_modes : tuple of int
        Number of Fourier modes per spatial dimension.
    hidden_channels : int
        Feature width. Default 16.
    n_layers : int
        Number of FNO blocks. Default 2.
    activation : callable
        Activation function. Default jax.nn.gelu.
    scalar_hidden : sequence of int
        Hidden sizes for the scalar bypass MLP. Default (16,).
    """

    mode = "derivative"

    def __init__(
        self,
        spatial_field: str,
        n_modes: tuple,
        hidden_channels: int = 16,
        n_layers: int = 2,
        activation: Callable = jax.nn.gelu,
        scalar_hidden: Sequence[int] = (16,),
    ):
        check_equinox()
        self.spatial_field = spatial_field
        self.n_modes = tuple(n_modes)
        self.ndim = len(n_modes)
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.activation = activation
        self.scalar_hidden = tuple(scalar_hidden)

    def init_params(self, rng_key, state_spec, boundary_spec):
        k1, k2 = jax.random.split(rng_key)

        fno_params = _build_fno_net(
            k1, in_channels=1, out_channels=1,
            hidden_channels=self.hidden_channels,
            n_modes=self.n_modes,
            n_layers=self.n_layers,
            ndim=self.ndim,
        )

        scalar_fields = {k: v for k, v in state_spec.items() if k != self.spatial_field}
        scalar_size = sum(math.prod(s) if s else 1 for s in scalar_fields.values())
        boundary_size = sum(math.prod(s) if s else 1 for s in boundary_spec.values())
        scalar_input = scalar_size + boundary_size + 1

        scalar_mlp_params = None
        if scalar_size > 0:
            mlp = eqx.nn.MLP(
                in_size=scalar_input,
                out_size=scalar_size,
                width_size=self.scalar_hidden[0] if self.scalar_hidden else 16,
                depth=len(self.scalar_hidden),
                activation=self.activation,
                key=k2,
            )
            scalar_mlp_params = eqx.partition(mlp, eqx.is_array)

        return (fno_params, scalar_mlp_params)

    def forward(self, params, state, boundary_inputs, dt):
        (fno_arrays, fno_static), scalar_mlp_params = params

        fno_net = eqx.combine(fno_arrays, fno_static)
        activation = self.activation

        spatial = state[self.spatial_field]
        spatial_in = spatial[..., None]
        spatial_out = fno_net(spatial_in, activation)
        spatial_out = spatial_out[..., 0]

        result = {self.spatial_field: spatial_out}

        scalar_fields = {k: v for k, v in sorted(state.items()) if k != self.spatial_field}
        if scalar_fields and scalar_mlp_params is not None:
            s_arrays, s_static = scalar_mlp_params
            scalar_mlp = eqx.combine(s_arrays, s_static)

            parts = []
            for k in sorted(scalar_fields.keys()):
                parts.append(jnp.ravel(scalar_fields[k]))
            for k in sorted(boundary_inputs.keys()):
                parts.append(jnp.ravel(boundary_inputs[k]))
            parts.append(jnp.atleast_1d(jnp.asarray(dt, dtype=jnp.float32)))
            scalar_in = jnp.concatenate(parts)

            scalar_out = scalar_mlp(scalar_in)

            offset = 0
            for k in sorted(scalar_fields.keys()):
                shape = scalar_fields[k].shape
                size = math.prod(shape) if shape else 1
                val = scalar_out[offset:offset + size]
                result[k] = val.reshape(shape) if shape else val.squeeze()
                offset += size

        return result
