"""Sensor / objective protocol for the wavelet node.

A sensor maps the wavelet coefficients ``c`` to an *observable* — what the node's
objective is computed from.  The node reads the observable through this protocol
and knows nothing about what it means physically; specific measurement models are
supplied by the caller.  Provided building blocks:

* **point** — a single value ``u(x_sensor)`` (a scalar);
* **multi-point** — a vector of ``u`` at several locations;
* **field functional** — a weighted integral ``Σ_i w_i u(x_i)`` (a scalar);
* **gradient** — components of ``∇u`` at specified locations (a vector per
  location), via :class:`GradientSensor`.

The point / multi-point / field observables are **linear in ``c``** —
``u(x_i) = Wn[i] · c`` — so each is a precomputed matrix ``R`` applied as
``R @ c``, keeping them differentiable and cheap.

The node's ``_sensor`` must stay **scalar** (the blindness diagnostic and the
grad tests differentiate it), so scalar sensors squeeze to 0-d; multi-point /
vector sensors are opt-in and used by objectives that reduce them to a scalar
themselves.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

__all__ = ["Sensor", "LinearSensor", "point_sensor", "multipoint_sensor",
           "field_functional", "GradientSensor"]


@runtime_checkable
class Sensor(Protocol):
    """Maps physical wavelet coefficients ``c`` to an observable."""

    def observe(self, c: jax.Array) -> jax.Array: ...


class LinearSensor:
    """Observable ``R @ c`` for a precomputed row/matrix ``R``.

    ``scalar=True`` squeezes a single-row result to 0-d (the node's ``_sensor``
    contract); ``scalar=False`` returns the full ``(M,)`` vector.
    """

    def __init__(self, R: jax.Array, *, scalar: bool = False):
        self.R = jnp.atleast_2d(R)
        self.scalar = bool(scalar)

    def observe(self, c: jax.Array) -> jax.Array:
        y = self.R @ c
        return jnp.squeeze(y) if self.scalar else y


def point_sensor(Wn: jax.Array, idx: int) -> LinearSensor:
    """Scalar ``φ(x_idx)`` — the default single-probe sensor."""
    return LinearSensor(Wn[idx][None, :], scalar=True)


def multipoint_sensor(Wn: jax.Array, idxs) -> LinearSensor:
    """Vector ``[φ(x_i) for i in idxs]`` — a probe array."""
    return LinearSensor(Wn[jnp.asarray(idxs)], scalar=False)


def field_functional(Wn: jax.Array, weights: jax.Array) -> LinearSensor:
    """Scalar weighted field integral ``Σ_i weights[i] · φ(x_i)``.

    ``weights`` is one value per grid point (same length as a column of ``Wn``);
    e.g. ``+1`` over one region and ``-1`` over another gives a differential
    functional.  Precomputes the single coefficient-space row ``weights @ Wn`` so
    the observable stays an ``R @ c``.
    """
    row = jnp.asarray(weights) @ Wn
    return LinearSensor(row[None, :], scalar=True)


class GradientSensor:
    """Components of ``∇u`` at specified locations — a vector per location.

    The solution field ``u`` is reconstructed from the wavelet coefficients
    matrix-free (``field_from_c = Wn·c``), the periodic central-difference
    :func:`~maddening.nodes.adaptive.wavelets.matrixfree.grid_gradient` gives
    ``∇u``, and the requested components are gathered at the probe indices.  Linear
    in ``c`` and fully differentiable.  Physics-agnostic: a caller that wants a
    signed field (e.g. a negated gradient) applies the sign in its objective.

    ``observe(c)`` returns shape ``(n_probe, n_component)``; an objective reduces
    that to a scalar itself (e.g. a misfit norm).
    """

    def __init__(self, field_from_c, side: int, dim: int, h: float,
                 probe_indices, components=None):
        self.field_from_c = field_from_c
        self.side = int(side)
        self.dim = int(dim)
        self.h = float(h)
        self.probes = jnp.asarray(probe_indices)
        self.components = tuple(range(dim)) if components is None else tuple(components)

    def observe(self, c: jax.Array) -> jax.Array:
        from maddening.nodes.adaptive.wavelets.matrixfree import grid_gradient
        u = self.field_from_c(c)
        grads = grid_gradient(u, self.side, self.dim, self.h)
        cols = [grads[d][self.probes] for d in self.components]
        return jnp.stack(cols, axis=-1)          # (n_probe, n_component)
