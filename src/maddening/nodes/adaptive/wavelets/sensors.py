"""Sensor / objective protocol for the wavelet node.

A sensor maps the physical wavelet coefficients ``c`` to an *observable* — what
the node's objective is computed from.  The default is a single point value
``φ(x_sensor)`` (a scalar), which is all the spike node exposed.  Real
applications need more:

* **multi-point** — a vector of ``φ`` at several probe locations (e.g. a
  magnetometer / Hall-sensor array, application 1);
* **field functional** — a weighted integral ``Σ_i w_i φ(x_i)`` (e.g. dose over a
  target region minus dose over healthy tissue, application 2);
* **vector / gradient** — components of ``∇φ`` at probes (Hall probes measure
  ``B = -∇φ``); the gradient operator lands with M21, and plugs in here as
  another :class:`LinearSensor` once the derivative rows are available.

Every observable here is **linear in ``c``** — ``φ(x_i) = Wn[i] · c`` — so each
sensor is just a precomputed matrix ``R`` (rows assembled from the synthesis
matrix) applied as ``R @ c``.  This keeps the sensor differentiable and cheap,
and lets point / multi-point / field all share one implementation.

The node's ``_sensor`` must stay **scalar** (the blindness diagnostic and the
grad tests differentiate it), so scalar sensors squeeze to 0-d; multi-point /
vector sensors are opt-in and used by application objectives that reduce them to
a scalar themselves.
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
    e.g. ``+1`` over a target region and ``-1`` over healthy tissue gives a
    differential-dose objective.  Precomputes the single coefficient-space row
    ``weights @ Wn`` so the observable stays an ``R @ c``.
    """
    row = jnp.asarray(weights) @ Wn
    return LinearSensor(row[None, :], scalar=True)


class GradientSensor:
    """Hall-probe sensor: the field perturbation ``B = -∇φ`` at probe points.

    Hall magnetometers measure a magnetic field, ``B = -∇φ`` for the scalar
    potential φ — a *vector* per probe, not a point value.  φ is reconstructed
    from the wavelet coefficients matrix-free (``phi_from_c = Wn·c``), the
    periodic central-difference :func:`~maddening.nodes.adaptive.wavelets.\
matrixfree.grid_gradient` gives ∇φ, and the requested components are gathered at
    the probe indices.  Linear in ``c`` and fully differentiable, so an inverse
    objective ``‖B(χ) − B_measured‖²`` differentiates w.r.t. χ.

    ``observe(c)`` returns shape ``(n_probe, n_component)``.  An application
    objective reduces that to a scalar itself (e.g. a misfit norm).
    """

    def __init__(self, phi_from_c, side: int, dim: int, h: float,
                 probe_indices, components=None):
        self.phi_from_c = phi_from_c
        self.side = int(side)
        self.dim = int(dim)
        self.h = float(h)
        self.probes = jnp.asarray(probe_indices)
        self.components = tuple(range(dim)) if components is None else tuple(components)

    def observe(self, c: jax.Array) -> jax.Array:
        from maddening.nodes.adaptive.wavelets.matrixfree import grid_gradient
        phi = self.phi_from_c(c)
        grads = grid_gradient(phi, self.side, self.dim, self.h)
        # B = -∇φ, gathered at the probes, stacked over requested components
        cols = [(-grads[d])[self.probes] for d in self.components]
        return jnp.stack(cols, axis=-1)          # (n_probe, n_component)
