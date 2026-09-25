"""A rod built on ``grid_points`` is held to its own Fourier limit.

The constructor's stability check (MADD-ANO-009) used to run for uniform rods
only.  A rod given ``grid_points`` spelling the very same cell centres at
Fourier number 5, ten times the limit, built without a word and reached inf
by step 32 (audit_040_p4_2, release-record, M3).  MADD-ANO-002 and the
release notes said the constructor refused such rods from 0.4.0.

The non-uniform stencil is the 2nd-order variable-spacing one, whatever
``stencil_order`` says.  Gershgorin on its rows bounds every eigenvalue by
``4 / min(h_L * h_R)``, where ``h_L`` and ``h_R`` are a cell's spacings to
its two neighbours and the ends repeat their end spacing.  So the check is
``dt * alpha / min(h_L * h_R) <= 1/2``.  That is exactly the uniform bound
on a uniform grid, and sufficient on any grid.  These tests hold it to both
properties on the node's own operator.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.heat import (
    MAX_FOURIER_NUMBER,
    HeatNode,
    _nonuniform_fourier_spacing,
)

ALPHA = 0.01


def _graded_grids():
    rng = np.random.default_rng(0)
    return {
        "geometric-1.2": np.cumsum(1.2 ** np.arange(20)) / 100.0,
        "geometric-2.0": np.cumsum(2.0 ** np.arange(10)) / 1000.0,
        "tanh-clustered": 0.5 * (1 + np.tanh(2.5 * np.linspace(-1, 1, 40))
                                 / np.tanh(2.5)),
        "jittered": np.sort((np.arange(30) + 0.5 + rng.uniform(-0.4, 0.4, 30)) / 30),
        "abrupt-10x": np.concatenate([np.arange(10) * 0.01,
                                      0.1 + np.arange(1, 11) * 0.1]),
    }


GRIDS = _graded_grids()


def _criterion_dt(points, alpha=ALPHA):
    return MAX_FOURIER_NUMBER[2] * _nonuniform_fourier_spacing(points) / alpha


def _operator(node):
    """The rod's Laplacian as a matrix, Dirichlet data zero, from the node's
    own ``_compute_laplacian`` (one call per unit vector)."""
    n = node.params["n_cells"]
    rows = jax.vmap(lambda e: node._compute_laplacian(e, 0.0, 0.0))(
        jnp.eye(n, dtype=jnp.float32))
    return np.asarray(rows, dtype=np.float64).T


def test_a_non_uniform_rod_past_its_limit_is_refused():
    """The audit's reproducer: the same 20 cell centres, uniform and as
    ``grid_points``, at Fo = 5.  Both are refused now."""
    n, length = 20, 1.0
    dx = length / n
    dt = 5.0 * dx * dx / ALPHA
    with pytest.raises(ValueError, match="Fourier number dt\\*alpha/dx\\^2 is 5,"):
        HeatNode("uniform", dt, n_cells=n, length=length, thermal_diffusivity=ALPHA)
    centres = np.linspace(dx / 2, length - dx / 2, n)
    with pytest.raises(ValueError, match=r"non-uniform grid .* is 5, above the 0.5 limit"):
        HeatNode("nonuniform", dt, n_cells=n, length=length,
                 thermal_diffusivity=ALPHA, grid_points=centres)


def test_a_uniform_grid_given_as_points_has_the_uniform_limit():
    """On a uniform grid every h_L * h_R is dx**2, so the two checks agree:
    both accept Fo = 0.499 and both refuse 0.501."""
    n, length = 16, 2.0
    dx = length / n
    centres = ((np.arange(n) + 0.5) * dx).tolist()
    assert _nonuniform_fourier_spacing(centres) == pytest.approx(dx * dx, rel=1e-12)
    for fourier, refused in ((0.499, False), (0.501, True)):
        dt = fourier * dx * dx / ALPHA
        for kw in ({}, {"grid_points": centres}):
            if refused:
                with pytest.raises(ValueError, match="unstable"):
                    HeatNode("r", dt, n_cells=n, length=length,
                             thermal_diffusivity=ALPHA, **kw)
            else:
                HeatNode("r", dt, n_cells=n, length=length,
                         thermal_diffusivity=ALPHA, **kw)


@pytest.mark.parametrize("name", sorted(GRIDS))
def test_the_criterion_is_conservative_on_graded_grids(name):
    """At the largest timestep the check accepts, the explicit update's
    spectral radius is at most 1, so the rod is stable.  The criterion is
    also not vacuous: the sharp limit is within 1.5x of it (1.007x on the
    abrupt 10x refinement, 1.42x on the jittered grid)."""
    points = GRIDS[name].tolist()
    dt = _criterion_dt(points)
    node = HeatNode("r", dt, n_cells=len(points), thermal_diffusivity=ALPHA,
                    grid_points=points)
    M = _operator(node)
    eig = np.linalg.eigvals(M)
    assert np.max(np.abs(eig.imag)) <= 1e-4 * np.max(np.abs(eig.real))
    rho = np.max(np.abs(1.0 + dt * ALPHA * eig.real))
    assert rho <= 1.0 + 1e-5, rho
    sharp_dt = 2.0 / (ALPHA * np.max(-eig.real))
    assert 1.0 - 1e-5 <= sharp_dt / dt <= 1.5


def _run(node, state, dt, steps):
    step = jax.jit(lambda s: node.update(s, {"left_temperature": 0.0,
                                             "right_temperature": 0.0}, dt))
    return jax.lax.fori_loop(0, steps, lambda _, s: step(s), state)


def test_a_graded_rod_at_its_limit_decays_and_one_past_the_sharp_limit_diverges():
    """The accepted edge is stable in practice.  The rejected side is real:
    the same rod stepped at 1.05x the *sharp* limit, through a ``dt`` handed
    to ``update()`` (the MADD-ANO-002 route the constructor cannot see),
    leaves float range."""
    points = GRIDS["geometric-1.2"].tolist()
    dt = _criterion_dt(points)
    node = HeatNode("r", dt, n_cells=len(points), thermal_diffusivity=ALPHA,
                    grid_points=points)
    x = np.asarray(points)
    start = {"temperature": jnp.asarray(np.sin(np.pi * (x - x[0]) / (x[-1] - x[0]))
                                        + 0.1 * (-1.0) ** np.arange(len(x)),
                                        dtype=jnp.float32)}
    bounded = np.asarray(_run(node, start, dt, 2000)["temperature"])
    assert np.all(np.isfinite(bounded))
    assert np.max(np.abs(bounded)) <= np.max(np.abs(np.asarray(start["temperature"])))
    sharp_dt = 2.0 / (ALPHA * np.max(-np.linalg.eigvals(_operator(node)).real))
    blown = np.asarray(_run(node, start, 1.05 * sharp_dt, 2000)["temperature"])
    assert not np.all(np.isfinite(blown)) or np.max(np.abs(blown)) > 1e10


@pytest.mark.parametrize("points", [
    [0.0, 0.1, 0.1, 0.6],
    [0.6, 0.4, 0.2, 0.0],
    [0.0, 0.3, 0.2, 0.6],
], ids=["repeated", "decreasing", "folded"])
def test_grid_points_that_are_not_strictly_increasing_are_refused(points):
    """A repeated point divides by zero on the first step, and a decreasing
    one silently gives a wrong Laplacian (the stencil's ``x[i+1] - x[i-1] >
    0`` guard replaces a negative span with 1.0).  Neither has a Fourier
    number."""
    with pytest.raises(ValueError, match="strictly increasing"):
        HeatNode("r", 1e-6, n_cells=4, thermal_diffusivity=ALPHA, grid_points=points)


def test_stencil_order_four_on_points_is_held_to_the_second_order_limit():
    """The non-uniform path runs the 2nd-order stencil whatever
    ``stencil_order`` says, so its limit is 1/2, not 5/16: Fo = 0.45 is
    accepted (a 5/16 check would refuse it) and is stable."""
    points = GRIDS["geometric-1.2"].tolist()
    dt = 0.9 * _criterion_dt(points)
    node = HeatNode("r", dt, n_cells=len(points), thermal_diffusivity=ALPHA,
                    stencil_order=4, grid_points=points)
    eig = np.linalg.eigvals(_operator(node)).real
    assert np.max(np.abs(1.0 + dt * ALPHA * eig)) <= 1.0 + 1e-5
    with pytest.raises(ValueError, match="2nd-order variable-spacing"):
        HeatNode("r", 1.1 * _criterion_dt(points), n_cells=len(points),
                 thermal_diffusivity=ALPHA, stencil_order=4, grid_points=points)


def test_the_criterion_is_the_product_of_neighbouring_spacings():
    """Not the smallest spacing squared, which would refuse stable rods: on
    the jittered grid it is 2.7x stricter than the product."""
    points = GRIDS["jittered"].tolist()
    h = np.diff(points)
    padded = np.concatenate([[h[0]], h, [h[-1]]])
    assert _nonuniform_fourier_spacing(points) == pytest.approx(
        float(np.min(padded[:-1] * padded[1:])), rel=1e-12)
    assert _nonuniform_fourier_spacing(points) > 2.5 * float(np.min(h)) ** 2
