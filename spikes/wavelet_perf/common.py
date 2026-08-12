"""Shared setup for the R2 (adaptive apply) measurements.

Builds the same scaled operator the production node uses:
    Ahat = D^-1 Wn^T A_phys Wn D^-1
matrix-free, via maddening.nodes.adaptive.wavelets.matrixfree.
"""
from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from maddening.nodes.adaptive.wavelets import matrixfree as mf
from maddening.nodes.adaptive.wavelets import operator as op
from maddening.nodes.adaptive.wavelets import precond as pc
from maddening.nodes.adaptive.wavelets import transform as T


def build(n_levels, n_coarse, dim, order=4, kind="laplacian", contrast=1.0,
          mass=1.0, precond="hybrid", seed=0):
    """Return dict with the scaled matvec and all the pieces."""
    side = n_coarse * (2 ** n_levels)
    N = side ** dim
    h = 1.0 / side
    norms = op.column_norms_fast(n_levels, n_coarse, order, dim, h)
    levels = {1: T.levels_1d, 2: T.levels_2d, 3: T.levels_3d}[dim](n_levels, n_coarse)

    if kind == "laplacian":
        a_phys = mf.make_laplacian_apply(side, dim, h, mass=mass)
        a_grid = None
    else:
        # a checkerboard-ish smooth-but-sharp contrast field, like the apps spikes
        coords = np.meshgrid(*([np.arange(side) / side] * dim), indexing="ij")
        r2 = sum((c - 0.5) ** 2 for c in coords)
        blob = (r2 < 0.2 ** 2).astype(np.float64)
        a_np = 1.0 + (contrast - 1.0) * blob
        a_grid = jnp.asarray(a_np.reshape(-1))
        a_phys = mf.make_varcoeff_apply(a_grid, side, dim, h, mass=mass)

    # lagged diagonal at a_ref = 1 (what the node does)
    a_ref = jnp.ones(N, dtype=jnp.float64)
    a_phys_ref = mf.make_varcoeff_apply(a_ref, side, dim, h, mass=mass)
    diag_ref = mf.wave_diagonal_fast(n_levels, n_coarse, order, dim, norms, a_phys_ref)
    D = pc.diagonal_scaling(diag_ref, levels, precond)

    apply = mf.make_wave_apply(n_levels, n_coarse, order, dim, norms, a_phys, D)

    lev_np = np.asarray(levels)
    coarse = jnp.asarray(lev_np == lev_np.min())

    return dict(side=side, N=N, h=h, norms=norms, levels=levels, lev_np=lev_np,
                D=D, apply=apply, a_phys=a_phys, a_grid=a_grid, coarse=coarse,
                n_levels=n_levels, n_coarse=n_coarse, dim=dim, order=order)


def columns(apply, N, batch=64):
    """Yield batches of columns of the operator: (start, cols[b, N])."""
    # build ONLY the batch rows of the identity (never the full N x N)
    f = jax.jit(jax.vmap(lambda j: apply(jax.nn.one_hot(j, N, dtype=jnp.float64))))
    for s in range(0, N, batch):
        e = min(s + batch, N)
        yield s, f(jnp.arange(s, e))
