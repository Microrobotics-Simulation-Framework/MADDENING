"""Galerkin wavelet operators ``A_wave = Wᵀ A_phys W`` (BCOO + matrix-free).

Production reimplementation of the spike operator construction
(``dd_jax_poc.py`` BCOO assembly, ``discontinuous_coeff.py`` variable
coefficient).  Two paths:

* **Assembled BCOO** (default, validated sizes): materialise the L²-normalised
  synthesis matrix ``Wn`` and the physical FD operator ``A_phys`` (constant- or
  variable-coefficient), form ``A_wave = Wnᵀ A_phys Wn``, sparsify to
  ``jax.experimental.sparse.BCOO``.  The whole assembly is JAX-traceable, so
  ``jax.grad`` of a solve objective flows through ``A_phys(a)`` w.r.t. the
  coefficient field ``a(x)`` (Amendment 1 -- differentiability *through operator
  assembly*, not merely a stencil parameterised by ``a``).

* **Matrix-free matvec** (large-N option): ``A_wave v = Wnᵀ A_phys (Wn v)``
  evaluated via the matrix-free synthesis and its ``jax.linear_transpose``,
  never materialising ``Wn`` or ``A_wave``.  Column L² norms are computed per
  ``(level, subband)`` representative (translation invariance), O(log N).

Channel convention (Amendment 4, optional/forward-compat): the *solution* and
*coefficient* arrays may carry a trailing channel axis ``C`` (C=1 scalar).  The
operator structure (BCOO sparsity, masked solve) is identical per channel; only
the physical operator's coefficient may be ``(N, C, C)``.  M0 ships the scalar
path; the assembly functions accept ``a`` of shape ``(N,)`` (scalar) and the
channel axis is left as a documented extension point, not built out.
"""

from __future__ import annotations

from typing import Callable, Optional

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

from maddening.nodes.adaptive.wavelets import transform as T

__all__ = [
    "physical_laplacian",
    "physical_varcoeff",
    "column_norms",
    "assemble_wave_operator",
    "make_masked_operator",
]


# ----------------------------------------------------------------------
# Physical-space FD operators (dense; used for assembly at validated sizes).
# ----------------------------------------------------------------------

def _lap1d(side: int, h: float, dtype) -> jax.Array:
    """Periodic 1D stiffness (-d²/dx²), circulant tridiagonal, ~1/h scaling."""
    idx = jnp.arange(side)
    S = jnp.zeros((side, side), dtype=dtype)
    S = S.at[idx, idx].set(2.0 / h)
    S = S.at[idx, (idx + 1) % side].add(-1.0 / h)
    S = S.at[idx, (idx - 1) % side].add(-1.0 / h)
    return S


def physical_laplacian(side: int, dim: int, h: float, mass: float = 1.0,
                       dtype=jnp.float64) -> jax.Array:
    """Constant-coefficient H¹ bilinear form (-Δ + mass·I), periodic, dense.

    Built as a tensor sum of 1D stiffness and lumped mass (``M = h·I`` per
    axis), matching the spike's Galerkin convention.
    """
    S = _lap1d(side, h, dtype)
    M = (h) * jnp.eye(side, dtype=dtype)
    if dim == 1:
        return S + mass * M  # lumped mass M = h·I, matches spike laplacian_periodic
    if dim == 2:
        A = jnp.kron(S, M) + jnp.kron(M, S) + (mass) * jnp.kron(M, M)
        return A
    if dim == 3:
        A = (jnp.kron(jnp.kron(S, M), M)
             + jnp.kron(jnp.kron(M, S), M)
             + jnp.kron(jnp.kron(M, M), S)
             + (mass) * jnp.kron(jnp.kron(M, M), M))
        return A
    raise ValueError(f"dim must be 1, 2, or 3; got {dim}")


def physical_varcoeff(a_grid: jax.Array, dim: int, h: float,
                      mass: float = 1.0) -> jax.Array:
    """Conservative variable-coefficient operator -∇·(a∇·) + mass, periodic.

    ``a_grid`` is the coefficient field on the grid (shape ``(side,)``,
    ``(side, side)``, or ``(side, side, side)``).  Face coefficients are the
    average of adjacent cell values.  Fully JAX-traceable in ``a_grid`` so
    ``jax.grad`` w.r.t. the coefficient field flows through assembly.

    Returns a dense ``(N, N)`` matrix (N = side**dim).  Used at validated sizes;
    the matrix-free path is preferred for large grids.
    """
    a = a_grid
    side = a.shape[0]
    N = side ** dim
    aflat = a.reshape(-1)
    # Build via index arithmetic on the flat grid with periodic wrap.
    A = jnp.zeros((N, N), dtype=a.dtype)
    coords = jnp.indices((side,) * dim).reshape(dim, -1)  # (dim, N)

    def flat_index(shifted):
        idx = jnp.zeros(shifted.shape[1], dtype=jnp.int32)
        for d in range(dim):
            idx = idx * side + (shifted[d] % side)
        return idx

    rows = flat_index(coords)
    diag = jnp.full((N,), mass, dtype=a.dtype)
    for d in range(dim):
        for s in (+1, -1):
            nbr = coords.at[d].add(s)
            cols = flat_index(nbr)
            # face coefficient = average of this cell and the neighbour
            a_face = 0.5 * (aflat[rows] + aflat[cols]) / h ** 2
            A = A.at[rows, cols].add(-a_face)
            diag = diag + a_face
    A = A.at[rows, rows].add(diag)
    return 0.5 * (A + A.T)


# ----------------------------------------------------------------------
# Column L² norms of W (for normalisation).
# ----------------------------------------------------------------------

def column_norms(n_levels: int, n_coarse: int, order: int, dim: int,
                 h: float) -> jax.Array:
    """L²(grid) norms of the synthesis columns, ``sqrt(h**dim * Σ W[:,j]²)``.

    Computed exactly from the dense ``W`` (cheap at validated sizes).  The
    matrix-free path uses a per-(level, subband) representative instead.
    """
    W = T.synthesis_matrix(n_levels, n_coarse, order, dim=dim)
    norms = jnp.sqrt((h ** dim) * jnp.sum(W ** 2, axis=0))
    return jnp.where(norms > 0, norms, 1.0)


# ----------------------------------------------------------------------
# Assemble the Galerkin wavelet operator.
# ----------------------------------------------------------------------

def assemble_wave_operator(
    n_levels: int,
    n_coarse: int,
    order: int = 4,
    dim: int = 1,
    *,
    mass: float = 1.0,
    a_grid: Optional[jax.Array] = None,
    sparse_threshold: float = 1e-12,
    dtype=jnp.float64,
):
    """Assemble ``A_wave = Wnᵀ A_phys Wn`` as a dense matrix and a BCOO.

    Parameters
    ----------
    a_grid : optional coefficient field; if ``None`` the constant-coefficient
        ``(-Δ + mass)`` is used, else the conservative ``-∇·(a∇·) + mass``.
    sparse_threshold : entries with ``|A| < threshold * max|A|`` are dropped in
        the BCOO.

    Returns
    -------
    dict with keys ``A_dense``, ``A_bcoo``, ``Wn`` (normalised synthesis),
    ``levels``, ``side``, ``N``, ``h``.
    """
    side = n_coarse * (2 ** n_levels)
    h = 1.0 / side
    N = side ** dim

    W = T.synthesis_matrix(n_levels, n_coarse, order, dim=dim)
    norms = jnp.sqrt((h ** dim) * jnp.sum(W ** 2, axis=0))
    norms = jnp.where(norms > 0, norms, 1.0)
    Wn = W / norms[None, :]

    if a_grid is None:
        A_phys = physical_laplacian(side, dim, h, mass=mass, dtype=dtype)
    else:
        A_phys = physical_varcoeff(a_grid, dim, h, mass=mass)

    A_dense = Wn.T @ A_phys @ Wn
    A_dense = 0.5 * (A_dense + A_dense.T)

    thr = sparse_threshold * jnp.max(jnp.abs(A_dense))
    A_sp = jnp.where(jnp.abs(A_dense) >= thr, A_dense, 0.0)
    A_bcoo = jsparse.BCOO.fromdense(A_sp)

    levels = {1: T.levels_1d, 2: T.levels_2d, 3: T.levels_3d}[dim](n_levels, n_coarse)
    return dict(A_dense=A_dense, A_bcoo=A_bcoo, Wn=Wn, levels=levels,
                side=side, N=N, h=h)


# ----------------------------------------------------------------------
# Masked operator for the frozen inner solve (static shape, BCOO).
# ----------------------------------------------------------------------

def make_masked_operator(A: jax.Array, mask: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """Build the frozen-active-set operator closure ``v -> A_eff v``.

    ``A_eff`` is ``A`` on the active block ``(mask, mask)`` and the identity on
    inactive rows/cols, so the solve returns ``c_k = 0`` outside the mask.
    Mirrors ``hierarchical_hat.py``: assemble as BCOO and return a matvec for
    ``ift_linear_solve``.  Static shape ``(N, N)`` -- no dynamic submatrix
    indexing (the spike used dynamically-sized ``np.linalg.solve``; this is the
    production static-shape path).
    """
    N = A.shape[0]
    outer = mask[:, None] & mask[None, :]
    A_eff = jnp.where(outer, A, jnp.eye(N, dtype=A.dtype))
    A_bcoo = jsparse.BCOO.fromdense(A_eff)

    def operator_fn(v: jax.Array) -> jax.Array:
        return A_bcoo @ v

    return operator_fn
