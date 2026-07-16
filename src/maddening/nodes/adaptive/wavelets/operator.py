"""Galerkin wavelet operators ``A_wave = Wᵀ A_phys W`` (dense assembly + BCOO).

Production reimplementation of the spike operator construction
(``dd_jax_poc.py`` BCOO assembly, ``discontinuous_coeff.py`` variable
coefficient).

**Assembled BCOO** (the only path): materialise the L²-normalised synthesis
matrix ``Wn`` and the physical FD operator ``A_phys`` (constant- or
variable-coefficient), form ``A_wave = Wnᵀ A_phys Wn``, sparsify to
``jax.experimental.sparse.BCOO``.  The whole assembly is JAX-traceable, so
``jax.grad`` of a solve objective flows through ``A_phys(a)`` w.r.t. the
coefficient field ``a(x)`` (Amendment 1 -- differentiability *through operator
assembly*, not merely a stencil parameterised by ``a``).

.. warning::
   **Assembly is dense and this is what bounds the problem size.**  Both
   ``Wn`` and ``A_phys`` are materialised as ``(N, N)`` arrays with
   ``N = side**dim`` before the BCOO sparsification, so peak memory is
   O(N²) regardless of how sparse ``A_wave`` ends up.  In practice this caps
   3D at the 8³-16³ the test suite exercises (16³ = 4096 ⇒ a 4096² dense
   operator) and 2D at ~64².  Only the *solve* is sparse/adaptive
   (:func:`gather_solve` is O(K³) in the active-set size K).

   A genuinely matrix-free matvec — ``A_wave v = Wnᵀ A_phys (Wn v)`` via the
   matrix-free synthesis in :mod:`~maddening.nodes.adaptive.wavelets.transform`
   and its ``jax.linear_transpose``, with column L² norms taken per
   ``(level, subband)`` representative in O(log N) — is the designed route past
   those sizes.  **It is not implemented.**  Do not size a problem on the
   assumption that it exists.

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
import numpy as np

from maddening.nodes.adaptive.wavelets import transform as T

__all__ = [
    "physical_laplacian",
    "physical_varcoeff",
    "column_norms",
    "column_norms_fast",
    "assemble_wave_dense",
    "assemble_wave_operator",
    "sparsity_pattern",
    "bcoo_with_traced_data",
    "make_masked_operator",
    "gather_solve",
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


def _lap1d_dirichlet(side: int, h: float, dtype) -> jax.Array:
    """Non-periodic 1D stiffness (-d²/dx²) on ``side`` interior nodes."""
    idx = jnp.arange(side)
    S = jnp.zeros((side, side), dtype=dtype)
    S = S.at[idx, idx].set(2.0 / h)
    S = S.at[idx[:-1], idx[:-1] + 1].set(-1.0 / h)
    S = S.at[idx[1:], idx[1:] - 1].set(-1.0 / h)
    return S


def physical_laplacian_dirichlet(side: int, dim: int, h: float,
                                 mass: float = 1.0, dtype=jnp.float64) -> jax.Array:
    """Constant-coefficient (-Δ + mass·I) with homogeneous Dirichlet BCs, dense.

    Tensor sum of the 1D Dirichlet stiffness and lumped mass ``M = h·I``.
    """
    S = _lap1d_dirichlet(side, h, dtype)
    M = h * jnp.eye(side, dtype=dtype)
    if dim == 1:
        return S + mass * M
    if dim == 2:
        return jnp.kron(S, M) + jnp.kron(M, S) + mass * jnp.kron(M, M)
    if dim == 3:
        return (jnp.kron(jnp.kron(S, M), M) + jnp.kron(jnp.kron(M, S), M)
                + jnp.kron(jnp.kron(M, M), S) + mass * jnp.kron(jnp.kron(M, M), M))
    raise ValueError(f"dim must be 1, 2, or 3; got {dim}")


def _d2_1d_periodic(side: int, h: float, dtype) -> jax.Array:
    """Periodic 1D second difference (~1/h²), circulant [1, -2, 1]/h²."""
    idx = jnp.arange(side)
    D2 = jnp.zeros((side, side), dtype=dtype)
    D2 = D2.at[idx, idx].set(-2.0 / h ** 2)
    D2 = D2.at[idx, (idx + 1) % side].add(1.0 / h ** 2)
    D2 = D2.at[idx, (idx - 1) % side].add(1.0 / h ** 2)
    return D2


def physical_biharmonic(side: int, dim: int, h: float, mass: float = 1.0,
                        dtype=jnp.float64) -> jax.Array:
    """H²-elliptic biharmonic form ``∫(Δu)² + mass·u²``, periodic, dense.

    ``B = Lapᵀ M Lap + mass·M`` with ``Lap`` the dim-D periodic 2nd-difference
    Laplacian and ``M = h^dim·I`` (lumped mass) -- the stream-function operator
    (t=2).  Needs DD order ≥ 4 (H² Riesz basis); FINDINGS §5: DD-4 t=2 κ≈8.6e3,
    Jacobi ≈1.1e3.
    """
    D2 = _d2_1d_periodic(side, h, dtype)
    I = jnp.eye(side, dtype=dtype)
    if dim == 1:
        Lap = D2
    elif dim == 2:
        Lap = jnp.kron(D2, I) + jnp.kron(I, D2)
    elif dim == 3:
        Lap = (jnp.kron(jnp.kron(D2, I), I) + jnp.kron(jnp.kron(I, D2), I)
               + jnp.kron(jnp.kron(I, I), D2))
    else:
        raise ValueError(f"dim must be 1, 2, or 3; got {dim}")
    M = (h ** dim) * jnp.eye(side ** dim, dtype=dtype)
    B = Lap.T @ M @ Lap + mass * M
    return 0.5 * (B + B.T)


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

    Computed exactly from the dense ``W``, which is materialised here — cheap
    at the validated sizes, O(N²) in memory beyond them (see the module
    docstring).  A per-(level, subband) representative would give the same
    norms in O(log N) by translation invariance, but that path is not
    implemented.
    """
    W = T.synthesis_matrix(n_levels, n_coarse, order, dim=dim)
    norms = jnp.sqrt((h ** dim) * jnp.sum(W ** 2, axis=0))
    return jnp.where(norms > 0, norms, 1.0)


def column_norms_fast(n_levels: int, n_coarse: int, order: int, dim: int,
                      h: float) -> jax.Array:
    """L²(grid) column norms in **O(N log N)** — no dense ``W``.

    By periodic translation invariance every synthesis column within one
    structural block shares its norm, so evaluating one representative column per
    block (``1 + n_levels·n_subband`` of them — O(log N)) and scattering by block
    id reproduces :func:`column_norms` exactly.  Keyed on
    :func:`~maddening.nodes.adaptive.wavelets.transform.structural_blocks`, **not**
    ``levels_*()`` (derisk D4: that key conflates the coarse block with the first
    detail level and is wrong by O(0.15)).

    This is the matrix-free replacement for :func:`column_norms`; it removes the
    only remaining ``O(N²)`` dependence on the normalisation path.
    """
    synth = T._SYNTH[dim]
    N = T.n_dofs(n_levels, n_coarse, dim)
    block_ids_np, reps_np = T.structural_blocks(n_levels, n_coarse, dim)
    reps = jnp.asarray(reps_np)
    # one basis vector per block representative; synthesise each (O(log N) × O(N))
    eye_reps = jax.nn.one_hot(reps, N, dtype=jnp.float64)
    cols = jax.vmap(lambda e: synth(e, n_levels, n_coarse, order))(eye_reps)
    block_norms = jnp.sqrt((h ** dim) * jnp.sum(cols ** 2, axis=1))
    norms = block_norms[jnp.asarray(block_ids_np)]
    return jnp.where(norms > 0, norms, 1.0)


# ----------------------------------------------------------------------
# Assemble the Galerkin wavelet operator.
# ----------------------------------------------------------------------

def assemble_wave_dense(
    n_levels: int,
    n_coarse: int,
    order: int = 4,
    dim: int = 1,
    *,
    mass: float = 1.0,
    a_grid: Optional[jax.Array] = None,
    boundary: str = "periodic",
    kind: str = "laplacian",
    dtype=jnp.float64,
):
    """Assemble the **dense** ``A_wave = Wnᵀ A_phys Wn`` (no BCOO).

    Fully JAX-traceable -- in particular ``jax.grad`` w.r.t. ``a_grid`` flows
    through this assembly (Amendment 1).  Safe under ``jax.jit`` (unlike
    :func:`assemble_wave_operator`, whose ``BCOO.fromdense`` needs a static
    ``nse`` and so must be assembled eagerly).  Use this for the
    variable-coefficient / coefficient-gradient path.

    ``boundary`` is ``"periodic"`` (matrix-free isotropic Mallat basis) or
    ``"dirichlet"`` (boundary-adapted DD basis, dense; tensor product in
    multi-D -- see :mod:`.dirichlet`).

    Returns dict with ``A_dense``, ``Wn``, ``levels``, ``side``, ``N``, ``h``.
    """
    if boundary == "periodic":
        side = n_coarse * (2 ** n_levels)
        h = 1.0 / side
        W = T.synthesis_matrix(n_levels, n_coarse, order, dim=dim)
        levels = {1: T.levels_1d, 2: T.levels_2d, 3: T.levels_3d}[dim](
            n_levels, n_coarse)
        if kind == "biharmonic":
            if a_grid is not None:
                raise NotImplementedError("variable-coefficient biharmonic not supported")
            A_phys = physical_biharmonic(side, dim, h, mass=mass, dtype=dtype)
        elif a_grid is None:
            A_phys = physical_laplacian(side, dim, h, mass=mass, dtype=dtype)
        else:
            A_phys = physical_varcoeff(a_grid, dim, h, mass=mass)
    elif boundary == "dirichlet":
        from maddening.nodes.adaptive.wavelets import dirichlet as _dir
        if a_grid is not None:
            raise NotImplementedError(
                "variable-coefficient Dirichlet assembly is not yet supported; "
                "use periodic for coefficient-gradient work")
        side = _dir.dirichlet_side(n_levels, n_coarse)
        h = 1.0 / (side + 1)
        W, levels, _ = _dir.synthesis_matrix_dirichlet(n_levels, n_coarse,
                                                       order, dim)
        A_phys = physical_laplacian_dirichlet(side, dim, h, mass=mass, dtype=dtype)
    else:
        raise ValueError(f"boundary must be 'periodic' or 'dirichlet'; got {boundary!r}")

    N = side ** dim
    norms = jnp.sqrt((h ** dim) * jnp.sum(W ** 2, axis=0))
    norms = jnp.where(norms > 0, norms, 1.0)
    Wn = W / norms[None, :]
    A_dense = Wn.T @ A_phys @ Wn
    A_dense = 0.5 * (A_dense + A_dense.T)
    return dict(A_dense=A_dense, Wn=Wn, levels=levels, side=side, N=N, h=h)


def assemble_wave_operator(
    n_levels: int,
    n_coarse: int,
    order: int = 4,
    dim: int = 1,
    *,
    mass: float = 1.0,
    a_grid: Optional[jax.Array] = None,
    boundary: str = "periodic",
    kind: str = "laplacian",
    sparse_threshold: float = 1e-12,
    dtype=jnp.float64,
):
    """Assemble ``A_wave`` as a dense matrix **and** a BCOO (eager only).

    Adds the sparse ``A_bcoo`` to :func:`assemble_wave_dense`.  ``BCOO.fromdense``
    requires a static ``nse``, so this must run eagerly (at node construction);
    for a traced (e.g. ``jax.jit``/coefficient-gradient) path use
    :func:`assemble_wave_dense` plus :func:`bcoo_with_traced_data`.

    Returns dict with keys ``A_dense``, ``A_bcoo``, ``Wn``, ``levels``,
    ``side``, ``N``, ``h``.
    """
    res = assemble_wave_dense(n_levels, n_coarse, order, dim, mass=mass,
                              a_grid=a_grid, boundary=boundary, kind=kind,
                              dtype=dtype)
    A_dense = res["A_dense"]
    thr = sparse_threshold * jnp.max(jnp.abs(A_dense))
    A_sp = jnp.where(jnp.abs(A_dense) >= thr, A_dense, 0.0)
    res["A_bcoo"] = jsparse.BCOO.fromdense(A_sp)
    return res


def sparsity_pattern(A_dense: jax.Array, threshold: float = 1e-12):
    """Static ``(rows, cols)`` index arrays of the structural nonzeros of ``A``.

    The wavelet operator's sparsity pattern is **independent of the coefficient
    field** ``a(x)`` (``a`` scales entries; the stencil support is fixed), so
    this pattern -- computed once from a reference operator -- is reused as the
    static BCOO structure across all ``a``.  Returns concrete numpy-backed index
    arrays (call eagerly).
    """
    A = np.asarray(A_dense)
    thr = threshold * np.max(np.abs(A))
    rows, cols = np.nonzero(np.abs(A) >= thr)
    return jnp.asarray(rows), jnp.asarray(cols)


def bcoo_with_traced_data(A_dense: jax.Array, rows: jax.Array, cols: jax.Array):
    """Build a BCOO with **static indices** and **traced data**.

    Separates structural assembly (the ``(rows, cols)`` pattern, fixed) from the
    value update (gathered from the traced ``A_dense(a)``), per the
    variable-coefficient design: the sparsity structure is static while the
    ``.data`` values remain a differentiable function of the coefficient field,
    so ``jax.grad`` w.r.t. ``a(x)`` flows through.  JIT-safe.
    """
    N = A_dense.shape[0]
    data = A_dense[rows, cols]                       # traced gather -> dJ/da flows
    indices = jnp.stack([rows, cols], axis=1)
    return jsparse.BCOO((data, indices), shape=(N, N))


# ----------------------------------------------------------------------
# Masked operator for the frozen inner solve (static shape, BCOO).
# ----------------------------------------------------------------------

def gather_solve(A: jax.Array, mask: jax.Array, rhs: jax.Array,
                 buf: int) -> jax.Array:
    """Frozen active-set solve via gather to a fixed-size buffer + dense K×K.

    Instead of an O(N) iterative solve on the full masked operator, gather the
    ``buf`` highest-priority active DOFs into a ``buf×buf`` dense system, solve
    it directly, and scatter the result back to the full ``(N,)`` vector.  This
    realises the adaptivity speedup (O(buf³) ≪ O(N·iters)) while staying
    static-shape / JIT-safe and differentiable (gather, dense solve, scatter are
    all differentiable; the discrete index selection is non-differentiable, like
    the mask).  Requires ``|mask| ≤ buf``.

    ``A`` is the (scaled) dense operator; ``rhs`` the (scaled) right-hand side.
    Inactive buffer slots are set to identity rows so they return 0.
    """
    N = A.shape[0]
    # active DOFs first (argsort puts mask=True before False), take a fixed buf
    ix = jnp.argsort(jnp.logical_not(mask))[:buf]
    active = mask[ix]                                  # (buf,) real-vs-pad
    Asub = A[jnp.ix_(ix, ix)]                          # (buf, buf)
    keep = active[:, None] & active[None, :]
    Asub = jnp.where(keep, Asub, jnp.eye(buf, dtype=A.dtype))
    bsub = jnp.where(active, rhs[ix], 0.0)
    csub = jnp.linalg.solve(Asub, bsub)
    csub = jnp.where(active, csub, 0.0)
    return jnp.zeros(N, dtype=A.dtype).at[ix].set(csub)


def make_masked_operator(A, mask: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """Build the frozen-active-set operator closure ``v -> A_eff v``.

    ``A_eff`` acts as ``A`` on the active block ``(mask, mask)`` and as the
    identity on inactive rows/cols, so the frozen solve returns ``c_k = 0``
    outside the mask::

        A_eff v = where(mask, A (where(mask, v, 0)), v)

    This is **jit-safe and static-shape**: ``A`` is a *pre-assembled constant*
    (dense matrix or ``BCOO`` -- the latter for an O(nnz) matvec), and the mask
    is applied with ``jnp.where`` (the round-6 masked-matvec closure, spike
    Inv 2).  Crucially it does **not** call ``BCOO.fromdense`` on a traced array
    per call -- that pattern (used by the ``hierarchical_hat`` toy) only works
    eagerly and raises under ``jax.jit`` because ``nse`` is not static.  The
    active block inherits ``A``'s symmetry/PSD, so CG remains valid.
    """
    def operator_fn(v: jax.Array) -> jax.Array:
        vm = jnp.where(mask, v, 0.0)
        Av = A @ vm
        return jnp.where(mask, Av, v)

    return operator_fn
