"""Galerkin wavelet operators ``A = Wn^T A_phys Wn`` and the frozen solves on them.

``A_phys`` is the second-order central-difference discretisation of
``-Laplacian + mass`` (lumped mass ``h**dim``) on the uniform grid,
periodic or with homogeneous Dirichlet walls; ``Wn`` is the synthesis
matrix of :mod:`.transform` (or :mod:`.dirichlet`) with L2-normalised
columns.  The change of basis is exact, so a solve on the **full** wavelet
basis reproduces the finite-difference solution to round-off and the
scheme's order of accuracy is that of the stencil: **two**.  Adaptivity
adds a truncation error controlled by the active-set budget, not by the
grid spacing.

Two frozen-active-set solves are provided, both differentiable:

:func:`gather_solve`
    Gather the active functions into a fixed ``buf x buf`` dense block
    and solve it directly -- ``O(buf**3)``, the solve that realises the
    adaptivity speed-up.  Requires ``|mask| <= buf``, which
    :func:`~maddening.nodes.adaptive.wavelets.cdd.cdd_select` guarantees
    for ``buf = K``.
:func:`make_masked_operator`
    The full-size operator that is ``A`` on the active block and the
    identity elsewhere, for
    :func:`~maddening.core.solver_utils.ift_linear_solve`.  Valid for
    any mask, ``O(nnz)`` per matvec on the sparse operator.

Assembly is eager and dense: the node validates sizes up to ``256`` in
1-D, ``64**2`` in 2-D and ``16**3`` in 3-D, where a dense ``N x N``
operator is cheap; a matrix-free assembly is the extension point for
anything larger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.nodes.adaptive.wavelets import dirichlet as _dir
from maddening.nodes.adaptive.wavelets import transform as _tr

__all__ = [
    "BOUNDARIES",
    "WaveletOperator",
    "assemble_operator",
    "gather_solve",
    "make_masked_operator",
]

#: The accepted ``boundary`` values.
BOUNDARIES: tuple[str, ...] = ("periodic", "dirichlet")


# ---------------------------------------------------------------------------
# Physical-space finite-difference operators (dense)
# ---------------------------------------------------------------------------

def _stiffness_periodic(side: int, h: float, dtype: Any) -> jax.Array:
    """Periodic 1-D stiffness of ``-d2/dx2`` (circulant ``[-1, 2, -1] / h``)."""
    idx = jnp.arange(side)
    S = jnp.zeros((side, side), dtype=dtype)
    S = S.at[idx, idx].set(2.0 / h)
    S = S.at[idx, (idx + 1) % side].add(-1.0 / h)
    S = S.at[idx, (idx - 1) % side].add(-1.0 / h)
    return S


def _stiffness_dirichlet(side: int, h: float, dtype: Any) -> jax.Array:
    """1-D stiffness of ``-d2/dx2`` on ``side`` interior nodes, zero at the walls."""
    idx = jnp.arange(side)
    S = jnp.zeros((side, side), dtype=dtype)
    S = S.at[idx, idx].set(2.0 / h)
    S = S.at[idx[:-1], idx[:-1] + 1].set(-1.0 / h)
    S = S.at[idx[1:], idx[1:] - 1].set(-1.0 / h)
    return S


def _tensor_sum(S: jax.Array, M: jax.Array, dim: int, mass: float) -> jax.Array:
    """``sum_axes kron(M, ..., S, ..., M) + mass * kron(M, ..., M)``."""
    if dim == 1:
        return S + mass * M
    factors = [[M] * dim for _ in range(dim)]
    for ax in range(dim):
        factors[ax][ax] = S
    out = None
    for f in factors:
        term = f[0]
        for g in f[1:]:
            term = jnp.kron(term, g)
        out = term if out is None else out + term
    mass_term = M
    for _ in range(dim - 1):
        mass_term = jnp.kron(mass_term, M)
    assert out is not None
    return out + mass * mass_term


def _physical_operator(side: int, dim: int, h: float, mass: float,
                       boundary: str, dtype: Any) -> jax.Array:
    """Dense ``(-Laplacian + mass)`` bilinear form with lumped mass ``M = h I`` per axis."""
    if boundary == "periodic":
        S = _stiffness_periodic(side, h, dtype)
    else:
        S = _stiffness_dirichlet(side, h, dtype)
    M = h * jnp.eye(side, dtype=dtype)
    return _tensor_sum(S, M, dim, mass)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

@stability(StabilityLevel.EXPERIMENTAL)
@dataclass(frozen=True)
class WaveletOperator:
    """The assembled Galerkin operator and the basis it was built in.

    Attributes
    ----------
    A : jax.Array
        Dense ``(n, n)`` operator ``Wn^T A_phys Wn``, symmetrised.
    A_sparse : BCOO
        The same operator as a ``jax.experimental.sparse.BCOO`` with
        entries below ``sparse_threshold * max|A|`` dropped, for an
        ``O(nnz)`` matvec on the selection and Krylov paths.
    Wn : jax.Array
        L2-normalised synthesis matrix ``(n, n)``; ``u = Wn @ c``.
    levels : jax.Array
        ``int32`` level label per basis function.
    side : int
        Grid points per axis.
    n : int
        ``side ** dim``.
    h : float
        Grid spacing: ``1 / side`` (periodic) or ``1 / (side + 1)`` (Dirichlet).
    dim : int
    boundary : str
    """

    A: jax.Array
    A_sparse: Any
    Wn: jax.Array
    levels: jax.Array
    side: int
    n: int
    h: float
    dim: int
    boundary: str


@stability(StabilityLevel.EXPERIMENTAL)
def assemble_operator(
    n_levels: int,
    n_coarse: int,
    *,
    order: int = 4,
    dim: int = 1,
    mass: float = 1.0,
    boundary: str = "periodic",
    dtype: Any = None,
    sparse_threshold: float = 1e-12,
) -> WaveletOperator:
    """Assemble ``A = Wn^T A_phys Wn`` for ``(-Laplacian + mass)`` on the unit cube.

    Eager only: ``BCOO.fromdense`` needs a concrete matrix, so this runs
    at node construction, never inside a trace.

    Parameters
    ----------
    n_levels, n_coarse : int
        Refinements and coarse points per axis.
    order : {2, 4, 6}
        Interpolating order of the basis.
    dim : {1, 2, 3}
    mass : float
        Coefficient of the zeroth-order term; must be positive for the
        periodic operator to be definite.
    boundary : {"periodic", "dirichlet"}
    dtype : optional
        Floating dtype of every array; JAX's canonical float when omitted.
    sparse_threshold : float
        Relative magnitude below which an entry is dropped from
        :attr:`WaveletOperator.A_sparse` (never from ``A``).

    Returns
    -------
    WaveletOperator
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")
    dt = jnp.zeros((), dtype=dtype).dtype
    if boundary == "periodic":
        side = _tr.side_length(n_levels, n_coarse)
        h = 1.0 / side
        W = _tr.synthesis_matrix(n_levels, n_coarse, order=order, dim=dim, dtype=dt)
        levels = _tr.level_labels(n_levels, n_coarse, dim)
    else:
        W, levels, side = _dir.synthesis_matrix_dirichlet(
            n_levels, n_coarse, order=order, dim=dim, dtype=dt,
        )
        h = 1.0 / (side + 1)
    n = side ** dim
    A_phys = _physical_operator(side, dim, h, float(mass), boundary, dt)
    norms = jnp.sqrt((h ** dim) * jnp.sum(W ** 2, axis=0))
    norms = jnp.where(norms > 0, norms, 1.0)
    Wn = W / norms[None, :]
    A = Wn.T @ A_phys @ Wn
    A = 0.5 * (A + A.T)
    thr = sparse_threshold * jnp.max(jnp.abs(A))
    A_sparse = jsparse.BCOO.fromdense(jnp.where(jnp.abs(A) >= thr, A, 0.0))
    return WaveletOperator(
        A=A, A_sparse=A_sparse, Wn=Wn, levels=levels, side=int(side), n=int(n),
        h=float(h), dim=int(dim), boundary=str(boundary),
    )


# ---------------------------------------------------------------------------
# Frozen solves
# ---------------------------------------------------------------------------

@stability(StabilityLevel.EXPERIMENTAL)
def gather_solve(A: jax.Array, mask: jax.Array, rhs: jax.Array, buf: int) -> jax.Array:
    """Frozen-active-set solve by gathering the active block into ``buf x buf``.

    The active indices are moved to the front of a fixed-size buffer
    (``argsort`` on the negated mask), the corresponding block of ``A``
    and ``rhs`` is gathered, unused buffer rows are set to identity rows
    with a zero right-hand side, the dense block is solved and the result
    is scattered back to a full ``(N,)`` vector that is exactly zero off
    the mask.  Gather, dense solve and scatter are all differentiable, so
    ``jax.grad`` through this is the frozen-set adjoint; the discrete
    index selection is not, exactly as the mask is not.

    Parameters
    ----------
    A : jax.Array
        Dense ``(N, N)`` operator (the node passes the preconditioned one).
    mask : jax.Array
        Boolean ``(N,)`` active set with **at most** ``buf`` entries set.
        A mask with more than ``buf`` active entries is silently
        truncated -- this function cannot tell under a trace -- which is
        why the node caps its selection at ``buf`` and overrides the
        base class's all-true full-basis gradient with a dense solve.
    rhs : jax.Array
        Right-hand side ``(N,)``.
    buf : int
        Static buffer size.

    Returns
    -------
    jax.Array
        Solution ``(N,)``, zero off the mask.
    """
    n = A.shape[0]
    ix = jnp.argsort(jnp.logical_not(mask))[:buf]      # active first
    active = mask[ix]                                  # (buf,) real-vs-padding
    Asub = A[jnp.ix_(ix, ix)]                          # (buf, buf)
    keep = active[:, None] & active[None, :]
    Asub = jnp.where(keep, Asub, jnp.eye(buf, dtype=A.dtype))
    bsub = jnp.where(active, rhs[ix], 0.0)
    csub = jnp.linalg.solve(Asub, bsub)
    csub = jnp.where(active, csub, 0.0)
    return jnp.zeros(n, dtype=A.dtype).at[ix].set(csub)


@stability(StabilityLevel.EXPERIMENTAL)
def make_masked_operator(A: Any, mask: jax.Array) -> Callable[[jax.Array], jax.Array]:
    """The full-size frozen operator ``v -> where(mask, A where(mask, v, 0), v)``.

    Acts as ``A`` on the active block and as the identity on inactive
    rows and columns, so a solve with a right-hand side that is zero off
    the mask returns coefficients that are zero there too.  ``A`` may be
    dense or a pre-assembled ``BCOO``; the mask is applied with
    ``jnp.where``, so the closure is JIT-safe with a static shape and
    never re-sparsifies under a trace.  The active block inherits ``A``'s
    symmetry and definiteness, so ``solver="cg"`` remains valid.
    """
    def operator_fn(v: jax.Array) -> jax.Array:
        vm = jnp.where(mask, v, 0.0)
        return jnp.where(mask, A @ vm, v)

    return operator_fn
