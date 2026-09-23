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
    for ``buf = K`` from a seed that fits; a mask that does not fit
    poisons the gathered block with ``NaN`` rather than silently
    dropping the excess (a jitted function cannot raise).
:func:`make_masked_operator`
    The full-size operator that is ``A`` on the active block and the
    identity elsewhere, for
    :func:`~maddening.core.solver_utils.ift_linear_solve`.  Valid for
    any mask, ``O(nnz)`` per matvec on the sparse operator.

Assembly is host-side (float64 NumPy) and dense: the node validates
sizes up to ``256`` in 1-D, ``64**2`` in 2-D and ``16**3`` in 3-D, where
a dense ``N x N`` operator is cheap; a matrix-free assembly is the
extension point for anything larger.  Every input to the assembly is a
static setting, so it runs under ``jax.ensure_compile_time_eval`` and a
node may be constructed inside a ``jax.jit`` trace (a ``residual_fn``
that builds a fresh graph per call, say).  The assembled ``A`` is
checked for symmetry *before* it is symmetrised: a non-symmetric
physical stencil is refused, not repaired (:data:`SYMMETRY_TOL`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.nodes.adaptive.wavelets import dirichlet as _dir
from maddening.nodes.adaptive.wavelets import precond as _pc
from maddening.nodes.adaptive.wavelets import transform as _tr

__all__ = [
    "BOUNDARIES",
    "LANCZOS_STEPS",
    "SYMMETRY_TOL",
    "WaveletOperator",
    "assemble_operator",
    "condition_estimate",
    "physical_condition_number",
    "gather_solve",
    "make_masked_operator",
]

#: The accepted ``boundary`` values.
BOUNDARIES: tuple[str, ...] = ("periodic", "dirichlet")

#: Largest relative asymmetry ``max|A - A^T| / max|A|`` the assembly
#: accepts before symmetrising.  Measured on the correct Galerkin assembly
#: in float64: 1.1e-17 .. 1.1e-16 over 15 configurations (periodic and
#: Dirichlet, 1-D to 3-D, 32 to 1024 functions, orders 2/4/6, mass 0.01 to
#: 100; jaxlib 0.11.0, but the product is NumPy).  A one-sided
#: ``[-1, 2, -1] / h`` stencil measures 1.07.  ``1e-12`` sits four orders
#: above the rounding floor and twelve below a wrong stencil, so the
#: symmetrisation that follows only removes rounding residue and can no
#: longer convert a first-order stencil into a consistent second-order one.
SYMMETRY_TOL: float = 1e-12

#: Lanczos steps :func:`condition_estimate` takes (all of them when the
#: basis is smaller).  Measured against ``numpy.linalg.eigvalsh`` of the
#: preconditioned operator over 40 periodic and 8 Dirichlet
#: configurations -- 48 to 4096 functions, 1-D to 3-D, orders 4 and 6,
#: mass 1e-12 to 100 -- with 48 steps the estimate was within 3% of the
#: exact condition number wherever it is above 1e-12 (0.4 s at 4096
#: functions); 64 adds margin for the extreme eigenvalue that is not
#: pinned by the closed form.
LANCZOS_STEPS: int = 64


# ---------------------------------------------------------------------------
# Physical-space finite-difference operators (dense, NumPy)
# ---------------------------------------------------------------------------
#
# Assembly happens once per node, eagerly.  It is written in NumPy rather
# than jnp on purpose: an eager jnp assembly pays a separate XLA compile
# for every distinct ``.at[].set`` / ``kron`` shape it meets (seconds per
# new grid size), and nothing here needs to be traced or differentiated.

def _stiffness_periodic(side: int, h: float) -> np.ndarray:
    """Periodic 1-D stiffness of ``-d2/dx2`` (circulant ``[-1, 2, -1] / h``)."""
    idx = np.arange(side)
    S = np.zeros((side, side))
    S[idx, idx] = 2.0 / h
    S[idx, (idx + 1) % side] += -1.0 / h
    S[idx, (idx - 1) % side] += -1.0 / h
    return S


def _stiffness_dirichlet(side: int, h: float) -> np.ndarray:
    """1-D stiffness of ``-d2/dx2`` on ``side`` interior nodes, zero at the walls."""
    idx = np.arange(side)
    S = np.zeros((side, side))
    S[idx, idx] = 2.0 / h
    S[idx[:-1], idx[:-1] + 1] = -1.0 / h
    S[idx[1:], idx[1:] - 1] = -1.0 / h
    return S


def _tensor_sum(S: np.ndarray, M: np.ndarray, dim: int, mass: float) -> np.ndarray:
    """``sum_axes kron(M, ..., S, ..., M) + mass * kron(M, ..., M)``."""
    if dim == 1:
        return S + mass * M
    out = None
    for ax in range(dim):
        factors = [M] * dim
        factors[ax] = S
        term = factors[0]
        for g in factors[1:]:
            term = np.kron(term, g)
        out = term if out is None else out + term
    mass_term = M
    for _ in range(dim - 1):
        mass_term = np.kron(mass_term, M)
    assert out is not None
    return out + mass * mass_term


def _physical_operator(side: int, dim: int, h: float, mass: float, boundary: str) -> np.ndarray:
    """Dense ``(-Laplacian + mass)`` bilinear form with lumped mass ``M = h I`` per axis."""
    if boundary == "periodic":
        S = _stiffness_periodic(side, h)
    else:
        S = _stiffness_dirichlet(side, h)
    M = h * np.eye(side)
    return _tensor_sum(S, M, dim, mass)


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------

def _lanczos_extremes(M: np.ndarray, steps: int) -> tuple[float, float]:
    """Extreme Ritz values of the symmetric ``M`` after ``steps`` Lanczos steps.

    Full reorthogonalisation (twice, classical Gram-Schmidt), float64,
    from a fixed-seed start vector: deterministic, and at most ``steps``
    dense matvecs.  The Ritz values lie inside the spectrum, so the pair
    brackets it from the inside.
    """
    n = M.shape[0]
    steps = max(1, min(int(steps), n))
    V = np.zeros((steps + 1, n))
    v = np.random.default_rng(0).standard_normal(n)
    V[0] = v / np.linalg.norm(v)
    alpha: list[float] = []
    beta: list[float] = []
    for j in range(steps):
        w = M @ V[j]
        alpha.append(float(V[j] @ w))
        for _ in range(2):
            w = w - V[: j + 1].T @ (V[: j + 1] @ w)
        b = float(np.linalg.norm(w))
        if j == steps - 1 or b <= 1e-14 * max(abs(alpha[-1]), 1.0):
            break
        beta.append(b)
        V[j + 1] = w / b
    k = len(alpha)
    T = np.diag(alpha) + np.diag(beta[: k - 1], 1) + np.diag(beta[: k - 1], -1)
    ritz = np.linalg.eigvalsh(T)
    return float(ritz[0]), float(ritz[-1])


@stability(StabilityLevel.EXPERIMENTAL)
def condition_estimate(A_hat: np.ndarray, *, rayleigh_bound: Optional[float] = None,
                       steps: int = LANCZOS_STEPS) -> float:
    """Spectral condition number of the symmetric positive-definite ``A_hat``.

    ``lambda_max`` is the largest Ritz value of :data:`LANCZOS_STEPS`
    Lanczos steps.  ``lambda_min`` is the smallest Ritz value, or
    ``rayleigh_bound`` when that is smaller -- any Rayleigh quotient is
    an upper bound on ``lambda_min``, and for the periodic operator the
    assembly passes the quotient of the constant function, which *is*
    the smallest eigenvalue to within a few percent once the mass is
    small (the eigenvalue a random-start Lanczos resolves last, and the
    one the ``1 / mass`` growth lives in).  Both extremes are therefore
    approached from inside the spectrum and, in exact arithmetic, the
    estimate is a lower bound on the true condition number: a refusal
    made on it is never spurious.  (Against ``eigvalsh`` of the
    assembled float64 operator it is within 3% below and at most 4e-7
    above -- the assembly's own rounding at the smallest masses.)

    Parameters
    ----------
    A_hat : numpy.ndarray
        Dense symmetric ``(n, n)`` float64 matrix.
    rayleigh_bound : float, optional
        A known Rayleigh quotient ``x^T A_hat x / x^T x``.
    steps : int
        Lanczos steps (capped at ``n``, where the estimate is exact).

    Returns
    -------
    float
        ``lambda_max / lambda_min``; ``inf`` when ``lambda_min <= 0``,
        i.e. the matrix is not positive definite to float64 precision.
    """
    lo, hi = _lanczos_extremes(np.asarray(A_hat, dtype=np.float64), steps)
    if rayleigh_bound is not None:
        lo = min(lo, float(rayleigh_bound))
    if not lo > 0.0:
        return float("inf")
    return hi / lo


@stability(StabilityLevel.EXPERIMENTAL)
def physical_condition_number(side: int, dim: int, mass: float, boundary: str) -> float:
    """Condition number of the grid operator ``-Laplacian_h + mass``, in closed form.

    The central-difference Laplacian on ``side`` points per axis has the
    eigenvalues ``sum_axes (2 - 2 cos(pi j_a / s)) / h**2``: periodic,
    ``s = side / 2`` and ``j_a = 0 .. side - 1``, so the smallest is ``0``
    (the constant) and the operator's is ``mass``; Dirichlet,
    ``s = side + 1`` and ``j_a = 1 .. side``, so the smallest is
    ``dim (2 - 2 cos(pi h)) / h**2 ~ dim pi**2``.  The ratio of the
    extremes of ``lambda + mass`` is returned.

    This is not the conditioning of the node's solve -- the wavelet
    change of basis and the diagonal scaling bring that down to
    :func:`condition_estimate` of ``D^-1 A D^-1`` -- but it bounds the
    error of *forming* ``A = Wn^T A_phys Wn`` in float64: the Laplacian
    annihilates the constant only through cancellation, and the rounding
    left over is ``~ eps * lambda_max``, which the smallest eigenvalue
    ``mass`` then divides.  Measured on the periodic basis at the full
    budget in float64 (jaxlib 0.11.0): the sensor-reading error against
    an FFT solve was 0.005 to 0.25 times ``physical_condition_number * eps``
    over 1-D 64 to 256 points, 2-D 8^2, 3-D 4^3 and order 6, mass 1e-5
    to 1e-10.
    """
    if boundary == "periodic":
        h = 1.0 / side
        top = 2.0 - 2.0 * np.cos(2.0 * np.pi * (side // 2) / side)
        lam_min, lam_max = 0.0, dim * top / h ** 2
    else:
        h = 1.0 / (side + 1)
        lam_min = dim * (2.0 - 2.0 * np.cos(np.pi * h)) / h ** 2
        lam_max = dim * (2.0 - 2.0 * np.cos(np.pi * side * h)) / h ** 2
    return float((lam_max + mass) / (lam_min + mass))


def _constant_mode_rayleigh(Wn: np.ndarray, D: np.ndarray, levels: np.ndarray,
                            mass: float, h: float, dim: int) -> Optional[float]:
    """Rayleigh quotient of the constant function for the periodic ``D^-1 A D^-1``.

    The periodic central-difference stencil annihilates constants, so
    ``1^T A_phys 1 = mass * h**dim * n`` exactly, and the constant has
    coefficients ``y = Wn^-1 1`` on the level-0 functions alone (the
    interpolating basis reproduces constants).  In the scaled
    coordinates ``x = D y`` the quotient is ``mass h^d n / |D y|^2`` --
    closed form in ``mass``, free of the cancellation that makes the
    smallest eigenvalue hard to compute at small mass.  ``None`` if the
    level-0 functions do not reproduce the constant (they always do on
    the periodic basis; the check keeps the bound honest).
    """
    lev0 = np.flatnonzero(levels == levels.min())
    ones = np.ones(Wn.shape[0])
    y, *_ = np.linalg.lstsq(Wn[:, lev0], ones, rcond=None)
    if float(np.max(np.abs(Wn[:, lev0] @ y - ones))) > 1e-10:
        return None
    energy = float(mass) * h ** dim * Wn.shape[0]
    return energy / float(np.sum((D[lev0] * y) ** 2))


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
    levels : numpy.ndarray
        ``int32`` level label per basis function.  A host array, never a
        tracer, so the seed size and the diagonal scaling can be computed
        from it when the node is built inside a trace.
    diagonal : numpy.ndarray
        ``float64`` host copy of ``diag(A)``, for the diagonal scaling.
    side : int
        Grid points per axis.
    n : int
        ``side ** dim``.
    h : float
        Grid spacing: ``1 / side`` (periodic) or ``1 / (side + 1)`` (Dirichlet).
    dim : int
    boundary : str
    condition_number : float or None
        :func:`condition_estimate` of ``D^-1 A D^-1`` for the diagonal
        scaling ``assemble_operator`` was asked to condition-check
        (``preconditioner=``), computed on the float64 operator before
        the cast to ``dtype``; ``None`` when none was requested.
    """

    A: jax.Array
    A_sparse: Any
    Wn: jax.Array
    levels: np.ndarray
    diagonal: np.ndarray
    side: int
    n: int
    h: float
    dim: int
    boundary: str
    condition_number: Optional[float] = None


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
    preconditioner: Optional[str] = None,
) -> WaveletOperator:
    """Assemble ``A = Wn^T A_phys Wn`` for ``(-Laplacian + mass)`` on the unit cube.

    Host-side: the basis, the physical operator and the triple product
    are built in float64 NumPy under ``jax.ensure_compile_time_eval``,
    so the call is legal inside a ``jax.jit`` trace (every input is a
    static setting) and costs the same there as eagerly.

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
    preconditioner : {"hybrid", "full", "level", "dk"}, optional
        When given, estimate the condition number of ``D^-1 A D^-1`` for
        that :func:`~maddening.nodes.adaptive.wavelets.precond.diagonal_scaling`
        with :func:`condition_estimate`, on the float64 operator and
        before the cast, and return it as
        :attr:`WaveletOperator.condition_number` (the node refuses a
        configuration its dtype cannot carry).  The periodic constant
        function's Rayleigh quotient is passed as the bound.

    Returns
    -------
    WaveletOperator

    Raises
    ------
    ValueError
        For an unknown ``boundary``, or when the assembled ``A`` is not
        symmetric to :data:`SYMMETRY_TOL` -- which can only mean the
        physical stencil is not, since the Galerkin product of a
        symmetric ``A_phys`` is symmetric to rounding.
    """
    if boundary not in BOUNDARIES:
        raise ValueError(f"boundary must be one of {BOUNDARIES}, got {boundary!r}")
    dt = jnp.zeros((), dtype=dtype).dtype
    # The basis is a function of static ints only.  Under an enclosing
    # trace ``jnp`` would stage it out and the ``np.asarray`` below would
    # fail on the tracer; ``ensure_compile_time_eval`` evaluates it on the
    # host instead, which is what a constant of the node should be.
    with jax.ensure_compile_time_eval():
        if boundary == "periodic":
            side = _tr.side_length(n_levels, n_coarse)
            h = 1.0 / side
            W = np.asarray(
                _tr.synthesis_matrix(n_levels, n_coarse, order=order, dim=dim, dtype=dt),
                dtype=np.float64,
            )
            levels = _tr._level_labels_np(n_levels, n_coarse, dim)
        else:
            W_j, _, side = _dir.synthesis_matrix_dirichlet(
                n_levels, n_coarse, order=order, dim=dim, dtype=dt,
            )
            W = np.asarray(W_j, dtype=np.float64)
            levels = _dir._level_labels_nd(n_levels, n_coarse, dim)
            h = 1.0 / (side + 1)
    n = side ** dim
    # Assembled in float64 NumPy whatever the requested dtype, then cast:
    # the Galerkin triple product is where a float32 basis would lose
    # symmetry at round-off, and the cast happens once.
    A_phys = _physical_operator(side, dim, h, float(mass), boundary)
    norms = np.sqrt((h ** dim) * np.sum(W ** 2, axis=0))
    norms = np.where(norms > 0, norms, 1.0)
    Wn = W / norms[None, :]
    A = Wn.T @ A_phys @ Wn
    # Check, THEN symmetrise.  Symmetrising unconditionally turned a
    # one-sided (first-order) stencil into a consistent second-order one,
    # and the MMS order gate passed against the defect (measured order
    # 2.04 with the defect seeded); with the check the defect is refused.
    asym = float(np.max(np.abs(A - A.T)))
    scale = float(np.max(np.abs(A)))
    if asym > SYMMETRY_TOL * scale:
        raise ValueError(
            f"assemble_operator: the Galerkin operator is not symmetric: "
            f"max|A - A^T| / max|A| = {asym / scale:.2e} > SYMMETRY_TOL = "
            f"{SYMMETRY_TOL:.0e} (boundary={boundary!r}, dim={dim}, side={side}).  "
            f"The correct assembly measures ~1e-16 here, so the physical stencil "
            f"is not symmetric; it is refused rather than symmetrised away"
        )
    A = 0.5 * (A + A.T)
    condition = None
    if preconditioner is not None:
        D = _pc._diagonal_scaling_np(np.diag(A), levels, preconditioner)
        bound = (
            _constant_mode_rayleigh(Wn, D, levels, mass, h, dim)
            if boundary == "periodic" else None
        )
        condition = condition_estimate(A / D[:, None] / D[None, :], rayleigh_bound=bound)
    thr = sparse_threshold * np.max(np.abs(A))
    rows, cols = np.nonzero(np.abs(A) >= thr)
    A_sparse = jsparse.BCOO(
        (jnp.asarray(A[rows, cols], dtype=dt),
         jnp.asarray(np.stack([rows, cols], axis=1), dtype=jnp.int32)),
        shape=(n, n),
    )
    return WaveletOperator(
        A=jnp.asarray(A, dtype=dt), A_sparse=A_sparse, Wn=jnp.asarray(Wn, dtype=dt),
        levels=levels, diagonal=np.ascontiguousarray(np.diag(A)), side=int(side),
        n=int(n), h=float(h), dim=int(dim), boundary=str(boundary),
        condition_number=condition,
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
        A mask with more than ``buf`` active entries cannot be solved in
        the buffer; rather than silently dropping the excess (a jitted
        function cannot raise) the gathered block is poisoned with
        ``NaN``, so the coefficients and the objective are non-finite
        and nothing downstream can mistake the result for the masked
        solve.  The node keeps this branch
        unreachable -- the seed is validated against ``buf`` at
        construction and the selection never grows past it -- refuses
        an oversized *concrete* mask on the eager path with a message,
        and overrides the base class's all-true full-basis gradient
        with a dense solve for the same reason.
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
    fits = jnp.sum(mask) <= buf                         # the guard (see ``mask``)
    ix = jnp.argsort(jnp.logical_not(mask))[:buf]      # active first
    active = mask[ix]                                  # (buf,) real-vs-padding
    Asub = A[jnp.ix_(ix, ix)]                          # (buf, buf)
    keep = active[:, None] & active[None, :]
    Asub = jnp.where(keep, Asub, jnp.eye(buf, dtype=A.dtype))
    bsub = jnp.where(active, rhs[ix], 0.0)
    csub = jnp.linalg.solve(Asub, bsub)
    csub = jnp.where(fits, csub, jnp.nan)               # poison, never truncate
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
