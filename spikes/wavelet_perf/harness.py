"""Shared harness for LEVER R1 (contrast-robust preconditioner) experiments.

Builds matrix-free wavelet operators + coefficient fields at a given contrast,
and provides dense materialisers (small N only) for exact kappa via eigh.

DO NOT import into src/ -- investigation only.
"""
from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from maddening.nodes.adaptive.wavelets import transform as T
from maddening.nodes.adaptive.wavelets import operator as OP
from maddening.nodes.adaptive.wavelets import matrixfree as MF
from maddening.nodes.adaptive.wavelets.precond import diagonal_scaling


# ----------------------------------------------------------------------
# Coefficient fields
# ----------------------------------------------------------------------

def coeff_field(kind: str, contrast: float, side: int, dim: int, seed: int = 0):
    """a(x) in [1, contrast]. Returns flat (side**dim,) float64."""
    idx = np.indices((side,) * dim) / float(side)          # (dim, side, ...)
    if kind == "const":
        a = np.ones((side,) * dim)
    elif kind == "smooth":
        # smooth log-varying: a = contrast**(0.5*(1+sin)) -> [1, contrast]
        s = np.zeros((side,) * dim)
        for d in range(dim):
            s = s + np.sin(2 * np.pi * idx[d])
        s = s / dim
        a = contrast ** (0.5 * (1.0 + s))
    elif kind == "jump":
        # centred square/cube inclusion of high coefficient
        m = np.ones((side,) * dim, dtype=bool)
        for d in range(dim):
            m &= (idx[d] > 0.25) & (idx[d] < 0.625)
        a = np.where(m, contrast, 1.0)
    elif kind == "checker":
        # random binary blocks (4x4 cells) -- the hard AMG-style case
        rng = np.random.default_rng(seed)
        nb = max(side // 8, 2)
        blk = rng.integers(0, 2, size=(nb,) * dim)
        rep = side // nb
        a = np.kron(blk, np.ones((rep,) * dim))
        a = np.where(a > 0, contrast, 1.0)
    else:
        raise ValueError(kind)
    return jnp.asarray(a.reshape(-1), dtype=jnp.float64)


# ----------------------------------------------------------------------
# Problem bundle
# ----------------------------------------------------------------------

class Problem:
    def __init__(self, n_levels: int, n_coarse: int, dim: int, order: int = 4,
                 kind: str = "smooth", contrast: float = 1.0, mass: float = 1.0,
                 seed: int = 0):
        self.n_levels, self.n_coarse, self.dim, self.order = n_levels, n_coarse, dim, order
        self.side = n_coarse * 2 ** n_levels
        self.h = 1.0 / self.side
        self.N = self.side ** dim
        self.mass = mass
        self.contrast = contrast
        self.a = coeff_field(kind, contrast, self.side, dim, seed)
        self.norms = OP.column_norms_fast(n_levels, n_coarse, order, dim, self.h)
        self.levels = T.levels_1d(n_levels, n_coarse) if dim == 1 else (
            T.levels_2d(n_levels, n_coarse) if dim == 2 else T.levels_3d(n_levels, n_coarse))
        # physical operator (scaled like operator.physical_varcoeff: h**dim * ...)
        self.a_phys = MF.make_varcoeff_apply(self.a, self.side, dim, self.h, mass)
        self.wn_apply, self.wn_transpose = MF.make_wn_ops(
            n_levels, n_coarse, order, dim, self.norms)

    # ---- unscaled wavelet operator ----
    def A_wave(self, v):
        return self.wn_transpose(self.a_phys(self.wn_apply(v)))

    # ---- Wn^{-1} and Wn^{-T} (the pullback maps) ----
    def wn_inv(self, u):
        """Wn^{-1} u : grid -> coeffs.  Wn = W diag(1/norms) => Wn^-1 = diag(norms) W^-1."""
        ana = {1: T.analysis_1d, 2: T.analysis_2d, 3: T.analysis_3d}[self.dim]
        return self.norms * ana(u, self.n_levels, self.n_coarse, self.order)

    def wn_inv_T(self, v):
        """Wn^{-T} v : coeffs -> grid, exact transpose of wn_inv via linear_transpose."""
        z = jnp.zeros(self.N, dtype=jnp.float64)
        (out,) = jax.linear_transpose(self.wn_inv, z)(v)
        return out

    def diag_wave(self):
        return MF.wave_diagonal_fast(self.n_levels, self.n_coarse, self.order,
                                     self.dim, self.norms, self.a_phys)

    def D(self, kind="hybrid"):
        return diagonal_scaling(self.diag_wave(), self.levels, kind)


# ----------------------------------------------------------------------
# Dense materialiser + exact kappa
# ----------------------------------------------------------------------

def dense_of(fn, N):
    return jax.vmap(lambda e: fn(e))(jnp.eye(N, dtype=jnp.float64)).T


def kappa_pre(A_dense, Minv_dense):
    """kappa of the preconditioned operator, via symmetric generalized eig."""
    import scipy.linalg as sla
    A = np.asarray(A_dense, dtype=np.float64)
    Mi = np.asarray(Minv_dense, dtype=np.float64)
    Mi = 0.5 * (Mi + Mi.T)
    A = 0.5 * (A + A.T)
    # eig of Mi @ A  <=>  gen eig  A x = lam M x. Use sqrt of Mi (SPD).
    w, V = np.linalg.eigh(Mi)
    if w.min() <= 0:
        # not SPD -> fall back to plain eigenvalues of Mi@A
        ev = np.real(np.linalg.eigvals(Mi @ A))
        ev = ev[ev > 1e-12 * ev.max()]
        return ev.max() / ev.min()
    S = V @ np.diag(np.sqrt(w)) @ V.T
    ev = np.linalg.eigvalsh(S @ A @ S)
    ev = ev[ev > 1e-12 * ev.max()]
    return float(ev.max() / ev.min())


def kappa_plain(A_dense):
    A = np.asarray(A_dense, dtype=np.float64)
    A = 0.5 * (A + A.T)
    ev = np.linalg.eigvalsh(A)
    ev = ev[ev > 1e-12 * ev.max()]
    return float(ev.max() / ev.min())


# ----------------------------------------------------------------------
# Plain PCG with iteration counting (numpy, for diagnostics)
# ----------------------------------------------------------------------

def pcg_count(apply, b, Minv=None, tol=1e-8, maxit=5000, x0=None):
    """Return (x, iters, resid_hist). apply/Minv are callables on jnp arrays."""
    b = jnp.asarray(b)
    x = jnp.zeros_like(b) if x0 is None else x0
    r = b - apply(x)
    z = r if Minv is None else Minv(r)
    p = z
    rz = float(jnp.vdot(r, z))
    bnorm = float(jnp.linalg.norm(b))
    hist = [float(jnp.linalg.norm(r)) / bnorm]
    for k in range(maxit):
        Ap = apply(p)
        pAp = float(jnp.vdot(p, Ap))
        if pAp <= 0:
            break
        alpha = rz / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rn = float(jnp.linalg.norm(r)) / bnorm
        hist.append(rn)
        if rn < tol:
            return x, k + 1, hist
        z = r if Minv is None else Minv(r)
        rz_new = float(jnp.vdot(r, z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x, maxit, hist
