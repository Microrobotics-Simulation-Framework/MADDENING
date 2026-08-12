"""Matrix-free periodic geometric multigrid V-cycle for -div(a grad u) + m u.

Vertex-centred, tensor-product linear prolongation, R = P^T / 2^dim,
rediscretised coarse operator with a coarsened coefficient.
Symmetric V-cycle (equal pre/post damped-Jacobi) => M^-1 is symmetric => CG-safe.

Investigation only.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from maddening.nodes.adaptive.wavelets import matrixfree as MF


# ---------------- transfer ----------------

def prolong(uc, dim):
    """Linear interpolation, vertex-centred periodic, coarse->fine (2x per axis)."""
    u = uc
    for d in range(dim):
        n = u.shape[d]
        sh = list(u.shape); sh[d] = 2 * n
        out = jnp.zeros(sh, dtype=u.dtype)
        idx_even = [slice(None)] * u.ndim; idx_even[d] = slice(0, None, 2)
        idx_odd = [slice(None)] * u.ndim; idx_odd[d] = slice(1, None, 2)
        out = out.at[tuple(idx_even)].set(u)
        out = out.at[tuple(idx_odd)].set(0.5 * (u + jnp.roll(u, -1, axis=d)))
        u = out
    return u


def restrict(uf, dim):
    """R = P^T / 2^dim  (full weighting), fine->coarse."""
    z = jnp.zeros(tuple(s // 2 for s in uf.shape), dtype=uf.dtype)
    (out,) = jax.linear_transpose(lambda c: prolong(c, dim), z)(uf)
    return out / (2.0 ** dim)


def coarsen_coeff(a, dim, how="arith"):
    if how == "inject":
        sl = tuple(slice(0, None, 2) for _ in range(dim))
        return a[sl]
    # weighted average over the 2^dim stencil via P^T (full weighting)
    if how == "arith":
        return _fw(a, dim)
    if how == "harm":
        return 1.0 / _fw(1.0 / a, dim)
    raise ValueError(how)


def _fw(a, dim):
    """Full-weighting average of a (rows sum to 1)."""
    ones = jnp.ones_like(a)
    num = restrict(a, dim)
    den = restrict(ones, dim)
    return num / den


# ---------------- operator diagonal ----------------

def varcoeff_diag(a, side, dim, h, mass):
    a = a.reshape((side,) * dim)
    inv_h2 = 1.0 / h ** 2
    d = mass * jnp.ones_like(a)
    for ax in range(dim):
        d = d + 0.5 * (a + jnp.roll(a, -1, axis=ax)) * inv_h2
        d = d + 0.5 * (a + jnp.roll(a, 1, axis=ax)) * inv_h2
    return d


# ---------------- V-cycle ----------------

class MG:
    def __init__(self, a, side, dim, h, mass, n_levels, how="arith",
                 nu=2, omega=0.8, coarse_iters=30):
        self.dim, self.nu, self.omega, self.coarse_iters = dim, nu, omega, coarse_iters
        self.ops, self.diags, self.shapes = [], [], []
        ac = a.reshape((side,) * dim)
        s, hh = side, h
        for lev in range(n_levels + 1):
            self.ops.append(MF.make_varcoeff_apply(ac.reshape(-1), s, dim, hh, mass))
            self.diags.append(varcoeff_diag(ac.reshape(-1), s, dim, hh, mass))
            self.shapes.append((s,) * dim)
            if lev == n_levels:
                break
            ac = coarsen_coeff(ac, dim, how)
            s //= 2
            hh *= 2.0
        self.nlev = len(self.ops)

    def _smooth(self, lev, u, f, n):
        A, d = self.ops[lev], self.diags[lev]
        sh = self.shapes[lev]
        for _ in range(n):
            r = f - A(u.reshape(-1)).reshape(sh)
            u = u + self.omega * r / d
        return u

    def _vcycle(self, lev, u, f):
        if lev == self.nlev - 1:
            return self._smooth(lev, u, f, self.coarse_iters)
        u = self._smooth(lev, u, f, self.nu)
        r = f - self.ops[lev](u.reshape(-1)).reshape(self.shapes[lev])
        rc = restrict(r, self.dim)
        ec = self._vcycle(lev + 1, jnp.zeros_like(rc), rc)
        u = u + prolong(ec, self.dim)
        u = self._smooth(lev, u, f, self.nu)
        return u

    def apply(self, r_flat):
        """M^-1 r : one symmetric V-cycle from a zero initial guess."""
        f = r_flat.reshape(self.shapes[0])
        u = self._vcycle(0, jnp.zeros_like(f), f)
        return u.reshape(-1)
