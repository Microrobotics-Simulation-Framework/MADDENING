"""V-cycle with operator-dependent prolongation + rediscretised coarse operator."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from maddening.nodes.adaptive.wavelets import matrixfree as MF
from mg import coarsen_coeff, varcoeff_diag
from opdep import make_opdep_prolong


class MGOpDep:
    def __init__(self, a, side, dim, h, mass, n_levels, how="harm",
                 nu=2, omega=0.8, coarse_iters=30):
        self.dim, self.nu, self.omega, self.coarse_iters = dim, nu, omega, coarse_iters
        self.ops, self.diags, self.shapes, self.Ps = [], [], [], []
        ac = a.reshape((side,) * dim)
        s, hh = side, h
        for lev in range(n_levels + 1):
            self.ops.append(MF.make_varcoeff_apply(ac.reshape(-1), s, dim, hh, mass))
            self.diags.append(varcoeff_diag(ac.reshape(-1), s, dim, hh, mass))
            self.shapes.append((s,) * dim)
            if lev == n_levels:
                break
            # P built from THIS level's (fine) coefficients
            self.Ps.append(make_opdep_prolong(ac, dim, hh, mass))
            ac = coarsen_coeff(ac, dim, how)
            s //= 2
            hh *= 2.0
        self.nlev = len(self.ops)

    def _R(self, lev, uf):
        nc = self.shapes[lev + 1][0]
        z = jnp.zeros((nc,) * self.dim, dtype=uf.dtype)
        (out,) = jax.linear_transpose(self.Ps[lev], z)(uf)
        return out / (2.0 ** self.dim)

    def _smooth(self, lev, u, f, n):
        A, d, sh = self.ops[lev], self.diags[lev], self.shapes[lev]
        for _ in range(n):
            u = u + self.omega * (f - A(u.reshape(-1)).reshape(sh)) / d
        return u

    def _vcycle(self, lev, u, f):
        if lev == self.nlev - 1:
            return self._smooth(lev, u, f, self.coarse_iters)
        u = self._smooth(lev, u, f, self.nu)
        r = f - self.ops[lev](u.reshape(-1)).reshape(self.shapes[lev])
        ec = self._vcycle(lev + 1, jnp.zeros_like(self._R(lev, r)), self._R(lev, r))
        u = u + self.Ps[lev](ec)
        return self._smooth(lev, u, f, self.nu)

    def apply(self, r_flat):
        f = r_flat.reshape(self.shapes[0])
        return self._vcycle(0, jnp.zeros_like(f), f).reshape(-1)
