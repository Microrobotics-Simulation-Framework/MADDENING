"""Matrix-free contrast-robust multigrid: op-dependent P + exact Galerkin RAP.

Everything is a 3^dim STENCIL on a periodic grid, stored as {offset: (n,)*dim array}.
Nothing is ever materialised as a matrix.

Two ideas make Galerkin RAP matrix-free:

1. **Op-dependent prolongation from a general stencil** (Dendy/BoxMG).  Fine points
   are visited in order of how many coordinates are odd (k).  For a k-odd point the
   stencil is LUMPED over the even directions and the collapsed row is solved for the
   centre; all its neighbours along odd axes have strictly fewer odd coords, so they
   are already set.  Closed on 3^dim stencils, so it works at every level.

2. **RAP by colored probing.**  A_c = P^T A P is a 3^dim stencil.  The offsets
   {-1,0,1} are DISTINCT mod 4, so with colors e_k = 1[i == k (mod 4)] each probe
   y_k = A_c(e_k) reads off exactly one stencil entry per point:
   S_c[s][i] = y_{(i+s) mod 4}[i].  4^dim probes per level recover the coarse
   operator EXACTLY (not an approximation), using only matvecs.

Requires every level's side to be divisible by 4.
Investigation only.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp
import numpy as np

OFFSETS = lambda dim: list(itertools.product((-1, 0, 1), repeat=dim))


# ---------------- stencil apply ----------------

def stencil_apply(S, dim):
    """S: dict offset->array.  Returns u -> A u  (periodic)."""
    def A(u):
        out = jnp.zeros_like(u)
        for s, c in S.items():
            if s == (0,) * dim:
                out = out + c * u
            else:
                v = u
                for d in range(dim):
                    if s[d]:
                        v = jnp.roll(v, -s[d], axis=d)
                out = out + c * v
        return out
    return A


def stencil_from_coeff(a, dim, h, mass):
    """5/7-point stencil of  m u - div(a grad u), matching make_varcoeff_apply."""
    a = a.reshape((a.shape[0],) if a.ndim == 1 else a.shape)
    inv_h2 = 1.0 / h ** 2
    S = {}
    diag = mass * jnp.ones_like(a)
    for d in range(dim):
        ap = 0.5 * (a + jnp.roll(a, -1, axis=d)) * inv_h2
        am = 0.5 * (a + jnp.roll(a, 1, axis=d)) * inv_h2
        diag = diag + ap + am
        sp = tuple(1 if e == d else 0 for e in range(dim))
        sm = tuple(-1 if e == d else 0 for e in range(dim))
        S[sp] = -ap
        S[sm] = -am
    S[(0,) * dim] = diag
    return S


# ---------------- op-dependent prolongation from a stencil ----------------

def make_P_from_stencil(S, dim):
    """Dendy prolongation coarse (n/2)^dim -> fine n^dim, built from stencil S."""
    def P(uc):
        n = 2 * uc.shape[0]
        u = jnp.zeros((n,) * dim, dtype=uc.dtype)
        u = u.at[tuple(slice(0, None, 2) for _ in range(dim))].set(uc)
        for k in range(1, dim + 1):
            for odd in itertools.combinations(range(dim), k):
                # lump: sum stencil over the EVEN components
                T = {}
                for s, c in S.items():
                    t = tuple(s[d] for d in odd)
                    T[t] = T.get(t, 0.0) + c
                sl = tuple(slice(1, None, 2) if d in odd else slice(0, None, 2)
                           for d in range(dim))
                num = jnp.zeros((n,) * dim, dtype=uc.dtype)
                for t, c in T.items():
                    if all(x == 0 for x in t):
                        continue
                    v = u
                    for j, d in enumerate(odd):
                        if t[j]:
                            v = jnp.roll(v, -t[j], axis=d)
                    num = num - c * v
                den = T[(0,) * k]
                u = u.at[sl].set((num / den)[sl])
        return u
    return P


# ---------------- Galerkin RAP by colored probing ----------------

def _color_ids(n, dim):
    """flat color id (0..4^dim-1) of every fine point, from (i mod 4) per axis."""
    idx = np.indices((n,) * dim)
    cid = np.zeros((n,) * dim, dtype=np.int32)
    for d in range(dim):
        cid = cid * 4 + (idx[d] % 4)
    return cid


def rap_stencil(S_f, P, dim, n_c):
    """Exact coarse stencil of P^T A_f P via 4^dim colored probes."""
    A_f = stencil_apply(S_f, dim)
    z = jnp.zeros((n_c,) * dim)

    def A_c(v):
        (out,) = jax.linear_transpose(P, z)(A_f(P(v)))
        return out

    cid_np = _color_ids(n_c, dim)
    cid = jnp.asarray(cid_np)
    ncol = 4 ** dim
    probes = []
    for k in range(ncol):
        e = (cid == k).astype(jnp.float64)
        probes.append(A_c(e))
    Y = jnp.stack(probes, axis=0)                       # (ncol, n_c^dim...)

    S_c = {}
    for s in OFFSETS(dim):
        # color id of the point i+s
        sh = cid_np
        for d in range(dim):
            if s[d]:
                sh = np.roll(sh, -s[d], axis=d)
        S_c[s] = jnp.take_along_axis(Y, jnp.asarray(sh)[None], axis=0)[0]
    return S_c


# ---------------- the V-cycle ----------------

class StencilMG:
    def __init__(self, a, side, dim, h, mass, n_levels, nu=2, omega=0.8,
                 coarse_iters=40):
        self.dim, self.nu, self.omega, self.coarse_iters = dim, nu, omega, coarse_iters
        S = stencil_from_coeff(jnp.asarray(a).reshape((side,) * dim), dim, h, mass)
        self.Ss, self.Ps, self.shapes = [S], [], [(side,) * dim]
        s = side
        for lev in range(n_levels):
            if s // 2 < 4:
                break
            P = make_P_from_stencil(self.Ss[-1], dim)
            self.Ps.append(P)
            self.Ss.append(rap_stencil(self.Ss[-1], P, dim, s // 2))
            s //= 2
            self.shapes.append((s,) * dim)
        self.nlev = len(self.Ss)
        self.ops = [stencil_apply(S, dim) for S in self.Ss]
        self.diags = [S[(0,) * dim] for S in self.Ss]

    def _sm(self, l, u, f, n):
        for _ in range(n):
            u = u + self.omega * (f - self.ops[l](u)) / self.diags[l]
        return u

    def _v(self, l, u, f):
        if l == self.nlev - 1:
            return self._sm(l, u, f, self.coarse_iters)
        u = self._sm(l, u, f, self.nu)
        r = f - self.ops[l](u)
        z = jnp.zeros(self.shapes[l + 1])
        (rc,) = jax.linear_transpose(self.Ps[l], z)(r)
        ec = self._v(l + 1, jnp.zeros_like(rc), rc)
        u = u + self.Ps[l](ec)
        return self._sm(l, u, f, self.nu)

    def apply(self, r_flat):
        f = r_flat.reshape(self.shapes[0])
        return self._v(0, jnp.zeros_like(f), f).reshape(-1)
