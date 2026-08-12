"""Operator-dependent (Dendy / BoxMG-style) prolongation, matrix-free, periodic.

Fine points are classified by how many of their coordinates are odd:
  0 odd  -> coincides with a coarse point: inject.
  k odd  -> collapse the operator row in the (dim-k) EVEN directions (lump those
            couplings onto the diagonal) and solve the resulting row for the
            centre in terms of its 2k neighbours along the odd axes -- which are
            (k-1)-odd points, already computed.  Processed in increasing k.

For the 5/7-point stencil  A u[i] = m u[i] + sum_{d,±} af_{d,±}(u[i] - u[i±e_d])
the collapsed row at a k-odd point is
  (m + sum_{d in ODD} (af_{d,+} + af_{d,-})) u[i] = sum_{d in ODD} af_{d,±} u[i±e_d]
(the even-direction couplings cancel against their diagonal contribution).

R = P^T exactly (via jax.linear_transpose) => Galerkin-consistent, symmetric.
Investigation only.
"""
from __future__ import annotations

import itertools

import jax
import jax.numpy as jnp


def face_coeffs(a, dim, h):
    """af[d] = (a_plus, a_minus) at every fine point, matching make_varcoeff_apply."""
    inv_h2 = 1.0 / h ** 2
    out = []
    for d in range(dim):
        ap = 0.5 * (a + jnp.roll(a, -1, axis=d)) * inv_h2   # face towards +e_d
        am = 0.5 * (a + jnp.roll(a, 1, axis=d)) * inv_h2    # face towards -e_d
        out.append((ap, am))
    return out


def make_opdep_prolong(a, dim, h, mass):
    """Return P: coarse array (n/2,)*dim -> fine array (n,)*dim."""
    a = jnp.asarray(a)
    af = face_coeffs(a, dim, h)

    def P(uc):
        n_c = uc.shape[0]
        n = 2 * n_c
        u = jnp.zeros((n,) * dim, dtype=uc.dtype)
        # k = 0: inject at all-even points
        sl_even = tuple(slice(0, None, 2) for _ in range(dim))
        u = u.at[sl_even].set(uc)
        # k = 1..dim in order
        for k in range(1, dim + 1):
            for odd in itertools.combinations(range(dim), k):
                sl = tuple(slice(1, None, 2) if d in odd else slice(0, None, 2)
                           for d in range(dim))
                # collapsed diagonal: mass + sum over ODD axes of both faces
                den = mass * jnp.ones((n,) * dim, dtype=uc.dtype)
                for d in odd:
                    den = den + af[d][0] + af[d][1]
                num = jnp.zeros((n,) * dim, dtype=uc.dtype)
                for d in odd:
                    # neighbours along odd axis d are (k-1)-odd points: already set
                    num = num + af[d][0] * jnp.roll(u, -1, axis=d)
                    num = num + af[d][1] * jnp.roll(u, 1, axis=d)
                u = u.at[sl].set((num / den)[sl])
        return u

    return P


def make_opdep_restrict(P, dim, n_coarse):
    def R(uf):
        z = jnp.zeros((n_coarse,) * dim, dtype=uf.dtype)
        (out,) = jax.linear_transpose(P, z)(uf)
        return out
    return R
