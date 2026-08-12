"""LEVER R1 item 3: does the operator-mode marking indicator select a sensible
active set at high contrast?  Compare CDD energy error vs best-K-term at matched K.

Variants:
  A  diag-mode  : coords c_hat = D c,   indicator |r_hat|            (current production)
  B  op-mode    : coords c (identity),  indicator |M^-1 r|           (protocol's operator mode)
  C  scaled+MG  : coords c_hat = D c,   indicator |D M^-1 D r_hat|   (proposed refinement)
  D  scaled+MG  : coords c_hat = D c,   indicator |r_hat|            (MG only in the inner solve)

Oracles (best-K-term proxies): K largest |D_hybrid c*| and K largest |sqrt(diag A) c*|,
each followed by an exact Galerkin solve on that set.
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/home/nick/MSF/msf/MADDENING/spikes/wavelet_perf")

import jax, jax.numpy as jnp, numpy as np
from harness import Problem, dense_of
from stencil_mg import StencilMG
from maddening.nodes.adaptive.wavelets.cdd import cdd_select
from maddening.nodes.adaptive.wavelets import transform as T


def energy_err(Aw, c, cstar):
    e = np.asarray(c - cstar)
    return float(np.sqrt(max(e @ (Aw @ e), 0.0)))


def galerkin_on(Aw, b, idx):
    sub = np.linalg.solve(Aw[np.ix_(idx, idx)], b[idx])
    c = np.zeros(Aw.shape[0]); c[idx] = sub
    return c


def study(dim, nl, nc, kind, ct, Ks, mass=1.0):
    p = Problem(nl, nc, dim, kind=kind, contrast=ct, mass=mass)
    N = p.N
    Aw = np.asarray(dense_of(p.A_wave, N))
    Aw = 0.5 * (Aw + Aw.T)
    f = jnp.asarray(np.random.default_rng(0).normal(size=N))
    b = np.asarray(p.wn_transpose(f))
    cstar = np.linalg.solve(Aw, b)

    D = np.asarray(p.D("hybrid"))
    Dfull = np.sqrt(np.abs(np.diag(Aw)))
    mg = StencilMG(p.a, p.side, dim, p.h, p.mass, n_levels=nl)
    Minv = lambda v: p.wn_inv(mg.apply(p.wn_inv_T(v)))

    blocks, _ = T.structural_blocks(nl, nc, dim)
    coarse = jnp.asarray(blocks == 0)

    Ahat_np = (Aw / D[:, None]) / D[None, :]
    bhat = b / D

    out = {}
    for K in Ks:
        row = {}
        # ---- oracles ----
        for tag, w in [("orcl-D", D), ("orcl-diag", Dfull)]:
            idx = np.argsort(-np.abs(w * cstar))[:K]
            idx = np.union1d(idx, np.where(np.asarray(coarse))[0])
            row[tag] = energy_err(Aw, galerkin_on(Aw, b, idx), cstar)
        # ---- CDD variants ----
        def run(apply_op, rhs, ind, unscale):
            Amat = jnp.asarray(Ahat_np if unscale is not None else Aw)
            def solve_masked(m, r):
                M2 = m[:, None] & m[None, :]
                Amod = jnp.where(M2, Amat, 0.0) + jnp.diag(jnp.where(m, 0.0, 1.0))
                return jnp.linalg.solve(Amod, jnp.where(m, r, 0.0))
            mask, c, conv = cdd_select(apply_op, solve_masked, jnp.asarray(rhs),
                                       coarse, K, indicator=ind, max_outer=200)
            c = np.asarray(c)
            if unscale is not None:
                c = c / unscale
            return energy_err(Aw, c, cstar), int(np.sum(np.asarray(mask)))

        Ahat_j = jnp.asarray(Ahat_np); Aw_j = jnp.asarray(Aw)
        Ahat_fn = lambda v: Ahat_j @ v
        Aw_fn = lambda v: Aw_j @ v
        row["A_diag|r|"], kA = run(Ahat_fn, bhat, None, D)
        row["B_op|Minv r|"], kB = run(Aw_fn, b, lambda r: jnp.abs(Minv(r)), None)
        Mhat = lambda r: jnp.asarray(D) * Minv(jnp.asarray(D) * r)
        row["C_scaled|MGr|"], kC = run(Ahat_fn, bhat, lambda r: jnp.abs(Mhat(r)), D)
        row["_K"] = (kA, kB, kC)
        row["_zero"] = energy_err(Aw, np.zeros(N), cstar)
        out[K] = row
    return out
