"""R1 item 3: does the operator-mode marking indicator |M^-1 r| select a sensible
active set at high contrast?  Compare CDD energy error vs best-K-term oracle at
matched |Lambda|, for hybrid-diag indicator (|r_hat|) vs operator-mode (|M^-1 r|).

Small dense problem so we can form A_wave and do exact Galerkin solves on subsets.
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
    Aw = np.asarray(dense_of(p.A_wave, N)); Aw = 0.5 * (Aw + Aw.T)
    f = jnp.asarray(np.random.default_rng(1).normal(size=N))
    b = np.asarray(p.wn_transpose(f))
    cstar = np.linalg.solve(Aw, b)
    zero_e = energy_err(Aw, np.zeros(N), cstar)

    D = np.asarray(p.D("hybrid"))
    Ahat = (Aw / D[:, None]) / D[None, :]
    bhat = b / D
    mg = StencilMG(p.a, p.side, dim, p.h, p.mass, n_levels=nl)
    Minv = lambda v: p.wn_inv(mg.apply(p.wn_inv_T(v)))

    blocks, _ = T.structural_blocks(nl, nc, dim)
    coarse = jnp.asarray(np.asarray(blocks) == 0)
    ncoarse = int(np.sum(np.asarray(coarse)))

    Ahat_j = jnp.asarray(Ahat); Aw_j = jnp.asarray(Aw)

    def run_cdd(apply_op, rhs, ind, unscale, K):
        def solve_masked(m, r):
            M2 = m[:, None] & m[None, :]
            Amod = jnp.where(M2, apply_op, 0.0) + jnp.diag(jnp.where(m, 0.0, 1.0))
            return jnp.linalg.solve(Amod, jnp.where(m, r, 0.0))
        opfn = lambda v: apply_op @ v
        mask, c, conv = cdd_select(opfn, solve_masked, jnp.asarray(rhs),
                                   coarse, K, indicator=ind, max_outer=300)
        c = np.asarray(c)
        if unscale is not None:
            c = c / unscale
        return energy_err(Aw, c, cstar), int(np.sum(np.asarray(mask)))

    print(f"\n### {kind} dim={dim} N={N} contrast={ct}  zero-err={zero_e:.3e}  ncoarse={ncoarse}")
    print(f"{'K':>5} | {'orcl-D':>10} {'orcl-diagA':>10} | {'A:|rhat|':>10} (K) | {'B:|Minv r|':>10} (K)")
    for K in Ks:
        # oracles: best-K-term in two weightings, exact Galerkin on that set
        row = {}
        for tag, w in [("orcl-D", D), ("orcl-diagA", np.sqrt(np.abs(np.diag(Aw))))]:
            idx = np.argsort(-np.abs(w * cstar))[:K]
            idx = np.union1d(idx, np.where(np.asarray(coarse))[0])
            row[tag] = energy_err(Aw, galerkin_on(Aw, b, idx), cstar)
        eA, kA = run_cdd(Ahat_j, bhat, None, D, K)                       # diag mode
        eB, kB = run_cdd(Aw_j, b, lambda r: jnp.abs(Minv(r)), None, K)   # operator mode
        print(f"{K:5d} | {row['orcl-D']:10.3e} {row['orcl-diagA']:10.3e} |"
              f" {eA:10.3e} ({kA:3d}) | {eB:10.3e} ({kB:3d})", flush=True)


if __name__ == "__main__":
    Ks = [32, 64, 96, 128]
    for ct in [1.0, 100.0, 10000.0]:
        study(2, 3, 2, "jump", ct, Ks)      # N=256
    for ct in [100.0, 10000.0]:
        study(2, 3, 2, "checker", ct, Ks)
