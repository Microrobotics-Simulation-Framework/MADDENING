"""Investigation 5 -- submatrix conditioning in 2D and 3D.

SPIKE CODE. Investigative.

The 1D post-gate result (kappa(A_LL) ~ kappa(A_full)) was only validated in 1D.
In 2D/3D the isotropic Mallat basis has 3/7 detail subbands per level, coupled
through the physical operator. Does cross-subband coupling make kappa(A_LL)
blow up for active-set configs that include some subbands but not others?

Configs at k=N/16:
  balanced     : natural CDD output (mixed subbands)
  subband-biased: all from one subband (LH 2D / LLH 3D) -- pathological
  level-biased : all from the finest level -- pathological
"""

from __future__ import annotations

import os
import numpy as np

import dd_wavelets as dd
from hybrid_jacobi import precond, scaled, _l2, cdd_idx


def subbands_2d(n_levels, n_coarse):
    labs = ["LL"] * (n_coarse ** 2)
    cur = n_coarse
    for _ in range(n_levels):
        for kind in ("LH", "HL", "HH"):
            labs += [kind] * (cur * cur)
        cur *= 2
    return np.array(labs)


def subbands_3d(n_levels, n_coarse):
    labs = ["LLL"] * (n_coarse ** 3)
    cur = n_coarse
    par = ["LLH", "LHL", "LHH", "HLL", "HLH", "HHL", "HHH"]
    for _ in range(n_levels):
        for p in par:
            labs += [p] * (cur ** 3)
        cur *= 2
    return np.array(labs)


def configs(levels, subs, N, K, A_scaled, b):
    coarse = np.where(levels == levels.min())[0]
    out = {}
    # balanced: natural CDD
    out["balanced"] = cdd_idx(A_scaled, b, levels, K)
    # subband-biased: coarse + one subband, coarsest-level first
    one = sorted(set(subs) - {subs[coarse[0]]})[0]  # first detail subband label
    pool = np.where(subs == one)[0]
    pool = pool[np.argsort(levels[pool])]  # coarse levels first
    sb = np.concatenate([coarse, pool])[:K]
    out["subband-biased"] = np.array(sorted(set(sb.tolist())))
    # level-biased: coarse + finest level only
    fine = np.where(levels == levels.max())[0]
    lb = np.concatenate([coarse, fine])[:K]
    out["level-biased"] = np.array(sorted(set(lb.tolist())))
    return out


def run_2d():
    print("=" * 78)
    print("PART A -- 2D submatrix conditioning (Nside=32, N=1024, k=N/16=64)")
    print("=" * 78)
    nl = 4
    W2, levels, Nside = dd.synthesis_matrix_2d_isotropic(nl, 2, 4)
    subs = subbands_2d(nl, 2)
    h = 1.0 / Nside
    W2n = W2 / np.sqrt((h*h) * np.sum(W2**2, axis=0))[None, :]
    S = dd.laplacian_periodic(Nside, mass=False); Mm = h*np.eye(Nside)
    A = W2n.T @ (np.kron(S, Mm) + np.kron(Mm, S) + np.kron(Mm, Mm)) @ W2n
    A = 0.5 * (A + A.T)
    N = Nside * Nside; K = N // 16
    rng = np.random.default_rng(0); b = rng.standard_normal(N)
    report(A, levels, subs, N, K, b)


def run_3d():
    print("=" * 78)
    print("PART B -- 3D submatrix conditioning (Nside=16, N=4096, k=N/16=256)")
    print("=" * 78)
    cache = "/tmp/g2_3d_env.npz"
    if os.path.exists(cache):
        z = np.load(cache); A = z["A"]; levels = z["levels"]; Nside = int(z["Nside"])
        print(f"  (loaded cached 3D operator from {cache})")
    else:
        W3, levels, Nside = dd.synthesis_matrix_3d_isotropic(4, 1, 4)
        h = 1.0 / Nside
        W3n = W3 / np.sqrt((h**3) * np.sum(W3**2, axis=0))[None, :]
        S = dd.laplacian_periodic(Nside, mass=False); Mm = h*np.eye(Nside); I = np.eye(Nside)
        Aph = (np.kron(np.kron(S, Mm), Mm) + np.kron(np.kron(Mm, S), Mm)
               + np.kron(np.kron(Mm, Mm), S) + np.kron(np.kron(Mm, Mm), Mm))
        A = W3n.T @ Aph @ W3n; A = 0.5*(A+A.T)
    n_coarse = 1 if Nside == 16 else 2
    nl = int(round(np.log2(Nside / n_coarse)))
    subs = subbands_3d(nl, n_coarse)
    assert len(subs) == A.shape[0], (len(subs), A.shape[0])
    N = A.shape[0]; K = N // 16
    rng = np.random.default_rng(0); b = rng.standard_normal(N)
    report(A, levels, subs, N, K, b)


def report(A, levels, subs, N, K, b):
    print(f"  {'config':>15} {'precond':>7} {'kap(A_ΛΛ)':>10} {'kap_full':>9} {'ratio':>6}")
    for kind in ("full", "hybrid", "dk"):
        D, _ = precond(A, levels, kind)
        As = scaled(A, D)
        kfull = np.linalg.cond(As)
        cfg = configs(levels, subs, N, K, As, b / D)
        for cname, idx in cfg.items():
            ksub = np.linalg.cond(As[np.ix_(idx, idx)])
            print(f"  {cname:>15} {kind:>7} {ksub:>10.1f} {kfull:>9.1f} "
                  f"{ksub/kfull:>6.2f}")
        print()


if __name__ == "__main__":
    run_2d()
    run_3d()
