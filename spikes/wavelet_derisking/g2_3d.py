"""Gate 2, §4 -- 3D sparsity break-even and 3D trap structure.

SPIKE CODE. Investigative.

Questions:
  (A) Does the adaptive solver achieve J_err < 5% at k_active < N/8 in 3D
      (the 'worth building' threshold)? Sweep k in {N/16,N/8,N/4,N/2}.
  (B) Is kappa(A_scaled) in 3D within a factor of 2 of 1D/2D?
  (C) 3D trap structure: blindness ratio on the 3x3x3 theta grid; minimum
      symmetry-break delta from the central trap.

16^3 = 4096 interior DOF, periodic, isotropic DD-4 + single-level DK 2^j
(per Correction C1). Algebraic Jacobi also reported.
"""

from __future__ import annotations

import numpy as np

import dd_wavelets as dd

SIGMA_SMOOTH = 0.10
SIGMA_SHARP = 0.02
SENSOR = (0.7, 0.6, 0.5)


import os

_CACHE = "/tmp/g2_3d_env.npz"


def build_3d(n_levels=4, order=4, use_cache=True):
    if use_cache and os.path.exists(_CACHE):
        z = np.load(_CACHE)
        return dict(W3n=z["W3n"], levels=z["levels"], Nside=int(z["Nside"]),
                    h=float(z["h"]), A=z["A"], X=z["X"], Y=z["Y"], Z=z["Z"],
                    sidx=int(z["sidx"]), N=int(z["N"]), _=None)
    W3, levels, Nside = dd.synthesis_matrix_3d_isotropic(n_levels, 1, order)
    h = 1.0 / Nside
    norms = np.sqrt((h ** 3) * np.sum(W3 ** 2, axis=0))
    norms[norms == 0] = 1.0
    W3n = W3 / norms[None, :]

    S = dd.laplacian_periodic(Nside, mass=False)
    Mm = h * np.eye(Nside)
    I = np.eye(Nside)
    # H^1 form: grad.grad + mass, isotropic
    A_phys = (np.kron(np.kron(S, Mm), Mm)
              + np.kron(np.kron(Mm, S), Mm)
              + np.kron(np.kron(Mm, Mm), S)
              + np.kron(np.kron(Mm, Mm), Mm))
    A = W3n.T @ A_phys @ W3n
    A = 0.5 * (A + A.T)

    coords = np.arange(Nside) / Nside
    X, Y, Z = np.meshgrid(coords, coords, coords, indexing="ij")
    sx = int(np.argmin(np.abs(coords - SENSOR[0])))
    sy = int(np.argmin(np.abs(coords - SENSOR[1])))
    sz = int(np.argmin(np.abs(coords - SENSOR[2])))
    sidx = (sx * Nside + sy) * Nside + sz

    env = dict(W3n=W3n, levels=levels, Nside=Nside, h=h, A=A,
               X=X, Y=Y, Z=Z, sidx=sidx, N=Nside ** 3, _=I)
    np.savez(_CACHE, W3n=W3n, levels=levels, Nside=Nside, h=h, A=A,
             X=X, Y=Y, Z=Z, sidx=sidx, N=Nside ** 3)
    return env


def load(env, theta, sigma):
    X, Y, Z = env["X"], env["Y"], env["Z"]
    f = np.exp(-(((X - theta[0]) ** 2 + (Y - theta[1]) ** 2
                  + (Z - theta[2]) ** 2) / sigma ** 2)).reshape(-1)
    return (env["h"] ** 3) * (env["W3n"].T @ f)


def scaled(env, which):
    lev = env["levels"].astype(float)
    A = env["A"]
    if which == "dk":
        D = 2.0 ** lev
    elif which == "jacobi":
        D = np.sqrt(np.abs(np.diag(A)))
    else:
        D = np.ones_like(lev)
    Ahat = (A / D[:, None]) / D[None, :]
    return Ahat, D


def cdd_snapshots(Ahat, bh, levels, srow_hat, J_full, ks, theta_D=0.5):
    """Grow CDD; record J_err the first time |Lambda| >= each k in ks."""
    N = len(bh)
    coarse = np.where(levels == levels.min())[0]
    Lam = set(coarse.tolist())
    want = sorted(ks)
    out = {}
    denom = abs(J_full) + 1e-30
    bnorm = np.linalg.norm(bh) + 1e-30
    n_outer = 0
    while want:
        idx = np.array(sorted(Lam))
        c = np.zeros(N)
        c[idx] = np.linalg.solve(Ahat[np.ix_(idx, idx)], bh[idx])
        while want and len(idx) >= want[0]:
            k = want.pop(0)
            out[k] = (len(idx), abs(srow_hat @ c - J_full) / denom, n_outer)
        r = bh - Ahat @ c
        if np.linalg.norm(r) <= 1e-6 * bnorm:
            for k in want:
                out[k] = (len(idx), abs(srow_hat @ c - J_full) / denom, n_outer)
            break
        mask_out = np.ones(N, dtype=bool); mask_out[idx] = False
        oi = np.where(mask_out)[0]
        order = oi[np.argsort(-np.abs(r[oi]))]
        csum = np.cumsum(r[order] ** 2)
        ktake = min(int(np.searchsorted(csum, (theta_D ** 2) * np.linalg.norm(r) ** 2) + 1), len(order))
        Lam.update(order[:ktake].tolist())
        n_outer += 1
        if n_outer > 300:
            break
    return out


def _cdd_idx(Ahat, bh, levels, K, theta_D=0.5):
    """Grow a CDD active set until |Lambda| >= K; return the index array."""
    N = len(bh)
    Lam = set(np.where(levels == levels.min())[0].tolist())
    while True:
        idx = np.array(sorted(Lam))
        if len(idx) >= K:
            return idx
        c = np.zeros(N)
        c[idx] = np.linalg.solve(Ahat[np.ix_(idx, idx)], bh[idx])
        r = bh - Ahat @ c
        mask_out = np.ones(N, dtype=bool); mask_out[idx] = False
        oi = np.where(mask_out)[0]
        order = oi[np.argsort(-np.abs(r[oi]))]
        csum = np.cumsum(r[order] ** 2)
        ktake = min(int(np.searchsorted(csum, (theta_D ** 2) * np.linalg.norm(r) ** 2) + 1), len(order))
        if ktake == 0:
            return idx
        Lam.update(order[:ktake].tolist())


def run_sparsity(env):
    N = env["N"]
    ks = [N // 16, N // 8, N // 4, N // 2]
    print(f"3D isotropic DD-4, N_side={env['Nside']}, N={N}, sensor idx={env['sidx']}")
    print(f"k thresholds N/16,N/8,N/4,N/2 = {ks}")
    for which in ("dk", "jacobi"):
        Ahat, D = scaled(env, which)
        srow_hat = env["W3n"][env["sidx"]] / D
        print(f"\n  scaling={which}")
        print(f"  {'sigma':>7} " + " ".join(f"k={k}".rjust(13) for k in ks))
        for sigma in (SIGMA_SMOOTH, SIGMA_SHARP):
            theta = (0.5, 0.5, 0.5)  # will move; use mid then sensor-ish
            theta = (0.42, 0.47, 0.51)
            bh = load(env, theta, sigma) / D
            c_full = np.linalg.solve(Ahat, bh)
            J_full = srow_hat @ c_full
            snaps = cdd_snapshots(Ahat, bh, env["levels"], srow_hat, J_full, ks)
            cells = []
            for k in ks:
                kk, jerr, _ = snaps[k]
                cells.append(f"{jerr:.2e}".rjust(13))
            print(f"  {sigma:>7.2f} " + " ".join(cells))


def run_kappa(env):
    print("\n--- 3D kappa(A_scaled) (compare to 1D ~48, 2D ~110 for DD-4) ---")
    for which in ("none", "dk", "jacobi"):
        Ahat, D = scaled(env, which)
        k = np.linalg.cond(Ahat)
        print(f"  {which:>7}: kappa = {k:.3e}")


def tag_of(th):
    n = sum(1 for t in th if abs(t - 0.5) < 1e-9)
    return {0: "interior", 1: "face", 2: "edge", 3: "centre"}[n]


def _sidx_for(env, coord):
    Nside = env["Nside"]
    coords = np.arange(Nside) / Nside
    sx = int(np.argmin(np.abs(coords - coord[0])))
    sy = int(np.argmin(np.abs(coords - coord[1])))
    sz = int(np.argmin(np.abs(coords - coord[2])))
    return (sx * Nside + sy) * Nside + sz


def run_traps(env, sensor_coord=None, label="off-axis"):
    """3D blindness ratio on the 3x3x3 grid (per-axis) + mechanism probe.

    The trap is a FROZEN-gradient phenomenon (Selection-Equivariance):
    blindness_ratio = |g_frozen| / |g_full|, where g_frozen differentiates
    J with the active set Lambda(theta) held FIXED (selection is
    non-differentiable). The theorem only forces g_frozen -> 0 in
    symmetry-breaking directions when the OBJECTIVE is also G-symmetric.
    """
    import scipy.linalg as sla
    Ahat, D = scaled(env, "dk")
    lu = sla.lu_factor(Ahat)  # factor ONCE; reuse for all full solves
    sidx = env["sidx"] if sensor_coord is None else _sidx_for(env, sensor_coord)
    srow_hat = env["W3n"][sidx] / D
    print(f"\n=== trap structure, sensor={label} (idx={sidx}) ===")
    levels = env["levels"]
    coarse = np.where(levels == levels.min())[0]
    K = env["N"] // 8

    def bhat_at(theta, sigma):
        return load(env, theta, sigma) / D

    def J_full_at(theta, sigma=SIGMA_SMOOTH):
        c = sla.lu_solve(lu, bhat_at(theta, sigma))
        return srow_hat @ c

    def select_at(theta, sigma=SIGMA_SMOOTH, method="cdd"):
        """Active set (size ~K) chosen AT theta -- the frozen mask."""
        bh = bhat_at(theta, sigma)
        if method == "cdd":
            return _cdd_idx(Ahat, bh, levels, K)
        elif method == "topb":
            # top-|b| (the deprecated selection that produced 1D blindness)
            return np.argsort(-np.abs(bh))[:K]
        raise ValueError(method)

    def J_frozen_at(theta, idx, sigma=SIGMA_SMOOTH):
        bh = bhat_at(theta, sigma)
        c = np.zeros_like(bh)
        c[idx] = np.linalg.solve(Ahat[np.ix_(idx, idx)], bh[idx])
        return srow_hat @ c

    def grad_vec(fn, theta, eps=1e-4):
        g = np.zeros(3)
        for d in range(3):
            tp = list(theta); tp[d] += eps
            tm = list(theta); tm[d] -= eps
            g[d] = (fn(tuple(tp)) - fn(tuple(tm))) / (2 * eps)
        return g

    grid = [0.25, 0.5, 0.75]
    print("  (trap if ratio<0.7; compare CDD vs deprecated top-|b| selection)")
    print(f"  {'theta':>20} {'tag':>8} {'ratio_CDD':>10} {'ratio_topb':>11}")
    worst = {"cdd": (1e9, None), "topb": (1e9, None)}
    for tx in grid:
        for ty in grid:
            for tz in grid:
                th = (tx, ty, tz)
                gf = np.linalg.norm(grad_vec(J_full_at, th))
                ratios = {}
                for m in ("cdd", "topb"):
                    idx = select_at(th, method=m)
                    gz = np.linalg.norm(grad_vec(lambda t: J_frozen_at(t, idx), th))
                    r = gz / (gf + 1e-30)
                    ratios[m] = r
                    if r < worst[m][0]:
                        worst[m] = (r, (th, tag_of(th)))
                if sum(1 for t in th if abs(t - 0.5) < 1e-9) >= 2:
                    print(f"  {str(th):>20} {tag_of(th):>8} "
                          f"{ratios['cdd']:>10.3f} {ratios['topb']:>11.3f}")
    print(f"  worst ratio: CDD={worst['cdd'][0]:.3f} @ {worst['cdd'][1]}, "
          f"top-|b|={worst['topb'][0]:.3f} @ {worst['topb'][1]}")


def main():
    import sys
    env = build_3d(n_levels=4, order=4)
    if "--traps-only" not in sys.argv:
        run_kappa(env)
        run_sparsity(env)
    # off-axis sensor (realistic): expect no traps
    run_traps(env, sensor_coord=None, label="off-axis (0.7,0.6,0.5)")
    # sensor on the x=y=0.5 symmetry plane
    run_traps(env, sensor_coord=(0.5, 0.5, 0.7), label="on xy-plane (0.5,0.5,0.7)")


if __name__ == "__main__":
    main()
