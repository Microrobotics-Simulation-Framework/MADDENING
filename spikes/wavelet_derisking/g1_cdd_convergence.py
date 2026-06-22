"""Gate 1, §2 Hyp A (part 2) -- CDD convergence and rolling comparison.

SPIKE CODE. Investigative.

Two questions from the plan:
  1. Does CDD with Doerfler theta_D=0.5 converge in <= 20 outer iterations
     on the smooth trajectory (and at near-sharp sigma=0.02)?
  2. Does CDD-with-DD beat rolling-top-|c_prev| by >= 10% mean J_err on
     both smooth (sigma>=0.05) and near-sharp (sigma=0.02) sources?

We work in Dahmen-Kunoth-scaled coordinates (well-conditioned A_hat), where
CDD's residual-greedy marking is near-optimal. DD-4 Dirichlet basis.
Trajectory theta(t) = 0.3 + 0.3 sin(2 pi t / T), T=30.
"""

from __future__ import annotations

import numpy as np

import dd_wavelets as dd

SENSOR_X = 0.30


def _l2_normalise(W, h):
    norms = np.sqrt(h * np.sum(W ** 2, axis=0))
    norms[norms == 0] = 1.0
    return W / norms[None, :]


def setup(n_levels=6, order=4):
    W, levels, x = dd.synthesis_matrix(n_levels, n_coarse=2, order=order,
                                       boundary="dirichlet")
    N = W.shape[0]
    h = 1.0 / (N + 1)
    Wn = _l2_normalise(W, h)
    A = Wn.T @ dd.laplacian_dirichlet(N, mass=True) @ Wn
    A = 0.5 * (A + A.T)
    D = 2.0 ** levels.astype(float)
    Ahat = (A / D[:, None]) / D[None, :]
    sidx = int(np.argmin(np.abs(x - SENSOR_X)))
    srow_hat = Wn[sidx] / D
    coarse = np.where(levels == levels.min())[0]
    return dict(Wn=Wn, levels=levels, x=x, h=h, D=D, Ahat=Ahat, A_raw=A,
                sidx=sidx, srow_hat=srow_hat, coarse=coarse, N=N)


def rescale(env, which):
    """Return a shallow env copy with Ahat/srow_hat/D rebuilt for a scaling.

    which: 'h1' (Dahmen-Kunoth 2^|l|), 'besov' (B^1_{1,1} 1D: 2^{|l|/2}),
    'jacobi' (sqrt diag A), 'none'.
    """
    lev = env["levels"].astype(float)
    A = env["A_raw"]
    if which == "h1":
        D = 2.0 ** lev
    elif which == "besov":
        D = 2.0 ** (lev / 2.0)
    elif which == "jacobi":
        D = np.sqrt(np.abs(np.diag(A)))
    elif which == "none":
        D = np.ones_like(lev)
    else:
        raise ValueError(which)
    Ahat = (A / D[:, None]) / D[None, :]
    e = dict(env)
    e["D"] = D
    e["Ahat"] = Ahat
    e["srow_hat"] = env["Wn"][env["sidx"]] / D
    return e


def source(env, theta, sigma, kind="gauss"):
    x = env["x"]
    if kind == "gauss":
        return np.exp(-((x - theta) / sigma) ** 2)
    if kind == "step":
        # moving Heaviside: solution of (-D+I)u=1_{x<theta} has a KINK
        # (in H^1 but not H^2) -- the sharp-interface stress test
        return (x < theta).astype(float)
    raise ValueError(kind)


def bhat(env, theta, sigma, kind="gauss"):
    f = source(env, theta, sigma, kind)
    b = env["h"] * (env["Wn"].T @ f)
    return b / env["D"]


def solve_on(Ahat, bh, idx):
    c = np.zeros_like(bh)
    c[idx] = np.linalg.solve(Ahat[np.ix_(idx, idx)], bh[idx])
    return c


def cdd_loop(env, bh, theta_D=0.5, tol=1e-3, Kmax=None, Kstop=None):
    """CDD SOLVE-ESTIMATE-MARK-REFINE loop. Returns (idx, c, n_outer)."""
    Ahat = env["Ahat"]
    N = env["N"]
    Kmax = Kmax or N
    Lam = set(env["coarse"].tolist())
    n_outer = 0
    bnorm = np.linalg.norm(bh) + 1e-30
    while True:
        idx = np.array(sorted(Lam))
        c = solve_on(Ahat, bh, idx)
        r = bh - Ahat @ c
        rnorm = np.linalg.norm(r)
        if rnorm <= tol * bnorm:
            break
        if len(idx) >= Kmax:
            break
        if Kstop is not None and len(idx) >= Kstop:
            break
        # Doerfler bulk marking on indices outside Lambda
        mask_out = np.ones(N, dtype=bool)
        mask_out[idx] = False
        out_idx = np.where(mask_out)[0]
        order = out_idx[np.argsort(-np.abs(r[out_idx]))]
        r2 = r[order] ** 2
        target = (theta_D ** 2) * (rnorm ** 2)
        csum = np.cumsum(r2)
        ktake = int(np.searchsorted(csum, target) + 1)
        ktake = min(ktake, len(order))
        Lam.update(order[:ktake].tolist())
        n_outer += 1
        if n_outer > 200:
            break
    return idx, c, n_outer


def J_of(env, c):
    return env["srow_hat"] @ c


def trajectory(env, sigma, T=30, nsteps=30, K=None):
    """Compare CDD / rolling / oracle at a fixed budget K over a theta sweep."""
    if K is None:
        K = env["N"] // 8
    ts = np.arange(nsteps)
    thetas = 0.3 + 0.3 * np.sin(2 * np.pi * ts / T)
    errs = {"cdd": [], "rolling": [], "oracle": []}
    prev_c = None
    for theta in thetas:
        bh = bhat(env, theta, sigma)
        c_full = np.linalg.solve(env["Ahat"], bh)
        J_full = J_of(env, c_full)
        denom = abs(J_full) + 1e-30

        # CDD at fixed budget K
        idxc, cc, _ = cdd_loop(env, bh, Kstop=K)
        errs["cdd"].append(abs(J_of(env, cc) - J_full) / denom)

        # rolling top-|c_prev| (first step: top-|b_hat|)
        if prev_c is None:
            sr = np.argsort(-np.abs(bh))[:K]
        else:
            sr = np.argsort(-np.abs(prev_c))[:K]
        sr = np.union1d(sr, env["coarse"])  # keep coarse (fair: CDD also does)
        c_roll = solve_on(env["Ahat"], bh, sr)
        errs["rolling"].append(abs(J_of(env, c_roll) - J_full) / denom)
        prev_c = c_roll

        # oracle top-|c_full|
        so = np.argsort(-np.abs(c_full))[:K]
        c_or = solve_on(env["Ahat"], bh, so)
        errs["oracle"].append(abs(J_of(env, c_or) - J_full) / denom)

    return {k: np.mean(v) for k, v in errs.items()}


def main():
    env = setup(n_levels=6, order=4)
    print(f"DD-4 Dirichlet, N={env['N']}, sensor x≈{env['x'][env['sidx']]:.3f}")
    print()
    print("--- CDD convergence (Doerfler theta_D=0.5, tol=1e-3 on scaled resid) ---")
    print(f"{'sigma':>7} {'n_outer':>8} {'|Lambda|':>9} {'|L|/N':>7} {'J_err':>10}")
    for sigma in (0.10, 0.05, 0.02):
        bh = bhat(env, 0.45, sigma)
        idx, c, nout = cdd_loop(env, bh, tol=1e-3)
        c_full = np.linalg.solve(env["Ahat"], bh)
        jerr = abs(J_of(env, c) - J_of(env, c_full)) / (abs(J_of(env, c_full)) + 1e-30)
        print(f"{sigma:>7.2f} {nout:>8d} {len(idx):>9d} {len(idx)/env['N']:>7.3f} {jerr:>10.2e}")

    print()
    print("--- Trajectory mean J_err at fixed budget K=N/8, theta(t)=0.3+0.3sin ---")
    print(f"{'sigma':>7} {'CDD':>10} {'rolling':>10} {'oracle':>10} {'CDD vs roll':>12}")
    for sigma in (0.10, 0.05, 0.02):
        r = trajectory(env, sigma, K=env["N"] // 8)
        impr = (r["rolling"] - r["cdd"]) / (r["rolling"] + 1e-30) * 100
        print(f"{sigma:>7.2f} {r['cdd']:>10.2e} {r['rolling']:>10.2e} "
              f"{r['oracle']:>10.2e} {impr:>11.1f}%")

    print()
    print("--- SHARP-INTERFACE stress test: step source (kink solution, H^1 not H^2) ---")
    print("    CDD convergence (tol=1e-3) under competing scalings")
    print(f"{'scaling':>8} {'n_outer':>8} {'|Lambda|':>9} {'|L|/N':>7} {'J_err':>10}")
    for which in ("h1", "besov", "jacobi", "none"):
        e = rescale(env, which)
        bh = bhat(e, 0.45, 0.02, kind="step")
        idx, c, nout = cdd_loop(e, bh, tol=1e-3, Kmax=e["N"] // 2)
        c_full = np.linalg.solve(e["Ahat"], bh)
        jerr = abs(J_of(e, c) - J_of(e, c_full)) / (abs(J_of(e, c_full)) + 1e-30)
        print(f"{which:>8} {nout:>8d} {len(idx):>9d} {len(idx)/e['N']:>7.3f} {jerr:>10.2e}")


if __name__ == "__main__":
    main()
