"""Investigation 1 -- Level-Jacobi hybrid: the production preconditioner.

SPIKE CODE. Investigative.

Four preconditioners (diagonal scalings D; we solve the symmetrically scaled
operator A_hat = D^-1 A D^-1):
  full   : D_ii = sqrt(a_ii) for every entry        (assemble N scalars)
  level  : D_ii = sqrt(mean diag at level |lambda_i|)(assemble O(#levels))
  hybrid : per-entry at level 0, level-mean for finer (assemble N_coarse+#lvl)
  dk     : D_ii = 2^{t |lambda_i|}, matrix-free      (assemble 0)

Part A: full-op kappa, submatrix kappa at k=N/16, GMRES iters, assembly count,
        across N in 1D {256,1024,4096,16384} and 2D {16^2,32^2,64^2,128^2}.
Part B: CDD outer-iteration count + total inner GMRES iters on the smooth
        trajectory, under each preconditioner.
Part C: crossover-N model (in the memo).
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import gmres, LinearOperator

import dd_wavelets as dd

T_ORDER = 4  # DD-4


def _l2(W, h):
    n = np.sqrt(h * np.sum(W ** 2, axis=0)); n[n == 0] = 1
    return W / n[None, :]


def build_1d(N):
    n_levels = int(round(np.log2(N / 2)))
    W, levels, x = dd.synthesis_matrix(n_levels, n_coarse=2, order=T_ORDER,
                                       boundary="periodic")
    assert W.shape[0] == N, (W.shape[0], N)
    h = 1.0 / N
    Wn = _l2(W, h)
    A = Wn.T @ dd.laplacian_periodic(N, mass=True) @ Wn
    return 0.5 * (A + A.T), levels


def build_2d(Nside):
    n_levels = int(round(np.log2(Nside / 2)))
    W2, levels, n = dd.synthesis_matrix_2d_isotropic(n_levels, 2, T_ORDER)
    assert n == Nside, (n, Nside)
    h = 1.0 / Nside
    norms = np.sqrt((h * h) * np.sum(W2 ** 2, axis=0)); norms[norms == 0] = 1
    W2n = W2 / norms[None, :]
    S = dd.laplacian_periodic(Nside, mass=False)
    Mm = h * np.eye(Nside)
    A_phys = np.kron(S, Mm) + np.kron(Mm, S) + np.kron(Mm, Mm)
    A = W2n.T @ A_phys @ W2n
    return 0.5 * (A + A.T), levels


def precond(A, levels, kind):
    """Return (D, n_assemble_scalars)."""
    diag = np.abs(np.diag(A))
    lev = levels.astype(int)
    uniq = sorted(set(lev.tolist()))
    if kind == "full":
        return np.sqrt(diag), len(diag)
    if kind == "level":
        D = np.zeros_like(diag)
        for l in uniq:
            D[lev == l] = np.sqrt(diag[lev == l].mean())
        return D, len(uniq)
    if kind == "hybrid":
        D = np.zeros_like(diag)
        l0 = uniq[0]
        D[lev == l0] = np.sqrt(diag[lev == l0])          # per-entry coarse
        n_coarse = int((lev == l0).sum())
        for l in uniq[1:]:
            D[lev == l] = np.sqrt(diag[lev == l].mean())  # level-mean fine
        return D, n_coarse + (len(uniq) - 1)
    if kind == "dk":
        return 2.0 ** (1.0 * lev), 0   # t=1 Laplacian
    raise ValueError(kind)


def scaled(A, D):
    Di = 1.0 / D
    return (Di[:, None] * A) * Di[None, :]


def kappa(M, n_extreme_only=False):
    if M.shape[0] <= 4096 and not n_extreme_only:
        ev = sla.eigvalsh(M)
        return ev[-1] / max(ev[0], 1e-30)
    from scipy.sparse.linalg import eigsh
    lo = eigsh(M, k=1, which="SA", return_eigenvectors=False, maxiter=5000)[0]
    hi = eigsh(M, k=1, which="LA", return_eigenvectors=False, maxiter=5000)[0]
    return hi / max(lo, 1e-30)


def gmres_iters(Ahat, rtol=1e-6):
    N = Ahat.shape[0]
    rng = np.random.default_rng(0)
    b = rng.standard_normal(N)
    cnt = {"n": 0}
    def cb(_): cnt["n"] += 1
    op = LinearOperator((N, N), matvec=lambda v: Ahat @ v)
    gmres(op, b, rtol=rtol, restart=min(N, 100), maxiter=2000,
          callback=cb, callback_type="pr_norm")
    return cnt["n"]


def cdd_idx(Ahat, bh, levels, K, theta_D=0.5):
    N = len(bh)
    Lam = set(np.where(levels == levels.min())[0].tolist())
    while True:
        idx = np.array(sorted(Lam))
        if len(idx) >= K:
            return idx
        c = np.zeros(N); c[idx] = np.linalg.solve(Ahat[np.ix_(idx, idx)], bh[idx])
        r = bh - Ahat @ c
        mo = np.ones(N, bool); mo[idx] = False
        oi = np.where(mo)[0]; order = oi[np.argsort(-np.abs(r[oi]))]
        cs = np.cumsum(r[order] ** 2)
        kt = min(int(np.searchsorted(cs, (theta_D ** 2) * np.linalg.norm(r) ** 2) + 1), len(order))
        if kt == 0:
            return idx
        Lam.update(order[:kt].tolist())


def part_A(dim, sizes):
    print(f"\n{'='*78}\nPART A -- {dim} preconditioner metrics (DD-4)\n{'='*78}")
    kinds = ["full", "level", "hybrid", "dk"]
    for N in sizes:
        t0 = time.time()
        A, levels = build_1d(N) if dim == "1D" else build_2d(int(round(N ** 0.5)))
        K = max(4, N // 16)
        rng = np.random.default_rng(1)
        b = rng.standard_normal(N)
        print(f"\n  N={N}  (build {time.time()-t0:.1f}s)")
        print(f"  {'precond':>7} {'kap_full':>9} {'kap_AΛΛ':>9} {'gmres':>6} {'assemble':>9}")
        for kind in kinds:
            D, nasm = precond(A, levels, kind)
            Ah = scaled(A, D)
            kf = kappa(Ah)
            idx = cdd_idx(Ah, b / D, levels, K)
            ksub = kappa(Ah[np.ix_(idx, idx)])
            gi = gmres_iters(Ah)
            print(f"  {kind:>7} {kf:>9.2f} {ksub:>9.2f} {gi:>6d} {nasm:>9d}")


def part_B(dim, N):
    print(f"\n{'='*78}\nPART B -- {dim} CDD trajectory under each preconditioner (N={N})\n{'='*78}")
    A, levels = build_1d(N) if dim == "1D" else build_2d(int(round(N ** 0.5)))
    K = max(4, N // 8)
    # build a coordinate grid for the source
    if dim == "1D":
        x = np.arange(N) / N
        def src(theta): return np.exp(-((x - theta) / 0.06) ** 2)
        Wn_for = None  # we project via the same basis used to build A? Use identity-ish
    # We need b(theta) in the wavelet basis. Rebuild Wn to project the source.
    # (Re-derive Wn quickly.)
    if dim == "1D":
        n_levels = int(round(np.log2(N / 2)))
        W, lv, xx = dd.synthesis_matrix(n_levels, 2, T_ORDER, "periodic")
        h = 1.0 / N; Wn = _l2(W, h)
        def bvec(theta):
            f = np.exp(-((xx - theta) / 0.06) ** 2)
            return h * (Wn.T @ f)
        srow = Wn[int(np.argmin(np.abs(xx - 0.30)))]
    else:
        Nside = int(round(N ** 0.5))
        n_levels = int(round(np.log2(Nside / 2)))
        W2, lv, n = dd.synthesis_matrix_2d_isotropic(n_levels, 2, T_ORDER)
        h = 1.0 / Nside
        norms = np.sqrt((h*h) * np.sum(W2 ** 2, axis=0)); norms[norms == 0] = 1
        W2n = W2 / norms[None, :]
        coords = np.arange(Nside) / Nside
        X, Y = np.meshgrid(coords, coords, indexing="ij")
        def bvec(theta):
            f = np.exp(-(((X - theta) ** 2 + (Y - 0.5) ** 2) / 0.10 ** 2)).reshape(-1)
            return (h * h) * (W2n.T @ f)
        sx = int(np.argmin(np.abs(coords - 0.70))); sy = int(np.argmin(np.abs(coords - 0.60)))
        srow = W2n[sx * Nside + sy]

    thetas = 0.3 + 0.3 * np.sin(2 * np.pi * np.arange(12) / 12)
    print(f"  {'precond':>7} {'mean_nout':>10} {'peak_nout':>10} {'mean_Jerr':>11} "
          f"{'tot_inner':>10}")
    for kind in ["full", "level", "hybrid", "dk"]:
        D, _ = precond(A, levels, kind)
        Ah = scaled(A, D)
        nouts, jerrs, inners = [], [], 0
        for theta in thetas:
            bh = bvec(theta) / D
            c_full = np.linalg.solve(Ah, bh)
            Jf = srow @ (c_full / D)
            # CDD loop counting outer iters + inner gmres per outer
            Lam = set(np.where(levels == levels.min())[0].tolist())
            nout = 0
            while True:
                idx = np.array(sorted(Lam))
                # inner solve via gmres on the submatrix (count its iters)
                sub = Ah[np.ix_(idx, idx)]
                cnt = {"n": 0}
                sol, _info = gmres(sub, bh[idx], rtol=1e-6,
                                   restart=min(len(idx), 100), maxiter=2000,
                                   callback=lambda _: cnt.__setitem__("n", cnt["n"]+1),
                                   callback_type="pr_norm")
                inners += cnt["n"]
                c = np.zeros(N); c[idx] = sol
                if len(idx) >= K:
                    break
                r = bh - Ah @ c
                mo = np.ones(N, bool); mo[idx] = False
                oi = np.where(mo)[0]; order = oi[np.argsort(-np.abs(r[oi]))]
                cs = np.cumsum(r[order] ** 2)
                kt = min(int(np.searchsorted(cs, 0.25 * np.linalg.norm(r) ** 2) + 1), len(order))
                if kt == 0:
                    break
                Lam.update(order[:kt].tolist())
                nout += 1
            Jfr = srow @ (c / D)
            nouts.append(nout); jerrs.append(abs(Jfr - Jf) / (abs(Jf) + 1e-30))
        print(f"  {kind:>7} {np.mean(nouts):>10.1f} {np.max(nouts):>10.0f} "
              f"{np.mean(jerrs):>11.2e} {inners:>10d}")


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "all"
    big = "--big" in sys.argv
    if sel in ("A1", "all"):
        part_A("1D", [256, 1024, 4096] + ([16384] if big else []))
    if sel in ("A2", "all"):
        part_A("2D", [256, 1024, 4096] + ([16384] if big else []))
    if sel in ("B1", "all"):
        part_B("1D", 4096)
    if sel in ("B2", "all"):
        part_B("2D", 4096)
