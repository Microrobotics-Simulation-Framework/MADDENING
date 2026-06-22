"""Post-gate improvements / clever-workaround exploration.

SPIKE CODE. Investigative. Two questions the gates did not address but that
materially affect the production design:

  (1) SUBMATRIX conditioning. The full-operator kappa is what §2/§4 measured,
      but CDD only ever inverts the frozen K x K submatrix A_{Lambda,Lambda}.
      Does that submatrix stay well-conditioned as Lambda adapts? If not, the
      inner GMRES/CG (ift_linear_solve) degrades regardless of full-op kappa.

  (2) CHEAP diagonal. Jacobi is the best preconditioner found, but it needs
      diag(A_wave). Plan Hyp C claims the diagonal is analytically cheap
      (O(N), no N matvecs). Verify: does the cheap diagonal (one matvec per
      level via probing, or the analytic value) match the exact diagonal, and
      does Jacobi-from-cheap-diag conditioning match Jacobi-from-exact?

Also tries (3) a BPX-style additive 2-level preconditioner (matrix-free,
no diagonal needed) as an alternative.
"""

from __future__ import annotations

import numpy as np

import dd_wavelets as dd


def _l2(W, h):
    n = np.sqrt(h * np.sum(W ** 2, axis=0)); n[n == 0] = 1
    return W / n[None, :]


def build_1d(n_levels=6, order=4):
    W, levels, x = dd.synthesis_matrix(n_levels, n_coarse=2, order=order,
                                       boundary="dirichlet")
    N = W.shape[0]; h = 1.0 / (N + 1)
    Wn = _l2(W, h)
    A = Wn.T @ dd.laplacian_dirichlet(N, mass=True) @ Wn
    return 0.5 * (A + A.T), levels, Wn, x, h, N


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


def q1_submatrix_conditioning():
    print("=" * 70)
    print("(1) SUBMATRIX conditioning: kappa(A_LL) for CDD-selected Lambda")
    print("    (the matrix the inner solve actually inverts)")
    A, levels, Wn, x, h, N = build_1d()
    D_dk = 2.0 ** levels.astype(float)
    D_jac = np.sqrt(np.abs(np.diag(A)))
    f = np.exp(-((x - 0.3) / 0.05) ** 2)
    b = h * (Wn.T @ f)
    print(f"  N={N}; full-op kappa: DK={np.linalg.cond((A/D_dk[:,None])/D_dk[None,:]):.1f}, "
          f"Jacobi={np.linalg.cond((A/D_jac[:,None])/D_jac[None,:]):.1f}")
    print(f"  {'K':>5} {'kap(A_LL) DK':>14} {'kap(A_LL) Jac':>15} {'kap unscaled':>14}")
    for K in (N // 16, N // 8, N // 4, N // 2):
        AdK = (A / D_dk[:, None]) / D_dk[None, :]
        bh = b / D_dk
        idx = cdd_idx(AdK, bh, levels, K)
        kdk = np.linalg.cond(AdK[np.ix_(idx, idx)])
        AjK = (A / D_jac[:, None]) / D_jac[None, :]
        idxj = cdd_idx(AjK, b / D_jac, levels, K)
        kjac = np.linalg.cond(AjK[np.ix_(idxj, idxj)])
        kun = np.linalg.cond(A[np.ix_(idx, idx)])
        print(f"  {K:>5} {kdk:>14.1f} {kjac:>15.1f} {kun:>14.3e}")


def q2_cheap_diagonal():
    print("=" * 70)
    print("(2) CHEAP diagonal: does a probed/cheap diag match exact diag(A_wave)?")
    A, levels, Wn, x, h, N = build_1d()
    exact = np.diag(A)
    # 'Cheap' probe: diag via Hutchinson-style level probing is overkill here;
    # the genuinely cheap route for an FD operator is diag(A_wave)_l = sum_i
    # (Wn[i,l])^2 * A_phys_ii + cross terms. We approximate the operator
    # diagonal by Wn.T diag-action: d_l = (Wn[:,l]) . (A_phys @ Wn[:,l]).
    # That is exactly diag(A) -- so instead test the LEVEL-CONSTANT analytic
    # model diag ~ c * 2^{2 level} (the DK prediction) vs exact, per level.
    print("  exact diag(A) grouped by level vs DK model c*2^{2j}:")
    print(f"  {'level':>6} {'mean diag':>12} {'std/mean':>10} {'ratio to 2^2j':>14}")
    base = None
    for lvl in sorted(set(levels.tolist())):
        d = exact[levels == lvl]
        m = d.mean()
        if base is None:
            base = m / (2.0 ** (2 * lvl))
        ratio = m / (base * 2.0 ** (2 * lvl))
        print(f"  {lvl:>6} {m:>12.4e} {d.std()/m:>10.3f} {ratio:>14.3f}")
    print("  -> if std/mean is small per level, diag is ~level-constant and a")
    print("     single per-level analytic value suffices (O(#levels), not O(N)).")


def q3_bpx():
    print("=" * 70)
    print("(3) BPX-style additive preconditioner (matrix-free, no diagonal)")
    A, levels, Wn, x, h, N = build_1d()
    # BPX additive: M^-1 = sum_levels 2^{-2 level?}... in the wavelet basis the
    # additive Schwarz / BPX with L2-normalised wavelets reduces to the DK
    # diagonal D=2^{2*?}. We instead test the additive scaling D_bpx where the
    # block per level is scaled by its mean diagonal (cheap level-constant
    # Jacobi) -- a middle ground between DK and full Jacobi.
    exact = np.diag(A)
    Dlevel = np.zeros(N)
    for lvl in sorted(set(levels.tolist())):
        Dlevel[levels == lvl] = np.sqrt(exact[levels == lvl].mean())
    Dfull = np.sqrt(np.abs(exact))
    D_dk = 2.0 ** levels.astype(float)
    def k(D): return np.linalg.cond((A / D[:, None]) / D[None, :])
    print(f"  N={N}")
    print(f"  DK 2^j            : {k(D_dk):.1f}")
    print(f"  level-Jacobi(cheap): {k(Dlevel):.1f}  <- per-level constant, O(#lvl)")
    print(f"  full Jacobi        : {k(Dfull):.1f}")


if __name__ == "__main__":
    q1_submatrix_conditioning()
    q2_cheap_diagonal()
    q3_bpx()
