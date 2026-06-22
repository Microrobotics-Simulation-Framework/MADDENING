"""Investigation 4 -- discontinuous coefficient (swimmer-body / Brinkman).

SPIKE CODE. Investigative.

Production MIME uses immersed-boundary / Brinkman penalisation: a(x) jumps from
mu_fluid to mu_penalised at the swimmer surface. -div(a(x) grad u)=f with
discontinuous a is in H^1 but the gradient jump changes wavelet decay near the
interface. Does CDD/Jacobi still work? Does Jacobi (adapts to the actual
diagonal) beat theory-DK (tuned for constant-coeff H^1) here?

Part A -- 1D: -(a u')'=f, a=1+9*1_{x>0.5} (10x jump), Dirichlet, f=sin(pi x).
Part B -- 2D: -div(a grad u)+u=f, circular inclusion a=1+99*1_{|x-xc|<r}
          (100x penalisation), moving centre swept on a circle.
"""

from __future__ import annotations

import numpy as np

import dd_wavelets as dd


def _l2(W, h):
    n = np.sqrt(h * np.sum(W ** 2, axis=0)); n[n == 0] = 1
    return W / n[None, :]


def part_A():
    print("=" * 78)
    print("PART A -- 1D discontinuous coefficient, a=1+9*1_{x>0.5}")
    print("=" * 78)
    n_levels = 6
    W, levels, x = dd.synthesis_matrix(n_levels, n_coarse=2, order=4, boundary="dirichlet")
    N = W.shape[0]
    h = 1.0 / (N + 1)
    Wn = _l2(W, h)
    # conservative variable-coeff stiffness, Dirichlet
    def a_of(xx): return 1.0 + 9.0 * (xx > 0.5)
    xfull = np.concatenate(([0.0], x, [1.0]))
    A_phys = np.zeros((N, N))
    for i in range(N):
        xm = 0.5 * (xfull[i] + xfull[i + 1])   # left face (i-1/2 in interior idx)
        xp = 0.5 * (xfull[i + 1] + xfull[i + 2])
        am, ap = a_of(xm), a_of(xp)
        A_phys[i, i] = (am + ap) / h**2
        if i > 0:
            A_phys[i, i - 1] = -am / h**2
        if i < N - 1:
            A_phys[i, i + 1] = -ap / h**2
    A = Wn.T @ A_phys @ Wn
    A = 0.5 * (A + A.T)
    f = np.sin(np.pi * x)
    b = h * (Wn.T @ f)
    u_full = np.linalg.solve(A_phys, f * 0 + np.sin(np.pi * x))  # ref physical solve
    # solve in wavelet coords (full) for reference functional
    c_ref = np.linalg.solve(A, b)
    sidx = int(np.argmin(np.abs(x - 0.30)))
    J_ref = Wn[sidx] @ c_ref

    lev = levels.astype(float)
    scalings = {
        "DK 2^j": 2.0 ** lev,
        "Besov 2^{j/2}": 2.0 ** (lev / 2),
        "Jacobi": np.sqrt(np.abs(np.diag(A))),
    }
    K = N // 16
    centres = x[np.argmax(np.abs(Wn), axis=0)]
    print(f"  N={N}, K=N/16={K}, jump at x=0.5")
    print(f"  {'scaling':>14} {'kappa':>9} {'CDD out':>8} {'%near 0.5':>10} {'J_err':>9}")
    for name, D in scalings.items():
        D = D.copy(); D[D == 0] = 1
        As = (A / D[:, None]) / D[None, :]
        bh = b / D
        kap = np.linalg.cond(As)
        # CDD
        coarse = np.where(levels == levels.min())[0]
        Lam = set(coarse.tolist()); nout = 0
        while True:
            idx = np.array(sorted(Lam))
            if len(idx) >= K: break
            c = np.zeros(N); c[idx] = np.linalg.solve(As[np.ix_(idx, idx)], bh[idx])
            r = bh - As @ c
            mo = np.ones(N, bool); mo[idx] = False
            oi = np.where(mo)[0]; order = oi[np.argsort(-np.abs(r[oi]))]
            cs = np.cumsum(r[order]**2)
            kt = min(int(np.searchsorted(cs, 0.25*np.linalg.norm(r)**2)+1), len(order))
            if kt == 0: break
            Lam.update(order[:kt].tolist()); nout += 1
        c = np.zeros(N); c[idx] = np.linalg.solve(As[np.ix_(idx, idx)], bh[idx])
        J_cdd = Wn[sidx] @ (c / D)
        jerr = abs(J_cdd - J_ref) / (abs(J_ref) + 1e-30)
        near = np.mean([abs(centres[i] - 0.5) < 0.1 for i in idx])
        print(f"  {name:>14} {kap:>9.1f} {nout:>8d} {near:>10.2f} {jerr:>9.2e}")
    # compare to smooth-coeff case: how many modes does CDD need?
    print("  (smooth-coeff a=1 reference for mode-count comparison:)")
    A0p = np.zeros((N, N))
    for i in range(N):
        A0p[i, i] = 2 / h**2
        if i > 0: A0p[i, i-1] = -1/h**2
        if i < N-1: A0p[i, i+1] = -1/h**2
    A0 = Wn.T @ A0p @ Wn; A0 = 0.5*(A0+A0.T)
    D0 = np.sqrt(np.abs(np.diag(A0)))
    print(f"    Jacobi kappa smooth={np.linalg.cond((A0/D0[:,None])/D0[None,:]):.1f} "
          f"vs jump={np.linalg.cond((A/scalings['Jacobi'][:,None])/scalings['Jacobi'][None,:]):.1f}")


def build_2d_varcoeff(Nside, a_field, mass=1.0):
    """Conservative -div(a grad)+mass, periodic, on Nside^2."""
    h = 1.0 / Nside
    N = Nside * Nside
    a = a_field  # (Nside,Nside)
    A = np.zeros((N, N))
    def idx(i, j): return (i % Nside) * Nside + (j % Nside)
    for i in range(Nside):
        for j in range(Nside):
            k = idx(i, j)
            aE = 0.5 * (a[i, j] + a[(i+1) % Nside, j])
            aW = 0.5 * (a[i, j] + a[(i-1) % Nside, j])
            aN = 0.5 * (a[i, j] + a[i, (j+1) % Nside])
            aS = 0.5 * (a[i, j] + a[i, (j-1) % Nside])
            A[k, k] = (aE + aW + aN + aS) / h**2 + mass
            A[k, idx(i+1, j)] -= aE / h**2
            A[k, idx(i-1, j)] -= aW / h**2
            A[k, idx(i, j+1)] -= aN / h**2
            A[k, idx(i, j-1)] -= aS / h**2
    return 0.5 * (A + A.T)


def part_B():
    print("=" * 78)
    print("PART B -- 2D circular inclusion (Brinkman swimmer-body proxy)")
    print("=" * 78)
    nl = 4
    W2, levels, Nside = dd.synthesis_matrix_2d_isotropic(nl, 2, 4)  # Nside=32
    h = 1.0 / Nside
    W2n = W2 / np.sqrt((h*h) * np.sum(W2**2, axis=0))[None, :]
    coords = (np.arange(Nside) + 0.5) / Nside
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    N = Nside * Nside
    r = 0.15
    f = np.exp(-(((X - 0.5)**2 + (Y - 0.5)**2) / 0.2**2)).reshape(-1)
    bvec = (h*h) * (W2n.T @ f)
    cen = np.argmax(np.abs(W2n), axis=0)
    ci, cj = np.unravel_index(cen, (Nside, Nside))
    cx, cy = coords[ci], coords[cj]
    print(f"  Nside={Nside}, N={N}, inclusion r={r}, a=1+99*1_inside, Jacobi precond")
    print(f"  {'theta':>6} {'xc':>10} {'CDD out':>8} {'%@bndry(N/8)':>13} {'J_err(N/16)':>12}")
    for th in (0.0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi):
        xc = (0.5 + 0.2*np.cos(th), 0.5 + 0.2*np.sin(th))
        inside = ((X - xc[0])**2 + (Y - xc[1])**2 < r**2).astype(float)
        a_field = 1.0 + 99.0 * inside
        A = build_2d_varcoeff(Nside, a_field)
        Aw = W2n.T @ A @ W2n; Aw = 0.5*(Aw+Aw.T)
        D = np.sqrt(np.abs(np.diag(Aw))); D[D == 0] = 1
        As = (Aw / D[:, None]) / D[None, :]
        bh = bvec / D
        c_ref = np.linalg.solve(As, bh)
        u_ref = W2n @ (c_ref / D)
        coarse = np.where(levels == levels.min())[0]
        # distance of each mode centre to the circle boundary
        dist_bndry = np.abs(np.sqrt((cx - xc[0])**2 + (cy - xc[1])**2) - r)
        res = {}
        for Kfrac in (8, 16):
            K = max(8, N // Kfrac)
            Lam = set(coarse.tolist()); nout = 0
            while True:
                idx = np.array(sorted(Lam))
                if len(idx) >= K: break
                c = np.zeros(N); c[idx] = np.linalg.solve(As[np.ix_(idx, idx)], bh[idx])
                rr = bh - As @ c
                mo = np.ones(N, bool); mo[idx] = False
                oi = np.where(mo)[0]; order = oi[np.argsort(-np.abs(rr[oi]))]
                cs = np.cumsum(rr[order]**2)
                kt = min(int(np.searchsorted(cs, 0.25*np.linalg.norm(rr)**2)+1), len(order))
                if kt == 0: break
                Lam.update(order[:kt].tolist()); nout += 1
            c = np.zeros(N); c[idx] = np.linalg.solve(As[np.ix_(idx, idx)], bh[idx])
            u_cdd = W2n @ (c / D)
            jerr = np.linalg.norm(u_cdd - u_ref) / np.linalg.norm(u_ref)
            atbndry = np.mean([dist_bndry[i] < 1.5 * h for i in idx])
            res[Kfrac] = (nout, atbndry, jerr)
        print(f"  {th:>6.2f} ({xc[0]:.2f},{xc[1]:.2f}) {res[16][0]:>8d} "
              f"{res[8][1]*100:>12.0f}% {res[16][2]:>12.2e}")


if __name__ == "__main__":
    part_A()
    part_B()
