"""Closeout Investigation 2 -- limitation probes.

SPIKE CODE. Investigative. Tests assumptions that don't match production.

  A  Periodic vs Dirichlet BC sensitivity (1D + 2D): kappa, CDD iters, wrong-sign
  B  Non-separable D_threshold (2D + 3D sine basis): blindness encounter rate
  C  Complex source geometry (2D): figure-eight source + crescent coefficient
  D  BCOO footprint 3D 16^3 (measured) + 32^3 (extrapolated)
  E  CDD outer-iteration count distribution (percentiles), 1D + 2D sweeps
"""

from __future__ import annotations

import sys
import numpy as np

import dd_wavelets as dd
from hybrid_jacobi import precond, scaled


def _l2(W, h):
    n = np.sqrt(h * np.sum(W**2, axis=0)); n[n == 0] = 1
    return W / n[None, :]


def cdd(As, bh, levels, K, theta_D=0.5):
    N = len(bh); Lam = set(np.where(levels == levels.min())[0].tolist()); nout = 0
    while True:
        idx = np.array(sorted(Lam))
        c = np.zeros(N); c[idx] = np.linalg.solve(As[np.ix_(idx, idx)], bh[idx])
        if len(idx) >= K:
            break
        r = bh - As @ c
        mo = np.ones(N, bool); mo[idx] = False
        oi = np.where(mo)[0]; order = oi[np.argsort(-np.abs(r[oi]))]
        cs = np.cumsum(r[order]**2)
        kt = min(int(np.searchsorted(cs, (theta_D**2)*np.linalg.norm(r)**2)+1), len(order))
        if kt == 0:
            break
        Lam.update(order[:kt].tolist()); nout += 1
    return idx, c, nout


# ----------------------------------------------------------------------
# Part A -- Dirichlet vs periodic
# ----------------------------------------------------------------------
def part_A():
    print("="*78); print("PART A -- periodic vs Dirichlet BC sensitivity"); print("="*78)
    SENS = 0.30; sigma = 0.06
    # ---- 1D ----
    print("  1D DD-4 (hybrid Jacobi):")
    print(f"  {'BC':>10} {'kappa':>8} {'mean CDD out':>13} {'wrong-sign?':>12}")
    for bc in ("periodic", "dirichlet"):
        nl = 7 if bc == "periodic" else 6
        W, levels, x = dd.synthesis_matrix(nl, 2, 4, bc)
        N = W.shape[0]
        h = 1.0/N if bc == "periodic" else 1.0/(N+1)
        Wn = _l2(W, h)
        Aph = dd.laplacian_periodic(N, True) if bc == "periodic" else dd.laplacian_dirichlet(N, True)
        A = Wn.T @ Aph @ Wn; A = 0.5*(A+A.T)
        D, _ = precond(A, levels, "hybrid"); As = scaled(A, D)
        kap = np.linalg.cond(As)
        sidx = int(np.argmin(np.abs(x - SENS))); srow_h = Wn[sidx]/D
        K = N//16; nouts = []; wrong = False
        for tx in np.arange(0.05, 0.96, 0.05):
            f = np.exp(-((x - tx)/sigma)**2); bh = (h*(Wn.T @ f))/D
            c_full = np.linalg.solve(As, bh); Jf = srow_h @ c_full
            idx, c, nout = cdd(As, bh, levels, K); nouts.append(nout)
            if (srow_h @ c)*Jf < 0:
                wrong = True
        print(f"  {bc:>10} {kap:>8.1f} {np.mean(nouts):>13.1f} {str(wrong):>12}")
    # ---- 2D ----
    print("  2D DD-4 (hybrid Jacobi); periodic=isotropic Mallat, Dirichlet=tensor:")
    print(f"  {'BC':>10} {'kappa':>8} {'mean CDD out':>13} {'wrong-sign?':>12}")
    # periodic isotropic
    W2, lev2, Ns = dd.synthesis_matrix_2d_isotropic(4, 2, 4); h = 1.0/Ns
    W2n = W2/np.sqrt((h*h)*np.sum(W2**2, axis=0))[None, :]
    S = dd.laplacian_periodic(Ns, False); Mm = h*np.eye(Ns)
    Aper = W2n.T @ (np.kron(S, Mm)+np.kron(Mm, S)+np.kron(Mm, Mm)) @ W2n; Aper = 0.5*(Aper+Aper.T)
    _run_2d_bc("periodic", Aper, lev2, W2n, Ns, h, SENS, sigma)
    # Dirichlet tensor
    W1, l1, x1 = dd.synthesis_matrix(4, 2, 4, "dirichlet"); Nd = W1.shape[0]; hd = 1.0/(Nd+1)
    W1n = _l2(W1, hd); W2d = np.kron(W1n, W1n)
    lev2d = (l1[:, None] + l1[None, :]).reshape(-1)
    S1 = dd.laplacian_dirichlet(Nd, False); M1 = hd*np.eye(Nd)
    A2d = W2d.T @ (np.kron(S1, M1)+np.kron(M1, S1)+np.kron(M1, M1)) @ W2d; A2d = 0.5*(A2d+A2d.T)
    _run_2d_bc("dirichlet", A2d, lev2d, W2d, Nd, hd, SENS, sigma, x1=x1)


def _run_2d_bc(bc, A, levels, W2n, Ns, h, SENS, sigma, x1=None):
    D, _ = precond(A, levels, "hybrid"); As = scaled(A, D)
    kap = np.linalg.cond(As); N = A.shape[0]; K = N//16
    coords = (np.arange(Ns)/Ns) if x1 is None else x1
    si = int(np.argmin(np.abs(coords-SENS))); sj = int(np.argmin(np.abs(coords-0.4)))
    srow_h = W2n[si*Ns+sj]/D
    X, Y = np.meshgrid(coords, coords, indexing="ij")
    nouts = []; wrong = False
    for tx in np.arange(0.1, 0.91, 0.1):
        f = np.exp(-(((X-tx)**2+(Y-0.5)**2)/sigma**2)).reshape(-1)
        bh = ((h*h)*(W2n.T @ f))/D
        c_full = np.linalg.solve(As, bh); Jf = srow_h @ c_full
        idx, c, nout = cdd(As, bh, levels, K); nouts.append(nout)
        if (srow_h @ c)*Jf < 0:
            wrong = True
    print(f"  {bc:>10} {kap:>8.1f} {np.mean(nouts):>13.1f} {str(wrong):>12}")


# ----------------------------------------------------------------------
# Part B -- non-separable D_threshold (sine basis, coupled solve)
# ----------------------------------------------------------------------
def _sine_problem(D_dim, n_per=12, Kfrac=16, sensor=0.3):
    """D-dim sine eigenbasis Poisson; returns J, gfull, gfrozen (top-|b|)."""
    ks = np.arange(1, n_per+1)
    # full mode grid
    grids = np.meshgrid(*([ks]*D_dim), indexing="ij")
    K_modes = np.stack([g.reshape(-1) for g in grids], axis=1)  # (M, D)
    lam = np.sum((K_modes*np.pi)**2, axis=1) + 1.0
    M = K_modes.shape[0]; Kact = max(4, M//Kfrac)
    xs = sensor  # sensor coordinate (same each axis, off-centre)
    sphi = np.prod(np.sqrt(2.0)*np.sin(np.pi*K_modes*xs), axis=1)  # sensor row

    def bcoef(theta):  # separable projection (1D factors), product over axes
        # b_k = prod_i <sin(k_i pi x), exp(-((x-theta_i)/sig)^2)> ; approximate by
        # sampling 1D integral on a grid
        sig = 0.08; xg = np.linspace(0, 1, 200); w = xg[1]-xg[0]
        out = np.ones(M)
        for i in range(D_dim):
            g = np.exp(-((xg - theta[i])/sig)**2)
            B1 = np.sqrt(2.0)*np.sin(np.pi*np.outer(ks, xg)) @ g * w  # (n_per,)
            out *= B1[K_modes[:, i]-1]
        return out

    def J(theta):
        b = bcoef(theta); c = b/lam; return sphi @ c

    def grad(fn, theta, eps=1e-5):
        g = np.zeros(D_dim)
        for d in range(D_dim):
            tp = theta.copy(); tp[d] += eps; tm = theta.copy(); tm[d] -= eps
            g[d] = (fn(tp)-fn(tm))/(2*eps)
        return g

    def gfull(theta): return grad(J, theta)

    def gfrozen(theta):
        b = bcoef(theta); S = np.argsort(-np.abs(b))[:Kact]
        def Jf(th):
            bb = bcoef(th); c = np.zeros(M); c[S] = bb[S]/lam[S]; return sphi @ c
        return grad(Jf, theta)
    return J, gfull, gfrozen


def part_B():
    print("="*78); print("PART B -- non-separable blindness encounter rate"); print("="*78)
    print("  genuinely-coupled sine-basis Poisson (lambda couples axes), top-|b|")
    print(f"  {'D':>3} {'mean enc':>9} {'(separable ref ~1-1.5)':>24}")
    for D_dim, n_per in ((2, 12), (3, 7)):
        J, gfull, gfrozen = _sine_problem(D_dim, n_per)
        rng = np.random.default_rng(0); encs = []
        for _ in range(20):
            theta = rng.uniform(0.1, 0.9, D_dim); run = 0; nenc = 0; intrap = False
            for _ in range(100):
                gf = gfull(theta); gz = gfrozen(theta)
                ratio = np.linalg.norm(gz)/(np.linalg.norm(gf)+1e-30)
                if ratio < 0.7:
                    run += 1
                    if run >= 5 and not intrap:
                        nenc += 1; intrap = True
                else:
                    run = 0; intrap = False
                theta = np.clip(theta - 0.04*gz, 0.1, 0.9)
            encs.append(nenc)
        print(f"  {D_dim:>3} {np.mean(encs):>9.2f} {'':>24}")


# ----------------------------------------------------------------------
# Part C -- complex source geometry (2D)
# ----------------------------------------------------------------------
def part_C():
    print("="*78); print("PART C -- complex geometry: figure-eight source + crescent coeff"); print("="*78)
    W2, levels, Ns = dd.synthesis_matrix_2d_isotropic(5, 2, 4); h = 1.0/Ns  # Ns=64
    W2n = W2/np.sqrt((h*h)*np.sum(W2**2, axis=0))[None, :]
    coords = (np.arange(Ns)+0.5)/Ns; X, Y = np.meshgrid(coords, coords, indexing="ij")
    N = Ns*Ns; K = N//16
    sig = 0.08
    # figure-eight source: two lobes
    f = (np.exp(-(((X-0.35)**2+(Y-0.5)**2)/sig**2))
         + np.exp(-(((X-0.65)**2+(Y-0.5)**2)/sig**2))).reshape(-1)
    # crescent coefficient: big circle minus offset circle
    big = ((X-0.5)**2+(Y-0.5)**2) < 0.20**2
    cut = ((X-0.58)**2+(Y-0.5)**2) < 0.16**2
    crescent = (big & ~cut).astype(float)
    a = 1.0 + 99.0*crescent
    Aph = _varcoeff_2d(Ns, a, h)
    Aw = W2n.T @ Aph @ W2n; Aw = 0.5*(Aw+Aw.T)
    D = np.sqrt(np.abs(np.diag(Aw))); D[D == 0] = 1; As = (Aw/D[:, None])/D[None, :]
    bh = ((h*h)*(W2n.T @ f))/D
    c_ref = np.linalg.solve(As, bh); u_ref = W2n @ (c_ref/D)
    idx, c, nout = cdd(As, bh, levels, K); u_cdd = W2n @ (c/D)
    jerr = np.linalg.norm(u_cdd-u_ref)/np.linalg.norm(u_ref)
    # multi-lobe identification: fraction of active modes near each lobe centre
    cen = np.argmax(np.abs(W2n), axis=0); ci, cj = np.unravel_index(cen, (Ns, Ns))
    cx, cy = coords[ci], coords[cj]
    near_L = np.mean([(cx[i]-0.35)**2+(cy[i]-0.5)**2 < 0.12**2 for i in idx])
    near_R = np.mean([(cx[i]-0.65)**2+(cy[i]-0.5)**2 < 0.12**2 for i in idx])
    print(f"  Ns={Ns}, N={N}, k=N/16={K}")
    print(f"  CDD outer iters={nout}, J_err(relL2)={jerr:.2e}")
    print(f"  active modes near LEFT lobe={near_L*100:.0f}%, near RIGHT lobe={near_R*100:.0f}% "
          f"(both lobes covered: {near_L>0.05 and near_R>0.05})")


def _varcoeff_2d(Ns, a, h):
    N = Ns*Ns; A = np.zeros((N, N))
    def idx(i, j): return (i % Ns)*Ns + (j % Ns)
    for i in range(Ns):
        for j in range(Ns):
            k = idx(i, j); tot = 0.0
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                af = 0.5*(a[i, j] + a[(i+di) % Ns, (j+dj) % Ns])
                A[k, idx(i+di, j+dj)] -= af/h**2; tot += af/h**2
            A[k, k] = tot + 1.0
    return 0.5*(A+A.T)


# ----------------------------------------------------------------------
# Part D -- 3D BCOO footprint
# ----------------------------------------------------------------------
def part_D():
    print("="*78); print("PART D -- 3D BCOO stiffness footprint"); print("="*78)
    import os
    cache = "/tmp/closeout_3d_env.npz"
    if not os.path.exists(cache):
        print("  (3D env cache missing -- run closeout_3d.py first); using extrapolation only")
        nnz_per_row_16 = None
    else:
        A = np.load(cache)["A"]; N = A.shape[0]
        thr = 1e-12*np.abs(A).max()
        nnz = int(np.count_nonzero(np.abs(A) >= thr))
        nnz_per_row_16 = nnz/N
        # BCOO: data (f64) + indices (2 x int32 for 2D coords) per nonzero
        mem_mb = nnz*(8 + 2*4)/1e6
        print(f"  N=16^3=4096 (MEASURED): nnz={nnz} ({100*nnz/N**2:.1f}% dense), "
              f"{nnz_per_row_16:.0f} nnz/row, BCOO mem={mem_mb:.1f} MB")
    # extrapolate to 32^3 = 32768
    npr = nnz_per_row_16 if nnz_per_row_16 else 300.0
    N32 = 32**3
    nnz32 = npr * N32  # nnz/row roughly constant for a local-ish operator (grows slowly)
    mem32 = nnz32*(8+2*4)/1e6
    print(f"  N=32^3=32768 (EXTRAPOLATED, nnz/row~{npr:.0f}): nnz~{nnz32:.3e}, "
          f"BCOO mem~{mem32:.0f} MB ({mem32/1024:.1f} GB)")
    print(f"  round-6 design estimate was ~120 KB STATE (the c/mask vectors, N floats),")
    print(f"  which is separate from the OPERATOR matrix. State at 32^3: "
          f"{N32*8/1e6:.2f} MB per vector.")


# ----------------------------------------------------------------------
# Part E -- CDD iteration-count distribution
# ----------------------------------------------------------------------
def part_E():
    print("="*78); print("PART E -- CDD outer-iteration count distribution"); print("="*78)
    counts = []; tagged = []
    # 1D sweep: theta in (0,1), sigma in {0.10,0.05,0.02}
    nl = 7; W, levels, x = dd.synthesis_matrix(nl, 2, 4, "periodic"); N = W.shape[0]
    h = 1.0/N; Wn = _l2(W, h); A = Wn.T @ dd.laplacian_periodic(N, True) @ Wn; A = 0.5*(A+A.T)
    D, _ = precond(A, levels, "hybrid"); As = scaled(A, D); K = N//16
    for sigma in (0.10, 0.05, 0.02):
        for tx in np.arange(0.02, 0.99, 0.04):
            f = np.exp(-((x-tx)/sigma)**2); bh = (h*(Wn.T @ f))/D
            _, _, nout = cdd(As, bh, levels, K); counts.append(nout)
            tagged.append((nout, sigma, tx))
    # 2D sweep
    W2, lev2, Ns = dd.synthesis_matrix_2d_isotropic(4, 2, 4); h2 = 1.0/Ns
    W2n = W2/np.sqrt((h2*h2)*np.sum(W2**2, axis=0))[None, :]
    S = dd.laplacian_periodic(Ns, False); Mm = h2*np.eye(Ns)
    A2 = W2n.T @ (np.kron(S, Mm)+np.kron(Mm, S)+np.kron(Mm, Mm)) @ W2n; A2 = 0.5*(A2+A2.T)
    D2, _ = precond(A2, lev2, "hybrid"); As2 = scaled(A2, D2); K2 = (Ns*Ns)//16
    coords = np.arange(Ns)/Ns; X, Y = np.meshgrid(coords, coords, indexing="ij")
    for sigma in (0.10, 0.05, 0.02):
        for tx in np.arange(0.1, 0.91, 0.1):
            f = np.exp(-(((X-tx)**2+(Y-0.5)**2)/sigma**2)).reshape(-1)
            bh = ((h2*h2)*(W2n.T @ f))/D2
            _, _, nout = cdd(As2, bh, lev2, K2); counts.append(nout); tagged.append((nout, sigma, tx))
    counts = np.array(counts)
    print(f"  collected {len(counts)} CDD runs (1D+2D, sigma in [0.02,0.10], theta swept)")
    print(f"  outer iters: p50={np.percentile(counts,50):.0f}, "
          f"p90={np.percentile(counts,90):.0f}, p99={np.percentile(counts,99):.0f}, "
          f"max={counts.max()}")
    worst = sorted(tagged, reverse=True)[:5]
    print(f"  worst 5: " + ", ".join(f"{n}it(σ={s},θ={t:.2f})" for n, s, t in worst))
    hi = [(n, s, t) for n, s, t in tagged if n > 40]
    print(f"  runs >40 iters: {len(hi)}" + (f" e.g. {hi[:3]}" if hi else " (none)"))


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "all"
    for p, fn in (("A", part_A), ("B", part_B), ("C", part_C), ("D", part_D), ("E", part_E)):
        if sel in (p, "all"):
            fn()
