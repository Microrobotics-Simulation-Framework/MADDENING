"""Closeout Investigation 1 -- 3D completeness battery.

SPIKE CODE. Investigative. Confirms the spike's 1D/2D conclusions extend to 3D.

All parts: isotropic Mallat DD-4, N=16^3=4096 (gate-2 grid), hybrid Jacobi.
BCs: PERIODIC (flagged) -- the isotropic 3D Dirichlet wavelet basis was not
built in the spike; building boundary-adapted 3D wavelets is disproportionate
for a closeout. Dirichlet BC sensitivity is probed separately in 1D/2D in
limitation_probes.py Part A.

Parts:
  A  CDD trajectory (CDD vs rolling vs oracle), sigma=0.10 and 0.02
  B  theta_D sensitivity {0.08,0.1,0.3,0.5,0.7}: outer iters + J_err
  C  wrong-sign safety sweep (CDD / top-|b| / top-|c|)
  D  trajectory adjoint under lax.scan (JAX) -- in dd_jax_closeout via --partD
  E  discontinuous spherical inclusion, moving
"""

from __future__ import annotations

import os
import sys
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator

import dd_wavelets as dd
from hybrid_jacobi import precond, scaled

CACHE = "/tmp/closeout_3d_env.npz"


def build_env(n_levels=4):
    if os.path.exists(CACHE):
        z = np.load(CACHE)
        return dict(A=z["A"], W3n=z["W3n"], levels=z["levels"], Nside=int(z["Nside"]),
                    h=float(z["h"]))
    W3, levels, Nside = dd.synthesis_matrix_3d_isotropic(n_levels, 1, 4)
    h = 1.0 / Nside
    W3n = W3 / np.sqrt((h**3) * np.sum(W3**2, axis=0))[None, :]
    S = dd.laplacian_periodic(Nside, mass=False); Mm = h*np.eye(Nside)
    Aph = (np.kron(np.kron(S, Mm), Mm) + np.kron(np.kron(Mm, S), Mm)
           + np.kron(np.kron(Mm, Mm), S) + np.kron(np.kron(Mm, Mm), Mm))
    A = W3n.T @ Aph @ W3n; A = 0.5*(A+A.T)
    np.savez(CACHE, A=A, W3n=W3n, levels=levels, Nside=Nside, h=h)
    return dict(A=A, W3n=W3n, levels=levels, Nside=Nside, h=h)


def coords_of(env):
    return np.arange(env["Nside"]) / env["Nside"]


def gridfn(env):
    c = coords_of(env)
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    return X, Y, Z


def source_vec(env, theta, sigma):
    X, Y, Z = gridfn(env)
    f = np.exp(-(((X-theta[0])**2 + (Y-theta[1])**2 + (Z-theta[2])**2)/sigma**2)).reshape(-1)
    return (env["h"]**3) * (env["W3n"].T @ f)


def sensor_row(env, sensor):
    c = coords_of(env); Ns = env["Nside"]
    si = int(np.argmin(np.abs(c - sensor[0])))
    sj = int(np.argmin(np.abs(c - sensor[1])))
    sk = int(np.argmin(np.abs(c - sensor[2])))
    return env["W3n"][(si*Ns + sj)*Ns + sk]


def cdd_grow(As, bh, levels, K, theta_D=0.5, count_inner=False):
    N = len(bh)
    Lam = set(np.where(levels == levels.min())[0].tolist())
    nout = 0; inner = 0
    while True:
        idx = np.array(sorted(Lam))
        if count_inner:
            cnt = {"n": 0}
            sol, _ = gmres(As[np.ix_(idx, idx)], bh[idx], rtol=1e-8,
                           restart=min(len(idx), 100), maxiter=2000,
                           callback=lambda _: cnt.__setitem__("n", cnt["n"]+1),
                           callback_type="pr_norm")
            inner += cnt["n"]; c = np.zeros(N); c[idx] = sol
        else:
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
    return idx, c, nout, inner


def part_A(env):
    print("="*78); print("PART A -- 3D CDD trajectory (periodic, hybrid Jacobi)"); print("="*78)
    A, levels = env["A"], env["levels"]
    D, _ = precond(A, levels, "hybrid"); As = scaled(A, D)
    srow = sensor_row(env, (0.7, 0.6, 0.55)); srow_h = srow / D
    N = A.shape[0]; K = N // 16
    thetas = [(0.3 + 0.3*np.sin(2*np.pi*t/30), 0.5, 0.5) for t in range(30)]
    print(f"  N={N}, k=N/16={K}, T=30, sensor=(0.7,0.6,0.55)")
    for sigma in (0.10, 0.02):
        errs = {"cdd": [], "rolling": [], "oracle": []}; inner_tot = 0
        prev_c = None
        for theta in thetas:
            bh = source_vec(env, theta, sigma) / D
            c_full = np.linalg.solve(As, bh); Jf = srow_h @ c_full
            den = abs(Jf) + 1e-30
            idx, c, nout, inner = cdd_grow(As, bh, levels, K, count_inner=True)
            inner_tot += inner
            errs["cdd"].append(abs(srow_h @ c - Jf)/den)
            # rolling
            if prev_c is None:
                sr = np.argsort(-np.abs(bh))[:K]
            else:
                sr = np.argsort(-np.abs(prev_c))[:K]
            sr = np.union1d(sr, np.where(levels == levels.min())[0])
            cr = np.zeros(N); cr[sr] = np.linalg.solve(As[np.ix_(sr, sr)], bh[sr])
            errs["rolling"].append(abs(srow_h @ cr - Jf)/den); prev_c = cr
            # oracle
            so = np.argsort(-np.abs(c_full))[:K]
            co = np.zeros(N); co[so] = np.linalg.solve(As[np.ix_(so, so)], bh[so])
            errs["oracle"].append(abs(srow_h @ co - Jf)/den)
        print(f"  sigma={sigma}:")
        for k in ("cdd", "rolling", "oracle"):
            print(f"    {k:>8}: mean J_err={np.mean(errs[k]):.2e}, peak={np.max(errs[k]):.2e}"
                  + (f", total inner GMRES={inner_tot}" if k == "cdd" else ""))


def part_B(env):
    print("="*78); print("PART B -- 3D theta_D sensitivity"); print("="*78)
    A, levels = env["A"], env["levels"]
    D, _ = precond(A, levels, "hybrid"); As = scaled(A, D)
    srow_h = sensor_row(env, (0.7, 0.6, 0.55)) / D
    N = A.shape[0]; K = N // 16
    kap = np.linalg.cond(As)
    print(f"  hybrid-Jacobi kappa={kap:.1f} -> theory bound theta_D<kappa^-1/2={kap**-0.5:.3f}")
    thetas = [(0.3 + 0.3*np.sin(2*np.pi*t/30), 0.5, 0.5) for t in range(0, 30, 3)]
    print(f"  {'theta_D':>8} {'mean outer':>11} {'peak outer':>11} {'mean J_err':>11}")
    for tD in (0.08, 0.1, 0.3, 0.5, 0.7):
        nouts, jerrs = [], []
        for theta in thetas:
            bh = source_vec(env, theta, 0.10) / D
            Jf = srow_h @ np.linalg.solve(As, bh)
            idx, c, nout, _ = cdd_grow(As, bh, levels, K, theta_D=tD)
            nouts.append(nout); jerrs.append(abs(srow_h @ c - Jf)/(abs(Jf)+1e-30))
        print(f"  {tD:>8.2f} {np.mean(nouts):>11.1f} {np.max(nouts):>11.0f} {np.mean(jerrs):>11.2e}")


def part_C(env):
    print("="*78); print("PART C -- 3D wrong-sign safety sweep"); print("="*78)
    A, levels = env["A"], env["levels"]
    D, _ = precond(A, levels, "hybrid"); As = scaled(A, D)
    srow = sensor_row(env, (0.3, 0.4, 0.6)); srow_h = srow / D
    N = A.shape[0]; K = N // 16
    print(f"  sensor=(0.3,0.4,0.6), k=N/16={K}; flag WRONG if sign(J_frozen)!=sign(J_full)")
    print(f"  {'theta_x':>8} {'J_full':>11} {'CDD':>8} {'top|b|':>8} {'top|c|':>8}")
    any_wrong = {"cdd": False, "topb": False, "topc": False}
    for tx in np.arange(0.05, 0.96, 0.1):
        theta = (tx, 0.5, 0.5)
        bh = source_vec(env, theta, 0.10) / D
        c_full = np.linalg.solve(As, bh); Jf = srow_h @ c_full
        # CDD
        idx, c, _, _ = cdd_grow(As, bh, levels, K); Jc = srow_h @ c
        # top-|b|
        sb = np.argsort(-np.abs(bh))[:K]; cb = np.zeros(N)
        cb[sb] = np.linalg.solve(As[np.ix_(sb, sb)], bh[sb]); Jb = srow_h @ cb
        # top-|c|
        sc = np.argsort(-np.abs(c_full))[:K]; cc = np.zeros(N)
        cc[sc] = np.linalg.solve(As[np.ix_(sc, sc)], bh[sc]); Jcc = srow_h @ cc
        def tag(J):
            return "WRONG" if J*Jf < 0 else "ok"
        for key, J in (("cdd", Jc), ("topb", Jb), ("topc", Jcc)):
            if J*Jf < 0: any_wrong[key] = True
        print(f"  {tx:>8.2f} {Jf:>11.3e} {tag(Jc):>8} {tag(Jb):>8} {tag(Jcc):>8}")
    print(f"  --> any wrong-sign: CDD={any_wrong['cdd']}, top|b|={any_wrong['topb']}, "
          f"top|c|={any_wrong['topc']}")


def part_E(env):
    print("="*78); print("PART E -- 3D discontinuous coefficient (moving sphere)"); print("="*78)
    A0, levels = env["A"], env["levels"]   # A0 unused; rebuild per-position
    Ns = env["Nside"]; h = env["h"]; W3n = env["W3n"]; N = Ns**3
    c = coords_of(env); X, Y, Z = gridfn(env)
    f = np.exp(-(((X-0.5)**2+(Y-0.5)**2+(Z-0.5)**2)/0.2**2)).reshape(-1)
    bvec = (h**3) * (W3n.T @ f)
    cen = np.argmax(np.abs(W3n), axis=0)
    ci, cj, ck = np.unravel_index(cen, (Ns, Ns, Ns))
    cx, cy, cz = c[ci], c[cj], c[ck]
    r = 0.15; K = N // 16
    print(f"  Nside={Ns}, r={r}, a=1+99*1_inside, hybrid Jacobi, k=N/16={K}")
    print(f"  {'xc':>20} {'CDD out':>8} {'%@bndry':>8} {'J_err':>10}")
    Sst = dd.laplacian_periodic(Ns, mass=False); Mm = h*np.eye(Ns)
    for s in np.linspace(0, 1, 5):
        xc = (0.35 + 0.30*s, 0.35 + 0.30*s, 0.35 + 0.30*s)
        inside = ((X-xc[0])**2 + (Y-xc[1])**2 + (Z-xc[2])**2 < r**2).astype(float)
        a = (1.0 + 99.0*inside)
        # variable-coeff operator via diagonal-weighted stiffness (approx: a at nodes)
        # build -div(a grad)+mass in physical space (7-pt, face-averaged a), periodic
        Aph = build_varcoeff_3d(Ns, a.reshape(Ns, Ns, Ns), h)
        Aw = W3n.T @ Aph @ W3n; Aw = 0.5*(Aw+Aw.T)
        D = np.sqrt(np.abs(np.diag(Aw))); D[D == 0] = 1
        As = (Aw/D[:, None])/D[None, :]; bh = bvec/D
        c_ref = np.linalg.solve(As, bh); u_ref = W3n @ (c_ref/D)
        idx, cc, nout, _ = cdd_grow(As, bh, levels, K)
        u_cdd = W3n @ (cc/D)
        jerr = np.linalg.norm(u_cdd-u_ref)/np.linalg.norm(u_ref)
        dist = np.abs(np.sqrt((cx-xc[0])**2+(cy-xc[1])**2+(cz-xc[2])**2) - r)
        nb = np.mean([dist[i] < 1.5*h for i in idx])
        print(f"  ({xc[0]:.2f},{xc[1]:.2f},{xc[2]:.2f}) {nout:>8d} {nb*100:>7.0f}% {jerr:>10.2e}")


def build_varcoeff_3d(Ns, a, h):
    N = Ns**3
    A = np.zeros((N, N))
    def idx(i, j, k): return ((i % Ns)*Ns + (j % Ns))*Ns + (k % Ns)
    for i in range(Ns):
        for j in range(Ns):
            for k in range(Ns):
                kk = idx(i, j, k); aijk = a[i, j, k]
                tot = 0.0
                for di, dj, dk in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                    af = 0.5*(aijk + a[(i+di) % Ns, (j+dj) % Ns, (k+dk) % Ns])
                    A[kk, idx(i+di, j+dj, k+dk)] -= af/h**2; tot += af/h**2
                A[kk, kk] = tot + 1.0
    return 0.5*(A+A.T)


def part_D():
    """3D trajectory adjoint under lax.scan (JAX, float64). Loads the cache."""
    print("="*78); print("PART D -- 3D trajectory adjoint under lax.scan (JAX)"); print("="*78)
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    z = np.load(CACHE)
    A = jnp.asarray(z["A"]); W3n = jnp.asarray(z["W3n"]); levels = z["levels"]
    Nside = int(z["Nside"]); h = float(z["h"]); N = A.shape[0]
    Dnp, _ = precond(z["A"], levels, "hybrid"); D = jnp.asarray(Dnp)
    c = np.arange(Nside)/Nside
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    Xj, Yj, Zj = jnp.asarray(X.reshape(-1)), jnp.asarray(Y.reshape(-1)), jnp.asarray(Z.reshape(-1))
    si = int(np.argmin(np.abs(c-0.7))); sj = int(np.argmin(np.abs(c-0.6))); sk = int(np.argmin(np.abs(c-0.55)))
    srow = W3n[(si*Nside+sj)*Nside+sk] / D
    Ah = (A / D[:, None]) / D[None, :]
    coarse_mask = jnp.zeros(N, bool).at[jnp.asarray(np.where(levels == levels.min())[0])].set(True)
    K = N // 16

    def bvec(theta_x, sigma=0.10):
        f = jnp.exp(-(((Xj-theta_x)**2 + (Yj-0.5)**2 + (Zj-0.5)**2)/sigma**2))
        return (h**3) * (W3n.T @ f) / D

    def select(bh, c_prev):
        r = bh - Ah @ c_prev
        score = jnp.where(coarse_mask, jnp.inf, jnp.abs(r))
        idx = jnp.argsort(-score)[:K]
        return jax.lax.stop_gradient(jnp.zeros(N, bool).at[idx].set(True))

    def masked_solve(bh, mask):
        mm = mask[:, None] & mask[None, :]
        Aeff = jnp.where(mm, Ah, 0.0) + jnp.diag(jnp.where(mask, 0.0, 1.0))
        return jnp.linalg.solve(Aeff, jnp.where(mask, bh, 0.0))

    def step(c_prev, theta_x):
        bh = bvec(theta_x)
        mask = select(bh, c_prev)
        c = masked_solve(bh, mask)
        return c, srow @ c

    def traj_J(theta0, T, dstep):
        xs = theta0 + dstep * jnp.arange(T)
        _, us = jax.lax.scan(step, jnp.zeros(N), xs)
        return jnp.sum(us**2)

    def mask_flip_in_window(theta0, T, dstep, eps):
        """True if any step's mask differs between theta0±eps (FD invalid)."""
        for s in (theta0 - eps, theta0 + eps):
            xs = s + dstep * np.arange(T)
        # compare full mask sequence at theta0-eps vs theta0+eps
        def masks(th):
            cp = jnp.zeros(N); ms = []
            for t in range(T):
                bh = bvec(jnp.asarray(th + dstep*t)); m = select(bh, cp)
                cp = masked_solve(bh, m); ms.append(np.asarray(m))
            return ms
        m1, m2 = masks(theta0 - eps), masks(theta0 + eps)
        return any(int(np.sum(a != b)) > 0 for a, b in zip(m1, m2))

    eps = 1e-6
    # Direct clean reference at a smooth interior point.
    print("  direct test at theta0=0.30 (dstep=0.025):")
    print(f"  {'T':>3} {'grad':>14} {'FD':>14} {'rel_err':>10}")
    for T in (1, 3, 5):
        g = float(jax.grad(lambda t: traj_J(t, T, 0.025))(jnp.asarray(0.30)))
        fd = float((traj_J(jnp.asarray(0.30+eps), T, 0.025) - traj_J(jnp.asarray(0.30-eps), T, 0.025))/(2*eps))
        print(f"  {T:>3} {g:>14.6e} {fd:>14.6e} {abs(g-fd)/(abs(fd)+1e-30):>10.2e}")
    # Robust smoothness classification via Richardson: a point is SMOOTH if
    # central-FD(eps) and central-FD(eps/2) agree (no kink in window). In 3D the
    # K=256 active set sits near many top-K ties, so kinks are DENSE -- report
    # the fraction smooth and the agreement at smooth points (detector-free).
    def fd_at(t0, T, e):
        return (traj_J(jnp.asarray(t0+e), T, 0.025) - traj_J(jnp.asarray(t0-e), T, 0.025))/(2*e)
    print("  grad vs FD over scan theta0=0.20..0.44 (Richardson-classified smooth points):")
    print(f"  {'T':>3} {'%smooth':>8} {'median rel_err@smooth':>22}")
    for T in (1, 3, 5):
        rels = []; nsm = 0; ntot = 0
        for t0 in np.linspace(0.20, 0.44, 25):
            ntot += 1
            f1 = float(fd_at(t0, T, eps)); f2 = float(fd_at(t0, T, eps/2))
            if abs(f1 - f2) / (abs(f2) + 1e-30) > 1e-3:   # kink in window -> skip
                continue
            nsm += 1
            g = float(jax.grad(lambda t: traj_J(t, T, 0.025))(jnp.asarray(t0)))
            rels.append(abs(g - f2) / (abs(f2) + 1e-30))
        med = np.median(rels) if rels else float("nan")
        print(f"  {T:>3} {100*nsm/ntot:>7.0f}% {med:>22.2e}")
    print("  (3D active set K=256 sits near many top-K ties -> kinks are dense along")
    print("   any trajectory; grad is the correct Clarke subgradient at kinks, below.)")

    # near-kink probe: find a theta0 where the mask flips within +/-1e-5 at T=1
    print("  near-kink Clarke probe (T=1): scan theta0 for a mask flip in [theta-eps,theta+eps]")
    eps = 1e-5
    found = None
    for t0 in np.linspace(0.2, 0.6, 400):
        m1 = np.asarray(select(bvec(jnp.asarray(t0-eps)), jnp.zeros(N)))
        m2 = np.asarray(select(bvec(jnp.asarray(t0+eps)), jnp.zeros(N)))
        if int(np.sum(m1 != m2)) > 0:
            found = (t0, int(np.sum(m1 != m2))); break
    if found:
        t0, nflip = found
        g = float(jax.grad(lambda t: traj_J(t, 1, 0.05))(jnp.asarray(t0)))
        fdp = float((traj_J(jnp.asarray(t0+eps), 1, 0.05) - traj_J(jnp.asarray(t0), 1, 0.05))/eps)
        fdm = float((traj_J(jnp.asarray(t0), 1, 0.05) - traj_J(jnp.asarray(t0-eps), 1, 0.05))/eps)
        print(f"    at theta0={t0:.4f} ({nflip} modes flip): grad={g:.4e}, "
              f"one-sided FD+={fdp:.4e}, FD-={fdm:.4e}")
        print(f"    grad lies between one-sided FDs (Clarke subgradient): "
              f"{min(fdm,fdp)-1e-6 <= g <= max(fdm,fdp)+1e-6}")
    else:
        print("    no mask flip found in scan range (mask stable) -- grad=FD applies")


if __name__ == "__main__":
    sel = sys.argv[1] if len(sys.argv) > 1 else "all"
    if sel == "D":
        part_D(); sys.exit(0)
    env = build_env()
    if sel in ("A", "all"): part_A(env)
    if sel in ("B", "all"): part_B(env)
    if sel in ("C", "all"): part_C(env)
    if sel in ("E", "all"): part_E(env)
