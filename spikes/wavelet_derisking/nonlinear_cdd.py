"""Investigation 3 -- CDD feasibility on a NONLINEAR residual.

SPIKE CODE. Investigative.

CDD's residual criterion was validated only on linear elliptic problems. Does
it still concentrate DOF correctly when the residual includes a nonlinear
convective term?

Part A -- 1D viscous Burgers u_t + u u_x = nu u_xx as a proxy. Implicit Euler
  + Newton. At each Newton step CDD selects active wavelet modes from the FULL
  nonlinear residual. We check: does CDD mark the shock-forming region? Stay
  below N/8? Converge in <=20 outer iters? Critical test = the step just
  before the shock (steepest gradient).
Part B -- 2D stream-function/vorticity driven cavity at Re=100 (feasibility,
  not a Ghia accuracy match).

Periodic BCs, IC u0 = sin(2 pi x) (the textbook shock-forming periodic IC; the
brief's "sin(pi x)" is not periodic, so we use the periodic analogue and note
it). DD-4, N=256.
"""

from __future__ import annotations

import numpy as np

import dd_wavelets as dd

NU = 0.01


def _l2(W, h):
    n = np.sqrt(h * np.sum(W ** 2, axis=0)); n[n == 0] = 1
    return W / n[None, :]


def setup(n_levels=7):
    W, levels, x = dd.synthesis_matrix(n_levels, n_coarse=2, order=4, boundary="periodic")
    N = W.shape[0]
    h = 1.0 / N
    Wn = _l2(W, h)
    Winv = np.linalg.inv(Wn)
    # periodic central differences
    idx = np.arange(N)
    Dx = np.zeros((N, N)); Dx[idx, (idx+1) % N] = 0.5/h; Dx[idx, (idx-1) % N] = -0.5/h
    Dxx = np.zeros((N, N)); Dxx[idx, idx] = -2/h**2
    Dxx[idx, (idx+1) % N] += 1/h**2; Dxx[idx, (idx-1) % N] += 1/h**2
    return dict(Wn=Wn, Winv=Winv, levels=levels, x=x, h=h, N=N, Dx=Dx, Dxx=Dxx)


def cdd_select_c(Jc, rc, levels, K, theta_D=0.5):
    """CDD active set in wavelet coords from the (scaled) residual rc."""
    N = len(rc)
    Lam = set(np.where(levels == levels.min())[0].tolist())
    while True:
        idx = np.array(sorted(Lam))
        if len(idx) >= K:
            return idx
        # residual restricted: we mark on the current global residual rc minus
        # the part explained by the current solve (approx: use rc directly for
        # marking, which is the nonlinear residual projected to wavelet coords)
        c = np.zeros(N)
        c[idx] = np.linalg.solve(Jc[np.ix_(idx, idx)], rc[idx])
        r = rc - Jc @ c
        mo = np.ones(N, bool); mo[idx] = False
        oi = np.where(mo)[0]; order = oi[np.argsort(-np.abs(r[oi]))]
        cs = np.cumsum(r[order] ** 2)
        kt = min(int(np.searchsorted(cs, (theta_D**2)*np.linalg.norm(r)**2)+1), len(order))
        if kt == 0:
            return idx
        Lam.update(order[:kt].tolist())


def part_A():
    print("=" * 78)
    print("PART A -- 1D viscous Burgers (nu=0.01), CDD on the nonlinear residual")
    print("=" * 78)
    env = setup(7)
    Wn, Winv, levels = env["Wn"], env["Winv"], env["levels"]
    x, h, N, Dx, Dxx = env["x"], env["h"], env["N"], env["Dx"], env["Dxx"]
    D = 2.0 ** levels.astype(float)   # DK; Jacobi computed per-step below

    u = np.sin(2 * np.pi * x)
    dt = 0.004
    nsteps = 40
    K = N // 8
    print(f"  N={N}, dt={dt}, K=N/8={K}, IC=sin(2πx)")
    print(f"  {'t':>6} {'max|u_x|':>9} {'newton':>7} {'cdd_out':>8} "
          f"{'|Λ|':>5} {'oracle∩':>8} {'%near_shock':>11} {'J_err':>9}")
    for step in range(1, nsteps + 1):
        u_prev = u.copy()
        # --- full Newton to convergence (reference) ---
        uk = u_prev.copy()
        n_newton = 0
        for _ in range(50):
            F = (uk - u_prev)/dt + uk*(Dx@uk) - NU*(Dxx@uk)
            J = np.eye(N)/dt + np.diag(Dx@uk) + np.diag(uk)@Dx - NU*Dxx
            duk = np.linalg.solve(J, -F)
            uk = uk + duk
            n_newton += 1
            if np.linalg.norm(duk) < 1e-10:
                break
        u = uk
        c_conv = Winv @ u
        # oracle active set: top-K |c| of converged solution
        oracle = set(np.argsort(-np.abs(c_conv))[:K].tolist())

        # --- CDD selection at the FIRST Newton iterate (nonlinear residual) ---
        u0 = u_prev
        F0 = (u0 - u_prev)/dt + u0*(Dx@u0) - NU*(Dxx@u0)
        J0 = np.eye(N)/dt + np.diag(Dx@u0) + np.diag(u0)@Dx - NU*Dxx
        # wavelet coords, Jacobi-scaled
        Jc = Wn.T @ J0 @ Wn
        Djac = np.sqrt(np.abs(np.diag(Jc))); Djac[Djac == 0] = 1
        Jc_s = (Jc / Djac[:, None]) / Djac[None, :]
        rc = (Wn.T @ (-F0)) / Djac
        # count CDD outer iters via a counting variant
        Lam = set(np.where(levels == levels.min())[0].tolist()); nout = 0
        while True:
            idx = np.array(sorted(Lam))
            if len(idx) >= K:
                break
            cc = np.zeros(N); cc[idx] = np.linalg.solve(Jc_s[np.ix_(idx, idx)], rc[idx])
            r = rc - Jc_s @ cc
            mo = np.ones(N, bool); mo[idx] = False
            oi = np.where(mo)[0]; order = oi[np.argsort(-np.abs(r[oi]))]
            cs = np.cumsum(r[order]**2)
            kt = min(int(np.searchsorted(cs, 0.25*np.linalg.norm(r)**2)+1), len(order))
            if kt == 0:
                break
            Lam.update(order[:kt].tolist()); nout += 1
        sel = set(idx.tolist())
        overlap = len(sel & oracle) / max(1, len(oracle))

        # shock location = argmax|u_x|; fraction of active modes whose spatial
        # centre is within 0.1 of the shock
        ux = Dx @ u
        shock_x = x[int(np.argmax(np.abs(ux)))]
        # spatial centre of each wavelet = argmax|Wn[:,col]|
        centres = x[np.argmax(np.abs(Wn), axis=0)]
        near = np.mean([abs(centres[i] - shock_x) < 0.1 or
                        abs(abs(centres[i]-shock_x)-1) < 0.1 for i in idx])

        # J_err: solve restricted to CDD set, compare a functional (mean u^2)
        c_cdd = np.zeros(N)
        c_cdd[idx] = np.linalg.solve(Jc[np.ix_(idx, idx)], (Wn.T @ (-F0))[idx])
        # one frozen Newton step from u_prev in the CDD subspace
        u_cdd = u_prev + Wn @ c_cdd
        Jf = np.mean(u**2); Jf_cdd = np.mean(u_cdd**2)
        jerr = abs(Jf_cdd - Jf) / (abs(Jf) + 1e-30)

        if step % 4 == 0 or step <= 2:
            print(f"  {step*dt:>6.3f} {np.max(np.abs(ux)):>9.2f} {n_newton:>7d} "
                  f"{nout:>8d} {len(idx):>5d} {overlap:>8.2f} {near:>11.2f} {jerr:>9.2e}")


def part_B():
    print("=" * 78)
    print("PART B -- 2D stream-function/vorticity cavity Re=100 (feasibility)")
    print("=" * 78)
    # NB the ψ-ω formulation solves a POISSON ∇²ψ=-ω each step (the brief's
    # '∆²ψ=-ω' reads as the Laplacian; the §5 biharmonic is the *pure*-ψ
    # alternative). CDD is tested on this elliptic ψ-solve with Jacobi.
    Re = 100.0
    nu = 1.0 / Re
    Ni = 47                      # interior points per side (matches DD dirichlet)
    h = 1.0 / (Ni + 1)
    M = Ni + 2                   # incl walls
    dt = 0.004
    nsteps = 3000

    psi = np.zeros((M, M)); omega = np.zeros((M, M))
    # interior 5-point Laplacian (dense inverse for the physical ψ-solve)
    n = Ni * Ni
    Lap = np.zeros((n, n))
    def ij(i, j): return (i - 1) * Ni + (j - 1)
    for i in range(1, Ni + 1):
        for j in range(1, Ni + 1):
            k = ij(i, j); Lap[k, k] = -4 / h**2
            for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):
                ii, jj = i + di, j + dj
                if 1 <= ii <= Ni and 1 <= jj <= Ni:
                    Lap[k, ij(ii, jj)] = 1 / h**2
    Lap_inv = np.linalg.inv(Lap)

    for step in range(nsteps):
        # velocities from psi (central); lid (top row i=M-1) u=1
        u = np.zeros((M, M)); v = np.zeros((M, M))
        u[1:-1, 1:-1] = (psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
        v[1:-1, 1:-1] = -(psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
        u[-1, :] = 1.0  # lid
        # Thom vorticity wall BCs
        omega[-1, :] = -2 * (psi[-2, :] - psi[-1, :]) / h**2 - 2 * 1.0 / h  # top (lid)
        omega[0, :] = -2 * (psi[1, :] - psi[0, :]) / h**2
        omega[:, 0] = -2 * (psi[:, 1] - psi[:, 0]) / h**2
        omega[:, -1] = -2 * (psi[:, -2] - psi[:, -1]) / h**2
        # vorticity transport (explicit), interior
        wx = (omega[1:-1, 2:] - omega[1:-1, :-2]) / (2 * h)
        wy = (omega[2:, 1:-1] - omega[:-2, 1:-1]) / (2 * h)
        lap_w = (omega[2:, 1:-1] + omega[:-2, 1:-1] + omega[1:-1, 2:]
                 + omega[1:-1, :-2] - 4 * omega[1:-1, 1:-1]) / h**2
        omega[1:-1, 1:-1] += dt * (-u[1:-1, 1:-1] * wx - v[1:-1, 1:-1] * wy + nu * lap_w)
        # psi Poisson solve: ∇²ψ = -ω
        rhs = -omega[1:-1, 1:-1].reshape(-1)
        psi[1:-1, 1:-1] = (Lap_inv @ rhs).reshape(Ni, Ni)

    # qualitative diagnostics
    pmin = psi.min(); loc = np.unravel_index(np.argmin(psi), psi.shape)
    xc, yc = loc[1] / (M - 1), loc[0] / (M - 1)
    # corner vortices: sign of psi opposite to primary in bottom corners
    bl = psi[1:8, 1:8].max(); br = psi[1:8, -8:-1].max()
    print(f"  grid {Ni}², dt={dt}, {nsteps} steps (t={nsteps*dt:.1f})")
    print(f"  primary vortex: psi_min={pmin:.4e} at (x={xc:.2f}, y={yc:.2f})")
    print(f"    [Ghia Re=100 reference: centre ≈ (0.62, 0.74)]")
    print(f"  bottom-corner counter-rotation: BL max psi={bl:.2e}, BR max psi={br:.2e}"
          f"  ({'visible' if bl>1e-6 and br>1e-6 else 'weak'})")

    # ---- CDD test on the psi-Poisson solve in the DD-4 wavelet basis ----
    W1, levels1, x1 = dd.synthesis_matrix(4, n_coarse=2, order=4, boundary="dirichlet")
    Nside = W1.shape[0]   # 47
    hw = 1.0 / (Nside + 1)
    W1n = _l2(W1, hw)
    # 2D anisotropic tensor basis (Jacobi makes basis structure irrelevant)
    W2 = np.kron(W1n, W1n)
    lev2 = (levels1[:, None] + levels1[None, :]).reshape(-1)  # only for bookkeeping
    # 2D Dirichlet Laplacian (interior), as bilinear operator
    Lap1 = -dd.laplacian_dirichlet(Nside, mass=False)  # = +d2/dx2 (we want -lap for SPD)
    # build A2 = -(Dxx⊗I + I⊗Dxx); use laplacian_dirichlet which is (-d2+I); take stiffness only
    S1 = dd.laplacian_dirichlet(Nside, mass=False)  # (-d2/dx2) SPD
    Mass = hw * np.eye(Nside)
    A2phys = np.kron(S1, Mass) + np.kron(Mass, S1)
    A2 = W2.T @ A2phys @ W2
    A2 = 0.5 * (A2 + A2.T)
    Djac = np.sqrt(np.abs(np.diag(A2))); Djac[Djac == 0] = 1
    A2s = (A2 / Djac[:, None]) / Djac[None, :]

    # RHS from the converged omega, interpolated onto the wavelet grid
    om_int = omega[1:-1, 1:-1]
    # wavelet grid coords
    xw = x1
    Xg, Yg = np.meshgrid(np.arange(M)/(M-1), np.arange(M)/(M-1), indexing="ij")
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((np.arange(M)/(M-1), np.arange(M)/(M-1)),
                                     omega, bounds_error=False, fill_value=0.0)
    XX, YY = np.meshgrid(xw, xw, indexing="ij")
    om_w = interp(np.stack([XX.ravel(), YY.ravel()], axis=1))
    f = om_w  # ∇²ψ=-ω → stiffness form: A2 ψ = (ω) since A2=-∇² (SPD)
    b = (hw * hw) * (W2.T @ f)
    bh = b / Djac
    psi_full = np.linalg.solve(A2s, bh) / Djac
    psi_full_grid = (W2 @ psi_full).reshape(Nside, Nside)

    N2 = Nside * Nside
    coarse = np.where(lev2 == lev2.min())[0]
    for Kfrac in (16, 8):
        K = max(8, N2 // Kfrac)
        Lam = set(coarse.tolist()); nout = 0
        while True:
            idx = np.array(sorted(Lam))
            if len(idx) >= K: break
            c = np.zeros(N2); c[idx] = np.linalg.solve(A2s[np.ix_(idx, idx)], bh[idx])
            r = bh - A2s @ c
            mo = np.ones(N2, bool); mo[idx] = False
            oi = np.where(mo)[0]; order = oi[np.argsort(-np.abs(r[oi]))]
            cs = np.cumsum(r[order]**2)
            kt = min(int(np.searchsorted(cs, 0.25*np.linalg.norm(r)**2)+1), len(order))
            if kt == 0: break
            Lam.update(order[:kt].tolist()); nout += 1
        c = np.zeros(N2); c[idx] = np.linalg.solve(A2s[np.ix_(idx, idx)], bh[idx])
        psi_cdd = (W2 @ (c / Djac)).reshape(Nside, Nside)
        jerr = np.linalg.norm(psi_cdd - psi_full_grid) / np.linalg.norm(psi_full_grid)
        # fraction of active modes whose 2D support centre is near the lid (top) or corners
        cen = np.argmax(np.abs(W2), axis=0)
        ci, cj = np.unravel_index(cen, (Nside, Nside))
        cx, cy = xw[ci], xw[cj]
        near_lid = np.mean([(cy[i] > 0.85) or (cx[i] < 0.15) or (cx[i] > 0.85)
                            for i in idx])
        print(f"  CDD ψ-solve k=N/{Kfrac} ({K}/{N2}): outer={nout}, "
              f"rel L2 err={jerr:.2e}, frac near lid/walls={near_lid:.2f}")


if __name__ == "__main__":
    part_A()
    part_B()
