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


if __name__ == "__main__":
    part_A()
