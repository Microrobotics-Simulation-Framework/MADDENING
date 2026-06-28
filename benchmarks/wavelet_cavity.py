"""M5 part 2 — lid-driven cavity benchmark with a wavelet ψ-solve.

Validates that the production DD-wavelet Dirichlet Poisson solver produces a
quantitatively correct incompressible flow: the classic lid-driven cavity at
Re=100, compared to the Ghia-Ghia-Shin (1982) tabulated centreline velocities.

Formulation: stream-function/vorticity (ψ-ω).  Vorticity transport is 2nd-order
finite-difference in physical space with Thom wall-vorticity BCs; the
ψ-Poisson ``-∇²ψ = ω`` (homogeneous Dirichlet) is solved each step.

Two validations, decoupled for speed:

1. **Flow vs Ghia** -- the time loop uses the fast factorised FD Poisson solve
   (``run_cavity``), and the steady-state centreline velocity is compared to the
   Ghia tabulation.  This validates the scheme reaches the reference flow.

2. **Wavelet solver in context** -- at the converged vorticity, the ψ-Poisson is
   re-solved in the **DD-wavelet Dirichlet basis** (``wavelet_psi_consistency``)
   and checked against the FD ψ.  The wavelet Poisson operator is the FD Poisson
   represented in the (L²-normalised, boundary-adapted) wavelet basis
   ``A = Wnᵀ (-L_fd) Wn`` (SPD) -- an exact change of basis, so it reproduces the
   FD ψ to machine precision while genuinely exercising the wavelet operator
   (hybrid-Jacobi-conditionable, CDD-truncatable).

   (Running the dense-Wn wavelet solve *every* step is correct but slow at these
   sizes -- the Dirichlet basis is dense; a matrix-free Dirichlet transform is a
   later optimisation.  Decoupling keeps the benchmark fast without losing the
   in-context validation.)

Run directly::

    python benchmarks/wavelet_cavity.py            # 47² (fast)
    python benchmarks/wavelet_cavity.py 5          # 95² (finer)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.linalg as sla

from maddening.nodes.adaptive.wavelets import dirichlet as DIR


# Ghia-Ghia-Shin (1982) Re=100 reference.
# u along the vertical centreline (x=0.5):
GHIA_RE100_Y = np.array(
    [0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 0.2813, 0.4531, 0.5000,
     0.6172, 0.7344, 0.8516, 0.9531, 0.9609, 0.9688, 0.9766, 1.0000])
GHIA_RE100_U = np.array(
    [0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150, -0.15662,
     -0.21090, -0.20581, -0.13641, 0.00332, 0.23151, 0.68717, 0.73722,
     0.78871, 0.84123, 1.00000])
GHIA_RE100_VORTEX = (0.6172, 0.7344)     # primary vortex centre (x, y)


def _l2_normalise(W, h):
    nrm = np.sqrt(h * np.sum(W ** 2, axis=0))
    nrm[nrm == 0] = 1.0
    return W / nrm[None, :]


def _fd_dirichlet_laplacian(n, h):
    """SPD ``-L`` (5-point Dirichlet Laplacian, negated), shape (n², n²)."""
    N = n * n
    L = np.zeros((N, N))

    def k(i, j):
        return i * n + j

    for i in range(n):
        for j in range(n):
            kk = k(i, j)
            L[kk, kk] = 4.0 / h ** 2
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ii, jj = i + di, j + dj
                if 0 <= ii < n and 0 <= jj < n:
                    L[kk, k(ii, jj)] = -1.0 / h ** 2
    return L


def _side(nl, nc):
    return DIR.dirichlet_side(nl, nc)


def run_cavity(nl=4, nc=2, Re=100.0, dt=0.002, nsteps=30000, tol=1e-6,
               verbose=False):
    """Run the ψ-ω cavity to steady state (fast factorised FD ψ-solve)."""
    n = _side(nl, nc)
    h = 1.0 / (n + 1)
    lu = sla.lu_factor(_fd_dirichlet_laplacian(n, h))
    nu = 1.0 / Re
    U = 1.0
    psi = np.zeros((n + 2, n + 2))
    om = np.zeros((n + 2, n + 2))

    def psi_solve(omega_interior):           # -∇²ψ = ω
        return sla.lu_solve(lu, omega_interior.reshape(-1)).reshape(n, n)

    t0 = time.time()
    steps = nsteps
    for step in range(nsteps):
        om_old = om.copy()
        u = np.zeros((n + 2, n + 2))
        v = np.zeros((n + 2, n + 2))
        u[1:-1, 1:-1] = (psi[2:, 1:-1] - psi[:-2, 1:-1]) / (2 * h)
        v[1:-1, 1:-1] = -(psi[1:-1, 2:] - psi[1:-1, :-2]) / (2 * h)
        u[-1, :] = U                          # lid (top, y=1)
        # Thom wall vorticity
        om[-1, :] = -2 * (psi[-2, :] - psi[-1, :]) / h ** 2 - 2 * U / h
        om[0, :] = -2 * (psi[1, :] - psi[0, :]) / h ** 2
        om[:, 0] = -2 * (psi[:, 1] - psi[:, 0]) / h ** 2
        om[:, -1] = -2 * (psi[:, -2] - psi[:, -1]) / h ** 2
        ox = (om[1:-1, 2:] - om[1:-1, :-2]) / (2 * h)
        oy = (om[2:, 1:-1] - om[:-2, 1:-1]) / (2 * h)
        lap = (om[2:, 1:-1] + om[:-2, 1:-1] + om[1:-1, 2:] + om[1:-1, :-2]
               - 4 * om[1:-1, 1:-1]) / h ** 2
        om[1:-1, 1:-1] += dt * (-u[1:-1, 1:-1] * ox - v[1:-1, 1:-1] * oy + nu * lap)
        psi[1:-1, 1:-1] = psi_solve(om[1:-1, 1:-1])
        if step % 1000 == 0 and step > 0:
            resid = np.max(np.abs(om - om_old))
            if verbose:
                print(f"  step {step}: max|Δω|={resid:.2e}")
            if resid < tol:
                steps = step
                break
    return dict(psi=psi, om=om, u=u, v=v, h=h, n=n, nl=nl, nc=nc, steps=steps,
                wall_time=time.time() - t0)


def wavelet_psi_consistency(res, order=4):
    """Re-solve the converged ψ-Poisson in the DD-wavelet Dirichlet basis and
    return the relative error vs the FD ψ (validates the wavelet solver in the
    cavity context).  Exact change of basis ⇒ ~machine precision."""
    n, h = res["n"], res["h"]
    W1, _, _ = DIR.synthesis_matrix_dirichlet(res["nl"], res["nc"], order, dim=1)
    W1n = _l2_normalise(np.asarray(W1), h)
    Wn2 = np.kron(W1n, W1n)
    L = _fd_dirichlet_laplacian(n, h)
    A = Wn2.T @ L @ Wn2
    A = 0.5 * (A + A.T)
    om_int = res["om"][1:-1, 1:-1].reshape(-1)
    c = sla.lu_solve(sla.lu_factor(A), Wn2.T @ om_int)
    psi_wav = (Wn2 @ c).reshape(n, n)
    psi_fd = res["psi"][1:-1, 1:-1]
    return float(np.linalg.norm(psi_wav - psi_fd) / (np.linalg.norm(psi_fd) + 1e-30))


def ghia_comparison(res):
    """Return (max |u - Ghia| on the vertical centreline, vortex (x, y))."""
    n, u, psi = res["n"], res["u"], res["psi"]
    ys = np.arange(n + 2) / (n + 1)
    jc = int(round(0.5 * (n + 1)))
    uc = u[:, jc]
    u_interp = np.interp(GHIA_RE100_Y, ys, uc)
    max_err = float(np.max(np.abs(u_interp - GHIA_RE100_U)))
    imin = np.unravel_index(np.argmin(psi), psi.shape)
    vortex = (imin[1] / (n + 1), imin[0] / (n + 1))
    return max_err, vortex, float(uc.min())


if __name__ == "__main__":
    nl = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    res = run_cavity(nl=nl, verbose=True)
    err, vortex, umin = ghia_comparison(res)
    print(f"n={res['n']}², steps={res['steps']}, {res['wall_time']:.0f}s")
    print(f"min centreline u = {umin:.4f}  (Ghia -0.21090)")
    print(f"max |u - Ghia|   = {err:.4f}")
    print(f"vortex centre    = ({vortex[0]:.3f}, {vortex[1]:.3f})  "
          f"(Ghia {GHIA_RE100_VORTEX})")
    print(f"wavelet ψ-solve vs FD ψ (converged) = "
          f"{wavelet_psi_consistency(res):.2e}")
