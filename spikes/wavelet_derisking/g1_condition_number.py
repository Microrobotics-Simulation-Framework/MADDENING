"""Gate 1, §2 Hypothesis A -- DD Dahmen-Kunoth condition number.

SPIKE CODE. Investigative.

Question: does Deslauriers-Dubuc + Dahmen-Kunoth scaling D = 2^|lambda|
bring kappa(D^-1 A D^-1) to O(1) for the 1D/2D Laplacian, where Haar
does not?

Method:
  1. Calibrate the harness on Haar: reproduce the round-7 baseline
     (1D N=256 scaled kappa ~ 9e2, unscaled ~ 2.4e4 ballpark).
  2. Build A_wave = W^T A_phys W in an L2-normalised wavelet basis.
  3. Apply D = diag(2^level); report kappa scaled vs unscaled across N.

A_phys is the H^1 bilinear form (-d2/dx2 + I).  We use periodic BCs:
the cleanest test of the *interior* Dahmen-Kunoth norm-equivalence
claim, free of boundary-adapted-wavelet artefacts.
"""

from __future__ import annotations

import numpy as np

import dd_wavelets as dd


def _l2_normalise(W: np.ndarray, h: float) -> np.ndarray:
    """Scale each column (basis function) to unit L2 norm (trapezoid)."""
    norms = np.sqrt(h * np.sum(W ** 2, axis=0))
    norms[norms == 0] = 1.0
    return W / norms[None, :]


def cond_1d(n_levels: int, order: int, basis: str = "dd", n_coarse: int = 2):
    if basis == "haar":
        W, levels, x = dd.haar_synthesis_matrix(n_levels, n_coarse=n_coarse)
    else:
        W, levels, x = dd.synthesis_matrix(
            n_levels, n_coarse=n_coarse, order=order, boundary="periodic"
        )
    N = W.shape[0]
    h = 1.0 / N
    Wn = _l2_normalise(W, h)
    A_phys = dd.laplacian_periodic(N, mass=True)
    A_wave = Wn.T @ A_phys @ Wn
    # symmetrise tiny asymmetries from roundoff
    A_wave = 0.5 * (A_wave + A_wave.T)

    D = 2.0 ** levels.astype(float)
    Dinv = 1.0 / D
    A_scaled = (Dinv[:, None] * A_wave) * Dinv[None, :]

    k_unscaled = np.linalg.cond(A_wave)
    k_scaled = np.linalg.cond(A_scaled)
    return N, k_unscaled, k_scaled


def cond_2d(n_levels: int, order: int, basis: str = "dd", n_coarse: int = 2):
    """2D tensor-product. A_2d = A x M + M x A on the H^1 form.

    We build the 1D normalised wavelet basis and tensor it. The 2D
    Dahmen-Kunoth scaling is D_lambda = 2^{|lx| + |ly|}.
    """
    if basis == "haar":
        W, levels, x = dd.haar_synthesis_matrix(n_levels, n_coarse=n_coarse)
    else:
        W, levels, x = dd.synthesis_matrix(
            n_levels, n_coarse=n_coarse, order=order, boundary="periodic"
        )
    n = W.shape[0]
    h = 1.0 / n
    Wn = _l2_normalise(W, h)

    # 1D physical stiffness S and mass M separately for the tensor form.
    S = dd.laplacian_periodic(n, mass=False)
    M = h * np.eye(n)
    # 1D wavelet-basis stiffness and mass
    Sw = Wn.T @ S @ Wn
    Mw = Wn.T @ M @ Wn
    Sw = 0.5 * (Sw + Sw.T)
    Mw = 0.5 * (Mw + Mw.T)

    # 2D H^1 bilinear form: (Sx Mx) + (Mx Sy) + (Mx My)  [grad + mass]
    A2 = np.kron(Sw, Mw) + np.kron(Mw, Sw) + np.kron(Mw, Mw)
    A2 = 0.5 * (A2 + A2.T)

    lx = (levels[:, None] + np.zeros_like(levels)[None, :]).reshape(-1).astype(float)
    ly = (np.zeros_like(levels)[:, None] + levels[None, :]).reshape(-1).astype(float)

    def kappa_with(D):
        Dinv = 1.0 / D
        A2s = (Dinv[:, None] * A2) * Dinv[None, :]
        return np.linalg.cond(A2s)

    out = {
        "unscaled": np.linalg.cond(A2),
        # naive sum-of-levels (wrong for an isotropic operator)
        "D_sum": kappa_with(2.0 ** (lx + ly)),
        # max-level
        "D_max": kappa_with(2.0 ** np.maximum(lx, ly)),
        # proper H^1 tensor energy scaling
        "D_energy": kappa_with(np.sqrt(2.0 ** (2 * lx) + 2.0 ** (2 * ly) + 1.0)),
        # algebraic Jacobi (Hyp C): sqrt of the actual diagonal
        "D_jacobi": kappa_with(np.sqrt(np.abs(np.diag(A2)))),
    }
    return n * n, out


def cond_2d_isotropic(n_levels: int, order: int, n_coarse: int = 2):
    """2D ISOTROPIC wavelet basis (Mallat pyramid) -- the correct basis
    for an isotropic PDE operator. DK scaling D = 2^j (single level)."""
    W2, levels, n = dd.synthesis_matrix_2d_isotropic(n_levels, n_coarse, order)
    h = 1.0 / n
    # L2 normalise each 2D basis column (area element h^2)
    norms = np.sqrt((h * h) * np.sum(W2 ** 2, axis=0))
    norms[norms == 0] = 1.0
    W2n = W2 / norms[None, :]

    S = dd.laplacian_periodic(n, mass=False)
    M = h * np.eye(n)
    A2_phys = np.kron(S, M) + np.kron(M, S) + np.kron(M, M)
    A2 = W2n.T @ A2_phys @ W2n
    A2 = 0.5 * (A2 + A2.T)

    lev = levels.astype(float)

    def kappa_with(D):
        Dinv = 1.0 / D
        return np.linalg.cond((Dinv[:, None] * A2) * Dinv[None, :])

    out = {
        "unscaled": np.linalg.cond(A2),
        "D_2^j": kappa_with(2.0 ** lev),
        "D_jacobi": kappa_with(np.sqrt(np.abs(np.diag(A2)))),
    }
    return n * n, out


def main():
    print("=" * 72)
    print("CALIBRATION: Haar, 1D, periodic H^1 form")
    print("  (round-7 reported scaled kappa ~9e2 @ N=256, unscaled ~2.4e4)")
    print("-" * 72)
    print(f"{'N':>6} {'kappa_unscaled':>16} {'kappa_scaled':>14}")
    for nl in range(4, 9):  # N = 2^nl * n_coarse
        N, ku, ks = cond_1d(nl, order=4, basis="haar", n_coarse=1)
        print(f"{N:>6} {ku:>16.3e} {ks:>14.3e}")

    for order in (2, 4, 6):
        print("=" * 72)
        print(f"DD-{order} (order {order}), 1D, periodic H^1 form")
        print("-" * 72)
        print(f"{'N':>6} {'kappa_unscaled':>16} {'kappa_scaled':>14}")
        for nl in range(3, 8):
            N, ku, ks = cond_1d(nl, order=order, basis="dd", n_coarse=2)
            print(f"{N:>6} {ku:>16.3e} {ks:>14.3e}")

    print("=" * 72)
    print("2D tensor-product, DD-4, competing scalings for kappa")
    print("  D_sum = 2^(lx+ly) [naive], D_max = 2^max(lx,ly),")
    print("  D_energy = sqrt(2^2lx + 2^2ly + 1) [proper H^1], D_jacobi = sqrt(diag A)")
    print("-" * 72)
    cols = ["unscaled", "D_sum", "D_max", "D_energy", "D_jacobi"]
    hdr = f"{'basis':>6} {'N':>6} " + " ".join(f"{c:>11}" for c in cols)
    print(hdr)
    for nl in (3, 4, 5):
        N, out = cond_2d(nl, order=4, basis="dd", n_coarse=2)
        row = f"{'DD-4':>6} {N:>6} " + " ".join(f"{out[c]:>11.3e}" for c in cols)
        print(row)
    for nl in (3, 4, 5):
        N, out = cond_2d(nl, order=4, basis="haar", n_coarse=1)
        row = f"{'haar':>6} {N:>6} " + " ".join(f"{out[c]:>11.3e}" for c in cols)
        print(row)

    print("=" * 72)
    print("2D ISOTROPIC basis (Mallat pyramid), DD-4 -- the CORRECT 2D test")
    print("-" * 72)
    cols2 = ["unscaled", "D_2^j", "D_jacobi"]
    print(f"{'N':>6} " + " ".join(f"{c:>12}" for c in cols2))
    for nl in (2, 3, 4):
        N, out = cond_2d_isotropic(nl, order=4, n_coarse=2)
        print(f"{N:>6} " + " ".join(f"{out[c]:>12.3e}" for c in cols2))


if __name__ == "__main__":
    main()
