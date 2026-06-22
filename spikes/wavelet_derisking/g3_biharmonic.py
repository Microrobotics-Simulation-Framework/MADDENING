"""Gate 3, §5 -- stream-function (biharmonic) preconditioning derisk.

SPIKE CODE. Investigative.

The plan's Option 1 reformulates the 2D incompressible cavity as a scalar
biharmonic problem in the stream function: Delta^2 psi = -omega. The whole
AdaptiveNode/CDD/DK stack then applies, IF the biharmonic (H^2-elliptic,
t=2) admits Dahmen-Kunoth scaling D = 2^{2|lambda|}.

The derisking-relevant question is exactly that conditioning claim. The full
Ghia-Ghia-Shin Navier-Stokes cavity match is a multi-week implementation
task (a nonlinear time-dependent solve), explicitly out of spike scope --
we note it as implementation-phase, consistent with the plan's §8 deferral
philosophy. Here we test:

  (a) Does D = 2^{2j} give O(1) kappa for the biharmonic in DD-4?
  (b) DD-2 (piecewise linear, NOT in H^2) should FAIL -- a control.
  (c) Compare t=2 DK vs t=1 DK vs algebraic Jacobi.
"""

from __future__ import annotations

import numpy as np

import dd_wavelets as dd


def _l2_normalise(W, h):
    norms = np.sqrt(h * np.sum(W ** 2, axis=0))
    norms[norms == 0] = 1.0
    return W / norms[None, :]


def biharmonic_periodic(N):
    """H^2 bilinear-form matrix ~ integral (u'')^2 + (u)^2, periodic.

    B = (D2)^T M (D2) + mass, with D2 the periodic 2nd-difference (~1/h^2)
    and M = h I (lumped). Entries of the stiffness part ~ 1/h^3.
    """
    h = 1.0 / N
    idx = np.arange(N)
    D2 = np.zeros((N, N))
    D2[idx, idx] = -2.0 / h ** 2
    D2[idx, (idx + 1) % N] += 1.0 / h ** 2
    D2[idx, (idx - 1) % N] += 1.0 / h ** 2
    M = h * np.eye(N)
    B = D2.T @ M @ D2 + M
    return 0.5 * (B + B.T)


def kappa_biharmonic(n_levels, order, n_coarse=2):
    W, levels, x = dd.synthesis_matrix(n_levels, n_coarse=n_coarse,
                                       order=order, boundary="periodic")
    N = W.shape[0]
    h = 1.0 / N
    Wn = _l2_normalise(W, h)
    B = Wn.T @ biharmonic_periodic(N) @ Wn
    B = 0.5 * (B + B.T)
    lev = levels.astype(float)

    def kap(D):
        return np.linalg.cond((B / D[:, None]) / D[None, :])

    return N, {
        "unscaled": np.linalg.cond(B),
        "t=1 (2^j)": kap(2.0 ** lev),
        "t=2 (2^2j)": kap(2.0 ** (2 * lev)),
        "jacobi": kap(np.sqrt(np.abs(np.diag(B)))),
    }


def main():
    print("Biharmonic (stream-function) preconditioning, 1D periodic")
    print("H^2-elliptic operator; DK theory predicts t=2 scaling D=2^{2j}")
    print("Approximation order needed for H^2: >=3 -> DD-4 ok, DD-2 should FAIL")
    for order in (2, 4, 6):
        print("=" * 70)
        print(f"DD-{order}")
        cols = ["unscaled", "t=1 (2^j)", "t=2 (2^2j)", "jacobi"]
        print(f"{'N':>6} " + " ".join(f"{c:>12}" for c in cols))
        for nl in range(3, 8):
            N, out = kappa_biharmonic(nl, order)
            print(f"{N:>6} " + " ".join(f"{out[c]:>12.3e}" for c in cols))


if __name__ == "__main__":
    main()
