"""D1 derisk — does periodic + zero-padding approximate the drug no-flux BC?

Reaction-diffusion -div(D grad c) + k c = s has decay length lambda = sqrt(D/k).
The wavelet node is periodic; the question is whether solving on a padded periodic
domain (c -> 0 in the pad, i.e. ~homogeneous Dirichlet at the pad edge) reproduces
the physically-correct no-flux (Neumann) solution inside the ROI.

Physics intuition: if lambda << distance-to-boundary, c ~ 0 at the wall regardless,
so Dirichlet-via-padding and Neumann agree. If lambda ~ ROI, the BC matters.

Method (1D FD reference, exact for the BC question, basis-independent):
  - ROI = [0, L], unit source at centre, constant D, k = D / lambda^2.
  - Reference: no-flux (Neumann) on [0, L].
  - Candidate: periodic domain [0, L*(1+2*pad)] with the ROI centred, zero Dirichlet
    at the padded ends (what zero-padding achieves), restricted back to the ROI.
Sweep lambda/L and pad fraction; report max rel. error in the ROI.

Pass: for a realistic regime (lambda/L <~ 0.25) with modest padding, rel err < 1%
=> M9 is a no-op.  If even generous padding can't get <1% at lambda/L ~ 0.5, M9
needs a true no-flux basis (STOP-and-report).
"""
from __future__ import annotations
import numpy as np


def solve_neumann(n, L, lam):
    """-D c'' + k c = s on [0,L], homogeneous Neumann, unit point source at centre."""
    h = L / (n - 1)
    D = 1.0
    k = D / lam ** 2
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = 2 * D / h ** 2 + k
        if i > 0:
            A[i, i - 1] = -D / h ** 2
        if i < n - 1:
            A[i, i + 1] = -D / h ** 2
    # Neumann: mirror (ghost = interior neighbour) -> halve the one-sided coupling
    A[0, 0] = D / h ** 2 + k; A[0, 1] = -D / h ** 2
    A[-1, -1] = D / h ** 2 + k; A[-1, -2] = -D / h ** 2
    s = np.zeros(n); s[n // 2] = 1.0 / h
    return np.linalg.solve(A, s)


def solve_padded_dirichlet(n, L, lam, pad):
    """Same PDE on a padded domain with Dirichlet-0 ends; return the ROI slice."""
    Lp = L * (1 + 2 * pad)
    npad = int(round(n * (1 + 2 * pad)))
    h = Lp / (npad - 1)
    D = 1.0; k = D / lam ** 2
    A = np.zeros((npad, npad))
    for i in range(npad):
        A[i, i] = 2 * D / h ** 2 + k
        if i > 0:
            A[i, i - 1] = -D / h ** 2
        if i < npad - 1:
            A[i, i + 1] = -D / h ** 2
    # Dirichlet-0 ends (rows 0 and npad-1 pinned)
    A[0, :] = 0; A[0, 0] = 1
    A[-1, :] = 0; A[-1, -1] = 1
    s = np.zeros(npad); s[npad // 2] = 1.0 / h
    c = np.linalg.solve(A, s)
    # ROI is the centred L-window
    i0 = (npad - n) // 2
    return c[i0:i0 + n]


if __name__ == "__main__":
    n, L = 401, 1.0
    print(f"1D reaction-diffusion, ROI=[0,{L}], no-flux reference vs padded-Dirichlet")
    print(f"{'lam/L':>7} {'pad':>5} {'max rel err in ROI':>19}")
    worst_realistic = 0.0
    for lam_over_L in [0.1, 0.25, 0.5]:
        for pad in [0.25, 0.5, 1.0]:
            ref = solve_neumann(n, L, lam_over_L * L)
            cand = solve_padded_dirichlet(n, L, lam_over_L * L, pad)
            err = np.max(np.abs(cand - ref)) / (np.max(np.abs(ref)) + 1e-30)
            print(f"{lam_over_L:>7.2f} {pad:>5.2f} {err:>19.3e}")
            if lam_over_L <= 0.25 and pad >= 0.5:
                worst_realistic = max(worst_realistic, err)
        print()
    print(f"worst @ (lam/L<=0.25, pad>=0.5): {worst_realistic:.3e}  ->",
          "PASS — periodic+padding suffices, M9 is a no-op" if worst_realistic < 1e-2
          else "MARGINAL/FAIL — M9 may need a true no-flux basis (see FINDINGS_D5 on scope)")
