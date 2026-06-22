"""Gate 2 follow-up -- is the blindness trap a NON-LOCAL-basis phenomenon?

SPIKE CODE. Investigative.

The 3D wavelet trap sweep found NO blindness (ratio>=0.9 everywhere) even
with top-|b| selection -- contradicting the 1D spike, which found strong
blindness (ratio->0) at theta=0.5. The 1D spike used the SINE (non-local)
basis; the 3D test uses the DD WAVELET (local) basis.

Hypothesis: selection-induced blindness is a non-local-basis phenomenon.
In a local basis, top-|b| picks modes near the (moving) source, so the
active set tracks theta and never goes symmetric-blind. Confirm in 1D:
canonical trap theta=0.5, sensor x=1/3, top-|b| selection.

blindness_ratio = |dJ_frozen/dtheta| / |dJ_full/dtheta|, active set frozen
at the evaluation theta.
"""

from __future__ import annotations

import numpy as np

import g1_wrong_sign as g1


def blindness_at(kind, theta, order=4, n_levels=6, Kfrac=16):
    Wn, levels, x, A, h, sidx = g1.build_basis(kind, n_levels, order)
    N = A.shape[0]
    K = max(4, N // Kfrac)
    srow = Wn[sidx]

    def J_full(th):
        b = g1.load_vector(Wn, x, h, th)
        return srow @ np.linalg.solve(A, b)

    def select(th):
        b = g1.load_vector(Wn, x, h, th)
        return np.argsort(-np.abs(b))[:K]  # top-|b|

    def J_frozen(th, S):
        b = g1.load_vector(Wn, x, h, th)
        c = np.zeros(N); c[S] = np.linalg.solve(A[np.ix_(S, S)], b[S])
        return srow @ c

    eps = 1e-5
    gfull = (J_full(theta + eps) - J_full(theta - eps)) / (2 * eps)
    S = select(theta)
    gfrz = (J_frozen(theta + eps, S) - J_frozen(theta - eps, S)) / (2 * eps)
    return abs(gfrz) / (abs(gfull) + 1e-30), gfull, gfrz


def main():
    print("Blindness ratio under top-|b| selection at the canonical 1D trap")
    print("theta=0.5 (source symmetric), sensor x=1/3 (asymmetric), K=N/16")
    print("ratio < 0.7 => trapped/blind")
    print(f"{'basis':>8} {'theta':>6} {'ratio':>8} {'g_full':>11} {'g_frozen':>11}")
    for theta in (0.50, 0.48, 0.42):
        for kind, order in (("sine", 4), ("dd", 2), ("dd", 4)):
            r, gf, gz = blindness_at(kind, theta, order=order)
            label = kind.upper() + (f"-{order}" if kind == "dd" else "")
            flag = "  <-- BLIND" if r < 0.7 else ""
            print(f"{label:>8} {theta:>6.2f} {r:>8.3f} {gf:>11.3e} {gz:>11.3e}{flag}")
        print()


if __name__ == "__main__":
    # sensor at 1/3 for the canonical trap (note: for DD this is a coarse
    # node; we deliberately use it to match the 1D spike's trap geometry,
    # and also report the generic-sensor case below)
    g1.SENSOR_X = 1.0 / 3.0
    main()
    print("=" * 60)
    print("Repeat with generic (non-coarse-node) sensor x=0.30:")
    g1.SENSOR_X = 0.30
    main()
