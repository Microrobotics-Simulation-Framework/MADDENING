"""Gate 1, §3 -- DD phi sign property and wrong-sign safety.

SPIKE CODE. Investigative.

The round-4 locality theorem: a LOCAL basis (each phi_lambda single-signed
at the sensor on its support) cannot produce a frozen-set solution whose
sensor reading has the wrong sign relative to the full solution. NON-LOCAL
bases (sine) can: an active set chosen on |b| may include modes whose
contributions to u(x_sensor) partially cancel, leaving a residual of the
wrong sign.

DD wavelets are smoother than Haar. The wavelet psi has zero mean (must
change sign); the DD-4+ scaling function has small negative side-lobes.
Question: does DD inherit the sine failure mode, or does locality still
protect it?

We measure, for a moving Gaussian source f(x;theta) on (0,1) with sensor
at x=1/3, the sign agreement between J_full (full solve) and J_frozen
(frozen active set of size K) under three selection rules:
  top-|b|, top-|c| (oracle), and a simple CDD residual bulk-chasing.

A SINE basis is included as a positive control: it must reproduce the
wrong-sign failure, proving the harness can detect it.
"""

from __future__ import annotations

import numpy as np

import dd_wavelets as dd

SIGMA = 0.04
# NB: with n_coarse=2 Dirichlet, coarse nodes sit at 1/3 and 2/3. Placing
# the sensor at 1/3 would land it ON a coarse node, where the interpolation
# property makes every detail wavelet vanish -> J_frozen == J_full trivially
# (the always-selected coarse coeff is all that matters). Pick a generic
# fine-only location so multiple levels genuinely contribute at the sensor.
SENSOR_X = 0.30


def _l2_normalise(W, h):
    norms = np.sqrt(h * np.sum(W ** 2, axis=0))
    norms[norms == 0] = 1.0
    return W / norms[None, :]


def build_basis(kind: str, n_levels: int, order: int = 4):
    """Return (Wn, levels, x, A_wave, h, sensor_idx) for a Dirichlet H^1 op."""
    if kind == "sine":
        # non-local control: sine eigenbasis of (-d2/dx2 + I) Dirichlet
        # choose N to roughly match dd sizes
        N = 2 * (2 ** n_levels) - 1
        x = np.arange(1, N + 1) / (N + 1)
        ks = np.arange(1, N + 1)
        # phi_k(x) = sqrt(2) sin(k pi x); A diagonal lambda_k=(k pi)^2+1
        Wn = np.sqrt(2.0) * np.sin(np.pi * np.outer(x, ks))
        lambdas = (ks * np.pi) ** 2 + 1.0
        A_wave = np.diag(lambdas)
        levels = np.floor(np.log2(ks)).astype(int)
        h = 1.0 / (N + 1)
        # L2 normalise (sqrt(2) sin is already ~unit on continuous; discretely:)
        Wn = _l2_normalise(Wn, h)
        sensor_idx = int(round(SENSOR_X * (N + 1))) - 1
        return Wn, levels, x, A_wave, h, sensor_idx
    else:
        W, levels, x = dd.synthesis_matrix(
            n_levels, n_coarse=2, order=order, boundary="dirichlet"
        )
        N = W.shape[0]
        h = 1.0 / (N + 1)
        Wn = _l2_normalise(W, h)
        A_phys = dd.laplacian_dirichlet(N, mass=True)
        A_wave = Wn.T @ A_phys @ Wn
        A_wave = 0.5 * (A_wave + A_wave.T)
        sensor_idx = int(np.argmin(np.abs(x - SENSOR_X)))
        return Wn, levels, x, A_wave, h, sensor_idx


def load_vector(Wn, x, h, theta):
    f = np.exp(-((x - theta) / SIGMA) ** 2)
    return h * (Wn.T @ f)


def cdd_select(A, b, levels, K):
    """Simple CDD-style residual bulk-chasing to active-set size K.

    Start from the coarsest level (always included), then greedily add
    the index with largest scaled residual until |S| = K. This is the
    spirit of Cohen-Dahmen-DeVore (residual-driven), not the full
    optimal-tree machinery.
    """
    n = len(b)
    S = list(np.where(levels == levels.min())[0])  # coarse always in
    c = np.zeros(n)
    # solve on current S
    while True:
        idx = np.array(sorted(set(S)))
        c_S = np.linalg.solve(A[np.ix_(idx, idx)], b[idx])
        c[:] = 0.0
        c[idx] = c_S
        if len(idx) >= K:
            break
        r = b - A @ c
        r[idx] = 0.0  # don't re-pick
        # bulk: add the single largest-residual index (Dörfler with small bulk)
        nxt = int(np.argmax(np.abs(r)))
        S.append(nxt)
    return idx


def cdd_select_nocoarse(A, b, levels, K):
    """CDD-style greedy residual WITHOUT the coarse-inclusion guarantee.

    Starts from the single largest-|b| index and adds by residual. Tests
    whether the wrong-sign protection comes from coarse inclusion (plan
    §3 resolution (i)) rather than from locality per se.
    """
    n = len(b)
    S = [int(np.argmax(np.abs(b)))]
    c = np.zeros(n)
    while True:
        idx = np.array(sorted(set(S)))
        c[:] = 0.0
        c[idx] = np.linalg.solve(A[np.ix_(idx, idx)], b[idx])
        if len(idx) >= K:
            break
        r = b - A @ c
        r[idx] = 0.0
        S.append(int(np.argmax(np.abs(r))))
    return idx


def frozen_solve(A, b, S):
    idx = np.array(sorted(set(S.tolist() if hasattr(S, "tolist") else S)))
    c = np.zeros(len(b))
    c[idx] = np.linalg.solve(A[np.ix_(idx, idx)], b[idx])
    return c


def run(kind, n_levels, order, thetas, K_frac):
    Wn, levels, x, A, h, sidx = build_basis(kind, n_levels, order)
    N = A.shape[0]
    K = max(4, N // K_frac)
    rows = []
    for theta in thetas:
        b = load_vector(Wn, x, h, theta)
        c_full = np.linalg.solve(A, b)
        J_full = Wn[sidx] @ c_full

        # top-|b|
        Sb = np.argsort(-np.abs(b))[:K]
        c_b = frozen_solve(A, b, Sb)
        J_b = Wn[sidx] @ c_b

        # top-|c| (oracle, from full c)
        Sc = np.argsort(-np.abs(c_full))[:K]
        c_c = frozen_solve(A, b, Sc)
        J_c = Wn[sidx] @ c_c

        # CDD (coarse-guaranteed)
        Scdd = cdd_select(A, b, levels, K)
        c_cdd = frozen_solve(A, b, Scdd)
        J_cdd = Wn[sidx] @ c_cdd

        # CDD without coarse guarantee (mechanism probe)
        Snc = cdd_select_nocoarse(A, b, levels, K)
        c_nc = frozen_solve(A, b, Snc)
        J_nc = Wn[sidx] @ c_nc

        rows.append((theta, J_full, J_b, J_c, J_cdd, J_nc))
    return N, K, rows


def fmt_sign(Jfull, J):
    ws = (Jfull * J < 0)
    err = abs(J - Jfull) / (abs(Jfull) + 1e-30)
    return f"{'WRONG' if ws else 'ok':>5}({err:5.2f})"


def phi_sign_analysis(order, n_levels=6):
    """Address plan §3 sub-questions (a) and (b) directly.

    (a) Do wavelets psi_lambda covering the sensor alternate sign?
    (b) Does the coarse scaling phi stay single-signed at the sensor?
    Also quantify the DD negative-lobe magnitude relative to the peak.
    """
    W, levels, x = dd.synthesis_matrix(
        n_levels, n_coarse=2, order=order, boundary="dirichlet"
    )
    sidx = int(np.argmin(np.abs(x - SENSOR_X)))
    print(f"--- DD-{order} phi/psi sign structure at sensor x≈{x[sidx]:.3f} ---")
    # negative-lobe magnitude of each basis fn relative to its peak
    peak = np.max(np.abs(W), axis=0)
    negfrac = -np.min(W, axis=0) / (peak + 1e-30)  # how deep the neg lobe is
    print(f"  max negative-lobe depth across all basis fns: "
          f"{negfrac.max():.3f} of peak (0 => strictly nonneg, e.g. hat)")
    # for each level, sign of basis fns evaluated at sensor (nonzero only)
    for lvl in sorted(set(levels)):
        cols = np.where(levels == lvl)[0]
        vals = W[sidx, cols]
        nz = vals[np.abs(vals) > 1e-9 * peak[cols].max()]
        if len(nz) == 0:
            continue
        npos = int((nz > 0).sum()); nneg = int((nz < 0).sum())
        print(f"  level {lvl}: {len(nz)} basis fns cover sensor "
              f"-> {npos} positive, {nneg} negative at sensor")


def main():
    thetas = [0.02, 0.04, 0.06, 0.5, 0.94, 0.96, 0.98]
    print("Sign agreement of J_frozen vs J_full; 'WRONG'=opposite sign, ()=rel err")
    print("sensor x=1/3, sigma=0.04, source Gaussian at theta")
    for kind, order, Kfrac in [
        ("sine", 4, 16),
        ("dd", 2, 16),
        ("dd", 4, 16),
        ("dd", 4, 8),
    ]:
        N, K, rows = run(kind, 6, order, thetas, Kfrac)
        label = f"{kind.upper()}" + (f"-{order}" if kind == "dd" else "")
        print("=" * 78)
        print(f"{label}  N={N}  K={K} (N/{Kfrac})")
        print(f"{'theta':>6} {'J_full':>11} {'top|b|':>13} {'top|c|':>13} "
              f"{'CDD':>13} {'CDD-nocrs':>13}")
        for theta, Jf, Jb, Jc, Jcdd, Jnc in rows:
            print(f"{theta:>6.2f} {Jf:>11.3e} "
                  f"{fmt_sign(Jf, Jb):>13} {fmt_sign(Jf, Jc):>13} "
                  f"{fmt_sign(Jf, Jcdd):>13} {fmt_sign(Jf, Jnc):>13}")
    print("=" * 78)
    phi_sign_analysis(2)
    phi_sign_analysis(4)


if __name__ == "__main__":
    main()
