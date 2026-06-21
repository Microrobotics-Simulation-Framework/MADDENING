"""Round-5 Investigation 2: diagonal oracle selection.

Hypothesis: score_i = |b_i| / a_ii (where a_ii = diag(A) in the basis)
approximates |c_i| at zero cost (no preliminary solve required).

For Haar 1D + FD Dirichlet (-Delta + I), a_ii is the diagonal of
A_HAAR.  In production we'd want it computable in O(N) without
materialising A; for the spike we read it off the precomputed
matrix.

Parts A (convergence sweep), B (two-pass warm start), C (cost
structure analysis -- discussion), D (selection criterion table).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from locality_theorem import (
    N, W_HAAR, A_HAAR, EYE_N, source_grid, x_grid, SENSOR_IDX,
    J_haar_full,
)

A_HAAR_DIAG = jnp.diag(A_HAAR)
THETAS = [0.30, 0.42, 0.55, 0.04]   # 0.04 is the boundary case


def topk_mask_np(score, k):
    sorted_s = np.sort(score)
    return score >= sorted_s[-k]


def J_haar_under(theta, mask):
    b = W_HAAR @ source_grid(jnp.asarray(theta))
    m = jnp.asarray(mask)
    A_eff = jnp.where(m[:, None], m[None, :] * A_HAAR, EYE_N)
    b_eff = m * b
    c = jnp.linalg.solve(A_eff, b_eff)
    u = W_HAAR.T @ c
    return float(u[SENSOR_IDX])


def grad_haar_under(theta, mask):
    def J_fn(th):
        b = W_HAAR @ source_grid(th)
        m = jax.lax.stop_gradient(jnp.asarray(mask))
        A_eff = jnp.where(m[:, None], m[None, :] * A_HAAR, EYE_N)
        b_eff = m * b
        c = jnp.linalg.solve(A_eff, b_eff)
        u = W_HAAR.T @ c
        return u[SENSOR_IDX]
    return float(jax.grad(J_fn)(jnp.asarray(theta)))


def grad_haar_full_local(theta):
    return float(jax.grad(J_haar_full)(jnp.asarray(theta)))


def relerr(x, ref):
    return abs(x - ref) / (abs(ref) + 1e-30)


# ---- Part A: convergence sweep ----
def part_a():
    print("# Part A -- Haar diagonal-oracle convergence sweep")
    print()
    print(f"  Haar basis, N={N}, (-Delta + I) Dirichlet FD")
    A_diag_np = np.asarray(A_HAAR_DIAG)
    print(f"  diag(A_HAAR) summary: min={A_diag_np.min():.3e}, "
          f"max={A_diag_np.max():.3e}, mean={A_diag_np.mean():.3e}")
    print()
    print(f"  Selection criteria:")
    print(f"    b      = top-|b|        (round-2 baseline)")
    print(f"    c      = top-|c| oracle (requires full solve)")
    print(f"    b/diag = top-|b/a_ii|   (diagonal oracle, 0 solve cost)")
    print()
    for theta in THETAS:
        J_ref = float(J_haar_full(jnp.asarray(theta)))
        g_ref = grad_haar_full_local(theta)
        print(f"## theta = {theta}")
        print(f"   J_full = {J_ref:+.5e}, dJ/dth_full = {g_ref:+.5e}")
        b = np.asarray(W_HAAR @ source_grid(jnp.asarray(theta)))
        for k in [N // 16, N // 8, N // 4, N // 2]:
            # top-|b|
            mb = topk_mask_np(np.abs(b), k)
            Jb = J_haar_under(theta, mb)
            gb = grad_haar_under(theta, mb)
            # top-|c|
            c_full = np.linalg.solve(np.asarray(A_HAAR), b)
            mc = topk_mask_np(np.abs(c_full), k)
            Jc = J_haar_under(theta, mc)
            gc = grad_haar_under(theta, mc)
            # top-|b/diag|
            mbd = topk_mask_np(np.abs(b) / A_diag_np, k)
            Jbd = J_haar_under(theta, mbd)
            gbd = grad_haar_under(theta, mbd)
            print(f"   k={k:>3d}  "
                  f"b: Jerr={relerr(Jb,J_ref):.2e} gerr={relerr(gb,g_ref):.2e}  | "
                  f"c: Jerr={relerr(Jc,J_ref):.2e} gerr={relerr(gc,g_ref):.2e}  | "
                  f"b/diag: Jerr={relerr(Jbd,J_ref):.2e} gerr={relerr(gbd,g_ref):.2e}")
            # Mask agreement between c and b/diag
            agreement = int(np.sum(mc == mbd)) / N
            print(f"            mask agreement (c vs b/diag): "
                  f"{int(np.sum(mc & mbd))}/{k} matched, "
                  f"{agreement:.3f} overall")
        print()


# ---- Part B: two-pass warm start on trajectory ----
def part_b():
    print("\n# Part B -- two-pass warm start on smooth trajectory")
    print()
    T = 30
    K_BASE = N // 8   # k_active

    def trajectory(t):
        return 0.3 + 0.3 * np.sin(2.0 * np.pi * t / T)

    A_diag_np = np.asarray(A_HAAR_DIAG)
    strategies = ["b", "c_prev", "b_div_diag", "two_pass", "oracle"]
    errs = {s: [] for s in strategies}
    c_prev = None

    for t in range(T):
        theta = trajectory(t)
        b = np.asarray(W_HAAR @ source_grid(jnp.asarray(theta)))
        J_ref = float(J_haar_full(jnp.asarray(theta)))

        # b
        mb = topk_mask_np(np.abs(b), K_BASE)
        errs["b"].append(relerr(J_haar_under(theta, mb), J_ref))

        # c_prev
        if c_prev is None:
            m_cp = mb
        else:
            m_cp = topk_mask_np(np.abs(c_prev), K_BASE)
        errs["c_prev"].append(relerr(J_haar_under(theta, m_cp), J_ref))

        # b/diag
        mbd = topk_mask_np(np.abs(b) / A_diag_np, K_BASE)
        errs["b_div_diag"].append(relerr(J_haar_under(theta, mbd), J_ref))

        # two_pass: K_HALF warm + K_HALF residual-selected
        K_HALF = K_BASE // 2
        m_warm = topk_mask_np(np.abs(b) / A_diag_np, K_HALF)
        A_eff = np.where(m_warm[:, None],
                         m_warm[None, :] * np.asarray(A_HAAR),
                         np.eye(N))
        c_warm = np.linalg.solve(A_eff, m_warm * b)
        # CDD-style residual criterion: add modes with large |residual|
        residual = b - np.asarray(A_HAAR) @ c_warm
        # restrict to modes NOT in warm
        res_score = np.where(m_warm, -np.inf, np.abs(residual) / A_diag_np)
        # take top-(K_BASE - K_HALF) from residual outside warm
        n_add = K_BASE - K_HALF
        sorted_res = np.sort(res_score)
        thr = sorted_res[-n_add]
        m_add = res_score >= thr
        m_2p = m_warm | m_add
        # enforce exact size: take top K_BASE by combined-score if oversized
        if int(m_2p.sum()) != K_BASE:
            comb = np.where(m_2p, 1.0, 0.0)   # set membership tie-break
            # at this point m_2p might be size > K_BASE if many ties
            # accept current size for diagnostic purposes
            pass
        errs["two_pass"].append(relerr(J_haar_under(theta, m_2p), J_ref))

        # oracle
        c_full = np.linalg.solve(np.asarray(A_HAAR), b)
        m_or = topk_mask_np(np.abs(c_full), K_BASE)
        errs["oracle"].append(relerr(J_haar_under(theta, m_or), J_ref))

        c_prev = c_full

    print(f"  T={T}, K_active={K_BASE}")
    print(f"  {'strategy':>12}  {'mean J_err':>11}  {'max J_err':>11}  "
          f"{'wrong-sign':>11}")
    for s in strategies:
        ar = np.array(errs[s])
        n_ws = int(np.sum(ar > 1.0))
        print(f"  {s:>12}  {ar.mean():>11.3e}  {ar.max():>11.3e}  {n_ws:>11d}")


if __name__ == "__main__":
    part_a()
    part_b()
