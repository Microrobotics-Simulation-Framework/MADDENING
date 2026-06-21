"""Round-4 Investigation 3: 2D extension -- locality theorem and
trap structure on the tensor-product Haar / sine bases.

Setup:
  - 32x32 interior grid, dx = 1/33, N = 1024
  - operator (-Delta + I) with Dirichlet BCs (physical-space FD)
  - source f = exp(-((x-tx)^2 + (y-ty)^2)/sigma^2), sigma = 0.1
  - sensor at (0.7, 0.6)
  - bases: sine = DST-I_x kron DST-I_y (diagonalises operator);
           haar = Haar_x kron Haar_y (non-diagonal)
  - selection: top-|b| (RHS), top-|c| (oracle full solve)

Parts:
  A. Setup validation -- cross-basis full-solve agreement < 1e-10
  B. 2D locality theorem -- sweep k_active, both bases, both
     selection criteria
  C. 2D trap structure -- 7x7 grid of (tx, ty)
  D. 2D rolling and cold-start on a smooth trajectory
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# ---- 1D building blocks ----
N_1D = 32
N = N_1D * N_1D  # 1024
SIGMA = 0.1
SENSOR_X, SENSOR_Y = 0.7, 0.6

dx = 1.0 / (N_1D + 1)
x_grid_1d_np = np.arange(1, N_1D + 1) * dx  # interior, length N_1D
x_grid_1d = jnp.asarray(x_grid_1d_np)

# 1D FD Laplacian (Dirichlet)
A_LAP_1D = (2.0 / dx ** 2) * np.eye(N_1D) - (1.0 / dx ** 2) * (
    np.eye(N_1D, k=1) + np.eye(N_1D, k=-1)
)
EYE_1D = np.eye(N_1D)

# 2D operator A = -Delta + I = (-Delta_x kron I + I kron -Delta_y) + I
A_PHYS_2D = (np.kron(A_LAP_1D, EYE_1D) + np.kron(EYE_1D, A_LAP_1D)
             + np.eye(N))


# DST-I (1D)
ks_1d_np = np.arange(1, N_1D + 1)
S_1D_np = np.sqrt(2.0 / (N_1D + 1)) * np.sin(
    np.pi * np.outer(ks_1d_np, ks_1d_np) / (N_1D + 1)
)
SINE_EIGS_1D_np = 2.0 * (1.0 - np.cos(np.pi * ks_1d_np / (N_1D + 1))) / dx ** 2

# 2D sine basis: kron of two 1D DST-I matrices.
# Action on flattened f (row-major, idx = ix*N_1D + iy):
#   c_2d_flat = S_2D @ f_flat
S_2D_np = np.kron(S_1D_np, S_1D_np)
# 2D eigenvalues for (-Delta + I): lambda_kx + lambda_ky + 1
SINE_EIGS_2D_np = (SINE_EIGS_1D_np[:, None] + SINE_EIGS_1D_np[None, :]).flatten() + 1.0


def haar_matrix(n):
    W = np.eye(n)
    sz = n
    while sz > 1:
        H = np.zeros((sz, sz))
        inv = 1.0 / np.sqrt(2.0)
        for i in range(sz // 2):
            H[i, 2 * i] = inv
            H[i, 2 * i + 1] = inv
            H[sz // 2 + i, 2 * i] = inv
            H[sz // 2 + i, 2 * i + 1] = -inv
        W[:sz] = H @ W[:sz]
        sz //= 2
    return W


W_HAAR_1D_np = haar_matrix(N_1D)
W_HAAR_2D_np = np.kron(W_HAAR_1D_np, W_HAAR_1D_np)
A_HAAR_2D_np = W_HAAR_2D_np @ A_PHYS_2D @ W_HAAR_2D_np.T

# Move to jax
S_2D = jnp.asarray(S_2D_np)
SINE_EIGS_2D = jnp.asarray(SINE_EIGS_2D_np)
W_HAAR_2D = jnp.asarray(W_HAAR_2D_np)
A_HAAR_2D = jnp.asarray(A_HAAR_2D_np)
A_PHYS_2D_J = jnp.asarray(A_PHYS_2D)
EYE_N = jnp.eye(N)

# Sensor index
SENSOR_IX = int(round(SENSOR_X * (N_1D + 1))) - 1
SENSOR_IY = int(round(SENSOR_Y * (N_1D + 1))) - 1
SENSOR_FLAT = SENSOR_IX * N_1D + SENSOR_IY


def source_grid_2d(theta_x, theta_y):
    X = x_grid_1d[:, None]
    Y = x_grid_1d[None, :]
    f = jnp.exp(-((X - theta_x) ** 2 + (Y - theta_y) ** 2) / SIGMA ** 2)
    return f.reshape(-1)


def J_sine_full(theta_x, theta_y):
    f = source_grid_2d(theta_x, theta_y)
    b = S_2D @ f
    c = b / SINE_EIGS_2D
    u = S_2D.T @ c
    return u[SENSOR_FLAT]


def J_sine_frozen(theta_x, theta_y, k_active, selection):
    """selection: 'b' (top-|b|) or 'c' (top-|c|)."""
    f = source_grid_2d(theta_x, theta_y)
    b = S_2D @ f
    if selection == "b":
        score = jnp.abs(b)
    else:
        score = jnp.abs(b / SINE_EIGS_2D)
    thr = jnp.sort(score)[-k_active]
    mask = jax.lax.stop_gradient(score >= thr)
    c = jnp.where(mask, b / SINE_EIGS_2D, 0.0)
    u = S_2D.T @ c
    return u[SENSOR_FLAT]


def J_haar_full(theta_x, theta_y):
    f = source_grid_2d(theta_x, theta_y)
    b = W_HAAR_2D @ f
    c = jnp.linalg.solve(A_HAAR_2D, b)
    u = W_HAAR_2D.T @ c
    return u[SENSOR_FLAT]


def J_haar_frozen(theta_x, theta_y, k_active, selection):
    f = source_grid_2d(theta_x, theta_y)
    b = W_HAAR_2D @ f
    if selection == "b":
        score = jnp.abs(b)
    else:
        c_full = jnp.linalg.solve(A_HAAR_2D, b)
        score = jnp.abs(c_full)
    thr = jnp.sort(score)[-k_active]
    mask = jax.lax.stop_gradient(score >= thr)
    A_eff = jnp.where(mask[:, None], mask[None, :] * A_HAAR_2D, EYE_N)
    b_eff = mask * b
    c = jnp.linalg.solve(A_eff, b_eff)
    u = W_HAAR_2D.T @ c
    return u[SENSOR_FLAT]


# Vector gradients
def grad_xy_full_sine(theta_x, theta_y):
    return (jax.grad(J_sine_full, argnums=0)(theta_x, theta_y),
            jax.grad(J_sine_full, argnums=1)(theta_x, theta_y))


def grad_xy_full_haar(theta_x, theta_y):
    return (jax.grad(J_haar_full, argnums=0)(theta_x, theta_y),
            jax.grad(J_haar_full, argnums=1)(theta_x, theta_y))


def grad_xy_frozen_sine(theta_x, theta_y, k, sel):
    return (jax.grad(J_sine_frozen, argnums=0)(theta_x, theta_y, k, sel),
            jax.grad(J_sine_frozen, argnums=1)(theta_x, theta_y, k, sel))


def grad_xy_frozen_haar(theta_x, theta_y, k, sel):
    return (jax.grad(J_haar_frozen, argnums=0)(theta_x, theta_y, k, sel),
            jax.grad(J_haar_frozen, argnums=1)(theta_x, theta_y, k, sel))


# ---- Part A ----
def part_a():
    print("# Part A -- 2D setup validation")
    tx, ty = 0.3, 0.4
    J_s = float(J_sine_full(jnp.asarray(tx), jnp.asarray(ty)))
    J_h = float(J_haar_full(jnp.asarray(tx), jnp.asarray(ty)))
    print(f"  N = {N} ({N_1D}x{N_1D}), sigma = {SIGMA}, "
          f"sensor = ({SENSOR_X}, {SENSOR_Y})")
    print(f"  Sensor flat idx = {SENSOR_FLAT} "
          f"(ix={SENSOR_IX}, iy={SENSOR_IY})")
    print(f"  J_sine_full  at theta=({tx}, {ty}) = {J_s:+.6e}")
    print(f"  J_haar_full  at theta=({tx}, {ty}) = {J_h:+.6e}")
    rel = abs(J_s - J_h) / (abs(J_s) + 1e-30)
    print(f"  cross-basis rel diff = {rel:.2e}  "
          f"{'PASS' if rel < 1e-10 else 'FAIL'}")


def relerr(x, ref):
    return float(abs(x - ref) / (abs(ref) + 1e-30))


def vec_relerr(g, gref):
    g = np.asarray(g); gref = np.asarray(gref)
    return float(np.linalg.norm(g - gref) / (np.linalg.norm(gref) + 1e-30))


# ---- Part B ----
def part_b():
    print("\n# Part B -- 2D locality theorem")
    tx, ty = 0.42, 0.35
    print(f"  theta = ({tx}, {ty}), sensor at "
          f"(x_s={SENSOR_X}, y_s={SENSOR_Y})")
    tx_j, ty_j = jnp.asarray(tx), jnp.asarray(ty)
    J_s_ref = float(J_sine_full(tx_j, ty_j))
    J_h_ref = float(J_haar_full(tx_j, ty_j))
    gx_s_ref, gy_s_ref = grad_xy_full_sine(tx_j, ty_j)
    gx_h_ref, gy_h_ref = grad_xy_full_haar(tx_j, ty_j)
    g_s_ref = (float(gx_s_ref), float(gy_s_ref))
    g_h_ref = (float(gx_h_ref), float(gy_h_ref))
    print(f"  sine: J = {J_s_ref:+.4e}, grad = ({g_s_ref[0]:+.4e}, "
          f"{g_s_ref[1]:+.4e})")
    print(f"  haar: J = {J_h_ref:+.4e}, grad = ({g_h_ref[0]:+.4e}, "
          f"{g_h_ref[1]:+.4e})")
    print()
    print(f"  {'k_active':>8}  {'sel':>4}  | {'sine_Jerr':>10} "
          f"{'sine_gerr':>10}  | {'haar_Jerr':>10} {'haar_gerr':>10}")
    for k in [N // 16, N // 8, N // 4, N // 2]:
        for sel in ["b", "c"]:
            J_s = float(J_sine_frozen(tx_j, ty_j, k, sel))
            g_s = grad_xy_frozen_sine(tx_j, ty_j, k, sel)
            J_h = float(J_haar_frozen(tx_j, ty_j, k, sel))
            g_h = grad_xy_frozen_haar(tx_j, ty_j, k, sel)
            sJe = relerr(J_s, J_s_ref)
            sGe = vec_relerr(g_s, g_s_ref)
            hJe = relerr(J_h, J_h_ref)
            hGe = vec_relerr(g_h, g_h_ref)
            print(f"  {k:>8d}  {sel:>4}  | {sJe:>10.3e} "
                  f"{sGe:>10.3e}  | {hJe:>10.3e} {hGe:>10.3e}")


# ---- Part C ----
def part_c():
    print("\n# Part C -- 2D trap structure (7x7 sweep, sine basis, K=N/8)")
    k = N // 8
    print(f"  N = {N}, k_active = {k}, top-|b| selection")
    print(f"  Sensor at (0.7, 0.6) -- off-centre")
    print()
    grid = np.linspace(0.15, 0.85, 7)

    def blind(tx, ty):
        tx_j, ty_j = jnp.asarray(tx), jnp.asarray(ty)
        gx_full = float(jax.grad(J_sine_full, argnums=0)(tx_j, ty_j))
        gy_full = float(jax.grad(J_sine_full, argnums=1)(tx_j, ty_j))
        norm_full = np.sqrt(gx_full ** 2 + gy_full ** 2)
        gx_fr = float(jax.grad(J_sine_frozen, argnums=0)(tx_j, ty_j, k, "b"))
        gy_fr = float(jax.grad(J_sine_frozen, argnums=1)(tx_j, ty_j, k, "b"))
        norm_fr = np.sqrt(gx_fr ** 2 + gy_fr ** 2)
        return norm_fr / (norm_full + 1e-30), gx_full, gy_full

    print(f"  {'tx \\ ty':>10}", end="")
    for ty in grid:
        print(f"  {ty:>8.3f}", end="")
    print()
    blind_grid = np.zeros((7, 7))
    for i, tx in enumerate(grid):
        print(f"  {tx:>8.3f}: ", end="")
        for j, ty in enumerate(grid):
            r, _, _ = blind(tx, ty)
            blind_grid[i, j] = r
            print(f"  {r:>8.4f}", end="")
        print()
    n_blind = int(np.sum(blind_grid < 0.3))
    n_partial = int(np.sum((blind_grid >= 0.3) & (blind_grid < 0.7)))
    n_good = int(np.sum(blind_grid >= 0.7))
    print()
    print(f"  Blind (<0.3): {n_blind}, partial (0.3-0.7): {n_partial}, "
          f"good (>=0.7): {n_good}")
    # Symmetry analysis
    print()
    print("  Symmetry check: which rows/cols are systematically low?")
    print(f"    column means (varying tx): "
          f"{[f'{x:.3f}' for x in blind_grid.mean(axis=0)]}")
    print(f"    row means (varying ty): "
          f"{[f'{x:.3f}' for x in blind_grid.mean(axis=1)]}")


# ---- Part D ----
def part_d():
    print("\n# Part D -- 2D rolling + cold-start on smooth trajectory")
    T = 30
    k = N // 8

    def trajectory(t):
        return (0.3 + 0.3 * np.sin(2.0 * np.pi * t / T),
                0.35 + 0.25 * np.cos(2.0 * np.pi * t / T))

    print(f"  theta(t) = (0.3+0.3sin, 0.35+0.25cos), T={T}, k={k}")
    print()
    for basis in ["sine", "haar"]:
        if basis == "sine":
            J_full_fn = J_sine_full
            J_frozen_fn = J_sine_frozen
            def transform(f): return S_2D @ f
            def inv_lambda_apply(b): return b / SINE_EIGS_2D
            def solve_full_for_c(b): return b / SINE_EIGS_2D
        else:
            J_full_fn = J_haar_full
            J_frozen_fn = J_haar_frozen
            def transform(f): return W_HAAR_2D @ f
            def solve_full_for_c(b): return jnp.linalg.solve(A_HAAR_2D, b)
        print(f"  Basis: {basis}")
        c_prev_np = None
        errs = {"b": [], "rolling": [], "cold-coarse": [], "oracle": []}
        for t in range(T):
            tx, ty = trajectory(t)
            tx_j, ty_j = jnp.asarray(tx), jnp.asarray(ty)
            J_ref = float(J_full_fn(tx_j, ty_j))

            # top-|b|
            J_b = float(J_frozen_fn(tx_j, ty_j, k, "b"))
            errs["b"].append(relerr(J_b, J_ref))

            # top-|c_prev|
            if c_prev_np is None:
                J_rolling = J_b
            else:
                # Use stored c_prev to build mask (compute outside jax)
                mag_prev = np.abs(c_prev_np)
                thr_np = np.sort(mag_prev)[-k]
                mask_np = mag_prev >= thr_np
                if basis == "sine":
                    f = np.asarray(source_grid_2d(tx_j, ty_j))
                    b_np = np.asarray(S_2D_np @ f)
                    c_np = np.where(mask_np, b_np / SINE_EIGS_2D_np, 0.0)
                    u = S_2D_np.T @ c_np
                    J_rolling = float(u[SENSOR_FLAT])
                else:
                    f = np.asarray(source_grid_2d(tx_j, ty_j))
                    b_np = W_HAAR_2D_np @ f
                    A_arr = A_HAAR_2D_np
                    A_eff = np.where(mask_np[:, None],
                                     mask_np[None, :] * A_arr,
                                     np.eye(N))
                    b_eff = mask_np * b_np
                    c = np.linalg.solve(A_eff, b_eff)
                    u = W_HAAR_2D_np.T @ c
                    J_rolling = float(u[SENSOR_FLAT])
            errs["rolling"].append(relerr(J_rolling, J_ref))

            # cold-start (only at t=0): coarse-then-fine
            if t == 0:
                k_coarse = N // 4
                # coarse mask
                if basis == "sine":
                    f = np.asarray(source_grid_2d(tx_j, ty_j))
                    b_np = np.asarray(S_2D_np @ f)
                    mag_b = np.abs(b_np)
                    thr_c = np.sort(mag_b)[-k_coarse]
                    mask_c = mag_b >= thr_c
                    c_coarse = np.where(mask_c, b_np / SINE_EIGS_2D_np, 0.0)
                    mag_cc = np.abs(c_coarse)
                    thr = np.sort(mag_cc)[-k]
                    mask = mag_cc >= thr
                    c_np = np.where(mask, b_np / SINE_EIGS_2D_np, 0.0)
                    u = S_2D_np.T @ c_np
                    J_cs = float(u[SENSOR_FLAT])
                else:
                    f = np.asarray(source_grid_2d(tx_j, ty_j))
                    b_np = W_HAAR_2D_np @ f
                    mag_b = np.abs(b_np)
                    thr_c = np.sort(mag_b)[-k_coarse]
                    mask_c = mag_b >= thr_c
                    A_eff = np.where(mask_c[:, None],
                                     mask_c[None, :] * A_HAAR_2D_np,
                                     np.eye(N))
                    b_eff_c = mask_c * b_np
                    c_coarse = np.linalg.solve(A_eff, b_eff_c)
                    mag_cc = np.abs(c_coarse)
                    thr = np.sort(mag_cc)[-k]
                    mask = mag_cc >= thr
                    A_eff = np.where(mask[:, None],
                                     mask[None, :] * A_HAAR_2D_np,
                                     np.eye(N))
                    b_eff = mask * b_np
                    c = np.linalg.solve(A_eff, b_eff)
                    u = W_HAAR_2D_np.T @ c
                    J_cs = float(u[SENSOR_FLAT])
            else:
                J_cs = float("nan")
            errs["cold-coarse"].append(relerr(J_cs, J_ref)
                                       if not np.isnan(J_cs) else float("nan"))

            # oracle
            J_o = float(J_frozen_fn(tx_j, ty_j, k, "c"))
            errs["oracle"].append(relerr(J_o, J_ref))

            # update c_prev: use the oracle c as the previous c
            f = np.asarray(source_grid_2d(tx_j, ty_j))
            if basis == "sine":
                b_np = np.asarray(S_2D_np @ f)
                c_prev_np = b_np / SINE_EIGS_2D_np
            else:
                b_np = W_HAAR_2D_np @ f
                c_prev_np = np.linalg.solve(A_HAAR_2D_np, b_np)

        for strat in errs:
            ar = np.array(errs[strat], dtype=float)
            ar = ar[~np.isnan(ar)]
            n_ws = int(np.sum(ar > 1.0))   # 'wrong sign / J_err > 1' count
            print(f"    {strat:>14}: mean={ar.mean():.3e}, "
                  f"max={ar.max():.3e}, J_err>1 count: {n_ws}")
        print()


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
    part_d()
