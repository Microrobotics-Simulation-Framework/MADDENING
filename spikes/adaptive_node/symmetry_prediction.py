"""Round-5 Investigation 3: 2D symmetry-trap prediction and angle accuracy.

Tests two hypotheses about the trap mechanism:
  H1: trap is determined by the SOURCE+OPERATOR symmetry of the
      problem.  Sensor location doesn't change it.
  H2: breaking the DOMAIN x-symmetry (asymmetric rectangle)
      shifts/eliminates the x-trap axis.

Part A tests H1 (sensor shift) and H2 (asymmetric domain).
Part B measures 2D angle accuracy at the 7x7 grid of round-4.
Part C is the analytical prevalence-scaling section (memo only).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import os, sys
sys.path.insert(0, os.path.dirname(__file__))


# ---- Symmetric setup (matches locality_2d.py) ----
def build_problem_2d(N_1D=32, sensor_xy=(0.7, 0.6),
                     domain_xy=(1.0, 1.0), sigma=0.1):
    """Build everything for a 2D Poisson problem on a possibly
    asymmetric rectangular domain."""
    sensor_x, sensor_y = sensor_xy
    Lx, Ly = domain_xy
    N = N_1D * N_1D
    dx = Lx / (N_1D + 1)
    dy = Ly / (N_1D + 1)
    x_grid = (np.arange(1, N_1D + 1) * dx)
    y_grid = (np.arange(1, N_1D + 1) * dy)

    # 1D Laplacian in each direction (with different dx, dy)
    A_LAP_X = (2.0 / dx ** 2) * np.eye(N_1D) - (1.0 / dx ** 2) * (
        np.eye(N_1D, k=1) + np.eye(N_1D, k=-1)
    )
    A_LAP_Y = (2.0 / dy ** 2) * np.eye(N_1D) - (1.0 / dy ** 2) * (
        np.eye(N_1D, k=1) + np.eye(N_1D, k=-1)
    )
    A_PHYS = (np.kron(A_LAP_X, np.eye(N_1D))
              + np.kron(np.eye(N_1D), A_LAP_Y) + np.eye(N))

    # DST-I for x and y (eigenfunctions of FD Dirichlet Laplacian
    # with appropriate scaling per dimension)
    ks = np.arange(1, N_1D + 1)
    S_X = np.sqrt(2.0 / (N_1D + 1)) * np.sin(
        np.pi * np.outer(ks, ks) / (N_1D + 1)
    )
    S_Y = S_X  # same shape (DST-I depends on N_1D, not dx)
    EIGS_X = 2.0 * (1.0 - np.cos(np.pi * ks / (N_1D + 1))) / dx ** 2
    EIGS_Y = 2.0 * (1.0 - np.cos(np.pi * ks / (N_1D + 1))) / dy ** 2

    S_2D = np.kron(S_X, S_Y)
    SINE_EIGS_2D = (EIGS_X[:, None] + EIGS_Y[None, :]).flatten() + 1.0

    SENSOR_IX = int(round(sensor_x / dx)) - 1
    SENSOR_IY = int(round(sensor_y / dy)) - 1
    SENSOR_FLAT = SENSOR_IX * N_1D + SENSOR_IY

    state = dict(
        N_1D=N_1D, N=N, dx=dx, dy=dy,
        x_grid=x_grid, y_grid=y_grid,
        S_2D=jnp.asarray(S_2D),
        SINE_EIGS_2D=jnp.asarray(SINE_EIGS_2D),
        SENSOR_FLAT=SENSOR_FLAT,
        sigma=sigma,
    )
    return state


def source_grid_2d(state, theta_x, theta_y):
    X = jnp.asarray(state['x_grid'])[:, None]
    Y = jnp.asarray(state['y_grid'])[None, :]
    f = jnp.exp(-((X - theta_x) ** 2 + (Y - theta_y) ** 2)
                / state['sigma'] ** 2)
    return f.reshape(-1)


def J_sine_full(state, theta_x, theta_y):
    f = source_grid_2d(state, theta_x, theta_y)
    b = state['S_2D'] @ f
    c = b / state['SINE_EIGS_2D']
    u = state['S_2D'].T @ c
    return u[state['SENSOR_FLAT']]


def J_sine_frozen(state, theta_x, theta_y, k_active):
    f = source_grid_2d(state, theta_x, theta_y)
    b = state['S_2D'] @ f
    mag = jnp.abs(b)
    thr = jnp.sort(mag)[-k_active]
    mask = jax.lax.stop_gradient(mag >= thr)
    c = jnp.where(mask, b / state['SINE_EIGS_2D'], 0.0)
    u = state['S_2D'].T @ c
    return u[state['SENSOR_FLAT']]


def blindness_grid_2d(state, k, n_grid=7,
                      grid_lo=0.15, grid_hi=0.85):
    """Compute blindness_ratio over a grid of (tx, ty)."""
    grid = np.linspace(grid_lo, grid_hi, n_grid)
    out = np.zeros((n_grid, n_grid))
    grad_full_x = jax.grad(lambda tx, ty: J_sine_full(state, tx, ty),
                           argnums=0)
    grad_full_y = jax.grad(lambda tx, ty: J_sine_full(state, tx, ty),
                           argnums=1)
    grad_fr_x = jax.grad(lambda tx, ty: J_sine_frozen(state, tx, ty, k),
                         argnums=0)
    grad_fr_y = jax.grad(lambda tx, ty: J_sine_frozen(state, tx, ty, k),
                         argnums=1)
    full_grads = []
    fr_grads = []
    for i, tx in enumerate(grid):
        for j, ty in enumerate(grid):
            tx_j, ty_j = jnp.asarray(tx), jnp.asarray(ty)
            gx_f = float(grad_full_x(tx_j, ty_j))
            gy_f = float(grad_full_y(tx_j, ty_j))
            gx_fr = float(grad_fr_x(tx_j, ty_j))
            gy_fr = float(grad_fr_y(tx_j, ty_j))
            full_grads.append((tx, ty, gx_f, gy_f))
            fr_grads.append((tx, ty, gx_fr, gy_fr))
            n_full = np.hypot(gx_f, gy_f)
            n_fr = np.hypot(gx_fr, gy_fr)
            out[i, j] = n_fr / (n_full + 1e-30)
    return grid, out, full_grads, fr_grads


# ---- Part A: H1 (sensor shift) + H2 (asymmetric domain) ----
def part_a():
    print("# Part A -- symmetry-group prediction of trap locations")
    print()
    print("## Baseline (round-4): sensor=(0.7, 0.6), domain=(1.0, 1.0)")
    state_base = build_problem_2d(sensor_xy=(0.7, 0.6),
                                  domain_xy=(1.0, 1.0))
    grid_b, blind_b, _, _ = blindness_grid_2d(state_base, k=state_base['N'] // 8)
    _print_blind(grid_b, blind_b)

    print("\n## H1: shift sensor to (0.3, 0.6) [mirror in x]")
    state_h1 = build_problem_2d(sensor_xy=(0.3, 0.6),
                                domain_xy=(1.0, 1.0))
    grid_h1, blind_h1, _, _ = blindness_grid_2d(state_h1, k=state_h1['N'] // 8)
    _print_blind(grid_h1, blind_h1)
    print(f"\n  Column tx=0.5 mean (baseline):      "
          f"{blind_b[:, grid_b.tolist().index(0.5) if 0.5 in grid_b else 3].mean():.3f}")
    # 7x7 grid has tx=0.5 at index 3
    print(f"  Column tx=0.5 mean (H1 sensor=0.3): {blind_h1[:, 3].mean():.3f}")
    print(f"  Row    ty=0.5 mean (baseline):      {blind_b[3, :].mean():.3f}")
    print(f"  Row    ty=0.5 mean (H1 sensor=0.3): {blind_h1[3, :].mean():.3f}")
    print()
    print("  -> Prediction was the trap structure should be the SAME under")
    print("     sensor reflection because the SELECTION criterion (top-|b|)")
    print("     depends only on source+operator symmetry, not sensor.")

    print("\n## H2: domain (1.2, 1.0) [breaks x-symmetry]")
    state_h2 = build_problem_2d(sensor_xy=(0.7, 0.6),
                                domain_xy=(1.2, 1.0))
    # New x-symmetry axis would be at x=0.6 if the source is also placed
    # symmetrically; the parameter sweep needs to be rescaled.
    # Sweep in [0.15, 0.85] * 1.2 for x:
    grid_h2 = np.linspace(0.15, 0.85, 7) * 1.2   # rescale to new domain
    grid_y = np.linspace(0.15, 0.85, 7)
    out = np.zeros((7, 7))
    grad_full_x = jax.grad(
        lambda tx, ty: J_sine_full(state_h2, tx, ty), argnums=0
    )
    grad_full_y = jax.grad(
        lambda tx, ty: J_sine_full(state_h2, tx, ty), argnums=1
    )
    grad_fr_x = jax.grad(
        lambda tx, ty: J_sine_frozen(state_h2, tx, ty, state_h2['N'] // 8),
        argnums=0
    )
    grad_fr_y = jax.grad(
        lambda tx, ty: J_sine_frozen(state_h2, tx, ty, state_h2['N'] // 8),
        argnums=1
    )
    for i, tx in enumerate(grid_h2):
        for j, ty in enumerate(grid_y):
            tx_j, ty_j = jnp.asarray(tx), jnp.asarray(ty)
            gx_f = float(grad_full_x(tx_j, ty_j))
            gy_f = float(grad_full_y(tx_j, ty_j))
            gx_fr = float(grad_fr_x(tx_j, ty_j))
            gy_fr = float(grad_fr_y(tx_j, ty_j))
            out[i, j] = np.hypot(gx_fr, gy_fr) / (
                np.hypot(gx_f, gy_f) + 1e-30
            )
    print(f"  Domain [0, 1.2] x [0, 1.0]")
    print(f"  tx sweep covers [0.18, 1.02] (rescaled), ty in [0.15, 0.85]")
    print(f"  New x-symmetry axis: tx = 0.6 (= 1.2/2)")
    print(f"  Look for trap at tx near 0.6 instead of 0.5")
    print()
    print(f"     tx \\ ty", end="")
    for ty in grid_y:
        print(f"  {ty:>7.3f}", end="")
    print()
    for i, tx in enumerate(grid_h2):
        print(f"  {tx:>9.4f}: ", end="")
        for j in range(len(grid_y)):
            print(f"  {out[i, j]:>7.4f}", end="")
        print()
    # Find column closest to tx=0.6
    idx_60 = int(np.argmin(np.abs(grid_h2 - 0.6)))
    print(f"\n  Column nearest tx=0.6 (idx={idx_60}, tx={grid_h2[idx_60]:.4f}) "
          f"mean: {out[idx_60, :].mean():.3f}")
    print(f"  Row nearest ty=0.5 (idx 3, ty=0.5) mean: {out[3, :].mean():.3f}")


def _print_blind(grid, blind):
    print(f"   {'tx \\ ty':>10}", end="")
    for ty in grid:
        print(f"  {ty:>7.3f}", end="")
    print()
    for i, tx in enumerate(grid):
        print(f"  {tx:>9.4f}: ", end="")
        for j in range(len(grid)):
            print(f"  {blind[i, j]:>7.4f}", end="")
        print()


# ---- Part B: 2D angle accuracy ----
def part_b():
    print("\n\n# Part B -- 2D angle accuracy at the 49-point grid")
    print()
    state = build_problem_2d(sensor_xy=(0.7, 0.6))
    grid, blind, full_g, fr_g = blindness_grid_2d(
        state, k=state['N'] // 8
    )
    angles = []
    for (tx, ty, gxf, gyf), (tx2, ty2, gxr, gyr) in zip(full_g, fr_g):
        nf = np.hypot(gxf, gyf)
        nr = np.hypot(gxr, gyr)
        if nf < 1e-30 or nr < 1e-30:
            angle = float("nan")
        else:
            cos_t = (gxf * gxr + gyf * gyr) / (nf * nr)
            cos_t = max(-1.0, min(1.0, float(cos_t)))
            angle = np.degrees(np.arccos(cos_t))
        ratio = nr / (nf + 1e-30)
        angles.append((tx, ty, ratio, angle))

    print(f"  {'tx':>6} {'ty':>6} {'ratio':>9} {'angle_deg':>10}  bucket")
    bucket_summary = {"ratio<0.3": [], "0.3-0.7": [], "0.7-1.0": [],
                      "1.0+": []}
    for tx, ty, ratio, angle in angles:
        if np.isnan(angle):
            bucket = "blind"
        elif ratio < 0.3:
            bucket = "ratio<0.3"
        elif ratio < 0.7:
            bucket = "0.3-0.7"
        elif ratio < 1.0:
            bucket = "0.7-1.0"
        else:
            bucket = "1.0+"
        if bucket in bucket_summary:
            bucket_summary[bucket].append(angle)
        if abs(ratio - 1.0) > 0.5 or angle > 10:
            print(f"  {tx:>6.3f} {ty:>6.3f} {ratio:>9.4f} {angle:>10.3f}  "
                  f"{bucket}")
    print()
    for b in ["ratio<0.3", "0.3-0.7", "0.7-1.0", "1.0+"]:
        if bucket_summary[b]:
            arr = np.array(bucket_summary[b])
            arr = arr[~np.isnan(arr)]
            if len(arr):
                print(f"  {b:>10}: n={len(arr)}, mean angle="
                      f"{arr.mean():.3f}deg, max={arr.max():.3f}deg")
            else:
                print(f"  {b:>10}: n=0")
        else:
            print(f"  {b:>10}: n=0")


if __name__ == "__main__":
    part_a()
    part_b()
