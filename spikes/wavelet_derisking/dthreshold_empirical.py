"""Investigation 6 -- D_threshold = 5 empirical validation.

SPIKE CODE. Investigative. (Only affects NON-wavelet TopKAdaptiveNode: the
wavelet basis is trap-immune, Gate 2.)

The base class monitors traps at runtime when D > 5, from the round-5 analytic
estimate E[trap encounters] ≈ 0.2·D per trajectory. Never empirically checked.

Model: a D-dim separable generalisation of the 1D sine-Poisson problem,
J(θ) = Σ_i J_1d(θ_i), with the sensor at x=1/3 (off-centre) and top-|b|
selection (the trap-prone non-local setup from Gate 2). The per-axis full and
frozen gradients are the *exact* 1D sine-Poisson gradients (precomputed on a
grid). The D-dim blindness ratio is ||g_frozen||/||g_full||. A trap encounter
= ≥5 consecutive descent steps with ratio < 0.7.

This separable model is faithful to the trap structure (Fix(G) = union of
hyperplanes θ_i=0.5; each axis goes blind near 0.5 exactly as in 1D) and is
tractable to D=20 (the true coupled D-dim mode sum is not).
"""

from __future__ import annotations

import numpy as np

import g1_wrong_sign as g1

g1.SENSOR_X = 1.0 / 3.0  # off-centre -> trap-prone (Gate 2 mechanism)


def build_1d_gradients(n_levels=6, Kfrac=16):
    """Precompute exact 1D full and frozen (top-|b|) gradients on a θ grid."""
    Wn, levels, x, A, h, sidx = g1.build_basis("sine", n_levels, 4)
    N = A.shape[0]; K = max(4, N // Kfrac); srow = Wn[sidx]

    def J_full(th):
        b = g1.load_vector(Wn, x, h, th)
        return srow @ np.linalg.solve(A, b)

    def J_frozen(th, S):
        b = g1.load_vector(Wn, x, h, th)
        c = np.zeros(N); c[S] = np.linalg.solve(A[np.ix_(S, S)], b[S])
        return srow @ c

    grid = np.linspace(0.1, 0.9, 161)
    gF = np.zeros_like(grid); gZ = np.zeros_like(grid)
    eps = 1e-5
    for i, th in enumerate(grid):
        gF[i] = (J_full(th + eps) - J_full(th - eps)) / (2 * eps)
        b = g1.load_vector(Wn, x, h, th)
        S = np.argsort(-np.abs(b))[:K]          # top-|b| frozen at θ
        gZ[i] = (J_frozen(th + eps, S) - J_frozen(th - eps, S)) / (2 * eps)
    return grid, gF, gZ


def interp_fn(grid, vals):
    return lambda th: np.interp(np.clip(th, grid[0], grid[-1]), grid, vals)


def run_trajectories(gF_fn, gZ_fn, D, n_traj=20, lr=0.04, steps=100, seed=0):
    rng = np.random.default_rng(seed + D)
    encounters = []; wasted_steps = []; peraxis = []
    for _ in range(n_traj):
        theta = rng.uniform(0.1, 0.9, D)
        blind_run = 0; n_enc = 0; n_wasted = 0; axis_blind_steps = 0
        in_trap = False
        for _ in range(steps):
            gfull = gF_fn(theta)
            gfroz = gZ_fn(theta)
            ratio = np.linalg.norm(gfroz) / (np.linalg.norm(gfull) + 1e-30)
            # per-axis blind count: axes whose individual frozen/full ratio < 0.7
            ax_ratio = np.abs(gfroz) / (np.abs(gfull) + 1e-30)
            axis_blind_steps += int(np.sum(ax_ratio < 0.7))
            if ratio < 0.7:
                blind_run += 1
                if blind_run >= 5 and not in_trap:
                    n_enc += 1; in_trap = True
                if in_trap:
                    n_wasted += 1
            else:
                blind_run = 0; in_trap = False
            theta = np.clip(theta - lr * gfroz, 0.1, 0.9)  # node uses frozen grad
        encounters.append(n_enc); wasted_steps.append(n_wasted)
        peraxis.append(axis_blind_steps / steps)  # mean #blind-axes per step
    return np.mean(encounters), np.mean(wasted_steps), np.mean(peraxis)


def part_A(gF_fn, gZ_fn):
    print("=" * 78)
    print("PART A -- trap encounters vs D (20 trajectories, lr=0.04, 100 steps)")
    print("=" * 78)
    print(f"  {'D':>4} {'global enc':>11} {'0.2·D':>7} {'enc/0.2D':>9} "
          f"{'per-axis blind/step':>20}")
    data = {}
    for D in (2, 3, 5, 7, 10, 15, 20):
        enc, wasted, perax = run_trajectories(gF_fn, gZ_fn, D)
        data[D] = (enc, wasted)
        pred = 0.2 * D
        print(f"  {D:>4} {enc:>11.2f} {pred:>7.1f} {enc/(pred+1e-9):>9.2f} "
              f"{perax:>20.2f}")
    return data


def part_B(data):
    print("=" * 78)
    print("PART B -- cost of monitoring vs missed traps (round-6 cost model)")
    print("=" * 78)
    print("  detected trap = 3 full solves; undetected trap = ~50 wasted frozen steps")
    print(f"  {'D':>4} {'E[enc]':>7} {'monitor cost':>13} {'no-monitor cost':>16} "
          f"{'favor':>10}")
    for D, (enc, wasted) in data.items():
        monitor = enc * 3            # detect+break each trap: 3 full solves
        no_monitor = enc * 50        # ~50 wasted frozen steps per undetected trap
        favor = "monitor" if monitor < no_monitor else "off"
        print(f"  {D:>4} {enc:>7.2f} {monitor:>13.1f} {no_monitor:>16.1f} {favor:>10}")


if __name__ == "__main__":
    grid, gF, gZ = build_1d_gradients()
    gF_fn = interp_fn(grid, gF); gZ_fn = interp_fn(grid, gZ)
    data = part_A(gF_fn, gZ_fn)
    part_B(data)
