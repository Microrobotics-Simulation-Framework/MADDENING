"""Round-3 Investigation 1: symmetry-trap characterisation and
three concrete mitigations.

Setup: same 1D Poisson + sine basis + Gaussian source as Q2/Q3.
N_BASIS=256, sigma=0.04, K=16 active set, lr=0.04, 30 GD steps.
MINIMIZING J = u(x_sensor).

Parts:
  A. Prevalence -- 20 starting points across theta_0 in [0.1, 0.9].
     Tabulate final theta, J, and "moved" flag.  Identify all
     trap points and characterise them.
  B. Mitigation 1: history-based active set (sticky).
     new_mask = top-k(|b|) UNION prev_mask, capped at 2k.
  C. Mitigation 2: blindness ratio = |grad J_frozen| / |grad J_full|.
     Validate at trap and well-behaved points; sweep across theta.
  D. Mitigation 3: augmented criterion |b| + lambda * |db/dtheta|.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# Toy setup matches q3_chattering.py
N_BASIS = 256
N_GRID = 512
SIGMA = 0.04
SENSOR_IDX = N_GRID // 3
K_ACTIVE = 16
N_STEPS = 30
LR = 0.04

ks = jnp.arange(1, N_BASIS + 1)
LAMBDAS = (ks * jnp.pi) ** 2 + 1.0
x_grid = jnp.linspace(0.0, 1.0, N_GRID)
dx = float(x_grid[1] - x_grid[0])
PHI = jnp.sin(jnp.pi * jnp.outer(x_grid, ks))


def rhs_coeffs(theta):
    f = jnp.exp(-((x_grid - theta) / SIGMA) ** 2)
    return 2.0 * dx * (PHI.T @ f)


def J_with_mask(theta, mask):
    b = rhs_coeffs(theta)
    m = jax.lax.stop_gradient(mask)
    c = jnp.where(m, b / LAMBDAS, 0.0)
    u = PHI @ c
    return u[SENSOR_IDX]


def J_full(theta):
    """Full-basis J -- all modes active."""
    b = rhs_coeffs(theta)
    c = b / LAMBDAS
    u = PHI @ c
    return u[SENSOR_IDX]


grad_full = jax.jit(jax.grad(J_full))
grad_with_mask = jax.jit(jax.grad(J_with_mask, argnums=0))
J_jit = jax.jit(J_with_mask)
db_dtheta = jax.jit(jax.jacrev(rhs_coeffs))


def topk_mask(mag, k):
    sorted_mag = np.sort(mag)
    return mag >= sorted_mag[-k]


# ----- Part A: trap prevalence sweep -----
def run_descent(theta0, lr=LR, n_steps=N_STEPS, build_mask=None):
    """Run gradient descent.  build_mask(mag, prev_mask, prev_b) -> mask."""
    if build_mask is None:
        build_mask = lambda mag, prev, _b: topk_mask(mag, K_ACTIVE)
    theta = float(theta0)
    prev_mask = None
    history = [theta]
    for _ in range(n_steps):
        b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
        mag = np.abs(b)
        mask = build_mask(mag, prev_mask, b)
        g = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
        theta = theta - lr * g
        history.append(theta)
        prev_mask = mask
    return np.array(history)


def part_a():
    print("# Part A -- trap prevalence sweep")
    print(f"# 20 starting points across theta_0 in [0.1, 0.9], "
          f"K={K_ACTIVE}, lr={LR}, {N_STEPS} steps")
    print()
    theta0s = np.linspace(0.10, 0.90, 20)
    rows = []
    for t0 in theta0s:
        traj = run_descent(t0)
        moved = abs(traj[-1] - traj[0]) > 1e-4
        # check blindness at start
        b = np.asarray(rhs_coeffs(jnp.asarray(t0)))
        mask = topk_mask(np.abs(b), K_ACTIVE)
        g_frozen = float(grad_with_mask(jnp.asarray(t0), jnp.asarray(mask)))
        g_full_val = float(grad_full(jnp.asarray(t0)))
        blind = abs(g_frozen) / (abs(g_full_val) + 1e-30)
        rows.append((t0, traj[-1], moved, g_frozen, g_full_val, blind))

    print(f"{'theta_0':>8} {'theta_final':>12} {'moved':>6} "
          f"{'g_frozen':>12} {'g_full':>12} {'blind_ratio':>11}")
    n_trapped = 0
    for r in rows:
        t0, tf, m, gf, gF, br = r
        flag = "" if m else "  <-- TRAP"
        if not m:
            n_trapped += 1
        print(f"{t0:>8.4f} {tf:>12.5f} {str(m):>6} "
              f"{gf:>+12.4e} {gF:>+12.4e} {br:>11.4e}{flag}")
    print(f"\n  trapped runs: {n_trapped} / 20")

    # Characterise trap points by symmetry
    print()
    print("  Trap-point characterisation:")
    for r in rows:
        t0, tf, m, gf, gF, br = r
        if not m:
            print(f"    theta_0 = {t0:.4f}: "
                  f"|g_full| = {abs(gF):.2e}  "
                  f"(intrinsic critical point? -- "
                  f"{'YES' if abs(gF) < 1e-6 else 'NO, frozen-set blindness'})")

    return rows


# ----- Part B: history-based "sticky" active set -----
def sticky_mask(mag, prev_mask, b, k=K_ACTIVE, max_size=None):
    if max_size is None:
        max_size = 2 * k
    cur = topk_mask(mag, k)
    if prev_mask is None:
        return cur
    union = cur | prev_mask
    if int(union.sum()) <= max_size:
        return union
    # cap to max_size: keep cur entirely + add top-(max_size - k) from prev\cur by |b|
    extras = prev_mask & ~cur
    extras_idx = np.where(extras)[0]
    extras_mag = mag[extras_idx]
    keep_n = max(0, max_size - int(cur.sum()))
    keep_idx = extras_idx[np.argsort(-extras_mag)[:keep_n]]
    out = cur.copy()
    out[keep_idx] = True
    return out


def part_b():
    print("\n# Part B -- mitigation 1: history-based 'sticky' active set")
    print(f"# Sticky rule: new_mask = top-K(|b|) UNION prev_mask, capped at 2K")
    print(f"# Test at trap points found in Part A + a control")
    print()
    test_thetas = [0.5, 0.4, 0.7]   # Part A will report the actual list; 0.5 is the trap
    for t0 in test_thetas:
        traj = run_descent(t0, build_mask=sticky_mask)
        moved = abs(traj[-1] - traj[0]) > 1e-4
        J0 = float(J_full(jnp.asarray(t0)))
        Jf = float(J_full(jnp.asarray(traj[-1])))
        # measure churn
        prev_mask = None
        churns = []
        sizes = []
        for theta in traj[:-1]:
            b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
            mag = np.abs(b)
            m = sticky_mask(mag, prev_mask, b)
            if prev_mask is not None:
                churns.append(int(np.sum(prev_mask ^ m)))
            sizes.append(int(np.sum(m)))
            prev_mask = m
        print(f"  theta_0 = {t0:.4f}: theta_f = {traj[-1]:.5f} "
              f"(moved: {moved})")
        print(f"    J0 = {J0:.5e}, Jf = {Jf:.5e}, deltaJ = {J0-Jf:+.3e}")
        print(f"    mean mask size: {np.mean(sizes):.1f}, "
              f"max: {max(sizes)}, total churn: {sum(churns)}")
        print(f"    theta trajectory: "
              f"{['%.4f' % x for x in traj[::5]]}")


# ----- Part C: blindness diagnostic -----
def blindness_ratio(theta, k=K_ACTIVE):
    b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
    mask = topk_mask(np.abs(b), k)
    g_frozen = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
    g_full_val = float(grad_full(jnp.asarray(theta)))
    return abs(g_frozen) / (abs(g_full_val) + 1e-30), g_frozen, g_full_val


def part_c():
    print("\n# Part C -- mitigation 2: blindness diagnostic")
    print("# blindness_ratio = |grad J_frozen| / |grad J_full|")
    print()
    # Validate at known points
    test_thetas = {
        "trap (0.5)": 0.5,
        "well-behaved (0.42)": 0.42,
        "well-behaved (0.3)": 0.3,
    }
    print("  Validation at specific points:")
    for label, t in test_thetas.items():
        r, gf, gF = blindness_ratio(t)
        print(f"    theta = {t:.3f} ({label:>22s}): "
              f"ratio = {r:.4f}   |g_frozen| = {abs(gf):.3e}   "
              f"|g_full| = {abs(gF):.3e}")

    # Sweep across [0.1, 0.9]
    print()
    print("  Sweep across [0.1, 0.9]:")
    print(f"  {'theta':>7} {'ratio':>9} {'|g_full|':>11} {'flag':>10}")
    thetas = np.linspace(0.1, 0.9, 41)
    n_low = 0
    for t in thetas:
        r, _, gF = blindness_ratio(t)
        flag = ""
        if r < 0.5:
            flag = "<-- BLIND"
            n_low += 1
        elif r < 0.9:
            flag = "<-- partial"
        print(f"  {t:>7.4f} {r:>9.4f} {abs(gF):>11.3e}  {flag}")
    print(f"\n  Points with blindness_ratio < 0.5: {n_low} / 41")

    # Cost
    print()
    print("  Computational cost:")
    print("    1 diagnostic eval = 2 solves (frozen + full) + 2 grads.")
    print(f"    Full solve = O(N_GRID) = O({N_GRID}) FLOPs in this toy.")
    print("    Diagnostic ~2x cost of one forward+adjoint step.")


# ----- Part D: augmented criterion -----
def augmented_mask(mag, db_mag, k, lam):
    score = mag + lam * db_mag
    sorted_score = np.sort(score)
    return score >= sorted_score[-k]


def run_augmented(theta0, lam, n_steps=N_STEPS, lr=LR):
    theta = float(theta0)
    history = [theta]
    prev_mask = None
    churns = []
    for _ in range(n_steps):
        b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
        db = np.asarray(db_dtheta(jnp.asarray(theta)))
        mask = augmented_mask(np.abs(b), np.abs(db), K_ACTIVE, lam)
        if prev_mask is not None:
            churns.append(int(np.sum(prev_mask ^ mask)))
        g = float(grad_with_mask(jnp.asarray(theta), jnp.asarray(mask)))
        theta = theta - lr * g
        history.append(theta)
        prev_mask = mask
    return np.array(history), churns


def part_d():
    print("\n# Part D -- mitigation 3: augmented criterion |b| + lambda|db/dtheta|")
    print()
    for lam in [0.0, 0.1, 0.5, 1.0]:
        print(f"\n## lambda = {lam}")
        for t0 in [0.5, 0.4, 0.7]:
            traj, churns = run_augmented(t0, lam)
            moved = abs(traj[-1] - traj[0]) > 1e-4
            J0 = float(J_full(jnp.asarray(t0)))
            Jf = float(J_full(jnp.asarray(traj[-1])))
            tot_churn = sum(churns)
            print(f"  theta_0={t0:.4f}: theta_f={traj[-1]:.5f} "
                  f"(moved: {moved}) "
                  f"deltaJ={J0-Jf:+.3e}  total_churn={tot_churn}")


if __name__ == "__main__":
    rows_A = part_a()
    part_b()
    part_c()
    part_d()
