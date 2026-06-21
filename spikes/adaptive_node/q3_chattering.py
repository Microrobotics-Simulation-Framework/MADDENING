"""Q3 (rerun): Does hysteresis prevent chattering, and what does
condition E (trust-region re-evaluation) buy?

Same 1D Poisson + sine basis + Gaussian source as q2.  J(theta) =
u(x_sensor).  We MINIMIZE J via gradient descent (theta drifts away
from the sensor).  Starting theta=0.5, x_sensor ~ 0.333, so descent
pushes theta toward the right boundary.  Across this range many
top-K boundary swaps occur.

K_baseline = 16.

Conditions:
  A — pure top-k (no hysteresis)
  B — eps_remove = 0.9 * eps_add
  C — eps_remove = 0.5 * eps_add  (proposed default)
  D — eps_remove = 0.1 * eps_add
  E — trust-region: when any element is within delta=0.1*eps_add of
      the threshold, compute adjoint under both current mask and an
      alternative mask (flip the near-threshold element); take a
      trial step under each and choose whichever produces lower J.

Per step we record: theta, J, gradient magnitude, mask size, and
symmetric-difference (churn) from the previous step.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# -- toy (matches q2) --
N_BASIS = 256
N_GRID = 512
SIGMA = 0.04
SENSOR_IDX = N_GRID // 3
K_ACTIVE = 16
N_STEPS = 30
LR = 0.04
THETA0 = 0.40
# NOTE: theta=0.5 is a SYMMETRY trap with this setup.  At exactly 0.5 the
# Gaussian source is symmetric on [0,1]; b_k = 0 for k even and
# db_k/dtheta = 0 for k odd; the top-|b| mask selects only odd k, so the
# frozen-set gradient is identically 0 and no descent occurs.  This is a
# stark illustration of Q2's selection-criterion failure mode.  We start
# at theta_0=0.40 to avoid the trap.

ks = jnp.arange(1, N_BASIS + 1)
LAMBDAS = (ks * jnp.pi) ** 2 + 1.0
x_grid = jnp.linspace(0.0, 1.0, N_GRID)
dx = float(x_grid[1] - x_grid[0])
PHI = jnp.sin(jnp.pi * jnp.outer(x_grid, ks))


def rhs_coeffs(theta: jax.Array) -> jax.Array:
    f = jnp.exp(-((x_grid - theta) / SIGMA) ** 2)
    return 2.0 * dx * (PHI.T @ f)


def J_with_mask(theta, mask):
    b = rhs_coeffs(theta)
    m = jax.lax.stop_gradient(mask)
    c = jnp.where(m, b / LAMBDAS, 0.0)
    u = PHI @ c
    return u[SENSOR_IDX]


J_jit = jax.jit(J_with_mask)
grad_J = jax.jit(jax.grad(J_with_mask, argnums=0))


def topk_mask(mag, k):
    sorted_mag = np.sort(mag)
    eps = sorted_mag[-k]
    return mag >= eps, float(eps)


def hysteresis_mask(mag, k, prev_mask, ratio):
    sorted_mag = np.sort(mag)
    eps_add = sorted_mag[-k]
    eps_remove = ratio * eps_add
    if prev_mask is None:
        return mag >= eps_add, float(eps_add)
    keep_prev = prev_mask & (mag >= eps_remove)
    add_new = (~prev_mask) & (mag >= eps_add)
    return keep_prev | add_new, float(eps_add)


def run_condition(label, build_mask, theta0=THETA0, lr=LR, n_steps=N_STEPS,
                  trust_region=False):
    theta = float(theta0)
    prev_mask = None
    history = []
    extra_solves = 0
    for step in range(n_steps):
        b = np.asarray(rhs_coeffs(jnp.asarray(theta)))
        mag = np.abs(b)
        mask, eps_add = build_mask(mag, prev_mask)

        if trust_region:
            delta = 0.1 * eps_add
            in_band = np.abs(mag - eps_add) < delta
            chosen_mask = mask.copy()
            if np.any(in_band):
                # pick the element whose mag is closest to eps_add
                i_near = int(np.argmin(np.abs(mag - eps_add)))
                alt_mask = mask.copy()
                alt_mask[i_near] = not alt_mask[i_near]
                # gradients under both
                g_cur = float(grad_J(jnp.asarray(theta), jnp.asarray(mask)))
                g_alt = float(grad_J(jnp.asarray(theta), jnp.asarray(alt_mask)))
                t_cur = theta - lr * g_cur
                t_alt = theta - lr * g_alt
                J_cur = float(J_jit(jnp.asarray(t_cur), jnp.asarray(mask)))
                J_alt = float(J_jit(jnp.asarray(t_alt), jnp.asarray(alt_mask)))
                extra_solves += 4  # 2 grads + 2 trial Js
                if J_alt < J_cur:
                    chosen_mask = alt_mask
            mask = chosen_mask

        J_val = float(J_jit(jnp.asarray(theta), jnp.asarray(mask)))
        g = float(grad_J(jnp.asarray(theta), jnp.asarray(mask)))
        churn = 0 if prev_mask is None else int(np.sum(prev_mask ^ mask))
        history.append(dict(
            step=step, theta=theta, J=J_val, g=g,
            mask_size=int(np.sum(mask)), churn=churn,
        ))
        theta = theta - lr * g
        prev_mask = mask
    history.append(dict(step=n_steps, theta=theta, J=float('nan'),
                        g=float('nan'), mask_size=-1, churn=-1))
    return history, extra_solves


def summarize(label, history, extra_solves=0):
    h = history[:-1]   # drop the final "theta only" entry
    churns = [r['churn'] for r in h if r['churn'] >= 0][1:]   # skip step 0
    sizes = [r['mask_size'] for r in h]
    Js = [r['J'] for r in h]
    thetas = [r['theta'] for r in h]
    print(f"\n## Condition {label}")
    print(f"   final theta = {history[-1]['theta']:.5f}   "
          f"J0 = {Js[0]:.5e}   Jfinal = {Js[-1]:.5e}   "
          f"(deltaJ = {Js[0] - Js[-1]:+.3e})")
    print(f"   churn  per step (steps 1..{len(h)-1}): "
          f"{churns}")
    print(f"   churn  mean = {np.mean(churns):.2f}   "
          f"max = {max(churns)}   nonzero steps = "
          f"{sum(1 for c in churns if c > 0)} / {len(churns)}")
    print(f"   mask sizes (mean / min / max): "
          f"{np.mean(sizes):.1f} / {min(sizes)} / {max(sizes)}")
    if extra_solves:
        print(f"   extra solves (E): {extra_solves} "
              f"(~{extra_solves/len(h):.1f}x per step)")
    print(f"   theta trajectory: {[f'{t:.4f}' for t in thetas[::5]]} "
          f"(every 5 steps)")
    print(f"   J trajectory:     {[f'{j:.3e}' for j in Js[::5]]} "
          f"(every 5 steps)")


def main():
    print(f"# Q3 chattering: 1D Poisson MIN-J descent, "
          f"theta0={THETA0} -> right boundary, K={K_ACTIVE}, "
          f"lr={LR}, steps={N_STEPS}")

    conds = [
        ("A (pure top-k)",
         lambda mag, prev: topk_mask(mag, K_ACTIVE), False),
        ("B (eps_remove=0.9*eps_add)",
         lambda mag, prev: hysteresis_mask(mag, K_ACTIVE, prev, 0.9), False),
        ("C (eps_remove=0.5*eps_add)",
         lambda mag, prev: hysteresis_mask(mag, K_ACTIVE, prev, 0.5), False),
        ("D (eps_remove=0.1*eps_add)",
         lambda mag, prev: hysteresis_mask(mag, K_ACTIVE, prev, 0.1), False),
        ("E (trust-region on top-k)",
         lambda mag, prev: topk_mask(mag, K_ACTIVE), True),
    ]
    results = {}
    for label, build, trust in conds:
        h, ex = run_condition(label, build, trust_region=trust)
        results[label] = (h, ex)
        summarize(label, h, ex)

    print("\n## Summary table")
    print(f"{'condition':<32} {'final_theta':>11} {'final_J':>11} "
          f"{'tot_churn':>9} {'mean_size':>9}")
    for label, (h, _) in results.items():
        churns = [r['churn'] for r in h[:-1] if r['churn'] >= 0][1:]
        sizes = [r['mask_size'] for r in h[:-1]]
        print(f"{label:<32} {h[-1]['theta']:>11.5f} "
              f"{h[-2]['J']:>11.4e} {sum(churns):>9d} "
              f"{np.mean(sizes):>9.1f}")


if __name__ == "__main__":
    main()
