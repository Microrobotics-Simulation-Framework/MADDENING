"""Gate 3, §6 -- time-dependent trajectory adjoint under lax.scan.

SPIKE CODE. Investigative. (JAX, float64.)

Production use runs the adaptive node inside a lax.scan time loop where the
mask changes each step and c_prev flows into the next step's residual /
selection. The plan asks: does jax.grad through the trajectory match FD, or
does gradient contamination via c_prev accumulate with the number of steps T?

Setup: 1D Poisson (-d2/dx2 + I) on a DD-4 basis. Source moves in time:
f(x; t, theta) = exp(-((x - theta(t))/sigma)^2), theta(t) = theta0 + 0.1 t.
State per step: (c, mask). Selection at each step uses residual r = b - A
c_prev (so c_prev genuinely seeds selection -- the plan's exact concern).
Mask is non-differentiable (argsort) and stop_gradient'd. Differentiable
masked solve uses the static-shape A_eff trick (inactive rows -> identity).

Objective J = sum_t u_t(x_sensor)^2. Compare jax.grad(J)/dtheta0 vs FD,
sweeping T in {1,2,5,10}.
"""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import dd_wavelets as dd

SIGMA = 0.06
SENSOR_X = 0.30


def build():
    W, levels, x = dd.synthesis_matrix(5, n_coarse=2, order=4, boundary="dirichlet")
    N = W.shape[0]
    h = 1.0 / (N + 1)
    norms = np.sqrt(h * np.sum(W ** 2, axis=0)); norms[norms == 0] = 1.0
    Wn = W / norms[None, :]
    A = Wn.T @ dd.laplacian_dirichlet(N, mass=True) @ Wn
    A = 0.5 * (A + A.T)
    D = 2.0 ** levels.astype(float)
    sidx = int(np.argmin(np.abs(x - SENSOR_X)))
    coarse = np.where(levels == levels.min())[0]
    return dict(
        Wn=jnp.asarray(Wn), A=jnp.asarray(A), D=jnp.asarray(D),
        x=jnp.asarray(x), levels=jnp.asarray(levels), h=h, N=N,
        sidx=sidx, srow=jnp.asarray(Wn[sidx]),
        coarse=jnp.asarray(coarse),
    )


def make_step(env, K):
    A = env["A"]; D = env["D"]; Wn = env["Wn"]; x = env["x"]; h = env["h"]
    srow = env["srow"]; N = env["N"]
    coarse = env["coarse"]
    coarse_mask = jnp.zeros(N, dtype=bool).at[coarse].set(True)

    def bvec(theta_t):
        f = jnp.exp(-((x - theta_t) / SIGMA) ** 2)
        return h * (Wn.T @ f)

    def select(b, c_prev):
        # residual-seeded selection (c_prev enters here) in scaled coords
        r = (b - A @ c_prev) / D
        score = jnp.abs(r)
        # always keep coarse: bump their score
        score = jnp.where(coarse_mask, jnp.inf, score)
        idx = jnp.argsort(-score)[:K]
        mask = jnp.zeros(N, dtype=bool).at[idx].set(True)
        return jax.lax.stop_gradient(mask)

    def masked_solve(b, mask):
        mm = (mask[:, None] & mask[None, :])
        A_eff = jnp.where(mm, A, 0.0) + jnp.diag(jnp.where(mask, 0.0, 1.0))
        rhs = jnp.where(mask, b, 0.0)
        return jnp.linalg.solve(A_eff, rhs)

    def step(c_prev, theta_t):
        b = bvec(theta_t)
        mask = select(b, c_prev)
        c = masked_solve(b, mask)
        u_sensor = srow @ c
        return c, u_sensor

    return step, bvec


def trajectory_J(env, theta0, T, K):
    step, _ = make_step(env, K)
    thetas = theta0 + 0.1 * jnp.arange(T)
    c0 = jnp.zeros(env["N"])

    def scan_body(c_prev, theta_t):
        c, u = step(c_prev, theta_t)
        return c, u

    _, us = jax.lax.scan(scan_body, c0, thetas)
    return jnp.sum(us ** 2)


def main():
    env = build()
    K = max(6, env["N"] // 8)
    print(f"DD-4 Dirichlet, N={env['N']}, K={K}, sensor x≈{float(env['x'][env['sidx']]):.3f}")
    print("trajectory adjoint: jax.grad vs FD through lax.scan, sweep T")
    print(f"{'T':>4} {'J':>13} {'grad':>13} {'FD':>13} {'rel_err':>10}")
    theta0 = 0.27  # chosen to avoid mask flips within the FD window
    for T in (1, 2, 5, 10):
        J = float(trajectory_J(env, jnp.asarray(theta0), T, K))
        g = float(jax.grad(lambda t: trajectory_J(env, t, T, K))(jnp.asarray(theta0)))
        eps = 1e-6
        Jp = float(trajectory_J(env, jnp.asarray(theta0 + eps), T, K))
        Jm = float(trajectory_J(env, jnp.asarray(theta0 - eps), T, K))
        fd = (Jp - Jm) / (2 * eps)
        rel = abs(g - fd) / (abs(fd) + 1e-30)
        flag = "  <-- MISMATCH" if rel > 1e-4 else ""
        print(f"{T:>4} {J:>13.6e} {g:>13.6e} {fd:>13.6e} {rel:>10.2e}{flag}")

    print("\nRobustness: max grad-vs-FD rel_err over several theta0 at T=10")
    worst = 0.0
    for theta0 in (0.20, 0.24, 0.27, 0.33, 0.38, 0.41):
        g = float(jax.grad(lambda t: trajectory_J(env, t, 10, K))(jnp.asarray(theta0)))
        eps = 1e-6
        fd = (float(trajectory_J(env, jnp.asarray(theta0 + eps), 10, K))
              - float(trajectory_J(env, jnp.asarray(theta0 - eps), 10, K))) / (2 * eps)
        rel = abs(g - fd) / (abs(fd) + 1e-30)
        worst = max(worst, rel)
        print(f"  theta0={theta0:.2f}: rel_err={rel:.2e}")
    print(f"  worst rel_err = {worst:.2e}  "
          f"({'PASS' if worst < 1e-4 else 'FAIL'} vs plan tol 1e-6 at non-kink)")


if __name__ == "__main__":
    main()
