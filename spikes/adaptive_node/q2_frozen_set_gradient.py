"""Q2: Does the frozen-set IFT adjoint give correct gradients?

Toy: 1D Poisson -u'' + u = f(x; theta) on [0,1] with Dirichlet BC.
Sine (Fourier) basis on N modes -- Laplacian + I is diagonal:
    A = diag(lambda_k + 1),   lambda_k = (k*pi)^2
So the "solve" is c_k = b_k / (lambda_k + 1).

f(x; theta) = exp(-((x-theta)/sigma)^2).   theta drives the source bump.
Objective J(theta) = u(x_sensor).

Active set: top-k coefficients of |b|.  Mask under stop_gradient.

We compare three gradients of J w.r.t. theta:
- IFT-frozen: jax.grad through the frozen-set solve (no flow through mask).
- FD-fine:    central difference with h=1e-4 -- baseline, agrees with IFT
              when mask doesn't flip in the FD window.
- FD-flip:    central difference with h chosen large enough to span a
              threshold-flip event, evaluated at a theta where a flip
              occurs.  This is the "gold standard" that includes
              partial(mask)/partial(theta) as a discrete jump.

Claim under test: as the active set grows (resolution improves), the
flip-induced gap shrinks because the boundary coefficient magnitude
shrinks.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

N_BASIS = 256
N_GRID = 512
SIGMA = 0.04
SENSOR_IDX = N_GRID // 3

ks = jnp.arange(1, N_BASIS + 1)
LAMBDAS = (ks * jnp.pi) ** 2 + 1.0          # diag(A)
x_grid = jnp.linspace(0.0, 1.0, N_GRID)
dx = float(x_grid[1] - x_grid[0])
# Sine basis evaluated on the grid: PHI[i, k-1] = sin(k*pi*x_i)
PHI = jnp.sin(jnp.pi * jnp.outer(x_grid, ks))


def rhs_coeffs(theta: jax.Array) -> jax.Array:
    f = jnp.exp(-((x_grid - theta) / SIGMA) ** 2)
    # crude L2 projection (Riemann sum * 2 for sine series on [0,1])
    return 2.0 * dx * (PHI.T @ f)


def J_frozen(theta: jax.Array, k_active: int) -> jax.Array:
    b = rhs_coeffs(theta)
    mag = jnp.abs(b)
    # Stop-gradient threshold via sorted top-k.
    sorted_mag = jnp.sort(mag)
    threshold = sorted_mag[-k_active]
    mask = jax.lax.stop_gradient(mag >= threshold)
    c = jnp.where(mask, b / LAMBDAS, 0.0)
    u = PHI @ c
    return u[SENSOR_IDX]


def J_full(theta: jax.Array) -> jax.Array:
    b = rhs_coeffs(theta)
    c = b / LAMBDAS
    u = PHI @ c
    return u[SENSOR_IDX]


grad_J_frozen = jax.jit(jax.grad(J_frozen), static_argnums=1)
grad_J_full = jax.jit(jax.grad(J_full))


def fd(fn, theta, h):
    return float((fn(theta + h) - fn(theta - h)) / (2.0 * h))


def active_set_of(theta, k_active):
    b = rhs_coeffs(jnp.asarray(theta))
    mag = jnp.abs(b)
    threshold = jnp.sort(mag)[-k_active]
    return np.asarray(mag >= threshold)


def find_flip_theta(k_active: int, theta_lo: float = 0.20,
                    theta_hi: float = 0.50, n: int = 4001):
    """Locate a theta where the active set changes between adjacent grid points."""
    thetas = np.linspace(theta_lo, theta_hi, n)
    prev = active_set_of(thetas[0], k_active)
    for i in range(1, n):
        cur = active_set_of(thetas[i], k_active)
        if not np.array_equal(prev, cur):
            return 0.5 * (thetas[i - 1] + thetas[i]), int(np.sum(prev ^ cur))
        prev = cur
    return None, 0


def main():
    print(f"# Basis size N_BASIS = {N_BASIS}, sigma = {SIGMA}")
    print(f"# Grid N_GRID = {N_GRID}, sensor at x = {float(x_grid[SENSOR_IDX]):.3f}")
    theta0 = jnp.asarray(0.35)
    g_full = float(grad_J_full(theta0))
    g_fd_full = fd(lambda t: J_full(t), theta0, 1e-4)
    print(f"# Sanity (no masking): IFT={g_full:+.6e}   FD={g_fd_full:+.6e}   "
          f"rel_err={abs(g_full - g_fd_full)/abs(g_full):.2e}\n")

    print("## Sweep 1 -- gradient error at a fixed theta away from any flip event")
    print(f"{'k_active':>8}  {'IFT-frozen':>14}  {'FD h=1e-4':>14}  {'rel_err':>10}")
    for k in [4, 8, 16, 32, 64, 128, 256]:
        g_ift = float(grad_J_frozen(theta0, k))
        g_fd = fd(lambda t: J_frozen(t, k), theta0, 1e-4)
        err = abs(g_ift - g_fd) / (abs(g_fd) + 1e-30)
        print(f"{k:>8d}  {g_ift:>+14.6e}  {g_fd:>+14.6e}  {err:>10.2e}")

    print("\n## Sweep 2 -- gradient error across a known flip event")
    print(f"{'k_active':>8}  {'theta_flip':>10}  {'jumped':>6}  "
          f"{'IFT-frozen':>14}  {'FD across flip':>14}  {'rel_err':>10}")
    for k in [8, 16, 32, 64, 128]:
        theta_flip, n_jumped = find_flip_theta(k)
        if theta_flip is None:
            print(f"{k:>8d}  {'(none)':>10}")
            continue
        # h large enough to span the flip (find the gap first)
        # Locate the nearest two thetas straddling the flip
        h = 5e-4
        g_ift = float(grad_J_frozen(jnp.asarray(theta_flip), k))
        g_fd_flip = fd(lambda t: J_frozen(t, k), jnp.asarray(theta_flip), h)
        err = abs(g_ift - g_fd_flip) / (abs(g_fd_flip) + 1e-30)
        print(f"{k:>8d}  {theta_flip:>10.5f}  {n_jumped:>6d}  "
              f"{g_ift:>+14.6e}  {g_fd_flip:>+14.6e}  {err:>10.2e}")

    print("\n## Sweep 3 -- convergence vs full-basis (no-mask) gradient")
    print(f"{'k_active':>8}  {'|J_k - J_full|':>16}  {'|g_k - g_full|':>16}  "
          f"{'rel_g':>10}")
    J_full_val = float(J_full(theta0))
    for k in [4, 8, 16, 32, 64, 128, 256]:
        Jv = float(J_frozen(theta0, k))
        g = float(grad_J_frozen(theta0, k))
        print(f"{k:>8d}  {abs(Jv - J_full_val):>16.6e}  "
              f"{abs(g - g_full):>16.6e}  "
              f"{abs(g - g_full)/abs(g_full):>10.2e}")


if __name__ == "__main__":
    main()
