"""Locality theorem: empirical comparison of forward vs adjoint
convergence under top-|c| selection in a LOCAL (Haar) vs NON-LOCAL
(sine) basis.

Same physical problem in both bases: 1D FD Dirichlet Poisson + I,
(-u'' + u) = f(x; theta), Gaussian source.  N = 256 interior points,
dx = 1/(N+1).

Sine basis (= DST-I eigenfunctions of the FD Dirichlet Laplacian):
non-local (global sinusoid support).  Haar wavelet basis: local
(multi-scale step-function support).

For each k_active in {4, 8, 16, 32, 64, 128, 192}:
  - J error  = |J_frozen(theta; k) - J_full(theta)| / |J_full|
  - g error  = |grad J_frozen - grad J_full| / |grad J_full|

CLAIM under test (memo, cross-cutting finding #2): in a LOCAL basis,
top-|c| selection makes J error and gradient error decay at the same
rate (a "locality theorem").  In a NON-LOCAL basis, they decouple
(Q2 Sweep 3 showed the sine case: J error 1e-14 at k=128, gradient
error 7%).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

N = 256                       # interior points (2^8, required for Haar)
SIGMA = 0.05
THETAS = [0.30, 0.35, 0.42, 0.55]
SENSOR_IDX = N // 3

# -- physical-space FD Dirichlet (-u'' + u) operator --
dx = 1.0 / (N + 1)
x_grid_np = np.arange(1, N + 1) * dx          # interior, length N
A_PHYS_np = (2.0 / dx ** 2) * np.eye(N) + np.eye(N) \
    - (1.0 / dx ** 2) * (np.eye(N, k=1) + np.eye(N, k=-1))

# -- DST-I basis (diagonalizes A_PHYS exactly) --
ks = np.arange(1, N + 1)
S_np = np.sqrt(2.0 / (N + 1)) * np.sin(
    np.pi * np.outer(ks, ks) / (N + 1)
)
EIGVALS_SINE_np = (
    2.0 * (1.0 - np.cos(np.pi * ks / (N + 1))) / dx ** 2 + 1.0
)


def haar_matrix(n: int) -> np.ndarray:
    """Standard orthogonal Haar DWT matrix: c = W f, f = W.T c."""
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


W_HAAR_np = haar_matrix(N)
A_HAAR_np = W_HAAR_np @ A_PHYS_np @ W_HAAR_np.T

# move to jax
x_grid = jnp.asarray(x_grid_np)
S_J = jnp.asarray(S_np)
EIGVALS_SINE = jnp.asarray(EIGVALS_SINE_np)
W_HAAR = jnp.asarray(W_HAAR_np)
A_HAAR = jnp.asarray(A_HAAR_np)
EYE_N = jnp.eye(N)


def source_grid(theta):
    return jnp.exp(-((x_grid - theta) / SIGMA) ** 2)


def J_sine_full(theta):
    b = S_J @ source_grid(theta)
    c = b / EIGVALS_SINE
    u = S_J.T @ c
    return u[SENSOR_IDX]


def J_sine_frozen(theta, k_active):
    b = S_J @ source_grid(theta)
    mag = jnp.abs(b)
    threshold = jnp.sort(mag)[-k_active]
    mask = jax.lax.stop_gradient(mag >= threshold)
    c = jnp.where(mask, b / EIGVALS_SINE, 0.0)
    u = S_J.T @ c
    return u[SENSOR_IDX]


def J_haar_full(theta):
    b = W_HAAR @ source_grid(theta)
    c = jnp.linalg.solve(A_HAAR, b)
    u = W_HAAR.T @ c
    return u[SENSOR_IDX]


def J_haar_frozen(theta, k_active):
    b = W_HAAR @ source_grid(theta)
    mag = jnp.abs(b)
    threshold = jnp.sort(mag)[-k_active]
    mask = jax.lax.stop_gradient(mag >= threshold)
    # masked dense solve: inactive rows -> identity, inactive cols -> zero
    A_eff = jnp.where(mask[:, None], mask[None, :] * A_HAAR, EYE_N)
    b_eff = mask * b
    c = jnp.linalg.solve(A_eff, b_eff)
    u = W_HAAR.T @ c
    return u[SENSOR_IDX]


def J_sine_topc_frozen(theta, k_active):
    """Sine basis with mask = top-|c|, where c is the full-basis solution."""
    b = S_J @ source_grid(theta)
    c_full = b / EIGVALS_SINE
    mag_c = jnp.abs(c_full)
    threshold = jnp.sort(mag_c)[-k_active]
    mask = jax.lax.stop_gradient(mag_c >= threshold)
    c = jnp.where(mask, b / EIGVALS_SINE, 0.0)
    u = S_J.T @ c
    return u[SENSOR_IDX]


def J_haar_topc_frozen(theta, k_active):
    """Haar basis with mask = top-|c| from a preliminary full solve."""
    b = W_HAAR @ source_grid(theta)
    c_full_for_mask = jnp.linalg.solve(A_HAAR, b)
    mag_c = jnp.abs(c_full_for_mask)
    threshold = jnp.sort(mag_c)[-k_active]
    mask = jax.lax.stop_gradient(mag_c >= threshold)
    A_eff = jnp.where(mask[:, None], mask[None, :] * A_HAAR, EYE_N)
    b_eff = mask * b
    c = jnp.linalg.solve(A_eff, b_eff)
    u = W_HAAR.T @ c
    return u[SENSOR_IDX]


grad_sine_full = jax.jit(jax.grad(J_sine_full))
grad_sine_frozen = jax.jit(jax.grad(J_sine_frozen), static_argnums=1)
grad_sine_topc = jax.jit(jax.grad(J_sine_topc_frozen), static_argnums=1)
grad_haar_full = jax.jit(jax.grad(J_haar_full))
grad_haar_frozen = jax.jit(jax.grad(J_haar_frozen), static_argnums=1)
grad_haar_topc = jax.jit(jax.grad(J_haar_topc_frozen), static_argnums=1)


def relerr(x, ref):
    return float(abs(x - ref) / (abs(ref) + 1e-30))


def main():
    print(f"# Locality theorem: N = {N}, sigma = {SIGMA}, "
          f"sensor at x = {float(x_grid[SENSOR_IDX]):.4f}")
    print(f"# (-u'' + u) = exp(-((x-theta)/sigma)^2),  Dirichlet BC")
    KS = [4, 8, 16, 32, 64, 128, 192]
    for theta_val in THETAS:
        theta = jnp.asarray(theta_val)
        Js_ref = float(J_sine_full(theta))
        gs_ref = float(grad_sine_full(theta))
        Jh_ref = float(J_haar_full(theta))
        gh_ref = float(grad_haar_full(theta))

        print(f"\n## theta = {theta_val}")
        print(f"   sine full  J = {Js_ref:+.6e}, dJ/dtheta = {gs_ref:+.6e}")
        print(f"   haar full  J = {Jh_ref:+.6e}, dJ/dtheta = {gh_ref:+.6e}")
        print(f"   (cross-basis J check: |sine - haar| / |sine| = "
              f"{abs(Js_ref-Jh_ref)/(abs(Js_ref)+1e-30):.2e})")
        print()
        print(f"   selection = top-|b| (RHS-magnitude)")
        print(f"   {'k':>4} | {'sine_J_err':>11} {'sine_g_err':>11} | "
              f"{'haar_J_err':>11} {'haar_g_err':>11}")
        for k in KS:
            Js = float(J_sine_frozen(theta, k))
            gs = float(grad_sine_frozen(theta, k))
            Jh = float(J_haar_frozen(theta, k))
            gh = float(grad_haar_frozen(theta, k))
            print(f"   {k:>4d} | "
                  f"{relerr(Js, Js_ref):>11.2e} {relerr(gs, gs_ref):>11.2e} | "
                  f"{relerr(Jh, Jh_ref):>11.2e} {relerr(gh, gh_ref):>11.2e}")
        print()
        print(f"   selection = top-|c| (solution-magnitude)")
        print(f"   {'k':>4} | {'sine_J_err':>11} {'sine_g_err':>11} | "
              f"{'haar_J_err':>11} {'haar_g_err':>11}")
        for k in KS:
            Js = float(J_sine_topc_frozen(theta, k))
            gs = float(grad_sine_topc(theta, k))
            Jh = float(J_haar_topc_frozen(theta, k))
            gh = float(grad_haar_topc(theta, k))
            print(f"   {k:>4d} | "
                  f"{relerr(Js, Js_ref):>11.2e} {relerr(gs, gs_ref):>11.2e} | "
                  f"{relerr(Jh, Jh_ref):>11.2e} {relerr(gh, gh_ref):>11.2e}")


if __name__ == "__main__":
    main()
