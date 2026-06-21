"""Q4: Does the frozen-set adjoint behave correctly inside lax.scan?

We thread the active-set selection inside the scan body (re-threshold per
timestep) under stop_gradient.  Verify jax.grad through the scan gives a
finite, sensible value and matches a Python-loop equivalent.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

N = 64
ks = jnp.arange(1, N + 1)
LAMBDAS = (ks * jnp.pi) ** 2 + 1.0
x_grid = jnp.linspace(0.0, 1.0, 128)
dx = float(x_grid[1] - x_grid[0])
PHI = jnp.sin(jnp.pi * jnp.outer(x_grid, ks))
SENSOR = 40
K_ACTIVE = 16
N_STEPS = 12


def rhs(theta):
    f = jnp.exp(-((x_grid - theta) / 0.05) ** 2)
    return 2.0 * dx * (PHI.T @ f)


def step(state, theta):
    """One 'timestep': re-threshold + diffusion smoothing in mask space.
    state = (c_padded, _).  c_padded propagates between steps."""
    c, _ = state
    # decay + new source projection
    new_b = rhs(theta)
    mag = jnp.abs(new_b)
    threshold = jnp.sort(mag)[-K_ACTIVE]
    mask = jax.lax.stop_gradient(mag >= threshold)
    # mix decayed coefficient with new RHS on active set
    c_new = jnp.where(mask, 0.5 * c + new_b / LAMBDAS, 0.0)
    return (c_new, mask), c_new[SENSOR // 2]  # carry forward + per-step output


def J_scan(theta):
    init = (jnp.zeros(N), jnp.zeros(N, dtype=bool))
    (final_c, _), per_step = jax.lax.scan(step, init, jnp.full((N_STEPS,), theta))
    u = PHI @ final_c
    return u[SENSOR] + per_step.sum()


def J_pyloop(theta):
    c = jnp.zeros(N)
    per_step_sum = 0.0
    for _ in range(N_STEPS):
        (c, _), out = step((c, None), theta)
        per_step_sum = per_step_sum + out
    u = PHI @ c
    return u[SENSOR] + per_step_sum


theta0 = jnp.asarray(0.4)
print(f"J_scan({theta0})   = {float(J_scan(theta0)):+.6e}")
print(f"J_pyloop({theta0}) = {float(J_pyloop(theta0)):+.6e}")
print()
g_scan = jax.grad(J_scan)(theta0)
g_py = jax.grad(J_pyloop)(theta0)
print(f"grad J_scan   = {float(g_scan):+.6e}")
print(f"grad J_pyloop = {float(g_py):+.6e}")
print(f"rel err scan vs py-loop: "
      f"{abs(float(g_scan - g_py))/(abs(float(g_py)) + 1e-30):.2e}")

# FD baseline on J_scan
eps = 1e-5
fd = (J_scan(theta0 + eps) - J_scan(theta0 - eps)) / (2.0 * eps)
print(f"FD on J_scan  = {float(fd):+.6e}   "
      f"rel_err vs grad: {abs(float(g_scan - fd))/(abs(float(fd))+1e-30):.2e}")

# Now check: WITHOUT stop_gradient -- does jax.grad still work, or fail/wrong?
def step_no_sg(state, theta):
    c, _ = state
    new_b = rhs(theta)
    mag = jnp.abs(new_b)
    threshold = jnp.sort(mag)[-K_ACTIVE]
    mask = mag >= threshold   # NO stop_gradient
    c_new = jnp.where(mask, 0.5 * c + new_b / LAMBDAS, 0.0)
    return (c_new, mask), c_new[SENSOR // 2]


def J_no_sg(theta):
    init = (jnp.zeros(N), jnp.zeros(N, dtype=bool))
    (final_c, _), per_step = jax.lax.scan(step_no_sg, init, jnp.full((N_STEPS,), theta))
    u = PHI @ final_c
    return u[SENSOR] + per_step.sum()


try:
    g_no_sg = float(jax.grad(J_no_sg)(theta0))
    print(f"\ngrad J without stop_gradient on mask = {g_no_sg:+.6e}")
    print(f"  diff vs WITH stop_gradient: "
          f"{abs(g_no_sg - float(g_scan)):.2e}")
    print("  (JAX silently treats `>=` as non-differentiable -- "
          "no error, but no contribution from the mask construction either)")
except Exception as e:
    print(f"\ngrad without stop_gradient raised: {type(e).__name__}: {e}")
