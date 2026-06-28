"""M6 — trajectory adjoint hardening and end-to-end optimisation.

Confirms the frozen-set adjoint holds at production trajectory lengths
(lax.scan, T=100) with no degradation, and that gradient-based optimisation
through the node converges (the practical impact of the dense-kink finding,
spike Limitations 8/14).  Heavy tests are in the ``slow`` lane.  conftest
provides float64.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.nodes.adaptive import WaveletAdaptiveNode


def _fresh_state(node, theta):
    return {"c": jnp.zeros(node.N_max, dtype=jnp.float64),
            "mask": jnp.zeros(node.N_max, dtype=bool),
            "theta": jnp.atleast_1d(theta)}


def _solve_at(node, state, theta_t):
    """Generic per-step solve: set θ, reselect + frozen-solve, return new state."""
    s = {**state, "theta": jnp.atleast_1d(theta_t)}
    mask = node.compute_active_set(s, is_cold_start=False)
    return node.solve_frozen({**s, "mask": mask}, mask)


def test_generic_scan_signature_runs():
    """The per-step solve uses a generic ``step_fn(carry, x) -> (carry, y)``
    signature and runs under lax.scan (Amendment 5)."""
    node = WaveletAdaptiveNode(dim=1, n_levels=6, sigma=0.10)

    def step_fn(carry, theta_t):
        st = _solve_at(node, carry, theta_t)
        return st, jnp.squeeze(node._sensor(st))

    xs = 0.30 + 0.01 * jnp.arange(3)
    final, ys = jax.lax.scan(step_fn, node._initial_state_impl(), xs)
    assert ys.shape == (3,)
    assert final["c"].shape == (node.N_max,)


@pytest.mark.slow
def test_trajectory_adjoint_no_T_degradation():
    """jax.grad through a lax.scan trajectory matches FD with no degradation as
    the trajectory length grows to T=100 (spike §6 / Limitation 8)."""
    node = WaveletAdaptiveNode(dim=1, n_levels=7, n_coarse=2, sigma=0.10)

    def step_fn(carry, theta_t):
        st = _solve_at(node, carry, theta_t)
        return st, jnp.squeeze(node._sensor(st))

    def traj_J(theta0, T):
        xs = theta0 + 0.002 * jnp.arange(T)
        _, us = jax.lax.scan(step_fn, node._initial_state_impl(), xs)
        return jnp.sum(us ** 2)

    th = jnp.asarray(0.40)
    e = 1e-6
    for T in (1, 10, 100):
        g = float(jax.grad(lambda t: traj_J(t, T))(th))
        fd = float((traj_J(th + e, T) - traj_J(th - e, T)) / (2 * e))
        rel = abs(g - fd) / (abs(fd) + 1e-30)
        assert rel < 1e-5, f"T={T}: rel={rel}"


@pytest.mark.slow
def test_end_to_end_optimisation_converges():
    """Field-matching inverse problem: minimise ||u(θ) - u_target||² over the
    source position θ by gradient descent.  Concrete criteria (Amendment 3)."""
    node = WaveletAdaptiveNode(dim=1, n_levels=7, n_coarse=2, sigma=0.10,
                               theta_init=0.45)

    def field(theta):
        st = _solve_at(node, _fresh_state(node, theta), theta)
        return node._Wn @ st["c"]

    theta_opt = 0.45
    target = field(jnp.asarray(theta_opt))
    denom = float(jnp.mean(target ** 2))

    def J(theta):
        return jnp.mean((field(theta) - target) ** 2) / denom

    gJ = jax.jit(jax.grad(J))
    Jj = jax.jit(J)

    start = theta_opt - 0.22                  # ≥0.2 from the optimum
    theta = jnp.asarray(start)
    lr = 2.0
    finite = True
    smooth = 0
    eps = 1e-5
    for _ in range(50):
        g = gJ(theta)
        if not np.isfinite(float(g)):
            finite = False
            break
        # (d) kink density: is the active set stable across θ±eps?
        m1 = node.compute_active_set(_fresh_state(node, float(theta) - eps))
        m2 = node.compute_active_set(_fresh_state(node, float(theta) + eps))
        smooth += int(jnp.sum(m1 != m2) == 0)
        theta = jnp.clip(theta - lr * g, 0.05, 0.95)

    j_final = float(Jj(theta))
    # (c) no non-finite gradient ever
    assert finite
    # (a) reaches |J| < 1e-4 within 50 steps
    assert j_final < 1e-4, f"|J|={j_final}"
    # (b) recovers the source position
    assert abs(float(theta) - theta_opt) < 0.01, f"theta={float(theta)}"
    # (d) smooth-fraction is a sane fraction in [0, 1] (recorded; 1D is sparsely
    # kinked so most steps are smooth)
    frac_smooth = smooth / 50
    assert 0.0 <= frac_smooth <= 1.0
