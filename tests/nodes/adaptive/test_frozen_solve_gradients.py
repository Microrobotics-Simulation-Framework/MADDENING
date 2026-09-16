"""Gradients through the frozen solve: against finite differences and
against dense closed-form references (float64 via the conftest fixture)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tests.nodes.adaptive._toys import MaskedDenseNode, PoissonSineTopKNode

H = 1e-5


def _central_fd(f, x, h=H):
    return (f(x + h) - f(x - h)) / (2 * h)


def _sine_J(node, s):
    def J(th):
        out = node.update(s, {}, 1.0, params={"theta": th})
        return node.objective(out, {**node.params, "theta": th})
    return J


def test_sine_gradient_matches_finite_differences():
    node = PoissonSineTopKNode(theta=0.42)
    s = node.initial_state()
    J = _sine_J(node, s)
    th0 = jnp.asarray(0.42)
    g, fd = float(jax.grad(J)(th0)), float(_central_fd(J, th0))
    assert abs(g - fd) / abs(fd) < 1e-6, (g, fd)


def test_sine_gradient_matches_closed_form_masked_solution():
    """``c = b / lambda`` on the active set is the exact frozen solve; its
    gradient is the exact frozen-set adjoint the Krylov path must reproduce."""
    node = PoissonSineTopKNode(theta=0.42)
    s = node.initial_state()
    th0 = jnp.asarray(0.42)
    mask = node.compute_active_set(s, {**node.params, "theta": th0})

    def J_ref(th):
        p = {**node.params, "theta": th}
        return node._phi_sensor @ jnp.where(mask, node.full_solution_coefficients(p), 0.0)

    assert jnp.allclose(jax.grad(_sine_J(node, s))(th0), jax.grad(J_ref)(th0), rtol=1e-9)


def test_krylov_path_on_the_sine_toy_matches_closed_form():
    """Same check as above through the preconditioned CG path."""
    node = PoissonSineTopKNode(theta=0.42, solver="cg", blindness_gate=False)
    s = node.initial_state()
    th0 = jnp.asarray(0.42)
    mask = s["mask"]

    def J_ref(th):
        p = {**node.params, "theta": th}
        return node._phi_sensor @ jnp.where(mask, node.full_solution_coefficients(p), 0.0)

    assert jnp.allclose(s["c"], jnp.where(mask, node.full_solution_coefficients(node.params), 0.0), atol=1e-10)
    assert jnp.allclose(jax.grad(_sine_J(node, s))(th0), jax.grad(J_ref)(th0), rtol=1e-8)


def test_default_full_basis_gradient_matches_closed_form():
    node = PoissonSineTopKNode(theta=0.42)
    s = node.initial_state()
    g = node.compute_full_basis_gradient(s)["theta"]

    def J_full(th):
        p = {**node.params, "theta": th}
        return node._phi_sensor @ node.full_solution_coefficients(p)

    ref = jax.grad(J_full)(jnp.asarray(0.42))
    assert jnp.allclose(g, ref, rtol=1e-6)


def test_gradient_with_respect_to_the_whole_parameter_pytree():
    node = PoissonSineTopKNode(theta=0.42)
    s = node.initial_state()
    pt = {k: v.astype(jnp.float64) for k, v in node.params_pytree().items()}

    def J(p):
        out = node.update(s, {}, 1.0, params=p)
        return node.objective(out, {**node.params, **p})

    g = jax.grad(J)(pt)
    assert set(g) == {"theta", "sigma", "sensor_x"}
    fd_sigma = float(_central_fd(lambda sg: J({**pt, "sigma": sg}), pt["sigma"], 1e-6))
    assert abs(float(g["sigma"]) - fd_sigma) / abs(fd_sigma) < 1e-5
    assert float(g["sensor_x"]) == 0.0  # the objective closes over the constructor value


@pytest.mark.parametrize("solver", ["cg", "gmres"])
def test_dense_solution_and_gradient_match_dense_sub_block_solve(solver):
    node = MaskedDenseNode(theta=0.3, blindness_gate=False, solver=solver)
    s = node.initial_state()
    th0 = jnp.asarray(0.3)
    mask = node.compute_active_set(s, {**node.params, "theta": th0})
    idx = jnp.nonzero(mask)[0]

    def J_ref(th):
        p = {"theta": th}
        A = node.matrix(p)[jnp.ix_(idx, idx)]
        c_m = jnp.linalg.solve(A, node.rhs(p)[idx])
        return node.s[idx] @ c_m

    def J(th):
        out = node.update(s, {}, 1.0, params={"theta": th})
        return node.objective(out, {})

    c_ref = jnp.zeros(node.n_max).at[idx].set(
        jnp.linalg.solve(node.matrix({"theta": th0})[jnp.ix_(idx, idx)], node.rhs({"theta": th0})[idx])
    )
    assert jnp.allclose(node.update(s, {}, 1.0, params={"theta": th0})["c"], c_ref, atol=1e-9)
    g, g_ref = float(jax.grad(J)(th0)), float(jax.grad(J_ref)(th0))
    fd = float(_central_fd(J, th0))
    assert abs(g - g_ref) / abs(g_ref) < 1e-8, (g, g_ref)
    assert abs(g - fd) / abs(fd) < 1e-6, (g, fd)
