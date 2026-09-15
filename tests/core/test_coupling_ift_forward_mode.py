"""Forward-mode AD through the IFT coupling solver.

``_ift_solve`` is a ``jax.custom_jvp``; JAX derives reverse mode by
transposing its tangent rule.  These tests pin down the consequence:
one definition serves ``jvp`` / ``jacfwd`` (the FMI ``FORWARD``
directional derivative), ``grad`` / ``jacrev``, and second order — all
through the *jitted* step, and all agreeing with the unrolled fori path.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi.directional_derivatives import (
    DirectionalDerivativeKind,
    get_directional_derivative,
)
from maddening.nodes.spring import SpringDamperNode


def _make_gm(solver: str, acceleration: str = "none", **group_kw) -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode(
        name="spring_a", timestep=0.001, stiffness=50.0, damping=1.0,
        mass=1.0, rest_length=1.0, initial_position=0.0,
    ))
    gm.add_node(SpringDamperNode(
        name="spring_b", timestep=0.001, stiffness=50.0, damping=1.0,
        mass=1.0, rest_length=1.0, initial_position=2.0,
    ))
    gm.add_edge("spring_a", "spring_b", "position", "anchor_position")
    gm.add_edge("spring_b", "spring_a", "position", "anchor_position")
    gm.add_coupling_group(
        ["spring_a", "spring_b"], max_iterations=30, tolerance=1e-8,
        acceleration=acceleration, solver=solver, **group_kw,
    )
    return gm


def _positions_fn(gm: GraphManager):
    """``(pos_a, pos_b) -> stacked positions after one jitted step``."""
    _ = gm.step()
    compiled = gm._compiled_step
    base = gm._state

    def f(p):
        state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        state["spring_a"]["position"] = p[0]
        state["spring_b"]["position"] = p[1]
        out = compiled(state, {})
        return jnp.stack([out["spring_a"]["position"], out["spring_b"]["position"]])

    return f


P0 = jnp.array([0.1, 1.9], dtype=jnp.float32)
V = jnp.array([1.0, -0.5], dtype=jnp.float32)


@pytest.mark.parametrize("acceleration", ["none", "aitken", "iqn-imvj"])
def test_jvp_through_jitted_step_matches_fori(acceleration):
    f_ift = _positions_fn(_make_gm("ift", acceleration))
    f_fori = _positions_fn(_make_gm("fori", acceleration))
    _, t_ift = jax.jvp(f_ift, (P0,), (V,))
    _, t_fori = jax.jvp(f_fori, (P0,), (V,))
    np.testing.assert_allclose(t_ift, t_fori, rtol=1e-3, atol=1e-4)


def test_jacfwd_and_jacrev_agree_and_match_fori():
    f_ift = _positions_fn(_make_gm("ift"))
    f_fori = _positions_fn(_make_gm("fori"))
    J_fwd = jax.jacfwd(f_ift)(P0)
    J_rev = jax.jacrev(f_ift)(P0)
    J_ref = jax.jacfwd(f_fori)(P0)
    np.testing.assert_allclose(J_fwd, J_rev, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(J_fwd, J_ref, rtol=1e-3, atol=1e-4)


def test_forward_reverse_adjoint_identity():
    """<w, J v> == <J^T w, v> through the same custom_jvp definition."""
    f = _positions_fn(_make_gm("ift"))
    w = jnp.array([0.3, -1.2], dtype=jnp.float32)
    _, Jv = jax.jvp(f, (P0,), (V,))
    _, vjp_fn = jax.vjp(f, P0)
    (JTw,) = vjp_fn(w)
    np.testing.assert_allclose(jnp.dot(w, Jv), jnp.dot(JTw, V), rtol=1e-4, atol=1e-5)


def test_hessian_through_ift_step():
    f_ift = _positions_fn(_make_gm("ift"))
    f_fori = _positions_fn(_make_gm("fori"))
    loss = lambda g: (lambda p: jnp.sum(g(p) ** 2))
    H_ift = jax.hessian(loss(f_ift))(P0)
    H_ref = jax.hessian(loss(f_fori))(P0)
    assert bool(jnp.all(jnp.isfinite(H_ift)))
    np.testing.assert_allclose(H_ift, H_ref, rtol=2e-2, atol=1e-3)


@pytest.mark.parametrize("linear_solver", ["gmres", "dense"])
def test_jvp_linear_solver_backends(linear_solver):
    f = _positions_fn(_make_gm("ift", linear_solver=linear_solver))
    f_ref = _positions_fn(_make_gm("fori"))
    _, t = jax.jvp(f, (P0,), (V,))
    _, t_ref = jax.jvp(f_ref, (P0,), (V,))
    np.testing.assert_allclose(t, t_ref, rtol=1e-3, atol=1e-4)


class TestFMIDirectionalDerivative:
    """The FMI surface must work in both directions through a coupled step."""

    def test_forward_kind_through_ift_step(self):
        f = _positions_fn(_make_gm("ift"))
        out = get_directional_derivative(
            f, kind=DirectionalDerivativeKind.FORWARD, x=P0, v=V,
        )
        expected = jax.jacfwd(_positions_fn(_make_gm("fori")))(P0) @ V
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-4)

    def test_reverse_kind_through_ift_step(self):
        f = _positions_fn(_make_gm("ift"))
        w = jnp.array([0.3, -1.2], dtype=jnp.float32)
        out = get_directional_derivative(
            f, kind=DirectionalDerivativeKind.REVERSE, x=P0, v=w,
        )
        expected = jax.jacfwd(_positions_fn(_make_gm("fori")))(P0).T @ w
        np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-4)
