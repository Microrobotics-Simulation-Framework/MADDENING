"""``update`` traces once: jit, scan and vmap with fixed shapes and no
data-dependent structure; the adjoint is the frozen-set one."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tests.nodes.adaptive._toys import MaskedDenseNode, PoissonSineTopKNode


def test_jit_update_matches_eager():
    node = PoissonSineTopKNode()
    s = node.initial_state()
    pt = node.params_pytree()
    eager = node.update(s, {}, 1.0, params=pt)
    jitted = jax.jit(lambda st, p: node.update(st, {}, 1.0, params=p))(s, pt)
    assert jnp.allclose(eager["c"], jitted["c"], atol=1e-12)
    assert bool(jnp.array_equal(eager["mask"], jitted["mask"]))


def test_scan_through_update_keeps_shapes_and_matches_python_loop():
    """The active set changes between steps (theta drifts) but the array
    shapes never do, so one trace serves the whole scan."""
    node = PoissonSineTopKNode(theta=0.42)
    s0 = node.initial_state()
    thetas = jnp.linspace(0.42, 0.7, 6)

    def body(st, theta):
        new = node.update(st, {}, 1.0, params={"theta": theta})
        return new, (new["c"].shape, int(new["mask"].shape[0]))

    final, shapes = jax.jit(lambda st: jax.lax.scan(body, st, thetas))(s0)
    assert final["c"].shape == (node.n_max,) and final["mask"].shape == (node.n_max,)
    assert final["mask"].dtype == jnp.bool_

    ref = s0
    masks = []
    for th in thetas:
        ref = node.update(ref, {}, 1.0, params={"theta": th})
        masks.append(ref["mask"])
    assert jnp.allclose(final["c"], ref["c"], atol=1e-12)
    assert bool(jnp.array_equal(final["mask"], ref["mask"]))
    # the set genuinely adapted along the way
    assert not bool(jnp.array_equal(masks[0], masks[-1]))


def test_vmap_over_parameters():
    node = PoissonSineTopKNode()
    s = node.initial_state()
    thetas = jnp.array([0.3, 0.42, 0.6])
    out = jax.vmap(lambda th: node.update(s, {}, 1.0, params={"theta": th}))(thetas)
    assert out["c"].shape == (3, node.n_max) and out["mask"].shape == (3, node.n_max)
    assert bool(jnp.all(out["mask"].sum(axis=1) == node.params["k"]))


def test_gradient_through_update_is_the_frozen_set_adjoint():
    """``jax.grad`` through ``update`` equals ``jax.grad`` through
    ``solve_frozen`` with the mask held fixed: nothing leaks through the
    selection."""
    node = PoissonSineTopKNode(theta=0.42)
    s = node.initial_state()
    th0 = jnp.asarray(0.42)
    mask = node.compute_active_set(s, {**node.params, "theta": th0})

    def J_update(th):
        out = node.update(s, {}, 1.0, params={"theta": th})
        return node.objective(out, {**node.params, "theta": th})

    def J_frozen(th):
        p = {**node.params, "theta": th}
        c = jnp.where(mask, node.solve_frozen(s, mask, p)["c"], 0.0)
        return node.objective({"c": c, "mask": mask}, p)

    assert jnp.allclose(jax.grad(J_update)(th0), jax.grad(J_frozen)(th0), rtol=1e-10)


def test_gradient_through_a_scan_is_finite_and_consistent():
    """Steady problem: the gradient through n steps equals the one-step
    gradient (each step re-solves from the parameters alone)."""
    node = MaskedDenseNode(blindness_gate=False)
    s0 = node.initial_state()

    def J(th, n):
        def body(st, _):
            return node.update(st, {}, 1.0, params={"theta": th}), None
        final, _ = jax.lax.scan(body, s0, None, length=n)
        return node.objective(final, {})

    g1 = jax.grad(J)(jnp.asarray(0.3), 1)
    g4 = jax.grad(J)(jnp.asarray(0.3), 4)
    assert bool(jnp.isfinite(g1)) and float(g1) != 0.0
    assert jnp.allclose(g1, g4, rtol=1e-10)
