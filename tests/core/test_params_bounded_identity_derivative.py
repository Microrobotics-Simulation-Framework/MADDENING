"""A bounded identity leaf (``transform=None``) on its bound has the derivative
of moving it into its range.

``ParamSpec.to_constrained`` clips such a leaf, and ``jnp.clip`` is
``minimum(maximum(u, lo), hi)``, whose tie rule gives 0.5 at ``u == lo``.
So the gradient of anything with respect to a parameter sitting exactly on
its bound -- a spring with ``damping = 0.0`` -- was half the derivative into
the range, the only direction it can move, and an FIM column there was a
quarter of its size.  (audit_040_p4_4/fmu-sysid/repro_fit_gradients_and_contracts.py,
section A.)  Off the bounds nothing may change, bit for bit.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import observations_from_history, windowed_loss


def _derivative(spec, u):
    return float(jax.grad(lambda x: spec.to_constrained(x))(jnp.float32(u)))


@pytest.mark.parametrize("bounds, u, expected", [
    ((0.0, 2.0), 0.0, 1.0),         # on the lower bound: into the range
    ((0.0, 2.0), 2.0, 1.0),         # on the upper bound: into the range
    ((0.0, 2.0), 1.0, 1.0),         # inside
    ((0.0, 2.0), -0.5, 0.0),        # beyond a bound: clipped, flat
    ((0.0, 2.0), 2.5, 0.0),
    ((0.0, None), 0.0, 1.0),        # one-sided bounds
    ((None, 3.0), 3.0, 1.0),
    ((-1.5, float("inf")), -1.5, 1.0),   # an infinite side means no bound
    ((-1.5, float("inf")), 1e30, 1.0),
])
def test_the_derivative_at_and_around_a_bound(bounds, u, expected):
    spec = ParamSpec(bounds=bounds)
    assert _derivative(spec, u) == expected
    # Forward mode (what fim and fit_lm differentiate with) agrees.
    _, tangent = jax.jvp(spec.to_constrained, (jnp.float32(u),), (jnp.float32(1.0),))
    assert float(tangent) == expected


def test_off_its_bounds_the_clip_is_jnp_clip_bit_for_bit():
    """Values everywhere, and gradients everywhere but on a bound, are what
    ``jnp.clip`` gives -- the change is confined to the tie."""
    lo, hi = -0.75, 1.25
    spec = ParamSpec(bounds=(lo, hi))
    rng = np.random.default_rng(1729)
    u = jnp.asarray(np.concatenate([rng.uniform(-3, 3, 4000), [lo, hi]]), jnp.float32)
    weights = jnp.asarray(rng.normal(size=u.shape), jnp.float32)

    def ours(x):
        return jnp.sum(weights * jnp.sin(spec.to_constrained(x)))

    def reference(x):
        return jnp.sum(weights * jnp.sin(jnp.clip(x, lo, hi)))

    np.testing.assert_array_equal(np.asarray(spec.to_constrained(u)),
                                  np.asarray(jnp.clip(u, lo, hi)))
    g, g_ref = np.asarray(jax.grad(ours)(u)), np.asarray(jax.grad(reference)(u))
    on_bound = (np.asarray(u) == lo) | (np.asarray(u) == hi)
    assert on_bound.sum() == 2
    np.testing.assert_array_equal(g[~on_bound], g_ref[~on_bound])
    # On the bound: the reference splits the tie, ours does not.
    np.testing.assert_allclose(g[on_bound], 2.0 * g_ref[on_bound], rtol=1e-6)
    # Under jit too, and the second derivative is still zero.
    np.testing.assert_array_equal(np.asarray(jax.jit(jax.grad(ours))(u)), g)
    curvature = jax.grad(lambda x: jax.grad(lambda y: spec.to_constrained(y))(x))
    assert float(curvature(jnp.float32(0.3))) == 0.0


def test_an_integer_leaf_keeps_its_dtype_and_its_clip():
    spec = ParamSpec(bounds=(0, 5))
    out = spec.to_constrained(jnp.asarray([-2, 3, 9], jnp.int32))
    assert out.dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(out), [0, 3, 5])


# ---------------------------------------------------------------------------
# Through a graph: the spring held at damping = 0.0, its lower bound
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def spring_at_its_damping_bound():
    def graph(**kw):
        p = dict(stiffness=30.0, damping=0.0, mass=1.0, rest_length=1.0,
                 initial_position=0.2, initial_velocity=0.0)
        p.update(kw)
        gm = GraphManager()
        gm.add_node(SpringDamperNode(name="s", timestep=0.01, **p))
        gm.compile()
        return gm

    truth = graph(stiffness=40.0, damping=3.0)
    s0 = truth._user_state(truth._state)  # noqa: SLF001
    obs = observations_from_history(s0, truth.run_scan_with_history(60)[1])
    gm = graph()
    spec = gm.param_specs()["nodes"]["s"]["damping"]
    assert spec.transform is None and spec.bounds == (0.0, None)
    start = gm.params
    flat_u, unravel = ravel_pytree(gm.unconstrain(start))
    names = [jax.tree_util.keystr(p) for p, _ in
             jax.tree_util.tree_flatten_with_path(start)[0]]
    j = names.index("['nodes']['s']['damping']")
    assert float(flat_u[j]) == 0.0

    def objective(u_damping):
        p = gm.constrain(unravel(flat_u.at[j].set(u_damping)))
        return windowed_loss(gm, p, obs, obs_fn=lambda h: h["s"]["position"], window=20)

    return gm, obs, objective, graph


def test_the_gradient_at_the_bound_is_the_derivative_into_the_range(spring_at_its_damping_bound):
    """Autodiff against one-sided differences into the range, at two step
    sizes (first order, so they bracket the derivative): the audit's
    numbers were -2.64e-2 by autodiff against -5.24e-2 by difference."""
    _, _, objective, _ = spring_at_its_damping_bound
    g = float(jax.grad(objective)(jnp.float32(0.0)))
    f0 = float(objective(jnp.float32(0.0)))
    one_sided = [(float(objective(jnp.float32(h))) - f0) / h for h in (1e-2, 5e-3)]
    # Richardson: the first-order error halves with the step.
    extrapolated = 2.0 * one_sided[1] - one_sided[0]
    assert g == pytest.approx(extrapolated, rel=5e-3)
    # And it is not the tie-split half any more.
    central = (float(objective(jnp.float32(1e-2))) - float(objective(jnp.float32(-1e-2)))) / 2e-2
    assert g == pytest.approx(2.0 * central, rel=2e-2)


def test_the_jacobian_fit_lm_forms_at_the_bound_is_full_sized(spring_at_its_damping_bound):
    """``fit_lm`` differentiates the residual in its own coordinates, through
    ``constrain``, by forward mode; at the bound that column was half the
    derivative into the range, and the Gauss-Newton entry it forms a
    quarter.  (``fim`` itself differentiates the physical parameters, where
    nothing is clipped, so it was never affected.)"""
    gm, obs, _, graph = spring_at_its_damping_bound
    start = gm.params
    flat_u, unravel = ravel_pytree(gm.unconstrain(start))
    names = [jax.tree_util.keystr(p) for p, _ in
             jax.tree_util.tree_flatten_with_path(start)[0]]
    j = names.index("['nodes']['s']['damping']")

    def residual(u_damping):
        p = gm.constrain(unravel(flat_u.at[j].set(u_damping)))
        # A fresh graph per evaluation: ``run_scan*`` stores its final state.
        return graph().run_scan_with_history(60, params=p)[1]["s"]["position"] - obs["s"]["position"][1:]

    column = np.asarray(jax.jacfwd(residual)(jnp.float32(0.0)), np.float64)
    base = np.asarray(residual(jnp.float32(0.0)), np.float64)
    one_sided = [(np.asarray(residual(jnp.float32(h)), np.float64) - base) / h
                 for h in (1e-2, 5e-3)]
    extrapolated = 2.0 * one_sided[1] - one_sided[0]
    assert column @ column == pytest.approx(extrapolated @ extrapolated, rel=1e-2)
    np.testing.assert_allclose(column, extrapolated, rtol=0, atol=2e-2 * np.abs(extrapolated).max())
