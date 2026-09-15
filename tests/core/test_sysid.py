"""maddening.sysid: windowed teacher-forced loss and Fisher information."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fim, observations_from_history, windowed_loss

K_TRUE, C_TRUE = 30.0, 2.0
N_STEPS, WINDOW = 200, 20


def _spring_gm(k=K_TRUE, c=C_TRUE, coupled=False):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=k, damping=c, mass=1.0,
                                 rest_length=1.0, initial_position=0.5))
    if coupled:
        gm.add_node(SpringDamperNode("t", 0.01, stiffness=k, damping=c, mass=1.0,
                                     rest_length=1.0, initial_position=3.0))
        gm.add_edge("s", "t", "position", "anchor_position")
        gm.add_edge("t", "s", "position", "anchor_position")
        gm.add_coupling_group(["s", "t"], max_iterations=20, tolerance=1e-6)
    gm.compile()
    return gm


def _observations(gm):
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(N_STEPS)
    return observations_from_history(init, hist)


@pytest.fixture(scope="module")
def spring():
    gm = _spring_gm()
    obs = _observations(gm)
    return gm, obs


def _loss_fn(gm, obs, **kw):
    return jax.jit(lambda p: windowed_loss(
        gm, p, obs, obs_fn=lambda h: h["s"]["position"], window=WINDOW, **kw,
    ))


def _with_k(params, k):
    p = jax.tree.map(lambda x: x, params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(k, dtype=jnp.float32)
    return p


def test_loss_zero_at_truth_positive_elsewhere(spring):
    gm, obs = spring
    loss = _loss_fn(gm, obs)
    assert float(loss(gm.params)) < 1e-9
    assert float(loss(_with_k(gm.params, 1.2 * K_TRUE))) > 1e-4


def test_loss_gradient_finite_and_nonzero(spring):
    gm, obs = spring
    loss = _loss_fn(gm, obs)
    g = jax.grad(loss)(_with_k(gm.params, 1.2 * K_TRUE))["nodes"]["s"]["stiffness"]
    assert bool(jnp.isfinite(g)) and float(g) > 0.0  # k too high -> increase loss


def test_recovers_stiffness_from_20_percent_perturbation(spring):
    gm, obs = spring
    loss = _loss_fn(gm, obs)
    grad = jax.jit(jax.grad(loss))
    k = jnp.asarray(1.2 * K_TRUE, dtype=jnp.float32)
    m = v = jnp.zeros(())
    b1, b2, lr = 0.9, 0.999, 0.5
    for i in range(1, 301):
        g = grad(_with_k(gm.params, k))["nodes"]["s"]["stiffness"]
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g ** 2
        k = k - lr * (m / (1 - b1 ** i)) / (jnp.sqrt(v / (1 - b2 ** i)) + 1e-8)
    assert abs(float(k) - K_TRUE) / K_TRUE < 0.05, float(k)


def test_windows_must_tile_the_observations(spring):
    gm, obs = spring
    with pytest.raises(ValueError, match="must divide"):
        windowed_loss(gm, gm.params, obs, obs_fn=lambda h: h["s"]["position"], window=7)


def test_mask_unconverged_through_coupled_group():
    gm = _spring_gm(coupled=True)
    obs = _observations(gm)
    loss = jax.jit(lambda p: windowed_loss(
        gm, p, obs, obs_fn=lambda h: (h["s"]["position"], h["t"]["position"]),
        window=WINDOW, mask_unconverged=True,
    ))
    assert float(loss(gm.params)) < 1e-6
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(1.3 * K_TRUE, dtype=jnp.float32)
    g = jax.grad(loss)(p)["nodes"]["s"]["stiffness"]
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


def _residual_fn(gm, obs, names):
    """Residual over a sub-pytree of the spring's params."""
    step_fn = gm._build_step_fn()
    ext = gm._default_external_inputs()
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs["s"]["position"][1:]

    def residual(sub):
        p = jax.tree.map(lambda x: x, gm.params)
        for n in names:
            p["nodes"]["s"][n] = sub[n]

        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["s"]["position"]

        _, pos = jax.lax.scan(body, init, None, length=N_STEPS)
        return pos - truth

    return residual


def test_fim_identifiable_pair_has_finite_condition_number(spring):
    gm, obs = spring
    names = ("stiffness", "damping")
    sub = {n: gm.params["nodes"]["s"][n] for n in names}
    rep = fim(_residual_fn(gm, obs, names), sub)
    assert rep.param_names == ("['damping']", "['stiffness']")
    assert np.isfinite(rep.cond) and rep.cond < 1e6, rep.cond
    assert bool(jnp.all(jnp.isfinite(rep.crb)))


def test_fim_flags_unidentifiable_scale_direction(spring):
    """With position-only data, k, c, m enter only as k/m and c/m: scaling
    all three together is invisible.  In relative coordinates that is the
    direction (1, 1, 1)/sqrt(3), and its eigenvalue must be ~0."""
    gm, obs = spring
    names = ("stiffness", "damping", "mass")
    sub = {n: gm.params["nodes"]["s"][n] for n in names}
    rep = fim(_residual_fn(gm, obs, names), sub)
    ratio = float(rep.eigvals[0] / rep.eigvals[-1])
    assert ratio < 1e-4, ratio
    v = np.asarray(rep.eigvecs[:, 0])
    assert abs(abs(v @ np.ones(3) / np.sqrt(3.0))) > 0.99, v
    assert rep.cond > 1e4
