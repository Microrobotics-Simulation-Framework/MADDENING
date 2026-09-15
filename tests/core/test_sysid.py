"""maddening.sysid: windowed teacher-forced loss and Fisher information."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fim, fit, observations_from_history, windowed_loss

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


# ---------------------------------------------------------------------------
# fit / fim under ParamSpec
# ---------------------------------------------------------------------------


def _perturbed(gm, k_factor, c_factor):
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(k_factor * K_TRUE, jnp.float32)
    p["nodes"]["s"]["damping"] = jnp.asarray(c_factor * C_TRUE, jnp.float32)
    return p


def test_fit_recovers_k_c_with_mass_frozen_by_spec(spring):
    gm, obs = spring
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    loss = _loss_fn(gm, obs)
    seen = []
    res = fit(gm, loss, params=_perturbed(gm, 1.5, 2.5), n_iter=300, lr=0.1,
              callback=lambda i, l, p: seen.append(float(p["nodes"]["s"]["stiffness"])))
    s = res.params["nodes"]["s"]
    assert abs(float(s["stiffness"]) - K_TRUE) / K_TRUE < 0.05, float(s["stiffness"])
    assert abs(float(s["damping"]) - C_TRUE) / C_TRUE < 0.10, float(s["damping"])
    assert float(s["mass"]) == 1.0                       # frozen, bit-identical
    assert float(s["initial_position"]) == 0.5           # initial_* never move
    assert res.losses[-1] < 1e-3 * res.losses[0]
    assert res.n_iter == 300 and not res.converged       # tol=0: ran to n_iter
    assert min(seen) > 0.0                               # log transform: never <= 0


def test_fit_mask_overrides_specs_and_tol_stops_early(spring):
    gm, obs = spring
    loss = _loss_fn(gm, obs)
    mask = jax.tree.map(lambda _: False, gm.params)
    mask["nodes"]["s"]["stiffness"] = True
    start = _perturbed(gm, 1.3, 1.0)
    res = fit(gm, loss, params=start, mask=mask, n_iter=400, lr=0.1, tol=1e-6)
    s = res.params["nodes"]["s"]
    assert float(s["damping"]) == float(start["nodes"]["s"]["damping"])
    assert abs(float(s["stiffness"]) - K_TRUE) / K_TRUE < 0.02
    assert res.converged and res.n_iter < 400
    assert res.losses[-1] <= 1e-6


def test_fit_rejects_start_outside_bounds(spring):
    gm, obs = spring
    bad = _perturbed(gm, -1.0, 1.0)      # stiffness < 0 with a log spec
    with pytest.raises(ValueError, match="stiffness"):
        fit(gm, _loss_fn(gm, obs), params=bad, n_iter=1)


def test_fit_raises_on_non_finite_gradient(spring):
    gm, obs = spring
    with pytest.raises(FloatingPointError):
        fit(gm, lambda p: jnp.sqrt(p["nodes"]["s"]["stiffness"] - 30.0), n_iter=3)


def test_fim_mask_restricts_to_selected_leaves(spring):
    """``fim`` over the full pytree with the graph's trainable mask equals
    ``fim`` over the hand-built sub-pytree of the same leaves."""
    gm, obs = spring
    gm._param_spec_overrides.clear()  # noqa: SLF001 - module-scoped fixture
    for key in gm.params["nodes"]["s"]:
        if key not in ("stiffness", "damping"):
            gm.set_param_spec("s", key, ParamSpec(trainable=False))
    step_fn = gm._build_step_fn()
    ext = gm._default_external_inputs()
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs["s"]["position"][1:]

    def residual_full(p):
        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s["s"]["position"]
        return jax.lax.scan(body, init, None, length=N_STEPS)[1] - truth

    rep = fim(residual_full, gm.params, mask=gm.trainable_mask())
    ref = fim(_residual_fn(gm, obs, ("stiffness", "damping")),
              {n: gm.params["nodes"]["s"][n] for n in ("stiffness", "damping")})
    assert rep.fim.shape == (2, 2)
    assert rep.param_names == ("['nodes']['s']['damping']", "['nodes']['s']['stiffness']")
    assert np.allclose(np.asarray(rep.fim), np.asarray(ref.fim), rtol=1e-5)
    with pytest.raises(ValueError, match="selects no parameters"):
        fim(residual_full, gm.params, mask=jax.tree.map(lambda _: False, gm.params))
    gm._param_spec_overrides.clear()  # noqa: SLF001
