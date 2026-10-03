"""``windowed_loss`` replays a coupling group's warm starts across windows.

A coupling group with a ``predictor`` or ``acceleration="iqn-imvj"`` carries
state that the user state does not hold -- its predictor history
(``coupling_<g>_pred_*``) and its IQN-IMVJ secant matrices
(``coupling_<g>_V`` / ``_W``) -- and with a finite ``max_iterations`` that
state changes what a step returns.  0.4.0 development builds rebuilt every
window start as the measured user state plus ``zeros_like(_meta)``, so each
window restarted the warm starts cold while the record had been made with
them warm: with a quadratic predictor at ``max_iterations=1`` the loss at the
generating parameters was 1.7e-2 over 10-sample windows, and a fit started
at the truth walked 6.4% away, with no warning
(audit_040_p4_8/fmu-sysid/repro_windowed_predictor.py).

Window ``w`` now starts from the warm starts window ``w - 1`` ended with,
with their gradient stopped, and window 0 from the cold ones ``compile()``
leaves.  What is pinned here:

* the loss, and its gradient, are exactly zero at the generating parameters
  for every predictor, for IQN-IMVJ, for ``solver="fori"``, a multi-rate
  graph, ``sample_every > 1`` and multiple shooting;
* a fit started at the truth stays there;
* the inherited warm starts carry no gradient -- each window's gradient is
  its own scan's -- and the loss is the sum of the windows' own losses,
  each from the warm starts the previous one left;
* a window ``mask_unconverged`` drops hands the next one cold warm starts,
  not its own diverged ones;
* a record that began after the graph had stepped (``start_step > 0``)
  warns that it cannot be replayed exactly.
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    fit,
    init_window_states,
    observations_from_history,
    windowed_loss,
)

N_STEPS = 40


def _springs(**group):
    """Two springs anchored on each other, solved as one coupling group.
    Damped past ``k * dt`` each, so the converged pair is stable (the
    compile-time check of MADD-ANO-098 stays quiet)."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01, initial_position=0.5,
                                 stiffness=100.0, damping=1.5))
    gm.add_node(SpringDamperNode("b", 0.01, initial_position=0.2,
                                 stiffness=80.0, damping=1.2))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], **group)
    gm.compile()
    return gm, (lambda h: {"a": h["a"]["position"], "b": h["b"]["position"]}), None


def _springs_multirate(**group):
    """The spring pair beside a ball that steps every other base step."""
    gm, obs_fn, _ = _springs(**group)
    gm.add_node(BallNode(name="ball", timestep=0.02, initial_position=1.0,
                         elasticity=0.7))
    gm.compile()
    return gm, obs_fn, None


def _rods(**group):
    """Two heat rods coupled through their boundary temperatures."""
    gm = GraphManager()
    gm.add_node(HeatNode("r1", 0.001, n_cells=8, thermal_diffusivity=1e-2, length=0.1))
    gm.add_node(HeatNode("r2", 0.001, n_cells=8, thermal_diffusivity=2e-2, length=0.1))
    gm.add_edge("r1", "r2", "temperature", "left_temperature", transform=lambda T: T[-1])
    gm.add_edge("r2", "r1", "temperature", "right_temperature", transform=lambda T: T[0])
    gm.add_external_input("r1", "left_temperature")
    gm.add_coupling_group(["r1", "r2"], **group)
    gm.compile()
    return gm, (lambda h: h["r2"]["temperature"]), {"r1": {"left_temperature": jnp.float32(100.0)}}


#: Each warm-start configuration.  The 0.4.0 development builds' cold
#: restart gave, at the truth over 10-sample windows on jaxlib 0.11.0,
#: 1.2e-2 for the quadratic predictor at one pass, 6.0e-11 for the linear
#: one at two, and 8e-17 to 2e-21 for the IQN-IMVJ rods (which converge to
#: nearly the same state either way -- tiny, and still not zero).
_CONFIGS = {
    "quadratic predictor, one pass": (
        _springs, dict(max_iterations=1, tolerance=1e-14, predictor="quadratic")),
    "linear predictor, two passes": (
        _springs, dict(max_iterations=2, tolerance=1e-14, predictor="linear")),
    "quadratic predictor, solver fori": (
        _springs, dict(max_iterations=1, tolerance=1e-14, predictor="quadratic",
                       solver="fori")),
    "quadratic predictor, multi-rate graph": (
        _springs_multirate, dict(max_iterations=1, tolerance=1e-14, predictor="quadratic")),
    "iqn-imvj warm start": (
        _rods, dict(max_iterations=3, tolerance=1e-14, acceleration="iqn-imvj",
                    jacobian_reuse=2)),
    "iqn-imvj and quadratic predictor": (
        _rods, dict(max_iterations=3, tolerance=1e-14, acceleration="iqn-imvj",
                    jacobian_reuse=2, predictor="quadratic")),
}


def _build(name):
    build, group = _CONFIGS[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)    # solver="fori"
        return build(**group)


def _record(gm, ext, n_steps=N_STEPS):
    """A record taken from ``compile()``: the warm starts begin cold."""
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(n_steps, external_inputs=ext)
    return observations_from_history(init, hist)


def _warm_slots(gm):
    return sorted(k for k in gm._state["_meta"]  # noqa: SLF001
                  if k.endswith(("_V", "_W")) or "_pred_" in k)


@pytest.fixture(scope="module", params=sorted(_CONFIGS))
def recorded(request):
    gm, obs_fn, ext = _build(request.param)
    obs = _record(gm, ext)
    return request.param, gm, obs_fn, ext, obs


def _largest(tree):
    return max(float(jnp.max(jnp.abs(leaf))) for leaf in jax.tree.leaves(tree))


def test_the_loss_is_exactly_zero_and_stationary_at_the_truth(recorded):
    """Each window starts from the warm starts the record had there, so it
    reproduces its slice of the record bit for bit, for one-sample and
    ten-sample windows.  The gradient there is zero up to the rounding of the
    differentiated forward pass (exactly zero for the springs; the IFT
    solve's forward under ``jax.grad`` is a separate program, and for the
    rods it lands a few ulps off the record) -- against the cold restart's
    0.35 for the quadratic predictor."""
    name, gm, obs_fn, ext, obs = recorded
    assert _warm_slots(gm), f"{name}: the fixture has no warm start to replay"
    kw = dict(obs_fn=obs_fn, external_inputs=ext)
    if "multi-rate" in name:
        kw["start_step"] = 0
    for window in (1, 10):
        loss = windowed_loss(gm, gm.params, obs, window=window, **kw)
        assert float(loss) == 0.0, (name, window, float(loss))

    grad = jax.jit(jax.grad(lambda q: windowed_loss(gm, q, obs, window=10, **kw)))
    off = jax.tree.map(lambda x: x * 1.01 if jnp.issubdtype(jnp.result_type(x), jnp.floating)
                       else x, gm.params)
    at_truth, away = _largest(grad(gm.params)), _largest(grad(off))
    assert away > 0.0, name
    assert at_truth <= 1e-4 * away, (name, at_truth, away)
    if _CONFIGS[name][0] is not _rods:
        assert at_truth == 0.0, (name, at_truth)


@pytest.mark.parametrize("name", ["quadratic predictor, one pass", "iqn-imvj warm start"])
def test_multiple_shooting_and_sparse_samples_replay_exactly(name):
    """The replay runs through ``window_states`` (each window from its free
    start, with the warm starts the previous window left) and through
    ``sample_every`` (every base step of a sample carries them on)."""
    gm, obs_fn, ext = _build(name)
    obs = _record(gm, ext)
    ws = init_window_states(obs, 10)
    shooting = windowed_loss(gm, gm.params, obs, obs_fn=obs_fn, window=10,
                             external_inputs=ext, window_states=ws,
                             continuity_weight=1.0)
    assert float(shooting) == 0.0, float(shooting)
    sparse = jax.tree.map(lambda x: x[::2], obs)
    every_other = windowed_loss(gm, gm.params, sparse, obs_fn=obs_fn, window=5,
                                sample_every=2, external_inputs=ext)
    assert float(every_other) == 0.0, float(every_other)


def test_a_fit_started_at_the_truth_stays_there():
    """The cold restart moved the loss's minimum: Adam started at the
    generating stiffness walked to +6.4% and returned that.  Started there
    now, the fit returns it: the loss there is the floor of the ``log``
    round trip (one ulp of the stiffness), and every step climbs."""
    gm, obs_fn, ext = _build("quadratic predictor, one pass")
    obs = _record(gm, ext)
    truth = jax.tree.map(lambda x: x, gm.params)
    for node in ("a", "b"):
        for key in ("stiffness", "damping", "mass", "rest_length"):
            if (node, key) != ("a", "stiffness"):
                gm.set_param_spec(node, key, ParamSpec(trainable=False))
    loss = jax.jit(lambda p: windowed_loss(gm, p, obs, obs_fn=obs_fn, window=10))
    assert float(loss(truth)) == 0.0
    res = fit(gm, loss, params=truth, n_iter=60, lr=0.01, notify_every=0)
    got = float(res.params["nodes"]["a"]["stiffness"])
    assert abs(got / 100.0 - 1.0) < 1e-6, got
    assert res.best_loss < 1e-10, res.best_loss


def _manual_window(gm, p, obs, start, window, meta):
    """``(loss, final _meta)`` of one teacher-forced window, run by hand
    from the graph's step function: the oracle for what a window is."""
    step_fn = gm._build_step_fn()                     # noqa: SLF001
    ext = gm._default_external_inputs()               # noqa: SLF001
    state = {n: {k: v[start] for k, v in f.items()} for n, f in obs.items()}
    state["_meta"] = meta

    def body(s, _):
        s = step_fn(s, ext, p)
        return s, {"a": s["a"]["position"], "b": s["b"]["position"]}

    final, sim = jax.lax.scan(body, state, None, length=window)
    loss = sum(jnp.sum((sim[n] - obs[n]["position"][start + 1:start + 1 + window]) ** 2)
               for n in ("a", "b"))
    return loss, final["_meta"]


def test_the_inherited_warm_starts_carry_no_gradient():
    """The loss is the sum of the windows' own losses, window 1 starting
    from the warm starts window 0 ended with -- and its gradient is the sum
    of the windows' gradients with those warm starts held as constants.  A
    replay that let the gradient through them would compound it across
    windows, which is what teacher forcing exists to prevent; at this
    off-truth point its extra term moves the gradient by far more than the
    tolerance."""
    gm, obs_fn, ext = _build("quadratic predictor, one pass")
    obs = _record(gm, ext, n_steps=20)
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["a"]["stiffness"] = jnp.float32(112.0)
    cold = jax.tree.map(jnp.zeros_like, gm._state["_meta"])   # noqa: SLF001

    loss0, meta1 = _manual_window(gm, p, obs, 0, 10, cold)
    meta1 = jax.tree.map(jax.lax.stop_gradient, meta1)
    loss1, _ = _manual_window(gm, p, obs, 10, 10, meta1)
    got = windowed_loss(gm, p, obs, obs_fn=obs_fn, window=10)
    np.testing.assert_allclose(float(got), float(loss0 + loss1), rtol=1e-5)

    def oracle(q):
        l0, m1 = _manual_window(gm, q, obs, 0, 10, cold)
        l1, _ = _manual_window(gm, q, obs, 10, 10,
                               jax.tree.map(jax.lax.stop_gradient, m1))
        return l0 + l1

    want = jax.jit(jax.grad(oracle))(p)["nodes"]["a"]["stiffness"]
    have = jax.jit(jax.grad(lambda q: windowed_loss(gm, q, obs, obs_fn=obs_fn, window=10)))(
        p)["nodes"]["a"]["stiffness"]
    np.testing.assert_allclose(float(have), float(want), rtol=1e-6)

    def leaky(q):
        l0, m1 = _manual_window(gm, q, obs, 0, 10, cold)
        l1, _ = _manual_window(gm, q, obs, 10, 10, m1)
        return l0 + l1

    # Measured 4.9e-4 relative on jaxlib 0.11.0: a hundred times the
    # tolerance above, so a leaked gradient cannot pass for a stopped one.
    through = jax.jit(jax.grad(leaky))(p)["nodes"]["a"]["stiffness"]
    assert abs(float(through) - float(want)) > 1e-4 * abs(float(want)), (
        "the fixture cannot tell a stopped gradient from a leaked one",
        float(through), float(want))


def test_a_window_the_mask_drops_hands_on_cold_warm_starts():
    """A window that diverged ends with non-finite warm starts.  Handed on,
    they made the next window diverge too, and the next: every later window
    was masked and the loss read 0.0 -- "nothing to fit" -- at parameters
    10% off.  The next window starts cold instead, so only the window that
    diverged is dropped."""
    gm, obs_fn, ext = _springs(max_iterations=30, tolerance=1e-6,
                               predictor="quadratic")
    obs = _record(gm, ext)
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["a"]["stiffness"] = jnp.float32(110.0)
    kw = dict(obs_fn=obs_fn, window=10, mask_unconverged=True)
    clean = float(windowed_loss(gm, p, obs, **kw))
    first = float(windowed_loss(gm, p, jax.tree.map(lambda x: x[:11], obs), **kw))
    assert clean > first > 0.0, (clean, first)
    poisoned = jax.tree.map(lambda x: x, obs)
    poisoned["a"]["velocity"] = poisoned["a"]["velocity"].at[0].set(jnp.nan)
    dropped = float(windowed_loss(gm, p, poisoned, **kw))
    assert np.isfinite(dropped), dropped
    # Windows 1-3 still count; they start cold rather than warm, which a
    # group converged to 1e-6 barely notices.
    np.testing.assert_allclose(dropped, clean - first, rtol=1e-3)


def test_a_record_that_began_after_the_graph_stepped_warns():
    """Its warm starts at sample 0 are not known, so window 0 starts them
    cold and the replay cannot be exact: said, as ``start_step=None`` on a
    multi-rate graph says it assumes 0."""
    gm, obs_fn, ext = _build("quadratic predictor, one pass")
    obs = _record(gm, ext)
    with pytest.warns(UserWarning, match="warm starts .* are not known"):
        windowed_loss(gm, gm.params, obs, obs_fn=obs_fn, window=10, start_step=7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        for start_step in (None, 0):
            windowed_loss(gm, gm.params, obs, obs_fn=obs_fn, window=10,
                          start_step=start_step)


def test_a_graph_without_warm_starts_has_nothing_to_warn_about():
    """``start_step`` moves no schedule on a single-rate graph, and a group
    with neither a predictor nor IQN-IMVJ carries no warm start: no
    warning, and the loss at the truth is zero as it always was."""
    gm, obs_fn, ext = _springs(max_iterations=1, tolerance=1e-14)
    obs = _record(gm, ext)
    assert not _warm_slots(gm)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        loss = windowed_loss(gm, gm.params, obs, obs_fn=obs_fn, window=10,
                             start_step=7)
    assert float(loss) == 0.0
