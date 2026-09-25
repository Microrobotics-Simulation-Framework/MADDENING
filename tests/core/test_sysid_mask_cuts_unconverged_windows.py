"""``windowed_loss(mask_unconverged=True)`` cuts a window out of the loss and its gradient.

Two invariants:

* **A masked window contributes exactly nothing, even when it diverged.**
  The mask used to multiply the window's loss by 0.  A window whose
  coupling diverged holds inf / NaN and ``0 * inf`` is NaN: the forward
  loss came out right only because XLA's CPU backend rewrote the product
  into a select, and the backward pass carried the zero cotangent through
  the window's non-finite intermediates and made the gradient NaN.
* **The mask's criterion is the error estimate, not the raw residual.**
  A group stops on ``estimated_error(residual, amplification) <= threshold``
  and the report calls a step converged by the same test.  The one test
  that compared the mask with the report ran on a group whose ratio was
  rejected, where the estimate *is* the residual, so a mask reading the
  raw residual passed every mask test there was.  Here the ratio is
  usable and the step's residual is under the threshold while its
  estimate is over it.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening import sysid
from maddening.core.coupling.acceleration import convergence_criterion
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


# ---------------------------------------------------------------------------
# A window that diverges
# ---------------------------------------------------------------------------


class Square(SimulationNode):
    """``x <- c + g * u**2``: a fixed point near 0.6 from a small start;
    from a start of 30 the coupling iteration blows up quadratically.

    The node ignores its own state, so a window's start state is only the
    coupling's initial guess.
    """

    def __init__(self, name, c, g):
        super().__init__(name=name, timestep=1.0, c=c, g=g)

    def initial_state(self):
        return {"x": jnp.array(0.0, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.array(0.0, jnp.float32))}

    def update(self, state, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["c"] + p["g"] * bi["u"] ** 2}


def _square_pair():
    gm = GraphManager()
    gm.add_node(Square("a", 0.5, 0.3))
    gm.add_node(Square("b", 0.4, 0.3))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=30, tolerance=1e-6)
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def diverging():
    """Six samples, five one-step windows; window 2 starts at 30 and diverges.

    Windows 0-1 and 3-4 are each three samples long, so the two halves
    the masked loss is compared against share one compiled program.
    """
    gm = _square_pair()
    gm.step()
    xa, xb = float(gm._state["a"]["x"]), float(gm._state["b"]["x"])
    gm.reset_state()
    obs = {"a": {"x": jnp.full((6,), xa, jnp.float32).at[2].set(30.0)},
           "b": {"x": jnp.full((6,), xb, jnp.float32).at[2].set(30.0)}}
    return gm, obs


def _loss(gm, mask=True):
    """``(params, observations) -> loss``, one window per sample."""
    def f(p, obs):
        return sysid.windowed_loss(gm, p, obs, obs_fn=lambda s: s["a"]["x"], window=1,
                                   mask_unconverged=mask)
    return f


def _samples(obs, lo, hi):
    return {n: {"x": obs[n]["x"][lo:hi]} for n in obs}


def test_the_diverging_window_is_non_finite_and_unconverged(diverging):
    """The precondition, checked rather than assumed."""
    gm, obs = diverging
    assert not np.isfinite(float(_loss(gm, mask=False)(gm.params, obs)))


def test_a_diverged_window_leaves_a_finite_gradient_of_the_other_windows(diverging):
    """Masked over all five windows = windows 0-1 plus windows 3-4, value and gradient."""
    gm, obs = diverging
    value_and_grad = jax.jit(jax.value_and_grad(_loss(gm)))
    loss, grad = value_and_grad(gm.params, obs)
    halves = [value_and_grad(gm.params, _samples(obs, lo, lo + 3)) for lo in (0, 3)]
    want_loss = sum(float(v) for v, _ in halves)
    want_grad = jax.tree.map(jnp.add, halves[0][1], halves[1][1])
    assert np.isfinite(float(loss)) and float(loss) == pytest.approx(want_loss, rel=1e-6)
    for got, want in zip(jax.tree.leaves(grad), jax.tree.leaves(want_grad)):
        assert np.all(np.isfinite(np.asarray(got))), grad
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-6)


def test_multiple_shooting_drops_a_diverged_window_and_its_continuity_term(diverging):
    """Under multiple shooting the diverged window's end ties nothing either.

    Its continuity penalty would be ``inf``; it goes with the window, so the
    loss and the gradient -- with respect to the parameters and the free
    window starts -- stay finite.
    """
    gm, obs = diverging
    ws = sysid.init_window_states(obs, 1)

    def loss(p, w):
        return sysid.windowed_loss(gm, p, obs, obs_fn=lambda s: s["a"]["x"], window=1,
                                   mask_unconverged=True, window_states=w,
                                   continuity_weight=1.0)

    value, (gp, gw) = jax.value_and_grad(loss, argnums=(0, 1))(gm.params, ws)
    assert np.isfinite(float(value))
    for leaf in jax.tree.leaves((gp, gw)):
        assert np.all(np.isfinite(np.asarray(leaf))), (gp, gw)


# ---------------------------------------------------------------------------
# The mask reads the error estimate
# ---------------------------------------------------------------------------


class Affine(SimulationNode):
    def __init__(self, name, gain, bias):
        super().__init__(name=name, timestep=0.01, gain=gain, bias=bias)

    def initial_state(self):
        return {"x": jnp.float32(0.0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt):
        u = boundary_inputs.get("u", jnp.float32(0.0))
        return {"x": self.params["gain"] * u + self.params["bias"]}


#: A Gauss-Seidel pair contracting at about 0.8 per pass (amplification
#: near 5), stopped at a cap of 8 passes from a start about 5 from its
#: fixed point: the residual it reports is near 0.045 and the estimate
#: about five times that, so a tolerance of 0.1 sits between them.
TOLERANCE = 0.1
CAP = 8
#: The pair's fixed point: ``a = 0.9 b + 1``, ``b = 0.9 a``.
FIXED_POINT = {"a": 1.0 / (1.0 - 0.81), "b": 0.9 / (1.0 - 0.81)}


def _affine_pair():
    gm = GraphManager()
    gm.add_node(Affine("a", 0.9, 1.0))
    gm.add_node(Affine("b", 0.9, 0.0))
    gm.add_edge("b", "a", "x", "u")
    gm.add_edge("a", "b", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=CAP, tolerance=TOLERANCE)
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def estimate_over_threshold():
    """One step from rest: residual under the threshold, estimate over it."""
    gm = _affine_pair()
    init = {n: dict(gm.get_node_state(n)) for n in ("a", "b")}
    gm.step()
    report = dict(gm.coupling_diagnostics()["a+b"])
    after = {n: dict(gm.get_node_state(n)) for n in ("a", "b")}
    threshold, _ = convergence_criterion(gm._coupling_groups[0])
    # The configuration that tells the two criteria apart, checked.
    assert report["ratio_usable"] is True, report
    assert report["iterations"] == CAP, report
    assert report["residual"] <= threshold < report["error_estimate"], report
    assert report["converged"] is False, report
    # Sample 1 is window 0's target and window 1's start: next to the fixed
    # point (the nodes ignore their own state, so a start is only the
    # solve's initial guess), so window 1 converges and must be kept, and
    # away from where window 0 stopped, so window 0 has a loss to drop.
    start1 = {n: {"x": jnp.float32(FIXED_POINT[n] + 0.01)} for n in ("a", "b")}
    assert abs(float(after["a"]["x"]) - float(start1["a"]["x"])) > 0.05
    gm2 = _affine_pair()
    for n in ("a", "b"):
        gm2.set_node_state(n, start1[n])
    gm2.step()
    assert gm2.coupling_diagnostics()["a+b"]["converged"] is True
    later = {n: dict(gm2.get_node_state(n)) for n in ("a", "b")}
    return {n: {"x": jnp.stack([init[n]["x"], start1[n]["x"], later[n]["x"] + 1.0])}
            for n in ("a", "b")}


def test_the_mask_drops_the_window_the_report_calls_unconverged(estimate_over_threshold):
    obs = estimate_over_threshold
    gm = _affine_pair()
    kw = dict(obs_fn=lambda s: s["a"]["x"], window=1)
    unmasked = float(sysid.windowed_loss(gm, gm.params, obs, mask_unconverged=False, **kw))
    masked = float(sysid.windowed_loss(gm, gm.params, obs, mask_unconverged=True, **kw))
    second = {n: {"x": obs[n]["x"][1:]} for n in obs}
    kept = float(sysid.windowed_loss(gm, gm.params, second, mask_unconverged=True, **kw))
    assert kept > 0.0 and unmasked > kept
    # Window 0 (residual under the threshold, estimate over it) is dropped,
    # window 1 is kept: a mask reading the raw residual keeps both.
    assert masked == pytest.approx(kept, rel=1e-6), (masked, kept, unmasked)
