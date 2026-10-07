"""``windowed_loss(mask_unconverged=True)`` reads every base step of a window.

The mask drops a window "when any coupling group exited at
``max_iterations`` unconverged *during* it".  Every other mask test uses a
window that is unconverged at its END (a diverged window stays diverged, a
one-step window has one verdict), so a mask that read only the window's
last step passed all of them.

Here the window recovers: a linear Gauss-Seidel pair (contraction 0.81 per
pass, cap 4) restarted far from its fixed point exits its first steps at the
cap, and, because each step starts from the previous step's iterate, the
later steps converge.  The last step's verdict alone says "keep"; the
window's says "drop".
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening import sysid
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

STEPS = 30
FAR = (100.0, -50.0)
# The pair's fixed point: a = 0.9 b + 1, b = 0.9 a + 2.
FIXED = (2.8 / 0.19, 2.0 + 0.9 * 2.8 / 0.19)


class Lin(SimulationNode):
    """``x <- g * u + b``; its own state is only the coupling's first guess."""

    def __init__(self, name, g, b):
        super().__init__(name, 0.01, g=jnp.float32(g), b=jnp.float32(b))

    def initial_state(self):
        return {"x": jnp.zeros((), jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.zeros((), jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * boundary_inputs["u"] + p["b"]}


def _pair(cap):
    gm = GraphManager()
    gm.add_node(Lin("a", 0.9, 1.0))
    gm.add_node(Lin("b", 0.9, 2.0))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=cap, tolerance=1e-6)
    gm.compile()
    return gm


def _verdicts(gm, start, steps):
    gm.reset_state()
    gm.set_node_state("a", {"x": jnp.float32(start[0])})
    gm.set_node_state("b", {"x": jnp.float32(start[1])})
    out = []
    for _ in range(steps):
        gm.step()
        out.append(bool(gm.coupling_diagnostics()["a+b"]["converged"]))
    gm.reset_state()
    return out


def _obs(starts, n_samples):
    """Observations whose window starts are ``starts`` and whose other
    samples are zero (so a kept window has a loss far from zero)."""
    per = n_samples
    a = np.zeros(1 + per * len(starts), np.float32)
    b = np.zeros_like(a)
    for w, (sa, sb) in enumerate(starts):
        a[w * per], b[w * per] = sa, sb
    return {"a": {"x": jnp.asarray(a)}, "b": {"x": jnp.asarray(b)}}


@pytest.fixture(scope="module")
def capped():
    return _pair(4)


@pytest.fixture(scope="module")
def uncapped():
    return _pair(400)


def test_the_window_recovers_its_last_step_converges_and_its_early_ones_do_not(capped):
    """The fixture expresses the case: early steps at the cap, the last one fine."""
    v = _verdicts(capped, FAR, STEPS)
    assert not v[0] and not v[1], v
    assert v[-1], v
    # Monotone: once the iterate is close enough, every later step converges.
    first_ok = v.index(True)
    assert all(v[first_ok:]) and 2 <= first_ok < STEPS - 1, v


# (window, sample_every): the verdict is and-ed over the base steps of one
# sample and over the samples of one window; each shape puts the recovery
# across a different one of the two reductions.
SHAPES = [(STEPS, 1), (1, STEPS), (6, 5)]


@pytest.mark.parametrize("window,sample_every", SHAPES)
def test_a_window_whose_early_steps_hit_the_cap_is_dropped_though_its_last_step_converged(
        capped, window, sample_every):
    obs = _obs([FAR], window)

    def loss(p, mask):
        return sysid.windowed_loss(capped, p, obs, obs_fn=lambda s: s, window=window,
                                   sample_every=sample_every, mask_unconverged=mask)

    value, grad = jax.value_and_grad(loss)(capped.params, True)
    assert float(value) == 0.0
    for leaf in jax.tree.leaves(grad):
        np.testing.assert_array_equal(np.asarray(leaf), 0.0)
    # The window is not trivially zero: unmasked, it has a loss and a gradient.
    kept, kept_grad = jax.value_and_grad(loss)(capped.params, False)
    assert float(kept) > 1.0
    assert float(abs(kept_grad["nodes"]["a"]["g"])) > 0.0


@pytest.mark.parametrize("window,sample_every", SHAPES)
def test_the_same_window_is_kept_when_the_cap_lets_every_step_converge(
        uncapped, window, sample_every):
    assert all(_verdicts(uncapped, FAR, STEPS))
    obs = _obs([FAR], window)

    def loss(p, mask):
        return sysid.windowed_loss(uncapped, p, obs, obs_fn=lambda s: s, window=window,
                                   sample_every=sample_every, mask_unconverged=mask)

    value, grad = jax.value_and_grad(loss)(uncapped.params, True)
    plain, plain_grad = jax.value_and_grad(loss)(uncapped.params, False)
    assert float(value) == float(plain) > 1.0
    for a, b in zip(jax.tree.leaves(grad), jax.tree.leaves(plain_grad)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_multiple_shooting_drops_the_recovering_window_and_keeps_the_converged_one(capped):
    """Two free-start windows: the first recovers (dropped, with its
    continuity term), the second starts at the fixed point (kept)."""
    assert all(_verdicts(capped, FIXED, STEPS))
    obs = _obs([FAR, FIXED], STEPS)
    ws = sysid.init_window_states(obs, STEPS)

    def loss(p, w, mask=True):
        return sysid.windowed_loss(capped, p, obs, obs_fn=lambda s: s, window=STEPS,
                                   mask_unconverged=mask, window_states=w,
                                   continuity_weight=1.0)

    value, (gp, gw) = jax.value_and_grad(loss, argnums=(0, 1))(capped.params, ws)

    # What is left is exactly the second window alone (it is the last
    # window, so it has no continuity term of its own).
    tail = jax.tree.map(lambda x: x[STEPS:], obs)
    tail_ws = sysid.init_window_states(tail, STEPS)

    def alone(p, w):
        return sysid.windowed_loss(capped, p, tail, obs_fn=lambda s: s, window=STEPS,
                                   mask_unconverged=True, window_states=w,
                                   continuity_weight=1.0)

    ref, (rp, rw) = jax.value_and_grad(alone, argnums=(0, 1))(capped.params, tail_ws)
    assert float(value) == float(ref) > 1.0
    for a, b in zip(jax.tree.leaves(gp), jax.tree.leaves(rp)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # The dropped window's free start has no gradient; the kept one's has
    # the gradient it has alone.
    for leaf, ref_leaf in zip(jax.tree.leaves(gw), jax.tree.leaves(rw)):
        np.testing.assert_array_equal(np.asarray(leaf[0]), 0.0)
        np.testing.assert_array_equal(np.asarray(leaf[1]), np.asarray(ref_leaf[0]))
    # Unmasked, the first window and its continuity term are in the loss.
    assert float(loss(capped.params, ws, mask=False)) > float(value)
