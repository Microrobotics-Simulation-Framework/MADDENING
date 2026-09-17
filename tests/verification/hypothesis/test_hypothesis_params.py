"""Property-based tests for the graph parameter pytree.

The compiled step is ``step_fn(state, external_inputs, params)``.  These
properties pin down the contract of the third argument:

(a) ``params=None`` (the compile-time snapshot, baked as constants) and
    ``params=gm.params`` (traced input) produce the same step to float32
    round-off — with and without an IFT-coupled group.
(b) The traced-params path is consistent across ``step``, ``run_scan``
    and ``run_sweep``.
(c) ``jax.grad`` w.r.t. params through a ``lax.scan`` rollout matches a
    float64 central finite difference.
(d) jvp and vjp w.r.t. params are adjoint (``<J v, w> == <v, J^T w>``),
    including through the ``custom_jvp`` IFT rule.
(e) A modified ``params`` never dirties the graph, never recompiles, and
    never mutates ``gm.params``.
(f) ``params_pytree()`` round-trips through ``save_state`` / ``load_state``.

Round-off contract: node constants are traced, so ``dt * g`` is a runtime
multiply feeding an add that XLA may or may not contract into an FMA
depending on shape.  Equality is therefore "a few ulps", not bit-exact.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import copy
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.checkpoint import load_state, save_state
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode

DT = 0.01
EPS32 = float(np.finfo(np.float32).eps)
#: Tolerance for two evaluations of the *same* float32 program; the
#: absolute part is scaled by the compared magnitudes in
#: :func:`_assert_close`, so it carries the same units as ``rtol``.
ULP_TOL = dict(rtol=4 * EPS32, atol=4 * EPS32)


def _ulp_tol(n_steps):
    """Per-step round-off compounds: ``step`` (one jit per step), ``run_scan``
    (a scan body) and ``run_sweep`` (a vmapped scan body) are three XLA
    programs whose FMA contraction of ``dt * const`` can differ, so the
    honest contract over ``n`` steps is a few ulps plus one per step.

    The absolute part is scaled by the magnitudes the rollout actually
    reaches (see :func:`_assert_close`), not by the final value.  Round-off
    accumulates with the numbers the arithmetic passes through, while a
    final value can be near zero through cancellation -- a ball's velocity
    crosses zero on every bounce.  A fixed ``atol=1e-7`` made this suite
    fail for about one sampled rollout in 230 (measured: 7 of 1600
    seed/step combinations, every one of them a 15-step ball velocity
    landing near zero with a difference of 2e-7 to 1e-6)."""
    return dict(rtol=(4 + n_steps) * EPS32, atol=(4 + n_steps) * EPS32)


# ---------------------------------------------------------------------------
# Graph factories (module-scoped: jit compile dominates test time)
# ---------------------------------------------------------------------------


def _spring_gm(k=30.0, c=2.0, m=1.0, x0=0.5):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=k, damping=c, mass=m,
                                 rest_length=1.0, initial_position=x0))
    gm.compile()
    return gm


def _ball_gm(x0=5.0, v0=0.0):
    gm = GraphManager()
    gm.add_node(BallNode("b", DT, initial_position=x0, initial_velocity=v0))
    gm.compile()
    return gm


def _coupled_gm(k=30.0, c=2.0, m=1.0):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("sa", DT, stiffness=k, damping=c, mass=m,
                                 rest_length=1.0, initial_position=0.0))
    gm.add_node(SpringDamperNode("sb", DT, stiffness=k, damping=c, mass=m,
                                 rest_length=1.0, initial_position=3.0))
    gm.add_edge("sa", "sb", "position", "anchor_position")
    gm.add_edge("sb", "sa", "position", "anchor_position")
    gm.add_coupling_group(["sa", "sb"], max_iterations=30, tolerance=1e-8)
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def spring():
    return _spring_gm()


@pytest.fixture(scope="module")
def ball():
    return _ball_gm()


@pytest.fixture(scope="module")
def coupled():
    return _coupled_gm()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _user(state):
    return {k: v for k, v in state.items() if k != "_meta"}


def _copy_params(params):
    return jax.tree.map(lambda x: x, params)


def _leaves_equal(a, b):
    la, lb = jax.tree.leaves(a), jax.tree.leaves(b)
    return len(la) == len(lb) and all(
        np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(la, lb)
    )


def _assert_close(a, b, msg, **tol):
    """``numpy.allclose`` over two pytrees, with the absolute tolerance
    scaled by the largest magnitude in either tree (floored at one).  That
    scale stands in for the magnitudes the arithmetic passed through, which
    is what float32 round-off is proportional to; comparing a near-zero
    leaf against a fixed ``atol`` alone is a flaky test, not a tight one."""
    scale = max(
        1.0,
        *(float(np.max(np.abs(np.asarray(v)))) if np.size(np.asarray(v)) else 1.0
          for tree in (a, b) for v in jax.tree_util.tree_leaves(tree)),
    )
    tol = {**tol, "atol": tol.get("atol", 0.0) * scale}
    for (pa, x), (_, y) in zip(
        jax.tree_util.tree_flatten_with_path(a)[0],
        jax.tree_util.tree_flatten_with_path(b)[0],
    ):
        x, y = np.asarray(x), np.asarray(y)
        assert np.allclose(x, y, **tol), (
            f"{msg}: {jax.tree_util.keystr(pa)} max abs diff "
            f"{np.max(np.abs(x - y))} ({x} vs {y})"
        )


def _random_spring_params(rng, gm, nodes):
    """A perturbed copy of ``gm.params`` (never the snapshot itself)."""
    p = _copy_params(gm.params)
    for n in nodes:
        p["nodes"][n]["stiffness"] = jnp.asarray(rng.uniform(1.0, 200.0), jnp.float32)
        p["nodes"][n]["damping"] = jnp.asarray(rng.uniform(0.0, 10.0), jnp.float32)
        p["nodes"][n]["mass"] = jnp.asarray(rng.uniform(0.5, 5.0), jnp.float32)
    return p


def _random_spring_state(rng, gm, nodes):
    s = copy.copy(gm._state)  # noqa: SLF001
    for n in nodes:
        s[n] = {
            "position": jnp.asarray(rng.uniform(-3.0, 3.0), jnp.float32),
            "velocity": jnp.asarray(rng.uniform(-3.0, 3.0), jnp.float32),
        }
    return s


def _random_ball_state(rng, gm):
    s = copy.copy(gm._state)  # noqa: SLF001
    s["b"] = {
        "position": jnp.asarray(rng.uniform(0.0, 10.0), jnp.float32),
        "velocity": jnp.asarray(rng.uniform(-5.0, 5.0), jnp.float32),
    }
    return s


def _rollout(step_fn, ext, state0, params, n_steps, nodes):
    def body(s, _):
        s = step_fn(s, ext, params)
        return s, jnp.stack([s[n]["position"] for n in nodes])

    _, traj = jax.lax.scan(body, state0, None, length=n_steps)
    return traj


# ---------------------------------------------------------------------------
# (a) baked snapshot == traced params
# ---------------------------------------------------------------------------


class TestBakedEqualsTraced:
    """``_compiled_step(state, ext, None)`` vs ``(state, ext, gm.params)``."""

    @given(seed=st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=40, deadline=None)
    def test_spring(self, spring, seed):
        rng = np.random.default_rng(seed)
        state = _random_spring_state(rng, spring, ("s",))
        ext = spring._default_external_inputs()  # noqa: SLF001
        baked = spring._compiled_step(state, ext, None)  # noqa: SLF001
        traced = spring._compiled_step(state, ext, spring.params)  # noqa: SLF001
        _assert_close(_user(baked), _user(traced), "spring baked vs traced", **ULP_TOL)

    @given(seed=st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=40, deadline=None)
    def test_ball(self, ball, seed):
        rng = np.random.default_rng(seed)
        state = _random_ball_state(rng, ball)
        ext = ball._default_external_inputs()  # noqa: SLF001
        baked = ball._compiled_step(state, ext, None)  # noqa: SLF001
        traced = ball._compiled_step(state, ext, ball.params)  # noqa: SLF001
        _assert_close(_user(baked), _user(traced), "ball baked vs traced", **ULP_TOL)

    @given(seed=st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=25, deadline=None)
    def test_coupled(self, coupled, seed):
        """Through the IFT-coupled group the fixed point is found by an
        early-exit while_loop; the same start must give the same iterate
        sequence, so ulp agreement still holds."""
        rng = np.random.default_rng(seed)
        state = _random_spring_state(rng, coupled, ("sa", "sb"))
        ext = coupled._default_external_inputs()  # noqa: SLF001
        baked = coupled._compiled_step(state, ext, None)  # noqa: SLF001
        traced = coupled._compiled_step(state, ext, coupled.params)  # noqa: SLF001
        _assert_close(_user(baked), _user(traced), "coupled baked vs traced", **ULP_TOL)


# ---------------------------------------------------------------------------
# (b) step == run_scan == run_sweep with an explicit (perturbed) params
# ---------------------------------------------------------------------------


class TestRunPathsAgree:

    @given(
        seed=st.integers(min_value=0, max_value=2**31),
        n_steps=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=25, deadline=None)
    def test_spring_step_scan_sweep(self, spring, seed, n_steps):
        rng = np.random.default_rng(seed)
        p = _random_spring_params(rng, spring, ("s",))
        init = _random_spring_state(rng, spring, ("s",))

        spring._state = init  # noqa: SLF001
        for _ in range(n_steps):
            stepped = spring.step(params=p)

        spring._state = init  # noqa: SLF001
        scanned = spring.run_scan(n_steps, params=p)

        batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (3,) + x.shape), init)
        swept = spring.run_sweep(n_steps, batched, params=p)
        row = jax.tree.map(lambda x: x[1], swept)

        _assert_close(stepped, scanned, "step vs run_scan", **_ulp_tol(n_steps))
        _assert_close(scanned, row, "run_scan vs run_sweep", **_ulp_tol(n_steps))

    @given(
        seed=st.integers(min_value=0, max_value=2**31),
        n_steps=st.integers(min_value=1, max_value=30),
    )
    @settings(max_examples=25, deadline=None)
    def test_ball_step_scan_sweep(self, ball, seed, n_steps):
        rng = np.random.default_rng(seed)
        p = _copy_params(ball.params)
        p["nodes"]["b"]["gravity"] = jnp.asarray(rng.uniform(-20.0, -1.0), jnp.float32)
        init = _random_ball_state(rng, ball)

        ball._state = init  # noqa: SLF001
        for _ in range(n_steps):
            stepped = ball.step(params=p)
        ball._state = init  # noqa: SLF001
        scanned = ball.run_scan(n_steps, params=p)
        batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (2,) + x.shape), init)
        row = jax.tree.map(lambda x: x[0], ball.run_sweep(n_steps, batched, params=p))

        _assert_close(stepped, scanned, "step vs run_scan", **_ulp_tol(n_steps))
        _assert_close(scanned, row, "run_scan vs run_sweep", **_ulp_tol(n_steps))

    @pytest.mark.parametrize("seed", [17, 61, 132, 222, 259, 337, 355])
    def test_ball_rollouts_agree_when_the_final_velocity_is_near_zero(self, ball, seed):
        """Seeds whose 15-step rollout ends with the velocity close to zero.

        The three rollout paths then differ by 2e-7 to 1e-6 in absolute
        terms -- ordinary float32 round-off for a velocity that reached
        several m/s on the way -- which a fixed ``atol`` rejected while a
        magnitude-scaled one accepts.  Pinned so the flake cannot return
        unnoticed: each of these was a real failure of
        ``test_ball_step_scan_sweep`` before the tolerance was fixed."""
        n_steps = 15
        rng = np.random.default_rng(seed)
        p = _copy_params(ball.params)
        p["nodes"]["b"]["gravity"] = jnp.asarray(rng.uniform(-20.0, -1.0), jnp.float32)
        init = _random_ball_state(rng, ball)

        ball._state = init  # noqa: SLF001
        for _ in range(n_steps):
            stepped = ball.step(params=p)
        ball._state = init  # noqa: SLF001
        scanned = ball.run_scan(n_steps, params=p)

        assert abs(float(scanned["b"]["velocity"])) < 0.5      # the cancellation case
        _assert_close(stepped, scanned, "step vs run_scan", **_ulp_tol(n_steps))

    @given(
        seed=st.integers(min_value=0, max_value=2**31),
        n_steps=st.integers(min_value=1, max_value=15),
    )
    # Explicit cap: each example traces step, run_scan and run_sweep
    # through a coupling group at a freshly drawn ``n_steps``, i.e. three
    # compiles per example.  25 matches the neighbouring parity properties.
    @settings(max_examples=25, deadline=None)
    def test_coupled_step_scan_sweep(self, coupled, seed, n_steps):
        """Same through the coupling group; ``_meta`` is batched too."""
        rng = np.random.default_rng(seed)
        p = _random_spring_params(rng, coupled, ("sa", "sb"))
        init = _random_spring_state(rng, coupled, ("sa", "sb"))

        coupled._state = init  # noqa: SLF001
        for _ in range(n_steps):
            stepped = coupled.step(params=p)
        coupled._state = init  # noqa: SLF001
        scanned = coupled.run_scan(n_steps, params=p)
        batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (2,) + x.shape), init)
        row = jax.tree.map(lambda x: x[1], coupled.run_sweep(n_steps, batched, params=p))

        _assert_close(stepped, scanned, "step vs run_scan", **_ulp_tol(n_steps))
        _assert_close(scanned, row, "run_scan vs run_sweep", **_ulp_tol(n_steps))


# ---------------------------------------------------------------------------
# (c) params gradient vs float64 finite differences
# ---------------------------------------------------------------------------


class TestGradientMatchesFiniteDifferences:

    @staticmethod
    def _loss(gm, params, state0, n_steps):
        step_fn = gm._build_step_fn()  # noqa: SLF001
        ext = gm._default_external_inputs()  # noqa: SLF001
        traj = _rollout(step_fn, ext, state0, params, n_steps, ("s",))
        return jnp.mean(traj ** 2)

    @given(
        seed=st.integers(min_value=0, max_value=2**31),
        n_steps=st.integers(min_value=5, max_value=40),
        which=st.sampled_from(["stiffness", "damping"]),
    )
    @settings(max_examples=20, deadline=None)
    def test_spring_params_gradient(self, seed, n_steps, which):
        rng = np.random.default_rng(seed)
        k = float(rng.uniform(1.0, 200.0))
        c = float(rng.uniform(0.0, 10.0))
        m = float(rng.uniform(0.5, 5.0))
        x0 = float(rng.uniform(-2.0, 2.0))
        v0 = float(rng.uniform(-2.0, 2.0))

        gm32 = _spring_gm(k, c, m, x0)
        state32 = copy.copy(gm32._state)  # noqa: SLF001
        state32["s"] = {"position": jnp.asarray(x0, jnp.float32),
                        "velocity": jnp.asarray(v0, jnp.float32)}

        def loss_of(gm, theta, dtype, state):
            params = jax.tree.map(lambda x: jnp.asarray(x, dtype), gm.params)
            params["nodes"]["s"][which] = jnp.asarray(theta, dtype)
            return self._loss(gm, params, state, n_steps)

        theta0 = float(gm32.params["nodes"]["s"][which])
        g32 = float(jax.grad(lambda t: loss_of(gm32, t, jnp.float32, state32))(
            jnp.asarray(theta0, jnp.float32)))

        prev = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:
            gm64 = _spring_gm(k, c, m, x0)
            state64 = jax.tree.map(
                lambda x: x.astype(jnp.float64) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                state32,
            )
            h = 1e-4 * max(abs(theta0), 1.0)
            lp = float(loss_of(gm64, theta0 + h, jnp.float64, state64))
            lm = float(loss_of(gm64, theta0 - h, jnp.float64, state64))
            g_fd = (lp - lm) / (2 * h)
            g64 = float(jax.grad(lambda t: loss_of(gm64, t, jnp.float64, state64))(
                jnp.asarray(theta0, jnp.float64)))
        finally:
            jax.config.update("jax_enable_x64", prev)

        assume(abs(g64) > 1e-6)
        assert abs(g64 - g_fd) / abs(g64) < 1e-4, (g64, g_fd)
        assert abs(g32 - g64) / abs(g64) < 1e-3, (g32, g64, g_fd)


# ---------------------------------------------------------------------------
# (d) jvp / vjp adjoint identity w.r.t. params
# ---------------------------------------------------------------------------


class TestAdjointIdentity:

    @staticmethod
    def _check(gm, nodes, rng, n_steps):
        ext = gm._default_external_inputs()  # noqa: SLF001
        step_fn = gm._build_step_fn()  # noqa: SLF001
        state = _random_spring_state(rng, gm, nodes)
        p0 = _random_spring_params(rng, gm, nodes)["nodes"]

        def f(node_params):
            params = {"nodes": node_params, "mappings": {}}
            s = state
            for _ in range(n_steps):
                s = step_fn(s, ext, params)
            return {n: s[n] for n in nodes}

        v = jax.tree.map(lambda x: jnp.asarray(rng.standard_normal(x.shape), x.dtype), p0)
        out, jv = jax.jvp(f, (p0,), (v,))
        w = jax.tree.map(lambda x: jnp.asarray(rng.standard_normal(x.shape), x.dtype), out)
        _, vjp_fn = jax.vjp(f, p0)
        (jtw,) = vjp_fn(w)

        lhs = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(jv), jax.tree.leaves(w)))
        rhs = sum(jnp.vdot(a, b) for a, b in zip(jax.tree.leaves(v), jax.tree.leaves(jtw)))
        lhs, rhs = float(lhs), float(rhs)
        scale = max(abs(lhs), abs(rhs), 1e-3)
        assert abs(lhs - rhs) / scale < 1e-4, (lhs, rhs)

    @given(
        seed=st.integers(min_value=0, max_value=2**31),
        n_steps=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=25, deadline=None)
    def test_single_spring(self, spring, seed, n_steps):
        self._check(spring, ("s",), np.random.default_rng(seed), n_steps)

    @given(
        seed=st.integers(min_value=0, max_value=2**31),
        n_steps=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=20, deadline=None)
    def test_coupled_pair(self, coupled, seed, n_steps):
        """The IFT rule is a ``custom_jvp``; its transpose (via
        ``custom_linear_solve``) must be the true adjoint."""
        self._check(coupled, ("sa", "sb"), np.random.default_rng(seed), n_steps)


# ---------------------------------------------------------------------------
# (e) modified params never dirty / recompile / mutate
# ---------------------------------------------------------------------------


class TestParamsAreARuntimeArgument:

    @given(
        seed=st.integers(min_value=0, max_value=2**31),
        n_steps=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=25, deadline=None)
    def test_no_dirty_no_recompile_no_mutation(self, spring, seed, n_steps):
        rng = np.random.default_rng(seed)
        p_mod = _random_spring_params(rng, spring, ("s",))
        snapshot = _copy_params(spring.params)
        snapshot_leaves = [np.asarray(x).copy() for x in jax.tree.leaves(snapshot)]

        compiled_id = id(spring._compiled_step)  # noqa: SLF001
        assert not spring._dirty  # noqa: SLF001

        spring.step(params=p_mod)
        spring.run_scan(n_steps, params=p_mod)

        assert not spring._dirty  # noqa: SLF001
        assert id(spring._compiled_step) == compiled_id  # noqa: SLF001
        assert spring.params is not p_mod
        for x, y in zip(jax.tree.leaves(spring.params), snapshot_leaves):
            assert np.array_equal(np.asarray(x), y), "gm.params mutated"

    @given(seed=st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=25, deadline=None)
    def test_default_params_after_modified_step(self, spring, seed):
        """``step(params=p_mod)`` followed by ``step()`` uses ``gm.params``
        for the second step — the modified pytree does not stick."""
        rng = np.random.default_rng(seed)
        p_mod = _random_spring_params(rng, spring, ("s",))
        init = _random_spring_state(rng, spring, ("s",))

        spring._state = init  # noqa: SLF001
        spring.step(params=p_mod)
        after_mod = spring._state  # noqa: SLF001
        second = spring.step()

        fresh = _spring_gm()
        fresh._state = after_mod  # noqa: SLF001
        expected = fresh.step()
        _assert_close(second, expected, "default step after modified step", **ULP_TOL)


# ---------------------------------------------------------------------------
# (f) params round-trip through checkpoint
# ---------------------------------------------------------------------------


class TestCheckpointRoundTrip:

    @given(seed=st.integers(min_value=0, max_value=2**31))
    @settings(max_examples=20, deadline=None)
    def test_params_round_trip(self, seed):
        rng = np.random.default_rng(seed)
        gm = _coupled_gm()
        for n in ("sa", "sb"):
            for key in gm.params["nodes"][n]:
                gm.params["nodes"][n][key] = jnp.asarray(rng.uniform(0.1, 50.0), jnp.float32)
        saved = _copy_params(gm.params)

        with tempfile.TemporaryDirectory() as d:
            path = save_state(gm, Path(d) / "ckpt")
            fresh = _coupled_gm()
            assert not _leaves_equal(fresh.params, saved)
            load_state(fresh, path)

        assert set(fresh.params["nodes"]) == set(saved["nodes"])
        for n in saved["nodes"]:
            assert set(fresh.params["nodes"][n]) == set(saved["nodes"][n])
            for key, val in saved["nodes"][n].items():
                assert np.array_equal(np.asarray(fresh.params["nodes"][n][key]),
                                      np.asarray(val)), (n, key)

    def test_unknown_params_are_ignored(self):
        """A checkpoint carrying params for a node the target graph lacks
        (same node set, but the target's node doesn't accept params) is
        loaded without error and only the known keys are restored."""
        gm = _spring_gm()
        gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(77.0, jnp.float32)

        with tempfile.TemporaryDirectory() as d:
            path = save_state(gm, Path(d) / "ckpt")
            # Inject an extra params key the target graph does not know.
            data = dict(np.load(path))
            data["_params/s/not_a_param"] = np.asarray(1.0, np.float32)
            data["_params/ghost/stiffness"] = np.asarray(2.0, np.float32)
            np.savez(path, **data)

            fresh = _spring_gm()
            load_state(fresh, path)

        assert float(fresh.params["nodes"]["s"]["stiffness"]) == 77.0
        assert "not_a_param" not in fresh.params["nodes"]["s"]
        assert "ghost" not in fresh.params["nodes"]
