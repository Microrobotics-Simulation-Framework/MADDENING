"""Property-based tests for ``maddening.sysid``.

``windowed_loss`` and ``fim`` are exercised over random truth parameters,
random initial states, random rollout lengths and random window tilings
of a spring-damper graph.  The graphs are compiled once per module; the
random parameters flow in through the ``params`` pytree (a traced input,
so no recompile per example), which is exactly the path system
identification uses.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import fim, observations_from_history, windowed_loss
from tests.conftest import EXAMPLES_COSTLY

DT = 0.01
REST = 1.0

# ---------------------------------------------------------------------------
# Graphs (compiled once; parameters are runtime inputs)
# ---------------------------------------------------------------------------


def _single_gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=2.0, mass=1.0,
                                 rest_length=REST, initial_position=0.5))
    gm.compile()
    return gm


def _coupled_gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=2.0, mass=1.0,
                                 rest_length=REST, initial_position=0.5))
    gm.add_node(SpringDamperNode("t", DT, stiffness=30.0, damping=2.0, mass=1.0,
                                 rest_length=REST, initial_position=3.0))
    gm.add_edge("s", "t", "position", "anchor_position")
    gm.add_edge("t", "s", "position", "anchor_position")
    gm.add_coupling_group(["s", "t"], max_iterations=30, tolerance=1e-8)
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def single():
    return _single_gm()


@pytest.fixture(scope="module")
def coupled():
    return _coupled_gm()


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Semi-implicit Euler on the spring is stable for omega*dt < 2 and
# (c/m)*dt < 2; the envelope below keeps omega*dt <= 0.2, (c/m)*dt <= 0.2.
def _finite(lo, hi):
    return st.floats(min_value=lo, max_value=hi, allow_nan=False, allow_infinity=False)


truth_params_st = st.fixed_dictionaries({
    "stiffness": _finite(1.0, 200.0),
    "damping": _finite(0.0, 10.0),
    "mass": _finite(0.5, 5.0),
})
initial_state_st = st.fixed_dictionaries({
    "position": _finite(-5.0, 5.0),
    "velocity": _finite(-2.0, 2.0),
})
# The spring's equilibrium is ``anchor_position + rest_length``, and with no
# anchor edge that is ``REST``.  A state started *at* equilibrium and at rest
# never moves, so a rollout from it carries no information about ``(k, c)``
# and the FIM properties below cannot say anything -- which is what
# ``assume(_position_variance(obs) > 1e-2)`` was throwing away.  Hypothesis
# samples exactly 0.0 far more often than a uniform draw would, so that gate
# fired on 13-31% of draws depending on the test, measured, against ~7% for
# uniform sampling of the same ranges.
#
# Displacing the start by at least half a unit removes the at-rest draws
# without narrowing the dynamics: the envelope on ``(k, c, m)`` is untouched,
# so the lightly-damped stiff spring at ``k=49, c=0.125, m=3`` that
# ``test_crb_is_finite_exactly_where_the_pair_is_identifiable`` documents as
# its counter-example is still drawn. It does not reach zero rejection --
# a soft, heavily damped spring still barely moves inside a 20-step window,
# and constraining *that* away would delete the counter-example. Each test
# records what it still rejects.
displaced_state_st = st.fixed_dictionaries({
    "position": st.one_of(_finite(-5.0, REST - 0.5), _finite(REST + 0.5, 5.0)),
    "velocity": _finite(-2.0, 2.0),
})
# Multiplicative perturbation of the fitted constants (stiffness, damping).
perturb_st = st.fixed_dictionaries({
    "stiffness": _finite(0.5, 2.0),
    "damping": _finite(0.5, 2.0),
})


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


# (n_steps, sample_every, window) triples where the windows tile the
# subsampled observations exactly: T - 1 = n_steps / sample_every is a
# multiple of window.  A fixed set keeps the number of distinct compiled
# shapes bounded.
_N_STEPS = (20, 24, 30, 36, 40, 48, 60, 72, 80, 96, 120)
TILINGS = tuple(
    (n, se, w)
    for n in _N_STEPS
    for se in (1, 2)
    if n % se == 0
    for w in _divisors(n // se)
)
tiling_st = st.sampled_from(TILINGS)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_params(gm, node, values):
    """Copy of ``gm.params`` with ``values`` written into ``node``'s entry."""
    p = jax.tree.map(lambda x: x, gm.params)
    for k, v in values.items():
        p["nodes"][node][k] = jnp.asarray(v, dtype=jnp.float32)
    return p


def _set_state(gm, node, position, velocity):
    gm.set_node_state(node, {
        "position": jnp.asarray(position, dtype=jnp.float32),
        "velocity": jnp.asarray(velocity, dtype=jnp.float32),
    })


def _observe(gm, n_steps, params, sample_every=1):
    """Ground-truth observations (T = n_steps/sample_every + 1 samples)
    of a rollout from the current ``gm`` state with ``params``."""
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(n_steps, params=params)
    obs = observations_from_history(init, hist)
    return jax.tree.map(lambda x: x[::sample_every], obs)


def _position_variance(obs, node="s"):
    return float(jnp.var(obs[node]["position"]))


def _finite_tree(tree):
    return all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(tree))


def _sum_sq(tree):
    return float(sum(np.sum(np.asarray(x, dtype=np.float64) ** 2)
                     for x in jax.tree.leaves(tree)))


# ---------------------------------------------------------------------------
# (g) windowed_loss
# ---------------------------------------------------------------------------


class TestWindowedLoss:

    @given(truth=truth_params_st, init=initial_state_st, tiling=tiling_st,
           factor=perturb_st)
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_zero_at_truth_nonneg_and_finite_elsewhere(
        self, single, truth, init, tiling, factor,
    ):
        """At the truth the loss vanishes; away from it it is finite and
        >= 0; its gradient is finite; a >= 20% stiffness error on a
        moving trajectory gives a strictly positive loss."""
        gm = single
        n_steps, sample_every, window = tiling
        note(f"truth={truth} init={init} tiling={tiling} factor={factor}")

        p_truth = _with_params(gm, "s", truth)
        _set_state(gm, "s", init["position"], init["velocity"])
        obs = _observe(gm, n_steps, p_truth, sample_every)

        loss = jax.jit(lambda p: windowed_loss(
            gm, p, obs, obs_fn=lambda h: h["s"]["position"],
            window=window, sample_every=sample_every,
        ))

        # Same step function, same params: the windowed re-simulation
        # reproduces the observations to round-off.  The scan nesting
        # differs from ``run_scan_with_history`` so XLA may schedule the
        # arithmetic differently; allow float32 ulps relative to the
        # signal energy rather than demanding bit equality.
        scale = _sum_sq(obs["s"]["position"])
        at_truth = float(loss(p_truth))
        assert 0.0 <= at_truth <= 1e-9 * (1.0 + scale), (at_truth, scale)

        p_off = _with_params(gm, "s", {
            "stiffness": truth["stiffness"] * factor["stiffness"],
            "damping": truth["damping"] * factor["damping"],
        })
        off = float(loss(p_off))
        assert np.isfinite(off) and off >= 0.0, off

        g = jax.grad(loss)(p_off)
        assert _finite_tree(g)

        if _position_variance(obs) > 1e-2:
            p_20 = _with_params(gm, "s", {"stiffness": 1.2 * truth["stiffness"]})
            assert float(loss(p_20)) > 0.0

    @given(truth=truth_params_st, init=initial_state_st,
           tiling=st.sampled_from([t for t in TILINGS if t[2] == t[0] // t[1]]),
           factor=perturb_st)
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_single_window_equals_direct_trajectory_loss(
        self, single, truth, init, tiling, factor,
    ):
        """``window == T - 1`` is one teacher-forced window from sample 0,
        i.e. the plain trajectory sum of squared errors."""
        gm = single
        n_steps, sample_every, window = tiling
        assert window * sample_every == n_steps
        note(f"truth={truth} init={init} tiling={tiling} factor={factor}")

        p_truth = _with_params(gm, "s", truth)
        _set_state(gm, "s", init["position"], init["velocity"])
        obs = _observe(gm, n_steps, p_truth, sample_every)
        p_off = _with_params(gm, "s", {
            "stiffness": truth["stiffness"] * factor["stiffness"],
            "damping": truth["damping"] * factor["damping"],
        })

        windowed = float(windowed_loss(
            gm, p_off, obs, obs_fn=lambda h: h["s"]["position"],
            window=window, sample_every=sample_every,
        ))

        step_fn = gm._build_step_fn()  # noqa: SLF001
        ext = gm._default_external_inputs()  # noqa: SLF001
        init_state = jax.tree.map(lambda x: x[0], obs)

        def body(s, _):
            s = step_fn(s, ext, p_off)
            return s, s["s"]["position"]

        _, pos = jax.lax.scan(body, init_state, None, length=n_steps)
        direct = float(jnp.sum((pos[sample_every - 1::sample_every]
                                - obs["s"]["position"][1:]) ** 2))
        assert np.isclose(windowed, direct, rtol=1e-5, atol=1e-7), (windowed, direct)

    @given(truth=truth_params_st, init=initial_state_st,
           tiling=st.sampled_from([t for t in TILINGS if t[1] == 1 and t[0] <= 60]),
           factor=perturb_st)
    # Absolute, not ``EXAMPLES_COSTLY``: an example builds an IFT-coupled
    # group, runs a rollout and differentiates a windowed loss through
    # it -- one JAX compile per (tiling, shape) draw, measured at 3.6 s
    # an example, the most expensive property in the tree.  At the
    # costly tier the ``ci`` profile would spend five minutes here
    # alone, so the depth stays pinned and ``ci`` buys its extra search
    # from the cheaper properties around it.
    @settings(max_examples=30, deadline=None)
    def test_mask_unconverged_through_coupled_group(
        self, coupled, truth, init, tiling, factor,
    ):
        """Through an IFT-coupled group: the masked loss is finite,
        non-negative, never exceeds the unmasked one, and its gradient
        w.r.t. both nodes' params is finite."""
        gm = coupled
        n_steps, sample_every, window = tiling
        note(f"truth={truth} init={init} tiling={tiling} factor={factor}")

        p_truth = _with_params(gm, "s", truth)
        p_truth = {**p_truth, "nodes": {**p_truth["nodes"],
                                        "t": _with_params(gm, "t", truth)["nodes"]["t"]}}
        _set_state(gm, "s", init["position"], init["velocity"])
        _set_state(gm, "t", init["position"] + 2.0, -init["velocity"])
        obs = _observe(gm, n_steps, p_truth)

        def make(mask):
            return jax.jit(lambda p: windowed_loss(
                gm, p, obs,
                obs_fn=lambda h: (h["s"]["position"], h["t"]["position"]),
                window=window, mask_unconverged=mask,
            ))

        masked, unmasked = make(True), make(False)
        scale = _sum_sq((obs["s"]["position"], obs["t"]["position"]))
        assert float(unmasked(p_truth)) <= 1e-9 * (1.0 + scale)

        p_off = _with_params(gm, "s", {
            "stiffness": truth["stiffness"] * factor["stiffness"],
            "damping": truth["damping"] * factor["damping"],
        })
        m, u = float(masked(p_off)), float(unmasked(p_off))
        assert np.isfinite(m) and np.isfinite(u)
        assert 0.0 <= m <= u * (1 + 1e-6) + 1e-12, (m, u)
        assert _finite_tree(jax.grad(masked)(p_off))

    def test_windows_must_tile(self, single):
        gm = single
        obs = _observe(gm, 20, gm.params)
        with pytest.raises(ValueError, match="must divide"):
            windowed_loss(gm, gm.params, obs, obs_fn=lambda h: h["s"]["position"], window=7)


# ---------------------------------------------------------------------------
# (h) fim
# ---------------------------------------------------------------------------

fim_n_st = st.sampled_from((20, 40, 80))

# ``scale="relative"`` multiplies each J column by the parameter value, so
# a parameter that is exactly 0 is (correctly) an exact null direction: a
# relative change of zero is no change.  Keep damping away from 0 for the
# identifiability properties so that null direction is not the one found.
fim_truth_st = st.fixed_dictionaries({
    "stiffness": _finite(1.0, 200.0),
    "damping": _finite(0.1, 10.0),
    "mass": _finite(0.5, 5.0),
})


def _residual_fn(gm, obs, base_params, names, node="s", n_steps=None,
                 combine=None):
    """``sub -> simulated - measured position`` over ``names`` of ``node``.

    ``combine``, if given, maps the sub-pytree to the node's param entries
    (used to inject a duplicated parameter); by default entries are
    copied one-to-one.
    """
    step_fn = gm._build_step_fn()  # noqa: SLF001
    ext = gm._default_external_inputs()  # noqa: SLF001
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs[node]["position"][1:]
    n = truth.shape[0] if n_steps is None else n_steps

    def residual(sub):
        p = jax.tree.map(lambda x: x, base_params)
        entries = combine(sub) if combine is not None else {k: sub[k] for k in names}
        for k, v in entries.items():
            p["nodes"][node][k] = v

        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s[node]["position"]

        _, pos = jax.lax.scan(body, init, None, length=n)
        return pos - truth

    return residual


class TestFIM:

    @given(truth=truth_params_st, init=initial_state_st, n=fim_n_st)
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_symmetric_psd_sorted_orthonormal_cond(self, single, truth, init, n):
        """Structural properties of the report for the (k, c) pair."""
        gm = single
        note(f"truth={truth} init={init} n={n}")
        p_truth = _with_params(gm, "s", truth)
        _set_state(gm, "s", init["position"], init["velocity"])
        obs = _observe(gm, n, p_truth)
        names = ("stiffness", "damping")
        sub = {k: p_truth["nodes"]["s"][k] for k in names}
        rep = fim(_residual_fn(gm, obs, p_truth, names), sub)

        F = np.asarray(rep.fim, dtype=np.float64)
        ev = np.asarray(rep.eigvals, dtype=np.float64)
        V = np.asarray(rep.eigvecs, dtype=np.float64)
        assert np.all(np.isfinite(F))
        assert np.allclose(F, F.T, rtol=1e-5, atol=1e-6 * (1 + np.abs(F).max()))
        assert np.all(ev >= -1e-6 * max(ev.max(), 0.0) - 1e-12), ev
        assert np.all(np.diff(ev) >= 0.0), ev
        tol = 1e-4 if rep.eigvecs.dtype == jnp.float32 else 1e-10
        assert np.allclose(V.T @ V, np.eye(2), atol=tol)
        if ev[0] <= 0.0:
            assert rep.cond == float("inf")
        else:
            assert np.isclose(rep.cond, ev[-1] / ev[0], rtol=1e-6)
        assert rep.param_names == ("['damping']", "['stiffness']")
        assert rep.crb.shape == (2,)

        name, weight = rep.least_identifiable()
        assert name in rep.param_names
        assert 0.0 < weight <= 1.0 + 1e-6

    @given(truth=fim_truth_st, init=displaced_state_st, n=fim_n_st)
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_crb_is_finite_exactly_where_the_pair_is_identifiable(
        self, single, truth, init, n,
    ):
        """``crb`` is finite where the data resolves a direction and ``+inf``
        where it does not, and ``rank`` says which.

        This used to assert that position data on a moving spring *always*
        identifies ``(k, c)``.  It does not, and the old assertion passed only
        because ``pinv`` returned a finite number for a direction the data
        cannot see -- precisely the misreporting ``rank`` was added to end.

        Counter-example found by this property: ``k=49, c=0.125, m=3`` over
        ``n=20`` samples.  A lightly damped stiff spring over a short window
        puts the damping direction at the float32 noise floor -- Fisher
        eigenvalues ``[9.47e-08, 4.55e-01]``, a ratio of 2.1e-07 against the
        ``n*eps`` cutoff of 2.4e-07.  ``numpy.linalg.matrix_rank`` calls that
        matrix rank 1 as well, on the same convention.

        Asserting the equivalence rather than the premise makes this strictly
        stronger than what it replaced: it holds for every draw, identifiable
        or not, and it would catch a ``rank`` that disagreed with its own
        ``crb`` in either direction.
        Rejected draws
        --------------
        ``assume(_position_variance(obs) > 1e-2)`` still rejects 16-26% of
        draws under the ``ci`` profile -- the two ends are two ci runs of
        the same test, which is how much sampling noise 80 examples carry --
        so ``EXAMPLES_COSTLY`` buys that much less search here than the
        number says.  It is not removable by generation: what is left is a soft, heavily damped
        spring that barely moves inside a 20-sample window, and an envelope
        that excluded those would also exclude the lightly-damped stiff
        spring at ``k=49, c=0.125, m=3`` that
        ``test_crb_is_finite_exactly_where_the_pair_is_identifiable``
        documents as this suite's counter-example.  See
        ``displaced_state_st`` and ``scripts/audit_property_rejection.py``.
        """
        gm = single
        note(f"truth={truth} init={init} n={n}")
        p_truth = _with_params(gm, "s", truth)
        _set_state(gm, "s", init["position"], init["velocity"])
        obs = _observe(gm, n, p_truth)
        assume(_position_variance(obs) > 1e-2)
        names = ("stiffness", "damping")
        sub = {k: p_truth["nodes"]["s"][k] for k in names}
        rep = fim(_residual_fn(gm, obs, p_truth, names), sub)
        note(f"eigvals={np.asarray(rep.eigvals)} rank={rep.rank} crb={rep.crb}")

        assert rep.rank in (0, 1, 2), rep.rank
        finite = np.asarray(jnp.isfinite(rep.crb))
        crb = np.asarray(rep.crb)

        if rep.rank == len(names):
            # Full rank: every bound is a real, strictly positive number, and
            # the matrix is invertible so cond is finite too.
            assert finite.all(), rep.crb
            assert bool((crb > 0.0).all()), rep.crb
            assert np.isfinite(rep.cond), rep.cond
        else:
            # Rank deficient: at least one parameter is unresolvable and must
            # say so with +inf rather than a small, confident-looking number.
            assert not finite.all(), rep.crb
            assert bool((crb[finite] > 0.0).all()), rep.crb
            assert bool(np.isinf(crb[~finite]).all()), rep.crb

    @given(truth=fim_truth_st, init=displaced_state_st, n=fim_n_st)
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_common_scale_of_k_c_m_is_the_null_direction(
        self, single, truth, init, n,
    ):
        """Position-only data sees k/m and c/m: scaling (k, c, m) together
        is invisible, so in relative coordinates (1, 1, 1)/sqrt(3) is the
        weakest eigenvector with a ~0 eigenvalue.

        Rejected draws
        --------------
        ``assume(_position_variance(obs) > 1e-2)`` still rejects 22-24% of
        draws under the ``ci`` profile -- the two ends are two ci runs of
        the same test, which is how much sampling noise 80 examples carry --
        so ``EXAMPLES_COSTLY`` buys that much less search here than the
        number says.  It is not removable by generation: what is left is a soft, heavily damped
        spring that barely moves inside a 20-sample window, and an envelope
        that excluded those would also exclude the lightly-damped stiff
        spring at ``k=49, c=0.125, m=3`` that
        ``test_crb_is_finite_exactly_where_the_pair_is_identifiable``
        documents as this suite's counter-example.  See
        ``displaced_state_st`` and ``scripts/audit_property_rejection.py``.
        """
        gm = single
        note(f"truth={truth} init={init} n={n}")
        p_truth = _with_params(gm, "s", truth)
        _set_state(gm, "s", init["position"], init["velocity"])
        obs = _observe(gm, n, p_truth)
        assume(_position_variance(obs) > 1e-2)
        names = ("stiffness", "damping", "mass")
        sub = {k: p_truth["nodes"]["s"][k] for k in names}
        rep = fim(_residual_fn(gm, obs, p_truth, names), sub)
        ev = np.asarray(rep.eigvals, dtype=np.float64)
        ratio = ev[0] / ev[-1]
        assert ratio < 1e-3, ratio
        # The common-scale direction must lie in the *near-null subspace*
        # (every eigenvalue below 1e-3 of the largest), not necessarily on
        # the single smallest eigenvector: when damping is weakly
        # identifiable too, the two smallest eigenvalues are close and any
        # rotation within their plane is an equally valid eigenbasis.
        V = np.asarray(rep.eigvecs, dtype=np.float64)
        null = V[:, ev < 1e-3 * ev[-1]]
        d = np.ones(3) / np.sqrt(3.0)
        proj = np.linalg.norm(null.T @ d)
        assert proj > 0.98, (proj, ev / ev[-1], V[:, 0])

    @given(truth=fim_truth_st, init=displaced_state_st, n=fim_n_st)
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_zero_valued_parameter_is_exact_null_direction_under_relative_scaling(
        self, single, truth, init, n,
    ):
        """At ``damping == 0`` the relative FIM has a zero damping row and
        column, so the weakest direction is exactly the damping axis with
        eigenvalue 0 (a relative change of zero is no change).

        Rejected draws
        --------------
        ``assume(_position_variance(obs) > 1e-2)`` still rejects 8-16% of
        draws under the ``ci`` profile -- the two ends are two ci runs of
        the same test, which is how much sampling noise 80 examples carry --
        so ``EXAMPLES_COSTLY`` buys that much less search here than the
        number says.  It is not removable by generation: what is left is a soft, heavily damped
        spring that barely moves inside a 20-sample window, and an envelope
        that excluded those would also exclude the lightly-damped stiff
        spring at ``k=49, c=0.125, m=3`` that
        ``test_crb_is_finite_exactly_where_the_pair_is_identifiable``
        documents as this suite's counter-example.  See
        ``displaced_state_st`` and ``scripts/audit_property_rejection.py``.
        """
        gm = single
        truth = {**truth, "damping": 0.0}
        note(f"truth={truth} init={init} n={n}")
        p_truth = _with_params(gm, "s", truth)
        _set_state(gm, "s", init["position"], init["velocity"])
        obs = _observe(gm, n, p_truth)
        assume(_position_variance(obs) > 1e-2)
        names = ("stiffness", "damping")
        sub = {k: p_truth["nodes"]["s"][k] for k in names}
        rep = fim(_residual_fn(gm, obs, p_truth, names), sub)
        assert float(rep.eigvals[0]) == 0.0
        assert rep.cond == float("inf")
        v = np.asarray(rep.eigvecs[:, 0], dtype=np.float64)
        assert abs(v[0]) > 0.999, v  # param_names[0] == "['damping']"
        assert rep.least_identifiable()[0] == "['damping']"

    @given(truth=fim_truth_st, init=displaced_state_st, n=fim_n_st,
           split=_finite(0.1, 0.9))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_duplicated_parameter_null_direction_is_difference(
        self, single, truth, init, n, split,
    ):
        """Inject an exact null direction: stiffness = a + b.  With raw
        sensitivities the two columns of J are identical, so the FIM's
        weakest eigenvector is (1, -1)/sqrt(2) with eigenvalue 0 and the
        strongest is (1, 1)/sqrt(2).

        Rejected draws
        --------------
        ``assume(_position_variance(obs) > 1e-2)`` still rejects 18-27% of
        draws under the ``ci`` profile -- the two ends are two ci runs of
        the same test, which is how much sampling noise 80 examples carry --
        so ``EXAMPLES_COSTLY`` buys that much less search here than the
        number says.  It is not removable by generation: what is left is a soft, heavily damped
        spring that barely moves inside a 20-sample window, and an envelope
        that excluded those would also exclude the lightly-damped stiff
        spring at ``k=49, c=0.125, m=3`` that
        ``test_crb_is_finite_exactly_where_the_pair_is_identifiable``
        documents as this suite's counter-example.  See
        ``displaced_state_st`` and ``scripts/audit_property_rejection.py``.
        """
        gm = single
        note(f"truth={truth} init={init} n={n} split={split}")
        p_truth = _with_params(gm, "s", truth)
        _set_state(gm, "s", init["position"], init["velocity"])
        obs = _observe(gm, n, p_truth)
        assume(_position_variance(obs) > 1e-2)
        k = truth["stiffness"]
        sub = {"a": jnp.float32(split * k), "b": jnp.float32((1 - split) * k)}
        rep = fim(
            _residual_fn(gm, obs, p_truth, ("a", "b"),
                         combine=lambda s: {"stiffness": s["a"] + s["b"]}),
            sub, scale=None,
        )
        ev = np.asarray(rep.eigvals, dtype=np.float64)
        assert abs(ev[0]) <= 1e-5 * ev[-1], ev
        v0 = np.asarray(rep.eigvecs[:, 0], dtype=np.float64)
        v1 = np.asarray(rep.eigvecs[:, 1], dtype=np.float64)
        assert abs(v0 @ np.array([1.0, -1.0]) / np.sqrt(2.0)) > 0.999, v0
        assert abs(v1 @ np.array([1.0, 1.0]) / np.sqrt(2.0)) > 0.999, v1
        # The CRB is NaN along a singular FIM (pinv), not a bogus number.
        assert rep.cond == float("inf") or rep.cond > 1e5

    @given(truth=truth_params_st, init=initial_state_st, n=fim_n_st)
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_relative_scaling_is_congruence_by_params(self, single, truth, init, n):
        """``F_rel = D F_raw D`` with ``D = diag(params)``."""
        gm = single
        note(f"truth={truth} init={init} n={n}")
        p_truth = _with_params(gm, "s", truth)
        _set_state(gm, "s", init["position"], init["velocity"])
        obs = _observe(gm, n, p_truth)
        names = ("stiffness", "damping")
        sub = {k: p_truth["nodes"]["s"][k] for k in names}
        residual = _residual_fn(gm, obs, p_truth, names)
        raw = np.asarray(fim(residual, sub, scale=None).fim, dtype=np.float64)
        rel = np.asarray(fim(residual, sub, scale="relative").fim, dtype=np.float64)
        d = np.array([float(sub["damping"]), float(sub["stiffness"])])  # sorted keys
        expect = np.diag(d) @ raw @ np.diag(d)
        assert np.allclose(rel, expect, rtol=1e-4, atol=1e-6 * (1 + np.abs(expect).max()))

    def test_invalid_scale_raises(self, single):
        gm = single
        obs = _observe(gm, 20, gm.params)
        names = ("stiffness", "damping")
        sub = {k: gm.params["nodes"]["s"][k] for k in names}
        with pytest.raises(ValueError, match="scale"):
            fim(_residual_fn(gm, obs, gm.params, names), sub, scale="absolute")


# ---------------------------------------------------------------------------
# Multiple shooting (blind-spot review 2026-09-16)
# ---------------------------------------------------------------------------


class TestMultipleShooting:

    @given(truth=truth_params_st, init=initial_state_st, tiling=tiling_st)
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_seeded_window_states_reproduce_teacher_forcing(self, single, truth, init, tiling):
        """With ``init_window_states`` (the measured window starts) and any
        continuity weight, multiple shooting equals the teacher-forced loss
        at the truth (both ~0) and its window-state gradient is finite —
        for every tiling incl. ``sample_every=2``."""
        from maddening.sysid import init_window_states
        gm = single
        n_steps, sample_every, window = tiling
        note(f"truth={truth} init={init} tiling={tiling}")
        p_truth = _with_params(gm, "s", truth)
        _set_state(gm, "s", init["position"], init["velocity"])
        obs = _observe(gm, n_steps, p_truth, sample_every)
        ws = init_window_states(obs, window)
        assert ws["s"]["position"].shape == ((n_steps // sample_every) // window,)
        ms = jax.jit(lambda p, w: windowed_loss(
            gm, p, obs, obs_fn=lambda h: h["s"]["position"], window=window,
            sample_every=sample_every, window_states=w, continuity_weight=0.7))
        tf = jax.jit(lambda p: windowed_loss(
            gm, p, obs, obs_fn=lambda h: h["s"]["position"], window=window,
            sample_every=sample_every))
        scale = _sum_sq(obs["s"]["position"])
        a, b = float(ms(p_truth, ws)), float(tf(p_truth))
        assert 0.0 <= a <= 1e-9 * (1.0 + scale) and abs(a - b) <= 1e-9 * (1.0 + scale), (a, b)
        g = jax.grad(lambda w: ms(p_truth, w))(ws)
        assert _finite_tree(g)
        # moving one window start off the data costs something
        if ws["s"]["position"].shape[0] > 1:
            bad = jax.tree.map(lambda x: x, ws)
            bad["s"]["position"] = bad["s"]["position"].at[1].add(0.3)
            assert float(ms(p_truth, bad)) > 0.0
