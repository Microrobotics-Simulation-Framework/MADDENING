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

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    _PRECISION_WARN_FACTOR,
    fim,
    observations_from_history,
    windowed_loss,
)
from maddening.warnings import PrecisionLimitWarning
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


@pytest.mark.filterwarnings(
    "ignore::maddening.warnings.PrecisionLimitWarning")
class TestFIM:
    """``fim`` over generated spring-damper parameters.

    The whole class filters ``PrecisionLimitWarning``, and not as a
    workaround: position-only data cannot separate a common scaling of
    ``(k, c, m)``, so for a good share of the generated parameter draws
    the weakest eigenvalue genuinely sits at the float32 noise floor and
    ``fim`` correctly says so -- which is also the honest headline about
    doing identifiability analysis in float32: for this project's
    canonical problem the rank verdict routinely sits within a small
    multiple of the floor.

    The multiples first recorded here -- 1.14x, 1.18x, 1.4x, 1.67x and
    1.97x the cutoff -- were measured against the ``n * eps`` cutoff,
    which no longer exists; ``rank_rtol`` now carries a ``sqrt(m)`` term
    and every one of them is quoted against a cutoff that has moved.
    Re-checked with the filter lifted under the ``ci`` profile, the
    warning still fires here, at 1.97x on a ``k=9, c=0.125, m=2`` draw
    over 40 samples (cutoff ``sqrt(40) * eps``), so the filter is still
    load-bearing and not a leftover.  Fewer draws cross the band than
    before -- one of the seven tests fired in that run rather than five
    -- but *which* draw crosses is exactly what is not stable between
    runs, which is why this stays on the class.

    What these tests assert -- symmetry, PSD-ness, eigenvector
    directions, the congruence identity, where ``crb`` is finite -- are
    statements about the *matrix*, and the warning is about the rank
    verdict read off it.  Marking the class rather than the methods
    because the generators are shared and which draw crosses the band is
    not stable between runs; a per-method mark left one test to fail on
    a later seed.  Every assertion *about* the warning lives in
    :class:`TestPrecisionLimitedRank`, which does not filter it.
    """


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
        ``assume(_position_variance(obs) > 1e-2)`` still rejects 12-26% of
        draws under the ``ci`` profile.  The spread is three ci runs of the
        same test: two from a worktree and one from a checkout path without
        a ``test`` component, which is the only difference that decides
        whether Hypothesis injects this tree's own literals into the draws
        (see ``scripts/audit_property_rejection.py``).  It is what an
        80-example estimate is worth here, so ``EXAMPLES_COSTLY`` buys that much less search here than the
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

        The ``PrecisionLimitWarning`` filter is not a workaround: a
        weakest eigenvalue at ~0 is exactly what this test is *for*, so
        for some draws ``fim`` correctly reports that the rank verdict
        sits at the float32 noise floor.  What is asserted here is the
        eigen*vector*, which the warning says nothing about.

        Rejected draws
        --------------
        ``assume(_position_variance(obs) > 1e-2)`` still rejects 15-24% of
        draws under the ``ci`` profile.  The spread is three ci runs of the
        same test: two from a worktree and one from a checkout path without
        a ``test`` component, which is the only difference that decides
        whether Hypothesis injects this tree's own literals into the draws
        (see ``scripts/audit_property_rejection.py``).  It is what an
        80-example estimate is worth here, so ``EXAMPLES_COSTLY`` buys that much less search here than the
        number says.  It is not removable by generation: what is left is a
        soft, heavily damped spring that barely moves inside a 20-sample
        window, and an envelope that excluded those would also exclude the
        lightly-damped stiff spring at ``k=49, c=0.125, m=3`` that
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
        draws under the ``ci`` profile.  The spread is three ci runs of the
        same test: two from a worktree and one from a checkout path without
        a ``test`` component, which is the only difference that decides
        whether Hypothesis injects this tree's own literals into the draws
        (see ``scripts/audit_property_rejection.py``).  It is what an
        80-example estimate is worth here, so ``EXAMPLES_COSTLY`` buys that much less search here than the
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
        ``assume(_position_variance(obs) > 1e-2)`` still rejects 12-27% of
        draws under the ``ci`` profile.  The spread is three ci runs of the
        same test: two from a worktree and one from a checkout path without
        a ``test`` component, which is the only difference that decides
        whether Hypothesis injects this tree's own literals into the draws
        (see ``scripts/audit_property_rejection.py``).  It is what an
        80-example estimate is worth here, so ``EXAMPLES_COSTLY`` buys that much less search here than the
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
        """``F_rel = D F_raw D`` with ``D = diag(params)``.

        Filtered for ``PrecisionLimitWarning`` for the reason above: the
        congruence identity is about the matrix and holds whatever the
        conditioning, while some generated spring parameters put the
        rank verdict at the noise floor and ``fim`` now says so.
        """
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


# ---------------------------------------------------------------------------
# The precision-limit warning on ``fim``'s rank verdict
# ---------------------------------------------------------------------------


_EPS32 = float(np.finfo(np.float32).eps)


def _fisher_with_known_ratio(n, m, ratio_x_cutoff, seed):
    """A linear residual whose Fisher matrix has a *known* float64
    smallest-to-largest eigenvalue ratio, placed at ``ratio_x_cutoff``
    times the cutoff ``fim`` will decide rank against.

    Built in float64 and handed to ``fim`` as float32, so the float64
    reference below is the same matrix at the other precision rather
    than a different matrix.

    The cutoff is ``max(n, sqrt(m)) * eps``, which is ``fim``'s own
    default and not merely a number of the same shape.  It has to be:
    every property here is stated in units of the cutoff, and a
    reference applying a *different* cutoff from the one under test
    disagrees for structural reasons that have nothing to do with
    precision.  Before ``rank_rtol`` gained its ``sqrt(m)`` term this
    read ``n * _EPS32``, and it was the same expression then.
    """
    rng = np.random.default_rng(seed)
    cutoff = max(n, np.sqrt(m)) * _EPS32
    mid = np.geomspace(1e-3, 1.0, max(n - 1, 1))
    mid[-1] = 1.0                       # geomspace(a, b, 1) is [a], not [b]
    lam = np.sort(np.concatenate([[ratio_x_cutoff * cutoff], mid]))[:n]
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    V = q * np.sign(np.diag(r))
    U = np.linalg.qr(rng.standard_normal((m, n)))[0]
    return (U * np.sqrt(lam)) @ V.T, cutoff


def _rank_at(eigvals, cutoff):
    ev = np.asarray(eigvals, dtype=np.float64)
    return int((ev > max(float(ev[-1]), 0.0) * cutoff).sum())


def _fim_of(J64):
    n = J64.shape[1]
    A = jnp.asarray(J64, dtype=jnp.float32)
    params = {f"p{i}": jnp.float32(1.0) for i in range(n)}
    keys = tuple(params)

    def residual_fn(p):
        return A @ jnp.stack([p[k] for k in keys])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        report = fim(residual_fn, params, scale=None)
    warned = any(isinstance(w.message, PrecisionLimitWarning) for w in caught)
    return report, warned


class TestPrecisionLimitedRank:
    """``fim``'s ``rank`` is a comparison of two numbers, and in float32
    it can be a comparison of two numbers that differ by less than the
    decomposition resolves.  The contract is that such a verdict
    announces itself and an ordinary one stays quiet.

    The generators here sweep ``m`` up to 1024, and that is the point.
    They used to cap it at 32, because the error in forming
    ``F = J.T @ J`` in float32 grows with the number of residual rows
    while the then-current ``n * eps`` cutoff could not see ``m`` at
    all: at long residuals a precision-limited verdict landed
    arbitrarily far from the cutoff and no symmetric factor reached it.
    The measured miss rate over the full sweep was ~8%, essentially all
    of it there, so asserting these properties over long residuals
    would have been asserting something measured to be false.

    ``rank_rtol`` now carries a ``sqrt(m)`` term, the cutoff tracks the
    floor, and the misses are gone: over 228,000 synthetic matrices
    (n = 2..25, m = 20..2000) recall is 1.000 at the current factor, by
    every ``m`` bucket separately -- ``m`` in [20, 50], [50, 200],
    [200, 800] and [800, 2000] all measure 1.000, against 0.912-0.916
    overall under the old cutoff.  The cap coming off is the property getting
    its teeth back, and a regression in the cutoff shows up here as a
    failure rather than as a region nobody asserts anything about.
    """

    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    @given(
        n=st.integers(min_value=2, max_value=6),
        m=st.integers(min_value=2, max_value=1024),
        # log-uniform across the cutoff: both verdicts occur, and
        # disagreements are common enough for the property to bite
        log_ratio=st.floats(min_value=np.log(0.05), max_value=np.log(20.0)),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_a_verdict_the_two_precisions_disagree_about_warns(
            self, n, m, log_ratio, seed):
        """The property the whole feature is: where float32 and float64
        arithmetic reach *different* ranks from the same matrix under
        the same rank rule, the float32 answer was decided by rounding,
        and saying so is the only thing that lets a user act.

        Both verdicts apply the float32 cutoff.  Comparing each
        precision's own default cutoff instead would be a different
        question with a useless answer: those disagree for every ratio
        between ``n * 2.2e-16`` and ``n * 1.2e-07``, nine decades of
        merely ill-conditioned problems, and a warning over all of them
        is the routine firing that gets warnings suppressed.
        """
        J64, cutoff = _fisher_with_known_ratio(n, max(m, n),
                                               float(np.exp(log_ratio)), seed)
        report, warned = _fim_of(J64)
        rank64 = _rank_at(np.linalg.eigh(J64.T @ J64)[0], cutoff)
        note(f"n={n} m={m} ratio/cutoff={np.exp(log_ratio):.4g} "
             f"rank32={report.rank} rank64={rank64} warned={warned}")
        if report.rank != rank64:
            assert warned, (
                f"rank={report.rank} in float32 but {rank64} in float64 "
                f"under the same cutoff {cutoff:.4g}, and nothing said so; "
                f"eigvals={np.asarray(report.eigvals)}"
            )

    # Twice the profile's count, because the assertion below is guarded by
    # ``if warned`` rather than reached through ``assume(warned)`` and only
    # about half of this generator's draws warn (measured: 51.5% of draws
    # were discarded when this was an ``assume``).  Doubling restores the
    # number of *warned* cases the property is checked on -- ~80, as before
    # -- at the same number of draws the ``assume`` form already cost, with
    # none of them thrown away.  See
    # docs/developer_guide/testing_standards.md on rejection budgets.
    # Re-measured at 57.5% (92 warned of 160 draws) after the cutoff gained
    # its ``sqrt(m)`` term and this generator's ``m`` cap came off, which
    # changed both inputs to that 51.5%: still about half, so the doubling
    # still buys the ~80 warned cases it was sized for.
    @settings(max_examples=2 * EXAMPLES_COSTLY, deadline=None)
    @given(
        n=st.integers(min_value=2, max_value=6),
        m=st.integers(min_value=2, max_value=1024),
        log_ratio=st.floats(min_value=np.log(0.05), max_value=np.log(20.0)),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_a_warning_is_always_backed_by_a_ratio_inside_the_band(
            self, n, m, log_ratio, seed):
        """The other half: a warning that fires anywhere else would be
        noise.  Whenever it fires, some eigenvalue ratio really is
        within the measured factor of the cutoff -- and the number the
        message quotes is that ratio, not ``eigvals[0]``, which can be
        decades away from the comparison being made.

        The implication is tested as an implication, the way
        :meth:`test_a_verdict_the_two_precisions_disagree_about_warns`
        does one line above.  It used to be ``assume(warned)``, which
        threw away every draw that did not warn -- half of them, the
        highest rejection rate in either property suite -- and narrowing
        ``log_ratio`` towards the band to raise that rate would have
        deleted exactly the draws a spuriously-fired warning would show
        up in, which is the bug this property hunts.
        """
        J64, cutoff = _fisher_with_known_ratio(n, max(m, n),
                                               float(np.exp(log_ratio)), seed)
        report, warned = _fim_of(J64)
        if not warned:
            return
        ev = np.asarray(report.eigvals, dtype=np.float64)
        assert float(ev[-1]) > 0.0
        ratios = ev / float(ev[-1])
        inside = [r for r in ratios
                  if r > 0 and cutoff / _PRECISION_WARN_FACTOR
                  <= r <= cutoff * _PRECISION_WARN_FACTOR]
        assert inside, (
            f"warned with no ratio inside "
            f"[{cutoff / _PRECISION_WARN_FACTOR:.4g}, "
            f"{cutoff * _PRECISION_WARN_FACTOR:.4g}]: {ratios}"
        )

    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    @given(
        n=st.integers(min_value=2, max_value=8),
        m=st.integers(min_value=2, max_value=1024),
        # 1e2 .. 1e6 times the cutoff: ordinary, well-conditioned work
        log_ratio=st.floats(min_value=np.log(1e2), max_value=np.log(1e6)),
        seed=st.integers(min_value=0, max_value=2**31 - 1),
    )
    def test_an_ordinary_well_conditioned_problem_is_never_warned_about(
            self, n, m, log_ratio, seed):
        """A warning that fires routinely gets suppressed, which is
        worse than silence.  This is the property that keeps the other
        two worth having, and it is the constraint that fixed the
        factor: every widening past 2x multiplied the fire rate on
        verdicts float64 agrees with, reaching two thirds of ordinary
        5x..10x reports at the 8x first tried.
        """
        J64, _ = _fisher_with_known_ratio(n, max(m, n),
                                          float(np.exp(log_ratio)), seed)
        report, warned = _fim_of(J64)
        note(f"n={n} m={m} ratio/cutoff={np.exp(log_ratio):.4g} "
             f"rank={report.rank}")
        assert not warned
        assert report.rank == n

    def test_the_threshold_still_separates_the_two_populations(self):
        """A calibration gate, not a property: fixed seed, no
        hypothesis.

        ``_PRECISION_WARN_FACTOR`` is a measured number, and the thing
        that would silently rot is its *separation* -- someone widens it
        to catch one more case and it starts firing on ordinary work, or
        narrows it and it stops catching anything.  Neither shows up in
        a test that only asks whether a particular matrix warns.  So
        this one measures both rates over a fixed population and holds
        them to floors well inside the measured values.

        The population spans short and long residuals, which it could
        not before: ``m = 400`` puts the cutoff at ``sqrt(m) * eps``
        rather than ``n * eps`` for every ``n`` here, so a cutoff that
        stopped seeing the residual length would show up as a collapse
        in recall on the ``m = 400`` cells rather than as nothing at
        all.
        """
        rng = np.random.default_rng(20260919)
        dis = caught = far = fired = long_dis = 0
        for n in (2, 3, 5):
            for m in (12, 40, 400):
                cutoff = max(n, np.sqrt(m)) * _EPS32
                for _ in range(150):
                    rc = float(np.exp(rng.uniform(np.log(0.2), np.log(50.0))))
                    J64, _ = _fisher_with_known_ratio(
                        n, m, rc, int(rng.integers(0, 2**31 - 1)))
                    report, warned = _fim_of(J64)
                    rank64 = _rank_at(np.linalg.eigh(J64.T @ J64)[0], cutoff)
                    if report.rank != rank64:
                        dis += 1
                        caught += warned
                        long_dis += m > n * n
                    elif rc >= 5.0:
                        far += 1
                        fired += warned
        where = (f"disagreements={dis} caught={caught} "
                 f"far={far} fired={fired} long_residual={long_dis}")
        assert dis >= 5, f"population produced too few disagreements; {where}"
        # Measured 1.000 / 0.0000 over six seeds; the floors sit inside
        # that.  0.90 was 0.75 before the cutoff saw ``m``, when the
        # long-residual misses made a tighter floor unmeetable.
        assert caught / dis >= 0.90, where
        assert fired / max(far, 1) <= 0.02, where
        # Non-vacuity for the regime this gate was widened to cover: the
        # ``m > n**2`` cells must actually be producing close calls, or
        # the recall figure above is a statement about short residuals
        # wearing a longer population's name.  Three to six over six
        # seeds at calibration.
        assert long_dis >= 1, where
