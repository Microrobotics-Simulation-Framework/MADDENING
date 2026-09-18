"""Contract properties for ``maddening.sysid`` and the second calibration path.

``tests/verification/hypothesis/test_hypothesis_sysid.py`` covers the
*mathematics* of the fitting helpers (does the loss vanish at the truth,
is the null direction the one theory predicts).  This module covers the
*contract* the rest of the system leans on:

* **the trainable contract** -- :class:`~maddening.core.params.ParamSpec`
  ``trainable`` (and an explicit ``mask``) decides what an optimiser may
  move, over arbitrary generated graphs rather than one spring;
* **bounds and transforms** -- a fit that starts inside the declared
  bounds finishes inside them, the constrain/unconstrain maps invert,
  and a transform that needs particular bounds refuses to exist without
  them;
* **the Fisher matrix against a finite difference**, with the masked
  matrix a submatrix of the unmasked one and the noise model scaling it
  the way :func:`~maddening.sysid.fim` documents;
* **windowed and multiple-shooting losses** -- how the window count, the
  sampling interval and the continuity penalty compose, including the
  tilings that do not divide;
* **the second calibration path** --
  :func:`~maddening.core.simulation.calibration.calibrate` and
  :func:`~maddening.core.simulation.calibration.tune_coupling_params`,
  which nothing in ``src/`` imports and no document mentions, but which
  are still importable public functions.

Precision
---------
``test_fim_matches_a_central_finite_difference`` runs under
``jax_enable_x64`` on an *analytic* residual.  ``fim`` is agnostic to
what the residual computes, so the honest way to check its
differentiation is against an oracle that is itself precise: in float64
with a step scaled to the parameter, a central difference of a smooth
residual is accurate to ~1e-10 relative, which leaves a tolerance of
1e-7 three orders of headroom and still fails on a sign error, a lost
chain-rule factor, or a forward-difference mix-up (all O(1) or O(h)).

A graph rollout cannot be made precise that way: the nodes pin their
state and constants to float32 whatever ``jax_enable_x64`` says, so the
finite difference measures float32 cancellation as much as the
derivative.  ``test_fim_matches_a_finite_difference_of_a_rollout``
therefore states the weaker, measured claim -- 2e-2 relative to the
matrix's own scale, at a relative step of 3e-3 -- and says so here rather
than pretending float32 buys more.

That bound is set from a measured distribution, not a guess.  Over 126
configurations drawn from this test's own strategy the relative deviation
has median 1.6e-4 and p90 8.7e-4, with a long tail at high stiffness and
large initial displacement reaching 9.997e-3.  A 1e-2 bound therefore held
by 0.03%, which is not a bound -- it is a flake waiting for a draw that
CI had not yet made.  2e-2 leaves 2x headroom on the measured worst case
while still catching what this property exists to catch: a sign flip
shifts the matrix by 2|F_ds|/scale, min 5.2e-2 and median 5.8e-1 over the
same 126 configurations, so the sign-flip margin is 2.6x rather than 5x.
(A transposed Jacobian is not caught by any tolerance -- ``J.T @ J`` is
symmetric -- only by a shape error.)

Cost
----
Every property that builds or compiles a graph is ``EXAMPLES_COSTLY``;
the pure-pytree ones are ``EXAMPLES_CHEAP``.  Generated problems are
deliberately tiny (three-node graphs, rollouts of tens of steps, fits of
a handful of iterations): these properties are about the contract, not
about convergence, and the machine is shared.
"""

from __future__ import annotations

import contextlib
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.params import DEFAULT_SPEC, ParamSpec
from maddening.core.simulation.calibration import (
    CalibrateResult,
    TuneResult,
    calibrate,
    tune_coupling_params,
)
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import (
    fim,
    fit,
    fit_lm,
    fit_multiple_shooting,
    init_window_states,
    observations_from_history,
    windowed_loss,
)

from tests.conftest import EXAMPLES_CHEAP, EXAMPLES_COSTLY, EXAMPLES_STANDARD
from tests.property.strategies import graph_recipes

DT = 0.01
REST = 1.0


@contextlib.contextmanager
def x64_enabled():
    """``jax_enable_x64`` for the duration of the block.

    Same shape as ``tests/property/test_adaptive_invariants.py``: a
    context manager rather than a fixture, because a function-scoped
    fixture around a ``@given`` test is a Hypothesis health-check
    failure and a module-scoped one would drag every other property here
    into float64.
    """
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


def _finite(lo, hi):
    return st.floats(min_value=lo, max_value=hi, allow_nan=False,
                     allow_infinity=False)


def _finite_f32_normal(lo, hi):
    """``_finite`` restricted to values that are *normal in float32*.

    XLA's CPU backend runs with denormals flushed to zero, so for a
    float32 subnormal ``p`` even ``p - lr * 0.0`` evaluates to ``0.0``
    (``jnp.float32(1.6e-43) + 0.0`` is ``0.0`` here).  A *bit*-identity
    claim about a parameter no optimiser touched is therefore false in
    that corner for reasons that belong to the backend's floating-point
    mode rather than to the code under test, and false on CPU but not
    necessarily elsewhere.  ``allow_subnormal=False`` alone is not
    enough: it is read at the strategy's ``width``, so at the default
    64 it still yields ``1.3e-42`` -- an ordinary float64 and a float32
    subnormal.  ``width=32`` moves both the generation and the
    subnormal test to the precision these parameters are stored in.
    """
    return st.floats(min_value=lo, max_value=hi, allow_nan=False,
                     allow_infinity=False, width=32, allow_subnormal=False)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _leaf_paths(tree):
    return [jax.tree_util.keystr(p)
            for p, _ in jax.tree_util.tree_flatten_with_path(tree)[0]]


def _leaf_values(tree):
    return [np.asarray(leaf) for leaf in jax.tree.leaves(tree)]


def _moved(before, after):
    """Paths whose leaf is not bit-identical between two params pytrees."""
    return {
        path
        for path, b, a in zip(_leaf_paths(before), _leaf_values(before),
                              _leaf_values(after))
        if not np.array_equal(b, a)
    }


def _in_bounds(gm, params=None) -> bool:
    try:
        gm.check_params(params)
    except ValueError:
        return False
    return True


def _spec_leaves(gm):
    """``[(path_string, ParamSpec)]`` in ``jax.tree.leaves`` order."""
    specs = gm.param_specs()
    out = []
    for path, _ in jax.tree_util.tree_flatten_with_path(gm.params)[0]:
        node = specs
        for key in path:
            k = getattr(key, "key", None)
            node = node.get(k) if (k is not None and isinstance(node, dict)) else None
            if node is None:
                break
        out.append((jax.tree_util.keystr(path),
                    node if isinstance(node, ParamSpec) else DEFAULT_SPEC))
    return out


def _spring_gm(stiffness=30.0, damping=2.0, mass=1.0, position=0.5):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=stiffness, damping=damping,
                                 mass=mass, rest_length=REST,
                                 initial_position=position))
    gm.compile()
    return gm


def _with_params(gm, node, values):
    p = jax.tree.map(lambda x: x, gm.params)
    for k, v in values.items():
        p["nodes"][node][k] = jnp.asarray(v, dtype=jnp.float32)
    return p


def _observe(gm, n_steps, params, sample_every=1):
    init = {n: gm.get_node_state(n) for n in gm.node_names}
    _, hist = gm.run_scan_with_history(n_steps, params=params)
    obs = observations_from_history(init, hist)
    return jax.tree.map(lambda x: x[::sample_every], obs)


def _sum_squares(p):
    """A loss with a non-zero gradient in every direction, and no rollout.

    Cast to float32 because a generated graph may carry an integer leaf
    (a trainable matrix-mapping weight), which ``jnp.sum`` would keep
    integral.
    """
    total = jnp.float32(0.0)
    for leaf in jax.tree.leaves(p):
        total = total + jnp.sum((jnp.asarray(leaf, jnp.float32) - 1.5) ** 2)
    return total


# ---------------------------------------------------------------------------
# 1. The trainable contract
# ---------------------------------------------------------------------------


class TestTrainableContract:
    """What an optimiser is allowed to move, over arbitrary graphs.

    The downstream parameter-provenance work needs to be able to say
    "this constant was fitted, that one was not" from the specs alone,
    so the statement has to be general: for *any* graph and *any* mask
    that is a subset of the trainable set, the leaves outside the mask
    come back bit-identical and the leaves inside it are the only ones
    that may have moved.
    """

    @given(recipe=graph_recipes(max_nodes=3), data=st.data(),
           n_iter=st.integers(min_value=1, max_value=4))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_fit_moves_only_the_masked_trainable_leaves(self, recipe, data, n_iter):
        gm = recipe.build()
        if not _in_bounds(gm):
            # ``graph_recipes`` multiplies a "calibrated" leaf by up to
            # 2.0, which can push a bounded one (BallNode.elasticity is
            # declared ``bounds=(0, 1)``) outside its ParamSpec.  ``fit``
            # rightly refuses such a start; fit the constructor values.
            gm.reset_params()
        assume(_in_bounds(gm))

        trainable = jax.tree.leaves(gm.trainable_mask())
        picked = data.draw(st.lists(st.booleans(), min_size=len(trainable),
                                    max_size=len(trainable)))
        flags = [bool(t and p) for t, p in zip(trainable, picked)]
        assume(any(flags))
        mask = jax.tree_util.tree_unflatten(
            jax.tree_util.tree_structure(gm.params), flags)
        masked = {path for path, f in zip(_leaf_paths(gm.params), flags) if f}
        note(f"masked={sorted(masked)}")

        before = jax.tree.map(lambda x: np.array(x), gm.params)
        result = fit(gm, _sum_squares, mask=mask, n_iter=n_iter, lr=0.05)

        # Structure and dtypes are part of the params contract: the
        # result must drop straight back into ``gm.run_scan(params=...)``.
        assert jax.tree_util.tree_structure(result.params) == \
            jax.tree_util.tree_structure(gm.params)
        for b, a in zip(jax.tree.leaves(gm.params), jax.tree.leaves(result.params)):
            assert jnp.asarray(b).dtype == jnp.asarray(a).dtype
            assert jnp.asarray(b).shape == jnp.asarray(a).shape

        # Only the masked leaves move, and they are the only ones that
        # differ *at all*: a leaf outside the mask is copied from the
        # starting pytree rather than round-tripped through
        # ``unconstrain``/``constrain`` (see
        # ``test_a_leaf_outside_the_mask_is_bit_identical_after_a_fit``),
        # which subsumes the weaker claim about the frozen leaves.
        moved = _moved(before, result.params)
        assert moved <= masked, sorted(moved - masked)
        # The result is still a legal starting point.
        assert _in_bounds(gm, result.params)

        # Non-vacuity: a masked leaf that is float-valued and strictly
        # inside its bounds has nowhere to be clipped to, so Adam's first
        # step must move it.
        movable = {
            path
            for (path, spec), f, leaf in zip(_spec_leaves(gm), flags,
                                             jax.tree.leaves(before))
            if f and jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)
            and _strictly_inside(spec, leaf)
        }
        if movable:
            assert moved, f"nothing moved although {sorted(movable)} could"

    # Parametrised rather than drawn because there is nothing to
    # generate: the point is one leaf at one value in each fitter.  A
    # plain method takes ``@pytest.mark.parametrize`` happily; only the
    # ``@given`` methods below have to draw the fitter instead.
    @pytest.mark.parametrize("fitter", ["adam", "lm", "multiple_shooting"])
    def test_a_leaf_outside_the_mask_is_bit_identical_after_a_fit(self, fitter):
        """A leaf the mask excluded comes back exactly, not merely close.

        Every fitter used to return ``gm.constrain(unravel(...))`` over
        the WHOLE tree, and for a ``log`` or ``logit`` leaf that round
        trip is ``exp(log(p))`` in float32, exact only to about one ulp,
        so a transformed trainable leaf came back perturbed even when
        the mask excluded it (``stiffness`` 30.0 -> 30.000001907348633).
        A provenance claim of the form "these constants were not fitted,
        here are their bits" needs the value itself back, so the fitters
        copy an unmasked leaf straight from the starting pytree.  All
        three share that map: a fix in one of them would be the bug in a
        new place.
        """
        gm = _spring_gm()
        # ``SpringDamperNode`` declares *stiffness* and *mass* as positive
        # constants (``transform='log'``); ``damping`` is bounded below but
        # carries no transform, and an untransformed leaf round-trips
        # exactly -- so damping is the one spring constant that could not
        # show this.  The leaf under test must also be a value ``exp(log
        # .))`` moves: 1.0 and 2.5 are fixed points in float32, 30.0 is not
        # (it came back as 30.000001907348633).
        specs = gm.param_specs()["nodes"]["s"]
        assert specs["stiffness"].transform == "log"
        assert specs["damping"].transform is None
        mask = jax.tree.map(lambda _: False, gm.params)
        mask["nodes"]["s"]["damping"] = True
        before = float(gm.params["nodes"]["s"]["stiffness"])
        assert before == 30.0
        before_damping = float(gm.params["nodes"]["s"]["damping"])

        if fitter == "adam":
            result = fit(gm, lambda p: p["nodes"]["s"]["damping"] ** 2,
                         mask=mask, n_iter=1, lr=0.05)
        elif fitter == "lm":
            result = fit_lm(
                gm, lambda p: jnp.atleast_1d(p["nodes"]["s"]["damping"] - 3.0),
                mask=mask, n_iter=2)
        else:
            # Observations of a *more* damped spring, so the fit has a
            # reason to move ``damping`` away from where it starts.
            obs = _observe(gm, 12, _with_params(gm, "s", {"damping": 4.0}))
            result, _ = fit_multiple_shooting(
                gm, obs, obs_fn=(lambda h: h["s"]["position"]), window=4,
                mask=mask, n_iter=2, lr=0.05)

        assert float(result.params["nodes"]["s"]["stiffness"]) == before
        # Non-vacuity: the masked leaf did move, so the fit was not a
        # no-op that would leave every leaf untouched for free.
        assert float(result.params["nodes"]["s"]["damping"]) != before_damping

    @given(recipe=graph_recipes(max_nodes=3))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_fit_defaults_its_mask_to_the_trainable_set(self, recipe):
        """With no ``mask``, a ``trainable=False`` leaf is bit-identical."""
        gm = recipe.build()
        if not _in_bounds(gm):
            gm.reset_params()
        assume(_in_bounds(gm))
        frozen = {path for path, spec in _spec_leaves(gm) if not spec.trainable}
        assume(frozen)
        # A graph whose every leaf is frozen (two TableNodes: ``position``
        # is an initial condition) has nothing to fit at all, and ``fit``
        # says so -- correctly, but it is not this property's subject.
        assume(any(jax.tree.leaves(gm.trainable_mask())))
        note(f"frozen={sorted(frozen)}")

        before = jax.tree.map(lambda x: np.array(x), gm.params)
        result = fit(gm, _sum_squares, n_iter=2, lr=0.05)
        assert not (_moved(before, result.params) & frozen)

    # ``fitter`` is drawn rather than ``@pytest.mark.parametrize``\ d: a
    # parametrised ``@given`` *method* gets a fresh class instance per
    # case, which Hypothesis rejects as ``HealthCheck.differing_executors``.
    @given(fitter=st.sampled_from(["adam", "lm", "multiple_shooting"]),
           freeze=st.lists(st.sampled_from(["stiffness", "damping", "mass",
                                            "rest_length"]),
                           min_size=1, max_size=3, unique=True))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_every_fitter_honours_the_same_frozen_set(self, fitter, freeze):
        """``fit``, ``fit_lm`` and ``fit_multiple_shooting`` share
        ``_masked_indices``; all three must leave a frozen leaf alone."""
        gm = _spring_gm()
        for key in freeze:
            gm.set_param_spec("s", key, ParamSpec(trainable=False))
        obs = _observe(gm, 12, gm.params)
        start = _with_params(gm, "s", {"stiffness": 45.0, "damping": 3.0})
        before = {k: float(v) for k, v in start["nodes"]["s"].items()}
        note(f"freeze={freeze}")

        obs_fn = (lambda h: h["s"]["position"])
        if fitter == "adam":
            res = fit(gm, lambda p: windowed_loss(
                gm, p, obs, obs_fn=obs_fn, window=4), params=start, n_iter=3, lr=0.05)
        elif fitter == "lm":
            res = fit_lm(gm, lambda p: _rollout_residual(gm, obs)(p),
                         params=start, n_iter=2)
        else:
            res, _ = fit_multiple_shooting(
                gm, obs, obs_fn=obs_fn, window=4, params=start, n_iter=3, lr=0.05)

        after = {k: float(v) for k, v in res.params["nodes"]["s"].items()}
        for key in freeze:
            assert after[key] == before[key], (key, before[key], after[key])
        # initial conditions are declared non-trainable by the node itself
        assert after["initial_position"] == before["initial_position"]

    @given(fitter=st.sampled_from(["adam", "lm", "multiple_shooting"]))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_a_mask_may_not_widen_the_trainable_set(self, fitter):
        """Every fitter refuses a ``mask`` that contradicts a frozen spec.

        ``unconstrain``/``constrain`` apply a leaf's transform and bounds
        only when its :class:`ParamSpec` is trainable, so a mask that
        *widened* the trainable set had the optimiser step the leaf in
        physical coordinates with nothing clipping it: ``damping`` declared
        ``bounds=(0, 1)`` reached 6.5 and ``fit`` returned a pytree its own
        ``check_params`` rejects.  The spec is the single source of truth
        for what may move, so the mask is refused rather than allowed to
        route around it -- and the error names the leaf, because the fix is
        to change that leaf's spec.
        """
        gm = _spring_gm()
        gm.set_param_spec("s", "damping",
                          ParamSpec(trainable=False, bounds=(0.0, 1.0)))
        mask = jax.tree.map(lambda _: False, gm.params)
        mask["nodes"]["s"]["damping"] = True
        start = _with_params(gm, "s", {"damping": 0.5})
        obs = _observe(gm, 12, start)
        obs_fn = (lambda h: h["s"]["position"])

        with pytest.raises(ValueError) as excinfo:
            if fitter == "adam":
                fit(gm, lambda p: -10.0 * p["nodes"]["s"]["damping"],
                    params=start, mask=mask, n_iter=40, lr=0.1)
            elif fitter == "lm":
                fit_lm(gm, lambda p: p["nodes"]["s"]["damping"][None],
                       params=start, mask=mask, n_iter=2)
            else:
                fit_multiple_shooting(gm, obs, obs_fn=obs_fn, window=4,
                                      params=start, mask=mask, n_iter=2)
        message = str(excinfo.value)
        # Actionable without reading the source: which leaf, what the spec
        # says, and that the spec -- not the mask -- is where to change it.
        assert "damping" in message and "'s'" in message, message
        assert "trainable=False" in message, message
        assert "set_param_spec" in message, message

    def test_a_frozen_leaf_made_trainable_in_the_spec_respects_its_bounds(self):
        """The fix the refusal advertises actually activates the bounds.

        This is the invariant the mask used to break, restated against the
        contract that replaced it: with ``trainable=True`` in the spec the
        bounded leaf is genuinely clipped, so a loss that pulls on it
        forever still leaves a pytree ``check_params`` accepts.
        """
        gm = _spring_gm()
        # ``bounds=(0, 1)`` with no transform: ``constrain`` clips a
        # trainable leaf into the interval, and nothing clips a frozen one.
        gm.set_param_spec("s", "damping",
                          ParamSpec(trainable=True, bounds=(0.0, 1.0)))
        mask = jax.tree.map(lambda _: False, gm.params)
        mask["nodes"]["s"]["damping"] = True
        start = _with_params(gm, "s", {"damping": 0.5})
        res = fit(gm, lambda p: -10.0 * p["nodes"]["s"]["damping"],
                  params=start, mask=mask, n_iter=40, lr=0.1)
        gm.check_params(res.params)
        assert 0.0 <= float(res.params["nodes"]["s"]["damping"]) <= 1.0


def _strictly_inside(spec: ParamSpec, leaf) -> bool:
    v = np.asarray(leaf, dtype=np.float64)
    lo, hi = spec.bounds
    if lo is not None and np.any(v <= lo + 1e-6 * (1.0 + abs(lo))):
        return False
    if hi is not None and np.any(v >= hi - 1e-6 * (1.0 + abs(hi))):
        return False
    return True


def _rollout_residual(gm, obs, node="s"):
    """``params -> simulated - measured`` position over the observations."""
    step_fn = gm._build_step_fn()          # noqa: SLF001
    ext = gm._default_external_inputs()    # noqa: SLF001
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs[node]["position"][1:]
    n = int(truth.shape[0])

    def residual(p):
        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s[node]["position"]
        return jax.lax.scan(body, init, None, length=n)[1] - truth

    return residual


# ---------------------------------------------------------------------------
# 2. Bounds and transforms
# ---------------------------------------------------------------------------


@st.composite
def spec_and_value(draw):
    """A constructible ``ParamSpec`` and a value strictly inside its bounds.

    The intervals are kept wide (at least 1e-2, and wide relative to
    their endpoints) because ``to_constrained`` documents that an
    interval only a few float32 ulps across has no interior to clamp to
    -- that degenerate case is owned by
    ``tests/core/test_param_spec_edge_cases.py``, not by a general
    round-trip property.
    """
    transform = draw(st.sampled_from([None, "log", "logit"]))
    lo = draw(_finite(-20.0, 20.0))
    span = draw(_finite(1e-2, 40.0))
    if transform == "logit":
        bounds = (lo, lo + span)
        value = lo + span * draw(_finite(0.05, 0.95))
    elif transform == "log":
        bounds = (draw(st.sampled_from([None, lo])), None)
        base = 0.0 if bounds[0] is None else bounds[0]
        value = base + draw(_finite(1e-3, 40.0))
    else:
        kind = draw(st.sampled_from(["none", "lower", "upper", "both"]))
        if kind == "none":
            bounds, value = (None, None), draw(_finite(-20.0, 20.0))
        elif kind == "lower":
            bounds, value = (lo, None), lo + draw(_finite(0.0, 40.0))
        elif kind == "upper":
            bounds, value = (None, lo), lo - draw(_finite(0.0, 40.0))
        else:
            bounds = (lo, lo + span)
            value = lo + span * draw(_finite(0.0, 1.0))
    return ParamSpec(bounds=bounds, transform=transform), float(np.float32(value))


class TestBoundsAndTransforms:

    @given(st.data())
    @settings(max_examples=EXAMPLES_CHEAP, deadline=None)
    def test_a_transform_refuses_the_bounds_it_cannot_work_with(self, data):
        """``logit`` needs both bounds, ``log`` refuses an upper one, and
        no spec accepts ``lo >= hi``."""
        lo = data.draw(_finite(-20.0, 20.0))
        hi = lo + data.draw(_finite(1e-3, 40.0))

        for bounds in [(None, None), (lo, None), (None, hi)]:
            with pytest.raises(ValueError, match="logit"):
                ParamSpec(transform="logit", bounds=bounds)
        ParamSpec(transform="logit", bounds=(lo, hi))          # the legal one

        with pytest.raises(ValueError, match="log"):
            ParamSpec(transform="log", bounds=(lo, hi))
        with pytest.raises(ValueError, match="log"):
            ParamSpec(transform="log", bounds=(None, hi))
        ParamSpec(transform="log", bounds=(lo, None))

        with pytest.raises(ValueError, match="lo < hi"):
            ParamSpec(bounds=(hi, lo))
        with pytest.raises(ValueError, match="lo < hi"):
            ParamSpec(bounds=(lo, lo))
        with pytest.raises(ValueError, match="transform"):
            ParamSpec(transform="softplus")

    @given(spec_value=spec_and_value())
    @settings(max_examples=EXAMPLES_CHEAP, deadline=None)
    def test_constrain_inverts_unconstrain_inside_the_bounds(self, spec_value):
        spec, value = spec_value
        note(f"spec={spec} value={value}")
        v = jnp.float32(value)
        spec.check(v)                       # the draw really is a legal value
        round_trip = float(spec.to_constrained(spec.to_unconstrained(v)))
        # float32 through log/exp or logit/sigmoid: the error is relative
        # to the distance from the bound the transform pivots on, which is
        # what ``value - lo`` measures.
        lo = 0.0 if spec.bounds[0] is None else spec.bounds[0]
        scale = max(1.0, abs(value), abs(value - lo))
        assert abs(round_trip - value) <= 1e-4 * scale, (round_trip, value)

    @given(spec_value=spec_and_value(), u=_finite(-1e4, 1e4))
    @settings(max_examples=EXAMPLES_CHEAP, deadline=None)
    def test_constrain_lands_inside_the_bounds_from_any_coordinate(
        self, spec_value, u,
    ):
        """An optimiser may hand ``constrain`` anything; what comes back
        is a value ``check`` accepts."""
        spec, _ = spec_value
        note(f"spec={spec} u={u}")
        spec.check(spec.to_constrained(jnp.float32(u)))

    @given(bounds=st.tuples(_finite(0.5, 5.0), _finite(6.0, 40.0)),
           transform=st.sampled_from([None, "log", "logit"]),
           n_iter=st.integers(min_value=1, max_value=6))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_a_fit_that_starts_inside_the_bounds_finishes_inside_them(
        self, bounds, transform, n_iter,
    ):
        """For any starting point inside the bounds and any number of
        iterations, ``fit``'s result is inside them -- whatever the
        transform, and however hard the loss pulls outwards."""
        lo, hi = bounds
        spec = ParamSpec(bounds=(lo, None) if transform == "log" else (lo, hi),
                         transform=transform)
        start_value = 0.5 * (lo + hi)
        gm = _spring_gm(stiffness=start_value)
        gm.set_param_spec("s", "stiffness", spec)
        for key in ("damping", "mass", "rest_length"):
            gm.set_param_spec("s", key, ParamSpec(trainable=False))
        note(f"spec={spec} start={start_value} n_iter={n_iter}")

        for direction in (-1.0, 1.0):
            # A linear loss: the gradient never vanishes, so Adam keeps
            # pushing at the bound for as long as it is given.
            res = fit(gm, lambda p: direction * 50.0 * p["nodes"]["s"]["stiffness"],
                      n_iter=n_iter, lr=0.5)
            gm.check_params(res.params)
            value = float(res.params["nodes"]["s"]["stiffness"])
            assert np.isfinite(value)
            if spec.bounds[1] is not None:
                assert value <= spec.bounds[1]
            assert value >= spec.bounds[0]


# ---------------------------------------------------------------------------
# 3. The Fisher matrix against a finite difference
# ---------------------------------------------------------------------------


@st.composite
def analytic_residual(draw):
    """A smooth, well-scaled ``params -> residual`` map and a point.

    ``sum_j a_ij sin(b_ij theta_j)`` has bounded derivatives of every
    order at unit scale, which is what makes a central difference a
    *precise* oracle rather than a noisy one.
    """
    n_par = draw(st.integers(min_value=2, max_value=4))
    n_res = draw(st.integers(min_value=3, max_value=8))
    coeff = draw(st.lists(_finite(-1.0, 1.0), min_size=n_res * n_par,
                          max_size=n_res * n_par))
    freq = draw(st.lists(_finite(0.5, 1.5), min_size=n_res * n_par,
                         max_size=n_res * n_par))
    theta = draw(st.lists(_finite(0.5, 2.0), min_size=n_par, max_size=n_par))
    keys = tuple(f"p{i}" for i in range(n_par))
    A = np.asarray(coeff, dtype=np.float64).reshape(n_res, n_par)
    B = np.asarray(freq, dtype=np.float64).reshape(n_res, n_par)

    def residual(p):
        th = jnp.stack([p[k] for k in keys])
        return jnp.sum(jnp.asarray(A) * jnp.sin(jnp.asarray(B) * th[None, :]),
                       axis=1)

    return residual, keys, tuple(theta)


class TestFIMAgainstFiniteDifference:

    @given(problem=analytic_residual())
    @settings(max_examples=EXAMPLES_STANDARD, deadline=None)
    def test_fim_matches_a_central_finite_difference(self, problem):
        residual, keys, theta = problem
        note(f"theta={theta}")
        with x64_enabled():
            params = {k: jnp.asarray(v, jnp.float64)
                      for k, v in zip(keys, theta)}
            report = fim(residual, params, scale=None)

            order = sorted(keys)                 # ``ravel_pytree`` sorts dicts
            jac = np.zeros((len(np.asarray(residual(params))), len(order)))
            for j, key in enumerate(order):
                h = 1e-5 * (1.0 + abs(float(params[key])))
                plus = {**params, key: params[key] + h}
                minus = {**params, key: params[key] - h}
                jac[:, j] = (np.asarray(residual(plus), np.float64)
                             - np.asarray(residual(minus), np.float64)) / (2 * h)

            F = np.asarray(report.fim, dtype=np.float64)
            F_fd = jac.T @ jac
            scale = max(1.0, float(np.abs(F).max()))
            assert np.allclose(F, F_fd, rtol=1e-7, atol=1e-7 * scale), (F, F_fd)

    @given(truth=st.fixed_dictionaries({
        "stiffness": _finite(1.0, 200.0),
        "damping": _finite(0.1, 10.0),
        "mass": _finite(0.5, 5.0),
    }), init=st.fixed_dictionaries({
        "position": _finite(-5.0, 5.0),
        "velocity": _finite(-2.0, 2.0),
    }), n=st.sampled_from((20, 40)))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_fim_matches_a_finite_difference_of_a_rollout(self, truth, init, n):
        """The same check against a real graph rollout, at the tolerance
        float32 actually supports (module docstring)."""
        gm = _spring_gm()
        gm.set_node_state("s", {"position": jnp.float32(init["position"]),
                                "velocity": jnp.float32(init["velocity"])})
        p_truth = _with_params(gm, "s", truth)
        obs = _observe(gm, n, p_truth)
        assume(float(jnp.var(obs["s"]["position"])) > 1e-2)
        note(f"truth={truth} init={init} n={n}")

        names = ("damping", "stiffness")     # ``ravel_pytree`` order
        base = {k: p_truth["nodes"]["s"][k] for k in names}
        residual = _sub_residual(gm, obs, p_truth, names)
        F = np.asarray(fim(residual, base, scale=None).fim, dtype=np.float64)

        jac = np.zeros((n, len(names)))
        for j, key in enumerate(names):
            h = 3e-3 * (1.0 + abs(float(base[key])))
            plus = {**base, key: jnp.float32(float(base[key]) + h)}
            minus = {**base, key: jnp.float32(float(base[key]) - h)}
            # The float32 round trip moves the endpoints; difference over
            # the step that was actually taken, not the one asked for.
            step = 0.5 * (float(plus[key]) - float(minus[key]))
            assume(step > 0.0)
            jac[:, j] = (np.asarray(residual(plus), np.float64)
                         - np.asarray(residual(minus), np.float64)) / (2 * step)

        F_fd = jac.T @ jac
        scale = float(np.abs(F).max())
        assume(scale > 0.0)
        assert np.abs(F - F_fd).max() <= 2e-2 * scale, (F, F_fd)

    @given(problem=analytic_residual())
    @settings(max_examples=EXAMPLES_STANDARD, deadline=None)
    def test_crb_is_consistent_with_the_matrix_it_came_from(self, problem):
        """For a non-singular FIM the reported bound is the diagonal of the
        inverse, and the Cramér--Rao inequality ``crb_i >= 1 / F_ii`` holds
        (a parameter is never easier to estimate jointly than alone)."""
        residual, keys, theta = problem
        with x64_enabled():
            params = {k: jnp.asarray(v, jnp.float64)
                      for k, v in zip(keys, theta)}
            report = fim(residual, params, scale=None)
            F = np.asarray(report.fim, dtype=np.float64)
            crb = np.asarray(report.crb, dtype=np.float64)
            diag = np.diag(F)
            assume(np.all(diag > 1e-8))
            assume(np.isfinite(report.cond) and report.cond < 1e8)
            note(f"cond={report.cond}")
            assert np.allclose(crb, np.diag(np.linalg.inv(F)), rtol=1e-6,
                               atol=1e-9)
            assert np.all(crb >= (1.0 / diag) * (1 - 1e-6)), (crb, 1.0 / diag)
            assert np.all(crb > 0.0)

    @pytest.mark.xfail(strict=True, reason=(
        "``fim`` documents ``crb`` as 'NaN where the FIM is singular', but "
        "it is ``diag(pinv(F))``: along an exact null direction the "
        "pseudo-inverse is finite and small, so an entirely unidentifiable "
        "parameter is reported with a *tight* variance bound instead of an "
        "infinite one.  Here the residual depends only on ``a + b``, "
        "``eigvals[0]`` is exactly 0 and ``cond`` is inf, yet crb is "
        "~0.008 for both.  Uncertainty quantification built on this output "
        "would understate the variance by an unbounded factor.  The fix is "
        "an API decision (NaN, +inf, or a documented pseudo-inverse with a "
        "rank field), so the behaviour is pinned rather than changed."))
    def test_crb_is_not_finite_along_an_exact_null_direction(self):
        def residual(p):
            t = jnp.arange(5, dtype=jnp.float32)
            return (p["a"] + p["b"]) * t

        report = fim(residual, {"a": jnp.float32(1.0), "b": jnp.float32(2.0)},
                     scale=None)
        assert report.cond == float("inf")
        assert float(report.eigvals[0]) == 0.0
        assert not bool(jnp.all(jnp.isfinite(report.crb))), report.crb


def _sub_residual(gm, obs, base_params, names, node="s"):
    """``{name: value} -> simulated - measured`` for a subset of a node's
    constants, the shape ``fim`` is normally called with."""
    step_fn = gm._build_step_fn()          # noqa: SLF001
    ext = gm._default_external_inputs()    # noqa: SLF001
    init = jax.tree.map(lambda x: x[0], obs)
    truth = obs[node]["position"][1:]
    n = int(truth.shape[0])

    def residual(sub):
        p = jax.tree.map(lambda x: x, base_params)
        for key, value in sub.items():
            p["nodes"][node][key] = value

        def body(s, _):
            s = step_fn(s, ext, p)
            return s, s[node]["position"]

        return jax.lax.scan(body, init, None, length=n)[1] - truth

    return residual


# ---------------------------------------------------------------------------
# 4. Masking and scaling
# ---------------------------------------------------------------------------


class TestFIMMaskingAndScaling:

    @given(problem=analytic_residual(), data=st.data(),
           scale=st.sampled_from([None, "relative"]))
    @settings(max_examples=EXAMPLES_STANDARD, deadline=None)
    def test_the_masked_matrix_is_the_submatrix_of_the_unmasked_one(
        self, problem, data, scale,
    ):
        residual, keys, theta = problem
        order = sorted(keys)
        flags = data.draw(st.lists(st.booleans(), min_size=len(order),
                                   max_size=len(order)))
        assume(any(flags))
        note(f"order={order} flags={flags} scale={scale}")

        params = {k: jnp.asarray(v, jnp.float32) for k, v in zip(keys, theta)}
        mask = {k: flags[order.index(k)] for k in keys}
        full = fim(residual, params, scale=scale)
        part = fim(residual, params, scale=scale, mask=mask)

        idx = [i for i, f in enumerate(flags) if f]
        expect = np.asarray(full.fim, dtype=np.float64)[np.ix_(idx, idx)]
        got = np.asarray(part.fim, dtype=np.float64)
        assert got.shape == (len(idx), len(idx))
        assert part.param_names == tuple(full.param_names[i] for i in idx)
        assert np.allclose(got, expect, rtol=1e-5,
                           atol=1e-6 * (1 + np.abs(expect).max()))

    def test_a_mask_that_selects_nothing_is_refused(self):
        def residual(p):
            return p["a"] * jnp.arange(4, dtype=jnp.float32)

        params = {"a": jnp.float32(2.0), "b": jnp.float32(3.0)}
        with pytest.raises(ValueError, match="selects no parameters"):
            fim(residual, params, mask={"a": False, "b": False})
        with pytest.raises(ValueError, match="same tree structure"):
            fim(residual, params, mask={"a": True})

    @given(problem=analytic_residual(), sigma=_finite(0.25, 4.0))
    @settings(max_examples=EXAMPLES_STANDARD, deadline=None)
    def test_the_noise_model_scales_the_information_by_one_over_sigma_squared(
        self, problem, sigma,
    ):
        """``F = J^T Sigma^-1 J``: a scalar sigma divides the matrix by
        sigma^2 and multiplies the CRB by it, and the same sigma spelled
        as a per-entry pytree gives the same matrix."""
        residual, keys, theta = problem
        note(f"sigma={sigma}")
        params = {k: jnp.asarray(v, jnp.float32) for k, v in zip(keys, theta)}
        base = fim(residual, params)
        scaled = fim(residual, params, noise_std=sigma)
        n_res = int(np.asarray(residual(params)).size)
        per_entry = fim(residual, params,
                        noise_std=jnp.full((n_res,), sigma, jnp.float32))

        b = np.asarray(base.fim, dtype=np.float64)
        s = np.asarray(scaled.fim, dtype=np.float64)
        assert np.allclose(s, b / sigma**2, rtol=1e-4,
                           atol=1e-6 * (1 + np.abs(s).max()))
        assert np.allclose(np.asarray(per_entry.fim, dtype=np.float64), s,
                           rtol=1e-5, atol=1e-7 * (1 + np.abs(s).max()))

        crb_b = np.asarray(base.crb, dtype=np.float64)
        crb_s = np.asarray(scaled.crb, dtype=np.float64)
        # Only where the unscaled matrix is comfortably invertible.  A
        # near-singular float32 ``eigh`` is not scale-invariant: dividing
        # such a matrix by sigma^2 can round its smallest eigenvalue to
        # zero, which turns a large finite ``cond`` into ``inf`` and makes
        # ``pinv`` drop a direction.  That is arithmetic, not the noise
        # model, and the identifiability properties in
        # ``tests/verification/hypothesis/test_hypothesis_sysid.py`` own it.
        assume(np.isfinite(base.cond) and base.cond < 1e4)
        assume(np.all(np.isfinite(crb_b)))
        # A uniform rescaling cannot change which direction is weakest.
        # ``cond`` is a ratio of float32 ``eigh`` outputs, so the two runs
        # agree to a few parts in a thousand, not to round-off.
        assert np.isclose(scaled.cond, base.cond, rtol=1e-2), (
            scaled.cond, base.cond)
        assert np.allclose(crb_s, crb_b * sigma**2, rtol=1e-3,
                           atol=1e-9 * (1 + np.abs(crb_s).max()))


# ---------------------------------------------------------------------------
# 5. Windowed and multiple-shooting losses
# ---------------------------------------------------------------------------


def _divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


#: ``(n_steps, sample_every, window)`` triples that tile exactly.
_TILINGS = tuple(
    (n, se, w)
    for n in (12, 16, 24, 36)
    for se in (1, 2, 3)
    if n % se == 0
    for w in _divisors(n // se)
)


class TestWindowTilings:

    @given(tiling=st.sampled_from(_TILINGS))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_a_tiling_that_divides_is_accepted_by_both_helpers(self, tiling):
        n_steps, sample_every, window = tiling
        gm = _spring_gm()
        obs = _observe(gm, n_steps, gm.params, sample_every)
        T = int(obs["s"]["position"].shape[0])
        n_windows = (T - 1) // window
        note(f"tiling={tiling} T={T} n_windows={n_windows}")
        assert (T - 1) % window == 0

        value = float(windowed_loss(gm, gm.params, obs,
                                    obs_fn=lambda h: h["s"]["position"],
                                    window=window, sample_every=sample_every))
        assert np.isfinite(value) and value >= 0.0
        ws = init_window_states(obs, window)
        assert ws["s"]["position"].shape[0] == n_windows

    @given(n_steps=st.sampled_from((12, 16, 24)),
           window=st.integers(min_value=1, max_value=30))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_a_tiling_that_does_not_divide_is_refused_by_both_helpers(
        self, n_steps, window,
    ):
        """The awkward cases: a window that leaves a remainder, a
        non-positive window, and a window wider than the data (which
        divides ``T - 1 == 0`` arithmetically but leaves nothing to
        integrate)."""
        gm = _spring_gm()
        obs = _observe(gm, n_steps, gm.params)
        T = int(obs["s"]["position"].shape[0])
        assume((T - 1) % window != 0)
        note(f"T={T} window={window}")
        for bad in (window, 0, -window):
            with pytest.raises(ValueError, match="window"):
                windowed_loss(gm, gm.params, obs,
                              obs_fn=lambda h: h["s"]["position"], window=bad)
            with pytest.raises(ValueError, match="window"):
                init_window_states(obs, bad)

    def test_a_window_wider_than_the_data_is_refused(self):
        gm = _spring_gm()
        obs = _observe(gm, 4, gm.params)
        single = jax.tree.map(lambda x: x[:1], obs)   # T == 1: nothing to fit
        for window in (1, 5):
            with pytest.raises(ValueError, match="window"):
                windowed_loss(gm, gm.params, single,
                              obs_fn=lambda h: h["s"]["position"], window=window)
            with pytest.raises(ValueError, match="window"):
                init_window_states(single, window)

    @given(sample_every=st.integers(min_value=-3, max_value=0))
    @settings(max_examples=EXAMPLES_CHEAP, deadline=None)
    def test_a_non_positive_sampling_interval_is_refused(self, sample_every):
        """``sample_every <= 0`` used to run ``lax.scan`` for zero steps
        per sample and return a plausible-looking number computed from the
        window's initial state repeated ``window`` times."""
        gm = _spring_gm()
        obs = _observe(gm, 8, gm.params)
        with pytest.raises(ValueError, match="sample_every"):
            windowed_loss(gm, gm.params, obs,
                          obs_fn=lambda h: h["s"]["position"], window=4,
                          sample_every=sample_every)


class TestMultipleShootingLoss:

    @given(tiling=st.sampled_from([t for t in _TILINGS if t[2] < t[0] // t[1]]),
           weight=_finite(1e-3, 1e4))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_the_continuity_penalty_is_affine_in_its_weight(self, tiling, weight):
        """``windowed_loss`` documents the penalty as
        ``weight * sum_w ||end_w - window_states[w+1]||^2``.  So the loss
        is ``data_term + weight * P`` with one ``P >= 0`` that does not
        depend on the weight -- which is what makes the weight a knob
        rather than a reparametrisation."""
        n_steps, sample_every, window = tiling
        gm = _spring_gm()
        obs = _observe(gm, n_steps, gm.params, sample_every)
        ws = init_window_states(obs, window)
        off = _with_params(gm, "s", {"stiffness": 48.0})
        note(f"tiling={tiling} weight={weight}")

        def loss(w):
            return float(windowed_loss(
                gm, off, obs, obs_fn=lambda h: h["s"]["position"],
                window=window, sample_every=sample_every, window_states=ws,
                continuity_weight=w))

        data_only = loss(0.0)
        teacher = float(windowed_loss(
            gm, off, obs, obs_fn=lambda h: h["s"]["position"], window=window,
            sample_every=sample_every))
        # Seeded from the measured window starts, multiple shooting *is*
        # teacher forcing plus the penalty.
        assert np.isclose(data_only, teacher, rtol=1e-5, atol=1e-8)

        one, two = loss(weight), loss(2.0 * weight)
        penalty = one - data_only
        assert penalty >= -1e-6 * (1.0 + abs(data_only)), penalty
        assert np.isclose(two - data_only, 2.0 * penalty,
                          rtol=1e-4, atol=1e-6 * (1.0 + abs(two)))

    @given(tiling=st.sampled_from([t for t in _TILINGS if t[2] < t[0] // t[1]]),
           weight=st.sampled_from([0.0, 1.0, 1e3, 1e6]))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_truth_seeded_windows_recover_the_unwindowed_loss(
        self, tiling, weight,
    ):
        """At the true parameters, windows seeded from the true trajectory
        are continuous, so the penalty vanishes *however large its weight*
        and the windowed loss is the single-window (unwindowed) loss.

        Away from the truth this is false by construction -- each window
        restarts from the data, so the penalty is a real extra term -- and
        the property above measures that term instead.
        """
        n_steps, sample_every, window = tiling
        gm = _spring_gm()
        truth = gm.params
        obs = _observe(gm, n_steps, truth, sample_every)
        ws = init_window_states(obs, window)
        T = int(obs["s"]["position"].shape[0])
        note(f"tiling={tiling} weight={weight} T={T}")

        obs_fn = (lambda h: h["s"]["position"])
        unwindowed = float(windowed_loss(gm, truth, obs, obs_fn=obs_fn,
                                         window=T - 1, sample_every=sample_every))
        shooting = float(windowed_loss(
            gm, truth, obs, obs_fn=obs_fn, window=window,
            sample_every=sample_every, window_states=ws,
            continuity_weight=weight))
        energy = float(jnp.sum(obs["s"]["position"] ** 2))
        # float32 round-off per sample, amplified by the penalty weight.
        tol = 1e-9 * (1.0 + energy) * (1.0 + weight)
        assert 0.0 <= unwindowed <= 1e-9 * (1.0 + energy), unwindowed
        assert abs(shooting - unwindowed) <= tol, (shooting, unwindowed)

    @given(tiling=st.sampled_from([t for t in _TILINGS if t[2] < t[0] // t[1]]),
           bump=_finite(0.1, 1.0))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_a_discontinuous_window_start_costs_the_penalty(self, tiling, bump):
        """The penalty is the thing that makes the fitted trajectory one
        solution: move a window start off the truth and it must be paid."""
        n_steps, sample_every, window = tiling
        gm = _spring_gm()
        obs = _observe(gm, n_steps, gm.params, sample_every)
        ws = init_window_states(obs, window)
        n_windows = int(ws["s"]["position"].shape[0])
        assume(n_windows >= 2)
        moved = jax.tree.map(lambda x: x, ws)
        moved["s"]["position"] = moved["s"]["position"].at[1].add(bump)
        note(f"tiling={tiling} bump={bump} n_windows={n_windows}")

        def loss(w, weight):
            return float(windowed_loss(
                gm, gm.params, obs, obs_fn=lambda h: h["s"]["position"],
                window=window, sample_every=sample_every, window_states=w,
                continuity_weight=weight))

        assert loss(moved, 1.0) > loss(ws, 1.0)
        # The extra cost grows with the weight: it is a continuity term,
        # not only a data term.
        assert loss(moved, 10.0) > loss(moved, 1.0)

    def test_window_states_must_have_one_entry_per_window(self):
        gm = _spring_gm()
        obs = _observe(gm, 12, gm.params)
        ws = init_window_states(obs, 4)
        with pytest.raises(ValueError, match="leading axis"):
            windowed_loss(gm, gm.params, obs,
                          obs_fn=lambda h: h["s"]["position"], window=4,
                          window_states=jax.tree.map(lambda x: x[:2], ws))


# ---------------------------------------------------------------------------
# 6. The second calibration path
# ---------------------------------------------------------------------------
#
# ``maddening.core.simulation.calibration`` predates ``maddening.sysid``
# and duplicates it badly: ``calibrate`` is plain gradient descent on a
# free-standing ``forward_fn`` with no ParamSpec, no bounds, no trainable
# mask and no graph; ``tune_coupling_params`` is an eager grid search.
# Nothing under ``src/`` imports either, ``maddening.core.simulation``
# does not re-export them, and no example or document mentions them:
# their only callers in the tree are their own two unit-test modules and
# this one.  Both are now ``StabilityLevel.DEPRECATED`` and warn towards
# ``maddening.sysid.fit``, for removal in 0.5.0 -- but a deprecated
# function is still an importable public one until then, and the
# migration it asks for is exactly what these properties describe, so it
# gets the same treatment as anything else here: what it promises is
# what it does.


class TestCalibrate:

    @given(a=_finite(0.5, 2.0), x0=_finite(-3.0, 3.0), x_true=_finite(-3.0, 3.0),
           n_iters=st.integers(min_value=20, max_value=120))
    @settings(max_examples=EXAMPLES_STANDARD, deadline=None)
    def test_gradient_descent_on_a_convex_problem_decreases_and_converges(
        self, a, x0, x_true, n_iters,
    ):
        """On ``r = a*x - a*x_true`` the loss is a convex quadratic with
        curvature ``2 a^2``; with a step below ``2 / curvature`` gradient
        descent is monotone and moves towards the solution."""
        def forward(p):
            return a * p["x"]

        lr = 0.5 / (a * a)              # < 2 / (2 a^2), the stability limit
        result = calibrate(
            forward_fn=forward,
            initial_params={"x": jnp.float32(x0)},
            reference_trajectory=jnp.float32(a * x_true),
            n_iters=n_iters, learning_rate=lr, tolerance=1e-8,
        )
        note(f"a={a} x0={x0} x_true={x_true} history[:3]={result.loss_history[:3]}")
        assert isinstance(result, CalibrateResult)
        assert 0 < len(result.loss_history) <= n_iters
        history = np.asarray(result.loss_history, dtype=np.float64)
        assert np.all(np.isfinite(history))
        assert np.all(np.diff(history) <= 1e-6 * (1.0 + history[:-1]))
        assert abs(float(result.params["x"]) - x_true) <= \
            abs(x0 - x_true) + 1e-5

    @given(a=_finite(0.5, 2.0), x0=_finite(-3.0, 3.0),
           ghost=_finite_f32_normal(-3.0, 3.0),
           n_iters=st.integers(min_value=1, max_value=20))
    @settings(max_examples=EXAMPLES_STANDARD, deadline=None)
    def test_a_parameter_the_forward_ignores_is_bit_identical_afterwards(
        self, a, x0, ghost, n_iters,
    ):
        """``calibrate`` has no ``ParamSpec`` and no mask: the *only*
        thing that freezes a parameter is a zero gradient.  That is worth
        pinning, because it is the whole difference from ``sysid.fit``.
        """
        def forward(p):
            return a * p["x"]

        result = calibrate(
            forward_fn=forward,
            initial_params={"x": jnp.float32(x0), "unused": jnp.float32(ghost)},
            reference_trajectory=jnp.float32(0.0),
            n_iters=n_iters, learning_rate=0.1, tolerance=0.0,
        )
        assert set(result.params) == {"x", "unused"}
        assert float(result.params["unused"]) == float(jnp.float32(ghost))

    @given(tolerance=_finite(1e-6, 1.0), n_iters=st.integers(0, 15))
    @settings(max_examples=EXAMPLES_CHEAP, deadline=None)
    def test_the_converged_flag_means_the_reported_loss_beat_the_tolerance(
        self, tolerance, n_iters,
    ):
        """``converged`` is true exactly when the last recorded loss is
        below ``tolerance`` -- and, because the check runs *before* each
        step, the recorded loss belongs to the returned parameters."""
        def forward(p):
            return p["x"]

        result = calibrate(
            forward_fn=forward, initial_params={"x": jnp.float32(1.0)},
            reference_trajectory=jnp.float32(0.0), n_iters=n_iters,
            learning_rate=0.5, tolerance=tolerance,
        )
        assert len(result.loss_history) <= max(n_iters, 0)
        if not result.loss_history:
            assert n_iters == 0 and not result.converged
            assert float(result.params["x"]) == 1.0
            return
        last = result.loss_history[-1]
        assert result.converged == (last < tolerance)
        if result.converged:
            # The early return keeps the parameters that produced ``last``.
            assert float(forward(result.params) ** 2) == pytest.approx(
                last, rel=1e-5, abs=1e-12)


def _build_springs(**coupling_kwargs):
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="a", timestep=0.001, stiffness=50.0,
                                 damping=1.0, mass=1.0, rest_length=1.0,
                                 initial_position=0.0))
    gm.add_node(SpringDamperNode(name="b", timestep=0.001, stiffness=50.0,
                                 damping=1.0, mass=1.0, rest_length=1.0,
                                 initial_position=2.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    gm.add_coupling_group(["a", "b"], diagnostics=True, **coupling_kwargs)
    return gm


class TestTuneCouplingParams:

    @given(tolerances=st.lists(st.sampled_from([1e-3, 1e-6, 1e-9]),
                               min_size=1, max_size=2, unique=True),
           iterations=st.lists(st.sampled_from([3, 6]), min_size=1, max_size=2,
                               unique=True),
           threshold=st.sampled_from([0.0, 1e-6, 1.0]))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_the_report_describes_the_grid_it_searched(
        self, tolerances, iterations, threshold,
    ):
        grid = {"tolerance": tolerances, "max_iterations": iterations}
        configs = [dict(zip(sorted(grid), values)) for values in
                   itertools.product(*(grid[k] for k in sorted(grid)))]
        note(f"grid={grid} threshold={threshold}")

        result = tune_coupling_params(
            build_graph_fn=_build_springs, param_grid=grid, n_steps=3,
            accuracy_threshold=threshold,
        )
        assert isinstance(result, TuneResult)
        assert len(result.all_trials) == len(configs)
        assert [t["params"] for t in result.all_trials] == configs
        assert result.best_params in configs

        chosen = [t for t in result.all_trials if t["params"] == result.best_params]
        assert len(chosen) == 1
        assert chosen[0]["total_iters"] == result.best_total_iters
        assert chosen[0]["max_error"] == result.best_max_error

        passing = [t for t in result.all_trials if t["passed"]]
        if passing:
            assert chosen[0]["passed"]
            assert result.best_total_iters == min(t["total_iters"] for t in passing)
        else:
            assert result.best_max_error == min(
                t["max_error"] for t in result.all_trials)

    @given(iterations=st.lists(st.sampled_from([3, 6, 9]), min_size=2,
                               max_size=2, unique=True))
    @settings(max_examples=EXAMPLES_COSTLY, deadline=None)
    def test_the_reference_configuration_reproduces_itself_exactly(
        self, iterations,
    ):
        """The reference run is the grid's tightest tolerance and largest
        iteration count, so the trial with those settings re-runs the same
        deterministic graph and must score an error of exactly zero."""
        grid = {"tolerance": [1e-9], "max_iterations": sorted(iterations)}
        result = tune_coupling_params(
            build_graph_fn=_build_springs, param_grid=grid, n_steps=3,
            accuracy_threshold=0.0,
        )
        reference = {"tolerance": 1e-9, "max_iterations": max(iterations)}
        trial = next(t for t in result.all_trials if t["params"] == reference)
        note(f"trials={[(t['params'], t['max_error']) for t in result.all_trials]}")
        assert trial["max_error"] == 0.0
        assert trial["passed"]
        assert result.best_max_error == 0.0
