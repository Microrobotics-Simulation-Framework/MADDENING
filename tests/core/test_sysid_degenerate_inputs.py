"""What ``maddening.sysid`` does when its inputs are degenerate.

The fitters and :func:`~maddening.sysid.fim` are the surface a user
trusts to tell them whether their data determine their constants, so the
failure that matters here is not an exception -- it is a confident
answer that is wrong.  Every test below fixes one way that used to
happen, with the reproducer it came from named in the test.

Reproducers: ``benchmarks/results/audit_040_final/params-io/repro/``
(``r6_fim.py``, ``r7_fim_crb.py``, ``r8_fim_relative_zero.py``,
``r9_fit_degenerate.py``, ``r10_noop_fit_drift.py``,
``r14_mask_structure.py``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import contextlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import _rank_and_crb, fim, fit, fit_lm
from maddening.warnings import PrecisionLimitWarning


def _spring_gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0,
                                 mass=1.5, rest_length=1.0,
                                 initial_position=0.5))
    gm.compile()
    return gm


_PARAMS = {"a": jnp.asarray(1.0), "b": jnp.asarray(2.0)}


def _residual(p):
    return jnp.asarray([p["a"], 2.0 * p["b"]])


# ---------------------------------------------------------------------------
# The +inf fail-safe (r7_fim_crb.py, r6_fim.py)
# ---------------------------------------------------------------------------


def test_an_eigendecomposition_that_is_not_finite_reports_no_finite_bound():
    """``FIMReport`` promises ``crb = +inf`` *because* it fails safe.

    With a NaN decomposition the old test (``support > n * eps``) was
    False for every parameter -- NaN compares False against everything --
    so the empty resolved-subspace sum, ``0.0``, survived as the bound
    and ``crb < tol`` answered "identified" for a matrix holding no
    information at all.
    """
    n = 3
    nan_ev = jnp.full((n,), jnp.nan, dtype=jnp.float32)
    nan_vecs = jnp.full((n, n), jnp.nan, dtype=jnp.float32)
    rank, crb = _rank_and_crb(nan_ev, nan_vecs, None)
    crb = np.asarray(crb)
    assert rank == 0
    assert np.all(np.isinf(crb)), crb
    assert not np.any(crb < 1e-6), "an unidentifiable parameter read as identified"


def test_a_partly_resolved_decomposition_with_a_nan_direction_fails_safe():
    """One good eigenvalue does not license a bound on the NaN direction."""
    ev = jnp.asarray([jnp.nan, 1.0], dtype=jnp.float32)
    vecs = jnp.asarray([[jnp.nan, 0.0], [jnp.nan, 1.0]], dtype=jnp.float32)
    _, crb = _rank_and_crb(ev, vecs, None)
    assert np.all(np.isinf(np.asarray(crb)))


def test_fim_refuses_to_report_on_a_non_finite_fisher_matrix():
    def nan_residual(p):
        return jnp.asarray([p["a"] * jnp.nan, p["b"]])

    with pytest.raises(FloatingPointError, match="non-finite Fisher matrix"):
        fim(nan_residual, _PARAMS, scale=None)


@pytest.mark.parametrize("sigma", [0.0, -1.0, float("nan"), float("inf"), 1e-320])
def test_fim_rejects_a_noise_std_that_is_not_positive_at_the_residual_precision(sigma):
    """``1e-320`` is ``0.0`` once cast to float32, which is why the check
    runs after the cast rather than on the caller's Python float."""
    with pytest.raises(ValueError, match="noise_std must be finite and strictly positive"):
        fim(_residual, _PARAMS, scale=None, noise_std=sigma)


def test_fim_rejects_a_non_positive_sigma_inside_a_pytree_noise_model():
    def tree_residual(p):
        return {"x": jnp.asarray([p["a"]]), "y": jnp.asarray([p["b"]])}

    with pytest.raises(ValueError, match="noise_std must be finite and strictly positive"):
        fim(tree_residual, _PARAMS, scale=None, noise_std={"x": 1.0, "y": 0.0})


def test_fit_lm_rejects_a_non_positive_sigma_on_the_same_path():
    gm = _spring_gm()
    step, ext = gm._compiled_step, gm._default_external_inputs()
    target = step(gm._state, ext, gm.params)["s"]

    def residual(p):
        return step(gm._state, ext, p)["s"]["position"] - target["position"]

    with pytest.raises(ValueError, match="noise_std must be finite and strictly positive"):
        fit_lm(gm, residual, n_iter=1, noise_std=-1.0)


def test_a_positive_noise_std_still_scales_the_information():
    base = fim(_residual, _PARAMS, scale=None)
    scaled = fim(_residual, _PARAMS, scale=None, noise_std=2.0)
    np.testing.assert_allclose(np.asarray(scaled.fim), np.asarray(base.fim) / 4.0,
                               rtol=1e-6)


# ---------------------------------------------------------------------------
# A mask is read by position, so its structure has to match (r14_mask_structure)
# ---------------------------------------------------------------------------


def _relabelled(tree, labels):
    """``tree``'s leaves under different keys, same flatten order."""
    leaves = jax.tree.leaves(tree)
    return {name: leaf for name, leaf in zip(sorted(labels), leaves)}


def test_a_mask_whose_keys_differ_from_params_is_refused_by_fit():
    """The count matched and the keys did not, so the flags landed by
    flatten position: the audit marked ``damping`` and the fit moved
    ``stiffness`` from 30.0 to 76.3."""
    gm = _spring_gm()
    names = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    mask = {"nodes": {"s": _relabelled(gm.params["nodes"]["s"], names)}}
    mask["nodes"]["s"] = {k: (k == "zeta") for k in mask["nodes"]["s"]}
    assert len(jax.tree.leaves(mask)) == len(jax.tree.leaves(gm.params))

    with pytest.raises(ValueError, match="same tree structure as params"):
        fit(gm, lambda p: jnp.sum(jnp.asarray(p["nodes"]["s"]["stiffness"]) ** 2),
            mask=mask, n_iter=1)


def test_the_structure_error_names_the_first_path_that_differs():
    gm = _spring_gm()
    mask = {"nodes": {"s": {k if k != "damping" else "zeta": False
                            for k in gm.params["nodes"]["s"]}}}
    mask["nodes"]["s"]["stiffness"] = True
    with pytest.raises(ValueError) as excinfo:
        fit(gm, lambda p: jnp.asarray(0.0), mask=mask, n_iter=1)
    message = str(excinfo.value)
    assert "They first differ at leaf" in message
    assert "['nodes']['s']['damping']" in message


def test_a_mask_built_from_the_params_tree_is_still_accepted():
    gm = _spring_gm()
    mask = jax.tree.map(lambda _: False, gm.params)
    mask["nodes"]["s"]["stiffness"] = True
    res = fit(gm, lambda p: jnp.sum(jnp.asarray(p["nodes"]["s"]["stiffness"]) ** 2),
              mask=mask, n_iter=2, lr=0.1)
    assert not np.array_equal(np.asarray(res.params["nodes"]["s"]["stiffness"]),
                              np.asarray(gm.params["nodes"]["s"]["stiffness"]))


def test_fim_refuses_a_mask_whose_keys_differ_from_params():
    params = {"a": jnp.float32(2.0), "b": jnp.float32(3.0)}
    with pytest.raises(ValueError, match="same tree structure as params"):
        fim(_residual, params, mask={"a": True, "c": False})


# ---------------------------------------------------------------------------
# Relative scaling at zero (r8_fim_relative_zero.py)
# ---------------------------------------------------------------------------


def test_fim_names_the_parameters_relative_scaling_zeroed():
    """With ``J = I`` every parameter is exactly identifiable; the one
    sitting at 0.0 still reports ``crb = +inf`` under the default scale,
    because its column was multiplied by its value.  The report has to
    say that is what happened."""
    params = {"a": jnp.float32(2.0), "zero": jnp.float32(0.0)}

    def identity_residual(p):
        return jnp.stack([p["a"], p["zero"]])

    rel = fim(identity_residual, params)          # scale="relative", the default
    absolute = fim(identity_residual, params, scale=None)

    assert rel.zero_scaled == ("['zero']",)
    assert np.isinf(np.asarray(rel.crb)[rel.param_names.index("['zero']")])
    assert rel.rank == 1
    # The absolute question has an answer, and the report does not
    # pretend the relative one was it.
    assert absolute.zero_scaled == ()
    assert np.all(np.isfinite(np.asarray(absolute.crb)))
    assert absolute.rank == 2


def test_a_report_with_no_zero_valued_parameter_names_none():
    params = {"a": jnp.float32(2.0), "b": jnp.float32(3.0)}
    assert fim(_residual, params).zero_scaled == ()


# ---------------------------------------------------------------------------
# Hyper-parameters that invert their own meaning (r9_fit_degenerate.py)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kwargs,match", [
    ({"lr": 0.0}, "lr must be greater than 0"),
    ({"lr": -0.1}, "lr must be greater than 0"),
    ({"eps": 0.0}, "eps must be greater than 0"),
    ({"eps": -1e-8}, "eps must be greater than 0"),
    ({"tol": float("nan")}, "tol must be a finite number"),
    ({"tol": -1.0}, "tol must be at least 0"),
    ({"betas": (1.0, 0.999)}, r"betas\[0\] must be less than 1"),
    ({"betas": (0.9, 1.0)}, r"betas\[1\] must be less than 1"),
    ({"n_iter": -1}, "n_iter must be an integer"),
    ({"notify_every": -1}, "notify_every must be an integer"),
])
def test_fit_rejects_a_hyper_parameter_with_no_reading(kwargs, match):
    gm = _spring_gm()
    call = {"n_iter": 2, **kwargs}
    with pytest.raises(ValueError, match=match):
        fit(gm, lambda p: jnp.asarray(0.0), **call)


@pytest.mark.parametrize("kwargs,match", [
    ({"lam0": 0.0}, "lam0 must be greater than 0"),
    ({"lam0": -1.0}, "lam0 must be greater than 0"),
    ({"lam_up": 0.5}, "lam_up must be greater than 1"),
    ({"lam_up": 1.0}, "lam_up must be greater than 1"),
    ({"lam_down": 2.0}, "lam_down must be at most 1"),
    ({"step_tol": -1.0}, "step_tol must be at least 0"),
    ({"tol": float("nan")}, "tol must be a finite number"),
])
def test_fit_lm_rejects_a_hyper_parameter_with_no_reading(kwargs, match):
    gm = _spring_gm()
    with pytest.raises(ValueError, match=match):
        fit_lm(gm, lambda p: jnp.ones(3, jnp.float32), n_iter=1, **kwargs)


def test_the_documented_defaults_are_inside_the_accepted_range():
    """A validator that rejected its own defaults would be worse than none."""
    gm = _spring_gm()
    fit(gm, lambda p: jnp.asarray(0.0), n_iter=1)
    fit_lm(gm, lambda p: jnp.ones(3, jnp.float32), n_iter=1)


# ---------------------------------------------------------------------------
# A fit that takes no step returns its input (r10_noop_fit_drift.py)
# ---------------------------------------------------------------------------


def _bits(tree):
    return {k: np.asarray(v).tobytes() for k, v in tree["nodes"]["s"].items()}


@pytest.mark.parametrize("kwargs", [{"n_iter": 0}, {"n_iter": 5, "tol": 1e30}])
def test_a_fit_that_takes_no_step_returns_its_input_bit_for_bit(kwargs):
    """``FitResult`` promises that comparing input and output leaf by
    leaf says exactly which constants the calibration touched.  A masked
    ``log`` leaf used to come back one ulp off -- ``exp(log(30.0))`` is
    ``30.000002`` -- although no gradient had been applied to it, so the
    comparison could not tell "not fitted" from "fitted and barely
    moved"."""
    gm = _spring_gm()
    before = _bits(gm.params)
    res = fit(gm, lambda p: jnp.asarray(0.0), **kwargs)
    assert _bits(res.params) == before


def test_a_fit_that_does_take_a_step_still_reports_the_leaf_as_moved():
    """Non-vacuity: the exactness above must not be a blanket copy."""
    gm = _spring_gm()

    def loss(p):
        return jnp.sum(jnp.asarray(p["nodes"]["s"]["stiffness"]) ** 2)

    res = fit(gm, loss, n_iter=3, lr=0.1)
    changed = {k for k, v in _bits(res.params).items() if v != _bits(gm.params)[k]}
    assert "stiffness" in changed
    assert "initial_position" not in changed   # frozen by its ParamSpec


# ---------------------------------------------------------------------------
# FIMReport is keyword-only: inserting a field must not reassign the rest
# ---------------------------------------------------------------------------


def test_fim_report_cannot_be_built_positionally():
    """``rank`` was inserted between ``eigvecs`` and ``cond`` during
    0.4.0, so positional construction silently shifted every field after
    it -- no ``TypeError``, no warning, and a wrong ``rank`` is a wrong
    identifiability verdict.  ``kw_only`` is what stops the next inserted
    field doing it again."""
    from maddening.sysid import FIMReport

    kwargs = dict(
        fim=jnp.eye(2), eigvals=jnp.ones(2), eigvecs=jnp.eye(2),
        rank=2, cond=1.0, crb=jnp.ones(2), param_names=("a", "b"),
    )
    report = FIMReport(**kwargs)
    assert report.rank == 2 and report.cond == 1.0
    with pytest.raises(TypeError):
        FIMReport(*kwargs.values())            # type: ignore[misc]


def test_fit_result_cannot_be_built_positionally():
    """The same guard on the other public dataclass in this module.

    ``FitResult`` has had no field inserted into it, so this assertion
    is expected to be redundant today -- and it is the only thing that
    keeps it so.  Its ``converged``/``n_iter`` pair is the worse case of
    the two: a shift there does not even change the field *types*, so a
    positional caller of a four-field ``FitResult`` that grew a fifth
    field would read ``n_iter`` as ``converged`` -- truthy for every
    run that took a step -- and report an unconverged fit as converged.
    """
    from maddening.sysid import FitResult

    kwargs = dict(
        params={"a": 1.0}, losses=np.zeros(3), converged=False, n_iter=12,
    )
    result = FitResult(**kwargs)
    assert result.converged is False and result.n_iter == 12
    with pytest.raises(TypeError):
        FitResult(*kwargs.values())            # type: ignore[misc]


# ---------------------------------------------------------------------------
# A rank verdict decided at the float32 noise floor says so
# ---------------------------------------------------------------------------


def _linear_fim(eig_ratio, *, n=2, m=3, seed=0, dtype=jnp.float32, **kw):
    """``fim`` of a linear residual whose Fisher matrix has a known
    smallest-to-largest eigenvalue ratio."""
    rng = np.random.default_rng(seed)
    # ``geomspace(a, b, 1)`` is ``[a]``, not ``[b]``, so pin the largest
    # eigenvalue explicitly: ``eig_ratio`` is a ratio to it and the whole
    # helper is meaningless if it is not 1.0.
    mid = np.geomspace(1e-2, 1.0, max(n - 1, 1))
    mid[-1] = 1.0
    lam = np.sort(np.concatenate([[eig_ratio], mid]))[:n]
    assert lam[-1] == 1.0 and lam[0] == eig_ratio
    q, rr = np.linalg.qr(rng.standard_normal((n, n)))
    V = q * np.sign(np.diag(rr))
    U = np.linalg.qr(rng.standard_normal((max(m, n), n)))[0]
    A = jnp.asarray((U * np.sqrt(lam)) @ V.T, dtype=dtype)
    params = {f"p{i}": jnp.asarray(1.0, dtype=dtype) for i in range(n)}
    keys = tuple(params)

    def residual_fn(p):
        return A @ jnp.stack([p[k] for k in keys])

    return fim(residual_fn, params, scale=None, **kw)


def test_a_rank_decided_at_the_float32_floor_warns_with_the_numbers():
    """The case this guard was built from: an eigenvalue ratio of
    2.08e-07 against a 2.38e-07 cutoff.

    A rank determination resting on a difference smaller than float32
    epsilon, reported as fact -- ``rank=1``, ``crb=[inf, inf]``, no
    signal of any kind that the verdict was a coin flip.  The warning
    has to carry the two numbers, or a reader cannot tell how close the
    call was, and it has to name ``jax_enable_x64``, which is the whole
    remedy.
    """
    with pytest.warns(PrecisionLimitWarning) as rec:
        report = _linear_fim(2.08e-07)
    assert report.rank == 1
    ev = np.asarray(report.eigvals, dtype=np.float64)
    ratio = ev[0] / ev[-1]
    cutoff = 2 * float(np.finfo(np.float32).eps)
    assert abs(ratio / 2.08e-07 - 1.0) < 0.05, ratio
    msg = str(rec[0].message)
    assert f"{ratio:.4g}" in msg and f"{cutoff:.4g}" in msg
    assert "jax_enable_x64" in msg


def test_a_well_conditioned_fim_is_silent():
    """Non-vacuity, and the property that keeps the warning worth
    reading: it must not fire on an ordinary problem.  A warning that
    fires routinely gets suppressed, which is worse than silence."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionLimitWarning)
        report = _linear_fim(1e-3)
    assert report.rank == 2


def test_an_exactly_singular_fim_is_silent_about_precision():
    """An exactly rank-deficient ``F`` is a real rank deficiency, not a
    verdict decided by rounding.

    ``scale="relative"`` with a parameter sitting at ``0.0`` produces
    exactly this -- an exactly zero column, so an exactly zero
    eigenvalue -- and it is the first thing a user meets on the default
    scale, since ``SpringDamperNode.initial_velocity`` defaults to
    ``0.0``.  Warning there would fire on every such report for a
    verdict float64 agrees with completely.
    """
    params = {"a": jnp.float32(1.0), "b": jnp.float32(0.0)}

    def residual_fn(p):
        return jnp.stack([p["a"] + p["b"], p["a"] - p["b"]])

    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionLimitWarning)
        report = fim(residual_fn, params, scale="relative")
    assert report.zero_scaled == ("['b']",)
    assert report.rank == 1


def test_rank_rtol_zero_leaves_no_threshold_to_sit_near():
    """``rank_rtol=0`` resolves every positive eigenvalue by request, so
    there is no cutoff a ratio can be near and nothing to warn about --
    rather than a division by zero inside the check."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionLimitWarning)
        report = _linear_fim(2.08e-07, rank_rtol=0.0)
    assert report.rank == 2


def test_the_warning_names_the_eigenvalue_nearest_the_cutoff():
    """Not ``eigvals[0]``.  A spectrum whose smallest eigenvalue is far
    below the cutoff can still have a *different* one sitting on it, and
    that one is what a rounding error would carry across.  Reporting the
    smallest would name a ratio nowhere near the number being compared.
    """
    from maddening.sysid import _precision_limited

    eps = float(np.finfo(np.float32).eps)
    cutoff = 3 * eps
    # smallest is 1e-4 of the cutoff; the middle one is sitting on it
    ev = jnp.asarray([1e-4 * cutoff, 1.1 * cutoff, 1.0], dtype=jnp.float32)
    limited = _precision_limited(ev, cutoff, cutoff)
    assert limited is not None
    ratio, cut = limited
    assert cut == cutoff
    assert abs(ratio / (1.1 * cutoff) - 1.0) < 1e-3, ratio


@contextlib.contextmanager
def _x64():
    """``jax_enable_x64`` for the duration of the block.

    Process-global and normally set before the first JAX import, which
    is exactly why the warning recommends it rather than doing it: it
    changes every library in the process.  Toggling it here is safe only
    because the block restores it.
    """
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


def test_the_x64_rerun_the_warning_recommends_actually_settles_the_verdict():
    """The warning is worth emitting only if its remedy works.

    The pinned case -- ratio 2.08e-07, cutoff 2.38e-07 -- reads
    ``rank=1`` in float32.  Under x64 the cutoff drops to ``n * 2.2e-16``
    and the same data resolve both directions, so the answer the warning
    said was undetermined is determined, and the other way round from
    the float32 verdict.  If this ever stopped holding, the warning
    would be sending users on an errand that does not pay.
    """
    with _x64():
        with warnings.catch_warnings():
            warnings.simplefilter("error", PrecisionLimitWarning)
            report = _linear_fim(2.08e-07, dtype=jnp.float64)
    assert report.rank == 2
    assert np.all(np.isfinite(np.asarray(report.crb)))


def test_under_x64_the_message_does_not_send_the_user_round_again():
    """At float64 the remedy has already been taken and there is no
    third precision, so the message has to say something else.  Pointing
    a user who is already at x64 back at x64 is a loop."""
    with _x64():
        with pytest.warns(PrecisionLimitWarning) as rec:
            # sitting on the float64 cutoff, n * 2.22e-16
            _linear_fim(2.0 * float(np.finfo(np.float64).eps),
                        dtype=jnp.float64)
    msg = str(rec[0].message)
    assert "float64 noise floor" in msg
    assert "jax_enable_x64" not in msg
    assert "widest precision" in msg


@contextlib.contextmanager
def _x64():
    """``jax_enable_x64`` for the duration of the block.

    Process-global and normally set before the first JAX import, which
    is exactly why the warning recommends it rather than doing it:
    flipping it changes every library in the process.
    """
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prior)


def test_the_x64_rerun_the_warning_recommends_actually_settles_the_verdict():
    """The warning is only actionable if its remedy works.

    The float32 report calls the 2.08e-07 direction unresolved because
    that ratio is under a 2.38e-07 cutoff.  Under x64 the cutoff drops
    to ``n * 2.22e-16``, the same direction clears it by nine decades,
    and the answer is rank 2 with a finite bound -- so the warning's
    "re-run under x64 to settle it" is a claim this test holds it to,
    not a form of words.  It is also the non-vacuity check on the
    warning: the two precisions really do answer differently here.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PrecisionLimitWarning)
        f32 = _linear_fim(2.08e-07)
    assert f32.rank == 1 and not np.isfinite(np.asarray(f32.crb)).any()

    with _x64():
        with warnings.catch_warnings():
            warnings.simplefilter("error", PrecisionLimitWarning)
            f64 = _linear_fim(2.08e-07, dtype=jnp.float64)
    assert f64.rank == 2
    assert np.isfinite(np.asarray(f64.crb)).all()


def test_under_x64_the_warning_stops_recommending_x64():
    """A verdict at the *float64* floor has no wider precision to
    escalate to, so repeating the x64 advice would send a reader round a
    loop they have already finished.  The message has to say what is
    actually left to do instead."""
    cutoff = 2 * float(np.finfo(np.float64).eps)
    with _x64():
        with pytest.warns(PrecisionLimitWarning) as rec:
            report = _linear_fim(0.9 * cutoff, dtype=jnp.float64)
    assert report.rank == 1
    msg = str(rec[0].message)
    assert "float64" in msg
    assert "jax_enable_x64" not in msg
    assert "widest precision" in msg


def test_a_raised_rank_rtol_is_a_modelling_choice_not_a_precision_limit():
    """Raising ``rank_rtol`` moves the cutoff decades above the noise
    floor, and a close call there is not a close call about precision.

    ``rank_rtol=1e-3`` says "I call anything below 1e-3 unidentifiable
    in practice".  An eigenvalue ratio of 4e-4 against that cutoff is
    within a factor of 2.5 of it -- but float32 knows both numbers to
    three further decimal places, so the verdict is exact and warning
    about rounding would be simply wrong.  The band is anchored to
    ``n * eps``, not to whatever cutoff the caller picked.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", PrecisionLimitWarning)
        report = _linear_fim(4e-4, rank_rtol=1e-3)
    assert report.rank == 1          # 4e-4 is below the 1e-3 cutoff


def test_a_rank_rtol_under_the_noise_floor_still_warns():
    """The guard cuts one way only.  Lowering ``rank_rtol`` below
    ``n * eps`` asks for a cutoff finer than the decomposition can
    resolve, so every verdict at it is rounding -- the case the warning
    exists for, reached from the other side."""
    eps = float(np.finfo(np.float32).eps)
    with pytest.warns(PrecisionLimitWarning):
        _linear_fim(0.4 * eps, rank_rtol=0.5 * eps)
