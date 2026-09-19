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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import _rank_and_crb, fim, fit, fit_lm


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
