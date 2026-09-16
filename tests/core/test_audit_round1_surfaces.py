"""Audit round 1 (feat/graph-params-sysid): ParamSpec and sysid surfaces.

Regression tests adapted from the audit's reproducers.  Each one failed
before the corresponding fix:

* #5  ``constrain`` with ``transform="log"`` and ``lo != 0`` landed exactly
  ON the strict bound (``8.0 + exp(-15)`` is ``8.0`` in float32), so
  ``check_params`` rejected the fit's own output and ``unconstrain``
  returned ``-inf``; same for ``logit`` with large ``|lo|, |hi|``.
* #10 ``ParamSpec.check`` / ``check_params`` accepted NaN / inf.
* #13 ``ParamSpec.from_dict({"bounds": null})`` crashed.
* #14 ``constrain`` promoted an integer bounded identity leaf to float.
* #9  ``fim`` / ``fit_lm`` with ``noise_std=<pytree>`` crashed because
  ``jnp.ndim(dict) == 0`` took the scalar branch.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode

SETTINGS = dict(max_examples=40, deadline=None,
                suppress_health_check=[HealthCheck.too_slow])


def _spring_gm(**kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=1.0, **kw))
    gm.compile()
    return gm


# ---------------------------------------------------------------------------
# #5: constrain lands strictly inside a strict bound, for any coordinate
# ---------------------------------------------------------------------------

# Bounds are float32-representable (``width=32``) so ``lo`` is the same
# number in the spec and in the float32 arithmetic.
@given(u=st.floats(-1e4, 1e4, allow_nan=False, width=32),
       lo=st.floats(-1e3, 1e3, allow_nan=False, width=32))
@settings(**SETTINGS)
def test_log_constrain_is_strictly_inside_and_re_unconstrainable(u, lo):
    spec = ParamSpec(bounds=(float(lo), None), transform="log")
    p = spec.to_constrained(jnp.float32(u))
    spec.check(p)                                   # strictly > lo
    assert np.isfinite(float(spec.to_unconstrained(p)))


# ``delta >= 2**-9`` at ``|lo| <= 1e3`` keeps the interval at least ~16
# float32 ulps wide; a narrower interval has no interior to clamp to.
@given(u=st.floats(-1e4, 1e4, allow_nan=False, width=32),
       lo=st.floats(-1e3, 1e3, allow_nan=False, width=32),
       delta=st.floats(0.001953125, 1e3, allow_nan=False, width=32))
@settings(**SETTINGS)
def test_logit_constrain_is_strictly_inside_for_large_bounds(u, lo, delta):
    hi = float(np.float32(lo + delta))
    assume(hi > lo)
    spec = ParamSpec(bounds=(float(lo), hi), transform="logit")
    p = spec.to_constrained(jnp.float32(u))
    spec.check(p)                                   # lo < p < hi
    assert np.isfinite(float(spec.to_unconstrained(p)))


@given(lo=st.floats(-1e3, 1e3, allow_nan=False, width=32),
       x=st.floats(0.0009765625, 1e6, allow_nan=False, width=32))
@settings(**SETTINGS)
def test_log_round_trip_inside_bounds_still_exact(lo, x):
    """The relative clamp must not disturb values well inside the range."""
    spec = ParamSpec(bounds=(float(lo), None), transform="log")
    p = jnp.float32(lo + x)
    assume(float(p) > lo)
    back = spec.to_constrained(spec.to_unconstrained(p))
    assert float(back) == pytest.approx(float(p), rel=1e-4, abs=1e-6)
    spec.check(back)


def test_log_constrain_with_nonzero_lower_bound_keeps_fit_continuable():
    """The audit's falsifying example through the GraphManager surface:
    an ordinary Adam coordinate must give a value ``check_params`` accepts
    and ``unconstrain`` can map back."""
    gm = _spring_gm()
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(8.0, None), transform="log"))
    u = gm.unconstrain()
    u["nodes"]["s"]["stiffness"] = jnp.float32(-15.0)
    p = gm.constrain(u)
    assert float(p["nodes"]["s"]["stiffness"]) > 8.0
    gm.check_params(p)
    assert np.isfinite(float(gm.unconstrain(p)["nodes"]["s"]["stiffness"]))


# ---------------------------------------------------------------------------
# #10: non-finite values are rejected
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_check_params_rejects_non_finite(bad):
    gm = _spring_gm()
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(bad, jnp.float32)
    with pytest.raises(ValueError, match="stiffness.*not finite"):
        gm.check_params(p)


def test_check_rejects_nan_even_without_bounds():
    with pytest.raises(ValueError, match="not finite"):
        ParamSpec().check(jnp.asarray(jnp.nan), name="x")
    with pytest.raises(ValueError, match="not finite"):
        ParamSpec(bounds=(0.0, 1.0)).check(jnp.asarray([0.5, jnp.nan]), name="x")
    ParamSpec().check(jnp.asarray(3, jnp.int32))          # integers are fine


# ---------------------------------------------------------------------------
# #13: JSON null bounds
# ---------------------------------------------------------------------------

def test_param_spec_from_dict_bounds_null():
    spec = ParamSpec.from_dict({"trainable": True, "bounds": None})
    assert spec.bounds == (None, None)
    assert ParamSpec.from_dict({"bounds": [None, None]}).bounds == (None, None)
    assert ParamSpec.from_dict({"bounds": [0, None]}).bounds == (0.0, None)


# ---------------------------------------------------------------------------
# #14: identity clip keeps the leaf dtype
# ---------------------------------------------------------------------------

def test_identity_bounded_int_leaf_keeps_dtype():
    spec = ParamSpec(bounds=(0.0, 10.0))
    p = jnp.asarray(5, jnp.int32)
    out = spec.to_constrained(spec.to_unconstrained(p))
    assert out.dtype == p.dtype and int(out) == 5
    assert int(spec.to_constrained(jnp.asarray(12, jnp.int32))) == 10
    f = jnp.asarray(5.0, jnp.float32)
    assert spec.to_constrained(f).dtype == jnp.float32


# ---------------------------------------------------------------------------
# #9: sysid noise_std as a pytree
# ---------------------------------------------------------------------------

def _pytree_residual(gm):
    step, ext = gm._compiled_step, gm._default_external_inputs()
    target = step(gm._state, ext, gm.params)["s"]

    def residual(p):
        s = step(gm._state, ext, p)["s"]
        return {"pos": s["position"] - target["position"],
                "vel": s["velocity"] - target["velocity"]}
    return residual


def test_fim_noise_std_pytree_dict():
    from maddening.sysid import fim
    gm = _spring_gm()
    residual = _pytree_residual(gm)
    unit = fim(residual, gm.params, mask=gm.trainable_mask(), noise_std=1.0)
    per = fim(residual, gm.params, mask=gm.trainable_mask(),
              noise_std={"pos": 0.1, "vel": 1.0})
    # The velocity residual keeps its weight; the position row is 10x more
    # informative, so per-leaf noise changes the matrix.
    assert np.all(np.isfinite(np.asarray(per.fim)))
    assert not np.allclose(np.asarray(per.fim), np.asarray(unit.fim))
    # scalar forms that must keep working
    for sd in (2.0, np.float32(2.0), jnp.float32(2.0)):
        fim(residual, gm.params, mask=gm.trainable_mask(), noise_std=sd)


def test_fit_lm_noise_std_pytree_dict():
    from maddening.sysid import fit_lm
    gm = _spring_gm()
    residual = _pytree_residual(gm)
    start = jax.tree.map(lambda x: x, gm.params)
    start["nodes"]["s"]["stiffness"] = jnp.float32(45.0)
    res = fit_lm(gm, residual, params=start, n_iter=5,
                 noise_std={"pos": 0.1, "vel": 1.0})
    gm.check_params(res.params)
    assert res.losses[-1] <= res.losses[0]


def test_fit_lm_never_accepting_a_step_returns_cleanly():
    """A residual no step can lower: the loop must exit as not converged
    rather than touch an unset ``step_norm``."""
    from maddening.sysid import fit_lm
    gm = _spring_gm()
    res = fit_lm(gm, lambda p: jnp.ones(3, jnp.float32), n_iter=4)
    assert res.converged is False and res.n_iter == 1
