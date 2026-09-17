"""``ParamSpec`` at the edges of its envelope.

* ``constrain`` with ``transform="log"`` and a non-zero lower bound, or
  ``"logit"`` with large bounds, lands strictly inside the open interval
  and stays re-``unconstrain``-able (a fit's own output must pass
  ``check_params``);
* ``check`` / ``check_params`` reject NaN and inf;
* ``from_dict`` accepts ``bounds: null``;
* a bounded identity leaf keeps an integer dtype.

Originally written from the independent audit of 2026-09-16 (round 1; report and
reproducers under ``benchmarks/results/audit1/``).
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
from tests.conftest import EXAMPLES_STANDARD

SETTINGS = dict(max_examples=EXAMPLES_STANDARD, deadline=None,
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
