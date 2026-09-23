"""``ParamSpec`` at the edges of its envelope.

* ``constrain`` with ``transform="log"`` and a non-zero lower bound, or
  ``"logit"`` with large bounds, lands strictly inside the open interval
  and stays re-``unconstrain``-able (a fit's own output must pass
  ``check_params``);
* ``check`` / ``check_params`` reject NaN and inf;
* ``from_dict`` accepts ``bounds: null``;
* a bounded identity leaf keeps an integer dtype;
* a spec keyed for a list/tuple-valued leaf reaches it in every
  consumer of ``_spec_for`` -- ``trainable_mask``, ``unconstrain`` /
  ``constrain`` and ``check_bounds`` -- rather than the leaf getting the
  default spec because a ``SequenceKey`` has no ``.key``.

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
from maddening.core.params import (
    ParamSpec,
    check_bounds,
    constrain,
    trainable_mask,
    unconstrain,
)
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


def test_a_log_or_logit_transform_refuses_a_leaf_it_cannot_return():
    """The identity branch preserves an integer leaf's dtype; the
    transform branches cannot -- ``exp(log(3))`` is float, and for
    ``logit`` it is also 2.4e-7 off -- so ``constrain(unconstrain(p))``
    came back as a different dtype and a different value than the one
    this module's docstring promises for the whole tree.  Refusing the
    combination is what keeps that promise true.

    Reproducer: ``benchmarks/results/audit_040_final/params-io/repro/
    r1_paramspec_roundtrip.py``.
    """
    p = jnp.asarray(3, jnp.int32)
    for spec in (ParamSpec(bounds=(0.0, None), transform="log"),
                 ParamSpec(bounds=(0.0, 100.0), transform="logit")):
        with pytest.raises(ValueError, match="needs a floating-point leaf"):
            spec.to_unconstrained(p)
    # The same parameter as a float still round-trips, and the identity
    # transform still accepts the integer leaf.
    q = jnp.asarray(3.0, jnp.float32)
    log_spec = ParamSpec(bounds=(0.0, None), transform="log")
    assert log_spec.to_constrained(log_spec.to_unconstrained(q)).dtype == jnp.float32
    assert int(ParamSpec(bounds=(0.0, 10.0)).to_unconstrained(p)) == 3


# ---------------------------------------------------------------------------
# #9: sysid noise_std as a pytree
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A list/tuple-valued leaf is governed by the spec keyed for it
# ---------------------------------------------------------------------------
#
# ``jax.tree_util`` flattens ``{"v": [0.0, 1.0]}`` to two leaves at
# ``SequenceKey`` paths.  ``_spec_for`` used to stop at the first one
# and answer the default spec, so the array spelling ``{"v": [0., 1.]}``
# (one leaf, one ``DictKey``) and the list spelling disagreed in every
# consumer -- and ``check_bounds`` passed an out-of-range list.  The
# tests hand-build ``params`` because ``params_pytree()`` never yields
# a list leaf; each asserts the list spelling against the array one.

_LIST_SPEC = {"v": ParamSpec(trainable=False, bounds=(-1.0, 1.0),
                             transform="logit")}
_PER_POSITION = {"v": [ParamSpec(trainable=False, bounds=(-1.0, 1.0)),
                       ParamSpec(bounds=(0.0, 4.0), transform="logit")]}


def _list_and_array(values=(0.0, 1.0)):
    as_list = {"v": [jnp.float32(x) for x in values]}
    as_array = {"v": jnp.array(values, dtype=jnp.float32)}
    return as_list, as_array


def test_trainable_mask_reaches_a_list_leaf():
    as_list, as_array = _list_and_array()
    assert trainable_mask(as_array, _LIST_SPEC) == {"v": False}
    assert trainable_mask(as_list, _LIST_SPEC) == {"v": [False, False]}
    assert trainable_mask({"v": tuple(as_list["v"])}, _LIST_SPEC) == {"v": (False, False)}
    assert trainable_mask(as_list, _PER_POSITION) == {"v": [False, True]}


def test_unconstrain_and_constrain_apply_the_transform_to_a_list_leaf():
    """A frozen leaf passes through; a trainable one gets its logit.
    The list spelling must match the array spelling element for
    element, and round-trip."""
    as_list, as_array = _list_and_array((0.25, 0.5))
    trainable = {"v": ParamSpec(bounds=(0.0, 1.0), transform="logit")}
    u_list = unconstrain(as_list, trainable)
    u_array = unconstrain(as_array, trainable)
    np.testing.assert_allclose(np.asarray(u_list["v"]), np.asarray(u_array["v"]))
    # logit(0.25) is not 0.25: the transform was applied, not skipped.
    assert not np.allclose(np.asarray(u_list["v"]), [0.25, 0.5])
    back = constrain(u_list, trainable)
    np.testing.assert_allclose(np.asarray(back["v"]), [0.25, 0.5], rtol=1e-6)
    # Frozen spec: identity both ways, as for an array leaf.
    frozen = unconstrain(as_list, _LIST_SPEC)
    np.testing.assert_array_equal(np.asarray(frozen["v"]), [0.25, 0.5])
    # Per-position: only the second position is trainable and transformed.
    per = unconstrain(as_list, _PER_POSITION)
    assert float(per["v"][0]) == 0.25
    assert float(per["v"][1]) != 0.5


def test_check_bounds_refuses_an_out_of_range_list_leaf():
    """The fail-open case: before the fix ``check_bounds`` accepted
    ``[5.0, 6.0]`` against ``bounds=(-1, 1)`` when the leaf was a list,
    while refusing the identical array."""
    as_list, as_array = _list_and_array((5.0, 6.0))
    with pytest.raises(ValueError, match=r"\['v'\]=.* above bound 1.0"):
        check_bounds(as_array, _LIST_SPEC)
    with pytest.raises(ValueError, match=r"\['v'\]\[0\]=5.0 above bound 1.0"):
        check_bounds(as_list, _LIST_SPEC)
    with pytest.raises(ValueError, match=r"\['v'\]\[1\]=6.0 above bound 4.0"):
        check_bounds({"v": [jnp.float32(0.5), jnp.float32(6.0)]}, _PER_POSITION)
    # In range passes in every spelling.
    ok_list, ok_array = _list_and_array((0.0, 0.5))
    check_bounds(ok_list, _LIST_SPEC)
    check_bounds(ok_array, _LIST_SPEC)
    check_bounds({"v": (jnp.float32(0.0), jnp.float32(0.5))}, _LIST_SPEC)


def test_a_spec_above_a_dict_level_still_means_the_default_for_the_maps():
    """``_spec_for`` stays lenient for the tree maps: a ``ParamSpec``
    placed above a *dict* level covers nothing (only sequence levels
    are covered), so the leaves below get the default spec -- the
    documented "missing entry" outcome, not an error.  ``fim`` is the
    caller that refuses this shape, on top of the same walk."""
    params = {"outer": {"a": jnp.float32(0.5)}}
    specs = {"outer": ParamSpec(trainable=False)}
    assert trainable_mask(params, specs) == {"outer": {"a": True}}
