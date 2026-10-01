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

import dataclasses
import os
from collections import namedtuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.params import (
    DEFAULT_SPEC,
    ParamSpec,
    _spec_for,
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
# from_dict refuses what it used to coerce
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw", ["false", "no", "0", "", None, 0, 1, 0.0, [], {}],
                         ids=["str_false", "str_no", "str_0", "str_empty", "null", "int_0",
                              "int_1", "float_0", "list", "dict"])
def test_from_dict_refuses_a_trainable_that_is_not_a_boolean(raw):
    """``bool()`` read the strings ``"false"``, ``"no"`` and ``"0"`` as
    trainable, and JSON ``null`` as frozen: a hand-edited or foreign
    document silently unfroze what it meant to freeze."""
    with pytest.raises(ValueError, match="trainable must be true or false"):
        ParamSpec.from_dict({"trainable": raw, "bounds": [0.0, None]})


@pytest.mark.parametrize("raw", [True, False])
def test_from_dict_takes_a_boolean_trainable_as_given(raw):
    assert ParamSpec.from_dict({"trainable": raw}).trainable is raw
    assert ParamSpec.from_dict({}).trainable is True           # the default


@pytest.mark.parametrize("bound", [True, False, "2", "1e3", [0.0]],
                         ids=["true", "false", "str_int", "str_float", "list"])
def test_from_dict_refuses_a_bound_that_is_not_a_number(bound):
    """``float()`` read ``true`` as ``1.0`` and ``"2"`` as ``2.0``, past the
    refusal ``__post_init__`` applies to exactly those values."""
    with pytest.raises(ValueError, match="not a real number"):
        ParamSpec.from_dict({"bounds": [bound, None]})
    with pytest.raises(ValueError, match="not a real number"):
        ParamSpec.from_dict({"bounds": [None, bound]})


def test_from_dict_reads_back_everything_to_dict_writes():
    """The refusals take nothing away from a document this class wrote."""
    inf = float("inf")
    for spec in (ParamSpec(), ParamSpec(trainable=False),
                 ParamSpec(bounds=(0.0, None), transform="log", units="m"),
                 ParamSpec(bounds=(-1.0, 2.0), transform="logit", description="d"),
                 ParamSpec(bounds=(-inf, inf)), ParamSpec(bounds=(0, 3))):
        back = ParamSpec.from_dict(spec.to_dict())
        assert back == spec and back.trainable is spec.trainable
        assert all(b is None or type(b) is float for b in back.bounds)


def test_a_saved_graph_with_a_string_trainable_is_refused_on_load():
    """``GraphManager.from_dict`` applies ``param_specs`` overrides through
    ``ParamSpec.from_dict``; a string there is an error naming the value,
    not a spec that silently reads as trainable."""
    from maddening.nodes.spring import SpringDamperNode

    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0))
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    gm.compile()
    config = gm.to_dict()
    assert config["param_specs"]["s"]["mass"]["trainable"] is False
    registry = {"SpringDamperNode": SpringDamperNode}
    reloaded = GraphManager.from_dict(config, registry)
    reloaded.compile()
    assert reloaded.param_specs()["nodes"]["s"]["mass"].trainable is False
    config["param_specs"]["s"]["mass"]["trainable"] = "false"
    with pytest.raises(ValueError, match="trainable must be true or false, got str 'false'"):
        GraphManager.from_dict(config, registry)


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


def test_a_spec_above_a_dict_level_is_refused_by_the_maps():
    """A ``ParamSpec`` placed above a *dict* level covers nothing (only
    sequence levels are covered).  The tree maps used to read that as
    "missing entry" and give the leaves below the default spec, which
    for ``check_bounds`` -- a safety check -- meant passing a value its
    author had bounded, and for ``trainable_mask`` marking a leaf its
    author had frozen.  They share ``fim``'s walk now and refuse it by
    key path.  ``_spec_for``, the per-path reader, stays lenient; the
    maps no longer go through it alone."""
    params = {"outer": {"a": jnp.float32(7.0)}}
    specs = {"outer": ParamSpec(trainable=False, bounds=(0.0, 1.0))}
    refusal = r"specs\['outer'\] is a ParamSpec but params\['outer'\] is a dict"
    for tree_map in (trainable_mask, unconstrain, constrain, check_bounds):
        with pytest.raises(ValueError, match=refusal):
            tree_map(params, specs)
    assert _spec_for(specs, jax.tree_util.tree_flatten_with_path(params)[0][0][0]) \
        is DEFAULT_SPEC


# ---------------------------------------------------------------------------
# Non-finite bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bounds", [
    (float("nan"), None), (None, float("nan")), (float("nan"), 1.0),
    (0.0, float("nan")),
], ids=["nan_lo", "nan_hi", "nan_lo_both", "nan_hi_both"])
def test_a_nan_bound_is_refused(bounds):
    """A one-sided ``NaN`` used to be accepted -- the ``lo < hi`` check
    only runs with both sides set -- and every comparison with ``NaN``
    is False: ``check_bounds`` passed ``-1e30`` and ``constrain``
    returned ``NaN``."""
    with pytest.raises(ValueError, match="NaN"):
        ParamSpec(bounds=bounds)


@pytest.mark.parametrize("bounds", [(float("inf"), None), (None, -float("inf"))],
                         ids=["plus_inf_lo", "minus_inf_hi"])
def test_an_infinity_pointing_inward_is_refused(bounds):
    with pytest.raises(ValueError, match="admits no value"):
        ParamSpec(bounds=bounds)


@pytest.mark.parametrize("transform, bounds", [
    ("logit", (0.0, float("inf"))),
    ("logit", (-float("inf"), 1.0)),
    ("log", (-float("inf"), None)),
])
def test_an_infinite_bound_under_a_transform_is_refused(transform, bounds):
    """``(0, inf)`` under ``"logit"`` mapped every value to ``u = -inf``
    and ``constrain(0)`` to ``NaN``."""
    with pytest.raises(ValueError, match="needs a finite"):
        ParamSpec(bounds=bounds, transform=transform)


@pytest.mark.parametrize("bad", ["1.0", True, [0.0]], ids=["string", "bool", "list"])
def test_a_bound_that_is_not_a_real_number_is_refused(bad):
    with pytest.raises(ValueError, match="not a real number"):
        ParamSpec(bounds=(bad, None))


def test_an_outward_infinity_means_exactly_none():
    """``(-inf, inf)`` is what serialised documents already hold for an
    unbounded spec, so it stays accepted and stays as given (the round
    trip keeps it) -- and every reader treats it as ``None``."""
    inf = float("inf")
    spec = ParamSpec(bounds=(-inf, inf))
    assert spec.bounds == (-inf, inf)
    assert ParamSpec.from_dict(spec.to_dict()).bounds == (-inf, inf)
    big = jnp.float32(3e38)
    spec.check(big)
    spec.check(-big)
    assert float(spec.to_constrained(big)) == float(big)
    ParamSpec(bounds=(0.0, inf), transform=None).check(jnp.float32(1e30))
    from maddening.sysid import _changes_no_column, _nominal_entry
    assert _nominal_entry(spec) == _nominal_entry(ParamSpec()) == (None, 0.0)
    assert _nominal_entry(ParamSpec(bounds=(2.0, inf))) == (None, 0.0)
    assert _changes_no_column(spec)


# ---------------------------------------------------------------------------
# transform="log" without a lower bound is measured from 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-1.0, 0.0, -1e-30], ids=["negative", "zero", "tiny_negative"])
def test_a_log_spec_without_a_lower_bound_refuses_a_value_at_or_below_zero(bad):
    """``log`` is measured from 0 when ``lo`` is ``None``: ``unconstrain``
    returned ``-inf`` for 0 and ``NaN`` below it, and ``check`` /
    ``check_bounds`` passed both."""
    spec = ParamSpec(transform="log")
    with pytest.raises(ValueError, match=r"k=.* below bound 0\.0 \(transform='log'"):
        spec.check(jnp.float32(bad), name="k")
    with pytest.raises(ValueError, match=r"below bound 0\.0"):
        check_bounds({"k": jnp.float32(bad)}, {"k": spec})
    with pytest.raises(ValueError, match=r"below bound 0\.0"):
        check_bounds({"k": jnp.asarray([1.0, bad], jnp.float32)}, {"k": spec})


def test_a_log_spec_without_a_lower_bound_accepts_what_it_can_unconstrain():
    spec = ParamSpec(transform="log")
    # Not a subnormal: XLA's CPU backend flushes them to zero, where log is
    # -inf, so refusing one is the right answer (and what check() gives).
    for value in (2.0, float(np.finfo(np.float32).tiny)):
        leaf = {"k": jnp.float32(value)}
        check_bounds(leaf, {"k": spec})
        u = unconstrain(leaf, {"k": spec})["k"]
        assert np.isfinite(float(u)), (value, float(u))
    # An explicit lower bound keeps its own rule.
    ParamSpec(bounds=(-2.0, None), transform="log").check(jnp.float32(-1.0))


# ---------------------------------------------------------------------------
# A ParamSpec over a record (namedtuple / dataclass) level
# ---------------------------------------------------------------------------

_Rec = namedtuple("_Rec", ["a", "b"])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class _RecDC:
    a: jax.Array
    b: jax.Array


@pytest.mark.parametrize("record", [
    _Rec(a=jnp.float32(7.0), b=jnp.float32(8.0)),
    _RecDC(a=jnp.float32(7.0), b=jnp.float32(8.0)),
], ids=["namedtuple", "dataclass"])
def test_a_spec_above_a_record_level_is_refused_by_the_maps(record):
    """A record's fields are named parameters, each needing its own entry:
    a ``ParamSpec`` covers a list/tuple level and nothing else.  Documented
    in ``check_bounds`` and in ``_spec_step``; covering a record would give
    ``b`` the bounds written for a vector and pass ``7.0`` against
    ``(0, 1)`` on nobody's say-so."""
    params = {"outer": record}
    specs = {"outer": ParamSpec(trainable=False, bounds=(0.0, 1.0))}
    refusal = (r"specs\['outer'\] is a ParamSpec but params\['outer'\] is a "
               r"\w+ record with fields \['a', 'b'\]")
    for tree_map in (trainable_mask, unconstrain, constrain, check_bounds):
        with pytest.raises(ValueError, match=refusal):
            tree_map(params, specs)
