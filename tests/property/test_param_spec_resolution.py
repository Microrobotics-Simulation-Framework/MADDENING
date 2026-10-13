"""Which ``ParamSpec`` governs a params leaf has one answer, whoever asks.

Three readers decide it: ``_validate_specs_mirror`` (the strict checker
``fim(scale="nominal")`` runs), ``_resolve_specs`` (the walk behind
``trainable_mask`` / ``unconstrain`` / ``constrain`` / ``check_bounds``)
and ``_spec_for`` (the per-path reader).  They used to be separate walks,
and on a ``namedtuple`` level they disagreed: the checker saw a tuple and
accepted a list of specs, the reader followed the ``GetAttrKey`` path and
read nothing, so ``check_bounds`` passed ``k=50`` against bounds
``(-5, 5)`` and ``trainable_mask`` ignored ``trainable=False``.  An
independent differential fuzz found 633 of 3,100 accepted random trees
resolving differently, every one involving a namedtuple.

The properties here are that fuzz, kept:

* **acceptance implies agreement** -- whenever a reader accepts a
  ``specs`` tree, every leaf's spec is the one an *independent* reference
  model assigns (:func:`_ref_assign`, the auditor's, not a call into the
  code under test), and the other readers and the tree maps agree;
* the auditor's own seeded fuzz, 4,000 trees, with zero disagreements
  and enough accepted namedtuple/dataclass trees that zero means
  something.

Pure pytree work on numpy scalars -- no JAX trace, no device -- so the
Hypothesis property runs at the ``EXAMPLES_CHEAP`` tier.
"""

from __future__ import annotations

import dataclasses
import random
from collections import OrderedDict, namedtuple

import jax
import numpy as np
from hypothesis import given, note, settings
from hypothesis import strategies as st

from maddening.core.params import (
    DEFAULT_SPEC,
    ParamSpec,
    _resolve_specs,
    _spec_for,
    _validate_specs_mirror,
    check_bounds,
    trainable_mask,
)
from tests.conftest import EXAMPLES_CHEAP

NT2 = namedtuple("NT2", ["a", "b"])


@jax.tree_util.register_dataclass
@dataclasses.dataclass
class DC:
    x: object
    y: object


class Keyless:
    """A pytree node registered without keys: ``jax.tree_util`` names its
    children with ``FlattenedIndexKey``, the one key type the old walks
    did not address at all."""

    def __init__(self, *children):
        self.children = children


jax.tree_util.register_pytree_node(
    Keyless, lambda k: (k.children, None), lambda _, c: Keyless(*c))


# ---------------------------------------------------------------------------
# The independent reference model
# ---------------------------------------------------------------------------


def _ref_assign(spec_node, param_node, prefix, out):
    """Which ``ParamSpec`` an accepted ``specs`` tree places at each leaf.

    The auditor's reference, transcribed rather than imported: a dict of
    specs is matched to children by the key each path entry carries
    (``.key``, ``.name`` or ``.idx``), a list/tuple by position, and a
    ``ParamSpec`` covers every leaf beneath it.  It does not know which
    spellings are *allowed* -- that is the readers' business -- only what
    an allowed one means.
    """
    if isinstance(spec_node, ParamSpec):
        for path, _ in jax.tree_util.tree_flatten_with_path(param_node)[0]:
            out[prefix + tuple(path)] = spec_node
        return
    entries = jax.tree_util.tree_flatten_with_path(
        param_node, is_leaf=lambda x: x is not param_node)[0]
    if isinstance(spec_node, dict):
        for path, child in entries:
            comp = path[0]
            key = getattr(comp, "key", getattr(comp, "name",
                                               getattr(comp, "idx", None)))
            if key in spec_node:
                _ref_assign(spec_node[key], child, prefix + (comp,), out)
        return
    if isinstance(spec_node, (list, tuple)):
        for (path, child), sv in zip(entries, spec_node):
            _ref_assign(sv, child, prefix + (path[0],), out)


def _assert_accepted_trees_agree(params, specs, per_leaf):
    """``per_leaf`` (a reader's answer) against the reference, and every
    other reader against both."""
    ref: dict = {}
    _ref_assign(specs, params, (), ref)
    entries = jax.tree_util.tree_flatten_with_path(params)[0]
    assert len(per_leaf) == len(entries)
    for (path, _), got in zip(entries, per_leaf):
        want = ref.get(tuple(path), DEFAULT_SPEC)
        where = jax.tree_util.keystr(path)
        assert got is want, (where, got, want)
        assert _spec_for(specs, path) is want, where
    mask = jax.tree_util.tree_leaves(trainable_mask(params, specs))
    assert mask == [s.trainable for s in per_leaf]


# ---------------------------------------------------------------------------
# Generated trees
# ---------------------------------------------------------------------------

_BOUNDS = [(None, None), (-1.0, None), (0.0, None), (2.0, 5.0), (None, 10.0),
           (-1.0, 10.0)]
_KINDS = ["scalar", "vector", "dict", "odict", "list", "tuple", "nt", "dc",
          "keyless", "none"]


@st.composite
def _spec(draw):
    return ParamSpec(trainable=draw(st.booleans()),
                     bounds=draw(st.sampled_from(_BOUNDS)),
                     description=str(draw(st.integers(0, 9))))


def _draw_pair(draw, depth):
    """``(params node, specs node)``; the specs node may be ``_OMIT``.

    Mostly the spellings the docs allow, sometimes a wrong one, so that
    both acceptance and refusal are exercised: a ``namedtuple`` gets a
    dict by field *or* a list by position (the audit's case), a dict gets
    an occasional stray key, a sequence an occasional wrong length.
    """
    kinds = _KINDS if depth < 3 else ["scalar", "vector"]
    kind = draw(st.sampled_from(kinds))
    wrong = draw(st.integers(0, 7)) == 0
    if kind == "scalar":
        leaf = np.float32(draw(st.sampled_from([0.0, 1.5, -2.0])))
        choices = ["S", "S", "omit"] + (["dict", "none"] if wrong else [])
        return leaf, draw(st.sampled_from(choices))
    if kind == "vector":
        n = draw(st.integers(1, 3))
        return np.arange(1, n + 1, dtype=np.float32), \
            "S" if not wrong else draw(st.sampled_from(["dict", "list"]))
    if kind == "none":
        return None, draw(st.sampled_from(["omit", "S", {}]))
    if kind in ("dict", "odict"):
        keys = [f"k{i}" for i in range(draw(st.integers(1, 3)))]
        pairs = [_draw_pair(draw, depth + 1) for _ in keys]
        params = (dict if kind == "dict" else OrderedDict)(
            zip(keys, (p for p, _ in pairs)))
        spec = {k: s for k, (_, s) in zip(keys, pairs)
                if not (isinstance(s, str) and s == "omit")}
        if wrong:
            which = draw(st.sampled_from(["stray", "paramspec", "list"]))
            if which == "stray":
                spec["stray"] = "S"
            elif which == "paramspec":
                spec = "S"
            else:
                spec = [s for _, s in pairs]
        return params, spec
    if kind in ("list", "tuple", "keyless"):
        pairs = [_draw_pair(draw, depth + 1)
                 for _ in range(draw(st.integers(1, 3)))]
        children = [p for p, _ in pairs]
        params = {"list": list, "tuple": tuple}.get(kind, lambda c: Keyless(*c))(children)
        spec = ["S" if isinstance(s, str) and s == "omit" else s
                for _, s in pairs]
        if kind != "keyless" and draw(st.integers(0, 4)) == 0:
            spec = "S"                     # one spec covering every position
        if wrong:
            which = draw(st.sampled_from(["short", "long", "dict"]))
            if which == "short":
                spec = spec[:-1] if isinstance(spec, list) else []
            elif which == "long":
                spec = (spec if isinstance(spec, list) else []) + ["S"]
            else:
                spec = {i: s for i, (_, s) in enumerate(pairs)}
        return params, spec
    # namedtuple / dataclass: named fields
    a, b = _draw_pair(draw, depth + 1), _draw_pair(draw, depth + 1)
    params = NT2(a[0], b[0]) if kind == "nt" else DC(a[0], b[0])
    names = ("a", "b") if kind == "nt" else ("x", "y")
    spec = {n: s for n, (_, s) in zip(names, (a, b))
            if not (isinstance(s, str) and s == "omit")}
    if draw(st.integers(0, 3)) == 0 or wrong:
        spec = draw(st.sampled_from([[a[1], b[1]], "S"]))
    return params, spec


def _materialise(draw, spec):
    """Replace the placeholders with real entries."""
    if isinstance(spec, str):
        return {"S": lambda: draw(_spec()), "dict": lambda: {"lo": 0.0},
                "list": lambda: [draw(_spec())], "none": lambda: None,
                "omit": lambda: draw(_spec())}[spec]()
    if isinstance(spec, dict):
        return {k: _materialise(draw, v) for k, v in spec.items()}
    if isinstance(spec, list):
        return [_materialise(draw, v) for v in spec]
    return spec


@st.composite
def params_and_specs(draw):
    params, spec = _draw_pair(draw, 0)
    spec = _materialise(draw, spec)
    return {"root": params}, {"root": spec}


class TestAcceptanceImpliesAgreement:

    @settings(max_examples=EXAMPLES_CHEAP)
    @given(pair=params_and_specs())
    def test_what_the_strict_checker_accepts_every_reader_reads_alike(
            self, pair):
        params, specs = pair
        note(f"params={params!r}\nspecs={specs!r}")
        try:
            per_leaf = _validate_specs_mirror(params, specs)
        except ValueError:
            return
        _assert_accepted_trees_agree(params, specs, per_leaf)

    @settings(max_examples=EXAMPLES_CHEAP)
    @given(pair=params_and_specs())
    def test_what_the_tree_maps_accept_they_read_as_the_reference_does(
            self, pair):
        """The maps' walk is lenient about entries that reach no leaf and
        strict about everything else; whatever it accepts it must read
        exactly as the reference model does -- which ignores unreached
        entries too."""
        params, specs = pair
        note(f"params={params!r}\nspecs={specs!r}")
        try:
            per_leaf = _resolve_specs(params, specs)
        except ValueError:
            return
        _assert_accepted_trees_agree(params, specs, per_leaf)

    @settings(max_examples=EXAMPLES_CHEAP)
    @given(pair=params_and_specs())
    def test_the_strict_checker_accepts_nothing_the_maps_refuse(self, pair):
        """Strict is lenient plus one refusal, never a different walk."""
        params, specs = pair
        try:
            _validate_specs_mirror(params, specs)
        except ValueError:
            return
        _resolve_specs(params, specs)


# ---------------------------------------------------------------------------
# The auditor's fuzz, verbatim in spirit: seeded, 4,000 trees
# ---------------------------------------------------------------------------


def _auditor_fuzz(trials=4000, seed=0):
    """The Wave D differential fuzz: random params trees over dict,
    OrderedDict, list, tuple, namedtuple and dataclass levels, a
    "natural" spec mirror for each (a namedtuple getting a list *or* a
    dict, sequences sometimes one covering spec, dict entries sometimes
    dropped), checked with :func:`_ref_assign`."""
    rng = random.Random(seed)

    def rand_spec():
        lo = rng.choice([None, -1.0, 0.0, 2.0])
        hi = rng.choice([None, 5.0, 10.0])
        if lo is not None and hi is not None and not lo < hi:
            hi = None
        return ParamSpec(trainable=rng.random() < 0.5, bounds=(lo, hi),
                         description=str(rng.random()))

    def rand_params(depth=0):
        kinds = ["scalar", "vec"] + (
            ["dict", "list", "tuple", "nt", "dc", "od"] if depth < 3 else [])
        k = rng.choice(kinds)
        if k == "scalar":
            return np.float32(rng.choice([0.0, 1.5, -2.0]))
        if k == "vec":
            return np.arange(rng.randint(1, 3), dtype=np.float32) + 1.0
        if k == "dict":
            return {f"k{i}": rand_params(depth + 1)
                    for i in range(rng.randint(1, 3))}
        if k == "od":
            return OrderedDict((f"z{i}", rand_params(depth + 1))
                               for i in range(rng.randint(1, 3)))
        if k == "list":
            return [rand_params(depth + 1) for _ in range(rng.randint(1, 3))]
        if k == "tuple":
            return tuple(rand_params(depth + 1)
                         for _ in range(rng.randint(1, 3)))
        if k == "nt":
            return NT2(rand_params(depth + 1), rand_params(depth + 1))
        return DC(rand_params(depth + 1), rand_params(depth + 1))

    def is_leaf(p):
        return jax.tree_util.all_leaves([p])

    def mirror(p):
        if is_leaf(p):
            return rand_spec()
        if isinstance(p, (list, tuple)) and not hasattr(p, "_fields"):
            if rng.random() < 0.2 and all(is_leaf(c) for c in p):
                return rand_spec()
            return [mirror(c) for c in p]
        if hasattr(p, "_fields"):
            if rng.random() < 0.5:
                return [mirror(c) for c in p]
            return {f: mirror(getattr(p, f)) for f in p._fields}
        if isinstance(p, DC):
            return {"x": mirror(p.x), "y": mirror(p.y)}
        d = {}
        for kk, c in p.items():
            if rng.random() < 0.8:
                d[kk] = mirror(c)
        return d

    stats = {"accepted": 0, "disagree": 0, "records_accepted": 0}
    for _ in range(trials):
        p = {"root": rand_params()}
        s = {"root": mirror(p["root"])}
        try:
            per_leaf = _validate_specs_mirror(p, s)
        except ValueError:
            continue
        stats["accepted"] += 1
        has_record = any(
            isinstance(k, jax.tree_util.GetAttrKey)
            for path, _ in jax.tree_util.tree_flatten_with_path(p)[0]
            for k in path)
        stats["records_accepted"] += has_record
        try:
            _assert_accepted_trees_agree(p, s, per_leaf)
        except AssertionError:
            stats["disagree"] += 1
    return stats


def test_the_auditors_fuzz_finds_no_accepted_tree_that_resolves_two_ways():
    """633 disagreements before the walks were shared; zero after.

    The two floors make zero mean something: most trees must still be
    accepted (a checker that refused everything would agree vacuously),
    and a real share of the accepted ones must contain a namedtuple or
    dataclass level -- the only kind that ever disagreed.
    """
    stats = _auditor_fuzz()
    assert stats["disagree"] == 0, stats
    assert stats["accepted"] >= 2500, stats
    assert stats["records_accepted"] >= 500, stats


# ---------------------------------------------------------------------------
# The audit's case, stated directly
# ---------------------------------------------------------------------------


def test_a_namedtuple_level_is_read_by_field_name_by_every_reader():
    """A dict keyed by field name is the spelling a namedtuple level is
    read by -- in ``check_bounds``, ``trainable_mask`` and ``_spec_for``
    alike -- and a list by position is refused by name rather than
    accepted and ignored."""
    KC = namedtuple("KC", ["k", "c"])
    params = {"spring": KC(k=np.float32(50.0), c=np.float32(3.0))}
    frozen = ParamSpec(trainable=False, bounds=(-5.0, 5.0))
    free = ParamSpec(bounds=(0.0, 10.0))
    by_name = {"spring": {"k": frozen, "c": free}}
    assert _validate_specs_mirror(params, by_name) == [frozen, free]
    assert trainable_mask(params, by_name) == {"spring": KC(k=False, c=True)}
    try:
        check_bounds(params, by_name)
    except ValueError as e:
        assert "['spring'].k=50.0 above bound 5.0" in str(e)
    else:
        raise AssertionError("check_bounds passed k=50 against (-5, 5)")
    by_position = {"spring": [frozen, free]}
    for reader in (_validate_specs_mirror, _resolve_specs, trainable_mask,
                   check_bounds):
        try:
            reader(params, by_position)
        except ValueError as e:
            assert "addressed by field name" in str(e), str(e)
        else:
            raise AssertionError(f"{reader.__name__} accepted a list for a "
                                 "namedtuple level")


def test_a_keyless_custom_node_is_read_by_position():
    """``FlattenedIndexKey`` children are positional: a list of specs by
    position reaches them, a dict is refused, and a covering
    ``ParamSpec`` is refused (coverage is for lists and tuples)."""
    params = {"n": Keyless(np.float32(1.0), np.float32(2.0))}
    a, b = ParamSpec(bounds=(0.0, 5.0)), ParamSpec(trainable=False)
    assert _validate_specs_mirror(params, {"n": [a, b]}) == [a, b]
    assert _spec_for({"n": [a, b]},
                     jax.tree_util.tree_flatten_with_path(params)[0][1][0]) is b
    for bad in ({"n": {0: a, 1: b}}, {"n": a}):
        try:
            _resolve_specs(params, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad!r} for a keyless node")
