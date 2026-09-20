"""Non-finite numbers leave MADDENING as valid JSON, and come back.

``MADD-ANO-006``: ``json.dumps`` defaults to ``allow_nan=True`` and
writes the bare tokens ``NaN``, ``Infinity`` and ``-Infinity``, which RFC
8259 has no literal for.  Python read them back, so every round trip
inside MADDENING was exact and nothing here noticed; every conforming
reader rejected the document.

Every assertion below parses with a ``parse_constant`` that *raises*.
Python's default leniency is what hid the defect, so a test that relied
on it would hide the regression too.

This file covers the codec and the ``to_dict`` / ``from_dict`` surface;
``tests/usd/test_usd_params.py`` covers the stage and
``tests/fmi/test_non_finite_json_tokens.py`` the FMI wire.
"""

import json
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.serialization.config import from_dict, to_dict
from maddening.serialization.json_codec import (
    INF_TOKEN,
    NAN_TOKEN,
    NEG_INF_TOKEN,
    NON_FINITE_TOKENS,
    decode_non_finite,
    dumps,
    encode_non_finite,
    loads,
)

REGISTRY = {"SpringDamperNode": SpringDamperNode}


def _refuse(token):
    """A strict reader: RFC 8259 has no literal for these."""
    raise ValueError(f"non-standard JSON token {token!r}")


def strict_loads(text):
    """``json.loads`` that rejects the three non-standard tokens."""
    return json.loads(text, parse_constant=_refuse)


def _same_number(a, b) -> bool:
    """Equality that treats two NaNs as the same value."""
    if isinstance(a, float) and isinstance(b, float):
        return (math.isnan(a) and math.isnan(b)) or a == b
    return type(a) is type(b) and a == b


# ----------------------------------------------------------------- codec

def test_the_three_non_finite_floats_encode_to_their_quoted_tokens():
    assert encode_non_finite(math.nan) == NAN_TOKEN
    assert encode_non_finite(math.inf) == INF_TOKEN
    assert encode_non_finite(-math.inf) == NEG_INF_TOKEN


def test_decoding_an_encoded_tree_restores_every_value():
    tree = {"a": math.nan, "b": [math.inf, -math.inf, 1.5], "c": {"d": 0.0},
            "e": [], "f": None, "g": True, "h": 7, "i": "text"}
    back = decode_non_finite(encode_non_finite(tree))
    assert math.isnan(back["a"])
    assert back["b"][0] == math.inf and back["b"][1] == -math.inf
    assert back["b"][2] == 1.5
    assert back["c"] == {"d": 0.0} and back["e"] == []
    assert back["f"] is None and back["g"] is True and back["h"] == 7
    assert back["i"] == "text"


def test_a_finite_tree_is_returned_unchanged_rather_than_copied():
    """The walk costs nothing on the overwhelmingly common input.

    Identity, not equality: a config with a large inlined point set is
    walked on every ``to_dict``, and rebuilding it each time would double
    the memory for no change.
    """
    tree = {"nodes": [{"params": {"k": 1.0}}], "edges": []}
    assert encode_non_finite(tree) is tree
    assert decode_non_finite(tree) is tree


def test_a_bool_is_not_mistaken_for_a_float():
    """``bool`` is an ``int``; ``isinstance(True, float)`` is False.

    Redundant by construction, and kept: if the leaf test were widened to
    "anything with ``__float__``", ``True`` would start encoding as a
    number and nothing else here would notice.
    """
    out = encode_non_finite({"flag": True, "off": False})
    assert out["flag"] is True and out["off"] is False


def test_dict_keys_are_never_encoded_or_decoded():
    """A key is a name, never a number.  Encoding one would rename it."""
    assert encode_non_finite({NAN_TOKEN: 1.0}) == {NAN_TOKEN: 1.0}
    assert decode_non_finite({NAN_TOKEN: 1.0}) == {NAN_TOKEN: 1.0}


@pytest.mark.parametrize("token", sorted(NON_FINITE_TOKENS))
def test_a_string_that_collides_with_a_token_is_refused_at_write_time(token):
    """The disambiguation, stated as a test.

    A blind decoder cannot tell ``float('nan')`` from the string
    ``"NaN"``.  Rather than decode it wrongly later, the encoder refuses
    the document now and names the path to the offending value.
    """
    with pytest.raises(ValueError) as exc:
        encode_non_finite({"nodes": [{"params": {"mode": token}}]})
    assert "$.nodes[0].params.mode" in str(exc.value)
    assert token in str(exc.value)


def test_dumps_refuses_a_non_finite_value_its_walk_cannot_reach():
    """``allow_nan=False`` closes the hole from the other side.

    The walk covers dicts, lists and tuples.  A float produced by a
    ``default=`` hook is not walked, and must fail loudly rather than
    reintroduce a bare token.
    """
    class _Odd:
        pass

    with pytest.raises(ValueError):
        dumps({"x": _Odd()}, default=lambda o: math.inf)


def test_loads_reads_both_the_quoted_and_the_bare_spelling():
    """Backward compatibility, pinned on literal documents."""
    assert math.isnan(loads('{"v": NaN}')["v"])
    assert loads('{"v": Infinity}')["v"] == math.inf
    assert loads('{"v": -Infinity}')["v"] == -math.inf
    assert math.isnan(loads('{"v": "NaN"}')["v"])
    assert loads('{"v": "Infinity"}')["v"] == math.inf
    assert loads('{"v": "-Infinity"}')["v"] == -math.inf


# ------------------------------------------------------ to_dict/from_dict

def _graph(**params):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, **params))
    return gm


def test_a_non_finite_scalar_param_serialises_as_strict_json():
    gm = _graph(damping=math.inf, mass=math.nan)
    text = json.dumps(to_dict(gm))

    stored = strict_loads(text)
    params = stored["nodes"][0]["params"]
    assert params["damping"] == INF_TOKEN and params["mass"] == NAN_TOKEN

    back = from_dict(strict_loads(text), REGISTRY).effective_node_params("s")
    assert float(back["damping"]) == math.inf
    assert math.isnan(float(back["mass"]))


def test_an_infinite_param_spec_bound_serialises_as_strict_json():
    """The route MADD-ANO-006 was first found through.

    ``ParamSpec(bounds=(None, None))`` is the idiomatic unbounded spec and
    was always valid JSON; ``(-inf, inf)`` says the same thing and was
    not.
    """
    gm = _graph()
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(-math.inf, math.inf)))
    text = json.dumps(to_dict(gm))

    stored = strict_loads(text)
    assert stored["param_specs"]["s"]["stiffness"]["bounds"] == [
        NEG_INF_TOKEN, INF_TOKEN]

    gm2 = from_dict(strict_loads(text), REGISTRY)
    assert gm2.param_spec_overrides()["s"]["stiffness"].bounds == (
        -math.inf, math.inf)


def test_a_none_bound_still_serialises_as_null_not_as_a_token():
    """``None`` means "no bound" and is not a non-finite number.

    The two spellings stay distinguishable through a round trip: a spec
    written unbounded must not come back bounded at infinity.
    """
    gm = _graph()
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(None, None)))
    stored = strict_loads(json.dumps(to_dict(gm)))
    assert stored["param_specs"]["s"]["stiffness"]["bounds"] == [None, None]
    assert from_dict(stored, REGISTRY).param_spec_overrides()[
        "s"]["stiffness"].bounds == (None, None)


def test_a_zero_size_array_param_survives_beside_non_finite_ones():
    """The empty container and the encoded scalars do not interfere.

    ``[]`` has nothing to encode and must not become something else, and
    the walk must still reach the non-finite values around it.
    """
    gm = GraphManager()
    node = SpringDamperNode("s", 0.01, stiffness=30.0)
    node.params["empty"] = np.zeros((0,)).tolist()
    node.params["mixed"] = [math.inf, 1.0, math.nan]
    gm.add_node(node)

    stored = strict_loads(json.dumps(to_dict(gm)))
    params = stored["nodes"][0]["params"]
    assert params["empty"] == []
    assert params["mixed"] == [INF_TOKEN, 1.0, NAN_TOKEN]


def test_a_config_written_before_040_still_loads():
    """A literal pre-0.4.0 document, bare tokens and all.

    ``json.loads`` parses the bare tokens into floats before ``from_dict``
    ever sees them, so nothing has to recognise the old spelling -- but
    nothing may *reject* it either, and that is what this pins.
    """
    legacy = (
        '{"nodes": [{"type": "SpringDamperNode", "name": "s", '
        '"timestep": 0.01, "params": {"stiffness": NaN, "damping": Infinity, '
        '"mass": 1.0, "rest_length": 1.0, "initial_position": -Infinity, '
        '"initial_velocity": 0.0}}], '
        '"param_specs": {"s": {"stiffness": {"trainable": true, '
        '"bounds": [-Infinity, Infinity], "transform": null, '
        '"description": "", "units": ""}}}, '
        '"edges": [], "external_inputs": []}'
    )
    gm = from_dict(json.loads(legacy), REGISTRY)
    params = gm.effective_node_params("s")
    assert math.isnan(float(params["stiffness"]))
    assert float(params["damping"]) == math.inf
    assert float(params["initial_position"]) == -math.inf
    assert gm.param_spec_overrides()["s"]["stiffness"].bounds == (
        -math.inf, math.inf)


def test_a_reloaded_non_finite_config_serialises_to_the_same_document():
    """Idempotence: the second write equals the first, character for character."""
    gm = _graph(damping=math.inf, mass=math.nan)
    first = json.dumps(to_dict(gm), sort_keys=True)
    second = json.dumps(to_dict(from_dict(strict_loads(first), REGISTRY)),
                        sort_keys=True)
    assert first == second


# ----------------------------------------------------------- the property

#: JSON-shaped trees whose scalars include the non-finite floats and the
#: strings that look like them.  ``allow_nan=True`` here is the whole
#: point: these are the values the codec exists for.
_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(2 ** 53), max_value=2 ** 53),
    st.floats(allow_nan=True, allow_infinity=True),
    st.sampled_from(sorted(NON_FINITE_TOKENS)),
    st.text(max_size=8),
)

_TREES = st.recursive(
    _SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.dictionaries(st.text(max_size=6), children, max_size=4),
    ),
    max_leaves=12,
)


def _contains_token_string(tree) -> bool:
    if isinstance(tree, str):
        return tree in NON_FINITE_TOKENS
    if isinstance(tree, dict):
        return any(_contains_token_string(v) for v in tree.values())
    if isinstance(tree, list):
        return any(_contains_token_string(v) for v in tree)
    return False


def _assert_same_tree(original, restored):
    if isinstance(original, dict):
        assert set(original) == set(restored)
        for key in original:
            _assert_same_tree(original[key], restored[key])
    elif isinstance(original, list):
        assert len(original) == len(restored)
        for a, b in zip(original, restored):
            _assert_same_tree(a, b)
    else:
        assert _same_number(original, restored), (original, restored)


@given(_TREES)
def test_any_tree_dumps_to_a_document_a_strict_parser_accepts(tree):
    """The invariant the whole branch exists for.

    Either the document is valid JSON and reads back to the same values,
    or writing it raised -- and it raises only for the one ambiguous
    case, a string that reads like a token.
    """
    try:
        text = dumps(tree)
    except ValueError as exc:
        assert "cannot be written to JSON" in str(exc)
        assert _contains_token_string(tree)
        return

    assert not _contains_token_string(tree)
    strict_loads(text)                    # no bare token survived
    _assert_same_tree(tree, loads(text))


# ``width=32`` because the param lands in a float32 leaf: a float64 that
# overflows float32 is a node-level narrowing question, not a codec one,
# and the full float64 range is covered by the tree property above.
@given(st.floats(width=32, allow_nan=True, allow_infinity=True))
def test_any_float_round_trips_through_a_param_value(value):
    """Every float, not only the three interesting ones.

    A finite float must be untouched by the codec, so this also pins that
    the fix did not start rewriting ordinary numbers.
    """
    gm = _graph(damping=value)
    text = json.dumps(to_dict(gm))
    back = float(from_dict(strict_loads(text), REGISTRY)
                 .effective_node_params("s")["damping"])
    assert _same_number(value, back)
