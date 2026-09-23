"""An inline point reference with no ``"dtype"`` refuses extended-precision
values instead of rounding them to float64.

Without a ``"dtype"`` an inline payload is read as ``float64``.
``np.longdouble(...).tolist()`` yields ``np.longdouble`` scalars, not
Python floats, so a payload built from an extended-precision array --
its ``tolist()``, the array itself, rows of it, an object array holding
its scalars -- used to go through ``np.asarray(raw, dtype=float64)`` and
lose its extra bits without a word.  The coercion is left as it was (it
is the property-tested boundary for object and string payloads); a
refusal runs before it.  An explicit ``"dtype"`` is the caller deciding,
and goes through unchanged.
"""

import json

import numpy as np
import pytest

from maddening.core.coupling.mapping import rbf_mapping
from maddening.core.coupling.mapping_spec import (
    INLINE_ELEMENT_LIMIT,
    INLINE_POINT_LIMIT,
    MappingSpec,
    PointReferenceError,
    make_point_resolver,
    normalise_point_reference,
    reference_for_array,
)

_needs_extended_precision = pytest.mark.skipif(
    np.dtype(np.longdouble).itemsize <= 8,
    reason="this platform's np.longdouble is float64, so rounding it to "
           "float64 loses nothing and there is no narrowing to refuse")

#: A value float64 cannot hold: 1 + 2**-60 rounds to exactly 1.0.
_X = np.longdouble(1) + np.longdouble(2) ** -60
_LD_2D = np.array([[_X, 0.0], [1.0, 2.0]], dtype=np.longdouble)


def _payloads():
    """Every shape a long-double payload can arrive in, by name."""
    obj = np.empty(2, dtype=object)
    obj[0], obj[1] = _X, np.longdouble(2.0)
    return {
        "scalars in a list": [_X, np.longdouble(2.0)],
        "tolist of a 2-D array": _LD_2D.tolist(),
        "the array itself": _LD_2D,
        "a list of array rows": [_LD_2D[0], _LD_2D[1]],
        "one scalar among floats": [[0.5, 1.5], [2.5, _X]],
        "an object array of scalars": obj,
    }


@_needs_extended_precision
def test_the_fixture_value_really_loses_precision_as_float64():
    """The refusal is about a real loss: the fixture value is not a float64."""
    assert _X != np.longdouble(float(_X))
    assert float(_X) == 1.0


@_needs_extended_precision
@pytest.mark.parametrize("label", list(_payloads()))
def test_a_long_double_payload_without_a_dtype_is_refused(label):
    with pytest.raises(PointReferenceError, match="extended-precision values"):
        normalise_point_reference({"inline": _payloads()[label]}, name="source_points")


@_needs_extended_precision
def test_the_plain_list_form_is_refused_too():
    """A bare list is shorthand for ``{"inline": [...]}`` with no dtype."""
    with pytest.raises(PointReferenceError, match="extended-precision values"):
        normalise_point_reference(_LD_2D.tolist())


@_needs_extended_precision
def test_a_dtype_of_none_is_not_an_explicit_dtype():
    """``"dtype": None`` resolves to numpy's float64 default just as a
    missing key does, so it must not bypass the refusal."""
    with pytest.raises(PointReferenceError, match="extended-precision values"):
        normalise_point_reference({"inline": _LD_2D.tolist(), "dtype": None})


@_needs_extended_precision
def test_the_refusal_names_the_dtype_and_the_ways_forward():
    """The message says what was found, that no reference form can keep
    the precision, and three things the format really supports."""
    with pytest.raises(PointReferenceError) as excinfo:
        normalise_point_reference({"inline": _LD_2D.tolist()}, name="source_points")
    message = str(excinfo.value)
    assert message.startswith("source_points: ")
    assert np.dtype(np.longdouble).name in message       # what was found
    assert f"about {np.finfo(np.longdouble).precision} significant digits" in message
    assert "names no 'dtype'" in message                 # why it fired
    assert "Extended precision cannot be kept by any point reference" in message
    assert "8-byte elements" in message
    # the three ways forward, each one something the format accepts
    assert "np.asarray(points, dtype=np.float64).tolist()" in message
    assert "add 'dtype': 'float64' to the reference" in message
    assert "{'asset': '<file>.npy'}" in message
    assert f"more than {INLINE_POINT_LIMIT} points" in message


@_needs_extended_precision
def test_each_suggested_way_forward_is_accepted():
    """The alternatives the message offers really work, and the explicit
    dtype genuinely rounds (the caller asked for it)."""
    explicit = normalise_point_reference({"inline": _LD_2D.tolist(), "dtype": "float64"})
    assert explicit == {"inline": [[1.0, 0.0], [1.0, 2.0]], "dtype": "float64"}
    converted = normalise_point_reference(
        {"inline": np.asarray(_LD_2D, dtype=np.float64).tolist()})
    assert converted == explicit
    assert json.loads(json.dumps(converted)) == converted


@_needs_extended_precision
def test_the_refusal_reaches_every_entry_point_that_normalises():
    """A config's spec, the point resolver, and a factory's recorded
    reference all go through the same normalisation."""
    points = {"source_points": {"inline": _LD_2D.tolist()},
              "target_points": {"inline": [[0.0, 0.0], [1.0, 1.0]]}}
    with pytest.raises(PointReferenceError, match="extended-precision values"):
        MappingSpec.from_dict({"kind": "rbf", "points": points})
    with pytest.raises(PointReferenceError, match="extended-precision values"):
        make_point_resolver()({"inline": _LD_2D.tolist()})
    with pytest.raises(PointReferenceError, match="extended-precision values"):
        reference_for_array(np.asarray(_LD_2D, dtype=np.float64),
                            {"inline": _LD_2D.tolist()}, name="source_points")


@_needs_extended_precision
def test_a_long_double_at_the_far_end_of_a_full_payload_is_still_found():
    """The walk is bounded by the element limit, and the bound must not
    cut off a payload of exactly the largest accepted size: 64 rows of 16
    numbers with the only long double in the element visited last."""
    rows, cols = INLINE_POINT_LIMIT, INLINE_ELEMENT_LIMIT // INLINE_POINT_LIMIT
    payload = [[float(r * cols + c) for c in range(cols)] for r in range(rows)]
    payload[0][0] = _X
    with pytest.raises(PointReferenceError, match="extended-precision values"):
        normalise_point_reference({"inline": payload})
    payload[0][0] = 0.0
    assert normalise_point_reference({"inline": payload})["dtype"] == "float64"


@_needs_extended_precision
def test_past_the_walk_bound_a_long_double_payload_is_still_refused():
    """Beyond the bound the walk stops looking, which is safe only because
    anything that large is refused after coercion anyway: nothing past the
    bound can be accepted narrowed."""
    too_many = [[1.0] * INLINE_ELEMENT_LIMIT + [_X]]
    with pytest.raises(PointReferenceError, match="INLINE_ELEMENT_LIMIT"):
        normalise_point_reference({"inline": too_many})


@pytest.mark.parametrize("payload,expected", [
    ([[0.1, 0.2], [0.3, 0.4]], [[0.1, 0.2], [0.3, 0.4]]),
    ([1, 2, 3], [1.0, 2.0, 3.0]),
    ([np.float32(0.5), np.float32(1.5)], [0.5, 1.5]),
    (np.array([[0.25, 0.5]], dtype=np.float64), [[0.25, 0.5]]),
    ([np.float64(0.1), 2.0], [0.1, 2.0]),
])
def test_a_float64_or_narrower_payload_without_a_dtype_is_unaffected(payload, expected):
    """Nothing legitimate is caught: payloads that float64 holds exactly
    normalise as before, to float64."""
    assert normalise_point_reference({"inline": payload}) == {
        "inline": expected, "dtype": "float64"}


def test_an_explicit_narrower_dtype_is_unaffected():
    ref = normalise_point_reference({"inline": [[0.0, 1.0]], "dtype": "float32"})
    assert ref == {"inline": [[0.0, 1.0]], "dtype": "float32"}


def test_a_factory_still_inlines_float64_points_with_no_reference():
    src = np.linspace(0.0, 1.0, 4).reshape(-1, 1)
    tgt = np.linspace(0.0, 1.0, 3).reshape(-1, 1)
    spec = rbf_mapping(src, tgt).spec
    assert spec.points["source_points"] == {"inline": src.tolist(), "dtype": "float64"}
