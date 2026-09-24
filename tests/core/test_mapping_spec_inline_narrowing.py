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


# ---------------------------------------------------------------------------
# Complex values, integers beyond 2**53 and Decimals
# ---------------------------------------------------------------------------
#
# The same silent coercion had three more holes.  A complex payload lost its
# imaginary part with only a NumPy ComplexWarning (the long-double walk
# checked float dtypes only, so even ``np.clongdouble`` went through), and a
# Python int above 2**53 or a ``decimal.Decimal`` was rounded to float64
# without a word.  The coercion is unchanged; the refusal before it covers
# them.

import decimal  # noqa: E402
import warnings  # noqa: E402

_COMPLEX_PAYLOADS = {
    "a Python complex": [[0.5 + 0.25j], [1.0]],
    "complex64 scalars": [np.complex64(0.5 + 1j), np.complex64(2.0)],
    "a complex128 array": np.array([[0.5 + 1j, 2.0]], dtype=np.complex128),
    "clongdouble scalars": [[np.clongdouble(np.longdouble("0.1"))], [np.clongdouble(0.2)]],
    "an imaginary part of zero": [np.complex128(1.0 + 0j), np.complex128(2.0)],
}


def _complex_object_array():
    obj = np.empty(2, dtype=object)
    obj[0], obj[1] = 1.0, 2.0 + 3.0j
    return obj


@pytest.mark.parametrize("label", list(_COMPLEX_PAYLOADS) + ["an object array"])
@pytest.mark.parametrize("dtype", [None, "float64", "float32", "int64"])
def test_a_complex_payload_is_refused_with_or_without_a_dtype(label, dtype):
    """No accepted dtype is complex, so the imaginary part would be lost,
    not rounded: refused whatever the dtype, and no ComplexWarning escapes
    on the way (the refusal runs before the coercion)."""
    raw = _complex_object_array() if label == "an object array" else _COMPLEX_PAYLOADS[label]
    ref = {"inline": raw} if dtype is None else {"inline": raw, "dtype": dtype}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(PointReferenceError) as excinfo:
            normalise_point_reference(ref, name="source_points")
    message = str(excinfo.value)
    assert message.startswith("source_points: the inline points hold complex values")
    assert "discard the imaginary parts" in message
    assert "np.real(points).tolist()" in message
    assert "np.stack([np.real(points), np.imag(points)], axis=-1).tolist()" in message
    if dtype is None:
        assert "without a word" in message
    else:
        assert f"'dtype' {dtype!r}" in message and "ComplexWarning" in message


def test_the_complex_refusal_names_the_width_it_found():
    with pytest.raises(PointReferenceError, match=np.dtype(np.clongdouble).name):
        normalise_point_reference({"inline": _COMPLEX_PAYLOADS["clongdouble scalars"]})
    with pytest.raises(PointReferenceError, match="a Python complex"):
        normalise_point_reference({"inline": _COMPLEX_PAYLOADS["a Python complex"]})


def test_each_way_forward_from_a_complex_payload_is_accepted():
    points = np.array([[0.5 + 1j], [2.0 - 1j]])
    real = normalise_point_reference({"inline": np.real(points).tolist()})
    assert real == {"inline": [[0.5], [2.0]], "dtype": "float64"}
    split = normalise_point_reference(
        {"inline": np.stack([np.real(points), np.imag(points)], axis=-1).tolist()})
    assert split == {"inline": [[[0.5, 1.0]], [[2.0, -1.0]]], "dtype": "float64"}


_BIG = 2 ** 53 + 1          # the smallest positive integer float64 rounds


@pytest.mark.parametrize("payload,found", [
    ([[_BIG], [1]], _BIG),
    ([[1.0, 2.0], [3.0, -_BIG]], -_BIG),
    (np.array([[_BIG, 0]], dtype=np.int64), _BIG),
    ([np.int64(_BIG), np.int64(0)], _BIG),
    (np.array([2 ** 64 - 1], dtype=np.uint64), 2 ** 64 - 1),
    ([2 ** 70 + 1], 2 ** 70 + 1),
])
def test_an_integer_float64_cannot_represent_is_refused_without_a_dtype(payload, found):
    with pytest.raises(PointReferenceError) as excinfo:
        normalise_point_reference({"inline": payload}, name="source_points")
    message = str(excinfo.value)
    assert message.startswith(f"source_points: the inline points hold the integer {found}")
    assert "2**53 = 9007199254740992" in message
    assert f"would become {int(float(found))}" in message
    assert "rounded without a word" in message
    assert "add 'dtype': 'float64' to the reference" in message


def test_the_integer_refusal_suggests_a_dtype_that_really_holds_the_value():
    for value, keep in ((_BIG, "'dtype': 'int64'"), (2 ** 64 - 1, "'dtype': 'uint64'"),
                        (2 ** 70 + 1, "No accepted dtype holds it exactly")):
        with pytest.raises(PointReferenceError, match=keep):
            normalise_point_reference({"inline": [value]})
    with pytest.raises(PointReferenceError, match="overflows float64 altogether"):
        normalise_point_reference({"inline": [10 ** 400]})


def test_integers_float64_holds_exactly_are_unaffected():
    """2**53 itself, and powers of two far beyond it, are float64 values:
    they normalise exactly as before (the lookalikes of the refused ones)."""
    for value in (2 ** 53, -(2 ** 53), 2 ** 60, 2 ** 1000, 3, True):
        ref = normalise_point_reference({"inline": [value, 0]})
        assert ref == {"inline": [float(value), 0.0], "dtype": "float64"}
    arr = np.array([2 ** 53, 7], dtype=np.int64)
    assert normalise_point_reference({"inline": arr})["inline"] == [2.0 ** 53, 7.0]


def test_each_way_forward_from_a_big_integer_is_accepted():
    exact = normalise_point_reference({"inline": [[_BIG], [1]], "dtype": "int64"})
    assert exact == {"inline": [[_BIG], [1]], "dtype": "int64"}
    rounded = normalise_point_reference({"inline": [[_BIG], [1]], "dtype": "float64"})
    assert rounded == {"inline": [[float(2 ** 53)], [1.0]], "dtype": "float64"}
    converted = normalise_point_reference(
        {"inline": np.asarray([[_BIG], [1]], dtype=np.float64).tolist()})
    assert converted == rounded


def test_a_decimal_float64_cannot_represent_is_refused_without_a_dtype():
    d = decimal.Decimal("0.10000000000000000000001")
    with pytest.raises(PointReferenceError) as excinfo:
        normalise_point_reference({"inline": [[d], [1.0]]}, name="source_points")
    message = str(excinfo.value)
    assert message.startswith(f"source_points: the inline points hold the Decimal {d}")
    assert "it would become 0.1" in message
    assert "No point reference keeps decimal digits" in message
    assert "np.asarray(points, dtype=np.float64).tolist()" in message
    assert "add 'dtype': 'float64' to the reference" in message
    with pytest.raises(PointReferenceError, match="the Decimal 0.1,"):
        normalise_point_reference({"inline": [decimal.Decimal("0.1")]})


def test_decimals_float64_holds_exactly_and_the_ways_forward_are_accepted():
    exact = [decimal.Decimal("0.5"), decimal.Decimal("-3"), decimal.Decimal("1E+2")]
    assert normalise_point_reference({"inline": exact})["inline"] == [0.5, -3.0, 100.0]
    d = decimal.Decimal("0.1")
    assert normalise_point_reference({"inline": [d], "dtype": "float64"})["inline"] == [0.1]
    assert normalise_point_reference(
        {"inline": np.asarray([d], dtype=np.float64).tolist()})["inline"] == [0.1]
    # A non-finite Decimal is left to the finiteness check, which names it.
    with pytest.raises(PointReferenceError, match="must be finite"):
        normalise_point_reference({"inline": [decimal.Decimal("NaN")]})


def test_the_new_refusals_reach_every_entry_point_that_normalises():
    for raw in ([[_BIG], [1]], [[decimal.Decimal("0.1")], [1.0]], [[1 + 1j], [1.0]]):
        points = {"source_points": {"inline": raw},
                  "target_points": {"inline": [[0.0], [1.0]]}}
        with pytest.raises(PointReferenceError, match="the inline points hold"):
            MappingSpec.from_dict({"kind": "rbf", "points": points})
        with pytest.raises(PointReferenceError, match="the inline points hold"):
            make_point_resolver()({"inline": raw})
