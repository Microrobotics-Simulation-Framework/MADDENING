"""``lost_as_integer`` decides whether a value is an integer of a type
without rounding either side.

The two tests it replaces were lossy themselves.  Comparing
``cast.astype(float64)`` with ``value.astype(float64)`` (the FMU's value
check) rounds both: an int64 above 2**53 equals its rounded neighbour, and
a float64 of exactly 2**63 equals the largest int64 wherever the
out-of-range cast saturates.  Casting there and back (``load_state``'s) is
a bijection between a signed and an unsigned type of one width: ``-1`` for
a ``uint64`` leaf came back as ``-1``, and loaded as 18446744073709551615.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from maddening.core._exact_integers import lost_as_integer

INTS = ("int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64")


def _lost(value, source, target) -> bool:
    return bool(lost_as_integer(np.asarray(value, dtype=source), np.dtype(target)))


@pytest.mark.parametrize("source, target", list(itertools.product(INTS, INTS)))
def test_an_integer_is_kept_exactly_when_the_target_type_holds_it(source, target):
    """Every pair of integer types, at the source's extremes, the target's
    extremes and one beyond each, in Python integers: lost exactly when
    the value is outside the target's range.  Signed against unsigned of
    one width included, in both directions."""
    s, t = np.iinfo(source), np.iinfo(target)
    candidates = {s.min, s.max, 0, 1, -1, t.min, t.max, t.min - 1, t.max + 1,
                  2 ** 53 + 1, -(2 ** 53) - 1}
    for value in sorted(v for v in candidates if s.min <= v <= s.max):
        assert _lost(value, source, target) is not (t.min <= value <= t.max), (
            value, source, target)


@pytest.mark.parametrize("target", INTS)
def test_a_float_is_an_integer_of_the_type_only_if_whole_and_in_range(target):
    """The bounds are the type's own, exactly: its minimum is taken and the
    float below it is not; the largest float below ``max + 1`` is taken and
    ``max + 1`` (a power of two, which a saturating cast turns into ``max``)
    is not."""
    t = np.iinfo(target)
    top = float(2 ** t.bits if t.min == 0 else 2 ** (t.bits - 1))        # max + 1, exact
    below_top = float(np.nextafter(top, 0.0))
    kept = [0.0, -0.0, 1.0, float(t.min), np.trunc(below_top)]
    lost = [top, top * 2, 0.5, -0.5, 1e300, -1e300, np.inf, -np.inf, np.nan,
            float(np.nextafter(float(t.min), -np.inf)) if t.min else -1.0]
    for value in kept:
        assert not _lost(value, np.float64, target), (value, target)
        assert int(np.float64(value).astype(target)) == int(value)       # and the cast is exact
    for value in lost:
        assert _lost(value, np.float64, target), (value, target)
    # a float32 and a float16 carrier are judged the same way
    assert not _lost(1.0, np.float32, target) and _lost(0.5, np.float32, target)
    assert _lost(np.float32(top), np.float32, target) and _lost(np.inf, np.float16, target)


def test_the_two_values_the_old_comparisons_took():
    """2**63 as a float64 for an int64 leaf, and 2**64 for a uint64 one --
    equal to the type's maximum through float64 wherever the cast saturates
    -- and the same-width sign change a cast there and back cannot see."""
    assert _lost(2.0 ** 63, np.float64, "int64") and _lost(2.0 ** 64, np.float64, "uint64")
    assert not _lost(-(2.0 ** 63), np.float64, "int64")
    for value, source, target in ((-1, "int64", "uint64"), (-1, "int32", "uint32"),
                                  (-1, "int8", "uint8"), (2 ** 63, "uint64", "int64"),
                                  (4_000_000_000, "uint32", "int32"), (200, "uint8", "int8")):
        assert _lost(value, source, target), (value, source, target)
        # the round trip the checkpoint loader used cannot see any of them
        a = np.asarray(value, dtype=source)
        with np.errstate(over="ignore"):
            assert a.astype(target).astype(source) == a
    # and an int64 above 2**53 is not its float64 neighbour
    assert not _lost(2 ** 53 + 1, "int64", "int64") and not _lost(2 ** 53 + 1, "uint64", "int64")
    assert _lost(2 ** 53 + 1, "int64", "int32")


def test_it_answers_elementwise_and_for_booleans_and_complex_values():
    a = np.asarray([0, 255, 256, -1, 7], np.int64)
    assert lost_as_integer(a, np.dtype("uint8")).tolist() == [False, False, True, True, False]
    assert lost_as_integer(np.zeros((2, 0), np.int64), np.dtype("int8")).shape == (2, 0)
    assert not lost_as_integer(np.asarray([True, False]), np.dtype("int8")).any()
    c = np.asarray([3 + 0j, 3 + 2j, 0.5 + 0j, 300 + 0j])
    assert lost_as_integer(c, np.dtype("uint8")).tolist() == [False, True, True, True]
    f = np.asarray([[1.0, 2.5], [np.nan, -3.0]])
    assert lost_as_integer(f, np.dtype("int8")).tolist() == [[False, True], [True, False]]
