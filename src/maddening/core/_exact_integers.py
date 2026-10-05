"""Is a value exactly an integer of a given type?  Answered without rounding.

Every door that writes a number into an integer leaf -- an FMU ``set``, an
FMU-state archive, ``FmuSidecar.set_params``, a checkpoint -- has to refuse
one the leaf cannot hold, and the obvious tests for that are lossy
themselves:

* comparing ``cast.astype(float64)`` with ``value.astype(float64)`` rounds
  both sides, so an int64 above 2**53 equals its rounded neighbour, and a
  float64 of exactly 2**63 (2**64 for an unsigned leaf) equals the type's
  largest value wherever an out-of-range float-to-integer cast saturates
  (it wraps on x86 and saturates on ARM: the answer depended on the
  machine);
* casting to the leaf's type and back to the value's (``cast.astype(a.dtype)
  != a``) is a bijection between a signed and an unsigned type of one
  width, so ``-1`` for a ``uint64`` leaf came back as ``-1`` and loaded as
  18446744073709551615.

:func:`lost_as_integer` decides by range and wholeness instead: an integer
is compared as an integer against the target's own bounds, and a float must
be whole and inside ``[min, max + 1)``, both ends of which are zero or a
power of two and therefore exact in any binary float type.  The cast that
follows a passing check is exact on every platform.
"""

from __future__ import annotations

import math

import numpy as np


def lost_as_integer(a: np.ndarray, target: np.dtype) -> np.ndarray:
    """Elementwise: is this entry of ``a`` *not* a value of the integer type
    ``target``?

    Parameters
    ----------
    a : numpy.ndarray
        Integers, booleans, floats or complex numbers of any width.
    target : numpy.dtype
        An integer dtype.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` where a cast to ``target`` would not hold the entry
        exactly: out of ``target``'s range, not whole, not finite, or with
        an imaginary part.  ``a``'s shape.
    """
    info = np.iinfo(target)
    kind = a.dtype.kind
    if kind == "b":
        return np.zeros(a.shape, bool)
    if kind in "iu":
        source = np.iinfo(a.dtype)
        lost = np.zeros(a.shape, bool)
        # Each bound is compared only where the source type reaches past
        # it, and is then a value of the source type: no promotion, and in
        # particular none to float64 (which a signed / unsigned 64-bit pair
        # promotes to).
        if source.min < info.min:
            lost |= a < a.dtype.type(info.min)
        if source.max > info.max:
            lost |= a > a.dtype.type(info.max)
        return lost
    if kind == "c":
        return (a.imag != 0) | lost_as_integer(np.asarray(a.real), target)
    signed = info.min < 0
    past = math.ldexp(1.0, info.bits - (1 if signed else 0))     # max + 1 = 2**k, exact
    low = -past if signed else 0.0
    wide = a.astype(np.float64) if a.dtype.itemsize < 8 else a
    with np.errstate(invalid="ignore"):
        return ~((wide >= low) & (wide < past) & (wide == np.trunc(wide)))


__all__ = ["lost_as_integer"]
