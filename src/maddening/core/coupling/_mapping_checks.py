"""Checks of the geometry an interface-mapping factory is given.

Every factory in :mod:`maddening.core.coupling.mapping` and every closure
factory in :mod:`maddening.core.coupling.interface_mapping` builds its
operator once, on the host, from coordinates.  Each of them used to
compute *something* from coordinates it cannot map: a descending boundary
array gave a projection whose rows are all zero, a NaN among the points
made ``argmin`` select it for every target, a point set with the wrong
number of columns was broadcast.  Nothing warned, and the operator went
into the graph.

The functions here refuse such input with a ``ValueError`` that names the
argument and the first offending index.  Nothing is repaired: a boundary
array is never sorted or reversed and a point is never dropped, because
the field the operator is applied to keeps the caller's order.  A
``ValueError`` is what ``GraphManager.from_dict`` and
``load_graph_from_usd`` turn into a ``MappingRebuildError`` naming the
edge, so the same refusal holds for a mapping rebuilt from a config.

The checks read their input and return it as the float64 array the
factories already worked on; an accepted input gives the operator it gave
before, bit for bit.

A coordinate array that is a JAX tracer cannot be read on the host and is
not checked (:func:`is_traced`): the closure factories that accept one
keep doing so, on the caller's word that it is valid.  A complex array is
not judged here either: its coercion to float64 is NumPy's (with its
``ComplexWarning``), and the point-reference check that follows in the
mapping factories refuses the dtype by name.
"""

from __future__ import annotations

from typing import Any, Optional

import jax.core
import numpy as np


def is_traced(*arrays: Any) -> bool:
    """Is any of *arrays* a JAX tracer (a value with no host data to check)?"""
    return any(isinstance(a, jax.core.Tracer) for a in arrays)


def _is_complex(values: Any) -> bool:
    return np.asarray(values).dtype.kind == "c"


def _first_index(mask: np.ndarray) -> tuple:
    """Index of the first ``True`` of *mask*, in C order, as a tuple."""
    return tuple(int(i) for i in np.unravel_index(int(np.argmax(mask)), mask.shape))


def _show(index: tuple) -> str:
    return str(index[0]) if len(index) == 1 else str(index)


def first_nonfinite(array: Any) -> Optional[tuple]:
    """Index of the first NaN or infinity in *array*, or ``None``.

    Integer and boolean arrays have none.  An array ``numpy.isfinite``
    cannot judge (strings, objects) is left to the caller's own coercion.
    """
    arr = np.asarray(array)
    if arr.dtype.kind in "biu" or arr.size == 0:
        return None
    try:
        bad = ~np.isfinite(arr)
    except TypeError:
        return None
    return _first_index(bad) if bool(bad.any()) else None


def check_finite(name: str, array: Any) -> None:
    """Refuse a NaN or an infinity in *array*, naming its index."""
    at = first_nonfinite(array)
    if at is not None:
        arr = np.asarray(array)
        raise ValueError(
            f"{name} holds a non-finite value at index {_show(at)} ({arr[at].item()!r}); "
            f"every entry must be finite"
        )


def checked_boundaries(name: str, values: Any) -> np.ndarray:
    """Cell boundaries of a 1-D grid as a float64 array, or a ``ValueError``.

    The boundaries must be one-dimensional, hold at least two values (one
    cell), be finite and be strictly increasing.
    """
    arr = np.asarray(values, dtype=np.float64)
    if _is_complex(values):
        return arr
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array of cell boundaries, got shape "
            f"{arr.shape}"
        )
    if arr.size < 2:
        raise ValueError(
            f"{name} needs at least two boundaries (one cell), got {arr.size}"
        )
    check_finite(name, arr)
    not_rising = np.diff(arr) <= 0
    if bool(not_rising.any()):
        i = int(np.argmax(not_rising)) + 1
        raise ValueError(
            f"{name} must be strictly increasing, but {name}[{i}] = {float(arr[i])!r} is "
            f"not greater than {name}[{i - 1}] = {float(arr[i - 1])!r}.  The boundaries are not "
            f"sorted or reversed for you: the field keeps its cell order, so pass the "
            f"boundaries (and the field) in increasing order"
        )
    return arr


def check_ascending(name: str, values: Any) -> None:
    """Refuse 1-D coordinates that are non-finite or that decrease anywhere.

    Equal neighbours are accepted: an interpolation over a repeated
    coordinate is defined (the zero-width interval is skipped).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be a one-dimensional array of coordinates, got shape {arr.shape}"
        )
    check_finite(name, arr)
    falling = np.diff(arr) < 0
    if bool(falling.any()):
        i = int(np.argmax(falling)) + 1
        raise ValueError(
            f"{name} must be sorted in ascending order, but {name}[{i}] = "
            f"{float(arr[i])!r} is less than {name}[{i - 1}] = {float(arr[i - 1])!r}.  The "
            f"coordinates are not "
            f"sorted for you: the field keeps its order"
        )


def checked_points(name: str, values: Any, *, allow_empty: bool = False) -> np.ndarray:
    """A point set as an ``(n, d)`` float64 array, or a ``ValueError``.

    ``(n,)`` is read as *n* points on a line.  The set must be finite and
    have at least one coordinate column; it must hold at least one point
    unless *allow_empty* (the side an operator only evaluates at may be
    empty, the side it interpolates from may not).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if _is_complex(values) and arr.ndim == 2:
        return arr
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be an (n,) or (n, d) array of point coordinates, got shape "
            f"{np.shape(values)}"
        )
    if arr.shape[1] == 0:
        raise ValueError(
            f"{name} has shape {arr.shape}: its points have no coordinates"
        )
    if arr.shape[0] == 0 and not allow_empty:
        raise ValueError(
            f"{name} holds no points; there is nothing to map from"
        )
    at = first_nonfinite(arr)
    if at is not None:
        raise ValueError(
            f"{name} holds a non-finite coordinate at index {at[0]} "
            f"(point {arr[at[0]].tolist()}); every coordinate must be finite"
        )
    return arr


def check_same_dimension(source: np.ndarray, target: np.ndarray) -> None:
    """Refuse two ``(n, d)`` point sets with different ``d``."""
    if source.shape[1] != target.shape[1]:
        raise ValueError(
            f"source and target points must share a dimension, got "
            f"{source.shape[1]} and {target.shape[1]}"
        )
