"""An independent reference for multilinear gather / scatter on a uniform grid.

Written from the specification of the ``multilinear_grid`` mapping kind
alone, as explicit Python loops over points and corners, and sharing no
code with any JAX implementation.  ``tests/core/test_multilinear_grid_kernel.py``
compares the library's kind with it; the tests at the foot of that module
check *this* file against the kernel's own identities, so a fault here is
not taken for one in the library.

**The grid** is the lattice of sample points: value ``(i_0, .., i_{d-1})``
sits at ``x_a = origin[a] + i_a * spacing[a]``, ``d = len(shape)`` is 1, 2
or 3, and a flat field lists the lattice in C order (last axis fastest).

**The stencil of a point**, per axis: the index coordinate
``u = (x - origin) / spacing``; clamped onto ``[0, n - 1]`` (a point
outside the lattice's hull is treated as the nearest point of the hull,
coordinate by coordinate); ``i0 = min(floor(u), max(n - 2, 0))``,
``t = u - i0``, ``i1 = min(i0 + 1, n - 1)``.  The ``2**d`` corners are
listed in lexicographic order with axis 0 most significant; corner ``c``
has flat index ``sum_a stride[a] * (i1[a] if c[a] else i0[a])`` and weight
``prod_a (t[a] if c[a] else 1 - t[a])``.  A point with a non-finite
coordinate has every index at the corners of the cell at index 0 and every
weight NaN.

**Gather** (grid to points) is ``out[p] = sum_s W[p, s] * field[I[p, s]]``;
**scatter** (points to grid) adds ``W[p, s] * field[p]`` into cell
``I[p, s]``, points in order, corners in order: the transpose of gather.

**Exact arithmetic.**  A float is a rational number, so the index
coordinate, the cell and the weights are computed here with
:class:`fractions.Fraction` and have no rounding of their own: the
reference is the real-number answer for the float inputs it is given (in
whatever dtype they come), rounded once to float64 at the end.  That makes
it valid at any units -- a spacing of ``2**-125`` is as exact as one of
1.0 -- and means a difference from the library is the library's rounding,
which the tests bound.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class Grid:
    """A uniform lattice of sample points."""

    origin: tuple
    spacing: tuple
    shape: tuple

    def __post_init__(self):
        object.__setattr__(self, "origin", tuple(float(o) for o in self.origin))
        object.__setattr__(self, "spacing", tuple(float(h) for h in self.spacing))
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        assert len(self.origin) == len(self.spacing) == len(self.shape) and 1 <= self.d <= 3
        assert all(n >= 1 for n in self.shape) and all(h > 0 for h in self.spacing)

    @property
    def d(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def corners(self) -> int:
        return 2 ** self.d

    @property
    def strides(self) -> tuple:
        return tuple(int(np.prod(self.shape[a + 1:])) for a in range(self.d))

    def scaled(self, factor: float) -> "Grid":
        """Every length multiplied by *factor*."""
        return Grid(tuple(o * factor for o in self.origin),
                    tuple(h * factor for h in self.spacing), self.shape)

    def coordinates(self) -> np.ndarray:
        """``(size, d)`` float64 coordinates of the lattice points, in flat order."""
        out = np.zeros((self.size, self.d))
        for flat, idx in enumerate(itertools.product(*[range(n) for n in self.shape])):
            for a in range(self.d):
                out[flat, a] = self.origin[a] + idx[a] * self.spacing[a]
        return out

    def points_at(self, index_coordinates, dtype=np.float64) -> np.ndarray:
        """Float points at the given index coordinates (``(n_points, d)``)."""
        u = np.asarray(index_coordinates, np.float64).reshape(-1, self.d)
        return np.asarray(np.asarray(self.origin) + u * np.asarray(self.spacing), dtype)


def _points(grid: Grid, x) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1 and grid.d == 1:
        x = x.reshape(-1, 1)
    assert x.ndim == 2 and x.shape[1] == grid.d, (x.shape, grid.d)
    return x


def index_coordinates(grid: Grid, x) -> list:
    """Per point, the exact index coordinates ``[Fraction] * d`` -- or ``None``
    for a point with a non-finite coordinate."""
    x = _points(grid, x)
    out = []
    for p in range(x.shape[0]):
        if not all(math.isfinite(float(v)) for v in x[p]):
            out.append(None)
            continue
        out.append([(Fraction(float(x[p, a])) - Fraction(grid.origin[a]))
                    / Fraction(grid.spacing[a]) for a in range(grid.d)])
    return out


def _axis_cell(u: Fraction, n: int):
    """``(i0, i1, t)`` of one axis, clamped onto ``[0, n - 1]``."""
    uc = min(max(u, Fraction(0)), Fraction(n - 1))
    i0 = min(math.floor(uc), max(n - 2, 0))
    return i0, min(i0 + 1, n - 1), uc - i0


def stencil(grid: Grid, x):
    """``(I, W)``: flat indices ``(n_points, 2**d)`` and exact weights.

    ``W`` is an object array of :class:`~fractions.Fraction`; a point with
    a non-finite coordinate has ``None`` weights and the indices of the
    corners of the cell at index 0.
    """
    coords = index_coordinates(grid, x)
    K, strides = grid.corners, grid.strides
    index = np.zeros((len(coords), K), np.int64)
    weight = np.empty((len(coords), K), object)
    for p, u in enumerate(coords):
        if u is None:
            cells = [(0, min(1, n - 1), None) for n in grid.shape]
        else:
            cells = [_axis_cell(u[a], grid.shape[a]) for a in range(grid.d)]
        for s, corner in enumerate(itertools.product((0, 1), repeat=grid.d)):
            flat, w = 0, Fraction(1)
            for a in range(grid.d):
                i0, i1, t = cells[a]
                flat += strides[a] * (i1 if corner[a] else i0)
                if t is None:
                    w = None
                elif w is not None:
                    w = w * (t if corner[a] else 1 - t)
            index[p, s] = flat
            weight[p, s] = w
    return index, weight


def weights_float64(weight) -> np.ndarray:
    """The exact weights rounded to float64 (NaN for a non-finite point)."""
    out = np.zeros(weight.shape, np.float64)
    for pos in np.ndindex(*weight.shape):
        out[pos] = float("nan") if weight[pos] is None else float(weight[pos])
    return out


def _frac(v) -> Optional[Fraction]:
    v = float(v)
    return Fraction(v) if math.isfinite(v) else None


def _columns(field) -> np.ndarray:
    """A field as ``(n, C)`` (a plain ``(n,)`` field is one column)."""
    field = np.asarray(field)
    return field.reshape(field.shape[0], -1)


def gather(grid: Grid, field, x) -> np.ndarray:
    """Grid to points: the exact ``sum_s W[p, s] field[I[p, s]]``, as float64.

    *field* is ``(size,)`` or ``(size, C)`` in flat order, finite.  NaN for
    a point with a non-finite coordinate.
    """
    index, weight = stencil(grid, x)
    cols = _columns(field)
    assert cols.shape[0] == grid.size, (cols.shape, grid.size)
    out = np.zeros((index.shape[0], cols.shape[1]), np.float64)
    for p in range(index.shape[0]):
        for c in range(cols.shape[1]):
            if weight[p, 0] is None:
                out[p, c] = float("nan")
                continue
            acc = Fraction(0)
            for s in range(grid.corners):
                acc += weight[p, s] * Fraction(float(cols[index[p, s], c]))
            out[p, c] = float(acc)
    return out.reshape((index.shape[0],) + np.shape(field)[1:])


def scatter(grid: Grid, field, x) -> np.ndarray:
    """Points to grid: cell ``I[p, s]`` receives ``W[p, s] field[p]``, exactly, as float64.

    *field* is ``(n_points,)`` or ``(n_points, C)``, finite.  A point with a
    non-finite coordinate puts NaN in the corner cells at index 0.
    """
    index, weight = stencil(grid, x)
    cols = _columns(field)
    assert cols.shape[0] == index.shape[0], (cols.shape, index.shape)
    acc = [[Fraction(0)] * cols.shape[1] for _ in range(grid.size)]
    poisoned = np.zeros(grid.size, bool)
    for p in range(index.shape[0]):
        for s in range(grid.corners):
            cell = int(index[p, s])
            if weight[p, s] is None:
                poisoned[cell] = True
                continue
            for c in range(cols.shape[1]):
                acc[cell][c] += weight[p, s] * Fraction(float(cols[p, c]))
    out = np.asarray([[float(v) for v in row] for row in acc], np.float64)
    out[poisoned] = float("nan")
    return out.reshape((grid.size,) + np.shape(field)[1:])


def dense_matrix(grid: Grid, x) -> np.ndarray:
    """``H`` with ``H[p, I[p, s]] += W[p, s]`` (float64): gather is ``H @ field``
    and scatter ``H.T @ field``.  Finite points only."""
    index, weight = stencil(grid, x)
    acc = [[Fraction(0)] * grid.size for _ in range(index.shape[0])]
    for p in range(index.shape[0]):
        for s in range(grid.corners):
            acc[p][int(index[p, s])] += weight[p, s]
    return np.asarray([[float(v) for v in row] for row in acc], np.float64)


def support(grid: Grid, x) -> np.ndarray:
    """Boolean ``(n_points, size)``: the cells in each point's stencil."""
    index, _weight = stencil(grid, x)
    out = np.zeros((index.shape[0], grid.size), bool)
    for p in range(index.shape[0]):
        out[p, index[p]] = True
    return out


def magnitudes(grid: Grid, field, x) -> np.ndarray:
    """``S1[p] = sum_s |W[p, s]| |field[I[p, s]]|`` per point (and channel), float64."""
    index, weight = stencil(grid, x)
    w = np.abs(np.nan_to_num(weights_float64(weight)))
    cols = np.abs(_columns(field).astype(np.float64))
    out = np.einsum("ps,psc->pc", w, cols[index])
    return out.reshape((index.shape[0],) + np.shape(field)[1:])


def contributions(grid: Grid, x) -> np.ndarray:
    """``m_c``: how many (point, corner) pairs land in each cell."""
    index, _weight = stencil(grid, x)
    return np.bincount(index.ravel(), minlength=grid.size)


def project_onto_hull(grid: Grid, x) -> np.ndarray:
    """Each finite point moved to the nearest point of the lattice's hull,
    coordinate by coordinate, in *x*'s dtype."""
    x = _points(grid, x)
    lo = np.asarray(grid.origin, np.float64)
    hi = np.asarray([grid.origin[a] + (grid.shape[a] - 1) * grid.spacing[a]
                     for a in range(grid.d)], np.float64)
    return np.asarray(np.minimum(np.maximum(x.astype(np.float64), lo), hi), x.dtype)


def interior(grid: Grid, x, margin: float = 0.0) -> np.ndarray:
    """Which points lie inside the hull, at least *margin* cells from its faces.

    An axis with a single lattice point has no interior (its hull is that
    point), and every position along it reads the same value: it is not
    asked.
    """
    out = []
    for u in index_coordinates(grid, x):
        out.append(u is not None and all(
            grid.shape[a] == 1 or margin <= u[a] <= grid.shape[a] - 1 - margin
            for a in range(grid.d)))
    return np.asarray(out, bool)


def index_extent(grid: Grid, x) -> np.ndarray:
    """``max(1, |u_a|)`` per point and axis (float64): what the weight
    tolerances scale with.  1 for a non-finite point."""
    coords = index_coordinates(grid, x)
    out = np.ones((len(coords), grid.d))
    for p, u in enumerate(coords):
        if u is not None:
            out[p] = [max(1.0, abs(float(v))) for v in u]
    return out


def sample(grid: Grid, fn) -> np.ndarray:
    """``fn(coordinates) -> (size,)`` sampled on the lattice, float64."""
    return np.asarray(fn(grid.coordinates()), np.float64)


def roll(grid: Grid, field, shift: int, axis: int) -> np.ndarray:
    """``np.roll`` of a flat field along one lattice axis."""
    field = np.asarray(field)
    shaped = field.reshape(grid.shape + field.shape[1:])
    return np.roll(shaped, shift, axis=axis).reshape(field.shape)


def cell_variation(grid: Grid, field) -> np.ndarray:
    """Per axis, the field's largest change across one cell (float64)."""
    f = np.asarray(field, np.float64).reshape(grid.shape + np.shape(field)[1:])
    out = np.zeros(grid.d)
    for a in range(grid.d):
        if grid.shape[a] > 1:
            out[a] = float(np.max(np.abs(np.diff(f, axis=a))))
    return out


__all__: Sequence[str] = (
    "Grid", "index_coordinates", "stencil", "weights_float64", "gather", "scatter",
    "dense_matrix", "support", "magnitudes", "contributions", "project_onto_hull",
    "interior", "index_extent", "sample", "roll", "cell_variation",
)
