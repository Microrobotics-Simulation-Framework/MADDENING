"""Multilinear gather / scatter between a uniform grid and moving points.

The reference geometry-dependent mapping kind, ``multilinear_grid``
(**experimental**).  A geometry-dependent mapping reads, besides the field
it transfers, a *geometry*: a state field of the edge's own source or
target node, named on the edge with ``add_edge(..., geometry=(anchor,
field))`` and read by the graph at the time level the edge's value has
(see the interface-mapping guide).  Here the geometry is the positions of
``n_points`` points, shape ``(n_points, d)`` (``(n_points,)`` is taken too
on a one-dimensional grid), and the other side is a uniform grid::

    fluid_to_body = multilinear_grid_mapping(origin, spacing, shape,
                                             n_points=64, mode="consistent")
    gm.add_edge("fluid", "body", "velocity", "velocity_at_markers",
                mapping=fluid_to_body, geometry=("target", "marker_position"))

**The grid is the lattice of sample points.**  Value ``(i_0, .., i_{d-1})``
sits at ``x_a = origin[a] + i_a * spacing[a]``; a cell-centred solver
passes the coordinate of its first cell centre as ``origin``.  There are
one to three axes, and the flat index is C order (last axis fastest).

**Two modes, one stencil.**  Each point reads the ``2**d`` lattice points
around it with the multilinear weights of its position in that cell.

* ``mode="consistent"`` *gathers* grid to points:
  ``out[p] = sum_s W[p, s] * field[I[p, s]]``.  A field that is multilinear
  in the coordinates is reproduced.
* ``mode="conservative"`` *scatters* points to grid:
  ``out[I[p, s]] += W[p, s] * field[p]``.  It preserves the plain sum
  to the rounding of the geometry's dtype (``sum(out) == sum(field)`` to
  that rounding: the weights of a point are computed in the geometry's
  dtype and sum to one there): it deposits amounts.  It divides by no cell volume and applies no
  quadrature weight; turning the result into a density belongs to a node
  or to the edge's ``transform``.

Both use the same indices and weights, so each is exactly the transpose of
the other, and :meth:`MultilinearGridMapping.apply_T` is the other mode.

**Points outside the grid are clamped** to the nearest point of the
lattice's hull, coordinate by coordinate: gather extrapolates constantly,
scatter deposits on the boundary, the weights still sum to one, and the
derivative with respect to a coordinate strictly outside is zero.  A
**non-finite** coordinate is not clamped: every weight of that point is
NaN, on the ``2**d`` corners of the grid's first cell (index 0 or 1 on
each axis), so gather returns NaN for that point only and scatter puts NaN
in those corner cells and no other.

**Precision.**  Indices and weights are computed in the geometry's dtype,
in a static power-of-two frame of the spacing (so a spacing far below one
is resolved), and the weights are cast to the field's dtype: the result
has the field's dtype.  A float32 geometry far from the origin, or on an
axis with very many cells, cannot resolve a cell;
:meth:`MultilinearGridMapping.geometry_dtype_problems` says so when the
graph is compiled.

**Determinism.**  Gather is a fixed left fold over the ``2**d`` slots.
Scatter is a scatter-add with repeated indices; on CPU it accumulates in
update order (points, then slots), so it is a deterministic function of
the inputs *including the order of the points*.  On an accelerator a cell
that receives several contributions may differ by rounding from run to
run.  The same holds for the reverse-mode derivative of a gather with
respect to its field.

The mapping has no weights (``params_pytree()`` is ``{}``) and keeps
nothing between calls: it is a pure function of the field and the
geometry.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional

import jax.numpy as jnp
import numpy as np

from maddening.core._pow2_frame import pow2_host_factor
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.coupling.mapping import _MODES
from maddening.core.coupling.mapping_registry import register_mapping
from maddening.core.coupling.mapping_spec import MappingSpec, reference_for_array

KIND = "multilinear_grid"
_LAYOUTS = ("flat", "shaped")
_OUTSIDE = ("clamp",)
#: The flat grid index is int32.
_MAX_GRID_POINTS = 2**31 - 1
#: Cells of position error at which a geometry dtype is refused / warned
#: about on an axis (``geometry_dtype_problems``).
_RESOLUTION_REFUSED = 1.0 / 16.0  # units: grid cells
_RESOLUTION_WARNED = 1.0 / 1024.0  # units: grid cells


@stability(StabilityLevel.EXPERIMENTAL)
class MultilinearGridMapping:
    """Multilinear transfer between a uniform grid and moving points.

    Built by :func:`multilinear_grid_mapping`; see the module docstring
    for what it computes.

    Attributes
    ----------
    origin, spacing : tuple of float
        The lattice: point ``i`` of axis ``a`` is at
        ``origin[a] + i * spacing[a]``.
    shape : tuple of int
        Lattice points per axis.
    n_points : int
        The number of moving points.
    mode : str
        ``"consistent"`` (gather, grid to points) or ``"conservative"``
        (scatter, points to grid).
    layout : str
        ``"flat"``: the grid field is ``(N,)`` or ``(N, C)``;
        ``"shaped"``: it is ``shape`` or ``shape + (C,)``.
    needs_geometry : bool
        ``True``: the edge carrying this mapping names a geometry.
    geometry_shape : tuple of int
        ``(n_points, d)``, the shape of the geometry it reads.
    """

    kind = KIND
    needs_geometry = True

    def __init__(self, origin, spacing, shape, n_points: int, mode: str, layout: str,
                 spec: Optional[MappingSpec]):
        self.origin = tuple(float(o) for o in origin)
        self.spacing = tuple(float(h) for h in spacing)
        self.shape = tuple(int(n) for n in shape)
        self.n_points = int(n_points)
        self.mode = mode
        self.layout = layout
        self.outside = "clamp"
        self.spec = spec
        self._d = len(self.shape)
        self._size = 1
        for n in self.shape:
            self._size *= n
        self.geometry_shape = (self.n_points, self._d)

    @property
    def n_source(self) -> int:
        return self._size if self.mode == "consistent" else self.n_points

    @property
    def n_target(self) -> int:
        return self.n_points if self.mode == "consistent" else self._size

    def params_pytree(self) -> dict:
        """``{}``: the kind has no weights."""
        return {}

    def field_shapes(self) -> tuple[tuple, tuple]:
        """``(source lead, target lead)``: the leading axes of the field
        read and of the field delivered (the grid's are ``shape`` under
        ``layout="shaped"``, ``(N,)`` otherwise)."""
        grid = self.shape if self.layout == "shaped" else (self._size,)
        points = (self.n_points,)
        return (grid, points) if self.mode == "consistent" else (points, grid)

    def accepts_geometry_shape(self, shape) -> bool:
        """Whether a geometry of *shape* can be read: ``(n_points, d)``,
        or ``(n_points,)`` on a one-dimensional grid."""
        shape = tuple(shape)
        return shape == self.geometry_shape or (self._d == 1 and shape == (self.n_points,))

    def geometry_dtype_problems(self, dtype) -> tuple[list[str], list[str]]:
        """``(errors, warnings)`` about reading positions held in *dtype*.

        A position ``x`` resolves ``eps * abs(x)``, so on axis ``a`` a
        point of the grid is located to ``eps * max(abs(origin[a]),
        abs(origin[a] + (n_a - 1) * spacing[a])) / spacing[a]`` cells.  At
        1/16 cell or worse that is an error (the weights mean nothing);
        at 1/1024 cell or worse, a warning.  An ``origin`` or ``spacing``
        that is not a normal number of *dtype* is an error: such a
        coordinate is read as zero or infinity (an origin of exactly zero
        is a normal case and is accepted).
        """
        info = np.finfo(np.dtype(dtype))
        eps, tiny, largest = float(info.eps), float(info.tiny), float(info.max)
        errors: list[str] = []
        warnings_: list[str] = []
        for a in range(self._d):
            origin, spacing, n = self.origin[a], self.spacing[a], self.shape[a]
            far = origin + (n - 1) * spacing
            if not tiny <= spacing <= largest or (origin and not tiny <= abs(origin) <= largest) \
                    or abs(far) > largest:
                errors.append(
                    f"axis {a} of the grid (origin {origin!r}, spacing {spacing!r}, {n} "
                    f"points) is outside the normal range of a {np.dtype(dtype)} geometry")
                continue
            cells = eps * max(abs(origin), abs(far)) / spacing
            text = (f"a {np.dtype(dtype)} geometry locates a point on axis {a} of the grid "
                    f"(origin {origin!r}, spacing {spacing!r}, {n} points) only to "
                    f"{cells:.3g} of a cell")
            if cells >= _RESOLUTION_REFUSED:
                errors.append(text + "; hold the geometry in float64, or move the origin "
                                     "of the coordinates closer to the grid")
            elif cells >= _RESOLUTION_WARNED:
                warnings_.append(text)
        return errors, warnings_

    def __repr__(self) -> str:
        return (f"MultilinearGridMapping({self.mode}, grid {self.shape}, "
                f"{self.n_points} points)")

    # -- the stencil ------------------------------------------------------

    def _stencil(self, geom):
        """``(I, W)``, both ``(n_points, 2**d)``: flat grid indices (int32)
        and weights (the geometry's dtype) of every point's corners, in
        lexicographic corner order with axis 0 most significant."""
        if geom is None:
            raise ValueError(
                f"a {KIND} mapping reads a moving geometry and was called without one; "
                f"put it on an edge with add_edge(..., geometry=(anchor, field))")
        geom = jnp.asarray(geom)
        if not self.accepts_geometry_shape(geom.shape):
            raise ValueError(
                f"{KIND}: the geometry has shape {tuple(geom.shape)}; expected "
                f"{self.geometry_shape}")
        if not jnp.issubdtype(geom.dtype, jnp.floating):
            raise TypeError(
                f"{KIND}: the geometry has dtype {geom.dtype}; positions are floating-point")
        # Asked here as well as by ``compile()``: a program is traced
        # again when the geometry's dtype changes, so a float32 geometry
        # written into a graph that was compiled with a float64 one (a
        # state write is not a recompile) is refused at its first step.
        unresolved, _ = self.geometry_dtype_problems(geom.dtype)
        if unresolved:
            raise ValueError(f"{KIND}: " + "; ".join(unresolved))
        if geom.ndim == 1:
            geom = geom[:, None]
        T = geom.dtype
        zero = jnp.zeros((), T)
        finite = jnp.all(jnp.isfinite(geom), axis=1)
        lower, upper, fraction = [], [], []
        for a in range(self._d):
            n = self.shape[a]
            # A static power-of-two frame of this axis' spacing: exact, and
            # the difference below stays a normal number at any spacing.
            p = pow2_host_factor(self.spacing[a], T)
            framed_origin = jnp.asarray(self.origin[a] * p, T)
            framed_spacing = jnp.asarray(self.spacing[a] * p, T)
            u = (geom[:, a] * jnp.asarray(p, T) - framed_origin) / framed_spacing
            u = jnp.where(finite, u, zero)
            top = jnp.asarray(n - 1, T)
            # Strict comparisons under ``where``: the derivative at a bound
            # is the interior one-sided one, and zero strictly outside.
            clamped = jnp.where(u < 0, zero, jnp.where(u > top, top, u))
            base = jnp.clip(jnp.floor(clamped), 0, max(n - 2, 0))
            fraction.append(clamped - base)
            i0 = base.astype(jnp.int32)
            lower.append(i0)
            upper.append(jnp.minimum(i0 + 1, n - 1))
        strides = [int(np.prod(self.shape[a + 1:], dtype=np.int64)) for a in range(self._d)]
        index, weight = [], []
        nan = jnp.asarray(jnp.nan, T)
        for corner in itertools.product((0, 1), repeat=self._d):
            flat = jnp.zeros((self.n_points,), jnp.int32)
            w = jnp.ones((self.n_points,), T)
            for a in range(self._d):
                flat = flat + strides[a] * (upper[a] if corner[a] else lower[a])
                w = w * (fraction[a] if corner[a] else (1 - fraction[a]))
            index.append(flat)
            weight.append(jnp.where(finite, w, nan))
        return jnp.stack(index, axis=1), jnp.stack(weight, axis=1)

    def _floating(self, field, what: str):
        field = jnp.asarray(field)
        if not jnp.issubdtype(field.dtype, jnp.floating):
            raise TypeError(
                f"a {KIND} mapping transfers floating-point fields; its {what} field has "
                f"dtype {field.dtype}")
        return field

    def _gather(self, field, geom):
        field = self._floating(field, "grid")
        lead = self.shape if self.layout == "shaped" else (self._size,)
        if tuple(field.shape[:len(lead)]) != lead or field.ndim > len(lead) + 1:
            raise ValueError(
                f"{KIND} (grid to points, layout={self.layout!r}): the grid field has "
                f"shape {tuple(field.shape)}; expected {lead} or {lead + ('C',)}")
        flat = field.reshape((self._size,) + tuple(field.shape[len(lead):]))
        index, weight = self._stencil(geom)
        weight = weight.astype(field.dtype)
        out = None
        for s in range(index.shape[1]):
            w = weight[:, s].reshape((-1,) + (1,) * (flat.ndim - 1))
            term = w * flat[index[:, s]]
            out = term if out is None else out + term
        return out

    def _scatter(self, field, geom):
        field = self._floating(field, "point")
        if tuple(field.shape[:1]) != (self.n_points,) or field.ndim > 2:
            raise ValueError(
                f"{KIND} (points to grid): the point field has shape "
                f"{tuple(field.shape)}; expected {(self.n_points,)} or "
                f"{(self.n_points, 'C')}")
        index, weight = self._stencil(geom)
        weight = weight.astype(field.dtype)
        w = weight.reshape(weight.shape + (1,) * (field.ndim - 1))
        updates = w * field[:, None]
        out = jnp.zeros((self._size,) + tuple(field.shape[1:]), field.dtype)
        out = out.at[index].add(updates)
        if self.layout == "shaped":
            out = out.reshape(self.shape + tuple(field.shape[1:]))
        return out

    def apply(self, field, weights: Optional[dict] = None, geom=None):
        """Transfer *field* at the positions *geom* (*weights* is unused:
        the kind has none)."""
        if self.mode == "consistent":
            return self._gather(field, geom)
        return self._scatter(field, geom)

    def apply_T(self, field, weights: Optional[dict] = None, geom=None):
        """The transpose of :meth:`apply` at the same positions: the other
        mode's transfer."""
        if self.mode == "consistent":
            return self._scatter(field, geom)
        return self._gather(field, geom)


@register_mapping(
    KIND,
    arrays=("origin", "spacing", "shape"),
    hyperparameters={"mode": str, "layout": str, "outside": str, "n_points": int},
    references={"origin": "origin_ref", "spacing": "spacing_ref", "shape": "shape_ref"},
    needs_geometry=True,
)
@stability(StabilityLevel.EXPERIMENTAL)
def multilinear_grid_mapping(origin: Any, spacing: Any, shape: Any, *, n_points: int,
                             mode: str = "consistent", layout: str = "flat",
                             outside: str = "clamp", origin_ref: Optional[dict] = None,
                             spacing_ref: Optional[dict] = None,
                             shape_ref: Optional[dict] = None) -> MultilinearGridMapping:
    """A multilinear gather or scatter between a uniform grid and moving points.

    Parameters
    ----------
    origin, spacing : sequence of float
        The lattice of sample points, one entry per axis (one to three
        axes): point ``i`` of axis ``a`` is at ``origin[a] + i *
        spacing[a]``.  ``origin`` finite, ``spacing`` finite and positive.
    shape : sequence of int
        Lattice points per axis, each at least one; at most ``2**31 - 1``
        points in all.
    n_points : int
        The number of moving points; the geometry the edge names has
        shape ``(n_points, len(shape))``.
    mode : {"consistent", "conservative"}
        Gather grid to points, or scatter points to grid.
    layout : {"flat", "shaped"}
        Whether the grid field is flat (C order) or has the grid's shape.
    outside : {"clamp"}
        What a point outside the lattice's hull is: its projection onto
        the hull.  Recorded so that a saved config keeps its meaning; no
        other value is supported.
    origin_ref, spacing_ref, shape_ref : dict, optional
        References for the three arrays (inlined when omitted).

    Returns
    -------
    MultilinearGridMapping

    Raises
    ------
    ValueError
        For a grid or an option outside the above.
    """
    origin_a = np.atleast_1d(np.asarray(origin, np.float64))
    spacing_a = np.atleast_1d(np.asarray(spacing, np.float64))
    shape_in = np.atleast_1d(np.asarray(shape))
    d = int(shape_in.size)
    if shape_in.ndim != 1 or d not in (1, 2, 3) or origin_a.shape != (d,) \
            or spacing_a.shape != (d,):
        raise ValueError(
            f"{KIND}: origin, spacing and shape must have the same number of entries, one "
            f"per axis of a grid of one to three axes; got {np.shape(origin)}, "
            f"{np.shape(spacing)} and {np.shape(shape)}")
    if shape_in.dtype.kind not in "iuf" or not np.all(np.isfinite(shape_in)) \
            or not np.all(shape_in == np.round(shape_in)):
        raise ValueError(f"{KIND}: shape must hold integers, got {shape!r}")
    shape_a = shape_in.astype(np.int64)
    if not np.all(np.isfinite(origin_a)):
        raise ValueError(f"{KIND}: origin must be finite, got {origin!r}")
    if not (np.all(np.isfinite(spacing_a)) and np.all(spacing_a > 0)):
        raise ValueError(f"{KIND}: spacing must be finite and positive, got {spacing!r}")
    if np.any(shape_a < 1):
        raise ValueError(f"{KIND}: every axis needs at least one point, got shape {shape!r}")
    total = 1
    for n in shape_a:
        total *= int(n)
    if total > _MAX_GRID_POINTS:
        raise ValueError(
            f"{KIND}: a grid of shape {tuple(int(n) for n in shape_a)} has {total} points; "
            f"at most {_MAX_GRID_POINTS} are supported (the flat index is int32)")
    if mode not in _MODES:
        raise ValueError(f"{KIND}: mode must be one of {_MODES}, got {mode!r}")
    if layout not in _LAYOUTS:
        raise ValueError(f"{KIND}: layout must be one of {_LAYOUTS}, got {layout!r}")
    if outside not in _OUTSIDE:
        raise ValueError(
            f"{KIND}: outside={outside!r} is not supported; points outside the grid are "
            f"clamped to its hull (outside='clamp')")
    if isinstance(n_points, bool) or int(n_points) != n_points or int(n_points) < 1:
        raise ValueError(f"{KIND}: n_points must be a positive integer, got {n_points!r}")
    spec = MappingSpec(KIND, {"mode": mode, "layout": layout, "outside": outside,
                              "n_points": int(n_points)}, {
        "origin": reference_for_array(origin_a, origin_ref, name="origin"),
        "spacing": reference_for_array(spacing_a, spacing_ref, name="spacing"),
        "shape": reference_for_array(shape_a, shape_ref, name="shape"),
    })
    return MultilinearGridMapping(origin_a, spacing_a, shape_a, int(n_points), mode, layout,
                                  spec)
