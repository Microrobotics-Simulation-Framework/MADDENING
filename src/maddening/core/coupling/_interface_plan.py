"""One description of a coupling group's edges: what each one delivers,
and which fields a norm, a weight, a floor, an accelerator or a report
reads.  Private.

A coupling group's *interface* is its internal edges: those whose source
and target are both members.  Until this module each consumer worked out
for itself which edges those are, in what order, what each delivers and
which state fields count as "read", and the answers differed.  Here the
group's edges are described once (:func:`interface_plan`, one
:class:`InterfaceEdge` per edge) and every consumer reads a *view* of
that description: a method of :class:`InterfacePlan` or one of the small
functions below.  Each view's docstring names its consumers.

**Where two consumers read the interface differently, each keeps its own
named view, and the docstring states the difference.**  Nothing is
reconciled here; the differences are pinned by
``tests/core/test_interface_plan.py``:

* the accelerator (:meth:`InterfacePlan.iqn_fields`) counts a
  source-anchored geometry field and stands a flux edge's producer's
  whole state in for the flux; the spectrum's weights
  (:meth:`InterfacePlan.source_fields`) count neither;
* a field that two internal edges read is two readings to the norm
  (:func:`interface_records`) and one field to the accelerator and to
  the spectrum's weights;
* whether a source field is floating is decided on the state a plan is
  built from (:attr:`InterfaceEdge.source_kind`), except by the norm's
  reading, which decides it on the first state it is handed
  (``acceleration._interface_readings``).

**The side an edge is read on** (:attr:`InterfaceEdge.norm_side`) is set
in one place, :func:`_norm_side`: the interface norm reads a mapped edge
on its *compact* side.  An edge whose static mapping delivers more
entries than the field it reads holds (a scatter: 30 marker forces onto
a grid) is read at its ``"source"``, the field itself, before the
mapping and so before the transform; every other edge -- a mapping onto
fewer entries, **a tie**, no mapping -- is read as ``"delivered"``, what
the edge hands its target.  What follows from the side branches on that
field and on nothing else: the value read
(:meth:`InterfaceEdge.reading`), whether that value is the source field
itself (:attr:`InterfaceEdge.reads_source_as_is`) and whether it depends
on the mapping's weights (:attr:`InterfaceEdge.reads_through_mapping`).
The side is a function of the edge's mapping alone (the sizes it
declares, which ``add_edge`` holds its two ends to), so every plan of a
group, and a bare edge described on its own, decide it alike.

**A reading is one or more parts** (:attr:`InterfaceEdge.parts`, one
:class:`ReadingPart` each), and every consumer iterates them
(:meth:`InterfaceEdge.read`, through ``acceleration._interface_readings``):
the residual, its float floor, the pooled count, the spectral analysis's
reading and the fields a solve returns as it accepted them
(:attr:`InterfaceEdge.measured_whole`).  Every edge without a geometry
has one part, measured against its own magnitude.  A geometry-dependent
mapping (experimental) is read by the same side rule:

* on its ``"delivered"`` side (a gather, a tie): what the step delivers,
  the source field through the mapping **at the geometry the step uses**
  (:meth:`InterfaceEdge.geometry_at`), then the transform.  One part;
* on its ``"source"`` side (a scatter: a few points onto a large grid):
  **the mapping's inputs**.  The source field as stored, and -- where the
  geometry moves with the iterate (a source anchor) -- the positions, a
  part of their own measured **in units of the mapping kind's own length
  scale** (``geometry_length_scale()``; the grid spacing per axis for
  ``multilinear_grid``) and not against their own magnitude, which
  depends on where the origin of the coordinates is.  A target-anchored
  geometry is the target's pre-step state, the same at every pass: a
  constant of the solve, with no residual, and no part.

The exact model and the numerical reference of the test suite
(``tests/property/coupled_topologies.py``, ``coupling_reference.py``) are
oracles and do not import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core._pow2_frame import pow2_host_factor
from maddening.core.edge import _delivered

#: What an edge's source is, in the state its plan was built from
#: (:attr:`InterfaceEdge.source_kind`): a floating field of the producer's
#: state; a field of it that is not floating (a counter, a flag, a key);
#: a boundary flux (not a state field, and the producer defines
#: ``compute_boundary_fluxes``); or absent (not a state field, and no flux
#: hook -- ``validate()`` refuses the edge -- or a plan built without the
#: nodes).
FLOATING = "floating"
NON_FLOATING = "non_floating"
FLUX = "flux"
ABSENT = "absent"
#: The two kinds that are fields of the producer's state.
_STATE_FIELD_KINDS = (FLOATING, NON_FLOATING)

#: The form of the mapping an edge carries (:attr:`InterfaceEdge.mapping_form`),
#: in the mapping registry's terms: none; a ``StaticLinearMapping`` (a dense
#: matrix); a ``StaticSparseMapping``; another static mapping (a registered
#: kind's own class, ``needs_geometry`` false); or a geometry-dependent one
#: (``needs_geometry`` true).
NO_MAPPING = "none"
STATIC_DENSE = "static_dense"
STATIC_SPARSE = "static_sparse"
STATIC_OTHER = "static"
NEEDS_GEOMETRY = "needs_geometry"

#: Where an edge sits relative to the group (:attr:`InterfaceEdge.role`).
INTERNAL = "internal"
INBOUND = "inbound"
OUTBOUND = "outbound"

#: The side of an edge the interface norm reads (:attr:`InterfaceEdge.norm_side`):
#: what the edge hands its target (the source field through the mapping,
#: then the transform), or the source field itself, before both.
DELIVERED = "delivered"
SOURCE = "source"
#: A third thing a part of a reading can be (:attr:`ReadingPart.what`,
#: beside the two sides): the geometry a geometry-dependent mapping read
#: at its source takes from the iterate.
GEOMETRY = "geometry"

#: What a part of a reading is measured against (:attr:`ReadingPart.unit`):
#: its own largest magnitude, as every field of every norm is; or one unit
#: of the mapping kind's own length scale, the part's value being already
#: expressed in it.
OWN_MAGNITUDE = "own_magnitude"
KERNEL_LENGTH = "kernel_length"


def is_internal(edge, members) -> bool:
    """Is *edge* internal to a group of *members*: both its ends are members.

    The one definition of a coupling group's interface edges.  Read by
    :func:`internal_edges`, :func:`interface_plan`,
    :meth:`InterfacePlan.is_internal` and
    ``_group_layout._loop_through_outside_nodes``.
    """
    return edge.source_node in members and edge.target_node in members


def internal_edges(edges, members) -> list:
    """The edges of *edges* internal to *members*, in the order given.

    For a reader that has the graph's edges and no state: the node
    classes' group advisories (``GraphManager._coupling_group_advisories``)
    and the inspection table (``core/inspection.py``).
    """
    return [e for e in edges if is_internal(e, members)]


def member_reads(edges, members) -> dict:
    """``{member: members it reads through an internal edge}``, itself included.

    Read by ``_group_layout._group_pass_structure`` (which members a
    member reads from the same pass: the evaluation count of the float
    floor) and by :meth:`InterfacePlan.member_reads`.
    """
    reads: dict[str, set] = {nn: set() for nn in members}
    for edge in internal_edges(edges, members):
        reads[edge.target_node].add(edge.source_node)
    return reads


def _interface_edge_order(edges, member_order) -> list:
    """A coupling group's internal *edges* in the order its interface norm sums them.

    By the source's place in *member_order* (the group's sweep), then the
    source field, the target's place, the target field and the ordinal:
    the order the L2 and mixed norms sum the members in, so the interface
    norm, like them, depends neither on the order of the ``add_edge``
    calls nor on the nodes' names.  Edges with the same endpoints (an
    additive pair with different transforms) keep their relative order.

    The canonical order of :attr:`InterfacePlan.internal`.
    """
    place = {nn: i for i, nn in enumerate(member_order)}
    last = len(place)
    return sorted(edges, key=lambda e: (place.get(e.source_node, last), e.source_field,
                                        place.get(e.target_node, last), e.target_field,
                                        e.ordinal))


def _defines_fluxes(node) -> bool:
    """Does *node*'s class define ``compute_boundary_fluxes``?"""
    from maddening.core.node import SimulationNode  # noqa: PLC0415

    return type(node).compute_boundary_fluxes is not SimulationNode.compute_boundary_fluxes


def _source_kind(edge, state, nodes) -> str:
    """What *edge*'s source is in *state* (see :data:`FLOATING`)."""
    fields = None if state is None else state.get(edge.source_node)
    if fields is not None and edge.source_field in fields:
        value = fields[edge.source_field]
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            try:
                dtype = jnp.asarray(value).dtype
            except TypeError:
                # Not an array at all (a nested container): no norm reads it.
                return NON_FLOATING
        return FLOATING if jnp.issubdtype(dtype, jnp.floating) else NON_FLOATING
    spec = None if nodes is None else nodes.get(edge.source_node)
    if spec is not None and _defines_fluxes(spec.node):
        return FLUX
    return ABSENT


def _mapping_form(mapping) -> str:
    """The form of *mapping* (see :data:`NO_MAPPING`)."""
    if mapping is None:
        return NO_MAPPING
    if getattr(mapping, "needs_geometry", False):
        return NEEDS_GEOMETRY
    from maddening.core.coupling.mapping import StaticLinearMapping  # noqa: PLC0415
    from maddening.core.coupling.sparse_mapping import StaticSparseMapping  # noqa: PLC0415

    if isinstance(mapping, StaticSparseMapping):
        return STATIC_SPARSE
    if isinstance(mapping, StaticLinearMapping):
        return STATIC_DENSE
    return STATIC_OTHER


def _mapping_leads(mapping) -> tuple:
    """``(source lead, target lead)`` of *mapping*: the leading axes of the
    field it reads and of the field it delivers.

    What the mapping declares, the sizes ``add_edge`` holds the edge's two
    ends to: ``field_shapes()`` where it has one, else ``(n_source,)`` and
    ``(n_target,)`` (a mapping acts on axis 0; any further axes pass
    through, so they do not enter a comparison of the two sides).
    """
    shapes = getattr(mapping, "field_shapes", None)
    if shapes is not None:
        source_lead, target_lead = shapes()
        return (tuple(int(n) for n in source_lead), tuple(int(n) for n in target_lead))
    try:
        return ((int(mapping.n_source),), (int(mapping.n_target),))
    except AttributeError:
        raise TypeError(
            f"mapping {mapping!r} declares neither n_source and n_target nor "
            f"field_shapes(): the interface norm cannot tell which side of it is "
            f"the compact one") from None


def _entries(lead) -> int:
    """How many entries (per trailing component) a field of leading axes *lead* holds."""
    count = 1
    for n in lead:
        count *= n
    return count


def _norm_side(edge) -> str:
    """The side of *edge* the interface norm reads.

    **The one place this is decided.**  The compact side:

    * ``"source"`` for an edge whose static mapping delivers **more**
      entries than the field it reads holds: the norm reads the source
      field itself, before the mapping, and so before the transform
      (the step applies the mapping, then the transform).  Read as
      delivered, such an edge put every entry of its large target into
      the norm's one RMS, of which a few change: a converged group's
      small field was 23 to 459 tolerances from its fixed point at 1e3
      to 1e6 target entries
      (``benchmarks/results/interface_norm_dilution``);
    * ``"delivered"`` for every other edge: a mapping onto fewer entries,
      **a tie**, and an edge with no mapping.

    A function of the edge's mapping alone -- the sizes it declares
    (:func:`_mapping_leads`), never the shape of its weights (a sparse
    layout's ``(rows, k)`` says nothing of the two sides) and never a
    state -- so the plans built by ``validate()``, by ``compile()`` and
    at trace, and a bare edge described for the report, cannot differ.

    A geometry-dependent mapping is decided by the same sizes (a
    ``multilinear_grid`` scatter of a few points onto a larger grid is
    read at its ``"source"``; its gather, and a tie, as ``"delivered"``),
    provided its kind declares the length scale a reading at its source
    needs (``geometry_length_scale``: :func:`_kernel_lengths`).  A kind
    that declares none has no reading there and stays ``"delivered"``;
    ``compile()`` refuses it inside a group under the interface norm
    (``_geometry_edge_coupling_errors``), so nothing reads it.
    """
    mapping = edge.mapping
    if mapping is None:
        return DELIVERED
    if getattr(mapping, "needs_geometry", False) and not callable(
            getattr(mapping, "geometry_length_scale", None)):
        return DELIVERED
    source_lead, target_lead = _mapping_leads(mapping)
    return SOURCE if _entries(target_lead) > _entries(source_lead) else DELIVERED


def _kernel_lengths(mapping) -> tuple:
    """The length scale of *mapping*'s kind on each coordinate axis of its geometry.

    What the kind declares (``geometry_length_scale()``): for
    ``multilinear_grid`` the grid spacing per axis.  The one unit a
    position is measured in by the interface norm; a user-supplied length
    is not offered.
    """
    declared: Any = getattr(mapping, "geometry_length_scale", None)
    if not callable(declared):
        raise TypeError(
            f"mapping {mapping!r} declares no geometry_length_scale(): the interface "
            f"norm cannot measure its geometry (a position is measured in units of the "
            f"mapping kind's own length scale)")
    lengths: Any = declared()
    return tuple(float(h) for h in lengths)


def _in_kernel_lengths(mapping, geom):
    """The geometry *geom* of *mapping*, each coordinate over the kind's
    length scale on its axis (:func:`_kernel_lengths`).

    The same shape and dtype as *geom* (``(points, axes)``, or
    ``(points,)`` on one axis).  Each axis is divided in a static
    power-of-two frame of its length, as the kernel divides it, so the
    quotient is exact in the frame at any spacing.  No origin is
    subtracted: this is what the **residual** reads of positions, and a
    difference of two readings does not see an origin.  It is not, by
    itself, where a position is rounded: the kernel subtracts its own
    origin in the positions' dtype (:func:`_rounded_at`, which the
    float floor reads).
    """
    lengths = _kernel_lengths(mapping)
    geom = jnp.asarray(geom)
    cols = geom if geom.ndim >= 2 else geom[:, None]
    if cols.shape[-1] != len(lengths):
        raise ValueError(
            f"mapping {mapping!r} declares {len(lengths)} length scale(s) for a geometry "
            f"of shape {tuple(geom.shape)}")
    dtype = geom.dtype
    scaled = []
    for axis, length in enumerate(lengths):
        frame = pow2_host_factor(length, dtype)
        scaled.append((cols[..., axis] * jnp.asarray(frame, dtype))
                      / jnp.asarray(length * frame, dtype))
    return jnp.stack(scaled, axis=-1).reshape(geom.shape)


def _rounded_at(mappings, geom):
    """The magnitude each coordinate of the geometry *geom* is **rounded
    at**, in the length scale of the mapping kind that reads it: what the
    float floor multiplies ``eps`` of the positions' dtype by
    (``acceleration._positions_resolution``).  Non-negative, the shape
    and dtype of *geom*.

    **The one place the rounding of a position is located.**  Per
    coordinate, the larger of two magnitudes, for each of *mappings*
    (the mappings of the group's edges that read *geom*: one, or several
    on different lattices; the largest over them):

    * its **stored** magnitude, :func:`_in_kernel_lengths`: a position
      ``u`` lengths from the coordinates' zero is stored to ``eps * |u|``
      lengths;
    * the magnitude of the **coordinate the kernel forms from it** in
      the positions' dtype, where the kind declares one
      (``geometry_kernel_coordinates``; for ``multilinear_grid`` the
      lattice coordinate ``(x - origin) / spacing``) and reads that
      coordinate (``geometry_coordinates_read``): the kernel's weights
      are resolved to ``eps`` times *that*.

    The second is what the first alone missed.  Counted from the
    coordinates' zero only, four float32 markers within 4 spacings of
    zero on a grid of 16001 points centred there (first point 8000
    spacings away) read a floor of 0.26 to 0.50 tolerances, no advisory
    and ``converged=True`` with the state 2 to 20 tolerances from the
    float64 fixed point (pooled; 42 in the worst entry), and 34 on a
    grid of 1e5 points (Gauss-Seidel at ``rtol=1e-5``, Jacobi at 3e-6;
    jaxlib 0.11.0, CPU): the kernel had rounded each weight at ``eps *
    8000`` and ``eps * 5e4`` of a cell.  And the same physical problem
    read a floor of 623 with the coordinates' zero at the grid's first
    point and 0.26 with it at the markers.  The larger of the two does
    not depend on where the zero is put once the lattice coordinate is
    the larger, which it is wherever the markers are further from the
    grid's first point than from the zero.

    A coordinate the kind does not read (an axis of one lattice point, a
    point clamped to the hull from further out than its rounding can
    cross) keeps its stored magnitude and takes nothing from the kernel:
    the kernel forms no weight from it.  A kind that declares no kernel
    coordinate has the stored magnitude alone.  Only ever at least the
    stored magnitude: a floor that reads this is never below the one
    that read :func:`_in_kernel_lengths`, and equals it, to the bit, for
    a grid whose origin is the coordinates' zero.
    """
    return jnp.maximum(_stored_magnitude(mappings, geom), _kernel_magnitude(mappings, geom))


def _stored_magnitude(mappings, geom):
    """``|u|``: how far each coordinate of *geom* is from the
    coordinates' zero, in the length scale of the kind that reads it
    (:func:`_in_kernel_lengths`), the largest over *mappings*.  One of
    the two magnitudes of :func:`_rounded_at`."""
    out = None
    for mapping in mappings:
        stored = jnp.abs(_in_kernel_lengths(mapping, geom))
        out = stored if out is None else jnp.maximum(out, stored)
    if out is None:
        raise ValueError("no mapping reads this geometry")
    return out


def _kernel_magnitude(mappings, geom):
    """The magnitude of the coordinate the kernel forms from each
    coordinate of *geom* in the positions' dtype (the kind's
    ``geometry_kernel_coordinates``; for ``multilinear_grid`` the
    lattice coordinate, the distance from the grid's first point in
    spacings), the largest over *mappings*; zero for a coordinate no
    mapping's kernel reads (``geometry_coordinates_read``) and for a
    kind that declares no such coordinate.  The other magnitude of
    :func:`_rounded_at`."""
    geom = jnp.asarray(geom)
    out = jnp.zeros(geom.shape, geom.dtype)
    for mapping in mappings:
        formed: Any = getattr(mapping, "geometry_kernel_coordinates", None)
        if not callable(formed):
            continue
        kernel = jnp.abs(jnp.asarray(formed(geom))).astype(geom.dtype)
        declared: Any = getattr(mapping, "geometry_coordinates_read", None)
        if callable(declared):
            read: Any = declared(geom)
            kernel = jnp.where(read, kernel, jnp.zeros((), geom.dtype))
        # units: the kind's lengths (dimensionless multiples of a spacing)
        out = jnp.maximum(out, kernel)
    return out


def _read_in_kernel_lengths(mapping, geom, lattices=None):
    """The magnitude at which the positions a value **delivered** through
    *mapping* at the geometry *geom* depends on are rounded, in the
    kind's length scale (:func:`_rounded_at`), with zero in place of
    every coordinate the kind says its transfer does not read.

    What the float floor of such a value takes the positions' rounding
    from (``acceleration._part_resolution``): a coordinate the kernel
    ignores -- for ``multilinear_grid`` one on an axis of one lattice
    point, or one clamped to the hull from further out than its rounding
    can cross (``geometry_coordinates_read``) -- moves nothing the edge
    delivers, whatever its last digits, so its rounding is none of the
    value's.  A kind that does not declare which coordinates it reads
    has all of them counted.  Positions that are themselves a **part**
    of a reading (a mapping read at its source and anchored there) are
    not passed through this: the residual reads every coordinate of
    them, and so does its floor.

    *lattices* are the mappings of every edge that reads the same
    geometry field (:func:`_position_lattices`; *mapping* alone where
    none is given): a coordinate this mapping reads is rounded at the
    largest magnitude over them.
    """
    reach = _rounded_at(lattices or (mapping,), geom)
    declared: Any = getattr(mapping, "geometry_coordinates_read", None)
    if not callable(declared):
        return reach
    read: Any = declared(geom)
    return jnp.where(read, reach, jnp.zeros((), reach.dtype))


def _position_lattices(records) -> dict:
    """``{(node, field): (mapping, ...)}``: for each geometry field the
    edges *records* are anchored at, the mappings that read it, each
    once, in the order of *records*.

    Only kinds that declare a length scale (``geometry_length_scale``:
    the ones the interface norm reads a geometry of).  Two edges on one
    field are usually a gather and a scatter on the same grid, which is
    one lattice (the kind's ``geometry_lattice()`` says so; mappings of
    a kind that declares none are each taken for their own); where they
    are on different lattices, a position's rounding is counted at the
    largest magnitude over them (:func:`_rounded_at`).  Read by
    ``acceleration._interface_readings`` (the floor) and through it by
    ``acceleration._positions_floors`` (``compile()``'s advisory), from
    the same records, so the two cannot count different lattices.
    """
    lattices: dict = {}
    for record in records:
        if record.anchor is None or not callable(
                getattr(record.mapping, "geometry_length_scale", None)):
            continue
        side, field = record.anchor
        holder = ((record.source if side == "source" else record.target)[0], field)
        declared: Any = getattr(record.mapping, "geometry_lattice", None)
        lattice = ((getattr(record.mapping, "kind", None), declared()) if callable(declared)
                   else id(record.mapping))
        lattices.setdefault(holder, {}).setdefault(lattice, record.mapping)
    return {holder: tuple(held.values()) for holder, held in lattices.items()}


@dataclass(frozen=True)
class ReadingPart:
    """One part of what the interface norm reads on an edge.

    Attributes
    ----------
    what : str
        ``"delivered"`` (what the edge hands its target), ``"source"``
        (the source field as stored) or ``"geometry"`` (the positions a
        mapping read at its source takes from the iterate).
    field : tuple of str
        ``(node, field)``: the state field of the iterate the part is
        read from.
    whole : bool
        Is the part that field, entry for entry (as stored, or each
        entry over a constant)?  Then the norm measures the field whole
        and a solve returns it as the iterate it accepted holds it
        (:attr:`InterfaceEdge.measured_whole`).
    unit : str
        ``"own_magnitude"``: the part's change is measured against its
        own largest magnitude, and a part within the dead band
        (``atol``) leaves the norm.  ``"kernel_length"``: the part is
        already expressed in the mapping kind's length scale and its
        change is measured against one unit of it; the dead band, a
        statement about a quantity's magnitude, does not apply to a
        position.
    """

    what: str
    field: tuple
    whole: bool
    unit: str


class PartReading(tuple):
    """One part of an edge's reading at one or more states.

    The tuple ``(edge, source_dtype, value, ...)``: the edge, the dtype
    of the stored field the part is read from at the first state, and
    the part's value at each state, in the unit it is measured in.  It
    also carries

    * ``part``: the :class:`ReadingPart` (what the value is measured
      against);
    * ``positions``: per state, the magnitude (in the kind's length
      scale) at which the stored positions the part rests on are
      rounded (:func:`_rounded_at`: the larger of a position's own
      magnitude and of the coordinate the kernel forms from it).  For a
      part that **is** positions, those positions'; for a value
      delivered through a geometry-dependent mapping, that of the
      positions it was computed at -- its rounding carries theirs --
      with zero for a coordinate the kind does not read
      (:func:`_read_in_kernel_lengths`); ``None`` for every other part.
      Read by the float floor only (``acceleration._part_resolution``),
      never by the residual;
    * ``delivered``: per state, what the edge delivers for a source field
      **read at its source**, where the reader asked for it
      (:meth:`InterfaceEdge.read` with ``band``: a group that declares a
      dead band); ``None`` otherwise.  Only its magnitude is read, and
      only by the dead band (``acceleration._kept_by_what_is_delivered``).
    """

    part: ReadingPart
    positions: tuple
    delivered: Optional[tuple]

    def __new__(cls, edge, source_dtype, values, part, positions, delivered=None):
        self = super().__new__(cls, (edge, source_dtype, *values))
        self.part = part
        self.positions = tuple(positions)
        self.delivered = None if delivered is None else tuple(delivered)
        return self

    @property
    def values(self) -> tuple:
        """The part's value at each state."""
        return tuple(self[2:])


@dataclass(frozen=True, eq=False)
class InterfaceEdge:
    """One edge of a coupling group, as every consumer reads it.

    Attributes
    ----------
    edge : EdgeSpec
        The edge itself: what the step's edge rule
        (``maddening.core.edge._delivered``) applies.
    key : str
        ``edge.key``, the edge's stable identifier and its slot in
        ``params["mappings"]``.
    declared : int
        The edge's place in the list the plan was built from (the order of
        the ``add_edge`` calls).
    role : str
        ``"internal"``, ``"inbound"`` (from outside the group into a
        member) or ``"outbound"`` (from a member to outside).
    source : tuple of str
        ``(source node, source field)``.
    source_kind : str
        ``"floating"``, ``"non_floating"``, ``"flux"`` or ``"absent"``, in
        the state the plan was built from.
    target : tuple of str
        ``(target node, target field)``.
    mapping
        The interface mapping the edge carries, or ``None``.
    mapping_kind : str or None
        The mapping's registered kind (``mapping.kind``).
    mapping_form : str
        ``"none"``, ``"static_dense"``, ``"static_sparse"``, ``"static"``
        or ``"needs_geometry"``.
    anchor : tuple of str or None
        ``(side, field)`` of the geometry a geometry-dependent mapping
        reads: a state field of the edge's source (``side == "source"``)
        or of its target (``"target"``).
    has_transform : bool
        Whether the edge applies a transform after the mapping.
    norm_side : str
        The side the interface norm reads the edge on (:func:`_norm_side`):
        ``"source"`` where a mapping delivers more entries than the
        field it reads holds, ``"delivered"`` otherwise.
    """

    edge: Any
    key: Optional[str]
    declared: int
    role: str
    source: tuple
    source_kind: str
    target: tuple
    mapping: Any
    mapping_kind: Optional[str]
    mapping_form: str
    anchor: Optional[tuple]
    has_transform: bool
    norm_side: str

    def reading(self, value, mappings=None, geom=None):
        """What the interface norm reads of *value*, this edge's source field.

        On the delivered side: *value* through the mapping, with the
        weights in *mappings* (``params["mappings"]``, ``None`` for the
        mapping's own) and, for a geometry-dependent mapping, at the
        geometry *geom* (:meth:`geometry_at`), then the transform -- the
        step's edge rule.  On the source side: *value* itself, through
        neither.  The first part of the edge's reading (:attr:`parts`);
        :meth:`read` evaluates all of them.

        The source side was decided on the sizes the mapping declares; a
        *value* that does not have them is refused rather than read on a
        side its own size would not have chosen.
        """
        if self.norm_side == DELIVERED:
            return _delivered(self.edge, value, mappings, geom)
        if self.norm_side == SOURCE:
            source_lead, _target_lead = _mapping_leads(self.mapping)
            shape = tuple(int(n) for n in jnp.shape(value))
            have = shape[:len(source_lead)]
            if have != source_lead and not (source_lead == (1,) and shape == ()):
                raise ValueError(
                    f"edge {self.key!r}: the interface norm reads this edge at its "
                    f"source because its mapping declares a field of leading axes "
                    f"{source_lead} delivered onto more entries, but the field read "
                    f"has shape {shape}")
            return value
        raise ValueError(
            f"edge {self.key!r}: the interface norm has no reading on side "
            f"{self.norm_side!r}")

    @property
    def parts(self) -> tuple:
        """The parts of the norm's reading of this edge (:class:`ReadingPart`).

        **The one statement of what is read on an edge, and against
        what.**  Delivered side: one part, what the edge delivers
        (whole only where the edge has no mapping and no transform).
        Source side: the source field as stored; and, for a
        geometry-dependent mapping whose geometry is a field of the
        source (a source anchor), that field in the kind's length scale.
        A target-anchored geometry is the target's pre-step state, a
        constant of the solve: it is no part.  Static.
        """
        if self.norm_side == DELIVERED:
            whole = self.mapping is None and not self.has_transform
            return (ReadingPart(DELIVERED, self.source, whole, OWN_MAGNITUDE),)
        if self.norm_side == SOURCE:
            parts = [ReadingPart(SOURCE, self.source, True, OWN_MAGNITUDE)]
            if self.anchor is not None and self.anchor[0] == "source":
                parts.append(ReadingPart(
                    GEOMETRY, (self.source[0], self.anchor[1]), True, KERNEL_LENGTH))
            return tuple(parts)
        raise ValueError(
            f"edge {self.key!r}: the interface norm has no reading on side "
            f"{self.norm_side!r}")

    @property
    def measured_whole(self) -> tuple:
        """``((node, field), ...)``: the state fields this edge's reading
        holds whole (:attr:`ReadingPart.whole`).

        The source field of an edge with no mapping and no transform or
        of one read at its source, and the geometry field a mapping read
        at its source takes from the iterate.  Read by
        ``_group_layout._fields_the_interface_norm_misses`` (the return
        rule keeps exactly these as the accepted iterate holds them).
        """
        return tuple(part.field for part in self.parts if part.whole)

    @property
    def reads_source_as_is(self) -> bool:
        """Is the norm's reading of this edge its source field, unchanged,
        and nothing else?

        On the source side: where the source field is the only part.
        A geometry-dependent mapping read at its source with a
        source-anchored geometry reads the positions too, in another
        unit, so its reading is not "the field as it is".  On the
        delivered side: an edge with no mapping and no transform.  Read
        by ``_group_layout._reading_is_the_fields`` (which spectral
        analysis the report takes).
        """
        if self.norm_side not in (DELIVERED, SOURCE):
            return False
        parts = self.parts
        return (len(parts) == 1 and parts[0].whole and parts[0].field == self.source
                and parts[0].unit == OWN_MAGNITUDE)

    @property
    def reads_through_mapping(self) -> bool:
        """Does the norm's reading of this edge go through its mapping, and
        so depend on the weights the step ran with?

        On the delivered side, where the edge carries a mapping.  Never on
        the source side: the value is read before the mapping.  Read by
        :meth:`InterfacePlan.norm_reads_mapping_weights` and
        :meth:`InterfacePlan.mapped_keys`.
        """
        return self.norm_side == DELIVERED and self.mapping is not None

    @property
    def reads_pre_step_geometry(self) -> bool:
        """Is this edge read as delivered through a mapping whose geometry
        is the target's pre-step state (a target anchor)?

        The state a solve returns does not hold that geometry, so
        nothing computed from the returned state alone can repeat the
        reading.  Read by
        :meth:`InterfacePlan.norm_reads_beyond_the_state`.
        """
        return (self.reads_through_mapping and self.anchor is not None
                and self.anchor[0] == "target")

    @property
    def kernel_rounds_at_stored_positions(self) -> bool:
        """Does this edge deliver through a geometry-dependent mapping of
        a kind that declares a length scale (``geometry_length_scale``;
        ``multilinear_grid``)?

        Such a kernel forms its weights from stored positions in their
        dtype, which the float floor of a group under ``"l2"`` and
        ``"mixed"`` counts for every value field
        (``acceleration._kernel_rounding``), whatever the edge's source
        is and whichever side the interface norm would read it on.
        """
        return self.anchor is not None and callable(
            getattr(self.mapping, "geometry_length_scale", None))

    @property
    def kernel_rounds_at_pre_step_positions(self) -> bool:
        """:attr:`kernel_rounds_at_stored_positions` for a mapping
        anchored at its **target**: the positions the kernel reads are
        the target's pre-step state, which the state a solve returns
        does not hold.  Read by
        :meth:`InterfacePlan.kernel_rounds_beyond_the_state`.
        """
        return (self.kernel_rounds_at_stored_positions and self.anchor is not None
                and self.anchor[0] == "target")

    @property
    def reads_weights_of_the_step(self) -> bool:
        """Is this edge read through a mapping that holds weights (which a
        caller may override for one step)?

        Every static mapping read as delivered.  A geometry-dependent
        kind without weights (``multilinear_grid``: its entry of
        ``params["mappings"]`` is ``{}``) reads none.
        """
        if not self.reads_through_mapping:
            return False
        if self.mapping_form != NEEDS_GEOMETRY:
            return True
        held = getattr(self.mapping, "params_pytree", None)
        return True if held is None else bool(held())

    @property
    def band_reads_weights_of_the_step(self) -> bool:
        """With a dead band declared, does whether this edge's reading is
        kept depend on mapping weights (which a caller may override for
        one step)?

        An edge read at its source through a mapping that holds weights:
        the band is asked of what it delivers as well as of the field
        (``acceleration._kept_by_what_is_delivered``).  Read by
        :meth:`InterfacePlan.band_reads_beyond_the_state`.
        """
        if self.norm_side != SOURCE:
            return False
        if self.mapping_form != NEEDS_GEOMETRY:
            return True
        held = getattr(self.mapping, "params_pytree", None)
        return True if held is None else bool(held())

    @property
    def band_reads_pre_step_geometry(self) -> bool:
        """With a dead band declared, does whether this edge's reading is
        kept depend on the target's pre-step geometry?

        An edge read at its source through a geometry-dependent mapping
        anchored at its target: what it delivers is taken at a geometry
        the returned state does not hold.
        """
        return (self.norm_side == SOURCE and self.anchor is not None
                and self.anchor[0] == "target")

    def geometry_at(self, state, pre_step=None):
        """The geometry this edge's mapping is read with at the iterate *state*.

        **The time level of a reading's geometry, in the one place it is
        written** (contract B of the geometry guide): a source-anchored
        geometry is the geometry field of the same state the value is
        read from, so it moves with the iterate; a target-anchored one is
        the field of the target's **pre-step** state, the same at every
        pass, which is what the target's ``update`` is resolved with.
        *pre_step* gives a member's pre-step state (a callable of the
        member's name, or a ``{name: state}`` dict); a reading that needs
        it outside a step, where it no longer exists, is refused.
        """
        if self.anchor is None:
            raise ValueError(f"edge {self.key!r} names no geometry")
        side, field = self.anchor
        if side == "source":
            return state[self.source[0]][field]
        if pre_step is None:
            raise ValueError(
                f"edge {self.key!r}: the interface norm reads what this edge delivers "
                f"at the pre-step {self.target[0]}.{field} (a target-anchored geometry), "
                f"and under every norm the float floor counts the rounding of the weights "
                f"its mapping forms there, which only the step that solved the group holds")
        held: Any = (pre_step(self.target[0]) if callable(pre_step)
                     else pre_step[self.target[0]])
        return held[field]

    def read(self, states, mappings=None, pre_step=None, band: bool = False,
             lattices=None) -> tuple:
        """This edge's reading at each of *states*: one :class:`PartReading` per part.

        *states* are ``{node: {field: value}}`` dicts (one for a floor,
        two successive iterates for a residual); *mappings* the weights
        the step ran with; *pre_step* the members' pre-step states
        (:meth:`geometry_at`).  Read by
        ``acceleration._interface_readings``, the one generator the
        residual, its float floor and the spectral analysis's reading
        iterate.

        *band* is that the reader's group declares a dead band
        (``atol > 0``).  The source-field part of an edge read at its
        source then also carries what the edge delivers at each state
        (``PartReading.delivered``: the step's edge rule on the stored
        field, with *mappings* and the geometry the step uses), for the
        dead band to ask of both.  Without it nothing more is built than
        ever was.

        *lattices* is ``{(node, field): (mapping, ...)}``, the mappings
        of every edge of the reader's group that reads each geometry
        field (:func:`_position_lattices`); ``None`` or a field it does
        not name is this edge's mapping alone.  It decides only
        ``PartReading.positions`` (where the positions a part rests on
        are rounded: :func:`_rounded_at`), which the float floor reads
        and the residual does not.
        """
        def among(holder):
            return (lattices or {}).get(holder) or (self.mapping,)

        node, field = self.source
        stored = tuple(s[node][field] for s in states)
        source_dtype = jnp.asarray(stored[0]).dtype
        out = []
        for part in self.parts:
            positions = (None,) * len(states)
            if part.what == SOURCE:
                delivered = None
                if band:
                    geoms = ((None,) * len(states) if self.anchor is None
                             else tuple(self.geometry_at(s, pre_step) for s in states))
                    delivered = tuple(_delivered(self.edge, v, mappings, g)
                                      for v, g in zip(stored, geoms))
                values = tuple(self.reading(v, mappings) for v in stored)
                out.append(PartReading(self.edge, source_dtype, values, part, positions,
                                       delivered))
                continue
            if part.what == GEOMETRY:
                g_node, g_field = part.field
                held = tuple(s[g_node][g_field] for s in states)
                values = tuple(_in_kernel_lengths(self.mapping, g) for g in held)
                positions = tuple(_rounded_at(among(part.field), g) for g in held)
                out.append(PartReading(self.edge, jnp.asarray(held[0]).dtype, values,
                                       part, positions))
                continue
            if part.what == DELIVERED and self.anchor is not None:
                geoms = tuple(self.geometry_at(s, pre_step) for s in states)
                values = tuple(self.reading(v, mappings, g) for v, g in zip(stored, geoms))
                if callable(getattr(self.mapping, "geometry_length_scale", None)):
                    side, g_field = self.anchor
                    holder = ((self.source if side == "source" else self.target)[0], g_field)
                    positions = tuple(
                        _read_in_kernel_lengths(self.mapping, g, among(holder)) for g in geoms)
            else:
                values = tuple(self.reading(v, mappings) for v in stored)
            out.append(PartReading(self.edge, source_dtype, values, part, positions))
        return tuple(out)

    @property
    def read_from_state(self) -> bool:
        """Is the source a field of the producer's state (floating or not)?"""
        return self.source_kind in _STATE_FIELD_KINDS

    def delivered_leaves(self, state, mappings=None) -> tuple:
        """``(dtype, size)`` of each array this edge hands its target, read from *state*.

        Static: the edge's own rule (``maddening.core.edge._delivered``:
        the mapping with the weights in *mappings*, then the transform)
        evaluated on the **abstract** value of the source field
        (:func:`jax.eval_shape`: shapes and dtypes, never values), so a
        transform is traced once more and is never run on numbers, and
        nothing is added to a program being traced around the call.  A
        geometry-dependent mapping is handed the abstract value of its
        geometry field in *state* (the pre-step field has the same shape
        and dtype).

        Empty where there is nothing to ask: an edge with neither a
        mapping nor a transform delivers its source field, which the
        caller already holds; a source that is not a field of *state* (a
        boundary flux, an absent field) or a geometry field *state* does
        not hold cannot be evaluated from it.

        Read by ``acceleration._group_coarsest_eps``: a transform (or a
        mapping) that delivers a coarser floating dtype than every field
        of the group is a rounding the pass goes through.
        """
        if not self.read_from_state or (self.mapping is None and not self.has_transform):
            return ()
        geom: tuple = ()
        if self.anchor is not None:
            side, field = self.anchor
            holder = (state.get(self.source[0] if side == "source" else self.target[0])
                      or {})
            if field not in holder:
                return ()
            geom = (holder[field],)
        fields = state.get(self.source[0]) or {}
        if self.source[1] not in fields:
            return ()
        edge = self.edge

        def rule(value, *geometry):
            return _delivered(edge, value, mappings, *geometry)

        out = jax.eval_shape(rule, fields[self.source[1]], *geom)
        return tuple((leaf.dtype, int(np.prod(leaf.shape, dtype=np.int64)))
                     for leaf in jax.tree_util.tree_leaves(out))


def _edge_record(edge, state=None, nodes=None, *, declared: int = 0,
                 role: str = INTERNAL) -> InterfaceEdge:
    """The record of *edge*: the one place an edge's attributes are read.

    Only the source, the mapping and the transform are required of
    *edge*: the static rule ``_reading_is_the_fields`` is also asked
    about edges that name nothing else.
    """
    mapping = edge.mapping
    return InterfaceEdge(
        edge=edge,
        key=getattr(edge, "key", None),
        declared=declared,
        role=role,
        source=(edge.source_node, edge.source_field),
        source_kind=_source_kind(edge, state, nodes),
        target=(getattr(edge, "target_node", None), getattr(edge, "target_field", None)),
        mapping=mapping,
        mapping_kind=getattr(mapping, "kind", None),
        mapping_form=_mapping_form(mapping),
        anchor=getattr(edge, "geometry", None),
        has_transform=edge.transform is not None,
        norm_side=_norm_side(edge),
    )


@dataclass(frozen=True, eq=False)
class InterfacePlan:
    """A coupling group's edges, described once (:func:`interface_plan`).

    Attributes
    ----------
    members : frozenset of str
        The group's nodes.
    order : tuple of str
        The members in the order the group sweeps them.
    internal : tuple of InterfaceEdge
        The internal edges in the canonical order
        (:func:`_interface_edge_order`), the order the interface norm
        sums them in.  Read by :func:`interface_records`.
    crossing : tuple of InterfaceEdge
        The edges with one end in the group, as declared.
    member_fields : dict
        ``{member: its state's field names}`` in the state the plan was
        built from.
    flux_members : frozenset of str
        The members whose class defines ``compute_boundary_fluxes``.
        Read by the group's passes (``one_pass_gs``, ``one_pass_jacobi``,
        ``_read_gain``): the nodes whose fluxes a pass computes.
    """

    members: frozenset
    order: tuple
    internal: tuple
    crossing: tuple
    member_fields: dict
    flux_members: frozenset

    # -- which edges ----------------------------------------------------

    def is_internal(self, edge) -> bool:
        """Is *edge* one of the group's internal edges?

        Read by the pass's boundary resolution (``_resolve_boundary``,
        ``_resolve_boundary_interpolated``): an internal edge is read
        from the iterate, forward, whatever the schedule says.
        """
        return is_internal(edge, self.members)

    def _declared(self, *roles) -> list:
        """The records of *roles*, as declared."""
        pool = (*self.internal, *self.crossing)
        return sorted((r for r in pool if r.role in roles), key=lambda r: r.declared)

    def declared_edges(self) -> tuple:
        """The internal edges (``EdgeSpec``), in the order they were declared.

        Read by the step for the pass's evaluation count
        (``_group_evaluations``, which asks only who reads whom).

        **Differs from** :meth:`norm_edges`, the order every sum of the
        norm is taken in.
        """
        return tuple(r.edge for r in self._declared(INTERNAL))

    def norm_edges(self) -> tuple:
        """The internal edges (``EdgeSpec``), in the canonical order.

        Read by ``compile()`` for the report's fallback float floor
        (``_committed_floor_inputs``, which ``coupling_diagnostics``
        hands ``residual_precision_floor``): the edges of
        :attr:`internal`, so the fallback sums them in the order the
        residual, the floor the step records and the spectral reading
        do.
        """
        return tuple(r.edge for r in self.internal)

    def member_reads(self) -> dict:
        """``{member: members it reads through an internal edge}``, itself included.

        Read by the step's measured evaluation count (``_read_gain``
        through ``_measured_pass_evaluations``).
        """
        reads: dict[str, set] = {nn: set() for nn in self.members}
        for rec in self.internal:
            reads[rec.target[0]].add(rec.source[0])
        return reads

    def coupled_inputs(self) -> dict:
        """``{member: the boundary inputs an internal edge feeds it}``.

        Read by the interface correction (``_apply_interface_overrides``):
        only inputs that come from coupling are corrected.
        """
        fed: dict[str, set] = {}
        for rec in self.internal:
            fed.setdefault(rec.target[0], set()).add(rec.target[1])
        return fed

    # -- which fields are read --------------------------------------------

    def source_fields(self) -> frozenset:
        """``{(source node, source field)}`` of every internal edge, whatever the source is.

        Read by the spectrum's weights under the interface norm
        (``_read_fields`` in ``_run_coupled_block_impl``, which keeps
        the floating state fields among them) and by the non-floating
        fields the IFT solve evaluates at the iterate (``live_nonfloat``).

        **Differs from** :meth:`iqn_fields`: no geometry field is here,
        and a flux edge contributes its flux's name, which is no state
        field, where the accelerator takes the producer's whole state.
        **Differs from** :func:`interface_records`: a field two edges
        read is here once.
        """
        return frozenset(rec.source for rec in self.internal)

    def iqn_fields(self) -> Optional[dict]:
        """Per-node state fields a quasi-Newton acceleration acts on by default.

        The fields the group's internal edges *read* from each producer.
        An edge whose source is not a state field (a boundary flux from
        ``compute_boundary_fluxes``) is a function of the producer's
        state, so the producer's whole state stands in for it.  A
        geometry an edge reads from its producer is read from the
        iterate too, as a second edge would read it.  ``None`` when no
        internal edge exists (accelerate everything).

        Read by ``_group_layout._group_accel_fields`` (the step's
        flattening and ``compile()``'s IQN-IMVJ warm-start seeding).

        **Differs from** :meth:`source_fields` in the geometry field and
        in the flux edge's stand-in.
        """
        ifields: dict[str, set] = {}
        for rec in self.internal:
            node, field = rec.source
            chosen = ifields.setdefault(node, set())
            if rec.read_from_state:
                chosen.add(field)
            else:
                chosen.update(self.member_fields.get(node, ()))
            if rec.anchor is not None and rec.anchor[0] == "source":
                chosen.add(rec.anchor[1])
        return {nn: tuple(sorted(fs)) for nn, fs in ifields.items()} if ifields else None

    def norm_reads_mapping_weights(self) -> bool:
        """Is an internal edge with a floating source read through its mapping?

        Read by ``_group_layout._reads_mapping_weights``: under the
        interface norm such a group's float floor depends on the mapping
        weights the step ran with, so the step records it.  An edge read
        at its source (:attr:`InterfaceEdge.reads_through_mapping`) does
        not count: its reading is the stored field, whatever the weights;
        nor does a geometry-dependent mapping that holds none
        (:attr:`InterfaceEdge.reads_weights_of_the_step`).
        """
        return any(rec.reads_weights_of_the_step and rec.source_kind == FLOATING
                   for rec in self.internal)

    def norm_reads_beyond_the_state(self) -> bool:
        """Does the norm read, on an internal edge with a floating source,
        something the state a solve returns does not hold?

        Either the weights of a mapping the edge is read through
        (:meth:`norm_reads_mapping_weights`: ``params["mappings"]``,
        which a caller may override for one step), or the pre-step
        geometry of a target-anchored mapping read as delivered
        (:attr:`InterfaceEdge.reads_pre_step_geometry`).  The float floor
        of such a group's residual cannot be taken from the returned
        state alone, so the step records it.  Read by
        ``_group_layout._reads_mapping_weights`` (who owns the
        ``reading_floor`` slot) and by the report's fallback floor
        (``_group_layout._floor_needs_the_step``).
        """
        return self.norm_reads_mapping_weights() or any(
            rec.reads_pre_step_geometry and rec.source_kind == FLOATING
            for rec in self.internal)

    def kernel_rounds_beyond_the_state(self) -> bool:
        """Does an internal edge form its kernel's weights from positions
        the state a solve returns does not hold (a geometry-dependent
        mapping of a kind with a length scale, anchored at its target:
        :attr:`InterfaceEdge.kernel_rounds_at_pre_step_positions`)?

        Under ``"l2"`` and ``"mixed"`` the float floor of such a group
        counts the rounding of those weights
        (``acceleration._kernel_rounding``), so it cannot be taken from
        the returned state alone and the step records it, as it does
        the floor of an interface reading taken at a pre-step geometry
        (:meth:`norm_reads_beyond_the_state`).  Read by
        ``_group_layout._reads_mapping_weights`` and
        ``_group_layout._floor_needs_the_step``.
        """
        return any(rec.kernel_rounds_at_pre_step_positions for rec in self.internal)

    def band_reads_beyond_the_state(self) -> bool:
        """With a dead band declared (``atol > 0``), does it ask, on an
        internal edge with a floating source, something the state a solve
        returns does not hold?

        The band of an edge read at its source is asked of the source
        field **and of what the edge delivers**: through the weights of
        its mapping (``params["mappings"]``, which a caller may override
        for one step) or at its target's pre-step geometry.  Which
        entries are in such a group's float floor then cannot be decided
        from the returned state alone, and the step records the floor,
        as it does for a group that reads an edge through its mapping
        (:meth:`norm_reads_beyond_the_state`).  Read by
        ``_group_layout._reads_mapping_weights``.
        """
        return any((rec.band_reads_weights_of_the_step or rec.band_reads_pre_step_geometry)
                   and rec.source_kind == FLOATING for rec in self.internal)

    def measured_whole(self) -> frozenset:
        """``{(node, field)}``: the state fields the norm measures whole on
        an internal edge (:attr:`InterfaceEdge.measured_whole`).

        The fields a solve returns as the iterate it accepted holds
        them; every other floating field is recomputed by one plain
        pass.  Read by ``_group_layout._fields_the_interface_norm_misses``.
        """
        return frozenset(field for rec in self.internal for field in rec.measured_whole)

    def mapped_keys(self) -> tuple:
        """The keys of the internal edges the norm reads through a mapping,
        in the canonical order.

        Read by the step for the mapping weights its reports are taken
        with (``report_mappings``).  An edge read at its source is not
        here: no report reads its weights.
        """
        return tuple(rec.key for rec in self.internal if rec.reads_through_mapping)

    # -- fluxes -----------------------------------------------------------

    def flux_edges(self) -> list:
        """The internal edges whose source is not a state field, as declared.

        Read by ``_group_layout._flux_edge_coupling_errors``: the
        interface norm and an interpolated sub-cycled read both take an
        internal edge's value from the state, and refuse these.
        """
        return [r for r in self._declared(INTERNAL) if not r.read_from_state]

    def resolves_a_flux(self) -> bool:
        """Does any edge with an end in the group carry a boundary flux?

        Read by the group's passes (``has_flux_edges``): whether a pass
        computes its members' fluxes at all.  It counts inbound and
        outbound edges too, which a pass cannot serve (MADD-ANO-156).
        """
        return any(rec.source_kind == FLUX for rec in (*self.internal, *self.crossing))

    def flux_producer_reads_a_flux(self) -> bool:
        """Does a member that produces fluxes read a flux itself?

        Read by the Jacobi pass (``jacobi_flux_sweeps``): such a group
        seeds its producers' fluxes in two sweeps.
        """
        return any(rec.source_kind == FLUX and rec.target[0] in self.flux_members
                   for rec in self._declared(INTERNAL, INBOUND))

    # -- geometry -----------------------------------------------------------

    def geometry_edges(self) -> list:
        """The internal edges whose mapping reads a geometry, as declared.

        Read by ``_group_layout._geometry_edge_coupling_errors``: the
        interface norm reads the geometry of the ``multilinear_grid``
        kind in a group that does not sub-cycle, and refuses the rest.
        """
        return [r for r in self._declared(INTERNAL) if r.anchor is not None]

    def resolved_geometry_edges(self) -> list:
        """The edges with a geometry-dependent mapping that the group's pass
        resolves: every one into a member, from inside the group or outside,
        as declared.

        Read by ``_group_layout._geometry_diagnostics_refusal``, by
        ``compile()`` (the ``geometry_gap`` slot and the report's
        ``_committed_geometry_edges``) and by :meth:`geometry_holders`.

        **Differs from** :meth:`geometry_edges`, which is the internal
        ones only.
        """
        return [r for r in self._declared(INTERNAL, INBOUND) if r.anchor is not None]

    def geometry_holders(self) -> list:
        """``[(node, field, mapping)]``: each geometry field the group's pass
        reads, once, with the first mapping that reads it (the step of the
        self-check is taken on that mapping's lattice).

        Read by the step's geometry self-check (``geometry_checked``).
        """
        seen: dict = {}
        for rec in self.resolved_geometry_edges():
            side, field = rec.anchor
            holder = rec.source[0] if side == "source" else rec.target[0]
            seen.setdefault((holder, field), rec.mapping)
        return [(node, field, mapping) for (node, field), mapping in seen.items()]

    def geometry_iterate_reads(self) -> list:
        """``[(node, field, mapping)]``: every geometry field of a member
        that the group's pass reads from the iterate or from the state the
        same pass has built, once per mapping that reads it so.

        A source-anchored internal edge reads its source's field with its
        value; a target-anchored edge into a member that computes fluxes
        is resolved again for the flux hook, from the member's in-pass
        state.  A target-anchored edge of ``update`` reads the member's
        pre-step state and a source-anchored edge from outside the group
        the outside node's: constants of the pass, not listed.

        Read by the step's lattice-plane limit
        (``_bounds._geometry_plane_limit``): these are the positions that
        differ between the returned iterate and the fixed point.

        **Differs from** :meth:`geometry_holders`, which lists every
        geometry field the pass reads, constants included, once.
        """
        out = []
        for rec in self.resolved_geometry_edges():
            side, field = rec.anchor
            if side == "source":
                holder, moves = rec.source[0], rec.source[0] in self.members
            else:
                holder, moves = rec.target[0], rec.target[0] in self.flux_members
            if moves:
                out.append((holder, field, rec.mapping))
        return out

    def reads_own_geometry(self, holder: str, field: str) -> bool:
        """Does an edge into member *holder* read *field* of *holder*'s own
        state as its geometry (a target anchor)?

        Read by the geometry self-check (``_geometry_gap_at``): only then
        is the member's pre-step geometry a constant the pass needs.
        """
        return any(rec.anchor is not None and rec.anchor == ("target", field)
                   and rec.target[0] == holder
                   for rec in self._declared(INTERNAL, INBOUND))


def interface_plan(members, edges, member_order, state, nodes) -> InterfacePlan:
    """The description of a group of *members* among *edges*.

    Built by ``_run_coupled_block_impl`` when a group's block is traced
    (from the state the step was handed), by ``compile()`` (from the
    graph's state: the seeding of the group's slots and what its reports
    rest on) and by ``validate()`` (the refusals).  The three read the
    same edges; a source's kind is that of the state each was given.

    Parameters
    ----------
    members : collection of str
        The group's nodes.
    edges : iterable of EdgeSpec
        Every edge of the graph, in the order of the ``add_edge`` calls.
    member_order : sequence of str
        The members in the order the group sweeps them (the schedule
        restricted to the group); decides the canonical order.
    state : dict
        ``{node: {field: value}}``: decides each source's kind.
    nodes : dict
        ``{name: node spec}``: decides which sources are fluxes.
    """
    members = frozenset(members)
    internal, crossing = [], []
    for declared, edge in enumerate(edges):
        inside_source = edge.source_node in members
        inside_target = edge.target_node in members
        if inside_source and inside_target:
            internal.append((declared, edge))
        elif inside_target:
            crossing.append(_edge_record(edge, state, nodes, declared=declared, role=INBOUND))
        elif inside_source:
            crossing.append(_edge_record(edge, state, nodes, declared=declared, role=OUTBOUND))
    place = {id(edge): declared for declared, edge in internal}
    ordered = _interface_edge_order([edge for _declared, edge in internal], member_order)
    return InterfacePlan(
        members=members,
        order=tuple(member_order),
        internal=tuple(
            _edge_record(edge, state, nodes, declared=place[id(edge)], role=INTERNAL)
            for edge in ordered),
        crossing=tuple(crossing),
        member_fields={nn: tuple((state or {}).get(nn, {})) for nn in members},
        flux_members=frozenset(
            nn for nn in members
            if nodes is not None and nn in nodes and _defines_fluxes(nodes[nn].node)),
    )


def interface_records(interface_edges, state=None) -> tuple:
    """The records the interface norm reads, in the order it sums them.

    A group's plan gives its internal edges in the canonical order.  A
    bare sequence of edges (a direct call of
    ``coupling_residual_interface`` or ``residual_precision_floor``, and
    the report's fallback floor, which is handed the plan's own order:
    :meth:`InterfacePlan.norm_edges`) is read in the order given, each
    edge described in *state* (without the nodes, so a source that is
    not a state field is ``"absent"``).  The side each is read on is the
    edge's own (:func:`_norm_side`), the same here as in a plan.

    Read by ``acceleration._interface_readings`` (the residual, its
    float floor and the spectral analysis's reading) and by
    ``_group_layout._reading_is_the_fields``.

    **Differs from** :meth:`InterfacePlan.source_fields` and
    :meth:`InterfacePlan.iqn_fields`: one record per *edge*, so a field
    that two edges read is read twice.
    """
    if isinstance(interface_edges, InterfacePlan):
        return interface_edges.internal
    return tuple(_edge_record(edge, state, declared=i)
                 for i, edge in enumerate(interface_edges))
