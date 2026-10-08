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

The exact model and the numerical reference of the test suite
(``tests/property/coupled_topologies.py``, ``coupling_reference.py``) are
oracles and do not import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp

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

    A geometry-dependent mapping stays ``"delivered"``: ``compile()``
    refuses one inside a group under the interface norm
    (``_geometry_edge_coupling_errors``), so nothing reads it.
    """
    mapping = edge.mapping
    if mapping is None or getattr(mapping, "needs_geometry", False):
        return DELIVERED
    source_lead, target_lead = _mapping_leads(mapping)
    return SOURCE if _entries(target_lead) > _entries(source_lead) else DELIVERED


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
        ``"source"`` where a static mapping delivers more entries than the
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

    def reading(self, value, mappings=None):
        """What the interface norm reads on this edge for *value* at its source.

        On the delivered side: *value* through the mapping, with the
        weights in *mappings* (``params["mappings"]``, ``None`` for the
        mapping's own), then the transform -- the step's edge rule.  On
        the source side: *value* itself, through neither.  Read by
        ``acceleration._interface_readings`` (the residual, its float
        floor and its eps, and the spectral analysis's reading).

        The source side was decided on the sizes the mapping declares; a
        *value* that does not have them is refused rather than read on a
        side its own size would not have chosen.
        """
        if self.norm_side == DELIVERED:
            return _delivered(self.edge, value, mappings)
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
    def reads_source_as_is(self) -> bool:
        """Is the norm's reading of this edge its source field, unchanged?

        On the source side: always.  On the delivered side: an edge with
        no mapping and no transform.  Read by
        ``_group_layout._reading_is_the_fields`` (which spectral analysis
        the report takes).
        """
        if self.norm_side == SOURCE:
            return True
        return (self.norm_side == DELIVERED and self.mapping is None
                and not self.has_transform)

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
    def read_from_state(self) -> bool:
        """Is the source a field of the producer's state (floating or not)?"""
        return self.source_kind in _STATE_FIELD_KINDS


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
        not count: its reading is the stored field, whatever the weights.
        """
        return any(rec.reads_through_mapping and rec.source_kind == FLOATING
                   for rec in self.internal)

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
        interface norm does not read a moving geometry and refuses them.
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
