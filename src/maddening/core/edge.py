"""
EdgeSpec -- an immutable description of a data dependency between two nodes.

An edge says: "before updating *target_node*, copy
*source_node.state[source_field]* into boundary_inputs[target_field],
optionally applying *transform* first."

``transform``, if provided, must be a JAX-traceable pure function.
"""

from dataclasses import dataclass, fields
from typing import Any, Callable, Optional

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


# Every :class:`EdgeSpec` field, mapped to the
# ``GraphManager.add_edge`` keyword that restores it.  Anything putting a
# saved edge back into a graph builds its call from this table, through
# :meth:`EdgeSpec.add_edge_kwargs`, instead of passing arguments
# positionally: a positional re-add that stopped at ``transform`` left
# ``add_edge``'s four remaining parameters on their defaults, so every
# ``replace_node`` reset ``additive`` to ``False``, both units to ``None``
# and ``mapping`` to ``None``.  Nothing errored; an additive boundary
# input simply measured 3.0 before a swap and 1.0 after it.
_ADD_EDGE_KWARGS = {
    "source_node": "source",
    "target_node": "target",
    "source_field": "source_field",
    "target_field": "target_field",
    "transform": "transform",
    "additive": "additive",
    "source_units": "source_units",
    "target_units": "target_units",
    "mapping": "mapping",
}

# ``ordinal`` is the one field ``add_edge`` does not take: it numbers the
# mapped edges sharing a field pair and is assigned by counting the ones
# already in the graph.  A caller re-adding a *set* of edges in their
# original order gets the original ordinals back; one re-adding a single
# edge into a graph that still holds its siblings does not, and should
# check ``key`` afterwards rather than assume.
_DERIVED_BY_ADD_EDGE = frozenset({"ordinal"})


@stability(StabilityLevel.STABLE)
@dataclass(frozen=True)
class EdgeSpec:
    source_node: str
    target_node: str
    source_field: str
    target_field: str
    transform: Optional[Callable] = None
    additive: bool = False  # If True, ADD to existing boundary_input value
    source_units: Optional[str] = None  # Physical units of the source field
    target_units: Optional[str] = None  # Physical units after transform
    # Interface mapping (``maddening.core.coupling.mapping.Mapping``)
    # applied before ``transform``; its weights live in
    # ``GraphManager.params["mappings"][edge.key]``.
    mapping: Optional[Any] = None
    # Position among mapped edges on the same field pair (two additive
    # mapped edges a.v->b.inp are legal); makes ``key`` -- and with it the
    # ``params["mappings"]`` slot -- unique.  Assigned by ``add_edge``.
    ordinal: int = 0

    @property
    def key(self) -> str:
        """Stable identifier: ``"<src>.<field>-><tgt>.<field>"`` (plus
        ``"#<n>"`` for the n-th further mapped edge on the same pair)."""
        base = (f"{self.source_node}.{self.source_field}->"
                f"{self.target_node}.{self.target_field}")
        return base if not self.ordinal else f"{base}#{self.ordinal}"

    def add_edge_kwargs(self) -> dict:
        """Arguments that re-add this edge with every attribute intact.

        The supported way to put a saved edge back into a graph::

            gm.add_edge(**edge.add_edge_kwargs())

        Returns
        -------
        dict
            Every field of this edge under the
            :meth:`~maddening.core.graph_manager.GraphManager.add_edge`
            keyword that restores it.  :attr:`ordinal` is absent:
            ``add_edge`` assigns it from the mapped edges already on the
            same field pair.

        Raises
        ------
        RuntimeError
            If this class has grown a field that the table does not
            name.  Loud is the point: the alternative is the caller
            silently re-adding the edge with that field on its default.

        Notes
        -----
        Derived from this dataclass's own fields rather than written out
        at each call site, so a tenth field cannot quietly stop being
        carried across a ``replace_node`` or a surrogate revert the way
        ``additive``, the units and ``mapping`` did.
        """
        unknown = sorted(
            f.name for f in fields(self)
            if f.name not in _ADD_EDGE_KWARGS and f.name not in _DERIVED_BY_ADD_EDGE
        )
        if unknown:
            raise RuntimeError(
                f"EdgeSpec field(s) {unknown} are in neither "
                f"maddening.core.edge._ADD_EDGE_KWARGS nor "
                f"_DERIVED_BY_ADD_EDGE, so re-adding this edge would reset "
                f"them to their defaults.  Add them to one table (and to "
                f"GraphManager.add_edge if it does not take them)."
            )
        return {kwarg: getattr(self, name) for name, kwarg in _ADD_EDGE_KWARGS.items()}

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        d = {
            "source_node": self.source_node,
            "target_node": self.target_node,
            "source_field": self.source_field,
            "target_field": self.target_field,
        }
        if self.mapping is not None:
            # A factory-built mapping describes itself as its MappingSpec
            # (kind, hyper-parameters, point references, shape -- never
            # the weights), which from_dict rebuilds; anything else can
            # only be named.  GraphManager.to_dict checks completeness.
            describe = getattr(self.mapping, "describe", None)
            d["mapping"] = describe() if callable(describe) else {
                "kind": getattr(self.mapping, "kind", type(self.mapping).__name__),
            }
        if self.ordinal:
            d["ordinal"] = self.ordinal
        if self.transform is not None:
            # The registered name reloads through add_edge(transform=str);
            # an unregistered callable can only be named, not rebuilt.
            from maddening.core.transforms import get_transform_name  # noqa: PLC0415
            d["transform"] = get_transform_name(self.transform) or self.transform.__qualname__
        if self.additive:
            d["additive"] = True
        if self.source_units is not None:
            d["source_units"] = self.source_units
        if self.target_units is not None:
            d["target_units"] = self.target_units
        return d

    def __repr__(self) -> str:
        arrow = f"{self.source_node}.{self.source_field} -> {self.target_node}.{self.target_field}"
        if self.mapping is not None:
            arrow += f"  (mapping {self.mapping!r})"
        if self.transform is not None:
            arrow += f"  (via {self.transform.__qualname__})"
        if self.source_units or self.target_units:
            arrow += f"  [{self.source_units or '?'} -> {self.target_units or '?'}]"
        return f"EdgeSpec({arrow})"
