"""
replace_node -- swap a physics node for a surrogate while preserving wiring.
"""

import logging
from dataclasses import fields as dataclass_fields

from maddening.surrogates.node import SurrogateNode


logger = logging.getLogger(__name__)


# Edges go back through :meth:`~maddening.core.edge.EdgeSpec.add_edge_kwargs`,
# which derives the call from the dataclass's own fields.  ``ordinal`` is
# the one field it leaves to ``add_edge``, and it does not need carrying
# across here: ``remove_node`` drops *every* edge touching the replaced
# node, and an edge that shares a saved edge's base key necessarily
# touches that node too -- the key is built from the two endpoint names
# and the surrogate must keep the original's name -- so no edge with a
# colliding base key survives the removal, and re-adding the saved edges
# in their saved order recomputes exactly the ordinals (and therefore the
# keys) they had.  ``replace_node`` verifies that per swap rather than
# trusting the argument.

# ``ExternalInputSpec`` has no equivalent method of its own, so its table
# lives here and is checked against the dataclass the same way.
EXTERNAL_INPUT_FIELD_TO_ADD_KWARG = {
    "target_node": "target_node",
    "target_field": "target_field",
    "shape": "shape",
    "dtype": "dtype",
}


def _restore_kwargs(spec, table):
    """Keyword arguments that re-add ``spec`` with every field intact.

    Parameters
    ----------
    spec : dataclass instance
        The saved ``ExternalInputSpec``.
    table : dict
        Maps each dataclass field name to the re-adding method's keyword.

    Raises
    ------
    RuntimeError
        If ``spec`` carries a field the table does not name -- it would
        otherwise be reset to its default silently, which is the bug this
        indirection exists to prevent.
    """
    unknown = sorted(
        f.name for f in dataclass_fields(spec) if f.name not in table
    )
    if unknown:
        raise RuntimeError(
            f"replace_node cannot preserve {type(spec).__name__} field(s) "
            f"{unknown}: they are not in this module's restore table, so "
            f"re-adding would silently reset them to their defaults.  Add "
            f"them to maddening.surrogates.replace._core (and to the "
            f"re-adding GraphManager method if it does not take them)."
        )
    return {kwarg: getattr(spec, name) for name, kwarg in table.items()}


def replace_node(gm, original_name: str, surrogate_node: SurrogateNode):
    """Replace a node in a GraphManager with a surrogate, preserving edges.

    Every attribute of every edge and external input that touches the
    replaced node survives the swap: ``additive``, ``source_units`` /
    ``target_units``, the interface ``mapping`` and its live weights in
    ``gm.params["mappings"]``, plus any :class:`ParamSpec` overrides set
    on a mapped edge.  The graph therefore produces the same numbers
    before and after the replacement, up to the surrogate's own error.

    Parameters
    ----------
    gm : GraphManager
        The graph manager containing the original node.
    original_name : str
        Name of the node to replace.
    surrogate_node : SurrogateNode
        The surrogate node (must have ``surrogate_node.name == original_name``).

    Raises
    ------
    ValueError
        If surrogate name doesn't match or node doesn't exist.
    RuntimeError
        If an edge or external input carries an attribute this function
        does not know how to carry across, or if the restored edges do
        not have the keys they had before the swap.

    Notes
    -----
    v0.2 #3 follow-up: if the original node and the replacement carry
    different :attr:`~maddening.core.node.SimulationNode.static_data`
    shapes, a warning is logged.  The replacement still proceeds — the
    next ``step()`` will recompile because the static_data hash drifted —
    but the warning makes the cache-invalidation visible.  Most genuine
    surrogate replacements should NOT trigger this: the surrogate is
    expected to mirror the physics node's shape contract.
    """
    if surrogate_node.name != original_name:
        raise ValueError(
            f"Surrogate name '{surrogate_node.name}' must match "
            f"original name '{original_name}'."
        )
    if original_name not in gm._nodes:
        raise KeyError(f"No node named '{original_name}' in graph.")

    # v0.2 #3 follow-up: log on static_data drift (advisory, non-blocking).
    old_node = gm._nodes[original_name].node
    old_hash = old_node.static_data_hash()
    new_hash = surrogate_node.static_data_hash()
    if old_hash != new_hash:
        logger.warning(
            "replace_node(%r): static_data_hash changed (%d -> %d). "
            "The next step() will recompile.  If the surrogate is "
            "supposed to mirror the physics node's static_data shape, "
            "double-check the new node's static_data property.",
            original_name, old_hash, new_hash,
        )

    # Save edges and external inputs referencing this node
    saved_edges = [
        e for e in gm._edges
        if e.source_node == original_name or e.target_node == original_name
    ]
    saved_external = [
        ei for ei in gm._external_inputs
        if ei.target_node == original_name
    ]

    # A mapped edge's weights live in ``gm.params["mappings"][edge.key]``
    # and are dropped by ``remove_node``.  The next compile re-snapshots
    # them from the mapping object, which silently undoes anything sysid
    # (or a hand edit of gm.params) had moved away from the recipe, so
    # save the live values and put them back.  Same for the ParamSpec
    # overrides that make a mapped edge's weights trainable.
    live_mappings = (gm.params or {}).get("mappings") or {}
    saved_mapping_params = {
        e.key: dict(live_mappings[e.key])
        for e in saved_edges
        if e.mapping is not None and e.key in live_mappings
    }
    saved_edge_param_specs = {
        e.key: dict(gm._param_spec_overrides[e.key])
        for e in saved_edges
        if e.key in gm._param_spec_overrides
    }

    # Build the re-add arguments *before* mutating the graph, so an edge
    # this function cannot preserve aborts the replacement instead of
    # leaving the graph half-rewired.
    edge_kwargs = [e.add_edge_kwargs() for e in saved_edges]
    external_kwargs = [
        _restore_kwargs(ei, EXTERNAL_INPUT_FIELD_TO_ADD_KWARG)
        for ei in saved_external
    ]

    # Remove original (this also removes edges and external inputs)
    gm.remove_node(original_name)

    # Add surrogate
    gm.add_node(surrogate_node)

    # Re-add saved edges, every attribute included
    for kwargs in edge_kwargs:
        gm.add_edge(**kwargs)

    # Re-add saved external inputs
    for kwargs in external_kwargs:
        gm.add_external_input(**kwargs)

    # ``ordinal`` -- and with it ``EdgeSpec.key``, which names the
    # ``params["mappings"]`` slot -- is recomputed by ``add_edge``.  It
    # must come out the same or the restored weights below would land in
    # the wrong slot; see the note above on ``ordinal``.
    restored_keys = [
        e.key for e in gm._edges
        if e.source_node == original_name or e.target_node == original_name
    ]
    if restored_keys != [e.key for e in saved_edges]:
        raise RuntimeError(
            f"replace_node({original_name!r}): restored edge keys "
            f"{restored_keys} differ from the saved ones "
            f"{[e.key for e in saved_edges]}; mapping weights would be "
            f"attached to the wrong edge."
        )

    if saved_mapping_params:
        gm.params.setdefault("mappings", {}).update(saved_mapping_params)
    for edge_key, overrides in saved_edge_param_specs.items():
        for param_key, spec in overrides.items():
            gm.set_param_spec(edge_key, param_key, spec)
