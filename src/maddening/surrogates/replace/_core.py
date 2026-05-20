"""
replace_node -- swap a physics node for a surrogate while preserving wiring.
"""

import logging

from maddening.surrogates.node import SurrogateNode


logger = logging.getLogger(__name__)


def replace_node(gm, original_name: str, surrogate_node: SurrogateNode):
    """Replace a node in a GraphManager with a surrogate, preserving edges.

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

    # Remove original (this also removes edges and external inputs)
    gm.remove_node(original_name)

    # Add surrogate
    gm.add_node(surrogate_node)

    # Re-add saved edges
    for e in saved_edges:
        gm.add_edge(
            e.source_node, e.target_node,
            e.source_field, e.target_field,
            e.transform,
        )

    # Re-add saved external inputs
    for ei in saved_external:
        gm.add_external_input(
            ei.target_node, ei.target_field,
            ei.shape, ei.dtype,
        )
