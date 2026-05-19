"""
replace_node -- swap a physics node for a surrogate while preserving wiring.
"""

from maddening.surrogates.node import SurrogateNode


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
    """
    if surrogate_node.name != original_name:
        raise ValueError(
            f"Surrogate name '{surrogate_node.name}' must match "
            f"original name '{original_name}'."
        )
    if original_name not in gm._nodes:
        raise KeyError(f"No node named '{original_name}' in graph.")

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
