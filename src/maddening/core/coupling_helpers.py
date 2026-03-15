"""
Coupling helper utilities for common coupling patterns.

Provides convenience functions for setting up value-based,
flux-based, Dirichlet-Neumann, Robin, and symmetric coupling
between nodes, plus conservation monitoring.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import jax.numpy as jnp

from maddening.core.edge import EdgeSpec


def add_value_coupling(
    gm,
    source: str,
    target: str,
    field: str,
    target_field: Optional[str] = None,
    transform: Optional[Callable] = None,
) -> None:
    """Add a simple value-passing edge (most common pattern).

    Parameters
    ----------
    gm : GraphManager
        The graph manager to add the edge to.
    source : str
        Source node name.
    target : str
        Target node name.
    field : str
        Source field name.
    target_field : str or None
        Target boundary input name.  Defaults to *field*.
    transform : callable or None
        Optional JAX-traceable transform.
    """
    if target_field is None:
        target_field = field
    gm.add_edge(source, target, field, target_field, transform=transform)


def add_flux_coupling(
    gm,
    source: str,
    target: str,
    flux_field: str,
    target_input: str,
    transform: Optional[Callable] = None,
    additive: bool = False,
) -> None:
    """Add a flux-based edge.

    The edge reads from the source node's ``compute_boundary_fluxes``
    output rather than from its state.

    Parameters
    ----------
    gm : GraphManager
        The graph manager to add the edge to.
    source : str
        Source node name.
    target : str
        Target node name.
    flux_field : str
        Field name in the source's flux output.
    target_input : str
        Target boundary input name.
    transform : callable or None
        Optional JAX-traceable transform.
    additive : bool
        If True, accumulate with existing boundary input.
    """
    edge = EdgeSpec(
        source_node=source,
        target_node=target,
        source_field=flux_field,
        target_field=target_input,
        transform=transform,
        additive=additive,
    )
    gm._edges.append(edge)
    gm._dirty = True


def add_dirichlet_neumann_pair(
    gm,
    dirichlet_node: str,
    neumann_node: str,
    value_field: str,
    flux_field: str,
    value_input: str,
    flux_input: str,
    value_transform: Optional[Callable] = None,
    flux_transform: Optional[Callable] = None,
) -> None:
    """Set up Dirichlet-Neumann coupling between two nodes.

    *dirichlet_node* receives the VALUE from *neumann_node*'s state.
    *neumann_node* receives the FLUX from *dirichlet_node*'s
    ``compute_boundary_fluxes``.

    Parameters
    ----------
    gm : GraphManager
        The graph manager.
    dirichlet_node : str
        Node that receives a Dirichlet (value) BC.
    neumann_node : str
        Node that receives a Neumann (flux) BC.
    value_field : str
        State field on the Neumann node that provides the value.
    flux_field : str
        Flux field on the Dirichlet node.
    value_input : str
        Boundary input name on the Dirichlet node.
    flux_input : str
        Boundary input name on the Neumann node.
    value_transform : callable or None
        Transform for the value edge.
    flux_transform : callable or None
        Transform for the flux edge.
    """
    # Value edge: neumann -> dirichlet (state field)
    gm.add_edge(
        neumann_node, dirichlet_node, value_field, value_input,
        transform=value_transform,
    )
    # Flux edge: dirichlet -> neumann (flux output)
    add_flux_coupling(
        gm, dirichlet_node, neumann_node, flux_field, flux_input,
        transform=flux_transform,
    )


def add_symmetric_value_coupling(
    gm,
    node_a: str,
    node_b: str,
    field_a: str,
    input_a: str,
    field_b: str,
    input_b: str,
    transform_a_to_b: Optional[Callable] = None,
    transform_b_to_a: Optional[Callable] = None,
) -> None:
    """Add bidirectional value coupling.

    A.field_a -> B.input_b and B.field_b -> A.input_a.

    Parameters
    ----------
    gm : GraphManager
        The graph manager.
    node_a, node_b : str
        Node names.
    field_a : str
        Source field on node A.
    input_a : str
        Boundary input on node A (receives from B).
    field_b : str
        Source field on node B.
    input_b : str
        Boundary input on node B (receives from A).
    transform_a_to_b : callable or None
        Transform for the A->B edge.
    transform_b_to_a : callable or None
        Transform for the B->A edge.
    """
    gm.add_edge(node_a, node_b, field_a, input_b, transform=transform_a_to_b)
    gm.add_edge(node_b, node_a, field_b, input_a, transform=transform_b_to_a)


def add_robin_coupling(
    gm,
    node_a: str,
    node_b: str,
    value_field_a: str,
    flux_field_a: str,
    value_field_b: str,
    flux_field_b: str,
    input_a: str,
    input_b: str,
    alpha: float = 1.0,
) -> None:
    """Robin-Robin coupling between two nodes.

    Each node receives a Robin BC constructed from the other's value
    and flux.  Requires both nodes to implement
    ``compute_boundary_fluxes``.

    The Robin combination is::

        robin_a = alpha * value_b + (1 - alpha) * flux_b
        robin_b = alpha * value_a + (1 - alpha) * flux_a

    Parameters
    ----------
    gm : GraphManager
        The graph manager.
    node_a, node_b : str
        Node names.
    value_field_a, flux_field_a : str
        State field and flux field on node A.
    value_field_b, flux_field_b : str
        State field and flux field on node B.
    input_a, input_b : str
        Boundary input names (receiving Robin BC).
    alpha : float
        Mixing coefficient.  ``alpha=1`` is pure Dirichlet,
        ``alpha=0`` is pure Neumann.
    """
    # Store alpha in closure for the Robin transform
    a = float(alpha)

    # B -> A: robin_a = alpha * value_b + (1-alpha) * flux_b
    # We need TWO edges to A's input: value and flux, both additive.
    # Value component
    gm._edges.append(EdgeSpec(
        source_node=node_b,
        target_node=node_a,
        source_field=value_field_b,
        target_field=input_a,
        transform=lambda v, _a=a: _a * v,
        additive=False,
    ))
    # Flux component (additive on top)
    gm._edges.append(EdgeSpec(
        source_node=node_b,
        target_node=node_a,
        source_field=flux_field_b,
        target_field=input_a,
        transform=lambda f, _a=a: (1.0 - _a) * f,
        additive=True,
    ))

    # A -> B: robin_b = alpha * value_a + (1-alpha) * flux_a
    gm._edges.append(EdgeSpec(
        source_node=node_a,
        target_node=node_b,
        source_field=value_field_a,
        target_field=input_b,
        transform=lambda v, _a=a: _a * v,
        additive=False,
    ))
    gm._edges.append(EdgeSpec(
        source_node=node_a,
        target_node=node_b,
        source_field=flux_field_a,
        target_field=input_b,
        transform=lambda f, _a=a: (1.0 - _a) * f,
        additive=True,
    ))
    gm._dirty = True


def check_conservation(
    gm,
    state: dict[str, dict],
    flux_pairs: list[tuple[str, str, str, str]],
) -> dict[str, float]:
    """Compute flux imbalance across coupling interfaces.

    An observer/diagnostic, not part of the iteration loop.

    For two domains sharing a boundary, conservation means the flux
    computed on each side is the same.  The imbalance is the
    difference ``flux_a - flux_b``, which should be near zero.

    Parameters
    ----------
    gm : GraphManager
        The graph manager (used for node lookup).
    state : dict
        Current graph state.
    flux_pairs : list of (node_a, flux_a, node_b, flux_b)
        Each tuple identifies an interface where ``flux_a`` and
        ``flux_b`` measure the same physical quantity from each side.
        Conservation means they are equal (difference is zero).

    Returns
    -------
    dict[str, float]
        ``{interface_name: imbalance}`` where imbalance is close
        to zero for conservative coupling.
    """
    result: dict[str, float] = {}
    for node_a, flux_a, node_b, flux_b in flux_pairs:
        # Compute fluxes
        spec_a = gm._nodes[node_a]
        spec_b = gm._nodes[node_b]
        bi_a: dict[str, Any] = {}
        bi_b: dict[str, Any] = {}
        # Resolve boundary inputs for flux computation
        for edge in gm._edges:
            if edge.target_node == node_a:
                src_dict = state.get(edge.source_node, {})
                if edge.source_field in src_dict:
                    val = src_dict[edge.source_field]
                else:
                    continue
                if edge.transform is not None:
                    val = edge.transform(val)
                bi_a[edge.target_field] = val
            if edge.target_node == node_b:
                src_dict = state.get(edge.source_node, {})
                if edge.source_field in src_dict:
                    val = src_dict[edge.source_field]
                else:
                    continue
                if edge.transform is not None:
                    val = edge.transform(val)
                bi_b[edge.target_field] = val
        fluxes_a = spec_a.node.compute_boundary_fluxes(
            state[node_a], bi_a, spec_a.timestep
        )
        fluxes_b = spec_b.node.compute_boundary_fluxes(
            state[node_b], bi_b, spec_b.timestep
        )
        fa = fluxes_a.get(flux_a, jnp.array(0.0))
        fb = fluxes_b.get(flux_b, jnp.array(0.0))
        imbalance = float(jnp.sum(fa - fb))
        interface_name = f"{node_a}.{flux_a}-{node_b}.{flux_b}"
        result[interface_name] = imbalance
    return result
