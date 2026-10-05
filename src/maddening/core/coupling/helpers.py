"""
Coupling helper utilities for common coupling patterns.

Provides convenience functions for setting up value-based,
flux-based, Dirichlet-Neumann, Robin, and symmetric coupling
between nodes, plus conservation monitoring.
"""

from __future__ import annotations

from typing import Callable, Optional

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

    Each node's flux is its ``compute_boundary_fluxes`` at *state*, given
    the boundary inputs the graph's edges deliver from *state*.  Those are
    resolved by the rule the compiled step uses
    (``GraphManager._boundary_inputs_from``, which
    ``resolve_boundary_inputs`` shares): the edge's interface mapping,
    then its transform, additive edges summed, with the mapping weights
    and the node constants of ``gm.params``.  An edge that reads another
    node's *flux output* takes that node's flux at *state*, computed the
    same way.

    Every value is taken from the one *state* passed in.  The step itself
    mixes time levels (a forward edge reads this step's value, a back
    edge the previous one), so on a transient the diagnostic is the
    imbalance *of that state*, not a replay of the last step; at a
    converged or steady state the two agree.

    Parameters
    ----------
    gm : GraphManager
        The graph manager (used for node lookup).
    state : dict
        Current graph state.  It must hold every node an edge into the
        compared nodes reads from.
    flux_pairs : list of (node_a, flux_a, node_b, flux_b)
        Each tuple identifies an interface where ``flux_a`` and
        ``flux_b`` measure the same physical quantity from each side.
        Conservation means they are equal (difference is zero).

    Returns
    -------
    dict[str, float]
        ``{interface_name: imbalance}`` where imbalance is close
        to zero for conservative coupling.

    Raises
    ------
    KeyError
        A node of a pair, or the source of an edge into one, is not in
        the graph or not in *state*; a node does not report the flux
        named for it (a misspelt flux used to read as ``0.0``); an edge
        reads a field that is neither a state field nor a flux output of
        its source.
    ValueError
        Two nodes read each other's flux outputs (Robin-Robin coupling,
        say).  Such inputs are what the coupling iteration converged to
        and cannot be rebuilt from a state alone; the diagnostic used to
        drop them and compare fluxes computed without them.
    """
    from maddening.core.graph_manager import _node_fluxes  # noqa: PLC0415

    node_params = gm._params_or_default(None).get("nodes", {})
    computed: dict[str, dict] = {}

    def fluxes_of(name: str, pending: tuple = ()) -> dict:
        """``name``'s flux outputs at *state*, with the inputs the edges give it."""
        if name in computed:
            return computed[name]
        if name not in gm._nodes:
            raise KeyError(f"check_conservation: the graph has no node {name!r}")
        if name not in state:
            raise KeyError(f"check_conservation: state has no entry for node {name!r}")
        if name in pending:
            loop = " -> ".join(pending[pending.index(name):] + (name,))
            raise ValueError(
                f"check_conservation: the boundary inputs of {name!r} read flux outputs "
                f"that depend on its own ({loop}).  Inside a coupling group those are "
                f"the values the iteration converged to, which cannot be rebuilt from a "
                f"state alone, so the fluxes at this interface cannot be compared here"
            )
        upstream: dict[str, dict] = {}
        for edge in gm._edges:
            if edge.target_node != name:
                continue
            src = edge.source_node
            if src not in state:
                raise KeyError(
                    f"check_conservation: state has no entry for node {src!r}, the "
                    f"source of edge {edge.key}"
                )
            if edge.source_field in state[src]:
                continue
            # Not a state field: a flux output of the source, as in the step.
            upstream[src] = fluxes_of(src, pending + (name,))
            if edge.source_field not in upstream[src]:
                raise KeyError(
                    f"check_conservation: edge {edge.key} reads {edge.source_field!r}, "
                    f"which is neither a state field of {src!r} "
                    f"({sorted(state[src])}) nor one of its flux outputs "
                    f"({sorted(upstream[src])})"
                )
        inputs = gm._boundary_inputs_from(state, name, fluxes=upstream)
        spec = gm._nodes[name]
        computed[name] = _node_fluxes(
            spec, state[name], inputs, spec.timestep, node_params.get(name))
        return computed[name]

    def flux(name: str, field: str):
        fluxes = fluxes_of(name)
        if field not in fluxes:
            raise KeyError(
                f"check_conservation: node {name!r} reports no flux {field!r}; its "
                f"compute_boundary_fluxes returns {sorted(fluxes)}"
            )
        return fluxes[field]

    result: dict[str, float] = {}
    for node_a, flux_a, node_b, flux_b in flux_pairs:
        fa = flux(node_a, flux_a)
        fb = flux(node_b, flux_b)
        imbalance = float(jnp.sum(fa - fb))
        interface_name = f"{node_a}.{flux_a}-{node_b}.{flux_b}"
        result[interface_name] = imbalance
    return result
