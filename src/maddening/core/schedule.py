"""
Scheduling utilities -- topological sort and cycle detection.

Uses Kahn's algorithm for topological ordering.  When cycles are
detected they are reported so the GraphManager can apply staggering
(use previous-timestep values for back-edges).
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Sequence

from maddening.core.edge import EdgeSpec


def _build_adjacency(
    node_names: Sequence[str],
    edges: Sequence[EdgeSpec],
) -> tuple[dict[str, set[str]], dict[str, int]]:
    """Return (adjacency list, in-degree map)."""
    adj: dict[str, set[str]] = defaultdict(set)
    in_deg: dict[str, int] = {n: 0 for n in node_names}
    for e in edges:
        if e.source_node != e.target_node and e.target_node not in adj[e.source_node]:
            adj[e.source_node].add(e.target_node)
            in_deg[e.target_node] = in_deg.get(e.target_node, 0) + 1
    return adj, in_deg


def topological_sort(
    node_names: Sequence[str],
    edges: Sequence[EdgeSpec],
) -> list[str]:
    """Return a topological ordering of *node_names* given *edges*.

    If a cycle exists the function still returns an ordering (omitting
    the nodes that participate in the cycle from the sorted portion).
    Callers should use :func:`detect_cycles` separately to decide how
    to handle cycles.
    """
    adj, in_deg = _build_adjacency(node_names, edges)
    queue: deque[str] = deque(n for n in node_names if in_deg[n] == 0)
    order: list[str] = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in sorted(adj[node]):  # sorted for determinism
            in_deg[neighbour] -= 1
            if in_deg[neighbour] == 0:
                queue.append(neighbour)

    # If some nodes were not reached they are part of a cycle.
    # Append them in their original order so every node appears exactly once.
    if len(order) < len(node_names):
        remaining = [n for n in node_names if n not in set(order)]
        order.extend(remaining)
    return order


def detect_cycles(
    node_names: Sequence[str],
    edges: Sequence[EdgeSpec],
) -> list[tuple[str, ...]]:
    """Return a list of cycles found in the graph.

    Each cycle is a tuple of node names forming the loop.
    Uses DFS-based cycle detection.
    """
    adj, _ = _build_adjacency(node_names, edges)
    WHITE, GRAY, BLACK = 0, 1, 2
    colour: dict[str, int] = {n: WHITE for n in node_names}
    path: list[str] = []
    cycles: list[tuple[str, ...]] = []

    def _dfs(u: str) -> None:
        colour[u] = GRAY
        path.append(u)
        for v in sorted(adj[u]):
            if colour[v] == GRAY:
                # Found cycle: extract the loop from path
                idx = path.index(v)
                cycles.append(tuple(path[idx:]))
            elif colour[v] == WHITE:
                _dfs(v)
        path.pop()
        colour[u] = BLACK

    for n in node_names:
        if colour[n] == WHITE:
            _dfs(n)
    return cycles


def identify_back_edges(
    schedule: Sequence[str],
    edges: Sequence[EdgeSpec],
) -> list[EdgeSpec]:
    """Given an execution *schedule*, return the edges that violate
    topological order (back-edges).  These must use staggered
    (previous-timestep) values.
    """
    pos = {name: i for i, name in enumerate(schedule)}
    return [e for e in edges if pos.get(e.source_node, -1) >= pos.get(e.target_node, -1)]


def find_strongly_connected_components(
    node_names: Sequence[str],
    edges: Sequence[EdgeSpec],
) -> list[list[str]]:
    """Return strongly connected components using Tarjan's algorithm.

    Only returns SCCs with more than one node (i.e. actual cycles).
    Each SCC is a list of node names.
    """
    adj: dict[str, list[str]] = {n: [] for n in node_names}
    for e in edges:
        if e.source_node in adj and e.target_node in adj:
            adj[e.source_node].append(e.target_node)

    index_counter = [0]
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    sccs: list[list[str]] = []

    def strongconnect(v: str) -> None:
        indices[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in adj[v]:
            if w not in indices:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            scc: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                scc.append(w)
                if w == v:
                    break
            if len(scc) > 1:
                sccs.append(scc[::-1])  # reverse for natural order

    for n in node_names:
        if n not in indices:
            strongconnect(n)

    return sccs
