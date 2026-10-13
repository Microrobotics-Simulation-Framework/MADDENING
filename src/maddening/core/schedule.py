"""
Scheduling utilities -- topological sort and cycle detection.

Uses Kahn's algorithm for topological ordering.  When cycles are
detected they are reported so the GraphManager can apply staggering
(use previous-timestep values for back-edges).
"""

from __future__ import annotations

import heapq
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

    Every node appears exactly once.  If a cycle exists the function still
    returns an ordering: Kahn's algorithm orders every node it can reach
    (each one after all of its sources), and the nodes it cannot -- the
    members of every cycle and everything downstream of one -- follow in
    the order of their strongly connected components.  An edge between two
    different components always points forward in that order, so the only
    edges that point backward (:func:`identify_back_edges`) are edges
    inside a cycle.  Within a cycle, and between components that do not
    depend on each other, the nodes keep their order in *node_names*.
    Callers should use :func:`detect_cycles` separately to decide how to
    handle cycles.

    Before 0.4.0 the unreached nodes were appended in their order in
    *node_names* alone, so a node downstream of a cycle that had been added
    to the graph before the cycle's members was scheduled ahead of them,
    and its input from the cycle -- an edge on no cycle -- was read from
    the previous step.
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

    if len(order) < len(node_names):
        placed = set(order)
        remaining = [n for n in node_names if n not in placed]
        order.extend(_order_by_components(remaining, edges))
    return order


def _order_by_components(nodes: Sequence[str], edges: Sequence[EdgeSpec]) -> list[str]:
    """*nodes* grouped into their strongly connected components, in dependency order.

    A stable topological sort of the condensation, a component at a time:
    repeatedly take, of the components whose sources in *other* components
    are all placed, the one whose first member comes first in *nodes*, and
    place all its members together, in their order in *nodes*.

    Together, because a coupling group runs as one block at its first
    member's place in the schedule: a component placed member by member
    could put a downstream group's first member ahead of an upstream
    group's last, and that block would then run before the group it reads.
    In *nodes* order inside a component, because that is the order a
    Gauss-Seidel sweep of a coupling group follows.  Where *nodes* already
    lists every component whole and after the components it reads -- every
    graph built without a downstream node added ahead of its cycle -- the
    result is *nodes* unchanged.
    """
    position = {n: i for i, n in enumerate(nodes)}
    head = {n: n for n in nodes}
    for scc in find_strongly_connected_components(list(nodes), edges):
        first = min(scc, key=position.__getitem__)
        for n in scc:
            head[n] = first
    members: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        members[head[n]].append(n)
    successors: dict[str, set[str]] = defaultdict(set)
    waiting = {h: 0 for h in members}
    for e in edges:
        src, dst = e.source_node, e.target_node
        if src not in position or dst not in position:
            continue
        a, b = head[src], head[dst]
        if a != b and b not in successors[a]:
            successors[a].add(b)
            waiting[b] += 1
    ready = [position[h] for h in members if waiting[h] == 0]
    heapq.heapify(ready)
    out: list[str] = []
    while ready:
        h = nodes[heapq.heappop(ready)]
        out.extend(members[h])
        for nxt in successors[h]:
            waiting[nxt] -= 1
            if waiting[nxt] == 0:
                heapq.heappush(ready, position[nxt])
    return out


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
