"""Node-to-device assignment for multi-GPU coupling.

Assigns simulation nodes to devices with the goals of:
1. Keeping coupled nodes on the same device (minimise all-gather).
2. Balancing memory usage across devices.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Optional


def assign_nodes_to_devices(
    node_names: list[str],
    edges: list[dict],
    coupling_groups: list[set[str]],
    n_devices: int,
    state_sizes: Optional[dict[str, int]] = None,
) -> dict[str, int]:
    """Assign each node to a device index.

    Parameters
    ----------
    node_names : list[str]
        All simulation node names.
    edges : list[dict]
        Edges with ``source_node`` and ``target_node`` keys.
    coupling_groups : list[set[str]]
        Sets of node names that are in the same coupling group.
    n_devices : int
        Number of available devices.
    state_sizes : dict[str, int], optional
        Estimated state size in bytes per node (for load balancing).

    Returns
    -------
    dict[str, int]
        Mapping from node name to device index.
    """
    assignment: dict[str, int] = {}

    # Phase 1: Assign coupled groups round-robin, keeping each group
    # co-located on one device when possible
    device_loads: dict[int, int] = defaultdict(int)
    sizes = state_sizes or {}

    for i, group in enumerate(coupling_groups):
        # Pick the device with the lightest load
        target_device = min(range(n_devices), key=lambda d: device_loads[d])
        for name in group:
            if name in assignment:
                continue
            assignment[name] = target_device
            device_loads[target_device] += sizes.get(name, 1)

    # Phase 2: Assign remaining (uncoupled) nodes round-robin by load
    for name in node_names:
        if name in assignment:
            continue
        target_device = min(range(n_devices), key=lambda d: device_loads[d])
        assignment[name] = target_device
        device_loads[target_device] += sizes.get(name, 1)

    return assignment


def coupling_colocation_rate(
    assignment: dict[str, int],
    coupling_groups: list[set[str]],
) -> float:
    """Fraction of coupled node pairs assigned to the same device.

    Returns a value in [0, 1].  Higher is better.
    """
    total_pairs = 0
    colocated_pairs = 0

    for group in coupling_groups:
        names = sorted(group)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                total_pairs += 1
                if assignment.get(names[i]) == assignment.get(names[j]):
                    colocated_pairs += 1

    if total_pairs == 0:
        return 1.0
    return colocated_pairs / total_pairs
