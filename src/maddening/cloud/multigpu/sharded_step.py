"""Per-device Jacobi coupling pass.

Each node in a Jacobi coupling group is placed on its assigned device
(via :func:`jax.device_put` upstream) and computed locally on that
device.  All nodes read from the previous-iteration ``latest_results``
(the Jacobi invariant), so the work is naturally parallel across
devices once the placement is in place.

This is *not* a ``shard_map``-based distributed implementation -- it
relies on JAX's runtime scheduler to run device-placed work in
parallel.  Stencil-node sharding within a coupling group (where one
node's state spans multiple devices) is provided by
:class:`ShardedStencilNode` instead.
"""

from __future__ import annotations

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def build_sharded_jacobi_pass(
    group_node_names: list[str],
    update_fns: dict[str, Callable],
    resolve_boundary_fn: Callable,
    device_map: dict[str, int],
    mesh: Mesh,
    get_dt_fn: Callable[[str], Any],
) -> Callable:
    """Build a sharded Jacobi pass function.

    Parameters
    ----------
    group_node_names : list[str]
        Node names in the coupling group.
    update_fns : dict[str, Callable]
        Per-node update functions ``(state, boundary, dt) -> new_state``.
    resolve_boundary_fn : Callable
        ``(node_name, full_state) -> boundary_inputs`` — resolves edges.
    device_map : dict[str, int]
        Node name -> device index mapping.
    mesh : Mesh
        JAX device mesh.
    get_dt_fn : Callable
        ``(node_name) -> dt`` — returns the timestep for a node.

    Returns
    -------
    Callable
        ``(initial_node_states, latest_results) -> updated_results``
        A Jacobi pass that computes all node updates from frozen state.
    """
    n_devices = len(mesh.devices)

    # Group nodes by device
    device_nodes: dict[int, list[str]] = {d: [] for d in range(n_devices)}
    for name in group_node_names:
        dev = device_map.get(name, 0)
        device_nodes[dev].append(name)

    def sharded_jacobi_pass(initial_node_states, latest_results):
        """Execute one Jacobi pass with device-placed per-node updates.

        Each node reads ``latest_results`` (frozen previous iteration)
        from wherever those arrays live; JAX collects what's needed and
        runs the local update on the node's assigned device.
        """
        results = {}

        # Each device computes its assigned nodes
        for dev_id in range(n_devices):
            for nn in device_nodes[dev_id]:
                # Resolve boundary from frozen state (all-gather equivalent)
                bi = resolve_boundary_fn(nn, latest_results)
                pre = initial_node_states[nn]
                dt = get_dt_fn(nn)
                results[nn] = update_fns[nn](pre, bi, dt)

        # Merge: start from latest_results, overwrite with new values
        s = {k: v for k, v in latest_results.items()}
        for nn in group_node_names:
            s[nn] = results[nn]

        return s

    return sharded_jacobi_pass


def shard_state(
    state: dict[str, dict],
    device_map: dict[str, int],
    mesh: Mesh,
) -> dict[str, dict]:
    """Place state arrays on their assigned devices.

    Parameters
    ----------
    state : dict[str, dict]
        Full simulation state ``{node: {field: array}}``.
    device_map : dict[str, int]
        Node name -> device index.
    mesh : Mesh
        JAX device mesh.

    Returns
    -------
    Sharded state with arrays placed on the correct devices.
    """
    sharded = {}
    devices = mesh.devices

    for node_name, node_state in state.items():
        dev_idx = device_map.get(node_name, 0)
        device = devices[min(dev_idx, len(devices) - 1)]
        sharded[node_name] = {
            field: jax.device_put(arr, device)
            for field, arr in node_state.items()
        }

    return sharded
