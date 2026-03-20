"""ShardedNode — data-parallel wrapper for large simulation nodes.

Wraps a ``SimulationNode`` to shard its state arrays across a JAX
device mesh.  Each device holds a slice of the state along
``shard_axis``, and ``update()`` runs under the mesh with each device
computing its shard independently.

Only nodes where ``requires_halo`` is ``False`` (pointwise operations)
can be sharded.  Stencil-based nodes (FD, LBM) require halo exchange
which is not yet implemented.
"""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maddening.core.node import SimulationNode


class ShardedNode(SimulationNode):
    """Data-parallel wrapper that shards a node's state across devices.

    Parameters
    ----------
    node : SimulationNode
        The node to wrap.
    mesh : Mesh
        JAX device mesh (1-D, axis name ``"devices"``).
    shard_axes : int or tuple[int, ...]
        Which axes of the state arrays to shard.  A single int is
        converted to a 1-tuple.  Only single-axis sharding is
        implemented currently; multi-axis is accepted and stored
        for future use but raises ``NotImplementedError`` if
        ``len(shard_axes) > 1``.

    Raises
    ------
    ValueError
        If the node's ``requires_halo`` property is ``True``.
    NotImplementedError
        If ``len(shard_axes) > 1`` (multi-axis sharding not yet
        implemented).
    """

    def __init__(
        self,
        node: SimulationNode,
        mesh: Mesh,
        shard_axes: int | tuple[int, ...] = (0,),
    ) -> None:
        if node.requires_halo:
            raise ValueError(
                f"{type(node).__name__} requires halo exchange for "
                f"spatial neighbor access, which is not yet supported "
                f"by ShardedNode. Only pointwise nodes (requires_halo=False) "
                f"can be sharded."
            )

        if isinstance(shard_axes, int):
            shard_axes = (shard_axes,)

        if len(shard_axes) > 1:
            raise NotImplementedError(
                f"Multi-axis sharding (axes={shard_axes}) is not yet "
                f"implemented. Use a single shard axis for now."
            )

        # Initialize the SimulationNode base
        super().__init__(
            name=node.name,
            timestep=node.delta_t,
            **node.params,
        )
        self._inner = node
        self._mesh = mesh
        self._shard_axes = shard_axes
        self._sharding = NamedSharding(mesh, P("devices"))

    @property
    def requires_halo(self) -> bool:
        """ShardedNode only wraps pointwise nodes."""
        return False

    def initial_state(self) -> dict:
        """Return sharded initial state."""
        state = self._inner.initial_state()
        sharded = {}
        for field, arr in state.items():
            if arr.ndim > self._shard_axes[0]:
                # Shard along the specified axis
                sharded[field] = jax.device_put(arr, self._sharding)
            else:
                # Scalar or too-small array — replicate
                sharded[field] = arr
        return sharded

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Run the inner node's update with sharded arrays."""
        return self._inner.update(state, boundary_inputs, dt)

    def state_fields(self) -> list[str]:
        return self._inner.state_fields()

    def boundary_input_spec(self):
        return self._inner.boundary_input_spec()

    def to_dict(self) -> dict:
        d = self._inner.to_dict() if hasattr(self._inner, 'to_dict') else {}
        d["sharded"] = True
        d["shard_axes"] = self._shard_axes
        return d
