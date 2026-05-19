"""Sharded wrappers for :class:`SimulationNode`.

Two flavours:

- :class:`ShardedPointwiseNode` wraps a pointwise node and shards its
  state along a single device-mesh axis.  No halo exchange.
- :class:`ShardedStencilNode` wraps a stencil node (non-empty
  ``halo_width``), pads each state field with halo cells from
  neighbouring shards via :func:`halo_exchange`, calls
  :meth:`SimulationNode.update_padded`, and strips halos from the
  result.

``ShardedNode`` is retained as a deprecated alias for
``ShardedPointwiseNode`` (v0.1 compatibility).
"""

from __future__ import annotations

import warnings
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maddening.cloud.multigpu.halo import halo_exchange
from maddening.core.node import SimulationNode


class ShardedPointwiseNode(SimulationNode):
    """Data-parallel wrapper for a pointwise :class:`SimulationNode`.

    Only nodes with empty ``halo_width()`` can be wrapped; stencil nodes
    must use :class:`ShardedStencilNode`.

    Parameters
    ----------
    node : SimulationNode
        The pointwise node to wrap.
    mesh : Mesh
        JAX device mesh (1-D, axis name ``"devices"``).
    shard_axes : int or tuple[int, ...]
        Which axes of the state arrays to shard.  Only single-axis
        sharding is implemented; multi-axis raises
        :class:`NotImplementedError`.
    """

    def __init__(
        self,
        node: SimulationNode,
        mesh: Mesh,
        shard_axes: int | tuple[int, ...] = (0,),
    ) -> None:
        if node.halo_width():
            raise ValueError(
                f"{type(node).__name__} declares halo_width="
                f"{node.halo_width()} -- use ShardedStencilNode for "
                "stencil sharding.  ShardedPointwiseNode only wraps "
                "pointwise nodes (halo_width() == {})."
            )

        if isinstance(shard_axes, int):
            shard_axes = (shard_axes,)

        if len(shard_axes) > 1:
            raise NotImplementedError(
                f"Multi-axis pointwise sharding (axes={shard_axes}) is "
                "not yet implemented. Use a single shard axis."
            )

        super().__init__(name=node.name, timestep=node.delta_t, **node.params)
        self._inner = node
        self._mesh = mesh
        self._shard_axes = shard_axes
        self._sharding = NamedSharding(mesh, P("devices"))

    def halo_width(self) -> dict[int, int]:
        """ShardedPointwiseNode only wraps pointwise nodes (no halo)."""
        return {}

    def initial_state(self) -> dict:
        state = self._inner.initial_state()
        sharded = {}
        for field, arr in state.items():
            if arr.ndim > self._shard_axes[0]:
                sharded[field] = jax.device_put(arr, self._sharding)
            else:
                sharded[field] = arr
        return sharded

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        return self._inner.update(state, boundary_inputs, dt)

    def state_fields(self) -> list[str]:
        return self._inner.state_fields()

    def boundary_input_spec(self):
        return self._inner.boundary_input_spec()

    def to_dict(self) -> dict:
        d = self._inner.to_dict() if hasattr(self._inner, "to_dict") else {}
        d["sharded"] = True
        d["shard_axes"] = self._shard_axes
        return d


class ShardedStencilNode(SimulationNode):
    """Pencil-decomposition wrapper for a stencil :class:`SimulationNode`.

    On each step, every state field listed in the node's
    ``halo_width()`` is halo-exchanged along the relevant mesh axis,
    :meth:`SimulationNode.update_padded` is called on the padded state,
    and halos are stripped before the result is returned.

    Parameters
    ----------
    node : SimulationNode
        The stencil node to wrap.  Must override ``update_padded`` and
        declare a non-empty ``halo_width()``.
    mesh : Mesh
        JAX device mesh whose axes match ``axis_map``.
    axis_map : dict[str, int]
        Maps each mesh axis name to the spatial axis index of the
        node's state arrays it shards.  Example for a 3-D LBM under a
        2-D pencil mesh::

            {"spatial_y": 1, "spatial_z": 2}

        Spatial axes not appearing as values of ``axis_map`` are
        replicated on every device; their halos do not need exchange.
    boundary : str
        Boundary mode for halo exchange (``"periodic"``, ``"edge"``,
        or ``"zero"``).  Default ``"edge"`` -- replicate own edge.
        ``update_padded`` applies the physical BCs after the exchange.
    """

    def __init__(
        self,
        node: SimulationNode,
        mesh: Mesh,
        axis_map: dict[str, int],
        boundary: str = "edge",
    ) -> None:
        halo = node.halo_width()
        if not halo:
            raise ValueError(
                f"{type(node).__name__} has empty halo_width() -- use "
                "ShardedPointwiseNode for pointwise sharding."
            )

        # Validate axis_map keys against the mesh and warn on covered axes
        for mesh_axis in axis_map:
            if mesh_axis not in mesh.axis_names:
                raise ValueError(
                    f"axis_map references mesh axis {mesh_axis!r} not "
                    f"present in mesh.axis_names={mesh.axis_names}"
                )

        # Every spatial axis the node sharding covers must have a declared
        # halo width (else it would not be a stencil axis).
        for spatial_axis in axis_map.values():
            if spatial_axis not in halo:
                raise ValueError(
                    f"axis_map shards spatial axis {spatial_axis} but "
                    f"node {type(node).__name__} reports halo_width="
                    f"{halo} with no entry for that axis."
                )

        super().__init__(name=node.name, timestep=node.delta_t, **node.params)
        self._inner = node
        self._mesh = mesh
        self._axis_map = dict(axis_map)
        self._boundary = boundary

        # axes for halo_exchange: list of (mesh_axis, spatial_axis, halo)
        self._exchange_axes: list[tuple[str, int, int]] = [
            (ma, sa, halo[sa]) for ma, sa in self._axis_map.items()
        ]

    def halo_width(self) -> dict[int, int]:
        """Same as the wrapped node -- sharding does not change the stencil."""
        return self._inner.halo_width()

    def initial_state(self) -> dict:
        state = self._inner.initial_state()
        return {
            field: jax.device_put(arr, self._sharding_for_field(arr))
            for field, arr in state.items()
        }

    def state_fields(self) -> list[str]:
        return self._inner.state_fields()

    def boundary_input_spec(self):
        return self._inner.boundary_input_spec()

    def update_padded(self, state_padded, boundary_inputs, dt):
        return self._inner.update_padded(state_padded, boundary_inputs, dt)

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Halo-pad every state field, call ``update_padded``, strip halos.

        Sharded spatial axes get halo cells from neighbour shards via
        :func:`halo_exchange`.  Spatial axes with halo but no sharding
        (replicated axes) get their halos filled locally according to
        ``boundary``.  This lets a node with halo on every axis run
        under a partial-pencil mesh without the stencil needing to
        know which axes are sharded.
        """
        in_state_specs = {f: self._spec_for_field(arr) for f, arr in state.items()}

        inner = self._inner
        mesh = self._mesh
        halo_widths = inner.halo_width()
        sharded_spatial = set(self._axis_map.values())
        replicated_halo_axes = {
            sa: h for sa, h in halo_widths.items() if sa not in sharded_spatial
        }
        exchange_axes = self._exchange_axes
        boundary = self._boundary

        def _pad_replicated(arr: jax.Array) -> jax.Array:
            out = arr
            for sa in sorted(replicated_halo_axes):
                h = int(replicated_halo_axes[sa])
                if h == 0 or sa >= out.ndim:
                    continue
                n = out.shape[sa]
                left = jax.lax.slice_in_dim(out, 0, h, axis=sa)
                right = jax.lax.slice_in_dim(out, n - h, n, axis=sa)
                if boundary == "periodic":
                    left_halo, right_halo = right, left
                elif boundary == "edge":
                    left_halo, right_halo = left, right
                else:  # zero
                    left_halo = jnp.zeros_like(left)
                    right_halo = jnp.zeros_like(right)
                out = jnp.concatenate([left_halo, out, right_halo], axis=sa)
            return out

        def _local_update(local_state):
            padded = {}
            for f, arr in local_state.items():
                arr2 = _pad_replicated(arr)
                if exchange_axes and self._field_needs_halo(arr):
                    arr2 = halo_exchange(
                        arr2, mesh=mesh, axes=exchange_axes, boundary=boundary,
                    )
                padded[f] = arr2
            new_padded = inner.update_padded(padded, boundary_inputs, dt)
            return {f: self._strip_halos(arr, original=local_state[f])
                    for f, arr in new_padded.items()}

        sharded_fn = shard_map(
            _local_update,
            mesh=self._mesh,
            in_specs=(in_state_specs,),
            out_specs=in_state_specs,
            check_rep=False,
        )
        return sharded_fn(state)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _spec_for_field(self, arr: jax.Array) -> P:
        """PartitionSpec for a single state field array.

        Only the spatial axes referenced by ``axis_map`` are sharded;
        all others are replicated.
        """
        spec: list[Optional[str]] = [None] * arr.ndim
        for mesh_axis, spatial_axis in self._axis_map.items():
            if spatial_axis < arr.ndim:
                spec[spatial_axis] = mesh_axis
        return P(*spec)

    def _sharding_for_field(self, arr: jax.Array) -> NamedSharding:
        return NamedSharding(self._mesh, self._spec_for_field(arr))

    def _field_needs_halo(self, arr: jax.Array) -> bool:
        """True if this field has at least one sharded spatial axis."""
        for spatial_axis in self._axis_map.values():
            if spatial_axis < arr.ndim:
                return True
        return False

    def _strip_halos(self, arr: jax.Array, *, original: jax.Array) -> jax.Array:
        """Strip every halo axis (sharded **and** replicated)."""
        halo = self._inner.halo_width()
        out = arr
        for spatial_axis in sorted(halo):
            if spatial_axis >= out.ndim:
                continue
            h = int(halo[spatial_axis])
            if h == 0:
                continue
            target = original.shape[spatial_axis]
            out = jax.lax.slice_in_dim(out, h, h + target, axis=spatial_axis)
        return out

    def to_dict(self) -> dict:
        d = self._inner.to_dict() if hasattr(self._inner, "to_dict") else {}
        d["sharded"] = True
        d["sharded_stencil"] = True
        d["axis_map"] = self._axis_map
        d["boundary"] = self._boundary
        return d


# ---------------------------------------------------------------------------
# Deprecated alias
# ---------------------------------------------------------------------------


class ShardedNode(ShardedPointwiseNode):
    """Deprecated alias for :class:`ShardedPointwiseNode`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "ShardedNode is deprecated; use ShardedPointwiseNode for "
            "pointwise sharding or ShardedStencilNode for stencil "
            "(halo-aware) sharding.  Removed in v0.3.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)
