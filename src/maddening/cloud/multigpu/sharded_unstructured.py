"""Graph-partitioned sharding wrapper — :class:`ShardedUnstructuredNode`.

v0.3.0 §A6 substrate.  Sibling of
:class:`maddening.cloud.multigpu.sharded_node.ShardedStencilNode`.
They share the substrate (state dict signature, halo exchange call
inside ``shard_map``, output classification via state_fields /
domain_integral_fields) but differ in what "halo" means:

* Cartesian sharded nodes: halo of width K along each spatial axis,
  obtained from the two neighbour shards along that axis.
* Unstructured sharded nodes: halo is the set of ghost cells — cells
  on another shard that this shard's stencil reads — fetched via a
  sparse :func:`exchange_unstructured` collective.

Class hierarchy (per ``plans/MADDENING_v0.3.0_PLAN.md`` §A6):

.. code-block:: text

    ShardedNode (Protocol — surface contract only)
    ├── ShardedPointwiseNode   (no halo, no spatial topology)         v0.2.0
    ├── ShardedStencilNode     (Cartesian, axis-aligned halos)        v0.2.1
    └── ShardedUnstructuredNode (graph-partition, sparse halos)       v0.3.0 (this)

v0.4.0 commitment
-----------------
MIME's ``FVMFluidNode`` will subclass this in v0.4.0.  The
constructor signature + ``update_padded`` plumbing + output
classification + partition-assignment handoff documented here is
``@stability(stable)``-ready: v0.4.0 hardens the implementation
(production-grade sparse halo exchange, real-mesh-size testing,
NCCL fast-path), it does not redesign the surface.

If a real design flaw emerges during v0.3.0 implementation, fix it
in v0.3.0 — surface a breaking change here, not in v0.4.0.
"""

from __future__ import annotations

import inspect
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maddening.cloud.multigpu.halo_unstructured import (
    UnstructuredPartitionLayout,
    exchange_unstructured,
    partition_value,
    gather_value,
)
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray


@stability(StabilityLevel.STABLE)
class ShardedUnstructuredNode(SimulationNode):
    """Sharded wrapper for a graph-partitioned :class:`SimulationNode`.

    The wrapped ``node`` exposes ``update_padded(state_padded,
    boundary_inputs, dt, *, static_padded=None, shard_info=None)``
    just as a stencil-sharded inner node does.  The difference is the
    layout of the padded arrays:

    * ``state_padded[<field>]`` is a 1-D-or-higher array whose first
      axis has length ``n_local_max + n_ghost_max``.  The first
      ``n_local_max`` slots are the shard's owned cells (in the order
      declared by the partition layout's ``local_global_ids[d]``); the
      remaining slots are ghost cells in the order declared by
      ``ghost_global_ids[d]``.
    * ``static_padded[<key>]`` follows the same layout when
      ``StaticArray(replication="partition")`` is declared.
    * ``shard_info`` is ``{0: (offset, n_local_max)}`` where ``offset``
      is a *traced* ``lax.axis_index(mesh_axis) * n_local_max`` —
      i.e. a JAX scalar.  Unlike the Cartesian case, the offset isn't
      a geometric coordinate (cells aren't contiguous in global ID
      space), but it's exposed for symmetry with
      :class:`ShardedStencilNode` and for nodes that want a unique
      per-shard tag.

    Output classification follows the same rules as
    :class:`ShardedStencilNode`:

    * Keys in ``inner.state_fields()``: padding is stripped (only the
      first ``n_local_max`` slots are returned).
    * Keys in ``inner.domain_integral_fields()``: ``lax.psum`` across
      the mesh axis (the partial sums from each shard are summed).
    * Any other key raises ``ValueError`` — we cannot infer a
      partition spec for unknown outputs.

    Parameters
    ----------
    node : SimulationNode
        The inner node.  Must implement ``update_padded`` with the
        signature above.  ``node.halo_width()`` is ignored for
        unstructured sharding — the halo is determined by the
        partition layout, not by an axis-aligned halo width.
    mesh : Mesh
        1-D JAX device mesh.
    layout : UnstructuredPartitionLayout
        Pre-computed partition / ghost / send-recv tables.
    mesh_axis : str, optional
        The name of the mesh axis to shard along.  Defaults to
        ``"devices"``.  Must match ``mesh.axis_names``.

    Notes
    -----
    The partition-assignment handoff pattern (PyMetis → layout → here)
    is part of the v0.3.0 contract.  See
    ``docs/developer_guide/sharding_topology.md`` for the developer-
    facing description.
    """

    def __init__(
        self,
        node: SimulationNode,
        mesh: Mesh,
        layout: UnstructuredPartitionLayout,
        *,
        mesh_axis: str = "devices",
    ) -> None:
        if mesh_axis not in mesh.axis_names:
            raise ValueError(
                f"ShardedUnstructuredNode: mesh_axis={mesh_axis!r} not in "
                f"mesh.axis_names={mesh.axis_names}"
            )
        mesh_size = int(mesh.shape[mesh_axis])
        if mesh_size != layout.n_devices:
            raise ValueError(
                f"ShardedUnstructuredNode: mesh axis {mesh_axis!r} has size "
                f"{mesh_size} but layout.n_devices={layout.n_devices}"
            )

        # Verify any StaticArray(replication="partition") on the inner
        # node uses a partition_assignment compatible with the layout.
        sharded_static: dict[str, StaticArray] = {}
        for k, v in node.static_data.items():
            if isinstance(v, StaticArray) and v.replication == "partition":
                # All partition_assignments must agree on the layout's
                # one — otherwise we'd need a different layout per array.
                pa = v.partition_assignment
                ref_pa = layout.partition_assignment
                if pa.shape != ref_pa.shape:
                    raise ValueError(
                        f"ShardedUnstructuredNode: StaticArray {k!r} has "
                        f"partition_assignment.shape={pa.shape} but layout "
                        f"was built with shape {ref_pa.shape}."
                    )
                # Element-wise equality (numpy-style — pa might be a
                # numpy or jax array).  The layout claims authority.
                import numpy as np  # noqa: PLC0415
                if not np.array_equal(np.asarray(pa), np.asarray(ref_pa)):
                    raise ValueError(
                        f"ShardedUnstructuredNode: StaticArray {k!r} has a "
                        "partition_assignment that disagrees with the "
                        "layout's.  All partitioned static arrays on a "
                        "node must share one assignment (build the layout "
                        "from that one assignment).",
                    )
                sharded_static[k] = v

        super().__init__(name=node.name, timestep=node.delta_t, **node.params)
        self._inner = node
        self._mesh = mesh
        self._mesh_axis = mesh_axis
        self._layout = layout
        self._sharded_static = sharded_static
        # Cached compiled fns keyed by the input signature.
        self._sharded_cache: dict[Any, Any] = {}

    # -----------------------------------------------------------------
    # Layout accessor — handoff for the experiment-setup contract.
    # -----------------------------------------------------------------
    @property
    def layout(self) -> UnstructuredPartitionLayout:
        """The partition layout used to shard the inner node."""
        return self._layout

    # -----------------------------------------------------------------
    # SimulationNode plumbing
    # -----------------------------------------------------------------
    def halo_width(self) -> dict[int, int]:
        """Unstructured sharding has no axis-aligned halo width."""
        return {}

    def state_fields(self) -> list[str]:
        return self._inner.state_fields()

    def boundary_input_spec(self):
        return self._inner.boundary_input_spec()

    def initial_state(self) -> dict:
        """Materialise the inner node's initial state onto the mesh.

        The inner node produces a global initial state (first axis =
        global cell axis).  We slice into (n_devices, n_local_max, *)
        and place on the mesh via ``NamedSharding``.
        """
        global_state = self._inner.initial_state()
        sharded = {}
        sharding = NamedSharding(self._mesh, P(self._mesh_axis))
        for k, arr in global_state.items():
            arr = jnp.asarray(arr)
            per_shard = partition_value(
                value=jax.device_get(arr), layout=self._layout,
            )
            sharded[k] = jax.device_put(jnp.asarray(per_shard.reshape(
                (self._layout.n_devices * self._layout.n_local_max,)
                + per_shard.shape[2:]
            )), sharding)
        return sharded

    def gather_global(self, sharded_state: dict) -> dict:
        """Inverse of :meth:`initial_state` for test / driver use.

        Strips per-shard padding and rebuilds the global state dict.
        Domain-integral fields (the ones declared via
        :meth:`SimulationNode.domain_integral_fields`) are already
        fully replicated after the ``psum`` and are passed through
        unchanged.  Not on the inner node's signature — this is an
        unstructured-sharding utility.
        """
        out = {}
        state_set = set(self._inner.state_fields())
        for k, arr in sharded_state.items():
            host = jax.device_get(arr)
            if k in state_set:
                per_shard = host.reshape(
                    (self._layout.n_devices, self._layout.n_local_max)
                    + host.shape[1:]
                )
                out[k] = gather_value(per_shard=per_shard, layout=self._layout)
            else:
                # Domain-integral (or otherwise replicated) output —
                # pass through.
                out[k] = host
        return out

    # -----------------------------------------------------------------
    # Pure-Python update plumbing (for tests / single-shard verification)
    # -----------------------------------------------------------------
    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Run one step under sharding.

        The wrapper compiles a shard_map'd implementation per
        (state_signature, bi_signature, static_signature) tuple and
        dispatches.
        """
        fn = self._get_sharded_fn(state, boundary_inputs)
        # Collect the per-partition static arrays from the inner node.
        static_partitioned = {}
        for k, sa in self._sharded_static.items():
            host = jax.device_get(sa.value) if hasattr(sa.value, "device") \
                else sa.value
            per_shard = partition_value(value=host, layout=self._layout)
            sharding = NamedSharding(self._mesh, P(self._mesh_axis))
            static_partitioned[k] = jax.device_put(
                jnp.asarray(per_shard.reshape(
                    (self._layout.n_devices * self._layout.n_local_max,)
                    + per_shard.shape[2:]
                )), sharding,
            )
        return fn(state, boundary_inputs, jnp.asarray(dt), static_partitioned)

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _get_sharded_fn(self, state, boundary_inputs):
        key = (
            tuple(sorted((k, tuple(a.shape), str(a.dtype))
                         for k, a in state.items())),
            tuple(sorted((k, tuple(a.shape) if hasattr(a, "shape") else (),
                          str(a.dtype) if hasattr(a, "dtype") else type(a).__name__)
                         for k, a in boundary_inputs.items())),
            tuple(sorted((k, tuple(self._sharded_static[k].value.shape),
                          str(self._sharded_static[k].value.dtype))
                         for k in self._sharded_static)),
            self._inner.static_data_hash(),
        )
        cached = self._sharded_cache.get(key)
        if cached is not None:
            return cached

        state_specs = {k: P(self._mesh_axis) for k in state}
        bi_specs = {k: P() for k in boundary_inputs}
        static_specs = {k: P(self._mesh_axis) for k in self._sharded_static}
        out_specs = {**state_specs}
        for k in self._inner.domain_integral_fields():
            out_specs[k] = P()  # fully replicated after psum

        local_fn = self._build_local_update()

        sm = shard_map(
            local_fn,
            mesh=self._mesh,
            in_specs=(state_specs, bi_specs, P(), static_specs),
            out_specs=out_specs,
            check_rep=False,
        )
        fn = jax.jit(sm)
        self._sharded_cache[key] = fn
        return fn

    def _build_local_update(self):
        inner = self._inner
        layout = self._layout
        mesh_axis = self._mesh_axis
        n_local_max = layout.n_local_max
        state_set = set(inner.state_fields())
        integrals = set(inner.domain_integral_fields())

        def _local_update(local_state, local_bi, local_dt, local_static):
            # Strip the leading device dimension that shard_map already
            # collapsed for us — local arrays now have shape (n_local_max, *).

            # 1. Halo-exchange each state field.
            padded_state = {}
            for k, arr in local_state.items():
                padded_state[k] = exchange_unstructured(
                    arr, layout=layout, mesh_axis=mesh_axis,
                )

            # 2. Halo-exchange each partitioned static.
            padded_static = {}
            for k, arr in local_static.items():
                padded_static[k] = exchange_unstructured(
                    arr, layout=layout, mesh_axis=mesh_axis,
                )

            # 3. shard_info — traced offset for nodes that want a per-shard tag.
            idx = lax.axis_index(mesh_axis)
            shard_info = {0: (idx * n_local_max, n_local_max)}

            # 4. Dispatch.
            new = inner.update_padded(
                padded_state, local_bi, local_dt,
                static_padded=(padded_static or None),
                shard_info=shard_info,
            )

            # 5. Classify outputs.
            out = {}
            for k, v in new.items():
                if k in state_set:
                    # Strip the ghost tail.
                    out[k] = v[:n_local_max]
                elif k in integrals:
                    out[k] = lax.psum(v, axis_name=mesh_axis)
                else:
                    raise ValueError(
                        f"{type(inner).__name__}.update_padded returned "
                        f"key {k!r} that is neither in state_fields() nor "
                        "in domain_integral_fields().",
                    )
            return out

        return _local_update

    def to_dict(self) -> dict:
        d = self._inner.to_dict() if hasattr(self._inner, "to_dict") else {}
        d["sharded"] = True
        d["sharding"] = "unstructured"
        d["n_devices"] = self._layout.n_devices
        return d


__all__ = [
    "ShardedUnstructuredNode",
]
