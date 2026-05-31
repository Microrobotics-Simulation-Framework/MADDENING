"""Sharded wrappers for :class:`SimulationNode`.

Two flavours:

- :class:`ShardedPointwiseNode` wraps a pointwise node and shards its
  state along a single device-mesh axis.  No halo exchange.
- :class:`ShardedStencilNode` wraps a stencil node (non-empty
  ``halo_width``), pads each state field with halo cells from
  neighbouring shards via :func:`halo_exchange`, calls
  :meth:`SimulationNode.update_padded`, and strips halos from the
  result.

The legacy ``ShardedNode`` alias was removed in v0.3.0 (per the v0.2.x
deprecation cycle).  Use :class:`ShardedPointwiseNode` for pointwise
sharding or :class:`ShardedStencilNode` for stencil sharding.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maddening.cloud.multigpu.halo import halo_exchange
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray, coerce_static_data_value


@stability(StabilityLevel.STABLE)
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


@stability(StabilityLevel.STABLE)
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

        sharded_spatial = set(self._axis_map.values())
        self._replicated_halo_axes = {
            sa: h for sa, h in halo.items() if sa not in sharded_spatial
        }

        # Classify the inner node's static_data: pick out every
        # StaticArray declared with ``replication="shard"`` and validate
        # that its shard_axis lines up with one of the spatial axes this
        # wrapper actually shards.
        self._sharded_static: dict[str, StaticArray] = {}
        for k, v in node.static_data.items():
            v_coerced = coerce_static_data_value(v, node_name=node.name, key=k)
            if isinstance(v_coerced, StaticArray) and v_coerced.replication == "shard":
                if v_coerced.shard_axis not in sharded_spatial:
                    raise ValueError(
                        f"StaticArray {k!r} declares shard_axis="
                        f"{v_coerced.shard_axis} but {type(node).__name__} "
                        f"shards spatial axes {sorted(sharded_spatial)} via "
                        "ShardedStencilNode (per axis_map.values())."
                    )
                self._sharded_static[k] = v_coerced

        # Probe inner.update_padded's signature once.  Nodes ported to
        # v0.2.1 accept `static_padded=` and `shard_info=`; v0.2-era
        # nodes do not.  If the node has sharded statics declared but
        # its signature does not accept `static_padded`, that is a
        # contract violation and we raise here rather than at first
        # trace.
        sig = inspect.signature(node.update_padded)
        params = sig.parameters
        has_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        self._inner_accepts_static_padded = (
            "static_padded" in params or has_var_kw
        )
        self._inner_accepts_shard_info = (
            "shard_info" in params or has_var_kw
        )
        if self._sharded_static and not self._inner_accepts_static_padded:
            raise ValueError(
                f"{type(node).__name__} declares sharded static_data "
                f"({sorted(self._sharded_static)}) but its update_padded "
                "signature does not accept 'static_padded'. Update the "
                "signature to '(self, state_padded, boundary_inputs, dt, "
                "*, static_padded=None, shard_info=None)'."
            )

        # Cache for shard_map-wrapped update functions, keyed by state
        # shape signature.  Built once per shape, then reused so the JAX
        # trace cache hits on every subsequent step.
        self._local_update_fn = self._build_local_update()
        self._sharded_cache: dict[tuple, Any] = {}

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

    def update_padded(
        self,
        state_padded,
        boundary_inputs,
        dt,
        *,
        static_padded=None,
        shard_info=None,
    ):
        kwargs = {}
        if self._inner_accepts_static_padded and static_padded is not None:
            kwargs["static_padded"] = static_padded
        if self._inner_accepts_shard_info and shard_info is not None:
            kwargs["shard_info"] = shard_info
        return self._inner.update_padded(
            state_padded, boundary_inputs, dt, **kwargs
        )

    def domain_integral_fields(self) -> set[str]:
        """Proxy to the wrapped node's declaration."""
        return self._inner.domain_integral_fields()

    # ------------------------------------------------------------------
    # update path
    # ------------------------------------------------------------------

    def _build_local_update(self):
        """Build the stable inner update function (closure on static data).

        The returned function takes ``(local_state, boundary_inputs, dt,
        local_static)`` and is reused across every step -- only its in/out
        specs depend on the input shapes, so the shard_map trace caches
        cleanly.
        """
        inner = self._inner
        mesh = self._mesh
        exchange_axes = self._exchange_axes
        boundary = self._boundary
        replicated_halo_axes = self._replicated_halo_axes
        strip_fn = self._strip_halos
        field_needs_halo = self._field_needs_halo
        sharded_static = self._sharded_static
        axis_map = self._axis_map  # {mesh_axis: spatial_axis}
        halo_widths = inner.halo_width()
        state_set = set(inner.state_fields())
        integrals = set(inner.domain_integral_fields())
        accepts_static_padded = self._inner_accepts_static_padded
        accepts_shard_info = self._inner_accepts_shard_info
        mesh_axis_tup = tuple(mesh.axis_names)

        # Pre-compute per-static halo-exchange descriptors. A sharded
        # static gets halo-exchanged only on the single mesh axis that
        # maps to its shard_axis, with ``boundary="edge"`` (statics
        # don't evolve in time -- periodic wrap would be wrong even when
        # the state uses periodic).
        static_exchange: dict[str, list[tuple[str, int, int]]] = {}
        for k, sa in sharded_static.items():
            descriptors: list[tuple[str, int, int]] = []
            if sa.shard_axis in halo_widths:
                h = halo_widths[sa.shard_axis]
                for ma, mapped_sax in axis_map.items():
                    if mapped_sax == sa.shard_axis:
                        descriptors.append((ma, sa.shard_axis, h))
                        break
            static_exchange[k] = descriptors

        def _pad_replicated(arr):
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

        def _local_update(local_state, local_bi, local_dt, local_static):
            # 1. Halo-pad state.
            padded = {}
            for f, arr in local_state.items():
                arr2 = _pad_replicated(arr)
                if exchange_axes and field_needs_halo(arr):
                    arr2 = halo_exchange(
                        arr2, mesh=mesh, axes=exchange_axes, boundary=boundary,
                    )
                padded[f] = arr2

            # 2. Halo-pad sharded statics (boundary="edge").
            padded_static: dict[str, Any] = {}
            for k, arr in local_static.items():
                descriptors = static_exchange[k]
                if descriptors:
                    padded_static[k] = halo_exchange(
                        arr, mesh=mesh, axes=descriptors, boundary="edge",
                    )
                else:
                    padded_static[k] = arr

            # 3. Compute shard_info: {spatial_axis: (global_offset,
            #    local_extent)} for every spatial axis the node shards.
            #    ``global_offset`` is a traced JAX scalar — usable in
            #    dynamic_slice, not in Python integer slicing.
            shard_info: dict[int, tuple[Any, int]] = {}
            for ma, sax in axis_map.items():
                extent = None
                for arr in local_state.values():
                    if sax < arr.ndim:
                        extent = arr.shape[sax]
                        break
                if extent is None:
                    for arr in local_static.values():
                        if sax < arr.ndim:
                            extent = arr.shape[sax]
                            break
                if extent is None:
                    continue
                offset = lax.axis_index(ma) * extent
                shard_info[sax] = (offset, extent)

            # 4. Dispatch.
            extra_kwargs = {}
            if accepts_static_padded and padded_static:
                extra_kwargs["static_padded"] = padded_static
            if accepts_shard_info and shard_info:
                extra_kwargs["shard_info"] = shard_info
            new_padded = inner.update_padded(
                padded, local_bi, local_dt, **extra_kwargs
            )

            # 5. Classify outputs: state fields → strip halos; declared
            #    integrals → psum across the full mesh; otherwise raise
            #    (the out_specs build below would also catch it).
            out: dict[str, Any] = {}
            for k, v in new_padded.items():
                if k in state_set:
                    out[k] = strip_fn(v, original=local_state[k])
                elif k in integrals:
                    out[k] = lax.psum(v, axis_name=mesh_axis_tup)
                else:
                    raise ValueError(
                        f"{type(inner).__name__}.update_padded returned "
                        f"key {k!r} that is neither in state_fields() "
                        "nor in domain_integral_fields()."
                    )
            return out

        return _local_update

    def _state_signature(self, state: dict) -> tuple:
        return tuple(sorted(
            (f, tuple(arr.shape), str(arr.dtype)) for f, arr in state.items()
        ))

    def _bi_signature(self, boundary_inputs: dict) -> tuple:
        return tuple(sorted(
            (k, tuple(jnp.asarray(v).shape), str(jnp.asarray(v).dtype))
            for k, v in boundary_inputs.items()
        ))

    def _static_signature(self, static: dict) -> tuple:
        return tuple(sorted(
            (k, tuple(a.shape), str(a.dtype)) for k, a in static.items()
        ))

    def _get_sharded_fn(
        self, state: dict, boundary_inputs: dict, static: dict
    ):
        key = (
            self._state_signature(state),
            self._bi_signature(boundary_inputs),
            self._static_signature(static),
            self._inner.static_data_hash(),
        )
        fn = self._sharded_cache.get(key)
        if fn is not None:
            return fn

        state_specs = {f: self._spec_for_field(arr) for f, arr in state.items()}
        bi_specs = {k: P() for k in boundary_inputs}
        static_specs = {
            k: self._spec_for_static_key(k, arr)
            for k, arr in static.items()
        }
        # Outputs: state fields keep their per-shard specs (halos are
        # stripped to original shape); declared domain integrals are
        # replicated after lax.psum.  Anything else would have raised
        # inside _local_update; we leave the out_specs key absent here
        # so shard_map's pytree consistency check catches it too.
        out_specs = dict(state_specs)
        for k in self._inner.domain_integral_fields():
            out_specs[k] = P()

        sm = shard_map(
            self._local_update_fn,
            mesh=self._mesh,
            in_specs=(state_specs, bi_specs, P(), static_specs),
            out_specs=out_specs,
            check_rep=False,
        )
        # Bare shard_map outside jit incurs ~250ms/call of Python dispatch
        # overhead on CPU; wrapping it in jit reduces that to microseconds.
        # When ShardedStencilNode is used inside GraphManager's jitted step
        # function the outer jit would absorb this anyway, but the eager
        # path (standalone .update calls in tests) needs the explicit jit.
        fn = jax.jit(sm)
        self._sharded_cache[key] = fn
        return fn

    def _materialise_sharded_statics(self) -> dict:
        """Per-device materialisation of every sharded StaticArray.

        Each StaticArray with ``replication="shard"`` is placed onto
        the device mesh via ``jax.device_put`` + ``NamedSharding`` whose
        PartitionSpec puts the matching mesh-axis at the array's
        ``shard_axis``.  This is the "3a materialisation" step from the
        v0.2.1 plan -- v0.2.0 only stored ``shard_axis`` as metadata.
        """
        out: dict[str, jax.Array] = {}
        for k, sa in self._sharded_static.items():
            arr = jnp.asarray(sa.value)
            spec = self._spec_for_static_key(k, arr)
            sharding = NamedSharding(self._mesh, spec)
            out[k] = jax.device_put(arr, sharding)
        return out

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Halo-pad every state field, call ``update_padded``, strip halos.

        Sharded spatial axes get halo cells from neighbour shards via
        :func:`halo_exchange`.  Spatial axes with halo but no sharding
        (replicated axes) get their halos filled locally according to
        ``boundary``.  This lets a node with halo on every axis run
        under a partial-pencil mesh without the stencil needing to
        know which axes are sharded.

        Sharded static arrays declared by the inner node (via
        :class:`~maddening.core.static_data.StaticArray` with
        ``replication="shard"``) are materialised per-device and
        halo-exchanged with ``boundary="edge"`` before being passed
        through as ``static_padded`` to :meth:`update_padded`.

        The shard_map wrapper is built once per
        ``(state_shape, bi_shape, static_shape, static_data_hash)``
        signature and cached so repeated steps hit JAX's compile cache.
        """
        static_materialised = self._materialise_sharded_statics()
        fn = self._get_sharded_fn(state, boundary_inputs, static_materialised)
        return fn(
            state,
            boundary_inputs,
            jnp.asarray(dt, dtype=jnp.float32),
            static_materialised,
        )

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

    def _spec_for_static_key(self, key: str, arr: jax.Array) -> P:
        """PartitionSpec for a sharded StaticArray.

        Only the array's own ``shard_axis`` gets a mesh-axis assignment;
        every other axis is replicated.  Picks the (unique) mesh axis
        whose ``axis_map`` entry maps to that spatial axis.
        """
        sa = self._sharded_static[key]
        spec: list[Optional[str]] = [None] * arr.ndim
        for mesh_axis, spatial_axis in self._axis_map.items():
            if spatial_axis == sa.shard_axis and spatial_axis < arr.ndim:
                spec[spatial_axis] = mesh_axis
                break
        return P(*spec)

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
