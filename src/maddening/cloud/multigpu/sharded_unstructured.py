"""Graph-partitioned sharding wrapper — :class:`ShardedUnstructuredNode`.

Added in v0.3.0.  Sibling of
:class:`maddening.cloud.multigpu.sharded_node.ShardedStencilNode`.
They share the substrate (state dict signature, halo exchange call
inside ``shard_map``, output classification via state_fields /
domain_integral_fields) but differ in what "halo" means:

* Cartesian sharded nodes: halo of width K along each spatial axis,
  obtained from the two neighbour shards along that axis.
* Unstructured sharded nodes: halo is the set of ghost cells — cells
  on another shard that this shard's stencil reads — fetched via a
  sparse :func:`exchange_unstructured` collective.

Class hierarchy:

.. code-block:: text

    ShardedNode (Protocol — surface contract only)
    ├── ShardedPointwiseNode   (no halo, no spatial topology)         v0.2.0
    ├── ShardedStencilNode     (Cartesian, axis-aligned halos)        v0.2.1
    └── ShardedUnstructuredNode (graph-partition, sparse halos)       v0.3.0 (this)

What 0.4.0 does
---------------
The constructor signature, ``update_padded`` plumbing, output
classification and partition-assignment handoff documented here have
been ``@stability(STABLE)`` since v0.3.0, and 0.4.0 kept them.  It
added the per-neighbour ``exchange="ppermute"`` transport (bit-identical
to the default ``all_to_all``) and sharded per-cell boundary inputs.  It
did not validate the path on real multi-GPU hardware: the default
transport stays ``all_to_all`` until NCCL timings justify a change, and
the path is tested on CPU virtual devices only
(``docs/developer_guide/sharding_topology.md``).
"""

from __future__ import annotations

import functools
import warnings
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maddening.cloud.multigpu.sharded_node import _params_signature
from maddening.cloud.multigpu.halo_unstructured import (
    UnstructuredPartitionLayout,
    exchange_unstructured,
    partition_value,
    gather_value,
)
from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.node import SimulationNode, _method_accepts_params
from maddening.core.static_data import StaticArray


def _partition_statics(static_data) -> dict[str, StaticArray]:
    """The ``replication="partition"`` entries of a ``static_data`` dict."""
    return {
        k: v for k, v in static_data.items()
        if isinstance(v, StaticArray) and v.replication == "partition"
    }


def _reads_partition_layout(node) -> bool:
    """Does ``node`` say its ``update_padded`` reads the partition layout?

    The private, provisional opt-in for a node that declares a Cartesian
    ``halo_width()`` (so that :class:`ShardedStencilNode` can wrap it)
    *and* whose ``update_padded`` also reads ``[owned cells | ghost
    cells]``: a class attribute or a method ``_reads_partition_layout``
    that is (or returns) true.  Not part of the node contract: an
    internal convention, like ``_halo_boundary_hint``, until a release
    has shown it earns a public name.  A node written only for this
    wrapper needs none of it -- it declares ``halo_width() == {}``.
    """
    hook = getattr(node, "_reads_partition_layout", None)
    if hook is None:
        return False
    return bool(hook() if callable(hook) else hook)


def _validate_partition_statics(
    sharded: dict[str, StaticArray], layout: UnstructuredPartitionLayout,
) -> None:
    """Every partitioned static must agree with the layout's assignment.

    All ``partition_assignment``s must be the layout's one -- otherwise
    we would need a different layout per array.  Kept out of the hot
    path: the element-wise comparison is O(n_cells) and only runs when
    the set of partitioned statics changes.
    """
    ref_pa = layout.partition_assignment
    for k, v in sharded.items():
        pa = v.partition_assignment
        # `StaticArray.__post_init__` requires one for replication="partition",
        # which is what `_partition_statics` selected on.
        assert pa is not None
        if pa.shape != ref_pa.shape:
            raise ValueError(
                f"ShardedUnstructuredNode: StaticArray {k!r} has "
                f"partition_assignment.shape={pa.shape} but layout "
                f"was built with shape {ref_pa.shape}."
            )
        # Element-wise equality (numpy-style -- pa might be a numpy or
        # jax array).  The layout claims authority.
        if not np.array_equal(np.asarray(pa), np.asarray(ref_pa)):
            raise ValueError(
                f"ShardedUnstructuredNode: StaticArray {k!r} has a "
                "partition_assignment that disagrees with the "
                "layout's.  All partitioned static arrays on a "
                "node must share one assignment (build the layout "
                "from that one assignment).",
            )


@stability(StabilityLevel.STABLE)
class ShardedUnstructuredNode(SimulationNode):
    """Sharded wrapper for a graph-partitioned :class:`SimulationNode`.

    The wrapped ``node`` exposes ``update_padded(state_padded,
    boundary_inputs, dt, *, static_padded=None, shard_info=None)``,
    the signature a stencil-sharded inner node has -- but not its
    contract.  The padded arrays are in *partition layout*, not the
    Cartesian ``[halo | interior | halo]`` one, so a node is written
    for one wrapper or the other:

    * ``state_padded[<field>]`` is a 1-D-or-higher array whose first
      axis has length ``n_local_max + n_ghost_max``.  The first
      ``n_local_max`` slots are the shard's owned cells (in the order
      declared by the partition layout's ``local_global_ids[d]``); the
      remaining slots are ghost cells in the order declared by
      ``ghost_global_ids[d]``.
    * ``static_padded[<key>]`` follows the same layout when
      ``StaticArray(replication="partition")`` is declared.
    * ``shard_info`` is ``{0: (offset, n_local_max), "n_local":
      n_owned}``.  ``n_local_max`` is the length of the owned block,
      the same on every shard so that it can size a slice; on a shard
      that owns fewer cells than the largest one, rows ``n_owned`` to
      ``n_local_max - 1`` of that block are padding.  ``n_owned`` is
      this shard's own cell count (``layout.n_local[d]``), a *traced*
      int32 scalar: a node that sums over its cells -- a domain
      integral -- masks the padding with ``jnp.arange(n_local_max) <
      shard_info["n_local"]``.  (Added in 0.4.0; before it nothing told
      the node which rows were padding, and a node that summed the
      block summed them into its integral.)  ``offset`` is a *traced*
      ``lax.axis_index(mesh_axis) * n_local_max``.  Unlike the
      Cartesian case, it isn't a geometric coordinate (cells aren't
      contiguous in global ID space), but it's exposed for symmetry
      with :class:`ShardedStencilNode` and for nodes that want a
      unique per-shard tag.

    A node that declares a non-empty ``halo_width()`` is written for
    the Cartesian layout -- its ``update_padded`` reads each field as
    ``[halo | interior | halo]`` -- and is refused at construction:
    handed the partition layout it reads the wrong cells, silently (a
    ``HeatNode`` rod ran 43 K off on two devices, with no error, before
    0.4.0).  Shard it with :class:`ShardedStencilNode`.  A node written
    for this wrapper declares ``halo_width() == {}``: its halo is the
    layout's ghost cells.  The rare node whose ``update_padded`` reads
    *both* layouts, and so declares a halo for the stencil wrapper, can
    say so with a private ``_reads_partition_layout = True``
    (provisional, and not part of the node contract).

    Output classification follows the same rules as
    :class:`ShardedStencilNode`:

    * Keys in ``inner.state_fields()``: padding is stripped (only the
      first ``n_local_max`` slots are returned).
    * Keys in ``inner.domain_integral_fields()``: ``lax.psum`` across
      the mesh axis (the partial sums from each shard are summed).
      Such a key is not a per-cell field: when the state carries it (a
      node that declares its initial value, or a graph feeding a step's
      output back) it is placed as the step returns it -- replicated
      once reduced, stacked along the mesh axis when not -- rather than
      partitioned, and reaches ``update_padded`` unpadded: the reduced
      value, or this shard's slice of the stacked one (a leading axis of
      length one).
    * Any other key raises ``ValueError`` — we cannot infer a
      partition spec for unknown outputs.

    Every other state field is per-cell: its leading axis has one row
    per cell of the layout.  A node whose cell count differs from the
    layout's is refused at construction, naming both counts; before
    0.4.0 the cells past the layout's count were silently dropped.

    Parameters
    ----------
    node : SimulationNode
        The inner node.  Must implement ``update_padded`` with the
        signature above, reading the partition layout, and declare an
        empty ``halo_width()`` -- the halo is determined by the
        partition layout, not by an axis-aligned halo width.  A node
        with a non-empty ``halo_width()`` is refused (see above).
    mesh : Mesh
        1-D JAX device mesh.
    layout : UnstructuredPartitionLayout
        Pre-computed partition / ghost / send-recv tables.
    exchange : {"all_to_all", "ppermute"}, optional
        Halo-exchange transport (see :func:`exchange_unstructured`):
        one dense ``all_to_all`` (default) or one ``ppermute`` per
        communicating cyclic shift, which moves far fewer cells when each
        shard has few neighbours.  Results are bit-identical.
    mesh_axis : str, optional
        The name of the mesh axis to shard along.  Defaults to
        ``"devices"``.  Must match ``mesh.axis_names``.

    Raises
    ------
    ValueError
        If ``mesh_axis`` is not a mesh axis, if the mesh and the layout
        disagree on the device count, if ``node`` declares a non-empty
        ``halo_width()`` (a Cartesian stencil node), if a per-cell state
        field of ``node`` does not have one row per cell of the layout,
        if a partitioned ``StaticArray`` disagrees with the layout, or
        if ``exchange`` is unknown.

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
        exchange: str = "all_to_all",
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

        # A Cartesian stencil node reads ``[halo | interior | halo]``; this
        # wrapper hands it ``[owned | ghosts]``.  Until 0.4.0 the halo was
        # "ignored" and the node stepped the wrong cells without a word
        # (a HeatNode rod 43 K off on two devices, every cell wrong).
        halo = dict(node.halo_width())
        if halo and not _reads_partition_layout(node):
            raise ValueError(
                f"ShardedUnstructuredNode cannot wrap {type(node).__name__} "
                f"{node.name!r}: it declares halo_width() == {halo}, the "
                "Cartesian stencil contract, under which update_padded reads "
                "each field as [halo | interior | halo] along those axes.  This "
                "wrapper hands update_padded the partition layout instead -- "
                "[the shard's own cells | ghost cells], in the order of "
                "layout.local_global_ids and layout.ghost_global_ids -- which "
                "such a node would read as the wrong cells, silently.  Shard a "
                "Cartesian stencil node with ShardedStencilNode, sizing each "
                "sharded axis to a multiple of its device count (or running on "
                "a device count that divides it).  A node written for the "
                "partition layout declares halo_width() == {}: its halo is the "
                "layout's ghost cells."
            )

        # Verify any StaticArray(replication="partition") on the inner
        # node uses a partition_assignment compatible with the layout.
        sharded_static = _partition_statics(node.static_data)
        _validate_partition_statics(sharded_static, layout)

        super().__init__(name=node.name, timestep=node.delta_t, **node.params)
        # Share the inner node's params dict rather than copying it, so a
        # write through any surface reaches the code that reads it.
        self.params = node.params
        if exchange not in ("all_to_all", "ppermute"):
            raise ValueError(
                f"ShardedUnstructuredNode: exchange must be 'all_to_all' or "
                f"'ppermute', got {exchange!r}"
            )
        self._inner = node
        # Graph parameter contract: the wrapper is a params node exactly
        # when the inner ``update_padded`` -- what the shard_map calls --
        # would receive ``params``.  The one rule (explicit keyword or
        # ``**kwargs``, asked through the inner node's own probe), read
        # once: this probe used to accept only the explicit keyword, so an
        # inner ``update_padded(..., **kwargs)`` that ``ShardedStencilNode``
        # calibrates silently left ``gm.params`` here, and
        # ``step(params=...)`` refused it with a false "takes no 'params'
        # keyword".
        self._inner_accepts_params = _method_accepts_params(node, "update_padded")
        self._mesh = mesh
        self._mesh_axis = mesh_axis
        self._layout = layout
        self._exchange = exchange
        self._sharded_static = sharded_static
        self._check_inner_cell_count()
        # Cached compiled fns keyed by the input signature.  Dropped by
        # :meth:`invalidate_static_cache` (every ``compile()``): each entry
        # is a trace of the inner node as it read its params then.
        self._sharded_cache: dict[Any, Any] = {}
        # Rows of a per-cell array in partition layout, and whether that
        # layout *is* global cell order (every shard full, and shard ``d``
        # owning the ``d``-th contiguous block in ascending order).  When
        # the two counts agree and it is not, a global-order array and a
        # partition-layout one have the same shape; see
        # ``_cell_boundary_inputs``.
        self._n_layout_rows = layout.n_devices * layout.n_local_max
        n_global = int(np.asarray(layout.partition_assignment).size)
        self._layout_is_global_order = (
            self._n_layout_rows == n_global
            and np.array_equal(
                np.concatenate([np.asarray(ids) for ids in layout.local_global_ids]),
                np.arange(n_global),
            )
        )
        # A misspelt mesh axis in ``domain_integral_axes`` used to be read
        # as "not this axis" and return the stacked per-shard partials in
        # place of the reduced value; refused here, by name, as
        # ``ShardedStencilNode`` refuses it (building the local update
        # below asks the same question of every integral).
        #
        # The per-shard update, built once and rebuilt by
        # ``invalidate_static_cache``, as ``ShardedStencilNode`` keeps it.
        # Held as an attribute, its closure over the inner node is what
        # tells the REST route's parameter probe (which answers "would
        # the running node use this write?" on a shallow copy) that this
        # wrapper cannot be copied, so it asks the node inside, which
        # shares the params dict.  Rebuilt on each cache miss instead, the
        # wrapper looked copyable, and the copy shared -- and wrote into --
        # the compiled-function cache of the original: it answered with the
        # original's trace and left its own traces behind for the original
        # to call.
        self._local_update_fn = self._build_local_update()
        # Cached per-device materialisation of the partitioned statics;
        # see ``_materialise_partitioned_statics``.
        self._static_device_cache: Optional[tuple] = None

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

    def update_evaluations(self) -> Optional[float]:
        """The wrapped node's declaration: sharding does not change how often it rounds.

        Without it a sub-stepping node lost its declaration when wrapped,
        and a coupling group containing it read its float floor as one
        evaluation (``spectral_usable=False`` at the floor, where the
        unwrapped node's group was usable).
        """
        return self._inner.update_evaluations()

    def interface_dof_indices(self) -> dict[str, tuple[str, int]]:
        """Refuse, by name, an inner node that declares interface DOFs.

        The graph corrects a coupled interface by writing
        ``state[field].at[index]``; here the state is in partition layout
        (``n_devices * n_local_max`` rows, cells permuted and padded), so
        the inner node's global ``index`` names a different cell, and its
        ``compute_interface_correction`` reads a state it was not written
        for.  The base-class answer, ``{}``, left the interface silently
        uncorrected.  An inner node without interface DOFs is unaffected.
        ``ShardedPointwiseNode`` and ``ShardedStencilNode`` keep the
        inner node's global view and forward both hooks.
        """
        iface = self._inner.interface_dof_indices()
        if iface:
            raise NotImplementedError(
                f"ShardedUnstructuredNode {self.name!r} cannot forward "
                f"{type(self._inner).__name__}.interface_dof_indices() "
                f"{sorted(iface)}: its state is in partition layout, where "
                "the inner node's global cell indices name different cells, "
                "so the coupled interface correction would be applied to the "
                "wrong cells.  Couple the unwrapped node, or shard it with "
                "ShardedStencilNode, which keeps the global view."
            )
        return {}

    def initial_state(self) -> dict:
        """Materialise the inner node's initial state onto the mesh.

        The inner node produces a global initial state (first axis =
        global cell axis).  We slice into (n_devices, n_local_max, *)
        and place on the mesh via ``NamedSharding``.
        """
        global_state = self._inner.initial_state()
        sharded = {}
        sharding = NamedSharding(self._mesh, P(self._mesh_axis))
        integrals = set(self._inner.domain_integral_fields())
        for k, arr in global_state.items():
            arr = jnp.asarray(arr)
            if k in integrals:
                # A domain integral is not a per-cell field: it is placed
                # as the step returns it, replicated once reduced.
                # Partitioned, a scalar raised an IndexError here.  A
                # per-shard (unreduced) integral comes back stacked along
                # the mesh axis, so its initial value is stacked too, and a
                # scan carry keeps its shape.
                if not self._integral_is_reduced(k):
                    arr = jnp.broadcast_to(
                        arr, (self._layout.n_devices,) + tuple(arr.shape),
                    )
                sharded[k] = jax.device_put(
                    arr, NamedSharding(self._mesh, self._integral_spec(k)),
                )
                continue
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
    def accepts_params(self, *, method: str = "update") -> bool:
        if method != "update":
            return super().accepts_params(method=method)
        return self._inner_accepts_params

    def params_pytree(self) -> dict:
        return self._inner.params_pytree() if self._inner_accepts_params else {}

    def param_specs(self) -> dict:
        return self._inner.param_specs() if self._inner_accepts_params else {}

    def update(
        self, state: dict, boundary_inputs: dict, dt: float, *, params=None,
    ) -> dict:
        """Run one step under sharding.

        The wrapper compiles a shard_map'd implementation per
        (state_signature, bi_signature, static_signature, params
        signature) tuple and dispatches.  ``params`` (the node's entry of
        ``GraphManager.params``) is replicated to every shard and handed
        to an inner ``update_padded(..., params=)``.

        ``state`` is in partition layout, as :meth:`initial_state` builds
        it: every field but a domain integral has ``n_devices *
        n_local_max`` rows.  A field with another leading axis -- a state
        in global cell order on a partition with padding, say -- is
        refused by name.  When the two counts agree (every shard full)
        and the partition does not keep cells in global order, a
        global-order state has the same shape as a partition-layout one
        and cannot be told from it here; convert it with
        :func:`~maddening.cloud.multigpu.halo_unstructured.partition_value`
        before writing it.
        """
        self._check_state_layout(state)
        # Materialise first: it refreshes ``self._sharded_static``, which
        # ``_get_sharded_fn`` keys its compiled-function cache on.
        static_partitioned = self._materialise_partitioned_statics()
        fn = self._get_sharded_fn(state, boundary_inputs, params)
        return fn(state, boundary_inputs, jnp.asarray(dt), static_partitioned,
                  params if params else {})

    @stability(StabilityLevel.STABLE)
    def invalidate_static_cache(self) -> None:
        """Drop the cached per-device copy of the partitioned statics.

        Call this after rewriting a partitioned ``StaticArray``'s buffer
        in place; replacing the array object is detected automatically.
        The ``super()`` call forwards to the wrapped node, so a cache
        further in is dropped too.

        The compiled ``shard_map`` step functions go with it, because
        :meth:`~maddening.core.graph_manager.GraphManager.compile` calls
        this on every node and a recompile has to trace the inner node
        afresh: a node on the three-argument contract that reads a
        constant from ``self.params`` has that value baked into each
        cached trace.  Before 0.4.0 they survived, so a parameter write
        followed by ``compile()`` left the sharded step on the old value
        (MADD-ANO-032).
        """
        self._static_device_cache = None
        self._sharded_cache.clear()
        self._local_update_fn = self._build_local_update()
        super().invalidate_static_cache()

    def _materialise_partitioned_statics(self) -> dict:
        """Per-device materialisation of every partitioned StaticArray.

        Partitioning a static array costs a device-to-host copy, a
        NumPy gather through the layout, and a host-to-device copy.
        None of that depends on the state, so doing it on every public
        ``update`` was the dominant host cost of an interactive step.
        The result is cached and reused; the key is the identity of each
        static's underlying array (see
        :meth:`ShardedStencilNode._static_cache_key` for why identity,
        and for the one case it cannot see).
        """
        sharded = _partition_statics(self._inner.static_data)
        key = tuple(
            (k, id(sharded[k].value), tuple(sharded[k].value.shape),
             str(sharded[k].value.dtype))
            for k in sorted(sharded)
        )
        cached = self._static_device_cache
        if cached is not None and cached[0] == key:
            return cached[1]

        _validate_partition_statics(sharded, self._layout)
        if sorted(sharded) != sorted(self._sharded_static):
            # A partitioned key appeared or vanished: every shard_map
            # compiled against the old set is stale.
            self._sharded_cache.clear()
        self._sharded_static = sharded

        # ``ensure_compile_time_eval``: this may run inside GraphManager's
        # trace of the compiled step, and a cached tracer would escape it.
        # The statics are compile-time constants, so evaluating the
        # placement eagerly is both correct and what we want cached.
        static_partitioned = {}
        sharding = NamedSharding(self._mesh, P(self._mesh_axis))
        with jax.ensure_compile_time_eval():
            for k, sa in sharded.items():
                host = jax.device_get(sa.value) if hasattr(sa.value, "device") \
                    else sa.value
                per_shard = partition_value(value=host, layout=self._layout)
                static_partitioned[k] = jax.device_put(
                    jnp.asarray(per_shard.reshape(
                        (self._layout.n_devices * self._layout.n_local_max,)
                        + per_shard.shape[2:]
                    )), sharding,
                )
        # ``sharded`` is retained so the ``id()``s in ``key`` stay pinned.
        self._static_device_cache = (key, static_partitioned, sharded)
        return static_partitioned

    # -----------------------------------------------------------------
    # Internals
    # -----------------------------------------------------------------
    def _get_sharded_fn(self, state, boundary_inputs, params=None):
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
            _params_signature(params),
        )
        cached = self._sharded_cache.get(key)
        if cached is not None:
            return cached

        # A domain integral fed back as state (a graph's second step) is
        # not a per-cell field: it is placed as the step returns it --
        # replicated once reduced, stacked along the mesh axis when not.
        # ``P(mesh_axis)`` on a reduced scalar was "too long" for it and
        # the step failed.
        integrals = set(self._inner.domain_integral_fields())
        state_specs = {
            k: (self._integral_spec(k) if k in integrals else P(self._mesh_axis))
            for k in state
        }
        cell_bi = self._cell_boundary_inputs(boundary_inputs)
        bi_specs = {
            k: (P(self._mesh_axis) if k in cell_bi else P())
            for k in boundary_inputs
        }
        static_specs = {k: P(self._mesh_axis) for k in self._sharded_static}
        out_specs = {**state_specs}
        for k in integrals:
            # fully replicated after psum, or the per-shard values stacked
            out_specs[k] = self._integral_spec(k)

        local_fn = functools.partial(self._local_update_fn, cell_bi=cell_bi)

        params_specs = jax.tree.map(lambda _: P(), params if params else {})
        sm = shard_map(
            local_fn,
            mesh=self._mesh,
            in_specs=(state_specs, bi_specs, P(), static_specs, params_specs),
            out_specs=out_specs,
        )
        fn = jax.jit(sm)
        self._sharded_cache[key] = fn
        return fn

    def _integral_spec(self, key: str) -> P:
        """Where domain integral ``key`` lives, as the step returns it."""
        return P() if self._integral_is_reduced(key) else P(self._mesh_axis)

    def _integral_is_reduced(self, key: str) -> bool:
        """Is domain integral ``key`` summed over the mesh axis?

        ``True`` unless the node's ``domain_integral_axes()`` names axes
        for ``key`` that leave this wrapper's mesh axis out, in which case
        the per-shard values are stacked.  An axis name the mesh does not
        have is refused rather than read as "not this axis".
        """
        axes = dict(getattr(self._inner, "domain_integral_axes", dict)()).get(key)
        if axes is None:
            return True
        axes = tuple(axes)
        mesh_axes = tuple(self._mesh.axis_names)
        unknown = [a for a in axes if a not in mesh_axes]
        if unknown:
            raise ValueError(
                f"ShardedUnstructuredNode: domain_integral_axes[{key!r}] of "
                f"{type(self._inner).__name__} {self._inner.name!r} names mesh "
                f"axes {unknown} not in mesh.axis_names={mesh_axes}"
            )
        return self._mesh_axis in axes

    def _check_inner_cell_count(self) -> None:
        """Refuse a node whose per-cell fields do not have the layout's cell count.

        ``partition_value`` gathers ``value[ids]`` for the ids the layout
        owns; before 0.4.0 it never compared ``value.shape[0]`` with the
        layout, so a 10-cell node on an 8-cell layout lost cells 8 and 9
        without a word (and its integral with them).  Checked here, on
        the shapes ``initial_state()`` builds (evaluated abstractly), so
        the refusal names the node at construction; ``partition_value``
        refuses the same thing wherever else it is called.  A node whose
        ``initial_state()`` cannot be evaluated that way (one waiting for
        a ``static_data_provider``, say) is left to that backstop.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                built = dict(jax.eval_shape(self._inner.initial_state))
        except Exception:  # noqa: BLE001 - cannot tell: the backstop checks
            return
        n_global = int(np.asarray(self._layout.partition_assignment).size)
        integrals = set(self._inner.domain_integral_fields())
        for k, v in built.items():
            if k in integrals or not hasattr(v, "shape"):
                continue
            shape = tuple(v.shape)
            if shape and shape[0] == n_global:
                continue
            rows = f"{shape[0]} rows" if shape else "no cell axis (0-d)"
            raise ValueError(
                f"ShardedUnstructuredNode cannot wrap {type(self._inner).__name__} "
                f"{self._inner.name!r}: its state field {k!r} has shape {shape}, "
                f"{rows}, but the layout partitions {n_global} cells.  Every "
                "state field but a domain integral is per-cell, one row per "
                "cell of the layout; the rows past the layout's count would be "
                "silently dropped.  Build the layout from this node's cells, "
                "or declare a field that is one value for the whole domain in "
                "domain_integral_fields()."
            )

    def _check_state_layout(self, state: dict) -> None:
        """Refuse a per-cell state field that is not in partition layout."""
        n_layout = self._n_layout_rows
        # Every field but a domain integral (replicated after its psum) is
        # per-cell: ``initial_state`` partitions them all.
        integrals = set(self._inner.domain_integral_fields())
        for k in state:
            if k in integrals:
                continue
            shape = tuple(jnp.shape(state[k]))
            if shape and shape[0] == n_layout:
                continue
            raise ValueError(
                f"ShardedUnstructuredNode {self.name!r}: state field {k!r} has "
                f"shape {shape}; the wrapper steps a state in partition layout, "
                f"{n_layout} rows (n_devices * n_local_max = "
                f"{self._layout.n_devices} * {self._layout.n_local_max}), as "
                "initial_state() builds it.  A state in global cell order has "
                "to go through partition_value(value=..., layout=...) and be "
                f"reshaped to ({n_layout}, ...) first."
            )

    def _cell_boundary_inputs(self, boundary_inputs: dict) -> frozenset[str]:
        """Names of the boundary inputs that are per-cell fields.

        A per-cell boundary input is laid out like this wrapper's state:
        leading axis of length ``n_devices * n_local_max`` in partition
        order (what :meth:`initial_state` produces, and what
        :func:`partition_value` gives for a global-order array).  It is
        sharded and ghost-exchanged like a state field, so the inner sees
        ``n_local_max + n_ghost_max`` rows.  Scalars and anything else are
        replicated.  A global-order array (leading axis ``n_global_cells``)
        is refused rather than silently misread.

        When the two lengths coincide -- every shard owns ``n_local_max``
        cells -- the shape cannot say which order an array is in.  If the
        partition keeps cells in global order (shard ``d`` owns the
        ``d``-th contiguous block, ascending) the two orders are the same
        array and it is accepted.  Otherwise it is refused: before the
        refusal it was read as partition layout, so a global-order input
        -- what an edge from an unsharded node delivers -- gave each cell
        another cell's value (an interleaved 8-cell, 2-device partition read
        ``[1..8]`` as ``[1, 5, 2, 6, 3, 7, 4, 8]``).  Renumber the cells so
        that each shard owns a contiguous ascending block, and both
        readings agree.
        """
        n_layout = self._n_layout_rows
        n_global = int(np.asarray(self._layout.partition_assignment).size)
        out = set()
        for k, v in boundary_inputs.items():
            shape = tuple(jnp.shape(v))
            if not shape:
                continue
            if shape[0] == n_layout and n_layout == n_global \
                    and not self._layout_is_global_order:
                raise ValueError(
                    f"boundary input {k!r} has leading axis {n_layout}, which "
                    "is both the global cell count and the partition-layout "
                    f"row count (n_devices * n_local_max = "
                    f"{self._layout.n_devices} * {self._layout.n_local_max}), "
                    "and this partition does not keep cells in global order: "
                    "ShardedUnstructuredNode cannot tell a global-order array "
                    "from a partition-layout one, and reading one as the other "
                    "gives each cell another cell's value.  Renumber the cells "
                    "so that each device owns a contiguous, ascending block of "
                    "global ids (e.g. relabel them in the order "
                    "np.argsort(partition_assignment, kind='stable')); the two "
                    "orders then coincide."
                )
            if shape[0] == n_layout:
                out.add(k)
            elif shape[0] == n_global:
                raise ValueError(
                    f"boundary input {k!r} has leading axis {n_global} (global "
                    f"cell order); ShardedUnstructuredNode expects per-cell "
                    f"inputs in partition layout ({n_layout} rows) -- run it "
                    "through partition_value(value=..., layout=...) and reshape "
                    f"to ({n_layout}, ...) first."
                )
        return frozenset(out)

    def _build_local_update(self):
        inner = self._inner
        layout = self._layout
        mesh_axis = self._mesh_axis
        exchange = self._exchange
        n_local_max = layout.n_local_max
        n_local_per_shard = np.asarray(layout.n_local, dtype=np.int32)
        state_set = set(inner.state_fields())
        integrals = set(inner.domain_integral_fields())
        reduced = {k: self._integral_is_reduced(k) for k in integrals}

        accepts_params = self._inner_accepts_params

        def _local_update(local_state, local_bi, local_dt, local_static,
                          local_params, *, cell_bi=frozenset()):
            # Strip the leading device dimension that shard_map already
            # collapsed for us — local arrays now have shape (n_local_max, *).

            # 1. Halo-exchange each state field.
            #    A domain integral carried in the state is not a per-cell
            #    field: passed through unpadded.
            padded_state = {}
            for k, arr in local_state.items():
                if k in integrals:
                    padded_state[k] = arr
                    continue
                padded_state[k] = exchange_unstructured(
                    arr, layout=layout, mesh_axis=mesh_axis, method=exchange,
                )
            # 1b. Per-cell boundary inputs get the same ghost exchange so
            #     the inner reads them at the padded local shape.
            local_bi = {
                k: (exchange_unstructured(v, layout=layout, mesh_axis=mesh_axis, method=exchange)
                    if k in cell_bi else v)
                for k, v in local_bi.items()
            }

            # 2. Halo-exchange each partitioned static.
            padded_static = {}
            for k, arr in local_static.items():
                padded_static[k] = exchange_unstructured(
                    arr, layout=layout, mesh_axis=mesh_axis, method=exchange,
                )

            # 3. shard_info — traced offset for nodes that want a per-shard
            #    tag, the padded block length (static, it sizes slices), and
            #    this shard's own cell count (traced), which is what tells a
            #    node summing its cells where the padding starts.
            idx = lax.axis_index(mesh_axis)
            shard_info = {
                0: (idx * n_local_max, n_local_max),
                "n_local": jnp.asarray(n_local_per_shard)[idx],
            }

            # 4. Dispatch.
            extra = {"params": local_params} if (accepts_params and local_params) else {}
            new = inner.update_padded(
                padded_state, local_bi, local_dt,
                static_padded=(padded_static or None),
                shard_info=shard_info, **extra,
            )

            # 5. Classify outputs.
            out = {}
            for k, v in new.items():
                if k in state_set:
                    # Strip the ghost tail.
                    out[k] = v[:n_local_max]
                elif k in integrals:
                    if reduced[k]:
                        out[k] = lax.psum(v, axis_name=mesh_axis)
                    else:
                        out[k] = v[None]          # stacked along the mesh axis
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
        d["exchange"] = self._exchange
        return d


__all__ = [
    "ShardedUnstructuredNode",
]
