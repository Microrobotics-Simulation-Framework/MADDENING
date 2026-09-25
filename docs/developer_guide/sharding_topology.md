# Sharding topology: structured vs unstructured

:::{versionadded} 0.3.0
The unstructured-sharding contract (``ShardedUnstructuredNode`` +
``StaticArray(replication="partition")`` + the
``halo_unstructured.exchange_unstructured`` collective) landed in
v0.3.0 as a sibling of the v0.2.1 structured path
(``ShardedStencilNode`` + ``StaticArray(replication="shard")``).
:::

## Choice criteria

MADDENING ships **two** first-class sharding wrappers — pick the one
whose topology matches your node's stencil:

| Wrapper | When to use it | Performance |
|---|---|---|
| `ShardedPointwiseNode` | Pointwise nodes (no spatial topology, no halo). | Trivial — every shard runs its slice independently. |
| `ShardedStencilNode` | **Cartesian** grids — `(nx, ny, nz)`-shaped state, axis-aligned halos via `halo_width()`. | XLA SIMD-vectorised, contiguous memory, ~10–100× faster than the equivalent unstructured topology at the same cell count. |
| `ShardedUnstructuredNode` | **Arbitrary connectivity** — FVM on unstructured meshes, GNNs over real graphs, finite-element on irregular grids. | Sparse halo exchange via `lax.all_to_all`; pays a per-cell scatter/gather cost compared to the Cartesian path. |

When both topologies are technically viable (some structured-on-irregular
problems can be expressed either way), the rule is:

* **Structured wins on Cartesian-like geometry**, even if there are
  irregular embedded boundaries.  Represent the geometry via masks
  in `static_data`, not via partitioning.
* **Unstructured wins on intrinsically graph-shaped problems** where
  trying to embed the connectivity in a Cartesian array would be
  wasteful (large blanked-out regions, very high anisotropy in the
  mesh, mesh adapts dynamically, etc.).

### The grid extent must divide by the device count (Cartesian paths only)

Both Cartesian wrappers split each sharded axis **evenly**, so a sharded
extent has to be a multiple of the devices on its mesh axis: 16 cells over
3 devices has no layout.  Since v0.4.0 `ShardedStencilNode` and
`ShardedPointwiseNode` refuse that at construction, with a `ValueError`
naming the node, the cell count and the device count; before that it
surfaced much later as a `jax.errors.IndivisibleError` raised from inside
`device_put`, naming none of them.

Three ways out, in the order most people want them:

1. **Size the grid to a multiple of the device count.**  A sharded axis of
   `n_devices * k` cells is the cheapest fix and keeps the Cartesian
   performance in the table above.
2. **Run on a device count that divides the grid.**  The error message
   lists the ones that do.
3. **Use `ShardedUnstructuredNode`.**  It carries an explicit padded
   layout and accepts **any** (device, cell) pair — 17 cells over 3
   devices included — at the per-cell scatter/gather cost the table
   above describes.  The constraint is a property of the Cartesian
   pencil decomposition, not of the framework.

## Class hierarchy

```text
ShardedNode (Protocol — surface contract only)
├── ShardedPointwiseNode      (no halo, no spatial topology)      v0.2.0
├── ShardedStencilNode        (Cartesian, axis-aligned halos)     v0.2.1
└── ShardedUnstructuredNode   (graph-partition, sparse halos)     v0.3.0
```

All three share the same substrate:

* The wrapped inner node's `update_padded(state_padded,
  boundary_inputs, dt, *, static_padded=None, shard_info=None)`
  signature is identical.
* Outputs are classified the same way: keys in `state_fields()` have
  halo/padding stripped; keys in `domain_integral_fields()` get
  `lax.psum`-ed across the mesh — or across the subset of mesh axes
  `domain_integral_axes()` names for them, keeping a leading axis per
  unreduced mesh axis; other keys raise.
* Both wrappers take part in the graph parameter contract: if the inner
  node's `update_padded` accepts `params`, the wrapper exposes the inner
  `params_pytree()` and hands the node's entry of `gm.params` (replicated
  to every shard) to `update_padded(..., params=)`.
* The sharded Krylov solvers (`sharded_cg` / `sharded_gmres`) take a
  `preconditioner=` (`jacobi_preconditioner`, `block_jacobi_preconditioner`
  ship) and `differentiable=True`, which routes the solve through
  `lax.custom_linear_solve` so a node that solves inside a differentiated
  step gets an exact linear-solve adjoint with the same preconditioner
  applied in the adjoint solve; the iteration count is then reported as
  -1.
* `StaticArray` carries the per-array sharding policy via
  `replication=` (`"replicate"` / `"shard"` / `"partition"`).
* Boundary inputs are classified by shape.  A **grid-shaped** input (on
  the stencil path: same extent as the state fields on every sharded
  spatial axis, e.g. an LBM per-cell `body_force` map or a
  `wall_mask_update`; on the unstructured path: leading axis of length
  `n_devices * n_local_max` in partition layout, i.e. what
  `partition_value` produces) is sharded and halo/ghost-padded exactly
  like a state field, so `update_padded` receives it at the padded local
  shape.  Everything else (a scalar pressure, a uniform `(D,)` force
  vector) is replicated to every shard.  The unstructured wrapper refuses
  a per-cell input given in *global* cell order rather than misreading it.

## Halo-exchange transport (unstructured path)

`exchange_unstructured` has two transports that return bit-identical
slabs.  `method="all_to_all"` (default) packs one `(n_devices,
n_ghost_max)` payload per shard and issues a single `lax.all_to_all`, so
every shard sends `n_devices * n_ghost_max` cells whether or not it
neighbours the receiver.  `method="ppermute"` issues one `lax.ppermute`
per cyclic shift that actually carries cells, sized to that shift's
largest message; a partition where shards talk to few neighbours moves a
fraction of the cells.  `ShardedUnstructuredNode(..., exchange=...)`
selects it per node, and `exchange_traffic(layout)` gives the cells
moved per shard for both, plus the useful count, straight from the
layout.  Which transport is *faster* under NCCL is hardware-dependent
and is measured in the real multi-GPU session; the correctness and the
byte counts are settled here.

## Partition-assignment handoff (unstructured path)

The contract for unstructured sharding has **three** participants:

```text
┌─────────────────────────┐     ┌──────────────────────────┐     ┌────────────────────────────┐
│ Experiment setup (host) │     │ build_unstructured_      │     │ ShardedUnstructuredNode    │
│  PyMetis / METIS / etc. │ ──▶ │   partition()  (host)    │ ──▶ │  (constructor)             │
│   ↓                     │     │   ↓                      │     │   ↓                        │
│ partition_assignment    │     │ UnstructuredPartition-   │     │ shard_map dispatch +       │
│  (int array)            │     │   Layout                 │     │  exchange_unstructured     │
└─────────────────────────┘     └──────────────────────────┘     └────────────────────────────┘
```

**Step 1 — partition the mesh on the host.**  This is *not* MADDENING's
job.  Use PyMetis (or an equivalent partitioner) at experiment-setup
time to map each global cell to a device index.  The output is a 1-D
int array of length `n_global_cells`, value = device index.

**Step 2 — compute the layout once, on the host.**  Pass the
partition assignment and the global connectivity table to
``maddening.cloud.multigpu.halo_unstructured.build_unstructured_partition``.
The result is an
``UnstructuredPartitionLayout`` carrying per-shard local→global ID
lists, ghost-cell global IDs, and the send/recv index tables the
``all_to_all`` collective needs at runtime.

**Step 3 — wrap the node.**  Pass the layout to
``ShardedUnstructuredNode(node, mesh, layout)``.  Any
``StaticArray(replication="partition", partition_assignment=...)`` on
the inner node must use the same assignment (the layout is
authoritative).

Reproducibility: the partition assignment is part of the experiment's
reproducibility bundle (alongside seeds and version pins).  It is not
recomputed every run.  Different PyMetis versions can produce
different assignments for the same mesh — that's a user-side
versioning concern, not a MADDENING-level guarantee.

## Performance trade-offs

The structured path benefits from XLA's contiguous-memory
optimisations: a stencil read on a `(nx, ny, nz)` array is a strided
gather that XLA can vectorise across SIMD lanes.  The unstructured
path goes through `jnp.take` on an explicit gather table, which is
not amenable to the same optimisations.

In practice that means: a 256³ Cartesian LBM runs ~10–100× faster than
the equivalent expressed as a graph with the same number of cells.
For the MICROROBOTICA Light cloud-rendered demos at 30 fps the gap is
load-bearing — that's why we keep both paths.

For the v0.3.0 unstructured substrate the toy test is 16 cells and the
intermediate smoke is 1024 cells; v0.4.0 work will tune the sparse
halo exchange for real-mesh sizes (10⁴–10⁶ cells) and add NCCL
fast-paths for actual GPUs.

## v0.4.0 commitment (hard downstream gate)

By MADDENING v0.4.0 (MIME v0.5.0) the unstructured sharded path must
support a real FVM `FVMFluidNode` in MIME.  That commitment is what
makes the v0.3.0 contract load-bearing: the constructor signature,
`update_padded` plumbing, output classification, and partition-
assignment handoff documented above are
``@stability(stable)``-ready.  Any breaking change here cascades into
a MIME rewrite; v0.4.0 hardens, it doesn't redesign.

If you find a flaw in the v0.3.0 contract while implementing v0.4.0,
fix it back in v0.3.0 — surface the break here, not in v0.4.0.

## See also

* {doc}`node_authoring` — the inner-node side of the contract.
* {doc}`stability_report` — current @stability tagging of the
  sharding API surface.
