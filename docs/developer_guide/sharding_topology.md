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
  `lax.psum`-ed across the mesh; other keys raise.
* `StaticArray` carries the per-array sharding policy via
  `replication=` (`"replicate"` / `"shard"` / `"partition"`).

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

For the v0.3.0 §A6 substrate the toy test is 16 cells and the
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
* ``plans/MADDENING_v0.3.0_PLAN.md`` §A6 — the source-of-truth scope
  document (not version-controlled; lives next to the repo).
