---
orphan: false
---

# Sharded static data and domain integrals (v0.2.1)

```{versionadded} v0.2.1
Per-device materialisation of `StaticArray(replication="shard")`
under {class}`~maddening.cloud.multigpu.sharded_node.ShardedStencilNode`,
and the {meth}`~maddening.core.node.SimulationNode.domain_integral_fields`
API for cross-device reduction of partial-sum outputs.
```

This page explains how to write a stencil node whose non-evolving
arrays are *sharded* alongside state — needed when the array is
big enough that replicating it per device would defeat the point of
sharding — and how to declare outputs that are domain integrals
over the lattice so the wrapper all-reduces them.

The motivating consumer is MIME's `IBLBMFluidNode`: a D3Q19 LBM
node carrying a `(nx, ny, nz)` pipe-wall mask and a
`(19, nx, ny, nz)` "missing-link" bounce-back mask, both far too
large to replicate on every device when the simulation is sharded
across a pencil mesh.

## The picture

```{mermaid}
flowchart LR
    subgraph "ShardedStencilNode.update()"
        A[state pytree<br/>and StaticArray] --> B[device_put<br/>NamedSharding per shard_axis]
        B --> C[shard_map]
        C --> D[halo_exchange<br/>state slab + static slab]
        D --> E[inner.update_padded<br/>static_padded=...<br/>shard_info=...]
        E --> F[strip halos<br/>or psum integrals]
        F --> G[next state]
    end
```

Three things happen per step:

1. The wrapper takes each `StaticArray(replication="shard")` on
   the inner node and `device_put`s it with a `NamedSharding`
   whose `PartitionSpec` puts the matching mesh axis on the
   array's `shard_axis` (cached by `static_data_hash`).
2. Inside `shard_map`, each device's slab is halo-exchanged along
   the matching spatial axis (boundary `"edge"` — static arrays
   don't evolve, so periodic wrap is wrong even if state uses
   periodic).
3. The inner node's `update_padded` receives the padded slab via
   `static_padded[<key>]` and uses it like any other padded array.

## Declaring a sharded static array

The declaration lives on the node:

```python
from maddening.core.node import SimulationNode
from maddening.core.static_data import StaticArray

class WallBouncebackLBM(SimulationNode):

    def __init__(self, name, timestep, *, nx, ny, nz, pipe_radius):
        super().__init__(name, timestep,
                          nx=nx, ny=ny, nz=nz, pipe_radius=pipe_radius)
        self._pipe_wall = _build_pipe_mask(nx, ny, nz, pipe_radius)

    def halo_width(self):
        # D3Q19: read ±1 neighbour on every spatial axis
        return {0: 1, 1: 1, 2: 1}

    @property
    def static_data(self):
        return {
            "pipe_wall": StaticArray(
                self._pipe_wall,
                replication="shard",
                shard_axis=0,           # shard along the x axis
            ),
        }
```

`shard_axis` is the array's *own* axis — it must match one of the
spatial axes the wrapping `ShardedStencilNode` actually shards
(`shard_axis ∈ axis_map.values()`) **and** the node must declare
a non-zero `halo_width()` entry on that axis (otherwise there's
no neighbour slab to exchange with).  Both invariants are checked
at `ShardedStencilNode.__init__` time:

```python
sharded = ShardedStencilNode(
    WallBouncebackLBM("lbm", 0.001, nx=128, ny=64, nz=64, pipe_radius=20),
    mesh=mesh,
    axis_map={"spatial_x": 0},   # shard axis 0 on mesh "spatial_x"
    boundary="periodic",
)
# Raises ValueError immediately if pipe_wall's shard_axis isn't in
# axis_map.values() (i.e. it's not on the sharded spatial axis), or
# if halo_width() has no entry for shard_axis.
```

Replicate-mode `StaticArray` entries are not affected — they stay
closure-captured in full on every device, exactly as in v0.2.0.

## Reading the sharded slab in `update_padded`

The inner node's `update_padded` gets a new keyword-only argument:

```python
def update_padded(
    self, state_padded, boundary_inputs, dt,
    *, static_padded=None, shard_info=None,
) -> dict:
    """Receives a per-shard slab of pipe_wall via `static_padded`."""
    f_pad      = state_padded["f"]               # (nx_local+2, ny, nz, 19)
    wall_pad   = static_padded["pipe_wall"]      # (nx_local+2, ny, nz)
    # interior view (halo stripped) is wall_pad[1:-1, :, :]
    # cells outside the interior come from the neighbour shard's
    # boundary cells via halo_exchange
    ...
```

The keyword is `None` when there are no sharded static_data
entries on the node, or when the node is run outside of
`ShardedStencilNode` (e.g. single-device path).  Default to the
closure-captured full array in that case:

```python
def update_padded(
    self, state_padded, boundary_inputs, dt,
    *, static_padded=None, shard_info=None,
):
    if static_padded is not None and "pipe_wall" in static_padded:
        wall = static_padded["pipe_wall"]
    else:
        # Unsharded path; the full wall is closure-captured on self.
        wall = jnp.pad(self._pipe_wall, [(1, 1), (0, 0), (0, 0)], mode="edge")
    ...
```

## Declaring domain-integral outputs

A node that computes a `jnp.sum`-over-lattice output (a drag
force, a total mass, a heat-flux integral) needs cross-device
reduction under sharding.  Declare which output keys are
integrals:

```python
def domain_integral_fields(self) -> set[str]:
    return {"drag_force", "drag_torque"}
```

and have `update_padded` compute the partial sum on the local
slab — no `psum` in the node:

```python
def update_padded(self, state_padded, boundary_inputs, dt,
                  *, static_padded=None, shard_info=None):
    ...
    force_partial = jnp.sum(
        per_cell_force * wall_pad[1:-1, :, :, None],
        axis=(0, 1, 2),
    )
    return {
        "f": f_new,
        "drag_force": force_partial,    # (3,) — partial sum on this shard
    }
```

ShardedStencilNode applies
`lax.psum(value, axis_name=tuple(mesh.axis_names))` to every key
listed in `domain_integral_fields()` after `update_padded`
returns, and sets the corresponding `out_spec` to `P()` (fully
replicated post-psum).  Returned integral values are
floating-point — `psum` on integer types risks wrap on large
meshes.

```{important}
Every key returned from `update_padded` must be either in
`state_fields()` or in `domain_integral_fields()`.  The wrapper
defensively raises `ValueError` at trace time on an unknown key
— it has no way to infer the partition spec for an unclassified
output.
```

## `shard_info`: when you'd rather recompute the mask

Not every static array needs to be materialised.  For analytic
masks (a sphere, a cylinder, a coordinate range), it's often
cheaper to recompute per shard than to ship the full mask through
`jax.device_put` once and `halo_exchange` it every step.

ShardedStencilNode populates a `shard_info` dict for the inner
node:

```python
def update_padded(self, state_padded, boundary_inputs, dt,
                  *, static_padded=None, shard_info=None):
    # shard_info = {0: (global_offset, local_extent)}
    #
    # global_offset is a TRACED jax scalar:
    #   lax.axis_index(mesh_axis) * local_extent
    # local_extent is a Python int.
    if shard_info is not None and 0 in shard_info:
        offset, extent = shard_info[0]
        # offset usable in dynamic_slice; NOT in Python int slicing
        global_x = offset + jnp.arange(extent + 2)    # +2 for halos
        ...
```

`shard_info` is `None` when the node is run outside of
`ShardedStencilNode`.

## Sharding policy is part of the JIT cache key

`SimulationNode.static_data_hash()` already incorporates
`(key, shape, dtype, replication, shard_axis)`.  Switching a
StaticArray from `replication="replicate"` to `replication="shard"`
between graph compiles invalidates the JIT trace — no extra
machinery needed in user code.  The shard_map cache inside
`ShardedStencilNode` also keys on `static_data_hash()`, so
*identity* changes (a user replacing `self._mask` between steps)
invalidate cleanly even when shape and dtype are unchanged.

## What's still TODO

* **Partial-axis psum.**  `domain_integral_fields()` triggers a
  full-mesh `psum` over every mesh axis.  Reductions over a
  subset of axes (e.g. for outputs that vary along one axis but
  integrate along the others) are out of scope for v0.2.1 — open
  an issue if you need it.
* **Sharded static arrays without halos.**  v0.2.1 rejects a
  sharded `StaticArray` whose `shard_axis` doesn't appear in
  `node.halo_width()`.  A future relaxation could allow such
  arrays by skipping the halo exchange, but the semantics get
  subtle near slab boundaries.

## See also

* {doc}`/release_notes/v0.2.1` — release notes for the cycle this
  shipped in.
* [`StaticArray` API
  reference](https://github.com/Microrobotics-Simulation-Framework/MADDENING/blob/main/src/maddening/core/static_data.py)
  — dataclass contract and hash semantics.
* {doc}`edge_validation_migration` — companion v0.2.1 change.
