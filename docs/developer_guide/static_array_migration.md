---
orphan: false
---

# `StaticArray` migration guide (v0.2 → v0.3.0)

```{versionchanged} v0.2
{class}`~maddening.core.static_data.StaticArray` introduced as the
typed wrapper for non-state arrays carried via
{attr}`SimulationNode.static_data <maddening.core.node.SimulationNode.static_data>`.
Bare arrays (anything with ``shape``/``dtype`` but not wrapped) were
coerced to ``StaticArray(value=arr)`` with a one-time
{class}`FutureWarning` per (node × key) pair.
```

```{versionchanged} v0.3.0
The bare-array coercion path was **removed**.  Putting a bare
ndarray / ``jnp.ndarray`` into ``static_data`` now raises
{class}`~maddening.warnings.MigrationError` immediately.
```

```{versionadded} v0.3.0
New ``replication="partition"`` variant for graph-partitioned
sharding (see [{doc}`sharding_topology`](sharding_topology.md)).
```

## Why this changed

`StaticArray` is the declaration site for **how** an array gets
materialised under sharding:

| Replication | What the wrapper does |
|---|---|
| `"replicate"` (default) | Every device gets the full array. |
| `"shard"` | Per-device slice along `shard_axis` (Cartesian — paired with `ShardedStencilNode`). |
| `"partition"` *(v0.3.0)* | Per-device slice via a partition assignment (graph — paired with `ShardedUnstructuredNode`). |

A bare array carries no replication policy, so the v0.2 coercion
path silently chose `"replicate"`.  That was harmless on a single
device but a bug magnet under sharding — the user couldn't see
why their stencil mask was bigger than expected per device.
v0.3.0 makes the declaration mandatory.

## What to do

### Wrap your `static_data` values explicitly

```{code-block} python
:caption: Before (v0.2.x; FutureWarning in v0.2.1, MigrationError in v0.3.0)
class MyNode(SimulationNode):
    @property
    def static_data(self) -> dict:
        return {
            "weights": self._weights_array,     # bare array
            "mask": self._mask_array,           # bare array
        }
```

```{code-block} python
:caption: After (v0.3.0+)
from maddening.core.static_data import StaticArray

class MyNode(SimulationNode):
    @property
    def static_data(self) -> dict:
        return {
            # Lookup table that every shard needs in full.
            "weights": StaticArray(self._weights_array),
            # Per-shard slab along axis 0 — paired with ShardedStencilNode
            # whose axis_map sends a mesh axis to the same spatial axis 0.
            "mask": StaticArray(
                self._mask_array,
                replication="shard", shard_axis=0,
            ),
        }
```

### Scalars, strings, and tuples pass through unchanged

Only array-like values (with `shape` AND `dtype`) trigger the
migration error.  Plain Python scalars, strings, and tuples stay
bare:

```{code-block} python
@property
def static_data(self) -> dict:
    return {
        "weights": StaticArray(self._w),    # wrapped
        "n_iters": 42,                      # scalar — no wrap needed
        "label": "demo",                    # string — no wrap needed
        "shape": (8, 8, 8),                 # tuple — no wrap needed
    }
```

### Nested structures stay unsupported

A list/dict of arrays under a single `static_data` key still raises
at `StaticArray.__post_init__`.  Unfold into multiple top-level keys:

```{code-block} python
# WRONG -- raises TypeError
"masks": [StaticArray(m1), StaticArray(m2)]

# RIGHT
"mask_a": StaticArray(m1),
"mask_b": StaticArray(m2),
```

### The new `"partition"` variant for unstructured sharding

If you're authoring an unstructured / graph-partitioned node for
{class}`~maddening.cloud.multigpu.sharded_unstructured.ShardedUnstructuredNode`,
declare any per-cell static data with the new `"partition"`
replication and a partition-assignment array:

```{code-block} python
import numpy as np
from maddening.core.static_data import StaticArray

class FVMNode(SimulationNode):
    def __init__(self, *, cell_volumes, partition_assignment, ...):
        self._cell_volumes = cell_volumes              # shape (n_global,)
        self._pa = partition_assignment.astype(np.int32)  # shape (n_global,)
        ...

    @property
    def static_data(self) -> dict:
        return {
            "cell_volumes": StaticArray(
                self._cell_volumes,
                replication="partition",
                partition_assignment=self._pa,
            ),
        }
```

By convention the partitioned axis is **axis 0** of `value` (the
global-cell axis).  The wrapper validates the assignment's length
matches.

See [{doc}`sharding_topology`](sharding_topology.md) for the full
unstructured-sharding contract.

## Auto-migration tooling

The {class}`~maddening.warnings.MigrationError` raised from
`coerce_static_data_value` carries structured detail:

```{code-block} python
from maddening.warnings import MigrationError

try:
    my_node.static_data_hash()
except MigrationError as err:
    print(f"api_name:    {err.api_name}")
    print(f"replacement: {err.replacement}")
    print(f"guide:       {err.migration_guide}")
```

The `api_name` follows the convention `bare-array-in-static_data
(<node>.<key>)`, so a tool can grep MIME / MICROROBOTICA for the
specific offending entries.

## Related

* {class}`~maddening.core.static_data.StaticArray` API reference.
* [{doc}`node_authoring`](node_authoring.md) — the
  `static_data` contract at the inner-node level.
* [{doc}`sharded_static_data`](sharded_static_data.md) — what the
  Cartesian `"shard"` variant delivers to `update_padded`.
* [{doc}`sharding_topology`](sharding_topology.md) — when to pick
  `"shard"` vs `"partition"`.
* {class}`~maddening.warnings.MigrationError` API reference.
