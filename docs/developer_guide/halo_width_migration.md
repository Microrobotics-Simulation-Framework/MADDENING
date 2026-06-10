---
orphan: false
---

# `halo_width` migration guide (v0.2 → v0.3.0)

```{versionchanged} v0.2
Per-axis halo widths via
{meth}`SimulationNode.halo_width() <maddening.core.node.SimulationNode.halo_width>`,
returning a ``dict[int, int]`` (``{axis: width}``).  Pointwise nodes
return ``{}``; stencil nodes return a non-empty dict.  The legacy
boolean ``requires_halo`` property was retained as a derived
fallback (``bool(self.halo_width())``) and any subclass overriding
``requires_halo`` directly emitted a
{class}`FutureWarning` at class-definition time.
```

```{versionchanged} v0.3.0
The ``requires_halo`` property and the ``__init_subclass__`` compat
shim were **removed**.  Subclasses that override ``requires_halo``
instead of ``halo_width`` now raise
{class}`~maddening.warnings.MigrationError` at class-definition time.
```

## Why this changed

Pencil decomposition (the multi-GPU sharding path —
{class}`~maddening.cloud.multigpu.sharded_node.ShardedStencilNode`
in v0.2.1, plus
{class}`~maddening.cloud.multigpu.sharded_unstructured.ShardedUnstructuredNode`
added in v0.3.0) needs to know each axis's halo width, not just
"do you need a halo at all".  The boolean was load-bearing
once — when stencils were 1-D and one axis per node — but it
silently lost information the moment we shipped 3-D LBM.

The v0.2 release kept the boolean as a derived fallback for
back-compat and emitted a {class}`FutureWarning`.  v0.3.0 closes
the deprecation cycle.

## What to do

### If you override `halo_width` already

Nothing.  You already write to the new API; the v0.3.0 hard-removal
doesn't affect you.

### If you override `requires_halo` only

Replace the property override with a {meth}`halo_width` method:

```{code-block} python
:caption: Before (v0.2.x; FutureWarning in v0.2, MigrationError in v0.3.0)
class MyStencilNode(SimulationNode):
    @property
    def requires_halo(self) -> bool:
        return True

    def update_padded(self, state_padded, boundary_inputs, dt, **kwargs):
        ...
```

```{code-block} python
:caption: After (v0.3.0+)
class MyStencilNode(SimulationNode):
    def halo_width(self) -> dict[int, int]:
        # axis → cells per side.  Two-cell halo on axis 0 (a 5-point
        # stencil), one-cell halo on axis 1 (a 3-point stencil).
        return {0: 2, 1: 1}

    def update_padded(self, state_padded, boundary_inputs, dt, **kwargs):
        ...
```

### Pointwise nodes

Pointwise nodes (no spatial neighbour access) return the empty dict:

```{code-block} python
class MyPointwiseNode(SimulationNode):
    def halo_width(self) -> dict[int, int]:
        return {}
```

If your node never overrode `halo_width` *or* `requires_halo`, the
base class already returns `{}` and you don't need to do anything.

### Auto-migration tooling

The {class}`~maddening.warnings.MigrationError` raised at class-
definition time carries structured detail rather than just a string
message — `err.api_name`, `err.affected_class`, `err.replacement`,
`err.migration_guide`.  Tools can introspect these without grepping
the message:

```{code-block} python
import importlib, traceback
from maddening.warnings import MigrationError

try:
    importlib.import_module("my_pkg.legacy_node")
except MigrationError as err:
    print(f"Migrate {err.affected_class.__qualname__}: "
          f"replace requires_halo with {err.replacement}")
```

## Detecting affected subclasses

A grep over downstream packages catches most cases.  For each match,
either replace with `halo_width` or delete the override entirely if
the parent class already returns the right value:

```{code-block} console
$ grep -rn "def requires_halo\|requires_halo =\|@property" --include="*.py" | grep -A1 "requires_halo"
```

MIME's CI runs a grep gate on this — see the v0.3.0 plan's §B6.

## Related

* [{doc}`node_authoring`](node_authoring.md) — the inner-node
  contract `halo_width` participates in.
* [{doc}`sharding_topology`](sharding_topology.md) — how stencil
  vs unstructured nodes consume halos differently.
* {class}`~maddening.warnings.MigrationError` API reference.
