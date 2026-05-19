---
orphan: false
---

# Edge validation: migration guide (v0.2 → v0.2.1)

```{versionadded} v0.2
Shape, dtype, and unit checks at {meth}`GraphManager.compile() <maddening.core.graph_manager.GraphManager.compile>`,
emitted as four warning subclasses of {class}`~maddening.warnings.EdgeValidationWarning`.
```

```{versionchanged} v0.2.1
The {class}`~maddening.warnings.ShapeMismatchWarning` and
{class}`~maddening.warnings.DtypeMismatchWarning` will be **promoted to
hard `EdgeValidationError`** exceptions in v0.2.1.  Unit warnings stay
as warnings (units are advisory, not load-bearing).
```

In v0.2 we added compile-time validation that walks every
{class}`~maddening.core.graph_manager.EdgeSpec` and compares the source
field's runtime shape, dtype, and units against the target node's
{class}`~maddening.core.node.BoundaryInputSpec`.  The intent is that a
lab newcomer wiring a 20-edge graph gets *every* mistake flagged in one
{meth}`~maddening.core.graph_manager.GraphManager.compile` pass, instead
of debugging one failure at a time at runtime.

This page is for users with existing graphs that started seeing
warnings after upgrading to v0.2.

## The four warning kinds

All four are subclasses of {class}`maddening.warnings.EdgeValidationWarning`
so callers can catch them in one `with warnings.catch_warnings()` block.

| Warning | Trigger | Common fix |
|---|---|---|
| `ShapeMismatchWarning` | Source field has shape `(5,)` but target spec declares `(3,)`. | Add a `transform=lambda x: x[:3]` on the edge, or fix one of the two shapes. |
| `DtypeMismatchWarning` | Source emits `int32`, target spec expects `float32`. | Add a `transform=lambda x: x.astype(jnp.float32)` on the edge, or pick a consistent dtype at the producer. |
| `UnitMismatchWarning` | Edge declares `source_units="kN"`, target spec says `expected_units="N"`. | Add `target_units="N"` or wrap with a units-converting transform. |
| `EdgeValidationWarning` (bare) | Future overflow category for warnings that don't fit the three above. | None currently emitted. |

## The escape hatch: any `transform=` on the edge suppresses the check

When an edge has a `transform=` callable, both shape and dtype checks
are skipped — we cannot statically reason about what the transform
will produce.  Use this when the mismatch is intentional (e.g. picking
a boundary slice from a 1-D field).

```python
gm.add_edge(
    source_node="heat_rod",
    target_node="ball",
    source_field="temperature",        # shape (N,)
    target_field="ambient_temperature",  # spec shape ()
    transform=lambda T: T[N // 2],      # → scalar at midpoint
)
```

No shape warning fires because the transform is doing the reshape.

## How `compile()` decides what's an error vs a warning

| Issue | v0.2 behaviour | v0.2.1 behaviour |
|---|---|---|
| Edge references a non-existent node | `RuntimeError` from `compile()` | (unchanged) |
| Source field not in source node's state | `RuntimeError` | (unchanged) |
| Shape mismatch (no transform) | `ShapeMismatchWarning` | `ShapeMismatchError` (raised) |
| Dtype mismatch (no transform) | `DtypeMismatchWarning` | `DtypeMismatchError` (raised) |
| Unit mismatch | `UnitMismatchWarning` | (unchanged — units stay advisory) |
| Disconnected node | `UserWarning` | (unchanged) |

## Common patterns

### "I'm slicing a boundary from a 1-D field"

```python
# Before v0.2: silent at compile, surprise at runtime
gm.add_edge("heat", "neighbor", "temperature", "wall_T")  # shape (N,) vs ()

# After v0.2: ShapeMismatchWarning
# Fix: add the transform that was implicit
gm.add_edge(
    "heat", "neighbor", "temperature", "wall_T",
    transform=lambda T: T[-1],
)
```

### "I'm passing kilonewtons but the target wants newtons"

```python
gm.add_edge(
    "thruster", "body", "force", "F_external",
    source_units="kN",
    transform=lambda x: x * 1000.0,   # kN → N
    target_units="N",
)
```

The transform suppresses the shape/dtype warnings; the explicit
`target_units` declaration matches the target spec so no unit
warning fires either.

### "I want my edge to keep working even after v0.2.1"

Fix the mismatch.  The compile-time-error path is intentional: it
catches a class of bug (passing the wrong field by name) that
currently fails at the first `step()` with a runtime shape error
deep in the traced JIT.

### "I have a synthetic test that *needs* the warning"

```python
import pytest
import warnings
from maddening.warnings import ShapeMismatchWarning

with pytest.warns(ShapeMismatchWarning):
    gm.compile()
```

In v0.2.1, swap the warning class for the corresponding error
class and the `pytest.warns` for `pytest.raises`.

### "I'm running MIME experiments and don't want CI to fail on warnings"

MADDENING ships `filterwarnings = ["error", ...]` in
`pyproject.toml` for the *internal* test suite — downstream
projects don't inherit that setting.  If you have your own
`filterwarnings = ["error"]` and want to grandfather edge
validation:

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "error",
    "ignore::maddening.warnings.ShapeMismatchWarning",   # remove for v0.2.1
    "ignore::maddening.warnings.DtypeMismatchWarning",   # remove for v0.2.1
    "ignore::maddening.warnings.UnitMismatchWarning",
]
```

Remove the shape/dtype lines before v0.2.1 lands.

## Aggregation: all problems in one pass

`compile()` walks the full `validate()` issue list and emits one
warning per problem.  A 20-edge graph with three different mismatch
classes produces three warnings, not one — by design.

```python
import warnings
from maddening.warnings import EdgeValidationWarning

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", EdgeValidationWarning)
    gm.compile()

for w in caught:
    print(type(w.message).__name__, ":", w.message)
```

## Tracking issue for the v0.2.1 flip

When v0.2.1 lands the `ShapeMismatchWarning` and
`DtypeMismatchWarning` flips, the change set will:

1. Add `ShapeMismatchError(EdgeValidationError)` and
   `DtypeMismatchError(EdgeValidationError)` to
   `maddening.warnings`.
2. Replace the `warnings.warn(issue, ShapeMismatchWarning)` calls
   in `GraphManager.compile()` with `raise
   ShapeMismatchError(issue)`.
3. Aggregate errors in a single `ExceptionGroup` so the lab-newcomer
   "see every problem at once" property is preserved.
4. Update this page's first admonition to `versionchanged` and move
   the "v0.2 → v0.2.1" caveat to "fixed in v0.2.1".

Track the cut on the GitHub `v0.2.1` milestone — see also the
`compile_time_edge_validation` test file at
`tests/core/test_edge_validation.py` which doubles as the migration
test bed.
