---
orphan: false
---

# Edge validation: migration guide (v0.2 → v0.2.1)

```{versionadded} v0.2
Shape, dtype, and unit checks at {meth}`GraphManager.compile() <maddening.core.graph_manager.GraphManager.compile>`,
emitted as warning subclasses of {class}`~maddening.warnings.EdgeValidationWarning`.
```

```{versionchanged} v0.2.1
Shape and dtype mismatches were **promoted to hard errors** —
{meth}`~maddening.core.graph_manager.GraphManager.compile` now raises
an :class:`ExceptionGroup` of
{class}`~maddening.warnings.ShapeMismatchError` /
{class}`~maddening.warnings.DtypeMismatchError`.  Unit mismatches
stay as warnings (units are documentation, not contract).
```

```{warning}
**Semver carve-out.**  v0.2.1 includes one breaking change under
strict semver — the edge-validation warning→error flip described
here.  We pre-announced this in v0.2.0 release notes and held the
deprecation calendar; the carve-out is the deliberate decision to
ship the flip as a PATCH release rather than bumping to 0.3.0,
because: (a) the change was published in advance, (b) the migration
path is documented on this page, and (c) the deprecated
``*Warning`` aliases stay importable through v0.2.x for one release
cycle.  If your CI pins ``maddening<0.3``, expect this change; if
you depend on the old warning behaviour, see "Backwards-compat
escape hatch" below.
```

In v0.2 we added compile-time validation that walks every
{class}`~maddening.core.graph_manager.EdgeSpec` and compares the source
field's runtime shape, dtype, and units against the target node's
{class}`~maddening.core.node.BoundaryInputSpec`.  The intent is that a
lab newcomer wiring a 20-edge graph gets *every* mistake flagged in one
{meth}`~maddening.core.graph_manager.GraphManager.compile` pass, instead
of debugging one failure at a time at runtime.

This page is for users with existing graphs that started seeing
warnings after upgrading to v0.2 and want to migrate to v0.2.1's
error-on-mismatch behaviour.

## The validation surface (current — v0.2.1)

| Issue | Behaviour |
|---|---|
| Edge references a non-existent node | `RuntimeError` from `compile()` |
| Source field not in source node's state | `RuntimeError` |
| Shape mismatch (no transform) | `ShapeMismatchError` raised inside an `ExceptionGroup` |
| Dtype mismatch (no transform) | `DtypeMismatchError` raised inside an `ExceptionGroup` |
| Unit mismatch | `UnitMismatchWarning` (advisory; never raises) |
| Disconnected node | `UserWarning` (suppressed in single-node graphs) |

All shape/dtype errors detected during a single `compile()` call are
aggregated and raised together — the lab-newcomer "see every problem
at once" property is preserved by the `ExceptionGroup` shape.

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

No `ShapeMismatchError` is raised because the transform is doing the
reshape.

## Catching the errors

`ExceptionGroup` is a builtin on Python 3.11+; MADDENING declares
`exceptiongroup` as a backport dependency on 3.10.  Two equivalent
ways to handle the group:

### Python 3.11+ — `except*`

```python
from maddening.warnings import ShapeMismatchError, DtypeMismatchError

try:
    gm.compile()
except* ShapeMismatchError as eg:
    for err in eg.exceptions:
        print("shape:", err)
except* DtypeMismatchError as eg:
    for err in eg.exceptions:
        print("dtype:", err)
```

### Python 3.10 (or version-agnostic) — explicit iteration

```python
from maddening.warnings import (
    EdgeValidationError, ExceptionGroup,
    ShapeMismatchError, DtypeMismatchError,
)

try:
    gm.compile()
except ExceptionGroup as eg:
    for err in eg.exceptions:
        if isinstance(err, ShapeMismatchError):
            print("shape:", err)
        elif isinstance(err, DtypeMismatchError):
            print("dtype:", err)
        elif isinstance(err, EdgeValidationError):
            print("other validation error:", err)
```

## Common patterns

### "I'm slicing a boundary from a 1-D field"

```python
# Before v0.2: silent at compile, surprise at runtime
gm.add_edge("heat", "neighbor", "temperature", "wall_T")  # shape (N,) vs ()

# v0.2+: ShapeMismatchError on compile() — fix by adding the
# transform that was previously implicit
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

The transform suppresses the shape/dtype checks; the explicit
`target_units` declaration matches the target spec so no
`UnitMismatchWarning` fires either.

### "I have a synthetic test that needs the error"

```python
import pytest
from maddening.warnings import ShapeMismatchError, ExceptionGroup

with pytest.raises(ExceptionGroup) as exc_info:
    gm.compile()
assert any(isinstance(e, ShapeMismatchError) for e in exc_info.value.exceptions)
```

### "I had `pytest.warns(ShapeMismatchWarning)` in my test suite"

The `ShapeMismatchWarning` and `DtypeMismatchWarning` classes remain
importable in v0.2.1 as deprecated aliases — but MADDENING no longer
emits them.  Update to the error form above; the alias is removed in
v0.3.

### "I'm running MIME experiments and don't want CI to fail"

MADDENING ships `filterwarnings = ["error", ...]` in `pyproject.toml`
for the *internal* test suite — downstream projects don't inherit
that setting.  In v0.2, the typical downstream override was:

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "error",
    "ignore::maddening.warnings.ShapeMismatchWarning",   # no-op since v0.2.1
    "ignore::maddening.warnings.DtypeMismatchWarning",   # no-op since v0.2.1
    "ignore::maddening.warnings.UnitMismatchWarning",
]
```

The first two lines are no-ops as of v0.2.1 (no warning is emitted to
ignore) — safe to remove.  The `UnitMismatchWarning` line stays.

## Aggregation: all problems in one pass

`compile()` walks the full `validate()` issue list and accumulates
every shape/dtype mismatch into a single `ExceptionGroup`.  Unit
mismatches still emit as warnings *before* the raise, so they show
up in `caught` records too.

```python
import warnings
import pytest
from maddening.warnings import (
    EdgeValidationWarning, ExceptionGroup,
    ShapeMismatchError, DtypeMismatchError, UnitMismatchWarning,
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always", EdgeValidationWarning)
    with pytest.raises(ExceptionGroup) as exc_info:
        gm.compile()

errors = exc_info.value.exceptions
unit_warns = [w for w in caught
              if issubclass(w.category, UnitMismatchWarning)]
```

A 20-edge graph with three different shape mismatches, two different
dtype mismatches, and one unit mismatch produces six items total
(five errors in the group + one warning) — by design.

## Backwards-compat escape hatch

The deprecated `*Warning` classes (`ShapeMismatchWarning`,
`DtypeMismatchWarning`, `EdgeValidationWarning`) stay importable
through v0.2.x.  Nothing in MADDENING emits them in v0.2.1, but they
are not removed — downstream `pytest.warns` references resolve and
just never fire.  In v0.3 the aliases are removed (see the v0.3.0
plan's compat-hygiene bucket).
