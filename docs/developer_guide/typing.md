# Static Typing Policy

MADDENING is type-checked with [pyright](https://github.com/microsoft/pyright).
The check is introduced in two phases so that the annotation work can be
scoped to the API surface that is actually frozen, instead of chasing
whole-tree cleanliness.

| Phase | When | What |
|---|---|---|
| **1 (current)** | v0.4.0 development | `pyrightconfig.json` in *basic* mode over `src/maddening`; a `typecheck` CI job that is **visible but non-blocking**; the baseline below.  No source annotations are changed in this phase. |
| **2** | after the 0.4.0 API freeze | Annotate the surfaces tagged `@stability(StabilityLevel.STABLE)` (see the [stability report](stability_report.md); 19 STABLE-tagged surfaces in 12 modules at the time of writing) *and* the internal packages that refactors touch, replace bare `dict` parameters with `TypedDict`/`Mapping` types, ship a `py.typed` marker ([PEP 561](https://peps.python.org/pep-0561/)), and make the pyright check **blocking in two tiers**: tier 1 (`core`, `nodes`, `fmi`, `cloud`, `sysid`, `serialization`, `testing`, `compliance`) must be at zero errors; tier 2 (`viz`, `usd`, `api`, `surrogates`, which sit on untyped or optional third-party libraries) gets every public signature annotated, so a `py.typed` package never exposes an `Any`-returning public call, while its module bodies are only ratcheted: the error count may not rise above the recorded baseline. |

Whole-tree cleanliness is explicitly *not* a goal of either phase.

## Running it

```bash
# In an environment with the package and its extras installed
# (pip install -e ".[ci]" or ".[dev]" -- both pull in pyright):
pyright                                   # plain diagnostics, exit 1 on errors
python scripts/typing_baseline.py         # per-rule / per-file summary (exit 0)
python scripts/typing_baseline.py --markdown --top 20

# Without installing pyright (uses the interpreter whose site-packages
# should be used for import resolution):
uvx pyright --pythonpath /path/to/.venv/bin/python
python scripts/typing_baseline.py --pyright "uvx pyright" -- --pythonpath /path/to/.venv/bin/python
```

`scripts/typing_baseline.py` runs `pyright --outputjson`, prints the totals,
the counts per rule and the files with the most errors, and always exits 0
unless `--fail-on-errors` is given (that flag is what the CI step uses so
that the step is red when there are errors, while `continue-on-error`
keeps the workflow green).  `--json FILE` summarises a stored run instead of
invoking pyright, which is how `tests/test_typing_baseline.py` tests it
without pyright installed.

## `pyrightconfig.json`

JSON has no comments, so the settings are explained here.

| Key | Value | Why |
|---|---|---|
| `include` | `["src/maddening"]` | The package only.  `tests/`, `benchmarks/`, `docs/`, `scripts/` are excluded for now; `src/maddening/examples` (61 example scripts shipped inside the package) is excluded because it is not an API surface and would dominate the numbers. |
| `typeCheckingMode` | `"basic"` | Start where the signal is actionable.  Measured distance to the stricter modes is given below. |
| `pythonVersion` | `"3.11"` | The floor in `pyproject.toml` (`requires-python = ">=3.11"`), so syntax/stdlib checks match the oldest supported interpreter. |
| `pythonPlatform` | `"Linux"` | Matches CI; avoids platform-conditional stdlib noise. |
| `reportMissingImports` | `"warning"` | Optional extras (`gi`/PyGObject, `pygfx`, `rendercanvas`, `skimage`, `fsspec`, `cupy`, plus `pxr`, `sky`, `zmq`, `lineax`, `fmpy` ... when those extras are not installed) are imported lazily or behind `try:`.  They must not count as errors in an environment without those extras. |
| `reportMissingModuleSource` | `"warning"` | Same reason, for packages that ship only stubs. |
| `reportMissingTypeStubs` | `false` | Several dependencies (`jax` internals, `pyvista`, `sky`, `pxr`) have no stubs; that is not something this repository can fix. |
| `venvPath` / `venv` | *not set* | CI installs the package and pyright into the same environment, so `python` on `PATH` is the right interpreter.  Locally, pass `--pythonpath` (see above) rather than hard-coding a path in the shared config. |

## Baseline (phase 1)

Measured with pyright 1.1.414 on 2026-09-16 (branch `feat/typing-pep561`,
forked from `release/0.4.0`), import resolution against a full dev
environment (everything in `[ci]` installed except `gi`, `pygfx`,
`rendercanvas`, `skimage`, `fsspec`, `cupy`).  133 files analysed.

| Mode | Errors | Warnings | Note |
|---|---|---|---|
| **basic** (configured) | **395** | 94 | the numbers below |
| standard | 466 | 94 | +69 `reportPossiblyUnboundVariable`, +2 `reportFunctionMemberAccess` |
| strict | 11754 | 18 | (the `__all__` warnings become errors in strict) dominated by `reportUnknown*` / `reportMissingParameterType` (every unannotated parameter and every value that flows from one) |

Of the 395 basic-mode errors, **30 are in the 12 modules that define a
STABLE surface** (`core/graph_manager.py` 20, `core/edge.py` 3,
`cloud/multigpu/iterative_solver.py` 3, `cloud/multigpu/sharded_unstructured.py` 2,
`cloud/multigpu/sharded_node.py` 1, `nodes/heat.py` 1); that is the phase-2
workload for the public contract.  The tier-1 packages (`core`, `nodes`,
`fmi`, `cloud`, `sysid`, `serialization`, `testing`, `compliance`) hold
roughly 140 of the 395; the remaining ~250 are in `viz`, `usd`, `api` and
`surrogates`, mostly `reportOptionalMemberAccess` / `reportAttributeAccessIssue`
against optional or untyped third-party libraries (pxr, pygfx, matplotlib,
fastapi extras), which is why those packages are ratcheted rather than
cleaned.  Decided 2026-09-16: whole-tree strictness is not the goal, but
the internals that refactors move through are, because pyright catches
broken call sites there just as it does on the public surface.

### Diagnostics by rule (basic)

| severity | rule | count |
|---|---|---|
| error | reportAttributeAccessIssue | 135 |
| error | reportOptionalMemberAccess | 111 |
| error | reportArgumentType | 88 |
| warning | reportUnsupportedDunderAll | 76 |
| error | reportCallIssue | 22 |
| warning | reportMissingImports | 18 |
| error | reportGeneralTypeIssues | 13 |
| error | reportRedeclaration | 9 |
| error | reportIndexIssue | 5 |
| error | reportAssignmentType | 3 |
| error | reportOptionalCall | 2 |
| error | reportOptionalOperand | 2 |
| error | reportUndefinedVariable | 2 |
| error | reportOperatorIssue | 1 |
| error | reportOptionalSubscript | 1 |
| error | reportPrivateImportUsage | 1 |

What the big buckets are:

- `reportUnsupportedDunderAll` (all 76 warnings): every one is in a package
  `__init__.py` that lists lazily-imported names in `__all__` and resolves
  them through a [PEP 562](https://peps.python.org/pep-0562/) module
  `__getattr__` (`maddening`, `maddening.cloud`, `maddening.surrogates`,
  `maddening.viz`, ...).  pyright cannot see through `__getattr__`; the
  phase-2 fix for the STABLE surface is `if TYPE_CHECKING:` re-exports next
  to the lazy table.
- `reportOptionalMemberAccess` (111) and most `reportAttributeAccessIssue`
  (135): attribute access on values typed `X | None` (matplotlib artists
  and figures in `api/frame_renderer*.py`, `viz/history_viewer.py`; optional
  network layers in `surrogates/architectures/*.py`) and on `pxr`/`pyvista`
  objects that have no stubs.  Concentrated in viz/USD/surrogate modules,
  none of which is STABLE.
- `reportUndefinedVariable` (2): string annotations (`"Path"`,
  `"subprocess.CompletedProcess"`) whose names are only imported inside the
  function body.  Harmless at runtime, trivially fixed in phase 2.

### Top 15 files by error count (basic)

| errors | file |
|---|---|
| 53 | src/maddening/viz/history_viewer.py |
| 45 | src/maddening/usd/live_stage.py |
| 34 | src/maddening/api/frame_renderer.py |
| 28 | src/maddening/api/frame_renderer_3d.py |
| 20 | src/maddening/core/graph_manager.py |
| 20 | src/maddening/surrogates/architectures/deeponet.py |
| 18 | src/maddening/usd/geometry.py |
| 16 | src/maddening/usd/writer.py |
| 15 | src/maddening/surrogates/architectures/fno.py |
| 12 | src/maddening/viz/backends/pyvista_live.py |
| 11 | src/maddening/api/vessel_renderer.py |
| 9 | src/maddening/viz/backends/pygfx_viewer.py |
| 9 | src/maddening/viz/usd_viewer.py |
| 8 | src/maddening/core/coupling/mapping.py |
| 8 | src/maddening/surrogates/architectures/mlp.py |

### Annotation coverage (measured before phase 1)

194 modules including examples, ~1498 function definitions, 57 % with a
return annotation, 235 parameters annotated as bare `dict` (the phase-2
`TypedDict` candidates: node states, edge payloads, node/graph config).

## Phase 2 checklist

1. After the 0.4.0 API freeze, regenerate the [stability report](stability_report.md);
   the STABLE list is the contract that must be *right*, the tier-1
   packages are the scope that must be *clean*.
2. Annotate tier 1 (`feat/typing-core`, `feat/typing-nodes`,
   `feat/typing-fmi-cloud`) and every public signature in tier 2 (the
   bodies there stay as they are); introduce `TypedDict`s for the recurring bare
   `dict` shapes (node `state`, `boundary_inputs`, `static_data`,
   `gm.params` sections, spec maps) and use them in the internals too.
3. Add `if TYPE_CHECKING:` re-exports for public names behind lazy
   `__getattr__` tables.
4. Add `src/maddening/py.typed` and the hatch `force-include` entry for it.
5. `chore/py-typed`: make the tier-1 run blocking (drop
   `continue-on-error`, keep `--fail-on-errors`) and add the tier-2
   ratchet (`scripts/typing_baseline.py` compares the count against a
   committed baseline and fails on growth).  Record the new baseline with
   `python scripts/typing_baseline.py --markdown` here.
