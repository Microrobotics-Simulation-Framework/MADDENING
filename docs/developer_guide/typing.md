# Static Typing Policy

MADDENING is type-checked with [pyright](https://github.com/microsoft/pyright).
The check is introduced in two phases so that the annotation work can be
scoped to the API surface that is actually frozen, instead of chasing
whole-tree cleanliness.

| Phase | When | What |
|---|---|---|
| **1** (done) | v0.4.0 development | `pyrightconfig.json` in *basic* mode over `src/maddening`; a `typecheck` CI job that is **visible but non-blocking**; the baseline below.  No source annotations are changed in this phase. |
| **2 (current)** | v0.4.0, after the STABLE list settled | Annotate the surfaces tagged `@stability(StabilityLevel.STABLE)` (the [stability report](stability_report.md) is the authoritative list, regenerated and equality-checked in CI; 17 STABLE-tagged surfaces in 11 modules at v0.4.0) *and* the internal packages that refactors touch, replace bare `dict` parameters with `TypedDict`/`Mapping` types, ship a `py.typed` marker ([PEP 561](https://peps.python.org/pep-0561/)), and make the pyright check **blocking in two tiers**: tier 1 (`__init__.py`, `core`, `nodes`, `fmi`, `cloud`, `sysid`, `serialization`, `testing`, `compliance`) must be at zero errors; tier 2 (`viz`, `usd`, `api`, `surrogates`, which sit on untyped or optional third-party libraries) gets every public signature annotated, so a `py.typed` package never exposes an `Any`-returning public call, while its module bodies are only ratcheted: the error count may not rise above the recorded baseline. |

Whole-tree cleanliness is explicitly *not* a goal of either phase.

## Running it

```bash
# In an environment with the package and its extras installed
# (pip install -e ".[ci]" or ".[dev]" -- both pull in pyright):
pyright                                   # plain diagnostics, exit 1 on errors
python scripts/typing_baseline.py         # per-rule / per-file summary (exit 0, or 2 if the run cannot be trusted)
python scripts/typing_baseline.py --markdown --top 20

# Without installing pyright (uses the interpreter whose site-packages
# should be used for import resolution):
uvx pyright --pythonpath /path/to/.venv/bin/python
python scripts/typing_baseline.py --pyright "uvx pyright" -- --pythonpath /path/to/.venv/bin/python
```

`scripts/typing_baseline.py` runs `pyright --outputjson`, prints the totals,
the counts per rule and the files with the most errors, and forwards
whatever pyright wrote to stderr.  Its exit status separates *what pyright
found* from *whether pyright ran*:

| Exit | Meaning | CI |
|---|---|---|
| 0 | pyright completed; no errors, or neither `--fail-on-errors` nor `--tier` given | green |
| 1 | pyright completed and reported errors (`--fail-on-errors`), or a package exceeded its tier ceiling (`--tier`) | **the job fails** (phase 2; under phase 1 this was `continue-on-error`) |
| 2 | **infrastructure failure**: the numbers cannot be trusted, so none are printed.  The pyright command is missing; pyright exited with a code other than 0/1 (3 = `pyrightconfig.json` could not be parsed, in which case it silently analysed the whole tree with the default config); stdout is not JSON; the `summary` block is missing or `filesAnalyzed` is below `--min-files` (default 1: a missing `include` path is reported by pyright on stderr only, with exit 0 and an empty, valid JSON document); a core import (`jax`, `numpy`, `yaml`) did not resolve; or more than `--max-missing-imports` (default 40; the baseline has 23, all optional extras) `reportMissingImports` diagnostics were reported.  With `--tier`, also: the tier file is missing, names no such tier, has an empty ceiling map (a gate over nothing would pass on anything), or records a **different pyright release** from the one running | the step **fails the job** |

The import-resolution guard exists because a mis-resolved interpreter does
not make the count go *up*: without `jax`/`numpy` most attribute and
argument errors become `Unknown` and disappear (measured: 395 -> 230 errors,
201 missing imports, nothing on stderr).  For the same reason a
`--pythonpath PYTHON` given after `--` is checked before pyright runs: the
path must exist and `PYTHON -c "import numpy"` must succeed; pyright itself
accepts a non-existent path without a word.

`--json FILE` summarises a stored run instead of invoking pyright (the same
checks apply to the stored document), which is how
`tests/test_typing_baseline.py` tests it without pyright installed; the
failure paths are tested with a fake pyright executable.  CI uses it to
run pyright **once** and gate both tiers off the stored document, so the
two verdicts cannot disagree about what was measured.

### The two-tier gate

```bash
pyright --outputjson > pyright.json
python scripts/typing_baseline.py --json pyright.json --tier tier1
python scripts/typing_baseline.py --json pyright.json --tier tier2
```

`--tier NAME` reads `typing_tiers.json` at the repository root, restricts
every count to that tier's top-level packages, and compares each
package's error count against the ceiling committed there.  Exit 1 means
at least one package is over; the verdict on stderr names it and the
excess, and prints the *slack* on every package that is under so a
ratchet can be tightened when it becomes free.

Ceilings are **per package** on purpose: a single total would let a
regression in `viz` hide behind an improvement in `api`.

The package root module is a tier of one.  `_package_of` gives a module
directly under `src/maddening` its own key, so `src/maddening/__init__.py`
is the ceiling key `__init__.py` -- and until 2026-09-20 that key was in
neither tier.  Because `--tier` filters the diagnostics to the tier's
packages *before* comparing, a key in no tier is not reported as
uncovered; it simply vanishes, and all three typing steps pass on any
number of errors in that file (measured: 500 synthetic errors, three
green steps).  That file is where the `if TYPE_CHECKING:` re-exports live
and is what every `from maddening import X` resolves through for a
`py.typed` consumer, and `tests/test_lazy_reexports.py` is `ast`-based by
design -- so pyright is the only thing that can see a stale name in it.
It is committed at **0**, which is its measured count (pyright 1.1.414).

The file also records `pyright_version` and `environment`, and a run
under a different pyright release is an *infrastructure* failure (exit
2), not a gate failure: every release changes diagnostics, so the
comparison would be meaningless in either direction.  Bump the
`pyright==` pin in the `[ci]` extra, re-measure, and update
`typing_tiers.json` and this document together -- never one alone.  The
environment matters just as much: installing `usd-core` or `pyvista`
resolves imports the CI environment does not, which makes real errors
visible and pushes the counts **up** (measured: 423 against 270 on the
same commit).  `typing_tiers.json` names the exact set the ceilings were
taken under.

## `pyrightconfig.json`

JSON has no comments, so the settings are explained here.

| Key | Value | Why |
|---|---|---|
| `include` | `["src/maddening"]` | The package only.  `tests/`, `benchmarks/`, `docs/`, `scripts/` are excluded for now; `src/maddening/examples` (61 example scripts shipped inside the package) is excluded because it is not an API surface and would dominate the numbers. |
| `typeCheckingMode` | `"basic"` | Start where the signal is actionable.  Measured distance to the stricter modes is given below. |
| `pythonVersion` | `"3.11"` | The floor in `pyproject.toml` (`requires-python = ">=3.11"`), so syntax/stdlib checks match the oldest supported interpreter. |
| `pythonPlatform` | `"Linux"` | Matches CI; avoids platform-conditional stdlib noise. |
| `reportMissingImports` | `"warning"` | Optional extras (`gi`/PyGObject, `pygfx`, `rendercanvas`, `skimage`, `fsspec`, `cupy`, plus `pxr`, `sky`, `zmq`, `fmpy` ... when those extras are not installed) are imported lazily or behind `try:`.  They must not count as errors in an environment without those extras. |
| `reportMissingModuleSource` | `"warning"` | Same reason, for packages that ship only stubs. |
| `reportMissingTypeStubs` | `false` | Several dependencies (`jax` internals, `pyvista`, `sky`, `pxr`) have no stubs; that is not something this repository can fix. |
| `venvPath` / `venv` | *not set* | CI installs the package and pyright into the same environment, so `python` on `PATH` is the right interpreter; the script's import-resolution guard (above) is what turns a wrong interpreter into a failed job instead of a lower count.  Locally, pass `--pythonpath` (see above) rather than hard-coding a path in the shared config. |

## Baseline (phase 2, current)

Measured with pyright 1.1.414 on 2026-09-20 (branch `feat/pep561-phase2`,
forked from `release/0.4.0`), against `pip install -e ".[ci]"` on Python
3.11 with `jax==0.10.2` -- the same environment the CI job builds.  141
files analysed.  The committed ceilings are in `typing_tiers.json`; these
are the numbers they were taken from.

| | errors | warnings |
|---|---|---|
| **before this phase** | 270 | 100 |
| **after** | **165** | **23** |
| tier 1 (`__init__.py`, `core`, `nodes`, `fmi`, `cloud`, `sysid.py`, `serialization`, `testing`, `compliance`, `transport_auth.py`, `warnings.py`) | 103 -> **0** | |
| tier 2 (`api`, `surrogates`, `viz`, `usd`) | 167 -> **165** | |

The 77 `reportUnsupportedDunderAll` warnings are gone: every name behind a
lazy `__getattr__` is now re-exported under `if TYPE_CHECKING:`.  The 23
that remain are all `reportMissingImports` for optional extras that the
CI environment does not install (`gi`, `pxr`, `pygfx`, `rendercanvas`,
`skimage`, `pyvista`, `cupy`).

### Per-package ceilings (tier 2)

| package | errors | why it is ratcheted, not cleaned |
|---|---|---|
| `api` | 61 | matplotlib artists and figures typed `X \| None` in the frame renderers |
| `surrogates` | 55 | optional network layers in `architectures/*.py`, equinox internals.  Lowered from 58 by the phase-3 `TypedDict` pass (below) |
| `viz` | 45 | matplotlib, pygfx, pyvista, rendercanvas -- mostly absent, none with stubs |
| `usd` | 1 | `pxr` has no stubs and is not installed in CI |

### Diagnostics by rule (after)

| severity | rule | count |
|---|---|---|
| error | reportOptionalMemberAccess | 106 |
| error | reportAttributeAccessIssue | 29 |
| warning | reportMissingImports | 23 |
| error | reportArgumentType | 10 |
| error | reportGeneralTypeIssues | 10 |
| error | reportCallIssue | 5 |
| error | reportOptionalOperand | 2 |
| error | reportIndexIssue | 1 |
| error | reportOperatorIssue | 1 |
| error | reportOptionalSubscript | 1 |

### Top files by error count (after)

| errors | file |
|---|---|
| 34 | `src/maddening/api/frame_renderer.py` |
| 31 | `src/maddening/viz/history_viewer.py` |
| 20 | `src/maddening/surrogates/architectures/deeponet.py` |
| 19 | `src/maddening/api/frame_renderer_3d.py` |
| 15 | `src/maddening/surrogates/architectures/fno.py` |
| 9 | `src/maddening/viz/backends/pygfx_viewer.py` |
| 8 | `src/maddening/surrogates/architectures/mlp.py` |
| 6 | `src/maddening/surrogates/training/callbacks.py` |
| 6 | `src/maddening/surrogates/weights/checkpoint.py` |
| 5 | `src/maddening/api/vessel_renderer.py` |

All 106 remaining `reportOptionalMemberAccess` are in tier 2, and they are
a real class of latent bug rather than noise: a renderer's matplotlib
artist or a viewer's `_plotter` is `None` until `setup()` runs, so calling
`update()` first is an `AttributeError` instead of a clear error.  Cleaning
them is control-flow work, not annotation work, and is deliberately left
for a later pass (see *What phase 2 did not do*).

## PEP 561: shipping the marker

`src/maddening/py.typed` is an empty file; `pyproject.toml` force-includes
it into the wheel.  Without it a conforming type checker **must ignore
every annotation in the package**, so all of the work above would be
invisible to a consumer of `pip install maddening`.  Measured on a
throwaway project importing `GraphManager` and `HeatNode` and making four
deliberate mistakes:

| checker | without `py.typed` | with |
|---|---|---|
| mypy 1.x | 0 of 4 found (2 `import-untyped` notices instead) | **4 of 4** |
| pyright, `useLibraryCodeForTypes: false` | **0 of 4 found** | **4 of 4** |
| pyright, default settings | 4 of 4 | 4 of 4 |

pyright's default `useLibraryCodeForTypes: true` infers from library
*source* and so masks a missing marker; mypy, and any checker following
PEP 561 to the letter, does not.  That is why the claim is tested against
the built artifact rather than the configuration:
`tests/test_packaging_py_typed.py` builds a real wheel and asserts
`maddening/py.typed` is a member of it, together with the two USD schema
resources that ride the same `force-include` mechanism.

## Baseline (phase 1, historical)

Measured with pyright 1.1.414 on 2026-09-16 (branch `feat/typing-pep561`,
forked from `release/0.4.0`), import resolution against a full dev
environment (everything in `[ci]` installed except `gi`, `pygfx`,
`rendercanvas`, `skimage`, `fsspec`, `cupy`).  133 files analysed.

| Mode | Errors | Warnings | Note |
|---|---|---|---|
| **basic** (configured) | **395** | 94 | the numbers below |
| standard | 466 | 94 | +69 `reportPossiblyUnboundVariable`, +2 `reportFunctionMemberAccess` |
| strict | 11754 | 18 | (the `__all__` warnings become errors in strict) dominated by `reportUnknown*` / `reportMissingParameterType` (every unannotated parameter and every value that flows from one) |

Of the 395 basic-mode errors, **33 are in 7 of the 12 modules that define
a STABLE surface** (`core/graph_manager.py` 20, `core/edge.py` 3,
`cloud/multigpu/iterative_solver.py` 3, `fmi/model_description.py` 3,
`cloud/multigpu/sharded_unstructured.py` 2, `cloud/multigpu/sharded_node.py` 1,
`nodes/heat.py` 1; the other five STABLE modules are clean); that is the
phase-2 workload for the public contract.  The tier-1 scope (`core`,
`nodes`, `fmi`, `cloud`, the module `sysid.py`, `serialization`, `testing`,
`compliance`) holds **83** of the 395 (`core` 51, `cloud` 17, `nodes` 5,
`testing` 5, `fmi` 4, `sysid.py` 1; `serialization` and `compliance` 0);
the remaining **312** are in `viz` (90), `usd` (88), `api` (76) and
`surrogates` (58), mostly `reportOptionalMemberAccess` /
`reportAttributeAccessIssue` against optional or untyped third-party
libraries (pxr, pygfx, matplotlib, fastapi extras), which is why those
packages are ratcheted rather than cleaned.  (Counts re-measured by an
independent audit of the phase-1 merge against the same pyright run;
per-package figures are errors per top-level package under
`src/maddening`.)  Decided 2026-09-16: whole-tree strictness is not the
goal, but the internals that refactors move through are, because pyright
catches broken call sites there just as it does on the public surface.

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

Delivered on `feat/pep561-phase2` (2026-09-20) except where noted.

1. ~~Regenerate the [stability report](stability_report.md)~~ -- done; the
   STABLE list had settled before this branch started, and the report is
   regenerated and equality-checked in CI, so a level change in the
   freeze round (release sequence phase 2.9) is caught rather than silent.
2. **Annotate tier 1 to zero and every public signature in tier 2** --
   done (103 -> 0; 219 public signatures across `api`, `usd`, `viz` and
   `surrogates`).  The `TypedDict` conversion in the original wording is
   **deliberately not done here**: it changes the representation of the
   data structure every node touches, and the original motivation was
   pytree layout and performance as much as typing.  It is phase 3 item 1
   and needs its own review.  Where an annotation wanted one, the
   parameter is `Mapping[str, Any]` / `dict[str, Any]` with a comment
   naming the shape, so phase 3 can find every site by grepping for those
   notes.
3. **`if TYPE_CHECKING:` re-exports for the lazy `__getattr__` tables** --
   done, for all seven packages; `tests/test_lazy_reexports.py` pins it
   statically so it cannot rot between pyright runs, in **both**
   directions: a name in `__all__` that no checker can see, and a name a
   checker can see that the runtime lazy table has no entry for.  The
   second one type-checks perfectly and raises `AttributeError`, which
   with the marker shipped is the direction that reaches a consumer.  A
   `TYPE_CHECKING` import of a symbol that does not exist is not
   reachable by an `ast`-based check at all; pyright is the backstop for
   that, which is why every one of the seven packages -- including the
   package root module -- has to be inside a tier.
4. **`src/maddening/py.typed` and the hatch `force-include`** -- done, and
   asserted against a built wheel rather than against the configuration
   (see *PEP 561: shipping the marker*).
5. **Blocking in two tiers** -- done: `continue-on-error` is gone, the
   `typecheck` job runs pyright once and gates each tier off the stored
   run, and the ceilings live in `typing_tiers.json`.  The baseline is
   recorded above.

## Phase 3 item 1: the PEP 589 `TypedDict` pass

Delivered on `typing/pep589-typed-dicts` (2026-09-21).  Phase 2 left 39
`TypedDict candidate (phase 3)` markers in 13 modules (`surrogates` 33,
`viz` 4, `usd` 2); none on the core graph/node path.  **Report the two numbers separately: 11 markers were
converted to a `TypedDict`, 28 were given a named alias instead.**

The dividing question is whether the keys are known statically:

- **Converted (11 markers, 7 `TypedDict`s plus two private
  required-key bases).**  `TrainState` and
  `TrainMetrics` in `surrogates.types`; `TerminalRendererConfig`,
  `TimeSeriesPlotConfig`, `SceneConfig` + `SceneObjectSpec` in the viz
  backends; `TubeConfig` in `viz.usd_viewer`.  Each key set is fixed by
  this package, so a checker rejects a misspelling and a wrong value
  type.
- **Aliased (28 markers).**  `StateDict`, `MutableStateDict`,
  `BatchedStateDict`, `SpecDict`, `FieldValues`, `WeightOverrides`,
  `DerivFn` and `Integrator` in
  `maddening.surrogates.types`, and `NodeStateDict` in
  `maddening.usd.live_stage`.  These sit on generic call sites -- an
  architecture's `forward`, a physics loss, a USD prim updater -- whose
  keys are whichever fields the caller's node declares.  `TypedDict`
  requires statically-known keys and cannot express that.  Every such
  site carries a one-line note saying so, greppable as
  `not a TypedDict`.

Two things to know before adding another one:

- **A `TypedDict` is a `dict` at runtime** -- same class, same
  `PyTreeDef`, invisible to `jax.jit`.  This pass changed no
  representation and no trace count.  Making one of these a
  `NamedTuple`, a `register_dataclass` or an `equinox.Module` *would*
  change the pytree; that is a different piece of work.
- **Split required from optional with a base class, not
  `NotRequired`.**  Under `from __future__ import annotations` (which
  all three affected modules use) CPython cannot see a `NotRequired`
  wrapper and reports every key in `__required_keys__`.  Measured on
  3.12.3; a checker gets it right either way, the runtime does not.

The pass was verified by making the checker the test: a probe calling
the converted surfaces with three misspelled keys, three wrong value
types and one invalid `Literal` draws **0 errors on `release/0.4.0` and
7 on this branch**.  It also found a real defect --
`MatplotlibSceneRenderer.setup()` passed a scene object's `"x"` straight
to `patches.Circle`, so the documented "or a state field name" spelling
died with `ConversionError` before the first frame (fixed, with
`tests/viz/test_scene_renderer_object_spec.py`).

### What phase 2 did not do

- **The PEP 589 `TypedDict` conversion** (checklist item 2, above) --
  done in phase 3, above.
- **The 106 remaining `reportOptionalMemberAccess`**, all in tier 2.
  These are *control-flow* changes -- a matplotlib artist or a PyVista
  plotter that is `None` until `setup()` -- not annotations, and several
  are latent `AttributeError`s rather than checker noise.  The ratchet
  holds the count; lowering it is its own piece of work.
- **Tier-1 public signatures are not all annotated.**  Tier 1's contract
  is *zero errors*, which is a different property: an unannotated
  parameter is not an error in basic mode.  145 tier-1 public functions
  still have no return annotation, so a `py.typed` consumer can still
  reach an implicitly-`Any` call there.  That is the obvious next
  ratchet, and it is larger than it looks (779 public definitions).
