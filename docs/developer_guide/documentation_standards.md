# Documentation Standards

This document defines the documentation standards for MADDENING. These standards support {term}`IEC 62304` {term}`SOUP` traceability and are enforced by CI where possible.

## Docstring Format

Use **NumPy-style** docstrings for all public classes and functions:

```python
def update(self, state, boundary_inputs, dt):
    """Advance the simulation state by one timestep.

    Parameters
    ----------
    state : dict
        Current state arrays.
    boundary_inputs : dict
        External inputs from coupled nodes.
    dt : float
        Timestep size in seconds.

    Returns
    -------
    dict
        Updated state arrays.
    """
```

## Docstring Examples Are Tests

Every `>>>` under `src/maddening/` is executed in CI, by the `compliance`
job:

```bash
python scripts/check_doctests.py          # add -v to see each example
```

An example is an API claim. Before this gate existed nothing ran them, and
one of the fifteen in the tree was materially wrong: `calibrate()` showed a
free-fall problem annotated `true g=-9.81`, ran with the default budget, and
asserted nothing — it actually stops at `g = -8.04` with `converged=False`.
Write examples so that a wrong one fails:

- **Assert something.** A snippet that only constructs objects proves they
  import. Print a value, or wrap a comparison in `bool(...)` /
  `round(..., n)` so the expected output is exact and stable.
- **No arrays, addresses or raw floats in expected output.** `repr` of a {term}`JAX`
  array carries dtype and formatting that change between releases; object
  addresses change every run. Compare instead:
  `>>> bool(jnp.allclose(x, 0.5 * jnp.ones(8)))`.
- **Round to a margin you can derive.** `round(float(g), 2)` is safe in the
  `calibrate` example because the tolerance bounds the answer to
  `|g + 9.81| < 2e-3`, well inside the rounding boundary. Rounding until the
  digits happen to match on your machine is not the same thing.
- **`filterwarnings = ["error"]` applies.** An example that emits a warning
  fails unless `pyproject.toml` already filters it.
- **`# doctest: +SKIP` fails the gate.** A skipped example is back to
  being prose that looks like a test. If an example genuinely cannot run,
  show it as a plain code block instead of behind `>>>`.
- **Stay inside the `ci` extra.** The `compliance` job installs `[ci,usd]`,
  a superset of what the test matrix installs; an example that needs
  anything further would pass there and fail everyone else.

The gate also fails if the collection *shrinks* — it scans the source for
docstrings containing examples and requires pytest to have collected a test
from each such file, on top of a floor on the number of `>>>` examples
actually executed (not docstrings: one docstring can hold many examples). A
doctest job that silently collects nothing exits 0 and gets counted as
coverage. Raise `MIN_EXAMPLES` in the script when you add examples.

## Code Blocks in the Docs Are Tests

Every fenced ` ```python ` block (also `py`, `python3`, and MyST
` ```{code-block} python `) in `docs/**/*.md`, `README.md`,
`CONTRIBUTING.md` and `SECURITY.md` runs in CI, in the `compliance` job:

```bash
pytest tests/compliance/test_docs_snippets.py
```

Each runnable block runs once, in a fresh `python` process, with a
timeout, its working directory a new temporary directory, `HOME` an empty
one, `JAX_PLATFORMS=cpu` and `MPLBACKEND=Agg`. The environment is built
from an allowlist, so no cloud credential or API token reaches it, and any
connection or name lookup that is not loopback is refused. A traceback
names the Markdown file and line. Deprecated API called from the snippet's
own code (`DeprecationWarning` or `FutureWarning`) fails it. The blocks run
in parallel, one process per CPU.

A block that is not meant to run says so, with a reason, in an HTML comment
on the line immediately before its opening fence and at the same
indentation:

```markdown
<!-- snippet: no-run, reason: fragment: lines inside update_padded -->
```

| Marker item | Meaning |
|---|---|
| `no-run, reason: <category>: <why>` | Not executed. The category is one of the list below; the explanation is required. |
| `continues` | Runs after the file's previous Python block, as one program in one process, and after whatever that block continues. A `no-run` block may carry it too: it is skipped and the chain runs on past it. The first block of a chain must run. |
| `requires: <module> ...` | Skipped, with the reason printed, where a module is not installed (`usd-core` is `pxr`). It runs in the job that installs it; keep such blocks inside the `ci` and `usd` extras. |

Items combine, separated by commas, with `reason:` last:
`<!-- snippet: continues, no-run, reason: fragment: ... -->`.

| Category | Use it for |
|---|---|
| `fragment` | An intentionally partial excerpt: a method body, a dict entry, or calls on a `gm` or {term}`node <Node>` the prose builds but the block does not |
| `pseudo-code` | Placeholders (`...`, `XXX`, `bounds={...}`) that stand for the reader's own code |
| `legacy` | A removed or pre-migration API, shown for comparison |
| `cloud` | Anything that names a cloud launch path (below) |
| `network` | Needs network access or a remote service |
| `gpu` | Needs a GPU or several devices |
| `external` | Needs a file, package, tool or configuration outside this repository |

Write a block so it runs if it is meant to: real imports, real keyword
names, sizes small enough to finish in a few seconds. Fix a failing
block's code rather than marking it `no-run`. A `no-run` block is not
unchecked either: its `maddening` imports must resolve, every keyword
argument it passes to a callable it imports from `maddening` must be one
that callable takes, and every `gm.<method>(...)` it calls must exist on
{codeterm}`GraphManager` with those keywords (every guide spells a graph `gm`). Only
the import statements are executed, never the block.

**Cloud launch paths never run.** A block that names `/cloud/launch`,
`launch_vm`, `CloudSession(`, `CloudLauncher(`, `.launch(`, `sky.` or the
`sky` CLI, `RUNPOD`, the `runpod` SDK, a cloud provider SDK, or `run_pod`
without `--dry-run` must be marked `no-run` with the `cloud` category. The
runner refuses such a block again before starting a process, and checks
its imports by reading the source instead of importing anything.

**How the marker renders.** An HTML comment is invisible on GitHub and in
the MyST/Sphinx HTML build, and dropped from LaTeX output, so a marker
changes nothing a reader sees. Two rules keep it that way, and the test
enforces both: the marker sits on the line directly before the fence, with
the fence's indentation (an unindented comment would end a list item), and
it contains no `--`, which an HTML comment may not.

The test also fails on a malformed or orphaned marker, an unknown item, a
reason on a block that runs, and a change in the number of blocks that run:
`EXPECTED_RUNNABLE` in the test is held equal to the count, so update it in
the commit that adds a runnable block or marks one `no-run`.

## Math in Code

- **Docstrings**: ASCII-art equations (e.g., `dT/dt = alpha * d^2T/dx^2`)
- **Algorithm guides**: LaTeX math blocks (`$$...$$`)
- **Math-heavy code exception**: mathematical variable names may use short names matching published formulas (e.g., `tau`, `f_eq`, `dx`, `rho`). This follows iMSTK's convention of exempting "highly math-based code" from standard naming rules to maintain correspondence with published formulas.

## Bibliography

All academic references go in `docs/bibliography.bib`. This is the single centralized reference store.

**In algorithm guides**: cite using Pandoc-style `[@Key]` syntax:
- Single citation: `[@Crank1975]`
- Multiple citations: `[@Crank1975; @LeVeque2007]`
- Each reference also gets a human-readable inline description
- CI validates all cited keys exist (`scripts/check_citations.py`)

**In code**: use the `Reference` type in `NodeMeta.references`:
<!-- snippet: no-run, reason: fragment: a NodeMeta keyword argument -->
```python
references=(
    Reference("Crank1975", "Analytical solutions for heat equation"),
)
```

## Algorithm Guide Documents

Every physics node must have a corresponding document in `docs/algorithm_guide/nodes/` following the template at `docs/algorithm_guide/nodes/_template.md`.

Required sections:

| Section | Purpose |
|---------|---------|
| Summary | 1-2 sentences |
| Governing Equations | Full math formulation (LaTeX) |
| Discretization | How continuous equations become discrete |
| Implementation Mapping | Every equation term traced to code (IEC 62304 Clause 5.4) |
| Assumptions and Simplifications | Numbered list of every assumption |
| Validated Physical Regimes | Quantitative parameter bounds with evidence |
| Known Limitations and Failure Modes | Feeds into SOUP anomaly assessment |
| Stability Conditions | Analytical/empirical stability bounds |
| State Variables | Field, shape, units, description |
| Parameters | Parameter, type, default, units, description |
| {term}`Boundary Inputs <Boundary inputs>` | Field, shape, default, description |
| References | `[@Key]` citations with inline descriptions |
| Verification Evidence | Links to benchmarks and test files |
| Changelog | Version, date, change |

## Implementation Mapping

The Implementation Mapping table is **mandatory** (IEC 62304 Clause 5.4 — detailed design traceability). It traces every term in the governing equations to a specific Python/JAX function:

```markdown
| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\alpha \nabla^2 T$ (diffusion) | `maddening.nodes.heat.HeatNode.update` | 2nd-order central FD |
| Time integration | Forward Euler in `maddening.nodes.heat.HeatNode.update` | 1st-order explicit |
```

Rules:
- Every governing equation term must appear — no silent omissions
- Terms handled by JAX primitives must document the primitive and calling convention
- CI validates all function names resolve to existing callables (`scripts/check_impl_mapping.py`)
- Every code span in the Implementation column is a fully qualified `maddening.*` name, except in a row whose Notes cell begins `JAX primitive` or `Third-party`
- A symbol the named class inherits rather than defines needs a Notes cell that begins ``Inherited from `BaseClass` ``; the gate checks the base and fails the marker once the class overrides the symbol
- Every symbol is a function, method or property, not a class: name the method the term is computed in.  A row that does mean a class (its constructor, say) begins its Notes cell ``Class `Name` ``; the gate fails the marker when no symbol in the row is that class

## Commit Message Convention

| Prefix | Meaning |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring (no behavior change) |
| `docs:` | Documentation only |
| `test:` | Test additions or changes |
| `perf:` | Performance improvement |
| `verify:` | Verification/validation evidence |
| `break:` | Breaking change |
| `deprecate:` | Deprecation notice |
| `security:` | Security-relevant change |

Commit messages should be concise (1-2 sentences) and focus on the "why" rather than the "what".

## Recording a change

A pull request never edits the shared record files: `CHANGELOG.md`, the
release notes under `docs/release_notes/`, the anomaly registry
`docs/validation/known_anomalies.yaml`, `_HIGHEST_ANOMALY_ID` in
`tests/compliance/test_soup_evidence.py`, the generated
`docs/validation/soup_package.md`, and the anomaly index in the development
record. Two open pull requests that both edited them always collided. Each
pull request writes its own small files under `docs/fragments/` instead,
named after its branch: lower-cased, with `/`, `_` and `.` turned into `-`.

| File | Holds |
|------|-------|
| `docs/fragments/changes/<stem>.yaml` | the changelog bullets and the release-notes text |
| `docs/fragments/anomalies/<stem>.yaml` | new registry entries, each under a provisional label |

`python scripts/fragments.py new --changes` (or `--anomaly`) writes either file
from a template for the branch you are on. `scripts/check_fragments.py`
checks both on every push, and `scripts/check_anomalies.py` checks the
entries of an anomalies fragment exactly as it checks the registry's own.

### The changelog and the release notes

A changes fragment maps the changelog's sections to bullets, and the release
notes' headings to Markdown:

```yaml
line: "0.4"
changelog:
  Fixed:
    - "**`HeatNode` keeps its timestep after a reset.** It fell back to the default."
release_notes:
  Fixed: |
    - **`HeatNode` keeps its timestep after a reset.** ...
```

`line` is the release line the tree was on when you wrote it, quoted. The
changelog's sections are `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`,
`Security`, `Verification` and `Known Anomalies`. The **Verification**,
**Security** and **Known Anomalies** sections are required for EU regulatory
workflows (IEC 62304, {term}`MDCG 2019-16`). The release notes' headings are
`Summary`, `What is new`, `What changes for existing code`, `Security`,
`Fixed`, `Known anomalies`, `API stability`, `Upgrade checklist` and
`For contributors`. The gate refuses any other section or heading.

Record every change that adds, changes, fixes or deprecates user-visible
functionality. A changelog bullet answers two questions only: what changed,
and what a user has to do about it. Measurements, file inventories, design
rationale and test counts belong in the pull request description, or in the
release-notes text. Internal test infrastructure (fuzz harnesses, Hypothesis
profiles, CI gating, tooling like this) is contributor-facing: it goes under
the notes' `For contributors` heading, or in a single changelog line at most.

`## [Unreleased]` in `CHANGELOG.md` stays empty. Before each release the
release assembly (`scripts/fragments.py assemble --release X.Y.Z --date D`,
run inside the version-bump pull request) writes the dated section and the
release notes from every fragment of the line, and deletes the fragments it
used.

### A new anomaly

A new registry entry goes in an anomalies fragment, in the registry's own
layout, with two differences:

```yaml
line: "0.4"
anomalies:
  - anomaly_id: "MADD-ANO-NEW-<stem>-1"
    title: "What a user sees"
    # ... every field the registry requires ...
    resolution_status: "open"
    release_note: >
      What a user sees and what to do about it: the release notes'
      Known-anomalies text.
```

- The id is a **label**, `MADD-ANO-NEW-<stem>-<n>`, with the file's own stem
  and `<n>` = 1, 2, … within the file. Name the entry by its label in tests,
  in claims inventories and in prose until it is numbered. Use this one
  spelling only: the assembler rewrites the label as an exact token.
- `release_note` is required for a reachable entry (one that is not
  `resolved` or `duplicate`). `record_index` is optional: the development
  record's index line, generated from the title when it is missing.

Do not pick a number. `scripts/fragments.py assemble --registry` runs as its
own pull request at quiet points on the numbering line, the first of
`RELEASE_LINES` in `scripts/fragments.py`. It appends the entries to the
registry in merge order, and moves `_HIGHEST_ANOMALY_ID` by exactly the number
it assigned. It records each label as a `# label:` comment after the entry's
id line, rewrites every label in the tree to its number, writes the index line
and the release note, and regenerates the SOUP table. Every entry from
`MADD-ANO-297` on carries that comment, so `scripts/check_fragments.py` fails
an entry numbered by hand. A fragment that was merged is never deleted by hand
(the ratchet in CI refuses it): close a wrong entry as `duplicate` or
`resolved` instead.

Changing an entry that is already numbered (resolving it, correcting it) is
still an edit to the registry. Its Known-anomalies text goes in a changes
fragment, under the `Known anomalies` heading.

Before writing a new entry, check that it does not exist already:
`python scripts/anom.py list --similar "a few words"` searches the registry
and every fragment. `scripts/anom.py show`, `refs` and `next` look entries up
by number or by label without opening the 21,000-line registry.

### Two release lines

While a 0.4.x line is open beside 0.5.0, fixes land on the 0.4 line first and
are merged forward. Only the numbering line numbers anomalies; an entry
written on the other line keeps its label until numbering moves (in 0.5.0's
release-assembly pull request). After each forward merge, run
`python scripts/fragments.py relabel`: it rewrites every label the registry
records to its number. `scripts/check_fragments.py` names each conflict a
forward merge can leave and its resolution.

A pull request opened before fragments existed moves its own registry
entries into a fragment with `python scripts/fragments.py from-branch --base
<its base>`.

## Anomaly Registry

Known bugs, limitations and failure modes go in
`docs/validation/known_anomalies.yaml`, by way of a fragment (above). See
[CONTRIBUTING.md](https://github.com/Microrobotics-Simulation-Framework/MADDENING/blob/main/CONTRIBUTING.md)
for the three-phase anomaly lifecycle and the release gate model.

## Versioning

MADDENING follows strict [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backward-compatible
- **PATCH** (0.0.X): Bug fixes, documentation

API {term}`stability levels <Stability level>`:

| Level | Meaning | SemVer Guarantee |
|-------|---------|-----------------|
| **{term}`stable <STABLE>`** | Breaking changes only in major versions | Full |
| **{term}`evolving <EVOLVING>`** | May change in minor versions with deprecation warning | One minor version notice |
| **{term}`experimental <EXPERIMENTAL>`** | May change without notice | None |
| **{term}`deprecated <DEPRECATED>`** | Scheduled for removal; the replacement is named in the docstring | Removed when the level it held allows: a `stable` surface in a major release, after two minor releases of notice; an `evolving` one in a minor release, after one; an `experimental` one in any minor release |

The table is a summary; [the deprecation policy](deprecation_policy.md) is the
rule, and also covers the two deprecated levels, `PROVISIONAL` and
`INTERNAL`.  A deprecated surface is not always kept until the next major
release: the CHANGELOG schedules `calibrate` and `tune_coupling_params` for
removal in 0.5.0, a minor release.
