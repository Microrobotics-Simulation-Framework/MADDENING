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
- **No arrays, addresses or raw floats in expected output.** `repr` of a JAX
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
| Boundary Inputs | Field, shape, default, description |
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

## CHANGELOG.md

Follow [Keep a Changelog](https://keepachangelog.com/) with these sections:

```markdown
## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Verification
### Security
### Known Anomalies
```

The **Verification**, **Security**, and **Known Anomalies** sections are required for EU regulatory workflows (IEC 62304, {term}`MDCG 2019-16`).

Update the changelog with every commit that adds, changes, fixes, or deprecates user-visible functionality. Empty sections can be omitted in the commit but must be present in release notes.

### Entry style

Add exactly **one contiguous block, at most three lines**, at the **top** of the relevant subsection, and never edit or reorder existing lines: branches merge in parallel, and appending at the top is what keeps the conflicts trivial.

An entry answers two questions only — what changed, and what a user has to do about it. Measurements, file inventories, design rationale, test counts and "why" paragraphs belong in the pull request description and in the per-version release notes under `docs/release_notes/`; link to the release notes from the top of `## [Unreleased]`. If three lines are not enough, the surplus is release-notes material.

Internal test infrastructure — fuzz harnesses, Hypothesis profiles and depth tiers, property-test packages, CI gating — is contributor-facing rather than user-facing. It belongs in the release notes' engineering section, or in a single changelog line at most.

## Anomaly Registry

Known bugs, limitations, and failure modes go in `docs/validation/known_anomalies.yaml`. See [CONTRIBUTING.md](https://github.com/Microrobotics-Simulation-Framework/MADDENING/blob/main/CONTRIBUTING.md) for the three-phase anomaly lifecycle and release gate model.

## Versioning

MADDENING follows strict [Semantic Versioning](https://semver.org/):

- **MAJOR** (X.0.0): Breaking API changes
- **MINOR** (0.X.0): New features, backward-compatible
- **PATCH** (0.0.X): Bug fixes, documentation

API stability levels:

| Level | Meaning | SemVer Guarantee |
|-------|---------|-----------------|
| **stable** | Breaking changes only in major versions | Full |
| **provisional** | May change in minor versions with deprecation warning | One minor version notice |
| **experimental** | May change without notice | None |
| **deprecated** | Scheduled for removal; the replacement is named in the docstring | Removed when the level it held allows: a `stable` surface in a major release, after two minor releases of notice; an `evolving` or `provisional` one in a minor release, after one; an `experimental` one in any minor release |

The table is a summary; [the deprecation policy](deprecation_policy.md) is the
rule, and also covers `evolving` and `internal`.  So a deprecated surface is not
always kept until the next major release: the CHANGELOG schedules
`calibrate` and `tune_coupling_params` for removal in 0.5.0, a minor release.
