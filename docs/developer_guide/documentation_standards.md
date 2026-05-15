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

## Anomaly Registry

Known bugs, limitations, and failure modes go in `docs/validation/known_anomalies.yaml`. See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the three-phase anomaly lifecycle and release gate model.

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
| **deprecated** | Scheduled for removal | Removed in next major |
