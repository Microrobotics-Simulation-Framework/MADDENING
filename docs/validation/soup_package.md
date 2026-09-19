# MADDENING SOUP Package Document

The version and date of record for this document are the ones in
[§1 Software Identification](#1-software-identification) below, which are
generated from `pyproject.toml`.  A second copy in a header is a second thing
to forget on a release; there is deliberately no longer one here.

## 1. Software Identification

<!-- BEGIN GENERATED: software-identification -- scripts/generate_soup_tables.py; do not edit by hand -->
| Field | Value |
|---|---|
| Name | MADDENING |
| Full Name | Modular Automatic Differentiation and Data Enhanced Neural-network INteracting Graph |
| Version | 0.4.0.dev0 |
| Release Date | unreleased (development build) |
| Licence | LGPL-3.0-or-later |
| Source Repository | https://github.com/Microrobotics-Simulation-Framework/MADDENING |
| Python Version | >=3.11 |
| Base Dependencies | jax>=0.10,<0.13, jaxlib>=0.10,<0.13, lineax>=0.0.7, numpy>=1.24, pyyaml>=6.0 |
| Build System | hatchling |
| Install | `pip install maddening` |
<!-- END GENERATED: software-identification -->

Optional extras (GPU, server, visualization, FMI, USD, …) are listed in
`pyproject.toml` under `[project.optional-dependencies]`; only the base
dependencies above are installed by `pip install maddening`.

## 2. Functional Description

### Core Capabilities

- **Graph-based multi-physics simulation**: Compose, couple, and run physics simulations as directed graphs of nodes
- **Functional state pattern**: Pure functions, no shared mutable state, explicit data flow
- **{term}`JIT compilation`**: Full simulation step compiled to {term}`XLA` via {term}`JAX`
- **Automatic differentiation**: End-to-end differentiable simulation graphs
- **Neural surrogates**: Train and deploy neural network replacements for physics nodes
- **Adaptive timestepping**: {term}`Richardson extrapolation` with {term}`PI controller`
- **Parameter sweeps**: Batched simulation via `jax.vmap`
- **{term}`Coupling`**: {term}`Gauss-Seidel` iterative coupling via an early-exit `jax.lax.while_loop` with implicit-function-theorem differentiation

### Capabilities NOT Provided

- Clinical decision support
- Patient data processing
- Medical device functionality
- Real-time safety monitoring (HealthCheckNode provides infrastructure only; configuration and response logic are downstream responsibilities)
- Input validation or sanitization (assumes trusted inputs)

### Infrastructure Nodes

- **HealthCheckNode** (`maddening.nodes.health_check`): Base health monitor for execution-layer fault detection (NaN/Inf trapping, physical boundary checks). Downstream libraries instantiate and configure this node. Part of the MADDENING {term}`SOUP` dependency, not a separate SOUP item.

## 3. Known Anomalies

`known_anomalies.yaml` (same directory) is the registry of record.  The table
below is generated from it — it is a reading aid, not a second source, and a
stale copy fails CI rather than shipping.

<!-- BEGIN GENERATED: known-anomalies -- scripts/generate_soup_tables.py; do not edit by hand -->
| ID | Title | Severity | Safety Relevance | Status | Affected Versions |
|---|---|---|---|---|---|
| MADD-ANO-001 | LBM GPU segfault on CUDA 12.2 + jaxlib 0.5.1 | `major` | `context_dependent` | `resolved` | <=0.3.1 |
| MADD-ANO-002 | HeatNode CFL stability not enforced at runtime | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-003 | AdaptiveNode frozen-set gradient omits a first-order term at active-set switches | `major` | `context_dependent` | `open` | >=0.4.0 |
| MADD-ANO-004 | ShardedPointwiseNode ignored every parameter write; PUT /graph/params answered 200 without changing the physics | `major` | `context_dependent` | `resolved` (in 0.4.0) | 0.3.0, 0.3.1 |
| MADD-ANO-005 | A coupling group's converged flag was a residual test, not a bound on the distance to the fixed point | `minor` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-006 | Non-finite numbers are written as non-standard JSON tokens | `minor` | `context_dependent` | `open` | >=0.1.0 |

*6 anomalies registered, 3 open.  Rationale, workaround, affected components and verification evidence for each: `known_anomalies.yaml`.*
<!-- END GENERATED: known-anomalies -->

## 4. Verification Evidence

See [`framework_verification.md`](framework_verification.md) for the registered
verification benchmarks and the shape of the test suite.  The
machine-readable registry is `maddening.compliance.get_benchmark_registry()`.

## 5. IEC 62304 Lifecycle Activities

See `docs/regulatory/iec62304_mapping.md` for the full lifecycle mapping.

## 6. Configuration Management

- Version control: Git (GitHub)
- Release tags: semantic versioning (`vX.Y.Z`)
- {term}`SBOM`: CycloneDX format (planned, Phase 3)
- CI: GitHub Actions

## 7. Anomaly Management Policy

See CONTRIBUTING.md for the three-phase anomaly lifecycle and three-tier release gate model.

## 8. Dependencies

The base dependencies — what `pip install maddening` pulls in — are listed in
[§1 Software Identification](#1-software-identification), generated from
`pyproject.toml`.  Optional extras, and the transitive tree, are in
`pyproject.toml` itself.  A CycloneDX {term}`SBOM` is still planned (§6).
