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
| Python Version | >=3.12 permitted; verified on 3.12 (the CI matrix) |
| JAX Version | jax>=0.10,<0.13 permitted; verified at 0.10.2, 0.11.2 (the versions CI installs) |
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
| MADD-ANO-001 | LBM GPU segfault on CUDA 12.2 + jaxlib 0.5.1 | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-002 | HeatNode CFL stability not enforced at runtime | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-003 | AdaptiveNode frozen-set gradient omits a first-order term at active-set switches | `major` | `context_dependent` | `open` | >=0.4.0.dev0 |
| MADD-ANO-004 | The sharded wrappers ignored a REST parameter write; PUT /graph/params answered 200 without changing the physics | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-005 | A coupling group's converged flag was a residual test, not a bound on the distance to the fixed point | `minor` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-006 | Non-finite numbers are written as non-standard JSON tokens | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-007 | HeatNode applies its Dirichlet boundary data at the first cell centre, not at the rod ends it documents | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-008 | HeatNode's 4th-order stencil converges at 1st order and is less accurate than the 2nd-order one | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-009 | HeatNode's documented CFL limit is the 2nd-order stencil's; the 4th-order stencil diverges below it | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-010 | A string spelling a non-finite token is refused by the JSON serialisers | `minor` | `context_dependent` | `open` | >=0.4.0.dev0 |
| MADD-ANO-015 | The ZeroMQ transports bound every interface with no authentication or encryption | `critical` | `safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-011 | BallNode's declared discretisation is forward Euler; the implementation is semi-implicit Euler | `minor` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-012 | HeartPumpNode samples its cardiac inflow at the end of the step, disagreeing with its own derivatives() | `minor` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-013 | Several nodes pin a float64 quantity to float32: HeartPumpNode's backpressure, and the rigid-body and HeatNode parameter and initial_state casts | `minor` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-014 | Every explicit integrator converges at 1st order when a time-varying boundary input is supplied once per step, including the one named 4th-order | `major` | `context_dependent` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-016 | cloud/_skypilot.py was written against a SkyPilot API older than the supported floor | `major` | `not_safety_relevant` | `partially_resolved` (in 0.4.0) | >=0.1.0 |
| MADD-ANO-017 | Under jax_enable_x64 a scan-shaped path refuses a float32 carry: implicit_euler_step in every release, and GraphManager's scans on a freshly compiled graph from 0.4.0 | `minor` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-018 | A calibrated params value reaches update() and cannot reach derivatives(), implicit_residual() or integrate_node() | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-019 | A coupling iteration that diverged to a non-finite state was reported residual=0.0, converged=True | `major` | `context_dependent` | `resolved` (in 0.4.0) | none |
| MADD-ANO-020 | LBMNode's Zou-He pressure boundary imposed rho_p + S_K instead of the prescribed density | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-021 | A node that branches on its own state inside update() returns the gradient of the branch it took; BallNode's gradient through a bounce omits the contact time and is exactly zero in the drop height | `major` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-022 | A mapped edge whose point reference names a grid derived from a trainable parameter keeps the constructor's geometry when that parameter is calibrated | `major` | `context_dependent` | `open` | >=0.4.0.dev0 |
| MADD-ANO-023 | The FMU TCP bridge authenticates no caller: any process that can reach its port can read, write, step and hold the model | `major` | `context_dependent` | `open` | >=0.4.0.dev0 |
| MADD-ANO-024 | PUT /graph/params accepted a value a node consumes at construction: reported and saved, never used by the running graph | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-025 | A sharded LBMNode computed a different model from the unsharded node: pressure faces imposed at every seam of a sharded face axis, and an edge-filled global halo by default | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-026 | With waveform_iterations > 1, coupling_diagnostics() reported only the last sweep's pass count: an earlier sweep stopped at max_iterations read as iterations=1 | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-027 | Sub-cycled coupling relaxes no waveform: waveform_iterations > 1 re-solves the same fixed point, and the sub-step interpolation runs between iterates, not across the step | `minor` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-028 | A sharded stencil node on a mesh axis of one device ran with periodic global halos whatever `boundary` said | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-029 | The default `"edge"` halo fill copied a halo two or more cells wide from the shard's first cells instead of repeating its edge cell | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-030 | A sharded HeatNode ignored its boundary temperature inputs: left_temperature and right_temperature never reached update_padded's rod ends | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-031 | A HeatNode at stencil_order=4 with no boundary input is not insulated: heat crosses a rod end that has no temperature given | `minor` | `context_dependent` | `open` | >=0.1.0 |
| MADD-ANO-032 | After a parameter write and compile(), the sharded wrappers kept computing with what they had built from the old value: a legacy node's constant, and ShardedStencilNode's halo width | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-033 | ShardedStencilNode stepped with dt rounded to float32 under jax_enable_x64 | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |

*33 anomalies registered.  15 have a defect reachable in this version — every entry whose `resolution_status` is not `resolved` or `duplicate`, which is 12 `open` plus 3 `partially_resolved` whose residual risk is still live.  The Affected Versions column is a PEP 440 specifier set read against this document's version; `none` marks a defect introduced and fixed within one development cycle, which no release carried.  The convention, and the gate that holds every range to it, are in the header of `known_anomalies.yaml`.  Rationale, workaround, affected components and verification evidence for each: `known_anomalies.yaml`.*
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
