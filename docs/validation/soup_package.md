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

Optional extras (GPU, server, visualization, USD, …) are listed in
`pyproject.toml` under `[project.optional-dependencies]`; only the base
dependencies above are installed by `pip install maddening`.  FMU export
(`maddening.fmi`) needs no extra.  The resolved dependency tree of the base
install, and of the extras listed in §6, is in the SBOMs described there.

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
| MADD-ANO-034 | LBMNode's wall_mask was dropped by every save/reload: the reloaded graph ran with no walls | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.1.0, <0.4.0 |
| MADD-ANO-035 | ShardedUnstructuredNode reads a global-order state as partition layout on a balanced partition not in global order | `major` | `context_dependent` | `open` | >=0.3.0 |
| MADD-ANO-036 | A config round trip dropped a node's sharding with no word: the reloaded graph ran unsharded | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-037 | ShardedUnstructuredNode accepted a Cartesian stencil node and stepped it wrong, every cell, with no error | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-038 | ShardedStencilNode edge-filled a sharded static's global halos under a periodic wrapper | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.1, <0.4.0 |
| MADD-ANO-039 | ShardedUnstructuredNode dropped the cells a node had past the layout's count, and gave a node no way to leave padding out of an integral | `major` | `context_dependent` | `resolved` (in 0.4.0) | >=0.3.0, <0.4.0 |
| MADD-ANO-040 | The sharded wrappers placed a domain integral carried in the state like a grid field: an integral-emitting node could not run in a graph | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.2.1, <0.4.0 |
| MADD-ANO-041 | halo_exchange ignored a per-axis boundary key naming no exchanged mesh axis, and that axis took the edge fill | `minor` | `context_dependent` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |
| MADD-ANO-042 | ShardedStencilNode accepted an empty axis_map and sharded nothing | `minor` | `not_safety_relevant` | `resolved` (in 0.4.0) | >=0.2.0, <0.4.0 |

*42 anomalies registered.  16 have a defect reachable in this version — every entry whose `resolution_status` is not `resolved` or `duplicate`, which is 13 `open` plus 3 `partially_resolved` whose residual risk is still live.  The Affected Versions column is a PEP 440 specifier set read against this document's version; `none` marks a defect introduced and fixed within one development cycle, which no release carried.  The convention, and the gate that holds every range to it, are in the header of `known_anomalies.yaml`.  Rationale, workaround, affected components and verification evidence for each: `known_anomalies.yaml`.*
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
- {term}`SBOM`: CycloneDX 1.6 JSON, one per covered install, in
  `docs/validation/sbom/` (below)
- CI: GitHub Actions

### Software Bill of Materials

Each SBOM is the environment that a clean install of MADDENING's wheel
resolved to.  `scripts/generate_sbom.py` builds the wheel from the tree,
installs it into a new, isolated virtual environment, and runs
`cyclonedx-py` against that environment from a separate tool environment, so
the tool is not in the SBOM.  MADDENING is the root component, with its
version, purl and licence.  Every other installed distribution is a
component with its name, version, `pkg:pypi` purl and the licence its
metadata declares.  The dependency graph is included, and the root's edges
are exactly the direct dependencies `pyproject.toml` declares for that
install.

| File | Install | What it covers |
|---|---|---|
| `maddening-0.4.0.dev0-core.cdx.json` | `pip install maddening` | The base dependencies: what every user gets, and the SOUP items of §1 |
| `maddening-0.4.0.dev0-server.cdx.json` | `pip install maddening[server]` | The network-facing bundle: the HTTP/WebSocket API, the ZeroMQ transports, terminal and matplotlib rendering, zstd frames.  A superset of the `api`, `network`, `terminal`, `viz` and `compression` extras |
| `maddening-0.4.0.dev0-surrogates.cdx.json` | `pip install maddening[surrogates]` | Neural surrogate training (`optax`).  A trained surrogate replaces a physics node, so this code is in the computed result |
| `maddening-0.4.0.dev0-usd.cdx.json` | `pip install maddening[usd]` | OpenUSD stage read and write (`usd-core`, a binary wheel that bundles OpenUSD's C++ libraries) |

**Why these installs, and not the others.**  These are the installs whose
code runs in a deployed simulation: the base install, the network surfaces
a deployment exposes, the surrogate path that feeds computed results, and
the geometry import path.  The rest are left out on purpose:

- `cuda12` and `tpu` install hardware runtimes, and GPU is not a verified
  configuration: CI runs on CPU (MADD-ANO-001).  On 2026-09-25 `cuda12`
  resolved fifteen more wheels on Linux: JAX's two CUDA plugin wheels and
  thirteen NVIDIA CUDA libraries.  A GPU deployment records its own SBOM.
  The cloud Docker image installs `jax[cuda12]` with `.[server]`, and
  `python scripts/generate_sbom.py --extra cuda12+server --output-dir <dir>`
  records that combination.
- `all`, `dev` and `ci` describe a developer's workstation, not a deployment.
  On 2026-09-25 `all` resolved 176 packages, including SkyPilot's cloud SDKs,
  PyGObject and the display stack.  That is the shape of the whole-machine
  `sbom.json` which sat at the repository root from 2026-03 until 0.4.0
  removed it.  It listed one machine's packages, not MADDENING's dependency
  set.
- The cloud extras (`runpod`, `lambda`, `aws`, `gcp`, `cloud`, `cloud-all`)
  are launch tooling on the operator's machine, not code in the simulation's
  process.  `viz3d`, `gpu-viz` and `streaming` are display-side
  visualisation.  `verify` and `sbom` are tooling, and `ift` is empty.

**What an SBOM records, and what it does not.**  Each file records the
Python version and platform it was resolved on (the PEP 508 marker
variables and the wheel platform tag, as `maddening:sbom:*` properties) and
the resolution cutoff (`maddening:sbom:exclude-newer`): only distributions
uploaded before it were considered.  All three decide the transitive
versions.  jaxlib, numpy and scipy ship per-platform wheels, and a later
date resolves newer releases inside the declared ranges.  An SBOM here
records one resolution, on Python 3.12 and `linux-x86_64`, as of the
recorded date.  It is not a lock file: `pip install maddening` resolves
afresh, and it is not the environment any CI run installed (see the warning
in `framework_verification.md`).  A deployment records its own environment
by running `generate_sbom.py` for its own install, or `cyclonedx-py` on its
own environment.

**Consistency.**  `scripts/check_sbom.py` runs in CI through
`tests/compliance/test_sbom_check.py`, with no network.  It fails if a
covered install has no SBOM at the version `pyproject.toml` declares, or if
the directory holds any other SBOM.  It fails if a direct dependency
`pyproject.toml` declares for an install is missing from that install's
SBOM, or is at a version outside its declared range.  It fails if a SOUP
item §1 lists is missing, or is at a version outside the `pyproject.toml`
range.  It also fails if a component has no purl, or a purl that disagrees
with it, and if the file was edited after generation: the `serialNumber`
is derived from the content.  Each failure names the discrepancy.

**Determinism.**  Components and the dependency graph are sorted, keys are
written sorted, `metadata.timestamp` is the resolution cutoff (or
`SOURCE_DATE_EPOCH` when set), and the `serialNumber` is a UUIDv5 of the
content.  Regenerating with the same cutoff on the same platform therefore
writes the same bytes.  To compare SBOMs generated on different days,
`python scripts/check_sbom.py --normalise <file>` prints one without the
three date-dependent fields (`serialNumber`, `metadata.timestamp`, the
cutoff).

**At release, the SBOMs are regenerated from the tagged build.**  The files
above are generated from this tree at the version `pyproject.toml` declares,
which their names carry.  They are not the release SBOM.  The release step,
done by the maintainer:

1. On the release commit, after the version bump and before the tag,
   change the version in the file names in the table above.
2. Run `python scripts/generate_sbom.py` (it needs `uv` and network
   access).  It builds the wheel from that tree, resolves each install
   afresh, writes `maddening-<version>-<install>.cdx.json` for the new
   version, removes the previous version's files, and runs the check.
   Until both steps are done the check fails on the version bump, which
   makes them hard to skip.
3. Commit the SBOMs with the release commit, tag that commit, and attach
   the four `.cdx.json` files to the GitHub release as assets.

## 7. Anomaly Management Policy

See CONTRIBUTING.md for the three-phase anomaly lifecycle and three-tier release gate model.

## 8. Dependencies

The base dependencies — what `pip install maddening` pulls in — are listed in
[§1 Software Identification](#1-software-identification), generated from
`pyproject.toml`.  The optional extras are declared in `pyproject.toml`.
The transitive tree, with the version and declared licence of every
package, is in the CycloneDX {term}`SBOM` of each covered install (§6): the
core SBOM for the base install, and one each for the `server`, `surrogates`
and `usd` extras.
