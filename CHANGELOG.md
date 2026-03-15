# Changelog

All notable changes to MADDENING will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Additional sections per release: **Verification**, **Security**, and **Known Anomalies**.

## [Unreleased]

### Added
- Coupling convergence infrastructure: per-field mixed atol/rtol norm (`convergence_norm="mixed"`), convergence diagnostics (`diagnostics=True`), Aitken delta-squared acceleration (`acceleration="aitken"`), fixed under-relaxation (`acceleration="fixed"`), and Jacobi iteration mode (`iteration_mode="jacobi"`)
- `coupling_acceleration` module with standalone JAX-traceable residual norms, state flatten/unflatten, and acceleration functions
- `GraphManager.coupling_diagnostics()` method for retrieving iteration counts and final residuals
- IQN-ILS quasi-Newton coupling acceleration (`acceleration="iqn-ils"`) with Aitken fallback, pre-allocated matrices for fori_loop compatibility, and automatic column management
- Subcycling within coupling groups (`subcycling=True`) for mixed-timestep coupling with linear/constant boundary interpolation
- Spatial interpolation map factories in `interface_mapping` module: `nearest_neighbor_1d`, `linear_interpolation_1d`, `rbf_interpolation` (4 kernels), `conservative_projection_1d`
- `auto_couple()` and `add_coupling_group()` accept `**kwargs` forwarded to `CouplingGroup`
- Coupling examples: acceleration comparison, Jacobi vs Gauss-Seidel, subcycling, spatial interpolation, convergence diagnostics
- IQN-ILS/IMVJ auto interface-field detection: `flatten_coupled_state` accepts `fields` parameter to accelerate only coupling-edge fields, reducing V/W matrix size for nodes with many internal DOFs
- IQN-IMVJ multi-timestep Jacobian reuse (`acceleration="iqn-imvj"`, `jacobian_reuse=N`): warm-starts V/W from previous timestep for faster convergence
- Interface residual convergence norm (`convergence_norm="interface"`): checks coupling-edge values between iterations instead of full state change
- Quadratic subcycling boundary interpolation (`boundary_interpolation="quadratic"`): Lagrange interpolation through three successive iteration values
- Waveform relaxation for subcycled groups (`waveform_iterations=N`): repeats coupling block to improve boundary data quality
- Flux-based coupling: `SimulationNode.compute_boundary_fluxes()` exposes derived quantities (heat flux, spring force) consumable via edges; `SimulationNode.boundary_input_spec()` declares expected inputs with `BoundaryInputSpec` descriptors
- `EdgeSpec.additive` flag: edges with `additive=True` accumulate values instead of overwriting, enabling multi-source force/flux coupling
- `coupling_helpers` module: `add_value_coupling`, `add_flux_coupling`, `add_dirichlet_neumann_pair`, `add_symmetric_value_coupling`, `add_robin_coupling`, `check_conservation`
- `BoundaryInputSpec` dataclass and `boundary_input_spec()` on HeatNode, SpringDamperNode, BallNode, RigidBody2DNode
- `compute_boundary_fluxes()` on HeatNode (left/right heat flux) and SpringDamperNode (spring force)
- Flux coupling demo and node authoring guide sections on flux coupling patterns
- `TransformRegistry` with `@register_transform` decorator for named, serializable edge transforms; built-in transforms (`extract_first`, `extract_last`, `negate`, `scale`, `identity`); `GraphManager.add_edge` accepts string transform names
- `scripts/check_transforms.py` CI validation script for string transform references
- Gradient health audit: verified `jax.grad` finite through 1000-step coupled rollouts for springs, heat rods, and multi-physics systems
- Parameter recovery baseline: gradient-based recovery of spring stiffness and damping from trajectory data (inline physics, proving differentiability concept)

### Fixed
- Subcycling dividers were inverted: fast nodes now correctly take multiple sub-steps while slow nodes take one step
- Coupling diagnostics were lost in multi-rate graphs when step counter overwrote `_meta`
- `maddening.compliance` namespace with schema types, anomaly validator, and CLI
- `NodeMeta` dataclass with `hazard_hints`, `validated_regimes`, `implementation_map` fields
- `StabilityLevel` and `UQReadiness` enums
- `@verification_benchmark` decorator and `ValidationBenchmark` registry
- `@stability` decorator (identity decorator; functional machinery in Phase 4)
- `HealthCheckNode` for execution-layer fault detection
- `NodeMeta` attached to all existing nodes (BallNode, TableNode, SpringDamperNode, RigidBody2DNode, HeatNode, LBMPipeNode)
- `AuditLogger` with `NullSink` and `JSONFileSink`
- `SimulationProvenance` for reproducibility tracking
- `UncertaintySpec` and `UncertainParameter` for UQ interface
- Regulatory documentation: `intended_use.md`, `downstream_integration.md`, `iec62304_mapping.md`, `eu_mdr_guidelines.md`, `mdcg_2019_11.md`
- `known_anomalies.yaml` with MADD-ANO-001 and MADD-ANO-002
- `soup_package.md` (skeleton)
- `SECURITY.md`, `CONTRIBUTING.md`, `CITATION.cff`
- Algorithm guide template and HeatNode algorithm guide
- `scripts/check_anomalies.py`, `scripts/check_impl_mapping.py`, and `scripts/check_citations.py`
- GitHub issue template for anomalies
- Developer guide: `docs/developer_guide/` with `node_authoring.md`, `documentation_standards.md`, `testing_standards.md`
- Bibliography citation system: Pandoc-style `[@Key]` syntax with CI validation
- Claude skill `.claude/skills/commit-and-push/` for commit/push compliance checklist
- Migrated to `src/` layout with hatchling build backend
- Reorganized tests into subdirectories: `core/`, `nodes/`, `surrogates/`, `api/`, `viz/`, `compliance/`, `verification/`

### Changed
- Build backend: setuptools → hatchling
- Package layout: flat → src/
- `pyproject.toml` URLs updated to Microrobotics-Simulation-Framework org

### Verification
- 512+ existing tests pass after restructure
- New compliance test suite validates all Phase 0-4 artifacts

### Security
- No security-relevant changes in this release

### Known Anomalies
- MADD-ANO-001: LBM GPU segfault on CUDA 12.2 + jaxlib 0.5.1 (open, context_dependent)
- MADD-ANO-002: HeatNode CFL stability not enforced at runtime (open, context_dependent)

## [0.1.0] - 2025-03-01

### Added
- Initial release: modular simulation framework with functional state pattern
- Core: GraphManager, SimulationNode ABC, EdgeSpec, scheduling, coupling, adaptive timestepping, parameter sweeps, checkpoint/restore
- Nodes: BallNode, TableNode, SpringDamperNode, RigidBody2DNode, HeatNode, LBMPipeNode
- Surrogate framework: SurrogateArchitecture ABC, SurrogateNode, SurrogateTrainer, DatasetGenerator, architectures (MLP, DeepONet, SDeepONet, FNO)
- Visualization: matplotlib, terminal, PyVista, pygfx backends; ZMQ network transport
- API: FastAPI server with REST, WebSocket (JSON + binary), server-side rendering
- 545+ tests
