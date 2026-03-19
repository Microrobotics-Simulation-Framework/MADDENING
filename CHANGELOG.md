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
- Interface DOF awareness: `interface_dof_indices()` and `compute_interface_correction()` on SimulationNode; coupling system re-applies interface values after node update, fixing the DD coupling "cold lock" where HeatNode's Dirichlet BC enforcement prevented heat transfer
- Coupling iteration predictors (`predictor="linear"` or `"quadratic"` on CouplingGroup): extrapolates initial guess from previous timesteps' converged states, reducing iteration count
- `tune_coupling_params()` utility for grid-search optimization of coupling parameters (tolerance, max_iterations, acceleration)
- `HybridNode` wrapper: composes a physics node with an additive correction function; `generate_correction_data()` for training integration error correctors
- `derivatives()` method on SimulationNode with implementations on BallNode, SpringDamperNode, HeatNode; `integrators` module with `euler_step`, `heun_step`, `rk4_step` and convenience `integrate_node()`
- `calibrate()` utility for gradient-based parameter recovery from reference trajectories using `jax.grad`
- Implicit node support: `implicit_residual()` on SimulationNode with fixed-count Newton iteration via `jax.lax.fori_loop`; implemented for SpringDamperNode and HeatNode; unconditionally stable for stiff problems
- OpenUSD integration: codeless schemas (`MaddeningSimulationGraph`, `MaddeningNode`, `MaddeningEdge`, `MaddeningCouplingGroup`, `MaddeningExternalInput`), `USDWriter` for time-sampled state output, `save_graph_to_usd()` / `load_graph_from_usd()` for full graph serialization, late-registration guard with RuntimeError
- HeatNode 4th-order FD stencil (`stencil_order=4`), non-uniform grid support (`grid_points` parameter)
- 2D spatial interpolation: `nearest_neighbor_2d()`, `rbf_interpolation_2d()` in interface_mapping
- USD geometry reader: `load_grid_from_usd()`, `create_vessel_phantom()` (Y-shaped bifurcating vessel)
- `geometry_source` attribute on SimulationNode for USD-initialized nodes
- Vessel bifurcation coupling example: three HeatNodes initialized from USD geometry, coupled at Y-junction
- `HistoryViewer3D.add_curve_tube()`: render 3D centerline tubes colored by scalar fields (vessels, pipes, rods)
- `HistoryViewer3D.add_line_plot()`: render 1D fields as 3D line plots (temperature profiles, wave solutions)
- `viewer_from_usd()`, `viewer_from_usd_with_geometry()`, `render_usd_frame()`: bridge USD results data to the general-purpose HistoryViewer3D for interactive replay and screenshots
- USD tests skip gracefully when `usd-core` is not installed (CI compatibility for Python 3.10/3.11)
- `LBMNode`: general 3D Lattice Boltzmann on boolean mask domains with D3Q19/D2Q9 lattices, Zou-He pressure BCs, Guo forcing, runtime clot injection via `wall_mask_update`
- `lbm_geometry.voxelize_vessel()`: analytical Y-bifurcation voxelizer parametric by vessel geometry
- `RigidBodyNode`: full 6DOF rigid body (quaternion orientation, diagonal inertia, DOF constraints). `RigidBody2DNode` deprecated with thin wrapper.
- `HeartPumpNode`: 2-element Windkessel model with pulsatile cardiac output, configurable heart rate / stroke volume / resistance / compliance, bidirectional pressure coupling
- `PyVistaLiveRenderer`: real-time 3D visualization backend with timer callbacks, pause/resume/speed keyboard controls
- Vessel bifurcation live example: real-time simulation + USD recording + PyVista visualization + heat pulse injection demo
- Vessel flow server: FastAPI server with HeartPump+LBM coupling, REST endpoints for heart rate / resistance / clot injection, WebSocket live vitals streaming, browser UI with pressure waveform chart

- Cloud module (`maddening.cloud`): `StreamingSession` ABC and `StreamConfig`/`StreamInfo`/`QualityPreset`/`GPUFramebuffer` data types for WebRTC viewport streaming; `MockStreamSession` for zero-dep testing; HMAC-SHA256 session token auth; `SelkiesSession` GStreamer/WebRTC implementation (requires PyGObject)
- `CloudSession` state machine with SkyPilot VM orchestration, typed health probes (`HealthProbeError` with stage attribution), `CloudReadyResult` with per-stage pass/fail, `MockCloudSession` for testing; preemption detection with configurable policy (CHECKPOINT/FAILOVER/ABORT)
- `SelkiesRenderer(Renderer)`: wraps inner renderer + `StreamingSession`, auto-detects GPU/CPU framebuffer path, emits `PerformanceWarning` on CPU fallback
- Multi-GPU Jacobi coupling: `create_device_mesh()`, `assign_nodes_to_devices()` with coupling co-location, `build_sharded_jacobi_pass()` for distributed node updates; `GraphManager.enable_multigpu()` method
- Cloud container: `docker/Dockerfile.cloud` (CUDA + GStreamer + MADDENING), `entrypoint.py` with JSON config deserialization
- Cloud API endpoints on `SimulationServer`: `POST /cloud/launch`, `GET /cloud/status`, `POST /cloud/teardown` (unconditionally registered, returns 501 if unconfigured)
- `CloudLauncher`: user-facing cloud job orchestration with `CloudJob` handle, `JobConfig` YAML loading, `CostPolicy` cost guards, credential context manager with cleanup, and `CloudJob.from_cluster_name()` reconnect
- `CloudProvider` ABC with `RunPodProvider` and `LambdaLabsProvider` (stub); per-provider credential file management with write/delete lifecycle
- Cloud examples consolidated under `src/maddening/examples/cloud/`: `01_validate.py` (dry-run), `02_runpod_launch.py` (real launch), config templates
- Restructured package extras: per-provider cloud (`runpod`, `lambda`, `aws`, `gcp`), hardware acceleration (`cuda12`, `tpu`), task bundles (`server`, `client`), combo (`cloud`, `cloud-all`)
- Consistent import guards across all optional dependencies: missing extras now raise `ImportError` with the exact `pip install maddening[extra]` command
- User guide: `docs/user_guide/installation.md` (full install reference), `docs/user_guide/quickstart.md` (5-minute intro)
- `CostPolicy.spot_fallback`: when spot instances are unavailable, auto-retry on-demand (subject to same cost guards); configurable via job config YAML
- `retry_until_up` on all SkyPilot launches to handle transient SSH/provisioning failures
- Concise error message for spot unavailability (truncates verbose per-region table); other errors preserved in full
- Multi-GPU Phase 1: `enable_multigpu()` wired into `_build_step_fn()` — Jacobi coupling uses `jax.device_put` for per-node device placement; correctness validated (single step, 100 steps, `lax.scan` all match non-sharded)
- Multi-job architecture: `Coordinator` (ZMQ ROUTER-based rendezvous with registration, topology broadcast, heartbeat monitoring), `CloudGroup` (provision rank-0 first, inject `COORDINATOR_ADDR` into workers, `teardown_all` / `teardown_one` with `ISOLATE` mode), `SubgraphSpec` + `GroupConfig`
- Cloud examples organized into subdirectories: `config/`, `launch/`, `server/`, `streaming/`

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
