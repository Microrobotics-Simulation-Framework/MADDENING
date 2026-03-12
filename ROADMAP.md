# MADDENING Roadmap

Prioritized by ease of implementation from the current state of the codebase.

---

## Completed

### v1 Core Framework
- Functional state pattern, GraphManager, JIT-compiled graph step
- Multi-rate scheduling, Gauss-Seidel coupling, adaptive timestepping
- Parameter sweeps (`run_sweep` via `jax.vmap`)
- Checkpoint/restore (NPZ), HistoryLogger
- FastAPI server + WebSocket + Cytoscape.js graph viz
- Physics nodes: BallNode, TableNode, SpringDamperNode, RigidBody2DNode, HeatNode

### Surrogate Framework Phase 1
- SurrogateArchitecture ABC, SurrogateNode, euler/rk4 integrators
- DatasetGenerator (from_graph, from_sweep)
- SurrogateTrainer (Optax loop), SurrogateValidator
- MLPDirect, MLPDerivative architectures
- replace_node() with edge/external input preservation
- 42 tests (node, dataset, trainer, replace, integration)

### Surrogate Framework Phase 2
- DeepONetDirect, DeepONetDerivative (branch-trunk architecture)
- SDeepONetDirect, SDeepONetDerivative (GRU branch for temporal history)
- FNODirect, FNODerivative (1D/2D/3D spectral convolution)
- Mixed spatial + scalar field support (FNO scalar bypass MLP)
- Shared architecture utilities (_utils.py)
- 36 tests (deeponet, fno)
- Total: 439 tests, 0 warnings

### Surrogate Framework Phase 3
- Weight serialization: `save_weights(path)` / `load_weights(path, arch)` (NPZ, no pickle)
- `TrainResult.save()` / `TrainResult.load()` convenience methods
- Training callbacks: `EarlyStopping`, `ModelCheckpoint`, `LRSchedule`
- Physics-informed losses: `residual_loss`, `energy_conservation_loss`, `momentum_conservation_loss`, `smoothness_loss`, `composite_loss`
- Total: 530+ tests

### Visualization & Server Infrastructure
- LBMPipeNode: D3Q19 + D3Q7 passive scalar, gravity, partial fill, propeller actuator
- HistoryViewer3D: generic 3D replay viewer (isosurfaces, slices, particles, rotating meshes)
- ServerFrameRenderer: matplotlib blitting (~50fps server-side 2D rendering)
- ServerFrameRenderer3D: PyVista/VTK offscreen 3D rendering
- BinaryStateEncoder: flat float32 packing for WebSocket streaming
- SimulationServer: REST + WebSocket (JSON/binary/render) + surrogate endpoints
- Interactive app: `/viz/app` with parameter tuning, surrogate training, binary WS

---

## Next Steps

### Tier 1: Easy (days of work, minimal dependencies)

#### 1. Field Subscriptions for State Streaming
Let clients request only specific node/field combinations instead of receiving full state.
- Add `subscribe` message to WebSocket protocol: `{"type": "subscribe", "fields": {"fluid": ["density", "velocity"], "robot": ["joint_positions"]}}`
- Server filters state before encoding/sending — reduces bandwidth by orders of magnitude for large grids
- Update `BinaryStateEncoder` to build schema from subscribed fields only
- Backward-compatible: no subscription = full state (current behavior)
- **Why easy**: Small change to WS endpoints and encoder. No core changes.
- **Enables**: O3DE integration (only send render-relevant fields), fast viz backends (less data to transfer)

#### 2. Render Stride / Decoupled Physics-Render Rates
Physics steps N times per frame sent to clients.
- Add `render_stride` parameter to `RealtimeRunner` and WS endpoints
- Server runs `gm.step()` N times, streams every Nth state
- Client can request stride dynamically: `{"type": "config", "stride": 10}`
- For scan-based runs: `run_scan(n_steps)` already does this internally, just need to expose the streaming hook
- **Why easy**: Thin wrapper around existing step loop.
- **Enables**: High-fidelity physics (small dt) with smooth rendering (60fps), O3DE integration

#### 3. Surrogate Example Notebooks
- Jupyter notebooks demonstrating end-to-end workflows for each architecture
- Heat equation with FNO, bouncing ball with MLP, time-series with S-DeepONet
- Comparison of direct vs derivative mode
- **Why easy**: No code changes, just documentation. Uses existing API.

### Tier 2: Moderate (1-2 weeks, some new abstractions)

#### 4. GPU-Native 3D Viz Backend (pygfx/wgpu)
Modern GPU-accelerated rendering backend using game-engine-like patterns.
- **pygfx** on **wgpu** (WebGPU): continuous render loop, persistent GPU buffers, shader-based updates
- Drop-in alternative to PyVista/VTK `HistoryViewer3D` for replay, and to `ServerFrameRenderer3D` for live
- Key advantages over VTK/PyVista:
  - Continuous render loop (no explicit `render()` calls needed)
  - GPU-resident buffers updated in place (no CPU→GPU copy per frame for mesh data)
  - Instanced rendering for particles/repeated geometry
  - WebGPU compute shaders for marching cubes, field slicing, particle advection
  - Double buffering built in
- Implementation: `maddening/viz/backends/pygfx_viewer.py`
- Reuse the same builder API as `HistoryViewer3D` (add_isosurface, add_static_mesh, etc.)
- **Why moderate**: pygfx API is clean, but need to implement isosurface extraction and field slicing in wgpu compute shaders or fall back to CPU.
- **Enables**: 60fps+ local visualization, foundation for O3DE-like rendering quality

#### 5. Scene Description Protocol
A renderer-agnostic description of what to visualize, decoupled from any specific backend.
- Dataclass-based scene graph: meshes, isosurfaces, slices, particles, transforms, materials
- Nodes can optionally provide a `scene_hints()` method describing their preferred visualization
- Renderers (PyVista, pygfx, O3DE, browser) consume the scene description
- Serializable to JSON/protobuf for network transport to remote renderers
- **Why moderate**: Design work to get the abstraction right. Implementation is straightforward.
- **Enables**: Any renderer backend (local or remote) works with any node, O3DE Gem consumes scene descriptions

#### 6. TI-DeepONet (Time-Integrated DeepONet)
DeepONet variant that integrates over a time window, better for stiff systems.
- Extends existing DeepONet architecture with time embedding
- Reuses `_BranchTrunkNet` machinery
- **Why moderate**: New architecture class, but follows established patterns.

#### 7. PIKAN (Physics-Informed Kolmogorov-Arnold Networks)
KAN-based surrogate architecture with learnable activation functions.
- B-spline or Chebyshev basis activations
- Pure JAX implementation (no extra dependencies)
- **Why moderate**: Novel architecture, but fits cleanly into SurrogateArchitecture ABC.

#### 8. Multi-Surrogate Training Pipeline
Train surrogates for multiple nodes simultaneously with shared data generation.
- `SurrogatePipeline` class: configure, generate data, train, validate, replace in one call
- Parallel training of independent nodes
- **Why moderate**: Orchestration layer over existing components.

#### 9. Adaptive Surrogate Switching (ANCHOR-style)
Dynamically switch between physics and surrogate at runtime based on error estimates.
- Monitor surrogate prediction confidence
- Fall back to physics solver when confidence is low
- Requires `jnp.where` conditional execution (already used for multi-rate)
- **Why moderate**: Control logic is straightforward, but error estimation needs design.

#### 10. Neural ODE Backend
Use Diffrax for adaptive-step integration of derivative-mode surrogates.
- `diffrax_integrator` as a drop-in for `euler_integrator`/`rk4_integrator`
- Automatic step-size control within surrogate updates
- **Why moderate**: Diffrax integration is clean, but adds a new optional dependency.

### Tier 3: Significant (weeks, infrastructure work)

#### 11. gRPC/Protobuf Transport Layer
Standard RPC transport for production deployments and O3DE integration.
- `GrpcSimulationServer` alongside existing `SimulationServer` (FastAPI)
- Protobuf messages for state, commands, scene descriptions
- Language-neutral: generates C++ stubs for O3DE Gem, Python for clients
- Bidirectional streaming RPCs for state push + command receipt
- Coexists with WebSocket (same `GraphManager`, different transport)
- **Why significant**: New dependency (grpcio), protobuf schema design, C++ codegen toolchain.
- **Enables**: Production O3DE integration, ROS 2 bridge, low-latency command/state loop

#### 12. Command Queue with Timestamps
Latency-tolerant command injection for networked clients.
- Commands arrive as `(t_sim_target, command_dict)` tuples
- Server maintains a priority queue, applies commands at correct simulation time
- Interpolation for commands that arrive between steps
- Handles out-of-order delivery and network jitter
- **Why significant**: Requires changes to the step loop and external_inputs injection.
- **Enables**: Accurate actuator control over network (robotics), deterministic replay of command sequences

#### 13. State Delta Compression
Send only what changed between frames for large-state simulations.
- zstd compression of full frames (consecutive LBM frames compress ~10:1)
- Optional: true delta encoding (XOR previous frame, compress)
- Per-field change detection (skip unchanged fields entirely)
- **Why significant**: Compression adds latency; need to benchmark throughput vs latency tradeoff.
- **Enables**: Large-grid simulations over network (256^3 LBM = ~64MB/frame uncompressed)

#### 14. Multi-GPU / Distributed Training
- `jax.pmap` or `jax.experimental.mesh` for data-parallel surrogate training
- Shard large FNO models across devices
- Distributed data generation with `run_sweep`
- **Why significant**: JAX multi-device APIs work well but require careful sharding design.

#### 15. External Training Backends
Adapters for training surrogates outside MADDENING's Optax loop.
- **DeepXDE (JAX backend)**: Adapter that converts `SurrogateDataset` to DeepXDE format, trains, extracts weights back to `SurrogateArchitecture.init_params` format
- **PyTorch adapters** (PhysicsNeMo, NeuroMANCER): Export dataset to PyTorch tensors, train externally, import weights to JAX
- **Why significant**: Cross-framework weight conversion is fiddly. PyTorch<->JAX array bridging.

#### 16. Online Learning / Continual Training
- Update surrogate weights during simulation based on live physics data
- Requires differentiable training step within the simulation loop
- Balance exploration (physics solver) vs exploitation (surrogate)
- **Why significant**: Mixing training and inference in a JIT-compiled loop is architecturally challenging.

### Tier 4: Significant (months, production infrastructure)

#### 17. O3DE Integration (Robotics Simulator)
Full integration with Open 3D Engine for cloud-hosted robotics simulation.
- **Architecture**: MADDENING physics on cloud GPU, O3DE rendering on cloud or local
- **O3DE Gem (C++)**: connects to physics server, deserializes state → entity transforms / mesh deformations / particle systems, sends actuator commands back
- **Transport**: gRPC (Tier 3, item 11) with protobuf state + scene descriptions
- **Latency-aware stepping**: real-time clock sync for hardware-in-the-loop robotics
- **Prerequisites**: gRPC transport (item 11), command queue (item 12), scene description protocol (item 5), field subscriptions (item 1)
- **Why tier 4**: C++ Gem development, O3DE build system integration, real-time sync design, extensive testing across network topologies.

```
┌─────────────────────────────┐         ┌──────────────────────────┐
│   MADDENING Physics Server  │         │     O3DE Client          │
│   (Cloud GPU instance)      │         │  (Cloud or Local)        │
│                             │  gRPC/  │                          │
│  GraphManager               │  WS     │  MADDENING Gem (C++)     │
│    ├─ LBMPipeNode           │◄───────►│    ├─ State deserializer │
│    ├─ RobotArmNode          │  binary │    ├─ Entity updater     │
│    ├─ SurrogateNode(s)      │  frames │    ├─ Command sender     │
│    └─ ...                   │         │    └─ Connection mgr     │
│                             │         │                          │
│  SimulationServer           │         │  O3DE Rendering          │
│    ├─ gRPC transport        │         │    ├─ 3D scene           │
│    ├─ WebSocket (existing)  │         │    ├─ Physics debug viz  │
│    └─ Command queue         │         │    └─ UI / controls      │
└─────────────────────────────┘         └──────────────────────────┘
```

#### 18. Cloud Infrastructure / HPC Deployment
- **Apache Libcloud**: Provider-agnostic cloud VM provisioning for training runs
- **Terraform modules**: Infrastructure-as-code for GPU cluster provisioning
- **SLURM integration**: Job submission scripts for HPC clusters
- **Containerization**: Docker/Singularity images with JAX+CUDA+MADDENING
- **Why tier 4**: Infrastructure, not library code. Orthogonal to the simulation framework.

### Tier 5: Research-Level (open problems)

#### 19. Graph Neural Network Surrogates
- Replace entire subgraphs (not just single nodes) with a GNN
- Learn inter-node coupling patterns
- Message-passing over the MADDENING graph topology
- **Why research**: Requires rethinking the node-level surrogate abstraction.

#### 20. Uncertainty Quantification
- Ensemble surrogates, MC dropout, or Bayesian neural networks
- Confidence intervals on surrogate predictions
- Integrate with adaptive switching (Tier 2, item 9)
- **Why research**: UQ for neural surrogates in dynamical systems is an active research area.

#### 21. Automatic Surrogate Selection
- Given a node's physics, automatically select the best architecture
- Meta-learning over architecture hyperparameters
- Benchmark suite for comparing architectures on standard problems
- **Why research**: AutoML for scientific surrogates is largely unsolved.
