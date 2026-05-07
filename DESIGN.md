# MADDENING v1 Design Document

**Modular Automatic Differentiation and Data Enhanced Neural-network INteracting Graph**

## Overview

MADDENING is a JAX-based modular simulation framework for multi-physics.
It represents a simulation as a directed graph of nodes (physics modules)
connected by edges (data dependencies).  The entire graph step is
JIT-compiled into a single XLA computation, enabling automatic
differentiation through the full simulation.

---

## 1. The Functional State Pattern

### Why It Exists

JAX's programming model requires **pure functions**: given the same inputs,
a function must always produce the same outputs and have no side effects.
`jax.jit` works by *tracing* a function with abstract values, so any
Python-level mutation or branching on concrete values will be baked into
the compiled program incorrectly.

This means simulation nodes **cannot** store mutable state as instance
attributes (the way the original prototype did with `self.state`).
Instead, state flows through function arguments and return values.

### How It Works

Each `SimulationNode` is a **descriptor** -- it holds metadata (name,
timestep, parameters) and exposes two pure functions:

```python
class SimulationNode(ABC):
    def initial_state(self) -> dict:
        """Return initial state as dict of JAX arrays."""

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        """Pure function: (state, inputs, dt) -> new_state."""
```

The `GraphManager` owns all state centrally in a `dict[str, dict]` mapping
node names to their state dictionaries.  During a step, the graph manager
passes each node's state *in* and captures the returned state *out*.

### Benefits

- The entire graph step is a pure function `full_state -> full_state`
- `jax.jit` compiles it into a single XLA program -- no Python overhead per step
- `jax.grad` can differentiate through the full simulation
- State is inspectable and serializable at any point

---

## 2. Node Authoring Contract

When writing a new `SimulationNode` subclass, you must follow these rules:

### Pure Functions

`update()` must be a pure function.  No side effects.  No mutation of the
input `state` dict.  Return a **new** dict.

### JAX Traceability

All operations inside `update()` must be JAX-traceable:

- **Use `jnp.where`** instead of Python `if` for any branching that depends
  on JAX array values.
- **Use `jnp` operations** instead of NumPy or plain Python math on arrays.
- **No Python loops** over array elements (use `jax.lax.scan`,
  `jax.lax.fori_loop`, or vectorized operations).

```python
# WRONG -- Python if on JAX value
if position < table_pos:
    position = table_pos

# RIGHT -- jnp.where is JAX-traceable
hit = position < table_pos
position = jnp.where(hit, table_pos, position)
```

### Parameters vs State

- **Parameters** (elasticity, mass, etc.) are set at construction time via
  `__init__(**params)` and stored in `self.params`.  They are constants
  during simulation.  Reading `self.params` inside `update()` is fine
  because JAX treats Python-level constants as literals.
- **State** (position, velocity, etc.) flows through the function signature.
  It must be a flat dict of JAX arrays.

### Boundary Inputs

`boundary_inputs` is a dict populated by the GraphManager from edge
definitions.  Use `.get(key, None)` for optional inputs so the node
works both standalone and connected.

---

## 3. GraphManager API

### Construction

```python
gm = GraphManager()

gm.add_node(ball_node)
gm.add_node(table_node)
gm.add_edge("table", "ball", "position", "table_position")
```

### Lifecycle

1. **Build** -- `add_node()`, `add_edge()`, `remove_node()`, `remove_edge()`
2. **Validate** -- `validate()` checks integrity, returns warnings/errors
3. **Compile** -- `compile()` topologically sorts + JIT-compiles the step
4. **Run** -- `step()` or `run(n_steps, callback=...)`

Modifying the graph (add/remove node/edge) sets a dirty flag.  The next
`step()` or `run()` call will automatically recompile.

### State Access

```python
state = gm.get_node_state("ball")
gm.set_node_state("ball", {"position": jnp.array(10.0), ...})
```

### Observer Pattern

```python
def my_observer(event, data):
    if event == "step":
        print(data["ball"]["position"])

gm.add_observer(my_observer)
```

Events: `node_added`, `node_removed`, `edge_added`, `edge_removed`,
`compiled`, `step`.

### Serialization

```python
config = gm.to_dict()   # JSON-compatible dict
gm2 = GraphManager.from_dict(config, {"BallNode": BallNode, ...})
```

---

## 4. Scheduling

Execution order is determined by **Kahn's algorithm** (topological sort)
over the edge dependency graph.

### Cycle Handling

Cycles are detected and reported as warnings.  Edges that violate
topological order ("back-edges") automatically use **staggered** values:
they read from the *previous* timestep's state instead of the current
in-progress state.

This is physically reasonable for explicit time integration and avoids
the need for iterative solvers in v1.

### Multi-Rate Timestep Scheduling

Nodes can have different timesteps. The framework computes a **base
timestep** as the GCD of all node timesteps and runs at that rate.
Each node has a **rate divider** (`node_dt / base_dt`), and only
updates when the internal step counter is a multiple of its divider.

```python
# Fast dynamics at 1kHz, slow thermal at 100Hz
fast_node = ContactNode(name="contact", timestep=0.001)
slow_node = ThermalNode(name="thermal", timestep=0.01)
gm.add_node(fast_node)
gm.add_node(slow_node)
gm.compile()
# base_timestep = 0.001, thermal fires every 10th step
```

**Design**:
- **Always-compute, conditionally-apply**: Every node's update is
  computed every step (maintaining static JAX trace structure for JIT),
  but results are applied via `jnp.where` only when the step counter
  fires. Fully differentiable.
- **Zero overhead for uniform rate**: When all nodes share the same
  timestep, the original fast path is used with no step counter or
  conditional logic.
- **Internal `_meta` state**: A step counter lives in `_state["_meta"]`
  but is automatically stripped from all user-facing APIs (returns,
  callbacks, observers, scan history).
- **Properties**: `gm.is_multirate`, `gm.rate_dividers`,
  `gm.base_timestep` (alias for `gm.timestep`).

---

## 5. v1 Scope

### Included

- `SimulationNode` ABC with functional state pattern
- `GraphManager` with add/remove node/edge, validate, compile, step, run
- `EdgeSpec` with optional transforms
- **External inputs**: `add_external_input()` for injecting controller
  commands, sensor data, or user input into the simulation each step
- Topological scheduling with cycle staggering
- JIT compilation of the full graph step
  (`(full_state, external_inputs) -> full_state`)
- Automatic differentiation through the graph
- Observer pattern for external hooks
- JSON-compatible serialization of graph structure (including external inputs)
- ZMQ network transport: `NetworkRelay`/`NetworkReceiver` for state,
  `CommandPublisher`/`CommandReceiver` for commands
- `RealtimeRunner` with optional `command_receiver` for closed-loop control
- `BallNode`, `TableNode`, `SpringDamperNode`, `RigidBody2DNode`, `HeatNode`
- Multi-rate timestep scheduling (GCD-based, JAX-traceable)
- **Gauss-Seidel iterative coupling**: `add_coupling_group()` for strongly-coupled
  subsystems; uses `jax.lax.fori_loop` for differentiability
- **Adaptive timestepping**: `run_adaptive()` with Richardson extrapolation and PI
  controller; `run_adaptive_scan()` for differentiable batch runs
- **Parameter sweeps**: `run_sweep()` via `jax.vmap` over initial conditions
- **State checkpointing**: `save_state()` / `load_state()` via NPZ format
- **HistoryLogger**: observer-based state accumulation during `run()`
- FastAPI + WebSocket server with interactive Cytoscape.js graph visualization
- Comprehensive example suite and 350+ tests

### `jax.lax.scan` Integration

For batch (non-real-time) simulations, the Python loop in `run()` is the
bottleneck -- each iteration crosses the Python/XLA boundary.  Two scan-based
methods eliminate this:

```python
# Fast batch run (no per-step callbacks or observers)
final_state = gm.run_scan(10000)

# With full history (all intermediate states as stacked JAX arrays)
final_state, history = gm.run_scan_with_history(10000)
# history["ball"]["position"] is shape (10000,)
```

**Design trade-offs**:
- `run_scan` pushes the entire loop into XLA — orders of magnitude faster
  for large simulations
- No per-step callbacks or observer notifications (those require Python)
- External inputs are **static** (same every step).  For dynamic inputs,
  use `step()` in a loop or `RealtimeRunner`
- The step-by-step `step()` and `run()` methods are **unchanged** and
  remain the correct choice for real-time operation

### Reference Nodes

**`BallNode`** — point mass under gravity with optional surface collision.
Uses `jnp.where` for JAX-traceable collision handling.

**`TableNode`** — static surface. Returns state unchanged.

**`SpringDamperNode`** — linear spring-damper connecting two points.
`F = -k*(x - anchor - rest_length) - c*v`, integrated with semi-implicit
Euler.  Anchors via `boundary_inputs["anchor_position"]`.

### Deferred to Future Versions

- **Implicit solvers**: Newton iteration for stiff systems
- **Parallel node execution**: nodes at the same topological level
  could run in parallel
- **GPU multi-device**: distribute nodes across devices via `jax.pmap`
- **PINN surrogate nodes**: neural network nodes trained via
  PhysicsNeMo to approximate expensive physics
- **Dynamic graph modification**: add/remove nodes mid-simulation
  without full recompilation
- **ROS 2 bridge**: bidirectional communication with ROS 2 topics

---

## 6. REST / WebSocket API (`maddening.api.server`)

The `SimulationServer` wraps a `GraphManager` with a FastAPI HTTP +
WebSocket interface. Install with `pip install maddening[api]`.

### REST Endpoints

| Method | Path                        | Description                    |
|--------|-----------------------------|--------------------------------|
| GET    | `/api/graph`                | Get graph structure            |
| POST   | `/api/graph/nodes`          | Add a node                     |
| DELETE | `/api/graph/nodes/{name}`   | Remove a node                  |
| POST   | `/api/graph/edges`          | Add an edge                    |
| DELETE | `/api/graph/edges`          | Remove an edge                 |
| POST   | `/api/graph/compile`        | Compile the graph              |
| GET    | `/api/state`                | Get full simulation state      |
| GET    | `/api/state/{node}`         | Get a single node's state      |
| PUT    | `/api/state/{node}`         | Set a node's state             |
| POST   | `/api/simulate/step`        | Advance one step               |
| POST   | `/api/simulate/run`         | Run N steps                    |
| POST   | `/api/simulate/stop`        | Stop a running simulation      |

### WebSocket

| Channel                  | Direction | Description                       |
|--------------------------|-----------|-----------------------------------|
| `/ws/state`              | Server->  | Stream state after each step      |
| `/ws/events`             | Server->  | Graph events (compile, error)     |
| `/ws/control`            | ->Server  | Start/stop/step commands          |

Usage:

```python
from maddening.api.server import SimulationServer
from maddening.nodes import BallNode, TableNode, SpringDamperNode

server = SimulationServer(
    node_registry={"BallNode": BallNode, "TableNode": TableNode,
                    "SpringDamperNode": SpringDamperNode},
)
app = server.create_app()

# uvicorn.run(app, host="0.0.0.0", port=8000)
```

The WebSocket endpoint `/ws/state` polls the `StateRelay` at ~30 Hz
and pushes JSON frames `{"sim_time": float, "state": {...}}` to
connected clients.

---

## 7. Visualization Architecture

### Design Principles

- **Simulation code never knows about visualization** -- coupling is via the
  existing observer pattern only
- **Renderers are swappable** -- matplotlib for 2D dev, O3DE for 3D robotics,
  WebGL for web UI, all behind the same `Renderer` ABC
- **Rate decoupled** -- simulation runs at sim rate, rendering at display rate
  (typically 30-60 fps), neither blocks the other

### Three-Layer Architecture

```
GraphManager (simulation thread)
    |
    | observer callback (EVENT_STEP)
    v
StateRelay (thread-safe snapshot buffer)
    |
    | latest_snapshot() poll at display rate
    v
Renderer ABC (backend-specific: matplotlib, O3DE, WebGL, ...)
```

### Components

**`Renderer` ABC** (`maddening/viz/renderer.py`)
- `setup(graph_info)` -- initialize from graph metadata (not the GraphManager itself)
- `update(sim_time, state)` -- process a new state snapshot at display rate
- `teardown()` -- release resources
- `requested_fields()` -- optional filter for which state fields to receive

**`GraphInfo`** -- frozen dataclass of graph metadata (node names, params,
state fields, edges, timestep).  Built via `GraphInfo.from_graph_manager(gm)`.
Ensures renderers never hold a reference to the mutable GraphManager.

**`StateRelay`** (`maddening/viz/relay.py`)
- Attaches as a GraphManager observer
- Captures latest state in a lock-protected single-slot buffer on each step
- `latest_snapshot() -> (sim_time, state_dict)` for renderer polling
- Thread-safe: lock protects only a reference swap (nanoseconds), never blocks sim

**`RealtimeRunner`** (`maddening/viz/runner.py`)
- Runs simulation on a daemon background thread
- Paces to wall-clock time via `time.perf_counter()` accumulation (drift-free)
- `time_scale` property: 1.0 = real-time, 2.0 = double speed, etc.
- `pause()` / `resume()` / `stop()` via `threading.Event`

### Threading Model

```
Main thread:         Renderer event loop (matplotlib plt.show(), or O3DE tick)
Background thread:   RealtimeRunner._loop() -> gm.step() -> observer -> StateRelay
```

Matplotlib requires the main thread for its event loop.  The simulation runs
on the background thread and communicates via the StateRelay.  For
network-separated renderers (O3DE on another machine), the StateRelay would
be replaced by a NetworkRelay that serializes snapshots to a socket.

### Usage Example

```python
gm = GraphManager()
# ... add nodes, edges, compile ...

relay = StateRelay()
relay.attach(gm)

renderer = MatplotlibRenderer(relay, plot_config={
    "fields": {"ball": ["position", "velocity"]},
})
renderer.setup(GraphInfo.from_graph_manager(gm))

runner = RealtimeRunner(gm, relay, time_scale=1.0)
runner.start()
renderer.run_event_loop()  # blocks on main thread
runner.stop()
```

### Extending to New Backends

To add a new visualization backend (e.g., O3DE, Godot, web):

1. Subclass `Renderer`
2. Implement `setup()`, `update()`, `teardown()`
3. Optionally implement `requested_fields()` for efficiency
4. For network-separated backends, replace `StateRelay` with a
   `NetworkRelay` that serializes over ZMQ/WebSocket

No changes to simulation code, GraphManager, or the Renderer ABC are needed.

### Remote Visualization (HPC)

For HPC environments where the simulation runs on a compute node and
visualization runs on a local workstation, MADDENING provides
network-transparent state transport via ZMQ:

**`NetworkRelay`** (`maddening/viz/network.py`) -- simulation side
- Attaches as a GraphManager observer (same as `StateRelay`)
- Serializes state to JSON and publishes on a ZMQ PUB socket
- `NetworkRelay("tcp://*:5555")` binds to port 5555

**`NetworkReceiver`** (`maddening/viz/network.py`) -- visualization side
- Subscribes to a `NetworkRelay` over ZMQ SUB
- Exposes `latest_snapshot()` with the **same interface** as `StateRelay`
- Every renderer works unchanged -- just pass the receiver where you
  would pass a relay

```
HPC Node                           Local Workstation
─────────                          ─────────────────
GraphManager                       NetworkReceiver
    │ observer                         │ latest_snapshot()
    ▼                                  ▼
NetworkRelay ──── ZMQ PUB/SUB ──── Renderer(s)
 tcp://*:5555    (or SSH tunnel)
```

Typical SSH tunnel setup:

```bash
# Terminal 1 — local machine, create the tunnel:
ssh -L 5555:localhost:5555 user@hpc-node

# Terminal 2 — on the HPC node (or in the SSH session):
python maddening/examples/remote_sim_server.py

# Terminal 3 — local machine:
python maddening/examples/remote_viz_client.py          # terminal mode
python maddening/examples/remote_viz_client.py --mode scene  # matplotlib
```

### Thread Safety Note

While the `RealtimeRunner` is active, do **not** access `GraphManager` state
directly from the main thread.  Always go through the `StateRelay`.  The
`StateRelay`'s lock-protected snapshot ensures safe cross-thread reads.

---

## 8. External Inputs

### The Problem

The original compiled step function was `full_state -> full_state` — a
closed system with no way for external controllers, sensors, or user
commands to inject data into the simulation.

### The Solution

The step function is now `(full_state, external_inputs) -> full_state`.
External inputs are declared with `add_external_input()` and injected
via `step(external_inputs=...)`.

```python
gm.add_external_input("robot", "joint_torques", shape=(7,), dtype=jnp.float32)
gm.compile()

# Each step, provide current commands:
gm.step(external_inputs={"robot": {"joint_torques": torques}})
```

External inputs appear in the target node's `boundary_inputs` dict
alongside edge-delivered values.  If not provided, they default to
zeros of the declared shape.

### Design Properties

- **Backwards-compatible**: `step()` with no arguments still works
  (defaults to zero for all declared external inputs, or empty dict
  if none declared).
- **JIT-compatible**: external inputs are JAX pytrees with fixed
  shapes, so `jax.jit` traces them correctly.
- **Differentiable**: `jax.grad` can differentiate through the
  compiled step w.r.t. external inputs (e.g., for learning control
  policies).
- **Serializable**: external input declarations are included in
  `to_dict()` / `from_dict()`.

---

## 9. Robotics Integration Architecture

MADDENING is designed to serve as the physics backend for robotics
simulators (specifically MIME).  This section documents how the
architecture supports that use case.

### Topology

```
Cloud / HPC                              Robot / Local Machine
───────────                              ─────────────────────
GraphManager                             ROS2 Controller
    │                                        │
    │ observer                               │ publishes commands
    ▼                                        ▼
NetworkRelay ─── state (PUB :5555) ──→  ROS-ZMQ Bridge
    ▲                                        │
    │                                        │ subscribes to state
CommandReceiver ← cmds (SUB :5556) ←── CommandPublisher
    │                                        │
    ▼                                        ▼
RealtimeRunner                           Renderer(s)
  step(external_inputs=commands)
```

### ROS Bridge (future, separate package)

The ROS-ZMQ bridge is a **standalone ROS2 node**, not a SimulationNode.
ROS communication is side-effectful and asynchronous — it cannot live
inside the JIT-compiled graph step.

The bridge translates between ZMQ messages and ROS topics:

- **State out**: subscribes to `NetworkRelay` (ZMQ SUB), publishes ROS
  topics (`/sim/joint_states`, `/sim/imu`, etc.)
- **Commands in**: subscribes to ROS command topics
  (`/joint_commands`, `/cmd_vel`), publishes to `CommandPublisher`
  (ZMQ PUB)
- **Configuration**: YAML mapping from simulation fields to ROS message
  types

The bridge should be a separate package (e.g., `mime_ros_bridge`) that
depends on MADDENING's ZMQ transport but lives outside this repo.
Target ROS2 only.

### Command Channel (ZMQ)

The command channel mirrors the state channel but flows in reverse:

- **`CommandPublisher`** (controller side): publishes command dicts
  over ZMQ PUB.
- **`CommandReceiver`** (simulation side): subscribes with
  `ZMQ.CONFLATE` (latest-value semantics), exposes
  `latest_commands()`.
- **`RealtimeRunner`**: reads from `CommandReceiver` before each
  `step()` and passes commands as `external_inputs`.

Uses PUB/SUB with CONFLATE because for continuous control inputs
(joint torques, velocity commands), only the latest value matters.

### Mapping Robot Components to Nodes

| Component | Node pattern | State | Boundary inputs |
|---|---|---|---|
| Articulated body | Single `CompositeNode` for whole chain | `q` (joint pos), `qdot` (joint vel), `link_poses` | `joint_torques` (ext), `external_wrenches` (from contact) |
| Actuator | `ActuatorNode` per motor group | motor velocity, current | commanded torque (ext), joint velocity (from body) |
| Contact | `ContactNode` | (stateless or penalty state) | link poses (from body) → outputs forces |
| IMU sensor | `IMUSensorNode` | bias state | link pose + velocity (from body) → outputs accel/gyro |
| F/T sensor | `ForceTorqueSensorNode` | (stateless) | contact forces (from contact) |
| Camera | Not a node — a `Renderer` | — | Consumes state via relay, produces images |

**Key convention**: articulated bodies should be modeled as a single
`SimulationNode` containing the full kinematic chain, not one node per
link.  Multi-body dynamics algorithms (ABA, CRBA) operate on the whole
chain for numerical stability, and this avoids explosion of nodes/edges.

### Contact and Cycles

Contact creates a cycle: body positions feed contact detection, contact
forces feed back into body dynamics.  The existing cycle-staggering
mechanism handles this: contact forces are computed from the previous
timestep's positions.  At typical robotics timesteps (0.001s), this is
standard practice and numerically stable for penalty-based contact.

For future complementarity-based contact (LCP), the contact solve must
happen within a single timestep.  This would require either embedding
contact in the body node or adding iterative coupling (deferred to v2).

### Cloud Physics and Latency

For cloud-computed physics with a real robot:

1. **Preferred**: co-locate controller and simulation in the cloud.
   ZMQ link is localhost (microsecond latency).  Only visualization
   and operator commands cross the network.

2. **Alternative**: accept network latency (10-50ms LAN) and model it
   as part of the system.  Robots have real communication latency too.

The `NetworkRelay` already tolerates dropped frames via `ZMQ.CONFLATE`,
making it robust to variable network conditions.

---

## 9. OpenUSD Integration Architecture

MADDENING's primary output bus is an OpenUSD stage, shared with
MICROBOTICA (the interactive simulation platform above MADDENING in
the stack).  Design decisions for the USD integration:

### Codeless Schemas (not hand-coded, not full usdGenSchema)

USD's codeless schema mechanism (available since USD 21.08) provides
formally typed prim schemas without C++ code generation.  This was
chosen over:

- **Hand-coded typeName strings**: prims are untyped in USD's type
  system — no property introspection, no fallback values, no
  validation.  Insufficient for IEC 62304 traceability.
- **Full usdGenSchema**: requires C++ compiler, CMake, Boost, TBB,
  and the full USD build tree.  ABI compatibility with `usd-core`
  from PyPI is an unsolved problem (Pixar issue #2531).

Codeless schemas are shipped as two static files (`plugInfo.json`,
`generatedSchema.usda`) as pip-distributable package data.  Prim
types: `MaddeningSimulationGraph`, `MaddeningNode`, `MaddeningEdge`,
`MaddeningCouplingGroup`, `MaddeningExternalInput`.

**Migration path**: if C++ accessor classes are ever needed, remove
`skipCodeGeneration = true` from `schema.usda`.  Type names and all
downstream consumers remain unchanged.

### Threading and Stage Ownership

- **Single writer (MADDENING physics thread), multiple readers
  (MICROBOTICA render threads).**
- Frame writes are atomic via `Sdf.ChangeBlock`.
- Readers are notified via `Tf.Notice` (USD's built-in push
  notification).  No polling.
- When physics outruns rendering, the renderer sees the latest
  frame (natural frame dropping).

### Late Registration Guard

`Plug.Registry().RegisterPlugins()` must be called BEFORE any
`Usd.Stage` operations (the `UsdSchemaRegistry` is a singleton
initialised once).  `import maddening.usd` is the registration
point.  Late registration raises `RuntimeError` with an actionable
message.

### Transform Registry

Edge transforms are Python callables that aren't directly
serializable.  The `TransformRegistry` (`maddening.core.transforms`)
bridges this gap:

- `@register_transform("name")` decorator registers a callable
  by name (serialized to USD).
- Registration is optional for local use; required for USD
  serialization (`save_graph_to_usd` raises
  `UnregisteredTransformError` for unregistered callables).
- `compile()` emits a warning for unregistered transforms.
- Built-in transforms: `extract_first`, `extract_last`, `negate`,
  `scale(factor)`, `identity`, etc.
- CI: `scripts/check_transforms.py` validates string references.

### Layer Composition

Simulation scenarios compose three USD layers:
1. **Geometry layer**: meshes, transforms, materials (e.g., from
   neuroimaging)
2. **Graph layer**: nodes, edges, coupling groups, parameters
3. **Results layer**: time-sampled state output per node per step

---

## 10. Parameter Differentiability

Node parameters (`self.params`) are Python floats baked into JIT
as compile-time constants.  They do NOT participate in JAX autodiff.

**Current path**: `jax.grad` works through initial conditions and
external inputs, which are JAX arrays in the state dict.

**Planned path** (Phase 4 calibration tooling): a `calibrate()`
utility that rebuilds the step function with parameters as JAX-traced
values, enabling gradient-based parameter recovery from trajectory
data.  The node contract itself does not change — the calibration
layer constructs differentiable wrappers around existing nodes.
