# Accuracy Improvements + OpenUSD Integration Plan

## Overview

This plan covers two interleaved tracks:

**Track A (Phases 0-5)**: Simulation accuracy improvements (items 1-9 from the analysis). These improve the physics, leverage differentiability, and build the calibration tooling that makes MADDENING's accuracy story credible.

**Track B (Phases 6-9)**: OpenUSD integration + spatial accuracy (items 10-12). USD is the data bus between MADDENING and MICROBOTICA. The write path is the immediate priority; graph serialization is a compliance requirement; the read path enables mesh-native geometry from neuroimaging data.

The tracks interleave because spatial accuracy improvements (non-uniform grids, higher-order stencils, conservative mapping) should be designed in the context of USD mesh representation, not independently.

---

## Phase 0: Validation Baseline

**Goal**: Prove that end-to-end differentiability already works before changing anything. This is the "before" measurement.

### 0a. Parameter recovery test (Tier 1)

Set up coupled springs with known "true" parameters (k=100, c=2). Generate a reference trajectory. Start with wrong parameters (k=50, c=5). Use `jax.grad` through 100+ coupled timesteps to recover the true values via gradient descent.

**Verification gate**:
- Recovered parameters within 2% of truth
- Loss drops by 3+ orders of magnitude
- Gradient is finite and nonzero at every optimization step

### 0b. Gradient health audit

Run `jax.grad` through increasingly long rollouts (10, 50, 200, 1000 steps) for coupled springs, coupled heat rods, and the ball+spring+table system. Check that gradients don't vanish, explode, or become NaN.

**Verification gate**:
- All gradients finite for all systems up to 1000 steps
- Gradient magnitude stays within 1e-10 to 1e+10 (no vanishing/exploding)

### 0c. Write results as a benchmark

Store the baseline calibration results (parameter recovery accuracy, convergence curve, gradient norms vs rollout length) so we can compare after each subsequent phase.

**Tests**: ~5 new tests in `tests/calibration/` or `tests/verification/`

---

## Phase 1: Foundational Correctness

**Goal**: Fix two correctness issues that affect all downstream work.

### 1a. Interface DOF awareness

**Problem**: HeatNode's Dirichlet BC enforcement overwrites boundary cells, causing DD coupling to silently fail (no heat transfer). Users must know to use interior cells — a trap.

**Solution**: Add an `is_interface=True` flag to `BoundaryInputSpec`. When the coupling system resolves boundary inputs for a node, and the target field has `is_interface=True`, it records which state fields are boundary DOFs. The node's `update` is called normally, but after the update, the coupling system re-applies the interface value to the boundary DOF (overriding the node's internal BC enforcement).

Alternative (simpler): add a `post_update_interface_fields()` method that returns `{field: index}` pairs. The coupling system calls it after `update` and writes the coupled value into `state[field][index]`.

**Verification gate**:
- DD coupling of two heat rods with `T[-1]` / `T[0]` transforms (the naive setup that currently fails) now correctly transfers heat
- Energy conservation holds (total energy constant to machine precision for equal grids)
- All existing tests still pass (backward compatible — nodes without interface DOFs behave identically)

**Tests**: ~5 new tests

### 1b. Coupling iteration predictors

**Problem**: Each coupling iteration starts from the beginning-of-step state. For smoothly evolving systems, the previous timestep's converged state is a much better initial guess.

**Solution**: Store the last 2 converged states in `_meta`. At the start of each coupling block, extrapolate: `x_pred = 2*x_{n} - x_{n-1}` (linear) or `x_pred = 3*x_{n} - 3*x_{n-1} + x_{n-2}` (quadratic). Use `x_pred` as the initial state for edge resolution in the first coupling pass. The integration still starts from the true beginning-of-step state.

**Verification gate**:
- Coupled springs with diagnostics: predictor reduces iteration count by 1-2 vs no predictor
- Coupled heat rods: same reduction
- Physics results identical (same converged state, just fewer iterations to get there)
- Pre-populate predictor states in `_meta` at compile time for `lax.scan` compatibility

**Tests**: ~5 new tests

**Run full test suite + compliance checks after Phase 1.**

---

## Phase 2: Leveraging Differentiability

**Goal**: Build the tooling that turns differentiability into a practical accuracy advantage. This is MADDENING's moat.

### 2a. Coupling parameter auto-tuning utility

A utility function: given a graph, a reference trajectory, and a list of coupling parameters to tune (relaxation, Robin alpha, tolerance), use `jax.grad` to find optimal values.

```
tune_coupling_params(gm, reference_trajectory, tunable_params) -> optimal_params
```

Internally: Adam optimizer over a loss that measures trajectory deviation from reference. The loss is differentiable because all coupling parameters flow through the JIT-compiled step function.

**Verification gate**:
- For Robin-coupled heat rods, auto-tuned alpha gives fewer coupling iterations than alpha=0.5 (the default)
- For under-relaxed springs, auto-tuned omega matches or beats hand-picked omega
- Optimizer converges in <100 steps

**Tests**: ~3 new tests

### 2b. Hybrid physics + learned residual node

A `HybridNode` wrapper that composes a physics node with a small correction network:

```
output = physics_node.update(state, bi, dt) + correction_network(state, bi, dt)
```

Uses the existing surrogate architecture + trainer infrastructure. The correction network is trained on the *residual* between a coarse simulation and a fine reference, not on the full dynamics.

**Verification gate**:
- Spring oscillator: Euler at dt=0.01 + trained correction matches Euler at dt=0.0001 (the reference) to within 1% over 100 steps
- Correction network generalizes to held-out initial conditions (not just the training set)
- `HybridNode` works with `jax.jit`, `jax.grad`, `lax.scan`, coupling groups
- `HybridNode` is a drop-in `SimulationNode` replacement (passes all graph integration tests)

**Tests**: ~8 new tests (unit + integration + generalization)

### 2c. Learned integration error corrector

Specialized application of 2b: train the correction specifically on `e(state, dt) = x_fine - x_coarse` using the `DatasetGenerator`. Provide a convenience function:

```
corrector = train_integration_corrector(node, dt_coarse, dt_fine, n_trajectories)
hybrid = HybridNode(node, corrector)
```

**Verification gate** (Tier 2 test):
- Corrected Euler at dt=0.01 has 10-100x less error than uncorrected Euler at dt=0.01, measured on held-out trajectories
- Works for springs (scalar state) and heat rods (array state)
- Training converges in <500 epochs

**Tests**: ~5 new tests

**Run full test suite + compliance checks after Phase 2.**

---

## Phase 3: Time Integration

**Goal**: Give nodes access to higher-order time integration without breaking the existing contract.

### 3a. Node `derivatives()` method + pluggable integrators

Add an optional method to `SimulationNode`:

```python
def derivatives(self, state, boundary_inputs):
    """Return state time-derivatives. Optional — enables higher-order integration."""
    raise NotImplementedError
```

Add integrator functions at the graph level:

```python
def euler_step(derivatives_fn, state, bi, dt): ...
def rk4_step(derivatives_fn, state, bi, dt): ...
def rk45_step(derivatives_fn, state, bi, dt): ...  # with error estimate
```

The GraphManager detects whether a node implements `derivatives()`. If yes, it uses the configured integrator (default: Euler for backward compatibility). If no, it calls `update()` directly (existing behavior).

Implement `derivatives()` on all existing physics nodes (BallNode, SpringDamperNode, RigidBody2DNode, HeatNode). These are straightforward — the dynamics are already expressed in `update()`, just factor out the derivative computation.

**Verification gate**:
- Spring oscillator with RK4: energy drift over 10000 steps is <1e-6 (vs ~0.1 with Euler at same dt)
- Heat rod with RK4: temporal convergence is O(dt^4) (measure error at dt, dt/2, dt/4 and verify 4th-order slope)
- Default integrator is Euler — all existing tests pass unchanged
- `jax.grad` works through RK4 steps
- RK4 works inside coupling groups and `lax.scan`

**Tests**: ~15 new tests (per-node derivatives, integrator unit tests, convergence order verification, JAX compatibility)

### 3b. Graph-level RK4

Apply RK4 at the graph level rather than per-node. This evaluates ALL nodes at intermediate RK stages, with boundary inputs recomputed at each stage. Eliminates the operator-splitting error that per-node RK4 still has.

The graph-level integrator needs a "graph derivative" function that computes all nodes' derivatives simultaneously given the full state and boundary inputs. This is built from the individual nodes' `derivatives()` methods + edge resolution.

**Verification gate**:
- Coupled springs: graph-level RK4 gives smaller error than per-node RK4 at the same dt (measures the splitting error)
- Convergence order: graph-level RK4 is O(dt^4) for the coupled system (per-node RK4 may be less due to splitting)
- Works with coupling groups, multi-rate, adaptive timestepping

**Tests**: ~8 new tests

**Run full test suite + compliance checks after Phase 3.**

---

## Phase 4: Calibration Tooling

**Goal**: Build the end-to-end calibration workflow that is MADDENING's strategic differentiator.

### 4a. `calibrate()` utility

```python
result = calibrate(
    gm,
    reference_trajectories,    # list of (initial_state, trajectory) pairs
    parameters={               # what to tune
        "spring": ["stiffness", "damping"],
        "heat": ["thermal_diffusivity"],
    },
    bounds={"stiffness": (1.0, 1000.0)},  # optional box constraints
    n_steps=100,               # rollout length per trajectory
    optimizer="adam",           # or "lbfgs"
    max_epochs=500,
)
# result.params, result.loss_history, result.trajectories
```

Internally: builds a differentiable loss over batched trajectories, optimizes with `jax.grad` + optax. Handles parameter bounds via projection or reparametrization. Supports multiple reference trajectories (different ICs) to prevent overfitting.

**Verification gate** (Tier 3 test):
- Multi-physics calibration: ball + spring + heat rod. True params: k=100, alpha=0.02. Start with k=50, alpha=0.005. Calibrate from coupled trajectory. Both recovered within 5%.
- Optimizer doesn't get stuck (loss decreases monotonically after warmup)
- Works with coupling groups
- Works with batched trajectories (multiple ICs)

### 4b. Calibration + correction composition

Demonstrate the full workflow: calibrate physical parameters first, then train a learned correction on the residual. Show that the combined result is more accurate than either alone.

**Verification gate**:
- Calibrated-only error: X
- Correction-only error: Y
- Calibrated + correction error: < min(X, Y)

**Tests**: ~10 new tests

**Run full test suite + compliance checks after Phase 4.**

---

## Phase 5: Implicit Node Support

**Goal**: Enable nodes to use implicit time integration for stiff problems.

### 5a. Implicit update contract

Nodes can optionally implement:

```python
def implicit_residual(self, state_new, state_old, boundary_inputs, dt):
    """F(x_new) = 0 residual for implicit integration."""
    ...
```

The GraphManager runs a fixed-count Newton iteration (via `lax.fori_loop`) to solve `F(x_new) = 0`. The iteration count is configurable per node. Falls back to `update()` if `implicit_residual` is not implemented.

**Why fixed-count**: `lax.while_loop` is not reverse-mode differentiable. Fixed-count Newton via `fori_loop` is. Choose a generous count (e.g., 10) and rely on early convergence.

### 5b. Implicit HeatNode

Implement `implicit_residual` for HeatNode. The implicit update is:

```
T_new = T_old + dt * alpha * L(T_new) + dt * source
```

where `L` is the Laplacian. The residual is `T_new - T_old - dt * alpha * L(T_new) - dt * source`. Newton iteration solves this via linearization. For the 1D Laplacian with Dirichlet BCs, this is a tridiagonal system — but we solve it iteratively (Jacobi iteration inside `fori_loop`) to stay JAX-traceable.

**Verification gate**:
- Implicit HeatNode is stable at dt = 10 * dx^2 / (2*alpha) — 10x beyond the explicit CFL limit
- Solution matches explicit HeatNode (at small dt) to within discretization error
- `jax.grad` works through the implicit solve
- Works in coupling groups

**Tests**: ~8 new tests

**Run full test suite + compliance checks after Phase 5.**

---

## Phase 6: USD Write Path

**Goal**: MADDENING writes simulation state to a live USD stage each tick. This is the immediate priority for MICROBOTICA integration.

### 6a. Threading and Ownership Model

MADDENING's physics loop and MICROBOTICA's render loop run at different rates on different threads (or processes). The USD stage is the shared data bus. The model:

**Single writer, multiple readers.** MADDENING's physics thread is the sole writer to the results layer. Render threads are read-only consumers. This is enforced by convention (documented) not by locking — USD stages are thread-safe for this pattern when writes are batched.

**Atomic frame writes via `Sdf.ChangeBlock`.** Each frame's state is written inside an `Sdf.ChangeBlock` context manager. This batches all attribute changes into a single atomic operation. Readers never see a partially-written frame — they see either the old frame or the new frame.

**Push notification via `Tf.Notice`.** Readers do NOT poll. When MADDENING writes a frame and the ChangeBlock closes, USD automatically fires `UsdNotice.ObjectsChanged` to all registered listeners. MICROBOTICA's render loop registers a listener that sets a flag or enqueues a render request. This is the idiomatic USD notification mechanism — the same one Omniverse, Houdini, and other USD-native tools use.

**Frame dropping is natural.** When physics runs faster than rendering (the normal case — e.g., 1000 Hz physics vs 60 fps render), the renderer simply reads the latest state when it gets around to rendering. No explicit frame buffer or queue needed. The USD stage always holds the most recent state. Missed frames are fine — the renderer shows the latest available, which is the correct behavior for a live viewport.

**Decoupled loop architecture:**

```
Physics thread                      Render thread
--------------                      -------------
while running:                      listener = register(ObjectsChanged)
    state = gm.step(ext)            while running:
    with Sdf.ChangeBlock():             wait_for_notification()
        writer.write_frame(             state = read_stage()
            state, sim_time)            render(state)
    # Notice fires automatically
    # when ChangeBlock exits
```

**For file export** (non-live, `run_scan` workflows): every frame gets a proper USD time sample. The results layer can be replayed at any speed in `usdview` or any USD reader. No notification mechanism needed — it's a batch write.

**For separate processes** (eventual MICROBOTICA deployment): the shared stage can be a memory-mapped `.usdc` on a shared filesystem, or MICROBOTICA can use USD's `Ar` (asset resolver) to read from a network location. The notification mechanism changes (OS-level file watch or a lightweight side-channel socket), but the stage structure and write pattern are identical.

### 6b. USD output layer

A `USDWriter` class that takes a GraphManager and a USD stage:

```python
writer = USDWriter(stage, gm)
# After each step:
writer.write_frame(state, time)
```

For each MADDENING node, the writer creates a USD prim (under a `/Simulation/` root). State fields become time-sampled USD attributes on those prims. Scalar fields are `float`/`double` attributes. Array fields (like temperature profiles) are `FloatArray` or `VtArray<float>`.

The writer handles:
- Prim creation on first write (lazy, based on actual state structure)
- Time-sampled attribute authoring (`attribute.Set(value, time)`)
- Efficient conversion from JAX arrays to USD-compatible numpy arrays
- Metadata attributes (node type, timestep, parameter snapshot)
- Selective field writing (field subscription, like `BinaryStateEncoder`)

Benchmark target: <1ms overhead per frame for a typical graph (5-10 nodes, mix of scalars and small arrays).

### 6c. Results layer file export

For non-live use (post-simulation analysis, sharing), write the results layer to a `.usdc` file. The results layer composes over the geometry and graph layers.

**Verification gate**:
- Round-trip: write a simulation to USD, read it back, compare values to original (bit-exact for float32)
- Performance: <1ms write overhead per frame for typical graph
- USD file opens correctly in `usdview` and shows time-varying attributes
- `Tf.Notice` fires on each `write_frame` (unit test with mock listener)
- Works with `run()`, `run_scan()`, and `step()` workflows

**Tests**: ~12 new tests
**New dependency**: `pxr` (OpenUSD Python bindings) as optional extra `[usd]`

---

## Phase 7: USD Graph Serialization + Transform Registry

**Goal**: GraphManager is fully serializable to/from USD with a defined custom schema namespace. A complete simulation scenario — geometry, graph, parameters, results — is a single self-describing USD artifact.

### 7a. Transform Registry

Edge transforms are Python callables, which aren't directly serializable to USD. A `TransformRegistry` provides the bridge, following the same pattern as `NodeMeta` and `@stability`:

```python
from maddening.core.transforms import register_transform

@register_transform("extract_right_boundary")
def extract_right_boundary(T):
    """Extract the rightmost cell of a temperature array."""
    return T[-1]

# Usage (either string name or direct callable — both work):
gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
            transform="extract_right_boundary")
gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
            transform=extract_right_boundary)  # also works, auto-resolved to name
```

**Design principles**:
- Registration is **optional for local/testing use**. Inline lambdas keep working. No existing code breaks.
- Registration is **required for USD serialization**. `save_graph_to_usd()` raises `UnregisteredTransformError` if any edge has an unregistered callable.
- `compile()` emits a **warning** (not error) for unregistered transforms, nudging users toward registration without blocking development.
- Each registered transform has a **name** (serialized to USD), a **callable** (used at runtime), and an optional **docstring** (stored as USD metadata).
- A `scripts/check_transforms.py` CI script validates that all transforms referenced in USD artifacts resolve in the current codebase. Analogous to `check_impl_mapping.py`.

**Built-in transforms**: register common transforms out of the box:
- `extract_first`, `extract_last`, `extract_index(n)` — array element extraction
- `negate`, `scale(factor)` — sign/scaling
- `broadcast_scalar(shape)` — scalar to array
- `slice_range(start, stop)` — subarray

**Developer guide update**: Add a "Transform Registration" section to `docs/developer_guide/node_authoring.md` explaining when and why to register, how it enables USD serialization, and the built-in transforms available.

**Tests**: ~8 new tests (registration, lookup, serialization round-trip, unregistered detection, built-in transforms)

### 7b. MADDENING USD schema namespace (codeless schemas)

**Approach**: Use USD's codeless schema mechanism (available since USD 21.08)
to define formally typed prim schemas without C++ code generation.

**Why codeless, not hand-coded or full usdGenSchema**:
- *Hand-coded* (just setting typeName strings) gives prims that USD tools
  treat as untyped — no property introspection, no fallback values, no
  validation.  Insufficient for IEC 62304 traceability where the USD
  artifact must be self-describing.
- *Full usdGenSchema* requires a C++ compiler, CMake, Boost, TBB, and
  the full USD build tree.  ABI compatibility with `usd-core` from
  PyPI is an unsolved problem (Pixar issue #2531).  Overkill for 5
  schema types in a Python-only package.
- *Codeless schemas* give formally typed prims recognised by the USD
  type system (usdview shows correct types, property panel lists
  expected attributes with fallbacks, `UsdSchemaRegistry` resolves
  them) with zero C++ dependency.  Shipped as two static files
  (`plugInfo.json`, `generatedSchema.usda`) as pip-distributable
  package data — proven pattern by `newton-usd-schemas`.

**Migration path**: If accessor classes or custom computed properties
are ever needed, switch to full usdGenSchema on top of the same
`schema.usda` source by removing `skipCodeGeneration = true`.  The
schema files, type names, and all downstream consumers remain
unchanged.

**Schema types** (concrete typed schemas inheriting from `UsdTyped`):

- `MaddeningSimulationGraph` — root prim, carries global config (base_dt, multi-rate info)
- `MaddeningNode` — a simulation node (type, parameters, timestep)
- `MaddeningEdge` — a data dependency (source prim, target prim, fields, transform name, additive flag)
- `MaddeningCouplingGroup` — coupling configuration (member node relationships, tolerance, acceleration, all CouplingGroup fields)
- `MaddeningExternalInput` — declared external input (target prim, field, shape, dtype)

Each schema declares its expected attributes with types and fallback
values in `generatedSchema.usda`.  Parameters are stored as USD
attributes with appropriate types.  Node type is stored as a string
attribute referencing the Python class qualified name.

**Files** (shipped as package data under `maddening/usd/schema/`):

```
maddening/usd/schema/
    plugInfo.json           # type declarations, plugin metadata
    generatedSchema.usda    # property definitions with fallbacks
```

Source schema definition (`schema.usda`) is also committed for
reference, but is not required at runtime.  The two runtime files
can be hand-authored following Pixar's codeless schema example
format, or generated once via `usdGenSchema` and committed.

**Registration** at import time:

```python
# maddening/usd/__init__.py
import pathlib
from pxr import Plug, Usd

# Detect late registration: if UsdSchemaRegistry has already been
# initialised (by any prior Usd.Stage operation), our schemas will
# NOT be picked up.  Fail loudly rather than silently producing
# untyped prims.
#
# Detection method: attempt registration, then verify our types
# actually resolved.  If they didn't, the registry was already
# frozen.
_schema_dir = pathlib.Path(__file__).parent / "schema"
Plug.Registry().RegisterPlugins([_schema_dir.absolute().as_posix()])

# Verification: check that our primary schema type is known.
# UsdSchemaRegistry is the singleton that resolves type names.
# If it was already frozen before our RegisterPlugins call, our
# types will not appear.
from pxr import Usd as _Usd
_registry = _Usd.SchemaRegistry()
if not _registry.FindConcretePrimDefinition("MaddeningNode"):
    raise RuntimeError(
        "MADDENING USD schema registration failed.  This happens "
        "when 'import maddening.usd' occurs AFTER a Usd.Stage has "
        "already been created, because USD's schema registry is a "
        "singleton initialised on first use and cannot be extended "
        "afterward.\n\n"
        "Fix: import maddening.usd BEFORE any Usd.Stage.Create*() "
        "or Usd.Stage.Open() calls.\n\n"
        "  import maddening.usd          # <-- FIRST\n"
        "  stage = Usd.Stage.CreateNew() # <-- AFTER"
    )
```

**Chosen behavior for late registration**: `RuntimeError` with an
actionable message.  NOT a warning.  Rationale:

- A warning would allow the process to continue with silently
  untyped prims.  `prim.GetTypeName()` returns `""` instead of
  `"MaddeningNode"`, `GetPropertyNames()` returns nothing, and the
  USD artifact is not self-describing.  This silently violates
  the IEC 62304 traceability requirement.
- The failure is always a code-ordering bug (import too late), never
  a transient condition.  The fix is deterministic and the error
  message tells the user exactly what to do.
- For a project on a trajectory toward Class III medical device
  compliance, silent data-integrity failures are unacceptable.

**Late-registration test** (`test_schema_late_registration`):

```python
def test_schema_late_registration():
    """Importing maddening.usd after a Usd.Stage exists must raise."""
    import subprocess, sys
    # Run in a subprocess to get a clean Python process where
    # no prior import of maddening.usd has occurred.
    result = subprocess.run(
        [sys.executable, "-c", """
from pxr import Usd
# Create a stage FIRST — this freezes the schema registry
stage = Usd.Stage.CreateInMemory()
# NOW try to import maddening.usd — should raise RuntimeError
import maddening.usd
"""],
        capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "schema registration failed" in result.stderr
```

This MUST run in a subprocess because `UsdSchemaRegistry` is a
process-level singleton — once registered in the parent test process,
it cannot be un-registered.

**Positive counterpart test** (`test_schema_correct_registration`):

```python
def test_schema_correct_registration():
    """Correct order: import maddening.usd → create stage → typed prims work."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c", """
import maddening.usd  # Register schemas FIRST
from pxr import Usd, Sdf

stage = Usd.Stage.CreateInMemory()
prim = stage.DefinePrim("/test", "MaddeningNode")

# Type is recognised
assert prim.GetTypeName() == "MaddeningNode", (
    f"Expected 'MaddeningNode', got '{prim.GetTypeName()}'")

# Schema-declared properties exist with correct types
prop_names = prim.GetPropertyNames()
assert "maddening:nodeType" in prop_names, (
    f"Schema property 'maddening:nodeType' not found in {prop_names}")
assert "maddening:timestep" in prop_names, (
    f"Schema property 'maddening:timestep' not found in {prop_names}")

# Fallback values are correct types
attr = prim.GetAttribute("maddening:timestep")
assert attr.GetTypeName() == Sdf.ValueTypeNames.Double, (
    f"Expected Double, got {attr.GetTypeName()}")

print("PASS")
"""],
        capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"Schema test failed:\nstdout: {result.stdout}\nstderr: {result.stderr}")
    assert "PASS" in result.stdout
```

Both tests run in subprocesses to ensure a clean schema registry
state.  This is unavoidable given the singleton nature of
`UsdSchemaRegistry` and is the standard pattern for testing USD
plugin registration.

**Minimum USD version**: 21.08.  `usd-core` on PyPI is at 26.3, well
above this.  The `[usd]` optional extra will pin `usd-core >= 21.8`.

### 7c. GraphManager to/from USD

```python
# Serialize
save_graph_to_usd(gm, stage, root_path="/Simulation")

# Deserialize
gm = load_graph_from_usd(stage, root_path="/Simulation")
```

Serialization writes all nodes, edges, coupling groups, and external inputs as USD prims under the root. Deserialization reconstructs a functional GraphManager by:
1. Reading node prims, instantiating the correct `SimulationNode` subclass from the type name
2. Reading edge prims, looking up transforms in the `TransformRegistry`
3. Reading coupling group prims, reconstructing `CouplingGroup` dataclasses
4. Calling `compile()` on the reconstructed graph

**Node type resolution**: A `NodeRegistry` (similar to `TransformRegistry`) maps type name strings to Python classes. All built-in nodes are auto-registered. Custom nodes register via `@register_node` decorator.

### 7d. Layer composition

The simulation scenario composes three USD layers:
1. **Geometry layer**: meshes, transforms, materials (authored externally, e.g., from neuroimaging)
2. **Graph layer**: MADDENING nodes, edges, coupling groups, parameters
3. **Results layer**: time-sampled state output

The layers compose via USD's native composition arcs. The graph layer references geometry prims (for nodes that have spatial extent). The results layer is written on top during simulation.

**Verification gate**:
- Round-trip: serialize a GraphManager to USD, deserialize, run the same simulation, get bit-identical results
- Layer composition: open a composed stage in `usdview`, see geometry + graph + results
- Schema validation: `UsdSchemaRegistry` resolves all `Maddening*` types; prims created with `stage.DefinePrim(path, "MaddeningNode")` report the correct type via `prim.GetTypeName()`
- Property introspection: `prim.GetPropertyNames()` returns the schema-declared attributes with correct types and fallback values
- All node types, edge configurations (including registered transforms), and coupling groups survive the round-trip
- Unregistered transforms produce clear error at save time
- Late schema registration: `import maddening.usd` after any `Usd.Stage` operation raises `RuntimeError` with actionable message (subprocess test)
- Correct registration order: `import maddening.usd` before stage creation produces typed prims where `prim.GetTypeName() == "MaddeningNode"` and `prim.GetPropertyNames()` includes all schema-declared attributes with correct types and fallback values (subprocess test)
- `scripts/check_transforms.py` passes

**Tests**: ~20 new tests (including 2 subprocess schema registration tests, typed prim creation, property introspection per schema type, round-trip, transform registry)

---

## Phase 8: Spatial Accuracy (in context of USD geometry)

**Goal**: Improve spatial discretization, designed from the start to work with USD-sourced geometry.

### 8a. Higher-order FD stencils (item 10)

Add a `stencil_order` parameter to HeatNode (default 2 for backward compatibility). Support 4th-order central differences. The stencil width increases (5-point vs 3-point), requiring additional ghost cells at boundaries.

**Verification gate**:
- Spatial convergence test: error vs dx shows O(dx^4) slope with stencil_order=4
- Backward compatible: stencil_order=2 gives identical results to current code

**Tests**: ~3 new tests

### 8b. Non-uniform grid support (item 11)

Allow HeatNode (and future spatial nodes) to accept a `grid` parameter — an array of cell positions — instead of assuming uniform spacing. The FD stencil coefficients become position-dependent.

**Design for USD**: The grid can come from a USD mesh prim's point positions (1D case: extract positions along a curve). The `geometry_source` seam (Phase 9) will feed this.

**Verification gate**:
- Non-uniform grid matches uniform grid solution when grid happens to be uniform
- Graded mesh (fine near boundary, coarse in interior) gives better accuracy per DOF than uniform mesh
- Works with coupling, RK4 integration, calibration

**Tests**: ~5 new tests

### 8c. Conservative 2D/3D interface mapping (item 12)

Extend `interface_mapping` to 2D/3D. For non-conforming meshes at coupling interfaces, implement:
- Nearest-neighbor 2D/3D (trivial extension)
- RBF interpolation 2D/3D (already dimension-agnostic in the kernel)
- Conservative projection via L2 projection or supermesh (significant work)

**Design for USD**: Interface meshes come from USD. The mapping functions accept arrays of point positions, which can be extracted from UsdGeom prims.

**Verification gate**:
- Patch test: constant field transferred across non-conforming 2D interface is exact
- Conservation test: integral of transferred field equals integral of source field
- Works inside coupling groups

**Tests**: ~8 new tests

---

## Phase 9: USD Read Path + Geometry Source

**Goal**: Nodes can optionally initialize from USD geometry prims. A synthetic bifurcating vessel phantom (standing in for eventual neuroimaging-derived anatomy) demonstrates the workflow.

### 9a. `geometry_source` on SimulationNode

Add an optional `geometry_source` parameter to SimulationNode:

```python
class SimulationNode(ABC):
    geometry_source: Optional[str] = None  # USD prim path
```

When set, the node's initialization can query the USD stage for mesh data (point positions, topology, material properties) instead of relying on explicit parameters. The explicit parameter path remains the default and the testing path.

### 9b. USD-initialized HeatNode

HeatNode with `geometry_source="/Anatomy/Vessel"` reads:
- Point positions from the prim's `UsdGeom.Points` or `UsdGeom.BasisCurves` → becomes the non-uniform grid (Phase 8b)
- Material properties from `UsdShade` attributes → thermal diffusivity
- Extent from `UsdGeom.Boundable` → rod length

### 9c. Bifurcating vessel phantom USD asset

Since no real neuroimaging mesh exists yet, create a synthetic USD asset representing a Y-shaped bifurcating vessel (a common blood vessel phantom geometry). This is more interesting than a straight pipe — it has:
- A parent vessel (single tube) that splits into two daughter branches
- Different diameters per branch (parent wider than daughters)
- Non-uniform point spacing (finer near the bifurcation)
- Material property variation (different wall properties per region)

The phantom is authored as a `.usdc` file with:
- `UsdGeom.BasisCurves` prims for each branch centerline
- `UsdGeom.Mesh` prims for the vessel wall surfaces
- Primvar attributes for region labels (`parent`, `daughter_left`, `daughter_right`)
- `UsdShade.Material` prims with thermal/diffusion properties per region

This phantom serves as the test geometry for:
- USD read path (Phase 9b)
- Non-uniform grid initialization (Phase 8b)
- Multi-domain coupling (parent rod → two daughter rods at bifurcation)
- Conservative interface mapping at a non-conforming junction (Phase 8c)

### 9d. Bifurcation coupling example

A complete example using the vessel phantom:
- Three HeatNodes (parent + two daughters), each initialized from their USD branch
- Coupling at the bifurcation point (parent's right boundary couples to both daughters' left boundaries via additive flux)
- Demonstrates the full workflow: USD geometry → node initialization → coupled simulation → USD results output

**Verification gate**:
- HeatNode initialized from USD mesh gives same results as HeatNode initialized from explicit parameters (when the mesh happens to be uniform)
- Non-uniform vessel geometry produces physically reasonable results
- Bifurcation coupling conserves energy (total heat in parent = total heat in daughters)
- The full scenario — geometry + graph + parameters + results — is a single composable USD artifact

**Tests**: ~10 new tests

---

## Dependency Graph

```
Phase 0 (baseline)
  |
  +-- Phase 1a (interface DOF)
  +-- Phase 1b (predictors)
  |
  +-- Phase 2a (coupling auto-tune)
  +-- Phase 2b (hybrid nodes) ----+
  +-- Phase 2c (error corrector) --+-- depends on 2b
  |
  +-- Phase 3a (derivatives + integrators)
  |     |
  |     +-- Phase 3b (graph-level RK4)
  |
  +-- Phase 4a (calibrate utility) -- depends on 0, benefits from 1-3
  +-- Phase 4b (calibrate + correct) -- depends on 2b, 4a
  |
  +-- Phase 5 (implicit nodes)
  |
  +-- Phase 6 (USD write + threading model) -- independent
  |     |
  |     +-- Phase 7a (transform registry) -- independent, but needed by 7b
  |     +-- Phase 7b-d (USD graph serialization)
  |           |
  |           +-- Phase 9 (USD read + geometry source + vessel phantom)
  |
  +-- Phase 8a (higher-order FD) -- independent
  +-- Phase 8b (non-uniform grids) -- before 9b
  +-- Phase 8c (conservative 2D/3D mapping) -- before 9d
```

Phases 0-2 can be done sequentially in a single push.
Phase 3 can overlap with Phase 6 (different code areas).
Phase 4 depends on earlier phases but benefits from running after 3.
Phase 5 is independent and can be done whenever.
Phase 7a (transform registry) can be done early — it's useful independent of USD.
Phases 6-7 are the USD track, can run in parallel with the accuracy track.
Phases 8-9 bridge both tracks.

---

## Milestone Verification Gates

After each phase, run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v --tb=short --ignore=tests/viz
python scripts/check_anomalies.py
python scripts/check_impl_mapping.py
python scripts/check_citations.py
```

Additionally:

| Milestone | Extra Verification |
|-----------|-------------------|
| After Phase 0 | Baseline calibration metrics recorded |
| After Phase 1 | DD heat coupling transfers heat with naive setup |
| After Phase 2 | Tier 2 test: corrected Euler matches fine-dt reference |
| After Phase 3 | RK4 convergence order verified (O(dt^4) slope) |
| After Phase 4 | Tier 3 test: multi-physics parameter recovery |
| After Phase 5 | Implicit heat stable at 10x CFL limit |
| After Phase 6 | USD file opens in usdview with time-varying data; Tf.Notice fires |
| After Phase 7 | GraphManager round-trips through USD; `UsdSchemaRegistry` resolves `Maddening*` types; `check_transforms.py` passes |
| After Phase 8 | O(dx^4) convergence; conservative mapping patch test passes |
| After Phase 9 | Vessel phantom bifurcation example runs end-to-end as single USD artifact |

---

## Estimated Scope

| Phase | New tests | New/modified files | Effort |
|-------|-----------|-------------------|--------|
| 0 | ~5 | 1 example + tests | 1 day |
| 1 | ~10 | node.py, graph_manager.py, coupling | 2-3 days |
| 2 | ~16 | hybrid_node.py, calibration utils | 3-4 days |
| 3 | ~23 | node.py, graph_manager.py, all nodes | 4-5 days |
| 4 | ~10 | calibrate.py, examples | 2-3 days |
| 5 | ~8 | node.py, graph_manager.py, heat.py | 3-4 days |
| 6 | ~12 | usd_writer.py, new [usd] extra | 4-5 days |
| 7 | ~28 | transforms.py, usd/__init__.py, usd/schema/{plugInfo.json, generatedSchema.usda}, usd_serialization.py, check_transforms.py | 7-9 days |
| 8 | ~16 | heat.py, interface_mapping.py | 4-5 days |
| 9 | ~10 | node.py, heat.py, usd_reader.py, vessel phantom asset, bifurcation example | 4-5 days |
| **Total** | **~138** | | **~6-8 weeks** |
