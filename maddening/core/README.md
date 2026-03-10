# maddening.core

Core simulation framework: node contract, graph orchestration, edge coupling, and scheduling.

## SimulationNode (`node.py`)

Abstract base class every physics node must implement. Nodes are **descriptors** -- they carry metadata and expose two pure functions, but never store mutable simulation state.

```python
class SimulationNode(ABC):
    def __init__(self, name: str, timestep: float, **params): ...
    def initial_state(self) -> dict: ...          # dict of JAX arrays
    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict: ...
```

**Rules:**
- `update` must be a **pure function** suitable for JAX tracing
- Use `jnp.where` instead of Python `if` for value-dependent branching
- `params` are fixed configuration; `state` is mutable and owned by GraphManager

**Helpers:** `state_fields()`, `to_dict()`

## EdgeSpec (`edge.py`)

Immutable dataclass describing a data dependency between two nodes.

```python
EdgeSpec(source_node, target_node, source_field, target_field, transform=None)
```

Meaning: before updating `target_node`, copy `source_node.state[source_field]` into `boundary_inputs[target_field]`, optionally applying `transform` (must be JAX-traceable).

## GraphManager (`graph_manager.py`)

Central orchestrator. Owns all node state, builds the execution schedule, and JIT-compiles the full graph step.

### Construction

```python
gm = GraphManager()
gm.add_node(node)                               # register + init state
gm.add_edge(src, tgt, src_field, tgt_field)      # data dependency
gm.add_external_input(node, field, shape, dtype)  # controller/sensor input
gm.remove_node(name)
gm.remove_edge(src, tgt, src_field, tgt_field)
```

### Validation & Compilation

```python
issues = gm.validate()   # returns list of ERROR/WARNING/INFO strings
gm.compile()             # topo-sort + JIT compile; raises on errors
```

### Execution

| Method | Loop | Per-step callback | External inputs | Returns |
|--------|------|-------------------|-----------------|---------|
| `step(external_inputs=None)` | Single step | Observers fire | Dynamic per-step | State dict |
| `run(n_steps, callback=None, external_inputs=None)` | Python loop | Yes | Static | None |
| `run_scan(n_steps, external_inputs=None)` | `jax.lax.scan` | No | Static | Final state |
| `run_scan_with_history(n_steps, external_inputs=None)` | `jax.lax.scan` | No | Static | (final, history) |

`run_scan` variants push the entire loop into XLA -- no Python dispatch overhead.

### Multi-rate

Nodes can have different timesteps. The graph steps at the GCD (base timestep). Each node fires only when `step_count % rate_divider == 0`, using `jnp.where` to keep the computation JAX-traceable.

### State Access & Observers

```python
gm.get_node_state(name) -> dict
gm.set_node_state(name, state)
gm.add_observer(callback)   # callback(event, data), events: node_added, edge_added, compiled, step, ...
```

### Properties

`timestep`, `base_timestep`, `is_multirate`, `rate_dividers`, `node_names`, `schedule`

### Serialization

`to_dict()` / `from_dict(config, node_registry)` -- JSON-compatible graph structure (not runtime state).

## Scheduling (`schedule.py`)

| Function | Description |
|----------|-------------|
| `topological_sort(node_names, edges)` | Kahn's algorithm; appends cycle participants at end |
| `detect_cycles(node_names, edges)` | DFS-based; returns list of `(node, ...)` tuples |
| `identify_back_edges(schedule, edges)` | Edges violating topo order (use previous-timestep values) |

## Quick Example

```python
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode

gm = GraphManager()
gm.add_node(TableNode("table", timestep=0.01, position=0.0))
gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
gm.add_edge("table", "ball", "position", "table_position")
gm.compile()

final, history = gm.run_scan_with_history(1000)
# history["ball"]["position"] is shape (1000,)
```
