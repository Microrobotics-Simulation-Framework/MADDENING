# maddening.serialization

Serialization helpers for graph structure (not runtime state).

## API (`config.py`)

```python
from maddening.serialization.config import to_dict, from_dict
```

| Function | Description |
|----------|-------------|
| `to_dict(graph_manager)` | Serialize graph structure to a JSON-compatible dict |
| `from_dict(config, node_registry)` | Reconstruct a GraphManager from a serialized config |

These are thin wrappers around `GraphManager.to_dict()` / `GraphManager.from_dict()`.

## Node Registry Pattern

`from_dict` needs a mapping from type-name strings to Python classes so it can instantiate nodes:

```python
from maddening.nodes import BallNode, TableNode, SpringDamperNode

registry = {
    "BallNode": BallNode,
    "TableNode": TableNode,
    "SpringDamperNode": SpringDamperNode,
}

gm_restored = from_dict(config, registry)
```

## Serialized Format

```json
{
  "nodes": [
    {"type": "BallNode", "name": "ball", "timestep": 0.01, "params": {"initial_position": 5.0, ...}}
  ],
  "edges": [
    {"source_node": "table", "target_node": "ball", "source_field": "position", "target_field": "table_position"}
  ],
  "external_inputs": [
    {"target_node": "ball", "target_field": "force", "shape": []}
  ]
}
```

## What is / is not serialized

- **Serialized:** node descriptors (type, name, timestep, params), edges, external input specs
- **Not serialized:** runtime state (JAX arrays), edge transforms (functions), observers, compiled step
- After `from_dict`, call `gm.compile()` before running
