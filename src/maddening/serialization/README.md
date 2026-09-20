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

## Non-finite numbers

`NaN` and the infinities have no JSON literal, so they are written as the
quoted tokens `"NaN"`, `"Infinity"` and `"-Infinity"` — a diverged param, an
infinite `ParamSpec` bound — and `from_dict` turns them back into floats:

```json
{"params": {"stiffness": "NaN"}, "bounds": ["-Infinity", "Infinity"]}
```

Before 0.4.0 these went out as the *bare* tokens `json.dumps` writes at its
default `allow_nan=True`, which only Python reads (`MADD-ANO-006`).  Those are
still accepted on load, so an older config needs no migration.

Because a decoder cannot tell the float `NaN` from a string that reads
`"NaN"`, `to_dict` **raises** on a string leaf equal to one of the three
tokens, naming its path.  Spell such a value differently.  This is a real
limitation on a wire format and is registered as **`MADD-ANO-010`**; the
registry entry has the reasoning, the workaround and why a tagged object
such as `{"__nonfinite__": "Infinity"}` was not used instead (it relocates
the ambiguity to a rarer shape and makes it silent on read rather than
loud on write, and it costs the FMU's C wrapper a JSON parser it does not
otherwise have).

The rule applies to a node *name* as well as a parameter value, and to a
header a client puts on the FMI wire, but not to dict **keys** — nothing
decodes a key.  Both the refusal and the decode are exact string matches,
so `"nan"`, `"inf"`, `"+Infinity"` and `"NaN "` are all stored and read
back as the strings they are.  Do not make them case-insensitive to match
the C wrapper's `strtod`: that would turn every one of those into a float
with nothing left to notice.

The same encoding is used by the USD stage (`maddening:paramsJson` and the
other JSON attributes) and the FMI sidecar wire; the shared implementation is
`maddening.serialization.json_codec`.

## What is / is not serialized

- **Serialized:** node descriptors (type, name, timestep, params), edges (registered transform names, interface mappings as their `MappingSpec` — kind, hyper-parameters and point references; `from_dict(..., base_dir=)` locates `{"asset": ...}` files), external input specs
- **Not serialized:** runtime state (JAX arrays) and mapping weights (checkpoints carry those), unregistered edge transforms (functions), observers, compiled step
- After `from_dict`, call `gm.compile()` before running
