# maddening.nodes

Reference node implementations and guide for writing custom nodes.

## BallNode (`ball.py`)

Point-mass ball under gravity with surface collision.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_position` | 0.0 | Starting height |
| `initial_velocity` | 0.0 | Starting velocity |
| `elasticity` | 0.8 | Coefficient of restitution |

**State:** `position`, `velocity`
**Boundary input:** `table_position` (optional) -- collision surface height

Collision uses `jnp.where` so the update is fully JAX-traceable. If `table_position` is not provided, the ball falls freely.

## TableNode (`table.py`)

Static horizontal surface. Its `update` returns state unchanged.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `position` | 0.0 | Surface height |

**State:** `position`

## SpringDamperNode (`spring.py`)

Linear spring-damper connecting two attachment points. Semi-implicit Euler integration.

```
F = -k * (position - anchor_position - rest_length) - c * velocity
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `stiffness` | 100.0 | Spring constant k (N/m) |
| `damping` | 1.0 | Damping coefficient c (N*s/m) |
| `mass` | 1.0 | Point mass (kg) |
| `rest_length` | 1.0 | Natural length |
| `initial_position` | 0.0 | Starting position |
| `initial_velocity` | 0.0 | Starting velocity |

**State:** `position`, `velocity`
**Boundary input:** `anchor_position` (defaults to 0.0 if absent)

## Writing a Custom Node

```python
import jax.numpy as jnp
from maddening.core.node import SimulationNode

class MyNode(SimulationNode):
    def __init__(self, name, timestep, my_param=1.0):
        super().__init__(name, timestep, my_param=my_param)

    def initial_state(self) -> dict:
        return {"x": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        # Pure function, JAX-traceable. Use jnp.where, not if/else on values.
        force = boundary_inputs.get("force", jnp.array(0.0))
        x = state["x"] + self.params["my_param"] * force * dt
        return {"x": x}
```

**Checklist:**
1. Subclass `SimulationNode`, pass all config as `**params` to `super().__init__`
2. `initial_state()` returns `dict[str, jnp.array]`
3. `update()` is a pure function -- no side effects, no Python-level branching on array values
4. Use `boundary_inputs.get(key, default)` for optional inputs so the node works in isolation
5. `self.params` holds fixed configuration; mutable data goes in `state`
