# Unit-Aware Edge Transforms

**Module**: `maddening.core.transforms_unit`
**Stability**: stable

## Overview

When coupling nodes that use different physical unit systems (e.g. an LBM
fluid solver in lattice units coupled to a rigid body in SI), the edge
between them must apply a unit conversion.  MADDENING supports this via:

1. **Unit annotations** on edges and boundary specs (documentation/validation).
2. **Transform factories** that produce JAX-traceable conversion callables.

## Unit Annotations

### On edges

```python
gm.add_edge(
    "fluid", "body", "drag_force", "force",
    transform=lbm_to_si_force(dx=1e-3, dt=1e-6, rho=1060.0),
    source_units="lattice",
    target_units="N",
)
```

### On nodes

Nodes declare expected units on their boundary inputs:

```python
def boundary_input_spec(self):
    return {
        "force": BoundaryInputSpec(
            shape=(3,), coupling_type="additive",
            expected_units="N",
        ),
    }
```

And on flux outputs:

```python
def boundary_flux_spec(self):
    return {
        "drag_force": BoundaryFluxSpec(
            shape=(3,), output_units="lattice",
        ),
    }
```

### Compile-time validation

`GraphManager.validate()` checks two conditions:

- If an edge declares `target_units` that differs from the target node's
  `expected_units`, a warning is emitted.
- If an edge has `source_units` different from `expected_units` and no
  `transform` is set, a warning is emitted.

These are warnings, not errors -- they do not block compilation.

## LBM Lattice Unit Conversions

LBM operates in lattice units where $\Delta x = \Delta t_{\text{LBM}} = 1$.
Physical conversion requires three parameters:

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `dx_physical` | $\Delta x$ | Physical lattice spacing [m] |
| `dt_physical` | $\Delta t$ | Physical timestep per LBM step [s] |
| `rho_physical` | $\rho_0$ | Reference fluid density [kg/m$^3$] |

### Conversion factors

| Quantity | Factor (multiply LBM to get SI) | Factory |
|----------|--------------------------------|---------|
| Length | $\Delta x$ | `lbm_to_si_length` |
| Velocity | $\Delta x / \Delta t$ | `lbm_to_si_velocity` |
| Force | $\rho_0 \, \Delta x^4 / \Delta t^2$ | `lbm_to_si_force` |
| Torque | $\rho_0 \, \Delta x^5 / \Delta t^2$ | `lbm_to_si_torque` |
| Pressure | $\rho_0 \, (\Delta x / \Delta t)^2$ | `lbm_to_si_pressure` |

### Usage

```python
from maddening.core.transforms_unit import lbm_to_si_force, lbm_to_si_torque

# Parameters for a specific LBM setup
dx = 10e-6     # 10 um lattice spacing
dt = 1e-7      # 100 ns per LBM step
rho = 1060.0   # blood density kg/m^3

# Create transform callables (pure JAX functions)
force_transform = lbm_to_si_force(dx, dt, rho)
torque_transform = lbm_to_si_torque(dx, dt, rho)

# Use on edges
gm.add_edge("fluid", "body", "drag_force", "force",
            transform=force_transform,
            source_units="lattice", target_units="N")
gm.add_edge("fluid", "body", "drag_torque", "torque",
            transform=torque_transform,
            source_units="lattice", target_units="N*m")
```

The returned callables are:

- **JAX-traceable**: work inside `jax.jit`, `jax.lax.scan`, `jax.lax.fori_loop`
- **Differentiable**: `jax.grad` flows through them (they are scalar multiplications)
- **vmap-compatible**: work with `jax.vmap` for parameter sweeps

### Optional registration

If you need named transforms for USD serialization, register them manually:

```python
from maddening.core.transforms import register_transform

register_transform("my_force_lbm_to_si")(force_transform)
```

## MIME Follow-on

When MADDENING's unit-aware edges are merged, `IBLBMFluidNode` in MIME
should be updated to:

- Declare `expected_units="lattice"` on boundary inputs (angular velocity,
  orientation).
- Override `boundary_flux_spec()` to declare `output_units="lattice"` on
  `drag_force` and `drag_torque`.
- Edges connecting `IBLBMFluidNode` to `RigidBodyNode` should carry
  `lbm_to_si_force` / `lbm_to_si_torque` transforms with appropriate
  `source_units="lattice"` and `target_units="N"` / `target_units="N*m"`.
