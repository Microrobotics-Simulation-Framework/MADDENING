# HeatNode

**Module**: `maddening.nodes.heat`
**Stability**: stable
**Algorithm ID**: `MADD-NODE-005`
**Version**: 1.0.0

## Summary

1D heat diffusion on a uniform rod with Dirichlet boundary conditions, solved using explicit finite differences.

## Governing Equations

$$
\frac{\partial T}{\partial t} = \alpha \nabla^2 T + S
$$

where $T$ is temperature, $\alpha$ is thermal diffusivity, and $S$ is a volumetric heat source term.

## Discretization

Explicit finite difference on a uniform grid of $N$ cells spanning a rod of length $L$:

- **Space**: 2nd-order central difference: $\nabla^2 T_i \approx \frac{T_{i+1} - 2T_i + T_{i-1}}{\Delta x^2}$
- **Time**: Forward Euler (1st-order explicit): $T_i^{n+1} = T_i^n + \Delta t \cdot [\alpha \frac{T_{i+1}^n - 2T_i^n + T_{i-1}^n}{\Delta x^2} + S_i]$
- **Boundary conditions**: Dirichlet at both ends, enforced by ghost cells and overwriting boundary values

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\alpha \nabla^2 T$ (diffusion) | `maddening.nodes.heat.HeatNode.update` | 2nd-order central FD stencil via `jnp.concatenate` padding + array slicing |
| $S$ (source term) | `maddening.nodes.heat.HeatNode.update` | Added as `source * dt` after diffusion step |
| Time integration ($\partial T / \partial t$) | `maddening.nodes.heat.HeatNode.update` | Forward Euler: `T + coeff * laplacian + source * dt` |
| Left Dirichlet BC | `maddening.nodes.heat.HeatNode.update` | `T_new.at[0].set(T_left)` — JAX primitive `jax.numpy.ndarray.at[].set()` |
| Right Dirichlet BC | `maddening.nodes.heat.HeatNode.update` | `T_new.at[-1].set(T_right)` — JAX primitive `jax.numpy.ndarray.at[].set()` |

## Assumptions and Simplifications

1. Uniform grid spacing ($\Delta x = L / N$)
2. Constant thermal diffusivity (no temperature dependence)
3. 1D geometry (rod)
4. Dirichlet boundary conditions at both ends
5. No convection or radiation terms

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| `thermal_diffusivity` | $10^{-6}$ – $1.0$ m²/s | |
| `n_cells` | 4 – 1000 | Convergence verified |
| CFL number | $< 0.5$ | $\Delta t \cdot \alpha / \Delta x^2 < 0.5$ for stability |

## Known Limitations and Failure Modes

1. **CFL stability limit**: $\Delta t < \frac{\Delta x^2}{2\alpha}$ — violating this produces silently incorrect results (MADD-ANO-002). Runtime enforcement not yet implemented.
2. **1st-order in time**: temporal accuracy is $O(\Delta t)$
3. **No convection**: pure diffusion only
4. **No radiation**: no radiative heat transfer

## Stability Conditions

$$
\Delta t < \frac{\Delta x^2}{2 \alpha d}
$$

where $d = 1$ is the spatial dimension. For the explicit scheme, the CFL number $\text{CFL} = \frac{\alpha \Delta t}{\Delta x^2}$ must satisfy $\text{CFL} < 0.5$.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `temperature` | `(n_cells,)` | K | Nodal temperatures |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `n_cells` | int | 10 | — | Number of grid cells |
| `length` | float | 1.0 | m | Physical length of the rod |
| `thermal_diffusivity` | float | 0.01 | m²/s | Thermal diffusivity $\alpha$ |
| `initial_temperature` | float | 0.0 | K | Uniform initial temperature |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| `left_temperature` | scalar | `T[0]` | Dirichlet BC at left end |
| `right_temperature` | scalar | `T[-1]` | Dirichlet BC at right end |
| `heat_source` | `(n_cells,)` or scalar | 0.0 | Volumetric heat source term |

## References

- Crank, J. (1975). *The Mathematics of Diffusion*. Oxford University Press.
- LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.

## Verification Evidence

- Benchmark: `MADD-VER-001` — Analytical solution comparison for constant-BC diffusion
- Test file: `tests/verification/test_heat_analytical.py`

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2025-03-01 | Initial implementation |
