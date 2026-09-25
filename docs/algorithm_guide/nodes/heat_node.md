---
bibliography: ../../bibliography.bib
---

# HeatNode

**Module**: `maddening.nodes.heat`
**Stability**: stable
**Algorithm ID**: `MADD-NODE-005`
**Version**: 2.0.0

## Summary

1D heat diffusion on a uniform rod with Dirichlet boundary conditions, solved using explicit finite differences [@Crank1975; @LeVeque2007].

## Governing Equations

$$
\frac{\partial T}{\partial t} = \alpha \nabla^2 T + S
$$

where $T$ is temperature, $\alpha$ is thermal diffusivity, and $S$ is a volumetric heat source term.

## Discretization

Explicit finite difference on a uniform, **cell-centred** grid of $N$ cells spanning a rod of length $L$.  Cell $i$ sits at $x_i = (i + \tfrac12)\Delta x$ with $\Delta x = L/N$, so the rod ends $x = 0$ and $x = L$ lie half a cell outside the first and last cell centre.

- **Space**: 2nd-order central difference: $\nabla^2 T_i \approx \frac{T_{i+1} - 2T_i + T_{i-1}}{\Delta x^2}$, or the 5-point 4th-order stencil $\frac{-T_{i+2} + 16T_{i+1} - 30T_i + 16T_{i-1} - T_{i-2}}{12\Delta x^2}$ when `stencil_order=4`
- **Time**: Forward Euler (1st-order explicit): $T_i^{n+1} = T_i^n + \Delta t \cdot [\alpha \nabla^2 T_i^n + S_i]$
- **Boundary conditions**: Dirichlet at the rod ends $x = 0$ and $x = L$, imposed entirely through ghost cells.  No cell is overwritten after the update.

### Boundary closure

`left_temperature` and `right_temperature` are $T(0)$ and $T(L)$.  Because the boundary sits half a cell outside the first cell centre, the value is imposed by choosing the ghost cell at $x = -\Delta x/2$ so that the reconstruction through it hits $T_b$ at $x = 0$:

- **2nd-order stencil** — linear reconstruction, $T_{-1} = 2T_b - T_0$.  This is the conservative finite-volume closure: the first row becomes the difference of two face fluxes, and the left face flux is the one at the rod end.  Its local truncation error is $O(1)$, but the conservative form still gives a globally 2nd-order scheme (supraconvergence on cell-centred grids), measured at 2.000.
- **4th-order stencil** — cubic through $(0, T_b)$ and the three nearest cell centres, evaluated at both ghost positions:

$$
T_{-1} = \frac{16 T_b - 15 T_0 + 5 T_1 - T_2}{5},
\qquad
T_{-2} = \frac{64 T_b - 90 T_0 + 40 T_1 - 9 T_2}{5}
$$

  A cubic is the lowest degree that works.  The 5-point stencil divides ghost errors by $12\Delta x^2$, so a linear or quadratic extrapolation caps the scheme at order 2 or 3 (measured 2.000 and 2.987 on the finest pair of a 10/20/40/80/160 ladder, against 3.957 for this cubic); a quartic is accurate enough but spectrally unstable.  The linear figure is worth reading twice: a linear ghost is genuinely 2nd order, so on a manufactured solution whose curvature vanishes at the rod ends it measures 4.08 and looks correct — see `maddening.nodes.heat._dirichlet_ghosts_4th_order` and `TestTheSteadyProfileCanSeeABrokenScheme`.  Under this closure the 5-point and 3-point forms are algebraically identical at cells $0$ and $N-1$, so the 2nd-order fallback applied there costs nothing.

Before 0.4.0 the Dirichlet value was written into the first and last cell after the update, which imposed it half a cell inside the rod and made the scheme globally 1st-order (MADD-ANO-007), and the 4th-order ghosts were built one cell out of position (MADD-ANO-008).

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\alpha \nabla^2 T$ (diffusion), stencil assembly | `maddening.nodes.heat.HeatNode._compute_laplacian` | Selects the stencil and builds the ghost-padded field via `jnp.concatenate` + array slicing |
| $\nabla^2 T$, 2nd-order interior stencil | `maddening.nodes.heat._laplacian_2nd_order_uniform` | $(T_{i+1} - 2T_i + T_{i-1})/\Delta x^2$ |
| $\nabla^2 T$, 4th-order interior stencil | `maddening.nodes.heat._laplacian_4th_order_uniform` | 5-point form, 3-point at cells $0$ and $N-1$ (identical there under the cubic closure) |
| $\nabla^2 T$, non-uniform grid | `maddening.nodes.heat._laplacian_nonuniform` | Variable-$\Delta x$ 2nd-order form; `stencil_order=4` is not offered here |
| Left/right Dirichlet BC, 2nd order | `maddening.nodes.heat._dirichlet_ghosts_2nd_order` | $T_{-1} = 2T_b - T_0$, imposing $T_b$ at the rod end |
| Left/right Dirichlet BC, 4th order | `maddening.nodes.heat._dirichlet_ghosts_4th_order` | Cubic through the rod end and the three nearest cell centres |
| Rod-end flux $-\alpha\,\partial T/\partial x$ at $x = 0, L$ | `maddening.nodes.heat._rod_end_gradient` | One-sided reconstruction anchored at the rod end; with the Dirichlet datum it reads `stencil_order` cells and is accurate to `stencil_order`, without one it extrapolates from three cells at 2nd order |
| Lagrange derivative at the end face | `maddening.nodes.heat._lagrange_gradient_weights` | Pure-Python weights, folded before tracing: one formula for both grids, both ends and both datum cases, and one dot product in the graph |
| $S$ (source term) | `maddening.nodes.heat.HeatNode.update` | Added as `source * dt` after diffusion step |
| Time integration ($\partial T / \partial t$) | `maddening.nodes.heat.HeatNode.update` | Forward Euler: `T + alpha * dt * laplacian + source * dt` |
| Stability bound on $\Delta t$ | `maddening.nodes.heat.HeatNode.__init__` | Refuses a configuration above the per-stencil Fourier limit in `MAX_FOURIER_NUMBER`, and a constant no rod can have (MADD-ANO-062) |
| Stability bound on a non-uniform grid | `maddening.nodes.heat._nonuniform_fourier_spacing` | $\min_i h_{L,i} h_{R,i}$, the length squared the Gershgorin bound divides by; the constructor refuses $\Delta t\,\alpha / \min(h_L h_R) > 1/2$ |
| Coupled-pair limit (warning) | `maddening.nodes.heat._coupled_pair_advisories` | `compile()` warns when two rods exchange end cells in a coupling group past $3/8$ (`stencil_order=2`) or $0.226$ (`stencil_order=4`), MADD-ANO-050 |
| Rod-end closure on a sharded rod | `maddening.nodes.heat.HeatNode.update_padded` | The block holding a rod end (from `shard_info`) rebuilds its ghosts with the two closures above, from `left_temperature` / `right_temperature`; the 4th-order cubic as one `(2, 3)` product (`_cubic_ghosts`), equal to `update` to float32 rounding |
| Which halo fill a sharded rod takes | `maddening.nodes.heat.HeatNode.halo_boundary` | `"edge"`; `ShardedStencilNode` refuses `"zero"` and `"periodic"`, which the closure would otherwise ignore |

## Assumptions and Simplifications

1. Uniform, cell-centred grid ($\Delta x = L / N$, cell $i$ at $(i + \tfrac12)\Delta x$)
2. Constant thermal diffusivity (no temperature dependence)
3. 1D geometry (rod)
4. Dirichlet boundary conditions at both ends
5. No convection or radiation terms

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| `thermal_diffusivity` | $10^{-6}$ – $1.0$ m²/s | |
| `n_cells` | 4 – 1000 | Convergence verified |
| Fourier number, `stencil_order=2` | $< 1/2$ | $\Delta t \cdot \alpha / \Delta x^2 < 0.5$; exact spectral bound |
| Fourier number, `stencil_order=4` | $< 5/16$ | $0.3125$, conservative; the sharp bound runs from $0.3169$ at $N=5$ to $0.3249$ as $N \to \infty$ (MADD-ANO-009) |
| Fourier number, non-uniform grid | $\le 1/2$ | $\Delta t \cdot \alpha / \min(h_L h_R)$, any `stencil_order`; Gershgorin, exact on a uniform grid |
| Fourier number, two rods coupled end to end | $< 3/8$ (order 2), $< 0.226$ (order 4) | Converged exchange through the Dirichlet inputs (MADD-ANO-050) |

## Known Limitations and Failure Modes

1. **Stability limit depends on the stencil**: Fourier number $< 1/2$ for `stencil_order=2`, $< 5/16$ for `stencil_order=4` (MADD-ANO-009), and $\Delta t\,\alpha / \min(h_L h_R) \le 1/2$ on a non-uniform grid. The constructor refuses a configuration above its limit on either grid, because the explicit update diverges to NaN there rather than degrading. A $\Delta t$ handed to `update()`, or a `thermal_diffusivity` / `length` moved by calibration, bypasses that check — MADD-ANO-002.
2. **1st-order in time**: temporal accuracy is $O(\Delta t)$
3. **No convection**: pure diffusion only
4. **No radiation**: no radiative heat transfer
5. **Non-uniform grids are 2nd-order only**: `stencil_order=4` requires a uniform grid
6. **The reported flux is not the scheme's own face flux**: `compute_boundary_fluxes` returns the physical rod-end flux, reconstructed to $O(\Delta x^{\texttt{stencil\_order}})$, while the conservative update's first row uses the 1st-order face flux $-\alpha (T_0 - T_b)/(\Delta x/2)$. The two agree in the limit but not bit-for-bit, so a discrete energy balance closed against the reported flux carries that difference. Until 0.4.0 the method instead returned $-\alpha (T_1 - T_0)/\Delta x$, the flux at $x = \Delta x$ rather than at the rod end: a 10.6% error at $N = 10$ on $T = e^x$, converging at order 1.005 against the node's own 2.000
7. **Units**: `boundary_flux_spec` declares `K*m/s`, not `W/m^2`. $-\alpha\,\partial T/\partial x$ is the conductive flux divided by $\rho c_p$, and neither is a parameter of this node
8. **An end with no boundary input is insulated only at `stencil_order=2`** (MADD-ANO-031). With no `left_temperature` the datum is the end cell, $T_b = T_0$: at 2nd order the ghost is then $T_0$ and the face flux is exactly zero, but the 4th-order cubic through $(0, T_0)$ and the three end cells has a non-zero slope at the rod end, so heat crosses it — the mean of an $x^2$ profile on 64 cells falls from 0.333313 to 0.332048 in 2000 steps (float64), where 2nd order conserves it exactly. Pass explicit end temperatures, or use `stencil_order=2`, for an insulated rod
9. **Sharding**: `ShardedStencilNode(HeatNode)` is the unsharded node for the same boundary inputs on any number of devices (to float32 rounding; MADD-ANO-030 until 0.4.0). Hold an end at a temperature with `left_temperature` / `right_temperature`, exactly as unsharded; the wrapper's `boundary` must be `"edge"`
10. **Two rods coupled end to end have a lower limit** (MADD-ANO-050). When each rod's end-cell temperature is the other's Dirichlet datum and a coupling group converges the exchange, the pair is unstable above $3/8$ at `stencil_order=2` and $0.226$ at `stencil_order=4`; see Stability Conditions. `compile()` warns about the pattern it recognises; nothing refuses it

## Stability Conditions

For the explicit scheme the {term}`CFL number` $\mathrm{Fo} = \frac{\alpha \Delta t}{\Delta x^2}$ must satisfy

$$
\mathrm{Fo} < \mathrm{Fo}_{\max}(\texttt{stencil\_order}),
\qquad
\mathrm{Fo}_{\max} = \begin{cases} 1/2 & \texttt{stencil\_order} = 2 \\ 5/16 & \texttt{stencil\_order} = 4 \end{cases}
$$

Both are spectral bounds on the *whole* discrete operator, boundary rows included, not on the interior symbol alone.

For `stencil_order=2` the mirror ghost is exact on the operator's eigenvectors $\sin(m\pi x/L)$, so the closure adds nothing to the interior spectrum and the classical $1/2$ survives unchanged.

For `stencil_order=4` the bare 5-point symbol would allow $3/8$, but the cubic boundary closure raises the spectral radius: the sharp bound is $0.3169$ at $N = 5$, rising monotonically to $0.3249$ as $N \to \infty$. $5/16 = 0.3125$ is below all of them, so one number is safe at every resolution the node accepts.

On a non-uniform grid (`grid_points`) the stencil is the 2nd-order variable-spacing one whatever `stencil_order` says. Row $i$ has diagonal $-2/(h_L h_R)$ and off-diagonal weights summing to $2/(h_L h_R)$, where $h_L$ and $h_R$ are the spacings to the two neighbours and the end spacing repeats beyond each end. The Dirichlet end row is $(T_1 - 3T_0 + 2T_b)/h^2$. Gershgorin puts every eigenvalue in $[-4/\min(h_L h_R), 0]$. The operator is tridiagonal with positive off-diagonal products, so it is similar to a symmetric matrix and the eigenvalues are real. Hence

$$
\frac{\alpha\,\Delta t}{\min_i h_{L,i}\,h_{R,i}} \le \frac{1}{2},
$$

which is the uniform bound on a uniform grid, sufficient on any grid, and within $1.5\times$ of the sharp limit on the graded grids of `tests/nodes/test_heat_nonuniform_fourier_limit.py`.

**Two rods coupled end to end** (MADD-ANO-050). Converging the exchange makes each datum the other rod's end cell at the *new* time, while the interiors stay explicit, so the step is $(I - \mathrm{Fo}\,Q)\,T' = (I + \mathrm{Fo}\,P)\,T$. An amplification of $-1$ needs $(P - Q)v = -(2/\mathrm{Fo})\,v$. The mode that gets there first is mirror-symmetric about the interface: each rod sees a datum of minus its own end cell, and the mode decays away from the interface. At `stencil_order=2`, $v_k = r^k$ with $1 + 1/r = -2$, so $r = -1/3$, the eigenvalue is $-16/3$ and $\mathrm{Fo} = 3/8$. At `stencil_order=4` the two decaying roots of the 5-point recurrence must satisfy both cubic-closure rows. Their product $p$ is then the root in $(-1, 0)$ of $p^6 - 294p^5 - 2333p^4 + 1692p^3 - 9p^2 + 362p + 5$, the eigenvalue is $-8.8446$, and $\mathrm{Fo} = 0.2261266$ ($0.2261215$ at $N = 5$). The node records $0.226$. The cubic ghost puts $16/5$ of the datum into the end cell's row where the mirror ghost puts $2$, which is why the order-4 figure is so much lower.

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
| `initial_temperature` | float or array | 0.0 | K | Uniform initial temperature, or one value per cell |
| `stencil_order` | int | 2 | — | 2 or 4; 4 requires $N \geq 5$ and a uniform grid |
| `grid_points` | array or None | None | m | Non-uniform cell centres; `length` is then ignored and `param_specs()` declares it `trainable=False` |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| `left_temperature` | scalar | `T[0]` | Dirichlet BC at the left rod end, $x = 0$ |
| `right_temperature` | scalar | `T[-1]` | Dirichlet BC at the right rod end, $x = L$ |
| `heat_source` | `(n_cells,)` or scalar | 0.0 | Volumetric heat source term |

## References

- [@Crank1975] Crank, J. (1975). *The Mathematics of Diffusion*. Oxford University Press. — Analytical solutions for the heat equation used in {term}`verification benchmarks <Verification benchmark>`.
- [@LeVeque2007] LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM. — Convergence theory and stability analysis for the explicit FD scheme.

## Verification Evidence

- Benchmark: `MADD-VER-001` — Analytical solution comparison for constant-BC diffusion
- Benchmark: `MADD-VER-002` — Global spatial convergence against the Fourier solution; measured rate 1.900 against a theoretical 2.0
- Benchmark: `MADD-VER-005` — Spatial order of accuracy by the Method of Manufactured Solutions; measured 2.000
- Benchmark: `MADD-VER-006` — Temporal order of accuracy by MMS; measured 1.000
- Test files: `tests/verification/test_heat_analytical.py`, `tests/verification/test_mms_order.py`
- The `stencil_order=4` claim is measured at 3.957 over a 10/20/40/80/160 ladder by `tests/verification/test_mms_order.py::test_heat_fourth_order_stencil_converges_at_its_declared_spatial_order`

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2025-03-01 | Initial implementation |
| 2.0.0 | 2026-09-20 | Dirichlet data imposed at the rod ends through ghost cells instead of by overwriting the end cells (MADD-ANO-007); 4th-order ghosts rebuilt by cubic extrapolation (MADD-ANO-008); per-stencil Fourier bound documented and enforced at construction (MADD-ANO-009) |
