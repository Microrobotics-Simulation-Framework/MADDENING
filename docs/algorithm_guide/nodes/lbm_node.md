---
bibliography: ../../bibliography.bib
---

# LBMNode

**Module**: `maddening.nodes.lbm`
**Stability**: experimental
**Algorithm ID**: `MADD-NODE-007`
**Version**: 1.1.0

## Summary

A lattice Boltzmann solver for weakly compressible, isothermal, laminar flow on a D2Q9 or D3Q19 lattice [@Kruger2017], with a BGK collision operator, Guo body forcing [@Guo2002], bounce-back walls on an arbitrary wall mask, and Zou-He pressure (density) boundaries on a chosen inlet and outlet face [@ZouHe1997; @HechtHarting2010].

## Governing Equations

The lattice Boltzmann equation with the BGK collision operator and a forcing term,

$$
f_q(\mathbf{x} + \mathbf{e}_q, t + 1) = f_q(\mathbf{x}, t) - \frac{1}{\tau}\left[f_q - f_q^{\text{eq}}\right] + S_q,
$$

with the second-order equilibrium

$$
f_q^{\text{eq}}(\rho, \mathbf{u}) = w_q\,\rho\left[1 + \frac{\mathbf{e}_q\cdot\mathbf{u}}{c_s^2} + \frac{(\mathbf{e}_q\cdot\mathbf{u})^2}{2c_s^4} - \frac{\mathbf{u}\cdot\mathbf{u}}{2c_s^2}\right],
\qquad c_s^2 = \tfrac13,
$$

and the moments

$$
\rho = \sum_q f_q, \qquad \rho\mathbf{u} = \sum_q f_q\,\mathbf{e}_q + \tfrac12\mathbf{F}, \qquad p = c_s^2 \rho.
$$

In the low-Mach limit this recovers the Navier-Stokes equations with kinematic viscosity $\nu = c_s^2(\tau - \tfrac12)$ and the isothermal equation of state $p = c_s^2\rho$. The node takes `viscosity` and sets $\tau = \tfrac12 + \nu / c_s^2$.

**Pressure boundary.** On an inlet or outlet face the node imposes a density $\rho_p = p / c_s^2$ and zero tangential velocity, and solves for the face-normal velocity $u_n$ and the populations that stream into the domain through the face.

## Discretization

Lattice units, $\Delta x = \Delta t = 1$; `update` ignores its `dt` argument. One call of `update` is, in order:

1. moments $\rho$, $\mathbf{u}$ from $f$ (with the Guo half-force correction);
2. BGK collision plus the Guo source term $S_q = (1 - \tfrac{1}{2\tau})\,w_q\left[\frac{\mathbf{e}_q - \mathbf{u}}{c_s^2} + \frac{\mathbf{e}_q\cdot\mathbf{u}}{c_s^4}\mathbf{e}_q\right]\cdot\mathbf{F}$;
3. streaming, periodic on every axis (`jnp.roll`), or across a halo in the sharded path;
4. bounce-back: in every wall cell each population is replaced by its opposite;
5. the Zou-He closure on the inlet face, then on the outlet face (fluid cells only);
6. moments of the result, the velocity zeroed in wall cells, $p = c_s^2 \rho$.

**Declared order of accuracy**: `DiscretizationOrder(spatial=2.0, temporal=None)`, measured on a wall-free periodic domain by MADD-VER-007. There is no independent timestep, so no temporal order is declared. Walls are first order (see the limitations).

### Zou-He pressure boundary

Let $n$ be the face axis and $\sigma = +1$ on a `*_min` face, $\sigma = -1$ on a `*_max` face, so that $\sigma\hat{\mathbf{n}}$ points into the domain. After streaming, the directions on the face split into three sets:

| Set | Definition | Meaning |
|-----|------------|---------|
| $U$ (unknown) | $e_{qn} = \sigma$ | arrive from outside the domain; rebuilt by the closure |
| $K$ (known) | $e_{qn} = -\sigma$ | streamed out of the interior; the opposites of $U$ |
| $T$ (tangential) | $e_{qn} = 0$ | the rest population included |

With $S_X = \sum_{q\in X} f_q$, the density and the normal momentum of the face are

$$
\rho_p = S_T + S_K + S_U, \qquad \rho_p u_n = \sigma\,(S_U - S_K),
$$

and eliminating the unknown $S_U$ gives the face-normal velocity

$$
u_n = \sigma\left[1 - \frac{S_T + 2 S_K}{\rho_p}\right].
$$

The known populations appear twice: once in the density and once, reflected, in the momentum. For each unknown $q$ with opposite $\bar q$, non-equilibrium bounce-back of the normal part plus a transverse-momentum correction gives

$$
f_q = f_{\bar q} + \frac{2 w_q}{c_s^2}\,\rho_p\,e_{qn}\,u_n - \sum_{t\neq n} \frac{e_{qt}\,N_t}{\sum_{p\in U} e_{pt}^2},
\qquad N_t = \sum_{p\in T} f_p\,e_{pt}.
$$

The rebuilt face has exactly density $\rho_p$ and zero tangential momentum. This rests on four identities of the velocity set, which `_zou_he_face_closure` checks for each face before the closure runs, refusing a set that fails one: the opposites of $U$ are exactly $K$; $2\sum_{U} w_q e_{qn}^2 / c_s^2 = 1$, so the bounce-back terms add up to $\rho_p u_n$; $\sum_U e_{qt} = \sum_U w_q e_{qn} e_{qt} = 0$, so the correction leaves density and normal momentum alone; and $\sum_U e_{qt} e_{qt'} = 0$ for $t \neq t'$, so each tangential axis is corrected on its own. On both supported lattices $\sum_U e_{pt}^2 = 2$ on every face, so the coefficient is $\tfrac12$.

**D2Q9** (this module's numbering: 1 $+x$, 2 $-x$, 3 $+y$, 4 $-y$, 5 $(+,+)$, 6 $(-,+)$, 7 $(+,-)$, 8 $(-,-)$), `x_min`:

$$
u = 1 - \frac{f_0 + f_3 + f_4 + 2(f_2 + f_6 + f_8)}{\rho_p},\quad
f_1 = f_2 + \tfrac23\rho_p u,\quad
f_5 = f_8 + \tfrac16\rho_p u - \tfrac12(f_3 - f_4),\quad
f_7 = f_6 + \tfrac16\rho_p u + \tfrac12(f_3 - f_4).
$$

`x_max` rebuilds $f_2$, $f_6$, $f_8$ from $f_1$, $f_7$, $f_5$, with $u = -1 + [f_0 + f_3 + f_4 + 2(f_1 + f_5 + f_7)]/\rho_p$ and the signs of the $\rho_p u$ terms reversed. These are the equations of Zou and He [@ZouHe1997] with $u_y = 0$. The $y$ faces follow by exchanging the roles of the axes.

**D3Q19**, `x_min`: $f_{(1,0,0)} = f_{(-1,0,0)} + \tfrac13\rho_p u$, and for the four edge directions

$$
f_{(1,\pm1,0)} = f_{(-1,\mp1,0)} + \tfrac16\rho_p u \mp \tfrac12 N_y, \qquad
f_{(1,0,\pm1)} = f_{(-1,0,\mp1)} + \tfrac16\rho_p u \mp \tfrac12 N_z,
$$

where $N_y$ and $N_z$ sum over the nine tangential populations, the $(0,\pm1,\pm1)$ edges included. This is the D3Q19 on-site closure of Hecht and Harting [@HechtHarting2010] with zero tangential velocity. The other five faces follow by symmetry.

Wall cells on a pressure face keep all of their populations. The closure is also used unchanged by the sharded step `update_padded`, which is valid while the inlet/outlet axis is not sharded.

```{note}
Before 0.4.0 the closure computed $u_n = \sigma[1 - (S_T + S_K)/\rho_p]$, with the factor 2 on $S_K$ missing, and had no transverse correction. The face density came out as $\rho_p + S_K$: 15.4% high for $p = 0.36$ on a unit-density lattice, on every face of both lattices. The pressure-driven channels measured carried 0.58 to 0.80 of the imposed pressure drop, and `outlet_pressure_avg` reported the wrong pressure. Recorded as **MADD-ANO-020**; MADD-VER-016 is the benchmark that would have caught it.
```

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| Velocity sets $\mathbf{e}_q$, $w_q$, opposites | `maddening.nodes.lbm.d2q9`, `maddening.nodes.lbm.d3q19` | numpy arrays, concrete inside `jit` |
| $f_q^{\text{eq}}(\rho, \mathbf{u})$ | `maddening.nodes.lbm._equilibrium` | Second-order Hermite equilibrium |
| $\rho$, $\rho\mathbf{u} = \sum f_q\mathbf{e}_q + \tfrac12\mathbf{F}$ | `maddening.nodes.lbm._compute_macroscopic` | Guo half-force correction |
| BGK collision, $\tau = \tfrac12 + \nu/c_s^2$ | `maddening.nodes.lbm.LBMNode.update` | `viscosity` read from the injected params when the graph supplies them |
| Guo source term $S_q$ | `maddening.nodes.lbm._guo_forcing` | [@Guo2002] |
| Streaming, periodic | `maddening.nodes.lbm._stream` | `jnp.roll` along each axis |
| Streaming across a halo (sharded) | `maddening.nodes.lbm._stream_padded`, `maddening.nodes.lbm.LBMNode.update_padded` | Slicing into halo-exchanged neighbours |
| Bounce-back in wall cells | `maddening.nodes.lbm.LBMNode.update` | `f_streamed[..., opp]` where the runtime wall mask is set |
| Runtime wall mask | `maddening.nodes.lbm.LBMNode._runtime_wall_mask` | `wall_mask_update`, else the `wall_mask` state field, else the constructor's mask |
| Opposite-direction map $\bar q$ | `maddening.nodes.lbm._get_opp_map` | $\mathbf{e}_{\bar q} = -\mathbf{e}_q$ |
| Sets $U$, $K$, $T$ of a face | `maddening.nodes.lbm._classify_directions` | By the sign of $e_{qn}$ |
| Lattice identities of the closure | `maddening.nodes.lbm._zou_he_face_closure` | Refuses a velocity set that fails one; returns $\sigma$ and the transverse coefficients |
| $u_n = \sigma[1 - (S_T + 2S_K)/\rho_p]$ | `maddening.nodes.lbm._zou_he_pressure_face` | [@ZouHe1997]; MADD-ANO-020 was the missing 2 |
| Unknown populations $f_q$ (bounce-back + transverse correction) | `maddening.nodes.lbm._zou_he_pressure_face` | [@ZouHe1997] (D2Q9), [@HechtHarting2010] (D3Q19) |
| $\rho_p = p/c_s^2$ from `inlet_pressure` / `outlet_pressure` | `maddening.nodes.lbm.LBMNode.update` | Inlet face first, then outlet |
| $p = c_s^2\rho$ | `maddening.nodes.lbm.LBMNode.update` | `pressure` state field |
| Outlet pressure published to a coupled node | `maddening.nodes.lbm.LBMNode.compute_boundary_fluxes` | Mean over the fluid cells of the outlet face, under the runtime wall mask |
| Initial condition $f = f^{\text{eq}}(1, \mathbf{0})$ | `maddening.nodes.lbm.LBMNode.initial_state` | The wall mask is carried as `uint8` state |
| Boundary inputs | `maddening.nodes.lbm.LBMNode.boundary_input_spec` | See below |
| Parameter bounds ($\nu > 0$) | `maddening.nodes.lbm.LBMNode.param_specs` | Log transform |

## Assumptions and Simplifications

1. Weakly compressible, isothermal flow: $p = c_s^2 \rho$, valid for Mach number $\ll 1$
2. BGK single-relaxation-time collision
3. Rigid, impermeable walls, modelled by bounce-back in wall cells
4. Pressure faces impose a density and zero tangential velocity; the face-normal velocity is whatever the interior delivers
5. Every axis without a pressure face or a wall is periodic

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| `tau` | 0.501 – 2.0 | $\tau > 0.5$ required; $\tau \gg 1$ adds numerical diffusion |
| Reynolds number | 0 – 100 | Laminar channel and pipe flow (MADD-VER-003, MADD-VER-016) |
| Inlet/outlet density difference | up to 1% of the mean | MADD-VER-016's range; larger drops add compressibility error |

## Known Limitations and Failure Modes

1. **Walls are first order, straight walls included.** Wall cells collide as well as reflect, so the hydrodynamic wall does not sit on the half-way plane. In MADD-VER-016's channel it sits about 0.1 lattice units from the wall node (0.097 at $H = 8$, $\tau = 1$; 0.12 at $\tau = 0.8$) instead of 0.5. The channel is effectively about 0.8 lattice units wider than nominal, and at $H = 8$ the centreline velocity is 20% above Hagen-Poiseuille. The excess halves with each doubling of $H$, and body-force driving shows nearly the same excess (+19.8% at $H = 8$).
2. **Pressure faces reflect acoustic waves.** A pressure-driven channel settles on a viscous time scale set by the slowest acoustic mode between the two faces, which is longer than the $H^2/\nu$ of the velocity profile.
3. **Entrance effect of a pressure face.** Near a face the gradient departs from the linear profile. In the middle half of MADD-VER-016's channel it is 0.91% steeper than $\Delta p / L$ at $H = 8$ and 0.18% at $H = 16$.
4. **Pressure faces and sharding**: `update_padded` applies the closure per shard, which is correct only while the inlet/outlet axis is replicated.
5. Compressibility errors at Mach number above about 0.1; no turbulence model; BGK is less stable than MRT at high Reynolds number.

## Stability Conditions

$\tau > \tfrac12$, which the constructor enforces through $\nu > 0$. The equilibrium is a low-Mach expansion, so $|\mathbf{u}| \lesssim 0.1$ in lattice units. $\tau$ close to $0.5$ with sharp gradients is prone to instability under BGK.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `f` | `(*grid_shape, Q)` | lattice | Distribution functions |
| `density` | `grid_shape` | lattice | $\rho = \sum_q f_q$ |
| `velocity` | `(*grid_shape, D)` | lattice | $\mathbf{u}$, zero in wall cells |
| `pressure` | `grid_shape` | lattice | $p = c_s^2\rho$ |
| `wall_mask` | `grid_shape` | — | Wall cells, `uint8` |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `grid_shape` | tuple of int | `(64, 32, 32)` | cells | 2 entries for D2Q9, 3 for D3Q19 |
| `viscosity` | float | 0.1 | lattice | $\nu$; trainable; $\tau = \tfrac12 + \nu/c_s^2$ |
| `lattice` | str | `"D3Q19"` | — | `"D3Q19"` or `"D2Q9"` |
| `wall_mask` | bool array or None | None | — | True in wall cells |
| `inlet_face` | str | `"x_min"` | — | One of `x_min` … `z_max` |
| `outlet_face` | str | `"x_max"` | — | One of `x_min` … `z_max` |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| `inlet_pressure` | scalar | none (no BC) | Pressure imposed on `inlet_face`, $\rho_p = p/c_s^2$ |
| `outlet_pressure` | scalar | none (no BC) | Pressure imposed on `outlet_face` |
| `body_force` | `(*grid_shape, D)` or `(D,)` | zero | Guo body force per unit volume |
| `wall_mask_update` | `grid_shape` | the state's mask | Runtime wall mask override |

## Boundary Fluxes

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `outlet_pressure_avg` | scalar | lattice | Mean pressure over the fluid cells of the outlet face (runtime wall mask); equals `outlet_pressure` when one is imposed |

## References

- [@Kruger2017] Krüger, T. et al. (2017). *The Lattice Boltzmann Method: Principles and Practice*. Springer. — BGK, equilibria, bounce-back.
- [@Guo2002] Guo, Z., Zheng, C., Shi, B. (2002). Discrete lattice effects on the forcing term in the lattice Boltzmann method. *Phys. Rev. E* 65, 046308. — Forcing term and half-force velocity.
- [@ZouHe1997] Zou, Q., He, X. (1997). On pressure and velocity boundary conditions for the lattice Boltzmann BGK model. *Phys. Fluids* 9, 1591. — The pressure closure (D2Q9).
- [@HechtHarting2010] Hecht, M., Harting, J. (2010). Implementation of on-site velocity boundary conditions for D3Q19 lattice Boltzmann simulations. *J. Stat. Mech.* P01018. — The D3Q19 transverse-momentum correction.

## Verification Evidence

- **MADD-VER-007** — spatial order 2 by manufactured solution (body-forced Kolmogorov flow, periodic): `tests/verification/test_mms_order.py`.
- **MADD-VER-016** — pressure-driven Poiseuille through the Zou-He faces, in absolute terms, under refinement; also the pressure drop and body-force equivalence checks: `tests/verification/test_lbm_pressure_poiseuille.py`.
- **MADD-VER-003** — body-force Hagen-Poiseuille in a pipe on the sharded path: `tests/cloud/multigpu/test_lbm_poiseuille.py`.
- Pressure face, cell by cell (density, tangential momentum, which populations change, fixed point, Zou and He's D2Q9 equations written out, node pressure field, reported outlet pressure, runtime wall mask): `tests/nodes/test_lbm_pressure_boundary.py`.
- Structural battery (finite, deterministic, jit-consistent, differentiable, every trainable parameter effective): `tests/verification/test_builtin_nodes_verified_lbm.py`.

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-03-16 | Initial implementation |
| 1.1.0 | 2026-09-24 | Zou-He closure corrected: the face now carries the prescribed density and zero tangential velocity (MADD-ANO-020). `outlet_pressure_avg` averages over the runtime wall mask |
