---
bibliography: ../../bibliography.bib
---

# SpringDamperNode

**Module**: `maddening.nodes.spring`
**Stability**: stable
**Algorithm ID**: `MADD-NODE-003`
**Version**: 1.0.0

## Summary

A linear spring with viscous damping connecting a point mass to a moving attachment point. The node owns one end of the spring; the other end arrives each step as the boundary input `anchor_position`.

## Governing Equations

$$
m \ddot{x} = -k\,(x - a(t) - \ell_0) - c\,\dot{x}
$$

where $x$ is the position of this end of the spring, $a(t)$ the anchor position, $\ell_0$ the rest length, $k$ the stiffness, $c$ the damping coefficient and $m$ the point mass.

## Discretization

Semi-implicit (symplectic) Euler: the velocity is advanced first and the position is advanced with the **already-updated** velocity.

$$
v^{n+1} = v^n + \frac{\Delta t}{m}\left[-k\,(x^n - a^n - \ell_0) - c\,v^n\right],
\qquad
x^{n+1} = x^n + \Delta t\, v^{n+1}
$$

**Declared order of accuracy**: `DiscretizationOrder(spatial=None, temporal=1.0)`.
There is no spatial order — the node integrates an ODE and has no grid. The temporal claim is 1st order globally in *both* state fields. Unlike a body under a state-independent force, the position does not gain an order here: the spring force depends on the position itself, so the leading error term does not cancel. The claim is measured, not asserted; see [Verification Evidence](#verification-evidence).

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $-k(x - a - \ell_0)$ (spring force) | `maddening.nodes.spring.SpringDamperNode.update` | `force = -k * (position - anchor - rest)` |
| $-c\dot{x}$ (viscous damping) | `maddening.nodes.spring.SpringDamperNode.update` | `- c * velocity`, same expression |
| $a(t)$ (anchor position) | `maddening.nodes.spring.SpringDamperNode.boundary_input_spec` | `anchor_position`; defaults to the origin when unsupplied |
| Velocity update ($\dot v = F/m$) | `maddening.nodes.spring.SpringDamperNode.update` | `velocity + (force / m) * dt` |
| Position update ($\dot x = v$) | `maddening.nodes.spring.SpringDamperNode.update` | `position + velocity * dt`, using the **new** velocity |
| Continuous right-hand side | `maddening.nodes.spring.SpringDamperNode.derivatives` | For `maddening.core.simulation.integrators.integrate_node`; not the scheme `update()` uses |
| Backward-Euler residual | `maddening.nodes.spring.SpringDamperNode.implicit_residual` | $x^{n+1} - x^n - \Delta t f(x^{n+1})$ |
| Reaction force delivered over a flux edge | `maddening.nodes.spring.SpringDamperNode.compute_boundary_fluxes` | `spring_force`, same constants as `update` |

## Assumptions and Simplifications

1. Linear spring (Hooke's law) — no hardening or softening
2. Viscous damping, linear in velocity
3. Point mass: no rotational dynamics
4. One translational degree of freedom
5. No collision detection with other objects

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| `stiffness` | $10^{-2}$ – $10^{6}$ N/m | Very stiff springs need a small $\Delta t$ |
| `damping` | $0$ – $10^{4}$ N·s/m | |
| `mass` | $> 0$ kg | Zero mass divides by zero; rejected in the constructor |

## Known Limitations and Failure Modes

1. **1st-order integration**: the error is $O(\Delta t)$ in both position and velocity
2. **No stability check**: $k \Delta t^2 / m \gtrsim 4$ makes the undamped scheme unstable, and nothing enforces it at runtime
3. **A calibrated constant reaches `derivatives()` and `implicit_residual()` only when it is passed.** Both take the injected `params` by the same `{**self.params, **params}` rule as `update()` (MADD-ANO-018, resolved in 0.4.0), so hand the node's `gm.params` entry to `integrate_node(..., params=)` or `implicit_euler_step(..., params=)`; called without it, they integrate the constructor constants
4. No nonlinear spring behaviour, and no contact

## Stability Conditions

Semi-implicit Euler applied to the undamped oscillator is stable for $\Delta t < 2\sqrt{m/k}$, i.e. $\omega \Delta t < 2$ with $\omega = \sqrt{k/m}$. Damping tightens the bound slightly.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `position` | scalar | m | Position of this end of the spring |
| `velocity` | scalar | m/s | Velocity of this end |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `stiffness` | float | 100.0 | N/m | Spring constant $k$ |
| `damping` | float | 1.0 | N·s/m | Damping coefficient $c$ |
| `mass` | float | 1.0 | kg | Point mass $m$ |
| `rest_length` | float | 1.0 | m | Natural length $\ell_0$ |
| `initial_position` | float | 0.0 | m | Starting position |
| `initial_velocity` | float | 0.0 | m/s | Starting velocity |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| `anchor_position` | scalar | 0.0 | Position of the other end of the spring |

## Boundary Fluxes

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `spring_force` | scalar | N | Spring-damper force for a downstream node |

## References

- [@Hairer2006] Hairer, E., Lubich, C. and Wanner, G. (2006). *Geometric Numerical Integration*. Springer. — Chapter VI derives the order and the energy behaviour of symplectic Euler, which is the scheme this node uses.
- [@LeVeque2007] LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM. — Convergence and stability theory for one-step ODE methods.
- [@Roache2002] Roache, P.J. (2002). Code verification by the Method of Manufactured Solutions. *J. Fluids Eng.* — The method the order claim above is measured with.

## Verification Evidence

- Benchmark: `MADD-VER-009` — observed temporal order of accuracy by the Method of Manufactured Solutions. A manufactured displacement is injected through `anchor_position`, which enters the force linearly, so the anchor that makes the trajectory exact is available in closed form. Over a 100/200/400/800 step ladder at fixed final time, in float64, the observed order over the finest pair is **1.029** against the declared 1.0.
- Test file: `tests/verification/test_mms_order_ode_nodes.py`
- The same file mutation-tests the study: a mis-scaled timestep, a timestep that drifts with the resolution (a monotone ladder at order 1/2) and a source frozen at $t=0$ are each required to fail the ladder and name this node.

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2025-03-01 | Initial implementation |
| 1.0.0 | 2026-09-20 | Declared order of accuracy added and measured (MADD-VER-009) |
