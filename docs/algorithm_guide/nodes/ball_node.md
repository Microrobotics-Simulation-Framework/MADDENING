---
bibliography: ../../bibliography.bib
---

# BallNode

**Module**: `maddening.nodes.ball`
**Stability**: stable
**Algorithm ID**: `MADD-NODE-001`
**Version**: 1.0.0

## Summary

A point mass falling under gravity in one dimension, with an optional perfectly rigid collision surface supplied as the boundary input `table_position`.

## Governing Equations

$$
\dot{v} = g, \qquad \dot{x} = v
$$

with, at contact ($x < x_{\text{table}}$), the restitution law

$$
x \leftarrow x_{\text{table}}, \qquad v \leftarrow -e\,v
$$

where $e$ is the coefficient of restitution. Below a velocity threshold of $10^{-4}$ the bounce is replaced by $v \leftarrow 0$, which is what stops the ball chattering on the surface.

## Discretization

Semi-implicit (symplectic) Euler: the velocity is advanced first and the position is advanced with the **already-updated** velocity.

$$
v^{n+1} = v^n + g\,\Delta t, \qquad x^{n+1} = x^n + \Delta t\, v^{n+1}
$$

Collision is resolved once per step, after the integration, with `jnp.where` so the whole update stays traceable.

**Declared order of accuracy**: `DiscretizationOrder(spatial=None, temporal=1.0)`.
There is no spatial order — the node integrates an ODE. The temporal claim is 1st order globally and covers the **smooth regime only**: a contact is a non-smooth event and no order is claimed across one.

```{warning}
`NodeMeta.discretization` names *forward* Euler, and the implementation is
semi-implicit Euler as written above.  Both are 1st order, so the declared
order is unaffected, but the sign of the leading position error and the
long-run energy behaviour are not what the metadata implies, and
`update()` does not agree with
`maddening.core.simulation.integrators.integrate_node(node, ..., method="euler")`.
Recorded as **MADD-ANO-011**.
```

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\dot v = g$ (gravity) | `maddening.nodes.ball.BallNode.update` | `velocity + gravity * dt`; `gravity` is an injectable parameter |
| $\dot x = v$ (kinematics) | `maddening.nodes.ball.BallNode.update` | `position + velocity * dt`, using the **new** velocity |
| Contact detection | `maddening.nodes.ball.BallNode.update` | `hit = position < table_pos`, branch-free via `jnp.where` |
| Restitution $v \to -e v$ | `maddening.nodes.ball.BallNode.update` | Nested `jnp.where`; below $10^{-4}$ the velocity is zeroed instead |
| $x_{\text{table}}$ (surface) | `maddening.nodes.ball.BallNode.boundary_input_spec` | `table_position`; omitting it disables contact entirely |
| Continuous right-hand side | `maddening.nodes.ball.BallNode.derivatives` | Collision-free; for `maddening.core.simulation.integrators.integrate_node` |

## Assumptions and Simplifications

1. Point mass: no rotational dynamics
2. Perfectly rigid collision surface
3. Constant coefficient of restitution (not velocity-dependent)
4. One translational degree of freedom
5. No air resistance or drag

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| `elasticity` | $0$ – $1$ | $e = 0$ perfectly inelastic, $e = 1$ perfectly elastic |
| `timestep` | $10^{-4}$ – $10^{-1}$ s | Tested range; smaller is more accurate |

## Known Limitations and Failure Modes

1. **No forcing input.** `boundary_input_spec` offers only `table_position`, a collision surface. A caller cannot drive this node with an external force; the only way in is the `gravity` parameter, which is the acceleration itself. This is why `MADD-VER-010` injects its manufactured source through `params`.
2. **1st-order integration**: the error is $O(\Delta t)$ in position and velocity.
3. **Tunnelling**: contact is tested once per step, so $|v|\Delta t$ larger than the gap passes through the surface.
4. **No order across a contact**: the restitution law is non-smooth, so the declared order does not apply to a step containing a bounce.
5. `derivatives()` ignores injected `params` and returns the gravity as float32.
6. **Scheme/metadata mismatch**: MADD-ANO-011, above.
7. **Gradients through a bounce omit the contact-time derivative** (MADD-ANO-021). Contact is a per-step branch, so `jax.grad` differentiates the branch the step took and never sees the moment of contact. With $\Delta t = 10^{-3}$, a drop from 1.0 and one bounce before $T = 0.8$ s, $\partial x(T)/\partial h$ is exactly $0$ against the exact $+0.5892$. The clamp pins the position to a constant, and the impact velocity depends on the number of steps the fall took rather than on $h$. $\partial x(T)/\partial |g|$ is $+0.0651$ against $+0.0051$. $\partial x(T)/\partial e$ is right, because the contact time does not depend on $e$. Values are correct to one step of travel ($3 \times 10^{-3}$), and nothing warns. This is a framework limitation shared by every node that branches on its own state (see the node-authoring guide, "Events inside `update()`"). Pinned in `tests/nodes/test_ball_event_gradient.py`.

## Stability Conditions

The collision-free scheme is unconditionally stable: the acceleration does not depend on the state, so there is no amplification factor to bound. The practical limit is tunnelling, not stability: keep $|v|\,\Delta t$ well below the distance to the surface.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `position` | scalar | m | Height of the ball |
| `velocity` | scalar | m/s | Vertical velocity |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `initial_position` | float | 0.0 | m | Starting height |
| `initial_velocity` | float | 0.0 | m/s | Starting velocity |
| `elasticity` | float | 0.8 | — | Coefficient of restitution $e$ |
| `gravity` | float | -9.81 | m/s² | Gravitational acceleration |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| `table_position` | scalar | absent | Surface position for collision; contact is skipped entirely when omitted |

## References

- [@Hairer2006] Hairer, E., Lubich, C. and Wanner, G. (2006). *Geometric Numerical Integration*. Springer. — Chapter VI: the order and the structure-preserving behaviour of the symplectic Euler scheme this node implements.
- [@LeVeque2007] LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM. — Convergence theory for one-step ODE methods.
- [@Roache2002] Roache, P.J. (2002). Code verification by the Method of Manufactured Solutions. *J. Fluids Eng.* — The method the order claim above is measured with.

## Verification Evidence

- Benchmark: `MADD-VER-010` — observed temporal order of accuracy by the Method of Manufactured Solutions, in the collision-free regime. Because the node exposes no force input, the manufactured acceleration is injected as a time-varying `gravity` parameter. Over a 100/200/400/800 step ladder at fixed final time, in float64, the observed order over the finest pair is **1.002** against the declared 1.0.
- Test file: `tests/verification/test_mms_order_ode_nodes.py`
- Anomaly: `MADD-ANO-011`, pinned as a strict xfail in the same file.

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2025-03-01 | Initial implementation |
| 1.0.0 | 2026-09-20 | Declared order of accuracy added and measured (MADD-VER-010); MADD-ANO-011 recorded |
