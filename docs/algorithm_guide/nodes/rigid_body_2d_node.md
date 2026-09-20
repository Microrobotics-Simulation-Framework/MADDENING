---
bibliography: ../../bibliography.bib
---

# RigidBody2DNode

**Module**: `maddening.nodes.rigid_body_2d`
**Stability**: deprecated
**Algorithm ID**: `MADD-NODE-004`
**Version**: 1.0.0

```{deprecated} 0.3.0
Use `maddening.nodes.rigid_body.RigidBodyNode` with
`constraints={"z": 0, "rx": 0, "ry": 0}`, which gives equivalent planar
behaviour inside the full 6-DOF node.  The constructor warns on every
instantiation.  This guide documents the node as it stands so that the
existing evidence for it is traceable.
```

## Summary

A planar rigid body with two translational degrees of freedom and one rotational, driven by an external force and torque applied at the centre of mass.

## Governing Equations

$$
m\,\ddot{\mathbf{x}} = \mathbf{F} + m\,\mathbf{g},
\qquad
I\,\ddot{\theta} = \tau
$$

where $\mathbf{x} = (x, y)$ is the centre-of-mass position, $\theta$ the orientation, $m$ the mass, $I$ the moment of inertia about the centre of mass, $\mathbf{F}$ the external force and $\tau$ the external torque.

## Discretization

Semi-implicit (symplectic) Euler in both degree-of-freedom groups: the linear and angular velocities are advanced first, and the position and angle are advanced with the **already-updated** velocities.

$$
\mathbf{v}^{n+1} = \mathbf{v}^n + \Delta t \frac{\mathbf{F}^n + m\mathbf{g}}{m},
\qquad
\omega^{n+1} = \omega^n + \Delta t \frac{\tau^n}{I}
$$

$$
\mathbf{x}^{n+1} = \mathbf{x}^n + \Delta t\,\mathbf{v}^{n+1},
\qquad
\theta^{n+1} = \theta^n + \Delta t\,\omega^{n+1}
$$

**Declared order of accuracy**: `DiscretizationOrder(spatial=None, temporal=1.0)`.
There is no spatial order — the node integrates an ODE. The temporal claim is 1st order globally over the state as a whole. For a state-independent force the position and angle *alone* converge at 2nd order, because they use the already-updated velocities; the declared order is the one that holds for every field, which is what a joint error norm measures.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\mathbf{F} + m\mathbf{g}$ (net force) | `maddening.nodes.rigid_body_2d.RigidBody2DNode.update` | `(force + gravity * mass) / mass`; gravity is cast to float32 |
| $\tau / I$ (angular acceleration) | `maddening.nodes.rigid_body_2d.RigidBody2DNode.update` | `alpha = torque / inertia` |
| Linear velocity update | `maddening.nodes.rigid_body_2d.RigidBody2DNode.update` | `v + acceleration * dt` |
| Angular velocity update | `maddening.nodes.rigid_body_2d.RigidBody2DNode.update` | `omega + alpha * dt` |
| Position update ($\dot{\mathbf{x}} = \mathbf{v}$) | `maddening.nodes.rigid_body_2d.RigidBody2DNode.update` | `x + v_new * dt`, using the **new** velocity |
| Angle update ($\dot\theta = \omega$) | `maddening.nodes.rigid_body_2d.RigidBody2DNode.update` | `angle + omega_new * dt`, using the **new** angular velocity |
| $\mathbf{F}$, $\tau$ (external loads) | `maddening.nodes.rigid_body_2d.RigidBody2DNode.boundary_input_spec` | `force` and `torque`, both `coupling_type="additive"` |

## Assumptions and Simplifications

1. Rigid body: no deformation
2. Constant mass and moment of inertia
3. Planar motion only — no out-of-plane translation or rotation
4. Forces and torques are applied at the centre of mass
5. No contact, collision or constraint handling

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| `mass` | $10^{-6}$ – $10^{6}$ kg | Zero mass divides by zero |
| `inertia` | $10^{-6}$ – $10^{6}$ kg·m² | Zero inertia divides by zero |

## Known Limitations and Failure Modes

1. **Deprecated**: superseded by `RigidBodyNode` with planar constraints
2. **1st-order integration**: the error is $O(\Delta t)$ over the state as a whole
3. **No collision detection**: bodies can overlap with no penalty
4. **No constraint handling**: no joints or hinges
5. **Gravity is truncated to float32** inside `update()`, so the gravitational term carries about $10^{-7}$ relative error even under `jax_enable_x64`; the manufactured-solution study below sets gravity to zero, so the measurement is unaffected
6. No `derivatives()`, so `integrate_node` and the implicit solver cannot drive this node

## Stability Conditions

Unconditionally stable for a state-independent force and torque: the accelerations do not depend on the state, so there is no amplification factor to bound. A force supplied by a coupled node that depends on this node's position reintroduces the usual explicit-scheme limit.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `x` | `(2,)` | m | Centre-of-mass position $[x, y]$ |
| `angle` | scalar | rad | Orientation $\theta$ |
| `v` | `(2,)` | m/s | Linear velocity $[v_x, v_y]$ |
| `omega` | scalar | rad/s | Angular velocity |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `mass` | float | 1.0 | kg | Body mass $m$ |
| `inertia` | float | 1.0 | kg·m² | Moment of inertia $I$ |
| `gravity` | tuple | (0.0, -9.81) | m/s² | Gravitational acceleration $[g_x, g_y]$ |
| `initial_x`, `initial_y` | float | 0.0 | m | Initial position |
| `initial_vx`, `initial_vy` | float | 0.0 | m/s | Initial velocity |
| `initial_angle` | float | 0.0 | rad | Initial orientation |
| `initial_omega` | float | 0.0 | rad/s | Initial angular velocity |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| `force` | `(2,)` | `[0, 0]` | External force $[F_x, F_y]$, additive |
| `torque` | scalar | 0.0 | External torque $\tau$, additive |

## References

- [@Hairer2006] Hairer, E., Lubich, C. and Wanner, G. (2006). *Geometric Numerical Integration*. Springer. — Chapter VI: order and structure preservation for symplectic Euler, the scheme used here in both DOF groups.
- [@LeVeque2007] LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM. — Convergence theory for one-step ODE methods.
- [@Roache2002] Roache, P.J. (2002). Code verification by the Method of Manufactured Solutions. *J. Fluids Eng.* — The method the order claim above is measured with.

## Verification Evidence

- Benchmark: `MADD-VER-011` — observed temporal order of accuracy by the Method of Manufactured Solutions. A manufactured planar trajectory and orientation history are injected as `force` and `torque`, the node's own additive boundary inputs, with gravity zeroed so the manufactured forcing is the only drive. Over a 100/200/400/800 step ladder at fixed final time, in float64, the observed order over the finest pair is **1.000** against the declared 1.0, measured over position, angle, velocity and angular velocity jointly.
- Test file: `tests/verification/test_mms_order_ode_nodes.py`

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2025-03-01 | Initial implementation |
| 1.0.0 | 2026-09-20 | Declared order of accuracy added and measured (MADD-VER-011) |
