---
bibliography: ../../bibliography.bib
---

# HeartPumpNode

**Module**: `maddening.nodes.heart_pump`
**Stability**: experimental
**Algorithm ID**: `MADD-NODE-008`
**Version**: 1.0.0

## Summary

A 2-element Windkessel model of the systemic arterial tree [@Westerhof2009]: a compliant reservoir emptying through a peripheral resistance, filled by a pulsatile cardiac inflow. It produces a physiologically plausible arterial pressure waveform for driving a downstream vascular domain.

## Governing Equations

$$
C\,\frac{dP_{\text{art}}}{dt} = Q_{\text{heart}}(t) - \frac{P_{\text{art}} - P_{\text{down}}}{R}
$$

with the cardiac inflow sinusoidal during systole and zero during diastole,

$$
Q_{\text{heart}}(\phi) =
\begin{cases}
Q_{\max}\,\sin\!\left(\dfrac{\pi \phi}{f_s}\right), & \phi < f_s \\[2ex]
0, & \phi \geq f_s
\end{cases}
\qquad
\dot\phi = \frac{\text{HR}}{60},\quad \phi \in [0, 1)
$$

$Q_{\max}$ is fixed by the stroke volume: integrating the half-sine over systole gives $Q_{\max} = \text{SV} \cdot \pi f / (2 f_s)$ with $f = \text{HR}/60$.

## Discretization

Explicit, 1st order in time. Per step:

$$
\phi^{n+1} = \left(\phi^n + \Delta t\,\frac{\text{HR}}{60}\right) \bmod 1,
\qquad
P^{n+1} = P^n + \frac{\Delta t}{C}\left[Q_{\text{heart}}(\phi^{n+1}) - \frac{P^n - P_{\text{down}}^n}{R}\right]
$$

The phase advance is exact — the phase is linear in $t$ — so the only discretisation error is in the pressure.

**Declared order of accuracy**: `DiscretizationOrder(spatial=None, temporal=1.0)`.
There is no spatial order: the arterial compartment is lumped, with no spatial extent.

```{warning}
`NodeMeta.discretization` names *forward* Euler, which would place every
right-hand-side term at $t^n$.  As written above, the inflow is evaluated
at $\phi^{n+1}$ — the **end** of the step — while the outflow uses $P^n$.
The node's own `derivatives()` evaluates the same waveform at $\phi^n$, so
the explicit path and the `derivatives()`-based paths
(`integrate_node`, `implicit_residual`) integrate inflow waveforms offset
by one timestep.  Both samplings are 1st order, so the declared order is
unaffected.  Recorded as **MADD-ANO-012**.

`backpressure` is also cast to float32 unconditionally, which pins the
coupling variable to single precision under any caller precision
(**MADD-ANO-013**).
```

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\dot\phi = \text{HR}/60$ (cycle phase) | `maddening.nodes.heart_pump.HeartPumpNode.update` | `jnp.fmod(phase + dt * hr / 60, 1.0)`; exact, not approximated |
| $Q_{\text{heart}}(\phi)$ (cardiac inflow) | `maddening.nodes.heart_pump._cardiac_output` | Half-sine during systole, zero otherwise, branch-free via `jnp.where` |
| $Q_{\max}$ from stroke volume | `maddening.nodes.heart_pump.HeartPumpNode._compute_q_max` | $\text{SV}\,\pi f / (2 f_s)$ |
| $(P_{\text{art}} - P_{\text{down}})/R$ (outflow) | `maddening.nodes.heart_pump.HeartPumpNode.update` | `Q_out`; `P_downstream` is cast to float32 (MADD-ANO-013) |
| $P_{\text{down}}$ (downstream pressure) | `maddening.nodes.heart_pump.HeartPumpNode.boundary_input_spec` | `backpressure`; falls back to `venous_pressure` when absent |
| Time integration of $P$ | `maddening.nodes.heart_pump.HeartPumpNode.update` | `P_art + dP_dt * dt` |
| Continuous right-hand side | `maddening.nodes.heart_pump.HeartPumpNode.derivatives` | Samples the inflow at $\phi^n$, not $\phi^{n+1}$ (MADD-ANO-012) |
| Backward-Euler residual | `maddening.nodes.heart_pump.HeartPumpNode.implicit_residual` | $x^{n+1} - x^n - \Delta t f(x^{n+1})$ |
| $P_{\text{art}}$ published downstream | `maddening.nodes.heart_pump.HeartPumpNode.compute_boundary_fluxes` | `inlet_pressure` |

## Assumptions and Simplifications

1. Lumped parameters: the arterial compartment is spatially uniform
2. Rigid arterial walls — compliance is constant
3. Sinusoidal systolic flow waveform
4. Instantaneous valve opening and closing; no valve dynamics
5. No inertial effects (2-element model, no inductance)
6. No wave propagation, no coronary circulation, no venous return dynamics

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| `heart_rate` | 40 – 180 bpm | |
| `systole_fraction` | 0.2 – 0.5 | Divides the phase; must stay strictly inside $(0, 1)$ |
| `resistance` | 0.01 – 100 | |
| `compliance` | 0.001 – 100 | |
| `timestep` | *not bounded* | No validated range is recorded; see the limitations below |

## Known Limitations and Failure Modes

1. **1st-order integration**: the pressure error is $O(\Delta t)$
2. **No stability check on $\Delta t$** relative to the $RC$ time constant; forward Euler on this ODE is stable only for $\Delta t < 2RC$, and nothing enforces it
3. **Negative pressures** are reachable with a large $\Delta t$ or a low compliance
4. **Source sampled at the end of the step**: MADD-ANO-012, above
5. **`backpressure` truncated to float32**: MADD-ANO-013, above. A float64 convergence study is floored at about $10^{-6}$ relative error, and a weakly typed float64 pressure state is demoted to float32 by the same cast, which makes `lax.scan` reject the carry
6. **`derivatives()` leaves `flow_rate` where it started.** It and `implicit_residual()` take the injected `params` by the same `{**self.params, **params}` rule as `update()` (MADD-ANO-018, resolved in 0.4.0), but `derivatives()` returns a zero rate for `flow_rate`, which `update()` recomputes from the phase every step: after 20 steps of 0.01 s from rest, `update()` reports 314.6 and `integrate_node` still 0.0
7. The inflow waveform is continuous but has a corner at the systole/diastole transition, so it is not $C^1$; higher-order integrators would not recover their order across it

## Stability Conditions

The homogeneous part of the pressure ODE decays with time constant $RC$. Forward Euler is stable for

$$
\Delta t < 2 R C
$$

and non-oscillatory for $\Delta t < RC$. With the defaults ($R = C = 1$) that is a 2-second bound, so stability is rarely the binding constraint; accuracy relative to the systolic pulse is (see the workaround recorded with MADD-ANO-012).

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `arterial_pressure` | scalar | Pa | Arterial pressure $P_{\text{art}}$ |
| `phase` | scalar | — | Position in the cardiac cycle, $[0, 1)$ |
| `flow_rate` | scalar | volume/s | Cardiac inflow at the end of the step |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `resistance` | float | 1.0 | — | Peripheral vascular resistance $R$ |
| `compliance` | float | 1.0 | — | Arterial compliance $C$ |
| `heart_rate` | float | 72.0 | bpm | Heart rate |
| `stroke_volume` | float | 70.0 | — | Volume ejected per beat |
| `venous_pressure` | float | 0.0 | Pa | Downstream pressure when `backpressure` is absent |
| `systole_fraction` | float | 0.35 | — | Fraction of the cycle that is systole, $f_s$ |
| `initial_pressure` | float | 80.0 | Pa | Starting arterial pressure |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| `backpressure` | scalar | `venous_pressure` | Downstream pressure feedback, e.g. from an LBM outlet |

## Boundary Fluxes

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| `inlet_pressure` | scalar | Pa | Arterial pressure, for a downstream vascular domain |

## References

- [@Westerhof2009] Westerhof, N., Lankhaar, J.-W. and Westerhof, B.E. (2009). The arterial Windkessel. *Med. Biol. Eng. Comput.* 47(2), 131–141. — The 2-element model implemented here, and the limits of a lumped description of the arterial tree.
- [@LeVeque2007] LeVeque, R.J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM. — Convergence and stability of forward Euler on a linear decay ODE.
- [@Roache2002] Roache, P.J. (2002). Code verification by the Method of Manufactured Solutions. *J. Fluids Eng.* — The method the order claim above is measured with.

## Verification Evidence

- Benchmark: `MADD-VER-012` — observed temporal order of accuracy by the Method of Manufactured Solutions. A manufactured arterial pressure history is injected through `backpressure`, which enters the outflow linearly, so the downstream pressure that makes the trajectory exact is available in closed form. Over a 200/400/800/1600 step ladder across one second of cardiac cycles, in float64, the observed order over the finest pair is **1.000** against the declared 1.0. The ladder stops at 1600 steps because MADD-ANO-013 turns it over below about $6 \times 10^{-6}$ relative error, two orders of magnitude finer than the $1.9 \times 10^{-3}$ this ladder reaches.
- Test file: `tests/verification/test_mms_order_ode_nodes.py`
- Anomalies: `MADD-ANO-012` and `MADD-ANO-013`, both pinned as strict xfails in the same file.

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2025-03-01 | Initial implementation |
| 1.0.0 | 2026-09-20 | Declared order of accuracy added and measured (MADD-VER-012); MADD-ANO-012 and MADD-ANO-013 recorded |
