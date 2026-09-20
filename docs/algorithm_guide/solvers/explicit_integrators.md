---
bibliography: ../../bibliography.bib
---

# Explicit ODE Integrators

**Module**: `maddening.core.simulation.integrators`
**Stability**: provisional
**Algorithm ID**: `MADD-ALG-INT-001`
**Version**: 1.1.0

## Summary

Three explicit Runge-Kutta steppers for nodes that implement
`derivatives()`, plus a dispatch helper.  They are pure JAX functions,
so each composes with `jit`, `grad` and `scan`.

## Governing Equations

Each stepper advances one step of an initial-value problem

$$
\frac{dx}{dt} = f(x, u), \qquad x(t_n) = x_n
$$

where $x$ is the node state and $u$ the boundary inputs.

## Discretization

An $s$-stage explicit Runge-Kutta method with Butcher tableau
$(A, b, c)$ computes

$$
k_i = f\!\left(x_n + h \sum_{j<i} a_{ij} k_j,\; u_i\right),
\qquad
x_{n+1} = x_n + h \sum_{i=1}^{s} b_i k_i
$$

with $u_i$ the input at stage time $t_n + c_i h$.  The three tableaux:

| method | $c$ | $A$ | $b$ | order |
|---|---|---|---|---|
| `euler_step` | $(0)$ | — | $(1)$ | 1 |
| `heun_step` | $(0, 1)$ | $a_{21} = 1$ | $(\tfrac12, \tfrac12)$ | 2 |
| `rk4_step` | $(0, \tfrac12, \tfrac12, 1)$ | $a_{21}=a_{32}=\tfrac12,\; a_{43}=1$ | $(\tfrac16, \tfrac13, \tfrac13, \tfrac16)$ | 4 |

### What the order claim is conditional on

The signature `(derivatives_fn, state, boundary_inputs, dt)` carries no
time.  Every stage is therefore given the *same* `boundary_inputs`
object, which is $u_i = u(t_n)$ for all $i$ — a zero-order hold on the
input.

For an **autonomous** problem, or one whose inputs are constant over the
step, this is exact and the orders above are delivered in full.

For a **non-autonomous** problem driven the obvious way — one
`boundary_inputs` value per step — the hold contributes an
$\mathcal{O}(h^2)$ local error independently of the stage arithmetic, so
**every method in this module converges at order 1**.  This is registered
as `MADD-ANO-014`.  The stages are not wrong; they are being asked to
integrate a different problem from the one the caller has in mind.

### Reaching stage times without widening the interface

Autonomise: carry time as a state field with $dt/dt = 1$ [@HairerNorsettWanner1993].
Every stepper here builds its stage states as
$x_n + h\sum_j a_{ij} k_j$, so a field whose derivative is $1$ arrives at
stage $i$ holding exactly $t_n + c_i h$ — the Butcher node — and a
forcing read from it is read at the right time.  This is exact, not an
interpolation, and it uses only the existing `derivatives_fn` argument:

```python
def forced(state, _unused):
    t = state["time"]
    rest = {k: v for k, v in state.items() if k != "time"}
    return {**node.derivatives(rest, {"u": u_of(t)}), "time": jnp.asarray(1.0)}

state = {**node.initial_state(), "time": jnp.asarray(0.0)}
for _ in range(n_steps):
    state = rk4_step(forced, state, {}, dt)
```

Two alternatives were measured and rejected.  Interpolating the input
linearly between step endpoints caps RK4 at order 2.000, since the
interpolant's own $\mathcal{O}(h^2)$ error becomes the leading term.
Evaluating the frozen input at the step midpoint instead of its start
likewise caps Heun and RK4 at 2.000.  Neither reaches 4, and both would
need a second input argument; autonomisation reaches 4 with none.

## Implementation Mapping

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $x_{n+1} = x_n + h k_1$ (Euler update) | `maddening.core.simulation.integrators.euler_step` | $b = (1)$, one stage |
| $k_1, k_2$ and the trapezoid weights (Heun) | `maddening.core.simulation.integrators.heun_step` | $c = (0,1)$, $b = (\tfrac12,\tfrac12)$ |
| $k_1 \ldots k_4$ and $b = (\tfrac16,\tfrac13,\tfrac13,\tfrac16)$ (RK4) | `maddening.core.simulation.integrators.rk4_step` | Stage offsets $\tfrac12 h$, $\tfrac12 h$, $h$ |
| $f(x, u)$, the right-hand side | `maddening.core.node.SimulationNode.derivatives` | Supplied by the node; default raises |
| Method selection by name | `maddening.core.simulation.integrators.integrate_node` | Dispatch only; adds no arithmetic |

## Assumptions and Simplifications

1. $f$ is evaluated with the caller's `boundary_inputs` at every stage,
   i.e. a zero-order hold on the input across the step.
2. The state is a flat `dict` of array leaves and the returned state
   carries exactly the fields `derivatives_fn` returned a derivative for;
   a field it omits is dropped rather than held.
3. Fixed step size.  Nothing here estimates or adapts $h$; see
   `maddening.core.simulation.adaptive`.
4. Explicit methods only, so stiff problems are bounded by stability
   rather than accuracy; see `maddening.core.simulation.implicit`.

## Validated Physical Regimes

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| Observed temporal order, autonomous input | 1.003 / 2.005 / 4.006 | euler / heun / rk4, $dt$ ladder $0.075 \to 0.0047$, float64 |
| Observed temporal order, per-step input | 1.022 / 1.017 / 1.016 | Same ladder, sinusoidal $u(t)$ — `MADD-ANO-014` |
| Observed temporal order, time in state | 1.022 / 1.999 / 4.002 | Same ladder and $u(t)$, autonomised |
| $z = h \lambda$ for the identity check | $-2.5, -0.3, 0.7$ | One step reproduces $R(z)$ to $10^{-13}$ relative |

## Known Limitations and Failure Modes

1. **A time-varying input supplied once per step gives order 1 from
   every method** (`MADD-ANO-014`).  Nothing warns: the input dict is
   opaque to the stepper, which cannot tell a constant from a sample of
   a waveform.
2. **`rk4_step` can be less accurate than `euler_step`** under that
   regime — measured 1.5x worse at $dt = 0.0047$ — so selecting a
   higher-order method is not a conservative default.
3. **`integrate_node` cannot be fixed by its caller.** It passes
   `node.derivatives` straight through, and its signature has nowhere to
   put a stage time; a caller who needs one composes the autonomised
   function above and calls `rk4_step` directly.
4. **Inputs that are not functions of time alone** — data exchanged with
   another node during the step — cannot be reached at stage times by any
   single-node integrator.  The order is then set by the coupling scheme;
   see `docs/algorithm_guide/coupling/`.
5. Explicit stability limits apply. `rk4_step` is unstable for real
   $h\lambda < -2.785$, `heun_step` for $h\lambda < -2$, `euler_step` for
   $h\lambda < -2$.

## Stability Conditions

One step of $x' = \lambda x$ multiplies the state by the stability
polynomial $R(z)$, $z = h\lambda$, which is $\exp(z)$ truncated at the
method's order:

$$
R_{\text{euler}} = 1 + z, \qquad
R_{\text{heun}} = 1 + z + \tfrac{z^2}{2}, \qquad
R_{\text{rk4}} = 1 + z + \tfrac{z^2}{2} + \tfrac{z^3}{6} + \tfrac{z^4}{24}
$$

The step is stable where $|R(z)| \le 1$.  $R$ also serves as an exact
identity check on the coefficients, which a convergence ladder cannot
provide: under a frozen time-varying input all three methods measure
order ~1.02, so the *order* alone cannot distinguish one scheme from
another.  The errors do not all agree — `heun` and `rk4` do, to three
significant figures (7.038e-3 and 7.034e-3 relative at $dt = 0.0047$),
but `euler` sits at 4.567e-3, a factor of 1.540 better than either, which
is the same measurement the limitation above states as "1.5x worse".  An
earlier revision of this sentence said all three errors agreed, which
would have made the two paragraphs contradict each other.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| caller-defined | any | any | Whatever `derivatives_fn` returns a derivative for |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| `dt` | float | — | s | Step size, fixed |
| `method` | str | `"rk4"` | — | `integrate_node` only: `"euler"`, `"heun"`, `"rk4"` |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| caller-defined | any | — | Passed unchanged to every stage |

## References

- [@HairerNorsettWanner1993] Hairer, E., Nørsett, S. P. & Wanner, G. (1993). *Solving Ordinary Differential Equations I: Nonstiff Problems*. Springer. — Butcher tableaux, the order conditions, stability polynomials, and the autonomisation of a non-autonomous system (§II.1-II.3).
- [@Hairer2006] Hairer, E., Lubich, C. & Wanner, G. (2006). *Geometric Numerical Integration*. Springer. — Why a fixed-step explicit RK method's long-time behaviour is not implied by its order.

## Verification Evidence

`tests/verification/test_integrator_order.py` — the three studies in the
table above, plus the stability-polynomial identity and the
`integrate_node` dispatch pin.  `tests/verification/hypothesis/test_hypothesis_integrators.py`
covers the autonomous orders under Hypothesis.

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-01-15 | Initial implementation |
| 1.1.0 | 2026-09-20 | Order claims stated conditionally and measured; `MADD-ANO-014` registered. No change to the arithmetic. |
