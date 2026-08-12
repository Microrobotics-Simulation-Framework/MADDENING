# Verification Suite

This directory contains formal verification (stelling) and property-based
testing (hypothesis) suites for MADDENING's core algorithms and nodes.

## Structure

```
verification/
├── stelling/           # Formal verification (proves properties for ALL inputs)
│   ├── test_stelling_aitken.py     # Aitken relaxation omega bounds
│   ├── test_stelling_adaptive.py   # Adaptive dt bounds, acceptance logic
│   ├── test_stelling_norms.py      # Convergence norm safety
│   └── nodes/                      # Per-node formal properties
├── hypothesis/         # Property-based testing (samples + shrinks)
│   ├── test_hypothesis_aitken.py   # Division guard, omega finiteness
│   ├── test_hypothesis_adaptive.py # dt/factor bounds, monotonicity
│   ├── test_hypothesis_coupling.py # Norm symmetry, triangle inequality
│   └── nodes/                      # Per-node properties
├── test_gradient_health.py         # Pre-existing gradient checks
└── test_heat_analytical.py         # Pre-existing analytical verification
```

## Two verification layers

### stelling (formal, sound)

Proves properties over ALL inputs in a declared bounded envelope using
interval arithmetic + SMT solver escalation (z3, cvc5). A VERIFIED
verdict means the property holds for every possible input — not just
sampled ones.

### hypothesis (sampling, finds bugs)

Samples from a richer input space using property-based testing with
counterexample shrinking. Catches bugs that stelling can't reach (e.g.,
properties involving control flow through `jnp.where`).

## Verified properties summary

| Property | Tool | Status | Notes |
|----------|------|--------|-------|
| Aitken omega in [0.01, 2.0] | stelling | VERIFIED | All input envelopes |
| Aitken omega bounded (with overflow guard) | stelling | VERIFIED | Includes isfinite check |
| Aitken denom > 0 (non-degenerate) | stelling | VERIFIED | Residuals bounded away from 0 |
| Aitken division guard (denom ≤ 1e-30) | hypothesis | PASS | stelling: UNKNOWN (select_n limitation) |
| Aitken overflow robustness | hypothesis | PASS | Large residuals → finite omega (bug fix verified) |
| Adaptive dt_next in [dt_min, dt_max] | stelling | VERIFIED | |
| Adaptive factor in [min_factor, max_factor] | stelling | VERIFIED | |
| Adaptive acceptance iff error ≤ 1 | stelling | VERIFIED | |
| Adaptive acceptance monotone | hypothesis | PASS | Lower error → also accepted |
| L2 norm ≥ 0 | stelling | VERIFIED | |
| L2 norm = 0 when identical | stelling | VERIFIED | Requires solver (correlation) |
| Mixed norm scale > 0 (atol > 0) | stelling | VERIFIED | |
| Mixed norm ≥ 0 | stelling | VERIFIED | |
| L2 symmetry, triangle inequality | hypothesis | PASS | |
| Mixed norm symmetry | hypothesis | PASS | |
| Integrator zero-step identity | hypothesis | PASS | Euler, Heun, RK4 (excl. subnormals) |
| Integrator constant-deriv exact | hypothesis | PASS | All produce x + dt*c |
| Integrator finite output | hypothesis | PASS | Bounded inputs → finite output |
| Integrator order (Euler=1, RK4=4) | hypothesis | SKIP | Requires float64; verified when x64 on |
| SpringDamperNode mass > 0 validated | pytest | PASS | ValueError on mass ≤ 0 |
| SpringDamperNode energy dissipation | hypothesis | PASS | E_after ≤ E_before with damping |
| HeatNode conservation (Neumann BC) | hypothesis | PASS | ≤5% leakage from ghost-cell BCs |
| HeatNode CFL stability (documented) | hypothesis | PASS | Confirms negativity above CFL |
| HeatNode finite (CFL-safe dt) | hypothesis | PASS | |
| Node update finite for bounded inputs | hypothesis | PASS | BallNode, SpringDamperNode |
| Node update preserves structure | hypothesis | PASS | All built-in nodes |

## Known limitations

1. **select_n both-branch tracing**: When a property depends on a
   conditional (`jnp.where`) not taking the division path, stelling
   produces UNKNOWN because JAX's jaxpr always traces both branches.
   The division-guard fallback in Aitken is the primary example.
   Hypothesis covers this gap.

2. **Array-wide assertions**: `jnp.all(pred)` over arrays is not
   well-supported by stelling's interval propagation. Use scalar
   declarations or per-element assertions instead.

3. **Correlation (a - a = 0)**: Interval arithmetic alone cannot
   prove that subtracting a variable from itself yields zero (it
   tracks independent intervals). The SMT solver handles this via
   QF_LRA, so properties requiring correlation need
   `solver_timeout_ms`.

4. **Outward-rounded boundaries**: At exact arithmetic boundaries
   (e.g., `atol + 0 >= atol`), 1-ulp outward rounding causes
   straddles. This is a precision artifact, not a real violation.

5. **XLA subnormal flush-to-zero**: JAX/XLA flushes subnormal
   float32 values (|x| < 1.18e-38) to zero. Properties involving
   subnormals cannot be verified in either tool; hypothesis tests
   exclude this range via `assume()`.

## Bugs found and fixed

| Bug | Location | Trigger | Fix |
|-----|----------|---------|-----|
| Aitken NaN on overflow | `acceleration.py:278` | float32 `delta_r^2` overflows to inf → inf/inf = nan | Added `jnp.isfinite(denom)` guard; falls back to input omega |
| SpringDamperNode div-by-zero | `spring.py:134` | `mass=0` → `force / 0` = inf | Added `mass > 0` validation in `__init__` |

## Known limitations in MADDENING (not fixable without algorithm changes)

| Issue | Location | Impact | Mitigation |
|-------|----------|--------|-----------|
| HeatNode CFL instability | `heat.py:443` | Negative temperatures above `dt > dx²/(2α)` | Use adaptive timestepping |
| IQN-ILS normal equations | `acceleration.py:378` | `V^T V` squares condition number; 1e-10 reg inadequate for float32 | Use QR factorization (future work) |
| Implicit Newton no line search | `implicit.py:101` | Can diverge for highly nonlinear residuals | Fixed iteration count; caller should check convergence |
| LBM high-Mach instability | `lbm.py:200-205` | Negative f_eq at Ma > ~0.3 | Inherent to BGK; document Mach limit |
| BallNode tunneling | `ball.py:96/101` | Collision missed at large v*dt | Use smaller dt or add subcycling |

## Stelling feature requests (for next release)

1. **isfinite transfer + select_n branch pruning**: The `isfinite`
   transfer (~30 lines) unlocks select_n pruning for free. Once
   landed, MADDENING's Aitken division-guard XFAIL becomes VERIFIED,
   and `node_no_overflow` can use `assert_(jnp.isfinite(x))` directly
   instead of the bounded-proxy. Expected in next stelling release.

2. **Float32 precision modeling**: Stelling operates in exact real (ℝ)
   semantics. A float32 mode would allow proving properties like
   "clip(x, 0.01, 2.0) >= 0.01" at the actual representable precision.
   May come in next release.

**Note on reduce_and:** NOT needed. stelling's `assert_` is already
elementwise on arrays — `assert_(x > 0)` checks each element without
needing `jnp.all()`. The bare array form is strictly better (fewer
equations, same semantics).

## Extending for your own nodes

Install `maddening[verify]` and use the testing harness:

```python
from maddening.testing.verification import verify_node
from maddening.testing.strategies import node_states, bounded_dt

# Formal verification
results = verify_node(
    my_node,
    bounds={"temperature": (200.0, 5000.0), "pressure": (1e3, 1e7)},
)
for name, result in results.items():
    print(f"{name}: {result.status}")

# Property-based testing
from hypothesis import given, settings

@given(state=node_states(my_node, bounds={...}), dt=bounded_dt())
@settings(max_examples=1000)
def test_my_node_finite(state, dt):
    out = my_node.update(state, {}, dt)
    for v in out.values():
        assert jnp.all(jnp.isfinite(v))
```
