# Verification Suite

Property-based tests (Hypothesis) for MADDENING's core algorithms and nodes,
plus the pre-existing analytical and gradient-health checks.

## Structure

```
verification/
├── hypothesis/
│   ├── test_hypothesis_aitken.py        # Division guard, omega finiteness
│   ├── test_hypothesis_adaptive.py      # dt/factor bounds, monotonicity
│   ├── test_hypothesis_coupling.py      # Norm symmetry, triangle inequality
│   ├── test_hypothesis_integrators.py   # Zero-step, constant-deriv, order
│   ├── test_hypothesis_graph.py         # Random topology robustness
│   ├── test_hypothesis_multirate.py     # Multi-rate consistency
│   ├── test_hypothesis_determinism.py   # Repeatability, JIT, vmap
│   ├── nodes/                           # Per-node properties (ball, spring, heat)
│   └── conftest.py                      # Disable hypothesis deadline (JIT warmup)
├── test_gradient_health.py              # Gradient checks
├── test_heat_analytical.py              # Analytical benchmark
├── test_mms_order.py                    # Observed order of convergence (MMS)
└── test_wavelet_mms_order.py            # WaveletAdaptiveNode: declared order by MMS, adaptive budget
```

## Properties covered

| Property | Notes |
|----------|-------|
| Aitken omega in [0.01, 2.0], finite, division guard | float32, wide envelopes |
| Adaptive dt_next in [dt_min, dt_max], factor bounds, acceptance iff error <= 1 | |
| Adaptive acceptance monotone | Lower error -> also accepted |
| L2 / mixed norm: non-negative, symmetric, triangle inequality, finite with atol=0 | |
| Integrator zero-step identity, constant-deriv exact, finite output | Euler, Heun, RK4 |
| Node update finite / structure-preserving / deterministic / JIT-consistent | All built-in nodes |
| Observed order of convergence matches the declared one | HeatNode (space + time), LBMNode (space), RigidBodyNode (time) |
| Graph topology robustness | Random topologies |
| Multi-rate sync consistency | Rate ratios |

## Bugs found by this suite (and its former stelling companion)

| Bug | Location | Trigger | Fix |
|-----|----------|---------|-----|
| Aitken NaN on overflow | `acceleration.py` | float32 `delta_r^2` overflows to inf | `jnp.isfinite(denom)` guard |
| Mixed norm div-by-zero | `acceleration.py` (3 sites) | `scale = 0` with `atol=0` | `jnp.where(scale > 0, diff/jnp.maximum(scale, 1e-300), 0.0)` |
| SpringDamperNode div-by-zero | `spring.py` | `mass=0` | `mass > 0` validation |
| IQN-ILS condition squaring | `acceleration.py` | `V.T @ V` squares cond# | `jnp.linalg.lstsq` |

| HeatNode `stencil_order=4` converges at order 1, not 4 | `heat.py` | MMS spatial refinement | fixed in 0.4.0, MADD-ANO-008 — cubic ghost extrapolation through the rod end; 0.954 -> 3.957 |
| HeatNode Dirichlet data lands at the cell centre, not the documented rod end | `heat.py` | MMS with the documented boundary data | fixed in 0.4.0, MADD-ANO-007 — mirror ghost, end cells no longer overwritten; 1.001 -> 2.000 |
| HeatNode 4th-order stencil diverges inside the documented CFL limit | `heat.py` | Fo = 0.40 with `stencil_order=4` | fixed in 0.4.0, MADD-ANO-009 — per-stencil bound documented and refused at construction |
| MADD-VER-002's acceptance band had been widened to accept the order-1 result it was measuring | `test_heat_analytical.py` | Re-reading the benchmark against what it asserts | fixed in 0.4.0 — band is now [1.7, 2.3] |

## Physical limitations (not code bugs)

| Finding | Notes |
|---------|-------|
| CFL violation -> negative T | dt > dx^2/(2*alpha) allows negative temperature |
| LBM BGK negativity at low tau | f_post < 0 at tau=0.501 (Mach limit) |

## Extending for your own nodes

```python
from maddening.testing.verification import assert_node_verified

def test_my_node():
    assert_node_verified(
        my_node, bounds={"temperature": (200.0, 5000.0), "pressure": (1e3, 1e7)},
    )
```

For the order of accuracy, declare it on the node and measure it:

```python
from maddening.testing.mms import RefinementAxis, assert_node_order_verified

def test_my_node_is_second_order_in_space():
    assert_node_order_verified(
        my_node, axis=RefinementAxis.SPACE,
        error_at=error_at, levels=(10, 20, 40, 80),
    )
```

See `docs/developer_guide/verification.md` for the full API.
