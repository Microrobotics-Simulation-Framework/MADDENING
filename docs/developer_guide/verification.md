# Property-Based Verification

MADDENING ships a property-based testing layer (built on
[Hypothesis](https://hypothesis.readthedocs.io/)) for physics nodes: sample
inputs from a declared envelope, check universal invariants, and shrink any
failure to a minimal counterexample.

Those checks compare the code to itself.  They cannot see a wrong
discretisation, because a stencil with the wrong weight is finite,
deterministic, JIT-consistent and differentiable.  [Order of
accuracy](#order-of-accuracy-the-method-of-manufactured-solutions) is the
other half of the battery: it compares the code to the mathematics.

Install via:

```bash
pip install maddening[verify]
```

## Verifying your own nodes

### One-liner battery

```python
from maddening.testing.verification import assert_node_verified

def test_my_node():
    assert_node_verified(
        my_node,
        bounds={"temperature": (200.0, 5000.0), "pressure": (1e3, 1e7)},
        dt_range=(1e-5, 0.001),
    )
```

This runs the default battery — every check a physics node should pass
regardless of what it models:

| Check | What it catches |
|-------|-----------------|
| `finite` | NaN/inf in any output field |
| `structure` | Dropped/added keys, shape or dtype drift |
| `deterministic` | Non-reproducible output (stray RNG, host side effects) |
| `jit_consistent` | Eager vs `jax.jit` disagreement (Python branching on array values) |
| `gradient_finite` | NaN in `d(outputs)/d(state)` — backward-pass-only failures such as `lstsq`/`solve` VJPs on rank-deficient inputs, `sqrt`/`norm` at zero, `jnp.where` guards that protect the forward but not the gradient |
| `params_consistent` | `update(..., params=node.params_pytree())` (and `compute_boundary_fluxes`, for a flux producer) disagreeing with the 3-argument form beyond float32 round-off; a producer that takes `params` in `update` but not in `compute_boundary_fluxes` fails outright — a constant read from `self.params` on one path and from the injected `params` on the other, so the graph (which always injects) calibrates a different model from the one tested in isolation |
| `params_gradient_finite` | NaN in `d(outputs)/d(params_pytree)` — the gradient an optimiser or `maddening.sysid` uses |
| `params_effective` | A trainable leaf of `params_pytree()` whose gradient (through `update` outputs *and* boundary fluxes) is zero on *every* sample — `update` still reads that constant from `self.params`, so the graph's injected value (and any calibration of it) is silently ignored.  Aggregated over the battery, so a parameter that only matters on some inputs passes as long as one sample exercised it; declare a leaf `ParamSpec(trainable=False)` if it is genuinely not a dynamics constant |

The `params_*` checks report `SKIP` (which counts as passed) for a
node whose `update` does not take a `params` keyword; see
`SimulationNode.accepts_params()`.  `tests/verification/test_builtin_nodes_verified.py`
(and `_lbm.py`) run the battery on every built-in node and are the
ledger of which nodes have migrated to `params`.

Failures list the shrunk counterexample. For programmatic access use
`verify_node`, which returns a `dict[str, VerificationResult]`:

```python
from maddening.testing.verification import verify_node

results = verify_node(my_node, bounds={...}, max_examples=500)
for name, r in results.items():
    print(name, r.status)          # PASS / FAIL / ERROR / SKIP
    if r.failed:
        print(r.counterexample)    # {"state": ..., "boundary_inputs": ..., "dt": ...}
```

### Opt-in physics checks

```python
verify_node(
    my_node,
    bounds={"T": (200.0, 5000.0)},
    output_bounds={"T": (0.0, 1e5)},            # boundedness
    energy_fn=lambda s: 0.5 * jnp.sum(s["v"]**2),  # energy_monotone
    invariants={                                # arbitrary predicates
        "mass_conserved": lambda s_in, s_out, bi, dt:
            jnp.allclose(jnp.sum(s_in["rho"]), jnp.sum(s_out["rho"]), rtol=1e-5),
    },
)
```

### Boundary inputs

Inputs declared in `boundary_input_spec()` are sampled automatically; bound
them with `boundary_bounds={...}`. If `update()` needs an input the node does
not declare, pass a fixed dict with `boundary_inputs={...}`.

### Writing your own `@given` tests

The strategies underneath the battery are public:

```python
from hypothesis import given, settings
from maddening.testing.strategies import node_states, bounded_dt, boundary_inputs_for

@given(
    state=node_states(my_node, bounds={"temperature": (200.0, 5000.0)}),
    bi=boundary_inputs_for(my_node, bounds={"inlet_T": (200.0, 400.0)}),
    dt=bounded_dt(1e-5, 0.01),
)
@settings(max_examples=500, deadline=None)
def test_my_node_cools(state, bi, dt):
    out = my_node.update(state, bi, dt)
    assert jnp.all(out["temperature"] <= state["temperature"].max())
```

| Strategy | Purpose |
|----------|---------|
| `node_states(node, bounds)` | State dicts matching `initial_state()` |
| `bounded_dt(min, max)` | Realistic timestep values |
| `boundary_inputs_for(node, bounds)` | Inputs matching `boundary_input_spec()` |

Sampling defaults to `float32` — the dtype most nodes execute in — so
overflow shows up where it actually happens.

## Order of accuracy: the Method of Manufactured Solutions

`maddening.testing.mms` measures the **observed order of convergence** of a
node's discretisation and fails when it falls short of what the node claims
[@Roache2002; @LeVeque2007].

Order, not error.  A wrong stencil weight, a mishandled boundary or an
off-by-one in a flux normally leaves the absolute error looking perfectly
acceptable on the one grid a threshold test runs on, and shows up only as
order 1 where order 2 was claimed.  Both defects this harness found in
MADDENING's own nodes (MADD-ANO-007 and MADD-ANO-008) sat inside the
thresholds of the tests that already covered them; both are fixed in
0.4.0.

### Declare the order

```python
from maddening.core.compliance.metadata import DiscretizationOrder, NodeMeta

class MyNode(SimulationNode):
    meta = NodeMeta(
        discretization="2nd-order central differences, forward Euler",
        discretization_order=DiscretizationOrder(
            spatial=2.0, temporal=1.0, notes="where the claim comes from",
        ),
        ...
    )
```

If the order depends on how the node was constructed — a selectable stencil,
say — override `discretization_order()` on the instance; the harness prefers
the hook over the class declaration.  A node that declares nothing is
**skipped explicitly**: `verify_node_order` returns `SKIP` and
`assert_node_order_verified` raises `UndeclaredOrderError`, so an undeclared
node can never look verified.

### Measure it

```python
from maddening.testing.mms import (
    ManufacturedSolution, RefinementAxis, assert_node_order_verified,
    diffusion_operator,
)

sol = ManufacturedSolution(
    exact=lambda x, t: jnp.sin(2 * jnp.pi * x) + 0.5 * x + 1.0,
    operator=diffusion_operator(alpha),          # du/dt = L[u] + S
)
# sol.source(x, t) = d(exact)/dt - L[exact](x, t), derived by AD rather
# than by hand: the derivation is the step MMS is most often got wrong on.

def error_at(n_cells):
    """Build the node at this resolution, drive it with sol, return one error."""
    ...

assert_node_order_verified(
    node, axis=RefinementAxis.SPACE, error_at=error_at,
    levels=(10, 20, 40, 80, 160),          # coarsest first
)
```

### Four things to get right

1. **Refine one axis at a time.**  Refining space and time together measures
   the *minimum* of the two orders, so a first-order integrator hides a
   second-order stencil.  Hold the other axis fixed, or make its error vanish
   identically: a manufactured solution with no time dependence, run to
   steady state, leaves only the spatial error (forward Euler's temporal
   truncation error is proportional to `d2u/dt2`); a solution quadratic in
   `x` is reproduced exactly by a second-order central difference and leaves
   only the temporal error.
2. **Check the node takes a source at all.**  MMS has to inject `S`.
   `HeatNode` has `heat_source`, `LBMNode` has `body_force`, `RigidBodyNode`
   has `force`/`torque`.  A node with no forcing input cannot be verified
   this way, and the honest finding is that MMS needs a hook the node does
   not expose — not a substitute study that does not test the discretisation.
3. **Stay above the arithmetic noise floor.**  The observed order is a ratio
   of small numbers.  In float32 these ladders measure a clean order to about
   80 cells and then turn over; the studies in
   `tests/verification/test_mms_order.py` run under `jax_enable_x64` for that
   reason.  `OrderMeasurement.monotone` is the guard: an error that stops
   falling fails as an inconclusive study, not as a wrong order.
4. **Read the band.**  `check_order` gates on the order over the *finest*
   pair, accepting `[declared - 0.25, declared + 1.0]`.  The lower half comes
   from measurement: across the three nodes covered, the finest pair lands
   within 0.02 of theory while the coarsest pair of the same ladder sits up
   to 0.16 low, and both defects found fall a full order or more short.  The
   upper half catches a study that is not exercising the scheme at all — a
   manufactured solution the discretisation represents exactly measures
   nothing.

### What is covered

| Node | Axis | Declared | Observed | Benchmark |
|------|------|----------|----------|-----------|
| `HeatNode` (`stencil_order=2`, boundary data at the rod ends) | space | 2 | 2.000 | MADD-VER-005 |
| `HeatNode` | time | 1 | 0.998 | MADD-VER-006 |
| `HeatNode` (`stencil_order=4`) | space | 4 | 3.957 | — |
| `LBMNode` (D2Q9, periodic, Guo forcing [@Guo2002]) | space | 2 | 1.998 | MADD-VER-007 |
| `RigidBodyNode` (symplectic Euler [@Hairer2006]) | time | 1 | 0.999 | MADD-VER-008 |

The two HeatNode spatial rows were strict xfails when this harness landed,
measuring 1.001 (MADD-ANO-007) and 0.954 (MADD-ANO-008).  Both node defects
are fixed in 0.4.0 and the xfails are now ordinary assertions; a strict xfail
that starts passing is a failure, so the two had to land together.

Every other node is undeclared and skips.  `LBMNode` declares no *temporal*
order on purpose: the lattice fixes `dx = dt = 1` and `update` ignores its
`dt`, so there is no timestep to refine.

## Running the suite

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/verification/
```

`tests/verification/hypothesis/conftest.py` disables the Hypothesis deadline
globally because the first example of every test pays JIT compilation.

## Limitations

1. **Sampling, not proof.** A PASS means no counterexample was found in
   `max_examples` draws. Raise `max_examples` for cheap nodes; use
   `derandomize=True` if CI must be bit-reproducible.
2. **Subnormals.** XLA flushes subnormals to zero on most backends, so
   properties that depend on them cannot be tested meaningfully.
3. **Cost scales with `update()`.** Nodes with large grids should use tight
   `bounds` and small shapes in the test fixture; the battery calls
   `update()` roughly `5 * max_examples` times plus one `jax.grad`.

## Checklist for new nodes

- [ ] `assert_node_verified(node, bounds=...)` with a physically meaningful envelope
- [ ] `NodeMeta(discretization_order=DiscretizationOrder(...))`, and an MMS
      study measuring it — an order nobody measured is a claim, not evidence
- [ ] `update(state, bi, dt=0)` is identity (zero-step) — add as an `invariants` entry
- [ ] If dissipative: `energy_fn=`
- [ ] If a conservation law applies: `invariants=` for the conserved quantity
- [ ] Document CFL / stability conditions in `meta.limitations`
- [ ] Document parameter constraints (e.g. `mass > 0`) with validation in `__init__`
