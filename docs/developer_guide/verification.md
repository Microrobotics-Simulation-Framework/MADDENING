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
| `params_effective` | A trainable leaf of `params_pytree()` whose gradient (through `update` outputs *and* boundary fluxes) is zero on *every* sample — `update` still reads that constant from `self.params`, so the graph's injected value (and any calibration of it) is silently ignored.  Aggregated over the battery, so a parameter that only matters on some inputs passes as long as one sample exercised it; declare a leaf `ParamSpec(trainable=False)` if it is genuinely not a dynamics constant.  Each perturbed element is also compared, per path, against a node rebuilt from `to_dict()` with that value (when the node rebuilds faithfully): a path that uses the injected value differently from a constructed one -- wholly or partly through a copy made in `__init__` -- fails |

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
   pair, accepting `[declared - 0.25, declared + 1.0]`.  Both halves come
   from the ladders recorded in `maddening.testing.mms._ORDER_BAND_FIXTURES`.
   The lower half: the largest honest shortfall at the finest pair is
   **0.043** (the 4th-order `HeatNode` stencil's 3.957 against 4), while
   both defects the harness found fell a full order or more short (1.001
   against a declared 2, 0.954 against a declared 4).  The upper half
   catches a study that is not exercising the scheme at all — a
   manufactured solution the discretisation represents exactly measures
   nothing — and the largest honest pairwise order recorded, **4.126** (the
   cubic closure on `tanh_bump`, over its coarsest pair), sits inside it.
   `DEFAULT_ORDER_SHORTFALL` and `DEFAULT_ORDER_EXCESS` carry the full
   derivation, and
   `tests/verification/test_mms_order.py::TestTheOrderBandIsJustifiedByTheseNumbers`
   fails if a quoted figure drifts out of the band.

### What is covered

| Node | Axis | Declared | Observed | Benchmark |
|------|------|----------|----------|-----------|
| `HeatNode` (`stencil_order=2`, boundary data at the rod ends) | space | 2 | 2.000 | MADD-VER-005 |
| `HeatNode` | time | 1 | 1.000 | MADD-VER-006 |
| `HeatNode` (`stencil_order=4`) | space | 4 | 3.957 | — |
| `LBMNode` (D2Q9, periodic, Guo forcing [@Guo2002]) | space | 2 | 1.998 | MADD-VER-007 |
| `RigidBodyNode` (symplectic Euler [@Hairer2006]) | time | 1 | 0.999 | MADD-VER-008 |
| `SpringDamperNode` | time | 1 | 1.029 | MADD-VER-009 |
| `BallNode` (smooth regime, no collision) | time | 1 | 1.002 | MADD-VER-010 |
| `RigidBody2DNode` | time | 1 | 1.000 | MADD-VER-011 |
| `HeartPumpNode` | time | 1 | 1.000 | MADD-VER-012 |
| `WaveletAdaptiveNode` (full budget, periodic 1-D; Dirichlet and 2-D ladders in the same module) | space | 2 | 2.000 | MADD-VER-014 |

The two HeatNode spatial rows were strict xfails when this harness landed,
measuring 1.001 (MADD-ANO-007) and 0.954 (MADD-ANO-008).  Both node defects
are fixed in 0.4.0 and the xfails are now ordinary assertions; a strict xfail
that starts passing is a failure, so the two had to land together.

The built-in nodes that declare no order — `LBMPipeNode`, `TableNode`,
`HealthCheckNode` and the `AdaptiveNode` base class — skip.  `LBMNode` declares no *temporal*
order on purpose: the lattice fixes `dx = dt = 1` and `update` ignores its
`dt`, so there is no timestep to refine.

## Grid convergence: the fallback where MMS cannot reach

MMS needs somewhere to inject `S`.  That is a real restriction, not a
formality: `LBMPipeNode` declares no boundary inputs at all, a generic source
term is not even well defined for a lattice Boltzmann collision operator (the
forcing scheme changes the order of accuracy, so the study would measure the
scheme rather than the operator), and a node a *user* writes will usually have
no forcing input either.  Adding an optional manufactured-source hook to
`SimulationNode` was proposed and rejected, for four reasons:

1. **It is ill-defined for the node that prompted it.**  A source cannot be
   added to lattice Boltzmann distribution functions without a forcing scheme
   (Guo, Shan-Chen, exact-difference), and the choice changes the order of
   accuracy.  A generic hook would measure the forcing scheme's error rather
   than the collision operator's: a harness confidently reporting the wrong
   number, which is worse than one that declines to run.
2. **It is ill-defined for constrained systems.**  A node that projects onto a
   manifold cannot take arbitrary forcing without violating the constraint it
   exists to maintain.
3. **It would be production API surface that serves only tests**, and every
   shipped surface has to be documented and justified.
4. **It would not deliver its own headline benefit.**  The argument for it was
   that every user node becomes testable, but most user nodes would never
   implement an optional hook, so coverage of user code would not improve.

The one node that exposed the gap, `LBMPipeNode`, has a single scalar forcing
input on a fixed actuator-disc mask, which says little about the nodes users
write.

**The node-authoring contract, in one line: a node with a natural forcing
input can be MMS-tested; everything else falls back to GCI.**

`maddening.testing.mms` therefore carries a second mode over the same
refinement ladder.  A Richardson / Grid Convergence Index study compares three
successively refined solutions to **each other** — no exact field, no source
term, no reference run — and yields the observed order of convergence, the
extrapolated limit, and an error band on the finest solution
[@Richardson1911; @Roache1994; @Celik2008; @ASMEVV20].

```python
from maddening.testing.mms import RefinementAxis, assert_node_gci_verified

def flow_shape_at(n_cells):
    """Build the node at this resolution, run it, return ONE scalar."""
    ...
    return u_max / u_mean

assert_node_gci_verified(
    node, axis=RefinementAxis.SPACE, solution_at=flow_shape_at,
    levels=(16, 24, 32),        # coarsest first, at least three
    max_gci=0.25,               # widest acceptable band, as a fraction
)
```

The callback returns the **solution**, not the error, and it must be one
scalar functional — a peak value, a flux, a drag coefficient, a norm of the
state — computed identically at every level.

### What it reports, and what it refuses

Every quantity comes off the convergence ratio `R = eps_fine / eps_coarse`,
the change over the finest pair divided by the change over the pair before it
[@Stern2001].  Only `0 < R < 1` admits an extrapolation:

| `R` | Regime | Outcome |
|---|---|---|
| `0 < R < 1` | monotone convergence | order, limit and GCI reported |
| `-1 < R < 0` | oscillatory convergence | **FAIL**, named — the solutions straddle their limit, and no single power law describes the approach |
| `R >= 1` | monotone divergence | **FAIL**, named |
| `R <= -1` | oscillatory divergence | **FAIL**, named |
| a difference vanishes | stagnant | **FAIL** — *the node did not respond to refinement* |
| a solution is not finite | invalid | **FAIL** |

The stagnant case is the one to know about.  A node that ignores its
refinement parameter returns the same number three times; every formula in a
GCI divides by the difference between them, so the naive implementation
returns `nan` or, worse, a plausible order read off round-off.  Solutions
identical to within `DEFAULT_STAGNATION_RTOL` (`1e-12`, double-precision
round-off — raise it for a float32 study) are refused by name.  This is
checked **before** the node's declared order is looked at, so a node that both
declares nothing and ignores refinement cannot skip its way to a pass.

### Non-constant and non-integer refinement ratios

With a constant ratio the observed order is the textbook
`log(eps_32 / eps_21) / log(r)`.  With a ladder of 10, 17 and 40 cells it is
not, and using that formula anyway is the classic way to get a GCI wrong: on a
second-order rule it reads **0.98**, which looks like a perfectly plausible
first-order scheme and would be believed.

The harness never assumes a constant ratio.  It solves the implicit equation
of the ASME V&V 20 procedure [@Celik2008],

```
p = |ln|eps_32/eps_21| + q(p)| / ln(r_21),
q(p) = ln((r_21^p - s) / (r_32^p - s)),   s = sgn(eps_32/eps_21)
```

by Celik et al.'s fixed-point iteration, falling back to a bracketed bisection
on the signed residual when that diverges — which it does for ratio pairs far
apart, `r_21 = 1.4` against `r_32 = 2.7` overflowing within twenty steps.  A
pair for which the equation has no root in `[1e-3, 40]` is reported as an
inconclusive study, never clipped to the nearest endpoint.

### The safety factor

`Fs = 1.25` where the asymptotic range has been demonstrated, `Fs = 3.0`
everywhere else [@Roache1994; @Roache1998].  Roache ties the narrow band to an
order *measured* from three or more grids that agrees with the formal one;
where that agreement cannot be shown, this harness takes the conservative
factor rather than quoting the narrower band on trust.  A node that declares
no order therefore gets a wider, honest band instead of a confident,
unsupported one.

### The asymptotic range

A GCI is an error band only inside the asymptotic range, where the leading
truncation term dominates.  Outside it the number is not conservative — it is
*wrong in the confident direction*.  On the `LBMPipeNode` ladder below, the
16/24/32 triple quotes a band of 1.2% around a solution that is 3.6% from the
answer.  A study **positively shown** to be outside the range therefore fails
rather than reporting.

Roache's published check, `GCI_coarse / (r^p * GCI_fine) ~ 1`, is reported but
is **not** what the verdict is taken from.  For a three-grid study at a
constant ratio it collapses algebraically to `|phi_fine| / |phi_medium|`, so it
is within a per cent of 1 for any ladder whose solutions are close together,
converging or not — it reads 1.006 on the LBM pipe study that is emphatically
not asymptotic.  What carries information is:

1. **the observed order against the declared one** — the criterion ASME V&V 20
   uses, and the reason declaring `DiscretizationOrder` pays off twice; and
2. **agreement between independent triples**, when the ladder has four or more
   levels — which needs nothing declared.

With three levels and no declared order, neither is available and the harness
answers `None`, not `True`.  Adding a fourth level can therefore turn a pass
into a failure, and that is the intended direction: more evidence, more that
can be falsified.

### What it covers, and what it cannot

| Node | Axis | Levels | `R` | Observed order | GCI (`Fs`) | Benchmark |
|------|------|--------|-----|----------------|------------|-----------|
| `LBMPipeNode` (D3Q19, uniform axial force, `u_max/u_mean`) | space | 12/16/24 | 0.846 | 1.49 | 10.7% (3.0) | MADD-VER-013 |
| `LBMPipeNode`, same ladder plus 32 | space | 12/16/24/32 | 0.214 | 1.49 and 3.35 | — | not asymptotic; fails |

`LBMPipeNode` is the node the mode was built for, and the four-level result is
the honest finding: the pipe wall is a circle staircased onto a Cartesian
lattice with bounce-back [@Kruger2017], the effective wall position jumps about
as the resolution changes, and the two independent triples disagree by nearly
two orders of accuracy.  The study converges, but it is not in the asymptotic
range, and the harness says so instead of quoting a band it cannot support.

**GCI never sees the true answer, so it cannot catch a scheme that converges
cleanly to the wrong limit.**  MMS can.  Where a node has a natural forcing
input, MMS is the stronger test and GCI is the uncertainty statement on top of
it — not a substitute for it.

## Running the suite

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/verification/
```

The root `tests/conftest.py` disables the Hypothesis deadline for every
profile it registers (`dev`, the default, and `ci`; select one with
`MADDENING_HYPOTHESIS_PROFILE`), because the first example of every test pays
JIT compilation.  `tests/verification/hypothesis/conftest.py` configures
nothing any more; it is kept as a signpost to the root file.

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
- [ ] A node with a natural forcing input can be MMS-tested; everything else
      falls back to GCI — `assert_node_gci_verified(node, solution_at=...)`,
      which needs nothing from the node but the ability to refine
- [ ] `update(state, bi, dt=0)` is identity (zero-step) — add as an `invariants` entry
- [ ] If dissipative: `energy_fn=`
- [ ] If a conservation law applies: `invariants=` for the conserved quantity
- [ ] Document CFL / stability conditions in `meta.limitations`
- [ ] Document parameter constraints (e.g. `mass > 0`) with validation in `__init__`
