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

A node is the first of three levels.  The transfer between two nodes and the
coupled graph have batteries of their own: see [Verifying a coupled model in
three levels](#verifying-a-coupled-model-in-three-levels).

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

<!-- snippet: no-run, reason: pseudo-code: my_node and bounds={...} are placeholders -->
```python
from maddening.testing.verification import verify_node

results = verify_node(my_node, bounds={...}, max_examples=500)
for name, r in results.items():
    print(name, r.status)          # PASS / FAIL / ERROR / SKIP
    if r.failed:
        print(r.counterexample)    # {"state": ..., "boundary_inputs": ..., "dt": ...}
```

### Opt-in physics checks

<!-- snippet: no-run, reason: fragment: my_node is the reader's node -->
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

### Fields with constraints

Every state field is sampled on its own, in a box.  A field that only
means something under a constraint — a unit quaternion, lattice populations
that are positive, a step counter the node uses as an index, two fields
that must agree — gets draws the node is never given.  The battery then
fails for a reason that is not a defect (the zero quaternion cannot be
normalised), or passes on states that say nothing about the node.

`constrain_state=` takes a function that is applied to every drawn state
before any check uses it; `constrain_boundary=` does the same for the drawn
boundary inputs.

```python
import jax.numpy as jnp
import numpy as np

from maddening.nodes.rigid_body import RigidBodyNode
from maddening.testing.verification import verify_node


def unit_quaternion(state):
    # In float64: the square of a tiny float32 draw underflows.  The zero
    # quaternion has no direction, so the identity stands in for it.
    q = np.asarray(state["orientation"], dtype=np.float64)
    norm = np.linalg.norm(q)
    q = q / norm if norm > 0 else np.array([1.0, 0.0, 0.0, 0.0])
    return {**state, "orientation": jnp.asarray(q, dtype=state["orientation"].dtype)}


body = RigidBodyNode("body", 0.01, mass=2.0, inertia=(1.0, 2.0, 3.0))
results = verify_node(
    body,
    bounds={"orientation": (-1.0, 1.0), "angular_velocity": (-5.0, 5.0)},
    boundary_bounds={"force": (-50.0, 50.0), "torque": (-50.0, 50.0)},
    constrain_state=unit_quaternion,
    checks=["finite", "gradient_finite"],
    max_examples=50,
    derandomize=True,
)
assert all(r.passed for r in results.values()), [str(r) for r in results.values()]
```

Every check sees the mapped state and only that — the eager and the
compiled call, the point a gradient is taken at, `energy_fn`, `invariants`
— and it is the state a counterexample reports.  `bounds` still set the box
the raw draw comes from.  The function must return the fields it was given,
each with the shape and dtype it was drawn with (`jnp.clip` keeps an `int32`
counter `int32`; cast back after computing in float64, as above).  Anything
else raises a `ValueError` that names the field: a refused function says
nothing about the node, so it is never reported as a `FAIL`.

For what a map cannot express — fields tied together through a shared draw,
a state taken from a short trajectory — pass your own Hypothesis strategy as
`state_strategy=` (or `boundary_strategy=`):

<!-- snippet: no-run, reason: fragment: my_node, n_cells and h are the reader's -->
```python
from hypothesis import strategies as st
from maddening.testing.verification import verify_node

@st.composite
def located(draw):
    cell = draw(st.integers(0, n_cells - 1))
    offset = draw(st.floats(0.0, 0.5, width=32))
    return {"x": jnp.asarray((cell + offset) * h, jnp.float32),
            "cell": jnp.asarray(cell, jnp.int32)}       # agrees with x

verify_node(my_node, state_strategy=located())
```

The strategy replaces the per-field sampling, so it is refused together with
`bounds` (and `boundary_strategy` with `boundary_bounds` or
`boundary_inputs`) rather than silently ignoring them.  Its examples reach
the node exactly as drawn (`dtype` is not applied to them), they shrink as
usual, and `constrain_state`, if also given, is applied to them.  In your
own `@given` test the same function is a `.map()`:
`node_states(my_node).map(unit_quaternion)`.

### Writing your own `@given` tests

The strategies underneath the battery are public:

<!-- snippet: no-run, reason: fragment: my_node is the reader's node -->
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

<!-- snippet: no-run, reason: pseudo-code: the dots stand for the other NodeMeta fields -->
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

<!-- snippet: no-run, reason: pseudo-code: error_at's body, alpha and node are the reader's -->
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

<!-- snippet: no-run, reason: pseudo-code: flow_shape_at's body and node are the reader's -->
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

## Verifying a coupled model in three levels

Verified nodes are not yet a verified model.  Each node was checked alone,
against its own equations; a coupled graph adds the transfer between two
discretisations and the scheme that advances the exchange in time, and
either can be wrong while every node is right.  The verification of a
coupled model therefore has three levels, and each establishes one thing:

| level | what is checked | with | what it establishes |
|---|---|---|---|
| 1. the node | `update()` against an analytical or manufactured solution | `verify_node`, `verify_node_order`, `verify_node_gci` (above) | each node solves *its* equations at its declared order |
| 2. the edge | the transfer between two nodes | `verify_mapping` | nothing is created or destroyed on the way, and a value arrives as the value it was |
| 3. the graph | the coupled model against a known solution, or against itself under refinement | `verify_graph_order`, `verify_graph_gci` | the *coupled* scheme converges, at the order the exchange and the edges allow |

**What none of them does.**  All three are *verification*: evidence that the
equations written down are solved correctly.  None says the equations are
the right ones.  *Validation* against experiments, the regime a model is
valid in, and the uncertainty of its parameters are separate work, and a
model that passes all three levels can still describe the wrong physics.
Manufactured solutions for a whole graph (a source term injected into each
node) are not in this release: level 3 here needs a coupled problem with a
known solution, or uses grid convergence, which needs none.

*`verify_mapping`, `verify_graph_order` and `verify_graph_gci` are
experimental in 0.4.0.*

### Level 2: the edge

`verify_mapping` is the battery for an interface mapping.  It takes any
object with the members of the `Mapping` protocol -- built by a shipped
factory, registered with `register_mapping`, or written by hand and never
registered -- or a whole `EdgeSpec`, in which case every check is made on what
the edge *delivers*: the mapping and then the edge's `transform`, through the
one function the step delivers an edge through.

```python
import numpy as np

from maddening.core.coupling.mapping import nearest_neighbor_mapping, rbf_mapping
from maddening.core.edge import EdgeSpec
from maddening.testing.mapping import assert_mapping_verified, verify_mapping

fluid = np.linspace(0.0, 1.0, 9)            # where the fluid's interface values are
solid = np.linspace(0.05, 0.95, 7)          # ... and the solid's

# A consistent transfer of a value, claimed to reproduce linear fields.
to_solid = rbf_mapping(fluid, solid, kernel="thin_plate_spline")
assert_mapping_verified(to_solid, polynomial_order=1,
                        source_coordinates=fluid, target_coordinates=solid)

# A conservative transfer of a force, on its edge: N to kN, and a sign.
edge = EdgeSpec("fluid", "solid", "traction", "force",
                mapping=nearest_neighbor_mapping(fluid, solid, mode="conservative"),
                transform=lambda force: -1e-3 * force)
results = verify_mapping(edge, scale=-1e-3)
assert all(r.passed for r in results.values())
assert results["conservative"].status == "PASS"      # the total, times the scale
assert results["consistent"].skipped                 # not claimed, so not checked

# Nearest neighbour reproduces constants and no more: a claim it does not keep fails.
too_much = verify_mapping(nearest_neighbor_mapping(fluid, solid), polynomial_order=1,
                          source_coordinates=fluid, target_coordinates=solid)
assert too_much["consistent"].failed
```

**What the mapping claims is your statement.**  The battery cannot know what
a kind of your own promises and does not guess.  `consistent=` and
`conservative=` default to the mapping's `mode`; `polynomial_order=` is the
degree the claim holds to (0: constants, the total); a property that is not
claimed is a `SKIP` that says "not claimed".  What the two words mean for a
transfer:

* **consistent** is the transfer of a *value* (a temperature, a
  displacement): a field that is uniform on the source arrives as the same
  uniform field, and, for degree `p`, every polynomial of the coordinates
  up to `p` arrives as itself.  The degree is the order of the transfer: a
  kind that reproduces degree `p` carries a smooth field with an error of
  order `h**(p + 1)`, which is what level 3 then measures.
* **conservative** is the transfer of an *amount* (a force, a heat flow):
  the total over the target is the total over the source, and for degree
  `p` so are the moments up to `p` (degree 1: where the amount sits).  With
  `source_measure=` / `target_measure=` the totals are weighted sums, for a
  field of densities on cells of different sizes.

**Conservation of work is a property of a pair.**  The `adjoint` check holds
the two applications of one mapping to `<apply(x), y> == <x, apply_T(y)>`.
If a displacement goes one way through a gather `H` and the force comes back
through the scatter `Hᵀ`, the work done on the two sides of the interface is
the same number: `fᵀ (H u) = (Hᵀ f)ᵀ u`.  Neither transfer has that property
alone.  A force returned through an independently built mapping, however
accurate, exchanges different amounts of work in the two directions, and the
coupled model gains or loses energy at the interface.

The checks, each a named result:

| result | what it holds the edge to |
|---|---|
| `structure` | the protocol's members; the `params_pytree()` contract `add_edge` enforces; the delivered shapes; no argument, weight or parameter changed by a call, nothing kept between calls |
| `linearity` | `delivered(a x + b y) == a delivered(x) + b delivered(y)` |
| `consistent`, `conservative` | as claimed, above |
| `adjoint` | `<delivered(x), y> == scale * <x, apply_T(y)>` |
| `geometry_derivative` | for a geometry-dependent kind: the derivative with respect to a position against a central difference |
| `outside_hull` | with `hull=` and `outside="clamp"`: a position outside the kernel's box is its projection onto it, and the derivative with respect to it is zero |
| `dtype_float32`, `dtype_float64` | the delivered dtype is the one the field and the weights promote to; float64 needs `jax_enable_x64` |
| `jit_consistent` | the compiled delivery equals the eager one |
| `round_trip` | a registered kind, written as a config writes it and rebuilt as a config reads it, has the same weights and delivers the same field, bit for bit |

A few things to know when reading a result:

* **An edge with a transform.**  A transform that multiplies by a known
  factor (a unit conversion, a sign) is declared with `scale=`.  One that is
  not that scaling -- a clamp, an offset, an undeclared factor -- fails
  `conservative` (and `linearity`, `adjoint`) with the transform named in the
  message: the mapping may conserve and the edge still not.
* **A geometry-dependent kind** needs positions: `geometry=` (one array of
  sample positions) or `geometry_strategy=` (a Hypothesis strategy that
  yields them, the vocabulary of `state_strategy=`).  Keep them where the
  kind's claims hold; a multilinear gather reproduces linear fields inside
  its hull.  The kernel of such a kind has *kinks* (a lattice plane, a face
  of the hull), where a finite difference is not a derivative.  The check
  lays a five-point stencil along one coordinate and compares only where
  the stencil's third differences vanish; a draw where they do not is *on a
  kink*, and the result's detail counts those draws beside the ones
  compared.  A check in which no draw could be compared is a `SKIP`.
* **Tolerances are in units of rounding**, `rounding_units * eps * gain *
  max|field|`, with `eps` the coarsest rounding among the dtypes involved
  and `gain` the largest absolute row sum of the operator.  A kind whose
  documented accuracy is not rounding needs a larger `rounding_units`, said
  where it is passed: the RBF factories solve a kernel system with a
  relative ridge of `1e-8`, so in float64 they reproduce and conserve to
  `1e-8`, which is some `5e7` float64 roundings.
* **A `SKIP` counts as passed**, as it does for a node.  Where a check must
  have been made, `assert_mapping_verified(..., require=("round_trip",))`
  raises on a skip of it.  An unregistered kind's `round_trip` is always a
  skip: it has no save/load route to check.

### Level 3: the coupled graph, and the iteration-error trap

`verify_graph_order` and `verify_graph_gci` are the graph-level forms of
`verify_node_order` and `verify_node_gci`, over the same machinery: you give
a function that builds the graph at a refinement level, runs it and returns
one number, and the levels.  Two things differ from a node.

**A graph declares no order, so you state the one you expect.**  Without
`expected=` the result is a `SKIP`.  What a partitioned coupling can reach:

* **first order in time**, whatever the nodes' own integrators are.  Each
  node receives its boundary inputs once per step and holds them across it
  (MADD-ANO-014), so the exchange is first order even where a coupling
  group iterates the step to convergence;
* **order `p + 1` in space** through an edge whose mapping reproduces
  polynomials of degree `p`, and no more than the nodes' own order.

**The iteration error is not the discretisation error.**  A coupling group
solves each step's exchange iteratively, to a tolerance.  What is left of
that iteration is an error of the *solver*; a refinement ladder reads it as
an error of the *scheme*.  Measured on the worked example below, refining
the timestep on a fixed grid (the coupled scheme is first order):

| group `tolerance` | pairwise orders (50, 100, 200, 400 steps) | last step's bound / error, finest level |
|---|---|---|
| `1e-8` | 1.005, 1.002, 1.001 | `3.2e-8` |
| `1e-4` | 1.211, 1.029, **0.645** | `1.9e-2` |

At `1e-4` the group takes two passes on almost every step of the two coarse
levels and stops after one on most steps of the two fine ones (168 of 200,
383 of 400), so the ladder compares two different schemes and the order it
reports belongs to neither.  Note the last column: the bound the
diagnostics report for the *last step* is 1.9% of the error and looks
harmless.  A fixed-point iteration stopped early from the previous step's
state errs the same way at every step, and the errors add: it is the bound
**times the number of steps** that has to be small.

Both functions therefore carry a guard and do not report an order as
verified without it.  At every level the iteration error must be at most
`iteration_factor` (default 0.05) of the discretisation error being
measured -- an error known to within 5% at two levels moves the order read
from them by at most 0.14, inside the band -- by one of two routes:

* **from the diagnostics**: `iteration_bound_at=lambda level:
  coupling_iteration_bound(graph_at(level), steps=...)`.  It reads
  `coupling_report()` and returns `steps` times the largest
  `spectral_error_bound`, usable where every group reports
  `spectral_usable=True` (`solver="ift"`, `diagnostics=True`; see *Reading
  `coupling_diagnostics()`* in the coupling algorithm guide).  A group can
  report a number and not call it usable; the result then has
  `usable=False` and the report's flag as its `reason`, and the number is
  not used;
* **by running the level again** at a tighter coupling tolerance:
  `tightened_error_at=` (or `tightened_solution_at=`).  The iteration error
  is then the difference of the two measurements, in the study's own units.
  This route assumes nothing about the graph, and is the one to use where a
  group reports no usable bound: `solver="fori"`, diagnostics off, a
  geometry-dependent mapping the diagnostics do not read, a wide interface,
  a custom edge.

Where the guard finds the iteration error too large the result is a `FAIL`
that says the study is inconclusive (tighten the tolerance), not that the
model is wrong.  Where neither route can be taken the result is a `SKIP`
that says the iteration error was not checked.  A graph with no coupling
group iterates nothing: say `iterated=False`, and the lag of its staggered
exchange is part of the time discretisation the study measures.

### A worked example

Two rods on the same interval, insulated at their ends, each heated along
its whole length at a rate proportional to the other's temperature:

$$
\partial_t T_a = \alpha\, \partial_x^2 T_a + k\, T_b, \qquad
\partial_t T_b = \alpha\, \partial_x^2 T_b + k\, T_a.
$$

From $T_a = 1 + \cos \pi x$, $T_b = 0$ the solution is that of one rod
alone, $u = 1 + \cos(\pi x)\, e^{-\alpha \pi^2 t}$, shared between the two:
$T_a = \cosh(kt)\, u$ and $T_b = \sinh(kt)\, u$.  Each rod is a `HeatNode`
(second order in space, forward Euler in time), and the coupling is two
edges, each rod's temperature into the other's `heat_source`.  On matching
grids the edges are plain; on non-matching grids they carry a mapping, and
levels 2 and 3 meet.

```python
import math

import jax

jax.config.update("jax_enable_x64", True)  # a ladder is a ratio of small numbers

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import nearest_neighbor_mapping, projection_1d_mapping
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode
from maddening.testing.coupled import (
    coupling_iteration_bound, verify_graph_gci, verify_graph_order,
)
from maddening.testing.mapping import assert_mapping_verified
from maddening.testing.mms import RefinementAxis

ALPHA, K, T_END = 0.1, 1.0, 0.5


def centres(n):
    return (np.arange(n) + 0.5) / n


def faces(n):
    return np.linspace(0.0, 1.0, n + 1)


def projection(n_from, n_to):
    return projection_1d_mapping(faces(n_from), faces(n_to))


def nearest(n_from, n_to):
    return nearest_neighbor_mapping(centres(n_from), centres(n_to))


def run(n_a, n_b, steps, mapping=None, **group):
    """The two rods after `steps` steps to T_END.  A fresh graph every time:
    a run leaves a graph at its final state."""
    dt = T_END / steps
    gm = GraphManager()
    gm.add_node(HeatNode("a", dt, n_cells=n_a, thermal_diffusivity=ALPHA))
    gm.add_node(HeatNode("b", dt, n_cells=n_b, thermal_diffusivity=ALPHA))
    for source, target, n_from, n_to in (("a", "b", n_a, n_b), ("b", "a", n_b, n_a)):
        gm.add_edge(source, target, "temperature", "heat_source",
                    transform=lambda temperature: K * temperature,
                    mapping=None if mapping is None else mapping(n_from, n_to))
    if group:
        gm.add_coupling_group(["a", "b"], max_iterations=100, **group)
    gm.compile()
    # A HeatNode's state is float32 until it is set.
    gm.set_node_state("a", {"temperature": jnp.asarray(1.0 + np.cos(np.pi * centres(n_a)))})
    gm.set_node_state("b", {"temperature": jnp.zeros(n_b, jnp.float64)})
    gm.run_scan(steps)
    return gm


def exact(x):
    alone = 1.0 + np.cos(np.pi * x) * math.exp(-ALPHA * np.pi ** 2 * T_END)
    return math.cosh(K * T_END) * alone, math.sinh(K * T_END) * alone


def error(n, ratio=1.0, mapping=None, **group):
    """Relative L2 error on grids of n and ratio * n cells.  The timestep is
    refined as dx**2: the scheme is first order in time, so its time error
    falls at order 2 in dx and cannot limit a ladder that expects 2 or 1."""
    n_b = round(ratio * n)
    gm = run(n, n_b, math.ceil(T_END * ALPHA * max(n, n_b) ** 2 / 0.2), mapping, **group)
    t_a, t_b = (np.asarray(gm.get_node_state(rod)["temperature"]) for rod in "ab")
    e_a, e_b = exact(centres(n))[0], exact(centres(n_b))[1]
    return math.sqrt((np.mean((t_a - e_a) ** 2) + np.mean((t_b - e_b) ** 2))
                     / (np.mean(e_a ** 2) + np.mean(e_b ** 2)))


SPACE, TIME = RefinementAxis.SPACE, RefinementAxis.TIME

# Level 2, on the edge the non-matching model will use: a cell average
# reproduces constants and preserves the integral (cell sizes as the measure).
assert_mapping_verified(projection(8, 12), consistent=True, conservative=True,
                        source_measure=np.diff(faces(8)), target_measure=np.diff(faces(12)))

# Level 3, matching grids on plain edges, in a group iterated to convergence.
# The guard takes the re-run route: the same error at a tighter tolerance.
matching = verify_graph_order(
    axis=SPACE, levels=(8, 16, 32), expected=2.0,
    error_at=lambda n: error(n, tolerance=1e-10),
    tightened_error_at=lambda n: error(n, tolerance=1e-12))
assert matching.status == "PASS", matching.detail

# Non-matching grids (three cells to two) through the cell-average projection:
# second order still.  No group here, so nothing iterates.
projected = verify_graph_order(
    axis=SPACE, levels=(8, 16, 32), expected=2.0, iterated=False,
    error_at=lambda n: error(n, ratio=1.5, mapping=projection))
assert projected.status == "PASS", projected.detail

# Through nearest neighbour, which reproduces degree 0: first order, and a
# claim of second is caught.
by_nearest = dict(axis=SPACE, levels=(8, 16, 32), iterated=False,
                  error_at=lambda n: error(n, ratio=1.5, mapping=nearest))
assert verify_graph_order(expected=1.0, **by_nearest).status == "PASS"
assert verify_graph_order(expected=2.0, **by_nearest).failed

# The order in time, on one grid, from three timesteps compared with each
# other (no exact solution needed), guarded from the diagnostics.  The rods
# are swept together here (iteration_mode="jacobi"): for that sweep this
# group reports a bound it calls usable.
runs = {}


def at(steps):
    if steps not in runs:
        runs[steps] = run(16, 16, steps, tolerance=1e-8, solver="ift", diagnostics=True,
                          iteration_mode="jacobi")
    return runs[steps]


in_time = verify_graph_gci(
    axis=TIME, levels=(50, 100, 200), expected=1.0,
    solution_at=lambda steps: float(
        jnp.sqrt(jnp.mean(at(steps).get_node_state("b")["temperature"] ** 2))),
    iteration_bound_at=lambda steps: coupling_iteration_bound(at(steps), steps=steps))
assert in_time.status == "PASS", in_time.detail
```

What the example measures, on ladders of 8, 16, 32 and 64 cells (the same
figures on jax 0.10.2, 0.11.0 and 0.11.2).  The mapped rows are without a
coupling group; in a group iterated to convergence the thin-plate spline at
both ratios, and the projection and nearest neighbour at 2:1, were measured
too and stay inside the band:

| edges | order observed | why |
|---|---|---|
| plain, matching grids, in a group | 2.02, 2.01, 2.00 | the rods' own second order |
| `rbf_mapping` (thin-plate spline), 3:2 | 2.02, 2.01, 2.00 | reproduces linear fields: order 2 |
| `rbf_mapping` (thin-plate spline), 2:1 | 2.05, 2.03, 2.02 | the same |
| `projection_1d_mapping`, 3:2 | 2.01, 2.00, 2.00 | see below |
| `projection_1d_mapping`, 2:1 | 2.02, 2.01, 2.00 | see below |
| `nearest_neighbor_mapping`, 3:2 | 1.00, 0.99, 0.99 | reproduces constants only: order 1 |
| `nearest_neighbor_mapping`, 2:1 | 0.69, 0.86, 0.94 | order 1, reached slowly |
| time, matching grids, in a group (50 to 400 steps) | 1.00, 1.00, 1.00 | inputs supplied once per step |

Two of those rows say something the edge check alone does not.

* **The cell-average projection keeps second order although, pointwise, it
  reproduces only constants** (`verify_mapping(..., polynomial_order=1)`
  fails for it at the cell centres).  Its error has zero mean over every
  source cell -- that is what preserving the integral means -- so it is a
  grid-scale oscillation that diffusion damps, and the coupled solution
  converges at the rods' order.  The degree an edge reproduces is the order
  it *guarantees*; the coupled study measures the order the model *has*.
* **Nearest neighbour at 2:1 is biased.**  A coarse cell centre is exactly
  between two fine ones, the tie goes to the lower index (as documented),
  and every coarse cell reads a value a quarter of a cell to its left.  The
  model is first order, and the ladder reaches that order slowly.

One more thing the example shows is where the diagnostics route stops.  The
time study sweeps the two rods together; swept one after the other, which
is the default, the same group reports a `spectral_error_bound` and
`spectral_usable=False` -- its pass map is far from normal, which is where
the diagnostics decline to certify the spectral radius they measured -- and
`coupling_iteration_bound` returns `usable=False` with that flag.  The
guard then takes the re-run route if it was given one, and is a `SKIP` if
it was not.

Stability, for this construction: each rod keeps its own explicit limit
(`dt * alpha / dx**2` below 1/2 on the finer rod; the example runs at 0.2)
and `k * dt` is small.  Two rods coupled *end to end* through their
Dirichlet inputs are a different construction with a lower limit of their
own, 3/8 at `stencil_order=2` (MADD-ANO-050).

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
- [ ] A state field with a constraint (a unit quaternion, positive populations,
      an index): `constrain_state=`, so the battery judges states the node is given
- [ ] `NodeMeta(discretization_order=DiscretizationOrder(...))`, and an MMS
      study measuring it — an order nobody measured is a claim, not evidence
- [ ] A node with a natural forcing input can be MMS-tested; everything else
      falls back to GCI — `assert_node_gci_verified(node, solution_at=...)`,
      which needs nothing from the node but the ability to refine
- [ ] `update(state, bi, dt=0)` is identity (zero-step) — add as an `invariants` entry
- [ ] If dissipative: `energy_fn=`
- [ ] If a conservation law applies: `invariants=` for the conserved quantity
- [ ] A mapping kind of your own on an edge: `assert_mapping_verified(mapping, ...)`
      with the claims it makes; a coupled model: `verify_graph_order(...)` with the
      iteration-error guard ([three levels](#verifying-a-coupled-model-in-three-levels))
- [ ] Document CFL / stability conditions in `meta.limitations`
- [ ] Document parameter constraints (e.g. `mass > 0`) with validation in `__init__`
