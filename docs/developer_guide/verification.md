# Formal Verification and Property-Based Testing

MADDENING provides a two-layer verification system for proving numerical
correctness of simulation code:

1. **Stelling** (formal verification) — proves properties over ALL inputs
   in a declared bounded envelope. A VERIFIED result is a mathematical
   proof, not a sample.
2. **Hypothesis** (property-based testing) — samples from a rich input
   space with counterexample shrinking. Finds real bugs through random
   testing.

Install both via:

```bash
pip install maddening[verify]
```

## When to use which

| Goal | Tool | Strength |
|------|------|----------|
| Prove a bound holds universally | stelling | Sound — no false negatives |
| Find edge-case bugs | hypothesis | Broad coverage + minimal counterexamples |
| Verify division-by-zero guards | hypothesis | stelling can't reason through `jnp.where` branches with division |
| Prove non-negativity of a scalar | stelling | Interval arithmetic handles this directly |
| Test array-wide properties | hypothesis | stelling's `reduce_and` support is limited |
| Test at float32 precision | hypothesis | stelling operates in exact ℝ semantics |

## Verifying your own nodes

### Quick start (hypothesis)

```python
from hypothesis import given, settings
from maddening.testing.strategies import node_states, bounded_dt

@given(
    state=node_states(my_node, bounds={"temperature": (200.0, 5000.0)}),
    dt=bounded_dt(1e-5, 0.01),
)
@settings(max_examples=500, deadline=None)
def test_my_node_finite(state, dt):
    out = my_node.update(state, {}, dt)
    for field, val in out.items():
        assert jnp.all(jnp.isfinite(val))
```

The `node_states` strategy reads your node's `initial_state()` to determine
field names and shapes, then generates random arrays within the declared
bounds. `bounded_dt` generates realistic timestep values.

### Quick start (stelling)

```python
from maddening.testing.verification import verify_node

results = verify_node(
    my_node,
    bounds={"temperature": (200.0, 5000.0), "pressure": (1e3, 1e7)},
    dt_range=(1e-5, 0.001),
    solver_timeout_ms=10000,
)
for name, result in results.items():
    print(f"{name}: {result.status}")
    # status is VERIFIED, UNKNOWN, or REFUTED
```

`verify_node` runs a default set of checks (shape stability, overflow
freedom). For custom properties, use stelling's API directly.

### Writing custom stelling harnesses

A stelling harness transcribes your node's arithmetic into a function that
stelling can trace. Declare inputs with `any_array`, assert properties with
`assert_`:

```python
import jax.numpy as jnp
from stelling.harness import any_array, assert_
from stelling.preconditions import check

def harness():
    # Declare inputs over a bounded envelope
    T = any_array((10,), "float64", (200.0, 5000.0))
    dt = any_array((), "float64", (1e-5, 0.001))

    # Transcribe the node's arithmetic
    dT = -0.1 * T  # exponential cooling
    T_new = T + dt * dT

    # Assert the property
    return (assert_(jnp.all(T_new > 0.0)),)

v = check(harness, vacuity_mode="inputs-only", solver_timeout_ms=10000)
assert v.status == "VERIFIED"
```

**Key rules for stelling harnesses:**

1. Transcribe the arithmetic directly — don't import and call the node's
   `update()` method (stelling needs to trace the jaxpr).
2. Use `"float64"` for declarations (stelling operates in real arithmetic).
3. Keep declarations scalar or small arrays — stelling scales with equation
   count, not array size.
4. Use `solver_timeout_ms` when the property requires SMT escalation (e.g.,
   correlation-dependent properties like `a - a == 0`).

### Writing custom hypothesis strategies

For boundary inputs or complex state structures, extend the built-in
strategies:

```python
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np

# Strategy for a specific boundary input format
my_boundary_inputs = st.fixed_dictionaries({
    "pressure_inlet": arrays(
        dtype=np.float32, shape=(10,),
        elements=st.floats(min_value=1e3, max_value=1e7,
                           allow_nan=False, allow_infinity=False),
    ),
})
```

## Available strategies (`maddening.testing.strategies`)

| Strategy | Purpose |
|----------|---------|
| `node_states(node, bounds)` | Generate valid state dicts matching a node's interface |
| `bounded_dt(min, max)` | Generate realistic timestep values |
| `boundary_inputs_for(node, bounds)` | Generate inputs matching `boundary_input_spec()` |

## Available verification checks (`maddening.testing.verification`)

| Check | What it proves |
|-------|---------------|
| `node_shape_stability` | Output pytree has same structure as input |
| `node_no_overflow` | All output elements are finite |
| `node_boundedness` | Output stays within declared bounds |
| `node_energy_monotone` | Energy is non-increasing (dissipative systems) |
| `verify_node` | Runs all applicable checks |

## Running the verification suite

```bash
# Full verification suite (stelling + hypothesis)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/verification/

# With stelling overflow detection (integer narrowing tripwire)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/verification/ -p stelling.overflow

# Just stelling (formal proofs)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/verification/stelling/

# Just hypothesis (property-based)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/verification/hypothesis/
```

**Note:** The stelling overflow plugin (`-p stelling.overflow`) must NOT be
loaded globally — it interferes with the multigpu virtual device setup (the
plugin triggers JAX initialization before XLA_FLAGS are set). Load it
explicitly only when running verification tests.

## CI integration

The CI compliance job runs the verification suite automatically:

```yaml
# .github/workflows/ci.yml (compliance job)
python -m pytest tests/compliance/ tests/verification/ -v --tb=short -p stelling.overflow
```

## Known limitations of the verification system

See `tests/verification/README.md` for the full table of verified
properties, known limitations, and stelling feature requests.

Key limitations:

1. **select_n branch tracing** — stelling cannot prove properties that
   depend on `jnp.where` NOT taking a branch with division-by-zero.
   Use hypothesis for these.
2. **Array-wide assertions** — `jnp.all(pred)` is not supported by
   stelling. Use scalar declarations or per-element assertions.
3. **Float32 precision** — stelling operates in exact ℝ. Properties
   that depend on float32 representability (e.g., `clip(x, 0.01) >= 0.01`)
   must be tested via hypothesis.
4. **JIT deadline** — hypothesis tests that trigger JIT compilation on
   first run may exceed the 200ms default deadline. The verification
   suite disables deadlines globally via a conftest profile.

## Checklist for new nodes

When adding a new physics node, include these verification steps:

- [ ] Add hypothesis test: `update()` returns finite values for bounded inputs
- [ ] Add hypothesis test: `update()` preserves state structure (keys + shapes)
- [ ] Add hypothesis test: `update(state, {}, dt=0)` is identity (zero-step)
- [ ] Add hypothesis test: `update()` is deterministic (same in → same out)
- [ ] If dissipative: add hypothesis test for energy non-increase
- [ ] If conservation law: add hypothesis test for conserved quantity
- [ ] Document any CFL or stability conditions in `meta.limitations`
- [ ] Document any parameter constraints (e.g., mass > 0) with validation in `__init__`
- [ ] If arithmetic is simple enough: add stelling harness for key bounds
