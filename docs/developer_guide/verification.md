# Property-Based Verification

MADDENING ships a property-based testing layer (built on
[Hypothesis](https://hypothesis.readthedocs.io/)) for physics nodes: sample
inputs from a declared envelope, check universal invariants, and shrink any
failure to a minimal counterexample.

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
| `params_consistent` | `update(..., params=node.params_pytree())` disagreeing with `update(...)` beyond float32 round-off — a constant read from `self.params` on one path and from the injected `params` on the other, so the graph (which always injects) calibrates a different model from the one tested in isolation |
| `params_gradient_finite` | NaN in `d(outputs)/d(params_pytree)` — the gradient an optimiser or `maddening.sysid` uses |
| `params_effective` | A trainable leaf of `params_pytree()` whose gradient is zero on *every* sample — `update` still reads that constant from `self.params`, so the graph's injected value (and any calibration of it) is silently ignored.  Aggregated over the battery, so a parameter that only matters on some inputs passes as long as one sample exercised it; declare a leaf `ParamSpec(trainable=False)` if it is genuinely not a dynamics constant |

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
- [ ] `update(state, bi, dt=0)` is identity (zero-step) — add as an `invariants` entry
- [ ] If dissipative: `energy_fn=`
- [ ] If a conservation law applies: `invariants=` for the conserved quantity
- [ ] Document CFL / stability conditions in `meta.limitations`
- [ ] Document parameter constraints (e.g. `mass > 0`) with validation in `__init__`
