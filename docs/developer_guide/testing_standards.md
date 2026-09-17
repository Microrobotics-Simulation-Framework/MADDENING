# Testing Standards

Test requirements at three levels, all enforced by CI.

## 1. Unit Tests (mandatory for all code)

Location: `tests/<module>/test_<name>.py`

Requirements:
- Test each public method
- Test normal operation and edge cases
- For nodes: verify {term}`JAX-traceability <JAX-traceable>` (`jax.jit`, `jax.grad`, `jax.vmap`)
- Test boundary input handling (missing inputs, default values)
- Test parameter edge cases

```python
def test_your_node_basic():
    node = YourNode("test", timestep=0.01)
    state = node.initial_state()
    new_state = node.update(state, {}, 0.01)
    assert "field" in new_state

def test_your_node_jit():
    node = YourNode("test", timestep=0.01)
    state = node.initial_state()
    jitted = jax.jit(node.update)
    new_state = jitted(state, {}, 0.01)
    # Must not raise
```

## 2. Integration Tests (mandatory for nodes)

Test the node within a `GraphManager`:

```python
def test_your_node_in_graph():
    gm = GraphManager()
    node = YourNode("test", timestep=0.01)
    gm.add_node(node)
    state = gm.run(n_steps=10)
    assert "test" in state
```

Test edge connections if the node consumes or produces coupled data.

## 3. Verification Benchmarks (mandatory for physics nodes)

Location: `tests/verification/test_<name>_<benchmark>.py`

Compare against analytical solutions or published reference data. Register with the `@verification_benchmark` decorator:

```python
from maddening.core.validation import verification_benchmark

@verification_benchmark(
    benchmark_id="MADD-VER-XXX",
    description="Analytical solution comparison for ...",
    node_class="YourNode",
    reference="AuthorYear",
)
def test_your_node_analytical():
    """Compare YourNode output to analytical solution."""
    node = YourNode("bench", timestep=0.001, ...)
    gm = GraphManager()
    gm.add_node(node)
    state = gm.run(n_steps=1000)

    # Compute analytical solution
    analytical = ...

    # Compare
    error = jnp.abs(state["bench"]["field"] - analytical)
    assert jnp.max(error) < tolerance, f"Max error {jnp.max(error)} exceeds {tolerance}"
```

{term}`Verification benchmarks <Verification benchmark>` must:
- Use a registered benchmark ID (`MADD-VER-XXX`)
- Cite the analytical solution or reference data source
- State the tolerance and justify it
- Be referenced in the node's algorithm guide (Verification Evidence section)

## Running Tests

```bash
# Activate venv
source ../venvs/.maddening/bin/activate

# Full test suite (exclude viz — requires display)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v --tb=short --ignore=tests/viz

# Just compliance + verification
python -m pytest tests/compliance/ tests/verification/ -v --tb=short

# Just your new node's tests
python -m pytest tests/nodes/test_your_node.py tests/verification/test_your_node_*.py -v

# Compliance CI scripts
python scripts/check_anomalies.py
python scripts/check_impl_mapping.py
python scripts/check_citations.py
```

## Environment Variables

| Variable | Purpose | Typical Value |
|----------|---------|---------------|
| `JAX_PLATFORMS` | Force CPU backend (avoids GPU issues) | `cpu` |
| `XLA_FLAGS` | Disable GPU autotune (avoids equinox segfaults) | `--xla_gpu_autotune_level=0` |
| `PYTEST_DISABLE_PLUGIN_AUTOLOAD` | Prevent plugin conflicts | `1` |
| `MADDENING_HYPOTHESIS_PROFILE` | Hypothesis depth/settings profile (`dev` or `ci`) | `dev` |

## 4. Property-Based Testing (recommended for physics nodes)

Location: `tests/verification/hypothesis/` for the numerics suite; property
tests that belong with a package (`tests/fmi/`, `tests/core/`) stay there.
Configuration is global either way — see *Hypothesis profiles* below.

Beyond analytical benchmarks, MADDENING provides a Hypothesis-based
property-testing layer. See the full [Verification Guide](verification.md).

**Property-based testing** (hypothesis) — test universal invariants:

```python
from maddening.testing.strategies import node_states, bounded_dt
from hypothesis import given

@given(state=node_states(my_node, bounds={...}), dt=bounded_dt())
def test_my_node_finite(state, dt):
    out = my_node.update(state, {}, dt)
    for val in out.values():
        assert jnp.all(jnp.isfinite(val))
```

Note the absence of `@settings`: `deadline`, `print_blob`, the example
database and `max_examples` all come from the active profile.

### Hypothesis profiles

Profiles are registered and loaded in the **root** `tests/conftest.py`, so
every property test in the tree gets them, whether or not the Hypothesis
pytest plugin is loaded (the suite runs with
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`, which disables the plugin and its
`--hypothesis-profile` flag). Do not register profiles in a sub-directory
`conftest.py`, and do not repeat `deadline=None` per test.

| Profile | `max_examples` | Used by | Notes |
|---------|----------------|---------|-------|
| `dev`   | 50  | default for every local run | fast enough to keep the full suite in its ~50-minute budget |
| `ci`    | 200 | the `verify-hypothesis` GitHub Actions job | 4x the search for every test that lets the profile own its depth |

A per-test `@settings(max_examples=...)` **overrides the profile**, so a test
that sets its own cap runs at that cap under both profiles and `ci` buys it
nothing. That is the whole reason for the house rule below.

Both set `deadline=None` (a JAX compile blows any wall-clock deadline),
`print_blob=True` (a failure prints a `@reproduce_failure` blob you can
paste into the test) and `suppress_health_check=[HealthCheck.too_slow]`.

Select a profile with the `MADDENING_HYPOTHESIS_PROFILE` environment
variable; an unknown name is a hard error rather than a silent fallback:

```bash
# local default -- same as setting nothing
MADDENING_HYPOTHESIS_PROFILE=dev pytest tests/verification/hypothesis/

# reproduce what CI runs
MADDENING_HYPOTHESIS_PROFILE=ci pytest tests/verification/hypothesis/
```

Every run prints the active profile in the pytest header, e.g.
`hypothesis profile: ci (max_examples=200, database=.../.hypothesis/examples)`.

### The example database

Failing examples are written to `<repo>/.hypothesis/examples` (git-ignored,
absolute path fixed in `tests/conftest.py` so it does not depend on the
working directory). Hypothesis replays them first on the next run, so a
falsifying example found once stays found. The `verify-hypothesis` CI job
caches that directory with `actions/cache`, keyed per run with a
`restore-keys` prefix, so the database survives across runs; a cache miss
is not an error, the job just starts from an empty database.

To clear a stale entry locally, delete `.hypothesis/`.

### `max_examples`: the house rule

**A property test carries no `max_examples`. The profile owns it.** That is
what makes "run the suite deeper" a one-line change instead of a hundred.

An explicit `max_examples` is an exception that must be justified *in a
comment on the line above it*, and is only justified when a single example
is genuinely expensive — a fresh JAX trace/compile per draw, an optimiser
loop, a multi-device `shard_map`, a full rollout. Then:

* the floor is **20**. Below that Hypothesis barely gets past reuse and
  generation and the test is decoration: it can pass for months and fail
  once, which is exactly the failure mode this configuration exists to
  remove;
* prefer the value already used by the sibling properties in the same file
  over inventing a new one;
* an explicit value *above* the profile default is fine and needs no
  special pleading — it only ever deepens the test.

`verify_node` / `assert_node_verified` are a separate knob: they build
their own `settings` internally (with `database=None`), so the profile does
not reach them. Their `max_examples` argument defaults to 200; a call site
that lowers it is subject to the same floor and the same
comment-your-reason rule.

**Node battery** — finite outputs, structure, determinism, jit/eager
agreement, finite gradients, in one call:

```python
from maddening.testing.verification import assert_node_verified

def test_my_node():
    assert_node_verified(my_node, bounds={"field": (lo, hi)})
```

Install with `pip install maddening[verify]`.

## Test Organization

```
tests/
├── core/           # GraphManager, scheduling, coupling, adaptive, checkpoint
├── nodes/          # Per-node unit tests
├── surrogates/     # Surrogate framework tests
├── api/            # Server, WebSocket, binary encoder tests
├── viz/            # Visualization tests (skipped in CI — require display)
├── compliance/     # Metadata, anomaly registry, stability tests
└── verification/   # Verification suite
    ├── hypothesis/     # Property-based tests (hypothesis)
    │   └── nodes/      # Per-node property tests
    ├── test_gradient_health.py    # Pre-existing gradient checks
    └── test_heat_analytical.py    # Pre-existing analytical benchmark
```

## pytest Configuration

Test warnings are escalated to errors by default (`filterwarnings = ["error"]` in `pyproject.toml`). Known-safe warnings are explicitly ignored. If your code produces a new warning, either fix the cause or add a documented filter to `pyproject.toml` with a comment explaining why.
