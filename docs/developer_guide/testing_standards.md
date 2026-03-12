# Testing Standards

Test requirements at three levels, all enforced by CI.

## 1. Unit Tests (mandatory for all code)

Location: `tests/<module>/test_<name>.py`

Requirements:
- Test each public method
- Test normal operation and edge cases
- For nodes: verify JAX-traceability (`jax.jit`, `jax.grad`, `jax.vmap`)
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

Verification benchmarks must:
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

## Test Organization

```
tests/
├── core/           # GraphManager, scheduling, coupling, adaptive, checkpoint
├── nodes/          # Per-node unit tests
├── surrogates/     # Surrogate framework tests
├── api/            # Server, WebSocket, binary encoder tests
├── viz/            # Visualization tests (skipped in CI — require display)
├── compliance/     # Metadata, anomaly registry, stability tests
└── verification/   # Analytical benchmarks (registered with @verification_benchmark)
```

## pytest Configuration

Test warnings are escalated to errors by default (`filterwarnings = ["error"]` in `pyproject.toml`). Known-safe warnings are explicitly ignored. If your code produces a new warning, either fix the cause or add a documented filter to `pyproject.toml` with a comment explaining why.
