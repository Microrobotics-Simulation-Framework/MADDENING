# Node Authoring Guide

This guide covers everything required to add a new `SimulationNode` to MADDENING. Every requirement here is derived from the [Documentation Architecture](../../DOCUMENTATION_ARCHITECTURE.md) and enforced by CI.

## The Contract

A MADDENING node is a **pure function wrapped in a descriptor class**:

- `initial_state(params) -> dict` — returns the initial state arrays
- `update(state, boundary_inputs, dt) -> new_state` — returns a new state dict
- `update()` must be **JAX-traceable**: no Python-level side effects, no data-dependent control flow, no print statements. Use `jnp.where` instead of `if/else`.
- State is **immutable** — return a new dict, don't mutate in place
- Parameters live in `self.params`, not in state

## Directory Structure

```
src/maddening/nodes/your_node.py        # Node implementation
tests/nodes/test_your_node.py           # Unit tests
tests/verification/test_your_node_*.py  # Verification benchmark(s)
docs/algorithm_guide/nodes/your_node.md # Algorithm documentation
```

## Step-by-Step

### 1. Implement the Node

```python
"""YourNode -- one-line description."""

import jax.numpy as jnp

from maddening.core.node import SimulationNode
from maddening.core.metadata import (
    NodeMeta, StabilityLevel, ValidatedRegime, Reference,
)
from maddening.core.stability import stability


@stability(StabilityLevel.EXPERIMENTAL)
class YourNode(SimulationNode):
    """NumPy-style docstring.

    Parameters
    ----------
    name : str
        Unique node name.
    timestep : float
        Simulation timestep in seconds.
    ...

    Boundary inputs
    ---------------
    ...
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-XXX",      # Get next available ID
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="One-line description",
        governing_equations=r"...",          # LaTeX
        discretization="...",
        assumptions=(
            "...",
        ),
        limitations=(
            "...",
        ),
        validated_regimes=(
            ValidatedRegime("param", min_val, max_val, "units"),
        ),
        references=(
            Reference("AuthorYear", "Description"),
        ),
        hazard_hints=(
            "...",
        ),
    )

    def __init__(self, name, timestep, **kwargs):
        params = {"key": value, ...}
        state_spec = {"field": shape_tuple, ...}
        super().__init__(name, timestep, params, state_spec)

    def initial_state(self):
        return {"field": jnp.zeros(self.state_spec["field"])}

    def update(self, state, boundary_inputs, dt):
        # Pure JAX operations only
        ...
        return new_state
```

### 2. Required Metadata (`NodeMeta`)

Every node **must** have a `meta` ClassVar. Required fields:

| Field | Required | Description |
|-------|----------|-------------|
| `algorithm_id` | Yes | `MADD-NODE-XXX` format (get next available from existing nodes) |
| `algorithm_version` | Yes | Semantic version of this algorithm implementation |
| `stability` | Yes | `StabilityLevel.EXPERIMENTAL` for new nodes |
| `description` | Yes | One-line description |
| `assumptions` | Yes | Tuple of strings — every physical/mathematical assumption |
| `limitations` | Yes | Tuple of strings — every known failure mode |
| `hazard_hints` | Yes | Tuple of strings — qualitative risks for ISO 14971 input |

Recommended fields:

| Field | Description |
|-------|-------------|
| `governing_equations` | LaTeX string of the governing equations |
| `discretization` | Description of the numerical method |
| `validated_regimes` | Tuple of `ValidatedRegime` — quantitative parameter bounds |
| `references` | Tuple of `Reference` — BibTeX keys from `docs/bibliography.bib` |
| `implementation_map` | Dict mapping equation terms to Python qualified names |

**Scope distinction**: `validated_regimes` is for quantitative parameter-bound risks ("CFL must be < 0.5"). `hazard_hints` is for qualitative non-parameter-bound risks ("wall bounce-back assumes rigid walls"). A given risk goes in exactly one, never both.

### 3. Write the Algorithm Guide

Copy `docs/algorithm_guide/nodes/_template.md` and fill in every section:

```
docs/algorithm_guide/nodes/your_node.md
```

**Mandatory sections**: Summary, Governing Equations, Discretization, Implementation Mapping, Assumptions and Simplifications, Validated Physical Regimes, Known Limitations and Failure Modes, Stability Conditions, State Variables, Parameters, Boundary Inputs, References, Verification Evidence, Changelog.

**Implementation Mapping**: Trace every equation term to a specific Python/JAX function. No silent omissions. Terms handled by JAX primitives (e.g., `jnp.fft.rfftn()`) must be documented as such.

**References**: Use Pandoc-style `[@Key]` citations where `Key` matches an entry in `docs/bibliography.bib`. Include YAML frontmatter:

```yaml
---
bibliography: ../../bibliography.bib
---
```

Each reference also gets a human-readable inline description for GitHub/VS Code readability.

### 4. Add Bibliography Entries

Add BibTeX entries for any cited references to `docs/bibliography.bib`:

```bibtex
@article{AuthorYear,
  author  = {Last, First and Last2, First2},
  title   = {Title of the Paper},
  journal = {Journal Name},
  year    = {2024},
  volume  = {1},
  pages   = {1--10},
  doi     = {10.xxxx/yyyy},
}
```

CI validates that every `[@Key]` citation in algorithm guides resolves to an entry in the bib file (`scripts/check_citations.py`).

### 5. Write Tests

**Unit tests** (`tests/nodes/test_your_node.py`) — mandatory:
- Test each public method
- Test normal operation and edge cases
- Verify JAX-traceability: `jax.jit(node.update)(state, {}, dt)` must work
- Verify `jax.grad` compatibility if the node supports differentiation
- Verify `jax.vmap` compatibility if the node supports batching

**Verification benchmark** (`tests/verification/test_your_node_*.py`) — mandatory for physics nodes:
- Compare against analytical solution or published reference data
- Register with `@verification_benchmark`:

```python
from maddening.core.validation import verification_benchmark

@verification_benchmark(
    benchmark_id="MADD-VER-XXX",
    description="Your benchmark description",
    node_class="YourNode",
    reference="AuthorYear",
)
def test_your_analytical_comparison():
    ...
```

**Integration test** — mandatory:
- Test the node within a `GraphManager` (add node, connect edges, run steps)

### 6. Apply the `@stability` Decorator

```python
from maddening.core.stability import stability

@stability(StabilityLevel.EXPERIMENTAL)
class YourNode(SimulationNode):
    ...
```

New nodes start as `EXPERIMENTAL`. Promote to `STABLE` when:
- At least one verification benchmark passes
- Algorithm guide is complete
- API has been stable for at least one minor release

### 7. Document Known Limitations

If your node has known failure modes or limitations, add entries to `docs/validation/known_anomalies.yaml`:

```yaml
- anomaly_id: "MADD-ANO-XXX"
  title: "YourNode: brief description of limitation"
  description: "Full description..."
  severity: "major"            # critical | major | minor
  safety_relevance: "context_dependent"
  safety_relevance_rationale: "..."
  affected_components: ["YourNode"]
  affected_versions: ["0.2.0"]
  status: "open"
  workaround: "..."
```

Run `python -m maddening.compliance check-anomalies docs/validation/known_anomalies.yaml` to validate.

### 8. Run CI Checks

Before committing, verify everything passes:

```bash
# Activate the virtual environment
source ../venvs/.maddening/bin/activate

# Run the full test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v --tb=short --ignore=tests/viz

# Run compliance checks
python scripts/check_anomalies.py
python scripts/check_impl_mapping.py
python scripts/check_citations.py
```

## New Node Checklist

- [ ] `SimulationNode` subclass with `initial_state()` and `update()`
- [ ] `update()` is JAX-traceable (jit, grad, vmap compatible)
- [ ] `@stability(StabilityLevel.EXPERIMENTAL)` decorator applied
- [ ] `NodeMeta` metadata attached (algorithm ID, stability, assumptions, limitations, hazard_hints)
- [ ] NumPy-style docstring with Parameters, Boundary inputs
- [ ] Algorithm guide document in `docs/algorithm_guide/nodes/` following `_template.md`
- [ ] All template sections filled in (no empty tables or placeholder text)
- [ ] Implementation Mapping traces every equation term to code
- [ ] `[@Key]` citations reference entries in `docs/bibliography.bib`
- [ ] YAML frontmatter with `bibliography: ../../bibliography.bib`
- [ ] Assumptions and simplifications listed
- [ ] Validated physical regimes documented
- [ ] Known limitations and failure modes documented
- [ ] At least one registered `@verification_benchmark`
- [ ] Unit tests covering normal operation, edge cases, JAX-traceability
- [ ] Integration test within a `GraphManager`
- [ ] Known limitations entered in `docs/validation/known_anomalies.yaml`
- [ ] Entry in `docs/bibliography.bib` for primary reference
- [ ] All CI checks pass: `check_anomalies.py`, `check_impl_mapping.py`, `check_citations.py`
