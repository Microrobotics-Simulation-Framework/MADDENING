# Contributing to MADDENING

## ID Prefix Convention

MADDENING uses the prefix `MADD` for all identifiers:

| Category | Format | Example |
|----------|--------|---------|
| Node algorithm IDs | `MADD-NODE-XXX` | `MADD-NODE-001` |
| Anomaly IDs | `MADD-ANO-XXX` | `MADD-ANO-001` |
| Verification benchmark IDs | `MADD-VER-XXX` | `MADD-VER-001` |

## Anomaly Lifecycle

MADDENING uses a three-phase anomaly lifecycle:

### Phase 1: Discovery (GitHub Issue)

When you discover a bug or anomaly:

1. Create a GitHub Issue using the **anomaly** template (`.github/ISSUE_TEMPLATE/anomaly.md`)
2. Fill in all mandatory fields: severity, safety relevance, rationale, affected components, affected versions, workaround
3. Apply the appropriate labels: `anomaly:critical`, `anomaly:major`, or `anomaly:minor`
4. If the anomaly could affect numerical correctness in a safety-relevant context, also apply the `safety-relevant` label

### Phase 2: Formalization (YAML Entry)

Once the anomaly is confirmed and understood:

1. Add an entry to `docs/validation/known_anomalies.yaml`
2. The entry must follow the schema: `anomaly_id`, `title`, `description`, `severity`, `safety_relevance`, `safety_relevance_rationale`
3. Run `python -m maddening.compliance check-anomalies docs/validation/known_anomalies.yaml` to validate
4. Cross-reference the GitHub Issue number in the YAML entry

### Phase 3: Verification

When the anomaly is resolved:

1. Update the YAML entry with resolution status and version
2. Add or update verification tests that demonstrate the fix
3. Never delete anomaly entries — mark them as resolved

### Release Gate (Three-Tier Model)

- **Tier 1** (`safety-relevant`): YAML entry required before release — no exceptions
- **Tier 2** (`anomaly:critical` or `anomaly:major`): YAML entry required before release — no grace period
- **Tier 3** (`anomaly:minor`): Two-cycle grace period; CI warns after one release cycle, blocks after two

## Code Style

- NumPy-style docstrings
- All node `update()` functions must be JAX-traceable (no Python-level side effects)
- All new nodes must have a `meta` ClassVar with `NodeMeta`

## Installation for Development

```bash
pip install -e ".[dev]"        # all features + pytest
# Or for GPU development:
pip install -e ".[dev,cuda12]"
```

See [docs/user_guide/installation.md](docs/user_guide/installation.md) for all available extras.

## Testing

Run the test suite:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v --tb=short
```

## Optional Dependency Guards

When writing code that uses an optional dependency, wrap the import with a try/except that tells the user which extra to install:

```python
# Module-level guard (for modules dedicated to one dep)
try:
    import pyvista as pv
except ImportError as _exc:
    raise ImportError(
        "HistoryViewer3D requires 'pyvista'. "
        "Install with:  pip install maddening[viz3d]"
    ) from _exc

# Lazy import guard (in __getattr__)
try:
    mod = importlib.import_module(module_path)
    return getattr(mod, name)
except ImportError as exc:
    raise ImportError(
        f"'{name}' requires additional dependencies. "
        f"Install with:  pip install maddening[extra_name]"
    ) from exc
```

The error message must always contain the exact `pip install maddening[...]` command.

## NodeMeta Requirements

Every `SimulationNode` subclass must have a `meta` ClassVar containing at minimum:
- `algorithm_id` (using `MADD-NODE-XXX` format)
- `stability` (StabilityLevel enum)
- `description`
- `assumptions`
- `limitations`
- `hazard_hints`
