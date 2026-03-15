# Commit and Push Compliance Skill

This skill enforces MADDENING's documentation architecture requirements on every commit and push. Run it before pushing to ensure all regulatory and documentation standards are met.

## Trigger

When the user asks to commit and push, or invokes `/commit-and-push`.

## Pre-Commit Checklist

Work through the following checklist systematically. Each section has a **gate** — if the gate fails, stop and fix before proceeding.

### 1. Tests Pass

Run the full test suite:

```bash
source ../venvs/.maddening/bin/activate
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -v --tb=short --ignore=tests/viz
```

**Gate**: All tests must pass. If any fail, fix them before continuing.

### 2. Compliance CI Scripts Pass

Run all three compliance validation scripts:

```bash
python scripts/check_anomalies.py
python scripts/check_impl_mapping.py
python scripts/check_citations.py
python scripts/check_transforms.py
```

**Gate**: All four must exit 0. Fix any errors before continuing.

### 3. Commit Message Convention

Use the correct prefix based on the nature of the change:

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature or capability |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring (no behavior change) |
| `docs:` | Documentation-only changes |
| `test:` | Test additions or changes |
| `perf:` | Performance improvement |
| `verify:` | Verification/validation evidence |
| `break:` | Breaking API change |
| `deprecate:` | Deprecation notice |
| `security:` | Security-relevant change |

The commit message body should be concise (1-2 sentences) and focus on **why** the change was made, not what was changed (the diff shows what).

### 4. CHANGELOG.md Updated

Check if the changes affect user-visible functionality. If yes, update `CHANGELOG.md` under `## [Unreleased]` in the appropriate section:

- **Added** — new features, new nodes, new capabilities
- **Changed** — changes to existing features or behavior
- **Deprecated** — features marked for future removal
- **Removed** — features removed in this change
- **Fixed** — bug fixes
- **Verification** — changes to V&V status, new benchmarks, benchmark results changes
- **Security** — security-relevant changes (required by MDCG 2019-16)
- **Known Anomalies** — changes to `known_anomalies.yaml` (required for IEC 62304 SOUP)

**When to skip**: Pure internal refactors, CI config changes, and developer tooling changes that don't affect the package's external behavior don't need changelog entries.

### 5. New Node Checks (if applicable)

If the commit adds or modifies a `SimulationNode` subclass, verify:

- [ ] `meta` ClassVar has `NodeMeta` with: `algorithm_id`, `stability`, `description`, `assumptions`, `limitations`, `hazard_hints`
- [ ] `@stability(StabilityLevel.X)` decorator applied
- [ ] NumPy-style docstring present
- [ ] `update()` is JAX-traceable (uses `jnp` ops, no Python side effects)
- [ ] Algorithm guide in `docs/algorithm_guide/nodes/` follows `_template.md`
- [ ] Implementation Mapping table traces all equation terms to code
- [ ] Unit tests in `tests/nodes/`
- [ ] At least one `@verification_benchmark` in `tests/verification/`

### 6. New Anomaly Checks (if applicable)

If the commit introduces or discovers a known limitation:

- [ ] Entry added to `docs/validation/known_anomalies.yaml` with all required fields
- [ ] `anomaly_id` uses `MADD-ANO-XXX` format
- [ ] `safety_relevance_rationale` is filled in (not empty)
- [ ] Changelog updated under `### Known Anomalies`
- [ ] `python scripts/check_anomalies.py` passes

### 7. Bibliography/Citation Checks (if applicable)

If the commit modifies algorithm guide documents:

- [ ] All `[@Key]` citations have matching entries in `docs/bibliography.bib`
- [ ] YAML frontmatter includes `bibliography: ../../bibliography.bib`
- [ ] References section has human-readable inline descriptions alongside `[@Key]`
- [ ] `python scripts/check_citations.py` passes

### 8. Implementation Mapping Checks (if applicable)

If the commit renames or moves any function referenced in an algorithm guide's Implementation Mapping table:

- [ ] Algorithm guide updated with new qualified name
- [ ] `python scripts/check_impl_mapping.py` passes

### 8b. Transform Registry Checks (if applicable)

If the commit adds or modifies edge transforms used in production examples or scenarios:

- [ ] Transforms registered via `@register_transform` from `maddening.core.transforms`
- [ ] String transform references in production code resolve in the registry
- [ ] `python scripts/check_transforms.py` passes

### 9. API Stability Checks (if applicable)

If the commit changes a public API surface:

- [ ] `@stability` decorator level is appropriate
- [ ] If `STABLE`: change is backward-compatible (or this is a major version bump)
- [ ] If `PROVISIONAL`: deprecation warning added for the old API
- [ ] If removing a `DEPRECATED` API: verify it's a major version bump

## Execution Steps

After all checks pass:

1. **Stage files**: Add specific files (avoid `git add -A` to prevent accidentally staging secrets or large files)
2. **Commit**: Use the appropriate prefix and a concise message
3. **Push**: Push to the remote

```bash
git add <specific files>
git commit -m "prefix: concise description of why

Longer explanation if needed.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
git push
```

4. **Verify CI**: After pushing, note that CI will run tests on Python 3.10/3.11/3.12 and the compliance job. If CI fails, fix and push again.

## Quick Reference: What Goes Where

| Change Type | CHANGELOG | Anomaly YAML | Algorithm Guide | Bibliography |
|-------------|-----------|-------------|-----------------|--------------|
| New node | Added | If limitations | Yes (new doc) | If citing papers |
| Bug fix | Fixed | If it was a known anomaly | Update if affects equations | No |
| New limitation discovered | Known Anomalies | Yes (new entry) | Update Known Limitations | No |
| Renamed function | No | No | Update Impl Mapping | No |
| New verification benchmark | Verification | No | Update Verification Evidence | If citing reference |
| Documentation only | No (unless user-facing) | No | If algorithm guide | If adding citations |
| Security fix | Security | If safety-relevant | No | No |
| Deprecation | Deprecated | No | No | No |
