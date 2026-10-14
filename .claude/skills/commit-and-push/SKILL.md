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

Run every compliance gate.  The gates are whatever `scripts/check_*.py`
holds -- **discover them with the glob, never copy a list**: the set grew
three times during 0.4.0, and a copied list (this section's own, which said
"six" and left out `check_doctests.py`) silently verified less than it
claimed.

```bash
for gate in scripts/check_*.py; do
  args=""
  [ "$gate" = scripts/check_anomalies.py ] && args="--prefix MADD-ANO-"  # as CI runs it
  python "$gate" $args || echo "GATE FAILED ($?): $gate"
done
python scripts/generate_soup_tables.py --check
```

What to expect from the gates whose behaviour is not simply "exit 0 or 1":

- `check_fragments.py` reads the history of HEAD for its ratchet.  On your
  own branch, an anomaly label you added and then dropped before merging
  shows as a dropped fragment: run it again with `--base origin/<the branch
  you merge into>`, which reads that line's history only, as CI does for a
  pull request.
- `check_doctests.py` executes every `>>>` example in `src/maddening/` under
  pytest (about two minutes).  It fails if an example fails or is skipped,
  or if fewer examples run than its committed floor (`MIN_EXAMPLES`).  Adding
  examples?  Raise the floor in the same commit.
- `check_transforms.py` exits **2** ("could not be trusted") when a module it
  must import needs an optional extra that is not installed -- in practice
  the `usd` extra (`usd-core`).  Install `.[ci,usd]`, as CI's compliance job
  does, or pass `--allow-missing-optional` to accept a partial check that
  names every reference it could not confirm.
- `check_stable_signatures.py` exits **2** if a module holding a `STABLE`
  surface cannot be imported; a missing extra is tolerated only while no
  `STABLE` surface lives in that module.  An intended signature change is
  accepted with `--update` in the same commit (section 9).
- `check_doctests.py` also exits **2** if pytest cannot run or no longer
  exposes the example count it reads.

**Gate**: every gate, and the SOUP `--check`, must exit 0.  Exit 2 is not a
pass: it means the gate could not see what it verifies.  Fix any errors
before continuing.

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

### 4. Change Recorded in a Fragment

A pull request never edits `CHANGELOG.md`, `docs/release_notes/`,
`docs/validation/known_anomalies.yaml`, `_HIGHEST_ANOMALY_ID`,
`docs/validation/soup_package.md` or the development record's anomaly index.
If the change affects user-visible functionality, record it in this branch's
changes fragment, `docs/fragments/changes/<branch>.yaml` (`<branch>`
lower-cased, `/ _ .` turned into `-`).  `python scripts/fragments.py new
--changes` writes one from a template:

```yaml
line: "0.4"
changelog:
  Fixed:
    - "**What changed.** What a user has to do about it."
release_notes:
  Fixed: |
    - The longer text for the release notes.
```

- **Changelog sections**: `Added`, `Changed`, `Deprecated`, `Removed`,
  `Fixed`, `Security` (required by MDCG 2019-16), `Verification`,
  `Known Anomalies` (required for IEC 62304 SOUP).
- **Release-notes headings**: `Summary`, `What is new`, `What changes for
  existing code`, `Security`, `Fixed`, `Known anomalies`, `API stability`,
  `Upgrade checklist`, `For contributors`.  Any other is refused.

**How to write the bullet**: say only what changed and what a user has to do
about it, in at most three lines.  Measurements, file inventories, rationale,
test counts and "why" paragraphs belong in the pull request description and
in the release-notes text.

**When to skip**: Pure internal refactors, CI config changes, and developer
tooling changes that don't affect the package's external behavior need no
changelog bullet.  Internal test infrastructure (fuzz harnesses, Hypothesis
profiles, property-test packages, CI gating) is contributor-facing: it goes
under the notes' `For contributors` heading, or one changelog line at most.

The release assembly (`scripts/fragments.py assemble --release`) writes the
changelog section and the notes before each release; `## [Unreleased]` stays
empty.  `scripts/check_fragments.py` (one of the gates in section 2) checks
the fragment.

### 5. New Node Checks (if applicable)

If the commit adds or modifies a `SimulationNode` subclass, verify:

- [ ] `meta` ClassVar has `NodeMeta` with: `algorithm_id`, `stability`, `description`, `assumptions`, `limitations`, `hazard_hints`
- [ ] `algorithm_id` is a new, unused ID written as a string literal, and the algorithm guide's `**Algorithm ID**` line states the same one (`check_impl_mapping.py` refuses a duplicate or a mismatch)
- [ ] `@stability(StabilityLevel.X)` decorator applied
- [ ] NumPy-style docstring present
- [ ] `update()` is JAX-traceable (uses `jnp` ops, no Python side effects)
- [ ] Algorithm guide in `docs/algorithm_guide/nodes/` follows `_template.md`
- [ ] Implementation Mapping table traces all equation terms to code
- [ ] Unit tests in `tests/nodes/`
- [ ] At least one `@verification_benchmark` in `tests/verification/`

### 6. New Anomaly Checks (if applicable)

If the commit introduces or discovers a known limitation:

- [ ] Not registered already: `python scripts/anom.py list --similar "a few words"`
- [ ] Entry added to this branch's anomalies fragment,
      `docs/fragments/anomalies/<branch>.yaml` (`python scripts/fragments.py
      new --anomaly`), with all required fields, never to the registry itself
- [ ] `anomaly_id` is the label `MADD-ANO-NEW-<branch>-<n>`: no number is
      picked by hand (the assembler numbers it and rewrites the label in
      every file); tests and other files name the entry by its label
- [ ] `safety_relevance_rationale` is filled in (not empty), and a reachable
      entry has a `release_note`
- [ ] `python scripts/check_anomalies.py --prefix MADD-ANO-` and
      `python scripts/check_fragments.py` pass
- [ ] Resolving a numbered entry: the registry edit, plus a changes-fragment
      text under the notes' `Known anomalies` heading

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

If the commit changes a public API surface, follow
`docs/developer_guide/deprecation_policy.md` (this is its short form):

- [ ] `@stability` decorator level is appropriate, and every newly public
      surface has one (an untagged export is a decision nobody has made)
- [ ] `python scripts/check_stable_signatures.py` passes.  If it flags a
      change, either the change is compatible and
      `python scripts/check_stable_signatures.py --update` runs in *this*
      commit with the reason in the message, or the change does not belong
      on a `STABLE` surface outside a major release
- [ ] If `STABLE`: change is backward-compatible (or this is a major version
      bump).  Lowering a surface's level is itself a breaking change
- [ ] If `EVOLVING`: an incompatible change ships the
      replacement first and gives one minor release of `DeprecationWarning`
      at `stacklevel=2`; `STABLE` gives two
- [ ] Deprecations retag to `DEPRECATED`, name the replacement and the removal
      release in the warning text, and have a `pytest.warns` test
- [ ] If removing a `DEPRECATED` API: verify it's a major version bump
- [ ] CHANGELOG `### Deprecated` / `### Removed` entry added
- [ ] `python scripts/generate_stability_report.py` re-run if any tag changed

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

The CHANGELOG column is the section of the changes fragment; the Anomaly
YAML column means the anomalies fragment.

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
