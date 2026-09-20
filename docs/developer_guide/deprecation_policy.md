# Deprecation policy

What a `@stability` level promises, how a promise is withdrawn, and how the
withdrawal is recorded. This is the *mechanism*; which surfaces sit at which
level is decided in
[the 1.0 API freeze proposal](api_freeze_proposal.md) and recorded in
[the stability report](stability_report.md).

MADDENING follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
Read "major bump" below as "2.0 at the earliest", because 1.0 is where the
`STABLE` set is first frozen.

## The levels

Every public surface carries
`@stability(StabilityLevel.X)` from `maddening.core.compliance.stability`.
An untagged public surface is a decision nobody has made, not an implicit
level; see [Untagged surfaces](#untagged-surfaces).

| Level | Promise | Incompatible change allowed in | Notice required |
|---|---|---|---|
| `stable` | The signature and the documented behaviour hold for the life of the major version. | A major release. | Two minor releases of `DeprecationWarning`. |
| `evolving` | The signature and wire format are settled; additions are expected. | A minor release. | One minor release of `DeprecationWarning`. |
| `provisional` | Synonym for `evolving`, kept for surfaces tagged before v0.3.0. Do not use it on anything new. | As `evolving`. | As `evolving`. |
| `experimental` | Opt-in. May change or disappear in any minor release. | Any minor release. | None, but a CHANGELOG entry is mandatory. |
| `internal` | Not public. No promise of any kind. | Any release. | None. |
| `deprecated` | Scheduled for removal; the replacement is named in the docstring. | — | Already given. |

Two rules that are easy to miss:

- **Lowering a level is itself a breaking change.** Moving a surface from
  `stable` to `evolving` withdraws a promise callers have already relied on,
  so it needs the same notice as removing it. Raising a level
  (`experimental` → `evolving` → `stable`) is free and needs only a
  CHANGELOG entry.
- **A `stable` class promises the whole object**, not just its constructor:
  every public method and property an instance answers to, including the ones
  it inherits. `docs/developer_guide/stable_api.json` records them all.

## What counts as a breaking change

The guard (below) compares signatures mechanically and cannot tell intent, so
it flags every difference. Use this table to decide whether a flagged
difference needs a deprecation cycle or is a compatible change the snapshot
should simply be told about.

| Change | Breaking? |
|---|---|
| Add a keyword-only parameter with a default | No |
| Add a key to a returned `dict` | No |
| Widen a parameter type (`float` → `float \| jax.Array`) | No |
| Annotate a previously unannotated parameter with the type it already accepted | No |
| Rename a parameter | **Yes** — callers pass it by keyword |
| Reorder positional parameters | **Yes** |
| Remove a parameter, or make an optional one required | **Yes** |
| Turn a keyword-only parameter into positional-or-keyword | Not for callers, **yes** for subclasses that override; treat as breaking |
| Change a default value | **Yes** — it changes results silently, which is worse than a failure |
| Narrow a parameter type, or tighten validation so a previously accepted value now raises | **Yes** |
| Remove a key from a returned `dict`, or change a value's dtype/units | **Yes** |
| Add a new abstract method to a `stable` base class | **Yes** — every downstream subclass stops instantiating |
| Change which exception type is raised | **Yes** if it is documented in the `Raises` section |

A behaviour change that keeps the signature — a different numerical result, a
new warning, a changed convergence default — is breaking at the same level as
a signature change and gets the same notice. Bug fixes that make a documented
behaviour actually work are not breaking; say so in the CHANGELOG under
`### Fixed`.

## Deprecating a `stable` surface

1. **Ship the replacement first**, at `evolving` or above, in the same release
   the deprecation is announced. A deprecation with nowhere to go is a bug
   report, not a policy.
2. **Keep the old surface working** for the whole notice period. It forwards to
   the replacement; it does not get a second implementation.
3. **Warn at the call site.** `DeprecationWarning`, `stacklevel=2`, naming the
   replacement and the release that removes it:

   ```python
   warnings.warn(
       "maddening.core.simulation.checkpoint.download_and_load_state moved to "
       "maddening.cloud.resume; the alias is removed in 1.0",
       DeprecationWarning,
       stacklevel=2,
   )
   ```

   Use `FutureWarning` instead when the call keeps working but will start
   *returning something different* — a changed default, a changed convention —
   because `DeprecationWarning` is hidden from end users by default and a
   silent result change must not be. `PendingDeprecationWarning` is not used
   in this repository; two minor releases of notice is already the pending
   stage.
4. **Retag it** `@stability(StabilityLevel.DEPRECATED)` and put the
   replacement in the docstring's first line, so
   `scripts/generate_stability_report.py` shows it.
5. **Test the warning**, not only the forwarding:
   `pytest.warns(DeprecationWarning, match="...")`, plus one test that the
   alias still produces the same result as the replacement.
6. **Record it** in the CHANGELOG (below).
7. **Remove it** no earlier than two minor releases after the announcement,
   and only in a major release. Removal is a `### Removed` entry and an
   `--update` of the signature snapshot in the same commit.

`maddening.core.simulation.checkpoint.download_and_load_state` is the worked
example in the tree: announced in 0.4.0, forwards to
`maddening.cloud.resume.download_and_load_state`, removed in 1.0.

### `evolving` and `experimental`

`evolving` differs only in the length of the notice — one minor release
instead of two — and in where removal may land: a minor release, not a major
one. Everything else (replacement first, forwarding alias, warning, retag,
tests, CHANGELOG) is identical. An *additive* change to an `evolving` surface
needs no notice at all; that is what the level is for.

`experimental` needs no warning and no alias. It still needs the CHANGELOG
entry, because "opt-in" means someone opted in. Prefer a one-release
`DeprecationWarning` anyway when the change is cheap to soften — the level
sets the floor, not the ceiling.

`internal` surfaces change freely and are not mentioned in the CHANGELOG.

## Recording it in the CHANGELOG

`CHANGELOG.md` follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
A deprecation is announced under `### Deprecated` in `## [Unreleased]`, and
the eventual removal under `### Removed`. Add exactly one contiguous block at
the top of the subsection; create the subsection directly after the
`## [Unreleased]` heading if it is missing, and never reorder existing lines
(parallel branches merge into this file).

Each entry names three things — the surface, the replacement, and the release
that removes it:

```markdown
### Deprecated
- `CouplingGroup.solver="fori"` emits `DeprecationWarning`; removed in the next
  minor release.
```

If the deprecation lands together with the replacement, the replacement goes
under `### Added` as usual; do not fold it into the `### Deprecated` bullet.

## The CI guard

`scripts/check_stable_signatures.py` records the signature of every `stable`
surface in `docs/developer_guide/stable_api.json` and compares the tree
against it. It runs as the **Check STABLE API signatures** step of the
`compliance` job on every push and pull request — that job has no
`changes`-gate, so the guard runs even for a documentation-only change.

```console
$ python scripts/check_stable_signatures.py
OK: 13 STABLE surface(s), 223 member(s) unchanged
```

A difference fails with the recorded and current signatures side by side.
Two kinds:

- **CHANGED / REMOVED** — the promise is broken. Revert, or make the change
  additive, or land it behind a major version bump.
- **not in the snapshot** — a new `stable` surface, or a newly promoted one.
  Not a break; the snapshot is simply out of date.

Either way, accepting the current tree is one command:

```console
$ python scripts/check_stable_signatures.py --update
```

Run it in the **same commit** as the change, and say in the commit message why
the change is compatible or which major release carries it. `--update` is the
only supported way to move the snapshot; hand-editing the JSON is how a break
gets through unnoticed.

What the snapshot does **not** cover, and therefore what review still has to
catch by reading: behaviour, exceptions, the contents of `dict`-shaped
arguments and returns (until the PEP 589 phase gives them `TypedDict`s — see
[typing.md](typing.md)), instance attributes assigned in `__init__`
(`gm.params` is one), and module-level constants.

## Untagged surfaces

An export that carries no `@stability` tag has no level and therefore no
promise — but callers cannot see that, and in practice they treat anything in
an `__all__` as public. Anything newly exported from `__all__`, documented in
a user guide, or used by `src/maddening/examples/` gets a tag in the same
commit. `docs/developer_guide/api_freeze_proposal.md` lists the surfaces that
are public in practice and still untagged today.

`scripts/generate_stability_report.py` only sees a tag if it imports the
module that carries it; `tests/compliance/test_stability.py` fails when a
tagged module is missing from its `STABILITY_MODULES` list, so a new tag
cannot silently drop out of the report.

## Checklist

For a commit that changes a public surface — the long form of section 9 of
`.claude/skills/commit-and-push/SKILL.md`:

- [ ] The surface carries a `@stability` tag, and the level is still right.
- [ ] `python scripts/check_stable_signatures.py` passes, or `--update` ran in
      this commit with the reason in the commit message.
- [ ] If the change is breaking and the surface is `stable`: it is behind a
      major version bump, or it is a `### Deprecated` announcement with the
      replacement shipped, a `DeprecationWarning` at `stacklevel=2`, the
      `DEPRECATED` retag, and a test asserting the warning.
- [ ] If the change is breaking and the surface is `evolving`: one minor
      release of notice, same mechanics.
- [ ] CHANGELOG entry under the right subsection, naming the replacement and
      the removal release.
- [ ] `python scripts/generate_stability_report.py` re-run if any tag changed.
