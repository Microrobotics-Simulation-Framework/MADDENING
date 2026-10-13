# Audit: documentation and compliance apparatus (`c51cd6a`, release/0.4.0)

> Transcribed by the orchestrator: a harness guard refused the auditor's own
> `Write` to this path. The reproducers and raw data in this directory were
> written by the auditor itself:
> `repro_capture_ratio_fraction.py`, `repro_paramspec_override_asymmetry.py`,
> `compare_sweep.py`, `compare_sweep_output.txt`,
> `coupling_sweep_cpu_HEAD_c51cd6a.json` (420-row re-run at HEAD, ~50 min).

Worktree clean at exit. All runs `JAX_PLATFORMS=cpu`, `PYTHONPATH=<wt>/src`.
**PR 59 had not landed when this finished** (still OPEN/MERGEABLE); its diff was
read and every finding below is checked against it.

## Summary

The prose is in good shape — 45 of 47 test paths cited in docs/README/CHANGELOG resolve (the 2 that don't are `test_your_node.py` template placeholders), every documented public signature matches the code, every anomaly's `affected_components` and `verification` entry resolves, the quickstart/README snippets run clean, and several measurements reproduce to three digits. **The defects are concentrated in the generated and machine-checked parts, and in numbers quoted from measurements that this release invalidated.**

---

## Findings

### HIGH — `docs/developer_guide/stability_report.md` publishes 42 of 85 registered surfaces, and omits both DEPRECATED entries

**What breaks:** the published API-stability contract is missing half the API, including everything deprecated. A user reading it sees no deprecations at all in a release that deprecates four things.

**Evidence:** committed file says `*42 API surfaces registered.*` (line 64), 13 `stable` rows. Regenerated from the live registry (without writing over the tracked file):
```
SKIPPED: {}
registered NOW: 85 {'EVOLVING': 55, 'STABLE': 17, 'EXPERIMENTAL': 11, 'DEPRECATED': 2}
committed rows: 42 | committed claims: 42
missing from committed report: 43
DEPRECATED in registry: ['...calibration.tune_coupling_params', '...calibration.calibrate']
DEPRECATED rows in committed report: []
```
Missing: all 9 `maddening.sysid.*`, all 4 `mapping_spec`, 6 `mapping`, 4 `nodes.adaptive`, 5 `fmi.*`, 3 `compile_cache`, 2 `profiler`, `cloud.resume.download_and_load_state`, `solver_utils.ift_linear_solve`, the four new STABLE methods, and both deprecations.

**Why it happens:** `git log -- docs/developer_guide/stability_report.md` → last touched by `6bddeb4`, a v0.3.0-era commit. `tests/compliance/test_stability.py` asserts only that `STABILITY_MODULES` covers every module using `@stability(` (`:173`) and that the generator refuses an incomplete report (`:218`). **Nothing compares the committed markdown to a fresh generation**, so CI cannot catch staleness.

**Knock-on:** three documents now give three different STABLE counts — `docs/developer_guide/typing.md:11` says "19 STABLE-tagged surfaces in 12 modules", the committed report has 13 in 11, the registry has 17 in 14. typing.md's phase-2 scope is defined by that number.

**Non-issue check:** ruled out a missing optional dependency (`SKIPPED: {}` — a complete run) and a broken detector (hand-verified `node.py:483`, `node.py:591`).

**Fix:** regenerate, and add a compliance test that a fresh generation equals the committed file. Risk: the file then changes on every `@stability` edit, which is the point.

---

### HIGH — `scripts/check_transforms.py` verifies zero references; the gate cannot fail on anything real

**Evidence:**
```
$ python scripts/check_transforms.py
OK: 0 string transform reference(s) verified (6 transforms in registry)
exit=0
```
It AST-walks only `src/maddening/{examples,core,nodes}` for **string-literal** `transform=` kwargs. Everything in `src/` uses lambdas or variables; the only string-literal edge transforms in the repo are in `tests/`, which the code explicitly excludes — while its own docstring says it validates "examples and **tests**". Mutation results (all reverted, worktree verified clean): an unregistered transform in `tests/` → PASS; in `src/maddening/usd/` → PASS (88 of 198 src files unscanned, including the USD serializer that `PLAN_accuracy_and_usd.md:376` names as the reason the gate exists); a name bound to a variable → PASS; deleting `@register_transform("extract_first")` while three tests reference it by name → PASS (registry 6→5, exit 0). A deliberately misspelled literal in `src/maddening/examples/` **is** caught, so the script is not a no-op — it just has nothing in scope.

`DESIGN.md:737` ("CI: validates string references") and `PLAN_accuracy_and_usd.md:602,777` ("check_transforms.py passes" as a Phase-7 exit criterion) both read as delivered coverage.

**Fix:** include `tests/` (allowlisting negative-test names like `test_heat_to_anchor`) and the remaining `src/` subpackages; assert `n_checked > 0`. Risk: it will immediately fail on the three test call sites until they are allowlisted.

---

### MEDIUM — the coupling guide's measured numbers: aggregate caveat is honest, five specific claims are now false

Full 420-row fast sweep re-run at HEAD with the recorded configuration. Aggregate: **mean iterations ×1.083** (median ×1.051), 323 rows up / 22 down / 75 unchanged — consistent with the guide's own "+10.3%" replay note, and the guide does say its counts are lower bounds. That caveat is accurate. These five are not covered by it:

**(a) "the interface norm removes −5% to 31% of the iterations" (line 410) → measured −9% to 28%.** Both endpoints outside the stated range. The supporting table's worst row reverses qualitatively:

| `stiff-pair-0.95`, gs/none | guide | measured at HEAD |
|---|---|---|
| l2 | 58.4 it, at cap 98% | 59.0 it, at cap 100%, converged **0%** |
| interface | **49.6 it, at cap 34%** | **57.7 it, at cap 94%, converged 8%** |

The guide presents the interface norm as taking this fixture from 98% at-cap to 34%; it now takes it from 100% to 94%, and the reduction fell from 15.1% to **2.2%**. `stiff-pair-0.95` is precisely the fixture behind "Start here" row 3 ("contraction above ~0.9").

**(b) "Gauss-Seidel needs 1.7–1.9x fewer iterations than Jacobi on every shape where both converge (1.69 on `chain-2` to 1.93 on `slow-drift`)" (line 57) → measured 1.59 to 2.04.** `chain-50` is **1.59**, below the stated floor; `chain-2` is 1.83, not 1.69; `slow-drift` 2.04, not 1.93.

**(c) Contradiction 3's headline evidence reverses.** "on `star-16` Aitken buys Jacobi **nothing at all** (42.4 → 42.5)" → measured **47.63 → 42.13, a 12% reduction**. (Gauss-Seidel's 53% → 58%.) The qualitative finding survives; the quoted number no longer supports it.

**(d) `jacobi` / `fixed` ω = 0.8 — the guide's one positive recommendation for a constant ω ("Start here" row 9, contradiction 4) — stopped converging on eight fixtures.** All recorded at 100%:

```
star-16  jac/fixed0.8/l2   100% -> 20%      star-8  jac/fixed0.8/l2  100% -> 40%
star-4   jac/fixed0.8/l2   100% -> 82%      star-2  jac/fixed0.8/l2  100% -> 88%
stiff-pair-0.8 jac/fixed0.8/l2 100% -> 92%  chain-20 jac/fixed0.8/l2  92% -> 58%
```
The recommendation for `slow-drift` itself still holds (100% converged, 6.80→4.56→3.62 against the documented 6.14→4.30→3.46). But a reader applying row 9 to a wide star now gets a group that converges on one step in five, and every table on the page reports it at 100%.

**(e) "Aitken … cannot exit in fewer than four" (line ~325) is false, and is disproved by the repo's own committed baseline** — no re-run needed:
```
ring-4          gs/aitken/interface   iterations_min=3
star-4          gs/aitken/interface   iterations_min=2
star-4          jac/aitken/interface  iterations_min=2
stiff-pair-0.25 gs/aitken/interface   iterations_min=3
stiff-pair-0.25 jac/aitken/interface  iterations_min=3
stiff-pair-0.95 gs/aitken/interface   iterations_min=2
stiff-pair-0.95 jac/aitken/interface  iterations_min=2
```
Seven rows in `coupling_sweep_cpu.json` exit in 2 or 3. The guide derives the floor of 4 from "the threshold on two consecutive passes"; whatever the mechanism, the stated floor does not hold.

**Verified still true:** "not one Gauss-Seidel row beats its unrelaxed counterpart on iterations" — zero violations across all 420 rows at HEAD. Also confirmed the guide's claim that `--steps` does not move iteration counts (chain-2 at `--steps 3`: 6.02 vs 6.06 at the fixture's default).

**Timings excluded.** Five agents shared the box; `chain-50 gs/iqn-ils/l2` measured 1073 ms against the recorded 169 ms. Not reported as fact — but the guide already says that row is "the least reproducible number on this page … treat it as one to two orders of magnitude"; 290× is outside even that. Flagged for re-measurement on a quiet machine.

**Fix:** re-record the baselines (already queued per the guide's own note) and, until then, mark the specific quoted figures rather than only the aggregate. The `jac/fixed0.8` convergence collapse and the `stiff-pair-0.95` interface row should be called out in the "Start here" table, not only in the header caveat.

---

### MEDIUM — after PR 59 lands, the "Start here" table's top two recommendations contradict the guide's own new accuracy caveat

PR 59 adds to `coupling_algorithm_guide.md` (after line 495): *"do not pair the auto-detected `accelerated_fields` with `convergence_norm="interface"` … **Prefer a norm that sees the whole state** (`"l2"`, `"mixed"`)."* Its cited evidence includes an Aitken case at 5.2e-01 disagreement.

The "Start here" table, 440 lines earlier and untouched by PR 59, recommends:
- row 1 (`anything, first attempt`): `gauss-seidel` / **`aitken`** / **`interface`**, auto fields
- row 3 (`contraction above ~0.9`): `gauss-seidel` / **`iqn-ils`** / **`interface`**, auto fields

Row 3 is exactly the pairing PR 59 says not to use; row 1 is the case its cited property test found at 5.2e-01. Rows 4 and 5 are fine (row 4 names `accelerated_fields` explicitly, row 5 uses `none`).

**Non-issue check:** PR 59's full diff touches `CHANGELOG.md`, `benchmarks/results/retire_known_disagreements/*`, `coupling_algorithm_guide.md` (this insertion only), `known_anomalies.yaml` (extends MADD-ANO-005's `residual_risk`), and two test files. It does **not** revise the "Start here" table.

**Fix:** amend rows 1 and 3 when PR 59 lands. One line each.

---

### MEDIUM — the IEC 62304 SOUP evidence set is stale in four independent places

1. **`docs/validation/soup_package.md` §3 lists 2 of 5 anomalies.** Missing MADD-ANO-003 (`open`, `major`, `context_dependent` — the AdaptiveNode gradient anomaly **opened by this release**), MADD-ANO-004 and MADD-ANO-005. The section names `known_anomalies.yaml` as authoritative and then gives a summary table that drifted — the classic two-documents-one-fact case.
2. **`docs/validation/framework_verification.md` lists 1 of 4 registered benchmarks** (only MADD-VER-001). MADD-VER-002/003 predate this release; **MADD-VER-004 was registered by it** (`tests/nodes/adaptive/test_verification.py:60`), and the branch that added it updated its own algorithm guide but not the central index.
3. **Version identification is two-to-three releases stale.** `soup_package.md` §1: `Version 0.1.0`, `Release Date 2025-03-01`. `known_anomalies.yaml` header: `maddening_version: "0.1.0"`, `generated_date: "2026-03-12"`. `CITATION.cff`: `version: 0.1.0`. Package version is `0.4.0.dev0`. This release *did* touch `soup_package.md` (two dependency rows) and left the version.
4. **MADD-ANO-001 and MADD-ANO-002 are `open` but record `affected_versions: "0.1.0"`.** ANO-002 confirmed still live: `HeatNode` at Fo = 5.0 → `max|T| = nan` after 60 steps, **no warning** (stable Fo = 0.4 → 0.057). So the anomaly is true and the version field is wrong. Minor nit in the same entry: the workaround says `dt * alpha / dx^2 < 0.5`; the constructor parameter is `thermal_diffusivity`, and `alpha` is a `TypeError`.

**Fix:** regenerate the summary tables from the YAML (or delete them and link), and bump the three version strings. Risk: none.

---

### MEDIUM — `docs/release_notes/v0.4.0.md` §"Verification evidence" quotes a v0.2.1 test count

**Evidence:** `docs/release_notes/v0.4.0.md:921` and `CHANGELOG.md:328`: *"Full MADDENING test suite: **1680 passed, 3 skipped** (1 deselected via `-m "not slow"`)."* Measured at HEAD:
```
$ pytest tests/ --collect-only -q -m "not slow"   ->  3413/3440 tests collected (27 deselected)
$ pytest tests/ --collect-only -q -m "slow or not slow" ->  3440 tests collected
```
Provenance: `git log -S"1680 passed"` → introduced by `6a597f2` (2026-05-30, *"chore: v0.2.1 release prep"*) and copied verbatim into the 0.4.0 notes by `c93bc89` (2026-09-17). Two releases and ~9,000 lines of new tests stale.

**Structural note worth acting on:** the CHANGELOG convention ("NEVER edit or reorder existing lines") *guarantees* that any count-bearing line goes stale across a parallel-merge release. Count-bearing lines should be generated at release time, not written by a branch.

---

### MEDIUM — `adaptive_node.md`'s "Validated Physical Regimes" states the gradient-capture rule as an active *fraction*; it is a function of the budget alone

`adaptive_node.md:165`: *"Below `K / n_max` ≈ 0.1 the gradient-capture ratio falls under the 0.7 threshold and construction *warns*."*
`docs/developer_guide/adaptive_node.md:191`: *"the ratio is a function of the budget alone — 0.16 / 0.57 / 0.85 / 1.00 at `k = 4 / 8 / 16 / 32`, **at every `n_max`**."* These cannot both be true.

**Evidence** (`repro_capture_ratio_fraction.py`):
```
 n_max    k  K/n_max   ratio  warns?  alg-guide predicts
   256   16   0.0625   0.855   False  warn (frac<0.1)      <-- CONTRADICTS
    64    8   0.1250   0.565    True  no warn (frac>=0.1)  <-- CONTRADICTS
    32    4   0.1250   0.163    True  no warn (frac>=0.1)  <-- CONTRADICTS
    32    8   0.2500   0.565    True  no warn (frac>=0.1)  <-- CONTRADICTS
```
The developer guide is correct and is pinned by `tests/nodes/adaptive/test_blindness_diagnostics.py::test_gradient_capture_ratio_tracks_the_active_set_budget_not_symmetry`. A user at `n_max = 32, k = 4` reads the table as inside the validated regime and gets a ratio of 0.163 — 84% of the full-basis gradient missing, which the same row calls "percent-level".

**What would make this a non-issue:** the runtime warning is correct and does fire, so nobody is left without a signal — hence MEDIUM not HIGH. **Fix:** restate the row in terms of `K` (threshold crossed between K = 8 and K = 16 on the reference problem).

---

### MEDIUM — the other three compliance gates each miss the realistic form of their own defect

Mutation-tested; **6 caught, 22 missed**; every mutation reverted and the worktree verified clean. All four run in `.github/workflows/ci.yml` job `compliance` with exit codes honoured and no `needs:` gate — the wiring is sound, the coverage is not. There is no pre-commit config; these exist only in CI.

**`check_impl_mapping.py`** — resolves with `getattr`, which walks the MRO. **8 of 16 mapping rows name an attribute the base class also defines**, so deleting or renaming the concrete implementation leaves the gate green:
```
adaptive_node.md: AdaptiveNode.update (x2), AdaptiveNode.initial_state
heat_node.md:     HeatNode.update (x5)   -> all still resolve via SimulationNode
```
Confirmed by mutation: renaming `HeatNode.update` → `step` in `src/` → `OK: 16 implementation mapping(s) verified`, exit 0. It also never calls `callable()` despite claiming to, reads only the *first* backticked symbol per row, and pins no minimum row count — so a row with the backticks dropped silently becomes `OK: 15 … verified`, and deleting the whole table becomes `OK: 11 … verified`. Scope is `os.listdir` (non-recursive) over `docs/algorithm_guide/nodes/` only: 2 of 6 algorithm-guide files, and node guides exist for 2 of 12 node modules. The coupling/uq guides carry no Implementation Mapping table today, so the scope gap is latent rather than live.

**`check_citations.py`** — catches a dangling key and a deleted entry, but a **`%`-commented bibliography entry passes** (`% @book{Crank1975,` → `OK: 13 citation(s) verified`, exit 0), which is exactly the broken-reference case the gate exists for: BibTeX ignores it, Sphinx renders a broken ref. Also misses duplicate keys (set-collapsed), mis-tokenises hyphenated keys, and scans only `docs/algorithm_guide/**` — 5 of 41 doc files, 0 of 198 src files. Pointing it at `docs/` finds 5 pre-existing dangling `[@Key]` placeholders in `developer_guide/documentation_standards.md` and `node_authoring.md` (template examples, so widening needs them excluded first). `DOCUMENTATION_ARCHITECTURE.md:472` claims "CI validates **all** cited keys exist".

**`check_anomalies.py`** — strongest of the four on its declared schema (caught a missing required field, a bad `severity`, a duplicate id) but the schema is a third of the file. `src/maddening/compliance/_validate.py:20` requires 6 fields and enum-checks 2. **`resolution_status` is neither required nor enum-checked** — `resolution_status: "probably fine tbh"` passes, and so does deleting it; it is the one field an IEC 62304 known-anomalies list is read for. Unknown keys are not rejected (`workaroud:` silently drops the workaround, exit 0); `verification:` test paths and `affected_components` symbols are never resolved, though `check_impl_mapping.py` already contains the resolver. CI invokes it without `--prefix`, so its prefix rule is dead there — but `tests/compliance/test_validator.py::test_repo_registry_uses_madd_prefix` covers it in the next CI step.

Separately verified that today **all 5 anomalies' `affected_components` import cleanly and all 7 `verification` entries resolve to a real file *and* a real `def` name** — so nothing is currently broken, only unguarded.

---

### MEDIUM — `FMIVariable`'s field order changed, with no CHANGELOG or release-note entry

```
FMIVariable in maddening.fmi.__all__: True
positional 8th arg -> node = (3,)   shape = None       # on 0.3.x this set shape=(3,)
```
`node` and `field` were inserted **between** `unit` and `shape` (`src/maddening/fmi/model_description.py:133-137`). Nothing in `CHANGELOG.md`, `docs/release_notes/v0.4.0.md` or `docs/` mentions `FMIVariable`, `node_field` or the reordering.

Full AST signature diff of the public surface (`origin/main` vs HEAD, 998 → 1060 defs): **0 public definitions removed**, 16 changed. All the rest are additive keyword-only parameters or appended dataclass fields. The only other insertion is `FIMReport.rank` between `eigvecs` and `cond` — that one *is* in the CHANGELOG as a new field, though the positional break is not called out.

---

### LOW — two serialisation paths disagree on a bad `ParamSpec` override, and neither behaviour is documented

`repro_paramspec_override_asymmetry.py`:
```
--- path 1: GraphManager.from_dict ---
    ValueError: param_specs['s']['no_such_parameter'] cannot be applied to this config …
--- path 2: load_graph_from_usd, same override ---
    loaded OK.  warnings: ['RuntimeWarning: ignoring ParamSpec override s.no_such_parameter from the USD stage: …']
```
`graph_manager.py:~4900` raises; `usd/serialization.py:417-424` warns and drops. `interface_mapping.md` §Serialisation describes config and USD as carrying the same thing. Secondary: the `from_dict` message says *"'s' is neither one of its nodes nor one of its mapped edge keys"* when `s` **is** one of its nodes — the real cause is only in the chained `__cause__`.

### LOW — the coupling guide contradicts itself, and the release's own last commit, on two mechanisms

1. `coupling_algorithm_guide.md:512-513`: *"the interface norm is a *relative* criterion (`atol`/`rtol`), so it stops earlier than **an absolute L2 tolerance**."* The same page's header (line 30) and `CouplingGroup`'s docstring (`group.py:53-70`) say all three norms are now relative. PR 59 sharpens this by recommending `"l2"` for accuracy 20 lines earlier.
2. `coupling_algorithm_guide.md:37`: ω = 0.5/0.8 *"halves every step and so has the largest gap between its residual and its error."* Commit `0ee18a0` (PR 58, the release's most recent merge) **retracted exactly this mechanism** in `benchmarks/results/convergence_error_bound/REPORT.md` — measured `rho` as low as 0.156 at ω = 0.5 over 20k random matrices. It fixed one of three copies; the other two survive here and at `tests/core/test_coupling_fixture_invariants.py:236-238`.

### LOW — smaller items
- `RigidBody2DNode` emits a `DeprecationWarning` on **every** instantiation and has since `CHANGELOG.md:624`, but `stability_report.md:56` publishes it as `experimental`. Four other surfaces warn deprecated without a `DEPRECATED` tag (`checkpoint.download_and_load_state`, `AdaptiveNode.blindness_ratio`, `blindness_threshold=`, `CouplingGroup(solver="fori")`) — those four *are* documented as deprecated, so only the tag is missing.
- Both genuinely-`DEPRECATED` symbols (`calibration.calibrate`, `tune_coupling_params`) warn correctly and are in the CHANGELOG, but appear nowhere in `docs/`.
- `docs/developer_guide/verification.md` is in no toctree; reachable only via a relative link from `parameters.md:83`.
- `src/maddening/sysid.py` has no `__all__`, so `maddening.sysid` re-exports `stability`, `StabilityLevel`, `estimated_error`, `jnp`, `jax` as public names.
- `typing.md:164` annotation snapshot ("194 modules, ~1498 function definitions, 57% with a return annotation, 235 bare `dict`") measures 198 / 1643 / 59.9% / 205 now. Labelled "measured before phase 1", so this is drift-by-design.
- Release notes line 1030: "eleven sites keep an absolute cap … each says so in a comment" → 20 such sites now, 6 with no comment in the preceding four lines. "95 call sites name a tier" → 153. No site is below the floor of 20.
- `typing.md`'s pyright baseline (395 errors / 94 warnings) could not be re-measured: pyright is not in the shared venv and installing it is forbidden.

---

## Unverified suspicions

- **`chain-50 gs/iqn-ils/l2` measured 1073 ms against a documented 169 ms** (290×). The guide already flags this row as irreproducible; five agents shared the machine. Needs a quiet box.
- The release notes' gcov claim ("26% to 57% of the wrapper's lines") and the hypothesis-profile wall times are not re-measurable without a full C build and a full property-suite run. The tier *values* they describe do check out exactly (`dev`: 200/50/20, `ci`: 800/200/80).
- `MADD-ANO-001` (LBM GPU segfault) is untestable here — no CUDA jaxlib in the venv.

## What I checked and found sound

- **Test-path traceability:** 45 of 47 `tests/…` paths cited across `docs/`, `README.md`, `CHANGELOG.md` resolve to a real file, and every `::test_name` to a real `def`. The two failures are `test_your_node.py` placeholders in authoring templates.
- **Anomaly registry integrity:** all 5 entries' `affected_components` import; all 7 `verification` entries resolve to file *and* test name. `tests/core/test_coupling_error_bound.py` — MADD-ANO-005's evidence — is 13/13 green.
- **Documented API:** every symbol and signature in `README.md` and the five `docs/user_guide/` pages exists and matches. AST signature diff: **0 public definitions removed** since `main`.
- **Executable docs:** all three quickstart/README snippets run to completion with zero warnings (final height 0.284; coupled 1.58; `jax.grad` through 100 steps = 0.0721).
- **Reproduced documented measurements to three digits:** gradient-capture ratios 0.855 / 0.174 / 0.000 at θ = 0.42 / 0.48 / 0.50 (doc: 0.86 / 0.17 / 0.0); hypothesis tier resolution; `MappingSpec` limits `INLINE_POINT_LIMIT=64`, `INLINE_ELEMENT_LIMIT=1024`, `MAX_ASSET_BYTES=256 MiB`; `parameters.md`'s FIM claim (rank 2 of 3, weakest eigenvector `(1,1,1)/√3`, all CRBs `+inf`).
- **Cross-document consistency where drift was expected and not found:** the four AdaptiveNode descriptions agree on −1.5808e-3, −2.3598e-3, 33%, 27 jumps, 8–10×, k=64→1e-8; `installation.md`'s Python ≥ 3.11 matches `pyproject.toml`; no `stelling` references survive outside the release note recording its removal; the `[ift]` extra's emptiness is documented in three consistent places; `unit_transforms.md` names only `lbm_to_si_*` factories that exist.
- **Algorithm-guide structure:** `adaptive_node.md` carries all 14 mandatory sections from `_template.md` (IEC 62304 Clause 5.4), in order.
- **Traceability IDs:** every `MADD-VER-*` and `MADD-NODE-*` referenced in docs exists in code.
- **Coupling guide claims that survived re-measurement:** "not one Gauss-Seidel row beats its unrelaxed counterpart" (0 violations in 420 rows); "`--steps` changes timings and nothing else" (6.02 vs 6.06); the `slow-drift` ω = 0.8 ranking; the aggregate "+10.3% iterations" caveat (measured +8.3%).
- **CI wiring of the four gates:** all in one ungated job, exit codes honoured, no `|| true`, no `continue-on-error`.
