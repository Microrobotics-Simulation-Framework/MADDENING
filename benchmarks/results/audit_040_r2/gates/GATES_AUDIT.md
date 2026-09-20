# Audit: CI gates and compliance checks (round 2)   (0219b82c9a5a22b209c1dd370cc7d406e4ec1079)

Worktree: `/home/nick/MSF/msf/MADDENING-wt/audit-r2-gates` (flattened name, no
`test` path component — `audit_property_rejection.py`'s own banner reports
`constant injection: ON for src/maddening (670 local constants harvested)`, so
measurements here match CI's configuration).

Reproducers: `repro/g1..g7`, raw output in `repro_output.txt`. Every reproducer
restores the tree; `git status --porcelain` is empty after each run.

## Summary

I mutation-tested 11 gates: the five `scripts/check_*.py` compliance gates, the
SOUP and stability-report equality gates, the compile-count gate, the property
draw-rejection gate, the MMS/GCI order harnesses, and the anomaly and benchmark
registries. **34 mutations, 25 caught, 9 missed.**

Nothing found is a wrong number in shipping code. What I found is that four
gates can report success having verified nothing or having verified less than
they say, and that **neither registry pins its own membership**: deleting an
open anomaly, or colliding two benchmark IDs, and then running the generator the
equality gate's error message tells you to run, leaves all 188 compliance tests
green with the entry gone from the IEC 62304 evidence set (G5). That is the
highest-value item here and it is not on the R2 floor list.

The tree is far better defended than round one's record suggests: the
stability-report equality gate, the SOUP equality gate, the compile-count gate,
the integrator scheme-identity check and the MMS ladder all caught every
mutation I threw at them, including a 1e-5 perturbation of an RK4 Butcher
weight. Those are recorded in "What I checked and found sound".

---

## Findings

### MEDIUM — Neither registry pins its membership: an anomaly or a benchmark can be deleted with every gate green

**What breaks:** two separate paths, same mechanism.

*A.* Remove `MADD-ANO-012` (`resolution_status: open`) from
`docs/validation/known_anomalies.yaml`. `scripts/check_anomalies.py` passes
(there is no count, ID-sequence or membership pin). `generate_soup_tables.py
--check` then fails — correctly — with a diff, and its message says
*"docs/validation/ is stale — run `python scripts/generate_soup_tables.py` and
commit the result."* Doing exactly that makes everything green and removes
`MADD-ANO-012` from `soup_package.md`. The released known-anomalies list is
short by one open defect.

*B.* Give two `@verification_benchmark` decorators the same `benchmark_id` (a
copy-paste slip; `MADD-VER-011` → `MADD-VER-010`).
`_BENCHMARK_REGISTRY[benchmark_id] = benchmark` at
`src/maddening/core/compliance/validation.py:76` is a plain dict assignment with
no duplicate check, so the second decorator silently deletes the first. Same
regenerate-and-commit step, same result: `MADD-VER-011` is gone from
`docs/validation/framework_verification.md`, 18/18 `test_soup_evidence.py`
green.

**Evidence:** `bash repro/g5_registry_silent_deletion.sh`

```
=== A. delete an OPEN anomaly from the registry ===
OK: anomaly registry at docs/validation/known_anomalies.yaml is valid
check_anomalies rc=0
soup --check after regenerating rc=0
MADD-ANO-012 in soup_package.md: 0
188 passed in 15.74s

=== B. two @verification_benchmark decorators with the same id ===
soup --check after regenerating rc=0
MADD-VER-011 in framework_verification.md: 0
18 passed in 1.03s
```

**Why it happens:** every check runs *registry → document*
(`tests/compliance/test_soup_evidence.py:59`
`test_every_anomaly_in_the_registry_reaches_the_summary_table`, and `:70` for
benchmarks). Nothing runs document → registry, and nothing pins the registry's
size or ID set. The equality gate compares the committed table against a *fresh
generation from the same registry*, so once the registry shrinks the two agree
again. The anomaly validator does check `Duplicate anomaly_id`
(`src/maddening/compliance/_validate.py:374`); the benchmark registry has no
equivalent.

**What would make this a non-issue:** (a) if the IDs were non-contiguous anyway,
so a gap meant nothing — they are not, the registry holds exactly
`MADD-ANO-001..015` and the benchmarks exactly `MADD-VER-001..013`, both
contiguous; (b) if code review always caught the registry diff — it is the one
line of defence, and this release has already recorded a band being widened and
a wrong count being quoted into three reports without review catching either;
(c) if some other artefact pinned the count — I grepped `tests/compliance/` for
`len(...anomalies)`, `MIN_ANOMAL`, `N_ANOMALIES` and found nothing.

**Suggested fix:** pin the ID sets. A test asserting the anomaly registry
contains a contiguous `MADD-ANO-001..N` with `N` a committed constant, and the
same for `MADD-VER`, turns both deletions red. Additionally make
`verification_benchmark` raise on a duplicate `benchmark_id` rather than
overwrite — a two-line change that catches B at import time, where it belongs.
Risk: the ID pin needs bumping with every new anomaly, which is the point; a
contiguity assertion would need relaxing if an ID is ever retired, so record
retirements explicitly rather than by deletion.

---

### MEDIUM — `MIN_MAPPINGS` is the only protection for the mapping tables, and nothing stops it being lowered

**What breaks:** lower `docs/algorithm_guide/nodes/heat_node.md` from `9` to `1`
in `scripts/check_impl_mapping.py:46`, delete 8 of the guide's 9 mapping rows:
gate green, all 9 `TestImplementationMappingGate` tests green.

**Evidence:** `bash repro/g6_min_mappings_no_ratchet.sh`

```
removed 8 mapping rows from heat_node.md
OK: 50 implementation mapping(s) verified across .../docs/algorithm_guide
gate rc=0
9 passed, 33 deselected in 0.63s
```

**Why it happens:** `check_pinned` (`scripts/check_impl_mapping.py:180`) compares
against `MIN_MAPPINGS`, which lives in the same file the author is editing. The
comment at line 44 says *"Raise a number when a guide gains rows; never lower
one to make CI pass"* — a convention, not a check.
`tests/compliance/test_gate_scripts.py:206`
(`test_every_pinned_guide_is_satisfied_by_the_repository`) reads the *current*
`MIN_MAPPINGS`, so it moves with the mutation.

Two neighbouring allowlists in the same surface *are* enforced:
`_MAX_ALLOWED_UNRESOLVABLE` (`check_transforms.py:65`) is capped and each entry's
reason and staleness is tested (`test_gate_scripts.py:364-402`), and
`_ALLOWED_UNSTABLE` (`check_heat_stability.py:67`) is pinned by
`test_the_allowlisted_file_still_plants_a_defect`. `MIN_MAPPINGS` and
`_TEMPLATE_CITATIONS` (`check_citations.py:42`) are the two with no guard at all.

**What would make this a non-issue:** the numbers are visible in a diff. So was
MADD-VER-002's `[0.7, 2.5]`. I checked whether any test snapshots the values
(`grep -rn MIN_MAPPINGS tests/`): only `test_gate_scripts.py:208`, which passes
the live dict in.

**Suggested fix:** commit the expected values as data next to the test and assert
`MIN_MAPPINGS[path] >= expected[path]`, so lowering a pin requires editing two
files in opposite directions. Risk: one more file to update on a legitimate
guide shrink — which should be rare and deliberate.

---

### MEDIUM — Three gates report success having verified nothing

`check_heat_stability.py` fails in both of these cases; `check_transforms.py` in
the first. The others have neither.

**What breaks:** `bash repro/g7_gates_that_verified_nothing.sh`

| gate | scope given | printed | rc |
|---|---|---|---|
| `check_anomalies.py` | registry with `anomalies: []` | `OK: ... is valid` | **0** |
| `check_anomalies.py` | live registry, every `affected_components`/`verification` stripped | `OK: ... is valid` | **0** |
| `check_impl_mapping.py` | an empty directory | `OK: 0 implementation mapping(s) verified` | **0** |
| `audit_property_rejection.py --check` | a suite where every property test skips | `no Hypothesis runs were observed` | **0** |

**Why it happens:**

* `scripts/check_anomalies.py:98` — the zero-coverage guard is
  `if notes and not args.no_resolve:`. `notes` is only populated by references
  that were *skipped as unavailable*, so a registry with no references at all
  never enters the guard. `validate_anomaly_registry` explicitly permits an empty
  list (`_validate.py:353`, "must be a list, may be empty").
* `scripts/check_impl_mapping.py:258` — no guard on `checked == 0`. The pins do
  still run (they resolve against `_REPO_ROOT`, not the scanned directory), so
  the gate is not blind, but the line it prints is a pass that names a scope it
  verified nothing in, and that line is what gets quoted.
* `scripts/audit_property_rejection.py:544` — `return 1 if over else 0`. Zero
  records means zero `over`. An empty *collection* is caught (pytest exits 5),
  but an all-skipped or all-deselected run, or a Hypothesis refactor that stops
  calling `hypothesis.statistics.collector`, is not. The banner even prints
  "no Hypothesis runs were observed" and then exits 0.

**What would make this a non-issue:** for the property gate, whether the CI scope
can actually go empty. `verify-hypothesis` runs
`tests/verification/hypothesis/`, whose tests carry no skip markers today — so
this is a latent failure mode, not a live one. For `check_anomalies` the empty
registry is implausible on its own; the realistic shape is a merge that truncates
the file, and the CI invocation
(`python scripts/check_anomalies.py --prefix MADD-ANO-`) would pass on it.

**Suggested fix:** propagate `check_heat_stability.py`'s two guards. For
`check_anomalies`, count anomalies and resolved references unconditionally and
fail on zero of either. For `audit_property_rejection`, fail `--check` when
`plugin.records` is empty. For `check_impl_mapping`, fail when `checked == 0` in
the scanned directory. Risk: a contributor legitimately pointing a gate at a
narrow scope now gets a failure — acceptable, and it is what the other two gates
already do.

---

### MEDIUM — `check_transforms.py` cannot see a transform passed positionally, and accepts a registration that never executes

Two independent holes in the same gate, whose stated contract is "every string
edge-transform reference resolves" and whose stated reason is USD serialization
(a stage records the *name* and resolves it on load).

**What breaks:**

*A.* `transform` is the 5th positional parameter of `GraphManager.add_edge`
(`src/maddening/core/graph_manager.py:2959`) and the 5th field of `EdgeSpec`
(`src/maddening/core/edge.py:55`). `gm.add_edge("a","b","x","y","extract_last")`
is accepted at runtime and resolves the string. The gate reads only
`node.keywords` (`check_transforms.py:108`), so the positional form is invisible.

*B.* `find_local_registrations` (`:119`) walks the whole AST for a
`register_transform("name")` *call expression*. A registration inside a function
nobody calls satisfies the gate; the name is not in the registry after import.

**Evidence:** `bash repro/g2_transforms_positional.sh`

```
--- keyword form (the gate sees it) ---
FAIL: 1 unresolvable transform reference(s):
  tests/core/_g2_probe.py:2: transform 'definitely_not_registered' is neither in
  the TransformRegistry nor registered in this file
rc=1
--- positional form, identical meaning (the gate does not) ---
OK: 31 string transform reference(s) verified (6 transforms in registry)
rc=0   <-- 0, the bogus name is invisible
```

`bash repro/g3_transforms_dead_registration.sh`

```
OK: 32 string transform reference(s) verified (6 transforms in registry)
gate rc=0
in registry after import? False
^ gate said verified; registry says no. add_edge would raise KeyError.
```

**What would make this a non-issue:** if nothing in the tree used the positional
form. Nothing does today (grep for module-qualified and 5-positional `add_edge`
calls found none), so both are latent. B's realistic shape is a registration in a
helper that a fixture forgot to call, or one behind a `try/except ImportError`
fallback — not the `if False:` a minimal reproducer would use, which is why
`repro/g3` uses the uncalled-helper form.

**Suggested fix:** for A, read `node.args[4]` as well as the keyword; for B,
resolve against the live registry after importing the module rather than trusting
the lexical presence of the decorator — or, cheaper and sufficient, keep the
lexical check but additionally assert that every name the gate credits to a local
registration is in `_TRANSFORM_REGISTRY` after that module is imported. Risk: B's
stricter form makes the gate import test modules, which is slower and can fail on
an optional extra; `resolve_dotted_name`'s `unavailable` handling is the
precedent for how to degrade.

---

### LOW — `check_heat_stability.py` sees only `HeatNode(...)` spelled as a bare name

**What breaks:** two rods at Fourier 0.6605 (the guard's limit is 0.5), written
`heat.HeatNode(...)` and via `from ... import HeatNode as Rod`, are invisible.
With the repository's own calls in scope the gate is green.

**Evidence:** `bash repro/g4_heat_call_form.sh`

```
OK: 124 HeatNode construction(s) verified within their stencil's stability
limit; 73 further construction(s) have computed arguments and were NOT checked
rc=0   <-- 0, with two rods in scope that HeatNode.__init__ refuses
runtime: timestep 0.0001 is unstable for this rod: the Fourier number
dt*alpha/dx^2 is 0.6605, abov[e the 0.5 limit]
```

**Why it happens:** `scripts/check_heat_stability.py:123` matches on
`getattr(node.func, "id", None) == "HeatNode"`, i.e. an `ast.Name` only. An
`ast.Attribute` (`heat.HeatNode`) has `.attr`, not `.id`; an aliased import binds
a different `.id`.

The gate's own zero-scope message anticipates exactly this — *"or the call is
spelled in a way this gate does not recognise"* — but that guard only fires when
there are **no** recognised calls anywhere, and the repository has 132.

**What would make this a non-issue:** nothing in the tree constructs a node
module-qualified or aliased today
(`grep -rn "nodes\.\(Heat\|Spring\|Ball\|RigidBody\|HeartPump\)[A-Za-z]*("` over
`src/ tests/ benchmarks/ docs/`: no hits), and no doc shows that style. LOW
rather than MEDIUM for that reason.

**Suggested fix:** match `getattr(node.func, "attr", None)` too, and resolve
module-level `import ... as` aliases the way `_module_level_string_constants`
already resolves string constants in `check_transforms.py`. Risk: `attr`-matching
could pick up an unrelated `something.HeatNode` — harmless, it would be reported
as unchecked at worst.

---

### LOW — `check_citations.py` reports 50 citations verified; it verified 45

**What breaks:** the five `_TEMPLATE_CITATIONS` occurrences are `continue`d
before the existence check (`scripts/check_citations.py:165`) but are still in
`len(citations)` on the OK line at `:187`.

**Evidence:** `python repro/g1_citations_count_conflation.py`

```
gate prints          : OK: 50 citation(s) verified
actually verified    : 45
skipped, but counted : 5
    docs/developer_guide/documentation_standards.md 39 Key
    docs/developer_guide/documentation_standards.md 71 Key
    docs/developer_guide/node_authoring.md 155 Key
    docs/developer_guide/node_authoring.md 181 Key
    docs/developer_guide/node_authoring.md 528 Key
```

Same class as `check_impl_mapping`'s "16 verified of 17" and PR 90's
"132 → 232": unverified references inflating the headline. The mapping gate and
the heat gate both now report the two numbers separately
(`"..., N not checked"` / `"; N further constructions ... were NOT checked"`);
the citation gate does not.

`_TEMPLATE_CITATIONS` is also the one allowlist in the tree with **no** guard: no
reason string, no size cap, no staleness check — compare `_ALLOWED_UNRESOLVABLE`
and `_ALLOWED_UNSTABLE`, both of which have all three.

**Suggested fix:** subtract the skipped occurrences from the headline and report
them separately, and give `_TEMPLATE_CITATIONS` the reason/cap/staleness
treatment `_ALLOWED_UNRESOLVABLE` has. Risk: none.

---

### LOW — `DEFAULT_ORDER_EXCESS = 1.0` is justified by a measurement its own band rejects

**What breaks:** `src/maddening/testing/mms.py:351` justifies `1.0` as *"wide
enough for the genuine superconvergence seen here — the corrected fourth-order
stencil measures 5.02 over one pair of a fourth-order ladder"*. For
`expected = 4`, `check_order` computes `high = 4 + 1.0 = 5.00`. A measurement of
**5.02 would FAIL the band the sentence is defending**. The stated justification
and the constant are mutually inconsistent as written, whichever number is
currently right.

This is the same docstring the R2 floor flags for citing figures that no longer
reproduce (5.02 vs "measures at most 3.98"; `DEFAULT_ORDER_SHORTFALL`'s cited
"HeatNode 1.982" against the registry's own MADD-VER-005 acceptance criteria,
which records `measured: 2.000`). The arithmetic contradiction is a second,
independent defect in the same block: it does not depend on which measurement is
current.

**Suggested fix:** re-measure and rewrite both justifications from the current
ladders, and state the band edge explicitly (`4 + 1.0 = 5.00`) so a cited figure
outside it is visible on the page.

---

### LOW — MADD-VER-003's acceptance criteria state ±25%; the executed assertion is [0.75, 1.30]

**What breaks:** `tests/cloud/multigpu/test_lbm_poiseuille.py` registers
`acceptance_criteria=... "centreline velocity within +/-25% of u_max = F R^2 /
(4 mu)"`, the inline comment at `:173` repeats *"Tolerance: within +/-25% of
nominal u_max"*, and the assertion at `:179` is `assert 0.75 < ratio < 1.30` —
−25%/+30%. The criteria string is what `generate_soup_tables.py` renders into
`docs/validation/framework_verification.md`, so the published verification index
overstates the strictness of the executed check.

Same class as MADD-VER-002's `0.7 < avg_rate < 2.5` against a claimed 1.5–2.5,
but much smaller: the measured ratio is ~1.13, comfortably inside ±25%, so the
extra 5 points are not currently covering anything. That is why this is LOW and
not a repeat of the MADD-VER-002 finding.

I swept the other twelve benchmarks' `acceptance_criteria` against their executed
assertions; this is the only disagreement. MADD-VER-002's is now `[1.7, 2.3]` in
both the criteria and the assert, with a paragraph of justification and both
edges defended.

**Suggested fix:** make the two agree — either widen the criteria string to
`-25%/+30%` with the reason (mid-link bounce-back puts the effective radius above
nominal, biasing the ratio up), or tighten the assert to 1.25.

---

## Unverified suspicions

**The draw-rejection gate never measures the population its threshold was derived
from.** `MAX_REJECTION = 0.40` is derived in
`scripts/audit_property_rejection.py:79` from *"over 170 tests that draw"* across
`tests/property` **and** `tests/verification/hypothesis` — the script's own
default `paths` are both. CI's `verify-hypothesis` job passes only
`tests/verification/hypothesis/` (`.github/workflows/ci.yml:294`).
`tests/property/` appears in no workflow except the blanket `pytest tests/`,
which does not load the plugin. So ~89 `@given` decorations across 16 modules,
including `test_a_mask_whose_keys_differ_from_params_is_refused` — the test the
script's own docstring names as the tree's worst overrun at 37.6% — are never
measured by the gate that cites them.

*What would make this a non-issue, and what I checked:* whether anything in
`tests/property/` is actually over a gate today. I measured it rather than
guessing (`property_scope_dev.log`, `property_scope_ci.log`,
`property_scope_dev.json`):

| profile | modules | worst filter rate | worst overrun rate |
|---|---|---|---|
| `dev` | sysid_contract, round_trips, mapping_spec_resolver (49 drawing tests) | 39.4% (`test_fim_matches_a_finite_difference_of_a_rollout`) | 37.0% |
| `ci` | sysid_contract, three worst classes (9 drawing tests) | 19.2% (same test) | 37.7% |

Nothing is over `MAX_REJECTION = 0.40` or `MAX_OVERRUN = 0.50` today, so this is
a scope gap and not a live violation — but at `dev` the worst is **0.6 points
under the gate**, in a directory the gate never runs against. Downgraded from a
finding to a suspicion on that measurement. Adding `tests/property` to the
`verify-hypothesis` invocation costs one extra suite run; the alternative is to
say in the docstring that the constant is derived from a wider population than it
governs.

**`allow_inherited` is triggered by the bare substring `inherited` anywhere in
the row.** `check_impl_mapping.py:147` is
`allow_inherited = "inherited" in row_text.lower()`. I confirmed the mechanism:
with the notes cell reading `Added as ...`, a row naming
`maddening.nodes.heat.HeatNode.to_dict` fails with *"resolves only through base
class SimulationNode"*; changing the same cell to `Inherited from the base; added
as ...` turns it into a NOTE and the gate passes. The hazard is a row whose prose
says the *opposite* — "overrides the inherited default", "not inherited" — which
also contains the substring and so also disables own-definition checking. No such
row exists today (I read all 58 references), so there is nothing to reproduce
against the live tree; it is a trap for the next author.

**`_ALLOWED_UNSTABLE` skips whole files, not constructions.**
`check_heat_stability.py:211` `continue`s on the file, so
`tests/compliance/test_gate_scripts.py`'s *stable* rods are also unexamined and
counted in neither `seen` nor `unchecked`. Harmless today (that file exists to
plant defects) but the exemption is wider than its stated reason. Not worth a fix
on its own.

**An anomaly's `verification:` entry is satisfied by a `def` existing.**
`resolve_test_reference` (`_validate.py:258`) checks the file exists and every
`::` component is defined; it does not check the test runs. Three of
MADD-ANO-004's entries are `@pytest.mark.skipif(not _HAS_4, ...)`. In CI the
multigpu conftest forces 16 virtual devices so they do run — checked — but the
registry would read as verified in an environment where they cannot. I did not
find a case where this is currently false evidence.

## What I checked and found sound

Recorded because the round asked for it. Every line below is a mutation I applied
and the gate caught, naming the right thing.

**Stability-report equality gate** (`test_stability.py:273`) — 2/2 caught.
Flipping one surface's level `stable` → `evolving` in the committed report, and
deleting one row outright, both fail with a unified diff naming the row. Its
companion coverage gate (`:176`, the one repaired in `0219b82`) — 1/1: removing
`maddening.api.auth` from `STABILITY_MODULES` while it still carries `@stability`
tags fails and names the module. I confirmed the repair is real: the old prefix
match could not fail because `maddening` is itself listed.

**SOUP equality gate** (`generate_soup_tables.py --check`) — 1/1 caught. A single
altered anomaly ID in `soup_package.md` fails with the diff. (Its blind spot is
the deletion path in the first finding, which is a membership problem, not an
equality problem.)

**Integrator scheme identity** (`tests/verification/test_integrator_order.py`) —
5/5 caught. Rewiring `rk4_step` to delegate to `heun_step` fails **12** tests,
including `test_each_stepper_reproduces_its_own_stability_polynomial` and
`test_the_three_steppers_are_distinguishable_on_a_single_step`. Perturbing the
RK4 `b` weights by `±eps` while preserving `sum(b)` and the order-2 condition is
caught at eps = 1e-2, 1e-3, 1e-4 **and 1e-5**. This is the strongest gate in the
surface and PR 88's answer to "a ladder cannot see which scheme you used" holds.

**`check_impl_mapping`** — 3/5 caught (misses in findings above). Renaming a
table-row symbol to a non-existent method fails and names the symbol and the
equation term. Stripping the `maddening.` prefix off one row's code span (leaving
a plausible-looking code span behind) is caught by the `MIN_MAPPINGS` pin. A row
naming a base-only attribute without the word "inherited" fails with the right
message. All 7 algorithm guides that carry an Implementation Mapping table are
pinned — I enumerated them; there is no unpinned guide with a table.

**`check_citations`** — 3/5 caught. A dangling `[@NoSuchKey2099]` in a `docs/`
file fails with file:line. Commenting out a cited BibTeX entry (`% @book{...`)
fails on all four citations of it. Duplicate keys and the hyphenated-key
tokenisation are covered by `test_gate_parsers.py`. *Scope gaps, both
deliberate-looking but worth knowing:* a dangling citation in a repo-root `.md`
(`README.md`, `CONTRIBUTING.md`, 12 files) or in a `_`-prefixed docs file is not
seen — the gate scans `docs/**/*.md` only. `DOCUMENTATION_ARCHITECTURE.md`
already contains `[@AuthorYear]`, unverified.

**`check_anomalies`** — 2/4 caught. A non-resolving `affected_components` symbol
and a `verification` entry naming a non-existent test both fail and name the
anomaly, the entry and the reason. (The two misses are the zero-scope cases
above.)

**`check_heat_stability`** — 2/3 caught. An unstable rod in plain `HeatNode(...)`
form fails with the Fourier number, the limit, the stencil order and the largest
admissible timestep. An empty scope fails. Its two zero-coverage guards, its
allowlist pin, its signature-derived defaults and its `textwrap.dedent` recursion
are all genuinely load-bearing; its summary line is the only one in the tree that
reports verified and unverified counts separately (`132 verified; 103 further ...
NOT checked`), which is the model the other four should follow.

**Compile-count gate** (`scripts/compile_counts.py`) — read in full, no mutation
needed to conclude it fails closed, and its own suite already mutation-tests it
(`test_the_gate_fails_on_an_extra_retrace_and_names_it`,
`test_the_gate_refuses_a_baseline_taken_on_another_device_count`,
`test_the_gate_refuses_a_baseline_that_records_no_device_count`,
`test_the_pinned_device_count_survives_any_inherited_xla_flags`). `compare()`
fails on a field present on only one side, on a workload present on only one
side, on any topology difference (unbanded), and exactly on
`retrace_count`/`scan_retrace_count`/`scan_steps`. `pin_device_count` strips both
absl spellings of the inherited flag. The docstring/code mismatch the R2 brief
records is fixed and pinned. `TOLERANCES` is two-sided and justified by a
34-number measurement on two JAX versions.

**MMS order harness** (`check_order`, `src/maddening/testing/mms.py:503`) — sound
by construction: a non-monotone ladder fails as *inconclusive* with its own
message rather than as a wrong node, a non-finite order fails with "the scheme is
exact on this manufactured solution and the study measures nothing", and the band
is two-sided. `tests/verification/test_mms_order.py` carries an explicit
fixture-expressiveness suite —
`test_the_curvature_does_not_vanish_at_either_rod_end`,
`test_the_fourth_derivative_does_not_vanish`,
`test_the_profile_is_not_symmetric_about_the_rod_centre` — which is PR 90's
lesson (`u'' = 0` at both ends) turned into standing tests. The ODE module has
the equivalent in `TestTheLaddersCanFail`, including
`test_an_order_study_cannot_tell_forward_from_semi_implicit_euler`, which asserts
the blind spot rather than pretending it is covered. I found nothing to add here.

**GCI harness** (`check_gci`, `:1711`) — ordering is right: non-convergent regime
first, then indeterminate order, then "positively outside the asymptotic range",
only then the order band and the GCI. `verify_node_gci` returns SKIP rather than
PASS for a node with no declared order, and the skip carries the whole
measurement. `check_gci(study)` with neither `expected` nor `max_gci` returns
PASS, but says so in the summary ("no order or band was asked of it") and no
caller does that.

**Benchmark acceptance bands** — all 13 `MADD-VER-*` criteria strings read
against their executed assertions. One disagreement (MADD-VER-003, above).
MADD-VER-002 is now consistent and both edges are defended with measurements.

---

## Method notes

- 34 mutations across 11 gates; each applied to the worktree, gate run, tree
  restored, `git status --porcelain` verified empty. No `src/` or `tests/` change
  survives.
- I did not run the full suite. The heaviest thing run was
  `pytest tests/compliance/` (188 tests, 16 s) inside `repro/g5`, plus two
  targeted `tests/property` audit runs.
- No process was killed by pattern.
