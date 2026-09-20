# Audit: compliance documentation as a body of claims (round 2)

Commit audited: `0219b82` ("fix(compliance): the stability coverage gate could
not fail"), via `git worktree add /home/nick/MSF/msf/MADDENING-wt/audit-r2-compliance origin/release/0.4.0 --detach`.

Surface: `docs/validation/known_anomalies.yaml`, `docs/validation/soup_package.md`,
`docs/validation/framework_verification.md`, `docs/release_notes/v0.4.0.md`,
`CHANGELOG.md`, `docs/algorithm_guide/**`, `docs/regulatory/**`, and the
registered benchmark criteria in `@verification_benchmark` decorators.

## Summary

Thirteen findings. The registry of record is wrong in four ways a downstream
SOUP assessment would act on: MADD-ANO-004 says 0.2.x is unaffected by a defect
that reproduces there; MADD-ANO-002's "re-verified at 0.4.0.dev0 and still live"
paragraph was written the day *before* the fix that refutes it; MADD-ANO-005's
workaround tells the reader to read two diagnostics keys this release deleted
from the report; MADD-ANO-015 says a critical safety-relevant defect affects
0.4.0 when its own description says it stopped at 0.3.1. The release notes'
known-anomalies list still shows 2 of 8 live anomalies and its breaking-changes
table still has the duplicated coupling row and a count that disagrees with it.
`check_anomalies.py` reports OK on an empty registry and on a registry where
nothing at all could be verified, despite a comment claiming the opposite.
MADD-ANO-001 is `resolved` with no regression test, and I reverted its
resolution with every gate and all 200 compliance tests green.

What is sound: all 15 `safety_relevance_rationale` fields are substantive; all
44 `verification:` test ids resolve and the 135 tests they name pass; the SOUP
equality gate caught all 5 mutations; 12 of 13 benchmark acceptance criteria
match what their tests assert; the corrected MADD-ANO-007/008 figures reproduce
to the digit.

Blocks a tag: F1, F2, F3, F4, F10, F11, F12 (a wrong number in the registry or
the release notes is what gets quoted onward, which is how this release got
here). F5, F6, F7, F8, F9, F13 should be fixed but do not falsify a shipped claim.

---

# Findings

## HIGH — F1. MADD-ANO-004 says versions 0.3.0 and 0.3.1 are affected; 0.2.0 and 0.2.1 are affected too

**What breaks:** `docs/validation/known_anomalies.yaml` MADD-ANO-004 carries
`affected_versions: "0.3.0, 0.3.1"` — the only *closed list* in the registry, so
it positively asserts that no other released version is affected. The SOUP
package publishes it verbatim (`docs/validation/soup_package.md`, row
`MADD-ANO-004 | ... | resolved (in 0.4.0) | 0.3.0, 0.3.1`). A downstream
manufacturer fielding 0.2.1 reads that row and concludes they are not impacted.

**Evidence:** `repro/compdocs_ano004_versions.py` drives the mechanism the entry
itself describes (`PUT /graph/params/{node}` writes `node.params[key] = value`;
`ShardedPointwiseNode.__init__` does `super().__init__(..., **node.params)` and
`update()` delegates to `self._inner.update`, so the write lands on a copy
nothing reads) against a `git archive` of each tag:

```
=== v0.2.0 ===
  wrapper position after 5 steps, k=10   : 0.014767
  wrapper position after params['k']=1000: 0.014767
  a genuinely k=1000 node                : 1.161203
  VERDICT: param write IGNORED (defect present)
=== v0.2.1 ===   (identical three lines)
=== v0.3.0 ===   (identical three lines)
=== v0.3.1 ===   (identical three lines)
```

The REST endpoint that makes this reachable also exists at v0.2.0:
`git show v0.2.0:src/maddening/api/server.py` line 361,
`@app.put("/graph/params/{node_name}")`, whose body is
`node.params[key] = value` + `return {"status": "ok", ...}`.

**Why it happens:** `git show v0.2.0:src/maddening/cloud/multigpu/sharded_node.py`
line 42 is byte-identical in the relevant respect to v0.3.1's line 79:
`super().__init__(name=node.name, timestep=node.delta_t, **node.params)`, with
`def update(...): return self._inner.update(state, boundary_inputs, dt)`.
`ShardedPointwiseNode` first appears at v0.2.0.

**What would make this a non-issue:** (a) the wrapper not existing before 0.3.0 —
it does, from v0.2.0; (b) the class not being public before 0.3.0 — neither
v0.2.1 nor v0.3.1 exports it from `maddening/cloud/multigpu/__init__.py`, so
0.3.x has no more public standing than 0.2.x; (c) the param contract
(`accepts_params`, `params_pytree`, `param_specs`) not existing at 0.3.x either —
correct, `git grep -l "def accepts_params" v0.3.1 -- src/` is empty, so *most* of
the entry's described mechanism is 0.4.0-development machinery that never
shipped. The one part that did ship — the REST write answering 200 and changing
nothing — shipped in four releases, not two. Either way the field is wrong.
The only defensible thing that changed at 0.3.0 is the `@stability(STABLE)`
decorator, which is `stable_since`, not `affected_versions`.

**Suggested fix:** `affected_versions: ">=0.2.0, <0.4.0"`, and split the entry's
description so the shipped defect (REST write ignored, 0.2.0–0.3.1) is separated
from the 0.4.0-development contract defects (`accepts_params` etc.) that no
release carried. Risk: none; it widens a disclosure.

---

## HIGH — F2. MADD-ANO-002's re-verification paragraph, the evidence that the defect is live, asserts three things 0.4.0 refutes

**What breaks:** the entry reads

> Re-verified at 0.4.0.dev0 and still live: on a 20-cell rod of length 1.0 with
> thermal_diffusivity=0.01 and a half-sine initial profile, a Fourier number
> Fo = dt*alpha/dx^2 of 5.0 gives max|T| = nan after 60 steps and emits no
> warning of any kind, while Fo = 0.4 and Fo = 0.5 decay normally (0.516 and
> 0.437). **The constructor accepts the unstable timestep without comment.**

The constructor refuses it. And the two decay figures do not reproduce under any
reading of the initial condition.

**Evidence:** `repro/compdocs_ano002.py`, exactly the entry's configuration:

```
--- Fo=0.4  dt=0.10000000000000002
    constructed ok; warnings=[]
    max|T| after 60 steps = 0.675313
--- Fo=0.5  dt=0.12500000000000003
    constructed ok; warnings=[]
    max|T| after 60 steps = 0.657864
--- Fo=5.0  dt=1.2500000000000002
    CONSTRUCTOR RAISED ValueError: timestep 1.2500000000000002 is unstable for
    this rod: the Fourier number dt*alpha/dx^2 is 5, above the 0.5 limit of the
    order-2 stencil (dx = length/n_cells = 0.05, alpha = 0.01). ...
```

`repro/compdocs_ano002b.py` tries the other reading (zero Dirichlet data
supplied) and still does not reproduce the recorded figures:

```
Fo=0.4   no bc        max|T| after 60 update() steps = 0.675313     warnings=0
Fo=0.4   dirichlet 0  max|T| after 60 update() steps = 0.550473     warnings=0
Fo=0.5   no bc        max|T| after 60 update() steps = 0.657864     warnings=0
Fo=0.5   dirichlet 0  max|T| after 60 update() steps = 0.474083     warnings=0
Fo=5.0   no bc        max|T| after 60 update() steps = nan          warnings=0
Fo=5.0   dirichlet 0  max|T| after 60 update() steps = nan          warnings=0
```

**Why it happens:** provenance, not opinion. The paragraph was written in
`a264130` "docs(validation): correct the version fields of the open anomalies",
2026-09-19 00:46. The constructor check and the boundary-closure change landed in
`ad06a8a` "fix(heat): impose Dirichlet data at the rod ends and fix the
4th-order ghosts", 2026-09-20 02:50. `git merge-base --is-ancestor ad06a8a a264130`
→ NO, and `git show a264130:src/maddening/nodes/heat.py | grep -c MAX_FOURIER_NUMBER`
→ 0. The registry was not revisited after the fix. The old decay figures are the
pre-fix values, taken when supplying no boundary data froze the end cells;
under the ghost-cell closure they now evolve, which is `MADD-ANO-007`'s own
recorded consequence.

**What would make this a non-issue:** the anomaly being wrongly left open — it is
not. The `update()` path is genuinely still live and silent (`Fo=5.0 ... nan,
warnings=0` above), and `MADD-ANO-009`'s `residual_risk` states the real
remaining gap: the check sees only the constructor's `timestep`,
`thermal_diffusivity` and `length`, so a `dt` handed to `update()` and a
calibration-moved `thermal_diffusivity`/`length` bypass it. That gap is recorded
in a *different, resolved* entry; MADD-ANO-002, the open one, does not state it.

**Also LOW, same cause:** MADD-ANO-009's description says "Measured at
0.4.0.dev0 ... Fo = 0.37 converges normally, Fo = 0.40 -- inside the documented
limit -- diverges". `repro/compdocs_ano009.py`: both constructions are now
refused (`MAX_FOURIER_NUMBER = {2: 0.5, 4: 0.3125}`). ANO-009 is `resolved` and
its `residual_risk` gives the reader the full picture, so this is a tense
problem rather than a misleading one.

**Suggested fix:** rewrite MADD-ANO-002's evidence paragraph against the
constructor check: say what the constructor now refuses, re-measure the decay
figures, and move MADD-ANO-009's "what the check does not cover" paragraph into
MADD-ANO-002, which is the entry that stays open because of it. Risk: none.

---

## HIGH — F3. MADD-ANO-005 instructs the reader to use two `coupling_diagnostics()` keys that 0.4.0 removed from the report

**What breaks:** MADD-ANO-005's description ("coupling_diagnostics() reports
error_estimate, amplification, **bound_valid** and **gradient_error_bound**
beside the residual"), its workaround ("read
`coupling_diagnostics()['bound_valid']` to see whether it did, and treat
`bound_valid=False` as the old behaviour") and its `residual_risk`
("`coupling_diagnostics()['bound_valid']` is False and `gradient_error_bound`
is +inf") all name keys this release renamed and deprecated. The workaround is
the operative instruction a downstream user follows.

**Evidence:** `repro/compdocs_ano005_keys.py`, on the suite's own `rho = 0.25`
contracting fixture:

```
keys actually reported: ['amplification', 'converged', 'error_estimate',
                         'gradient_error_estimate', 'iterations',
                         'ratio_usable', 'residual']
  'bound_valid' in keys(): False
  d['bound_valid'] -> True   warnings: ["DeprecationWarning:
      coupling_diagnostics()['bound_valid'] is deprecated: the flag checks one
      of the four condi..."]
  'gradient_error_bound' in keys(): False
  d['gradient_error_bound'] -> 8.63167441123025e-05   warnings:
      ["DeprecationWarning: ... is deprecated: it is numerically 'error_est..."]
```

**Why it happens:** `src/maddening/core/graph_manager.py:1024-1025`
`_DIAGNOSTICS_RENAMES = {"bound_valid": "ratio_usable", "gradient_error_bound":
"gradient_error_estimate"}`; `_CouplingDiagnostics` deliberately keeps the old
names out of `keys()`, iteration and `len()` "so a recorded artefact should not
preserve a name the next release deletes", and `CHANGELOG.md` "Deprecated" says
they are removed in 0.5.0. The registry is precisely the recorded artefact the
docstring is talking about.

**What would make this a non-issue:** the aliases living past 0.5.0 — the
CHANGELOG says they do not. Note the irony that makes this worth fixing rather
than ignoring: the *reason* for the rename ("the flag checks one of the four
conditions the estimate rests on, not that it bounds the error") is the same
caveat MADD-ANO-005 exists to record, and the entry's last line
("`bound_valid` reports a usable *ratio*, not a valid *bound*") says it in the
deleted vocabulary.

**Suggested fix:** s/bound_valid/ratio_usable/ and
s/gradient_error_bound/gradient_error_estimate/ throughout MADD-ANO-005, and
regenerate the SOUP table. Risk: none.

---

## HIGH — F4. The release notes' known-anomalies section lists 2 of the 8 anomalies that are open or partially resolved, and the breaking-changes table still has the duplicated coupling row and a count that disagrees with it

Delegated verification; I have reviewed the evidence and reproduced the registry
side independently. Both defects the brief described as previously found are
**still present**. Details, with commands and output, in the sub-report
appended as `SUBREPORT_release_notes.md`. Headlines:

* `docs/release_notes/v0.4.0.md:1939-1961` — the section named "Known anomalies"
  mentions `MADD-ANO-003`, `-006`, `-010` only. The registry has 8 entries that
  are `open` or `partially_resolved`: 002, 003, 005, 010, 011, 012, 013, 014.
  Six are omitted. (The brief said five; it is six now.) All six *are* discussed
  correctly in the narrative body, so this is an enumeration failure, not a
  misstatement — but a reader who goes to the section named "Known anomalies" to
  learn what is still open sees 2 of 8.
* `docs/release_notes/v0.4.0.md:2223` states "nine things change behaviour"; the
  table at 2226 has **11** data rows and **10** distinct ones. Row 2238 is a
  *stale* duplicate of row 2233 — it silently drops the `atol` default change
  (`git show v0.3.1:src/maddening/core/coupling/group.py:132` `atol: float = 1e-8`
  vs `src/maddening/core/coupling/group.py:239` `atol: float = 0.0`), so a reader
  who reads the last row misses that the noise-floor default moved.

**Blocks a tag.** These are the two items the brief named as previously found;
neither fix landed.

---

## HIGH — F5. The release notes assert that `POST /graph/nodes` status codes are unchanged; five classes of input flipped 201 to 400

`docs/release_notes/v0.4.0.md:1749-1752`: "Status codes on `POST /graph/nodes`
are unchanged for every input that already had one". Measured against
`git show v0.3.1:src/maddening/api/server.py` (no finiteness check, no
`initial_state()` call, no dry-run, no size cap; `gm.add_node` ValueError → 409):
a non-finite params leaf, a node whose `initial_state()` raises, a node above
`MAX_NODE_STATE_ELEMENTS`, a node failing the new dry-run trace, and every
`HeatNode` above its Fourier limit now return 400 where they returned 201; a
non-duplicate `ValueError` out of `add_node` returns 400 where it returned 409.
This change is also **not listed** in the breaking-changes table. Full evidence
in `SUBREPORT_release_notes.md` (F3/F4 there).

Also not listed, and genuinely breaking: the **three ZMQ bind defaults** moving
to loopback (`src/maddening/viz/network.py:95,302`,
`src/maddening/cloud/multigpu/coordinator.py:105`, vs `tcp://*:5555`,
`tcp://*:5556`, `tcp://0.0.0.0:{port}` at v0.3.1). `FitResult` keyword-only and
the `crb` rank-cutoff change are correctly absent — `maddening.sysid` does not
exist at v0.3.1, so nothing downstream can break.

---

## MEDIUM — F6. `check_anomalies.py` reports OK on an empty registry and on a registry where nothing could be verified, while its own comment claims it cannot

**What breaks:** the R2 brief's property 1, both cases, in the gate that guards
the registry of record.

**Evidence:** `repro/compdocs_mutate_anomalies.py` (16 mutations against copies
in scratch; the shipped file was never modified):

```
MISSED  M1_empty_registry: OK: anomaly registry at .../M1_empty_registry.yaml is valid
CAUGHT  M2_bogus_component: ERROR: MADD-ANO-002: affected_components entry
        'maddening.nodes.heat.NoSuchThing' does not resolve
CAUGHT  M3_bogus_test_id / M4_bogus_test_file / M5_duplicate_id
MISSED  M6_resolved_no_version
MISSED  M7_resolved_no_verif
MISSED  M8_bogus_versions          (affected_versions: "banana" accepted)
CAUGHT  M9_empty_rationale
MISSED  M10_rationale_is_title
CAUGHT  M11_bogus_status / M12_bogus_severity
MISSED  M13_version_drift          (caught by generate_soup_tables --check instead)
MISSED  M14_no_components
CAUGHT  M15_wrong_prefix / M16_safety_relevant_no_rationale
```

and the sharper one — 15 anomalies present, every `affected_components` and
`verification` stripped, so the gate can verify nothing:

```
M17 (15 anomalies, zero resolvable references): rc= 0
    OK: anomaly registry at .../M17_nothing_verifiable.yaml is valid
```

Score: 8 caught / 8 missed.

**Why it happens:** `scripts/check_anomalies.py:95-107`. The zero-scope guard is
inside `if notes and not args.no_resolve:`, and `notes` is populated only when a
reference was *skipped as unavailable*. It is empty both when everything
resolved and when there was nothing to resolve, so `_components_checked()` is
never called in the case it exists for. The comment two lines above reads
"...and if *nothing* could be checked, the run proves nothing. Same guard as
`check_transforms.py`'s: a gate that verified zero references must not report
OK." It is not the same guard.

Secondary: the summary line carries no count at all
(`OK: anomaly registry at ... is valid`), so CI records no coverage number. That
is safer than overstating one, but it means nothing about this gate is quotable
as evidence.

**What would make this a non-issue:** another gate catching an emptied registry.
`generate_soup_tables.py --check` would fail on the *table* mismatch, but only
until someone regenerates — see F7, where exactly that happens.

**Suggested fix:** hoist the `_components_checked()` call out of `if notes`, add
a `len(anomalies) == 0` failure, and print "N anomalies; M references verified;
K not checked" as two numbers the way `check_heat_stability.py` does. Risk: a
contributor without the `usd` extra sees a NOTE-driven number; that is already
the case.

---

## MEDIUM — F7. MADD-ANO-001 is `resolved` with no verification entry, and its resolution reverts with every gate green

**What breaks:** MADD-ANO-001 is the only `resolved` entry with no `verification:`
key. Its resolution is stated as "the dependency floor moving past the affected
jaxlib rather than by any change to MADDENING's own code". Nothing pins the floor.

**Evidence:** setting `jax>=0.4,<0.6` / `jaxlib>=0.4,<0.6` in `pyproject.toml`
(base and the `cuda12` extra, so `tests/test_import_guards.py` stays happy) —
the exact condition the entry says permitted jaxlib 0.5.1:

```
### (a) no regeneration
  generate_soup_tables --check rc=1
  check_anomalies rc=0
### (b) regenerate then check
  generate_soup_tables --check rc=0
  check_anomalies rc=0
  what the regenerated SOUP now advertises:
20:| Base Dependencies | jax>=0.4,<0.6, jaxlib>=0.4,<0.6, lineax>=0.0.7, ... |
24:| JAX | pinned to `0.10.2` in CI; `jax>=0.4,<0.6` supported |
```

and with the docs regenerated, `pytest tests/compliance/ tests/test_import_guards.py`
→ **200 passed, 1 skipped**. `pyproject.toml` and `docs/` were restored with
`git checkout --` and `git status --porcelain` verified clean afterwards.

Note the second half of that output: the regenerated table says CI pins `0.10.2`
while the supported range is `>=0.4,<0.6` — a row that contradicts itself — and
`generate_soup_tables --check` reports OK, because it checks *document equals
source* and never *source is consistent with source*. `render_test_suite`
(`scripts/generate_soup_tables.py:369-386`) reads the pin and the range and never
compares them.

**What would make this a non-issue:** MADD-ANO-001 being unverifiable in CI
(true: no GPU runner, and the entry honestly records that "the original
configuration cannot be reconstructed under the current floor"). But the floor
itself is trivially assertable, and it is the whole of the resolution.

**Suggested fix:** add `verification:` pointing at a test that asserts the base
`jax`/`jaxlib` floor is `>= 0.10`, and make `generate_soup_tables` fail when the
CI pin falls outside the declared range. Risk: the test has to be updated with
every floor bump, which is the point.

---

## MEDIUM — F8. MADD-ANO-015's `affected_versions: ">=0.1.0"` claims 0.4.0 is affected by a critical, safety-relevant defect its own description says ended at 0.3.1

**What breaks:** the SOUP table publishes
`MADD-ANO-015 | ... | critical | safety_relevant | resolved (in 0.4.0) | >=0.1.0`.
Read plainly that row says 0.4.0 both has and does not have the defect. The
entry's own first sentence says "From v0.1.0 to v0.3.1".

**Evidence:** `repro/compdocs_ano015_defaults.py` at HEAD —

```
NetworkRelay       address default = 'tcp://127.0.0.1:5555'
NetworkReceiver    address default = 'tcp://localhost:5555'
CommandPublisher   address default = 'tcp://127.0.0.1:5556'
CommandReceiver    address default = 'tcp://localhost:5556'
Coordinator params: {... 'port': 5580, 'bind_host': '127.0.0.1', 'secure': None, 'token': None}
```

and all six cited security tests pass (`logs/compdocs_anoverif.log`).

**Why it happens:** `generate_soup_tables.py:418-450` has
`_check_unresolved_anomalies_are_open_ended` — a check that an `open` entry must
*not* close its range — and no converse check that a `resolved` entry must close
it. MADD-ANO-007/008/009 use `">=0.1.0, <0.4.0"`; 006 and 015 do not.

**What would make this a non-issue:** an explicit convention that
`affected_versions` means "first affected version onward, see
`resolution_version` for the end". MADD-ANO-006 argues for exactly that in its
description ("The range is open-ended because the defect was reachable on every
version up to the fix"). But three resolved entries use the closed form, so
there is no convention, and the SOUP table shows the field without the
description that explains it.

**Suggested fix:** `">=0.1.0, <0.4.0"` on 015 and 006, and extend the generator's
check to require a closed range whenever `resolution_version` is set. Risk: none.

---

## MEDIUM — F9. The SOUP known-anomalies headline counts 6 open and omits the 2 partially resolved, which both have a live residual defect

`docs/validation/soup_package.md`: "*15 anomalies registered, **6 open**.*"
`scripts/generate_soup_tables.py:331-333` counts only
`resolution_status == "open"`. MADD-ANO-005 and MADD-ANO-014 are
`partially_resolved` and both carry a `residual_risk` describing behaviour
present in 0.4.0 — ANO-014's in so many words: "What is resolved in 0.4.0 is the
claim, not the behaviour ... the degraded path is still the default and still
silent". Eight anomalies have a reachable defect; the headline of the document a
downstream manufacturer reads first says six. This is the count-vs-coverage
class the R2 brief names, in the sentence most likely to be quoted.

Suggested fix: "*15 registered; 6 open, 2 partially resolved, 7 resolved.*"

---

## MEDIUM — F10. MADD-VER-003's published acceptance criterion is narrower than the band the test enforces

`docs/validation/framework_verification.md` and the `@verification_benchmark`
decorator at `tests/cloud/multigpu/test_lbm_poiseuille.py:71` both state
"centreline velocity within **+/-25%** of u_max = F R^2 / (4 mu)". Line 179 of
the same file asserts `0.75 < ratio < 1.30` — minus 25 per cent, plus **30**.
Measured value is ~1.13, so a regression to 1.27 passes the executed band and
violates the published criterion. The band came in with `e2f93e8`; the criteria
string was written later, in `acde157` "docs(validation): generate the SOUP
summary tables from their sources", and got the number wrong. The registry is
described on the same page as "the acceptance criteria below are the ones
actually asserted"; here they are not.

Every other acceptance criterion I checked matches its assertion (see "checked
and sound").

---

## MEDIUM — F11. The published acceptance band `[-0.25, +1.0]` on eight benchmarks is justified by three measurements that no longer reproduce, one of which the registry itself records as wrong

`docs/validation/framework_verification.md` publishes "within [-0.25, +1.0] of
the declared order" as the acceptance criterion for MADD-VER-005 through -012.
The band comes from `DEFAULT_ORDER_SHORTFALL = 0.25` and
`DEFAULT_ORDER_EXCESS = 1.0` in `src/maddening/testing/mms.py:326-351`, whose
docstrings are the only stated justification for either number.

**Evidence:** `repro/compdocs_mms_bands.py`, re-running the node's own ladders:

```
HeatNode stencil_order=2, rod-end BC, Fo=0.4 (MADD-VER-005 ladder):
    pair   10->20   observed order = 2.023
    pair   20->40   observed order = 2.006
    pair   40->80   observed order = 2.001
    pair   80->160  observed order = 2.000
HeatNode stencil_order=4, Fo=0.3:
    pair   10->20   observed order = 3.760
    pair   20->40   observed order = 3.831
    pair   40->80   observed order = 3.913
    pair   80->160  observed order = 3.957
```

* SHORTFALL cites "HeatNode 1.982 against 2" for the finest pair and "the
  *coarsest* pair of the same ladders sits as much as 0.16 low (1.847) ... 0.25
  clears that coarse-grid wobble". Measured: finest 2.000, coarsest **2.023**.
  No pair on this ladder is low at all; the wobble the constant is sized for is
  now on the *other* side of the declared order, which is the EXCESS constant's
  territory. 1.982 and 1.847 are the pre-MADD-ANO-007 cell-centre readings.
* EXCESS cites "the corrected fourth-order stencil measures **5.02** over one
  pair of a fourth-order ladder". Nothing measures 5.02; the maximum over the
  whole ladder is 3.957. **The registry already knows this**: MADD-ANO-008's
  `residual_risk` says "the oracle experiment ... restores observed orders of
  3.759/3.898/3.954 ... not the 3.775/**5.021**/4.792 recorded", i.e. the
  figure the live constant is justified by is one the registry records as having
  failed re-derivation.

The numbers 0.25 and 1.0 may still be the right numbers. The point is that the
document publishing them says they were "Measured, not chosen by intuition", and
the measurements cited are gone. This is the same failure the release has already
had twice — a band whose stated justification no longer holds is one edit away
from being widened again with nobody able to say why it was 0.25.

---

## MEDIUM — F12. `CHANGELOG.md` states MADD-ANO-005 as "open" with the pre-0.4.0 workaround, contradicting the registry, the SOUP table and two of its own sections

`CHANGELOG.md:489-491`:

> `- MADD-ANO-005: `converged=True` is a residual test, not a bound on the
>   distance to the fixed point -- calibrate it by re-solving at a 100x tighter
>   tolerance (minor, **open**, context_dependent)`

The registry says `partially_resolved`; the SOUP table says
`partially_resolved (in 0.4.0)`. The same file says so twice: line 30, "MADDENING's
own `known_anomalies.yaml` has used `partially_resolved` since MADD-ANO-005 was
written", and under Changed, "**`converged=True` means 'within `tolerance` of the
fixed point'**, not 'the last step was small'". The prescribed workaround
(re-solve at 100x tighter tolerance) is the one the registry marks "On 0.3.x and
earlier".

Two more in the same block:

* `CHANGELOG.md:479` — "MADD-ANO-**011/011**/012 (BallNode, HeartPumpNode, all
  open)". The sentence then describes three defects, the third being
  "`backpressure` is truncated to float32", which is **MADD-ANO-013**. 011 is
  named twice and 013 never.
* MADD-ANO-007/008/009 is listed **twice** in the Known Anomalies block
  (lines 469 and 483), once with the ANO-008 re-derivation correction and once
  without.

Suggested fix: correct the status, the ids, and drop one of the duplicate
007/008/009 entries. Risk: the file's house rule forbids reordering existing
lines; these are corrections in place, which is what the 3440-count line already
did precedent for.

---

## MEDIUM — F13. No compliance artefact records the version of any SOUP dependency the verification evidence was generated against

`docs/validation/soup_package.md` §1 lists Base Dependencies as version
*ranges*: `jax>=0.10,<0.13, jaxlib>=0.10,<0.13, lineax>=0.0.7, numpy>=1.24,
pyyaml>=6.0`. §8 says the transitive tree "is in `pyproject.toml` itself". CI
hard-pins only jax and jaxlib (`pip install "jax==0.10.2" "jaxlib==0.10.2"`)
and floats everything else — `lineax`, `numpy`, `pyyaml` and the whole `[ci]`
extra — from ranges resolved fresh on every run.

Two consequences:

1. `docs/validation/framework_verification.md` asserts "`jax>=0.10,<0.13`
   **supported**" on evidence generated at exactly one point in that range. No
   job in `.github/workflows/ci.yml` exercises 0.11 or 0.12. The word carrying
   the weight is "supported"; nothing tests the range.
2. There is no record anywhere of the `lineax`/`numpy`/`pyyaml` versions any
   verification run used, so the evidence cannot be reproduced and the SOUP
   items are not identified by version — which is what IEC 62304 §5.3.3 and
   §8.1.2 ask of a SOUP item, and what
   `docs/regulatory/iec62304_mapping.md` presents this package as supporting.

This is a documentation-of-record gap rather than a falsified sentence, except
for "supported". Suggested fix: have `generate_soup_tables` emit, or CI upload,
a resolved `pip freeze` for the run that produced the verification evidence, and
soften "supported" to "declared compatible; verified at 0.10.2".

---

## LOW — findings that are real but do not change a decision

L1 through L5 come from the delegated algorithm-guide audit; full evidence,
commands and output in `SUBREPORT_algorithm_guides.md`, with the lead auditor's
correction to L4 marked there.

**L1. `docs/algorithm_guide/nodes/heat_node.md:45` states a rejected boundary
closure is a whole order worse than it is.** The guide says "a linear or
quadratic extrapolation caps the scheme at order 1 or 3 (measured 0.95 and
2.97)". Measured with the shipped ghost builder monkeypatched to a degree-1
Lagrange extrapolation on the same profile and ladder: **2.000**, not 0.95. The
0.954 is MADD-ANO-008's measurement of the *mispositioned* ghosts, not of a
linear closure. Every other copy in the tree is right —
`src/maddening/nodes/heat.py:127-129`, MADD-ANO-008's `residual_risk`, and
`tests/verification/test_mms_order.py:135-137` all say 2.00 — so the guide is the
only artefact carrying the stale number, and it is the artefact the IEC 62304
mapping names as the Clause 5.4 detailed-design record. (Delegated; the degree-3
arm of the same harness reproduces the shipped 3.760/3.831/3.913/3.957 exactly,
which validates it, and I independently measured the same four numbers.)

**L2. `docs/algorithm_guide/solvers/explicit_integrators.md:157-158` asserts an
error agreement its own Known Limitations section denies.** "all three methods
measure order ~1.02 with errors **agreeing to three significant figures**". At
dt = 0.0047 the measured errors are euler 4.567e-3, heun 7.038e-3, rk4 7.034e-3 —
euler is 1.54x away, and heun and rk4 differ in the third figure. Line 128-130 of
the same file says the opposite and is the correct one ("`rk4_step` can be less
accurate than `euler_step` ... measured 1.5x worse at dt = 0.0047"), matching
MADD-ANO-014. The conclusion the sentence supports (an order ladder cannot
distinguish two first-order schemes) survives; the evidence offered for it does
not. Propagated from `tests/verification/test_integrator_order.py:26-28`.

**L3. `docs/algorithm_guide/uq/index.md:9` names `maddening.core.uq`, which does
not exist** (`ModuleNotFoundError`). The real path is
`maddening.core.compliance.uq` (`src/maddening/core/compliance/uq.py:26,56`).
It survives because `scripts/check_impl_mapping.py` only reads rows under an
`## Implementation Mapping` heading, and three guides — `uq/index.md`,
`coupling/interface_mapping.md`, `coupling/unit_transforms.md` — have no such
section and no `MIN_MAPPINGS` pin, so **no gate checks a single symbol claim in
them**. Hand-resolution of every backticked symbol in all three found this as
the only bad one.

**L4. `docs/algorithm_guide/nodes/adaptive_node.md:301` cites design evidence
that is not in the package.** It sources every constant in its Validated
Physical Regimes table to `plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md`.
That path does not exist on any ref of this repository
(`git log --all -- plans/...` is empty, `plans/` has never been tracked).
**Correction to the delegated finding:** the document does exist, at
`/home/nick/MSF/msf/plans/MADDENING_ADAPTIVE_NODE_SPIKE_FINDINGS.md` — the
maintainer's workspace, one directory above the repo — with substantive content
(40 references to spike rounds). So this is design evidence that ships nowhere
and is unreachable for any recipient of the release, not a fabricated citation.
`docs/developer_guide/adaptive_node.md:39` cites it too. The constants
themselves are correct against the class attributes.

**L5. `scripts/check_citations.py` reports 50 citations verified; it verified
45.** `main()` prints `len(citations)`, the raw scan count, but the loop at line
165-166 `continue`s past the five `_TEMPLATE_CITATIONS` entries (the `[@Key]`
teaching examples in `docs/developer_guide/documentation_standards.md` and
`node_authoring.md`) before testing them against the bibliography. No real
citation is dangling, so this is count honesty only — but it is the fifth
instance this release of a gate's summary line overstating what it checked, and
the adjacent "18 unique keys" *is* honest, which makes the mismatch easy to miss.

**L6. `docs/validation/soup_package.md` §2 "Capabilities NOT Provided: Input
validation or sanitization (assumes trusted inputs)" is contradicted by what
0.4.0 ships**: `POST /graph/nodes` returns 400 naming the offending parameter for
a non-finite constant, `PUT /graph/params` validates dtype, shape, finiteness and
`ParamSpec` bounds before writing, the FMU sidecar caps archive member sizes, and
mapping assets are opened `O_NOFOLLOW`. The error direction is conservative (the
document under-claims, so a manufacturer over-allocates validation to itself),
which is why this is LOW.

---

# Unverified suspicions

* `docs/regulatory/iec62304_mapping.md` Clause 5.6 says "`tests/` with 500+
  tests". Collection at the audited tip is 4322. "500+" is true and useless; I
  did not treat it as a finding because it is not false.
* `framework_verification.md` gives the reason GPU tests are absent from CI as
  "MADD-ANO-001", which is now `resolved`. The real reason is that there is no
  GPU runner. Not falsifiable as written; worth rewording.
* MADD-VER-013's stated measurements (R = 0.846, order 1.49, GCI 10.7%) are
  descriptive rather than asserted, and I did not re-measure them — the LBM pipe
  ladder is expensive and five other agents share this box. The test's
  assertions (monotone, finite order, GCI < 25% at the cautious safety factor)
  match the criteria text.
* MADD-ANO-014's `verification:` entry is a bare file path
  (`tests/verification/test_integrator_order.py`) with no test id, the only such
  entry in the registry. The gate accepts it. For a `partially_resolved` entry
  whose resolution is "the claim, not the behaviour", a file-level pointer does
  not name what would fail. Not a false claim.

---

# What I checked and found sound

**The registry's content, where it is right.**

* All 15 `safety_relevance_rationale` fields are substantive, not restatements
  of the title: 60 to 155 words, 1% to 27% title-word reuse, every one naming a
  failure mode, a consequence and the downstream condition the relevance depends
  on. Scored in `repro/compdocs_rationale.py`.
* All 44 `verification:` entries resolve: every file exists, and every
  `file::test` id matches a collected node (261 collected across the 13 files).
* All of them pass. `pytest` on the nine registry-cited files outside
  `tests/verification/`: **135 passed, 2 xfailed** (`logs/compdocs_anoverif.log`);
  both xfails are the documented strict xfails for MADD-ANO-005's residual risk,
  with reason strings that say what flipping them to a pass would mean.
  `pytest tests/verification/ tests/nodes/adaptive/test_verification.py
  tests/compliance/`: **537 passed, 1 deselected, 4 xfailed**
  (`logs/compdocs_verif.log`).
* `affected_versions` checked against the tags: MADD-ANO-001 `<=0.3.1` is right —
  `git show v0.1.0:pyproject.toml` through `v0.3.1` all carry `jax>=0.4,<0.6`,
  which permits the affected jaxlib 0.5.1, and 0.4.0's floor does not.
  MADD-ANO-003 `>=0.4.0` is right — `src/maddening/nodes/adaptive/base.py` is
  absent at v0.3.1. MADD-ANO-007/008/009 `>=0.1.0, <0.4.0` are right.
* The registry's *corrected* figures reproduce to the digit. MADD-ANO-007's
  residual_risk says the rod-end ladder moves to "2.022/2.006/2.001/2.000"; I
  measure 2.023/2.006/2.001/2.000. MADD-ANO-008's says the cubic closure gives
  "3.760/3.831/3.913/3.957"; I measure exactly those four. The entries that were
  re-derived are trustworthy; it is the ones that were not that are stale.
* Every node that declares "Forward Euler" and implements something else has an
  anomaly (BallNode → 011, HeartPumpNode → 012). `RigidBodyNode`,
  `RigidBody2DNode` and `SpringDamperNode` all declare "Semi-implicit Euler",
  which is what they implement, so the registry is complete on that axis.

**Gates I proved sound.**

* `generate_soup_tables.py --check` caught **all five** mutations: a drifted
  `maddening_version`, a tampered "6 open" count, a tampered "13 benchmarks"
  count, a deleted anomaly row, and a widened MADD-VER-002 band in the criteria
  text. It is a genuine equality gate on every generated block. Its limit,
  stated rather than discovered: it proves the tables match the registry, never
  that the registry matches reality — which is the whole of F1 through F3.
* `check_anomalies.py` caught 8 of 16 (listed in F6): a component symbol that
  does not resolve, a test id that does not exist in a file that does, a test
  file that does not exist, a duplicate `anomaly_id`, an empty rationale, an
  invalid `resolution_status`, an invalid `severity`, and a wrong prefix.
* `check_impl_mapping.py`'s arithmetic is honest — the "16 verified of 17" class
  of defect is not present. It reports 58; an independent parse of the Markdown
  finds 56 data rows carrying 58 distinct `maddening.*` symbols (two rows carry
  two each), every one resolves, none is inherited, and the per-guide counts
  match `MIN_MAPPINGS` exactly with zero slack. Four mutations caught
  (renamed symbol, stripped backticks, base-class-only resolution, deleted
  table). Its one hole is guides with no Implementation Mapping section — L3.
* The five compliance gates all pass at HEAD:
  `check_anomalies` OK; `check_impl_mapping` "58 verified";
  `check_citations` "50 verified" (see L5); `check_transforms` "31 references, 6
  transforms"; `check_heat_stability` "132 verified; 103 further constructions
  have computed arguments and were NOT checked" — the only one of the five that
  reports coverage the way the R2 brief asks for.

**Documents that check out.**

* Software identification matches `pyproject.toml` exactly (name, version
  0.4.0.dev0, LGPL-3.0-or-later, Python >=3.11, all five base dependencies,
  hatchling). Mutating the version in the registry is caught.
* "Test packages | 13" matches `tests/` exactly (14 directories minus
  `__pycache__`), and every directory has a scope row.
* 12 of the 13 benchmark acceptance criteria match what their tests assert:
  VER-001 (`< 1e-4` asserted, `1e-4` published), VER-002 (`1.7 < avg_rate < 2.3`
  asserted, `[1.7, 2.3]` published, over the mean of pairwise rates on a 20/40/80
  ladder as described — the defect the brief named is fixed), VER-004 (all four
  clauses asserted, including the deliberately non-monotone one, with a companion
  test pinning why the stronger wording cannot be restored), VER-005 through -012
  (all delegate to `assert_node_order_verified` with the harness defaults the
  criteria quote), VER-013 (monotone, finite order, GCI < 25% at the cautious
  factor, plus a companion test proving the ladder is outside the asymptotic
  range). VER-003 is the exception — F10.
* Order of accuracy agrees across guide, `NodeMeta` and registered benchmark for
  all six node guides, including `HeatNode`'s per-instance hook returning
  `spatial == 4.0` for `stencil_order=4`. The three open node anomalies
  (011/012/013) are correctly *asserted* by the guides, not contradicted.
* `explicit_integrators.md` does carry the MADD-ANO-014 caveat — "every method in
  this module converges at order 1", plus two Known Limitations and three
  Validated-Regime rows — and every quantitative figure in it reproduces except
  L2's sentence. Stability intervals verified analytically (RK4 real boundary at
  z = -2.785293563 against the stated -2.785).
* `heat_node.md`'s stability numbers are exact: building the full discrete
  operator and solving for the sharp forward-Euler bound gives 0.316920 at N=5
  rising monotonically to 0.324851, against the guide's "0.3169 ... rising
  monotonically to 0.3249", with `MAX_FOURIER_NUMBER[4] = 5/16 = 0.3125` below
  all of them. The cubic ghost weights are algebraically exact.
* `unit_transforms.md` and `interface_mapping.md`: every symbol exists, all five
  conversion factors are right, and the three documented `validate()` unit
  outcomes reproduce (two warnings, one clean control).
* Every citation in every algorithm guide resolves; 19 bib entries, no
  duplicates, one unused (`ShanChen1993`, non-blocking warning).
* `docs/regulatory/iec62304_mapping.md` cites only documents that exist
  (`DESIGN.md`, `ROADMAP.md`, `CONTRIBUTING.md`, `CHANGELOG.md`,
  `tests/core/test_integration.py`).
* `CHANGELOG.md`'s claim that "Every anomaly whose defect is still reachable now
  records an open-ended `affected_versions`" is true — all six open and both
  partially-resolved entries use `>=X`. And `[verify]` does now pull only
  `hypothesis`, as Removed claims.
* The API-authentication contradiction the brief flagged is fixed: the Security
  section's last entry states "Since this release the API *does* authenticate on
  a non-loopback bind". One residual phrase, "the unauthenticated API now caps
  `n_steps`..." at line 449, reads oddly beside it but is true of the loopback
  case; wording, not a finding.

---

# Reproducers

All in `repro/`, all read-only except the two that mutate a copy in scratch or
mutate and restore via `git checkout --` (verified clean afterwards with
`git status --porcelain`).

| script | what it shows |
|---|---|
| `compdocs_ano002.py` | MADD-ANO-002: the constructor refuses Fo=5.0 |
| `compdocs_ano002b.py` | MADD-ANO-002: the recorded decay figures do not reproduce; the `update()` path is still live |
| `compdocs_ano004_versions.py` | MADD-ANO-004: the defect reproduces at v0.2.0, v0.2.1, v0.3.0, v0.3.1 |
| `compdocs_ano005_keys.py` | MADD-ANO-005: `bound_valid` / `gradient_error_bound` are gone from `keys()` and warn |
| `compdocs_ano009.py` | MADD-ANO-009: both stated configurations are now refused |
| `compdocs_mms_bands.py` | the MMS band's justification figures (1.982, 1.847, 5.02) against what the ladders measure |
| `compdocs_mutate_anomalies.py` | 16 mutations against `check_anomalies.py`; 8 caught, 8 missed |
| `compdocs_rationale.py` | substance scoring of all 15 `safety_relevance_rationale` fields |

`logs/compdocs_verif.log` — `tests/verification/` + adaptive verification +
`tests/compliance/`, 537 passed.
`logs/compdocs_anoverif.log` — the nine other registry-cited files, 135 passed.
