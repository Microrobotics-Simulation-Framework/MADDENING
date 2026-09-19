# MADDENING 0.4.0 — pre-release independent audit brief

Durable location (not the session scratchpad, which is volatile).

You are one of six independent auditors of the MADDENING 0.4.0 release
candidate.  Read `MADDENING_AGENT_COMMON_BRIEF.md` in this directory first:
its rules on PYTHONPATH, the shared virtualenv, never killing by pattern, and
repository conventions are mandatory and are not repeated here.

## What this audit is

`release/0.4.0` is ~7,000 changed lines of `src/maddening` over `main`,
developed across ~45 pull requests merged in parallel over two days.  Each PR
was reviewed and CI-green *against the base at the time*.  Nothing has audited
the **combined** result.  That is your job.

This release has already produced three defects that no individual branch
caught and only the merged state exposed.  Interactions between separately
reviewed changes are therefore the highest-yield thing you can look for, not
the lowest.

## You are deliberately uninformed

You have not been told what the orchestrator believes, what earlier audits
found, or which areas are thought to be risky.  This is on purpose.  Report
what the code actually does.  If you independently rediscover something already
recorded in `docs/validation/known_anomalies.yaml`, say so and move on — an
independent confirmation is a useful result, not a wasted one.

## Method

1. Read your assigned surface as it now stands, then read
   `git diff origin/main...HEAD -- <your paths>` to see what this release
   changed about it.  Both: the diff tells you what is new, the file tells you
   what is true.
2. For each candidate finding, **build a reproducer that fails** before you
   write it down.  A reasoned argument that something looks wrong is a
   hypothesis; a script that prints the wrong number is a finding.  Findings
   without a reproducer go in a clearly separate "unverified suspicions"
   section.
3. Prefer differential evidence: same input, `main` vs `HEAD`, or two
   configurations that should agree.  "This is wrong" is weaker than "these
   two paths that must agree return 3.0 and 5.0".
4. Actively try to *disprove* each finding before reporting it.  State what
   would have to be true for it to be a non-issue, and check that.

## Severity

- **CRITICAL** — silent wrong numerical results, data loss, or a security hole.
- **HIGH** — a documented guarantee that does not hold; a crash on a supported
  configuration; a gradient that is wrong or silently zero.
- **MEDIUM** — a real defect with a workaround, or a guarantee that holds only
  under conditions the docs do not state.
- **LOW** — naming, docstrings that overstate, missing validation with no
  reachable consequence.

Be honest about severity.  Inflating a LOW to a HIGH costs the release more
attention than it is worth; burying a CRITICAL in a list costs more than that.

## What NOT to do

- Do **not** fix anything.  Report only.  Fixes are dispatched separately, by
  someone who can see all six reports and the interactions between them.
  (Exception: a throwaway reproducer script is expected — put it in your
  report directory, not in `tests/`.)
- Do **not** run the full test suite.  Run only the tests relevant to your
  surface.  Five other agents share this machine, one filesystem and one
  virtualenv.  CI runs the full suite.
- Do not modify anything outside your own worktree and report directory.

## Deliverable

Write `benchmarks/results/audit_040_final/<your-name>/REPORT.md` **in the main
checkout** at `/home/nick/MSF/msf/MADDENING/` (create the directory; it is
untracked scratch until the orchestrator commits it).  Put reproducer scripts
and raw output next to it.

Structure:

```
# Audit: <surface>   (<commit sha audited>)

## Summary
<= 10 lines.  If nothing significant: say so plainly, and say what you
checked so the absence of findings is worth something.

## Findings
### <SEVERITY> — one-line claim
**What breaks:** concrete inputs -> wrong output.
**Evidence:** the command, and its actual output pasted.
**Why it happens:** the mechanism, with file:line.
**What would make this a non-issue:** what you checked to rule that out.
**Suggested fix:** one paragraph, and what it would risk.

## Unverified suspicions
## What I checked and found sound
```

That last section matters.  A surface that was examined and is clean is a
result the release needs recorded.
