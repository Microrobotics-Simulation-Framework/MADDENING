# P0.3b CI follow-ups: progress notes

Worktree /home/nick/MSF/msf/MADDENING-wt/ci-followups, branch ci/followups from origin/release/0.4.2 (f6a70006).
Brief: plans/050_briefs/P0_3b_ci_followups.md. Core slice 5-7, 6G cap, CPU only.

## Plan
1. pin ubuntu-24.04 (4 workflows + generated SOUP doc + tests)
2. scripts/ci_runner_health.py (info, mark, canary, before/after tests) wired into both lanes
3. scripts/ci_variance_report.py + test + doc
4. slow_lane_weights.py: CPU normalisation + median default for slow lane
5. canary + rerun workflow (.github/workflows/rerun-degraded.yml, logic inline, no checkout)
6. measure thresholds from a sampler throwaway branch (ci-trial/...), delete after

## Status
(see below, updated per step)

## Status (updated)
- DONE locally: pin; scripts/ci_runner_health.py (limits measured, thresholds.md); both workflows wired; rerun-degraded.yml;
  scripts/ci_variance_report.py; slow_lane_weights.py normalisation + --latest; docs section; tests (compliance/test_ci_runner_health,
  test_rerun_degraded, test_ci_variance_report, test_weights_cpu_normalisation); 29 guard mutants all caught.
- Sampler branch ci-trial/p3b-sampler (throwaway, DELETE at the end), runs 38502198313/38502608977/38502639034.
- TODO: push ci/followups, open PR to release/0.4.2; throwaway forced-trip branches (release/p3b-trial-degraded for ci.yml, ci-trial/p3b-slow for slow lane
  dispatch narrowed to shard 1 + forced shard 2); real-API dry decide of rerun script on the forced run; quote real log lines in PR + docs;
  merge base in before final push (no rebase); delete throwaway branches.
