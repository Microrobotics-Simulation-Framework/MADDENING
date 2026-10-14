# P0.3c progress (slow-lane shard 4)

Worktree `/home/nick/MSF/msf/MADDENING-wt/slow-shard4`, branch `ci/slow-shard4`
(from origin/release/0.4.2 @ 8d2d6bd3; upstream unset on purpose, push with
`git push -u origin ci/slow-shard4`). Slice 8-13, 12G, CPU only.

Scratch data (not committed, 100 MB): `/tmp/claude-1000/-home-nick-MSF-msf-MADDENING/6705ca06-9e31-48a7-911f-0df1e90c341b/scratchpad/shard4/`
(junit/ = shard 4 of each run, junit_all/ = all 60 artifacts of runs
38182068116, 38206585198, 38332673699; scripts junit_files.py,
seeded_cmp.py, lane_cmp.py, speed_model.py).

## Step 1: CI data (done)
- Weights-table run = 38182068116 (commit e69536b3); seeded_faults was on
  SHARD 6 there (older table weighed it 614 s), measured 30.0 / 26.7 min.
- Shard 4 in 38206585198 / 38332673699: seeded 56.6/52.3 and 56.9/48.6 min.
- Per function: every test of the file slowed by the same 1.4-1.9x
  (subprocess-heavy ~1.9, in-process ~1.5-1.7). Uniform, no growth through
  the file, not a few tests.
- 27 tests (control + 26 wrapper seeds) each copy the package and run a
  cold `run_pod.py --goal checklist --dry-run` child: ~92% of the file.
- Runner-speed model (log t = log w_file + log s_runner, robust medians,
  seeded excluded from fit): runner factors are bimodal (fast 0.61-0.87,
  slow 0.95-1.23, ratio ~1.6 = the CPU lottery). Shard 6 of the weights run
  was on two FAST runners (x0.65, x0.61); shard 4 of both later runs on
  four SLOW ones (x1.03-1.13). Seeded residual after the runner factor:
  0.94-1.08 in all six observations => the file is not slow relative to
  its runner; the weight (1798 s) came from a fast runner. Median-runner
  time of the file: 46.9 min.
- Other shard 4 files ran slower than their weights by the same factor.

## Step 1b: a runner-speed proxy from the job logs
- Independent runner-speed proxy: pytest's collection time (session start
  -> "collected" line in the job log; same work on every shard, before any
  test). r = 0.86 with the fitted runner factor over 60 jobs; log factor ~
  1.54 log collect. Weights-run shard 6 runners: collection 26.3 / 28.0 s =
  ranks 1 and 3 (fastest) of 60. Shard 4 in the later runs: 37.4-41.5 s =
  ranks 35-59. Normalising by (collect/median)^1.5 halves the per-file
  spread across runners (sd of log time 0.169 -> 0.087, 146 files).
- Max RSS on shard 4 4.5 GB of 16: memory pressure out.

## Step 2: local reproduction (done; taskset 8-11 = 4 CPUs like CI, 12G)
- Control checklist test alone: 58.7 s (CI fast runners 54-60, slow 102-117).
- Subset (control + 3 seeds) alone 58-67 s each; after the 12 prefix files
  (tests/api x7 + multigpu x5) with RELEASE=1: 58-61 s; after a full
  CI-like collection (tests/, SHARD=4/10:weighted, RELEASE=1, -k limited,
  skypilot excluded): 58-60 s. No context effect.
- Parent after prefix: 53-57 threads (JAX 16 virtual devices, servers), all
  idle: parent CPU +0.0 s per 5 s while children ran. Children ~2.5 of 4
  CPUs busy, 28 threads, ~700 MB.
- Child breakdown (JAX_LOG_COMPILES via the harness, scratch probe):
  XLA compile 45 s of ~60 (jit(loss) value_and_grad 32 s, 61 programs),
  lowering 4 s, tracing ~11 s; goals: stencil 35, coupled 17, hybrid 3.4.
- Cheaper-children probes: shared JAX persistent cache across children:
  control 78 s (+30%), seeds 50-70 s (-15..25%) -- seeds change the
  wrapper's HLO, so the expensive programs miss. --xla_backend_optimization
  _level=0: 52 vs 70 s on a noisy box (-25%), changes the compiled code.
  Neither adopted.

## Decision: fix = runner-speed-normalised weights (done: f5c9cce0, merge 634ea1e3)
- tests/_sharding.py Plugin records runner_collect_seconds / runner_cpu /
  runner_cpus as JUnit testsuite properties.
- scripts/slow_lane_weights.py normalises each report by
  (median collect / report's collect) ** 1.5 before merging.
- Regenerate tests/slow_lane_weights.json from A, C, B with collection
  seconds injected from the job logs (scratch collect_speed.json).
- New table: seeded file 3152 s (was 1798). Old deal in median-runner
  minutes: shard 4 at 99.3 (0.10.2); new deal: worst 74.2 / 74.5.
- Drift script scales its prediction by the report's runner speed.
- Mutants G4, G6 (moved), G8, G9, S26, Z18 (moved), Z21, Z22: all caught
  locally (slow guard test, -k by id). Gates green; compliance files green.
- Merged origin/release/0.4.2 (#347 per-push weights) at 634ea1e3.

## Step 5: PR + one slow-lane dispatch (next)
