# Degraded-runner limits: where they come from (2026-10-14)

Rule (brief P0.3b): well outside every CPU model's normal range, more than 3x
the slowest model's median.

## Data

A throwaway branch (`ci-trial/p3b-sampler`, deleted afterwards) ran
`scripts/ci_runner_health.py machine`, the real install step of the per-push
lane (constraints file, `jax==0.10.2`, `.[ci]`) and `canary`, twice in a row,
in 20 jobs per run on `ubuntu-24.04`. Runs 38502608977 and 38502639034 used the
final canary (NumPy matmul 2200x2200 x4 plus a 200-layer `jax.jit` compile); an
earlier run (38502198313, a canary a quarter the size) was used to size it.

| CPU model | jobs | canary median (s) | canary max (s) | install median (s) | install max (s) |
|---|---|---|---|---|---|
| AMD EPYC 7763 | 18 | 2.83 | 3.09 | 68 | 97 |
| AMD EPYC 9V74 | 16 | 1.98 | 2.75 | 64 | 91 |
| Intel Xeon Platinum 8573C | 3 | 2.00 | 2.03 | 61 | 66 |
| Intel Xeon Platinum 8370C | 1 | 2.04 | 2.04 | 52 | 52 |
| AMD EPYC 9V45 | 2 | 1.43 | 1.45 | 48 | 52 |

The two canaries of one job differ by a median of 2.1 percent (max 7.4).
Slowest model to fastest, the canary spans 2.0x (the test lanes span 1.5x).

Install steps of both lanes, last 50 runs of `ci.yml` and 20 of
`slow-tests.yml` (read with `gh run view --json jobs`, the "Install
dependencies" step): 653 steps, median 62 s, p99 93 s, max 131 s.
The degraded slow-lane runner of 2026-10-13: about 540 s (9 minutes).

## Limits

* canary: 3 x 2.83 = 8.5 s (the healthy maximum seen is 3.09 s);
* install: 3 x 68 = 204 s (the healthy maximum seen over 653 steps is 131 s).

## Collect-only (optional item), measured and not adopted

`pytest --collect-only -q` of the whole tree (a shard still imports every
test module before it deselects) took 34 to 45 s on 6 of 6 sampler jobs
(run 38502198313). That is 3 to 4 percent of a median per-push job on every
job of both lanes, for a signal the install time and the canary already give
for the 2026-10-13 case (366 s to collect came with a 9-minute install). Not
added.

## Redo

Change the runner image or the canary's size, and these numbers move:
re-sample and change `CANARY_LIMIT_S` and `INSTALL_LIMIT_S` in
`scripts/ci_runner_health.py`.
