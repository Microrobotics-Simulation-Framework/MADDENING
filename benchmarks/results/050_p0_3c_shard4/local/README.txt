Local reproduction of slow-lane shard 4 (P0.3c), 2026-10-14, taskset -c 8-11
(4 CPUs, like a hosted runner), MemoryMax=12G, CPU only, jax 0.11.0.

run.sh / monitor.py: the CI-like pytest command, sampled every 5 s (*.mon:
one JSON line per sample: parent threads, CPU, RSS, busiest threads, children).
  control1   - the control checklist test alone
  sub_alone  - control + 3 seeds, alone (RELEASE_BETWEEN_FILES=1)
  sub_prefix - the 12 files before it on shard 4, then the same 4 tests
  full_k     - tests/ collected, MADDENING_TEST_SHARD=4/10:weighted, -k limited
               to tests/api + multigpu (skypilot excluded), 3 seeds
  sub_cache_exp - sub_alone with a JAX persistent cache shared by the children
probe_*.py: scratch probes (pytest files, renamed so nothing collects them) that run one checklist child through the
  harness (run_pod_support.run_pod, --dry-run asserted), with JAX_LOG_COMPILES
  or extra XLA flags. Not part of the suite.
speed_model.py / collect_speed.py: the runner-speed fit over the 60 CI jobs.
