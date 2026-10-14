# Slow-lane confirmation runs for the median-runner table (PR #352)

Both runs are narrowed to JAX 0.10.2, on throwaway branches (deleted), with the table of this branch (runner_speed reference 36.39 s, exponent 1.5). Columns: the pytest step and the JUnit test minutes; pred, the table's minutes for the files the shard ran; coll, the runner's collection seconds; xspd, (coll / 36.39) ** 1.5; expect = pred * xspd. Per-shard tables from a scratch script, numbers as printed.

## Run 38526100615: before #355, with the temporary pin (commit ec562d46)

```text
lane    sh   step  tests   pred  coll  xspd expect tests/exp tests/pred  watched  cpu
0.10.2   1   82.0   81.2   71.5  39.8  1.14   81.8      0.99       1.14  [seeded 57.6]  AMD EPYC 7763 64-Core Processor success
0.10.2   2   78.9   78.1   71.6  38.7  1.10   78.6      0.99       1.09  []  AMD EPYC 7763 64-Core Processor success
0.10.2   3   80.0   79.3   71.6  37.1  1.03   73.7      1.08       1.11  [ckpt 3.5]  INTEL(R) XEON(R) PLATINUM 8573C failure
0.10.2   4   79.0   78.2   71.6  40.1  1.16   83.0      0.94       1.09  []  AMD EPYC 7763 64-Core Processor success
0.10.2   5   82.7   81.8   71.6  41.4  1.21   86.8      0.94       1.14  []  AMD EPYC 7763 64-Core Processor success
0.10.2   6   83.8   83.0   71.6  40.9  1.19   85.5      0.97       1.16  []  AMD EPYC 7763 64-Core Processor success
0.10.2   7  132.3  131.4   71.6  42.7  1.27   91.0      1.44       1.83  [rest 64.5]  AMD EPYC 9V74 80-Core Processor success
0.10.2   8   81.9   81.1   71.6  38.2  1.08   77.2      1.05       1.13  []  AMD EPYC 7763 64-Core Processor success
0.10.2   9   79.9   79.1   71.6  38.8  1.10   78.9      1.00       1.10  []  AMD EPYC 7763 64-Core Processor success
0.10.2  10   58.2   57.5   71.6  33.9  0.90   64.4      0.89       0.80  []  AMD EPYC 9V74 80-Core Processor success
tests/expected: 0.89 to 1.44
```

Shard 3's failure is guard mutant S15, which the pin had turned into a no-op (fixed, then removed with the pin). Shard 7 still held the REST schema file (64.5 min) although the pin had moved the checkpoint-load file to shard 3: the slowdown came from Hypothesis's LINE events (#355), not from the pair.

## Run 38537609605: after #355, no pin (commit 4e5d7c15)

```text
lane    sh   step  tests   pred  coll  xspd expect tests/exp tests/pred  watched  cpu
0.10.2   1   54.2   53.5   71.5  34.4  0.92   65.7      0.81       0.75  [seeded 36.0]  AMD EPYC 9V74 80-Core Processor success
0.10.2   2   73.7   72.9   71.6  43.9  1.32   94.9      0.77       1.02  []  AMD EPYC 9V74 80-Core Processor success
0.10.2   3   89.1   88.3   71.6  38.0  1.07   76.4      1.16       1.23  []  AMD EPYC 7763 64-Core Processor success
0.10.2   4   84.7   83.8   71.6  41.2  1.20   86.3      0.97       1.17  []  AMD EPYC 7763 64-Core Processor success
0.10.2   5   54.2   53.6   71.6  28.3  0.68   49.0      1.09       0.75  []  Intel(R) Xeon(R) 6973P-C success
0.10.2   6   63.2   62.4   71.6  35.5  0.96   68.9      0.91       0.87  []  AMD EPYC 9V74 80-Core Processor success
0.10.2   7   79.4   78.6   71.6  40.5  1.17   84.0      0.94       1.10  [ckpt 2.8 rest 9.5]  AMD EPYC 7763 64-Core Processor success
0.10.2   8   62.0   61.1   71.6  39.7  1.14   81.5      0.75       0.85  []  AMD EPYC 9V74 80-Core Processor success
0.10.2   9   58.7   58.1   71.6  29.2  0.72   51.4      1.13       0.81  []  Intel(R) Xeon(R) 6973P-C success
0.10.2  10   73.0   72.1   71.6  42.4  1.26   90.1      0.80       1.01  []  AMD EPYC 9V74 80-Core Processor success
tests/expected: 0.75 to 1.16
```

## How much the Hypothesis events inflated the table

Per file, the 0.10.2 lane, every time scaled to the table's median runner; table = the three-run median that lane contributed.

```text
102 files of 60 s or more on the table's 0.10.2 lane
post/table: median 0.91, quartiles 0.81-1.05, min 0.58, max 1.33
post/pre:   median 0.97, quartiles 0.85-1.11
totals (min): table 621, pre 640, post 571  (post/table 0.92)

files the table over-weighs most (post/table), with the pre run beside them:
   0.58  pre/table  0.60  table    335 s  post    194 s  tests/compliance/test_gate_scripts.py
   0.63  pre/table  0.92  table    213 s  post    134 s  tests/property/test_differential_param_writes.py
   0.66  pre/table  0.72  table    246 s  post    163 s  tests/verification/test_verify_mapping.py
   0.69  pre/table  1.09  table    284 s  post    197 s  tests/property/test_sysid_targeted_search.py
   0.70  pre/table  1.00  table    474 s  post    333 s  tests/property/test_differential_coupling_solvers.py
   0.71  pre/table  1.01  table    985 s  post    697 s  tests/core/test_coupling_pairs_of_two_sizes_in_every_domain.py
   0.71  pre/table  0.97  table    193 s  post    137 s  tests/property/test_coupling_invariances.py
   0.71  pre/table  0.96  table   2667 s  post   1905 s  tests/property/test_sysid_truth_recovery.py
   0.71  pre/table  0.79  table    869 s  post    621 s  tests/property/test_differential_coupling_interactions.py
   0.72  pre/table  0.99  table    202 s  post    145 s  tests/api/test_a_checkpoint_load_answers_as_a_params_write_on_every_parameter.py
   0.72  pre/table  0.99  table    168 s  post    122 s  tests/verification/hypothesis/test_hypothesis_sysid.py
   0.72  pre/table  0.96  table    107 s  post     77 s  tests/core/test_step_program_digests.py

files the table under-weighs most:
   1.33  pre/table  1.25  table    119 s  post    158 s  tests/verification/test_builtin_nodes_verified_lbm.py
   1.33  pre/table  1.20  table    790 s  post   1050 s  tests/core/test_the_float_floor_counts_a_geometry_kernels_rounding.py
   1.32  pre/table  1.19  table    214 s  post    282 s  tests/property/test_differential_checkpoint.py
   1.28  pre/table  1.13  table   2282 s  post   2921 s  tests/compliance/test_guard_mutations.py
   1.23  pre/table  1.07  table    192 s  post    236 s  tests/property/test_differential_transforms.py
   1.22  pre/table  1.08  table   1051 s  post   1287 s  tests/property/test_coupling_targeted_search.py
```

Without the REST schema file (5.8 times its weight on shard 7 of the first run), over the other 101 files: table 612 min, before #355 590, after 563. So the events cost the lane about 5% in total (after/before 0.95). The files they inflate most, the Hypothesis-heavy ones, run 25-30% under the table after #355. Two files near the top of that list (`test_gate_scripts.py`, `test_verify_mapping.py`) were already under the table before #355, so not all of the gap is the events: those files changed, or their runs drew differently. The deal absorbs it: every shard of the second run is within 0.75-1.16 of its prediction. The next refresh of the table should take post-#355 runs only.

## Collection time against the canary, as the runner's speed

The 19 shards of both runs (shard 7 of the first left out); tests/prediction against each measure.

```text
19 shards; log-log r with tests/prediction: collection 0.73, canary jit 0.91, canary score 0.86
  AMD EPYC 7763 64-Core    n=10  tests / runner-scaled prediction: median 0.98 (0.94-1.16)
  AMD EPYC 9V74 80-Core    n=6  tests / runner-scaled prediction: median 0.81 (0.75-0.91)
  INTEL(R) XEON(R) PLATI   n=1  tests / runner-scaled prediction: median 1.08 (1.08-1.08)
  Intel(R) Xeon(R) 6973P   n=2  tests / runner-scaled prediction: median 1.11 (1.09-1.13)
```
