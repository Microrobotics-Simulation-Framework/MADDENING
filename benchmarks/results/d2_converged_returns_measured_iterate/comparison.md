## Sweep comparison (100 rows of 350)

- `rows_compared`: 100
- `rows_ok`: 100
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 100
- `rows_with_step0_converged_identical`: 100
- `rows_with_window_iterations_identical`: 61
- `rows_with_window_converged_identical`: 100
- `prediction1_single_step_holds`: True
- `rows_at_cap_fraction_1`: 10
- `rows_at_cap_fraction_1_bit_identical`: 10
- `prediction2_at_cap_bit_identical_holds`: True
- `at_cap_violations`: `[]`
- `rows_that_move`: 90
- `rows_bit_identical`: 10
- `rows_with_a_converged_exit`: 90
- `rel_step1_l2_median`: 1.995e-06
- `rel_step1_l2_p90`: 2.751e-05
- `rel_step1_l2_max`: 2.345e-04
- `rel_final_l2_median`: 1.569e-05
- `rel_final_l2_p90`: 2.618e-04
- `rel_final_l2_max`: 1.204e-03
- `worst_step1_row`: `{"fixture": "star-2", "label": "jac/aitken/interface", "rel_l2": 0.0002345065605689583}`
- `worst_final_row`: `{"fixture": "star-2", "label": "jac/aitken/interface", "rel_l2": 0.0012040643332082816}`
- `rows_only_in_before`: `[]`
- `rows_only_in_after`: `[]`

### By acceleration

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `aitken` | 34 | 0 | 8.65e-07 | 2.35e-04 | 6.56e-06 | 1.20e-03 |
| `fixed` | 56 | 0 | 2.28e-06 | 1.01e-04 | 1.87e-05 | 8.44e-04 |

### By convergence norm

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `interface` | 39 | 0 | 1.54e-05 | 2.35e-04 | 9.34e-05 | 1.20e-03 |
| `l2` | 51 | 0 | 8.26e-07 | 6.64e-06 | 7.37e-06 | 5.17e-05 |

### By fixture

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `chain-2` | 10 | 0 | 2.13e-06 | 1.66e-05 | 8.72e-06 | 7.69e-05 |
| `chain-5` | 11 | 0 | 1.16e-06 | 2.29e-05 | 1.51e-05 | 1.37e-04 |
| `mixed-modes` | 6 | 0 | 4.82e-06 | 2.12e-05 | 1.60e-05 | 2.38e-04 |
| `ring-4` | 10 | 0 | 2.06e-06 | 2.40e-05 | 2.17e-05 | 2.60e-04 |
| `ring-8` | 11 | 0 | 1.14e-06 | 4.98e-05 | 2.49e-05 | 2.87e-04 |
| `slow-drift` | 10 | 0 | 5.75e-08 | 3.39e-07 | 1.01e-07 | 1.14e-06 |
| `star-2` | 10 | 0 | 5.49e-06 | 2.35e-04 | 3.44e-05 | 1.20e-03 |
| `star-4` | 11 | 0 | 3.02e-06 | 9.05e-05 | 4.89e-05 | 8.01e-04 |
| `stiff-pair-0.5` | 11 | 0 | 6.19e-07 | 1.69e-05 | 1.36e-05 | 2.31e-04 |

