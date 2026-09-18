## Sweep comparison (150 rows of 350)

- `rows_compared`: 150
- `rows_ok`: 150
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 150
- `rows_with_step0_converged_identical`: 150
- `rows_with_window_iterations_identical`: 98
- `rows_with_window_converged_identical`: 150
- `prediction1_single_step_holds`: True
- `rows_at_cap_fraction_1`: 12
- `rows_at_cap_fraction_1_bit_identical`: 12
- `prediction2_at_cap_bit_identical_holds`: True
- `at_cap_violations`: `[]`
- `rows_that_move`: 138
- `rows_bit_identical`: 12
- `rows_with_a_converged_exit`: 138
- `rel_step1_l2_median`: 1.282e-06
- `rel_step1_l2_p90`: 3.346e-05
- `rel_step1_l2_max`: 2.345e-04
- `rel_final_l2_median`: 1.788e-05
- `rel_final_l2_p90`: 2.961e-04
- `rel_final_l2_max`: 1.633e-03
- `worst_step1_row`: `{"fixture": "star-2", "label": "jac/aitken/interface", "rel_l2": 0.0002345065605689583}`
- `worst_final_row`: `{"fixture": "stiff-pair-0.8", "label": "jac/aitken/interface", "rel_l2": 0.0016328767384159283}`
- `rows_only_in_before`: `[]`
- `rows_only_in_after`: `[]`

### By acceleration

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `aitken` | 42 | 0 | 7.13e-07 | 2.35e-04 | 6.56e-06 | 1.63e-03 |
| `fixed` | 84 | 0 | 2.83e-06 | 1.07e-04 | 2.46e-05 | 8.44e-04 |
| `iqn-ils` | 12 | 0 | 1.06e-07 | 8.23e-07 | 9.68e-07 | 3.12e-05 |

### By convergence norm

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `interface` | 63 | 0 | 1.59e-05 | 2.35e-04 | 1.11e-04 | 1.63e-03 |
| `l2` | 75 | 0 | 5.46e-07 | 6.64e-06 | 6.27e-06 | 7.32e-05 |

### By fixture

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `chain-2` | 13 | 0 | 2.42e-06 | 3.31e-05 | 1.09e-05 | 1.42e-04 |
| `chain-5` | 13 | 0 | 1.16e-06 | 2.29e-05 | 1.51e-05 | 1.37e-04 |
| `mixed-modes` | 7 | 0 | 1.07e-06 | 2.12e-05 | 7.76e-06 | 2.38e-04 |
| `ring-4` | 13 | 0 | 2.10e-06 | 3.20e-05 | 2.46e-05 | 2.60e-04 |
| `ring-8` | 13 | 0 | 1.14e-06 | 4.98e-05 | 3.12e-05 | 2.87e-04 |
| `slow-drift` | 13 | 0 | 7.36e-08 | 3.39e-07 | 1.04e-07 | 1.14e-06 |
| `star-2` | 13 | 0 | 4.34e-06 | 2.35e-04 | 3.68e-05 | 1.20e-03 |
| `star-4` | 13 | 0 | 3.02e-06 | 1.07e-04 | 4.89e-05 | 8.01e-04 |
| `stiff-pair-0.25` | 13 | 0 | 4.03e-07 | 9.55e-06 | 3.73e-06 | 1.64e-04 |
| `stiff-pair-0.5` | 13 | 0 | 6.19e-07 | 1.69e-05 | 1.36e-05 | 3.47e-04 |
| `stiff-pair-0.8` | 13 | 0 | 1.68e-06 | 6.67e-05 | 7.32e-05 | 1.63e-03 |
| `stiff-pair-1.2` | 1 | 0 | 8.23e-07 | 8.23e-07 | 3.44e-07 | 3.44e-07 |

