## Sweep comparison (200 rows of 350)

- `rows_compared`: 200
- `rows_ok`: 200
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 200
- `rows_with_step0_converged_identical`: 200
- `rows_with_window_iterations_identical`: 130
- `rows_with_window_converged_identical`: 200
- `prediction1_single_step_holds`: True
- `rows_at_cap_fraction_1`: 14
- `rows_at_cap_fraction_1_bit_identical`: 14
- `prediction2_at_cap_bit_identical_holds`: True
- `at_cap_violations`: `[]`
- `rows_that_move`: 186
- `rows_bit_identical`: 14
- `rows_with_a_converged_exit`: 186
- `rel_step1_l2_median`: 1.101e-06
- `rel_step1_l2_p90`: 6.563e-05
- `rel_step1_l2_max`: 3.841e+00
- `rel_final_l2_median`: 1.938e-05
- `rel_final_l2_p90`: 6.964e-04
- `rel_final_l2_max`: 4.469e+03
- `worst_step1_row`: `{"fixture": "stiff-pair-1.2", "label": "gs/iqn-ils/interface", "rel_l2": 3.8405328659088767}`
- `worst_final_row`: `{"fixture": "stiff-pair-1.2", "label": "gs/iqn-ils/interface", "rel_l2": 4468.814332002661}`
- `rows_only_in_before`: `[]`
- `rows_only_in_after`: `[]`

### By acceleration

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `aitken` | 50 | 0 | 7.13e-07 | 2.35e-04 | 1.53e-05 | 1.63e-03 |
| `fixed` | 98 | 0 | 2.06e-06 | 1.07e-04 | 2.46e-05 | 8.44e-04 |
| `iqn-ils` | 38 | 0 | 3.99e-07 | 3.84e+00 | 5.75e-06 | 4.47e+03 |

### By convergence norm

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `interface` | 89 | 0 | 1.59e-05 | 3.84e+00 | 1.84e-04 | 4.47e+03 |
| `l2` | 97 | 0 | 3.75e-07 | 7.52e-06 | 5.84e-06 | 5.32e-04 |

### By fixture

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `chain-2` | 15 | 0 | 2.42e-06 | 3.57e-03 | 1.09e-05 | 7.81e-03 |
| `chain-20` | 14 | 0 | 9.10e-08 | 1.44e-05 | 1.66e-04 | 7.83e-04 |
| `chain-5` | 15 | 0 | 1.16e-06 | 6.52e-03 | 1.51e-05 | 1.64e-03 |
| `mixed-modes` | 8 | 0 | 4.82e-06 | 1.45e-02 | 1.60e-05 | 5.08e-03 |
| `ring-4` | 15 | 0 | 2.10e-06 | 2.51e-03 | 2.46e-05 | 2.62e-03 |
| `ring-8` | 15 | 0 | 1.14e-06 | 4.98e-05 | 3.12e-05 | 3.52e-04 |
| `slow-drift` | 15 | 0 | 4.13e-08 | 3.39e-07 | 9.78e-08 | 1.14e-06 |
| `star-2` | 15 | 0 | 4.34e-06 | 1.80e-01 | 3.68e-05 | 3.06e-01 |
| `star-4` | 15 | 0 | 3.29e-06 | 2.26e-01 | 4.89e-05 | 2.96e-01 |
| `stiff-pair-0.25` | 15 | 0 | 4.03e-07 | 5.62e-04 | 3.73e-06 | 1.05e-03 |
| `stiff-pair-0.5` | 15 | 0 | 6.19e-07 | 1.67e-02 | 1.36e-05 | 4.15e-02 |
| `stiff-pair-0.8` | 14 | 0 | 2.01e-06 | 6.34e-02 | 8.83e-05 | 3.33e-01 |
| `stiff-pair-0.95` | 12 | 0 | 0.00e+00 | 4.71e-01 | 1.52e-04 | 7.66e-01 |
| `stiff-pair-1.2` | 3 | 0 | 8.23e-07 | 3.84e+00 | 2.05e-06 | 4.47e+03 |

