## Sweep comparison (350 rows of 350)

### All rows

- `rows_compared`: 350
- `rows_ok`: 350
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 231
- `rows_with_step0_converged_identical`: 325
- `rows_with_window_iterations_identical`: 196
- `rows_with_window_converged_identical`: 292
- `prediction1_single_step_holds`: False
- `rows_at_cap_fraction_1`: 29
- `rows_at_cap_fraction_1_bit_identical`: 29
- `prediction2_at_cap_bit_identical_holds`: True
- `at_cap_violations`: `[]`
- `rows_never_reporting_converged`: 28
- `rows_never_reporting_converged_bit_identical`: 28
- `prediction2_restated_holds`: True
- `restated_violations`: `[]`
- `rows_that_move`: 154
- `rows_bit_identical`: 196
- `rows_with_a_converged_exit`: 322
- `rel_step1_l2_median`: 0.000e+00
- `rel_step1_l2_p90`: 2.137e-04
- `rel_step1_l2_max`: 2.191e+00
- `rel_final_l2_median`: 0.000e+00
- `rel_final_l2_p90`: 3.151e-02
- `rel_final_l2_max`: 3.526e+00
- `worst_step1_row`: `{"fixture": "stiff-pair-0.95", "label": "jac/iqn-ils/interface", "rel_l2": 2.191129983955112}`
- `worst_final_row`: `{"fixture": "stiff-pair-0.95", "label": "gs/iqn-ils/interface", "rel_l2": 3.5263188478703063}`
- `rows_only_in_before`: `[]`
- `rows_only_in_after`: `[]`

### Excluding `stiff-pair-1.2` (built to diverge)

- `rows_compared`: 330
- `rows_ok`: 330
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 215
- `rows_with_step0_converged_identical`: 305
- `rows_with_window_iterations_identical`: 180
- `rows_with_window_converged_identical`: 274
- `prediction1_single_step_holds`: False
- `rows_at_cap_fraction_1`: 17
- `rows_at_cap_fraction_1_bit_identical`: 17
- `prediction2_at_cap_bit_identical_holds`: True
- `at_cap_violations`: `[]`
- `rows_never_reporting_converged`: 16
- `rows_never_reporting_converged_bit_identical`: 16
- `prediction2_restated_holds`: True
- `restated_violations`: `[]`
- `rows_that_move`: 150
- `rows_bit_identical`: 180
- `rows_with_a_converged_exit`: 314
- `rel_step1_l2_median`: 0.000e+00
- `rel_step1_l2_p90`: 9.890e-05
- `rel_step1_l2_max`: 2.191e+00
- `rel_final_l2_median`: 0.000e+00
- `rel_final_l2_p90`: 1.225e-02
- `rel_final_l2_max`: 3.526e+00
- `worst_step1_row`: `{"fixture": "stiff-pair-0.95", "label": "jac/iqn-ils/interface", "rel_l2": 2.191129983955112}`
- `worst_final_row`: `{"fixture": "stiff-pair-0.95", "label": "gs/iqn-ils/interface", "rel_l2": 3.5263188478703063}`

### By acceleration

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `aitken` | 65 | 35 | 0.00e+00 | 2.19e-04 | 0.00e+00 | 9.58e-03 |
| `fixed` | 117 | 58 | 0.00e+00 | 9.91e-05 | 0.00e+00 | 2.38e-03 |
| `iqn-ils` | 70 | 38 | 0.00e+00 | 2.19e+00 | 0.00e+00 | 3.53e+00 |
| `iqn-imvj` | 70 | 37 | 0.00e+00 | 2.19e+00 | 0.00e+00 | 2.47e+00 |

### By convergence norm

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `interface` | 166 | 12 | 1.56e-05 | 2.19e+00 | 3.54e-04 | 3.53e+00 |
| `l2` | 156 | 156 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

### By fixture

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `chain-2` | 20 | 12 | 0.00e+00 | 1.88e-05 | 0.00e+00 | 3.97e-02 |
| `chain-20` | 19 | 9 | 0.00e+00 | 2.64e-04 | 2.36e-04 | 2.21e-01 |
| `chain-5` | 20 | 10 | 0.00e+00 | 2.53e-05 | 7.05e-06 | 4.19e-01 |
| `chain-50` | 17 | 8 | 0.00e+00 | 8.40e-06 | 1.31e-05 | 4.31e-04 |
| `mixed-modes` | 10 | 5 | 1.95e-06 | 1.38e-02 | 3.42e-05 | 2.92e-01 |
| `ring-16` | 20 | 10 | 5.00e-06 | 7.06e-05 | 1.22e-04 | 2.38e-03 |
| `ring-4` | 20 | 10 | 0.00e+00 | 8.36e-02 | 8.43e-06 | 3.41e-02 |
| `ring-8` | 20 | 10 | 0.00e+00 | 4.26e-05 | 2.82e-05 | 6.57e-01 |
| `slow-drift` | 20 | 20 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| `star-16` | 19 | 9 | 0.00e+00 | 3.03e-01 | 6.48e-05 | 7.61e-01 |
| `star-2` | 19 | 9 | 0.00e+00 | 1.89e-01 | 2.03e-04 | 2.47e+00 |
| `star-4` | 19 | 9 | 0.00e+00 | 1.31e-01 | 1.77e-04 | 1.48e+00 |
| `star-8` | 19 | 9 | 0.00e+00 | 2.90e-01 | 1.70e-04 | 2.38e+00 |
| `stiff-pair-0.25` | 20 | 10 | 2.58e-10 | 1.20e-03 | 1.65e-07 | 7.97e-02 |
| `stiff-pair-0.5` | 20 | 10 | 0.00e+00 | 1.44e-05 | 0.00e+00 | 1.28e-01 |
| `stiff-pair-0.8` | 19 | 9 | 0.00e+00 | 3.10e-01 | 5.46e-05 | 6.23e-01 |
| `stiff-pair-0.95` | 13 | 5 | 0.00e+00 | 2.19e+00 | 1.21e-06 | 3.53e+00 |
| `stiff-pair-1.2` | 8 | 4 | 5.80e-01 | 1.18e+00 | 4.56e-01 | 1.62e+00 |

