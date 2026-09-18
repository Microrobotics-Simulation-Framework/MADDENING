## Sweep comparison (350 rows of 350)

### All rows

- `rows_compared`: 350
- `rows_ok`: 350
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 350
- `rows_with_step0_converged_identical`: 350
- `rows_with_window_iterations_identical`: 184
- `rows_with_window_converged_identical`: 349
- `prediction1_single_step_holds`: True
- `rows_at_cap_fraction_1`: 17
- `rows_at_cap_fraction_1_bit_identical`: 16
- `prediction2_at_cap_bit_identical_holds`: False
- `at_cap_violations`: `[{"fixture": "chain-50", "label": "jac/fixed0.8/l2", "first_differing_step": 27, "first_difference_on_a_converged_step": true, "rel_step1_l2": 0.0, "converged_fraction_before": 0.03333333333333333}]`
- `rows_never_reporting_converged`: 16
- `rows_never_reporting_converged_bit_identical`: 16
- `prediction2_restated_holds`: False
- `restated_violations`: `[{"fixture": "chain-5", "label": "jac/fixed0.5/l2", "first_differing_step": 1}, {"fixture": "star-4", "label": "jac/fixed0.5/l2", "first_differing_step": 1}]`
- `rows_that_move`: 334
- `rows_bit_identical`: 16
- `rows_with_a_converged_exit`: 334
- `rel_step1_l2_median`: 1.291e-06
- `rel_step1_l2_p90`: 1.650e-02
- `rel_step1_l2_max`: 3.841e+00
- `rel_final_l2_median`: 3.025e-05
- `rel_final_l2_p90`: 1.006e-01
- `rel_final_l2_max`: 4.469e+03
- `worst_step1_row`: `{"fixture": "stiff-pair-1.2", "label": "gs/iqn-ils/interface", "rel_l2": 3.8405328659088767}`
- `worst_final_row`: `{"fixture": "stiff-pair-1.2", "label": "gs/iqn-ils/interface", "rel_l2": 4468.814332002661}`
- `rows_only_in_before`: `[]`
- `rows_only_in_after`: `[]`

### Excluding `stiff-pair-1.2` (built to diverge)

- `rows_compared`: 330
- `rows_ok`: 330
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 330
- `rows_with_step0_converged_identical`: 330
- `rows_with_window_iterations_identical`: 164
- `rows_with_window_converged_identical`: 329
- `prediction1_single_step_holds`: True
- `rows_at_cap_fraction_1`: 5
- `rows_at_cap_fraction_1_bit_identical`: 4
- `prediction2_at_cap_bit_identical_holds`: False
- `at_cap_violations`: `[{"fixture": "chain-50", "label": "jac/fixed0.8/l2", "first_differing_step": 27, "first_difference_on_a_converged_step": true, "rel_step1_l2": 0.0, "converged_fraction_before": 0.03333333333333333}]`
- `rows_never_reporting_converged`: 4
- `rows_never_reporting_converged_bit_identical`: 4
- `prediction2_restated_holds`: False
- `restated_violations`: `[{"fixture": "chain-5", "label": "jac/fixed0.5/l2", "first_differing_step": 1}, {"fixture": "star-4", "label": "jac/fixed0.5/l2", "first_differing_step": 1}]`
- `rows_that_move`: 326
- `rows_bit_identical`: 4
- `rows_with_a_converged_exit`: 326
- `rel_step1_l2_median`: 1.291e-06
- `rel_step1_l2_p90`: 1.453e-02
- `rel_step1_l2_max`: 7.900e-01
- `rel_final_l2_median`: 3.025e-05
- `rel_final_l2_p90`: 6.055e-02
- `rel_final_l2_max`: 8.964e-01
- `worst_step1_row`: `{"fixture": "stiff-pair-0.95", "label": "jac/iqn-ils/interface", "rel_l2": 0.7899693056363033}`
- `worst_final_row`: `{"fixture": "stiff-pair-0.95", "label": "jac/iqn-imvj5/interface", "rel_l2": 0.8963625495117571}`

### By acceleration

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `aitken` | 66 | 0 | 8.31e-07 | 2.35e-04 | 2.19e-05 | 2.40e-03 |
| `fixed` | 128 | 0 | 2.06e-06 | 1.07e-04 | 3.44e-05 | 2.47e-03 |
| `iqn-ils` | 70 | 0 | 7.93e-07 | 3.84e+00 | 4.22e-05 | 4.47e+03 |
| `iqn-imvj` | 70 | 0 | 7.93e-07 | 3.84e+00 | 3.80e-05 | 5.41e+01 |

### By convergence norm

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `interface` | 169 | 0 | 3.07e-05 | 3.84e+00 | 4.39e-04 | 4.47e+03 |
| `l2` | 165 | 0 | 2.54e-07 | 7.52e-06 | 5.84e-06 | 5.32e-04 |

### By fixture

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `chain-2` | 20 | 0 | 3.27e-06 | 1.60e-02 | 1.23e-05 | 1.60e-01 |
| `chain-20` | 20 | 0 | 1.29e-07 | 2.62e-04 | 1.68e-04 | 5.73e-01 |
| `chain-5` | 20 | 0 | 5.57e-06 | 6.52e-03 | 2.23e-05 | 3.03e-01 |
| `chain-50` | 19 | 0 | 4.50e-08 | 8.01e-06 | 1.63e-05 | 4.45e-04 |
| `mixed-modes` | 10 | 0 | 4.82e-06 | 1.45e-02 | 1.60e-05 | 1.55e-01 |
| `ring-16` | 20 | 0 | 4.47e-06 | 4.98e-05 | 1.26e-04 | 2.47e-03 |
| `ring-4` | 20 | 0 | 2.37e-06 | 8.02e-02 | 3.08e-05 | 3.85e-01 |
| `ring-8` | 20 | 0 | 5.27e-06 | 4.98e-05 | 4.19e-05 | 4.12e-01 |
| `slow-drift` | 20 | 0 | 1.49e-09 | 3.39e-07 | 2.10e-08 | 1.14e-06 |
| `star-16` | 19 | 0 | 1.30e-06 | 2.47e-01 | 1.49e-04 | 6.48e-01 |
| `star-2` | 20 | 0 | 5.49e-06 | 1.89e-01 | 1.94e-04 | 6.53e-01 |
| `star-4` | 20 | 0 | 3.50e-06 | 2.26e-01 | 5.03e-05 | 6.51e-01 |
| `star-8` | 20 | 0 | 2.36e-06 | 2.49e-01 | 1.27e-04 | 6.50e-01 |
| `stiff-pair-0.25` | 20 | 0 | 4.14e-07 | 1.20e-03 | 1.28e-05 | 9.58e-02 |
| `stiff-pair-0.5` | 20 | 0 | 6.76e-07 | 2.35e-02 | 1.57e-05 | 2.85e-01 |
| `stiff-pair-0.8` | 20 | 0 | 2.01e-06 | 2.69e-01 | 8.83e-05 | 6.61e-01 |
| `stiff-pair-0.95` | 18 | 0 | 5.87e-07 | 7.90e-01 | 1.52e-04 | 8.96e-01 |
| `stiff-pair-1.2` | 8 | 0 | 5.58e-01 | 3.84e+00 | 6.18e-01 | 4.47e+03 |

