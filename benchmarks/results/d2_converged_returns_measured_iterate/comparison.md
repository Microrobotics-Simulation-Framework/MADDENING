## Sweep comparison (50 rows of 350)

- `rows_compared`: 50
- `rows_ok`: 50
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 50
- `rows_with_step0_converged_identical`: 50
- `rows_with_window_iterations_identical`: 32
- `rows_with_window_converged_identical`: 50
- `prediction1_single_step_holds`: True
- `rows_at_cap_fraction_1`: 7
- `rows_at_cap_fraction_1_bit_identical`: 7
- `prediction2_at_cap_bit_identical_holds`: True
- `at_cap_violations`: `[]`
- `rows_that_move`: 43
- `rows_bit_identical`: 7
- `rows_with_a_converged_exit`: 43
- `rel_step1_l2_median`: 6.837e-07
- `rel_step1_l2_p90`: 1.795e-05
- `rel_step1_l2_max`: 6.453e-05
- `rel_final_l2_median`: 8.412e-06
- `rel_final_l2_p90`: 1.225e-04
- `rel_final_l2_max`: 3.938e-04
- `worst_step1_row`: `{"fixture": "star-4", "label": "gs/fixed0.5/interface", "rel_l2": 6.452528419185685e-05}`
- `worst_final_row`: `{"fixture": "star-4", "label": "gs/fixed0.5/interface", "rel_l2": 0.00039375107829475687}`
- `rows_only_in_before`: `[]`
- `rows_only_in_after`: `[]`

### By acceleration

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `aitken` | 17 | 0 | 5.31e-07 | 3.44e-05 | 4.16e-06 | 2.74e-04 |
| `fixed` | 26 | 0 | 8.85e-07 | 6.45e-05 | 1.69e-05 | 3.94e-04 |

### By convergence norm

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `interface` | 15 | 0 | 1.25e-05 | 6.45e-05 | 9.34e-05 | 3.94e-04 |
| `l2` | 28 | 0 | 5.12e-07 | 3.02e-06 | 5.03e-06 | 5.17e-05 |

### By fixture

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `chain-5` | 8 | 0 | 8.52e-07 | 2.29e-05 | 9.06e-06 | 1.25e-04 |
| `mixed-modes` | 5 | 0 | 1.07e-06 | 1.21e-05 | 7.76e-06 | 1.05e-04 |
| `ring-8` | 8 | 0 | 9.53e-07 | 2.81e-05 | 2.24e-05 | 1.11e-04 |
| `slow-drift` | 7 | 0 | 4.13e-08 | 3.39e-07 | 1.04e-07 | 9.04e-07 |
| `star-4` | 7 | 0 | 2.15e-06 | 6.45e-05 | 2.22e-05 | 3.94e-04 |
| `stiff-pair-0.5` | 8 | 0 | 4.64e-07 | 1.69e-05 | 8.97e-06 | 2.31e-04 |

