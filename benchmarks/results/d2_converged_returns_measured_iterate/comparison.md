## Sweep comparison (250 rows of 350)

- `rows_compared`: 250
- `rows_ok`: 250
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 250
- `rows_with_step0_converged_identical`: 250
- `rows_with_window_iterations_identical`: 158
- `rows_with_window_converged_identical`: 250
- `prediction1_single_step_holds`: True
- `rows_at_cap_fraction_1`: 14
- `rows_at_cap_fraction_1_bit_identical`: 14
- `prediction2_at_cap_bit_identical_holds`: True
- `at_cap_violations`: `[]`
- `rows_that_move`: 236
- `rows_bit_identical`: 14
- `rows_with_a_converged_exit`: 236
- `rel_step1_l2_median`: 1.510e-06
- `rel_step1_l2_p90`: 2.480e-04
- `rel_step1_l2_max`: 3.841e+00
- `rel_final_l2_median`: 2.565e-05
- `rel_final_l2_p90`: 2.269e-03
- `rel_final_l2_max`: 4.469e+03
- `worst_step1_row`: `{"fixture": "stiff-pair-1.2", "label": "gs/iqn-ils/interface", "rel_l2": 3.8405328659088767}`
- `worst_final_row`: `{"fixture": "stiff-pair-1.2", "label": "gs/iqn-ils/interface", "rel_l2": 4468.814332002661}`
- `rows_only_in_before`: `[]`
- `rows_only_in_after`: `[]`

### By acceleration

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `aitken` | 58 | 0 | 8.31e-07 | 2.35e-04 | 1.78e-05 | 2.40e-03 |
| `fixed` | 114 | 0 | 2.06e-06 | 1.07e-04 | 3.44e-05 | 2.47e-03 |
| `iqn-ils` | 62 | 0 | 7.93e-07 | 3.84e+00 | 4.22e-05 | 4.47e+03 |
| `iqn-imvj` | 2 | 0 | 6.78e-08 | 1.35e-07 | 8.45e-06 | 1.42e-05 |

### By convergence norm

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `interface` | 118 | 0 | 2.13e-05 | 3.84e+00 | 3.33e-04 | 4.47e+03 |
| `l2` | 118 | 0 | 3.89e-07 | 7.52e-06 | 6.88e-06 | 5.32e-04 |

### By fixture

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `chain-2` | 16 | 0 | 3.27e-06 | 1.60e-02 | 1.10e-05 | 6.42e-02 |
| `chain-20` | 16 | 0 | 1.12e-07 | 2.62e-04 | 1.67e-04 | 1.65e-03 |
| `chain-5` | 17 | 0 | 1.16e-06 | 6.52e-03 | 1.51e-05 | 4.18e-02 |
| `mixed-modes` | 8 | 0 | 4.82e-06 | 1.45e-02 | 1.60e-05 | 5.08e-03 |
| `ring-16` | 16 | 0 | 4.47e-06 | 4.98e-05 | 1.26e-04 | 2.47e-03 |
| `ring-4` | 16 | 0 | 2.37e-06 | 8.02e-02 | 3.08e-05 | 4.90e-02 |
| `ring-8` | 16 | 0 | 5.27e-06 | 4.98e-05 | 4.10e-05 | 4.59e-03 |
| `slow-drift` | 16 | 0 | 2.14e-08 | 3.39e-07 | 5.02e-08 | 1.14e-06 |
| `star-2` | 16 | 0 | 5.49e-06 | 1.89e-01 | 1.94e-04 | 3.06e-01 |
| `star-4` | 16 | 0 | 3.39e-06 | 2.26e-01 | 5.03e-05 | 2.96e-01 |
| `star-8` | 16 | 0 | 2.36e-06 | 2.49e-01 | 1.27e-04 | 3.55e-01 |
| `stiff-pair-0.25` | 16 | 0 | 4.14e-07 | 1.20e-03 | 5.33e-06 | 2.59e-02 |
| `stiff-pair-0.5` | 17 | 0 | 6.19e-07 | 2.35e-02 | 1.42e-05 | 1.63e-01 |
| `stiff-pair-0.8` | 16 | 0 | 2.01e-06 | 2.69e-01 | 8.83e-05 | 3.33e-01 |
| `stiff-pair-0.95` | 14 | 0 | 0.00e+00 | 7.90e-01 | 1.52e-04 | 7.66e-01 |
| `stiff-pair-1.2` | 4 | 0 | 5.58e-01 | 3.84e+00 | 6.70e-01 | 4.47e+03 |

