## Sweep comparison (300 rows of 350)

- `rows_compared`: 300
- `rows_ok`: 300
- `rows_failed`: 0
- `rows_with_step0_iterations_identical`: 300
- `rows_with_step0_converged_identical`: 300
- `rows_with_window_iterations_identical`: 172
- `rows_with_window_converged_identical`: 299
- `prediction1_single_step_holds`: True
- `rows_at_cap_fraction_1`: 16
- `rows_at_cap_fraction_1_bit_identical`: 15
- `prediction2_at_cap_bit_identical_holds`: False
- `at_cap_violations`: `[{"fixture": "chain-50", "label": "jac/fixed0.8/l2", "first_differing_step": 27, "rel_step1_l2": 0.0, "converged_fraction_before": 0.03333333333333333}]`
- `rows_that_move`: 285
- `rows_bit_identical`: 15
- `rows_with_a_converged_exit`: 285
- `rel_step1_l2_median`: 1.285e-06
- `rel_step1_l2_p90`: 3.572e-03
- `rel_step1_l2_max`: 3.841e+00
- `rel_final_l2_median`: 2.640e-05
- `rel_final_l2_p90`: 3.529e-02
- `rel_final_l2_max`: 4.469e+03
- `worst_step1_row`: `{"fixture": "stiff-pair-1.2", "label": "gs/iqn-ils/interface", "rel_l2": 3.8405328659088767}`
- `worst_final_row`: `{"fixture": "stiff-pair-1.2", "label": "gs/iqn-ils/interface", "rel_l2": 4468.814332002661}`
- `rows_only_in_before`: `[]`
- `rows_only_in_after`: `[]`

### By acceleration

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `aitken` | 64 | 0 | 8.31e-07 | 2.35e-04 | 1.78e-05 | 2.40e-03 |
| `fixed` | 121 | 0 | 2.03e-06 | 1.07e-04 | 3.20e-05 | 2.47e-03 |
| `iqn-ils` | 66 | 0 | 7.93e-07 | 3.84e+00 | 4.22e-05 | 4.47e+03 |
| `iqn-imvj` | 34 | 0 | 7.29e-07 | 3.84e+00 | 3.57e-05 | 5.41e+01 |

### By convergence norm

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `interface` | 144 | 0 | 2.15e-05 | 3.84e+00 | 3.50e-04 | 4.47e+03 |
| `l2` | 141 | 0 | 2.88e-07 | 7.52e-06 | 6.51e-06 | 5.32e-04 |

### By fixture

| key | rows | bit-identical | rel step1 median | rel step1 max | rel final median | rel final max |
|---|---:|---:|---:|---:|---:|---:|
| `chain-2` | 18 | 0 | 3.27e-06 | 1.60e-02 | 1.23e-05 | 6.42e-02 |
| `chain-20` | 18 | 0 | 1.12e-07 | 2.62e-04 | 1.67e-04 | 4.44e-01 |
| `chain-5` | 18 | 0 | 5.57e-06 | 6.52e-03 | 2.23e-05 | 3.03e-01 |
| `chain-50` | 17 | 0 | 4.50e-08 | 8.01e-06 | 1.63e-05 | 3.28e-04 |
| `mixed-modes` | 10 | 0 | 4.82e-06 | 1.45e-02 | 1.60e-05 | 1.55e-01 |
| `ring-16` | 18 | 0 | 4.47e-06 | 4.98e-05 | 1.26e-04 | 2.47e-03 |
| `ring-4` | 18 | 0 | 2.37e-06 | 8.02e-02 | 3.08e-05 | 4.95e-02 |
| `ring-8` | 18 | 0 | 5.27e-06 | 4.98e-05 | 4.10e-05 | 1.96e-01 |
| `slow-drift` | 18 | 0 | 1.49e-09 | 3.39e-07 | 2.10e-08 | 1.14e-06 |
| `star-16` | 2 | 0 | 5.43e-06 | 1.05e-05 | 7.50e-05 | 1.49e-04 |
| `star-2` | 18 | 0 | 5.49e-06 | 1.89e-01 | 1.94e-04 | 3.06e-01 |
| `star-4` | 18 | 0 | 3.39e-06 | 2.26e-01 | 5.03e-05 | 2.96e-01 |
| `star-8` | 18 | 0 | 2.36e-06 | 2.49e-01 | 1.27e-04 | 4.40e-01 |
| `stiff-pair-0.25` | 18 | 0 | 4.14e-07 | 1.20e-03 | 5.33e-06 | 2.59e-02 |
| `stiff-pair-0.5` | 18 | 0 | 6.76e-07 | 2.35e-02 | 1.57e-05 | 1.63e-01 |
| `stiff-pair-0.8` | 18 | 0 | 2.01e-06 | 2.69e-01 | 8.83e-05 | 3.33e-01 |
| `stiff-pair-0.95` | 16 | 0 | 2.93e-07 | 7.90e-01 | 1.52e-04 | 8.38e-01 |
| `stiff-pair-1.2` | 6 | 0 | 5.58e-01 | 3.84e+00 | 6.70e-01 | 4.47e+03 |

