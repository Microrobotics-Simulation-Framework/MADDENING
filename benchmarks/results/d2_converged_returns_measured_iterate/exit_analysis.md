## What predicts the shift

### `criterion_exit_at_step0`

```json
{
 "rows": 308,
 "rel_step1": {
  "n": 308,
  "median": 2.283368268389153e-06,
  "p90": 0.023470594716185362,
  "max": 3.8405328659088767,
  "min": 0.0
 },
 "rel_final": {
  "n": 308,
  "median": 3.609183200353626e-05,
  "p90": 0.15653867947690842,
  "max": 4468.814332002661,
  "min": 0.0
 },
 "bit_identical": 0
}
```

### `cap_exit_at_step0`

```json
{
 "rows": 42,
 "bit_identical_whole_window": 16,
 "rel_step1": {
  "n": 42,
  "median": 0.0,
  "p90": 0.0,
  "max": 1.0816204747268494e-06,
  "min": 0.0
 }
}
```

### `unconverged_at_step0`

```json
{
 "rows": 39,
 "bit_identical_whole_window": 16,
 "rel_step1": {
  "n": 39,
  "median": 0.0,
  "p90": 0.0,
  "max": 0.0,
  "min": 0.0
 }
}
```

### `one_residual_check_l2_norm`

```json
{
 "rows": 148,
 "measured_over_predicted": {
  "n": 147,
  "median": 1.0,
  "p90": 1.4142135416093207,
  "max": 2.000000056810502,
  "min": 0.0
 },
 "predicted_residual_over_norm": {
  "n": 148,
  "median": 3.455067057488445e-07,
  "p90": 2.612491455358375e-06,
  "max": 5.701359261149155e-06,
  "min": 0.0
 }
}
```

### `fewest_pass_criterion_exits`

```json
{
 "iterations": 3,
 "rows": 28,
 "rel_step1": {
  "n": 28,
  "median": 0.04007314624550032,
  "p90": 0.47081937124667145,
  "max": 3.8405328659088767,
  "min": 4.713169787439813e-10
 }
}
```

### `by_state_field`

```json
{
 "position": {
  "criterion_exit_rows": {
   "n": 288,
   "median": 1.6719514535523258e-06,
   "p90": 7.057184589486462e-05,
   "max": 0.00019169679626533922,
   "min": 0.0
  },
  "criterion_exit_rows_iqn": {
   "n": 66,
   "median": 8.112705600298028e-07,
   "p90": 0.0001026636148569307,
   "max": 0.00019169679626533922,
   "min": 9.64559692235471e-08
  }
 },
 "temperature": {
  "criterion_exit_rows": {
   "n": 20,
   "median": 1.490435152744361e-09,
   "p90": 1.6644796645916517e-07,
   "max": 3.3920156157303157e-07,
   "min": 0.0
  },
  "criterion_exit_rows_iqn": {
   "n": 4,
   "median": 9.808760657441712e-10,
   "p90": 1.490435152744361e-09,
   "max": 1.490435152744361e-09,
   "min": 4.713169787439813e-10
  }
 },
 "velocity": {
  "criterion_exit_rows": {
   "n": 288,
   "median": 3.606212589759679e-06,
   "p90": 0.03810255285505669,
   "max": 3.842215438177094,
   "min": 0.0
  },
  "criterion_exit_rows_iqn": {
   "n": 66,
   "median": 0.016737000144900552,
   "p90": 0.4713465300577286,
   "max": 3.842215438177094,
   "min": 6.57708138292464e-06
  }
 }
}
```

### Fifteen largest single-step shifts

| fixture | config | step0 iters | converged | rel shift after 1 step | rel shift after the window |
|---|---|---:|---|---:|---:|
| `stiff-pair-1.2` | `gs/iqn-ils/interface` | 3 | yes | 3.84e+00 | 4.47e+03 |
| `stiff-pair-1.2` | `gs/iqn-imvj5/interface` | 3 | yes | 3.84e+00 | 5.41e+01 |
| `stiff-pair-1.2` | `jac/iqn-ils/interface` | 4 | yes | 1.12e+00 | 1.34e+00 |
| `stiff-pair-1.2` | `jac/iqn-imvj5/interface` | 4 | yes | 1.12e+00 | 1.24e+00 |
| `stiff-pair-0.95` | `jac/iqn-ils/interface` | 4 | yes | 7.90e-01 | 1.03e-01 |
| `stiff-pair-0.95` | `jac/iqn-imvj5/interface` | 4 | yes | 7.90e-01 | 8.96e-01 |
| `stiff-pair-0.95` | `gs/iqn-ils/interface` | 3 | yes | 4.71e-01 | 7.66e-01 |
| `stiff-pair-0.95` | `gs/iqn-imvj5/interface` | 3 | yes | 4.71e-01 | 8.38e-01 |
| `stiff-pair-0.8` | `jac/iqn-ils/interface` | 4 | yes | 2.69e-01 | 7.71e-02 |
| `stiff-pair-0.8` | `jac/iqn-imvj5/interface` | 4 | yes | 2.69e-01 | 6.61e-01 |
| `star-8` | `gs/iqn-ils/interface` | 3 | yes | 2.49e-01 | 2.87e-01 |
| `star-8` | `gs/iqn-imvj5/interface` | 3 | yes | 2.49e-01 | 4.40e-01 |
| `star-16` | `gs/iqn-ils/interface` | 3 | yes | 2.47e-01 | 2.40e-01 |
| `star-16` | `gs/iqn-imvj5/interface` | 3 | yes | 2.47e-01 | 6.48e-01 |
| `star-4` | `gs/iqn-ils/interface` | 3 | yes | 2.26e-01 | 2.96e-01 |

