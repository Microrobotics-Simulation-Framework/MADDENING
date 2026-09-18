# Retiring `_KNOWN_DISAGREEMENTS` — what the prescribed remedy actually does

Evidence for item 4 of the 0.4.0 maintainer list, measured on
`feat/convergence-error-bound` (PR #56) @ `62bccc5`.

`tests/core/test_coupling_fixture_invariants.py` lists ten configurations that
do not reach the common fixed point, all `iqn-*` under
`convergence_norm="interface"`. Each entry prescribed its own remedy:
*"Retire by tightening the fixture's atol/rtol."*

**Measured: it does not.** At the tightest `rtol` that keeps every
configuration converging, four of the ten retire, for +17.7% iterations across
the benchmark sweep. The other six — including all four `stiff-pair-0.5`
rows — stay open at every setting tried, and one of them returns a
**bit-identical** state across two decades of `rtol`. Nothing was tightened.
The defect is not in the tolerance.

## Verdict, row by row

Deviation from the lane's `gs/none/l2` reference after 25 driven steps, at the
lane's cap of 120, over `rtol` (`atol` held at the fixtures' 1e-08). **Bold**
is inside the lane's 2.5e-02 interface threshold.

| fixture | configuration | 1e-4 (today) | 1e-5 | 1e-6 | 3e-7 | 1e-7 | 1e-8 |
|---|---|---:|---:|---:|---:|---:|---:|
| `stiff-pair-0.5` | `gs/iqn-ils/interface` | 1.59e-01 | 1.59e-01 | 1.59e-01 | 1.60e-01 | 1.48e-01 | 4.42e-02 |
| `stiff-pair-0.5` | `gs/iqn-imvj5/interface` | 4.58e-01 | 3.82e-01 | 3.15e-01 | 3.18e-01 | 4.55e-01 | 4.01e-02 |
| `stiff-pair-0.5` | `jac/iqn-ils/interface` | 4.43e-01 | 3.73e-01 | 3.55e-01 | 3.37e-01 | 1.67e-01 | **7.42e-04** |
| `stiff-pair-0.5` | `jac/iqn-imvj5/interface` | 5.61e-01 | 5.61e-01 | 4.15e-01 | 3.14e-01 | 3.27e-01 | 2.58e-01 |
| `chain-5` | `gs/iqn-imvj5/interface` | 3.50e-01 | 3.90e-02 | 7.30e-02 | **2.65e-04** | **9.83e-06** | **6.90e-06** |
| `chain-5` | `jac/iqn-ils/interface` | 2.55e-02 | 5.08e-02 | 4.72e-02 | 5.32e-02 | 5.43e-02 | **5.93e-06** |
| `chain-5` | `jac/iqn-imvj5/interface` | 5.55e-01 | 1.17e-01 | 6.70e-02 | 1.56e-01 | **6.55e-03** | **2.53e-05** |
| `ring-8` | `gs/iqn-imvj5/interface` | 2.25e+00 | 1.58e+00 | **1.92e-03** | **2.72e-04** | **2.46e-04** | **1.47e-04** |
| `ring-8` | `jac/iqn-ils/interface` | 3.38e-02 | **1.11e-03** | **1.31e-03** | **1.31e-03** | **1.25e-03** | **1.49e-04** |
| `ring-8` | `jac/iqn-imvj5/interface` | 1.69e+00 | 5.05e-01 | 5.02e-01 | **5.11e-04** | **1.56e-04** | **1.29e-04** |

| | 1e-4 | 1e-5 | 1e-6 | 3e-7 | 1e-7 | 1e-8 |
|---|---:|---:|---:|---:|---:|---:|
| rows retired (of 10) | 0 | 1 | 2 | **4** | 5 | 6 |
| interface mean iterations | 13.54 | 16.95 | 20.31 | 22.23 | 26.81 | 45.09 |
| interface converged fraction | 1.000 | 1.000 | 1.000 | 0.997 | 0.969 | 0.758 |

**3e-07 is the floor of the usable range.** Below it the criterion asks float32
for exact bit stagnation of the interface field — 1e-07 is ~0.8 ulp of these
fixtures' positions — and rows start leaving on the cap instead of on the
criterion, which is the premise the same test asserts. The column is not
monotone (`chain-5 jac/iqn-imvj5` runs 6.70e-02 → 1.56e-01 → 6.55e-03): these
are 25-step trajectory separations, not a converging sequence, and a row
retired on a non-monotone column is a row that can come back.

## Three findings that rule the remedy out

**1. On `stiff-pair-0.5 gs/iqn-ils/interface` the knob is inert.** The returned
state is identical to the last bit at `rtol` 1e-04, 1e-05 and 1e-06, in the
same 3.00 iterations. Its residual at the exit is 1.6e-07 of the interface
quantity — about one float32 ulp, three decades inside its own threshold of
1.0. There is no tightening left to apply: the criterion is not what stops this
row. Pinned as a test rather than quoted, in
`test_tightening_rtol_does_not_move_the_resistant_stiff_pair_row`.

**2. `atol` — the other half of the prescription — is inert everywhere here.**
It is the dead band `_scaled_change` uses to decide which elements enter the
norm at all, and every interface element of every spring fixture is orders of
magnitude above 1e-08. Moving it to 1e-12 changes no digit of any row.

**3. The disagreement is in a field the criterion never looks at.** Per-field,
at the fixtures' own `rtol=1e-4`:

| row | `position` | `velocity` |
|---|---:|---:|
| `stiff-pair-0.5 gs/iqn-ils` | 1.71e-02 | 1.59e-01 |
| `chain-5 gs/iqn-imvj5` | 1.11e-02 | 3.50e-01 |
| `ring-8 jac/iqn-imvj5` | 5.32e-02 | 1.69e+00 |

Sweep-wide the split is sharper. Over the 350-row replay, on rows that exited
on their criterion under `iqn-*`, `position` moves by at most **1.0e-04** while
`velocity` moves by up to **2.2** (median 7.1e-05, p90 0.31) —
`raw/sweep_exit_analysis_rtol3e-7.json`.

## What is actually holding them open

`convergence_norm="interface"` takes its residual over the edge source
fields — `position` on these fixtures. `accelerated_fields=None` auto-detects
the **same** set. So the quasi-Newton step lands on `position`, `velocity` is
carried out of `_build_accel_state` at whatever the raw pass produced, and the
criterion is taken over exactly the fields the accelerator fixed. The
accelerator and the criterion share a blind spot, and each hides the other's.

This is why the `l2` rows are clean: that norm is over every float field, and
all twelve `l2` configurations of all three fixtures agree to 2e-05 or better
at the fixtures' existing tolerances. Twelve interface configurations, same
fixtures, same tolerances: ten of them are on this list.

It is `MADD-ANO-005` either way — a residual criterion that does not bound the
distance to the fixed point — but by a mechanism the anomaly's text did not
cover. Its `residual_risk` names cases where the *rate* `rho` is not
trustworthy; this is a case where the *norm* is not taken over the state that
moved, so `error_estimate` is a correct bound on the wrong quantity. Recorded
against the anomaly; the status is unchanged (`partially_resolved`) and is the
maintainer's to move.

## The remedy that does work, and what it costs

Point the accelerator at the whole state (`accel_scope="all"`, i.e.
`accelerated_fields` over every field rather than the auto-detected interface
set), change nothing else, and **all ten close** — at the fixtures' existing
`rtol=1e-4`, and within 0.8% of the same iteration count. `scope_experiment.py`:

| fixture | configuration | `auto` (today) | `all` | iterations |
|---|---|---:|---:|---|
| `stiff-pair-0.5` | `gs/iqn-ils` | 1.59e-01 | **2.16e-05** | 3.00 → 3.00 |
| `stiff-pair-0.5` | `jac/iqn-ils` | 4.43e-01 | **2.81e-05** | 3.60 → 3.60 |
| `stiff-pair-0.5` | `gs/iqn-imvj5` | 4.58e-01 | **3.56e-04** | 2.04 → 2.04 |
| `stiff-pair-0.5` | `jac/iqn-imvj5` | 5.61e-01 | **2.57e-05** | 2.08 → 2.08 |
| `chain-5` | `gs/iqn-imvj5` | 3.50e-01 | **2.01e-04** | 2.16 → 2.20 |
| `chain-5` | `jac/iqn-ils` | 2.55e-02 | **1.16e-05** | 4.80 → 4.84 |
| `chain-5` | `jac/iqn-imvj5` | 5.55e-01 | **1.08e-03** | 2.16 → 2.16 |
| `ring-8` | `gs/iqn-imvj5` | 2.25e+00 | **1.34e-03** | 3.32 → 3.24 |
| `ring-8` | `jac/iqn-ils` | 3.38e-02 | **5.33e-03** | 7.08 → 7.08 |
| `ring-8` | `jac/iqn-imvj5` | 1.69e+00 | **2.41e-03** | 3.04 → 3.04 |

**Not applied here.** `accel_scope="auto"` is the library's default and what
the sweep measures; `"all"` is a variant the sweep already runs deliberately
(`sweep_configs(extra_fields=True)`), and the algorithm guide has a section on
the difference. Changing the default would change what is being measured, and
the cleaner fix may be in `src/` — either auto-detection that covers the fields
the returned state carries, or refusing to pair an interface-only criterion
with an interface-only accelerated set. Both are the maintainer's call.
Generalised beyond these ten rows in
`tests/property/test_coupling_acceleration_agreement.py`, which asserts over
generated graphs that an accelerator given the whole state agrees with plain
iteration.

## What tightening would cost, if it is wanted anyway

Both trees replayed over the full 350-row sweep at the **registry's own caps**
with `benchmarks/results/d2_converged_returns_measured_iterate/replay_sweep.py`;
`iteration_cost.py` reads the records. No wall-clock timings: the machine was
shared throughout.

### `rtol` 1e-4 → 3e-7 (retires 4 of 10)

| slice | steps | iters before | iters after | change | converged before | converged after | at-cap before | at-cap after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 15800 | 18.668 | 21.967 | **+17.7%** | 0.9053 | 0.8549 | 1652 | 2478 |
| interface | 7900 | 16.667 | 23.266 | **+39.6%** | 0.9219 | 0.8213 | 651 | 1477 |
| l2 (control) | 7900 | 20.668 | 20.668 | +0.0% | 0.8886 | 0.8886 | 1001 | 1001 |
| interface + iqn | 3160 | 3.918 | 5.967 | +52.3% | 1.0000 | 0.9908 | 0 | 29 |

### `rtol` 1e-4 → 1e-6 (retires 2 of 10)

| slice | steps | iters before | iters after | change | converged before | converged after | at-cap before | at-cap after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | 15800 | 18.668 | 21.340 | **+14.3%** | 0.9053 | 0.8656 | 1652 | 2304 |
| interface | 7900 | 16.667 | 22.012 | **+32.1%** | 0.9219 | 0.8425 | 651 | 1303 |
| l2 (control) | 7900 | 20.668 | 20.668 | +0.0% | 0.8886 | 0.8886 | 1001 | 1001 |
| interface + iqn | 3160 | 3.918 | 5.132 | +31.0% | 1.0000 | 0.9984 | 0 | 5 |

For scale: PR #56 measured its own change at +10.3% mean iterations and
reported it as a headline. The `l2` control moving 0.0% is the check that the
experiment did not leak — `rtol` is not read under that norm.

### Two collateral effects, both against tightening

**The registry's caps cannot pay for it.** Five points of converged fraction
overall and ten on the interface rows are lost to the cap, and 826 more steps
land at it. This is the shortfall
`benchmarks/results/convergence_error_bound/REPORT.md` already records for
`fixed` at omega = 0.5, met from the other side: at `rtol=1e-6`
`jac/fixed0.5/interface` needs ~72 iterations against `_SPRING_MAXIT = 60`. Any
decision to tighten has to raise the caps in the same change, and then the
recorded sweep is stale for a second reason.

**It inverts a documented, tested claim.** The algorithm guide states the
interface norm "removes -5% to 31% of the iterations" on `gs/none`, asserted by
`test_interface_norm_iteration_change_is_inside_the_quoted_range` against the
recorded sweep. At `rtol=1e-6` the interface norm *adds* 16-20% on the same
fixtures (`stiff-pair-0.5` 8.32 → 10.00, `chain-5` 14.92 → 17.28, `ring-8`
14.36 → 17.24); at 3e-07, 35-40%. Part of that documented advantage was never
the norm — it was the looser criterion, which is the same fact this whole
report is about, seen from the benchmark side. Worth a line in the guide
whether or not anything is tightened.

## Reproducing

```
# the ladder (one run per rtol; --norms must stay "l2,interface")
PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python tolerance_ladder.py \
    --rtol 3e-7 --out raw/ladder_3e-7.json

# the counterfactual
PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python scope_experiment.py

# the sweep-wide cost: replay both trees, then read the records
cd ../d2_converged_returns_measured_iterate
PYTHONPATH=<wt>/src        JAX_PLATFORMS=cpu python replay_sweep.py --out before.jsonl
PYTHONPATH=<tightened>/src JAX_PLATFORMS=cpu python replay_sweep.py --out after.jsonl
python ../retire_known_disagreements/iteration_cost.py "rtol 1e-4 -> 3e-7" \
    before.jsonl after.jsonl
```

The tightened tree is a copy of `benchmarks/` with `_finish`'s `rtol` default
and `build_mixed_modes`' literal changed; `src/` is identical on both sides, so
`PYTHONPATH` points at the same tree for both runs.

## Limits of this evidence

- The ladder is three fixtures. The sweep-wide cost is all eighteen, but "does
  the row retire" was only asked of the three the slow lane runs, because those
  are the three the entries are about.
- `atol` was varied only on `stiff-pair-0.5`, where it was inert by four
  decades. The argument that it is inert on the others is the magnitudes, not a
  sweep.
- Non-monotonicity in the ladder means a row's number at a given `rtol` is one
  sample of a separating trajectory, not a limit. The four that retire at 3e-07
  all sit 20-90x inside the threshold and are monotone below 1e-06, which is
  why they are the four named; `chain-5 jac/iqn-imvj5` at 1e-07 is not, and is
  not counted.
- `accel_scope="all"` was measured on the ten rows and generalised by a
  property test over generated graphs. It has not been run over the full sweep,
  so its effect on the eighteen-fixture iteration counts is unmeasured.
