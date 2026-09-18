# Convergence as an error bound, and a scale-aware norm — measured

Evidence for `feat/convergence-error-bound` (items 11-13). The raw records in
`raw/` are the two 350-row sweep runs the numbers come from, so the comparison
can be recomputed without replaying either tree.

## Reproducing

```
# pre-change tree: origin/release/0.4.0 @ 6ada908
benchmarks/results/d2_converged_returns_measured_iterate/replay_sweep.py --out before.jsonl
# this branch
benchmarks/results/d2_converged_returns_measured_iterate/replay_sweep.py --out after.jsonl
python benchmarks/results/convergence_error_bound/verdicts.py before.jsonl after.jsonl
```

350 rows on both trees, no errors either side. No wall-clock timings: the
machine was shared throughout and they would be meaningless.

## Verdicts

| | value |
|---|---|
| converged step-verdicts lost | **438 of 14 733** (−2.9 % net) |
| converged step-verdicts gained | 8 |
| rows changing verdict at all | 50 of 350 |
| rows that used to converge and now never do | **12** |
| rows losing some steps but still converging | 36 |

## Cost

| | before | after |
|---|---:|---:|
| mean iterations / step | 16.884 | **18.620** (+10.3 %) |
| at-cap steps | 1 103 | 1 652 |
| groups with more iterations | — | 244 |
| groups with fewer | — | 22 |
| groups unchanged | — | 94 |

+10.3 % iterations is **not** +10.3 % wall time; no time was measured.

## Returned states

| | median | p90 | max |
|---|---:|---:|---:|
| relative L2 shift after one step | 0 | 5.3e-05 | 7.2e-04 |
| shift over the whole window | 1.9e-05 | 1.9e-03 | **2.7** |

68 of 350 rows are bit-identical throughout. The tail is entirely the
`iqn-*` / `interface` family — `stiff-pair-0.95 gs/iqn-imvj5/interface` at 2.7,
`chain-20 jac/iqn-imvj5/interface` at 0.84, `star-2 gs/iqn-imvj5/interface` at
0.76. Those rows converge on 100 % of steps before *and* after; what moved is
where they converge **to**. They are the `_KNOWN_DISAGREEMENTS` entries, whose
recorded cause is a criterion loose enough to stop before the quasi-Newton step
lands, so a large shift there is that entry being paid down.

## The twelve that stop converging are one family

```
star-4          jac/fixed0.5/l2      stiff-pair-0.95  gs/fixed0.5/interface
star-2          jac/fixed0.5/l2      stiff-pair-0.95  jac/aitken/l2
stiff-pair-0.8  jac/fixed0.5/l2      stiff-pair-0.95  gs/fixed0.8/l2
chain-20        jac/fixed0.5/l2      stiff-pair-0.95  jac/fixed0.8/l2
star-8          jac/fixed0.5/l2      stiff-pair-0.95  jac/fixed0.5/interface
chain-50        gs/fixed0.5/l2       chain-50         jac/fixed0.5/interface
```

Eleven of twelve are **fixed under-relaxation**; the twelfth is Aitken on the
stiffest fixture. At omega = 0.5 every step is halved by construction, so
rho >= 0.5 and the error is at least 2x the residual — 6x measured on
`chain-5`, 20x on `stiff-pair-0.95`. These are exactly the configurations whose
residual was least informative about their distance from the answer: they were
already inaccurate and reporting success.

## What did not change

- **`_KNOWN_DISAGREEMENTS`: none of the ten retire.** The slow lane still
  passes while asserting `not repaired`, so every `iqn-*` / `interface` row
  still leaves the trajectory by more than its limit after 25 steps.
  Tightening the criterion moves those states a long way without closing the
  gap; the entries' own prescription (tighten the fixtures' `atol`/`rtol`)
  remains the change that retires them.
- **Gradients** are bit-for-bit identical on the equivalence fixture, whose
  `F` is affine — so `dF/dx` is constant and the IFT adjoint is the fixed
  point's sensitivity wherever the forward stopped. That is *not* general:
  where `F` is non-linear and `x_star` moved (282 of 350 rows), the adjoint is
  linearised at the new `x_star` and moves with it, bounded by the state
  movement above.

## Limits of this evidence

- The two halves (error bound, scale-aware norm) **cannot be separated** in
  these numbers. A run with the norm change alone would attribute the 8 gained
  verdicts and part of the state movement; it was not made.
- MIME was not measured. The `interface`-norm criterion on a ~1e-5 quantity is
  the case this change was aimed at, so the MIME re-run is now needed for a
  second reason.
- `benchmarks/coupling_fixtures.py`'s registry caps are demonstrably too small
  for `fixed` omega = 0.5 under an honest criterion. Left alone deliberately;
  the next baseline re-record should decide whether the sweep wants a larger
  cap or wants to keep reporting those rows as unconverged.
