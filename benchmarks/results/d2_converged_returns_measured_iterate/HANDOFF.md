# D2 — `ift` returns the iterate its criterion passed

Branch `fix/converged-returns-measured-iterate`, forked from
`origin/repro/ift-fori-divergence` (PR 46) @ `4a6f26a`.

**Everything this branch set out to do is done.**  The fix, the tests
and the housekeeping were finished on 2026-09-18 before an unplanned
power-off; the sweep quantification that the power-off interrupted was
redone and completed later the same day, 350 of 350 rows on both trees
— see "DONE — the sweep replay" below and `REPORT.md`.  What is left is
a decision for the maintainer, not work: the `_KNOWN_DISAGREEMENTS`
entries under "THE FINDING THAT NEEDS A DECISION", and whether
`stiff-pair-1.2` under `iqn-*`/`interface` should stay as it is.

## DONE — the fix itself (commit `6866695`)

`_fixed_point_while` in `src/maddening/core/graph_manager.py` now carries
the measured iterate alongside the produced one and returns the measured
one when the criterion fired:

    x_next, x_meas, final_res, prev, n_iters, acc = while_loop(...)
    criterion_met = ...                     # unchanged
    x_star = jnp.where(criterion_met, x_meas, x_next)

The cap exit is untouched (returns `x_next`, pays one `F` to measure it).
`cond` and the stopping rule are untouched, so **no row changes how many
passes it runs** — only which of two adjacent iterates leaves the loop.
This is exactly what the fori path's `_merge` has always done.

Verified, on the two-node affine cycle in
`tests/core/test_coupling_solver_equivalence.py` (x64, CPU):

| | before | after |
|---|---|---|
| `fori` tau / `ift` tau | 1.0 / 1.0214 | 1.0 / 1.0 (**bit-identical**) |
| `ift` analytic d tau/d gain | 1.0218679905 | 1.0218679905 (unchanged) |
| exact `1/(1-rho)` | 1.0218679747 | — |
| `ift` central FD of its own forward | 1.0213990268 | 1.0 |
| analytic vs its own FD | 4.59e-04 | **2.19e-02** |
| same at `tolerance=1e-6`, cap 200 | — | analytic 1.0218679905, FD 1.0218667868, rel 1.2e-06 |

On `_amplifying_graph` (the non-normal pin) both solvers now return
`(7.0, 0.6)` with residual 0.25 == the returned state's own residual, at
caps 3/4/5.

## DONE — tests (commits `6866695`, `a23df66`)

- `tests/core/test_coupling_solver_equivalence.py` rewritten: the whole
  "the gap exists" half of the file asserted a 2.14% divergence and had
  to be flipped, not just the two xfails.  37 passed.
- `tests/core/test_coupling_convergence_reporting.py`: the strict xfail
  `test_converged_is_proof_the_returned_state_is_within_tolerance` is
  now a positive assertion; `test_the_fori_solver_never_claims_a_state_
  it_did_not_measure` became `test_neither_solver_claims_...`,
  parametrised over both solvers.  98 passed, 3 skipped.
- `tests/core/test_coupling_fixture_invariants.py`: ten
  `_KNOWN_DISAGREEMENTS` entries added — see below.  42 passed.
- Profiler + the two pinning files together: 166 passed, 3 skipped.
- All coupling files (19 of them, `-m "slow or not slow"`): 313 passed
  before the `_KNOWN_DISAGREEMENTS` entries, the 3 failures were the
  fixture-invariant rows now recorded.
- Four compliance scripts: 4/4 OK (two pre-existing `check_citations`
  warnings about uncited bib entries).

**The gradient-vs-its-own-FD xfail does NOT xpass, and cannot.**  The
brief expected it to.  A finite difference of `ift`'s forward
differentiates a truncated iterate; the adjoint differentiates the fixed
point.  Neither D2 option changes that, and option 2 makes the gap
*larger* (rho^2/(1+rho) -> rho) because the discarded successor was the
nearer of the two to the fixed point.  It is replaced by a positive
assertion of the documented `residual * cond(I - dF/dx)` bound, plus a
test that both gradients agree to round-off once the live criterion is
tightened.

## DONE — housekeeping

- CHANGELOG: one 3-line block at the top of `### Changed`.
- `MADD-ANO-005`: **kept open, not retired.**  Its claim (a residual
  criterion is not a bound on the distance to the fixed point) is
  untouched by D2.  The description now separates the two: D2 removes
  the sharper failure (a state 2.5x its own tolerance reported as
  converged) and leaves the spectral one the entry was written about.

## THE FINDING THAT NEEDS A DECISION

`test_coupling_fixture_invariants.py::test_every_configuration_reaches_
the_same_fixed_point` (slow-marked) failed on 10 rows, all `iqn-*` under
`convergence_norm="interface"`:

    stiff-pair-0.5  gs/iqn-ils 1.59e-01  gs/iqn-imvj5 4.58e-01
                    jac/iqn-ils 4.43e-01 jac/iqn-imvj5 5.61e-01
    chain-5         gs/iqn-imvj5 3.50e-01 jac/iqn-ils 2.56e-02
                    jac/iqn-imvj5 5.55e-01
    ring-8          gs/iqn-imvj5 2.25e+00 jac/iqn-ils 5.16e-02
                    jac/iqn-imvj5 1.90e+00      (limit 2.5e-02)

Cause: the discarded update is IQN's quasi-Newton step, and these
fixtures' `rtol=1e-4` interface criterion is loose enough to stop before
it.  Measured on ring-8 `gs/iqn-imvj5/interface`, per-step deviation from
a `tolerance=1e-9` reference: **5e-07 before -> 6e-04 after**; 25 driven
steps compound that to 2.25.  Every step is inside its stated criterion
throughout — this is MADD-ANO-005, previously masked on the `ift` path by
the free extra pass.

**The same rows under `solver="fori"` have always drifted** — measured on
the pre-fix tree: ring-8 `jac/iqn-ils/interface` 5.16e-02,
`gs/iqn-imvj5/interface` 1.46e-02, both over the 2.5e-02 limit.  The
invariant held only because these tests run the default `ift`.

Recorded as `_KNOWN_DISAGREEMENTS` entries rather than by widening the
threshold, so the test asserts they are still disagreeing.  **Retire them
by tightening the fixtures' `atol`/`rtol`** — that is the change the
maintainer may want instead.

One further, separate, pre-existing difference found on the way: for
`iqn-imvj`, `fori` stores secant matrices that are zeroed out by the
post-convergence frozen passes (shift-and-insert pushes the real columns
out), so `fori`'s IMVJ is numerically identical to its IQN-ILS.  `ift`
stores genuine columns.  That is *not* addressed here and is why `ift`
and `fori` IMVJ trajectories still differ across timesteps even though
they now agree within a step.

## DONE — the sweep replay (2026-09-18, a later session)

**Complete: 350 of 350 rows on both trees.**  Written up in
`REPORT.md`; the numbers are in `comparison.json` / `comparison.md` and
`exit_analysis.json` / `exit_analysis.md`, and the raw per-step records
in `raw/before.jsonl` and `raw/after.jsonl` so the comparison can be
recomputed without replaying either tree.  Harness: `replay_sweep.py`,
`diff_sweep.py`, `exit_analysis.py`, all in this directory and all
committed.  The run took minutes per tree, not the ~55 the lost attempt
estimated: recording what each coupling group returned is far cheaper
than profiling it, which is what `bench_coupling_sweep` spends its time
on.

To redo it from scratch:

    git archive 4a6f26a src | tar -x -C <scratch>/before
    cd benchmarks/results/d2_converged_returns_measured_iterate
    PYTHONPATH=<scratch>/before/src JAX_PLATFORMS=cpu python \
        replay_sweep.py --out raw/before.jsonl
    PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python \
        replay_sweep.py --out raw/after.jsonl
    python diff_sweep.py raw/before.jsonl raw/after.jsonl \
        --out comparison.json --markdown comparison.md
    python exit_analysis.py raw/before.jsonl raw/after.jsonl \
        --out exit_analysis.json --markdown exit_analysis.md

`replay_sweep.py` skips rows already present in its `--out` file, so it
is restartable; `--start` / `--limit` cut it into chunks.

Two departures from the recipe above, both deliberate and both recorded
in the harness docstrings.  `warmup` is 0 for every fixture, so both
trees start from the identical deterministic initial state and the
state after step 1 is the uncompounded single-exit shift the brief
asked for.  Rows run in a staircase order over (fixture, configuration)
rather than fixture by fixture, so any prefix is a near-complete
rectangle of the grid — which, with a commit every 50 rows, is what
makes another power-off cost minutes.

Headline: **the quantity the group was converging on moves by less than
the tolerance it was given** (interface fields, worst case 1.9e-04
relative against the fixtures' `rtol=1e-4`; under `l2` the shift is
exactly one residual, median ratio 1.0000 over 148 rows).  **A state
field the criterion never looked at is not bounded that way**: under
`iqn-*` with `convergence_norm="interface"`, `velocity` moves by a
median of 1.7% and by up to 79% on a non-divergent fixture.  Every one
of the fifteen largest single-step shifts is an `iqn-*` interface-norm
row.

The two predictions:

- **"Iteration counts and converged fractions do not move for a single
  step" — HELD**, 350/350 on both, exactly.  Over the 50-step window
  166 rows land on a different iteration count and one row
  (`chain-50 jac/aitken/l2`) on a different converged flag, which is
  the returned state changing the next timestep's problem.
- **"Rows with `at_cap_fraction == 1.0` must be bit-identical" —
  FAILED, 16 of 17.**  The proxy is wrong, not the fix.
  `at_cap_fraction` is the profiler's `iters >= cap - 1`, and `cond`
  stops at `i >= max_iter - 1`, so a group meeting its criterion on the
  last pass the cap allows is counted as at-cap while having taken a
  criterion exit.  `chain-50 jac/fixed0.8/l2` is exactly that: 59
  iterations on all 30 steps, residual dipping below `tolerance=1e-4`
  on step 27 alone, states differing from step 27 alone.  Restated
  about the flag rather than the proxy the invariant holds: all 16 rows
  that never report `converged=True` are bit-identical over the whole
  window, and all 39 rows unconverged at step 0 are identical after
  step 1.  There is a ~1 float32 ULP floor under "bit-identical" —
  two rows differ by 4e-08 / 1e-07 on an unconverged step because the
  added `jnp.where` changes the HLO and XLA rounds differently.

`stiff-pair-1.2` (built past the convergence limit on purpose) is the
one alarming row and deserves the maintainer's eye: under
`gs/iqn-ils/interface` it reports `converged=True` on all ten steps in
both trees at three iterations, and the fixed tree's state norm runs
115 -> 5.3e+04 where the pre-fix tree's stayed bounded at 35 -> 12.
The fix removes an accidental damping that was hiding a divergence the
interface criterion never noticed.  That is `MADD-ANO-005`, which this
branch deliberately left open.

## Not touched

`docs/release_notes/v0.4.0.md` — owned by the `docs/release-notes-040`
branch; the user-visible statement is in CHANGELOG.  Someone should add
the D2 numbers there once the sweep is redone.
