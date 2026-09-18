# D2 — `ift` returns the iterate its criterion passed: state at power-off

Branch `fix/converged-returns-measured-iterate`, forked from
`origin/repro/ift-fori-divergence` (PR 46) @ `4a6f26a`.  Snapshot taken
2026-09-18 under an unplanned power-off; the code and tests are complete
and green, the **sweep quantification is not finished**.

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

## NOT DONE — the sweep replay (the "quantify what moves" deliverable)

Harness written and working, run **incomplete at power-off: 72 of 350
rows on the "before" tree, 0 on the "after" tree.**  Nothing usable was
produced; no `.npz` was written.  All of it lived in the session
scratchpad (`/tmp/claude-1000/.../scratchpad/work`) and is gone.

To redo it (roughly 55 min per tree on this laptop, CPU, the two trees
sequentially):

1. Export the pre-fix source once:
   `git archive 4a6f26a src | tar -x -C <scratch>/before`
2. A replay script that, for every `sweep_configs(("l2","interface"))`
   entry with `acceleration != "none"` (350 rows over the 18 fast
   fixtures; skip jacobi on `mixed-modes`, which is `mode_fixed`), builds
   the fixture, runs `spec.warmup` then `min(spec.steps, 50)` steps,
   and records per group from `gm.coupling_diagnostics()` each step:
   iterations mean/min/max, `at_cap_fraction` (`iters >= cap - 1`),
   `converged_fraction`, the last residual — plus the concatenated
   float state of the coupled nodes.  **No timings** (other agents share
   the box).
3. Run it under `PYTHONPATH=<scratch>/before/src` and under
   `PYTHONPATH=<wt>/src`, then diff.

Expected shape of the answer, stated as a prediction so it can be
checked rather than assumed:

- **Iteration counts and converged fractions should not move at all**
  for a single step — `cond` is untouched.  Any movement is the returned
  state feeding the next timestep, so it accumulates over the 50-step
  window; the 72 rows measured on the "before" tree are only a baseline,
  not a comparison.
- **The state moves on every row that exits on its criterion.**  Rows
  with `at_cap_fraction == 1.0` must be bit-identical; that is the
  sharpest available check that the harness is sound.
- Single-step magnitude is one residual: ~`residual/||x||` for
  `acceleration="none"`, and up to three decades more for `iqn-*` under
  the interface norm (measured 6e-04 on ring-8, above).

A one-step variant of the same harness (`warmup=0`, one step, from the
deterministic initial state, so both trees start identical) is the clean
way to get the *uncompounded* "relative shift in the returned state on a
converged exit" the brief asks for; a 5-fixture subset is enough.

## Not touched

`docs/release_notes/v0.4.0.md` — owned by the `docs/release-notes-040`
branch; the user-visible statement is in CHANGELOG.  Someone should add
the D2 numbers there once the sweep is redone.
