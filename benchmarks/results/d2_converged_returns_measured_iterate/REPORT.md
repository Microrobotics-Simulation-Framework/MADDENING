# D2 — how far does the returned state move?

Replay of the accelerated coupling sweep against the pre-fix tree
(`4a6f26a`, the commit this branch forked from) and against this branch,
to size the shift that `fix(core): the ift solver returns the iterate its
criterion passed` introduces in every converged coupling group.

**350 of 350 rows, both trees, complete.**  The grid is
`sweep_configs(("l2", "interface"))` restricted to `acceleration !=
"none"`, over the 18 fast fixtures of `benchmarks/coupling_fixtures.py`,
`solver="ift"` (the default), float32, CPU.  **No timings anywhere** —
this machine is shared with two other agents, so a wall clock here
measures the neighbours.

Reproduce with `replay_sweep.py`, `diff_sweep.py` and `exit_analysis.py`
in this directory.  `raw/before.jsonl` and `raw/after.jsonl` are the
records those scripts consume, committed so the comparison can be redone
without replaying either tree.

## The answer in one table

Relative L2 shift of the state returned after **one** step from the
identical initial state — the uncompounded effect of a single converged
exit — over the 308 rows that took a criterion exit at step 0 (288 of
them spring fixtures, which carry `position` and `velocity`; 20 heat,
which carry `temperature`):

| what moves | median | p90 | worst |
|---|---:|---:|---:|
| interface fields (`position`), every row | 1.7e-06 | 7.1e-05 | **1.9e-04** |
| interface field (`temperature`), every row | 1.5e-09 | 1.7e-07 | **3.4e-07** |
| non-interface field (`velocity`), every row | 3.6e-06 | 3.8e-02 | 3.8e+00 |
| non-interface field (`velocity`), `iqn-*` + `interface` norm (66 rows) | **1.7e-02** | **4.7e-01** | **3.8e+00** |

**The quantity the group was converging on moves by less than the
tolerance you asked for.**  Every fixture here runs `rtol=1e-4`, and no
interface field moved by more than 1.9e-04 relative on any of the 350
rows.  For `convergence_norm="l2"` the shift is *exactly* one residual:
measured over predicted (`residual / ||x||`) has median 1.0000 across
148 rows, p90 1.414, max 2.000 — the discarded successor differs from
the returned iterate by one update, no more and no less.

**A state field the criterion never looked at is a different story.**
Under `iqn-ils` / `iqn-imvj` with `convergence_norm="interface"` the
accelerator corrects only `accelerated_fields` (the interface set) and
the criterion measures only the same set, so the two adjacent iterates
are free to disagree about everything else.  On the spring fixtures
`velocity` is exactly that: it moves by a median of 1.7% and by up to
79% on a non-divergent fixture.  Every one of the fifteen largest
single-step shifts in the sweep is an `iqn-*` row under the interface
norm; see `exit_analysis.md`.

Direct confirmation on `star-4`, `gs/iqn-ils/interface`, one step, both
trees reporting `iterations=3, residual=0.0022670708, converged=True`:

    before  position 0.73470634   velocity 14.69413
    after   position 0.7347067    velocity 17.415796

## 1. Iteration counts and converged fractions

| | rows |
|---|---:|
| step-0 iteration counts identical | **350 / 350** |
| step-0 converged flags identical | **350 / 350** |
| iteration counts identical across the whole window | 184 / 350 |
| converged flags identical across the whole window | 349 / 350 |

Nothing moves within a step, which is what `cond` being untouched
predicts.  Over the window the returned state feeds the next timestep,
so 166 of 350 rows land on a different iteration count at some later
step — that is the returned state changing the *next* problem, not the
stopping rule changing.  Converged flags are more robust still: only one
row (`chain-50`, `jac/aitken/l2`) reports a different flag anywhere in
its window, and by one step out of thirty (`converged_fraction`
1.000 -> 0.967).

Group-level `converged_fraction` and `at_cap_fraction` are recorded per
row in `comparison.json` (`converged_fraction_before` / `_after`,
`at_cap_fraction_before` / `_after`).

## 2. Magnitude of the state change

Whole-state relative L2 shift, all 334 rows that have a converged exit
somewhere in their window:

| | after 1 step | after the window |
|---|---:|---:|
| median | 1.3e-06 | 3.0e-05 |
| p90 | 1.7e-02 | 1.0e-01 |
| max | 3.8e+00 | 4.5e+03 |

Both maxima are `stiff-pair-1.2`, `gs/iqn-ils/interface`.  That fixture
is **built past the convergence limit on purpose** (gain 1.2; the
registry comment says "a divergent group grows the state by
~rho**max_iterations every step"), and it is the sharpest illustration
of `MADD-ANO-005` in the sweep: at `iterations=3` it reports
`converged=True` on all ten steps in both trees, while on the fixed tree
its state norm runs 115 → 5.3e+04 and on the pre-fix tree it stays
bounded at 35 → 12.  Neither trajectory is right — the group is not
converging and the interface criterion is not noticing — but the fix
removes an accidental damping that was hiding it.  Percentiles with that
fixture excluded are in `comparison.md`
(`summary_excluding_divergent_fixture`).

By configuration, single step, median and max over rows:

| | `l2` median | `l2` max | `interface` median | `interface` max |
|---|---:|---:|---:|---:|
| `aitken` (35+35) | 2.3e-07 | 7.5e-06 | 8.0e-06 | 2.4e-04 |
| `fixed` 0.5/0.8 (70+70) | 5.5e-07 | 4.6e-06 | 1.8e-05 | 1.1e-04 |
| `iqn-ils` (35+35) | 1.3e-07 | 3.5e-06 | **1.6e-02** | **3.8e+00** |
| `iqn-imvj` reuse 5 (35+35) | 1.3e-07 | 3.5e-06 | **1.6e-02** | **3.8e+00** |

Read as a rule: under `convergence_norm="l2"` nothing in this sweep
moved by more than 7.5e-06 in one step, whatever the accelerator.
Everything large is `iqn-*` under the interface norm.

## 3. The two predictions from `HANDOFF.md`

### "Iteration counts and converged fractions should not move at all for a single step, because `cond` is untouched" — **HELD**

350 of 350 rows agree on both, exactly, at step 0.  This is also the
strongest evidence the harness is sound: the two trees are running the
same solver up to the returned iterate.

### "Rows with `at_cap_fraction == 1.0` must be bit-identical" — **FAILED**, and the prediction was wrong rather than the fix

17 rows have `at_cap_fraction == 1.0`; 16 are bit-identical and one is
not: `chain-50`, `jac/fixed0.8/l2`, first differing at step 27.

The cause is the proxy, not the code.  `at_cap_fraction` is the
profiler's `iters >= cap - 1` (`simulation/profiler.py:495`), and
`cond` stops at `i >= max_iter - 1`, so a group that meets its criterion
on the *last pass the cap allows* records `iterations == cap - 1` and is
counted as at-cap while having taken a criterion exit.  That is exactly
this row: cap 60, 59 iterations on every one of its 30 steps, residual
dipping to 8.52e-05 against `tolerance=1e-4` on step 27 and only on step
27 — which is precisely the step the states start to differ.  Its
`converged_fraction` is 1/30.

**The invariant the prediction was reaching for holds.**  Stated about
the flag instead of the proxy: of the 16 rows that never report
`converged=True` anywhere in their window, **16 of 16 are bit-identical
over the whole window**, and of the 39 rows unconverged at step 0, all
39 are bit-identical after step 1.  A step that reports unconverged
never reaches the changed line.

One caveat found while checking this, worth knowing but not worth
acting on: two rows (`chain-5` and `star-4`, both `jac/fixed0.5/l2`)
first differ on a step that reports unconverged.  The differences are
1.3e-07 and 4.4e-08 relative — one float32 ULP (`eps = 1.19e-07`) — and
those rows go back to bit-identical on the following steps.  Adding the
`jnp.where` changes the emitted HLO, and XLA rounds the unconverged
steps differently at the last bit.  There is therefore a ~1 ULP floor
under "bit-identical", not a semantic change.

## What this means for re-running MIME

1. **Any experiment whose coupling groups use `convergence_norm="l2"`
   does not need re-running on account of D2.**  The returned state
   moves by one residual, worst case 7.5e-06 relative across the whole
   sweep, and the residual is the number the experiment already reports.
2. **Any experiment reading only interface / coupled quantities does not
   need re-running.**  Those moved by at most 1.9e-04 relative — below
   the `rtol=1e-4` the groups were asked for.
3. **Experiments using `acceleration="iqn-ils"` or `"iqn-imvj"` with
   `convergence_norm="interface"` must be re-run if they report any
   state outside the interface set.**  Velocities and similar derived
   fields move by a median of 1.7% and by tens of percent at p90; that
   is large enough to change a plotted curve, and on a group near or
   past its convergence limit it is unbounded.
4. **MIME's D2 drag-force group is a case 3 by signature.**  The D1/D2
   analysis (`plans/MADDENING_040_DECISIONS.md`) records it satisfying
   its criterion on pass one — a 1.7e-05 N force against `atol=1e-8` —
   and the rows here that exit in the fewest passes are the ones that
   move most: the 28 rows exiting at three passes have a median
   single-step shift of 4.0e-02, against 1.3e-06 for the sweep as a
   whole.  An early exit means the discarded update is a whole
   quasi-Newton step rather than a converged tail.
5. **Tightening the live knob remains the way to make the choice
   irrelevant.**  Under the interface and mixed norms that is
   `atol`/`rtol`, never `tolerance`, which is hard-coded out
   (`graph_manager.py:1146`).

## Caveats

- float32, CPU, `solver="ift"`, the 18 fast fixtures.  `expensive-pair`
  and `heterogeneous` (the 1e5-cell rows) are not in the grid.
- `warmup` is 0 for every fixture, a deliberate departure from
  `bench_coupling_sweep`'s defaults, so that both trees start from the
  identical deterministic initial state and the step-1 number is the
  uncompounded shift.  `slow-drift` normally takes 40 warmup steps to
  leave its transient; its rows here therefore describe the transient.
- The window is `min(spec.steps, 50)` steps — 10 for `stiff-pair-1.2`,
  30 for the largest shapes, 50 for the rest — so the compounded column
  is not a long-run statistic.
- Relative shifts are `||x_after - x_before||_2 / ||x_before||_2` over
  the concatenated float32 state of the coupled nodes; driver nodes are
  excluded because they sit outside every group.
- `by_fixture` / `by_acceleration` tables cover only rows with a
  converged exit, so their row counts fall short of 20 where a fixture
  never converged.
