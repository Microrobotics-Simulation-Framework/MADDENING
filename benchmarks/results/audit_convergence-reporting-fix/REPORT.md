# Independent audit — `fix/convergence-reporting-regression`

Diff audited: `git diff origin/release/0.4.0...HEAD` at `a2ad49d`
(204 lines in `core/graph_manager.py`, 7 in `core/coupling/group.py`, tests).
Worktree: `MADDENING-wt/fix/convergence-reporting-regression`.
Everything below was run with `PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu`.
Where I say "on base", I exported `origin/release/0.4.0` with
`git archive` into a scratch tree and ran the *same* script against it,
so base and head comparisons are the same code on the same machine.

---

## Answer to question 1: yes, `converged` can still be wrong — in both directions

* **False `True` (returns a state outside tolerance and calls it converged).**
  Still reachable on the `ift` path, on the exit that *met* its criterion.
  The re-measurement the fix adds is gated on `criterion_met`, so the one
  exit that reports success keeps the one-update lag. Proven on an ordinary
  linear graph (no scripted residuals): reported `0.25`, `converged=True`,
  returned state's own residual `2.5`, tolerance `1.0`. **Identical on base**
  — the fix neither introduced nor closed it. MAJOR-1 below.
* **False `False` is essentially gone** where the fix reaches. On the cap
  exit both solvers now measure the state they return, and the residual
  reported for a group sitting on its fixed point went from `2.5e-6`
  (`converged=False`) on base to `0.0` (`converged=True`) on head. That part
  of the claim holds.
* **A third direction the author did not consider: no verdict at all.** A
  NaN residual is `False` against `>` *and* `<=`, so `strict_convergence`
  said nothing while `coupling_diagnostics()` said `converged=False`. The
  fix made that gap reachable in a case where the guard used to fire.
  MAJOR-2 — **found, fixed, regression-tested** in commit `e6a119f`.

---

## CRITICAL

None.

---

## MAJOR-1 — `converged=True` does not survive the caller re-measuring (pre-existing)

*Confidence: proven. Reproduced end-to-end on a plain graph, and identical
on `origin/release/0.4.0`, so this is not a regression from this diff.*

`_fixed_point_while` measures the iterate each pass *starts from*. After
`n` bodies, `final_res = r(x_{n-1})` while `x_star = x_n`. The fix
re-measures only when `criterion_met` is False, so the criterion-met exit
still reports a number about a state the caller never receives — for
*every* acceleration, including the two-pass-guarded `aitken`.

Fixture (`_amplifying_graph` in the test file): Jacobi on
`a = 10b + 1`, `b = 0.05a`, from `(12, 1.1)`. It contracts by `0.5` every
two passes but the iteration matrix is strongly non-normal, so the measured
residual sequence is `0.5, 5, 0.25, 2.5, 0.125, …`. With `tolerance=1.0`:

| solver | reported | converged | returned state | its own residual |
|---|---|---|---|---|
| `ift`  | 0.250 | **True** | (7.00, 0.35) | **2.50** |
| `fori` | 0.250 | True | (7.00, 0.60) | 0.25 |

Both report the same *number*, so the author's new solver-parity test
passes — but the number means different things. `fori` freezes on the
iterate it measured, so its flag is sound; `ift` hands back one further
update, so its flag is not. `strict_convergence` reads the same number and
therefore does not fire either: the IFT adjoint is taken at a state 2.5×
outside the tolerance the step just claimed to meet.

This directly contradicts three statements in the diff:
* the new `coupling_diagnostics` docstring, *"Both values are independent of
  `solver` … the flag does not move when a graph migrates between them"* —
  the flag does not move, but its truth value does;
* the author's summary that **cap 1 is the only place the lag survives** — it
  is not; the criterion-met exit at every cap survives it too (the
  `coupling_diagnostics` docstring does admit this, the summary does not);
* the new test `test_converged_means_the_returned_state_is_within_tolerance`,
  which asserts a guarantee the code does not provide. It passes only
  because its fixture is contractive.

**Not fixed here, deliberately.** The two ways to close it are both design
calls with costs the maintainer should choose between, and neither is
minimal:
1. drop the `criterion_met` gate and always re-measure — one extra `F`
   per converged group per step on the framework's default path, and it
   breaks the author's new solver-parity test (the numbers would then
   differ, because `fori` reports the latched measurement);
2. carry the previous iterate and return the one whose residual passed, as
   `fori` does — no extra `F`, closes the "two solvers return different
   states" item as well, but it changes the state every converged `ift`
   group returns (by up to `tolerance`) and makes the solve slightly less
   accurate in the ordinary contractive case.

Recorded as `test_converged_is_proof_the_returned_state_is_within_tolerance`,
`@pytest.mark.xfail(strict=True)` with the reasoning in the marker, so it
turns into a hard error the moment someone closes it.

---

## MAJOR-2 — `strict_convergence` went silent on an overflowed solve (introduced here; FIXED)

*Confidence: proven, with a deterministic reproduction that raises on base
and did not raise on head. Fixed in `e6a119f`.*

Both guards asked `residual > threshold`. NaN is `False` against `>` exactly
as against `<=`, so a NaN residual raised nothing. Meanwhile
`coupling_diagnostics()` asks `residual <= threshold` and reported
`converged=False` for the same run — the two disagreed on the one case where
the IFT gradient is certainly invalid.

The diff made a previously-guarded case unreachable by the guard. A solve
that overflows reports a finite-or-`inf` residual from the pass *before* the
cap, but measuring the state it *returns* is `inf - inf` = NaN. Deterministic
two-node fixture (`a = 10b + 1`, `b = 1e30·a`, `strict_convergence=True`,
`acceleration="none"`, `tolerance=1e-6`):

| `max_iterations` | base | head (audited) | head + fix |
|---|---|---|---|
| 2 | **raises** | silent, returns `b=inf` | raises |
| 3, 4 | silent | silent | raises |

The same thing on the diverging `gain=3.0` fixture: base raises up to and
including `max_iterations=41`, head stops raising at 41 and returns
`a=1.66e38` with `converged=False` and no error.

Fix: both `eqx.error_if` predicates (the `max_iterations=1` branch and the
`ift` branch) now ask `jnp.logical_not(residual <= threshold)`, which is
identical for every finite residual and agrees with the flag on NaN. That
also closes the pre-existing hole at caps ≥ 42.

Tests: `test_a_residual_that_is_not_a_number_raises_under_strict_convergence`
(caps 2/3/4) and `test_a_diverged_group_is_never_reported_as_converged`.
All three fail on head before the fix (`DID NOT RAISE`) and pass after.

---

## MINOR

### M-1 — "can only flip `converged` False→True" is false at `max_iterations=2` with `aitken`
*Confidence: proven (scripted loop).* Base skipped the `jnp.maximum` when
`n_iters == 1`, so it reported a single sub-threshold measurement as
`converged=True`. With the `first_res` seeding the streak is answerable
there, so an exit whose pre-loop residual was above threshold now
re-measures. Scripted: `first_res=1.0`, schedule `[1e-3, 5.0]`, `cap=2`,
`threshold=1e-2` → base `1e-3`/`True`, head `5.0`/`False`. The new answer is
the correct one (the returned state really is at `5.0`), but it **is** a new
`strict_convergence` raise path for `aitken` + `max_iterations=2` users and
belongs in the release notes rather than in an argument that no new errors
are possible.

### M-2 — "only that branch pays" is not true under `vmap`
*Confidence: proven.* A batched predicate turns `lax.cond` into `select_n`
and both branches are evaluated (checked on this JAX build: the jaxpr of a
vmapped `cond` contains `select_n` and no `cond`, with the false branch's
work hoisted). Any `vmap`ed coupled step therefore pays the extra `F`
unconditionally, on the `ift` path and on the `fori` path with
`diagnostics=True`. Values are still correct — `select_n` discards the
untaken result and its transpose feeds it a zero cotangent — so this is cost,
not correctness. Worth one sentence in the docstring.

### M-3 — `diagnostics=True` now costs a pass on the `fori` path
*Confidence: proven by counting traced node updates.* `fori` at any cap ≥ 2:
2 traced passes with `diagnostics=False`, 3 with `diagnostics=True` (base: 2
either way). Expected from the design, but it means turning diagnostics on
changes the cost of a `fori` step by ~1/N, which the CHANGELOG does not say.

### M-4 — `iterations` is off by exactly one between the solvers (pre-existing)
*Confidence: proven; identical on base.* At a cap exit `ift` reports
`max_iterations - 1` and `fori` reports `max_iterations`, on the same graph
with the same residual. Slightly more serious than "counts differently": the
obvious user test for a cap exit, `iterations == max_iterations`, is never
true on the default solver. The author's parity test excludes `iterations`
from the comparison, which documents the gap without closing it.

---

## VERIFIED SAFE (negative results)

* **The `lax.cond` does not change the returned state.** Structural: in both
  paths the cond rebinds only `final_res`, after `x_star` / `final_state` is
  fixed. Empirical: spring fixture, `repr`-exact positions identical between
  base and head at every cap and acceleration tested.
* **`jax.grad` through a coupling group is unchanged, bit for bit.**
  `d(loss)/d(position_a)` through the jitted step is
  `0.0007996251806616783` on base and on head, for `none`, `fixed`,
  `aitken` and `iqn-ils`. The `custom_jvp` still owns the derivative, so the
  extra primal evaluation cannot reach it; `aux_dot` is zeros over an `aux`
  that now contains a cond-produced leaf, which is fine.
* **The `nondiff_argnums` shift is right.** `_ift_solve_impl`'s parameters
  are `0=step_pure, 1=x0, 2=consts, 3=accel_init, 4=first_res, 5=threshold,
  6=max_iter, 7=acceleration, 8=relaxation, 9=n_reuse, 10=sub_idx,
  11=linear_solver`; `nondiff_argnums=(0,5,6,7,8,9,10,11)` leaves exactly
  `(x0, consts, accel_init, first_res)` as primals, which is the 4-tuple
  `_ift_solve_jvp` unpacks, and the rule's positional prefix matches the
  nondiff order. Forward and reverse mode both exercised.
* **No side effect fires on the untaken branch.** `one_pass_gs`,
  `one_pass_jacobi` and the closure-converted `step_pure` contain no
  `error_if`, no `debug.print`, no callbacks and no host recording; the only
  non-arithmetic call is `jax.device_put` in the multi-GPU Jacobi branch,
  which is placement, not effect. (Under `vmap` the branch runs anyway — M-2.)
* **NaN/inf in the re-measure branch is safe as an evaluation.** A NaN or
  `inf` state produces a NaN residual rather than a trace-time or runtime
  failure, and `coupling_diagnostics()` reads NaN as `converged=False`. The
  only thing that went wrong was the guard, MAJOR-2.
* **`first_res` seeding at cap 2 is the right value, not just an answerable
  one.** `first_r = ||one_pass(s_in) - s_in||` and the loop's first body
  measures `||F(x0) - x0||` with `x0 = one_pass(s_in)`, so the seeded pair is
  two genuinely consecutive measurements on successive iterates — not a
  measurement from a different sequence. The `fori` path already seeded
  `first_below` from the same quantity before this diff, so the change makes
  `ift` match `fori` rather than inventing a rule.
* **The cap-1 exception's dependency is real.** `_one_iteration_variant`
  rewrites every group to `max_iterations=1` and the profiler computes
  `coupling_overhead_ms = mean_step_ms - one_iteration_step_ms`, so a second
  evaluation there would silently deflate the reported coupling overhead.
  Counting traced node updates: cap 1 costs exactly 1 pass on both solvers,
  with and without diagnostics. The claim holds — but see MAJOR-1: cap 1 is
  *not* the only surviving lag.
* **Question 4 — the author's correction is right, and I verified it
  independently.** At `9aeba17^1` (the commit before the PR #32 merge)
  `graph_manager.py` already contains the `eqx.error_if` on
  `final_res > conv_threshold_value` in the `ift` branch, and
  `_fixed_point_while` at that commit has
  `init = (x0, jnp.array(jnp.inf, dtype=dtype), …)` with no `_TWO_PASS_EXIT`,
  no `prev` slot and no `jnp.maximum` — it returned the last pass's residual
  unmodified. PR #32 introduced the two-pass machinery and the `maximum`,
  which can only make the reported number larger and therefore only make the
  raise more likely. So: the raise predates PR #32; PR #32 made the number it
  reads worse. Describe the branch that way.
* **Question 5, first half — `iterations` divergence is pre-existing.**
  Confirmed against base, see M-4.
* **Question 5, second half — the two solvers returning different states on a
  converged exit is pre-existing but more serious than judged.** It is not a
  cosmetic difference: it is exactly what makes `ift`'s `converged=True`
  unsound while `fori`'s is sound (MAJOR-1). Fixing MAJOR-1 by option (2)
  above would close both at once.

---

## What I ran

```
pytest tests/core/test_coupling*.py tests/core/test_profiler.py -q -rs
  -> 389 passed, 3 skipped, 4 deselected, 3 xfailed in 347s
```
The 3 skips are the author's `cap 1 returns before the accelerator is built`
(cap-1 has no accelerator to compare, covered by the other accelerations);
the 3 xfails are MAJOR-1, strict, deliberate.

```
scripts/check_anomalies.py      OK
scripts/check_impl_mapping.py   OK (16 mappings)
scripts/check_citations.py      OK (13 citations; 2 pre-existing uncited-bib warnings)
scripts/check_transforms.py     OK
```

Timings on this box are not trustworthy (another agent is running); the
`[IFT perf] fori 75.6 us, ift 83.8 us` line printed by the existing perf test
should be re-read on a quiet machine before anyone concludes anything from it.

## Commit

`e6a119f fix(core): strict_convergence stops ignoring a NaN residual`
— the MAJOR-2 fix, its three regression tests, the MAJOR-1 xfail record, and
one `### Fixed` CHANGELOG bullet. Not pushed; no PR touched.
