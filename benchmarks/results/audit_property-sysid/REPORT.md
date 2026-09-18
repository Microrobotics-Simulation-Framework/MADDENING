# Independent audit — `test/property-sysid`

Diff audited: `git diff origin/release/0.4.0...HEAD` in
`/home/nick/MSF/msf/MADDENING-wt/test/property-sysid` —
`tests/property/test_sysid_contract.py` (new, 29 tests: 23 Hypothesis
properties, 6 plain, 3 of them `xfail(strict=True)`), `src/maddening/sysid.py`
(+20/-6, input validation), `CHANGELOG.md` (3 lines).

The auditor did not write this code. Every number below was measured on this
machine, which was shared with other agents throughout, so wall-clock timings
are indicative only; correctness verdicts are not.

All commands used the mandated prefix
`PYTHONPATH=<WT>/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
with `/home/nick/MSF/msf/.venv/bin/python`, under the default `dev`
Hypothesis profile (`EXAMPLES_CHEAP/STANDARD/COSTLY = 200/50/20`).

---

## Headline — are the properties load-bearing?

**Yes.** Nothing here passes vacuously:

* **Filtering is healthy.** The worst acceptance rate in the file is 55%
  (`test_the_masked_matrix_is_the_submatrix_of_the_unmasked_one`); most are
  70-100%. No property is running on a degenerate corner of its input space.
* **13 of 18 seeded defects in `maddening.sysid` were caught.** Every
  off-by-one in the window tiling, every misapplied index, the dropped
  factor of 2, the inverted mask, the inverted noise model and both halves
  of the new validation were caught. The 5 survivors are all outside this
  file's declared scope, and 4 of the 5 are pinned elsewhere in the repo.
* **The tolerances still discriminate**: the `1e-7` float64 claim is sound
  (~160x headroom, and a sign error or a lost chain-rule factor is eight
  orders above it), and the `1e-2` float32 bound still catches a factor of 2
  and a sign flip. But the quoted worst case behind the `1e-2` bound is out
  by a factor of ten, and the real worst case I measured lands 0.03% below
  the bound (MAJ-3).

**Three things need action before this merges (two are fixed here):**

1. **MAJ-1 — the branch is red.** `TestCalibrate::
   test_a_parameter_the_forward_ignores_is_bit_identical_afterwards` fails on
   a fresh Hypothesis search somewhere between 1 run in 12 and 1 in 2 (two
   independent estimates). CI's main `Run tests` job does not cache the
   example database, so every run searches fresh.
2. **MAJ-2 — one of the three `strict=True` xfails xfails for the wrong
   reason.** It dies on its own premise assertion, not on the defect its
   `reason` describes. The defect it names is real; the test does not reach
   it, and would `XPASS` (i.e. fail) if the premise line were repaired
   naively.

3. **MAJ-3 — the rollout finite-difference assertion has no margin left.**
   Measured worst case 9.997e-03 against a 1.000e-02 bound over 126
   configurations. Reported, deliberately not fixed: widening a tolerance is
   the author's call, and it spends detection power.

MAJ-1 and MAJ-2 are fixed on `fix/property-sysid-audit`.

---

## Question 1 — load-bearing, in detail

### 1a. Filtering statistics (`--hypothesis-show-statistics`, dev profile)

Measured with `-p hypothesis.extra.pytestplugin --hypothesis-show-statistics`;
"accepted" = passing / (passing + invalid).

| property | passing | invalid | accepted | dominant filter |
|---|---|---|---|---|
| `test_fit_moves_only_the_masked_trainable_leaves` | 20 | 11 | 65% | `assume(any(flags))` 19%, `assume(_in_bounds)` 10% |
| `test_fit_defaults_its_mask_to_the_trainable_set` | 20 | 6 | 77% | `assume(frozen)` 12% |
| `test_every_fitter_honours_the_same_frozen_set` | 20 | 10 | 67% | strategy-level (`unique=True` lists) |
| `test_a_transform_refuses_the_bounds_it_cannot_work_with` | 200 | 0 | 100% | — |
| `test_constrain_inverts_unconstrain_inside_the_bounds` | 200 | 0 | 100% | — |
| `test_constrain_lands_inside_the_bounds_from_any_coordinate` | 200 | 0 | 100% | — |
| `test_a_fit_that_starts_inside_the_bounds_finishes_inside_them` | 20 | 0 | 100% | — |
| `test_fim_matches_a_central_finite_difference` | 50 | 2 | 96% | — |
| `test_fim_matches_a_finite_difference_of_a_rollout` | 20 | 14 | 59% | `assume(var(position) > 1e-2)` 32% |
| `test_crb_is_consistent_with_the_matrix_it_came_from` | 50 | 18 | 73% | `assume(diag > 1e-8)` 9%, `assume(cond < 1e8)` 1.5% |
| `test_the_masked_matrix_is_the_submatrix_of_the_unmasked_one` | 50 | 41 | **55%** | `assume(any(flags))` 37% |
| `test_the_noise_model_scales_...` | 50 | 15 | 77% | `assume(cond < 1e4)` 12% |
| `test_a_tiling_that_divides_is_accepted_by_both_helpers` | 20 | 0 | 100% | — |
| `test_a_tiling_that_does_not_divide_is_refused_by_both_helpers` | 20 | 4 | 83% | `assume((T-1) % window != 0)` 17% |
| `test_a_non_positive_sampling_interval_is_refused` | 4 | 0 | 100% | exhausts its 4-value domain |
| all three `TestMultipleShootingLoss` properties | 20 | 0 | 100% | — |
| `test_gradient_descent_on_a_convex_problem_...` | 50 | 0 | 100% | — |
| `test_the_converged_flag_...` | 200 | 0 | 100% | — |
| `TestTuneCouplingParams::test_the_report_describes_the_grid_it_searched` | 20 | 0 | 100% | — |
| `...::test_the_reference_configuration_reproduces_itself_exactly` | 6 | 0 | 100% | exhausts its 6-value domain |

**Verdict: no vacuous filtering.** The two heaviest filters
(`assume(any(flags))` at 37%, `assume(var > 1e-2)` at 32%) reject exactly the
inputs the property has nothing to say about (an empty mask; a spring that
barely moves), and both still leave a majority of draws alive. The two
"exhausted" properties are exhaustive over small finite domains, which is
stronger than sampling, not weaker.

### 1b. The `rtol=1e-7` float64 analytic claim — verified, slightly overstated

The docstring claims the central difference is "accurate to ~1e-10 relative,
which leaves a tolerance of 1e-7 three orders of headroom".

Measured over 400 random draws from the same distribution as
`analytic_residual()` (`benchmarks/results/audit_property-sysid/` scripts were
run from the session scratchpad):

```
worst |F - F_fd|.max() / scale = 6.36e-10      (scale = max(1, |F|.max()))
worst pure-relative error      = 7.38e-08      (on entries near zero, where
                                                the atol term dominates)
```

The assertion is `np.allclose(F, F_fd, rtol=1e-7, atol=1e-7 * scale)`, so the
binding term is `atol` and the headroom is **6.36e-10 vs 1e-7, i.e. ~160x
(2.2 orders), not three**. The claimed worst case of 3.9e-10 is the right
order; my larger sample found 6.4e-10. The error is bounded by theory
(truncation `h²f'''/6 ≈ 4e-11` plus round-off `ε|f|/h ≈ 1.5e-11` per Jacobian
entry at `h ≈ 1.5e-5`), so the tolerance cannot drift. **Load-bearing:** a
sign error, a dropped chain-rule factor or a forward-difference mix-up are all
O(1) or O(h) — eight or more orders above the bound. Mutation M9 (a spurious
factor of 2) was caught here.

### 1c. The `1e-2` float32 rollout claim — sound bound, wrong stated margin

The docstring claims "1e-2 relative to the matrix's own scale, against a worst
case of 1e-3 measured over 25 random spring configurations at a relative step
of 3e-3".

Measured over 126 usable configurations (two seeds, 55 + 71) drawn from the
test's own strategy (stiffness 1-200, damping 0.1-10, mass 0.5-5, position ±5,
velocity ±2, n ∈ {20,40}, same `h = 3e-3(1+|θ|)` and the same
step-actually-taken correction):

```
seed 1 (55 usable):  worst 3.59e-03   median 2.72e-04   p90 8.87e-04
seed 7 (71 usable):  worst 9.997e-03  median 1.61e-04   p90 8.72e-04
                           ^^^^^^^^^ the assertion bound is 1.000e-02
worst config: k=115.3 c=0.493 m=4.107 x0=4.60 v0=1.416 n=20
```

The *typical* case is better than the claimed 1e-3, but the **observed worst
case is 10x worse than stated and sits 0.03% below the assertion bound** — a
near-miss, not a margin. See MAJ-3.

Is `1e-2` still able to catch a real defect? `F` here is 2x2 and symmetric:

* **dropped/extra factor of 2** on a Jacobian column — `F_ss` scales by 4,
  `F_ds` by 2: O(1) relative, ~100x above the bound. Caught (M9).
* **transposed Jacobian** — `J` is `(n_res, n_par) = (n, 2)`, so `J @ J.T`
  is `(n, n)`. This is caught as a *shape* error, not by the tolerance; the
  properties cannot distinguish `J` from `Jᵀ` numerically because `JᵀJ` is
  symmetric by construction. Worth knowing, but not a gap this file can close.
* **wrong sign on one parameter** — flips only the off-diagonal, so the
  detectable signal is `2|F_ds| / max|F|`. Measured over the same 126
  configurations: `min = 5.19e-02`, `median = 5.81e-01`, **0% of
  configurations below the 1e-2 bound**. So yes, the loose bound still
  catches a sign error — with a worst-case margin of ~5x, which is what
  constrains how far the bound could be widened (MAJ-3). It would *not*
  catch a sign error on a parameter whose sensitivity is orthogonal to the
  other's — inherent to a 2x2 `JᵀJ` check, not to the tolerance.


### 1d. Mutation testing — which seeded defects the suite noticed

Method: apply one exact-string mutation to `src/maddening/sysid.py`, run the
properties that could see it (`-x`), restore with `git checkout --` and assert
`git status --porcelain src/maddening` is empty before the next one. The tree
was verified clean at the end (`git status` shows only my own test edit and
this report directory). Selections were ordered cheapest-discriminating-first,
so "CAUGHT" names the first property to fail, not necessarily the only one.
Caveat: the Hypothesis example database was warm across the campaign, which
can only make the suite look *better* at catching — so the SURVIVED rows are
the trustworthy ones, and they are the interesting ones.

| # | mutation | verdict | first property to fail |
|---|---|---|---|
| M1 | `windowed_loss`: truth slice off by one (`start + 1` → `start`) | **CAUGHT** | `test_truth_seeded_windows_recover_the_unwindowed_loss` |
| M2 | `windowed_loss`: `n_windows = T // window` | **CAUGHT** | `test_truth_seeded_windows_recover_the_unwindowed_loss` |
| M3 | `windowed_loss`: inner scan ignores `sample_every` (`length=1`) | **CAUGHT** | `test_truth_seeded_windows_recover_the_unwindowed_loss` |
| M4 | `windowed_loss`: drop the last-window gate on the continuity penalty | **CAUGHT** | `test_truth_seeded_windows_recover_the_unwindowed_loss` |
| M5 | `windowed_loss`: continuity penalty **doubled** | *SURVIVED* | — (see MIN-4) |
| M6 | `init_window_states`: window starts shifted by one sample | **CAUGHT** | `test_a_tiling_that_divides_is_accepted_by_both_helpers` |
| M7 | `windowed_loss`: shooting start indexed by `start` instead of `w` | **CAUGHT** | window/shooting properties |
| M8 | `windowed_loss`: `mask_unconverged` never applied | *SURVIVED* | — (out of scope; pinned in `tests/core/test_sysid.py:95` and `test_hypothesis_sysid.py:256`) |
| M9 | `fim`: `F = 2 * JᵀJ` (spurious factor of 2) | **CAUGHT** | FIM finite-difference properties |
| M10 | `fim`: relative scaling loses the sign (`θ` → `|θ|`) | *SURVIVED* | — (all drawn `θ` are positive; pinned in `test_hypothesis_sysid.py:500`) |
| M11 | `fim`: `scale="relative"` silently ignored | *SURVIVED* | — (nothing here anchors relative scaling absolutely; pinned in `test_hypothesis_sysid.py:500`) |
| M12 | `_masked_indices`: mask inverted | **CAUGHT** | `test_a_mask_that_selects_nothing_is_refused` |
| M13 | `_inverse_noise_std`: returns σ instead of 1/σ | **CAUGHT** | `test_the_noise_model_scales_the_information_by_one_over_sigma_squared` |
| M14 | `fim`: `crb` from `diag(F)` instead of `diag(pinv(F))` | **CAUGHT** | `test_crb_is_consistent_with_the_matrix_it_came_from` |
| M15 | `fit`: Adam **ascends** instead of descending | *SURVIVED* | — (see MIN-5; pinned in `tests/core/test_sysid.py`) |
| M16 | `windowed_loss`: new `window > T-1` bound removed (accepts `T == 1`) | **CAUGHT** | `test_a_window_wider_than_the_data_is_refused` |
| M17 | bound removed from `init_window_states` only (helpers disagree) | **CAUGHT** | `test_a_window_wider_than_the_data_is_refused` |
| M18 | `windowed_loss`: `sample_every <= 0` check dropped | **CAUGHT** | `test_a_non_positive_sampling_interval_is_refused` |

**Reading of the survivors.** None of them is a hole in the properties the
file *claims* to state:

* **M5** is the only one I would call a genuine weakness of a property that
  exists (MIN-4): `test_the_continuity_penalty_is_affine_in_its_weight` pins
  affinity but not the constant, and no other test in the repo pins it either.
* **M8, M10, M11, M15** are subjects this file deliberately leaves to
  `tests/verification/hypothesis/test_hypothesis_sysid.py` and
  `tests/core/test_sysid.py` (the module docstring says so: "covers the
  *contract*… the mathematics is covered there"). M10 is worth a footnote:
  `analytic_residual` draws `θ ∈ [0.5, 2.0]`, all positive, so `θ` and `|θ|`
  are indistinguishable — the relative-scaling sign is untested *here* for a
  reason nobody wrote down.

The three mutations aimed squarely at the diff's own new validation (M16, M17,
M18) were all caught, and caught by the property whose name says what it
checks. The new validation is not decoration.

---

## Question 2 — is the `src/maddening/sysid.py` change correct and complete?

**Verdict: correct, and it breaks nothing.** Two neighbouring gaps it did not
reach are recorded as minor findings.

### `sample_every <= 0` → `ValueError` (new)

* `sample_every == 0` was the real hole. `_advance_one_sample` ran
  `lax.scan(..., length=0)`, so the state never advanced and the loss compared
  each window's *initial* state, repeated `window` times, against the next
  `window` observations — a finite, plausible, monotone-looking number that is
  not a loss. Rejecting it is right.
* `sample_every < 0` already raised, from inside `lax.scan` ("length must be
  non-negative"). The new message is clearer; nothing that worked stops
  working.
* No call site in `src/`, `examples/`, `docs/` or `benchmarks/` passes a
  non-positive `sample_every`.

### `window > T - 1` → `ValueError` (new)

* **The new clause is reachable only at `T == 1`.** For `T - 1 > 0`,
  `window > T - 1` implies `(T - 1) % window == T - 1 != 0`, so the
  pre-existing divisibility clause already rejected it (only the message
  changes). At `T == 1` every positive window "divides" `T - 1 == 0`.
* What `T == 1` used to do: `window == 1` returned a silent `0.0` — a false
  "perfect fit" from zero windows; `window >= 2` died inside
  `dynamic_slice_in_dim` while tracing the zero-length scan. Neither is a
  meaningful loss, so **no legitimate call that produced a meaningful loss is
  now rejected**. `T == 1` is the initial state alone, with nothing to
  integrate against.
* `T == 0` (empty leading axis) also becomes a clean `ValueError` instead of
  `lax.scan(length=-1)`.
* **The boundary is right, not merely stricter:** `window == T - 1` (a single
  window, i.e. the unwindowed loss) is still accepted, and
  `test_truth_seeded_windows_recover_the_unwindowed_loss` exercises exactly
  that call every run.

### Do the two helpers agree?

Yes, exactly. Both strip `_META_KEY` first, both take `T` from
`_leading_len`, and both apply the identical predicate
`window <= 0 or (T - 1) % window != 0 or window > T - 1`. Boundary behaviour
is identical at `window == 1`, `window == T - 1`, `window == T`, `window == 0`
and `T == 1`. Two properties assert the agreement directly
(`test_a_tiling_that_does_not_divide_is_refused_by_both_helpers`,
`test_a_window_wider_than_the_data_is_refused`), and seeded defect **M17**
(remove the new bound from `init_window_states` only, leaving the two out of
step) was **caught**.

### Backward compatibility of the message

Every pre-existing assertion matches on `"must divide"`
(`tests/core/test_sysid.py:91`, `:261`;
`tests/verification/hypothesis/test_hypothesis_sysid.py:296`), which the new
text preserves.

---

## Findings

### CRITICAL

None.

### MAJOR

#### MAJ-1 — the branch is red: `test_a_parameter_the_forward_ignores_is_bit_identical_afterwards` fails

`tests/property/test_sysid_contract.py`, `TestCalibrate`.

A plain run of the new file fails:

```
$ ... python -m pytest tests/property/test_sysid_contract.py -q
.x..x.......x............F...
E  AssertionError: assert 0.0 == 1.5694542800437951e-43
   Falsifying example: a=1.0, x0=0.0, ghost=1.567879084204046e-43, n_iters=1
```

**Cause.** `ghost` is drawn from `_finite(-3.0, 3.0)`, i.e. `st.floats` at the
default `width=64`. A value like `1.57e-43` is a perfectly ordinary float64
but a **float32 subnormal**, and XLA's CPU backend runs with denormals flushed
to zero, so `calibrate`'s update `p - lr * 0.0` returns `0.0` even though the
gradient is exactly zero:

```
>>> float(jnp.float32(1.567879084204046e-43) + jnp.float32(0.0))
0.0
```

The property's claim (a parameter the forward ignores comes back bit-identical)
is therefore false in that corner for reasons that belong to the backend's
floating-point mode, not to `calibrate`.

**Frequency.** Reproduced from a *clean* example database: 1 failure in 2 runs
of the pytest node, and 1 in 12 independent 50-example searches in a
standalone harness. CI's `Run tests` job (`.github/workflows/ci.yml:108`,
which is what runs `tests/property/`) has **no Hypothesis database cache** —
only the `verify-hypothesis` job does, and that job runs
`tests/verification/hypothesis/` only. So every CI run searches fresh and this
is a coin-flip red build, not a one-off.

**Fix applied** (`fix/property-sysid-audit`): a `_finite_f32_normal` strategy
using `width=32, allow_subnormal=False`, used for `ghost`.
`allow_subnormal=False` **alone is not sufficient** — it is evaluated at the
strategy's width, so at the default 64 it still yields `1.34e-42`, which is
normal in float64 and subnormal in float32. That was verified the hard way: the
naive fix still failed within the first 6 of 20 fresh searches (and 3 of 12
in a separate harness). With `width=32` the property is clean
over 20 independent fresh searches.

#### MAJ-2 — xfail #1 xfails on its own premise, not on the defect it documents

`tests/property/test_sysid_contract.py::TestTrainableContract::
test_a_leaf_outside_the_mask_is_bit_identical_after_a_fit`,
`xfail(strict=True)`.

Run with `--runxfail`:

```
    gm = _spring_gm()
>   assert gm.param_specs()["nodes"]["s"]["damping"].transform == "log"
E   AssertionError: assert None == 'log'
E    +  where None = ParamSpec(trainable=True, bounds=(0.0, None),
                               transform=None, ...).transform
tests/property/test_sysid_contract.py:320
```

`SpringDamperNode.param_specs` (`src/maddening/nodes/spring.py:113-117`)
declares `stiffness` and `mass` with `transform="log"` but **`damping` with no
transform** — it is the one spring constant the property cannot be shown on.
The test never reaches the `fit` call at all.

**The documented defect is real** — I reproduced it separately:

```
gm.constrain(gm.unconstrain(p)):  stiffness 30.0 -> 30.000001907348633
```

(`unconstrain`/`constrain` transform every *trainable* leaf, masked or not, so
a `log` leaf comes back ~1 ulp off; 69% of random float32 values in 0.1-100
are not fixed points of `exp(log(·))`.) But `damping`, having no transform,
round-trips exactly — so if the premise line were simply deleted, the test
would **XPASS** and, being `strict=True`, turn into a confusing failure. This
is precisely the "xfail passing for the wrong reason" hazard.

**Fix applied:** assert on `stiffness` (which *is* `transform="log"`, and whose
default value 30.0 is *not* a float32 fixed point of `exp(log(·))` — 1.0 and
2.5 are, so the choice of value matters too), with `damping` in the mask
instead. The `reason` text needed no change; it was accurate all along.

### MINOR (reported, not fixed)

#### MAJ-3 (risk, not fixed) — the rollout finite-difference assertion has no margin left

`tests/property/test_sysid_contract.py:38-45` states "a worst case of 1e-3
measured over 25 random spring configurations at a relative step of 3e-3", and
asserts `np.abs(F - F_fd).max() <= 1e-2 * scale`.

Over 126 configurations drawn from the test's own strategy I measured a worst
case of **9.997e-03** against a bound of **1.000e-02** — the assertion would
have held by 0.03%. The distribution has a long right tail (median 1.6e-4,
p90 8.7e-4) driven by high stiffness with a large initial displacement, where
float32 cancellation in the rollout dominates the central difference. The
quoted worst case of 1e-3 is out by a factor of ten.

Eight repeat runs of the property itself with a *cleared* Hypothesis database
all passed (20 examples each, dev profile), so I have a near-miss and not an
observed failure. Extrapolating from the sample, roughly 1 draw in 126 lands
within a factor of 1.001 of the bound, which at `EXAMPLES_COSTLY = 80` under
the `ci` profile is a meaningful per-run failure probability.

**Not fixed, deliberately.** The only fixes are to widen the bound or narrow
the strategy, and both are calls about detection power that belong to the
author: at `3e-2` the margin over a sign-flip on one parameter's Jacobian
column drops from ~5x to ~1.7x (measured above). What is unambiguously wrong
is the quoted measurement; that one line should be corrected whichever way the
bound goes. The imprecision is in the *oracle* (a float32 central difference),
not in `fim`, so widening does not hide a defect in the code under test — but
it does spend the test's remaining power.

#### MIN-2 — `continuity_weight < 0` is silently ignored

The validation pass hardened `window` and `sample_every` but not the argument
next to them. `windowed_loss` gates the penalty behind
`if window_states is not None and continuity_weight > 0.0`
(`src/maddening/sysid.py:248`), so `continuity_weight=-1.0` returns the
*un-penalised* loss rather than raising or (nonsensically) rewarding
discontinuity. A user sweeping the weight through zero would get a silently
different objective on the negative side. One line in the same `if` chain
would close it.

#### MIN-3 — the `T == 1` error message names an empty interval

`f"window={window} must divide T-1={T - 1} and lie in [1, T-1] (T={T} samples)"`
renders `[1, T-1]` literally, so at `T == 1` it reads "must divide T-1=0 and
lie in [1, T-1] (T=1 samples)" — i.e. `[1, 0]`, without saying that there is
nothing to fit. The source comment explains it; the message does not.

#### MIN-4 — the absolute scale of the continuity penalty is unpinned

`test_the_continuity_penalty_is_affine_in_its_weight` checks that the loss is
`data + w * P` with a `w`-independent `P >= 0`. Doubling the penalty inside
`windowed_loss` keeps it affine, so the property cannot see it: seeded defect
**M5 survived**, and `tests/verification/hypothesis/test_hypothesis_sysid.py::
TestMultipleShooting` does not pin it either (it only checks the penalty
vanishes at the truth and is positive off it). Comparing `penalty` against an
independently computed `Σ_w ||end_w − ws[w+1]||²` would close it.

#### MIN-5 — no property in this file asserts that a fitter makes progress

`test_fit_moves_only_the_masked_trainable_leaves` has a non-vacuity clause
("something moved"), but nothing asserts that `fit`, `fit_lm` or
`fit_multiple_shooting` *decrease* their loss. Seeded defect **M15** (Adam
ascending instead of descending) is therefore invisible to this file — see the
table. It is caught by `tests/core/test_sysid.py` and
`tests/verification/hypothesis/test_hypothesis_sysid.py`, so the repo is
covered; this is a scope observation about the file, not a hole.

### VERIFIED SAFE (chased, found to be fine — recorded so nobody chases them again)

* **xfail #2, `test_a_mask_that_overrides_a_frozen_spec_still_respects_its_bounds`,
  fails for exactly the stated reason.** With `--runxfail`:
  `ValueError: ['nodes']['s']['damping']=4.499972343444824 above bound 1.0`,
  raised by the test's own `gm.check_params(res.params)`. An explicit mask that
  widens the trainable set really does move a leaf in physical coordinates past
  its bounds, and `fit` really does return a pytree its own `check_params`
  rejects. The `reason` text is accurate.
* **xfail #3, `test_crb_is_not_finite_along_an_exact_null_direction`, fails for
  exactly the stated reason.** With `--runxfail` the failure is on the *last*
  assertion, the crb one:
  `crb = Array([0.00833333, 0.00833333])` with `fim = [[30,30],[30,30]]`,
  `eigvals = [0., 60.]`, `cond = inf`. The two preceding assertions
  (`cond == inf`, `eigvals[0] == 0.0` exactly) both hold in float32 on this
  machine, so the xfail is not being propped up by a brittle premise. An
  entirely unidentifiable parameter really is reported with a tight variance
  bound.
* **`_ROUND_TRIP_RTOL = 8 * float32 eps` is not an excuse to be loose.** The
  measured worst-case perturbation of the `log` round trip over 20 000 random
  float32 values in `[0.1, 100]` is `3.34e-07` relative, i.e. **2.8 eps**; the
  8-eps allowance is about 3x that, and Adam's first step moves a leaf by
  ~`lr` (5%), eight orders above the allowance. The weakening in
  `_materially_moved` cannot hide a leaf the optimiser actually touched.
* **`_TILINGS` really tiles.** `_observe(..., sample_every=se)` takes
  `x[::se]` of `T = n+1` samples, giving `T-1 = n//se`, and the generator only
  emits `w ∈ divisors(n//se)` for `n % se == 0`. Every triple is a legal
  tiling; none of the 20 examples per run is silently skipped.
* **No regression in the existing suite's error-message expectations** — see
  Question 2.
* **The two "exhausted" properties are not under-run.**
  `test_a_non_positive_sampling_interval_is_refused` (4 examples) and
  `test_the_reference_configuration_reproduces_itself_exactly` (6) stop because
  Hypothesis has enumerated their whole domain, not because a filter starved
  them.
* **Every pre-existing `sysid` caller still works.** Nothing in `src/`,
  `examples/`, `docs/` or `benchmarks/` passes `sample_every <= 0` or a
  single-sample `observations` pytree.

---

## The three `strict=True` xfails — spot check

All three were run with `--runxfail` to see the *actual* failure.

| xfail | fails for the stated reason? | evidence |
|---|---|---|
| `test_a_leaf_outside_the_mask_is_bit_identical_after_a_fit` | **NO — see MAJ-2** | died on `assert ...["damping"].transform == "log"` (`None == 'log'`), never reaching `fit`. Fixed on `fix/property-sysid-audit`; it now fails on `assert 30.000001907348633 == 30.0`, which *is* the documented round trip. |
| `test_a_mask_that_overrides_a_frozen_spec_still_respects_its_bounds` | YES | `ValueError: ['nodes']['s']['damping']=4.499972343444824 above bound 1.0` from the test's own `gm.check_params`. |
| `test_crb_is_not_finite_along_an_exact_null_direction` | YES | fails on the final assertion with `crb = [0.00833333, 0.00833333]`; the two premise assertions (`cond == inf`, `eigvals[0] == 0.0`) both hold. |

---

## Changes made on `fix/property-sysid-audit`

Branched from `test/property-sysid` (`f6863d6`). Two surgical edits, both in
`tests/property/test_sysid_contract.py`; `src/` is untouched.

1. **MAJ-1** — new `_finite_f32_normal(lo, hi)` helper
   (`st.floats(..., width=32, allow_subnormal=False)`) with a docstring saying
   why, used for the `ghost` draw in
   `test_a_parameter_the_forward_ignores_is_bit_identical_afterwards`.
2. **MAJ-2** — `test_a_leaf_outside_the_mask_is_bit_identical_after_a_fit` now
   masks `damping` and asserts on `stiffness` (`transform="log"`, value 30.0),
   with both premises asserted explicitly and a comment recording that the
   *value* matters as much as the transform.

The `xfail` `reason` strings were not touched: all three were accurate
descriptions of real defects. No `src/` behaviour was changed, so nothing in
the `sysid` public API moves.

### Verification run

```
$ ... pytest tests/property/test_sysid_contract.py::TestTrainableContract::\
test_a_leaf_outside_the_mask_is_bit_identical_after_a_fit \
  tests/property/test_sysid_contract.py::TestCalibrate::\
test_a_parameter_the_forward_ignores_is_bit_identical_afterwards -q
1 passed, 1 xfailed in 5.67s
```

and with `--runxfail` the xfail now fails on
`assert 30.000001907348633 == 30.0`.

The de-flaked property was additionally checked over **20 independent
50-example searches with `database=None`** in a standalone harness: clean.
The same harness with the original strategy falsifies it, and with the naive
`allow_subnormal=False` (width 64) still falsifies it.

CI runs the full suite; I did not.

---

## Appendix — what was run

```
# baseline + filtering statistics (dev profile)
cd <WT> && PYTHONPATH=<WT>/src JAX_PLATFORMS=cpu PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest tests/property/test_sysid_contract.py -q -p no:cacheprovider -rs \
  -p hypothesis.extra.pytestplugin --hypothesis-show-statistics
# -> .x..x.......x............F...   26 passed, 3 xfailed, 1 FAILED   (MAJ-1)

# the three xfails, unmasked
  python -m pytest <the three node ids> -q --runxfail

# mutation campaign: 18 exact-string mutations of src/maddening/sysid.py,
# each applied, tested with -x against the properties that could see it, then
# reverted with `git checkout --` and `git status --porcelain src/maddening`
# asserted empty.  Harness + logs in the session scratchpad.

# measurements (standalone scripts, session scratchpad):
#   400 draws of the analytic residual in float64  -> 1b
#   126 spring rollouts, two seeds                 -> 1c, MAJ-3
#   20 000 float32 log round trips                 -> _ROUND_TRIP_RTOL check
#   20 fresh 50-example searches, database=None    -> MAJ-1 fix verification
#   8 repeats of the rollout FD property with a cleared .hypothesis  -> MAJ-3

# compliance
  python scripts/check_anomalies.py      OK
  python scripts/check_impl_mapping.py   OK
  python scripts/check_citations.py      OK
  python scripts/check_transforms.py     OK
```

The worktree's `.hypothesis/` database was cleared at the end of the audit
(it had accumulated counterexamples produced by the seeded mutations, which
are not real failures and would only slow future runs down).

`git status` at hand-off: clean apart from this report and the one test-file
commit. `src/maddening/` is byte-identical to `test/property-sysid`.
