# `error_estimate` / `bound_valid`: what they can honestly be called in 0.4.0

Written for the maintainer's decision. Renaming or demoting a diagnostics
field whose criterion moved 438 of 14 733 converged step-verdicts when it
landed is not a subagent's call, so this recommends and does not act. Everything below is measured on
`release/0.4.0` at `fix/coupling-bound-and-adjoint`; reproducers are in
`benchmarks/results/audit_040_final/coupling/`.

## Recommendation, in one line

**Keep `error_estimate`. Rename `bound_valid` to `ratio_usable`, keep
`bound_valid` as a deprecated alias through 0.4.x, and keep
`gradient_error_bound`'s value while renaming it `gradient_error_estimate`.**
`error_estimate` already says *estimate*; `bound_valid` is the field that
claims something the code cannot check, and it is the one that has to move.

## What the quantity is

`error_estimate = residual * max(omega * amplification, 1)` with
`amplification = 1/(1 - rho)`, `rho = max(r_k/r_{k-1}, sqrt(r_k/r_{k-2}))`
and `omega` the step scale (this branch added the `omega`). It is the sum
of a geometric series of remaining step lengths. That sum bounds the
distance to the fixed point only if four things hold. **Three of them can
fail with `bound_valid=True`.**

| # | Condition | Checked? | Worst measured understatement |
|---|-----------|----------|-------------------------------|
| 1 | the measure obeys the triangle inequality | no | unbounded in principle; `test_the_triangle_inequality_does_not_hold`, already recorded |
| 2 | `rho` is at least the asymptotic rate | **no** | **122x** (modes `(0.999, 0.2)`, `tolerance=1e-4`) |
| 3 | the step scale is the one applied | partly | was 1.97x (`fixed`, omega=1.95) — **fixed on this branch**; still 2.04x under `aitken`, 4.5x under `iqn-*` |
| 4 | the ratio is monotone and finite | **yes** | this is all `bound_valid` reports |

Condition 4 is the only one the flag named `bound_valid` actually tests.

## What this branch fixed, and what it did not

**Fixed — condition 3 for constant relaxation.** The series summed
`||F(x) - x||` while the iterate moved `omega * (F(x) - x)`, so
`est/true` tracked `1/omega`: 0.68 at omega=1.5, 0.51 at omega=1.95.
`relaxation_step_scale` now supplies omega to every criterion site (both
solvers, both `strict_convergence` guards, `coupling_diagnostics`,
`sysid`'s mask). On the affine fixture `est/true` is now **exactly 1.0
wherever the relaxed iteration does not overshoot** (`mu = 1 - omega(1-rho) >= 0`)
and conservative where it does — which is the correct behaviour of a
geometric series over an alternating sequence, not slack. Measured range
over an 88-point (gain, omega) grid: **0.992 to 4.53** — the 0.992 is
float32 noise on a relative norm, not a systematic shortfall.

**Not fixed — condition 2.** `rho` reads the mode that dominates the
*step*. On a two-mode contraction the residual sequence is a clean
geometric decay at the *fast* rate for as long as the fast mode's
amplitude dominates, while the distance still to travel already belongs
to the slow one. Measured: `(0.999, 0.2)`, `tolerance=1e-4` →
`error_estimate = 9.19e-05`, true distance `1.12e-02`, `bound_valid=True`,
`converged=True`.

This is not an oversight and I do not recommend forcing a fix at a freeze.
The reason is sharper than "estimating a rate is hard": for the first
several passes the two-mode sequence and a genuine single-mode decay at
0.2 are *the same sequence*. The measured consecutive ratios at the exit are 0.2000 and
0.2024 — stationary to three figures, so the two-step `sqrt` guard agrees with
the one-step ratio, and a longer window, a stationarity test, or a
multi-term fit would all agree with both. **No function of the residual
norms separates the valid case from the invalid one.** Recovering the
slow mode needs information the residual sequence does not contain.

**Not fixed — condition 3 for `aitken` and `iqn-*`.** Aitken's factor is
re-derived per pass and clipped to `[0.01, 2.0]`; it saturates at 2.0 on
the audit fixture and understates by **2.04x**. IQN understates by
**4.5x**, but by mechanism 2: a superlinear sequence reads `rho -> 0`, so
`amplification` collapses to 1 while the true remaining error is still
`1/(1 - rho_spectral)` of the residual. Correcting Aitken means carrying a
dynamic scalar through both solvers' carries and the `_meta` payload, and
the value that matters is the one the *next* step will use — an estimate,
not a bound. A static `2.0` would be a bound and would tighten the
criterion for every Aitken group. Both are documented on
`relaxation_step_scale` rather than corrected.

## Is there a statable condition the code could check?

I looked for one and could not find one that is both sound and cheap.

* **From the residual sequence: no.** Argued above — the counterexample
  is stationary.
* **From the spectrum: yes, and this is the real fix.** Under
  `solver="ift"` the machinery for `I - dF/dx` already exists. A few
  power iterations on `dF/dx` at the fixed point give `rho_spectral`
  directly, and `residual / (1 - rho_spectral)` *is* a bound (modulo
  condition 1). Cost is a handful of extra JVPs per group per step,
  available only under `"ift"`, and it is a new numerical component —
  post-0.4.0.
* **Conservative fallback: sound, but a different product.** Reporting
  `+inf` unless the spectrum is known would fail every
  `strict_convergence` run.

So for 0.4.0 the quantity is an estimate. The question is what it is called.

## Recommendation in full

1. **`error_estimate` keeps its name and its value.** The name already
   says estimate, it is strictly better than the pre-0.4.0 residual test
   in every measured case, and it is never below the residual. Renaming
   it would churn the 438 verdicts for no gain in honesty.
2. **`bound_valid` → `ratio_usable`.** This is the one that misleads:
   users read `bound_valid=True` as "the bound holds", and it holds only
   under conditions 1-3, none of which it looks at. `ratio_usable` says
   exactly what the code computes — the contraction ratio was monotone,
   finite, and had a non-zero predecessor. Ship `bound_valid` as a
   deprecated alias returning the same value through 0.4.x so no caller
   breaks; remove in 0.5.0.
3. **`gradient_error_bound` → `gradient_error_estimate`**, same value,
   same alias treatment. It is numerically `error_estimate` and inherits
   every way that can understate; calling it a bound is the same overclaim
   one level down.
4. **If (2) and (3) are judged too much churn at a freeze**, the minimum
   acceptable action is documentation-only, and it is already done on this
   branch: `_fixed_point_while`, `coupling_diagnostics` and
   `error_amplification` now enumerate all four conditions and state that
   `bound_valid=True` does not certify the estimate, with the 122x number
   in the text. I would not consider "leave the name and say nothing"
   defensible — three independent mechanisms is no longer a caveat.
5. **MADD-ANO-005 needs a substantive revision**, proposed in the PR body
   rather than edited here (that file is another branch's). Its
   `residual_risk` currently names unexcited modes and non-linear `F`;
   neither covers a *linear, fully excited* two-mode map, which is the
   122x case. Its `description` still says the fourth defect "is now
   closed".

## Tests that pin this

`test_the_estimate_is_invariant_to_the_relaxation_factor` (example +
property) pins the omega fix and the exact `mu >= 0` tightness relation.
`test_the_estimate_is_never_smaller_than_the_distance_it_estimates`
(example + property) is a **strict xfail** with the audit's case pinned
by `@example`: when condition 2 is fixed it XPASSes and fails the suite,
which is the signal to revisit this memo.
`test_a_hidden_slow_mode_is_the_recorded_size_and_is_not_flagged` pins
the 122x itself, so a partial improvement is visible rather than
silently still-failing.
