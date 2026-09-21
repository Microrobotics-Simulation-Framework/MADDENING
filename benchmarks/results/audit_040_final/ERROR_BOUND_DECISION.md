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


## 2026-09-22 — the spectrum is measured: `spectral_error_bound`

Appended, not rewritten: everything above stands as the record of the
decision it describes.  This section records what landed on
`feat/spectral-error-bound` (Phase 3 item 3) against it.  Measurements
on this machine, jaxlib 0.11.0, CPU, float32; reproducers were run
from the session scratchpad and are reproduced by the tests named
below.

**What was built.**  Under `solver="ift"` with `diagnostics=True`,
after the fixed-point solve, eight Arnoldi steps on the Jacobian-vector
product the IFT adjoint already builds — at the *returned* iterate, in
the group's own norm coordinates (`D J D^+`, `D` the per-field weights
`_scaled_change` applies there) — give three scalars in `_meta`: the
Ritz spectral radius `rho_spectral` (Gelfand's formula on the k×k
Hessenberg, log-space, from above; no non-symmetric `eigvals`, so it
lowers on every backend), the Arnoldi residual `h_{k+1,k}`, and the
resolvent norm `||(I - H)^{-1}||_2`.  `coupling_diagnostics()` gains
`rho_spectral`, `spectral_error_bound = residual * max(||(I - H)^{-1}||,
1 / (1 - rho_spectral - 2 h_{k+1,k}))` and `spectral_usable`
(finite, and `h_{k+1,k} <= 0.05 (1 - rho_spectral)`).  The seven
existing keys keep their names and values, the criterion is unchanged,
and the returned state is bit-identical to the base tree on 12
fixture/configuration rows.  Cost: 8 Jacobian-vector products per group
per step plus an 8×8 SVD and 24 8×8 products, charged only with
`diagnostics=True`.  `fori`, `diagnostics=False` and `max_iterations=1`
report NaN and `False`.

**Why Arnoldi and not the power iteration this memo proposed.**  Built
first as written above.  On 60 random symmetric contractions of
dimension 2–6 eight power-iteration steps under-resolved clustered
spectra (true 0.983 read 0.949) and put the bound *below* the true
distance in about one draw in six, with no stationarity test able to
see it; on the heterogeneous fixture's Jacobi map, whose spectrum is
symmetric under sign, the norm-ratio sequence oscillated and the bound
read `inf` on every step.  A coupling Jacobian's rank is at most the
number of boundary scalars crossing the group's edges, so an eight-step
Krylov space is its whole range for any group with up to eight of them
and the Ritz spectrum is exact — 80/80 draws `spectral_usable` with a
minimum `bound/distance` of 1.0001 after the change.

**Why the resolvent and not only the radius.**  Item 6 asked for the
heterogeneous fixture (60 000-cell heat grid + 4 probes, Jacobi, Aitken,
L2, tolerance 1e-4).  `rho_spectral` resolved exactly on every step
(0.693), and on the two steps where Aitken produced the residual dip
this release documented, `residual / (1 - rho_spectral)` read
**0.031x and 0.024x** of the true per-step distance — 30–40x under, with
the radius right.  The condition that failed is normality: the grid
responds strongly to the probes and the probes weakly back, the
`±lambda` eigenvector pair is nearly parallel, and an error of the shape
"grid consistent with probes, both off" has a residual `(1 - lambda²)`
times its probe part while its size is the grid's response to it.  When
the Krylov space is invariant (`h_{k+1,k} = 0` certifies it) `A Q = Q H`
and `||(I - A)^{-1} r|| <= ||(I - H)^{-1}|| ||r||` for any `r` in the
space, whatever the eigenvectors do.  With that term the bound holds on
all 20 steps: **1.47x and 3.31x** on the two dip steps.

**Measured ratios `spectral_error_bound / true distance`.**

| fixture | configuration | `error_estimate` | `spectral_error_bound` |
|---|---|---|---|
| two-mode `(0.999, 0.2)`, tol 1e-4 | gs / none | 0.0082 (122x under) | **7.95** |
| two-mode | gs / fixed ω=1.3 | 0.0022 | 1.98 |
| two-mode | gs / aitken (cap 60) | 0.50 | 1.22 |
| two-mode | gs / iqn-ils | 0.0010 | 1.22 |
| single mode ρ=0.25 | gs / none | 1.00 | 1.06 |
| log map `a + g log(1+u)` (non-linear) | gs / none | — | 1.10 |
| random normal contractions, n=2–6, 80 draws | none / fixed ω≤1 | 0.84–240 | 1.0001–228 |
| heterogeneous | jacobi / aitken, 20 steps | 0.008–0.72 | **1.47–119** |
| heterogeneous | gs / none | 0.98–2.3 | 0.991–1.93 |
| heterogeneous | gs / aitken | 0.72–18 | 1.13–3.49 |

The 1.22 on every two-mode row is the resolvent factor of the
`a → b → a` relay shape itself (`[[0, R], [0, R]]` is not normal); the
7.95 is that times the fast mode's share of the residual at the exit,
both in the conservative direction.  The 0.991 is the float32 floor of a
60 000-entry L2 norm (`eps sqrt(n) = 3e-5`, where the residual sits).
The 119 is the price of a rigorous bound on a badly non-normal map: the
resolvent norm is the worst direction in the space and the actual
residual is rarely in it.

**Criterion: unchanged, and no opt-in criterion offered.**  On the
heterogeneous Jacobi/Aitken fixture the 0.4.0 criterion already
exhausts the cap on 19 of 20 steps (true distance 4–1150x the
tolerance; the "15–31x converged" this anomaly recorded was the
pre-0.4.0 flag).  On the one step it passes, the state *is* within
tolerance (0.41x) and the spectral bound is 48.8x the tolerance: a
spectral gate would refuse the one correct pass.  A bound that loose is
right to report and wrong to gate on by default, and an opt-in that
fails every Jacobi group with a non-normal loop is not worth a
`CouplingGroup` field at a freeze.  Iteration counts and `converged`
are therefore exactly what they were, by construction; the
compile-count baseline and the recorded sweep rows do not move.

**Honesty of the name.**  It is a bound on `||x - x*||` in the group's
norm for a linear `F` with a resolved Krylov space, whatever the
iteration did.  What it is not: for a non-linear `F` it is asymptotic
(Ostrowski; exact to float32 on the log map within tolerance of its
fixed point); it is in the norm at the returned state, so the dead
band's excluded fields are outside it and the scale drifts as it does
for `residual`; it inherits `residual`'s float32 floor; and a group
with more than eight independent interface scalars gets a radius from
below and `spectral_usable=False`.  Each of these is on
`spectral_error_bound`'s docstring, on `coupling_diagnostics`' and in
MADD-ANO-005's residual risk.

**Tests.**  `test_the_estimate_is_never_smaller_than_the_distance_it_estimates`
in `tests/core/` is now
`test_the_spectral_bound_is_never_smaller_than_the_distance_it_bounds`
and passes on the new key; the property-file twin stays a strict xfail
on `error_estimate`, whose value is unchanged, with a passing sibling on
the same draws.  `test_a_hidden_slow_mode_is_the_recorded_size_and_is_not_flagged`
still pins 122x two-sided; its sibling pins 7.95x two-sided and
`rho_spectral = 0.999` to 1e-4.  Random normal contractions, the
accelerators, the relaxation factor, the non-linear map, the absent
cases and the numerics each have a named test in the same two files.

**Side finding, not acted on here.**  A relaxed iteration that diverges
past float32 range is reported `residual=0.0, converged=True,
ratio_usable=True` on the base tree: the NaN successor makes
`_field_reference` NaN, the dead band drops the field, the norm reads
zero and the loop returns the last finite iterate.  Reproduced on
`release/0.4.0` with `acceleration="fixed", relaxation=1.5` on a
`(-0.95, 0.3)` cycle.  It belongs to the dead band, not to this branch;
recorded for a registry entry.
