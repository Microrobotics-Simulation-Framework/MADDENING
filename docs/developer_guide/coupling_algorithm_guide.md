# Choosing a coupling algorithm

A `CouplingGroup` offers two iteration modes and five accelerations, plus
`relaxation`, `convergence_norm`, `accelerated_fields` and
`jacobian_reuse`.  Until now everything this project said about which
combination suits which problem was extrapolated from one measured graph
(MIME's AR4, in `benchmarks/results/ar4_*.json`) and a heat pair.  This
page replaces that with a sweep over eight synthetic graphs chosen to
span the four properties that actually decide the answer.

Read [Profiling and benchmarking](profiling.md) first if you want to know
how the per-step numbers are obtained; this page is about what they say.

> **These numbers post-date the Aitken correction.**  Every row in the
> recorded files was measured after two defects in `acceleration="aitken"`
> were fixed: its first pass of each timestep relaxed with ω clipped to
> the 0.01 floor, because the previous residual is seeded to zeros and
> the resulting zero ω hit the clip; and the loop exited on the first
> residual below the threshold, which under Aitken is a transient dip in
> a non-monotone sequence rather than a bound on the error — on
> `heterogeneous` under Jacobi that meant `converged=True` at a point
> 15–31x the tolerance it claimed to have met, and a hundredfold tighter
> tolerance barely moved the answer.  The exit now requires the
> threshold on two consecutive passes for the accelerations where one
> pass is not enough.  Aitken rows recorded earlier in this branch's
> history are not comparable, and neither is any Aitken figure quoted
> from them.

> **These numbers also pre-date the error-bound criterion.**  In 0.4.0
> a group's threshold is applied to an estimate of the distance to the
> fixed point, `residual / (1 - rho)`, rather than to the residual, and
> every norm scales each field's change by that field's own magnitude.
> Replaying the 350-row sweep across the change: mean iterations per
> step +10.3%, 438 of 14 733 converged step-verdicts lost against 8
> gained, and 12 of 350 rows that used to converge now exhaust their
> cap — eleven of them `acceleration="fixed"` at ω = 0.5 or 0.8.  Those
> rows *contract slowly on these fixtures*, so their error is a large
> multiple of their residual.  An earlier wording here said ω = 0.5
> "halves every step and so has the largest gap between its residual
> and its error", which reads as a floor ρ ≥ 1 − ω.  **There is no such
> floor, and that mechanism is retracted.**  Relaxing maps an eigenvalue
> λ to `1 - ω + ωλ`, so a λ near −1 drives ρ towards zero — damping
> oscillatory modes is what under-relaxation is *for* — and a search
> over 20 000 random iteration matrices at ω = 0.5 found ρ = **0.156**
> (`benchmarks/results/convergence_error_bound/REPORT.md`, corrected
> 2026-09-18 in `0ee18a0`).  The measured numbers are unaffected; the
> generality was not there.  Iteration counts and `converged_fraction`
> on this page are therefore lower bounds; re-recording the baselines is
> queued (`plans/MADDENING_040_DECISIONS.md`, "Not decisions").

> **Where each number on this page comes from.**  Unless it says
> otherwise, a figure is from `benchmarks/results/coupling_sweep_cpu.json`
> (the 420-row fast sweep) or `coupling_sweep_expensive_cpu.json` (the
> two grid fixtures), both recorded **2026-09-18**, `689d7c2`.  A figure
> tagged **(re-measured `c51cd6a`, 2026-09-19)** is *not* from those
> files: the fast sweep was re-run at that commit with the same
> configuration
> (`benchmarks/results/audit_040_final/docs-compliance/coupling_sweep_cpu_HEAD_c51cd6a.json`)
> and the recorded value no longer held, so the re-run's value is
> quoted instead.  A figure marked **[not re-evidenced]** is one the
> re-run contradicted with nothing on this page left to support it; it
> is marked rather than deleted, so that it is not quoted again.  The
> grid file was **not** re-run, so every `expensive-pair` and
> `heterogeneous` figure is still the 2026-09-18 recording.

> **And they pre-date the relaxation correction.**  That estimate sums
> the steps the iterate takes, and under `acceleration="fixed"` a step
> is `relaxation` times the residual that is measured, which the first
> version left out.  The criterion is now `ω · residual / (1 - rho)`.
> It therefore moved in *both* directions for `fixed` rows: ω < 1 was
> being held to a criterion `1/ω` too strict — which is where the
> eleven capped rows above came from — and ω > 1 to one `ω` too loose,
> understating its distance from the fixed point by that factor (0.51x
> at ω = 1.95).  Expect the ω = 0.5 / 0.8 rows to converge again and
> the over-relaxed rows to cost more passes than the figures here.

## Start here

Pick the row that matches the *shape* of your graph.  Every recommendation
is `iteration_mode` / `acceleration` / `convergence_norm`.

One thing to hold on to while reading it: **fewer iterations is not the
same as a faster step.**  On the small fixtures the whole step is a
fraction of a millisecond of kernel dispatch, and an accelerator that
quarters the iteration count can still triple the wall time because its
own arithmetic is not free.  The table recommends on measured step time
where the step times are separable and on iteration count where they are
not, and says which.

| your graph | start with | why |
|---|---|---|
| anything, first attempt | `gauss-seidel` / `aitken` / **`l2`** | Gauss-Seidel needs **1.6–2.0x** fewer iterations than Jacobi on every shape where both converge — 1.59 on `chain-50` to 2.04 on `slow-drift` (re-measured `c51cd6a`, 2026-09-19); the 2026-09-18 recording read 1.69 to 1.93 and this row used to quote it as 1.7–1.9x, which `chain-50` now sits below.  Aitken removes up to half of what is left (0–49% under the interface norm).  It is the sweep's own best configuration on ten of the eighteen fast fixtures (nine under the interface norm, one under L2).  Its arithmetic is inside the dispatch floor on every launch-bound fixture; on a compute-bound one it is not free.  **The norm here is `l2`, not `interface`, and that is a change**: the interface norm is the iteration-cheaper of the two and this row used to recommend it, but it measures only the coupling-edge fields, so a group whose interface has gone stationary can exit while the rest of its state is still moving.  Over generated graphs that is measurable at **5.2e-01** on exactly this configuration — a Jacobi three-spring group under Aitken, both solves reporting `converged`.  Switch to `interface` once you have read the accuracy caveat at the end of [Four combinations that theory says should win](#four-combinations-that-theory-says-should-win) and checked your own graph against it |
| cheap nodes, few interface DOFs, contraction below ~0.8 | `gauss-seidel` / `aitken` / `interface` | launch-bound: differences between configurations smaller than the dispatch floor — which is most of the step there — are not measuring the algorithm, so pick the fewest iterations and the least machinery among the rows that time the same |
| contraction above ~0.9, or unknown and possibly divergent | `gauss-seidel` / `iqn-ils` / **`l2`** | the only family that converges *past* the limit at all, and at gain 0.95 it takes 4.0 iterations at 100% converged where `gs/none/l2` exhausts its cap of 60 on **100%** of steps and converges on **0%** (re-measured `c51cd6a`, 2026-09-19) (the 2026-09-18 recording read 4.1, 98% and 2%).  Not faster there (0.21 ms against 0.10) — right rather than fast.  **The norm here is `l2`, not `interface`, and that is a change**: `iqn-*` with the default `accelerated_fields=None` is the one pairing the accuracy caveat at the end of [Four combinations that theory says should win](#four-combinations-that-theory-says-should-win) names outright — the auto-detected set and the interface criterion are the same fields, so the quasi-Newton step lands on exactly what the criterion then measures and every other field rides out of the last raw pass unwatched.  The interface norm also stopped being the lever here: on `stiff-pair-0.95` it now cuts 2.2% of the iterations, not 15.1%, and converges 8% of steps rather than 68% |
| one expensive node among cheap ones | `gauss-seidel` / `iqn-imvj` / `interface`, with `accelerated_fields` naming **only the cheap nodes** | the sweep's best configuration on `heterogeneous`: 3.0 iterations at 1.78 ms, against 5.4 and 2.02 for plain iteration, and 2.0 at 116 ms for the same accelerator on every field.  The quasi-Newton problem needs enough degrees of freedom to model the interface response, not the grid.  Under **Jacobi** the same restriction costs convergence — see below |
| every node expensive (grid-to-grid) | `gauss-seidel` / `none` / `interface` | the interface norm alone takes `expensive-pair` from 6.0 iterations and 6.17 ms to 1.5 and 1.85 ms; IQN buys 0.05 of an iteration for 91–148x the step time and is not affordable at 2x10⁵ accelerated DOFs |
| deep chain (information must cross many nodes) | `gauss-seidel` / `aitken` / `interface` | Gauss-Seidel's advantage is real but *flat* in depth — it does not grow with the chain length |
| wide star (independent leaves) | `gauss-seidel` / `aitken` / `interface` | both accelerators are flat in width (Aitken 7.9 → 8.1 iterations from 2 to 16 leaves under the interface norm, IQN 3.0 → 3.0) but IQN's step cost is not: 1.45 ms against 0.51 ms at 16 leaves, for 5 fewer iterations that the dispatch floor hides |
| ring / cycle with no natural first node | `jacobi` if the answer must not depend on how the graph was built, otherwise `gauss-seidel` / `aitken` | Gauss-Seidel on a ring is measurably order-dependent; Jacobi is bit-identical under rotation and reversal |
| fixed point that barely moves between steps, **and nothing else** | `jacobi` / `fixed` ω = 0.8 / `l2` — one of only two places a constant ω wins | 6.80 → 3.62 iterations at 100% converged on `slow-drift` (re-measured `c51cd6a`, 2026-09-19), and the sweep's best configuration there.  **Do not carry it to another shape.**  Every table on this page records `jac/fixed0.8/l2` at 100% converged, and at `c51cd6a` it converges on **20%** of `star-16` steps, 40% of `star-8`, 82% of `star-4`, 88% of `star-2`, 92% of `stiff-pair-0.8` and 58% of `chain-20` — a wide star under this row now converges one step in five.  `iqn-imvj` with `jacobian_reuse` does cut iterations further (3.0 → 2.0) but costs ~20x the plain step to do it |
| two subsystems with different shapes | one group each, with its own settings | groups in one graph keep independent schedules, iteration counts and convergence flags |

**On the `interface` norm elsewhere in this table.**  The rows above
that still recommend it do so on *iteration count*, which is what the
sweep measures and what they say.  The norm's accuracy caveat applies
to every one of them, not only to the two rows that were changed: it
sees the coupling-edge fields and nothing else.  Where the answer
matters more than the iteration count — training, calibration, anything
reading state that is not on an edge — start from `"l2"` or `"mixed"`
and move to `"interface"` only after measuring your own graph's
`fixed_point_agreement`.

Three settings that are nearly always right and are not in the table:

* **Leave `relaxation` alone under Gauss-Seidel.** Not one Gauss-Seidel
  row in the sweep beats its unrelaxed counterpart, usually by a factor
  of two in iterations, and no ω rescued a divergent case.  Under
  *Jacobi* it is worth measuring: ω = 0.8 cuts iterations 44% on
  `slow-drift` and 18% on `expensive-pair`.  See below.
* **Cap `max_iterations` at what you actually expect.** It is not a free
  safety margin: IQN allocates `max_iterations - 1` secant columns, so
  the cap sets the size of its least-squares problem.
* **Turn on `strict_convergence` for training and calibration runs.**
  The implicit-function-theorem gradient is only valid at a converged
  fixed point.
* **Leave `linear_solver` at `"gmres"` on anything grid-shaped.** The
  `"dense"` alternative is exact and is sometimes offered as the thing
  to try when the adjoint struggles; on a grid it cannot run at all.
  See [`linear_solver="dense"` is not an escape hatch on a
  grid](#linear_solverdense-is-not-an-escape-hatch-on-a-grid).

## Reading `coupling_diagnostics()`

Each group reports fourteen fields: nine for every group, and five --
three spectral, two about the gradient -- that carry a value only under
`solver="ift"` with `diagnostics=True`.  Three of the nine need
reading carefully, and one of them was renamed in 0.4.0 because its old
name said more than it checks.  A group that has not taken a step yet
(before the first `step()`, and after `reset_state()`) has no entry at
all: its `_meta` slots hold seeds that keep the scan carry's structure,
and read as a report they said `converged=True` about a group that had
never run.

| field | what it is |
|---|---|
| `iterations` | passes used, counting the first staggered one.  Equal to `max_iterations` exactly when the group exhausted its budget.  Under waveform relaxation (a sub-cycling group with `waveform_iterations > 1`) each sweep has a budget of its own and this is the largest sweep's count, so `iterations >= max_iterations` still reads "some sweep hit the cap" — see below |
| `total_iterations` | passes the step ran, summed over its waveform sweeps: the work done.  Equal to `iterations` for a group that runs one sweep |
| `residual` | `\|F(x) - x\|` in the group's norm, for the state the step returned.  Carries a float32 noise floor |
| `amplification` | the estimated `1 / (1 - rho)` of the mode the residual sequence reveals.  `nan` when the ratio was rejected |
| `error_estimate` | `residual · max(ω · amplification, 1)` — **an estimate** of the distance to the fixed point, not of the last step.  Falls back to `residual` when the ratio was rejected |
| `ratio_usable` | whether the contraction *ratio* was usable — see below.  Renamed from `bound_valid` |
| `gradient_error_estimate` | how far the IFT adjoint may sit from a finite difference of the same forward.  Numerically `error_estimate`, so it inherits every way that number can understate.  `inf` when `ratio_usable` is false.  Renamed from `gradient_error_bound` |
| `converged` | the *error estimate* met the group's threshold.  **`True` on a stalled float32 iterate** — see below |
| `rho_spectral` | the spectral radius of `dF/dx` at the returned state, from eight Arnoldi steps on the Jacobian-vector product the IFT adjoint already builds.  Sees every mode, not only the one dominating the step.  NaN for `fori`, for `diagnostics=False` and at `max_iterations=1` |
| `spectral_error_bound` | `(residual + floor) · max(‖(I − H)⁻¹‖₂, 1/(1 − rho_spectral))`, with `floor` the residual's own float resolution and `H` the Krylov-compressed Jacobian in the group's own norm — **a bound** on the distance to the fixed point for a linear `F`, whatever the accelerator did; asymptotic for a non-linear one.  See below |
| `spectral_usable` | the bound is finite and the Arnoldi space had settled (`h_{k+1,k} ≤ 0.05 (1 − rho_spectral)`).  False where nothing was computed, for a group with more than eight independent interface scalars, and where the residual is at its float floor (`precision_limited`) in a group with a node that has not declared `update_evaluations()` — see below |
| `gradient_relative_error_bound` | a bound on the relative error of the IFT gradient caused by the forward stopping early: `spectral_error_bound` × the resolvent factor it applies × the change in the map's linearisation per unit distance, for the worst of one probe per floating constant.  **About the gradient, not the solve** — reads 0.0 on an affine group whose state is far off.  See below |
| `gradient_bound_usable` | the gradient bound is finite and `spectral_usable` is true.  False where nothing was computed and where the Newton–Kantorovich check fails |
| `precision_limited` | the residual is at or below its own float resolution: `residual` and `error_estimate` are rounding, at least half of each bound is the floor, and only a wider dtype can shrink them.  Reported for every group; clears `spectral_usable` only where a node's evaluation count is undeclared |

### A stalled float32 iterate reads `converged=True`

When `(1 − rho) · |x − x*|` falls below half an ulp, a pass changes
nothing: `F(x) == x` bitwise, the residual is exactly `0.0`, and
`converged` — and `strict_convergence` — report success although the
state can be far from its fixed point.  Measured (jaxlib 0.11.0,
float32): a relay contracting at 0.99999, started 0.3% short of its
fixed point, stops after one pass with `residual=0.0` and
`converged=True` at **38 348 ulps**, a relative distance of 4.2e-3
against a tolerance of 1e-6.  The criterion is deliberately not changed
for this — a precision floor in it would make a tight float32 tolerance
unreachable and move iteration counts everywhere — so the bound keys
are where it shows: `spectral_error_bound` adds the residual's float
resolution before amplifying it (it read `0.0` there; it now reads
9.4e-2, which covers the 4.2e-3), and `precision_limited` is `True`.
On a slow group, turn on `diagnostics=True` and read those two before
trusting `converged`.  Planned for 0.5.0, not promised by this
release: `strict_convergence` consulting the spectral bound when
diagnostics are on.

The floor is `PRECISION_FLOOR_ULPS = 4` units of `eps · max|field|` in
every entry the norm reads, **per evaluation**, in the norm's units —
`4 m eps √n` under `"l2"` over its `n` entries, `4 m eps / rtol` under
`"mixed"` and `"interface"` (`residual_precision_floor`), each field at
its own dtype's `eps`.  Four is 2.6x the sum of the two measured
sources of one evaluation's rounding: the evaluation error of a dense
update `A @ u + c` near its fixed point (at most 0.72 of a unit over
3 000 random contractions) and the disagreement between two
compilations of the same pass (at most 0.82, the solver-equivalence
sweep).

`m` is how many evaluations one coupling pass rounds like: the largest
sub-cycling divider times `SimulationNode.update_evaluations()` in the
group.  A composite map's error grows with its evaluations: explicit
Euler in `N` sub-steps, each moving its field by less than half an ulp,
is 5.8 units off the exact map at `N = 20` and 29.4 at `N = 100`, and
while the floor was a flat four units a stalled relay built on such a
node read a bound 0.07–0.96x its true distance at `N` = 15–200 with
`spectral_usable=True`.  The framework's own `subcycling=True` is
counted without help; a node whose `update` loops over its own state
must say so:

```python
class SubSteppedNode(SimulationNode):
    def update_evaluations(self):
        return self.n_substeps      # N explicit sub-steps per update()
```

A node that does not declare (`None`, the default) is counted as one
evaluation, and because nothing outside `update` can check that, a
group containing it reports `spectral_usable=False` — and so
`gradient_bound_usable=False` — wherever its residual is at the floor,
where that count *is* the bound.  Above the floor the count carries
less than half of the bound, but a long undeclared loop can still
exceed it; declare it.  The floor models the map's rounding; a node
whose update cancels catastrophically inside itself can exceed it
whatever it declares.

### What `ratio_usable` checks, and what it does not

`error_estimate` sums a geometric series of remaining step lengths.
That sum is a bound on the distance to the fixed point only if **four**
conditions hold, and `ratio_usable` reports **the fourth one alone**:

1. the measure obeys the triangle inequality — **not checked**;
2. `rho` is at least the asymptotic rate — **not checked**;
3. the step scale is the one actually applied — checked only for
   `acceleration="fixed"` and `"none"`, where ω is a constant;
4. the ratio is monotone and finite — **this, and only this, is what
   `ratio_usable` reports**.

So `ratio_usable=True` means: `rho < 1`, the predecessor residual was
non-zero, and every residual in the ratio was finite, therefore the
criterion used the estimate rather than falling back to the raw residual
test.  It does **not** mean the estimate bounds the error.  Each of the
three unchecked conditions is measured broken in this release:

| mechanism | measured understatement | status |
|---|---|---|
| a `rho` read from the mode dominating the *step*, not the remaining error | **122x** on a linear two-mode contraction, modes `(0.999, 0.2)` at `tolerance=1e-4`, with `ratio_usable=True` **and** `converged=True` | characterised, not fixable from the residual sequence: over the passes that matter that sequence and a genuine single-mode decay at 0.2 are the *same sequence* (consecutive ratios 0.2000 and 0.2024).  Needs the spectrum |
| over-relaxation, `acceleration="fixed"` | was 1.97x at ω = 1.95 | **fixed in 0.4.0** — `est/true` now runs 0.992 to 4.53 over an 88-point (gain, ω) grid |
| a step scale the series does not carry | 2.04x under `aitken` (its per-pass factor saturating at its 2.0 clip), 4.5x under `iqn-*` | measured, uncorrected, documented |

`ratio_usable=False` is the honest case, not the alarming one: the
criterion degrades to the pre-0.4.0 raw residual test and says so.  It
is false on a non-monotone sequence, on a zero or non-finite
predecessor, and at `max_iterations=1`, where there is no pair of
residuals to take a ratio of.

Read `error_estimate` as a *better* number than the residual — it is
never smaller than it, and strictly stronger than the pre-0.4.0
criterion in every measured case — and not as a certificate.  Where you
need one, read `spectral_error_bound`.
The full argument, with reproducers, is in
`benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md`; the
standing caveat is `MADD-ANO-005`.

### `spectral_error_bound`: the spectrum, measured

Under `solver="ift"` with `diagnostics=True` the group spends eight
Jacobian-vector products per step — the same `jax.jvp` of the one-pass
map the IFT adjoint solves with — on an Arnoldi iteration at the
returned state, in the coordinates of the group's own norm.  Three
things come out of it: the Ritz spectral radius `rho_spectral`, the
Arnoldi residual `h_{k+1,k}`, and the resolvent norm `‖(I − H)⁻¹‖₂` of
the compressed Jacobian.  A coupling Jacobian's rank is at most the
number of boundary scalars crossing the group's edges, so for a group
with up to eight of them the Krylov space is the whole range, the
non-zero spectrum is exact and `h_{k+1,k}` is zero; for a larger group
the radius is an estimate from below, `spectral_usable` is false, and
the bound carries a margin of `2 h_{k+1,k}` on the radius.

For a *linear* map the error of any iterate is `(A − I)⁻¹` of its
residual — no step sequence, relaxation factor or accelerator enters —
so `residual · ‖(I − A)⁻¹‖` bounds the distance to the fixed point
whatever the iteration did.  `1/(1 − rho)` is that norm for a normal
`A`; the resolvent term is what holds when `A` is not normal, which a
Jacobi loop between a node that responds strongly and one that
responds weakly is measured to be.  Measured `spectral_error_bound /
true distance` (jaxlib 0.11.0, CPU):

| fixture | `error_estimate` | `spectral_error_bound` |
|---|---|---|
| two-mode `(0.999, 0.2)`, gs / none | 0.0082 (the 122x) | 8.05 |
| two-mode, gs / aitken and gs / iqn-ils | 0.50 and 0.0010 | 1.34 and 1.32 |
| random normal contractions, n = 2–6, 80 draws, none and fixed ω ≤ 1 | 0.92–88 | 1.009–96 |
| heterogeneous, jacobi / aitken, 20 steps | 0.008–0.71 | 2.9–95 (911 on the one precision-limited step) |
| heterogeneous, gs / none, 20 steps, all precision-limited | 0.98–2.3 | 3.4–25 |

The 95 is what a rigorous bound on a badly non-normal map costs — the
resolvent norm is the worst direction in the space and the residual is
rarely in it — and it is why the bound is **reported and not applied**:
`converged`, the iteration counts and the recorded sweep rows are
exactly what they were.  On the heterogeneous fixture the float floor of
a 60 000-entry L2 norm, `4 eps √n = 1.2e-4`, is above the tolerance, so
every step that meets it is precision-limited and the floor is most of
the bound (the 911, and every gs / none row); the fixture's nodes do not
declare `update_evaluations()`, so those rows report
`spectral_usable=False`.  Measured on the final tree (jaxlib 0.11.0,
the random draws the property test's generator seeded from numpy); the
earlier recording, 1.47–119 and 0.991–1.93, predated the floor.

Three things make the resolvent term hold where the table's rows did
not test it:

* **the residual is in the space.**  The resolvent norm bounds
  `(I − J)⁻¹ r` only for an `r` inside the invariant space it was
  measured on.  A Krylov space from one start vector is that vector's
  cyclic subspace, which a repeated eigenvalue breaks down early: on
  `A = B ⊗ I₂` it stopped at dimension 2 with a range of dimension 4,
  the zero Arnoldi residual read as "settled", and the bound read
  0.92x the true distance.  The residual is now passed in and the space
  continues from its missing part at every breakdown (1.05x there);
  whatever it never absorbs is reported as unresolved;
* **a dead-banded field stays on the loop.**  The dead band takes a
  field out of the *residual*, not out of the coupling loop.  Weighted
  zero in the spectrum it cut the loop: `rho_spectral` 0.000 for a loop
  gain of 0.9 (a displacement of 1e-9 m feeding a stiffness of 1e9),
  and the bound 0.15–0.29x the kept field's distance.  It now keeps its
  own magnitude's weight in the spectrum, and its share of the residual
  — which `residual` does not contain — is measured and folded into
  the factor (2.8–5.5x there);
* **a non-finite state reports NaN**, not a spectral radius computed at
  a state that has left float range.

What it is not: for a non-linear `F` it is asymptotic (Ostrowski) — exact
to float32 on a log map within tolerance of its fixed point, an estimate
far from one; it is taken in the norm at the returned state, so the dead
band's excluded fields are outside it; and it reads `inf` where
`rho_spectral` (with margin) is at or above one.  `spectral_usable`
reports what the code checked — a finite bound and a settled space — and
not linearity, which nothing checks.

### `gradient_relative_error_bound`: the gradient, not the solve

The IFT adjoint solves `(I − dF/dx)ᵀ λ = ∂L/∂x` at the iterate the
forward *returned*, `x_k`, not at the fixed point `x*`.  With `t_k` the
tangent it returns and `G(x) = J(x) t_k + F_c(x) ċ` the one-pass map's
Jacobian-vector product along it, exactly

    t_k − t* = (I − J(x*))⁻¹ [G(x_k) − G(x*)]

so the error is the resolvent applied to how much the linearisation
moves between the two points.  Each factor is bounded by something the
group already has or measures cheaply:

1. **the distance** `‖x_k − x*‖` is `spectral_error_bound` — not the
   residual and never `error_estimate`, which reads 100x short on a
   hidden slow mode where this bound holds;
2. **the resolvent** is bounded by the factor `spectral_error_bound`
   applies to a residual, the larger of `‖(I − H)⁻¹‖₂` and
   `1/(1 − rho_spectral)`;
3. **the curvature** is a directional second difference of the
   adjoint's own matvec: `G` evaluated by the same Jacobian-vector
   product at `x_k` and at `x_k + δ`, `δ = (I − J)⁻¹ (F(x_k) − x_k)` the
   Newton correction (which supplies the direction only), divided by
   `‖δ‖`.  No Hessian is formed.  Where the residual is at its float
   resolution it carries no direction, and a floor-sized vector's
   resolvent image — the slow mode — stands in;
4. **how far the linearisation reaches**, by Newton–Kantorovich: with
   `h = amp · L · ‖δ‖`, `L` the Jacobian's change along `δ` per unit
   length squared (one more pair of Jacobian-vector products), the
   resolvent at `x*` is at most `amp / √(1 − 2h)` and a fixed point lies
   within Kantorovich's radius, so the bound carries that factor and at
   least that distance — and is `inf`, unusable, at `h ≥ ½`, where
   nothing measured at `x_k` bounds the resolvent at the fixed point.
   On a convex map at `F'(x*) = 0.99`, 0.65–4.5% short, the bound without
   it read 0.20–0.96x the true error with the flag true; `h` there is
   0.48–0.58, so two of those points now read `inf` and two hold at
   3.0–3.5x.  `h` is exactly zero on an affine map.

The bound is `amplification · distance · ‖G(x_k + δ) − G(x_k)‖ / (‖δ‖ ‖t_k‖)`,
relative to the tangent, taken for **one probe per floating constant**
the closure-converted map reads (each parameter, the pre-step states,
the outside states it reads) and reported for the worst.  Per constant,
because a combined direction can cancel: one probe over every constant
read 0.0 on the stiff spring pair while its stiffness gradient was
0.8–4.8% off — the random signs moved each node's stiffness and mass by
the same relative amount, and the dynamics see only their ratio.  The
tangents and `δ` come from a Woodbury solve on an eight-vector basis of
the Jacobian's range (`jacobian_range_basis`, `resolvent_apply`), so
the cost is `11 + k + 4 n_c` Jacobian-vector products per group per step
(`k ≤ 8`, `n_c` the floating constants) beside the spectral bound's
eight — which is why it shares its gate.

Measured `bound / true` (jaxlib 0.11.0, float32), the fixed point's
gradient from tight `ift` and `fori` arms that agree, every point a
fresh graph stepped through `gm.step()` and stopped early by
construction:

| fixture | `bound / true` |
|---|---|
| concave `a + g log(1 + u)`, caps 3–8 (26% → 0.2% from `x*`) | 1.21–2.37 for `d/da`, 9.4–11.5 for `d/dg` |
| convex `a + g u²`, caps 3–8 (6.8% → 0.3%) | 1.29–1.36 for `d/dg`, 3.57–3.80 for `d/da` |
| affine `a + g u`, `d/dg` (`d/da` is exact) | 1.81 at every cap |
| stiff spring pair, stiffness and mass, caps 2–6 | 7–11 |
| two-mode, concave slow mode, `converged=True` | 83 (with `error_estimate`'s distance: 12x short) |

The parameter with the larger relative error reads near the product of
the two conservative factors (the distance 1.1x, the relay's resolvent
1.22x) from cap 4 on, and more at cap 3 (2.37 on the concave map),
where the Newton–Kantorovich factor below is largest; the other reads
its gap to the worst probe as well.

**What it is not — read this before using it.**  It is a statement
about the gradient, not about the solve.  On a map affine in its state
with additive parameters the IFT gradient is the fixed point's from
*any* iterate, so the bound truthfully reads **0.0** while the state is
far off: on the two-mode case it is 0.0 while the state sits 1.1e-2
from the fixed point with `converged=True`, and on the stiff spring pair
under `iqn-ils` with the interface norm and explicit
`accelerated_fields` it is 3.1e-7 — zero to float32 — while the
velocities are 1.7% off.  The
returned `(value, gradient)` pair is then **mutually inconsistent** —
the gradient is `d(fixed point)/dθ`, the value is not the fixed point —
and this key cannot say so.  For the health of the solve read
`spectral_error_bound`, within its norm: under the interface norm it
covers the interface fields only and on that spring pair reads 9.0e-3.
Beyond that: it inherits every condition of `spectral_error_bound`; it
is leading-order in the distance (the curvature is measured over `δ` and
extrapolated linearly, and `h` checks the Jacobian's variation along `δ`
only); a field-valued constant is probed along one
random direction; and it is relative to the tangent's norm, so a scalar
loss whose gradient nearly cancels across the state can carry a larger
relative error.  `gradient_bound_usable` reports a finite bound and a
settled spectrum, not those conditions.  It is not spelled
`gradient_error_bound`: that spelling is the deprecated alias of
`gradient_error_estimate` (below), and code written against 0.3.x
would read a new meaning under it as the old number.

### Waveform relaxation: which sweep each field describes

A sub-cycling group with `waveform_iterations=N` runs `N` sweeps per
step, and each sweep is a whole fixed-point solve with a budget of
`max_iterations` passes of its own.  Every sweep iterates the same
one-pass map, starting from where the sweep before it stopped: the
residual the second sweep measures on its first pass is, bit for bit,
the residual the first sweep reported for the state it handed on.  So:

- `iterations` is the **largest** sweep's count and `total_iterations`
  the sum.  The cap check `iterations >= max_iterations` is exact: true
  when, and only when, some sweep exhausted its budget.
- Every other field is the **last** sweep's, which produced the
  returned state: `residual` recomputed on that state reproduces it, and
  `converged` is the verdict on it.  Under `solver="ift"` the gradient
  is the last sweep's too, because the implicit-function derivative
  ignores the initial guess.  An earlier sweep that stopped at the cap
  shows in `iterations`, not in `converged`.

Measured on the sub-cycled spring pair of
`tests/core/test_phases_5_7_8.py` (timesteps 0.001 / 0.01,
`tolerance=1e-8`, `waveform_iterations=3`), first step, jaxlib 0.11.0,
CPU; both solvers give the same numbers:

| `max_iterations` | acceleration | passes per sweep (converged) | `iterations` before the fix | `iterations` / `total_iterations` |
|---|---|---|---|---|
| 10 | none | 3 (yes) · 1 (yes) · 1 (yes) | 1 | 3 / 5 |
| 2 | none | 2 (no) · 1 (yes) · 1 (yes) | 1 | 2 / 4 |
| 10 | fixed, `relaxation=0.7` | 10 (no) · 1 (yes) · 1 (yes) | 1 | 10 / 12 |

Before 0.4.0 `iterations` was the last sweep's count, which is the
fourth column (MADD-ANO-026).  `strict_convergence=True` under
`solver="ift"` checks every sweep, so the second and third rows raise
there although their last sweep converges.

### The old names

`bound_valid` and `gradient_error_bound` still read through 0.4.x and
emit a `DeprecationWarning` naming their replacement; they are removed
in 0.5.0.  They are **not** in `keys()`, so `dict(diag)` and anything
that records the report carry only the new names.

## How the numbers were produced

`benchmarks/coupling_fixtures.py` defines eight graph shapes, each
isolating one property:

| fixture | property | built from |
|---|---|---|
| `chain-N`, N ∈ {2,5,20,50} | sequential depth | spring-damper nodes in a line |
| `star-N`, N ∈ {2,4,8,16} | width with no leaf-to-leaf path | hub plus independent leaves |
| `ring-N`, N ∈ {4,8,16} | a cycle with no natural first node | bidirectional ring |
| `stiff-pair-G`, G ∈ {0.25,0.5,0.8,0.95,1.2} | contraction factor, weak to divergent | two mutually anchored nodes |
| `expensive-pair` | cost per iteration | two 10⁵-cell heat grids |
| `heterogeneous` | one expensive node among cheap ones | 6x10⁴-cell grid plus four scalars |
| `mixed-modes` | two groups, two schedules | Gauss-Seidel chain + Jacobi star |
| `slow-drift` | a fixed point that barely moves | heat pair relaxing a step discontinuity |

The spring fixtures are parameterised by a dimensionless **coupling gain**
`g` rather than a stiffness.  `SpringDamperNode` integrates with
semi-implicit Euler using its own old position and its anchor's new one,
so the derivative of its new position with respect to its anchor is
`dt²k/m`; fixing that at 1 and putting the coupling strength in the edge
weight makes `g` exactly the gain the fixed-point iteration sees, with
the remaining stiffness acting as a spring to ground.  Two mutually
coupled nodes then contract as `g` under Jacobi and `g²` under
Gauss-Seidel, and the line and ring shapes add the usual
`cos(π/(N+1))`-type factor.  The heat fixtures use the Fourier number
`Fo = α·dt/dx²` the same way.  The module docstring explains why the
damping has to be a function of `g` and why every spring fixture carries
a driver node outside the group.

`benchmarks/bench_coupling_sweep.py` runs the product of two iteration
modes, five accelerations (`fixed` at ω = 0.5 and 0.8) and two
convergence norms over each fixture, recording per row: mean / median /
p95 step time, the dispatch floor, iterations mean and max against the
cap, the fraction of steps that converged, the final residual, a
fingerprint of the resulting state, and the device trace where the
platform provides one.

```
# 420 rows, ~14 min: every fast fixture
JAX_PLATFORMS=cpu python benchmarks/bench_coupling_sweep.py \
    --json benchmarks/results/coupling_sweep_cpu.json
# 96 rows: the two grid fixtures, plus accelerated_fields variants
JAX_PLATFORMS=cpu python benchmarks/bench_coupling_sweep.py \
    --fixtures expensive-pair,heterogeneous --fields \
    --json benchmarks/results/coupling_sweep_expensive_cpu.json
# 48 rows, ~2 min: the same graphs with a tighter iteration cap
JAX_PLATFORMS=cpu python benchmarks/bench_coupling_sweep.py \
    --fixtures chain-20,chain-50,star-16,stiff-pair-0.8 \
    --norms l2 --max-iterations 16 \
    --json benchmarks/results/coupling_sweep_cap16_cpu.json
```

The grid fixtures are opt-in (`--include-slow`, or named explicitly)
because they are minutes rather than seconds.  The third run exists
because `max_iterations` is not a free safety margin for IQN — see
[What IQN costs](#what-iqn-costs).

`--steps` changes how many timings are averaged and nothing else.  The
iteration counts, convergence fractions and residuals come from a
separate statistics pass over a window that belongs to the *fixture*,
so two runs of the same fixture at different `--steps` differ only in
their timings.  That was not true of the first recording of these
files: the pass ran `min(n_steps, 50)` steps from wherever the timed run
stopped, so a shortened run moved the window as well as the sample
size, and on fixtures driven by a 44-step oscillator the mean iteration
count moved with it — `jac/iqn-imvj5/l2` on `chain-5` reads 4.00 at ten
timed steps and 3.00 at twenty.  An earlier version of this page said
those three quantities were unaffected by `--steps`; they were not, and
the grid file was the one recorded at ten.  It has been re-recorded at
the fixtures' own twenty.

### Platform caveat — read this before quoting a step time

Every recorded row was measured on a **24-core laptop CPU**; the venv has
no CUDA jaxlib, so `jax.devices()` reports `TFRT_CPU_0` and
`jax.profiler` reports no device kernels at all.  That has three
consequences.

* `n_kernels_per_step` and `device_busy_fraction` are **zero in every
  recorded row**.  The `regime` column falls back to
  `dispatch_floor_ms / mean_step_ms`, which is a defensible CPU proxy
  and is labelled as such in `regime_from`.  On a GPU the sweep uses the
  device-busy fraction instead, with no change to the script.
* **Nothing here measures Jacobi's parallelism.**  Jacobi's structural
  advantage is that its node updates are independent and can run
  concurrently; on one CPU device they run one after another, exactly
  like Gauss-Seidel's.  A Jacobi row's *iteration count* is exact and
  platform-independent — that is a property of the algorithm — but its
  *step time* is a pessimistic upper bound, because the concurrency the
  mode exists for is not available.  The open question, and the first
  thing to re-measure on a GPU box, is whether a wide `star-N` or a
  sharded grid recovers Jacobi's 1.8x iteration penalty in wall time.
  Nothing in this document answers it either way.
* On the launch-bound fixtures the whole step is 0.1–2 ms and the spread
  between configurations is tens of microseconds — inside the noise.
  **The iteration counts are the reliable signal there**, and the sweep's
  own "best configuration" picker ties rows whose medians differ by less
  than the dispatch floor — on `star-8` that floor is 0.217 ms against a
  0.288 ms step — or than the rows' own median sampling spread,
  whichever is larger, and breaks the tie on iteration count.  It used a
  fixed 10% band before, well inside the within-row spread these rows
  actually record (up to 84% from median to p95), so it excluded rows
  that were not measurably slower and named an 18.6-iteration
  configuration as `star-8`'s best over a 7.1-iteration one.  Every
  `best` field in the recorded files comes from the corrected rule, and
  each one also records the fewest-iterations row, so a fixture where
  the two disagree says so.

## Where the measurement contradicted the theory

Four expectations that the sweep did not support.  Each was written down
before the fixtures were built, which is the only reason they count as
predictions rather than commentary.

**1. The Gauss-Seidel advantage does not widen with chain length.**  The
expectation was that sequential information flow would let Gauss-Seidel
pull further ahead as the chain grew.  It does not:

| fixture | Gauss-Seidel | Jacobi | ratio | ratio (re-measured `c51cd6a`, 2026-09-19) |
|---|---|---|---|---|
| `chain-2` | 5.6 | 9.4 | 1.69 | 1.83 |
| `chain-5` | 13.4 | 23.6 | 1.77 | 1.88 |
| `chain-20` | 23.7 | 40.7 | 1.72 | 1.79 |
| `chain-50` | 30.8 | 53.0 | 1.72 | **1.59** |

The ratio was flat because a tridiagonal coupling operator is
*consistently ordered*, for which the Gauss-Seidel spectral radius is
exactly the square of the Jacobi one — a constant factor of two in
iterations at any depth, not a growing one.  What grows with N is the
radius itself (0.40 → 0.80), and so the absolute iteration count.

The last column is why row 1 of "Start here" now says 1.6–2.0x.  The
prediction this paragraph tested — that the advantage *widens* with
depth — is still refuted, and more firmly: at `c51cd6a` the ratio
*falls*, from 1.88 at N = 5 to 1.59 at N = 50.  The flatness the
explanation above accounts for is what did not survive.  Whether the
difference is the operator or the error-bound criterion charging the two
modes differently has not been measured.

**2. The star is one of the shapes where Gauss-Seidel wins by the most,
not the least.**
The expectation was that leaves which cannot see each other would leave
Gauss-Seidel's ordering nothing to exploit, making Jacobi competitive.
Measured, the star family shows the *largest* Gauss-Seidel advantage of
any shape family — 1.83–1.85 against 1.69–1.77 on the chain, with only
`stiff-pair-0.8` at 1.83 reaching into the same band — and it is flat in
width (20.5 iterations at 2 leaves, 22.9 at 16).  The hub-to-leaf
dependency alone is enough to give the full squared radius; ordering
*among* the leaves was never what produced the factor.  On one device
there is no shape at which Jacobi is competitive on iterations.

Re-measured at `c51cd6a` on 2026-09-19, the *flat in width* half holds:
1.92–1.96 across the four stars, 23.1 iterations at 2 leaves and 24.8 at
16.  The **"largest of any shape family" half does not** —
`stiff-pair-0.8` measures 1.98 and `slow-drift` 2.04, both above every
star.  What survives is the finding the paragraph was written to test:
a star gives Gauss-Seidel the full squared radius, so leaves that cannot
see each other were never what produced the factor.  The ranking against
the other families was a by-product of that, and it no longer holds.

**3. Aitken helps Gauss-Seidel more than Jacobi, except on a two-node
pair.**  The expectation was the reverse: Jacobi's error history decays
more cleanly, which is what Aitken's scalar relaxation assumes.  That
holds where the coupling really is a single mode —

All four rows re-measured at `c51cd6a` on 2026-09-19; the recorded
2026-09-18 values are in brackets.

| fixture | gs/none | gs/aitken | jac/none | jac/aitken |
|---|---|---|---|---|
| `stiff-pair-0.5` | 8.2 (7.8) | 4.0 (4.0) | 16.2 (13.9) | 5.3 (5.0) |
| `stiff-pair-0.8` | 22.3 (20.8) | 10.0 (9.9) | 44.1 (38.0) | 23.0 (20.7) |
| `star-8` | 24.1 (21.7) | 10.1 (10.4) | 46.8 (40.2) | 38.3 (37.4) |
| `star-16` | 24.8 (22.9) | 10.4 (10.7) | 47.6 (42.4) | 42.1 (42.5) |

— on `stiff-pair-0.5` Aitken very nearly erases the Gauss-Seidel
advantage, taking a 2.0x gap down to 1.3x, and on `chain-2` it closes
it outright (4.0 against 4.3).  But as soon as the error is a mixture of
modes with comparable magnitudes, the single Aitken ω cannot cancel them
and Jacobi's cleaner decay stops helping: on `star-16` Aitken removes
**12%** of Jacobi's iterations against **58%** of Gauss-Seidel's
(47.6 → 42.1 against 24.8 → 10.4), and on `star-8` 18% against 58%
(re-measured `c51cd6a`, 2026-09-19).

The old wording here was stronger and no longer holds: it said Aitken
buys Jacobi "nothing at all" on `star-16`, on the recorded 42.4 → 42.5.
That figure is gone — 42.1 from 47.6 is a real reduction, and 12% is
not nothing.  The finding this paragraph exists to report survives on
the *ratio*, which is what it was always about: Aitken is worth three
to five times as much to Gauss-Seidel as to Jacobi on a wide star
(58%/18% at 8 leaves, 58%/12% at 16), and the gap widens with width.
One caveat on the Jacobi column: `star-16` `jac/aitken/l2` converges on
97% of steps, not 100%, so part of its lower count is steps that
stopped without arriving.

**4. Fixed under-relaxation never helps Gauss-Seidel, and helps Jacobi
on two fixtures out of twenty.**  The prediction was that it would lose
everywhere, and the shape of where it does not is more useful than the
prediction was.  Across all twenty fixtures, both norms and both ω
values, **not one Gauss-Seidel row beats its unrelaxed counterpart on
iterations** — it usually roughly doubles them (`chain-20`:
23.7 → 51.3; `star-16`: 22.9 → 49.4; `slow-drift`: 3.2 → 4.3).  Every
row where relaxation wins is a Jacobi row, and there are two of them:

| fixture / norm | ω = 1 | ω = 0.5 | ω = 0.8 | ω = 0.8 (re-measured `c51cd6a`, 2026-09-19) |
|---|---|---|---|---|
| `slow-drift`, jacobi, L2 | 6.14 | 4.30 | **3.46** | **3.62**, 100% converged |
| `slow-drift`, jacobi, interface | 6.40 | 4.54 | **3.84** | **4.00**, 100% converged |
| `expensive-pair`, jacobi, interface | 2.45 | 2.20 | **2.00** | **[not re-evidenced]** — the grid file was not re-run |

`jacobi`/`fixed` ω = 0.8 / L2 is the sweep's own best configuration for
`slow-drift`, and the `expensive-pair` row is the more interesting of
the two because that fixture is compute-bound, so its 3.33 → 2.60 ms is
a real 22% and not dispatch.

**The scope of this finding is narrower than it looks, and narrower
than it was when it was written.**  `jac/fixed0.8/l2` is recorded at
100% converged on every fixture in the table above and on eleven more.
At `c51cd6a` it converges on 20% of `star-16` steps, 40% of `star-8`,
82% of `star-4`, 88% of `star-2`, 92% of `stiff-pair-0.8` and 58% of
`chain-20`, against 100% (92% for `chain-20`) in the 2026-09-18
recording — the largest single block of the 18 rows that stopped
converging across the release.  The `slow-drift` recommendation itself
stands: it still converges every step, at 6.80 → 3.62.  Read this
finding as "on a graph whose fixed point barely moves, measure both
ω values", never as "ω = 0.8 is a safe default under Jacobi".

The split is the textbook one, and it is the same sentence as the next
paragraph read the other way.  Under-relaxation damps an iteration that
*overshoots*.  A Jacobi iteration matrix on these shapes has eigenvalues
of both signs, so its error alternates and overshoots; Gauss-Seidel's is
the square of it, positive, so the error approaches monotonically and
damping it only slows the approach.  Read that way, "leave `relaxation`
alone" is advice about Gauss-Seidel, and under Jacobi on a graph whose
fixed point barely moves it is worth measuring both ω values.

What under-relaxation does *not* do, under either mode, is rescue a
divergent group.  This is arithmetic, not bad luck: relaxing maps an eigenvalue λ
to `1 - ω + ωλ`, which for a *positive* λ > 1 stays above 1 for every
ω > 0.  `stiff-pair-1.2` has a Gauss-Seidel eigenvalue of +1.44 and
neither ω = 0.5 nor ω = 0.8 converged a single step of it.
Under-relaxation is the cure for an eigenvalue outside the unit circle on
the *negative* side, i.e. for an oscillating divergence; it is not a
general stabiliser, and it should not be reached for first.

## The finding that matters most

**IQN converges problems that no fixed-point method can.**  On
`stiff-pair-1.2`, whose coupling gain is past the convergence limit by
construction.  Straight from the recorded rows of
`coupling_sweep_cpu.json` (Gauss-Seidel, L2 norm, the fixture's 5 warmup
+ 10 timed + 10 statistics steps — it is deliberately short, because a
divergent group's state grows by ρ^cap every step):

| acceleration | iterations | at cap | converged | \|a.position\| |
|---|---|---|---|---|
| `none` | 11.0 | 100% | 0% | inf |
| `aitken` | 11.0 | 100% | 0% | 9.0e+03 |
| `fixed` ω=0.5 | 11.0 | 100% | 0% | 5.5e+11 |
| `fixed` ω=0.8 | 11.0 | 100% | 0% | 3.5e+17 |
| `iqn-ils` | 4.0 | 0% | 100% | 0.496 |
| `iqn-imvj`, reuse 5 | 3.0 | 0% | 100% | 0.496 |

One reading note: once the residual is not finite the loop condition
`res > threshold` is false and the pass count stops rising, so a longer
run does not show a larger `iterations` — the honest signal is
`converged`, which is 0% for the whole fixed-point family and 100% for
both quasi-Newton rows.

IQN-ILS is a quasi-Newton root solver, not a contraction, so a spectral
radius above one is not fatal to it; it solves the coupled algebraic
system directly.  This is why the invariant is phrased as "a divergent
group must *report itself* unconverged" rather than "must fail": the
fixed-point family must be honest about hitting the cap, and IQN is
allowed to succeed where they cannot.  Both are asserted in
`tests/core/test_coupling_fixture_invariants.py`.

The same effect shows at gain 0.95, which is inside the convergence
limit but too slow to be useful: `gs/none/l2` exhausts its cap of 60 on
98% of steps and converges on 2% of them, while `gs/iqn-ils/l2`
converges every step in 4.1 iterations and `gs/iqn-imvj` in 3.4.
`gs/aitken/l2` also converges every step there, in 38.0 — so at this
gain the choice is between an accelerator and no useful answer, not
between two speeds.  IQN is not *faster* here (0.21 ms against 0.10 for
the row that does not converge); it is the one that is right.

One property of Aitken worth knowing before reaching for it: since the
correction it needs the convergence threshold met on **two consecutive
passes**.  The mechanism is real — `_TWO_PASS_EXIT` in
`core/graph_manager.py`, argued in `_fixed_point_while`'s docstring —
but an earlier version of this paragraph drew the wrong floor from it
and said Aitken "cannot exit in fewer than four".  **It can, and the
repo's own recorded baseline says so.**  The streak's first member is
the pass that ran *before* the group's iteration loop (`first_res`,
which is exactly why it is seeded from a measurement rather than from
infinity), so the guard costs at most one extra pass, not three.  Seven
Aitken rows in `benchmarks/results/coupling_sweep_cpu.json` record an
`iterations_min` of 2 or 3 — `star-4` and `stiff-pair-0.95` at 2 under
both iteration modes, `ring-4` and `stiff-pair-0.25` at 3 — and all
seven are **interface-norm** rows.  Re-measured at `c51cd6a` on
2026-09-19, `star-4 gs/aitken/interface` still exits some steps in 3,
while `gs/none/interface` on `stiff-pair-0.95` exits some in 1: 2 is
the lowest Aitken count anywhere in the sweep against 1 for `none`,
`fixed` and IQN, which is the single pass the guard is specified to
cost.

What the old sentence generalised from is the L2 rows it sampled, where
Aitken does land on 4.0: on a group that already converges in three it
costs a pass — `slow-drift` goes 3.3 → 4.0 and `stiff-pair-0.25`
3.6 → 4.0.  Budget one pass over the unaccelerated exit, not a floor of
four.  It is an accelerator for iterations you have, not for iterations
you do not.

## What IQN costs

IQN's iteration counts are the best of any option on nearly every
fixture.  Its *time* is not, and the reason is worth stating precisely:
each iteration solves a least-squares problem with `jnp.linalg.pinv`
over an `n_dof x max_cols` matrix, where `n_dof` is the number of
accelerated degrees of freedom and **`max_cols = max_iterations - 1`**.

| fixture | accelerated DOFs | gs/none | gs/iqn-ils | factor |
|---|---|---|---|---|
| `chain-5` | 5 | 0.194 ms | 0.71 ms | 3.7x |
| `chain-20` | 20 | 0.744 ms | 2.15 ms | 2.9x |
| `chain-50` | 50 | 1.41 ms | 169 ms | 120x |
| `slow-drift` | 4 000 | 0.456 ms | 8.94 ms | 20x |

`chain-50` is the cautionary row: 16.9 iterations against 30.8, and a
step 120 times slower for them.  That factor is also the least
reproducible number on this page — the same row has measured 107 ms and
332 ms on earlier runs of code that differed in nothing that touches it,
because XLA's CPU SVD inside a `while_loop` is erratic.  Treat it as
"one to two orders of magnitude", not as 120.  Two levers, in order:

**1. Lower `max_iterations`, but size it from a measurement.**  A cap of
60 gives IQN 59 secant columns whether or not it ever needs them, and
the least-squares problem is that wide every pass.  Re-running the same
graphs with `--max-iterations 16` (`coupling_sweep_cap16_cpu.json`):

| fixture / config | cap 60 | cap 16 |
|---|---|---|
| `star-16` `gs/iqn-ils/l2` | 4.0 it, 1.33 ms, 100% converged | 4.0 it, 1.55 ms, **100%** |
| `chain-20` `gs/iqn-ils/l2` | 14.1 it, 2.15 ms, 100% | 13.9 it, 1.85 ms, **86%** |
| `chain-50` `gs/iqn-ils/l2` | 16.9 it, 169 ms, 100% | 14.9 it, 2.93 ms, **23%** |
| `chain-50` `gs/iqn-imvj5/l2` | 17.5 it, 41.8 ms, 100% | 15.0 it, 3.05 ms, **13%** |

Read the third column, not the second.  `star-16` converges in 4
iterations, so 59 columns were pure overhead and taking them away costs
nothing — that is the case the lever is for.  `chain-50` needs 16.9, so
a cap of 16 cannot hold it: it sits at the cap on 93–100% of steps and
converges on 13–23% of them.  The 58x saving on that row is real and it
is bought by not converging.  **`max_cols = max_iterations - 1`, so the
cap is simultaneously the secant history length and the iteration
budget, and it is a sizing parameter in both directions** — lowering it
shrinks the least-squares problem *and* the number of passes available
to use it.

The unaccelerated rows show the same boundary more bluntly.  At cap 16,
`gs/none/l2` converged on **0%** of `chain-50` and `star-16` steps and
2% of `chain-20` and `stiff-pair-0.8` — it needs 21–31.  Set the cap
from the iteration count you measured, not from the one you hoped for,
and read `converged_fraction` afterwards rather than the step time.

**2. Set `accelerated_fields` explicitly** so a large-state node does not
put its whole array into the least-squares problem.  `None`
auto-detects the fields the group's internal edges *read*, which for
a grid node coupled on one cell is still the entire grid.

## Four combinations that theory says should win

**Jacobi + Aitken.** Nearly closes the gap on two-node pairs
(`stiff-pair-0.5`: 16.2 → 5.3 against Gauss-Seidel + Aitken's 4.0;
`chain-2`: 11.1 → 4.3 against 4.0) and buys far less on anything wider
(`star-16`: 47.6 → 42.1, a 12% reduction, against Gauss-Seidel's 58%)
(re-measured `c51cd6a`, 2026-09-19).  Not a general recommendation.  An earlier version of
this line read "does nothing on anything wider (`star-16`: 42.4 →
42.5)"; see contradiction 3 for why that figure no longer stands.

**Jacobi + IQN.** Marginally better than Gauss-Seidel + IQN on the
chain and ring families (`chain-5` 5.3 vs 7.0, `chain-20` 13.2 vs 14.1,
`ring-4` 5.0 vs 5.9, `chain-50` 16.7 vs 16.9) and marginally worse on
the stars (`star-8` 4.2 vs 4.0, `star-16` 4.5 vs 4.0).  The hypothesis — a residual history from a single
iterate spans a cleaner space — is weakly supported on the chains and
rings only, and the effect is smaller than the noise in step time.  Worth knowing, not
worth defaulting to.

**Gauss-Seidel + fixed under-relaxation.** Never won.  See contradiction
4 above.

**Interface norm with `accelerated_fields` restricted to interface DOFs.**
Half right, and the half that is wrong is the more useful half.

The interface norm alone generalised from AR4.  Measured on
`gs/none` across the eighteen fast fixtures it removes **−9% to 28%** of
the iterations (re-measured `c51cd6a`, 2026-09-19): 18–29% on the chains, the stars and the
rings, 20–28% on the stiff pairs up to gain 0.8, **2%** on
`stiff-pair-0.95`, nothing at all on `stiff-pair-1.2` (which converges
nowhere), and −9% on `slow-drift`, where it costs a third of an
iteration.  The 2026-09-18 recording read −5% to 31% and this paragraph
quoted that; both endpoints have moved outside it.  A blanket "25–35%"
was the top of the range quoted as the whole of it.  On a grid fixture, where the
L2 residual is mostly bulk change that does not iterate, the cut is far
larger: `expensive-pair` goes 6.0 → 1.5, a 75% reduction.

| fixture | l2 (2026-09-18) | interface (2026-09-18) | l2 (re-measured `c51cd6a`, 2026-09-19) | interface (re-measured `c51cd6a`, 2026-09-19) |
|---|---|---|---|---|
| `chain-20` | 23.7 | 16.7 | 26.4 | 18.9 |
| `chain-50` | 30.8 | 23.0 | 33.8 | 26.0 |
| `star-16` | 22.9 | 15.9 | 24.8 | 18.2 |
| `stiff-pair-0.95` | 58.4 (at cap on 98% of steps, converged 2%) | 49.6 (at cap on 34%, converged 68%) | 59.0 (at cap on **100%**, converged **0%**) | 57.7 (at cap on **94%**, converged **8%**) |

**The last row reversed qualitatively and is the one to read.**  On the
2026-09-18 recording the interface norm took `stiff-pair-0.95` from 98%
at-cap to 34% — a 15.1% cut in iterations and a group that converged on
two steps in three.  At `c51cd6a` it takes it from 100% at-cap to 94%,
a **2.2%** cut, and the interface arm converges on **8%** of steps.
Near the convergence limit the interface norm is no longer the lever
this table was written to show; `stiff-pair-0.95` is the fixture behind
"Start here" row 3, which is why that row now recommends IQN on its own
merits rather than on this one.

"At no measurable cost" is also not quite true, and in the direction you
would not guess: nine of the eighteen fast fixtures record a *higher*
step time under the interface norm than under L2 (`ring-4` 0.144 → 0.179 ms,
`stiff-pair-0.95` 0.100 → 0.173 ms), while others record a lower one
(`chain-20` 0.744 → 0.594 ms).  On launch-bound fixtures those
differences are dispatch, not work.  The iteration reduction is the real
effect; the step time is a coin flip at this scale.

Combined with IQN it is better still, and there the saving is not a coin
flip (`chain-50`: 16.9 iterations and 169 ms under l2, 11.4 iterations
and 51 ms under the interface norm — the smaller residual vector also
shrinks the least-squares problem).

The `accelerated_fields` half did not generalise the way it was
expected to.  The expectation was that restricting the quasi-Newton
problem to the *expensive* node's interface would beat accelerating
everything.  On `heterogeneous` — one 6x10⁴-cell grid plus four scalar
nodes — restricting it to the expensive node is the **worst** of the
four options, and restricting it to the cheap ones is the best by two
orders of magnitude (Gauss-Seidel, `iqn-ils`, L2 norm):

| `accelerated_fields` | accelerated DOFs | iterations | ms/step |
|---|---|---|---|
| `None` (auto: grid + scalars) | ~6x10⁴ | 4.0 | 177 |
| everything, incl. velocities | ~6x10⁴ | 3.0 | 121 |
| the expensive node only | 6x10⁴ | 6.7 | 300 |
| **the four cheap nodes only** | **4** | **4.0** | **2.10** |
| (no acceleration, for scale) | — | 6.7 | 2.13 |

Two comparisons, stated separately because the earlier version of this
page mixed them: cheap-only matches the **auto-detected** set on
iterations exactly (4.0 against 4.0) at **1/85th** of its cost, and is
one iteration behind accelerating **everything** (4.0 against 3.0) at
**1/58th** of its cost.  Neither ratio is 1/60 or 1/76 paired with
"same iteration count"; those were an iteration claim from one pair and
a cost ratio from another.

The reading is that the secant history needs enough degrees of freedom
to model how the *interface* responds, and the four scalars carry that;
the grid's sixty thousand cells add nothing to the model and the entire
cost.  Restricting to the grid alone removes the degrees of freedom that
were doing the work.

Two caveats the recommendation needs.  First, the win is over IQN's own
cost, not over doing nothing: cheap-only `iqn-ils` at 2.10 ms is a
wash against `gs/none/l2` at 2.13 ms, and it is `iqn-imvj` under the
interface norm — 3.0 iterations at 1.78 ms against `gs/none/interface`'s
5.4 at 2.02 — that actually wins, by 12%.  The point of
`accelerated_fields` here is that it makes IQN *affordable* on a
heterogeneous group, not that IQN is a large win on one.

Second, **under Jacobi the same restriction costs convergence.**  All
four Jacobi cheap-only rows fall short:

| row | iterations | converged |
|---|---|---|
| `jac/iqn-ils/l2/fields-cheap` | 16.0 | 60% |
| `jac/iqn-imvj5/l2/fields-cheap` | 15.8 | 65% |
| `jac/iqn-ils/interface/fields-cheap` | 13.6 | 75% |
| `jac/iqn-imvj5/interface/fields-cheap` | 13.6 | 75% |

against 100% for every one of the corresponding Gauss-Seidel rows and
for the Jacobi rows that accelerate everything.  Restricting the secant
basis to four scalars leaves Jacobi without enough of the interface
response to model, and unlike Gauss-Seidel it has no sequential update
to make up the difference.

The corollary for a grid-to-grid group, where there is no cheap subset
to fall back on: IQN is simply not affordable.  On `expensive-pair` it
costs **91–148x** the plain step for 0.05 of an iteration under the
interface norm, and under the L2 norm `gs/iqn-ils` converged on only
75% of steps.

Third, and this one is an accuracy caveat rather than a cost one:
**do not pair the auto-detected `accelerated_fields` with
`convergence_norm="interface"`.**  Both are the edge source fields, so
the quasi-Newton step lands on exactly the fields the criterion then
measures, and every other field of the group is carried out of the last
raw pass with nothing looking at it.  Measured over the sweep, on an
`iqn-*` row exiting on its criterion, the interface field moves by at
most 1.0e-04 while `velocity` moves by up to **2.2**.  This is what the
ten `_KNOWN_DISAGREEMENTS` rows in
`tests/core/test_coupling_fixture_invariants.py` are; naming every
field in `accelerated_fields` closes all ten at the same iteration
count, and tightening `atol`/`rtol` — the remedy those entries first
named — does not, because the criterion is already three decades inside
its threshold when they stop.  Naming every field is necessary and not
sufficient: the criterion is still over the edge fields alone, so a
group whose interface goes stationary before the rest of its state has
can still exit, and over generated graphs that is measurable at
5.2e-01.  **Prefer a norm that sees the whole state** (`"l2"`,
`"mixed"`);
`benchmarks/results/retire_known_disagreements/REPORT.md` has the
numbers, and `MADD-ANO-005` carries it as residual risk.

## Do all these configurations agree?

The sweep records `fixed_point_agreement` per fixture: the largest
deviation between the state fingerprints of any two fully converged
rows, measured against the first of them.  Two numbers, because one is
not enough on a mixed-scale fixture — `max_relative_deviation` scales
each entry by its *field's* largest magnitude across the group, and
`max_node_relative_deviation` scales it by that entry's own amplitude,
which is the one that catches a small node hiding behind a large one
sharing its field name.

Across all twenty fixtures the field-scaled figure runs from 1.0x10⁻⁷
(`slow-drift`) to 4.8x10⁻³ (`mixed-modes`), and every fixture's worst
row is an interface-norm one.  That is the expected shape, but not for
the reason an earlier version of this page gave.  It said the interface
norm "is a *relative* criterion (`atol`/`rtol`), so it stops earlier
than an absolute L2 tolerance".  That sentence pre-dates 0.4.0: **all
three norms are relative now** — each divides a field's change by that
field's own magnitude — so relativeness is not what separates them, as
the header of this page and `CouplingGroup`'s docstring
(`src/maddening/core/coupling/group.py`) both say.  What separates them
is *what they look at*.  The interface norm measures only the
coupling-edge fields, so a group whose interface has gone stationary
stops while the rest of its state is still moving, and the trajectories
drift correspondingly further apart.  The node-scaled figure agrees with it everywhere except
`chain-50`, where it reports 6.0x10⁻² for `link30.velocity` — a node
passing near zero, measured against its own small amplitude.  That is
what the second number is for, and it is also why the fixture-level
threshold in the test suite is per-norm rather than one band for both.

Two notes on how these figures got here, because the earlier version of
this page quoted a different one.  The headline used to be 4.2x10⁻² on
`heterogeneous` — a number that was **not** a disagreement of 4.2x10⁻².
It was the *sum* of a 60 000-cell grid field: an extensive quantity
compared against a scale taken from an intensive one, so a per-cell
difference of 7x10⁻⁷ scored 4x10⁻².  And the scale it was divided by
came from the driver node, which sits outside every coupling group and
carries a `position` eighty times the coupled probes', so the probes'
own disagreement — a real 20.6%, caused by the Aitken exit criterion
described at the top of this page — was reported as 2.6x10⁻³ and passed
under a 5x10⁻³ threshold.  Both are fixed: the signature records each
entry's element count and the comparison divides it out, and the
signature covers only the nodes inside a coupling group.
`heterogeneous` now reports 1.8x10⁻³, from `jacobi/fixed` ω = 0.8 /
`interface`, and the Aitken rows are unremarkable.

The practical advice is unchanged and now rests on the right numbers:
the deviation grows with the fixture's condition number and with how
early the criterion lets you stop, so set `rtol` deliberately rather
than inheriting it, and read `fixed_point_agreement` for your own graph
rather than trusting a figure from this page.

## A note on the global L2 norm and large grids

On a graph where one node carries a large state and the coupling touches
a few cells of it, the global L2 residual is dominated by bulk change
that does not iterate at all.  `expensive-pair` converges in **1.5
iterations** under the interface norm, at 1.85 ms, because the interface
agrees almost immediately at that tolerance; under the global L2 norm
the same graph reports 6.0 iterations and 6.17 ms.  The two norms are measuring
different things, and on a grid the L2 number is mostly a statement
about the grid, not about the coupling.  Use the
interface norm on grid couplings, and read `coupling_iter_stats` rather
than trusting a residual whose units you have not thought about.

## `linear_solver="dense"` is not an escape hatch on a grid

When the GMRES adjoint fails to converge — an ill-conditioned
`I - dF/dx`, which is what a stiff group produces — the obvious move is
to swap the matrix-free solve for the exact one and accept the cost.
Read the cost first.

`"dense"` materialises the full `N x N` coupling Jacobian **and** the
identity basis `jacfwd` builds it from, so both are live at once and
the peak working set is `2 * N**2 * itemsize`. In float32:

| coupled DOF `N` | peak working set | single Jacobian |
|---|---|---|
| 1,024 | 8.0 MiB | 4.0 MiB |
| 8,000 | 0.48 GiB | 0.24 GiB |
| 16,384 | 2.0 GiB | 1.0 GiB |
| 65,536 | 32.0 GiB | 16.0 GiB |
| ~3.6e5 | ~975 GiB | **523 GB** |

`jax_enable_x64` doubles every row. The figures are XLA's own
compiled-module memory analysis of the `_dense` body in
`maddening.core.graph_manager`, taken on CPython 3.12.3 with
jax/jaxlib 0.11.0 on the CPU backend; nothing was allocated to produce
them, and the analysis agrees with `2 * N**2 * 4` to within a few tens
of kilobytes at every size from 64 DOF to 3.6e5.

The last row is the point. There is no soft edge to this curve: the
solve does not thrash, or slow down, or lose accuracy. It does not
start. A grid-coupled group at that size reports

```text
Out of memory allocating 523186046552 bytes
```

— a single allocation, the Jacobian, before the first matvec. And
those are ordinary sizes for a grid coupling: one scalar field over a
20³ volume is already 8,000 DOF, a volume coupled to a surface
discretisation runs to 10⁵–10⁶, and the matrix-free path exists
precisely so that `N` never appears squared.

So the advice, plainly:

* **Small group (N up to a few thousand):** `"dense"` is a real escape
  hatch. It is exact, it costs megabytes, and
  `MADDENING_IFT_DENSE_SOLVE=1` is a reasonable triage switch.
* **Grid-coupled group:** it is not an option at any resolution you
  would run. The remedy is to make the group less stiff — stronger
  relaxation, a smaller timestep, or splitting the cycle — which
  attacks `cond(A) ~ 1 / (1 - rho)` itself. Raising GMRES's `restart`
  does not help either; it is already `min(N, 50)`.

MADDENING does re-solve densely on its own, but only below
`_DENSE_ADJOINT_FALLBACK_MAX_DOF` (50), where the Krylov space is
already the whole space and `N**2` floats of scratch are negligible.
Above that cap the failure is raised rather than silently paid for, and
the message it raises now prices the dense path at your own `N`.

## Invariants

`tests/core/test_coupling_fixture_invariants.py` asserts the properties
that matter more than any timing:

* every configuration of a fixture reaches the same fixed point.  Two
  lanes: the six L2 configurations on a spring fixture and a
  2 000-cell grid fixture run by default, and the full
  24-configuration sweep over three spring fixtures is slow-marked.
  The deviation is measured against the coupling group's own nodes;
  including the driver divided it by the driver's amplitude, which on
  `heterogeneous` is a factor of eighty.  The rows that do *not* reach
  the common fixed point are listed in `_KNOWN_DISAGREEMENTS` with the
  defect that explains each, and the test fails if a listed row starts
  agreeing, so the list cannot outlive its defect;
* Gauss-Seidel on a ring is order-dependent and Jacobi is not — the test
  first checks that rotating the build really does rotate the schedule,
  so it cannot pass vacuously;
* a group whose contraction factor exceeds one reports itself unconverged
  for every fixed-point acceleration, and IQN converges the same group;
* IQN on a single-degree-of-freedom interface, where its least-squares is
  rank-deficient, agrees with plain iteration rather than returning a
  silently wrong answer.  The comparison runs over a driven trajectory
  and is scaled by that trajectory's own amplitude: against an undriven
  pair, whose state is ~5e-7 by step 20, the assertion was a flat 1e-3
  absolute and an accelerator returning zero would have passed it;
* `accelerated_fields` that names no field in the group is rejected at
  construction;
* the two groups of `mixed-modes` keep their own schedules and iteration
  counts, and the graph steps deterministically.  "Keep their own
  schedules" is asserted by building the counterfactual with one
  group's mode flipped and requiring the iterate to move — reading
  `iteration_mode` back off the dataclass, which is what the test did
  before, would pass even if the solver ignored the field;
* the sweep driver's own summary fields: the agreement metric is
  size-invariant and ignores nodes outside every group, the best-row
  picker prefers fewer iterations when two rows' timings overlap, and
  the recorded JSON files share one schema.
