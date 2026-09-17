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
| anything, first attempt | `gauss-seidel` / `aitken` / `interface` | Gauss-Seidel needs 1.7–1.9x fewer iterations than Jacobi on every shape measured, and Aitken removes up to half of what is left (0–49% under the interface norm).  It is the sweep's own best configuration on eight of the eighteen fast fixtures.  Its arithmetic is inside the dispatch floor on every launch-bound fixture; on a compute-bound one it is not free |
| cheap nodes, few interface DOFs, contraction below ~0.8 | `gauss-seidel` / `aitken` / `interface` | launch-bound: differences between configurations smaller than the dispatch floor — which is most of the step there — are not measuring the algorithm, so pick the fewest iterations and the least machinery among the rows that time the same |
| contraction above ~0.9, or unknown and possibly divergent | `gauss-seidel` / `iqn-ils` / `interface` | the only family that converges *past* the limit at all, and at gain 0.95 it takes 4.1 iterations where `gs/none/l2` exhausts its cap of 60 on 98% of steps and converges on 2%.  Not faster there (0.21 ms against 0.10) — right rather than fast |
| one expensive node among cheap ones | `gauss-seidel` / `iqn-imvj` / `interface`, with `accelerated_fields` naming **only the cheap nodes** | same iteration count as accelerating everything, at 1/60th the cost — the quasi-Newton problem needs enough degrees of freedom to model the interface response, not the grid |
| every node expensive (grid-to-grid) | `gauss-seidel` / `none` / `interface` | the interface norm alone takes `expensive-pair` from 5.6 iterations and 7.70 ms to 1.0 and 3.38 ms; IQN costs 70–160x and is not affordable at 2x10⁵ accelerated DOFs |
| deep chain (information must cross many nodes) | `gauss-seidel` / `aitken` / `interface` | Gauss-Seidel's advantage is real but *flat* in depth — it does not grow with the chain length |
| wide star (independent leaves) | `gauss-seidel` / `aitken` / `interface` | both accelerators are flat in width (Aitken 7.9 → 8.1 iterations from 2 to 16 leaves under the interface norm, IQN 3.0 → 3.0) but IQN's step cost is not: 1.45 ms against 0.51 ms at 16 leaves, for 5 fewer iterations that the dispatch floor hides |
| ring / cycle with no natural first node | `jacobi` if the answer must not depend on how the graph was built, otherwise `gauss-seidel` / `aitken` | Gauss-Seidel on a ring is measurably order-dependent; Jacobi is bit-identical under rotation and reversal |
| fixed point that barely moves between steps | `jacobi` / `fixed` ω = 0.8 / `l2` — the one place a constant ω wins | 6.14 → 3.46 iterations and the fastest step of any configuration on `slow-drift`; `iqn-imvj` with `jacobian_reuse` does cut iterations further (3.0 → 2.0) but costs ~20x the plain step to do it |
| two subsystems with different shapes | one group each, with its own settings | groups in one graph keep independent schedules, iteration counts and convergence flags |

Three settings that are nearly always right and are not in the table:

* **Leave `relaxation` alone unless your fixed point creeps.** Fixed
  under-relaxation lost on seventeen of the eighteen fast fixtures,
  usually by a factor of two in iterations, and it rescued no divergent
  case at all.  The exception is `slow-drift` under Jacobi, where
  ω = 0.8 cuts iterations 44% and is the fixture's best configuration.
  See below.
* **Cap `max_iterations` at what you actually expect.** It is not a free
  safety margin: IQN allocates `max_iterations - 1` secant columns, so
  the cap sets the size of its least-squares problem.
* **Turn on `strict_convergence` for training and calibration runs.**
  The implicit-function-theorem gradient is only valid at a converged
  fixed point.

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
  own "best configuration" picker treats rows whose recorded spreads
  overlap as tied and breaks the tie on iterations.  It used a fixed 10%
  band before, which is well inside the within-row spread these rows
  actually record — up to 84% from median to p95 — so it excluded rows
  that were not measurably slower and named an 18.6-iteration
  configuration as `star-8`'s best over a 7.1-iteration one.  Every
  `best` field in the recorded files comes from the corrected rule.

## Where the measurement contradicted the theory

Four expectations that the sweep did not support.  Each was written down
before the fixtures were built, which is the only reason they count as
predictions rather than commentary.

**1. The Gauss-Seidel advantage does not widen with chain length.**  The
expectation was that sequential information flow would let Gauss-Seidel
pull further ahead as the chain grew.  It does not:

| fixture | Gauss-Seidel | Jacobi | ratio |
|---|---|---|---|
| `chain-2` | 5.6 | 9.4 | 1.69 |
| `chain-5` | 13.4 | 23.6 | 1.77 |
| `chain-20` | 23.7 | 40.7 | 1.72 |
| `chain-50` | 30.8 | 53.0 | 1.72 |

The ratio is flat because a tridiagonal coupling operator is
*consistently ordered*, for which the Gauss-Seidel spectral radius is
exactly the square of the Jacobi one — a constant factor of two in
iterations at any depth, not a growing one.  What grows with N is the
radius itself (0.40 → 0.80), and so the absolute iteration count.

**2. The star is where Gauss-Seidel wins by the most, not the least.**
The expectation was that leaves which cannot see each other would leave
Gauss-Seidel's ordering nothing to exploit, making Jacobi competitive.
Measured, the star shows the *largest* Gauss-Seidel advantage of any
shape — 1.83–1.85 against 1.69–1.77 on the chain — and it is flat in
width (20.5 iterations at 2 leaves, 22.9 at 16).  The hub-to-leaf
dependency alone is enough to give the full squared radius; ordering
*among* the leaves was never what produced the factor.  On one device
there is no shape at which Jacobi is competitive on iterations.

**3. Aitken helps Gauss-Seidel more than Jacobi, except on a two-node
pair.**  The expectation was the reverse: Jacobi's error history decays
more cleanly, which is what Aitken's scalar relaxation assumes.  That
holds where the coupling really is a single mode —

| fixture | gs/none | gs/aitken | jac/none | jac/aitken |
|---|---|---|---|---|
| `stiff-pair-0.5` | 7.8 | 4.0 | 13.9 | 5.0 |
| `stiff-pair-0.8` | 20.8 | 9.9 | 38.0 | 20.7 |
| `star-8` | 21.7 | 10.4 | 40.2 | 37.4 |
| `star-16` | 22.9 | 10.7 | 42.4 | 42.5 |

— on `stiff-pair-0.5` Aitken very nearly erases the Gauss-Seidel
advantage, taking a 1.8x gap down to 1.25x, and on `chain-2` it closes
it outright (4.0 against 4.2).  But as soon as the error is a mixture of
modes with comparable magnitudes, the single Aitken ω cannot cancel them
and Jacobi's cleaner decay stops helping: on `star-16` Aitken buys
Jacobi **nothing at all** (42.4 → 42.5, inside the sampling spread of an
iteration count that varies by step) and Gauss-Seidel 53%.

**4. Fixed under-relaxation lost everywhere except one regime, and that
regime is `slow-drift`.**  The prediction was that it would lose
everywhere, and on seventeen of the eighteen fast fixtures it does,
roughly doubling the iteration count at ω = 0.5 (`chain-20`:
23.7 → 51.3; `star-16`: 22.9 → 49.4).  The exception is Jacobi on
`slow-drift`, the fixture that exists for the "fixed point barely moves"
regime, where ω = 0.8 cuts iterations by 44%:

| `slow-drift`, Jacobi | ω = 1 | ω = 0.5 | ω = 0.8 |
|---|---|---|---|
| L2 norm | 6.14 | 4.30 | **3.46** |
| interface norm | 6.40 | 4.54 | **3.84** |

`jacobi`/`fixed` ω = 0.8 / L2 is in fact the sweep's own best
configuration for that fixture.  That is not luck: a constant ω is a
one-parameter model of the error's decay, and a fixed point that barely
moves between steps is the one case where that model is right for the
whole run.  Everywhere else the error is a mixture of modes and a
constant ω is the wrong constant.  So: leave `relaxation` alone unless
your fixed point creeps, and if it does, measure both ω values.

What under-relaxation does *not* do, anywhere, is rescue a divergent
group.  This is arithmetic, not bad luck: relaxing maps an eigenvalue λ
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
passes**, so it cannot exit in fewer than four.  On a group that already
converges in three that costs a pass — `slow-drift` goes 3.3 → 4.0 and
`stiff-pair-0.25` 3.6 → 4.0 — and every fixture where it is fast lands
on exactly 4.0.  It is an accelerator for iterations you have, not for
iterations you do not.

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

**1. Lower `max_iterations`.**  A cap of 60 gives IQN 59 secant columns
whether or not it ever needs them.  Re-running the same graphs with
`--max-iterations 16` (`coupling_sweep_cap16_cpu.json`):

| fixture / config | cap 60 | cap 16 |
|---|---|---|
| `chain-50` `gs/iqn-ils/l2` | 14.1 it, 107 ms | 14.0 it, 3.00 ms |
| `chain-50` `gs/iqn-imvj5/l2` | 13.6 it, 65.0 ms | 13.8 it, 3.44 ms |
| `chain-20` `gs/iqn-ils/l2` | 13.6 it, 2.56 ms | 13.6 it, 1.75 ms |
| `star-16` `gs/iqn-ils/l2` | 5.0 it, 1.43 ms | 5.0 it, 1.35 ms |

Same iteration count, a thirty-fifth of the cost on the worst row.  The
cap is not a free safety margin, it is a sizing parameter.

The obvious caveat: a cap only costs nothing for configurations that
already converge inside it.  In the same run, `gs/none/l2` at cap 16
converged on **0%** of `chain-20` and `chain-50` steps, 3% of `star-16`
and 4% of `stiff-pair-0.8` — it needs 20–25.  Set the cap from the
iteration count you measured, not from the one you hoped for, and read
`converged_fraction` afterwards.

**2. Set `accelerated_fields` explicitly** so a large-state node does not
put its whole array into the least-squares problem.  `None`
auto-detects the fields the group's internal edges *read*, which for
a grid node coupled on one cell is still the entire grid.

## Four combinations that theory says should win

**Jacobi + Aitken.** Nearly closes the gap on two-node pairs
(`stiff-pair-0.5`: 13.9 → 5.0 against Gauss-Seidel + Aitken's 4.0;
`chain-2`: 9.4 → 4.2 against 4.0) and does nothing on anything wider
(`star-16`: 42.4 → 42.5).  Not a general recommendation.

**Jacobi + IQN.** Marginally better than Gauss-Seidel + IQN on the
chain and ring families (`chain-5` 5.3 vs 7.0, `chain-20` 13.2 vs 14.1,
`ring-4` 5.0 vs 5.9, `chain-50` 16.7 vs 16.9) and marginally worse on
the stars (`star-8` 4.2 vs 4.0, `star-16` 4.5 vs 4.0).  The hypothesis — a residual history from a single
iterate spans a cleaner space — is weakly supported on chains only, and
the effect is smaller than the noise in step time.  Worth knowing, not
worth defaulting to.

**Gauss-Seidel + fixed under-relaxation.** Never won.  See contradiction
4 above.

**Interface norm with `accelerated_fields` restricted to interface DOFs.**
Half right, and the half that is wrong is the more useful half.

The interface norm alone generalised from AR4.  Measured on
`gs/none` across all eighteen fast fixtures it removes **−5% to 31%** of
the iterations: 25–31% on the deep chains, the wide stars and the larger
rings, 13–25% on the small and the stiff pairs, nothing at all on
`stiff-pair-1.2` (which converges nowhere), and −5% on `slow-drift`,
where it costs a fifth of an iteration.  A blanket "25–35%" was the top
of that range quoted as the whole of it.

| fixture | l2 | interface |
|---|---|---|
| `chain-20` | 23.7 | 16.7 |
| `chain-50` | 30.8 | 23.0 |
| `star-16` | 22.9 | 15.9 |
| `stiff-pair-0.95` | 58.4 (at cap on 98% of steps) | 49.6 (at cap on 34%) |

"At no measurable cost" is also not quite true, and in the direction you
would not guess: eight of the eighteen fixtures record a *higher* step
time under the interface norm than under L2 (`ring-4` 0.144 → 0.179 ms,
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
orders of magnitude (Gauss-Seidel, L2 norm, 10-step run):

| `accelerated_fields` | accelerated DOFs | iterations | ms/step |
|---|---|---|---|
| `None` (auto: grid + scalars) | ~6x10⁴ | 4.0 | 188 |
| everything, incl. velocities | ~6x10⁴ | 3.0 | 170 |
| the expensive node only | 6x10⁴ | 8.5 | 343 |
| **the four cheap nodes only** | **4** | **4.0** | **2.49** |
| (no acceleration, for scale) | — | 6.2 | 2.22 |

Same iteration count as accelerating everything, at one seventy-sixth of
the cost — and `iqn-imvj` on the cheap nodes does better still, 3.0
iterations at 2.02 ms.  The reading is that the secant history needs
enough degrees of freedom to model how the *interface* responds, and the
four scalars carry that; the grid's sixty thousand cells add nothing to
the model and the entire cost.  Restricting to the grid alone removes
the degrees of freedom that were doing the work and doubles the
iteration count.

The corollary for a grid-to-grid group, where there is no cheap subset
to fall back on: IQN is simply not affordable.  On `expensive-pair` it
costs 70–160x the plain step, and under the L2 norm it converged on only
90% of steps.
One honest caveat: the interface norm is a *relative* criterion
(`atol`/`rtol`), so it stops earlier than an absolute L2 tolerance, and
the trajectories drift further apart as a result.  Measured against the
`gauss-seidel/none/l2` trajectory over the profiled run, and scaling
each field by its own magnitude rather than the graph's largest, the
worst interface-norm row per fixture runs from 3.8x10⁻⁵ (`chain-2`)
through 2x10⁻³ (`chain-20`, `star-4`) to 3.7x10⁻² on `chain-50`; every
L2 row on the fast fixtures stays within 5x10⁻⁴.

The largest disagreement anywhere in the sweep is 4.2x10⁻² on
`heterogeneous`, from `jacobi/aitken/l2` — a row that reports itself
fully converged.  That is a fixture with a four-orders-of-magnitude
spread of scales between its grid and its scalar nodes, and the figure
is small enough to be consistent with each configuration stopping at its
own tolerance, but it is the one number here that has not been run down
to a cause.  If you are relying on `heterogeneous` for anything load-
bearing, start by reproducing that row.  That is the same fixed point reached
to a looser tolerance, not a different one — the sweep records
`fixed_point_agreement` per fixture and the suite asserts it — but the
deviation grows with the fixture's condition number, so set `rtol`
deliberately rather than inheriting it.

## A note on the global L2 norm and large grids

On a graph where one node carries a large state and the coupling touches
a few cells of it, the global L2 residual is dominated by bulk change
that does not iterate at all.  `expensive-pair` converges in **1.0
iterations** under the interface norm, at 3.38 ms, because the interface
agrees immediately at that tolerance; under the global L2 norm the same
graph reports 5.6 iterations and 7.70 ms.  The two norms are measuring
different things, and on a grid the L2 number is mostly a statement
about the grid, not about the coupling.  Use the
interface norm on grid couplings, and read `coupling_iter_stats` rather
than trusting a residual whose units you have not thought about.

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
