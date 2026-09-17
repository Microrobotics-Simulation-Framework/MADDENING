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
| anything, first attempt | `gauss-seidel` / `aitken` / `interface` | Gauss-Seidel needs 1.6–1.9x fewer iterations than Jacobi on every shape measured; Aitken costs nothing measurable and removes 30–60% of the remaining iterations |
| cheap nodes, few interface DOFs, contraction below ~0.8 | `gauss-seidel` / `aitken` / `interface` | launch-bound: every configuration's step time is within noise of every other, so pick the one with the fewest iterations and the least machinery |
| contraction above ~0.9, or unknown and possibly divergent | `gauss-seidel` / `iqn-ils` / `interface` | the only family that converged at all past the contraction limit, and the one case where IQN also wins on time: 6.0 iterations and 0.34 ms against a cap of 60 exhausted on 96% of steps |
| one or more nodes with a large state (grids, meshes) | `gauss-seidel` / `aitken` / `interface`, and set `accelerated_fields` before trying IQN | IQN's least-squares runs over every accelerated degree of freedom; on a 2x10⁵-cell interface it cost 80x the plain step |
| deep chain (information must cross many nodes) | `gauss-seidel` / `aitken` / `interface` | Gauss-Seidel's advantage is real but *flat* in depth — it does not grow with the chain length |
| wide star (independent leaves) | `gauss-seidel` / `aitken` / `interface` | both accelerators are flat in width (Aitken 9.2 → 9.6 iterations from 2 to 16 leaves, IQN 4.9 → 5.0) but IQN's step cost is not: 1.55 ms against 0.69 ms at 16 leaves, for 4.6 fewer iterations that the dispatch floor hides |
| ring / cycle with no natural first node | `jacobi` if the answer must not depend on how the graph was built, otherwise `gauss-seidel` / `aitken` | Gauss-Seidel on a ring is measurably order-dependent; Jacobi is bit-identical under rotation and reversal |
| fixed point that barely moves between steps | `gauss-seidel` / `aitken` / `interface`; try `iqn-imvj` with `jacobian_reuse` only if the interface is small | reuse does cut iterations (3 → 2) but the quasi-Newton machinery cost 20x the saving here |
| two subsystems with different shapes | one group each, with its own settings | groups in one graph keep independent schedules, iteration counts and convergence flags |

Three settings that are nearly always right and are not in the table:

* **Leave `relaxation` alone.** Fixed under-relaxation lost on every
  fixture in the sweep, usually by a factor of two in iterations, and it
  did not rescue a single divergent case.  See below.
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
# 96 rows, ~16 min: the two grid fixtures, plus accelerated_fields variants
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
  own "best configuration" picker treats anything within 10% of the
  fastest as tied and breaks the tie on iterations.

## Where the measurement contradicted the theory

Four expectations that the sweep did not support.  Each was written down
before the fixtures were built, which is the only reason they count as
predictions rather than commentary.

**1. The Gauss-Seidel advantage does not widen with chain length.**  The
expectation was that sequential information flow would let Gauss-Seidel
pull further ahead as the chain grew.  It does not:

| fixture | Gauss-Seidel | Jacobi | ratio |
|---|---|---|---|
| `chain-2` | 5.5 | 9.2 | 1.68 |
| `chain-5` | 13.3 | 23.4 | 1.76 |
| `chain-20` | 22.9 | 39.2 | 1.71 |
| `chain-50` | 25.4 | 43.5 | 1.71 |

The ratio is flat because a tridiagonal coupling operator is
*consistently ordered*, for which the Gauss-Seidel spectral radius is
exactly the square of the Jacobi one — a constant factor of two in
iterations at any depth, not a growing one.  What grows with N is the
radius itself (0.40 → 0.80), and so the absolute iteration count.

**2. The star is where Gauss-Seidel wins by the most, not the least.**
The expectation was that leaves which cannot see each other would leave
Gauss-Seidel's ordering nothing to exploit, making Jacobi competitive.
Measured, the star shows the *largest* Gauss-Seidel advantage of any
shape — 1.83–1.85 against 1.68–1.76 on the chain — and it is flat in
width (20.4 iterations at 2 leaves, 22.0 at 16).  The hub-to-leaf
dependency alone is enough to give the full squared radius; ordering
*among* the leaves was never what produced the factor.  On one device
there is no shape at which Jacobi is competitive on iterations.

**3. Aitken helps Gauss-Seidel more than Jacobi, except on a two-node
pair.**  The expectation was the reverse: Jacobi's error history decays
more cleanly, which is what Aitken's scalar relaxation assumes.  That
holds where the coupling really is a single mode —

| fixture | gs/none | gs/aitken | jac/none | jac/aitken |
|---|---|---|---|---|
| `stiff-pair-0.5` | 6.7 | 3.0 | 12.0 | 3.1 |
| `stiff-pair-0.8` | 20.1 | 9.0 | 36.3 | 22.3 |
| `star-8` | 21.4 | 9.5 | 39.7 | 35.9 |
| `star-16` | 22.0 | 9.6 | 40.6 | 38.3 |

— on `stiff-pair-0.5` Aitken erases the Gauss-Seidel advantage entirely,
exactly as predicted.  But as soon as the error is a mixture of modes
with comparable magnitudes, the single Aitken ω cannot cancel them and
Jacobi's cleaner decay stops helping: on `star-16` Aitken buys Jacobi 6%
and Gauss-Seidel 56%.

**4. Fixed under-relaxation never won anything.**  It lost on every
fixture, roughly doubling the iteration count at ω = 0.5 (`chain-20`:
22.9 → 49.4; `star-16`: 22.0 → 47.6), and it did not rescue a single
divergent case.  This is arithmetic, not bad luck: relaxing maps an
eigenvalue λ to `1 - ω + ωλ`, which for a *positive* λ > 1 stays above 1
for every ω > 0.  `stiff-pair-1.2` has a Gauss-Seidel eigenvalue of
+1.44 and neither ω = 0.5 nor ω = 0.8 converged a single step of it.
Under-relaxation is the cure for an eigenvalue outside the unit circle on
the *negative* side, i.e. for an oscillating divergence; it is not a
general stabiliser, and it should not be reached for first.

## The finding that matters most

**IQN converges problems that no fixed-point method can.**  On
`stiff-pair-1.2`, whose coupling gain is past the convergence limit by
construction, after 40 steps:

| acceleration | iterations | converged | state after 40 steps |
|---|---|---|---|
| `none` | 11 of 12 (at cap) | no | NaN |
| `aitken` | 11 of 12 (at cap) | no | ~1e+06 |
| `fixed` ω=0.5, 0.8 | 11 of 12 (at cap) | no | 1e+09 … 1e+13 |
| `iqn-ils` | 6 | yes | −0.704 |
| `iqn-imvj`, reuse 5 | 3 | yes | −0.704 |

IQN-ILS is a quasi-Newton root solver, not a contraction, so a spectral
radius above one is not fatal to it; it solves the coupled algebraic
system directly.  This is why the invariant is phrased as "a divergent
group must *report itself* unconverged" rather than "must fail": the
fixed-point family must be honest about hitting the cap, and IQN is
allowed to succeed where they cannot.  Both are asserted in
`tests/core/test_coupling_fixture_invariants.py`.

The same effect shows at gain 0.95, which is inside the convergence
limit but too slow to be useful: `gs/none/l2` exhausts its cap of 60 on
96% of steps, while `gs/iqn-ils/l2` converges every step in 6.0
iterations and `gs/iqn-imvj` in 3.5.

## What IQN costs

IQN's iteration counts are the best of any option on nearly every
fixture.  Its *time* is not, and the reason is worth stating precisely:
each iteration solves a least-squares problem with `jnp.linalg.pinv`
over an `n_dof x max_cols` matrix, where `n_dof` is the number of
accelerated degrees of freedom and **`max_cols = max_iterations - 1`**.

| fixture | accelerated DOFs | gs/none | gs/iqn-ils | factor |
|---|---|---|---|---|
| `chain-5` | 5 | 0.170 ms | 0.953 ms | 5.6x |
| `chain-20` | 20 | 0.620 ms | 2.767 ms | 4.5x |
| `chain-50` | 50 | 1.381 ms | 331.6 ms | 240x |
| `slow-drift` | 4 000 | 0.447 ms | 9.065 ms | 20x |

`chain-50` is the cautionary row: 14.1 iterations against 25.4, and a
step that is 240 times slower.  Two levers, in order:

1. **Lower `max_iterations`.**  A cap of 60 gives IQN 59 secant columns
   whether or not it ever needs them.
2. **Set `accelerated_fields` explicitly** so a large-state node does not
   put its whole array into the least-squares problem.  `None`
   auto-detects the fields the group's internal edges *read*, which for
   a grid node coupled on one cell is still the entire grid.

## Four combinations that theory says should win

**Jacobi + Aitken.** Wins on two-node pairs (`stiff-pair-0.5`: 12.0 → 3.1,
matching Gauss-Seidel + Aitken's 3.0) and loses on everything wider or
deeper (`star-16`: 40.6 → 38.3).  Not a general recommendation.

**Jacobi + IQN.** Marginally better than Gauss-Seidel + IQN on the middle
of the chain family (`chain-5` 5.2 vs 7.0, `chain-20` 11.9 vs 13.6,
`ring-4` 5.1 vs 6.0) and not elsewhere (`chain-50` 15.3 vs 14.1, star
family ~equal).  The hypothesis — a residual history from a single
iterate spans a cleaner space — is weakly supported on chains only, and
the effect is smaller than the noise in step time.  Worth knowing, not
worth defaulting to.

**Gauss-Seidel + fixed under-relaxation.** Never won.  See contradiction
4 above.

**Interface norm with `accelerated_fields` on interface DOFs.**  This is
the one that generalised from AR4.  The interface norm alone removes
25–35% of the iterations at no measurable cost:

| fixture | l2 | interface |
|---|---|---|
| `chain-20` | 22.9 | 14.7 |
| `chain-50` | 25.4 | 15.5 |
| `star-16` | 22.0 | 14.1 |
| `stiff-pair-0.95` | 58.8 (at cap on 96% of steps) | 46.2 (at cap on 20%) |

Combined with IQN it is better still (`chain-50`: 14.1 iterations and
331 ms under l2, 7.6 iterations and 30.7 ms under the interface norm —
the smaller residual vector also shrinks the least-squares problem).
One honest caveat: the interface norm is a *relative* criterion
(`atol`/`rtol`), so it stops earlier than an absolute L2 tolerance, and
the trajectories drift further apart as a result.  Measured against the
`gauss-seidel/none/l2` trajectory over the profiled run, the worst
interface-norm row per fixture sits at 4x10⁻⁵ (`chain-2`) to 2x10⁻³
(`chain-20`, `star-4`, `ring-16`), and 1.4x10⁻² on `chain-50`; the L2
rows stay within 2x10⁻⁴ everywhere.  That is the same fixed point reached
to a looser tolerance, not a different one — the sweep records
`fixed_point_agreement` per fixture and the suite asserts it — but the
deviation grows with the fixture's condition number, so set `rtol`
deliberately rather than inheriting it.

## A note on the global L2 norm and large grids

On a graph where one node carries a large state and the coupling touches
a few cells of it, the global L2 residual is dominated by bulk change
that does not iterate at all.  `expensive-pair` converges in **1.0
iterations** under the interface norm, at 2.33 ms, because the interface
agrees immediately at that tolerance; under the global L2 norm the same
graph reports 5.3 iterations and 6.86 ms.  The two norms are measuring
different things, and on a grid the L2 number is mostly a statement
about the grid, not about the coupling.  Use the
interface norm on grid couplings, and read `coupling_iter_stats` rather
than trusting a residual whose units you have not thought about.

## Invariants

`tests/core/test_coupling_fixture_invariants.py` asserts the properties
that matter more than any timing:

* every configuration of a fixture reaches the same fixed point (slow-marked);
* Gauss-Seidel on a ring is order-dependent and Jacobi is not — the test
  first checks that rotating the build really does rotate the schedule,
  so it cannot pass vacuously;
* a group whose contraction factor exceeds one reports itself unconverged
  for every fixed-point acceleration, and IQN converges the same group;
* IQN on a single-degree-of-freedom interface, where its least-squares is
  rank-deficient, agrees with plain iteration rather than returning a
  silently wrong answer;
* `accelerated_fields` that names no field in the group is rejected at
  construction;
* the two groups of `mixed-modes` keep their own schedules and iteration
  counts, and the graph steps deterministically.
