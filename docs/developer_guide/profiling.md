# Profiling and benchmarking

Two tools, one question: *where does a step's time go, and is a change
faster without changing the answer?*

## `profile_graph`

```python
from maddening.core.simulation.profiler import profile_graph

report = profile_graph(gm, n_steps=200, n_warmup=5, trace=True)
print(report)
```

What it measures, and how:

| number | how it is obtained |
|--------|--------------------|
| `jit_compile_ms` | the first step (compile + run) |
| `mean_step_ms`, `median`, `p95` | `n_steps` warmed steps, each blocked on the device |
| `dispatch_floor_ms` | a jitted identity on the same `(state, ext, params)` pytree: the cost of dispatching a step before any physics runs |
| `node_times_ms` | each node's `update` jitted and timed in isolation (an estimate: one dispatch per node, no coupling) |
| `coupling_iter_stats[group]` | over up to 50 further steps: iterations used (mean / min / max) against `max_iterations`, fraction of steps that exited at the cap, fraction whose residual met the threshold, from the always-on `_meta` diagnostics |
| `coupling_overhead_ms` (`"measured"`) | the graph is recompiled with every group capped at one iteration and timed; the difference to the real step is what the extra iterations cost, and `coupling_per_iteration_ms` divides it by the mean number of extra iterations.  The graph, its state and compiled step are restored afterwards.  `measure_coupling=False` falls back to the old estimate (step minus the sum of isolated node times) |
| `trace` (`trace=True`) | a short `jax.profiler` trace; device kernel time per step, kernels per step, host dispatch of the step's own `PjitFunction`, and kernel time by scope |

### Reading the trace attribution

Node updates run under `jax.named_scope("node:<name>")`, the residual
under `coupling:residual`, acceleration under `coupling:accelerate`,
interface corrections under `coupling:interface_override`, mappings under
`edge:mapping`.  XLA labels a *fused* kernel with the longest common
prefix of the fused ops' names, so a kernel that fuses two nodes' work
is reported under the shared prefix (`while/body` for anything inside
the coupling loop) rather than under either node.  Attribution is
therefore exact for unfused kernels and coarse inside the loop body —
which still separates "inside the fixed-point loop" from "outside it".

`device_busy_fraction` is the number to look at first on a GPU: kernel
time over wall step time.  Well below one with hundreds of kernels per
step means the step is **launch-bound** — fewer, larger nodes or fewer
iterations help; optimising a single node's arithmetic does not.

### What the profiler does *not* tell you

It cannot see a compile that happens inside a caller's timing loop.  A
weak-typed leaf in the seed state (`jnp.array(0.0)` without a dtype)
retraces the step when the leaf comes back strongly typed; `compile()`
normalises the seed state, but a driver that assigns its own arrays
into `gm._state` can reintroduce it — use `gm.set_node_state()` and
`gm.reset_state()` instead, and check that `gm._compiled_step._cache_size()`
stays at 1 across a run.

`gm.trace_count` covers the compiled step only.  `run_scan`,
`run_scan_with_history`, `run_sweep` and `run_adaptive_scan` build a
separate `lax.scan` program around that step, counted by
`gm.scan_trace_count`: one per `compile()` per (entry point, step count)
in a healthy run.  A driver that loops over `run_scan` and sees that
number climb is paying a compile per call, which `trace_count` alone
would not show.

## `benchmarks/bench_coupling.py`

```bash
python benchmarks/bench_coupling.py --graph mime-ar4 \
    --experiment ../MIME/experiments/ar4_helical_drive --trace --json out.json
JAX_PLATFORMS=cpu python benchmarks/bench_coupling.py --graph heat-chain --n-cells 256
```

Runs `profile_graph` on the graph with its coupling group(s) and on the
same graph with them disabled (staggered back-edges), prints both
reports and a summary (coupling cost vs baseline, iterations used vs
cap, PERF-1 acceptance at `--acceptance-ms`, default 30), and writes a
JSON record.  Keep before/after records under `benchmarks/results/` when
a change claims a speed-up; a claim without the two JSON files is a
guess.

## `benchmarks/bench_coupling_sweep.py`

```bash
JAX_PLATFORMS=cpu python benchmarks/bench_coupling_sweep.py \
    --json benchmarks/results/coupling_sweep_cpu.json
```

Sweeps every `iteration_mode` x `acceleration` x `convergence_norm` over
the graph shapes in `benchmarks/coupling_fixtures.py` — chain, star,
ring, a stiffness sweep, two grid fixtures, a two-group graph and a
slow-drift case — and records per configuration the step time, the
iterations used against the cap, the fraction of steps converged, the
final residual and the launch-bound / compute-bound verdict.  The
results and what they mean for a given graph shape are in
[Choosing a coupling algorithm](coupling_algorithm_guide.md).

`--steps` changes only how many timings are averaged.  The iteration
counts, convergence fractions and residuals come from a separate
statistics pass whose length and starting point are the *fixture's*,
not the run's (`--stat-steps` overrides it, `profile_graph`'s
`n_stat_steps` is the underlying knob).  That matters because those
three are the numbers people quote across runs: while the pass was
`min(n_steps, 50)` steps taken from wherever the timed run stopped, a
shortened run moved the window as well as the sample size, and on a
periodically driven graph the mean iteration count moved with it.

## Persistent compilation cache

```bash
export MADDENING_COMPILATION_CACHE_DIR=~/.cache/maddening/xla
```

`GraphManager.compile()` picks the variable up and points JAX's
persistent cache at it with thresholds that cache MADDENING's
sub-second compiles (JAX's defaults skip them).  `compile_cache.enable(dir)`
does the same from code, and `compile_cache.warm_cache(gm_factory,
n_steps=2, scan_steps=N)` compiles a graph's `step` and `run_scan`
ahead of a run.  The cache key covers the JAX/XLA version, backend and
traced program, so a stale entry is a miss, never a wrong executable.

## Compilation counts, and the regression gate

Compile time is what a user of MADDENING feels first — the pitch is that
the whole graph compiles to one jitted step — and until v0.4.0 nothing
stopped a silent 3× compile-time regression from shipping.

The gate that guards it does not read the clock.  This repository has
already paid for a wall-clock gate: a test comparing per-step cost at
two coupling iteration caps with a 3× bound, on quantities around
1e-4 s, failed CI at 3.27× on a pull request that changed 122
documentation files and no code at all.  On an idle box the real ratio
is 0.97–1.31.  The number it read was the runner.

So the gate reads integers.  `profile_graph` reports a `CompileCounts`
alongside the timings (`compile_counts(gm)` gets them on their own):

| count | what it is | why it matters |
| --- | --- | --- |
| `retrace_count` | Python traces of the compiled step since `compile()`, i.e. XLA compilations | One is healthy.  An unexpected retrace is the classic silent regression — a weak-typed leaf, a dtype that drifts, an argument that stopped being static — and costs a full compile on every run |
| `jaxpr_primitive_count` | primitives in the step's jaxpr, recursing into `scan`/`while`/`cond` bodies | how much work the graph builder emits.  A body counts once, not once per iteration, so the number describes the program and not the trip count |
| `hlo_op_count` | operations in the lowered StableHLO module | what MADDENING hands to XLA.  Deliberately *pre*-optimisation: the post-fusion count is the backend's decision and differs by platform |

`scan_*` variants cover a `run_scan` program when `scan_steps` is passed.

All of these reproduce exactly: the same graph on the same JAX version
gives the same integers on any machine, under any load.  Wall-clock
stays in the report, worth trending, and never gates.

### Running the gate

```bash
python scripts/compile_counts.py --check    # what CI runs
python scripts/compile_counts.py --show     # print, write nothing
python scripts/compile_counts.py            # regenerate the baseline
```

Five workloads are measured — a single node, a coupled pair, a
multi-rate graph, a heat chain with a `run_scan` program, and a
`ShardedStencilNode` over a four-device mesh — and compared to
`benchmarks/compile_counts_baseline.json`.  The script pins
`JAX_PLATFORMS=cpu` and the device count before importing JAX, because
both change the counts; `tests/core/test_compile_counts.py` therefore
runs it as a subprocess.

### If the gate fails on your branch

It prints every count that moved, both tables, and the regenerate
command.  Two cases:

- **A retrace count moved.**  Compared exactly, always, because it is
  counted in Python and does not depend on the JAX version at all.  An
  increase is an extra XLA compile on every run; find the leaf whose
  dtype or weak type changes after the first step, or the argument that
  stopped being static, before touching the baseline.
- **An op count moved.**  Compared against a band: ±max(2 ops, 2%) when
  the running JAX matches the baseline's, ±max(10 ops, 25%) otherwise.
  If the change is intended, regenerate and commit the new baseline,
  saying in the commit message why the counts moved.

Regenerate under the JAX version CI pins where you can — the baseline
records the version it was taken on, and one matching CI's gets the
tighter band.  The counts were measured identical on JAX 0.10.2 and
0.11.0, so this costs strictness and nothing else.

`CompileCounts` reads the counts after four warmup steps, not one.  The
retrace bugs this project has actually had did not appear on the first
step: the weak-typed-seed bug traced again on step 2 when a weak leaf
came back strongly typed, and a third time on step 3.  Measured after a
single step it looked healthy — which is how the gate's own mutation
test found the flaw.
