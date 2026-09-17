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
