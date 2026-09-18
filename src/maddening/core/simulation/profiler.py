"""
Graph profiler -- measure per-node and per-step performance.

Provides :func:`profile_graph` which runs a graph for a specified
number of steps and reports timing breakdowns: total step time,
per-node update time, coupling overhead, and JIT compilation time.

Usage::

    from maddening.core.simulation.profiler import profile_graph

    report = profile_graph(gm, n_steps=100)
    print(report)

The profiler works by:

1. Measuring JIT compilation time (first step) and the warmed-up step
   time (mean / median / p95 over ``n_steps``).
2. Measuring the *dispatch floor*: a jitted identity on the same state
   pytree, i.e. what a step costs before any physics runs.
3. Collecting per-group coupling statistics over a run — iterations
   used (mean / min / max against ``max_iterations``), the fraction of
   steps that hit the cap, the fraction that converged — from the
   always-on ``_meta`` diagnostics.
4. **Measuring** coupling overhead: the same graph is recompiled with
   every group capped at one iteration and timed; the difference to
   the real step is what the extra iterations cost, and divided by the
   mean number of extra iterations gives the per-iteration cost.
   (The older "total minus sum of isolated node costs" figure is kept
   as ``sum_node_ms`` / an *estimate* when measurement is disabled.)
5. Optionally recording a ``jax.profiler`` trace of a few steps and
   attributing device kernel time to the graph's ``jax.named_scope``
   labels (``node:<name>``, ``coupling:residual``, ...), plus the
   device-busy fraction of the step — the number that says whether a
   GPU step is compute-bound or kernel-launch-bound.
"""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability


@dataclass
class ProfileReport:
    """Profiling results for a simulation graph."""
    graph_name: str = ""
    n_steps: int = 0
    n_nodes: int = 0

    # Overall timing
    jit_compile_ms: float = 0.0
    mean_step_ms: float = 0.0
    std_step_ms: float = 0.0
    median_step_ms: float = 0.0
    p95_step_ms: float = 0.0
    total_run_ms: float = 0.0
    steps_per_second: float = 0.0
    # A jitted identity on the same (state, ext, params) pytree: the
    # cost of dispatching a step before any physics runs.
    dispatch_floor_ms: float = 0.0
    device: str = ""

    # Per-node timing (each node's update jitted in isolation; an
    # estimate — it includes one dispatch per node and ignores coupling)
    node_times_ms: dict[str, float] = field(default_factory=dict)
    sum_node_ms: float = 0.0

    # Coupling
    n_coupling_groups: int = 0
    # ``"measured"``: real step minus the same graph capped at one
    # iteration per group; ``"estimated"``: real step minus the sum of
    # isolated node costs (the pre-0.4 definition); ``""``: no groups.
    coupling_overhead_method: str = ""
    coupling_overhead_ms: float = 0.0
    one_iteration_step_ms: float = 0.0
    coupling_per_iteration_ms: float = 0.0
    # Last-step iteration count per group (kept for compatibility).
    coupling_iters: dict[str, int] = field(default_factory=dict)
    # Per group over the profiled run: mean / min / max iterations,
    # max_iterations, fraction of steps at the cap, fraction converged.
    coupling_iter_stats: dict[str, dict] = field(default_factory=dict)

    # Optional jax.profiler trace attribution (``trace=True``)
    trace: Optional["TraceSummary"] = None

    # State sizes
    node_sizes: dict[str, int] = field(default_factory=dict)
    total_state_elements: int = 0

    # Recommendations
    bottleneck: str = ""
    recommendations: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"=== Graph Profile: {self.graph_name} ===",
            f"",
            f"  Nodes: {self.n_nodes}",
            f"  Total state: {self.total_state_elements:,} elements "
            f"({self.total_state_elements * 4 / 1024:.0f} KB float32)",
            f"",
            f"  Device:       {self.device or 'unknown'}",
            f"  JIT compile:  {self.jit_compile_ms:>8.1f} ms",
            f"  Mean step:    {self.mean_step_ms:>8.2f} ms "
            f"(+/- {self.std_step_ms:.2f}; median {self.median_step_ms:.2f}, "
            f"p95 {self.p95_step_ms:.2f})",
            f"  Dispatch floor: {self.dispatch_floor_ms:>6.2f} ms "
            f"(jitted identity on the state)",
            f"  Throughput:   {self.steps_per_second:>8.0f} steps/s",
            f"",
        ]

        if self.node_times_ms:
            lines.append("  Per-node estimated cost:")
            sorted_nodes = sorted(self.node_times_ms.items(),
                                  key=lambda x: -x[1])
            for name, ms in sorted_nodes:
                pct = ms / max(self.mean_step_ms, 1e-10) * 100
                size = self.node_sizes.get(name, 0)
                lines.append(
                    f"    {name:20s} {ms:>7.2f} ms ({pct:>5.1f}%)  "
                    f"[{size:,} elems]"
                )

        if self.n_coupling_groups > 0:
            lines.append(f"")
            lines.append(f"  Coupling groups: {self.n_coupling_groups}")
            lines.append(
                f"  Coupling overhead: {self.coupling_overhead_ms:.2f} ms "
                f"({self.coupling_overhead_method or 'n/a'}"
                + (f"; one-iteration step {self.one_iteration_step_ms:.2f} ms, "
                   f"{self.coupling_per_iteration_ms:.2f} ms per extra iteration"
                   if self.coupling_overhead_method == "measured" else "")
                + ")"
            )
            for group_key, st in self.coupling_iter_stats.items():
                lines.append(
                    f"    {group_key}: {st['mean']:.1f} iterations mean "
                    f"(min {st['min']}, max {st['max']}, cap {st['cap']}); "
                    f"at cap {st['at_cap_fraction'] * 100:.0f}% of steps, "
                    f"converged {st['converged_fraction'] * 100:.0f}%"
                )
            for group_key, iters in self.coupling_iters.items():
                if group_key not in self.coupling_iter_stats:
                    lines.append(f"    {group_key}: {iters} iterations")

        if self.trace is not None:
            lines.append(f"")
            lines.extend(self.trace.lines())

        if self.bottleneck:
            lines.append(f"")
            lines.append(f"  Bottleneck: {self.bottleneck}")

        if self.recommendations:
            lines.append(f"")
            lines.append(f"  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    - {rec}")

        return "\n".join(lines)


@stability(StabilityLevel.EVOLVING)
@dataclass
class TraceSummary:
    """Device-side attribution from a short ``jax.profiler`` trace.

    ``device_ms_by_scope`` sums kernel time per step by the first
    ``jax.named_scope`` component the graph attaches (``node:<name>``,
    ``coupling:residual``, ``coupling:accelerate``,
    ``coupling:interface_override``, ``edge:mapping``); kernels without
    a scope land in ``"unscoped"``.  ``device_busy_fraction`` is device
    kernel time over wall step time: well below 1 on a GPU means the
    step is kernel-launch / dispatch bound, not compute bound.
    """
    n_steps: int = 0
    device_busy_ms_per_step: float = 0.0
    device_busy_fraction: float = 0.0
    n_kernels_per_step: float = 0.0
    host_dispatch_ms_per_step: float = 0.0
    device_ms_by_scope: dict[str, float] = field(default_factory=dict)
    top_kernels: list[tuple[str, float]] = field(default_factory=list)
    log_dir: str = ""

    def lines(self) -> list[str]:
        out = [
            f"  Trace ({self.n_steps} steps, {self.log_dir}):",
            f"    device busy: {self.device_busy_ms_per_step:.2f} ms/step "
            f"({self.device_busy_fraction * 100:.0f}% of wall step); "
            f"{self.n_kernels_per_step:.0f} kernels/step; "
            f"host dispatch {self.host_dispatch_ms_per_step:.2f} ms/step",
        ]
        if self.device_ms_by_scope:
            out.append("    device time by scope:")
            for k, v in sorted(self.device_ms_by_scope.items(), key=lambda kv: -kv[1]):
                out.append(f"      {k:32s} {v:>7.3f} ms/step")
        if self.top_kernels:
            out.append("    top kernels:")
            for name, v in self.top_kernels[:8]:
                out.append(f"      {name[:48]:48s} {v:>7.3f} ms/step")
        return out


def _meta_group_keys(gm) -> list[tuple[str, str, str, float, int]]:
    """``(group_key, iter_key, res_key, threshold, cap)`` per group."""
    out = []
    for g in gm._coupling_groups:
        key = "+".join(sorted(g.nodes))
        thr = 1.0 if g.convergence_norm in ("mixed", "interface") else float(g.tolerance)
        out.append((key, f"coupling_{key}_iterations", f"coupling_{key}_residual",
                    thr, int(g.max_iterations)))
    return out


def _time_steps(gm, external_inputs, n: int) -> np.ndarray:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        gm.step(external_inputs)
        jax.block_until_ready(jax.tree.leaves(gm._state))
        times.append((time.perf_counter() - t0) * 1000)
    return np.asarray(times)


def _one_iteration_variant(gm):
    """Context: every coupling group capped at one iteration (and not
    strict), recompiled; restores groups, state and compiled step on exit.

    Used to *measure* what the fixed-point iterations cost: the one-
    iteration step still resolves edges, updates every node once and
    writes diagnostics, so the difference to the real step is the extra
    iterations only.
    """
    import contextlib
    import dataclasses

    @contextlib.contextmanager
    def _cm():
        saved_groups = list(gm._coupling_groups)
        saved_state = jax.tree.map(lambda x: x, gm._state)
        saved_params = gm.params
        saved_step = gm._compiled_step
        try:
            gm._coupling_groups = [
                dataclasses.replace(g, max_iterations=1, strict_convergence=False)
                for g in saved_groups
            ]
            gm.compile()
            gm._state = jax.tree.map(lambda x: x, saved_state)
            yield
        finally:
            gm._coupling_groups = saved_groups
            gm.compile()
            gm._state = saved_state
            gm.params = saved_params
            # ``compile`` rebuilt the step; the original object is fine
            # to keep for callers holding a reference (same graph).
            del saved_step

    return _cm()


def _scope_of(path: str) -> str:
    """Scope label for a kernel from its op-name path.

    XLA labels a fused kernel with the longest common prefix of the
    fused ops' names, so a kernel that fuses two nodes' work carries
    only the shared part (``while/body``).  We strip the ``jit(...)``
    prefix and keep up to two components, preferring a ``named_scope``
    component (``node:x``, ``coupling:residual``) when the prefix
    reaches one; kernels without an op name (copies, sorts) are
    ``"unscoped"``.
    """
    if not path:
        return "unscoped"
    parts = [p for p in path.split("/") if p and not p.startswith("jit(")]
    if not parts:
        return "unscoped"
    scoped = [p for p in parts if ":" in p]
    if scoped:
        return scoped[0]
    return "/".join(parts[:2])


def _summarise_trace(
    log_dir: str, n_steps: int, wall_step_ms: float, step_fn_name: str = "",
) -> TraceSummary:
    """Aggregate the Perfetto JSON ``jax.profiler`` wrote under ``log_dir``.

    ``host_dispatch_ms_per_step`` counts only the compiled step's own
    ``PjitFunction`` events (``step_fn_name``), not nested jits.
    """
    import glob
    import gzip

    summary = TraceSummary(n_steps=n_steps, log_dir=log_dir)
    files = glob.glob(os.path.join(log_dir, "**", "perfetto_trace.json.gz"), recursive=True)
    if not files or n_steps <= 0:
        return summary
    with gzip.open(sorted(files)[-1]) as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    pids = {
        e["pid"]: e["args"].get("name", "")
        for e in events
        if e.get("ph") == "M" and e.get("name") == "process_name"
    }
    by_scope: dict[str, float] = {}
    by_kernel: dict[str, float] = {}
    device_us = 0.0
    n_kernels = 0
    host_dispatch_us = 0.0
    for e in events:
        if e.get("ph") != "X":
            continue
        pname = pids.get(e.get("pid"), "")
        dur = float(e.get("dur", 0.0))
        name = e.get("name", "")
        if "device:" in pname and "host" not in pname.lower():
            device_us += dur
            n_kernels += 1
            by_kernel[name] = by_kernel.get(name, 0.0) + dur
            path = (e.get("args") or {}).get("name", "")
            by_scope[_scope_of(path)] = by_scope.get(_scope_of(path), 0.0) + dur
        elif step_fn_name and name == f"PjitFunction({step_fn_name})":
            host_dispatch_us += dur
    per_step = 1.0 / (n_steps * 1000.0)
    summary.device_busy_ms_per_step = device_us * per_step
    summary.device_busy_fraction = (
        summary.device_busy_ms_per_step / wall_step_ms if wall_step_ms > 0 else 0.0
    )
    summary.n_kernels_per_step = n_kernels / n_steps
    summary.host_dispatch_ms_per_step = host_dispatch_us * per_step
    summary.device_ms_by_scope = {k: v * per_step for k, v in by_scope.items()}
    summary.top_kernels = sorted(
        ((k, v * per_step) for k, v in by_kernel.items()), key=lambda kv: -kv[1],
    )[:12]
    return summary


@stability(StabilityLevel.EVOLVING)
def profile_graph(
    gm,
    n_steps: int = 100,
    n_warmup: int = 3,
    external_inputs: Optional[dict] = None,
    *,
    measure_coupling: bool = True,
    n_stat_steps: Optional[int] = None,
    trace: bool = False,
    trace_steps: int = 20,
    trace_dir: Optional[str] = None,
) -> ProfileReport:
    """Profile a compiled GraphManager.

    Parameters
    ----------
    gm : GraphManager
        A compiled (or compilable) graph.  Its state is reset to the
        nodes' initial state for the run.
    n_steps : int
        Number of steps to benchmark (after warmup).
    n_warmup : int
        Warmup steps (included in JIT timing, excluded from
        per-step timing).
    external_inputs : dict or None
        External inputs for each step.
    measure_coupling : bool
        Recompile the graph with every coupling group capped at one
        iteration and time it, so ``coupling_overhead_ms`` is measured
        rather than inferred (costs one extra compile; the graph is
        restored afterwards).  Ignored without coupling groups.
    n_stat_steps : int or None
        Steps in the coupling-iteration statistics pass.  ``None``
        keeps the historical behaviour: ``min(n_steps, 50)`` steps taken
        from wherever the timed run left the state, which makes
        ``coupling_iter_stats`` depend on ``n_steps`` twice over — the
        sample count follows it, and so does the window's *position* in
        the trajectory.  On a periodically driven graph that moves the
        mean iteration count by tens of percent, which is a trap for
        any caller that shortens a timing run believing the iteration
        counts are properties of the step.  Passing a value pins both:
        the state is reset and re-warmed first, so the statistics come
        from the same window whatever ``n_steps`` is.
    trace : bool
        Record ``trace_steps`` steps with ``jax.profiler`` and attribute
        device kernel time to the graph's named scopes
        (:class:`TraceSummary`).
    trace_steps, trace_dir
        Trace length and directory (a temporary directory by default).

    Returns
    -------
    ProfileReport
        Detailed profiling results.
    """
    if gm._dirty or gm._compiled_step is None:
        gm.compile()

    report = ProfileReport(
        n_nodes=len(gm._nodes),
        n_coupling_groups=len(gm._coupling_groups),
        device=str(jax.devices()[0]),
    )

    # State sizes
    for name in gm._nodes:
        s = gm._state.get(name, {})
        elems = sum(np.asarray(v).size for v in s.values())
        report.node_sizes[name] = elems
        report.total_state_elements += elems

    # JIT compile timing (first step); reset to the initial state the way
    # ``compile`` seeds it (normalised, ``_meta`` intact) so the timed
    # steps never include a retrace.
    gm.reset_state()

    t0 = time.perf_counter()
    gm.step(external_inputs)
    jax.block_until_ready(jax.tree.leaves(gm._state))
    report.jit_compile_ms = (time.perf_counter() - t0) * 1000

    # Warmup remaining
    for _ in range(max(0, n_warmup - 1)):
        gm.step(external_inputs)
    jax.block_until_ready(jax.tree.leaves(gm._state))

    # Benchmark steps (nothing else touches the device in this loop)
    step_arr = _time_steps(gm, external_inputs, n_steps)
    report.mean_step_ms = float(np.mean(step_arr))
    report.std_step_ms = float(np.std(step_arr))
    report.median_step_ms = float(np.median(step_arr))
    report.p95_step_ms = float(np.percentile(step_arr, 95))
    report.total_run_ms = float(np.sum(step_arr))
    report.n_steps = n_steps
    if report.mean_step_ms > 0:
        report.steps_per_second = 1000.0 / report.mean_step_ms

    # Dispatch floor: a jitted identity on the same pytree.
    ext = external_inputs if external_inputs is not None else gm._default_external_inputs()
    identity = jax.jit(lambda s, e, p: s)
    jax.block_until_ready(identity(gm._state, ext, gm.params))
    floor = []
    for _ in range(min(n_steps, 50)):
        t0 = time.perf_counter()
        jax.block_until_ready(identity(gm._state, ext, gm.params))
        floor.append((time.perf_counter() - t0) * 1000)
    report.dispatch_floor_ms = float(np.mean(floor)) if floor else 0.0

    # Coupling iteration statistics over a run (a separate loop: reading
    # _meta forces a device->host sync that must not pollute the timing).
    group_keys = _meta_group_keys(gm)
    if group_keys:
        n_stat = min(n_steps, 50) if n_stat_steps is None else n_stat_steps
        if n_stat_steps is not None:
            # Pin the window's position as well as its length: without
            # this the pass starts wherever the timed run happened to
            # stop, which is n_warmup + n_steps into the trajectory.
            gm.reset_state()
            for _ in range(max(0, n_warmup)):
                gm.step(external_inputs)
            jax.block_until_ready(jax.tree.leaves(gm._state))
        iters = {k[0]: [] for k in group_keys}
        conv = {k[0]: [] for k in group_keys}
        for _ in range(n_stat):
            gm.step(external_inputs)
            meta = gm._state.get("_meta", {})
            for key, iter_key, res_key, thr, cap in group_keys:
                if iter_key in meta:
                    iters[key].append(int(meta[iter_key]))
                    conv[key].append(float(meta[res_key]) <= thr)
        for key, _ik, _rk, _thr, cap in group_keys:
            if iters[key]:
                a = np.asarray(iters[key])
                report.coupling_iter_stats[key] = {
                    "mean": float(a.mean()), "min": int(a.min()), "max": int(a.max()),
                    "cap": cap,
                    # ``iterations`` counts body iterations after the
                    # first pass, so the cap is reached at cap - 1.
                    "at_cap_fraction": float(np.mean(a >= cap - 1)),
                    "converged_fraction": float(np.mean(conv[key])),
                    "n": int(a.size),
                }

    # Per-node cost estimation: run each node's update in isolation
    for name, spec in gm._nodes.items():
        node_state = gm._state.get(name, spec.node.initial_state())
        bi = {}  # empty boundary inputs
        update_fn = jax.jit(spec.update_fn)
        # Warmup
        _ = update_fn(node_state, bi, spec.timestep)
        jax.block_until_ready(_)
        # Measure
        times = []
        for _ in range(min(50, n_steps)):
            t0 = time.perf_counter()
            r = update_fn(node_state, bi, spec.timestep)
            jax.block_until_ready(jax.tree.leaves(r))
            times.append((time.perf_counter() - t0) * 1000)
        report.node_times_ms[name] = float(np.mean(times))

    # Coupling overhead: measured (one-iteration variant) or estimated.
    report.sum_node_ms = sum(report.node_times_ms.values())
    if group_keys and measure_coupling:
        with _one_iteration_variant(gm):
            gm.step(external_inputs)
            jax.block_until_ready(jax.tree.leaves(gm._state))
            for _ in range(max(0, n_warmup - 1)):
                gm.step(external_inputs)
            one = _time_steps(gm, external_inputs, n_steps)
        report.one_iteration_step_ms = float(np.mean(one))
        report.coupling_overhead_ms = max(0.0, report.mean_step_ms - report.one_iteration_step_ms)
        report.coupling_overhead_method = "measured"
        extra = sum(
            max(st["mean"] - 1.0, 0.0) for st in report.coupling_iter_stats.values()
        )
        report.coupling_per_iteration_ms = (
            report.coupling_overhead_ms / extra if extra > 0 else 0.0
        )
    elif group_keys:
        report.coupling_overhead_ms = max(0.0, report.mean_step_ms - report.sum_node_ms)
        report.coupling_overhead_method = "estimated"

    # Last-step coupling diagnostics (compat)
    diag = gm.coupling_diagnostics()
    for key, info in diag.items():
        report.coupling_iters[key] = info.get("iterations", 0)

    # Optional trace attribution
    if trace:
        log_dir = trace_dir or tempfile.mkdtemp(prefix="maddening_profile_")
        # Warm (the coupling measurement recompiled the step) so the
        # trace holds steady-state steps only.
        for _ in range(max(1, n_warmup)):
            gm.step(external_inputs)
        jax.block_until_ready(jax.tree.leaves(gm._state))
        step_fn_name = getattr(gm._compiled_step, "__name__", "")
        jax.profiler.start_trace(log_dir, create_perfetto_trace=True)
        try:
            for _ in range(trace_steps):
                gm.step(external_inputs)
            jax.block_until_ready(jax.tree.leaves(gm._state))
        finally:
            jax.profiler.stop_trace()
        report.trace = _summarise_trace(
            log_dir, trace_steps, report.mean_step_ms, step_fn_name,
        )

    # Identify bottleneck
    if report.node_times_ms:
        worst = max(report.node_times_ms, key=report.node_times_ms.get)
        worst_ms = report.node_times_ms[worst]
        if worst_ms > 0.8 * report.mean_step_ms:
            report.bottleneck = (
                f"{worst} ({worst_ms:.2f}ms, "
                f"{worst_ms/report.mean_step_ms*100:.0f}% of step)"
            )
        elif report.coupling_overhead_ms > 0.3 * report.mean_step_ms:
            report.bottleneck = (
                f"Coupling overhead ({report.coupling_overhead_ms:.2f}ms, "
                f"{report.coupling_overhead_ms/report.mean_step_ms*100:.0f}%)"
            )

    # Recommendations
    for key, st in report.coupling_iter_stats.items():
        if st["at_cap_fraction"] > 0.0:
            report.recommendations.append(
                f"Coupling group '{key}' hit max_iterations={st['cap']} on "
                f"{st['at_cap_fraction'] * 100:.0f}% of steps "
                f"(converged {st['converged_fraction'] * 100:.0f}%): raise the "
                f"cap or loosen the tolerance; the IFT gradient is unreliable "
                f"on unconverged steps."
            )
        elif st["mean"] >= 8:
            report.recommendations.append(
                f"Coupling group '{key}' uses {st['mean']:.1f} iterations on "
                f"average — consider acceleration='iqn-ils' or removing the "
                f"group if the coupling is weak."
            )
    for key, iters in report.coupling_iters.items():
        if key not in report.coupling_iter_stats and iters >= 8:
            report.recommendations.append(
                f"Coupling group '{key}' uses {iters} iterations — "
                f"consider acceleration='iqn-ils' or removing the coupling "
                f"group if the coupling is weakly coupled."
            )
    if report.trace is not None and "cpu" not in report.device.lower():
        if report.trace.device_busy_fraction < 0.3 and report.trace.n_kernels_per_step > 50:
            report.recommendations.append(
                f"Device busy only {report.trace.device_busy_fraction * 100:.0f}% of "
                f"the step with {report.trace.n_kernels_per_step:.0f} kernels/step: "
                f"the step is kernel-launch bound, not compute bound.  Fewer, larger "
                f"nodes (or fusing per-iteration work) beats optimising any one node."
            )
    if report.dispatch_floor_ms > 0.5 * report.mean_step_ms > 0:
        report.recommendations.append(
            f"Dispatch floor ({report.dispatch_floor_ms:.2f} ms) is more than half "
            f"the step: batch steps with run_scan instead of calling step() per "
            f"timestep."
        )
    for name, ms in report.node_times_ms.items():
        size = report.node_sizes.get(name, 0)
        if size > 100000 and ms > 5.0:
            report.recommendations.append(
                f"Node '{name}' has {size:,} elements and takes {ms:.1f}ms. "
                f"Consider GPU acceleration or a coarser grid."
            )
    if report.mean_step_ms > 50:
        report.recommendations.append(
            f"Step time ({report.mean_step_ms:.1f}ms) limits real-time "
            f"to {report.steps_per_second:.0f} Hz. Use --gpu for faster "
            f"execution or reduce grid resolution."
        )

    return report


# ---------------------------------------------------------------------------
# v0.2 #9: Perfetto trace export + jax.profiler integration
# ---------------------------------------------------------------------------


def profile_report_to_perfetto(report: ProfileReport) -> dict:
    """Convert a :class:`ProfileReport` to Chrome/Perfetto Trace Event format.

    The result is JSON-serialisable; write it to ``profile.json`` and
    open in https://ui.perfetto.dev (drag-and-drop) for an interactive
    flame-graph view.  No conversion tooling needed — the schema is
    documented at
    https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/
    and natively understood by Perfetto's "JSON" frontend.

    The trace contains:

    * A ``"step"`` event spanning the entire benchmarked run.
    * Per-node ``"<node>.update"`` events laid out in series within
      each step (since we measured them in isolation, the timeline
      is reconstructed rather than a literal recording).
    * A meta event with ``args`` capturing throughput, coupling
      overhead, and the bottleneck summary so the front-matter
      survives in the trace.

    Notes
    -----
    Because the source data is mean-step timings rather than a
    recorded trace, the per-node bars are an aggregate approximation,
    not literal sequential timing.  For genuine wall-clock traces
    use :func:`start_jax_trace` / :func:`stop_jax_trace` which emit
    XLA-level events from ``jax.profiler``.
    """
    events = []
    pid = 1
    tid = 1

    # Microsecond-resolution timestamps (Perfetto convention).
    step_us = report.mean_step_ms * 1000.0
    total_us = report.total_run_ms * 1000.0

    events.append({
        "name": f"run x{report.n_steps}",
        "cat": "graph",
        "ph": "X",  # complete event (begin+dur)
        "ts": 0.0,
        "dur": total_us,
        "pid": pid,
        "tid": tid,
        "args": {
            "n_steps": report.n_steps,
            "mean_step_ms": report.mean_step_ms,
            "steps_per_second": report.steps_per_second,
            "bottleneck": report.bottleneck or "none",
        },
    })

    # Lay out per-node updates inside one representative step.
    cursor_us = 0.0
    for name, ms in sorted(report.node_times_ms.items(), key=lambda x: -x[1]):
        dur_us = ms * 1000.0
        events.append({
            "name": f"{name}.update",
            "cat": "node",
            "ph": "X",
            "ts": cursor_us,
            "dur": dur_us,
            "pid": pid,
            "tid": tid + 1,
            "args": {
                "node": name,
                "elements": report.node_sizes.get(name, 0),
                "share_of_step_pct": (
                    dur_us / step_us * 100 if step_us > 0 else 0.0
                ),
            },
        })
        cursor_us += dur_us

    if report.coupling_overhead_ms > 0:
        events.append({
            "name": "coupling_overhead",
            "cat": "coupling",
            "ph": "X",
            "ts": cursor_us,
            "dur": report.coupling_overhead_ms * 1000.0,
            "pid": pid,
            "tid": tid + 1,
            "args": {
                "n_groups": report.n_coupling_groups,
                "method": report.coupling_overhead_method,
                "iterations_per_group": dict(report.coupling_iters),
                "iteration_stats": dict(report.coupling_iter_stats),
            },
        })

    return {
        "traceEvents": events,
        "displayTimeUnit": "us",
        "otherData": {
            "source": "maddening.core.simulation.profiler",
            "n_steps": report.n_steps,
            "n_nodes": report.n_nodes,
            "recommendations": list(report.recommendations),
            "jit_compile_ms": report.jit_compile_ms,
        },
    }


class JaxProfilerSession:
    """Context manager wrapper around :mod:`jax.profiler`.

    JAX writes a TensorBoard-style profile under ``log_dir/plugins/profile/
    <timestamp>/...xplane.pb`` which the TensorBoard "Trace Viewer" plugin
    visualises with a Perfetto front-end.

    Usage::

        with JaxProfilerSession() as sess:
            for _ in range(100):
                gm.step()
        sess.log_dir   # the captured trace directory

    For one-shot REST use, :func:`start_jax_trace` and
    :func:`stop_jax_trace` provide a non-blocking begin/end pair.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir: Optional[str] = log_dir
        self._owns_dir = log_dir is None
        self._active = False

    def __enter__(self) -> "JaxProfilerSession":
        if self.log_dir is None:
            self.log_dir = tempfile.mkdtemp(prefix="maddening_jaxtrace_")
        jax.profiler.start_trace(self.log_dir)
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._active:
            jax.profiler.stop_trace()
            self._active = False

    @property
    def active(self) -> bool:
        return self._active


# Module-level singleton state for the REST endpoints.
_active_jax_trace: Optional[JaxProfilerSession] = None


def start_jax_trace(log_dir: Optional[str] = None) -> str:
    """Begin a JAX trace; subsequent step() calls are recorded.

    Returns the directory path the trace will be written to.  Pair
    with :func:`stop_jax_trace`; starting a second trace while one is
    active raises ``RuntimeError``.
    """
    global _active_jax_trace
    if _active_jax_trace is not None and _active_jax_trace.active:
        raise RuntimeError("A JAX trace is already active; stop it first.")
    sess = JaxProfilerSession(log_dir=log_dir)
    sess.__enter__()
    _active_jax_trace = sess
    return sess.log_dir or ""


def stop_jax_trace() -> str:
    """End the active JAX trace and return the path to its log dir."""
    global _active_jax_trace
    if _active_jax_trace is None or not _active_jax_trace.active:
        raise RuntimeError("No JAX trace is active.")
    log_dir = _active_jax_trace.log_dir or ""
    _active_jax_trace.__exit__(None, None, None)
    _active_jax_trace = None
    return log_dir


def jax_trace_active() -> bool:
    """Return whether a JAX trace is currently recording."""
    return _active_jax_trace is not None and _active_jax_trace.active


def tar_trace_dir(log_dir: str) -> bytes:
    """Pack a trace directory into a gzipped tarball (in-memory).

    Useful for shipping JAX traces out of an ephemeral cloud instance
    via the REST endpoint.
    """
    import io
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(log_dir, arcname=Path(log_dir).name)
    return buf.getvalue()
