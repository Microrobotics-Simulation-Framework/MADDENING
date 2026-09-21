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
    # A difference of two windows timed at *different* points of the
    # trajectory.  See ``measure_coupling`` in :func:`profile_graph` for
    # which points, why that is sound, and what it was measured to cost.
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

    # Deterministic compilation counts (``counts=True``, the default).
    # Unlike every timing above, these reproduce exactly on any machine.
    counts: Optional["CompileCounts"] = None

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

        if self.counts is not None:
            lines.append(f"")
            lines.extend(self.counts.lines())

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


# ---------------------------------------------------------------------------
# Deterministic compilation counts
# ---------------------------------------------------------------------------


@stability(StabilityLevel.EVOLVING)
@dataclass
class CompileCounts:
    """Compilation counts for a graph: integers, never durations.

    Compile time is what a user of MADDENING feels first, and a silent
    regression in it would ship unnoticed -- but wall-clock cannot gate
    it.  A per-step timing comparison in this repository failed CI at
    3.27x against a 3.0x bound on a pull request that changed no code at
    all, because the number it read was the runner and not the solver
    (see ``tests/core/test_coupling_while_default.py``).

    These counts are what that comparison should have been measuring.
    They are exactly reproducible: the same graph on the same JAX
    version yields the same integers on any machine, under any load,
    however many other jobs share the box.

    Attributes
    ----------
    retrace_count : int
        Python traces of the compiled step since ``compile()``
        (:attr:`GraphManager.trace_count`).  One is healthy.  More means
        something in the call signature keeps changing -- a weak-typed
        leaf, a drifting dtype, a non-static argument -- and every one of
        them is a full XLA compile the user pays for.  This is the single
        most valuable number here: an unexpected retrace is the classic
        silent regression, and it is the one count that does not depend
        on the JAX version at all, because it is counted in Python.
    jaxpr_primitive_count : int
        Primitives in the step's jaxpr, recursing into every sub-jaxpr
        (``scan``/``while``/``cond`` bodies are counted once, not once
        per iteration).  A structural measure of how much work the graph
        builder emits.
    hlo_op_count : int
        Operations in the lowered StableHLO module, excluding the
        top-level ``module`` op.  Deliberately the *pre*-optimisation
        module: it is what MADDENING hands to XLA, so it measures
        MADDENING's own graph construction rather than the backend's
        fusion decisions, which differ by platform and would make the
        number machine-dependent.
    scan_steps : int
        Length of the ``run_scan`` program the scan counts describe;
        ``0`` when no scan was measured.
    scan_retrace_count : int
        Python traces of scan programs since ``compile()``
        (:attr:`GraphManager.scan_trace_count`).
    scan_jaxpr_primitive_count, scan_hlo_op_count : int
        The same two structural counts for the scan program.

    Notes
    -----
    Not timing.  :class:`ProfileReport` carries the wall-clock numbers;
    they are worth trending and must not gate.
    """

    retrace_count: int = 0
    jaxpr_primitive_count: int = 0
    hlo_op_count: int = 0
    scan_steps: int = 0
    scan_retrace_count: int = 0
    scan_jaxpr_primitive_count: int = 0
    scan_hlo_op_count: int = 0

    def as_dict(self) -> dict[str, int]:
        """The counts as a plain ``dict`` of ``int``, for serialisation.

        Scan fields are omitted entirely when no scan was measured, so a
        baseline never records a zero that could be mistaken for a
        measured count of nothing.
        """
        out = {
            "retrace_count": int(self.retrace_count),
            "jaxpr_primitive_count": int(self.jaxpr_primitive_count),
            "hlo_op_count": int(self.hlo_op_count),
        }
        if self.scan_steps:
            out.update(
                scan_steps=int(self.scan_steps),
                scan_retrace_count=int(self.scan_retrace_count),
                scan_jaxpr_primitive_count=int(self.scan_jaxpr_primitive_count),
                scan_hlo_op_count=int(self.scan_hlo_op_count),
            )
        return out

    def lines(self) -> list[str]:
        """Rendered lines for :meth:`ProfileReport.__str__`."""
        out = [
            "  Compile counts (deterministic; no wall-clock):",
            f"    step retraces:     {self.retrace_count:>7d}"
            + ("" if self.retrace_count == 1 else "   <-- expected 1"),
            f"    jaxpr primitives:  {self.jaxpr_primitive_count:>7d}",
            f"    lowered HLO ops:   {self.hlo_op_count:>7d}",
        ]
        if self.scan_steps:
            out += [
                f"    scan ({self.scan_steps} steps) retraces: "
                f"{self.scan_retrace_count:>3d}",
                f"    scan jaxpr primitives: {self.scan_jaxpr_primitive_count:>7d}",
                f"    scan lowered HLO ops:  {self.scan_hlo_op_count:>7d}",
            ]
        return out


def _subjaxprs(jaxpr) -> list:
    """Sub-jaxprs of *jaxpr*, one level down.

    ``jax.extend.core.subjaxprs`` is the supported spelling; the
    duck-typed fallback keeps this working on a JAX that moves it again
    (``jax.core.ClosedJaxpr`` disappeared in 0.11, which is exactly the
    breakage this fallback exists for).
    """
    try:
        from jax.extend.core import subjaxprs
    except ImportError:  # pragma: no cover - JAX layout change
        pass
    else:
        return list(subjaxprs(jaxpr))

    found = []  # pragma: no cover - JAX layout change
    for eqn in jaxpr.eqns:  # pragma: no cover - JAX layout change
        for value in eqn.params.values():
            items = value if isinstance(value, (tuple, list)) else (value,)
            for item in items:
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    found.append(inner)
    return found


@stability(StabilityLevel.EVOLVING)
def count_jaxpr_primitives(jaxpr) -> int:
    """Primitives in *jaxpr*, recursing into every sub-jaxpr.

    Parameters
    ----------
    jaxpr : Jaxpr or ClosedJaxpr
        The jaxpr to count.  A ``ClosedJaxpr`` is unwrapped.

    Returns
    -------
    int
        Total equation count.  A ``scan`` contributes its own primitive
        plus the primitives of its body **once**, not once per
        iteration, so the number describes the program's structure and
        not the trip count -- which is what makes it comparable between
        a 10-step and a 1000-step run of the same graph.
    """
    jaxpr = getattr(jaxpr, "jaxpr", jaxpr)
    return len(jaxpr.eqns) + sum(count_jaxpr_primitives(s) for s in _subjaxprs(jaxpr))


@stability(StabilityLevel.EVOLVING)
def count_hlo_ops(lowered) -> int:
    """Operations in a ``jax.stages.Lowered``'s StableHLO module.

    Every MLIR operation in the module is counted, at every nesting
    depth, excluding the top-level ``module`` op itself (``func.func``,
    ``return`` and the bodies of ``stablehlo.while`` / ``stablehlo.case``
    are all included).  Counting everything rather than filtering by
    dialect keeps the number stable across JAX versions that rename or
    re-dialect an op.

    Parameters
    ----------
    lowered : jax.stages.Lowered
        The result of ``jitted_fn.lower(*args)``.

    Returns
    -------
    int
        Operation count of the pre-optimisation module.
    """
    total = 0
    stack = [lowered.compiler_ir().operation]
    while stack:
        op = stack.pop()
        for region in op.regions:
            for block in region.blocks:
                for child in block.operations:
                    total += 1
                    stack.append(child)
    return total


@stability(StabilityLevel.EVOLVING)
def compile_counts(
    gm,
    *,
    external_inputs: Optional[dict] = None,
    params: Optional[dict] = None,
    scan_steps: int = 0,
    warmup_steps: int = 4,
) -> CompileCounts:
    """Measure :class:`CompileCounts` for a compiled graph.

    Parameters
    ----------
    gm : GraphManager
        The graph.  Compiled if dirty.
    external_inputs : dict or None
        External inputs for the step whose compilation is measured.
        ``None`` uses the graph's defaults.
    params : dict or None
        Graph parameter pytree; ``None`` uses ``gm.params``.
    scan_steps : int
        When positive, also build a ``run_scan`` program of this length
        and count it.  Costs one scan compile.
    warmup_steps : int
        Steps to run before reading the counts.  **A single step is not
        enough**, and the default is 4 for a specific reason: the retrace
        bugs this repository has actually had did not retrace on the
        first step.  The weak-typed-seed bug fixed in
        ``tests/core/test_step_retrace.py`` traced once on step 1, again
        on step 2 when a weak leaf came back strongly typed, and a third
        time on step 3 for leaves that only changed later.  Measured
        after one step it looks perfectly healthy.  Pass ``0`` when the
        caller has already run the graph far enough, as
        :func:`profile_graph` has.

    Returns
    -------
    CompileCounts
        The counts.

    Notes
    -----
    The graph is left ``warmup_steps`` steps further on than it was
    found: warming up is the price of a meaningful retrace count, and
    silently rewinding it would hide that from the caller.  The scan
    measurement is different -- it advances the graph by ``scan_steps``
    purely as an implementation detail -- so that one *is* rolled back.

    Lowering the step to read its jaxpr and its HLO re-enters
    ``jax.jit``, which serves both from its jaxpr cache and so does not
    retrace a warm graph.  ``_n_traces`` is nevertheless snapshotted and
    restored around the measurement, so that measuring a graph can never
    move the very number being measured -- on any JAX whose caching
    differs from today's.  ``tests/core/test_compile_counts.py`` pins
    that: it asserts ``trace_count`` is unchanged by a measurement, an
    assertion that passes today and would be the only warning if a JAX
    upgrade changed it.
    """
    if gm._dirty or gm._compiled_step is None:
        gm.compile()

    for _ in range(max(0, warmup_steps)):
        gm.step(external_inputs, params=params)
    # A graph nobody has stepped has a retrace count of 0 and no compiled
    # program to lower; one step is the floor even when warmup is off.
    if gm.trace_count == 0:
        gm.step(external_inputs, params=params)
    jax.block_until_ready(jax.tree.leaves(gm._state))

    resolved_ext = gm._resolve_external_inputs(external_inputs)
    resolved_params = gm._params_or_default(params)

    counts = CompileCounts(retrace_count=int(gm.trace_count))

    step_fn = gm._compiled_step
    assert step_fn is not None  # the compile guard at the top of this function

    saved_traces = gm._n_traces
    try:
        lowered = step_fn.lower(gm._state, resolved_ext, resolved_params)
        counts.hlo_op_count = count_hlo_ops(lowered)
        counts.jaxpr_primitive_count = count_jaxpr_primitives(
            jax.make_jaxpr(step_fn)(gm._state, resolved_ext, resolved_params)
        )
    finally:
        gm._n_traces = saved_traces

    if scan_steps > 0:
        counts.scan_steps = int(scan_steps)
        saved_state = jax.tree.map(lambda x: x, gm._state)
        try:
            # Populates ``_scan_cache``; ``scan_trace_count`` counts the
            # Python traces, one per XLA compile of a scan program.  It
            # is deliberately *not* restored afterwards the way
            # ``_n_traces`` is: this call really did build a scan
            # program, the program stays in ``_scan_cache``, and a
            # counter rolled back below a cache that still holds the
            # program would report 0 traces for a program that exists.
            gm.run_scan(scan_steps, external_inputs, params=params)
            counts.scan_retrace_count = int(gm.scan_trace_count)
            key = (gm._compile_generation, "run_scan", int(scan_steps))
            scan_fn = gm._scan_cache[key]
            scan_lowered = scan_fn.lower(saved_state, resolved_ext, resolved_params)
            counts.scan_hlo_op_count = count_hlo_ops(scan_lowered)
            counts.scan_jaxpr_primitive_count = count_jaxpr_primitives(
                jax.make_jaxpr(scan_fn)(saved_state, resolved_ext, resolved_params)
            )
        finally:
            gm._state = saved_state

    return counts


def _meta_group_keys(gm) -> list[tuple[str, str, str, str, float, int]]:
    """``(group_key, iter_key, res_key, amp_key, threshold, cap)``.

    ``amp_key`` carries the amplification ``1/(1 - rho)`` the
    convergence flag is built from: ``converged_fraction`` has to test
    the same estimated distance to the fixed point that
    ``coupling_diagnostics()`` does, not the raw residual.
    """
    out = []
    for g in gm._coupling_groups:
        key = "+".join(sorted(g.nodes))
        thr = 1.0 if g.convergence_norm in ("mixed", "interface") else float(g.tolerance)
        out.append((key, f"coupling_{key}_iterations", f"coupling_{key}_residual",
                    f"coupling_{key}_amplification", thr, int(g.max_iterations)))
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
    counts: bool = True,
    count_scan_steps: int = 0,
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

        The two windows it subtracts start at **different points of the
        trajectory**, and stay that way on purpose.  ``mean_step_ms`` is
        timed ``n_warmup`` steps after a ``reset_state``;
        ``one_iteration_step_ms`` is timed from wherever the timed run
        and the coupling-statistics pass left the state, plus another
        ``n_warmup``.  Nothing leaks -- :func:`_one_iteration_variant`
        saves and restores state, groups and compiled step -- but the
        *start* is not pinned the way ``n_stat_steps`` pins the
        statistics pass, and unlike that pass it does not need to be.

        The reason is structural rather than lucky:
        ``max_iterations <= 1`` returns straight after the single
        staggered pass, before any ``while_loop``, accelerator or IFT
        solve is reached, so the capped step is straight-line code on
        fixed shapes and costs the same whatever state it starts from.
        Its measured iteration count is exactly one from every
        trajectory position, which
        ``test_one_iteration_variant_runs_one_pass_from_any_position``
        pins.  Pinning both windows to the same start was measured on
        the compute-bound ``expensive-pair`` fixture (two 1e5-cell heat
        grids; 10 interleaved repeats, a fresh graph per measurement)
        and moved ``coupling_overhead_ms`` by -0.11%, against a
        run-to-run scatter of 8.1% -- two orders of magnitude below the
        noise, so the computation is left as it is.  Should a cap of one
        ever regain a data-dependent trip count, the subtraction would
        begin comparing two different workloads and this window would
        have to be pinned.
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
    counts : bool
        Measure :class:`CompileCounts` -- retraces, jaxpr primitives and
        lowered HLO ops.  Cheap (no device work) and, unlike every
        timing in the report, exactly reproducible, which is why the
        regression gate reads these and not the clock.
    count_scan_steps : int
        When positive and ``counts`` is set, also build a ``run_scan``
        program of this length and count it.  Costs one scan compile.
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

    # Compilation counts, taken here rather than at the end: the coupling
    # measurement below recompiles the step (resetting ``trace_count``),
    # so anywhere after it the retrace count would describe the
    # profiler's own recompile instead of the run that was just timed.
    if counts:
        # ``warmup_steps=0``: the timed run above has already stepped the
        # graph ``n_warmup + n_steps`` times, well past the point where a
        # late retrace would have shown up, and stepping further here
        # would move the window the coupling statistics below are taken
        # from.
        report.counts = compile_counts(
            gm, external_inputs=external_inputs, scan_steps=count_scan_steps,
            warmup_steps=0,
        )

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
            for key, iter_key, res_key, amp_key, thr, cap in group_keys:
                if iter_key in meta:
                    iters[key].append(int(meta[iter_key]))
                    amp = float(meta.get(amp_key, 0.0))
                    est = float(meta[res_key]) * (amp if amp >= 1.0 else 1.0)
                    conv[key].append(est <= thr)
        for key, _ik, _rk, _ak, _thr, cap in group_keys:
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
    #
    # The one-iteration window is deliberately *not* pinned to where
    # ``mean_step_ms`` was measured -- it starts from wherever the timed
    # run and the statistics pass left the state.  A group capped at one
    # iteration has no data-dependent control flow, so the capped step
    # costs the same from any state and the subtraction stays valid;
    # measured impact of pinning it, -0.11% against 8.1% run-to-run
    # scatter.  Full reasoning on ``measure_coupling`` above.
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
        # `dict.get` as the key function is typed as returning `float |
        # None`, which is not orderable; subscripting says what is meant.
        worst = max(report.node_times_ms, key=lambda k: report.node_times_ms[k])
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
