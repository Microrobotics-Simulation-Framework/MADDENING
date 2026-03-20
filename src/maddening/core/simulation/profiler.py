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
1. Measuring JIT compilation time (first step)
2. Measuring warmed-up step time (subsequent steps)
3. Estimating per-node cost by running each node in isolation
4. Computing coupling overhead as total - sum(nodes)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


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
    total_run_ms: float = 0.0
    steps_per_second: float = 0.0

    # Per-node timing (estimated)
    node_times_ms: dict[str, float] = field(default_factory=dict)

    # Coupling overhead
    n_coupling_groups: int = 0
    coupling_overhead_ms: float = 0.0
    coupling_iters: dict[str, int] = field(default_factory=dict)

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
            f"  JIT compile:  {self.jit_compile_ms:>8.1f} ms",
            f"  Mean step:    {self.mean_step_ms:>8.2f} ms "
            f"(+/- {self.std_step_ms:.2f})",
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
                f"  Coupling overhead: {self.coupling_overhead_ms:.2f} ms"
            )
            for group_key, iters in self.coupling_iters.items():
                lines.append(f"    {group_key}: {iters} iterations")

        if self.bottleneck:
            lines.append(f"")
            lines.append(f"  Bottleneck: {self.bottleneck}")

        if self.recommendations:
            lines.append(f"")
            lines.append(f"  Recommendations:")
            for rec in self.recommendations:
                lines.append(f"    - {rec}")

        return "\n".join(lines)


def profile_graph(
    gm,
    n_steps: int = 100,
    n_warmup: int = 3,
    external_inputs: Optional[dict] = None,
) -> ProfileReport:
    """Profile a compiled GraphManager.

    Parameters
    ----------
    gm : GraphManager
        A compiled (or compilable) graph.
    n_steps : int
        Number of steps to benchmark (after warmup).
    n_warmup : int
        Warmup steps (included in JIT timing, excluded from
        per-step timing).
    external_inputs : dict or None
        External inputs for each step.

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
    )

    # State sizes
    for name in gm._nodes:
        s = gm._state.get(name, {})
        elems = sum(np.asarray(v).size for v in s.values())
        report.node_sizes[name] = elems
        report.total_state_elements += elems

    # JIT compile timing (first step)
    # Reset state to initial
    for name, spec in gm._nodes.items():
        gm._state[name] = spec.node.initial_state()
    if gm._is_multirate:
        gm._state["_meta"] = {"step_count": jnp.array(0, dtype=jnp.int32)}

    t0 = time.perf_counter()
    gm.step(external_inputs)
    jax.block_until_ready(jax.tree.leaves(gm._state))
    report.jit_compile_ms = (time.perf_counter() - t0) * 1000

    # Warmup remaining
    for _ in range(max(0, n_warmup - 1)):
        gm.step(external_inputs)
    jax.block_until_ready(jax.tree.leaves(gm._state))

    # Benchmark steps
    step_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        gm.step(external_inputs)
        jax.block_until_ready(jax.tree.leaves(gm._state))
        step_times.append((time.perf_counter() - t0) * 1000)

    step_arr = np.array(step_times)
    report.mean_step_ms = float(np.mean(step_arr))
    report.std_step_ms = float(np.std(step_arr))
    report.total_run_ms = float(np.sum(step_arr))
    report.n_steps = n_steps
    if report.mean_step_ms > 0:
        report.steps_per_second = 1000.0 / report.mean_step_ms

    # Per-node cost estimation: run each node's update in isolation
    ext = gm._default_external_inputs()
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

    # Coupling overhead
    sum_node_ms = sum(report.node_times_ms.values())
    report.coupling_overhead_ms = max(0, report.mean_step_ms - sum_node_ms)

    # Coupling diagnostics
    diag = gm.coupling_diagnostics()
    for key, info in diag.items():
        report.coupling_iters[key] = info.get("iterations", 0)

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
    for key, iters in report.coupling_iters.items():
        if iters >= 8:
            report.recommendations.append(
                f"Coupling group '{key}' uses {iters} iterations — "
                f"consider acceleration='iqn-ils' or removing the coupling "
                f"group if the coupling is weakly coupled."
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
