#!/usr/bin/env python
"""
Performance comparison: run() vs run_scan() vs run_scan_with_history().

Demonstrates the performance advantages of using jax.lax.scan-based
execution over Python-loop execution in MADDENING.

Benchmarks:
1. run()                  -- Python loop calling JIT-compiled step
2. run_scan()             -- full loop pushed into XLA via lax.scan
3. run_scan_with_history() -- lax.scan with intermediate state recording

Also shows how to use the history arrays from run_scan_with_history()
for post-hoc analysis (min, max, mean, energy calculations).

Usage
-----
    cd /home/nick/MSF/MADDENING
    source ../venvs/.maddening/bin/activate
    python maddening/examples/scan_performance.py
"""

import sys
import os
import time

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import jax
import jax.numpy as jnp

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


def build_graph() -> GraphManager:
    """Create a fresh bouncing-ball graph."""
    gm = GraphManager()
    table = TableNode(name="table", timestep=0.01, position=0.0)
    ball = BallNode(
        name="ball",
        timestep=0.01,
        initial_position=5.0,
        initial_velocity=0.0,
        elasticity=0.8,
    )
    gm.add_node(table)
    gm.add_node(ball)
    gm.add_edge(
        source="table", target="ball",
        source_field="position", target_field="table_position",
    )
    gm.compile()
    return gm


def benchmark_run(n_steps: int) -> tuple[float, dict]:
    """Benchmark gm.run() with Python loop (includes first-call JIT overhead)."""
    gm = build_graph()

    start = time.perf_counter()
    gm.run(n_steps)
    # Block until computation is done
    final = gm.get_node_state("ball")
    _ = float(final["position"])
    elapsed = time.perf_counter() - start

    return elapsed, {"position": float(final["position"]),
                     "velocity": float(final["velocity"])}


def benchmark_scan(n_steps: int) -> tuple[float, dict]:
    """Benchmark gm.run_scan() with lax.scan (includes trace + compile time)."""
    gm = build_graph()

    start = time.perf_counter()
    final_state = gm.run_scan(n_steps)
    # Block until computation is done
    _ = float(final_state["ball"]["position"])
    elapsed = time.perf_counter() - start

    return elapsed, {"position": float(final_state["ball"]["position"]),
                     "velocity": float(final_state["ball"]["velocity"])}


def benchmark_scan_history(n_steps: int) -> tuple[float, dict, dict]:
    """Benchmark gm.run_scan_with_history() (includes trace + compile time)."""
    gm = build_graph()

    start = time.perf_counter()
    final_state, history = gm.run_scan_with_history(n_steps)
    # Block until computation is done
    _ = float(final_state["ball"]["position"])
    elapsed = time.perf_counter() - start

    return elapsed, {"position": float(final_state["ball"]["position"]),
                     "velocity": float(final_state["ball"]["velocity"])}, history


def main() -> None:
    n_steps = 10000
    dt = 0.01

    print(f"Benchmarking {n_steps} steps (dt={dt}, total time={n_steps * dt:.0f}s)")
    print("(All timings include first-call JIT compilation overhead)")
    print("=" * 60)

    # ---- Benchmark run() ------------------------------------------------
    print("\n1. gm.run() [Python loop + JIT step]...")
    t_run, final_run = benchmark_run(n_steps)
    print(f"   Time: {t_run:.4f}s")
    print(f"   Final: pos={final_run['position']:.6f}, vel={final_run['velocity']:.6f}")

    # ---- Benchmark run_scan() -------------------------------------------
    print("\n2. gm.run_scan() [lax.scan, no history]...")
    t_scan, final_scan = benchmark_scan(n_steps)
    print(f"   Time: {t_scan:.4f}s")
    print(f"   Final: pos={final_scan['position']:.6f}, vel={final_scan['velocity']:.6f}")

    # ---- Benchmark run_scan_with_history() ------------------------------
    print("\n3. gm.run_scan_with_history() [lax.scan + history]...")
    t_hist, final_hist, history = benchmark_scan_history(n_steps)
    print(f"   Time: {t_hist:.4f}s")
    print(f"   Final: pos={final_hist['position']:.6f}, vel={final_hist['velocity']:.6f}")

    # ---- Speedup report -------------------------------------------------
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    print(f"  run():                   {t_run:.4f}s")
    print(f"  run_scan():              {t_scan:.4f}s")
    print(f"  run_scan_with_history(): {t_hist:.4f}s")

    if t_scan > 0:
        print(f"\n  Speedup run_scan over run:         {t_run / t_scan:.1f}x")
    if t_hist > 0:
        print(f"  Speedup run_scan_hist over run:    {t_run / t_hist:.1f}x")
    if t_scan > 0 and t_hist > 0:
        print(f"  History overhead vs scan:           {t_hist / t_scan:.2f}x")

    # ---- Verify consistency ---------------------------------------------
    print("\n" + "=" * 60)
    print("CONSISTENCY CHECK")
    print("=" * 60)
    pos_match = abs(final_run["position"] - final_scan["position"]) < 1e-4
    vel_match = abs(final_run["velocity"] - final_scan["velocity"]) < 1e-4
    hist_match = abs(final_scan["position"] - final_hist["position"]) < 1e-4
    print(f"  run vs scan positions match:       {'PASS' if pos_match else 'FAIL'}")
    print(f"  run vs scan velocities match:      {'PASS' if vel_match else 'FAIL'}")
    print(f"  scan vs scan_history match:         {'PASS' if hist_match else 'FAIL'}")

    # ---- History-based analysis -----------------------------------------
    print("\n" + "=" * 60)
    print("HISTORY ANALYSIS (from run_scan_with_history)")
    print("=" * 60)

    ball_pos = history["ball"]["position"]
    ball_vel = history["ball"]["velocity"]

    print(f"  Position min:   {float(jnp.min(ball_pos)):.6f}")
    print(f"  Position max:   {float(jnp.max(ball_pos)):.6f}")
    print(f"  Position mean:  {float(jnp.mean(ball_pos)):.6f}")
    print(f"  Position std:   {float(jnp.std(ball_pos)):.6f}")
    print(f"  Velocity min:   {float(jnp.min(ball_vel)):.6f}")
    print(f"  Velocity max:   {float(jnp.max(ball_vel)):.6f}")
    print(f"  Velocity mean:  {float(jnp.mean(ball_vel)):.6f}")

    # Approximate kinetic energy over time: 0.5 * v^2 (unit mass)
    ke = 0.5 * ball_vel ** 2
    print(f"\n  Kinetic energy (mean):  {float(jnp.mean(ke)):.6f}")
    print(f"  Kinetic energy (max):   {float(jnp.max(ke)):.6f}")
    print(f"  Kinetic energy (final): {float(ke[-1]):.6f}")

    # Energy should decrease over time (elasticity < 1)
    ke_first_half = float(jnp.mean(ke[:n_steps // 2]))
    ke_second_half = float(jnp.mean(ke[n_steps // 2:]))
    energy_decreasing = ke_second_half < ke_first_half
    print(f"\n  Mean KE first half:     {ke_first_half:.6f}")
    print(f"  Mean KE second half:    {ke_second_half:.6f}")
    print(f"  Energy decreasing:      {'YES' if energy_decreasing else 'NO'}")

    # ---- sanity checks --------------------------------------------------
    assert pos_match, "Position mismatch between run() and run_scan()!"
    assert hist_match, "Position mismatch between run_scan() and run_scan_with_history()!"
    assert float(jnp.min(ball_pos)) >= -0.01, "Ball fell through the table!"

    # ---- plot -----------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        t = jnp.arange(n_steps) * dt

        axes[0, 0].plot(t, ball_pos, "b-", linewidth=0.3)
        axes[0, 0].set_ylabel("Position (m)")
        axes[0, 0].set_title("Ball Position")
        axes[0, 0].axhline(0, color="k", linewidth=0.5)

        axes[0, 1].plot(t, ball_vel, "r-", linewidth=0.3)
        axes[0, 1].set_ylabel("Velocity (m/s)")
        axes[0, 1].set_title("Ball Velocity")

        axes[1, 0].plot(t, ke, "g-", linewidth=0.3)
        axes[1, 0].set_ylabel("Kinetic Energy (J)")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_title("Kinetic Energy")

        # Performance bar chart
        methods = ["run()", "run_scan()", "scan_history()"]
        times = [t_run, t_scan, t_hist]
        colors = ["#e74c3c", "#2ecc71", "#3498db"]
        bars = axes[1, 1].bar(methods, times, color=colors)
        axes[1, 1].set_ylabel("Time (s)")
        axes[1, 1].set_title("Execution Time")
        for bar, val in zip(bars, times):
            axes[1, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{val:.3f}s", ha="center", va="bottom", fontsize=9)

        plt.suptitle(f"Scan Performance Comparison ({n_steps} steps)", fontsize=13)
        plt.tight_layout()
        out_path = os.path.join(_project_root, "scan_performance_result.png")
        plt.savefig(out_path, dpi=150)
        print(f"\nPlot saved to {out_path}")
    except ImportError:
        print("matplotlib not available, skipping plot.")

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
