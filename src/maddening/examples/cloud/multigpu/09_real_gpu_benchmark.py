#!/usr/bin/env python3
"""Benchmark multi-GPU vs single-GPU Jacobi coupling on real hardware.

Provisions a 2xA100-80GB instance on RunPod, runs coupled simulations
with varying node state sizes, measures the crossover point where
multi-GPU device placement starts to pay off.

JIT warmup is accounted for: 10 warmup steps before timing.

Usage:
    python 09_real_gpu_benchmark.py
    python 09_real_gpu_benchmark.py --keep
"""

import argparse
import json
import os
import shlex
import sys
import time

from maddening.cloud.launcher import (
    CloudLauncher,
    CostPolicy,
    JobConfig,
    LaunchError,
)


INSTALL_CMD = (
    "pip install -q --root-user-action=ignore"
    ' "jax[cuda12]>=0.4,<0.6" "numpy>=1.24" "pyyaml>=6.0"'
    " && [ -d ~/sky_workdir/src ]"
    " && pip install -q --root-user-action=ignore -e ~/sky_workdir"
    " ; echo INSTALL_DONE"
)

BENCHMARK_SCRIPT = r"""
import json, time, warnings, sys
import jax
import jax.numpy as jnp

print(f"JAX devices: {jax.devices()}")
print(f"Device count: {len(jax.devices())}")

if len(jax.devices()) < 2:
    print("ERROR: Need 2+ GPUs for benchmark")
    json.dump({"error": "need 2+ GPUs"}, open("/tmp/benchmark.json", "w"))
    sys.exit(1)

from maddening import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode

results = []

# Sweep node sizes to find the crossover point
# Small nodes (ball+spring) → overhead dominates
# Large nodes (HeatNode with many cells) → compute dominates
for n_cells in [10, 50, 100, 500, 1000, 5000]:
    print(f"\n--- n_cells={n_cells} ---")

    def build_graph():
        gm = GraphManager()
        gm.add_node(BallNode("ball", timestep=0.01, initial_position=5.0))
        gm.add_node(TableNode("table", timestep=0.01))
        gm.add_node(SpringDamperNode(
            "spring", timestep=0.01, stiffness=50.0, damping=2.0,
            mass=0.5, rest_length=1.5, initial_position=3.0,
        ))
        gm.add_node(HeatNode(
            "heat", timestep=0.01, n_cells=n_cells,
            thermal_diffusivity=0.01, initial_temperature=20.0,
        ))
        gm.add_edge("table", "ball", "position", "table_position")
        gm.add_edge("ball", "spring", "position", "anchor_position")
        gm.add_edge(
            "ball", "heat", "velocity", "left_temperature",
            transform=lambda v: jnp.clip(jnp.abs(v) * 10.0, 0.0, 100.0),
        )
        gm.add_coupling_group(
            nodes=["ball", "spring"],
            max_iterations=5, tolerance=1e-6,
            iteration_mode="jacobi",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.compile()
        return gm

    n_steps = 200
    n_warmup = 10

    # Single-GPU baseline
    gm1 = build_graph()
    for _ in range(n_warmup):
        gm1.step()
    jax.block_until_ready(gm1._state)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        gm1.step()
    jax.block_until_ready(gm1._state)
    single_time = time.perf_counter() - t0
    single_ms = single_time / n_steps * 1000

    # Multi-GPU
    gm2 = build_graph()
    gm2.enable_multigpu(n_devices=2)
    gm2.compile()
    for _ in range(n_warmup):
        gm2.step()
    jax.block_until_ready(gm2._state)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        gm2.step()
    jax.block_until_ready(gm2._state)
    multi_time = time.perf_counter() - t0
    multi_ms = multi_time / n_steps * 1000

    speedup = single_time / multi_time
    overhead_ms = multi_ms - single_ms

    print(f"  Single-GPU: {single_ms:.2f} ms/step")
    print(f"  Multi-GPU:  {multi_ms:.2f} ms/step")
    print(f"  Speedup:    {speedup:.2f}x")
    print(f"  Overhead:   {overhead_ms:.2f} ms/step")

    # Verify correctness
    pos1 = float(gm1._state["ball"]["position"])
    pos2 = float(gm2._state["ball"]["position"])
    match = abs(pos1 - pos2) < 1e-3
    print(f"  Correctness: {'PASS' if match else 'FAIL'} (diff={abs(pos1-pos2):.6f})")

    # Check device placement
    try:
        devices = set()
        for node_name in ["ball", "spring"]:
            for field, arr in gm2._state[node_name].items():
                d = list(arr.devices()) if hasattr(arr, 'devices') else ['?']
                devices.update(str(x) for x in d)
        print(f"  Devices used: {devices}")
    except Exception as e:
        print(f"  Device check: {e}")

    results.append({
        "n_cells": n_cells,
        "single_ms": round(single_ms, 2),
        "multi_ms": round(multi_ms, 2),
        "speedup": round(speedup, 2),
        "overhead_ms": round(overhead_ms, 2),
        "correct": match,
    })

# Write results
with open("/tmp/benchmark.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("CROSSOVER ANALYSIS")
print("=" * 60)
print(f"{'n_cells':>8} | {'Single':>10} | {'Multi':>10} | {'Speedup':>8} | {'Overhead':>10}")
print("-" * 60)
for r in results:
    print(f"{r['n_cells']:>8} | {r['single_ms']:>8.2f}ms | {r['multi_ms']:>8.2f}ms | {r['speedup']:>7.2f}x | {r['overhead_ms']:>8.2f}ms")
print("=" * 60)
crossover = next((r for r in results if r["speedup"] >= 1.0), None)
if crossover:
    print(f"Crossover at n_cells={crossover['n_cells']}: multi-GPU breaks even")
else:
    print("Multi-GPU never faster (overhead too high for tested sizes)")
"""


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU benchmark")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    while project_root != "/" and not os.path.exists(
        os.path.join(project_root, "pyproject.toml")
    ):
        project_root = os.path.dirname(project_root)

    # 2x cheapest GPU for multi-GPU testing
    config = JobConfig(
        provider="runpod",
        gpu_type="RTX4090",
        gpu_count=2,
        use_spot=False,
        region="US",
        cost=CostPolicy(
            max_cost_per_hour=2.0,
            max_total_budget=10.0,
            autostop_minutes=10,
            auto_teardown=False,
            spot_fallback=True,
        ),
        run="echo 'VM ready'; sleep 7200",
        workdir=project_root,
    )

    launcher = CloudLauncher()

    print("Launching 2xRTX4090 in US...")
    try:
        job = launcher.launch(config)
    except LaunchError as e:
        print(f"Launch failed: {e}")
        sys.exit(1)

    print(f"  Cluster: {job.cluster_name}")
    print(f"  VM IP: {job.vm_ip}:{job.ssh_port}")
    print(f"  Cost: ${job._hourly_cost:.2f}/hr")

    print("\nInstalling deps...")
    result = job.ssh_run(INSTALL_CMD, timeout=300, capture=True)
    print(f"  {(result.stdout or '').strip().split(chr(10))[-1]}")

    print("\nVerifying 2 GPUs visible...")
    result = job.ssh_run(
        'python3.12 -c "import jax; print(jax.devices()); assert len(jax.devices()) >= 2"',
        timeout=30, capture=True, check=False,
    )
    print(f"  {(result.stdout or '').strip()}")
    if "CudaDevice" not in (result.stdout or ""):
        print("  WARNING: GPUs may not be visible")

    print("\nRunning benchmark (this takes a few minutes)...")
    job.ssh_run(
        f"cat > /tmp/benchmark.py << 'PYEOF'\n{BENCHMARK_SCRIPT}\nPYEOF",
        check=True,
    )
    result = job.ssh_run(
        "python3.12 /tmp/benchmark.py",
        timeout=600, capture=True, check=False,
    )
    print(result.stdout or "")
    if result.stderr:
        print(f"stderr: {result.stderr[-300:]}")

    # Retrieve results
    result = job.ssh_run("cat /tmp/benchmark.json 2>/dev/null", capture=True, check=False)
    if result.stdout:
        try:
            data = json.loads(result.stdout.strip())
            print("\nResults saved.")
        except json.JSONDecodeError:
            print("Could not parse benchmark results")

    if args.keep:
        print(f"\nKeeping alive: ssh -p {job.ssh_port} root@{job.vm_ip}")
    else:
        print("\nTearing down...")
        job.teardown()
        print("  Done.")


if __name__ == "__main__":
    main()
