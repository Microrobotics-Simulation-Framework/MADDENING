"""Does a compiled GraphManager step pay the unstructured wrapper's host cost?  (2026-10-12)

The session's `forward` goal measured the public `ShardedUnstructuredNode.update()`
at about 250 ms/step at 1e7 cells against 3.4 ms on the device.  The suspected
cause: the runner's `NeighbourMeanNode.static_data` builds a new array on every
access, so the wrapper's identity-keyed static cache misses and re-partitions
the statics on every eager call.  A compiled graph should pay that once, at trace.

Three variants, same node, same mesh, same partition:
  A  the runner's node, stepped eagerly through the public update();
  B  the same with static_data cached (the same objects on every access), eagerly;
  C  node A inside a GraphManager: gm.step(), and gm.run_scan().
For each: ms per step, how many times the wrapper re-partitioned a static
(calls to partition_value in sharded_unstructured), and the final field.

Usage (from the checkout's root):
  python3 graph_vs_eager.py --cells 10000000 --steps 20
Runs on CPU virtual devices too (JAX_PLATFORMS=cpu,
XLA_FLAGS=--xla_force_host_platform_device_count=4).  Gates nothing.
"""
import argparse
import json
import os
import statistics
import sys
import time
import types

sys.path.insert(0, os.path.join(os.getcwd(), "benchmarks", "multigpu"))
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import run_pod as rp  # noqa: E402

import maddening.cloud.multigpu.sharded_unstructured as su  # noqa: E402
from maddening.core.graph_manager import GraphManager  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--cells", type=int, default=10_000_000)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--devices", type=int, default=4)
ap.add_argument("--out", default="")
ap.add_argument("--profile-dir", default="", help="also run profile_graph (trace=True) on 4 cards and on 1")
ap.add_argument("--profile-steps", type=int, default=50)
args = ap.parse_args()

calls = {"n": 0}
_partition_value = su.partition_value


def counting_partition_value(*a, **k):
    calls["n"] += 1
    return _partition_value(*a, **k)


su.partition_value = counting_partition_value

rp._load_backend()
D = args.devices
mesh = rp._mesh_for(D)
t0 = time.perf_counter()
ns = types.SimpleNamespace(mesh=None, synthetic="grid", partition="contiguous")
n, edges, pa, source = rp._mesh_source(ns, args.cells)
pa, how = rp._partition_for(ns, source, n, edges, pa, D)
edges, pa, _ = rp.cells_in_an_order_the_wrapper_takes(n, edges, pa, D)
setup_s = time.perf_counter() - t0
print(f"devices {jax.devices()}  cells {n}  partition {how}  host setup {setup_s:.1f} s", flush=True)

Node = rp.NeighbourMeanNode


class NoSource(Node):
    """The runner's node without its per-cell source (the graph has no input to feed it)
    and without its diagnostic integrals (``total``, ``per_shard``: not in its initial
    state, which GraphManager refuses as a change of layout).  The field is the same."""

    def boundary_input_spec(self):
        return {}

    def domain_integral_fields(self):
        return set()

    def domain_integral_axes(self):
        return {}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": super().update(state, boundary_inputs, dt, params=params)["x"]}

    def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                      shard_info=None, params=None):
        return {"x": super().update_padded(state_padded, boundary_inputs, dt,
                                           static_padded=static_padded, shard_info=shard_info,
                                           params=params)["x"]}


class CachedStatics(NoSource):
    """Variant B: static_data returns the same objects on every access."""

    @property
    def static_data(self):
        if not hasattr(self, "_static_cache"):
            self._static_cache = NoSource.static_data.fget(self)
        return self._static_cache


def build(cls):
    layout = su.build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=D) \
        if hasattr(su, "build_unstructured_partition") else \
        rp.build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=D)
    tbl = rp.neighbour_table(n, edges)
    inner = cls("mesh", n, rp.slab_table(layout, tbl), partition_assignment=pa,
                weight=rp.NODE_WEIGHT)
    return rp.ShardedUnstructuredNode(inner, mesh, layout, exchange="ppermute"), layout


params = {"weight": jnp.asarray(rp.NODE_WEIGHT, jnp.float32)}
results = {"cells": n, "steps": args.steps, "devices": [str(d) for d in jax.devices()]}


def eager(label, cls):
    sharded, _ = build(cls)
    state = {"x": sharded.initial_state()["x"]}
    jax.block_until_ready(sharded.update(state, {}, 1.0, params=params)["x"])   # compile
    before = calls["n"]
    times = []
    for _ in range(args.steps):
        t = time.perf_counter()
        state = {"x": sharded.update(state, {}, 1.0, params=params)["x"]}
        jax.block_until_ready(state["x"])
        times.append((time.perf_counter() - t) * 1e3)
    r = {"median_ms_per_step": statistics.median(times),
         "repartitions_during_steps": calls["n"] - before}
    print(f"[{label}] eager update: {r['median_ms_per_step']:.2f} ms/step, "
          f"re-partitions in {args.steps} steps: {r['repartitions_during_steps']}", flush=True)
    return r, np.asarray(sharded.gather_global(state)["x"])


results["A_eager_fresh_statics"], xA = eager("A", NoSource)
results["B_eager_cached_statics"], xB = eager("B", CachedStatics)

sharded, _ = build(NoSource)
gm = GraphManager()
gm.add_node(sharded)
before_compile = calls["n"]
t = time.perf_counter()
gm.compile()
st = gm.step()                # the first step traces and compiles
jax.block_until_ready(st["mesh"]["x"])
compile_s = time.perf_counter() - t
before = calls["n"]
times = []
for _ in range(args.steps - 1):
    t = time.perf_counter()
    st = gm.step()
    jax.block_until_ready(st["mesh"]["x"])
    times.append((time.perf_counter() - t) * 1e3)
xC = np.asarray(sharded.gather_global({"x": st["mesh"]["x"]})["x"])
results["C_graph_step"] = {"median_ms_per_step": statistics.median(times),
                           "repartitions_at_compile_and_first_step": before - before_compile,
                           "repartitions_during_steps": calls["n"] - before,
                           "compile_and_first_step_s": compile_s}
print(f"[C] gm.step(): {statistics.median(times):.2f} ms/step, re-partitions: "
      f"{before - before_compile} at compile+first step, {calls['n'] - before} during "
      f"{args.steps - 1} steps (compile+first step {compile_s:.1f} s)", flush=True)

gm2 = GraphManager()
sharded2, _ = build(NoSource)
gm2.add_node(sharded2)
gm2.compile()
jax.block_until_ready(gm2.run_scan(args.steps)["mesh"]["x"])   # compile the scan (one per step count)
gm2.reset_state()
before = calls["n"]
t = time.perf_counter()
out = gm2.run_scan(args.steps)
jax.block_until_ready(out["mesh"]["x"])
scan_ms = (time.perf_counter() - t) * 1e3 / args.steps
results["C_graph_run_scan"] = {"ms_per_step": scan_ms, "repartitions": calls["n"] - before}
print(f"[C] gm.run_scan({args.steps}): {scan_ms:.2f} ms/step, re-partitions: {calls['n'] - before}",
      flush=True)

results["A_vs_B_max_abs"] = float(np.max(np.abs(xA - xB)))
results["A_vs_graph_step_max_abs"] = float(np.max(np.abs(xA - xC)))
print(f"final field: A vs B max|dx| {results['A_vs_B_max_abs']:.3e}; "
      f"A vs graph step max|dx| {results['A_vs_graph_step_max_abs']:.3e}", flush=True)

if args.profile_dir:
    from maddening.core.simulation.profiler import profile_graph
    os.makedirs(args.profile_dir, exist_ok=True)
    tbl = rp.neighbour_table(n, edges)
    graphs = {}
    g4 = GraphManager(); g4.add_node(build(CachedStatics)[0]); g4.compile(); graphs["4_cards_sharded"] = g4
    one = NoSource("mesh", n, tbl, weight=rp.NODE_WEIGHT)   # the unsharded node: one device
    g1 = GraphManager(); g1.add_node(one); g1.compile(); graphs["1_card_unsharded"] = g1
    for label, g in graphs.items():
        d = os.path.join(args.profile_dir, label)
        rep = profile_graph(g, n_steps=args.profile_steps, n_warmup=3, measure_coupling=False,
                            counts=True, trace=True, trace_steps=20, trace_dir=d)
        with open(os.path.join(args.profile_dir, f"{label}.txt"), "w") as f:
            f.write(str(rep))
        keep = {k: getattr(rep, k) for k in ("jit_compile_ms", "mean_step_ms", "median_step_ms",
                "std_step_ms", "steps_per_second", "dispatch_floor_ms", "device", "node_times_ms")
                if hasattr(rep, k)}
        tr = getattr(rep, "trace", None)
        if tr is not None:
            keep["trace"] = tr if isinstance(tr, dict) else getattr(tr, "__dict__", str(tr))
        results[f"profile_{label}"] = keep
        print(f"[profile {label}] median {keep.get('median_step_ms', float('nan')):.3f} ms/step, "
              f"compile {keep.get('jit_compile_ms', float('nan')):.0f} ms, "
              f"dispatch floor {keep.get('dispatch_floor_ms', float('nan')):.3f} ms", flush=True)
if args.out:
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("wrote", args.out)
