#!/usr/bin/env python
"""A sparse interface mapping inside a coupling group, saved and reloaded.

``StaticSparseMapping`` (EXPERIMENTAL in 0.4.0, from
``maddening.core.coupling.sparse_mapping``) stores only the entries each
row of an interface mapping uses: an integer index and one weight per
entry.  A ``StaticLinearMapping`` stores the whole ``n_target x
n_source`` matrix.  This example measures where that matters, couples two
large rods through sparse mappings in a coupling group, and saves and
reloads the graph.

What it shows, and checks:

1. **When a dense matrix is too large.**  The dense
   ``nearest_neighbor_mapping`` is built at a size it can manage, and its
   memory is measured with ``tracemalloc``: ``H`` holds 4 bytes for each
   (target, source) pair, and building it holds about 16 bytes a pair at
   its peak, for its table of distances.  Both grow with the product of
   the two sizes.  ``sparse_nearest_neighbor_mapping`` builds the same
   matrix (checked bit for bit) from 8 bytes per target.  For the
   interface below, 200 000 cells against 100 000, the dense ``H`` would
   be 80 GB for each edge; the sparse mappings hold 2.4 MB between them.
2. **Sparse edges in a coupling group.**  A fine rod (200 000 cells) and
   a coarse one (100 000) on one interval, each heating the other in
   proportion to its temperature, one sparse mapping each way, in one
   coupling group with ``solver``, ``acceleration``, ``convergence_norm``,
   ``rtol`` and ``iteration_mode`` set.  It is not a physical model: it is
   the smallest graph in which a group iterates over an interface of this
   size.  The mapping weights are parameters (``W`` in
   ``gm.params["mappings"]``); one edge's are scaled to 0.5 by a write.
3. **Save and reload.**  ``gm.to_dict()``, taken when the graph is
   built, carries the group's options and each mapping's recipe (kind,
   references to the rods' ``grid_x``), never its index or weights;
   ``gm.save_state()``, taken after the run, carries the state and the
   weights, with a digest of the index they were made for.
   ``GraphManager.from_dict`` rebuilds the group and calls the same
   builder on the same points, so the index is the same, bit for bit;
   ``load_state`` restores the state and the scaled weights.  The reloaded
   graph then steps to the same state as the original, bit for bit, and
   reports the same iterations.  Without the checkpoint the reloaded graph
   would hold the builder's weights, not the scaled ones: a config does
   not carry a weight write (``to_dict()`` warns when it is taken after
   one).

Usage
-----
    python -m maddening.examples.coupling.sparse_mapping_round_trip_demo
    python -m maddening.examples.coupling.sparse_mapping_round_trip_demo --cells 20000 --steps 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import tracemalloc
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import nearest_neighbor_mapping
from maddening.core.coupling.sparse_mapping import (
    StaticSparseMapping, sparse_nearest_neighbor_mapping,
)
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode

DT = 1e-3
DIFFUSIVITY = 100.0        # cell width 1, so the Fourier number is 0.1
GROUP = "coarse+fine"
TO_FINE = "coarse.temperature->fine.heat_source"
TO_COARSE = "fine.temperature->coarse.heat_source"
REGISTRY = {"HeatNode": HeatNode}
#: The group's options: what the round trip has to carry.
GROUP_OPTIONS = dict(max_iterations=20, solver="ift", acceleration="aitken",
                     convergence_norm="mixed", rtol=1e-5,
                     iteration_mode="gauss-seidel")
#: The dense mapping is measured at this many targets (and half as many sources).
DENSE_PROBE = 3000


def section(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def measured(build):
    """``build()``'s result and the peak of what it allocated, in bytes.

    ``tracemalloc`` sees NumPy's array buffers as well as Python objects,
    which is where the factories below build their matrices.
    """
    tracemalloc.start()
    try:
        result = build()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return result, peak


def structure_bytes(mapping: StaticSparseMapping) -> int:
    """What a sparse mapping holds: its index and its weights."""
    return mapping.indices.nbytes + np.asarray(mapping.params_pytree()["W"]).nbytes


def physical_memory():
    """This machine's memory in bytes, where the OS says (not on Windows)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def build(n_fine: int) -> GraphManager:
    """The two rods, a sparse mapping each way, and the coupling group."""
    n_coarse = n_fine // 2
    length = float(n_fine)
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", DT, n_cells=n_coarse, length=length,
                         thermal_diffusivity=DIFFUSIVITY))
    gm.add_node(HeatNode("fine", DT, n_cells=n_fine, length=length,
                         thermal_diffusivity=DIFFUSIVITY))
    x = {name: np.asarray(gm.get_node(name).static_data["grid_x"].value)
         for name in ("coarse", "fine")}
    for src, tgt in (("coarse", "fine"), ("fine", "coarse")):
        # Consistent nearest neighbour: each target cell takes the value of
        # the source cell nearest to it.  The references are what a saved
        # config records in place of the arrays.
        gm.add_edge(src, tgt, "temperature", "heat_source",
                    mapping=sparse_nearest_neighbor_mapping(
                        x[src], x[tgt],
                        source_ref={"node": src, "field": "grid_x"},
                        target_ref={"node": tgt, "field": "grid_x"}))
    gm.add_coupling_group(["coarse", "fine"], **GROUP_OPTIONS)
    gm.compile()
    return gm


def states(gm: GraphManager) -> dict:
    return {n: {f: np.asarray(v) for f, v in gm.get_node_state(n).items()}
            for n in gm.node_names}


def part_1_memory(n_fine: int) -> None:
    section("1. When a dense interface matrix is too large")
    n_t, n_s = DENSE_PROBE, DENSE_PROBE // 2
    probe_t, probe_s = np.linspace(0.0, 1.0, n_t), np.linspace(0.0, 1.0, n_s)
    sparse_nearest_neighbor_mapping(probe_s[:8], probe_t[:8])   # import scipy first
    dense, dense_peak = measured(lambda: nearest_neighbor_mapping(probe_s, probe_t))
    sparse = sparse_nearest_neighbor_mapping(probe_s, probe_t)
    h_bytes = np.asarray(dense.H).nbytes
    per_pair = dense_peak / (n_t * n_s)
    print(f"  dense nearest_neighbor_mapping, {n_t} targets x {n_s} sources:")
    print(f"    H holds {h_bytes / 1e6:.1f} MB (4 bytes a pair); building it "
          f"peaked at {dense_peak / 1e6:.1f} MB ({per_pair:.1f} bytes a pair)")
    assert h_bytes == 4 * n_t * n_s and dense_peak >= h_bytes

    # The sparse rows, scattered into a zero matrix, are the dense H.
    rows = np.zeros((n_t, n_s), np.float32)
    weights = np.asarray(sparse.params_pytree()["W"])
    for j in range(sparse.indices.shape[1]):
        rows[np.arange(n_t), sparse.indices[:, j]] += weights[:, j]
    same = np.array_equal(rows, np.asarray(dense.H))
    print(f"  sparse_nearest_neighbor_mapping of the same points: index "
          f"{sparse.indices.shape}, {structure_bytes(sparse) / 1e3:.0f} kB; "
          f"the same matrix bit for bit: {same}")
    assert same

    n_coarse = n_fine // 2
    x_fine = np.linspace(0.5, n_fine - 0.5, n_fine)
    x_coarse = np.linspace(1.0, n_fine - 1.0, n_coarse)
    to_fine, peak = measured(lambda: sparse_nearest_neighbor_mapping(x_coarse, x_fine))
    to_coarse = sparse_nearest_neighbor_mapping(x_fine, x_coarse)
    held = structure_bytes(to_fine) + structure_bytes(to_coarse)
    dense_h = 4 * n_fine * n_coarse
    print(f"  the interface of part 2, {n_fine} fine cells and {n_coarse} coarse, "
          f"one mapping each way:")
    print(f"    dense: H would hold {dense_h / 1e9:.2f} GB per edge, and building it "
          f"about {per_pair * n_fine * n_coarse / 1e9:.1f} GB at its peak")
    print(f"    sparse: {held / 1e6:.2f} MB for both; building the larger peaked at "
          f"{peak / 1e6:.1f} MB")
    memory = physical_memory()
    if memory is not None:
        print(f"    (this machine has {memory / 1e9:.0f} GB of memory)")
    assert held == 8 * (n_fine + n_coarse) and dense_h > 1000 * held


def part_2_group(n_fine: int, n_steps: int):
    section("2. Sparse edges in a coupling group")
    gm = build(n_fine)
    # The graph's structure, saved once, as built: the group's options and
    # each mapping's recipe.  (Taken after the weight write below, to_dict()
    # would warn that the config does not carry the written weights, and
    # name save_state as the way to keep them -- as part 3 does.)
    config = json.loads(json.dumps(gm.to_dict()))        # through real JSON
    for key in (TO_FINE, TO_COARSE):
        edge = next(e for e in gm.edges
                    if f"{e.source_node}.{e.source_field}->{e.target_node}.{e.target_field}" == key)
        W = gm.params["mappings"][key]["W"]
        print(f"  {key}: {type(edge.mapping).__name__}, kind "
              f"{edge.mapping.describe()['kind']!r}, W {tuple(W.shape)}")
        assert isinstance(edge.mapping, StaticSparseMapping) and W.shape == (edge.mapping.indices.shape)

    # The weights are a parameter: scale one edge's, no recompile needed.
    gm.params["mappings"][TO_COARSE]["W"] = 0.5 * gm.params["mappings"][TO_COARSE]["W"]
    # A warm bump in the middle of the fine rod; the coarse one starts at 0.
    x_fine = np.asarray(gm.get_node("fine").static_data["grid_x"].value)
    bump = np.exp(-((x_fine - n_fine / 2) / (n_fine / 20)) ** 2)
    gm.set_node_state("fine", {**gm.get_node_state("fine"),
                               "temperature": jnp.asarray(bump, jnp.float32)})

    gm.run(n_steps)
    info = gm.coupling_diagnostics()[GROUP]
    peak_coarse = float(np.max(gm.get_node_state("coarse")["temperature"]))
    print(f"  {n_steps} steps: the last converged={bool(info['converged'])} in "
          f"{info['iterations']} passes (cap {GROUP_OPTIONS['max_iterations']}); "
          f"the coarse rod has warmed to {peak_coarse:.2e}")
    assert info["converged"] and peak_coarse > 0.0
    return gm, config


def part_3_round_trip(gm: GraphManager, config: dict, n_steps: int, tmp: Path) -> None:
    section("3. Save the state, reload the graph and the state, step both")
    checkpoint = gm.save_state(tmp / "state.npz")
    (group,) = config["coupling_groups"]
    print(f"  config: {len(json.dumps(config))} bytes; checkpoint: "
          f"{checkpoint.stat().st_size / 1e6:.1f} MB")
    print("  the group in the config: "
          + ", ".join(f"{k}={group[k]!r}" for k in GROUP_OPTIONS))
    assert all(group[k] == v for k, v in GROUP_OPTIONS.items())
    for edge in config["edges"]:
        spec = edge["mapping"]
        print(f"  {edge['source_node']}->{edge['target_node']}: kind={spec['kind']!r}, points "
              + ", ".join(f"{k}={v['node']}.{v['field']}" for k, v in spec["points"].items()))
        assert "W" not in spec and "indices" not in spec, "a recipe, not the arrays"

    fresh = GraphManager.from_dict(config, REGISTRY)
    fresh.compile()
    assert fresh.to_dict()["coupling_groups"] == config["coupling_groups"]
    for old, new in zip(gm.edges, fresh.edges):
        assert new.mapping.structure_digest() == old.mapping.structure_digest()
    print("  rebuilt: the same group options, and the same index on each edge "
          "(structure_digest)")

    builder_w = np.asarray(fresh.params["mappings"][TO_COARSE]["W"])
    fresh.load_state(checkpoint)
    loaded_w = np.asarray(fresh.params["mappings"][TO_COARSE]["W"])
    print(f"  {TO_COARSE} weights: {builder_w.max():.1f} from the builder, "
          f"{loaded_w.max():.1f} after load_state")
    assert builder_w.max() == 1.0 and np.array_equal(
        loaded_w, np.asarray(gm.params["mappings"][TO_COARSE]["W"]))

    gm.run(n_steps)
    fresh.run(n_steps)
    a, b = states(gm), states(fresh)
    identical = all(np.array_equal(a[n][f], b[n][f]) for n in a for f in a[n])
    passes = (gm.coupling_diagnostics()[GROUP]["iterations"],
              fresh.coupling_diagnostics()[GROUP]["iterations"])
    print(f"  {n_steps} more steps on each: states identical: {identical}; "
          f"passes on the last step {passes[0]} and {passes[1]}")
    assert identical and passes[0] == passes[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells", type=int, default=200_000,
                        help="Cells of the fine rod; the coarse one has half (default 200000)")
    parser.add_argument("--steps", type=int, default=20,
                        help="Steps before and after the round trip (default 20)")
    args = parser.parse_args(argv)
    if args.cells < 1000 or args.cells % 2:
        parser.error("--cells must be even and at least 1000")

    part_1_memory(args.cells)
    gm, config = part_2_group(args.cells, args.steps)
    with tempfile.TemporaryDirectory(prefix="maddening_sparse_demo_") as tmp:
        part_3_round_trip(gm, config, args.steps, Path(tmp))
    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
