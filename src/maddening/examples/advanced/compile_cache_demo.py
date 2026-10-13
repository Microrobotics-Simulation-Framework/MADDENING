#!/usr/bin/env python3
"""The persistent compile cache: compile a graph once, reuse it in the next process.

A graph's step compiles once per process.  :mod:`maddening.core.simulation.
compile_cache` keeps the compiled programs on disk, so the next process that
builds the same graph reads them back instead of compiling them again:

1. **A cold run.**  A fresh process calls ``compile_cache.enable(dir)`` on an
   empty temporary directory, then ``compile_cache.warm_cache(build,
   n_steps=2, scan_steps=N)``, which builds the graph and compiles its
   ``step`` and its ``run_scan(N)``.  The directory then holds the compiled
   programs.
2. **A warm run.**  A second fresh process does the same with the same
   directory.  It reads the programs back (JAX reports cache hits) and its
   compile time is a fraction of the first.

Both processes run with ``MADDENING_COMPILATION_CACHE_DIR`` pointing at a
second temporary directory, as if it were set in your shell.  The explicit
``enable(dir)`` wins over the variable: every entry lands in ``dir``, and
the variable's directory stays empty.  (``GraphManager.compile()`` enables
the variable's directory only while no directory is enabled yet.)

The timings are printed, not asserted: they depend on the machine.  What
the example asserts is what does not: the cold run finds nothing in the
empty directory, the directory holds entries after it, the warm run reads
them back and adds none, both processes report the directory they were
given, and the variable's directory holds nothing.

Everything goes into temporary directories, removed on exit; nothing is
written to your home directory.  To use a cache in your own runs, set
``MADDENING_COMPILATION_CACHE_DIR`` (``GraphManager.compile()`` picks it
up) or call ``compile_cache.enable(dir)``.

Usage
-----
    python -m maddening.examples.advanced.compile_cache_demo
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

N_CELLS = 64
SCAN_STEPS = 100


def build():
    """Two heat rods meeting end to end, solved in a coupling group."""
    from maddening.core.graph_manager import GraphManager
    from maddening.nodes.heat import HeatNode

    dx, alpha = 1e-3, 1e-4                    # [m], [m^2/s]
    dt = 0.2 * dx ** 2 / alpha                # Fourier number 0.2
    length = N_CELLS * dx
    gm = GraphManager()
    gm.add_node(HeatNode("cold", dt, n_cells=N_CELLS, length=length,
                         thermal_diffusivity=alpha, initial_temperature=290.0))
    gm.add_node(HeatNode("hot", dt, n_cells=N_CELLS, length=length,
                         thermal_diffusivity=alpha, initial_temperature=350.0))
    gm.add_edge("cold", "hot", "temperature", "left_temperature",
                transform="extract_last")
    gm.add_edge("hot", "cold", "temperature", "right_temperature",
                transform="extract_first")
    gm.add_coupling_group(["cold", "hot"], max_iterations=20, tolerance=1e-6)
    return gm


def one_run(cache_dir: str) -> dict:
    """Enable the cache, warm it, and report timings, hits and directories."""
    import jax

    from maddening.core.simulation import compile_cache

    asked = compile_cache.enable(cache_dir)   # wins over the variable
    hits = []
    jax.monitoring.register_event_listener(
        lambda event, **_: hits.append(event)
        if event == "/jax/compilation_cache/cache_hits" else None)
    timings = compile_cache.warm_cache(build, n_steps=2, scan_steps=SCAN_STEPS)
    timings["cache_hits"] = len(hits)
    timings["asked"] = asked
    timings["enabled_dir"] = compile_cache.enabled_dir()
    return timings


def entries(cache_dir: str) -> list[str]:
    return sorted(p.name for p in Path(cache_dir).rglob("*") if p.is_file())


def run_in_fresh_process(cache_dir: str, variable_dir: str) -> dict:
    env = {**os.environ, "MADDENING_COMPILATION_CACHE_DIR": variable_dir}
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        [sys.executable, "-m", __spec__.name if __spec__ else "compile_cache_demo",
         "--child", cache_dir],
        capture_output=True, text=True, check=True, env=env)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--child", metavar="DIR", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        print(json.dumps(one_run(args.child)))
        return 0

    root = tempfile.mkdtemp(prefix="maddening-xla-")
    try:
        cache_dir = os.path.join(root, "asked")
        variable_dir = os.path.join(root, "variable")
        os.makedirs(variable_dir)
        print(f"Cache directory, given to enable():  {cache_dir}")
        print(f"MADDENING_COMPILATION_CACHE_DIR:     {variable_dir}")
        cold = run_in_fresh_process(cache_dir, variable_dir)
        stored = entries(cache_dir)
        warm = run_in_fresh_process(cache_dir, variable_dir)
        print()
        print(f"  {'':24s} {'cold':>10s} {'warm':>10s}")
        print(f"  {'step compile + 1 step':24s} {cold['compile_s']:9.2f}s {warm['compile_s']:9.2f}s")
        print(f"  {'run_scan compile + run':24s} {cold['scan_compile_s']:9.2f}s {warm['scan_compile_s']:9.2f}s")
        print(f"  {'cache hits':24s} {cold['cache_hits']:10d} {warm['cache_hits']:10d}")
        print(f"  entries in the cache after the cold run: {len(stored)}")
        print(f"  entries in the variable's directory:     {len(entries(variable_dir))}")
        for run in (cold, warm):
            assert run["asked"] == run["cache_dir"] == run["enabled_dir"] == cache_dir, run
        assert cold["cache_hits"] == 0, cold
        assert stored, "the cold run wrote nothing to the directory it was given"
        assert warm["cache_hits"] > 0, warm
        assert entries(cache_dir) == stored, "the warm run compiled something new"
        assert not entries(variable_dir), "an entry went to the variable's directory"
        print("\n  The warm process read the compiled programs back instead of "
              "compiling them,\n  from the directory it was given, not the "
              "variable's.")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print()
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
