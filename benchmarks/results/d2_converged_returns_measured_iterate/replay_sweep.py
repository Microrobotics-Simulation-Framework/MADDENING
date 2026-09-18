#!/usr/bin/env python
"""Replay the accelerated coupling sweep and record what each row returns.

This is the D2 quantification harness: it runs the same (fixture,
configuration) grid as ``benchmarks/bench_coupling_sweep.py`` but
records *what the group returned*, not how long it took, so the same
grid can be replayed against the pre-fix and the fixed source tree and
the two compared.

Deliberately **no timings**: several agents share this machine and a
wall clock here measures the neighbours.

Usage
-----
Run the same row range against both trees, then diff them::

    PYTHONPATH=<before>/src JAX_PLATFORMS=cpu python replay_sweep.py \\
        --out before.jsonl --start 0 --limit 50
    PYTHONPATH=<wt>/src     JAX_PLATFORMS=cpu python replay_sweep.py \\
        --out after.jsonl  --start 0 --limit 50
    python diff_sweep.py before.jsonl after.jsonl --out comparison.json

The output is JSON Lines, one row per (fixture, configuration), flushed
after every row: an interruption costs the row in flight and nothing
else.  Re-running with the same ``--out`` skips rows already present, so
the recipe above is restartable.

Row order is a *staircase* over (fixture, configuration) -- rows are
sorted by ``max(fixture_rank, config_rank)`` -- so any prefix of the run
is a near-complete rectangle of the grid rather than a few fixtures with
every configuration.  The first 50 rows cover seven fixtures crossed
with seven configurations; a truncated run is still a usable answer.

What a row carries
------------------
Per coupling group, per step: iterations used, the reported residual and
the converged flag.  Per step: a hash of the concatenated float32 state
of the coupled nodes (bit-identity across trees is checkable), and its
L2 norm.  Plus the full state after step 1 and after the last step, so
the relative shift can be computed exactly rather than from norms.

``warmup`` is 0 for every fixture, which is a deliberate departure from
the benchmark driver.  Both trees start from the same deterministic
initial state, so the state after step 1 is the *uncompounded* shift
from a single converged exit -- the number the brief asks for -- while
the state after the last step is the same shift compounded through the
window.  A warmup would have mixed the two.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

_BENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BENCH))

import numpy as np  # noqa: E402

from coupling_fixtures import (  # noqa: E402
    FIXTURES,
    CouplingConfig,
    ScopeNotApplicable,
    sweep_configs,
)

#: Steps per row, capped.  ``bench_coupling_sweep`` caps its statistics
#: window the same way and for the same reason.
_STEP_CAP = 50

#: Fixtures in the order they are first reached, chosen so the first six
#: rows of the staircase touch every family once.
_FIXTURE_ORDER = [
    "stiff-pair-0.5", "chain-5", "ring-8", "star-4", "mixed-modes",
    "slow-drift", "stiff-pair-1.2", "chain-2", "star-2", "ring-4",
    "stiff-pair-0.25", "stiff-pair-0.8", "stiff-pair-0.95", "chain-20",
    "star-8", "ring-16", "chain-50", "star-16",
]


def _staircase(pairs_a, pairs_b):
    """Order the product so every prefix is a near-complete rectangle."""
    out = [(a, b) for a in range(len(pairs_a)) for b in range(len(pairs_b))]
    out.sort(key=lambda p: (max(p), p))
    return out


def _configs(include_none: bool) -> list[CouplingConfig]:
    """The swept configurations, ordered to span accelerations early."""
    all_cfgs = sweep_configs(("l2", "interface"))
    picked = [c for c in all_cfgs
              if include_none or c.acceleration != "none"]
    # Two independent axes: the accelerator, and the (mode, norm) pair.
    accels, modenorms = [], []
    for c in picked:
        a = (c.acceleration, c.relaxation, c.jacobian_reuse)
        mn = (c.iteration_mode, c.convergence_norm)
        if a not in accels:
            accels.append(a)
        if mn not in modenorms:
            modenorms.append(mn)
    by_key = {((c.acceleration, c.relaxation, c.jacobian_reuse),
               (c.iteration_mode, c.convergence_norm)): c for c in picked}
    return [by_key[(accels[i], modenorms[j])]
            for i, j in _staircase(accels, modenorms)]


def build_rows(include_none: bool = False) -> list[tuple[str, CouplingConfig]]:
    """The full row list, in run order, identical on both trees."""
    fixtures = [n for n in _FIXTURE_ORDER if n in FIXTURES]
    missing = sorted(set(FIXTURES) - set(fixtures)
                     - {n for n, f in FIXTURES.items() if f.slow})
    fixtures += sorted(missing)          # never silently drop a fixture
    configs = _configs(include_none)
    rows = []
    for i, j in _staircase(fixtures, configs):
        name, cfg = fixtures[i], configs[j]
        if FIXTURES[name].mode_fixed and cfg.iteration_mode != "gauss-seidel":
            continue                      # would only duplicate the row
        rows.append((name, cfg))
    return rows


def _flat_state(gm, nodes) -> tuple[list[str], list[int], np.ndarray]:
    """Concatenated float32 state of *nodes*, with its field layout."""
    names, sizes, chunks = [], [], []
    for node in sorted(gm.node_names):
        if node not in nodes:
            continue
        state = gm.get_node_state(node)
        for field in sorted(state):
            arr = np.asarray(state[field], dtype=np.float32).ravel()
            if arr.size == 0:
                continue
            names.append(f"{node}.{field}")
            sizes.append(int(arr.size))
            chunks.append(arr)
    flat = (np.concatenate(chunks) if chunks
            else np.zeros(0, dtype=np.float32))
    return names, sizes, flat


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def _coupled_nodes(built) -> frozenset:
    out: set[str] = set()
    for key in built.group_keys:
        out.update(key.split("+"))
    return frozenset(out)


def measure(name: str, cfg: CouplingConfig, index: int) -> dict:
    """Run one row and return its record."""
    spec = FIXTURES[name]
    row: dict = {
        "row": index,
        "fixture": name,
        "label": cfg.label,
        "iteration_mode": cfg.iteration_mode,
        "acceleration": cfg.acceleration,
        "relaxation": cfg.relaxation,
        "convergence_norm": cfg.convergence_norm,
        "jacobian_reuse": cfg.jacobian_reuse,
    }
    try:
        built = spec.build(cfg)
    except ScopeNotApplicable as exc:
        row.update(ok=False, skipped=True, error=str(exc))
        return row
    except Exception as exc:                       # a build failure is a result
        row.update(ok=False, skipped=False,
                   error=f"{type(exc).__name__}: {exc}")
        return row

    gm = built.gm
    ext = built.external_inputs
    coupled = _coupled_nodes(built)
    n_steps = min(spec.steps, _STEP_CAP)

    per_group: dict[str, dict[str, list]] = {}
    hashes: list[str] = []
    l2s: list[float] = []
    first_state = None
    names = sizes = None

    try:
        for step in range(n_steps):
            gm.step(ext) if ext is not None else gm.step()
            for key, d in gm.coupling_diagnostics().items():
                g = per_group.setdefault(
                    key, {"iterations": [], "residual": [], "converged": []})
                g["iterations"].append(int(d["iterations"]))
                g["residual"].append(float(d["residual"]))
                g["converged"].append(bool(d["converged"]))
            names, sizes, flat = _flat_state(gm, coupled)
            hashes.append(hashlib.blake2b(
                np.ascontiguousarray(flat).tobytes(), digest_size=12).hexdigest())
            l2s.append(float(np.sqrt(np.sum(
                np.asarray(flat, dtype=np.float64) ** 2))))
            if step == 0:
                first_state = flat.copy()
    except Exception as exc:
        row.update(ok=False, skipped=False,
                   error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc()[-2000:],
                   steps_done=len(hashes))
        return row

    row.update({
        "ok": True,
        "skipped": False,
        "error": None,
        "n_steps": n_steps,
        # Per group: ``at_cap_fraction`` is ``iters >= cap - 1`` (the
        # profiler's rule -- ``iterations`` counts body passes after the
        # first), and ``mixed-modes`` gives its two groups different
        # caps, so one number for the row would be wrong.
        "caps": {"+".join(sorted(g.nodes)): int(g.max_iterations)
                 for g in gm._coupling_groups},
        "group_keys": sorted(per_group),
        "groups": per_group,
        "state_fields": names,
        "state_sizes": sizes,
        "step_hashes": hashes,
        "step_l2": l2s,
        "step1_state": _b64(first_state),
        "final_state": _b64(flat),
    })
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0,
                    help="0 means 'to the end'")
    ap.add_argument("--include-none", action="store_true",
                    help="also sweep acceleration='none' (the brief's grid "
                         "is the accelerated rows only)")
    ap.add_argument("--list", action="store_true",
                    help="print the row order and exit")
    args = ap.parse_args()

    rows = build_rows(args.include_none)
    if args.list:
        for i, (name, cfg) in enumerate(rows):
            print(f"{i:4d}  {name:<16s} {cfg.label}")
        print(f"total {len(rows)} rows")
        return 0

    done: set[int] = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["row"])

    stop = len(rows) if not args.limit else min(len(rows),
                                                args.start + args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as fh:
        for i in range(args.start, stop):
            if i in done:
                continue
            name, cfg = rows[i]
            rec = measure(name, cfg, i)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass                       # e.g. --out /dev/null
            flag = "ok" if rec.get("ok") else f"FAIL {rec.get('error')}"
            print(f"{i:4d}/{len(rows)}  {name:<16s} {cfg.label:<28s} {flag}",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
