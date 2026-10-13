#!/usr/bin/env python
"""Does tightening the fixtures' ``rtol`` retire ``_KNOWN_DISAGREEMENTS``?

Replays the lane that owns the ten entries --
``test_coupling_fixture_invariants.py::test_every_configuration_reaches_
the_same_fixed_point``, all 24 sweep configurations of a fixture over 25
driven steps -- once per ``rtol``, and records what each configuration's
trajectory costs and how far it ends up from the reference.

The reference is the same one the test uses: the first configuration
``sweep_configs(("l2", "interface"))`` yields, ``gs/none/l2``.  Both
norms therefore have to be run even when only the ``interface`` rows are
of interest; running ``--norms interface`` alone silently changes the
reference to ``gs/none/interface`` and every number with it.

``atol`` is settable and was measured, but it is inert on these
fixtures: it is the dead band ``_scaled_change`` uses to decide which
elements are in the norm at all, and every interface element of every
spring fixture is orders of magnitude above 1e-08.  1e-12 and 1e-08
produce identical trajectories.

Deliberately **no timings**: several agents share this machine and a
wall clock here measures the neighbours.  What is recorded instead is
iterations per step and the converged fraction, which is what the cost
of a tighter criterion is actually paid in.

Usage
-----
One run per ``rtol``, then read them side by side::

    for r in 1e-4 1e-5 1e-6 3e-7 1e-7 1e-8; do
        PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python tolerance_ladder.py \\
            --rtol $r --out raw/ladder_$r.json
    done

``raw/tolerance_ladder.json`` in this directory is the six merged runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_BENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BENCH))

import numpy as np  # noqa: E402

import coupling_fixtures as cf  # noqa: E402

#: The iteration caps the invariant lane overrides the registry with,
#: copied from ``tests/core/test_coupling_fixture_invariants.py``.  The
#: registry's own caps are what the benchmark sweep reports against and
#: are a measurement; these are a test premise, and the question here is
#: about the criterion, not about the budget.
_REACH_CAP = 120
_REACH_CAP_GRID = 60

#: Steps per configuration.  The lane's own number: enough for a
#: single-exit disagreement to compound into something a relative
#: measure can resolve.
_STEPS = 25


def _fixture_build(name, cap=_REACH_CAP):
    if name == "heterogeneous-2000":
        return lambda config: cf.build_heterogeneous(
            dataclasses.replace(config, max_iterations=_REACH_CAP_GRID),
            n_cells=2000,
        )
    build = cf.FIXTURES[name].build
    return lambda config: build(
        dataclasses.replace(config, max_iterations=cap))


def _run(built, n_steps):
    """(state, per-step total iterations, per-step converged flags)."""
    gm = built.gm
    iterations, converged = [], []
    for _ in range(n_steps):
        gm.step()
        diagnostics = gm.coupling_diagnostics()
        iterations.append(
            sum(int(d["iterations"]) for d in diagnostics.values()))
        converged.append(all(bool(d["converged"])
                             for d in diagnostics.values()))
    state = {
        (name, field): np.asarray(value, dtype=np.float64).ravel()
        for name in sorted(gm.node_names)
        for field, value in sorted(gm.get_node_state(name).items())
    }
    return state, iterations, converged


def _group_nodes(built):
    out = set()
    for key in built.group_keys:
        out.update(key.split("+"))
    return frozenset(out)


def _relative_spread(a, b, nodes):
    """The lane's own measure: per field name, over the group's nodes."""
    scales = {}
    for (node, field), arr in a.items():
        if node not in nodes:
            continue
        m = float(np.max(np.abs(arr))) if arr.size else 0.0
        scales[field] = max(scales.get(field, 0.0), m)
    worst = 0.0
    for key, ref in a.items():
        if key[0] not in nodes:
            continue
        scale = max(scales[key[1]], 1e-12)
        worst = max(worst, float(np.max(np.abs(ref - b[key]))) / scale)
    return worst


def measure(fixture, norms, atol, rtol, *, n_steps=_STEPS, cap=_REACH_CAP):
    """One fixture's 24 configurations at one (atol, rtol)."""
    build = _fixture_build(fixture, cap)
    reference = ref_nodes = None
    rows = {}
    for config in cf.sweep_configs(norms):
        if config.convergence_norm != "l2":
            overrides = {k: v for k, v in (("atol", atol), ("rtol", rtol))
                         if v is not None}
            config = dataclasses.replace(config, **overrides)
        built = build(config)
        state, iterations, converged = _run(built, n_steps)
        finite = bool(np.all(np.isfinite(
            np.concatenate(list(state.values())))))
        row = {
            "label": config.label,
            "norm": config.convergence_norm,
            "mean_iters": float(np.mean(iterations)),
            "conv_frac": float(np.mean(converged)),
            "finite": finite,
            "dev": 0.0,
        }
        if reference is None:
            reference, ref_nodes = state, _group_nodes(built)
        else:
            row["dev"] = _relative_spread(reference, state, ref_nodes)
        rows[config.label] = row
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", default="stiff-pair-0.5,chain-5,ring-8")
    ap.add_argument("--norms", default="l2,interface",
                    help="both, or the reference changes; see the docstring")
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--steps", type=int, default=_STEPS)
    ap.add_argument("--cap", type=int, default=_REACH_CAP)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    everything = {}
    for fixture in args.fixtures.split(","):
        rows = measure(fixture, tuple(args.norms.split(",")),
                       args.atol, args.rtol,
                       n_steps=args.steps, cap=args.cap)
        everything[fixture] = rows
        for label, row in sorted(rows.items()):
            print(f"{fixture:16s} {label:28s} dev={row['dev']:.3e} "
                  f"iters={row['mean_iters']:7.2f} "
                  f"conv={row['conv_frac']:.2f}"
                  f"{'' if row['finite'] else '  NONFINITE'}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(everything, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
