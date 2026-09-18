#!/usr/bin/env python
"""What the ten disagreements cost when the accelerator sees every field.

The counterfactual to :mod:`tolerance_ladder`.  That script moves the
knob ``_KNOWN_DISAGREEMENTS`` prescribes (``rtol``) and finds it cannot
retire the list; this one leaves the fixtures' own ``rtol=1e-4`` exactly
where it is and moves ``accelerated_fields`` instead, from the
auto-detected interface set (``accel_scope="auto"``) to the whole state
(``accel_scope="all"``).

Both columns are printed for every IQN row of the three fixtures the
slow lane runs, with the ten listed rows marked ``*``.  The two numbers
that matter are the deviation -- does the row come inside the lane's
2.5e-02 threshold -- and the iteration count, because a remedy that
costs passes is a trade-off and one that does not is free.

``accel_scope="all"`` is already a configuration the sweep knows how to
run (``sweep_configs(extra_fields=True)`` crosses the IQN rows with
``all`` / ``expensive`` / ``cheap`` on the heterogeneous fixture).  What
it is *not* is the sweep's default, and making it one would change what
the sweep measures about ``accelerated_fields`` -- which is the subject
of a section of the algorithm guide.  This script reports the number;
the change is the maintainer's.

No timings: the machine is shared.

Usage
-----
::

    PYTHONPATH=<wt>/src JAX_PLATFORMS=cpu python scope_experiment.py
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_BENCH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_BENCH))

import numpy as np  # noqa: E402

import coupling_fixtures as cf  # noqa: E402

from tolerance_ladder import (  # noqa: E402
    _REACH_CAP,
    _STEPS,
    _group_nodes,
    _relative_spread,
    _run,
)

#: The lane's agreement threshold for interface-norm rows.
_INTERFACE_AGREEMENT = 2.5e-2

#: The ten entries, as ``(fixture, label)``, so the output says which
#: rows the question is actually about.
_LISTED = frozenset({
    ("stiff-pair-0.5", "gs/iqn-ils/interface"),
    ("stiff-pair-0.5", "gs/iqn-imvj5/interface"),
    ("stiff-pair-0.5", "jac/iqn-ils/interface"),
    ("stiff-pair-0.5", "jac/iqn-imvj5/interface"),
    ("chain-5", "gs/iqn-imvj5/interface"),
    ("chain-5", "jac/iqn-ils/interface"),
    ("chain-5", "jac/iqn-imvj5/interface"),
    ("ring-8", "gs/iqn-imvj5/interface"),
    ("ring-8", "jac/iqn-ils/interface"),
    ("ring-8", "jac/iqn-imvj5/interface"),
})

_IQN = (("iqn-ils", {}), ("iqn-imvj", {"jacobian_reuse": 5}))


def _solve(fixture, config, cap=_REACH_CAP, n_steps=_STEPS):
    built = cf.FIXTURES[fixture].build(
        dataclasses.replace(config, max_iterations=cap))
    state, iterations, converged = _run(built, n_steps)
    return state, _group_nodes(built), float(np.mean(iterations)), \
        float(np.mean(converged))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fixtures", default="stiff-pair-0.5,chain-5,ring-8")
    args = ap.parse_args()

    for fixture in args.fixtures.split(","):
        # The lane's reference: the first configuration of the sweep.
        reference, nodes, _, _ = _solve(fixture, cf.CouplingConfig())
        for acceleration, extra in _IQN:
            for mode in ("gauss-seidel", "jacobi"):
                cells = []
                for scope in ("auto", "all"):
                    config = cf.CouplingConfig(
                        iteration_mode=mode, acceleration=acceleration,
                        convergence_norm="interface", accel_scope=scope,
                        **extra,
                    )
                    state, _, iterations, converged = _solve(fixture, config)
                    cells.append((
                        scope,
                        _relative_spread(reference, state, nodes),
                        iterations, converged,
                    ))
                label = cf.CouplingConfig(
                    iteration_mode=mode, acceleration=acceleration,
                    convergence_norm="interface", **extra,
                ).label
                mark = "*" if (fixture, label) in _LISTED else " "
                print(f"{mark}{fixture:16s} {label:24s} " + "  ".join(
                    f"{scope}: dev={dev:.3e}{'' if dev > _INTERFACE_AGREEMENT else ' OK'}"
                    f" it={iterations:6.2f} conv={converged:.2f}"
                    for scope, dev, iterations, converged in cells
                ), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
