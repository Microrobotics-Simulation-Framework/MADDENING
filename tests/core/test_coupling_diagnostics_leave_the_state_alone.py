"""``diagnostics=True`` reports on the forward; it does not change it.

A diagnostic that moves the answer it diagnoses is a second, silent
configuration knob.  Both solvers compute extra quantities when a group
asks for diagnostics -- the ``fori`` path carries an iteration count, a
residual and an amplification through its loop; the ``ift`` path takes
Jacobian-vector products for the spectral and gradient bounds -- and
XLA's simplifier rewrites a computation according to how many users
each of its values has.  The ``ift`` path isolated its extra work in
``lax.cond`` branches after the returned state moved by an ulp; the
``fori`` path kept its amplification carry sharing the criterion's
arithmetic, and on 4 of 36 chain-5 configurations its returned state
moved by one ulp from the first step.  These are those configurations,
pinned bit-for-bit on both solvers.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

import maddening

_FIXTURES_PY = Path(maddening.__file__).resolve().parents[2] / "benchmarks" / "coupling_fixtures.py"


def _fixtures():
    name = "maddening_coupling_fixtures"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(_FIXTURES_PY))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: ``(iteration_mode, convergence_norm)`` of the four chain-5
#: configurations whose ``fori`` state moved under ``diagnostics=True``
#: (jaxlib 0.11.0, CPU), all at ``acceleration="none"``.
_MOVED = (
    ("gauss-seidel", "interface"),
    ("jacobi", "l2"),
    ("jacobi", "mixed"),
    ("jacobi", "interface"),
)

_STEPS = 3


def _trajectory(solver, diagnostics, iteration_mode, norm):
    cf = _fixtures()
    config = cf.CouplingConfig(iteration_mode=iteration_mode, acceleration="none",
                               convergence_norm=norm)
    gm = cf.FIXTURES["chain-5"].build(config).gm
    gm._coupling_groups = [
        dataclasses.replace(g, solver=solver, diagnostics=diagnostics)
        for g in gm._coupling_groups
    ]
    gm.compile()
    out = []
    for _ in range(_STEPS):
        gm.step()
        out.append({n: {f: np.asarray(v).tobytes() for f, v in gm.get_node_state(n).items()}
                    for n in gm.node_names})
    return out


@pytest.mark.parametrize("solver", ("fori", "ift"))
@pytest.mark.parametrize("iteration_mode,norm", _MOVED)
def test_diagnostics_leave_the_returned_state_bit_identical(solver, iteration_mode, norm):
    off = _trajectory(solver, False, iteration_mode, norm)
    on = _trajectory(solver, True, iteration_mode, norm)
    for k, (a, b) in enumerate(zip(off, on), start=1):
        moved = sorted(f"{n}.{f}" for n in a for f in a[n] if a[n][f] != b[n][f])
        assert not moved, (
            f"{solver} {iteration_mode}/{norm}: diagnostics=True moved "
            f"{moved} at step {k}"
        )
