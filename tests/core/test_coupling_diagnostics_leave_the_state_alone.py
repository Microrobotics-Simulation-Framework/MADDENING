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

They are all ``acceleration="none"``, and a gate that sees one acceleration
cannot fail for the others: a mutant that moved the state one ulp only
under an accelerator passed it.  So a per-push subset covers every
accelerator and both solvers (and, under ``"ift"``, the recorded iteration
count, residual and amplification, which ``"ift"`` writes whatever
``diagnostics`` says), and a slow-marked sweep covers every chain-5
configuration -- the fixture the drift was found on.  The 0.4.0 audit ran
the same comparison over 432 configurations on six fixtures, with none
differing (jaxlib 0.11.0).
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

#: The ``_meta`` slots both settings write under ``solver="ift"``.
_REPORT_SLOTS = ("iterations", "residual", "amplification")

#: ``acceleration -> CouplingConfig`` keywords.
_ACCELERATIONS = {
    "none": {},
    "aitken": {"acceleration": "aitken"},
    "fixed0.5": {"acceleration": "fixed", "relaxation": 0.5},
    "fixed1.5": {"acceleration": "fixed", "relaxation": 1.5},
    "iqn-ils": {"acceleration": "iqn-ils"},
    "iqn-imvj3": {"acceleration": "iqn-imvj", "jacobian_reuse": 3},
}


def _trajectory(solver, diagnostics, iteration_mode, norm, acceleration="none"):
    cf = _fixtures()
    config = cf.CouplingConfig(iteration_mode=iteration_mode, convergence_norm=norm,
                               **_ACCELERATIONS[acceleration])
    gm = cf.FIXTURES["chain-5"].build(config).gm
    gm._coupling_groups = [
        dataclasses.replace(g, solver=solver, diagnostics=diagnostics)
        for g in gm._coupling_groups
    ]
    gm.compile()
    out = []
    for _ in range(_STEPS):
        gm.step()
        state = {f"{n}.{f}": np.asarray(v).tobytes()
                 for n in gm.node_names for f, v in gm.get_node_state(n).items()}
        meta = gm._state.get("_meta", {})
        slots = {k: np.asarray(v).tobytes() for k, v in meta.items()
                 if solver == "ift" and k.startswith("coupling_")
                 and k.rsplit("_", 1)[-1] in _REPORT_SLOTS}
        out.append((state, slots))
    return out


def _assert_bit_identical(solver, iteration_mode, norm, acceleration="none"):
    off = _trajectory(solver, False, iteration_mode, norm, acceleration)
    on = _trajectory(solver, True, iteration_mode, norm, acceleration)
    label = f"{solver} {iteration_mode}/{acceleration}/{norm}"
    for k, ((state0, slots0), (state1, slots1)) in enumerate(zip(off, on), start=1):
        moved = sorted(f for f in state0 if state0[f] != state1[f])
        assert not moved, f"{label}: diagnostics=True moved {moved} at step {k}"
        assert solver == "fori" or slots0, label
        changed = sorted(key for key in slots0 if slots1.get(key) != slots0[key])
        assert not changed, f"{label}: diagnostics=True changed {changed} at step {k}"


@pytest.mark.parametrize("solver", ("fori", "ift"))
@pytest.mark.parametrize("iteration_mode,norm", _MOVED)
def test_diagnostics_leave_the_returned_state_bit_identical(solver, iteration_mode, norm):
    _assert_bit_identical(solver, iteration_mode, norm)


#: Per push: every accelerator once, each solver twice, every norm and
#: both iteration modes -- ``(solver, iteration_mode, acceleration, norm)``.
_ACCELERATED_PER_PUSH = (
    ("ift", "gauss-seidel", "aitken", "mixed"),
    ("ift", "jacobi", "iqn-imvj3", "l2"),
    ("fori", "jacobi", "fixed1.5", "l2"),
    ("fori", "gauss-seidel", "iqn-ils", "interface"),
)


@pytest.mark.parametrize("solver,iteration_mode,acceleration,norm", _ACCELERATED_PER_PUSH)
def test_diagnostics_leave_an_accelerated_group_bit_identical(
        solver, iteration_mode, acceleration, norm):
    _assert_bit_identical(solver, iteration_mode, norm, acceleration)


def _broad_cases():
    covered = {(s, m, "none", n) for m, n in _MOVED for s in ("fori", "ift")}
    covered |= set(_ACCELERATED_PER_PUSH)
    return [
        (solver, mode, acceleration, norm)
        for solver in ("fori", "ift")
        for mode in ("gauss-seidel", "jacobi")
        for acceleration in _ACCELERATIONS
        for norm in ("l2", "mixed", "interface")
        if (solver, mode, acceleration, norm) not in covered
    ]


@pytest.mark.slow
@pytest.mark.parametrize("solver,iteration_mode,acceleration,norm", _broad_cases())
def test_diagnostics_leave_every_chain_configuration_bit_identical(
        solver, iteration_mode, acceleration, norm):
    _assert_bit_identical(solver, iteration_mode, norm, acceleration)
