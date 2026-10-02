"""Differential oracle: a coupled group's verdict does not depend on its units.

``CouplingGroup.convergence_norm`` divides every field's change by that
field's own magnitude so that "a group's verdict does not depend on the
units its quantities are written in".  The same group written at another
scale -- every bias and initial state multiplied by ``s``, so on a linear
group every iterate is ``s`` times the unscaled one -- must therefore take
the same passes and give the same verdict.  Under ``acceleration="aitken"``
it did not: the guard on Aitken's denominator was an absolute
``sum(delta_r**2) > 1e-30``, so at ``s = 1e-12`` omega froze part-way
through the solve and the group hit its cap unconverged where it converged
in 88 passes at ``s = 1``; IQN's blow-up guard had an absolute floor and its
least-squares solve was rescaled by LAPACK at extreme scales.

**The scales are powers of two** (``2**-20``, ``2**-40``, ``2**20`` stand in
for ``1e-6``, ``1e-12``, ``1e6``).  Multiplying a float32 by a power of two is
exact, so the scaled group's inputs are the unscaled ones' bit patterns with
another exponent, and a units-invariant pipeline reproduces *every* bit:
the pass count, the verdict, the residual, and the state divided by ``s``.
A decimal scale would change the inputs' rounding and so test rounding
sensitivity as much as units -- Aitken, whose omega is a ratio of
near-cancelling dot products, takes 84-92 passes at decimal scales of one
group (all converged) for that reason alone.  So the claim here is exact,
and anything short of bit-identity is a constant in the pipeline that does
not scale.

What makes a group scale-equivariant, and so what the strategy keeps:
linear nodes (no ``tanh``), no ``beta * dt`` term (an absolute forcing), no
leaves with absolute updates, and the dead band off (``atol=0``, the
default: a declared absolute "zero" is units by definition).
"""

from __future__ import annotations

import dataclasses
import functools

import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.property import coupled_graphs as cg

#: Powers of two standing in for 1e-6, 1e-12 and 1e6 (see the module docstring).
SCALES = (2.0 ** -20, 2.0 ** -40, 2.0 ** 20)
STEPS = 2


def _equivariant(gdef: cg.GraphDef) -> cg.GraphDef:
    """*gdef* with every absolute term removed: no ``beta * dt``, no ``tanh``, no leaves."""
    nodes = tuple(dataclasses.replace(nd, beta=0.0, nonlinear=False, leaves=())
                  for nd in gdef.nodes)
    return dataclasses.replace(gdef, nodes=nodes)


def _scaled(values: dict, s: float) -> dict:
    return {nm: {"G": v["G"], "b": np.asarray(v["b"] * np.float32(s), np.float32),
                 "x0": np.asarray(v["x0"] * np.float32(s), np.float32)}
            for nm, v in values.items()}


def assert_units_invariant(gdef, gm, values):
    """Every scale reproduces the unscaled run's passes, verdict, residual and state."""
    base = cg.trajectory(gm, gdef, values, STEPS)
    for s in SCALES:
        run = cg.trajectory(gm, gdef, _scaled(values, s), STEPS)
        for k, ((st0, _m0, r0), (st1, _m1, r1)) in enumerate(zip(base, run)):
            where = f"scale {s!r}, step {k + 1}"
            if r0 is not None:
                cg.note(f"{where}: unscaled {r0}, scaled {r1}")
                assert (r1["iterations"], r1["converged"]) == (
                    r0["iterations"], r0["converged"]), (
                    f"{where}: {r1['iterations']} passes, converged={r1['converged']} "
                    f"against {r0['iterations']}, converged={r0['converged']} unscaled")
                assert r1["residual"] == r0["residual"], (where, r0, r1)
                assert r1["amplification"] == r0["amplification"], (where, r0, r1)
            for nm in gdef.group_nodes:
                got = np.asarray(st1[nm]["x"], np.float32) / np.float32(s)
                np.testing.assert_array_equal(
                    got, st0[nm]["x"], err_msg=f"{where}: {nm}.x / s differs from unscaled")


#: Per-push: every acceleration, with the norms, modes and solvers rotated
#: through them.  One graph each, compiled once; the values are drawn.
_CASES = {
    "none-l2-gs-ift": (dict(acceleration="none", tolerance=1e-5), "triangle"),
    "fixed-mixed-jacobi-fori": (dict(acceleration="fixed", relaxation=0.8,
                                     convergence_norm="mixed", rtol=1e-4,
                                     iteration_mode="jacobi", solver="fori",
                                     diagnostics=True), "triangle"),
    "aitken-l2-jacobi-ift": (dict(acceleration="aitken", tolerance=1e-5,
                                  iteration_mode="jacobi"), "triangle"),
    "aitken-interface-gs-fori": (dict(acceleration="aitken", convergence_norm="interface",
                                      rtol=1e-4, solver="fori", diagnostics=True), "ring"),
    "iqn-ils-mixed-gs-ift": (dict(acceleration="iqn-ils", convergence_norm="mixed",
                                  rtol=1e-4), "ring"),
    "iqn-imvj-l2-jacobi-ift": (dict(acceleration="iqn-imvj", jacobian_reuse=2,
                                    tolerance=1e-5, iteration_mode="jacobi"), "triangle"),
    "iqn-ils-interface-gs-ift-predictor": (dict(acceleration="iqn-ils",
                                                convergence_norm="interface", rtol=1e-4,
                                                predictor="linear"), "triangle"),
}

_STRUCTURES = {
    # The triangle with its outside driver and sink; a ring with a chord.
    "triangle": _equivariant(cg._cycle(3, 2, chords=((0, 2),), leaves=())),
    "ring": _equivariant(cg._cycle(4, 1, chords=((1, 3),), leaves=(), outside=False)),
}


@functools.lru_cache(maxsize=None)
def _graph(case):
    group, structure = _CASES[case]
    gdef = _STRUCTURES[structure]
    return gdef, cg.build_graph(gdef, dict(group, max_iterations=40))


@pytest.mark.parametrize("case", sorted(_CASES))
# Costly tier: each example runs 2 steps at 4 scales on one compiled graph.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_coupled_group_takes_the_same_passes_at_every_scale(case, data):
    """Per push; slow sibling :func:`test_units_invariance_on_generated_graphs`."""
    gdef, gm = _graph(case)
    values = data.draw(cg.drawn_values(gdef, rhos=(0.5, 0.9, 0.99), bias_scales=(1.0,)))
    assert_units_invariant(gdef, gm, values)


def test_aitken_takes_the_same_passes_at_every_scale_on_the_finding():
    """The finding's own pair: converged in 88 passes at scale 1, capped at 1e-12.

    ``a <- diag(0.95, 0.9) b + s (1, 2)``, ``b <- diag(0.9, 0.8) a``, Jacobi,
    tolerance 1e-5, from zero, a cap of 100; both solvers.
    """
    gdef = cg.GraphDef(n=2, nodes=(cg.NodeDef("a", 1), cg.NodeDef("b", 1)),
                       edges=(cg.EdgeDef("b", "a", 0), cg.EdgeDef("a", "b", 0)),
                       group_nodes=("a", "b"))
    values = {"a": {"G": [np.diag([0.95, 0.9]).astype(np.float32)],
                    "b": np.array([1.0, 2.0], np.float32), "x0": np.zeros(2, np.float32)},
              "b": {"G": [np.diag([0.9, 0.8]).astype(np.float32)],
                    "b": np.zeros(2, np.float32), "x0": np.zeros(2, np.float32)}}
    for solver in ("ift", "fori"):
        gm = cg.build_graph(gdef, dict(acceleration="aitken", iteration_mode="jacobi",
                                       tolerance=1e-5, max_iterations=100, solver=solver,
                                       diagnostics=True))
        (_s, _m, rep), = cg.trajectory(gm, gdef, values, 1)
        assert rep["converged"] and rep["iterations"] < 100, rep
        assert_units_invariant(gdef, gm, values)


# Slow: structure and configuration are drawn, so every example builds and
# compiles a graph of its own (seconds each on CI).
# Per push: tests/property/test_differential_coupling_units.py::test_a_coupled_group_takes_the_same_passes_at_every_scale
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_units_invariance_on_generated_graphs(data):
    """The same claim with the structure, every knob and the solver drawn.

    Per-push sibling: :func:`test_a_coupled_group_takes_the_same_passes_at_every_scale`.
    """
    gdef = _equivariant(data.draw(cg.graph_defs(allow_nonlinear=False, leaves=())))
    group = data.draw(cg.group_configs(gdef, caps=(2, 5, 12, 40),
                                       thresholds=(1e-6, 1e-4, 1e-2)))
    group["solver"] = data.draw(st.sampled_from(["ift", "fori"]))
    group["diagnostics"] = data.draw(st.booleans())
    note(f"{gdef}\n{group}")
    gm = cg.build_graph(gdef, cg.live_knobs(group))
    values = data.draw(cg.drawn_values(gdef, rhos=(0.5, 0.9, 0.99), bias_scales=(1.0,)))
    assert_units_invariant(gdef, gm, values)
