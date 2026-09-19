"""``converged=True`` implies the state is near the fixed point.

This is the one invariant the 0.4.0 convergence work exists to
establish, and it is stated here over *generated* graphs rather than
over the two or three hand-built fixtures that motivated it.  The
example-based cases live in ``tests/core/test_coupling_error_bound.py``;
what they cannot do is rule out a configuration nobody thought of, and
the coupling group has eighteen fields.

The measurement is deliberately independent of the criterion it checks.
Where ``coupling_diagnostics()`` reports ``error_estimate`` --
``residual / (1 - rho)``, with ``rho`` read off the residual sequence --
this module solves the *same group again* with a criterion a thousand
times tighter and a much larger iteration cap, and treats that answer as
the fixed point.  The distance between the two returned states is then
measured in the group's own convergence norm, which is the norm its
threshold is quoted in.  Nothing here reads the estimate to decide
whether the estimate was right.

Slack, and why there is any
---------------------------
``_SLACK`` is ``4``.  Three things make an exact ``distance <=
threshold`` the wrong assertion over arbitrary graphs:

* the bound is a *linear* extrapolation (``sum of a geometric series``)
  of a map that these fixtures do not promise is linear;
* the reference is itself only converged to ``threshold / 1000``, not to
  the exact fixed point;
* node state is float32, so a relative norm carries ~1e-7 of noise per
  field and a threshold near float32 resolution is mostly noise.

Four is loose enough to survive all three and tight enough to fail the
defect: ``MADD-ANO-005`` was measured at 15-31x its own tolerance on the
coupling audit's heterogeneous fixture, and the ``ift`` lag D2 recorded
was 2.5x on a two-node group.

The ``"interface"`` norm is excluded, and only because the edge list its
norm is taken over is not reachable from the public surface, so the
distance could not be measured in the same units the threshold is quoted
in.  The criterion machinery sees only a scalar residual and cannot tell
the three norms apart; ``tests/core/test_coupling_error_bound.py``
covers ``"interface"`` by example.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st

from maddening.core.coupling.acceleration import (
    coupling_residual_l2,
    coupling_residual_mixed,
)

from tests.conftest import EXAMPLES_COSTLY
from tests.core.test_coupling_solver_equivalence import (
    residual_noise_floor,
)
from tests.property.strategies import graph_recipes, without_inert_knobs

#: How much the linear extrapolation, the reference's own residual and
#: float32 are jointly allowed to be wrong by.  See the module docstring.
_SLACK = 4.0

#: How much tighter the reference solve is than the group under test.
_REFERENCE_FACTOR = 1e-3

#: The reference's iteration budget.  Generated groups draw caps of 2-6;
#: the reference has to be able to spend what a tighter criterion costs.
_REFERENCE_CAP = 60


def _threshold(group) -> float:
    return (1.0 if group.convergence_norm in ("mixed", "interface")
            else float(group.tolerance))


def _distance(group, got: dict, reference: dict, nodes: list[str]) -> float:
    """The group's own convergence norm between two states.

    The same function the solver measures its residual with, applied to
    the returned state and the reference fixed point instead of to two
    successive iterates -- so the number is directly comparable with the
    threshold, in the units the threshold is quoted in.
    """
    if group.convergence_norm == "mixed":
        return float(coupling_residual_mixed(
            reference, got, nodes, group.atol, group.rtol,
        ))
    return float(coupling_residual_l2(reference, got, nodes, group.atol))


def _tightened(recipe):
    """The same recipe, solved to a criterion ``_REFERENCE_FACTOR`` tighter.

    ``atol`` is deliberately *not* tightened: it is the dead band that
    decides which fields are in the norm at all, so moving it would
    measure the distance in different units from the ones the group's
    own threshold is quoted in.

    Only the knob the drawn norm actually reads is tightened.  ``"l2"``
    reads ``tolerance`` and never sees ``rtol``; ``"mixed"`` and
    ``"interface"`` read ``rtol`` and never see ``tolerance``.  Moving
    the inert one would be a no-op on the reference *and* would trip
    ``CouplingGroup``'s inert-knob ``UserWarning``, which
    ``filterwarnings = ["error"]`` makes fatal -- so the reference would
    fail to build rather than be loose.
    """
    def _tighter(g):
        live = ({"tolerance": g.tolerance * _REFERENCE_FACTOR}
                if g.convergence_norm == "l2"
                else {"rtol": g.rtol * _REFERENCE_FACTOR})
        return dataclasses.replace(
            g,
            max_iterations=_REFERENCE_CAP,
            diagnostics=True,
            strict_convergence=False,
            **live,
        )

    return dataclasses.replace(
        recipe,
        coupling_groups=tuple(_tighter(g) for g in recipe.coupling_groups),
    )


def _state_gap(a: dict, b: dict, nodes: list[str]) -> float:
    """Largest relative difference between two returned states."""
    worst = 0.0
    for node in nodes:
        for field in a[node]:
            x = np.asarray(a[node][field], np.float64)
            y = np.asarray(b[node][field], np.float64)
            scale = max(np.max(np.abs(x)), np.max(np.abs(y)), 1e-30)
            worst = max(worst, float(np.max(np.abs(x - y)) / scale))
    return worst


def _n_float_entries(state: dict, nodes: list[str]) -> int:
    """Float scalars in the group's state -- the L2 norm's sum length."""
    return sum(
        int(np.asarray(v).size)
        for node in nodes for v in state[node].values()
        if np.issubdtype(np.asarray(v).dtype, np.floating)
    )


def _diagnostics_recipe(recipe):
    """The recipe under test, with reporting forced on."""
    return dataclasses.replace(
        recipe,
        coupling_groups=tuple(
            dataclasses.replace(g, diagnostics=True, strict_convergence=False)
            for g in recipe.coupling_groups
        ),
    )


_RECIPES = graph_recipes(
    min_nodes=2, max_nodes=3,
    require_coupling_group=True,
    allow_mappings=False,
)


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(recipe=_RECIPES)
def test_converged_implies_the_state_is_within_tolerance_of_the_fixed_point(
    recipe,
):
    """The invariant, measured against an independently converged answer.

    Before 0.4.0 ``converged=True`` meant ``||F(x) - x|| <= tol``: a
    statement about the last step, which is short of the distance to the
    fixed point by ``1/(1 - rho)``.  A group whose slowest mode has
    ``rho`` near 1 could therefore report success arbitrarily far from
    the answer, and nothing in the library said so.  That is
    ``MADD-ANO-005``, and this is the assertion that retires it.
    """
    recipe = _diagnostics_recipe(recipe)
    gm = recipe.build()
    gm.step()
    diagnostics = gm.coupling_diagnostics()
    assume(diagnostics)

    groups = {"+".join(sorted(g.nodes)): g for g in gm._coupling_groups}  # noqa: SLF001
    converged = {
        key: groups[key] for key, d in diagnostics.items()
        if d["converged"] and groups[key].convergence_norm != "interface"
    }
    assume(converged)

    reference = _tightened(recipe).build()
    reference.step()
    ref_diagnostics = reference.coupling_diagnostics()

    checked = 0
    for key, group in converged.items():
        # A reference that did not itself converge is not a fixed point,
        # so it cannot be used to measure a distance to one.
        if not ref_diagnostics.get(key, {}).get("converged", False):
            continue
        nodes = sorted(group.nodes)
        got = {n: dict(gm.get_node_state(n)) for n in nodes}
        want = {n: dict(reference.get_node_state(n)) for n in nodes}
        distance = _distance(group, got, want, nodes)
        threshold = _threshold(group)
        d = diagnostics[key]
        note(f"{key}: distance={distance:.3e} threshold={threshold:.3e} "
             f"residual={d['residual']:.3e} "
             f"estimate={d['error_estimate']:.3e} "
             f"amp={d['amplification']} valid={d['bound_valid']} "
             f"norm={group.convergence_norm} accel={group.acceleration}")
        if not jnp.isfinite(distance):
            continue                 # a diverged reference measures nothing
        assert distance <= threshold * _SLACK, (
            f"{key} reported converged at threshold {threshold} but the "
            f"state it returned is {distance} from a reference solved "
            f"{1 / _REFERENCE_FACTOR:.0f}x tighter"
        )
        assert d["error_estimate"] * _SLACK >= distance, (
            f"{key}: the reported error estimate {d['error_estimate']} does "
            f"not bound the measured distance {distance}"
        )
        checked += 1
    assume(checked)


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(recipe=_RECIPES)
def test_the_new_criterion_is_never_looser_than_the_residual_test(recipe):
    """The compatibility half of the change, as an exact statement.

    ``error_estimate = residual * max(amplification, 1)`` and a valid
    amplification is ``1/(1 - rho)`` with ``rho`` in ``[0, 1)``, so the
    estimate is never below the residual.  Two consequences follow and
    both are asserted here: a group that meets the new criterion also
    meets the old one -- so D2's guarantee that ``converged=True``
    describes the state you were handed is not weakened -- and a
    rejected estimate degrades to exactly the old criterion rather than
    to something unpredictable.
    """
    gm = _diagnostics_recipe(recipe).build()
    gm.step()
    diagnostics = gm.coupling_diagnostics()
    assume(diagnostics)

    groups = {"+".join(sorted(g.nodes)): g for g in gm._coupling_groups}  # noqa: SLF001
    for key, d in diagnostics.items():
        threshold = _threshold(groups[key])
        note(f"{key}: {d}")
        if d["bound_valid"]:
            assert d["amplification"] >= 1.0
            assert d["error_estimate"] == pytest.approx(
                d["residual"] * d["amplification"], rel=1e-5,
            )
            assert d["gradient_error_bound"] == pytest.approx(
                d["error_estimate"],
            )
        else:
            assert d["error_estimate"] == pytest.approx(d["residual"])
            assert d["gradient_error_bound"] == float("inf")
        if d["converged"]:
            assert d["residual"] <= threshold, (
                "the new criterion let through a state the residual test "
                "would have rejected"
            )


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(recipe=_RECIPES, solver=st.sampled_from(["ift", "fori"]))
def test_the_bound_is_the_same_on_both_solvers(recipe, solver):
    """``solver`` stays invisible in the report, new fields included.

    D2's other guarantee: the two paths return the same state and the
    same verdict.  The error bound is derived from the same residual
    sequence on both, so a divergence here would mean one of them is
    measuring a different sequence.

    *The state and the verdict are exact claims; the residual is not.*
    This asserted ``residual`` equality at ``abs=1e-9``, and a
    generated multi-rate group falsified it: ``ift`` read ``0.0`` where
    ``fori`` read ``1.04e-05`` on a graph both returned the same state
    for.  Neither was measuring a different sequence.  Every norm here
    divides ``F(x) - x`` by a scale, so a converged group's residual is
    a *cancellation*, and the two solvers run their passes in different
    loop constructs -- ``lax.while_loop`` for the early-exiting
    ``ift``, ``lax.fori_loop`` for ``fori`` -- which XLA compiles to
    differently rounded arithmetic.  One ulp on the map's output is a
    full-size change to a residual that small.  ``1e-9`` was three
    thousand times below the measurement's own resolution; the honest
    comparison is against that resolution, which
    :func:`residual_noise_floor` derives from the norm.  The worked
    reproducer is in ``tests/core/test_coupling_solver_equivalence.py``.
    """
    base = _diagnostics_recipe(recipe)
    other = "fori" if solver == "ift" else "ift"
    built = {}
    states = {}
    groups = {}
    for name in (solver, other):
        # ``linear_solver`` and ``strict_convergence`` are read inside
        # the IFT path alone, so flipping to ``"fori"`` strands whatever
        # the draw put in them and the group warns -- fatally, under
        # ``filterwarnings = ["error"]``.  Resetting them is not a
        # weakening of the comparison: they are exactly the fields
        # ``"fori"`` does not read.
        r = dataclasses.replace(
            base,
            coupling_groups=tuple(
                without_inert_knobs(dataclasses.replace(g, solver=name))
                for g in base.coupling_groups
            ),
        )
        gm = r.build()
        gm.step()
        built[name] = gm.coupling_diagnostics()
        groups[name] = {"+".join(sorted(g.nodes)): g
                        for g in gm._coupling_groups}       # noqa: SLF001
        states[name] = {n: dict(gm.get_node_state(n))
                        for n in gm.node_names}
    assume(built[solver])
    assert set(built[solver]) == set(built[other])
    for key in built[solver]:
        a, b = built[solver][key], built[other][key]
        group = groups[solver][key]
        nodes = sorted(group.nodes)
        note(f"{key}: {solver}={a} {other}={b}")

        # The claim that matters, and the one this test did not make:
        # whatever the two reports say, the states they describe are the
        # same one.  Float32 round-off only -- the acceleration carries
        # the last-bit difference of the map into the iterate.
        gap = _state_gap(states[solver], states[other], nodes)
        assert gap <= 1e-5, (
            f"{key}: the two solvers returned states {gap} apart "
            "relatively, which is a solver defect and not round-off"
        )
        assert a["converged"] == b["converged"]

        floor = residual_noise_floor(
            group.convergence_norm, group.rtol,
            _n_float_entries(states[solver], nodes),
        )
        assert a["residual"] == pytest.approx(b["residual"], rel=1e-4,
                                              abs=floor, nan_ok=True)
        # ``bound_valid`` is a statement about the *ratio* of the last
        # two residuals.  Where both are at the noise floor that ratio
        # is a ratio of rounding, and one solver rejecting it while the
        # other accepts it says nothing about either.  Above the floor
        # the two must agree.
        if min(a["residual"], b["residual"]) > floor:
            assert a["bound_valid"] == b["bound_valid"]
