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
from hypothesis import assume, example, given, note, settings
from hypothesis import strategies as st

from maddening.core.coupling.acceleration import (
    coupling_residual_l2,
    coupling_residual_mixed,
    relaxation_step_scale,
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


#: Floor on the scale a state difference is divided by.  A generated
#: graph can leave a field at 1e-20 or at exactly zero, where a pure
#: relative measure reads a one-subnormal difference as O(1) and the
#: comparison becomes noise about nothing.  Below this the check is
#: absolute instead, which at the 1e-5 tolerance it is used with is
#: 1e-11 -- orders above the few ulps float32 can put there.
_STATE_SCALE_FLOOR = 1e-6


def _state_gap(a: dict, b: dict, nodes: list[str]) -> float:
    """Largest relative difference between two returned states."""
    worst = 0.0
    for node in nodes:
        for field in a[node]:
            x = np.asarray(a[node][field], np.float64)
            y = np.asarray(b[node][field], np.float64)
            scale = max(np.max(np.abs(x)), np.max(np.abs(y)),
                        _STATE_SCALE_FLOOR)
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
             f"amp={d['amplification']} valid={d['ratio_usable']} "
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

    ``error_estimate = residual * max(omega * amplification, 1)``,
    where ``omega`` is the step scale
    (:func:`~maddening.core.coupling.acceleration.relaxation_step_scale`:
    the relaxation factor under ``acceleration="fixed"``, 1 otherwise)
    and a valid amplification is ``1/(1 - rho)`` with ``rho`` in
    ``[0, 1)``.  The ``max`` is what keeps the estimate from dropping
    below the residual under *under*-relaxation, and it is why the two
    consequences asserted here survive ``omega``: a group that meets
    the new criterion also meets the old one -- so D2's guarantee that
    ``converged=True`` describes the state you were handed is not
    weakened -- and a rejected estimate degrades to exactly the old
    criterion rather than to something unpredictable.
    """
    gm = _diagnostics_recipe(recipe).build()
    gm.step()
    diagnostics = gm.coupling_diagnostics()
    assume(diagnostics)

    groups = {"+".join(sorted(g.nodes)): g for g in gm._coupling_groups}  # noqa: SLF001
    for key, d in diagnostics.items():
        threshold = _threshold(groups[key])
        note(f"{key}: {d}")
        if d["ratio_usable"]:
            assert d["amplification"] >= 1.0
            scale = relaxation_step_scale(
                groups[key].acceleration, groups[key].relaxation,
            )
            assert d["error_estimate"] == pytest.approx(
                d["residual"] * max(scale * d["amplification"], 1.0),
                rel=1e-5,
            )
            assert d["gradient_error_estimate"] == pytest.approx(
                d["error_estimate"],
            )
        else:
            assert d["error_estimate"] == pytest.approx(d["residual"])
            assert d["gradient_error_estimate"] == float("inf")
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
        # ``ratio_usable`` is a statement about the *ratio* of the last
        # two residuals.  Where both are at the noise floor that ratio
        # is a ratio of rounding, and one solver rejecting it while the
        # other accepts it says nothing about either.  Above the floor
        # the two must agree.
        if min(a["residual"], b["residual"]) > floor:
            assert a["ratio_usable"] == b["ratio_usable"]


# ---------------------------------------------------------------------------
# Two further mechanisms, from the 2026-09-19 coupling audit
#
# The generated recipes above draw two- and three-node graphs of library
# nodes, and neither mechanism showed up there: one needs a *chosen*
# relaxation factor on a group slow enough to have a tail, the other
# needs a *chosen* spectrum with a stiff mode hiding behind a fast one.
# Both are cheap to build directly from an affine node, and the fixed
# point is then in closed form, so these two properties measure against
# the exact answer rather than against a tighter solve.
#
# See ``benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md``.
# ---------------------------------------------------------------------------

from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.core.node import (  # noqa: E402
    BoundaryInputSpec,
    SimulationNode,
)


class _Affine(SimulationNode):
    """``x <- gain * u + bias`` on ``n`` independent modes."""

    def __init__(self, name, gain, bias):
        super().__init__(name=name, timestep=1.0)
        self._gain = jnp.asarray(gain, jnp.float32)
        self._bias = jnp.asarray(bias, jnp.float32)
        self._shape = jnp.shape(self._gain)

    def initial_state(self):
        return {"x": jnp.zeros(self._shape, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=self._shape, dtype=jnp.float32,
                                       description="u")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": self._gain * boundary_inputs["u"] + self._bias}


def _affine_cycle(gain, bias, **group_kw):
    """``a -> b -> a`` carrying ``diag(gain)``; one sweep is ``rho = gain``.

    ``b`` is the identity relay, so a Gauss-Seidel pass over the group
    advances ``a`` by ``x -> gain * x + bias`` and the fixed point is
    ``bias / (1 - gain)`` per mode.
    """
    ones = jnp.ones_like(jnp.asarray(gain, jnp.float32))
    gm = GraphManager()
    gm.add_node(_Affine("a", gain, bias))
    gm.add_node(_Affine("b", ones, jnp.zeros_like(ones)))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, **group_kw)
    gm.compile()
    return gm


def _exact_distance(gm, gain, bias):
    """Distance to the analytic fixed point, in the group's own L2 norm."""
    exact = [b / (1.0 - g) for g, b in zip(jnp.atleast_1d(jnp.asarray(gain)),
                                           jnp.atleast_1d(jnp.asarray(bias)))]
    exact = [float(v) for v in exact]
    total = 0.0
    for node in ("a", "b"):
        got = [float(v) for v in jnp.atleast_1d(gm.get_node_state(node)["x"])]
        ref = max(max(abs(v) for v in got), max(abs(v) for v in exact))
        if ref == 0.0:
            continue
        total += sum(((g - e) / ref) ** 2 for g, e in zip(got, exact))
    return total ** 0.5


#: Not tighter than 1e-3: the norm is relative, the fixed points here
#: are O(1)-O(100), and float32 resolves ~1e-7 of them, so below about
#: 1e-4 the residual *ratio* the estimate rests on is reading round-off.
_ANALYTIC_TOLERANCE = 1e-3


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(
    gain=st.floats(min_value=0.4, max_value=0.95),
    omega=st.floats(min_value=0.3, max_value=1.95),
)
def test_the_estimate_is_invariant_to_the_relaxation_factor(gain, omega):
    """``relaxation`` changes how far each pass goes, not how far is left.

    The exact statement, for a single mode of rate ``rho`` under
    constant relaxation ``omega``.  The iterate contracts at
    ``mu = 1 - omega * (1 - rho)`` and the step it takes is
    ``omega * (F(x) - x)``, so the distance to the fixed point is
    ``residual / (1 - rho)`` -- free of ``omega`` -- while the
    geometric series of steps sums to
    ``omega * residual / (1 - |mu|)``.  The two agree exactly when
    ``mu >= 0`` and the series is *larger* when ``mu < 0``, because an
    over-relaxed iterate that overshoots walks further than the
    straight-line distance it covers.  So:

    * the estimate never understates, which is what summing residuals
      instead of steps used to break (``est/true`` tracked ``1/omega``:
      0.68 at ``omega=1.5``, 0.51 at ``omega=1.95``);
    * and it is tight, not merely safe, wherever the iteration does not
      overshoot.
    """
    bias = 1.0
    gm = _affine_cycle(
        gain, bias, max_iterations=500, tolerance=_ANALYTIC_TOLERANCE,
        acceleration="fixed", relaxation=omega,
    )
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assume(d["converged"] and d["ratio_usable"])
    distance = _exact_distance(gm, gain, bias)
    assume(distance > 0.0)

    ratio = d["error_estimate"] / distance
    mu = 1.0 - omega * (1.0 - gain)
    note(f"gain={gain} omega={omega} mu={mu} ratio={ratio} {d}")
    assert ratio >= 0.9, (
        f"understated by {1 / ratio:.2f}x at relaxation={omega}: the series "
        f"is summing residuals rather than the steps the iterate takes"
    )
    if mu >= 0.05:
        assert ratio <= 1.2, (
            f"a non-overshooting iteration should be estimated tightly; "
            f"got {ratio:.3f} at gain={gain}, omega={omega}"
        )


@pytest.mark.xfail(strict=True, reason=(
    "Recorded, not accepted: `rho` is read from the residual sequence, "
    "which reports the mode dominating the *step*.  Until the fast mode "
    "has decayed, that sequence is indistinguishable from a single-mode "
    "decay at the fast rate -- so no test on it, including the two-step "
    "sqrt guard, can tell that the remaining error already belongs to a "
    "much slower mode.  The pinned example is the audit's: (0.999, 0.2) "
    "reports 9.19e-05 against a true distance of 1.12e-02, 122x, with "
    "ratio_usable=True.  A real fix needs the spectrum rather than the "
    "residual sequence and is post-0.4.0 work.  Flipping this to a pass "
    "means the estimate became a bound: update "
    "benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md, the "
    "caveat on _fixed_point_while, and the example-based twin in "
    "tests/core/test_coupling_error_bound.py."
))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(
    rho_slow=st.floats(min_value=0.99, max_value=0.9999),
    rho_fast=st.floats(min_value=0.0, max_value=0.5),
    c_slow=st.floats(min_value=1e-6, max_value=1e-3),
)
@example(rho_slow=0.999, rho_fast=0.2, c_slow=1e-5)
def test_the_estimate_is_never_smaller_than_the_distance_it_estimates(
    rho_slow, rho_fast, c_slow,
):
    """The property ``error_estimate`` would need for its name to hold.

    A two-mode contraction, generated: a stiff mode carrying very
    little per pass but amplified by ``1/(1 - rho_slow)``, and a fast
    one carrying O(1) per pass.  The stiff mode owns the distance to
    the fixed point long before it owns the residual.
    """
    gain = (rho_slow, rho_fast)
    bias = (c_slow, 1.0)
    gm = _affine_cycle(gain, bias, max_iterations=60,
                       tolerance=_ANALYTIC_TOLERANCE)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    assume(d["converged"] and d["ratio_usable"])
    distance = _exact_distance(gm, gain, bias)
    assume(distance > 0.0)
    note(f"rho={gain} c={bias} distance={distance} {d}")
    assert d["error_estimate"] >= distance, (
        f"reported {d['error_estimate']:.4e} for a true distance of "
        f"{distance:.4e} ({distance / d['error_estimate']:.0f}x)"
    )
