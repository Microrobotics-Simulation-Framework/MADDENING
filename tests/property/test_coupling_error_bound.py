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

The slow companion
------------------
The generated groups alone could not fail on the defect this property is
named for.  Library nodes coupled at their drawn settings contract
fast -- a residual and the distance it leaves differ by ``rho / (1 -
rho)``, which is inside ``_SLACK`` for every ``rho`` below 0.8 -- so a
seeded fault that put the raw residual test back (the criterion reading
``residual`` where it reads ``residual / (1 - rho)``) passed this
property under both the ``dev`` and ``ci`` profiles.  Every drawn graph
therefore also carries a *slow companion*: a scalar relay cycle in a
group of its own, contracting at a drawn ``rho`` in ``[0.9, 0.95]``,
started a drawn distance short of its fixed point, under a drawn norm,
solver and step scale.  Its fixed point is arithmetic, so it is measured
against that rather than against a tighter solve.  At those rates the raw
residual test stops 10-20x its threshold from the fixed point, and the
property fails on the first draw.
"""

from __future__ import annotations

import dataclasses
import functools

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
from maddening.core.node import BoundaryInputSpec, SimulationNode

from tests.conftest import EXAMPLES_COSTLY
from tests.core.test_coupling_solver_equivalence import (
    residual_noise_floor,
)
from tests.property.strategies import (
    CouplingGroupRecipe,
    EdgeRecipe,
    NodeRecipe,
    graph_recipes,
    without_inert_knobs,
)

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


@st.composite
def _measurable_norm_recipes(draw):
    """``_RECIPES`` with no group left on the ``"interface"`` norm.

    The distance this property measures is a distance *in the norm the
    group's threshold is quoted in*, and ``"interface"`` measures only the
    fields that cross an edge -- so a group on it reports ``converged``
    about a subset of the state and the comparison below is not the one the
    threshold promises.  The property therefore does not cover that norm.

    It used to say so with ``assume``: ``convergence_norm`` is drawn
    uniformly from three values, so a third of every draw built a graph,
    stepped it, and threw the result away.  Measured at 42.9% rejection over
    all three of this test's gates -- the worst in ``tests/property/``.
    Remapping the norm at draw time costs nothing and rejects nothing, and
    the remapped draw gets a freshly drawn tolerance for the knob its new
    norm actually reads, so the search is no narrower than it was.

    ``without_inert_knobs`` closes the question the flip re-opens: the knob
    the old norm made live keeps its drawn value otherwise, and
    ``CouplingGroup`` warns about a knob its configuration never reads --
    fatally, under ``filterwarnings = ["error"]``.
    """
    recipe = draw(_RECIPES)
    groups = []
    for g in recipe.coupling_groups:
        if g.convergence_norm == "interface":
            norm = draw(st.sampled_from(("l2", "mixed")))
            live = ({"tolerance": draw(st.sampled_from([1e-8, 1e-6, 1e-3]))}
                    if norm == "l2"
                    else {"rtol": draw(st.sampled_from([1e-6, 1e-3]))})
            g = without_inert_knobs(
                dataclasses.replace(g, convergence_norm=norm, **live))
        groups.append(g)
    return dataclasses.replace(recipe, coupling_groups=tuple(groups))


class _SlowRelay(SimulationNode):
    """``x <- gain * u + bias``, a scalar started at ``x0``.

    Two of these in a cycle, one of them the identity (``gain=1``,
    ``bias=0``), are the slow companion: a Gauss-Seidel pass advances the
    other by ``x -> gain * x + bias``, so the group contracts at exactly
    ``gain`` and its fixed point is ``bias / (1 - gain)`` on both nodes.
    """

    def __init__(self, name, timestep, gain, bias, x0):
        super().__init__(name=name, timestep=timestep)
        self._gain = float(gain)
        self._bias = float(bias)
        self._x0 = float(x0)

    def initial_state(self):
        return {"x": jnp.asarray(self._x0, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0),
                                       description="u")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": jnp.float32(self._gain) * boundary_inputs["u"]
                + jnp.float32(self._bias)}

    def update_evaluations(self):
        return 1


@dataclasses.dataclass(frozen=True)
class _SlowRelayRecipe(NodeRecipe):
    """A :class:`NodeRecipe` for :class:`_SlowRelay`.

    ``GraphRecipe.build`` calls ``node.build()`` on every node, so a
    companion rides along in the recipe -- and so in every graph built
    from it -- without a registry entry of its own.
    """

    def build(self):
        return _SlowRelay(self.name, self.timestep, **self.params)


#: The companion's node names; nothing in ``NODE_NAME_POOL`` collides.
_COMPANION = ("slow_companion", "slow_companion_relay")
_COMPANION_KEY = "+".join(sorted(_COMPANION))

#: Where the companion starts, as a multiple of its threshold short of
#: its fixed point.  Far enough that the raw residual test would stop
#: well outside ``_SLACK`` (it stops once the distance is below
#: ``threshold / (omega * (1 - rho))``, 10-20x the threshold here), near
#: enough that the error-bound criterion arrives inside the cap.
_COMPANION_HEAD_START = 30.0

#: Passes the companion may spend.  The slowest draw contracts at
#: ``1 - 0.8 * (1 - 0.95) = 0.96`` per pass and has to cover a factor
#: of ``_COMPANION_HEAD_START``: 84 passes.
_COMPANION_CAP = 150

#: The bias; the fixed point is ``bias / (1 - rho)``, O(10).
_COMPANION_BIAS = 1.0


def _companion_fixed_point(rate: float) -> float:
    """``bias / (1 - rate)`` of the float32 map the companion evaluates."""
    g = float(np.float32(rate))
    return float(np.float32(_COMPANION_BIAS)) / (1.0 - g)


@st.composite
def _slow_companion(draw, timestep: float):
    """The companion's nodes, edges and group, drawn.

    Only ``"none"`` and ``"fixed"`` are drawn.  They are the two
    accelerations whose step scale the estimate corrects (it is constant
    there, ``relaxation_step_scale``); Aitken's clipped factor and IQN's
    quasi-Newton step are documented as uncorrected (``MADD-ANO-005``'s
    residual risk), and the generated groups draw them anyway.  The
    relaxation stays in ``[0.8, 1.6]``: at ``rho >= 0.9`` that keeps the
    relaxed rate ``1 - omega * (1 - rho)`` positive, the regime where the
    estimate is tight rather than merely safe, and the pass count under
    the cap.
    """
    rate = draw(st.floats(min_value=0.9, max_value=0.95))
    norm = draw(st.sampled_from(("l2", "mixed")))
    threshold = draw(st.sampled_from((1e-3, 1e-4)))
    acceleration = draw(st.sampled_from(("none", "fixed")))
    relaxation = (draw(st.floats(min_value=0.8, max_value=1.6))
                  if acceleration == "fixed" else 1.0)
    solver = draw(st.sampled_from(("ift", "fori")))
    x_star = _companion_fixed_point(rate)
    x0 = x_star * (1.0 - _COMPANION_HEAD_START * threshold)
    slow, relay = _COMPANION
    nodes = (
        _SlowRelayRecipe("SlowRelay", slow, timestep, (
            ("gain", rate), ("bias", _COMPANION_BIAS), ("x0", x0))),
        _SlowRelayRecipe("SlowRelay", relay, timestep, (
            ("gain", 1.0), ("bias", 0.0), ("x0", x0))),
    )
    edges = (EdgeRecipe(relay, slow, "x", "u"), EdgeRecipe(slow, relay, "x", "u"))
    live = ({"tolerance": threshold} if norm == "l2"
            else {"rtol": threshold})
    group = CouplingGroupRecipe(
        nodes=_COMPANION, max_iterations=_COMPANION_CAP,
        convergence_norm=norm, diagnostics=True, acceleration=acceleration,
        relaxation=relaxation, solver=solver, **live,
    )
    return nodes, edges, group, rate


@st.composite
def _recipes_with_a_slow_companion(draw):
    """``(recipe, rate)``: :func:`_measurable_norm_recipes` plus the companion.

    The companion steps at the graph's base timestep (the recipe's
    timesteps are one base times 1, 2 or 4, so the smallest is the GCD)
    and so is solved on every ``step()``.
    """
    recipe = draw(_measurable_norm_recipes())
    timestep = min(n.timestep for n in recipe.nodes)
    nodes, edges, group, rate = draw(_slow_companion(timestep))
    return dataclasses.replace(
        recipe,
        nodes=recipe.nodes + nodes,
        edges=recipe.edges + edges,
        coupling_groups=recipe.coupling_groups + (group,),
    ), rate


def _check_the_companion(gm, group, diagnostics, rate) -> bool:
    """The invariant on the companion, against its arithmetic fixed point.

    Returns whether it was checked: a companion that ran out of passes
    reported ``converged=False``, which is the honest answer and says
    nothing about this property.
    """
    d = diagnostics[_COMPANION_KEY]
    if not d["converged"]:
        return False
    nodes = sorted(_COMPANION)
    x_star = np.float32(_companion_fixed_point(rate))
    got = {n: dict(gm.get_node_state(n)) for n in nodes}
    want = {n: {"x": jnp.asarray(x_star, jnp.float32)} for n in nodes}
    distance = _distance(group, got, want, nodes)
    threshold = _threshold(group)
    note(f"companion: rho={rate} distance={distance:.3e} "
         f"threshold={threshold:.3e} residual={d['residual']:.3e} "
         f"estimate={d['error_estimate']:.3e} norm={group.convergence_norm} "
         f"accel={group.acceleration} omega={group.relaxation} "
         f"solver={group.solver} iterations={d['iterations']}")
    assert distance <= threshold * _SLACK, (
        f"the slow companion (rho={rate}) reported converged at threshold "
        f"{threshold} but is {distance} from its fixed point"
    )
    assert d["error_estimate"] * _SLACK >= distance, (
        f"the companion's error estimate {d['error_estimate']} does not "
        f"bound its distance {distance} from the fixed point"
    )
    return True


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(case=_recipes_with_a_slow_companion())
def test_converged_implies_the_state_is_within_tolerance_of_the_fixed_point(
    case,
):
    """The invariant, measured against an independently converged answer.

    Before 0.4.0 ``converged=True`` meant ``||F(x) - x|| <= tol``: a
    statement about the last step, which is short of the distance to the
    fixed point by ``1/(1 - rho)``.  A group whose slowest mode has
    ``rho`` near 1 could therefore report success arbitrarily far from
    the answer, and nothing in the library said so.  That is
    ``MADD-ANO-005``, and this is the assertion that retires it.

    Every generated group that converged is measured against the same
    recipe re-solved 1000x tighter, and the slow companion against its
    arithmetic fixed point (module docstring).  The companion is what
    makes the property able to fail on the defect: without it a raw
    residual criterion passed.

    Rejected draws
    --------------
    The property rejects an example only when nothing was checked: the
    companion ran out of passes *and* no generated group converged
    against a converged reference.  The ``"interface"`` norm is excluded by the
    strategy rather than assumed away (see
    :func:`_measurable_norm_recipes`); whether a drawn group reaches its
    threshold in the passes it was given is an outcome of the solve, not a
    shape of the input, so it cannot be generated.
    """
    recipe, rate = case
    recipe = _diagnostics_recipe(recipe)
    # The companion reports under ``"ift"`` without diagnostics; turning
    # them on there compiles the spectral and gradient bounds for a group
    # whose verdict is all this reads, which doubled the property's time.
    # ``"fori"`` reports only with them.
    recipe = dataclasses.replace(recipe, coupling_groups=tuple(
        dataclasses.replace(g, diagnostics=g.solver == "fori")
        if g.nodes == _COMPANION else g
        for g in recipe.coupling_groups))
    gm = recipe.build()
    gm.step()
    diagnostics = gm.coupling_diagnostics()

    groups = {"+".join(sorted(g.nodes)): g for g in gm._coupling_groups}  # noqa: SLF001
    assert all(g.convergence_norm != "interface" for g in groups.values()), (
        "_measurable_norm_recipes must leave no group on the interface norm"
    )
    checked = int(_check_the_companion(
        gm, groups[_COMPANION_KEY], diagnostics, rate))

    converged = {key: groups[key] for key, d in diagnostics.items()
                 if d["converged"] and key != _COMPANION_KEY}
    if not converged:
        assume(checked)
        return

    # The reference re-solves the generated groups only: the companion
    # is measured against arithmetic, and the tightened criterion would
    # spend its whole cap on it for nothing.
    generated = dataclasses.replace(
        recipe,
        nodes=tuple(n for n in recipe.nodes if n.name not in _COMPANION),
        edges=tuple(e for e in recipe.edges if e.target not in _COMPANION),
        coupling_groups=tuple(g for g in recipe.coupling_groups
                              if g.nodes != _COMPANION),
    )
    reference = _tightened(generated).build()
    reference.step()
    ref_diagnostics = reference.coupling_diagnostics()

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


# Slow-marked (still run by slow-tests.yml): the recipe *is* the draw, so
# every example builds and compiles a different graph -- 42-48 s on the CI
# runner, and no fixed-shape rewrite applies.  The same algebra is pinned
# on every push by example in ``tests/core/test_coupling_error_bound.py``.
@pytest.mark.slow
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


# Slow-marked (still run by slow-tests.yml): two graphs built and compiled
# per drawn recipe, 63-73 s on the CI runner.  Solver parity is pinned on
# every push by ``tests/core/test_coupling_solver_equivalence.py``, including
# this property's own shrunk counterexample.
@pytest.mark.slow
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


class _Affine(SimulationNode):
    """``x <- gain * u + bias`` on ``n`` independent modes.

    ``gain`` and ``bias`` are node parameters, so they reach the compiled
    step as traced arguments: one compiled graph serves every drawn
    spectrum of a given size (see :func:`_compiled_affine_cycle`).
    """

    def __init__(self, name, gain, bias):
        gain = jnp.asarray(gain, jnp.float32)
        super().__init__(name=name, timestep=1.0, gain=gain,
                         bias=jnp.asarray(bias, jnp.float32))
        self._shape = jnp.shape(gain)

    def initial_state(self):
        return {"x": jnp.zeros(self._shape, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=self._shape, dtype=jnp.float32,
                                       description="u")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else params
        return {"x": p["gain"] * boundary_inputs["u"] + p["bias"]}

    def update_evaluations(self):
        # One evaluation, declared: the tests below assert the bound and
        # its usable flag at float32 convergence, which an undeclared
        # node's group does not get at the floor.
        return 1


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


@functools.lru_cache(maxsize=None)
def _compiled_affine_cycle(n_modes, **group_kw):
    """One compiled ``_affine_cycle`` per mode count and group config.

    The two-mode properties below draw ``gain`` and ``bias`` and nothing
    that is static in the step, so building and compiling a graph per
    example spent almost all of each example compiling the same program:
    20 of them took 30 s on the CI runner.  The draws are passed to this
    one graph as its nodes' parameters instead (:func:`_step_affine`),
    which is the same compiled program a fresh graph would build --
    ``_Affine`` reads both from ``params`` either way.
    """
    ones = np.ones(n_modes, np.float32)
    return _affine_cycle(ones * 0.5, ones, **group_kw)


def _step_affine(gm, gain, bias, x0=None):
    """One step of *gm* from its initial state, with ``a``'s constants set.

    ``reset_state`` puts the state *and* every coupling seed back to what
    ``compile()`` left (``test_reset_state_restores_the_meta_compile_seeds``
    in ``tests/core/test_coupling_error_bound.py`` pins that), so the step
    is the one a freshly built graph would take.  *x0*, when given, is
    written into both nodes' ``x`` first -- a start other than zero, with
    the coupling seeds still fresh.  Returns the group's diagnostics.
    """
    gm.reset_state()
    if x0 is not None:
        start = jnp.asarray(x0, jnp.float32)
        gm.set_node_state("a", {"x": start})
        gm.set_node_state("b", {"x": start})
    gm.step(params={"nodes": {"a": {
        "gain": jnp.asarray(gain, jnp.float32),
        "bias": jnp.asarray(bias, jnp.float32),
    }}})
    return gm.coupling_diagnostics()["a+b"]


def _exact_distance(gm, gain, bias):
    """Distance to the analytic fixed point, in the group's own L2 norm.

    The fixed point of the map the graph *evaluates*: ``_Affine`` holds
    its gain and bias in float32, so they are rounded to float32 first
    and the fixed point is then taken in float64.  Taking it from the
    unrounded draws measured the parameters' rounding as well -- at a
    gain of 0.9999 a one-ulp change in the gain moves the fixed point by
    6e-4 of itself, which is the same order as the bound being checked.
    """
    gains = np.atleast_1d(np.asarray(gain, np.float32)).astype(np.float64)
    biases = np.atleast_1d(np.asarray(bias, np.float32)).astype(np.float64)
    exact = [float(b / (1.0 - g)) for g, b in zip(gains, biases)]
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

#: The group the two-mode properties solve: one config, so one compile.
_TWO_MODE_GROUP = dict(max_iterations=60, tolerance=_ANALYTIC_TOLERANCE)


# Slow-marked (still run by slow-tests.yml): ``relaxation`` is a static
# knob of the compiled step, so a drawn ``omega`` is a compile per example
# (26 s on the CI runner) and no shared graph can take it as an argument.
# ``test_the_estimate_is_invariant_to_the_relaxation_factor`` in
# ``tests/core/test_coupling_error_bound.py`` pins the same statement at
# fixed factors on every push.
@pytest.mark.slow
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
    "ratio_usable=True.  The spectrum is the fix and it is in the tree "
    "as a separate key, `spectral_error_bound` (see the property just "
    "below, which holds on the same draws); `error_estimate` keeps its "
    "value because 438 recorded verdicts depend on it.  Flipping this to "
    "a pass therefore means `error_estimate`'s own value changed: update "
    "benchmarks/results/audit_040_final/ERROR_BOUND_DECISION.md, the "
    "caveat on _fixed_point_while, and the recorded 122x in "
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
    gm = _compiled_affine_cycle(2, **_TWO_MODE_GROUP)
    d = _step_affine(gm, gain, bias)
    assume(d["converged"] and d["ratio_usable"])
    distance = _exact_distance(gm, gain, bias)
    assume(distance > 0.0)
    note(f"rho={gain} c={bias} distance={distance} {d}")
    assert d["error_estimate"] >= distance, (
        f"reported {d['error_estimate']:.4e} for a true distance of "
        f"{distance:.4e} ({distance / d['error_estimate']:.0f}x)"
    )


#: How far short of its fixed point the state may start, relatively.
#: ``None`` starts it at zero, as the strict xfail does.  The rest reach
#: the float32 floor: a mode contracting at ``rho`` stalls -- ``F(x) ==
#: x`` bitwise -- once ``(1 - rho) * |x - x*|`` is under half an ulp,
#: which at ``rho >= 0.99`` is a relative distance under ``6e-8 / (1 -
#: rho)``: every slow mode at 1e-6, the slowest at 1e-5 and 1e-4.
_HEAD_STARTS = (None, 1e-6, 1e-5, 1e-4)

#: The group a head start is solved in: a threshold no float32 residual
#: above zero meets, so the loop runs until the fast mode is bitwise
#: stationary too, and the residual left is the rounding the key's floor
#: is for.  At the analytic group's 1e-3 the fast mode's last change is
#: what stops the loop, and multiplied by the slow mode's amplification
#: it covers the stalled distance by itself, floor or no floor.
_STALL_GROUP = dict(max_iterations=60, tolerance=1e-12)


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(
    rho_slow=st.floats(min_value=0.99, max_value=0.9999),
    rho_fast=st.floats(min_value=0.0, max_value=0.5),
    c_slow=st.floats(min_value=1e-6, max_value=1e-3),
    head_start=st.sampled_from(_HEAD_STARTS),
)
@example(rho_slow=0.999, rho_fast=0.2, c_slow=1e-5, head_start=None)
@example(rho_slow=0.9999, rho_fast=0.2, c_slow=1e-3, head_start=1e-6)
def test_the_spectral_bound_is_never_smaller_than_the_distance_it_bounds(
    rho_slow, rho_fast, c_slow, head_start,
):
    """The strict xfail's draws, held by the spectral key -- and a stall.

    With ``head_start=None`` nothing about the two-mode map changes
    between the two tests -- same generator, same criterion, same
    returned state.  What changes is where ``rho`` comes from: the
    residual sequence reads the fast mode for as long as it dominates the
    step, the Arnoldi space of ``dF/dx`` contains both modes from the
    first product.  For a linear map the error is ``(A - I)^{-1}`` of the
    residual whatever the iteration did, so the bound holds wherever the
    residual is above its own noise.

    Below its noise is the other half, and those draws alone did not
    reach it: started at zero, the group stops on a criterion three
    decades above float32 resolution, so dropping the float floor from
    the key passed this property.  A head start puts both modes near
    their fixed point instead (:data:`_HEAD_STARTS`) and runs the loop
    until nothing moves (:data:`_STALL_GROUP`): the slow mode stalls a
    measurable distance away with a residual that reads nothing, the
    bound is then the floor times the amplification, and without the
    floor it reads under the distance.  The comparison stays bare -- the
    floor is the key's business, and a margin added here is how a
    stalled iterate reading a bound of ``0.0`` went unnoticed before.
    """
    gain = (rho_slow, rho_fast)
    bias = (c_slow, 1.0)
    if head_start is None:
        gm = _compiled_affine_cycle(2, **_TWO_MODE_GROUP)
        d = _step_affine(gm, gain, bias)
    else:
        gm = _compiled_affine_cycle(2, **_STALL_GROUP)
        gains = np.asarray(gain, np.float32).astype(np.float64)
        biases = np.asarray(bias, np.float32).astype(np.float64)
        x0 = biases / (1.0 - gains) * (1.0 - head_start)
        d = _step_affine(gm, gain, bias, x0=x0)
    distance = _exact_distance(gm, gain, bias)
    note(f"rho={gain} c={bias} head_start={head_start} distance={distance} {d}")
    assert d["spectral_usable"] is True, d
    assert d["rho_spectral"] == pytest.approx(rho_slow, abs=1e-4)
    assert d["spectral_error_bound"] >= distance, (
        f"reported {d['spectral_error_bound']:.4e} for a true distance of "
        f"{distance:.4e} ({distance / d['spectral_error_bound']:.2f}x)"
    )


# ---------------------------------------------------------------------------
# Random normal contractions: the spectral bound holds whatever the spectrum
# ---------------------------------------------------------------------------


class _Linear(SimulationNode):
    """``x <- A u + c`` on a vector field."""

    def __init__(self, name, A, c):
        super().__init__(name=name, timestep=1.0)
        self._A = jnp.asarray(A, jnp.float32)
        self._c = jnp.asarray(c, jnp.float32)

    def initial_state(self):
        return {"x": jnp.zeros(self._c.shape, jnp.float32)}

    def state_fields(self):
        return ["x"]

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=self._c.shape, dtype=jnp.float32,
                                       description="u")}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": self._A @ boundary_inputs["u"] + self._c}

    def update_evaluations(self):
        # One dense evaluation, declared (see ``_Affine``).
        return 1


def _linear_cycle(A, c, **group_kw):
    """``a -> b -> a`` carrying a full matrix; ``b`` is the identity relay.

    The Gauss-Seidel one-pass Jacobian is ``[[0, A], [0, A]]``: rank at
    most ``n``, eigenvalues those of ``A``, so an eight-step Krylov
    space is its whole range for every ``n`` drawn here and the Ritz
    spectrum is exact.  The fixed point is ``(I - A)^{-1} c`` on both
    nodes.
    """
    n = len(c)
    gm = GraphManager()
    gm.add_node(_Linear("a", A, c))
    gm.add_node(_Linear("b", np.eye(n), np.zeros(n)))
    gm.add_edge(source="b", target="a", source_field="x", target_field="u")
    gm.add_edge(source="a", target="b", source_field="x", target_field="u")
    gm.add_coupling_group(["a", "b"], diagnostics=True, **group_kw)
    gm.compile()
    return gm


def _distance_to(gm, x_star):
    """Distance to ``x_star`` on both nodes, in the group's own L2 norm."""
    total = 0.0
    for node in ("a", "b"):
        got = np.asarray(gm.get_node_state(node)["x"], np.float64)
        ref = max(np.max(np.abs(got)), np.max(np.abs(x_star)))
        if ref > 0.0:
            total += float(np.sum(((got - x_star) / ref) ** 2))
    return total ** 0.5


@st.composite
def _normal_contractions(draw):
    """``(A, c, acceleration, relaxation)`` with ``A`` symmetric and ``rho < 1``.

    Eigenvalues are drawn directly, negative ones included, and one of
    them is pushed towards ``+/-1`` on most draws so the slow-mode
    regime is exercised rather than found by luck; a random orthogonal
    basis then makes ``A`` normal with that spectrum.  ``relaxation``
    stays at or below one: above it a negative eigenvalue near ``-1``
    makes the *relaxed* iteration diverge, and a state that has left
    float32 is not a fixture for anything.
    """
    n = draw(st.integers(min_value=2, max_value=6))
    lam = np.asarray(draw(st.lists(
        st.floats(min_value=-0.9, max_value=0.9), min_size=n, max_size=n,
    )))
    edge = draw(st.sampled_from(["none", "slow", "alternating"]))
    if edge == "slow":
        lam[0] = draw(st.floats(min_value=0.9, max_value=0.98))
    elif edge == "alternating":
        lam[0] = -draw(st.floats(min_value=0.9, max_value=0.98))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    A = Q @ np.diag(lam) @ Q.T
    c = rng.uniform(-2.0, 2.0, size=n)
    acceleration = draw(st.sampled_from(["none", "fixed"]))
    relaxation = (draw(st.floats(min_value=0.3, max_value=1.0))
                  if acceleration == "fixed" else 1.0)
    return A, c, acceleration, relaxation


# Slow-marked (still run by slow-tests.yml): the draw varies the mode count
# (a shape) and, under ``"fixed"``, the static relaxation factor, so most
# examples are a compile of their own (31-37 s on the CI runner).  The
# spectral bound is held on every push by the two-mode property above, which
# compiles once, and by the example tests in
# ``tests/core/test_coupling_error_bound.py``, relaxation included.
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(case=_normal_contractions())
def test_the_spectral_bound_holds_on_random_normal_contractions(case):
    """``spectral_error_bound >= ||x - x*||`` for any normal ``A``.

    Under ``"none"`` and ``"fixed"`` at any relaxation, with negative
    eigenvalues, alternation and a spectral radius up to 0.98, and
    whether or not the group met its criterion within the cap -- the
    bound is about the returned iterate, not about convergence.  The
    spectrum is exact here (``n <= 6 < SPECTRAL_KRYLOV_STEPS``), so
    ``spectral_usable`` is asserted True rather than assumed: a False
    would mean the Arnoldi breakdown handling stopped resolving a
    resolvable spectrum.
    """
    A, c, acceleration, relaxation = case
    kw = dict(acceleration=acceleration)
    if acceleration == "fixed":
        kw["relaxation"] = relaxation
    gm = _linear_cycle(A, c, max_iterations=400,
                       tolerance=_ANALYTIC_TOLERANCE, **kw)
    gm.step()
    d = gm.coupling_diagnostics()["a+b"]
    # The fixed point of the float32 map the graph evaluates (see
    # ``_exact_distance``), taken in float64.
    A32 = np.asarray(A, np.float32).astype(np.float64)
    c32 = np.asarray(c, np.float32).astype(np.float64)
    x_star = np.linalg.solve(np.eye(len(c)) - A32, c32)
    distance = _distance_to(gm, x_star)
    rho = float(np.max(np.abs(np.linalg.eigvalsh(A))))
    note(f"rho={rho} {acceleration} omega={relaxation} distance={distance} {d}")
    assert d["spectral_usable"] is True, d
    assert d["rho_spectral"] == pytest.approx(rho, abs=1e-4), (
        f"rho_spectral={d['rho_spectral']} for a spectral radius of {rho}"
    )
    # Bare: the key carries its own float resolution (see the two-mode
    # property above).
    assert d["spectral_error_bound"] >= distance, (
        f"{acceleration} omega={relaxation}: bound {d['spectral_error_bound']:.4e} "
        f"below the true distance {distance:.4e} at rho={rho:.4f}"
    )
