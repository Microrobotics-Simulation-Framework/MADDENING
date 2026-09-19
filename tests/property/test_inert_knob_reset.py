"""Flipping a gate on a drawn recipe leaves no knob the group ignores.

``CouplingGroup`` warns when a knob is set that the chosen
configuration never reads -- that property is stated in
``test_coupling_group_inert_knobs.py`` -- and ``filterwarnings =
["error"]`` makes the warning fatal.  ``tests/property/strategies.py``
answers it at *draw* time: ``linear_solver`` is only varied under
``solver="ift"``, ``accelerated_fields`` only under the quasi-Newton
methods, and so on, so every recipe it hands out is quiet.

A test that then overrides one of those gates re-opens the question, and
two of them did.  ``test_the_bound_is_the_same_on_both_solvers`` flips
``solver`` to ``"fori"`` over a recipe drawn with
``linear_solver="dense"``; ``_retune`` in
``test_coupling_acceleration_agreement.py`` overrides ``acceleration``
to ``"aitken"`` over a recipe carrying ``accelerated_fields``.  Neither
is a defect in the library -- the warning is right both times -- and
neither is fixed by silencing it: the recipe is simply still asking for
something it no longer means.

:func:`~tests.property.strategies.without_inert_knobs` is the shared
answer, and this module is what keeps it honest.  The helper is driven
off ``_INERT_RULES``, the library's own table, so that a gate added
there is respected at every override site without a second edit.  The
price of that is the failure mode this module is mostly about: a field
renamed on either side makes a rule match nothing, the helper skips it
in silence, and a property test goes on passing for the wrong reason
until Hypothesis happens to draw the combination again.
"""

from __future__ import annotations

import dataclasses
import warnings

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.coupling.group import (
    _FIELD_DEFAULTS,
    _INERT_RULES,
    CouplingGroup,
)
from tests.conftest import EXAMPLES_STANDARD
from tests.property.strategies import (
    COUPLING_ENUM_OPTIONS,
    CouplingGroupRecipe,
    EdgeRecipe,
    GraphRecipe,
    NodeRecipe,
    graph_recipes,
    without_inert_knobs,
)

#: Every field with a domain small enough to flip exhaustively, and its
#: values.  A superset of the real gates on purpose: the four that gate
#: something today are ``convergence_norm``, ``acceleration``,
#: ``subcycling`` and ``solver``, and the rest are here so that a rule
#: newly gated on, say, ``predictor`` is covered the day it lands rather
#: than the day somebody remembers this list.
_FLIPS: dict[str, tuple] = {
    **COUPLING_ENUM_OPTIONS,
    "subcycling": (False, True),
    "diagnostics": (False, True),
    "strict_convergence": (False, True),
}

#: Field names some rule governs -- the only ones the helper may touch.
_GOVERNED = frozenset(name for rule in _INERT_RULES for name in rule.fields)

_RECIPES = graph_recipes(
    min_nodes=2, max_nodes=3,
    require_coupling_group=True,
    allow_mappings=False,
)


def _emitted(group: CouplingGroupRecipe) -> list[str]:
    """The inert-knob warnings constructing *group* raises, if any.

    The :class:`CouplingGroup` is built directly rather than through
    :meth:`GraphRecipe.build`, because that is where the warning lives
    (``CouplingGroup.__post_init__``) and because a graph build is a JAX
    compile -- a cost this module would pay some thousands of times over
    for an answer it can have exactly.  The two regression cases at the
    bottom do go through a real graph.

    ``solver="fori"`` raises a ``DeprecationWarning`` of its own, which
    is about the solver rather than about a dead knob.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CouplingGroup(nodes=frozenset(group.nodes), **group.kwargs)
    return [
        str(w.message) for w in caught
        if issubclass(w.category, UserWarning)
        and not issubclass(w.category, DeprecationWarning)
    ]


# ---------------------------------------------------------------------------
# The table and the recipe have to agree on names
# ---------------------------------------------------------------------------

def test_every_field_the_inert_table_governs_exists_on_the_recipe():
    """A rename on either side must fail here, not go quiet.

    ``without_inert_knobs`` looks each governed field up on the recipe
    by the name ``_INERT_RULES`` gives it.  Let the two drift apart and
    the helper stops resetting that field, with nothing to say so until
    a property test fails on a knob it never meant to set.
    """
    declared = {f.name for f in dataclasses.fields(CouplingGroupRecipe)}
    missing = sorted(_GOVERNED - declared)
    assert not missing, (
        f"_INERT_RULES governs {missing}, which CouplingGroupRecipe does "
        "not declare.  Either the field was renamed on CouplingGroup and "
        "not here or here and not there; until they match, "
        "without_inert_knobs skips it in silence."
    )


def test_every_gate_the_inert_table_reads_resolves_on_the_recipe():
    """The ``live`` predicates have to apply to a recipe, duck-typed.

    They are written against :class:`CouplingGroup` and read their gate
    by attribute name; the helper hands them a
    :class:`CouplingGroupRecipe` instead.  That works only while the two
    classes share the *gate* names -- which the test above does not
    cover, since a gate need not be a governed field -- and an
    ``AttributeError`` from inside a strategy is a much worse way to
    find out.
    """
    recipe = CouplingGroupRecipe(nodes=("a", "b"))
    for rule in _INERT_RULES:
        verdict = rule.live(recipe)
        assert isinstance(verdict, bool), (
            f"the rule governing {rule.fields} returned {verdict!r} for a "
            "recipe rather than a verdict"
        )


def test_every_governed_field_defaults_the_same_on_both_classes():
    """The helper resets to ``CouplingGroup``'s default, not the recipe's.

    They are meant to be the same value -- ``test_round_trips.py``
    checks the two classes name the same fields, not that they default
    them alike -- and the reset is a no-op round trip only while they
    are.  A recipe that defaulted a governed field differently would
    come back from the helper holding a value no draw ever chose.
    """
    declared = {f.name: f for f in dataclasses.fields(CouplingGroupRecipe)}
    for name in sorted(_GOVERNED):
        assert declared[name].default == _FIELD_DEFAULTS[name], (
            f"{name} defaults to {declared[name].default!r} on the recipe "
            f"and {_FIELD_DEFAULTS[name]!r} on CouplingGroup"
        )


# ---------------------------------------------------------------------------
# The property
# ---------------------------------------------------------------------------

@settings(max_examples=EXAMPLES_STANDARD, deadline=None)
@given(
    recipe=_RECIPES,
    combination=st.fixed_dictionaries(
        {name: st.sampled_from(values) for name, values in _FLIPS.items()}
    ),
)
def test_a_reset_recipe_is_quiet_whatever_gate_was_flipped(recipe, combination):
    """The guarantee both override sites rely on.

    Every single-field flip is tried exhaustively -- the space is small
    enough that drawing from it would only make the coverage
    probabilistic, which is the property this module exists to stop
    depending on -- and one simultaneous assignment of all of them is
    drawn on top, because a rule gated on two fields would be satisfied
    by every single flip and broken by the pair.
    """
    for group in recipe.coupling_groups:
        flips = [{name: value}
                 for name, values in _FLIPS.items() for value in values]
        flips.append(combination)
        for flip in flips:
            reset = without_inert_knobs(dataclasses.replace(group, **flip))
            emitted = _emitted(reset)
            assert not emitted, (
                f"after {flip} and a reset the group still names a knob "
                f"its configuration ignores: {emitted}"
            )


@settings(max_examples=EXAMPLES_STANDARD, deadline=None)
@given(recipe=_RECIPES)
def test_a_reset_moves_nothing_the_configuration_actually_reads(recipe):
    """The reset is minimal: only governed, only if inert, only to default.

    A helper that quietly normalised a *live* knob would make the two
    property tests agree by having them measure something other than the
    recipe they were handed.  That is a worse failure than the warning
    it was written to stop, because no warning would report it.
    """
    for group in recipe.coupling_groups:
        for name, values in _FLIPS.items():
            for value in values:
                flipped = dataclasses.replace(group, **{name: value})
                reset = without_inert_knobs(flipped)
                for field in dataclasses.fields(CouplingGroupRecipe):
                    before = getattr(flipped, field.name)
                    after = getattr(reset, field.name)
                    if before == after:
                        continue
                    assert field.name in _GOVERNED, (
                        f"the helper changed {field.name}, which no rule "
                        f"governs, from {before!r} to {after!r}"
                    )
                    assert after == _FIELD_DEFAULTS[field.name], (
                        f"the helper set {field.name} to {after!r} rather "
                        f"than to its default "
                        f"{_FIELD_DEFAULTS[field.name]!r}"
                    )
                    # ``all``, not ``any``: a field can be governed by
                    # more than one rule -- ``linear_solver`` is dead
                    # both under ``solver="fori"`` and at
                    # ``max_iterations=1`` -- and it is read only where
                    # every rule that governs it says so.  That is the
                    # same conjunction ``without_inert_knobs`` applies
                    # (it resets on the first rule that says inert), so
                    # testing it with ``any`` would call a correct reset
                    # a violation the moment a second rule landed on a
                    # field.
                    assert not all(
                        rule.live(reset) for rule in _INERT_RULES
                        if field.name in rule.fields
                    ), f"the helper reset {field.name}, which is live here"


# ---------------------------------------------------------------------------
# The two counterexamples CI found, pinned
# ---------------------------------------------------------------------------

#: The recipe Hypothesis reported against ``_retune``: an accelerated
#: field set that only the two IQN methods read, under Aitken.
_STRANDED_BY_ACCELERATION = CouplingGroupRecipe(
    nodes=("node_1", "rod"),
    acceleration="aitken",
    accelerated_fields=(("node_1", ("position",)), ("rod", ("position",))),
    jacobian_reuse=3,
    relaxation=0.5,
    diagnostics=True,
)

#: The recipe Hypothesis reported against
#: ``test_the_bound_is_the_same_on_both_solvers``: a tangent-system
#: choice and a gradient guard that only ``solver="ift"`` reads, under
#: ``"fori"``.  A latent defect on ``release/0.4.0`` rather than one
#: this branch introduced -- the flip has always stranded the knob, and
#: it took the validator plus a lucky draw to say so.
_STRANDED_BY_SOLVER = CouplingGroupRecipe(
    nodes=("node_1", "rod"),
    solver="fori",
    linear_solver="dense",
    strict_convergence=True,
    diagnostics=True,
)


@pytest.mark.parametrize(
    "group,stranded",
    [
        pytest.param(
            _STRANDED_BY_ACCELERATION,
            ("accelerated_fields", "jacobian_reuse", "relaxation"),
            id="acceleration",
        ),
        pytest.param(
            _STRANDED_BY_SOLVER,
            ("linear_solver", "strict_convergence"),
            id="solver",
        ),
    ],
)
def test_the_counterexamples_ci_found_warn_before_the_reset(group, stranded):
    """Each pinned recipe really is a warning, so the fix is load-bearing.

    Without this the regression below would go on passing if the
    validator were weakened or dropped, and nobody would notice that the
    property tests had stopped covering the case.
    """
    emitted = _emitted(group)
    assert emitted, f"{stranded} no longer warns when set inertly"
    named = " ".join(emitted)
    for field in stranded:
        assert field in named, (
            f"{field} is inert here but no warning names it: {named}"
        )


def _spring(name: str, position: float) -> NodeRecipe:
    return NodeRecipe(
        type_name="SpringDamperNode", name=name, timestep=0.01,
        kwargs=(
            ("stiffness", 10.0), ("damping", 0.5), ("mass", 1.0),
            ("rest_length", 1.0), ("initial_position", position),
            ("initial_velocity", 0.0),
        ),
    )


def _two_spring_graph(group: CouplingGroupRecipe) -> GraphRecipe:
    """The shape both counterexamples were found on: ``rod`` and
    ``node_1`` exchanging position, which is a cycle and so a group."""
    return GraphRecipe(
        nodes=(_spring("rod", 0.5), _spring("node_1", -0.25)),
        edges=(
            EdgeRecipe("rod", "node_1", "position", "anchor_position",
                       units="m"),
            EdgeRecipe("node_1", "rod", "position", "anchor_position",
                       units="m"),
        ),
        coupling_groups=(group,),
    )


@pytest.mark.parametrize(
    "group",
    [
        pytest.param(_STRANDED_BY_ACCELERATION, id="acceleration"),
        pytest.param(_STRANDED_BY_SOLVER, id="solver"),
    ],
)
def test_the_counterexamples_ci_found_build_after_the_reset(group):
    """Both CI failures, end to end on a real graph.

    :func:`_emitted` is enough to state the property; this pays for a
    compile once per case so that the fix is also pinned at the level
    the two property tests fail at, ``GraphRecipe.build()``.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm = _two_spring_graph(without_inert_knobs(group)).build()
        gm.step()
    inert = [
        str(w.message) for w in caught
        if issubclass(w.category, UserWarning)
        and not issubclass(w.category, DeprecationWarning)
    ]
    assert not inert, inert
