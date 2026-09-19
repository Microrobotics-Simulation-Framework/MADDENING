"""An accelerator may not move a coupling group to a different answer.

``benchmarks/coupling_fixtures.py`` carries ten rows — all ``iqn-*``
under ``convergence_norm="interface"`` — where it does, listed in
``tests/core/test_coupling_fixture_invariants.py``'s
``_KNOWN_DISAGREEMENTS``.  They are ten hand-picked configurations of
three hand-built graphs, and the measurement that explains them
(``benchmarks/results/retire_known_disagreements/REPORT.md``) is about
*which fields the accelerator was pointed at*, not about the fixtures:

    the ``"interface"`` criterion is taken over the edge source fields,
    ``accelerated_fields=None`` auto-detects the **same** set, so the
    quasi-Newton step lands on the interface and every other state
    field is left at whatever the raw pass produced — where nothing
    measures it.  ``position`` moves by at most 1.0e-04 while
    ``velocity`` moves by up to 2.2.

This module states the other half of that sentence over generated
graphs: point the accelerator at the *whole* state and the answer stops
depending on which accelerator ran.  On the ten fixture rows that
single change closes every one of them at no iteration cost, which is
the evidence the fixtures give; what they cannot give is the assurance
that it holds for a graph nobody built by hand.

Stated under ``convergence_norm="l2"``, which is the half of the
sentence that generalises.  Trying it under ``"interface"`` as well
finds a counterexample in a few hundred draws — 5.2e-01 on a
three-spring Jacobi group under Aitken, both solves reporting
``converged`` and the accelerated one reporting ``bound_valid=False``
with ``residual`` exactly zero — and the counterexample is the same
defect one layer down.  Widening ``accelerated_fields`` stops the
accelerator from leaving fields behind, but the *criterion* is still
over the edge fields alone, so a group whose interface goes stationary
while the rest of its state has not can still exit.  On the spring
fixtures that cannot happen — their ``position`` and ``velocity`` are
one integration apart, so a converged interface is a converged state,
which is why all ten rows close there.  On an arbitrary graph it can.
The guarantee belongs to the norm, not to the accelerated set; naming
every field is necessary and, under ``"interface"``, not sufficient.

Why this is not the same property as
``tests/property/test_coupling_error_bound.py``.  That module asks
whether one solve's ``converged=True`` bounds its own distance to its
own fixed point.  This one asks whether *two* solves of the same graph
agree, which is a comparison between two returned states and needs no
norm at all.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from hypothesis import assume, given, note, settings
from hypothesis import strategies as st

from maddening.core.coupling.group import _FIELD_DEFAULTS

from tests.conftest import EXAMPLES_COSTLY
from tests.property.strategies import (
    NODE_KINDS,
    graph_recipes,
    without_inert_knobs,
)

#: Accelerations compared against plain fixed-point iteration.  Every
#: one of the ten fixture rows is ``iqn-ils`` or ``iqn-imvj``; Aitken is
#: in because it is the other accelerator that rewrites the iterate, and
#: it held an entry of its own until the corrected exit criterion
#: cleared it.
_ACCELERATIONS = ("aitken", "iqn-ils", "iqn-imvj")

#: Steps per comparison.  Two solves of a *driven* graph that disagree
#: at all disagree more with every step, so a handful is enough to tell
#: a different answer from round-off — the fixture lane needs 25 only
#: because it is looking for the smallest such gap it can still resolve.
#: Each step is a compiled call on both graphs, so this is the knob that
#: decides what the module costs.
_STEPS = 4

#: How far apart the two answers may be, relative to each field's own
#: scale.  Measured rather than guessed: over 400 examples at the ``ci``
#: profile's depth, 387 were bit-identical and the worst of the rest was
#: 8.6e-07, so this is ~100x the observed worst and still three decades
#: under the smallest defect it has to catch (the fixture rows run 0.16
#: to 2.25).  It is not float32 that sets the floor -- the two solves
#: genuinely stop at different iterates of the same sequence -- so the
#: headroom is for a graph shape the 400 did not draw.
_AGREEMENT = 1e-4

#: The criterion both solves are held to: ``CouplingGroup``'s own
#: defaults for all three knobs, paired with a cap that can afford
#: them.  Tighter than the fixtures' ``rtol=1e-4``, so what is left
#: between the two answers is the accelerator's choice of iterate
#: rather than the tolerance.
#:
#: All three are set, not just the live one, and set to the *defaults*
#: -- because the group warns about a knob its norm does not read and
#: ``filterwarnings = ["error"]`` makes that fatal, so a drawn recipe
#: carrying a non-default ``tolerance`` fails to build under the
#: interface norm rather than being measured.
_TOLERANCE = _FIELD_DEFAULTS["tolerance"]
_ATOL = _FIELD_DEFAULTS["atol"]
_RTOL = _FIELD_DEFAULTS["rtol"]
_CAP = 60

#: Node kinds drawn.  ``BallNode`` is excluded: its update branches on
#: contact, so ``F`` is piecewise and two iterates on opposite sides of
#: the switch are not two approximations of one fixed point.  An
#: accelerator that extrapolates across the branch lands somewhere
#: plain iteration does not -- measured at 4.0e-01 under
#: ``aitken``/``interface`` on a two-ball graph -- and that is a
#: statement about the graph, not about the accelerator.  Everything
#: left is smooth.
_SMOOTH_KINDS = tuple(k for k in NODE_KINDS if k != "BallNode")


def _float_fields(gm, nodes):
    """``{node: (field, ...)}`` over the float leaves of *nodes*.

    Integer leaves are excluded because the fixed-point vector excludes
    them: ``ift`` holds them out of its state vector entirely, so naming
    one in ``accelerated_fields`` would ask the least-squares problem to
    extrapolate a counter.
    """
    out = {}
    for name in nodes:
        fields = tuple(
            field for field, value in sorted(gm.get_node_state(name).items())
            if np.issubdtype(np.asarray(value).dtype, np.floating)
        )
        if fields:
            out[name] = fields
    return out


def _spread(a, b, nodes):
    """Worst deviation between two states over *nodes*, per field scale.

    The measure ``_assert_same_fixed_point`` uses, for the same reason:
    a single global scale is a velocity scale on these graphs and hides
    a disagreement about position, while normalising each element
    against itself explodes wherever a trajectory passes through zero.
    """
    scales = {}
    for (_node, field), arr in a.items():
        m = float(np.max(np.abs(arr))) if arr.size else 0.0
        scales[field] = max(scales.get(field, 0.0), m)
    worst = 0.0
    for key, ref in a.items():
        scale = max(scales[key[1]], 1e-12)
        worst = max(worst, float(np.max(np.abs(ref - b[key]))) / scale)
    return worst


def _state(gm, nodes):
    return {
        (name, field): np.asarray(value, dtype=np.float64).ravel()
        for name in sorted(nodes)
        for field, value in sorted(gm.get_node_state(name).items())
        if np.issubdtype(np.asarray(value).dtype, np.floating)
    }


def _run(gm, group_keys):
    """*_STEPS* steps; ``None`` unless every group converged every time."""
    for _ in range(_STEPS):
        gm.step()
        diagnostics = gm.coupling_diagnostics()
        if not all(diagnostics[key]["converged"] for key in group_keys):
            return None
    return diagnostics


#: The norm this property is stated under.  See the module docstring
#: for why it is not drawn: ``"interface"`` measures the edge fields
#: alone, and its guarantee is conditional on the interface controlling
#: the rest of the state -- true of the coupling fixtures, not of an
#: arbitrary graph.
_NORM = "l2"


def _retune(recipe, *, acceleration, accelerated_fields=None,
            jacobian_reuse=0):
    """*recipe* with every group on one criterion and one accelerator.

    ``acceleration`` is a gate: ``accelerated_fields`` is read by the
    two IQN methods alone, ``jacobian_reuse`` by ``iqn-imvj`` alone and
    ``relaxation`` (which the draw may have set) by ``"fixed"`` alone.
    Overriding the gate here therefore strands up to three knobs, and
    ``CouplingGroup`` warns about each -- fatally, under
    ``filterwarnings = ["error"]``.  :func:`without_inert_knobs` puts
    every one the new accelerator does not read back to its default, so
    what reaches ``build()`` is the configuration this module means to
    measure and nothing else.  Under ``acceleration="none"`` that is
    what makes the plain arm plain.
    """
    def _one(group):
        fields = (None if accelerated_fields is None else tuple(
            (name, tuple(f)) for name, f in sorted(accelerated_fields.items())
            if name in group.nodes
        ))
        return without_inert_knobs(dataclasses.replace(
            group,
            convergence_norm=_NORM,
            acceleration=acceleration,
            accelerated_fields=fields or None,
            jacobian_reuse=jacobian_reuse,
            max_iterations=_CAP,
            diagnostics=True,
            strict_convergence=False,
            tolerance=_TOLERANCE,
            atol=_ATOL,
            rtol=_RTOL,
        ))

    return dataclasses.replace(
        recipe, coupling_groups=tuple(_one(g) for g in recipe.coupling_groups),
    )


#: ``allow_subcycling=False`` is a *generation* constraint, not a
#: filter.  A subcycled group is not one fixed-point iteration over one
#: state: a pass runs several sub-steps and reapplies the interface
#: override inside each, so its residual measures the last sub-step
#: rather than the pass, and the two solves interpolate two different
#: interface histories.  Measured at 3.4e+00 on a three-node graph with
#: timesteps 0.04 / 0.02 / 0.01, the accelerated exit reporting
#: ``amplification=nan`` and ``bound_valid=False``.  That is a statement
#: about waveform relaxation, and this property does not make it.
#:
#: It used to be said with ``assume``, which is the expensive way to say
#: it: better than half the drawn recipes carried a subcycled group
#: (55% measured on the base strategy, 64% after the inert-knob rules
#: narrowed what a demoted group may draw), every one of them paid for a
#: graph that was then thrown away, and together with the five
#: ``assume`` calls further down it put this test over Hypothesis's
#: ``filter_too_much`` threshold -- 9 examples kept out of 59.  Asking
#: for uniform timesteps costs nothing and rejects nothing.
_RECIPES = graph_recipes(
    min_nodes=2, max_nodes=3,
    kinds=_SMOOTH_KINDS,
    require_coupling_group=True,
    allow_mappings=False,
    allow_subcycling=False,
)


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(
    recipe=_RECIPES,
    acceleration=st.sampled_from(_ACCELERATIONS),
    jacobian_reuse=st.integers(min_value=0, max_value=5),
)
def test_accelerating_every_field_lands_on_the_same_answer_as_plain_iteration(
    recipe, acceleration, jacobian_reuse,
):
    """The guarantee the ten fixture rows are ten counter-examples to.

    An accelerator is a way of reaching the fixed point in fewer passes,
    not a different fixed point, so two solves of one graph that both
    report ``converged=True`` owe each other the same answer.  What the
    fixture rows show is that the guarantee is conditional: it holds
    over the fields the accelerator was given, and the auto-detected set
    is the interface, which under ``convergence_norm="interface"`` is
    also the only set the criterion measures.  Give the accelerator the
    whole state, under a norm that measures the whole state, and the
    condition is met -- which is the recommendation this stands behind.

    ``jacobian_reuse`` is drawn rather than fixed because
    ``iqn-imvj5`` — five columns carried across timesteps — is the
    configuration seven of the ten rows are in, and reuse is what makes
    an IMVJ answer depend on steps before this one.
    """
    # The strategy is asked for uniform timesteps (see ``_RECIPES``), so
    # no group here subcycles.  Asserted rather than assumed: if the
    # generator ever stops holding up its end this must fail loudly, not
    # quietly go back to throwing half its examples away.
    assert not any(g.subcycling for g in recipe.coupling_groups), (
        "allow_subcycling=False must not produce a subcycled group"
    )

    plain = _retune(recipe, acceleration="none")
    gm_plain = plain.build()
    group_keys = ["+".join(sorted(g.nodes)) for g in plain.coupling_groups]
    assume(group_keys)

    nodes = sorted({n for g in plain.coupling_groups for n in g.nodes})
    fields = _float_fields(gm_plain, nodes)
    # A group of integer-only state has no fixed point to disagree
    # about, and ``accelerated_fields`` naming no float field is
    # rejected at construction.
    assume(all(any(n in fields for n in g.nodes)
               for g in plain.coupling_groups))

    accelerated = _retune(
        recipe, acceleration=acceleration,
        accelerated_fields=fields, jacobian_reuse=jacobian_reuse,
    ).build()

    # An unconverged exit is not a claim about a fixed point, so the two
    # answers owe each other nothing; the fixture lane can assert
    # convergence because its graphs were built to converge and these
    # were drawn.
    diag_plain = _run(gm_plain, group_keys)
    assume(diag_plain is not None)
    diag_accel = _run(accelerated, group_keys)
    assume(diag_accel is not None)

    got_plain = _state(gm_plain, nodes)
    got_accel = _state(accelerated, nodes)
    assume(all(np.all(np.isfinite(v)) for v in got_plain.values()))
    assume(all(np.all(np.isfinite(v)) for v in got_accel.values()))

    deviation = _spread(got_plain, got_accel, nodes)
    note(f"accel={acceleration} reuse={jacobian_reuse} "
         f"deviation={deviation:.3e} "
         f"iterations={ {k: int(diag_accel[k]['iterations']) for k in group_keys} } "
         f"plain={ {k: int(diag_plain[k]['iterations']) for k in group_keys} }")
    assert deviation <= _AGREEMENT, (
        f"{acceleration} under the {_NORM} norm, accelerating every field, "
        f"left plain iteration's trajectory by {deviation:.3e} after "
        f"{_STEPS} steps with both reporting converged — an accelerator "
        f"that reaches a different answer is not an accelerator"
    )
