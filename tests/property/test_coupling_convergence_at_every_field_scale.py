"""``converged=True`` means *every* field arrived, whatever its magnitude.

``tests/property/test_coupling_error_bound.py`` already states "a group
that reports ``converged=True`` is near its fixed point" over generated
graphs.  It could not see this defect, for two reasons that are worth
writing down because they are the reasons a property test passes a bug:

* its reference is *the same recipe solved again* with a tighter
  threshold, and :func:`_tightened` deliberately does not move ``atol``
  -- so the dead band applied identically to the reference, and the
  reference was wrong in exactly the same way as the answer;
* its fixtures are O(1).  A dead band is an *absolute* number, so it is
  invisible until the physics is written in units small enough to fall
  inside it.

So this module fixes both.  The fixed point is known in closed form --
each field runs its own scalar cycle ``a = g*b + bias``, ``b = g*a``,
whose fixed point is ``bias / (1 - g**2)`` -- and the field magnitudes
are drawn across fifteen decades, from 1e-12 to 1e3, independently for
the two fields of every node.  A group that says it converged is then
checked field by field against arithmetic, not against itself.

The property is one sentence:

    If ``coupling_diagnostics()`` reports ``converged=True``, then every
    float field of every node in the group is at its fixed point to
    within the criterion the group was given -- at every scale, and
    whichever of the three norms measured it.

Its counter-example, before the fix, was the audit's: an O(1) field and
an O(1e-9) field in one group under the default ``atol=1e-8``, reported
``{'iterations': 1, 'residual': 0.0, 'converged': True}`` with the small
field 50% away.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.conftest import EXAMPLES_COSTLY


#: How far past its own threshold a converged group is allowed to be.
#:
#: The threshold is a statement about a *norm* over fields; this module
#: asserts per field, the reference is float32 (~1e-7 of relative noise
#: a field), and the bound extrapolates a rate read off two residuals.
#: Twenty decades of slack over the tightest threshold drawn here is
#: 2e-4 of relative error, which every one of those three can hide
#: inside and which is three orders below the 50% the dead band was
#: measured producing.  A slack that had to be widened to keep this
#: green would itself be the finding.
_SLACK = 20.0

#: Floor under the budget: below ~1e-6 relative, float32 state is noise.
_FLOAT32_FLOOR = 1e-5


class _TwoScale(SimulationNode):
    """Two independent scalar fields in one node, at unrelated magnitudes.

    ``big`` and ``small`` never interact, so the group is a direct
    product of two scalar cycles and each field's fixed point is
    arithmetic rather than a second solve.  That is the whole point:
    the reference cannot inherit the defect under test.
    """

    def __init__(self, name, gain, bias_big, bias_small):
        super().__init__(name=name, timestep=1.0, gain=gain,
                         bias_big=bias_big, bias_small=bias_small)

    def initial_state(self):
        return {"big": jnp.asarray(0.0, jnp.float32),
                "small": jnp.asarray(0.0, jnp.float32)}

    def state_fields(self):
        return ["big", "small"]

    def boundary_input_spec(self):
        z = jnp.float32(0.0)
        return {"ub": BoundaryInputSpec(shape=(), dtype=jnp.float32, default=z),
                "us": BoundaryInputSpec(shape=(), dtype=jnp.float32, default=z)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        g = jnp.asarray(p["gain"])
        return {"big": g * boundary_inputs["ub"] + jnp.asarray(p["bias_big"]),
                "small": g * boundary_inputs["us"] + jnp.asarray(p["bias_small"])}


def _graph(*, big, small, gain, norm, solver, acceleration, threshold):
    """The two-field cycle, with only the knobs ``norm`` actually reads.

    Setting a knob a norm ignores raises a ``UserWarning``, which
    ``filterwarnings = ["error"]`` makes fatal, so the live threshold is
    ``tolerance`` under ``"l2"`` and ``rtol`` under the other two.
    """
    gm = GraphManager()
    gm.add_node(_TwoScale("a", gain=gain, bias_big=big, bias_small=small))
    gm.add_node(_TwoScale("b", gain=gain, bias_big=0.0, bias_small=0.0))
    for src, dst in (("a", "b"), ("b", "a")):
        gm.add_edge(source=src, target=dst,
                    source_field="big", target_field="ub")
        gm.add_edge(source=src, target=dst,
                    source_field="small", target_field="us")
    live = ({"tolerance": threshold} if norm == "l2" else {"rtol": threshold})
    gm.add_coupling_group(
        ["a", "b"], convergence_norm=norm, solver=solver,
        acceleration=acceleration, max_iterations=60, diagnostics=True,
        **live,
    )
    gm.compile()
    return gm


#: Fifteen decades, which is where an absolute dead band becomes
#: visible: the shipped default was 1e-8, in the middle of this range.
_MAGNITUDES = st.sampled_from(
    [1e-12, 1e-9, 1e-8, 1e-7, 1e-5, 1e-3, 1.0, 1e3]
)


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(
    big=_MAGNITUDES,
    small=_MAGNITUDES,
    gain=st.sampled_from([0.2, 0.5, 0.7]),
    norm=st.sampled_from(["l2", "mixed", "interface"]),
    solver=st.sampled_from(["ift", "fori"]),
    acceleration=st.sampled_from(["none", "aitken", "fixed"]),
    threshold=st.sampled_from([1e-3, 1e-5]),
)
def test_a_converged_group_is_at_its_fixed_point_in_every_field_at_every_scale(
    big, small, gain, norm, solver, acceleration, threshold,
):
    """The invariant, over magnitudes rather than over one fixture.

    A verdict about a group is a verdict about all of it.  Reporting
    ``converged=True`` while one field is nowhere near its fixed point
    is worse than reporting ``converged=False``, because the second is
    a number the caller can act on and the first is one they will
    trust -- and ``strict_convergence`` raises on the second and stays
    silent on the first.
    """
    gm = _graph(big=big, small=small, gain=gain, norm=norm, solver=solver,
                acceleration=acceleration, threshold=threshold)
    gm.step()
    diag = gm.coupling_diagnostics()["a+b"]
    if not diag["converged"]:
        return          # the honest answer; this property says nothing of it
    budget = max(_SLACK * threshold, _FLOAT32_FLOOR)
    exact = {
        "a": {"big": big / (1.0 - gain ** 2),
              "small": small / (1.0 - gain ** 2)},
        "b": {"big": gain * big / (1.0 - gain ** 2),
              "small": gain * small / (1.0 - gain ** 2)},
    }
    for node in ("a", "b"):
        state = gm.get_node_state(node)
        for field, want in exact[node].items():
            got = float(state[field])
            if want == 0.0:
                assert got == 0.0, (node, field, got)
                continue
            rel = abs(got - want) / abs(want)
            assert rel <= budget, (
                f"{node}.{field} is {rel:.3g} from its fixed point "
                f"({got!r} against {want!r}) while the group reports "
                f"{diag}; budget {budget:.3g}"
            )


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(
    small=st.sampled_from([1e-12, 1e-9, 1e-7]),
    gain=st.sampled_from([0.5, 0.7]),
    norm=st.sampled_from(["l2", "mixed", "interface"]),
)
def test_a_small_field_is_not_dropped_merely_for_being_small(
    small, gain, norm,
):
    """The mechanism, stated separately from its consequence.

    The consequence above -- a wrong answer called converged -- needs
    the small field to be the *only* thing unconverged, which not every
    draw arranges.  This states the mechanism directly: a group
    carrying a field six or more decades below its largest one must
    still measure that field, so its residual has to move when the
    field does.  Under the shipped default the residual was exactly
    ``0.0`` and the group exited after one pass, whatever the small
    field was doing.
    """
    assume(small < 1e-6)
    threshold = 1e-5
    gm = _graph(big=1.0, small=small, gain=gain, norm=norm, solver="ift",
                acceleration="none", threshold=threshold)
    gm.step()
    diag = gm.coupling_diagnostics()["a+b"]
    got = float(gm.get_node_state("a")["small"])
    want = small / (1.0 - gain ** 2)
    assert got == pytest.approx(want, rel=max(_SLACK * threshold,
                                              _FLOAT32_FLOOR)), (
        f"the small field stopped at {got!r} against {want!r}; {diag}"
    )
    assert diag["iterations"] > 1, (
        f"one pass cannot converge a contraction of {gain ** 2}; {diag}"
    )
