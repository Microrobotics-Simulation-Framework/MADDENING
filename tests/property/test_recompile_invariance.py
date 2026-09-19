"""A recompile changes no trajectory.

``compile()`` preserves node state and ``params`` across a rebuild --
that is the whole reason ``load_state`` compiles first -- but it used to
throw ``_meta`` away.  ``step_count`` is what decides which sub-steps a
node with a rate divider > 1 fires on, so every structural edit re-phased
the schedule mid-run: in the case that found this, declaring one unused,
zero-valued external input after four of eight steps moved the final
velocity by 33 %.

A mid-run edit is supported.  ``add_node``, ``add_edge``,
``add_external_input``, ``remove_node``, ``remove_edge``,
``add_coupling_group``, ``remove_coupling_group`` and ``enable_multigpu``
all dirty a live graph; ``maddening.api.server`` sets ``_dirty`` from its
node-parameter write endpoint, the interactive-slider path, which has no
reason to restart anything; and ``docs/user_guide/parameters.md`` sells
the recompile as transparent in exactly this scenario.

So the property is bit-identity, not closeness: an edit that changes no
physics must change no number.  Each edit below is inert by
construction -- an external input nothing reads, a bare dirty flag, a
node nobody is connected to -- and the schedules are generated rather
than chosen, because the failure only shows on a divider > 1 and grows
with the run.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import _META_KEY, GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.conftest import EXAMPLES_COSTLY

_BASE_DT = 0.01


class _Ramp(SimulationNode):
    """Integrates its own rate plus whatever it is fed.

    Both halves matter: the self term makes a node's own firing pattern
    visible in its state, and the fed term carries a neighbour's phase
    into it, so a re-phased upstream node cannot cancel out.
    """

    def halo_width(self) -> dict[int, int]:
        return {}

    def initial_state(self) -> dict:
        return {"y": jnp.array(0.0, dtype=jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        u = boundary_inputs.get("u", jnp.array(0.0, dtype=jnp.float32))
        return {"y": state["y"] + dt * (1.0 + u)}

    def boundary_input_spec(self) -> dict:
        return {"u": BoundaryInputSpec(shape=(), description="upstream y")}


@st.composite
def _schedules(draw):
    """``(dividers, n_pre, n_post)`` for a chain of multi-rate nodes."""
    dividers = draw(st.lists(
        st.integers(min_value=1, max_value=4), min_size=2, max_size=3,
    ))
    # A graph whose nodes all share a timestep is uniform-rate whatever
    # the drawn numbers were, and has no phase to lose; the interesting
    # space is the one with at least one node that skips sub-steps.
    if len(set(dividers)) == 1:
        dividers[-1] += 1
    return dividers, draw(st.integers(1, 6)), draw(st.integers(1, 6))


def _build(dividers):
    gm = GraphManager()
    names = [f"n{i}" for i in range(len(dividers))]
    for name, divider in zip(names, dividers):
        gm.add_node(_Ramp(name=name, timestep=_BASE_DT * divider))
    for upstream, downstream in zip(names, names[1:]):
        gm.add_edge(upstream, downstream, "y", "u")
    gm.compile()
    return gm, names


def _add_an_unread_external_input(gm, names):
    gm.add_external_input(names[0], "nothing_reads_this", shape=())


def _mark_dirty(gm, names):
    # What maddening.api.server's node-parameter endpoint does.
    gm._dirty = True


def _add_an_unconnected_node(gm, names):
    # At a timestep the graph already has, so no surviving node's divider
    # moves and the running schedule keeps its meaning.
    existing = gm._nodes[names[0]].timestep
    gm.add_node(_Ramp(name="bystander", timestep=existing))


_EDITS = (
    _add_an_unread_external_input,
    _mark_dirty,
    _add_an_unconnected_node,
)


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(schedule=_schedules(), edit=st.sampled_from(_EDITS))
def test_an_inert_mid_run_edit_changes_no_number(schedule, edit):
    dividers, n_pre, n_post = schedule

    reference, names = _build(dividers)
    reference.run(n_pre + n_post)

    gm, _ = _build(dividers)
    gm.run(n_pre)
    edit(gm, names)
    gm.run(n_post)

    for name in names:
        for field, expected in reference.get_node_state(name).items():
            np.testing.assert_array_equal(
                np.asarray(gm.get_node_state(name)[field]),
                np.asarray(expected),
                err_msg=f"{name}.{field} moved across the recompile",
            )


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(schedule=_schedules(), edit=st.sampled_from(_EDITS))
def test_the_counter_keeps_counting_across_the_edit(schedule, edit):
    """The mechanism, stated on its own.

    The trajectory property above is the thing users feel; this is the
    one line of state it reduces to, so a future change that preserves
    the numbers by accident still has to preserve this.
    """
    dividers, n_pre, n_post = schedule
    gm, names = _build(dividers)
    gm.run(n_pre)
    edit(gm, names)
    gm.run(n_post)
    assert int(gm._state[_META_KEY]["step_count"]) == n_pre + n_post


@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(schedule=_schedules())
def test_reset_state_is_still_the_way_to_zero_the_counter(schedule):
    """Preserving the counter must not take the explicit reset away.

    ``tests/core/test_step_retrace.py`` pins ``reset_state()`` as the
    documented way to restart a schedule; it is the reason ``compile()``
    doing the same thing silently was the anomaly.
    """
    dividers, n_pre, _ = schedule
    gm, _ = _build(dividers)
    gm.run(n_pre)
    assert int(gm._state[_META_KEY]["step_count"]) == n_pre
    gm.reset_state()
    assert int(gm._state[_META_KEY]["step_count"]) == 0
