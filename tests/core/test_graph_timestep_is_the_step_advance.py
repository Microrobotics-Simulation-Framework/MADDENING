"""``GraphManager.timestep`` is the simulated time one ``step()`` advances.

A ``subcycling=True`` coupling group solves once per macro step -- its
largest member timestep -- and sub-steps its faster members inside that
solve, so ``compile()`` schedules every member at the macro timestep.
``timestep`` used to be the GCD of the nodes' *own* timesteps, which on
such a graph is shorter than a step: a group of 0.01 and 0.02 read 0.01
while every step advanced 0.02 (MADD-ANO-096), and everything that clocks
a run with it -- ``RealtimeRunner``, the state relays, the FMU's default
step, USD ``baseDt`` -- ran at half the real rate.  It is now the GCD of
the timesteps ``compile()`` schedules, from the same helper.

The probe is a node that counts its own updates: after ``n`` steps (a
multiple of every rate divider) each node has advanced ``count * own dt``,
and that must be ``n * gm.timestep`` for every node of every graph.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode


class _Counter(SimulationNode):
    """``n <- n + 1`` per update; ``x <- 0.5 * u + 1`` (``u`` from a partner)."""

    def __init__(self, name, timestep):
        super().__init__(name=name, timestep=timestep)

    def initial_state(self):
        return {"n": jnp.int32(0), "x": jnp.float32(0.0)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=jnp.float32,
                                       default=jnp.float32(0.0))}

    def update(self, state, boundary_inputs, dt):
        u = boundary_inputs.get("u", jnp.float32(0.0))
        return {"n": state["n"] + 1, "x": 0.5 * u + 1.0}


def _group(gm, a, b):
    gm.add_edge(a, b, "x", "u")
    gm.add_edge(b, a, "x", "u")
    gm.add_coupling_group([a, b], max_iterations=20, tolerance=1e-6, subcycling=True)


def _graph(kind: str) -> GraphManager:
    """Every schedule ``compile()`` builds from a set of timesteps."""
    gm = GraphManager()
    if kind == "uniform":
        gm.add_node(_Counter("a", 0.01))
        gm.add_node(_Counter("b", 0.01))
        gm.add_edge("a", "b", "x", "u")
    elif kind == "multirate":
        gm.add_node(_Counter("fast", 0.01))
        gm.add_node(_Counter("slow", 0.02))
        gm.add_edge("fast", "slow", "x", "u")
    elif kind == "multirate_not_dividing":
        # 0.002 and 0.003: the step is 0.001, neither node's own timestep.
        gm.add_node(_Counter("a", 0.002))
        gm.add_node(_Counter("b", 0.003))
        gm.add_edge("a", "b", "x", "u")
    elif kind == "subcycled":
        # Uniform rate at the macro step: one step is 0.02 (it read 0.01).
        gm.add_node(_Counter("fine", 0.01))
        gm.add_node(_Counter("coarse", 0.02))
        _group(gm, "fine", "coarse")
    elif kind == "subcycled_with_a_node_at_the_member_rate":
        # Mixed: a node outside at 0.01 keeps the step at 0.01, and the group
        # fires every second step.  The one case the old GCD got right.
        gm.add_node(_Counter("fine", 0.01))
        gm.add_node(_Counter("coarse", 0.02))
        _group(gm, "fine", "coarse")
        gm.add_node(_Counter("outside", 0.01))
        gm.add_edge("coarse", "outside", "x", "u")
    elif kind == "subcycled_mixed":
        # Group 0.005/0.02 beside nodes at 0.01 and 0.03: scheduled
        # {0.02, 0.02, 0.01, 0.03}, a step of 0.01 (it read 0.005).
        gm.add_node(_Counter("fine", 0.005))
        gm.add_node(_Counter("coarse", 0.02))
        _group(gm, "fine", "coarse")
        gm.add_node(_Counter("ref", 0.01))
        gm.add_node(_Counter("slow", 0.03))
        gm.add_edge("coarse", "ref", "x", "u")
        gm.add_edge("ref", "slow", "x", "u")
    elif kind == "two_subcycled_groups":
        # Two groups, macro 0.02 and 0.03: a step of 0.01, no node at it.
        gm.add_node(_Counter("f1", 0.01))
        gm.add_node(_Counter("c1", 0.02))
        _group(gm, "f1", "c1")
        gm.add_node(_Counter("f2", 0.015))
        gm.add_node(_Counter("c2", 0.03))
        _group(gm, "f2", "c2")
        gm.add_edge("c1", "f2", "x", "u")
    else:  # pragma: no cover - a typo in the parametrisation
        raise AssertionError(kind)
    return gm


#: kind -> the step every update count below agrees on.
STEP = {
    "uniform": 0.01,
    "multirate": 0.01,
    "multirate_not_dividing": 0.001,
    "subcycled": 0.02,
    "subcycled_with_a_node_at_the_member_rate": 0.01,
    "subcycled_mixed": 0.01,
    "two_subcycled_groups": 0.01,
}


@pytest.fixture(scope="module", params=sorted(STEP))
def ran(request):
    """``(kind, gm, n_steps)``: the graph after ``n_steps`` steps."""
    kind = request.param
    gm = _graph(kind)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
    n_steps = 12          # a multiple of every rate divider above
    assert all(n_steps % d == 0 for d in gm.rate_dividers.values()), gm.rate_dividers
    gm.run_scan(n_steps)
    return kind, gm, n_steps


def test_every_node_advanced_n_steps_of_gm_timestep(ran):
    """The invariant: ``n`` steps cover ``n * gm.timestep`` of every node's
    clock, sub-cycled members included (they update twice per step at half
    the timestep)."""
    kind, gm, n_steps = ran
    for name in gm.node_names:
        advanced = int(gm.get_node_state(name)["n"]) * gm._nodes[name].timestep  # noqa: SLF001
        assert advanced == pytest.approx(n_steps * gm.timestep, rel=1e-9), (kind, name)


def test_the_step_is_the_documented_one(ran):
    kind, gm, _ = ran
    assert gm.timestep == pytest.approx(STEP[kind], rel=1e-9)
    assert gm.base_timestep == gm.timestep


def test_the_rate_dividers_count_in_gm_timestep(ran):
    """``rate_dividers[n] * timestep`` is node ``n``'s scheduled timestep --
    its own, or its sub-cycling group's largest -- so ``compile()`` and
    ``timestep`` read one rule."""
    kind, gm, _ = ran
    for group in gm._coupling_groups:  # noqa: SLF001
        macro = max(gm._nodes[n].timestep for n in group.nodes)  # noqa: SLF001
        for n in group.nodes:
            assert gm.rate_dividers[n] * gm.timestep == pytest.approx(macro, rel=1e-9), (kind, n)
    grouped = {n for g in gm._coupling_groups for n in g.nodes}  # noqa: SLF001
    for n in set(gm.node_names) - grouped:
        assert gm.rate_dividers[n] * gm.timestep == pytest.approx(
            gm._nodes[n].timestep, rel=1e-9), (kind, n)  # noqa: SLF001


def test_an_uncompiled_graph_reports_the_step_its_compile_will_take():
    gm = _graph("subcycled")
    assert gm.timestep == pytest.approx(0.02)
    gm.compile()
    assert gm.timestep == pytest.approx(0.02)


def test_a_group_without_subcycling_keeps_its_members_own_rate():
    """``subcycling=False`` with equal timesteps: nothing to schedule."""
    gm = GraphManager()
    gm.add_node(_Counter("a", 0.01))
    gm.add_node(_Counter("b", 0.01))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-6)
    gm.add_node(_Counter("slow", 0.03))
    gm.add_edge("b", "slow", "x", "u")
    assert gm.timestep == pytest.approx(0.01)


def test_an_empty_graph_still_raises():
    with pytest.raises(RuntimeError, match="No nodes registered"):
        _ = GraphManager().timestep


def test_editing_the_graph_moves_the_step():
    """Live, not cached: adding a faster node or a sub-cycling group moves
    it before any recompile."""
    gm = GraphManager()
    gm.add_node(_Counter("fine", 0.01))
    gm.add_node(_Counter("coarse", 0.02))
    gm.add_edge("fine", "coarse", "x", "u")
    gm.add_edge("coarse", "fine", "x", "u")
    assert gm.timestep == pytest.approx(0.01)
    gm.add_coupling_group(["fine", "coarse"], max_iterations=20, tolerance=1e-6,
                          subcycling=True)
    assert gm.timestep == pytest.approx(0.02)
    gm.add_node(_Counter("fast", 0.005))
    assert gm.timestep == pytest.approx(0.005)


@pytest.mark.parametrize("kind, multirate", [
    ("subcycled", False), ("subcycled_mixed", True), ("multirate_not_dividing", True),
])
def test_validate_reports_the_schedule_compile_builds(kind, multirate):
    """``validate()``'s multi-rate line reads the scheduled timesteps: a
    sub-cycling group of 0.01 and 0.02 is one macro rate, not a multi-rate
    graph with a step of 0.01 (what it used to say)."""
    gm = _graph(kind)
    info = [i for i in gm.validate() if i.startswith("INFO: multi-rate")]
    assert bool(info) is multirate, info
    if multirate:
        assert f"Base timestep: {gm.timestep}" in info[0], info
        gm.compile()
        assert f"rate dividers: {gm.rate_dividers}" in info[0], info
