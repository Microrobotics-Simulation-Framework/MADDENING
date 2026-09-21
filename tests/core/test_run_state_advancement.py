"""Which ``GraphManager`` run entry points advance the graph's own state.

Six of them do -- ``step``, ``run``, ``run_scan``,
``run_scan_with_history``, ``run_adaptive`` and ``run_adaptive_scan`` all
end by calling ``_store_state`` -- and ``run_sweep`` does not.  Until this
module none of it was pinned, and only ``step`` hinted at it in its
docstring.

The consequence when it is missed is not a crash.  A 0.4.0 measurement
harness built observations more than once from one ``GraphManager``,
silently measured a different initial condition each time, and reported
plausible wrong numbers (+3.256% and -3.6% / -10.0% / +87.6% / +36.4%,
against -0.849% and -1.97% to -4.80% re-derived from a fresh graph per
measurement).  So the load-bearing assertion here is not "the state
moved" but "a **second** call continues from where the first ended
rather than restarting" -- that is the one the harness bug would have
tripped.

The graph used is a ball in free fall: the table sits far below it, so
no collision ever fires and the velocity is strictly monotone.  A
fixture whose state a step leaves unchanged -- a spring at its rest
length with zero velocity is the known example in this repo -- cannot
express the defect at all, so every test below either checks progress
strictly or pairs its equality check with a demonstration that the same
comparison does detect a change.
"""

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


#: Steps per call for the fixed-step entry points.  Small enough to stay
#: cheap, large enough that one call's displacement is far outside any
#: floating-point tolerance used below.
N_STEPS = 20

#: ``t_end`` per call for the adaptive entry points.
T_END = 0.2

#: Scan allocation for ``run_adaptive_scan``.  ``T_END / dt_initial`` is
#: about 20 accepted steps, so 64 leaves room for rejections without
#: paying for the 10000-step default.
MAX_STEPS = 64


def _free_fall_graph() -> GraphManager:
    """A compiled ball falling freely, far above the table.

    The table is at -1e6 so the collision branch never fires: velocity
    is then strictly decreasing at every step, which is what lets
    "continued" be told apart from "restarted" without relying on
    bit-exact arithmetic.
    """
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01, position=-1.0e6))
    gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0,
                         initial_velocity=0.0, elasticity=0.7))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


def _snapshot(gm: GraphManager) -> dict:
    """Every node field the graph currently holds, detached from it."""
    return {
        name: {f: jnp.asarray(v) for f, v in gm.get_node_state(name).items()}
        for name in gm.node_names
    }


def _identical(a: dict, b: dict) -> bool:
    """True when two snapshots agree on every field, bit for bit."""
    assert a.keys() == b.keys()
    return all(
        jnp.array_equal(a[name][field], b[name][field])
        for name in a
        for field in a[name]
    )


def _velocity(gm: GraphManager) -> float:
    return float(gm.get_node_state("ball")["velocity"])


def _velocity_of(snapshot: dict) -> float:
    return float(snapshot["ball"]["velocity"])


# ``n`` is a step count for the fixed-step runners and a ``t_end`` for
# the adaptive ones; each runner interprets its own unit.
_FIXED_STEP_RUNNERS = {
    "step": lambda gm, n: [gm.step() for _ in range(int(n))],
    "run": lambda gm, n: gm.run(int(n)),
    "run_scan": lambda gm, n: gm.run_scan(int(n)),
    "run_scan_with_history": lambda gm, n: gm.run_scan_with_history(int(n)),
}

_ADAPTIVE_RUNNERS = {
    "run_adaptive": lambda gm, t: gm.run_adaptive(t),
    "run_adaptive_scan": lambda gm, t: gm.run_adaptive_scan(
        t, max_steps=MAX_STEPS,
    ),
}

_MUTATING = [
    *((name, fn, N_STEPS) for name, fn in _FIXED_STEP_RUNNERS.items()),
    *((name, fn, T_END) for name, fn in _ADAPTIVE_RUNNERS.items()),
]

_MUTATING_IDS = [name for name, _fn, _unit in _MUTATING]


# ------------------------------------------------------------------
# The fixture must be able to express the defect
# ------------------------------------------------------------------

def test_free_fall_graph_state_moves_under_a_single_step():
    """The fixture can show that state advanced.

    Guards every test below: a graph whose one-step residual is zero --
    a spring started at its rest length with zero velocity -- would make
    "the state moved" unfalsifiable, and both this module's equality
    checks and its progress checks would pass on a `GraphManager` that
    never stored anything.
    """
    gm = _free_fall_graph()
    before = _snapshot(gm)
    gm.step()
    assert not _identical(before, _snapshot(gm)), (
        "the free-fall fixture's state is unchanged by a step, so it "
        "cannot express a missing _store_state"
    )
    assert _velocity(gm) < 0.0, "a falling ball must gain negative velocity"


def test_free_fall_graph_velocity_is_strictly_monotone():
    """No collision fires, so each call must leave velocity strictly lower.

    The strict-progress assertions below are only meaningful while this
    holds; a bounce would let a later velocity coincide with an earlier
    one and make "continued" indistinguishable from "restarted".
    """
    gm = _free_fall_graph()
    seen = [_velocity(gm)]
    for _ in range(4):
        gm.run(N_STEPS)
        seen.append(_velocity(gm))
    assert all(b < a for a, b in zip(seen, seen[1:])), seen
    assert float(gm.get_node_state("ball")["position"]) > 0.0, (
        "the ball reached the table; the fixture no longer free-falls"
    )


# ------------------------------------------------------------------
# The six entry points that advance the graph's own state
# ------------------------------------------------------------------

@pytest.mark.parametrize("name, run, unit", _MUTATING, ids=_MUTATING_IDS)
def test_run_entry_point_advances_the_graphs_own_state(name, run, unit):
    """Calling it leaves the graph at the state the run reached."""
    gm = _free_fall_graph()
    before = _snapshot(gm)

    run(gm, unit)

    after = _snapshot(gm)
    assert not _identical(before, after), (
        f"GraphManager.{name} left the graph's own state untouched; it is "
        f"documented as advancing the graph to the final state it reaches"
    )
    assert _velocity(gm) < _velocity_of(before), (
        f"GraphManager.{name} did not advance the graph's velocity"
    )


@pytest.mark.parametrize("name, run, unit", _MUTATING, ids=_MUTATING_IDS)
def test_second_call_continues_rather_than_restarting(name, run, unit):
    """Two calls advance twice as far as one, from one graph.

    This is the assertion the 0.4.0 harness bug would have tripped.  A
    graph that restarted each call would leave *twice* and *once* at the
    same state; the reference graph shows what continuing actually
    reaches.
    """
    gm = _free_fall_graph()

    run(gm, unit)
    after_first = _snapshot(gm)
    run(gm, unit)
    after_second = _snapshot(gm)

    assert not _identical(after_first, after_second), (
        f"a second GraphManager.{name} left the graph where the first "
        f"did: it restarted from the initial condition instead of "
        f"continuing from the state the first call reached"
    )
    assert _velocity_of(after_second) < _velocity_of(after_first), (
        f"a second GraphManager.{name} did not carry the graph further "
        f"than the first"
    )

    # What "continuing" reaches, measured the only safe way: a fresh
    # graph per measurement.
    reference = _free_fall_graph()
    run(reference, 2 * unit)

    assert jnp.allclose(
        after_second["ball"]["velocity"],
        jnp.asarray(_velocity(reference)),
        rtol=1e-5, atol=1e-6,
    ), (
        f"two GraphManager.{name} calls of {unit} did not reach the "
        f"state one call of {2 * unit} reaches on a fresh graph: "
        f"{_velocity_of(after_second)} vs {_velocity(reference)}.  The "
        f"second call did not continue from where the first ended."
    )


# ------------------------------------------------------------------
# The one that does not
# ------------------------------------------------------------------

def _batched_initial_states(batch: int = 3) -> dict:
    return {
        "ball": {
            "position": jnp.linspace(5.0, 7.0, batch),
            "velocity": jnp.zeros(batch),
        },
        "table": {"position": jnp.full((batch,), -1.0e6)},
    }


def test_run_sweep_leaves_the_graphs_own_state_untouched():
    """``run_sweep`` is the one run entry point that does not mutate.

    It runs from the caller's ``initial_states`` and writes nothing
    back.  Pinned because an unmarked non-mutator is as easy to misread
    as an unmarked mutator -- and because the six siblings above make
    "every ``run_*`` advances the state" the natural wrong guess.
    """
    gm = _free_fall_graph()
    before = _snapshot(gm)

    gm.run_sweep(N_STEPS, _batched_initial_states())

    assert _identical(before, _snapshot(gm)), (
        "GraphManager.run_sweep advanced the graph's own state; it is "
        "documented as leaving it exactly as it was"
    )

    # The comparison above must be able to fail.  A mutating sibling on
    # the same graph, compared the same way, has to show a difference --
    # otherwise the assertion passes for the wrong reason.
    gm.step()
    assert not _identical(before, _snapshot(gm)), (
        "_identical cannot detect a state change on this graph, so the "
        "run_sweep assertion above proves nothing"
    )


def test_run_sweep_twice_returns_identical_results():
    """The user-facing consequence of not mutating: it is idempotent.

    Two identical sweeps from one ``GraphManager`` agree exactly.  The
    same pair of calls to any of the six mutating entry points would
    not.
    """
    gm = _free_fall_graph()
    initial_states = _batched_initial_states()

    first = gm.run_sweep(N_STEPS, initial_states)
    second = gm.run_sweep(N_STEPS, initial_states)

    for node in first:
        for field in first[node]:
            assert jnp.array_equal(first[node][field], second[node][field]), (
                f"run_sweep is not idempotent on {node}.{field}: a second "
                f"identical sweep gave a different answer, so something "
                f"about the graph changed between them"
            )

    # And the batch really did simulate something, so equality above is
    # not equality of two untouched inputs.
    assert jnp.all(first["ball"]["velocity"] < 0.0), (
        "the sweep returned its initial velocities; it did not run"
    )


# ------------------------------------------------------------------
# The contract is documented, not only enforced
# ------------------------------------------------------------------

def test_every_run_entry_point_documents_whether_it_advances_state():
    """The six say they advance the state and ``run_sweep`` says it does not.

    The behaviour above is pinned either way; this pins that a reader
    can find out which is which without reading ``_store_state`` call
    sites.  Deleting the Notes sections would otherwise be silent.
    """
    for name in _MUTATING_IDS:
        doc = getattr(GraphManager, name).__doc__ or ""
        assert "state is advanced" in doc, (
            f"GraphManager.{name} advances the graph's own state (pinned "
            f"by the tests above) but its docstring does not say so"
        )

    sweep_doc = GraphManager.run_sweep.__doc__ or ""
    assert "Not stateful" in sweep_doc, (
        "GraphManager.run_sweep does not advance the graph's own state "
        "(pinned above) but its docstring does not say so"
    )
