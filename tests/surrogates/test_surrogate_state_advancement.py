"""The surrogate entry points that advance the graph they are handed.

``DatasetGenerator.from_graph`` and ``SurrogateValidator.compare_graphs``
both run ``GraphManager.run_scan_with_history`` on a graph the caller
owns, so both leave that graph at the end of the rollout.
``DatasetGenerator.from_sweep`` goes through ``run_sweep`` and does not.

This is the same contract ``tests/core/test_run_state_advancement.py``
pins on the six ``GraphManager`` entry points, one level up: the scan
that found those six also found these two, and they were left
undocumented because that work was scoped to ``GraphManager``.  The
module structure here deliberately mirrors that one.

The consequence is never a crash.  A caller generating several datasets
from one graph gets silently continued trajectories -- the second
dataset is drawn from wherever the first run ended -- and a caller
comparing a physics graph against a surrogate twice measures the
*second* ``n_steps`` of divergence while believing they re-measured the
first.  So the load-bearing assertion below is not "the state moved" but
"a **second** call continues rather than restarting".

The graph is a ball in free fall with the table placed far below it, so
no collision fires and velocity is strictly monotone.  A fixture whose
state a step leaves unchanged -- a spring started at its rest length
with zero velocity is the known example in this repo -- could not
express a missing state write at all, so the first test here checks the
fixture itself.
"""

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode
from maddening.surrogates.dataset import DatasetGenerator
from maddening.surrogates.validator import SurrogateValidator


#: Steps per call.  Small enough to stay cheap, large enough that one
#: call's change in velocity is far outside any tolerance used below.
N_STEPS = 20


def _free_fall_graph() -> GraphManager:
    """A compiled ball falling freely, far above the table.

    The table sits at -1e6 so the collision branch never fires: velocity
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


def _batched_initial_states(batch: int = 3) -> dict:
    return {
        "ball": {
            "position": jnp.linspace(5.0, 7.0, batch),
            "velocity": jnp.zeros(batch),
        },
        "table": {"position": jnp.full((batch,), -1.0e6)},
    }


# ------------------------------------------------------------------
# The fixture must be able to express the defect
# ------------------------------------------------------------------

def test_free_fall_graph_state_moves_under_a_single_step():
    """The fixture can show that state advanced.

    Guards every test below.  On a graph whose one-step residual is zero
    both the equality checks and the progress checks would pass against
    an entry point that stored nothing, and the module would prove
    nothing at all.
    """
    gm = _free_fall_graph()
    before = _snapshot(gm)
    gm.step()
    assert not _identical(before, _snapshot(gm)), (
        "the free-fall fixture's state is unchanged by a step, so it "
        "cannot express a missing state write"
    )
    assert _velocity(gm) < 0.0, "a falling ball must gain negative velocity"


def test_free_fall_graph_velocity_is_strictly_monotone():
    """No collision fires, so each call leaves velocity strictly lower.

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
# DatasetGenerator.from_graph
# ------------------------------------------------------------------

def test_from_graph_advances_the_graph_it_is_given():
    """It leaves the caller's graph at the end of the trajectory."""
    gm = _free_fall_graph()
    before = _snapshot(gm)

    DatasetGenerator.from_graph(gm, "ball", n_steps=N_STEPS)

    after = _snapshot(gm)
    assert not _identical(before, after), (
        "DatasetGenerator.from_graph left the graph's own state "
        "untouched; it runs run_scan_with_history, which advances it"
    )
    assert _velocity(gm) < _velocity_of(before), (
        "DatasetGenerator.from_graph did not advance the graph's velocity"
    )


def test_second_from_graph_continues_rather_than_restarting():
    """Two datasets from one graph are consecutive, not two of the same.

    This is the assertion that catches the real defect: a caller
    generating several datasets from a single ``GraphManager`` believes
    it is resampling one initial condition and is in fact walking down a
    single trajectory.
    """
    gm = _free_fall_graph()

    first = DatasetGenerator.from_graph(gm, "ball", n_steps=N_STEPS)
    after_first = _snapshot(gm)
    second = DatasetGenerator.from_graph(gm, "ball", n_steps=N_STEPS)
    after_second = _snapshot(gm)

    assert not _identical(after_first, after_second), (
        "a second DatasetGenerator.from_graph left the graph where the "
        "first did: it restarted from the initial condition instead of "
        "continuing from the state the first call reached"
    )
    assert _velocity_of(after_second) < _velocity_of(after_first), (
        "a second DatasetGenerator.from_graph did not carry the graph "
        "further than the first"
    )

    # The datasets themselves differ, which is what a caller sees.
    assert not jnp.array_equal(
        first.states["velocity"], second.states["velocity"],
    ), (
        "two from_graph datasets off one graph hold identical samples; "
        "the second was expected to be drawn from the continuation"
    )
    assert float(second.states["velocity"][0]) < float(
        first.states["velocity"][-1],
    ), (
        "the second dataset does not begin after the first ends, so it "
        "is not the continuation the advanced state implies"
    )

    # What "continuing" reaches, measured the only safe way: a fresh
    # graph per measurement.
    reference = _free_fall_graph()
    DatasetGenerator.from_graph(reference, "ball", n_steps=2 * N_STEPS)

    assert jnp.allclose(
        after_second["ball"]["velocity"],
        jnp.asarray(_velocity(reference)),
        rtol=1e-5, atol=1e-6,
    ), (
        f"two from_graph calls of {N_STEPS} did not reach the state one "
        f"call of {2 * N_STEPS} reaches on a fresh graph: "
        f"{_velocity_of(after_second)} vs {_velocity(reference)}"
    )


def test_reset_state_between_from_graph_calls_restores_the_initial_condition():
    """The documented workaround actually works.

    Without this the docstring's advice is untested advice, and a reader
    following it has no more assurance than one who does not.
    """
    gm = _free_fall_graph()

    first = DatasetGenerator.from_graph(gm, "ball", n_steps=N_STEPS)
    gm.reset_state()
    second = DatasetGenerator.from_graph(gm, "ball", n_steps=N_STEPS)

    assert jnp.array_equal(
        first.states["velocity"], second.states["velocity"],
    ), (
        "reset_state() between two from_graph calls did not reproduce "
        "the first dataset, so the workaround the docstring recommends "
        "does not restore the initial condition"
    )


def test_from_sweep_leaves_the_graphs_own_state_untouched():
    """``from_sweep`` is the generator entry point that does not mutate.

    It runs through ``run_sweep``, which writes nothing back.  Pinned
    because an unmarked non-mutator is as easy to misread as an unmarked
    mutator, and because its sibling above makes "every generator
    advances the graph" the natural wrong guess.
    """
    gm = _free_fall_graph()
    before = _snapshot(gm)

    DatasetGenerator.from_sweep(
        gm, "ball", N_STEPS, _batched_initial_states(),
    )

    assert _identical(before, _snapshot(gm)), (
        "DatasetGenerator.from_sweep advanced the graph's own state; it "
        "goes through run_sweep, which leaves it exactly as it was"
    )

    # The comparison above must be able to fail.
    gm.step()
    assert not _identical(before, _snapshot(gm)), (
        "_identical cannot detect a state change on this graph, so the "
        "from_sweep assertion above proves nothing"
    )


def test_from_sweep_says_it_is_not_stateful():
    """Same marker ``run_sweep`` uses, for the same reason.

    An unmarked non-mutator standing next to a documented mutator reads
    as an oversight; saying so explicitly is what stops a reader
    generalising ``from_graph``'s Notes to both.
    """
    doc = DatasetGenerator.from_sweep.__doc__ or ""
    assert "Not stateful" in doc, (
        "DatasetGenerator.from_sweep leaves the graph untouched (pinned "
        "above) but its docstring does not say so"
    )


# ------------------------------------------------------------------
# SurrogateValidator.compare_graphs
# ------------------------------------------------------------------

def test_compare_graphs_advances_both_graphs_it_is_given():
    """Neither argument is left where the caller passed it."""
    gm_a = _free_fall_graph()
    gm_b = _free_fall_graph()
    before_a = _snapshot(gm_a)
    before_b = _snapshot(gm_b)

    SurrogateValidator.compare_graphs(gm_a, gm_b, N_STEPS, "ball")

    for label, gm, before in (("gm_physics", gm_a, before_a),
                              ("gm_surrogate", gm_b, before_b)):
        assert not _identical(before, _snapshot(gm)), (
            f"SurrogateValidator.compare_graphs left {label}'s own state "
            f"untouched; it runs run_scan_with_history on both"
        )
        assert _velocity(gm) < _velocity_of(before), (
            f"SurrogateValidator.compare_graphs did not advance {label}"
        )


def test_second_compare_graphs_continues_rather_than_restarting():
    """A repeated comparison measures the next window, not the same one.

    ``compare_graphs`` exists to show error accumulation over a rollout,
    so a second call silently reporting the *following* ``n_steps`` is
    the failure mode that matters here.
    """
    gm_a = _free_fall_graph()
    gm_b = _free_fall_graph()

    SurrogateValidator.compare_graphs(gm_a, gm_b, N_STEPS, "ball")
    after_first = (_snapshot(gm_a), _snapshot(gm_b))
    SurrogateValidator.compare_graphs(gm_a, gm_b, N_STEPS, "ball")
    after_second = (_snapshot(gm_a), _snapshot(gm_b))

    for label, first, second in (("gm_physics", after_first[0], after_second[0]),
                                 ("gm_surrogate", after_first[1], after_second[1])):
        assert not _identical(first, second), (
            f"a second SurrogateValidator.compare_graphs left {label} "
            f"where the first did: it restarted from the initial "
            f"condition instead of continuing"
        )
        assert _velocity_of(second) < _velocity_of(first), (
            f"a second SurrogateValidator.compare_graphs did not carry "
            f"{label} further than the first"
        )

    reference = _free_fall_graph()
    reference.run_scan_with_history(2 * N_STEPS)
    assert jnp.allclose(
        after_second[0]["ball"]["velocity"],
        jnp.asarray(_velocity(reference)),
        rtol=1e-5, atol=1e-6,
    ), (
        f"two compare_graphs calls of {N_STEPS} did not reach the state "
        f"one rollout of {2 * N_STEPS} reaches on a fresh graph: "
        f"{_velocity_of(after_second[0])} vs {_velocity(reference)}"
    )


# ------------------------------------------------------------------
# The contract is documented, not only enforced
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "owner, name",
    [
        (DatasetGenerator, "from_graph"),
        (SurrogateValidator, "compare_graphs"),
    ],
)
def test_surrogate_entry_point_documents_that_it_advances_state(owner, name):
    """Both say so, so a reader need not go find the call site.

    The behaviour is pinned above either way; this pins that deleting
    the Notes section would not be silent.
    """
    doc = getattr(owner, name).__doc__ or ""
    # The same marker phrase the six GraphManager entry points use, so
    # one grep finds every mutating run path in the tree.
    assert "**Stateful:" in doc and "state is advanced" in doc, (
        f"{owner.__name__}.{name} advances the graph it is given "
        f"(pinned by the tests above) but its docstring does not carry "
        f"the '**Stateful: ... state is advanced**' marker the "
        f"GraphManager entry points use"
    )
