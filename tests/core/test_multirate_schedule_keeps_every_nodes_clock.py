"""A multi-rate schedule keeps every node's clock, or the graph does not compile.

The base step of a multi-rate graph is the GCD of its scheduled timesteps
and each node is stepped every ``round(dt / base)``-th base step.  The GCD
takes a remainder at or below ``1e-9`` of the largest timestep for float
noise, so a timestep smaller than that *was* the noise: nodes at ``1.0`` and
``1e-10`` got a base step of ``1.0`` and rate dividers of 1 and 0, the fast
node was stepped once per base step with its own ``1e-10``, and after four
steps its clock read ``4e-10`` against the graph's ``4.0`` -- add, compile,
validate and run all succeeding, in process and over REST.  A little closer
together the divider was merely wrong (``1.0`` and ``1.5e-9``: a base of
``1e-9`` and a divider of 2, a clock a third fast).

``compile()`` now refuses such a schedule (``_graph_specs._rate_dividers``),
naming the two nodes and their timesteps; the dividers of every schedule it
keeps are the ones it computed before.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest
from hypothesis import given
from hypothesis import strategies as st

from maddening.core._graph_specs import _RATE_RTOL, _multi_gcd, _rate_dividers
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes import BallNode


class _Clock(SimulationNode):
    """Accumulates the ``dt`` it is handed and counts its updates."""

    def halo_width(self) -> dict[int, int]:
        return {}

    def initial_state(self) -> dict:
        return {"t": jnp.zeros(()), "n": jnp.zeros((), jnp.int32)}

    def update(self, state: dict, boundary_inputs: dict, dt: float) -> dict:
        return {"t": state["t"] + dt, "n": state["n"] + 1}


def _graph(slow: float, fast: float) -> GraphManager:
    gm = GraphManager()
    gm.add_node(_Clock("slow", timestep=slow))
    gm.add_node(_Clock("fast", timestep=fast))
    return gm


#: ``(slow, fast)`` no schedule keeps: the fast timestep is below 1e-9 of
#: the slow one or equal to it (a divider of 0), or the common step found
#: leaves a divider that does not reproduce it (1 for a step of 1.5).
REFUSED = [(1.0, 1e-10), (0.01, 2e-12), (0.001, 5e-13), (1.0, 1e-9), (1.0, 1.5e-9),
           (0.0001, 1.1e-12)]
#: ... and what compiles, with the dividers it always had.
KEPT = [((1.0, 3e-9), {"slow": 999999993, "fast": 3}),
        ((0.002, 0.001), {"slow": 2, "fast": 1}),
        ((0.3, 0.1), {"slow": 3, "fast": 1}),
        ((2e-9, 1e-9), {"slow": 2, "fast": 1}),
        ((1.0, 2e-9), {"slow": 500000016, "fast": 1})]


@pytest.mark.parametrize("slow, fast", REFUSED)
def test_a_schedule_that_would_not_keep_a_nodes_clock_does_not_compile(slow, fast):
    gm = _graph(slow, fast)
    with pytest.raises(ValueError) as refused:
        gm.compile()
    text = str(refused.value)
    assert "'fast'" in text and "'slow'" in text, text
    assert repr(fast) in text and repr(slow) in text, text
    # ... validate says the same, and no longer reports a divider of 0
    issues = gm.validate()
    assert [i for i in issues if i.startswith("ERROR") and "'fast'" in i], issues
    assert not [i for i in issues if "rate dividers" in i], issues
    with pytest.raises(ValueError, match="cannot be scheduled together"):
        gm.step()
    # the refusal is of the schedule, not of the graph: without the fast
    # node it compiles and runs
    gm.remove_node("fast")
    gm.compile()
    gm.step()
    assert float(gm._state["slow"]["t"]) == pytest.approx(slow)


@pytest.mark.parametrize("timesteps, dividers", KEPT)
def test_a_schedule_that_is_kept_has_the_dividers_it_always_had(timesteps, dividers):
    slow, fast = timesteps
    gm = _graph(slow, fast)
    gm.compile()
    assert gm._rate_dividers == dividers
    base = _multi_gcd(sorted(timesteps))
    assert gm.timestep == base
    assert dividers == {"slow": round(slow / base), "fast": round(fast / base)}
    assert not [i for i in gm.validate() if i.startswith("ERROR")]


def test_the_nodes_of_a_kept_schedule_read_the_graphs_clock():
    gm = GraphManager()
    for name, dt in (("a", 0.01), ("b", 0.002), ("c", 0.005)):
        gm.add_node(_Clock(name, timestep=dt))
    gm.compile()
    assert gm._rate_dividers == {"a": 10, "b": 2, "c": 5}
    for _ in range(20):
        gm.step()
    for name, fired in (("a", 2), ("b", 10), ("c", 4)):
        assert int(gm._state[name]["n"]) == fired
        assert float(gm._state[name]["t"]) == pytest.approx(20 * gm.timestep, rel=1e-5)


_MANTISSAS = (1.0, 1.1, 1.25, 1.5, 2.0, 2.5, 3.0, 3.3, 4.0, 5.0, 6.0, 6.25, 7.0, 7.5, 8.0, 9.0)
_TIMESTEPS = st.builds(lambda m, e: float(f"{m}e{e}"), st.sampled_from(_MANTISSAS),
                       st.integers(-13, 1))


@given(st.lists(_TIMESTEPS, min_size=2, max_size=4, unique=True))
def test_any_set_of_timesteps_is_scheduled_on_one_clock_or_refused(timesteps):
    """Timesteps up to fourteen decades apart: the schedule either keeps
    every node's clock to ``_RATE_RTOL`` with a divider of at least one --
    and then the dividers are ``round(dt / base)``, as before -- or it is a
    ``ValueError`` naming the node it cannot schedule, the slowest node and
    both timesteps."""
    scheduled = {f"n{i}": dt for i, dt in enumerate(timesteps)}
    base = _multi_gcd(sorted(timesteps))
    try:
        dividers = _rate_dividers(scheduled)
    except ValueError as exc:
        text = str(exc)
        slowest = max(scheduled, key=scheduled.get)
        assert repr(slowest) in text and repr(scheduled[slowest]) in text, text
        named = [n for n, dt in scheduled.items()
                 if n != slowest and repr(n) in text and repr(dt) in text]
        assert named, text
        bad = scheduled[named[0]]
        assert round(bad / base) < 1 or abs(round(bad / base) * base - bad) > _RATE_RTOL * bad
    else:
        assert dividers == {n: round(dt / base) for n, dt in scheduled.items()}
        assert list(dividers) == list(scheduled)
        for name, dt in scheduled.items():
            assert dividers[name] >= 1
            assert abs(dividers[name] * base - dt) <= _RATE_RTOL * dt
        if max(timesteps) / min(timesteps) < 1e3:
            # nothing a few decades apart is refused: the GCD is exact there
            assert all(abs(d * base - scheduled[n]) <= 1e-9 * scheduled[n]
                       for n, d in dividers.items())


@given(st.lists(_TIMESTEPS, min_size=2, max_size=4, unique=True))
def test_timesteps_within_three_decades_are_never_refused(timesteps):
    lo = min(timesteps)
    close = [dt for dt in timesteps if dt / lo < 1e3]
    if len(close) > 1:
        _rate_dividers({f"n{i}": dt for i, dt in enumerate(close)})


# ---------------------------------------------------------------------------
# Over REST
# ---------------------------------------------------------------------------


def _post(client, name: str, timestep: float):
    return client.post("/graph/nodes", json={"type": "BallNode", "name": name,
                                             "timestep": timestep, "params": {}})


def test_a_node_whose_timestep_the_graph_cannot_schedule_is_not_added():
    """Over REST the add, the compile, the validation and the run of nodes
    at 1.0 and 1e-10 all succeeded.  The node is refused where its
    timestep enters, naming both nodes, and the graph is as it was."""
    from tests.property.rest_oracle import assert_nothing_changed, serve, snapshot

    served = serve(registry={"BallNode": BallNode})
    try:
        client = served.client
        assert _post(client, "slow", 1.0).status_code == 201
        assert client.post("/sim/step").status_code == 200
        before = snapshot(served)
        for dt in (1e-10, 1e-9, 1.5e-9, 2 ** 63, 1e-50):
            resp = _post(client, "fast", dt)
            assert resp.status_code == 400, (dt, resp.text)
            detail = resp.json()["detail"]
            assert "'fast'" in detail and "'slow'" in detail and repr(float(dt)) in detail, detail
            assert_nothing_changed(before, snapshot(served), f"the refused timestep {dt}")
        assert _post(client, "fast", 0.25).status_code == 201
        assert client.post("/sim/step").status_code == 200
        assert served.gm._rate_dividers == {"slow": 4, "fast": 1}
    finally:
        served.close()


def test_a_removal_that_would_leave_timesteps_with_no_schedule_is_refused():
    """The common step is found afresh for the timesteps that are left,
    and four decades and more apart it can be one that no longer keeps a
    remaining node's clock: these four nodes compile, and three of them
    without ``d`` do not."""
    from tests.property.rest_oracle import assert_nothing_changed, serve, snapshot

    timesteps = {"a": 0.00011, "b": 6e-05, "c": 1.1e-10, "d": 2.5e-08}
    _rate_dividers(timesteps)
    with pytest.raises(ValueError):
        _rate_dividers({n: dt for n, dt in timesteps.items() if n != "d"})
    served = serve(registry={"BallNode": BallNode})
    try:
        client = served.client
        for name in ("a", "b", "d", "c"):
            assert _post(client, name, timesteps[name]).status_code == 201, name
        assert client.post("/graph/compile").status_code == 200
        before = snapshot(served)
        resp = client.delete("/graph/nodes/d")
        assert resp.status_code == 400, resp.text
        assert "cannot be removed" in resp.json()["detail"] and "'c'" in resp.json()["detail"]
        assert_nothing_changed(before, snapshot(served), "the refused removal")
        assert client.delete("/graph/nodes/c").status_code == 200
        assert client.delete("/graph/nodes/d").status_code == 200
        assert client.post("/graph/compile").status_code == 200
    finally:
        served.close()


def test_the_routes_say_why_a_graph_built_with_such_timesteps_cannot_run():
    """A graph built in process and served: the compile, the validation
    and the step say why the two nodes cannot run together, and the node
    that is the reason can be removed."""
    from tests.property.rest_oracle import serve

    gm = GraphManager()
    gm.add_node(BallNode("slow", 1.0))
    gm.add_node(BallNode("fast", 1e-10))
    served = serve(gm, registry={"BallNode": BallNode})
    try:
        client = served.client
        for route in ("/graph/compile", "/sim/step", "/sim/run?n_steps=2"):
            resp = client.post(route)
            assert resp.status_code == 400, (route, resp.text)
            detail = str(resp.json()["detail"])
            assert "'fast'" in detail and "'slow'" in detail and "1e-10" in detail, detail
        issues = client.post("/graph/validate").json()["issues"]
        assert [i for i in issues if i.startswith("ERROR") and "'fast'" in i], issues
        assert client.delete("/graph/nodes/fast").status_code == 200
        assert client.post("/sim/step").status_code == 200
    finally:
        served.close()
