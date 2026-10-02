"""The live runner and the state relays clock a run by what its steps advanced.

``RealtimeRunner`` paces to the wall clock and reports ``sim_time``; the
``StateRelay`` behind the REST server's WebSocket frames and the ZMQ
``NetworkRelay`` stamp each snapshot with a simulated time.  All three
used ``gm.timestep``, read once, and ``gm.timestep`` was the GCD of the
nodes' own timesteps.  On a graph with a sub-cycling coupling group that is
shorter than a step (MADD-ANO-096): a group of 0.01 and 0.02 steps by
0.02, and every clock ran at half the real rate -- after one wall-clock
second at ``time_scale=100`` the runner said 50 s and the graph's own
clocks read 100 s.  A value read once also goes stale when the graph is
edited between runs.

The probe is a node that counts its own updates, and the runner is driven
for an exact number of steps (a command receiver pauses it), so the
expected clock is ``count * timestep`` of a node that fires every step:
the sum of the real step advances.
"""

from __future__ import annotations

import json
import os
import threading
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.viz.relay import StateRelay
from maddening.viz.renderer import GraphInfo
from maddening.viz.runner import RealtimeRunner


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


def _subcycled_group(gm, fine, coarse, fine_dt, coarse_dt):
    gm.add_node(_Counter(fine, fine_dt))
    gm.add_node(_Counter(coarse, coarse_dt))
    gm.add_edge(fine, coarse, "x", "u")
    gm.add_edge(coarse, fine, "x", "u")
    gm.add_coupling_group([fine, coarse], max_iterations=20, tolerance=1e-6,
                          subcycling=True)


def _subcycled():
    gm = GraphManager()
    _subcycled_group(gm, "fine", "coarse", 0.01, 0.02)    # steps by 0.02
    return gm


def _multirate():
    gm = GraphManager()
    gm.add_node(_Counter("fast", 0.01))
    gm.add_node(_Counter("slow", 0.02))
    gm.add_edge("fast", "slow", "x", "u")
    return gm


def _mixed():
    gm = GraphManager()
    _subcycled_group(gm, "fine", "coarse", 0.005, 0.02)   # scheduled at 0.02
    gm.add_node(_Counter("ref", 0.01))
    gm.add_node(_Counter("slow", 0.03))
    gm.add_edge("coarse", "ref", "x", "u")
    gm.add_edge("ref", "slow", "x", "u")
    return gm                                              # steps by 0.01


#: kind -> (builder, a node that fires on every step, the step).
GRAPHS = {
    "subcycled": (_subcycled, "coarse", 0.02),
    "multirate": (_multirate, "fast", 0.01),
    "mixed": (_mixed, "ref", 0.01),
}


def _count(gm, node):
    return int(gm.get_node_state(node)["n"])


class _PauseAfter:
    """A command receiver that pauses the runner after ``n`` frames.

    The runner asks for commands once per frame, before the frame's steps;
    with ``steps_per_frame=1`` the run therefore takes exactly ``n`` steps.
    """

    def __init__(self):
        self.runner = None
        self.paused = threading.Event()
        self.n = 0
        self.calls = 0

    def arm(self, n):
        self.n, self.calls = n, 0
        self.paused.clear()

    def latest_commands(self):
        self.calls += 1
        if self.calls > self.n:
            self.runner.pause()
            self.paused.set()
        return None


def _runner(gm, relay):
    receiver = _PauseAfter()
    runner = RealtimeRunner(gm, relay, time_scale=1e6, steps_per_frame=1,
                            command_receiver=receiver)
    receiver.runner = runner
    return runner, receiver


def _run(runner, receiver, n):
    receiver.arm(n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # a disconnected-node notice
        runner.start()
    assert receiver.paused.wait(120), "the runner never reached its last frame"
    assert runner.stop(timeout=30), "the runner thread did not stop"


@pytest.mark.parametrize("kind", sorted(GRAPHS))
def test_the_runner_and_the_relay_clock_the_real_step_advance(kind):
    build, ref, step = GRAPHS[kind]
    gm = build()
    relay = StateRelay()
    relay.attach(gm)
    runner, receiver = _runner(gm, relay)
    _run(runner, receiver, 6)
    assert _count(gm, ref) == 6
    expected = _count(gm, ref) * gm._nodes[ref].timestep    # noqa: SLF001
    assert expected == pytest.approx(6 * step)
    assert runner.sim_time == pytest.approx(expected, rel=1e-9)
    sim_time, snapshot = relay.latest_snapshot()
    assert snapshot is not None
    assert sim_time == pytest.approx(expected, rel=1e-9)


def test_the_clocks_keep_time_when_the_graph_is_edited_between_runs():
    """Five steps of the sub-cycled pair (0.02 each), then a node at 0.005
    joins and the graph steps by 0.005 with the group every fourth step.
    The runner used to read the step once per run and the relay once at
    attach time, so the second run was clocked at the first run's step."""
    gm = _subcycled()
    relay = StateRelay()
    relay.attach(gm)
    runner, receiver = _runner(gm, relay)
    _run(runner, receiver, 5)
    gm.add_node(_Counter("fast", 0.005))
    gm.add_edge("coarse", "fast", "x", "u")
    _run(runner, receiver, 8)
    assert _count(gm, "fast") == 8
    expected = 5 * 0.02 + 8 * 0.005
    # the group's own clock agrees: it fired on steps 0 and 4 of the second run
    assert _count(gm, "coarse") * 0.02 == pytest.approx(expected)
    assert runner.sim_time == pytest.approx(expected, rel=1e-9)
    assert relay.latest_snapshot()[0] == pytest.approx(expected, rel=1e-9)


def test_relay_stride_reports_the_time_of_the_captured_step():
    """With ``stride=3`` a snapshot is taken every third step and stamped
    with that step's time, the sum of every step before it."""
    gm = _subcycled()
    relay = StateRelay(stride=3)
    relay.attach(gm)
    gm.run(7)
    sim_time, _ = relay.latest_snapshot()
    assert sim_time == pytest.approx(6 * 0.02, rel=1e-9)


def test_the_network_relay_stamps_each_message_with_the_real_advance():
    pytest.importorskip("zmq", reason="the ZMQ relay needs pyzmq")
    from maddening.viz.network import NetworkRelay

    class _Recorder:
        def __init__(self, socket):
            self.socket, self.sent = socket, []

        def send(self, payload, flags=0):
            self.sent.append(json.loads(payload))

        def close(self):
            self.socket.close()

    relay = NetworkRelay(address="tcp://127.0.0.1:*")
    relay._socket = _Recorder(relay._socket)        # noqa: SLF001
    try:
        gm = _subcycled()
        relay.attach(gm)
        gm.run(4)
        # a node at 0.005 joins: the next two steps advance 0.005 each
        gm.add_node(_Counter("fast", 0.005))
        gm.add_edge("coarse", "fast", "x", "u")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.run(2)
        times = [m["t"] for m in relay._socket.sent]  # noqa: SLF001
        assert times == pytest.approx([0.02, 0.04, 0.06, 0.08, 0.085, 0.09], rel=1e-9)
    finally:
        relay.close()


def test_a_server_reset_restarts_the_frames_clock():
    """The REST server's WebSocket frames carry the relay's ``sim_time``:
    after ``POST /sim/run`` it is the run's simulated time, and ``POST
    /sim/reset`` sets it back to zero with the state -- and publishes the
    reset state at that time, so the streams show it at once."""
    gm = _subcycled()
    gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    resp = client.post("/sim/run", params={"n_steps": 5})
    assert resp.status_code == 200, resp.text
    assert server.relay.latest_snapshot()[0] == pytest.approx(5 * 0.02, rel=1e-9)
    reset = client.post("/sim/reset")
    assert reset.status_code == 200
    sim_time, snapshot = server.relay.latest_snapshot()
    assert sim_time == 0.0 and server.relay.step_count == 0
    assert {node: {f: float(v) for f, v in fields.items()}
            for node, fields in snapshot.items()} == {
        node: {f: float(v) for f, v in fields.items()}
        for node, fields in reset.json()["state"].items() if node != "_meta"}
    assert client.post("/sim/run", params={"n_steps": 2}).status_code == 200
    assert server.relay.latest_snapshot()[0] == pytest.approx(2 * 0.02, rel=1e-9)


def test_the_renderers_graph_info_carries_the_step():
    assert GraphInfo.from_graph_manager(_subcycled()).timestep == pytest.approx(0.02)
