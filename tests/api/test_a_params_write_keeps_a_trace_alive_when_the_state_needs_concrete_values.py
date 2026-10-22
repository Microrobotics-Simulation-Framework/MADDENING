"""A served node whose state needs concrete values keeps a whole JAX trace alive per params write.

``PUT /graph/params`` and ``POST /checkpoint/load`` ask, before anything
is built, whether the state the node would build with the written values
has the running layout: ``_abstract_initial_state`` in
``maddening/api/server.py`` evaluates ``node.initial_state()`` under
``jax.eval_shape`` and catches what it raises.  A node whose
``initial_state`` needs a concrete value -- the built-in
``WaveletAdaptiveNode`` is one -- raises there every time (the answer is
"unknown", and the write goes on to the concrete checks).  The exception
leaves the trace JAX opened for the evaluation, and JAX never frees a
trace that raised: ``core.ensure_no_leaks`` invalidates the trace after a
bare ``yield``, which an exception skips, and the finalizer the trace
registered for a staged constant keeps it reachable from
``weakref.finalize``'s registry.  Each accepted write therefore leaves one
whole trace alive, and nothing frees it: not ``gc.collect()``, not
dropping the server and the graph
(MADD-ANO-NEW-records-known-anomalies-0-4-2-4, open).  The replies and
the states are right; only the memory grows.

Measured (CPU; jax 0.10.2, 0.11.0 and 0.11.2, the same counts): a served
``WaveletAdaptiveNode`` keeps 476 tracers per ``PUT`` and 476 per
checkpoint load, linearly (28,560 after 60 writes; the process grew from
469 MB to 706 MB); the small node below keeps 63 per write, and the same
node with a state that evaluates abstractly keeps none.

What is held here, per push on the small node (the wavelet node costs
three seconds to build and add, so its own cases are behind the slow
mark): the premise for both routes as what happens today, the control,
and a strict xfail of the claim "an accepted write keeps no trace alive".
The traces the tests leave stay for the rest of the process (under 4 MB).
"""

from __future__ import annotations

import gc
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.adaptive import WaveletAdaptiveNode
from tests._loopback_client import LoopbackTestClient as TestClient

_LABEL = "MADD-ANO-NEW-records-known-anomalies-0-4-2-4"

#: Fewer live tracers than one kept trace of the small node holds (63 on
#: every JAX measured), and more than a write that keeps nothing leaves
#: (none: the count is flat between such writes).
ONE_SMALL_TRACE = 20
#: The same for ``WaveletAdaptiveNode`` (476 on every JAX measured).
ONE_WAVELET_TRACE = 100

_TABLE = np.linspace(0.0, 1.0, 16)


class ConcreteStart(SimulationNode):
    """A node whose ``initial_state`` reads a value it computes: it builds
    its state when run, and raises under abstract evaluation, after
    staging a non-scalar constant and twenty operations on it."""

    reads_a_concrete_value = True

    def __init__(self, name: str, timestep: float, gain: float = 1.0):
        super().__init__(name, timestep, gain=gain)

    def initial_state(self):
        w = jnp.asarray(_TABLE)
        for _ in range(20):
            w = jnp.sin(w * self.params["gain"])
        n = 2 + int(w.sum() > 0) if self.reads_a_concrete_value else 3
        return {"x": jnp.zeros(n) + w[:1]}

    def update(self, state, boundary_inputs, dt, *, params=None):
        gain = (self.params if params is None else params)["gain"]
        return {"x": state["x"] + dt * gain}


class AbstractStart(ConcreteStart):
    """The control: the same node, with a state that evaluates abstractly."""

    reads_a_concrete_value = False


#: The parameter written, and the two values written in turn.
_WRITES = {ConcreteStart: ("gain", (2.0, 3.0)), AbstractStart: ("gain", (2.0, 3.0)),
           WaveletAdaptiveNode: ("sigma", (0.12, 0.08))}


def _accepted(response):
    assert response.status_code == 200, (response.status_code, response.text[:300])
    return response.json()


def _live_tracers() -> int:
    gc.collect()
    return sum(isinstance(o, jax.core.Tracer) for o in gc.get_objects())


@pytest.fixture
def served(tmp_path):
    """``served(node, compiled=False)``: *node* served, and its client, with
    one write already made (whatever a first write sets up once is not
    counted).  *compiled*: the graph compiled and a checkpoint of it saved,
    for the load.  The clients are closed when the test ends."""
    clients = []

    def make(node, *, compiled=False):
        gm = GraphManager()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gm.add_node(node)
            if compiled:
                gm.compile()
        server = SimulationServer({type(node).__name__: type(node)}, graph_manager=gm,
                                  checkpoint_root=str(tmp_path))
        client = TestClient(server.create_app())
        client.__enter__()
        clients.append(client)
        if compiled:
            _accepted(client.post("/checkpoint/save", params={"path": "c.npz"}))
        key, values = _WRITES[type(node)]
        _accepted(client.put(f"/graph/params/{node.name}", json={"params": {key: values[0]}}))
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


def _kept_by_writes(client, node) -> int:
    """Live tracers added by accepted ``PUT /graph/params`` of the two
    values in turn (three writes; two for the wavelet node), each read back
    as written."""
    key, values = _WRITES[type(node)]
    order = (values[1], values[0]) if isinstance(node, WaveletAdaptiveNode) else (
        values[1], values[0], values[1])
    before = _live_tracers()
    for value in order:
        _accepted(client.put(f"/graph/params/{node.name}", json={"params": {key: value}}))
        assert _accepted(client.get(f"/graph/params/{node.name}"))[key] == pytest.approx(value)
    return _live_tracers() - before


def _kept_by_a_load(client) -> int:
    """Live tracers added by one accepted checkpoint load."""
    before = _live_tracers()
    _accepted(client.post("/checkpoint/load", params={"path": "c.npz"}))
    return _live_tracers() - before


def test_each_accepted_params_write_keeps_a_whole_trace_alive(served):
    """A premise, as what happens today: three accepted writes, each read
    back as written, keep three traces' worth of tracers alive."""
    node = ConcreteStart("c", 0.1)
    assert _kept_by_writes(served(node), node) >= 3 * ONE_SMALL_TRACE


def test_a_checkpoint_load_keeps_a_whole_trace_alive(served):
    """A premise, as what happens today: a checkpoint load is answered as
    a params write, by the same abstract evaluation."""
    assert _kept_by_a_load(served(ConcreteStart("c", 0.1), compiled=True)) >= ONE_SMALL_TRACE


def test_a_node_whose_state_evaluates_abstractly_keeps_nothing(served):
    """The control: the evaluation that does not raise is freed."""
    node = AbstractStart("c", 0.1)
    assert _kept_by_writes(served(node), node) < ONE_SMALL_TRACE


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    f"{_LABEL} (open): every accepted PUT /graph/params or checkpoint load on a served node "
    "whose initial_state raises under jax.eval_shape (WaveletAdaptiveNode: 476 tracers) keeps "
    "a whole JAX trace alive for good"))
def test_an_accepted_params_write_keeps_no_trace_alive(served):
    """The claim that does not hold today."""
    node = ConcreteStart("c", 0.1)
    assert _kept_by_writes(served(node), node) < ONE_SMALL_TRACE


def _wavelet():
    return WaveletAdaptiveNode("wavelet", 1.0, n_levels=3, n_coarse=2, theta=0.4, sigma=0.1)


# Per push: tests/api/test_a_params_write_keeps_a_trace_alive_when_the_state_needs_concrete_values.py::test_each_accepted_params_write_keeps_a_whole_trace_alive
@pytest.mark.slow
def test_a_served_wavelet_node_keeps_a_whole_trace_alive_per_params_write(served):
    """The built-in node: each accepted ``PUT`` keeps one trace of its
    ``initial_state``."""
    node = _wavelet()
    assert _kept_by_writes(served(node), node) >= 2 * ONE_WAVELET_TRACE


# Per push: tests/api/test_a_params_write_keeps_a_trace_alive_when_the_state_needs_concrete_values.py::test_a_checkpoint_load_keeps_a_whole_trace_alive
@pytest.mark.slow
def test_a_checkpoint_load_on_a_wavelet_node_keeps_a_whole_trace_alive(served):
    """The built-in node, compiled: a checkpoint load keeps one trace."""
    assert _kept_by_a_load(served(_wavelet(), compiled=True)) >= ONE_WAVELET_TRACE
