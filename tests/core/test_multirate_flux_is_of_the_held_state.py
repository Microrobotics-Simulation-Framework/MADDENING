"""A slow node's flux is the flux of the state the node holds.

On a multi-rate graph a node whose rate divider is above one fires on one
base step in ``divider`` and holds its state on the others.  A reader of
one of its STATE fields sees the held state.  A reader of one of its
FLUXES (a key ``compute_boundary_fluxes`` returns) must see the same
thing: the hook is called on every base step with the state the node
holds after that step and the boundary inputs resolved at that step.

The hook used to be called on the result of ``update`` before the step
decided whether to keep it, so between firings a reader received the flux
of a state the graph never held, and a geometry the node holds (an edge
with ``geometry=("target", g)``) was read from that discarded state.

Every model here is integer-valued in float32, so each comparison is
exact on any JAX version; the reference is a NumPy float64 simulation
written from the documented rule, not from the step.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

_BASE_DT = 0.01
_N_STEPS = 9


class _Counter(SimulationNode):
    """``x`` grows by one per firing; the flux ``f`` is ``x``."""

    def initial_state(self):
        return {"x": jnp.zeros(())}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] + 1.0}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"f": state["x"]}


class _Reader(SimulationNode):
    def __init__(self, name, timestep, shape=()):
        super().__init__(name, timestep)
        self._shape = shape

    def initial_state(self):
        return {"got": jnp.zeros(self._shape)}

    def boundary_input_spec(self):
        return {"in": BoundaryInputSpec(shape=self._shape)}

    def update(self, state, boundary_inputs, dt):
        return {"got": boundary_inputs["in"]}


def _trajectory(gm, fields, n_steps=_N_STEPS):
    rows = []
    for _ in range(n_steps):
        gm.step()
        rows.append([np.asarray(gm.get_node_state(n)[f]).ravel()[0]
                     for n, f in fields])
    return np.asarray(rows, dtype=np.float64)


def test_a_reader_of_a_slow_nodes_flux_receives_the_flux_of_the_held_state():
    gm = GraphManager()
    gm.add_node(_Counter("slow", 2 * _BASE_DT))
    gm.add_node(_Reader("reader", _BASE_DT))
    gm.add_edge("slow", "reader", "f", "in")
    gm.compile()
    assert gm.rate_dividers == {"slow": 2, "reader": 1}
    got = _trajectory(gm, [("slow", "x"), ("reader", "got")])
    # The node fires on base steps 0, 2, 4, ...
    np.testing.assert_array_equal(got[:, 0], [1, 1, 2, 2, 3, 3, 4, 4, 5])
    # The flux is ``x``: the reader holds exactly what the node holds.
    np.testing.assert_array_equal(got[:, 1], got[:, 0])


_N_CELLS = 12


class _Lattice(SimulationNode):
    """A field whose value at a point is the point's coordinate."""

    def initial_state(self):
        return {"x": jnp.arange(float(_N_CELLS))}

    def update(self, state, boundary_inputs, dt):
        return state


class _Markers(SimulationNode):
    """Moves one cell per firing; its flux is the sample it was handed."""

    def initial_state(self):
        return {"pos": jnp.full((1, 1), 0.25)}

    def boundary_input_spec(self):
        return {"sampled": BoundaryInputSpec(shape=(1,))}

    def update(self, state, boundary_inputs, dt):
        return {"pos": state["pos"] + 1.0}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        # ``compile`` probes the hook's keys with no inputs.
        return {"f": boundary_inputs.get("sampled", jnp.zeros(1))}


def test_a_slow_nodes_flux_hook_reads_its_geometry_from_the_held_state():
    gm = GraphManager()
    gm.add_node(_Lattice("grid", _BASE_DT))
    gm.add_node(_Markers("markers", 2 * _BASE_DT))
    gm.add_node(_Reader("reader", _BASE_DT, shape=(1,)))
    gm.add_edge(
        "grid", "markers", "x", "sampled",
        mapping=multilinear_grid_mapping(
            [0.0], [1.0], [_N_CELLS], n_points=1, mode="consistent"),
        geometry=("target", "pos"))
    gm.add_edge("markers", "reader", "f", "in")
    gm.compile()
    got = _trajectory(gm, [("markers", "pos"), ("reader", "got")])
    np.testing.assert_array_equal(
        got[:, 0], [1.25, 1.25, 2.25, 2.25, 3.25, 3.25, 4.25, 4.25, 5.25])
    # The field is the coordinate, so a sample IS the position it was
    # taken at: the hook's inputs are resolved at the position held.
    np.testing.assert_array_equal(got[:, 1], got[:, 0])


# ---------------------------------------------------------------------------
# The rule against a NumPy reference, over dividers, which node is slow
# and the order the nodes were added in.
#
#   driver.t  -> producer.u     (state edge; the driver fires every step)
#   producer.f -> consumer.in   (flux edge)
#
# producer: x <- x + b * u + 1;   f = c * x + u   (state AND input)
# consumer: y <- y + in;  got <- in
# ---------------------------------------------------------------------------

class _Driver(SimulationNode):
    def initial_state(self):
        return {"t": jnp.zeros(())}

    def update(self, state, boundary_inputs, dt):
        return {"t": state["t"] + 1.0}


class _Producer(SimulationNode):
    def initial_state(self):
        return {"x": jnp.zeros(()), "b": jnp.ones(()), "c": jnp.ones(())}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=())}

    def update(self, state, boundary_inputs, dt):
        return {**state,
                "x": state["x"] + state["b"] * boundary_inputs["u"] + 1.0}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        u = boundary_inputs.get("u", jnp.zeros(()))
        return {"f": state["c"] * state["x"] + u}


class _Consumer(SimulationNode):
    def initial_state(self):
        return {"y": jnp.zeros(()), "got": jnp.zeros(())}

    def boundary_input_spec(self):
        return {"in": BoundaryInputSpec(shape=())}

    def update(self, state, boundary_inputs, dt):
        return {"y": state["y"] + boundary_inputs["in"],
                "got": boundary_inputs["in"]}


@functools.lru_cache(maxsize=None)
def _chain(divider, slow, reverse):
    dts = {"driver": _BASE_DT, "producer": _BASE_DT, "consumer": _BASE_DT}
    dts[slow] = divider * _BASE_DT
    build = {
        "driver": lambda: _Driver("driver", dts["driver"]),
        "producer": lambda: _Producer("producer", dts["producer"]),
        "consumer": lambda: _Consumer("consumer", dts["consumer"]),
    }
    order = ["driver", "producer", "consumer"]
    gm = GraphManager()
    for name in (reversed(order) if reverse else order):
        gm.add_node(build[name]())
    edges = [("driver", "producer", "t", "u"),
             ("producer", "consumer", "f", "in")]
    for edge in (reversed(edges) if reverse else edges):
        gm.add_edge(*edge)
    gm.compile()
    assert gm.rate_dividers[slow] == divider
    return gm


def _reference(divider, slow, x, t, y, b, c, n_steps):
    """The documented rule in float64.

    At base step ``n`` every node is visited in schedule order.  A node
    fires when ``n % divider == 0`` and holds its state otherwise; a
    reader of a state field or of a flux reads what its source holds
    after the source's visit, and the flux hook is evaluated on that
    held state with the inputs resolved at this base step.
    """
    div = {"driver": 1, "producer": 1, "consumer": 1}
    div[slow] = divider
    got = 0.0
    rows = []
    for n in range(n_steps):
        if n % div["driver"] == 0:
            t = t + 1.0
        if n % div["producer"] == 0:
            x = x + b * t + 1.0
        flux = c * x + t
        if n % div["consumer"] == 0:
            y, got = y + flux, flux
        rows.append((t, x, y, got))
    return np.asarray(rows, dtype=np.float64)


_small = st.integers(min_value=-6, max_value=6)


@pytest.mark.parametrize("reverse", [False, True], ids=["as_read", "reversed"])
@pytest.mark.parametrize("slow", ["producer", "consumer"])
@pytest.mark.parametrize("divider", [2, 3, 4])
@given(x=_small, t=_small, y=_small, b=_small, c=_small)
def test_a_flux_reader_follows_the_held_state_at_every_base_step(
        divider, slow, reverse, x, t, y, b, c):
    gm = _chain(divider, slow, reverse)
    gm.reset_state()
    gm.set_node_state("driver", {"t": jnp.asarray(float(t))})
    gm.set_node_state("producer", {
        "x": jnp.asarray(float(x)), "b": jnp.asarray(float(b)),
        "c": jnp.asarray(float(c))})
    gm.set_node_state("consumer", {
        "y": jnp.asarray(float(y)), "got": jnp.zeros(())})
    got = _trajectory(gm, [("driver", "t"), ("producer", "x"),
                           ("consumer", "y"), ("consumer", "got")])
    want = _reference(divider, slow, float(x), float(t), float(y),
                      float(b), float(c), _N_STEPS)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("divider", [2, 3])
def test_the_scanned_and_batched_steps_publish_the_held_flux_too(divider):
    """``run_scan_with_history`` and a ``vmap`` over phases run the same rule."""
    gm = GraphManager()
    gm.add_node(_Counter("slow", divider * _BASE_DT))
    gm.add_node(_Reader("reader", _BASE_DT))
    gm.add_edge("slow", "reader", "f", "in")
    gm.compile()
    _, history = gm.run_scan_with_history(_N_STEPS)
    np.testing.assert_array_equal(
        np.asarray(history["reader"]["got"]),
        np.asarray(history["slow"]["x"]))
    np.testing.assert_array_equal(
        np.asarray(history["slow"]["x"]),
        [1 + n // divider for n in range(_N_STEPS)])

    # One batch element per phase of the divider: each holds or fires by
    # its own step count.
    gm.reset_state()
    state = gm._state
    step = gm._raw_step_fn
    phases = jnp.arange(divider)

    def at_phase(phase):
        meta = {**state["_meta"],
                "step_count": state["_meta"]["step_count"] + phase}
        out = step({**state, "_meta": meta}, {})
        return out["slow"]["x"], out["reader"]["got"]

    x, got = jax.vmap(at_phase)(phases)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(x))
    np.testing.assert_array_equal(np.asarray(x), [1.0] + [0.0] * (divider - 1))
