"""A 200 from ``PUT /graph/params`` means the saved graph reloads and runs
what the running graph runs.

``to_dict()`` saves each node as its class and its effective params, and
``from_dict()`` calls the class with them, so a write the route accepts has
to be one the constructor takes -- with every changed key at once and the
live values a save would carry -- and one the running node honours the way
the rebuilt node would.  Four ways it was not, each driven here through the
real server, each checked for "nothing was written" and, where the route
answers 200, for the reload reproducing the running graph:

* a constructor-fixed branch -- ``HeatNode``'s grid (uniform vs
  ``grid_points``) and ``LBMPipeNode``'s single-/multiphase switch
  (``G != 0``) -- flipped by a write the running node did not rebuild;
* a live leaf the constructor refuses (``HeatNode`` above its Fourier limit,
  ``LBMPipeNode`` with ``rho_gas > rho_liquid``), which skipped the
  constructor because only structural keys were asked;
* a non-finite structural value, written and then a 500 from the reply's
  encoder, leaving every later ``GET /graph/params`` a 500;
* a value naming a huge array dimension, refused only after the node and
  its state had been built at that size.

Originally written from the round-2 confirmation audit of the 0.4.0 tree
(reproducer under
``benchmarks/results/audit_040_p4_2/sharding-params/repro_rest_param_writes.py``).
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api import server as server_module
from maddening.api.server import MAX_NODE_PARAM_INT, SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.lbm_pipe import LBMPipeNode


# ---------------------------------------------------------------------------
# Test nodes for the machinery (not the two audited nodes)
# ---------------------------------------------------------------------------


class _BranchOnK(SimulationNode):
    """Picks its update at construction from ``k != 0`` and reads ``k``
    from the injected params inside the branch -- the shape of
    ``LBMPipeNode``'s ``G``: every write of ``k`` is *used*, and the one
    that crosses zero still runs another model than the reload."""

    def __init__(self, name="n", timestep=0.1, k=1.0):
        super().__init__(name, timestep, k=k)
        self._decays = k != 0.0

    def initial_state(self):
        return {"x": jnp.asarray(1.0, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        if self._decays:
            return {"x": state["x"] * (1.0 - dt * p["k"])}
        return {"x": state["x"] + dt}


class _WeightsFromScale(SimulationNode):
    """Builds a weight array from ``scale`` at construction and also reads
    ``scale`` live: the new value changes a constant of the rebuilt node's
    trace, not its text."""

    def __init__(self, name="n", timestep=0.1, scale=1.0):
        super().__init__(name, timestep, scale=scale)
        self._w = np.linspace(0.5, 1.5, 4, dtype=np.float32) * np.float32(scale)

    def initial_state(self):
        return {"x": jnp.ones(4, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] * (1.0 - dt * self._w) + 0.0 * p["scale"]}


class _StartsFromK(SimulationNode):
    """Copies ``k`` into its initial condition at construction and reads
    ``k`` live in the step: the step agrees with the reload, the reset
    state does not."""

    def __init__(self, name="n", timestep=0.1, k=1.0):
        super().__init__(name, timestep, k=k)
        self._x0 = float(k)

    def initial_state(self):
        return {"x": jnp.asarray(self._x0, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] * (1.0 - dt * p["k"])}


class _DrawsNoise(SimulationNode):
    """Draws an unseeded array at construction: it never matches its own
    rebuild, so no write can be blamed for a difference, and a write of the
    live ``k`` it reads is accepted."""

    def __init__(self, name="n", timestep=0.1, k=1.0):
        super().__init__(name, timestep, k=k)
        self._noise = np.random.default_rng().normal(size=3).astype(np.float32)

    def initial_state(self):
        return {"x": jnp.zeros(3, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": state["x"] + dt * (self._noise + p["k"])}


class _Grid(SimulationNode):
    """State ``(nx, ny)``, both read from ``params``; counts every
    construction and every concrete (not traced) state it builds."""

    constructed: list = []
    built: list = []

    def __init__(self, name="g", timestep=0.1, nx=4, ny=4):
        super().__init__(name, timestep, nx=nx, ny=ny)
        type(self).constructed.append((nx, ny))

    def initial_state(self):
        grid = jnp.zeros((self.params["nx"], self.params["ny"]), jnp.float32)
        if not isinstance(grid, jax.core.Tracer):
            type(self).built.append(grid.shape)
        return {"grid": grid}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"grid": state["grid"] + dt}


REGISTRY = {
    "HeatNode": HeatNode,
    "LBMPipeNode": LBMPipeNode,
    "_BranchOnK": _BranchOnK,
    "_WeightsFromScale": _WeightsFromScale,
    "_StartsFromK": _StartsFromK,
    "_DrawsNoise": _DrawsNoise,
    "_Grid": _Grid,
}

N = 12
_X = (np.arange(N) + 0.5) / N
T0 = (300 + 40 * np.sin(2.3 * np.pi * _X)).tolist()
#: Clearly non-uniform, and stable at the rod's dt on both grids.
NONUNIFORM = (np.linspace(0, 1, N) ** 1.6 * 0.9 + 0.05).tolist()

PIPE = dict(nx=8, ny=10, nz=10, pipe_radius=0.8, propeller_x=2,
            propeller_strength=0.0, G=-5.0, rho_liquid=2.0, rho_gas=0.2,
            fill_fraction=0.5)


def _client(gm):
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm)
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _graph(node, *inputs, compile=True):
    gm = GraphManager()
    gm.add_node(node)
    for field in inputs:
        gm.add_external_input(node.name, field)
    if compile:
        gm.compile()
    return gm


def _rod(**kw):
    return _graph(
        HeatNode("rod", 0.05, **{"n_cells": N, "length": 1.0,
                                 "thermal_diffusivity": 1e-3,
                                 "initial_temperature": T0, **kw}),
        "left_temperature", "right_temperature",
    )


def _pipe(compile=False, **kw):
    """Uncompiled unless asked: the route answers the same one compile()
    earlier or later (it probes the node's own pytree), and compiling the
    multiphase step is most of a pipe test's time."""
    return _graph(LBMPipeNode("p", 1.0, **{**PIPE, **kw}), compile=compile)


def _reloaded(gm):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = GraphManager.from_dict(gm.to_dict(), REGISTRY)
        again.compile()
    return again


def _snapshot(gm, name):
    node = gm._nodes[name].node
    live = {k: np.asarray(v).copy()
            for k, v in (gm.params.get("nodes", {}).get(name) or {}).items()}
    return dict(node.params), live, gm._dirty


def _assert_nothing_written(gm, name, before):
    node_params, live, dirty = before
    assert dict(gm._nodes[name].node.params) == node_params
    now = gm.params.get("nodes", {}).get(name) or {}
    assert now.keys() == live.keys()
    for key, value in live.items():
        np.testing.assert_array_equal(np.asarray(now[key]), value)
    assert gm._dirty is dirty


def _assert_reload_runs_the_same(gm, name, field, steps):
    """The saved graph, reloaded, steps to the running graph's numbers."""
    again = _reloaded(gm)
    gm.reset_state()
    running = np.asarray(gm.run_scan(steps)[name][field])
    saved = np.asarray(again.run_scan(steps)[name][field])
    np.testing.assert_array_equal(running, saved)


def _put(gm, name, params):
    return _client(gm).put(f"/graph/params/{name}", json={"params": params})


# ---------------------------------------------------------------------------
# A branch the constructor fixed
# ---------------------------------------------------------------------------


def test_a_grid_written_into_a_uniform_rod_is_refused_and_the_reload_matches():
    """The audited write: 200, echoed by GET, while the running rod stepped
    the variable-dx stencil on its uniform coordinates and the reload ran
    the new grid -- 166 K apart after 200 steps."""
    gm = _rod()
    before = _snapshot(gm, "rod")
    resp = _put(gm, "rod", {"grid_points": NONUNIFORM})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith("grid_points: node 'rod' cannot take")
    _assert_nothing_written(gm, "rod", before)
    assert _client(gm).get("/graph/params/rod").json()["grid_points"] is None
    _assert_reload_runs_the_same(gm, "rod", "temperature", 50)


def test_a_uniform_rod_keeps_its_stencil_when_grid_points_is_written_into_its_params():
    """The node itself, with no route in between: the grid is fixed at
    construction with the coordinates the stencil reads, so a list landing
    in ``params`` (any write surface) cannot switch the step onto the
    variable-dx branch over the stale uniform coordinates."""
    node = HeatNode("rod", 0.05, n_cells=N, length=1.0, thermal_diffusivity=1e-3,
                    initial_temperature=T0)
    state = node.initial_state()
    boundary = {"left_temperature": jnp.float32(300.0),
                "right_temperature": jnp.float32(320.0)}
    before = np.asarray(node.update(state, boundary, 0.05)["temperature"])
    node.params["grid_points"] = NONUNIFORM
    assert node._is_nonuniform is False
    assert node.static_data_deps() == {}
    after = np.asarray(node.update(state, boundary, 0.05)["temperature"])
    np.testing.assert_array_equal(after, before)
    fluxes = node.compute_boundary_fluxes(state, boundary, 0.05)
    node.params["grid_points"] = None
    np.testing.assert_array_equal(
        np.asarray(fluxes["left_heat_flux"]),
        np.asarray(node.compute_boundary_fluxes(state, boundary, 0.05)["left_heat_flux"]),
    )


def test_a_non_uniform_rod_refuses_both_a_new_grid_and_none():
    gm = _rod(grid_points=NONUNIFORM)
    before = _snapshot(gm, "rod")
    for value in (None, (np.asarray(NONUNIFORM) * 0.9).tolist()):
        resp = _put(gm, "rod", {"grid_points": value})
        assert resp.status_code == 400, resp.text
        _assert_nothing_written(gm, "rod", before)
    _assert_reload_runs_the_same(gm, "rod", "temperature", 20)


def test_switching_a_multiphase_pipe_to_single_phase_is_refused_and_the_reload_matches():
    """``G=0`` was used (the multiphase step reads ``G``) and answered 200;
    the running pipe stayed multiphase with no interaction and the save
    reloaded single-phase, 0.645 apart in the tracer after ten steps."""
    gm = _pipe()
    before = _snapshot(gm, "p")
    resp = _put(gm, "p", {"G": 0.0})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("G: node 'p' cannot take a new value")
    assert "reload a node that computes something else" in detail and "_G" in detail
    _assert_nothing_written(gm, "p", before)


def test_a_single_phase_pipe_refuses_a_nonzero_interaction_strength():
    gm = _pipe(G=0.0)
    before = _snapshot(gm, "p")
    resp = _put(gm, "p", {"G": -5.0})
    assert resp.status_code == 400, resp.text
    _assert_nothing_written(gm, "p", before)


@pytest.mark.slow  # two multiphase-pipe compiles and scans: ~8 s on 3 cores
def test_a_compiled_multiphase_pipe_takes_writes_within_its_branch_and_the_reload_matches():
    """On the compiled graph the audit drove: ``G=0`` refused, and each
    in-branch write -- changing an attribute the rebuilt node's
    constructor derives (``_G``, the stale copy ``_tau`` the step never
    reads, nothing at all) -- accepted, with the reload running what the
    running graph runs.  Over-refusal guard for the reload comparison."""
    gm = _pipe(compile=True)
    client = _client(gm)
    before = _snapshot(gm, "p")
    assert client.put("/graph/params/p", json={"params": {"G": 0.0}}).status_code == 400
    _assert_nothing_written(gm, "p", before)
    for write in ({"G": -4.5}, {"tau": 0.9}, {"rho_liquid": 1.8, "rho_gas": 0.3}):
        resp = client.put("/graph/params/p", json={"params": write})
        assert resp.status_code == 200, (write, resp.text)
    _assert_reload_runs_the_same(gm, "p", "tracer", 2)


def test_a_uniform_rod_takes_a_new_length_and_the_reload_matches():
    """``length`` rebuilds the uniform coordinates, which the uniform step
    never reads: accepted, and the reload agrees."""
    gm = _rod()
    resp = _put(gm, "rod", {"length": 1.1})
    assert resp.status_code == 200, resp.text
    _assert_reload_runs_the_same(gm, "rod", "temperature", 50)


def test_a_constructor_derived_branch_is_refused_for_any_node():
    """The machinery, not the pipe: a node that picks its update from a
    live leaf at construction refuses the write that crosses the switch
    and takes one that does not."""
    gm = _graph(_BranchOnK())
    before = _snapshot(gm, "n")
    resp = _put(gm, "n", {"k": 0.0})
    assert resp.status_code == 400, resp.text
    assert "the step traces to a different computation" in resp.json()["detail"]
    _assert_nothing_written(gm, "n", before)
    assert _put(gm, "n", {"k": 2.0}).status_code == 200
    _assert_reload_runs_the_same(gm, "n", "x", 3)


def test_an_array_the_constructor_derives_from_a_live_value_is_refused():
    gm = _graph(_WeightsFromScale())
    before = _snapshot(gm, "n")
    resp = _put(gm, "n", {"scale": 2.0})
    assert resp.status_code == 400, resp.text
    assert "the step closes over different constants" in resp.json()["detail"]
    _assert_nothing_written(gm, "n", before)


def test_an_initial_condition_the_constructor_copied_from_a_live_value_is_refused():
    gm = _graph(_StartsFromK())
    before = _snapshot(gm, "n")
    resp = _put(gm, "n", {"k": 2.0})
    assert resp.status_code == 400, resp.text
    assert "initial_state() builds a different state" in resp.json()["detail"]
    _assert_nothing_written(gm, "n", before)


def test_a_node_that_differs_from_its_own_rebuild_is_not_blamed_on_the_write():
    """Unseeded noise drawn at construction: the running node never matches
    a rebuild, before the write or after it, so the difference is not this
    write's, and a write of the live leaf the step reads is accepted."""
    gm = _graph(_DrawsNoise())
    resp = _put(gm, "n", {"k": 2.0})
    assert resp.status_code == 200, resp.text
    assert float(gm.params["nodes"]["n"]["k"]) == 2.0


# ---------------------------------------------------------------------------
# A live leaf the constructor refuses
# ---------------------------------------------------------------------------


def test_a_live_leaf_above_the_fourier_limit_is_refused_and_the_graph_still_saves():
    """``thermal_diffusivity`` is a live leaf, so the constructor was never
    asked: Fourier number 0.6 was answered 200, and ``from_dict`` then
    raised "timestep ... is unstable" on the saved graph."""
    gm = _graph(HeatNode("rod", 1.0, n_cells=16, length=1.0,
                         thermal_diffusivity=0.2 / 256, initial_temperature=300.0))
    before = _snapshot(gm, "rod")
    resp = _put(gm, "rod", {"thermal_diffusivity": 0.6 / 256})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("thermal_diffusivity: node 'rod' cannot take")
    assert "HeatNode's constructor refuses it" in detail and "unstable" in detail
    _assert_nothing_written(gm, "rod", before)
    _reloaded(gm)
    # The stable side of the limit is still a live write.
    assert _put(gm, "rod", {"thermal_diffusivity": 0.4 / 256}).status_code == 200


def test_a_pipe_gas_density_above_its_liquid_density_is_refused():
    gm = _pipe()
    before = _snapshot(gm, "p")
    resp = _put(gm, "p", {"rho_gas": 3.0})
    assert resp.status_code == 400, resp.text
    assert "rho_liquid must be > rho_gas" in resp.json()["detail"]
    _assert_nothing_written(gm, "p", before)
    _reloaded(gm)


def test_two_values_each_acceptable_alone_are_refused_together():
    """``rho_gas=1.5`` is fine against ``rho_liquid=2.0`` and ``rho_liquid=1.0``
    against ``rho_gas=0.2``; together the save would not load."""
    gm = _pipe()
    before = _snapshot(gm, "p")
    resp = _put(gm, "p", {"rho_gas": 1.5, "rho_liquid": 1.0})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("rho_gas, rho_liquid: node 'p' cannot take these values together")
    _assert_nothing_written(gm, "p", before)
    _reloaded(gm)


def test_the_constructor_is_asked_with_the_live_values_a_save_would_carry():
    """A fit moved ``thermal_diffusivity`` in ``gm.params`` only.  A shorter
    rod is stable against the constructor's value and unstable against the
    fitted one -- and the fitted one is what ``to_dict()`` saves."""
    gm = _graph(HeatNode("rod", 1.0, n_cells=16, length=1.0,
                         thermal_diffusivity=0.1 / 256, initial_temperature=300.0))
    gm.params["nodes"]["rod"]["thermal_diffusivity"] = jnp.float32(0.4 / 256)
    before = _snapshot(gm, "rod")
    resp = _put(gm, "rod", {"length": 0.8})    # Fo 0.156 (ctor alpha), 0.625 (fitted)
    assert resp.status_code == 400, resp.text
    assert "unstable" in resp.json()["detail"]
    _assert_nothing_written(gm, "rod", before)


# ---------------------------------------------------------------------------
# A non-finite value
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("literal, where", [
    ("[0.1, NaN, 0.5, 0.9]", "params.grid_points[1]"),
    ("[0.1, Infinity, 0.5, 0.9]", "params.grid_points[1]"),
    ("[-Infinity, 0.2, 0.5, 0.9]", "params.grid_points[0]"),
    ("[0.1, 0.2, 0.5, 1" + "0" * 400 + "]", "params.grid_points[3]"),
])
def test_a_non_finite_structural_value_is_refused_before_anything_is_written(literal, where):
    """It used to skip the live-leaf checks, be written, and fail in the
    reply's JSON encoder: a 500 after the write, and every later
    ``GET /graph/params/<node>`` a 500 too."""
    gm = _graph(HeatNode("rod", 0.5, n_cells=4, length=1.0, thermal_diffusivity=1e-3))
    before = _snapshot(gm, "rod")
    client = _client(gm)
    resp = client.put(
        "/graph/params/rod",
        content='{"params": {"length": 2.0, "grid_points": ' + literal + '}}',
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == f"{where}: value must be finite"
    _assert_nothing_written(gm, "rod", before)
    assert client.get("/graph/params/rod").status_code == 200
    assert client.get("/graph").status_code == 200


# ---------------------------------------------------------------------------
# A huge dimension
# ---------------------------------------------------------------------------


def test_an_oversized_integer_is_refused_by_the_request_model_before_any_node_is_built(monkeypatch):
    """``n_cells=3e7`` was a 400 only after a 30-million-cell rod and its
    state had been built (+235 MB, 0.5 s), twice over."""
    built = []
    real_init = HeatNode.__init__

    def counting_init(self, *args, **kwargs):
        built.append(kwargs.get("n_cells"))
        real_init(self, *args, **kwargs)

    gm = _graph(HeatNode("rod", 1.0, n_cells=8, length=1.0, thermal_diffusivity=1e-18))
    before = _snapshot(gm, "rod")
    monkeypatch.setattr(HeatNode, "__init__", counting_init)
    resp = _put(gm, "rod", {"n_cells": 3 * MAX_NODE_PARAM_INT})
    assert resp.status_code == 422, resp.text
    assert "integer magnitude must be at most" in resp.text
    assert built == []
    _assert_nothing_written(gm, "rod", before)


def test_a_new_cell_count_is_refused_before_a_rod_of_that_size_is_built(monkeypatch):
    """Under the integer bound, the layout change is still told without
    constructing a rod of the new size."""
    built = []
    real_init = HeatNode.__init__

    def counting_init(self, *args, **kwargs):
        built.append(kwargs.get("n_cells"))
        real_init(self, *args, **kwargs)

    gm = _graph(HeatNode("rod", 1.0, n_cells=8, length=1.0, thermal_diffusivity=1e-18))
    before = _snapshot(gm, "rod")
    monkeypatch.setattr(HeatNode, "__init__", counting_init)
    resp = _put(gm, "rod", {"n_cells": 16})
    assert resp.status_code == 400, resp.text
    assert "changes the layout of the state" in resp.json()["detail"]
    assert built == []
    _assert_nothing_written(gm, "rod", before)


def test_a_state_over_the_cap_is_refused_before_it_is_built():
    """Each dimension under the integer bound, the product over the state
    cap: refused from the abstract state, with no node constructed and no
    concrete state of that size built."""
    gm = _graph(_Grid())
    before = _snapshot(gm, "g")
    _Grid.constructed.clear()
    _Grid.built.clear()
    resp = _put(gm, "g", {"nx": MAX_NODE_PARAM_INT})
    assert resp.status_code == 400, resp.text
    assert (f"{4 * MAX_NODE_PARAM_INT} state elements"
            in resp.json()["detail"])
    assert _Grid.constructed == [] and _Grid.built == []
    _assert_nothing_written(gm, "g", before)


def test_a_pipe_length_is_refused_before_a_pipe_of_that_length_is_built(monkeypatch):
    """A pipe's ``nx`` multiplies into its whole volume, and its masks are
    built by the constructor: the state it would build with the new ``nx``
    and the old masks does not broadcast, which is told abstractly."""
    built = []
    real_init = LBMPipeNode.__init__

    def counting_init(self, *args, **kwargs):
        built.append(kwargs.get("nx"))
        real_init(self, *args, **kwargs)

    gm = _pipe(G=0.0, fill_fraction=1.0)
    before = _snapshot(gm, "p")
    monkeypatch.setattr(LBMPipeNode, "__init__", counting_init)
    resp = _put(gm, "p", {"nx": 20_000})
    assert resp.status_code == 400, resp.text
    assert "initial_state() raises with it" in resp.json()["detail"]
    assert built == []
    _assert_nothing_written(gm, "p", before)


def test_the_state_cap_before_building_applies_the_servers_limit(monkeypatch):
    """The cap is the module's, read when the request is answered."""
    monkeypatch.setattr(server_module, "MAX_NODE_STATE_ELEMENTS", 20)
    gm = _graph(_Grid())
    resp = _put(gm, "g", {"nx": 6})
    assert resp.status_code == 400, resp.text
    assert "24 state elements" in resp.json()["detail"]
