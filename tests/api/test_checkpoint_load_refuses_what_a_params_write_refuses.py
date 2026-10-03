"""``POST /checkpoint/load`` refuses the parameter values ``PUT /graph/params`` refuses.

A checkpoint carries the graph's live parameter leaves, and a load writes
them into ``gm.params``.  It used to check only that they fitted the graph's
shapes and dtypes, so it restored values the params route refuses -- one
the node's constructor refuses at this graph's timestep (a ``HeatNode``
past its Fourier limit), one outside the leaf's ``ParamSpec`` bounds (a
negative damping) -- answered 200, and left a graph whose save did not
reload and whose next steps diverged (the rod's temperatures reached
4e32 in 40 steps).  Both routes now ask the same code
(``_params_write_refusal``, after the same finiteness and bounds checks),
so for every value the two answer alike: the property test below writes
each value with ``PUT`` on one server and loads a checkpoint carrying it
on another, and requires the same verdict.

``GraphManager.load_state`` itself does not ask these, by decision: a
value outside a ``ParamSpec``'s bounds is one Python code may hold on
purpose (a ``gm.params`` write is not checked either), and a graph so
written must resume its own checkpoint into a freshly built graph.  It
refuses only text and booleans for a numeric leaf, which NumPy cast
without a word and no save writes.  The last tests pin both, and that the
route asks nothing of a value the graph holds or was built with (a graph
built outside its bounds reloads its own checkpoint).  The domain oracle,
``tests/property/test_differential_param_acceptance.py``, holds the route
to every write door and documents ``load_state`` as difference D6.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import HeatNode, SpringDamperNode

REGISTRY = {"HeatNode": HeatNode, "SpringDamperNode": SpringDamperNode}
ROD = dict(n_cells=10, length=1.0)
SPRING = dict(stiffness=100.0, mass=1.0, rest_length=0.5, initial_position=1.0)


def _client(gm: GraphManager, root) -> tuple[SimulationServer, TestClient]:
    server = SimulationServer(REGISTRY, graph_manager=gm, checkpoint_root=str(root))
    return server, TestClient(server.create_app(), raise_server_exceptions=False)


def _graph(*nodes) -> GraphManager:
    gm = GraphManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for node in nodes:
            gm.add_node(node)
        gm.compile()
    return gm


def _rod(dt: float, alpha: float) -> HeatNode:
    return HeatNode("rod", dt, thermal_diffusivity=alpha, **ROD)


def _spring(damping: float) -> SpringDamperNode:
    return SpringDamperNode("s", 0.01, damping=damping, **SPRING)


def _mapped_graph() -> GraphManager:
    """A 3-cell rod feeding a 5-cell rod through a cell-average projection,
    whose weight matrix holds zeros (cells that do not overlap)."""
    from maddening.core.coupling.mapping import projection_1d_mapping

    gm = GraphManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_node(HeatNode("coarse", 0.01, n_cells=3, length=1.0, thermal_diffusivity=0.01))
        gm.add_node(HeatNode("fine", 0.01, n_cells=5, length=1.0, thermal_diffusivity=0.01))
        gm.add_edge("coarse", "fine", "temperature", "heat_source",
                    mapping=projection_1d_mapping(np.linspace(0.0, 1.0, 4),
                                                  np.linspace(0.0, 1.0, 6)))
        gm.compile()
    return gm


def _save_with_live_value(root, name: str, node, key: str, value: float) -> None:
    """A checkpoint of a one-node graph whose live leaf *key* holds
    *value* -- written into ``gm.params`` the way Python code (a fit) may
    write any value."""
    gm = _graph(node)
    gm.params["nodes"][node.name][key] = jnp.asarray(value, jnp.float32)
    gm.save_state(str(root / name))


def _snapshot(gm: GraphManager) -> tuple:
    state = {n: {f: np.asarray(v).copy() for f, v in fields.items()}
             for n, fields in gm._state.items()}
    params = {o: {k: np.asarray(v).copy() for k, v in leaves.items()}
              for o, leaves in gm.params["nodes"].items()}
    return state, params


def _same(a: tuple, b: tuple) -> bool:
    def eq(x, y):
        return x.keys() == y.keys() and all(
            x[k].keys() == y[k].keys()
            and all(np.array_equal(x[k][f], y[k][f], equal_nan=True) for f in x[k])
            for k in x)
    return eq(a[0], b[0]) and eq(a[1], b[1])


def test_a_value_the_constructor_refuses_at_this_timestep_is_not_loaded(tmp_path):
    """The auditor's case: a rod saved at dt=0.01 with alpha=0.4 (Fourier
    0.4), loaded into a rod built at dt=0.05 (Fourier 2.0 with that alpha)."""
    gm_a = _graph(_rod(0.01, 0.4))
    gm_a.save_state(str(tmp_path / "a.npz"))
    gm = _graph(_rod(0.05, 0.05))
    _server, client = _client(gm, tmp_path)
    before = _snapshot(gm)

    put = client.put("/graph/params/rod", json={"params": {"thermal_diffusivity": 0.4}})
    assert put.status_code == 400 and "constructor refuses" in put.json()["detail"]
    load = client.post("/checkpoint/load", params={"path": "a.npz"})
    assert load.status_code == 400, load.text
    detail = load.json()["detail"]
    assert "does not fit this graph, nothing was loaded" in detail
    assert "thermal_diffusivity" in detail and "constructor refuses" in detail, detail
    assert _same(before, _snapshot(gm))
    # The graph still saves a config that reloads, and steps stably.
    GraphManager.from_dict({k: v for k, v in client.get("/graph").json().items()
                            if k != "active_surrogates"}, node_registry=REGISTRY)
    spike = [0.0] * 10
    spike[5] = 1.0
    assert client.put("/graph/state/rod", json={"state": {"temperature": spike}}).status_code == 200
    temps = client.post("/sim/run", params={"n_steps": 40}).json()["rod"]["temperature"]
    assert max(abs(t) for t in temps) <= 1.0


def test_a_value_outside_its_bounds_is_not_loaded(tmp_path):
    _save_with_live_value(tmp_path, "b.npz", _spring(1.0), "damping", -5.0)
    gm = _graph(_spring(1.0))
    _server, client = _client(gm, tmp_path)
    before = _snapshot(gm)
    put = client.put("/graph/params/s", json={"params": {"damping": -5.0}})
    assert put.status_code == 400
    load = client.post("/checkpoint/load", params={"path": "b.npz"})
    assert load.status_code == 400, load.text
    assert "-5.0 below bound 0.0" in load.json()["detail"]
    assert _same(before, _snapshot(gm))
    assert client.get("/graph/params/s").json()["damping"] == 1.0


def test_a_non_finite_parameter_in_a_checkpoint_is_not_loaded(tmp_path):
    _save_with_live_value(tmp_path, "n.npz", _spring(1.0), "stiffness", 100.0)
    with np.load(tmp_path / "n.npz") as data:
        arrays = {k: data[k] for k in data.files}
    key = next(k for k in arrays if k.endswith("/s/stiffness"))
    arrays[key] = np.asarray(np.nan, arrays[key].dtype)
    np.savez(tmp_path / "n.npz", **arrays)
    gm = _graph(_spring(1.0))
    _server, client = _client(gm, tmp_path)
    before = _snapshot(gm)
    load = client.post("/checkpoint/load", params={"path": "n.npz"})
    assert load.status_code == 400, load.text
    assert "stiffness: value must be finite" in load.json()["detail"]
    assert _same(before, _snapshot(gm))


def test_a_value_every_check_takes_still_loads_and_runs(tmp_path):
    """The control: alpha=0.09 at dt=0.05 is Fourier 0.45, under the limit."""
    _save_with_live_value(tmp_path, "ok.npz", _rod(0.05, 0.05), "thermal_diffusivity", 0.09)
    gm = _graph(_rod(0.05, 0.05))
    _server, client = _client(gm, tmp_path)
    load = client.post("/checkpoint/load", params={"path": "ok.npz"})
    assert load.status_code == 200, load.text
    assert client.get("/graph/params/rod").json()["thermal_diffusivity"] == pytest.approx(0.09)
    assert float(gm.params["nodes"]["rod"]["thermal_diffusivity"]) == pytest.approx(0.09)


def test_a_checkpoint_that_puts_a_leaf_back_to_the_nodes_own_value_loads(tmp_path):
    """A leaf a fit moved, put back by the load to the value the node was
    built with, writes no new value: it is asked only in the combined
    checks, and loads."""
    gm_src = _graph(_spring(1.0))
    gm_src.save_state(str(tmp_path / "own.npz"))
    gm = _graph(_spring(1.0))
    gm.params["nodes"]["s"]["damping"] = jnp.asarray(3.0, jnp.float32)   # a fit's value
    _server, client = _client(gm, tmp_path)
    load = client.post("/checkpoint/load", params={"path": "own.npz"})
    assert load.status_code == 200, load.text
    assert float(gm.params["nodes"]["s"]["damping"]) == 1.0


def test_a_load_that_puts_a_baked_leaf_back_to_the_nodes_own_value_loads(tmp_path):
    """A leaf the step does not read (``initial_position``), edited in
    ``gm.params`` from Python -- which the graph refuses at its next run --
    is put back by the load to the value the node was built with: a repair,
    not a new value, so it is not asked whether anything reads it (asked,
    the node's own value would be refused as "nothing reads it")."""
    gm_src = _graph(_spring(1.0))
    gm_src.save_state(str(tmp_path / "clean.npz"))
    gm = _graph(_spring(1.0))
    gm.params["nodes"]["s"]["initial_position"] = jnp.asarray(3.0, jnp.float32)
    _server, client = _client(gm, tmp_path)
    load = client.post("/checkpoint/load", params={"path": "clean.npz"})
    assert load.status_code == 200, load.text
    assert float(gm.params["nodes"]["s"]["initial_position"]) == SPRING["initial_position"]
    assert client.post("/sim/step").status_code == 200


@pytest.mark.parametrize("node, key, value", [
    (lambda: _spring(1.0), "damping", -5.0),
    (lambda: _spring(1.0), "damping", -1e-30),
    (lambda: _spring(1.0), "damping", 0.0),
    (lambda: _spring(1.0), "damping", 7.5),
    (lambda: _spring(1.0), "stiffness", 0.0),            # a log transform: 0 has no coordinate
    (lambda: _spring(1.0), "stiffness", 2.0e7),
    (lambda: _rod(0.05, 0.05), "thermal_diffusivity", 0.09),
    (lambda: _rod(0.05, 0.05), "thermal_diffusivity", 0.1),     # float32 0.1 is over the limit
    (lambda: _rod(0.05, 0.05), "thermal_diffusivity", 0.1001),
    (lambda: _rod(0.05, 0.05), "thermal_diffusivity", 0.4),
    (lambda: _rod(0.05, 0.05), "thermal_diffusivity", -0.01),
    (lambda: _rod(0.05, 0.05), "length", 0.5),            # Fourier 2.0 with dx = 0.05
    (lambda: _rod(0.05, 0.05), "length", 2.0),
], ids=lambda v: v if isinstance(v, (str, float)) else "")
def test_a_load_and_a_params_write_give_the_same_verdict_on_every_value(tmp_path, node, key,
                                                                      value):
    """One decision on both routes: whatever ``PUT`` answers for a value
    (2xx or 4xx), a load of a checkpoint carrying it answers the same, and
    a refused load changes nothing."""
    _save_with_live_value(tmp_path, "v.npz", node(), key, value)
    gm_put = _graph(node())
    put = _client(gm_put, tmp_path)[1].put(f"/graph/params/{gm_put.node_names[0]}",
                                           json={"params": {key: value}})
    gm_load = _graph(node())
    before = _snapshot(gm_load)
    load = _client(gm_load, tmp_path)[1].post("/checkpoint/load", params={"path": "v.npz"})
    assert put.status_code < 500 and load.status_code < 500, (put.text, load.text)
    assert (put.status_code == 200) == (load.status_code == 200), (put.text, load.text)
    if load.status_code != 200:
        assert _same(before, _snapshot(gm_load))
    else:
        assert float(gm_load.params["nodes"][gm_load.node_names[0]][key]) \
            == pytest.approx(value, rel=1e-6)


@pytest.mark.parametrize("key, member, refusal", [
    ("damping", np.asarray(True), "holds a boolean"),
    ("damping", np.asarray("1.5"), "not a number"),
], ids=["boolean", "numeric-string"])
def test_load_state_refuses_text_and_a_boolean_for_a_number(tmp_path, key, member, refusal):
    """NumPy cast a boolean member to 1.0 and a numeric string to its number,
    and ``load_state`` restored both; no save writes either, and every write
    door refuses them.  A ``ValueError`` now, nothing loaded, on both doors."""
    _save_with_live_value(tmp_path, "py.npz", _spring(1.0), key, 1.0)
    with np.load(tmp_path / "py.npz") as data:
        arrays = {k: data[k] for k in data.files}
    arrays[next(k for k in arrays if k.endswith(f"/s/{key}"))] = member
    np.savez(tmp_path / "py.npz", **arrays)
    gm = _graph(_spring(1.0))
    before = _snapshot(gm)
    with pytest.raises(ValueError, match=refusal):
        gm.load_state(str(tmp_path / "py.npz"))
    assert _same(before, _snapshot(gm))
    _server, client = _client(gm, tmp_path)
    load = client.post("/checkpoint/load", params={"path": "py.npz"})
    assert load.status_code == 400 and refusal in load.json()["detail"], load.text
    assert _same(before, _snapshot(gm))


@pytest.mark.parametrize("node, key, value", [
    ("spring", "damping", -5.0), ("spring", "stiffness", float("nan")), ("rod", None, 0.4),
], ids=["below-bound", "nan", "constructor-refuses"])
def test_load_state_restores_what_python_may_hold_and_the_route_refuses(tmp_path, node, key,
                                                                        value):
    """The decision this PR records: ``GraphManager.load_state`` restores a
    value outside a ``ParamSpec``'s bounds, a NaN, or one the node's
    constructor refuses at this graph's timestep -- a ``gm.params`` write
    takes each, so a graph written so resumes its own checkpoint -- while
    ``POST /checkpoint/load`` refuses the same file and loads nothing."""
    if node == "rod":
        _graph(_rod(0.01, value)).save_state(str(tmp_path / "c.npz"))
        build, read = (lambda: _graph(_rod(0.05, 0.05))), ("rod", "thermal_diffusivity")
    else:
        _save_with_live_value(tmp_path, "c.npz", _spring(1.0), key, value)
        build, read = (lambda: _graph(_spring(1.0))), ("s", key)
    gm = build()
    _server, client = _client(gm, tmp_path)
    before = _snapshot(gm)
    assert client.post("/checkpoint/load", params={"path": "c.npz"}).status_code == 400
    assert _same(before, _snapshot(gm))
    gm.load_state(str(tmp_path / "c.npz"))
    np.testing.assert_array_equal(np.asarray(gm.params["nodes"][read[0]][read[1]]),
                                  np.float32(value))


def test_a_graph_built_outside_its_bounds_reloads_its_own_checkpoint(tmp_path):
    """A graph runs whatever its constructor was given, bounds being
    metadata to it: a stiffness of 100 under a declared lower bound of 150
    is the graph's own value.  The route asks nothing of a value that is
    the leaf's now or the node's own -- the FMU's ``set_fmu_state`` rule --
    so the graph reloads its own checkpoint, and goes back to it after a
    ``gm.params`` write moved the leaf (to 200, inside the bounds); a
    checkpoint that installs another value outside the bounds is still
    refused."""
    from maddening.core.params import ParamSpec

    gm = _graph(_spring(1.0))
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(150.0, None)))
    gm.save_state(str(tmp_path / "own.npz"))
    _server, client = _client(gm, tmp_path)
    assert client.post("/checkpoint/load", params={"path": "own.npz"}).status_code == 200
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(200.0, jnp.float32)
    back = client.post("/checkpoint/load", params={"path": "own.npz"})
    assert back.status_code == 200, back.text
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 100.0

    _save_with_live_value(tmp_path, "moved.npz", _spring(1.0), "stiffness", 120.0)
    before = _snapshot(gm)
    load = client.post("/checkpoint/load", params={"path": "moved.npz"})
    assert load.status_code == 400 and "120.0 below bound 150.0" in load.json()["detail"], \
        load.text
    assert _same(before, _snapshot(gm))


def test_a_python_write_the_constructor_refuses_still_reloads_its_own_checkpoint(tmp_path):
    """A ``gm.params`` write in Python is not checked against the node's
    constructor (``MADD-ANO-047``'s residual), so a graph can hold a rod past
    its Fourier limit.  Its own checkpoint, which changes nothing, loads
    through the route; the same checkpoint into a graph at a stable value
    is refused there, since there the load installs the value."""
    gm = _graph(_rod(0.05, 0.05))
    gm.params["nodes"]["rod"]["thermal_diffusivity"] = jnp.asarray(0.4, jnp.float32)
    gm.save_state(str(tmp_path / "own.npz"))
    _server, client = _client(gm, tmp_path)
    assert client.post("/checkpoint/load", params={"path": "own.npz"}).status_code == 200
    fresh = _graph(_rod(0.05, 0.05))
    _server, fresh_client = _client(fresh, tmp_path)
    before = _snapshot(fresh)
    load = fresh_client.post("/checkpoint/load", params={"path": "own.npz"})
    assert load.status_code == 400 and "constructor refuses" in load.json()["detail"], load.text
    assert _same(before, _snapshot(fresh))


def test_a_weight_matrix_is_asked_about_the_entries_a_load_moves(tmp_path):
    """Per element: a mapping weight matrix holding zeros under a ``log``
    spec (which refuses 0, and which the graph was built with) loads when
    the checkpoint moves only its non-zero entries, and is refused when it
    moves one to a value the spec refuses."""
    from maddening.core.params import ParamSpec

    gm = _mapped_graph()
    edge, = gm.params["mappings"]
    held = np.asarray(gm.params["mappings"][edge]["H"])
    assert (held == 0).any() and (held != 0).any(), held
    gm.set_param_spec(edge, "H", ParamSpec(bounds=(0.0, None), transform="log"))
    scaled = jnp.asarray(np.where(held > 0, held * 0.5, held), held.dtype)
    gm.params["mappings"][edge]["H"] = scaled
    gm.save_state(str(tmp_path / "trained.npz"))
    gm.params["mappings"][edge]["H"] = jnp.asarray(np.where(held > 0, -held, held), held.dtype)
    gm.save_state(str(tmp_path / "negative.npz"))

    target = _mapped_graph()
    target.set_param_spec(edge, "H", ParamSpec(bounds=(0.0, None), transform="log"))
    _server, client = _client(target, tmp_path)
    before = _snapshot(target)
    bad = client.post("/checkpoint/load", params={"path": "negative.npz"})
    assert bad.status_code == 400 and "below bound 0.0" in bad.json()["detail"], bad.text
    assert _same(before, _snapshot(target))
    ok = client.post("/checkpoint/load", params={"path": "trained.npz"})
    assert ok.status_code == 200, ok.text
    np.testing.assert_array_equal(np.asarray(target.params["mappings"][edge]["H"]),
                                  np.asarray(scaled))
