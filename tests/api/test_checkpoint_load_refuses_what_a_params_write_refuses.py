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

``GraphManager.load_state`` itself asks the first three -- a leaf it
changes finite and inside its bounds, as ``check_params`` asks, and the
constructor -- for a Python caller, and refuses text and booleans for a
numeric leaf; a leaf it leaves where it was is not asked, so a graph built
outside its bounds reloads its own checkpoint.  The last tests pin that.  (The domain oracle,
``tests/property/test_differential_param_acceptance.py``, holds both loads
to every write door.)
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
    assert "stiffness" in load.json()["detail"] and "not finite" in load.json()["detail"]
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
    ("damping", np.float32(-5.0), "below bound 0.0"),
    ("stiffness", np.float32(np.nan), "not finite"),
    ("damping", np.asarray(True), "holds a boolean"),
    ("damping", np.asarray("1.5"), "not a number"),
], ids=["below-bound", "nan", "boolean", "numeric-string"])
def test_graph_manager_load_state_refuses_what_check_params_and_the_constructor_refuse(
        tmp_path, key, member, refusal):
    """A Python caller had the same hole as the route: ``load_state``
    restored a damping below its bound (which ``check_params`` refuses), a
    NaN, and -- NumPy casting without a word -- a boolean as 1.0 and a
    numeric string as its number.  Each is a ``ValueError`` now, with
    nothing loaded."""
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


def test_graph_manager_load_state_refuses_a_value_the_constructor_refuses(tmp_path):
    gm_a = _graph(_rod(0.01, 0.4))
    gm_a.save_state(str(tmp_path / "a.npz"))
    gm = _graph(_rod(0.05, 0.05))
    before = _snapshot(gm)
    with pytest.raises(ValueError, match="constructor refuses"):
        gm.load_state(str(tmp_path / "a.npz"))
    assert _same(before, _snapshot(gm))
    _save_with_live_value(tmp_path, "ok.npz", _rod(0.05, 0.05), "thermal_diffusivity", 0.09)
    gm.load_state(str(tmp_path / "ok.npz"))      # the control
    assert float(gm.params["nodes"]["rod"]["thermal_diffusivity"]) == pytest.approx(0.09)


def test_a_graph_built_outside_its_bounds_reloads_its_own_checkpoint(tmp_path):
    """A graph runs whatever its constructor was given, bounds being
    metadata to it: a stiffness of 100 under a declared lower bound of 150
    is the graph's own value.  Both loads ask nothing of a leaf they leave
    where it was, so the graph reloads its own checkpoint (asked as a whole
    tree, ``load_state`` refused it); a checkpoint that moves the leaf to
    another value outside the bounds is still refused by both."""
    from maddening.core.params import ParamSpec

    gm = _graph(_spring(1.0))
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(150.0, None)))
    gm.save_state(str(tmp_path / "own.npz"))
    gm.load_state(str(tmp_path / "own.npz"))
    _server, client = _client(gm, tmp_path)
    assert client.post("/checkpoint/load", params={"path": "own.npz"}).status_code == 200
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 100.0

    _save_with_live_value(tmp_path, "moved.npz", _spring(1.0), "stiffness", 120.0)
    before = _snapshot(gm)
    with pytest.raises(ValueError, match="120.0 below bound 150.0"):
        gm.load_state(str(tmp_path / "moved.npz"))
    load = client.post("/checkpoint/load", params={"path": "moved.npz"})
    assert load.status_code == 400 and "below bound 150.0" in load.json()["detail"], load.text
    assert _same(before, _snapshot(gm))


def test_a_python_write_the_constructor_refuses_still_reloads_its_own_checkpoint(tmp_path):
    """A ``gm.params`` write in Python is not checked against the node's
    constructor (``MADD-ANO-047``'s residual), so a graph can hold a rod past
    its Fourier limit.  Its own checkpoint, which leaves every leaf where it
    was, reloads; the same checkpoint into a graph at a stable value is
    refused, since there the load installs the value."""
    gm = _graph(_rod(0.05, 0.05))
    gm.params["nodes"]["rod"]["thermal_diffusivity"] = jnp.asarray(0.4, jnp.float32)
    gm.save_state(str(tmp_path / "own.npz"))
    gm.load_state(str(tmp_path / "own.npz"))
    assert float(gm.params["nodes"]["rod"]["thermal_diffusivity"]) == pytest.approx(0.4)
    fresh = _graph(_rod(0.05, 0.05))
    before = _snapshot(fresh)
    with pytest.raises(ValueError, match="constructor refuses"):
        fresh.load_state(str(tmp_path / "own.npz"))
    assert _same(before, _snapshot(fresh))
