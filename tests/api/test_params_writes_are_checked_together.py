"""``PUT /graph/params`` asks the constructor about the values of a request
together, and holds a structural value to the float range a leaf is held to.

* Each structural key was checked against the node's constructor *alone*,
  with every other key at its old value: HeatNode ``stencil_order: 4`` at a
  Fourier number of 0.4 is unstable (the 4th-order limit is 5/16), and
  together with ``thermal_diffusivity: 0.2`` in the same request it is
  not, yet the request was refused.  The per-key check now builds with the
  request's other changes applied; a combination the constructor refuses is
  refused naming every key of it, and nothing is written.
* A finite number inside a structural value that float32 cannot hold --
  RigidBodyNode ``constraints: {"z": 1e39}`` -- was taken (200), and every
  step after it produced infinities.  It is a 400 now, as a live leaf's
  1e39 already was, on ``PUT /graph/params`` and ``POST /graph/nodes``.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import HeatNode, RigidBodyNode

REGISTRY = {"HeatNode": HeatNode, "RigidBodyNode": RigidBodyNode}


def _served(node, compile_first=True):
    gm = GraphManager()
    gm.add_node(node)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if compile_first:
            gm.compile()
    server = SimulationServer(REGISTRY, graph_manager=gm)
    return gm, TestClient(server.create_app(), raise_server_exceptions=False)


def _rod(**kw):
    """Fourier number dt * alpha / dx**2 = 0.01 * alpha / 0.01 = alpha."""
    return _served(HeatNode("n", 0.01, n_cells=10, length=1.0,
                            initial_temperature=1.0, **{"thermal_diffusivity": 0.4, **kw}))


def _snapshot(gm):
    node = gm._nodes["n"].node
    live = {k: np.asarray(v).copy() for k, v in (gm.params.get("nodes", {}).get("n") or {}).items()}
    return dict(node.params), live, gm._dirty


def _assert_nothing_written(gm, before):
    params, live, dirty = before
    assert dict(gm._nodes["n"].node.params) == params
    now = gm.params.get("nodes", {}).get("n") or {}
    assert now.keys() == live.keys()
    for key, value in live.items():
        np.testing.assert_array_equal(np.asarray(now[key]), value)
    assert gm._dirty is dirty


def _assert_the_saved_graph_runs_the_same(gm, steps=20):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = GraphManager.from_dict(gm.to_dict(), REGISTRY)
        again.compile()
    gm.reset_state()
    np.testing.assert_array_equal(np.asarray(gm.run_scan(steps)["n"]["temperature"]),
                                  np.asarray(again.run_scan(steps)["n"]["temperature"]))


@pytest.mark.parametrize("compile_first", [True, False])
def test_a_value_valid_only_with_another_key_of_the_request_is_taken(compile_first):
    gm, client = _served(HeatNode("n", 0.01, n_cells=10, length=1.0, initial_temperature=1.0,
                                  thermal_diffusivity=0.4), compile_first)
    resp = client.put("/graph/params/n", json={"params": {"stencil_order": 4,
                                                          "thermal_diffusivity": 0.2}})
    assert resp.status_code == 200, resp.text
    node = gm._nodes["n"].node
    assert node.params["stencil_order"] == 4
    assert node.params["thermal_diffusivity"] == pytest.approx(0.2)
    assert client.post("/sim/run", params={"n_steps": 10}).status_code == 200
    _assert_the_saved_graph_runs_the_same(gm)


def test_the_same_value_alone_is_still_refused_and_nothing_is_written():
    gm, client = _rod()
    before = _snapshot(gm)
    resp = client.put("/graph/params/n", json={"params": {"stencil_order": 4}})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith(
        "stencil_order: node 'n' cannot take a new value for this parameter")
    assert "Fourier number" in resp.json()["detail"]
    _assert_nothing_written(gm, before)


def test_a_combination_the_constructor_refuses_is_refused_naming_its_keys():
    gm, client = _rod()
    before = _snapshot(gm)
    # Fourier 0.35 > 5/16 with the 4th-order stencil: neither key saves it.
    resp = client.put("/graph/params/n", json={"params": {"stencil_order": 4,
                                                          "thermal_diffusivity": 0.35}})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("stencil_order, thermal_diffusivity: node 'n' cannot take "
                             "these values together"), detail
    _assert_nothing_written(gm, before)


# ---------------------------------------------------------------------------
# float32 inside a structural value
# ---------------------------------------------------------------------------

def _body(**kw):
    return _served(RigidBodyNode("n", 0.01, initial_velocity=[0.1, 0.2, 0.3], **kw))


@pytest.mark.parametrize("constraints", [{"z": 1e39}, {"x": -3.5e38}])
def test_a_structural_value_float32_cannot_hold_is_refused(constraints):
    gm, client = _body()
    before = _snapshot(gm)
    resp = client.put("/graph/params/n", json={"params": {"constraints": constraints}})
    assert resp.status_code == 400, resp.text
    key = next(iter(constraints))
    assert resp.json()["detail"].startswith(f"constraints.{key}: value does not fit its "
                                            "type float32")
    _assert_nothing_written(gm, before)
    assert client.post("/sim/run", params={"n_steps": 3}).status_code == 200


def test_a_large_structural_value_float32_holds_is_still_taken():
    gm, client = _body()
    resp = client.put("/graph/params/n", json={"params": {"constraints": {"z": 3e38}}})
    assert resp.status_code == 200, resp.text
    assert gm._nodes["n"].node.params["constraints"] == {"z": 3e38}


@pytest.mark.parametrize("params, where", [
    ({"constraints": {"z": 1e39}}, "constraints.z"),
    ({"initial_position": [0.0, 1e39, 0.0]}, "initial_position[1]"),
])
def test_a_new_node_with_a_value_float32_cannot_hold_is_refused(params, where):
    gm, client = _body()
    resp = client.post("/graph/nodes", json={"type": "RigidBodyNode", "name": "m",
                                             "timestep": 0.01, "params": params})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == f"params.{where}: value does not fit its type float32"
    assert list(gm._nodes) == ["n"]


def test_a_live_leaf_keeps_its_own_float_range_message():
    gm, client = _rod()
    resp = client.put("/graph/params/n", json={"params": {"thermal_diffusivity": 1e39}})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == ("thermal_diffusivity: value does not fit its type "
                                     "float32, got 1e+39")
