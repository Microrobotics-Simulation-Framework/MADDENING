"""A value a float32 leaf would overflow is refused before it is cast.

``PUT /graph/params`` and ``PUT /graph/state`` cast the request value to
the live leaf's dtype with ``jnp.asarray(value, dtype=...)`` and checked it
afterwards.  ``1e39`` into ``float32`` is ``inf``, and NumPy says so with a
``RuntimeWarning`` ("overflow encountered in cast"): a 500 wherever warnings
are errors (this suite's ``filterwarnings = ["error"]``), and otherwise a
400 calling a finite value non-finite.  Nothing was written either way.
The value is now refused before the cast, in the words of the FMU's own
write paths ("does not fit its type float32").
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer, _unrepresentable
from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode

REGISTRY = {"BallNode": BallNode, "HeatNode": HeatNode}
FLOAT32_MAX = float(np.finfo(np.float32).max)


@pytest.fixture(scope="module")
def gm():
    g = GraphManager()
    g.add_node(BallNode("ball", 0.01, initial_position=1.0))
    g.add_node(HeatNode("rod", 0.01, n_cells=4, thermal_diffusivity=0.001,
                        initial_temperature=1.0))
    g.compile()
    return g


@pytest.fixture
def client(gm, tmp_path):
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm,
                              checkpoint_root=str(tmp_path))
    return TestClient(server.create_app(), raise_server_exceptions=False)


@pytest.mark.parametrize("value, refused", [
    (1e39, True), (-1e39, True), ([1.0, 1e39], True), ([[1e39]], True),
    (FLOAT32_MAX, False), (-FLOAT32_MAX, False), (3.0, False), ([1.0, 2.0], False),
    # left to the checks after the cast: not numbers, or not finite at all
    (math.inf, False), (math.nan, False), ("abc", False), (None, False),
    ([1.0, [2.0, 3.0]], False),
])
def test_only_a_finite_value_the_dtype_overflows_is_called_unrepresentable(value, refused):
    problem = _unrepresentable(value, np.float32)
    assert (problem is not None) == refused, (value, problem)
    if refused:
        assert problem == "value does not fit its type float32"


def test_a_wider_dtype_takes_what_float32_cannot():
    assert _unrepresentable(1e39, np.float64) is None
    assert _unrepresentable(1e39, np.int32) is None       # not a float leaf: not asked


@pytest.mark.parametrize("key, value", [("gravity", 1e39), ("gravity", -1e39)])
def test_a_parameter_write_of_an_overflowing_value_is_a_clean_400(client, gm, key, value):
    before = float(gm.params["nodes"]["ball"][key])
    resp = client.put("/graph/params/ball", json={"params": {key: value}})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith(f"{key}: value does not fit its type float32")
    assert float(gm.params["nodes"]["ball"][key]) == before


def test_an_array_parameter_with_one_overflowing_element_is_refused(client, gm):
    before = np.asarray(gm.params["nodes"]["rod"]["initial_temperature"]).copy()
    bad = [1.0] * before.size
    bad[-1] = 1e39
    resp = client.put("/graph/params/rod", json={"params": {"initial_temperature": bad}})
    assert resp.status_code == 400, resp.text
    assert "does not fit its type float32" in resp.json()["detail"]
    np.testing.assert_array_equal(np.asarray(gm.params["nodes"]["rod"]["initial_temperature"]),
                                  before)


def test_a_state_write_of_an_overflowing_value_is_a_clean_400(client, gm):
    before = {k: np.asarray(v).copy() for k, v in gm.get_node_state("ball").items()}
    resp = client.put("/graph/state/ball", json={"state": {
        "position": 1e39, "velocity": 0.0}})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == "position: value does not fit its type float32"
    for field, value in before.items():
        np.testing.assert_array_equal(np.asarray(gm.get_node_state("ball")[field]), value)


def test_the_largest_float32_still_reaches_the_checks_after_the_cast(client, gm):
    """The check refuses an overflow, not a large number: ``float32``'s own
    maximum is cast and then held to the node's own refusals."""
    before = {k: float(v) for k, v in gm.get_node_state("ball").items()}
    resp = client.put("/graph/state/ball", json={"state": {
        "position": FLOAT32_MAX, "velocity": 0.0}})
    assert resp.status_code == 200, resp.text
    assert float(gm.get_node_state("ball")["position"]) == FLOAT32_MAX
    assert client.put("/graph/state/ball", json={"state": before}).status_code == 200
