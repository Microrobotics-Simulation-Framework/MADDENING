"""``PUT /graph/state`` refuses the values every other write surface refuses.

The route cast each field with ``jnp.asarray(value, dtype=...)`` and asked
nothing else of an integer field: in LBMNode's ``uint8`` wall mask 0.5 and
1.7 were stored as 0 and 1 with a 200, and 256, -1 or 1e10 were a 500
(``OverflowError``).  On every field ``"1.5"``, ``" 2 "`` and ``true`` were
parsed as numbers, where ``PUT /graph/params``, a checkpoint load
(``checkpoint._checked_cast``) and the FMU refuse text and booleans.  Each
is now a 400 naming the field, and nothing is written.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
from maddening.nodes.lbm import LBMNode
from tests._loopback_client import LoopbackTestClient as TestClient

GRID = (4, 3)


@pytest.fixture(scope="module")
def served():
    gm = GraphManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_node(LBMNode("lbm", 1.0, grid_shape=GRID, lattice="D2Q9", viscosity=0.1))
        gm.add_node(BallNode("ball", 0.01, initial_position=5.0))
        gm.compile()
    server = SimulationServer({"LBMNode": LBMNode, "BallNode": BallNode}, graph_manager=gm)
    return gm, TestClient(server.create_app(), raise_server_exceptions=False)


def _snapshot(gm, node):
    return {k: np.asarray(v).copy() for k, v in gm.get_node_state(node).items()}


def _put_wall_mask_entry(served, value):
    gm, client = served
    state = client.get("/graph/state/lbm").json()
    mask = np.zeros(GRID, dtype=int).tolist()
    mask[0][0] = value
    return client.put("/graph/state/lbm", json={"state": {**state, "wall_mask": mask}})


@pytest.mark.parametrize("value", [0.5, 1.7, -0.25, 256, -1, 1e10, 2 ** 64, "1", " 2 ", True,
                                   None, [1]])
def test_an_integer_field_refuses_what_its_dtype_cannot_hold_exactly(served, value):
    gm, _client = served
    before = _snapshot(gm, "lbm")
    resp = _put_wall_mask_entry(served, value)
    assert resp.status_code == 400, (value, resp.status_code, resp.text)
    assert resp.json()["detail"].startswith("wall_mask: "), resp.text
    after = _snapshot(gm, "lbm")
    assert all(np.array_equal(before[k], after[k], equal_nan=True) for k in before)


@pytest.mark.parametrize("value", [0, 1, 2, 255, 7.0])
def test_an_integer_field_takes_integral_values_in_its_range(served, value):
    gm, _client = served
    resp = _put_wall_mask_entry(served, value)
    assert resp.status_code == 200, resp.text
    assert np.asarray(gm.get_node_state("lbm")["wall_mask"])[0, 0] == int(value)
    assert np.asarray(gm.get_node_state("lbm")["wall_mask"]).dtype == np.uint8


@pytest.mark.parametrize("value, words", [
    ("1.5", "got a string"), (" 2 ", "got a string"), (True, "got a boolean"),
    (None, "got null"), ({"a": 1}, "got dict"),
    (10 ** 400, "does not fit its type"),
])
def test_a_float_field_refuses_text_booleans_and_what_float_cannot_hold(served, value, words):
    gm, client = served
    before = _snapshot(gm, "ball")
    resp = client.put("/graph/state/ball", json={"state": {"position": value, "velocity": 0.0}})
    assert resp.status_code == 400, (value, resp.status_code, resp.text)
    assert resp.json()["detail"].startswith("position: ") and words in resp.json()["detail"]
    after = _snapshot(gm, "ball")
    assert all(np.array_equal(before[k], after[k]) for k in before)


@pytest.mark.parametrize("literal", ["Infinity", "-Infinity", "NaN"])
def test_an_integer_field_refuses_a_non_finite_json_number(served, literal):
    gm, client = served
    state = client.get("/graph/state/lbm").json()
    mask = np.zeros(GRID, dtype=int).tolist()
    import json
    body = json.dumps({"state": {**state, "wall_mask": mask}}).replace(
        '"wall_mask": [[0', f'"wall_mask": [[{literal}', 1)
    assert literal in body
    before = _snapshot(gm, "lbm")
    resp = client.put("/graph/state/lbm", content=body,
                      headers={"content-type": "application/json"})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"].startswith("wall_mask: ")
    after = _snapshot(gm, "lbm")
    assert all(np.array_equal(before[k], after[k], equal_nan=True) for k in before)


def test_text_nested_in_a_field_is_refused(served):
    gm, _client = served
    before = _snapshot(gm, "lbm")
    resp = _put_wall_mask_entry(served, "x")
    assert resp.status_code == 400 and "got a string" in resp.json()["detail"], resp.text
    after = _snapshot(gm, "lbm")
    assert all(np.array_equal(before[k], after[k], equal_nan=True) for k in before)


def test_a_float_field_still_takes_a_number_and_an_integral_json_number(served):
    gm, client = served
    for value in (1.25, 3):
        resp = client.put("/graph/state/ball", json={"state": {"position": value,
                                                              "velocity": 0.0}})
        assert resp.status_code == 200, resp.text
        assert float(gm.get_node_state("ball")["position"]) == value


def test_the_state_rule_is_the_checkpoint_loads():
    """The same values, asked of the two surfaces' own helpers: what the
    state route refuses, a checkpoint load of the same array refuses."""
    from maddening.api.server import _state_value_refusal
    from maddening.core.simulation.checkpoint import _checked_cast

    for value in (0.5, 256, -1):
        assert _state_value_refusal([value], np.uint8) is not None
        with pytest.raises(ValueError):
            _checked_cast(np.asarray([value]), np.uint8, "field 'lbm/wall_mask'")
    for value in (0, 255, 3.0):
        assert _state_value_refusal([value], np.uint8) is None
        _checked_cast(np.asarray([value]), np.uint8, "field 'lbm/wall_mask'")
    assert _state_value_refusal([True], np.bool_) is None
    assert _state_value_refusal([2], np.bool_) is not None
