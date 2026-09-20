"""No single request may name an unbounded amount of work or memory.

A loopback bind is served without a credential, so the caller of every
endpoint here can be anyone with a shell on the box, and the bounds are
what keeps one request from naming the whole machine: ``POST /sim/run``
used to accept any
``n_steps``, ``POST /graph/nodes`` let the caller pick an array dimension
(the audit measured +433 MB of RSS from one request), and
``TrainSurrogateRequest`` had no bounds at all.  The limits are on the
request models, so they appear in ``/openapi.json`` and a violation is a
422 that names the field.

Also covers the startup warning that tells an operator the port they just
opened is unauthenticated.
"""

import logging
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

from maddening.api import server as server_module
from maddening.api.server import (
    MAX_NODE_PARAM_INT,
    MAX_RUN_STEPS,
    MAX_SURROGATE_EPOCHS,
    SimulationServer,
    TrainSurrogateRequest,
    warn_if_publicly_bound,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

DT = 0.01


class GridNode(SimulationNode):
    """A node whose state size is the product of two caller-chosen ints."""

    def __init__(self, name, timestep, nx: int = 2, ny: int = 2):
        super().__init__(name, timestep)
        self.nx, self.ny = int(nx), int(ny)

    def initial_state(self):
        return {"grid": jnp.zeros((self.nx, self.ny), jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"grid": state["grid"] + dt}


REGISTRY = {
    "SpringDamperNode": SpringDamperNode,
    "HeatNode": HeatNode,
    "GridNode": GridNode,
}


def _node_names(client):
    return {n["name"] for n in client.get("/graph").json()["nodes"]}


def _client():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", DT, stiffness=30.0, damping=2.0,
                                 initial_position=1.0))
    gm.compile()
    return TestClient(
        SimulationServer(node_registry=REGISTRY, graph_manager=gm).create_app(),
        raise_server_exceptions=False,
    )


class TestSimRunIsBounded:
    def test_shipped_step_counts_still_run(self):
        c = _client()
        # 100 is what the example script and the bundled UI send; 200 is
        # the "Run 200" button in app.html.
        for n in (1, 100, 200):
            assert c.post(f"/sim/run?n_steps={n}").status_code == 200
        assert c.post("/sim/run").status_code == 200

    def test_step_count_above_the_limit_is_422(self):
        c = _client()
        r = c.post(f"/sim/run?n_steps={MAX_RUN_STEPS + 1}")
        assert r.status_code == 422
        assert "n_steps" in r.text

    def test_zero_steps_is_a_no_op_not_an_error(self):
        # A caller stepping a loop to zero gets the current state back, which
        # is what the stateful REST model in tests/property encodes.
        c = _client()
        before = c.get("/graph/state").json()
        r = c.post("/sim/run?n_steps=0")
        assert r.status_code == 200
        assert r.json() == before

    def test_negative_step_count_is_422(self):
        assert _client().post("/sim/run?n_steps=-5").status_code == 422

    def test_the_limit_is_published_in_the_schema(self):
        schema = _client().get("/openapi.json").json()
        params = schema["paths"]["/sim/run"]["post"]["parameters"]
        n_steps = next(p for p in params if p["name"] == "n_steps")
        assert n_steps["schema"]["maximum"] == MAX_RUN_STEPS
        assert n_steps["schema"]["minimum"] == 0


class TestAddNodeIsBounded:
    def test_an_array_dimension_above_the_limit_is_rejected(self):
        c = _client()
        before = _node_names(c)
        r = c.post("/graph/nodes", json={
            "type": "HeatNode", "name": "huge", "timestep": DT,
            "params": {"n_cells": 100_000_000},
        })
        # 422 from the schema: nothing was constructed, so nothing was
        # allocated.
        assert r.status_code == 422, r.text
        assert _node_names(c) == before

    def test_the_bound_is_on_magnitude_not_sign(self):
        c = _client()
        r = c.post("/graph/nodes", json={
            "type": "HeatNode", "name": "huge", "timestep": DT,
            "params": {"n_cells": -(MAX_NODE_PARAM_INT + 1)},
        })
        assert r.status_code == 422

    def test_a_nested_oversized_integer_is_rejected(self):
        c = _client()
        r = c.post("/graph/nodes", json={
            "type": "HeatNode", "name": "huge", "timestep": DT,
            "params": {"grid_points": {"n": [1, MAX_NODE_PARAM_INT + 1]}},
        })
        assert r.status_code == 422

    def test_ordinary_integer_and_float_params_still_work(self):
        # thermal_diffusivity is chosen to keep dt*alpha/dx^2 under the
        # stencil_order=4 Fourier limit of 5/16; above it HeatNode's own
        # constructor refuses the node and the server correctly reports a
        # 400, which would test the stability guard rather than the
        # request bounds this class is about.
        c = _client()
        r = c.post("/graph/nodes", json={
            "type": "HeatNode", "name": "h", "timestep": DT,
            "params": {"n_cells": 64, "thermal_diffusivity": 0.001,
                       "stencil_order": 4},
        })
        assert r.status_code == 201, r.text
        assert "h" in _node_names(c)

    def test_an_integer_too_large_to_be_a_float_stays_a_400_not_finite(self):
        # JSON allows an integer literal of any length.  One that float()
        # cannot represent is the unusable constant the finiteness check
        # already owns, and it keeps that check's documented 400 rather than
        # being re-reported as an oversized dimension.
        c = _client()
        r = c.post(
            "/graph/nodes",
            content=(b'{"type":"HeatNode","name":"n","timestep":0.01,'
                     b'"params":{"thermal_diffusivity": ' + b"9" * 400 + b'}}'),
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400, r.text
        assert "finite" in r.json()["detail"]

    def test_a_large_float_is_not_treated_as_a_dimension(self):
        # A float is a physical constant; only integers become shapes.
        # ``length`` rather than ``thermal_diffusivity``, because a huge
        # diffusivity is a genuinely unstable rod and HeatNode's
        # constructor now refuses it with a 400 -- a different rejection
        # from the 422 this class is about, and one that would let the
        # test pass for the wrong reason if the bound ever went away.
        c = _client()
        r = c.post("/graph/nodes", json={
            "type": "HeatNode", "name": "h", "timestep": DT,
            "params": {"n_cells": 8, "length": 1e12},
        })
        assert r.status_code == 201, r.text

    def test_a_large_float_that_is_physically_unstable_is_a_400(self):
        # The companion to the above: an oversized *float* is not a
        # dimension error (422), but a thermal_diffusivity that puts the
        # rod past its Fourier limit is refused by the node itself (400).
        c = _client()
        r = c.post("/graph/nodes", json={
            "type": "HeatNode", "name": "h", "timestep": DT,
            "params": {"n_cells": 8, "thermal_diffusivity": 1e12},
        })
        assert r.status_code == 400, r.text
        assert "Fourier number" in r.json()["detail"]

    def test_dimensions_that_multiply_are_caught_by_the_state_cap(self, monkeypatch):
        # Each factor is under the per-integer bound; their product is not.
        # The cap is exercised at a small value so the test does not have to
        # allocate the megabytes the real limit permits.
        monkeypatch.setattr(server_module, "MAX_NODE_STATE_ELEMENTS", 100)
        c = _client()
        before = _node_names(c)
        r = c.post("/graph/nodes", json={
            "type": "GridNode", "name": "grid", "timestep": DT,
            "params": {"nx": 40, "ny": 40},
        })
        assert r.status_code == 400
        assert "1600 state elements" in r.json()["detail"]
        assert _node_names(c) == before
        # ... and a state under the cap is still accepted.
        assert c.post("/graph/nodes", json={
            "type": "GridNode", "name": "small", "timestep": DT,
            "params": {"nx": 5, "ny": 5},
        }).status_code == 201


class TestTrainSurrogateIsBounded:
    def test_shipped_ui_maxima_are_accepted(self):
        # app.html's sliders top out at 2000 data steps and 500 epochs.
        req = TrainSurrogateRequest(
            node_name="s", n_data_steps=2000, n_epochs=500,
            hidden_sizes=[64, 64], batch_size=64,
        )
        assert req.n_data_steps == 2000 and req.n_epochs == 500

    @pytest.mark.parametrize("payload", [
        {"n_epochs": MAX_SURROGATE_EPOCHS + 1},
        {"n_data_steps": 10 ** 9},
        {"batch_size": 0},
        {"hidden_sizes": [1] * 17},
        {"hidden_sizes": [10 ** 9]},
        {"hidden_sizes": []},
    ])
    def test_out_of_range_training_arguments_are_422(self, payload):
        c = _client()
        r = c.post("/surrogate/train", json={"node_name": "s", **payload})
        assert r.status_code == 422, r.text

    def test_the_limits_are_published_in_the_schema(self):
        schema = _client().get("/openapi.json").json()
        model = schema["components"]["schemas"]["TrainSurrogateRequest"]
        assert model["properties"]["n_epochs"]["maximum"] == MAX_SURROGATE_EPOCHS
        assert model["properties"]["n_epochs"]["minimum"] == 1


class TestPublicBindWarning:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1",
                                      "LOCALHOST", "127.0.1.1"])
    def test_loopback_is_silent(self, host, caplog):
        with caplog.at_level(logging.WARNING):
            assert warn_if_publicly_bound(host, 8000) is False
        assert caplog.records == []

    @pytest.mark.parametrize("host", ["0.0.0.0", "10.1.2.3", "::", "example.net"])
    def test_non_loopback_warns(self, host, caplog):
        with caplog.at_level(logging.WARNING):
            assert warn_if_publicly_bound(host, 8000) is True
        message = "\n".join(r.getMessage() for r in caplog.records)
        # The warning has to name the actual risk, not just "be careful".
        # Such a bind now demands a bearer token, so the risk it names is
        # the one that is left: no TLS, and two exempt route families.
        assert host in message
        assert "NO TLS" in message
        assert "Bearer" in message
        assert "/viz/" in message
        assert "127.0.0.1" in message
