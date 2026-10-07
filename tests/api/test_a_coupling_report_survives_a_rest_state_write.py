"""A coupling report describes the step that ran, through the REST write doors too.

``PUT /graph/state/{node}`` writes through ``set_node_state`` and
``POST /checkpoint/save`` / ``load`` through ``save_state`` /
``load_state``; the report is read in process (no route returns it).  See
``tests/core/test_coupling_report_describes_the_step_that_ran.py`` for the
invariant and the in-process doors.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests._loopback_client import LoopbackTestClient as TestClient

N = 3
KEY = "a+b"


class Lin(SimulationNode):
    def __init__(self, name, b):
        super().__init__(name, 0.01, g=jnp.float32(0.99), b=jnp.full(N, b, jnp.float32))

    def initial_state(self):
        return {"x": jnp.zeros(N, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N,), dtype=jnp.float32,
                                       default=jnp.zeros(N, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["g"] * boundary_inputs["u"] + p["b"]}

    def update_evaluations(self):
        return 1


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    gm = GraphManager()
    gm.add_node(Lin("a", 1.0))
    gm.add_node(Lin("b", 2.0))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    gm.add_coupling_group(["a", "b"], max_iterations=400000, tolerance=1e-9,
                          diagnostics=True)
    gm.compile()
    server = SimulationServer({"Lin": Lin}, graph_manager=gm,
                              checkpoint_root=str(tmp_path_factory.mktemp("checkpoints")))
    return gm, TestClient(server.create_app(), raise_server_exceptions=False)


def _report(gm):
    d = gm.coupling_diagnostics().get(KEY)
    if d is None:
        return None
    return {k: ("nan" if isinstance(v, float) and v != v else v) for k, v in d.items()}


def _zero(client):
    for name in ("a", "b"):
        reply = client.put(f"/graph/state/{name}", json={"state": {"x": [0.0] * N}})
        assert reply.status_code == 200, reply.text


def test_a_state_put_after_the_step_does_not_move_its_report(served):
    gm, client = served
    assert client.post("/sim/reset").status_code == 200
    assert client.post("/sim/step").status_code == 200
    first = _report(gm)
    assert first["spectral_usable"] and first["precision_limited"], first
    assert first["spectral_error_bound"] > 0.0
    _zero(client)
    assert client.get("/graph/state/a").json()["x"] == [0.0] * N
    assert _report(gm) == first


def test_a_checkpoint_saved_over_rest_after_a_put_reloads_the_state_and_withholds_the_bound(served):
    gm, client = served
    assert client.post("/sim/reset").status_code == 200
    assert client.post("/sim/step").status_code == 200
    first = _report(gm)
    assert client.post("/checkpoint/save", params={"path": "stepped.npz"}).status_code == 200
    _zero(client)
    assert client.post("/checkpoint/save", params={"path": "written.npz"}).status_code == 200
    assert _report(gm) == first
    before = client.get("/graph/state").json()

    reply = client.post("/checkpoint/load", params={"path": "written.npz"})
    assert reply.status_code == 200, reply.text
    # A checkpoint is a copy of the state, ``_meta`` included.
    assert client.get("/graph/state").json() == before
    loaded = _report(gm)
    assert loaded["spectral_usable"] is False and loaded["precision_limited"] is False
    assert "written" in loaded["not_usable_reason"]
    assert loaded["iterations"] == first["iterations"]
    reply = client.post("/checkpoint/load", params={"path": "stepped.npz"})
    assert reply.status_code == 200, reply.text
    assert _report(gm) == first
    _zero(client)
    assert _report(gm) == first
