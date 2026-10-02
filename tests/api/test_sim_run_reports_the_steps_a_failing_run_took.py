"""A ``POST /sim/run`` whose step raises part-way through says how many
steps it took.

A step can raise at run time -- an in-graph check such as
``CouplingGroup(strict_convergence=True)``, or ``equinox.error_if`` in a
node -- after the run's earlier steps were stored.  The 400 said "nothing
was stepped" and carried no count, while the graph had moved by every
step before the one that raised.  It now carries ``steps_run`` and says
where the run stopped; a failure at the first step still says nothing was
stepped, and a single ``POST /sim/step`` that raises stores nothing.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.testclient import TestClient

eqx = pytest.importorskip("equinox", reason="the run-time check is equinox.error_if")

from maddening.api.server import SimulationServer  # noqa: E402
from maddening.core.graph_manager import GraphManager  # noqa: E402
from maddening.nodes import BallNode  # noqa: E402

DT = 1.0 / 64.0


class _GuardedCounter(BallNode):
    """A counter that raises at run time once its position passes 0.5:
    the 33rd step from zero."""

    def update(self, state, boundary_inputs, dt, *, params=None):
        out = super().update(state, boundary_inputs, dt, params=params)
        return {**out, "position": eqx.error_if(out["position"], out["position"] > 0.5,
                                                "counter passed 0.5")}


def _client(start: float = 0.0):
    gm = GraphManager()
    gm.add_node(_GuardedCounter("c", timestep=DT, initial_position=start,
                                initial_velocity=1.0, gravity=0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    server = SimulationServer({}, graph_manager=gm)
    return server, TestClient(server.create_app(), raise_server_exceptions=False)


def test_a_run_that_fails_part_way_reports_the_steps_it_took(monkeypatch):
    # Slices that always double (1, 2, 4, 8, 16, 32 ...): the 33rd step is
    # the second of the sixth slice, so the count must include the steps a
    # failing slice took, not only the slices that finished.
    from maddening.api import server as server_module

    monkeypatch.setattr(server_module, "_RUN_SLICE_SECONDS", 1e9)
    server, client = _client()
    resp = client.post("/sim/run", params={"n_steps": 100})
    assert resp.status_code == 400, resp.text
    body = resp.json()
    assert body["steps_run"] == 32 and body["n_steps"] == 100
    assert "nothing was stepped" not in body["detail"]
    assert "the run stopped at step 33 of 100" in body["detail"]
    assert "left after the 32 step(s) it took" in body["detail"]
    assert client.get("/graph/state/c").json()["position"] == 0.5
    assert server.relay.step_count == 32


def test_a_run_that_fails_at_its_first_step_says_nothing_was_stepped():
    server, client = _client(start=0.5)
    resp = client.post("/sim/run", params={"n_steps": 10})
    assert resp.status_code == 400, resp.text
    assert resp.json()["steps_run"] == 0
    assert "nothing was stepped" in resp.json()["detail"]
    assert client.get("/graph/state/c").json()["position"] == 0.5


def test_a_single_step_that_fails_stores_nothing():
    server, client = _client(start=0.5)
    resp = client.post("/sim/step")
    assert resp.status_code == 400, resp.text
    assert client.get("/graph/state/c").json()["position"] == 0.5
