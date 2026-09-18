"""``/graph/params/{node}``: the live pytree is what the endpoint reads and
writes.  A JSON integer for a float leaf keeps the constructor's Python
type (so the leaf stays in the pytree across a recompile), ``GET``
returns the live view (what a fit or a checkpoint restore wrote), and a
``PUT`` before the first compile is validated exactly like one after it.

Originally written from the independent audit of 2026-09-16 (round 2; report and
reproducers under ``benchmarks/results/audit2/``).
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode

REGISTRY = {"SpringDamperNode": SpringDamperNode}


def _spring(compile=True):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    if compile:
        gm.compile()
    return gm


def _client(gm):
    return TestClient(SimulationServer(node_registry=REGISTRY, graph_manager=gm).create_app(),
                      raise_server_exceptions=False)


def test_put_json_int_keeps_constructor_type_and_next_step_works():
    gm = _spring()
    c = _client(gm)
    assert c.put("/graph/params/s", json={"params": {"stiffness": 40}}).status_code == 200
    assert type(gm._nodes["s"].node.params["stiffness"]) is float
    gm.step()
    gm.check_params()
    gm._dirty = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 40.0


def test_get_params_returns_the_live_view():
    gm = _spring()
    c = _client(gm)
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    assert c.get("/graph/params/s").json()["stiffness"] == pytest.approx(300.0)
    put = c.put("/graph/params/s", json={"params": {"stiffness": 55.0}}).json()["params"]
    assert c.get("/graph/params/s").json()["stiffness"] == pytest.approx(put["stiffness"])


def test_put_before_first_compile_is_validated_like_after():
    gm = _spring(compile=False)
    c = _client(gm)
    r = c.put("/graph/params/s", json={"params": {"stiffness": -5.0}})
    assert r.status_code == 400 and "below bound" in r.json()["detail"]
    r = c.put("/graph/params/s", json={"params": {"stiffness": "x"}})
    assert r.status_code == 400
    assert c.put("/graph/params/s", json={"params": {"stiffness": 45}}).status_code == 200
    gm.compile()
    gm.check_params()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 45.0
    assert type(gm._nodes["s"].node.params["stiffness"]) is float


def test_a_param_write_reaches_an_already_cached_run_scan():
    """A slider write must not be served past by the cached scan program.

    ``run_scan`` builds its ``lax.scan`` once per ``compile()`` and reuses
    it; the params pytree is an argument of that program, not a constant
    baked into it, so a ``PUT`` between two scans changes the trajectory
    without rebuilding anything.
    """
    gm = GraphManager()
    # Off the rest length, so the trajectory actually depends on stiffness.
    gm.add_node(SpringDamperNode(
        "s", 0.01, stiffness=30.0, damping=2.0, initial_position=0.5,
    ))
    gm.compile()
    start = dict(gm.get_node_state("s"))

    soft = float(gm.run_scan(10)["s"]["position"])
    assert gm.scan_trace_count == 1

    c = _client(gm)
    assert c.put("/graph/params/s",
                 json={"params": {"stiffness": 400.0}}).status_code == 200

    gm.set_node_state("s", start)
    stiff = float(gm.run_scan(10)["s"]["position"])
    assert gm.scan_trace_count == 1, "the PUT rebuilt the cached scan"
    assert soft != pytest.approx(stiff), (
        "the cached scan served the pre-PUT stiffness"
    )
