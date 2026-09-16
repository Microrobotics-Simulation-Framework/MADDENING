"""Audit round 2, batch 2: REST entry points, serialisation, retrace, remove_edge."""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

REGISTRY = {"BallNode": BallNode, "TableNode": TableNode, "SpringDamperNode": SpringDamperNode}


def _gm(compile=True):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    if compile:
        gm.compile()
    return gm


def _client(gm):
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm)
    return TestClient(server.create_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# REST: same operation, different graph state
# ---------------------------------------------------------------------------

def test_put_out_of_bounds_before_first_compile_is_400():
    """Entry-point agreement: a PUT that is a 400 on a compiled graph must
    not be a 200 on the same graph before its first compile."""
    gm = _gm(compile=True)
    c = _client(gm)
    r = c.put("/graph/params/s", json={"params": {"stiffness": -5.0}})
    assert r.status_code == 400

    gm2 = _gm(compile=False)
    c2 = _client(gm2)
    r2 = c2.put("/graph/params/s", json={"params": {"stiffness": -5.0}})
    assert r2.status_code == 400, (r2.status_code, r2.json())
    # and the negative stiffness never reaches the step
    gm2.compile()
    with pytest.raises(ValueError):
        gm2.check_params()


def test_get_params_reflects_live_values():
    gm = _gm()
    c = _client(gm)
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)   # what sysid.fit does
    got = c.get("/graph/params/s").json()
    assert got["stiffness"] == pytest.approx(300.0), got


def test_get_params_agrees_with_put_response():
    gm = _gm()
    c = _client(gm)
    put = c.put("/graph/params/s", json={"params": {"stiffness": 55.0}}).json()["params"]
    got = c.get("/graph/params/s").json()
    assert got["stiffness"] == pytest.approx(put["stiffness"])


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class Vec(SimulationNode):
    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)}

    def update(self, s, bi, dt):
        return {"v": s["v"] + dt * bi.get("inp", jnp.zeros_like(s["v"]))}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


def test_unregistered_lambda_transform_round_trip_is_loud_or_faithful():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", 0.01)); gm.add_node(SpringDamperNode("b", 0.01))
    gm.add_edge("a", "b", "position", "anchor_position", transform=lambda x: 2.0 * x)
    d = gm.to_dict()
    try:
        gm2 = GraphManager.from_dict(d, {"SpringDamperNode": SpringDamperNode})
    except Exception as exc:  # loud is acceptable
        assert "transform" in str(exc).lower(), exc
        return
    # silent: the transform must survive
    e = gm2._edges[0]
    assert e.transform is not None and float(e.transform(jnp.float32(1.0))) == 2.0


def test_stale_ordinal_spec_override_after_remove_edge():
    from maddening.core.coupling.mapping import matrix_mapping

    gm = GraphManager()
    gm.add_node(Vec("a", 1.0)); gm.add_node(Vec("b", 1.0))
    H = jnp.eye(3, dtype=jnp.float32)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H), additive=True)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(2 * H), additive=True)
    gm.compile()
    gm.set_param_spec("a.v->b.inp#1", "H", ParamSpec(description="second"))
    gm.set_param_spec("a.v->b.inp", "H", ParamSpec(description="first"))
    gm.remove_edge("a", "b", "v", "inp")
    assert gm.param_spec_overrides() == {}, gm.param_spec_overrides()
    gm.compile()
    d = gm.to_dict()
    GraphManager.from_dict(d, {"Vec": Vec})


def test_remove_node_drops_ordinal_mapping_slots_and_overrides():
    from maddening.core.coupling.mapping import matrix_mapping

    gm = GraphManager()
    gm.add_node(Vec("a", 1.0)); gm.add_node(Vec("b", 1.0))
    H = jnp.eye(3, dtype=jnp.float32)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H), additive=True)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(2 * H), additive=True)
    gm.compile()
    gm.set_param_spec("a.v->b.inp#1", "H", ParamSpec(description="second"))
    gm.remove_node("b")
    assert gm.param_spec_overrides() == {}
    assert gm.params["mappings"] == {}


# ---------------------------------------------------------------------------
# Retrace: params leaves given as Python floats
# ---------------------------------------------------------------------------

def test_python_float_params_leaf_does_not_retrace_the_step():
    gm = _gm()
    gm.step()
    n0 = gm.trace_count
    gm.step(params={"nodes": {"s": {"stiffness": 31.0}}})
    gm.step(params={"nodes": {"s": {"stiffness": 32.0}}})
    gm.step()
    n1 = gm.trace_count
    assert n1 == n0, (n0, n1)


def test_user_assigned_python_float_in_gm_params_does_not_retrace():
    gm = _gm()
    gm.step()
    n0 = gm.trace_count
    gm.params["nodes"]["s"]["stiffness"] = 31.0
    gm.step()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(32.0, jnp.float32)
    gm.step()
    n1 = gm.trace_count
    assert n1 == n0, (n0, n1)


# ---------------------------------------------------------------------------
# reset_state / reset_params / replace-by-remove semantics
# ---------------------------------------------------------------------------

def test_readd_node_after_remove_uses_new_constructor_value():
    gm = _gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    gm.remove_node("s")
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=50.0, damping=2.0, initial_position=1.0))
    gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 50.0


def test_structural_put_on_a_live_key_of_a_params_node():
    """A bool for a live float takes the structural path (documented) --
    the live value must then follow the constructor, not the other way
    round, and the step must not crash."""
    gm = _gm()
    c = _client(gm)
    r = c.put("/graph/params/s", json={"params": {"stiffness": 40}})   # int -> live path?
    assert r.status_code == 200, r.json()
    gm.step()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 40.0
