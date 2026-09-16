"""Audit round 2, batch 3: REST int minimised, remove_edge/from_dict, Hypothesis
partial-params agreement, ParamSpec interior, subcycling with wide ints."""
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode

REGISTRY = {"SpringDamperNode": SpringDamperNode}


def _gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.compile()
    return gm


def _client(gm):
    return TestClient(SimulationServer(node_registry=REGISTRY, graph_manager=gm).create_app(),
                      raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# REST: JSON integer for a live float leaf
# ---------------------------------------------------------------------------

def test_put_json_int_for_live_float_then_first_step():
    gm = _gm()
    c = _client(gm)
    r = c.put("/graph/params/s", json={"params": {"stiffness": 40}})
    assert r.status_code == 200
    assert type(gm._nodes["s"].node.params["stiffness"]) is int      # root cause
    gm.step()                                                       # ValueError today


def test_put_json_int_for_live_float_after_a_step_breaks_check_and_recompile():
    gm = _gm()
    gm.step()
    c = _client(gm)
    assert c.put("/graph/params/s", json={"params": {"stiffness": 40}}).status_code == 200
    gm.step()                       # cache hit, fine
    gm.check_params()               # ValueError: unknown key 'stiffness'
    gm._dirty = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()                # RuntimeWarning: dropped ... stiffness
    assert "stiffness" in gm.params["nodes"]["s"]


# ---------------------------------------------------------------------------
# remove_edge leaves "#1" override -> to_dict/from_dict
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


def test_from_dict_after_remove_edge_with_ordinal_override():
    from maddening.core.coupling.mapping import matrix_mapping

    gm = GraphManager()
    gm.add_node(Vec("a", 1.0)); gm.add_node(Vec("b", 1.0))
    H = jnp.eye(3, dtype=jnp.float32)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H), additive=True)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(2 * H), additive=True)
    gm.compile()
    gm.set_param_spec("a.v->b.inp#1", "H", ParamSpec(description="second"))
    gm.remove_edge("a", "b", "v", "inp")
    gm.compile()
    GraphManager.from_dict(gm.to_dict(), {"Vec": Vec})


# ---------------------------------------------------------------------------
# Hypothesis: partial params agree across entry points
# ---------------------------------------------------------------------------

KEYS = ["stiffness", "damping", "mass", "rest_length"]


@settings(max_examples=25, deadline=None, suppress_health_check=list(HealthCheck))
@given(
    subset=st.lists(st.sampled_from(KEYS), unique=True, min_size=0, max_size=4),
    vals=st.lists(st.floats(0.5, 50.0), min_size=4, max_size=4),
    live_k=st.floats(5.0, 60.0),
)
def test_partial_params_agree_across_entry_points(subset, vals, live_k):
    gm = _gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(live_k, jnp.float32)
    partial = {"nodes": {"s": {k: jnp.asarray(v, jnp.float32) for k, v in zip(subset, vals)}}}
    full = jax.tree.map(lambda x: x, gm.params)
    full["nodes"]["s"].update(partial["nodes"]["s"])

    gm.reset_state(); a = gm.run_scan(3, params=partial)["s"]["position"]
    gm.reset_state(); b = gm.run_scan(3, params=full)["s"]["position"]
    gm.reset_state()
    for _ in range(3):
        c = gm.step(params=partial)["s"]["position"]
    gm.reset_state()
    s = gm._state
    for _ in range(3):
        s = gm._compiled_step(s, gm._default_external_inputs(), full)
    d = s["s"]["position"]
    np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(c), np.asarray(b), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(d), np.asarray(b), rtol=1e-6)
    # sysid sees the same completed tree
    assert set(gm.trainable_mask(partial)["nodes"]["s"]) == set(KEYS) | {"initial_position", "initial_velocity"}


# ---------------------------------------------------------------------------
# ParamSpec: constrain lands strictly inside for signed / large bounds
# ---------------------------------------------------------------------------

@settings(max_examples=200, deadline=None)
@given(lo=st.floats(-1e6, 1e6), u=st.floats(-120.0, 120.0))
def test_log_constrain_strict_interior_signed_bounds(lo, u):
    spec = ParamSpec(bounds=(lo, None), transform="log")
    p = spec.to_constrained(jnp.asarray(u, jnp.float32))
    spec.check(p)
    back = spec.to_unconstrained(p)
    assert bool(jnp.isfinite(back))
    p2 = spec.to_constrained(back)
    spec.check(p2)


@settings(max_examples=200, deadline=None)
@given(lo=st.floats(-1e6, 1e6), width=st.floats(1e-3, 1e6), u=st.floats(-120.0, 120.0))
def test_logit_constrain_strict_interior_signed_bounds(lo, width, u):
    spec = ParamSpec(bounds=(lo, lo + width), transform="logit")
    p = spec.to_constrained(jnp.asarray(u, jnp.float32))
    spec.check(p)
    back = spec.to_unconstrained(p)
    assert bool(jnp.isfinite(back))
    spec.check(spec.to_constrained(back))


# ---------------------------------------------------------------------------
# Subcycling inside an IFT group with a wide integer leaf
# ---------------------------------------------------------------------------

class Counter(SimulationNode):
    def __init__(self, name, dt):
        super().__init__(name, dt)

    def initial_state(self):
        return {"x": jnp.array(1.0, jnp.float32), "n": jnp.array(0xDEADBEEF, jnp.uint32),
                "big": jnp.array(2**30 + 12345, jnp.int32)}

    def update(self, s, bi, dt):
        other = bi.get("other", jnp.array(0.0, jnp.float32))
        return {"x": s["x"] + dt * (other - s["x"]), "n": s["n"], "big": s["big"] + 1}

    def boundary_input_spec(self):
        return {"other": BoundaryInputSpec(shape=(), description="o")}


@pytest.mark.parametrize("subcycle", ["linear", "constant"])
def test_wide_int_leaves_survive_subcycled_ift_group(subcycle):
    gm = GraphManager()
    gm.add_node(Counter("a", 0.1)); gm.add_node(Counter("b", 0.3))
    gm.add_edge("a", "b", "x", "other"); gm.add_edge("b", "a", "x", "other")
    gm.add_coupling_group(["a", "b"], solver="ift", max_iterations=5, subcycling=True,
                          boundary_interpolation=subcycle)
    gm.compile()
    out = gm.run_scan(3)
    assert int(out["a"]["n"]) == 0xDEADBEEF
    assert int(out["b"]["n"]) == 0xDEADBEEF
    assert int(out["a"]["big"]) == 2**30 + 12345 + 9, int(out["a"]["big"]) - (2**30 + 12345)
    assert int(out["b"]["big"]) == 2**30 + 12345 + 3, int(out["b"]["big"]) - (2**30 + 12345)


# ---------------------------------------------------------------------------
# checkpoint param restore with a shape mismatch
# ---------------------------------------------------------------------------

def test_load_state_rejects_param_shape_mismatch(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state
    gm = _gm()
    path = save_state(gm, tmp_path / "ck")
    data = dict(np.load(path))
    data["_params/s/stiffness"] = np.ones((3,), np.float32)
    np.savez(tmp_path / "bad.npz", **data)
    fresh = _gm()
    with pytest.raises(ValueError):
        load_state(fresh, tmp_path / "bad.npz")
    fresh.step()
