import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import io, base64, json, math, warnings
import numpy as np
import jax, jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode
from maddening.nodes.heat import HeatNode
from maddening.fmi import build_model_description
from maddening.fmi.package import MODEL_IDENTIFIER
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge

DT = 1e-2


def _bridge(gm, **kw):
    md = build_model_description(gm, model_name="P", model_identifier=MODEL_IDENTIFIER, **kw)
    sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token, step_fn=gm._compiled_step,
                                  initial_state=gm._state, params=gm.params,
                                  param_specs=gm.param_specs()))
    return md, FmuTcpBridge(sc, md, master_dt=DT)


def _vr(md, name):
    return next(v.value_reference for v in md.variables if v.name == name)


# 1. bridge: an input the importer never set is NOT the advertised start value
def test_bridge_unset_input_is_not_the_advertised_zero():
    gm = GraphManager()
    gm.add_node(HeatNode(name="rod", timestep=DT, n_cells=8, thermal_diffusivity=0.1,
                         initial_temperature=100.0, length=1.0))
    gm.add_external_input("rod", "left_temperature")
    gm.compile()
    md, bridge = _bridge(gm, include_evolving=True)
    lt = _vr(md, "rod.left_temperature")
    var = next(v for v in md.variables if v.name == "rod.left_temperature")
    assert var.start == "0.0"
    assert bridge.handle({"op": "get", "vr": [lt]})["values"] == [0.0]
    for _ in range(5):
        assert bridge.handle({"op": "step", "t": 0.0, "dt": DT})["ok"]
    T_fmu = np.asarray(bridge._sidecar.state["rod"]["temperature"])
    ref = GraphManager()
    ref.add_node(HeatNode(name="rod", timestep=DT, n_cells=8, thermal_diffusivity=0.1,
                          initial_temperature=100.0, length=1.0))
    ref.add_external_input("rod", "left_temperature")
    ref.run(5)                      # zeros for every declared external input
    T_ref = np.asarray(ref.get_node_state("rod")["temperature"])
    np.testing.assert_allclose(T_fmu, T_ref, rtol=1e-6)


# 2. params shape at step: not validated
def test_step_params_with_wrong_shape_is_rejected():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="s", timestep=DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.compile()
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.ones(3, jnp.float32) * 30.0
    with pytest.raises((ValueError, TypeError)):
        gm.step(params=p)
    assert np.shape(gm.get_node_state("s")["position"]) == ()


def test_assigning_wrong_shape_into_gm_params_is_rejected():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="s", timestep=DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.compile()
    gm.params["nodes"]["s"]["stiffness"] = jnp.ones(3, jnp.float32) * 30.0
    with pytest.raises((ValueError, TypeError)):
        gm.step()
    assert np.shape(gm.get_node_state("s")["position"]) == ()


# 3. trace_count under run_scan
def test_trace_count_under_run_scan():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="s", timestep=DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.compile()
    gm.run_scan(3)
    gm.run_scan(3)
    print("trace_count after 2x run_scan:", gm.trace_count)
    assert gm.trace_count >= 1


# 4. checkpoint with a node name containing '/'
def test_checkpoint_node_name_with_slash(tmp_path):
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="a/b", timestep=DT, stiffness=30.0, damping=2.0,
                                 initial_position=0.5))
    gm.compile(); gm.run(2)
    p = gm.save_state(tmp_path / "c.npz")
    gm2 = GraphManager()
    gm2.add_node(SpringDamperNode(name="a/b", timestep=DT, stiffness=30.0, damping=2.0,
                                  initial_position=0.5))
    gm2.load_state(p)
    np.testing.assert_allclose(np.asarray(gm2.get_node_state("a/b")["position"]),
                               np.asarray(gm.get_node_state("a/b")["position"]))


# 5. checkpoint _meta mismatch: checkpoint from a non-multirate graph into a multirate one
def test_checkpoint_meta_missing_for_multirate_graph(tmp_path):
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="s", timestep=DT, stiffness=30.0, damping=2.0, initial_position=0.5))
    gm.add_node(BallNode(name="b", timestep=DT, initial_position=1.0, elasticity=0.7))
    gm.compile(); gm.run(2)
    p = gm.save_state(tmp_path / "c.npz")
    gm2 = GraphManager()
    gm2.add_node(SpringDamperNode(name="s", timestep=DT, stiffness=30.0, damping=2.0, initial_position=0.5))
    gm2.add_node(BallNode(name="b", timestep=2 * DT, initial_position=1.0, elasticity=0.7))
    gm2.load_state(p)
    gm2.run(3)      # KeyError '_meta'?


# 6. bridge: get_state / set_state round trip after reset keeps params & inputs coherent
def test_bridge_set_state_after_reset_and_many_small_vs_one_large():
    def graph():
        gm = GraphManager()
        gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=30.0, damping=2.0, initial_position=0.5))
        gm.add_external_input("spring", "anchor_position")
        gm.compile()
        return gm
    gm = graph()
    md, bridge = _bridge(gm)
    pos, anchor, k = _vr(md, "spring.position"), _vr(md, "spring.anchor_position"), _vr(md, "spring.params.stiffness")
    bridge.handle({"op": "set", "vr": [anchor, k], "values": [0.3, 45.0]})
    for i in range(10):
        bridge.handle({"op": "step", "t": i * DT, "dt": DT})
    small = bridge.handle({"op": "get", "vr": [pos]})["values"]
    snap = bridge.handle({"op": "get_state"})["state"]
    bridge.handle({"op": "reset"})
    assert bridge.handle({"op": "get", "vr": [k, anchor]})["values"] == [30.0, 0.0]
    bridge.handle({"op": "set", "vr": [anchor, k], "values": [0.3, 45.0]})
    bridge.handle({"op": "step", "t": 0.0, "dt": 10 * DT})
    large = bridge.handle({"op": "get", "vr": [pos]})["values"]
    assert small == pytest.approx(large, rel=1e-6)
    bridge.handle({"op": "reset"})
    assert bridge.handle({"op": "set_state", "state": snap})["ok"]
    assert bridge.handle({"op": "get", "vr": [pos, k, anchor]})["values"] == pytest.approx(small + [45.0, 0.3])


# 7. bridge: zip-bomb-ish npz (compressed) DoS -- report only size
def test_bridge_compressed_npz_is_loaded_before_shape_check():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=DT, stiffness=30.0, damping=2.0, initial_position=0.5))
    gm.compile()
    md, bridge = _bridge(gm)
    buf = io.BytesIO()
    big = np.zeros(50_000_000, np.float64)          # 400 MB decompressed
    np.savez_compressed(buf, _token=np.array(md.instantiation_token), **{"s/spring/position": big})
    blob = buf.getvalue()
    print("compressed size:", len(blob))
    assert len(blob) < 1_000_000
    import tracemalloc
    tracemalloc.start()
    r = bridge.handle({"op": "set_state", "state": base64.b64encode(blob).decode()})
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    print("peak MB:", peak / 1e6, r)
    assert not r["ok"]
    assert peak < 100e6
