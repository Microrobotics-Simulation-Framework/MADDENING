"""Calibrated params and ParamSpec overrides round-trip through USD."""

import json

import jax.numpy as jnp
import numpy as np
from pxr import Usd

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd


def _gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5))
    gm.add_node(SpringDamperNode("t", 0.01, stiffness=10.0, damping=1.0))
    gm.add_edge("s", "t", "position", "anchor_position")
    gm.compile()
    return gm


def test_calibrated_params_and_overrides_round_trip():
    gm = _gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(45.5, jnp.float32)
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)

    prim = stage.GetPrimAtPath("/Simulation/nodes/s")
    stored = json.loads(prim.GetAttribute("maddening:paramsJson").Get())
    assert stored["stiffness"] == 45.5            # live value, not constructor's
    assert stored["damping"] == 2.0
    overrides = json.loads(prim.GetAttribute("maddening:paramSpecOverridesJson").Get())
    assert overrides == {"mass": ParamSpec(trainable=False).to_dict()}
    # a node without overrides carries no override attribute
    t_attr = stage.GetPrimAtPath("/Simulation/nodes/t").GetAttribute(
        "maddening:paramSpecOverridesJson")
    assert not t_attr or not t_attr.Get()

    gm2 = load_graph_from_usd(stage)
    gm2.compile()
    assert float(gm2.params["nodes"]["s"]["stiffness"]) == 45.5
    assert gm2.param_specs()["nodes"]["s"]["mass"].trainable is False
    assert gm2.param_specs()["nodes"]["t"]["mass"].trainable is True
    a, b = gm.run_scan(20), gm2.run_scan(20)
    np.testing.assert_allclose(np.asarray(a["t"]["position"]),
                               np.asarray(b["t"]["position"]), rtol=1e-6)
