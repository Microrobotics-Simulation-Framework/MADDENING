"""Calibrated params and ParamSpec overrides round-trip through
``GraphManager.to_dict`` / ``from_dict`` (and the config helpers)."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import json

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.rigid_body import RigidBodyNode
from maddening.nodes.spring import SpringDamperNode
from maddening.serialization import config as cfg

REGISTRY = {"SpringDamperNode": SpringDamperNode, "RigidBodyNode": RigidBodyNode}


def _gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5))
    gm.add_node(RigidBodyNode("r", 0.01, mass=2.0, inertia=(1.0, 2.0, 3.0)))
    gm.compile()
    return gm


def test_effective_params_overlay_live_values():
    gm = _gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(45.5, jnp.float32)
    gm.params["nodes"]["r"]["inertia"] = jnp.asarray([4.0, 5.0, 6.0], jnp.float32)
    eff = gm.effective_node_params("s")
    assert eff["stiffness"] == 45.5 and eff["damping"] == 2.0
    assert isinstance(eff["stiffness"], float)
    assert gm.effective_node_params("r")["inertia"] == [4.0, 5.0, 6.0]
    assert gm.effective_node_params("r")["constraints"] == {}     # structural kept
    # the node object itself is not mutated
    assert gm._nodes["s"].node.params["stiffness"] == 30.0


def test_to_dict_from_dict_round_trip_calibrated_values_and_overrides():
    gm = _gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(45.5, jnp.float32)
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    gm.set_param_spec("r", "gravity", ParamSpec(bounds=(-20.0, 0.0), units="m/s^2"))

    d = json.loads(json.dumps(cfg.to_dict(gm)))     # must be JSON-clean
    assert d["param_specs"]["s"]["mass"]["trainable"] is False
    assert d["param_specs"]["r"]["gravity"]["bounds"] == [-20.0, 0.0]

    gm2 = cfg.from_dict(d, REGISTRY)
    gm2.compile()
    assert float(gm2.params["nodes"]["s"]["stiffness"]) == 45.5
    assert gm2.param_specs()["nodes"]["s"]["mass"].trainable is False
    assert gm2.param_specs()["nodes"]["r"]["gravity"] == ParamSpec(
        bounds=(-20.0, 0.0), units="m/s^2")
    # and the trajectory of the reloaded graph matches the calibrated one
    a = gm.run_scan(20)
    b = gm2.run_scan(20)
    for n in ("s", "r"):
        for f in a[n]:
            np.testing.assert_allclose(np.asarray(a[n][f]), np.asarray(b[n][f]),
                                       rtol=1e-6)


def test_no_overrides_means_no_param_specs_key():
    assert "param_specs" not in _gm().to_dict()
