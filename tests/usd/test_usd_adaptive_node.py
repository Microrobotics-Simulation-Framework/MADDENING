"""An ``AdaptiveNode`` must survive a USD save/load round trip.

``save_graph_to_usd`` stores ``gm.effective_node_params(...)`` and
``load_graph_from_usd`` rebuilds with ``cls(name=..., timestep=...,
**params)``.  Before the audit fix that path raised ``TypeError: got
multiple values for keyword argument 'n_max'``, and a node deliberately
built with ``blindness_gate=False`` came back with the gate on and
failed ``add_node`` (audit A4).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from pxr import Usd

import maddening.usd  # noqa: F401 (schema registration)
from maddening.core.graph_manager import GraphManager
from maddening.usd.serialization import (
    load_graph_from_usd,
    register_node_class,
    save_graph_to_usd,
)

from tests.nodes.adaptive._toys import PoissonSineTopKNode

register_node_class(PoissonSineTopKNode)


def _round_trip(gm):
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    return load_graph_from_usd(stage)


def test_adaptive_node_survives_a_usd_round_trip_with_a_non_default_n_max():
    gm = GraphManager()
    gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=0.42, n=64, k=16))
    gm.compile()
    before = gm.run_scan(2)["adaptive"]

    gm2 = _round_trip(gm)
    node2 = gm2._nodes["adaptive"].node
    assert node2.n_max == 64
    assert node2.params["k"] == 16
    gm2.compile()
    after = gm2.run_scan(2)["adaptive"]

    assert np.array_equal(np.asarray(before["mask"]), np.asarray(after["mask"]))
    assert jnp.allclose(before["c"], after["c"], atol=1e-6)


def test_a_gate_disabled_adaptive_node_reloads_with_the_gate_still_off():
    """A scene saved at an operating point the diagnostic rejects must
    reload: ``blindness_gate`` is recorded in ``params``."""
    gm = GraphManager()
    gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=0.5, n=64, k=16,
                                    blindness_gate=False))
    gm.compile()

    gm2 = _round_trip(gm)   # add_node would raise otherwise
    node2 = gm2._nodes["adaptive"].node
    assert node2.blindness_gate is False
    assert float(node2.params["theta"]) == 0.5


def test_calibrated_parameters_and_diagnostic_settings_round_trip():
    gm = GraphManager()
    gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=0.42, n=32, k=8,
                                    blindness_gate=False,
                                    gradient_capture_threshold=0.3,
                                    on_blind="ignore"))
    gm.compile()
    gm.params["nodes"]["adaptive"]["theta"] = 0.45

    node2 = _round_trip(gm)._nodes["adaptive"].node
    assert node2.n_max == 32
    assert node2.gradient_capture_threshold == 0.3
    assert node2.on_blind == "ignore"
    assert float(node2.params["theta"]) == np.float32(0.45).item()
