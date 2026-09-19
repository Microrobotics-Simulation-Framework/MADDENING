"""An ``AdaptiveNode`` must survive a config round trip.

``GraphManager.effective_node_params`` / ``node.params`` is what every
serialisation path (USD, config) stores, and every one of them rebuilds
the node with ``cls(name=..., timestep=..., **params)``.  The class
therefore has to be reconstructible from its own ``params`` -- with the
diagnostic settings it was built with, and without ``n_max`` arriving
twice (audit A4).
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.adaptive import AdaptiveNodeBlindnessError

from tests.nodes.adaptive._toys import MaskedDenseNode, PoissonSineTopKNode


def _rebuild(node):
    return type(node)(name=node.name, timestep=node.delta_t, **node.params)


def test_structural_n_max_is_not_a_constructor_parameter_twice():
    """``n_max`` used to be stored in ``self.params`` and replayed as a
    keyword, so every subclass following the documented pattern got it
    twice: ``TypeError: got multiple values for keyword argument``."""
    node = PoissonSineTopKNode("adaptive", 1.0, n=64, k=16)
    assert "n_max" not in node.params
    assert _rebuild(node).n_max == 64


def test_adaptive_node_survives_a_params_round_trip_reconstruction():
    node = PoissonSineTopKNode(
        "adaptive", 1.0, theta=0.5, n=32, k=8,
        blindness_gate=False, gradient_capture_threshold=0.4,
        blindness_break_delta=0.02, D_threshold=3,
    )
    clone = _rebuild(node)

    assert clone.n_max == 32 == node.n_max
    assert clone.blindness_gate is False
    assert clone.gradient_capture_threshold == 0.4
    assert clone.blindness_break_delta == 0.02
    assert clone.D_threshold == 3
    assert clone.params == node.params
    assert set(clone.params_pytree()) == set(node.params_pytree())

    a, b = node.initial_state(), clone.initial_state()
    assert jnp.array_equal(a["mask"], b["mask"])
    assert jnp.allclose(a["c"], b["c"], atol=1e-14)
    a2 = node.update(a, {}, 1.0, params={"theta": 0.44})
    b2 = clone.update(b, {}, 1.0, params={"theta": 0.44})
    assert jnp.allclose(a2["c"], b2["c"], atol=1e-14)


def test_a_gate_disabled_node_does_not_reload_with_the_gate_on():
    """A scene saved at a legitimate operating point the gate dislikes
    must be reloadable: the setting is part of ``params``."""
    node = PoissonSineTopKNode("adaptive", 1.0, theta=0.5, n=64, k=16,
                               blindness_gate=False)
    clone = _rebuild(node)
    assert clone.blindness_gate is False
    clone.initial_state()   # would raise if the gate came back on

    gated = PoissonSineTopKNode("adaptive", 1.0, theta=0.5, n=64, k=16,
                                on_blind="raise")
    with pytest.raises(AdaptiveNodeBlindnessError):
        _rebuild(gated).initial_state()


def test_the_on_blind_policy_survives_the_round_trip():
    node = PoissonSineTopKNode("adaptive", 1.0, theta=0.48, n=64, k=16,
                               on_blind="ignore")
    clone = _rebuild(node)
    assert clone.on_blind == "ignore"
    clone.initial_state()   # silent: a warning here would fail the suite


def test_effective_node_params_of_a_calibrated_graph_rebuild_the_node():
    """The serialisation path proper: live (calibrated) values written
    over the constructor ones must still reconstruct."""
    gm = GraphManager()
    gm.add_node(PoissonSineTopKNode("adaptive", 1.0, theta=0.42, n=64, k=16))
    gm.compile()
    gm.params["nodes"]["adaptive"]["theta"] = 0.45

    params = gm.effective_node_params("adaptive")
    assert params["theta"] == pytest.approx(0.45, abs=1e-6)
    clone = PoissonSineTopKNode(name="adaptive", timestep=1.0, **params)
    assert clone.n_max == 64
    assert float(clone.params["theta"]) == pytest.approx(0.45, abs=1e-6)


def test_a_dense_subclass_round_trips_too():
    node = MaskedDenseNode("dense", 1.0, n=16, k=4, blindness_gate=False)
    clone = _rebuild(node)
    assert clone.n_max == 16
    assert jnp.allclose(node.initial_state()["c"], clone.initial_state()["c"],
                        atol=1e-12)
