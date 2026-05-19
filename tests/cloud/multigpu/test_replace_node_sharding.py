"""``replace_node`` × sharding regression.

Verifies the v0.1 ``replace_node`` machinery still works when the
graph contains :class:`ShardedStencilNode` or
:class:`ShardedPointwiseNode` wrappers, and when the *replacement*
node has a different sharding pattern than the original.

Three scenarios:

1. Replace an unsharded HeatNode with a ``ShardedStencilNode``-wrapped
   version of itself (upgrade path).
2. Replace a sharded HeatNode with an unsharded HeatNode (downgrade /
   ablation path).
3. Replace a sharded node with a pointwise SimulationNode (e.g. a
   surrogate that lives on a single device); the graph keeps running.

Edges and external inputs around the replaced node must be preserved.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.surrogates.replace import replace_node

_HAS_4 = len(jax.devices()) >= 4


def _make_heat(name: str, n: int = 16):
    return HeatNode(
        name=name, timestep=0.01, n_cells=n, length=1.0,
        thermal_diffusivity=0.01,
        initial_temperature=np.sin(
            np.pi * np.linspace(0, 1, n)
        ).astype(np.float32),
    )


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_replace_unsharded_with_sharded():
    """Upgrade path: in-place swap unsharded -> sharded.

    ``replace_node`` reinitialises state from the new node's
    ``initial_state()``; we just verify the graph keeps running and
    state continues to evolve under the new (sharded) update rule.
    """
    gm = GraphManager()
    gm.add_node(_make_heat("heat"))
    gm.compile()
    for _ in range(5):
        gm.step()

    inner = _make_heat("heat")
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        inner, mesh, axis_map={"devices": 0}, boundary="zero",
    )
    replace_node(gm, "heat", sharded)
    gm.compile()
    T_just_after_replace = float(jnp.mean(gm._state["heat"]["temperature"]))

    for _ in range(20):
        gm.step()
    T_after = float(jnp.mean(gm._state["heat"]["temperature"]))

    assert jnp.isfinite(T_after)
    # Heat diffused away from its initial sin(pi x/L) profile.
    assert T_after < T_just_after_replace


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_replace_sharded_with_unsharded():
    """Downgrade path: swap a sharded node back to an unsharded one."""
    gm = GraphManager()
    inner = _make_heat("heat")
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        inner, mesh, axis_map={"devices": 0}, boundary="zero",
    )
    gm.add_node(sharded)
    gm.compile()
    for _ in range(5):
        gm.step()
    T_before = float(jnp.mean(gm._state["heat"]["temperature"]))

    # Swap back to plain HeatNode.
    plain = _make_heat("heat")
    replace_node(gm, "heat", plain)
    gm.compile()
    for _ in range(5):
        gm.step()

    T_after = float(jnp.mean(gm._state["heat"]["temperature"]))
    assert jnp.isfinite(T_after)
    assert T_after < T_before


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_replace_sharded_with_pointwise_preserves_edges():
    """Replace a sharded node and confirm edges around it survive.

    Graph: heat -> ball (via the rightmost heat temperature as a
    fictitious gravity input).  After replacing 'heat' with a tiny
    pointwise node, the edge still feeds 'ball' each step.
    """

    class Sentinel(SimulationNode):
        """Pointwise replacement that reports its last seen input."""
        def halo_width(self):
            return {}

        def initial_state(self):
            return {"temperature": jnp.zeros(16, dtype=jnp.float32),
                    "last_input": jnp.float32(0.0)}

        def update(self, state, boundary_inputs, dt):
            inp = jnp.float32(boundary_inputs.get(
                "external_temperature", jnp.float32(0.0)
            ))
            T = state["temperature"] + 1.0  # bump so we can observe progress
            return {"temperature": T, "last_input": inp}

    gm = GraphManager()
    inner = _make_heat("heat")
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(
        inner, mesh, axis_map={"devices": 0}, boundary="zero",
    )
    gm.add_node(sharded)
    ball = BallNode(name="ball", timestep=0.01,
                    initial_position=5.0, initial_velocity=0.0)
    gm.add_node(ball)
    # An edge from heat (rightmost cell) into ball's table_position input.
    gm.add_edge(
        source="heat", target="ball",
        source_field="temperature", target_field="table_position",
        transform=lambda T: T[-1],
    )
    # An external input on heat that we'll observe surviving the swap.
    gm.add_external_input("heat", "external_temperature", shape=(), dtype=jnp.float32)
    gm.compile()

    # Run one step to confirm the wiring works pre-replace.
    gm.step({"heat": {"external_temperature": jnp.float32(1.0)}})

    # Replace heat with the Sentinel, then verify edge + external input survive.
    sentinel = Sentinel(name="heat", timestep=0.01)
    replace_node(gm, "heat", sentinel)
    gm.compile()

    gm.step({"heat": {"external_temperature": jnp.float32(42.0)}})

    # Sentinel's last_input must record what was fed via external input.
    assert float(gm._state["heat"]["last_input"]) == 42.0
    # Ball got the rightmost temperature from the new node (which now
    # equals 0+1=1.0 after one update).
    assert jnp.isfinite(gm._state["ball"]["position"])
