"""Coupling under sharding regression.

Covers part of M8 of the v0.2 halo-exchange roadmap: verifies that
:class:`ShardedStencilNode` integrates cleanly with the existing
:class:`GraphManager` (edge resolution, step loop, scan).  The
explicit Heat↔LBM thermal-coupling test on the same pencil mesh axis
is a known follow-up (it needs sharded ``wall_mask`` + sharded Zou-He
pressure BCs, both deferred to v0.2.x along with the full
Hagen-Poiseuille validation).

What we verify here:

- A single sharded HeatNode runs inside a ``GraphManager.compile()``
  + ``step()`` loop, its ends held at 0 by external inputs.
- Two sharded HeatNodes coupled via boundary-temperature edges produce
  the unsharded trajectory to float32 rounding -- and the coupling moves
  it, so a sharded path that dropped the edges would fail.
- ``run_scan`` works on the sharded graph.

Until 0.4.0 the coupled test compared means within 10%, which it passed
with the coupling ignored: ``HeatNode.update_padded`` never read the
``left_temperature`` / ``right_temperature`` the edges deliver
(MADD-ANO-030).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.nodes.heat import HeatNode

_HAS_4 = len(jax.devices()) >= 4


def _make_heat(name: str, n_cells: int = 16):
    return HeatNode(
        name=name, timestep=0.01, n_cells=n_cells, length=1.0,
        thermal_diffusivity=0.01,
        initial_temperature=np.sin(
            np.pi * np.linspace(0, 1, n_cells)
        ).astype(np.float32),
    )


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_single_sharded_heat_in_graph_manager():
    """A graph with one sharded HeatNode runs through compile + step."""
    n = 16
    node = _make_heat("heat", n)
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(node, mesh, axis_map={"devices": 0})

    def graph(heat):
        gm = GraphManager()
        gm.add_node(heat)
        # Both ends at 0: external inputs default to zero every step.
        gm.add_external_input("heat", "left_temperature")
        gm.add_external_input("heat", "right_temperature")
        gm.compile()
        return gm

    gm = graph(sharded)
    gm_u = graph(_make_heat("heat", n))

    t0 = float(jnp.mean(gm._state["heat"]["temperature"]))
    for _ in range(20):
        gm.step()
        gm_u.step()
    t_after = float(jnp.mean(gm._state["heat"]["temperature"]))

    # Heat diffuses out through the cold ends -> mean drops
    assert t_after < t0
    assert jnp.isfinite(t_after)
    np.testing.assert_allclose(np.asarray(gm._state["heat"]["temperature"]),
                               np.asarray(gm_u._state["heat"]["temperature"]),
                               rtol=0, atol=1e-6)


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_two_sharded_heats_coupled_via_edges():
    """Two sharded HeatNodes coupled by passing right-edge cell to neighbour.

    Each node's end cell is the OTHER node's end temperature -- a simple
    replicated scalar edge.  The sharded run is the unsharded run to
    float32 rounding: ``update_padded`` closes the rod ends from those
    inputs exactly as ``update`` does.  The coupling is also checked to
    matter, against the same pair uncoupled, so that a sharded path which
    dropped the edge inputs (as every release before 0.4.0 did) cannot
    pass by agreeing with an unsharded run it no longer resembles.
    """
    n = 16

    def pair(wrap, coupled=True):
        gm = GraphManager()
        gm.add_node(wrap(_make_heat("a", n)))
        gm.add_node(wrap(_make_heat("b", n)))
        if coupled:
            gm.add_edge(
                source="a", target="b",
                source_field="temperature", target_field="left_temperature",
                transform=lambda T: T[-1],
            )
            gm.add_edge(
                source="b", target="a",
                source_field="temperature", target_field="right_temperature",
                transform=lambda T: T[0],
            )
        gm.compile()
        return gm

    mesh = create_device_mesh(shape=(4,))
    gm_u = pair(lambda node: node)
    gm_s = pair(lambda node: ShardedStencilNode(node, mesh, axis_map={"devices": 0}))
    gm_free = pair(lambda node: node, coupled=False)

    for _ in range(30):
        gm_u.step()
        gm_s.step()
        gm_free.step()

    for name in ("a", "b"):
        T_u = np.asarray(gm_u._state[name]["temperature"])
        T_s = np.asarray(gm_s._state[name]["temperature"])
        T_free = np.asarray(gm_free._state[name]["temperature"])
        np.testing.assert_allclose(T_s, T_u, rtol=0, atol=1e-6, err_msg=name)
        # The coupling moves the answer by far more than the tolerance.
        assert np.max(np.abs(T_u - T_free)) > 1e-3, name


@pytest.mark.skipif(not _HAS_4, reason="needs >=4 virtual devices")
def test_sharded_heat_run_scan():
    """``run_scan`` works on a single sharded HeatNode graph."""
    n = 16
    node = _make_heat("heat", n)
    mesh = create_device_mesh(shape=(4,))
    sharded = ShardedStencilNode(node, mesh, axis_map={"devices": 0})

    gm = GraphManager()
    gm.add_node(sharded)
    gm.compile()

    final = gm.run_scan(n_steps=30)
    assert jnp.all(jnp.isfinite(final["heat"]["temperature"]))
