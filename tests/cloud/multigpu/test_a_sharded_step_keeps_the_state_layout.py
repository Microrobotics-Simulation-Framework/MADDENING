"""A graph with a sharded node is held to its state's layout at a step,
like any other (``GraphManager._store_stepped_state``, MADD-ANO-220).

The stepped state of a sharded node is compared as it is stored: leaves
placed over the mesh have their global shapes, which a step keeps.  A
node beside the sharded one that broadcasts a leaf at its first step is
refused by name, and the sharded state is the one the graph held.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode
from maddening.nodes.heat import HeatNode

DT = 1e-4

pytestmark = pytest.mark.skipif(len(jax.devices()) < 4, reason="needs >=4 devices")


def _graph(velocity) -> GraphManager:
    gm = GraphManager()
    rod = HeatNode("rod", DT, n_cells=32, thermal_diffusivity=0.1, initial_temperature=300.0)
    gm.add_node(ShardedStencilNode(rod, create_device_mesh(n_devices=4),
                                   axis_map={"devices": 0}))
    gm.add_node(BallNode("b", DT, initial_velocity=velocity))
    gm.compile()
    return gm


def _layout(gm: GraphManager) -> dict:
    return {f"{node}/{key}": (np.shape(leaf), np.asarray(leaf).dtype.kind)
            for node, fields in gm._state.items() for key, leaf in fields.items()}


def test_a_healthy_sharded_graph_steps_and_keeps_its_layout():
    gm = _graph(1.0)
    layout = _layout(gm)
    gm.step()
    gm.run(2)
    gm.run_adaptive(3 * DT, dt_initial=DT, dt_max=DT)
    # (The step of this graph is traced twice -- the plain node's leaves
    # come back placed on the mesh -- so it is compared twice, and passes.)
    assert _layout(gm) == layout


@pytest.mark.parametrize("entry", ["step", "run", "run_adaptive"])
def test_a_node_beside_a_sharded_one_that_reshapes_its_state_is_refused(entry):
    gm = _graph([1.0, 2.0])
    held = gm._state
    call = {"step": gm.step, "run": lambda: gm.run(2),
            "run_adaptive": lambda: gm.run_adaptive(3 * DT, dt_initial=DT, dt_max=DT)}[entry]
    with pytest.raises(ValueError, match=r"'b/position' has shape \(\) before the update "
                                         r"and \(2,\) after it"):
        call()
    assert gm._state is held
