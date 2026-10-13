"""The inspection methods on graphs holding sharded wrappers (CPU virtual devices).

Read-only like everywhere else -- reading a sharded array's placement and
gathering it to the host for statistics compiles nothing -- and they
report the wrapper, the mesh, the sharded axis and the per-device memory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode, ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes.heat import HeatNode
from tests.core.inspection_guard_support import (
    INSPECTION_CALLS,
    assert_read_only,
    eager_first_call,
    unavailable,
)

pytestmark = pytest.mark.skipif(len(jax.devices()) < 4, reason="needs >=4 (virtual) devices")


class Pointwise(SimulationNode):
    def halo_width(self) -> dict[int, int]:
        return {}

    def initial_state(self):
        n = self.params.get("n", 16)
        return {"values": jnp.linspace(0.0, 1.0, n, dtype=jnp.float32),
                "rate": jnp.ones(n, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"values": state["values"] + dt * state["rate"], "rate": state["rate"]}


_GRAPHS: dict[str, GraphManager] = {}


def _graph(kind: str) -> GraphManager:
    if kind not in _GRAPHS:
        mesh = create_device_mesh(n_devices=4)
        gm = GraphManager()
        if kind == "pointwise":
            gm.add_node(ShardedPointwiseNode(Pointwise("field", 0.01, n=16), mesh))
            gm.compile()
            gm.step()
        elif kind == "stencil":
            rod = HeatNode("rod", 1e-4, n_cells=32, thermal_diffusivity=0.1,
                           initial_temperature=300.0)
            gm.add_node(ShardedStencilNode(rod, mesh, axis_map={"devices": 0}))
            gm.compile()
            gm.step()
        elif kind == "pointwise_uncompiled":
            gm.add_node(ShardedPointwiseNode(Pointwise("field", 0.01, n=16), mesh))
        _GRAPHS[kind] = gm
    return _GRAPHS[kind]


KINDS = ("pointwise", "stencil", "pointwise_uncompiled")


@pytest.mark.parametrize("method", sorted(INSPECTION_CALLS))
@pytest.mark.parametrize("kind", KINDS)
def test_inspection_of_a_sharded_graph_changes_nothing_and_compiles_nothing(kind, method):
    if (reason := unavailable(method)) is not None:
        pytest.skip(reason)
    gm = _graph(kind)
    call = INSPECTION_CALLS[method]
    assert_read_only(gm, call,
                     allow_eager_compile=eager_first_call(method, uncompiled=kind.endswith(
                         "uncompiled")))
    assert_read_only(gm, call)


def test_graph_text_names_the_wrapper_mesh_and_sharded_axis():
    text = _graph("pointwise").format_graph()
    assert "field  ShardedPointwiseNode(Pointwise)" in text
    assert "sharded: ShardedPointwiseNode over mesh {devices: 4}; state axis (0,)" in text
    assert "rate float32[16@devices]" in text and "values float32[16@devices]" in text
    stencil = _graph("stencil").format_graph()
    assert "rod  ShardedStencilNode(HeatNode)" in stencil
    assert "axis map devices -> state axis 0" in stencil


def test_mermaid_and_dot_label_the_wrapped_type():
    assert "ShardedPointwiseNode(Pointwise)" in _graph("pointwise").to_mermaid()
    assert "ShardedPointwiseNode(Pointwise)" in _graph("pointwise").to_dot()


@pytest.mark.parametrize("kind", KINDS)
def test_memory_estimate_reports_per_device_shares(kind):
    gm = _graph(kind)
    rows = {r["node"]: r for r in gm.memory_estimate()}
    name = "rod" if kind == "stencil" else "field"
    row = rows[name]
    total = sum(int(np.prod(v.shape)) * v.dtype.itemsize for v in gm._state[name].values())
    assert row["bytes"] == total
    assert row["devices"] == 4
    assert row["per_device_bytes"] == total // 4
    assert row["sharding"] == "devices@0"


def test_state_summary_gathers_the_sharded_values():
    gm = _graph("pointwise")
    rows = {r["field"]: r for r in gm.state_summary()}
    values = np.asarray(gm.get_node_state("field")["values"])
    assert rows["values"]["shape"] == (16,)
    assert rows["values"]["min"] == float(values.min())
    assert rows["values"]["max"] == float(values.max())
