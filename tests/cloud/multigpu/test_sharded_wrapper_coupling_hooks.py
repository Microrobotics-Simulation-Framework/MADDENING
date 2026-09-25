"""The sharded wrappers forward the flux and interface-correction hooks.

``ShardedPointwiseNode`` and ``ShardedStencilNode`` used to forward
neither ``compute_boundary_fluxes`` / ``boundary_flux_spec`` nor
``interface_dof_indices`` / ``compute_interface_correction``.  The first
half failed loudly (a flux edge from a wrapped node did not compile:
"source field 'spring_force' not in state"); the second was silent -- the
wrapper declared no interface DOFs, so a coupled interface was never
corrected.  Both wrappers keep the graph-level state in the inner node's
global view, so they now forward all four.  ``ShardedUnstructuredNode``
cannot (its state is in partition layout, where the inner node's global
indices name other cells) and refuses by name instead of answering ``{}``.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode, ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")


#: One spring per device.  ``ShardedPointwiseNode`` refuses a node with
#: nothing to shard, which a single spring's 0-d state is, so the springs
#: here are a batch of four and every device owns one.
_N_SPRINGS = 4


class _Relay(SimulationNode):
    """``y = a * inp + b``: consumes a flux edge, feeds a BC edge back."""

    def __init__(self, name="R"):
        super().__init__(name, 1e-2)

    def initial_state(self):
        return {"y": jnp.full(_N_SPRINGS, 0.1, jnp.float32)}

    def update(self, state, bi, dt):
        inp = bi.get("inp", jnp.zeros(_N_SPRINGS, jnp.float32))
        return {"y": 0.01 * jnp.asarray(inp, jnp.float32) + 0.1}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(_N_SPRINGS,))}


class _Springs(SpringDamperNode):
    """``SpringDamperNode`` over a batch of springs: its update and flux are
    elementwise, so only the declared input and flux shapes change."""

    def boundary_input_spec(self):
        return {"anchor_position": BoundaryInputSpec(shape=(_N_SPRINGS,))}

    def boundary_flux_spec(self):
        spec = super().boundary_flux_spec()["spring_force"]
        return {"spring_force": dataclasses.replace(spec, shape=(_N_SPRINGS,))}


def _spring(stiffness=40.0):
    return _Springs("s", 1e-2, stiffness=stiffness, rest_length=0.6,
                    initial_position=[0.2, 0.3, 0.4, 0.5],
                    initial_velocity=[0.0] * _N_SPRINGS)


def _spring_relay(node, solver):
    gm = GraphManager()
    gm.add_node(node)
    gm.add_node(_Relay())
    gm.add_edge("s", "R", "spring_force", "inp")
    gm.add_edge("R", "s", "y", "anchor_position")
    if solver is not None:
        gm.add_coupling_group(["s", "R"], max_iterations=30, tolerance=1e-7, solver=solver)
    gm.compile()
    return gm


@pytest.mark.parametrize("solver", [None, "ift"], ids=["staggered", "ift"])
def test_a_flux_edge_from_a_wrapped_pointwise_node_delivers_the_unwrapped_flux(solver):
    """Compiles, and the relay reads the same force as from the bare node."""
    mesh = create_device_mesh(shape=(4,))
    plain = _spring_relay(_spring(), solver)
    wrapped = _spring_relay(ShardedPointwiseNode(_spring(), mesh), solver)
    for _ in range(3):
        a, b = plain.step(), wrapped.step()
    for n in ("s", "R"):
        for f in a[n]:
            np.testing.assert_allclose(np.asarray(b[n][f]), np.asarray(a[n][f]), rtol=1e-6)


def test_a_calibrated_constant_reaches_the_flux_through_the_wrapper():
    """``params`` reaches the inner flux under the one params rule: the
    relay sees the calibrated stiffness's force, not the constructor's."""
    mesh = create_device_mesh(shape=(4,))
    wrapped = _spring_relay(ShardedPointwiseNode(_spring(), mesh), None)
    wrapped.params["nodes"]["s"]["stiffness"] = jnp.asarray(52.0, jnp.float32)
    ref = _spring_relay(_spring(stiffness=52.0), None)
    for _ in range(3):
        a, b = ref.step(), wrapped.step()
    np.testing.assert_allclose(np.asarray(b["R"]["y"]), np.asarray(a["R"]["y"]), rtol=1e-6)


def _heat(**kw):
    return HeatNode("h", 1e-2, n_cells=16, thermal_diffusivity=0.02,
                    initial_temperature=[300.0 + 3 * i for i in range(16)], **kw)


def test_a_stencil_wrapper_forwards_every_coupling_hook_with_params():
    mesh = create_device_mesh(shape=(4,))
    inner = _heat()
    wrapped = ShardedStencilNode(inner, mesh, {"devices": 0})
    assert wrapped.boundary_flux_spec() == inner.boundary_flux_spec()
    assert wrapped.interface_dof_indices() == inner.interface_dof_indices() != {}
    state = wrapped.initial_state()
    bi = {"left_temperature": jnp.asarray(350.0, jnp.float32),
          "right_temperature": jnp.asarray(290.0, jnp.float32)}
    p = {**inner.params_pytree(), "length": jnp.asarray(1.7, jnp.float32)}
    for kwargs in ({}, {"params": p}):
        fw = wrapped.compute_boundary_fluxes(state, bi, 1e-2, **kwargs)
        fi = inner.compute_boundary_fluxes(inner.initial_state(), bi, 1e-2, **kwargs)
        assert set(fw) == set(fi) == {"left_heat_flux", "right_heat_flux"}
        for k in fi:
            np.testing.assert_allclose(float(fw[k]), float(fi[k]), rtol=1e-6)
        cw = wrapped.compute_interface_correction(state, bi, 1e-2, **kwargs)
        ci = inner.compute_interface_correction(inner.initial_state(), bi, 1e-2, **kwargs)
        assert [i for i, _ in cw["temperature"]] == [i for i, _ in ci["temperature"]] == [0, -1]
        for (_, vw), (_, vi) in zip(cw["temperature"], ci["temperature"]):
            np.testing.assert_allclose(float(vw), float(vi), rtol=1e-6)
    # ...and the injected length is the one used: it moves the flux.
    base = wrapped.compute_boundary_fluxes(state, bi, 1e-2)["left_heat_flux"]
    calibrated = wrapped.compute_boundary_fluxes(state, bi, 1e-2, params=p)["left_heat_flux"]
    assert float(base) != pytest.approx(float(calibrated), rel=1e-3)


def test_the_unstructured_wrapper_refuses_interface_dofs_by_name():
    """Partition layout: a global index names another cell.  An answer of
    ``{}`` left the interface silently uncorrected; the wrapper now says
    so, and an inner node without interface DOFs is unaffected."""
    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode

    class Ring(SimulationNode):
        def __init__(self, iface):
            super().__init__("ring", 0.1, k=2.0)
            self._iface = iface

        def initial_state(self):
            return {"x": jnp.arange(16, dtype=jnp.float32)}

        def update(self, state, bi, dt):
            return dict(state)

        def update_padded(self, state_padded, bi, dt, **kwargs):
            return dict(state_padded)

        def interface_dof_indices(self):
            return {"x_bc": ("x", 0)} if self._iface else {}

    mesh = create_device_mesh(shape=(4,))
    pa = (np.arange(16) * 4 // 16).astype(np.int32)
    edges = np.array([[i, (i + 1) % 16] for i in range(16)], dtype=np.int32)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=4)
    assert ShardedUnstructuredNode(Ring(False), mesh, layout).interface_dof_indices() == {}
    with pytest.raises(NotImplementedError, match=r"'ring'.*\['x_bc'\].*partition layout"):
        ShardedUnstructuredNode(Ring(True), mesh, layout).interface_dof_indices()
