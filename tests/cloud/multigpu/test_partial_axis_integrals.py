"""C4 (v0.4.0 plan): partial-axis reductions for domain integrals.

A body-surface drag on a partitioned mesh lives on a subset of shards and
a per-slab integral must not be summed across slabs: ``domain_integral_axes``
names the mesh axes to reduce over; the result keeps one leading axis per
unreduced mesh axis.  Checked on a 2x2 CPU-virtual pencil mesh against
the unsharded sums, and on the unstructured wrapper (stacked per-shard
values when no axis is reduced).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.node import SimulationNode

_HAS_4 = len(jax.devices()) >= 4


class Toy3D(SimulationNode):
    """3-D Laplacian with three integrals: full mesh, along one axis, none."""

    def halo_width(self):
        return {0: 1, 1: 1, 2: 1}

    def initial_state(self):
        rng = np.random.default_rng(1)
        return {"field": jnp.asarray(rng.standard_normal((4, 4, 8)).astype(np.float32))}

    def domain_integral_fields(self):
        return {"total", "per_y", "per_shard"}

    def domain_integral_axes(self):
        return {"per_y": ("spatial_z",), "per_shard": ()}   # "total": full mesh

    def _integrals(self, core):
        s = jnp.sum(core ** 2)
        return {"total": s, "per_y": s, "per_shard": s}

    def update(self, state, boundary_inputs, dt):
        f = state["field"]
        out = {"field": f}
        out.update(self._integrals(f))
        return out

    def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                      shard_info=None):
        f = state_padded["field"]
        core = f[1:-1, 1:-1, 1:-1]
        out = {"field": f}
        out.update(self._integrals(core))
        return out


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_partial_axis_reduction_on_2x2_pencil_mesh():
    mesh = create_device_mesh(shape=(2, 2), axis_names=("spatial_y", "spatial_z"))
    node = Toy3D(name="toy", timestep=0.1)
    sharded = ShardedStencilNode(node, mesh, axis_map={"spatial_y": 1, "spatial_z": 2},
                                 boundary="periodic")
    state = node.initial_state()
    f = np.asarray(state["field"])
    out = sharded.update(state, {}, 0.1)

    # full mesh: one replicated scalar
    total = np.asarray(jax.device_get(out["total"]))
    assert total.shape == () and np.isclose(total, np.sum(f ** 2), rtol=1e-5)

    # reduced over spatial_z only: one value per spatial_y shard (2 slabs of y)
    per_y = np.asarray(jax.device_get(out["per_y"]))
    assert per_y.shape == (2,)
    expected_y = [np.sum(f[:, :2, :] ** 2), np.sum(f[:, 2:, :] ** 2)]
    np.testing.assert_allclose(per_y, expected_y, rtol=1e-5)

    # no reduction: the 2x2 grid of per-shard partial sums
    per_shard = np.asarray(jax.device_get(out["per_shard"]))
    assert per_shard.shape == (2, 2)
    expected = [[np.sum(f[:, yi * 2:(yi + 1) * 2, zi * 4:(zi + 1) * 4] ** 2) for zi in range(2)]
                for yi in range(2)]
    np.testing.assert_allclose(per_shard, expected, rtol=1e-5)
    np.testing.assert_allclose(per_shard.sum(), total, rtol=1e-5)


def test_unknown_axis_rejected():
    class Bad(Toy3D):
        def domain_integral_axes(self):
            return {"per_y": ("nope",)}

    mesh = create_device_mesh(shape=(1,))
    with pytest.raises(ValueError, match="not in mesh.axis_names"):
        ShardedStencilNode(Bad(name="b", timestep=0.1), mesh, axis_map={"devices": 1},
                           boundary="periodic").update(Bad(name="b", timestep=0.1).initial_state(), {}, 0.1)


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_unstructured_stacked_per_shard_values():
    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
    from tests.cloud.multigpu.test_sharded_unstructured import _NeighbourAverageNode

    class Stacked(_NeighbourAverageNode):
        def domain_integral_axes(self):
            return {"total_mass": ()}

    n_global, n_devices = 16, 4
    pa = (np.arange(n_global) % n_devices).astype(np.int32)
    edges = np.array([[i, (i + 1) % n_global] for i in range(n_global)], dtype=np.int32)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                          n_devices=n_devices)
    mesh = create_device_mesh(shape=(n_devices,))
    mass = np.arange(n_global, dtype=np.float32) + 1.0
    node = Stacked(name="toy", n_global_cells=n_global, edges=edges, mass=mass,
                   partition_assignment=pa)
    sharded = ShardedUnstructuredNode(node, mesh, layout)
    out = sharded.update(sharded.initial_state(), {}, 1.0)
    per = np.asarray(jax.device_get(out["total_mass"]))
    assert per.shape == (n_devices,)
    expected = [mass[pa == d].sum() for d in range(n_devices)]
    np.testing.assert_allclose(per, expected, rtol=1e-6)
    assert np.isclose(per.sum(), 136.0)


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_unstructured_unknown_axis_rejected_at_construction():
    """The stencil wrapper refuses a mesh axis the mesh does not have; the
    unstructured one read a misspelt name as "not this axis" and returned
    the stacked per-shard partials where the declared reduction was the
    scalar.  Refused by name at construction, and the correct spelling
    still reduces."""
    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
    from tests.cloud.multigpu.test_sharded_unstructured import _NeighbourAverageNode

    def declaring(axes):
        class Declares(_NeighbourAverageNode):
            def domain_integral_axes(self):
                return {"total_mass": axes}
        return Declares

    n_global, n_devices = 16, 4
    pa = (np.arange(n_global) % n_devices).astype(np.int32)
    edges = np.array([[i, (i + 1) % n_global] for i in range(n_global)], dtype=np.int32)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                          n_devices=n_devices)
    mesh = create_device_mesh(shape=(n_devices,))

    def build(axes):
        node = declaring(axes)(name="toy", n_global_cells=n_global, edges=edges,
                               partition_assignment=pa)
        return ShardedUnstructuredNode(node, mesh, layout)

    with pytest.raises(ValueError, match=r"domain_integral_axes\['total_mass'\].*\['devics'\] "
                                         r"not in mesh.axis_names"):
        build(("devics",))
    sharded = build(("devices",))
    out = sharded.update(sharded.initial_state(), {}, 1.0)
    assert np.asarray(jax.device_get(out["total_mass"])).shape == ()
    assert np.isclose(float(jax.device_get(out["total_mass"])), n_global)
