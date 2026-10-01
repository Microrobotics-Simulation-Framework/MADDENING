"""``exchange_unstructured`` on a layout without ghost cells returns the local block.

It used to join the block to a zero-size ghost tail, under both
transports, and on jaxlib 0.11.2 the transpose of that inside a
``lax.scan`` -- reverse mode through ``run_scan`` -- segfaulted XLA's
compiler.  The crash itself is pinned in a subprocess by
``test_differential_sharding.py::test_a_gradient_through_a_partition_without_ghosts_does_not_crash``
(both transports).  These are the in-process neighbours: what the
exchange returns, for every field shape the wrapper hands it (state,
per-cell inputs, partitioned statics), and the forward graph on such a
partition, which the change also reaches.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import PartitionSpec as P

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import (
    build_unstructured_partition,
    exchange_unstructured,
)
from tests.cloud.multigpu import differential_sharding_support as D

pytestmark = pytest.mark.skipif(
    len(jax.devices()) < 4,
    reason="needs 4 devices (the directory conftest forces 16 virtual CPU devices)")


def _edge_disjoint_layout():
    """Two devices, two cells each, no edge between the shards."""
    layout = build_unstructured_partition(
        partition_assignment=np.array([0, 0, 1, 1], np.int32),
        edges=np.array([[0, 1], [2, 3]], np.int32), n_devices=2)
    assert layout.n_ghost_max == 0
    return layout


@pytest.mark.parametrize("trailing", [(), (3,), (2, 2)], ids=["scalar-cells", "vector", "matrix"])
@pytest.mark.parametrize("method", ["all_to_all", "ppermute"])
def test_the_exchange_returns_the_local_block_unchanged(method, trailing):
    layout = _edge_disjoint_layout()
    mesh = create_device_mesh(shape=(2,))
    values = jnp.arange(4 * int(np.prod(trailing, dtype=int)), dtype=jnp.float32).reshape(
        (4,) + trailing)

    def body(local):
        out = exchange_unstructured(local, layout=layout, mesh_axis="devices", method=method)
        assert out.shape == local.shape        # no tail, not even a zero-size one
        return out

    got = jax.jit(shard_map(body, mesh=mesh, in_specs=P("devices"), out_specs=P("devices")))(values)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(values))


def test_an_unknown_method_is_still_refused_on_a_layout_without_ghosts():
    with pytest.raises(ValueError, match="method must be 'all_to_all' or 'ppermute'"):
        exchange_unstructured(jnp.zeros(2), layout=_edge_disjoint_layout(),
                              mesh_axis="devices", method="alltoall")


@pytest.mark.parametrize("exchange", ["all_to_all", "ppermute"])
def test_a_graph_with_every_cell_on_one_device_answers_as_unsharded(exchange):
    """An uneven partition that leaves a shard empty: no edge crosses, no ghosts."""
    cfg = D.UnstructuredConfig(
        n_devices=2, n_cells=5, assignment=(0, 0, 0, 0, 0), partition="uneven",
        chords=((1, 3),), contract="params", integral="vector", integral_name="a_total",
        integral_listed=False, weight="halo", source="per_cell", misshapen_len=0, gain=True,
        dtype="float32", wrapping="hybrid", steps=2, seed=10, surface="run_scan",
        exchange=exchange)
    assert int(D.unstructured_layout(cfg).n_ghost_max) == 0
    D.check_config(cfg, surfaces=("run_scan", "step", "set_state", "write_compile"))
