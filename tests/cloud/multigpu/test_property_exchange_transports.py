"""Properties of the unstructured halo exchange, over generated partitions.

``exchange_unstructured`` has two transports -- a dense ``all_to_all``
and a per-neighbour ``ppermute`` -- and the docstring promises they
"return bit-identical slabs".  ``test_exchange_ppermute.py`` checks that
on random partitions of a ring with four devices and at least eight
cells; what it cannot reach is the shape of partition a real partitioner
occasionally produces and a hand-written fixture never does: a shard
that owns nothing, one cell per device, an edge-disjoint partition with
no ghosts at all, a self edge, a duplicated edge, a single device.
Those are exactly the cases where an index table is built from a maximum
that happens to be zero, or a shift table from an empty message.

Three invariants, over :func:`partition_layouts`:

* the two transports agree bit for bit;
* every ghost slot holds the value its owner holds (the transports could
  agree on the same wrong answer);
* the sparse transport never moves more cells than the dense one, and
  never fewer than the partition genuinely needs.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from jax import shard_map
from jax.sharding import PartitionSpec as P

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import (
    exchange_traffic,
    exchange_unstructured,
    partition_value,
)
from tests.cloud.multigpu.property_support import partition_layouts
from tests.conftest import EXAMPLES_COSTLY


def _slab(layout, trailing=(), seed=0):
    """A per-shard slab of distinct values, one per global cell."""
    rng = np.random.default_rng(seed)
    n = layout.partition_assignment.size
    values = rng.standard_normal((n,) + trailing).astype(np.float32)
    per_shard = partition_value(value=values, layout=layout)
    return values, jnp.asarray(per_shard.reshape(
        (layout.n_devices * layout.n_local_max,) + trailing))


def _exchanged(layout, x, method):
    mesh = create_device_mesh(shape=(layout.n_devices,))

    def local(v):
        return exchange_unstructured(v, layout=layout, mesh_axis="devices",
                                     method=method)

    fn = jax.jit(shard_map(local, mesh=mesh, in_specs=P("devices"),
                           out_specs=P("devices")))
    return np.asarray(fn(x))


@given(layout=partition_layouts(), trailing=st.sampled_from([(), (2,)]),
       seed=st.integers(0, 2**16))
@settings(max_examples=EXAMPLES_COSTLY)
def test_the_two_exchange_transports_agree_on_any_valid_partition(
        layout, trailing, seed):
    """``all_to_all`` and ``ppermute`` return the same slab, bit for bit."""
    _, x = _slab(layout, trailing, seed)
    dense = _exchanged(layout, x, "all_to_all")
    sparse = _exchanged(layout, x, "ppermute")
    assert dense.shape == sparse.shape
    np.testing.assert_array_equal(dense, sparse)


@given(layout=partition_layouts(),
       method=st.sampled_from(["all_to_all", "ppermute"]),
       seed=st.integers(0, 2**16))
@settings(max_examples=EXAMPLES_COSTLY)
def test_every_ghost_slot_holds_the_value_its_owner_holds(layout, method, seed):
    """The ghost tail is the neighbours' data, not merely a consistent one.

    Both transports could agree on the same wrong permutation, so the
    slab is checked against the global array it was partitioned from:
    device ``d``'s ghost slot ``j`` must be the value of global cell
    ``layout.ghost_global_ids[d][j]``, and the slots past that device's
    own ghost count must be zero.
    """
    values, x = _slab(layout, (), seed)
    out = _exchanged(layout, x, method)
    width = layout.n_local_max + layout.n_ghost_max
    per_device = out.reshape((layout.n_devices, width))
    for d in range(layout.n_devices):
        for j, g in enumerate(layout.ghost_global_ids[d]):
            assert per_device[d, layout.n_local_max + j] == values[int(g)], (
                f"device {d} ghost slot {j} (global cell {g})")
        for j in range(len(layout.ghost_global_ids[d]), layout.n_ghost_max):
            assert per_device[d, layout.n_local_max + j] == 0.0
        # The owned block is untouched by the exchange.
        owned = layout.local_global_ids[d]
        np.testing.assert_array_equal(per_device[d, :len(owned)],
                                      values[owned])


@given(layout=partition_layouts())
@settings(max_examples=EXAMPLES_COSTLY)
def test_the_sparse_transport_never_moves_more_cells_than_the_dense_one(layout):
    """``useful <= ppermute <= all_to_all``, and no message without a cell.

    ``useful`` is a per-shard figure computed by integer division, so it
    reads 0 whenever fewer than ``n_devices`` cells move in total -- it
    is a floor of the average, not a lower bound on the real traffic,
    and "nothing to send" has to be read off ``send_counts`` instead.
    (Hypothesis made the distinction concrete with a four-device
    partition where only two shards exchange one cell each.)
    """
    traffic = exchange_traffic(layout)
    assert traffic["useful"] <= traffic["ppermute"] <= traffic["all_to_all"]
    assert traffic["ppermute_messages"] <= max(layout.n_devices - 1, 0)
    if int(np.asarray(layout.send_counts).sum()) == 0:
        assert traffic["ppermute"] == 0 and traffic["ppermute_messages"] == 0


# ---------------------------------------------------------------------------
# The awkward partitions, pinned explicitly.
#
# The strategy above reaches these, but only by chance and only for as
# many examples as the profile allows; named cases keep them in every
# run and say out loud which shapes are meant to work.
# ---------------------------------------------------------------------------

_AWKWARD = {
    # label: (partition_assignment, edges, n_devices)
    "single_device": ([0, 0, 0, 0], [[0, 1], [1, 2], [2, 3]], 1),
    "empty_shard": ([0, 0, 0, 0], [[0, 1], [1, 2], [2, 3]], 4),
    "one_cell_per_device": ([0, 1, 2, 3], [[0, 1], [1, 2], [2, 3], [3, 0]], 4),
    "edge_disjoint_shards": ([0, 0, 1, 1], [[0, 1], [2, 3]], 2),
    "no_edges_at_all": ([0, 1, 0, 1], [], 2),
    "self_edges_only": ([0, 1, 0, 1], [[0, 0], [1, 1]], 2),
    "duplicate_edges": ([0, 1], [[0, 1], [0, 1], [1, 0]], 2),
    "all_cells_on_the_last_shard": ([3, 3, 3], [[0, 1], [1, 2]], 4),
}


@pytest.mark.parametrize("label", sorted(_AWKWARD))
def test_an_awkward_partition_exchanges_the_same_under_both_transports(label):
    from maddening.cloud.multigpu.halo_unstructured import (
        build_unstructured_partition,
    )

    assignment, edge_list, n_devices = _AWKWARD[label]
    if n_devices > len(jax.devices()):
        pytest.skip(f"needs >={n_devices} CPU-virtual devices")
    layout = build_unstructured_partition(
        partition_assignment=np.asarray(assignment, dtype=np.int32),
        edges=(np.asarray(edge_list, dtype=np.int32) if edge_list
               else np.zeros((0, 2), dtype=np.int32)),
        n_devices=n_devices)
    values, x = _slab(layout, (), seed=3)
    dense = _exchanged(layout, x, "all_to_all")
    np.testing.assert_array_equal(dense, _exchanged(layout, x, "ppermute"))
    width = layout.n_local_max + layout.n_ghost_max
    per_device = dense.reshape((layout.n_devices, width))
    for d in range(layout.n_devices):
        for j, g in enumerate(layout.ghost_global_ids[d]):
            assert per_device[d, layout.n_local_max + j] == values[int(g)]
