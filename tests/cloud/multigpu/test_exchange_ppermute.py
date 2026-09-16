"""Per-neighbour (``ppermute``) unstructured halo exchange vs the dense
``all_to_all`` one (v0.4.0 plan: production sparse halo exchange, the
part that can be settled on CPU-virtual devices).

* bit parity on random partitions of random graphs;
* gradient parity through the exchange;
* traffic accounting: the sparse path moves at most as many cells as the
  dense one and exactly the useful number on a ring;
* the sharded wrapper matches its unsharded node under either transport,
  including per-cell boundary inputs;
* a 10^5-cell ring in the slow lane.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import shard_map
from jax.sharding import PartitionSpec as P

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import (
    build_unstructured_partition,
    exchange_traffic,
    exchange_unstructured,
    partition_value,
)

_N_DEV = 4
pytestmark = pytest.mark.skipif(len(jax.devices()) < _N_DEV, reason="needs 4 CPU-virtual devices")


def _random_graph(rng, n, extra_edges):
    edges = [[i, (i + 1) % n] for i in range(n)]                     # ring keeps it connected
    for _ in range(extra_edges):
        a, b = rng.integers(0, n, 2)
        if a != b:
            edges.append([int(a), int(b)])
    return np.array(edges, dtype=np.int32)


def _layout(rng, n, extra_edges, contiguous):
    if contiguous:
        pa = (np.arange(n) * _N_DEV // n).astype(np.int32)
    else:
        pa = rng.integers(0, _N_DEV, n).astype(np.int32)
        pa[:_N_DEV] = np.arange(_N_DEV)                               # every shard owns a cell
    edges = _random_graph(rng, n, extra_edges)
    return build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=_N_DEV)


def _exchange_fn(mesh, layout, method, trailing=()):
    def local(x):
        return exchange_unstructured(x, layout=layout, mesh_axis="devices", method=method)

    return jax.jit(shard_map(local, mesh=mesh, in_specs=P("devices"), out_specs=P("devices")))


def _slab(rng, layout, trailing=()):
    n = layout.partition_assignment.size
    values = rng.standard_normal((n,) + trailing).astype(np.float32)
    per = partition_value(value=values, layout=layout)
    return jnp.asarray(per.reshape((layout.n_devices * layout.n_local_max,) + trailing))


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("trailing", [(), (3,)])
def test_ppermute_matches_all_to_all_bit_for_bit(seed, contiguous, trailing):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 40))
    layout = _layout(rng, n, extra_edges=int(rng.integers(0, 12)), contiguous=contiguous)
    mesh = create_device_mesh(shape=(_N_DEV,))
    x = _slab(rng, layout, trailing)
    dense = _exchange_fn(mesh, layout, "all_to_all")(x)
    sparse = _exchange_fn(mesh, layout, "ppermute")(x)
    assert dense.shape == sparse.shape
    np.testing.assert_array_equal(np.asarray(dense), np.asarray(sparse))
    # and the ghosts really are the neighbours' values (spot check device 0)
    d0 = np.asarray(sparse).reshape(_N_DEV, layout.n_local_max + layout.n_ghost_max, *trailing)[0]
    gids = layout.ghost_global_ids[0]
    src = np.asarray(x).reshape(_N_DEV, layout.n_local_max, *trailing)
    for j, g in enumerate(gids):
        owner = int(layout.partition_assignment[g])
        np.testing.assert_array_equal(d0[layout.n_local_max + j],
                                      src[owner, layout.local_index_of(owner, int(g))])


def test_gradient_through_ppermute_matches_all_to_all():
    rng = np.random.default_rng(11)
    layout = _layout(rng, 24, extra_edges=6, contiguous=False)
    mesh = create_device_mesh(shape=(_N_DEV,))
    x = _slab(rng, layout)
    w = jnp.asarray(rng.standard_normal(layout.n_devices * (layout.n_local_max + layout.n_ghost_max)),
                    jnp.float32)

    def loss(method):
        f = _exchange_fn(mesh, layout, method)
        return lambda v: jnp.sum(jnp.tanh(f(v)) * w)

    ga = jax.grad(loss("all_to_all"))(x)
    gp = jax.grad(loss("ppermute"))(x)
    np.testing.assert_allclose(np.asarray(gp), np.asarray(ga), rtol=1e-6, atol=1e-7)
    assert float(jnp.max(jnp.abs(gp))) > 0


def test_traffic_accounting_ring_and_random():
    rng = np.random.default_rng(3)
    n = 64
    ring = build_unstructured_partition(
        partition_assignment=(np.arange(n) * _N_DEV // n).astype(np.int32),
        edges=np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int32),
        n_devices=_N_DEV,
    )
    t = exchange_traffic(ring)
    # each shard needs exactly 2 ghosts (its two ring neighbours), one from
    # each adjacent shard: the sparse path moves exactly that, the dense
    # path moves n_devices * n_ghost_max.
    assert t["useful"] == 2 and t["ppermute"] == 2 and t["ppermute_messages"] == 2
    assert t["all_to_all"] == _N_DEV * ring.n_ghost_max == 8
    rnd = _layout(rng, 200, extra_edges=80, contiguous=False)
    t2 = exchange_traffic(rnd)
    assert t2["useful"] <= t2["ppermute"] <= t2["all_to_all"]
    assert t2["ppermute_messages"] <= _N_DEV - 1


def test_wrapper_matches_unsharded_under_both_transports():
    from tests.cloud.multigpu.test_sharded_boundary_inputs import _ring_setup, _SourcedRing  # noqa: E402
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode

    rng = np.random.default_rng(5)
    node, _, layout = _ring_setup(n=32)
    mesh = create_device_mesh(shape=(_N_DEV,))
    src_global = rng.standard_normal(32).astype(np.float32)
    src_layout = jnp.asarray(partition_value(value=src_global, layout=layout).reshape(-1))
    ref = node.initial_state()
    for _ in range(4):
        ref = node.update(ref, {"source": jnp.asarray(src_global)}, 1.0)
    outs = {}
    for method in ("all_to_all", "ppermute"):
        sh = ShardedUnstructuredNode(node, mesh, layout, exchange=method)
        st = sh.initial_state()
        for _ in range(4):
            st = sh.update(st, {"source": src_layout}, 1.0)
        outs[method] = np.asarray(sh.gather_global(st)["x"])
        np.testing.assert_allclose(outs[method], np.asarray(ref["x"]), rtol=1e-6, atol=1e-6)
        assert sh.to_dict()["exchange"] == method
    np.testing.assert_array_equal(outs["all_to_all"], outs["ppermute"])
    with pytest.raises(ValueError, match="exchange"):
        ShardedUnstructuredNode(node, mesh, layout, exchange="carrier-pigeon")


@pytest.mark.slow
def test_hundred_thousand_cell_ring_ppermute_matches_dense():
    n = 100_000
    rng = np.random.default_rng(7)
    pa = rng.integers(0, _N_DEV, n).astype(np.int32)          # scattered: many ghosts
    edges = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int32)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=_N_DEV)
    mesh = create_device_mesh(shape=(_N_DEV,))
    x = _slab(rng, layout, (2,))
    dense = _exchange_fn(mesh, layout, "all_to_all")(x)
    sparse = _exchange_fn(mesh, layout, "ppermute")(x)
    np.testing.assert_array_equal(np.asarray(dense), np.asarray(sparse))
    t = exchange_traffic(layout)
    assert t["ppermute"] <= t["all_to_all"]
