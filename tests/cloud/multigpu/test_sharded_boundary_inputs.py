"""Grid-shaped boundary inputs on the sharded wrappers (blind-spot review).

``ShardedStencilNode`` used to replicate *every* boundary input, so a
per-cell field (an LBM ``body_force`` map, a ``wall_mask_update``) reached
each shard at its global shape and broke the inner ``update_padded`` (or
silently mismatched).  Grid-shaped inputs are now sharded and halo-padded
exactly like state; scalars and uniform ``(D,)`` vectors stay replicated.
The same rule holds for the unstructured wrapper's per-cell inputs.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.nodes.lbm import LBMNode

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")


def _lbm():
    return LBMNode(name="lbm", timestep=1.0, grid_shape=(8, 8), viscosity=0.05,
                   lattice="D2Q9")


def _perturbed_state(node, seed=0):
    rng = np.random.default_rng(seed)
    st = node.initial_state()
    f = st["f"] * jnp.asarray(1 + 0.05 * rng.standard_normal(st["f"].shape), jnp.float32)
    return {**st, "f": f}


def _grid_inputs(seed=1):
    rng = np.random.default_rng(seed)
    mask = np.zeros((8, 8), np.uint8)
    mask[0, :] = 1
    mask[:, 3] = 1          # a wall column that crosses the shard boundaries
    return {
        "body_force": jnp.asarray(rng.standard_normal((8, 8, 2)) * 1e-3, jnp.float32),
        "wall_mask_update": jnp.asarray(mask),
    }


def test_grid_shaped_inputs_match_unsharded():
    # LBMNode streams periodically, so only the periodic halo policy can
    # match the unsharded node (edge/zero differ at the domain boundary
    # regardless of boundary inputs).
    mesh = create_device_mesh(shape=(4,))
    sh = ShardedStencilNode(_lbm(), mesh, axis_map={"devices": 1}, boundary="periodic")
    un = _lbm()
    bi = _grid_inputs()
    a = b = _perturbed_state(un)
    for _ in range(3):
        a, b = sh.update(a, bi, 1.0), un.update(b, bi, 1.0)
    for k in b:
        np.testing.assert_allclose(np.asarray(a[k], np.float32), np.asarray(b[k], np.float32),
                                   rtol=1e-5, atol=1e-6, err_msg=k)
    # the wall actually acted: cells next to the wall column differ from a
    # run without the mask update
    c = un.update(_perturbed_state(un), {"body_force": bi["body_force"]}, 1.0)
    assert not np.allclose(np.asarray(c["f"]), np.asarray(un.update(_perturbed_state(un), bi, 1.0)["f"]))


def test_uniform_vector_and_scalars_stay_replicated():
    mesh = create_device_mesh(shape=(4,))
    sh = ShardedStencilNode(_lbm(), mesh, axis_map={"devices": 1}, boundary="periodic")
    un = _lbm()
    st = _perturbed_state(un)
    bi = {"body_force": jnp.asarray([1e-4, 0.0], jnp.float32),
          "inlet_pressure": jnp.float32(0.34), "outlet_pressure": jnp.float32(0.33)}
    assert sh._grid_shaped_boundary_inputs(st, bi) == frozenset()
    assert sh._grid_shaped_boundary_inputs(st, _grid_inputs()) == {"body_force", "wall_mask_update"}
    a, b = sh.update(st, bi, 1.0), un.update(st, bi, 1.0)
    np.testing.assert_allclose(np.asarray(a["f"]), np.asarray(b["f"]), rtol=1e-5, atol=1e-6)


def test_switching_input_shapes_recompiles_correctly():
    """The shard_map cache is keyed on the input shape, so a uniform force
    followed by a per-cell force (and back) each get the right specs."""
    mesh = create_device_mesh(shape=(4,))
    sh = ShardedStencilNode(_lbm(), mesh, axis_map={"devices": 1}, boundary="periodic")
    un = _lbm()
    st = _perturbed_state(un)
    grid = _grid_inputs()
    for bi in ({"body_force": jnp.asarray([1e-4, 0.0], jnp.float32)}, grid,
               {"body_force": jnp.asarray([1e-4, 0.0], jnp.float32)}):
        a, b = sh.update(st, bi, 1.0), un.update(st, bi, 1.0)
        np.testing.assert_allclose(np.asarray(a["f"]), np.asarray(b["f"]), rtol=1e-5, atol=1e-6)


def test_verify_node_battery_passes_on_the_sharded_wrapper():
    """``verify_node`` samples every declared boundary input at its
    declared (grid) shape; the wrapper must accept that like the inner."""
    from maddening.testing.verification import verify_node

    mesh = create_device_mesh(shape=(4,))
    sh = ShardedStencilNode(_lbm(), mesh, axis_map={"devices": 1}, boundary="periodic")
    p = (0.3, 0.4)
    res = verify_node(
        sh, bounds={"f": (0.02, 0.2), "wall_mask": (0.0, 1.0)}, max_examples=6,
        dt_range=(1.0, 1.0), derandomize=True,
        boundary_bounds={"inlet_pressure": p, "outlet_pressure": p,
                         "body_force": (-1e-3, 1e-3), "wall_mask_update": (0.0, 1.0)},
    )
    bad = {k: r.status for k, r in res.items() if not r.passed}
    assert not bad, bad
    assert res["params_effective"].status == "PASS"      # viscosity is live through shard_map


def test_gradient_wrt_per_cell_force_through_sharded_step():
    mesh = create_device_mesh(shape=(4,))
    sh = ShardedStencilNode(_lbm(), mesh, axis_map={"devices": 1}, boundary="periodic")
    un = _lbm()
    st = _perturbed_state(un)
    bi = _grid_inputs()

    def loss(node, force):
        out = node.update(st, {**bi, "body_force": force}, 1.0)
        return jnp.sum(out["velocity"] ** 2)

    ga = jax.grad(lambda f: loss(sh, f))(bi["body_force"])
    gb = jax.grad(lambda f: loss(un, f))(bi["body_force"])
    assert np.isfinite(np.asarray(ga)).all()
    np.testing.assert_allclose(np.asarray(ga), np.asarray(gb), rtol=1e-4, atol=1e-7)


# ---------------------------------------------------------------------------
# Unstructured wrapper: per-cell boundary inputs in partition layout
# ---------------------------------------------------------------------------

from maddening.cloud.multigpu.halo_unstructured import (  # noqa: E402
    build_unstructured_partition, partition_value,
)
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode  # noqa: E402
from maddening.core.node import SimulationNode  # noqa: E402
from maddening.core.static_data import StaticArray  # noqa: E402


class _SourcedRing(SimulationNode):
    """``x[i] <- 0.5 * (x[i] + x[next(i)]) + dt * source[i]`` on a ring.

    ``next(i)`` of a shard's last owned cell is a ghost, so the per-cell
    ``source`` must be laid out and ghost-exchanged like ``x`` for the
    sharded result to match.  The slab index of ``next(i)`` is carried as
    a partitioned static (``next_idx``) computed from the layout.
    """

    def __init__(self, name, n, timestep=1.0, next_idx=None, partition_assignment=None):
        super().__init__(name=name, timestep=timestep)
        self._n = n
        self._next_idx = next_idx
        self._pa = partition_assignment

    def state_fields(self):
        return ["x"]

    @property
    def static_data(self):
        if self._next_idx is None:
            return {}
        return {"next_idx": StaticArray(self._next_idx, replication="partition",
                                        partition_assignment=self._pa)}

    def initial_state(self):
        return {"x": jnp.arange(self._n, dtype=jnp.float32) + 1.0}

    def update(self, state, boundary_inputs, dt):
        x = state["x"]
        nxt = jnp.roll(x, -1)
        src = boundary_inputs.get("source", jnp.zeros_like(x))
        return {"x": 0.5 * (x + nxt) + dt * src}

    def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                      shard_info=None):
        x = state_padded["x"]
        n_local = shard_info[0][1]
        nxt = jnp.take(x, static_padded["next_idx"][:n_local], axis=0)
        src = boundary_inputs.get("source")
        src = jnp.zeros(n_local, x.dtype) if src is None else src[:n_local]
        return {"x": 0.5 * (x[:n_local] + nxt) + dt * src}


def _slab_index(layout, device, g):
    """Slab position of global cell ``g`` as seen by ``device``."""
    owned = layout.local_index_of(device, g)
    if owned >= 0:
        return owned
    pos = np.where(layout.ghost_global_ids[device] == g)[0]
    assert len(pos) == 1, (device, g)
    return layout.n_local_max + int(pos[0])


def _ring_setup(n=16, n_devices=4):
    pa = (np.arange(n) * n_devices // n).astype(np.int32)      # contiguous blocks
    edges = np.array([[i, (i + 1) % n] for i in range(n)], dtype=np.int32)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                          n_devices=n_devices)
    mesh = create_device_mesh(shape=(n_devices,))
    next_idx = np.array([_slab_index(layout, int(pa[g]), (g + 1) % n) for g in range(n)],
                        dtype=np.int32)
    node = _SourcedRing("ring", n, next_idx=next_idx, partition_assignment=pa)
    return node, ShardedUnstructuredNode(node, mesh, layout), layout


def test_unstructured_per_cell_input_matches_unsharded():
    node, sharded, layout = _ring_setup()
    rng = np.random.default_rng(3)
    source_global = rng.standard_normal(16).astype(np.float32)
    src_layout = jnp.asarray(partition_value(value=source_global, layout=layout).reshape(-1))
    assert sharded._cell_boundary_inputs({"source": src_layout, "gain": 2.0}) == {"source"}

    ref = node.initial_state()
    st = sharded.initial_state()
    for _ in range(3):
        ref = node.update(ref, {"source": jnp.asarray(source_global)}, 1.0)
        st = sharded.update(st, {"source": src_layout}, 1.0)
    got = sharded.gather_global(st)["x"]
    np.testing.assert_allclose(np.asarray(got), np.asarray(ref["x"]), rtol=1e-6, atol=1e-6)


def test_unstructured_global_order_input_is_refused():
    node, sharded, layout = _ring_setup(n=14)      # 14 cells over 4 shards: layout is 16 rows
    assert layout.n_devices * layout.n_local_max != 14
    with pytest.raises(ValueError, match="global cell order"):
        sharded.update(sharded.initial_state(), {"source": jnp.ones(14, jnp.float32)}, 1.0)
    # a scalar is still fine (replicated)
    sharded.update(sharded.initial_state(), {"gain": jnp.float32(2.0)}, 1.0)
