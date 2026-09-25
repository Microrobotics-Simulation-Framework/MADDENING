"""A sharded node takes the input shapes the unsharded node takes, no others.

``ShardedStencilNode`` shards a boundary input that is grid-shaped and
replicates anything else, so every shard used to receive a mis-shaped
input whole -- and could not tell it from a local one:

* ``HeatNode.update_padded`` broadcast any 1-D ``heat_source`` of
  ``n_local`` values and stripped the ends off one of ``n_local + 2*halo``,
  so on a 16-cell rod split four ways a source of 4 values was applied on
  every block (one pattern repeated along the rod) and one of 6 as a
  halo-padded block;
* ``LBMNode.update_padded`` zero-padded any ``body_force`` that was neither
  ``(D,)`` nor the padded block, so a ``(4, 8, 2)`` force on a ``(16, 8)``
  grid was applied on every slab.

The unsharded node refuses all of them.  Now the wrapper refuses a
replicated input the node declares per cell unless it broadcasts to the
declared shape, and both nodes' ``update_padded`` take only what their
``update`` takes, translated to the block.
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
from maddening.nodes.lbm import LBMNode

_HAS_4 = len(jax.devices()) >= 4
pytestmark = pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")

N = 16
DT = 0.2 * (1.0 / N) ** 2 / 0.01


def _rod(order=2):
    return HeatNode("rod", DT, n_cells=N, initial_temperature=300.0,
                    stencil_order=order)


def _sharded_rod(n_devices, order=2):
    return ShardedStencilNode(_rod(order), create_device_mesh(shape=(n_devices,)),
                              {"devices": 0}, boundary="edge")


def _source(length):
    return jnp.arange(1.0, length + 1.0, dtype=jnp.float32) * 100.0


# ---------------------------------------------------------------------------
# HeatNode heat_source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_devices, order, length", [
    (4, 2, 4),     # n_local: used to repeat along the rod
    (4, 2, 6),     # n_local + 2*halo: used to be read as a padded block
    (2, 2, 8),
    (2, 2, 10),
    (4, 4, 4),
    (4, 4, 8),     # n_local + 2*halo at halo 2
    (4, 2, 3),
    (4, 2, 18),    # the global rod plus halos
], ids=lambda v: str(v))
def test_a_heat_source_the_rod_refuses_is_refused_sharded(n_devices, order, length):
    node = _rod(order)
    with pytest.raises(ValueError):
        node.update(node.initial_state(), {"heat_source": _source(length)}, DT)
    wrapped = _sharded_rod(n_devices, order)
    with pytest.raises(ValueError, match=rf"heat_source.*\({length},\)"):
        wrapped.update(wrapped.initial_state(), {"heat_source": _source(length)}, DT)


@pytest.mark.parametrize("source", [
    jnp.float32(50.0), jnp.asarray([50.0], jnp.float32), _source(N),
], ids=["scalar", "one", "per-cell"])
@pytest.mark.parametrize("n_devices", [2, 4])
def test_a_heat_source_the_rod_takes_steps_the_sharded_rod_the_same(source, n_devices):
    node = _rod()
    want = node.initial_state()
    wrapped = _sharded_rod(n_devices)
    got = wrapped.initial_state()
    for _ in range(3):
        want = node.update(want, {"heat_source": source}, DT)
        got = wrapped.update(got, {"heat_source": source}, DT)
    np.testing.assert_allclose(np.asarray(got["temperature"]),
                               np.asarray(want["temperature"]), rtol=1e-6)
    assert float(jnp.max(want["temperature"])) > 300.0


def test_update_padded_refuses_an_unpadded_block_source_on_a_split_rod():
    """The node's own check, for a caller that is not the wrapper.

    A block of 4 of the rod's 16 cells is handed its cells halo-padded, 6
    values; 4 values are not something the wrapper ever delivers, and used
    to be broadcast as if they were this block's.
    """
    node = _rod()
    padded = {"temperature": jnp.full(4 + 2, 300.0, jnp.float32)}
    info = {0: (jnp.int32(4), 4)}
    with pytest.raises(ValueError, match=r"heat_source has shape \(4,\)"):
        node.update_padded(padded, {"heat_source": _source(4)}, DT, shard_info=info)
    out = node.update_padded(padded, {"heat_source": _source(6)}, DT, shard_info=info)
    np.testing.assert_allclose(
        np.asarray(out["temperature"][1:-1]), 300.0 + np.asarray(_source(6)[1:-1]) * DT,
        rtol=1e-6)


@pytest.mark.parametrize("padded_source", [False, True], ids=["cells", "padded"])
def test_update_padded_on_the_whole_rod_takes_a_per_cell_source(padded_source):
    """A direct call holds the whole rod: ``(n_cells,)`` is its block."""
    node = _rod()
    rng = np.random.default_rng(3)
    T = jnp.asarray(300.0 + rng.standard_normal(N), jnp.float32)
    src = jnp.asarray(rng.standard_normal(N) * 100.0, jnp.float32)
    ends = {"left_temperature": jnp.float32(250.0), "right_temperature": jnp.float32(350.0)}
    want = node.update({"temperature": T}, {**ends, "heat_source": src}, DT)
    given = jnp.pad(src, 1) if padded_source else src
    got = node.update_padded({"temperature": jnp.pad(T, 1, mode="edge")},
                             {**ends, "heat_source": given}, DT)
    np.testing.assert_allclose(np.asarray(got["temperature"][1:-1]),
                               np.asarray(want["temperature"]), rtol=1e-6)


def test_a_mis_shaped_external_heat_source_is_refused_by_a_sharded_graph():
    """The path the finding was reached by: ``add_external_input``."""
    for node, match in ((_rod(), None), (_sharded_rod(4), r"heat_source.*\(4,\)")):
        gm = GraphManager()
        gm.add_node(node)
        gm.add_external_input("rod", "heat_source", shape=(4,))
        with pytest.raises(ValueError, match=match):
            gm.compile()
            gm.run_scan(2, external_inputs={"rod": {"heat_source": _source(4)}})


# ---------------------------------------------------------------------------
# LBMNode body_force
# ---------------------------------------------------------------------------


GRID = (16, 8)


def _lbm():
    return LBMNode("lbm", 1.0, grid_shape=GRID, viscosity=0.1, lattice="D2Q9")


def _sharded_lbm(axis):
    return ShardedStencilNode(_lbm(), create_device_mesh(shape=(4,)),
                              {"devices": axis}, boundary="periodic")


@pytest.mark.parametrize("axis, shape", [
    (0, (4, 8, 2)),      # the block's cells: used to be zero-padded, per slab
    (0, (6, 10, 2)),     # the padded block
    (1, (16, 2, 2)),
    (1, (18, 4, 2)),
    (0, ()),             # update indexes force[..., None, :]
], ids=lambda v: str(v))
def test_a_body_force_the_node_refuses_is_refused_sharded(axis, shape):
    force = jnp.full(shape, 1e-4, jnp.float32)
    node = _lbm()
    with pytest.raises((TypeError, ValueError, IndexError)):
        node.update(node.initial_state(), {"body_force": force}, 1.0)
    wrapped = _sharded_lbm(axis)
    with pytest.raises(ValueError, match="body_force"):
        wrapped.update(wrapped.initial_state(), {"body_force": force}, 1.0)


@pytest.mark.parametrize("axis, kind", [
    (0, "uniform"), (1, "one"), (0, "per-cell"), (1, "per-cell"),
], ids=lambda v: str(v))
def test_a_body_force_the_node_takes_steps_the_sharded_node_the_same(axis, kind):
    rng = np.random.default_rng(5)
    force = {
        "uniform": jnp.asarray([1e-4, -2e-5], jnp.float32),
        "one": jnp.asarray([1e-4], jnp.float32),
        "per-cell": jnp.asarray(rng.standard_normal(GRID + (2,)) * 1e-4, jnp.float32),
    }[kind]
    node = _lbm()
    step = jax.jit(lambda s: node.update(s, {"body_force": force}, 1.0))
    want = node.initial_state()
    wrapped = _sharded_lbm(axis)
    got = wrapped.initial_state()
    for _ in range(3):
        want = step(want)
        got = wrapped.update(got, {"body_force": force}, 1.0)
    # atol: the velocity (~3e-4) is a difference of populations ~0.1, so
    # float32 rounding of f alone moves it by ~1e-8.
    np.testing.assert_allclose(np.asarray(got["velocity"]), np.asarray(want["velocity"]),
                               rtol=1e-4, atol=1e-7)
    assert float(jnp.max(jnp.abs(want["velocity"]))) > 1e-5


def test_update_padded_refuses_an_unpadded_block_force_on_a_split_grid():
    """The node's own check, for a caller that is not the wrapper: a slab of
    4 of the grid's 16 rows is handed its cells halo-padded; a force of the
    slab's unpadded shape used to be zero-padded and applied."""
    node = _lbm()
    state = node.initial_state()
    padded = {k: jnp.pad(v[:4], [(1, 1), (1, 1)] + [(0, 0)] * (v.ndim - 2), mode="wrap")
              for k, v in state.items()}
    info = {0: (jnp.int32(4), 4)}
    with pytest.raises(ValueError, match=r"body_force has shape \(4, 8, 2\)"):
        node.update_padded(padded, {"body_force": jnp.full((4, 8, 2), 1e-4, jnp.float32)},
                           1.0, shard_info=info)


def test_update_padded_on_the_whole_grid_fills_a_per_cell_force_periodically():
    """A direct call on the whole grid, with its halos filled periodically as
    the node declares: the force's halos must be the opposite edge's too.
    Zero-padded (before 0.4.0), the edge cells' neighbours collided with no
    force and the step differed from ``update`` at every edge."""
    node = _lbm()
    rng = np.random.default_rng(7)
    force = jnp.asarray(rng.standard_normal(GRID + (2,)) * 1e-3, jnp.float32)
    state = node.initial_state()
    want = jax.jit(lambda s, f: node.update(s, {"body_force": f}, 1.0))(state, force)
    padded = {k: jnp.pad(v, [(1, 1), (1, 1)] + [(0, 0)] * (v.ndim - 2), mode="wrap")
              for k, v in state.items()}
    got = jax.jit(lambda s, f: node.update_padded(s, {"body_force": f}, 1.0))(padded, force)
    np.testing.assert_allclose(np.asarray(got["f"][1:-1, 1:-1]), np.asarray(want["f"]),
                               rtol=1e-6, atol=1e-9)
