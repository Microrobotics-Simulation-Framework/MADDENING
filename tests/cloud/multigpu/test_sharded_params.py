"""Graph parameter contract on the sharded path (blind-spot review).

A ``ShardedStencilNode`` / ``ShardedUnstructuredNode`` wrapping an inner
node whose ``update_padded`` takes ``params`` is itself a params node:
``gm.params`` carries the inner's pytree, a changed value takes effect
without recompiling, and the sharded step matches the unsharded one for
the same params.  Without this, calibrating e.g. LBM viscosity on a
sharded graph silently did nothing.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.sharded_node import ShardedStencilNode
from maddening.core.graph_manager import GraphManager
from maddening.nodes.lbm import LBMNode

_HAS_4 = len(jax.devices()) >= 4


def _lbm(grid=(8, 8)):
    return LBMNode(name="lbm", timestep=1.0, grid_shape=grid, viscosity=0.05, lattice="D2Q9")


def _graph(sharded):
    node = _lbm()
    gm = GraphManager()
    if sharded:
        mesh = create_device_mesh(shape=(4,))
        gm.add_node(ShardedStencilNode(node, mesh, axis_map={"devices": 1}, boundary="periodic"))
    else:
        gm.add_node(node)
    gm.compile()
    rng = np.random.default_rng(0)
    f0 = gm.get_node_state("lbm")["f"]
    gm.set_node_state("lbm", {**gm.get_node_state("lbm"),
                              "f": f0 + jnp.asarray(rng.standard_normal(f0.shape) * 1e-3, f0.dtype)})
    return gm


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_sharded_lbm_exposes_viscosity_and_matches_unsharded():
    sh, un = _graph(True), _graph(False)
    assert sh._nodes["lbm"].node.accepts_params()
    assert set(sh.params["nodes"]["lbm"]) == set(un.params["nodes"]["lbm"]) == {"viscosity"}
    for nu in (0.05, 0.12):
        p_sh = jax.tree.map(lambda x: x, sh.params); p_sh["nodes"]["lbm"]["viscosity"] = jnp.float32(nu)
        p_un = jax.tree.map(lambda x: x, un.params); p_un["nodes"]["lbm"]["viscosity"] = jnp.float32(nu)
        a = sh.run_scan(3, params=p_sh)["lbm"]["f"]
        b = un.run_scan(3, params=p_un)["lbm"]["f"]
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)
    # the two viscosities give different results (the param is live)
    a1 = sh.run_scan(3, params=p_sh)["lbm"]["f"]
    p_sh["nodes"]["lbm"]["viscosity"] = jnp.float32(0.05)
    a0 = sh.run_scan(3, params=p_sh)["lbm"]["f"]
    assert not np.allclose(np.asarray(a0), np.asarray(a1))
    # step() with two different params values: one compile, not two
    sh.step(params=p_sh)
    p_sh["nodes"]["lbm"]["viscosity"] = jnp.float32(0.08)
    sh.step(params=p_sh)
    assert sh._compiled_step._cache_size() == 1


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_gradient_wrt_viscosity_through_sharded_step():
    sh = _graph(True)
    step = sh._build_step_fn(); ext = sh._default_external_inputs()

    def loss(nu):
        p = jax.tree.map(lambda x: x, sh.params); p["nodes"]["lbm"]["viscosity"] = nu
        final, _ = jax.lax.scan(lambda s, _: (step(s, ext, p), None), sh._state, None, length=3)
        return jnp.sum(final["lbm"]["f"] ** 2)

    g = jax.grad(loss)(jnp.float32(0.05))
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_unstructured_wrapper_forwards_params():
    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
    from tests.cloud.multigpu.test_sharded_unstructured import _NeighbourAverageNode

    class Gained(_NeighbourAverageNode):
        def update_padded(self, state_padded, boundary_inputs, dt, *, static_padded=None,
                          shard_info=None, params=None):
            out = super().update_padded(state_padded, boundary_inputs, dt,
                                        static_padded=static_padded, shard_info=shard_info)
            p = self.params if params is None else {**self.params, **params}
            out["x"] = out["x"] * p.get("gain", 1.0)
            return out

    n_global, n_devices = 16, 4
    pa = (np.arange(n_global) % n_devices).astype(np.int32)
    edges = np.array([[i, (i + 1) % n_global] for i in range(n_global)], dtype=np.int32)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                          n_devices=n_devices)
    mesh = create_device_mesh(shape=(n_devices,))
    node = Gained(name="toy", n_global_cells=n_global, edges=edges, partition_assignment=pa)
    node.params["gain"] = 1.0
    sharded = ShardedUnstructuredNode(node, mesh, layout)
    assert sharded.accepts_params() and set(sharded.params_pytree()) == {"gain"}
    s0 = sharded.initial_state()
    base = sharded.update(s0, {}, 1.0)
    doubled = sharded.update(s0, {}, 1.0, params={"gain": jnp.float32(2.0)})
    np.testing.assert_allclose(np.asarray(sharded.gather_global(doubled)["x"]),
                               2 * np.asarray(sharded.gather_global(base)["x"]))
