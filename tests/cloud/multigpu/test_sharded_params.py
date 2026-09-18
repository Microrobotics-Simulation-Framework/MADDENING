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
    assert sh.trace_count == 1


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


# ---------------------------------------------------------------------------
# The wrapper is the node: whatever surface a user reaches for, it must see
# the inner node's parameter contract.  ShardedPointwiseNode used to report
# ``accepts_params() == False`` while still returning the base class's six
# pytree leaves, so ``PUT /graph/params`` answered 200 with a value the
# physics never read (whole-tree audit W2).
# ---------------------------------------------------------------------------


def _pointwise_params_pair():
    from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
    from maddening.nodes.spring import SpringDamperNode

    inner = SpringDamperNode("s", 0.01, stiffness=10.0, initial_position=2.0)
    mesh = create_device_mesh(shape=(4,))
    return inner, ShardedPointwiseNode(inner, mesh, shard_axes=(0,))


def _stencil_params_pair():
    inner = _lbm()
    mesh = create_device_mesh(shape=(4,))
    return inner, ShardedStencilNode(
        inner, mesh, axis_map={"devices": 1}, boundary="periodic",
    )


def _unstructured_params_pair():
    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
    from tests.cloud.multigpu.test_sharded_unstructured import _NeighbourAverageNode

    class Gained(_NeighbourAverageNode):
        def update_padded(self, state_padded, boundary_inputs, dt, *,
                          static_padded=None, shard_info=None, params=None):
            out = super().update_padded(state_padded, boundary_inputs, dt,
                                        static_padded=static_padded,
                                        shard_info=shard_info)
            p = self.params if params is None else {**self.params, **params}
            out["x"] = out["x"] * p.get("gain", 1.0)
            return out

    n_global, n_devices = 16, 4
    pa = (np.arange(n_global) % n_devices).astype(np.int32)
    edges = np.array([[i, (i + 1) % n_global] for i in range(n_global)], dtype=np.int32)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                          n_devices=n_devices)
    mesh = create_device_mesh(shape=(n_devices,))
    inner = Gained(name="toy", n_global_cells=n_global, edges=edges,
                   partition_assignment=pa)
    inner.params["gain"] = 1.0
    return inner, ShardedUnstructuredNode(inner, mesh, layout)


_WRAPPER_PAIRS = {
    "pointwise": _pointwise_params_pair,
    "stencil": _stencil_params_pair,
    "unstructured": _unstructured_params_pair,
}


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
@pytest.mark.parametrize("build", list(_WRAPPER_PAIRS.values()), ids=list(_WRAPPER_PAIRS))
def test_every_sharded_wrapper_proxies_the_param_contract_of_its_inner_node(build):
    inner, wrapped = build()
    assert wrapped.accepts_params() is True
    # params_pytree must agree with accepts_params, key for key.
    assert set(wrapped.params_pytree()) == set(inner.params_pytree())
    # ...and the specs must keep the inner node's bounds and transforms, not
    # fall back to the base class's "initial conditions only" default.
    assert wrapped.param_specs() == inner.param_specs()
    # One node, one params dict: a write through any surface must reach the
    # ``self.params`` the inner update reads.
    assert wrapped.params is inner.params


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_a_sharded_wrapper_around_a_legacy_node_reports_no_params_consistently():
    """A node on the 3-argument contract stays a no-params node when wrapped.

    ``accepts_params()``, ``params_pytree()`` and ``param_specs()`` have to
    agree: a wrapper that answers ``False`` while still listing pytree leaves
    invites every caller to write a parameter nothing reads.
    """
    from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
    from maddening.core.node import SimulationNode

    class _Legacy(SimulationNode):
        def initial_state(self):
            return {"x": jnp.zeros(4, dtype=jnp.float32)}

        def state_fields(self):
            return ["x"]

        def update(self, state, boundary_inputs, dt):
            return {"x": state["x"] + self.params["rate"] * dt}

    inner = _Legacy(name="legacy", timestep=0.1, rate=2.0)
    wrapped = ShardedPointwiseNode(inner, create_device_mesh(shape=(4,)))
    assert wrapped.accepts_params() is False
    assert wrapped.params_pytree() == {}
    assert wrapped.param_specs() == {}


def _sharded_spring_graph(stiffness):
    from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
    from maddening.nodes.spring import SpringDamperNode

    gm = GraphManager()
    gm.add_node(ShardedPointwiseNode(
        SpringDamperNode("s", 0.01, stiffness=stiffness, initial_position=2.0),
        create_device_mesh(shape=(4,)), shard_axes=(0,),
    ))
    gm.compile()
    return gm


def _position_after(gm, steps=5):
    gm.reset_state()
    for _ in range(steps):
        gm.step()
    return float(gm.get_node_state("s")["position"])


def _client(gm):
    from fastapi.testclient import TestClient
    from maddening.api.server import SimulationServer
    from maddening.nodes.spring import SpringDamperNode

    server = SimulationServer({"SpringDamperNode": SpringDamperNode}, gm)
    return TestClient(server.create_app(), raise_server_exceptions=False)


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_a_rest_param_write_to_a_sharded_node_changes_the_trajectory():
    """``PUT /graph/params`` answering 200 must mean the physics changed.

    The wrapper used to report no params at all, so the write landed on a
    private copy: the endpoint echoed ``stiffness=1000`` and the node kept
    integrating at 10.
    """
    gm = _sharded_spring_graph(10.0)
    response = _client(gm).put("/graph/params/s", json={"params": {"stiffness": 1000.0}})
    assert response.status_code == 200
    assert response.json()["params"]["stiffness"] == 1000.0

    written = _position_after(gm)
    assert written == pytest.approx(_position_after(_sharded_spring_graph(1000.0)))
    assert written != pytest.approx(_position_after(_sharded_spring_graph(10.0)))


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_a_rest_param_write_outside_a_sharded_nodes_bounds_is_refused_by_name():
    """The inner node's ``ParamSpec`` still guards the wrapped node.

    Proxying ``param_specs()`` is what makes the bound reachable at all --
    with the base-class fallback the wrapper exposed no bound on
    ``stiffness``, so a negative stiffness was a 200.
    """
    from maddening.core.params import ParamSpec

    gm = _sharded_spring_graph(10.0)
    client = _client(gm)

    refused = client.put("/graph/params/s", json={"params": {"stiffness": -5.0}})
    assert refused.status_code == 400
    assert "stiffness" in refused.json()["detail"]
    assert "below bound 0.0" in refused.json()["detail"]

    # A graph-level override is enforceable too: ``set_param_spec`` used to
    # refuse the node outright with "takes no params".
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(1.0, 100.0)))
    too_stiff = client.put("/graph/params/s", json={"params": {"stiffness": 1000.0}})
    assert too_stiff.status_code == 400
    assert "above bound 100.0" in too_stiff.json()["detail"]

    # A refused write leaves both the live pytree and the node untouched.
    assert float(gm.params["nodes"]["s"]["stiffness"]) == pytest.approx(10.0)
    assert gm._nodes["s"].node.params["stiffness"] == pytest.approx(10.0)
