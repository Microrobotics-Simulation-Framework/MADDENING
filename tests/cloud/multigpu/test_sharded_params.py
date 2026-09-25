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


def _kwargs_ring_node(k=2.0):
    from maddening.core.node import SimulationNode

    class KwargsRing(SimulationNode):
        """``update_padded(**kwargs)`` reading ``params`` out of the kwargs:
        a spelling the one params rule counts as taking the keyword."""

        def __init__(self, n=16):
            super().__init__("ring", 0.1, k=k)
            self._n = n

        def initial_state(self):
            return {"x": jnp.arange(1, self._n + 1, dtype=jnp.float32)}

        def halo_width(self):
            return {0: 1}

        def update(self, state, boundary_inputs, dt, *, params=None):
            p = self.params if params is None else {**self.params, **params}
            return {"x": state["x"] * (1 - dt * p["k"])}

        def update_padded(self, state_padded, boundary_inputs, dt, **kwargs):
            p = {**self.params, **(kwargs.get("params") or {})}
            return {"x": state_padded["x"] * (1 - dt * p["k"])}

    return KwargsRing()


def _kwargs_ring_wrapped(kind, k=2.0):
    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode

    mesh = create_device_mesh(shape=(4,))
    if kind == "stencil":
        return ShardedStencilNode(_kwargs_ring_node(k), mesh, {"devices": 0})
    pa = (np.arange(16) * 4 // 16).astype(np.int32)
    edges = np.array([[i, (i + 1) % 16] for i in range(16)], dtype=np.int32)
    layout = build_unstructured_partition(partition_assignment=pa, edges=edges, n_devices=4)
    return ShardedUnstructuredNode(_kwargs_ring_node(k), mesh, layout)


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
@pytest.mark.parametrize("kind", ["stencil", "unstructured"])
def test_a_var_keyword_update_padded_is_calibratable_under_both_sharded_wrappers(kind):
    """The same inner node, the same answer under either wrapper.

    ``ShardedUnstructuredNode`` used to accept only an explicit ``params``
    keyword on ``update_padded``, so this node was calibratable under
    ``ShardedStencilNode`` and silently absent from ``gm.params`` under the
    unstructured wrapper, where ``step(params=...)`` then refused it with a
    false "takes no 'params' keyword".
    """
    wrapped = _kwargs_ring_wrapped(kind)
    gm = GraphManager()
    gm.add_node(wrapped)
    gm.compile()
    assert gm.nodes_without_params() == []
    assert set(gm.params["nodes"]["ring"]) == {"k"}
    gm.step(params={"nodes": {"ring": {"k": jnp.asarray(5.0, jnp.float32)}}})
    x = gm.get_node_state("ring")["x"]
    if kind == "unstructured":
        x = wrapped.gather_global({"x": x})["x"]
    # k = 5, dt = 0.1: every cell halves; the constructor's k = 2 gives 0.8x.
    np.testing.assert_allclose(np.asarray(x), 0.5 * np.arange(1, 17), rtol=1e-6)


# ---------------------------------------------------------------------------
# The wrapper is the node: whatever surface a user reaches for, it must see
# the inner node's parameter contract.  ShardedPointwiseNode used to report
# ``accepts_params() == False`` while still returning the base class's six
# pytree leaves, so ``PUT /graph/params`` answered 200 with a value the
# physics never read (whole-tree audit W2).
# ---------------------------------------------------------------------------


#: Four springs, one per device: ``ShardedPointwiseNode`` refuses a node
#: with nothing to shard, which a single spring's 0-d state is.
_SPRINGS = dict(initial_position=[2.0, 1.5, 1.0, 0.5], initial_velocity=[0.0] * 4)


def _pointwise_params_pair():
    from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
    from maddening.nodes.spring import SpringDamperNode

    inner = SpringDamperNode("s", 0.01, stiffness=10.0, **_SPRINGS)
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
        SpringDamperNode("s", 0.01, stiffness=stiffness, **_SPRINGS),
        create_device_mesh(shape=(4,)), shard_axes=(0,),
    ))
    gm.compile()
    return gm


def _position_after(gm, steps=5):
    gm.reset_state()
    for _ in range(steps):
        gm.step()
    return np.asarray(gm.get_node_state("s")["position"])


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
    np.testing.assert_allclose(written, _position_after(_sharded_spring_graph(1000.0)),
                               rtol=1e-6)
    assert not np.allclose(written, _position_after(_sharded_spring_graph(10.0)), rtol=1e-6)


def _rest_stencil_graph(diffusivity):
    """``ShardedStencilNode(HeatNode)`` built at ``diffusivity``."""
    from maddening.nodes.heat import HeatNode

    gm = GraphManager()
    gm.add_node(ShardedStencilNode(
        HeatNode("h", 1e-2, n_cells=16, length=1.3, thermal_diffusivity=diffusivity,
                 initial_temperature=[300.0 + 3 * i for i in range(16)]),
        create_device_mesh(shape=(4,)), {"devices": 0},
    ))
    gm.compile()
    return gm


def _rest_unstructured_graph(k):
    """``ShardedUnstructuredNode`` around a decaying ring built at ``k``."""
    gm = GraphManager()
    gm.add_node(_kwargs_ring_wrapped("unstructured", k))
    gm.compile()
    return gm


_REST_WRITE_CASES = {
    # name: (graph builder, node, parameter, old value, new value, state field)
    "stencil": (_rest_stencil_graph, "h", "thermal_diffusivity", 0.02, 0.026, "temperature"),
    "unstructured": (_rest_unstructured_graph, "ring", "k", 2.0, 5.0, "x"),
}


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
@pytest.mark.parametrize("case", list(_REST_WRITE_CASES.values()), ids=list(_REST_WRITE_CASES))
def test_a_rest_param_write_to_a_stencil_or_unstructured_wrapper_changes_the_trajectory(case):
    """The pointwise wrapper's siblings carried its defect at every tag.

    ``ShardedStencilNode`` (v0.2.0 to v0.3.1) and ``ShardedUnstructuredNode``
    (v0.3.0, v0.3.1) copied the inner node's params into a dict of their
    own at construction; ``PUT /graph/params`` wrote that copy and answered
    200 while the inner update kept reading the constructor value.  Driven
    through this endpoint against ``git archive`` of each tag, the write
    left both wrappers on the old trajectory.  The proxy test above pins
    the shared dict; this pins what a caller of the endpoint observes.
    """
    build, node, key, old, new, field = case

    def after(gm, steps=3):
        gm.reset_state()
        for _ in range(steps):
            gm.step()
        return np.asarray(gm.get_node_state(node)[field])

    gm = build(old)
    response = _client(gm).put(f"/graph/params/{node}", json={"params": {key: new}})
    assert response.status_code == 200
    assert response.json()["params"][key] == pytest.approx(new)

    written = after(gm)
    np.testing.assert_allclose(written, after(build(new)), rtol=1e-6)
    assert not np.allclose(written, after(build(old)), rtol=1e-6)


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


# ---------------------------------------------------------------------------
# The injected value reaches the inner update through each wrapper -- without
# the node's own params being written (a REST write writes both, so a test
# through REST cannot tell a wrapper that forwards params from one that
# does not).
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
def test_a_gm_params_write_reaches_a_pointwise_wrapped_update():
    from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
    from maddening.nodes.spring import SpringDamperNode

    def graph(k):
        gm = GraphManager()
        gm.add_node(ShardedPointwiseNode(
            SpringDamperNode("s", 0.01, stiffness=k, rest_length=0.6, **_SPRINGS),
            create_device_mesh(shape=(4,)), shard_axes=(0,)))
        gm.compile()
        return gm

    gm = graph(10.0)
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(1000.0, jnp.float32)
    assert gm._nodes["s"].node.params["stiffness"] == 10.0     # only gm.params moved
    got = gm.run_scan(5)["s"]["position"]
    np.testing.assert_allclose(np.asarray(got), np.asarray(graph(1000.0).run_scan(5)["s"]["position"]),
                               rtol=1e-6)
    assert not np.allclose(np.asarray(got), np.asarray(graph(10.0).run_scan(5)["s"]["position"]))


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
@pytest.mark.parametrize("leaf, value", [("thermal_diffusivity", 0.026), ("length", 1.69)])
def test_an_injected_heat_constant_reaches_the_sharded_stencil_update(leaf, value):
    """``update_padded`` reads both trainable constants from the injected
    params; only the diffusivity used to be exercised on the sharded path,
    so one reading ``self.params["length"]`` passed."""
    from maddening.nodes.heat import HeatNode

    mesh = create_device_mesh(shape=(4,))
    kw = dict(n_cells=16, thermal_diffusivity=0.02, length=1.3,
              initial_temperature=[300.0 + 3 * i for i in range(16)])

    def wrapped(**over):
        return ShardedStencilNode(HeatNode("h", 1e-2, **{**kw, **over}), mesh, {"devices": 0})

    w0, w1 = wrapped(), wrapped(**{leaf: value})
    s = w0.initial_state()
    injected = w0.update(s, {}, 1e-2, params={**w0.params_pytree(), leaf: jnp.asarray(value, jnp.float32)})
    built = w1.update(s, {}, 1e-2)
    base = w0.update(s, {}, 1e-2)
    np.testing.assert_allclose(np.asarray(injected["temperature"]), np.asarray(built["temperature"]),
                               rtol=1e-6)
    assert not np.allclose(np.asarray(base["temperature"]), np.asarray(built["temperature"]))


# ---------------------------------------------------------------------------
# A node on the three-argument contract reads its constants from
# ``self.params`` when it is traced.  A write followed by a recompile has to
# reach the sharded step: both wrappers kept a jitted ``shard_map`` per
# input signature, which ``compile()`` did not drop, so the recompiled
# graph called a trace holding the old constant (the REST route answered
# 200 and the physics did not move).
# ---------------------------------------------------------------------------


def _legacy_diffusion(D=0.1):
    from maddening.core.node import SimulationNode

    class LegacyDiffusion(SimulationNode):
        """``u <- u + dt*D*lap(u)``, periodic, ``D`` read from ``self.params``."""

        def __init__(self):
            super().__init__("d", 0.1, D=D)

        def halo_width(self):
            return {0: 1}

        def initial_state(self):
            u = np.sin(np.linspace(0.0, 2 * np.pi, 16, endpoint=False))
            return {"u": jnp.asarray(u, jnp.float32)}

        def _new(self, up, dt):
            return up[1:-1] + dt * self.params["D"] * (up[2:] - 2 * up[1:-1] + up[:-2])

        def update(self, state, boundary_inputs, dt):
            return {"u": self._new(jnp.pad(state["u"], 1, mode="wrap"), dt)}

        def update_padded(self, state_padded, boundary_inputs, dt, *,
                          static_padded=None, shard_info=None):
            return {"u": state_padded["u"].at[1:-1].set(self._new(state_padded["u"], dt))}

    return LegacyDiffusion()


def _legacy_decay(w=0.5):
    from maddening.core.node import SimulationNode

    class LegacyDecay(SimulationNode):
        """``x <- x * (1 - w)``, ``w`` read from ``self.params``."""

        def __init__(self):
            super().__init__("d", 0.1, w=w)

        def initial_state(self):
            return {"x": jnp.arange(1, 17, dtype=jnp.float32)}

        def update(self, state, boundary_inputs, dt):
            return {"x": state["x"] * (1.0 - self.params["w"])}

        def update_padded(self, state_padded, boundary_inputs, dt, *,
                          static_padded=None, shard_info=None):
            n = shard_info[0][1]
            return {"x": state_padded["x"][:n] * (1.0 - self.params["w"])}

    return LegacyDecay()


def _legacy_graph(kind, value):
    """``(gm, inner, field, read)`` for a legacy node under ``kind``'s wrapper."""
    from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
    from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode

    mesh = create_device_mesh(shape=(4,))
    if kind == "stencil":
        inner = _legacy_diffusion(value)
        wrapped = ShardedStencilNode(inner, mesh, {"devices": 0}, boundary="periodic")
        field = "u"

        def read(gm):
            return np.asarray(gm.get_node_state("d")["u"])
    else:
        inner = _legacy_decay(value)
        pa = (np.arange(16) % 4).astype(np.int32)
        edges = np.array([[i, (i + 1) % 16] for i in range(16)], dtype=np.int32)
        layout = build_unstructured_partition(partition_assignment=pa, edges=edges,
                                              n_devices=4)
        wrapped = ShardedUnstructuredNode(inner, mesh, layout)
        field = "x"

        def read(gm):
            return np.asarray(wrapped.gather_global(gm.get_node_state("d"))["x"])
    gm = GraphManager()
    gm.add_node(wrapped)
    gm.compile()
    return gm, inner, field, read


_LEGACY_WRITES = {"stencil": ("D", 0.1, 2.0), "unstructured": ("w", 0.5, 0.9)}


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
@pytest.mark.parametrize("surface", ["rest", "node_params_and_compile"])
@pytest.mark.parametrize("kind", list(_LEGACY_WRITES))
def test_a_legacy_nodes_param_write_reaches_the_sharded_step_after_a_recompile(kind, surface):
    """One step at the old value (which traces the sharded step), the write,
    one more step: the second must use the new value, as unsharded."""
    from fastapi.testclient import TestClient
    from maddening.api.server import SimulationServer

    key, old, new = _LEGACY_WRITES[kind]
    gm, inner, field, read = _legacy_graph(kind, old)
    client = TestClient(SimulationServer({}, gm).create_app(),
                        raise_server_exceptions=False)
    assert client.post("/sim/step", json={}).status_code == 200
    if surface == "rest":
        response = client.put("/graph/params/d", json={"params": {key: new}})
        assert response.status_code == 200, response.text
        assert client.post("/sim/step", json={}).status_code == 200
    else:
        inner.params[key] = new
        gm.compile()
        gm.step()

    build = _legacy_diffusion if kind == "stencil" else _legacy_decay
    first = build(old).update(build(old).initial_state(), {}, 0.1)
    with_new = np.asarray(build(new).update(first, {}, 0.1)[field])
    with_old = np.asarray(build(old).update(first, {}, 0.1)[field])
    np.testing.assert_allclose(read(gm), with_new, rtol=1e-6, atol=1e-7)
    assert not np.allclose(read(gm), with_old, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not _HAS_4, reason="needs 4 CPU-virtual devices")
@pytest.mark.parametrize("kind", list(_LEGACY_WRITES))
def test_a_rest_write_to_a_value_the_wrapped_node_copied_at_construction_is_refused(kind):
    """The REST route decides a structural write on a copy of the node that
    reads the new value.  A sharded wrapper closes over the node it wraps,
    so no faithful copy of it can be made; the route used to accept every
    such write on the stencil wrapper unseen, and on the unstructured one
    its copy shared the original's compiled-function cache -- answering
    with the original's trace and leaving its own behind.  The wrapped
    node, which shares the params dict, is asked instead: a value it copied
    in ``__init__`` is refused, one its step reads (above) is accepted, and
    the wrapper's cache is not touched by the asking."""
    from fastapi.testclient import TestClient
    from maddening.api.server import SimulationServer

    gm, inner, _, _ = _legacy_graph(kind, _LEGACY_WRITES[kind][1])
    inner.params["baked"] = 3.0
    inner._baked = 3.0           # copied at construction, never read again
    wrapped = gm._nodes["d"].node
    client = TestClient(SimulationServer({}, gm).create_app(),
                        raise_server_exceptions=False)
    response = client.put("/graph/params/d", json={"params": {"baked": 9.0}})
    assert response.status_code == 400, response.text
    assert "trace identically" in response.json()["detail"]
    assert inner.params["baked"] == 3.0 and not gm._dirty
    assert wrapped._sharded_cache == {}
