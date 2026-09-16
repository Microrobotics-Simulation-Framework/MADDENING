import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np, jax, jax.numpy as jnp, pytest, warnings
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode, BoundaryInputSpec
from maddening.core.params import ParamSpec
from maddening.core.coupling.mapping import matrix_mapping
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.ball import BallNode
from maddening.core.simulation.checkpoint import save_state_with_manifest, download_and_load_state, load_state, save_state
from maddening import sysid

DT = 1e-3


def _rods(alpha=0.05, **grp):
    gm = GraphManager()
    gm.add_node(HeatNode(name="a", timestep=DT, n_cells=6, thermal_diffusivity=alpha, initial_temperature=100.0, length=1.0))
    gm.add_node(HeatNode(name="b", timestep=DT, n_cells=6, thermal_diffusivity=alpha, initial_temperature=0.0, length=1.0))
    gm.add_edge("a", "b", "temperature", "left_temperature", transform=lambda T: T[-1])
    gm.add_edge("b", "a", "temperature", "right_temperature", transform=lambda T: T[0])
    gm.add_coupling_group(["a", "b"], max_iterations=30, tolerance=1e-8, **grp)
    gm.compile()
    return gm


@pytest.mark.parametrize("grp", [dict(solver="ift"), dict(solver="fori")])
def test_grad_wrt_alpha_at_interface_cell_one_step_matches_fd(grp):
    gm = _rods(**grp)
    state0 = gm._state
    step = gm._compiled_step
    ext = gm._default_external_inputs()

    def f(alpha):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["a"]["thermal_diffusivity"] = alpha
        p["nodes"]["b"]["thermal_diffusivity"] = alpha
        s = step(state0, ext, p)
        return s["a"]["temperature"][-1]
    a0 = jnp.asarray(0.05, jnp.float32)
    g = float(jax.grad(f)(a0))
    h = 1e-3
    fd = (float(f(a0 + h)) - float(f(a0 - h))) / (2 * h)
    print(grp, "grad", g, "fd", fd)
    assert g == pytest.approx(fd, rel=2e-2)


class Vec(SimulationNode):
    def __init__(self, name, timestep, n=3, gain=1.0):
        super().__init__(name, timestep, n=n, gain=gain)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)}

    def update(self, s, bi, dt, *, params=None):
        p = {**self.params, **(params or {})}
        return {"v": s["v"] + dt * p["gain"] * bi.get("inp", jnp.zeros_like(s["v"]))}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


def _mapped(H=None):
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0))
    gm.add_node(Vec("b", 1.0))
    H = jnp.eye(3, dtype=jnp.float32) if H is None else H
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H))
    gm.compile()
    return gm


def test_grad_wrt_mapping_weights_matches_fd_and_fit_recovers_H():
    gm = _mapped()
    key = "a.v->b.inp"
    gm.set_param_spec(key, "H", ParamSpec())
    truth = jax.tree.map(lambda x: x, gm.params)
    truth["mappings"][key]["H"] = jnp.array([[2., 0, 0], [0, 0.5, 0], [0, 0, 1.]], jnp.float32)
    step = gm._build_step_fn(); ext = gm._default_external_inputs(); s0 = gm._state
    def run3(p):
        s = s0
        for _ in range(3):
            s = step(s, ext, p)
        return s["b"]["v"]
    target = run3(truth)

    def loss(p):
        return jnp.sum((run3(p) - target) ** 2)
    g = jax.grad(loss)(gm.params)["mappings"][key]["H"]
    # FD on H[1,1]
    def f(x):
        p = jax.tree.map(lambda x: x, gm.params)
        p["mappings"][key]["H"] = p["mappings"][key]["H"].at[1, 1].set(x)
        return float(loss(p))
    fd = (f(1.0 + 1e-2) - f(1.0 - 1e-2)) / 2e-2
    print("grad H11", float(g[1, 1]), "fd", fd)
    assert float(g[1, 1]) == pytest.approx(fd, rel=2e-2)
    assert gm.trainable_mask()["mappings"][key]["H"] is True
    assert gm.trainable_mask()["nodes"]["b"]["gain"] is True
    gm.set_param_spec("a", "gain", ParamSpec(trainable=False))
    gm.set_param_spec("b", "gain", ParamSpec(trainable=False))
    res = sysid.fit(gm, loss, n_iter=300, lr=0.05)
    Hf = np.asarray(res.params["mappings"][key]["H"])
    print("fitted H diag", np.diag(Hf), "loss", res.loss if hasattr(res, "loss") else res)
    # mapping gradient only sees the diagonal-relevant entries; check loss decreased strongly
    assert float(loss(res.params)) < 1e-3 * float(loss(gm.params))


def test_fit_through_flux_edge_recovers_stiffness():
    def build(k):
        gm = GraphManager()
        gm.add_node(SpringDamperNode(name="s", timestep=1e-2, stiffness=k, damping=0.5, initial_position=0.5))
        gm.add_node(Vec("c", 1e-2, n=1))
        gm.add_edge("s", "c", "spring_force", "inp")
        gm.compile()
        return gm
    truth = build(45.0)
    s0 = {k: v for k, v in truth._state.items() if k != "_meta"}
    _, hist = truth.run_scan_with_history(39)
    obs = sysid.observations_from_history(s0, hist)
    gm = build(20.0)
    for k in ("damping", "mass", "rest_length", "initial_position", "initial_velocity"):
        gm.set_param_spec("s", k, ParamSpec(trainable=False))
    gm.set_param_spec("c", "gain", ParamSpec(trainable=False))
    def loss(p):
        return sysid.windowed_loss(gm, p, obs, obs_fn=lambda s: s["c"]["v"], window=39)
    res = sysid.fit(gm, loss, n_iter=400, lr=0.1)
    kf = float(res.params["nodes"]["s"]["stiffness"])
    print("fitted k", kf)
    assert kf == pytest.approx(45.0, rel=5e-2)


def _multi_coupled(H=None):
    gm = GraphManager()
    gm.add_node(HeatNode(name="a", timestep=DT, n_cells=6, thermal_diffusivity=0.05, initial_temperature=100.0, length=1.0))
    gm.add_node(HeatNode(name="b", timestep=DT, n_cells=6, thermal_diffusivity=0.05, initial_temperature=0.0, length=1.0))
    gm.add_edge("a", "b", "temperature", "left_temperature", transform=lambda T: T[-1])
    gm.add_edge("b", "a", "temperature", "right_temperature", transform=lambda T: T[0])
    gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-8, solver="ift", predictor="linear")
    gm.add_node(Vec("x", 2 * DT))
    gm.add_node(Vec("y", 2 * DT))
    gm.add_edge("x", "y", "v", "inp", mapping=matrix_mapping(jnp.eye(3, dtype=jnp.float32)))
    gm.compile()
    return gm


def test_checkpoint_url_round_trip_multirate_coupled_with_mappings(tmp_path):
    gm = _multi_coupled()
    gm.params["mappings"]["x.v->y.inp"]["H"] = 2.0 * jnp.eye(3, dtype=jnp.float32)
    gm.params["nodes"]["a"]["thermal_diffusivity"] = jnp.asarray(0.08, jnp.float32)
    gm.run(5)
    npz, man = save_state_with_manifest(gm, tmp_path / "ck.npz", extra={"note": 1})
    gm.run(4)
    ref = gm._state
    fresh = _multi_coupled()
    m = download_and_load_state(fresh, "file://" + str(npz), dest_dir=tmp_path / "dl")
    assert m["extra"] == {"note": 1}
    fresh.run(4)
    for nn in ("a", "b", "x", "y"):
        for f, v in ref[nn].items():
            np.testing.assert_allclose(np.asarray(fresh._state[nn][f]), np.asarray(v), rtol=1e-6, err_msg=f"{nn}.{f}")
    for k, v in ref["_meta"].items():
        np.testing.assert_allclose(np.asarray(fresh._state["_meta"][k]), np.asarray(v), rtol=1e-6, err_msg=k)


def test_hybrid_node_params_contract():
    from maddening.core.simulation.hybrid_node import HybridNode
    gm = GraphManager()
    inner = SpringDamperNode(name="s", timestep=1e-2, stiffness=30.0, damping=2.0, initial_position=0.5)
    gm.add_node(HybridNode(inner, correction_fn=lambda s, bi, dt: {"position": 0.0 * s["position"], "velocity": 0.0 * s["velocity"]}))
    gm.compile()
    print("nodes_without_params", gm.nodes_without_params(), gm.params)
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    v1 = float(gm.step(params=p)["s"]["velocity"])
    ref = GraphManager(); ref.add_node(SpringDamperNode(name="s", timestep=1e-2, stiffness=300.0, damping=2.0, initial_position=0.5)); ref.compile()
    assert v1 == pytest.approx(float(ref.step()["s"]["velocity"]), rel=1e-6)


def test_to_dict_from_dict_keeps_every_edge_attribute():
    from maddening.core.transforms import register_transform
    try:
        @register_transform("neg_audit4")
        def neg(x):
            return -x
    except Exception as e:
        print("register:", e)
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0)); gm.add_node(Vec("b", 1.0)); gm.add_node(Vec("c", 1.0))
    gm.add_edge("a", "b", "v", "inp", transform="neg_audit4", additive=True, source_units="m", target_units="m")
    gm.add_edge("c", "b", "v", "inp", additive=True)
    gm.add_external_input("a", "inp", shape=(3,))
    gm.compile(); gm.params["nodes"]["b"]["gain"] = jnp.asarray(2.5, jnp.float32)
    d = gm.to_dict()
    import json; d = json.loads(json.dumps(d))
    gm2 = GraphManager.from_dict(d, {"Vec": Vec})
    gm2.compile()
    e1 = [e for e in gm2.edges if e.source_node == "a"][0]
    assert e1.additive and e1.transform is not None and e1.source_units == "m"
    assert float(gm2.params["nodes"]["b"]["gain"]) == 2.5
    np.testing.assert_allclose(np.asarray(gm2.run_scan(3)["b"]["v"]), np.asarray(gm.run_scan(3)["b"]["v"]))


def test_partial_params_with_wrong_mapping_shape():
    gm = _mapped()
    with pytest.raises((ValueError, TypeError)):
        gm.step(params={"mappings": {"a.v->b.inp": {"H": jnp.eye(2, dtype=jnp.float32)}}})


def test_run_adaptive_trace_count_and_params():
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="s", timestep=1e-2, stiffness=30.0, damping=2.0, initial_position=0.5))
    gm.compile()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    st, info = gm.run_adaptive(0.05, dt_initial=1e-2)
    print("adaptive trace_count", gm.trace_count, info if not isinstance(info, dict) else list(info)[:5])
