"""Blind spots found by reviewing tonight's changes against their failure
modes (not by a failing run).  Each test pins one behaviour a future
change could silently break."""

import os
import subprocess
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode


# ---------------------------------------------------------------------------
# Integer state leaves inside a coupling group (dtype restore) — gradient
# ---------------------------------------------------------------------------


class Counter(SimulationNode):
    """A coupled node with a float and an int32 leaf."""
    def initial_state(self):
        return {"y": jnp.array(0.0, jnp.float32), "n": jnp.array(0, jnp.int32)}

    def update(self, s, bi, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        gain = p.get("gain", 1.0)
        x = bi.get("x", jnp.array(0.0, jnp.float32))
        return {"y": s["y"] + dt * (gain * x - s["y"]), "n": s["n"] + 1}

    def boundary_input_spec(self):
        from maddening.core.node import BoundaryInputSpec
        return {"x": BoundaryInputSpec(shape=(), description="drive")}


def _coupled_with_counter(**group_kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(Counter("c", 0.01, gain=2.0))
    gm.add_edge("s", "c", "position", "x")
    gm.add_edge("c", "s", "y", "anchor_position")
    gm.add_coupling_group(["s", "c"], max_iterations=20, tolerance=1e-8, **group_kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("group_kw", [dict(), dict(acceleration="aitken"),
                                      dict(acceleration="iqn-ils"),
                                      dict(iteration_mode="jacobi")])
def test_int_leaf_in_coupling_group_keeps_dtype_single_step_grad_and_jvp(group_kw):
    gm = _coupled_with_counter(**group_kw)
    for _ in range(3):
        gm.step()
    assert gm._state["c"]["n"].dtype == jnp.int32 and int(gm._state["c"]["n"]) == 3
    assert gm._compiled_step._cache_size() == 1
    step = gm._build_step_fn()
    ext = gm._default_external_inputs()

    def one(p):
        return step(gm._state, ext, p)["c"]["y"] ** 2

    g = jax.grad(one)(gm.params)
    assert bool(jnp.isfinite(g["nodes"]["c"]["gain"])) and float(g["nodes"]["c"]["gain"]) != 0.0
    # forward mode through a scan is fine too
    def scanned(gain):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["c"]["gain"] = gain
        final, _ = jax.lax.scan(lambda s, _: (step(s, ext, p), None), gm._state, None, length=4)
        return final["c"]["y"]
    _, t = jax.jvp(scanned, (jnp.float32(2.0),), (jnp.float32(1.0),))
    assert bool(jnp.isfinite(t)) and float(t) != 0.0


@pytest.mark.parametrize("group_kw", [dict(), dict(acceleration="iqn-ils"),
                                      dict(iteration_mode="jacobi")])
def test_int_leaf_in_coupling_group_reverse_and_forward_mode_through_scan(group_kw):
    """Was an UnexpectedTracerError (reverse) / missing constant handler
    (forward): closure_convert hoisted the int leaf as an integer constant
    of the IFT custom_jvp.  Now float images are hoisted instead."""
    gm = _coupled_with_counter(**group_kw)
    step = gm._build_step_fn()
    ext = gm._default_external_inputs()

    def loss(p):
        final, _ = jax.lax.scan(lambda s, _: (step(s, ext, p), None), gm._state, None, length=4)
        return final["c"]["y"] ** 2

    g = jax.grad(loss)(gm.params)
    assert bool(jnp.isfinite(g["nodes"]["c"]["gain"])) and float(g["nodes"]["c"]["gain"]) != 0.0
    # forward mode agrees with reverse mode on the gain direction
    v = jax.tree.map(jnp.zeros_like, gm.params)
    v["nodes"]["c"]["gain"] = jnp.float32(1.0)
    _, t = jax.jvp(loss, (gm.params,), (v,))
    assert np.isclose(float(t), float(g["nodes"]["c"]["gain"]), rtol=1e-4, atol=1e-6)
    # and the integer leaf is still an integer after the scan
    final, _ = jax.lax.scan(lambda s, _: (step(s, ext, gm.params), None), gm._state, None, length=4)
    assert final["c"]["n"].dtype == jnp.int32 and int(final["c"]["n"]) == 4


# ---------------------------------------------------------------------------
# Explicit accelerated_fields naming a flux is a clear error, not KeyError
# ---------------------------------------------------------------------------


def test_accelerated_fields_naming_a_non_state_field_is_a_clear_error():
    from maddening.nodes.heat import HeatNode
    gm = GraphManager()
    gm.add_node(HeatNode("rod", 1e-3, n_cells=6))
    gm.add_node(SpringDamperNode("s", 1e-3))
    gm.add_edge("s", "rod", "position", "right_temperature")
    gm.add_edge("rod", "s", "left_heat_flux", "anchor_position")
    gm.add_coupling_group(["rod", "s"], max_iterations=5, acceleration="iqn-ils",
                          accelerated_fields={"rod": ("left_heat_flux",)})
    with pytest.raises(ValueError, match="left_heat_flux.*not a state field"):
        gm.compile()


# ---------------------------------------------------------------------------
# sysid: empty mask, LM through a coupled group, fit on non-params graph
# ---------------------------------------------------------------------------


def test_fit_and_fim_reject_an_all_false_mask():
    from maddening.sysid import fim, fit, fit_lm
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01))
    gm.compile()
    mask = jax.tree.map(lambda _: False, gm.params)
    with pytest.raises(ValueError, match="selects no parameters"):
        fit(gm, lambda p: p["nodes"]["s"]["stiffness"] ** 2, mask=mask, n_iter=1)
    with pytest.raises(ValueError, match="selects no parameters"):
        fit_lm(gm, lambda p: p["nodes"]["s"]["stiffness"][None], mask=mask, n_iter=1)
    with pytest.raises(ValueError, match="selects no parameters"):
        fim(lambda p: p["nodes"]["s"]["stiffness"][None], gm.params, mask=mask)


def test_fit_lm_through_ift_coupled_group_recovers_stiffness():
    from maddening.sysid import fit_lm, observations_from_history
    def build(k):
        gm = GraphManager()
        gm.add_node(SpringDamperNode("a", 0.01, stiffness=k, damping=2.0, initial_position=0.0))
        gm.add_node(SpringDamperNode("b", 0.01, stiffness=k, damping=2.0, initial_position=3.0))
        gm.add_edge("a", "b", "position", "anchor_position")
        gm.add_edge("b", "a", "position", "anchor_position")
        gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-8)
        gm.compile()
        return gm
    truth = build(30.0)
    init = {n: truth.get_node_state(n) for n in truth.node_names}
    _, hist = truth.run_scan_with_history(60)
    obs = observations_from_history(init, hist)
    gm = build(30.0)
    for n in ("a", "b"):
        for key in ("mass", "rest_length", "damping"):
            gm.set_param_spec(n, key, ParamSpec(trainable=False))
    step = gm._build_step_fn(); ext = gm._default_external_inputs()
    start = jax.tree.map(lambda x: x[0], obs)
    def residual(p):
        def body(s, _):
            s = step(s, ext, p)
            return s, jnp.stack([s["a"]["position"], s["b"]["position"]])
        return jax.lax.scan(body, start, None, length=60)[1] - jnp.stack(
            [obs["a"]["position"][1:], obs["b"]["position"][1:]], axis=1)
    p0 = jax.tree.map(lambda x: x, gm.params)
    for n in ("a", "b"):
        p0["nodes"][n]["stiffness"] = jnp.asarray(45.0, jnp.float32)
    res = fit_lm(gm, residual, params=p0, n_iter=25)
    for n in ("a", "b"):
        assert abs(float(res.params["nodes"][n]["stiffness"]) - 30.0) < 0.6


# ---------------------------------------------------------------------------
# differentiable sharded solve ignores x0 (documented) but still converges
# ---------------------------------------------------------------------------


def test_differentiable_sharded_cg_with_x0_matches_dense():
    from maddening.cloud.multigpu.iterative_solver import sharded_cg
    n = 12
    A = jnp.diag(jnp.full((n,), 2.0)) - jnp.diag(jnp.ones(n - 1), 1) - jnp.diag(jnp.ones(n - 1), -1)
    b = jnp.arange(n, dtype=jnp.float32)
    x_ref = jnp.linalg.solve(A, b)
    x0 = x_ref + 0.1
    for diff in (False, True):
        r = sharded_cg(lambda x: A @ x, b, x0=x0, max_iters=200, rtol=1e-6, backend="loop",
                       differentiable=diff)
        assert jnp.allclose(r.value, x_ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# benchmark script does not rot
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bench_coupling_script_runs_on_springs(tmp_path):
    out = tmp_path / "b.json"
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONWARNINGS": "ignore"}
    r = subprocess.run(
        [sys.executable, "benchmarks/bench_coupling.py", "--graph", "springs", "--steps", "10",
         "--warmup", "2", "--json", str(out)],
        capture_output=True, text=True, env=env, cwd=os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), check=True,
    )
    assert "PERF-1 acceptance" in r.stdout
    import json
    d = json.loads(out.read_text())
    assert d["graph"] == "springs" and d["coupled_mean_step_ms"] > 0
    assert {v["variant"] for v in d["variants"]} == {"coupled", "baseline"}


# ---------------------------------------------------------------------------
# Mapped edges through checkpoint / FMI / sysid.fit (mapping weights are
# graph params but not node params)
# ---------------------------------------------------------------------------


def _mapped_rods():
    from maddening.core.coupling.mapping import rbf_mapping
    from maddening.nodes.heat import HeatNode
    xc, xf = np.linspace(0, 1, 5), np.linspace(0, 1, 9)
    gm = GraphManager()
    gm.add_node(HeatNode("c", 1e-4, n_cells=5, thermal_diffusivity=0.1, initial_temperature=300.0))
    gm.add_node(HeatNode("f", 1e-4, n_cells=9, thermal_diffusivity=0.1, initial_temperature=350.0))
    gm.add_edge("c", "f", "temperature", "heat_source", mapping=rbf_mapping(xc, xf, epsilon=4.0))
    gm.add_edge("f", "c", "temperature", "heat_source", mapping=rbf_mapping(xf, xc, epsilon=4.0))
    gm.compile()
    return gm


def test_checkpoint_round_trip_with_mapped_edges(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state
    gm = _mapped_rods()
    gm.run(5)
    gm.params["nodes"]["c"]["thermal_diffusivity"] = jnp.asarray(0.2, jnp.float32)
    path = save_state(gm, tmp_path / "ck.npz")
    fresh = _mapped_rods()
    load_state(fresh, path)
    assert float(fresh.params["nodes"]["c"]["thermal_diffusivity"]) == float(np.float32(0.2))
    # mapping weights are rebuilt from the graph definition, not the checkpoint
    np.testing.assert_array_equal(np.asarray(fresh.params["mappings"]["c.temperature->f.heat_source"]["H"]),
                                  np.asarray(gm.params["mappings"]["c.temperature->f.heat_source"]["H"]))
    a, b = gm.run_scan(3), fresh.run_scan(3)
    np.testing.assert_allclose(np.asarray(a["f"]["temperature"]), np.asarray(b["f"]["temperature"]), rtol=1e-6)


def test_fmi_model_description_ignores_mapping_weights():
    from maddening.fmi.model_description import build_model_description
    gm = _mapped_rods()
    md = build_model_description(gm, model_name="m", include_evolving=True)
    names = [v.name for v in md.variables if v.causality == "parameter"]
    assert names and all(".params." in n for n in names)
    assert not any("->" in n for n in names)


def test_sysid_fit_leaves_mapping_weights_untouched():
    from maddening.sysid import fit
    gm = _mapped_rods()
    key = "c.temperature->f.heat_source"
    H0 = np.asarray(gm.params["mappings"][key]["H"]).copy()
    step = gm._build_step_fn(); ext = gm._default_external_inputs()
    target = gm.run_scan(4)["f"]["temperature"]
    for n in ("c", "f"):
        gm.set_param_spec(n, "length", ParamSpec(trainable=False))

    def loss(p):
        final, _ = jax.lax.scan(lambda s, _: (step(s, ext, p), None), gm._state, None, length=4)
        return jnp.sum((final["f"]["temperature"] - target) ** 2)

    start = jax.tree.map(lambda x: x, gm.params)
    start["nodes"]["f"]["thermal_diffusivity"] = jnp.asarray(0.3, jnp.float32)
    res = fit(gm, loss, params=start, n_iter=5, lr=0.05)
    np.testing.assert_array_equal(np.asarray(res.params["mappings"][key]["H"]), H0)
    assert res.losses[-1] <= res.losses[0]


def test_sidecar_without_params_ignores_params_in_snapshot():
    from maddening.fmi.fmu_state import serialize_fmu_state
    from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
    sc = FmuSidecar(SidecarConfig(schema_token="t", step_fn=lambda s, e: s,
                                  initial_state={"n": {"x": jnp.array(0.0)}}))
    snap = serialize_fmu_state(state={"n": {"x": jnp.array(2.0)}}, schema_token="t",
                               params={"nodes": {"n": {"k": jnp.array(1.0)}}, "mappings": {}})
    sc.set_fmu_state(snap)
    assert sc.params is None and float(sc.state["n"]["x"]) == 2.0
