"""Audit round 2, batch 1: interface correction vs params, checkpoint meta/mappings,
odd dtypes in an IFT group."""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


# ---------------------------------------------------------------------------
# 1. compute_interface_correction reads self.params (heat-heat coupled group)
# ---------------------------------------------------------------------------

def _heat_rods(alpha, **group_kw):
    gm = GraphManager()
    a = HeatNode(name="rod_a", timestep=0.001, n_cells=10,
                 thermal_diffusivity=alpha, length=1.0, initial_temperature=100.0)
    b = HeatNode(name="rod_b", timestep=0.001, n_cells=10,
                 thermal_diffusivity=alpha, length=1.0, initial_temperature=0.0)
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature",
                transform=lambda T: T[-1])
    gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature",
                transform=lambda T: T[0])
    gm.add_coupling_group(["rod_a", "rod_b"], **group_kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("group_kw", [
    dict(solver="ift", max_iterations=10),
    dict(solver="fori", max_iterations=10),
])
def test_interface_correction_uses_injected_diffusivity(group_kw):
    truth = _heat_rods(1.0, **group_kw)
    ref = truth.run_scan(1)

    gm = _heat_rods(0.01, **group_kw)
    gm.params["nodes"]["rod_a"]["thermal_diffusivity"] = jnp.asarray(1.0, jnp.float32)
    gm.params["nodes"]["rod_b"]["thermal_diffusivity"] = jnp.asarray(1.0, jnp.float32)
    out = gm.run_scan(1)

    # Interior cells: update() honours params
    np.testing.assert_allclose(np.asarray(out["rod_a"]["temperature"][1:-1]),
                               np.asarray(ref["rod_a"]["temperature"][1:-1]), rtol=1e-5)
    # Interface cells: compute_interface_correction reads self.params
    np.testing.assert_allclose(np.asarray(out["rod_a"]["temperature"][-1]),
                               np.asarray(ref["rod_a"]["temperature"][-1]), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(out["rod_b"]["temperature"][0]),
                               np.asarray(ref["rod_b"]["temperature"][0]), rtol=1e-5)


def test_gradient_wrt_diffusivity_at_interface_cell_matches_fd():
    gm = _heat_rods(0.5, solver="ift", max_iterations=10)

    def iface_cell(alpha):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["rod_b"]["thermal_diffusivity"] = alpha
        p["nodes"]["rod_a"]["thermal_diffusivity"] = alpha
        step = gm._build_step_fn()
        s = gm._state
        for _ in range(3):
            s = step(s, gm._default_external_inputs(), p)
        return s["rod_b"]["temperature"][0]

    a0 = jnp.asarray(0.5, jnp.float32)
    g = jax.grad(iface_cell)(a0)
    h = 1e-2
    fd = (iface_cell(a0 + h) - iface_cell(a0 - h)) / (2 * h)
    np.testing.assert_allclose(float(g), float(fd), rtol=5e-2)


# ---------------------------------------------------------------------------
# 2. load_state before first compile: compile() clobbers the restored _meta
# ---------------------------------------------------------------------------

def _multirate_gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("fast", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=1.0))
    gm.add_node(SpringDamperNode("slow", 0.03, stiffness=10.0, damping=1.0,
                                 initial_position=0.5))
    gm.add_edge("fast", "slow", "position", "anchor_position")
    return gm


def test_load_state_before_compile_keeps_multirate_step_count(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    gm = _multirate_gm()
    gm.compile()
    gm.run(4)                                   # step_count == 4 (not a multiple of 3)
    saved_meta = {k: np.asarray(v) for k, v in gm._state["_meta"].items()}
    assert int(saved_meta["step_count"]) == 4
    path = save_state(gm, tmp_path / "ck")
    gm.run(3)
    ref = gm.get_node_state("slow")

    # Entry point A: compile, then load
    a = _multirate_gm(); a.compile(); load_state(a, path)
    assert int(a._state["_meta"]["step_count"]) == 4
    a.run(3)

    # Entry point B: load before the first compile
    b = _multirate_gm(); load_state(b, path)
    b.run(3)
    assert int(b._state["_meta"]["step_count"]) == 7, b._state["_meta"]
    np.testing.assert_allclose(np.asarray(b.get_node_state("slow")["position"]),
                               np.asarray(ref["position"]), rtol=1e-6)
    np.testing.assert_allclose(np.asarray(a.get_node_state("slow")["position"]),
                               np.asarray(ref["position"]), rtol=1e-6)


def test_load_state_before_compile_keeps_predictor_history(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    def build():
        gm = GraphManager()
        gm.add_node(SpringDamperNode("a", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
        gm.add_node(SpringDamperNode("b", 0.01, stiffness=20.0, damping=1.0, initial_position=0.5))
        gm.add_edge("a", "b", "position", "anchor_position")
        gm.add_edge("b", "a", "position", "anchor_position")
        gm.add_coupling_group(["a", "b"], solver="fori", max_iterations=4, predictor="linear")
        return gm

    gm = build(); gm.compile(); gm.run(3)
    path = save_state(gm, tmp_path / "ck")
    gm.run(2)
    ref = gm.get_node_state("a")["position"]

    b = build(); load_state(b, path); b.run(2)
    np.testing.assert_allclose(np.asarray(b.get_node_state("a")["position"]), np.asarray(ref), rtol=1e-6)


# ---------------------------------------------------------------------------
# 3. checkpoints do not carry params["mappings"]
# ---------------------------------------------------------------------------

class Vec(SimulationNode):
    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)}

    def update(self, s, bi, dt):
        return {"v": s["v"] + dt * bi.get("inp", jnp.zeros_like(s["v"]))}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


def test_checkpoint_keeps_calibrated_mapping_weights(tmp_path):
    from maddening.core.coupling.mapping import matrix_mapping
    from maddening.core.simulation.checkpoint import load_state, save_state

    def build():
        gm = GraphManager()
        gm.add_node(Vec("a", 1.0)); gm.add_node(Vec("b", 1.0))
        gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(jnp.eye(3, dtype=jnp.float32)))
        gm.compile()
        return gm

    gm = build()
    gm.params["mappings"]["a.v->b.inp"]["H"] = 3.0 * jnp.eye(3, dtype=jnp.float32)
    path = save_state(gm, tmp_path / "ck")
    fresh = build(); load_state(fresh, path)
    np.testing.assert_allclose(np.asarray(fresh.params["mappings"]["a.v->b.inp"]["H"]),
                               3.0 * np.eye(3))


# ---------------------------------------------------------------------------
# 4. odd dtypes inside an IFT coupling group
# ---------------------------------------------------------------------------

class Odd(SimulationNode):
    def __init__(self, name, dt, dtype, zero=False):
        super().__init__(name, dt)
        self._dtype = dtype
        self._zero = zero

    def initial_state(self):
        s = {"x": jnp.array(1.0, jnp.float32), "h": jnp.array(0.5, self._dtype)}
        if self._zero:
            s["z"] = jnp.zeros((0,), jnp.float32)
        return s

    def update(self, s, bi, dt):
        other = bi.get("other", jnp.array(0.0, jnp.float32))
        out = {"x": s["x"] + dt * (other - s["x"]), "h": (s["h"] * 1.5).astype(self._dtype)}
        if self._zero:
            out["z"] = s["z"]
        return out

    def boundary_input_spec(self):
        return {"other": BoundaryInputSpec(shape=(), description="o")}


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_half_precision_leaf_in_group_keeps_dtype_and_value(dtype, solver):
    gm = GraphManager()
    gm.add_node(Odd("a", 0.1, dtype)); gm.add_node(Odd("b", 0.1, dtype))
    gm.add_edge("a", "b", "x", "other"); gm.add_edge("b", "a", "x", "other")
    gm.add_coupling_group(["a", "b"], solver=solver, max_iterations=5)
    gm.compile()
    out = gm.step()
    assert out["a"]["h"].dtype == dtype
    np.testing.assert_allclose(float(out["a"]["h"]), 0.75, rtol=1e-2)
    out = gm.run_scan(2)
    assert out["a"]["h"].dtype == dtype


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_zero_size_leaf_in_group(solver):
    gm = GraphManager()
    gm.add_node(Odd("a", 0.1, jnp.float32, zero=True)); gm.add_node(Odd("b", 0.1, jnp.float32))
    gm.add_edge("a", "b", "x", "other"); gm.add_edge("b", "a", "x", "other")
    gm.add_coupling_group(["a", "b"], solver=solver, max_iterations=5)
    gm.compile()
    out = gm.run_scan(2)
    assert out["a"]["z"].shape == (0,)
