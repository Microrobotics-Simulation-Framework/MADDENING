"""Calibrated parameters through persistence and the coupling internals.

* ``compute_interface_correction`` honours the params pytree (a calibrated
  diffusivity corrects coupled interface cells, both solvers) and
  ``HybridNode`` is on its physics node's params contract;
* ``load_state`` before the first compile equals compile-then-load
  (multirate step counter included), checkpoints carry mapping weights,
  and a params leaf of the wrong shape is refused;
* ``remove_edge`` drops ordinal-key overrides so ``to_dict`` round-trips;
* Python-float params leaves keep the leaf dtype (no retrace, kept on
  recompile); the logit clamp stays inside a few-ulp-wide interval.

Originally written from the independent audit of 2026-09-16 (round 2; report and
reproducers under ``benchmarks/results/audit2/``).
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode



def _spring(compile=True):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    if compile:
        gm.compile()
    return gm




# ------------------------------------------------------------ interface correction

def _heat_rods(alpha, **group_kw):
    gm = GraphManager()
    gm.add_node(HeatNode(name="rod_a", timestep=0.001, n_cells=10, thermal_diffusivity=alpha,
                         length=1.0, initial_temperature=100.0))
    gm.add_node(HeatNode(name="rod_b", timestep=0.001, n_cells=10, thermal_diffusivity=alpha,
                         length=1.0, initial_temperature=0.0))
    gm.add_edge("rod_a", "rod_b", "temperature", "left_temperature", transform=lambda T: T[-1])
    gm.add_edge("rod_b", "rod_a", "temperature", "right_temperature", transform=lambda T: T[0])
    gm.add_coupling_group(["rod_a", "rod_b"], **group_kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("group_kw", [dict(solver="ift", max_iterations=10),
                                      dict(solver="fori", max_iterations=10)])
def test_interface_correction_uses_injected_diffusivity(group_kw):
    ref = _heat_rods(1.0, **group_kw).run_scan(2)
    gm = _heat_rods(0.01, **group_kw)
    gm.params["nodes"]["rod_a"]["thermal_diffusivity"] = jnp.asarray(1.0, jnp.float32)
    gm.params["nodes"]["rod_b"]["thermal_diffusivity"] = jnp.asarray(1.0, jnp.float32)
    out = gm.run_scan(2)
    for n in ("rod_a", "rod_b"):
        np.testing.assert_allclose(np.asarray(out[n]["temperature"]),
                                   np.asarray(ref[n]["temperature"]), rtol=1e-5)


def test_hybrid_node_is_on_its_physics_node_params_contract():
    from maddening.core.simulation.hybrid_node import HybridNode

    h = HybridNode(SpringDamperNode("s", 0.01, stiffness=30.0, initial_position=1.0,
                                    rest_length=0.0),
                   lambda s, bi, dt: {"position": 0.0 * s["position"]})
    assert h.accepts_params() and "stiffness" in h.params_pytree()
    gm = GraphManager()
    gm.add_node(h)
    gm.compile()
    assert gm.nodes_without_params() == []
    base = float(gm.run_scan(3)["s"]["velocity"])
    gm.reset_state()
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    assert float(gm.run_scan(3, params=p)["s"]["velocity"]) != base


# --------------------------------------------------------------- checkpoints

def _multirate_gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("fast", 0.01, initial_position=1.0))
    gm.add_node(SpringDamperNode("slow", 0.03, initial_position=0.5))
    gm.add_edge("fast", "slow", "position", "anchor_position")
    return gm


def test_load_state_before_compile_equals_compile_then_load(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    gm = _multirate_gm()
    gm.compile()
    gm.run(4)
    assert int(gm._state["_meta"]["step_count"]) == 4
    path = save_state(gm, tmp_path / "ck")
    gm.run(3)
    ref = np.asarray(gm.get_node_state("slow")["position"])

    a = _multirate_gm(); a.compile(); load_state(a, path); a.run(3)
    b = _multirate_gm(); load_state(b, path); b.run(3)
    assert int(b._state["_meta"]["step_count"]) == 7
    np.testing.assert_allclose(np.asarray(a.get_node_state("slow")["position"]), ref, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(b.get_node_state("slow")["position"]), ref, rtol=1e-6)


class Vec(SimulationNode):
    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)}

    def update(self, s, bi, dt):
        return {"v": s["v"] + dt * bi.get("inp", jnp.zeros_like(s["v"]))}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


def _mapped():
    from maddening.core.coupling.mapping import matrix_mapping

    gm = GraphManager()
    gm.add_node(Vec("a", 1.0))
    gm.add_node(Vec("b", 1.0))
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(jnp.eye(3, dtype=jnp.float32)))
    gm.compile()
    return gm


def test_checkpoint_carries_mapping_weights(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    gm = _mapped()
    gm.params["mappings"]["a.v->b.inp"]["H"] = 3.0 * jnp.eye(3, dtype=jnp.float32)
    path = save_state(gm, tmp_path / "ck")
    fresh = _mapped()
    load_state(fresh, path)
    np.testing.assert_allclose(np.asarray(fresh.params["mappings"]["a.v->b.inp"]["H"]),
                               3.0 * np.eye(3))
    np.testing.assert_allclose(np.asarray(fresh.step()["b"]["v"]), np.asarray(gm.step()["b"]["v"]))


def test_load_state_refuses_a_params_leaf_of_the_wrong_shape(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    gm = _spring()
    path = save_state(gm, tmp_path / "ck")
    data = dict(np.load(path))
    data["_params/s/stiffness"] = np.ones((3,), np.float32)
    np.savez(tmp_path / "bad.npz", **data)
    fresh = _spring()
    with pytest.raises(ValueError, match="shape"):
        load_state(fresh, tmp_path / "bad.npz")


# ---------------------------------------------------- overrides / dtypes / clamp

def test_remove_edge_drops_ordinal_overrides_and_round_trips():
    from maddening.core.coupling.mapping import matrix_mapping

    gm = GraphManager()
    gm.add_node(Vec("a", 1.0))
    gm.add_node(Vec("b", 1.0))
    H = jnp.eye(3, dtype=jnp.float32)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H), additive=True)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(2 * H), additive=True)
    gm.compile()
    gm.set_param_spec("a.v->b.inp#1", "H", ParamSpec(description="second"))
    gm.remove_edge("a", "b", "v", "inp")
    assert not any(k.startswith("a.v->b.inp") for k in gm.param_spec_overrides())
    gm.compile()
    assert gm.params["mappings"] == {}
    GraphManager.from_dict(gm.to_dict(), {"Vec": Vec})


def test_python_float_params_leaves_do_not_retrace_and_survive_recompile():
    gm = _spring()
    gm.step()
    n0 = gm.trace_count
    gm.step(params={"nodes": {"s": {"stiffness": 31.0}}})
    gm.params["nodes"]["s"]["stiffness"] = 32.0
    gm.step()
    assert gm.trace_count == n0
    gm._dirty = True
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
    leaf = gm.params["nodes"]["s"]["stiffness"]
    assert float(leaf) == 32.0 and leaf.dtype == jnp.float32 and not leaf.weak_type


def test_logit_clamp_strictly_inside_a_few_ulp_interval():
    lo, hi = 262144.0, 262144.0625
    spec = ParamSpec(bounds=(lo, hi), transform="logit")
    for u in (-50.0, -2.0, 0.0, 2.0, 50.0):
        p = spec.to_constrained(jnp.asarray(u, jnp.float32))
        spec.check(p)                                   # strictly inside
        assert bool(jnp.isfinite(spec.to_unconstrained(p)))
