"""Audit reproducers (core params / checkpoint / flux edges / mappings / coupling)."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.heat import HeatNode


class Sink(SimulationNode):
    """Integrates a scalar 'force' boundary input: x += force*dt."""
    def initial_state(self):
        return {"x": jnp.array(0.0, jnp.float32)}

    def update(self, s, bi, dt, *, params=None):
        f = bi.get("force", jnp.array(0.0, jnp.float32))
        return {"x": s["x"] + f * dt}

    def boundary_input_spec(self):
        return {"force": BoundaryInputSpec(shape=(), description="force")}


def _spring_gm(**kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=1.0, **kw))
    gm.compile()
    return gm


# ---------------------------------------------------------------------------
# F1: load_state before compile silently drops the checkpointed params
# ---------------------------------------------------------------------------

def test_load_state_before_compile_drops_params(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state
    gm = _spring_gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(77.0, jnp.float32)
    path = save_state(gm, tmp_path / "ck")

    fresh = GraphManager()
    fresh.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0,
                                    initial_position=1.0))
    # Natural resume sequence: build graph -> load checkpoint -> run.
    load_state(fresh, path)
    fresh.run(1)
    assert float(fresh.params["nodes"]["s"]["stiffness"]) == 77.0


# ---------------------------------------------------------------------------
# F2: any recompile resets gm.params to the constructor snapshot
# ---------------------------------------------------------------------------

def test_recompile_after_calibration_discards_live_params():
    gm = _spring_gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(77.0, jnp.float32)
    # A structural change that has nothing to do with the spring's constants.
    gm.add_external_input("s", "anchor_position")
    gm.step()   # recompiles
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 77.0


def test_static_data_dirty_recompile_discards_live_params():
    gm = _spring_gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(77.0, jnp.float32)
    gm.add_edge  # noqa: B018 (no-op; keep the fixture identical to the above)
    gm._dirty = True  # what _check_static_data_dirty / replace_node do
    gm.step()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 77.0


# ---------------------------------------------------------------------------
# F3: boundary fluxes read self.params, not the injected params
# ---------------------------------------------------------------------------

def _flux_gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=2.0))   # stretched: F != 0
    gm.add_node(Sink("sink", 0.01))
    gm.add_edge("s", "sink", "spring_force", "force")
    gm.compile()
    return gm


def _spring_force(k, c, rest, s):
    return -k * (s["position"] - 0.0 - rest) - c * s["velocity"]


def test_spring_force_flux_ignores_graph_params():
    gm = _flux_gm()
    step = gm._compiled_step
    ext = gm._default_external_inputs()
    p2 = jax.tree.map(lambda x: x, gm.params)
    p2["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    s2 = step(gm._state, ext, p2)
    delivered = float(s2["sink"]["x"]) / 0.01
    with_injected_k = float(_spring_force(300.0, 2.0, 1.0, s2["s"]))
    with_constructor_k = float(_spring_force(30.0, 2.0, 1.0, s2["s"]))
    assert delivered != pytest.approx(with_constructor_k, rel=1e-6), \
        "flux was computed with the constructor stiffness"
    assert delivered == pytest.approx(with_injected_k, rel=1e-6)


def test_gradient_of_flux_consumer_wrt_stiffness_is_only_the_state_path():
    """d sink.x / d k should be dt * d spring_force / d k = -dt*(x - rest) + O(dt^2)."""
    gm = _flux_gm()
    step = gm._compiled_step
    ext = gm._default_external_inputs()

    def sink_x(p):
        return step(gm._state, ext, p)["sink"]["x"]

    g = float(jax.grad(sink_x)(gm.params)["nodes"]["s"]["stiffness"])
    s1 = step(gm._state, ext, gm.params)["s"]
    expected = -0.01 * (float(s1["position"]) - 1.0)     # leading term
    assert g == pytest.approx(expected, rel=1e-2), (g, expected)


def test_heat_flux_ignores_graph_params():
    gm = GraphManager()
    gm.add_node(HeatNode("h", 1e-4, n_cells=6, thermal_diffusivity=0.1,
                         initial_temperature=300.0))
    gm.add_node(Sink("sink", 1e-4))
    gm.add_edge("h", "sink", "left_heat_flux", "force")
    gm.compile()
    st = gm._state["h"]
    gm._state["h"] = {"temperature": st["temperature"].at[0].set(0.0)}
    step, ext = gm._compiled_step, gm._default_external_inputs()
    p2 = jax.tree.map(lambda x: x, gm.params)
    p2["nodes"]["h"]["thermal_diffusivity"] = jnp.asarray(1.0, jnp.float32)
    s2 = step(gm._state, ext, p2)
    T = s2["h"]["temperature"]
    dx = 1.0 / 6
    delivered = float(s2["sink"]["x"]) / 1e-4
    assert delivered == pytest.approx(float(-1.0 * (T[1] - T[0]) / dx), rel=1e-5), \
        "left_heat_flux used the constructor diffusivity"


# ---------------------------------------------------------------------------
# F4: two mapped edges with the same key share one params["mappings"] slot
# ---------------------------------------------------------------------------

class Vec(SimulationNode):
    def __init__(self, name, timestep, n=3):
        super().__init__(name, timestep, n=n)

    def initial_state(self):
        return {"v": jnp.arange(1, self.params["n"] + 1, dtype=jnp.float32)}

    def update(self, s, bi, dt):
        inp = bi.get("inp", jnp.zeros_like(s["v"]))
        return {"v": s["v"] + dt * inp}

    def boundary_input_spec(self):
        return {"inp": BoundaryInputSpec(shape=(self.params["n"],), description="i")}


def test_duplicate_mapped_edges_collide_in_params_mappings():
    from maddening.core.coupling.mapping import matrix_mapping
    gm = GraphManager()
    gm.add_node(Vec("a", 1.0, n=3))
    gm.add_node(Vec("b", 1.0, n=3))
    H1 = jnp.eye(3, dtype=jnp.float32)
    H2 = 2.0 * jnp.eye(3, dtype=jnp.float32)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H1), additive=True)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H2), additive=True)
    gm.compile()
    assert len(gm.params["mappings"]) == 2, "second edge overwrote the first's weights"
    out = gm.step()["b"]["v"]
    v = jnp.array([1.0, 2.0, 3.0])
    expected = v + 1.0 * (H1 @ v + H2 @ v)
    np.testing.assert_allclose(np.asarray(out), np.asarray(expected))


# ---------------------------------------------------------------------------
# F5: integer / uint32 state leaves in a coupling group go through float32
# ---------------------------------------------------------------------------

class KeyHolder(SimulationNode):
    """A coupled node carrying a PRNG-like uint32 leaf that it never changes."""
    def initial_state(self):
        return {"y": jnp.array(0.0, jnp.float32),
                "key": jnp.array([0xDEADBEEF, 0x12345678], jnp.uint32),
                "big": jnp.array(2**24 + 1, jnp.int32)}

    def update(self, s, bi, dt, *, params=None):
        x = bi.get("x", jnp.array(0.0, jnp.float32))
        return {"y": s["y"] + dt * (x - s["y"]), "key": s["key"], "big": s["big"]}

    def boundary_input_spec(self):
        return {"x": BoundaryInputSpec(shape=(), description="drive")}


def _coupled_keyholder(**kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(KeyHolder("k", 0.01))
    gm.add_edge("s", "k", "position", "x")
    gm.add_edge("k", "s", "y", "anchor_position")
    gm.add_coupling_group(["s", "k"], max_iterations=5, tolerance=1e-8, **kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("kw", [dict(), dict(solver="fori")])
def test_uint32_and_large_int_leaves_survive_a_coupled_step(kw):
    gm = _coupled_keyholder(**kw)
    out = gm.step()
    np.testing.assert_array_equal(np.asarray(out["k"]["key"]),
                                  np.array([0xDEADBEEF, 0x12345678], np.uint32))
    assert int(out["k"]["big"]) == 2**24 + 1


class TypedKeyHolder(SimulationNode):
    def initial_state(self):
        return {"y": jnp.array(0.0, jnp.float32), "key": jax.random.key(0)}

    def update(self, s, bi, dt, *, params=None):
        x = bi.get("x", jnp.array(0.0, jnp.float32))
        return {"y": s["y"] + dt * (x - s["y"]), "key": s["key"]}

    def boundary_input_spec(self):
        return {"x": BoundaryInputSpec(shape=(), description="drive")}


def test_typed_prng_key_leaf_in_coupled_group():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, initial_position=1.0))
    gm.add_node(TypedKeyHolder("k", 0.01))
    gm.add_edge("s", "k", "position", "x")
    gm.add_edge("k", "s", "y", "anchor_position")
    gm.add_coupling_group(["s", "k"], max_iterations=5)
    gm.compile()
    gm.step()


def test_typed_prng_key_leaf_outside_group_with_ift_group():
    """A node *outside* the group still goes through the float32 image."""
    class Free(TypedKeyHolder):
        pass
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, initial_position=1.0))
    gm.add_node(KeyHolder("k", 0.01))
    gm.add_node(Free("free", 0.01))
    gm.add_edge("s", "k", "position", "x")
    gm.add_edge("k", "s", "y", "anchor_position")
    gm.add_coupling_group(["s", "k"], max_iterations=5)
    gm.compile()
    gm.step()


# ---------------------------------------------------------------------------
# F6: fim(noise_std=<pytree>) when the residual is a pytree
# ---------------------------------------------------------------------------

def test_fim_noise_std_pytree_dict():
    from maddening.sysid import fim
    gm = _spring_gm()
    step, ext = gm._compiled_step, gm._default_external_inputs()

    def residual(p):
        s = step(gm._state, ext, p)
        return {"pos": s["s"]["position"], "vel": s["s"]["velocity"]}

    fim(residual, gm.params, mask=gm.trainable_mask(),
        noise_std={"pos": 0.1, "vel": 1.0})


# ---------------------------------------------------------------------------
# F7: a partial params pytree silently falls back to *constructor* constants
# ---------------------------------------------------------------------------

def test_partial_params_pytree_uses_constructor_constants_not_gm_params():
    gm = _spring_gm(rest_length=0.0)   # stretched, so k matters
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    step, ext = gm._compiled_step, gm._default_external_inputs()
    ref = step(gm._state, ext, gm.params)
    # A pytree that names no node at all passes validation ...
    out = step(gm._state, ext, {"nodes": {}, "mappings": {}})
    # ... and should either error or use gm.params; it uses k=30 (constructor).
    assert float(out["s"]["velocity"]) == float(ref["s"]["velocity"])


def test_partial_node_entry_keyerror_or_fallback():
    gm = _spring_gm(rest_length=0.0)
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    step, ext = gm._compiled_step, gm._default_external_inputs()
    ref = step(gm._state, ext, gm.params)
    out = step(gm._state, ext, {"nodes": {"s": {"damping": jnp.asarray(2.0)}}, "mappings": {}})
    assert float(out["s"]["velocity"]) == float(ref["s"]["velocity"])


# ---------------------------------------------------------------------------
# F8: set_param_spec override survives remove_node -> to_dict/from_dict fails
# ---------------------------------------------------------------------------

def test_stale_spec_override_after_remove_node_breaks_round_trip():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01))
    gm.add_node(SpringDamperNode("t", 0.01))
    gm.compile()
    gm.set_param_spec("t", "mass", ParamSpec(trainable=False))
    gm.remove_node("t")
    d = gm.to_dict()
    GraphManager.from_dict(d, {"SpringDamperNode": SpringDamperNode})


# ---------------------------------------------------------------------------
# F9: sidecar rejects the value the XML advertises as `min` (log transform)
# ---------------------------------------------------------------------------

def test_fmi_min_attribute_is_not_settable_for_log_leaves():
    from maddening.fmi.model_description import build_model_description
    from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
    gm = _spring_gm()
    md = build_model_description(gm, model_name="m", include_evolving=True)
    var = next(v for v in md.variables if v.name == "s.params.stiffness")
    assert var.min == 0.0
    sc = FmuSidecar(SidecarConfig(schema_token=md.instantiation_token,
                                  step_fn=gm._compiled_step, initial_state=gm._state,
                                  params=gm.params, param_specs=gm.param_specs()))
    sc.set_params({"s.params.stiffness": var.min})


# ---------------------------------------------------------------------------
# F10: ParamSpec.from_dict with bounds=null (JSON) crashes
# ---------------------------------------------------------------------------

def test_param_spec_from_dict_bounds_null():
    ParamSpec.from_dict({"trainable": True, "bounds": None})


# ---------------------------------------------------------------------------
# F11: check() / check_params accept NaN
# ---------------------------------------------------------------------------

def test_check_params_rejects_nan():
    gm = _spring_gm()
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stiffness"] = jnp.asarray(jnp.nan, jnp.float32)
    with pytest.raises(ValueError):
        gm.check_params(p)
