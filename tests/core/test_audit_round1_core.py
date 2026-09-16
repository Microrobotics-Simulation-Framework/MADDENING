"""Regression tests for the independent audit of 2026-09-16 (round 1),
graph-manager side.  Each test failed before its fix.

* flux edges honour the params pytree (value and gradient);
* compile() keeps calibrated params; a checkpoint loaded before the
  first compile keeps its params; ``reset_params`` is the explicit way
  back to the constructor snapshot;
* integer / uint32 / PRNG-key leaves survive a coupled step bit-exactly
  (16-bit-limb float images), inside and outside the group;
* two mapped edges on the same field pair are refused (they would share
  one weights slot);
* a partial params pytree passed through ``step``/``run_scan`` is
  completed from the *live* ``gm.params``; the raw compiled step refuses
  an incomplete one instead of silently using constructor constants;
* ``set_param_spec`` overrides do not outlive their node/edge;
* edge transform / additive / units survive ``to_dict`` -> ``from_dict``;
* the predictor extrapolates floating fields only;
* ``verify_node`` flags a flux producer that takes params in ``update``
  but not in ``compute_boundary_fluxes``.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.core.transforms import register_transform
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode


class Sink(SimulationNode):
    """x += force * dt for a scalar 'force' boundary input."""

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


def _flux_gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0,
                                 initial_position=2.0))     # stretched: F != 0
    gm.add_node(Sink("sink", 0.01))
    gm.add_edge("s", "sink", "spring_force", "force")
    gm.compile()
    return gm


def _spring_force(k, c, rest, s):
    return -k * (s["position"] - 0.0 - rest) - c * s["velocity"]


# ---------------------------------------------------------------------------
# Flux edges and the params pytree
# ---------------------------------------------------------------------------

def test_spring_force_flux_uses_injected_stiffness():
    gm = _flux_gm()
    step, ext = gm._compiled_step, gm._default_external_inputs()
    p2 = jax.tree.map(lambda x: x, gm.params)
    p2["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    s2 = step(gm._state, ext, p2)
    delivered = float(s2["sink"]["x"]) / 0.01
    assert delivered == pytest.approx(float(_spring_force(300.0, 2.0, 1.0, s2["s"])), rel=1e-6)
    assert delivered != pytest.approx(float(_spring_force(30.0, 2.0, 1.0, s2["s"])), rel=1e-6)


def test_gradient_through_flux_edge_matches_finite_differences():
    gm = _flux_gm()
    step, ext = gm._compiled_step, gm._default_external_inputs()

    def sink_x(k):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["s"]["stiffness"] = k
        return step(gm._state, ext, p)["sink"]["x"]

    k0 = jnp.asarray(30.0, jnp.float32)
    g = float(jax.grad(sink_x)(k0))
    h = 1e-1
    fd = (float(sink_x(k0 + h)) - float(sink_x(k0 - h))) / (2 * h)
    assert g == pytest.approx(fd, rel=2e-3), (g, fd)
    # and it is dominated by the flux path, not the O(dt^2) state path
    assert abs(g) > 5e-3


def test_heat_flux_uses_injected_diffusivity():
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
    delivered = float(s2["sink"]["x"]) / 1e-4
    assert delivered == pytest.approx(float(-1.0 * (T[1] - T[0]) / (1.0 / 6)), rel=1e-5)


def test_flux_edge_inside_a_coupling_group_uses_injected_stiffness():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=2.0))
    gm.add_node(Sink("sink", 0.01))
    gm.add_edge("s", "sink", "spring_force", "force")
    gm.add_edge("sink", "s", "x", "anchor_position")
    gm.add_coupling_group(["s", "sink"], max_iterations=4)
    gm.compile()
    base = float(gm.run_scan(3)["sink"]["x"])
    p2 = jax.tree.map(lambda x: x, gm.params)
    p2["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    gm.reset_state()
    stiff = float(gm.run_scan(3, params=p2)["sink"]["x"])
    assert not np.isclose(base, stiff)


def test_verify_node_flags_flux_producer_without_params():
    from maddening.testing.verification import verify_node

    class Trap(SpringDamperNode):
        def compute_boundary_fluxes(self, state, boundary_inputs, dt):     # no params kw
            return super().compute_boundary_fluxes(state, boundary_inputs, dt)

    res = verify_node(Trap("t", 0.01), bounds={"position": (-1, 1), "velocity": (-1, 1)},
                      checks=["params_consistent"], max_examples=3)
    assert res["params_consistent"].status == "FAIL"
    assert "compute_boundary_fluxes" in res["params_consistent"].detail
    ok = verify_node(SpringDamperNode("s", 0.01), bounds={"position": (-1, 1), "velocity": (-1, 1)},
                     checks=["params_consistent", "params_effective"], max_examples=5,
                     derandomize=True)
    assert all(r.passed for r in ok.values()), ok


# ---------------------------------------------------------------------------
# compile() keeps calibrated params
# ---------------------------------------------------------------------------

def test_load_state_before_compile_keeps_params(tmp_path):
    from maddening.core.simulation.checkpoint import load_state, save_state

    gm = _spring_gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(77.0, jnp.float32)
    path = save_state(gm, tmp_path / "ck")

    fresh = GraphManager()
    fresh.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    load_state(fresh, path)             # build -> load -> run, no explicit compile
    fresh.run(1)
    assert float(fresh.params["nodes"]["s"]["stiffness"]) == 77.0


@pytest.mark.parametrize("change", ["external_input", "dirty_flag", "edge"])
def test_recompile_keeps_live_params(change):
    gm = _spring_gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(77.0, jnp.float32)
    if change == "external_input":
        gm.add_external_input("s", "anchor_position")
    elif change == "edge":
        gm.add_node(SpringDamperNode("t", 0.01))
        gm.add_edge("t", "s", "position", "anchor_position")
    else:
        gm._dirty = True                # what replace_node / static-data changes do
    gm.step()                           # recompiles
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 77.0
    # the value is really live: the step used it
    assert float(gm.effective_node_params("s")["stiffness"]) == 77.0


def test_remove_node_discards_its_params_without_warning():
    import warnings

    gm = _spring_gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(77.0, jnp.float32)
    gm.remove_node("s")
    gm.add_node(SpringDamperNode("u", 0.01))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()
    assert set(gm.params["nodes"]) == {"u"}


def test_recompile_drops_leaves_that_no_longer_fit_with_a_warning():
    """A leaf whose shape changed under the same node/key cannot be kept."""
    gm = _spring_gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.zeros(3, jnp.float32)    # wrong shape
    gm._dirty = True
    with pytest.warns(RuntimeWarning, match="dropped live gm.params"):
        gm.compile()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 30.0


def test_reset_params_restores_constructor_snapshot():
    gm = _spring_gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(77.0, jnp.float32)
    gm.reset_params()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 30.0
    gm.step()
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 30.0


def test_profiler_one_iteration_variant_keeps_live_params():
    from maddening.core.simulation.profiler import profile_graph

    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, initial_position=1.0))
    gm.add_node(Sink("sink", 0.01))
    gm.add_edge("s", "sink", "spring_force", "force")
    gm.add_edge("sink", "s", "x", "anchor_position")
    gm.add_coupling_group(["s", "sink"], max_iterations=4)
    gm.compile()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(77.0, jnp.float32)
    profile_graph(gm, n_steps=3, n_warmup=1, measure_coupling=True)
    assert float(gm.params["nodes"]["s"]["stiffness"]) == 77.0


# ---------------------------------------------------------------------------
# Non-float leaves in coupled graphs are bit-exact
# ---------------------------------------------------------------------------

class KeyHolder(SimulationNode):
    def initial_state(self):
        return {"y": jnp.array(0.0, jnp.float32),
                "key": jnp.array([0xDEADBEEF, 0x12345678], jnp.uint32),
                "big": jnp.array(2**24 + 1, jnp.int32),
                "neg": jnp.array(-(2**31), jnp.int32),
                "flag": jnp.array(True)}

    def update(self, s, bi, dt, *, params=None):
        x = bi.get("x", jnp.array(0.0, jnp.float32))
        return {**s, "y": s["y"] + dt * (x - s["y"])}

    def boundary_input_spec(self):
        return {"x": BoundaryInputSpec(shape=(), description="drive")}


class TypedKeyHolder(SimulationNode):
    def initial_state(self):
        return {"y": jnp.array(0.0, jnp.float32), "key": jax.random.key(0)}

    def update(self, s, bi, dt, *, params=None):
        x = bi.get("x", jnp.array(0.0, jnp.float32))
        return {"y": s["y"] + dt * (x - s["y"]), "key": s["key"]}

    def boundary_input_spec(self):
        return {"x": BoundaryInputSpec(shape=(), description="drive")}


def _coupled(holder_cls, extra=None, **kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.add_node(holder_cls("k", 0.01))
    if extra is not None:
        gm.add_node(extra)
    gm.add_edge("s", "k", "position", "x")
    gm.add_edge("k", "s", "y", "anchor_position")
    gm.add_coupling_group(["s", "k"], max_iterations=5, tolerance=1e-8, **kw)
    gm.compile()
    return gm


@pytest.mark.parametrize("kw", [dict(), dict(solver="fori"), dict(predictor="linear"),
                                dict(acceleration="iqn-ils")])
def test_wide_integer_leaves_survive_a_coupled_step(kw):
    gm = _coupled(KeyHolder, **kw)
    out = gm.run_scan(3)
    np.testing.assert_array_equal(np.asarray(out["k"]["key"]),
                                  np.array([0xDEADBEEF, 0x12345678], np.uint32))
    assert int(out["k"]["big"]) == 2**24 + 1
    assert int(out["k"]["neg"]) == -(2**31)
    assert bool(out["k"]["flag"]) is True
    assert out["k"]["key"].dtype == jnp.uint32 and out["k"]["big"].dtype == jnp.int32


def test_typed_prng_key_inside_and_outside_group():
    gm = _coupled(TypedKeyHolder)
    out = gm.run_scan(2)
    assert jax.dtypes.issubdtype(out["k"]["key"].dtype, jax.dtypes.prng_key)
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(out["k"]["key"])),
                                  np.asarray(jax.random.key_data(jax.random.key(0))))
    gm2 = _coupled(KeyHolder, extra=TypedKeyHolder("free", 0.01))
    out2 = gm2.run_scan(2)
    np.testing.assert_array_equal(np.asarray(jax.random.key_data(out2["free"]["key"])),
                                  np.asarray(jax.random.key_data(jax.random.key(0))))


def test_gradient_still_flows_with_wide_integer_leaves_in_the_group():
    gm = _coupled(KeyHolder)

    def loss(p):
        return jnp.sum(gm.run_scan(4, params=p)["k"]["y"] ** 2)

    g = jax.grad(loss)(gm.params)["nodes"]["s"]["stiffness"]
    assert np.isfinite(float(g)) and float(g) != 0.0


# ---------------------------------------------------------------------------
# Mapped edges, partial params, overrides, edge round trip
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


def test_two_mapped_edges_on_same_field_pair_get_their_own_weights():
    from maddening.core.coupling.mapping import matrix_mapping

    gm = GraphManager()
    gm.add_node(Vec("a", 1.0, n=3))
    gm.add_node(Vec("b", 1.0, n=3))
    H1 = jnp.eye(3, dtype=jnp.float32)
    H2 = 2.0 * jnp.eye(3, dtype=jnp.float32)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H1), additive=True)
    gm.add_edge("a", "b", "v", "inp", mapping=matrix_mapping(H2), additive=True)
    gm.compile()
    assert set(gm.params["mappings"]) == {"a.v->b.inp", "a.v->b.inp#1"}
    out = gm.step()["b"]["v"]
    v = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(out), np.asarray(v + (H1 @ v + H2 @ v)))
    # each slot is live on its own
    p = jax.tree.map(lambda x: x, gm.params)
    p["mappings"]["a.v->b.inp#1"]["H"] = jnp.zeros((3, 3), jnp.float32)
    gm.reset_state()
    out2 = gm.step(params=p)["b"]["v"]
    np.testing.assert_allclose(np.asarray(out2), np.asarray(v + H1 @ v))
    # ParamSpec overrides address the slot by its key
    gm.set_param_spec("a.v->b.inp#1", "H", ParamSpec(description="second"))
    assert gm.trainable_mask()["mappings"]["a.v->b.inp#1"]["H"] is True
    assert gm.trainable_mask()["mappings"]["a.v->b.inp"]["H"] is False


def test_partial_params_complete_from_live_values_through_step():
    gm = _spring_gm(rest_length=0.0)
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(300.0, jnp.float32)
    ref = gm.run_scan(1)["s"]["velocity"]
    gm.reset_state()
    a = gm.run_scan(1, params={"nodes": {}, "mappings": {}})["s"]["velocity"]
    gm.reset_state()
    b = gm.step(params={"nodes": {"s": {"damping": jnp.asarray(2.0)}}})["s"]["velocity"]
    assert float(a) == float(ref) == float(b)


def test_raw_compiled_step_refuses_an_incomplete_pytree():
    gm = _spring_gm(rest_length=0.0)
    step, ext = gm._compiled_step, gm._default_external_inputs()
    with pytest.raises(ValueError, match="missing"):
        step(gm._state, ext, {"nodes": {}, "mappings": {}})
    with pytest.raises(ValueError, match="missing key"):
        step(gm._state, ext, {"nodes": {"s": {"damping": jnp.asarray(2.0)}}, "mappings": {}})


def test_spec_override_does_not_outlive_its_node():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01))
    gm.add_node(SpringDamperNode("t", 0.01))
    gm.compile()
    gm.set_param_spec("t", "mass", ParamSpec(trainable=False))
    gm.remove_node("t")
    assert "t" not in gm.param_spec_overrides()
    d = gm.to_dict()
    gm2 = GraphManager.from_dict(d, {"SpringDamperNode": SpringDamperNode})
    gm2.compile()


@register_transform("audit_round1_negate")
def _negate(x):
    return -x


def test_edge_transform_additive_units_round_trip():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, initial_position=1.0))
    gm.add_node(SpringDamperNode("t", 0.01))
    gm.add_edge("s", "t", "position", "anchor_position", transform="audit_round1_negate",
                additive=True, source_units="m", target_units="m")
    gm.compile()
    d = gm.to_dict()
    assert d["edges"][0]["transform"] == "audit_round1_negate"
    gm2 = GraphManager.from_dict(d, {"SpringDamperNode": SpringDamperNode})
    e = gm2.edges[0]
    assert e.transform is _negate and e.additive and e.source_units == "m" and e.target_units == "m"
    gm2.compile()
    a, b = gm.run_scan(5), gm2.run_scan(5)
    np.testing.assert_array_equal(np.asarray(a["t"]["position"]), np.asarray(b["t"]["position"]))


def test_predictor_leaves_integer_fields_alone():
    class Counter(SimulationNode):
        def initial_state(self):
            return {"y": jnp.array(0.0, jnp.float32), "n": jnp.array(0, jnp.int32),
                    "flag": jnp.array(False)}

        def update(self, s, bi, dt, *, params=None):
            x = bi.get("x", jnp.array(0.0, jnp.float32))
            return {"y": s["y"] + dt * (x - s["y"]), "n": s["n"] + 1, "flag": ~s["flag"]}

        def boundary_input_spec(self):
            return {"x": BoundaryInputSpec(shape=(), description="drive")}

    gm = _coupled(Counter, predictor="quadratic")
    out = gm.run_scan(6)
    assert int(out["k"]["n"]) == 6 and out["k"]["n"].dtype == jnp.int32
    assert bool(out["k"]["flag"]) is False and out["k"]["flag"].dtype == jnp.bool_
