"""GraphManager params API: validation of explicit pytrees, ParamSpec
merging/overrides, trainable mask, constrain/unconstrain, bounds."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode
from tests.conftest import EXAMPLES_STANDARD


class Legacy(SimulationNode):
    """3-argument contract: constants baked, no params."""
    def initial_state(self):
        return {"x": jnp.array(1.0, jnp.float32)}

    def update(self, s, bi, dt):
        return {"x": s["x"] * (1.0 - dt * self.params["k"])}


def _gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5))
    gm.add_node(BallNode("b", 0.01, initial_position=2.0, elasticity=0.7))
    gm.add_node(Legacy("l", 0.01, k=0.5))
    gm.compile()
    return gm


# ---------------------------------------------------------------------------
# Loud errors for pytrees the step cannot use
# ---------------------------------------------------------------------------


def test_snapshot_excludes_nodes_without_params():
    gm = _gm()
    assert set(gm.params["nodes"]) == {"s", "b"}
    assert gm.nodes_without_params() == ["l"]


def test_entry_for_node_without_params_is_an_error():
    gm = _gm()
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["l"] = {"k": jnp.asarray(0.5)}
    with pytest.raises(ValueError, match=r"takes no 'params' keyword"):
        gm.step(params=p)
    with pytest.raises(ValueError, match=r"takes no 'params' keyword"):
        gm.run_scan(3, params=p)


def test_unknown_node_and_unknown_key_are_errors():
    gm = _gm()
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["ghost"] = {"stiffness": jnp.asarray(1.0)}
    with pytest.raises(ValueError, match="unknown node 'ghost'"):
        gm.step(params=p)
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s"]["stifness"] = jnp.asarray(1.0)       # typo
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['stifness'\]"):
        gm.step(params=p)


def test_gradient_wrt_legacy_node_cannot_be_silently_zero():
    """The failure mode the error exists for: differentiating a loss with
    respect to a legacy node's constant."""
    gm = _gm()

    def loss(p):
        return gm._compiled_step(gm._state, gm._default_external_inputs(), p)["l"]["x"]

    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["l"] = {"k": jnp.asarray(0.5)}
    with pytest.raises(ValueError, match="takes no 'params' keyword"):
        jax.grad(loss)(p)


def test_adaptive_path_validates_too():
    gm = _gm()
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["ghost"] = {"k": jnp.asarray(1.0)}
    with pytest.raises(ValueError, match="unknown node"):
        gm.check_params(p)


# ---------------------------------------------------------------------------
# ParamSpec merging and overrides
# ---------------------------------------------------------------------------


def test_param_specs_merge_node_declarations_and_initial_heuristic():
    gm = _gm()
    specs = gm.param_specs()
    assert set(specs["nodes"]) == {"s", "b"}
    s = specs["nodes"]["s"]
    assert s["stiffness"].transform == "log" and s["stiffness"].bounds == (0.0, None)
    assert s["initial_position"].trainable is False
    assert s["initial_velocity"].trainable is False
    assert specs["nodes"]["b"]["elasticity"].bounds == (0.0, 1.0)
    # Every spec key is a real leaf of the pytree.
    for n, node_specs in specs["nodes"].items():
        assert set(node_specs) <= set(gm.params["nodes"][n]), n


def test_set_param_spec_overrides_and_validates():
    gm = _gm()
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    assert gm.param_specs()["nodes"]["s"]["mass"].trainable is False
    assert gm.param_specs()["nodes"]["s"]["stiffness"].trainable is True
    assert not gm._dirty
    with pytest.raises(KeyError, match="unknown node"):
        gm.set_param_spec("ghost", "mass", ParamSpec())
    with pytest.raises(ValueError, match="takes no params"):
        gm.set_param_spec("l", "k", ParamSpec())
    with pytest.raises(KeyError, match="no parameter 'nope'"):
        gm.set_param_spec("s", "nope", ParamSpec())
    with pytest.raises(TypeError):
        gm.set_param_spec("s", "mass", "frozen")


def test_trainable_mask_mirrors_params():
    gm = _gm()
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    mask = gm.trainable_mask()
    assert jax.tree.structure(mask) == jax.tree.structure(gm.params)
    s = mask["nodes"]["s"]
    assert s["mass"] is False and s["initial_position"] is False
    assert s["stiffness"] is True and s["damping"] is True
    assert mask["nodes"]["b"]["gravity"] is True


# ---------------------------------------------------------------------------
# constrain / unconstrain / check_params through the graph
# ---------------------------------------------------------------------------


def test_round_trip_and_bounds_through_graph():
    gm = _gm()
    u = gm.unconstrain()
    assert np.isclose(float(u["nodes"]["s"]["stiffness"]), np.log(30.0), rtol=1e-6)
    assert float(u["nodes"]["s"]["damping"]) == 2.0            # identity
    assert float(u["nodes"]["s"]["initial_position"]) == 0.0   # passthrough
    back = gm.constrain(u)
    for (pa, a), (pb, b) in zip(
        jax.tree_util.tree_leaves_with_path(gm.params),
        jax.tree_util.tree_leaves_with_path(back),
    ):
        assert pa == pb
        assert np.allclose(np.asarray(a), np.asarray(b), rtol=1e-6, atol=1e-7), pa
    gm.check_params()                                          # in range
    bad = jax.tree.map(lambda x: x, gm.params)
    bad["nodes"]["b"]["elasticity"] = jnp.asarray(1.5, jnp.float32)
    with pytest.raises(ValueError, match="elasticity.*above bound 1.0"):
        gm.check_params(bad)


@given(
    u_k=st.floats(-40.0, 40.0, allow_nan=False, allow_infinity=False),
    u_e=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=EXAMPLES_STANDARD, deadline=None)
def test_constrained_params_from_any_coordinates_are_a_valid_step_input(u_k, u_e):
    """An optimiser step in unconstrained coordinates, however wild, maps
    to a params pytree the graph accepts and that stays inside bounds."""
    gm = _GM_MODULE
    u = gm.unconstrain()
    u["nodes"]["s"]["stiffness"] = jnp.asarray(u_k, jnp.float32)
    u["nodes"]["b"]["elasticity"] = jnp.asarray(u_e, jnp.float32)
    p = gm.constrain(u)
    gm.check_params(p)
    out = gm._compiled_step(gm._state, gm._default_external_inputs(), p)
    assert float(p["nodes"]["s"]["stiffness"]) > 0.0
    assert 0.0 <= float(p["nodes"]["b"]["elasticity"]) <= 1.0
    assert all(bool(jnp.all(jnp.isfinite(v))) for v in jax.tree.leaves(out["b"]))


_GM_MODULE = _gm()


def test_heat_length_is_a_live_parameter():
    """Regression: ``length`` was in the pytree but read from
    ``self.params`` in the stencil, so injecting it did nothing."""
    gm = GraphManager()
    gm.add_node(HeatNode("h", 1e-4, n_cells=8, thermal_diffusivity=1.0, length=1.0,
                         initial_temperature=300.0))
    gm.compile()
    # A curved profile so the Laplacian (and hence dx = L / n) matters.
    x = jnp.linspace(0.0, 1.0, 8, dtype=jnp.float32)
    gm.set_node_state("h", {"temperature": 300.0 + 100.0 * x ** 2})
    ext = gm._default_external_inputs()

    def temp_after(length):
        p = jax.tree.map(lambda x: x, gm.params)
        p["nodes"]["h"]["length"] = jnp.asarray(length, jnp.float32)
        return gm._compiled_step(gm._state, ext, p)["h"]["temperature"]

    a, b = temp_after(1.0), temp_after(2.0)
    assert float(jnp.max(jnp.abs(a - b))) > 1e-3
    g = jax.grad(lambda L: jnp.sum(temp_after(L)))(jnp.asarray(1.0, jnp.float32))
    assert float(g) != 0.0 and bool(jnp.isfinite(g))
