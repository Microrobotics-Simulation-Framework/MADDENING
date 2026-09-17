"""Mapped edges: ``add_edge(..., mapping=)`` with weights in
``gm.params["mappings"]`` — a traced input like node params, never a
closure constant."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.coupling.interface_mapping import rbf_interpolation
from maddening.core.coupling.mapping import (
    matrix_mapping,
    projection_1d_mapping,
    rbf_mapping,
)
from maddening.nodes.heat import HeatNode

N_COARSE, N_FINE = 6, 12
X_COARSE = np.linspace(0.0, 1.0, N_COARSE)
X_FINE = np.linspace(0.0, 1.0, N_FINE)


def _two_rods(coupled=True, use_closure=False, kernel="thin_plate_spline"):
    """Coarse and fine rods exchanging their full temperature profile as a
    heat source (a stand-in for a non-conforming interface)."""
    gm = GraphManager()
    gm.add_node(HeatNode("coarse", 1e-4, n_cells=N_COARSE, thermal_diffusivity=0.1,
                         initial_temperature=300.0))
    gm.add_node(HeatNode("fine", 1e-4, n_cells=N_FINE, thermal_diffusivity=0.1,
                         initial_temperature=350.0))
    if use_closure:
        c2f = rbf_interpolation(X_COARSE.reshape(-1, 1), X_FINE.reshape(-1, 1),
                                epsilon=2.0, kernel=kernel)
        f2c = rbf_interpolation(X_FINE.reshape(-1, 1), X_COARSE.reshape(-1, 1),
                                epsilon=2.0, kernel=kernel)
        gm.add_edge("coarse", "fine", "temperature", "heat_source", transform=c2f)
        gm.add_edge("fine", "coarse", "temperature", "heat_source", transform=f2c)
    else:
        gm.add_edge("coarse", "fine", "temperature", "heat_source",
                    mapping=rbf_mapping(X_COARSE, X_FINE, epsilon=2.0, kernel=kernel))
        gm.add_edge("fine", "coarse", "temperature", "heat_source",
                    mapping=rbf_mapping(X_FINE, X_COARSE, epsilon=2.0, kernel=kernel))
    if coupled:
        gm.add_coupling_group(["coarse", "fine"], max_iterations=30, tolerance=1e-8)
    gm.compile()
    # Curved profiles so the mapping is not trivially constant.
    gm.set_node_state("coarse", {"temperature": jnp.asarray(300 + 50 * X_COARSE ** 2, jnp.float32)})
    gm.set_node_state("fine", {"temperature": jnp.asarray(350 - 40 * X_FINE, jnp.float32)})
    return gm


C2F = "coarse.temperature->fine.heat_source"
F2C = "fine.temperature->coarse.heat_source"


def test_weights_live_in_params_mappings():
    gm = _two_rods()
    assert set(gm.params["mappings"]) == {C2F, F2C}
    H = gm.params["mappings"][C2F]["H"]
    assert H.shape == (N_FINE, N_COARSE)
    assert gm.edges[0].mapping.n_source == N_COARSE


@pytest.mark.parametrize("coupled", [False, True])
def test_mapped_edge_matches_closure_transform(coupled):
    a = _two_rods(coupled=coupled, use_closure=False).run_scan(20)
    b = _two_rods(coupled=coupled, use_closure=True).run_scan(20)
    for n in ("coarse", "fine"):
        np.testing.assert_allclose(np.asarray(a[n]["temperature"]),
                                   np.asarray(b[n]["temperature"]), rtol=1e-5)


def test_changed_weights_take_effect_without_recompile():
    gm = _two_rods()
    compiled = gm._compiled_step
    base = gm.run_scan(10)
    p = jax.tree.map(lambda x: x, gm.params)
    p["mappings"][C2F]["H"] = jnp.zeros_like(p["mappings"][C2F]["H"])   # cut the edge
    cut = gm.run_scan(10, params=p)
    assert gm._compiled_step is compiled and not gm._dirty
    assert not np.allclose(np.asarray(base["fine"]["temperature"]),
                           np.asarray(cut["fine"]["temperature"]))


@pytest.mark.parametrize("coupled", [False, True])
def test_gradient_wrt_mapping_weights(coupled):
    gm = _two_rods(coupled=coupled)
    step = gm._build_step_fn()
    ext = gm._default_external_inputs()

    def loss(p):
        def body(s, _):
            s = step(s, ext, p)
            return s, None
        final, _ = jax.lax.scan(body, gm._state, None, length=5)
        return jnp.sum(final["fine"]["temperature"] ** 2)

    g = jax.jit(jax.grad(loss))(gm.params)["mappings"][C2F]["H"]
    assert g.shape == (N_FINE, N_COARSE)
    assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0


def test_mapping_then_transform_order_and_additive():
    """``mapping`` applies first, then the scalar ``transform``; additive
    edges accumulate the mapped value."""
    gm = GraphManager()
    gm.add_node(HeatNode("a", 1e-4, n_cells=4, initial_temperature=1.0))
    gm.add_node(HeatNode("b", 1e-4, n_cells=2, initial_temperature=0.0))
    m = matrix_mapping(np.array([[1, 0, 0, 0], [0, 0, 0, 1]], np.float32))
    gm.add_edge("a", "b", "temperature", "heat_source", mapping=m,
                transform=lambda x: 2.0 * x)
    gm.add_edge("a", "b", "temperature", "heat_source", mapping=m, additive=True)
    gm.compile()
    gm.set_node_state("a", {"temperature": jnp.array([10.0, 0.0, 0.0, 30.0], jnp.float32)})
    # b's heat source should be 2*[10, 30] + [10, 30] = [30, 90]; one step of
    # dt=1e-4 adds source*dt to the interior... b has only 2 cells, both
    # boundary cells, so check the resolved boundary input directly.
    bi = gm.resolve_boundary_inputs("b")
    np.testing.assert_allclose(np.asarray(bi["heat_source"]), [30.0, 90.0])


def test_shape_mismatch_rejected_at_add_edge():
    gm = GraphManager()
    gm.add_node(HeatNode("a", 1e-4, n_cells=4))
    gm.add_node(HeatNode("b", 1e-4, n_cells=2))
    with pytest.raises(ValueError, match="n_source=3.*temperature.*4"):
        gm.add_edge("a", "b", "temperature", "heat_source",
                    mapping=matrix_mapping(np.zeros((2, 3), np.float32)))
    with pytest.raises(ValueError, match="n_target=5.*heat_source.*2"):
        gm.add_edge("a", "b", "temperature", "heat_source",
                    mapping=matrix_mapping(np.zeros((5, 4), np.float32)))


def test_unknown_mapping_key_in_params_is_an_error():
    gm = _two_rods()
    p = jax.tree.map(lambda x: x, gm.params)
    p["mappings"]["ghost->edge"] = {"H": jnp.zeros((2, 2))}
    with pytest.raises(ValueError, match="mappings.*unknown edge"):
        gm.step(params=p)


def test_conservative_projection_edge_preserves_integral():
    """A conservative 1D projection edge between rods with different
    resolutions preserves the integral of the transferred field."""
    gm = GraphManager()
    gm.add_node(HeatNode("src", 1e-4, n_cells=10))
    gm.add_node(HeatNode("dst", 1e-4, n_cells=5))
    sb, tb = np.linspace(0, 1, 11), np.linspace(0, 1, 6)
    gm.add_edge("src", "dst", "temperature", "heat_source",
                mapping=projection_1d_mapping(sb, tb))
    gm.compile()
    v = np.sin(np.linspace(0.05, 0.95, 10) * np.pi).astype(np.float32)
    gm.set_node_state("src", {"temperature": jnp.asarray(v)})
    out = np.asarray(gm.resolve_boundary_inputs("dst")["heat_source"])
    assert np.sum(v * np.diff(sb)) == pytest.approx(np.sum(out * np.diff(tb)), abs=1e-5)


def test_serialisation_of_mapped_edges_round_trips_without_weights():
    """``to_dict`` writes the MappingSpec (never ``H``); ``from_dict``
    rebuilds the same weights and registers the ``params['mappings']``
    slot.  Detailed coverage: tests/core/test_mapping_spec_serialisation.py."""
    gm = _two_rods()
    d = gm.to_dict()
    e = d["edges"][0]
    assert e["mapping"]["kind"] == "rbf" and e["mapping"]["shape"] == [N_FINE, N_COARSE]
    assert "H" not in e["mapping"] and "points" in e["mapping"]
    gm2 = GraphManager.from_dict(d, {"HeatNode": HeatNode})
    gm2.compile()
    assert set(gm2.params["mappings"]) == {C2F, F2C}
    np.testing.assert_array_equal(np.asarray(gm2.params["mappings"][C2F]["H"]),
                                  np.asarray(gm.params["mappings"][C2F]["H"]))


def test_mapping_weights_frozen_by_default_and_opt_in_trainable():
    """``sysid.fit`` must not move an interface operator unless asked."""
    from maddening.core.params import ParamSpec

    gm = _two_rods()
    mask = gm.trainable_mask()
    assert mask["mappings"][C2F]["H"] is False and mask["mappings"][F2C]["H"] is False
    assert mask["nodes"]["fine"]["thermal_diffusivity"] is True
    u = gm.unconstrain()
    np.testing.assert_array_equal(np.asarray(u["mappings"][C2F]["H"]),
                                  np.asarray(gm.params["mappings"][C2F]["H"]))
    gm.set_param_spec(C2F, "H", ParamSpec(description="learned edge"))
    assert gm.trainable_mask()["mappings"][C2F]["H"] is True
    assert gm.param_spec_overrides()[C2F]["H"].description == "learned edge"
    with pytest.raises(KeyError, match="no weight 'W'"):
        gm.set_param_spec(C2F, "W", ParamSpec())


# ---------------------------------------------------------------------------
# Property: a mapped edge equals the closure transform for random interfaces
# ---------------------------------------------------------------------------

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tests.conftest import EXAMPLES_COSTLY  # noqa: E402


@given(seed=st.integers(0, 2**31), n_c=st.integers(3, 10), n_f=st.integers(3, 14),
       kernel=st.sampled_from(["gaussian", "thin_plate_spline", "multiquadric"]),
       coupled=st.booleans())
@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
def test_mapped_edge_equals_closure_for_random_interfaces(seed, n_c, n_f, kernel, coupled):
    rng = np.random.default_rng(seed)
    xc = np.sort(rng.uniform(0, 1, n_c)); xf = np.sort(rng.uniform(0, 1, n_f))
    eps = 1.0 / max(np.diff(xc).mean(), 1e-3) if n_c > 1 else 1.0

    def build(use_closure):
        rng = np.random.default_rng(seed + 1)   # same states for both builds
        gm = GraphManager()
        gm.add_node(HeatNode("c", 1e-4, n_cells=n_c, thermal_diffusivity=0.1,
                             initial_temperature=300.0))
        gm.add_node(HeatNode("f", 1e-4, n_cells=n_f, thermal_diffusivity=0.1,
                             initial_temperature=350.0))
        if use_closure:
            gm.add_edge("c", "f", "temperature", "heat_source",
                        transform=rbf_interpolation(xc.reshape(-1, 1), xf.reshape(-1, 1),
                                                    epsilon=eps, kernel=kernel))
            gm.add_edge("f", "c", "temperature", "heat_source",
                        transform=rbf_interpolation(xf.reshape(-1, 1), xc.reshape(-1, 1),
                                                    epsilon=eps, kernel=kernel))
        else:
            gm.add_edge("c", "f", "temperature", "heat_source",
                        mapping=rbf_mapping(xc, xf, epsilon=eps, kernel=kernel))
            gm.add_edge("f", "c", "temperature", "heat_source",
                        mapping=rbf_mapping(xf, xc, epsilon=eps, kernel=kernel))
        if coupled:
            gm.add_coupling_group(["c", "f"], max_iterations=20, tolerance=1e-8)
        gm.compile()
        gm.set_node_state("c", {"temperature": jnp.asarray(300 + 40 * rng.random(n_c), jnp.float32)})
        gm.set_node_state("f", {"temperature": jnp.asarray(350 - 30 * rng.random(n_f), jnp.float32)})
        return gm

    a = build(False)
    b = build(True)
    fa, fb = a.run_scan(5), b.run_scan(5)
    for n in ("c", "f"):
        np.testing.assert_allclose(np.asarray(fa[n]["temperature"]),
                                   np.asarray(fb[n]["temperature"]), rtol=1e-5, atol=1e-4)
    assert set(a.params["mappings"]) == {C2F.replace("coarse", "c").replace("fine", "f"),
                                          F2C.replace("coarse", "c").replace("fine", "f")}
