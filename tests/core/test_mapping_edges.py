"""Mapped edges: ``add_edge(..., mapping=)`` with weights in
``gm.params["mappings"]`` — a traced input like node params, never a
closure constant.

What is said here of the weights -- where they live, that a changed one
takes effect without a recompile, that a gradient reaches each, that they
are frozen until made trainable and that a fit then moves them -- is held
for the built-in RBF mapping and for each kind registered the way another
library registers one (``tests/registered_mapping_kinds.py``), weight by
weight."""

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
# The kinds registered the way another library registers one, and the
# library's own sparse nearest neighbour, registered the same way.
from tests.sparse_mapping_support import REGISTERED_AND_SPARSE as REGISTERED_KINDS

N_COARSE, N_FINE = 6, 12
X_COARSE = np.linspace(0.0, 1.0, N_COARSE)
X_FINE = np.linspace(0.0, 1.0, N_FINE)

#: The weights each mapping kind puts into ``params["mappings"]``, and one
#: ``(kind, weight)`` case per weight.
WEIGHTS = {"rbf": ("H",), **{name: kind.weights for name, kind in REGISTERED_KINDS.items()}}
EVERY_KIND = sorted(WEIGHTS)
EVERY_WEIGHT = [(kind, weight) for kind in EVERY_KIND for weight in WEIGHTS[kind]]
_WEIGHT_IDS = [f"{kind}.{weight}" for kind, weight in EVERY_WEIGHT]


def _two_rods(coupled=True, use_closure=False, kernel="thin_plate_spline", kind="rbf"):
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
    elif kind == "rbf":
        gm.add_edge("coarse", "fine", "temperature", "heat_source",
                    mapping=rbf_mapping(X_COARSE, X_FINE, epsilon=2.0, kernel=kernel))
        gm.add_edge("fine", "coarse", "temperature", "heat_source",
                    mapping=rbf_mapping(X_FINE, X_COARSE, epsilon=2.0, kernel=kernel))
    else:
        build = REGISTERED_KINDS[kind].build
        gm.add_edge("coarse", "fine", "temperature", "heat_source",
                    mapping=build(X_COARSE, X_FINE))
        gm.add_edge("fine", "coarse", "temperature", "heat_source",
                    mapping=build(X_FINE, X_COARSE))
    if coupled:
        gm.add_coupling_group(["coarse", "fine"], max_iterations=30, tolerance=1e-8)
    gm.compile()
    _curved_profiles(gm)
    return gm


def _curved_profiles(gm):
    """Curved profiles so the mapping is not trivially constant."""
    gm.set_node_state("coarse", {"temperature": jnp.asarray(300 + 50 * X_COARSE ** 2, jnp.float32)})
    gm.set_node_state("fine", {"temperature": jnp.asarray(350 - 40 * X_FINE, jnp.float32)})


C2F = "coarse.temperature->fine.heat_source"
F2C = "fine.temperature->coarse.heat_source"


@pytest.mark.parametrize("kind", EVERY_KIND)
def test_weights_live_in_params_mappings(kind):
    gm = _two_rods(kind=kind)
    assert set(gm.params["mappings"]) == {C2F, F2C}
    assert tuple(gm.params["mappings"][C2F]) == WEIGHTS[kind]
    own = gm.edges[0].mapping.params_pytree()
    for weight in WEIGHTS[kind]:
        live = gm.params["mappings"][C2F][weight]
        assert live.shape == own[weight].shape and live.dtype == own[weight].dtype
        np.testing.assert_array_equal(np.asarray(live), np.asarray(own[weight]))
    if kind == "rbf":
        assert gm.params["mappings"][C2F]["H"].shape == (N_FINE, N_COARSE)
    assert gm.edges[0].mapping.n_source == N_COARSE
    assert gm.edges[0].mapping.n_target == N_FINE


@pytest.mark.parametrize("coupled", [False, True])
def test_mapped_edge_matches_closure_transform(coupled):
    a = _two_rods(coupled=coupled, use_closure=False).run_scan(20)
    b = _two_rods(coupled=coupled, use_closure=True).run_scan(20)
    for n in ("coarse", "fine"):
        np.testing.assert_allclose(np.asarray(a[n]["temperature"]),
                                   np.asarray(b[n]["temperature"]), rtol=1e-5)


@pytest.mark.parametrize("kind, weight", EVERY_WEIGHT, ids=_WEIGHT_IDS)
def test_changed_weights_take_effect_without_recompile(kind, weight):
    gm = _two_rods(kind=kind)
    compiled = gm._compiled_step

    def fine_after_ten_steps(params=None):
        # ``run_scan`` leaves the graph at the state it reached, so every
        # run starts again from the curved profiles: what differs between
        # two of them is then the weights and nothing else.
        gm.reset_state()
        _curved_profiles(gm)
        return np.asarray(gm.run_scan(10, params=params)["fine"]["temperature"])

    base = fine_after_ten_steps()
    np.testing.assert_array_equal(fine_after_ten_steps(), base)   # the runs are repeatable
    p = jax.tree.map(lambda x: x, gm.params)
    p["mappings"][C2F][weight] = jnp.zeros_like(p["mappings"][C2F][weight])   # cut the edge
    cut = fine_after_ten_steps(p)
    assert gm._compiled_step is compiled and not gm._dirty
    assert not np.allclose(base, cut)
    # ... and written into gm.params in place, the next run uses it too
    gm.params["mappings"][C2F][weight] = jnp.zeros_like(gm.params["mappings"][C2F][weight])
    written = fine_after_ten_steps()
    assert gm._compiled_step is compiled and not gm._dirty
    np.testing.assert_array_equal(written, cut)


@pytest.mark.parametrize("coupled", [False, True])
@pytest.mark.parametrize("kind", ["rbf", "inverse_distance"])
def test_gradient_wrt_mapping_weights(coupled, kind):
    """A gradient reaches every weight of the edge, staggered and through
    the IFT rule of a coupling group, whatever class holds the weights."""
    gm = _two_rods(coupled=coupled, kind=kind)
    step = gm._build_step_fn()
    ext = gm._default_external_inputs()

    def loss(p):
        def body(s, _):
            s = step(s, ext, p)
            return s, None
        final, _ = jax.lax.scan(body, gm._state, None, length=5)
        return jnp.sum(final["fine"]["temperature"] ** 2)

    grads = jax.jit(jax.grad(loss))(gm.params)["mappings"][C2F]
    assert tuple(grads) == WEIGHTS[kind]
    for weight in WEIGHTS[kind]:
        g = grads[weight]
        assert g.shape == gm.params["mappings"][C2F][weight].shape
        assert bool(jnp.all(jnp.isfinite(g))) and float(jnp.max(jnp.abs(g))) > 0.0
    if kind == "rbf":
        assert grads["H"].shape == (N_FINE, N_COARSE)


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


@pytest.mark.parametrize("kind", EVERY_KIND)
def test_unknown_mapping_key_in_params_is_an_error(kind):
    gm = _two_rods(kind=kind)
    p = jax.tree.map(lambda x: x, gm.params)
    p["mappings"]["ghost->edge"] = {"H": jnp.zeros((2, 2))}
    with pytest.raises(ValueError, match="mappings.*unknown edge"):
        gm.step(params=p)
    # ... and so is a weight the edge's mapping does not expose
    p = jax.tree.map(lambda x: x, gm.params)
    p["mappings"][C2F] = {**p["mappings"][C2F], "ghost": jnp.zeros((2, 2))}
    with pytest.raises(ValueError, match=r"unknown key\(s\) \['ghost'\]; the mapping "
                                         r"exposes"):
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


@pytest.mark.parametrize("kind", EVERY_KIND)
def test_serialisation_of_mapped_edges_round_trips_without_weights(kind):
    """``to_dict`` writes the MappingSpec (never a weight); ``from_dict``
    rebuilds the same weights and registers the ``params['mappings']``
    slot.  Detailed coverage: tests/core/test_mapping_spec_serialisation.py."""
    gm = _two_rods(kind=kind)
    d = gm.to_dict()
    e = d["edges"][0]
    assert e["mapping"]["kind"] == kind and e["mapping"]["shape"] == [N_FINE, N_COARSE]
    assert not set(WEIGHTS[kind]) & set(e["mapping"]) and "points" in e["mapping"]
    gm2 = GraphManager.from_dict(d, {"HeatNode": HeatNode})
    gm2.compile()
    assert set(gm2.params["mappings"]) == {C2F, F2C}
    assert tuple(gm2.params["mappings"][C2F]) == WEIGHTS[kind]
    for weight in WEIGHTS[kind]:
        np.testing.assert_array_equal(np.asarray(gm2.params["mappings"][C2F][weight]),
                                      np.asarray(gm.params["mappings"][C2F][weight]))


@pytest.mark.parametrize("kind, weight", EVERY_WEIGHT, ids=_WEIGHT_IDS)
def test_mapping_weights_frozen_by_default_and_opt_in_trainable(kind, weight):
    """``sysid.fit`` must not move an interface operator unless asked."""
    from maddening.core.params import ParamSpec

    gm = _two_rods(kind=kind)
    mask = gm.trainable_mask()
    assert mask["mappings"][C2F] == {w: False for w in WEIGHTS[kind]}
    assert mask["mappings"][F2C] == {w: False for w in WEIGHTS[kind]}
    assert mask["nodes"]["fine"]["thermal_diffusivity"] is True
    u = gm.unconstrain()
    np.testing.assert_array_equal(np.asarray(u["mappings"][C2F][weight]),
                                  np.asarray(gm.params["mappings"][C2F][weight]))
    gm.set_param_spec(C2F, weight, ParamSpec(description="learned edge"))
    assert gm.trainable_mask()["mappings"][C2F] == {w: w == weight for w in WEIGHTS[kind]}
    assert gm.param_spec_overrides()[C2F][weight].description == "learned edge"
    with pytest.raises(KeyError, match="no weight 'absent'"):
        gm.set_param_spec(C2F, "absent", ParamSpec())


@pytest.mark.parametrize("kind, weight", EVERY_WEIGHT, ids=_WEIGHT_IDS)
def test_a_fit_moves_the_one_mapping_weight_made_trainable_and_no_other(kind, weight):
    """System identification over a mapping's weights: with one weight of
    one edge trainable, ``fit`` lowers the loss by moving that weight and
    hands every other leaf of the tree back bit for bit -- the other
    weights of the same mapping included."""
    from maddening.core.params import ParamSpec
    from maddening.sysid import fit

    gm = _two_rods(coupled=False, kind=kind)
    gm.set_param_spec(C2F, weight, ParamSpec())
    for node in gm.node_names:                       # only the weight is free
        for key in gm.params["nodes"][node]:
            gm.set_param_spec(node, key, ParamSpec(trainable=False))
    step, ext, state0 = gm._build_step_fn(), gm._default_external_inputs(), gm._state

    def loss(p):
        state = step(step(state0, ext, p), ext, p)
        return jnp.mean((state["fine"]["temperature"] - 350.0) ** 2)

    start = jax.tree.map(np.asarray, gm.params)
    result = fit(gm, loss, n_iter=5, lr=1e-3)
    assert float(result.losses[-1]) < float(result.losses[0])
    fitted = jax.tree.map(np.asarray, result.params)
    for path, before in jax.tree_util.tree_leaves_with_path(start):
        after = fitted
        for step_key in path:
            after = after[step_key.key]
        moved = path[0].key == "mappings" and path[1].key == C2F and path[2].key == weight
        assert (after.tobytes() != before.tobytes()) is moved, jax.tree_util.keystr(path)
        assert after.dtype == before.dtype and after.shape == before.shape
    assert np.all(np.isfinite(fitted["mappings"][C2F][weight]))


@pytest.mark.parametrize("kind", EVERY_KIND)
def test_a_fit_of_node_constants_leaves_frozen_mapping_weights_alone(kind):
    """The default: every mapping weight frozen.  A fit of the rods' own
    constants runs beside the mapped edges -- including one whose mapping
    has no weights -- and returns the weights untouched."""
    from maddening.sysid import fit

    gm = _two_rods(coupled=False, kind=kind)
    start = jax.tree.map(np.asarray, gm.params["mappings"])
    result = fit(gm, lambda p: jnp.sum((p["nodes"]["fine"]["thermal_diffusivity"] - 0.2) ** 2),
                 n_iter=3, lr=0.05)
    assert result.n_iter == 3
    assert set(result.params["mappings"]) == {C2F, F2C}
    for edge, weights in start.items():
        assert tuple(sorted(result.params["mappings"][edge])) == tuple(sorted(weights))
        for name, before in weights.items():
            assert np.asarray(result.params["mappings"][edge][name]).tobytes() == \
                before.tobytes()


# ---------------------------------------------------------------------------
# Property: a mapped edge equals the closure transform for random interfaces
# ---------------------------------------------------------------------------

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tests.conftest import EXAMPLES_COSTLY  # noqa: E402


# Per push: tests/core/test_mapping_edges.py::test_mapped_edge_matches_closure_transform (one fixed interface).
@pytest.mark.slow  # two graphs compiled per example: 8-16 s on CI
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
