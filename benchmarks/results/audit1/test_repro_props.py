"""Audit property tests (Hypothesis) for the params / sysid / mapping invariants."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st, assume, HealthCheck

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec, constrain, unconstrain, trainable_mask, check_bounds
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.ball import BallNode
from maddening.nodes.heat import HeatNode

SETTINGS = dict(max_examples=40, deadline=None,
                suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])

# ---------------------------------------------------------------------------
# ParamSpec leaf maps
# ---------------------------------------------------------------------------

finite = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=32)


@given(lo=st.floats(-1e3, 1e3, allow_nan=False, width=32),
       delta=st.floats(0.001953125, 1e3, allow_nan=False, width=32),
       t=st.floats(0.001953125, 0.998046875, allow_nan=False, width=32))
@settings(**SETTINGS)
def test_logit_round_trip_inside_bounds(lo, delta, t):
    hi = lo + delta
    assume(hi > lo)
    spec = ParamSpec(bounds=(float(lo), float(hi)), transform="logit")
    p = jnp.float32(lo + t * (hi - lo))
    assume(float(p) > lo and float(p) < hi)
    back = spec.to_constrained(spec.to_unconstrained(p))
    assert np.isfinite(float(back))
    assert float(back) == pytest.approx(float(p), rel=1e-4, abs=1e-5 * delta)
    spec.check(back)


@given(lo=st.floats(-1e3, 1e3, allow_nan=False, width=32),
       x=st.floats(0.0009765625, 1e6, allow_nan=False, width=32))
@settings(**SETTINGS)
def test_log_round_trip_inside_bounds(lo, x):
    spec = ParamSpec(bounds=(float(lo), None), transform="log")
    p = jnp.float32(lo + x)
    assume(float(p) > lo)
    back = spec.to_constrained(spec.to_unconstrained(p))
    assert np.isfinite(float(back))
    assert float(back) == pytest.approx(float(p), rel=1e-4, abs=1e-6)
    spec.check(back)


@given(u=st.floats(-1e4, 1e4, allow_nan=False, width=32),
       lo=st.floats(-1e3, 1e3, allow_nan=False, width=32))
@settings(**SETTINGS)
def test_log_constrain_is_strictly_inside_and_re_unconstrainable(u, lo):
    spec = ParamSpec(bounds=(float(lo), None), transform="log")
    p = spec.to_constrained(jnp.float32(u))
    spec.check(p)   # strictly > lo
    u2 = spec.to_unconstrained(p)
    assert np.isfinite(float(u2))


@given(u=st.floats(-1e4, 1e4, allow_nan=False, width=32))
@settings(**SETTINGS)
def test_logit_constrain_is_strictly_inside(u):
    spec = ParamSpec(bounds=(0.0, 1.0), transform="logit")
    p = spec.to_constrained(jnp.float32(u))
    spec.check(p)
    assert np.isfinite(float(spec.to_unconstrained(p)))


def test_identity_bounded_int_leaf_keeps_dtype():
    """A bounded identity leaf with an integer dtype changes dtype through constrain."""
    spec = ParamSpec(bounds=(0.0, 10.0))
    p = jnp.asarray(5, jnp.int32)
    out = spec.to_constrained(spec.to_unconstrained(p))
    assert out.dtype == p.dtype, out.dtype


# ---------------------------------------------------------------------------
# Graph-level: structure invariants for arbitrary graphs
# ---------------------------------------------------------------------------

class Legacy(SimulationNode):
    """3-argument contract; no params."""
    def initial_state(self):
        return {"z": jnp.array(0.0, jnp.float32)}

    def update(self, s, bi, dt):
        return {"z": s["z"] + dt}


class Empty(SimulationNode):
    """Accepts params but exposes no float leaf (only structural ints)."""
    def __init__(self, name, timestep):
        super().__init__(name, timestep, n=3, flag=True, label="x")

    def initial_state(self):
        return {"z": jnp.zeros(self.params["n"], jnp.float32)}

    def update(self, s, bi, dt, *, params=None):
        return {"z": s["z"] + dt}


def _random_graph(rng, n_spring, n_ball, with_legacy, with_empty, with_mapping):
    gm = GraphManager()
    for i in range(n_spring):
        gm.add_node(SpringDamperNode(f"s{i}", 0.01, stiffness=float(rng.uniform(1, 50)),
                                     damping=float(rng.uniform(0, 5)), initial_position=1.0))
    for i in range(n_ball):
        gm.add_node(BallNode(f"b{i}", 0.01, initial_position=float(rng.uniform(1, 5))))
    if with_legacy:
        gm.add_node(Legacy("legacy", 0.01))
    if with_empty:
        gm.add_node(Empty("empty", 0.01))
    for i in range(1, n_spring):
        gm.add_edge(f"s{i-1}", f"s{i}", "position", "anchor_position")
    if with_mapping:
        from maddening.core.coupling.mapping import rbf_mapping
        gm.add_node(HeatNode("hc", 0.01, n_cells=4, thermal_diffusivity=0.1))
        gm.add_node(HeatNode("hf", 0.01, n_cells=7, thermal_diffusivity=0.1))
        gm.add_edge("hc", "hf", "temperature", "heat_source",
                    mapping=rbf_mapping(np.linspace(0, 1, 4), np.linspace(0, 1, 7), epsilon=3.0))
    gm.compile()
    return gm


@given(seed=st.integers(0, 2**31), n_spring=st.integers(0, 3), n_ball=st.integers(0, 2),
       with_legacy=st.booleans(), with_empty=st.booleans(), with_mapping=st.booleans())
@settings(max_examples=25, deadline=None)
def test_mask_unconstrain_constrain_share_structure_and_round_trip(
        seed, n_spring, n_ball, with_legacy, with_empty, with_mapping):
    assume(n_spring + n_ball + with_legacy + with_empty + with_mapping > 0)
    rng = np.random.default_rng(seed)
    gm = _random_graph(rng, n_spring, n_ball, with_legacy, with_empty, with_mapping)
    p = gm.params
    tp = jax.tree.structure(p)
    assert jax.tree.structure(gm.trainable_mask()) == tp
    u = gm.unconstrain()
    assert jax.tree.structure(u) == tp
    back = gm.constrain(u)
    assert jax.tree.structure(back) == tp
    for a, b in zip(jax.tree.leaves(p), jax.tree.leaves(back)):
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-5, atol=1e-7)
    gm.check_params(back)
    # nodes without params are absent; nodes with params but no float leaf present+empty
    if with_legacy:
        assert "legacy" not in p["nodes"]
    if with_empty:
        assert p["nodes"]["empty"] == {}


# ---------------------------------------------------------------------------
# run_scan(n, params) == n x step(params); run_scan_with_history consistent
# ---------------------------------------------------------------------------

@given(seed=st.integers(0, 2**31), n=st.integers(1, 6))
@settings(max_examples=20, deadline=None)
def test_run_scan_equals_repeated_step_with_params(seed, n):
    rng = np.random.default_rng(seed)
    gm = _random_graph(rng, 2, 1, False, False, True)
    p = jax.tree.map(lambda x: x, gm.params)
    p["nodes"]["s0"]["stiffness"] = jnp.asarray(rng.uniform(1, 50), jnp.float32)
    p["nodes"]["hc"]["thermal_diffusivity"] = jnp.asarray(rng.uniform(0.01, 0.3), jnp.float32)
    s0 = gm._state
    a = gm.run_scan(n, params=p)
    gm._state = s0
    for _ in range(n):
        b = gm.step(params=p)
    for k in a:
        for f in a[k]:
            np.testing.assert_allclose(np.asarray(a[k][f]), np.asarray(b[k][f]), rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# reset_state then step is bit-identical to a fresh compile
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [dict(), dict(acceleration="iqn-imvj", jacobian_reuse=2),
                                dict(predictor="linear"), dict(acceleration="aitken", diagnostics=True)])
def test_reset_state_then_step_matches_fresh(kw):
    def build():
        gm = GraphManager()
        gm.add_node(SpringDamperNode("a", 0.01, initial_position=1.0))
        gm.add_node(SpringDamperNode("b", 0.01, initial_position=2.0))
        gm.add_edge("a", "b", "position", "anchor_position")
        gm.add_edge("b", "a", "position", "anchor_position")
        gm.add_coupling_group(["a", "b"], max_iterations=8, **kw)
        gm.compile()
        return gm
    gm = build()
    gm.run(7)
    gm.reset_state()
    fresh = build()
    for _ in range(3):
        x, y = gm.step(), fresh.step()
        for k in x:
            for f in x[k]:
                np.testing.assert_array_equal(np.asarray(x[k][f]), np.asarray(y[k][f]), err_msg=f"{k}.{f}")
    for k, v in gm._state["_meta"].items():
        np.testing.assert_array_equal(np.asarray(v), np.asarray(fresh._state["_meta"][k]), err_msg=k)


# ---------------------------------------------------------------------------
# fit never moves a masked-out leaf; bounds respected
# ---------------------------------------------------------------------------

@given(seed=st.integers(0, 2**31))
@settings(max_examples=8, deadline=None)
def test_fit_never_moves_masked_leaf(seed):
    from maddening.sysid import fit
    rng = np.random.default_rng(seed)
    gm = _random_graph(rng, 1, 0, False, False, False)
    gm.set_param_spec("s0", "damping", ParamSpec(trainable=False, bounds=(0.0, None)))
    gm.set_param_spec("s0", "rest_length", ParamSpec(trainable=False))
    step, ext = gm._compiled_step, gm._default_external_inputs()
    target = step(gm._state, ext, gm.params)["s0"]["position"]

    def loss(p):
        return (step(gm._state, ext, p)["s0"]["position"] - target) ** 2 + p["nodes"]["s0"]["damping"] * 0

    start = jax.tree.map(lambda x: x, gm.params)
    start["nodes"]["s0"]["stiffness"] = jnp.asarray(rng.uniform(1, 50), jnp.float32)
    res = fit(gm, loss, params=start, n_iter=5, lr=0.5)
    for k in ("damping", "rest_length", "mass", "initial_position", "initial_velocity"):
        assert np.array_equal(np.asarray(res.params["nodes"]["s0"][k]), np.asarray(start["nodes"]["s0"][k])), k
    gm.check_params(res.params)


# ---------------------------------------------------------------------------
# consistent mapping reproduces a constant field
# ---------------------------------------------------------------------------

@given(seed=st.integers(0, 2**31), ns=st.integers(3, 12), nt=st.integers(2, 15),
       kernel=st.sampled_from(["gaussian", "thin_plate_spline", "multiquadric", "inverse_multiquadric"]))
@settings(max_examples=40, deadline=None)
def test_consistent_rbf_mapping_reproduces_constants(seed, ns, nt, kernel):
    from maddening.core.coupling.mapping import rbf_mapping
    rng = np.random.default_rng(seed)
    src = np.sort(rng.uniform(0, 1, ns))
    assume(np.min(np.diff(src)) > 1e-3)
    tgt = rng.uniform(0, 1, nt)
    m = rbf_mapping(src, tgt, kernel=kernel, epsilon=2.0)
    out = m.apply(jnp.full((ns,), 3.0, jnp.float32))
    np.testing.assert_allclose(np.asarray(out), 3.0, rtol=1e-3, atol=1e-3)
    # conservative: sum preserved
    mc = rbf_mapping(src, tgt, kernel=kernel, epsilon=2.0, mode="conservative")
    f = jnp.asarray(rng.uniform(-1, 1, ns), jnp.float32)
    assert float(jnp.sum(mc.apply(f))) == pytest.approx(float(jnp.sum(f)), rel=1e-3, abs=1e-3)


def test_projection_1d_mapping_degenerate_target_cell():
    from maddening.core.coupling.mapping import projection_1d_mapping
    m = projection_1d_mapping([0.0, 0.5, 1.0], [0.0, 0.5, 0.5, 1.0])
    H = np.asarray(m.H)
    assert np.all(np.isfinite(H))


# ---------------------------------------------------------------------------
# fim / masks
# ---------------------------------------------------------------------------

def test_fim_mask_with_numpy_bools_and_arrays():
    from maddening.sysid import fim
    gm = _random_graph(np.random.default_rng(0), 1, 0, False, False, False)
    step, ext = gm._compiled_step, gm._default_external_inputs()
    mask = jax.tree.map(lambda x: np.bool_(True), gm.params)
    mask["nodes"]["s0"]["initial_position"] = np.bool_(False)
    r = fim(lambda p: step(gm._state, ext, p)["s0"]["position"], gm.params, mask=mask)
    assert "initial_position" not in "".join(r.param_names)


# ---------------------------------------------------------------------------
# log transform with lo != 0: constrain can land exactly ON the strict bound
# ---------------------------------------------------------------------------

def test_log_constrain_with_nonzero_lower_bound_lands_on_bound_and_breaks_fit_continuation():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, initial_position=1.0))
    gm.compile()
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(8.0, None), transform="log"))
    u = gm.unconstrain()
    u["nodes"]["s"]["stiffness"] = jnp.float32(-15.0)      # a perfectly ordinary Adam coordinate
    p = gm.constrain(u)
    assert float(p["nodes"]["s"]["stiffness"]) > 8.0, "constrain landed on the strict bound"
    gm.check_params(p)                                      # ValueError: below bound 8.0
    assert np.isfinite(float(gm.unconstrain(p)["nodes"]["s"]["stiffness"]))
