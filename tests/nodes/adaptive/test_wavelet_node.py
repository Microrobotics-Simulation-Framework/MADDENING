"""``WaveletAdaptiveNode`` against the ``AdaptiveNode`` author-facing contract.

The 0.4.0 adaptive-node audit listed seven things a second subclass
author gets wrong that the base class does not stop.  Each has a section
below naming how the wavelet node handles it and pinning that it does:

1. an empty active set at cold start;
2. a non-boolean mask;
3. a basis array baked from a trainable parameter in ``__init__``;
4. ``update`` ignoring ``dt`` / ``boundary_inputs`` (edge source only);
5. ``extra_initial_state()`` fields carried unmasked;
6. hooks receiving the non-numeric diagnostic settings in ``params``;
7. ``compute_full_basis_gradient`` assuming an all-true mask is a valid solve.

The rest is the node's own behaviour: traceability, the frozen-set
adjoint against finite differences with the active set held fixed,
round trips, graph integration, the alternative solver and boundary.
The suite's ``conftest.py`` enables float64 per test.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.graph_manager import GraphManager
from maddening.core.node import static_data_dep_violations
from maddening.core.params import ParamSpec
from maddening.nodes.adaptive import AdaptiveNode, WaveletAdaptiveNode
from maddening.nodes.ball import BallNode
from maddening.testing.verification import DEFAULT_CHECKS, verify_node

THETA = 0.42


def _node(**kw) -> WaveletAdaptiveNode:
    kw.setdefault("n_levels", 6)
    kw.setdefault("theta", THETA)
    return WaveletAdaptiveNode("wavelet", 1.0, **kw)


def _J(node, state, **leaves):
    return node.objective(node.update(state, {}, 1.0, params=leaves), {})


def _fd(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def _same_mask_across(node, state, key, x, h=1e-5) -> bool:
    lo = node.compute_active_set(state, node._merged({key: x - h}))
    hi = node.compute_active_set(state, node._merged({key: x + h}))
    return bool(jnp.array_equal(lo, hi))


# ---------------------------------------------------------------------------
# construction and metadata
# ---------------------------------------------------------------------------

def test_defaults_size_the_buffer_and_the_budget_from_the_levels():
    node = _node()
    assert node.n_max == 128 == node.side and node.dim == 1
    assert node.params["k"] == 8 == node.k
    assert node.state_fields() == ["c", "mask"]
    assert node.grid_shape == (128,)
    assert set(node.params_pytree()) == {"theta", "sigma", "mass", "sensor"}
    assert "n_max" not in node.params


@pytest.mark.parametrize("bad", [
    dict(dim=4), dict(order=3), dict(boundary="neumann"), dict(preconditioner="ilu"),
    dict(frozen_solver="lu"), dict(mass=0.0), dict(mass=-1.0), dict(sigma=0.0),
    dict(k=1), dict(k=129), dict(sensor=(0.3, 0.4)), dict(sensor=(1.5,)),
    dict(n_levels=0),
])
def test_an_invalid_structural_setting_is_refused_at_construction(bad):
    with pytest.raises(ValueError):
        _node(**bad)


def test_the_node_is_experimental_and_its_metadata_is_filled_in():
    assert WaveletAdaptiveNode._stability_level == StabilityLevel.EXPERIMENTAL
    m = WaveletAdaptiveNode.meta
    assert m.algorithm_id == "MADD-NODE-010"
    assert m.stability == StabilityLevel.EXPERIMENTAL
    assert m.description and m.assumptions and m.limitations and m.hazard_hints
    assert m.implementation_map and m.references
    assert m.discretization_order is not None
    assert m.discretization_order.spatial == 2.0
    assert m.discretization_order.temporal is None


# ---------------------------------------------------------------------------
# 1. an empty active set at cold start
# ---------------------------------------------------------------------------

def test_the_cold_start_set_is_never_empty_because_the_coarse_level_seeds_it():
    """The selection reads the parameters, never ``c``, so the all-zero
    cold-start coefficients cannot empty it; the coarse level is always in."""
    node = _node()
    empty = {"c": jnp.zeros(node.n_max, dtype=node.dtype),
             "mask": jnp.zeros(node.n_max, dtype=bool)}
    mask = node.compute_active_set(empty, node.params, prev=None, is_cold_start=True)
    assert bool(jnp.all(mask[node._coarse]))
    assert int(node._coarse.sum()) <= int(mask.sum()) <= node.k
    state = node.initial_state()
    assert bool(jnp.array_equal(state["mask"], mask))


@pytest.mark.parametrize("theta", [0.02, 0.25, 0.5, 0.75, 0.98])
def test_the_selection_is_non_empty_and_within_budget_across_the_source_range(theta):
    node = _node(blindness_gate=False)
    state = node.initial_state()
    mask = node.compute_active_set(state, node._merged({"theta": theta}), prev=state["mask"])
    assert bool(jnp.all(mask[node._coarse])) and int(mask.sum()) <= node.k


# ---------------------------------------------------------------------------
# 2. a non-boolean mask
# ---------------------------------------------------------------------------

def test_the_hook_itself_returns_a_boolean_mask_not_only_the_validated_state():
    node = _node()
    state = node.initial_state()
    mask = node.compute_active_set(state, node.params, prev=state["mask"])
    assert mask.dtype == jnp.bool_ and mask.shape == (node.n_max,)
    full = _node(k=128, blindness_gate=False)
    mask_full = full.compute_active_set(state, full.params)
    assert mask_full.dtype == jnp.bool_ and bool(jnp.all(mask_full))


# ---------------------------------------------------------------------------
# 3. a basis array baked from a trainable parameter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,x", [("theta", THETA), ("sigma", 0.10)])
def test_gradients_with_respect_to_the_trainable_leaves_match_finite_differences(key, x):
    """Both trainable leaves enter through the right-hand side recomputed
    from ``params`` on every call, so nothing is baked.  Central
    differences with the active set asserted unchanged across the step."""
    node = _node()
    state = node.initial_state()
    assert _same_mask_across(node, state, key, x)
    f = lambda v: _J(node, state, **{key: v})
    g = float(jax.grad(f)(jnp.asarray(x)))
    fd = float(_fd(f, jnp.asarray(x)))
    assert g != 0.0
    assert abs(g - fd) / abs(fd) < 1e-6, (g, fd)


def test_mass_and_sensor_are_frozen_and_their_statics_are_declared():
    node = _node()
    specs = node.param_specs()
    assert specs["mass"].trainable is False and specs["sensor"].trainable is False
    assert specs["theta"].trainable and specs["sigma"].trainable
    assert set(node.static_data) == {"scaling", "sensor_row"}
    assert node.static_data_deps() == {"scaling": ("mass",), "sensor_row": ("sensor",)}
    assert static_data_dep_violations(node) == []


@pytest.mark.parametrize("key", ["mass", "sensor"])
def test_compile_refuses_a_graph_that_unfreezes_a_baked_parameter(key):
    """The guard the developer guide promises: a graph-level override that
    makes ``mass`` or ``sensor`` trainable is refused, naming the static."""
    gm = GraphManager()
    gm.add_node(_node())
    gm.set_param_spec("wavelet", key, ParamSpec(trainable=True))
    with pytest.raises(ValueError, match=f"derived from parameter '{key}'"):
        gm.compile()


def test_the_full_basis_gradient_is_not_bitwise_zero_for_any_trainable_leaf():
    node = _node()
    state = node.initial_state()      # would warn (-> error) on a bitwise-zero leaf
    g = node.compute_full_basis_gradient(state, None)
    assert set(g) == {"theta", "sigma", "mass", "sensor"}
    assert float(g["theta"]) != 0.0 and float(g["sigma"]) != 0.0
    assert float(g["mass"]) == 0.0 and bool(jnp.all(g["sensor"] == 0.0))


# ---------------------------------------------------------------------------
# 4. update ignores dt and boundary_inputs
# ---------------------------------------------------------------------------

def test_update_is_a_steady_solve_that_reads_neither_dt_nor_boundary_inputs():
    """The node solves a steady elliptic problem, so this is correct rather
    than a limitation it works around: two different ``dt`` and an
    unexpected boundary input give bitwise the same state.  The consequence
    is that the node is an edge *source* only (it declares no boundary
    inputs); a time-dependent wavelet node would need the hooks to receive
    ``dt``, which is the open freeze question the base class documents."""
    node = _node()
    s = node.initial_state()
    a = node.update(s, {}, 1.0)
    b = node.update(s, {}, 1e-3)
    c = node.update(s, {"anything": jnp.ones(3)}, 7.0)
    for other in (b, c):
        assert bool(jnp.array_equal(a["c"], other["c"])) and bool(jnp.array_equal(a["mask"], other["mask"]))
    assert node.boundary_input_spec() == {}


# ---------------------------------------------------------------------------
# 5. extra_initial_state fields carried unmasked
# ---------------------------------------------------------------------------

def test_the_state_is_exactly_c_and_mask_so_nothing_can_carry_a_stale_value():
    node = _node()
    assert node.extra_initial_state() == {}
    s = node.initial_state()
    assert set(s) == {"c", "mask"}
    moved = node.update(s, {}, 1.0, params={"theta": 0.7})
    assert set(moved) == {"c", "mask"}
    assert not bool(jnp.array_equal(moved["mask"], s["mask"]))
    assert bool(jnp.all(moved["c"][~moved["mask"]] == 0.0))


# ---------------------------------------------------------------------------
# 6. hooks receive the non-numeric diagnostic settings in params
# ---------------------------------------------------------------------------

def test_the_hooks_read_named_leaves_and_tolerate_the_merged_dict_with_its_strings():
    """``params`` in the hooks is ``{**self.params, **injected}``: it carries
    ``on_blind`` (str), ``blindness_gate`` (bool), ``boundary`` (str) and the
    ints alongside the physics.  The node indexes the leaves it needs and
    never maps over the dict, so the merged dict and the numeric-only one
    give the same answer."""
    node = _node()
    s = node.initial_state()
    merged = node._merged(None)
    assert isinstance(merged["on_blind"], str) and isinstance(merged["blindness_gate"], bool)
    numeric = {k: merged[k] for k in ("theta", "sigma", "mass", "sensor")}
    m1 = node.compute_active_set(s, merged, prev=s["mask"])
    m2 = node.compute_active_set(s, numeric, prev=s["mask"])
    assert bool(jnp.array_equal(m1, m2))
    c1 = node.solve_frozen(s, m1, merged)["c"]
    c2 = node.solve_frozen(s, m1, numeric)["c"]
    assert bool(jnp.array_equal(c1, c2))
    assert float(node.source_field({**merged, "junk": "ignored"})[0]) == float(node.source_field(numeric)[0])


# ---------------------------------------------------------------------------
# 7. compute_full_basis_gradient and the all-true mask
# ---------------------------------------------------------------------------

def test_the_full_basis_gradient_override_matches_a_dense_finite_difference():
    node = _node()
    s = node.initial_state()
    g = float(node.compute_full_basis_gradient(s, None)["theta"])

    def dense_J(theta):
        b = node._rhs(node._merged({"theta": theta}))
        return node._sensor_row @ jnp.linalg.solve(node._A, b)

    fd = float(_fd(dense_J, jnp.asarray(THETA)))
    assert abs(g - fd) / abs(fd) < 1e-6


def test_the_base_default_would_truncate_the_gathered_solve_which_is_why_it_is_overridden():
    """``AdaptiveNode.compute_full_basis_gradient`` runs ``solve_frozen`` with
    an all-true mask.  For the gathered solve that mask has ``n_max`` active
    entries in a ``k``-sized buffer and is silently wrong; for the masked-CG
    solve it is a genuine full solve.  Both facts pinned, so the override
    cannot be removed as redundant."""
    s = _node().initial_state()
    gathered = _node()
    override = float(gathered.compute_full_basis_gradient(s, None)["theta"])
    base_default = float(AdaptiveNode.compute_full_basis_gradient(gathered, s, None)["theta"])
    assert abs(base_default - override) / abs(override) > 1e-3

    cg = _node(frozen_solver="cg")
    base_cg = float(AdaptiveNode.compute_full_basis_gradient(cg, s, None)["theta"])
    assert abs(base_cg - override) / abs(override) < 1e-8


# ---------------------------------------------------------------------------
# accuracy, diagnostics, solver variants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_levels", [6, 7])
def test_the_adaptive_sensor_reading_is_close_to_the_full_basis_one(n_levels):
    node = _node(n_levels=n_levels)
    s = node.initial_state()
    j_cdd = float(node.objective(s, {}))
    full = _node(n_levels=n_levels, k=node.n_max, blindness_gate=False)
    j_full = float(full.objective(full.initial_state(), {}))
    assert abs(j_cdd - j_full) / abs(j_full) < 1e-2
    assert j_cdd * j_full > 0


def test_the_gradient_capture_ratio_is_near_one_for_the_local_basis():
    node = _node()
    r = node.gradient_capture_ratio(node.initial_state())
    assert 0.9 < r < 1.1, r


def test_the_masked_cg_solve_agrees_with_the_gathered_one():
    a = _node().initial_state()
    b = _node(frozen_solver="cg").initial_state()
    assert bool(jnp.array_equal(a["mask"], b["mask"]))
    assert float(jnp.max(jnp.abs(a["c"] - b["c"]))) < 1e-10


def test_the_full_budget_turns_adaptivity_off_and_reproduces_the_dense_solve():
    node = _node(k=128, blindness_gate=False)
    s = node.initial_state()
    assert bool(jnp.all(s["mask"]))
    c = jnp.linalg.solve(node._A, node._rhs(node.params))
    assert float(jnp.max(jnp.abs(s["c"] - c))) < 1e-10 * float(jnp.max(jnp.abs(c)))


def test_field_reshapes_to_the_grid_and_is_the_synthesis_of_c():
    node = _node()
    s = node.initial_state()
    u = node.field(s)
    assert u.shape == (node.n_max,) and u.reshape(node.grid_shape).shape == (128,)
    assert bool(jnp.allclose(u, node._op.Wn @ s["c"]))


def test_the_selection_is_memoryless_so_repeated_updates_at_fixed_params_do_not_chatter():
    node = _node()
    s = node.initial_state()
    a = node.update(s, {}, 1.0)
    b = node.update(a, {}, 1.0)
    assert bool(jnp.array_equal(a["mask"], b["mask"])) and bool(jnp.array_equal(a["c"], b["c"]))


@pytest.mark.parametrize("kind", ["hybrid", "full", "level", "dk"])
def test_every_preconditioner_constructs_and_solves(kind):
    node = _node(preconditioner=kind, blindness_gate=False)
    s = node.initial_state()
    assert int(s["mask"].sum()) >= int(node._coarse.sum())
    assert bool(jnp.all(jnp.isfinite(s["c"])))


# ---------------------------------------------------------------------------
# traceability
# ---------------------------------------------------------------------------

def test_jit_update_matches_eager_and_traces_once():
    node = _node()
    s = node.initial_state()
    pt = node.params_pytree()
    eager = node.update(s, {}, 1.0, params=pt)
    count = {"n": 0}

    @jax.jit
    def step(st, p):
        count["n"] += 1
        return node.update(st, {}, 1.0, params=p)

    a = step(s, pt)
    b = step(a, pt)
    step(b, pt)
    assert count["n"] == 1
    assert bool(jnp.array_equal(a["mask"], eager["mask"]))
    assert float(jnp.max(jnp.abs(a["c"] - eager["c"]))) < 1e-12


def test_scan_with_a_per_step_source_position_matches_the_python_loop():
    node = _node()
    s0 = node.initial_state()
    thetas = jnp.linspace(0.42, 0.7, 5)

    def body(st, th):
        new = node.update(st, {}, 1.0, params={"theta": th})
        return new, node.objective(new, {})

    final, ys = jax.jit(lambda st: jax.lax.scan(body, st, thetas))(s0)
    ref, masks = s0, []
    for th in thetas:
        ref = node.update(ref, {}, 1.0, params={"theta": th})
        masks.append(ref["mask"])
    assert ys.shape == (5,)
    assert bool(jnp.array_equal(final["mask"], ref["mask"]))
    assert float(jnp.max(jnp.abs(final["c"] - ref["c"]))) < 1e-12
    assert not bool(jnp.array_equal(masks[0], masks[-1]))


def test_vmap_over_the_source_position():
    node = _node()
    s = node.initial_state()
    out = jax.vmap(lambda th: node.update(s, {}, 1.0, params={"theta": th}))(jnp.array([0.3, 0.42, 0.6]))
    assert out["c"].shape == (3, node.n_max) and out["mask"].shape == (3, node.n_max)
    assert bool(jnp.all(out["mask"].sum(axis=1) <= node.k))


def test_jit_grad_matches_eager_grad_and_finite_differences():
    node = _node()
    s = node.initial_state()
    f = lambda th: _J(node, s, theta=th)
    g_eager = float(jax.grad(f)(jnp.asarray(THETA)))
    g_jit = float(jax.jit(jax.grad(f))(jnp.asarray(THETA)))
    fd = float(_fd(f, jnp.asarray(THETA)))
    assert abs(g_eager - g_jit) < 1e-12 * (1 + abs(g_eager))
    assert abs(g_jit - fd) / abs(fd) < 1e-6


def test_gradient_through_a_scan_of_updates_matches_finite_differences():
    node = _node(n_levels=5)
    s0 = node.initial_state()

    def traj(theta0):
        def body(st, i):
            new = node.update(st, {}, 1.0, params={"theta": theta0 + 0.002 * i})
            return new, node.objective(new, {})
        _, ys = jax.lax.scan(body, s0, jnp.arange(6))
        return jnp.sum(ys ** 2)

    th = jnp.asarray(0.40)
    g = float(jax.grad(traj)(th))
    fd = float(_fd(traj, th, h=1e-6))
    assert abs(g - fd) / abs(fd) < 1e-5


# ---------------------------------------------------------------------------
# other dimensions and the Dirichlet basis
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(dim=2, n_levels=3),
    dict(dim=3, n_levels=2, n_coarse=1),
    dict(boundary="dirichlet", n_levels=5),
], ids=["2d", "3d", "dirichlet"])
def test_other_dimensions_and_the_dirichlet_basis_cold_start_and_differentiate(kw):
    node = _node(**kw)
    s = node.initial_state()
    assert bool(jnp.all(s["mask"][node._coarse])) and int(s["mask"].sum()) <= node.k
    assert node.grid_shape == (node.side,) * node.dim
    assert _same_mask_across(node, s, "theta", THETA)
    f = lambda th: _J(node, s, theta=th)
    g = float(jax.grad(f)(jnp.asarray(THETA)))
    fd = float(_fd(f, jnp.asarray(THETA)))
    assert abs(g - fd) / abs(fd) < 1e-6


@pytest.mark.slow
def test_three_dimensional_node_at_the_validated_size():
    node = _node(dim=3, n_levels=3, n_coarse=1)
    s = node.initial_state()
    assert node.n_max == 512 and int(s["mask"].sum()) == node.k
    assert 0.9 < node.gradient_capture_ratio(s) < 1.1


# ---------------------------------------------------------------------------
# round trips and graph integration
# ---------------------------------------------------------------------------

def test_the_node_is_rebuilt_from_its_own_params():
    node = _node(n_levels=5, k=6, boundary="dirichlet", frozen_solver="cg",
                 sensor=(0.6,), mass=2.0, sigma=0.2, blindness_gate=False)
    clone = type(node)(name=node.name, timestep=node.delta_t, **node.params)
    assert clone.params == node.params and clone.n_max == node.n_max
    a, b = node.initial_state(), clone.initial_state()
    assert bool(jnp.array_equal(a["mask"], b["mask"]))
    assert bool(jnp.array_equal(a["c"], b["c"]))


def test_a_list_valued_sensor_from_a_json_round_trip_is_accepted():
    node = _node(sensor=[0.25], blindness_gate=False)
    assert node.params["sensor"] == (0.25,)


def test_add_node_compile_run_scan_and_the_params_pytree():
    gm = GraphManager()
    gm.add_node(_node())
    gm.compile()
    leaves = gm.params["nodes"]["wavelet"]
    assert set(leaves) == {"theta", "sigma", "mass", "sensor"}
    before = gm.run_scan(2)["wavelet"]
    assert before["mask"].dtype == jnp.bool_ and int(before["mask"].sum()) <= 8
    gm.params["nodes"]["wavelet"]["theta"] = 0.8
    after = gm.run_scan(1)["wavelet"]
    assert not bool(jnp.array_equal(before["mask"], after["mask"]))


def test_gradient_through_the_compiled_step_reaches_theta_and_sigma():
    gm = GraphManager()
    gm.add_node(_node())
    gm.compile()
    node = gm._nodes["wavelet"].node

    def loss(p):
        out = gm._compiled_step(gm._state, gm._default_external_inputs(), p)
        return node.objective(out["wavelet"], {})

    g = jax.grad(loss)(jax.tree.map(lambda x: x, gm.params))["nodes"]["wavelet"]
    assert float(g["theta"]) != 0.0 and float(g["sigma"]) != 0.0
    assert bool(jnp.isfinite(g["theta"]))


def test_a_config_round_trip_of_a_graph_preserves_the_trajectory():
    gm = GraphManager()
    gm.add_node(_node(n_levels=5, blindness_gate=False))
    gm.compile()
    reloaded = GraphManager.from_dict(gm.to_dict(), {"WaveletAdaptiveNode": WaveletAdaptiveNode})
    reloaded.compile()
    a, b = gm.run_scan(2)["wavelet"], reloaded.run_scan(2)["wavelet"]
    assert bool(jnp.array_equal(a["mask"], b["mask"]))
    assert bool(jnp.array_equal(a["c"], b["c"]))


def test_the_node_drives_a_downstream_node_through_an_edge_at_the_default_precision():
    """An edge out of ``c`` into a ``BallNode`` (the smallest graph that makes
    the node an edge source), at the float32 precision the framework runs
    at by default -- x64 is switched off inside this test because the
    suite's conftest turns it on."""
    prior = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        node = _node(blindness_gate=False)
        gm = GraphManager()
        gm.add_node(node)
        gm.add_node(BallNode("ball", 1.0, initial_position=2.0, elasticity=0.5))
        gm.add_edge("wavelet", "ball", "c", "table_position", transform="extract_first")
        gm.compile()
        out = gm.run_scan(3)
        assert out["wavelet"]["c"].dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(out["ball"]["position"])))
    finally:
        jax.config.update("jax_enable_x64", prior)


def test_under_x64_the_mixed_graph_fails_on_the_downstream_carry_not_on_the_wavelet_one():
    """The dtype-policy measurement for the port, pinned.

    The strict xfail in ``tests/property/test_adaptive_invariants.py``
    attributes the ``run_scan`` failure under ``jax_enable_x64`` to the
    adaptive node being "the only node whose state dtype follows the
    flag".  Measured here: build the wavelet node with ``dtype=float32``
    requested explicitly, so the base class enforces float32 on ``c`` --
    and the graph still fails, on ``carry['ball']``.  The ball promotes
    its own float32 state because ``params_pytree()`` injects float64
    leaves under x64 (a plain ``BallNode`` alone fails the same way on
    this tree).  So "``AdaptiveNode`` stops following the flag" cannot be
    the fix on its own; the graph coercing each node's update output back
    to its carry dtype is the candidate that also covers the other nodes.
    This test flips when that lands: replace it with the positive run."""
    node = _node(dtype=jnp.float32, blindness_gate=False)
    gm = GraphManager()
    gm.add_node(node)
    gm.add_node(BallNode("ball", 1.0, initial_position=2.0, elasticity=0.5))
    gm.add_edge("wavelet", "ball", "c", "table_position", transform="extract_first")
    gm.compile()
    assert node.initial_state()["c"].dtype == jnp.float32
    with pytest.raises(TypeError, match="carry input and carry output") as info:
        gm.run_scan(3)
    message = str(info.value)
    assert "carry['ball']" in message
    assert "carry['wavelet']" not in message


def test_verify_node_battery_passes():
    node = _node(n_levels=4, blindness_gate=False)
    results = verify_node(node, bounds={"c": (-1.0, 1.0)}, dtype=np.float64,
                          max_examples=10, derandomize=True)
    assert set(results) == set(DEFAULT_CHECKS)
    bad = [str(r) for r in results.values() if not r.passed]
    assert not bad, "\n".join(bad)
    assert all(results[k].status == "PASS" for k in
               ("params_consistent", "params_gradient_finite", "params_effective"))
