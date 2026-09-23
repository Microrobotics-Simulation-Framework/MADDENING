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
round trips, graph integration, the alternative solver and boundary,
and the pins from the merged-tree audit -- the CDD seed (every level-0
function, not the coarse block alone) fits the budget in every
dimension and both boundaries, an oversized set is refused rather than
truncated, construction inside a trace, the selection diagnostics, the
sensor at the periodic seam, and the largest sizes the metadata claims.
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


def _fd4(f, x, h=1e-4):
    """Fourth-order central difference; truncation ~1e-12 relative on these problems."""
    return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / (12 * h)


def _J_fixed(node, state, mask, **leaves):
    """The objective through the frozen solve on ``mask`` -- the set held fixed by construction."""
    return node.objective(node.solve_frozen(state, mask, node._merged(leaves)), {})


def _assert_frozen_gradient_matches_fd(node, state, x=None, tol=1e-8):
    """The audit's method: ``grad`` through ``update`` (which re-selects the
    set) equals ``grad`` through the frozen solve on the set ``state`` was
    solved on, and that equals a fourth-order central difference of the
    frozen solve.  Differencing through ``update`` instead would compare
    two active-set regions whenever ``theta`` sits at a switch, which it
    does for the 2-D Dirichlet configuration at 0.42."""
    x = node.params["theta"] if x is None else x
    mask = state["mask"]
    g_upd = float(jax.grad(lambda th: _J(node, state, theta=th))(jnp.asarray(x)))
    g_fix = float(jax.grad(lambda th: _J_fixed(node, state, mask, theta=th))(jnp.asarray(x)))
    fd = float(_fd4(lambda th: _J_fixed(node, state, mask, theta=th), jnp.asarray(x)))
    assert abs(g_upd - g_fix) < 1e-12 * (1.0 + abs(g_fix)), (g_upd, g_fix)
    assert abs(g_fix - fd) / abs(fd) < tol, (g_fix, fd)
    assert g_fix != 0.0


def _same_mask_across(node, state, key, x, h=1e-5) -> bool:
    lo = node.compute_active_set(state, node._merged({key: x - h}))
    hi = node.compute_active_set(state, node._merged({key: x + h}))
    return bool(jnp.array_equal(lo, hi))


def _masked_dense_reference(node, mask, params):
    """The frozen solve on ``mask`` done the obvious way: index, solve, scatter."""
    idx = jnp.flatnonzero(mask)
    sub = jnp.linalg.solve(node._A[jnp.ix_(idx, idx)], node._rhs(params)[idx])
    return jnp.zeros(node.n_max, dtype=node.dtype).at[idx].set(sub)


def _level0_count(node) -> int:
    nc = node.params["n_coarse"]
    return (2 * nc) ** node.dim if node.params["boundary"] == "periodic" else (2 * nc + 1) ** node.dim


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
    # the seed is level 0 = coarse block + first band = 4 functions here:
    # k=3 passed the old n_coarse**dim bound and was silently truncated
    dict(k=3),
])
def test_an_invalid_structural_setting_is_refused_at_construction(bad):
    with pytest.raises(ValueError):
        _node(**bad)


def test_a_budget_below_the_seed_is_refused_naming_both_numbers_and_the_remedy():
    with pytest.raises(ValueError, match=r"got k=3 with seed=4 .*Pass k=4 or larger"):
        _node(k=3)
    with pytest.raises(ValueError, match=r"got k=10 with seed=16 .*\(2 n_coarse\)\*\*dim = 16"):
        _node(dim=2, n_levels=2, k=10)
    with pytest.raises(ValueError, match=r"seed=25 .*\(2 n_coarse \+ 1\)\*\*dim = 25"):
        _node(dim=2, n_levels=2, boundary="dirichlet", k=9)


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


def test_the_base_default_full_basis_gradient_is_refused_for_the_gathered_solve_which_is_why_it_is_overridden():
    """``AdaptiveNode.compute_full_basis_gradient`` runs ``solve_frozen`` with
    an all-true mask.  For the gathered solve that mask has ``n_max`` active
    entries in a ``k``-sized buffer: the node refuses it by name (it used to
    be silently truncated to a plausible wrong gradient), and the override
    -- a dense solve -- is what the diagnostics use.  For the masked-CG solve
    the base default is a genuine full solve and agrees with the override.
    Both facts pinned, so the override cannot be removed as redundant."""
    s = _node().initial_state()
    gathered = _node()
    override = float(gathered.compute_full_basis_gradient(s, None)["theta"])
    with pytest.raises(ValueError, match="active set has 128 functions .* k = 8"):
        AdaptiveNode.compute_full_basis_gradient(gathered, s, None)

    cg = _node(frozen_solver="cg")
    base_cg = float(AdaptiveNode.compute_full_basis_gradient(cg, s, None)["theta"])
    assert abs(base_cg - override) / abs(override) < 1e-8


# ---------------------------------------------------------------------------
# the CDD seed fits the budget -- every dimension, both boundaries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    dict(dim=2, n_levels=1), dict(dim=2, n_levels=2), dict(dim=2, n_levels=2, n_coarse=3),
    dict(dim=3, n_levels=1), dict(dim=3, n_levels=2),
    dict(boundary="dirichlet", dim=2, n_levels=1, n_coarse=1),
    dict(boundary="dirichlet", dim=2, n_levels=1),
    dict(boundary="dirichlet", dim=2, n_levels=2, n_coarse=1),
    dict(boundary="dirichlet", dim=2, n_levels=2),
    dict(boundary="dirichlet", dim=2, n_levels=2, n_coarse=3),
    dict(boundary="dirichlet", dim=3, n_levels=1, n_coarse=1),
    dict(boundary="dirichlet", dim=3, n_levels=1),
    dict(boundary="dirichlet", dim=3, n_levels=2, n_coarse=1),
], ids=lambda kw: "-".join(f"{k}={v}" for k, v in kw.items()))
def test_the_default_budget_holds_the_whole_level_zero_seed_so_no_coefficient_is_dropped(kw):
    """The CDD seed is every level-0 function -- the coarse block AND the
    first detail band, ``(2 n_coarse)**dim`` periodic or ``(2 n_coarse + 1)**dim``
    Dirichlet -- and the default ``k`` must hold it.  These are default
    constructions on which the seed used to exceed ``k`` and the gathered
    solve silently dropped the excess (``|mask| = 16`` in a buffer of 8 at
    ``dim=2, n_levels=2``, ``c`` 4% off and ``dJ/dtheta`` five times too
    small); 13 of the audit's 16, the three 3-D ones above 1000 functions
    left out for time.  Pinned: the seed count, the bound, and that every
    active coefficient is solved -- ``c`` equals the masked dense solve."""
    node = _node(blindness_gate=False, **kw)
    seed = int(node._coarse.sum())
    assert seed == _level0_count(node) == node._seed_size
    assert seed <= node.k <= node.n_max
    s = node.initial_state()
    assert seed <= int(s["mask"].sum()) <= node.k
    assert int(s["mask"].sum()) == int((s["c"] != 0).sum())
    ref = _masked_dense_reference(node, s["mask"], node.params)
    assert float(jnp.max(jnp.abs(s["c"] - ref))) < 1e-12 * float(jnp.max(jnp.abs(ref)))


@pytest.mark.parametrize("kw", [dict(dim=2, n_levels=2), dict(dim=3, n_levels=2)],
                         ids=["2d_8x8", "3d_8x8x8"])
def test_the_frozen_gradient_matches_finite_differences_where_the_seed_used_to_exceed_the_budget(kw):
    """The two configurations the audit measured: ``dJ/dtheta`` was -4.9e-3
    against -2.6e-2 (2-D) and -1.1e-3 against -7.7e-3 (3-D) because half
    the seed was dropped, while the capture ratio read 0.97 through the
    same truncated solve.  Gate on, mask held fixed across the stencil."""
    node = _node(**kw)
    s = node.initial_state()
    assert int(s["mask"].sum()) == int((s["c"] != 0).sum()) <= node.k
    _assert_frozen_gradient_matches_fd(node, s)
    assert 0.95 < node.gradient_capture_ratio(s) < 1.05


@pytest.mark.parametrize("kw", [dict(dim=2, n_levels=1, n_coarse=3), dict(dim=1, n_levels=1, n_coarse=16)])
def test_the_default_budget_never_violates_the_constructors_own_bound(kw):
    """``dim=2, n_levels=1, n_coarse=3`` used to raise ``k must satisfy ...
    got 8`` for a ``k`` the caller never passed.  The default is sized from
    the seed, so an unset ``k`` always constructs."""
    node = _node(blindness_gate=False, **kw)
    assert node.k == node.params["k"] == min(node.n_max, max(node._seed_size, 8, node.n_max // 16))
    assert node._seed_size <= node.k <= node.n_max


def test_an_active_set_larger_than_the_budget_is_refused_eagerly_and_poisoned_under_jit():
    """The truncation the audit found cannot happen silently any more.  The
    node's own selection never exceeds ``k`` (seed validated, marking
    capped); a larger mask handed to the gathered solve is refused with a
    message on the eager path, and under ``jit`` -- where nothing can raise
    -- the gathered block is NaN rather than a plausible wrong answer.  A
    mask exactly at the budget solves, and the masked-CG solve takes any."""
    node = _node()
    s = node.initial_state()
    too_many = jnp.zeros(node.n_max, dtype=bool).at[: node.k + 1].set(True)
    with pytest.raises(ValueError, match=r"active set has 9 functions .* k = 8"):
        node.solve_frozen(s, too_many, node.params)
    c = jax.jit(lambda m: node.solve_frozen(s, m, node.params)["c"])(too_many)
    assert bool(jnp.all(jnp.isnan(c[: node.k]))) and bool(jnp.all(c[node.k + 1:] == 0.0))
    exact = jnp.zeros(node.n_max, dtype=bool).at[: node.k].set(True)
    assert bool(jnp.all(jnp.isfinite(node.solve_frozen(s, exact, node.params)["c"])))
    cg = _node(frozen_solver="cg")
    assert bool(jnp.all(jnp.isfinite(cg.solve_frozen(s, too_many, cg.params)["c"])))


# ---------------------------------------------------------------------------
# construction inside a trace
# ---------------------------------------------------------------------------

def test_the_node_can_be_constructed_inside_a_jit_trace():
    """Every constant is built on the host from static settings, so a
    function that builds a fresh node per call traces under ``jax.jit`` --
    it used to fail in the operator assembly with a
    ``TracerArrayConversionError`` -- and gives the eager answer."""
    def run(th):
        node = WaveletAdaptiveNode("w", 1.0, n_levels=4, blindness_gate=False)
        out = node.update(node.initial_state(), {}, 1.0, params={"theta": th})
        return node.objective(out, {}), out["mask"]

    j_eager, m_eager = run(jnp.asarray(THETA))
    j_jit, m_jit = jax.jit(run)(jnp.asarray(THETA))
    assert bool(jnp.array_equal(m_eager, m_jit))
    assert abs(float(j_jit) - float(j_eager)) < 1e-12
    g = jax.grad(lambda th: run(th)[0])(jnp.asarray(THETA))
    assert bool(jnp.isfinite(g)) and float(g) != 0.0


def test_a_fresh_graph_holding_the_node_can_be_built_inside_the_fim_trace():
    """The audit's scenario: ``fim`` jits a ``residual_fn`` that constructs a
    fresh graph per call.  Construction used to fail in ``assemble_operator``
    with a ``TracerArrayConversionError`` whose traceback pointed at
    ``sysid.py``."""
    from maddening.sysid import fim

    def build():
        gm = GraphManager()
        gm.add_node(_node(n_levels=4, blindness_gate=False))
        gm.compile()
        return gm

    gm0 = build()
    truth = gm0.run_scan_with_history(3)[1]["wavelet"]["c"].reshape(-1)

    def residual(p):
        return build().run_scan_with_history(3, params=p)[1]["wavelet"]["c"].reshape(-1) - truth

    rep = fim(residual, gm0.params, scale="relative")
    assert rep.rank >= 1
    finite = {n for n, c in zip(rep.param_names, np.asarray(rep.crb)) if np.isfinite(c)}
    assert any("theta" in n for n in finite) and any("sigma" in n for n in finite), rep.param_names


# ---------------------------------------------------------------------------
# selection diagnostics and the sensor at the periodic seam
# ---------------------------------------------------------------------------

def test_selection_diagnostics_report_whether_the_budget_or_the_iteration_bound_ended_the_selection():
    """At the default budget the loop stops on the budget in a few
    iterations.  At ``k = 64`` on the 128-point basis it stops on the bound
    (30) short of the budget (54 measured; the exact count is not pinned
    across jaxlib lanes) and ``k = 96`` stops at the same set.  The mask is
    still valid; its reading is what ``MADD-VER-015`` measures."""
    default = _node(blindness_gate=False).selection_diagnostics()
    assert default["budget_reached"] and default["active"] == default["k"] == 8
    assert 0 < default["outer_iterations"] < default["max_outer"] == 30
    stalled = _node(k=64, blindness_gate=False).selection_diagnostics()
    assert not stalled["budget_reached"] and stalled["outer_iterations"] == 30
    assert 32 < stalled["active"] < 64
    also = _node(k=96, blindness_gate=False).selection_diagnostics()
    assert also["active"] == stalled["active"] and not also["budget_reached"]
    full = _node(k=128, blindness_gate=False).selection_diagnostics()
    assert full == {"active": 128, "k": 128, "outer_iterations": 0, "max_outer": 30,
                    "budget_reached": True}
    moved = _node(blindness_gate=False).selection_diagnostics({"theta": 0.7})
    assert moved["budget_reached"] and moved["active"] == 8


def test_a_periodic_sensor_at_one_snaps_to_its_periodic_image_and_a_dirichlet_one_to_the_wall_neighbour():
    """``x = 1.0`` is ``x = 0.0`` on a periodic axis, so the sensor row is the
    same (it used to snap to the last point ``(side - 1) / side``); on a
    Dirichlet axis there is no grid point at the wall and the nearest
    interior point is taken."""
    at_one = _node(sensor=(1.0,), blindness_gate=False)
    at_zero = _node(sensor=(0.0,), blindness_gate=False)
    assert at_one._sensor_index == at_zero._sensor_index == 0
    assert bool(jnp.array_equal(at_one._sensor_row, at_zero._sensor_row))
    assert _node(sensor=(0.999,), blindness_gate=False)._sensor_index == 0      # nearer 1.0 than 127/128
    assert _node(sensor=(0.99,), blindness_gate=False)._sensor_index == 127     # nearer 127/128 than 1.0
    wall = _node(boundary="dirichlet", n_levels=5, sensor=(1.0,), blindness_gate=False)
    assert wall._sensor_index == wall.side - 1
    two_d = _node(dim=2, n_levels=3, sensor=(1.0, 0.5), blindness_gate=False)
    assert two_d._sensor_index == 0 * two_d.side + two_d.side // 2


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
    dict(boundary="dirichlet", dim=2, n_levels=3),
    dict(boundary="dirichlet", dim=3, n_levels=2, n_coarse=1, k=40),
], ids=["2d", "3d", "dirichlet", "dirichlet_2d", "dirichlet_3d"])
def test_other_dimensions_and_the_dirichlet_basis_cold_start_and_differentiate(kw):
    """Periodic 2-D / 3-D and the Dirichlet basis in 1-D, 2-D (23^2, budget
    33 over a seed of 25) and 3-D (7^3, budget 40 over a seed of 27): cold
    start, every active coefficient solved, the frozen gradient against
    central differences with the set held fixed."""
    node = _node(**kw)
    s = node.initial_state()
    assert bool(jnp.all(s["mask"][node._coarse])) and int(s["mask"].sum()) <= node.k
    assert int(s["mask"].sum()) == int((s["c"] != 0).sum())
    assert node.grid_shape == (node.side,) * node.dim
    _assert_frozen_gradient_matches_fd(node, s)


@pytest.mark.slow
def test_three_dimensional_node_at_the_validated_size():
    node = _node(dim=3, n_levels=3, n_coarse=1)
    s = node.initial_state()
    assert node.n_max == 512 and int(s["mask"].sum()) == node.k
    assert 0.9 < node.gradient_capture_ratio(s) < 1.1


@pytest.mark.slow
@pytest.mark.parametrize("kw", [dict(dim=2, n_levels=5), dict(dim=3, n_levels=3)],
                         ids=["64x64", "16x16x16"])
def test_the_largest_sizes_the_metadata_claims_construct_select_solve_and_differentiate(kw):
    """The sizes ``NodeMeta.limitations`` claims as validated (``64^2`` and
    ``16^3``, 4096 functions), actually constructed: default budget 256,
    seed inside it, the gathered solve equal to the masked dense solve,
    ``jax.grad`` against central differences with the set held fixed, and
    the capture ratio.  About 10-15 s each on a loaded 24-core box, of
    which construction is ~7 s."""
    node = _node(blindness_gate=False, **kw)
    assert node.n_max == 4096 and node.k == 256 and node._seed_size <= node.k
    s = node.initial_state()
    assert int(s["mask"].sum()) == int((s["c"] != 0).sum()) == node.k
    ref = _masked_dense_reference(node, s["mask"], node.params)
    assert float(jnp.max(jnp.abs(s["c"] - ref))) < 1e-12 * float(jnp.max(jnp.abs(ref)))
    _assert_frozen_gradient_matches_fd(node, s)
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
