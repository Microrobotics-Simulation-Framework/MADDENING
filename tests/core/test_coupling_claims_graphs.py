"""The coupling claims inventory's graph-level rows, each at the edge of its domain.

``docs/validation/coupling_claims.yaml`` lists every documented claim about
coupling with a test that can fail.  The rows below had no test, or none at
the edge of the conditions the documentation states; each test names the row
it pins.  A test that fails on the tree is a strict ``xfail`` whose reason
starts with its row id, and the row is ``failing``; the compliance test
``tests/compliance/test_coupling_claims.py`` holds the two together.

Most graphs are the synthetic relays of :mod:`tests.property.coupled_graphs`
(``x <- alpha x_pre + sum_j G_j u_j + b``), whose fixed point is a float64
linear solve, built once per module.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import (
    convergence_criterion,
    reported_error_estimate,
    residual_precision_floor,
)
from maddening.core.graph_manager import GraphManager, _group_evaluations
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.simulation.profiler import profile_graph
from maddening.nodes.spring import SpringDamperNode
from maddening.sysid import observations_from_history, windowed_loss
from tests.property import coupled_graphs as cg

ACCELERATIONS = ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj")
F32 = jnp.float32


def _knobs(acceleration, **extra):
    group = dict(acceleration=acceleration, **extra)
    if acceleration == "fixed":
        group.setdefault("relaxation", 0.8)
    if acceleration == "iqn-imvj":
        group.setdefault("jacobian_reuse", 2)
    return group


def _build(gdef, group, *, node_order=None, edge_order=None, compile=True):
    """*gdef* built with its nodes and edges added in the given orders."""
    gm = GraphManager()
    nodes = list(gdef.nodes) if node_order is None else [gdef.nodes[i] for i in node_order]
    for nd in nodes:
        gm.add_node(cg.make_node(nd, gdef.n))
    edges = list(gdef.edges) if edge_order is None else [gdef.edges[i] for i in edge_order]
    for e in edges:
        gm.add_edge(e.src, e.dst, e.field, f"u{e.port}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        gm.add_coupling_group(list(gdef.group_nodes), **cg.live_knobs(group))
    if compile:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*multi-rate.*")
            gm.compile()
    return gm


def _step_params(gm, values):
    return cg.params_for(gm, values)


# ---------------------------------------------------------------------------
# CPL-006: one staggered pass differentiates like fori
# ---------------------------------------------------------------------------


def test_a_single_pass_group_differentiates_like_fori():
    """CPL-006: at ``max_iterations=1`` ``"ift"`` and ``"fori"`` give the same gradient, bitwise."""
    gdef = cg._cycle(2, 2, outside=False, leaves=())
    values = cg.draw_values(np.random.default_rng(3), gdef, 0.9)
    grads = {}
    for solver in ("ift", "fori"):
        gm = _build(gdef, dict(max_iterations=1, solver=solver, diagnostics=True))
        cg.set_initial(gm, values)
        params = _step_params(gm, values)
        state, ext = gm._state, gm._default_external_inputs()

        def loss(p, gm=gm, state=state, ext=ext):
            out = gm._compiled_step(state, ext, p)
            return jnp.sum(out["g0"]["x"]) + jnp.sum(out["g1"]["x"] ** 2)

        grads[solver] = jax.grad(loss)(params)["nodes"]
    for node in ("g0", "g1"):
        for leaf, g in grads["ift"][node].items():
            np.testing.assert_array_equal(np.asarray(g), np.asarray(grads["fori"][node][leaf]),
                                          err_msg=f"{node}.{leaf}")


# ---------------------------------------------------------------------------
# CPL-010: a power-of-two scale at the edge of float32's range
# ---------------------------------------------------------------------------

_UNITS_GDEF = dataclasses.replace(
    cg._cycle(3, 2, chords=((0, 2),), leaves=()),
    nodes=tuple(dataclasses.replace(nd, beta=0.0, nonlinear=False, leaves=())
                for nd in cg._cycle(3, 2, chords=((0, 2),), leaves=()).nodes))


def _scaled_values(values, s):
    return {nm: {"G": v["G"], "b": np.asarray(v["b"] * np.float32(s), np.float32),
                 "x0": np.asarray(v["x0"] * np.float32(s), np.float32)}
            for nm, v in values.items()}


# Aitken, "fixed" and IQN-IMVJ failed this until the accelerators formed their
# steps in power-of-two frames (MADD-ANO-108).
@pytest.mark.parametrize("acceleration", ["none", "aitken", "fixed", "iqn-ils", "iqn-imvj"])
def test_a_group_at_2_to_the_minus_110_takes_the_unscaled_passes(acceleration):
    """CPL-010: "a group at any power-of-two scale reproduce[s] the unscaled run bit for bit".

    ``2**-110``: every state ~1e-33, inside float32's normal range, with the
    map's own products (gains of 1e-3 times states of 1e-33) still normal;
    a one-ulp change of a field is subnormal.  The release notes' harness
    stops at ``2**-53``.
    """
    gdef = _UNITS_GDEF
    gm = _build(gdef, _knobs(acceleration, max_iterations=60, tolerance=1e-5))
    values = cg.draw_values(np.random.default_rng(11), gdef, 0.9)
    s = 2.0 ** -110
    base = cg.trajectory(gm, gdef, values, 2)
    run = cg.trajectory(gm, gdef, _scaled_values(values, s), 2)
    for k, ((st0, _m0, r0), (st1, _m1, r1)) in enumerate(zip(base, run)):
        assert (r1["iterations"], r1["converged"], r1["residual"]) == (
            r0["iterations"], r0["converged"], r0["residual"]), (k, r0, r1)
        for nm in gdef.group_nodes:
            np.testing.assert_array_equal(np.asarray(st1[nm]["x"], np.float32) / np.float32(s),
                                          st0[nm]["x"], err_msg=f"step {k + 1}, {nm}")


# ---------------------------------------------------------------------------
# CPL-014: iqn-imvj without reuse is iqn-ils
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_imvj_without_reuse_is_iqn_ils(solver):
    """CPL-014: ``jacobian_reuse=0`` "means no reuse (same as IQN-ILS)", step after step."""
    gdef = cg.TRIANGLE
    values = cg.draw_values(np.random.default_rng(5), gdef, 0.9, nonnormal=True)
    runs = {}
    for acc in ("iqn-ils", "iqn-imvj"):
        gm = _build(gdef, dict(acceleration=acc, jacobian_reuse=0, max_iterations=20,
                               tolerance=1e-5, solver=solver, diagnostics=solver == "fori"))
        runs[acc] = cg.trajectory(gm, gdef, values, 4)
    for k, ((s0, _, r0), (s1, _, r1)) in enumerate(zip(runs["iqn-ils"], runs["iqn-imvj"])):
        assert not cg.bitwise_differences(s0, s1), k
        assert r0 == r1, k


# ---------------------------------------------------------------------------
# CPL-016: one pass reads what its mode documents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
def test_one_pass_reads_the_iterate_its_mode_documents(mode):
    """CPL-016: Gauss-Seidel reads the in-pass values of earlier nodes; Jacobi the previous iterate.

    A ring ``g0 -> g1 -> g2 -> g0`` of scalar relays (``alpha = 0``),
    swept in insertion order, one pass (``max_iterations=1``) from a known
    state, against the float64 sweep.
    """
    gdef = cg._cycle(3, 1, outside=False, leaves=(), alpha=0.0, beta=0.0)
    gains = {"g0": 0.5, "g1": -0.75, "g2": 0.25}
    bias = {"g0": 1.0, "g1": 2.0, "g2": -1.5}
    x0 = {"g0": 0.3, "g1": -0.6, "g2": 1.2}
    values = {nm: {"G": [np.array([[gains[nm]]], np.float32)],
                   "b": np.array([bias[nm]], np.float32),
                   "x0": np.array([x0[nm]], np.float32)} for nm in gdef.group_nodes}
    gm = _build(gdef, dict(max_iterations=1, iteration_mode=mode))
    assert [n for n in gm.schedule if n in gdef.group_nodes] == ["g0", "g1", "g2"]
    (state, _m, _r), = cg.trajectory(gm, gdef, values, 1)
    src = {"g0": "g2", "g1": "g0", "g2": "g1"}
    new = {}
    for nm in ("g0", "g1", "g2"):
        read = new[src[nm]] if (mode == "gauss-seidel" and src[nm] in new) else x0[src[nm]]
        new[nm] = gains[nm] * read + bias[nm]
    for nm, want in new.items():
        assert float(state[nm]["x"][0]) == pytest.approx(want, rel=1e-6), (nm, mode)


# ---------------------------------------------------------------------------
# CPL-025: a node downstream of a group reads it in the same step
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "CPL-025: a node downstream of a coupling group added before the group's members "
    "is scheduled before them and reads their previous-step output; pending fix"))
def test_a_node_downstream_of_a_group_reads_it_in_the_same_step():
    """CPL-025: only a cycle's back edges are staggered, whatever order the nodes were added in."""
    gdef = cg._cycle(2, 1, outside=True, leaves=())
    names = [nd.name for nd in gdef.nodes]
    assert names == ["drv", "g0", "g1", "sink"]
    values = cg.draw_values(np.random.default_rng(1), gdef, 0.5)
    group = dict(max_iterations=20, tolerance=1e-6)
    last = _build(gdef, group)
    first = _build(gdef, group, node_order=[names.index(n) for n in ("sink", "drv", "g0", "g1")])
    a = cg.trajectory(last, gdef, values, 2)
    b = cg.trajectory(first, gdef, values, 2)
    for k, ((sa, _, _), (sb, _, _)) in enumerate(zip(a, b)):
        np.testing.assert_array_equal(sb["sink"]["x"], sa["sink"]["x"],
                                      err_msg=f"step {k + 1}: schedules {last.schedule} "
                                              f"and {first.schedule}")


# ---------------------------------------------------------------------------
# CPL-031: forward mode through solver="fori"
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True, raises=pytest.fail.Exception, reason=(
    "CPL-031: forward-mode AD does work through solver='fori' (the docstring says it does "
    "not); pending a docs fix"))
def test_forward_mode_through_a_fori_group_is_refused():
    """CPL-031: "Forward-mode AD does not work through it" (``solver="fori"``)."""
    gdef = cg._cycle(2, 2, outside=False, leaves=())
    values = cg.draw_values(np.random.default_rng(2), gdef, 0.6)
    gm = _build(gdef, dict(max_iterations=10, tolerance=1e-6, solver="fori"))
    cg.set_initial(gm, values)
    params = _step_params(gm, values)
    state, ext = gm._state, gm._default_external_inputs()

    def f(p):
        return gm._compiled_step(state, ext, p)["g0"]["x"]

    with pytest.raises(Exception):
        jax.jvp(f, (params,), (jax.tree.map(jnp.ones_like, params),))


# ---------------------------------------------------------------------------
# CPL-035 / CPL-052: strict_convergence and waveform sweeps; CPL-122: the counts
# ---------------------------------------------------------------------------


def _springs(**group):
    """The sub-cycled spring pair of ``test_phases_5_7_8`` (timesteps 0.001 / 0.01)."""
    gm = GraphManager()
    for name, dt, x0 in (("fast", 0.001, 0.0), ("slow", 0.01, 3.0)):
        gm.add_node(SpringDamperNode(name=name, timestep=dt, stiffness=50.0,
                                     damping=1.0, mass=1.0, rest_length=1.0,
                                     initial_position=x0))
    gm.add_edge("fast", "slow", "position", "anchor_position")
    gm.add_edge("slow", "fast", "position", "anchor_position")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        gm.add_coupling_group(["fast", "slow"], subcycling=True, **group)
    gm.compile()
    return gm


def test_strict_convergence_checks_every_waveform_sweep():
    """CPL-035, CPL-052: the report says converged at the cap; the strict group raises.

    ``max_iterations=2``, ``waveform_iterations=3``, ``tolerance=1e-8``: the
    guide's table reads sweeps of 2 (no), 1 (yes), 1 (yes).
    """
    group = dict(max_iterations=2, waveform_iterations=3, tolerance=1e-8)
    gm = _springs(**group)
    gm.step()
    d = gm.coupling_diagnostics()["fast+slow"]
    assert d["iterations"] >= 2 and d["converged"], dict(d)
    strict = _springs(strict_convergence=True, **group)
    with pytest.raises(Exception, match="without converging"):
        strict.step()


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_every_sweep_at_its_cap_counts_exactly_n_times_the_cap(solver):
    """CPL-122: ``total_iterations`` is at most ``N * max_iterations`` and excludes the measuring pass.

    A rotation coupled to an identity relay never stops moving, so each of
    the three sweeps runs its two passes and then spends one more
    evaluation measuring its residual; ``g0`` sub-steps twice per pass.
    """
    theta = 0.3
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
                   np.float32)
    base = cg._cycle(2, 2, outside=False, leaves=(), alpha=0.0, beta=0.0)
    gdef = dataclasses.replace(base, nodes=(
        dataclasses.replace(base.nodes[0], timestep=0.5), base.nodes[1]))
    values = {"g0": {"G": [rot], "b": np.zeros(2, np.float32), "x0": np.zeros(2, np.float32)},
              "g1": {"G": [np.eye(2, dtype=np.float32)], "b": np.zeros(2, np.float32),
                     "x0": np.array([1.0, 0.0], np.float32)}}
    gm = _build(gdef, dict(subcycling=True, max_iterations=2, waveform_iterations=3,
                           tolerance=1e-6, solver=solver, diagnostics=solver == "fori"))
    (_state, _meta, rep), = cg.trajectory(gm, gdef, values, 1)
    assert (rep["iterations"], rep["total_iterations"]) == (2, 6), rep
    assert not rep["converged"]
    d = gm.coupling_diagnostics()[gdef.key]
    assert (d["iterations"], d["total_iterations"]) == (2, 6), dict(d)


# ---------------------------------------------------------------------------
# CPL-053: unconverged means at the cap, or not finite
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _triangle(acceleration, mode, solver, cap):
    gm = _build(cg.TRIANGLE, _knobs(acceleration, iteration_mode=mode, solver=solver,
                                    max_iterations=cap, tolerance=1e-5,
                                    diagnostics=solver == "fori"))
    return gm


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("acceleration", ACCELERATIONS)
def test_unconverged_means_at_the_cap_or_not_finite(acceleration, mode):
    """CPL-053: ``converged=False`` only beside ``iterations == max_iterations`` or a non-finite state."""
    gdef = cg.TRIANGLE
    cap = 12
    seen = set()
    for solver in ("ift", "fori"):
        gm = _triangle(acceleration, mode, solver, cap)
        for seed, rho in ((0, 0.3), (2, 1.3)):
            values = cg.draw_values(np.random.default_rng(seed), gdef, rho, nonnormal=True)
            for state, _meta, rep in cg.trajectory(gm, gdef, values, 2):
                finite = all(np.all(np.isfinite(state[nm]["x"])) for nm in gdef.group_nodes)
                seen.add(rep["converged"])
                if not rep["converged"]:
                    assert rep["iterations"] == cap or not finite, (solver, seed, rep)
                assert rep["iterations"] <= cap
    assert True in seen, "the draws must exercise a converged verdict"
    if not acceleration.startswith("iqn"):
        # A quasi-Newton root solver may converge the divergent draw too.
        assert False in seen, "the draws must exercise an unconverged verdict"


# ---------------------------------------------------------------------------
# CPL-066: the secant history
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cap", [2, 7])
def test_the_secant_history_is_max_iterations_minus_one_columns(cap):
    """CPL-066: "IQN allocates max_iterations - 1 secant columns"."""
    gdef = cg.TRIANGLE
    gm = _build(gdef, dict(acceleration="iqn-imvj", jacobian_reuse=1, max_iterations=cap,
                           tolerance=1e-5))
    values = cg.draw_values(np.random.default_rng(0), gdef, 0.9)
    (_s, meta, _r), = cg.trajectory(gm, gdef, values, 1)
    n_iface = gdef.n * len(gdef.group_nodes)        # every internal edge reads x
    assert meta["V"].shape == (n_iface, cap - 1)
    assert meta["W"].shape == (n_iface, cap - 1)


# ---------------------------------------------------------------------------
# CPL-072: 16-bit groups under every accelerator
# ---------------------------------------------------------------------------


class _Lin16(SimulationNode):
    """``x <- G @ u + b`` in a fixed dtype."""

    def __init__(self, name, G, b, dtype):
        super().__init__(name, 1.0, G=jnp.asarray(G, dtype), b=jnp.asarray(b, dtype))
        self._dtype = dtype

    def initial_state(self):
        return {"x": jnp.zeros(2, self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=self._dtype,
                                       default=jnp.zeros(2, self._dtype))}

    def update(self, state, boundary_inputs, dt):
        return {"x": (self.params["G"] @ boundary_inputs["u"] + self.params["b"]).astype(self._dtype)}


# The fori rows raised a carry TypeError until the accelerators' returns were
# cast to their carries' dtypes (MADD-ANO-109).
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float16], ids=["bfloat16", "float16"])
@pytest.mark.parametrize("acceleration,solver", [
    *[(acc, "ift") for acc in ACCELERATIONS],
    ("none", "fori"), ("fixed", "fori"),
    ("aitken", "fori"), ("iqn-ils", "fori"), ("iqn-imvj", "fori"),
])
def test_a_sixteen_bit_group_steps_under_every_accelerator(dtype, acceleration, solver):
    """CPL-072: a bfloat16 / float16 group steps and scans and keeps its dtype."""
    gm = GraphManager()
    gm.add_node(_Lin16("a", np.eye(2) * 0.6, [1.0, 2.0], dtype))
    gm.add_node(_Lin16("b", np.eye(2) * 0.9, [0.0, 0.0], dtype))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        gm.add_coupling_group(["a", "b"], **_knobs(acceleration, solver=solver,
                                                   max_iterations=20, tolerance=1e-2,
                                                   diagnostics=solver == "fori"))
    gm.compile()
    gm.step()
    gm.run_scan(2)
    assert gm.get_node_state("a")["x"].dtype == dtype
    assert gm.coupling_diagnostics()["a+b"]["iterations"] >= 1


# ---------------------------------------------------------------------------
# CPL-078: Jacobi and the build order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("acceleration", ["none", "fixed"])
def test_jacobi_states_do_not_depend_on_the_build_order_under_a_constant_iterator(
        acceleration, solver):
    """CPL-078 (the reading that holds): Jacobi's returned states under every build order.

    Under a constant iterator every member reads the stored previous
    iterate, so the members' order cannot reach the states; the norm sums
    the members in the schedule's order, so a residual can move by an ulp,
    and these draws keep the verdict away from the threshold.  (Under
    ``"aitken"`` and the IQN pair the states differ in their last bits --
    the accelerator's dot products follow the schedule -- which is the
    row's ambiguous half.)
    """
    gdef = cg._cycle(3, 2, chords=((0, 2),), outside=False, leaves=())
    values = cg.draw_values(np.random.default_rng(5), gdef, 0.9, nonnormal=True)
    group = _knobs(acceleration, iteration_mode="jacobi", solver=solver, max_iterations=12,
                   tolerance=1e-5, diagnostics=solver == "fori")
    base = cg.trajectory(_build(gdef, group), gdef, values, 3)
    for order in ([1, 2, 0], [2, 1, 0]):
        gm = _build(gdef, group, node_order=order)
        assert [n for n in gm.schedule] == [gdef.nodes[i].name for i in order]
        run = cg.trajectory(gm, gdef, values, 3)
        for k, ((s0, _, r0), (s1, _, r1)) in enumerate(zip(base, run)):
            assert not cg.bitwise_differences(s0, s1), (order, k)
            assert (r0["iterations"], r0["converged"]) == (r1["iterations"], r1["converged"])


# ---------------------------------------------------------------------------
# CPL-083: the report's derived keys follow its slots
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("acceleration,cap,solver", [
    ("none", 12, "ift"), ("fixed", 12, "fori"), ("aitken", 12, "ift"),
    ("iqn-ils", 12, "ift"), ("none", 1, "ift"),
])
def test_the_report_derives_every_value_from_its_slots(acceleration, cap, solver):
    """CPL-083: amplification, error_estimate, gradient_error_estimate and precision_limited."""
    gdef = cg.TRIANGLE
    gm = _build(gdef, _knobs(acceleration, max_iterations=cap, tolerance=1e-5, solver=solver,
                             diagnostics=solver == "fori"))
    values = cg.draw_values(np.random.default_rng(7), gdef, 0.8)
    group = gm._coupling_groups[0]
    threshold, scale = convergence_criterion(group)
    cg.set_initial(gm, values)
    params = _step_params(gm, values)
    for _ in range(3):
        gm.step(params=params)
        d = gm.coupling_diagnostics()[gdef.key]
        meta = cg.group_meta(gm, gdef.key)
        res, amp = meta["residual"], meta["amplification"]
        valid = float(amp) >= 1.0
        assert d["ratio_usable"] == valid
        assert (math.isnan(d["amplification"]) if not valid else d["amplification"] == float(amp))
        est = reported_error_estimate(res, amp, scale)
        assert d["error_estimate"] == est
        if not valid:
            assert d["error_estimate"] == float(res)
        assert d["gradient_error_estimate"] == (est if valid else math.inf)
        evaluations, _declared = _group_evaluations(group, gm._nodes, gm.schedule, gm._edges)
        internal = [e for e in gm._edges if e.source_node in group.nodes
                    and e.target_node in group.nodes]
        floor = float(residual_precision_floor(gm._state, sorted(group.nodes),
                                               group.convergence_norm, group.atol,
                                               group.rtol, internal, evaluations=evaluations))
        assert d["precision_limited"] == (floor > 0.0 and math.isfinite(d["residual"])
                                          and d["residual"] <= floor)
    if cap == 1:
        assert not d["ratio_usable"] and d["gradient_error_estimate"] == math.inf


def test_a_group_whose_norm_reads_nothing_is_not_precision_limited():
    """CPL-083, CPL-011: every field dead-banded: residual 0, converged after one pass, floor 0."""
    gdef = cg._cycle(2, 1, outside=False, leaves=())
    gm = _build(gdef, dict(max_iterations=10, tolerance=1e-6, atol=1e6, diagnostics=True))
    values = cg.draw_values(np.random.default_rng(4), gdef, 0.99)
    cg.set_initial(gm, values)
    gm.step(params=_step_params(gm, values))
    d = gm.coupling_diagnostics()[gdef.key]
    assert d["residual"] == 0.0 and d["converged"], dict(d)
    assert not d["precision_limited"], dict(d)


# ---------------------------------------------------------------------------
# CPL-086: the report is the last step's, judged under the graph it ran
# ---------------------------------------------------------------------------


def test_an_edge_added_after_the_step_does_not_rejudge_its_floor():
    """CPL-086: an edge added after a step, before any recompile, does not move that step's bound.

    Three scalar relays around a hub ``g0`` (``g0 <-> g1``, ``g0 <-> g2``):
    under Gauss-Seidel the longest same-pass chain is two.  ``g1 -> g2``
    added afterwards would make it three, but the step did not run with it.
    The floor is added to the residual at any distance, so the bound moved
    by it whether or not the group had stalled (on the base tree it read
    0.000506 before the edit and 0.000547 after); the report now reads the
    count the step measured or ``compile()`` snapshotted.  This test's
    first edition also asserted ``precision_limited``, which the 200-pass
    run never reaches (residual 0.10) -- the assertion that failed was
    that premise, on the base tree and the fixed one alike.
    """
    nodes = (cg.NodeDef("g0", 2), cg.NodeDef("g1", 1), cg.NodeDef("g2", 2))
    edges = (cg.EdgeDef("g1", "g0", 0), cg.EdgeDef("g2", "g0", 1),
             cg.EdgeDef("g0", "g1", 0), cg.EdgeDef("g0", "g2", 0))
    gdef = cg.GraphDef(n=1, nodes=nodes, edges=edges, group_nodes=("g0", "g1", "g2"))
    gm = _build(gdef, dict(max_iterations=200, tolerance=1e-30, diagnostics=True))
    values = cg.draw_values(np.random.default_rng(9), gdef, 0.99)
    cg.set_initial(gm, values)
    gm.step(params=_step_params(gm, values))
    before = dict(gm.coupling_diagnostics()[gdef.key])
    assert math.isfinite(before["spectral_error_bound"]), before
    gm.add_edge("g1", "g2", "x", "u1")                  # no recompile
    after = dict(gm.coupling_diagnostics()[gdef.key])
    assert after["spectral_error_bound"] == before["spectral_error_bound"]


# ---------------------------------------------------------------------------
# CPL-088: a usable bound on a stalled sub-cycled or multi-rate group
# ---------------------------------------------------------------------------


def _exact_subcycled(values, alpha, divider, pre):
    """Fixed point of ``a <- divider sub-steps of x <- alpha x + G_a b + b_a``, ``b <- G_b a + b_b``."""
    Ga = np.asarray(values["g0"]["G"][0], np.float32).astype(np.float64)
    Gb = np.asarray(values["g1"]["G"][0], np.float32).astype(np.float64)
    ba = np.asarray(values["g0"]["b"], np.float32).astype(np.float64)
    bb = np.asarray(values["g1"]["b"], np.float32).astype(np.float64)
    a32 = np.float64(np.float32(alpha))
    geo = sum(a32 ** j for j in range(divider))
    n = Ga.shape[0]
    # a = a32**d pre + geo (Ga b + ba);  b = Gb a + bb
    M = np.block([[np.zeros((n, n)), geo * Ga], [Gb, np.zeros((n, n))]])
    c = np.concatenate([a32 ** divider * pre + geo * ba, bb])
    x = np.linalg.solve(np.eye(2 * n) - M, c)
    return {"g0": x[:n], "g1": x[n:]}


@pytest.mark.parametrize("case", ["subcycled", "multirate"])
def test_a_usable_bound_holds_on_a_stalled_subcycled_or_multirate_group(case):
    """CPL-088: ``spectral_error_bound >= ||x - x*||`` where usable, on the two schedules the bound's floor counts differently.

    A two-relay group contracting at 0.99, run to its float32 stall
    (``tolerance=1e-30``).  *subcycled*: ``g0`` sub-steps four times per
    pass with ``alpha = 0.5`` (``x <- 0.5 x + ...``, a composite map the
    floor counts as four evaluations).  *multirate*: the group at twice a
    driver's timestep, read on the step it fires.
    """
    if case == "subcycled":
        base = cg._cycle(2, 2, outside=False, leaves=(), alpha=0.0, beta=0.0)
        gdef = dataclasses.replace(base, nodes=(
            dataclasses.replace(base.nodes[0], alpha=0.5, timestep=0.25), base.nodes[1]))
        group = dict(subcycling=True, boundary_interpolation="constant")
    else:
        base = cg._cycle(2, 2, outside=True, leaves=(), alpha=0.0, beta=0.0)
        gdef = dataclasses.replace(base, nodes=tuple(
            dataclasses.replace(nd, timestep=0.5 if nd.name in ("drv", "sink") else 1.0,
                                alpha=1.0 if nd.name == "drv" else nd.alpha)
            for nd in base.nodes))
        group = {}
    gm = _build(gdef, dict(group, max_iterations=400, tolerance=1e-30, diagnostics=True))
    # The draw fixes the Jacobi rate of the gains; the sub-cycled member's
    # four sub-steps multiply its gain by 1 + 0.5 + 0.25 + 0.125.
    rho = 0.99 / math.sqrt(1.875) if case == "subcycled" else 0.99
    values = cg.draw_values(np.random.default_rng(21), gdef, rho)
    cg.set_initial(gm, values)
    pre = {nm: {"x": np.asarray(gm.get_node_state(nm)["x"], np.float64)}
           for nm in gdef.group_nodes}
    gm.step(params=_step_params(gm, values))
    d = gm.coupling_diagnostics()[gdef.key]
    got = cg.snapshot(gm)
    if case == "subcycled":
        exact = _exact_subcycled(values, 0.5, 4, pre["g0"]["x"])
    else:
        # The driver is scheduled first and fires on step 0: the group
        # reads its new value through a forward edge.
        drv_new = np.asarray(got["drv"]["x"], np.float32)
        exact = cg.exact_fixed_point(gdef, values, pre, {("g0", 1): drv_new}, 1.0)
    dist = cg.group_distance(gm, gdef, dict(group), got, exact)
    assert d["spectral_usable"], dict(d)
    assert d["spectral_error_bound"] >= dist, (dict(d), dist)


# ---------------------------------------------------------------------------
# CPL-110: the report flags an unsettled spectral bound only where one exists
# ---------------------------------------------------------------------------


def test_the_report_flags_an_unsettled_spectral_bound_only_where_one_was_computed():
    """CPL-110: twelve interface scalars against eight Krylov steps; diagnostics off reports none."""
    gdef = cg._cycle(2, 6, outside=False, leaves=())
    values = cg.draw_values(np.random.default_rng(6), gdef, 0.9)
    flags = {}
    for diagnostics in (True, False):
        # Jacobi: each member's update reads the other's previous iterate,
        # so the pass's Jacobian has rank twelve (Gauss-Seidel's has six).
        gm = _build(gdef, dict(max_iterations=30, tolerance=1e-5, iteration_mode="jacobi",
                               diagnostics=diagnostics))
        cg.set_initial(gm, values)
        gm.step(params=_step_params(gm, values))
        (row,) = gm.coupling_report()
        flags[diagnostics] = [f for f in row["flags"] if f.startswith("spectral_usable")]
        assert row["spectral_usable"] is False
    assert flags[True], "an unsettled bound was computed and not flagged"
    assert not flags[False], "no bound was computed, yet it was flagged"


# ---------------------------------------------------------------------------
# CPL-134: the two adaptive steppers at dt_min
# ---------------------------------------------------------------------------


# run_adaptive kept a rejected attempt when the shrunk step would fall to
# dt_min until both steppers took one rule (adaptive.step_decision; MADD-ANO-111).
def test_both_adaptive_steppers_accept_the_same_steps_at_dt_min():
    """CPL-134: "Like run_adaptive but fully JIT-compiled" -- the same accepted steps at ``dt_min``."""
    def graph():
        gm = GraphManager()
        gm.add_node(SpringDamperNode("a", 0.01, stiffness=30000.0, damping=50.0,
                                     initial_position=-1.0))
        gm.add_node(SpringDamperNode("b", 0.01, stiffness=30000.0, damping=50.0,
                                     initial_position=1.0))
        gm.add_edge("a", "b", "position", "anchor_position")
        gm.add_edge("b", "a", "position", "anchor_position")
        gm.add_coupling_group(["a", "b"], max_iterations=30, tolerance=1e-5)
        gm.compile()
        return gm

    kw = dict(dt_initial=0.01, dt_max=0.01, dt_min=0.005, atol=1e-3, rtol=1e-3)
    with pytest.warns(UserWarning, match="hit dt_min"):
        host, info = graph().run_adaptive(0.01, **kw)
    scan, _hist, sinfo = graph().run_adaptive_scan(0.01, max_steps=8, **kw)
    assert int(sinfo["n_steps"]) == info["n_steps"], (int(sinfo["n_steps"]), info)
    for nm in ("a", "b"):
        np.testing.assert_array_equal(np.asarray(scan[nm]["position"]),
                                      np.asarray(host[nm]["position"]))


# ---------------------------------------------------------------------------
# CPL-141: the IFT gradient of an affine group under every acceleration
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _affine_gradient(acceleration):
    """``d(sum of the group's x*)/db`` per node, through one IFT step."""
    gdef = _AFFINE_GDEF
    values = cg.draw_values(np.random.default_rng(8), gdef, 0.8)
    gm = _build(gdef, _knobs(acceleration, max_iterations=40, tolerance=1e-6))
    cg.set_initial(gm, values)
    params = _step_params(gm, values)
    state, ext = gm._state, gm._default_external_inputs()

    def loss(p):
        out = gm._compiled_step(state, ext, p)
        return sum(jnp.sum(out[nm]["x"]) for nm in gdef.group_nodes)

    g = jax.grad(loss)(params)["nodes"]
    return {nm: np.asarray(g[nm]["b"]) for nm in gdef.group_nodes}


_AFFINE_GDEF = cg._cycle(3, 2, chords=((0, 2),), outside=False, leaves=(), beta=0.0)


@pytest.mark.parametrize("acceleration", ACCELERATIONS[1:])
def test_the_ift_gradient_of_an_affine_group_is_the_same_under_every_acceleration(acceleration):
    """CPL-141 (the reading that holds): on an affine map the rule's inputs do not depend on the iterate."""
    got, ref = _affine_gradient(acceleration), _affine_gradient("none")
    for nm in _AFFINE_GDEF.group_nodes:
        np.testing.assert_array_equal(got[nm], ref[nm], err_msg=f"{acceleration}, d/db of {nm}")


# ---------------------------------------------------------------------------
# CPL-143: the adjoint and the size of the cotangent
# ---------------------------------------------------------------------------


# 1e-12 read exactly 0.0 until the solve dropped its absolute 1e-8 (MADD-ANO-106).
@pytest.mark.parametrize("scale", [1.0, 1e-12])
def test_the_ift_gradient_is_scale_equivariant_in_the_cotangent(scale):
    """CPL-143: ``d(s * L)/dtheta = s * dL/dtheta``: the adjoint is relative to its right-hand side.

    ``a <- 0.6 b + c``, ``b <- 0.9 a``: ``d(sum a*)/dc = 1 / (1 - 0.54)``.
    """
    gdef = cg._cycle(2, 2, outside=False, leaves=(), alpha=0.0, beta=0.0)
    values = {"g0": {"G": [np.eye(2, dtype=np.float32) * 0.6], "b": np.array([1.0, 2.0], np.float32),
                     "x0": np.zeros(2, np.float32)},
              "g1": {"G": [np.eye(2, dtype=np.float32) * 0.9], "b": np.zeros(2, np.float32),
                     "x0": np.zeros(2, np.float32)}}
    gm = _build(gdef, dict(max_iterations=200, tolerance=1e-7))
    cg.set_initial(gm, values)
    params = _step_params(gm, values)
    state, ext = gm._state, gm._default_external_inputs()

    def loss(p):
        return jnp.float32(scale) * jnp.sum(gm._compiled_step(state, ext, p)["g0"]["x"])

    g = np.asarray(jax.grad(loss)(params)["nodes"]["g0"]["b"], np.float64) / scale
    np.testing.assert_allclose(g, 1.0 / (1.0 - 0.54), rtol=1e-4)


# ---------------------------------------------------------------------------
# CPL-144: a Hessian through the IFT rule
# ---------------------------------------------------------------------------


def test_jax_hessian_runs_through_an_ift_step():
    """CPL-144: ``jax.hessian`` through a coupled step, against the analytic Hessian.

    ``a <- 0.6 b + c_a``, ``b <- 0.5 a + c_b``: ``x* = (I - M)^-1 c``, so the
    loss ``|x*|^2`` has the Hessian ``2 B^T B`` in ``c``, ``B = (I - M)^-1``.
    The smallest coupled group, so the second-order program compiles on
    every push (the spring-pair version is slow-marked).
    """
    gdef = cg._cycle(2, 1, outside=False, leaves=(), alpha=0.0, beta=0.0)
    values = {"g0": {"G": [np.array([[0.6]], np.float32)], "b": np.array([1.0], np.float32),
                     "x0": np.zeros(1, np.float32)},
              "g1": {"G": [np.array([[0.5]], np.float32)], "b": np.array([0.5], np.float32),
                     "x0": np.zeros(1, np.float32)}}
    gm = _build(gdef, dict(max_iterations=100, tolerance=1e-7))
    cg.set_initial(gm, values)
    base = _step_params(gm, values)
    state, ext = gm._state, gm._default_external_inputs()

    def loss(c):
        nodes = {**base["nodes"], "g0": {**base["nodes"]["g0"], "b": c[:1]},
                 "g1": {**base["nodes"]["g1"], "b": c[1:]}}
        out = gm._compiled_step(state, ext, {**base, "nodes": nodes})
        return jnp.sum(out["g0"]["x"] ** 2) + jnp.sum(out["g1"]["x"] ** 2)

    H = np.asarray(jax.hessian(loss)(jnp.asarray([1.0, 0.5], F32)), np.float64)
    B = np.linalg.inv(np.eye(2) - np.array([[0.0, 0.6], [0.5, 0.0]]))
    np.testing.assert_allclose(H, 2.0 * B.T @ B, rtol=1e-4)


# ---------------------------------------------------------------------------
# CPL-150: the profiler's cap check with waveform sweeps
# ---------------------------------------------------------------------------


def test_the_profilers_cap_check_sees_a_capped_first_sweep():
    """CPL-150: ``at_cap_fraction`` counts a step whose first sweep ran out ("when some sweep did")."""
    gm = _springs(max_iterations=2, waveform_iterations=3, tolerance=1e-8)
    # ``n_warmup=0``: the statistics pass is the trajectory's first step,
    # whose sweeps the guide's table records as 2 (capped), 1 and 1.
    report = profile_graph(gm, n_steps=2, n_warmup=0, measure_coupling=False, counts=False,
                           n_stat_steps=1)
    st = report.coupling_iter_stats["fast+slow"]
    assert st["sweeps"] == 3 and st["cap"] == 2
    assert st["at_cap_fraction"] == 1.0, st
    assert st["total_mean"] > st["mean"], st


# ---------------------------------------------------------------------------
# CPL-162: the sysid mask where every window converges
# ---------------------------------------------------------------------------


def test_the_mask_changes_nothing_where_every_window_converges():
    """CPL-162: with every window converged the masked loss and gradient are the unmasked ones, bitwise."""
    gdef = cg._cycle(2, 1, outside=False, leaves=())
    values = cg.draw_values(np.random.default_rng(12), gdef, 0.5)
    gm = _build(gdef, dict(max_iterations=40, tolerance=1e-6))
    cg.set_initial(gm, values)
    params = _step_params(gm, values)
    init = {nm: dict(gm.get_node_state(nm)) for nm in gdef.group_nodes}
    _final, hist = gm.run_scan_with_history(4, params=params)
    obs = observations_from_history(init, {nm: hist[nm] for nm in gdef.group_nodes})
    cg.set_initial(gm, values)

    def loss(p, mask):
        return windowed_loss(gm, p, obs, obs_fn=lambda s: s["g0"]["x"], window=2,
                             mask_unconverged=mask)

    off = jax.value_and_grad(lambda p: loss(p, False))(params)
    on = jax.value_and_grad(lambda p: loss(p, True))(params)
    assert float(on[0]) == float(off[0])
    for a, b in zip(jax.tree.leaves(on[1]), jax.tree.leaves(off[1])):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
