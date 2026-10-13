"""Metamorphic invariances of a coupled step: names, build order, an identity relay.

Three changes to how a coupled graph is *written* that must not change what
it *computes* (rows CPL-180 to CPL-183 of
``docs/validation/coupling_claims.yaml``):

**Renaming every node.**  A group is keyed by its sorted node names joined
with ``"+"``, and nothing else about a name reaches the solve: the sweep
follows the compiled schedule, which follows insertion order and topology,
never the alphabet.  So a graph built with every node renamed -- including
names whose sorted order differs from the schedule, and names that sort
before ``"_meta"`` -- must return bit-identical states, verdicts and
``coupling_diagnostics()`` reports, up to the name keys.  Measured to hold
for both solvers, both schedules, every acceleration and every norm, the
spectral and gradient-bound keys included.

**The build order.**  The order of the ``add_edge`` calls, and of the
``add_node`` calls for every node outside the group, must not change the
result either -- a node downstream of the group included, which used to
read the group's previous-step output when it was added before the
group's members (CPL-025).  One order is *not* free, by design, and the
strategies hold it fixed: the relative order of a group's members, which
a Gauss-Seidel sweep follows on purpose (CPL-077; under Jacobi the
members' order still reaches the states through the accelerators' dot
products, CPL-078).  Every norm is drawn: the interface norm used to sum
its terms in the order the edges were added (CPL-182).  Three or more
additive edges into one input still sum in insertion order (float
addition does not associate; documented in the coupling guide); the
structures here carry none.

**An identity relay on an internal edge** (``x <- I u``, exact in float32)
is the same coupled system with one more copy of a field, so it cannot
move the fixed point.  Three statements, from strongest to weakest:

* under Gauss-Seidel with the relay added straight after its source and a
  constant iterator (``"none"``, ``"fixed"``), every other node's iterate
  is bit-identical pass for pass: the relay is recomputed from its source
  before its reader reads it, in the same pass.  With a criterion only
  exact stationarity meets, both graphs run the same passes and return the
  same bits, the relay holding a copy of its source;
* under every schedule and acceleration, run to the float32 stall, the two
  graphs' states differ by rounding only.  The derived bound: each stall
  is within ``A * PRECISION_FLOOR_ULPS * eps * max|x|`` of the exact fixed
  point (the floor model's evaluation error, amplified by
  ``A = ||(I - M)^-1||_inf`` of the group's float64 coupling operator), so
  the two are within twice that.  Measured: at most 0.26 of
  ``A eps max|x|`` (jaxlib 0.11.0, CPU);
* away from the threshold the verdict is the same: a convergent group
  converges with and without the relay.

Per-push tests fix the structure and the build orders and draw the values;
the slow siblings draw the structure, the configuration and the
permutation too.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.core.coupling.acceleration import PRECISION_FLOOR_ULPS
from maddening.core.graph_manager import GraphManager
from tests.conftest import EXAMPLES_COSTLY
from tests.property import coupled_graphs as cg

EPS32 = float(np.finfo(np.float32).eps)
_STEPS = 2

# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------

#: Two drivers, a three-node group (a cycle and a chord) and two sinks.
_NODES = (
    cg.NodeDef("d0", 0, alpha=1.0, beta=1.0, leaves=("count",)),
    cg.NodeDef("d1", 0, alpha=0.5, beta=-0.5),
    cg.NodeDef("g0", 2, alpha=0.5, beta=1.0, leaves=cg.LEAF_POOL),
    cg.NodeDef("g1", 2, alpha=-0.25, beta=0.0, leaves=("flag", "unread")),
    cg.NodeDef("g2", 2, alpha=0.0, beta=-0.5, leaves=("tag",)),
    cg.NodeDef("s0", 1, alpha=0.5, leaves=("tag",)),
    cg.NodeDef("s1", 1, alpha=0.25),
)
_EDGES = (
    cg.EdgeDef("g0", "g1", 0), cg.EdgeDef("g1", "g2", 0), cg.EdgeDef("g2", "g0", 0),
    cg.EdgeDef("g0", "g2", 1), cg.EdgeDef("d0", "g0", 1), cg.EdgeDef("d1", "g1", 1),
    cg.EdgeDef("g2", "s0", 0), cg.EdgeDef("g1", "s1", 0),
)
STRUCT = cg.GraphDef(n=2, nodes=_NODES, edges=_EDGES, group_nodes=("g0", "g1", "g2"))

#: Names whose sorted order is not the schedule's, one of them sorting
#: before ``"_meta"`` (upper case) -- the state dict a scan returns has
#: sorted keys.
RENAME = {"d0": "y_drive", "d1": "b_drive", "g0": "zeta", "g1": "Alpha", "g2": "mu",
          "s0": "a_sink", "s1": "z_sink"}

#: A build order that keeps the members' relative order and moves
#: everything else, a sink ahead of the members included; the edges
#: reversed.
NODE_ORDER = ("s0", "d1", "g0", "d0", "g1", "g2", "s1")
EDGE_ORDER = tuple(reversed(range(len(_EDGES))))


def _knobs(acceleration, **extra):
    group = dict(acceleration=acceleration, **extra)
    if acceleration == "fixed":
        group.setdefault("relaxation", 0.8)
    if acceleration == "iqn-imvj":
        group.setdefault("jacobian_reuse", 2)
    return group


#: The per-push configurations: every acceleration once, the solvers,
#: schedules and norms rotated through them, and a predictor.
CASES = {
    "none-gs-ift-l2": _knobs("none", tolerance=1e-5),
    "aitken-jacobi-fori-mixed": _knobs("aitken", iteration_mode="jacobi", solver="fori",
                                       diagnostics=True, convergence_norm="mixed", rtol=1e-4),
    "fixed-gs-ift-l2-predictor": _knobs("fixed", predictor="linear", tolerance=1e-5),
    "iqn-ils-jacobi-ift-mixed": _knobs("iqn-ils", iteration_mode="jacobi",
                                       convergence_norm="mixed", rtol=1e-4),
    "iqn-imvj-gs-fori-l2": _knobs("iqn-imvj", solver="fori", diagnostics=True,
                                  tolerance=1e-5),
    "fixed-gs-ift-interface": _knobs("fixed", convergence_norm="interface", rtol=1e-4),
}

#: The renaming oracle also takes a diagnostics group, whose spectral and
#: gradient-bound keys must not move either -- on a two-member group, the
#: diagnostics' compile being the cost.
RENAME_CASES = {
    **CASES,
    "none-gs-ift-l2-diag": _knobs("none", diagnostics=True, tolerance=1e-5),
}

SMALL = cg._cycle(2, 2, outside=True, leaves=("count",))
SMALL_RENAME = {"drv": "Zdrive", "g0": "zz", "g1": "aa", "sink": "m_sink"}


def _structure(case):
    """``(structure, renaming)`` a renaming case runs on."""
    if RENAME_CASES[case].get("diagnostics") and RENAME_CASES[case].get("solver") != "fori":
        return SMALL, SMALL_RENAME
    return STRUCT, RENAME


# ---------------------------------------------------------------------------
# Building and running
# ---------------------------------------------------------------------------


def build(gdef, group, *, node_order=None, edge_order=None):
    """*gdef* with its nodes and edges added in the given orders (names / indices)."""
    gm = GraphManager()
    by_name = {nd.name: nd for nd in gdef.nodes}
    nodes = gdef.nodes if node_order is None else [by_name[n] for n in node_order]
    for nd in nodes:
        gm.add_node(cg.make_node(nd, gdef.n))
    edges = gdef.edges if edge_order is None else [gdef.edges[i] for i in edge_order]
    for e in edges:
        gm.add_edge(e.src, e.dst, e.field, f"u{e.port}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        gm.add_coupling_group(list(gdef.group_nodes), **cg.live_knobs(group))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*multi-rate.*")
        gm.compile()
    return gm


def run(gm, gdef, values, steps=_STEPS):
    """``[(state, report)]`` after each step, from the drawn initial state."""
    cg.set_initial(gm, values)
    params = cg.params_for(gm, values)
    out = []
    for _ in range(steps):
        gm.step(params=params)
        report = gm.coupling_diagnostics().get(gdef.key)
        out.append((cg.snapshot(gm), None if report is None else dict(report)))
    return out


def rename(gdef, mapping):
    nodes = tuple(dataclasses.replace(nd, name=mapping[nd.name]) for nd in gdef.nodes)
    edges = tuple(dataclasses.replace(e, src=mapping[e.src], dst=mapping[e.dst])
                  for e in gdef.edges)
    return cg.GraphDef(n=gdef.n, nodes=nodes, edges=edges,
                       group_nodes=tuple(mapping[g] for g in gdef.group_nodes))


def report_differences(a, b):
    """Report keys whose values differ, NaN equal to NaN."""
    if a is None or b is None:
        return [] if a is b else ["<missing report>"]
    out = []
    for k in sorted(set(a) | set(b)):
        x, y = a.get(k), b.get(k)
        same = x == y or (isinstance(x, float) and isinstance(y, float)
                          and math.isnan(x) and math.isnan(y))
        if not same:
            out.append(f"{k}: {x!r} != {y!r}")
    return out


def assert_same_runs(a, b, mapping=None, nodes=None):
    """States bit for bit (renamed back) and reports value for value."""
    mapping = mapping or {}
    for k, ((sa, ra), (sb, rb)) in enumerate(zip(a, b)):
        sb = {inv: sb[mapping.get(inv, inv)] for inv in (nodes or sa)}
        sa = {n: sa[n] for n in (nodes or sa)}
        diff = cg.bitwise_differences(sa, sb)
        assert not diff, f"step {k + 1}: states differ in {diff}"
        rdiff = report_differences(ra, rb)
        assert not rdiff, f"step {k + 1}: reports differ: {rdiff}"


_VALUES = cg.drawn_values


# ---------------------------------------------------------------------------
# CPL-180: renaming every node
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _renamed_pair(case):
    group = RENAME_CASES[case]
    gdef, mapping = _structure(case)
    return build(gdef, group), build(rename(gdef, mapping), group)


@pytest.mark.parametrize("case", sorted(RENAME_CASES))
# Costly tier: two compiled graphs, two steps each, per example.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_renaming_every_node_changes_nothing_but_the_keys(case, data):
    """CPL-180, per push; slow sibling :func:`test_renaming_is_invariant_on_generated_graphs`."""
    gm, gm_renamed = _renamed_pair(case)
    gdef, mapping = _structure(case)
    values = data.draw(_VALUES(gdef))
    renamed = rename(gdef, mapping)
    a = run(gm, gdef, values)
    b = run(gm_renamed, renamed, {mapping[k]: v for k, v in values.items()})
    assert_same_runs(a, b, mapping)


def _rename_strategy(gdef):
    pool = ["Q", "zz", "a", "_x", "m+n", "beta", "Z9", "omega", "k_1", "ALPHA", "q.r"]
    names = [nd.name for nd in gdef.nodes]
    return st.permutations(pool).map(lambda p: dict(zip(names, p[:len(names)])))


# Per push: tests/property/test_coupling_invariances.py::test_renaming_every_node_changes_nothing_but_the_keys
@pytest.mark.slow
# Costly tier: compiles two graphs per example.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(data=st.data())
def test_renaming_is_invariant_on_generated_graphs(data):
    """CPL-180 with the structure, the configuration and the new names drawn."""
    gdef = data.draw(cg.graph_defs(max_group=4))
    group = data.draw(cg.group_configs(gdef))
    mapping = data.draw(_rename_strategy(gdef))
    values = data.draw(_VALUES(gdef))
    renamed = rename(gdef, mapping)
    group = dict(group, diagnostics=data.draw(st.booleans()))
    a = run(build(gdef, group), gdef, values)
    b = run(build(renamed, group), renamed, {mapping[k]: v for k, v in values.items()})
    assert_same_runs(a, b, mapping)


# ---------------------------------------------------------------------------
# CPL-181 / CPL-182: the build order
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _ordered_pair(case):
    group = CASES[case]
    return build(STRUCT, group), build(STRUCT, group, node_order=NODE_ORDER,
                                       edge_order=EDGE_ORDER)


@pytest.mark.parametrize("case", sorted(CASES))
# Costly tier: two compiled graphs, two steps each, per example.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_the_build_order_changes_nothing(case, data):
    """CPL-181, per push; slow sibling :func:`test_the_build_order_is_invariant_on_generated_graphs`."""
    gm, gm_reordered = _ordered_pair(case)
    assert gm.schedule != gm_reordered.schedule, "the reordering must reach the schedule"
    values = data.draw(_VALUES(STRUCT))
    assert_same_runs(run(gm, STRUCT, values), run(gm_reordered, STRUCT, values))


def _order_strategy(gdef):
    """Any node order that keeps the group's members in their relative order."""
    members = list(gdef.group_nodes)
    names = [nd.name for nd in gdef.nodes]

    @st.composite
    def order(draw):
        drawn = draw(st.permutations(names))
        slots = iter(members)
        return tuple(next(slots) if n in members else n for n in drawn)

    return order()


# Per push: tests/property/test_coupling_invariances.py::test_the_build_order_changes_nothing
@pytest.mark.slow
# Costly tier: compiles two graphs per example.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(data=st.data())
def test_the_build_order_is_invariant_on_generated_graphs(data):
    """CPL-181 with the structure, the configuration and both orders drawn."""
    gdef = data.draw(cg.graph_defs(max_group=4))
    group = data.draw(cg.group_configs(gdef))
    node_order = data.draw(_order_strategy(gdef))
    edge_order = data.draw(st.permutations(range(len(gdef.edges))))
    values = data.draw(_VALUES(gdef))
    a = run(build(gdef, group), gdef, values)
    b = run(build(gdef, group, node_order=node_order, edge_order=edge_order), gdef, values)
    assert_same_runs(a, b)


def test_the_interface_residual_does_not_depend_on_the_edge_order():
    """CPL-182: the same group, its edges added in reverse, reports the same residual.

    The draw on which the residual moved in its last bits (228.80760
    against 228.80762) while the interface norm summed in insertion order;
    both solvers, and every step's state and report.
    """
    gdef = cg.TRIANGLE
    values = cg.draw_values(np.random.default_rng(3), gdef, 0.9, nonnormal=True)
    for solver in ("ift", "fori"):
        group = _knobs("fixed", convergence_norm="interface", rtol=1e-5, max_iterations=12,
                       solver=solver, diagnostics=solver == "fori")
        a = run(build(gdef, group), gdef, values, steps=2)
        b = run(build(gdef, group, edge_order=tuple(reversed(range(len(gdef.edges))))),
                gdef, values, steps=2)
        assert a[0][1]["residual"] == b[0][1]["residual"], (solver, a[0][1], b[0][1])
        assert_same_runs(a, b)


# ---------------------------------------------------------------------------
# CPL-183: an identity relay on an internal edge
# ---------------------------------------------------------------------------

RELAY_BASE = cg._cycle(3, 2, chords=((0, 2),), outside=True, leaves=())


def with_relay(gdef, k=0):
    """*gdef* with an identity relay ``r`` on its ``k``-th internal edge, added after the source."""
    e = gdef.internal_edges[k]
    idx = gdef.edges.index(e)
    edges = list(gdef.edges)
    edges[idx] = cg.EdgeDef(e.src, "r", 0, e.field)
    edges.insert(idx + 1, cg.EdgeDef("r", e.dst, e.port, "x"))
    nodes = list(gdef.nodes)
    nodes.insert([nd.name for nd in nodes].index(e.src) + 1, cg.NodeDef("r", 1))
    members = list(gdef.group_nodes)
    members.insert(members.index(e.src) + 1, "r")
    return cg.GraphDef(n=gdef.n, nodes=tuple(nodes), edges=tuple(edges),
                       group_nodes=tuple(members)), e


RELAYED, RELAY_EDGE = with_relay(RELAY_BASE)


def relay_values(values, gdef=RELAY_BASE, edge=RELAY_EDGE):
    """*values* plus the relay: ``G = I``, ``b = 0``, started at its source's state."""
    out = dict(values)
    out["r"] = {"G": [np.eye(gdef.n, dtype=np.float32)], "b": np.zeros(gdef.n, np.float32),
                "x0": np.asarray(values[edge.src]["x0"], np.float32)}
    return out


@functools.lru_cache(maxsize=None)
def _relay_pair(key):
    group = dict(key)
    return build(RELAY_BASE, group), build(RELAYED, group)


def _exact_only(norm):
    """A criterion only exact stationarity meets, in the norm's own knob."""
    return {"tolerance": 0.0} if norm == "l2" else {"convergence_norm": norm, "rtol": 1e-30}


_GS_CONSTANT = [("none", "ift", "l2"), ("none", "fori", "interface"),
                ("fixed", "ift", "mixed"), ("fixed", "fori", "l2")]


@pytest.mark.parametrize("acceleration,solver,norm", _GS_CONSTANT)
# Costly tier: two compiled graphs, two steps each, per example.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_an_identity_relay_moves_no_bit_under_a_gauss_seidel_constant_iterator(
        acceleration, solver, norm, data):
    """CPL-183, the strongest statement: pass-for-pass identity of every other node."""
    group = _knobs(acceleration, solver=solver, max_iterations=6,
                   diagnostics=solver == "fori", **_exact_only(norm))
    gm, gm_relay = _relay_pair(tuple(sorted(group.items())))
    values = data.draw(_VALUES(RELAY_BASE, rhos=(0.5, 0.9, 0.99)))
    a = cg.trajectory(gm, RELAY_BASE, values, _STEPS)
    b = cg.trajectory(gm_relay, RELAYED, relay_values(values), _STEPS)
    others = [nd.name for nd in RELAY_BASE.nodes]
    for k, ((sa, _ma, ra), (sb, _mb, rb)) in enumerate(zip(a, b)):
        diff = cg.bitwise_differences({n: sa[n] for n in others}, {n: sb[n] for n in others})
        assert not diff, f"step {k + 1}: {diff}"
        np.testing.assert_array_equal(sb["r"]["x"], sb[RELAY_EDGE.src]["x"])
        assert (ra["iterations"], ra["converged"]) == (rb["iterations"], rb["converged"]), (ra, rb)


def _stall_bound(values):
    """Twice the floor model's distance of a float32 stall from the exact fixed point."""
    M = cg.coupling_matrix(RELAY_BASE, values)
    amp = float(np.linalg.norm(np.linalg.inv(np.eye(M.shape[0]) - M), np.inf))
    return 2.0 * amp * PRECISION_FLOOR_ULPS * EPS32


_ALL = [(acc, mode) for acc in ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj")
        for mode in ("gauss-seidel", "jacobi")]


@pytest.mark.parametrize("acceleration,mode", _ALL)
# Costly tier: two compiled graphs, run to their stall, per example.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_an_identity_relay_moves_the_fixed_point_only_by_rounding(acceleration, mode, data):
    """CPL-183, every schedule and acceleration: the two stalls agree to rounding."""
    group = _knobs(acceleration, iteration_mode=mode, max_iterations=600, tolerance=1e-30)
    gm, gm_relay = _relay_pair(tuple(sorted(group.items())))
    values = data.draw(_VALUES(RELAY_BASE, rhos=(0.3, 0.6, 0.9)))
    a = cg.trajectory(gm, RELAY_BASE, values, 1)[-1][0]
    b = cg.trajectory(gm_relay, RELAYED, relay_values(values), 1)[-1][0]
    scale = max(float(np.max(np.abs(a[nm]["x"]))) for nm in RELAY_BASE.group_nodes)
    gap = max(float(np.max(np.abs(a[nm]["x"].astype(np.float64) - b[nm]["x"])))
              for nm in RELAY_BASE.group_nodes)
    bound = _stall_bound(values) * scale
    assert gap <= bound, (gap, bound, gap / (bound / (2.0 * PRECISION_FLOOR_ULPS)))


@pytest.mark.parametrize("acceleration", ["none", "aitken", "fixed", "iqn-ils", "iqn-imvj"])
# Costly tier: two compiled graphs, one step each, per example.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_an_identity_relay_keeps_the_verdict(acceleration, data):
    """CPL-183: a convergent group converges with and without the relay; the verdicts agree.

    Aitken is drawn under Gauss-Seidel only.  Under Jacobi its single
    relaxation factor, re-derived from residuals that now count the relayed
    field twice, can stagnate where the relay-free group crept in under
    the threshold through the raw-residual fallback (measured at rate 0.9:
    converged at 278 passes against capped at 400) -- the documented
    failure of Aitken's single-mode assumption on a Jacobi spectrum
    (``_fixed_point_while``), not rounding.
    """
    mode = "jacobi" if acceleration in ("iqn-ils", "iqn-imvj") else "gauss-seidel"
    group = _knobs(acceleration, iteration_mode=mode, max_iterations=400, tolerance=1e-5)
    gm, gm_relay = _relay_pair(tuple(sorted(group.items())))
    rho = data.draw(st.sampled_from((0.3, 0.6, 0.9, 1.3)))
    seed = data.draw(st.integers(0, 2**32 - 1))
    # Normal gains: a non-normal draw at a rate of 0.9 can stay in its
    # transient for hundreds of passes, which is a slow group, not a verdict.
    values = cg.draw_values(np.random.default_rng(seed), RELAY_BASE, rho)
    (_sa, _ma, ra), = cg.trajectory(gm, RELAY_BASE, values, 1)
    (_sb, _mb, rb), = cg.trajectory(gm_relay, RELAYED, relay_values(values), 1)
    assert ra["converged"] == rb["converged"], (rho, ra, rb)
    if rho <= 0.6:
        assert ra["converged"], (rho, ra)
