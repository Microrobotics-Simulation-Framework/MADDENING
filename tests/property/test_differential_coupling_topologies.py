"""Differential oracles over coupled topologies the coupled_graphs generator never draws.

Coupling's defects have lived in graph *shapes* as much as in settings: a
node downstream of a group added before the group's members read it a step
late (CPL-025, MADD-ANO-120), an outside node added between two members of
a group inside a larger loop did too (MADD-ANO-144).  The structures here
(:func:`~tests.property.coupled_topologies.named_topologies` per push,
:func:`drawn_topologies` in the slow lane) put those shapes under test:
rings, chains, stars and nested cycles; outside nodes inside a group's
strongly connected component and several groups in one component; groups
beside acyclic outside nodes and ungrouped cycles; additive edges (three
into one port), flux edges, mapped edges between nodes of different sizes
and transformed edges; integer, boolean and typed PRNG-key leaves; nodes on
the three-argument contract.  Build orders and names are permuted on top.

**The monolithic reference** (:class:`~tests.property.coupled_topologies.LinearModel`):
every node is affine, so one step is one linear system, back edges read
from the pre-step state by the documented rule (restated, not called).
Every node outside a group must sit within its own float rounding of its
update at the values that rule says it reads; every group within what its
reported residual allows; the whole state within the propagated allowance
of the exact solve.  ``compile()`` must warn exactly for the groups that
are a strict subset of a strongly connected component.

**Invariances** on the same structures: renaming every node; permuting the
``add_node`` and ``add_edge`` calls, except the orders the documentation
says reach the result (a group's members, CPL-077; the nodes of a
component that is not exactly one group, whose order picks the edge read
late; three or more additive edges into one port, whose sum follows their
order); and an identity relay on a group-internal edge (bit for bit under
Gauss-Seidel with a constant iterator, and on its reference everywhere).

**Sparse mapped edges** (``mapping_kind``, see
:data:`~tests.property.coupled_topologies.MAPPING_KINDS`): the same
structures with every mapped edge a ``StaticSparseMapping`` over a pattern
fixed per structure, in both layouts.  The oracle is differential -- a
sparse edge must do what a dense edge holding the same matrix does -- and
is stated three ways.  Against the reference: the sparse graph satisfies
the monolithic reference the dense one is held to, unchanged, with the
drawn ``H`` zeroed outside the pattern.  Against the dense graph itself:
under a constant iterator both graphs take the same passes, so nothing but
rounding separates them, in every domain where a rounding is small against
the state.  And against themselves: two sparse builds of one structure
(renamed, reordered) are bit-identical, in every domain.

What this cannot see: non-linear nodes (no closed form), sub-cycling and
multi-rate stepping on these shapes (the covering array and the schedule
harness hold those), and any fault the reference's restatement of the
schedule shares with the library's (the restatement is from the
documentation, so a documented rule that is itself wrong passes).
"""

from __future__ import annotations

import functools

import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.property import coupled_graphs as cg
from tests.property import coupled_topologies as ct
from tests.property.test_coupling_invariances import report_differences

NAMED = ct.named_topologies()
#: The structure for the mapping kinds (wider interfaces), and every
#: structure by name.
MAPPED = ct.mapped_topologies()
STRUCTURES = {**NAMED, **MAPPED}
_STEPS = 3

#: The sparse kinds the monolithic reference is run over: every entry, a
#: ragged pattern with padded rows, and that pattern in the scatter layout.
_SPARSE_REFERENCE_KINDS = ("sparse-full", "sparse-ragged", "sparse-scatter")


@functools.lru_cache(maxsize=48)
def _built(name: str, choice: int, node_order=None, edge_order=None, rename=None,
           relay=None, constant=False, mapping_kind="matrix") -> tuple:
    """``(topology, knobs, Built)`` of a named structure, built once per variant."""
    topo = STRUCTURES[name]
    if relay is not None:
        topo = ct.with_identity_relay(topo, relay)
    knobs = ct.topology_knobs(topo, choice)
    if constant:
        knobs = [dict(g, acceleration="none", iteration_mode="gauss-seidel", tolerance=0.0,
                      convergence_norm="l2", max_iterations=8) for g in knobs]
        for g in knobs:
            g.pop("rtol", None)
            g.pop("relaxation", None)
            g.pop("jacobian_reuse", None)
    if rename is not None:
        mapping = dict(rename)
        topo = topo.renamed(mapping)
        node_order = tuple(mapping[n] for n in (node_order or STRUCTURES[name].names))
    return topo, knobs, ct.build(topo, knobs, node_order=node_order, edge_order=edge_order,
                                 mapping_kind=mapping_kind)


@st.composite
def _values(draw, topo, knobs, mapping_kind="matrix"):
    rho = draw(st.sampled_from([0.3, 0.6, 0.9]))
    seed = draw(st.integers(0, 2**32 - 1))
    return ct.draw_values(topo, np.random.default_rng(seed), rho,
                          nonnormal=draw(st.booleans()),
                          bias_scale=draw(st.sampled_from([1e-3, 1.0, 1e3])),
                          group_cfgs=ct.group_cfgs_of(knobs), mapping_kind=mapping_kind)


def assert_reproduces_the_reference(topo, knobs, built, values, steps=_STEPS, traj=None,
                                    dtype="float32"):
    """Every step of *built*'s run (or of *traj*, a run taken elsewhere) on the
    monolithic reference, in the graph's *dtype*."""
    model = ct.LinearModel(topo, values, node_order=built.node_order,
                           group_cfgs=ct.group_cfgs_of(knobs), dtype=dtype)
    traj = ct.run(built, values, steps) if traj is None else traj
    for k, step in enumerate(traj, start=1):
        where = f"{topo.label} step {k}"
        model.check_step(step.pre, step.state, step.reports,
                         thresholds=ct.thresholds_of(knobs), where=where)
        ct.check_leaves(topo, step.state, k, dividers=model.divider, where=where)
    return traj


#: The text of ``compile()``'s warning for an edge a group's block staggers
#: between two strongly connected components (MADD-ANO-159).
CROSSING = "is read from the previous step although no cycle runs through it"


def assert_warns_exactly_for_groups_inside_larger_loops(topo, built):
    # Every named topology reproduces the reference, which reads an edge
    # between two components forward: none is staggered across components.
    assert not any(CROSSING in w for w in built.warnings), built.warnings
    for gi, members in enumerate(topo.groups):
        said = any(f"coupling group {sorted(members)} is part of a larger feedback loop" in w
                   for w in built.warnings)
        assert said == ct.group_in_larger_loop(topo, gi), (
            f"{topo.label}: group {members} in a larger loop: "
            f"{ct.group_in_larger_loop(topo, gi)}, warned: {said} ({built.warnings})")


def assert_same_runs(a, b, *, what=""):
    """States bit for bit and reports value for value, step by step."""
    for k, (sa, sb) in enumerate(zip(a, b), start=1):
        diff = cg.bitwise_differences(sa.state, sb.state)
        assert not diff, f"{what} step {k}: states differ in {diff}"
        for gi in set(sa.reports) | set(sb.reports):
            rdiff = report_differences(sa.reports.get(gi), sb.reports.get(gi))
            assert not rdiff, f"{what} step {k} group {gi}: reports differ: {rdiff}"


# ---------------------------------------------------------------------------
# The monolithic reference, per push
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["as-built", "interleaved"])
@pytest.mark.parametrize("choice", [0, 1])
@pytest.mark.parametrize("name", sorted(NAMED))
# Costly tier: one compiled graph per (structure, configuration, order);
# the examples draw the gains, biases, mapping matrices, scale and start.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_named_topology_reproduces_the_monolithic_reference(name, choice, order, data):
    """Per push; slow sibling :func:`test_a_drawn_topology_reproduces_the_monolithic_reference`.

    ``"interleaved"`` adds every reader of a cycle before the cycle and
    every outside node of a group's component between the group's first
    two members (:func:`~tests.property.coupled_topologies.interleaved_order`):
    the two build orders the schedule has read a step late before.
    """
    node_order = None if order == "as-built" else ct.interleaved_order(NAMED[name])
    topo, knobs, built = _built(name, choice, node_order=node_order)
    values = data.draw(_values(topo, knobs))
    assert_reproduces_the_reference(topo, knobs, built, values)


#: The structure whose ring carries an interface mapping (then a transform)
#: on one internal edge and a transform alone on the other, and the
#: configuration that puts the interface norm on it.
_MAPPED_RING, _INTERFACE_CHOICE = "chain-into-ring", 2


def _assert_the_interface_norm_reads_a_mapped_edge(topo, knobs):
    """The premise of the mapped-ring tests."""
    (gi,) = range(len(topo.groups))
    assert knobs[gi]["convergence_norm"] == "interface", knobs
    internal = [topo.edges[i] for i in topo.internal_edges(gi)]
    assert any(e.mapped and e.transform for e in internal), internal
    assert any(not e.mapped and e.transform for e in internal), internal


@pytest.mark.parametrize("order", ["as-built", "interleaved"])
# Costly tier: one compiled graph per order; the examples draw the gains,
# biases, the mapping matrix, scale and start.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_the_interface_norm_on_a_mapped_ring_reproduces_the_monolithic_reference(order, data):
    """Aitken, Gauss-Seidel and ``convergence_norm="interface"`` on a ring with a
    mapped internal edge: the reported residual is ``||F(x) - x||`` of the
    returned state *in what the edges deliver* -- the mapping matrix the step
    ran with, then the transform -- the defect is within what that residual
    allows, and a converged group is within its threshold of its exact fixed
    point in that norm.

    While the norm read the source field and left the mapping out, the
    reference restated that rule and agreed with it; with the reference
    reading what the step delivers, the same library fails this test (the
    mapping object holds zeros: the matrix reaches the step as a parameter).
    Slow sibling: :func:`test_a_drawn_topology_reproduces_the_monolithic_reference`,
    which draws mapped and transformed internal edges under every norm.
    """
    node_order = None if order == "as-built" else ct.interleaved_order(NAMED[_MAPPED_RING])
    topo, knobs, built = _built(_MAPPED_RING, _INTERFACE_CHOICE, node_order=node_order)
    _assert_the_interface_norm_reads_a_mapped_edge(topo, knobs)
    values = data.draw(_values(topo, knobs))
    assert_reproduces_the_reference(topo, knobs, built, values)


def _sparse_reference_cases():
    """Every sparse kind on every structure, and the dense kind on the
    structure built for the mapping kinds (the others have their dense run
    above).  Per push: the structure with wide interfaces under both
    configurations, and every other structure under the first, each in its
    own build order.  The rest -- the second configuration on the narrow
    structures, the interleaved order -- is slow."""
    cases = []
    for name in sorted(STRUCTURES):
        kinds = _SPARSE_REFERENCE_KINDS + (("matrix",) if name in MAPPED else ())
        for kind in kinds:
            for choice in (0, 1):
                for order in ("as-built", "interleaved"):
                    per_push = order == "as-built" and (choice == 0 or name in MAPPED)
                    cases.append(pytest.param(
                        name, kind, choice, order, id=f"{name}-{kind}-{choice}-{order}",
                        marks=() if per_push else pytest.mark.slow))
    return cases


# Per push: tests/property/test_differential_coupling_topologies.py::test_a_named_topology_with_sparse_mapped_edges_reproduces_the_monolithic_reference
# (every structure and sparse kind under configuration 0 in its own build
# order; the slow cases are the same check under configuration 1 and the
# interleaved order)
@pytest.mark.parametrize("name, mapping_kind, choice, order", _sparse_reference_cases())
# Costly tier: one compiled graph per case; the examples draw the gains,
# biases, mapping matrices, scale and start.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_named_topology_with_sparse_mapped_edges_reproduces_the_monolithic_reference(
        name, mapping_kind, choice, order, data):
    """The dense oracle, unchanged, over a graph whose mapped edges are
    sparse: every node outside a group within its own rounding of its
    update, every group within what its reported residual allows, with
    ``H`` the drawn matrix zeroed outside the edge's pattern -- for every
    entry in the gather layout, for a ragged pattern with padded slots, and
    for that pattern in the scatter layout."""
    node_order = None if order == "as-built" else ct.interleaved_order(STRUCTURES[name])
    topo, knobs, built = _built(name, choice, node_order=node_order, mapping_kind=mapping_kind)
    mapped = [type(e.mapping).__name__ for e in built.gm.edges if e.mapping is not None]
    assert mapped and set(mapped) == {
        "StaticLinearMapping" if mapping_kind == "matrix" else "StaticSparseMapping"}, (
        "premise: the mapped edges are of the kind under test")
    values = data.draw(_values(topo, knobs, mapping_kind))
    assert_reproduces_the_reference(topo, knobs, built, values)


def test_the_sparse_patterns_of_the_named_structures_can_express_a_wrong_index():
    """The fixture can express the defect: across the structures the ragged
    pattern leaves entries out, pads rows (so a mask is traced), has a row
    with two entries (so a row sum is one) and an entry off the diagonal;
    and the scatter layout has a source feeding two targets and one feeding
    none."""
    ragged = []
    for topo in STRUCTURES.values():
        for i, e in enumerate(topo.edges):
            if e.mapped:
                ragged.append(ct.mapping_pattern(topo, i, "sparse-ragged"))
                assert ct.mapping_pattern(topo, i, "sparse-full").all()
                assert ct.mapping_pattern(topo, i, "matrix") is None
    assert len(ragged) >= 12
    wide = MAPPED["mapped-ring-wide"]
    keys = [(e.src, e.dst, e.port) for e in wide.edges if e.mapped]
    assert len(keys) == 7 and len(set(keys)) == 6, "two mapped edges on one field pair"
    assert any(wide.internal(e) and wide.node(e.src).n == wide.node(e.dst).n
               for e in wide.edges if e.mapped), "a mapped edge between equal sizes"
    assert any(e.field == "q" for e in wide.edges if e.mapped), "a mapped flux edge"
    assert any(not mask.all() for mask in ragged)
    gather = [ct.pattern_slots(mask, scatter=False) for mask in ragged]
    scatter = [ct.pattern_slots(mask, scatter=True) for mask in ragged]
    assert any(not s.valid.all() for s in gather), "a padded gather row"
    assert any(s.valid.sum(axis=1).max() > 1 for s in gather), "a row of two entries"
    assert any((s.source[s.valid] != 0).any() for s in gather), "an index other than 0"
    assert any(s.valid.sum(axis=1).max() > 1 for s in scatter), "a source with two targets"
    assert any(s.valid.sum(axis=1).min() == 0 for s in scatter), "a source with no target"


@pytest.mark.parametrize("name", sorted(NAMED))
def test_compile_warns_exactly_for_a_group_inside_a_larger_loop(name):
    """``compile()`` names every group that is a strict subset of a component, and no other."""
    topo, _knobs, built = _built(name, 0)
    assert_warns_exactly_for_groups_inside_larger_loops(topo, built)


# ---------------------------------------------------------------------------
# Invariances, per push
# ---------------------------------------------------------------------------

#: Names that sort differently from the schedule, one before ``"_meta"``,
#: one with the ``"+"`` group keys are joined with, one with a dot.
_NAME_POOL = ("zeta", "Alpha", "_m", "m+n", "q.r", "beta", "Z9", "omega", "k_1", "ALPHA",
              "a-b", "yy", "c0x", "Q")


def _renaming(topo, seed):
    rng = np.random.default_rng(seed)
    pool = list(_NAME_POOL)
    rng.shuffle(pool)
    return tuple(zip(topo.names, pool[:len(topo.names)]))


@pytest.mark.parametrize("name", sorted(NAMED))
# Costly tier: two compiled graphs per structure, the examples draw values.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_renaming_every_node_of_a_named_topology_changes_nothing(name, data):
    """CPL-180 on the new shapes; slow sibling :func:`test_renaming_a_drawn_topology_changes_nothing`."""
    topo, knobs, built = _built(name, 0)
    rename = _renaming(topo, 11)
    _rt, _rk, renamed = _built(name, 0, rename=rename)
    values = data.draw(_values(topo, knobs))
    assert_same_runs(ct.run(built, values, 2),
                     ct.run(renamed, values, 2, rename=dict(rename)), what=f"{name} renamed")


def _orders(topo, seed):
    rng = np.random.default_rng(seed)
    keep_nodes, keep_edges = ct.invariant_orders(topo)
    return (tuple(ct.ordered_permutation(topo.names, keep_nodes, rng)),
            tuple(ct.ordered_permutation(list(range(len(topo.edges))), keep_edges, rng)))


@pytest.mark.parametrize("name", sorted(NAMED))
# Costly tier: two compiled graphs per structure, the examples draw values.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_the_build_order_of_a_named_topology_changes_nothing(name, data):
    """CPL-181 on the new shapes, outside nodes included; slow sibling draws the orders too."""
    topo, knobs, built = _built(name, 0)
    node_order, edge_order = _orders(topo, 5)
    _t, _k, reordered = _built(name, 0, node_order=node_order, edge_order=edge_order)
    assert reordered.gm.schedule != built.gm.schedule or edge_order != tuple(
        range(len(topo.edges))), "the permutation must reach the graph"
    values = data.draw(_values(topo, knobs))
    assert_same_runs(ct.run(built, values, 2), ct.run(reordered, values, 2),
                     what=f"{name} reordered")


def _relayable(topo):
    """The first group-internal, unmapped edge (the relay's place)."""
    return next(i for i, e in enumerate(topo.edges) if topo.internal(e) and not e.mapped)


@pytest.mark.parametrize("name", sorted(NAMED))
# Costly tier: two compiled graphs per structure, the examples draw values.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_an_identity_relay_on_a_named_topology_moves_no_bit(name, data):
    """CPL-183's strongest statement on the new shapes.

    Gauss-Seidel, no acceleration and a criterion only exact stationarity
    meets (``tolerance=0``, eight passes): the relay is recomputed from
    its source in the same pass before anything reads it, so every other
    node's iterate is bit-identical pass for pass and the relay holds what
    it relays.
    """
    topo = NAMED[name]
    k = _relayable(topo)
    t0, knobs, plain = _built(name, 0, constant=True)
    t1, _k1, relayed = _built(name, 0, relay=k, constant=True)
    values = data.draw(_values(t0, knobs))
    a = ct.run(plain, values, 2)
    b = ct.run(relayed, ct.relay_values(t0, values, t1, k), 2)
    others = list(t0.names)
    for step, (sa, sb) in enumerate(zip(a, b), start=1):
        diff = cg.bitwise_differences({n: sa.state[n] for n in others},
                                      {n: sb.state[n] for n in others})
        assert not diff, f"{name} step {step}: the relay moved {diff}"


@pytest.mark.parametrize("name", sorted(NAMED))
# Costly tier: one compiled graph per structure, the examples draw values.
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_an_identity_relay_keeps_a_named_topology_on_its_reference(name, data):
    """CPL-183 under every configuration: the relayed graph reproduces the same exact solve."""
    topo = NAMED[name]
    k = _relayable(topo)
    t1, knobs, relayed = _built(name, 1, relay=k)
    values = ct.relay_values(topo, data.draw(_values(topo, ct.topology_knobs(topo, 1))), t1, k)
    assert_reproduces_the_reference(t1, knobs, relayed, values)


# ---------------------------------------------------------------------------
# Drawn topologies, slow
# ---------------------------------------------------------------------------


@st.composite
def drawn_topologies(draw):
    """One or two groups of 2-4 members (ring, nested cycles or star), joined or not,
    with drivers, readers, an outside node in a group's component, an
    ungrouped cycle, additive ports, flux, mapped and transformed edges,
    every leaf kind and three-argument nodes, each drawn."""
    b = ct.TopologyBuilder()
    sizes = st.sampled_from([1, 2])
    transforms = st.sampled_from([None, None, "negate", "scale_2.0", "scale_0.5", "identity"])

    def node(name, **kw):
        return b.node(name, draw(sizes), alpha=draw(st.sampled_from([0.0, 0.5, -0.25, 1.0])),
                      beta=draw(st.sampled_from([0.0, 1.0, -0.5])),
                      leaves=tuple(draw(st.lists(st.sampled_from(ct.LEAF_KINDS), unique=True,
                                                 max_size=3))),
                      three_arg=draw(st.integers(0, 3)) == 0, **kw)

    def edge(src, dst, **kw):
        mapped = None if b._nodes[src]["n"] != b._nodes[dst]["n"] else draw(  # noqa: SLF001
            st.integers(0, 3)) == 0
        return b.edge(src, dst, transform=draw(transforms), mapped=mapped, **kw)

    groups = []
    for gi in range(draw(st.integers(1, 2))):
        m = draw(st.integers(2, 4))
        shape = draw(st.sampled_from(["ring", "nested", "star"]))
        flux_member = draw(st.integers(-1, m - 1))
        members = [node(f"g{gi}{k}", flux=(k == flux_member)) for k in range(m)]
        pairs = ([(members[0], leaf) for leaf in members[1:]]
                 + [(leaf, members[0]) for leaf in members[1:]] if shape == "star"
                 else [(members[i], members[(i + 1) % m]) for i in range(m)]
                 + ([(members[m - 1], members[1])] if shape == "nested" and m >= 3 else []))
        for src, dst in pairs:
            field = "q" if b._nodes[src]["flux"] and draw(st.booleans()) else "x"  # noqa: SLF001
            edge(src, dst, field=field)
        b.group(*members)
        groups.append(members)
    if len(groups) == 2:
        link = draw(st.sampled_from(["one-scc", "chain", "apart"]))
        if link in ("one-scc", "chain"):
            edge(draw(st.sampled_from(groups[0])), draw(st.sampled_from(groups[1])))
        if link == "one-scc":
            edge(draw(st.sampled_from(groups[1])), draw(st.sampled_from(groups[0])))
    everyone = [m for g in groups for m in g]
    drivers = [node(f"d{k}", flux=k == 0) for k in range(draw(st.integers(0, 2)))]
    if len(drivers) == 2 and draw(st.booleans()):
        edge(drivers[0], drivers[1], field="q")
    if drivers:
        target = draw(st.sampled_from(everyone))
        if len(drivers) == 2 and draw(st.booleans()):
            # Three additive edges into one port: both drivers and a member.
            port = edge(drivers[0], target)
            edge(drivers[1], target, port=port)
            other = draw(st.sampled_from([m for m in everyone if m != target]))
            edge(other, target, port=port)
        else:
            for d in drivers:
                edge(d, draw(st.sampled_from(everyone)))
    if draw(st.booleans()):
        o = node("o")
        edge(draw(st.sampled_from(groups[0])), o)
        edge(o, draw(st.sampled_from(groups[0])))
    if draw(st.booleans()):
        u0, u1 = node("u0"), node("u1")
        edge(draw(st.sampled_from(everyone)), u0)
        edge(u0, u1)
        edge(u1, u0)
    for k in range(draw(st.integers(0, 2))):
        edge(draw(st.sampled_from(everyone)), node(f"r{k}"))
    topo = b.build("drawn")
    order = tuple(draw(st.permutations(topo.names)))
    return topo, order


def _drawn_case(data):
    topo, order = data.draw(drawn_topologies())
    knobs = ct.topology_knobs(topo, data.draw(st.integers(0, 4)))
    note(f"{topo}\norder={order}\nknobs={knobs}")
    return topo, order, knobs


# Slow: the structure is the draw, so every example compiles a graph of its
# own (1-2 s each on CI).
# Per push: tests/property/test_differential_coupling_topologies.py::test_a_named_topology_reproduces_the_monolithic_reference
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_drawn_topology_reproduces_the_monolithic_reference(data):
    topo, order, knobs = _drawn_case(data)
    built = ct.build(topo, knobs, node_order=order)
    assert_warns_exactly_for_groups_inside_larger_loops(topo, built)
    assert_reproduces_the_reference(topo, knobs, built, data.draw(_values(topo, knobs)))


# Slow: two graphs compiled per example.
# Per push: tests/property/test_differential_coupling_topologies.py::test_renaming_every_node_of_a_named_topology_changes_nothing
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_renaming_a_drawn_topology_changes_nothing(data):
    topo, order, knobs = _drawn_case(data)
    mapping = dict(zip(topo.names, data.draw(st.permutations(_NAME_POOL))))
    values = data.draw(_values(topo, knobs))
    a = ct.run(ct.build(topo, knobs, node_order=order), values, 2)
    b = ct.run(ct.build(topo.renamed(mapping), knobs,
                        node_order=tuple(mapping[n] for n in order)),
               values, 2, rename=mapping)
    assert_same_runs(a, b, what="renamed")


# Slow: two graphs compiled per example.
# Per push: tests/property/test_differential_coupling_topologies.py::test_the_build_order_of_a_named_topology_changes_nothing
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_the_build_order_of_a_drawn_topology_changes_nothing(data):
    topo, order, knobs = _drawn_case(data)
    seed = data.draw(st.integers(0, 2**32 - 1))
    keep_nodes, keep_edges = ct.invariant_orders(topo)
    rng = np.random.default_rng(seed)
    # The keep-groups take their relative order from the first build's.
    other = tuple(ct.ordered_permutation(order, keep_nodes, rng))
    edges = tuple(ct.ordered_permutation(list(range(len(topo.edges))), keep_edges, rng))
    values = data.draw(_values(topo, knobs))
    a = ct.run(ct.build(topo, knobs, node_order=order), values, 2)
    b = ct.run(ct.build(topo, knobs, node_order=other, edge_order=edges), values, 2)
    assert_same_runs(a, b, what=f"orders {order} / {other}")


# Slow: two graphs compiled per example.
# Per push: tests/property/test_differential_coupling_topologies.py::test_an_identity_relay_keeps_a_named_topology_on_its_reference
@pytest.mark.slow
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_an_identity_relay_keeps_a_drawn_topology_on_its_reference(data):
    topo, order, knobs = _drawn_case(data)
    candidates = [i for i, e in enumerate(topo.edges) if topo.internal(e) and not e.mapped]
    if not candidates:
        return      # every internal edge of this draw is mapped; nothing to relay
    k = data.draw(st.sampled_from(candidates))
    relayed = ct.with_identity_relay(topo, k)
    src = topo.edges[k].src
    r_order = list(order)
    r_order.insert(r_order.index(src) + 1, "rly")
    r_knobs = ct.topology_knobs(relayed, 0)
    built = ct.build(relayed, r_knobs, node_order=tuple(r_order))
    values = ct.relay_values(topo, data.draw(_values(topo, r_knobs)), relayed, k)
    assert_reproduces_the_reference(relayed, r_knobs, built, values)


# ---------------------------------------------------------------------------
# What the harness found
# ---------------------------------------------------------------------------


def _ring_with(extra_nodes, extra_edges, *, flux_members=()):
    b = ct.TopologyBuilder()
    b.node("g0", 1, alpha=0.5, flux="g0" in flux_members)
    b.node("g1", 1, alpha=0.0, beta=1.0, flux="g1" in flux_members)
    for name, kw in extra_nodes:
        b.node(name, 1, **kw)
    b.edge("g0", "g1")
    b.edge("g1", "g0")
    for src, dst, kw in extra_edges:
        b.edge(src, dst, **kw)
    b.group("g0", "g1")
    return b.build("finding")


def _steps_like_its_reference(topo, knobs=None, node_order=None):
    knobs = knobs or [dict(tolerance=1e-6, max_iterations=100)]
    built = ct.build(topo, knobs, node_order=node_order)
    values = ct.draw_values(topo, np.random.default_rng(0), 0.5,
                            group_cfgs=ct.group_cfgs_of(knobs))
    assert_reproduces_the_reference(topo, knobs, built, values)


@pytest.mark.xfail(strict=True, raises=KeyError, reason=(
    "CPL-186: a flux edge from an outside node into a coupling group member fails to "
    "trace with a bare KeyError naming the flux; pending fix"))
def test_a_flux_edge_from_an_outside_node_into_a_group_member_steps():
    """The flux an outside producer computes earlier in the step reaches a member.

    ``compute_boundary_fluxes``'s keys "become available as source_field
    on edges" (node authoring guide), and a forward flux edge between two
    outside nodes works.  ``_run_coupled_block_impl`` resolves a member's
    inputs from the group's own flux dictionary only, never from the
    fluxes the outside nodes produced earlier in the step
    (``coupling/_coupled_block.py``, ``_resolve_value``), so the step raises
    ``KeyError: 'q'`` at trace.  Measured at 0.4.0.dev0 under both solvers.
    """
    _steps_like_its_reference(_ring_with([("drv", dict(alpha=1.0, beta=1.0, flux=True))],
                                         [("drv", "g0", dict(field="q"))]))


@pytest.mark.xfail(strict=True, raises=KeyError, reason=(
    "CPL-186: a flux edge from a coupling group member to an outside reader fails to "
    "trace with a bare KeyError naming the flux; pending fix"))
def test_a_flux_edge_from_a_group_member_to_an_outside_reader_steps():
    """A member's flux reaches a reader outside the group.

    The group computes its members' fluxes in a dictionary local to
    ``_run_coupled_block_impl``; the outside reader looks them up in the
    step's own ``flux_state`` (``_build_step_fn``'s
    ``_resolve_and_update_node``), which the group never writes, so the
    step raises ``KeyError: 'q'`` at trace.
    """
    _steps_like_its_reference(_ring_with([("sink", dict(alpha=0.5))],
                                         [("g1", "sink", dict(field="q"))],
                                         flux_members=("g1",)))


def _ungrouped_flux_ring():
    b = ct.TopologyBuilder()
    b.node("a", 1, alpha=0.5, flux=True)
    b.node("b", 1, alpha=0.25, beta=1.0)
    b.edge("a", "b", field="q")
    b.edge("b", "a")
    return b.build("ungrouped-flux-ring")


def test_a_flux_edge_read_forward_in_an_ungrouped_cycle_steps():
    """The control: built ``a, b`` the flux edge ``a.q -> b`` is read this step, and works."""
    _steps_like_its_reference(_ungrouped_flux_ring(), knobs=[], node_order=("a", "b"))


@pytest.mark.xfail(strict=True, raises=KeyError, reason=(
    "CPL-186: a flux edge an ungrouped cycle reads from the previous step fails to trace "
    "with a bare KeyError naming the flux; pending fix"))
def test_a_flux_edge_read_late_in_an_ungrouped_cycle_steps():
    """Built ``b, a``, the cycle is staggered on the flux edge, which then fails.

    A back edge reads its source from the previous step's state
    (``_build_step_fn``), and a flux is not in the state: the lookup falls
    through to ``state['a']['q']`` and raises ``KeyError: 'q'`` at trace,
    so whether the graph runs at all depends on the order the two nodes
    were added.  The adaptive steppers refuse the same edge with a
    ``ValueError`` that names it (``_build_dt_step_fn``).
    """
    _steps_like_its_reference(_ungrouped_flux_ring(), knobs=[], node_order=("b", "a"))


@pytest.mark.parametrize("acceleration", ["aitken", "fixed"])
def test_a_typed_prng_key_in_a_member_steps_under_aitken_and_fixed_relaxation(acceleration):
    """CPL-003: a PRNG-key field is computed by every pass and never relaxed.

    Under ``solver="ift"`` with ``"aitken"`` or ``"fixed"`` the step
    builder sized the accelerator from ``_flatten(state_after_first)`` with
    ``accel_fields=None`` (``_run_coupled_block_impl``: "the IFT path
    relaxes on its own floating vector and reads this only for IQN's index
    map"), which concatenated *every* field of every member, the typed key
    included: ``ValueError: dtype=key<fry> is not a valid dtype for JAX
    type promotion`` (MADD-ANO-158).  It now flattens the floating fields
    only there, as ``"none"``, the IQN pair and ``solver="fori"`` do.
    """
    b = ct.TopologyBuilder()
    b.node("a", 2, alpha=0.5, leaves=("key",))
    b.node("b", 2, alpha=0.0, beta=1.0)
    b.edge("a", "b")
    b.edge("b", "a")
    b.group("a", "b")
    knobs = [dict(acceleration=acceleration, relaxation=0.7, tolerance=1e-6,
                  max_iterations=100)]
    _steps_like_its_reference(b.build("keyed-pair"), knobs=knobs)


def test_a_typed_prng_key_in_a_member_steps_under_every_other_iterator():
    """The control for the finding above: the same group steps everywhere else."""
    b = ct.TopologyBuilder()
    b.node("a", 2, alpha=0.5, leaves=("key",))
    b.node("b", 2, alpha=0.0, beta=1.0)
    b.edge("a", "b")
    b.edge("b", "a")
    b.group("a", "b")
    topo = b.build("keyed-pair")
    for knobs in (dict(acceleration="none"), dict(acceleration="iqn-ils"),
                  dict(acceleration="aitken", solver="fori", diagnostics=True)):
        _steps_like_its_reference(topo, knobs=[dict(knobs, tolerance=1e-6,
                                                    max_iterations=100)])


def _joined_through_an_outside_node():
    """Group ``{a, b}`` with ``a -> c -> b`` and ``a -> b``: no node-level cycle at all."""
    b = ct.TopologyBuilder()
    b.node("a", 1, alpha=0.5, beta=1.0)
    b.node("b", 1, alpha=0.25)
    b.node("c", 1, alpha=0.0)
    b.edge("a", "c")
    b.edge("c", "b")
    b.edge("a", "b")
    b.group("a", "b")
    return b.build("joined-off-cycle")


def test_a_group_joined_through_an_outside_node_off_any_cycle_says_it_reads_it_late():
    """The group runs as one block, so ``c -> b`` is read from the previous step -- silently.

    ``CouplingGroup`` documents that its members "form (part of) a cycle";
    nothing checks it.  With ``a -> c -> b`` and no edge back, the graph
    is acyclic and an uncoupled step would read every edge this step, but
    the group's block runs ``a`` and ``b`` together before ``c``
    (``_block_schedule``), so ``identify_back_edges`` staggers ``c -> b``
    -- an edge between two strongly connected components, which CPL-025
    says always points forward.  ``compile()``'s warning for a group inside
    a larger loop (``_loop_through_outside_nodes``) looks for strongly
    connected components only and stayed silent (MADD-ANO-159); a second
    warning (``_staggered_across_components``) now names every staggered
    edge between two components and the group whose block forced it.
    """
    topo = _joined_through_an_outside_node()
    try:
        built = ct.build(topo, [dict(tolerance=1e-6, max_iterations=50)])
    except (ValueError, RuntimeError):
        return
    assert any("c.x -> b.u" in w and CROSSING in w for w in built.warnings), built.warnings
    named = [w for w in built.warnings if CROSSING in w]
    assert len(named) == 1 and "['a', 'b']" in named[0], named


def test_an_edge_staggered_inside_a_component_is_not_named_as_crossing_components():
    """The control for the warning above: an ungrouped cycle's back edge, and a
    group inside a larger loop, stagger an edge *inside* one component --
    the documented schedule (CPL-025, CPL-181) -- and the crossing warning
    says nothing about them."""
    ring = ct.TopologyBuilder()
    ring.node("a", 1, alpha=0.5, beta=1.0)
    ring.node("p", 1, alpha=0.25)
    ring.edge("a", "p")
    ring.edge("p", "a")
    for order in (("a", "p"), ("p", "a")):
        built = ct.build(ring.build("ungrouped-ring"), [], node_order=order)
        assert not any(CROSSING in w for w in built.warnings), built.warnings
    loop = ct.TopologyBuilder()
    loop.node("a", 1, alpha=0.5, beta=1.0)
    loop.node("b", 1, alpha=0.25)
    loop.node("c", 1, alpha=0.0)
    loop.edge("a", "b")
    loop.edge("b", "a")
    loop.edge("b", "c")
    loop.edge("c", "a")
    loop.group("a", "b")
    built = ct.build(loop.build("group-in-a-loop"), [dict(tolerance=1e-6, max_iterations=50)])
    assert any("is part of a larger feedback loop" in w for w in built.warnings), built.warnings
    assert not any(CROSSING in w for w in built.warnings), built.warnings


def test_the_build_order_of_an_ungrouped_cycle_picks_the_edge_read_late():
    """An ungrouped cycle is staggered on the edge its build order makes a back edge.

    Documented in ``topological_sort`` ("within a cycle ... the nodes keep
    their order in node_names") and for a group inside a larger loop
    (CPL-181's exclusion), but not stated where CPL-181 says the order of
    every outside node's ``add_node`` call "does not reach the result": a
    cycle of outside nodes is the counterexample.  Both orders step as the
    documented rule says, and differently from each other.
    """
    b = ct.TopologyBuilder()
    b.node("a", 1, alpha=0.5, beta=1.0)
    b.node("p", 1, alpha=0.25)
    b.edge("a", "p")
    b.edge("p", "a")
    topo = b.build("ungrouped-ring")
    values = ct.draw_values(topo, np.random.default_rng(1), 0.5)
    runs = []
    for order in (("a", "p"), ("p", "a")):
        built = ct.build(topo, [], node_order=order)
        runs.append(assert_reproduces_the_reference(topo, [], built, values))
    assert cg.bitwise_differences(runs[0][-1].state, runs[1][-1].state)



# ---------------------------------------------------------------------------
# The invariances in the other numeric domains, per push
#
# The renaming, build-order and identity-relay invariances (CPL-180 to
# CPL-183) and Jacobi's indifference to its members' order under a constant
# iterator (CPL-078) are stated for float32 single-rate graphs above.  Here
# each runs again on the named structures in eight more domains, comparing two
# builds of one structure bit for bit as above:
#
# * ``f64``: every relay float64, under x64;
# * ``bfloat16`` and ``float16`` (the ``16bit`` domain): every relay in the
#   16-bit dtype;
# * ``multi_rate``: an isolated clock at half the base step, so the graph is
#   multi-rate and every group fires on every other base step;
# * ``sub_cycled``: the last member of every group at half the group's step,
#   ``subcycling=True``, sub-stepped twice per pass;
# * ``vmap``: three draws through one ``jax.vmap`` of the step, each member's
#   state and report compared;
# * ``predictors_warm_starts``: every group with ``predictor="quadratic"``
#   (IQN-IMVJ keeps its Jacobian reuse), over four steps;
# * ``checkpoint_restart``: each run saved after two steps, run on, reset,
#   loaded and run again; the steps after the load must reproduce the
#   uninterrupted run's, and are what is compared.
#
# Three fixed draws per case rather than Hypothesis examples: one compile
# per build, the draws cheap after it.
# ---------------------------------------------------------------------------

import contextlib  # noqa: E402
import dataclasses  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

DOMAINS = ("f64", "bfloat16", "float16", "multi_rate", "sub_cycled", "vmap",
           "predictors_warm_starts", "checkpoint_restart")
#: An isolated relay at half the step: the graph's base step halves, so it is
#: multi-rate and every other node fires on every other base step.
_TICK = ct.TNode("tick", n=1, alpha=1.0, beta=1.0, timestep=0.5)
_DRAWS = ((3, 0.3), (17, 0.6), (29, 0.9))
#: Per push the domains run on ``chain-into-ring`` (the cheapest structure,
#: and the one configuration 2 puts the interface norm on); the other three
#: structures are slow, two compiles each per domain.
_DOMAIN_NAMES = [n if n == "chain-into-ring" else pytest.param(n, marks=pytest.mark.slow)
                 for n in sorted(NAMED)]


@contextlib.contextmanager
def _x64(on: bool):
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", on)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _domain_dtype(domain) -> str:
    return {"f64": "float64", "bfloat16": "bfloat16", "float16": "float16"}.get(domain,
                                                                              "float32")


def _domain_topology(topo, domain):
    if domain == "multi_rate":
        return dataclasses.replace(topo, nodes=topo.nodes + (_TICK,))
    if domain == "sub_cycled":
        return topo.with_timesteps({members[-1]: 0.5 for members in topo.groups})
    if domain == "checkpoint_restart":
        # ``save_state`` cannot write a typed PRNG key (MADD-ANO-171): the
        # restart runs on the structure without its key leaves.
        return dataclasses.replace(topo, nodes=tuple(
            dataclasses.replace(nd, leaves=tuple(lf for lf in nd.leaves if lf != "key"))
            for nd in topo.nodes))
    return topo


def _domain_knobs(knobs, domain):
    if domain == "sub_cycled":
        return [dict(g, subcycling=True, boundary_interpolation="constant") for g in knobs]
    if domain == "predictors_warm_starts":
        return [dict(g, predictor="quadratic") for g in knobs]
    return knobs


def _domain_steps(domain) -> int:
    return {"multi_rate": 4, "predictors_warm_starts": 4}.get(domain, 2)


@functools.lru_cache(maxsize=96)
def _built_in(domain: str, name: str, choice: int, node_order=None, edge_order=None,
              rename=None, relay=None, constant: str | None = None,
              mapping_kind: str = "matrix") -> tuple:
    """``(topology, knobs, Built)`` of a named structure in *domain*.

    *constant* is ``"gauss-seidel"`` or ``"jacobi"``: every group under no
    acceleration and a criterion only exact stationarity meets.
    """
    topo = _domain_topology(STRUCTURES[name], domain)
    names = topo.names
    if relay is not None:
        topo = ct.with_identity_relay(topo, relay)
        names = topo.names
    knobs = ct.topology_knobs(topo, choice)
    if constant:
        knobs = [dict(max_iterations=8, tolerance=0.0, convergence_norm="l2",
                      acceleration="none", iteration_mode=constant) for _ in knobs]
    knobs = _domain_knobs(knobs, domain)
    if rename is not None:
        mapping = dict(rename)
        topo = topo.renamed(mapping)
        node_order = tuple(mapping[n] for n in (node_order or names))
    with _x64(domain == "f64"):
        built = ct.build(topo, knobs, dtype=_domain_dtype(domain), node_order=node_order,
                         edge_order=edge_order, mapping_kind=mapping_kind)
    return topo, knobs, built


def _domain_values(topo, knobs, domain, mapping_kind="matrix") -> list:
    with _x64(domain == "f64"):
        return [ct.draw_values(topo, np.random.default_rng(seed), rho,
                               dtype=_domain_dtype(domain), group_cfgs=ct.group_cfgs_of(knobs),
                               mapping_kind=mapping_kind)
                for seed, rho in _DRAWS]


_BATCHED: dict = {}


def _batched_start(built, values_list, rename=None):
    """``(state, params)`` of the batch: member *i* is draw *i*'s own start.

    ``set_initial`` rewrites the graph's state dictionary in place
    (``reset_state`` updates it and ``set_node_state`` assigns into it), so
    each member is a copy taken before the next draw is written: a list of
    the live dictionary holds the last draw once per member.
    """
    gm = built.gm
    states, params = [], []
    for v in values_list:
        ct.set_initial(built, v, rename)
        states.append(jax.tree.map(lambda x: x, gm._state))
        params.append(ct.params_for(built, v, rename))
    return (jax.tree.map(lambda *xs: jnp.stack(xs), *states),
            jax.tree.map(lambda *xs: jnp.stack(xs), *params))


def _batched_runs(built, values_list, steps, rename):
    """One trajectory per draw, all three through one ``jax.vmap`` of the step."""
    gm = built.gm
    rename = rename or {}
    back = {v: k for k, v in rename.items()}
    if id(gm) not in _BATCHED:
        _BATCHED[id(gm)] = (gm, jax.jit(jax.vmap(gm._raw_step_fn, in_axes=(0, None, 0))))
    step = _BATCHED[id(gm)][1]
    state, p = _batched_start(built, values_list, rename)
    ext = gm._default_external_inputs()
    out = [[] for _ in values_list]
    pres = [None] * len(values_list)
    saved = gm._state
    try:
        for _ in range(steps):
            state = step(state, ext, p)
            for i in range(len(values_list)):
                gm._state = jax.tree.map(lambda x, i=i: x[i], state)
                snap = ct._snapshot(gm, back)
                diag = gm.coupling_diagnostics()
                reports = {gi: dict(diag[built.topo.group_key(gi)])
                           for gi in range(len(built.topo.groups))
                           if built.topo.group_key(gi) in diag}
                out[i].append(ct.Step(pres[i], snap, reports, {}))
                pres[i] = snap
    finally:
        gm._state = saved
    return out


def _restarted_runs(built, values_list, steps, rename):
    """Each draw run two steps, saved, run on, reset, loaded and run again."""
    out = []
    for v in values_list:
        straight = ct.run(built, v, steps + 2, rename=rename)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.npz"
            first = ct.run(built, v, 2, rename=rename)      # leaves the graph at step 2
            built.gm.save_state(path)
            built.gm.reset_state()
            built.gm.load_state(path)
            assert not cg.bitwise_differences(first[-1].state, straight[1].state)
            back = {b: a for a, b in (rename or {}).items()}
            after = []
            for k in range(steps):
                built.gm.step(params=ct.params_for(built, v, rename))
                snap = ct._snapshot(built.gm, back)
                diag = built.gm.coupling_diagnostics()
                reports = {gi: dict(diag[built.topo.group_key(gi)])
                           for gi in range(len(built.topo.groups))
                           if built.topo.group_key(gi) in diag}
                after.append(ct.Step(None, snap, reports, {}))
        for k, (a, b) in enumerate(zip(straight[2:], after)):
            diff = cg.bitwise_differences(a.state, b.state)
            assert not diff, f"the run after the checkpoint restart left the straight one: {diff}"
        out.append(after)
    return out


def _runs_in(domain, built, values_list, rename=None) -> list:
    """One trajectory per draw, in *domain*."""
    steps = _domain_steps(domain)
    with _x64(domain == "f64"):
        if domain == "vmap":
            runs = _batched_runs(built, values_list, steps, rename)
        elif domain == "checkpoint_restart":
            runs = _restarted_runs(built, values_list, steps, rename)
        else:
            runs = [ct.run(built, v, steps, rename=rename) for v in values_list]
    _assert_in_domain(domain, built, runs)
    return runs


def _assert_in_domain(domain, built, runs):
    """The premise: the graph is in *domain* (a fixture that fell back to the
    default lane would check float32 single-rate twice)."""
    gm = built.gm
    if domain == "f32":
        return
    assert bool(gm._is_multirate) == (domain == "multi_rate"), "multi-rate premise"
    groups = gm._committed_coupling_groups.values()
    assert all(g.subcycling == (domain == "sub_cycled") for g in groups), "sub-cycling premise"
    assert all((g.predictor == "quadratic") == (domain == "predictors_warm_starts")
               for g in groups), "predictor premise"
    want = jnp.dtype(_domain_dtype(domain))
    for run in runs:
        for step in run:
            for fields in step.state.values():
                assert fields["x"].dtype == want, (domain, fields["x"].dtype)


def test_each_member_of_a_vmap_batch_starts_from_its_own_draw():
    """The ``vmap`` domain's premise: member *i* of the batch is draw *i*.

    ``set_initial`` rewrites the graph's state dictionary in place, so a
    batch stacked from the live dictionary started every member from the
    last draw, and the domain compared three runs from one start."""
    topo, knobs, built = _built_in("vmap", "chain-into-ring", 0)
    rename = _renaming(topo, 11)
    _rt, _rk, renamed = _built_in("vmap", "chain-into-ring", 0, rename=rename)
    values = _domain_values(topo, knobs, "vmap")
    starts = [{n: np.asarray(v["x0"]) for n, v in draw["nodes"].items()} for draw in values]
    for i, a in enumerate(starts):
        for b in starts[i + 1:]:
            assert any(not np.array_equal(a[n], b[n]) for n in a), (
                "premise: the draws start apart")
    for what, graph, names in (("as built", built, {}), ("renamed", renamed, dict(rename))):
        state, _params = _batched_start(graph, values, names)
        for i, start in enumerate(starts):
            for n, x0 in start.items():
                got = np.asarray(state[names.get(n, n)]["x"][i])
                assert np.array_equal(got, x0.astype(got.dtype)), (
                    f"{what}: member {i} of the batch does not start node {n} from draw "
                    f"{i}: {got} against {x0}")


def _started_elsewhere(values: dict) -> dict:
    """*values* with every node's start moved by a part in 1024."""
    return {**values, "nodes": {
        n: {**v, "x0": (np.asarray(v["x0"]) * (1 + 2.0 ** -10)).astype(np.asarray(v["x0"]).dtype)}
        for n, v in values["nodes"].items()}}


def test_a_fault_in_one_member_of_a_vmap_batch_fails_that_member_alone():
    """The ``vmap`` comparisons can fail, and member by member: of two
    batches that differ in one draw's start, the comparison fails for that
    member and holds, bit for bit, for the other two.  (Stacked from the
    live dictionary, a fault in the first or second draw reached no member
    and one in the last reached all three.)"""
    topo, knobs, built = _built_in("vmap", "chain-into-ring", 0)
    values = _domain_values(topo, knobs, "vmap")
    clean = _runs_in("vmap", built, values)
    for member in range(len(values)):
        faulty = list(values)
        faulty[member] = _started_elsewhere(values[member])
        for k, (a, b) in enumerate(zip(clean, _runs_in("vmap", built, faulty))):
            what = f"draw {k} beside a fault in draw {member}"
            if k == member:
                with pytest.raises(AssertionError, match="states differ"):
                    assert_same_runs(a, b, what=what)
            else:
                assert_same_runs(a, b, what=what)


# Per push: tests/property/test_differential_coupling_topologies.py::test_renaming_a_named_topology_changes_nothing_in_every_domain
# (on chain-into-ring, in every domain; the slow structures are the same check)
@pytest.mark.parametrize("name", _DOMAIN_NAMES)
@pytest.mark.parametrize("domain", DOMAINS)
def test_renaming_a_named_topology_changes_nothing_in_every_domain(domain, name):
    """CPL-180 in *domain*: every node renamed, bit-identical states and reports."""
    topo, knobs, built = _built_in(domain, name, 0)
    rename = _renaming(topo, 11)
    _rt, _rk, renamed = _built_in(domain, name, 0, rename=rename)
    values = _domain_values(topo, knobs, domain)
    for k, (a, b) in enumerate(zip(_runs_in(domain, built, values),
                                   _runs_in(domain, renamed, values, rename=dict(rename)))):
        assert_same_runs(a, b, what=f"{domain}/{name} draw {k} renamed")


# Per push: tests/property/test_differential_coupling_topologies.py::test_the_build_order_of_a_named_topology_changes_nothing_in_every_domain
# (on chain-into-ring, in every domain; the slow structures are the same check)
@pytest.mark.parametrize("name", _DOMAIN_NAMES)
@pytest.mark.parametrize("domain", DOMAINS)
def test_the_build_order_of_a_named_topology_changes_nothing_in_every_domain(domain, name):
    """CPL-181 and CPL-182 in *domain*: nodes and edges added in another order
    (the documented exceptions kept), bit-identical states and reports.
    Configuration 2 puts Aitken and the interface norm on ``chain-into-ring``,
    whose residual used to sum in the edges' insertion order (CPL-182)."""
    topo, knobs, built = _built_in(domain, name, 2)
    if name == "chain-into-ring":
        assert any(g.get("convergence_norm") == "interface" for g in knobs), (
            "premise: CPL-182's interface norm is on this structure's group")
    node_order, edge_order = _orders(topo, 5)
    _t, _k, reordered = _built_in(domain, name, 2, node_order=node_order,
                                  edge_order=edge_order)
    assert reordered.gm.schedule != built.gm.schedule or edge_order != tuple(
        range(len(topo.edges))), "the permutation must reach the graph"
    values = _domain_values(topo, knobs, domain)
    for k, (a, b) in enumerate(zip(_runs_in(domain, built, values),
                                   _runs_in(domain, reordered, values))):
        assert_same_runs(a, b, what=f"{domain}/{name} draw {k} reordered")


# Per push: tests/property/test_differential_coupling_topologies.py::test_an_identity_relay_on_a_named_topology_moves_no_bit_in_every_domain
# (on chain-into-ring, in every domain; the slow structures are the same check)
@pytest.mark.parametrize("name", _DOMAIN_NAMES)
@pytest.mark.parametrize("domain", DOMAINS)
def test_an_identity_relay_on_a_named_topology_moves_no_bit_in_every_domain(domain, name):
    """CPL-183's strongest statement in *domain*: Gauss-Seidel, no acceleration,
    a criterion only exact stationarity meets; every other node's state is
    bit-identical with the relay, and the relay holds what it relays."""
    topo = _domain_topology(NAMED[name], domain)
    k = _relayable(topo)
    t0, knobs, plain = _built_in(domain, name, 0, constant="gauss-seidel")
    t1, _k1, relayed = _built_in(domain, name, 0, relay=k, constant="gauss-seidel")
    values = _domain_values(t0, knobs, domain)
    with _x64(domain == "f64"):
        relay_values = [ct.relay_values(t0, v, t1, k) for v in values]
    others = list(t0.names)
    for d, (a, b) in enumerate(zip(_runs_in(domain, plain, values),
                                   _runs_in(domain, relayed, relay_values))):
        for step, (sa, sb) in enumerate(zip(a, b), start=1):
            diff = cg.bitwise_differences({n: sa.state[n] for n in others},
                                          {n: sb.state[n] for n in others})
            assert not diff, f"{domain}/{name} draw {d} step {step}: the relay moved {diff}"


#: The structures whose groups are each a strongly connected component of
#: their own, so their members' build order is free to move (in the other
#: two an outside node or a second group shares the component, and the
#: order picks the edge read late: CPL-181's exclusion).
_FREE_MEMBERS = ("chain-into-ring", "star-and-ungrouped-ring")


def _member_orders(topo, seed):
    """A node order that moves the groups' members among themselves too."""
    keep_nodes, _keep_edges = ct.invariant_orders(topo)
    keep = [k for k in keep_nodes if not any(set(k) == set(g) for g in topo.groups)]
    rng = np.random.default_rng(seed)
    for _ in range(50):
        order = tuple(ct.ordered_permutation(topo.names, keep, rng))
        if any([m for m in order if m in g] != list(g) for g in topo.groups):
            return order
    raise AssertionError("no draw moved the members' order")


@pytest.mark.parametrize("name", _FREE_MEMBERS)
@pytest.mark.parametrize("domain", ("f32",) + DOMAINS)
def test_jacobi_under_a_constant_iterator_ignores_the_members_build_order(domain, name):
    """CPL-078: under ``iteration_mode="jacobi"`` with no acceleration every
    member reads the stored previous iterate, so the order the members were
    added in -- free to move here -- cannot reach the states or the
    verdicts; with a criterion only exact stationarity meets, both builds
    run the same passes and return the same bits.  The norm sums the
    members in the schedule's order, so the residual agrees to round-off
    (and the amplification and estimate derived from its ratios move with
    it), as documented."""
    domain_ = domain
    topo = _domain_topology(NAMED[name], domain_)
    order = _member_orders(topo, 7)
    _t, knobs, built = _built_in(domain_, name, 0, constant="jacobi")
    _t2, _k2, moved = _built_in(domain_, name, 0, node_order=order, constant="jacobi")
    values = _domain_values(topo, knobs, domain_)
    eps = float(jnp.finfo(jnp.dtype(_domain_dtype(domain_))).eps)
    for k, (a, b) in enumerate(zip(_runs_in(domain_, built, values),
                                   _runs_in(domain_, moved, values))):
        for step, (sa, sb) in enumerate(zip(a, b), start=1):
            where = f"{domain}/{name} draw {k} step {step} members moved"
            diff = cg.bitwise_differences(sa.state, sb.state)
            assert not diff, f"{where}: states differ in {diff}"
            for gi in sa.reports:
                ra, rb = sa.reports[gi], sb.reports[gi]
                assert (ra["iterations"], ra["converged"]) == (rb["iterations"],
                                                               rb["converged"]), (where, ra, rb)
                # The norm sums the members in the schedule's order: the
                # residual -- and what is derived from its ratios -- may
                # move in its last bits, nothing else.
                assert ra["residual"] == pytest.approx(rb["residual"], rel=16 * eps,
                                                       abs=1e-300), (where, ra, rb)


@pytest.mark.xfail(strict=True, raises=TypeError, reason=(
    "MADD-ANO-171: save_state cannot write a typed PRNG key leaf (np.asarray of a key "
    "array raises TypeError), so a graph holding one cannot be checkpointed; deferred "
    "to 0.5.0"))
def test_a_structure_with_a_typed_prng_key_survives_a_checkpoint_restart():
    """A typed key is a supported leaf (CPL-145: keys travel as their uint32 data),
    and ``save_state`` persists "all node states": the named structure with key
    leaves in and around its group, saved after two steps and loaded, must
    step on as the uninterrupted run does.  The checkpoint domain above runs
    the structures without their key leaves until it can."""
    topo, knobs, built = _built("nested-loop-additive", 0)
    assert any("key" in nd.leaves for nd in topo.nodes), "premise: a typed key leaf"
    values = ct.draw_values(topo, np.random.default_rng(3), 0.6,
                            group_cfgs=ct.group_cfgs_of(knobs))
    straight = ct.run(built, values, 3)
    ct.run(built, values, 2)            # leaves the graph at step 2
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.npz"
        built.gm.save_state(path)
        built.gm.reset_state()
        built.gm.load_state(path)
    built.gm.step(params=ct.params_for(built, values))
    after = ct._snapshot(built.gm, {})
    assert not cg.bitwise_differences(after, straight[-1].state)


#: The domains the monolithic reference models: a float64 graph, a sub-cycled
#: member (its update composed with the inputs held), and a predictor's other
#: starting guess.
_REFERENCE_DOMAINS = ("f64", "sub_cycled", "predictors_warm_starts")


# Per push: tests/property/test_differential_coupling_topologies.py::test_an_identity_relay_keeps_a_named_topology_on_its_reference_in_every_domain
# (on chain-into-ring, in each domain; the slow structures are the same check)
@pytest.mark.parametrize("name", _DOMAIN_NAMES)
@pytest.mark.parametrize("domain", _REFERENCE_DOMAINS)
def test_an_identity_relay_keeps_a_named_topology_on_its_reference_in_every_domain(
        domain, name):
    """CPL-183 under configuration 1 (IQN-ILS under Jacobi and the mixed norm
    first) in *domain*: the relayed graph reproduces the exact solve of the
    same structure, every group within what its reported residual allows."""
    topo = _domain_topology(NAMED[name], domain)
    k = _relayable(topo)
    t1, knobs, relayed = _built_in(domain, name, 1, relay=k)
    base_knobs = _domain_knobs(ct.topology_knobs(topo, 1), domain)
    for values in _domain_values(topo, base_knobs, domain):
        with _x64(domain == "f64"):
            rv = ct.relay_values(topo, values, t1, k)
        traj, = _runs_in(domain, relayed, [rv])
        with _x64(domain == "f64"):
            assert_reproduces_the_reference(t1, knobs, relayed, rv, traj=traj,
                                            dtype=_domain_dtype(domain))


@pytest.mark.parametrize("domain", _REFERENCE_DOMAINS)
def test_the_interface_norm_on_a_mapped_ring_reproduces_its_reference_in_every_domain(domain):
    """The mapped ring under the interface norm in each domain the reference
    models: float64, a sub-cycled member and a predictor's starting guess.
    (The other domains hold the same structure and configuration to the
    build-order invariance above.)"""
    topo, knobs, built = _built_in(domain, _MAPPED_RING, _INTERFACE_CHOICE)
    _assert_the_interface_norm_reads_a_mapped_edge(topo, knobs)
    for values in _domain_values(topo, knobs, domain):
        traj, = _runs_in(domain, built, [values])
        with _x64(domain == "f64"):
            assert_reproduces_the_reference(topo, knobs, built, values, traj=traj,
                                            dtype=_domain_dtype(domain))


# ---------------------------------------------------------------------------
# Sparse mapped edges in every domain
#
# Three statements.  The monolithic reference, in the domains it models.
# The dense twin: the same structure with dense mapped edges holding the
# same matrices, on the same draws, under a constant iterator (Gauss-Seidel,
# no acceleration, up to eight passes, stopping only at exact
# stationarity), so that no convergence criterion decides how far each
# graph goes and nothing but rounding separates them.  And two sparse
# builds of one structure against each other, bit for bit.
#
# The twin comparison is to rounding and not exact, even for one entry per
# target and weights that round nothing.  An earlier form of this test
# asked for equal states and failed, by one rounding, on the structure with
# three additive mapped edges into one port: the two compiled programs are
# different programs, and the compiler evaluates the arithmetic around a
# mapping differently in each (a product fused with the addition next to
# it in one and not in the other, for one).  A mapping applied on its own
# is exact for one entry per row (``tests/core/test_sparse_mapping.py``);
# a step is not.  It is not run in the 16-bit dtypes, where a rounding is
# a percent of the state and the comparison could not tell a wrong weight
# from one; there the mapping alone is held to its bound, and two sparse
# builds to each other.
# ---------------------------------------------------------------------------

#: The domains the dense twin is compared in.
_TWIN_DOMAINS = ("f32", "f64", "multi_rate", "sub_cycled", "vmap",
                 "predictors_warm_starts", "checkpoint_restart")
#: How far a sparse graph may sit from its dense twin, in epsilons of the
#: dtype times the largest state entry.  The largest gap measured over
#: every structure, sparse kind and twin domain was 7.5 (jaxlib 0.11.0);
#: this leaves a factor of thirty for another compiler's choices.
_TWIN_EPSILONS = 256


def _twin_cases():
    """Per push: the ragged pattern in both layouts on the structure with
    wide interfaces, in every twin domain.  Slow: every row full there, and
    the ragged pattern in both layouts on the other structures."""
    cases = []
    for domain in _TWIN_DOMAINS:
        for name in sorted(STRUCTURES):
            for kind in _SPARSE_REFERENCE_KINDS:
                if name not in MAPPED and kind == "sparse-full":
                    continue
                per_push = name in MAPPED and kind != "sparse-full"
                cases.append(pytest.param(
                    domain, name, kind, id=f"{domain}-{name}-{kind}",
                    marks=() if per_push else pytest.mark.slow))
    return cases


def _largest_gap(a, b) -> float:
    """The largest difference between two runs' float states, over the
    largest entry of the first, step by step; every other leaf must be
    equal (they do not depend on the mapping)."""
    worst = 0.0
    for sa, sb in zip(a, b):
        scale = max(float(np.max(np.abs(np.asarray(f["x"], np.float64))))
                    for f in sa.state.values())
        for node, fields in sa.state.items():
            for field, value in fields.items():
                other = sb.state[node][field]
                assert other.dtype == value.dtype and other.shape == value.shape
                if field == "x":
                    gap = np.max(np.abs(np.asarray(value, np.float64)
                                        - np.asarray(other, np.float64)), initial=0.0)
                    worst = max(worst, float(gap) / scale)
                else:
                    assert np.array_equal(value, other), f"{node}.{field}"
    return worst


# Per push: tests/property/test_differential_coupling_topologies.py::test_a_sparse_mapped_topology_steps_as_its_dense_twin_to_rounding_in_every_domain
# (the ragged pattern in both layouts on mapped-ring-wide, in every twin
# domain; the slow cases are the same check on the other structures)
@pytest.mark.parametrize("domain, name, mapping_kind", _twin_cases())
def test_a_sparse_mapped_topology_steps_as_its_dense_twin_to_rounding_in_every_domain(
        domain, name, mapping_kind):
    """Sparse equals dense, to rounding: the same structure built with
    dense mapped edges and with sparse ones holding the same matrices, on
    the same draws and under a constant iterator, returns the same states
    to within a few hundred epsilons of the largest entry -- in float32
    and float64, on a multi-rate graph, in a sub-cycled group, under
    ``jax.vmap``, with a predictor's warm start, and across a checkpoint
    restart (whose archive carries the sparse weights and their pattern's
    digest, and which must resume bit for bit).  Mapped matrices a tenth
    off are far outside that.  (The two may stop a pass apart: each stops
    when its own iterate stops moving at all.)"""
    topo, knobs, dense = _built_in(domain, name, 0, constant="gauss-seidel")
    _t, _k, sparse = _built_in(domain, name, 0, constant="gauss-seidel",
                               mapping_kind=mapping_kind)
    assert sparse.slots and not dense.slots
    assert {s.scatter for s in sparse.slots.values()} == {mapping_kind.endswith("scatter")}
    values = _domain_values(topo, knobs, domain, mapping_kind)
    eps = float(jnp.finfo(jnp.dtype(_domain_dtype(domain))).eps)
    dense_runs = _runs_in(domain, dense, values)
    for k, (a, b) in enumerate(zip(dense_runs, _runs_in(domain, sparse, values))):
        gap = _largest_gap(a, b)
        assert gap <= _TWIN_EPSILONS * eps, (
            f"{domain}/{name}/{mapping_kind} draw {k}: the sparse graph is {gap / eps:.0f} "
            f"epsilons from its dense twin")
    # The premise: the comparison can see a wrong operator.  The mapped
    # matrices a tenth off move the dense graph itself far outside the
    # allowance: the states depend on them.
    with _x64(domain == "f64"):
        off = {**values[0], "H": {i: np.asarray(H * H.dtype.type(1.1), H.dtype)
                                  for i, H in values[0]["H"].items()}}
    moved = _largest_gap(dense_runs[0], _runs_in(domain, dense, [off])[0])
    assert moved > 16 * _TWIN_EPSILONS * eps, (
        f"premise: mapped matrices a tenth off move the states by {moved / eps:.0f} epsilons")


def _sparse_domain_reference_cases():
    """Per push the ragged pattern in both layouts on the structure with
    wide interfaces; every row full, and the other structures, are slow."""
    cases = []
    for domain in _REFERENCE_DOMAINS:
        for kind in _SPARSE_REFERENCE_KINDS:
            for name in sorted(STRUCTURES):
                per_push = name in MAPPED and kind != "sparse-full"
                cases.append(pytest.param(
                    domain, name, kind, id=f"{domain}-{name}-{kind}",
                    marks=() if per_push else pytest.mark.slow))
    return cases


# Per push: tests/property/test_differential_coupling_topologies.py::test_a_named_topology_with_sparse_mapped_edges_reproduces_its_reference_in_every_domain
# (the ragged pattern in both layouts on mapped-ring-wide, in each domain
# the reference models; the slow cases are the same check)
@pytest.mark.parametrize("domain, name, mapping_kind", _sparse_domain_reference_cases())
def test_a_named_topology_with_sparse_mapped_edges_reproduces_its_reference_in_every_domain(
        domain, name, mapping_kind):
    """The dense oracle over sparse mapped edges with several entries in a
    row, in the domains the monolithic reference models: a float64 graph, a
    sub-cycled member, and a predictor's other starting guess."""
    topo, knobs, built = _built_in(domain, name, 1, mapping_kind=mapping_kind)
    assert built.slots, "premise: sparse mapped edges"
    for values in _domain_values(topo, knobs, domain, mapping_kind):
        traj, = _runs_in(domain, built, [values])
        with _x64(domain == "f64"):
            assert_reproduces_the_reference(topo, knobs, built, values, traj=traj,
                                            dtype=_domain_dtype(domain))


def _flat_gradient(tree) -> dict:
    return {jax.tree_util.keystr(path): np.asarray(leaf, np.float64)
            for path, leaf in jax.tree_util.tree_leaves_with_path(tree)}


@pytest.mark.parametrize("mapping_kind", ["sparse-ragged", "sparse-scatter"])
@pytest.mark.parametrize("domain", ["f32", "f64"])
def test_the_gradient_through_sparse_mapped_edges_is_the_dense_graphs(domain, mapping_kind):
    """``jax.grad`` of a loss over two steps of the structure with wide
    interfaces, with respect to every parameter leaf: the gradient to each
    node constant is the dense graph's, and the gradient to a sparse edge's
    ``W`` is the dense graph's gradient to ``H`` read through the edge's
    pattern -- zero in a padded slot.  Through the coupling group's
    implicit rule and through the staggered edges, in float32 and under
    x64; the two graphs converge to rounding of one another, so the
    gradients agree to the group's tolerance, not bit for bit."""
    name = "mapped-ring-wide"
    topo, knobs, dense = _built_in(domain, name, 0)
    _t, _k, sparse = _built_in(domain, name, 0, mapping_kind=mapping_kind)
    values = _domain_values(topo, knobs, domain, mapping_kind)[1]

    def gradient(built):
        with _x64(domain == "f64"):
            ct.set_initial(built, values)
            gm = built.gm
            state0, ext, step = gm._state, gm._default_external_inputs(), gm._raw_step_fn

            def loss(params):
                state = step(step(state0, ext, params), ext, params)
                return sum(jnp.sum(state[n]["x"] ** 2) for n in topo.names)

            return jax.jit(jax.grad(loss))(ct.params_for(built, values))

    g_dense, g_sparse = gradient(dense), gradient(sparse)
    rtol = 1e-7 if domain == "f64" else 2e-3
    nodes_d, nodes_s = _flat_gradient(g_dense["nodes"]), _flat_gradient(g_sparse["nodes"])
    assert nodes_d.keys() == nodes_s.keys() and nodes_d
    scale = max(float(np.max(np.abs(leaf))) for leaf in nodes_d.values())
    assert scale > 0.0
    for key, leaf in nodes_d.items():
        np.testing.assert_allclose(nodes_s[key], leaf, rtol=rtol, atol=rtol * scale,
                                   err_msg=f"{domain}/{mapping_kind} d/d{key}")
    moved = 0
    for i, key in sparse.mapping_keys.items():
        slots = sparse.slots[i]
        dH = np.asarray(g_dense["mappings"][dense.mapping_keys[i]]["H"], np.float64)
        dW = np.asarray(g_sparse["mappings"][key]["W"], np.float64)
        np.testing.assert_allclose(dW, slots.weights(dH), rtol=rtol, atol=rtol * scale,
                                   err_msg=f"{domain}/{mapping_kind} d/dW of {key}")
        assert not dW[~slots.valid].any(), "a padded weight has no derivative"
        moved += int(np.count_nonzero(dW))
    assert moved > 0, "premise: the loss depends on the mapping weights"


def _sparse_invariance_cases():
    """Per push: the 16-bit dtypes on the structure with wide interfaces,
    in both layouts -- the domains the dense twin is not compared in.  Slow:
    the other six domains there, and all eight on ``chain-into-ring``."""
    cases = []
    for domain in DOMAINS:
        for kind in ("sparse-ragged", "sparse-scatter"):
            for name in ("chain-into-ring", *sorted(MAPPED)):
                per_push = name in MAPPED and domain in ("bfloat16", "float16")
                cases.append(pytest.param(
                    domain, kind, name, id=f"{domain}-{kind}-{name}",
                    marks=() if per_push else pytest.mark.slow))
    return cases


# Per push: tests/property/test_differential_coupling_topologies.py::test_renaming_and_reordering_a_sparse_mapped_topology_changes_nothing_in_every_domain
# (bfloat16 and float16 on mapped-ring-wide, in both layouts; the slow cases
# are the same check in the other domains and on chain-into-ring)
@pytest.mark.parametrize("domain, mapping_kind, name", _sparse_invariance_cases())
def test_renaming_and_reordering_a_sparse_mapped_topology_changes_nothing_in_every_domain(
        domain, mapping_kind, name):
    """CPL-180 and CPL-181 over sparse mapped edges: every node renamed, and
    nodes and edges added in another order, bit-identical states and reports
    (a sparse edge's pattern belongs to the structure, not to a name or to
    the order the edge was added in)."""
    topo, knobs, built = _built_in(domain, name, 0, mapping_kind=mapping_kind)
    rename = _renaming(topo, 11)
    _rt, _rk, renamed = _built_in(domain, name, 0, rename=rename, mapping_kind=mapping_kind)
    node_order, edge_order = _orders(topo, 5)
    _t, _k, reordered = _built_in(domain, name, 0, node_order=node_order,
                                  edge_order=edge_order, mapping_kind=mapping_kind)
    values = _domain_values(topo, knobs, domain, mapping_kind)
    base = _runs_in(domain, built, values)
    for k, (a, b) in enumerate(zip(base, _runs_in(domain, renamed, values,
                                                  rename=dict(rename)))):
        assert_same_runs(a, b, what=f"{domain}/{name}/{mapping_kind} draw {k} renamed")
    for k, (a, b) in enumerate(zip(base, _runs_in(domain, reordered, values))):
        assert_same_runs(a, b, what=f"{domain}/{name}/{mapping_kind} draw {k} reordered")
