"""Tests for scheduling utilities (topological sort, cycle detection, back-edges)."""

from maddening.core.edge import EdgeSpec
from maddening.core.schedule import topological_sort, detect_cycles, identify_back_edges


class TestTopologicalSort:
    def test_linear_chain(self):
        """A -> B -> C should produce [A, B, C]."""
        nodes = ["A", "B", "C"]
        edges = [
            EdgeSpec("A", "B", "x", "x"),
            EdgeSpec("B", "C", "x", "x"),
        ]
        order = topological_sort(nodes, edges)
        assert order.index("A") < order.index("B") < order.index("C")

    def test_diamond(self):
        """Diamond: A -> {B, C} -> D. A must come first, D last."""
        nodes = ["A", "B", "C", "D"]
        edges = [
            EdgeSpec("A", "B", "x", "x"),
            EdgeSpec("A", "C", "x", "x"),
            EdgeSpec("B", "D", "x", "x"),
            EdgeSpec("C", "D", "x", "x"),
        ]
        order = topological_sort(nodes, edges)
        assert order[0] == "A"
        assert order[-1] == "D"

    def test_no_edges(self):
        """All nodes present even with no edges."""
        nodes = ["X", "Y", "Z"]
        order = topological_sort(nodes, [])
        assert set(order) == {"X", "Y", "Z"}

    def test_single_node(self):
        order = topological_sort(["solo"], [])
        assert order == ["solo"]

    def test_cycle_all_nodes_present(self):
        """Even with a cycle, all nodes must appear in the output."""
        nodes = ["A", "B"]
        edges = [
            EdgeSpec("A", "B", "x", "x"),
            EdgeSpec("B", "A", "x", "x"),
        ]
        order = topological_sort(nodes, edges)
        assert set(order) == {"A", "B"}

    def test_self_loop_ignored(self):
        """Self-loops (A -> A) should be ignored by adjacency builder."""
        nodes = ["A", "B"]
        edges = [
            EdgeSpec("A", "A", "x", "x"),
            EdgeSpec("A", "B", "x", "x"),
        ]
        order = topological_sort(nodes, edges)
        assert order.index("A") < order.index("B")

    def test_deterministic(self):
        """Same input should always produce the same output."""
        nodes = ["C", "B", "A"]
        edges = [EdgeSpec("A", "B", "x", "x")]
        order1 = topological_sort(nodes, edges)
        order2 = topological_sort(nodes, edges)
        assert order1 == order2


class TestDetectCycles:
    def test_no_cycle(self):
        nodes = ["A", "B", "C"]
        edges = [
            EdgeSpec("A", "B", "x", "x"),
            EdgeSpec("B", "C", "x", "x"),
        ]
        assert detect_cycles(nodes, edges) == []

    def test_simple_cycle(self):
        nodes = ["A", "B"]
        edges = [
            EdgeSpec("A", "B", "x", "x"),
            EdgeSpec("B", "A", "x", "x"),
        ]
        cycles = detect_cycles(nodes, edges)
        assert len(cycles) >= 1
        # Cycle should contain both A and B
        cycle_nodes = set()
        for c in cycles:
            cycle_nodes.update(c)
        assert "A" in cycle_nodes and "B" in cycle_nodes

    def test_three_node_cycle(self):
        nodes = ["A", "B", "C"]
        edges = [
            EdgeSpec("A", "B", "x", "x"),
            EdgeSpec("B", "C", "x", "x"),
            EdgeSpec("C", "A", "x", "x"),
        ]
        cycles = detect_cycles(nodes, edges)
        assert len(cycles) >= 1

    def test_no_false_positive_on_diamond(self):
        """A diamond (A -> B,C -> D) has no cycle."""
        nodes = ["A", "B", "C", "D"]
        edges = [
            EdgeSpec("A", "B", "x", "x"),
            EdgeSpec("A", "C", "x", "x"),
            EdgeSpec("B", "D", "x", "x"),
            EdgeSpec("C", "D", "x", "x"),
        ]
        assert detect_cycles(nodes, edges) == []


class TestIdentifyBackEdges:
    def test_no_back_edges(self):
        schedule = ["A", "B", "C"]
        edges = [
            EdgeSpec("A", "B", "x", "x"),
            EdgeSpec("B", "C", "x", "x"),
        ]
        assert identify_back_edges(schedule, edges) == []

    def test_back_edge_detected(self):
        schedule = ["A", "B"]
        edges = [
            EdgeSpec("A", "B", "x", "x"),
            EdgeSpec("B", "A", "x", "x"),  # back-edge
        ]
        back = identify_back_edges(schedule, edges)
        assert len(back) == 1
        assert back[0].source_node == "B"
        assert back[0].target_node == "A"

    def test_self_loop_is_back_edge(self):
        schedule = ["A"]
        edges = [EdgeSpec("A", "A", "x", "x")]
        back = identify_back_edges(schedule, edges)
        assert len(back) == 1


# ---------------------------------------------------------------------------
# A cycle's downstream nodes are scheduled after it, whatever the build order
# (CPL-025 in docs/validation/coupling_claims.yaml)
# ---------------------------------------------------------------------------

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from maddening.core.schedule import find_strongly_connected_components  # noqa: E402
from tests.conftest import EXAMPLES_CHEAP  # noqa: E402


def _edges(pairs):
    return [EdgeSpec(a, b, "x", f"in_{a}") for a, b in pairs]


class TestCyclesDownstreamNodes:
    def test_a_node_downstream_of_a_cycle_added_first_is_scheduled_after_it(self):
        """The defect: ``sink`` reads ``b`` (an edge on no cycle) and was added first."""
        edges = _edges([("a", "b"), ("b", "a"), ("b", "sink")])
        order = topological_sort(["sink", "a", "b"], edges)
        assert order == ["a", "b", "sink"]
        assert identify_back_edges(order, edges) == [edges[1]]   # only the cycle's own

    def test_a_cycle_keeps_its_members_order(self):
        """The members of a cycle keep their build order (a Gauss-Seidel sweep follows it)."""
        edges = _edges([("a", "b"), ("b", "c"), ("c", "a"), ("c", "d")])
        assert topological_sort(["d", "c", "a", "b"], edges) == ["c", "a", "b", "d"]

    def test_a_component_is_placed_whole_ahead_of_its_reader(self):
        """A downstream cycle's members never come between an upstream cycle's."""
        edges = _edges([("a0", "a1"), ("a1", "a0"), ("b0", "b1"), ("b1", "b0"), ("a1", "b0")])
        for order in (["b1", "b0", "a0", "a1"], ["b0", "a0", "b1", "a1"],
                      ["a0", "b0", "a1", "b1"]):
            want = [n for n in order if n[0] == "a"] + [n for n in order if n[0] == "b"]
            assert topological_sort(order, edges) == want, order

    def test_a_build_order_that_was_already_right_is_unchanged(self):
        """Independent cycles stay interleaved where nothing reads across them."""
        edges = _edges([("a0", "a1"), ("a1", "a0"), ("b0", "b1"), ("b1", "b0"),
                        ("d", "a0"), ("a1", "s")])
        order = ["d", "a0", "a1", "b0", "b1", "s"]
        assert topological_sort(order, edges) == order


_NAMES = ["n0", "n1", "n2", "n3", "n4", "n5", "n6"]


@st.composite
def _graphs(draw):
    k = draw(st.integers(2, len(_NAMES)))
    names = _NAMES[:k]
    pairs = draw(st.lists(st.tuples(st.sampled_from(names), st.sampled_from(names)),
                          max_size=3 * k))
    order = draw(st.permutations(names))
    return list(order), _edges(pairs)


def _component_of(order, edges):
    comp = {n: n for n in order}
    for scc in find_strongly_connected_components(order, edges):
        for n in scc:
            comp[n] = min(scc)
    return comp


@settings(max_examples=EXAMPLES_CHEAP)
@given(_graphs())
def test_only_a_cycles_own_edges_point_backward(graph):
    """Every edge between two components points forward, whatever the build order."""
    order, edges = graph
    sched = topological_sort(order, edges)
    assert sorted(sched) == sorted(order)
    comp = _component_of(order, edges)
    for e in identify_back_edges(sched, edges):
        assert comp[e.source_node] == comp[e.target_node], (sched, e.key)


@settings(max_examples=EXAMPLES_CHEAP)
@given(_graphs())
def test_every_component_is_contiguous_and_keeps_its_build_order(graph):
    """A cycle's members sit together, in the order they were added."""
    order, edges = graph
    sched = topological_sort(order, edges)
    for scc in find_strongly_connected_components(order, edges):
        idx = sorted(sched.index(n) for n in scc)
        assert idx == list(range(idx[0], idx[0] + len(scc))), (sched, scc)
        assert [n for n in sched if n in scc] == [n for n in order if n in scc]


@settings(max_examples=EXAMPLES_CHEAP)
@given(_graphs())
def test_a_schedule_rebuilt_in_its_own_order_is_itself(graph):
    """A graph built in the order its schedule gives is scheduled the same way."""
    order, edges = graph
    sched = topological_sort(order, edges)
    assert topological_sort(sched, edges) == sched
