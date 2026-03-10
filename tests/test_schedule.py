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
