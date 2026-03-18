"""Tests for node-to-device partitioning."""

import pytest

from maddening.cloud.multigpu.partition import (
    assign_nodes_to_devices,
    coupling_colocation_rate,
)


class TestAssignNodesToDevices:
    def test_single_device(self):
        assignment = assign_nodes_to_devices(
            node_names=["a", "b", "c"],
            edges=[],
            coupling_groups=[],
            n_devices=1,
        )
        assert all(v == 0 for v in assignment.values())

    def test_coupled_nodes_colocated(self):
        assignment = assign_nodes_to_devices(
            node_names=["a", "b", "c", "d"],
            edges=[{"source_node": "a", "target_node": "b"}],
            coupling_groups=[{"a", "b"}],
            n_devices=2,
        )
        # a and b should be on the same device
        assert assignment["a"] == assignment["b"]

    def test_uncoupled_nodes_spread(self):
        assignment = assign_nodes_to_devices(
            node_names=["a", "b", "c", "d"],
            edges=[],
            coupling_groups=[],
            n_devices=2,
        )
        # With no coupling, nodes should be spread across devices
        devices_used = set(assignment.values())
        assert len(devices_used) == 2

    def test_all_nodes_assigned(self):
        names = ["n1", "n2", "n3", "n4", "n5"]
        assignment = assign_nodes_to_devices(
            node_names=names,
            edges=[],
            coupling_groups=[{"n1", "n2"}, {"n3", "n4"}],
            n_devices=2,
        )
        assert set(assignment.keys()) == set(names)

    def test_load_balancing_with_sizes(self):
        assignment = assign_nodes_to_devices(
            node_names=["big", "small1", "small2"],
            edges=[],
            coupling_groups=[],
            n_devices=2,
            state_sizes={"big": 1000, "small1": 10, "small2": 10},
        )
        # big should be alone, smalls together
        assert assignment["small1"] == assignment["small2"]
        assert assignment["big"] != assignment["small1"]


class TestCouplingColocationRate:
    def test_all_colocated(self):
        assignment = {"a": 0, "b": 0, "c": 0}
        rate = coupling_colocation_rate(
            assignment, [{"a", "b", "c"}],
        )
        assert rate == 1.0

    def test_none_colocated(self):
        assignment = {"a": 0, "b": 1}
        rate = coupling_colocation_rate(
            assignment, [{"a", "b"}],
        )
        assert rate == 0.0

    def test_partial_colocation(self):
        assignment = {"a": 0, "b": 0, "c": 1}
        rate = coupling_colocation_rate(
            assignment, [{"a", "b", "c"}],
        )
        # 3 pairs: (a,b)=same, (a,c)=diff, (b,c)=diff → 1/3
        assert abs(rate - 1.0 / 3.0) < 1e-6

    def test_no_coupling_groups(self):
        rate = coupling_colocation_rate({"a": 0}, [])
        assert rate == 1.0

    def test_threshold_80_percent(self):
        """Partition quality: coupling group co-location >= 80%."""
        assignment = assign_nodes_to_devices(
            node_names=["a", "b", "c", "d", "e", "f"],
            edges=[],
            coupling_groups=[{"a", "b", "c"}, {"d", "e", "f"}],
            n_devices=2,
        )
        rate = coupling_colocation_rate(
            assignment, [{"a", "b", "c"}, {"d", "e", "f"}],
        )
        assert rate >= 0.8, f"Co-location rate {rate:.2f} < 0.80 threshold"
