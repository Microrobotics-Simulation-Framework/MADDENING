"""Tests for CloudGroup orchestration."""

import pytest
from unittest.mock import MagicMock, patch

from maddening.cloud.group import (
    CloudGroup,
    GroupConfig,
    GroupFailureMode,
    SubgraphSpec,
)
from maddening.cloud.launcher import (
    CloudJob,
    CostPolicy,
    JobConfig,
    JobPhase,
    LaunchError,
)


def _make_spec(subgraph_id: str, **kwargs) -> SubgraphSpec:
    """Create a SubgraphSpec with defaults."""
    config = JobConfig(
        provider="runpod",
        gpu_type="RTX4090",
        cost=CostPolicy(max_cost_per_hour=2.0, max_total_budget=10.0),
        **kwargs,
    )
    return SubgraphSpec(
        subgraph_id=subgraph_id,
        job_config=config,
        zmq_ports={"state": 5555},
    )


class TestCloudGroupConstruction:
    def test_requires_specs(self):
        with pytest.raises(ValueError, match="at least one"):
            CloudGroup(specs=[], edges=[])

    def test_rank0_is_first_spec(self):
        group = CloudGroup(
            specs=[_make_spec("flow"), _make_spec("structure")],
            edges=[],
        )
        assert group._rank0_id == "flow"


class TestCloudGroupEnvInjection:
    def test_rank0_gets_coordinator_env(self):
        group = CloudGroup(
            specs=[_make_spec("flow"), _make_spec("structure")],
            edges=[{"source": "flow", "target": "structure",
                    "source_field": "p", "target_field": "f"}],
        )
        rank0_config = group._inject_rank0_env(group._specs[0])
        extra = getattr(rank0_config, "_extra_envs", {})
        assert extra["SUBGRAPH_ID"] == "flow"
        assert extra["IS_RANK0"] == "1"
        assert "EXPECTED_WORKERS" in extra
        assert "INTER_JOB_EDGES" in extra

    def test_worker_gets_coordinator_addr(self):
        group = CloudGroup(
            specs=[_make_spec("flow"), _make_spec("structure")],
            edges=[],
        )
        worker_config = group._inject_worker_env(
            group._specs[1], "10.0.0.1:5580",
        )
        extra = getattr(worker_config, "_extra_envs", {})
        assert extra["SUBGRAPH_ID"] == "structure"
        assert extra["COORDINATOR_ADDR"] == "10.0.0.1:5580"
        assert extra["IS_RANK0"] == "0"

    def test_reserved_envs_not_overwritten(self):
        """User envs should be preserved alongside injected ones."""
        spec = _make_spec("flow", envs={"MY_VAR": "hello"})
        group = CloudGroup(specs=[spec], edges=[])
        config = group._inject_rank0_env(spec)
        assert config.envs["MY_VAR"] == "hello"
        extra = getattr(config, "_extra_envs", {})
        assert extra["SUBGRAPH_ID"] == "flow"


class TestCloudGroupLifecycle:
    def test_teardown_all(self):
        group = CloudGroup(
            specs=[_make_spec("a"), _make_spec("b")],
            edges=[],
        )
        # Mock jobs
        job_a = MagicMock(spec=CloudJob)
        job_b = MagicMock(spec=CloudJob)
        group._jobs = {"a": job_a, "b": job_b}

        group.teardown_all()
        job_a.teardown.assert_called_once()
        job_b.teardown.assert_called_once()

    def test_teardown_one_requires_isolate(self):
        group = CloudGroup(
            specs=[_make_spec("a")],
            edges=[],
            group_config=GroupConfig(failure_mode=GroupFailureMode.TEARDOWN_ALL),
        )
        group._jobs = {"a": MagicMock(spec=CloudJob)}
        with pytest.raises(ValueError, match="ISOLATE"):
            group.teardown_one("a")

    def test_teardown_one_isolate_mode(self):
        group = CloudGroup(
            specs=[_make_spec("a"), _make_spec("b")],
            edges=[],
            group_config=GroupConfig(failure_mode=GroupFailureMode.ISOLATE),
        )
        job_a = MagicMock(spec=CloudJob)
        job_b = MagicMock(spec=CloudJob)
        group._jobs = {"a": job_a, "b": job_b}

        group.teardown_one("a")
        job_a.teardown.assert_called_once()
        job_b.teardown.assert_not_called()

    def test_cost_so_far(self):
        group = CloudGroup(specs=[_make_spec("a"), _make_spec("b")], edges=[])
        job_a = MagicMock(spec=CloudJob)
        job_b = MagicMock(spec=CloudJob)
        job_a.cost_so_far.return_value = 0.15
        job_b.cost_so_far.return_value = 0.20
        group._jobs = {"a": job_a, "b": job_b}

        assert group.cost_so_far() == pytest.approx(0.35)

    def test_from_cluster_names(self):
        with patch.object(CloudJob, "from_cluster_name") as mock_fcn:
            mock_fcn.return_value = MagicMock(spec=CloudJob)
            group = CloudGroup.from_cluster_names(
                {"flow": "cluster-1", "structure": "cluster-2"},
            )
        assert "flow" in group._jobs
        assert "structure" in group._jobs
        assert mock_fcn.call_count == 2
