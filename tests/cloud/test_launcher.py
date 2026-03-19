"""Tests for CloudLauncher, CloudJob, JobConfig, and cost guards."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from maddening.cloud.launcher import (
    CloudJob,
    CloudLauncher,
    CostLimitError,
    CostPolicy,
    CredentialError,
    JobConfig,
    JobPhase,
    LaunchError,
    _credential_context,
)
from maddening.cloud.providers import RunPodProvider


# -- Fixtures ----------------------------------------------------------

@pytest.fixture
def creds_file(tmp_path):
    """Write a minimal credentials file and return its path."""
    data = {
        "version": 1,
        "runpod": {"api_key": "rp_test_key"},
        "lambda_labs": {"api_key": "ll_test_key"},
    }
    path = tmp_path / "cloud_credentials.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


@pytest.fixture
def job_config_file(tmp_path):
    """Write a minimal job config YAML and return its path."""
    data = {
        "version": 1,
        "provider": "runpod",
        "gpu_type": "A4000",
        "gpu_count": 1,
        "use_spot": True,
        "region": "US",
        "disk_size": 50,
        "cost": {
            "max_cost_per_hour": 2.0,
            "max_total_budget": 5.0,
            "autostop_minutes": 15,
            "auto_teardown": True,
        },
        "container_image": "nvcr.io/nvidia/cuda:12.2.2-runtime-ubuntu22.04",
        "stream_preset": "standard",
    }
    path = tmp_path / "job_config.yaml"
    path.write_text(yaml.dump(data), encoding="utf-8")
    return path


# -- JobConfig tests ---------------------------------------------------

class TestJobConfig:
    def test_from_yaml(self, job_config_file):
        cfg = JobConfig.from_yaml(job_config_file)
        assert cfg.provider == "runpod"
        assert cfg.gpu_type == "A4000"
        assert cfg.cost.max_cost_per_hour == 2.0
        assert cfg.cost.autostop_minutes == 15
        assert cfg.cost.auto_teardown is True

    def test_from_yaml_missing_provider(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({"gpu_type": "A4000"}))
        with pytest.raises(ValueError, match="provider"):
            JobConfig.from_yaml(path)

    def test_from_yaml_missing_gpu_type(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(yaml.dump({"provider": "runpod"}))
        with pytest.raises(ValueError, match="gpu_type"):
            JobConfig.from_yaml(path)

    def test_from_dict_round_trip(self):
        cfg = JobConfig(provider="runpod", gpu_type="A4000")
        d = {
            "provider": cfg.provider,
            "gpu_type": cfg.gpu_type,
            "gpu_count": cfg.gpu_count,
            "use_spot": cfg.use_spot,
            "region": cfg.region,
            "disk_size": cfg.disk_size,
            "stream_preset": cfg.stream_preset,
        }
        cfg2 = JobConfig.from_dict(d)
        assert cfg2.provider == cfg.provider
        assert cfg2.gpu_type == cfg.gpu_type

    def test_reserved_env_var_raises(self):
        with pytest.raises(ValueError, match="COORDINATOR_ADDR"):
            JobConfig(
                provider="runpod", gpu_type="A4000",
                envs={"COORDINATOR_ADDR": "10.0.0.1:5580"},
            )

    def test_reserved_env_var_maddening_config(self):
        with pytest.raises(ValueError, match="MADDENING_CLOUD_CONFIG"):
            JobConfig(
                provider="runpod", gpu_type="A4000",
                envs={"MADDENING_CLOUD_CONFIG": "{}"},
            )


class TestCostPolicy:
    def test_defaults(self):
        p = CostPolicy()
        assert p.max_cost_per_hour == 2.0
        assert p.max_total_budget == 5.0
        assert p.autostop_minutes == 15
        assert p.auto_teardown is True


# -- CloudLauncher tests -----------------------------------------------

class TestCloudLauncherCredentials:
    def test_nonexistent_credentials_path(self, tmp_path):
        launcher = CloudLauncher(credentials_path=tmp_path / "nope.yaml")
        with pytest.raises(CredentialError, match="not found"):
            launcher.validate(JobConfig(provider="runpod", gpu_type="A4000"))

    def test_missing_provider_block(self, tmp_path):
        creds_path = tmp_path / "creds.yaml"
        creds_path.write_text(yaml.dump({"version": 1, "aws": {"key": "x"}}))
        launcher = CloudLauncher(credentials_path=creds_path)
        with pytest.raises(CredentialError, match="runpod"):
            launcher.validate(JobConfig(provider="runpod", gpu_type="A4000"))

    def test_unknown_provider(self, creds_file):
        launcher = CloudLauncher(credentials_path=creds_file)
        with pytest.raises(CredentialError, match="notacloud"):
            launcher.validate(
                JobConfig(provider="notacloud", gpu_type="A4000"),
            )


class TestCloudLauncherCostGuards:
    def test_hourly_cost_exceeds_limit(self, creds_file):
        launcher = CloudLauncher(credentials_path=creds_file)
        config = JobConfig(
            provider="runpod", gpu_type="A4000",
            cost=CostPolicy(max_cost_per_hour=0.10),
        )
        with patch.object(
            launcher, "_resolve_resources", return_value=("inst-1", 0.50),
        ), patch.object(
            launcher, "_get_budget_used", return_value=0.0,
        ):
            with pytest.raises(CostLimitError) as exc_info:
                launcher.validate(config)
            assert exc_info.value.guard_type == "hourly"
            assert exc_info.value.actual == 0.50
            assert exc_info.value.limit == 0.10

    def test_budget_exceeded(self, creds_file):
        launcher = CloudLauncher(credentials_path=creds_file)
        config = JobConfig(
            provider="runpod", gpu_type="A4000",
            cost=CostPolicy(max_total_budget=3.00),
        )
        with patch.object(
            launcher, "_resolve_resources", return_value=("inst-1", 0.20),
        ), patch.object(
            launcher, "_get_budget_used", return_value=4.50,
        ):
            with pytest.raises(CostLimitError) as exc_info:
                launcher.validate(config)
            assert exc_info.value.guard_type == "budget"

    def test_validate_returns_info(self, creds_file):
        launcher = CloudLauncher(credentials_path=creds_file)
        config = JobConfig(provider="runpod", gpu_type="A4000")
        with patch.object(
            launcher, "_resolve_resources",
            return_value=("1x_A4000_SECURE", 0.20),
        ), patch.object(
            launcher, "_get_budget_used", return_value=1.50,
        ):
            result = launcher.validate(config)
        assert result["provider"] == "runpod"
        assert result["instance_type"] == "1x_A4000_SECURE"
        assert result["hourly_cost"] == 0.20
        assert result["budget_used"] == 1.50
        assert result["budget_remaining"] == 3.50


class TestCloudLauncherLaunch:
    def test_launch_success(self, creds_file, tmp_path):
        import sys
        mock_sky = MagicMock()
        mock_handle = MagicMock()
        mock_handle.head_ip = "10.0.0.42"
        mock_sky.launch.return_value = "req-123"
        mock_sky.get.return_value = (1, mock_handle)
        mock_sky.stream_and_get.return_value = (1, mock_handle)
        mock_sky.clouds = MagicMock()
        mock_sky.RunPod = MagicMock  # Cloud class lookup fallback

        launcher = CloudLauncher(credentials_path=creds_file)
        config = JobConfig(provider="runpod", gpu_type="A4000")

        with patch.object(
            launcher, "_resolve_resources",
            return_value=("1x_A4000_SECURE", 0.20),
        ), patch.object(
            launcher, "_get_budget_used", return_value=0.0,
        ), patch.object(
            launcher, "_resolve_provider",
            return_value=(RunPodProvider(), {"api_key": "rp_test"}),
        ), patch.dict(sys.modules, {"sky": mock_sky}):
            job = launcher.launch(config)

        assert job.cluster_name.startswith("maddening-")
        assert job.phase == JobPhase.EXECUTING
        assert job.vm_ip == "10.0.0.42"
        assert job.ports == {}

    def test_dry_run(self, creds_file):
        import sys
        mock_sky = MagicMock()
        mock_sky.launch.return_value = "req-dry"
        mock_sky.clouds = MagicMock()
        mock_sky.RunPod = MagicMock

        launcher = CloudLauncher(credentials_path=creds_file)
        config = JobConfig(provider="runpod", gpu_type="A4000")

        with patch.object(
            launcher, "_resolve_resources",
            return_value=("1x_A4000_SECURE", 0.20),
        ), patch.object(
            launcher, "_get_budget_used", return_value=0.0,
        ), patch.object(
            launcher, "_resolve_provider",
            return_value=(RunPodProvider(), {"api_key": "rp_test"}),
        ), patch.dict(sys.modules, {"sky": mock_sky}):
            job = launcher.launch(config, dry_run=True)

        assert job.cluster_name == "dry-run"
        assert job.phase == JobPhase.DONE

    def test_launch_credential_cleanup_on_failure(self, creds_file, tmp_path):
        import sys
        mock_sky = MagicMock()
        mock_sky.launch.side_effect = RuntimeError("SkyPilot exploded")
        mock_sky.clouds = MagicMock()
        mock_sky.RunPod = MagicMock

        launcher = CloudLauncher(credentials_path=creds_file)
        config = JobConfig(provider="runpod", gpu_type="A4000")

        with patch.object(
            launcher, "_resolve_resources",
            return_value=("1x_A4000_SECURE", 0.20),
        ), patch.object(
            launcher, "_get_budget_used", return_value=0.0,
        ), patch.object(
            launcher, "_resolve_provider",
            return_value=(RunPodProvider(), {"api_key": "rp_test"}),
        ), patch.object(
            launcher, "_poll_until_up",
            side_effect=LaunchError("poll failed"),
        ), patch.dict(sys.modules, {"sky": mock_sky}):
            with pytest.raises(LaunchError, match="SkyPilot exploded"):
                launcher.launch(config)

        # Credential file should be cleaned up despite the exception


# -- CloudJob tests ----------------------------------------------------

class TestCloudJob:
    def test_teardown_twice_is_safe(self):
        job = CloudJob("dry-run")
        job.teardown()
        job.teardown()  # no exception

    def test_is_done(self):
        job = CloudJob("test-cluster")
        assert not job.is_done()
        job._phase = JobPhase.DONE
        assert job.is_done()
        job._phase = JobPhase.FAILED
        assert job.is_done()

    def test_from_cluster_name(self):
        job = CloudJob.from_cluster_name("test-cluster")
        assert job.cluster_name == "test-cluster"
        assert job.phase == JobPhase.EXECUTING

    def test_phase_initial(self):
        job = CloudJob("test-cluster")
        assert job.phase == JobPhase.PROVISIONING

    def test_vm_ip_before_up(self):
        job = CloudJob("test-cluster")
        assert job.vm_ip is None

    def test_ports_empty_for_single_job(self):
        job = CloudJob("test-cluster")
        assert job.ports == {}

    def test_status_dry_run(self):
        job = CloudJob("dry-run")
        s = job.status()
        assert s["cluster_status"] == "dry-run"

    def test_cost_dry_run(self):
        job = CloudJob("dry-run")
        assert job.cost_so_far() == 0.0

    def test_list_gpu_types_empty_on_no_catalog(self, creds_file):
        import sys
        mock_sky = MagicMock()
        mock_catalog = MagicMock()
        mock_catalog.list_accelerators.side_effect = Exception("no catalog")
        launcher = CloudLauncher(credentials_path=creds_file)
        with patch.dict(sys.modules, {
            "sky": mock_sky,
            "sky.catalog": mock_catalog,
        }):
            result = launcher.list_gpu_types("runpod")
        assert result == []
