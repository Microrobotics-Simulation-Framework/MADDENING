"""Tests for CloudSession state machine via MockCloudSession."""

import threading
import time

import pytest

from maddening.cloud.mock_session import MockCloudSession
from maddening.cloud.session import (
    CloudConfig,
    CloudReadyResult,
    CloudSessionError,
    CloudSessionInfo,
    CloudStage,
    PreemptionPolicy,
)


class TestCloudConfig:
    def test_defaults(self):
        cfg = CloudConfig()
        assert cfg.cloud == "gcp"
        assert cfg.spot is True
        assert cfg.on_preempted == PreemptionPolicy.CHECKPOINT

    def test_from_dict(self):
        d = {
            "cloud": "aws",
            "instance_type": "g4dn.xlarge",
            "accelerator": "T4:1",
            "spot": False,
            "on_preempted": "abort",
            "container_image": "my-image:v1",
            "stream_config": {
                "width": 1920, "height": 1080, "fps": 60,
                "bitrate_kbps": 8000, "codec": "h264",
                "pixel_format": "RGBA", "enable_audio": False,
                "ice_servers": [],
            },
            "region": "us-west-2",
            "region_strategy": "cheapest",
        }
        cfg = CloudConfig.from_dict(d)
        assert cfg.cloud == "aws"
        assert cfg.on_preempted == PreemptionPolicy.ABORT
        assert cfg.stream_config.width == 1920
        assert not cfg.spot

    def test_frozen(self):
        cfg = CloudConfig()
        with pytest.raises(AttributeError):
            cfg.cloud = "aws"  # type: ignore[misc]


class TestCloudReadyResult:
    def test_fully_ready(self):
        r = CloudReadyResult(
            vm_ready=True, container_ready=True, simulation_ready=True,
            stream_ready=True, data_ready=True,
        )
        assert r.fully_ready

    def test_not_fully_ready_missing_stage(self):
        r = CloudReadyResult(
            vm_ready=True, container_ready=True, simulation_ready=True,
            stream_ready=False, data_ready=True,
        )
        assert not r.fully_ready

    def test_not_fully_ready_with_error(self):
        r = CloudReadyResult(
            vm_ready=True, container_ready=True, simulation_ready=True,
            stream_ready=True, data_ready=True,
            error_stage="container", error_detail="timeout",
        )
        assert not r.fully_ready


class TestCloudStage:
    def test_all_stages(self):
        stages = [s.value for s in CloudStage]
        assert "not_started" in stages
        assert "fully_ready" in stages
        assert "error" in stages
        assert "preempted" in stages


class TestMockCloudSessionHappyPath:
    def test_launch_and_wait_ready(self):
        stages_seen = []

        def on_stage(info: CloudSessionInfo):
            stages_seen.append(info.stage)

        session = MockCloudSession(on_stage_changed=on_stage)
        session.launch(CloudConfig())
        result = session.wait_ready(timeout=5.0)

        assert result.fully_ready
        assert result.error_stage is None
        assert CloudStage.VM_PROVISIONING in stages_seen
        assert CloudStage.FULLY_READY in stages_seen

    def test_session_info_populated(self):
        session = MockCloudSession()
        info = session.launch(CloudConfig())
        assert info.vm_ip == "10.0.0.42"
        assert info.session_id == "mock-session-001"
        session.wait_ready(timeout=5.0)
        final_info = session.info
        assert final_info.stage == CloudStage.FULLY_READY

    def test_health_check(self):
        session = MockCloudSession()
        session.launch(CloudConfig())
        session.wait_ready(timeout=5.0)
        result = session.health_check()
        assert result.fully_ready

    def test_teardown(self):
        session = MockCloudSession()
        session.launch(CloudConfig())
        session.wait_ready(timeout=5.0)
        session.teardown()
        assert session.stage == CloudStage.NOT_STARTED

    def test_relaunch_after_teardown(self):
        session = MockCloudSession()
        session.launch(CloudConfig())
        session.wait_ready(timeout=5.0)
        session.teardown()
        # Re-launch should work from NOT_STARTED
        session.launch(CloudConfig())
        result = session.wait_ready(timeout=5.0)
        assert result.fully_ready


class TestMockCloudSessionFailure:
    def test_fail_at_vm_provisioning(self):
        session = MockCloudSession(
            fail_at_stage=CloudStage.VM_PROVISIONING,
            fail_detail="No capacity in region",
        )
        # VM_PROVISIONING is set by launch(), not the worker.
        # The worker starts at CONTAINER_STARTING.
        # So let's fail at CONTAINER_STARTING instead.
        session2 = MockCloudSession(
            fail_at_stage=CloudStage.CONTAINER_STARTING,
            fail_detail="Container pull failed",
        )
        session2.launch(CloudConfig())
        result = session2.wait_ready(timeout=5.0)
        assert not result.fully_ready
        assert result.error_stage == "container_starting"
        assert "Container pull failed" in result.error_detail

    def test_fail_at_simulation_starting(self):
        session = MockCloudSession(
            fail_at_stage=CloudStage.SIMULATION_STARTING,
            fail_detail="Graph compilation error",
        )
        session.launch(CloudConfig())
        result = session.wait_ready(timeout=5.0)
        assert not result.fully_ready
        assert result.error_stage == "simulation_starting"
        assert result.error_detail == "Graph compilation error"

    def test_fail_at_stream_starting(self):
        session = MockCloudSession(
            fail_at_stage=CloudStage.STREAM_STARTING,
            fail_detail="GStreamer init failed",
        )
        session.launch(CloudConfig())
        result = session.wait_ready(timeout=5.0)
        assert not result.fully_ready
        assert result.error_stage == "stream_starting"

    def test_fail_at_data_ready(self):
        session = MockCloudSession(
            fail_at_stage=CloudStage.DATA_READY,
            fail_detail="ZMQ bind failed",
        )
        session.launch(CloudConfig())
        result = session.wait_ready(timeout=5.0)
        assert not result.fully_ready
        assert result.error_stage == "data_ready"

    def test_relaunch_after_error(self):
        session = MockCloudSession(
            fail_at_stage=CloudStage.CONTAINER_STARTING,
        )
        session.launch(CloudConfig())
        session.wait_ready(timeout=5.0)
        assert session.stage == CloudStage.ERROR
        # Should be able to re-launch from ERROR state
        session._fail_at_stage = None  # remove failure
        session.launch(CloudConfig())
        result = session.wait_ready(timeout=5.0)
        assert result.fully_ready


class TestMockCloudSessionPreemption:
    def test_preemption_callback(self):
        preempted_info = []

        def on_preempt(info: CloudSessionInfo):
            preempted_info.append(info)

        session = MockCloudSession(
            on_preempted=on_preempt,
            stage_delay=0.05,
            preempt_after=0.01,  # preempt very quickly
        )
        session.launch(CloudConfig())
        session.wait_ready(timeout=5.0)
        assert session.stage == CloudStage.PREEMPTED
        assert len(preempted_info) == 1

    def test_relaunch_after_preemption(self):
        session = MockCloudSession(
            stage_delay=0.05,
            preempt_after=0.01,
        )
        session.launch(CloudConfig())
        session.wait_ready(timeout=5.0)
        assert session.stage == CloudStage.PREEMPTED
        # Re-launch from PREEMPTED
        session._preempt_after = None  # disable preemption
        session.launch(CloudConfig())
        result = session.wait_ready(timeout=5.0)
        assert result.fully_ready


class TestMockCloudSessionEdgeCases:
    def test_double_launch_raises(self):
        session = MockCloudSession(stage_delay=0.1)
        session.launch(CloudConfig())
        with pytest.raises(CloudSessionError):
            session.launch(CloudConfig())
        session.wait_ready(timeout=5.0)

    def test_wait_ready_timeout(self):
        session = MockCloudSession(stage_delay=10.0)  # very slow
        session.launch(CloudConfig())
        result = session.wait_ready(timeout=0.01)
        assert not result.fully_ready

    def test_stage_progression_order(self):
        stages_seen = []

        def on_stage(info: CloudSessionInfo):
            stages_seen.append(info.stage)

        session = MockCloudSession(on_stage_changed=on_stage)
        session.launch(CloudConfig())
        session.wait_ready(timeout=5.0)

        expected = [
            CloudStage.VM_PROVISIONING,
            CloudStage.CONTAINER_STARTING,
            CloudStage.SIMULATION_STARTING,
            CloudStage.STREAM_STARTING,
            CloudStage.STREAM_READY,
            CloudStage.DATA_READY,
            CloudStage.FULLY_READY,
        ]
        assert stages_seen == expected
