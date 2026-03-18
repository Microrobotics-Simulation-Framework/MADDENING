"""Tests for health probes and wait_for retry logic."""

import time

import pytest

from maddening.cloud._health import HealthProbeError, wait_for


class TestHealthProbeError:
    def test_stage_and_detail(self):
        err = HealthProbeError("vm", "SSH timeout")
        assert err.stage == "vm"
        assert err.detail == "SSH timeout"
        assert "vm" in str(err)
        assert "SSH timeout" in str(err)


class TestWaitFor:
    def test_immediate_success(self):
        call_count = [0]

        def probe():
            call_count[0] += 1

        wait_for(probe, timeout=5.0, interval=0.1)
        assert call_count[0] == 1

    def test_success_after_retries(self):
        attempts = [0]

        def probe():
            attempts[0] += 1
            if attempts[0] < 3:
                raise HealthProbeError("test", f"attempt {attempts[0]}")

        wait_for(probe, timeout=5.0, interval=0.01)
        assert attempts[0] == 3

    def test_timeout_raises_last_error(self):
        def probe():
            raise HealthProbeError("container", "not ready")

        with pytest.raises(HealthProbeError) as exc_info:
            wait_for(probe, timeout=0.05, interval=0.01)

        assert exc_info.value.stage == "container"
        assert "not ready" in exc_info.value.detail

    def test_preserves_stage_attribution(self):
        """Error attribution is preserved through wait_for."""
        def probe():
            raise HealthProbeError("data_channel", "ZMQ unreachable")

        with pytest.raises(HealthProbeError) as exc_info:
            wait_for(probe, timeout=0.05, interval=0.01)

        assert exc_info.value.stage == "data_channel"

    def test_timeout_with_no_error_raises(self):
        """Edge case: wait_for with a probe that never runs due to zero timeout."""
        # This shouldn't normally happen, but test the fallback path.
        # With a very short timeout and a probe that always fails:
        def probe():
            raise HealthProbeError("vm", "down")

        with pytest.raises(HealthProbeError):
            wait_for(probe, timeout=0.001, interval=0.001)
