"""Tests for device mesh creation."""

import os

import pytest

# Force 2 CPU devices for testing
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=2")

import jax
from maddening.cloud.multigpu.device_mesh import create_device_mesh


class TestCreateDeviceMesh:
    def test_default_uses_all_devices(self):
        mesh = create_device_mesh()
        assert mesh.shape["devices"] == len(jax.devices())
        assert mesh.axis_names == ("devices",)

    def test_explicit_n_devices(self):
        mesh = create_device_mesh(n_devices=2)
        assert mesh.shape["devices"] == 2

    def test_single_device(self):
        mesh = create_device_mesh(n_devices=1)
        assert mesh.shape["devices"] == 1

    def test_too_many_devices_raises(self):
        n = len(jax.devices()) + 100
        with pytest.raises(ValueError, match="only .* available"):
            create_device_mesh(n_devices=n)
