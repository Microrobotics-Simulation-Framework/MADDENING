"""Tests for 2-D (pencil) device mesh creation.

Forces 8 virtual CPU devices via ``XLA_FLAGS`` before JAX is imported.
Covers M2 of the v0.2 halo-exchange roadmap.
"""

from __future__ import annotations

import os

import pytest

# Force 8 virtual CPU devices. Must be set before JAX is imported.
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax  # noqa: E402

from maddening.cloud.multigpu.device_mesh import (  # noqa: E402
    create_device_mesh,
    factor_devices,
)

_HAS_8_DEVICES = len(jax.devices()) >= 8
_HAS_16_DEVICES = len(jax.devices()) >= 16
_SKIP_8 = "Requires >=8 JAX devices (set XLA_FLAGS before JAX import)"
_SKIP_16 = "Requires >=16 JAX devices (set XLA_FLAGS before JAX import)"


# ---------------------------------------------------------------------------
# factor_devices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,expected", [
    (1, (1, 1)),
    (2, (1, 2)),
    (4, (2, 2)),
    (6, (2, 3)),
    (8, (2, 4)),
    (9, (3, 3)),
    (12, (3, 4)),
    (16, (4, 4)),
])
def test_factor_devices(n, expected):
    assert factor_devices(n) == expected


def test_factor_devices_rejects_zero():
    with pytest.raises(ValueError):
        factor_devices(0)


# ---------------------------------------------------------------------------
# 1-D mesh (regression — must remain unchanged)
# ---------------------------------------------------------------------------


class Test1DMeshUnchanged:
    def test_default_uses_all_devices(self):
        mesh = create_device_mesh()
        assert mesh.shape["devices"] == len(jax.devices())
        assert mesh.axis_names == ("devices",)

    def test_explicit_n_devices(self):
        mesh = create_device_mesh(n_devices=2)
        assert mesh.shape["devices"] == 2
        assert mesh.axis_names == ("devices",)

    def test_single_device(self):
        mesh = create_device_mesh(n_devices=1)
        assert mesh.shape["devices"] == 1


# ---------------------------------------------------------------------------
# 2-D pencil mesh
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_8_DEVICES, reason=_SKIP_8)
class TestPencilMesh8:
    def test_default_axis_names_2d(self):
        mesh = create_device_mesh(shape=(2, 4))
        assert mesh.axis_names == ("spatial_y", "spatial_z")
        assert mesh.shape["spatial_y"] == 2
        assert mesh.shape["spatial_z"] == 4
        assert mesh.devices.shape == (2, 4)

    def test_transposed_shape(self):
        mesh = create_device_mesh(shape=(4, 2))
        assert mesh.devices.shape == (4, 2)
        assert mesh.shape["spatial_y"] == 4
        assert mesh.shape["spatial_z"] == 2

    def test_degenerate_2d_shape(self):
        """(8, 1) is a valid 2-D mesh with a trivial second axis."""
        mesh = create_device_mesh(shape=(8, 1))
        assert mesh.devices.shape == (8, 1)

    def test_custom_axis_names(self):
        mesh = create_device_mesh(
            shape=(2, 4), axis_names=("py", "pz")
        )
        assert mesh.axis_names == ("py", "pz")

    def test_shape_product_must_equal_n_devices(self):
        with pytest.raises(ValueError, match="does not match"):
            create_device_mesh(n_devices=4, shape=(2, 4))

    def test_n_devices_derived_from_shape(self):
        mesh = create_device_mesh(shape=(2, 4))
        # 8 devices total; mesh covers the requested 2*4
        assert mesh.devices.size == 8

    def test_axis_names_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match="len"):
            create_device_mesh(shape=(2, 4), axis_names=("only_one",))

    def test_duplicate_axis_names_rejected(self):
        with pytest.raises(ValueError, match="not unique"):
            create_device_mesh(shape=(2, 4), axis_names=("p", "p"))


@pytest.mark.skipif(not _HAS_16_DEVICES, reason=_SKIP_16)
class TestPencilMesh16:
    def test_4x4_pencil(self):
        mesh = create_device_mesh(shape=(4, 4))
        assert mesh.devices.shape == (4, 4)
        assert mesh.shape["spatial_y"] == 4
        assert mesh.shape["spatial_z"] == 4


# ---------------------------------------------------------------------------
# Higher-dimensional meshes (3-D)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_8_DEVICES, reason=_SKIP_8)
def test_3d_mesh_default_axis_names():
    mesh = create_device_mesh(shape=(2, 2, 2))
    assert mesh.axis_names == ("axis_0", "axis_1", "axis_2")
    assert mesh.devices.shape == (2, 2, 2)
