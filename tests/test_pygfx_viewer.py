"""Tests for GPUHistoryViewer (pygfx backend).

These tests verify the builder API, data processing, and scene construction
logic WITHOUT requiring a GPU or display.  The actual pygfx rendering is
mocked where necessary.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

# Skip entire module if pygfx is not installed
pygfx = pytest.importorskip("pygfx")


from maddening.viz.backends.pygfx_viewer import (
    GPUHistoryViewer,
    _hex_to_rgba,
    _make_sphere,
    _pyvista_mesh_to_arrays,
)


# -- Helpers --

def _make_history(n_frames=10, nx=8, ny=6, nz=6):
    """Create a synthetic history dict for testing."""
    return {
        "fluid": {
            "density": np.random.rand(n_frames, nx, ny, nz).astype(np.float32),
            "velocity": np.random.rand(n_frames, nx, ny, nz, 3).astype(np.float32) * 0.01,
            "tracer": np.random.rand(n_frames, nx, ny, nz).astype(np.float32),
        },
    }


class TestHexToRGBA:
    def test_white(self):
        assert _hex_to_rgba("#FFFFFF") == pytest.approx((1.0, 1.0, 1.0, 1.0))

    def test_black(self):
        assert _hex_to_rgba("#000000") == pytest.approx((0.0, 0.0, 0.0, 1.0))

    def test_with_opacity(self):
        r, g, b, a = _hex_to_rgba("#FF0000", opacity=0.5)
        assert r == pytest.approx(1.0)
        assert a == pytest.approx(0.5)


class TestMakeSphere:
    def test_sphere_shape(self):
        pos, idx, nrm = _make_sphere((0, 0, 0), 1.0, resolution=8)
        assert pos.dtype == np.float32
        assert idx.dtype == np.int32
        assert nrm.dtype == np.float32
        assert pos.shape[1] == 3
        assert idx.shape[1] == 3
        assert nrm.shape[0] == pos.shape[0]

    def test_sphere_center(self):
        center = (5.0, 3.0, 1.0)
        pos, _, _ = _make_sphere(center, 1.0, resolution=8)
        centroid = pos.mean(axis=0)
        # Centroid should be near the center
        assert centroid == pytest.approx(center, abs=0.3)

    def test_sphere_radius(self):
        pos, _, _ = _make_sphere((0, 0, 0), 2.0, resolution=16)
        distances = np.linalg.norm(pos, axis=1)
        assert distances.max() == pytest.approx(2.0, abs=0.01)


class TestGPUHistoryViewerBuilder:
    """Test the builder API without opening a window."""

    def test_constructor(self):
        history = _make_history()
        viewer = GPUHistoryViewer(history, dt=0.01)
        assert viewer._n_frames == 10
        assert viewer._dt == 0.01
        assert viewer._autoplay is True
        assert viewer._camera_up is None

    def test_constructor_camera_up(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01, camera_up=(0, 0, 1))
        assert viewer._camera_up == (0, 0, 1)

    def test_add_isosurface(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        viewer.add_isosurface("fluid", "tracer", threshold=0.5,
                              color="#4488CC", opacity=0.6)
        assert len(viewer._isosurfaces) == 1
        assert viewer._isosurfaces[0].threshold == 0.5

    def test_add_volume_slice(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        viewer.add_volume_slice("fluid", "velocity", component=0,
                                normal="y", origin_frac=0.5)
        assert len(viewer._slices) == 1
        assert viewer._slices[0].normal == "y"

    def test_add_particle(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        viewer.add_particle("fluid", "velocity",
                            start_pos=(4, 3, 3), radius=0.5)
        assert len(viewer._particles) == 1

    def test_add_rotating_mesh(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        # Use a raw tuple instead of PyVista mesh
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        idx = np.array([[0, 1, 2]], dtype=np.int32)
        nrm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        viewer.add_rotating_mesh(
            (pos, idx, nrm),
            axis="x", speed=5.0, center=(0.5, 0.5, 0.0),
        )
        assert len(viewer._rotating_meshes) == 1

    def test_add_static_mesh(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        idx = np.array([[0, 1, 2]], dtype=np.int32)
        nrm = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
        viewer.add_static_mesh((pos, idx, nrm), color="#AABBCC", opacity=0.3)
        assert len(viewer._static_meshes) == 1

    def test_add_arrows(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        viewer.add_arrows("fluid", "velocity")
        assert len(viewer._arrows) == 1

    def test_add_streamlines(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        viewer.add_streamlines("fluid", "velocity", dims=(8, 6, 6))
        assert len(viewer._streamlines) == 1


class TestFieldExtraction:
    """Test internal field extraction helpers."""

    def test_get_field(self):
        history = _make_history(n_frames=5, nx=4, ny=4, nz=4)
        viewer = GPUHistoryViewer(history, dt=0.01)
        field = viewer._get_field("fluid", "density", 2)
        assert field.shape == (4, 4, 4)

    def test_extract_scalar_magnitude(self):
        history = _make_history(n_frames=3, nx=4, ny=4, nz=4)
        viewer = GPUHistoryViewer(history, dt=0.01)
        scalar = viewer._extract_scalar("fluid", "velocity", -1, 0)
        assert scalar.shape == (4, 4, 4)
        # Magnitude should be >= 0
        assert scalar.min() >= 0

    def test_extract_scalar_component(self):
        history = _make_history(n_frames=3, nx=4, ny=4, nz=4)
        viewer = GPUHistoryViewer(history, dt=0.01)
        scalar = viewer._extract_scalar("fluid", "velocity", 0, 0)
        assert scalar.shape == (4, 4, 4)

    def test_slice_index(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        axis, idx = viewer._slice_index("y", 0.5, (8, 6, 6))
        assert axis == 1
        assert idx == 2  # floor(0.5 * 5) = 2


class TestParticleTrajectory:
    """Test particle advection pre-computation."""

    def test_trajectory_shape(self):
        history = _make_history(n_frames=10, nx=8, ny=6, nz=6)
        viewer = GPUHistoryViewer(history, dt=0.01)
        from maddening.viz.history_viewer import _ParticleDef
        pdef = _ParticleDef(
            node="fluid", field_name="velocity",
            start_pos=(4.0, 3.0, 3.0), radius=0.5,
            color="#22AA44", opacity=1.0,
            clamp_y=None, clamp_z=None, periodic_x=None,
        )
        traj = viewer._precompute_particle_trajectory(pdef)
        assert traj.shape == (10, 3)
        # First position matches start
        assert traj[0] == pytest.approx((4.0, 3.0, 3.0))

    def test_trajectory_with_periodic(self):
        history = _make_history(n_frames=5, nx=8, ny=6, nz=6)
        viewer = GPUHistoryViewer(history, dt=0.01)
        from maddening.viz.history_viewer import _ParticleDef
        pdef = _ParticleDef(
            node="fluid", field_name="velocity",
            start_pos=(7.5, 3.0, 3.0), radius=0.5,
            color="#22AA44", opacity=1.0,
            clamp_y=None, clamp_z=None, periodic_x=8.0,
        )
        traj = viewer._precompute_particle_trajectory(pdef)
        # All x positions should be in [0, 8)
        assert np.all(traj[:, 0] >= 0)
        assert np.all(traj[:, 0] < 8.0)


class TestIsosurfaceExtraction:
    """Test CPU marching cubes isosurface extraction."""

    def test_isosurface_arrays(self):
        skimage = pytest.importorskip("skimage")
        # Create a field with a clear threshold crossing
        n = 5
        history = {"test": {
            "scalar": np.zeros((1, 16, 16, 16), dtype=np.float32),
        }}
        # Put a sphere of value 1.0 in the center
        for x in range(16):
            for y in range(16):
                for z in range(16):
                    if (x - 8)**2 + (y - 8)**2 + (z - 8)**2 < 25:
                        history["test"]["scalar"][0, x, y, z] = 1.0

        viewer = GPUHistoryViewer(history, dt=0.01)
        from maddening.viz.history_viewer import _IsosurfaceDef
        idef = _IsosurfaceDef(
            node="test", field_name="scalar", threshold=0.5,
            color="#4488CC", opacity=0.6, smooth_n_iter=0,
        )
        result = viewer._compute_isosurface_arrays(idef, 0)
        assert result is not None
        positions, indices, normals = result
        assert positions.shape[1] == 3
        assert indices.shape[1] == 3
        assert normals.shape == positions.shape

    def test_isosurface_none_when_no_crossing(self):
        skimage = pytest.importorskip("skimage")
        history = {"test": {
            "scalar": np.zeros((1, 8, 8, 8), dtype=np.float32),
        }}
        viewer = GPUHistoryViewer(history, dt=0.01)
        from maddening.viz.history_viewer import _IsosurfaceDef
        idef = _IsosurfaceDef(
            node="test", field_name="scalar", threshold=0.5,
            color="#4488CC", opacity=0.6, smooth_n_iter=0,
        )
        result = viewer._compute_isosurface_arrays(idef, 0)
        assert result is None


class TestScalarToColors:
    """Test colormap mapping."""

    def test_output_shape(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        values = np.linspace(0, 1, 100).astype(np.float32)
        colors = viewer._scalar_to_colors(values, (0, 1), "coolwarm", 0.8)
        assert colors.shape == (100, 4)
        assert colors.dtype == np.float32

    def test_opacity(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        values = np.array([0.5], dtype=np.float32)
        colors = viewer._scalar_to_colors(values, (0, 1), "coolwarm", 0.7)
        assert colors[0, 3] == pytest.approx(0.7)


class TestPlaybackControls:
    """Test playback logic without a window."""

    def test_toggle_play(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        assert viewer._playing is False
        viewer._toggle_play()
        assert viewer._playing is True
        viewer._toggle_play()
        assert viewer._playing is False

    def test_speed_up_down(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        assert viewer._playback_speed == 1
        viewer._speed_up()
        assert viewer._playback_speed == 2
        viewer._speed_up()
        assert viewer._playback_speed == 4
        viewer._speed_down()
        assert viewer._playback_speed == 2
        viewer._speed_down()
        assert viewer._playback_speed == 1
        viewer._speed_down()
        assert viewer._playback_speed == 1  # clamped

    def test_speed_max(self):
        viewer = GPUHistoryViewer(_make_history(), dt=0.01)
        for _ in range(20):
            viewer._speed_up()
        assert viewer._playback_speed == 64

    def test_toggle_play_restarts_at_end(self):
        viewer = GPUHistoryViewer(_make_history(n_frames=10), dt=0.01)
        viewer._frame_idx = 9  # last frame
        # Mock _update_frame to avoid pygfx calls
        updated = []
        viewer._update_frame = lambda f, light=False: updated.append(f)
        viewer._toggle_play()
        assert viewer._playing is True
        assert 0 in updated  # should have reset to frame 0


class TestPyVistaConversion:
    """Test PyVista mesh → arrays conversion."""

    def test_pyvista_mesh(self):
        pv = pytest.importorskip("pyvista")
        sphere = pv.Sphere(radius=1.0, center=(0, 0, 0))
        pos, idx, nrm = _pyvista_mesh_to_arrays(sphere)
        assert pos.dtype == np.float32
        assert idx.dtype == np.int32
        assert nrm.dtype == np.float32
        assert pos.shape[1] == 3
        assert idx.shape[1] == 3
        assert nrm.shape == pos.shape

    def test_pyvista_box(self):
        pv = pytest.importorskip("pyvista")
        box = pv.Box(bounds=[0, 1, 0, 1, 0, 1])
        pos, idx, nrm = _pyvista_mesh_to_arrays(box)
        assert pos.shape[0] > 0
        assert idx.shape[0] > 0
