"""Tests for v0.2 #8 preempt/checkpoint lifecycle wiring.

Covers:
  - Integrity manifest (write/read/verify, mismatch detection)
  - save_state_with_manifest / load_state_with_manifest round-trip
  - download_and_load_state via file:// URL
  - make_preempt_snapshot_hook builds a callback that snapshots on
    a CloudSession preemption event
  - resume_from_url entry-point helper
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import jax.numpy as jnp
import pytest

from maddening.cloud.entrypoint import (
    make_preempt_snapshot_hook,
    resume_from_url,
)
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointIntegrityError,
    CheckpointVersionError,
    compute_checkpoint_hash,
    download_and_load_state,
    load_state_with_manifest,
    read_manifest,
    save_state,
    save_state_with_manifest,
    verify_manifest,
    write_manifest,
)
from maddening.nodes.ball import BallNode
from maddening.nodes.table import TableNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bouncing_ball_graph():
    """Compiled bouncing-ball graph with a few steps under its belt."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01))
    gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    for _ in range(5):
        gm.step()
    return gm


@pytest.fixture
def matching_empty_graph():
    """Same nodes/edges but fresh state — destination for load tests."""
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01))
    gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


# ---------------------------------------------------------------------------
# Manifest write/read/verify
# ---------------------------------------------------------------------------


class TestManifest:
    def test_write_manifest_creates_sidecar(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        manifest_path = write_manifest(npz_path)
        assert manifest_path.exists()
        assert manifest_path.name == "snap.npz.manifest.json"

    def test_manifest_has_required_fields(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        write_manifest(npz_path)
        manifest = read_manifest(npz_path)
        assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        assert isinstance(manifest["sha256"], str)
        assert len(manifest["sha256"]) == 64  # full SHA-256 hex
        assert manifest["size_bytes"] > 0
        assert isinstance(manifest["extra"], dict)

    def test_manifest_extra_carries_through(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        write_manifest(npz_path, extra={"commit": "deadbeef", "sim_time": 1.5})
        manifest = read_manifest(npz_path)
        assert manifest["extra"]["commit"] == "deadbeef"
        assert manifest["extra"]["sim_time"] == 1.5

    def test_read_missing_manifest_raises(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        with pytest.raises(FileNotFoundError, match="manifest"):
            read_manifest(npz_path)

    def test_verify_manifest_passes_on_match(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        write_manifest(npz_path)
        verify_manifest(npz_path)  # no exception

    def test_verify_manifest_fails_on_tampered_npz(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        write_manifest(npz_path)
        # Tamper with the .npz body
        body = npz_path.read_bytes()
        npz_path.write_bytes(body + b"\x00")
        with pytest.raises(CheckpointIntegrityError, match="SHA-256"):
            verify_manifest(npz_path)

    def test_verify_manifest_fails_on_schema_mismatch(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        write_manifest(npz_path)
        # Tamper with the manifest schema version
        manifest_path = npz_path.with_suffix(npz_path.suffix + ".manifest.json")
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = CHECKPOINT_SCHEMA_VERSION + 99
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(CheckpointVersionError):
            verify_manifest(npz_path)

    def test_verify_manifest_accepts_inmemory_manifest(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "sha256": compute_checkpoint_hash(npz_path),
            "size_bytes": npz_path.stat().st_size,
            "extra": {},
        }
        verify_manifest(npz_path, manifest=manifest)


class TestSchemaVersionPolicy:
    """v0.2 #8 follow-up: explicit version-window policy.

    A release running schema version N reads N and N-1; older or newer
    raises CheckpointVersionError naming the intermediate release.
    """

    def _make_manifest(self, version: int, sha: str, size: int = 100) -> dict:
        return {
            "schema_version": version,
            "sha256": sha,
            "size_bytes": size,
            "extra": {},
        }

    def test_current_version_passes(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        verify_manifest(
            npz_path,
            manifest=self._make_manifest(
                CHECKPOINT_SCHEMA_VERSION,
                compute_checkpoint_hash(npz_path),
                npz_path.stat().st_size,
            ),
        )  # no exception

    def test_too_new_raises_version_error(self, bouncing_ball_graph, tmp_path):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        future = CHECKPOINT_SCHEMA_VERSION + 5
        with pytest.raises(CheckpointVersionError) as ei:
            verify_manifest(
                npz_path,
                manifest=self._make_manifest(future, "x" * 64),
            )
        assert ei.value.file_version == future
        assert "newer than this release" in str(ei.value)

    def test_too_old_raises_version_error_with_hint(self, bouncing_ball_graph, tmp_path):
        # When CHECKPOINT_SCHEMA_VERSION ≥ 2, a v(N-2) checkpoint is
        # out-of-window and the error message names the intermediate
        # release.  v1 today → no "too old" case exists; we exercise
        # the code path by simulating a future bump.
        if CHECKPOINT_SCHEMA_VERSION < 3:
            pytest.skip("requires schema_version ≥ 3 to be out-of-window in the past")
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        ancient = CHECKPOINT_SCHEMA_VERSION - 2
        with pytest.raises(CheckpointVersionError) as ei:
            verify_manifest(
                npz_path,
                manifest=self._make_manifest(ancient, "x" * 64),
            )
        assert ei.value.file_version == ancient
        assert "Load it with" in str(ei.value)

    def test_non_int_schema_version_raises_integrity_error(
        self, bouncing_ball_graph, tmp_path,
    ):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        bad_manifest = {
            "schema_version": "v1",  # string, not int
            "sha256": "x" * 64,
            "size_bytes": 0,
            "extra": {},
        }
        with pytest.raises(CheckpointIntegrityError, match="schema_version"):
            verify_manifest(npz_path, manifest=bad_manifest)

    def test_version_error_is_integrity_error_subclass(self):
        # Callers that catch the parent class still see version errors.
        assert issubclass(CheckpointVersionError, CheckpointIntegrityError)

    def test_version_error_carries_structured_fields(self):
        err = CheckpointVersionError(
            path="/tmp/x.npz",
            file_version=0,
            readable_min=1,
            readable_max=2,
        )
        assert err.file_version == 0
        assert err.readable_min == 1
        assert err.readable_max == 2
        assert err.path == "/tmp/x.npz"


# ---------------------------------------------------------------------------
# save_state_with_manifest + load_state_with_manifest
# ---------------------------------------------------------------------------


class TestSaveLoadWithManifest:
    def test_round_trip(self, bouncing_ball_graph, matching_empty_graph, tmp_path):
        npz_path, manifest_path = save_state_with_manifest(
            bouncing_ball_graph, tmp_path / "snap.npz",
            extra={"step": 5},
        )
        assert npz_path.exists()
        assert manifest_path.exists()
        manifest = load_state_with_manifest(matching_empty_graph, npz_path)
        assert manifest["extra"]["step"] == 5
        # Ball position should match
        src_pos = bouncing_ball_graph.get_node_state("ball")["position"]
        dst_pos = matching_empty_graph.get_node_state("ball")["position"]
        assert float(src_pos) == pytest.approx(float(dst_pos), rel=1e-6)

    def test_load_with_skip_integrity(self, bouncing_ball_graph, matching_empty_graph, tmp_path):
        npz_path, _ = save_state_with_manifest(
            bouncing_ball_graph, tmp_path / "snap.npz",
        )
        # Corrupt the manifest hash; the load with skip should still succeed
        mp = npz_path.with_suffix(npz_path.suffix + ".manifest.json")
        m = json.loads(mp.read_text())
        m["sha256"] = "0" * 64
        mp.write_text(json.dumps(m))
        load_state_with_manifest(
            matching_empty_graph, npz_path, skip_integrity_check=True,
        )

    def test_load_rejects_tampered_checkpoint_by_default(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        npz_path, _ = save_state_with_manifest(
            bouncing_ball_graph, tmp_path / "snap.npz",
        )
        # Tamper with the npz body
        npz_path.write_bytes(npz_path.read_bytes() + b"\xff")
        with pytest.raises(CheckpointIntegrityError):
            load_state_with_manifest(matching_empty_graph, npz_path)


# ---------------------------------------------------------------------------
# download_and_load_state via file://
# ---------------------------------------------------------------------------


class TestDownloadAndLoad:
    def test_file_scheme_url(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        npz_path, _ = save_state_with_manifest(
            bouncing_ball_graph, tmp_path / "snap.npz",
        )
        url = f"file://{npz_path}"
        dest_dir = tmp_path / "resume"
        manifest = download_and_load_state(
            matching_empty_graph, url, dest_dir=dest_dir,
        )
        assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
        # Resumed graph should have the same ball position
        src = float(bouncing_ball_graph.get_node_state("ball")["position"])
        dst = float(matching_empty_graph.get_node_state("ball")["position"])
        assert src == pytest.approx(dst, rel=1e-6)

    def test_bare_path_treated_as_file_scheme(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        npz_path, _ = save_state_with_manifest(
            bouncing_ball_graph, tmp_path / "snap.npz",
        )
        dest_dir = tmp_path / "resume2"
        # No scheme — should still work as a file path
        manifest = download_and_load_state(
            matching_empty_graph, str(npz_path), dest_dir=dest_dir,
        )
        assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION

    def test_unsupported_scheme_raises(self, matching_empty_graph, tmp_path):
        with pytest.raises(ValueError, match="scheme"):
            download_and_load_state(
                matching_empty_graph, "ftp://example.com/x.npz",
                dest_dir=tmp_path,
            )

    def test_missing_manifest_triggers_failure(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        # Save without manifest
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        url = f"file://{npz_path}"
        with pytest.raises(FileNotFoundError):
            download_and_load_state(
                matching_empty_graph, url, dest_dir=tmp_path / "resume",
            )

    def test_skip_integrity_tolerates_missing_manifest(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        url = f"file://{npz_path}"
        manifest = download_and_load_state(
            matching_empty_graph, url,
            dest_dir=tmp_path / "resume",
            skip_integrity_check=True,
        )
        assert manifest == {}


# ---------------------------------------------------------------------------
# resume_from_url (entrypoint helper)
# ---------------------------------------------------------------------------


class _FakeServer:
    def __init__(self, gm):
        self.gm = gm


class TestResumeFromUrl:
    def test_resumes_from_local_file(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        npz_path, _ = save_state_with_manifest(
            bouncing_ball_graph, tmp_path / "snap.npz",
        )
        server = _FakeServer(matching_empty_graph)
        manifest = resume_from_url(server, f"file://{npz_path}")
        assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION

    def test_skip_integrity_passes_through(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        npz_path = save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        server = _FakeServer(matching_empty_graph)
        manifest = resume_from_url(
            server, f"file://{npz_path}",
            skip_integrity_check=True,
        )
        assert manifest == {}


# ---------------------------------------------------------------------------
# make_preempt_snapshot_hook
# ---------------------------------------------------------------------------


class _FakeInfo:
    def __init__(self, session_id="abc", stage_value="preempted"):
        self.session_id = session_id

        class _Stage:
            value = stage_value
        self.stage = _Stage()


class TestPreemptSnapshotHook:
    def test_hook_writes_snapshot_and_manifest(
        self, bouncing_ball_graph, tmp_path,
    ):
        server = _FakeServer(bouncing_ball_graph)
        snap = tmp_path / "preempt.npz"
        hook = make_preempt_snapshot_hook(
            server, snapshot_path=str(snap),
            extra_meta={"commit": "abc123"},
        )
        hook(_FakeInfo())
        assert snap.exists()
        manifest = read_manifest(snap)
        assert manifest["extra"]["commit"] == "abc123"
        assert manifest["extra"]["session_id"] == "abc"
        assert manifest["extra"]["stage_at_snapshot"] == "preempted"

    def test_hook_uses_env_var_when_path_not_supplied(
        self, bouncing_ball_graph, tmp_path, monkeypatch,
    ):
        server = _FakeServer(bouncing_ball_graph)
        monkeypatch.setenv("MADDENING_SNAPSHOT_PATH", str(tmp_path / "env.npz"))
        hook = make_preempt_snapshot_hook(server)
        hook(_FakeInfo())
        assert (tmp_path / "env.npz").exists()

    def test_hook_uses_env_dir_when_path_not_supplied(
        self, bouncing_ball_graph, tmp_path, monkeypatch,
    ):
        server = _FakeServer(bouncing_ball_graph)
        monkeypatch.delenv("MADDENING_SNAPSHOT_PATH", raising=False)
        monkeypatch.setenv("MADDENING_SNAPSHOT_DIR", str(tmp_path))
        hook = make_preempt_snapshot_hook(server)
        hook(_FakeInfo())
        assert (tmp_path / "maddening_preempt_snapshot.npz").exists()

    def test_hook_does_not_raise_on_failure(
        self, bouncing_ball_graph, tmp_path,
    ):
        # Snapshot path inside a non-existent unwritable dir would raise.
        # We accept the failure silently (logged) so a preemption hook
        # never propagates an exception into the CloudSession monitor.
        server = _FakeServer(bouncing_ball_graph)
        # On Linux, /proc/0/foo is rejected with EACCES; safer to use a
        # path that's actually denied for non-root.
        hook = make_preempt_snapshot_hook(
            server, snapshot_path="/proc/0/snap.npz",
        )
        # Should not raise.
        hook(_FakeInfo())


# ---------------------------------------------------------------------------
# Integration: CloudSession on_preempted callback wiring
# ---------------------------------------------------------------------------


class TestCloudSessionIntegration:
    def test_preemption_signal_invokes_snapshot_hook(
        self, bouncing_ball_graph, tmp_path,
    ):
        from maddening.cloud.session import CloudSession

        server = _FakeServer(bouncing_ball_graph)
        snap = tmp_path / "auto_snapshot.npz"
        hook = make_preempt_snapshot_hook(server, snapshot_path=str(snap))

        sess = CloudSession(on_preempted=hook)
        # Simulate a preemption signal directly
        sess._on_preemption_signal()
        assert snap.exists()
        manifest = read_manifest(snap)
        assert manifest["schema_version"] == CHECKPOINT_SCHEMA_VERSION
