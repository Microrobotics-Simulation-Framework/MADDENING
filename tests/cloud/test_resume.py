"""Resume-from-URL transport (``maddening.cloud.resume``).

Covers ``download_and_load_state`` over every supported scheme family:
``file://`` and bare paths (end to end), the fsspec cloud-storage
schemes (end to end on fsspec's in-memory filesystem, plus the
actionable error when the backend is missing), scheme rejection,
manifest handling, the deprecated alias left in the core checkpoint
module, and the invariant that the core module never imports the cloud
package or fsspec at import time.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.cloud.resume import download_and_load_state
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation import checkpoint as ck
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _spring_graph():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, initial_position=0.0))
    gm.compile()
    return gm


def _ball_graph():
    gm = GraphManager()
    gm.add_node(TableNode(name="table", timestep=0.01))
    gm.add_node(BallNode(name="ball", timestep=0.01, initial_position=5.0))
    gm.add_edge("table", "ball", "position", "table_position")
    gm.compile()
    return gm


@pytest.fixture
def bouncing_ball_graph():
    """Compiled bouncing-ball graph with a few steps under its belt."""
    gm = _ball_graph()
    for _ in range(5):
        gm.step()
    return gm


@pytest.fixture
def matching_empty_graph():
    """Same nodes/edges but fresh state — destination for load tests."""
    return _ball_graph()


def _assert_ball_position_restored(src_gm, dst_gm):
    src = float(src_gm.get_node_state("ball")["position"])
    dst = float(dst_gm.get_node_state("ball")["position"])
    assert src == pytest.approx(dst, rel=1e-6)


# ---------------------------------------------------------------------------
# file:// and bare paths
# ---------------------------------------------------------------------------


class TestFileScheme:
    def test_file_scheme_url(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        npz_path, _ = ck.save_state_with_manifest(
            bouncing_ball_graph, tmp_path / "snap.npz",
        )
        url = f"file://{npz_path}"
        dest_dir = tmp_path / "resume"
        manifest = download_and_load_state(
            matching_empty_graph, url, dest_dir=dest_dir,
        )
        assert manifest["schema_version"] == ck.CHECKPOINT_SCHEMA_VERSION
        _assert_ball_position_restored(bouncing_ball_graph, matching_empty_graph)

    def test_bare_path_treated_as_file_scheme(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        npz_path, _ = ck.save_state_with_manifest(
            bouncing_ball_graph, tmp_path / "snap.npz",
        )
        dest_dir = tmp_path / "resume2"
        # No scheme — should still work as a file path
        manifest = download_and_load_state(
            matching_empty_graph, str(npz_path), dest_dir=dest_dir,
        )
        assert manifest["schema_version"] == ck.CHECKPOINT_SCHEMA_VERSION

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
        npz_path = ck.save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        url = f"file://{npz_path}"
        with pytest.raises(FileNotFoundError):
            download_and_load_state(
                matching_empty_graph, url, dest_dir=tmp_path / "resume",
            )

    def test_skip_integrity_tolerates_missing_manifest(
        self, bouncing_ball_graph, matching_empty_graph, tmp_path,
    ):
        npz_path = ck.save_state(bouncing_ball_graph, tmp_path / "snap.npz")
        url = f"file://{npz_path}"
        manifest = download_and_load_state(
            matching_empty_graph, url,
            dest_dir=tmp_path / "resume",
            skip_integrity_check=True,
        )
        assert manifest == {}


# ---------------------------------------------------------------------------
# fsspec cloud-storage schemes (C3, v0.4.0)
# ---------------------------------------------------------------------------


def test_unknown_scheme_still_rejected():
    with pytest.raises(ValueError, match="Unsupported URL scheme 'ftp'"):
        download_and_load_state(_spring_graph(), "ftp://host/x.npz")


def test_missing_fsspec_gives_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(ImportError, match=r"pip install fsspec s3fs"):
        download_and_load_state(_spring_graph(), "s3://bucket/run/ckpt.npz")


def test_memory_filesystem_round_trip(tmp_path):
    fsspec = pytest.importorskip(
        "fsspec",
        reason="fsspec is an optional [ci]/[dev] extra; the cloud-storage "
        "scheme round trip needs its in-memory filesystem",
    )
    gm = _spring_graph()
    for _ in range(7):
        gm.step()
    local, _mpath = ck.save_state_with_manifest(gm, tmp_path / "ckpt.npz")
    # "Upload" both files to the in-memory filesystem.
    fs = fsspec.filesystem("memory")
    for src, name in ((local, "ckpt.npz"), (Path(str(local) + ".manifest.json"),
                                            "ckpt.npz.manifest.json")):
        with fs.open(f"/run/{name}", "wb") as f:
            f.write(Path(src).read_bytes())
    fresh = _spring_graph()
    manifest = download_and_load_state(fresh, "memory:///run/ckpt.npz")
    assert manifest
    assert float(fresh.get_node_state("s")["position"]) == float(gm.get_node_state("s")["position"])


# ---------------------------------------------------------------------------
# Package export and the deprecated alias in the core checkpoint module
# ---------------------------------------------------------------------------


def test_cloud_package_lazily_exports_transport():
    import maddening.cloud

    assert "download_and_load_state" in maddening.cloud.__all__
    assert maddening.cloud.download_and_load_state is download_and_load_state


def test_old_import_path_still_works_and_warns(
    bouncing_ball_graph, matching_empty_graph, tmp_path,
):
    npz_path, _ = ck.save_state_with_manifest(
        bouncing_ball_graph, tmp_path / "snap.npz",
    )
    with pytest.warns(DeprecationWarning, match=r"moved to maddening\.cloud\.resume"):
        manifest = ck.download_and_load_state(
            matching_empty_graph, f"file://{npz_path}", dest_dir=tmp_path / "resume",
        )
    assert manifest["schema_version"] == ck.CHECKPOINT_SCHEMA_VERSION
    _assert_ball_position_restored(bouncing_ball_graph, matching_empty_graph)


def test_old_import_path_forwards_arguments_and_errors(matching_empty_graph, tmp_path):
    # Same error, same message, through the alias — the warning fires first.
    with pytest.warns(DeprecationWarning):
        with pytest.raises(ValueError, match="Unsupported URL scheme 'ftp'"):
            ck.download_and_load_state(
                matching_empty_graph, "ftp://host/x.npz", dest_dir=tmp_path,
            )


def test_importing_old_path_does_not_warn():
    # Only the *call* is deprecated; importing the name must stay silent so
    # ``from ... import download_and_load_state`` keeps working under -W error.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        from maddening.core.simulation.checkpoint import (  # noqa: F401
            download_and_load_state as _alias,
        )


def test_core_checkpoint_module_does_not_import_cloud_or_fsspec():
    # Fresh interpreter: importing the core module must not pull in the cloud
    # package (which would invert the core <- cloud dependency direction) nor
    # fsspec (an optional extra).
    code = (
        "import sys\n"
        "import maddening.core.simulation.checkpoint\n"
        "bad = sorted(m for m in sys.modules "
        "if m == 'maddening.cloud' or m.startswith('maddening.cloud.') or m == 'fsspec')\n"
        "print(repr(bad))\n"
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env,
    )
    assert out.stdout.strip() == "[]", out.stdout
