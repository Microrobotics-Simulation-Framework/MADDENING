"""Robustness of the resume-from-URL transport (``maddening.cloud.resume``).

Each test names the invariant it pins:

* the manifest URL is derived on the URL *path* (query string preserved),
  with an explicit ``manifest_url=`` override and ``RESUME_MANIFEST_URL``
  in the entry point;
* HTTP fetches time out instead of hanging container start-up;
* downloads stream to disk in chunks and multi-MiB files round-trip;
* the default temporary directory is removed on success and on failure,
  a caller-supplied one is kept;
* URL edge cases (empty, directory, percent-encoded ``file://``) fail
  clearly or work;
* the entry point redacts the query string in its logs, logs the
  manifest's key fields, and says plainly when the graph is empty.
"""

from __future__ import annotations

import functools
import http.server
import logging
import os
import socket
import socketserver
import threading
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.cloud import entrypoint, resume
from maddening.cloud.resume import download_and_load_state
from maddening.core.graph_manager import GraphManager
from maddening.core.simulation import checkpoint as ck
from maddening.nodes.spring import SpringDamperNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _spring_graph(steps: int = 0) -> GraphManager:
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, initial_position=0.0))
    gm.compile()
    for _ in range(steps):
        gm.step()
    return gm


def _position(gm: GraphManager) -> float:
    return float(gm.get_node_state("s")["position"])


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args, **kwargs):  # noqa: D401 — silence the server
        pass


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


@pytest.fixture
def http_dir(tmp_path):
    """Serve ``tmp_path / "www"`` over a local ``http.server``; yields (url, dir)."""
    www = tmp_path / "www"
    www.mkdir()
    handler = functools.partial(_QuietHandler, directory=str(www))
    httpd = _Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", www
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def saved_checkpoint(tmp_path):
    """(source graph, npz path, manifest path) of a stepped spring graph."""
    gm = _spring_graph(steps=7)
    (tmp_path / "src").mkdir()
    npz, manifest = ck.save_state_with_manifest(gm, tmp_path / "src" / "snap.npz")
    return gm, Path(npz), Path(manifest)


# ---------------------------------------------------------------------------
# F2: manifest URL derived on the path; query string preserved; override
# ---------------------------------------------------------------------------


class TestManifestUrl:
    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://h/run/snap.npz", "https://h/run/snap.npz.manifest.json"),
            (
                "https://h/run/snap.npz?X-Amz-Signature=abc&X-Amz-Expires=60",
                "https://h/run/snap.npz.manifest.json?X-Amz-Signature=abc&X-Amz-Expires=60",
            ),
            ("https://h/snap.npz#frag", "https://h/snap.npz.manifest.json#frag"),
            ("https://h/snap.npz?q=1#frag", "https://h/snap.npz.manifest.json?q=1#frag"),
            ("file:///tmp/a%20b/snap.npz", "file:///tmp/a%20b/snap.npz.manifest.json"),
            ("s3://bucket/run/snap.npz?versionId=7", "s3://bucket/run/snap.npz.manifest.json?versionId=7"),
            ("memory:///run/snap.npz", "memory:///run/snap.npz.manifest.json"),
            ("/tmp/snap.npz", "/tmp/snap.npz.manifest.json"),
        ],
    )
    def test_manifest_url_rewrites_path_and_keeps_query_and_fragment(self, url, expected):
        assert resume._manifest_url_for(url) == expected

    def test_http_url_with_query_string_loads_manifest(self, http_dir, saved_checkpoint, tmp_path):
        base, www = http_dir
        src_gm, npz, manifest = saved_checkpoint
        (www / npz.name).write_bytes(npz.read_bytes())
        (www / manifest.name).write_bytes(manifest.read_bytes())
        fresh = _spring_graph()
        got = download_and_load_state(
            fresh, f"{base}/snap.npz?X-Amz-Signature=abc&X-Amz-Expires=60",
            dest_dir=tmp_path / "dl",
        )
        assert got["schema_version"] == ck.CHECKPOINT_SCHEMA_VERSION
        assert _position(fresh) == _position(src_gm)

    def test_explicit_manifest_url_overrides_derivation(self, http_dir, saved_checkpoint, tmp_path):
        base, www = http_dir
        src_gm, npz, manifest = saved_checkpoint
        (www / "snap.npz").write_bytes(npz.read_bytes())
        # Manifest lives under a different name (presigned per object).
        (www / "other-name.json").write_bytes(manifest.read_bytes())
        fresh = _spring_graph()
        got = download_and_load_state(
            fresh, f"{base}/snap.npz?sig=A",
            manifest_url=f"{base}/other-name.json?sig=B",
            dest_dir=tmp_path / "dl",
        )
        assert got["sha256"] == ck.compute_checkpoint_hash(npz)
        assert _position(fresh) == _position(src_gm)

    def test_explicit_manifest_url_is_validated_against_the_allow_list(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported URL scheme 'ftp'"):
            download_and_load_state(
                _spring_graph(), "https://h/snap.npz",
                manifest_url="ftp://h/snap.npz.manifest.json", dest_dir=tmp_path,
            )

    def test_entrypoint_honours_resume_manifest_url_env(
        self, http_dir, saved_checkpoint, caplog,
    ):
        base, www = http_dir
        src_gm, npz, manifest = saved_checkpoint
        (www / "snap.npz").write_bytes(npz.read_bytes())
        (www / "presigned-manifest.json").write_bytes(manifest.read_bytes())
        server = _FakeServer(_spring_graph())
        with caplog.at_level(logging.INFO, logger="maddening.cloud.entrypoint"):
            got = entrypoint.resume_from_env(server, {
                "RESUME_FROM_URL": f"{base}/snap.npz?sig=A",
                "RESUME_MANIFEST_URL": f"{base}/presigned-manifest.json?sig=B",
            })
        assert got and got["schema_version"] == ck.CHECKPOINT_SCHEMA_VERSION
        assert _position(server.gm) == _position(src_gm)


# ---------------------------------------------------------------------------
# F3: timeout
# ---------------------------------------------------------------------------


@pytest.fixture
def stalled_server():
    """A socket that accepts connections and never replies; yields its URL."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    accepted: list[socket.socket] = []
    stop = threading.Event()

    def _accept_forever():
        sock.settimeout(0.1)
        while not stop.is_set():
            try:
                conn, _ = sock.accept()
            except (socket.timeout, OSError):
                continue
            accepted.append(conn)  # keep it open, never respond

    thread = threading.Thread(target=_accept_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}/snap.npz"
    finally:
        stop.set()
        thread.join(timeout=2)
        for conn in accepted:
            conn.close()
        sock.close()


class TestTimeout:
    def test_stalled_http_server_raises_timeout_error_within_timeout(self, stalled_server, tmp_path):
        t0 = time.monotonic()
        with pytest.raises(TimeoutError, match=r"timed out after 0\.5 s"):
            download_and_load_state(
                _spring_graph(), stalled_server, dest_dir=tmp_path, timeout=0.5,
            )
        assert time.monotonic() - t0 < 10.0

    def test_timeout_error_message_redacts_the_query_string(self, stalled_server, tmp_path):
        with pytest.raises(TimeoutError) as ei:
            download_and_load_state(
                _spring_graph(), stalled_server + "?X-Amz-Signature=SECRET",
                dest_dir=tmp_path, timeout=0.3,
            )
        assert "SECRET" not in str(ei.value)
        assert "<redacted>" in str(ei.value)

    def test_default_timeout_is_finite_and_passed_to_urlopen(self, monkeypatch, tmp_path):
        seen = {}

        def fake_urlopen(url, timeout=None):
            seen["timeout"] = timeout
            raise OSError("stop here")

        monkeypatch.setattr(resume.urllib.request, "urlopen", fake_urlopen)
        with pytest.raises(OSError, match="stop here"):
            resume._fetch("http://h/snap.npz", tmp_path / "x.npz")
        assert seen["timeout"] == resume.DEFAULT_TIMEOUT
        assert 0 < resume.DEFAULT_TIMEOUT < float("inf")

    def test_entrypoint_reads_timeout_from_env(self, monkeypatch, tmp_path):
        seen = {}

        def fake_download(gm, url, **kwargs):
            seen.update(kwargs)
            return {"schema_version": 1, "extra": {}}

        monkeypatch.setattr(resume, "download_and_load_state", fake_download)
        server = _FakeServer(_spring_graph())
        entrypoint.resume_from_env(server, {
            "RESUME_FROM_URL": "https://h/snap.npz", "MADDENING_RESUME_TIMEOUT": "7.5",
        })
        assert seen["timeout"] == 7.5

    def test_fsspec_timeout_options_are_best_effort_per_backend(self):
        assert resume._fsspec_timeout_options("s3", 5.0) == {
            "config_kwargs": {"connect_timeout": 5.0, "read_timeout": 5.0},
        }
        assert resume._fsspec_timeout_options("gs", 5.0) == {"timeout": 5.0}
        assert resume._fsspec_timeout_options("memory", 5.0) == {}


# ---------------------------------------------------------------------------
# F7: streaming downloads
# ---------------------------------------------------------------------------


class TestStreaming:
    def test_multi_mib_http_download_round_trips(self, http_dir, tmp_path):
        base, www = http_dir
        payload = os.urandom(3 * (1 << 20) + 12345)  # > 3 chunks, not chunk-aligned
        (www / "big.npz").write_bytes(payload)
        dest = tmp_path / "big.npz"
        resume._fetch(f"{base}/big.npz", dest)
        assert dest.read_bytes() == payload

    def test_http_fetch_never_reads_the_whole_body_at_once(self, monkeypatch, tmp_path):
        payload = os.urandom(2 * (1 << 20) + 7)
        reads: list[int | None] = []

        class _Stream:
            def __init__(self):
                self._pos = 0

            def read(self, n=None):
                reads.append(n)
                assert n is not None and 0 < n <= resume._CHUNK_SIZE, "unbounded read"
                chunk = payload[self._pos:self._pos + n]
                self._pos += len(chunk)
                return chunk

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        monkeypatch.setattr(resume.urllib.request, "urlopen", lambda url, timeout=None: _Stream())
        dest = tmp_path / "x.npz"
        resume._fetch("https://h/x.npz", dest)
        assert dest.read_bytes() == payload
        assert len(reads) >= 3

    def test_multi_mib_file_url_round_trips(self, tmp_path):
        payload = os.urandom(2 * (1 << 20) + 1)
        src = tmp_path / "big.npz"
        src.write_bytes(payload)
        dest = tmp_path / "out" / "big.npz"
        dest.parent.mkdir()
        resume._fetch(f"file://{src}", dest)
        assert dest.read_bytes() == payload

    def test_multi_mib_fsspec_download_round_trips(self, tmp_path):
        fsspec = pytest.importorskip(
            "fsspec",
            reason="fsspec is an optional [ci]/[dev] extra; the streaming fsspec "
            "download needs its in-memory filesystem",
        )
        payload = os.urandom(2 * (1 << 20) + 99)
        fs = fsspec.filesystem("memory")
        with fs.open("/stream/big.npz", "wb") as f:
            f.write(payload)
        dest = tmp_path / "big.npz"
        resume._fetch("memory:///stream/big.npz", dest)
        assert dest.read_bytes() == payload


# ---------------------------------------------------------------------------
# F6: temporary directory lifecycle
# ---------------------------------------------------------------------------


def _resume_tmp_dirs(root: Path) -> set[Path]:
    return {p for p in root.iterdir() if p.name.startswith("maddening_resume_")}


class TestTempDirLifecycle:
    @pytest.fixture(autouse=True)
    def _isolated_tmp(self, tmp_path, monkeypatch):
        self.tmp_root = tmp_path / "tmproot"
        self.tmp_root.mkdir()
        monkeypatch.setenv("TMPDIR", str(self.tmp_root))
        import tempfile
        monkeypatch.setattr(tempfile, "tempdir", None)  # re-read TMPDIR

    def test_default_dest_dir_is_removed_after_successful_load(self, saved_checkpoint):
        src_gm, npz, _ = saved_checkpoint
        fresh = _spring_graph()
        download_and_load_state(fresh, f"file://{npz}")
        assert _position(fresh) == _position(src_gm)
        assert _resume_tmp_dirs(self.tmp_root) == set()

    def test_default_dest_dir_is_removed_after_failed_download(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            download_and_load_state(_spring_graph(), f"file://{tmp_path}/missing.npz")
        assert _resume_tmp_dirs(self.tmp_root) == set()

    def test_default_dest_dir_is_removed_after_failed_integrity_check(self, saved_checkpoint):
        _, npz, manifest = saved_checkpoint
        npz.write_bytes(npz.read_bytes() + b"\xff")
        with pytest.raises(ck.CheckpointIntegrityError):
            download_and_load_state(_spring_graph(), f"file://{npz}")
        assert _resume_tmp_dirs(self.tmp_root) == set()

    def test_caller_supplied_dest_dir_is_kept_with_its_files(self, saved_checkpoint, tmp_path):
        src_gm, npz, _ = saved_checkpoint
        dest = tmp_path / "keep"
        download_and_load_state(_spring_graph(), f"file://{npz}", dest_dir=dest)
        assert sorted(p.name for p in dest.iterdir()) == ["snap.npz", "snap.npz.manifest.json"]

    def test_caller_supplied_dest_dir_is_kept_after_failure(self, tmp_path):
        dest = tmp_path / "keep"
        with pytest.raises(FileNotFoundError):
            download_and_load_state(
                _spring_graph(), f"file://{tmp_path}/missing.npz", dest_dir=dest,
            )
        assert dest.is_dir()


# ---------------------------------------------------------------------------
# F8: URL edge cases
# ---------------------------------------------------------------------------


class TestUrlEdgeCases:
    @pytest.mark.parametrize("url", ["", "   "])
    def test_empty_url_is_rejected_with_value_error(self, url, tmp_path):
        with pytest.raises(ValueError, match="empty checkpoint URL"):
            download_and_load_state(_spring_graph(), url, dest_dir=tmp_path)

    def test_directory_file_url_is_rejected_with_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="is a directory"):
            download_and_load_state(_spring_graph(), f"file://{tmp_path}", dest_dir=tmp_path / "d")

    def test_bare_directory_path_is_rejected_with_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="is a directory"):
            download_and_load_state(_spring_graph(), str(tmp_path), dest_dir=tmp_path / "d")

    def test_percent_encoded_file_url_is_unquoted(self, tmp_path):
        gm = _spring_graph(steps=3)
        (tmp_path / "a b").mkdir()
        npz, _ = ck.save_state_with_manifest(gm, tmp_path / "a b" / "my snap.npz")
        url = "file://" + str(npz).replace(" ", "%20")
        fresh = _spring_graph()
        download_and_load_state(fresh, url, dest_dir=tmp_path / "dl")
        assert _position(fresh) == _position(gm)
        assert (tmp_path / "dl" / "my snap.npz").exists()

    def test_windows_drive_letter_is_reported_as_unsupported_scheme(self, tmp_path):
        with pytest.raises(ValueError, match=r"Unsupported URL scheme 'c'.*drive-letter"):
            download_and_load_state(_spring_graph(), r"C:\Users\n\snap.npz", dest_dir=tmp_path)

    def test_non_string_url_is_a_type_error(self, tmp_path):
        with pytest.raises(TypeError, match="must be a str"):
            download_and_load_state(_spring_graph(), tmp_path / "x.npz", dest_dir=tmp_path)


# ---------------------------------------------------------------------------
# F4: entry-point logging
# ---------------------------------------------------------------------------


class _FakeServer:
    def __init__(self, gm):
        self.gm = gm


class TestEntrypointLogging:
    def test_redact_url_replaces_query_string_and_keeps_host_and_path(self):
        assert (
            entrypoint.redact_url("https://b.s3.amazonaws.com/run/sim.npz?X-Amz-Signature=SECRET")
            == "https://b.s3.amazonaws.com/run/sim.npz?<redacted>"
        )
        assert entrypoint.redact_url("https://h/sim.npz") == "https://h/sim.npz"
        assert entrypoint.redact_url("s3://bucket/sim.npz#frag") == "s3://bucket/sim.npz"

    def test_failed_resume_logs_redacted_url_and_is_non_fatal(self, caplog):
        server = _FakeServer(_spring_graph())
        url = "file:///nonexistent/dir/sim.npz?X-Amz-Signature=SECRET"
        with caplog.at_level(logging.INFO, logger="maddening.cloud.entrypoint"):
            got = entrypoint.resume_from_env(server, {"RESUME_FROM_URL": url})
        assert got is None
        text = caplog.text
        assert "starting fresh" in text
        assert "SECRET" not in text
        assert "file:///nonexistent/dir/sim.npz?<redacted>" in text

    def test_successful_resume_logs_manifest_key_fields(self, saved_checkpoint, caplog):
        src_gm, npz, _ = saved_checkpoint
        server = _FakeServer(_spring_graph())
        with caplog.at_level(logging.INFO, logger="maddening.cloud.entrypoint"):
            got = entrypoint.resume_from_env(server, {"RESUME_FROM_URL": f"file://{npz}"})
        assert got["schema_version"] == ck.CHECKPOINT_SCHEMA_VERSION
        line = next(r.getMessage() for r in caplog.records if "Resumed simulation state" in r.getMessage())
        assert f"schema_version={ck.CHECKPOINT_SCHEMA_VERSION}" in line
        assert f"size_bytes={npz.stat().st_size}" in line
        assert f"sha256={got['sha256'][:12]}" in line
        assert _position(server.gm) == _position(src_gm)

    def test_empty_graph_logs_that_resume_is_impossible(self, saved_checkpoint, caplog):
        from maddening.api.server import SimulationServer

        _, npz, _ = saved_checkpoint
        server = SimulationServer(node_registry={})  # exactly what main() builds
        with caplog.at_level(logging.INFO, logger="maddening.cloud.entrypoint"):
            got = entrypoint.resume_from_env(
                server, {"RESUME_FROM_URL": f"file://{npz}?sig=SECRET"},
            )
        assert got is None
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1
        msg = errors[0].getMessage()
        assert "graph has no nodes" in msg
        assert "resume is impossible until a graph is loaded" in msg
        assert "SECRET" not in msg
        assert "Checkpoint node mismatch" not in caplog.text

    def test_nothing_requested_is_a_silent_no_op(self, caplog):
        server = _FakeServer(_spring_graph())
        with caplog.at_level(logging.DEBUG, logger="maddening.cloud.entrypoint"):
            assert entrypoint.resume_from_env(server, {}) is None
        assert caplog.records == []
