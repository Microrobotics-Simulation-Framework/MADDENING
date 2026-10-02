"""``/checkpoint/save`` and ``/checkpoint/load`` touch only files strictly
inside the checkpoint root.

``path=""``, ``"."`` or ``"sub/.."`` resolved to the root itself, which the
check accepted; ``numpy.savez`` then appended ``.npz`` and wrote
``<root>.npz`` -- in the root's *parent* -- and ``/checkpoint/load`` read it
back.  A NUL byte in the path was a 500.  The check now runs on the file
NumPy actually touches.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode


@pytest.fixture
def server_dirs(tmp_path):
    parent = tmp_path / "server_cwd"
    root = parent / "checkpoints"
    root.mkdir(parents=True)
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
        gm.step()
    server = SimulationServer({"BallNode": BallNode}, graph_manager=gm,
                              checkpoint_root=str(root))
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    return gm, client, tmp_path, root


def _files(base: Path) -> list[str]:
    return sorted(str(p.relative_to(base)) for p in base.rglob("*") if p.is_file())


@pytest.mark.parametrize("path", ["", ".", "./", "sub/..", "a/b/../..", "a/../.",
                                  "a\x00b", "../checkpoints", "x" * 5000])
@pytest.mark.parametrize("op", ["save", "load"])
def test_a_path_that_is_not_a_file_inside_the_root_is_a_400(server_dirs, op, path):
    gm, client, base, root = server_dirs
    if op == "load":
        # the file the old check let a load read back
        (root.parent / "checkpoints.npz").write_bytes(b"not a checkpoint")
    before = _files(base)
    resp = client.post(f"/checkpoint/{op}", params={"path": path})
    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"]
    assert _files(base) == before


@pytest.mark.parametrize("path", ["", ".", "sub/.."])
def test_a_load_does_not_read_the_file_beside_the_root(server_dirs, path):
    """The audited read-back: with the root directory absent, the old load
    fell back to ``<root>.npz`` -- the file a root-equal save had written
    in the root's parent -- and restored it."""
    gm, client, base, root = server_dirs
    assert client.post("/checkpoint/save", params={"path": "real.npz"}).status_code == 200
    (root / "real.npz").rename(root.parent / "checkpoints.npz")
    root.rmdir()
    gm.step()
    before = float(gm.get_node_state("ball")["position"])
    resp = client.post("/checkpoint/load", params={"path": path})
    assert resp.status_code == 400, resp.text
    assert float(gm.get_node_state("ball")["position"]) == before


def test_a_symlink_inside_the_root_does_not_lead_out_of_it(server_dirs):
    gm, client, base, root = server_dirs
    outside = base / "outside.npz"
    assert client.post("/checkpoint/save", params={"path": "real.npz"}).status_code == 200
    (root / "real.npz").rename(outside)
    (root / "link.npz").symlink_to(outside)
    for op, path in [("load", "link.npz"), ("load", "link"), ("save", "link")]:
        resp = client.post(f"/checkpoint/{op}", params={"path": path})
        assert resp.status_code == 400, (op, path, resp.text)
        assert "must stay under" in resp.json()["detail"]


@pytest.mark.parametrize("path, written", [("ok.npz", "ok.npz"), ("plain", "plain.npz"),
                                           ("nested/deep.npz", "nested/deep.npz"),
                                           ("a/../b.npz", "b.npz")])
def test_a_file_inside_the_root_still_saves_and_loads(server_dirs, path, written):
    gm, client, base, root = server_dirs
    resp = client.post("/checkpoint/save", params={"path": path})
    assert resp.status_code == 200, resp.text
    assert Path(resp.json()["path"]) == (root / written).resolve()
    assert _files(base) == [f"server_cwd/checkpoints/{written}"]
    gm.step()
    resp = client.post("/checkpoint/load", params={"path": path})
    assert resp.status_code == 200, resp.text


def test_a_name_that_is_a_directory_loads_the_file_saved_under_it(server_dirs):
    """``nested`` is a directory (``nested/deep.npz`` was saved) and
    ``nested.npz`` the file ``save path=nested`` wrote: the load reads the
    file, where it used to hand the directory to NumPy."""
    gm, client, base, root = server_dirs
    assert client.post("/checkpoint/save", params={"path": "nested/deep.npz"}).status_code == 200
    assert client.post("/checkpoint/save", params={"path": "nested"}).status_code == 200
    resp = client.post("/checkpoint/load", params={"path": "nested"})
    assert resp.status_code == 200, resp.text


def test_a_missing_checkpoint_is_a_404(server_dirs):
    gm, client, base, root = server_dirs
    resp = client.post("/checkpoint/load", params={"path": "never.npz"})
    assert resp.status_code == 404, resp.text
