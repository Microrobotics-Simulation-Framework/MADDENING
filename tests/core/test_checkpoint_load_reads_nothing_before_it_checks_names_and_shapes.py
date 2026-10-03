"""A checkpoint load checks every member's name, shape and dtype before it
reads any member's data, and never reads a member it does not restore.

``load_state`` decompressed every member of the ``.npz`` first
(``{key: data[key] for key in data.files}``) and only then compared names
and shapes with the graph.  A small file therefore cost as much memory as
its members *declared*: a 2 MB archive holding one extra member of 2 GiB of
zeros filled 2 GiB before ``POST /checkpoint/load`` answered "node
mismatch".  The members' ``.npy`` headers are now read first; a name the
graph does not have, a shape that is not the live leaf's, and a dtype that
is not a number are refused from the header, and a member the load ignores
(an unknown params owner, a stale ``_meta`` key) is never read.  What a load
reads is bounded by the live graph's own sizes.
"""

from __future__ import annotations

import io
import warnings
import zipfile

import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.simulation import checkpoint as ckpt
from maddening.nodes import BallNode, SpringDamperNode


def _graph() -> GraphManager:
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=2.0))
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=40.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.compile()
    return gm


def _member(shape, dtype="<f4", data: bytes = b"") -> bytes:
    """A ``.npy`` member: a header declaring *shape* and *dtype*, then
    *data* -- which may be far shorter than the header declares."""
    out = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        out, {"descr": dtype, "fortran_order": False, "shape": tuple(shape)})
    return out.getvalue() + data


def _archive(path, good: str, extra: dict) -> None:
    """*good*'s members, plus or instead of *extra* (``{name: bytes}``,
    ``None`` to drop a member)."""
    with zipfile.ZipFile(good) as src, zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        for name in src.namelist():
            if name in extra:
                continue
            out.writestr(name, src.read(name))
        for name, data in extra.items():
            if data is not None:
                out.writestr(name, data)


@pytest.fixture
def reads(monkeypatch):
    """The members a load reads the data of."""
    seen: list[str] = []
    original = ckpt._CheckpointArchive.read

    def read(self, key):
        seen.append(key)
        return original(self, key)

    monkeypatch.setattr(ckpt._CheckpointArchive, "read", read)
    return seen


@pytest.fixture
def good(tmp_path):
    gm = _graph()
    path = tmp_path / "good.npz"
    gm.save_state(str(path))
    return str(path)


#: A member that declares 1 GiB and holds none of it: reading it would
#: allocate the declared size before it found the data missing.
HUGE = (2 ** 28,)


def test_an_extra_node_is_refused_before_any_member_is_read(tmp_path, good, reads):
    path = tmp_path / "extra.npz"
    _archive(path, good, {"intruder/x.npy": _member(HUGE)})
    with pytest.raises(ValueError, match="extra in checkpoint: \\['intruder'\\]"):
        _graph().load_state(str(path))
    assert reads == []


def test_a_field_of_another_shape_is_refused_from_its_header(tmp_path, good, reads):
    path = tmp_path / "shape.npz"
    _archive(path, good, {"ball/position.npy": _member(HUGE)})
    with pytest.raises(ValueError, match="has shape \\(268435456,\\)"):
        _graph().load_state(str(path))
    assert reads == []


def test_a_params_leaf_of_another_shape_is_refused_from_its_header(tmp_path, good, reads):
    gm = _graph()
    names = zipfile.ZipFile(good).namelist()
    leaf = next(n for n in names if n.startswith("_params/spring/"))
    path = tmp_path / "param.npz"
    _archive(path, good, {leaf: _member(HUGE)})
    with pytest.raises(ValueError, match="has shape"):
        gm.load_state(str(path))
    assert not any(k == leaf[:-4] for k in reads)


def test_members_the_load_ignores_are_never_read(tmp_path, good, reads):
    """An unknown params owner or key and a stale ``_meta`` key are ignored
    by a load: their declared sizes cost nothing now."""
    path = tmp_path / "ignored.npz"
    _archive(path, good, {"_params/ghost/k.npy": _member(HUGE),
                          "_params/ball/ghost_key.npy": _member(HUGE),
                          "_meta/stale_key.npy": _member(HUGE)})
    gm = _graph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.load_state(str(path))
    assert not any(k in reads for k in ("_params/ghost/k", "_params/ball/ghost_key",
                                        "_meta/stale_key"))
    assert reads, "the members the load restores are read"


def test_a_member_of_text_is_refused_from_its_header(tmp_path, good, reads):
    path = tmp_path / "text.npz"
    _archive(path, good, {"ball/position.npy": _member((), "<U1000000")})
    with pytest.raises(ValueError, match="holds <U1000000 data, not a number"):
        _graph().load_state(str(path))
    assert "ball/position" not in reads


@pytest.mark.parametrize("extra, words", [
    ({"notes.txt": b"hello"}, "is not an .npy array"),
    ({"ball/position.npy": _member((), "|O")}, "holds Python objects"),
    ({"ball/position.npy": b"\x93NUMPY\x09\x00garbage"}, "npy header version"),
    ({"ball/position.npy": b"\x93NUMPY\x01\x00\x10\x00{not a header}\n"}, "damaged"),
    ({"ball/position.npy": b"not npy at all"}, "damaged"),
])
def test_a_member_that_is_not_a_plain_array_is_not_a_checkpoint(tmp_path, good, reads,
                                                                  extra, words):
    path = tmp_path / "odd.npz"
    _archive(path, good, extra)
    with pytest.raises(ckpt.CheckpointFormatError, match=words):
        _graph().load_state(str(path))
    assert reads == []


def test_a_valid_checkpoint_reads_each_member_it_restores_once(tmp_path, good, reads):
    gm = _graph()
    gm.step()
    gm.load_state(good)
    names = {n[:-4] for n in zipfile.ZipFile(good).namelist()}
    assert sorted(reads) == sorted(names)
    assert float(gm.get_node_state("ball")["position"]) == 2.0


def test_the_rest_route_refuses_the_archive_without_reading_it(tmp_path, reads):
    """The auditor's case through ``POST /checkpoint/load``: a 400, and the
    intruder's data never read."""
    from maddening.api.server import SimulationServer
    from tests._loopback_client import LoopbackTestClient as TestClient

    gm = _graph()
    server = SimulationServer({"BallNode": BallNode, "SpringDamperNode": SpringDamperNode},
                              graph_manager=gm, checkpoint_root=str(tmp_path))
    client = TestClient(server.create_app(), raise_server_exceptions=False)
    assert client.post("/checkpoint/save?path=good.npz").status_code == 200
    _archive(tmp_path / "bomb.npz", str(tmp_path / "good.npz"),
             {"intruder/x.npy": _member(HUGE)})
    reads.clear()
    resp = client.post("/checkpoint/load?path=bomb.npz")
    assert resp.status_code == 400 and "intruder" in resp.text, resp.text
    assert reads == []
    assert client.post("/checkpoint/load?path=good.npz").status_code == 200
