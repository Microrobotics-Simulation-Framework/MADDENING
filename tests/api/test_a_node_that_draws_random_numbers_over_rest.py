"""The REST doors of a node that keeps its key's raw data in its state.

The convention and its in-process doors are in
``tests/core/test_a_node_that_draws_random_numbers.py``.  Here: the state
routes carry the key data as a list of integers, a ``PUT`` of it and the
checkpoint routes resume the stream, and a write that is not key data is a
400 naming the field.  The contrast: the server refuses a node whose state
holds a *typed* key, by name, because a key has no JSON form.

In process only.  The requests go to the paths in ``PATHS`` and nowhere
else -- none under ``/cloud``, ``/surrogate`` or ``/ws`` -- and the module
runs under :func:`tests.property.differential.no_cloud_launch`.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from tests._loopback_client import LoopbackTestClient as TestClient
from tests.core.test_a_node_that_draws_random_numbers import (
    AMP,
    DT,
    SEED,
    NoisySensor,
    TypedKeySensor,
    assert_draws,
    stream,
)
from tests.property.differential import no_cloud_launch

#: Every path this module requests.
PATHS = ("/graph/state", "/graph/state/n", "/graph/nodes", "/sim/step", "/sim/reset",
         "/checkpoint/save", "/checkpoint/load")
_NEVER = ("/cloud", "/surrogate", "/ws")
assert not [path for path in PATHS if path.startswith(_NEVER)]

REGISTRY = {"NoisySensor": NoisySensor, "TypedKeySensor": TypedKeySensor}


def _call(client, method, path, **kw):
    assert path in PATHS and not path.startswith(_NEVER), path
    return getattr(client, method)(path, **kw)


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    gm = GraphManager()
    gm.add_node(NoisySensor("n", DT, seed=SEED, amplitude=AMP))
    gm.compile()
    server = SimulationServer(REGISTRY, graph_manager=gm,
                              checkpoint_root=str(tmp_path_factory.mktemp("checkpoints")))
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _reset(client):
    assert _call(client, "post", "/sim/reset").status_code == 200


def _step(client):
    reply = _call(client, "post", "/sim/step")
    assert reply.status_code == 200, reply.text
    return _call(client, "get", "/graph/state/n").json()


def test_the_state_routes_carry_the_key_as_integers_and_a_put_resumes_the_stream(client):
    want, _ = stream(SEED, 5)
    _reset(client)
    state = _call(client, "get", "/graph/state/n").json()
    assert state["key"] == np.asarray(jax.random.key_data(jax.random.key(SEED))).tolist()
    assert all(isinstance(word, int) for word in state["key"])
    after_three = [_step(client) for _ in range(3)][-1]
    assert after_three["key"] == stream(SEED, 3)[1].tolist()
    assert_draws([_step(client)["noise"] for _ in range(2)], want[3:])
    # Written back, the state of three steps ago draws those two again.
    reply = _call(client, "put", "/graph/state/n", json={"state": after_three})
    assert reply.status_code == 200, reply.text
    assert _call(client, "get", "/graph/state/n").json()["key"] == after_three["key"]
    assert_draws([_step(client)["noise"] for _ in range(2)], want[3:])


def test_the_checkpoint_routes_resume_the_stream(client):
    want, _ = stream(SEED, 5)
    _reset(client)
    for _ in range(3):
        _step(client)
    saved = _call(client, "post", "/checkpoint/save", params={"path": "noise.npz"})
    assert saved.status_code == 200, saved.text
    assert_draws([_step(client)["noise"] for _ in range(2)], want[3:])
    loaded = _call(client, "post", "/checkpoint/load", params={"path": "noise.npz"})
    assert loaded.status_code == 200, loaded.text
    assert loaded.json()["state"]["n"]["key"] == stream(SEED, 3)[1].tolist()
    assert_draws([_step(client)["noise"] for _ in range(2)], want[3:])


@pytest.mark.parametrize("key, problem", [
    ([1.5, 2], "key: value 1.5 is not an integer, and the field holds uint32"),
    ([-1, 2], "key: value -1 is outside the range of uint32 [0, 4294967295]"),
    ([2**32, 1], "key: value 4294967296 is outside the range of uint32 [0, 4294967295]"),
    (["a", 1], "key: expected a number, got a string"),
    ([1], "key: expected 2 value(s) (shape (2,)), got 1"),
], ids=["a-fraction", "negative", "beyond-uint32", "text", "one-word"])
def test_a_write_that_is_not_key_data_is_refused_by_name_and_writes_nothing(client, key, problem):
    _reset(client)
    before = _call(client, "get", "/graph/state/n").json()
    reply = _call(client, "put", "/graph/state/n", json={"state": {**before, "key": key}})
    assert reply.status_code == 400, reply.text
    assert reply.json()["detail"] == problem
    assert _call(client, "get", "/graph/state/n").json() == before


def test_the_server_refuses_to_add_a_node_whose_state_holds_a_typed_key(client):
    """The exact refusal, and the same request with the key's data accepted."""
    typed = _call(client, "post", "/graph/nodes", json={
        "type": "TypedKeySensor", "name": "typed", "timestep": DT, "params": {"seed": 3}})
    assert typed.status_code == 400, typed.text
    assert typed.json()["detail"] == (
        "node 'typed' cannot run with these params: its state holds a PRNG key ('key'), "
        "which has no JSON form, so no reply could carry the state")
    assert "typed" not in _call(client, "get", "/graph/state").json()
    raw = _call(client, "post", "/graph/nodes", json={
        "type": "NoisySensor", "name": "second", "timestep": DT,
        "params": {"seed": 3, "amplitude": 1.0}})
    assert raw.status_code == 201, raw.text
    assert raw.json()["node"]["params"]["seed"] == 3


def test_a_served_graph_that_already_holds_a_typed_key_has_no_checkpoint_and_no_state_reply(
        tmp_path):
    """A graph handed to the server with a typed key in it: MADD-ANO-171 over REST.

    The checkpoint route answers 400 with ``save_state``'s refusal and
    writes nothing; no reply can carry the state, so the state route does
    not answer 200.
    """
    gm = GraphManager()
    gm.add_node(TypedKeySensor("t", DT, seed=SEED))
    gm.compile()
    server = SimulationServer(REGISTRY, graph_manager=gm, checkpoint_root=str(tmp_path))
    typed = TestClient(server.create_app(), raise_server_exceptions=False)
    saved = _call(typed, "post", "/checkpoint/save", params={"path": "typed.npz"})
    assert saved.status_code == 400, saved.text
    assert saved.json()["detail"].startswith(
        "could not save checkpoint 'typed.npz', nothing was written: JAX array with "
        "PRNGKey dtype cannot be converted to a NumPy array")
    assert not list(tmp_path.iterdir())
    assert _call(typed, "get", "/graph/state").status_code != 200
