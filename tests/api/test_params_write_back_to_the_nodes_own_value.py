"""A params write that puts a leaf back to the node's own value is asked
the combined checks with the graph's other live values.

``PUT /graph/params`` counted a leaf as changed when it differed from the
node's own value (``node.params``), not from the live leaf the graph runs
with.  After a fit, or a ``POST /checkpoint/load`` (which moves
``gm.params`` and leaves ``node.params``), the two differ; a leaf written
back to the node's own value was then dropped from the write's checks and
still written, so every combined check -- the constructor asked with every
key at once, the graph a save would reload -- was asked with the leaf's
*old* live value.  A ``HeatNode`` rod took ``length`` and
``thermal_diffusivity`` past its Fourier limit with a 200, ran to NaN, and
saved a graph whose constructor refused it.  Plain REST reached it: PUT,
save, PUT, load, then PUT the original value back.

A leaf now counts as changed when it differs from the live leaf, and one
that goes back to the node's own value is asked as ``at_own_value`` -- the
per-key checks skip it (the node was built with it), the combined checks
take it -- the way a checkpoint load already asked it.  These tests pin
every door that writes a leaf: the multi-key and single-key ``PUT``, after
a fit and after a load, and the load itself; and that a write back that the
graph's live values do take is still a 200 whose save reloads.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import HeatNode

REGISTRY = {"HeatNode": HeatNode}
DT = 0.01
N = 16
# Fourier number dt * alpha / dx**2 = 0.45 at length 1 (the explicit limit is
# 0.5); 0.648 at the shorter length, where a fitted alpha of 0.05 gives 0.184.
A0 = 0.9 * 0.5 * (1.0 / N) ** 2 / DT
A_FIT = 0.05
L_SHORT = 1.0 / 1.2


def _rod_graph() -> GraphManager:
    gm = GraphManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_node(HeatNode("rod", DT, n_cells=N, length=1.0, thermal_diffusivity=A0,
                             initial_temperature=np.linspace(0.0, 1.0, N).tolist()))
        gm.compile()
    return gm


def _client(gm: GraphManager, root) -> TestClient:
    server = SimulationServer(REGISTRY, graph_manager=gm, checkpoint_root=str(root))
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _fit(gm: GraphManager, **values: float) -> None:
    """Move live leaves the way Python code (a fit) may: a ``gm.params``
    write, which leaves ``node.params`` at the constructor's values."""
    params = gm.params
    for key, value in values.items():
        params["nodes"]["rod"][key] = jnp.asarray(value, jnp.float32)
    gm.params = params


def _live(gm: GraphManager) -> dict:
    return {k: float(np.asarray(v)) for k, v in gm.params["nodes"]["rod"].items()
            if k in ("length", "thermal_diffusivity")}


def _fourier(live: dict) -> float:
    return DT * live["thermal_diffusivity"] / (live["length"] / N) ** 2


def _put(client: TestClient, **params):
    return client.put("/graph/params/rod", json={"params": params})


def _assert_the_save_reloads(gm: GraphManager) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        GraphManager.from_dict(gm.to_dict(), REGISTRY)


def test_a_write_back_after_a_fit_is_asked_with_the_fitted_values(tmp_path):
    """The auditor's case: after a fit moved alpha to 0.05, ``{alpha: own,
    length: short}`` is the pair the constructor refuses (Fourier 0.648)."""
    gm = _rod_graph()
    client = _client(gm, tmp_path)
    _fit(gm, thermal_diffusivity=A_FIT)
    before = _live(gm)
    r = _put(client, thermal_diffusivity=A0, length=L_SHORT)
    assert r.status_code == 400, r.text
    assert "thermal_diffusivity" in r.json()["detail"] and "length" in r.json()["detail"]
    assert "constructor refuses" in r.json()["detail"]
    assert _live(gm) == before
    assert gm._nodes["rod"].node.params["length"] == 1.0
    _assert_the_save_reloads(gm)


def test_a_single_key_written_back_after_a_load_is_asked_with_the_loaded_values(tmp_path):
    """REST alone: a load leaves ``length`` short and alpha fitted, the
    node's own alpha is the original; putting alpha back alone is the
    Fourier-0.648 rod."""
    gm = _rod_graph()
    client = _client(gm, tmp_path)
    assert _put(client, thermal_diffusivity=A_FIT).status_code == 200
    assert _put(client, length=L_SHORT).status_code == 200
    assert client.post("/checkpoint/save?path=short.npz").status_code == 200
    assert _put(client, length=1.0).status_code == 200
    assert _put(client, thermal_diffusivity=A0).status_code == 200
    assert client.post("/checkpoint/load?path=short.npz").status_code == 200
    node = gm._nodes["rod"].node
    assert node.params["thermal_diffusivity"] == pytest.approx(A0)   # the node's own
    loaded = _live(gm)
    assert loaded["length"] == pytest.approx(L_SHORT)
    r = _put(client, thermal_diffusivity=A0)
    assert r.status_code == 400, r.text
    assert _live(gm) == loaded
    assert _fourier(_live(gm)) < 0.5
    _assert_the_save_reloads(gm)


def test_a_two_key_write_back_after_a_load_is_asked_with_the_loaded_values(tmp_path):
    gm = _rod_graph()
    client = _client(gm, tmp_path)
    assert _put(client, thermal_diffusivity=A_FIT).status_code == 200
    assert client.post("/checkpoint/save?path=small.npz").status_code == 200
    assert _put(client, thermal_diffusivity=A0).status_code == 200
    assert client.post("/checkpoint/load?path=small.npz").status_code == 200
    loaded = _live(gm)
    r = _put(client, thermal_diffusivity=A0, length=L_SHORT)
    assert r.status_code == 400, r.text
    assert _live(gm) == loaded
    _assert_the_save_reloads(gm)


def test_a_write_back_the_live_values_take_is_still_a_200_whose_save_reloads(tmp_path):
    """The control: alpha back to its own value with the length at 1 is
    the rod as built (Fourier 0.45)."""
    gm = _rod_graph()
    client = _client(gm, tmp_path)
    _fit(gm, thermal_diffusivity=A_FIT)
    r = _put(client, thermal_diffusivity=A0)
    assert r.status_code == 200, r.text
    assert _live(gm)["thermal_diffusivity"] == pytest.approx(A0)
    _assert_the_save_reloads(gm)
    assert np.all(np.isfinite(np.asarray(gm.run_scan(50)["rod"]["temperature"])))


def test_a_write_of_the_live_value_changes_nothing_and_is_a_200(tmp_path):
    """A leaf equal to the live leaf is not a change, even where it
    differs from the node's own value: the graph runs, and saves, that
    value already."""
    gm = _rod_graph()
    client = _client(gm, tmp_path)
    _fit(gm, thermal_diffusivity=A_FIT, length=L_SHORT)
    before = _live(gm)
    r = _put(client, thermal_diffusivity=A_FIT, length=L_SHORT)
    assert r.status_code == 200, r.text
    assert _live(gm) == before
    _assert_the_save_reloads(gm)


def test_a_load_that_puts_a_leaf_back_to_the_nodes_own_value_is_asked_with_the_live_values(
        tmp_path):
    """The load door: a checkpoint whose alpha is the node's own value and
    whose length is the live (fitted) one.  Only alpha changes, and it is
    asked together with the live length."""
    source = _rod_graph()
    _fit(source, length=L_SHORT)              # alpha stays the node's own A0
    source.save_state(str(tmp_path / "own_alpha_short_rod.npz"))
    gm = _rod_graph()
    client = _client(gm, tmp_path)
    _fit(gm, thermal_diffusivity=A_FIT, length=L_SHORT)
    before = _live(gm)
    r = client.post("/checkpoint/load?path=own_alpha_short_rod.npz")
    assert r.status_code == 400, r.text
    assert "thermal_diffusivity" in r.json()["detail"]
    assert _live(gm) == before


def test_a_write_back_before_the_first_compile_is_still_a_200(tmp_path):
    """Before a compile the live view is the node's own pytree, so the
    node's own value is unchanged and asks nothing."""
    gm = GraphManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_node(HeatNode("rod", DT, n_cells=N, length=1.0, thermal_diffusivity=A0))
    client = _client(gm, tmp_path)
    assert _put(client, thermal_diffusivity=A0).status_code == 200
    assert _put(client, thermal_diffusivity=A0, length=L_SHORT).status_code == 400
