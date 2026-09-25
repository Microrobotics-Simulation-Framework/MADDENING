"""A walled ``LBMNode`` is rebuilt with its walls by every save/reload path.

Until 0.4.0 the ``wall_mask`` constructor argument was kept on the node
alone -- not in ``params`` -- so the node that ``GraphManager.from_dict``
(or a USD stage) rebuilds from ``cls(name=, timestep=, **params)`` had no
walls, and the reloaded graph ran a different flow with no error
(MADD-ANO-034, since 0.1.0).  The mask is now recorded in ``params`` as
nested lists of ``bool``.

The fixture is a 12 x 8 D2Q9 channel with walls on both long sides and one
obstacle cell, driven by a uniform body force so that the walls shape the
flow: after 30 steps the wall-free domain's velocity is several times the
walled one's (``test_the_fixture_can_express_a_lost_mask``), so a mask lost
anywhere is visible in the trajectory, not only in the node's attributes.
"""

from __future__ import annotations

import importlib.util
import json

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.nodes.lbm import LBMNode

NX, NY = 12, 8
STEPS = 30
FORCE = jnp.asarray([2e-5, 0.0], jnp.float32)
REGISTRY = {"LBMNode": LBMNode}


def _walls() -> np.ndarray:
    wall = np.zeros((NX, NY), bool)
    wall[:, 0] = wall[:, -1] = True      # a channel
    wall[5, 3] = True                    # and an obstacle in it
    return wall


def _node(wall_mask=None, **kw) -> LBMNode:
    return LBMNode("fluid", 1.0, grid_shape=(NX, NY), lattice="D2Q9",
                   viscosity=0.1, wall_mask=wall_mask, **kw)


def _graph(node: LBMNode) -> GraphManager:
    gm = GraphManager()
    gm.add_node(node)
    gm.add_external_input("fluid", "body_force", shape=(2,))
    gm.compile()
    return gm


def _run(gm: GraphManager) -> dict:
    """``STEPS`` steps of the jitted step under the body force (a fresh
    graph per call: stepping moves the graph's own state)."""
    for _ in range(STEPS):
        gm.step(external_inputs={"fluid": {"body_force": FORCE}})
    return {k: np.asarray(v) for k, v in gm.get_node_state("fluid").items()}


@pytest.fixture(scope="module")
def walled_run() -> dict:
    return _run(_graph(_node(_walls())))


def _assert_same_run(got: dict, want: dict) -> None:
    assert sorted(got) == sorted(want)
    for k in want:
        assert got[k].dtype == want[k].dtype, k
        np.testing.assert_array_equal(got[k], want[k], err_msg=k)


def _usd_installed() -> bool:
    try:
        return importlib.util.find_spec("pxr") is not None
    except (ImportError, ValueError):
        return False


def test_the_fixture_can_express_a_lost_mask(walled_run):
    """Without the check below every other test could pass on a flow the
    walls do not shape."""
    open_run = _run(_graph(_node(None)))
    walled = float(np.abs(walled_run["velocity"]).max())
    wall_free = float(np.abs(open_run["velocity"]).max())
    assert wall_free > 3 * walled > 0.0


def test_the_mask_is_recorded_in_params_as_nested_lists_of_bool():
    node = _node(_walls())
    assert node.params["wall_mask"] == _walls().tolist()
    json.dumps(node.to_dict())          # JSON-faithful, no encoder needed


def test_a_node_rebuilt_from_its_params_has_the_same_walls():
    node = _node(_walls())
    d = json.loads(json.dumps(node.to_dict()))
    rebuilt = LBMNode(name=d["name"], timestep=d["timestep"], **d["params"])
    np.testing.assert_array_equal(np.asarray(rebuilt.initial_state()["wall_mask"]),
                                  _walls().astype(np.uint8))
    np.testing.assert_array_equal(np.asarray(rebuilt._wall_mask), _walls())
    assert rebuilt._has_walls
    assert rebuilt.to_dict() == node.to_dict()


def test_a_config_round_trip_through_json_steps_the_same_flow(walled_run):
    config = json.loads(json.dumps(_graph(_node(_walls())).to_dict()))
    reloaded = GraphManager.from_dict(config, REGISTRY)
    reloaded.compile()
    _assert_same_run(_run(reloaded), walled_run)


@pytest.mark.skipif(not _usd_installed(), reason="usd-core not installed")
def test_a_usd_round_trip_steps_the_same_flow(walled_run):
    from pxr import Usd

    from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(_graph(_node(_walls())), stage)
    reloaded = load_graph_from_usd(stage, node_registry=REGISTRY)
    reloaded.compile()
    _assert_same_run(_run(reloaded), walled_run)


def test_every_array_like_spelling_of_the_mask_builds_the_same_node():
    """A NumPy bool array, a JAX array, 0/1 integers and the nested lists a
    saved config carries are the same mask."""
    reference = _node(_walls()).to_dict()
    for spelling in (_walls().tolist(), jnp.asarray(_walls()),
                     _walls().astype(np.uint8), _walls().astype(np.float32)):
        assert _node(spelling).to_dict() == reference


def test_a_mask_of_the_wrong_shape_is_refused_in_every_spelling():
    wrong = np.zeros((NX, NY + 1), bool)
    for spelling in (wrong, wrong.tolist()):
        with pytest.raises(ValueError, match=r"wall_mask shape \(12, 9\) != grid_shape"):
            _node(spelling)


def test_a_wall_free_node_writes_the_config_it_wrote_before():
    """No mask, no key: a config of a wall-free node is unchanged, and the
    key never appears as ``null``."""
    assert "wall_mask" not in _node(None).params
    assert "wall_mask" not in _graph(_node(None)).to_dict()["nodes"][0]["params"]


def test_the_mask_is_structural_not_a_differentiable_parameter():
    """Recorded in ``params`` but never a leaf of the parameter pytree: a
    fit or an FIM over the graph must not see 96 booleans as parameters."""
    gm = _graph(_node(_walls()))
    assert "wall_mask" not in gm.params["nodes"]["fluid"]
    assert "wall_mask" not in _node(_walls()).params_pytree()
