"""A USD stage naming a node no machine holds is refused before the node's
constructor runs, as ``GraphManager.from_dict`` refuses the same config.

A stage is untrusted input, and ``load_graph_from_usd`` calls each node's
class with the params the stage carries: a ``WaveletAdaptiveNode`` with
``n_levels=10_000_000`` used to run the loading process into the OOM killer.
"""

import functools
import json
import warnings

import pytest
from pxr import Usd

from maddening.core.graph_manager import GraphManager
from maddening.nodes.adaptive import WaveletAdaptiveNode
from maddening.nodes.lbm import LBMNode
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd


#: The classes these stages name; a stage is untrusted input, so the load
#: is told which classes it may build.
REGISTRY = {f"{cls.__module__}.{cls.__name__}": cls for cls in (WaveletAdaptiveNode, LBMNode)}


def _stage_with(node, edit):
    gm = GraphManager()
    gm.add_node(node)
    stage = Usd.Stage.CreateInMemory()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        save_graph_to_usd(gm, stage)
    attr = stage.GetPrimAtPath(f"/Simulation/nodes/{node.name}").GetAttribute(
        "maddening:paramsJson")
    attr.Set(json.dumps({**json.loads(attr.Get()), **edit}))
    return stage


@pytest.mark.parametrize("cls, kwargs, edit", [
    (WaveletAdaptiveNode, {"dim": 1, "n_levels": 3}, {"n_levels": 10_000_000}),
    (LBMNode, {"grid_shape": (4, 4, 4)}, {"grid_shape": [10 ** 6] * 3}),
], ids=["wavelet-levels", "lbm-grid"])
def test_a_stage_naming_an_impossible_node_is_refused_unbuilt(monkeypatch, cls, kwargs, edit):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stage = _stage_with(cls("w", 1.0, **kwargs), edit)
    calls = []
    real_init = cls.__init__

    @functools.wraps(real_init)
    def guarded(self, *args, **kwargs):
        calls.append(kwargs)
        raise AssertionError(f"{cls.__name__} was constructed")

    monkeypatch.setattr(cls, "__init__", guarded)
    with pytest.raises(ValueError, match=r"node 'w' \(" + cls.__name__
                       + r"\) cannot be built on this machine"):
        load_graph_from_usd(stage, node_registry=REGISTRY)
    assert calls == []


def test_a_stage_of_an_ordinary_node_still_loads():
    stage = _stage_with(LBMNode("w", 1.0, grid_shape=(4, 4, 4)), {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm = load_graph_from_usd(stage, node_registry=REGISTRY)
    assert list(gm.get_node("w").params["grid_shape"]) == [4, 4, 4]
