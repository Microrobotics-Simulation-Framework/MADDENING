"""``save_graph_to_usd`` says which constructor arguments a stage does not
carry, as ``to_dict()`` does for a config.

A stage holds, for each node, its class, name, timestep and ``params``, and
``load_graph_from_usd`` rebuilds ``cls(name=..., timestep=..., **params)``:
a constructor argument a class keeps outside ``params`` is dropped by this
door exactly as by a config (MADD-ANO-273).  The writer reads
``node.params`` itself, not ``node.to_dict()``, so the warning ``to_dict()``
gives did not reach it, and for a time during 0.4.0 development the config
warned and the stage did not.  Every door that saves a node's constants now
makes the one call.
"""

from __future__ import annotations

import json
import warnings

import pytest
from pxr import Usd

from maddening.core.graph_manager import GraphManager
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd
from tests.core.test_to_dict_says_which_constructor_arguments_it_does_not_save import (
    DT, WARNING, AttributeSpring, BounceNode, ParamsSpring, _graph, _height,
)


def _registry(*classes):
    return {f"{cls.__module__}.{cls.__qualname__}": cls for cls in classes}


def _saved(gm):
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    return stage


def test_the_stage_door_gives_the_warning_the_config_door_gives():
    """The same node, the same text, attributed to the line that saved;
    and the stage reloads as the default graph, which is the loss it warns
    of."""
    gm = _graph(AttributeSpring(stiffness=40.0, name="spring", timestep=DT))
    with pytest.warns(UserWarning) as from_config:
        gm.to_dict()
    with pytest.warns(UserWarning) as from_stage:
        stage = _saved(gm)
    assert len(from_stage) == 1 and WARNING in str(from_stage[0].message)
    assert str(from_stage[0].message) == str(from_config[0].message)
    assert from_stage[0].filename == __file__
    prim = stage.GetPrimAtPath("/Simulation/nodes/spring")
    assert json.loads(prim.GetAttribute("maddening:paramsJson").Get()) == {}

    reloaded = load_graph_from_usd(stage, node_registry=_registry(BounceNode, AttributeSpring))
    reloaded.compile()
    at_12 = _height(_graph(AttributeSpring(name="spring", timestep=DT)))
    at_40 = _height(_graph(AttributeSpring(stiffness=40.0, name="spring", timestep=DT)))
    assert _height(reloaded) == at_12 != at_40


def test_a_node_that_hands_its_constants_on_is_saved_to_a_stage_without_a_word():
    gm = _graph(ParamsSpring(stiffness=40.0, name="spring", timestep=DT))
    stage = _saved(gm)                                      # warnings are errors here
    reloaded = load_graph_from_usd(stage, node_registry=_registry(BounceNode, ParamsSpring))
    reloaded.compile()
    assert reloaded.get_node("spring").params["stiffness"] == 40.0
    assert _height(reloaded) == _height(_graph(ParamsSpring(stiffness=40.0, name="spring",
                                                            timestep=DT)))


def test_a_wrapper_saved_to_a_stage_is_told_of_the_node_it_wraps():
    """The writer saves a wrapper by its ``params``, which are the wrapped
    node's: the warning names that node, as it does through ``to_dict()``,
    and a wrapper around a node that saves everything is silent."""
    def hybrid(spring):
        gm = GraphManager()
        gm.add_node(HybridNode(spring, lambda s, b, dt: {}))
        gm.compile()
        return gm

    with pytest.warns(UserWarning, match=rf"node 'spring' \(AttributeSpring\) {WARNING} stiffness"):
        _saved(hybrid(AttributeSpring(stiffness=40.0, name="spring", timestep=DT)))
    stage = _saved(hybrid(ParamsSpring(stiffness=40.0, name="spring", timestep=DT)))
    # What the guide says of this door: the stage names the wrapper's class,
    # which is not rebuilt from ``name, timestep, **params``, so the load
    # fails, and says so.  Pinned so that the sentence stays true.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        load_graph_from_usd(stage, node_registry=_registry(HybridNode, ParamsSpring))


def test_only_this_warning_is_new_at_the_stage_door():
    """A graph of nodes that save everything writes its stage as it did:
    nothing is said (recorded, so that a second kind of warning would be
    seen too)."""
    gm = _graph(ParamsSpring(name="spring", timestep=DT))
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        _saved(gm)
    assert [str(w.message) for w in said] == []
