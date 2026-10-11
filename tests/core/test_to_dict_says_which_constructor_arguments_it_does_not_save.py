"""``to_dict()`` says which constructor arguments a saved node cannot rebuild.

A saved node is its ``params`` and nothing else: every reload calls
``cls(name=..., timestep=..., **params)``.  A constructor argument a class
keeps as an attribute (``self.stiffness = stiffness``, the spelling the
quickstart taught until 0.4.0) is dropped by the save, and the reloaded
graph runs the constructor's default with nothing said: the quickstart's
own coupled graph built with ``SpringNode(stiffness=40.0)`` came back as
the ``k = 12`` graph, in every release since 0.1.0.

``SimulationNode.to_dict`` now warns, and only where a loss is certain:
the node was *constructed with* an argument for a named parameter, at a
value other than the parameter's default, and ``node.params`` has no such
key.  The tests hold the rule from both sides: every such node is named,
and nothing that saves correctly is (every test here runs with warnings
as errors, so an unexpected one fails it).
"""

from __future__ import annotations

import copy
import inspect
import json
import pickle
import re
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core import node as node_module
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.nodes import LBMNode
from tests.nodes.test_builtin_node_round_trip import CASES as BUILT_IN_CASES
from tests.nodes.test_builtin_node_round_trip import _build as _built_in

DT = 0.01
STEPS = 200
WARNING = "does not save the constructor argument"


class BounceNode(SimulationNode):
    """The quickstart's ball."""

    def initial_state(self):
        return {"position": jnp.array(5.0), "velocity": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        force = boundary_inputs.get("external_force", 0.0)
        velocity = state["velocity"] + (-9.81 + force) * dt
        position = state["position"] + velocity * dt
        velocity = jnp.where(position < 0, jnp.abs(velocity) * 0.8, velocity)
        return {"position": jnp.maximum(position, 0.0), "velocity": velocity}


class AttributeSpring(SimulationNode):
    """The quickstart's spring as it was taught until 0.4.0: its two
    constants kept as attributes, outside ``params``."""

    def __init__(self, stiffness=12.0, rest_height=2.0, **kwargs):
        super().__init__(**kwargs)
        self.stiffness = stiffness
        self.rest_height = rest_height

    def initial_state(self):
        return {"force": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        anchor = boundary_inputs.get("anchor_position", self.rest_height)
        return {"force": -self.stiffness * (anchor - self.rest_height)}


class ParamsSpring(SimulationNode):
    """The quickstart's spring as it is taught now: the constants handed
    to the base constructor, so they are in ``params``."""

    def __init__(self, stiffness=12.0, rest_height=2.0, **kwargs):
        super().__init__(stiffness=stiffness, rest_height=rest_height, **kwargs)

    def initial_state(self):
        return {"force": jnp.array(0.0)}

    def update(self, state, boundary_inputs, dt):
        stiffness, rest_height = self.params["stiffness"], self.params["rest_height"]
        anchor = boundary_inputs.get("anchor_position", rest_height)
        return {"force": -stiffness * (anchor - rest_height)}


def _graph(spring):
    gm = GraphManager()
    gm.add_node(BounceNode(name="ball", timestep=DT))
    gm.add_node(spring)
    gm.add_edge("ball", "spring", source_field="position", target_field="anchor_position")
    gm.add_edge("spring", "ball", source_field="force", target_field="external_force")
    gm.compile()
    return gm


def _height(gm):
    return float(gm.run_scan(STEPS)["ball"]["position"])


def _reload(config, spring_class):
    gm = GraphManager.from_dict(json.loads(json.dumps(config)),
                                {"BounceNode": BounceNode, spring_class.__name__: spring_class})
    gm.compile()
    return gm


# -- the defect, and the corrected node ------------------------------------

def test_the_quickstart_idiom_is_named_and_reloads_as_the_default_model():
    """The audit's reproducer: the graph built with ``stiffness=40.0`` saves
    an entry with no stiffness, and reloads as the default graph.  The save
    says so, naming the node, its class and the one argument that was
    given (``rest_height`` was left at its default and is not named)."""
    gm = _graph(AttributeSpring(stiffness=40.0, name="spring", timestep=DT))
    with pytest.warns(UserWarning) as said:
        config = gm.to_dict()
    assert len(said) == 1
    message = str(said[0].message)
    assert "node 'spring' (AttributeSpring)" in message
    assert f"{WARNING} stiffness." in message and "rest_height" not in message
    assert "runs with the default stiffness=12.0" in message
    assert "super().__init__(name, timestep, stiffness=stiffness)" in message
    assert next(n for n in config["nodes"] if n["name"] == "spring")["params"] == {}

    at_40 = _height(_graph(AttributeSpring(stiffness=40.0, name="spring", timestep=DT)))
    at_12 = _height(_graph(AttributeSpring(name="spring", timestep=DT)))
    assert abs(at_40 - at_12) > 0.05                      # the two models differ
    assert _height(_reload(config, AttributeSpring)) == at_12


def test_the_warning_reads_as_written():
    """The whole text, for one argument and for several (one of them with
    no default: the reload cannot build the node at all)."""
    node = AttributeSpring(stiffness=40.0, name="spring", timestep=DT)
    with pytest.warns(UserWarning) as said:
        node.to_dict()
    assert str(said[0].message) == (
        "node 'spring' (AttributeSpring): to_dict() does not save the constructor argument "
        "stiffness.  AttributeSpring.__init__ does not pass it to super().__init__(), so it is "
        "not in node.params, and a saved node is rebuilt as AttributeSpring(name=..., "
        "timestep=..., **params): a graph loaded from this config runs with the default "
        "stiffness=12.0.  Pass it on -- super().__init__(name, timestep, stiffness=stiffness) "
        "-- and read it from self.params; or override to_dict() if the node saves itself "
        "another way.")

    class Grid(SimulationNode):
        def __init__(self, name, timestep, shape, spacing=1.0, origin=(0.0, 0.0)):
            super().__init__(name, timestep)
            self.shape, self.spacing, self.origin = shape, spacing, origin

        def initial_state(self):
            return {"u": jnp.zeros(self.shape)}

        def update(self, state, boundary_inputs, dt):
            return state

    with pytest.warns(UserWarning) as said:
        Grid("g", DT, (4, 4), spacing=0.5, origin=(1.0, 0.0)).to_dict()
    assert str(said[0].message) == (
        "node 'g' (Grid): to_dict() does not save the constructor arguments origin, shape, "
        "spacing.  Grid.__init__ does not pass them to super().__init__(), so they are not in "
        "node.params, and a saved node is rebuilt as Grid(name=..., timestep=..., **params): "
        "a graph loaded from this config runs with the default origin=(0.0, 0.0), spacing=1.0 "
        "and cannot build the node (shape: no default).  Pass them on -- "
        "super().__init__(name, timestep, origin=origin, shape=shape, spacing=spacing) -- and "
        "read them from self.params; or override to_dict() if the node saves itself another "
        "way.")


def test_the_corrected_quickstart_node_round_trips_at_the_value_it_was_built_with():
    """Constants handed to ``super().__init__``: no warning, the entry
    carries them, and the reloaded graph is the graph that was saved, bit
    for bit."""
    gm = _graph(ParamsSpring(stiffness=40.0, name="spring", timestep=DT))
    config = gm.to_dict()
    entry = next(n for n in config["nodes"] if n["name"] == "spring")
    assert entry["params"] == {"stiffness": 40.0, "rest_height": 2.0}
    reloaded = _reload(config, ParamsSpring)
    assert reloaded.get_node("spring").params["stiffness"] == 40.0
    saved_height = _height(_graph(ParamsSpring(stiffness=40.0, name="spring", timestep=DT)))
    assert _height(reloaded) == saved_height
    # ... and it is the model the attribute-holding node ran, not the default.
    assert saved_height == _height(_graph(AttributeSpring(stiffness=40.0, name="spring",
                                                          timestep=DT)))
    assert saved_height != _height(_graph(ParamsSpring(name="spring", timestep=DT)))


# -- nothing that saves correctly is warned about ---------------------------

def _keeps(**given):
    """A node of a class that keeps ``gain`` (default 2.0) and ``taps``
    (default ``(0.0, 0.0)``) as attributes, built with ``given``."""

    class Keeps(SimulationNode):
        def __init__(self, name, timestep, gain=2.0, taps=(0.0, 0.0), mask=None):
            super().__init__(name, timestep)
            self.gain, self.taps, self.mask = gain, taps, mask

        def initial_state(self):
            return {"x": jnp.zeros(())}

        def update(self, state, boundary_inputs, dt):
            return state

    return Keeps("k", DT, **given)


@pytest.mark.parametrize("given", [
    {},                                                 # built with its defaults
    {"gain": 2.0},                                      # the default, spelt out
    {"gain": 2},                                        # the same number as an int
    {"gain": np.float32(2.0)},                          # ... and as a NumPy scalar
    {"taps": (0.0, 0.0)},
    {"taps": [0, 0]},                                   # the same numbers in a list
    {"taps": np.zeros(2)},                              # ... and in an array
    {"mask": None},
], ids=repr)
def test_an_argument_at_its_default_is_not_reported(given):
    """The reload passes the default, which is what the node was built
    with: nothing is lost, and nothing is said."""
    entry = _keeps(**given).to_dict()                   # warnings are errors here
    assert entry["params"] == {}


@pytest.mark.parametrize("given, named", [
    ({"gain": 2.5}, "argument gain."),
    ({"taps": (0.0, 1.0)}, "argument taps."),
    ({"taps": np.array([0.0, 1.0])}, "argument taps."),
    ({"taps": (0.0, 0.0, 0.0)}, "argument taps."),      # another length
    ({"mask": np.zeros(3, bool)}, "argument mask."),    # an array where the default is None
    ({"gain": float("nan")}, "argument gain."),
    ({"gain": 2.5, "mask": [True]}, "arguments gain, mask."),
], ids=lambda value: repr(value)[:40])
def test_an_argument_off_its_default_and_outside_params_is_reported(given, named):
    with pytest.warns(UserWarning, match=re.escape(f"{WARNING}{named[8:]}")):
        _keeps(**given).to_dict()


def test_what_the_rule_cannot_tell_is_not_reported():
    """Stated in ``_unsaved_constructor_arguments``: an argument a
    ``**kwargs`` took in, a class that saves itself its own way, a copy,
    and an attribute written after construction."""

    class Absorbs(SimulationNode):
        def __init__(self, name, timestep, **kwargs):
            self.gain = kwargs.pop("gain", 2.0)             # kept, and not saved
            super().__init__(name, timestep, **kwargs)

        def initial_state(self):
            return {"x": jnp.zeros(())}

        def update(self, state, boundary_inputs, dt):
            return state

    assert Absorbs("a", DT, gain=5.0, k=1.0).to_dict()["params"] == {"k": 1.0}

    class SavesItself(Absorbs):
        def __init__(self, name, timestep, gain=2.0):
            super().__init__(name, timestep)
            self.gain = gain

        def to_dict(self):
            return {**super().to_dict(), "gain": self.gain}

    assert SavesItself("s", DT, gain=5.0).to_dict()["gain"] == 5.0

    original = _keeps(gain=5.0)
    for duplicate in (copy.deepcopy(original), pickle.loads(pickle.dumps(BounceNode("b", DT)))):
        duplicate.to_dict()                                 # no record: nothing is said
    with pytest.warns(UserWarning, match=WARNING):
        original.to_dict()                                  # the original still is

    later = _keeps()
    later.gain = 5.0                                        # after construction
    later.to_dict()


def test_the_base_constructors_own_arguments_are_not_reported():
    """``name`` and ``timestep`` are in the entry beside ``params``;
    ``geometry_source`` is taken out of ``params`` by the base constructor
    and is saved by nothing (a statement of what is, pinned so that a
    change to it is a decision)."""
    node = BounceNode("b", DT, geometry_source="/World/ball", restitution=0.8)
    entry = node.to_dict()
    assert entry == {"type": "BounceNode", "name": "b", "timestep": DT,
                     "params": {"restitution": 0.8}}
    assert node.geometry_source == "/World/ball"
    base = list(inspect.signature(SimulationNode.__init__).parameters)[1:3]
    assert set(base) | {"geometry_source"} == node_module._BASE_CONSTRUCTOR_NAMES


@pytest.mark.parametrize("case", sorted(BUILT_IN_CASES))
def test_no_built_in_node_is_reported(case):
    """Every built-in node, built with every value off its default (the
    round-trip table's cases), saves all it was given."""
    node = _built_in(case)
    node.to_dict()                                          # warnings are errors here
    assert node_module._unsaved_constructor_arguments(node) == []


def test_a_built_in_argument_saved_only_when_given_is_not_reported():
    """``LBMNode`` writes no ``wall_mask`` key for a node built without
    one: a parameter whose name is not in ``params``, and no loss.  A rule
    that went by the names alone would warn of every wall-free node."""
    node = LBMNode("lbm", 1.0, grid_shape=(6, 4), lattice="D2Q9", viscosity=0.1)
    assert "wall_mask" not in node.to_dict()["params"]
    walls = np.zeros((6, 4), bool)
    walls[:, 0] = True
    walled = LBMNode("lbm", 1.0, grid_shape=(6, 4), lattice="D2Q9", viscosity=0.1,
                     wall_mask=walls)
    assert "wall_mask" in walled.to_dict()["params"]


# -- where it is said, and through what --------------------------------------

def test_a_wrapper_passes_on_what_its_node_cannot_save():
    """``HybridNode`` saves the node it wraps (as the sharded wrappers do):
    the wrapper's own arguments are not asked about, the wrapped node's
    are, and the warning names the wrapped node."""
    saved = HybridNode(ParamsSpring(stiffness=40.0, name="spring", timestep=DT),
                       lambda s, b, dt: {}).to_dict()
    assert saved["params"]["stiffness"] == 40.0
    hybrid = HybridNode(AttributeSpring(stiffness=40.0, name="spring", timestep=DT),
                        lambda s, b, dt: {})
    with pytest.warns(UserWarning, match=r"node 'spring' \(AttributeSpring\).*stiffness"):
        hybrid.to_dict()


def test_the_warning_points_at_the_line_that_saved():
    """Attributed to the caller's line, through the graph and through a
    wrapper, not to a line of this package."""
    gm = _graph(AttributeSpring(stiffness=40.0, name="spring", timestep=DT))
    hybrid = HybridNode(AttributeSpring(stiffness=40.0, name="spring", timestep=DT),
                        lambda s, b, dt: {})
    for save in (gm.to_dict, gm.get_node("spring").to_dict, hybrid.to_dict):
        with pytest.warns(UserWarning, match=WARNING) as said:
            save()
        assert said[0].filename == __file__


def test_it_is_a_warning_and_the_node_is_still_saved():
    """An error would stop every node written the old way from being saved
    at all; the entry is written, less the argument."""
    node = AttributeSpring(stiffness=40.0, rest_height=3.0, k=1.0, name="spring", timestep=DT)
    with warnings.catch_warnings(record=True) as said:
        warnings.simplefilter("always")
        entry = node.to_dict()
    assert [w.category for w in said] == [UserWarning]
    assert "arguments rest_height, stiffness" in str(said[0].message)
    assert entry == {"type": "AttributeSpring", "name": "spring", "timestep": DT,
                     "params": {"k": 1.0}}


# -- the record costs a node nothing ------------------------------------------

def test_noting_the_arguments_changes_nothing_else_about_a_node():
    """The names are kept off the node: its attributes, its pickle, its
    copy and its class's signature are what they were, and the record goes
    when the node does."""
    node = AttributeSpring(stiffness=40.0, name="spring", timestep=DT)
    assert set(vars(node)) == {"name", "delta_t", "geometry_source", "params",
                               "stiffness", "rest_height"}
    assert vars(pickle.loads(pickle.dumps(node))).keys() == vars(node).keys()
    assert vars(copy.deepcopy(node)).keys() == vars(node).keys()

    assert str(inspect.signature(AttributeSpring)) == "(stiffness=12.0, rest_height=2.0, **kwargs)"
    # A class that inherits the base constructor shows that constructor,
    # not the (*args, **kwargs) of the hook that notes the arguments.
    assert str(inspect.signature(BounceNode)) == "(name: str, timestep: float, **params)"
    assert str(inspect.signature(SimulationNode)) == "(name: str, timestep: float, **params)"

    key = id(node)
    assert node_module._CONSTRUCTED_WITH[key][1] == frozenset({"stiffness"})
    del node
    assert key not in node_module._CONSTRUCTED_WITH


def test_a_constructor_call_that_fails_is_the_constructors_error():
    """The hook binds the arguments before ``__init__`` runs; a call the
    constructor refuses must fail as it always did, and an abstract class
    is still not instantiable."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'stifness'"):
        _Relay("r", DT, gain=1.0, stifness=2.0)
    with pytest.raises(TypeError, match="missing .* required positional argument"):
        BounceNode("only a name")
    with pytest.raises(TypeError, match="abstract"):
        SimulationNode("n", DT)

    class WithItsOwnNew(BounceNode):
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)                     # hands nothing on

    assert WithItsOwnNew("w", DT, restitution=0.5).to_dict()["params"] == {"restitution": 0.5}


# -- arrays in a config ---------------------------------------------------------

class _Relay(SimulationNode):
    """A node whose constant is an array, as a gain matrix is."""

    def __init__(self, name, timestep, gain):
        super().__init__(name, timestep, gain=gain)

    def initial_state(self):
        return {"x": jnp.ones(2, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": jnp.tanh(jnp.asarray(p["gain"]) @ state["x"])}


@pytest.mark.parametrize("array", [np.asarray, jnp.asarray], ids=["numpy", "jax"])
def test_an_array_in_params_is_saved_as_the_array_and_json_needs_a_default(array):
    """What ``GraphManager.to_dict``'s docstring says since 0.4.0: an array
    is not converted, plain ``json.dumps`` of such a config raises, and
    written as lists it reloads into a graph that steps bit for bit."""
    def graph():
        gm = GraphManager()
        gm.add_node(_Relay("r", DT, array([[0.5, -0.25], [0.125, 0.75]], np.float32)))
        gm.compile()
        return gm

    def run(gm):
        return np.asarray(gm.run_scan(3)["r"]["x"])

    config = graph().to_dict()
    assert not isinstance(config["nodes"][0]["params"]["gain"], list)
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(config)
    text = json.dumps(config, default=lambda a: np.asarray(a).tolist())
    want = run(graph())
    for reloaded in (GraphManager.from_dict(config, {"_Relay": _Relay}),
                     GraphManager.from_dict(json.loads(text), {"_Relay": _Relay})):
        reloaded.compile()
        assert np.array_equal(run(reloaded), want)
