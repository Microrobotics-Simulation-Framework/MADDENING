"""``to_dict()`` says which constructor arguments a saved node cannot rebuild.

A saved node is its ``params`` and nothing else: every reload calls
``cls(name=..., timestep=..., **params)``.  A constructor argument a class
keeps as an attribute (``self.stiffness = stiffness``, the spelling the
quickstart taught until 0.4.0) is dropped by the save, and the reloaded
graph runs the constructor's default with nothing said: the quickstart's
own coupled graph built with ``SpringNode(stiffness=40.0)`` came back as
the ``k = 12`` graph, in every release since 0.1.0.

``SimulationNode.to_dict`` now warns.  The rule goes by the name: the node
was *constructed with* an argument for a named parameter, at a value that
is not exactly the parameter's default, and ``node.params`` has no key of
that name.  The tests hold the rule as it is stated: every such node is
named; a node built with its defaults, and every built-in node with every
argument off its default, is not (every test here runs with warnings as
errors, so an unexpected one fails it); what the rule cannot tell is
silent; and a node that saves an argument another way is named too, until
its class says so by overriding ``to_dict``.
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
from maddening import nodes as built_in_nodes
from maddening.nodes import (
    BallNode,
    HealthCheckNode,
    HeartPumpNode,
    HeatNode,
    LBMNode,
    LBMPipeNode,
    RigidBody2DNode,
    RigidBodyNode,
    SpringDamperNode,
    TableNode,
)
from maddening.nodes import adaptive as adaptive_nodes
from maddening.nodes.adaptive import WaveletAdaptiveNode

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
        "node 'spring' (AttributeSpring) does not save the constructor argument stiffness.  "
        "AttributeSpring's constructor (or one it inherits) does not hand it to "
        "SimulationNode.__init__ under that name, so it is not in node.params, and a saved "
        "node is rebuilt as AttributeSpring(name=..., timestep=..., **params): a graph loaded "
        "from this save runs with the default stiffness=12.0.  Hand it on -- "
        "super().__init__(name, timestep, stiffness=stiffness) -- and read it from "
        "self.params; or, if the node saves it another way, override to_dict() to say so.")

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
        "node 'g' (Grid) does not save the constructor arguments origin, shape, spacing.  "
        "Grid's constructor (or one it inherits) does not hand them to "
        "SimulationNode.__init__ under those names, so they are not in node.params, and a "
        "saved node is rebuilt as Grid(name=..., timestep=..., **params): a graph loaded from "
        "this save runs with the default origin=(0.0, 0.0), spacing=1.0 and cannot build the "
        "node (shape: no default).  Hand them on -- super().__init__(name, timestep, "
        "origin=origin, shape=shape, spacing=spacing) -- and read them from self.params; or, "
        "if the node saves them another way, override to_dict() to say so.")


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
    {"gain": float("2.0")},                             # an equal float that is another object
    {"taps": (0.0, 0.0)},
    {"taps": tuple(float(text) for text in ("0", "0"))},    # an equal tuple, another object
    {"mask": None},
], ids=["defaults", "the same float", "an equal float", "the same tuple", "an equal tuple",
        "None"])
def test_an_argument_at_exactly_its_default_is_not_reported(given):
    """The reload passes the default, and the node was built with a value
    that cannot be told from it -- the same object, or another of the same
    type and bits (a literal repeated in one module is one object, so the
    equal ones are built at run time): nothing is lost, nothing is said."""
    entry = _keeps(**given).to_dict()                   # warnings are errors here
    assert entry["params"] == {}


@pytest.mark.parametrize("given, named", [
    ({"gain": 2.5}, "argument gain."),
    ({"gain": 2}, "argument gain."),                    # an int where the default is a float
    ({"gain": np.float32(2.0)}, "argument gain."),      # a dtype the default does not carry
    ({"gain": True}, "argument gain."),
    ({"gain": float("nan")}, "argument gain."),
    ({"taps": (0.0, 1.0)}, "argument taps."),
    ({"taps": (0.0, -0.0)}, "argument taps."),          # the sign of a zero
    ({"taps": (0, 0)}, "argument taps."),               # ints in the tuple
    ({"taps": [0.0, 0.0]}, "argument taps."),           # a list where the default is a tuple
    ({"taps": np.zeros(2)}, "argument taps."),          # an array of the same numbers
    ({"taps": (0.0, 0.0, 0.0)}, "argument taps."),      # another length
    ({"mask": np.zeros(3, bool)}, "argument mask."),    # an array where the default is None
    ({"gain": 2.5, "mask": [True]}, "arguments gain, mask."),
], ids=lambda value: repr(value)[:40])
def test_an_argument_that_is_not_exactly_its_default_and_outside_params_is_reported(given, named):
    """"The default" is the same object, or a value of the same type that
    is the same bit for bit: a reload passes the default itself, and only
    what cannot be told from it is known to build the same node."""
    with pytest.warns(UserWarning, match=re.escape(f"{WARNING}{named[8:]}")):
        _keeps(**given).to_dict()


def _reloaded_states(node, steps=3):
    """``(states of the node's graph, states of the graph reloaded from its
    config or the error the reload ran into)``, the save's warnings set
    aside."""
    def run(gm):
        gm.compile()
        return [np.asarray(gm.step()[node.name]["x"]) for _ in range(steps)]

    gm = GraphManager()
    gm.add_node(node)
    want = run(gm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        config = gm.to_dict()
    try:
        return want, run(GraphManager.from_dict(config, {type(node).__name__: type(node)}))
    except Exception as error:   # noqa: BLE001 - the reload's own failure is the result
        return want, error


def test_a_zero_of_the_other_sign_and_an_int_for_a_float_are_not_the_default():
    """Two arguments that used to pass for the default, and what the
    reload did with nothing said: a node that reads the sign of its
    constant came back with the opposite one, and one that multiplies an
    integer state by its constant came back unable to step."""

    class Signed(SimulationNode):
        def __init__(self, name, timestep, floor=0.0):
            super().__init__(name, timestep)
            self.floor = floor

        def initial_state(self):
            return {"x": jnp.asarray(1.5, jnp.float32)}

        def update(self, state, boundary_inputs, dt):
            return {"x": jnp.copysign(state["x"], jnp.float32(self.floor))}

    class Counter(SimulationNode):
        def __init__(self, name, timestep, factor=2.0):
            super().__init__(name, timestep)
            self.factor = factor

        def initial_state(self):
            return {"x": jnp.asarray(3, jnp.int32)}

        def update(self, state, boundary_inputs, dt):
            return {"x": state["x"] * self.factor}

    with pytest.warns(UserWarning, match=f"{WARNING} floor"):
        Signed("n", DT, floor=-0.0).to_dict()
    want, got = _reloaded_states(Signed("n", DT, floor=-0.0))
    assert float(want[-1]) == -1.5 and float(got[-1]) == 1.5       # the loss it warns of

    with pytest.warns(UserWarning, match=f"{WARNING} factor"):
        Counter("n", DT, factor=2).to_dict()
    want, got = _reloaded_states(Counter("n", DT, factor=2))
    assert int(want[-1]) == 24 and isinstance(got, ValueError)     # the reload cannot step


class _KeepsGain(SimulationNode):
    """Keeps ``gain`` as an attribute and decays by it."""

    def __init__(self, name, timestep, gain=2.0):
        super().__init__(name, timestep)
        self.gain = gain

    def initial_state(self):
        return {"x": jnp.asarray(1.5, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] * (1.0 - dt * self.gain)}


def test_an_argument_a_parent_keeps_is_lost_through_the_childs_kwargs_and_nothing_says_so():
    """What the rule does not read: a subclass that hands ``**kwargs`` up
    to a parent that keeps one of them.  The argument is lost by the save
    and no warning is given; a class that inherits the parent's constructor
    as it is, is asked.  Pinned as what is, with the loss shown, so that
    the guide's sentence about it stays true and closing it is a decision
    (a parent may save such an argument under another name, as the
    built-in adaptive node does with ``blindness_threshold``)."""

    class Child(_KeepsGain):
        def __init__(self, name, timestep, offset=0.0, **kwargs):
            super().__init__(name, timestep, **kwargs)
            self.params["offset"] = offset

    class Inherits(_KeepsGain):
        """No constructor of its own: the parent's is its constructor."""

    assert Child("n", DT, gain=5.0, offset=0.1).to_dict()["params"] == {"offset": 0.1}
    want, got = _reloaded_states(Child("n", DT, gain=5.0, offset=0.1))
    assert not np.array_equal(want[-1], got[-1])                # lost, in silence
    with pytest.warns(UserWarning, match=rf"\(Inherits\) {WARNING} gain\."):
        Inherits("n", DT, gain=5.0).to_dict()


def test_a_node_that_saves_an_argument_another_way_says_so_by_overriding_to_dict():
    """The rule goes by the name, so it also names a node that reloads bit
    for bit: an argument saved under another spelling, one handed on under
    another name, a flag that says how to read another argument.  Such a
    class answers for its own saving by overriding ``to_dict`` -- calling
    the base one is enough -- and is then not asked."""

    class Oscillator(SimulationNode):
        """Takes the constant under either of two spellings and saves one."""

        def __init__(self, name, timestep, frequency=1.0, period=None):
            if period is not None:
                frequency = 1.0 / period
            super().__init__(name, timestep, frequency=frequency)

        def initial_state(self):
            return {"x": jnp.asarray(1.5, jnp.float32)}

        def update(self, state, boundary_inputs, dt):
            return {"x": state["x"] * (1.0 - dt * self.params["frequency"])}

    class Normalises(SimulationNode):
        """A flag that says how to read another argument; what is saved is
        the rate as it is read."""

        def __init__(self, name, timestep, rate=1.0, per_minute=False):
            super().__init__(name, timestep, rate=rate / 60.0 if per_minute else rate)

        def initial_state(self):
            return {"x": jnp.asarray(1.5, jnp.float32)}

        def update(self, state, boundary_inputs, dt):
            return {"x": state["x"] * (1.0 - dt * self.params["rate"])}

    class SaysSo(Oscillator):
        def to_dict(self):
            return super().to_dict()        # ``period`` is saved, as ``frequency``

    for node in (Oscillator("n", DT, period=0.25), Normalises("n", DT, rate=120.0,
                                                               per_minute=True)):
        want, got = _reloaded_states(node)
        assert all(np.array_equal(a, b) for a, b in zip(want, got))   # nothing is lost ...
        with pytest.warns(UserWarning, match=WARNING):                # ... and it is named
            node.to_dict()

    entry = SaysSo("n", DT, period=0.25).to_dict()                    # warnings are errors here
    assert entry["params"] == {"frequency": 4.0}
    want, got = _reloaded_states(SaysSo("n", DT, period=0.25))
    assert all(np.array_equal(a, b) for a, b in zip(want, got))


def test_what_the_rule_cannot_tell_is_not_reported():
    """Stated in ``_unsaved_constructor_arguments``: an argument a
    ``**kwargs`` took in that no parent names, what an ``*args`` took in,
    a class that saves itself its own way, a copy, a class whose own
    ``__new__`` hands no arguments on, and an attribute written after
    construction.  Each of these loses the argument in silence: pinned as
    what is, so that closing one is a decision."""

    class StarArgs(_KeepsGain):
        def __init__(self, name, timestep, *gains):
            super().__init__(name, timestep)
            self.gain = gains[0] if gains else 2.0

    assert StarArgs("n", DT, 5.0).to_dict()["params"] == {}

    class OwnNew(_KeepsGain):
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)                     # hands nothing on

    assert OwnNew("n", DT, gain=5.0).to_dict()["params"] == {}

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


def _channel_walls():
    walls = np.zeros((10, 6), bool)
    walls[:, 0] = walls[:, -1] = True
    return walls


#: ``class -> (timestep, kwargs)``: every built-in node class with EVERY
#: named parameter of its constructor off its default, the wavelet node's
#: inherited ones (which reach ``AdaptiveNode`` through its ``**kw``)
#: included.  ``test_the_table_moves_every_argument_of_every_built_in_node``
#: holds the table to that sentence.
EVERY_ARGUMENT_OFF_ITS_DEFAULT = {
    BallNode: (0.01, dict(initial_position=1.3, initial_velocity=0.2, elasticity=0.6,
                          gravity=-3.0)),
    TableNode: (0.01, dict(position=0.4)),
    SpringDamperNode: (0.01, dict(stiffness=50.0, damping=0.3, mass=2.0, rest_length=0.5,
                                  initial_position=0.9, initial_velocity=0.1)),
    HeartPumpNode: (0.01, dict(resistance=1.2, compliance=0.9, heart_rate=60.0,
                               stroke_volume=60.0, venous_pressure=2.0, systole_fraction=0.3,
                               initial_pressure=70.0)),
    HealthCheckNode: (0.01, dict(checks={"x": {"finite": True, "min": -1.0, "max": 2.0}})),
    HeatNode: (0.01, dict(n_cells=8, length=2.0, thermal_diffusivity=0.001, stencil_order=4,
                          grid_points=np.linspace(0.0, 1.0, 8) ** 2,
                          initial_temperature=np.linspace(0.0, 1.0, 8).tolist())),
    LBMNode: (1.0, dict(grid_shape=(10, 6), lattice="D2Q9", viscosity=0.05,
                        wall_mask=_channel_walls(), inlet_face="y_min", outlet_face="y_max")),
    LBMPipeNode: (1.0, dict(nx=8, ny=6, nz=6, tau=0.7, pipe_radius=0.8, propeller_x=3,
                            propeller_radius=0.7, propeller_strength=0.001,
                            initial_velocity=0.01, gravity=1e-5, fill_fraction=0.6,
                            tau_tracer=0.7, G=-5.0, rho_liquid=1.2, rho_gas=0.3, rho_0=1.1,
                            rho_wall=0.5, initial_rho_liquid=1.15, initial_rho_gas=0.35)),
    RigidBodyNode: (0.01, dict(
        mass=2.0, inertia=np.array([1.0, 2.0, 3.0]), gravity=np.array([0.0, -1.0, -9.81]),
        constraints={"z": 0.0}, initial_position=np.array([0.1, 0.2, 0.3]),
        initial_velocity=(1.0, 0.0, 0.0), initial_orientation=(0.0, 1.0, 0.0, 0.0),
        initial_angular_velocity=(0.0, 0.1, 0.0))),
    RigidBody2DNode: (0.01, dict(mass=2.0, inertia=0.5, gravity=np.array([0.0, -2.0]),
                                 initial_x=0.3, initial_y=0.2, initial_vx=0.3, initial_vy=0.1,
                                 initial_angle=0.1, initial_omega=0.2)),
    WaveletAdaptiveNode: (0.5, dict(
        dim=2, n_levels=3, n_coarse=3, order=2, k=64, theta=0.3, sigma=0.2, mass=2.0,
        sensor=(0.25, 0.5), preconditioner="level", boundary="dirichlet", frozen_solver="cg",
        gradient_capture_threshold=0.5, blindness_threshold=0.25, blindness_break_delta=0.125,
        D_threshold=12, blindness_gate=False, on_blind="raise", dtype="float32")),
}


def _built_in(cls):
    timestep, kwargs = EVERY_ARGUMENT_OFF_ITS_DEFAULT[cls]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)    # RigidBody2DNode
        return cls("n", timestep, **kwargs)


def _named_parameters(cls):
    """``{name: parameter}`` for every named constructor parameter a node
    of ``cls`` can be given: its own, and those its ``**kwargs`` hands to a
    parent, less what the base constructor takes and ``n_max``, which the
    wavelet node derives and sets for its parent."""
    named = {}
    for klass in cls.__mro__:
        if klass is SimulationNode:
            break
        if "__init__" not in vars(klass):
            continue
        parameters = list(inspect.signature(klass.__init__).parameters.values())[1:]
        for parameter in parameters:
            if parameter.kind in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY) \
                    and parameter.name not in node_module._BASE_CONSTRUCTOR_NAMES:
                named.setdefault(parameter.name, parameter)
        if not any(p.kind is p.VAR_KEYWORD for p in parameters):
            break
    named.pop("n_max", None)
    return named


#: Saved under another name: ``AdaptiveNode`` takes it as a deprecated
#: alias and saves it as ``gradient_capture_threshold``.
SAVED_UNDER_ANOTHER_NAME = {"blindness_threshold"}


def test_the_table_moves_every_argument_of_every_built_in_node():
    """The table is what its comment says: every concrete node class the
    two packages export has a case, and the case gives every named
    parameter -- inherited ones included -- a value that is not exactly
    its default, by the rule's own test of "the default"."""
    exported = {getattr(module, name) for module in (built_in_nodes, adaptive_nodes)
                for name in dir(module) if name.endswith("Node")}
    concrete = {cls for cls in exported if not inspect.isabstract(cls)}
    assert concrete == set(EVERY_ARGUMENT_OFF_ITS_DEFAULT)
    for cls, (_, kwargs) in EVERY_ARGUMENT_OFF_ITS_DEFAULT.items():
        named = _named_parameters(cls)
        assert set(kwargs) == set(named), cls.__name__
        at_default = [name for name, value in kwargs.items()
                      if named[name].default is not inspect.Parameter.empty
                      and node_module._is_the_default(value, named[name].default)]
        assert not at_default, (cls.__name__, at_default)
        # ... and the node's own record agrees, for the parameters its own
        # constructor names (what a **kwargs takes in is not noted).
        own = {p for p in inspect.signature(cls.__init__).parameters if p in named}
        assert node_module._CONSTRUCTED_WITH[id(node := _built_in(cls))][1] == own, node.name


@pytest.mark.parametrize("cls", list(EVERY_ARGUMENT_OFF_ITS_DEFAULT), ids=lambda c: c.__name__)
def test_no_built_in_node_is_reported(cls):
    """Every built-in node class, built with every argument off its
    default, saves all it was given."""
    node = _built_in(cls)
    entry = node.to_dict()                                  # warnings are errors here
    assert node_module._unsaved_constructor_arguments(node) == []
    assert set(_named_parameters(cls)) - SAVED_UNDER_ANOTHER_NAME <= set(entry["params"])


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
