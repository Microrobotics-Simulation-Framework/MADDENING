"""Calibrated params and ParamSpec overrides round-trip through USD."""

import json

import jax.numpy as jnp
import numpy as np
import pytest
from pxr import Usd

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.usd.serialization import load_graph_from_usd, save_graph_to_usd


def _gm():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0, mass=1.5))
    gm.add_node(SpringDamperNode("t", 0.01, stiffness=10.0, damping=1.0))
    gm.add_edge("s", "t", "position", "anchor_position")
    gm.compile()
    return gm


def test_calibrated_params_and_overrides_round_trip():
    gm = _gm()
    gm.params["nodes"]["s"]["stiffness"] = jnp.asarray(45.5, jnp.float32)
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)

    prim = stage.GetPrimAtPath("/Simulation/nodes/s")
    stored = json.loads(prim.GetAttribute("maddening:paramsJson").Get())
    assert stored["stiffness"] == 45.5            # live value, not constructor's
    assert stored["damping"] == 2.0
    overrides = json.loads(prim.GetAttribute("maddening:paramSpecOverridesJson").Get())
    assert overrides == {"mass": ParamSpec(trainable=False).to_dict()}
    # a node without overrides carries no override attribute
    t_attr = stage.GetPrimAtPath("/Simulation/nodes/t").GetAttribute(
        "maddening:paramSpecOverridesJson")
    assert not t_attr or not t_attr.Get()

    gm2 = load_graph_from_usd(stage)
    gm2.compile()
    assert float(gm2.params["nodes"]["s"]["stiffness"]) == 45.5
    assert gm2.param_specs()["nodes"]["s"]["mass"].trainable is False
    assert gm2.param_specs()["nodes"]["t"]["mass"].trainable is True
    a, b = gm.run_scan(20), gm2.run_scan(20)
    np.testing.assert_allclose(np.asarray(a["t"]["position"]),
                               np.asarray(b["t"]["position"]), rtol=1e-6)


def test_stale_override_on_load_warns_instead_of_failing():
    """A ParamSpec override for a parameter the node class no longer has
    must not make the whole stage unloadable."""
    import warnings

    gm = _gm()
    gm.set_param_spec("s", "mass", ParamSpec(trainable=False))
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    prim = stage.GetPrimAtPath("/Simulation/nodes/s")
    prim.GetAttribute("maddening:paramSpecOverridesJson").Set(
        json.dumps({"renamed_away": ParamSpec(trainable=False).to_dict(),
                    "mass": ParamSpec(trainable=False).to_dict()}))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        gm2 = load_graph_from_usd(stage)
    assert any("renamed_away" in str(x.message) for x in w)
    gm2.compile()
    assert gm2.param_specs()["nodes"]["s"]["mass"].trainable is False


# --------------------------------------------------------------------------
# What a stage can and cannot represent.  Both from the independent audit of
# 2026-09-19 (``params-io``; reproducers ``r20_misc.py``, ``r2_usd_params.py``).
# --------------------------------------------------------------------------

def test_a_param_the_encoder_cannot_represent_raises_instead_of_being_repred():
    """``json.dumps(..., default=str)`` used to write a param's ``repr``.

    The documented ``static_data_provider`` pattern stores an object in
    ``self.params`` so it survives a checkpoint round trip; through USD it
    came back as the string ``"<Provider ...>"``, while the config path
    raised ``TypeError`` for the same graph.  The two serialisers now
    agree, on the answer that fails at save time.
    """
    class _Provider:
        def __repr__(self):
            return "<Provider /data/mesh.vtu>"

    gm = GraphManager()
    node = SpringDamperNode("s", 0.01, stiffness=30.0)
    node.params["provider"] = _Provider()
    gm.add_node(node)

    stage = Usd.Stage.CreateInMemory()
    with pytest.raises(TypeError) as exc:
        save_graph_to_usd(gm, stage)
    assert "provider" in str(exc.value) and "'s'" in str(exc.value)


class _ArrayParamNode(SpringDamperNode):
    """A spring that also carries an array constant in ``params``."""

    def __init__(self, name, timestep, *, empty_2d=None, **kwargs):
        super().__init__(name, timestep, **kwargs)
        self.params["empty_2d"] = np.asarray(
            np.zeros((0, 3)) if empty_2d is None else empty_2d)


_ARRAY_NODE_QUALNAME = f"{_ArrayParamNode.__module__}.{_ArrayParamNode.__qualname__}"


def test_a_zero_size_array_param_keeps_its_shape_through_a_round_trip():
    """``np.zeros((0, 3)).tolist()`` is ``[]``: every axis after a
    zero-length one is lost unless the shape is recorded beside it."""
    gm = GraphManager()
    gm.add_node(_ArrayParamNode("s", 0.01, stiffness=30.0))

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    reloaded = load_graph_from_usd(
        stage, node_registry={_ARRAY_NODE_QUALNAME: _ArrayParamNode})

    assert np.asarray(reloaded.get_node("s").params["empty_2d"]).shape == (0, 3)


def test_a_graph_with_no_degenerate_shapes_writes_no_extra_attribute():
    gm = _gm()
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    attr = stage.GetPrimAtPath("/Simulation/nodes/s").GetAttribute(
        "maddening:paramArrayShapesJson")
    assert not attr or not attr.Get()


def test_an_external_inputs_declared_dtype_survives_a_round_trip():
    """The stage carried a shape but no dtype, so an ``int32`` or ``bool``
    boundary input came back ``float32``."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0))
    gm.add_external_input("s", "anchor_position", shape=(), dtype=jnp.int32)

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    reloaded = load_graph_from_usd(stage)

    assert np.dtype(reloaded._external_inputs[0].dtype) == np.dtype("int32")


def test_a_stage_without_a_dtype_attribute_loads_as_it_always_did():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0))
    gm.add_external_input("s", "anchor_position", shape=())

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    prim = stage.GetPrimAtPath("/Simulation/external_inputs/ext0")
    prim.GetAttribute("maddening:dtype").Clear()          # a pre-0.4.0 stage

    reloaded = load_graph_from_usd(stage)
    assert np.dtype(reloaded._external_inputs[0].dtype) == np.dtype("float32")


def test_an_unknown_dtype_on_a_stage_warns_and_falls_back():
    import warnings

    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0))
    gm.add_external_input("s", "anchor_position", shape=())

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    prim = stage.GetPrimAtPath("/Simulation/external_inputs/ext0")
    prim.GetAttribute("maddening:dtype").Set("object")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reloaded = load_graph_from_usd(stage)
    assert any("object" in str(w.message) for w in caught)
    assert np.dtype(reloaded._external_inputs[0].dtype) == np.dtype("float32")


def _refuse(token):
    """A strict reader: RFC 8259 has no literal for these."""
    raise ValueError(f"non-standard JSON token {token!r}")


def _strict_loads(text):
    return json.loads(text, parse_constant=_refuse)


class _NonFiniteArrayNode(SpringDamperNode):
    """A spring carrying a diverged array and an empty one in ``params``."""

    def __init__(self, name, timestep, *, diverged=None, empty_2d=None, **kwargs):
        super().__init__(name, timestep, **kwargs)
        self.params["diverged"] = np.asarray(
            [np.inf, 1.0, np.nan] if diverged is None else diverged)
        self.params["empty_2d"] = np.asarray(
            np.zeros((0, 3)) if empty_2d is None else empty_2d)


_NON_FINITE_NODE_QUALNAME = (
    f"{_NonFiniteArrayNode.__module__}.{_NonFiniteArrayNode.__qualname__}")


def test_a_non_finite_param_writes_a_stage_a_strict_reader_accepts():
    """The USD third of ``MADD-ANO-006``.

    ``maddening:paramsJson`` used to be written with ``json.dumps`` at its
    default ``allow_nan=True``, so a diverged param went onto the stage as
    the bare token ``Infinity``.  Python read it back, which is why
    nothing here noticed; every other reader of the ``.usda`` rejected the
    attribute.  It is now the quoted token, which is JSON.
    """
    gm = GraphManager()
    node = SpringDamperNode("s", 0.01, stiffness=30.0)
    node.params["cap"] = float("inf")
    node.params["floor"] = float("-inf")
    node.params["bad"] = float("nan")
    gm.add_node(node)

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    stored = stage.GetPrimAtPath("/Simulation/nodes/s").GetAttribute(
        "maddening:paramsJson").Get()

    parsed = _strict_loads(stored)
    assert parsed["cap"] == "Infinity"
    assert parsed["floor"] == "-Infinity"
    assert parsed["bad"] == "NaN"


def test_a_non_finite_array_param_round_trips_beside_a_zero_size_one():
    """The two awkward shapes together, through a real save/load.

    A zero-size array carries no value to encode and its shape is
    recorded in a separate attribute; the non-finite entries beside it
    must still be found by the walk and come back as the same floats.
    """
    gm = GraphManager()
    gm.add_node(_NonFiniteArrayNode("s", 0.01, stiffness=30.0))

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)

    stored = stage.GetPrimAtPath("/Simulation/nodes/s").GetAttribute(
        "maddening:paramsJson").Get()
    assert _strict_loads(stored)["diverged"] == ["Infinity", 1.0, "NaN"]

    reloaded = load_graph_from_usd(
        stage, node_registry={_NON_FINITE_NODE_QUALNAME: _NonFiniteArrayNode})
    params = reloaded.get_node("s").params
    back = np.asarray(params["diverged"])
    assert np.isposinf(back[0]) and back[1] == 1.0 and np.isnan(back[2])
    assert np.asarray(params["empty_2d"]).shape == (0, 3)


def test_an_infinite_param_spec_bound_is_written_as_a_quoted_token():
    """The override attribute is JSON too, and carries ParamSpec bounds."""
    gm = _gm()
    gm.set_param_spec("s", "mass", ParamSpec(bounds=(-float("inf"), float("inf"))))

    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    stored = stage.GetPrimAtPath("/Simulation/nodes/s").GetAttribute(
        "maddening:paramSpecOverridesJson").Get()
    assert _strict_loads(stored)["mass"]["bounds"] == ["-Infinity", "Infinity"]

    reloaded = load_graph_from_usd(stage)
    reloaded.compile()
    assert reloaded.param_specs()["nodes"]["s"]["mass"].bounds == (
        -float("inf"), float("inf"))


def test_a_stage_written_before_040_still_loads_its_bare_tokens():
    """Backward compatibility, pinned on a literal attribute value.

    ``json.loads`` parses the bare tokens itself, so the loader needs no
    special case -- but it must not acquire one that rejects them.
    """
    gm = _gm()
    gm.set_param_spec("s", "mass", ParamSpec(trainable=True))
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(gm, stage)
    prim = stage.GetPrimAtPath("/Simulation/nodes/s")
    prim.GetAttribute("maddening:paramsJson").Set(
        '{"stiffness": Infinity, "damping": NaN, "mass": 1.0, '
        '"rest_length": 1.0, "initial_position": -Infinity, '
        '"initial_velocity": 0.0}')
    prim.GetAttribute("maddening:paramSpecOverridesJson").Set(
        '{"mass": {"trainable": true, "bounds": [-Infinity, Infinity], '
        '"transform": null, "description": "", "units": ""}}')

    reloaded = load_graph_from_usd(stage)
    params = reloaded.get_node("s").params
    assert float(params["stiffness"]) == float("inf")
    assert np.isnan(float(params["damping"]))
    assert float(params["initial_position"]) == -float("inf")
    reloaded.compile()
    assert reloaded.param_specs()["nodes"]["s"]["mass"].bounds == (
        -float("inf"), float("inf"))


def test_a_finite_graph_writes_strict_json_to_the_stage():
    """The fix is confined to non-finite numbers, not to the attribute."""
    stage = Usd.Stage.CreateInMemory()
    save_graph_to_usd(_gm(), stage)
    stored = stage.GetPrimAtPath("/Simulation/nodes/s").GetAttribute(
        "maddening:paramsJson").Get()

    assert _strict_loads(stored)["stiffness"] == 30.0


# ---------------------------------------- node names agree with the config surface

@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_a_stage_carrying_a_token_spelled_node_name_is_refused_on_load(tmp_path, token):
    """The stage surface and the config surface agree about node names.

    A node name goes to a typed USD ``String`` attribute, which never
    meets the JSON codec, while ``GraphManager.to_dict`` puts it in the
    JSON tree and refuses it (``MADD-ANO-010``).  A ``.usda`` written
    before the refusal therefore reloaded into a graph that could not be
    written as a config -- the same graph accepted or refused depending
    on the surface.  ``load_graph_from_usd`` builds through ``add_node``,
    so the refusal now lands on the load, naming the node.
    """
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0, damping=2.0))
    gm.compile()
    written = tmp_path / "written.usda"
    stage = Usd.Stage.CreateNew(str(written))
    save_graph_to_usd(gm, stage)
    stage.Save()

    # a stage as an older MADDENING would have written it
    text = (written.read_text()
            .replace('string maddening:nodeName = "s"',
                     f'string maddening:nodeName = "{token}"'))
    assert f'maddening:nodeName = "{token}"' in text
    legacy = tmp_path / "legacy.usda"          # a new path: USD caches layers
    legacy.write_text(text)

    with pytest.raises(ValueError, match="non-finite JSON token"):
        load_graph_from_usd(Usd.Stage.Open(str(legacy)))


def test_an_ordinary_node_name_still_round_trips_through_the_stage(tmp_path):
    """The refusal is exact: a name that merely resembles a token is fine."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("nan", 0.01, stiffness=30.0, damping=2.0))
    gm.compile()
    path = tmp_path / "lookalike.usda"
    stage = Usd.Stage.CreateNew(str(path))
    save_graph_to_usd(gm, stage)
    stage.Save()

    assert sorted(load_graph_from_usd(Usd.Stage.Open(str(path)))._nodes) == ["nan"]
