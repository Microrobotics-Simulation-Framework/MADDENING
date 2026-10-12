"""A field has one external-input declaration: a second that differs is
refused where it is made, and one that says the same again changes nothing.

``add_external_input`` appended every call to a list, and the readers of a
declaration did not take the same one.  Declared ``int32`` and then
``float32``, the step and the FMU export ran ``float32`` (the last), the
cast warning said the value ran cast to ``int32`` (the first), and
``to_dict`` saved both.  The second declaration is now a ``ValueError`` that
names the field, the shape and dtype of each, and what to do, at every door
that declares one: the method, ``from_dict`` of a config that holds two
(``tests/usd/test_usd_params.py`` holds the stage), and the node replacement,
which declares a node's inputs again after removing them
(``tests/surrogates/test_replace.py``).  No REST route declares one.

A second declaration with the same shape and dtype is accepted and kept
once.  Nothing a reader could disagree about has changed, and a builder
that declares its inputs and is run twice worked before 0.4.0 (both were
kept there, and the last decided the zeros of an omitted input).
"""

from __future__ import annotations

import json
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.nodes.spring import SpringDamperNode

DT = 0.01
HEAD = "external input 'spring.anchor_position' is already declared"
REGISTRY = {"SpringDamperNode": SpringDamperNode}

#: Pairs of declarations of one field that differ, as ``(shape, dtype)``:
#: each dtype family against another in both orders, two widths of one
#: family, two shapes, and a shape with its axes the other way round.
DIFFERENT = {
    "int32_then_float32": (((), jnp.int32), ((), jnp.float32)),
    "float32_then_int32": (((), jnp.float32), ((), jnp.int32)),
    "float32_then_bool": (((), jnp.float32), ((), jnp.bool_)),
    "bool_then_uint8": (((), jnp.bool_), ((), jnp.uint8)),
    "int8_then_int32": (((), jnp.int8), ((), jnp.int32)),
    "float32_then_float64": (((), jnp.float32), ((), jnp.float64)),
    "scalar_then_vector": (((), jnp.float32), ((3,), jnp.float32)),
    "vector_then_scalar": (((3,), jnp.float32), ((), jnp.float32)),
    "three_then_four": (((3,), jnp.float32), ((4,), jnp.float32)),
    "axes_swapped": (((2, 3), jnp.float32), ((3, 2), jnp.float32)),
    "shape_and_dtype": (((3,), jnp.int32), ((), jnp.float32)),
}

#: Two spellings of one declaration.
SAME = {
    "the_defaults_twice": ({}, {}),
    "the_default_and_its_values": ({}, {"shape": (), "dtype": jnp.float32}),
    "a_dtype_and_its_name": ({"dtype": jnp.int32}, {"dtype": "int32"}),
    "a_dtype_and_numpys": ({"dtype": jnp.float32}, {"dtype": np.dtype("float32")}),
    "a_tuple_and_a_list": ({"shape": (3,)}, {"shape": [3]}),
    "bool_and_bool_": ({"dtype": bool}, {"dtype": jnp.bool_}),
}


def _spring():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", DT, stiffness=30.0, rest_length=0.4))
    return gm


def _declared(gm):
    return [(tuple(e.shape), np.dtype(e.dtype).name) for e in gm._external_inputs]


def _run(gm, value=0.7, steps=3):
    """The spring's position after ``steps`` steps fed ``value``, the cast
    warnings, the type the FMU gives the input and what ``to_dict`` saves."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
        for _ in range(steps):
            gm.step({"spring": {"anchor_position": value}})
        md = build_model_description(gm, model_name="X")
    return {
        "position": np.asarray(gm.get_node_state("spring")["position"]).tobytes(),
        "warned": [str(w.message).split(":")[0] for w in caught
                   if "is declared" in str(w.message)],
        "fmu": [(v.name, v.dtype) for v in md.variables if v.causality == "input"],
        "saved": [(e["shape"], e["dtype"]) for e in gm.to_dict()["external_inputs"]],
    }


# ------------------------------------------------------------- the method


@pytest.mark.parametrize("pair", sorted(DIFFERENT))
def test_a_second_declaration_that_differs_is_refused_and_the_first_is_kept(pair):
    (shape_a, dtype_a), (shape_b, dtype_b) = DIFFERENT[pair]
    gm = _spring()
    gm.add_external_input("spring", "anchor_position", shape=shape_a, dtype=dtype_a)
    with pytest.raises(ValueError) as refused:
        gm.add_external_input("spring", "anchor_position", shape=shape_b, dtype=dtype_b)
    text = str(refused.value)
    assert text.startswith(HEAD)
    # each of the two, by its shape and dtype, in the order they were made
    was = f"with shape {shape_a!r} and dtype {np.dtype(dtype_a).name}"
    now = f"declares it again with shape {shape_b!r} and dtype {np.dtype(dtype_b).name}"
    assert was in text and now in text and text.index(was) < text.index(now)
    assert _declared(gm) == [(shape_a, np.dtype(dtype_a).name)]


def test_the_message_says_what_to_do_and_what_the_graph_used_to_do():
    gm = _spring()
    gm.add_external_input("spring", "anchor_position", dtype=jnp.int32)
    with pytest.raises(ValueError) as refused:
        gm.add_external_input("spring", "anchor_position", dtype=jnp.float32)
    text = str(refused.value)
    assert "add_external_input('spring', 'anchor_position', ...)" in text
    assert "A field has one declaration" in text
    assert "Declare the input once" in text
    assert "there is no remove_external_input" in text and "remove_node" in text
    # which of the two an upgraded script was running
    assert "Until 0.4.0 the second declaration was taken and the graph ran the last one" in text
    assert "Declaring it again with the same shape and dtype is accepted" in text
    assert not hasattr(GraphManager, "remove_external_input")


@pytest.mark.parametrize("order", ["int32_then_float32", "float32_then_int32"])
def test_the_graph_runs_reports_exports_and_saves_its_one_declaration(order):
    """What the two declarations disagreed about, on the graph the refusal
    leaves: the step, the cast warning, the FMU's type and the saved
    config all speak of the first, as a graph that declares it alone."""
    (_, first), (_, second) = DIFFERENT[order]
    gm = _spring()
    gm.add_external_input("spring", "anchor_position", dtype=first)
    with pytest.raises(ValueError, match="already declared"):
        gm.add_external_input("spring", "anchor_position", dtype=second)
    alone = _spring()
    alone.add_external_input("spring", "anchor_position", dtype=first)
    got, want = _run(gm), _run(alone)
    assert got == want
    name = np.dtype(first).name
    assert got["fmu"] == [("spring.anchor_position", name)]
    assert got["saved"] == [([], name)]
    # 0.7 is not an int32: the warning, said once, names the dtype it ran in
    told = ["external input 'spring.anchor_position' is declared int32 and was handed a "
            "float32 value that int32 does not hold as given"] if name == "int32" else []
    assert got["warned"] == told
    # and the two dtypes do give two results, so "the first" is a statement
    other = _spring()
    other.add_external_input("spring", "anchor_position", dtype=second)
    assert _run(other)["position"] != got["position"]


def test_a_refusal_on_a_compiled_graph_leaves_it_compiled():
    gm = _spring()
    gm.add_external_input("spring", "anchor_position", dtype=jnp.int32)
    gm.compile()
    generation = gm._compile_generation
    with pytest.raises(ValueError, match="already declared"):
        gm.add_external_input("spring", "anchor_position", shape=(3,))
    assert gm._dirty is False
    gm.step({"spring": {"anchor_position": 1}})
    assert gm._compile_generation == generation


@pytest.mark.parametrize("pair", sorted(SAME))
def test_the_same_declaration_again_is_accepted_and_kept_once(pair):
    first, again = SAME[pair]
    gm = _spring()
    gm.add_external_input("spring", "anchor_position", **first)
    gm.add_external_input("spring", "anchor_position", **again)
    alone = _spring()
    alone.add_external_input("spring", "anchor_position", **first)
    assert _declared(gm) == _declared(alone) and len(_declared(gm)) == 1
    assert gm.to_dict()["external_inputs"] == alone.to_dict()["external_inputs"]
    if tuple(first.get("shape", ())) == ():
        assert _run(gm, value=1.0) == _run(alone, value=1.0)


def test_the_same_declaration_again_does_not_mark_a_compiled_graph_for_a_recompile():
    gm = _spring()
    gm.add_external_input("spring", "anchor_position", dtype=jnp.int32)
    gm.compile()
    generation = gm._compile_generation
    gm.add_external_input("spring", "anchor_position", dtype="int32")
    assert gm._dirty is False
    gm.step({"spring": {"anchor_position": 1}})
    assert gm._compile_generation == generation


def test_float64_and_float32_are_two_declarations_without_x64_too():
    """The comparison is of what was written down, which ``to_dict`` saves
    and the FMU types the input by, not of what this process canonicalises
    it to."""
    gm = _spring()
    gm.add_external_input("spring", "anchor_position", dtype=jnp.float64)
    with pytest.raises(ValueError, match="dtype float64.*again with shape \\(\\) and dtype float32"):
        gm.add_external_input("spring", "anchor_position", dtype=jnp.float32)
    gm.add_external_input("spring", "anchor_position", dtype="float64")
    assert _declared(gm) == [((), "float64")]


def test_another_field_and_the_same_field_of_another_node_are_other_declarations():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", DT, stiffness=30.0))
    gm.add_node(SpringDamperNode("other", DT, stiffness=30.0))
    gm.add_external_input("spring", "anchor_position", dtype=jnp.int32)
    gm.add_external_input("other", "anchor_position", dtype=jnp.float32)
    gm.add_external_input("spring", "force", shape=(3,))
    assert _declared(gm) == [((), "int32"), ((), "float32"), ((3,), "float32")]


def test_removing_the_node_is_the_way_to_declare_its_input_otherwise():
    """What the message offers: ``remove_node`` drops the node's
    declarations, and the node added again takes another one."""
    gm = _spring()
    gm.add_external_input("spring", "anchor_position", dtype=jnp.int32)
    with pytest.raises(ValueError, match="remove_node"):
        gm.add_external_input("spring", "anchor_position", dtype=jnp.float32)
    gm.remove_node("spring")
    gm.add_node(SpringDamperNode("spring", DT, stiffness=30.0, rest_length=0.4))
    gm.add_external_input("spring", "anchor_position", dtype=jnp.float32)
    alone = _spring()
    alone.add_external_input("spring", "anchor_position", dtype=jnp.float32)
    assert _run(gm) == _run(alone)


# --------------------------------------------------------------- from_dict


def _config(*entries):
    """The config of a spring with one declared input, its declarations
    replaced by ``entries``."""
    gm = _spring()
    gm.add_external_input("spring", "anchor_position")
    config = json.loads(json.dumps(gm.to_dict()))
    config["external_inputs"] = [
        {"target_node": "spring", "target_field": "anchor_position", **entry}
        for entry in entries]
    return config


@pytest.mark.parametrize("pair", sorted(DIFFERENT))
def test_a_config_that_holds_two_declarations_that_differ_is_refused_by_its_entries(pair):
    (shape_a, dtype_a), (shape_b, dtype_b) = DIFFERENT[pair]
    config = _config({"shape": list(shape_a), "dtype": np.dtype(dtype_a).name},
                     {"shape": list(shape_b), "dtype": np.dtype(dtype_b).name})
    with pytest.raises(ValueError) as refused:
        GraphManager.from_dict(config, REGISTRY)
    text = str(refused.value)
    assert text.startswith(HEAD)
    assert f"with shape {shape_a!r} and dtype {np.dtype(dtype_a).name} " \
           "(external_inputs[0] of this config)" in text
    assert f"external_inputs[1] declares it again with shape {shape_b!r} and dtype " \
           f"{np.dtype(dtype_b).name}" in text
    assert 'Delete from the config\'s "external_inputs" the entry' in text


def test_a_config_saved_with_both_by_a_graph_that_took_the_second_is_refused():
    """The file an earlier commit of 0.4.0 wrote for the graph of the
    report: the two entries, in the order they were declared.  It named
    neither as the one that ran; loading it must not pick."""
    saved = _config({"shape": [], "dtype": "int32"}, {"shape": [], "dtype": "float32"})
    with pytest.raises(ValueError, match="spring.anchor_position.*external_inputs\\[1\\]"):
        GraphManager.from_dict(saved, REGISTRY)
    # the entry its author keeps loads, either one
    for keep in (0, 1):
        one = dict(saved, external_inputs=[saved["external_inputs"][keep]])
        assert _declared(GraphManager.from_dict(one, REGISTRY)) == [
            ((), saved["external_inputs"][keep]["dtype"])]


def test_a_config_from_before_the_dtype_was_saved_is_held_to_the_rule():
    """An entry without ``dtype`` is ``float32``, as it has always loaded:
    two such entries of two shapes differ, and one beside an ``int32``
    entry does."""
    with pytest.raises(ValueError, match="shape \\(\\) and dtype float32.*shape \\(3,\\) and "
                                         "dtype float32"):
        GraphManager.from_dict(_config({"shape": []}, {"shape": [3]}), REGISTRY)
    with pytest.raises(ValueError, match="dtype float32.*dtype int32"):
        GraphManager.from_dict(_config({"shape": []}, {"shape": [], "dtype": "int32"}),
                               REGISTRY)


@pytest.mark.parametrize("entries", [
    ({"shape": []}, {"shape": []}),
    ({"shape": [], "dtype": "int32"}, {"shape": [], "dtype": "int32"}),
    ({"shape": []}, {"shape": [], "dtype": "float32"}),
    ({"shape": [3], "dtype": "bool"}, {"shape": [3], "dtype": "bool"}, {"shape": [3], "dtype": "bool"}),
], ids=["no_dtype_twice", "int32_twice", "no_dtype_and_float32", "bool_three_times"])
def test_a_config_that_holds_one_declaration_several_times_loads_it_once(entries):
    """A graph that declared its input twice alike saved two entries before
    0.4.0, and such a file still loads: as the one declaration it is."""
    gm = GraphManager.from_dict(_config(*entries), REGISTRY)
    alone = GraphManager.from_dict(_config(entries[0]), REGISTRY)
    assert _declared(gm) == _declared(alone) and len(_declared(gm)) == 1
    assert gm.to_dict()["external_inputs"] == alone.to_dict()["external_inputs"]


def test_a_third_entry_is_compared_with_the_first_of_its_field():
    with pytest.raises(ValueError, match="external_inputs\\[0\\] of this config.*"
                                         "external_inputs\\[2\\] declares it again"):
        GraphManager.from_dict(_config({"shape": []}, {"shape": []}, {"shape": [2]}), REGISTRY)


def test_a_saved_graph_loads_with_the_one_declaration_it_has():
    gm = _spring()
    gm.add_external_input("spring", "anchor_position", dtype=jnp.int32)
    gm.add_external_input("spring", "force", shape=(3,))
    back = GraphManager.from_dict(json.loads(json.dumps(gm.to_dict())), REGISTRY)
    assert _declared(back) == _declared(gm) == [((), "int32"), ((3,), "float32")]
