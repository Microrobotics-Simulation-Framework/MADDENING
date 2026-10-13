"""An external input survives a config round trip with its dtype.

``to_dict`` wrote ``target_node``, ``target_field`` and ``shape`` and
nothing about ``dtype``, so an input declared ``jnp.int32`` came back
from its own config as ``float32``: a node using it as an index failed
on the reloaded graph, and one doing arithmetic with it silently got a
different trace.  Serialisation is now driven off
:class:`~maddening.core.graph_manager.ExternalInputSpec` itself, and the
field-coverage test below is what stops the next field added to that
dataclass from being dropped the same way.
"""

from __future__ import annotations

import json
from dataclasses import fields

import jax.numpy as jnp
import pytest

from maddening.core.graph_manager import ExternalInputSpec, GraphManager
from maddening.nodes.ball import BallNode

REGISTRY = {"BallNode": BallNode}


def _graph(dtype):
    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0))
    gm.add_external_input("ball", "mode", shape=(3,), dtype=dtype)
    return gm


def test_every_field_of_the_spec_has_a_slot_in_the_config():
    """The guard: a new field must be written, not silently defaulted."""
    written = set(ExternalInputSpec("n", "f", (), jnp.int32).to_dict())
    assert written == {f.name for f in fields(ExternalInputSpec)}


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32, jnp.bool_])
def test_a_config_round_trip_keeps_the_declared_dtype(dtype):
    gm = _graph(dtype)
    config = json.loads(json.dumps(gm.to_dict()))
    back = GraphManager.from_dict(config, REGISTRY)

    assert jnp.dtype(back._external_inputs[0].dtype) == jnp.dtype(dtype)
    # And the zeros the graph fills an omitted input with follow it.
    leaf = back._default_external_inputs()["ball"]["mode"]
    assert leaf.dtype == jnp.dtype(dtype)
    assert leaf.shape == (3,)


def test_a_config_written_before_dtype_was_recorded_still_loads():
    """Forward compatibility runs one way; backward has to keep working.

    Such a config described a graph that ran at ``float32``, which is
    what it reloads as.
    """
    config = _graph(jnp.float32).to_dict()
    for ei in config["external_inputs"]:
        del ei["dtype"]
    back = GraphManager.from_dict(config, REGISTRY)
    assert jnp.dtype(back._external_inputs[0].dtype) == jnp.dtype(jnp.float32)


def test_the_spec_round_trips_on_its_own():
    spec = ExternalInputSpec("ball", "mode", (2, 3), jnp.int32)
    back = ExternalInputSpec.from_dict(spec.to_dict())
    assert back.target_node == spec.target_node
    assert back.target_field == spec.target_field
    assert back.shape == spec.shape
    assert jnp.dtype(back.dtype) == jnp.dtype(spec.dtype)
