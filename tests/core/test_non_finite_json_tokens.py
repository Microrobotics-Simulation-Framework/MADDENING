"""`to_dict` writes a non-finite number as a non-standard JSON token.

Pins the config half of ``MADD-ANO-006``.  ``json.dumps`` defaults to
``allow_nan=True``, which emits the bare tokens ``NaN``, ``Infinity`` and
``-Infinity``.  Python reads them back, so every MADDENING round trip is
exact; they are not JSON, so a conforming reader rejects the document.

The anomaly is **open**: closing it means choosing an encoding for a
format that has shipped, which is a maintainer's decision rather than a
fix.  Until then this test states the behaviour the registry documents,
so that changing it fails here and the registry entry is updated with it.
"""

import json
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.nodes.spring import SpringDamperNode
from maddening.serialization.config import to_dict


def _graph_with_an_unbounded_spec():
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0))
    gm.set_param_spec("s", "stiffness",
                      ParamSpec(bounds=(-math.inf, math.inf)))
    return gm


def test_an_infinite_param_spec_bound_makes_to_dict_non_standard_json():
    text = json.dumps(to_dict(_graph_with_an_unbounded_spec()))

    assert "-Infinity" in text, (
        "MADD-ANO-006 says to_dict emits the bare token; if that changed, "
        "update the registry entry"
    )
    # Python reads its own output back exactly...
    assert json.loads(text) is not None
    # ...and a conforming reader does not.
    with pytest.raises(ValueError):
        json.loads(text, parse_constant=_refuse)


def test_a_finite_graph_serialises_as_strict_json():
    """The anomaly is confined to non-finite numbers, not to `to_dict`."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("s", 0.01, stiffness=30.0))
    text = json.dumps(to_dict(gm))

    assert json.loads(text, parse_constant=_refuse) is not None


def _refuse(token):
    """A strict reader: RFC 8259 has no literal for these."""
    raise ValueError(f"non-standard JSON token {token!r}")
