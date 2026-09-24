"""The FMI ``min`` / ``max`` attributes are the *settable* envelope: for a
strict (log / logit) bound the description advertises the next
representable float inside, for an inclusive one the bound itself.

Originally written from the independent audit of 2026-09-16 (round 1; report and
reproducers under ``benchmarks/results/audit1/``).
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec
from maddening.fmi.model_description import build_model_description
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.nodes.ball import BallNode
from maddening.nodes.spring import SpringDamperNode
from maddening.nodes.table import TableNode

F32 = np.finfo(np.float32)


@pytest.fixture
def gm():
    g = GraphManager()
    # A table edge, so the ball's step reads ``elasticity``: the FMU exports
    # only parameters its step reads, and without a table ``elasticity`` is
    # a knob that does nothing.
    g.add_node(TableNode(name="table", timestep=1e-2))
    g.add_node(BallNode(name="ball", timestep=1e-2, initial_position=1.0, elasticity=0.7))
    g.add_node(SpringDamperNode(name="s", timestep=1e-2, stiffness=30.0, damping=2.0))
    g.add_edge("table", "ball", "position", "table_position")
    g.compile()
    return g


def _sidecar(gm, md):
    return FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
    ))


def test_log_leaf_with_zero_bound_advertises_smallest_normal(gm):
    md = build_model_description(gm, model_name="m", include_evolving=True)
    var = next(v for v in md.variables if v.name == "s.params.stiffness")
    # not 0.0 (rejected as "below bound"), not nextafter(0) (a subnormal)
    assert var.min == float(F32.tiny) and var.min > 0.0
    assert var.max is None
    _sidecar(gm, md).set_params({"s.params.stiffness": var.min})    # accepted
    el = next(v for v in ET.fromstring(md.to_xml()).find("ModelVariables")
              if v.get("name") == "s.params.stiffness")
    assert float(el.get("min")) == var.min


def test_log_leaf_with_nonzero_bound_advertises_next_float(gm):
    gm.set_param_spec("s", "stiffness", ParamSpec(bounds=(8.0, None), transform="log"))
    md = build_model_description(gm, model_name="m", include_evolving=True)
    var = next(v for v in md.variables if v.name == "s.params.stiffness")
    assert var.min == float(np.nextafter(np.float32(8.0), np.float32(np.inf)))
    assert var.min > 8.0
    sc = _sidecar(gm, md)
    sc.set_params({"s.params.stiffness": var.min})
    with pytest.raises(ValueError, match="below bound"):
        sc.set_params({"s.params.stiffness": 8.0})


def test_logit_leaf_advertises_both_open_bounds(gm):
    gm.set_param_spec("ball", "elasticity", ParamSpec(bounds=(0.0, 1.0), transform="logit"))
    md = build_model_description(gm, model_name="m")
    var = next(v for v in md.variables if v.name == "ball.params.elasticity")
    assert var.min == float(F32.tiny)
    assert var.max == float(np.nextafter(np.float32(1.0), np.float32(-np.inf)))
    assert 0.0 < var.min < var.max < 1.0
    sc = _sidecar(gm, md)
    sc.set_params({"ball.params.elasticity": var.min})
    sc.set_params({"ball.params.elasticity": var.max})
    with pytest.raises(ValueError, match="above bound"):
        sc.set_params({"ball.params.elasticity": 1.0})


def test_inclusive_identity_bounds_are_unchanged(gm):
    md = build_model_description(gm, model_name="m")
    var = next(v for v in md.variables if v.name == "ball.params.elasticity")
    assert (var.min, var.max) == (0.0, 1.0)
    damping = next(v for v in md.variables if v.name == "s.params.damping")
    assert (damping.min, damping.max) == (0.0, None)
    sc = _sidecar(gm, md)
    sc.set_params({"ball.params.elasticity": 0.0, "s.params.damping": 0.0})
