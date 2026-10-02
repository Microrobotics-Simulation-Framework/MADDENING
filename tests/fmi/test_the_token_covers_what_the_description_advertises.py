"""The instantiation token covers every variable's start values and bounds.

FMI 3.0: the token "is used to check that the modelDescription.xml file is
compatible with the C code of the FMU"; here the bridge's description is
that code.  The token hashed only names, types, causalities and shapes, so
an FMU packaged from a graph with stiffness 30 instantiated, ``fmi3OK``,
against a bridge serving the same structure at 45.  The bridge started the
instance at *its* 45, as its own description says, while the importer's
``modelDescription.xml`` said 30.  Now the start values, ``min`` / ``max``,
variability, value references and clocks are hashed too, and the C wrapper
refuses the mismatched FMU at instantiation.
"""

from __future__ import annotations

import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.package import find_c_compiler
from maddening.nodes.spring import SpringDamperNode
from tests.fmi.test_start_values_have_the_lexical_form_of_their_type import _graph as _gate


def _spring(**kw):
    gm = GraphManager()
    gm.add_node(SpringDamperNode(name="spring", timestep=0.01, **{
        "stiffness": 30.0, "damping": 0.5, "initial_position": 0.5, "rest_length": 0.0, **kw}))
    gm.compile()
    return gm


def _token(gm, **kw):
    return build_model_description(gm, model_name="m", **kw).instantiation_token


def test_the_same_graph_gives_the_same_token():
    assert _token(_spring()) == _token(_spring())


@pytest.mark.parametrize("change", [{"stiffness": 45.0}, {"damping": 0.75},
                                    {"rest_length": 0.1}])
def test_another_parameter_start_gives_another_token(change):
    assert _token(_spring()) != _token(_spring(**change))


def test_other_bounds_give_another_token(monkeypatch):
    from maddening.core.params import ParamSpec

    base = _token(_gate())
    original = type(_gate()._nodes["gate"].node).param_specs

    def wider(self):
        return {**original(self), "rate": ParamSpec(bounds=(0.0, 50.0))}

    monkeypatch.setattr(type(_gate()._nodes["gate"].node), "param_specs", wider)
    assert _token(_gate()) != base


def test_clocks_and_variability_are_part_of_the_token():
    gm = _spring()
    assert _token(gm) != _token(gm, multi_clock=True)


@pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")
def test_an_fmu_with_other_start_values_does_not_instantiate_against_the_bridge(
        tmp_path, monkeypatch):
    from maddening.fmi.package import build_fmu_binary
    from tests.fmi.test_c_wrapper_refuses_what_it_used_to_coerce import _Wrapper
    from tests.fmi.test_bridge_master_dt_is_the_graph_step import _sidecar
    from maddening.fmi.tcp_bridge import FmuTcpBridge

    old = build_model_description(_spring(), model_name="m")
    gm = _spring(stiffness=45.0)
    new = build_model_description(gm, model_name="m")
    wrapper = _Wrapper(build_fmu_binary(tmp_path / "bin"))
    with FmuTcpBridge(_sidecar(gm, new), new, master_dt=gm.timestep) as bridge:
        monkeypatch.setenv("MADDENING_FMU_ENDPOINT", bridge.endpoint)
        assert not wrapper.instantiate(old.instantiation_token)
        assert "instantiation token does not match" in wrapper.logs[-1]
        inst = wrapper.instantiate(new.instantiation_token)
        assert inst
        wrapper.lib.fmi3FreeInstance(inst)
