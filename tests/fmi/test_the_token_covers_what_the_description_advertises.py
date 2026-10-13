"""The instantiation token covers every variable's start values, bounds and
unit.

FMI 3.0: the token "is used to check that the modelDescription.xml file is
compatible with the C code of the FMU"; here the bridge's description is
that code.  The token hashed only names, types, causalities and shapes, so
an FMU packaged from a graph with stiffness 30 instantiated, ``fmi3OK``,
against a bridge serving the same structure at 45.  The bridge started the
instance at *its* 45, as its own description says, while the importer's
``modelDescription.xml`` said 30.  Now the start values, ``min`` / ``max``,
variability, value references and clocks are hashed too, and the C wrapper
refuses the mismatched FMU at instantiation.

And the unit.  The bridge holds its description to the graph's unit -- it
refuses to start once a parameter's unit has been re-declared -- so the unit
is advertised as the bounds are, yet the token left it out: an FMU packaged
when stiffness was declared in N/m instantiated against a bridge serving
the description that says kN/m.  Every attribute of a variable is now
either hashed or named, with the reason, as not covered.
"""

from __future__ import annotations

import dataclasses

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


# ------------------------------------------------------------------- the unit

def _respec(gm, **changes):
    """``spring.stiffness``'s ParamSpec with ``changes`` (a re-declaration
    dirties nothing: specs are metadata to the step)."""
    old = gm.param_specs()["nodes"]["spring"]["stiffness"]
    gm.set_param_spec("spring", "stiffness", dataclasses.replace(old, **changes))
    return gm


def _unit(md, name):
    return next(v.unit for v in md.variables if v.name == name)


def test_another_unit_gives_another_token():
    """The audit's case: stiffness re-declared from N/m to kN/m."""
    packaged = build_model_description(_spring(), model_name="m")
    served = build_model_description(_respec(_spring(), units="kN/m"), model_name="m")
    assert (_unit(packaged, "spring.params.stiffness"),
            _unit(served, "spring.params.stiffness")) == ("N/m", "kN/m")
    assert packaged.instantiation_token != served.instantiation_token
    # a unit removed, or given to a variable that had none, too
    bare = build_model_description(_respec(_spring(), units=""), model_name="m")
    assert len({packaged.instantiation_token, served.instantiation_token,
                bare.instantiation_token}) == 3
    # and the same unit is the same token (the hash is of the value)
    again = build_model_description(_respec(_spring(), units="N/m"), model_name="m")
    assert again.instantiation_token == packaged.instantiation_token


def test_a_description_text_is_not_part_of_the_token():
    """What the token leaves out, stated: the free text an importer shows a
    user.  Nothing computes with it and the bridge does not check it."""
    base = build_model_description(_spring(), model_name="m")
    retold = build_model_description(_respec(_spring(), description="how stiff it is"),
                                     model_name="m")
    described = {v.name: v.description for v in retold.variables}
    assert described["spring.params.stiffness"] == "how stiff it is"
    assert retold.instantiation_token == base.instantiation_token


def _a_variable():
    from maddening.fmi.model_description import FMIVariable

    return FMIVariable(name="n.x", value_reference=7, dtype="float32", causality="input",
                       variability="continuous", description="a thing", unit="m", node="n",
                       field="x", shape=(3,), start="0.0 0.0 0.0", min=-1.0, max=1.0,
                       clocks=(2,), interval_decimal=None)


#: Another value for each attribute of a variable (one its type allows).
_OTHER = {"name": "n.y", "value_reference": 8, "dtype": "float64", "causality": "parameter",
          "variability": "tunable", "description": "another thing", "unit": "mm", "node": "m",
          "field": "y", "shape": (4,), "start": "1.0 0.0 0.0", "min": -2.0, "max": 2.0,
          "clocks": (3,), "interval_decimal": 0.5}


def test_every_attribute_of_a_variable_is_hashed_or_named_as_not_covered():
    """Field by field over the dataclass, so a field added to
    ``FMIVariable`` fails here until it is put under the token or in the
    list of what the token leaves out, with a reason."""
    from maddening.fmi.model_description import FMIVariable, _NOT_IN_THE_TOKEN, _token_part

    fields = {f.name for f in dataclasses.fields(FMIVariable)}
    assert fields == set(_OTHER), fields ^ set(_OTHER)
    assert set(_NOT_IN_THE_TOKEN) <= fields
    assert all(isinstance(why, str) and len(why) > 20 for why in _NOT_IN_THE_TOKEN.values())
    base = _a_variable()
    for name in sorted(fields):
        if name == "interval_decimal":
            # a clock's interval: judged on a clock, which carries one
            clock = FMIVariable(name="clock_0", value_reference=2, dtype="clock",
                                causality="input", variability="discrete", interval_decimal=0.01)
            changed = _token_part(dataclasses.replace(clock, interval_decimal=0.02)) != \
                _token_part(clock)
        else:
            changed = _token_part(dataclasses.replace(base, **{name: _OTHER[name]})) != \
                _token_part(base)
        assert changed is (name not in _NOT_IN_THE_TOKEN), name
    # what the bridge checks against its description is all under the token:
    # start values, min / max and unit of a parameter, and the step
    assert not {"start", "min", "max", "unit"} & set(_NOT_IN_THE_TOKEN)


def test_what_the_xml_writes_for_a_variable_is_under_the_token():
    """The other direction: every attribute ``to_xml`` writes on a variable
    element, except its description text, moves the token."""
    from xml.etree import ElementTree

    md = build_model_description(_respec(_spring(), bounds=(1.0, 99.0), transform=None),
                                 model_name="m", multi_clock=True)
    written = set()
    for el in ElementTree.fromstring(md.to_xml()).find("ModelVariables"):
        written |= set(el.attrib)
    assert written == {"name", "valueReference", "causality", "variability", "description",
                       "unit", "start", "min", "max", "clocks", "intervalVariability",
                       "intervalDecimal"}, written


@pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")
def test_an_fmu_packaged_before_a_unit_was_redeclared_does_not_instantiate(tmp_path, monkeypatch):
    """End to end: the FMU's ``modelDescription.xml`` says N/m, the bridge
    serves the description that says kN/m, and
    ``fmi3InstantiateCoSimulation`` fails on the token."""
    from maddening.fmi.package import build_fmu_binary
    from tests.fmi.test_c_wrapper_refuses_what_it_used_to_coerce import _Wrapper
    from tests.fmi.test_bridge_master_dt_is_the_graph_step import _sidecar
    from maddening.fmi.tcp_bridge import FmuTcpBridge

    gm = _spring()
    packaged = build_model_description(gm, model_name="m")
    _respec(gm, units="kN/m")
    served = build_model_description(gm, model_name="m")
    wrapper = _Wrapper(build_fmu_binary(tmp_path / "bin"))
    with FmuTcpBridge(_sidecar(gm, served), served, master_dt=gm.timestep) as bridge:
        monkeypatch.setenv("MADDENING_FMU_ENDPOINT", bridge.endpoint)
        assert not wrapper.instantiate(packaged.instantiation_token)
        assert "instantiation token does not match" in wrapper.logs[-1]
        inst = wrapper.instantiate(served.instantiation_token)
        assert inst
        wrapper.lib.fmi3FreeInstance(inst)
