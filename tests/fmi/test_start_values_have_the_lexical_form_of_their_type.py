"""A variable's ``start``, ``min`` and ``max`` are written in the lexical form
of its FMI type, and a Boolean or integer input or output is ``discrete``.

``build_model_description`` wrote every input's start as ``"0.0"`` and every
parameter's through ``repr(float(x))``, whatever the variable's type.  FMI
3.0's schema types a ``<Boolean>``'s ``start`` as ``xs:boolean`` and an
``<Int32>``'s as ``xs:int``, so ``"0.0"`` is valid for neither: FMPy's
``validate_fmu`` reported it, and ``simulate_fmu``, which validates by
default, refused the FMU.  Once the literals were valid, FMPy's
model-structure check refused the next thing: a Boolean or integer variable
may not be ``continuous``.  The same graph is checked here against FMPy's
validator, against the FMI 3.0 XSD FMPy ships (when ``lxml`` is there), and
end to end through ``simulate_fmu`` with its default ``validate=True``.

The ``unit`` attribute was the next thing of the same kind.  The writer set
it on every element, and FMI 3.0's schema has it on ``<Float32>`` and
``<Float64>`` only: an ``int32`` input into a field whose node declares
``expected_units`` gave ``Element 'Int32', attribute 'unit': The attribute
'unit' is not allowed``.  The last section holds a unit-bearing input of
every type the writer knows, and unit-bearing integer and Boolean
parameters, to the same validators.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zipfile

import jax
import jax.numpy as jnp
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.params import ParamSpec
from maddening.fmi import build_model_description
from maddening.fmi.model_description import (
    _DTYPE_TO_FMI_TYPE,
    FMIVariable,
    ModelDescription,
    _fmi_kind,
    _parse_xs_value,
    _xs_value,
)
from maddening.nodes import BallNode, SpringDamperNode
from maddening.fmi.package import MODEL_IDENTIFIER, build_fmu_binary, find_c_compiler, write_fmu
from maddening.fmi.sidecar import FmuSidecar, SidecarConfig
from maddening.fmi.tcp_bridge import FmuTcpBridge, values_of

DT = 0.01


@stability(StabilityLevel.STABLE)
class TypedGate(SimulationNode):
    """A Boolean and an Int32 input, a Boolean and an Int32 parameter beside
    a float one, and an output of each type."""

    def __init__(self, name, timestep, rate=1.0, enabled=True, gain=3):
        super().__init__(name, timestep, rate=rate, enabled=enabled, gain=gain)

    def params_pytree(self):
        # Ints and bools are structural by default; this node makes them
        # tunable leaves on purpose, so they become FMI parameters.
        return {"rate": jnp.asarray(self.params["rate"], jnp.float32),
                "enabled": jnp.asarray(self.params["enabled"]),
                "gain": jnp.asarray(self.params["gain"], jnp.int32)}

    def param_specs(self):
        return {"rate": ParamSpec(bounds=(0.0, 5.0)),
                "gain": ParamSpec(bounds=(-2.5, 10.5)),
                # bounds on a Boolean, which the XML must not carry over
                "enabled": ParamSpec(bounds=(0.0, 1.0))}

    def initial_state(self):
        return {"level": jnp.asarray(0.0, jnp.float32), "was_open": jnp.asarray(False),
                "count": jnp.asarray(0, jnp.int32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        is_open = boundary_inputs.get("open", jnp.asarray(False))
        n = boundary_inputs.get("n", jnp.asarray(0, jnp.int32))
        on = jnp.logical_and(is_open, p["enabled"])
        return {"level": state["level"] + jnp.where(on, p["rate"] * dt, 0.0),
                "was_open": jnp.asarray(is_open, bool),
                "count": state["count"] + p["gain"] * n}


def _graph(**kw):
    gm = GraphManager()
    gm.add_node(TypedGate("gate", DT, **kw))
    gm.add_external_input("gate", "open", dtype=jnp.bool_)
    gm.add_external_input("gate", "n", dtype=jnp.int32)
    gm.compile()
    return gm


@pytest.fixture(scope="module")
def gm():
    return _graph()


@pytest.fixture(scope="module")
def md(gm):
    return build_model_description(gm, model_name="G", model_identifier=MODEL_IDENTIFIER)


def _elements(md):
    root = ET.fromstring(md.to_xml())
    return {el.get("name"): el for el in root.find("ModelVariables")}


def test_every_start_min_and_max_is_a_literal_of_the_variables_type(md):
    el = _elements(md)
    assert el["gate.open"].tag == "Boolean" and el["gate.open"].get("start") == "false"
    assert el["gate.n"].tag == "Int32" and el["gate.n"].get("start") == "0"
    assert el["gate.params.enabled"].tag == "Boolean"
    assert el["gate.params.enabled"].get("start") == "true"
    assert el["gate.params.gain"].tag == "Int32" and el["gate.params.gain"].get("start") == "3"
    assert el["gate.params.rate"].tag == "Float32" and el["gate.params.rate"].get("start") == "1.0"
    # an integer's bounds are the integers inside the declared interval
    assert (el["gate.params.gain"].get("min"), el["gate.params.gain"].get("max")) == ("-2", "10")
    assert (el["gate.params.rate"].get("min"), el["gate.params.rate"].get("max")) == ("0.0", "5.0")
    # a Boolean has no min / max in the schema
    assert el["gate.params.enabled"].get("min") is None
    assert el["gate.params.enabled"].get("max") is None


def test_a_boolean_or_integer_input_or_output_is_discrete(md):
    el = _elements(md)
    for name in ("gate.open", "gate.n", "gate.was_open", "gate.count"):
        assert el[name].get("variability") == "discrete", name
    assert el["gate.level"].get("variability") == "continuous"


def test_fmpy_validates_the_description(md, tmp_path):
    pytest.importorskip("fmpy")
    from fmpy.model_description import read_model_description
    from fmpy.validation import validate_fmu

    fmu = tmp_path / "g.fmu"
    with zipfile.ZipFile(fmu, "w") as zf:
        zf.writestr("modelDescription.xml", md.to_xml())
    assert validate_fmu(str(fmu)) == []
    read_model_description(str(fmu), validate=True)


def test_the_fmi3_schema_validates_the_description(md):
    fmpy = pytest.importorskip("fmpy")
    etree = pytest.importorskip("lxml.etree")
    xsd = os.path.join(os.path.dirname(fmpy.__file__), "schema", "fmi3",
                       "fmi3ModelDescription.xsd")
    if not os.path.exists(xsd):
        pytest.skip(f"FMPy {fmpy.__version__} ships no FMI 3.0 schema at {xsd}")
    schema = etree.XMLSchema(etree.parse(xsd))
    doc = etree.fromstring(md.to_xml().encode())
    assert schema.validate(doc), [e.message for e in schema.error_log]


@pytest.mark.skipif(find_c_compiler() is None, reason="no C compiler")
def test_simulate_fmu_with_its_default_validation_runs_the_fmu(gm, md, tmp_path):
    """End to end: the description FMPy used to refuse, served by a bridge,
    simulated with typed start values for both inputs and both typed
    parameters, against the graph."""
    fmpy = pytest.importorskip("fmpy")
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=gm._compiled_step,
        initial_state=gm._state, params=gm.params, param_specs=gm.param_specs(),
        input_resolver=gm._resolve_external_inputs))
    so = build_fmu_binary(tmp_path / "bin")
    with FmuTcpBridge(sidecar, md, master_dt=gm.timestep) as bridge:
        path = write_fmu(md, tmp_path / "gate.fmu", binary=so, endpoint=bridge.endpoint)
        res = fmpy.simulate_fmu(
            str(path), stop_time=5 * DT, step_size=DT, output_interval=DT,
            start_values={"gate.open": True, "gate.n": 2, "gate.params.gain": 4,
                          "gate.params.rate": 2.0},
            output=["gate.level", "gate.count", "gate.was_open"])
    ref = _graph(rate=2.0, gain=4)
    out = ref.run_scan(5, external_inputs={"gate": {"open": jnp.asarray(True),
                                                    "n": jnp.asarray(2, jnp.int32)}})
    assert res["gate.level"][-1] == pytest.approx(float(out["gate"]["level"]), rel=1e-6)
    assert res["gate.count"][-1] == int(out["gate"]["count"]) == 40
    assert bool(res["gate.was_open"][-1])


def test_the_bridge_reads_typed_starts_back_exactly(gm, md):
    """The bridge starts every instance at the description's starts, read
    in their type: a sidecar built with other values is brought into line
    for the Boolean and the Int32 parameters too."""
    other = _graph(enabled=False, gain=7)
    sidecar = FmuSidecar(SidecarConfig(
        schema_token=md.instantiation_token, step_fn=other._compiled_step,
        initial_state=other._state, params=other.params))
    with pytest.warns(UserWarning, match="gate.params.enabled.*gate.params.gain"):
        bridge = FmuTcpBridge(sidecar, md, master_dt=DT)
    try:
        vr = {v.name: v.value_reference for v in md.variables}
        got = values_of(bridge.handle({"op": "get", "vr": [vr["gate.params.enabled"],
                                                           vr["gate.params.gain"]]}))
        assert got.tolist() == [1.0, 3.0]
        assert sidecar.params["nodes"]["gate"]["gain"].dtype == jnp.int32
        assert sidecar.params["nodes"]["gate"]["enabled"].dtype == jnp.bool_
    finally:
        bridge.stop()


def test_a_hand_built_start_in_float_form_is_written_in_the_types_form():
    md = ModelDescription(model_name="m", instantiation_token="tok", variables=[
        FMIVariable(name="time", value_reference=1, dtype="float64",
                    causality="independent", variability="continuous"),
        FMIVariable(name="k", value_reference=2, dtype="int32", causality="parameter",
                    variability="tunable", start="3.0", min=-1.0, max=4.0),
        FMIVariable(name="b", value_reference=3, dtype="bool", causality="parameter",
                    variability="tunable", start="1.0 0.0", shape=(2,)),
        FMIVariable(name="u8", value_reference=4, dtype="uint8", causality="parameter",
                    variability="tunable", start="7", min=-5.0, max=300.0)])
    el = _elements(md)
    assert el["k"].get("start") == "3" and (el["k"].get("min"), el["k"].get("max")) == ("-1", "4")
    assert el["b"].get("start") == "true false"
    # a bound beyond the type's own range adds nothing and is left out
    assert el["u8"].get("min") is None and el["u8"].get("max") is None


@pytest.mark.parametrize("token, dtype, value", [
    ("true", "bool", True), ("0", "bool", False), ("1.0", "bool", True),
    ("-3", "int32", -3), ("4.0", "int64", 4), ("18446744073709551615", "uint64", 2**64 - 1),
    ("9007199254740993", "int64", 9007199254740993), ("INF", "float32", float("inf")),
])
def test_a_start_token_is_read_back_in_its_type(token, dtype, value):
    assert _parse_xs_value(token, dtype) == value


@pytest.mark.parametrize("token, dtype", [("0.5", "int32"), ("2", "bool"), ("yes", "bool"),
                                          ("abc", "float64")])
def test_a_start_token_that_is_not_a_value_of_its_type_is_refused(token, dtype):
    with pytest.raises(ValueError):
        _parse_xs_value(token, dtype)



# ------------------------------------------------------------------ the unit
#
# FMI 3.0's schema gives ``unit`` to ``<Float32>`` and ``<Float64>`` and to
# no other variable element.  The dtypes below are read from the writer's own
# table, so a type added to it is held to the rule here without being named.

#: The dtype of every FMI type the writer knows (``bool_`` is its second
#: spelling of ``bool``).
WRITTEN = sorted({"bool" if name == "bool_" else name for name in _DTYPE_TO_FMI_TYPE})
#: The ones a process has only under ``jax_enable_x64``.
NEED_X64 = ("float64", "int64", "uint64")


def _unit_of(dtype):
    return f"{dtype}_unit"


@stability(StabilityLevel.STABLE)
class Metered(SimulationNode):
    """A boundary field per dtype in ``WRITTEN``, each declared with a unit
    of its own, and a float, an Int32 and a Boolean parameter whose
    ``ParamSpec`` names a unit."""

    def __init__(self, name, timestep, rate=1.0, enabled=True, gain=3):
        super().__init__(name, timestep, rate=rate, enabled=enabled, gain=gain)

    def params_pytree(self):
        return {"rate": jnp.asarray(self.params["rate"], jnp.float32),
                "enabled": jnp.asarray(self.params["enabled"]),
                "gain": jnp.asarray(self.params["gain"], jnp.int32)}

    def param_specs(self):
        return {"rate": ParamSpec(units="1/s"), "gain": ParamSpec(units="counts"),
                "enabled": ParamSpec(units="flag")}

    def boundary_input_spec(self):
        return {f"u_{d}": BoundaryInputSpec(expected_units=_unit_of(d)) for d in WRITTEN}

    def initial_state(self):
        return {"level": jnp.asarray(0.0, jnp.float32)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        fed = sum(jnp.asarray(v, jnp.float32) for v in boundary_inputs.values())
        return {"level": state["level"] + jnp.where(p["enabled"], p["rate"] * dt, 0.0)
                + p["gain"] * fed}


def _metered(dtypes):
    gm = GraphManager()
    gm.add_node(Metered("meter", DT))
    for d in dtypes:
        gm.add_external_input("meter", f"u_{d}", dtype=d)
    gm.compile()
    return build_model_description(gm, model_name="M", model_identifier=MODEL_IDENTIFIER)


def _built_in_nodes_with_integer_inputs():
    """Two built-in nodes whose fields declare ``expected_units``, fed
    through an ``int32`` and an ``int8`` input: the graph the defect was
    reported on."""
    gm = GraphManager()
    gm.add_node(SpringDamperNode("spring", DT, stiffness=15.0, rest_length=0.3))
    gm.add_node(BallNode("ball", DT, initial_position=0.6))
    gm.add_external_input("spring", "anchor_position", dtype=jnp.int32)
    gm.add_external_input("ball", "table_position", dtype=jnp.int8)
    gm.compile()
    return build_model_description(gm, model_name="B", model_identifier=MODEL_IDENTIFIER)


def _hand_built():
    """An input, a parameter and an output of every key of the writer's
    table, each with a unit, and a clock with one."""
    variables = [FMIVariable(name="time", value_reference=1, dtype="float64",
                             causality="independent", variability="continuous", unit="s")]
    for name in sorted(_DTYPE_TO_FMI_TYPE):
        moving = "continuous" if _fmi_kind(name) == "float" else "discrete"
        for causality, variability in (("input", moving), ("parameter", "tunable"),
                                       ("output", moving)):
            variables.append(FMIVariable(
                name=f"{causality}_{name}", value_reference=len(variables) + 1, dtype=name,
                causality=causality, variability=variability, unit=_unit_of(name),
                start=None if causality == "output" else _xs_value(0, name)))
    variables.append(FMIVariable(
        name="tick", value_reference=len(variables) + 1, dtype="clock", causality="input",
        variability="discrete", interval_decimal=0.5, unit="tick_unit"))
    return ModelDescription(model_name="H", instantiation_token="tok", variables=variables,
                            co_simulation_model_identifier=MODEL_IDENTIFIER)


@pytest.fixture(scope="module")
def unit_bearing():
    """Descriptions that hold a unit-bearing variable of every type the
    writer knows: an input of each dtype a default process has, an input of
    all of them under ``jax_enable_x64`` (process-global, so it is on for
    that build alone), the built-in nodes' integer inputs, and a hand-built
    description."""
    cases = {"default": _metered([d for d in WRITTEN if d not in NEED_X64]),
             "built-in nodes": _built_in_nodes_with_integer_inputs(),
             "hand-built": _hand_built()}
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        cases["x64"] = _metered(WRITTEN)
    finally:
        jax.config.update("jax_enable_x64", previous)
    return cases


def test_the_unit_bearing_descriptions_hold_every_type_with_a_unit(unit_bearing):
    """What the tests below read is there to be read: every type of the
    writer's table is a variable that carries a unit in the description
    (``FMIVariable.unit`` keeps what the graph declares, whatever the XML
    may say), an input of each among them, and so are an integer and a
    Boolean parameter."""
    for label in ("x64", "hand-built"):
        md = unit_bearing[label]
        with_unit = {_DTYPE_TO_FMI_TYPE[v.dtype] for v in md.variables
                     if v.unit and v.causality == "input" and not v.is_clock}
        assert with_unit == set(_DTYPE_TO_FMI_TYPE.values()), label
    default = {v.dtype for v in unit_bearing["default"].variables if v.unit and
               v.causality == "input"}
    assert default == set(WRITTEN) - set(NEED_X64)
    for label in ("default", "x64"):
        unit = {v.name: (v.dtype, v.unit) for v in unit_bearing[label].variables}
        for d in WRITTEN if label == "x64" else sorted(default):
            assert unit[f"meter.u_{d}"] == (d, _unit_of(d))
        assert unit["meter.params.gain"] == ("int32", "counts")
        assert unit["meter.params.enabled"] == ("bool", "flag")
        assert unit["meter.params.rate"] == ("float32", "1/s")
    built_in = {v.name: (v.dtype, bool(v.unit)) for v in unit_bearing["built-in nodes"].variables}
    assert built_in["spring.anchor_position"] == ("int32", True)
    assert built_in["ball.table_position"] == ("int8", True)
    assert any(v.is_clock and v.unit for v in unit_bearing["hand-built"].variables)


def test_a_unit_is_written_on_a_float_variable_and_on_no_other(unit_bearing):
    for label, md in unit_bearing.items():
        root = ET.fromstring(md.to_xml())
        declared = {v.name: v.unit for v in md.variables}
        written, kept_out = set(), 0
        for el in root.find("ModelVariables"):
            name = el.get("name")
            if el.tag in ("Float32", "Float64"):
                assert el.get("unit") == (declared[name] or None), (label, name)
                written.add(el.get("unit"))
            else:
                assert el.get("unit") is None, (label, el.tag, name)
                kept_out += bool(declared[name])
        assert kept_out, label
        # ... and a unit no element carries is not defined: the units of the
        # integers, the Booleans and the clock are in no <Unit>
        defined = [u.get("name") for u in root.find("UnitDefinitions")]
        assert sorted(defined) == sorted(written - {None}), label


def test_a_units_definition_is_left_out_with_its_last_float_variable():
    """``<UnitDefinitions>`` goes when no element carries a unit, and a
    unit an integer shares with a float stays for the float."""
    def description(*variables):
        return ET.fromstring(ModelDescription(
            model_name="m", instantiation_token="tok", variables=list(variables)).to_xml())

    count = FMIVariable(name="n", value_reference=1, dtype="int32", causality="output",
                        variability="discrete", unit="m")
    flag = FMIVariable(name="b", value_reference=2, dtype="bool", causality="output",
                       variability="discrete", unit="m")
    length = FMIVariable(name="x", value_reference=3, dtype="float32", causality="output",
                         variability="continuous", unit="m")
    assert description(count, flag).find("UnitDefinitions") is None
    shared = description(count, flag, length)
    assert [u.get("name") for u in shared.find("UnitDefinitions")] == ["m"]
    assert {el.get("name"): el.get("unit") for el in shared.find("ModelVariables")} == {
        "n": None, "b": None, "x": "m"}


def test_fmpy_validates_a_unit_bearing_variable_of_every_type(unit_bearing, tmp_path):
    pytest.importorskip("fmpy")
    from fmpy.model_description import read_model_description
    from fmpy.validation import validate_fmu

    for i, (label, md) in enumerate(unit_bearing.items()):
        fmu = tmp_path / f"case{i}.fmu"
        with zipfile.ZipFile(fmu, "w") as zf:
            zf.writestr("modelDescription.xml", md.to_xml())
        assert validate_fmu(str(fmu)) == [], label
        read_model_description(str(fmu), validate=True)


def test_the_fmi3_schema_validates_a_unit_bearing_variable_of_every_type(unit_bearing):
    fmpy = pytest.importorskip("fmpy")
    etree = pytest.importorskip("lxml.etree")
    xsd = os.path.join(os.path.dirname(fmpy.__file__), "schema", "fmi3",
                       "fmi3ModelDescription.xsd")
    if not os.path.exists(xsd):
        pytest.skip(f"FMPy {fmpy.__version__} ships no FMI 3.0 schema at {xsd}")
    schema = etree.XMLSchema(etree.parse(xsd))
    for label, md in unit_bearing.items():
        doc = etree.fromstring(md.to_xml().encode())
        assert schema.validate(doc), (label, [e.message for e in schema.error_log])
