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
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zipfile

import jax.numpy as jnp
import pytest

from maddening.core.compliance.metadata import StabilityLevel
from maddening.core.compliance.stability import stability
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.core.params import ParamSpec
from maddening.fmi import build_model_description
from maddening.fmi.model_description import FMIVariable, ModelDescription, _parse_xs_value
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

