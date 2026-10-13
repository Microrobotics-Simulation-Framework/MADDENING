"""An FMU's ``DefaultExperiment`` step is one step of the graph.

``build_model_description`` defaulted ``stepSize`` to a ``_base_dt``
attribute that nothing set, falling back to the smallest node timestep
(MADD-ANO-097).  That is the graph's step on a uniform-rate graph and on a
multi-rate one whose timesteps divide each other -- which is what
MADD-ANO-078's fix verified, and still holds -- but not otherwise: nodes at
0.002 and 0.003 step by 0.001, and a sub-cycling coupling group of 0.01
and 0.02 steps by 0.02.  One sidecar step is one graph step, so a master
following the old default ran each 0.02 step as if it were 0.01.  The
default is now ``gm.timestep``, and the ``multi_clock`` intervals come from
the same scheduled timesteps.
"""

from __future__ import annotations

import math
import os
import re
import warnings
import xml.etree.ElementTree as ET

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
from maddening.fmi.model_description import _CLOCK_DTYPE, FMIVariable, ModelDescription
from maddening.nodes.spring import SpringDamperNode


def _springs(*dts, subcycled=(), outside=()):
    """Springs named ``s0, s1, ...``; *subcycled* indexes one sub-cycling group."""
    gm = GraphManager()
    for i, dt in enumerate(dts):
        gm.add_node(SpringDamperNode(f"s{i}", dt, rest_length=1.0 - 2.0 * (i % 2),
                                     initial_position=float(i)))
    for i in range(len(dts) - 1):
        gm.add_edge(f"s{i}", f"s{i + 1}", "position", "anchor_position")
    gm.add_edge(f"s{len(dts) - 1}", "s0", "position", "anchor_position")
    if subcycled:
        gm.add_coupling_group([f"s{i}" for i in subcycled], max_iterations=20,
                              tolerance=1e-6, subcycling=True)
    with warnings.catch_warnings():
        # A group of two springs in a ring of three is part of a larger loop
        # through the third, and compile() says so (CPL-181); the clocks are
        # this file's point, not which edge closes that loop.
        warnings.filterwarnings("ignore", message=".*part of a larger feedback loop.*",
                                category=UserWarning)
        gm.compile()
    return gm


#: name -> (graph, the step)
CASES = {
    "single_rate": (lambda: _springs(0.01, 0.01), 0.01),
    "multirate_dividing": (lambda: _springs(0.01, 0.05), 0.01),
    "multirate_not_dividing": (lambda: _springs(0.002, 0.003), 0.001),
    "subcycled": (lambda: _springs(0.01, 0.02, subcycled=(0, 1)), 0.02),
    "subcycled_beside_a_slower_node": (
        lambda: _springs(0.005, 0.02, 0.03, subcycled=(0, 1)), 0.01),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_default_step_is_one_graph_step(case):
    build, step = CASES[case]
    gm = build()
    md = build_model_description(gm, model_name="m")
    assert md.default_step_size == pytest.approx(step, rel=1e-12)
    assert md.default_step_size == gm.timestep
    experiment = ET.fromstring(md.to_xml()).find("DefaultExperiment")
    assert float(experiment.get("stepSize")) == pytest.approx(step, rel=1e-12)


def test_a_single_rate_graph_keeps_madd_ano_078s_step():
    """The case MADD-ANO-078 fixed is unchanged: the fastest node's timestep."""
    gm = _springs(0.01, 0.01)
    assert build_model_description(gm, model_name="m").default_step_size == 0.01


def test_an_explicit_default_step_size_still_wins():
    """... where it is a whole number of graph steps (0.02 here, the
    sub-cycling group's step); a quarter of one is refused when the
    description is built, not by the bridge that would serve it."""
    gm = _springs(0.01, 0.02, subcycled=(0, 1))
    md = build_model_description(gm, model_name="m", default_step_size=0.04)
    assert md.default_step_size == 0.04 and md.graph_timestep == 0.02
    with pytest.raises(ValueError, match="not a whole number of steps of graph_timestep=0.02"):
        build_model_description(gm, model_name="m", default_step_size=0.005)


@pytest.mark.parametrize("case", sorted(CASES))
def test_every_clock_is_a_whole_number_of_default_steps(case):
    build, step = CASES[case]
    gm = build()
    md = build_model_description(gm, model_name="m", multi_clock=True)
    for clock in md.clocks():
        ratio = clock.interval_decimal / md.default_step_size
        assert ratio == pytest.approx(round(ratio), rel=1e-9), (clock.name, ratio)
        assert round(ratio) >= 1


def test_a_sub_cycled_member_is_on_its_groups_clock():
    """``s0`` (0.005) is sub-stepped inside a group whose macro step is
    0.02: its outputs change every 0.02, so it shares the coarse member's
    clock rather than ticking at 0.005, faster than the 0.01 master step."""
    gm = _springs(0.005, 0.02, 0.03, subcycled=(0, 1))
    md = build_model_description(gm, model_name="m", multi_clock=True)
    clocks = {c.value_reference: c.interval_decimal for c in md.clocks()}
    assert sorted(clocks.values()) == pytest.approx([0.02, 0.03])
    by_name = {v.name: v for v in md.variables}
    s0, s1 = by_name["s0.position"].clocks, by_name["s1.position"].clocks
    assert s0 == s1 and clocks[s0[0]] == pytest.approx(0.02)
    assert clocks[by_name["s2.position"].clocks[0]] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# A NumPy scalar step size is a number, and is written as one
# ---------------------------------------------------------------------------
#
# ``default_step_size=np.float64(0.05)`` -- also ``np.int64(5) * dt``,
# ``np.round(...)`` -- was stored as given and written with ``repr``:
# ``stepSize="np.float64(0.05)"`` under NumPy 2, not an xs:double, so FMPy
# refused the FMU and nothing on the exporting side said so (B1 round 11).

#: The attribute forms FMI 3.0's schema reads as an xs:double.
XS_DOUBLE = re.compile(r"^(?:[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|INF|-INF|NaN)$")

#: Every NumPy spelling of "two steps of a 0.25 s graph" (0.25 so that the
#: float16 and float32 ones are the number itself).
NUMPY_STEPS = [np.float64(0.5), np.float32(0.5), np.float16(0.5), np.int64(2) * 0.25,
               np.round(0.5, 6), np.longdouble(0.5), np.float64(0.25) * 2]
NUMPY_INTEGER_STEPS = [np.int64(1), np.int32(1), np.uint8(1), np.int8(1), 1]


def _schema_problems(xml: str) -> list:
    fmpy = pytest.importorskip("fmpy")
    etree = pytest.importorskip("lxml.etree")
    xsd = os.path.join(os.path.dirname(fmpy.__file__), "schema", "fmi3",
                       "fmi3ModelDescription.xsd")
    if not os.path.exists(xsd):
        pytest.skip(f"FMPy {fmpy.__version__} ships no FMI 3.0 schema at {xsd}")
    schema = etree.XMLSchema(etree.parse(xsd))
    schema.validate(etree.fromstring(xml.encode()))
    return [e.message for e in schema.error_log]


def _no_attribute_is_a_repr(xml: str) -> None:
    for element in ET.fromstring(xml).iter():
        for key, value in element.attrib.items():
            if key in ("description", "name", "unit", "generationTool"):
                continue                                # prose, not a number
            assert "np." not in value and "(" not in value and "Array" not in value, (
                element.tag, key, value)


@pytest.mark.parametrize("step", NUMPY_STEPS + NUMPY_INTEGER_STEPS,
                         ids=lambda s: f"{type(s).__name__}-{s}")
def test_a_numpy_scalar_step_size_is_stored_and_written_as_the_float_it_is(step):
    gm = _springs(0.25)
    md = build_model_description(gm, model_name="m", default_step_size=step)
    plain = build_model_description(gm, model_name="m", default_step_size=float(step))
    assert type(md.default_step_size) is float and md.default_step_size == float(step)
    assert md.instantiation_token == plain.instantiation_token
    xml = md.to_xml()
    assert xml == plain.to_xml()
    written = ET.fromstring(xml).find("DefaultExperiment").attrib
    assert XS_DOUBLE.match(written["stepSize"]) and float(written["stepSize"]) == float(step)
    _no_attribute_is_a_repr(xml)
    assert _schema_problems(xml) == []


@pytest.mark.parametrize("bad, why", [
    (True, "must be a number"), (np.True_, "must be a number"), ("0.5", "must be a number"),
    (jnp.asarray(0.5), "must be a number"), (np.asarray(0.5), "must be a number"),
    (np.asarray([0.5]), "must be a number"), ([0.5], "must be a number"),
    (0.5 + 0j, "must be a number"), (np.complex128(0.5), "must be a number"),
    (math.nan, "must be finite and positive"), (np.float64("nan"), "must be finite and positive"),
    (math.inf, "must be finite and positive"), (np.float32("inf"), "must be finite and positive"),
    (0, "must be finite and positive"), (0.0, "must be finite and positive"), (-0.5, "must be finite and positive"),
    (np.int64(-1), "must be finite and positive"), (10 ** 400, "must be finite and positive"),
], ids=repr)
def test_a_step_size_that_is_not_a_finite_positive_number_is_refused_at_build(bad, why):
    gm = _springs(0.25)
    with pytest.raises(ValueError, match=f"default_step_size {why}"):
        build_model_description(gm, model_name="m", default_step_size=bad)


@pytest.mark.parametrize("scalar", [np.float64, np.float32, np.float16, np.int64, np.longdouble],
                         ids=lambda t: t.__name__)
def test_every_number_a_hand_built_description_writes_is_an_xs_double(scalar):
    """The other three ``DefaultExperiment`` attributes and a clock's
    interval, which only a hand-built description can set: each NumPy
    scalar type is written as the number, and the result is schema-valid."""
    def described(one, two, half_or_one):
        return ModelDescription(
            model_name="m", instantiation_token="tok",
            default_start_time=one, default_stop_time=two,
            default_tolerance=half_or_one, default_step_size=one,
            variables=[
                FMIVariable(name="time", value_reference=1, dtype="float64",
                            causality="independent", variability="continuous"),
                FMIVariable(name="clock_0", value_reference=2, dtype=_CLOCK_DTYPE,
                            causality="input", variability="discrete",
                            interval_decimal=two),
            ])
    xml = described(scalar(1), scalar(2), scalar(1)).to_xml()
    assert xml == described(1.0, 2.0, 1.0).to_xml()
    written = ET.fromstring(xml).find("DefaultExperiment").attrib
    assert written == {"startTime": "1.0", "stopTime": "2.0", "tolerance": "1.0",
                       "stepSize": "1.0"}
    _no_attribute_is_a_repr(xml)
    assert _schema_problems(xml) == []


def test_a_non_finite_experiment_time_is_written_in_the_schemas_spelling():
    md = ModelDescription(model_name="m", instantiation_token="tok", variables=[],
                          default_stop_time=np.float64("inf"), default_start_time=-math.inf,
                          default_tolerance=np.float32("nan"))
    written = ET.fromstring(md.to_xml()).find("DefaultExperiment").attrib
    assert (written["stopTime"], written["startTime"], written["tolerance"]) == (
        "INF", "-INF", "NaN")
    assert all(XS_DOUBLE.match(v) for v in written.values())
