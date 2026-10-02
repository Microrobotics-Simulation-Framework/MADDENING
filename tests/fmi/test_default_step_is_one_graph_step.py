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

import xml.etree.ElementTree as ET

import pytest

from maddening.core.graph_manager import GraphManager
from maddening.fmi import build_model_description
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
    gm = _springs(0.01, 0.02, subcycled=(0, 1))
    md = build_model_description(gm, model_name="m", default_step_size=0.005)
    assert md.default_step_size == 0.005


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
