"""The root prim's ``maddening:baseDt`` is one step of the graph, written and read back.

Both writers read a ``_base_dt`` attribute that nothing ever set
(MADD-ANO-097).  ``USDWriter`` therefore wrote 0.01 for every graph, and
``save_graph_to_usd`` fell back to the smallest node timestep, which is not
the step of a graph whose timesteps do not divide each other (0.002 and
0.003 step by 0.001) or of one with a sub-cycling coupling group (0.01 and
0.02 in one group step by 0.02).  The writers now write ``gm.timestep``,
and ``maddening:isMultirate`` from the same scheduled timesteps, so a stage
saved before the first compile says what the compile will do.

Each case writes a stage to a ``.usda`` file, opens it again and reads the
two attributes back.
"""

from __future__ import annotations

import pytest
from pxr import Usd

from maddening.core.graph_manager import GraphManager
from maddening.nodes.spring import SpringDamperNode
from maddening.usd.serialization import save_graph_to_usd
from maddening.usd.writer import USDWriter


def _pair(dt_a, dt_b, *, subcycled=False):
    gm = GraphManager()
    gm.add_node(SpringDamperNode("a", dt_a, rest_length=1.0, initial_position=1.0))
    gm.add_node(SpringDamperNode("b", dt_b, rest_length=-1.0))
    gm.add_edge("a", "b", "position", "anchor_position")
    gm.add_edge("b", "a", "position", "anchor_position")
    if subcycled:
        gm.add_coupling_group(["a", "b"], max_iterations=20, tolerance=1e-6,
                              subcycling=True)
    return gm


#: name -> (graph builder, baseDt, isMultirate)
CASES = {
    "single_rate_off_the_schema_default": (lambda: _pair(0.001, 0.001), 0.001, False),
    "multirate_dividing": (lambda: _pair(0.001, 0.005), 0.001, True),
    "multirate_not_dividing": (lambda: _pair(0.002, 0.003), 0.001, True),
    "subcycled": (lambda: _pair(0.01, 0.02, subcycled=True), 0.02, False),
}


def _read_back(path):
    stage = Usd.Stage.Open(str(path))
    root = stage.GetPrimAtPath("/Simulation")
    return (root.GetAttribute("maddening:baseDt").Get(),
            root.GetAttribute("maddening:isMultirate").Get())


@pytest.mark.parametrize("compiled", [True, False], ids=["compiled", "never_compiled"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_save_graph_to_usd_writes_the_graph_step(case, compiled, tmp_path):
    build, base_dt, multirate = CASES[case]
    gm = build()
    if compiled:
        gm.compile()
    path = tmp_path / "graph.usda"
    stage = Usd.Stage.CreateNew(str(path))
    save_graph_to_usd(gm, stage)
    stage.Save()
    got_dt, got_multirate = _read_back(path)
    assert got_dt == pytest.approx(base_dt, rel=1e-12)
    assert got_dt == pytest.approx(gm.timestep, rel=1e-12)
    assert got_multirate is multirate
    if compiled:
        assert got_multirate is gm.is_multirate


@pytest.mark.parametrize("case", sorted(CASES))
def test_the_frame_writer_writes_the_graph_step(case, tmp_path):
    build, base_dt, multirate = CASES[case]
    gm = build()
    gm.compile()
    path = tmp_path / "frames.usda"
    stage = Usd.Stage.CreateNew(str(path))
    writer = USDWriter(stage, gm)
    writer.write_frame(gm.step(), 1.0)
    stage.Save()
    got_dt, got_multirate = _read_back(path)
    assert got_dt == pytest.approx(base_dt, rel=1e-12)
    assert got_multirate is multirate


def test_an_empty_graph_keeps_the_schema_defaults(tmp_path):
    path = tmp_path / "empty.usda"
    stage = Usd.Stage.CreateNew(str(path))
    save_graph_to_usd(GraphManager(), stage)
    USDWriter(stage, GraphManager(), root_path="/Frames")
    stage.Save()
    assert _read_back(path) == (0.01, False)
    reopened = Usd.Stage.Open(str(path))
    assert reopened.GetPrimAtPath("/Frames").GetAttribute("maddening:baseDt").Get() == 0.01
