"""A graph without a geometry-dependent mapping compiles to the program it always did.

``scripts/capture_step_programs.py`` lowers five programs of each of a
fixed set of graphs -- the step with traced and with baked parameters, a
scan of three steps, a sweep of two, the adaptive step -- and
``tests/core/data/step_program_digests.json`` holds the sha256 of their
text as captured on the commit the geometry feature starts from, per jax
version.  The tests here lower them again and compare.

Program text, not results: result bits depend on the CPU and on jaxlib,
so a committed capture of results does not travel; identical program text
gives identical results on any machine, and depends only on the jax
version.

* **A jax version with no captured entry fails.**  It does not skip: a
  gate that passes for lack of a capture is not a gate.  Until the capture
  is taken (the first step of the feature work) every comparison below
  fails, by design.
* **The gate can fail** (the two mutation tests): a numerically neutral
  ``* 1.0`` on the edge path, and the incoming edges of a node resolved in
  the opposite order, each change the digests, so "unchanged" is not a
  property of the comparison.
* **Trace counts** of the same graphs, and of a graph that does carry a
  geometry edge: one trace for any number of steps, none for a parameter
  write, one per scan length, one for a sweep.

The sharded gate graph needs two devices, so its digests are taken in a
fresh process that forces two virtual CPU devices before jax is imported.
"""

from __future__ import annotations

import functools
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import maddening
import maddening.core._graph_specs as _graph_specs
import maddening.core.coupling._coupled_block as _coupled_block
import maddening.core.coupling._fixed_point as _fixed_point
import maddening.core.graph_manager as graph_manager

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "capture_step_programs.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("capture_step_programs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture = _load_script()
GRAPHS = capture.gate_graphs()
#: Every gate graph this process can build (the sharded one needs two devices).
IN_PROCESS = [name for name in GRAPHS if name != capture.SHARDED]
#: Small graphs the mutation tests rebuild.
CHEAP = ["rbf-edge", "flux-outside-a-group", "named/chain-into-ring/choice-0"]
_DIGEST = re.compile(r"[0-9a-f]{64}|refused:[A-Za-z_]+:[0-9a-f]{16}")


@functools.lru_cache(maxsize=None)
def _digests(name: str) -> dict:
    return capture.graph_digests(GRAPHS[name])


def _assert_captured(current: dict) -> None:
    problems = capture.differences(capture.load(), current)
    assert not problems, (
        "the compiled programs of graphs that use no geometry-dependent mapping are not "
        "the captured ones:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", IN_PROCESS)
def test_the_programs_of_a_graph_without_a_geometry_dependent_mapping_are_unchanged(name):
    _assert_captured({name: _digests(name)})


def _fresh_process_digests(*names: str) -> dict:
    """The digests of *names* from ``capture_step_programs.py --print`` in a new process."""
    env = dict(os.environ)
    src = str(Path(maddening.__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([src] + [p for p in env.get("PYTHONPATH", "").split(
        os.pathsep) if p and p != src])
    only = [arg for name in names for arg in ("--only", name)]
    out = subprocess.run([sys.executable, str(SCRIPT), "--print", *only], env=env,
                         capture_output=True, text=True, timeout=900, cwd=str(REPO_ROOT))
    assert out.returncode == 0, out.stderr[-2000:]
    payload = json.loads(out.stdout)
    assert payload["jax"] == jax.__version__
    return payload["graphs"]


def test_the_programs_of_a_sharded_graph_are_unchanged():
    _assert_captured(_fresh_process_digests(capture.SHARDED))


def test_the_capture_holds_exactly_the_gate_graphs_of_this_tree():
    """Every gate graph, all five programs, for the jax that is running --
    and no entry for a graph the gate no longer builds, which would be a
    capture nothing compares."""
    data = capture.load()
    assert data["captures"], (
        f"{capture.DIGESTS.relative_to(REPO_ROOT)} does not exist or is empty: capture it on "
        f"the base commit with `python scripts/capture_step_programs.py --write`")
    assert re.fullmatch(r"[0-9a-f]{40}", str(data["commit"])), data["commit"]
    assert jax.__version__ in data["captures"], (
        f"no capture for jax {jax.__version__}; the file holds {sorted(data['captures'])}")
    for version, entry in data["captures"].items():
        assert sorted(entry) == sorted(GRAPHS), (version, sorted(set(entry) ^ set(GRAPHS)))
        for graph, programs in entry.items():
            assert tuple(programs) == capture.PROGRAMS, (version, graph, list(programs))
            for program, digest in programs.items():
                assert _DIGEST.fullmatch(digest), (version, graph, program, digest)


def test_the_gate_graphs_are_the_documented_set():
    """Four named structures at two configurations, the same with sparse
    mappings where the tree has them, a multi-rate graph, a sub-cycled
    group, a group with a predictor, a flux edge outside a group, an
    ``rbf`` edge and a sharded node."""
    names = set(GRAPHS)
    named = {n for n in names if n.startswith("named/")}
    assert len(named) == 8 and {n.rsplit("/", 1)[1] for n in named} == {"choice-0", "choice-1"}
    assert capture.has_sparse_mappings()
    sparse = {n for n in names if n.startswith("sparse-ragged/")}
    assert len(sparse) == 8
    assert names - named - sparse == {"sparse-scatter/chain-into-ring/choice-0",
                                      "multi-rate/chain-into-ring",
                                      "sub-cycled/chain-into-ring",
                                      "predictor/chain-into-ring", capture.AITKEN,
                                      "flux-outside-a-group",
                                      "rbf-edge", capture.SHARDED}


def test_every_gate_graph_lowers_a_step_and_a_scan():
    """A refusal is a legitimate record for the adaptive step of a graph the
    adaptive steppers do not take -- not for the step itself."""
    for name in IN_PROCESS:
        digests = _digests(name)
        for program in ("step", "step_baked_params", "scan3", "sweep2"):
            assert re.fullmatch(r"[0-9a-f]{64}", digests[program]), (name, program,
                                                                     digests[program])
        assert len({digests[p] for p in ("step", "step_baked_params", "scan3", "sweep2")}) == 4


# ---------------------------------------------------------------------------
# The gate can fail
# ---------------------------------------------------------------------------


def test_the_digests_do_not_depend_on_the_process():
    """Twice here, and once in a fresh process: the same digests (a digest
    that moved by itself would make the gate fail for no reason, or teach
    people to recapture it)."""
    name = "rbf-edge"
    again = capture.graph_digests(GRAPHS[name])
    assert again == _digests(name)
    assert _fresh_process_digests(name)[name] == again


@pytest.mark.parametrize("name", CHEAP)
def test_a_numerically_neutral_edit_of_the_edge_path_changes_every_program(name, monkeypatch):
    """``value * 1.0`` after every edge: no result moves, and every program
    that resolves an edge has another text."""
    base = dict(_digests(name))
    original = _graph_specs._apply_edge                     # noqa: SLF001

    def neutral(*args, **kwargs):
        return original(*args, **kwargs) * 1.0

    # Both readers: the plain step reads the attribute of ``_graph_specs``,
    # the group block holds its own name for it.
    assert _coupled_block._apply_edge is original           # noqa: SLF001
    monkeypatch.setattr(_graph_specs, "_apply_edge", neutral)
    monkeypatch.setattr(_coupled_block, "_apply_edge", neutral)
    mutated = capture.graph_digests(GRAPHS[name])
    changed = {p for p in capture.PROGRAMS if mutated[p] != base[p]}
    lowered = {p for p in capture.PROGRAMS if not base[p].startswith("refused:")}
    assert lowered >= {"step", "step_baked_params", "scan3", "sweep2"}
    assert changed == lowered, (name, sorted(lowered - changed))


class _ReversedList(list):
    """A list that iterates last to first."""

    def __iter__(self):
        return reversed(list(super().__iter__()))


def test_resolving_a_node_s_incoming_edges_in_another_order_changes_every_program(monkeypatch):
    """The same operations in another order: ``edges_by_target`` handing each
    node its incoming edges last to first.  The gate is that strict, so a
    change that passes it has kept the order of every existing trace
    operation."""
    name = "named/chain-into-ring/choice-0"
    base = dict(_digests(name))
    real = graph_manager.defaultdict

    def reversing(factory=None, *args, **kwargs):
        return real(_ReversedList if factory is list else factory, *args, **kwargs)

    monkeypatch.setattr(graph_manager, "defaultdict", reversing)
    mutated = capture.graph_digests(GRAPHS[name])
    for program in ("step", "step_baked_params", "scan3", "sweep2"):
        assert mutated[program] != base[program], program


def test_dropping_aitken_s_second_pass_changes_the_programs_of_the_aitken_graph(monkeypatch):
    """The exit rule of the default solver is part of what the gate holds:
    with ``_TWO_PASS_EXIT`` emptied (Aitken leaving on the first pass under
    the threshold) every program of the Aitken graph that runs its group
    has another text, and a graph without Aitken has the same one."""
    assert _fixed_point._TWO_PASS_EXIT == ("aitken",)       # noqa: SLF001
    base = dict(_digests(capture.AITKEN))
    other = "named/chain-into-ring/choice-0"
    other_base = dict(_digests(other))
    monkeypatch.setattr(_fixed_point, "_TWO_PASS_EXIT", ())
    mutated = capture.graph_digests(GRAPHS[capture.AITKEN])
    for program in ("step", "step_baked_params", "scan3", "sweep2"):
        assert mutated[program] != base[program], program
    assert capture.graph_digests(GRAPHS[other]) == other_base


def test_the_comparison_reports_a_missing_capture_a_missing_graph_and_a_changed_program():
    """``differences`` itself: no entry for the running jax, a graph the
    capture does not hold, and one digest off are each reported; an exact
    match is not."""
    current = {"g": {p: "0" * 64 for p in capture.PROGRAMS}}
    version = jax.__version__
    assert capture.differences({"captures": {version: current}}, current) == []
    assert "no capture for jax" in capture.differences({"captures": {}}, current)[0]
    assert "no capture for jax" in capture.differences({"captures": {"0.0.0": current}},
                                                       current)[0]
    missing = capture.differences({"captures": {version: {}}}, current)
    assert missing == [f"g: no captured entry for jax {version}"]
    other = {"g": dict(current["g"], scan3="1" * 64)}
    changed = capture.differences({"captures": {version: other}}, current)
    assert len(changed) == 1 and changed[0].startswith("g: scan3 is ")


# ---------------------------------------------------------------------------
# Trace counts
# ---------------------------------------------------------------------------

_COUNTED_PER_PUSH = ("named/chain-into-ring/choice-0", "rbf-edge")
_COUNTED = [name if name in _COUNTED_PER_PUSH else pytest.param(name, marks=pytest.mark.slow)
            for name in IN_PROCESS]


def assert_traces_once(gm) -> None:
    """One trace for three steps and for a parameter write between steps;
    one more scan trace for a first ``run_scan`` and none for a second of
    the same length."""
    assert gm.trace_count == 0
    for _ in range(3):
        gm.step()
    assert gm.trace_count == 1
    params = gm.params
    gm.params = {**params, "mappings": jax.tree.map(lambda v: v * 0.5, params["mappings"])}
    gm.step()
    assert gm.trace_count == 1, "a parameter write retraced the step"
    before = gm.scan_trace_count
    gm.run_scan(3)
    assert gm.scan_trace_count == before + 1
    gm.run_scan(3)
    assert gm.scan_trace_count == before + 1, "a second run_scan of the same length retraced"
    assert gm.trace_count == 1


# Per push: tests/core/test_step_program_digests.py::test_a_gate_graph_traces_its_step_and_its_scan_once
# (on chain-into-ring and the rbf pair; the slow parameters are the same
# check on the other gate graphs, each a step and a scan compiled)
@pytest.mark.parametrize("name", _COUNTED)
def test_a_gate_graph_traces_its_step_and_its_scan_once(name):
    assert_traces_once(GRAPHS[name]())


def test_a_graph_with_a_geometry_edge_traces_its_step_its_scan_and_its_sweep_once():
    """The same counts for a coupling group whose two edges read a moving
    geometry, a write of the parameter that moves it, and a sweep."""
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    c = gg.case("counted", adv=0.3, down="target", up="source",
                group=dict(max_iterations=200, tolerance=1e-5))
    gm = gg.build(gg.two_body(c))
    assert gm.params["mappings"] == {"F.x->P.u": {}, "P.x->F.u": {}}
    assert gm.trace_count == 0
    for _ in range(3):
        gm.step()
    assert gm.trace_count == 1
    params = gm.params
    nodes = {name: dict(leaves) for name, leaves in params["nodes"].items()}
    nodes["P"]["rate"] = nodes["P"]["rate"] * 2
    gm.params = {**params, "nodes": nodes}
    gm.step()
    assert gm.trace_count == 1, "a write of the parameter that moves the geometry retraced"
    before = gm.scan_trace_count
    gm.run_scan(3)
    gm.run_scan(3)
    assert gm.scan_trace_count == before + 1
    batch = {name: {f: jnp.stack([v, v + 0.01]) for f, v in gm.get_node_state(name).items()}
             for name in gm.node_names}
    gm.run_sweep(3, batch)
    gm.run_sweep(3, batch)
    assert gm.scan_trace_count == before + 2, "the sweep traced more than once"
    assert gm.trace_count == 1
