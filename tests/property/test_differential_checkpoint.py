"""Differential oracle 3: a checkpoint resumes the uninterrupted run.

``save_state`` at step *k*, then ``load_state`` and ``N`` more steps, must be
the run that never stopped -- bit for bit, ``_meta`` included (the
multi-rate step counter and sub-step phase, a coupling group's predictor
history, IQN-IMVJ warm starts, diagnostics).  Stated over:

* every cheap built-in node alone, with calibrated (written) parameters;
* coupled groups carrying predictor history and IQN-IMVJ warm starts, a
  sub-cycling multi-rate group, and a multi-rate graph with no group;
* the REST ``/checkpoint/save`` and ``/checkpoint/load`` routes under the
  server's checkpoint root, crossed with the in-process calls;
* generated multi-node graphs and the costly nodes, in the slow lane.

For each split the checkpoint is loaded into a second graph of the same
family -- newly built, or one that has already stepped past *k* -- whose
parameter leaves have been moved first, so every slot and every leaf has
to be overwritten by the checkpoint rather than merely kept.  Then:

1. the restored state (``_meta`` included) and parameters equal the saved
   ones field for field *before* stepping, so a slot a checkpoint drops is
   caught even where it would not move the trajectory (a diagnostic
   counter);
2. ``N`` more steps equal the uninterrupted run;
3. ``reset_state()`` gives the state of a freshly compiled graph, ``_meta``
   seeds included (the state a checkpoint of step 0 holds).

Tolerance: none.  ``run`` re-enters one compiled step, so a rollout split
anywhere is bit-identical to an unsplit one.

What it cannot see: state held outside ``gm._state`` and ``gm.params`` (a
node's private attribute changed after construction), and a defect that
``save_state`` and ``load_state`` share (both use one key scheme).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient
from hypothesis import given, settings
from hypothesis import strategies as st

from maddening.api.server import SimulationServer
from maddening.core.graph_manager import GraphManager
from maddening.nodes import BallNode, HeatNode, SpringDamperNode

from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_STANDARD
from tests.property.differential import (
    note,
    assert_trees_identical,
    checkpoint_path,
    full_state,
    no_cloud_launch,
    params_tree,
    rollout,
    tmp_dir,
)
from tests.property.node_catalogue import CHEAP_KINDS, COSTLY_KINDS, KINDS, REGISTRY

N_STEPS = 3


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


def check_checkpoint(original: GraphManager, resumed: GraphManager, pristine: dict,
                     *, split: int, ahead: int) -> None:
    """Hold one graph family to the oracle at ``split``.

    ``original`` and ``resumed`` are two graphs of the family at their
    initial state (freshly built, or reset); ``pristine`` is the full state
    of a freshly compiled graph of the family.  ``resumed`` steps ``split +
    ahead`` and has every parameter leaf its step reads moved
    (:func:`_detune`) before the checkpoint is loaded into it (``ahead ==
    0`` on a freshly built graph is the plain resume), so every slot and
    every leaf it carries has to be overwritten rather than merely kept.  Afterwards ``original`` is
    reset and must hold ``pristine`` bit for bit, ``_meta`` seeds included;
    the same compiled step from the same state is the same rollout.
    """
    original.run(split)
    with tmp_dir() as tmp:
        path = original.save_state(checkpoint_path(tmp))
        saved_state, saved_params = full_state(original), params_tree(original)
        resumed.run(split + ahead)
        _detune(resumed)
        resumed.load_state(path)
    assert_trees_identical(saved_state, full_state(resumed), what="restored state")
    assert_trees_identical(saved_params, params_tree(resumed), what="restored params")
    assert_trees_identical(rollout(original, N_STEPS), rollout(resumed, N_STEPS),
                           what="continued trajectory")
    original.reset_state()
    assert_trees_identical(pristine, full_state(original), what="reset state")


def _detune(gm: GraphManager) -> None:
    """Move every parameter leaf the step reads, so that only the checkpoint
    can put it back: a resumed graph built like the original already holds
    the original's values, and a ``load_state`` that skipped one would go
    unseen."""
    reads = gm._params_read_by_step() or set()  # noqa: SLF001
    for owner, key in reads:
        leaf = gm.params["nodes"][owner][key]
        if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating):
            gm.params["nodes"][owner][key] = (jnp.asarray(leaf) * 1.5 + 0.25).astype(
                jnp.asarray(leaf).dtype)


def check_fresh_family(build, *, split: int, ahead: int) -> None:
    """:func:`check_checkpoint` on two freshly built graphs of a family."""
    original = build()
    pristine = full_state(original)
    check_checkpoint(original, build(), pristine, split=split, ahead=ahead)


# ---------------------------------------------------------------------------
# Per push: every cheap node, with written parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind_name", CHEAP_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_checkpoint_of_a_built_in_node_resumes_the_uninterrupted_run(kind_name, data):
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    split = data.draw(st.integers(min_value=0, max_value=3), label="split")
    factors = {key: data.draw(st.sampled_from([0.5, 0.9, 1.1, 2.0]), label=key)
               for key in sorted(kind.safe) if data.draw(st.booleans(), label=f"move {key}")}

    def write(gm):
        live = gm.params["nodes"].get(kind.name, {})
        for key, factor in factors.items():
            if key in live:
                lo, hi = kind.safe[key]
                moved = np.clip(np.asarray(live[key]) * factor, lo, hi)
                live[key] = jnp.asarray(moved.astype(np.asarray(live[key]).dtype))

    def build():
        gm = kind.graph(kwargs)
        write(gm)
        return gm

    check_fresh_family(build, split=split,
                       ahead=data.draw(st.sampled_from([0, 2]), label="ahead"))


# ---------------------------------------------------------------------------
# Per push: _meta -- predictor history, warm starts, multi-rate phase
# ---------------------------------------------------------------------------

def _coupled(predictor: str, acceleration: str, **group):
    def build():
        gm = GraphManager()
        gm.add_node(BallNode("ball", 0.01, initial_position=1.0, gravity=-3.0))
        gm.add_node(SpringDamperNode("spring", 0.01, stiffness=20.0, rest_length=0.5,
                                     initial_position=0.2, damping=0.5))
        gm.add_edge("ball", "spring", "position", "anchor_position")
        gm.add_edge("spring", "ball", "position", "table_position")
        gm.add_coupling_group(["ball", "spring"], predictor=predictor,
                              acceleration=acceleration, max_iterations=5,
                              tolerance=1e-7, **group)
        gm.compile()
        return gm
    return build


def _two_springs(**group):
    """Two springs anchored to each other: a coupling strong enough that an
    IQN-IMVJ group stores a secant history (``_V``) every step."""
    def build():
        gm = GraphManager()
        gm.add_node(SpringDamperNode("a", 0.01, stiffness=40.0, rest_length=0.5,
                                     initial_position=0.2, damping=0.5))
        gm.add_node(SpringDamperNode("b", 0.01, stiffness=30.0, rest_length=0.3,
                                     initial_position=-0.4, damping=0.2))
        gm.add_edge("a", "b", "position", "anchor_position")
        gm.add_edge("b", "a", "position", "anchor_position")
        gm.add_coupling_group(["a", "b"], max_iterations=6, tolerance=1e-8, **group)
        gm.compile()
        return gm
    return build


def _two_rods(subcycling: bool, **group):
    """Two rods end to end, the right one at twice the left's timestep."""
    def build():
        gm = GraphManager()
        gm.add_node(HeatNode("left", 0.01, n_cells=6, thermal_diffusivity=0.01,
                             initial_temperature=np.linspace(1.0, 2.0, 6).tolist()))
        gm.add_node(HeatNode("right", 0.02, n_cells=6, thermal_diffusivity=0.01,
                             initial_temperature=0.5))
        gm.add_edge("left", "right", "temperature", "left_temperature",
                    transform="extract_last")
        gm.add_edge("right", "left", "temperature", "right_temperature",
                    transform="extract_first")
        if subcycling:
            gm.add_coupling_group(["left", "right"], subcycling=True, max_iterations=4,
                                  **group)
        gm.compile()
        return gm
    return build


def _multirate_chain():
    """A spring at 0.01 driven by a ball at 0.04: rate dividers 1 and 4, so
    the sub-step phase in ``_meta`` decides when the ball moves."""
    def build():
        gm = GraphManager()
        gm.add_node(BallNode("ball", 0.04, initial_position=2.0, gravity=-5.0))
        gm.add_node(SpringDamperNode("spring", 0.01, stiffness=10.0, rest_length=0.3))
        gm.add_edge("ball", "spring", "position", "anchor_position")
        gm.compile()
        return gm
    return build


META_GRAPHS = {
    "linear-predictor-aitken": _coupled("linear", "aitken"),
    "quadratic-predictor-iqn-imvj-reuse": _two_springs(predictor="quadratic",
                                                        acceleration="iqn-imvj",
                                                        jacobian_reuse=2),
    "iqn-ils": _coupled("none", "iqn-ils"),
    "subcycling-waveform": _two_rods(True, waveform_iterations=2,
                                     boundary_interpolation="quadratic", predictor="linear"),
    "multirate-staggered": _two_rods(False),
    "multirate-chain": _multirate_chain(),
}

#: Families whose compile alone is over the per-push budget: a group with
#: ``diagnostics=True`` adds the spectral estimates (about 2.5 s a compile
#: on 3 cores).  Run by the slow-lane sibling of the property below.
SLOW_META_GRAPHS = {
    "iqn-ils-diagnostics": _coupled("none", "iqn-ils", diagnostics=True),
}


_BUILT: dict[str, tuple[GraphManager, GraphManager, dict]] = {}


def _family(graph: str) -> tuple[GraphManager, GraphManager, dict]:
    """Two graphs of a ``META_GRAPHS`` family, built and compiled once per
    module and reset before every use, and the family's pristine state.

    Reusing them is what keeps these properties inside the per-push
    budget (a coupled compile is about 0.4 s, four per example before),
    and it is sound because :func:`check_checkpoint` itself checks that a
    reset graph holds the pristine state bit for bit."""
    if graph not in _BUILT:
        build = {**META_GRAPHS, **SLOW_META_GRAPHS}[graph]
        original = build()
        _BUILT[graph] = (original, build(), full_state(original))
    original, resumed, pristine = _BUILT[graph]
    original.reset_state()
    resumed.reset_state()
    return original, resumed, pristine


@pytest.mark.parametrize("graph", sorted(META_GRAPHS))
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(split=st.integers(min_value=1, max_value=9),
       ahead=st.integers(min_value=0, max_value=3))
def test_a_checkpoint_of_a_graph_with_meta_slots_resumes_the_uninterrupted_run(
        graph, split, ahead):
    """Every split point, including the ones between a slow node's updates."""
    original, resumed, pristine = _family(graph)
    check_checkpoint(original, resumed, pristine, split=split, ahead=ahead)


@pytest.mark.parametrize("graph", sorted(META_GRAPHS))
def test_a_checkpoint_loads_into_a_freshly_built_graph_of_each_meta_family(graph):
    """The plain resume -- into a graph built and compiled and never
    stepped -- at a split between a slow node's updates."""
    original, _, pristine = _family(graph)
    check_checkpoint(original, META_GRAPHS[graph](), pristine, split=5, ahead=0)


# Per push: tests/property/test_differential_checkpoint.py::test_a_checkpoint_of_a_graph_with_meta_slots_resumes_the_uninterrupted_run
@pytest.mark.slow  # a diagnostics group's compile: ~2.5 s, two per family
@pytest.mark.parametrize("graph", sorted(SLOW_META_GRAPHS))
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(split=st.integers(min_value=1, max_value=9),
       ahead=st.integers(min_value=0, max_value=3))
def test_a_checkpoint_of_a_diagnostics_group_resumes_the_uninterrupted_run(
        graph, split, ahead):
    """The spectral-estimate and gradient-bound slots a diagnostics group
    keeps in ``_meta``."""
    original, resumed, pristine = _family(graph)
    check_checkpoint(original, resumed, pristine, split=split, ahead=ahead)


def test_every_meta_graph_carries_the_slots_it_is_for():
    """The fixtures can express the defect: each one's ``_meta`` holds the
    slots a dropped key would lose, with values that have moved by step 5."""
    expected = {
        "linear-predictor-aitken": ("pred_0", "pred_count"),
        # ``_W`` is seeded and carried but did not move in five steps of
        # any fixture tried; it is compared by the restored-state check all
        # the same, but a dropped ``_W`` would not change these numbers.
        "quadratic-predictor-iqn-imvj-reuse": ("pred_2", "_V"),
        "iqn-ils": ("iterations",),
        "subcycling-waveform": ("pred_0",),
        "multirate-staggered": ("step_count",),
        "multirate-chain": ("step_count",),
    }
    for graph, slots in expected.items():
        gm = META_GRAPHS[graph]()
        seeded = full_state(gm).get("_meta", {})
        gm.run(5)
        meta = full_state(gm).get("_meta", {})
        for slot in slots:
            keys = [k for k in meta if k.endswith(slot)]
            assert keys, f"{graph}: no _meta slot ending {slot!r} in {sorted(meta)}"
            assert any(meta[k].tobytes() != seeded[k].tobytes() for k in keys), (
                f"{graph}: {keys} never moved, so dropping it could not be seen")


# ---------------------------------------------------------------------------
# Per push: the REST routes, under the checkpoint root
# ---------------------------------------------------------------------------

def _client(gm, root):
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm, checkpoint_root=root)
    return TestClient(server.create_app(), raise_server_exceptions=False)


@pytest.mark.parametrize("graph", ["quadratic-predictor-iqn-imvj-reuse", "multirate-chain"])
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(split=st.integers(min_value=1, max_value=6),
       saver=st.sampled_from(["rest", "python"]),
       loader=st.sampled_from(["rest", "python"]),
       name=st.sampled_from(["checkpoint.npz", "nested/dir/run1.npz", "plain"]))
def test_a_rest_checkpoint_is_the_in_process_checkpoint(graph, split, saver, loader, name):
    """``POST /checkpoint/save`` / ``load`` under the root, crossed with
    ``save_state`` / ``load_state``: either pair resumes the same run."""
    original, resumed, _ = _family(graph)
    original.run(split)
    with tmp_dir() as root:
        if saver == "rest":
            resp = _client(original, root).post("/checkpoint/save", params={"path": name})
            assert resp.status_code == 200, resp.text
            path = Path(resp.json()["path"])
            assert Path(root).resolve() in path.resolve().parents
        else:
            # ``save_state`` does not create directories (the route does).
            (Path(root) / name).parent.mkdir(parents=True, exist_ok=True)
            path = original.save_state(Path(root) / name)
        resumed.run(1)
        if loader == "rest":
            rel = str(path.resolve().relative_to(Path(root).resolve()))
            resp = _client(resumed, root).post("/checkpoint/load", params={"path": rel})
            assert resp.status_code == 200, resp.text
        else:
            resumed.load_state(path)
    assert_trees_identical(full_state(original), full_state(resumed), what="restored state")
    assert_trees_identical(rollout(original, N_STEPS), rollout(resumed, N_STEPS),
                           what="continued trajectory")


@pytest.mark.parametrize("name", ["../escape.npz", "/etc/passwd", "a/../../b.npz"])
def test_a_checkpoint_path_outside_the_root_is_refused_by_both_routes(name):
    gm = META_GRAPHS["multirate-chain"]()
    gm.run(2)
    before = full_state(gm)
    with tmp_dir() as root:
        client = _client(gm, root)
        for route in ("/checkpoint/save", "/checkpoint/load"):
            resp = client.post(route, params={"path": name})
            assert resp.status_code == 400, (route, resp.text)
    assert_trees_identical(before, full_state(gm), what="state after refused routes")


# ---------------------------------------------------------------------------
# Known, loud: a checkpoint carrying a value the node cannot read
# ---------------------------------------------------------------------------

def test_a_checkpoint_of_a_param_the_node_cannot_read_makes_the_next_run_refuse():
    """Documented and loud, so a pass: ``load_state`` restores the leaf,
    and the next ``run_scan`` refuses it rather than computing with the
    constructor's value while reporting the checkpoint's."""
    gm = META_GRAPHS["multirate-chain"]()
    with tmp_dir() as tmp:
        # A donor built with another initial position: its own leaf matches
        # its own node, so it saves; this graph's node holds 0.0.
        donor = GraphManager()
        donor.add_node(BallNode("ball", 0.04, initial_position=2.0, gravity=-5.0))
        donor.add_node(SpringDamperNode("spring", 0.01, stiffness=10.0, rest_length=0.3,
                                        initial_position=0.75))
        donor.add_edge("ball", "spring", "position", "anchor_position")
        donor.compile()
        path = donor.save_state(checkpoint_path(tmp))
        gm.load_state(path)
    assert float(gm.params["nodes"]["spring"]["initial_position"]) == 0.75
    with pytest.raises(ValueError, match="differs from the node's own value"):
        gm.run_scan(2)
    with pytest.raises(ValueError, match="differs from the node's own value"):
        gm.to_dict()


# ---------------------------------------------------------------------------
# Slow lane
# ---------------------------------------------------------------------------

# Per push: tests/property/test_differential_checkpoint.py::test_a_checkpoint_of_a_built_in_node_resumes_the_uninterrupted_run
@pytest.mark.slow  # an LBM / pipe / wavelet compile per example: 2-6 s each
@pytest.mark.parametrize("kind_name", COSTLY_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_checkpoint_of_a_costly_node_resumes_the_uninterrupted_run(kind_name, data):
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    split = data.draw(st.integers(min_value=0, max_value=3), label="split")
    check_fresh_family(lambda: kind.graph(kwargs), split=split,
                       ahead=data.draw(st.sampled_from([0, 2]), label="ahead"))


# Per push: tests/property/test_differential_checkpoint.py::test_a_checkpoint_of_a_graph_with_meta_slots_resumes_the_uninterrupted_run
@pytest.mark.slow  # a generated multi-node graph built and compiled per example
@settings(max_examples=EXAMPLES_STANDARD, derandomize=True)
@given(data=st.data())
def test_a_checkpoint_of_a_generated_graph_resumes_the_uninterrupted_run(data):
    """Every node kind ``strategies`` can build, coupling groups over every
    valid combination of their settings, transforms, mappings, ParamSpec
    overrides and calibrated leaves."""
    from tests.property.strategies import ALL_NODE_KINDS, graph_recipes

    recipe = data.draw(graph_recipes(kinds=ALL_NODE_KINDS), label="recipe")
    split = data.draw(st.integers(min_value=0, max_value=4), label="split")
    note(f"recipe: {recipe}")
    check_fresh_family(recipe.build, split=split,
                       ahead=data.draw(st.sampled_from([0, 2]), label="ahead"))
