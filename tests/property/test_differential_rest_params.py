"""Differential oracle 1: a ``PUT /graph/params`` write is a reload.

``to_dict()`` saves each node as its class and its effective params, and
``from_dict()`` calls the class with them.  So once ``PUT
/graph/params/{node}`` answers 200, the graph a save would carry has to run
what the running graph runs -- and when it answers 4xx, nothing may have been
written.  Stated over **generated** writes to every cheap built-in node
(``node_catalogue.KINDS``): valid values, values on and past their
``ParamSpec`` bounds, non-finite and oversized numbers, wrong types, unknown
keys, initial conditions, structural constructor arguments, cross-parameter
pairs, and a valid key beside a refused one -- sometimes on a graph whose
``gm.params`` a calibration has already moved away from the constructor's
values.

The oracle, for every write:

* **never a 5xx**;
* **4xx** -- every node's ``params``, every ``gm.params`` leaf, the whole
  state (``_meta`` included) and the dirty flag are bit for bit what they
  were, and ``GET`` still serves the old values;
* **200** --

  - the reply's echo equals ``GET``, and ``GET`` equals what the running
    graph computes with (the ``gm.params`` leaf, or ``node.params`` for a
    structural key) and what a save carries (the config's node params);
  - **save after the write**: the config through JSON, rebuilt, with a
    checkpoint of the running graph loaded on top, restores the state bit
    for bit and continues ``N_STEPS`` bit for bit as the running graph does;
  - **after a reset** (``POST /sim/reset`` on the running graph,
    ``reset_state()`` on the reload) the two step bit for bit too -- which
    is where an initial condition the route wrote takes effect;
  - the reload writes the same config as the running graph.

Tolerance: none.  Both sides run the same computation through the same
compiled step shape, and ``run`` re-enters one compiled step, so any
difference is a difference in what was computed.

What it cannot see: a defect shared by both paths -- a node whose
constructor and whose live step agree on a wrong value -- and a write the
generator does not propose.  The node classes and value ranges are
``node_catalogue``'s; the multi-node graphs are ``strategies.graph_recipes``.
"""

from __future__ import annotations

import json
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
from tests._loopback_client import LoopbackTestClient as TestClient
from hypothesis import event, given, settings
from hypothesis import strategies as st

from maddening.api.server import SimulationServer, _jax_to_python
from maddening.core.graph_manager import GraphManager
from maddening.core.params import ParamSpec

from tests.conftest import EXAMPLES_COSTLY, EXAMPLES_STANDARD
from tests.property.differential import (
    note,
    assert_nothing_written,
    assert_trees_identical,
    canonical,
    checkpoint_path,
    full_state,
    graph_snapshot,
    leaves_identical,
    no_cloud_launch,
    reload_from_config,
    rollout,
    tmp_dir,
)
from tests.property.node_catalogue import (
    CHEAP_KINDS,
    COSTLY_KINDS,
    KINDS,
    REGISTRY,
    Write,
    writes,
)

#: Steps run before the write (so the state is not the initial one) and
#: compared after it.
WARM_STEPS = 2
N_STEPS = 3


@pytest.fixture(scope="module", autouse=True)
def _no_cloud():
    with no_cloud_launch():
        yield


def _client(gm: GraphManager, root: str, registry: dict) -> TestClient:
    server = SimulationServer(node_registry=registry, graph_manager=gm,
                              checkpoint_root=root)
    return TestClient(server.create_app(), raise_server_exceptions=False)


def _node_config(config: dict, name: str) -> dict:
    return next(n for n in config["nodes"] if n["name"] == name)


def _same_value(a, b, dtype=None) -> bool:
    """Two JSON values agree; numbers compared in ``dtype`` when given."""
    if dtype is not None:
        try:
            x = np.asarray(a, dtype=dtype)
            y = np.asarray(b, dtype=dtype)
        except (TypeError, ValueError):
            return canonical(a) == canonical(b)
        return leaves_identical(x, y) is None
    return canonical(a) == canonical(b)


def assert_get_echoes_what_runs(gm: GraphManager, name: str, got: dict,
                                config: dict) -> None:
    """``GET /graph/params/{name}`` is what the step computes with and what a
    save carries."""
    node = gm.get_node(name)
    live = gm.params.get("nodes", {}).get(name) or {}
    saved = _node_config(config, name).get("params", {})
    for key, value in got.items():
        if key in live:
            dtype = np.asarray(live[key]).dtype
            assert _same_value(value, np.asarray(live[key]), dtype), (
                f"GET {name}.{key} = {value!r}, the step reads {np.asarray(live[key])!r}")
            if key in saved:
                assert _same_value(value, saved[key], dtype), (
                    f"GET {name}.{key} = {value!r}, a save carries {saved[key]!r}")
        else:
            here = _jax_to_python(node.params.get(key))
            assert canonical(value) == canonical(here), (
                f"GET {name}.{key} = {value!r}, node.params holds {here!r}")
            if key in saved:
                assert canonical(json.loads(json.dumps(value))) == canonical(saved[key]), (
                    f"GET {name}.{key} = {value!r}, a save carries {saved[key]!r}")


def check_rest_write(gm: GraphManager, registry: dict, name: str, write: Write,
                     *, root: str, rest_reset: bool = True) -> str:
    """Drive one write through the real route and hold it to the oracle.

    Returns ``"accepted"`` or ``"refused"``.  ``rest_reset=False`` resets the
    running graph in process instead of through ``POST /sim/reset`` -- the
    call the route makes before it marks the graph dirty -- which saves the
    recompile a route reset costs.
    """
    gm.run(WARM_STEPS)
    client = _client(gm, root, registry)
    before = graph_snapshot(gm)
    get_before = client.get(f"/graph/params/{name}").json()
    # Under this suite's ``filterwarnings = ["error"]``: a warning the route
    # raises on the way to its answer is a 500 here, and so a failure (the
    # overflowing cast it used to make before refusing 1e39 was one).
    resp = client.put(f"/graph/params/{name}", content=write.body(),
                      headers={"content-type": "application/json"})
    note(f"PUT {write.params!r} -> {resp.status_code} {resp.text[:400]}")
    assert resp.status_code < 500, f"{resp.status_code}: {resp.text}"
    if resp.status_code >= 400:
        assert_nothing_written(gm, before, what=f"refused write {write.params!r}")
        assert canonical(client.get(f"/graph/params/{name}").json()) == canonical(get_before)
        return "refused"
    assert resp.status_code == 200, resp.text

    echo = resp.json()["params"]
    got = client.get(f"/graph/params/{name}").json()
    assert canonical(echo) == canonical(got), (echo, got)
    with warnings.catch_warnings():
        # A live mapping weight is not this oracle's business: the config
        # warns about it, and the checkpoint below carries it.
        warnings.filterwarnings("ignore", message=r".*live mapping weights.*")
        config = json.loads(json.dumps(gm.to_dict(), allow_nan=True))
    assert_get_echoes_what_runs(gm, name, got, config)

    # Save after the write: config + checkpoint, rebuilt, continue.
    with tmp_dir() as tmp:
        ckpt = gm.save_state(checkpoint_path(tmp))
        reloaded = reload_from_config(config, registry)
        reloaded.load_state(ckpt)
    assert_trees_identical(full_state(gm), full_state(reloaded), what="restored state")
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what="continued after the write")
    # After a reset: where an initial condition the route wrote takes effect.
    if rest_reset:
        resp = client.post("/sim/reset")
        assert resp.status_code == 200, resp.text
    else:
        gm.reset_state()
    reloaded.reset_state()
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what="after a reset")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*live mapping weights.*")
        # Compared as a file would hold them: ``to_dict`` returns a node's
        # params as given (a tuple stays a tuple), JSON makes them lists.
        assert (json.loads(json.dumps(reloaded.to_dict(), allow_nan=True))
                == json.loads(json.dumps(gm.to_dict(), allow_nan=True)))
    return "accepted"


# ---------------------------------------------------------------------------
# Per push: every cheap built-in node alone, every write category
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind_name", CHEAP_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_rest_param_write_is_refused_whole_or_runs_as_its_reload(kind_name, data):
    """One built-in node, one generated write: refused with nothing written,
    or the saved graph runs what the running graph runs."""
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    write = data.draw(writes(kind, kwargs), label="write")
    event(f"category={write.category}")
    gm = kind.graph(kwargs)
    # Sometimes a calibration has already moved a leaf in gm.params alone,
    # so the live value and the constructor's differ when the request
    # arrives: GET must serve the live one, and a save must carry it.
    calibrate = data.draw(st.sampled_from(sorted(kind.safe) + [None]), label="calibrated")
    if calibrate is not None:
        live = gm.params["nodes"][kind.name]
        lo, hi = kind.safe[calibrate]
        moved = np.clip(np.asarray(live[calibrate]) * np.float32(1.25), lo, hi)
        live[calibrate] = jnp.asarray(moved.astype(np.asarray(live[calibrate]).dtype))
    # ``POST /sim/reset`` marks the graph dirty, and the recompile after it
    # is the largest single cost of an example (a heat rod went past the
    # per-push budget on CI); half the examples reset in process instead,
    # which is the call the route makes before it marks the graph dirty.
    rest_reset = data.draw(st.booleans(), label="reset through the route")
    with tmp_dir() as root:
        outcome = check_rest_write(gm, REGISTRY, kind.name, write, root=root,
                                   rest_reset=rest_reset)
    event(f"{write.category}: {outcome}")
    if write.category in ("non_finite", "oversized", "wrong_type", "unknown", "invalid",
                          "mixed"):
        assert outcome == "refused", f"{write.category} write {write.params!r} was accepted"


def test_an_overflowing_value_is_refused_without_a_warning():
    """Found by the property above (category ``oversized``).  The route cast
    1e39 to the leaf's float32 before refusing it, and NumPy warned
    ("overflow encountered in cast"): a 500 wherever warnings are errors, as
    in this suite, and otherwise a 400 calling a finite value non-finite."""
    from maddening.nodes import BallNode

    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01))
    gm.compile()
    with tmp_dir() as root:
        client = _client(gm, root, REGISTRY)
        before = graph_snapshot(gm)
        resp = client.put("/graph/params/ball", content=json.dumps(
            {"params": {"elasticity": 1e39}}), headers={"content-type": "application/json"})
        assert_nothing_written(gm, before, what="overflowing write")
    assert resp.status_code == 400, resp.text
    assert "does not fit its type float32" in resp.json()["detail"], resp.text


def _diagnostics_pair():
    from maddening.nodes import BallNode, SpringDamperNode

    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0, gravity=-3.0))
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=20.0, rest_length=0.5))
    gm.add_edge("ball", "spring", "position", "anchor_position")
    gm.add_edge("spring", "ball", "position", "table_position")
    gm.add_coupling_group(["ball", "spring"], diagnostics=True, max_iterations=3)
    gm.compile()
    return gm


def test_a_rest_reset_of_a_diagnostics_group_answers_like_the_in_process_reset():
    """Found by the generated-graph property below.  The in-process reset
    works; the route applied it and then failed to encode its own reply
    (``SimulationServer._state_json`` returned ``gm._state`` with ``_meta``,
    whose spectral slots are seeded NaN, to an encoder that refuses NaN)."""
    gm = _diagnostics_pair()
    gm.run(2)
    reference = _diagnostics_pair()
    with tmp_dir() as root:
        client = _client(gm, root, REGISTRY)
        resp = client.post("/sim/reset")
        # The reset happened whatever the status says.
        assert_trees_identical(full_state(reference), full_state(gm), what="reset state")
        assert resp.status_code == 200, resp.text
        assert client.get("/graph/state").status_code == 200


def rods_mapped_by_grid() -> GraphManager:
    """Two uniform rods joined by an RBF mapping whose point sets are
    references to each rod's ``grid_x`` -- the form ``to_dict`` re-resolves
    and hash-checks.  ``grid_x`` is built from ``length`` when a rod is
    constructed.

    The mapping is built from ``grid_x`` *as the rod holds it* (float32): a
    reference's recorded hash covers the dtype, so points widened to float64
    first describe another array, and ``to_dict()`` refused this graph
    before any write -- which is how the pins below used to fail for the
    wrong reason (``test_the_mapped_rods_save_and_reload_before_any_write``).
    """
    from maddening.core.coupling.mapping import rbf_mapping
    from maddening.nodes import HeatNode

    gm = GraphManager()
    a = HeatNode("a", 0.01, n_cells=6, length=1.0, thermal_diffusivity=0.005,
                 initial_temperature=np.linspace(1.0, 2.0, 6).tolist())
    b = HeatNode("b", 0.01, n_cells=5, length=1.0, thermal_diffusivity=0.005,
                 initial_temperature=0.5)
    gm.add_node(a)
    gm.add_node(b)
    gm.add_edge("a", "b", "temperature", "heat_source", mapping=rbf_mapping(
        np.asarray(a.static_data["grid_x"].value),
        np.asarray(b.static_data["grid_x"].value),
        source_ref={"node": "a", "field": "grid_x"},
        target_ref={"node": "b", "field": "grid_x"}))
    gm.compile()
    return gm


def test_the_mapped_rods_save_and_reload_before_any_write():
    """The fixture can express the defect: unwritten, the graph saves, its
    config reloads, and the reload steps bit for bit as the original -- so a
    refusal or a disagreement below is the write's."""
    gm = rods_mapped_by_grid()
    gm.run(WARM_STEPS)
    config = json.loads(json.dumps(gm.to_dict(), allow_nan=True))
    with tmp_dir() as tmp:
        ckpt = gm.save_state(checkpoint_path(tmp))
        reloaded = reload_from_config(config, REGISTRY)
        reloaded.load_state(ckpt)
    assert_trees_identical(rollout(gm, N_STEPS), rollout(reloaded, N_STEPS),
                           what="mapped rods against their reload")


def test_a_rod_length_under_a_mapping_reference_is_refused_or_runs_as_its_reload():
    """Found while widening the generated writes to geometry: ``length`` is a
    live leaf of a uniform rod (its step reads it), so every check the route
    made passed and it answered 200; ``grid_x`` is derived from it at
    construction and never rebuilt, the mapping's ``MappingSpec`` points at
    ``grid_x``, and the running graph kept the old grid's weights while the
    saved config no longer loaded.  Refused now, naming the mapped edge."""
    gm = rods_mapped_by_grid()
    with tmp_dir() as root:
        assert check_rest_write(gm, REGISTRY, "a", Write("geometry", {"length": 1.5}),
                                root=root) == "refused"
        resp = _client(gm, root, REGISTRY).put(
            "/graph/params/a", json={"params": {"length": 1.5}})
    assert resp.status_code == 400, resp.text
    assert "a.temperature->b.heat_source" in resp.json()["detail"]
    assert "grid_x" in resp.json()["detail"]


# The two graph features a one-node graph cannot carry, per push: a
# ParamSpec override (the route must refuse a value outside it, and the
# reload must carry it), and a coupling group with a predictor (a write into
# a member must leave the group's warm starts consistent with the reload's).

def _spring_pair(**spec):
    from maddening.nodes import BallNode, SpringDamperNode

    gm = GraphManager()
    gm.add_node(BallNode("ball", 0.01, initial_position=1.0, gravity=-3.0))
    gm.add_node(SpringDamperNode("spring", 0.01, stiffness=20.0, rest_length=0.5,
                                 initial_position=0.2))
    gm.add_edge("ball", "spring", "position", "anchor_position")
    gm.add_edge("spring", "ball", "position", "table_position")
    gm.add_coupling_group(["ball", "spring"], predictor="linear", max_iterations=4,
                          acceleration="aitken")
    gm.compile()
    if spec:
        gm.set_param_spec("spring", "stiffness", ParamSpec(**spec))
    return gm


#: ``(stiffness, damping)`` writes around the override's bounds ``[1, 100]``:
#: inside, on each bound, just past each, far past, and with a second key.
#: A fixed table rather than a Hypothesis draw: each case builds and compiles
#: a coupled graph and its reload (about 0.4 s), so twenty draws would take a
#: per-push test past the time budget, while these six points are the whole
#: decision the route makes.  The slow lane draws generated coupled graphs
#: (``test_a_rest_param_write_into_a_generated_graph_...``).
_SPEC_CASES = [(37.5, None), (1.0, None), (100.0, 2.5), (100.5, None), (0.5, 1.0),
               (-3.0, None)]


@pytest.mark.parametrize("stiffness, damping", _SPEC_CASES)
def test_a_rest_write_into_a_coupled_member_with_a_param_spec_runs_as_its_reload(
        stiffness, damping):
    """A ``ParamSpec`` override bounds ``stiffness`` to ``[1, 100]``: the route
    refuses outside it, and an accepted write -- into a member of a coupling
    group that carries predictor history in ``_meta`` -- runs as the reload,
    which carries the override."""
    gm = _spring_pair(bounds=(1.0, 100.0))
    params = {"stiffness": stiffness}
    if damping is not None:
        params["damping"] = damping
    with tmp_dir() as root:
        outcome = check_rest_write(gm, REGISTRY, "spring", Write("coupled", params),
                                   root=root)
    assert (outcome == "accepted") == (1.0 <= stiffness <= 100.0), (params, outcome)


# ---------------------------------------------------------------------------
# Slow lane: the costly nodes, and generated multi-node graphs
# ---------------------------------------------------------------------------

# Per push: tests/property/test_differential_rest_params.py::test_a_rest_param_write_is_refused_whole_or_runs_as_its_reload
@pytest.mark.slow  # an LBM / pipe / wavelet compile per example: 2-6 s each
@pytest.mark.parametrize("kind_name", COSTLY_KINDS)
@settings(max_examples=EXAMPLES_COSTLY, derandomize=True)
@given(data=st.data())
def test_a_rest_param_write_to_a_costly_node_is_refused_whole_or_runs_as_its_reload(
        kind_name, data):
    kind = KINDS[kind_name]
    kwargs = data.draw(kind.kwargs, label="kwargs")
    write = data.draw(writes(kind, kwargs), label="write")
    gm = kind.graph(kwargs)
    with tmp_dir() as root:
        outcome = check_rest_write(gm, REGISTRY, kind.name, write, root=root)
    if write.category in ("non_finite", "oversized", "wrong_type", "unknown", "invalid"):
        assert outcome == "refused", f"{write.category} write {write.params!r} was accepted"


#: ``strategies`` node type -> the ``node_catalogue`` family whose write
#: categories apply to it.
_CATALOGUE_OF = {"BallNode": "ball", "TableNode": "table", "SpringDamperNode": "spring",
                 "HeatNode": "heat_uniform", "RigidBody2DNode": "rigid_body_2d",
                 "HeartPumpNode": "heart_pump", "RigidBodyNode": "rigid_body"}


def recipe_kwargs(target) -> dict:
    """A ``strategies.NodeRecipe``'s constructor arguments as the node takes them."""
    kw = dict(target.params)
    if target.type_name == "RigidBodyNode":
        kw["constraints"] = dict(kw.get("constraints") or [])
    return kw


@st.composite
def recipe_and_write(draw):
    """A generated graph, one of its nodes, and a write to it: a perturbable
    leaf scaled the way a calibration would move it, or any of the node's
    catalogue categories."""
    from tests.property.strategies import ALL_NODE_KINDS, graph_recipes

    recipe = draw(graph_recipes(kinds=ALL_NODE_KINDS, max_nodes=3), label="recipe")
    target = draw(st.sampled_from(recipe.nodes), label="target")
    kw = recipe_kwargs(target)
    if target.perturbable and draw(st.booleans()):
        key = draw(st.sampled_from(target.perturbable))
        factor = draw(st.sampled_from([0.5, 0.9, 1.1, 2.0]))
        value = (np.asarray(kw[key], np.float32) * np.float32(factor)).astype(np.float32)
        return recipe, target.name, Write("scaled", {key: value.tolist()})
    kind = KINDS[_CATALOGUE_OF[target.type_name]]
    return recipe, target.name, draw(writes(kind, kw), label="write")


# Per push: tests/property/test_differential_rest_params.py::test_a_rest_param_write_is_refused_whole_or_runs_as_its_reload
@pytest.mark.slow  # a generated multi-node graph built and compiled per example
@settings(max_examples=EXAMPLES_STANDARD, derandomize=True)
@given(case=recipe_and_write())
def test_a_rest_param_write_into_a_generated_graph_is_refused_whole_or_runs_as_its_reload(case):
    """Coupling groups (with predictors, IQN warm starts, subcycling),
    edges with transforms, interface mappings, ParamSpec overrides and
    calibrated leaves around the node written."""
    from tests.property.strategies import NODE_REGISTRY

    recipe, name, write = case
    note(f"recipe: {recipe}")
    gm = recipe.build()
    with tmp_dir() as root:
        # Every reset through the route, a diagnostics group's included.
        check_rest_write(gm, dict(NODE_REGISTRY), name, write, root=root)
