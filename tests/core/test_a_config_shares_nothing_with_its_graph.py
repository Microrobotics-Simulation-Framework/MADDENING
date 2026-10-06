"""A config is a copy: editing what ``to_dict()`` returned edits nothing else.

The natural way to build a variant of a graph is to take its config and
change it::

    cfg = gm.to_dict()
    cfg["nodes"][0]["params"]["stiffness"] = 5000.0
    variant = GraphManager.from_dict(cfg, registry)

``to_dict()`` used to hand out the graph's own containers, so that edit
also wrote into ``gm`` (MADD-ANO-205).  In every release from 0.1.0 to
0.3.1 the config's ``params`` *was* ``node.params``: the original graph ran
the variant's value from its next recompile on (a spring's position 0.9626
against 2.8842 one step later) and wrote it into its own next config, with
nothing said.  On the 0.4.0 tree the same held for a node on the
three-argument ``update`` contract, now from its next *step*; for the lists
and dicts inside any node's params; for the point sets of a mapped edge;
and for a sharded wrapper's ``axis_map``.

It is also why ``yaml.safe_dump(gm.to_dict())`` failed for a node on the
three-argument contract: its ``params`` was the node's own counting
``dict`` subclass, which a safe dumper will not represent.

These tests hold the config to "shares no mutable container with the
graph", by identity and by what an edit does, on every part of a graph
that writes one.
"""

from __future__ import annotations

import collections
import copy
import json
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest
import yaml

from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode, ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core.coupling.mapping import rbf_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode, _ParamsDict
from maddening.core.params import ParamSpec
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.nodes.heat import HeatNode
from maddening.nodes.spring import SpringDamperNode

DT = 0.01  # units: s


class ThreeArgument(SimulationNode):
    """A node on the three-argument contract: it reads ``self.params``."""

    def __init__(self, name, timestep, k=2.0, table=(1.0, 1.0, 1.0), nested=None):
        super().__init__(name, timestep, k=k, table=list(table),
                         nested=nested if nested is not None else {"a": [1, 2]})

    def initial_state(self):
        return {"x": jnp.ones(3, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"] * self.params["k"]
                * jnp.asarray(self.params["table"], jnp.float32)}


class FourArgument(ThreeArgument):
    """The same node on the ``params`` contract."""

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": state["x"] * params["k"]
                * jnp.asarray(self.params["table"], jnp.float32)}


class Cells(SimulationNode):
    """Eight cells with no stencil, for the wrappers."""

    def __init__(self, name="cells", timestep=DT, weights=(1.0, 2.0)):
        super().__init__(name, timestep, weights=list(weights))

    def initial_state(self):
        return {"x": jnp.ones(8, jnp.float32)}

    def update(self, state, boundary_inputs, dt):
        return dict(state)

    def update_padded(self, state_padded, boundary_inputs, dt):
        return dict(state_padded)


class Stencil(Cells):
    def halo_width(self):
        return {0: 1}


REGISTRY = {"ThreeArgument": ThreeArgument, "FourArgument": FourArgument,
            "HeatNode": HeatNode, "SpringDamperNode": SpringDamperNode}


def _rich_graph() -> GraphManager:
    """Every part of a graph that writes into a config: a node on each
    contract with a list and a nested dict among its params, a rod with a
    list initial condition, a mapped edge with inline points, a coupling
    group, an external input and a ``ParamSpec`` override."""
    gm = GraphManager()
    gm.add_node(ThreeArgument("three", DT))
    gm.add_node(FourArgument("four", DT))
    gm.add_node(HeatNode("a", DT, n_cells=6, length=1.0, thermal_diffusivity=0.005,
                         initial_temperature=np.linspace(1.0, 2.0, 6).tolist()))
    gm.add_node(HeatNode("b", DT, n_cells=5, length=1.0, thermal_diffusivity=0.005,
                         initial_temperature=0.5, grid_points=[0.1, 0.3, 0.5, 0.7, 0.9]))
    gm.add_edge("a", "b", "temperature", "heat_source", mapping=rbf_mapping(
        np.linspace(0.05, 0.95, 6), np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])))
    for name, rest, start in (("s1", 1.0, 0.0), ("s2", -1.0, 5.0)):
        gm.add_node(SpringDamperNode(name=name, timestep=DT, stiffness=50.0, damping=5.0,
                                     mass=0.5, rest_length=rest, initial_position=start))
    gm.add_edge("s1", "s2", "position", "anchor_position")
    gm.add_edge("s2", "s1", "position", "anchor_position")
    gm.add_coupling_group(["s1", "s2"], max_iterations=20, tolerance=1e-6)
    gm.add_external_input("three", "drive", shape=(3,), dtype=jnp.float32)
    gm.set_param_spec("s1", "stiffness", ParamSpec(bounds=(1.0, 100.0)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the advisories of a deliberately mixed graph
        gm.compile()
        gm.step()
    return gm


def _config(gm: GraphManager) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return gm.to_dict()


def _containers(value, path="config", found=None) -> dict:
    """``{id: (path, container)}`` for every dict, list and array in *value*."""
    found = {} if found is None else found
    if isinstance(value, dict):
        found[id(value)] = (path, value)
        for key, item in value.items():
            _containers(item, f"{path}[{key!r}]", found)
    elif isinstance(value, (list, tuple)):
        if isinstance(value, list):
            found[id(value)] = (path, value)
        for i, item in enumerate(value):
            _containers(item, f"{path}[{i}]", found)
    elif isinstance(value, np.ndarray):
        found[id(value)] = (path, value)
    return found


@pytest.fixture(scope="module")
def rich():
    return _rich_graph()


def test_two_configs_of_one_graph_share_no_container(rich):
    """Asked twice, the graph hands out two trees.  A container that is the
    same object in both is one the graph kept: its own."""
    first, second = _containers(_config(rich)), _containers(_config(rich))
    shared = sorted(first[i][0] for i in set(first) & set(second))
    assert len(first) > 30, "the fixture's config is smaller than this test assumes"
    assert not shared, shared


def test_no_container_of_a_config_is_one_the_graph_holds(rich):
    """The same by reachability: nothing in the config is a node's params
    mapping, a value inside one, or a point set of a mapping's spec."""
    held = {}
    for name in rich.node_names:
        _containers(rich.get_node(name).params, f"{name}.params", held)
    for edge in rich._edges:
        spec = getattr(edge.mapping, "spec", None)
        if spec is not None:
            _containers(spec.points, f"{edge.key}.spec.points", held)
            _containers(spec.hyperparameters, f"{edge.key}.spec.hyperparameters", held)
    config = _containers(_config(rich))
    both = sorted((config[i][0], held[i][0]) for i in set(config) & set(held))
    assert len(held) >= 8, "the fixture holds fewer containers than this test assumes"
    assert not both, both


def _paths():
    gm = _rich_graph()
    return sorted(path for path, _ in _containers(_config(gm)).values())


@pytest.mark.parametrize("path", _paths())
def test_emptying_any_container_of_a_config_leaves_the_graphs_next_config_alone(rich, path):
    """Each container of the config in turn is emptied in place.  The
    graph's next config is the one it gave before: nothing the caller did
    to a config reached the graph."""
    pristine = copy.deepcopy(_config(rich))
    config = _config(rich)
    target = next(c for p, c in _containers(config).values() if p == path)
    target.clear()
    assert _config(rich) == pristine, path


@pytest.mark.parametrize("contract", [ThreeArgument, FourArgument])
@pytest.mark.parametrize("edit", ["a key", "an element of a list", "a nested list"])
@pytest.mark.parametrize("then", ["step", "recompile and step"])
def test_a_variant_built_from_an_edited_config_does_not_change_the_original(
        contract, edit, then):
    """The use the defect broke.  The variant takes the edited value; the
    graph the config came from steps exactly as one nobody touched -- at
    its next step, and after a recompile, which is where the releases
    picked the edit up."""
    def graph():
        gm = GraphManager()
        gm.add_node(contract("n", DT))
        gm.compile()
        gm.step()
        return gm

    original, untouched = graph(), graph()
    config = original.to_dict()
    params = config["nodes"][0]["params"]
    if edit == "a key":
        params["k"] = 5.0
    elif edit == "an element of a list":
        params["table"][0] = 5.0
    else:
        params["nested"]["a"].append(3)
    variant = GraphManager.from_dict(config, {contract.__name__: contract})
    assert dict(variant.get_node("n").params) != dict(untouched.get_node("n").params)

    assert dict(original.get_node("n").params) == dict(untouched.get_node("n").params)
    for gm in (original, untouched):
        if then == "recompile and step":
            gm.compile()
        gm.step()
    np.testing.assert_array_equal(np.asarray(original.get_node_state("n")["x"]),
                                  np.asarray(untouched.get_node_state("n")["x"]))
    assert original.to_dict() == untouched.to_dict()


def test_the_effective_params_of_a_node_are_a_copy_all_the_way_down(rich):
    """``effective_node_params`` is what the config's ``params`` is built
    from, and what ``GET /graph/params`` serves: a list in it was the
    node's own."""
    for name in ("three", "four"):
        node = rich.get_node(name)
        before = copy.deepcopy(dict(node.params))
        effective = rich.effective_node_params(name)
        assert type(effective) is dict
        effective["table"][0] = 9.0
        effective["nested"]["a"].clear()
        effective["k"] = 9.0
        assert dict(node.params) == before, name


def _wrappers():
    mesh = create_device_mesh(shape=(1,))
    layout = build_unstructured_partition(
        partition_assignment=np.zeros(8, np.int32),
        edges=np.array([[i, (i + 1) % 8] for i in range(8)], np.int32), n_devices=1)
    return {
        "a node": lambda: ThreeArgument("n", DT),
        "ShardedPointwiseNode": lambda: ShardedPointwiseNode(Cells(), mesh),
        "ShardedStencilNode": lambda: ShardedStencilNode(Stencil(), mesh, {"devices": 0}),
        "ShardedUnstructuredNode": lambda: ShardedUnstructuredNode(Cells(), mesh, layout),
        "HybridNode": lambda: HybridNode(ThreeArgument("n", DT), lambda *a: {}),
    }


@pytest.mark.parametrize("kind", sorted(_wrappers()))
def test_a_nodes_own_descriptor_shares_no_container_with_the_node(kind):
    """``node.to_dict()`` called directly, for a node and for each wrapper
    that writes its own: two descriptors share nothing, and emptying every
    container of one changes neither the next descriptor nor the node --
    a sharded wrapper's ``axis_map`` was the wrapper's own dict."""
    node = _wrappers()[kind]()
    pristine = copy.deepcopy(node.to_dict())
    first, second = _containers(node.to_dict()), _containers(node.to_dict())
    assert not set(first) & set(second), sorted(first[i][0] for i in set(first) & set(second))
    assert first, "a descriptor with no container tests nothing"
    for _, container in first.values():
        container.clear()
    assert node.to_dict() == pristine


def test_an_edited_axis_map_does_not_move_the_wrappers_decomposition():
    wrapper = _wrappers()["ShardedStencilNode"]()
    before = dict(wrapper._axis_map)
    wrapper.to_dict()["axis_map"]["devices"] = 7
    assert wrapper._axis_map == before


@pytest.mark.parametrize("contract", [ThreeArgument, FourArgument])
@pytest.mark.parametrize("compiled", [False, True])
def test_a_config_is_plain_data_that_yaml_and_json_write_alike(contract, compiled):
    """``yaml.safe_dump`` refuses a ``dict`` subclass, which a node on the
    three-argument contract used to have for ``params``.  Every container
    of a config is exactly a ``dict`` or a ``list`` (or a tuple), and YAML
    reads back what JSON does."""
    gm = GraphManager()
    gm.add_node(contract("n", DT))
    if compiled:
        gm.compile()
        gm.step()
    config = gm.to_dict()
    odd = sorted(path for path, c in _containers(config).values()
                 if type(c) not in (dict, list))
    assert not odd, odd
    assert yaml.safe_load(yaml.safe_dump(config)) == json.loads(json.dumps(config))
    reloaded = GraphManager.from_dict(yaml.safe_load(yaml.safe_dump(config)),
                                      {contract.__name__: contract})
    assert dict(reloaded.get_node("n").params) == dict(gm.get_node("n").params)


def test_the_rich_config_is_plain_data_and_reloads(rich):
    config = _config(rich)
    odd = sorted(path for path, c in _containers(config).values()
                 if type(c) not in (dict, list))
    assert not odd, odd
    assert yaml.safe_load(yaml.safe_dump(config)) == json.loads(json.dumps(config))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reloaded = GraphManager.from_dict(json.loads(json.dumps(config)), REGISTRY)
        assert reloaded.to_dict() == config


Point = collections.namedtuple("Point", "x y")


def test_the_copy_keeps_values_and_drops_only_what_could_be_written_through():
    """What ``_detached_config`` copies, kind by kind: a ``dict`` of any
    class becomes a ``dict``, a list a new list, a tuple a tuple of
    copies (a named tuple keeps its class), an array a copy; a number, a
    string and a JAX array, which nothing can change in place, pass
    through as they are."""
    # Imported here: the tests above name no helper, so against a tree
    # without one they fail on what its configs do.
    from maddening.core.node import _detached_config  # noqa: PLC0415

    array, jax_array = np.arange(3.0), jnp.arange(3.0)
    inner = [1, {"deep": [2]}]
    source = _ParamsDict(a=inner, t=(inner, 4), p=Point([5], 6), n=array, j=jax_array,
                         s="text", f=1.5, none=None)
    out = _detached_config(source)
    assert type(out) is dict and out == dict(source) | {"n": out["n"]}
    assert np.array_equal(out["n"], array) and out["n"] is not array
    assert out["j"] is jax_array and out["s"] is source["s"]
    assert out["a"] is not inner and out["a"][1] is not inner[1]
    assert out["a"][1]["deep"] is not inner[1]["deep"]
    assert type(out["t"]) is tuple and out["t"][0] is not inner
    assert type(out["p"]) is Point and out["p"].x is not source["p"].x
    out["a"][1]["deep"].append(3)
    out["p"].x.append(7)
    out["n"][0] = 9.0
    assert inner == [1, {"deep": [2]}] and source["p"].x == [5] and array[0] == 0.0
