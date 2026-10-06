"""Every "does this take ``params``" probe in ``src/`` gives one answer.

The params contract has one rule -- an explicit ``params`` keyword *or* a
``**kwargs`` that would forward it -- with a callable half
(:func:`maddening.core.node._signature_takes_params`) and a node half
(:func:`maddening.core.node._method_accepts_params`, which asks the node's
own ``accepts_params`` probe first).  Until 0.4.0 shipped there were three
rules in the tree:

* explicit keyword or ``**kwargs``: ``accepts_params``, the implicit
  binder, the graph's hook probes, ``ShardedStencilNode``;
* explicit keyword only: ``ShardedUnstructuredNode`` (both of its
  ``update_padded`` probes) and the duck-typed fallbacks of the graph's
  ``_update_accepts_params`` and of ``sharded_node._accepts_params``;
* always ``False`` for a node object without ``accepts_params``:
  ``verify_node`` and the REST server's pre-compile probe.

What a user saw: an inner node with ``update_padded(**kwargs)`` was
calibratable under ``ShardedStencilNode``, but under
``ShardedUnstructuredNode`` it silently left ``gm.params`` and
``step(params=)`` failed with a false "takes no 'params' keyword"; a
duck-typed node with an explicit ``params`` keyword was injected by the
graph while ``verify_node`` reported its params checks as ``SKIP``, which
counts as passed.

This module runs every probe against one matrix of spellings and asserts
that each probe agrees with the ground truth (does calling the hook with
``params=X`` actually deliver ``X``?), and pins with a source scan that no
module outside ``core/node.py`` inspects a signature on its own -- the
way the second and third rules crept in.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ast
import functools
import inspect
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest
from tests._loopback_client import LoopbackTestClient as TestClient

import maddening
from maddening.api.server import SimulationServer
from maddening.cloud.multigpu import sharded_node as sn_mod
from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode, ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core import graph_manager as gm_mod
from maddening.core.graph_manager import GraphManager
from maddening.core.node import (
    SimulationNode,
    _method_accepts_params,
    _named_where_no_keyword_reaches,
    _signature_required_arguments,
    _signature_takes_keyword,
    _signature_takes_params,
)
from maddening.core.simulation.hybrid_node import HybridNode
from maddening.testing import verification as ver_mod

N = 4
STATE = {"x": jnp.ones(N, jnp.float32)}
METHODS = ("update", "update_padded", "compute_boundary_fluxes", "compute_interface_correction")
_OUTPUT = {
    "update": lambda s, *_: dict(s),
    "update_padded": lambda s, *_: dict(s),
    "compute_boundary_fluxes": lambda *_: {},
    "compute_interface_correction": lambda *_: {},
}


def _impl(node, method, *args, params=None):
    """The body every spelling reaches: record what arrived."""
    node.received[method] = params
    return _OUTPUT[method](*args)


# ------------------------------------------------------------------
# Spellings.  Class-level ones build a method; instance-level ones
# install a callable as an instance attribute over a class method that
# must never be reached.
# ------------------------------------------------------------------

def _explicit(method):
    def hook(self, *args, params=None):
        return _impl(self, method, *args, params=params)
    return hook


def _var_keyword(method):
    def hook(self, *args, **kwargs):
        return _impl(self, method, *args, **kwargs)
    return hook


def _legacy(method):
    def hook(self, *args):
        return _impl(self, method, *args)
    return hook


def _positional_only_params(method):
    """Names ``params``, where no keyword reaches it."""
    def hook(self, a, b, c, params=None, /):
        return _impl(self, method, a, b, c, params=params)
    return hook


def _positional_only_required(method):
    """Positional-only and without a default."""
    def hook(self, a, b, c, params, /):
        return _impl(self, method, a, b, c, params=params)
    return hook


def _var_positional_named_params(method):
    """``*params``: the name of the rest of the positional arguments."""
    def hook(self, *params):
        return _impl(self, method, *params)
    return hook


def _positional_only_beside_var_keyword(method):
    """A positional-only ``params`` beside ``**kwargs``: the keyword is
    delivered, through ``**kwargs``."""
    def hook(self, a, b, c, params=None, /, **kwargs):
        return _impl(self, method, a, b, c, **kwargs)
    return hook


def _no_wraps(fn):
    """A decorator that forgets ``functools.wraps``."""
    def inner(*args, **kwargs):
        return fn(*args, **kwargs)
    return inner


def _unreachable(method):
    def hook(self, *args):
        raise AssertionError(f"class-level {method} reached past the instance attribute")
    return hook


class _Delegate:
    """Owner of the bound method a node borrows."""

    def __init__(self, node, method):
        self.node, self.method = node, method

    def hook(self, *args, params=None):
        return _impl(self.node, self.method, *args, params=params)


class _CallObj:
    def __init__(self, node, method):
        self.node, self.method = node, method

    def __call__(self, *args, params=None):
        return _impl(self.node, self.method, *args, params=params)


_CLASS_SPELLINGS = {
    "explicit": _explicit,
    "var_keyword": _var_keyword,
    "no_wraps": lambda m: _no_wraps(_explicit(m)),
    "legacy": _legacy,
    "positional_only": _positional_only_params,
    "var_positional_named_params": _var_positional_named_params,
    "positional_only_beside_var_keyword": _positional_only_beside_var_keyword,
}
_INSTANCE_SPELLINGS = {
    "partial": lambda node, m: functools.partial(_impl, node, m),
    "bound_other": lambda node, m: _Delegate(node, m).hook,
    "callable_object": lambda node, m: _CallObj(node, m),
}


class _Base(SimulationNode):
    def __init__(self, name="n", halo=True):
        super().__init__(name, 0.1, k=2.0)
        self.received = {}
        self._halo = halo

    def initial_state(self):
        return dict(STATE)

    def halo_width(self):
        return {0: 1} if self._halo else {}


class _Duck:
    """Not a ``SimulationNode``: everything the graph and the wrappers read,
    and no ``accepts_params``."""

    def __init__(self, name="n", halo=True):
        self.name, self.delta_t, self.params = name, 0.1, {"k": 2.0}
        self.received = {}
        self._halo = halo

    def initial_state(self):
        return dict(STATE)

    def halo_width(self):
        return {0: 1} if self._halo else {}

    def params_pytree(self):
        return {"k": jnp.asarray(self.params["k"], jnp.float32)}

    def param_specs(self):
        return {}

    def state_fields(self):
        return list(STATE)

    def boundary_input_spec(self):
        return {}

    def domain_integral_fields(self):
        return set()

    static_data: dict = {}

    def static_data_hash(self):
        return 0


SPELLINGS = (
    [("node", s) for s in (*_CLASS_SPELLINGS, *_INSTANCE_SPELLINGS)]
    + [("duck", s) for s in (*_CLASS_SPELLINGS, *_INSTANCE_SPELLINGS)]
)


def _make(kind, spelling, *, halo=True):
    base = _Base if kind == "node" else _Duck
    if spelling in _CLASS_SPELLINGS:
        build = _CLASS_SPELLINGS[spelling]
        cls = type(f"{kind}_{spelling}", (base,), {m: build(m) for m in METHODS})
        return cls(halo=halo)
    cls = type(f"{kind}_{spelling}", (base,), {m: _unreachable(m) for m in METHODS})
    node = cls(halo=halo)
    for m in METHODS:
        setattr(node, m, _INSTANCE_SPELLINGS[spelling](node, m))
    return node


def _delivers(node, method) -> bool:
    """Ground truth: does ``node.<method>(..., params=X)`` deliver ``X``?"""
    sentinel = {"k": jnp.asarray(5.0, jnp.float32)}
    node.received.clear()
    try:
        getattr(node, method)(dict(STATE), {}, 0.1, params=sentinel)
    except TypeError:
        return False
    return node.received.get(method) is sentinel


_MESH = create_device_mesh(shape=(1,))
_LAYOUT = build_unstructured_partition(
    partition_assignment=np.zeros(N, np.int32),
    edges=np.array([[i, (i + 1) % N] for i in range(N)], np.int32),
    n_devices=1,
)


def _hybrid_forwards(method):
    """Does ``HybridNode`` hand ``params`` on to its physics node's hook?"""
    def probe(kind, spelling):
        node = _make(kind, spelling)
        hybrid = HybridNode(node, lambda *a: {})
        sentinel = {"k": jnp.asarray(7.0, jnp.float32)}
        node.received.clear()
        getattr(hybrid, method)(dict(STATE), {}, 0.1, params=sentinel)
        return node.received.get(method) is sentinel
    return probe


def _graph_spec(attr):
    def probe(kind, spelling):
        gm = GraphManager()
        gm.add_node(_make(kind, spelling))
        return getattr(gm._nodes["n"], attr)
    return probe


def _server_treats_as_params_node(kind, spelling):
    """``PUT /graph/params`` before the first compile validates a params
    node's leaf against its pytree (a wrong shape is a 400) and writes a
    structural key straight through otherwise."""
    gm = GraphManager()
    gm.add_node(_make(kind, spelling))
    client = TestClient(SimulationServer(node_registry={}, graph_manager=gm).create_app(),
                        raise_server_exceptions=False)
    resp = client.put("/graph/params/n", json={"params": {"k": [1.0, 2.0]}})
    assert resp.status_code in (200, 400), resp.text
    # The shape refusal is the pytree validation this probe asks about; a
    # structural write can also be a 400 for a value the node never reads.
    return resp.status_code == 400 and "expected shape" in resp.json()["detail"]


def _node_probe(fn):
    return lambda kind, spelling: fn(_make(kind, spelling))


class _Door:
    """A probe that hands the node to a graph or to a wrapper, where the
    others only ask.  A hook that names ``params`` where no keyword
    reaches it is refused at every door, and answered ``False`` by every
    probe that only asks."""

    def __init__(self, probe):
        self.probe = probe

    def __call__(self, kind, spelling):
        return self.probe(kind, spelling)


#: The spellings of ``_CLASS_SPELLINGS`` a door refuses.
_NO_KEYWORD_REACHES = ("positional_only", "var_positional_named_params")
_REFUSED = "refused: names `params` where no keyword reaches it"


def _ask(probe, kind, spelling):
    """The probe's answer, or ``_REFUSED`` for the refusal by name."""
    try:
        return probe(kind, spelling)
    except ValueError as exc:
        if "names `params`" not in str(exc) or "where no keyword reaches it" not in str(exc):
            raise
        return _REFUSED


PROBES = {
    "update": {
        "SimulationNode.accepts_params": lambda kind, s: (
            _make(kind, s).accepts_params() if kind == "node" else None),
        "_method_accepts_params": _node_probe(lambda n: _method_accepts_params(n, "update")),
        "_signature_takes_params": _node_probe(lambda n: _signature_takes_params(n.update)),
        "graph _update_accepts_params": _node_probe(gm_mod._update_accepts_params),
        "graph add_node spec": _Door(_graph_spec("accepts_params")),
        "graph nodes_without_params": _Door(lambda kind, s: (
            lambda gm: (gm.add_node(_make(kind, s)), "n" not in gm.nodes_without_params())[1]
        )(GraphManager())),
        "sharded_node _accepts_params": _node_probe(sn_mod._accepts_params),
        "ShardedPointwiseNode.accepts_params": _Door(lambda kind, s: ShardedPointwiseNode(
            _make(kind, s, halo=False), _MESH).accepts_params()),
        "HybridNode.accepts_params": _Door(_node_probe(
            lambda n: HybridNode(n, lambda *a: {}).accepts_params())),
        "HybridNode forwards": _Door(_hybrid_forwards("update")),
        # Through a wrapper that answers for what it wraps: a probe that
        # read the wrapper's own signature (explicit ``params``) instead of
        # asking it would say True for a legacy inner node.
        "HybridNode(ShardedPointwiseNode).accepts_params": _Door(lambda kind, s: HybridNode(
            ShardedPointwiseNode(_make(kind, s, halo=False), _MESH), lambda *a: {},
        ).accepts_params()),
        "graph add_node spec of ShardedPointwiseNode": _Door(lambda kind, s: (
            lambda gm: (gm.add_node(ShardedPointwiseNode(_make(kind, s, halo=False), _MESH)),
                        gm._nodes["n"].accepts_params)[1]
        )(GraphManager())),
        "verification _node_accepts_params": _node_probe(ver_mod._node_accepts_params),
        "REST pre-compile probe": _Door(_server_treats_as_params_node),
    },
    "update_padded": {
        "_method_accepts_params": _node_probe(lambda n: _method_accepts_params(n, "update_padded")),
        "_signature_takes_params": _node_probe(lambda n: _signature_takes_params(n.update_padded)),
        "ShardedStencilNode.accepts_params": _Door(_node_probe(
            lambda n: ShardedStencilNode(n, _MESH, {"devices": 0}).accepts_params())),
        # Built without a Cartesian halo: ShardedUnstructuredNode refuses a
        # node that declares one (it hands update_padded the partition
        # layout, not [halo | interior | halo]).
        "ShardedUnstructuredNode.accepts_params": _Door(lambda kind, s: ShardedUnstructuredNode(
            _make(kind, s, halo=False), _MESH, _LAYOUT).accepts_params()),
        "ShardedUnstructuredNode params_pytree": _Door(lambda kind, s: bool(
            ShardedUnstructuredNode(_make(kind, s, halo=False), _MESH, _LAYOUT).params_pytree())),
    },
    "compute_boundary_fluxes": {
        "_method_accepts_params": _node_probe(
            lambda n: _method_accepts_params(n, "compute_boundary_fluxes")),
        "graph _flux_accepts_params": _node_probe(gm_mod._flux_accepts_params),
        "graph add_node spec": _Door(_graph_spec("flux_accepts_params")),
        "verification _flux_accepts_params": _node_probe(ver_mod._flux_accepts_params),
        "HybridNode forwards": _Door(_hybrid_forwards("compute_boundary_fluxes")),
    },
    "compute_interface_correction": {
        "_method_accepts_params": _node_probe(
            lambda n: _method_accepts_params(n, "compute_interface_correction")),
        "graph _correction_accepts_params": _node_probe(gm_mod._correction_accepts_params),
        "HybridNode forwards": _Door(_hybrid_forwards("compute_interface_correction")),
    },
}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind, spelling", SPELLINGS, ids=[f"{k}-{s}" for k, s in SPELLINGS])
def test_every_params_probe_agrees_with_what_the_hook_receives(kind, spelling, method):
    truth = _delivers(_make(kind, spelling), method)
    answers = {name: _ask(probe, kind, spelling) for name, probe in PROBES[method].items()}
    answers = {name: a for name, a in answers.items() if a is not None}
    # Every hook of these nodes has the one spelling, so for a ``params``
    # no keyword reaches each door refuses the node -- and nothing else
    # does, for that spelling or any other.
    doors = {name for name, probe in PROBES[method].items() if isinstance(probe, _Door)}
    refused = {name for name, a in answers.items() if a is _REFUSED}
    assert refused == (doors if spelling in _NO_KEYWORD_REACHES else set()), (
        f"{kind} {spelling} {method}(): refused by {sorted(refused)}")
    wrong = {name: a for name, a in answers.items()
             if a is not _REFUSED and bool(a) is not truth}
    assert not wrong, (
        f"{kind} {spelling} {method}(): calling it with params= "
        f"{'delivers' if truth else 'does not deliver'} them, but {wrong}"
    )


def test_the_matrix_exercises_both_answers():
    """A matrix in which every spelling delivers could not tell a probe
    that always says ``True`` from the rule; the legacy spelling is the
    ``False`` row, for a node and for a duck-typed object."""
    truths = {(k, s): _delivers(_make(k, s), "update") for k, s in SPELLINGS}
    assert {v for v in truths.values()} == {True, False}
    assert not truths[("node", "legacy")] and not truths[("duck", "legacy")]


def test_a_params_parameter_no_keyword_reaches_is_not_a_params_keyword():
    """The rows this matrix lacked.  A hook that names ``params``
    positional-only, or as ``*params``, was answered ``True`` by the name
    alone, so the graph passed the keyword and Python raised ``TypeError``
    from inside the first trace.  Calling with ``params=`` does not
    deliver there, and beside ``**kwargs`` it does."""
    truths = {(k, s): _delivers(_make(k, s), "update") for k, s in SPELLINGS}
    for kind in ("node", "duck"):
        assert not truths[(kind, "positional_only")]
        assert not truths[(kind, "var_positional_named_params")]
        assert truths[(kind, "positional_only_beside_var_keyword")]


def _f_positional_only(a, /): ...
def _f_positional_only_default(a=1, /): ...
def _f_var_positional(*a): ...
def _f_var_keyword(**a): ...
def _f_plain(a): ...
def _f_keyword_only(*, a): ...
def _f_other_positional_only_beside_kwargs(b, /, **kwargs): ...
def _f_same_positional_only_beside_kwargs(a=1, /, **kwargs): ...
def _f_absent(b): ...


@pytest.mark.parametrize("fn, call", [
    (_f_positional_only, lambda f: f(a=3)),
    (_f_positional_only_default, lambda f: f(a=3)),
    (_f_var_positional, lambda f: f(a=3)),
    (_f_var_keyword, lambda f: f(a=3)),
    (_f_plain, lambda f: f(a=3)),
    (_f_keyword_only, lambda f: f(a=3)),
    (_f_other_positional_only_beside_kwargs, lambda f: f(0, a=3)),
    (_f_same_positional_only_beside_kwargs, lambda f: f(a=3)),
    (_f_absent, lambda f: f(0, a=3)),
], ids=lambda v: getattr(v, "__name__", ""))
def test_the_keyword_rule_answers_what_a_call_with_the_keyword_does(fn, call):
    """``_signature_takes_keyword(fn, "a")`` against the call itself, for
    every kind of parameter a signature can give the name to."""
    try:
        call(fn)
        accepted = True
    except TypeError:
        accepted = False
    assert _signature_takes_keyword(fn, "a") is accepted
    # ... and the refusal built on it: only for a signature that holds the
    # name, and never where the call with the keyword is accepted.
    holds_the_name = "a" in inspect.signature(fn).parameters
    assert (_named_where_no_keyword_reaches(fn, "a") is not None) is (
        holds_the_name and not accepted)


# ------------------------------------------------------------------
# A hook that names ``params`` where no keyword reaches it is refused
# where its node is handed over.  ``False`` above is the right answer to
# "would the keyword deliver", and the wrong thing to act on: called
# without the value, a node that plainly means to take ``params`` would
# run on its constructor's constants with every write to ``gm.params``
# ignored.  Every release raised ``TypeError`` from the first trace.
# ------------------------------------------------------------------

_UNREACHABLE_FORMS = {
    "positional_only_with_a_default": (_positional_only_params, "as a positional-only parameter"),
    "positional_only_required": (_positional_only_required, "as a positional-only parameter"),
    "var_positional": (_var_positional_named_params, r"as its \*args"),
}
_GRAPH_HOOKS = ("update", "compute_boundary_fluxes", "compute_interface_correction")
_PADDED_HOOKS = ("update_padded", "compute_boundary_fluxes", "compute_interface_correction")
#: door -> (hand the node over, built with a halo?, the hooks it hands ``params`` to)
_HANDOVERS = {
    "GraphManager.add_node": (lambda node: GraphManager().add_node(node), True, _GRAPH_HOOKS),
    "ShardedPointwiseNode": (lambda node: ShardedPointwiseNode(node, _MESH), False, _GRAPH_HOOKS),
    "HybridNode": (lambda node: HybridNode(node, lambda *a: {}), True, _GRAPH_HOOKS),
    "ShardedStencilNode": (
        lambda node: ShardedStencilNode(node, _MESH, {"devices": 0}), True, _PADDED_HOOKS),
    "ShardedUnstructuredNode": (
        lambda node: ShardedUnstructuredNode(node, _MESH, _LAYOUT), False, ("update_padded",)),
}
_HANDOVER_CASES = [(door, hook) for door, (_, _, hooks) in _HANDOVERS.items() for hook in hooks]


def _one_hook(kind, hook, build, *, halo=True):
    """A node whose hooks all take ``params`` by keyword, except *hook*."""
    base = _Base if kind == "node" else _Duck
    methods = {m: _explicit(m) for m in METHODS}
    methods[hook] = build(hook)
    return type(f"{kind}_one_{hook}", (base,), methods)(halo=halo)


@pytest.mark.parametrize("kind", ["node", "duck"])
@pytest.mark.parametrize("form", list(_UNREACHABLE_FORMS))
@pytest.mark.parametrize("door, hook", _HANDOVER_CASES, ids=[f"{d}:{h}" for d, h in _HANDOVER_CASES])
def test_a_hook_naming_params_where_no_keyword_reaches_is_refused_where_its_node_is_handed_over(
        door, hook, form, kind):
    """One hook at a time, the others taking ``params`` by keyword: the
    graph, and each wrapper for the node it wraps, refuses by name -- the
    node, the hook, how it names ``params`` and what to write instead."""
    hand_over, halo, _ = _HANDOVERS[door]
    build, how = _UNREACHABLE_FORMS[form]
    hand_over(_one_hook(kind, hook, _explicit, halo=halo))        # the same node, spelt right
    node = _one_hook(kind, hook, build, halo=halo)
    with pytest.raises(ValueError, match=(
            rf"Node 'n': {type(node).__name__}\.{hook}\(\) names `params` {how}, where no "
            r"keyword reaches it, .*Accept `params` as a keyword argument")):
        hand_over(node)


def test_a_refused_node_leaves_the_graph_as_it_was_and_its_name_free():
    gm = GraphManager()
    with pytest.raises(ValueError, match="names `params` as a positional-only parameter"):
        gm.add_node(_one_hook("node", "update", _positional_only_params, halo=False))
    assert not gm._nodes and "n" not in gm._state
    gm.add_node(_one_hook("node", "update", _explicit, halo=False))
    gm.compile()
    gm.step()
    assert "n" in gm.params["nodes"]


@pytest.mark.parametrize("door", ["GraphManager.add_node", "ShardedPointwiseNode", "HybridNode",
                                  "ShardedStencilNode"])
@pytest.mark.parametrize("hook", ["compute_boundary_fluxes", "compute_interface_correction"])
def test_a_node_without_params_may_name_them_anywhere_on_a_hook_that_is_never_handed_them(
        door, hook):
    """The refusal is for what used to raise, and no wider.  A node whose
    ``update`` takes no ``params`` has no entry in ``gm.params``, so its
    flux and correction hooks are never handed any: one that names
    ``params`` positional-only there ran in every release, and runs."""
    hand_over, halo, hooks = _HANDOVERS[door]
    methods = {m: _legacy(m) for m in METHODS}
    methods[hook] = _positional_only_params(hook)
    node = type("legacy_update", (_Base,), methods)(halo=halo)
    hand_over(node)
    assert not _method_accepts_params(node, hooks[0])
    if door == "GraphManager.add_node":
        gm = GraphManager()
        gm.add_node(node)
        gm.compile()
        gm.step()
        assert gm.nodes_without_params() == ["n"] and "n" not in gm.params.get("nodes", {})


def _static_padded_positional_only(self, padded, boundary_inputs, dt, static_padded=None, /,
                                   *, params=None):
    return dict(padded)


def _shard_info_positional_only(self, padded, boundary_inputs, dt, shard_info=None, /,
                                *, params=None):
    return dict(padded)


@pytest.mark.parametrize("keyword, update_padded", [
    ("static_padded", _static_padded_positional_only),
    ("shard_info", _shard_info_positional_only),
])
def test_the_stencil_wrapper_refuses_its_own_keywords_where_no_keyword_reaches_them(
        keyword, update_padded):
    """``ShardedStencilNode`` reads the same rule for ``static_padded`` and
    ``shard_info``: named positional-only, each was answered "not taken"
    and the inner ``update_padded`` called without it."""
    node = type("inner", (_Base,), {"update": _explicit("update"),
                                    "update_padded": update_padded})()
    with pytest.raises(ValueError, match=(
            rf"Node 'n': inner\.update_padded\(\) names `{keyword}` as a positional-only "
            rf"parameter, .*ShardedStencilNode passes it as `{keyword}=\.\.\.`")):
        ShardedStencilNode(node, _MESH, {"devices": 0})
    ShardedStencilNode(                     # the same node with the keyword reachable
        type("inner", (_Base,), {"update": _explicit("update"),
                                 "update_padded": _explicit("update_padded")})(),
        _MESH, {"devices": 0})


def test_the_battery_fails_a_node_the_graph_refuses_where_it_skips_one_without_params():
    """``SKIP`` counts as passed, so the three params checks may not
    answer it for a node no graph would take."""
    checks = ["params_consistent", "params_gradient_finite", "params_effective"]
    refused = ver_mod.verify_node(
        _one_hook("node", "update", _positional_only_params, halo=False), {"x": (-1.0, 1.0)},
        checks=checks, max_examples=20, derandomize=True)
    for name in checks:
        assert refused[name].status == "FAIL", refused[name]
        assert "names `params` as a positional-only parameter" in refused[name].detail
    legacy = ver_mod.verify_node(
        _make("node", "legacy", halo=False), {"x": (-1.0, 1.0)},
        checks=checks, max_examples=20, derandomize=True)
    assert {legacy[name].status for name in checks} == {"SKIP"}


class _PositionalOnlyOverRest(SimulationNode):
    """Built the way ``POST /graph/nodes`` builds a node."""

    def __init__(self, name, timestep, k=2.0):
        super().__init__(name, timestep, k=k)

    def initial_state(self):
        return dict(STATE)

    def update(self, state, boundary_inputs, dt, params=None, /):
        return dict(state)


def test_the_rest_route_answers_the_refusal_and_adds_nothing():
    """The route's dry run calls ``update`` without ``params``, so the node
    passed it, was added, and every step after answered 400."""
    gm = GraphManager()
    client = TestClient(
        SimulationServer(node_registry={"Odd": _PositionalOnlyOverRest},
                         graph_manager=gm).create_app(),
        raise_server_exceptions=False)
    resp = client.post("/graph/nodes", json={"type": "Odd", "name": "n", "timestep": 0.1})
    assert resp.status_code == 400, resp.text
    assert "_PositionalOnlyOverRest.update() names `params`" in resp.json()["detail"]
    assert "Accept `params` as a keyword argument" in resp.json()["detail"]
    assert not gm._nodes


def test_verify_node_runs_the_params_checks_on_a_duck_typed_params_node():
    """The graph injects a duck-typed node with a ``params`` keyword, so the
    battery must check it rather than report ``SKIP`` (which counts as
    passed).  A duck whose ``update`` ignores the injected value is the
    case the checks exist for."""

    class IgnoringDuck(_Duck):
        def update(self, state, bi, dt, *, params=None):
            return {"x": state["x"] * (1 - dt * self.params["k"])}

    r = ver_mod.verify_node(IgnoringDuck(halo=False), {"x": (-1.0, 1.0)},
                            checks=["params_consistent", "params_effective"],
                            max_examples=20, derandomize=True)
    assert r["params_consistent"].status == "PASS"
    assert r["params_effective"].status == "FAIL", r["params_effective"]


# ------------------------------------------------------------------
# No module inspects a signature on its own
# ------------------------------------------------------------------

_SIGNATURE_READERS = {"signature", "getfullargspec", "getargspec", "getcallargs"}
_ALLOWED = {
    ("core/node.py", "_signature_takes_keyword"),
    # Not a params probe: binds a request's params to a node constructor's
    # signature to apply its defaults, so a class's size estimate sees the
    # arguments the constructor would run with (``maddening.core._size_estimate``).
    # It never decides whether a method takes ``params``.
    ("core/_size_estimate.py", "constructor_arguments"),
    # Not a keyword probe: lists the arguments a call must supply (those
    # without a default), which a yes/no answer about one keyword cannot
    # enumerate.  The mapping registry needs both to check a declaration
    # against its factory, and asks ``_signature_takes_keyword`` whether a
    # declared name is taken; this one never decides that.  It lives in
    # ``core/node.py`` so that one module still holds every signature read
    # behind a keyword decision.
    ("core/node.py", "_signature_required_arguments"),
    # Not a second keyword rule either: it asks ``_signature_takes_keyword``
    # first and answers only where that said ``False`` -- *how* the
    # signature holds the name (positional-only, ``*args``) or that it does
    # not, for the refusal of a hook that names ``params`` where no keyword
    # reaches it.  The test below the matrix holds the two to each other.
    ("core/node.py", "_named_where_no_keyword_reaches"),
}


def _signature_reads():
    root = pathlib.Path(maddening.__file__).parent
    found = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(path.read_text(), filename=str(path))
        stack = []

        class V(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                stack.append(node.name)
                self.generic_visit(node)
                stack.pop()

            visit_AsyncFunctionDef = visit_FunctionDef

            def visit_Call(self, node):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
                if name in _SIGNATURE_READERS:
                    found.append((rel, stack[-1] if stack else "<module>"))
                self.generic_visit(node)

            def visit_Attribute(self, node):
                if node.attr in ("co_varnames", "co_argcount", "__code__"):
                    found.append((rel, stack[-1] if stack else "<module>"))
                self.generic_visit(node)

        V().visit(tree)
    return found


def test_only_the_shared_helper_reads_a_signature():
    """A probe that inspects a signature itself is how the explicit-only
    rule got into ``ShardedUnstructuredNode`` and the duck-typed
    fallbacks.  Every reader goes through
    ``core/node.py::_signature_takes_keyword``; ``_ALLOWED`` says what the
    three other reads are for, and none of them decides whether a callable
    takes a keyword."""
    reads = _signature_reads()
    assert set(reads) <= _ALLOWED, sorted(set(reads) - _ALLOWED)
    assert ("core/node.py", "_signature_takes_keyword") in reads


# What ``_signature_required_arguments`` is asked about.  Each body is
# trivial, so a ``TypeError`` from a call below is a failure to bind.

def _by_name(a, b, c=0):
    return "called"


def _keyword_only(a, *, b, c=0):
    return "called"


def _positional_only(a, b=0, /, *, c):
    return "called"


def _variadic(*args, **kwargs):
    return "called"


def _positional_only_beside_kwargs(a, /, b, **kwargs):
    return "called"


class _RequiringOwner:
    def __init__(self, a, b=0, *, c):
        pass

    def method(self, a, *, b):
        return "called"

    @classmethod
    def build(cls, a, b=0):
        return "called"

    def __call__(self, a, /, b):
        return "called"


_REQUIRED = {
    "by_name": (_by_name, {"a": True, "b": True}),
    "keyword_only": (_keyword_only, {"a": True, "b": True}),
    "positional_only": (_positional_only, {"a": False, "c": True}),
    "variadic": (_variadic, {}),
    "positional_only_beside_kwargs": (_positional_only_beside_kwargs, {"a": False, "b": True}),
    "class": (_RequiringOwner, {"a": True, "c": True}),
    "bound_method": (_RequiringOwner(1, c=2).method, {"a": True, "b": True}),
    "classmethod": (_RequiringOwner.build, {"a": True}),
    "callable_object": (_RequiringOwner(1, c=2), {"a": False, "b": True}),
    "partial_by_keyword": (functools.partial(_by_name, b=1), {"a": True}),
    "partial_by_position": (functools.partial(_by_name, 1), {"b": True}),
}


@pytest.mark.parametrize("spelling", sorted(_REQUIRED))
def test_the_required_arguments_are_the_ones_a_call_cannot_leave_out(spelling):
    """``_signature_required_arguments`` against the call itself: supplying
    exactly what it lists binds (a positional-only one by position), and
    leaving any one of them out does not.  A positional-only argument is
    one no keyword supplies."""
    fn, expected = _REQUIRED[spelling]
    required = _signature_required_arguments(fn)
    assert required == expected
    positional = [name for name, by_keyword in required.items() if not by_keyword]
    named = {name: 0 for name, by_keyword in required.items() if by_keyword}
    fn(*[0] * len(positional), **named)
    for left_out in named:
        with pytest.raises(TypeError):
            fn(*[0] * len(positional), **{k: v for k, v in named.items() if k != left_out})
    if positional:
        with pytest.raises(TypeError):
            fn(*[0] * (len(positional) - 1), **named)
        with pytest.raises(TypeError):
            fn(**dict.fromkeys(required, 0))


def test_an_unreadable_signature_has_no_required_arguments_to_report():
    """``None``, not an empty answer: the caller decides what an unreadable
    signature means, where the keyword rule answers ``False`` (it fails
    closed)."""
    class Opaque:
        __signature__ = "not a signature"

        def __call__(self, **kwargs):
            return "called"

    assert _signature_required_arguments(Opaque()) is None
    assert _signature_required_arguments(3) is None
    assert _signature_takes_keyword(Opaque(), "anything") is False


class _OldStyleProbe(SimulationNode):
    """A third-party node written against the pre-0.4 probe: its
    ``accepts_params(self)`` override has no ``method`` keyword, and its
    ``update`` forwards ``**kwargs`` (so the signature alone says "takes
    params")."""

    def __init__(self, answer, name="n"):
        super().__init__(name, 0.1, k=2.0)
        self._answer = answer

    def accepts_params(self):  # pyright: ignore[reportIncompatibleMethodOverride]
        return self._answer

    def initial_state(self):
        return {"x": jnp.zeros(3, jnp.float32)}

    def update(self, state, boundary_inputs, dt, **kwargs):
        return dict(state)

    def derivatives(self, state, boundary_inputs):
        return {"x": jnp.zeros(3, jnp.float32)}


@pytest.mark.parametrize("answer", [True, False])
def test_an_old_accepts_params_override_is_asked_the_update_question(answer):
    """``_method_accepts_params`` calls ``accepts_params(method=...)``; an
    override without the keyword raises ``TypeError`` there and is then
    asked the ``"update"`` question it was written for -- its answer, not
    the signature of ``update``, which forwards ``**kwargs`` and would say
    ``True`` either way.  Any other method is read off its own signature,
    so the old override cannot switch the integrators' refusal off."""
    node = _OldStyleProbe(answer)
    assert _method_accepts_params(node, "update") is answer
    assert _method_accepts_params(node, "derivatives") is False
    gm = GraphManager()
    gm.add_node(node)
    assert gm._nodes["n"].accepts_params is answer
