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
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest
from fastapi.testclient import TestClient

import maddening
from maddening.api.server import SimulationServer
from maddening.cloud.multigpu import sharded_node as sn_mod
from maddening.cloud.multigpu.device_mesh import create_device_mesh
from maddening.cloud.multigpu.halo_unstructured import build_unstructured_partition
from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode, ShardedStencilNode
from maddening.cloud.multigpu.sharded_unstructured import ShardedUnstructuredNode
from maddening.core import graph_manager as gm_mod
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode, _method_accepts_params, _signature_takes_params
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


PROBES = {
    "update": {
        "SimulationNode.accepts_params": lambda kind, s: (
            _make(kind, s).accepts_params() if kind == "node" else None),
        "_method_accepts_params": _node_probe(lambda n: _method_accepts_params(n, "update")),
        "_signature_takes_params": _node_probe(lambda n: _signature_takes_params(n.update)),
        "graph _update_accepts_params": _node_probe(gm_mod._update_accepts_params),
        "graph add_node spec": _graph_spec("accepts_params"),
        "graph nodes_without_params": lambda kind, s: (
            lambda gm: (gm.add_node(_make(kind, s)), "n" not in gm.nodes_without_params())[1]
        )(GraphManager()),
        "sharded_node _accepts_params": _node_probe(sn_mod._accepts_params),
        "ShardedPointwiseNode.accepts_params": lambda kind, s: ShardedPointwiseNode(
            _make(kind, s, halo=False), _MESH).accepts_params(),
        "HybridNode.accepts_params": _node_probe(
            lambda n: HybridNode(n, lambda *a: {}).accepts_params()),
        "HybridNode forwards": _hybrid_forwards("update"),
        # Through a wrapper that answers for what it wraps: a probe that
        # read the wrapper's own signature (explicit ``params``) instead of
        # asking it would say True for a legacy inner node.
        "HybridNode(ShardedPointwiseNode).accepts_params": lambda kind, s: HybridNode(
            ShardedPointwiseNode(_make(kind, s, halo=False), _MESH), lambda *a: {},
        ).accepts_params(),
        "graph add_node spec of ShardedPointwiseNode": lambda kind, s: (
            lambda gm: (gm.add_node(ShardedPointwiseNode(_make(kind, s, halo=False), _MESH)),
                        gm._nodes["n"].accepts_params)[1]
        )(GraphManager()),
        "verification _node_accepts_params": _node_probe(ver_mod._node_accepts_params),
        "REST pre-compile probe": _server_treats_as_params_node,
    },
    "update_padded": {
        "_method_accepts_params": _node_probe(lambda n: _method_accepts_params(n, "update_padded")),
        "_signature_takes_params": _node_probe(lambda n: _signature_takes_params(n.update_padded)),
        "ShardedStencilNode.accepts_params": _node_probe(
            lambda n: ShardedStencilNode(n, _MESH, {"devices": 0}).accepts_params()),
        # Built without a Cartesian halo: ShardedUnstructuredNode refuses a
        # node that declares one (it hands update_padded the partition
        # layout, not [halo | interior | halo]).
        "ShardedUnstructuredNode.accepts_params": lambda kind, s: ShardedUnstructuredNode(
            _make(kind, s, halo=False), _MESH, _LAYOUT).accepts_params(),
        "ShardedUnstructuredNode params_pytree": lambda kind, s: bool(ShardedUnstructuredNode(
            _make(kind, s, halo=False), _MESH, _LAYOUT).params_pytree()),
    },
    "compute_boundary_fluxes": {
        "_method_accepts_params": _node_probe(
            lambda n: _method_accepts_params(n, "compute_boundary_fluxes")),
        "graph _flux_accepts_params": _node_probe(gm_mod._flux_accepts_params),
        "graph add_node spec": _graph_spec("flux_accepts_params"),
        "verification _flux_accepts_params": _node_probe(ver_mod._flux_accepts_params),
        "HybridNode forwards": _hybrid_forwards("compute_boundary_fluxes"),
    },
    "compute_interface_correction": {
        "_method_accepts_params": _node_probe(
            lambda n: _method_accepts_params(n, "compute_interface_correction")),
        "graph _correction_accepts_params": _node_probe(gm_mod._correction_accepts_params),
        "HybridNode forwards": _hybrid_forwards("compute_interface_correction"),
    },
}


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("kind, spelling", SPELLINGS, ids=[f"{k}-{s}" for k, s in SPELLINGS])
def test_every_params_probe_agrees_with_what_the_hook_receives(kind, spelling, method):
    truth = _delivers(_make(kind, spelling), method)
    answers = {name: probe(kind, spelling) for name, probe in PROBES[method].items()}
    answers = {name: a for name, a in answers.items() if a is not None}
    wrong = {name: a for name, a in answers.items() if bool(a) is not truth}
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
_ALLOWED = {("core/node.py", "_signature_takes_keyword")}


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
    ``core/node.py::_signature_takes_keyword``."""
    reads = _signature_reads()
    assert set(reads) <= _ALLOWED, sorted(set(reads) - _ALLOWED)
    assert ("core/node.py", "_signature_takes_keyword") in reads


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
