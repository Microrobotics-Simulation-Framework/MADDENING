"""A node too large to accept is refused before any of it is built.

``POST /graph/nodes`` measured the state of the node it had just built, and
``PUT /graph/params`` checked a write by calling the node's constructor with
the new value.  A constructor that turns a parameter into an array size
therefore allocated it all before any refusal: a D3Q19 ``LBMNode`` of 140^3
cells took 1.1 GB before the state cap refused it, and one ``PUT
{"n_levels": 10000000}`` to a ``WaveletAdaptiveNode`` grew the server to
57.7 GB -- the level labels were a Python list of ``n_coarse * 2**n_levels``
entries -- until the kernel's OOM killer stopped the machine.

The built-in grid and basis nodes now say what they would allocate
(``_allocation_estimate``, read by :mod:`maddening.core._size_estimate`),
and the routes and :meth:`GraphManager.from_dict` ask before any
constructor call.  Every test that sends a size the old code could not
survive guards the constructor so that it records, and refuses, a call: a
regression fails here, it does not allocate.
"""

from __future__ import annotations

import functools
import os
import resource
import subprocess
import sys
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import numpy as np
import pytest
from fastapi.testclient import TestClient

from maddening.api import server as server_module
from maddening.api.server import (
    MAX_NODE_BUILD_BYTES,
    MAX_NODE_STATE_ELEMENTS,
    SimulationServer,
)
from maddening.core import _size_estimate as size_estimate
from maddening.core.graph_manager import GraphManager
from maddening.core.node import SimulationNode
from maddening.nodes import HeatNode, LBMNode, LBMPipeNode
from maddening.nodes.adaptive import WaveletAdaptiveNode
from maddening.nodes.adaptive import wavelet as wavelet_module
from maddening.nodes.adaptive.wavelets import dirichlet as dirichlet_module
from maddening.nodes.adaptive.wavelets import transform as transform_module

REGISTRY = {cls.__name__: cls for cls in (HeatNode, LBMNode, LBMPipeNode,
                                          WaveletAdaptiveNode)}

#: Small configurations of every node class with an estimate, for the
#: exactness check (the estimate must equal the state the node builds).
SMALL = [
    (HeatNode, 0.01, {"n_cells": 8}),
    (HeatNode, 0.01, {"n_cells": 17, "stencil_order": 4}),
    (HeatNode, 0.01, {}),
    (LBMNode, 1.0, {"grid_shape": (8, 8, 8)}),
    (LBMNode, 1.0, {"grid_shape": (9, 7, 5)}),
    (LBMNode, 1.0, {"grid_shape": [6, 5], "lattice": "d2q9"}),
    (LBMPipeNode, 1.0, {"nx": 12, "ny": 8, "nz": 8, "propeller_x": 4}),
    (LBMPipeNode, 1.0, {"nx": 12, "ny": 9, "nz": 7, "propeller_x": 4, "G": -5.0}),
    # One periodic and one Dirichlet basis: each construction compiles the
    # synthesis and estimates a condition number (seconds, not the budget).
    (WaveletAdaptiveNode, 1.0, {"dim": 2, "n_levels": 1, "n_coarse": 3}),
    (WaveletAdaptiveNode, 1.0, {"dim": 1, "n_levels": 2, "boundary": "dirichlet"}),
]

#: Configurations the API must refuse for their size: (class, params, what
#: the refusal names).  Each was measured allocating before it was refused
#: (or never being refused), and none may now reach a constructor.
OVERSIZED = [
    (WaveletAdaptiveNode, {"dim": 1, "n_levels": 26}, "state elements"),
    (WaveletAdaptiveNode, {"dim": 1, "n_levels": 10_000_000}, "state elements"),
    # A float the request model does not bound as an integer, which the
    # constructor's own count check accepts as one.
    (WaveletAdaptiveNode, {"dim": 1, "n_levels": 1e7}, "state elements"),
    (WaveletAdaptiveNode, {"dim": 3, "n_levels": 8, "boundary": "dirichlet"},
     "state elements"),
    # Small state, dense operator: 8192 functions assemble in about 4 GiB.
    (WaveletAdaptiveNode, {"dim": 1, "n_levels": 12}, "to build"),
    (LBMNode, {"grid_shape": [140, 140, 140]}, "state elements"),
    (LBMNode, {"grid_shape": [16000, 16000], "lattice": "D2Q9"}, "state elements"),
    (LBMNode, {"grid_shape": [10_000_000] * 3}, "state elements"),
    (LBMPipeNode, {"nx": 2000, "ny": 200, "nz": 200, "propeller_x": 4},
     "state elements"),
]


def _state_elements(state) -> int:
    return sum(int(np.prod(np.shape(leaf))) for leaf in jax.tree_util.tree_leaves(state))


def _guard_constructor(monkeypatch, cls) -> list:
    """Make *cls*'s constructor record its arguments and raise instead of
    building anything; returns the record.  A refusal that came from the
    constructor would carry its message, not the size refusal's.  The
    guard keeps the constructor's signature (``functools.wraps``), which
    the estimate binds the params against."""
    calls: list = []
    real_init = cls.__init__

    @functools.wraps(real_init)
    def guarded(self, *args, **kwargs):
        calls.append(kwargs)
        raise AssertionError(f"{cls.__name__} was constructed")

    monkeypatch.setattr(cls, "__init__", guarded)
    return calls


def _client(gm=None) -> TestClient:
    server = SimulationServer(node_registry=REGISTRY, graph_manager=gm)
    return TestClient(server.create_app(), raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# The estimate itself
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, dt, params", SMALL,
                         ids=[f"{c.__name__}-{i}" for i, (c, _, _) in enumerate(SMALL)])
def test_the_estimate_is_the_state_the_node_builds(cls, dt, params):
    """What the routes refuse on before building is exactly what
    ``POST /graph/nodes`` measured after building, so moving the check
    earlier changes which requests are refused by nothing."""
    estimate = size_estimate.estimate_allocation(cls, params)
    assert estimate is not None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        node = cls(name="n", timestep=dt, **params)
    assert estimate.state_elements == _state_elements(node.initial_state())
    assert estimate.peak_bytes >= 4 * estimate.state_elements


def test_the_wavelet_estimate_counts_the_dense_operator():
    """The wavelet node's memory is its ``n_max x n_max`` operator, not
    its ``2 n_max``-element state: the estimate has to say so, or a basis
    with a small state and a 4 GiB assembly passes the state cap."""
    estimate = size_estimate.estimate_allocation(
        WaveletAdaptiveNode, {"dim": 1, "n_levels": 12})
    n_max = 2 * 2 ** 12
    assert estimate == size_estimate.AllocationEstimate(2 * n_max, 64 * n_max ** 2)
    assert estimate.state_elements < MAX_NODE_STATE_ELEMENTS
    assert estimate.peak_bytes > MAX_NODE_BUILD_BYTES


@pytest.mark.parametrize("cls, params", [
    (LBMNode, {"grid_shape": "abc"}),
    (LBMNode, {"grid_shape": [8, 8]}),                 # rank 2 on D3Q19
    (LBMNode, {"grid_shape": [8, 8, 8], "lattice": "D3Q27"}),
    (LBMNode, {"grid_shape": [8, -1, 8]}),
    (HeatNode, {"n_cells": 2.5}),
    (HeatNode, {"n_cells": True}),
    (HeatNode, {"no_such_argument": 1}),
    (WaveletAdaptiveNode, {"dim": 4}),
    (WaveletAdaptiveNode, {"boundary": "neumann"}),
    (WaveletAdaptiveNode, {"n_levels": 0}),
])
def test_arguments_the_constructor_refuses_have_no_estimate(cls, params):
    """Left to the constructor, whose own message is the better 400."""
    assert size_estimate.estimate_allocation(cls, params) is None


def test_a_class_without_an_estimate_is_not_estimated():
    class Plain(SimulationNode):
        def __init__(self, name, timestep, n: int = 2):
            super().__init__(name, timestep, n=n)

        def initial_state(self):
            return {}

        def update(self, state, boundary_inputs, dt):
            return state

    assert size_estimate.estimate_allocation(Plain, {"n": 10 ** 9}) is None
    size_estimate.refuse_beyond_memory(Plain, "p", {"n": 10 ** 9})   # does not raise


def test_an_absurd_exponent_is_estimated_in_no_time():
    """``(2 * 2**10**7) ** 3`` takes seconds to form; the estimate
    saturates instead, and the messages print a power of two."""
    estimate = size_estimate.estimate_allocation(
        WaveletAdaptiveNode, {"dim": 3, "n_levels": 10_000_000})
    assert estimate.state_elements.bit_length() <= size_estimate.SATURATION_BITS + 2
    assert size_estimate.format_count(estimate.state_elements).startswith("at least 2^")
    assert size_estimate.format_bytes(estimate.peak_bytes).startswith("at least 2^")


# ---------------------------------------------------------------------------
# POST /graph/nodes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, params, names", OVERSIZED,
                         ids=[f"{c.__name__}-{i}" for i, (c, _, _) in enumerate(OVERSIZED)])
def test_post_refuses_an_oversized_node_before_constructing_it(monkeypatch, cls, params, names):
    calls = _guard_constructor(monkeypatch, cls)
    client = _client()
    resp = client.post("/graph/nodes", json={
        "type": cls.__name__, "name": "m", "timestep": 1.0, "params": params})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert names in detail and "before anything of that size is built" in detail
    assert calls == []
    assert client.get("/graph").json()["nodes"] == []


def test_post_still_builds_a_node_under_the_caps():
    resp = _client().post("/graph/nodes", json={
        "type": "LBMNode", "name": "m", "timestep": 1.0,
        "params": {"grid_shape": [8, 8], "lattice": "D2Q9"}})
    assert resp.status_code == 201, resp.text


def test_post_reads_the_servers_cap_when_it_answers(monkeypatch):
    """The module constant, as the post-construction check reads it."""
    monkeypatch.setattr(server_module, "MAX_NODE_STATE_ELEMENTS", 50)
    calls = _guard_constructor(monkeypatch, HeatNode)
    resp = _client().post("/graph/nodes", json={
        "type": "HeatNode", "name": "m", "timestep": 0.01, "params": {"n_cells": 51}})
    assert resp.status_code == 400, resp.text
    assert "51 state elements" in resp.json()["detail"]
    assert calls == []


# ---------------------------------------------------------------------------
# PUT /graph/params
# ---------------------------------------------------------------------------

def _graph(node, *, compile=False) -> GraphManager:
    gm = GraphManager()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_node(node)
        if compile:
            gm.compile()
    return gm


@pytest.fixture(scope="module")
def wavelet_graph():
    """One compiled, stepped wavelet graph for every PUT refusal (a refusal
    writes nothing, which each test checks)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm = _graph(WaveletAdaptiveNode("n", 1.0, dim=1, n_levels=4), compile=True)
        gm.step()
    return gm


@pytest.mark.parametrize("value", [22, 26, 10_000_000, 1e7])
def test_put_refuses_a_wavelet_level_count_before_constructing_it(
        monkeypatch, wavelet_graph, value):
    """The write that crashed the machine, and the sizes that cost 56 to
    655 MB before their 400."""
    gm = wavelet_graph
    before = (dict(gm.get_node("n").params), gm._dirty)
    calls = _guard_constructor(monkeypatch, WaveletAdaptiveNode)
    resp = _client(gm).put("/graph/params/n", json={"params": {"n_levels": value}})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("n_levels: ") and "before anything of that size" in detail
    assert calls == []
    assert (dict(gm.get_node("n").params), gm._dirty) == before


@pytest.mark.parametrize("shape", [[4000, 4000], [16000, 16000]])
def test_put_refuses_a_grid_shape_before_constructing_it(monkeypatch, shape):
    gm = _graph(LBMNode("n", 1.0, grid_shape=(8, 8), lattice="D2Q9"))
    calls = _guard_constructor(monkeypatch, LBMNode)
    resp = _client(gm).put("/graph/params/n", json={"params": {"grid_shape": shape}})
    assert resp.status_code == 400, resp.text
    assert "state elements" in resp.json()["detail"]
    assert calls == []
    assert gm.get_node("n").params["grid_shape"] == (8, 8)


def test_put_asks_every_key_at_once_as_well_as_one_at_a_time(monkeypatch):
    """``nx`` and ``ny`` each alone keep the pipe under the cap; together
    they do not, and the save check builds them together."""
    gm = _graph(LBMPipeNode("p", 1.0, nx=12, ny=8, nz=8, propeller_x=4))
    # 24*8*8*31 = 12*16*8*31 = 47616 each alone, 24*16*8*31 = 95232 together.
    monkeypatch.setattr(server_module, "MAX_NODE_STATE_ELEMENTS", 60_000)
    calls = _guard_constructor(monkeypatch, LBMPipeNode)
    resp = _client(gm).put("/graph/params/p", json={"params": {"nx": 24, "ny": 16}})
    assert resp.status_code == 400, resp.text
    detail = resp.json()["detail"]
    assert detail.startswith("nx, ny: ") and "95232 state elements" in detail
    assert calls == []
    assert (gm.get_node("p").params["nx"], gm.get_node("p").params["ny"]) == (12, 8)


def test_put_on_a_sharded_node_asks_the_class_it_wraps(monkeypatch):
    """A wrapper has no estimate of its own; the node it wraps, which
    shares its params and is the one the checks rebuild, answers."""
    from maddening.cloud.multigpu.device_mesh import create_device_mesh
    from maddening.cloud.multigpu.sharded_node import ShardedStencilNode

    monkeypatch.setattr(server_module, "MAX_NODE_STATE_ELEMENTS", 100)
    rod = HeatNode("rod", 1.0, n_cells=8, thermal_diffusivity=1e-18)
    gm = _graph(ShardedStencilNode(rod, create_device_mesh(shape=(1,)), {"devices": 0}))
    calls = _guard_constructor(monkeypatch, HeatNode)
    resp = _client(gm).put("/graph/params/rod", json={"params": {"n_cells": 101}})
    assert resp.status_code == 400, resp.text
    # The estimate's refusal, not the abstract state check's (which would
    # also name 101 elements here, under the patched cap).
    detail = resp.json()["detail"]
    assert "the HeatNode would hold 101 state elements" in detail
    assert "before anything of that size is built" in detail
    assert calls == []
    assert gm.get_node("rod").params["n_cells"] == 8


# ---------------------------------------------------------------------------
# GraphManager.from_dict
# ---------------------------------------------------------------------------

def _config_with(cls, dt, params, edit):
    gm = _graph(cls("n", dt, **params))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        config = gm.to_dict()
    config["nodes"][0]["params"].update(edit)
    return config


@pytest.mark.parametrize("cls, dt, params, edit", [
    (WaveletAdaptiveNode, 1.0, {"dim": 1, "n_levels": 4}, {"n_levels": 10_000_000}),
    (LBMNode, 1.0, {"grid_shape": (8, 8, 8)}, {"grid_shape": [10 ** 6] * 3}),
    (LBMPipeNode, 1.0, {"nx": 12, "ny": 8, "nz": 8, "propeller_x": 4},
     {"nx": 10 ** 7, "ny": 10 ** 7}),
])
def test_from_dict_refuses_a_node_no_machine_holds_before_constructing_it(
        monkeypatch, cls, dt, params, edit):
    config = _config_with(cls, dt, params, edit)
    calls = _guard_constructor(monkeypatch, cls)
    with pytest.raises(ValueError, match=r"node 'n' \(" + cls.__name__
                       + r"\) cannot be built on this machine"):
        GraphManager.from_dict(config, REGISTRY)
    assert calls == []


def test_from_dict_reads_the_machines_memory(monkeypatch):
    """The bound is what this machine has: a pipe that fits a large
    machine is refused on a small one, and loads where it fits."""
    config = _config_with(LBMPipeNode, 1.0, {"nx": 12, "ny": 8, "nz": 8, "propeller_x": 4}, {})
    monkeypatch.setattr(size_estimate, "physical_memory_bytes", lambda: 2 ** 10)
    with pytest.raises(ValueError, match="more than the"):
        GraphManager.from_dict(config, REGISTRY)
    monkeypatch.setattr(size_estimate, "physical_memory_bytes", lambda: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        again = GraphManager.from_dict(config, REGISTRY)
    assert again.get_node("n").params["nx"] == 12


# ---------------------------------------------------------------------------
# The wavelet constructor and its labels
# ---------------------------------------------------------------------------

def _labels_reference(n_levels, n_coarse, dim):
    """The list-built labels the node used before 0.4.0, for small sizes."""
    per_level = 2 ** dim - 1
    labs = [0] * (n_coarse ** dim)
    cur = n_coarse
    for lvl in range(n_levels):
        labs += [lvl] * (per_level * cur ** dim)
        cur *= 2
    return np.asarray(labs, dtype=np.int32)


def _dirichlet_labels_reference(n_levels, n_coarse):
    labs = [0] * n_coarse
    cur = n_coarse
    for lvl in range(n_levels):
        labs += [lvl] * (cur + 1)
        cur = 2 * cur + 1
    return np.asarray(labs, dtype=np.int32)


@pytest.mark.parametrize("dim", [1, 2, 3])
def test_the_vectorised_labels_are_the_labels_the_node_always_used(dim):
    for n_levels in range(0, 5):
        for n_coarse in (1, 2, 3):
            got = transform_module._level_labels_np(n_levels, n_coarse, dim)
            want = _labels_reference(n_levels, n_coarse, dim)
            assert got.dtype == want.dtype and np.array_equal(got, want)
            if dim == 1:
                got = dirichlet_module._level_labels_1d(n_levels, n_coarse)
                want = _dirichlet_labels_reference(n_levels, n_coarse)
                assert got.dtype == want.dtype and np.array_equal(got, want)


#: Run in a child process under an address-space cap, so that a regression
#: to the entry-by-entry list fails with MemoryError there instead of
#: taking this process (and the machine) with it.
_IMPOSSIBLE_SIZES_CHILD = r"""
import warnings
warnings.simplefilter("ignore")
from maddening.nodes.adaptive import WaveletAdaptiveNode
from maddening.nodes.adaptive.wavelets import dirichlet, transform
calls = [
    lambda: transform._level_labels_np(10_000_000, 2, 1),
    lambda: transform._level_labels_np(21, 2, 3),
    lambda: dirichlet._level_labels_1d(10_000_000, 2),
    lambda: dirichlet._level_labels_nd(30, 2, 3),
    lambda: WaveletAdaptiveNode("w", 1.0, dim=1, n_levels=10_000_000),
    lambda: WaveletAdaptiveNode("w", 1.0, dim=3, n_levels=1e7, boundary="dirichlet"),
]
for call in calls:
    try:
        call()
    except ValueError as exc:
        assert "too large" in str(exc) or "cannot index" in str(exc) \
            or "more than" in str(exc), exc
    else:
        raise SystemExit("built an impossible basis")
print("refused all")
"""


def test_an_impossible_basis_is_refused_at_once_not_built():
    """``n_levels=10_000_000``: a 2**10**7-function basis, refused by the
    labels and by the constructor before anything of that size exists.
    The pre-0.4.0 labels ran this into the OOM killer (57.7 GB)."""
    cap = 6 * 2 ** 30

    def limit():
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    proc = subprocess.run(
        [sys.executable, "-c", _IMPOSSIBLE_SIZES_CHILD],
        capture_output=True, text=True, timeout=120, preexec_fn=limit,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
    )
    assert proc.returncode == 0 and "refused all" in proc.stdout, proc.stderr[-2000:]


def test_the_constructor_refuses_a_basis_beyond_the_machines_memory(monkeypatch):
    """8192 functions assemble in about 4 GiB: refused on a machine with
    less, before the labels or the operator are built."""
    monkeypatch.setattr(wavelet_module, "physical_memory_bytes", lambda: 2 ** 30)

    def not_built(*args, **kwargs):
        raise AssertionError("the basis was built")

    monkeypatch.setattr(transform_module, "_level_labels_np", not_built)
    monkeypatch.setattr(wavelet_module._op, "assemble_operator", not_built)
    with pytest.raises(ValueError, match=r"assembles its operator dense .* more than "
                                         r"the 1\.0 GiB"):
        WaveletAdaptiveNode("w", 1.0, dim=1, n_levels=12)


def test_the_constructor_still_builds_a_basis_that_fits(monkeypatch):
    monkeypatch.setattr(wavelet_module, "physical_memory_bytes", lambda: 2 ** 30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        node = WaveletAdaptiveNode("w", 1.0, dim=1, n_levels=4)
    assert node.n_max == 32
