"""Coupled topologies of linear relays, and a monolithic float64 reference for a step of them.

:mod:`tests.property.coupled_graphs` draws one coupling group in a cycle,
with a driver upstream and a sink downstream.  Coupling's defects have also
lived in graph *shapes* that generator never draws: a node downstream of a
group added before its members (CPL-025), an outside node in a group's
strongly connected component (MADD-ANO-144), flux edges outside a group,
the summation order of additive edges.  This module describes a whole graph
-- several groups, outside nodes anywhere, every edge kind -- and restates
in float64 NumPy what one step of it must compute.

**The nodes** are the relays of ``coupled_graphs`` generalised:
``x <- alpha x_pre + sum_j G_j u_j + b + beta dt`` with ``u_j`` the sum of
the edges into port ``j``, at any float dtype (float64 runs under
``jax_enable_x64``), with any of four non-float leaves recomputed from the
pre-step state (an ``int32`` counter, a ``uint32`` LCG tag, a ``bool``
flag and a typed PRNG key), optionally a flux ``q = 2 x``, and optionally
on the three-argument ``update`` contract, whose constants are read from
``self.params`` when the step is traced.

**The edges** carry a field (``x``, or the flux ``q``), an optional linear
transform (``negate``, ``scale_2.0``, ``scale_0.5``, ``identity``), an
optional interface mapping (``matrix_mapping``: a dense ``H`` from the
source's size to the target's, a traced parameter) and the ``additive``
flag; two or more edges into one port sum.

**The reference.**  Every node is affine in its pre-step state and its
inputs, so a step is one linear system: each edge reads its source's new
value, except a *back edge*, which reads the pre-step value.  The
documented rule decides which edges are back edges
(:func:`documented_back_edges`, restated from ``topological_sort``,
``_block_schedule`` and CPL-025 rather than called): an edge between two
strongly connected components always reads the new value; inside a
component the nodes run in the order they were added, each coupling group
as one block at its first member's place, and an edge whose source does
not run before its target reads the old value -- except an edge inside a
group, which the group iterates.  :meth:`LinearModel.monolithic` solves the
system exactly.

**The oracle it supports** (:meth:`LinearModel.check_step`) is local and
exact.  For the state a step returned, every node's *defect* -- its value
minus its update evaluated in float64 at the values it read -- is:

* for a node outside every group, its own float rounding: at most
  ``T eps sum|term|`` per entry (:meth:`LinearModel.rounding`);
* for a group, ``-(I - L) r + epsilon`` exactly, where ``r`` is the
  residual of the group's one-pass map at the returned state, ``L`` the
  part of the pass read from the same pass (Gauss-Seidel), and
  ``epsilon`` the members' rounding.  So the defect is bounded by the
  group's *reported* residual through a computable operator norm
  (:meth:`LinearModel.group_defect_bound`), whatever the acceleration, the
  predictor or the solver did to reach that state.

The returned state minus the monolithic solution is ``(I - M)^{-1}`` of
the defect vector, exactly, so the defect bounds give the distance to the
reference.  Nothing here is a test.
"""

from __future__ import annotations

import dataclasses
import functools
import warnings
from typing import Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.transforms import scale
from tests.property import coupled_graphs as cg

# The scaling transforms are registered on first use; the generator names them.
scale(2.0)
scale(0.5)

#: The linear transforms an edge may carry, and the factor each applies.
#: Powers of two (and the identity), so a transform rounds nothing.
TRANSFORM_FACTORS = {None: 1.0, "identity": 1.0, "negate": -1.0,
                     "scale_2.0": 2.0, "scale_0.5": 0.5}

#: The non-float leaves a node may carry, all recomputed from the pre-step
#: state, so independent of the coupling iterate.
LEAF_KINDS = ("count", "tag", "flag", "key")

#: The typed PRNG key a ``"key"`` leaf starts at, and the datum each update
#: folds into it.
KEY_SEED = 271828
KEY_FOLD = 7

#: The reference's working precision.  The defects it measures are of the
#: order of the graph's own rounding, so it computes them in a precision
#: finer than the graph's: x86-64's 80-bit extended (``eps`` 1.1e-19), with
#: linear solves refined in it (:func:`_solve`).  A float64 graph's
#: rounding (``eps`` 2.2e-16) is then 2000 times the reference's.
LD = np.longdouble
EPS64 = float(np.finfo(np.float64).eps)


def reference_precision_ok() -> bool:
    """Is ``LD`` finer than float64 here (it is float64 on some platforms)?"""
    return float(np.finfo(LD).eps) < 1e-18


def _solve(A, b):
    """``A^{-1} b`` in ``LD``: a float64 solve refined by residuals computed in ``LD``."""
    A = np.asarray(A, LD)
    b = np.asarray(b, LD)
    A64 = np.asarray(A, np.float64)
    x = np.asarray(np.linalg.solve(A64, np.asarray(b, np.float64)), LD)
    for _ in range(4):
        r = b - A @ x
        x = x + np.asarray(np.linalg.solve(A64, np.asarray(r, np.float64)), LD)
    return x


def _dt(dtype) -> jnp.dtype:
    return jnp.dtype(dtype)


class TRelay(SimulationNode):
    """``x <- alpha x_pre + sum_j G_j u_j + b + beta dt``, at any float dtype.

    The arithmetic of :class:`~tests.property.coupled_graphs.Relay` in the
    same order (so a float32 ``TRelay`` evaluates the same expression), with
    a typed PRNG key among the leaves.
    """

    def __init__(self, name, timestep, *, n, ports, alpha=0.0, beta=0.0, leaves=(),
                 dtype="float32", constants=None):
        dt_ = _dt(dtype)
        params = {f"G{j}": jnp.zeros((n, n), dt_) for j in range(ports)}
        params["b"] = jnp.zeros(n, dt_)
        if constants is not None:
            for j, G in enumerate(constants["G"]):
                params[f"G{j}"] = jnp.asarray(G, dt_)
            params["b"] = jnp.asarray(constants["b"], dt_)
        super().__init__(name, timestep, **params)
        self._n = int(n)
        self._k = int(ports)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._leaves = tuple(leaves)
        self._dtype = dt_

    def initial_state(self):
        s = {"x": jnp.zeros(self._n, self._dtype)}
        if "count" in self._leaves:
            s["count"] = jnp.asarray(cg.COUNT0, jnp.int32)
        if "tag" in self._leaves:
            s["tag"] = jnp.asarray(cg.TAG0, jnp.uint32)
        if "flag" in self._leaves:
            s["flag"] = jnp.asarray(True)
        if "key" in self._leaves:
            s["key"] = jax.random.key(KEY_SEED)
        return s

    def boundary_input_spec(self):
        return {f"u{j}": BoundaryInputSpec(shape=(self._n,), dtype=self._dtype,
                                           default=jnp.zeros(self._n, self._dtype))
                for j in range(self._k)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        dtype = self._dtype
        f = (jnp.asarray(self._alpha, dtype) * state["x"] + p["b"]
             + jnp.asarray(self._beta, dtype) * jnp.asarray(dt, dtype))
        for j in range(self._k):
            u = boundary_inputs.get(f"u{j}", jnp.zeros(self._n, dtype))
            f = f + p[f"G{j}"] @ u
        out = {"x": f.astype(dtype)}
        if "count" in self._leaves:
            out["count"] = state["count"] + jnp.int32(1)
        if "tag" in self._leaves:
            out["tag"] = state["tag"] * jnp.uint32(cg.TAG_MUL) + jnp.uint32(cg.TAG_ADD)
        if "flag" in self._leaves:
            out["flag"] = jnp.logical_not(state["flag"])
        if "key" in self._leaves:
            out["key"] = jax.random.fold_in(state["key"], KEY_FOLD)
        return out

    def update_evaluations(self):
        return 1


class FluxTRelay(TRelay):
    """A :class:`TRelay` that also produces the boundary flux ``q = 2 x``."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": jnp.asarray(2.0, self._dtype) * state["x"]}


class TRelay3(TRelay):
    """A :class:`TRelay` on the three-argument contract: constants read at trace time."""

    def update(self, state, boundary_inputs, dt):
        return TRelay.update(self, state, boundary_inputs, dt)


class FluxTRelay3(TRelay3):
    """A three-argument relay with the flux ``q = 2 x`` (three-argument too)."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"q": jnp.asarray(2.0, self._dtype) * state["x"]}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class TNode:
    """One relay: everything about it the compiled step bakes in."""

    name: str
    n: int = 1
    ports: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    leaves: tuple = ()
    flux: bool = False
    three_arg: bool = False
    timestep: float = 1.0


@dataclasses.dataclass(frozen=True)
class TEdge:
    """``src.field -> dst.u{port}``, through ``mapping`` (``H``) then ``transform``."""

    src: str
    dst: str
    port: int
    field: str = "x"
    transform: Optional[str] = None
    additive: bool = False
    mapped: bool = False


@dataclasses.dataclass(frozen=True)
class Topology:
    """Nodes, edges and coupling groups (each a tuple of members in their build order)."""

    nodes: tuple
    edges: tuple
    groups: tuple = ()
    label: str = ""

    def node(self, name: str) -> TNode:
        return next(nd for nd in self.nodes if nd.name == name)

    @property
    def names(self) -> tuple:
        return tuple(nd.name for nd in self.nodes)

    def group_of(self, name: str) -> Optional[int]:
        for gi, members in enumerate(self.groups):
            if name in members:
                return gi
        return None

    def internal(self, e: TEdge) -> bool:
        g = self.group_of(e.src)
        return g is not None and g == self.group_of(e.dst)

    def group_key(self, gi: int, rename: Optional[dict] = None) -> str:
        members = self.groups[gi]
        if rename:
            members = [rename[m] for m in members]
        return "+".join(sorted(members))

    def internal_edges(self, gi: int) -> list:
        return [i for i, e in enumerate(self.edges) if self.group_of(e.src) == gi
                and self.group_of(e.dst) == gi]

    def renamed(self, mapping: dict) -> "Topology":
        return Topology(
            nodes=tuple(dataclasses.replace(nd, name=mapping[nd.name]) for nd in self.nodes),
            edges=tuple(dataclasses.replace(e, src=mapping[e.src], dst=mapping[e.dst])
                        for e in self.edges),
            groups=tuple(tuple(mapping[m] for m in g) for g in self.groups),
            label=self.label)

    def with_timesteps(self, timesteps: dict) -> "Topology":
        return dataclasses.replace(self, nodes=tuple(
            dataclasses.replace(nd, timestep=timesteps.get(nd.name, nd.timestep))
            for nd in self.nodes))

    def check(self) -> None:
        """Refuse a structure the generator must never produce (shape errors, port overlaps)."""
        names = self.names
        assert len(set(names)) == len(names), names
        for e in self.edges:
            src, dst = self.node(e.src), self.node(e.dst)
            assert 0 <= e.port < dst.ports, e
            assert e.mapped or src.n == dst.n, f"unmapped edge between sizes: {e}"
            assert e.field == "x" or src.flux, e
        for nd in self.nodes:
            for j in range(nd.ports):
                into = [e for e in self.edges if e.dst == nd.name and e.port == j]
                assert into, f"{nd.name}.u{j} has no edge"
                if len(into) > 1:
                    assert all(e.additive for e in into), f"{nd.name}.u{j}: replacive overlap"
        seen = [m for g in self.groups for m in g]
        assert len(seen) == len(set(seen)), "groups overlap"


def node_class(nd: TNode) -> type:
    if nd.three_arg:
        return FluxTRelay3 if nd.flux else TRelay3
    return FluxTRelay if nd.flux else TRelay


def fixed_constants(topo: Topology, dtype="float32") -> dict:
    """The constants of every three-argument node: fixed per structure.

    A three-argument node reads its gains from ``self.params`` when the
    step is traced, so they cannot be drawn per example on one compiled
    graph.  Small (``0.2 / n`` scale), so that a group the other gains are
    rescaled around still has room to reach its drawn rate.  Seeded by the
    node's place among the three-argument nodes and its shape -- not by its
    name or its index -- so a renamed topology, or one with a relay
    inserted, holds the same constants.
    """
    out = {}
    three = [nd for nd in topo.nodes if nd.three_arg]
    for k, nd in enumerate(three):
        rng = np.random.default_rng(10_000 + 97 * k + 13 * nd.ports + nd.n)
        G = [np.asarray(rng.normal(size=(nd.n, nd.n)) * 0.2 / nd.n, _dt(dtype))
             for _ in range(nd.ports)]
        out[nd.name] = {"G": G, "b": np.asarray(rng.normal(size=nd.n), _dt(dtype))}
    return out


def make_node(nd: TNode, dtype="float32", constants=None) -> TRelay:
    return node_class(nd)(nd.name, nd.timestep, n=nd.n, ports=nd.ports, alpha=nd.alpha,
                          beta=nd.beta, leaves=nd.leaves, dtype=dtype, constants=constants)


@dataclasses.dataclass
class Built:
    """A compiled graph of a :class:`Topology` and what the build recorded."""

    gm: GraphManager
    topo: Topology
    dtype: str
    #: ``{edge index: the library's edge key}`` for every mapped edge.
    mapping_keys: dict
    #: The ``UserWarning`` texts ``compile()`` raised.
    warnings: list
    node_order: tuple
    edge_order: tuple


def build(topo: Topology, knobs, *, dtype="float32", node_order=None, edge_order=None,
          compile: bool = True) -> Built:
    """A :class:`GraphManager` for *topo*, nodes and edges added in the given orders.

    *knobs* is one ``CouplingGroup`` configuration for every group, or a
    sequence of them, one per group; knobs a configuration leaves inert are
    dropped (:func:`coupled_graphs.live_knobs`).  ``compile()``'s
    ``UserWarning`` texts are recorded rather than raised.
    """
    topo.check()
    node_order = tuple(topo.names if node_order is None else node_order)
    edge_order = tuple(range(len(topo.edges)) if edge_order is None else edge_order)
    assert sorted(node_order) == sorted(topo.names) and sorted(edge_order) == list(
        range(len(topo.edges)))
    constants = fixed_constants(topo, dtype)
    gm = GraphManager()
    for name in node_order:
        nd = topo.node(name)
        gm.add_node(make_node(nd, dtype, constants.get(name)))
    for i in edge_order:
        e = topo.edges[i]
        mapping = None
        if e.mapped:
            n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
            mapping = matrix_mapping(np.zeros((n_dst, n_src), _dt(dtype)))
        gm.add_edge(e.src, e.dst, e.field, f"u{e.port}", transform=e.transform,
                    additive=e.additive, mapping=mapping)
    per_group = knobs if isinstance(knobs, (list, tuple)) else [knobs] * len(topo.groups)
    for members, group in zip(topo.groups, per_group):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "CouplingGroup solver='fori' is deprecated", DeprecationWarning)
            gm.add_coupling_group(list(members), **cg.live_knobs(group))
    keys = {}
    for pos, i in enumerate(edge_order):
        if topo.edges[i].mapped:
            keys[i] = gm._edges[pos].key  # noqa: SLF001
    recorded: list = []
    if compile:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gm.compile()
        for w in caught:
            if issubclass(w.category, DeprecationWarning):
                continue
            if "multi-rate" in str(w.message):
                continue
            recorded.append(str(w.message))
    return Built(gm, topo, str(dtype), keys, recorded, node_order, edge_order)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _spectral_radius(M: np.ndarray) -> float:
    M = np.asarray(M, np.float64)
    return float(np.max(np.abs(np.linalg.eigvals(M)))) if M.size else 0.0


def draw_values(topo: Topology, rng: np.random.Generator, rho: float, *,
                nonnormal: bool = False, bias_scale: float = 1.0, dtype="float32",
                group_cfgs=None) -> dict:
    """Gains, biases, mapping matrices and initial states; each group at rate *rho*.

    ``{"nodes": {name: {"G": [...], "b": ..., "x0": ...}}, "H": {edge: ...}}``.
    Every group's coupling operator (its fixed-point map, sub-cycling
    included: see :class:`LinearModel`) is rescaled so its spectral radius
    -- the Jacobi rate -- is *rho*: the drawn gains of its four-argument
    members' internal ports are multiplied by one factor, found by
    bisection (a three-argument member's gains are fixed per structure).
    Gains into outside nodes, and across groups, are divided by the
    port's size so no driver or reader dominates a spectrum.
    """
    constants = fixed_constants(topo, dtype)
    values: dict = {"nodes": {}, "H": {}}
    for nd in topo.nodes:
        if nd.name in constants:
            v = {"G": [np.asarray(G, np.float64) for G in constants[nd.name]["G"]],
                 "b": np.asarray(constants[nd.name]["b"], np.float64)}
        else:
            Gs = []
            for _ in range(nd.ports):
                G = rng.normal(size=(nd.n, nd.n))
                if nonnormal and nd.n > 1:
                    G = np.triu(G) + np.triu(rng.normal(size=(nd.n, nd.n)) * 4.0, 1)
                Gs.append(G)
            v = {"G": Gs, "b": rng.normal(size=nd.n) * bias_scale}
        v["x0"] = rng.normal(size=nd.n) * bias_scale
        values["nodes"][nd.name] = v
    for i, e in enumerate(topo.edges):
        if e.mapped:
            n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
            values["H"][i] = rng.normal(size=(n_dst, n_src)) / np.sqrt(n_src)
    # Ports that read anything other than their own group's members are
    # scaled down; ports wholly inside a group are what the rescale moves.
    for nd in topo.nodes:
        if nd.name in constants:
            continue
        gi = topo.group_of(nd.name)
        for j in range(nd.ports):
            into = [e for e in topo.edges if e.dst == nd.name and e.port == j]
            if gi is None or any(topo.group_of(e.src) != gi for e in into):
                values["nodes"][nd.name]["G"][j] = values["nodes"][nd.name]["G"][j] / max(nd.n, 1)
    for gi in range(len(topo.groups)):
        _rescale_group(topo, values, gi, rho, constants, dtype, group_cfgs)
    for nd in topo.nodes:
        v = values["nodes"][nd.name]
        v["G"] = [np.asarray(G, _dt(dtype)) for G in v["G"]]
        v["b"] = np.asarray(v["b"], _dt(dtype))
        v["x0"] = np.asarray(v["x0"], _dt(dtype))
    values["H"] = {i: np.asarray(H, _dt(dtype)) for i, H in values["H"].items()}
    return values


def _rescale_group(topo, values, gi, rho, constants, dtype, group_cfgs):
    members = topo.groups[gi]
    scalable = []
    for m in members:
        if m in constants:
            continue
        nd = topo.node(m)
        for j in range(nd.ports):
            into = [e for e in topo.edges if e.dst == m and e.port == j]
            if into and all(topo.group_of(e.src) == gi for e in into):
                scalable.append((m, j))
    base = {(m, j): np.asarray(values["nodes"][m]["G"][j], np.float64) for m, j in scalable}

    def radius(s):
        for (m, j), G in base.items():
            values["nodes"][m]["G"][j] = G * s
        model = LinearModel(topo, values, dtype="float64", group_cfgs=group_cfgs, exact=True)
        return _spectral_radius(model.group_operator(gi))

    if not scalable or radius(1.0) == 0.0:
        radius(1.0)
        return
    lo, hi = 0.0, 1.0
    while radius(hi) < abs(rho):
        hi *= 2.0
        if hi > 1e8:
            break
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if radius(mid) < abs(rho):
            lo = mid
        else:
            hi = mid
    radius(0.5 * (lo + hi))


def params_for(built: Built, values: dict, rename: Optional[dict] = None) -> dict:
    """``gm.params`` with every four-argument node's gains, bias and every mapping's ``H``."""
    gm = built.gm
    rename = rename or {}
    base = gm.params
    nodes = {name: dict(p) for name, p in base["nodes"].items()}
    for name, v in values["nodes"].items():
        live = rename.get(name, name)
        if live not in nodes or built.topo.node(live).three_arg:
            continue
        for j, G in enumerate(v["G"]):
            nodes[live][f"G{j}"] = jnp.asarray(G, nodes[live][f"G{j}"].dtype)
        nodes[live]["b"] = jnp.asarray(v["b"], nodes[live]["b"].dtype)
    mappings = {k: dict(p) for k, p in base.get("mappings", {}).items()}
    for i, key in built.mapping_keys.items():
        mappings[key]["H"] = jnp.asarray(values["H"][i], mappings[key]["H"].dtype)
    return {**base, "nodes": nodes, "mappings": mappings}


def set_initial(built: Built, values: dict, rename: Optional[dict] = None) -> None:
    """Reset (state and coupling seeds) and write every node's ``x0``."""
    gm = built.gm
    rename = rename or {}
    cg.recover(gm)
    gm.reset_state()
    for name, v in values["nodes"].items():
        live = rename.get(name, name)
        s = dict(gm.get_node_state(live))
        s["x"] = jnp.asarray(v["x0"], s["x"].dtype)
        gm.set_node_state(live, s)


class Step(tuple):
    """``(pre, state, reports, metas)`` of one step; ``reports`` / ``metas`` keyed by group."""

    __slots__ = ()

    def __new__(cls, pre, state, reports, metas):
        return super().__new__(cls, (pre, state, reports, metas))

    pre = property(lambda self: self[0])
    state = property(lambda self: self[1])
    reports = property(lambda self: self[2])
    metas = property(lambda self: self[3])


def run(built: Built, values: dict, steps: int, *, rename: Optional[dict] = None,
        report: bool = True) -> list:
    """``[Step(pre, state, reports, metas)]`` for each of *steps* steps from the drawn start.

    ``pre`` and ``state`` are ``{node: {field: array}}`` under the original
    names; ``reports`` is ``coupling_diagnostics()`` and ``metas`` each
    group's ``_meta`` slots, both keyed by group index.
    """
    rename = rename or {}
    back = {v: k for k, v in rename.items()}
    set_initial(built, values, rename)
    params = params_for(built, values, rename)
    gm = built.gm
    out = []
    pre = _snapshot(gm, back)
    for _ in range(steps):
        gm.step(params=params)
        state = _snapshot(gm, back)
        reports, metas = {}, {}
        diag = gm.coupling_diagnostics() if report else {}
        for gi in range(len(built.topo.groups)):
            key = built.topo.group_key(gi)
            if key in diag:
                reports[gi] = dict(diag[key])
            metas[gi] = cg.group_meta(gm, key)
        out.append(Step(pre, state, reports, metas))
        pre = state
    return out


def _snapshot(gm, back) -> dict:
    out = {}
    for n in gm.node_names:
        s = gm.get_node_state(n)
        out[back.get(n, n)] = {f: (np.asarray(jax.random.key_data(v))
                                   if jax.dtypes.issubdtype(v.dtype, jax.dtypes.prng_key)
                                   else np.asarray(v)) for f, v in s.items()}
    return out


@functools.lru_cache(maxsize=None)
def _key_after(updates: int) -> np.ndarray:
    key = jax.random.key(KEY_SEED)
    for _ in range(updates):
        key = jax.random.fold_in(key, KEY_FOLD)
    return np.asarray(jax.random.key_data(key))


def expected_leaves(nd: TNode, updates: int) -> dict:
    """The closed-form non-float leaves of *nd* after *updates* calls of ``update``."""
    base = cg.NodeDef(nd.name, 0, leaves=tuple(lf for lf in nd.leaves if lf != "key"))
    out = cg.expected_leaves(base, updates)
    if "key" in nd.leaves:
        out["key"] = _key_after(updates)
    return out


# ---------------------------------------------------------------------------
# The documented schedule
# ---------------------------------------------------------------------------


def strongly_connected_components(names: Sequence[str], edges) -> dict:
    """``{node: component id}``, every node in one (singletons included).

    Kosaraju's two passes, written out here rather than imported, so the
    reference does not share the library's component code.
    """
    succ = {n: [] for n in names}
    pred = {n: [] for n in names}
    for e in edges:
        succ[e.src].append(e.dst)
        pred[e.dst].append(e.src)
    seen, finish = set(), []
    for root in names:
        if root in seen:
            continue
        stack = [(root, iter(succ[root]))]
        seen.add(root)
        while stack:
            node, it = stack[-1]
            nxt = next((m for m in it if m not in seen), None)
            if nxt is None:
                stack.pop()
                finish.append(node)
            else:
                seen.add(nxt)
                stack.append((nxt, iter(succ[nxt])))
    comp: dict = {}
    for root in reversed(finish):
        if root in comp:
            continue
        cid = len(set(comp.values()))
        todo = [root]
        comp[root] = cid
        while todo:
            node = todo.pop()
            for m in pred[node]:
                if m not in comp:
                    comp[m] = cid
                    todo.append(m)
    return comp


def documented_back_edges(topo: Topology, node_order: Sequence[str]) -> set:
    """Indices of the edges a step reads from the previous step's state.

    The documented rule (``topological_sort``, ``_block_schedule``,
    CPL-025): an edge between two strongly connected components always
    points forward; inside a component the nodes run in their ``add_node``
    order, each coupling group as one block at its first member's place;
    an edge inside a group is iterated by the group, never staggered; any
    other edge whose source does not run before its target (a self-loop
    included) reads the previous step.
    """
    comp = strongly_connected_components(list(node_order), topo.edges)
    blocks: dict = {}
    for name in node_order:
        gi = topo.group_of(name)
        tag = ("g", gi) if gi is not None else ("n", name)
        blocks.setdefault(comp[name], [])
        if tag not in blocks[comp[name]]:
            blocks[comp[name]].append(tag)

    def place(name):
        gi = topo.group_of(name)
        tag = ("g", gi) if gi is not None else ("n", name)
        return blocks[comp[name]].index(tag)

    out = set()
    for i, e in enumerate(topo.edges):
        if topo.internal(e) or comp[e.src] != comp[e.dst]:
            continue
        if place(e.src) >= place(e.dst):
            out.add(i)
    return out


def gauss_seidel_order(topo: Topology, gi: int, node_order: Sequence[str]) -> list:
    """A group's sweep order: its members in the order they were added (CPL-077)."""
    members = set(topo.groups[gi])
    return [n for n in node_order if n in members]


def group_in_larger_loop(topo: Topology, gi: int) -> bool:
    """Is group *gi* a strict subset of one strongly connected component?"""
    comp = strongly_connected_components(list(topo.names), topo.edges)
    members = topo.groups[gi]
    cids = {comp[m] for m in members}
    if len(cids) != 1:
        return False
    cid = cids.pop()
    return sum(1 for n in topo.names if comp[n] == cid) > len(members)


# ---------------------------------------------------------------------------
# The float64 reference
# ---------------------------------------------------------------------------


class LinearModel:
    """One step of a :class:`Topology` of linear relays, in float64.

    Parameters
    ----------
    topo : Topology
    values : dict
        From :func:`draw_values` (already rounded to the graph's dtype).
    dtype : str
        The graph's float dtype: constants are rounded to it first, so the
        map solved is the one the graph evaluates, and its ``eps`` sizes
        the rounding allowances.
    node_order : sequence of str, optional
        The ``add_node`` order (decides the back edges and the sweeps).
    group_cfgs : sequence of dict, optional
        Per group: ``iteration_mode``, ``subcycling``,
        ``boundary_interpolation``, ``convergence_norm``, ``rtol``,
        ``tolerance``.  Defaults are ``CouplingGroup``'s.
    exact : bool
        Skip the dtype rounding of the constants (used while drawing).
    """

    def __init__(self, topo: Topology, values: dict, *, dtype="float32", node_order=None,
                 group_cfgs=None, exact: bool = False):
        self.topo = topo
        self.values = values
        self.dtype = _dt(dtype)
        self.eps = float(np.finfo(self.dtype).eps)
        self.order = tuple(topo.names if node_order is None else node_order)
        self.cfgs = [dict(c) for c in (group_cfgs or [{} for _ in topo.groups])]
        while len(self.cfgs) < len(topo.groups):
            self.cfgs.append({})
        self.back = documented_back_edges(topo, self.order)
        self.offset, k = {}, 0
        for nd in topo.nodes:
            self.offset[nd.name] = k
            k += nd.n
        self.dim = k
        rnd = (lambda a: np.asarray(a, np.float64)) if exact else (
            lambda a: np.asarray(np.asarray(a, self.dtype), np.float64))
        self._rnd = rnd
        self.divider = {}
        for gi, members in enumerate(topo.groups):
            if self.cfgs[gi].get("subcycling"):
                macro = max(topo.node(m).timestep for m in members)
                for m in members:
                    self.divider[m] = int(round(macro / topo.node(m).timestep))
        self.terms = {nd.name: self._node_terms(nd) for nd in topo.nodes}

    # -- per node -------------------------------------------------------------

    def _node_terms(self, nd: TNode):
        """``(P, c, [(edge index, K)])``: ``x = P x_pre + c + sum K v_e``.

        A node sub-cycled ``d`` times is its update composed ``d`` times
        with the inputs held (the fixed point's map, under either
        interpolation): ``P = alpha**d``, and ``c`` and every ``K`` carry
        ``s_d = sum_k alpha**k``.
        """
        rnd = self._rnd
        v = self.values["nodes"][nd.name]
        alpha = float(rnd(nd.alpha))
        beta = float(rnd(nd.beta))
        dt = float(rnd(nd.timestep))
        d = self.divider.get(nd.name, 1)
        a = LD(alpha)
        s_d = sum((a ** k for k in range(d)), LD(0))
        P = a ** d
        c = s_d * (np.asarray(rnd(v["b"]), LD) + LD(beta) * LD(dt))
        inputs = []
        for i, e in enumerate(self.topo.edges):
            if e.dst != nd.name:
                continue
            G = np.asarray(rnd(v["G"][e.port]), LD)
            fac = LD(TRANSFORM_FACTORS[e.transform] * (2.0 if e.field == "q" else 1.0))
            H = (np.asarray(rnd(self.values["H"][i]), LD) if e.mapped
                 else np.eye(nd.n, dtype=LD))
            inputs.append((i, s_d * fac * (G @ H)))
        return P, c, inputs

    def _vec(self, state: dict) -> np.ndarray:
        x = np.zeros(self.dim, LD)
        for nd in self.topo.nodes:
            o = self.offset[nd.name]
            x[o:o + nd.n] = np.asarray(np.asarray(state[nd.name]["x"]), LD)
        return x

    def _split(self, x: np.ndarray) -> dict:
        return {nd.name: x[self.offset[nd.name]:self.offset[nd.name] + nd.n]
                for nd in self.topo.nodes}

    # -- the whole step -------------------------------------------------------

    def matrices(self):
        """``(M_new, M_old, P, c)``: ``x = M_new x + M_old x_pre + P x_pre + c``."""
        M_new = np.zeros((self.dim, self.dim), LD)
        M_old = np.zeros((self.dim, self.dim), LD)
        Pd = np.zeros(self.dim, LD)
        c = np.zeros(self.dim, LD)
        for nd in self.topo.nodes:
            P, cn, inputs = self.terms[nd.name]
            o = self.offset[nd.name]
            Pd[o:o + nd.n] = P
            c[o:o + nd.n] = cn
            for i, K in inputs:
                e = self.topo.edges[i]
                s = self.offset[e.src]
                tgt = M_old if i in self.back else M_new
                tgt[o:o + nd.n, s:s + self.topo.node(e.src).n] += K
        return M_new, M_old, Pd, c

    def monolithic(self, pre: dict) -> dict:
        """The step's exact result: every group at its fixed point, back edges from *pre*."""
        M_new, M_old, Pd, c = self.matrices()
        xp = self._vec(pre)
        x = _solve(np.eye(self.dim, dtype=LD) - M_new, M_old @ xp + Pd * xp + c)
        return self._split(x)

    def defects(self, pre: dict, state: dict) -> dict:
        """Each node's value minus its update evaluated exactly at what it read."""
        M_new, M_old, Pd, c = self.matrices()
        xp, x = self._vec(pre), self._vec(state)
        return self._split(x - (M_new @ x + M_old @ xp + Pd * xp + c))

    def distance_operator(self) -> np.ndarray:
        """``(I - M_new)^{-1}``: the returned state minus the reference is this times the defects."""
        M_new, _M_old, _Pd, _c = self.matrices()
        return np.linalg.inv(np.eye(self.dim) - np.asarray(M_new, np.float64))

    # -- rounding ---------------------------------------------------------------

    def _input_magnitudes(self, name, read: dict):
        """Per port, ``sum_e |fac| |H| |v_e|`` (elementwise) at the values *read*."""
        nd = self.topo.node(name)
        out = [np.zeros(nd.n) for _ in range(nd.ports)]
        for i, e in enumerate(self.topo.edges):
            if e.dst != name:
                continue
            fac = abs(TRANSFORM_FACTORS[e.transform]) * (2.0 if e.field == "q" else 1.0)
            v = np.abs(np.asarray(read[(i,)], np.float64))
            H = np.abs(self._rnd(self.values["H"][i])) if e.mapped else np.eye(nd.n)
            out[e.port] = out[e.port] + fac * (H @ v)
        return out

    def _flops(self, name) -> int:
        """The longest chain of roundings in one evaluation of *name*'s update, per entry."""
        nd = self.topo.node(name)
        t = 4
        for j in range(nd.ports):
            into = [e for e in self.topo.edges if e.dst == name and e.port == j]
            t += nd.n + 1 + len(into) + max(
                (self.topo.node(e.src).n if e.mapped else 0) for e in into)
        return t

    def rounding(self, name, pre: dict, read: dict) -> np.ndarray:
        """Elementwise bound on how far *name*'s computed update is from the exact one.

        One evaluation of a sum of ``T`` rounded terms is within
        ``gamma_T sum |term|`` of the exact sum (the standard forward error
        of summation and of an inner product); ``gamma_T`` is taken as
        ``T eps``, twice ``T u``, which also covers the ``1 / (1 - T u)``
        factor and any reassociation or FMA contraction XLA applies (none
        adds a rounding to the chain).  ``T`` counts the chain
        (:meth:`_flops`): the three constant terms and, per port, the
        product, the additive edges' sum and a mapping's inner product.  A
        node sub-cycled ``d`` times adds each sub-step's rounding, carried
        by ``alpha`` to the end.  *read* maps ``(edge index,)`` to the value
        the edge read (the larger of the candidates, where a pass may read
        either).
        """
        rnd = self._rnd
        nd = self.topo.node(name)
        v = self.values["nodes"][name]
        alpha = abs(float(rnd(nd.alpha)))
        const = np.abs(rnd(v["b"])) + abs(float(rnd(nd.beta)) * float(rnd(nd.timestep)))
        inputs = self._input_magnitudes(name, read)
        drive = sum((np.abs(rnd(v["G"][j])) @ inputs[j] for j in range(nd.ports)),
                    np.zeros(nd.n))
        gamma = self._flops(name) * self.eps
        d = self.divider.get(name, 1)
        x = np.abs(np.asarray(pre[name]["x"], np.float64))
        total = np.zeros(nd.n)
        for k in range(d):
            terms = alpha * x + const + drive
            total = alpha * total + gamma * terms
            x = terms          # an upper bound on |x| after the sub-step
        return total

    # -- one group --------------------------------------------------------------

    def group_operator(self, gi: int) -> np.ndarray:
        """The group's coupling operator ``M_g`` over its members (the Jacobi map's linear part)."""
        members = self.topo.groups[gi]
        off, k = {}, 0
        for m in members:
            off[m] = k
            k += self.topo.node(m).n
        M = np.zeros((k, k), LD)
        for m in members:
            for i, K in self.terms[m][2]:
                e = self.topo.edges[i]
                if self.topo.group_of(e.src) == gi:
                    M[off[m]:off[m] + self.topo.node(m).n,
                      off[e.src]:off[e.src] + self.topo.node(e.src).n] += K
        return M

    def _group_layout(self, gi):
        members = self.topo.groups[gi]
        off, k = {}, 0
        for m in members:
            off[m] = k
            k += self.topo.node(m).n
        return members, off, k

    def group_pass(self, gi: int):
        """``(L, U)``: the one-pass map ``y = L y + U x + c_g`` over the group's members.

        Under Jacobi every internal read is of the incoming iterate.  Under
        Gauss-Seidel a member reads every member swept before it from the
        same pass and every other (itself included) from the incoming
        iterate; a sub-cycled member under ``boundary_interpolation`` other
        than ``"constant"`` reads a source swept before it at sub-step
        ``k`` as ``incoming + w_k (in-pass - incoming)``, ``w_k = (k + 1)
        / d`` (``_resolve_boundary_interpolated``), so that source's weight
        splits between ``L`` and ``U``.
        """
        cfg = self.cfgs[gi]
        members, off, k = self._group_layout(gi)
        jacobi = cfg.get("iteration_mode", "gauss-seidel") == "jacobi"
        interp = cfg.get("boundary_interpolation", "linear") if cfg.get("subcycling") else None
        sweep = gauss_seidel_order(self.topo, gi, self.order)
        rank = {m: sweep.index(m) for m in members}
        L = np.zeros((k, k), LD)
        U = np.zeros((k, k), LD)
        rnd = self._rnd
        for m in members:
            nd = self.topo.node(m)
            d = self.divider.get(m, 1)
            alpha = LD(float(rnd(nd.alpha)))
            s_d = sum((alpha ** j for j in range(d)), LD(0))
            w_in = (sum((alpha ** (d - 1 - j) * LD(j + 1) / LD(d) for j in range(d)), LD(0))
                    / s_d if (interp in ("linear", "quadratic") and d > 1 and s_d != 0)
                    else LD(1))
            for i, K in self.terms[m][2]:
                e = self.topo.edges[i]
                if self.topo.group_of(e.src) != gi:
                    continue
                rows = slice(off[m], off[m] + nd.n)
                cols = slice(off[e.src], off[e.src] + self.topo.node(e.src).n)
                if not jacobi and rank[e.src] < rank[m]:
                    L[rows, cols] += w_in * K
                    U[rows, cols] += (LD(1) - w_in) * K
                else:
                    U[rows, cols] += K
        return L, U

    def group_constant(self, gi: int, pre: dict, state: dict) -> np.ndarray:
        """``c_g``: everything a member's update adds besides its group's own reads."""
        members, off, k = self._group_layout(gi)
        c = np.zeros(k, LD)
        for m in members:
            P, cn, inputs = self.terms[m]
            acc = P * np.asarray(np.asarray(pre[m]["x"]), LD) + cn
            for i, K in inputs:
                e = self.topo.edges[i]
                if self.topo.group_of(e.src) == gi:
                    continue
                src = pre if i in self.back else state
                acc = acc + K @ np.asarray(np.asarray(src[e.src]["x"]), LD)
            c[off[m]:off[m] + self.topo.node(m).n] = acc
        return c

    def group_fixed_point(self, gi: int, pre: dict, state: dict) -> dict:
        """The group's exact fixed point given the outside values it read in *state*."""
        members, off, k = self._group_layout(gi)
        x = _solve(np.eye(k, dtype=LD) - self.group_operator(gi),
                   self.group_constant(gi, pre, state))
        return {m: x[off[m]:off[m] + self.topo.node(m).n] for m in members}

    def norm_fields(self, gi: int, *, raw: bool = False) -> list:
        """The fields group *gi*'s norm reads: ``[(B, gamma), ...]``.

        Each ``B`` maps the members' stacked ``x`` to one field, and the
        norm divides every entry of a field by ``rtol`` times that
        field's own largest magnitude.  Under ``"l2"`` and ``"mixed"`` a
        field is a member's ``x``.  Under ``"interface"`` it is **what one
        internal edge delivers**: the source's ``x`` through the edge's
        mapping matrix ``H``, then its transform's factor -- the value the
        step hands the target, with the weights the step ran with.  (An
        internal flux edge under that norm is refused at compile,
        CPL-075.)  While the library read the source's ``x`` there and
        left ``H`` out, this model restated that rule and agreed with it.

        ``gamma`` bounds the float evaluation of the reading itself,
        relative to ``|B| |x|``: a mapped edge's delivered value is an
        inner product the norm computes again from the stored field
        (``(n + 1) eps``); every other field is read as stored, a
        power-of-two factor being exact.

        ``raw=True`` gives the fields the *gradient* bound's norm is
        documented in under ``"interface"``: the source fields the edges
        read, before any mapping or transform.
        """
        norm = self.cfgs[gi].get("convergence_norm", "l2")
        members, off, k = self._group_layout(gi)
        fields = []
        if norm != "interface":
            for m in members:
                n = self.topo.node(m).n
                B = np.zeros((n, k))
                B[:, off[m]:off[m] + n] = np.eye(n)
                fields.append((B, 0.0))
            return fields
        for i in self.topo.internal_edges(gi):
            e = self.topo.edges[i]
            assert e.field == "x", f"the interface norm cannot read a flux edge: {e}"
            n_src = self.topo.node(e.src).n
            if raw or not e.mapped:
                block = np.eye(n_src)
                gamma = 0.0
            else:
                block = np.asarray(self._rnd(self.values["H"][i]), np.float64)
                gamma = (n_src + 1) * self.eps
            if not raw:
                block = TRANSFORM_FACTORS[e.transform] * block
            B = np.zeros((block.shape[0], k))
            B[:, off[e.src]:off[e.src] + n_src] = block
            fields.append((B, gamma))
        return fields

    def norm_parts(self, gi: int, state: dict, F: Optional[np.ndarray] = None, *,
                   raw: bool = False):
        """``(S, w, rtol_eff, rms)`` of the group's norm at the returned state.

        ``S`` maps the members' stacked ``x`` to the norm's entries (the
        rows of :meth:`norm_fields`: each member once for ``"l2"`` /
        ``"mixed"``, what each internal edge delivers for
        ``"interface"``); ``w`` weights each entry by ``1 / (rtol
        max|field|)`` at *state* (``rtol`` 1 under ``"l2"``); a field
        with no magnitude leaves the norm, as ``atol = 0`` makes it.
        ``rms`` divides the sum of squares by the count.
        """
        cfg = self.cfgs[gi]
        norm = cfg.get("convergence_norm", "l2")
        rtol_eff = 1.0 if norm == "l2" else float(cfg.get("rtol", 1e-6))
        members, _off, _k = self._group_layout(gi)
        x = np.concatenate([np.asarray(state[m]["x"], np.float64) for m in members])
        blocks, weights = [], []
        for B, _gamma in self.norm_fields(gi, raw=raw):
            ref = float(np.max(np.abs(B @ x))) if B.shape[0] else 0.0
            blocks.append(B)
            weights.append(np.full(B.shape[0], 1.0 / (rtol_eff * ref) if ref > 0 else 0.0))
        S = np.vstack(blocks) if blocks else np.zeros((0, len(x)))
        w = np.concatenate(weights) if weights else np.zeros(0)
        return S, w, rtol_eff, norm != "l2"

    def _field_slices(self, gi: int) -> list:
        """``[(rows of S, B, gamma), ...]``, one per field of :meth:`norm_fields`."""
        out, at = [], 0
        for B, gamma in self.norm_fields(gi):
            out.append((slice(at, at + B.shape[0]), B, gamma))
            at += B.shape[0]
        return out

    def group_report_consistency(self, gi: int, pre: dict, state: dict, residual: float):
        """``(defect_norm, bound, detail)``: is the reported residual that of the returned state?

        With ``F`` the group's one-pass map (:meth:`group_pass`) and ``Phi``
        its stationary map (every read of the same iterate), ``x - Phi(x)
        = -(I - L)(F(x) - x)`` exactly for an affine pass, and the float
        pass the library ran is ``F + (I - L)^{-1} epsilon``.  So the
        defect ``x - Phi(x)`` of the returned state, measured in the
        group's norm with the returned state's weights ``D``, is at most
        ``||D S (I - L) S^+ diag(rho_up)|| * R + ||D S epsilon||``:
        ``R`` the reported residual (as a raw 2-norm, times ``1 + (N + 4)
        eps`` for the float evaluation of the norm itself), ``rho_up`` an
        upper bound on the scale the residual divided each entry by
        (``rtol max(|x|, |F(x)| + |eta|)``), ``epsilon`` the members'
        rounding (:meth:`rounding`).  Holds whatever the acceleration, the
        predictor and the solver did to reach the state, converged or not.

        ``S`` is the norm's own reading (:meth:`norm_fields`), which under
        ``"interface"`` is what each internal edge delivers.  Every member
        reads the others only through those edges, so ``L = B S`` for some
        ``B`` and ``S (I - L) S^+`` applied to ``S r`` is ``(I - S B) S r``
        exactly, whether or not ``S`` has an inverse.  A mapped edge's
        delivered value is computed again by the norm, in float: both
        readings the residual compared carry that rounding (``gamma |S|
        |x|``, :meth:`norm_fields`), which is allowed for beside ``eta``.
        """
        members, off, k = self._group_layout(gi)
        L, U = self.group_pass(gi)
        cg_ = self.group_constant(gi, pre, state)
        x = np.concatenate([np.asarray(np.asarray(state[m]["x"]), LD) for m in members])
        F = _solve(np.eye(k, dtype=LD) - L, U @ x + cg_)
        phi = (L + U) @ x + cg_
        d = np.asarray(x - phi, np.float64)
        x = np.asarray(x, np.float64)
        F = np.asarray(F, np.float64)
        L = np.asarray(L, np.float64)
        read = {}
        for m in members:
            for i, _K in self.terms[m][2]:
                e = self.topo.edges[i]
                if self.topo.group_of(e.src) == gi:
                    a = np.abs(x[off[e.src]:off[e.src] + self.topo.node(e.src).n])
                    b = np.abs(F[off[e.src]:off[e.src] + self.topo.node(e.src).n])
                    read[(i,)] = np.maximum(a, b)
                else:
                    src = pre if i in self.back else state
                    read[(i,)] = np.asarray(src[e.src]["x"], np.float64)
        eps_vec = np.concatenate([self.rounding(m, pre, read) for m in members])
        eta = np.abs(np.linalg.inv(np.eye(k) - L)) @ eps_vec
        S, w, rtol_eff, rms = self.norm_parts(gi, state)
        absS = np.abs(S)
        fields = self._field_slices(gi)
        # Per entry of the reading: how far the float pass's reading can be
        # from the exact pass's (``|S| eta``), and how far a reading the
        # norm computed can be from the exact reading of the same state
        # (``read_x`` at the returned state, ``read_F`` after the pass).
        s_eta = absS @ eta
        read_x = np.zeros(len(w))
        read_F = np.zeros(len(w))
        rho_up = np.zeros(len(w))
        for rows, B, gamma in fields:
            read_x[rows] = gamma * (np.abs(B) @ np.abs(x))
            read_F[rows] = gamma * (np.abs(B) @ (np.abs(F) + eta))
            if B.shape[0]:
                rho_up[rows] = rtol_eff * max(
                    float(np.max(np.abs(B @ x) + read_x[rows])),
                    float(np.max(np.abs(B @ F) + s_eta[rows] + read_F[rows])))
        read = read_x + read_F
        Splus = np.linalg.pinv(S)
        A = (w[:, None] * S) @ (np.eye(k) - L) @ Splus @ np.diag(rho_up)
        K_up = float(np.linalg.norm(A, 2))
        N = len(w)
        R2 = float(residual) * (np.sqrt(N) if rms else 1.0) * (1.0 + (N + 4) * self.eps)
        A_read = np.abs((w[:, None] * S) @ (np.eye(k) - L) @ Splus)
        bound = (K_up * R2 + float(np.linalg.norm(w * (absS @ eps_vec)))
                 + float(np.linalg.norm(A_read @ read)))
        dnorm = float(np.linalg.norm(w * (S @ d)))
        scale = np.sqrt(N) if rms else 1.0
        # The other direction: the reported residual *is* ``||F(x) - x||`` of
        # the returned state (``coupling_diagnostics``: "for the state x this
        # step returned").  The float pass is within ``eta`` of ``F`` per
        # entry, which moves the norm by at most ``||D |S| eta||`` (``D`` the
        # returned-state weights, never below the residual's own), and the
        # two readings it compared by their own rounding; the residual's
        # weights move by at most ``(max |S| eta + reading) / ref`` per
        # field; the norm's own evaluation by ``(N + 4) eps``.
        r_true = F - x
        w_true = np.zeros(N)
        delta = 0.0
        for rows, B, _gamma in fields:
            if not B.shape[0]:
                continue
            ref = max(float(np.max(np.abs(B @ x))), float(np.max(np.abs(B @ F))))
            w_true[rows] = 1.0 / (rtol_eff * ref) if ref > 0 else 0.0
            if ref > 0:
                delta = max(delta, float(np.max(s_eta[rows] + np.maximum(
                    read_x[rows], read_F[rows]))) / ref)
        R_true = float(np.linalg.norm(w_true * (S @ r_true))) / scale
        R_tol = (float(np.linalg.norm(w * (s_eta + read))) / scale
                 + 2.0 * delta * R_true
                 + (N + 4) * self.eps * max(float(residual), R_true))
        return dnorm / scale, bound / scale, dict(K_up=K_up, eps=float(np.max(eps_vec)),
                                                  residual=float(residual), d=d,
                                                  residual_true=R_true, residual_tol=R_tol)

    def converged_distance(self, gi: int, pre: dict, state: dict, threshold: float):
        """``(distance, tolerance)``: the returned state against the group's exact fixed point.

        ``x - x* = (I - M_g)^{-1} (x - Phi(x))`` exactly, and a converged
        group's residual is at most its threshold (its error estimate is
        never below the residual, CPL-048), so the distance in the
        group's norm (returned-state weights) is at most
        ``||D S (I - M_g)^{-1} S^+ D^{-1}||`` times the defect bound of
        :meth:`group_report_consistency` taken at the threshold.
        """
        members, off, k = self._group_layout(gi)
        xs = self.group_fixed_point(gi, pre, state)
        x = np.concatenate([np.asarray(np.asarray(state[m]["x"]), LD) for m in members])
        xstar = np.concatenate([xs[m] for m in members])
        S, w, _rtol, rms = self.norm_parts(gi, state)
        N = len(w)
        scale = np.sqrt(N) if rms else 1.0
        dist = float(np.linalg.norm(w * (S @ np.asarray(x - xstar, np.float64)))) / scale
        _dn, bound_at_thr, _det = self.group_report_consistency(gi, pre, state, threshold)
        Winv = np.where(w > 0, 1.0 / np.where(w > 0, w, 1.0), 0.0)
        kappa = float(np.linalg.norm(
            (w[:, None] * S)
            @ np.linalg.inv(np.eye(k) - np.asarray(self.group_operator(gi), np.float64))
            @ np.linalg.pinv(S) @ np.diag(Winv), 2))
        return dist, kappa * bound_at_thr

    def returned_weight_distance(self, gi: int, pre: dict, state: dict) -> float:
        """The group's distance to its exact fixed point in its norm at the returned state.

        The norm ``spectral_error_bound`` documents (CPL-088, MADD-ANO-146):
        each field divided by its own ``max|field|`` at the returned state.
        """
        members, off, k = self._group_layout(gi)
        xs = self.group_fixed_point(gi, pre, state)
        x = np.concatenate([np.asarray(np.asarray(state[m]["x"]), LD) for m in members])
        xstar = np.concatenate([xs[m] for m in members])
        S, w, _rtol, rms = self.norm_parts(gi, state)
        scale = np.sqrt(len(w)) if rms else 1.0
        return float(np.linalg.norm(w * (S @ np.asarray(x - xstar, np.float64)))) / scale

    # -- the oracle ---------------------------------------------------------------

    def outside_reads(self, name, pre, state) -> dict:
        read = {}
        for i, _K in self.terms[name][2]:
            e = self.topo.edges[i]
            src = pre if i in self.back else state
            read[(i,)] = np.asarray(src[e.src]["x"], np.float64)
        return read

    def check_step(self, pre: dict, state: dict, reports: dict, *, thresholds=None,
                   where: str = "") -> dict:
        """Assert the step from *pre* to *state* is the documented one; return measurements.

        * every node outside the groups: its defect within its own rounding
          (the schedule's back edges, the edge kinds and the summation);
        * every reported group: its defect within the bound its reported
          residual gives (:meth:`group_report_consistency`), and, where it
          reports ``converged``, its distance to its exact fixed point within
          the tolerance its threshold gives (:meth:`converged_distance`);
        * the whole state against :meth:`monolithic`: ``|x - x_ref| <=
          |(I - M)^{-1}| a`` with ``a`` the defect allowances above.
        """
        topo = self.topo
        defects = self.defects(pre, state)
        allowance = {}
        out = {"outside": {}, "groups": {}}
        for nd in topo.nodes:
            if topo.group_of(nd.name) is not None:
                continue
            eps_i = self.rounding(nd.name, pre, self.outside_reads(nd.name, pre, state))
            d = np.abs(np.asarray(defects[nd.name], np.float64))
            allowance[nd.name] = eps_i
            assert np.all(d <= eps_i), (
                f"{where}: outside node {nd.name!r} is {d} from its update at the values the "
                f"documented schedule reads (rounding allows {eps_i}); back edges "
                f"{sorted(self.back)}")
            out["outside"][nd.name] = float(np.max(d))
        for gi, members in enumerate(topo.groups):
            rep = reports.get(gi)
            if rep is None:
                for m in members:
                    allowance[m] = None
                continue
            dnorm, bound, det = self.group_report_consistency(gi, pre, state, rep["residual"])
            gap = abs(float(rep["residual"]) - det["residual_true"])
            assert gap <= det["residual_tol"], (
                f"{where}: group {gi} {members} reports residual {rep['residual']:.6e}, but "
                f"||F(x) - x|| of the state it returned is {det['residual_true']:.6e} "
                f"(rounding allows {det['residual_tol']:.3e}; iterations="
                f"{rep.get('iterations')}, converged={rep.get('converged')})")
            assert dnorm <= bound, (
                f"{where}: group {gi} {members}: the returned state's defect is {dnorm:.4e} in "
                f"its norm, beyond the {bound:.4e} its reported residual "
                f"{rep['residual']:.4e} allows (iterations={rep.get('iterations')}, "
                f"converged={rep.get('converged')}, K={det['K_up']:.3g})")
            S, w, _rt, rms = self.norm_parts(gi, state)
            scale = np.sqrt(len(w)) if rms else 1.0
            _m, off, _k = self._group_layout(gi)
            for m in members:
                # ``||w (S d)|| <= bound scale``, so each entry the norm
                # reads of this member's defect is within ``bound scale /
                # w``; the defect itself is those entries through the
                # pseudo-inverse of the member's rows -- where they
                # determine it.  A member the norm reads through a mapping
                # that loses a direction (or does not read at all) has no
                # allowance, and the monolithic comparison is skipped.
                sl = slice(off[m], off[m] + topo.node(m).n)
                rows = (S[:, sl] != 0).any(axis=1) & (w > 0)
                C = S[rows][:, sl]
                if C.size and np.linalg.matrix_rank(C) == topo.node(m).n:
                    allowance[m] = np.abs(np.linalg.pinv(C)) @ (bound * scale / w[rows])
                else:
                    allowance[m] = None
            entry = {"defect": dnorm, "bound": bound}
            if rep.get("converged") and thresholds is not None:
                dist, tol = self.converged_distance(gi, pre, state, thresholds[gi])
                assert dist <= tol, (
                    f"{where}: group {gi} {members} reports converged at {dist:.4e} from its "
                    f"exact fixed point in its norm, beyond the {tol:.4e} its threshold "
                    f"{thresholds[gi]} allows")
                entry["distance"], entry["tolerance"] = dist, tol
            out["groups"][gi] = entry
        if all(a is not None for a in allowance.values()):
            ref = self.monolithic(pre)
            R = np.abs(self.distance_operator())
            a = np.concatenate([allowance[nd.name] for nd in topo.nodes])
            lim = self._split(R @ a)
            for nd in topo.nodes:
                gap = np.abs(np.asarray(
                    np.asarray(np.asarray(state[nd.name]["x"]), LD) - ref[nd.name], np.float64))
                # ``lim`` is evaluated in float64: 64 ulps cover that evaluation.
                assert np.all(gap <= lim[nd.name] * (1.0 + 64 * EPS64)), (
                    f"{where}: node {nd.name!r} is {gap} from the monolithic float64 solve, "
                    f"beyond {lim[nd.name]}")
            out["monolithic"] = True
        return out


def check_leaves(topo: Topology, state: dict, step: int, dividers: Optional[dict] = None,
                 where: str = "") -> None:
    """Every non-float leaf at its closed form after *step* steps (sub-steps counted)."""
    dividers = dividers or {}
    for nd in topo.nodes:
        updates = step * dividers.get(nd.name, 1)
        for field, want in expected_leaves(nd, updates).items():
            got = np.asarray(state[nd.name][field])
            want = np.asarray(want)
            assert got.dtype == want.dtype and got.tobytes() == want.tobytes(), (
                f"{where}: {nd.name}.{field} = {got!r} after {updates} updates, want {want!r}")


# ---------------------------------------------------------------------------
# The topology generator
# ---------------------------------------------------------------------------


class TopologyBuilder:
    """Assemble a :class:`Topology` by name: nodes, edges (ports assigned) and groups.

    ``edge(src, dst)`` opens a new port on *dst*; ``port=`` adds to an
    existing one (every edge into a shared port is additive).  An edge
    between nodes of different sizes is mapped; ``mapped=True`` maps one
    between equal sizes too.
    """

    def __init__(self):
        self._nodes: dict = {}
        self._edges: list = []
        self._groups: list = []
        self._ports: dict = {}

    def node(self, name, n=1, *, alpha=0.5, beta=0.0, leaves=(), flux=False,
             three_arg=False) -> str:
        assert name not in self._nodes, name
        self._nodes[name] = dict(name=name, n=n, alpha=alpha, beta=beta, leaves=tuple(leaves),
                                 flux=flux, three_arg=three_arg)
        self._ports[name] = 0
        return name

    def edge(self, src, dst, *, field="x", transform=None, mapped=None, port=None) -> int:
        if port is None:
            port = self._ports[dst]
            self._ports[dst] += 1
            additive = False
        else:
            additive = True
            for k, e in enumerate(self._edges):
                if e.dst == dst and e.port == port and not e.additive:
                    self._edges[k] = dataclasses.replace(e, additive=True)
        if mapped is None:
            mapped = self._nodes[src]["n"] != self._nodes[dst]["n"]
        self._edges.append(TEdge(src, dst, port, field, transform, additive, bool(mapped)))
        return port

    def group(self, *members) -> int:
        self._groups.append(tuple(members))
        return len(self._groups) - 1

    def build(self, label="") -> Topology:
        nodes = tuple(TNode(ports=self._ports[name], **spec) for name, spec in self._nodes.items())
        topo = Topology(nodes, tuple(self._edges), tuple(self._groups), label)
        topo.check()
        return topo


def _nested_cycle_shape(b: TopologyBuilder, members, *, transform=None):
    """A ring through *members* plus a chord closing a shorter inner cycle."""
    m = len(members)
    for i in range(m):
        b.edge(members[i], members[(i + 1) % m])
    if m >= 3:
        b.edge(members[m - 1], members[1], transform=transform)


def _star_shape(b: TopologyBuilder, hub, leaves, *, flux_leaf=None):
    """``hub <-> leaf`` for every leaf (a flux edge back from *flux_leaf*)."""
    for leaf in leaves:
        b.edge(hub, leaf)
        b.edge(leaf, hub, field="q" if leaf == flux_leaf else "x")


def named_topologies() -> dict:
    """The per-push structures, each a shape the coupled_graphs generator never draws.

    * ``nested-loop-additive``: a nested-cycle group (a ring and an inner
      chord) with members of two sizes joined by mapped edges, an outside
      node inside its strongly connected component (so ``compile()`` warns
      and one edge of that loop is staggered), three additive edges into
      one port, a forward flux edge between two drivers, a three-argument
      node, and a reader added before the group's members (CPL-025);
    * ``two-groups-one-scc``: a ring and a star joined both ways (two groups
      in one component, both warn), a flux edge inside the star, two
      additive edges into a three-argument member, a mapped reader;
    * ``star-and-ungrouped-ring``: a star group feeding an ungrouped ring
      (a cycle outside every group, staggered by its build order) and a
      reader;
    * ``chain-into-ring``: a chain of outside nodes into a two-member ring,
      a node beside the group reading the chain and the group additively,
      and a reader added first.  The ring's forward edge carries an
      interface mapping and then a transform, its return edge a transform
      alone, so configuration 2 -- the interface norm -- reads one edge of
      each kind as the step delivers it.
    """
    out = {}

    b = TopologyBuilder()
    b.node("early", 2, alpha=0.5, leaves=("key",))
    b.node("d0", 1, alpha=1.0, beta=1.0, flux=True)
    b.node("d1", 1, alpha=0.25, leaves=("tag",))
    b.node("d2", 1, alpha=-0.5, beta=0.5, three_arg=True)
    b.node("g0", 2, alpha=0.5, beta=1.0, leaves=("count", "tag"))
    b.node("g1", 2, alpha=-0.25, leaves=("flag", "key"))
    b.node("g2", 1, alpha=0.0, beta=-0.5)
    b.node("o", 2, alpha=0.5, leaves=("count",), three_arg=True)
    b.edge("d0", "d1", field="q")
    _nested_cycle_shape(b, ("g0", "g1", "g2"), transform="negate")
    b.edge("g2", "o")
    b.edge("o", "g1", transform="scale_0.5")
    p = b.edge("d0", "g0", transform="scale_2.0")
    b.edge("d1", "g0", port=p)
    b.edge("d2", "g0", port=p, transform="scale_0.5")
    b.edge("g1", "early", transform="scale_0.5")
    b.group("g0", "g1", "g2")
    out["nested-loop-additive"] = b.build("nested-loop-additive")

    b = TopologyBuilder()
    b.node("u", 2, alpha=1.0, beta=1.0, leaves=("count",))
    b.node("a0", 2, alpha=0.5, three_arg=True)
    b.node("a1", 2, alpha=0.0, beta=1.0, leaves=("key", "flag"))
    b.node("b0", 2, alpha=0.25, leaves=("tag",))
    b.node("b1", 2, alpha=0.0, flux=True)
    b.node("b2", 2, alpha=-0.5, leaves=("count",))
    b.node("r", 1, alpha=0.5, leaves=("tag",))
    b.edge("a0", "a1")
    b.edge("a1", "a0")
    _star_shape(b, "b0", ("b1", "b2"), flux_leaf="b1")
    b.edge("a1", "b0")
    p = b.edge("b2", "a0")
    b.edge("u", "a0", port=p, transform="negate")
    b.edge("b1", "r")
    b.group("a0", "a1")
    b.group("b0", "b1", "b2")
    out["two-groups-one-scc"] = b.build("two-groups-one-scc")

    b = TopologyBuilder()
    b.node("h", 2, alpha=0.5, beta=1.0, leaves=("count", "key"))
    b.node("l0", 2, alpha=0.0)
    b.node("l1", 1, alpha=0.25, leaves=("flag",))
    b.node("l2", 2, alpha=-0.25, three_arg=True)
    b.node("r0", 2, alpha=0.5, leaves=("tag",))
    b.node("r1", 2, alpha=0.25)
    b.node("z", 1, alpha=0.5, leaves=("key",))
    _star_shape(b, "h", ("l0", "l1", "l2"))
    b.edge("l0", "r0", transform="scale_2.0")
    b.edge("r0", "r1")
    b.edge("r1", "r0")
    b.edge("r1", "z")
    b.group("h", "l0", "l1", "l2")
    out["star-and-ungrouped-ring"] = b.build("star-and-ungrouped-ring")

    b = TopologyBuilder()
    b.node("q", 1, alpha=0.5, leaves=("flag",))
    b.node("c0", 1, alpha=1.0, beta=1.0, flux=True, leaves=("count",))
    b.node("c1", 2, alpha=0.5, three_arg=True)
    b.node("c2", 2, alpha=0.0, leaves=("key",))
    b.node("g0", 2, alpha=0.5, beta=1.0, leaves=("tag",))
    b.node("g1", 2, alpha=-0.5)
    b.node("w", 2, alpha=0.25, leaves=("count", "flag"))
    b.edge("c0", "c1", field="q")
    b.edge("c1", "c2", transform="negate")
    b.edge("c2", "g0")
    b.edge("g0", "g1", transform="negate", mapped=True)
    b.edge("g1", "g0", transform="scale_0.5")
    p = b.edge("c1", "w")
    b.edge("g0", "w", port=p)
    b.edge("g1", "q")
    b.group("g0", "g1")
    out["chain-into-ring"] = b.build("chain-into-ring")
    return out


def topology_knobs(topo: Topology, choice: int = 0) -> list:
    """One quiet, converging configuration per group, rotated by *choice*.

    Every acceleration, both schedules and every norm appear across the
    groups and choices; the interface norm is never paired with a
    group-internal flux edge (``compile()`` refuses it, CPL-075), and
    Aitken and fixed relaxation are never paired with a typed PRNG key in
    a member (that pairing fails to trace: see the strict xfail in
    ``test_differential_coupling_topologies.py``).
    """
    table = [
        dict(acceleration="none", iteration_mode="gauss-seidel", convergence_norm="l2",
             tolerance=1e-6),
        dict(acceleration="iqn-ils", iteration_mode="jacobi", convergence_norm="mixed",
             rtol=1e-4),
        dict(acceleration="aitken", iteration_mode="gauss-seidel", convergence_norm="interface",
             rtol=1e-4),
        dict(acceleration="fixed", relaxation=0.7, iteration_mode="jacobi",
             convergence_norm="l2", tolerance=1e-6),
        dict(acceleration="iqn-imvj", jacobian_reuse=2, iteration_mode="gauss-seidel",
             convergence_norm="mixed", rtol=1e-4, predictor="linear"),
    ]
    out = []
    for gi, members in enumerate(topo.groups):
        has_flux = any(topo.edges[i].field == "q" for i in topo.internal_edges(gi))
        has_key = any("key" in topo.node(m).leaves for m in members)
        k = (choice + 2 * gi) % len(table)
        while True:
            cfg = dict(table[k], max_iterations=200)
            bad = (has_flux and cfg["convergence_norm"] == "interface") or (
                has_key and cfg["acceleration"] in ("aitken", "fixed"))
            if not bad:
                break
            k = (k + 1) % len(table)
        out.append(cfg)
    return out


def group_cfgs_of(knobs) -> list:
    """:class:`LinearModel`'s per-group view of ``CouplingGroup`` configurations."""
    return [dict(iteration_mode=g.get("iteration_mode", "gauss-seidel"),
                 subcycling=bool(g.get("subcycling", False)),
                 boundary_interpolation=g.get("boundary_interpolation", "linear"),
                 convergence_norm=g.get("convergence_norm", "l2"),
                 rtol=g.get("rtol", 1e-6), tolerance=g.get("tolerance", 1e-6)) for g in knobs]


def thresholds_of(knobs) -> list:
    return [float(g.get("tolerance", 1e-6)) if g.get("convergence_norm", "l2") == "l2" else 1.0
            for g in knobs]


def invariant_orders(topo: Topology):
    """``(node groups that keep their relative order, edge groups that keep theirs)``.

    The documented exceptions to build-order invariance (CPL-181): a
    group's members keep their relative order (a Gauss-Seidel sweep follows
    it, CPL-077); so does every node of a strongly connected component that
    is not exactly one group -- an ungrouped cycle, or a loop through a
    group and outside nodes -- because which of its edges is read from the
    previous step follows that order; and three or more additive edges
    into one port keep theirs, because their sum does.
    """
    comp = strongly_connected_components(list(topo.names), topo.edges)
    keep_nodes = [list(g) for g in topo.groups]
    by_comp: dict = {}
    for name in topo.names:
        by_comp.setdefault(comp[name], []).append(name)
    for members in by_comp.values():
        if len(members) == 1:
            continue
        if any(set(members) == set(g) for g in topo.groups):
            continue
        keep_nodes.append(members)
    keep_edges = []
    ports: dict = {}
    for i, e in enumerate(topo.edges):
        ports.setdefault((e.dst, e.port), []).append(i)
    for idx in ports.values():
        if len(idx) >= 3:
            keep_edges.append(idx)
    return keep_nodes, keep_edges


def merge_keep(groups: list) -> list:
    """Overlapping keep-groups merged, so one relative order is imposed on their union."""
    merged: list = []
    for g in groups:
        g = list(g)
        hit = [m for m in merged if set(m) & set(g)]
        for m in hit:
            merged.remove(m)
            g = [x for x in m if x not in g] + g
        merged.append(g)
    return merged


def ordered_permutation(seq: Sequence, keep: list, rng: np.random.Generator) -> list:
    """A random permutation of *seq* in which each list in *keep* keeps its order in *seq*."""
    keep = merge_keep(keep)
    perm = list(seq)
    rng.shuffle(perm)
    for group in keep:
        order = [x for x in seq if x in set(group)]
        slots = sorted(perm.index(x) for x in order)
        for pos, x in zip(slots, order):
            perm[pos] = x
    return perm


def with_identity_relay(topo: Topology, edge_index: int):
    """*topo* with an identity relay ``rly`` on internal edge *edge_index*, added after its source.

    The edge's field and transform move onto ``src -> rly``; ``rly -> dst``
    is plain, on the original port (additive as it was).  ``rly`` joins the
    group straight after the source, and the build order straight after
    the source too.  Mapped edges are not relayed (the relay's start value
    would be a float matrix product the reference cannot reproduce bit for
    bit).
    """
    e = topo.edges[edge_index]
    assert topo.internal(e) and not e.mapped, e
    src, dst = topo.node(e.src), topo.node(e.dst)
    relay = TNode("rly", dst.n, ports=1, alpha=0.0, beta=0.0)
    nodes = list(topo.nodes)
    nodes.insert([nd.name for nd in nodes].index(e.src) + 1, relay)
    edges = list(topo.edges)
    edges[edge_index] = TEdge(e.src, "rly", 0, e.field, e.transform, False, False)
    edges.insert(edge_index + 1, TEdge("rly", e.dst, e.port, "x", None, e.additive, False))
    groups = []
    for g in topo.groups:
        g = list(g)
        if e.src in g:
            g.insert(g.index(e.src) + 1, "rly")
        groups.append(tuple(g))
    del src
    return Topology(tuple(nodes), tuple(edges), tuple(groups), topo.label + "+relay")


def relay_values(topo: Topology, values: dict, relayed: Topology, edge_index: int) -> dict:
    """*values* for the relayed topology: the relay is ``x <- I u``, started at what it relays."""
    e = topo.edges[edge_index]
    n = topo.node(e.dst).n
    dtype = np.asarray(values["nodes"][e.src]["x0"]).dtype
    fac = TRANSFORM_FACTORS[e.transform] * (2.0 if e.field == "q" else 1.0)
    out = {"nodes": dict(values["nodes"]), "H": {}}
    out["nodes"]["rly"] = {"G": [np.eye(n, dtype=dtype)], "b": np.zeros(n, dtype),
                           "x0": np.asarray(np.asarray(values["nodes"][e.src]["x0"]) * dtype.type(fac),
                                            dtype)}
    # Re-index the mapping matrices past the inserted edge.
    for i, H in values["H"].items():
        out["H"][i if i <= edge_index else i + 1] = H
    return out


def interleaved_order(topo: Topology) -> tuple:
    """A build order made of the shapes the schedule has got wrong before.

    Every node downstream of a cycle is added first, before the cycle it
    reads (CPL-025, MADD-ANO-120); every outside node of a group's
    strongly connected component is added between the group's first two
    members (MADD-ANO-144).  The members keep their relative order.
    """
    comp = strongly_connected_components(list(topo.names), topo.edges)
    sizes: dict = {}
    for n in topo.names:
        sizes[comp[n]] = sizes.get(comp[n], 0) + 1
    cyclic = {n for n in topo.names if sizes[comp[n]] > 1}
    succ: dict = {n: [] for n in topo.names}
    for e in topo.edges:
        succ[e.src].append(e.dst)
    downstream, todo = set(), list(cyclic)
    while todo:
        n = todo.pop()
        for m in succ[n]:
            if m not in cyclic and m not in downstream:
                downstream.add(m)
                todo.append(m)
    order = [n for n in topo.names if n not in downstream]
    for gi, members in enumerate(topo.groups):
        cid = comp[members[0]]
        outside = [n for n in topo.names if comp[n] == cid and topo.group_of(n) != gi]
        if not outside or len(members) < 2:
            continue
        order = [n for n in order if n not in outside]
        at = order.index(members[0]) + 1
        order[at:at] = outside
    return tuple([n for n in topo.names if n in downstream] + order)
