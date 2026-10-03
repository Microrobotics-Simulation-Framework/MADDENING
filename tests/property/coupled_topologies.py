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
from tests.property import coupled_graphs as cg

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
    rescaled around still has room to reach its drawn rate.
    """
    out = {}
    for k, nd in enumerate(topo.nodes):
        if not nd.three_arg:
            continue
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
            key = built.topo.group_key(gi, rename)
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


def _norm_rows(topo: Topology, gi: int, norm: str) -> list:
    """The norm's entries, as ``(node, index)``: every member's ``x``, or each internal edge's source."""
    if norm == "interface":
        return [(topo.edges[i].src, k) for i in topo.internal_edges(gi)
                for k in range(topo.node(topo.edges[i].src).n)]
    return [(m, k) for m in topo.groups[gi] for k in range(topo.node(m).n)]


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

    def norm_parts(self, gi: int, state: dict, F: Optional[np.ndarray] = None):
        """``(S, w, rtol_eff, rms)`` of the group's norm at the returned state.

        ``S`` selects the norm's entries from the members' stacked ``x``
        (each member once for ``"l2"`` / ``"mixed"``, each internal edge's
        source for ``"interface"``); ``w`` weights each entry by
        ``1 / (rtol max|field|)`` at *state* (``rtol`` 1 under ``"l2"``);
        a field with no magnitude leaves the norm, as ``atol = 0`` makes
        it.  ``rms`` divides the sum of squares by the count.
        """
        cfg = self.cfgs[gi]
        norm = cfg.get("convergence_norm", "l2")
        rtol_eff = 1.0 if norm == "l2" else float(cfg.get("rtol", 1e-6))
        members, off, k = self._group_layout(gi)
        rows = _norm_rows(self.topo, gi, norm)
        S = np.zeros((len(rows), k))
        w = np.zeros(len(rows))
        for r, (node, idx) in enumerate(rows):
            S[r, off[node] + idx] = 1.0
            ref = float(np.max(np.abs(np.asarray(state[node]["x"], np.float64))))
            w[r] = 1.0 / (rtol_eff * ref) if ref > 0 else 0.0
        return S, w, rtol_eff, norm != "l2"

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
        rho_up = np.zeros(len(w))
        rows = _norm_rows(self.topo, gi, self.cfgs[gi].get("convergence_norm", "l2"))
        for r, (node, idx) in enumerate(rows):
            sl = slice(off[node], off[node] + self.topo.node(node).n)
            rho_up[r] = rtol_eff * max(float(np.max(np.abs(x[sl]))),
                                       float(np.max(np.abs(F[sl]) + eta[sl])))
        Splus = np.linalg.pinv(S)
        A = (w[:, None] * S) @ (np.eye(k) - L) @ Splus @ np.diag(rho_up)
        K_up = float(np.linalg.norm(A, 2))
        N = len(w)
        R2 = float(residual) * (np.sqrt(N) if rms else 1.0) * (1.0 + (N + 4) * self.eps)
        bound = K_up * R2 + float(np.linalg.norm(w * (S @ eps_vec)))
        dnorm = float(np.linalg.norm(w * (S @ d)))
        scale = np.sqrt(N) if rms else 1.0
        return dnorm / scale, bound / scale, dict(K_up=K_up, eps=float(np.max(eps_vec)),
                                                  residual=float(residual), d=d)

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
            assert dnorm <= bound, (
                f"{where}: group {gi} {members}: the returned state's defect is {dnorm:.4e} in "
                f"its norm, beyond the {bound:.4e} its reported residual "
                f"{rep['residual']:.4e} allows (iterations={rep.get('iterations')}, "
                f"converged={rep.get('converged')}, K={det['K_up']:.3g})")
            S, w, _rt, rms = self.norm_parts(gi, state)
            scale = np.sqrt(len(w)) if rms else 1.0
            _m, off, _k = self._group_layout(gi)
            for m in members:
                sl = slice(off[m], off[m] + topo.node(m).n)
                wm = np.max(w[(S[:, sl] != 0).any(axis=1)]) if np.any(S[:, sl]) else 0.0
                allowance[m] = (np.full(topo.node(m).n, bound * scale / wm)
                                if wm > 0 else None)
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


def permutations_keeping(seq: Sequence, fixed_relative: Sequence[Sequence], rng) -> list:
    """A random permutation of *seq* in which each list in *fixed_relative* keeps its order."""
    perm = list(seq)
    rng.shuffle(perm)
    for group in fixed_relative:
        slots = iter(sorted((perm.index(x) for x in group)))
        positions = list(slots)
        for pos, x in zip(positions, group):
            perm[pos] = x
    return perm


