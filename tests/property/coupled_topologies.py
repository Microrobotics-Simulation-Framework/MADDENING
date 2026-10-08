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

**The mapping kind** (``build(..., mapping_kind=)``, :data:`MAPPING_KINDS`)
says how a mapped edge holds that ``H``: as the dense matrix (``"matrix"``,
the default and the code path every caller had), or as a
``StaticSparseMapping`` over a sparsity pattern fixed per structure
(:func:`mapping_pattern`).  The pattern is structure -- a sparse mapping's
index is baked into the compiled step -- so it cannot be drawn per example
on one compiled graph; :func:`draw_values` zeroes the drawn ``H`` outside
it instead, before a group's spectral radius is rescaled, and the model
below is handed that same ``H``.  The reference, the oracle and
:class:`LinearModel` are therefore the dense ones, unchanged: a sparse edge
must reproduce what a dense edge with the same matrix does.  That is a
statement about the *step*.  The two edges do not hold the same
*parameters*: a dense edge holds every entry of its matrix, a zero one
included, and a sparse edge only its pattern's, so an oracle that
differentiates with respect to "every mapping weight" asks
:func:`parameter_entries` which entries those are.

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

**Geometry-dependent mappings** (:class:`Geometry`).  A mapped edge may be
built as an edge whose mapping reads a moving geometry held in the state of
its source or target relay, in one of two ways:

* ``"moving"``: the mapping is ``test_geom_matrix`` and the geometry *is*
  the matrix, a second state field ``M<edge>`` of the relay that holds it,
  updated ``M <- M + dM`` with ``dM`` a per-relay parameter.  ``M`` depends
  on nothing else, so a step is still one linear system in ``x``; the
  reference (:meth:`LinearModel.bind`) repeats the same float additions in
  the graph's dtype, so its matrices are the graph's bit for bit, and takes
  for every edge the matrix the documented time level names: the source's
  new matrix on a forward or group-internal edge and its pre-step one on a
  back edge; the target's pre-step matrix, and at sub-step ``k`` of a
  sub-cycled target ``pre + k dM``.  Both ends of the edge hold a field of
  that name and shape, with different values, so an anchor read from the
  wrong end is another matrix.
* ``"frozen"``: the mapping is the library's ``multilinear_grid`` kind
  between a one-dimensional grid and points held in a field ``P<edge>``
  that no update moves; ``values["H"]`` is then the fixed matrix of that
  gather or scatter and every oracle of the static case applies unchanged.

:func:`single_pass` restates one coupling pass with explicit loops over the
members and their sub-steps, for the one case a fixed point cannot show:
what a sub-cycled member reads *during* a pass.
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
from maddening.core.coupling.sparse_mapping import (
    StaticSparseMapping,
    sparse_matrix_mapping,
)
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

#: How a mapped edge holds its matrix.  ``"matrix"`` is the dense
#: ``matrix_mapping``; the others are a ``StaticSparseMapping`` over a
#: pattern fixed per structure (:func:`mapping_pattern`):
#:
#: * ``"sparse-full"``: every entry, in the gather layout -- every row
#:   full, so no slot is padded and any dense ``H`` is its weights;
#: * ``"sparse-ragged"``: a seeded pattern with rows of different lengths
#:   (one row full, the others shorter where the sizes allow), in the
#:   gather layout: padded slots, masked;
#: * ``"sparse-scatter"``: the ragged pattern in the scatter layout (a row
#:   lists the targets of a *source*).
MAPPING_KINDS = ("matrix", "sparse-full", "sparse-ragged", "sparse-scatter")
SPARSE_KINDS = MAPPING_KINDS[1:]

#: Mapping kinds whose matrix is *local*: an interpolation between a small
#: field and a large one, two entries per entry of the small side
#: (:func:`mapping_pattern`), so a scatter leaves most of its target
#: untouched -- the shape on which a reading of the large side dilutes a
#: norm.  ``"matrix-local"`` holds it dense, ``"sparse-local"`` as a
#: ``StaticSparseMapping`` in its natural layout (a scatter lists each
#: source's targets, anything else each target's sources).  Kept out of
#: :data:`MAPPING_KINDS`, which the differential grids iterate.
LOCAL_KINDS = ("matrix-local", "sparse-local")

#: Where ``convergence_norm="interface"`` reads a mapped internal edge, as
#: **this tree's library** does it: ``"delivered"`` (every edge at the
#: value it delivers).  The decision of 2026-10-07 is ``"compact"``: a
#: mapping whose target is larger than its source is read at its source
#: value, before the mapping and the transform; a tie and a smaller target
#: at the delivered value (:meth:`LinearModel.interface_side_of`).  The
#: library change makes this one line ``"compact"``; every reference that
#: does not name a rule follows it.
INTERFACE_SIDE = "delivered"

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


def geometry_dtype(dtype) -> jnp.dtype:
    """The dtype of a relay's geometry fields: the relay's own when that is
    float32 or float64 (what a geometry may be), float32 for a 16-bit relay."""
    dtype = jnp.dtype(dtype)
    return dtype if dtype in (jnp.dtype("float32"), jnp.dtype("float64")) else jnp.dtype("float32")


class TRelay(SimulationNode):
    """``x <- alpha x_pre + sum_j G_j u_j + b + beta dt``, at any float dtype.

    The arithmetic of :class:`~tests.property.coupled_graphs.Relay` in the
    same order (so a float32 ``TRelay`` evaluates the same expression), with
    a typed PRNG key among the leaves.
    """

    def __init__(self, name, timestep, *, n, ports, alpha=0.0, beta=0.0, leaves=(),
                 dtype="float32", constants=None, geom=()):
        dt_ = _dt(dtype)
        params = {f"G{j}": jnp.zeros((n, n), dt_) for j in range(ports)}
        params["b"] = jnp.zeros(n, dt_)
        # Geometry fields ``(name, shape, moves)``: state a mapping reads.  A
        # moving one advances by the parameter ``d<name>`` on every update.
        self._geom = tuple(geom)
        self._geom_dtype = geometry_dtype(dt_)
        for field, shape, moves in self._geom:
            if moves:
                params[f"d{field}"] = jnp.zeros(shape, self._geom_dtype)
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
        for field, shape, _moves in self._geom:
            s[field] = jnp.zeros(shape, self._geom_dtype)
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
        for field, _shape, moves in self._geom:
            out[field] = state[field] + p[f"d{field}"] if moves else state[field]
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
    #: Geometry fields ``(name, shape, moves)`` (see :class:`Geometry`).
    geom: tuple = ()


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
                          beta=nd.beta, leaves=nd.leaves, dtype=dtype, constants=constants,
                          geom=nd.geom)


@dataclasses.dataclass(frozen=True)
class Geometry:
    """How :func:`build` realises mapped edges as geometry-dependent mappings.

    ``kind`` is ``"moving"`` or ``"frozen"`` (module docstring).  ``anchors``
    maps the index of a mapped edge to ``"source"`` or ``"target"``: the end
    whose state the edge reads its geometry from; a mapped edge it does not
    name stays a static ``matrix_mapping``.  ``modes`` (``"frozen"`` only)
    maps an edge to ``"consistent"`` (the source is the grid, gathered at
    the points) or ``"conservative"`` (the target is the grid, scattered
    onto); consistent when not named.
    """

    kind: str
    anchors: tuple          # ((edge index, anchor), ...)
    modes: tuple = ()       # ((edge index, mode), ...)

    def __post_init__(self):
        assert self.kind in ("moving", "frozen"), self.kind
        assert all(a in ("source", "target") for _i, a in self.anchors), self.anchors

    @property
    def anchor(self) -> dict:
        return dict(self.anchors)

    def mode(self, i: int) -> str:
        return dict(self.modes).get(i, "consistent")

    def field(self, i: int) -> str:
        """The state field edge *i* reads, held by both of its ends."""
        return f"{'M' if self.kind == 'moving' else 'P'}{i}"

    def shape(self, topo: "Topology", i: int) -> tuple:
        e = topo.edges[i]
        n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
        if self.kind == "moving":
            return (n_dst, n_src)
        return (n_dst, 1) if self.mode(i) == "consistent" else (n_src, 1)

    def holder(self, topo: "Topology", i: int) -> str:
        e = topo.edges[i]
        return e.src if self.anchor[i] == "source" else e.dst


def with_geometry(topo: Topology, geometry: Optional[Geometry]) -> Topology:
    """*topo* with the geometry fields of *geometry* on both ends of each edge it names."""
    if geometry is None:
        return topo
    extra: dict = {}
    for i in sorted(geometry.anchor):
        e = topo.edges[i]
        assert e.mapped, f"edge {i} ({e}) is not mapped"
        for name in dict.fromkeys((e.src, e.dst)):
            extra.setdefault(name, []).append(
                (geometry.field(i), geometry.shape(topo, i), geometry.kind == "moving"))
    return dataclasses.replace(topo, nodes=tuple(
        dataclasses.replace(nd, geom=nd.geom + tuple(extra.get(nd.name, ())))
        for nd in topo.nodes))


def _geometry_mapping(topo: Topology, geometry: Geometry, i: int):
    """The geometry-dependent mapping of edge *i* (kinds imported when used)."""
    from tests.property import geometry_graphs as gg  # noqa: PLC0415

    e = topo.edges[i]
    n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
    if geometry.kind == "moving":
        return gg.geom_matrix_mapping(n_dst, n_src)
    if geometry.mode(i) == "consistent":
        return gg.multilinear((0.0,), (1.0,), (n_src,), n_points=n_dst, mode="consistent")
    return gg.multilinear((0.0,), (1.0,), (n_dst,), n_points=n_src, mode="conservative")


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
    mapping_kind: str = "matrix"
    #: ``{edge index: Slots}`` for every mapped edge of a sparse build.
    slots: dict = dataclasses.field(default_factory=dict)
    geometry: Optional[Geometry] = None


def mapping_pattern(topo: Topology, edge_index: int, mapping_kind: str) -> Optional[np.ndarray]:
    """The entries mapped edge *edge_index* holds under *mapping_kind*, as a
    boolean ``(n_target, n_source)`` mask; ``None`` for ``"matrix"`` (all).

    Fixed per structure, like :func:`fixed_constants`: seeded by the edge's
    place among the mapped edges and its shape -- not by a name or an edge
    index -- so a renamed topology, or one with a relay inserted, holds the
    same pattern.
    """
    assert mapping_kind in MAPPING_KINDS + LOCAL_KINDS, mapping_kind
    if mapping_kind == "matrix":
        return None
    e = topo.edges[edge_index]
    assert e.mapped, e
    n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
    if mapping_kind in LOCAL_KINDS:
        return local_pattern(topo, edge_index)
    if mapping_kind == "sparse-full":
        return np.ones((n_dst, n_src), dtype=bool)
    place = sum(1 for other in topo.edges[:edge_index] if other.mapped)
    rng = np.random.default_rng(20_000 + 131 * place + 17 * n_dst + n_src)
    mask = np.zeros((n_dst, n_src), dtype=bool)
    # Ragged: one row full and every other row a proper, non-empty subset,
    # so the shorter rows are padded; a single row is a proper subset
    # itself (where it has more than one column to choose from).
    full = int(rng.integers(0, n_dst)) if n_dst > 1 else -1
    for i in range(n_dst):
        size = n_src if i == full or n_src == 1 else int(rng.integers(1, n_src))
        mask[i, rng.choice(n_src, size=size, replace=False)] = True
    return mask


def parameter_entries(topo: Topology, edge_index: int, mapping_kind: str) -> np.ndarray:
    """The entries of mapped edge *edge_index*'s matrix that a graph built
    under *mapping_kind* holds as parameters: a boolean ``(n_target,
    n_source)`` mask.

    :func:`build` gives a ``"matrix"`` or a ``"matrix-local"`` edge the
    dense ``matrix_mapping``: every entry is a weight of the graph, also
    where the drawn matrix is zero (a local matrix outside its pattern),
    and a user can write any of them and ask for a derivative with respect
    to it.  Every other kind is a ``StaticSparseMapping`` laid out from
    :func:`mapping_pattern`: the pattern's entries are its weights -- one
    whose drawn weight happens to be zero included -- and an entry outside
    it is structure.  The graph holds no number there, so nothing can be
    differentiated with respect to one and no reported bound speaks of it.

    Not ``values["H"] != 0`` (a weight at zero is still a weight) and not
    :func:`mapping_pattern` alone (it also says where a *dense* local
    matrix is drawn non-zero, which is not what that edge holds).
    """
    assert mapping_kind in MAPPING_KINDS + LOCAL_KINDS, mapping_kind
    e = topo.edges[edge_index]
    assert e.mapped, e
    if mapping_kind in ("matrix", "matrix-local"):
        return np.ones((topo.node(e.dst).n, topo.node(e.src).n), dtype=bool)
    return mapping_pattern(topo, edge_index, mapping_kind)


def local_pattern(topo: Topology, edge_index: int) -> np.ndarray:
    """The ``(n_target, n_source)`` mask of a local mapping on edge *edge_index*.

    Every entry of the smaller side touches two neighbouring entries of
    the larger one, at a seeded place (two markers may share a place): a
    gather's rows have two entries each and a scatter's *columns* do, so
    a scatter onto a target of more than twice its source's size leaves
    rows empty.  Between equal sizes every row holds its own entry and the
    next.  Seeded like :func:`mapping_pattern`, by place and shape.
    """
    e = topo.edges[edge_index]
    n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
    place = sum(1 for other in topo.edges[:edge_index] if other.mapped)
    rng = np.random.default_rng(30_000 + 131 * place + 17 * n_dst + n_src)
    small, large = min(n_src, n_dst), max(n_src, n_dst)
    at = (np.arange(small) if small == large
          else np.sort(rng.integers(0, max(large - 1, 1), size=small)))
    by_small = np.zeros((small, large), dtype=bool)
    by_small[np.arange(small), at] = True
    by_small[np.arange(small), (at + 1) % large] = True
    return by_small.T if n_dst > n_src else by_small


@dataclasses.dataclass(frozen=True)
class Slots:
    """Where each weight slot of a sparse edge reads the dense ``H``.

    ``index`` is the mapping's row structure (``-1`` in a padded slot);
    slot ``(r, m)`` holds ``H[target[r, m], source[r, m]]`` where ``valid``.
    """

    scatter: bool
    index: np.ndarray
    target: np.ndarray
    source: np.ndarray
    valid: np.ndarray

    def weights(self, H: np.ndarray) -> np.ndarray:
        H = np.asarray(H)
        picked = H[np.where(self.valid, self.target, 0), np.where(self.valid, self.source, 0)]
        return np.where(self.valid, picked, np.zeros((), H.dtype))


def pattern_slots(mask: np.ndarray, *, scatter: bool) -> Slots:
    """The row structure of *mask*: per target its sources (gather), or per
    source its targets (scatter), each row's entries in descending order
    (a builder keeps the order it is given), padded to the longest row."""
    rows_of = mask.T if scatter else mask
    n_rows = rows_of.shape[0]
    lists = [np.nonzero(row)[0][::-1] for row in rows_of]
    k = max(1, max(len(entries) for entries in lists))
    index = np.full((n_rows, k), -1, dtype=np.int64)
    for r, entries in enumerate(lists):
        index[r, :len(entries)] = entries
    valid = index >= 0
    own = np.broadcast_to(np.arange(n_rows)[:, None], index.shape)
    target, source = (index, own) if scatter else (own, index)
    return Slots(scatter, index, np.asarray(target), np.asarray(source), valid)


def _sparse_edge_mapping(slots: Slots, n_src: int, n_dst: int, dtype):
    """A sparse mapping over *slots* with zero weights (a draw fills them)."""
    zeros = np.zeros(slots.index.shape, _dt(dtype))
    if not slots.scatter:
        return sparse_matrix_mapping(slots.index, zeros, n_source=n_src)
    counts = slots.valid.sum(axis=1)
    return StaticSparseMapping(np.where(slots.valid, slots.index, 0), jnp.asarray(zeros),
                               n_source=n_src, n_target=n_dst, counts=counts,
                               layout="scatter")


def build(topo: Topology, knobs, *, dtype="float32", node_order=None, edge_order=None,
          compile: bool = True, mapping_kind: str = "matrix",
          geometry: Optional[Geometry] = None) -> Built:
    """A :class:`GraphManager` for *topo*, nodes and edges added in the given orders.

    *knobs* is one ``CouplingGroup`` configuration for every group, or a
    sequence of them, one per group; knobs a configuration leaves inert are
    dropped (:func:`coupled_graphs.live_knobs`).  ``compile()``'s
    ``UserWarning`` texts are recorded rather than raised.  *mapping_kind*
    (one of :data:`MAPPING_KINDS`) says how every mapped edge holds its
    matrix; ``"matrix"`` is the dense mapping every caller had.  *geometry*
    builds the mapped edges it names as geometry-dependent mappings, with
    their geometry fields on both ends; the returned ``topo`` carries them.
    """
    topo = with_geometry(topo, geometry)
    anchors = {} if geometry is None else geometry.anchor
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
    slots = {}
    for i in edge_order:
        e = topo.edges[i]
        mapping = None
        extra = {}
        if e.mapped and i in anchors:
            mapping = _geometry_mapping(topo, geometry, i)
            extra = {"geometry": (anchors[i], geometry.field(i))}
        elif e.mapped:
            n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
            if mapping_kind in ("matrix", "matrix-local"):
                mapping = matrix_mapping(np.zeros((n_dst, n_src), _dt(dtype)))
            else:
                # A local scatter in its natural layout: a row per source.
                scatter = (mapping_kind.endswith("scatter")
                           or (mapping_kind == "sparse-local" and n_dst > n_src))
                slots[i] = pattern_slots(mapping_pattern(topo, i, mapping_kind),
                                         scatter=scatter)
                mapping = _sparse_edge_mapping(slots[i], n_src, n_dst, dtype)
        gm.add_edge(e.src, e.dst, e.field, f"u{e.port}", transform=e.transform,
                    additive=e.additive, mapping=mapping, **extra)
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
    return Built(gm, topo, str(dtype), keys, recorded, node_order, edge_order,
                 mapping_kind, slots, geometry)


# ---------------------------------------------------------------------------
# Values
# ---------------------------------------------------------------------------


def _spectral_radius(M: np.ndarray) -> float:
    M = np.asarray(M, np.float64)
    return float(np.max(np.abs(np.linalg.eigvals(M)))) if M.size else 0.0


def draw_values(topo: Topology, rng: np.random.Generator, rho: float, *,
                nonnormal: bool = False, bias_scale: float = 1.0, dtype="float32",
                group_cfgs=None, mapping_kind: str = "matrix",
                geometry: Optional[Geometry] = None) -> dict:
    """Gains, biases, mapping matrices and initial states; each group at rate *rho*.

    Under a sparse *mapping_kind* every drawn ``H`` is zeroed outside its
    edge's pattern (:func:`mapping_pattern`) before the groups are rescaled,
    so ``values["H"]`` is the matrix a sparse edge and a dense edge both
    hold.  ``"matrix"`` draws what it always drew.

    ``{"nodes": {name: {"G": [...], "b": ..., "x0": ...}}, "H": {edge: ...}}``.
    With *geometry*, also ``"geometry": {(node, field): {"start": ...,
    "move": ...}}`` for both ends of every edge it names, and ``H`` of such
    an edge is the matrix its anchor's start gives (:func:`_draw_geometry`).
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
            pattern = mapping_pattern(topo, i, mapping_kind)
            if pattern is not None:
                values["H"][i] = np.where(pattern, values["H"][i], 0.0)
    if geometry is not None:
        _draw_geometry(topo, rng, values, geometry, dtype)
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


#: A moving matrix's step, relative to the matrix: small enough that a
#: group drawn at rate ``rho`` is still a contraction a few steps later,
#: with a member that takes four sub-steps (at 0.05 a group drawn at 0.5
#: reached 1.34 on its third step), and four orders of magnitude above a
#: float32 step's rounding.
MOVE_SCALE = 0.01


def _draw_geometry(topo, rng, values, geometry: Geometry, dtype) -> None:
    """The geometry fields of both ends of every geometry edge, and the edge's ``H``.

    ``"moving"``: a start matrix and a step ``dM`` per end; ``H`` is the
    anchor's start (what the group's rate is drawn around).  ``"frozen"``:
    points on the one-dimensional grid at index coordinates that are
    multiples of 1/8, some outside the hull, so the multilinear weights
    are exact in every float dtype, 16-bit ones included; ``H`` is the
    matrix of that gather (or its transpose, for a scatter) from the
    independent reference stencil.
    """
    from tests.core import multilinear_reference as mref  # noqa: PLC0415

    gd = np.dtype(str(geometry_dtype(dtype)))
    out = values.setdefault("geometry", {})
    for i in sorted(geometry.anchor):
        e = topo.edges[i]
        n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
        field, shape = geometry.field(i), geometry.shape(topo, i)
        for name in dict.fromkeys((e.src, e.dst)):
            if geometry.kind == "moving":
                start = rng.normal(size=shape) / np.sqrt(n_src)
                move = rng.normal(size=shape) / np.sqrt(n_src) * MOVE_SCALE
                out[(name, field)] = {"start": np.asarray(start, gd),
                                      "move": np.asarray(move, gd)}
            else:
                n_grid = n_src if geometry.mode(i) == "consistent" else n_dst
                index = rng.integers(-3, 8 * (n_grid - 1) + 4, size=shape) / 8.0
                out[(name, field)] = {"start": np.asarray(index, gd), "move": None}
        start = np.asarray(out[(geometry.holder(topo, i), field)]["start"], np.float64)
        if geometry.kind == "moving":
            values["H"][i] = start
        elif geometry.mode(i) == "consistent":
            values["H"][i] = mref.dense_matrix(mref.Grid((0.0,), (1.0,), (n_src,)), start)
        else:
            values["H"][i] = mref.dense_matrix(mref.Grid((0.0,), (1.0,), (n_dst,)), start).T


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
    """``gm.params`` with every four-argument node's gains, bias and every
    mapping's weights: ``H`` itself for a dense edge, ``H`` read through the
    edge's pattern into ``W`` for a sparse one."""
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
    for (name, field), g in values.get("geometry", {}).items():
        live = rename.get(name, name)
        if g["move"] is not None and f"d{field}" in nodes.get(live, {}):
            nodes[live][f"d{field}"] = jnp.asarray(g["move"], nodes[live][f"d{field}"].dtype)
    mappings = {k: dict(p) for k, p in base.get("mappings", {}).items()}
    for i, key in built.mapping_keys.items():
        if not mappings[key]:           # a geometry-dependent mapping has no weights
            continue
        (leaf, live), = mappings[key].items()
        weights = values["H"][i] if i not in built.slots else built.slots[i].weights(
            values["H"][i])
        mappings[key][leaf] = jnp.asarray(weights, live.dtype)
    return {**base, "nodes": nodes, "mappings": mappings}


def set_initial(built: Built, values: dict, rename: Optional[dict] = None) -> None:
    """Reset (state and coupling seeds) and write every node's ``x0``."""
    gm = built.gm
    rename = rename or {}
    cg.recover(gm)
    gm.reset_state()
    starts: dict = {}
    for (name, field), g in values.get("geometry", {}).items():
        starts.setdefault(name, {})[field] = g["start"]
    for name, v in values["nodes"].items():
        live = rename.get(name, name)
        s = dict(gm.get_node_state(live))
        s["x"] = jnp.asarray(v["x0"], s["x"].dtype)
        for field, start in starts.get(name, {}).items():
            if field in s:      # (the static twin of a geometry graph holds none)
                s[field] = jnp.asarray(start, s[field].dtype)
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
    geometry : Geometry, optional
        The geometry-dependent mapped edges.  For ``"moving"`` ones the
        model must be bound to each step (:meth:`bind`) before it is asked
        about it.
    interface_side : str or dict, optional
        Where ``convergence_norm="interface"`` reads a mapped internal
        edge (:meth:`interface_side_of`): ``"delivered"``, ``"compact"``,
        or ``{edge index: "source" | "delivered"}`` for mappings that
        declare their side (the edges it does not name follow
        ``"compact"``).  ``None`` is :data:`INTERFACE_SIDE`, the rule this
        tree's library implements.
    """

    def __init__(self, topo: Topology, values: dict, *, dtype="float32", node_order=None,
                 group_cfgs=None, exact: bool = False, geometry: Optional[Geometry] = None,
                 interface_side=None):
        self.interface_side = INTERFACE_SIDE if interface_side is None else interface_side
        assert isinstance(self.interface_side, dict) or self.interface_side in (
            "delivered", "compact"), self.interface_side
        self.geometry = geometry
        #: Edge -> the matrix it applies at each sub-step of its target, in
        #: the graph's dtype (``"moving"`` edges, once bound).
        self._bound: dict = {}
        #: Edge -> ``[fac G H_k]`` per sub-step of its target.
        self._substep: dict = {}
        #: Nodes that do not fire on the bound step (a multi-rate graph).
        self.idle: frozenset = frozenset()
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
        if nd.name in self.idle:
            # A node that does not fire keeps its state: nothing is read.
            return LD(1), np.zeros(nd.n, LD), []
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
            if i in self._bound:
                # One matrix per sub-step: ``x_d = alpha**d x_0 + sum_k
                # alpha**(d-1-k) (G fac H_k v + ...)``.
                Ks = [fac * (G @ np.asarray(np.asarray(H, np.float64), LD))
                      for H in self._bound[i]]
                assert len(Ks) == d, (i, len(Ks), d)
                self._substep[i] = Ks
                inputs.append((i, sum((a ** (d - 1 - k) * Ks[k] for k in range(d)),
                                      np.zeros_like(Ks[0]))))
                continue
            H = (np.asarray(rnd(self.values["H"][i]), LD) if e.mapped
                 else np.eye(nd.n, dtype=LD))
            inputs.append((i, s_d * fac * (G @ H)))
        return P, c, inputs

    # -- moving geometry ------------------------------------------------------

    def _updates(self, name) -> int:
        """How many times *name*'s ``update`` runs in the bound step."""
        return 0 if name in self.idle else self.divider.get(name, 1)

    def _chain(self, name, field, pre: dict) -> list:
        """A moving field before each update of the bound step and after the last.

        ``[M, M + dM, (M + dM) + dM, ...]`` by the float additions the relay
        performs, in its own dtype, so the entries are the graph's bit for bit.
        """
        M = np.asarray(pre[name][field])
        dM = np.asarray(self.values["geometry"][(name, field)]["move"], M.dtype)
        chain = [M]
        for _ in range(self._updates(name)):
            chain.append(np.asarray(chain[-1] + dM, M.dtype))
        return chain

    def bind(self, pre: dict, firing=None) -> dict:
        """Fix the step that starts at *pre*; return every moving field's value after it.

        *firing* is the set of nodes whose update the step applies (every
        node when ``None``; fewer on a multi-rate graph).  For each moving
        edge the matrix the documented time level names:

        * source anchor: the source's matrix after the step on a forward or
          group-internal edge (at the fixed point the iterate's matrix is
          the updated one from the first pass on), its pre-step matrix on a
          back edge;
        * target anchor: the target's pre-step matrix, and at sub-step
          ``k`` of a sub-cycled target the matrix after ``k`` of its updates.

        Returns ``{(node, field): array}``, the graph-dtype value every
        moving field must hold after the step, bit for bit.
        """
        self.idle = frozenset() if firing is None else frozenset(
            n for n in self.topo.names if n not in firing)
        self._bound, self._substep, expected = {}, {}, {}
        g = self.geometry
        if g is not None and g.kind == "moving":
            chains = {}
            for i in sorted(g.anchor):
                e = self.topo.edges[i]
                for name in dict.fromkeys((e.src, e.dst)):
                    chains[(name, g.field(i))] = self._chain(name, g.field(i), pre)
                    expected[(name, g.field(i))] = chains[(name, g.field(i))][-1]
            for i in sorted(g.anchor):
                e = self.topo.edges[i]
                d = self.divider.get(e.dst, 1)
                if g.anchor[i] == "source":
                    chain = chains[(e.src, g.field(i))]
                    self._bound[i] = [chain[0] if i in self.back else chain[-1]] * d
                else:
                    chain = chains[(e.dst, g.field(i))]
                    self._bound[i] = [chain[min(k, len(chain) - 1)] for k in range(d)]
        self.terms = {nd.name: self._node_terms(nd) for nd in self.topo.nodes}
        return expected

    def _geometry_norm_entries(self, gi: int, norm: str, state: dict) -> int:
        """Entries a group's RMS norm counts beyond the members' ``x``.

        A geometry field is state: the mixed norm reads every floating
        field of every member, and the interface norm reads the geometry a
        group-internal edge takes from its source as it reads the edge's
        value.  Such a field does not depend on the iterate here, so it
        adds nothing to the sum of squares -- only to the count the mean is
        taken over (a field that is zero everywhere leaves the norm).
        """
        g = self.geometry
        if g is None or norm == "l2":
            return 0
        members = set(self.topo.groups[gi])
        fields = []
        for i in sorted(g.anchor):
            e = self.topo.edges[i]
            if norm == "mixed":
                fields += [(name, g.field(i)) for name in dict.fromkeys((e.src, e.dst))
                           if name in members]
            elif e.src in members and e.dst in members and g.anchor[i] == "source":
                fields.append((e.src, g.field(i)))
        if norm == "mixed":
            fields = list(dict.fromkeys(fields))
        return sum(int(np.asarray(state[n][f]).size) for n, f in fields
                   if np.any(np.asarray(state[n][f]) != 0))

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
            if i in self._bound:
                H = np.max([np.abs(np.asarray(Hk, np.float64)) for Hk in self._bound[i]], axis=0)
            else:
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
        if name in self.idle:
            return np.zeros(nd.n)       # kept, not computed: exact
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
                Ks = self._substep.get(i)
                if not jacobi and rank[e.src] < rank[m]:
                    if Ks is not None and interp in ("linear", "quadratic") and d > 1:
                        # One matrix per sub-step: the read at sub-step ``k``
                        # is ``H_k (incoming + w_k (in-pass - incoming))``.
                        for kk in range(d):
                            wk = LD(kk + 1) / LD(d)
                            L[rows, cols] += alpha ** (d - 1 - kk) * wk * Ks[kk]
                            U[rows, cols] += alpha ** (d - 1 - kk) * (LD(1) - wk) * Ks[kk]
                    else:
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

    def interface_side_of(self, i: int) -> str:
        """``"source"`` or ``"delivered"``: where the interface norm reads edge *i*.

        The rule (``interface_side``): under ``"delivered"`` every edge is
        read at the value it delivers.  Under ``"compact"`` a mapped edge
        whose target is *larger* than its source is read at its source
        value; a mapped edge onto a smaller target, **a tie (equal
        sizes)** and every unmapped edge are read as delivered.  A dict
        names the side of the edges whose mapping declares one; the rest
        follow ``"compact"``.  The sizes are the ones the edge was built
        with: a mapping's side is structure, like its sparsity pattern.
        """
        e = self.topo.edges[i]
        rule = self.interface_side
        if not e.mapped or rule == "delivered":
            return "delivered"
        if isinstance(rule, dict) and i in rule:
            assert rule[i] in ("source", "delivered"), rule
            return rule[i]
        return "source" if self.topo.node(e.dst).n > self.topo.node(e.src).n else "delivered"

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

        **Per side** (``interface_side``, :meth:`interface_side_of`): an
        edge the rule reads at its *source* contributes the source's
        ``x`` as stored -- before the mapping and before the transform,
        which the step applies after the mapping (``edge._delivered``:
        the mapping first, then the transform), so a pre-mapping value
        has not been through it -- with no rounding of its own; every
        other edge is read as delivered, as above.  One field read by a
        gather edge and by a scatter edge is read once per edge, each by
        its own edge's rule.
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
            at_source = self.interface_side_of(i) == "source"
            if raw or not e.mapped or at_source:
                block = np.eye(n_src)
                gamma = 0.0
            else:
                block = np.asarray(self._rnd(self.values["H"][i]), np.float64)
                gamma = (n_src + 1) * self.eps
            if not raw and not at_source:
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
        extra = self._geometry_norm_entries(gi, norm, state)
        if extra:
            # Entries the norm counts and that carry no residual: rows that
            # select nothing.
            S = np.vstack([S, np.zeros((extra, S.shape[1]))])
            w = np.concatenate([w, np.ones(extra)])
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

    def interface_claim(self, gi: int, pre: dict, state: dict, side="compact"):
        """``(distance, K)``: what ``converged=True`` promises of an interface quantity.

        In the interface norm of *side* (whatever rule this model was
        built with): ``distance`` of the returned state to the group's
        exact fixed point, over the tolerance, and the constant ``K`` a
        residual at the threshold allows it.  With ``e`` the error, ``r =
        F(x) - x`` the pass's residual, ``S`` the reading, ``M`` the
        stationary map and ``L`` its same-pass part, ``S e = -[S (I -
        M)^{-1} (I - L) S^+] S r`` exactly (every member reads the others
        through the readings, so the bracket acts on ``S r`` alone), a
        converged group's residual is at most its threshold (CPL-048),
        and so ``||D_x S e|| <= K = ||D_x [.] D_r^{-1}||_2``: ``D_x``
        divides each reading by its magnitude at the returned state,
        ``D_r`` by the larger of that and its magnitude after the pass,
        as the residual does.  The norm of the product, not the product
        of two norms: nothing looser than the identity gives.  ``K`` is a
        property of the loop on its readings -- about ``1 / (1 - gain)``
        for a normal one -- and does not see how many entries the large
        side of a mapping has.
        """
        model = LinearModel(self.topo, self.values, dtype=str(self.dtype),
                            node_order=self.order, group_cfgs=self.cfgs,
                            interface_side=side)
        members, _off, k = model._group_layout(gi)
        L, U = (np.asarray(a, np.float64) for a in model.group_pass(gi))
        c = np.asarray(model.group_constant(gi, pre, state), np.float64)
        x = np.concatenate([np.asarray(state[m]["x"], np.float64) for m in members])
        F = np.linalg.solve(np.eye(k) - L, U @ x + c)
        S, w, rtol_eff, _rms = model.norm_parts(gi, state)
        scale = np.zeros(len(w))
        for rows, B, _gamma in model._field_slices(gi):
            if B.shape[0]:
                scale[rows] = rtol_eff * max(float(np.max(np.abs(B @ x))),
                                             float(np.max(np.abs(B @ F))))
        T = S @ np.linalg.solve(np.eye(k) - (L + U), np.eye(k) - L) @ np.linalg.pinv(S)
        K = float(np.linalg.norm((w[:, None] * T) * scale[None, :], 2))
        return model.returned_weight_distance(gi, pre, state), K

    def residual_between(self, gi: int, new: np.ndarray, old: np.ndarray) -> float:
        """The group's residual of the iterate ``new`` against ``old`` (the
        members' stacked ``x``): each field of :meth:`norm_fields` divided
        by ``rtol`` times its largest magnitude over both, a field with
        no magnitude left out, pooled as the norm pools."""
        cfg = self.cfgs[gi]
        norm = cfg.get("convergence_norm", "l2")
        assert self.geometry is None, "a geometry field is counted too: not restated here"
        rtol_eff = 1.0 if norm == "l2" else float(cfg.get("rtol", 1e-6))
        total, count = 0.0, 0
        for B, _gamma in self.norm_fields(gi):
            a, b = B @ np.asarray(new, np.float64), B @ np.asarray(old, np.float64)
            ref = max(float(np.max(np.abs(a))), float(np.max(np.abs(b)))) if a.size else 0.0
            if ref > 0:
                total += float(np.sum(((a - b) / (rtol_eff * ref)) ** 2))
                count += a.size
        return float(np.sqrt(total / max(count, 1) if norm != "l2" else total))

    def pass_residual(self, gi: int, pre: dict, state: dict) -> float:
        """``||F(x) - x||`` of the returned state in the group's norm under this
        model's reading rule, in float64: what the step should report, to
        the rounding of its own dtype."""
        members, _off, k = self._group_layout(gi)
        L, U = (np.asarray(a, np.float64) for a in self.group_pass(gi))
        c = np.asarray(self.group_constant(gi, pre, state), np.float64)
        x = np.concatenate([np.asarray(state[m]["x"], np.float64) for m in members])
        return self.residual_between(gi, np.linalg.solve(np.eye(k) - L, U @ x + c), x)

    def plain_exit(self, gi: int, pre: dict, state: dict, cap: int) -> dict:
        """Where ``acceleration="none"`` stops, restated: the pass count, the
        residual, the verdict and the state, by this model's reading rule.

        One pass from the pre-step state gives ``x_0`` and the seed
        residual; each further pass measures ``r = ||F(x) - x||`` of the
        iterate it started from and stops *on that iterate* when ``r *
        max(1, amplification)`` is at most the threshold (1 under the
        relative norms), the amplification from the last three residuals
        by the library's own ``error_amplification`` -- the estimate rule
        is not what this restates; the reading is.  *state* supplies what
        the group read from outside.  No predictor, no sub-cycling.

        ``margin`` is the least factor any deciding estimate was from the
        threshold: a predicted pass count holds where that is well above
        the residual's rounding, and says nothing where it is not.
        """
        from maddening.core.coupling.acceleration import error_amplification  # noqa: PLC0415

        cfg = self.cfgs[gi]
        assert not cfg.get("subcycling"), "the plain loop restated has no sub-steps"
        norm = cfg.get("convergence_norm", "l2")
        threshold = float(cfg.get("tolerance", 1e-6)) if norm == "l2" else 1.0
        members, off, k = self._group_layout(gi)
        L, U = self.group_pass(gi)
        c = self.group_constant(gi, pre, state)
        step = np.linalg.inv(np.eye(k) - np.asarray(L, np.float64))
        U64, c64 = np.asarray(U, np.float64), np.asarray(c, np.float64)

        def one_pass(x):
            return step @ (U64 @ x + c64)

        def split(x):
            return {m: x[off[m]:off[m] + self.topo.node(m).n] for m in members}

        start = np.concatenate([np.asarray(pre[m]["x"], np.float64) for m in members])
        x = one_pass(start)
        res = prev = prev2 = self.residual_between(gi, x, start)
        margin = float("inf")
        for i in range(1, int(cap)):
            y = one_pass(x)
            res, prev, prev2 = self.residual_between(gi, y, x), res, prev
            amp = float(error_amplification(np.float64(res), np.float64(prev),
                                            np.float64(prev2)))
            estimate = max(res * max(amp, 1.0), 1e-300)
            margin = min(margin, estimate / threshold if estimate > threshold
                         else threshold / estimate)
            if estimate <= threshold:
                return dict(iterations=i, residual=res, converged=True, state=split(x),
                            margin=margin)
            x = y
        return dict(iterations=int(cap), residual=float("nan"), converged=False,
                    state=split(x), margin=margin)

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


def single_pass(model: LinearModel, gi: int, pre: dict, state: Optional[dict] = None) -> dict:
    """One coupling pass of group *gi* from the pre-step state, with explicit loops.

    What ``max_iterations=1`` returns: every member, in sweep order,
    integrates from its pre-step state through its sub-steps, reading

    * a member of the group from the incoming iterate (here the pre-step
      state) -- or, under Gauss-Seidel, a member swept before it from this
      pass, and then at sub-step ``k`` of ``d`` under linear interpolation
      ``incoming + w_k (in-pass - incoming)``, ``w_k = (k + 1) / d``;
    * a node outside the group from *state* (a forward edge) or from *pre*
      (a back edge).

    A moving geometry is read by the same rule as the value it travels
    with: a source-anchored matrix from the dict the value came from, and
    under linear interpolation the *interpolated matrix applied to the
    interpolated value* (the interpolant of ``M @ v`` is another number);
    a target-anchored matrix from the member's own sub-step state, which
    has advanced ``k`` times at sub-step ``k``.

    Returns ``{member: {"x": LD vector, "magnitude": float64 vector,
    <geometry field>: graph-dtype array}}``; ``magnitude`` bounds the
    absolute terms that were summed, for the caller's rounding allowance.
    """
    topo, g, rnd = model.topo, model.geometry, model._rnd  # noqa: SLF001
    moving = g is not None and g.kind == "moving"
    cfg = model.cfgs[gi]
    jacobi = cfg.get("iteration_mode", "gauss-seidel") == "jacobi"
    interp = cfg.get("boundary_interpolation", "linear") if cfg.get("subcycling") else None
    state = pre if state is None else state

    def x_of(fields):
        return np.asarray(np.asarray(fields["x"], np.float64), LD)

    def matrix_of(fields, field):
        return np.asarray(np.asarray(fields[field], np.float64), LD)

    in_pass = {m: dict(pre[m]) for m in topo.groups[gi]}
    out = {}
    for m in gauss_seidel_order(topo, gi, model.order):
        nd = topo.node(m)
        v = model.values["nodes"][m]
        d = model.divider.get(m, 1)
        alpha = LD(float(rnd(nd.alpha)))
        const = np.asarray(rnd(v["b"]), LD) + LD(float(rnd(nd.beta))) * LD(float(rnd(nd.timestep)))
        x = x_of(pre[m])
        mag = np.abs(np.asarray(x, np.float64))
        own = {field: np.asarray(pre[m][field]) for field, _shape, _moves in nd.geom}
        for k in range(d):
            w = LD(k + 1) / LD(d)
            blend = interp in ("linear", "quadratic") and d > 1
            ports = [np.zeros(nd.n, LD) for _ in range(nd.ports)]
            port_mag = [np.zeros(nd.n) for _ in range(nd.ports)]
            for i, e in enumerate(topo.edges):
                if e.dst != m:
                    continue
                fac = LD(TRANSFORM_FACTORS[e.transform] * (2.0 if e.field == "q" else 1.0))
                internal = topo.group_of(e.src) == gi
                if internal:
                    before, now = pre[e.src], (pre[e.src] if jacobi else in_pass[e.src])
                    value = x_of(before) + w * (x_of(now) - x_of(before)) if blend else x_of(now)
                else:
                    before = now = pre[e.src] if i in model.back else state[e.src]
                    value = x_of(now)
                if not e.mapped:
                    H = np.eye(nd.n, dtype=LD)
                elif moving and i in g.anchor:
                    field = g.field(i)
                    if g.anchor[i] == "target":
                        H = np.asarray(np.asarray(own[field], np.float64), LD)
                    elif internal and blend:
                        H = matrix_of(before, field) + w * (
                            matrix_of(now, field) - matrix_of(before, field))
                    else:
                        H = matrix_of(now, field)
                else:
                    H = np.asarray(rnd(model.values["H"][i]), LD)
                ports[e.port] = ports[e.port] + fac * (H @ value)
                port_mag[e.port] = port_mag[e.port] + abs(float(fac)) * (
                    np.abs(np.asarray(H, np.float64)) @ np.abs(np.asarray(value, np.float64)))
            drive = sum((np.asarray(rnd(v["G"][j]), LD) @ ports[j] for j in range(nd.ports)),
                        np.zeros(nd.n, LD))
            drive_mag = sum((np.abs(rnd(v["G"][j])) @ port_mag[j] for j in range(nd.ports)),
                            np.zeros(nd.n))
            x = alpha * x + const + drive
            mag = abs(float(alpha)) * mag + np.abs(np.asarray(const, np.float64)) + drive_mag
            for field, _shape, moves in nd.geom:
                if moves:
                    dM = np.asarray(model.values["geometry"][(m, field)]["move"],
                                    own[field].dtype)
                    own[field] = np.asarray(own[field] + dM, own[field].dtype)
        in_pass[m] = {"x": x, **own}
        out[m] = {"x": x, "magnitude": mag, **own}
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


def mapped_topologies() -> dict:
    """Structures for the mapping kinds, with interfaces wide enough for a
    sparsity pattern to have rows of different lengths.

    The mapped edges of :func:`named_topologies` all join a node of size 2
    and one of size 1, so a pattern over them has one row or one column: no
    row can be shorter than another.  Here the interfaces have 2, 3 and 4
    entries:

    * ``mapped-ring-wide``: a ring group ``m0 (3) -> m1 (4) -> m2 (4) ->
      m0`` whose three edges are all mapped (one between equal sizes), a
      fourth mapped edge inside the group that shares a port with an
      unmapped edge from outside, a mapped *flux* edge between two outside
      nodes, and two mapped edges on one field pair into a reader (so two
      mappings share an edge key but for its ordinal).
    """
    b = TopologyBuilder()
    b.node("drv", 2, alpha=1.0, beta=1.0, flux=True)
    b.node("pre", 3, alpha=0.25, leaves=("tag",))
    b.node("m0", 3, alpha=0.5, beta=1.0, leaves=("count",))
    b.node("m1", 4, alpha=-0.25, leaves=("flag",))
    b.node("m2", 4, alpha=0.25)
    b.node("out", 3, alpha=0.5, leaves=("count", "tag"))
    b.edge("drv", "pre", field="q")
    b.edge("m0", "m1")
    b.edge("m1", "m2", mapped=True)
    b.edge("m2", "m0", transform="negate")
    p = b.edge("pre", "m0")
    b.edge("m1", "m0", port=p, transform="scale_0.5")
    p = b.edge("m2", "out")
    b.edge("m2", "out", port=p, transform="scale_2.0")
    b.group("m0", "m1", "m2")
    return {"mapped-ring-wide": b.build("mapped-ring-wide")}


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


# ---------------------------------------------------------------------------
# Shapes and value moves for the coupling targeted search
# ---------------------------------------------------------------------------


def search_topologies() -> dict:
    """One-group structures for ``test_coupling_targeted_search.py``, each a
    shape a past defect of ``coupling_diagnostics()`` lived on.

    The group is the whole graph, every member declares its evaluation
    count (:class:`TRelay`) and none carries a non-float leaf, so every
    acceleration traces and every bound can be usable.

    * ``ring-K`` (``K`` = 2, 3, 5, 8): a ring of scalars, the chain lengths;
    * ``ring-3-multirate``: ``ring-3`` with one member at half the
      timestep (a sub-cycled, multi-rate group under ``subcycling=True``);
    * ``pair-3``: two members of three entries, ``alpha = 0``: the ring a
      prescribed spectrum is put on (near-degenerate slow modes);
    * ``pair-6``: the same with six entries, twelve scalars under Jacobi:
      more than the eight Krylov steps resolve, so the flags must refuse;
    * ``hub``: a fan-out hub (one field read by four internal edges), a
      chord between two leaves, and a leaf that reads the hub's field
      twice, once through a transform;
    * ``mapped``: a ring of sizes 3, 2, 2 whose three edges are all mapped
      (a mapping with more columns than rows, whose rows can cancel; one
      between equal sizes under a transform).
    """
    out = {}
    for k in (2, 3, 5, 8):
        for multirate in ((False, True) if k == 3 else (False,)):
            b = TopologyBuilder()
            names = [f"r{i}" for i in range(k)]
            for i, name in enumerate(names):
                b.node(name, 1, alpha=(0.5, 0.0, -0.25)[i % 3])
            for i in range(k):
                b.edge(names[i], names[(i + 1) % k])
            b.group(*names)
            label = f"ring-{k}" + ("-multirate" if multirate else "")
            topo = b.build(label)
            if multirate:
                topo = topo.with_timesteps({"r1": 0.5})
            out[label] = topo

    for n in (3, 6):
        b = TopologyBuilder()
        b.node("a", n, alpha=0.0)
        b.node("b", n, alpha=0.0)
        b.edge("b", "a")
        b.edge("a", "b")
        b.group("a", "b")
        out[f"pair-{n}"] = b.build(f"pair-{n}")

    b = TopologyBuilder()
    b.node("h", 2, alpha=0.5)
    for leaf in ("l0", "l1", "l2"):
        b.node(leaf, 2, alpha=0.0 if leaf == "l1" else 0.25)
    _star_shape(b, "h", ("l0", "l1", "l2"))
    b.edge("l0", "l1")
    b.edge("h", "l2", transform="scale_2.0")
    b.group("h", "l0", "l1", "l2")
    out["hub"] = b.build("hub")

    b = TopologyBuilder()
    b.node("m0", 3, alpha=0.5)
    b.node("m1", 2, alpha=0.0)
    b.node("m2", 2, alpha=-0.25)
    b.edge("m0", "m1")
    b.edge("m1", "m2", mapped=True, transform="negate")
    b.edge("m2", "m0")
    b.group("m0", "m1", "m2")
    out["mapped"] = b.build("mapped")
    return out


#: The member sizes of the ``side-<small>-<large>`` pairs: the size ratio of
#: their two mapped edges runs from 1/300 to 300, with a tie.
SIDE_PAIRS = ((1, 300), (3, 300), (2, 60), (2, 12), (2, 8), (3, 6), (4, 4))


def side_topologies() -> dict:
    """One-group structures whose mapped internal edges join a small field
    and a large one: where a reading rule per side of a mapping shows.

    * ``side-<s>-<g>``: a pair ``s`` (the small member, swept first) and
      ``g``, both edges mapped -- ``g -> s`` onto the smaller target (a
      gather: ratio ``s / g``) and ``s -> g`` onto the larger one (a
      scatter: ratio ``g / s``); ``side-4-4`` is the tie.  A cycle of
      mapped edges returns to the size it left, so every such structure
      but the tie holds an edge of each direction;
    * ``side-<s>-<g>-r``: the same with the large member swept first;
    * ``side-hub``: a hub ``h`` (4) whose one field is read by a gather
      edge (to ``s``, 2) and by a scatter edge (to ``g``, 8), each leaf
      feeding the hub back through a mapped edge of its own: one source
      field read once per edge, each by its own edge's rule.

    How to add a mapping shape: a size pair here (its cells follow in
    ``test_coupling_targeted_search.py``), or a pattern in
    :func:`mapping_pattern` under a new kind of :data:`LOCAL_KINDS`.
    """
    out = {}
    for small, large in SIDE_PAIRS:
        for reverse in ((False, True) if (small, large) in ((3, 300), (2, 12), (2, 8))
                        else (False,)):
            b = TopologyBuilder()
            for name in (("g", "s") if reverse else ("s", "g")):
                b.node(name, small if name == "s" else large, alpha=0.0)
            b.edge("g", "s", mapped=True)
            b.edge("s", "g", mapped=True)
            b.group(*(("g", "s") if reverse else ("s", "g")))
            label = f"side-{small}-{large}" + ("-r" if reverse else "")
            out[label] = b.build(label)
    b = TopologyBuilder()
    b.node("h", 4, alpha=0.0)
    b.node("s", 2, alpha=0.25)
    b.node("g", 8, alpha=0.0)
    b.edge("h", "s", mapped=True)
    b.edge("h", "g", mapped=True)
    b.edge("s", "h", mapped=True)
    b.edge("g", "h", mapped=True)
    b.group("h", "s", "g")
    out["side-hub"] = b.build("side-hub")
    return out


def side_values(topo: Topology, rng: np.random.Generator, rho: float, *, dtype="float32",
                group_cfgs=None, mapping_kind: str = "matrix", offset: float = 1.0) -> dict:
    """:func:`draw_values` for the structures of :func:`side_topologies`.

    The same layout and the same promise -- the one group's coupling
    operator has spectral radius *rho*, ``H`` zeroed outside the kind's
    pattern -- with the rescale in closed form (every gain of a
    four-argument member scales the operator linearly), so a member of
    300 entries costs one eigenvalue problem and not a bisection of them.
    A local kind's weights are positive (interpolation weights); the
    start is *offset* field magnitudes from the biases.
    """
    assert len(topo.groups) == 1 and not any(nd.three_arg for nd in topo.nodes), topo.label
    values: dict = {"nodes": {}, "H": {}}
    for nd in topo.nodes:
        values["nodes"][nd.name] = {
            "G": [rng.normal(size=(nd.n, nd.n)) / np.sqrt(nd.n) for _ in range(nd.ports)],
            "b": rng.uniform(0.5, 2.0, nd.n) * rng.choice([-1.0, 1.0], nd.n),
            "x0": rng.normal(size=nd.n) * offset}
    for i, e in enumerate(topo.edges):
        if not e.mapped:
            continue
        n_src, n_dst = topo.node(e.src).n, topo.node(e.dst).n
        if mapping_kind in LOCAL_KINDS:
            H = np.where(local_pattern(topo, i), rng.uniform(0.05, 0.95, (n_dst, n_src)), 0.0)
        else:
            H = rng.normal(size=(n_dst, n_src)) / np.sqrt(n_src)
            pattern = mapping_pattern(topo, i, mapping_kind)
            if pattern is not None:
                H = np.where(pattern, H, 0.0)
        values["H"][i] = H
    model = LinearModel(topo, values, dtype="float64", group_cfgs=group_cfgs, exact=True)
    radius = _spectral_radius(model.group_operator(0))
    assert radius > 0, topo.label
    dt = _dt(dtype)
    for nd in topo.nodes:
        v = values["nodes"][nd.name]
        v["G"] = [np.asarray(G * (abs(rho) / radius), dt) for G in v["G"]]
        v["b"] = np.asarray(v["b"], dt)
        v["x0"] = np.asarray(v["x0"], dt)
    values["H"] = {i: np.asarray(H, dt) for i, H in values["H"].items()}
    return values


def place_group_fixed_point(topo: Topology, values: dict, gi: int, target: dict, *,
                            group_cfgs=None) -> None:
    """Move the biases of group *gi*'s members so that its fixed point, for
    the step from ``values[...]["x0"]``, is *target* (``{member: array}``).

    Exact in the reference's precision for the values as given; rounding
    them to a graph's dtype afterwards moves the fixed point by that
    rounding, so a caller measures against :class:`LinearModel` of the
    rounded values, never against *target*.  This is how a field is made
    small beside what drives it: its bias cancels its inputs.
    """
    model = LinearModel(topo, values, dtype="float64", group_cfgs=group_cfgs, exact=True)
    pre = {nd.name: {"x": np.asarray(values["nodes"][nd.name]["x0"], np.float64)}
           for nd in topo.nodes}
    state = model.monolithic(pre)
    members, off, k = model._group_layout(gi)
    t = np.concatenate([np.asarray(target[m], LD) for m in members])
    want = (np.eye(k, dtype=LD) - model.group_operator(gi)) @ t
    shift = want - model.group_constant(gi, pre, state)
    for m in members:
        nd = topo.node(m)
        alpha = LD(nd.alpha)
        s_d = sum((alpha ** j for j in range(model.divider.get(m, 1))), LD(0))
        values["nodes"][m]["b"] = np.asarray(
            np.asarray(values["nodes"][m]["b"], LD) + shift[off[m]:off[m] + nd.n] / s_d,
            np.float64)


def rescale_node_units(topo: Topology, values: dict, name: str, unit: float) -> None:
    """Restate node *name*'s field in a unit *unit* times smaller (its
    values *unit* times larger): its bias, start and gains are multiplied,
    and what reads it is divided -- a mapped edge's ``H``, or the gain of a
    port that only *name* feeds.  The coupled system is the same one."""
    v = values["nodes"][name]
    v["b"] = np.asarray(v["b"], np.float64) * unit
    v["x0"] = np.asarray(v["x0"], np.float64) * unit
    v["G"] = [np.asarray(G, np.float64) * unit for G in v["G"]]
    ports = set()
    for i, e in enumerate(topo.edges):
        if e.src != name:
            continue
        if e.mapped:
            values["H"][i] = np.asarray(values["H"][i], np.float64) / unit
        else:
            ports.add((e.dst, e.port))
    for dst, port in ports:
        assert all(e.src == name and not e.mapped for e in topo.edges
                   if e.dst == dst and e.port == port), (name, dst, port)
        values["nodes"][dst]["G"][port] = np.asarray(
            values["nodes"][dst]["G"][port], np.float64) / unit
