"""Members of two sizes in every numeric domain: a mapped edge that expands, and one that reduces.

:mod:`tests.core.coupling_domains` builds both members of its pair at one
size, so a mapped edge between them is a *tie*: as many entries delivered
as read.  Under ``convergence_norm="interface"`` a tie is read as
delivered, and the rule the norm has for an edge whose static mapping
delivers **more** entries than its source field holds -- read at its
source, the compact side (CPL-188) -- never ran in a domain of that
battery.  This module gives the battery the pair it lacked: the small
field coupled to a large one of
:mod:`tests.property.interface_side_graphs` (its three kinds, its three
ways of holding a mapping, its exact reference and its marker-side twin),
built in a :class:`~tests.core.coupling_domains.Domain` and run by that
module's :func:`~tests.core.coupling_domains.run` and
:func:`~tests.core.coupling_domains.run_sequence`.

* **The members** are ``a`` and ``b``, as the domains name them (``b`` is
  the member a domain sub-steps or shards; in the mixed-dtype domain ``a``
  is float32 and ``b`` float64).  A :class:`Cell` says which of them holds
  the *small* field of a two-way pair, so each domain can put either size
  on the member it treats differently.  Both are memoryless (``x <- b + a
  T u``), as the domains need: a step's fixed point depends only on the
  parameters it ran with.
* **The edges** are ``p -> q`` and ``q -> p`` of the reference, each
  through a static mapping (dense, sparse in its natural layout, sparse in
  the other layout) built in the dtype of the member it feeds, with the
  cast the mixed-dtype domain needs after it.  In the two-way kind the
  first *expands* (``m`` values scattered onto ``N``: read at its source)
  and the second *reduces* (``N`` gathered onto ``m``: read as delivered).
* **The reference** (:class:`Stored`) is the float64 reference of
  ``interface_side_graphs`` with every number as the member that stores it
  rounds it (a float32 gain beside a float64 bias, a bfloat16 weight), so
  the model is of the problem the graph holds.  It calls nothing of the
  library's reading: the side rule is its own (``side_of``).
* **The twin** (:func:`build_twin`) is the two-way pair with the scatter
  applied inside the large member (:class:`Spread`) and a plain edge
  carrying the ``m`` values: its interface norm reads the compact side by
  construction, in any domain.

**A pair coupled through a geometry-dependent mapping** (experimental; the
last section): the markers and the grid of
:mod:`tests.property.geometry_interface_graphs`, whose ``multilinear_grid``
edges read positions that move with the iterate, built in a domain the
same way (:class:`GeoCell`, :func:`build_geometry`), with that module's
reference as the members store its numbers (:class:`GeoStored`) and its
marker-side twin (:func:`build_geometry_twin`).

Nothing here is a test.
"""

from __future__ import annotations

import dataclasses
import math
import tempfile
import warnings
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.coupling.sparse_mapping import StaticSparseMapping, sparse_matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core import coupling_domains as cd
from tests.property import geometry_interface_graphs as gi
from tests.property import interface_side_graphs as sg

#: The sizes of a cell on every push: a loop on six numbers, which the
#: report's eight Krylov steps resolve, and a large field twelve times it.
N_SMALL = 3
N_LARGE = 36
#: A tolerance a 16-bit dtype resolves (bfloat16's eps is 2**-7), and the
#: cap its cells stop at, unconverged, as the tie's cells of those domains do.
RTOL_16 = 0.02
CAP_16 = 2


@dataclasses.dataclass(frozen=True)
class Cell:
    """One pair of two sizes in one domain."""

    label: str                      # the domain
    kind: str = "two-way"           # sg.KINDS
    form: str = "matrix"            # sg.MAPPINGS: how the edges hold their matrices
    schedule: str = "gauss-seidel"
    acceleration: str = "none"
    #: The member that holds the small field of a two-way pair.
    small: str = "a"
    diagnostics: bool = False
    n_small: int = N_SMALL
    n_large: int = N_LARGE
    layout_seed: int = 0

    def __post_init__(self):
        assert self.small in ("a", "b"), self
        assert self.small == "a" or self.kind == "two-way", (
            "the members of a gather-only or a scatter-only pair have one size")

    @property
    def domain(self) -> cd.Domain:
        return cd.DOMAINS[self.label]

    @property
    def sixteen(self) -> bool:
        return jnp.dtype(self.domain.coarsest).itemsize == 2

    @property
    def rtol(self) -> float:
        return RTOL_16 if self.sixteen else sg.RTOL

    @property
    def shape(self) -> sg.Shape:
        """The reference's description of the pair (its numbers drawn in
        float64; :class:`Stored` rounds them as the members store them)."""
        return sg.Shape(self.kind, self.n_large, self.n_small, self.form, self.schedule,
                        "float64", self.acceleration, cap=CAP_16 if self.sixteen else sg.CAP,
                        layout_seed=self.layout_seed, diagnostics=self.diagnostics)

    @property
    def names(self) -> dict:
        """The reference's ``p`` (small in a two-way pair) and ``q`` as members."""
        return {"p": "a", "q": "b"} if self.small == "a" else {"p": "b", "q": "a"}

    @property
    def id(self) -> str:
        parts = [self.label, self.kind, self.form, self.schedule]
        if self.acceleration != "none":
            parts.append(self.acceleration)
        if self.small != "a":
            parts.append("small-b")
        if self.diagnostics:
            parts.append("diagnosed")
        if (self.n_small, self.n_large) != (N_SMALL, N_LARGE):
            parts.append(f"{self.n_small}-{self.n_large}")
        return "-".join(parts)


@dataclasses.dataclass
class Built:
    gm: GraphManager
    cell: Cell
    #: ``{edge position: (library edge key, Slots or None)}`` of the mapped edges.
    mappings: dict
    twin: bool = False


class Spread(SimulationNode):
    """The twin's large member: ``x <- b + a S u``, ``S`` the scatter of a
    port of ``m`` values onto ``n``.

    The arithmetic of ``interface_side_graphs.SideNode``'s spread, with the
    scatter's weights ``w`` a parameter beside ``a`` and ``b``: each
    scenario of a run brings its own, as the weights of an edge's mapping
    reach a step through ``params["mappings"]``.
    """

    def __init__(self, name, timestep, *, n, dtype, index):
        dt_ = jnp.dtype(dtype)
        self._index = np.asarray(index, np.int32)
        super().__init__(name, timestep, a=jnp.zeros((), dt_), b=jnp.zeros(n, dt_),
                         w=jnp.zeros(self._index.shape, dt_))
        self._n, self._port, self._dtype = int(n), int(self._index.shape[0]), dt_

    def initial_state(self):
        return {"x": jnp.zeros(self._n, self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._port,), dtype=self._dtype,
                                       default=jnp.zeros(self._port, self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        t = jnp.zeros(self._n, self._dtype).at[self._index].add(p["w"] * u[:, None])
        return {"x": (p["b"] + p["a"] * t).astype(self._dtype)}

    def update_evaluations(self):
        return 1


def _copies(cell: Cell) -> int:
    return cd.N_SHARD if cell.domain.sharded else 1


def _tiled_index(index, copies: int, span: int):
    """*index* into a field of *span* entries, once per copy of the field."""
    return np.concatenate([index + c * span for c in range(copies)])


def _placed(cell: Cell, sg_name: str) -> tuple:
    """``(member name, dtype, timestep)`` of the reference's node *sg_name*."""
    d = cell.domain
    name = cell.names[sg_name]
    dtype = d.dtype_a if name == "a" else d.dtype_b
    return name, dtype, cd.DT / 2 if (d.subcycled and name == "b") else cd.DT


def _node(cell: Cell, sg_name: str) -> sg.SideNode:
    """Member ``sg_name`` of the reference in the cell's domain."""
    shape = cell.shape
    name, dtype, step = _placed(cell, sg_name)
    n, port = sg.node_sizes(shape)[sg_name]
    cells = sg.inner_cells(shape, sg_name)
    index = weight = None
    if cells is not None:
        index = np.stack([cells, cells + 1], axis=1)
        weight = np.full(index.shape, 0.5)
    copies = _copies(cell)
    if index is not None and copies > 1:
        index = _tiled_index(index, copies, n if port < n else port)
        weight = np.tile(weight, (copies, 1))
    return sg.SideNode(name, step, n=n * copies, port=port * copies, dtype=dtype,
                       index=index, weight=weight)


def _spread_node(cell: Cell) -> Spread:
    """The twin's ``q``: the large member, applying the scatter of ``p -> q`` itself."""
    shape = cell.shape
    name, dtype, step = _placed(cell, "q")
    n, port = sg.node_sizes(shape)["q"]
    assert n == port, "the twin's member is the two-way pair's large one"
    cells = sg.edges_of(shape)[0].cells
    index = _tiled_index(np.stack([cells, cells + 1], axis=1), _copies(cell), n)
    return Spread(name, step, n=n * _copies(cell), dtype=dtype, index=index)


def _mapping(cell: Cell, edge: sg.Edge, dtype):
    """``(mapping, Slots or None)``: *edge*'s matrix, all zeros, in *dtype*.

    The weights reach each step through ``params["mappings"]``
    (:func:`params_of`): a reading taken with the mapping's own is a
    reading of zeros.
    """
    shape, sizes = cell.shape, sg.node_sizes(cell.shape)
    n_src, n_dst = sizes[edge.src][0], sizes[edge.dst][1]
    layout = sg._scatter_layout(shape, edge)  # noqa: SLF001
    copies = _copies(cell)
    if layout is None:
        return matrix_mapping(jnp.zeros((n_dst * copies, n_src * copies), dtype)), None
    assert copies == 1, "the sharded domain's copies are built for the dense kind"
    slots = sg.slots_of(edge, shape, scatter=layout)
    zeros = jnp.zeros(slots.index.shape, dtype)
    if not layout:
        return sparse_matrix_mapping(slots.index, zeros, n_source=n_src), slots
    return StaticSparseMapping(
        np.where(slots.valid, slots.index, 0), zeros, n_source=n_src, n_target=n_dst,
        counts=slots.valid.sum(axis=1), layout="scatter"), slots


def _compiled(cell: Cell, nodes: list, edges: list, group_kw: dict) -> GraphManager:
    """The graph of *nodes* and *edges* (``(source, target, mapping)``) in the
    cell's domain: its casts, its clock, its group settings."""
    d = cell.domain
    gm = GraphManager()
    for node in nodes:
        if d.sharded and node.name == "b":
            from maddening.cloud.multigpu.device_mesh import create_device_mesh
            from maddening.cloud.multigpu.sharded_node import ShardedPointwiseNode
            node = ShardedPointwiseNode(node, create_device_mesh(shape=(cd.N_SHARD,)),
                                        shard_axes=0)
        gm.add_node(node)
    dtype = {"a": d.dtype_a, "b": d.dtype_b}
    for source, target, mapping in edges:
        to = dtype[target]
        cast = {} if dtype[source] == to else {"transform": lambda v, to=to: v.astype(to)}
        gm.add_edge(source, target, "x", "u", mapping=mapping, **cast)
    if d.multirate:
        gm.add_node(cd.Ticker("tick", cd.DT / 2))
    kw = {**cell.shape.group, "rtol": cell.rtol, **group_kw}
    gm.add_coupling_group([cell.names["p"], cell.names["q"]], **cd.group_kwargs(d, **kw))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")     # the multi-rate notice
        gm.compile()
    return gm


def build(cell: Cell, **group_kw) -> Built:
    """The pair of *cell*, compiled; call it inside ``cd.entered``.

    The reference's ``p`` is added first: its pass, which
    :class:`Stored` restates, sweeps ``p`` then ``q``.
    """
    d, shape = cell.domain, cell.shape
    dtype = {"a": d.dtype_a, "b": d.dtype_b}
    edges, mappings = [], {}
    for pos, e in enumerate(sg.edges_of(shape)):
        target = cell.names[e.dst]
        mapping, slots = _mapping(cell, e, dtype[target])
        edges.append((cell.names[e.src], target, mapping))
        mappings[pos] = slots
    gm = _compiled(cell, [_node(cell, "p"), _node(cell, "q")], edges, group_kw)
    keys = [e.key for e in gm._edges]  # noqa: SLF001
    return Built(gm, cell, {pos: (keys[pos], slots) for pos, slots in mappings.items()})


def build_twin(cell: Cell, **group_kw) -> Built:
    """The marker-side twin of a two-way *cell*.

    ``p -> q`` is a plain edge of the ``m`` small values and ``q`` spreads
    them itself (:class:`Spread`: the scatter's weights are the node's
    parameters, in the member's dtype, the dtype the edge-mapped graph
    holds them in); ``q -> p`` is the same gather.  Both internal edges
    carry ``m`` numbers, so its norm reads the compact side whatever rule
    the library has for a mapping that expands.
    """
    assert cell.kind == "two-way", cell
    d, shape = cell.domain, cell.shape
    _scatter, gather = sg.edges_of(shape)
    target = cell.names["p"]
    mapping, slots = _mapping(cell, gather, d.dtype_a if target == "a" else d.dtype_b)
    edges = [(cell.names["p"], cell.names["q"], None), (cell.names["q"], target, mapping)]
    gm = _compiled(cell, [_node(cell, "p"), _spread_node(cell)], edges, group_kw)
    return Built(gm, cell, {1: (gm._edges[1].key, slots)}, twin=True)  # noqa: SLF001


# ---------------------------------------------------------------------------
# The reference, as the members store its numbers
# ---------------------------------------------------------------------------
def _as_stored(value, dtype) -> np.ndarray:
    """*value* in float64 after a round trip through *dtype* (under the
    domain's ``x64``: a float64 member keeps it)."""
    return np.asarray(jnp.asarray(np.asarray(value, np.float64), dtype)).astype(np.float64)


class Stored(sg.Reference):
    """The exact reference of ``(cell, draw)``, every number as its member holds it.

    The gains and biases are rounded in their own member's dtype and each
    mapping's weights in the dtype of the member its edge feeds; *scale*
    multiplies the biases (a forcing that moves from step to step).  Build
    it inside ``cd.entered``.  The run starts from the graph's initial
    state, zeros (:attr:`start`).
    """

    def __init__(self, cell: Cell, draw: sg.Draw, scale: float = 1.0):
        super().__init__(cell.shape, draw)
        self.cell = cell
        d = cell.domain
        dtype = {sg_name: (d.dtype_a if name == "a" else d.dtype_b)
                 for sg_name, name in cell.names.items()}
        self.a = {n: float(_as_stored(self.a[n], dtype[n])) for n in ("p", "q")}
        self.b = {n: _as_stored(scale * self.b[n], dtype[n]) for n in ("p", "q")}
        for i, e in enumerate(self.edges):
            self.entries[i] = _as_stored(self.entries[i], dtype[e.dst])
            rows, cols, _ = e.coo(np.zeros(len(e.cells)))
            self.H[i] = sp.csr_matrix((self.entries[i], (rows, cols)), shape=self.H[i].shape)
        self.start = {n: np.zeros(self.sizes[n][0]) for n in ("p", "q")}
        self.compact = self._compact(cell.shape.schedule)
        for cached in ("fixed_point", "K"):
            self.__dict__.pop(cached, None)

    def started_at(self, start: dict) -> "Stored":
        """This reference with its iteration started at *start* (``{"p", "q"}``)."""
        self.start = {n: np.asarray(start[n], np.float64) for n in ("p", "q")}
        return self

    def in_tolerances(self, value: float) -> float:
        """A residual or a distance of the reference (stated at its own
        ``RTOL``) in the cell's tolerance: both are linear in ``1 / rtol``."""
        return value * sg.RTOL / self.cell.rtol


def params_of(ref: Stored, built: Built) -> dict:
    """The graph's parameter pytree with the numbers of *ref*, each leaf in its own dtype."""
    gm, cell = built.gm, built.cell
    base = gm.params
    copies = _copies(cell)
    nodes = {name: dict(p) for name, p in base["nodes"].items()}
    for sg_name, name in cell.names.items():
        leaves = nodes[name]
        leaves["a"] = jnp.asarray(ref.a[sg_name], leaves["a"].dtype)
        leaves["b"] = jnp.asarray(np.tile(ref.b[sg_name], copies), leaves["b"].dtype)
        if "w" in leaves:               # the twin's scatter, inside its large member
            w = np.tile(np.asarray(ref.entries[0]).reshape(-1, 2), (copies, 1))
            assert w.shape == leaves["w"].shape, (w.shape, leaves["w"].shape)
            leaves["w"] = jnp.asarray(w, leaves["w"].dtype)
    mappings = {k: dict(p) for k, p in base.get("mappings", {}).items()}
    for pos, (key, slots) in built.mappings.items():
        (leaf, live), = mappings[key].items()
        if slots is None:
            weights = np.kron(np.eye(copies), np.asarray(ref.H[pos].todense()))
        else:
            weights = slots.weights(ref.entries[pos], np.float64)
        assert weights.shape == live.shape, (weights.shape, live.shape)
        mappings[key][leaf] = jnp.asarray(weights, live.dtype)
    out = {**base, "nodes": nodes}
    if mappings:
        out["mappings"] = mappings
    return out


def state_of(cell: Cell, solve: cd.Solve, pre: bool = False) -> dict:
    """The members' fields of *solve* as the reference names them, in
    float64: ``{"p": ..., "q": ...}`` (one copy of a sharded domain's four,
    which must agree)."""
    out = {}
    source = solve.pre if pre else solve.state
    for sg_name, name in cell.names.items():
        x = np.asarray(source[name]["x"]).astype(np.float64)
        copies = _copies(cell)
        if copies > 1:
            parts = x.reshape(copies, -1)
            assert all(cd.bitwise(parts[0], part) for part in parts[1:]), (
                f"{cell.id}: the copies of {name}.x differ: {parts!r}")
            x = parts[0]
        out[sg_name] = x
    return out


def slot_of(solve: cd.Solve):
    """The group's ``reading_floor`` slot of *solve*, or ``None`` where it has none."""
    return solve.meta.get(f"coupling_{cd.KEY}_reading_floor")


def assert_sized(built: Built, solves: list) -> None:
    """The premise: the graph is the cell's pair, in its domain.

    One edge of a two-way pair delivers at least ten times the entries it
    reads and the other the reverse; a gather-only pair's both reduce and
    a scatter-only pair's both expand; the group sweeps the reference's
    ``p`` first.
    """
    gm, cell = built.gm, built.cell
    cd.assert_in_domain(cell.domain, gm, solves)
    order = [n for n in gm.schedule if n in ("a", "b")]
    assert order == [cell.names["p"], cell.names["q"]], (cell.id, order)
    ratios = []
    for e in gm._edges:  # noqa: SLF001
        if e.mapping is not None:
            ratios.append(e.mapping.n_target / e.mapping.n_source)
    want = {"two-way": [(10.0, None), (None, 0.1)], "gather-only": [(None, 0.1)] * 2,
            "scatter-only": [(10.0, None)] * 2}[cell.kind]
    if built.twin:
        want = want[1:]
    assert len(ratios) == len(want), (cell.id, ratios)
    for ratio, (above, below) in zip(ratios, want):
        assert (above is None or ratio >= above) and (below is None or ratio <= below), (
            cell.id, ratios)


def restart_pairs(built, params_seq: list, start=None) -> tuple:
    """``(uninterrupted, restarted, saved, loaded)`` around a checkpoint taken
    two steps into *params_seq*.

    The first two are the solves after it, run straight on and run again
    after a reset and a load (``cd.run_sequence`` returns the second list
    and compares the states and reports itself; here both are handed back,
    ``_meta`` and all).  *saved* is the graph as the checkpoint was
    written and *loaded* the graph as the load left it, before any step.
    *start* (a callable of the graph) writes the state the run starts
    from, after the first reset and never after the second: what the load
    brings back is the checkpoint's.
    """
    gm, d = built.gm, built.cell.domain
    gm.reset_state()
    if start is not None:
        start(gm)
    start = min(2, len(params_seq) - 1)
    for p in params_seq[:start]:
        cd._one(d, gm, p)  # noqa: SLF001
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "checkpoint.npz"
        gm.save_state(path)
        saved = cd._solve(gm, params_seq[start - 1])  # noqa: SLF001
        straight = [cd._one(d, gm, p) for p in params_seq[start:]]  # noqa: SLF001
        gm.reset_state()
        gm.load_state(path)
        loaded = cd._solve(gm, params_seq[start - 1])  # noqa: SLF001
        restarted = [cd._one(d, gm, p) for p in params_seq[start:]]  # noqa: SLF001
    return straight, restarted, saved, loaded


# ---------------------------------------------------------------------------
# A pair coupled through a geometry-dependent mapping (experimental)
# ---------------------------------------------------------------------------
#: The markers and the grid points of a geometry cell.
GEO_MARKERS, GEO_GRID = 5, 40
#: How far a marker moves in a step at most, in cells.  A power of two
#: (every member dtype stores it exactly, so both members of the
#: mixed-dtype domain hold one number), and small enough that the five
#: steps of a sequence leave every marker in the cell it started in (0.3
#: to 0.7 of the way across): each pass of each step is smooth.
GEO_PULL = 0.03125
#: Where the lattice of a geometry cell starts, in spacings from zero: the
#: grid is centred on zero, so the markers (0.2 to 0.8 of the way along
#: it) are within twelve spacings of it and a float32 member resolves
#: them to 0.06 of the tolerance per evaluation.  ("Coordinates local to
#: the grid", the advisory's own remedy; with the lattice starting at zero
#: the float floor of these pairs is 2.7 times that, and with it what a
#: float32 residual can be held to.)
GEO_ORIGIN = -GEO_GRID / 2
#: The domains a pair with ``multilinear_grid`` edges runs in under the
#: interface norm, and the one in which it is refused at ``compile()``.
#: (Not built here, and not claimed: the 16-bit dtypes, ``run_adaptive``,
#: a sharded member.)
GEO_ACCEPTED = ("f32", "f64", "mixed_dtype", "vmap", "multi_rate", "predictors_warm_starts",
                "checkpoint_restart")
GEO_REFUSED = ("sub_cycled",)
#: What ``compile()`` says of positions a dtype cannot resolve to the
#: tolerance (its advisory's own words).
POSITIONS_ADVISORY = "cannot be resolved to this tolerance"


@dataclasses.dataclass(frozen=True)
class GeoCell:
    """One pair of markers and a grid in one domain."""

    label: str                      # the domain
    kind: str = "two-way"           # gi.KINDS
    #: Where ``(p -> q, q -> p)`` read their positions.
    anchors: tuple = ("source", "target")
    schedule: str = "gauss-seidel"
    acceleration: str = "none"
    #: The member that is the reference's ``p`` (the markers' values of a
    #: two-way pair).
    small: str = "a"
    #: The lattice's origin, in spacings from zero.
    origin: float = GEO_ORIGIN

    def __post_init__(self):
        assert self.small in ("a", "b") and self.kind in gi.KINDS, self

    @property
    def domain(self) -> cd.Domain:
        return cd.DOMAINS[self.label]

    @property
    def shape(self) -> gi.Shape:
        """The reference's description of the pair, in float64
        (:class:`GeoStored` rounds its numbers as the members store them)."""
        return gi.Shape(self.kind, GEO_GRID, GEO_MARKERS, self.anchors, self.schedule,
                        "float64", None, self.acceleration, origin=self.origin)

    @property
    def names(self) -> dict:
        """The reference's ``p`` and ``q`` as members."""
        return {"p": "a", "q": "b"} if self.small == "a" else {"p": "b", "q": "a"}

    @property
    def dtypes(self) -> dict:
        """The dtype of the reference's ``p`` and ``q``: of the member's
        value and of its positions."""
        d = self.domain
        return {n: (d.dtype_a if name == "a" else d.dtype_b) for n, name in self.names.items()}

    @property
    def exact(self) -> bool:
        """Does every member hold float64?"""
        return all(jnp.dtype(t) == jnp.dtype(jnp.float64) for t in self.dtypes.values())

    @property
    def keys(self) -> tuple:
        """The library's keys of ``p -> q`` and ``q -> p``."""
        return tuple(f"{self.names[s]}.x->{self.names[t]}.u" for s, t in gi.EDGES)

    @property
    def id(self) -> str:
        parts = [self.label, self.kind, "-".join(a[0] for a in self.anchors), self.schedule]
        if self.acceleration != "none":
            parts.append(self.acceleration)
        if self.small != "a":
            parts.append("small-b")
        if self.origin != GEO_ORIGIN:
            parts.append(f"at-{self.origin:g}")
        return "-".join(parts)


@dataclasses.dataclass
class GeoBuilt:
    gm: GraphManager
    cell: GeoCell
    #: The advisories ``compile()`` gave about unresolved positions.
    advisories: tuple = ()
    twin: bool = False


def _geo_node(cell: GeoCell, sg_name: str, *, placed=None, scattering=None):
    """Member *sg_name* of the reference in the cell's domain: its value and
    its positions in the member's dtype.  ``p``'s first marker is held
    still, in every graph of a cell: the unit entry of the twin's position
    edge (``geometry_interface_graphs.twin_reference``)."""
    shape = cell.shape
    name, dtype, step = _placed(cell, sg_name)
    n, port = gi.node_sizes(shape)[sg_name]
    lay = gi.layout_of(shape)[sg_name]
    direction = gi.pinned(lay["direction"]) if sg_name == "p" else lay["direction"]
    common = dict(n=n, port=port, m=shape.n_small, d=shape.d, dtype=dtype, geom_dtype=dtype,
                  pair=lay["pair"], direction=direction, spacing=shape.spacing, placed=placed)
    if scattering is None:
        return gi.GeoSideNode(name, step, **common)
    return gi.ScatteringGrid(name, step, **scattering, **common)


def _geo_cast(cell: GeoCell, src: str, dst: str, first=None) -> dict:
    """The transform of an edge from the reference's *src* to its *dst*:
    *first* if given, then the cast the mixed-dtype domain needs."""
    to = cell.dtypes[dst]
    cast = jnp.dtype(cell.dtypes[src]) != jnp.dtype(to)
    if first is None:
        return {"transform": lambda v, to=to: v.astype(to)} if cast else {}
    if not cast:
        return {"transform": first}
    return {"transform": lambda v, to=to, first=first: first(v).astype(to)}


def geometry_group(cell: GeoCell, gm: GraphManager, **group_kw) -> GraphManager:
    """*gm* with the domain's clock and the pair's group under the
    interface norm (*group_kw* over the cell's settings), not compiled."""
    d = cell.domain
    if d.multirate:
        gm.add_node(cd.Ticker("tick", cd.DT / 2))
    kw = {**cell.shape.group, **group_kw}
    gm.add_coupling_group([cell.names["p"], cell.names["q"]], **cd.group_kwargs(d, **kw))
    return gm


def _geo_compiled(cell: GeoCell, gm: GraphManager, twin: bool, group_kw: dict) -> GeoBuilt:
    """*gm* with the domain's clock and group, compiled.  Its warnings are
    read, not silenced: the advisories about unresolved positions are
    handed back, and any other warning but the multi-rate domain's notice
    is an error as everywhere in the suite."""
    d = cell.domain
    geometry_group(cell, gm, **group_kw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
    texts = [str(w.message) for w in caught]
    advisories = tuple(t for t in texts if POSITIONS_ADVISORY in t)
    others = [t for t in texts if POSITIONS_ADVISORY not in t]
    assert d.multirate or not others, (cell.id, others)
    return GeoBuilt(gm, cell, advisories, twin)


def geometry_graph(cell: GeoCell, placed=None) -> GraphManager:
    """The pair of *cell*, its nodes and its two ``multilinear_grid`` edges,
    with no group yet (call it inside ``cd.entered``).  *placed*
    (``{"p": positions, "q": positions}``) starts the markers there; by
    default they start at zero and are written after ``compile()``
    (:meth:`GeoStored.start`), as a step's state is."""
    shape = cell.shape
    gm = GraphManager()
    for sg_name in ("p", "q"):
        gm.add_node(_geo_node(cell, sg_name, placed=(placed or {}).get(sg_name)))
    for (src, dst), way, anchor in zip(gi.EDGES, gi.WAYS[shape.kind], shape.anchors):
        gm.add_edge(cell.names[src], cell.names[dst], "x", "u",
                    mapping=gi._mapping(shape, way),  # noqa: SLF001
                    geometry=(anchor, "pos"), **_geo_cast(cell, src, dst))
    return gm


def build_geometry(cell: GeoCell, placed=None, **group_kw) -> GeoBuilt:
    """The pair of *cell*, compiled; call it inside ``cd.entered``."""
    return _geo_compiled(cell, geometry_graph(cell, placed), False, group_kw)


def build_geometry_twin(cell: GeoCell, start_pos, **group_kw) -> GeoBuilt:
    """The marker-side twin of a two-way *cell*
    (``geometry_interface_graphs.marker_side_twin``, in the cell's domain).

    The scatter is applied inside ``q``; ``p -> q`` is a plain edge of the
    marker values and, for a source-anchored scatter, a second plain edge
    of the markers' positions in grid spacings relative to *start_pos*
    (``p``'s positions as the run starts); ``q -> p`` is the same gather
    edge.  Every internal edge is read as a plain or a gather edge always
    was, the positions against one spacing.
    """
    assert cell.kind == "two-way", cell
    shape = cell.shape
    at_source = shape.anchors[0] == "source"
    reference = gi.twin_reference(shape, start_pos) if at_source else None
    gm = GraphManager()
    gm.add_node(_geo_node(cell, "p"))
    gm.add_node(_geo_node(cell, "q", scattering=dict(
        mapping=gi._mapping(shape, "scatter"), reference=reference,  # noqa: SLF001
        anchored_at_source=at_source)))
    p, q = cell.names["p"], cell.names["q"]
    gm.add_edge(p, q, "x", "f", **_geo_cast(cell, "p", "q"))
    if at_source:
        carried = gi.in_spacings(reference, shape.spacing, cell.dtypes["p"])
        gm.add_edge(p, q, "pos", "g", **_geo_cast(cell, "p", "q", first=carried))
    gm.add_edge(q, p, "x", "u", mapping=gi._mapping(shape, "gather"),  # noqa: SLF001
                geometry=(shape.anchors[1], "pos"), **_geo_cast(cell, "q", "p"))
    return _geo_compiled(cell, gm, True, group_kw)


class GeoStored(gi.Reference):
    """The exact reference of ``(cell, draw)``, every number as its member holds it.

    The gains, the biases and the positions are rounded in their own
    member's dtype (the pull is a power of two: :data:`GEO_PULL`); *scale*
    multiplies the biases (a forcing that moves from step to step).
    ``p``'s first marker is held still, as in the cell's graphs.  Build it
    inside ``cd.entered``.  :attr:`pre`, the state a step starts from, is
    the reference's own until :meth:`from_state` hands it the graph's.
    """

    def __init__(self, cell: GeoCell, draw: gi.Draw, scale: float = 1.0):
        assert draw.pull == GEO_PULL, draw
        super().__init__(cell.shape, draw, pin_first=True)
        self.cell = cell
        dtype = cell.dtypes
        self.a = {n: float(_as_stored(self.a[n], dtype[n])) for n in ("p", "q")}
        self.b = {n: _as_stored(scale * self.b[n], dtype[n]) for n in ("p", "q")}
        self.pre = {n: {"x": self.b[n].copy(),
                        "pos": _as_stored(self.pre[n]["pos"], dtype[n])} for n in ("p", "q")}
        self.__dict__.pop("fixed_point", None)

    def from_state(self, pre: dict) -> "GeoStored":
        """This reference with its step started from *pre* (``{"p", "q"}``,
        each ``{"x", "pos"}``): what the members integrate from, and where
        a target-anchored geometry is read."""
        self.pre = {n: {f: np.asarray(pre[n][f], np.float64) for f in ("x", "pos")}
                    for n in ("p", "q")}
        self.__dict__.pop("fixed_point", None)
        return self

    def floor(self, x=None) -> float:
        """The float floor of the residual at *x* per evaluation, each part
        at the resolution of the members that hold it: a scatter's source
        value at its member's eps; its positions at ``eps |u|`` of one
        spacing; a gathered value at the coarsest of its source's eps, its
        target's (the cast) and the eps of the member whose positions it
        was gathered at, and at that eps times those positions' distance
        from zero in spacings where that is more."""
        eps = {n: float(cd.finfo(t).eps) for n, t in self.cell.dtypes.items()}
        x = self.pre if x is None else x
        total, count = 0.0, 0
        for i, value, unit in self.parts(x):
            src, dst = gi.EDGES[i]
            if unit == gi.SPACINGS:
                e = eps[src] * float(np.max(np.abs(value)))
            elif self.way(i) == "gather":
                holder = src if self.shape.anchors[i] == "source" else dst
                reach = float(np.max(np.abs(self.geometry(i, x[src]) / self.h)))
                e = max(eps[src], eps[dst], eps[holder], eps[holder] * reach)
            else:
                e = eps[src]
            total += value.size * (e / gi.RTOL) ** 2
            count += value.size
        return 4.0 * math.sqrt(total / count)

    def params(self, gm: GraphManager) -> dict:
        """The graph's parameter pytree with this reference's numbers."""
        base = gm.params
        nodes = {name: dict(p) for name, p in base["nodes"].items()}
        for sg_name, name in self.cell.names.items():
            for leaf, value in (("a", self.a[sg_name]), ("b", self.b[sg_name]),
                                ("pull", self.pull)):
                nodes[name][leaf] = jnp.asarray(value, nodes[name][leaf].dtype)
        return {**base, "nodes": nodes}

    def start(self, gm: GraphManager) -> None:
        """Reset *gm* and write the pre-step state: after ``compile()``, so
        the markers are not in the state it sees."""
        gm.reset_state()
        self.write(gm)

    def write(self, gm: GraphManager) -> None:
        for sg_name, name in self.cell.names.items():
            held = gm.get_node_state(name)
            gm.set_node_state(name, {f: jnp.asarray(self.pre[sg_name][f], held[f].dtype)
                                     for f in ("x", "pos")})


def geo_state_of(cell: GeoCell, solve: cd.Solve, pre: bool = False) -> dict:
    """The members' fields of *solve* as the reference names them, in float64."""
    source = solve.pre if pre else solve.state
    return {sg_name: {f: np.asarray(source[name][f]).astype(np.float64) for f in ("x", "pos")}
            for sg_name, name in cell.names.items()}


def geo_measured_whole(cell: GeoCell) -> tuple:
    """``((reference node, field), ...)``: the fields the pair's norm
    measures whole (``geometry_interface_graphs.measured_whole``).  A cast
    after the mapping does not enter: a scatter is read before both."""
    return gi.measured_whole(cell.shape)
