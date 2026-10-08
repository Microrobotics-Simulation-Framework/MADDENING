"""A small field coupled to a large one through mapped edges, and its exact reference.

``convergence_norm="interface"`` measures a coupling group on what crosses
its internal edges.  The decision the instruments here hold the library to
(maintainer, 2026-10-07): **the interface norm reads a mapped edge on its
compact side** -- a mapping whose target is *larger* than its source is
read at its source value, before the mapping; every other edge (a mapping
onto a smaller or an equal target, an unmapped edge) is read at the value
it delivers.  :func:`side_of` is that rule, stated once for this module.

**The problem** is the two-way "robot and grid" pair of
``benchmarks/results/interface_norm_dilution/measure.py``, generalised to
three *kinds* (:data:`KINDS`), each a pair of nodes ``p`` and ``q`` and two
mapped internal edges built from linear interpolation at ``m`` markers on a
line of ``N`` cells (a *gather*, ``N -> m``, two weights per marker) and
its transpose (a *scatter*, ``m -> N``):

* ``"two-way"``: ``p`` holds ``m`` marker forces, ``q`` holds ``N`` cell
  values; ``q -> p`` gathers and ``p -> q`` scatters;
* ``"gather-only"``: both nodes hold ``N`` values and read ``m`` (each
  spreads what it reads inside itself), so both edges gather;
* ``"scatter-only"``: both nodes hold ``m`` values and read ``N`` (each
  picks what it reads inside itself), so both edges scatter.

Every node is ``x <- b + a T u`` with ``T`` the identity, a spread or a
pick, so one pass is affine and the whole problem closes on the *compact
readings* -- ``m`` numbers per edge, whatever ``N`` is.  The gains ``a``,
the biases ``b`` and the mapping weights are arguments of the compiled
step: a :class:`Shape` compiles once and any number of :class:`Draw` s run
on it.

**The reference** (:class:`Reference`) is float64 NumPy over sparse
matrices: the exact fixed point from the compact system, the pass restated
with explicit reads (Jacobi and Gauss-Seidel), the interface residual under
either reading rule (``"delivered"``: what every edge delivers, the rule of
0.4.0 before the decision; ``"compact"``: the decision), the plain
iteration's exit pass under that residual (:func:`plain_exit`), and the
constant ``K`` of the claim the property scores:

    ``converged=True`` under the interface norm implies the distance to
    the fixed point, in the compact readings and the norm's own weights,
    is at most ``K`` times the tolerance, with ``K`` independent of ``N``.

``K`` is not chosen: for an affine pass ``F`` with stationary map ``A`` and
same-pass part ``L`` the error of an iterate is ``-(I - A)^{-1} (I - L)``
of its residual ``F(x) - x`` exactly (the identity CPL-088's bound rests
on), a converged group's residual is at most its threshold (its error
estimate is never below the residual, CPL-048), so ``K = || D (I - A)^{-1}
(I - L) D^{-1} ||_2`` on the compact readings, ``D`` dividing each reading
by its own magnitude.  For a normal loop that is about ``1 / (1 - gain)``.
The compact system has ``2 m`` unknowns and does not see ``N``.

Nothing here is a test.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Optional

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from maddening.core.coupling.acceleration import error_amplification
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.coupling.sparse_mapping import StaticSparseMapping, sparse_matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.property.sysid_transform_grid import precision

KINDS = ("two-way", "gather-only", "scatter-only")

#: How an edge holds its matrix: the dense ``matrix_mapping``; a
#: ``StaticSparseMapping`` in its natural layout (a gather lists each
#: target's sources, a scatter each source's targets); or in the other
#: layout (a scatter held as rows of the large target, most of them empty,
#: and a gather held as rows of the large source).
MAPPINGS = ("matrix", "sparse", "sparse-transposed")

#: The stock accelerations, each with the knobs it reads.
ACCELERATIONS = {
    "none": dict(acceleration="none"),
    "aitken": dict(acceleration="aitken"),
    "fixed": dict(acceleration="fixed", relaxation=0.8),
    "iqn-ils": dict(acceleration="iqn-ils"),
    "iqn-imvj": dict(acceleration="iqn-imvj", jacobian_reuse=2),
}

RTOL = 1e-4
CAP = 400


def offset_by(c: float):
    """The edge transform ``v -> v + c``."""
    def _offset(v, _c=float(c)):
        return v + jnp.asarray(_c, v.dtype)
    return _offset


#: The side rule the library is held to (``coupled_topologies.INTERFACE_SIDE``
#: says the same; this module restates the rule from sizes and imports no
#: other reference).
LIBRARY_RULE = "compact"


def measured_whole(gm) -> tuple:
    """The members whose field the interface norm of *gm*'s group measures whole.

    A solve returns such a field as the iterate it accepted holds it, and
    every other one plain pass on.  Restated from the edges' own sizes,
    not from the library's plan: the source of an edge with no mapping
    and no transform, or of a static mapping the rule reads at its source
    (:func:`side_of`: onto more entries than its source holds) -- there
    the reading is the source field itself, before the mapping and the
    transform."""
    whole = set()
    for e in gm._edges:                                   # noqa: SLF001
        if e.mapping is None:
            if e.transform is None:
                whole.add(e.source_node)
        elif LIBRARY_RULE == "compact" and side_of(
                int(e.mapping.n_source), int(e.mapping.n_target)) == "source":
            whole.add(e.source_node)
    return tuple(sorted(whole))


def side_of(n_source: int, n_target: int, declared: Optional[str] = None) -> str:
    """Where the interface norm reads a mapped edge: ``"source"`` or ``"delivered"``.

    The decision: a mapping whose target is larger than its source is read
    at its source; a tie, and a target that is smaller, at what the edge
    delivers.  A mapping kind may declare its side (*declared*), which
    overrides the sizes.
    """
    if declared is not None:
        assert declared in ("source", "delivered"), declared
        return declared
    return "source" if n_target > n_source else "delivered"


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


class SideNode(SimulationNode):
    """``x <- b + a T u``: ``n`` values from a port of ``port`` entries.

    ``T`` is the identity (``port == n``), a spread (``port < n``: entry
    ``j`` of the port goes to ``index[j]`` with ``weight[j]``, summed) or a
    pick (``port > n``: entry ``i`` of the state is ``sum_k weight[i, k]
    u[index[i, k]]``).  ``a`` (a scalar) and ``b`` are parameters.
    """

    def __init__(self, name, timestep, *, n, port, dtype, index=None, weight=None):
        dt_ = jnp.dtype(dtype)
        super().__init__(name, timestep, a=jnp.zeros((), dt_), b=jnp.zeros(n, dt_))
        self._n, self._port, self._dtype = int(n), int(port), dt_
        assert (index is None) == (n == port), (n, port)
        self._index = None if index is None else np.asarray(index, np.int32)
        self._weight = None if weight is None else np.asarray(weight, dt_)

    def initial_state(self):
        return {"x": jnp.zeros(self._n, self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._port,), dtype=self._dtype,
                                       default=jnp.zeros(self._port, self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        if self._index is None:
            t = u
        elif self._port < self._n:
            t = jnp.zeros(self._n, self._dtype).at[self._index].add(
                jnp.asarray(self._weight) * u[:, None])
        else:
            t = jnp.sum(jnp.asarray(self._weight) * u[self._index], axis=1)
        return {"x": (p["b"] + p["a"] * t).astype(self._dtype)}

    def update_evaluations(self):
        return 1


# ---------------------------------------------------------------------------
# Structure: what a compiled graph bakes in
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Shape:
    """One compiled graph."""

    kind: str
    n_large: int
    n_small: int
    mapping: str = "matrix"
    schedule: str = "gauss-seidel"
    dtype: str = "float64"
    acceleration: str = "none"
    norm: str = "interface"
    cap: int = CAP
    #: Seeds the marker cells (structure: a sparse index is baked in).
    layout_seed: int = 0
    #: An offset the ``p -> q`` edge's *transform* adds to what its mapping
    #: delivers (the step applies the mapping, then the transform).  A
    #: reading at the source is the value before both, so it does not carry
    #: the offset; a delivered reading does, and an offset -- unlike a scale
    #: -- changes the magnitude the norm divides by.
    offset: float = 0.0
    #: ``diagnostics=True`` on the group (the spectral keys of the report).
    diagnostics: bool = False

    def __post_init__(self):
        assert self.kind in KINDS and self.mapping in MAPPINGS, self
        assert self.acceleration in ACCELERATIONS, self
        assert 2 <= self.n_small <= self.n_large, self

    @property
    def group(self) -> dict:
        return dict(ACCELERATIONS[self.acceleration], iteration_mode=self.schedule,
                    convergence_norm=self.norm, rtol=RTOL, max_iterations=self.cap,
                    solver="ift", **({"diagnostics": True} if self.diagnostics else {}))


def marker_cells(shape: Shape, which: int) -> np.ndarray:
    """The left cell of each marker's interpolation pair, sorted (seeded, per edge)."""
    rng = np.random.default_rng(5_000 + 977 * shape.layout_seed + 31 * which
                                + 7 * shape.n_small + shape.n_large)
    lo = int(0.2 * shape.n_large)
    hi = max(lo + 1, int(0.8 * shape.n_large) - 1)
    return np.sort(rng.integers(lo, hi, shape.n_small))


@dataclasses.dataclass(frozen=True)
class Edge:
    """``src.x -> dst.u`` through a matrix with two entries per marker."""

    src: str
    dst: str
    #: ``"gather"`` (``N -> m``) or ``"scatter"`` (``m -> N``).
    way: str
    cells: np.ndarray

    def coo(self, frac: np.ndarray):
        """``(rows, cols, values)`` of the matrix at interpolation fractions *frac*."""
        m = len(self.cells)
        marker = np.repeat(np.arange(m), 2)
        cell = np.stack([self.cells, self.cells + 1], axis=1).ravel()
        vals = np.stack([1.0 - frac, frac], axis=1).ravel()
        return (marker, cell, vals) if self.way == "gather" else (cell, marker, vals)


def edges_of(shape: Shape) -> tuple:
    """The two internal edges, ``p -> q`` first.

    In the two-way kind the scatter is the gather's transpose: the same
    markers (and, in a :class:`Reference`, the same weights).
    """
    ways = {"two-way": ("scatter", "gather"), "gather-only": ("gather", "gather"),
            "scatter-only": ("scatter", "scatter")}[shape.kind]
    second = 0 if shape.kind == "two-way" else 1
    return (Edge("p", "q", ways[0], marker_cells(shape, 0)),
            Edge("q", "p", ways[1], marker_cells(shape, second)))


def node_sizes(shape: Shape) -> dict:
    """``{name: (n, port)}``."""
    N, m = shape.n_large, shape.n_small
    return {"two-way": {"p": (m, m), "q": (N, N)},
            "gather-only": {"p": (N, m), "q": (N, m)},
            "scatter-only": {"p": (m, N), "q": (m, N)}}[shape.kind]


def inner_cells(shape: Shape, name: str) -> Optional[np.ndarray]:
    """The cells a node's own spread or pick uses (``None`` for the identity).

    A spread puts what the node read where the edge out of it gathers; a
    pick reads where the edge into it scattered: the loop closes on the
    markers.
    """
    n, port = node_sizes(shape)[name]
    if n == port:
        return None
    p_to_q, q_to_p = edges_of(shape)
    leaving = p_to_q if name == "p" else q_to_p
    entering = q_to_p if name == "p" else p_to_q
    return (leaving if port < n else entering).cells


@dataclasses.dataclass(frozen=True)
class Slots:
    """A sparse layout of one edge: where each weight slot reads the entry list."""

    scatter: bool
    index: np.ndarray
    entry: np.ndarray
    valid: np.ndarray

    def weights(self, values: np.ndarray, dtype) -> np.ndarray:
        return np.where(self.valid, np.asarray(values)[np.where(self.valid, self.entry, 0)],
                        0.0).astype(dtype)


def slots_of(edge: Edge, shape: Shape, *, scatter: bool) -> Slots:
    """Rows of targets listing sources (gather layout) or of sources listing targets."""
    rows, cols, _ = edge.coo(np.zeros(len(edge.cells)))
    sizes = node_sizes(shape)
    n_rows = sizes[edge.src][0] if scatter else sizes[edge.dst][1]
    own, other = (cols, rows) if scatter else (rows, cols)
    order = np.argsort(own, kind="stable")
    own_s = own[order]
    start = np.searchsorted(own_s, np.arange(n_rows))
    place = np.arange(len(own_s)) - start[own_s]
    k = int(place.max()) + 1
    index = np.full((n_rows, k), -1, np.int64)
    entry = np.zeros((n_rows, k), np.int64)
    index[own_s, place] = other[order]
    entry[own_s, place] = order
    return Slots(scatter, index, entry, index >= 0)


@dataclasses.dataclass
class Built:
    gm: GraphManager
    shape: Shape
    edges: tuple
    #: ``{edge position: (library edge key, Slots or None)}``.
    mappings: dict


def _scatter_layout(shape: Shape, edge: Edge) -> Optional[bool]:
    if shape.mapping == "matrix":
        return None
    natural = edge.way == "scatter"
    return natural if shape.mapping == "sparse" else not natural


def build(shape: Shape) -> Built:
    """Compile *shape* (call under :func:`precision` for float64)."""
    sizes = node_sizes(shape)
    edges = edges_of(shape)
    gm = GraphManager()
    for name in ("p", "q"):
        n, port = sizes[name]
        cells = inner_cells(shape, name)
        index = None if cells is None else np.stack([cells, cells + 1], axis=1)
        # The inner operator's weights are fixed halves: structure.
        weight = None if cells is None else np.full(index.shape, 0.5)
        gm.add_node(SideNode(name, 1.0, n=n, port=port, dtype=shape.dtype,
                             index=index, weight=weight))
    mappings = {}
    for pos, e in enumerate(edges):
        n_src, n_dst = sizes[e.src][0], sizes[e.dst][1]
        layout = _scatter_layout(shape, e)
        slots = None
        if layout is None:
            mapping = matrix_mapping(np.zeros((n_dst, n_src), shape.dtype))
        else:
            slots = slots_of(e, shape, scatter=layout)
            zeros = np.zeros(slots.index.shape, shape.dtype)
            if not layout:
                mapping = sparse_matrix_mapping(slots.index, zeros, n_source=n_src)
            else:
                mapping = StaticSparseMapping(
                    np.where(slots.valid, slots.index, 0), jnp.asarray(zeros),
                    n_source=n_src, n_target=n_dst, counts=slots.valid.sum(axis=1),
                    layout="scatter")
        transform = offset_by(shape.offset) if pos == 0 and shape.offset else None
        gm.add_edge(e.src, e.dst, "x", "u", mapping=mapping, transform=transform)
        mappings[pos] = (gm._edges[pos].key, slots)  # noqa: SLF001
    gm.add_coupling_group(["p", "q"], **shape.group)
    gm.compile()
    return Built(gm, shape, edges, mappings)


# ---------------------------------------------------------------------------
# Values and the reference
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Draw:
    """The numbers put on a compiled shape."""

    seed: int
    #: The loop gain: the spectral radius of one round trip (the
    #: Gauss-Seidel rate; the Jacobi rate is its square root).
    gain: float
    #: The sign of the round trip (a negative loop alternates).
    sign: float = 1.0


class Reference:
    """The exact problem of ``(shape, draw)`` in float64, as the graph's dtype holds it."""

    def __init__(self, shape: Shape, draw: Draw):
        self.shape, self.draw = shape, draw
        self.edges = edges_of(shape)
        self.sizes = node_sizes(shape)
        dt = np.dtype(shape.dtype)
        rng = np.random.default_rng(draw.seed)
        rnd = lambda a: np.asarray(np.asarray(a, dt), np.float64)  # noqa: E731
        self.entries, self.H = [], []
        fracs = [rng.uniform(0.05, 0.95, shape.n_small) for _ in self.edges]
        if shape.kind == "two-way":
            fracs[1] = fracs[0]
        for e, frac in zip(self.edges, fracs):
            rows, cols, vals = e.coo(frac)
            vals = rnd(vals)
            self.entries.append(vals)
            self.H.append(sp.csr_matrix(
                (vals, (rows, cols)),
                shape=(self.sizes[e.dst][1], self.sizes[e.src][0])))
        self.T = {}
        for name in ("p", "q"):
            n, port = self.sizes[name]
            cells = inner_cells(shape, name)
            if cells is None:
                self.T[name] = sp.identity(n, format="csr")
                continue
            own = np.repeat(np.arange(len(cells)), 2)
            cell = np.stack([cells, cells + 1], axis=1).ravel()
            half = np.full(len(own), 0.5)
            self.T[name] = (sp.csr_matrix((half, (cell, own)), shape=(n, port)) if port < n
                            else sp.csr_matrix((half, (own, cell)), shape=(n, port)))
        #: What each edge's transform adds to the delivered value.
        self.shift = [float(shape.offset), 0.0]
        self.b = {name: rnd(rng.uniform(0.5, 1.5, self.sizes[name][0])) for name in ("p", "q")}
        # Unit gains first: the compact Jacobi pass's spectral radius is the
        # square root of the round trip's, and linear in each gain.
        self.a = {"p": 1.0, "q": 1.0}
        radius = float(np.max(np.abs(np.linalg.eigvals(self._compact("jacobi")[0])))) ** 2
        s = math.sqrt(draw.gain / radius)
        self.a = {"p": float(rnd(draw.sign * s)), "q": float(rnd(s))}
        self.start = {name: self.b[name].copy() for name in ("p", "q")}
        self.compact = self._compact(shape.schedule)

    # -- the pass -------------------------------------------------------------

    def _into(self, name: str) -> int:
        return next(i for i, e in enumerate(self.edges) if e.dst == name)

    def update(self, name: str, x_src: np.ndarray) -> np.ndarray:
        """``name``'s new value from the value its edge's source holds."""
        i = self._into(name)
        return self.b[name] + self.a[name] * (self.T[name] @ self.delivered(i, x_src))

    def delivered(self, i: int, x_src: np.ndarray) -> np.ndarray:
        """What edge *i* hands its target: the mapping, then the transform."""
        return self.H[i] @ x_src + self.shift[i]

    def one_pass(self, x: dict) -> dict:
        """``F(x)``: ``p`` then ``q``, ``q`` reading ``p``'s new value under Gauss-Seidel."""
        new_p = self.update("p", x["q"])
        read = x["p"] if self.shape.schedule == "jacobi" else new_p
        return {"p": new_p, "q": self.update("q", read)}

    # -- the readings ---------------------------------------------------------

    def at_source(self, i: int, rule: str) -> bool:
        """Does *rule* read edge *i* at its source (before the mapping and the transform)?"""
        assert rule in ("compact", "delivered"), rule
        n_target, n_source = self.H[i].shape
        return rule == "compact" and side_of(n_source, n_target) == "source"

    def reading_matrix(self, i: int, rule: str):
        """The linear part of what the norm reads on edge *i* under *rule*."""
        if self.at_source(i, rule):
            return sp.identity(self.H[i].shape[1], format="csr")
        return self.H[i]

    def readings(self, x: dict, rule: str) -> list:
        """The fields the norm reads at *x*, in the library's order (source ``p`` first):
        the source's ``x`` itself, or the delivered value, transform included."""
        return [x[self.edges[i].src] if self.at_source(i, rule)
                else self.delivered(i, x[self.edges[i].src]) for i in (0, 1)]

    def residual(self, new: dict, old: dict, rule: str) -> float:
        """The interface residual of ``new`` against ``old``: each reading's change over
        ``rtol`` times its own largest magnitude (over both), pooled into one RMS."""
        total, count = 0.0, 0
        for a, b in zip(self.readings(new, rule), self.readings(old, rule)):
            ref = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
            if ref > 0:
                total += float(np.sum(((a - b) / (RTOL * ref)) ** 2))
                count += a.size
        return math.sqrt(total / max(count, 1))

    # -- the fixed point and the claim's constant -----------------------------

    def _compact(self, schedule: str):
        """``(A, L, c, to_state, off)``: the pass on the compact readings ``z = (z_0, z_1)``.

        ``z_i`` is edge ``i``'s compact reading.  ``x_dst = b + a T (H z)``
        where the edge is read at its source and ``b + a T z`` where it is
        read as delivered, so each node's value -- and the next reading --
        is an affine function of the reading of the edge into it.
        """
        m_of, lift, shift, R, own = [], [], [], [], []
        for i in range(2):
            src = self.at_source(i, "compact")
            Ri = self.reading_matrix(i, "compact")
            R.append(Ri)
            m_of.append(Ri.shape[0])
            # The delivered value from the reading: ``H z + shift`` or ``z``.
            lift.append(self.H[i] if src else sp.identity(self.H[i].shape[0]))
            shift.append(self.shift[i] if src else 0.0)
            own.append(0.0 if src else self.shift[i])
        off = [0, m_of[0]]
        k = sum(m_of)

        def value(name, zj, j):
            """``name``'s value from the reading of the edge into it."""
            return self.b[name] + self.a[name] * (self.T[name] @ (lift[j] @ zj + shift[j]))

        A = np.zeros((k, k))
        c = np.zeros(k)
        for i, e in enumerate(self.edges):          # edge i leaves e.src
            j = self._into(e.src)                   # the edge into its source
            name = e.src
            block = (R[i] @ self.T[name] @ lift[j]) * self.a[name]
            A[off[i]:off[i] + m_of[i], off[j]:off[j] + m_of[j]] = np.asarray(block.todense())
            c[off[i]:off[i] + m_of[i]] = R[i] @ value(name, np.zeros(m_of[j]), j) + own[i]
        L = np.zeros((k, k))
        if schedule != "jacobi":
            # ``q`` reads ``p``'s new value: the reading of the edge out of
            # ``p`` (edge 0) taken in the same pass feeds edge 1's reading.
            L[off[1]:, :off[1]] = A[off[1]:, :off[1]]

        def to_state(z):
            return {name: value(name, z[off[j]:off[j] + m_of[j]], j)
                    for name in ("p", "q") for j in (self._into(name),)}

        return A, L, c, to_state, off

    @functools.cached_property
    def fixed_point(self) -> dict:
        A, _L, c, to_state, _off = self.compact
        return to_state(np.linalg.solve(np.eye(len(c)) - A, c))

    @functools.cached_property
    def K(self) -> float:
        """``|| D (I - A)^{-1} (I - L) D^{-1} ||_2`` on the compact readings at the fixed point."""
        A, L, c, _to_state, off = self.compact
        z = np.linalg.solve(np.eye(len(c)) - A, c)
        d = np.empty(len(c))
        d[:off[1]] = 1.0 / np.max(np.abs(z[:off[1]]))
        d[off[1]:] = 1.0 / np.max(np.abs(z[off[1]:]))
        M = np.linalg.solve(np.eye(len(c)) - A, np.eye(len(c)) - L)
        return float(np.linalg.norm(d[:, None] * M / d[None, :], 2))

    def returned(self, x: dict, whole=()) -> dict:
        """What a solve returns for the iterate *x* it accepted: a field the
        interface norm measures whole (*whole*, :func:`measured_whole`: read
        by an edge with no mapping and no transform, or at its source by a
        mapping onto more entries) as *x* holds it, every other as one plain
        pass at *x* computes it.  In a :func:`build` graph the source of a
        scatter is kept and the source of a gather or a tie is ``F(x)``'s;
        in the marker-side twin ``p`` is read by a plain edge."""
        after = self.one_pass(x)
        return {name: x[name] if name in whole else after[name] for name in ("p", "q")}

    def K_returned(self, whole=()) -> float:
        """:attr:`K` for the state :meth:`returned`: its error is the accepted
        iterate's plus the pass's own step on the recomputed fields, so on
        the compact readings the bracket loses the identity on the readings
        of a recomputed source."""
        A, L, c, _to_state, off = self.compact
        z = np.linalg.solve(np.eye(len(c)) - A, c)
        d = np.empty(len(c))
        d[:off[1]] = 1.0 / np.max(np.abs(z[:off[1]]))
        d[off[1]:] = 1.0 / np.max(np.abs(z[off[1]:]))
        P = np.zeros(len(c))
        for i, e in enumerate(self.edges):
            if e.src not in whole:
                P[off[i]:(off[i + 1] if i + 1 < len(off) else len(c))] = 1.0
        M = np.linalg.solve(np.eye(len(c)) - A, np.eye(len(c)) - L) - np.diag(P)
        return float(np.linalg.norm(d[:, None] * M / d[None, :], 2))

    def distance(self, x: dict, rule: str = "compact") -> float:
        """The distance of *x* to the fixed point in the norm of *rule*, over the tolerance:
        each reading's error over ``rtol`` times its magnitude at *x*, pooled RMS."""
        total, count = 0.0, 0
        for a, b in zip(self.readings(x, rule), self.readings(self.fixed_point, rule)):
            ref = float(np.max(np.abs(a)))
            if ref > 0:
                total += float(np.sum(((a - b) / (RTOL * ref)) ** 2))
                count += a.size
        return math.sqrt(total / max(count, 1))

    def marker_error(self, x: dict) -> float:
        """The largest relative error of any entry of a compact reading, over the tolerance
        (the number ``measure.py`` reports for the marker forces)."""
        worst = 0.0
        for a, b in zip(self.readings(x, "compact"), self.readings(self.fixed_point, "compact")):
            there = b != 0      # (a square scatter leaves the cells no marker touches at 0)
            worst = max(worst, float(np.max(np.abs(a - b)[there] / np.abs(b)[there])))
        return worst / RTOL

    # -- the plain iteration --------------------------------------------------

    def plain_exit(self, rule: str, cap: Optional[int] = None) -> dict:
        """Where ``acceleration="none"`` stops under the residual of *rule*.

        The loop restated: one pass from the start gives ``x_0`` and the
        seed residual; each further pass measures ``r = ||F(x) - x||`` of
        the iterate it started from and stops on that iterate when the
        estimate ``r * max(1, amplification)`` is at most 1, the
        amplification from the last three residuals (the library's
        ``error_amplification``: the estimate rule is not what this
        reference restates).  ``margin`` is how far the deciding
        estimates were from 1, as a ratio: a prediction is only as good
        as that is above the residual's rounding.
        """
        cap = self.shape.cap if cap is None else cap
        x = self.one_pass(self.start)
        res = prev = prev2 = self.residual(x, self.start, rule)
        margin = math.inf
        for i in range(1, cap):
            y = self.one_pass(x)
            res, prev, prev2 = self.residual(y, x, rule), res, prev
            amp = float(error_amplification(np.float64(res), np.float64(prev),
                                            np.float64(prev2)))
            estimate = res * max(amp, 1.0)
            margin = min(margin, max(estimate, 1e-300) if estimate > 1 else
                         1.0 / max(estimate, 1e-300))
            if estimate <= 1.0:
                return dict(iterations=i, residual=res, converged=True, state=x,
                            margin=margin)
            x = y
        return dict(iterations=cap, residual=math.nan, converged=False, state=x, margin=margin)

    # -- handing the numbers to the graph -------------------------------------

    def params(self, built: Built) -> dict:
        gm, dt = built.gm, np.dtype(self.shape.dtype)
        base = gm.params
        nodes = {name: dict(p) for name, p in base["nodes"].items()}
        for name in ("p", "q"):
            nodes[name]["a"] = jnp.asarray(self.a[name], dt)
            nodes[name]["b"] = jnp.asarray(self.b[name], dt)
        mappings = {k: dict(p) for k, p in base["mappings"].items()}
        for pos, (key, slots) in built.mappings.items():
            (leaf, live), = mappings[key].items()
            if slots is None:
                weights = np.asarray(self.H[pos].todense(), dt)
            else:
                weights = slots.weights(self.entries[pos], dt)
            assert weights.shape == live.shape, (weights.shape, live.shape)
            mappings[key][leaf] = jnp.asarray(weights, live.dtype)
        return {**base, "nodes": nodes, "mappings": mappings}


def marker_side_twin(shape: Shape, ref: "Reference") -> Built:
    """The two-way graph of *shape* with its scatter applied inside ``q``.

    ``p -> q`` is a plain edge of ``m`` marker values and ``q`` spreads
    them itself, with the weights of *ref*; ``q -> p`` is the same gather
    edge.  The same coupled problem, and both internal edges carry ``m``
    numbers: **its interface norm reads the compact side by construction**,
    whatever rule the library has.  An edge-mapped graph under the compact
    rule must report what this one reports -- every number of the report
    at once, with no model of any of them.
    """
    assert shape.kind == "two-way" and not shape.offset, shape
    N, m = shape.n_large, shape.n_small
    scatter, gather = edges_of(shape)
    gm = GraphManager()
    gm.add_node(SideNode("p", 1.0, n=m, port=m, dtype=shape.dtype))
    gm.add_node(SideNode("q", 1.0, n=N, port=m, dtype=shape.dtype,
                         index=np.stack([scatter.cells, scatter.cells + 1], axis=1),
                         weight=np.asarray(ref.entries[0]).reshape(m, 2)))
    gm.add_edge("p", "q", "x", "u")
    slots = None
    if shape.mapping == "matrix":
        mapping = matrix_mapping(np.zeros((m, N), shape.dtype))
    else:
        slots = slots_of(gather, shape, scatter=False)
        mapping = sparse_matrix_mapping(slots.index, np.zeros(slots.index.shape, shape.dtype),
                                        n_source=N)
    gm.add_edge("q", "p", "x", "u", mapping=mapping)
    gm.add_coupling_group(["p", "q"], **shape.group)
    gm.compile()
    return Built(gm, shape, (scatter, gather), {1: (gm._edges[1].key, slots)})  # noqa: SLF001


@functools.lru_cache(maxsize=8)
def built(shape: Shape) -> Built:
    with precision(shape.dtype == "float64"):
        return build(shape)


def run(shape: Shape, draw: Draw, *, graph: Optional[Built] = None) -> dict:
    """One step of *draw* on *shape*, and what the reference says of the state it returned.

    ``excess`` is the claim's score: the distance to the fixed point in
    the compact readings over ``K`` times the tolerance, where the group
    reports ``converged`` (0.0 where it does not).  ``K`` is the constant
    of the state a solve *returns* (:meth:`Reference.K_returned`): the
    iterate it accepted with the fields in no ``whole`` member one pass
    on.  The report's residual is the accepted iterate's, which the
    returned state does not determine: a caller compares it with the
    reference's own loop (:meth:`Reference.plain_exit`).
    """
    ref = Reference(shape, draw)
    with precision(shape.dtype == "float64"):
        b = built(shape) if graph is None else graph
        gm = b.gm
        gm.reset_state()
        for name in ("p", "q"):
            gm.set_node_state(name, {"x": jnp.asarray(ref.start[name], shape.dtype)})
        gm.step(params=ref.params(b))
        (report,) = gm.coupling_diagnostics().values()
        report = dict(report)
        state = {name: np.asarray(gm.get_node_state(name)["x"], np.float64)
                 for name in ("p", "q")}
        whole = measured_whole(gm)
    converged = bool(report["converged"])
    distance = ref.distance(state, "compact")
    K = ref.K_returned(whole)
    return dict(
        reference=ref, state=state, report=report, converged=converged, whole=whole,
        iterations=int(report["iterations"]), residual=float(report["residual"]),
        K=K, distance=distance, marker_error=ref.marker_error(state),
        excess=distance / K if converged else 0.0)
