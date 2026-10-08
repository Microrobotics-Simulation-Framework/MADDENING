"""Markers coupled to a grid through ``multilinear_grid`` edges, and their exact reference.

``convergence_norm="interface"`` measures a coupling group on what crosses
its internal edges.  The decision the instruments here hold the library to
(maintainer, 2026-10-07), for an edge whose mapping reads a moving
geometry:

* **a gather** (grid to markers, and any mapping that does not deliver
  more entries than it reads: the delivered value is the compact side) is
  read as delivered, at the geometry the step uses -- the source's
  positions as the iterate holds them for a source anchor, the target's
  *pre-step* positions for a target anchor;
* **a scatter** (markers to grid: the delivered value is the large side)
  is read at its inputs: the source value, over its own magnitude as every
  reading is, **and** -- for a source anchor -- the positions, **in units
  of the grid spacing on each axis** and not over their own magnitude,
  which depends on where the origin is.  Positions are always read (a dead
  band is a statement about a magnitude).  A target-anchored geometry is
  the target's pre-step state, the same at every pass: it has no residual
  and is no reading.

:meth:`Reference.parts` is that rule, stated once for this module, with
the pooled count; :func:`measured_whole` is what follows for the state a
solve returns (a field a part holds whole is kept as the accepted iterate
holds it; every other floating field is one plain pass on).

**The problem** is the pair of ``tests/property/interface_side_graphs.py``
with the static matrices replaced by the library's ``multilinear_grid``
kind and positions that *move with the iterate*.  Two nodes ``p`` and
``q``, each holding a value ``x`` and the positions ``pos`` of ``m``
markers, in three kinds (:data:`KINDS`):

* ``"two-way"``: ``p`` holds ``m`` marker values, ``q`` holds ``N`` grid
  values; ``p -> q`` scatters and ``q -> p`` gathers;
* ``"gather-only"``: both hold ``N`` values and read ``m`` (each spreads
  what it reads inside itself), so both edges gather;
* ``"scatter-only"``: both hold ``m`` values and read ``N`` (each picks
  what it reads inside itself), so both edges scatter.

Each edge reads the positions of its source or of its target
(``anchors``).  A node is ``x <- b + a T u`` and ``pos <- pos_pre + pull h
dir tanh(w)``, ``w`` the ``m`` numbers its input comes down to: the
positions move by at most ``pull`` of a cell, from the middle of one, so
no marker crosses a lattice plane and the pass is smooth.

**The grid** covers one fixed length whatever its size (``h = LENGTH /
n``) and the markers sit at fixed fractions of it (to a cell), so the
sizes of a sweep are *refinements of one problem*; its origin is
``origin`` spacings from zero, so two shapes that differ in it are *the
same problem translated*.  Both are what a wrong scale of the positions
breaks: measured against their own magnitude, a position's tolerance in
cells grows with the grid and with the distance from the origin.

**The reference** (:class:`Reference`) is float64 NumPy with a stencil of
its own: the pass restated with explicit reads at the time levels of the
geometry guide, the residual of the rule above, the plain iteration's exit
under it, the fixed point, and the constant ``K`` of the claim

    ``converged=True`` implies the distance to the fixed point in the
    compact readings (marker values, and positions in grid spacings) is
    at most ``K`` tolerances, ``K`` independent of the grid's size and of
    where its origin is.

``K = || D (I - A)^{-1} (I - L) D^{-1} ||_2`` on the compact readings, as
in the static module, with ``A`` the Jacobian of the pass on the readings
at the fixed point (central differences in float64: the pass closes on the
readings, a few dozen numbers whatever ``N`` is) and ``D`` dividing each
part by what it is measured against.  The pass is nonlinear in the
positions, so the identity holds to first order in the distance; the
allowance for that is the caller's and is stated where it is used.

Nothing here is a test.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Optional

import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.acceleration import error_amplification
from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.property.interface_side_graphs import ACCELERATIONS
from tests.property.sysid_transform_grid import precision

KINDS = ("two-way", "gather-only", "scatter-only")
RTOL = 1e-4
CAP = 400
#: The length the grid covers along its first axis, whatever its size.
LENGTH = 50.0
#: Lattice points across, on a grid of two axes.
WIDTH = 3
#: The spacing of the second axis over the first's: another number, so a
#: position measured in the wrong axis' spacing is four times off.
ASPECT = 0.25
DT = 1.0


@dataclasses.dataclass(frozen=True)
class Shape:
    """One compiled graph."""

    kind: str
    n_large: int
    n_small: int
    #: Where ``(p -> q, q -> p)`` read their positions.
    anchors: tuple = ("source", "target")
    schedule: str = "gauss-seidel"
    dtype: str = "float64"
    #: The positions' dtype (the values' when ``None``).
    geom_dtype: Optional[str] = None
    acceleration: str = "none"
    cap: int = CAP
    d: int = 1
    #: The lattice's origin on every axis, in spacings.
    origin: float = 0.0
    layout_seed: int = 0
    solver: str = "ift"
    diagnostics: bool = False

    def __post_init__(self):
        assert self.kind in KINDS and self.acceleration in ACCELERATIONS, self
        assert self.d in (1, 2) and all(a in ("source", "target") for a in self.anchors), self
        assert 2 <= self.n_small < self.n_large and self.n_large % self.width == 0, self

    @property
    def width(self) -> int:
        return WIDTH if self.d == 2 else 1

    @property
    def grid_shape(self) -> tuple:
        n0 = self.n_large // self.width
        return (n0, WIDTH) if self.d == 2 else (n0,)

    @property
    def spacing(self) -> tuple:
        h = LENGTH / self.grid_shape[0]
        return (h, ASPECT * h) if self.d == 2 else (h,)

    @property
    def grid_origin(self) -> tuple:
        return tuple(self.origin * h for h in self.spacing)

    @property
    def geometry_dtype(self) -> str:
        return self.geom_dtype or self.dtype

    @property
    def needs_x64(self) -> bool:
        return "float64" in (self.dtype, self.geometry_dtype)

    @property
    def group(self) -> dict:
        return dict(ACCELERATIONS[self.acceleration], iteration_mode=self.schedule,
                    convergence_norm="interface", rtol=RTOL, max_iterations=self.cap,
                    solver=self.solver, **({"diagnostics": True} if self.diagnostics else {}))


#: ``(p -> q, q -> p)``: what each edge is, by kind.
WAYS = {"two-way": ("scatter", "gather"), "gather-only": ("gather", "gather"),
        "scatter-only": ("scatter", "scatter")}
EDGES = (("p", "q"), ("q", "p"))


def node_sizes(shape: Shape) -> dict:
    """``{name: (entries held, entries read)}``."""
    N, m = shape.n_large, shape.n_small
    return {"two-way": {"p": (m, m), "q": (N, N)},
            "gather-only": {"p": (N, m), "q": (N, m)},
            "scatter-only": {"p": (m, N), "q": (m, N)}}[shape.kind]


def layout_of(shape: Shape) -> dict:
    """What the markers of each node are, as structure: the cell each sits
    in along the first axis (at a fixed fraction of the length, the same
    cell for both nodes, so that the loop closes on the markers), where in
    that cell (each node's own: a geometry read from the wrong end is
    another number), and the direction its input moves it in."""
    out = {}
    n0 = shape.grid_shape[0]
    shared = np.random.default_rng(9_000 + 977 * shape.layout_seed + 7 * shape.n_small)
    along = np.sort(shared.uniform(0.2, 0.8, shape.n_small))
    cells = np.minimum(np.floor(along * n0).astype(np.int64), n0 - 2)
    for k, name in enumerate(("p", "q")):
        rng = np.random.default_rng(9_100 + 977 * shape.layout_seed + 31 * k + 7 * shape.n_small)
        index = np.empty((shape.n_small, shape.d))
        index[:, 0] = cells + rng.uniform(0.3, 0.7, shape.n_small)
        if shape.d == 2:
            # In the first cell across, whose lower lattice row the nodes'
            # own spread and pick use.
            index[:, 1] = rng.uniform(0.3, 0.7, shape.n_small)
        direction = rng.uniform(0.5, 1.0, (shape.n_small, shape.d)) * rng.choice(
            (-1.0, 1.0), (shape.n_small, shape.d))
        # The pair of flat cells a node's own spread or pick uses.
        pair = np.stack([cells * shape.width, (cells + 1) * shape.width], axis=1)
        out[name] = dict(index=index, direction=direction, pair=pair)
    return out


def positions_at(shape: Shape, index: np.ndarray) -> np.ndarray:
    """Positions at the index coordinates *index*, as the geometry's dtype holds them."""
    pos = np.asarray(shape.grid_origin) + index * np.asarray(shape.spacing)
    return np.asarray(np.asarray(pos, np.dtype(shape.geometry_dtype)), np.float64)


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------


class GeoSideNode(SimulationNode):
    """``x <- b + a T u`` and ``pos <- pos_pre + pull h dir tanh(w)``.

    ``T`` is the identity, a spread (``port < n``) or a pick (``port >
    n``) over the node's own ``pair`` of cells, with fixed halves; ``w``
    is the input itself where it has ``m`` entries and the pick of it
    otherwise.  ``a``, ``b`` and ``pull`` are parameters.
    """

    def __init__(self, name, timestep, *, n, port, m, d, dtype, geom_dtype, pair, direction,
                 spacing):
        dt_, gd = jnp.dtype(dtype), jnp.dtype(geom_dtype)
        super().__init__(name, timestep, a=jnp.zeros((), dt_), b=jnp.zeros(n, dt_),
                         pull=jnp.zeros((), gd))
        self._n, self._port, self._m, self._d = int(n), int(port), int(m), int(d)
        self._dtype, self._gd = dt_, gd
        self._pair = np.asarray(pair, np.int32)
        self._reach = np.asarray(direction, np.float64) * np.asarray(spacing, np.float64)

    def initial_state(self):
        return {"x": jnp.zeros(self._n, self._dtype),
                "pos": jnp.zeros((self._m, self._d), self._gd)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._port,), dtype=self._dtype,
                                       default=jnp.zeros(self._port, self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs["u"]
        half = jnp.asarray(0.5, self._dtype)
        picked = None if self._port == self._m else jnp.sum(half * u[self._pair], axis=1)
        if self._port == self._n:
            t = u
        elif self._port < self._n:
            t = jnp.zeros(self._n, self._dtype).at[self._pair].add(half * u[:, None])
        else:
            t = picked
        w = u if picked is None else picked
        move = p["pull"] * jnp.asarray(self._reach, self._gd) * jnp.tanh(w.astype(self._gd))[:, None]
        return {"x": (p["b"] + p["a"] * t).astype(self._dtype),
                "pos": (state["pos"] + move).astype(self._gd)}

    def update_evaluations(self):
        return 1


@dataclasses.dataclass
class Built:
    gm: GraphManager
    shape: Shape


def _mapping(shape: Shape, way: str):
    return multilinear_grid_mapping(
        shape.grid_origin, shape.spacing, shape.grid_shape, n_points=shape.n_small,
        mode="conservative" if way == "scatter" else "consistent")


def _nodes(shape: Shape, timesteps: Optional[dict] = None) -> dict:
    sizes, lay = node_sizes(shape), layout_of(shape)
    return {name: GeoSideNode(
        name, (timesteps or {}).get(name, DT), n=sizes[name][0], port=sizes[name][1], m=shape.n_small, d=shape.d,
        dtype=shape.dtype, geom_dtype=shape.geometry_dtype, pair=lay[name]["pair"],
        direction=lay[name]["direction"], spacing=shape.spacing) for name in ("p", "q")}


def build(shape: Shape) -> Built:
    """Compile *shape* (call under :func:`precision` for a float64 field or geometry)."""
    gm = GraphManager()
    for node in _nodes(shape).values():
        gm.add_node(node)
    for (src, dst), way, anchor in zip(EDGES, WAYS[shape.kind], shape.anchors):
        gm.add_edge(src, dst, "x", "u", mapping=_mapping(shape, way), geometry=(anchor, "pos"))
    gm.add_coupling_group(["p", "q"], **shape.group)
    gm.compile()
    return Built(gm, shape)


# ---------------------------------------------------------------------------
# The marker-side twin: the rule stated with plain edges
# ---------------------------------------------------------------------------


class ScatteringGrid(GeoSideNode):
    """The grid node of a two-way shape with the scatter inside it.

    It reads the marker values on ``f`` and -- for a source-anchored
    scatter -- the markers' positions on ``g``, as ``(pos - reference) /
    h``: the edge that carries them applies that transform, and this node
    undoes it.  For a target anchor it scatters at its own pre-step
    positions.  The scatter is the library's kernel, so the two graphs
    deposit in the same order.
    """

    def __init__(self, *args, mapping, reference, anchored_at_source, **kwargs):
        super().__init__(*args, **kwargs)
        self._mapping = mapping
        self._reference = None if reference is None else np.asarray(reference, np.float64)
        self._at_source = bool(anchored_at_source)
        self._h = np.asarray(kwargs["spacing"], np.float64)

    def boundary_input_spec(self):
        spec = {"f": BoundaryInputSpec(shape=(self._m,), dtype=self._dtype,
                                       default=jnp.zeros(self._m, self._dtype))}
        if self._at_source:
            spec["g"] = BoundaryInputSpec(shape=(self._m, self._d), dtype=self._gd,
                                          default=jnp.zeros((self._m, self._d), self._gd))
        return spec

    def update(self, state, boundary_inputs, dt, *, params=None):
        if self._at_source:
            geom = (jnp.asarray(self._reference, self._gd)
                    + boundary_inputs["g"] * jnp.asarray(self._h, self._gd))
        else:
            geom = state["pos"]
        u = self._mapping.apply(boundary_inputs["f"], None, geom)
        return super().update(state, {"u": u}, dt, params=params)


def in_spacings(reference, spacing, dtype):
    """The edge transform ``pos -> (pos - reference) / h``, per axis."""
    ref = np.asarray(reference, np.float64)
    h = np.asarray(spacing, np.float64)

    def _in_spacings(v, _ref=ref, _h=h):
        return (v - jnp.asarray(_ref, v.dtype)) / jnp.asarray(_h, v.dtype)
    return _in_spacings


def twin_reference(shape: Shape, start_pos: np.ndarray) -> np.ndarray:
    """The constant the twin's positions are carried relative to: each
    marker's own start, except the first's, which is one spacing below it.

    The interface norm reads a plain edge over the largest magnitude of
    what it delivers.  Carried this way the first marker's entry is
    exactly one (it must not move: the caller pins it, ``direction`` zero)
    and every other entry is the marker's motion, under a cell: the
    magnitude is one spacing, which is the rule's unit, with the rule's
    count.
    """
    ref = np.array(start_pos, np.float64)
    ref[0] = ref[0] - np.asarray(shape.spacing)
    return ref


def marker_side_twin(shape: Shape, start_pos: np.ndarray) -> Built:
    """The two-way graph of *shape* with its scatter applied inside ``q``.

    ``p -> q`` is a plain edge of ``m`` marker values and, for a
    source-anchored scatter, a second plain edge of the markers'
    positions in grid spacings (:func:`twin_reference`); ``q -> p`` is the
    same gather edge.  The same coupled problem, and every internal edge
    is read as a plain or a gather edge always was: **its interface norm
    reads the scatter's inputs by construction**, the positions in grid
    spacings, whatever rule the library has for a geometry edge.
    """
    assert shape.kind == "two-way", shape
    at_source = shape.anchors[0] == "source"
    sizes, lay = node_sizes(shape), layout_of(shape)
    common = dict(m=shape.n_small, d=shape.d, dtype=shape.dtype,
                  geom_dtype=shape.geometry_dtype, spacing=shape.spacing)
    reference = twin_reference(shape, start_pos) if at_source else None
    gm = GraphManager()
    gm.add_node(GeoSideNode("p", DT, n=sizes["p"][0], port=sizes["p"][1], pair=lay["p"]["pair"],
                            direction=pinned(lay["p"]["direction"]), **common))
    gm.add_node(ScatteringGrid("q", DT, n=sizes["q"][0], port=sizes["q"][1],
                               pair=lay["q"]["pair"], direction=lay["q"]["direction"],
                               mapping=_mapping(shape, "scatter"), reference=reference,
                               anchored_at_source=at_source, **common))
    gm.add_edge("p", "q", "x", "f")
    if at_source:
        gm.add_edge("p", "q", "pos", "g",
                    transform=in_spacings(reference, shape.spacing, shape.geometry_dtype))
    gm.add_edge("q", "p", "x", "u", mapping=_mapping(shape, "gather"),
                geometry=(shape.anchors[1], "pos"))
    gm.add_coupling_group(["p", "q"], **shape.group)
    gm.compile()
    return Built(gm, shape)


def pinned(direction: np.ndarray) -> np.ndarray:
    """*direction* with the first marker held still (the twin's unit entry)."""
    out = np.array(direction, np.float64)
    out[0] = 0.0
    return out


def build_pinned(shape: Shape) -> Built:
    """:func:`build` with ``p``'s first marker held still: the edge-mapped
    graph the marker-side twin is the twin of."""
    nodes = _nodes(shape)
    lay = layout_of(shape)
    nodes["p"]._reach = pinned(lay["p"]["direction"]) * np.asarray(shape.spacing)   # noqa: SLF001
    gm = GraphManager()
    for node in nodes.values():
        gm.add_node(node)
    for (src, dst), way, anchor in zip(EDGES, WAYS[shape.kind], shape.anchors):
        gm.add_edge(src, dst, "x", "u", mapping=_mapping(shape, way), geometry=(anchor, "pos"))
    gm.add_coupling_group(["p", "q"], **shape.group)
    gm.compile()
    return Built(gm, shape)


# ---------------------------------------------------------------------------
# Values and the reference
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Draw:
    """The numbers put on a compiled shape."""

    seed: int
    #: The loop gain of the values at rest (positions held).
    gain: float
    #: How far, in cells, a marker moves at most.
    pull: float = 0.15
    sign: float = 1.0


#: What a part is measured against: its own largest magnitude, or one grid spacing.
OWN, SPACINGS = "own", "spacings"


def measured_whole(shape: Shape) -> tuple:
    """``((node, field), ...)``: the fields the interface norm of *shape*'s
    group measures whole.  Restated from the kinds and the anchors, not
    from the library's plan: the source value of a scatter, and its
    source's positions where it is anchored there."""
    whole = []
    for (src, _dst), way, anchor in zip(EDGES, WAYS[shape.kind], shape.anchors):
        if way == "scatter":
            whole.append((src, "x"))
            if anchor == "source":
                whole.append((src, "pos"))
    return tuple(whole)


class Reference:
    """The exact problem of ``(shape, draw)`` in float64, as the graph's dtypes hold it."""

    def __init__(self, shape: Shape, draw: Draw, *, pin_first: bool = False):
        self.shape, self.draw = shape, draw
        self.sizes = node_sizes(shape)
        self.layout = layout_of(shape)
        if pin_first:
            self.layout["p"]["direction"] = pinned(self.layout["p"]["direction"])
        self.h = np.asarray(shape.spacing, np.float64)
        self.o = np.asarray(shape.grid_origin, np.float64)
        dt = np.dtype(shape.dtype)
        rng = np.random.default_rng(draw.seed)
        rnd = lambda a: np.asarray(np.asarray(a, dt), np.float64)  # noqa: E731
        self.b = {name: rnd(rng.uniform(0.5, 1.5, self.sizes[name][0])) for name in ("p", "q")}
        self.pull = float(np.asarray(draw.pull, np.dtype(shape.geometry_dtype)))
        #: The pre-step state: what each member integrates from.
        self.pre = {name: {"x": self.b[name].copy(),
                           "pos": positions_at(shape, self.layout[name]["index"])}
                    for name in ("p", "q")}
        # Unit gains with the positions held: the round trip is linear, and
        # its radius scales with the product of the gains.
        self.a = {"p": 1.0, "q": 1.0}
        held, self.pull = self.pull, 0.0
        A, _L, _scale = self.linearised(self.pre, "jacobi")
        self.pull = held
        radius = float(np.max(np.abs(np.linalg.eigvals(A)))) ** 2
        s = math.sqrt(draw.gain / radius)
        self.a = {"p": float(rnd(draw.sign * s)), "q": float(rnd(s))}

    # -- the kernel, restated ---------------------------------------------------

    def stencil(self, pos: np.ndarray):
        """``(index, weight)``, both ``(m, 2**d)``: the lattice points around
        each position and their multilinear weights (clamped to the hull)."""
        lower, upper, frac = [], [], []
        for a, n in enumerate(self.shape.grid_shape):
            u = np.clip((pos[:, a] - self.o[a]) / self.h[a], 0.0, n - 1.0)
            base = np.minimum(np.floor(u), max(n - 2, 0))
            frac.append(u - base)
            lower.append(base.astype(np.int64))
            upper.append(np.minimum(base.astype(np.int64) + 1, n - 1))
        strides = [self.shape.width, 1] if self.shape.d == 2 else [1]
        index, weight = [], []
        for corner in np.ndindex(*(2,) * self.shape.d):
            flat = np.zeros(len(pos), np.int64)
            w = np.ones(len(pos))
            for a in range(self.shape.d):
                flat += strides[a] * (upper[a] if corner[a] else lower[a])
                w = w * (frac[a] if corner[a] else 1.0 - frac[a])
            index.append(flat)
            weight.append(w)
        return np.stack(index, axis=1), np.stack(weight, axis=1)

    def gather(self, field: np.ndarray, pos: np.ndarray) -> np.ndarray:
        index, weight = self.stencil(pos)
        return np.sum(weight * field[index], axis=1)

    def scatter(self, values: np.ndarray, pos: np.ndarray) -> np.ndarray:
        index, weight = self.stencil(pos)
        out = np.zeros(self.shape.n_large)
        np.add.at(out, index, weight * values[:, None])
        return out

    # -- the pass ---------------------------------------------------------------

    def way(self, i: int) -> str:
        return WAYS[self.shape.kind][i]

    def geometry(self, i: int, source: dict) -> np.ndarray:
        """The positions edge *i* is resolved with when its source holds
        *source*: the source's own for a source anchor (they move with the
        iterate), the target's pre-step ones for a target anchor."""
        if self.shape.anchors[i] == "source":
            return source["pos"]
        return self.pre[EDGES[i][1]]["pos"]

    def delivered(self, i: int, source: dict) -> np.ndarray:
        """What edge *i* hands its target when its source holds *source*."""
        geom = self.geometry(i, source)
        return (self.scatter(source["x"], geom) if self.way(i) == "scatter"
                else self.gather(source["x"], geom))

    def update(self, name: str, u: np.ndarray) -> dict:
        """``name``'s new state from its input *u* (it integrates from the pre-step state)."""
        n, port = self.sizes[name]
        lay = self.layout[name]
        picked = None if port == self.shape.n_small else np.sum(0.5 * u[lay["pair"]], axis=1)
        if port == n:
            t = u
        elif port < n:
            t = np.zeros(n)
            np.add.at(t, lay["pair"], 0.5 * u[:, None])
        else:
            t = picked
        w = u if picked is None else picked
        move = self.pull * self.h * lay["direction"] * np.tanh(w)[:, None]
        return {"x": self.b[name] + self.a[name] * t, "pos": self.pre[name]["pos"] + move}

    def one_pass(self, x: dict) -> dict:
        """``F(x)``: ``p`` then ``q``, ``q`` reading ``p``'s new state under Gauss-Seidel."""
        new_p = self.update("p", self.delivered(1, x["q"]))
        read = x["p"] if self.shape.schedule == "jacobi" else new_p
        return {"p": new_p, "q": self.update("q", self.delivered(0, read))}

    # -- the readings: the rule -----------------------------------------------

    def parts(self, x: dict, rule: str = "decision") -> list:
        """``[(edge, value, unit)]``: what the norm reads at *x*, edge
        ``p -> q`` first, each part with what it is measured against.

        *rule* ``"decision"`` is the rule of the module docstring.  The
        others are the readings it is not, for premises: ``"delivered"``
        (every edge as delivered), ``"no-positions"`` (a scatter at its
        source value alone), ``"own-magnitude"`` (the positions over their
        own magnitude), ``"anchored-anywhere"`` (a scatter's positions
        read from whichever end holds them).
        """
        out = []
        for i, (src, dst) in enumerate(EDGES):
            if self.way(i) == "gather" or rule == "delivered":
                out.append((i, self.delivered(i, x[src]), OWN))
                continue
            out.append((i, x[src]["x"], OWN))
            if rule == "no-positions":
                continue
            if self.shape.anchors[i] == "source":
                unit = OWN if rule == "own-magnitude" else SPACINGS
                out.append((i, x[src]["pos"] / self.h, unit))
            elif rule == "anchored-anywhere":
                out.append((i, self.pre[dst]["pos"] / self.h, SPACINGS))
        return out

    def count(self, rule: str = "decision") -> int:
        """The entries the pooled root mean square divides by."""
        return sum(v.size for _i, v, _unit in self.parts(self.pre, rule))

    def residual(self, new: dict, old: dict, rule: str = "decision") -> float:
        """The interface residual of ``new`` against ``old``: each part's
        change over ``rtol`` times what it is measured against (its own
        largest magnitude over both, or one spacing), pooled into one RMS."""
        total, count = 0.0, 0
        for (_i, a, unit), (_j, b, _unit) in zip(self.parts(new, rule), self.parts(old, rule)):
            ref = 1.0 if unit == SPACINGS else max(float(np.max(np.abs(a))),
                                                   float(np.max(np.abs(b))))
            if ref > 0:
                total += float(np.sum(((a - b) / (RTOL * ref)) ** 2))
                count += a.size
        return math.sqrt(total / max(count, 1))

    def floor(self, x: Optional[dict] = None) -> float:
        """The float floor of that residual at the state *x* (the pre-step
        state by default), per evaluation: four units of each entry's resolution over what it is
        measured against -- ``eps`` of the coarsest dtype a value was
        computed from (its own, and the positions' for a delivered one,
        whose rounding ``eps |u|`` in spacings also moves the weights it
        was gathered with), and ``eps |u|`` for a position ``u`` spacings
        from zero -- over ``rtol``, pooled as the residual is."""
        eps_x = float(np.finfo(np.dtype(self.shape.dtype)).eps)
        eps_g = float(np.finfo(np.dtype(self.shape.geometry_dtype)).eps)
        x = self.pre if x is None else x
        total, count = 0.0, 0
        for i, value, unit in self.parts(x):
            if unit == SPACINGS:
                eps = eps_g * float(np.max(np.abs(value)))
            elif self.way(i) == "gather":
                reach = float(np.max(np.abs(self.geometry(i, x[EDGES[i][0]]) / self.h)))
                eps = max(eps_x, eps_g, eps_g * reach)
            else:
                eps = eps_x
            total += value.size * (eps / RTOL) ** 2
            count += value.size
        return 4.0 * math.sqrt(total / count)

    # -- the pass on the compact readings --------------------------------------

    def _flat(self, x: dict):
        parts = self.parts(x)
        return np.concatenate([np.ravel(v) for _i, v, _unit in parts]), parts

    def _from_readings(self, z: np.ndarray, like: list) -> dict:
        """The state of each node computed from the readings of the edge into it."""
        at, by_edge = 0, {0: [], 1: []}
        for i, value, _unit in like:
            by_edge[i].append(z[at:at + value.size].reshape(value.shape))
            at += value.size
        new = {}
        for i, (src, dst) in enumerate(EDGES):
            got = by_edge[i]
            if self.way(i) == "gather":
                u = got[0]
            else:
                geom = got[1] * self.h if len(got) == 2 else self.pre[dst]["pos"]
                u = self.scatter(got[0], geom)
            new[dst] = self.update(dst, u)
        return new

    def linearised(self, x: dict, schedule: Optional[str] = None):
        """``(A, L, scale)`` at *x*: the Jacobian of the pass on the compact
        readings (central differences), its same-pass part under
        Gauss-Seidel, and what each entry is measured against."""
        schedule = self.shape.schedule if schedule is None else schedule
        z, parts = self._flat(x)

        def H(v):
            return self._flat(self._from_readings(v, parts))[0]

        k = z.size
        A = np.zeros((k, k))
        # A value steps by a millionth of its part's size; a position by a
        # millionth of a spacing, wherever it is, so that it stays in its
        # cell (the kernel is a polynomial there: the difference is exact
        # to rounding).
        steps = np.concatenate([
            np.full(v.size, 1e-6 * (1.0 if unit == SPACINGS else float(np.max(np.abs(v)))))
            for _i, v, unit in parts])
        for j in range(k):
            step = steps[j]
            up, down = z.copy(), z.copy()
            up[j] += step
            down[j] -= step
            A[:, j] = (H(up) - H(down)) / (2.0 * step)
        first = sum(v.size for i, v, _unit in parts if i == 0)
        L = np.zeros((k, k))
        if schedule != "jacobi":
            L[first:, :first] = A[first:, :first]
        scale = np.concatenate([
            np.full(v.size, 1.0 if unit == SPACINGS else float(np.max(np.abs(v))))
            for _i, v, unit in parts])
        return A, L, scale

    @functools.cached_property
    def fixed_point(self) -> dict:
        x = self.pre
        for _ in range(20_000):
            y = self.one_pass(x)
            moved = max(float(np.max(np.abs(y[n][f] - x[n][f])) / max(
                float(np.max(np.abs(y[n][f]))), 1e-300)) for n in y for f in y[n])
            x = y
            if moved <= 4e-16:
                break
        else:
            raise AssertionError(f"{self.shape} {self.draw}: the plain iteration did not settle")
        return x

    def K(self, whole: Optional[tuple] = None) -> float:
        """``|| D ((I - A)^{-1} (I - L) - P) D^{-1} ||_2`` at the fixed point.

        ``P`` is zero for the accepted iterate (*whole* ``None``).  For
        the state a solve returns it is the identity on the parts read
        from a field that was recomputed (not in *whole*): that field's
        error is the accepted iterate's plus the pass's own step.
        """
        A, L, scale = self.linearised(self.fixed_point)
        eye = np.eye(len(scale))
        M = np.linalg.solve(eye - A, eye - L)
        if whole is not None:
            P = np.concatenate([
                np.full(v.size, 0.0 if (EDGES[i][0], field) in whole else 1.0)
                for (i, v, _unit), field in zip(self.parts(self.fixed_point),
                                                self._part_fields())])
            M = M - np.diag(P)
        return float(np.linalg.norm(M / scale[:, None] * scale[None, :], 2))

    def _part_fields(self) -> list:
        """The source's field each part is read from, in the order of :meth:`parts`."""
        return ["pos" if unit == SPACINGS else "x" for _i, _v, unit in self.parts(self.pre)]

    def returned(self, x: dict, whole: tuple) -> dict:
        """What a solve returns for the iterate *x* it accepted: a field
        the norm measures whole (*whole*) as *x* holds it, every other as
        one plain pass at *x* computes it."""
        after = self.one_pass(x)
        return {n: {f: (x if (n, f) in whole else after)[n][f] for f in after[n]}
                for n in after}

    def distance(self, x: dict) -> float:
        """The distance of *x* to the fixed point in the rule's norm, over the tolerance."""
        total, count = 0.0, 0
        for (_i, a, unit), (_j, b, _unit) in zip(self.parts(x), self.parts(self.fixed_point)):
            ref = 1.0 if unit == SPACINGS else float(np.max(np.abs(a)))
            if ref > 0:
                total += float(np.sum(((a - b) / (RTOL * ref)) ** 2))
                count += a.size
        return math.sqrt(total / max(count, 1))

    # -- the plain iteration ------------------------------------------------

    def plain_exit(self, rule: str = "decision", cap: Optional[int] = None) -> dict:
        """Where ``acceleration="none"`` stops under the residual of *rule*
        (the loop of ``interface_side_graphs.Reference.plain_exit``)."""
        cap = self.shape.cap if cap is None else cap
        x = self.one_pass(self.pre)
        res = prev = prev2 = self.residual(x, self.pre, rule)
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
                return dict(iterations=i, residual=res, converged=True, state=x, margin=margin)
            x = y
        return dict(iterations=cap, residual=math.nan, converged=False, state=x, margin=margin)

    # -- handing the numbers to the graph -------------------------------------

    def params(self, gm: GraphManager) -> dict:
        base = gm.params
        nodes = {name: dict(p) for name, p in base["nodes"].items()}
        for name in ("p", "q"):
            for leaf, value in (("a", self.a[name]), ("b", self.b[name]), ("pull", self.pull)):
                nodes[name][leaf] = jnp.asarray(value, nodes[name][leaf].dtype)
        return {**base, "nodes": nodes}

    def start(self, gm: GraphManager) -> None:
        """Reset *gm* and write the pre-step state."""
        gm.reset_state()
        for name in ("p", "q"):
            held = gm.get_node_state(name)
            gm.set_node_state(name, {f: jnp.asarray(self.pre[name][f], held[f].dtype)
                                     for f in ("x", "pos")})


@functools.lru_cache(maxsize=16)
def built(shape: Shape) -> Built:
    with precision(shape.needs_x64):
        return build(shape)


def state_of(gm: GraphManager) -> dict:
    return {name: {f: np.asarray(v, np.float64) for f, v in gm.get_node_state(name).items()}
            for name in ("p", "q")}


def run(shape: Shape, draw: Draw, *, graph: Optional[Built] = None,
        reference: Optional[Reference] = None) -> dict:
    """One step of *draw* on *shape*, and what the reference says of the state it returned.

    ``excess`` is the claim's score: the distance of the returned state to
    the fixed point in the rule's norm over ``K`` tolerances, ``K`` the
    constant of the state a solve returns, where the group reports
    ``converged`` (0.0 where it does not).
    """
    ref = Reference(shape, draw) if reference is None else reference
    with precision(shape.needs_x64):
        gm = (built(shape) if graph is None else graph).gm
        ref.start(gm)
        gm.step(params=ref.params(gm))
        (report,) = gm.coupling_diagnostics().values()
        report = dict(report)
        state = state_of(gm)
        meta = {k: np.asarray(v) for k, v in gm._state["_meta"].items()}     # noqa: SLF001
    whole = measured_whole(shape)
    converged = bool(report["converged"])
    distance = ref.distance(state)
    K = ref.K(whole)
    return dict(reference=ref, state=state, report=report, meta=meta, converged=converged,
                iterations=int(report["iterations"]), residual=float(report["residual"]),
                whole=whole, K=K, distance=distance,
                excess=distance / K if converged else 0.0)
