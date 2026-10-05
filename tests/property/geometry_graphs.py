"""Graphs whose edges carry geometry-dependent mappings, and their node-inlined twins.

An edge ``add_edge(S, T, sf, tf, mapping=m, geometry=(anchor, g))`` hands
its mapping a moving geometry: the state field ``g`` of the edge's own
source (``anchor="source"``) or target (``"target"``), read at a time
level the graph decides at each of its call sites.  This module is what
the tests of that feature are built from.  Nothing here is a test, and
nothing here ships: the mapping kinds are registered for the test process
through the public ``register_mapping``.

**The mappings.**

* :class:`GeomMatrixMapping` (kind ``test_geom_matrix``): the geometry *is*
  the matrix, ``apply(field, weights, geom) = geom @ field``.  No weights
  (its entry of ``gm.params["mappings"]`` is ``{}``), linear in the field
  for a given geometry, so a graph of linear relays stays one linear
  system per step whatever the matrix does
  (``tests/property/test_geometry_time_levels.py``).
* :func:`multilinear` is the library's own ``multilinear_grid`` kind, a
  gather / scatter between a uniform grid and moving points, imported when
  it is called.

**The nodes.**  :class:`Body` holds a value field ``x`` and any number of
geometry fields that *move*: at a constant rate, and with the node's own
mapped input (``adv``), which makes a geometry held by a coupling-group
member depend on the group's iterate.  :class:`FluxBody` adds a boundary
flux that reads the mapped input; :class:`Reader` consumes one.

**The node-inlined twin** (:func:`inline_geometry`).  An edge with a
geometry-dependent mapping means, by definition, what the same graph
computes with the mapping moved inside the target node: the edge becomes
the plain edge ``S.sf -> T."<tf>@value"`` and, for a source anchor, a
second plain edge ``S.g -> T."<tf>@geometry"`` straight after it;
:class:`InlinedMappingNode` wraps ``T`` and, in ``update``, in
``compute_boundary_fluxes`` and in ``compute_interface_correction``,
applies the mapping itself -- to the geometry port for a source anchor, to
the field ``g`` of the hook's own ``state`` argument for a target anchor --
before calling ``T``'s hook.  The twin uses ordinary edges only, so it runs
on a tree that knows nothing of geometry edges, and every time level in it
is the one an ordinary edge or a node's own state already has.  The
geometry edge must be the last edge into its port, so that an additive
port sums in the same order in both graphs.

:func:`two_body` builds the graph the differential and gradient tests
draw from: a grid-side body ``F`` and a point-side body ``P`` joined in
both directions, in or out of a coupling group, at one rate or two.
"""

from __future__ import annotations

import contextlib
import dataclasses
import warnings
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np

from maddening.core.coupling.mapping import register_mapping
from maddening.core.coupling.mapping_spec import MappingSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.transforms import resolve_transform, scale

# The scaling transform a drawn edge may name is registered on first use.
scale(0.5)

GEOM_MATRIX = "test_geom_matrix"


@contextlib.contextmanager
def x64(on: bool):
    """``jax_enable_x64`` for the block, restored afterwards."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", bool(on))
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


# ---------------------------------------------------------------------------
# The mapping kinds
# ---------------------------------------------------------------------------


class GeomMatrixMapping:
    """``target = geom @ source``: the geometry is the ``(n_target, n_source)`` matrix."""

    kind = GEOM_MATRIX
    mode = "consistent"
    needs_geometry = True

    def __init__(self, n_target: int, n_source: int, spec: Optional[MappingSpec]):
        self._n_target = int(n_target)
        self._n_source = int(n_source)
        self.geometry_shape = (self._n_target, self._n_source)
        self.spec = spec

    @property
    def n_source(self) -> int:
        return self._n_source

    @property
    def n_target(self) -> int:
        return self._n_target

    def params_pytree(self) -> dict:
        return {}

    def apply(self, field, weights: Optional[dict] = None, geom=None):
        return geom @ field

    def apply_T(self, field, weights: Optional[dict] = None, geom=None):
        return geom.T @ field

    def __repr__(self) -> str:
        return f"GeomMatrixMapping({self._n_target}x{self._n_source})"


def _build_geom_matrix(*, n_target: int, n_source: int) -> GeomMatrixMapping:
    spec = MappingSpec(GEOM_MATRIX, {"n_target": int(n_target), "n_source": int(n_source)}, {})
    return GeomMatrixMapping(n_target, n_source, spec)


_REGISTERED = False


def register_test_kinds() -> None:
    """Register ``test_geom_matrix`` for the rest of the process (once).

    Through the public ``register_mapping`` with ``needs_geometry=True``,
    as another library would register a geometry-dependent kind.  On a
    tree whose registry has no such flag this raises ``TypeError`` every
    time it is called, so each test that needs the kind fails by itself.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    register_mapping(GEOM_MATRIX, arrays=(), hyperparameters={"n_target": int, "n_source": int},
                     needs_geometry=True)(_build_geom_matrix)
    _REGISTERED = True


def geom_matrix_mapping(n_target: int, n_source: int) -> GeomMatrixMapping:
    """A registered ``test_geom_matrix`` mapping."""
    register_test_kinds()
    return _build_geom_matrix(n_target=n_target, n_source=n_source)


def multilinear(origin, spacing, shape, *, n_points: int, mode: str, **kw):
    """The library's ``multilinear_grid_mapping`` (imported here, so that a
    tree without the kind fails the test that asked for it, not the module)."""
    from maddening.core.coupling.grid_mapping import (  # noqa: PLC0415
        multilinear_grid_mapping,
    )
    return multilinear_grid_mapping(origin, spacing, shape, n_points=n_points, mode=mode, **kw)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class Body(SimulationNode):
    """A value field and the geometry fields that move with it.

    ``x <- a x_pre + g u + c`` with ``u`` the node's one input port (shape
    ``(n,)``: whatever a mapping hands it is already on this node's
    interface), and for every geometry field ``p``

    ``p <- p_pre + dt (rate D_p + adv Q_p u)``

    with ``D_p`` and ``Q_p`` constants fixed by ``seed``.  ``c`` (the
    value side) and ``rate`` (what moves the geometry) are parameters;
    ``adv != 0`` makes the geometry depend on the input, which inside a
    coupling group is the iterate.
    """

    def __init__(self, name, timestep, *, n, geoms=None, a=0.4, g=1.0, adv=0.0, seed=0,
                 dtype="float32", geom_dtype=None, geom_scale=1.0, rate=0.3):
        rng = np.random.default_rng(seed)
        self._dtype = jnp.dtype(dtype)
        self._gd = jnp.dtype(geom_dtype or dtype)
        c = rng.uniform(-0.5, 1.0, size=n)
        super().__init__(name, timestep, c=jnp.asarray(c, self._dtype),
                         rate=jnp.asarray(rate, self._gd))
        self._n = int(n)
        self._a, self._g, self._adv = float(a), float(g), float(adv)
        self._x0 = rng.uniform(-0.5, 1.0, size=n)
        self._geoms = {}
        for field, init in (geoms or {}).items():
            init = np.asarray(init, np.float64)
            drift = rng.uniform(-1.0, 1.0, size=init.shape) * geom_scale
            pull = rng.uniform(-1.0, 1.0, size=(init.size, n)) / n * geom_scale
            self._geoms[field] = (init, drift, pull)

    def initial_state(self):
        state = {"x": jnp.asarray(self._x0, self._dtype)}
        for field, (init, _drift, _pull) in self._geoms.items():
            state[field] = jnp.asarray(init, self._gd)
        return state

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=self._dtype,
                                       default=jnp.zeros(self._n, self._dtype))}

    def _u(self, boundary_inputs):
        return boundary_inputs.get("u", jnp.zeros(self._n, self._dtype))

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = self._u(boundary_inputs)
        dtype, gd = self._dtype, self._gd
        out = {"x": (jnp.asarray(self._a, dtype) * state["x"]
                     + jnp.asarray(self._g, dtype) * u + p["c"]).astype(dtype)}
        for field, (init, drift, pull) in self._geoms.items():
            move = p["rate"] * jnp.asarray(drift, gd)
            if self._adv:
                move = move + jnp.asarray(self._adv, gd) * (
                    jnp.asarray(pull, gd) @ u.astype(gd)).reshape(init.shape)
            out[field] = (state[field] + jnp.asarray(dt, gd) * move).astype(gd)
        return out

    def update_evaluations(self):
        return 1


class FluxBody(Body):
    """A :class:`Body` with the boundary flux ``q = 2 x + 0.5 u``: it reads the mapped input."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": jnp.asarray(2.0, self._dtype) * state["x"]
                + jnp.asarray(0.5, self._dtype) * self._u(boundary_inputs)}


class Reader(SimulationNode):
    """``x <- 0.5 x_pre + q``: the reader of a flux."""

    def __init__(self, name, timestep, *, n, dtype="float32"):
        super().__init__(name, timestep)
        self._n, self._dtype = int(n), jnp.dtype(dtype)

    def initial_state(self):
        return {"x": jnp.zeros(self._n, self._dtype)}

    def boundary_input_spec(self):
        return {"q": BoundaryInputSpec(shape=(self._n,), dtype=self._dtype,
                                       default=jnp.zeros(self._n, self._dtype))}

    def update(self, state, boundary_inputs, dt):
        q = boundary_inputs.get("q", jnp.zeros(self._n, self._dtype))
        return {"x": (jnp.asarray(0.5, self._dtype) * state["x"] + q).astype(self._dtype)}

    def update_evaluations(self):
        return 1


# ---------------------------------------------------------------------------
# A graph as data, and its node-inlined twin
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class GEdge:
    """``src.sf -> dst.tf`` through ``mapping`` (read at ``geometry``) then ``transform``."""

    src: str
    dst: str
    sf: str
    tf: str
    mapping: Any = None
    geometry: Optional[tuple] = None
    transform: Optional[str] = None
    additive: bool = False


@dataclasses.dataclass
class GGraph:
    """Nodes (in ``add_node`` order), edges (in ``add_edge`` order) and coupling groups."""

    nodes: list
    edges: list
    groups: list = dataclasses.field(default_factory=list)   # [(members, knobs)]

    def node(self, name: str):
        return next(nd for nd in self.nodes if nd.name == name)


@dataclasses.dataclass(frozen=True)
class _Inlined:
    tf: str
    mapping: Any
    anchor: str
    gfield: str
    transform: Any
    additive: bool
    value_spec: BoundaryInputSpec
    geometry_spec: Optional[BoundaryInputSpec]


class InlinedMappingNode(SimulationNode):
    """A node with the geometry-dependent mappings of its incoming edges inside it.

    Same name, timestep, state, parameters, parameter specs and evaluation
    count as the node it wraps.  For each inlined edge into port ``tf`` it
    takes ``"<tf>@value"`` (the source field, unmapped) and, for a source
    anchor, ``"<tf>@geometry"``; before handing the inner hook its inputs
    it computes ``v = mapping.apply(value, None, geom)`` -- ``geom`` the
    geometry port, or for a target anchor the field of the hook's own
    ``state`` argument -- applies the edge's transform, and adds ``v``
    into or sets port ``tf`` as the edge's additive flag says.
    """

    def __init__(self, inner: SimulationNode, inlined):
        super().__init__(inner.name, inner.delta_t, **dict(inner.params))
        self._wrapped = inner
        self._inlined = tuple(inlined)

    def initial_state(self):
        return self._wrapped.initial_state()

    def param_specs(self):
        return self._wrapped.param_specs()

    def update_evaluations(self):
        return self._wrapped.update_evaluations()

    def interface_dof_indices(self):
        return self._wrapped.interface_dof_indices()

    def boundary_input_spec(self):
        spec = dict(self._wrapped.boundary_input_spec())
        for it in self._inlined:
            spec[f"{it.tf}@value"] = it.value_spec
            if it.geometry_spec is not None:
                spec[f"{it.tf}@geometry"] = it.geometry_spec
        return spec

    def _inner_inputs(self, state, boundary_inputs):
        bi = dict(boundary_inputs)
        for it in self._inlined:
            value = bi.pop(f"{it.tf}@value", None)
            geom = (bi.pop(f"{it.tf}@geometry", None) if it.anchor == "source"
                    else state[it.gfield])
            if value is None:
                continue        # not resolved for this call (a flux seeded later)
            v = it.mapping.apply(value, None, geom)
            if it.transform is not None:
                v = it.transform(v)
            bi[it.tf] = bi[it.tf] + v if (it.additive and it.tf in bi) else v
        return bi

    def update(self, state, boundary_inputs, dt, *, params=None):
        return self._wrapped.update(state, self._inner_inputs(state, boundary_inputs), dt,
                                    params=params)

    def compute_interface_correction(self, pre_state, boundary_inputs, dt, *, params=None):
        return self._wrapped.compute_interface_correction(
            pre_state, self._inner_inputs(pre_state, boundary_inputs), dt, params=params)


class InlinedFluxMappingNode(InlinedMappingNode):
    """:class:`InlinedMappingNode` around a flux producer (a flux hook of its
    own, so the graph sees a producer exactly where the inner node is one)."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return self._wrapped.compute_boundary_fluxes(
            state, self._inner_inputs(state, boundary_inputs), dt, params=params)


def _produces_fluxes(node) -> bool:
    return type(node).compute_boundary_fluxes is not SimulationNode.compute_boundary_fluxes


def _source_value(node, field: str):
    """A source field's initial value: a state field, or a flux the node produces."""
    state = node.initial_state()
    if field in state:
        return state[field]
    return node.compute_boundary_fluxes(state, {}, 0.0)[field]


def _port_spec(value) -> BoundaryInputSpec:
    value = jnp.asarray(value)
    return BoundaryInputSpec(shape=tuple(value.shape), dtype=value.dtype,
                             default=jnp.zeros(value.shape, value.dtype))


def inline_geometry(graph: GGraph) -> GGraph:
    """*graph* with every geometry-dependent mapping moved inside its target node.

    Node names, the order of the nodes, the groups and their knobs are
    unchanged; each geometry edge keeps its place in the edge order as
    ``S.sf -> T."<tf>@value"``, followed for a source anchor by
    ``S.g -> T."<tf>@geometry"``.
    """
    edges: list = []
    inlined: dict = {}
    for k, e in enumerate(graph.edges):
        if e.geometry is None:
            edges.append(e)
            continue
        assert e.mapping is not None, e
        assert not any(later.dst == e.dst and later.tf == e.tf and later.geometry is None
                       for later in graph.edges[k + 1:]), (
            f"{e}: a geometry edge must be the last edge into its port")
        anchor, gfield = e.geometry
        source = graph.node(e.src)
        edges.append(GEdge(e.src, e.dst, e.sf, f"{e.tf}@value"))
        geometry_spec = None
        if anchor == "source":
            edges.append(GEdge(e.src, e.dst, gfield, f"{e.tf}@geometry"))
            geometry_spec = _port_spec(source.initial_state()[gfield])
        inlined.setdefault(e.dst, []).append(_Inlined(
            e.tf, e.mapping, anchor, gfield, resolve_transform(e.transform), e.additive,
            _port_spec(_source_value(source, e.sf)), geometry_spec))
    nodes = []
    for nd in graph.nodes:
        if nd.name in inlined:
            cls = InlinedFluxMappingNode if _produces_fluxes(nd) else InlinedMappingNode
            nd = cls(nd, inlined[nd.name])
        nodes.append(nd)
    return GGraph(nodes, edges, list(graph.groups))


def build(graph: GGraph, *, compile: bool = True) -> GraphManager:
    """A ``GraphManager`` of *graph*, compiled.

    ``geometry=`` is passed only for an edge that has one, so the inlined
    twin builds on any tree.  ``compile()``'s deprecation notice for
    ``solver="fori"`` is the only warning let through.
    """
    gm = GraphManager()
    for nd in graph.nodes:
        gm.add_node(nd)
    for e in graph.edges:
        kw = {} if e.geometry is None else {"geometry": e.geometry}
        gm.add_edge(e.src, e.dst, e.sf, e.tf, transform=e.transform, additive=e.additive,
                    mapping=e.mapping, **kw)
    for members, knobs in graph.groups:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", "CouplingGroup solver='fori' is deprecated", DeprecationWarning)
            gm.add_coupling_group(list(members), **knobs)
    if compile:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*multi-rate.*", UserWarning)
            gm.compile()
    return gm


# ---------------------------------------------------------------------------
# The two-body graph
# ---------------------------------------------------------------------------

DT = 0.1
#: Grid-side and point-side sizes of the ``geom_matrix`` flavour.
N_MATRIX, M_POINTS = 5, 4
#: The grids of the ``multilinear`` flavour, by dimension: a different size
#: on every axis.
GRIDS = {1: ((0.25,), (0.5,), (7,)),
         2: ((0.25, -1.0), (0.5, 0.25), (4, 3)),
         3: ((0.25, -1.0, 0.0), (0.5, 0.25, 2.0), (3, 2, 2))}
#: Index coordinates of the point-side body's points and of the grid-side
#: body's own (different) set, per axis: every fraction between 0.2 and
#: 0.8 of a cell, so three steps of motion cross no lattice plane.
_P_INDEX = np.asarray([[0.42, 0.34, 0.46], [1.34, 1.57, 0.61], [2.54, 0.66, 0.33],
                       [4.61, 1.41, 0.52]])
_F_INDEX = np.asarray([[1.37, 1.62, 0.58], [3.27, 0.44, 0.39], [0.61, 0.52, 0.64],
                       [5.44, 1.36, 0.47]])


@dataclasses.dataclass(frozen=True)
class Case:
    """One two-body graph: ``F`` (grid side) and ``P`` (point side).

    ``down`` is the anchor of the edge ``F.x -> P.u`` and ``up`` that of
    ``P.x -> F.u`` (``None``: no such edge); ``"source"`` reads the
    geometry from the edge's source, ``"target"`` from its target.  Both
    bodies hold a geometry field of the same name and shape with
    different values and different motion, so an anchor read from the
    wrong end is a different number.

    ``kind="geom_matrix"``: ``down`` multiplies by the field ``A`` (``M x
    N``), ``up`` by ``B`` (``N x M``).  ``kind="multilinear"``: ``down``
    gathers the grid field at the points ``pos`` (consistent) and ``up``
    scatters the point field onto the grid (conservative).

    ``flux="reader"`` makes ``P`` a flux producer whose flux reads its
    mapped input, read by a node ``R`` outside any group;
    ``flux="internal"`` sends that flux, not ``x``, up the ``up`` edge.
    ``extra`` adds a driver ``E`` with a plain additive edge into ``P.u``
    ahead of the ``down`` edge, which then is additive and carries the
    transform ``scale_0.5``.
    """

    label: str
    kind: str = "geom_matrix"
    d: int = 1
    order: tuple = ("F", "P")
    down: Optional[str] = "target"
    up: Optional[str] = "source"
    group: Optional[tuple] = None          # CouplingGroup knobs as sorted items
    dt_f: float = DT
    dt_p: float = DT
    adv: float = 0.0
    flux: Optional[str] = None
    extra: bool = False
    dtype: str = "float32"
    geom_dtype: Optional[str] = None
    steps: int = 3

    @property
    def knobs(self) -> Optional[dict]:
        return None if self.group is None else dict(self.group)

    @property
    def needs_x64(self) -> bool:
        return "float64" in (self.dtype, self.geom_dtype or self.dtype)

    def __repr__(self) -> str:      # the pytest id
        return self.label


def case(label: str, *, group: Optional[dict] = None, **kw) -> Case:
    return Case(label, group=None if group is None else tuple(sorted(group.items())), **kw)


def _grid_of(c: Case):
    return GRIDS[c.d]


def _positions(c: Case, index: np.ndarray) -> np.ndarray:
    origin, spacing, _shape = _grid_of(c)
    return np.asarray(origin) + index[:, :c.d] * np.asarray(spacing)


def geometry_fields(c: Case) -> dict:
    """``{body: {field: initial value}}`` for the case's two bodies."""
    if c.kind == "multilinear":
        return {"P": {"pos": _positions(c, _P_INDEX)}, "F": {"pos": _positions(c, _F_INDEX)}}
    out = {}
    for k, name in enumerate(("F", "P")):
        rng = np.random.default_rng(700 + k)
        out[name] = {"A": rng.uniform(-0.6, 0.6, size=(M_POINTS, N_MATRIX)),
                     "B": rng.uniform(-0.6, 0.6, size=(N_MATRIX, M_POINTS))}
    return out


def sizes(c: Case) -> tuple:
    """``(grid-side size, point-side size)``."""
    if c.kind == "multilinear":
        return int(np.prod(_grid_of(c)[2])), M_POINTS
    return N_MATRIX, M_POINTS


def _mappings(c: Case) -> tuple:
    """``(down mapping, its geometry field, up mapping, its geometry field)``."""
    n_grid, n_pts = sizes(c)
    if c.kind == "multilinear":
        origin, spacing, shape = _grid_of(c)
        return (multilinear(origin, spacing, shape, n_points=n_pts, mode="consistent"), "pos",
                multilinear(origin, spacing, shape, n_points=n_pts, mode="conservative"), "pos")
    return (geom_matrix_mapping(n_pts, n_grid), "A", geom_matrix_mapping(n_grid, n_pts), "B")


def holder(c: Case, edge: str) -> str:
    """The body that holds the geometry of the ``"down"`` or ``"up"`` edge."""
    anchor = c.down if edge == "down" else c.up
    ends = ("F", "P") if edge == "down" else ("P", "F")
    return ends[0] if anchor == "source" else ends[1]


def two_body(c: Case) -> GGraph:
    """The edge-mapped graph of *c* (fresh nodes and mappings on every call)."""
    n_grid, n_pts = sizes(c)
    geoms = geometry_fields(c)
    spacing = min(_grid_of(c)[1]) if c.kind == "multilinear" else 1.0
    common = dict(dtype=c.dtype, geom_dtype=c.geom_dtype, adv=c.adv, geom_scale=spacing)
    p_cls = FluxBody if c.flux else Body
    make = {
        "F": lambda: Body("F", c.dt_f, n=n_grid, geoms=geoms["F"], a=0.3, g=0.15, seed=11,
                          **common),
        "P": lambda: p_cls("P", c.dt_p, n=n_pts, geoms=geoms["P"], a=0.4, g=1.0, seed=12,
                           **common),
    }
    nodes = [make[name]() for name in c.order]
    edges = []
    down_map, down_field, up_map, up_field = _mappings(c)
    if c.extra:
        nodes.append(Body("E", c.dt_p, n=n_pts, a=0.5, seed=13, dtype=c.dtype))
        edges.append(GEdge("E", "P", "x", "u", additive=True))
    if c.down is not None:
        edges.append(GEdge("F", "P", "x", "u", mapping=down_map, geometry=(c.down, down_field),
                           transform="scale_0.5" if c.extra else None, additive=c.extra))
    if c.up is not None:
        edges.append(GEdge("P", "F", "q" if c.flux == "internal" else "x", "u",
                           mapping=up_map, geometry=(c.up, up_field)))
    if c.flux == "reader":
        nodes.append(Reader("R", c.dt_p, n=n_pts, dtype=c.dtype))
        edges.append(GEdge("P", "R", "q", "q"))
    groups = [] if c.group is None else [(("F", "P"), c.knobs)]
    return GGraph(nodes, edges, groups)


def graphs(c: Case) -> tuple:
    """``(edge-mapped GraphManager, node-inlined GraphManager)``, both compiled."""
    return build(two_body(c)), build(inline_geometry(two_body(c)))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------


def snapshot(gm: GraphManager) -> dict:
    """``{node: {field: numpy array}}`` of the graph's current state."""
    return {n: {f: np.asarray(v) for f, v in gm.get_node_state(n).items()}
            for n in gm.node_names}


def run_steps(gm: GraphManager, steps: int) -> list:
    """The state after each of *steps* calls of ``gm.step()`` from the initial state."""
    gm.reset_state()
    out = []
    for _ in range(steps):
        gm.step()
        out.append(snapshot(gm))
    return out


def rollout(gm: GraphManager, steps: int):
    """``(run, state0, params0)``: ``run(state, params)`` is *steps* steps of the
    graph's pure step in a ``lax.scan``, from the graph's own initial state
    and parameters by default."""
    step = gm._raw_step_fn                               # noqa: SLF001
    ext = gm._default_external_inputs()                  # noqa: SLF001
    gm.reset_state()
    state0 = gm._state                                   # noqa: SLF001

    def run(state, params):
        return jax.lax.scan(lambda carry, _: (step(carry, ext, params), None),
                            state, None, length=steps)[0]

    return run, state0, gm.params


def output_loss(state) -> Any:
    """Sum of squares of every node's ``x``: a loss on the graph's outputs."""
    return sum(jnp.sum(fields["x"].astype(jnp.result_type(float)) ** 2)
               for name, fields in sorted(state.items()) if not name.startswith("_"))


def with_theta(state, params, theta: dict):
    """*state* and *params* with ``theta``'s entries written in.

    Keys: ``("state", node, field)`` or ``("params", node, name)``.
    """
    state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in state.items()}
    params = {**params, "nodes": {k: dict(v) for k, v in params["nodes"].items()}}
    for (where, node, name), value in theta.items():
        if where == "state":
            state[node][name] = value
        else:
            params["nodes"][node][name] = value
    return state, params


def theta_of(c: Case, state0, params0) -> dict:
    """What a gradient is taken with respect to: the initial geometry of each
    geometry edge's holder, the ``rate`` that moves it, and the value-side
    parameter ``c`` of the grid-side body."""
    theta = {("params", "F", "c"): params0["nodes"]["F"]["c"]}
    fields = {"down": "A", "up": "B"} if c.kind == "geom_matrix" else {"down": "pos",
                                                                         "up": "pos"}
    for edge in ("down", "up"):
        if (c.down if edge == "down" else c.up) is None:
            continue
        node = holder(c, edge)
        theta[("state", node, fields[edge])] = state0[node][fields[edge]]
        theta[("params", node, "rate")] = params0["nodes"][node]["rate"]
    return theta


def loss_of(gm: GraphManager, c: Case):
    """``(loss, theta0)``: ``loss(theta)`` runs ``c.steps`` steps and sums the outputs."""
    run, state0, params0 = rollout(gm, c.steps)

    def loss(theta):
        state, params = with_theta(state0, params0, theta)
        return output_loss(run(state, params))

    return loss, theta_of(c, state0, params0)
