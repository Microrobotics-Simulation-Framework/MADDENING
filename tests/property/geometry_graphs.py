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

**The twins with the edge-mapped graph's interface reading**
(:func:`transform_twin`, :func:`relay_twin`; their docstrings say what
each holds equal) and **the faults later work is held to.**  The bounds
of a single-rate ``multilinear_grid`` group under ``"l2"`` and ``"mixed"``
read the geometry (``tests/property/test_coupling_geometry_search.py``
holds them to the numerical reference and lists the faults seeded on
them: 1, 2, 3 and 6 below are caught there per push).  The interface norm
as a criterion with a geometry edge, and the bounds in that reading with
the geometry term, do not exist in 0.4.0.  The table lists the faults
that work must be caught on, the instrument expected to catch each, and
-- where the fault can be seeded on today's tree in an analogous
static-mapping or solve-path form -- the signal measured when it was (scratch copies of ``src/``, jaxlib 0.11.0,
CPU; the instruments are ``tests/property/test_interface_reading_twins.py``
and, for the numerical reference, ``tests/property/coupling_reference.py``
and ``tests/property/test_coupling_nonlinear_search.py``):

==  =========================================  ==================================  ==========================================
#   fault                                      instrument                          analogue today, and the measured signal
==  =========================================  ==================================  ==========================================
1   the geometry term dropped from the         the numerical reference: its        none (a static mapping has no geometry
    reading's Jacobian                         Jacobian is ``jacfwd`` of the       term).  The reference's own sensitivity:
                                               pass with the geometry field in     a radius reported 10% low fails the
                                               the iterate (radius, bound and      per-push radius search; an error bound
                                               gradient scores); the relay         halved, or a gradient bound a twentieth,
                                               twin's ``rho_spectral``             fails a per-push seed (mutants R1 to R3).
2   the geometry read at the wrong time        the relay twin (residual, pass      solve path: a group's source-anchored
    level inside the reading                   count, ``rho_spectral``); the       geometry read from the pre-step state
                                               time-level reference of             in place of the dict its value is read
                                               ``test_geometry_time_levels.py``    from.  The relay twin's step-for-step
                                               for the solve it must agree with    equality fails after two passes: a
                                                                                   field 2.6e-4 apart, 5.4e-7 allowed, zero
                                                                                   unmutated (mutant F2-F3).  A converged
                                                                                   solve does not see it: at a fixed point
                                                                                   the iterate and the pass agree.
3   the geometry taken from the iterate        the relay twin, whose relay         the same seeded fault, which is the
    instead of the pre-step state (or the      computes the mapped value from      reverse (the pre-step state where the
    reverse)                                   the fields of one update; the       iterate was due): F2-F3.
                                               numerical reference
4   the geometry's own rounding left out of    the numerical reference's floor     not seeded.  The relay twin does not
    the precision floor                        score (the exact residual of the    judge a floor: its own is 13% from the
                                               returned state above the reported   edge-mapped graph's by construction
                                               one, over the reported floor) on    (a field read as stored against a
                                               cells with a float32 geometry       mapping the norm evaluates).
5   the mapping dropped from the interface     both twins                          seeded (mutant F5): the transform twin's
    reading                                                                        and the relay twin's equalities both
                                                                                   fail on the per-push case (residuals
                                                                                   8.8% apart, 1.2e-4 allowed).
6   the reading's Jacobian-vector product      ``PassReference                     a reading with a term its tangent does
    disagreeing with a finite difference       .finite_difference_gap`` (the       not see: gap 1e-3 or more where the
    (a later run-time self-check)              check itself, for the pass or       honest reading's is under 1e-7
                                               for any reading written in JAX)     (``test_the_reference_s_jacobian_is_...``).
==  =========================================  ==================================  ==========================================

**Seeded on the bounds under ``"l2"`` and ``"mixed"``** (the stage that made
them read a ``multilinear_grid`` geometry; scratch copies of ``src/``,
jaxlib 0.11.0, each caught by a per-push test of
``test_coupling_geometry_search.py`` unless another file is named):

* 1, in the estimator only (the position columns of the iterate behind
  ``stop_gradient`` in the product handed to the spectrum): the reference's
  radius score reads 3.0 (limit 1) and the radius seeds fail.  Seeded at
  the pass's own read of the geometry instead, the same fault reaches the
  reference's ``jacfwd`` too, which then agrees with the estimator: only
  the run-time self-check, a finite difference, sees it (a gap of 1,
  tolerance 0.25), and with the self-check's comparison disabled as well
  the radius seeds fail on their premise alone.
* 2, the product taken with the first pass's positions: radius score 25.
* 3, a target-anchored geometry read from the iterate (seeded in the pass:
  in this stage the product is the pass's): the node-inlined twin, a field
  5.4e-4 apart with 5.3e-7 allowed (``test_differential_geometry_edges.py``),
  and the time-level reference (``test_geometry_time_levels.py``).
* 6, the comparison disabled in the step (the gap returned as zero, or
  taken over all fields at once), in the report, or never traced: the
  self-check's own three tests.
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
    #: The ports the unmapped value and (source anchor) the geometry arrive
    #: on: ``"<tf>@value"`` and ``"<tf>@geometry"``, with ``"@<k>"`` appended
    #: for the ``k``-th further geometry edge into the same port.
    value_port: str
    geometry_port: str
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
            spec[it.value_port] = it.value_spec
            if it.geometry_spec is not None:
                spec[it.geometry_port] = it.geometry_spec
        return spec

    def _inner_inputs(self, state, boundary_inputs):
        bi = dict(boundary_inputs)
        for it in self._inlined:
            value = bi.pop(it.value_port, None)
            geom = (bi.pop(it.geometry_port, None) if it.anchor == "source"
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
        # A port of its own for each geometry edge: two into one target
        # port used to share ``"<tf>@value"``, where the second edge's
        # value replaced the first's before either was mapped.
        nth = sum(it.tf == e.tf for it in inlined.get(e.dst, []))
        suffix = f"@{nth}" if nth else ""
        value_port, geometry_port = f"{e.tf}@value{suffix}", f"{e.tf}@geometry{suffix}"
        edges.append(GEdge(e.src, e.dst, e.sf, value_port))
        geometry_spec = None
        if anchor == "source":
            edges.append(GEdge(e.src, e.dst, gfield, geometry_port))
            geometry_spec = _port_spec(source.initial_state()[gfield])
        inlined.setdefault(e.dst, []).append(_Inlined(
            e.tf, value_port, geometry_port, e.mapping, anchor, gfield,
            resolve_transform(e.transform), e.additive,
            _port_spec(_source_value(source, e.sf)), geometry_spec))
    nodes = []
    for nd in graph.nodes:
        if nd.name in inlined:
            cls = InlinedFluxMappingNode if _produces_fluxes(nd) else InlinedMappingNode
            nd = cls(nd, inlined[nd.name])
        nodes.append(nd)
    return GGraph(nodes, edges, list(graph.groups))


# ---------------------------------------------------------------------------
# The twin with an identical interface reading
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _Relayed:
    sf: str                     # the source field the edge read
    out: str                    # the state field that holds the mapped value
    mapping: Any
    gfield: Optional[str]       # the source's own geometry field, or None (a static mapping)
    leaves: tuple               # the mapping's weight leaves, held as node parameters


class RelayedSourceNode(SimulationNode):
    """A source node with the mappings of its outgoing edges inside it: the
    relay of :func:`relay_twin`.

    Same name, timestep and evaluation count as the node it wraps, the
    same state and one more field per relayed edge: ``out =
    mapping.apply(new[sf], weights, new[gfield])``, computed by ``update``
    from the fields it has just produced (and by ``initial_state`` from
    the initial ones), so in every state the graph holds -- and in every
    iterate of a group, which is a state some pass returned -- ``out`` is
    the mapped value of the ``sf`` beside it.  The mapping's weights are
    parameters of this node (``"<out>_<leaf>"``), so they stay constants
    of the step that a gradient or a per-step override can reach.
    """

    def __init__(self, inner: SimulationNode, relayed):
        params = dict(inner.params)
        for r in relayed:
            weights = r.mapping.params_pytree()
            for leaf in r.leaves:
                params[f"{r.out}_{leaf}"] = jnp.asarray(weights[leaf])
        super().__init__(inner.name, inner.delta_t, **params)
        self._wrapped = inner
        self._relayed = tuple(relayed)
        self._outs = frozenset(r.out for r in relayed)

    def _mapped(self, r: _Relayed, fields, p):
        weights = {leaf: p[f"{r.out}_{leaf}"] for leaf in r.leaves} or None
        if r.gfield is None:
            return r.mapping.apply(fields[r.sf], weights)
        return r.mapping.apply(fields[r.sf], weights, fields[r.gfield])

    def initial_state(self):
        state = dict(self._wrapped.initial_state())
        for r in self._relayed:
            assert r.sf in state, f"{self.name}.{r.sf} is not a state field (a flux is not relayed)"
            state[r.out] = self._mapped(r, state, self.params)
        return state

    def param_specs(self):
        return self._wrapped.param_specs()

    def update_evaluations(self):
        return self._wrapped.update_evaluations()

    def boundary_input_spec(self):
        return self._wrapped.boundary_input_spec()

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        inner = {k: v for k, v in state.items() if k not in self._outs}
        new = dict(self._wrapped.update(inner, boundary_inputs, dt, params=params))
        for r in self._relayed:
            new[r.out] = self._mapped(r, new, p)
        return new


def read_at_its_source(e: GEdge) -> bool:
    """Does ``convergence_norm="interface"`` read static mapped edge *e* at
    its source, before the mapping?

    Under the compact-side rule (``coupled_topologies.INTERFACE_SIDE``):
    where the mapping delivers more entries than its source field holds.
    Such an edge's reading is the source field itself, so a twin that
    moves the mapping onto the *delivered* side of the reading (a relay
    on the source, a transform) reads something else there: the two twins
    below leave such an edge as it is (``keep=read_at_its_source``), and
    the twin that moves its mapping is the one with the mapping inside
    the target (``interface_side_graphs.marker_side_twin``).
    """
    from tests.property import coupled_topologies as ct  # noqa: PLC0415

    return (ct.INTERFACE_SIDE == "compact" and e.mapping is not None and e.geometry is None
            and int(e.mapping.n_target) > int(e.mapping.n_source))


def relay_twin(graph: GGraph, *, keep=None) -> GGraph:
    """*graph* with every mapped edge's mapping moved into a relay on its
    source, so that the twin's **interface reading is the edge-mapped
    graph's** wherever the norm reads an edge as delivered.  An edge
    *keep* names (a predicate; :func:`read_at_its_source` for a comparison
    under the interface norm) stays as it is: read at its source, its
    reading is not the relay's field.

    ``convergence_norm="interface"`` reads such an internal edge as it
    is delivered: for ``S.sf -> T.tf`` through mapping ``m`` and transform
    ``t``, ``t(m(S.sf))``.  The node-inlined twin of
    :func:`inline_geometry` moves ``m`` into the *target*, so its edge
    delivers the raw ``S.sf`` and its interface norm is another norm: the
    two graphs' diagnostics agree only as far as two norms of one solve
    do.  Here the mapped value is a state field of the source,
    ``S."<sf>_to_<T>_<tf>" = m(S.sf)``, and the edge is the plain edge
    ``S."<sf>_to_<T>_<tf>" -> T.tf`` with the same transform and additive
    flag in the same place of the edge order: it delivers ``t(m(S.sf))``,
    the same reading entry for entry, with the mapping (and, for a source
    anchor, its geometry) inside the relay.

    The relay is fused with the source (:class:`RelayedSourceNode`) and is
    not a node of its own: a separate relay in the group would be fed by
    an internal edge ``S.sf -> relay`` that the interface norm reads too,
    and outside the group it would sit in the group's cycle.

    What the twin holds equal by construction, and what it does not:

    * equal: every value a node reads, so the fixed point and every
      iterate of an unaccelerated solve; the interface norm's reading, so
      the residual, the pass count, the verdict, the spectrum taken on the
      reading (``rho_spectral``, ``spectral_error_bound``) and the flags;
    * not equal: anything that reads the members' *state*, which has the
      relay's field in it here -- the ``"l2"`` and ``"mixed"`` norms, an
      accelerator's secant vectors, the gradient bound's norm (the raw
      source fields: ``S.sf`` there, the relay's field here) -- and the
      float floor, which counts an inner product for a mapping the norm
      evaluates and none for a field read as stored.

    A geometry-dependent mapping is relayed where its geometry is its
    source's (``anchor="source"``: the relay reads the geometry field
    beside the value, both as the update has just produced them).  A
    target-anchored geometry is refused: the relay would need the
    target's geometry over one more internal edge, which the norm would
    read.  A flux source field is refused too (it is not state).
    """
    edges: list = []
    relayed: dict = {}
    for e in graph.edges:
        if e.mapping is None or (keep is not None and keep(e)):
            edges.append(e)
            continue
        gfield = None
        if e.geometry is not None:
            anchor, gfield = e.geometry
            if anchor != "source":
                raise NotImplementedError(
                    f"{e}: a target-anchored geometry has no relay twin (the relay would read "
                    f"the target's geometry over an internal edge of its own)")
        out = f"{e.sf}_to_{e.dst}_{e.tf}"
        leaves = tuple(sorted(e.mapping.params_pytree()))
        relayed.setdefault(e.src, []).append(_Relayed(e.sf, out, e.mapping, gfield, leaves))
        edges.append(GEdge(e.src, e.dst, out, e.tf, transform=e.transform, additive=e.additive))
    nodes = [RelayedSourceNode(nd, relayed[nd.name]) if nd.name in relayed else nd
             for nd in graph.nodes]
    return GGraph(nodes, edges, list(graph.groups))


def transform_twin(graph: GGraph, *, keep=None) -> GGraph:
    """*graph* with every *static* mapping written as its edge's transform:
    the same nodes, the same state, plain edges, and on each formerly
    mapped edge the transform ``v -> t(m(v))`` (the mapping with its own
    weights, then the edge's transform ``t``).  An edge *keep* names (a
    predicate; :func:`read_at_its_source` for a comparison under the
    interface norm) stays as it is.

    The interface norm reads an edge as delivered, through its mapping
    and then its transform -- unless the mapping delivers more entries
    than its source holds, when it reads the source field and a transform
    in the mapping's place would be read as delivered (*keep*) -- so the
    twin's reading is the edge-mapped graph's and --
    unlike :func:`relay_twin` -- so is its state.  A mapping that reads a
    geometry cannot be written this way (a transform sees the value
    only), which is what :func:`relay_twin` is for.  The weights are
    constants of the transform here and parameters of the step there, so
    the gradient bound, which probes every parameter, is the one number
    the two graphs do not share.
    """
    edges = []
    for e in graph.edges:
        if e.mapping is None or (keep is not None and keep(e)):
            edges.append(e)
            continue
        assert e.geometry is None, f"{e}: a geometry-dependent mapping is not a transform"
        then = resolve_transform(e.transform)

        def through(v, _m=e.mapping, _t=then):
            v = _m.apply(v, None)
            return v if _t is None else _t(v)

        edges.append(GEdge(e.src, e.dst, e.sf, e.tf, transform=through, additive=e.additive))
    return GGraph(list(graph.nodes), edges, list(graph.groups))


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


# =============================================================================
# PHASE 1 OF GEOMETRY EDGES (0.4.0): THE ONE PLACE THIS HARNESS IS NARROWED
#
# The harness was written for diagnostics that read a moving geometry.
#
# **What reads one now** (``DIAGNOSTICS_READ_GEOMETRY = True``): a
# single-rate coupling group under ``convergence_norm="l2"`` or ``"mixed"``
# whose geometry edges all carry the ``multilinear_grid`` kind reports
# every bound and flag as any other group does, and the tests compare every
# key of its report with the node-inlined twin's (:func:`withheld` returns
# ``None`` for the case).
#
# **What still does not**, and what the library says instead of reporting
# numbers that leave the geometry out:
#
# * a group with a geometry edge of another kind (``test_geom_matrix``
#   here), or a sub-cycled one, reports the solve's own outcome
#   (``SOLVE_OUTCOME``) and nothing else: every bound NaN (the gradient
#   estimate ``inf``), every ``*_usable`` flag False, and
#   ``not_usable_reason`` saying which of the two it is
#   (:func:`withheld`, :func:`assert_not_diagnosed`);
# * ``convergence_norm="interface"`` on a group with a geometry-dependent
#   mapping on an internal edge is refused by ``compile()``
#   (``INTERFACE_NORM_READS_GEOMETRY = False``,
#   :func:`assert_interface_norm_refused`).
#
# Nothing was deleted: set ``INTERFACE_NORM_READS_GEOMETRY = True`` when a
# later stage makes the interface norm read the geometry, and the cases
# that ran it run again.
#
# Waiting for that stage too: ``RELAY_INTERFACE_CASES`` (defined after
# ``case``, below), the interface norm over source-anchored geometry edges,
# whose relay twin (``relay_twin``) has the edge-mapped graph's interface
# reading.  ``tests/property/test_interface_reading_twins.py`` asserts each
# refused today and its relay twin reporting; with the switch set it
# compares the two reports as it compares a static mapping's today.
# =============================================================================

#: Whether coupling diagnostics account for a moving geometry at all (the
#: ``multilinear_grid`` kind on a single-rate group under ``"l2"`` or
#: ``"mixed"``: see :func:`withheld`).
DIAGNOSTICS_READ_GEOMETRY = True
#: Whether the interface norm reads a geometry (and compiles over one).
INTERFACE_NORM_READS_GEOMETRY = False
#: What a report still says about a group whose bounds are withheld.
SOLVE_OUTCOME = ("iterations", "total_iterations", "residual", "converged")
_NOT_USABLE = {"amplification": "nan", "error_estimate": "nan", "ratio_usable": False,
               "gradient_error_estimate": "inf", "rho_spectral": "nan",
               "spectral_error_bound": "nan", "spectral_usable": False,
               "gradient_relative_error_bound": "nan", "gradient_bound_usable": False,
               "precision_limited": False}
#: What each reason for withholding says, beside the stem every one has.
WHY = {"kind": "other than 'multilinear_grid'",
       "sub-cycled": "in a sub-cycled group",
       "norm": "under convergence_norm='interface'",
       "self-check": "disagrees with a finite difference of the pass"}


def interface_norm_refused(knobs) -> bool:
    """Whether a group with these knobs and a geometry-dependent mapping on
    an internal edge is refused at compile (phase 1)."""
    return (not INTERFACE_NORM_READS_GEOMETRY and knobs is not None
            and dict(knobs).get("convergence_norm") == "interface")


def withheld(c: "Case") -> Optional[str]:
    """Why the report of *c*'s group withholds its bounds (a key of
    :data:`WHY`), or ``None`` where the diagnostics read its geometry."""
    if not DIAGNOSTICS_READ_GEOMETRY:
        return "kind"
    if c.kind != "multilinear":
        return "kind"
    if c.dt_f != c.dt_p and dict(c.knobs or {}).get("subcycling"):
        return "sub-cycled"
    return None


def assert_not_diagnosed(report, keys, why: Optional[str] = None) -> None:
    """*report* (one group of ``coupling_diagnostics()``) says the solve's
    outcome, no bound, no usable flag, and why, naming the edges *keys*;
    *why* is the key of :data:`WHY` the reason must be."""
    assert set(report) == {*SOLVE_OUTCOME, *_NOT_USABLE, "not_usable_reason"}, sorted(report)
    for name, want in _NOT_USABLE.items():
        got = report[name]
        if want == "nan":
            assert np.isnan(got), (name, got)
        elif want == "inf":
            assert got == float("inf"), (name, got)
        else:
            assert got is want, (name, got)
    reason = report["not_usable_reason"]
    assert isinstance(reason, str) and "geometry-dependent mapping" in reason, reason
    assert "do not read a moving geometry" in reason, reason
    for key in keys:
        assert key in reason, (key, reason)
    if why is not None:
        assert WHY[why] in reason, (why, reason)
        assert not any(text in reason for name, text in WHY.items() if name != why), reason
    assert np.isfinite(report["residual"]) and int(report["iterations"]) >= 1, report


def assert_interface_norm_refused(make, keys) -> None:
    """``make()`` builds and compiles a graph whose group uses the interface
    norm over the geometry edges *keys*: a ``RuntimeError`` from ``compile()``
    naming each edge and the norms that work."""
    import pytest  # noqa: PLC0415

    with pytest.raises(RuntimeError) as refused:
        make()
    message = str(refused.value)
    for phrase in ("convergence_norm='interface'", "geometry-dependent mapping",
                   "does not read a moving geometry in 0.4.0",
                   "Use convergence_norm='mixed' or 'l2'", *keys):
        assert phrase in message, (phrase, message)


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
#: body's own (different) set, by dimension: inside the hull, every
#: fraction between 0.2 and 0.8 of a cell, so three steps of motion cross
#: no lattice plane.
_INDEX = {
    1: (np.asarray([[0.42], [1.34], [2.54], [4.61]]),
        np.asarray([[1.37], [3.27], [0.61], [5.44]])),
    2: (np.asarray([[0.42, 0.34], [1.34, 1.57], [2.54, 0.66], [0.61, 1.41]]),
        np.asarray([[1.37, 1.62], [2.27, 0.44], [0.61, 0.52], [2.44, 1.36]])),
    3: (np.asarray([[0.42, 0.34, 0.46], [1.34, 0.57, 0.61], [1.54, 0.66, 0.33],
                    [0.61, 0.41, 0.52]]),
        np.asarray([[1.37, 0.62, 0.58], [0.27, 0.44, 0.39], [0.61, 0.52, 0.64],
                    [1.44, 0.36, 0.47]])),
}


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
    #: The rate every geometry field moves at; with ``rate=0`` and ``adv=0``
    #: the geometry is frozen.
    rate: float = 0.3

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


#: PHASE 1 (the block of that name above): the cases whose interface
#: diagnostics are held to their relay twin's once diagnostics read a
#: geometry.  Source anchors on both edges (a target-anchored geometry has
#: no relay twin), the geometry moving with the iterate, Gauss-Seidel (the
#: relay doubles the scalars a Jacobi pass carries, past what the spectrum
#: resolves).
RELAY_INTERFACE_CASES = [
    case("geometry edges, interface norm, three passes", adv=0.3, dtype="float64",
         down="source", up="source",
         group=dict(max_iterations=3, convergence_norm="interface", rtol=1e-6,
                    diagnostics=True)),
    case("multilinear geometry edges, interface norm, three passes", kind="multilinear",
         adv=0.3, down="source", up="source",
         group=dict(max_iterations=3, convergence_norm="interface", rtol=1e-6,
                    diagnostics=True)),
]


def _grid_of(c: Case):
    return GRIDS[c.d]


def _positions(c: Case, index: np.ndarray) -> np.ndarray:
    origin, spacing, _shape = _grid_of(c)
    return np.asarray(origin) + index * np.asarray(spacing)


def geometry_fields(c: Case) -> dict:
    """``{body: {field: initial value}}`` for the case's two bodies."""
    if c.kind == "multilinear":
        return {"P": {"pos": _positions(c, _INDEX[c.d][0])},
                "F": {"pos": _positions(c, _INDEX[c.d][1])}}
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
    common = dict(dtype=c.dtype, geom_dtype=c.geom_dtype, adv=c.adv, geom_scale=spacing,
                  rate=c.rate)
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


def static_matrix(c: Case, edge: str) -> np.ndarray:
    """The fixed matrix a geometry edge of *c* applies while its geometry
    stays where it starts: the holder's matrix itself, or for the
    multilinear kind the gather matrix of the holder's points from the
    independent reference stencil (its transpose for the scatter edge)."""
    from tests.core import multilinear_reference as mref  # noqa: PLC0415

    start = geometry_fields(c)[holder(c, edge)]
    gd = np.dtype(c.geom_dtype or c.dtype)
    if c.kind == "geom_matrix":
        return np.asarray(start["A" if edge == "down" else "B"], gd).astype(np.float64)
    gather = mref.dense_matrix(mref.Grid(*_grid_of(c)), np.asarray(start["pos"], gd))
    return gather if edge == "down" else gather.T


def static_twin(c: Case) -> GGraph:
    """*c*'s graph with every geometry edge replaced by a static ``matrix_mapping``
    of :func:`static_matrix` (the bodies still hold their geometry fields)."""
    from maddening.core.coupling.mapping import matrix_mapping  # noqa: PLC0415

    graph = two_body(c)
    edges = []
    for e in graph.edges:
        if e.geometry is not None:
            which = "down" if e.src == "F" else "up"
            e = dataclasses.replace(e, geometry=None, mapping=matrix_mapping(
                np.asarray(static_matrix(c, which), np.dtype(c.dtype))))
        edges.append(e)
    return GGraph(graph.nodes, edges, graph.groups)


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
