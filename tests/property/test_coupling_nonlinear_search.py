"""The coupling search on nonlinear groups, scored against a numerical float64 reference.

``test_coupling_targeted_search.py`` scores ``coupling_diagnostics()`` on
groups of *linear* relays, whose fixed point and Jacobian are closed
forms.  Its own docstring names what that leaves out: a nonlinear node
(the Newton-Kantorovich factor of the gradient bound is exactly one on an
affine map, and the Jacobian is the same at every iterate).  This module
adds nonlinear cells to that search, scored by the same four scores
against :mod:`tests.property.coupling_reference` -- the pass map of an x64
twin of the graph, its fixed point by Newton and its dense Jacobian by
``jax.jacfwd``.

**The reference is validated first** (the first tests below): on the
linear cells of the search it reproduces the closed form's fixed point,
Jacobian, spectral radius, distance, residual and gradient error to
float64 rounding, and several passes of an iterating twin are the same
number of compositions of its single pass.

**The nonlinear cells** (:data:`CELLS`) are structures of the linear
search with every member a :class:`NRelay`: the linear relay with each
port's input ``u`` passed through a nonlinearity ``phi`` centred on a
parameter ``c`` before its gain,

* ``"saturating"``: ``c + tanh(s (u - c)) / s`` (a saturating gain);
* ``"quadratic"``: ``u + s (u - c)^2`` (a quadratic term);
* ``"product"``: ``u_j + s (u_j - c_j) (u_j' - c_j')`` with ``j'`` the
  node's next port (the product of two fields' deviations; a node with
  one port squares its own).

``phi(c) = c`` and ``phi'(c) = 1`` for each, so with ``c`` the input at
the linear group's fixed point the nonlinear group has *the same fixed
point and the same Jacobian there* as the linear one: the numbers of a
case are drawn exactly as the linear search draws them (loop gain,
non-normal gains, a small field, a change of units, the start's offset:
:func:`tests.property.test_coupling_targeted_search.values_of`), and the
linear closed form is a second, independent check of the reference's
fixed point on every nonlinear example.  One more number is drawn, the
*curve* ``s max|c|``: how nonlinear the map is over a field's own size.
Away from the fixed point -- where a capped or early-stopped solve
returns -- the Jacobian is another matrix, and only the numerical
reference knows it.

A cell's index is its name: the pins at the foot of the module say
``Case(26, ...)``.  The 43 cells of the rotation (:data:`ROTATED`) keep
their places, a new configuration is appended (:data:`APPENDED`), and
``test_coupling_search_cells_are_pinned.py`` holds every cell's
configuration per push.

**The scores** are those of the linear search, restated for a map that
is not affine:

1. *bound* (CPL-088, which claims the bound for a linear ``F`` and calls
   it asymptotic for a nonlinear one): the true distance over
   ``spectral_error_bound`` times ``1 / (1 - h)``, with ``h`` the
   reference's measure of how far the Jacobian moves between the returned
   iterate and the fixed point
   (:meth:`~tests.property.coupling_reference.PassReference.nonlinearity`).
   ``x - x* = (I - J_mean)^{-1} (x - F(x))`` exactly, so a bound that is
   right for the linearisation at the returned iterate can be short by
   that factor and no more: a score over one is a wrong number, not a
   nonlinear map.  Not scored where ``h >= 1`` (nothing taken at the
   returned iterate bounds the distance; the fraction is reported).
2. *radius* / *radius_strict* (CPL-087: "the spectral radius of dF/dx at
   the returned state"): as the linear search scores them, against the
   reference's Jacobian **at the returned iterate**.
3. *gradient* (CPL-093, CPL-095): the true relative error of the implicit
   derivative taken at the returned iterate, the worst over the scalar
   constants, over ``gradient_relative_error_bound`` where
   ``gradient_bound_usable`` -- the flag that certifies the
   Newton-Kantorovich check, which on these cells is not trivially
   passed.  Two scores, by the constant: ``"gradient"`` over the gains,
   biases and mapping weights, and ``"gradient_vanishing"`` over the
   centre ``c`` and the curve ``s`` of every nonlinearity, which the
   fixed point does not respond to (``phi`` depends on neither at ``u =
   c``) and every other iterate does, so that ``|g_k - g*|`` is about
   ``|g_k|``: the relative error is of order one however close the
   iterate is, and a usable bound below that is short for that constant.
   Both allow what the float64 reference itself cannot resolve
   (:meth:`~tests.property.coupling_reference.PassReference.gradient_resolution`).
4. *floor* (CPL-097, CPL-100): the exact residual of the returned state
   above the reported one, over the reported floor.

Per push: the validation on one linear cell and each score on
:data:`PER_PUSH_CELLS` at the house floor of examples.  Slow: the
validation on every per-push draw of the linear search, and the hunt over
every nonlinear cell.  What goes over a threshold is pinned at the foot
of the module, as in the linear search.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import strategies as st
from jax.flatten_util import ravel_pytree

from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.property import coupled_graphs as cg
from tests.property import coupled_topologies as ct
from tests.property import coupling_reference as cr
from tests.property import test_coupling_targeted_search as linear
from tests.property.sysid_transform_grid import precision
from tests.property.targeted_search import PER_PUSH, SLOW, targeted_search

EPS64 = cr.EPS64


# ---------------------------------------------------------------------------
# What both halves share: the norm's reading on a topology of relays
# ---------------------------------------------------------------------------


def edge_fields(topo: ct.Topology, values: dict, ref: cr.PassReference, *, raw: bool = False):
    """The interface norm's reading of a flat iterate: what each internal
    edge delivers (its source's ``x`` through the edge's ``H`` and its
    transform's factor), or with *raw* the source field it reads -- one
    field per internal edge either way, as that norm counts them."""
    edges = [(e, np.asarray(values["H"][i], np.float64) if e.mapped else None)
             for i, e in enumerate(topo.edges) if topo.internal(e)]

    def fields(x):
        out = []
        for e, H in edges:
            src = np.asarray(ref.field(x, e.src, "x"), np.float64)
            if raw:
                out.append(src)
            else:
                out.append(ct.TRANSFORM_FACTORS[e.transform] * (src if H is None else H @ src))
        return out

    return fields


def norms_of(cfg: dict, topo: ct.Topology, values: dict, ref: cr.PassReference) -> tuple:
    """``(norm, raw norm)`` of a group under *cfg*: the norm its error
    bound is stated in, and the one its gradient bound is (the raw source
    fields under ``"interface"``; the same norm otherwise)."""
    kind, rtol = cfg["convergence_norm"], cfg["rtol"]
    if kind != "interface":
        norm = ref.norm(kind, rtol)
        return norm, norm
    return (ref.norm(kind, rtol, edge_fields(topo, values, ref)),
            ref.norm(kind, rtol, edge_fields(topo, values, ref, raw=True)))


def state_weights(ref: cr.PassReference, x: np.ndarray) -> np.ndarray:
    """``1 / max|field|`` per entry of flat iterate *x*."""
    w = np.ones(ref.size)
    for _n, _f, _shape, a, b in ref.layout:
        top = float(np.max(np.abs(x[a:b])))
        w[a:b] = 1.0 / top if top > 0 else 1.0
    return w


def bound_reference(ref: cr.PassReference, values: dict, built_twin: ct.Built) -> cr.PassReference:
    """*ref* bound to the start and the constants of *values* on its twin."""
    with cr.x64():
        ct.set_initial(built_twin, values)
        params = params_for(built_twin, values)
        return ref.at(built_twin.gm._state, params)       # noqa: SLF001


# ---------------------------------------------------------------------------
# The reference against the closed form, on the linear cells
# ---------------------------------------------------------------------------


#: The example a twin's layout probe steps (see ``PassReference.of``).
_PROBE = linear.Case(0, 1, 0.6, False, 1.0, 0.0, 1.0, 0, 0.5, 0)
#: The linear search's multi-rate cell, found by what it is (the cell's
#: place in ``linear.CELLS`` is the linear search's to keep).
_LINEAR_MULTIRATE = linear.CELLS.index(linear.Cell("ring-3-multirate", "float32", 6, 5))


@functools.lru_cache(maxsize=8)
def _linear_twin(index: int) -> tuple:
    cell = linear.CELLS[index]
    with cr.x64():
        built = ct.build(cell.topo, cr.twin_knobs(cell.knobs), dtype="float64",
                         mapping_kind=cell.mapping_kind)
        # A drawn example, so that the members' fields differ after a step
        # and the predictor slot's layout is determined.
        values = linear.values_of(dataclasses.replace(_PROBE, cell=index))
        ct.set_initial(built, values)
        return built, cr.PassReference.of(built.gm, params=ct.params_for(built, values))


def against_the_closed_form(case: linear.Case) -> dict:
    """How far each answer of the numerical reference is from the closed
    form's, on one drawn linear case (the step the search itself scores)."""
    cell = linear.CELLS[case.cell]
    topo = cell.topo
    values = linear.values_of(case)
    with precision(cell.dtype == "float64"):
        (step,) = ct.run(linear._built(case.cell), values, 1)   # noqa: SLF001
    out = dict(finite=all(np.all(np.isfinite(s["x"])) for s in step.state.values()))
    if not out["finite"]:
        return out
    model = ct.LinearModel(topo, values, dtype=cell.dtype, group_cfgs=cell.cfgs)
    built_twin, ref = _linear_twin(case.cell)
    ref = bound_reference(ref, values, built_twin)
    x = ref.flat(step.state)
    fixed = ref.fixed_point(x)
    members, off, _k = model._group_layout(0)             # noqa: SLF001
    # The model's stacking order, in the reference's.
    perm = np.concatenate([np.arange(off[n], off[n] + topo.node(n).n)
                           for n, _f, _s, _a, _b in ref.layout])
    exact = model.group_fixed_point(0, step.pre, step.state)
    x_star = np.concatenate([np.asarray(exact[m], np.float64) for m in members])[perm]
    J_exact = linear._pass_jacobian(model)[np.ix_(perm, perm)]   # noqa: SLF001
    J = ref.jacobian(fixed.x)
    norm, raw = norms_of(cell.cfgs[0], topo, values, ref)
    resolvent = float(np.linalg.norm(np.linalg.inv(np.eye(ref.size) - J_exact), 2))
    dist, dist_exact = ref.distance(x, fixed, norm), model.returned_weight_distance(
        0, step.pre, step.state)
    _dn, _b, detail = model.group_report_consistency(0, step.pre, step.state, 0.0)
    # The reference differentiates the twin's own parameter tree; the
    # closed form is told how the cell holds its mapped edges, so that it
    # takes the same constants (a sparse edge has none outside its pattern).
    grad, grad_exact = ref.gradient_error(x, fixed, raw)[0], linear._gradient_error(   # noqa: SLF001
        model, step.pre, step.state, cell.mapping_kind)
    scale = max(float(np.max(np.abs(x_star))), np.finfo(np.float64).tiny)
    # A relative norm divides a field by ``rtol`` times its size: one float64
    # rounding of a field is this much of the norm's unit.
    unit = EPS64 / norm.rtol
    out.update(
        converged=fixed.converged, ulps=fixed.ulps, resolvent=resolvent,
        fixed_point=float(np.max(np.abs(fixed.x - x_star))) / (EPS64 * scale * resolvent),
        jacobian=float(np.max(np.abs(J - J_exact))) / (
            EPS64 * max(float(np.max(np.abs(J_exact))), np.finfo(np.float64).tiny)),
        radius=abs(cr.radius(J) - linear._radius(J_exact)),   # noqa: SLF001
        distance=abs(dist - dist_exact), distance_exact=dist_exact, norm_unit=unit,
        residual=abs(ref.residual(x, norm) - detail["residual_true"]),
        residual_exact=detail["residual_true"],
        gradient=abs(grad - grad_exact), gradient_exact=grad_exact)
    return out


#: What "to float64 rounding" allows each difference, as measured over
#: every per-push draw of the linear search (see
#: ``test_the_reference_reproduces_the_closed_form_on_every_per_push_draw``):
#: the fixed point in float64 ``eps`` of the field times the resolvent's
#: norm; the Jacobian in ``eps`` of its largest entry; the radius absolute;
#: the distance and the residual in the norm's own rounding unit ``eps /
#: rtol`` times the resolvent's norm (a relative norm divides a field by
#: ``rtol`` times its size); the gradient error absolute beside one.
ALLOWED = dict(fixed_point=2.0 ** 10, jacobian=2.0 ** 6, radius=2.0 ** 10 * EPS64,
               distance=2.0 ** 12, residual=2.0 ** 12, gradient=1e-9)


def closed_form_misses(seen: dict) -> dict:
    """The differences of :func:`against_the_closed_form` over what
    :data:`ALLOWED` gives each (at most 1 where the reference agrees)."""
    if not seen["finite"]:
        return {}
    unit = seen["norm_unit"] * max(seen["resolvent"], 1.0)
    return dict(
        converged=0.0 if seen["converged"] else math.inf,
        fixed_point=seen["fixed_point"] / ALLOWED["fixed_point"],
        jacobian=seen["jacobian"] / ALLOWED["jacobian"],
        radius=seen["radius"] / ALLOWED["radius"],
        distance=seen["distance"] / (ALLOWED["distance"] * (
            unit + EPS64 * seen["distance_exact"])),
        residual=seen["residual"] / (ALLOWED["residual"] * (
            unit + EPS64 * seen["residual_exact"])),
        gradient=seen["gradient"] / (ALLOWED["gradient"] * max(seen["gradient_exact"], 1.0)))


# ---------------------------------------------------------------------------
# The nonlinear relay and its cells
# ---------------------------------------------------------------------------

KINDS = ("saturating", "quadratic", "product")


def phi(kind: str, u: list, c: list, s: list, xp=np) -> list:
    """Each port's input through the nonlinearity (module docstring);
    ``xp`` is ``numpy`` (the restatement the scores use) or ``jax.numpy``."""
    d = [uj - cj for uj, cj in zip(u, c)]
    k = len(u)
    if kind == "saturating":
        return [c[j] + xp.tanh(s[j] * d[j]) / s[j] for j in range(k)]
    if kind == "quadratic":
        return [u[j] + s[j] * d[j] * d[j] for j in range(k)]
    assert kind == "product", kind
    return [u[j] + s[j] * d[j] * d[(j + 1) % k] for j in range(k)]


class NRelay(SimulationNode):
    """``x <- alpha x_pre + b + sum_j G_j phi(u_j; c_j, s_j)``, at any float
    dtype: :class:`~tests.property.coupled_topologies.TRelay` with a
    nonlinearity on every port.  Every constant is a parameter of the
    step, so a compiled graph serves every draw."""

    def __init__(self, name, timestep, *, kind, n, ports, alpha=0.0, dtype="float32"):
        dt_ = jnp.dtype(dtype)
        params = {"b": jnp.zeros(n, dt_)}
        for j in range(ports):
            params[f"G{j}"] = jnp.zeros((n, n), dt_)
            params[f"c{j}"] = jnp.zeros(n, dt_)
            params[f"s{j}"] = jnp.ones((), dt_)
        super().__init__(name, timestep, **params)
        self._kind, self._n, self._k = kind, int(n), int(ports)
        self._alpha, self._dtype = float(alpha), dt_

    def initial_state(self):
        return {"x": jnp.zeros(self._n, self._dtype)}

    def boundary_input_spec(self):
        return {f"u{j}": BoundaryInputSpec(shape=(self._n,), dtype=self._dtype,
                                           default=jnp.zeros(self._n, self._dtype))
                for j in range(self._k)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        dtype = self._dtype
        f = jnp.asarray(self._alpha, dtype) * state["x"] + p["b"]
        u = [boundary_inputs.get(f"u{j}", jnp.zeros(self._n, dtype)) for j in range(self._k)]
        v = phi(self._kind, u, [p[f"c{j}"] for j in range(self._k)],
                [p[f"s{j}"] for j in range(self._k)], xp=jnp)
        for j in range(self._k):
            f = f + p[f"G{j}"] @ v[j]
        return {"x": f.astype(dtype)}

    def update_evaluations(self):
        return 1


def _tri() -> ct.Topology:
    """Three members of two entries: ``a`` reads ``c``; ``b`` reads ``a``;
    ``c`` reads ``a`` and ``b`` on two ports (the two fields a product
    multiplies)."""
    b = ct.TopologyBuilder()
    b.node("a", 2, alpha=0.5)
    b.node("b", 2, alpha=0.0)
    b.node("c", 2, alpha=-0.25)
    b.edge("c", "a")
    b.edge("a", "b")
    b.edge("a", "c")
    b.edge("b", "c")
    b.group("a", "b", "c")
    return b.build("tri")


#: The single-rate structures of the linear search (the centre ``c`` of a
#: nonlinearity is the input at the fixed point, which a sub-cycled
#: member reads interpolated), and ``tri``.
STRUCTURES = {**{name: linear.STRUCTURES[name]
                 for name in ("ring-2", "ring-3", "ring-5", "pair-3", "hub", "mapped")},
              "tri": _tri()}


@dataclasses.dataclass(frozen=True)
class Cell:
    """What one compiled nonlinear graph bakes in: a structure, a dtype,
    a configuration and a cap as the linear search's cells hold them, and
    the nonlinearity: one of :data:`KINDS` on every member, or ``"each"``
    -- the three in turn, by the members' order."""

    structure: str
    dtype: str
    knob: int
    cap: int
    kind: str
    mapping_kind: str = "matrix"

    @property
    def topo(self) -> ct.Topology:
        return STRUCTURES[self.structure]

    @property
    def knobs(self) -> dict:
        return linear.Cell("ring-3", self.dtype, self.knob, self.cap).knobs

    @property
    def cfgs(self) -> list:
        return ct.group_cfgs_of([self.knobs])

    def kind_of(self, name: str) -> str:
        return (KINDS[self.topo.names.index(name) % len(KINDS)] if self.kind == "each"
                else self.kind)

    @property
    def sweeps_a_product_of_two_members(self) -> bool:
        """Whether a member multiplies the deviations of two fields that
        two *other* members hold, in float32, under Gauss-Seidel.

        ``phi_j = u_j + s_j (u_j - c_j)(u_j' - c_j')`` with ``s_j = curve /
        max|c_j|``: the curve is measured against the port's own field and
        multiplies the *other* port's deviation, so where the two fields
        are in units a factor ``U`` apart the product is ``curve * U``
        over a field's own size, and the member's derivative ``1 + s_j
        (u_j' - c_j')`` moves by ``curve * U * eps`` when ``u_j'`` moves by
        one rounding.  Under Gauss-Seidel ``u_j'`` is a value this pass has
        just computed -- in float32, to a float32 rounding -- so at ``U =
        1e6`` the pass the group runs and its float64 twin have other
        Jacobians at one state, and the twin is no reference for the
        group's radius (MADD-ANO-239; the two radius scores are drawn with
        the change of units on such a cell within
        :data:`SWEPT_PRODUCT_DECADES`).
        Under Jacobi ``u_j' - c_j'`` is a difference of two floats of the
        iterate, exact in either dtype."""
        if self.dtype != "float32" or self.knobs["iteration_mode"] != "gauss-seidel":
            return False
        topo = self.topo
        for nd in topo.nodes:
            if self.kind_of(nd.name) != "product":
                continue
            read = {}
            for e in topo.edges:
                if e.dst == nd.name:
                    read.setdefault(e.port, set()).add(e.src)
            if len({frozenset(v) for v in read.values()}) > 1:
                return True
        return False


#: The rows of ``linear.KNOBS`` and the caps the rotation below takes,
#: FROZEN: the seven rows and two caps the linear search held when these
#: cells were laid out.  The rotation is over these tuples and never over
#: the length of a table of another module: a row appended to
#: ``linear.KNOBS`` (2026-10-07) silently turned 21 of the 43 cells into
#: other configurations, under pins whose comments described the old ones.
#: A new configuration is a new cell, appended (:data:`APPENDED`);
#: ``tests/property/test_coupling_search_cells_are_pinned.py`` holds every
#: cell's configuration per push.
ROTATED_KNOBS = (0, 1, 2, 3, 4, 5, 6)
ROTATED_CAPS = (5, 120)


def _cells() -> tuple:
    """Every structure with every nonlinearity at both dtypes, the
    configurations and the caps rotated as the linear search rotated them
    over :data:`ROTATED_KNOBS` and :data:`ROTATED_CAPS`."""
    out = []
    for s, name in enumerate(STRUCTURES):
        for q, kind in enumerate(KINDS):
            for t, dtype in enumerate(("float32", "float64")):
                knob = ROTATED_KNOBS[(s + 3 * t + 2 * q) % len(ROTATED_KNOBS)]
                out.append(Cell(name, dtype, knob,
                                ROTATED_CAPS[(s + t + q) % len(ROTATED_CAPS)], kind))
    return tuple(out)


#: The per-push cell: ``tri`` with a saturating gain on ``a``, a quadratic
#: term on ``b`` and the product of two fields on ``c``, in float32,
#: stopped after five Jacobi passes (a returned iterate away from the
#: fixed point, where the Jacobian is another matrix).  One cell, because
#: a cell's cost is the compile of the graph under test with its
#: diagnostics (seconds); the three nonlinearities apart, at both dtypes
#: and under every configuration, are the slow hunt's.
_FIRST = (Cell("tri", "float32", 1, 5, "each"),)
#: The 43 cells of the rotation, the per-push one first.  Their indices
#: are what the pins at the foot of the module name: never reordered.
ROTATED = _FIRST + tuple(c for c in _cells() if c not in _FIRST)
#: The linear search's eighth row, found by what it is: no acceleration,
#: Jacobi, the interface norm.
_JACOBI_INTERFACE = linear.KNOBS.index(
    dict(acceleration="none", iteration_mode="jacobi", convergence_norm="interface"))
#: Cells added on purpose, after every cell of the rotation (a new cell
#: goes at the end of this tuple).  From 2026-10-07 to the day the
#: rotation was frozen, 21 cells of the rotation were other configurations
#: by accident (:data:`ROTATED_KNOBS`); these are the ones that accident
#: visited and the rotation does not.
APPENDED = (
    # What cell 41 was by accident: ``tri`` with a product of two fields on
    # every member in float32, Aitken under Gauss-Seidel, the interface
    # norm, stopped after five passes.  The first cell on which a member
    # multiplies two other members' fields in a float32 sweep
    # (``sweeps_a_product_of_two_members``; MADD-ANO-239 is pinned on it).
    Cell("tri", "float32", 2, 5, "product"),
    # The same on the fan-out hub with no acceleration under the l2 norm
    # (what cell 29 was by accident).
    Cell("hub", "float32", 0, 5, "product"),
    # The eighth row, which the rotation over seven never takes: on ``tri``
    # at both dtypes (the per-push cell's nonlinearities, and the product),
    # and on the five structures the accident put it on.
    Cell("tri", "float32", _JACOBI_INTERFACE, 5, "each"),
    Cell("tri", "float64", _JACOBI_INTERFACE, 120, "product"),
    Cell("mapped", "float32", _JACOBI_INTERFACE, 5, "quadratic"),
    Cell("hub", "float64", _JACOBI_INTERFACE, 120, "saturating"),
    Cell("pair-3", "float32", _JACOBI_INTERFACE, 120, "product"),
    Cell("ring-5", "float64", _JACOBI_INTERFACE, 5, "quadratic"),
    Cell("ring-2", "float64", _JACOBI_INTERFACE, 120, "product"),
)
CELLS = ROTATED + APPENDED
PER_PUSH_CELLS = tuple(range(len(_FIRST)))
ALL_CELLS = tuple(range(len(CELLS)))
APPENDED_CELLS = ALL_CELLS[len(ROTATED):]
#: The slow hunt's blocks (each cell compiles a graph and a twin): the
#: cells of the rotation six ways, as they were before a cell was
#: appended (so a hunt over one draws what it drew), and the appended
#: cells.
BLOCKS = tuple(ALL_CELLS[:len(ROTATED)][k::6] for k in range(6)) + (APPENDED_CELLS,)


def build(cell: Cell, knobs: dict, dtype: str) -> ct.Built:
    """A compiled graph of *cell*'s structure with every member an
    :class:`NRelay` (``ct.build`` for these nodes: dense mappings, the
    group last)."""
    topo = cell.topo
    assert all(nd.timestep == topo.nodes[0].timestep and not nd.leaves and not nd.flux
               and not nd.beta for nd in topo.nodes), "single-rate plain relays only"
    assert len(topo.groups) == 1 and set(topo.groups[0]) == set(topo.names)
    gm = GraphManager()
    for nd in topo.nodes:
        gm.add_node(NRelay(nd.name, nd.timestep, kind=cell.kind_of(nd.name), n=nd.n, ports=nd.ports,
                           alpha=nd.alpha, dtype=dtype))
    keys = {}
    for i, e in enumerate(topo.edges):
        mapping = (matrix_mapping(np.zeros((topo.node(e.dst).n, topo.node(e.src).n),
                                           jnp.dtype(dtype))) if e.mapped else None)
        gm.add_edge(e.src, e.dst, e.field, f"u{e.port}", transform=e.transform,
                    additive=e.additive, mapping=mapping)
        if e.mapped:
            keys[i] = gm._edges[i].key                    # noqa: SLF001
    gm.add_coupling_group(list(topo.groups[0]), **cg.live_knobs(knobs))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        gm.compile()
    return ct.Built(gm, topo, str(dtype), keys, [], topo.names,
                    tuple(range(len(topo.edges))))


def params_for(built: ct.Built, values: dict) -> dict:
    """``ct.params_for`` and, where *values* holds them, every node's
    nonlinearity constants."""
    params = ct.params_for(built, values)
    for name, v in values.get("nonlinear", {}).items():
        node = params["nodes"][name]
        for j, (c, s) in enumerate(zip(v["c"], v["s"])):
            node[f"c{j}"] = jnp.asarray(c, node[f"c{j}"].dtype)
            node[f"s{j}"] = jnp.asarray(s, node[f"s{j}"].dtype)
    return params


@functools.lru_cache(maxsize=max(len(b) for b in BLOCKS) + 1)
def _built(index: int) -> tuple:
    """``(the graph under test, its x64 twin, the twin's reference)``."""
    cell = CELLS[index]
    with precision(cell.dtype == "float64"):
        built = build(cell, cell.knobs, cell.dtype)
    with cr.x64():
        twin = build(cell, cr.twin_knobs(cell.knobs), "float64")
        values = values_of(Case(index, _PROBE, 1.0))
        ct.set_initial(twin, values)
        return built, twin, cr.PassReference.of(twin.gm, params=params_for(twin, values))


@functools.lru_cache(maxsize=max(len(b) for b in BLOCKS) + 1)
def _own_twin(index: int) -> tuple:
    """``(a one-pass twin of float32 cell *index* in the cell's OWN dtype, a
    slot for its compiled tangents)``: :func:`cr.twin_knobs` without x64,
    so the pass rounds as the graph under test rounds it."""
    cell = CELLS[index]
    assert cell.dtype == "float32", cell
    with precision(False):
        return build(cell, cr.twin_knobs(cell.knobs), cell.dtype), {}


def own_sensitivities(index: int, values: dict, x: np.ndarray, ref: cr.PassReference) -> np.ndarray:
    """``dP/dc`` at iterate *x* through the pass of float32 cell *index*
    as the cell's own dtype evaluates it: the columns of
    ``ref.sensitivities``, each the tangent the group's own pass has for
    that constant (which is exactly zero where the pass, in float32, does
    not move with it)."""
    cell = CELLS[index]
    twin, compiled = _own_twin(index)
    p0, p1, count = (f"coupling_{cell.topo.group_key(0)}_pred_{s}" for s in ("0", "1", "count"))
    with precision(False):
        ct.set_initial(twin, values)
        params = params_for(twin, values)
        pre = twin.gm._state                              # noqa: SLF001
        theta, restore = ravel_pytree(ref._constants_of(params))   # noqa: SLF001
        if not compiled:
            step, ext = twin.gm._raw_step_fn, twin.gm._default_external_inputs()   # noqa: SLF001

            def moved(theta_, x_, pre_, params_):
                c = restore(theta_)
                nodes = {n: {**params_["nodes"].get(n, {}), **c["nodes"].get(n, {})}
                         for n in params_["nodes"]}
                mappings = {k: {**v, **c.get("mappings", {}).get(k, {})}
                            for k, v in params_.get("mappings", {}).items()}
                meta = {**pre_["_meta"], p0: x_, p1: x_,
                        count: jnp.asarray(2, pre_["_meta"][count].dtype)}
                after = step({**pre_, "_meta": meta}, ext,
                             {**params_, "nodes": nodes, "mappings": mappings})
                return jnp.concatenate([jnp.ravel(after[n][f])
                                        for n, f, _s, _a, _b in ref.layout])

            compiled["both"] = jax.jit(lambda *a: (moved(*a), jax.jacfwd(moved)(*a)))
        value, tangents = compiled["both"](theta, jnp.asarray(x, jnp.float32), pre, params)
    # The twin is the pass the reference differentiates, to float32
    # rounding (it reads the iterate through the same three slots).
    exact = ref.apply(x)
    for _n, _f, _shape, a, b in ref.layout:
        size = float(np.max(np.abs(exact[a:b])))
        assert np.max(np.abs(np.asarray(value, np.float64)[a:b] - exact[a:b])) <= (
            2.0 ** 10 * float(np.finfo(np.float32).eps) * size), (cell, _n)
    return np.asarray(tangents, np.float64)


@dataclasses.dataclass(frozen=True)
class Case:
    """One drawn problem on one nonlinear cell: the numbers of the linear
    search's case *base* (its ``cell`` is not read) and the curve."""

    cell: int
    base: linear.Case
    #: ``s max|c|``: the nonlinearity over a field's own size.
    curve: float

    @property
    def eps(self) -> float:
        return float(np.finfo(CELLS[self.cell].dtype).eps)


def delivered(topo: ct.Topology, values: dict, x: dict) -> dict:
    """``{node: [u_0, ...]}``: what every port reads when each source
    holds ``x[source]`` (float64)."""
    out = {nd.name: [np.zeros(nd.n) for _ in range(nd.ports)] for nd in topo.nodes}
    for i, e in enumerate(topo.edges):
        src = np.asarray(x[e.src], np.float64)
        v = np.asarray(values["H"][i], np.float64) @ src if e.mapped else src
        out[e.dst][e.port] = out[e.dst][e.port] + ct.TRANSFORM_FACTORS[e.transform] * v
    return out


def values_of(case: Case) -> dict:
    """The linear search's values for ``case.base`` on this cell, and each
    port's nonlinearity centred on its input at the linear fixed point."""
    cell = CELLS[case.cell]
    topo = cell.topo
    values = linear.values_of(case.base, cell=cell)
    model = ct.LinearModel(topo, values, dtype=cell.dtype, group_cfgs=cell.cfgs)
    pre = {m: {"x": np.asarray(values["nodes"][m]["x0"])} for m in topo.names}
    exact = model.group_fixed_point(0, pre, pre)
    values["fixed_point"] = {m: np.asarray(exact[m], np.float64) for m in topo.names}
    centres = delivered(topo, values, values["fixed_point"])
    dt = np.dtype(cell.dtype)
    values["nonlinear"] = {}
    for m in topo.names:
        c = [np.asarray(cj, dt) for cj in centres[m]]
        s = [np.asarray(case.curve / max(float(np.max(np.abs(cj))), np.finfo(dt).tiny), dt)
             for cj in c]
        values["nonlinear"][m] = {"c": c, "s": s}
    return values


def _cancellation(cell: Cell, values: dict, pre: dict, state: dict) -> float:
    """How far the worst member cancels inside itself (the linear search's
    ``_cancellation``, for an :class:`NRelay`): the magnitudes its update
    sums over its field's largest entry."""
    topo = cell.topo
    reads = delivered(topo, values, {m: state[m]["x"] for m in topo.names})
    worst = 1.0
    for nd in topo.nodes:
        v, nl = values["nodes"][nd.name], values["nonlinear"][nd.name]
        terms = (abs(nd.alpha) * np.abs(np.asarray(pre[nd.name]["x"], np.float64))
                 + np.abs(np.asarray(v["b"], np.float64)))
        through = phi(cell.kind_of(nd.name), reads[nd.name], [np.asarray(c, np.float64) for c in nl["c"]],
                      [float(s) for s in nl["s"]])
        for j in range(nd.ports):
            terms = terms + np.abs(np.asarray(v["G"][j], np.float64)) @ np.abs(through[j])
        size = float(np.max(np.abs(np.asarray(state[nd.name]["x"], np.float64))))
        if size > 0:
            worst = max(worst, float(np.max(terms)) / size)
    return worst


def run_once(built: ct.Built, values: dict) -> ct.Step:
    """One step from the drawn start (``ct.run`` with this module's constants)."""
    ct.set_initial(built, values)
    params = params_for(built, values)
    gm = built.gm
    pre = ct._snapshot(gm, {})                            # noqa: SLF001
    gm.step(params=params)
    key = built.topo.group_key(0)
    return ct.Step(pre, ct._snapshot(gm, {}), {0: dict(gm.coupling_diagnostics()[key])},   # noqa: SLF001
                   {0: cg.group_meta(gm, key)})


def does_not_move_the_fixed_point(constant: str) -> bool:
    """Whether *constant* (``node.leaf[i]``, as the reference names it) is
    a nonlinearity's centre ``c<j>`` or curve ``s<j>``: at the fixed point
    ``u = c`` and ``phi`` depends on neither, so the fixed point's
    derivative with respect to it is zero (to the rounding of ``c``)."""
    leaf = constant.split(".", 1)[1]
    return leaf[0] in "cs" and leaf[1].isdigit()


@functools.lru_cache(maxsize=4096)
def observe(case: Case) -> dict:
    """One step of *case* and the scores of what it reported."""
    cell = CELLS[case.cell]
    topo = cell.topo
    values = values_of(case)
    built, twin, ref = _built(case.cell)
    ref = bound_reference(ref, values, twin)
    with precision(cell.dtype == "float64"):
        step = run_once(built, values)
        d = dict(step.reports[0])
        floor = linear._reported_floor(built.gm, topo.group_key(0), step.metas[0], d)   # noqa: SLF001
    out = dict(bound=0.0, radius=0.0, radius_strict=0.0, gradient=0.0, gradient_vanishing=0.0,
               floor=0.0, vanishing_scored=False, spectral_usable=bool(d["spectral_usable"]),
               gradient_usable=bool(d["gradient_bound_usable"]),
               floor_reported=math.isfinite(floor), referenced=False, near=False, stepped=True,
               finite=False,
               report={k: d[k] for k in ("iterations", "converged", "residual", "rho_spectral",
                                         "spectral_error_bound", "spectral_usable",
                                         "gradient_relative_error_bound",
                                         "gradient_bound_usable", "precision_limited")})
    finite = all(np.all(np.isfinite(s["x"])) for s in step.state.values())
    if not finite or not math.isfinite(d["residual"]):
        return out
    out["finite"] = True
    x = ref.flat(step.state)
    fixed = ref.fixed_point(x)
    out["report"].update(reference_ulps=fixed.ulps, reference_steps=(fixed.picard, fixed.newton))
    if not fixed.converged:
        return out          # no reference: nothing is scored (the fraction is held to a floor)
    out["referenced"] = True
    exact = ref.flat({m: {"x": values["fixed_point"][m]} for m in topo.names})
    out["report"]["fixed_point_vs_linear"] = float(
        np.max(np.abs(fixed.x - exact)) / max(float(np.max(np.abs(exact))), 1e-300))
    norm, raw = norms_of(cell.cfgs[0], topo, values, ref)
    residual = float(d["residual"])
    cancels = _cancellation(cell, values, step.pre, step.state)
    allowed = ((residual + cancels * floor) / (residual + floor)
               if out["floor_reported"] and residual + floor > 0 else 1.0)
    out["report"].update(floor=floor, cancellation=cancels)

    if out["floor_reported"]:
        true = ref.residual(x, norm)
        above = true - residual * (1.0 + 2.0 ** 8 * case.eps)
        out["floor"] = max(0.0, above) / max(cancels * floor, 1e-300)
        out["report"]["residual_true"] = true

    # CPL-087: the Jacobian at the returned state.
    J = ref.jacobian(x)
    out["report"].update(rho_true=cr.radius(J), rho_at_fixed_point=cr.radius(
        ref.jacobian(fixed.x)))
    linear.radius_scores(out, J, state_weights(ref, x), float(d["rho_spectral"]), case.eps,
                         case.base.seed)

    if out["spectral_usable"]:
        dist = ref.distance(x, fixed, norm)
        h = ref.nonlinearity(x, fixed, norm)
        bound = float(d["spectral_error_bound"]) * allowed
        out["report"].update(distance=dist, nonlinearity=h,
                             distance_over_bound=dist / bound if bound > 0 else math.inf)
        out["near"] = h < 1.0
        if out["near"]:
            # The bound of the linearisation at the returned iterate is
            # short of the distance by at most ``1 / (1 - h)``.
            reach = bound / (1.0 - h)
            out["bound"] = (math.inf if math.isnan(bound) else
                            dist / reach if reach > 0 else (math.inf if dist > 0 else 0.0))

    if out["gradient_usable"]:
        bound = (float(d["gradient_relative_error_bound"]) * allowed + 64.0 * case.eps
                 + ref.gradient_resolution(x, raw))
        names = ref.constant_names()
        # CPL-093 is made for a constant the pass resolves: moving it by
        # its own magnitude moves one pass by more than the residual's
        # float floor.  Held here to ``RESOLVED_MARGIN`` floors (the two
        # sides are two computations of one comparison).
        response = ref.pass_responses(x, raw)
        resolved = [not out["floor_reported"] or r > RESOLVED_MARGIN * floor for r in response]
        own_too = [does_not_move_the_fixed_point(n) and r for n, r in zip(names, resolved)]
        if cell.dtype == "float32" and out["floor_reported"] and any(own_too):
            # CPL-093 is "a bound on the relative error of the IFT gradient
            # ... |g_k - g*| <= bound * |g_k|", made for "a constant the
            # pass resolves: moving it by its own magnitude moves one pass
            # from x_k by more than the residual's float floor ... and its
            # tangent is not exactly zero".  The pass, the tangent and
            # g_k are the group's own, in float32.  The tangent of a
            # nonlinearity's centre or curve is a difference that cancels
            # at the centre, and where that difference is a float32
            # rounding of a value the pass has just computed the float64
            # twin holds another number: two floors where the float32
            # pass's is exactly zero (``_NO_TANGENT_IN_FLOAT32``: not in
            # the bound, "which says nothing of it"), or 2.4 floors beside
            # 695, where the error is 6.7 of the twin's g_k and 0.97 of the
            # group's own under a bound of 1.93.  A centre or a curve is
            # scored where the float32 pass resolves it too and the twin's
            # response is the float32 pass's within ``RESOLVED_MARGIN``:
            # where the twin's g_k is the gradient the bound is about.
            own = ref.pass_responses(x, raw, sensitivities=own_sensitivities(
                case.cell, values, x, ref))

            def the_twin_s(r, o) -> bool:
                return (o > RESOLVED_MARGIN * floor and o <= RESOLVED_MARGIN * r
                        and r <= RESOLVED_MARGIN * o)

            out["report"]["unresolved_by_the_float32_pass"] = {
                n: (float(r / floor), float(o / floor))
                for n, mine, r, o in zip(names, own_too, response, own)
                if mine and not the_twin_s(r, o)}
            resolved = [r_ and not (mine and not the_twin_s(r, o))
                        for r_, mine, r, o in zip(resolved, own_too, response, own)]
        out["report"]["constants_resolved"] = (sum(resolved), len(resolved))
        for score, vanishing in (("gradient", False), ("gradient_vanishing", True)):
            mine = [does_not_move_the_fixed_point(n) is vanishing for n in names]
            out["report"][f"{score}_constants"] = (
                sum(m and r for m, r in zip(mine, resolved)), sum(mine))
            true, column = ref.gradient_error(x, fixed, raw, columns=[
                m and r for m, r in zip(mine, resolved)])
            if vanishing:
                out["vanishing_scored"] = column is not None
            out[score] = math.inf if math.isnan(bound) else true / bound
            out["report"].update({f"{score}_error": true, f"{score}_constant": (
                None if column is None else ref.constant_names()[column])})
    return out


# ---------------------------------------------------------------------------
# The strategy and the searches
# ---------------------------------------------------------------------------

#: The curves drawn, in decades: a hundredth (all but linear over a
#: field's own size) to a hundred (a returned iterate a thousandth of its
#: field from the fixed point still sees the Jacobian move by a tenth).
CURVES = (-2.0, 2.0)

THRESHOLD = {**linear.THRESHOLD, "gradient_vanishing": linear.THRESHOLD["gradient"]}
FLAG = {**linear.FLAG, "gradient_vanishing": "gradient_usable"}
#: The linear search's four and the gradient bound over a nonlinearity's
#: own constants (its centre and its curve), which the fixed point does
#: not respond to: scored wherever the pass resolves one.
SEARCHES = linear.SEARCHES + ("gradient_vanishing",)
#: The least fraction of a hunt's examples with the flag set (measured:
#: 0.64 to 1.00 by block and score; the linear search holds a half, and a
#: hunt here climbs towards starts that diverge), and the least fraction of
#: the examples that returned a finite state for which the reference found
#: a fixed point (measured: 0.93 to 1.00; a capped solve from a start a
#: whole field away can return where Newton reaches none).
USABLE_FLOOR = 0.25
#: A constant is scored for the gradient bound where its pass response
#: (``PassReference.pass_responses``) is above this many floors: the
#: bound drops a probe at one floor, by its own float32 or float64
#: arithmetic, and the reference measures the response in float64.
RESOLVED_MARGIN = 2.0
REFERENCED_FLOOR = 0.75


#: The largest change of units, in decades, the two radius scores are
#: drawn at on a cell whose pass multiplies two members' fields in a
#: float32 sweep (:attr:`Cell.sweeps_a_product_of_two_members`).  Within a
#: decade the product is at most a thousand over a field's own size, and
#: one float32 rounding moves the Jacobian by 1e-4; at the six decades the
#: linear search draws, by more than the whole of it (MADD-ANO-239, pinned
#: at the foot of the module on what the search read there).  The other
#: scores are drawn at every change of units on those cells too.
SWEPT_PRODUCT_DECADES = 1
#: The scores of CPL-087, which MADD-ANO-239 is about.
RADIUS_SCORES = ("radius", "radius_strict")


def within_the_units_a_swept_product_is_claimed_for(case: "Case") -> "Case":
    """*case* with its change of units within :data:`SWEPT_PRODUCT_DECADES`
    where its cell sweeps a product of two members' fields; any other
    case unchanged."""
    unit = case.base.unit
    if abs(unit) <= SWEPT_PRODUCT_DECADES or not CELLS[case.cell].sweeps_a_product_of_two_members:
        return case
    return dataclasses.replace(case, base=dataclasses.replace(
        case.base, unit=int(math.copysign(SWEPT_PRODUCT_DECADES, unit))))


def cases(cells=ALL_CELLS, domain: linear.Domain = linear.CLAIMED, curves=CURVES, *,
          swept_units_held: bool = False):
    """Draw a :class:`Case` on one of *cells*: the linear search's numbers
    within *domain* and a curve.  *swept_units_held*: with the change of
    units held within :data:`SWEPT_PRODUCT_DECADES` on a cell that sweeps
    a product of two members' fields (the same draws otherwise)."""
    drawn = st.builds(Case, cell=st.sampled_from(tuple(cells)), base=linear.cases((0,), domain),
                      curve=st.floats(*curves).map(lambda x: 10.0 ** x))
    return (drawn.map(within_the_units_a_swept_product_is_claimed_for) if swept_units_held
            else drawn)


def search(name: str, *, cells=PER_PUSH_CELLS, domain: linear.Domain = linear.CLAIMED,
           profile=None, fail: bool = True):
    """Run search *name*; returns ``(report, {fraction name: value})``.

    The default profile is the per-push one, the same draws for every
    score (an example is a step and a Newton solve; the four searches
    share what the random phase draws)."""
    profile = PER_PUSH if profile is None else profile
    drawn = []

    def score(case: Case):
        drawn.append(case)
        seen = observe(case)
        return seen[name], seen["report"]

    report = targeted_search(cases(cells, domain, swept_units_held=name in RADIUS_SCORES), score,
                             THRESHOLD[name], profile=profile, label=name, fail=fail)
    seen = [observe(c) for c in drawn]
    count = max(len(seen), 1)
    return report, dict(usable=sum(s[FLAG[name]] for s in seen) / count,
                        referenced=sum(s["referenced"] for s in seen) / count,
                        near=sum(s["near"] for s in seen) / count,
                        stepped=sum(s["stepped"] for s in seen) / count,
                        vanishing_scored=sum(s["vanishing_scored"] for s in seen) / count,
                        finite=sum(s["finite"] for s in seen) / count)


# ---------------------------------------------------------------------------
# The reference, validated
# ---------------------------------------------------------------------------


def _per_push_draws(cells) -> list:
    """The cases the linear search's four per-push searches draw, on *cells*."""
    drawn = []
    for name in linear.SEARCHES:
        def score(case, name=name):
            drawn.append(case)
            return linear.scorer(name)(case)
        targeted_search(linear.cases(cells, linear.CLAIMED), score, math.inf,
                        profile=linear.EVERY_PUSH.seeded(sorted(linear.THRESHOLD).index(name)),
                        label=name)
    return list(dict.fromkeys(drawn))


def assert_the_reference_reproduces_the_closed_form(drawn) -> dict:
    """Every answer of the reference within :data:`ALLOWED` of the closed
    form on each of *drawn*; the worst miss per answer."""
    worst, compared = {}, 0
    for case in drawn:
        seen = against_the_closed_form(case)
        misses = closed_form_misses(seen)
        compared += bool(misses)
        over = {k: v for k, v in misses.items() if v > 1.0}
        assert not over, f"{case}: {over} ({seen})"
        for k, v in misses.items():
            worst[k] = max(worst.get(k, 0.0), v)
    assert compared >= 0.9 * len(drawn), (compared, len(drawn))
    return worst


def test_the_reference_reproduces_the_closed_form_on_a_linear_cell():
    """Per push: the ring of three float64 scalars under Jacobi and the
    seed shapes on it.  Slow sibling:
    :func:`test_the_reference_reproduces_the_closed_form_on_every_per_push_draw`."""
    drawn = []
    # Forty derandomised draws of the linear search's generator (not
    # scored here: the slow sibling takes the search's own draws).
    targeted_search(linear.cases((0,), linear.CLAIMED), lambda c: (drawn.append(c) or 0.0, None),
                    math.inf, profile=dataclasses.replace(PER_PUSH, max_examples=40))
    drawn += [c for c in linear.SEEDS.values() if c.cell == 0]
    assert len(set(drawn)) >= 20, len(set(drawn))
    assert_the_reference_reproduces_the_closed_form(drawn)


# Slow: a twin compiled for each per-push cell of the linear search and for
# its multi-rate cell, and every one of its per-push draws (about 600).
# Per push: tests/property/test_coupling_nonlinear_search.py::test_the_reference_reproduces_the_closed_form_on_a_linear_cell
@pytest.mark.slow
def test_the_reference_reproduces_the_closed_form_on_every_per_push_draw():
    """Every per-push draw of the linear search (the five per-push cells:
    both dtypes, both schedules, the l2 and interface norms, a mapped
    ring, twelve scalars), the seed shapes, and the same number of draws
    on the multi-rate cell (a sub-cycled member, linear interpolation)."""
    drawn = _per_push_draws(linear.PER_PUSH_CELLS) + list(linear.SEEDS.values())
    drawn += _per_push_draws((_LINEAR_MULTIRATE,))[:150]
    worst = assert_the_reference_reproduces_the_closed_form(drawn)
    print(f"{len(drawn)} draws; the worst miss over what is allowed: "
          + ", ".join(f"{k} {v:.3g}" for k, v in sorted(worst.items())))


# Slow: a sparse cell of the linear search and its twin compiled.
# Per push: tests/property/test_coupling_nonlinear_search.py::test_the_reference_reproduces_the_closed_form_on_a_linear_cell
@pytest.mark.slow
def test_the_reference_reproduces_the_closed_form_on_a_sparse_cell():
    """The two oracles name a sparse edge's constants by different routes
    and must take the same ones: the reference differentiates the twin's
    own parameter tree, which holds the weights of the pattern and nothing
    else, and the closed form asks the mapping kind which entries of its
    dense matrix are weights.  On the local sparse cell, stopped after five
    passes, every answer agrees, the gradient error among them; taken over
    every entry of the matrix the closed form's is five times the
    reference's on ``linear.OUTSIDE_THE_PATTERN``.

    (A local pattern, because the relative error of the derivative with
    respect to a weight is that of the source entry it reads, whatever its
    row: the entries outside a pattern change the worst only where a
    column holds none, and a ragged pattern has a full row.)"""
    index = linear.OUTSIDE_THE_PATTERN.cell
    cell = linear.CELLS[index]
    kind = cell.mapping_kind
    assert kind == "sparse-local" and cell.cap == 5
    drawn = [linear.OUTSIDE_THE_PATTERN]
    targeted_search(linear.cases((index,), linear.CLAIMED),
                    lambda c: (drawn.append(c) or 0.0, None), math.inf,
                    profile=dataclasses.replace(PER_PUSH, max_examples=40))
    worst = assert_the_reference_reproduces_the_closed_form(drawn)
    # The premise: the pass of the twin (bound by the comparison) moves
    # with one mapping weight per pattern entry, the patterns hold fewer
    # entries than the matrices have, and a column of one holds none.
    _twin, ref = _linear_twin(index)
    patterns = [ct.mapping_pattern(cell.topo, i, kind)
                for i, e in enumerate(cell.topo.edges) if e.mapped]
    held, entries = sum(int(p.sum()) for p in patterns), sum(p.size for p in patterns)
    moves = np.any(ref.sensitivities(np.ones(ref.size)) != 0.0, axis=0)
    weights = sum(bool(m) for name, m in zip(ref.constant_names(), moves)
                  if name.startswith("mapping:"))
    assert weights == held < entries, (weights, held, entries)
    assert any(not p.any(axis=0).all() for p in patterns)
    print(f"{kind}, cell {index}: {len(drawn)} draws; the worst miss over what is allowed: "
          + ", ".join(f"{k} {v:.3g}" for k, v in sorted(worst.items())))


# Slow (the marked cells): a twin and an iterating twin compiled per cell.
# Per push: tests/property/test_coupling_nonlinear_search.py::test_several_passes_of_an_iterating_twin_are_compositions_of_the_single_pass
@pytest.mark.parametrize("index", [0, pytest.param(2, marks=pytest.mark.slow),
                                   pytest.param(_LINEAR_MULTIRATE, marks=pytest.mark.slow)])
def test_several_passes_of_an_iterating_twin_are_compositions_of_the_single_pass(index):
    """The single-pass branch of the step (``max_iterations=1``) runs the
    pass an iterating group iterates: three passes of a twin that iterates
    are three compositions of the reference's map, to float64 rounding.
    (The ring under Jacobi per push; the mapped ring under Gauss-Seidel
    and the multi-rate ring in the slow lane.)"""
    cell = linear.CELLS[index]
    values = linear.values_of(dataclasses.replace(_PROBE, cell=index))
    built_twin, ref = _linear_twin(index)
    ref = bound_reference(ref, values, built_twin)
    knobs = {**cr.twin_knobs(cell.knobs), "max_iterations": 3, "predictor": "none",
             "tolerance": 0.0}
    with cr.x64():
        several = ct.build(cell.topo, knobs, dtype="float64", mapping_kind=cell.mapping_kind)
        ct.set_initial(several, values)
        start = ref.flat({m: {"x": values["nodes"][m]["x0"]} for m in cell.topo.groups[0]})
        assert cr.passes_compose(ref, several.gm, 3, start) <= 2.0 ** 6
    # The premise: the pass depends on the iterate, apart from the pre-step state.
    assert np.any(ref.apply(start) != ref.apply(start + 1.0))


def test_the_reference_refuses_a_twin_it_cannot_read():
    """One group, one pass, the linear predictor: anything else is refused
    when the reference is built, not answered wrongly."""
    cell = linear.CELLS[0]
    with cr.x64():
        iterating = ct.build(cell.topo, {"max_iterations": 3}, dtype="float64")
        with pytest.raises(AssertionError, match="one pass under the linear predictor"):
            cr.PassReference.of(iterating.gm)
        fresh = ct.build(cell.topo, cr.twin_knobs(cell.knobs), dtype="float64")
        # A fresh graph: every field zero after its step, so the slot does
        # not say which member is where.
        with pytest.raises(AssertionError, match="does not determine the layout"):
            cr.PassReference.of(fresh.gm)


# ---------------------------------------------------------------------------
# The searches on the nonlinear cells
# ---------------------------------------------------------------------------


def test_the_nonlinear_fixed_point_is_the_linear_one_and_the_jacobian_moves():
    """The premise of the cells, on the per-push one: the reference's fixed
    point is the linear closed form's (the nonlinearity is centred on
    it), and away from it the Jacobian is another matrix -- the part no
    closed form here knows."""
    case = Case(0, linear.Case(0, 7, 0.6, True, 1.0, 0.0, 1.0, 0, 1e-1, 0), 30.0)
    seen = observe(case)
    assert seen["stepped"] and seen["referenced"], seen
    report = seen["report"]
    assert report["fixed_point_vs_linear"] <= 2.0 ** 10 * case.eps ** 2 + 2.0 ** 10 * EPS64, report
    assert abs(report["rho_at_fixed_point"] - 0.6) <= 1e-5, report
    assert abs(report["rho_true"] - report["rho_at_fixed_point"]) >= 1e-3, report
    assert report["nonlinearity"] >= 1e-3, report


#: Examples on the per-push cell on which a number is close to its limit
#: (found by a 500-example search of that cell), so that the per-push lane
#: sees a bound that became too small without drawing for it:
#: ``{name: (case, score, the least the score must still be)}``.
SEEDS = {
    # Five Jacobi passes from a whole field away: the distance is 0.88 of
    # ``spectral_error_bound``.
    "the-error-bound-an-eighth-above-the-distance": (
        Case(0, linear.Case(0, 65535, 0.05, True, 1.0, 0.0, 1.0, -3, 1.0, 2), 1.0),
        "bound", 0.75),
    # A curve of 21 over a field: the radius is 0.399 at the returned
    # iterate and 0.333 at the fixed point (``h`` = 0.14), and the
    # derivative with respect to a gain is 24% off, a fourteenth of
    # ``gradient_relative_error_bound``.
    "the-jacobian-a-fifth-from-the-fixed-point-s": (
        Case(0, linear.Case(0, 1, 1.0 / 3.0, False, 1.0, 0.0, 1.0, -1, 1.0, 0),
             21.544346900318832), "gradient", 0.05),
    # Five Jacobi passes from a whole field away: the pass resolves six of
    # the twelve centres and curves, the relative error of the gradient
    # with respect to one is of order one (the fixed point does not
    # respond to it), and the bound reads 1.27, the error 0.79 of it.
    "a-nonlinearity-s-own-constant-resolved-short-of-the-fixed-point": (
        Case(0, linear.Case(0, 1, 0.3, False, 1.0, 0.0, 1.0, 0, 1.0, 0), 1.0),
        "gradient_vanishing", 0.5),
}


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_every_score_holds_on_the_nonlinear_seed_shapes(seed):
    case, score, least = SEEDS[seed]
    seen = observe(case)
    assert seen["stepped"] and seen["referenced"] and seen[FLAG[score]], seen
    over = {name: seen[name] for name in SEARCHES + ("radius_strict",)
            if seen[name] > THRESHOLD[name]}
    assert not over, f"{seed}: {over} ({seen['report']})"
    assert seen[score] >= least, (
        f"premise: {score} is {seen[score]!r} on {seed}, no longer near its limit "
        f"({seen['report']})")


def _held(name: str, fractions: dict) -> None:
    assert fractions["referenced"] >= REFERENCED_FLOOR * fractions["finite"], (name, fractions)
    assert fractions["usable"] > 0, f"{name}: no example had its flag set ({fractions})"


def test_a_usable_error_bound_reaches_the_distance_on_the_nonlinear_cell_per_push():
    _report, fractions = search("bound")
    _held("bound", fractions)
    assert fractions["near"] > 0, fractions


def test_a_settled_spectral_radius_is_the_radius_at_the_returned_iterate_per_push():
    _report, fractions = search("radius")
    _held("radius", fractions)


def test_a_usable_gradient_bound_is_never_below_the_error_on_the_nonlinear_cell_per_push():
    _report, fractions = search("gradient")
    _held("gradient", fractions)


def test_a_usable_gradient_bound_covers_a_nonlinearity_s_own_constants_per_push():
    """A nonlinearity's centre and curve do not move the fixed point, so
    the relative error of the gradient with respect to one is of order one
    at any other iterate: the bound reads above it wherever the pass
    resolves the constant, and leaves the constant out where it does not.
    (This cell's twenty examples end within a float32 rounding of their
    fixed points, where none is resolved; the seed
    ``a-nonlinearity-s-own-constant-resolved-short-of-the-fixed-point``
    holds one that is, and the slow hunt holds that some are.)"""
    _report, fractions = search("gradient_vanishing")
    _held("gradient_vanishing", fractions)


def test_the_floor_covers_what_the_reported_residual_misses_on_the_nonlinear_cell_per_push():
    _report, fractions = search("floor")
    _held("floor", fractions)


# Slow: 115 random examples a search on each of six blocks of seven cells
# and on the block of the appended cells, each cell a compile of the graph
# with its diagnostics and of its twin; the blocks outermost, so the
# searches share a block's compiled graphs.
# Per push: tests/property/test_coupling_nonlinear_search.py::test_a_usable_error_bound_reaches_the_distance_on_the_nonlinear_cell_per_push
@pytest.mark.slow
@pytest.mark.parametrize("block,name", [(b, n) for b in range(len(BLOCKS))
                                        for n in SEARCHES + ("radius_strict",)])
def test_the_hunt_finds_no_number_on_the_wrong_side_of_a_nonlinear_group(block, name):
    """Not shrunk: the example that fails is reported as drawn (a shrink
    here is thousands of steps)."""
    profile = dataclasses.replace(SLOW, max_examples=115).seeded(1000 + block, shrink=False)
    report, fractions = search(name, cells=BLOCKS[block], profile=profile)
    print(f"{name}, block {block}: worst {report}; {fractions}")
    assert fractions["referenced"] >= REFERENCED_FLOOR * fractions["finite"], fractions
    assert fractions["usable"] >= USABLE_FLOOR, (
        f"{name}, block {block}: only {fractions['usable']:.2f} of the examples had the flag "
        f"set (floor {USABLE_FLOOR})")
    # No floor on fractions["vanishing_scored"] here.  Whether the pass
    # resolves a nonlinearity's own constant depends on where the float32
    # iterate stops relative to its fixed point, and that is a matter of
    # rounding: one block reads 0.17 to 0.57 with jaxlib 0.11.0 on the
    # development machine and 0.0 on CI's runners (same seed, jaxlib 0.10.2
    # and 0.11.2).  That the score is not empty is held per push, by the
    # seed ``a-nonlinearity-s-own-constant-resolved-short-of-the-fixed-point``
    # (test_every_score_holds_on_the_nonlinear_seed_shapes), which CI runs.


def test_the_reference_s_jacobian_is_the_central_difference_of_its_pass():
    """The dense Jacobian against a central difference of the pass map, on
    the per-push nonlinear cell away from its fixed point, and the same
    check on a reading (the interface norm's, written in JAX): the
    self-check a reading's Jacobian-vector product can be held to."""
    case = Case(0, linear.Case(0, 7, 0.6, True, 1.0, 0.0, 1.0, 0, 1e-1, 0), 30.0)
    cell = CELLS[0]
    values = values_of(case)
    _built_graph, twin, ref = _built(0)
    ref = bound_reference(ref, values, twin)
    x = ref.flat({m: {"x": values["nodes"][m]["x0"]} for m in cell.topo.names})
    assert ref.finite_difference_gap(x) <= 1e-7
    H = {i: jnp.asarray(values["H"][i], jnp.float64) for i in values["H"]}

    def reading(z):
        out = []
        for i, e in enumerate(cell.topo.edges):
            src = ref.field(z, e.src, "x")
            out.append(ct.TRANSFORM_FACTORS[e.transform] * (H[i] @ src if e.mapped else src))
        return out

    assert ref.finite_difference_gap(x, reading) <= 1e-7
    # A reading whose product drops a term (one the tangent does not see)
    # is caught by the same check.

    def dropped(z):
        return [f + jnp.sum(jax.lax.stop_gradient(z) ** 2) for f in reading(z)]

    assert ref.finite_difference_gap(x, dropped) >= 1e-3


# ---------------------------------------------------------------------------
# What the search found
# ---------------------------------------------------------------------------

#: The examples the hunt reached on the tree before the gradient bound
#: left out a probe the pass does not resolve, as drawn (the hunt does not
#: shrink).  On each the nonlinearity's own constants (a centre ``c``, a
#: curve ``s``) are below the pass's float floor: the fixed point does not
#: respond to one, and the returned iterate, a rounding away, by a
#: rounding.  ``(case, the bound it read, the relative error it stood
#: beside)``.
UNRESOLVED = {
    # A two-member float32 ring with saturating gains, started on its
    # fixed point: one pass, residual 0, precision_limited.  The float32
    # tangents of the centres and curves are exactly zero (never in the
    # bound); in float64 at the returned iterate the curve's is 6e-26
    # beside gradients of order one, a relative error of 2.7.
    "a-ring-stalled-on-its-start": (
        Case(1, linear.Case(0, 0, 0.05, False, 1.0, 0.0, 1.0, 0, 0.0, 0), 1.0), 1.4e-6, 2.7),
    # A float64 hub converged in four passes from a start a whole field
    # away: right-hand sides of 2e-33 to 5e-18 for the centres and curves
    # beside ones of order one.  The bound read 1.09 (their probes') for a
    # relative error of 1.134 on one of them; without them it is the
    # gains', 5.3e-12.
    "a-converged-hub": (
        Case(26, linear.Case(0, 2619, 0.05, False, 1.0, 0.0, 1.0, 3, 1.0, 3), 10.0), 1.09, 1.134),
}


# Slow: each example compiles its cell (a graph with diagnostics and a twin).
# Per push: tests/property/test_coupling_nonlinear_search.py::test_a_usable_gradient_bound_covers_a_nonlinearity_s_own_constants_per_push
@pytest.mark.slow
@pytest.mark.parametrize("name", list(UNRESOLVED))
def test_a_constant_the_pass_does_not_resolve_is_not_in_the_gradient_bound(name):
    """CPL-093 is a relative bound for a constant the pass resolves.  On
    these examples none of the nonlinearity's own constants is, and the
    usable bound is the other constants': far below one, and not below
    their error."""
    case, _read, _beside = UNRESOLVED[name]
    seen = observe(case)
    report = seen["report"]
    assert seen["stepped"] and seen["referenced"] and seen["gradient_usable"], seen
    resolved, total = report["gradient_vanishing_constants"]
    assert total > 0 and resolved == 0, (
        f"premise: {resolved} of the {total} constants of a nonlinearity are resolved: {report}")
    scored, others = report["gradient_constants"]
    assert scored == others > 0, report
    assert seen["gradient"] <= THRESHOLD["gradient"], report
    # units: a relative error; the hub read 1.09 with those probes in it.
    assert report["gradient_relative_error_bound"] < 1e-5, report


#: The example the hunt stopped on: the fan-out hub with products of two
#: fields in float64, Jacobi under IQN-ILS, cap 120, non-normal gains,
#: started a whole field from its fixed point.  Plain passes from that
#: start leave float64 range on the eighth.
_NEVER_RETURNS = Case(30, linear.Case(0, 3397, 0.6568215254870227, True, 1.0, 0.0, 1.0, 6, 1.0, 6),
                      1.0)
_STEP_IN_A_SUBPROCESS = """
import sys
from tests.property import test_coupling_nonlinear_search as nl
from tests.property.sysid_transform_grid import precision
case = nl._NEVER_RETURNS
cell = nl.CELLS[case.cell]
from tests.property import coupled_graphs as cg
with precision(True):
    built = nl.build(cell, cg.live_knobs({**cell.knobs, "acceleration": sys.argv[1]}), cell.dtype)
    values = nl.values_of(case)
    nl.run_once(built, nl.values_of(nl.Case(case.cell, nl._PROBE, 1.0)))   # compiled, and returns
    print("COMPILED", flush=True)
    step = nl.run_once(built, values)
    print("RETURNED", int(step.reports[0]["iterations"]), bool(step.reports[0]["converged"]),
          flush=True)
"""


def _steps_in_a_subprocess(acceleration: str, seconds: float) -> str:
    """The output of one step of :data:`_NEVER_RETURNS` under
    *acceleration*, in a process of its own given *seconds* after its
    compile (a call that does not return cannot be scored in this one)."""
    import pathlib  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    root = pathlib.Path(__file__).resolve().parents[2]
    proc = subprocess.Popen([sys.executable, "-c", _STEP_IN_A_SUBPROCESS, acceleration],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                            cwd=root, env={**os.environ, "PYTHONPATH": os.pathsep.join(
                                [str(root / "src"), str(root),
                                 os.environ.get("PYTHONPATH", "")])})
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "COMPILED", "the benign step did not return"
        try:
            out, _err = proc.communicate(timeout=seconds)
        except subprocess.TimeoutExpired:
            return "TIMEOUT"
        return out.strip()
    finally:
        proc.kill()
        proc.wait()


# Slow: a subprocess that compiles the cell and is then given 30 s for a
# step that takes 0.05 s where it returns.
# Per push: tests/property/test_coupling_nonlinear_search.py::test_a_usable_error_bound_reaches_the_distance_on_the_nonlinear_cell_per_push
@pytest.mark.slow
def test_a_group_that_leaves_float_range_without_a_quasi_newton_acceleration_returns():
    """The control of the pin below: the same graph and start with no
    acceleration returns from its step at the cap, not converged."""
    assert _steps_in_a_subprocess("none", 30.0) == "RETURNED 120 False"


# Slow: as the control above.
# Per push: tests/property/test_coupling_nonlinear_search.py::test_a_usable_error_bound_reaches_the_distance_on_the_nonlinear_cell_per_push
@pytest.mark.slow
@pytest.mark.parametrize("acceleration", ["iqn-ils", "iqn-imvj"])
def test_a_group_that_leaves_float_range_under_iqn_still_returns(acceleration):
    """``converged`` False "means ... the group hit max_iterations"
    (CPL-053) and ``residual`` is "inf on a non-finite state" (CPL-084):
    both say the step returns, and it runs to its cap as the unaccelerated
    group does.  It did not return (no return in 240 s): the secant
    least-squares handed LAPACK's SVD a matrix holding an ``inf``."""
    assert _steps_in_a_subprocess(acceleration, 30.0) == "RETURNED 120 False"


# ---------------------------------------------------------------------------
# What the search read on the cells an accident visited
# ---------------------------------------------------------------------------

#: ``tri`` with a product of two fields on every member in float32, Aitken
#: under Gauss-Seidel, the interface norm, five passes: the cell the three
#: examples below were drawn on (found by what it is, not by its index).
_SWEPT_PRODUCT = CELLS.index(Cell("tri", "float32", 2, 5, "product"))
#: Their numbers: non-normal gains at a loop gain of a third, member ``a``
#: in units of 1e6, started ON the fixed point (one pass, a float32
#: rounding from it, ``precision_limited``).
_ON_THE_FIXED_POINT_IN_OTHER_UNITS = dict(rho=1.0 / 3.0, nonnormal=True, small=1.0, spread=0.0,
                                          cancel=1.0, unit=6, offset=0.0, member=5)


def _swept(seed: int, curve: float) -> Case:
    return Case(_SWEPT_PRODUCT, linear.Case(0, seed, **_ON_THE_FIXED_POINT_IN_OTHER_UNITS), curve)


class SettledOnAJacobianFloat32DoesNotDetermine(AssertionError):
    """MADD-ANO-239: ``rho_spectral`` with ``spectral_usable`` set, outside
    what CPL-087 says of it, where one float32 rounding of a value the
    pass has just computed moves the pass's Jacobian by more."""


#: MADD-ANO-239, as drawn by the hunt of 2026-10-07 on this cell.  Member
#: ``c`` reads ``a`` (about 1e6, a float32 rounding 0.125) and ``b`` (about
#: 1) and its derivative with respect to ``b`` is ``1 + curve (a' - c_0) /
#: max|c_1|``: ``a'`` is the value this Gauss-Seidel pass has just computed
#: for ``a``, and ``a' - c_0`` is whatever rounding the float32 pass left in
#: it.  ``rho_spectral`` is, to six digits, the radius of the Jacobian at
#: the same-pass values the float32 pass forms; the float64 reference's is
#: at the exact ones.  ``{name: (case, score)}``.
UNDETERMINED = {
    # A curve of 19: 0.324 for 0.0238, 8.9 margins of the flag (the radius
    # is 0.048 to 0.59 over the float32 values within two roundings of the
    # exact same-pass ones).
    "a-radius-a-rounding-of-a-same-pass-value-moves-past-the-flag-s-margin": (
        _swept(65535, 19.05131244763185), "radius_strict"),
    # A curve of 0.19: 0.11014 for 0.11156, inside the flag's margin and
    # 1.3% off where CPL-087 says 1e-4 and the movement under a rounding
    # of the Jacobian (0.110 to 0.113 over the sixteen roundings).
    "a-radius-a-rounding-of-a-same-pass-value-moves-by-a-hundredth": (
        _swept(65536, 0.19051312447631852), "radius"),
}


@pytest.mark.parametrize("case,score", [pytest.param(*row, marks=pytest.mark.xfail(
    strict=True, raises=SettledOnAJacobianFloat32DoesNotDetermine,
    reason="MADD-ANO-239: spectral_usable does not see a Jacobian that one float32 rounding of "
           "a same-pass value moves")) for row in UNDETERMINED.values()], ids=list(UNDETERMINED))
def test_a_settled_radius_is_the_radius_where_a_same_pass_rounding_moves_the_jacobian(
        case, score):
    """CPL-087 with the flag set and eight scalars crossing the group's
    edges.  Strict: withdrawing the flag where the Jacobian moves with a
    rounding of the iterate turns both green (a score is 0 where its flag
    is False), and :data:`SWEPT_PRODUCT_DECADES` then goes."""
    seen = observe(case)
    cell = CELLS[case.cell]
    assert cell.sweeps_a_product_of_two_members and abs(case.base.unit) > SWEPT_PRODUCT_DECADES
    assert seen["stepped"] and seen["referenced"], seen
    if seen[score] > THRESHOLD[score]:
        raise SettledOnAJacobianFloat32DoesNotDetermine(
            f"{score} is {seen[score]!r} with spectral_usable={seen['spectral_usable']}: "
            f"{seen['report']}")


#: The third example of that hunt: a curve of 0.019.  The float64 twin
#: holds the same-pass ``a`` 0.006 and 0.28 of a float32 rounding off its
#: centre, so for it the pass moves with ``c``'s centres ``c1[0]`` and
#: ``c1[1]`` by 2 and 146 floors, the gradient with respect to ``c1[0]``
#: at the returned iterate is a forty-seventh of the fixed point's, and
#: "gradient_vanishing" read 1.28 (47.2 beside a bound of 6.77).  In
#: float32 ``a' - c_0`` is exactly zero and so are both tangents: CPL-093's
#: bound "says nothing of" a constant whose tangent is exactly zero.
_NO_TANGENT_IN_FLOAT32 = _swept(65535, 0.019051312447631853)
#: The same on the hub with no acceleration under the l2 norm (the other
#: cell that sweeps a product), a member in units of 1e-6: the twin's pass
#: moves with three centres of ``l1`` by 34 to 222 floors where the float32
#: pass has no tangent, and with a fourth by 2.4 where the float32 pass's
#: is 695 -- the error is 6.7 of the twin's gradient at the returned
#: iterate ("gradient_vanishing" read 2.59) and 0.97 of the group's own,
#: under a bound of 1.93.
_ANOTHER_TANGENT_IN_FLOAT32 = Case(
    CELLS.index(Cell("hub", "float32", 0, 5, "product")),
    linear.Case(0, 34949, 0.05000000000000001, False, 1.0, 0.0, 1.0, -6, 0.0, 0), 1.0)


# Slow (the hub): one more cell compiled, with its twins.
# Per push: tests/property/test_coupling_nonlinear_search.py::test_a_centre_whose_tangent_is_a_float32_rounding_is_not_in_the_gradient_score
@pytest.mark.parametrize("case,centre", [
    pytest.param(_NO_TANGENT_IN_FLOAT32, "c.c1[1]", id="tri"),
    pytest.param(_ANOTHER_TANGENT_IN_FLOAT32, "l1.c0[0]", id="hub", marks=pytest.mark.slow)])
def test_a_centre_whose_tangent_is_a_float32_rounding_is_not_in_the_gradient_score(case, centre):
    """CPL-093 bounds the error of the gradient the group has, for a
    constant its pass resolves.  Where the float64 twin's response to a
    centre is not the float32 pass's (here: a hundred floors and more
    beside exactly none), the twin's gradient is not the group's, and the
    score leaves the centre out: it holds on what remains."""
    seen = observe(case)
    report = seen["report"]
    assert seen["stepped"] and seen["referenced"] and seen["gradient_usable"], seen
    assert CELLS[case.cell].sweeps_a_product_of_two_members
    dropped = report["unresolved_by_the_float32_pass"]
    twin_floors, own_floors = dropped[centre]
    # units: floors -- the twin's response is far above RESOLVED_MARGIN, the float32 pass's is none.
    assert twin_floors > 16.0 and own_floors == 0.0, dropped
    assert report["gradient_vanishing_constant"] not in dropped, report
    assert seen["gradient_vanishing"] <= THRESHOLD["gradient_vanishing"], report
    assert seen["gradient"] <= THRESHOLD["gradient"], report


def test_the_float32_pass_resolves_what_the_twin_resolves_away_from_the_centre():
    """The other side of the condition above, on the seed whose iterate is
    a thousandth of a field from its centres: every centre and curve the
    float64 twin's pass resolves, the float32 pass resolves too, so the
    score still takes them (the seed test holds that it reads 0.79)."""
    case, score, _least = SEEDS["a-nonlinearity-s-own-constant-resolved-short-of-the-fixed-point"]
    seen = observe(case)
    assert score == "gradient_vanishing" and seen["vanishing_scored"], seen
    assert seen["report"]["unresolved_by_the_float32_pass"] == {}, seen["report"]
    resolved, total = seen["report"]["gradient_vanishing_constants"]
    assert 0 < resolved <= total, seen["report"]


def test_only_a_cell_that_sweeps_a_product_of_two_members_has_its_units_held():
    """The radius scores are drawn as the linear search draws on every
    cell but the two that multiply two members' fields in a float32
    sweep, where a change of units is held within a decade (MADD-ANO-239);
    the other scores are drawn at every change of units there too."""
    swept = [i for i, c in enumerate(CELLS) if c.sweeps_a_product_of_two_members]
    assert swept == [_SWEPT_PRODUCT, _ANOTHER_TANGENT_IN_FLOAT32.cell]
    assert all(i in APPENDED_CELLS for i in swept)
    for index in ALL_CELLS:
        for unit in (0, -6, -3, -1, 1, 3, 6):
            case = Case(index, dataclasses.replace(_PROBE, unit=unit), 1.0)
            held = within_the_units_a_swept_product_is_claimed_for(case)
            if index in swept and abs(unit) > SWEPT_PRODUCT_DECADES:
                assert held.base.unit == (1 if unit > 0 else -1), (index, unit)
                assert dataclasses.replace(held, base=dataclasses.replace(
                    held.base, unit=unit)) == case
            else:
                assert held is case, (index, unit)
