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
   derivative taken at the returned iterate, the worst over every scalar
   constant (the nonlinearity's ``c`` and ``s`` included), over
   ``gradient_relative_error_bound`` where ``gradient_bound_usable`` --
   the flag that certifies the Newton-Kantorovich check, which on these
   cells is not trivially passed.
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

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import strategies as st

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
    grad, grad_exact = ref.gradient_error(x, fixed, raw)[0], linear._gradient_error(   # noqa: SLF001
        model, step.pre, step.state)
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


def _cells() -> tuple:
    """Every structure with every nonlinearity at both dtypes, the
    configurations and the caps rotated as the linear search rotates them."""
    out = []
    for s, name in enumerate(STRUCTURES):
        for q, kind in enumerate(KINDS):
            for t, dtype in enumerate(("float32", "float64")):
                knob = (s + 3 * t + 2 * q) % len(linear.KNOBS)
                out.append(Cell(name, dtype, knob, linear.CAPS[(s + t + q) % 2], kind))
    return tuple(out)


#: The per-push cell: ``tri`` with a saturating gain on ``a``, a quadratic
#: term on ``b`` and the product of two fields on ``c``, in float32,
#: stopped after five Jacobi passes (a returned iterate away from the
#: fixed point, where the Jacobian is another matrix).  One cell, because
#: a cell's cost is the compile of the graph under test with its
#: diagnostics (seconds); the three nonlinearities apart, at both dtypes
#: and under every configuration, are the slow hunt's.
_FIRST = (Cell("tri", "float32", 1, 5, "each"),)
CELLS = _FIRST + tuple(c for c in _cells() if c not in _FIRST)
PER_PUSH_CELLS = tuple(range(len(_FIRST)))
ALL_CELLS = tuple(range(len(CELLS)))
#: The slow hunt's blocks (each cell compiles a graph and a twin).
BLOCKS = tuple(ALL_CELLS[k::6] for k in range(6))


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


def leaves_float_range(cell: Cell, values: dict, ref: cr.PassReference) -> bool:
    """Whether plain passes from the drawn start leave *cell*'s float
    range before its cap, on a cell with a quasi-Newton acceleration.

    Such an example is not stepped: under ``"iqn-ils"`` and ``"iqn-imvj"``
    a group whose iterate becomes non-finite before the cap never returns
    from ``step()`` (the finding pinned by
    ``test_a_group_that_leaves_float_range_under_iqn_still_returns``), and
    a search cannot score a call that does not return.  Every other
    acceleration returns a non-finite state, which is scored as unusable.
    *ref* is bound to *values*.
    """
    if not str(cell.knobs.get("acceleration", "none")).startswith("iqn"):
        return False
    x = ref.flat({m: {"x": values["nodes"][m]["x0"]} for m in cell.topo.names})
    limit = math.sqrt(float(np.finfo(cell.dtype).max))
    for _ in range(min(cell.cap, 60)):
        with np.errstate(all="ignore"):
            x = ref.apply(x)
        if not np.all(np.isfinite(x)) or float(np.max(np.abs(x))) > limit:
            return True
    return False


@functools.lru_cache(maxsize=4096)
def observe(case: Case) -> dict:
    """One step of *case* and the scores of what it reported."""
    cell = CELLS[case.cell]
    topo = cell.topo
    values = values_of(case)
    built, twin, ref = _built(case.cell)
    ref = bound_reference(ref, values, twin)
    if leaves_float_range(cell, values, ref):
        # Not stepped (see ``leaves_float_range``): nothing is scored.
        return dict(bound=0.0, radius=0.0, radius_strict=0.0, gradient=0.0, floor=0.0,
                    spectral_usable=False, gradient_usable=False, floor_reported=False,
                    referenced=False, near=False, stepped=False, report={})
    with precision(cell.dtype == "float64"):
        step = run_once(built, values)
        d = dict(step.reports[0])
        floor = linear._reported_floor(built.gm, topo.group_key(0), step.metas[0], d)   # noqa: SLF001
    out = dict(bound=0.0, radius=0.0, radius_strict=0.0, gradient=0.0, floor=0.0,
               spectral_usable=bool(d["spectral_usable"]),
               gradient_usable=bool(d["gradient_bound_usable"]),
               floor_reported=math.isfinite(floor), referenced=False, near=False, stepped=True,
               report={k: d[k] for k in ("iterations", "converged", "residual", "rho_spectral",
                                         "spectral_error_bound", "spectral_usable",
                                         "gradient_relative_error_bound",
                                         "gradient_bound_usable", "precision_limited")})
    finite = all(np.all(np.isfinite(s["x"])) for s in step.state.values())
    if not finite or not math.isfinite(d["residual"]):
        return out
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
        true, column = ref.gradient_error(x, fixed, raw)
        bound = float(d["gradient_relative_error_bound"]) * allowed + 64.0 * case.eps
        out["gradient"] = math.inf if math.isnan(bound) else true / bound
        out["report"].update(gradient_error=true, gradient_constant=(
            None if column is None else ref.constant_names()[column]))
    return out


# ---------------------------------------------------------------------------
# The strategy and the searches
# ---------------------------------------------------------------------------

#: The curves drawn, in decades: a hundredth (all but linear over a
#: field's own size) to a hundred (a returned iterate a thousandth of its
#: field from the fixed point still sees the Jacobian move by a tenth).
CURVES = (-2.0, 2.0)

THRESHOLD = linear.THRESHOLD
FLAG = linear.FLAG
SEARCHES = linear.SEARCHES
#: The least fraction of a hunt's examples with the flag set, and with a
#: reference (a fixed point Newton reached).
USABLE_FLOOR = 0.25
REFERENCED_FLOOR = 0.9


def cases(cells=ALL_CELLS, domain: linear.Domain = linear.CLAIMED, curves=CURVES):
    """Draw a :class:`Case` on one of *cells*: the linear search's numbers
    within *domain* and a curve."""
    return st.builds(Case, cell=st.sampled_from(tuple(cells)), base=linear.cases((0,), domain),
                     curve=st.floats(*curves).map(lambda x: 10.0 ** x))


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

    report = targeted_search(cases(cells, domain), score, THRESHOLD[name], profile=profile,
                             label=name, fail=fail)
    seen = [observe(c) for c in drawn]
    count = max(len(seen), 1)
    return report, dict(usable=sum(s[FLAG[name]] for s in seen) / count,
                        referenced=sum(s["referenced"] for s in seen) / count,
                        near=sum(s["near"] for s in seen) / count,
                        stepped=sum(s["stepped"] for s in seen) / count)
