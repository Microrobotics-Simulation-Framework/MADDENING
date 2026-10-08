"""Coupling groups with a ``multilinear_grid`` geometry edge, scored against the numerical reference.

``tests/property/test_coupling_nonlinear_search.py`` scores
``coupling_diagnostics()`` on nonlinear groups against
:mod:`tests.property.coupling_reference`.  This module is the same search
on groups whose pass is nonlinear *through a geometry*: a grid-side member
``F`` and a point-side member ``P`` joined by a gather (``F.x -> P.u``,
consistent) and a scatter (``P.x -> F.u``, conservative) of the library's
``multilinear_grid`` kind, each reading the positions ``pos`` of one of
its ends.  Nothing here is a test.

**The node** (:class:`GeoRelay`): ``x <- alpha x_pre + g u + c`` and, where
it holds positions, ``pos <- pos_pre + drift + (Q u)``: the positions move
with the node's own mapped input, which inside a group is the iterate.
Every number is a parameter or a state field, so one compiled graph serves
every draw of its cell.

**The cells** (:class:`Cell`) are named by where each edge reads its
geometry, which decides what the pass's Jacobian holds:

* ``anchors=("source", "source")``: each edge reads its source's
  positions *from the iterate* (the dict its value is read from).  With
  ``moving`` those positions follow the coupling, and the Jacobian has a
  geometry block in both directions; without, the positions the pass
  returns do not depend on the iterate, but the pass still reads them
  from it.
* ``anchors=("target", "target")``: each edge reads its target's
  positions from the member's *pre-step* state, a constant of the pass:
  no column of the Jacobian belongs to a geometry, and the positions are
  constants the gradient bound probes.
* ``anchors=("target", "source")``: both edges read ``P.pos``, the gather
  before the step and the scatter from the iterate.

**The scores** are the nonlinear search's (its module docstring), against
the reference's ``jacfwd`` Jacobian, which has every geometry field in the
iterate: the error bound with the ``1 / (1 - h)`` allowance of a nonlinear
map, the radius at the returned iterate, the gradient bound over every
constant (gains, biases, drifts and the pulls ``Q``), and the floor.  An
example is scored only where the returned iterate and the fixed point hold
every point in the same lattice cell (:func:`same_cells`): a multilinear
stencil is piecewise smooth, and across a lattice plane the Jacobian at
the returned iterate is another polynomial's.  The fraction scored is
returned beside the scores.

**Across a lattice plane** (:class:`PlaneCase`, :func:`observe_plane`).
The draws above start every point in the middle of a cell and leave the
unscored examples unscored.  A plane draw moves one drift so that the
*fixed point* holds one coordinate of one point a drawn distance (1e-6 to
1e-2 of a spacing) on either side of a lattice plane or of a face of the
hull, and is scored whether or not the returned iterate is in the fixed
point's cell: with ``spectral_usable`` set, every position the pass reads
from the iterate must be in the same lattice cell at the returned iterate
and at the fixed point,
or the fixed point must be within twice the bound all the same (the
``"plane"`` score, with no allowance for a nonlinearity; MAP-049), and
the bound is scored as on any other example.  :func:`plane_limit_reference` is the
slot ``geometry_plane_limit`` computed here from the returned state.

:func:`without_geometry` is the reference's own measure of what a product
that dropped the geometry would report: the radius of the Jacobian with
every geometry column zeroed.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import warnings
from typing import Optional

import jax.numpy as jnp
import numpy as np
from hypothesis import strategies as st

from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.property import coupled_graphs as cg
from tests.property import coupling_reference as cr
from tests.property import test_coupling_targeted_search as linear
from tests.property.sysid_transform_grid import precision
from tests.property.targeted_search import PER_PUSH, targeted_search

DT = 0.1
ALPHA = 0.3
#: The lattices, by dimension: ``(spacing, shape)``.  A different size on
#: each axis.
GRIDS = {1: ((0.5,), (4,)), 2: ((0.5, 0.25), (3, 2))}


class GeoRelay(SimulationNode):
    """``x <- alpha x_pre + g u + c``; with *points*, ``pos <- pos_pre + drift + Q u``.

    ``g`` (a scalar), ``c``, ``drift`` and ``Q`` are parameters; ``x`` and
    ``pos`` are state.  ``u`` is the node's one input, of the size of ``x``.
    """

    def __init__(self, name, timestep, *, n, points: Optional[tuple] = None, dtype="float32"):
        dt = jnp.dtype(dtype)
        params = {"g": jnp.asarray(0.0, dt), "c": jnp.zeros((n,), dt)}
        if points is not None:
            m, d = points
            params.update(drift=jnp.zeros((m, d), dt), Q=jnp.zeros((m * d, n), dt))
        super().__init__(name, timestep, **params)
        self._n, self._points, self._dtype = int(n), points, dt

    def initial_state(self):
        state = {"x": jnp.zeros((self._n,), self._dtype)}
        if self._points is not None:
            state["pos"] = jnp.zeros(self._points, self._dtype)
        return state

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=self._dtype,
                                       default=jnp.zeros((self._n,), self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs.get("u", jnp.zeros((self._n,), self._dtype))
        out = {"x": jnp.asarray(ALPHA, self._dtype) * state["x"] + p["g"] * u + p["c"]}
        if self._points is not None:
            out["pos"] = state["pos"] + p["drift"] + (p["Q"] @ u).reshape(self._points)
        return out

    def update_evaluations(self):
        return 1


#: One quiet ``solver="ift"`` configuration per row, under the two norms
#: that read the members' state: each stock acceleration, both schedules.
KNOBS = (
    dict(acceleration="none", iteration_mode="gauss-seidel", convergence_norm="l2"),
    dict(acceleration="none", iteration_mode="jacobi", convergence_norm="mixed"),
    dict(acceleration="aitken", iteration_mode="gauss-seidel", convergence_norm="mixed"),
    dict(acceleration="fixed", relaxation=0.7, iteration_mode="jacobi", convergence_norm="l2"),
    dict(acceleration="iqn-ils", iteration_mode="jacobi", convergence_norm="mixed"),
    dict(acceleration="iqn-imvj", jacobian_reuse=2, iteration_mode="gauss-seidel",
         convergence_norm="l2"),
)


@dataclasses.dataclass(frozen=True)
class Cell:
    """What one compiled graph bakes in."""

    #: Where the gather ``F.x -> P.u`` and the scatter ``P.x -> F.u`` read
    #: their positions.
    anchors: tuple
    #: Whether the positions move with their holder's input.
    moving: bool
    dtype: str
    knob: int
    cap: int
    d: int = 1
    m: int = 2
    #: The lattice's origin, in spacings (positions far from zero round
    #: coarsely against the spacing).
    origin: float = 0.5
    #: The order the members are added in (the Gauss-Seidel order).
    order: tuple = ("F", "P")
    #: The group's ``tolerance`` (``"l2"``) or ``rtol`` (``"mixed"``);
    #: zero for the linear search's.  The plane cells stop early, as a
    #: user's tolerance does: an iterate a lattice plane away from its
    #: fixed point.
    tolerance: float = 0.0

    @property
    def knobs(self) -> dict:
        g = dict(KNOBS[self.knob], solver="ift", diagnostics=True, max_iterations=self.cap)
        if g["convergence_norm"] == "l2":
            g["tolerance"] = self.tolerance or linear.TOLERANCE[self.dtype]
        else:
            g["rtol"] = self.tolerance or linear.RTOL
        return cg.live_knobs(g)

    @property
    def grid(self) -> tuple:
        spacing, shape = GRIDS[self.d]
        return tuple(self.origin * s for s in spacing), spacing, shape

    @property
    def n_grid(self) -> int:
        return int(np.prod(GRIDS[self.d][1]))

    @property
    def holders(self) -> tuple:
        """The members that hold the positions of ``(gather, scatter)``."""
        down, up = self.anchors
        return ("F" if down == "source" else "P", "P" if up == "source" else "F")

    @property
    def iterate_reads(self) -> tuple:
        """The members whose positions the pass reads from the iterate."""
        down, up = self.anchors
        return tuple(h for h, a in zip(self.holders, (down, up)) if a == "source")

    def __repr__(self) -> str:
        k = KNOBS[self.knob]
        return (f"{'-'.join(self.anchors)} {'moving' if self.moving else 'still'} {self.dtype} "
                f"{k['acceleration']} {k['iteration_mode']} {k['convergence_norm']} "
                f"cap{self.cap} {self.d}d m{self.m} o{self.origin:g} {''.join(self.order)}"
                + (f" tol{self.tolerance:g}" if self.tolerance else ""))


def build(cell: Cell, knobs: dict, dtype: str, *, node=GeoRelay) -> GraphManager:
    """A compiled graph of *cell* under *knobs* in *dtype*; *node* is the
    members' class (a test of a faulty member passes its own)."""
    origin, spacing, shape = cell.grid
    sizes = {"F": cell.n_grid, "P": cell.m}
    gm = GraphManager()
    for name in cell.order:
        gm.add_node(node(name, DT, n=sizes[name], dtype=dtype,
                         points=(cell.m, cell.d) if name in cell.holders else None))
    down, up = cell.anchors
    gm.add_edge("F", "P", "x", "u", geometry=(down, "pos"), mapping=multilinear_grid_mapping(
        origin, spacing, shape, n_points=cell.m, mode="consistent"))
    gm.add_edge("P", "F", "x", "u", geometry=(up, "pos"), mapping=multilinear_grid_mapping(
        origin, spacing, shape, n_points=cell.m, mode="conservative"))
    gm.add_coupling_group(["F", "P"], **cg.live_knobs(knobs))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        gm.compile()
    return gm


KEY = "F+P"


@dataclasses.dataclass(frozen=True)
class Case:
    """One drawn problem on one cell."""

    cell: int
    seed: int
    #: The product of the two members' gains over their fields' sizes (the
    #: value loop's gain up to the stencils' norms).
    loop: float
    #: How far a field of nominal size pulls a position, in lattice cells.
    pull: float
    #: The sizes of ``F.x`` and of ``P.x``.
    size_f: float = 1.0
    size_p: float = 1.0

    def eps(self, cells) -> float:
        return float(np.finfo(cells[self.cell].dtype).eps)


def values_of(case: Case, cell: Cell) -> dict:
    """The numbers of *case* on *cell* (float64; the builders cast)."""
    rng = np.random.default_rng(case.seed)
    origin, spacing, shape = cell.grid
    sizes = {"F": (cell.n_grid, case.size_f, case.size_p),
             "P": (cell.m, case.size_p, case.size_f)}
    out = {}
    for name in ("F", "P"):
        n, size, read = sizes[name]
        v = {"x": size * rng.uniform(-1.0, 1.0, n), "c": size * rng.uniform(-1.0, 1.0, n),
             "g": math.sqrt(case.loop) * size / read * (1.0 if rng.random() < 0.5 else -1.0)}
        # Drawn for every member, held or not, so a cell's draws do not
        # depend on its anchors.
        base = np.stack([rng.integers(0, max(s - 1, 1), cell.m) for s in shape], axis=1)
        index = base + rng.uniform(0.3, 0.7, (cell.m, cell.d))
        drift = 0.05 * rng.uniform(-1.0, 1.0, (cell.m, cell.d))
        Q = rng.uniform(-1.0, 1.0, (cell.m * cell.d, n)) / math.sqrt(n)
        if name in cell.holders:
            sp = np.asarray(spacing)
            v["pos"] = np.asarray(origin) + index * sp
            v["drift"] = drift * sp
            scale = np.tile(sp, cell.m)[:, None]
            v["Q"] = (case.pull if cell.moving else 0.0) * scale * Q / read
        out[name] = v
    return out


def set_initial(gm: GraphManager, values: dict) -> None:
    """Reset *gm* and write every member's state from *values*."""
    cg.recover(gm)
    gm.reset_state()
    for name, v in values.items():
        state = dict(gm.get_node_state(name))
        for field in state:
            state[field] = jnp.asarray(v[field], state[field].dtype)
        gm.set_node_state(name, state)


def params_for(gm: GraphManager, values: dict) -> dict:
    """``gm.params`` with every member's constants replaced by *values*."""
    base = gm.params
    nodes = {name: dict(p) for name, p in base["nodes"].items()}
    for name, v in values.items():
        for leaf in nodes[name]:
            nodes[name][leaf] = jnp.asarray(v[leaf], nodes[name][leaf].dtype)
    return {**base, "nodes": nodes}


_PROBE = Case(0, 7, 0.25, 0.1)


def built(cell: Cell, **kw) -> tuple:
    """``(the graph under test, its x64 twin, the twin's reference)``."""
    with precision(cell.dtype == "float64"):
        gm = build(cell, cell.knobs, cell.dtype, **kw)
    with cr.x64():
        twin = build(cell, cr.twin_knobs(cell.knobs), "float64", **kw)
        values = values_of(_PROBE, cell)
        set_initial(twin, values)
        return gm, twin, cr.PassReference.of(twin, params=params_for(twin, values))


def bound_reference(ref: cr.PassReference, twin: GraphManager, values: dict) -> cr.PassReference:
    """*ref* bound to the start and the constants of *values* on its twin."""
    with cr.x64():
        set_initial(twin, values)
        return ref.at(twin._state, params_for(twin, values))      # noqa: SLF001


def snapshot(gm: GraphManager) -> dict:
    return {n: {f: np.asarray(v) for f, v in gm.get_node_state(n).items()}
            for n in gm.node_names}


def run_once(gm: GraphManager, values: dict) -> tuple:
    """``(pre, state, report, meta)`` of one step from the drawn start."""
    set_initial(gm, values)
    pre = snapshot(gm)
    gm.step(params=params_for(gm, values))
    return pre, snapshot(gm), dict(gm.coupling_diagnostics()[KEY]), cg.group_meta(gm, KEY)


def lattice_cells(cell: Cell, ref: cr.PassReference, x) -> list:
    """The lattice cell of every point the pass reads from iterate *x*."""
    origin, spacing, _shape = cell.grid
    out = []
    for name in sorted(set(cell.holders)):
        pos = np.asarray(ref.field(np.asarray(x, np.float64), name, "pos"))
        out.append(np.floor((pos - np.asarray(origin)) / np.asarray(spacing)).astype(int))
    return out


def same_cells(cell: Cell, ref: cr.PassReference, *iterates) -> bool:
    """Whether every point is in the same lattice cell at each of *iterates*."""
    first = lattice_cells(cell, ref, iterates[0])
    return all(all(np.array_equal(a, b) for a, b in zip(first, lattice_cells(cell, ref, x)))
               for x in iterates[1:])


def geometry_columns(cell: Cell, ref: cr.PassReference) -> np.ndarray:
    """A mask over the flat iterate: the entries of every ``pos`` field."""
    mask = np.zeros(ref.size, bool)
    for _n, f, _shape, a, b in ref.layout:
        if f == "pos":
            mask[a:b] = True
    return mask


def without_geometry(cell: Cell, ref: cr.PassReference, J: np.ndarray) -> float:
    """The spectral radius of *J* with every geometry column zeroed: what
    a product that did not see the pass's dependence on the positions it
    reads from the iterate would be the product of."""
    dropped = np.array(J, np.float64)
    dropped[:, geometry_columns(cell, ref)] = 0.0
    return cr.radius(dropped)


def _cancellation(cell: Cell, values: dict, pre: dict, state: dict) -> float:
    """How far the worst field cancels inside its own update: the
    magnitudes the update sums over the field's largest entry."""
    worst = 1.0
    reads = {"F": "P", "P": "F"}
    for name in ("F", "P"):
        v = values[name]
        u = float(np.max(np.abs(np.asarray(state[reads[name]]["x"], np.float64))))
        terms = [(ALPHA * np.abs(np.asarray(pre[name]["x"], np.float64)) + abs(v["g"]) * u
                  + np.abs(v["c"]), state[name]["x"])]
        if "pos" in state[name]:
            terms.append((np.abs(np.asarray(pre[name]["pos"], np.float64)) + np.abs(v["drift"])
                          + (np.abs(v["Q"]) @ np.full(v["Q"].shape[1], u)).reshape(
                              v["drift"].shape), state[name]["pos"]))
        for total, field in terms:
            size = float(np.max(np.abs(np.asarray(field, np.float64))))
            if size > 0:
                worst = max(worst, float(np.max(total)) / size)
    return worst


#: A constant is scored for the gradient bound where its pass response is
#: above this many floors (the nonlinear search's ``RESOLVED_MARGIN``).
RESOLVED_MARGIN = 2.0


def crossed(cell: Cell, ref: cr.PassReference, a, b) -> bool:
    """Whether a coordinate the pass reads from the iterate is in another
    lattice cell at *b* than at *a* (a coordinate exactly on a plane is in
    both cells beside it: the stencil is continuous there)."""
    origin, spacing, shape = cell.grid
    for name in sorted(set(cell.iterate_reads)):
        ua, ub = ((np.asarray(ref.field(np.asarray(x, np.float64), name, "pos"))
                   - np.asarray(origin)) / np.asarray(spacing) for x in (a, b))
        lo, hi = np.minimum(ua, ub), np.maximum(ua, ub)
        top = np.asarray(shape, np.float64) - 1.0
        # A plane strictly between the two, among the planes 0 .. top.
        first = np.maximum(np.floor(lo) + 1.0, 0.0)
        if np.any((first < hi) & (first <= top)):
            return True
    return False


def plane_limit_reference(cell: Cell, ref: cr.PassReference, x, norm: cr.Norm) -> float:
    """``geometry_plane_limit`` of returned iterate *x*, from its
    definition: the smallest distance, in *norm* at the returned state's
    weights, from *x* to a state with one coordinate the pass reads from
    the iterate on a lattice plane (or on the face of the hull it is
    clamped to), halved (``GEOMETRY_PLANE_REACH``); zero where the pass
    itself moves a coordinate across; ``inf`` where the pass reads no
    position from the iterate."""
    origin, spacing, shape = cell.grid
    x = np.asarray(x, np.float64)
    after = ref.apply(x)
    weights = norm.weights(x)
    best = math.inf
    for name in sorted(set(cell.iterate_reads)):
        k = [(n, f) for n, f, _s, _a, _b in ref.layout].index((name, "pos"))
        _n, _f, _shape, a, b = ref.layout[k]
        for j in range(a, b):
            axis = (j - a) % cell.d
            n = shape[axis]
            if n < 2:
                continue
            u = (x[j] - origin[axis]) / spacing[axis]
            near = -u if u < 0 else u - (n - 1) if u > n - 1 else min(u - math.floor(u),
                                                                    1.0 - (u - math.floor(u)))
            dist = near * spacing[axis]
            if abs(after[j] - x[j]) > dist:
                dist = 0.0
            moved = x.copy()
            moved[j] += dist
            best = min(best, norm.of_difference(moved, x, weights))
    return best / 2.0


def without_plane_limit(gm: GraphManager) -> dict:
    """The report of *gm*'s last step with the lattice-plane criterion
    out of it (the slot read as ``inf``): what ``spectral_usable`` said
    before the criterion existed."""
    slot = f"coupling_{KEY}_geometry_plane_limit"
    kept = gm._state                                               # noqa: SLF001
    if slot not in kept.get("_meta", {}):
        return dict(gm.coupling_diagnostics()[KEY])
    try:
        gm._state = {**kept, "_meta": {**kept["_meta"], slot: np.asarray(        # noqa: SLF001
            np.inf, np.asarray(kept["_meta"][slot]).dtype)}}
        return dict(gm.coupling_diagnostics()[KEY])
    finally:
        gm._state = kept                                           # noqa: SLF001


def observe(cell: Cell, case: Case, trio: tuple, *, values: Optional[dict] = None,
            across: bool = False) -> dict:
    """One step of *case* on *cell* (``trio``: :func:`built`) and the
    scores of what it reported.  *values*: the numbers, where they are
    not ``values_of(case, cell)``.  With *across* the example is scored
    whether or not the fixed point is in the returned iterate's lattice
    cells (:func:`observe_plane`)."""
    gm, twin, ref = trio
    values = values_of(case, cell) if values is None else values
    ref = bound_reference(ref, twin, values)
    eps = float(np.finfo(cell.dtype).eps)
    with precision(cell.dtype == "float64"):
        pre, state, d, meta = run_once(gm, values)
        floor = (linear._reported_floor(gm, KEY, meta, d)          # noqa: SLF001
                 if not math.isnan(d["error_estimate"]) or "not_usable_reason" not in d
                 else math.nan)
        before = without_plane_limit(gm)
    out = dict(bound=0.0, radius=0.0, radius_strict=0.0, gradient=0.0, floor=0.0, plane=0.0,
               plane_before=0.0, usable_before=bool(before["spectral_usable"]), crossed=False,
               spectral_usable=bool(d["spectral_usable"]),
               gradient_usable=bool(d["gradient_bound_usable"]),
               floor_reported=math.isfinite(floor), referenced=False, near=False,
               scored=False, finite=False,
               reason=d.get("not_usable_reason"),
               report={k: d[k] for k in ("iterations", "converged", "residual", "rho_spectral",
                                         "spectral_error_bound", "spectral_usable",
                                         "gradient_relative_error_bound",
                                         "gradient_bound_usable", "precision_limited")})
    if "geometry_gap" in meta:
        out["report"]["geometry_gap"] = float(meta["geometry_gap"])
    if "geometry_plane_limit" in meta:
        out["report"]["plane_limit"] = float(meta["geometry_plane_limit"])
    finite = all(np.all(np.isfinite(f)) for s in state.values() for f in s.values())
    if not finite or not math.isfinite(d["residual"]):
        return out
    out["finite"] = True
    x = ref.flat(state)
    fixed = ref.fixed_point(x)
    out["report"].update(reference_ulps=fixed.ulps)
    if not fixed.converged:
        return out
    out["referenced"] = True
    kind = cell.knobs["convergence_norm"]
    norm = ref.norm(kind, cell.knobs.get("rtol", 1e-6))
    out["crossed"] = crossed(cell, ref, x, fixed.x)
    out["report"]["plane_limit_reference"] = plane_limit_reference(cell, ref, x, norm)
    if out["crossed"]:
        # The fixed point in another lattice cell than the iterate the
        # spectrum was taken at.  With the flag set the fixed point is
        # within the Newton-Kantorovich ball, twice the bound: no
        # allowance for a nonlinearity, which is the plane's own.
        dist = ref.distance(x, fixed, norm)
        bound = float(d["spectral_error_bound"])
        over = dist / (2.0 * bound) if bound > 0 else math.inf
        out["report"].update(distance=dist, distance_over_bound=2.0 * over)
        out["plane_before"] = over if out["usable_before"] else 0.0
        if out["spectral_usable"]:
            out["plane"] = math.inf if math.isnan(over) else over
    if across:
        if out["crossed"] or crossed(cell, ref, x, ref.apply(x)):
            return out      # the Jacobian at the iterate is another polynomial's
    elif not same_cells(cell, ref, x, fixed.x, ref.apply(x)):
        return out          # across a lattice plane: another polynomial's Jacobian
    out["scored"] = True
    residual = float(d["residual"])
    cancels = _cancellation(cell, values, pre, state)
    allowed = ((residual + cancels * floor) / (residual + floor)
               if out["floor_reported"] and residual + floor > 0 else 1.0)
    out["report"].update(floor=floor, cancellation=cancels)

    if out["floor_reported"]:
        true = ref.residual(x, norm)
        above = true - residual * (1.0 + 2.0 ** 8 * eps)
        out["floor"] = max(0.0, above) / max(cancels * floor, 1e-300)
        out["report"]["residual_true"] = true

    J = ref.jacobian(x)
    out["report"].update(rho_true=cr.radius(J), rho_without_geometry=without_geometry(
        cell, ref, J))
    weights = np.ones(ref.size)
    for _n, _f, _shape, a, b in ref.layout:
        top = float(np.max(np.abs(x[a:b])))
        weights[a:b] = 1.0 / top if top > 0 else 1.0
    linear.radius_scores(out, J, weights, float(d["rho_spectral"]), eps, case.seed)

    if out["spectral_usable"]:
        dist = ref.distance(x, fixed, norm)
        h = ref.nonlinearity(x, fixed, norm)
        bound = float(d["spectral_error_bound"]) * allowed
        out["report"].update(distance=dist, nonlinearity=h,
                             distance_over_bound=dist / bound if bound > 0 else math.inf)
        out["near"] = h < 1.0
        if out["near"]:
            reach = bound / (1.0 - h)
            out["bound"] = (math.inf if math.isnan(bound) else
                            dist / reach if reach > 0 else (math.inf if dist > 0 else 0.0))

    if out["gradient_usable"]:
        bound = (float(d["gradient_relative_error_bound"]) * allowed + 64.0 * eps
                 + ref.gradient_resolution(x, norm))
        response = ref.pass_responses(x, norm)
        resolved = [not out["floor_reported"] or r > RESOLVED_MARGIN * floor for r in response]
        true, column = ref.gradient_error(x, fixed, norm, columns=resolved)
        out["gradient"] = math.inf if math.isnan(bound) else true / bound
        out["report"].update(gradient_error=true, constants_resolved=(sum(resolved),
                                                                       len(resolved)),
                             gradient_constant=(None if column is None
                                                else ref.constant_names()[column]))
    return out


THRESHOLD = {**linear.THRESHOLD, "plane": 1.0}
FLAG = {**linear.FLAG, "plane": "usable_before"}
SEARCHES = linear.SEARCHES
#: The scores of a plane draw: the flag's statement about lattice planes,
#: and the two numbers it covers.
PLANE_SEARCHES = ("plane", "bound", "radius")


@dataclasses.dataclass(frozen=True)
class PlaneCase(Case):
    """A drawn problem whose fixed point holds one coordinate beside a
    lattice plane."""

    #: Which coordinate of the positions (an index into the flattened
    #: ``pos`` of the first member the pass reads positions of from the
    #: iterate, or of ``P`` where it reads none), modulo their number.
    which: int = 0
    #: Which lattice plane of that coordinate's axis, modulo their number:
    #: the first and the last are the faces of the hull.
    plane: int = 0
    #: The signed distance of the fixed point's coordinate from the plane,
    #: in spacings.
    offset: float = 1e-4


def plane_holder(cell: Cell) -> str:
    return cell.iterate_reads[0] if cell.iterate_reads else "P"


def plane_values(cell: Cell, case: PlaneCase, trio: tuple, *, rounds: int = 12) -> Optional[dict]:
    """The numbers of *case* with one drift moved so that the fixed point
    of the pass from the drawn start holds the chosen coordinate *offset*
    spacings from the chosen plane; ``None`` where the reference did not
    get there (no fixed point, or one the drift does not steer)."""
    _gm, twin, ref = trio
    origin, spacing, shape = cell.grid
    values = values_of(case, cell)
    holder = plane_holder(cell)
    entry = case.which % (cell.m * cell.d)
    point, axis = divmod(entry, cell.d)
    target = (origin[axis] + (case.plane % shape[axis] + case.offset) * spacing[axis])
    drift = np.array(values[holder]["drift"], np.float64)
    for _ in range(rounds):
        values[holder]["drift"] = drift.copy()
        bound = bound_reference(ref, twin, values)
        with cr.x64():
            start = bound.flat({n: {f: values[n][f] for f in ("x", "pos") if f in values[n]}
                                for n in ("F", "P")})
        fixed = bound.fixed_point(start, picard=40)
        if not fixed.converged:
            return None
        miss = target - float(np.asarray(bound.field(fixed.x, holder, "pos"))[point, axis])
        if abs(miss) <= 1e-3 * abs(case.offset) * spacing[axis]:
            return values
        drift[point, axis] += miss
    return None


def observe_plane(cell: Cell, case: PlaneCase, trio: tuple) -> dict:
    """:func:`observe` of a plane draw; an example the generator could not
    place is returned unscored with ``placed=False``."""
    values = plane_values(cell, case, trio)
    if values is None:
        return dict(bound=0.0, radius=0.0, radius_strict=0.0, gradient=0.0, floor=0.0,
                    plane=0.0, plane_before=0.0, usable_before=False, crossed=False,
                    spectral_usable=False,
                    gradient_usable=False, floor_reported=False, referenced=False,
                    near=False, scored=False, finite=False, placed=False, reason=None,
                    report={})
    return {**observe(cell, case, trio, values=values, across=True), "placed": True}


def plane_cases(indices):
    """Draw a :class:`PlaneCase` on one of the cells *indices*: the usual
    numbers, a coordinate, a plane and a signed distance of 1e-6 to 1e-2
    of a spacing."""
    offsets = st.tuples(st.floats(-6.0, -2.0), st.sampled_from((-1.0, 1.0))).map(
        lambda t: t[1] * 10.0 ** t[0])
    return st.builds(PlaneCase, cell=st.sampled_from(tuple(indices)),
                     seed=st.integers(0, 2 ** 16), loop=st.floats(0.01, 1.2),
                     pull=st.floats(0.02, 0.2), which=st.integers(0, 11),
                     plane=st.integers(0, 7), offset=offsets)


def cases(cells: tuple, indices):
    """Draw a :class:`Case` on one of the cells *indices* of *cells*."""
    del cells
    decades = st.floats(-2.0, 2.0).map(lambda v: 10.0 ** v)
    return st.builds(Case, cell=st.sampled_from(tuple(indices)), seed=st.integers(0, 2 ** 16),
                     loop=st.floats(0.01, 1.2), pull=st.floats(0.0, 0.2),
                     size_f=decades, size_p=decades)


class Search:
    """The searches over a tuple of cells, each compiled once.

    *twins*: another search whose cell at every index is this one's but
    for its tolerance.  Its twin and reference are used (a twin runs one
    pass whatever the tolerance), and only the graph under test is
    compiled here.
    """

    def __init__(self, cells: tuple, *, cache: int = 8, twins: Optional["Search"] = None):
        self.cells = cells

        def trio(index: int) -> tuple:
            cell = cells[index]
            if twins is None:
                return built(cell)
            assert dataclasses.replace(twins.cells[index], tolerance=cell.tolerance) == cell
            with precision(cell.dtype == "float64"):
                gm = build(cell, cell.knobs, cell.dtype)
            return (gm, *twins._built(index)[1:])

        self._built = functools.lru_cache(maxsize=cache)(trio)
        self._seen: dict = {}

    def observe(self, case: Case) -> dict:
        if case not in self._seen:
            look = observe_plane if isinstance(case, PlaneCase) else observe
            self._seen[case] = look(self.cells[case.cell], case, self._built(case.cell))
        return self._seen[case]

    def run(self, name: str, indices, *, profile=None, fail: bool = True,
            planes: bool = False) -> tuple:
        """Run search *name* over the cells *indices*; returns ``(report,
        {fraction name: value})``.  With *planes* the draws are
        :func:`plane_cases`."""
        profile = PER_PUSH if profile is None else profile
        drawn = []

        def score(case: Case):
            drawn.append(case)
            seen = self.observe(case)
            return seen[name], seen["report"]

        strategy = plane_cases(indices) if planes else cases(self.cells, indices)
        report = targeted_search(strategy, score, THRESHOLD[name],
                                 profile=profile, label=name, fail=fail)
        seen = [self.observe(c) for c in drawn]
        count = max(len(seen), 1)
        return report, {k: sum(bool(s[k]) for s in seen) / count
                        for k in (FLAG[name], "referenced", "scored", "near", "finite",
                                  "crossed", "usable_before", "spectral_usable")}
