"""A search for wrong numbers in ``coupling_diagnostics()``, between the shapes a grid samples.

``coupling_diagnostics()`` reports four numbers a user is told to rely on
where a flag says so: ``spectral_error_bound`` (a bound on the distance to
the fixed point), ``rho_spectral`` (the spectral radius of the pass), the
``gradient_relative_error_bound`` and the residual's float floor (which
``precision_limited`` reads and the bound adds).  Each has been wrong with
its flag set on one more graph shape per review.  This module searches for
the next one with ``tests/property/targeted_search.py``: four scores say
how far each number is on the wrong side of an exact answer, and
Hypothesis climbs each of them.

**The generator** (:func:`cases`) draws a compiled *cell* and the numbers
put on it.  A cell (:data:`CELLS`) is a structure of
:func:`tests.property.coupled_topologies.search_topologies` -- rings of 2
to 8, a multi-rate ring, a pair that carries a prescribed spectrum, a
fan-out hub with a chord and a field read twice, a ring of mapped edges
(dense or sparse) -- at float32 or float64, under one of
:data:`KNOBS` (each stock acceleration, both schedules, the three norms)
and a cap of a few passes or many.  The numbers are arguments of the
compiled step, so a cell compiles once: the loop gain (0.05 to 0.98, and
up to ``1 - 1e-7`` where :class:`Domain` allows), non-normal gains, a
field small beside what drives it (its bias cancels its inputs:
:func:`~tests.property.coupled_topologies.place_group_fixed_point`),
slow modes within a drawn fraction of each other, a mapping row that
differences two entries of a large field, a decimal change of units on
one node, and how far from the fixed point the step starts.

**The reference** is :class:`~tests.property.coupled_topologies.LinearModel`
of the values as the graph holds them: the fixed point and the pass's
Jacobian in closed form, in the reference's extended precision.

**The scores** (each 0 where its flag is False: an unusable number is
never a failure; the fraction of examples whose flag was set is reported
and held to a floor in the slow profile):

1. *error bound* (CPL-088): the true distance to the fixed point, in the
   group's norm at the returned state, over ``spectral_error_bound``,
   where ``spectral_usable``;
2. *spectral radius* (CPL-087): ``|rho_spectral - rho|`` over what the
   claim allows -- "exact ... to float32" where the space settled
   (``spectral_usable``), read as 1e-4 of the radius plus the movement of
   the radius under a perturbation of the weighted Jacobian of the
   analysis dtype's rounding (:func:`_radius_allowance`); and, where the
   weighted Jacobian is normal, how far the estimate is *above* the
   radius ("from below for a normal dF/dx"), whatever the flag.  No row
   promises ``rho_spectral`` as an upper estimate, so none is scored;
3. *gradient bound* (CPL-093): the true relative error of the implicit
   derivative taken at the returned iterate against the same dense solve
   at the fixed point, the worst over every scalar gain and mapping
   weight, over ``gradient_relative_error_bound``, where
   ``gradient_bound_usable``;
4. *floor* (CPL-097, CPL-100): how far the exact residual of the returned
   state is *above* the reported one, over the floor the report used --
   at a stalled iterate, whose reported residual is zero, the plateau
   the solve cannot go below over the floor.

The floor is promised only where no node "cancels inside itself"
(CPL-092's condition).  A relay whose bias cancels its inputs does, by
the factor :func:`_cancellation` measures, so scores 1 and 4 allow the
floor that factor: a mapping row that cancels is *not* inside a node and
is not allowed for (MADD-ANO-212).

Per push: each search on :data:`PER_PUSH_CELLS`, derandomised, at the
house floor of examples, over the domain the claims are made on and
this tree holds (:data:`CLAIMED`).  Slow: the random hunt over every
cell.  What the widened domains reach is pinned at the foot of the
module as strict xfails naming the finding each is.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from hypothesis import strategies as st

from tests.property import coupled_graphs as cg
from tests.property import coupled_topologies as ct
from tests.property.sysid_transform_grid import precision
from tests.property.targeted_search import PER_PUSH, SLOW, targeted_search

STRUCTURES = ct.search_topologies()

#: One quiet ``solver="ift"`` configuration per row: each stock
#: acceleration, both schedules, the three norms.
KNOBS = (
    dict(acceleration="none", iteration_mode="gauss-seidel", convergence_norm="l2"),
    dict(acceleration="none", iteration_mode="jacobi", convergence_norm="l2"),
    dict(acceleration="aitken", iteration_mode="gauss-seidel", convergence_norm="interface"),
    dict(acceleration="fixed", relaxation=0.7, iteration_mode="jacobi",
         convergence_norm="mixed"),
    dict(acceleration="iqn-ils", iteration_mode="jacobi", convergence_norm="mixed"),
    dict(acceleration="iqn-imvj", jacobian_reuse=2, iteration_mode="gauss-seidel",
         convergence_norm="l2"),
    dict(acceleration="none", iteration_mode="gauss-seidel", convergence_norm="interface"),
)
#: A cap that stops the solve early and one that lets it converge or stall.
CAPS = (5, 120)
#: The ``"l2"`` tolerance per dtype (the default in float32; float64 is
#: asked for what it can resolve) and the relative norms' ``rtol``.
TOLERANCE = {"float32": 1e-6, "float64": 1e-10}
RTOL = 1e-4


@dataclasses.dataclass(frozen=True)
class Cell:
    """What one compiled graph bakes in."""

    structure: str
    dtype: str
    knob: int
    cap: int
    mapping_kind: str = "matrix"

    @property
    def topo(self) -> ct.Topology:
        return STRUCTURES[self.structure]

    @property
    def knobs(self) -> dict:
        g = dict(KNOBS[self.knob], solver="ift", diagnostics=True, max_iterations=self.cap)
        if g["convergence_norm"] == "l2":
            g["tolerance"] = TOLERANCE[self.dtype]
        else:
            g["rtol"] = RTOL
        if self.structure.endswith("multirate"):
            g["subcycling"] = True
        return cg.live_knobs(g)

    @property
    def cfgs(self) -> list:
        return ct.group_cfgs_of([self.knobs])


def _cells() -> tuple:
    """Every structure and mapping kind at both dtypes, under three of the
    configurations and both caps, rotated so each configuration meets each
    structure at one dtype or the other."""
    shapes = [(name, "matrix") for name in STRUCTURES] + [("mapped", "sparse-ragged")]
    out = []
    for s, (name, kind) in enumerate(shapes):
        for t, dtype in enumerate(("float32", "float64")):
            for j in range(3):
                knob = (s + 3 * t + 2 * j) % len(KNOBS)
                out.append(Cell(name, dtype, knob, CAPS[(s + t + j) % 2], kind))
    return tuple(dict.fromkeys(out))


#: The first cells are the per-push ones (:data:`PER_PUSH_CELLS`), chosen
#: for the shapes past defects lived on: the ring of three scalars under
#: Jacobi in float64 (a small field), the pair with a prescribed spectrum
#: stopped early in float64 (near-degenerate modes), the mapped ring under
#: Gauss-Seidel in float32 (a cancelling mapping row at the floor) and the
#: hub under the interface norm in float32 (fan-out).
_FIRST = (
    Cell("ring-3", "float64", 1, 120),
    Cell("pair-3", "float64", 0, 5),
    Cell("mapped", "float32", 0, 120),
    Cell("hub", "float32", 6, 5),
)
CELLS = _FIRST + tuple(c for c in _cells() if c not in _FIRST)
PER_PUSH_CELLS = tuple(range(len(_FIRST)))
ALL_CELLS = tuple(range(len(CELLS)))


@functools.lru_cache(maxsize=12)
def _built(index: int) -> ct.Built:
    cell = CELLS[index]
    with precision(cell.dtype == "float64"):
        return ct.build(cell.topo, cell.knobs, dtype=cell.dtype, mapping_kind=cell.mapping_kind)


@dataclasses.dataclass(frozen=True)
class Case:
    """One drawn problem on one cell."""

    cell: int
    seed: int
    #: The pass's spectral radius under Jacobi (the loop gain).
    rho: float
    nonnormal: bool
    #: One member's field at the fixed point, as a fraction of the others'.
    small: float
    #: 0: the drawn gains.  Otherwise the slow modes sit within this
    #: fraction of each other (where the structure can carry a spectrum).
    spread: float
    #: 1: the drawn mapping.  Otherwise one mapping row differences two
    #: entries of a source field this many times their difference.
    cancel: float
    #: The decade of a decimal change of units on one member.
    unit: int
    #: How far from the fixed point the step starts, relative to each field.
    offset: float
    #: Which member is the small one (the next is the one whose unit moves).
    member: int = 0

    @property
    def eps(self) -> float:
        return float(np.finfo(CELLS[self.cell].dtype).eps)


def values_of(case: Case) -> dict:
    """The gains, biases, mappings and start of *case*, rounded to its dtype."""
    cell = CELLS[case.cell]
    topo, cfgs = cell.topo, cell.cfgs
    members = topo.groups[0]
    rng = np.random.default_rng(case.seed)
    v = ct.draw_values(topo, rng, case.rho, nonnormal=case.nonnormal, dtype="float64",
                       group_cfgs=cfgs, mapping_kind=cell.mapping_kind)
    reshaped = False
    sizes = {topo.node(m).n for m in members}
    if case.spread > 0 and sizes == {3}:
        # A symmetric gain with eigenvalues within ``spread`` of each other
        # on the first member, the identity on the rest.
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        lam = 1.0 - case.spread * np.array([0.0, 5.0 / 11.0, 1.0])
        for k, m in enumerate(members):
            v["nodes"][m]["G"] = [Q @ np.diag(lam) @ Q.T if k == 0 else np.eye(3)
                                  for _ in v["nodes"][m]["G"]]
        reshaped = True
    differenced = None
    if case.cancel > 1:
        for i, e in enumerate(topo.edges):
            pattern = ct.mapping_pattern(topo, i, cell.mapping_kind) if e.mapped else None
            if e.mapped and topo.node(e.src).n >= 2 and (pattern is None or pattern[0, :2].all()):
                H = np.array(v["H"][i], np.float64)
                H[0, :] = 0.0
                H[0, :2] = (1.0, -1.0)
                v["H"][i] = H
                differenced = e.src
                reshaped = True
                break
    if reshaped:
        ct._rescale_group(topo, v, 0, case.rho, {}, "float64", cfgs)  # noqa: SLF001
    target = {m: rng.uniform(0.5, 2.0, topo.node(m).n) * rng.choice([-1.0, 1.0], topo.node(m).n)
              for m in members}
    small = members[case.member % len(members)]
    target[small] = target[small] * case.small
    if differenced is not None:
        base = case.cancel * rng.uniform(0.5, 2.0)
        target[differenced][:2] = (base, base + rng.uniform(0.5, 2.0))
    for m in members:
        scale = float(np.max(np.abs(target[m])))
        v["nodes"][m]["x0"] = target[m] + case.offset * scale * rng.normal(size=topo.node(m).n)
    ct.place_group_fixed_point(topo, v, 0, target, group_cfgs=cfgs)
    if case.unit:
        ct.rescale_node_units(topo, v, members[(case.member + 1) % len(members)],
                              10.0 ** case.unit)
    dt = np.dtype(cell.dtype)
    for m in members:
        v["nodes"][m] = {"G": [np.asarray(G, dt) for G in v["nodes"][m]["G"]],
                         "b": np.asarray(v["nodes"][m]["b"], dt),
                         "x0": np.asarray(v["nodes"][m]["x0"], dt)}
    v["H"] = {i: np.asarray(H, dt) for i, H in v["H"].items()}
    return v


# ---------------------------------------------------------------------------
# The exact answers
# ---------------------------------------------------------------------------


def _reported_floor(gm, key: str, meta: dict) -> float:
    """The floor ``coupling_diagnostics()`` used for group *key* (it reports
    the flag and the bound that rest on it, not the number): the step's own
    measurement per evaluation times the evaluation count where it took
    one, the public function of the returned state otherwise."""
    from maddening.core.coupling.acceleration import residual_precision_floor  # noqa: PLC0415

    evaluations, _declared, internal = gm._committed_floor_inputs[key]  # noqa: SLF001
    measured = float(meta.get("pass_evaluations", np.nan))
    if math.isfinite(measured):
        evaluations = max(evaluations, measured)
    unit = meta.get("reading_floor")
    if unit is not None and np.isfinite(unit):
        return float(unit * np.asarray(evaluations, unit.dtype))
    (group,) = gm._coupling_groups  # noqa: SLF001
    return float(residual_precision_floor(
        gm._state, sorted(group.nodes), group.convergence_norm, group.atol,  # noqa: SLF001
        group.rtol, list(internal), evaluations=evaluations,
        mappings=(gm._params or {}).get("mappings")))  # noqa: SLF001


def _cancellation(model: ct.LinearModel, pre: dict, state: dict) -> float:
    """How far the worst member "cancels inside itself": the sum of the
    magnitudes of the terms its update adds, over its field's largest
    entry (1: none).  The floor is ``eps * max|field|`` per entry and an
    update rounds at ``eps`` of its largest *term*, so this is the factor
    by which a node the floor's condition excludes (CPL-092) outruns it.
    The terms are the node's own: each port's delivered value times its
    gain, so a cancellation inside a mapping, upstream of the node, does
    not count."""
    topo = model.topo
    worst = 1.0
    for m in topo.groups[0]:
        nd = topo.node(m)
        v = model.values["nodes"][m]
        d = model.divider.get(m, 1)
        s_d = sum(abs(nd.alpha) ** j for j in range(d))
        terms = (abs(nd.alpha) ** d * np.abs(np.asarray(pre[m]["x"], np.float64))
                 + s_d * np.abs(np.asarray(v["b"], np.float64) + nd.beta * nd.timestep))
        for j in range(nd.ports):
            u = np.zeros(nd.n)
            for i, e in enumerate(topo.edges):
                if e.dst == m and e.port == j:
                    x = np.asarray(state[e.src]["x"], np.float64)
                    H = np.asarray(model.values["H"][i], np.float64) if e.mapped else np.eye(nd.n)
                    u = u + ct.TRANSFORM_FACTORS[e.transform] * (H @ x)
            terms = terms + s_d * (np.abs(np.asarray(v["G"][j], np.float64)) @ np.abs(u))
        size = float(np.max(np.abs(np.asarray(state[m]["x"], np.float64))))
        if size > 0:
            worst = max(worst, float(np.max(terms)) / size)
    return worst


def _pass_jacobian(model: ct.LinearModel) -> np.ndarray:
    """``dF/dx`` of the group's one-pass map, in float64."""
    L, U = model.group_pass(0)
    k = L.shape[0]
    return np.asarray(ct._solve(np.eye(k, dtype=ct.LD) - L, U), np.float64)  # noqa: SLF001


def _state_weights(model: ct.LinearModel, state: dict) -> np.ndarray:
    """``1 / max|field|`` per entry of the members' stacked state."""
    out = []
    for m in model.topo.groups[0]:
        x = np.abs(np.asarray(state[m]["x"], np.float64))
        out.append(np.full(x.shape, 1.0 / float(np.max(x)) if np.max(x) > 0 else 1.0))
    return np.concatenate(out)


def _radius(A: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.eigvals(A)))) if A.size else 0.0


def _radius_allowance(A: np.ndarray, rho: float, eps: float, seed: int) -> float:
    """How far ``rho_spectral`` may be from *rho* and still be "exact to
    float32": 1e-4 of it, plus four times the largest movement of the
    radius under eight seeded perturbations of the weighted Jacobian *A*
    of norm ``8 eps ||A||`` -- the backward error of an Arnoldi process in
    a float of that ``eps``, which a non-normal matrix turns into a large
    movement of its eigenvalues and a normal one does not."""
    rng = np.random.default_rng(seed)
    size = 8.0 * eps * float(np.linalg.norm(A, 2))
    moved = 0.0
    for _ in range(8):
        E = rng.normal(size=A.shape)
        E *= size / max(float(np.linalg.norm(E, 2)), 1e-300)
        moved = max(moved, abs(_radius(A + E) - rho))
    return 1e-4 * rho + 4.0 * moved


def _gradient_error(model: ct.LinearModel, pre: dict, state: dict) -> float:
    """The worst relative error of ``d x* / d c`` taken at the returned
    iterate, over every scalar gain and mapping weight ``c`` of the group.

    With the pass ``F(x) = (I - L)^{-1} (U x + c_g)`` and ``M = L + U``,
    the implicit derivative at an iterate ``x`` is ``(I - M)^{-1} (dL F(x)
    + dU x)`` for a constant that moves ``L`` and ``U`` by ``dL`` and
    ``dU`` (both linear in one gain or weight), and the fixed point's is
    the same at ``x*``.  In the norm the bound documents: the raw fields
    the group's norm reads, each over its magnitude at the returned state.
    """
    topo = model.topo
    members, off, k = model._group_layout(0)  # noqa: SLF001
    L, U = model.group_pass(0)
    cg_ = model.group_constant(0, pre, state)
    eye = np.eye(k, dtype=ct.LD)
    x = np.concatenate([np.asarray(np.asarray(state[m]["x"]), ct.LD) for m in members])
    F = ct._solve(eye - L, U @ x + cg_)  # noqa: SLF001
    xs = ct._solve(eye - (L + U), cg_)  # noqa: SLF001
    resolvent = np.linalg.inv(np.asarray(eye - (L + U), np.float64))
    S, w, _rtol, _rms = model.norm_parts(0, state, raw=True)
    WS = w[:, None] * S

    def moved(node=None, port=None, edge=None, entry=None):
        values = {"nodes": {m: dict(nv) for m, nv in model.values["nodes"].items()},
                  "H": dict(model.values["H"])}
        if edge is None:
            G = [np.array(g, np.float64) for g in values["nodes"][node]["G"]]
            G[port][entry] += 1.0
            values["nodes"][node]["G"] = G
        else:
            H = np.array(values["H"][edge], np.float64)
            H[entry] += 1.0
            values["H"][edge] = H
        other = ct.LinearModel(topo, values, dtype="float64", group_cfgs=model.cfgs, exact=True)
        L1, U1 = other.group_pass(0)
        return L1 - L, U1 - U

    constants = []
    for m in members:
        nd = topo.node(m)
        for j in range(nd.ports):
            constants += [dict(node=m, port=j, entry=(a, b))
                          for a in range(nd.n) for b in range(nd.n)]
    for i, e in enumerate(topo.edges):
        if e.mapped:
            H = np.asarray(model.values["H"][i])
            pattern = np.ones(H.shape, bool)
            constants += [dict(edge=i, entry=(a, b)) for a in range(H.shape[0])
                          for b in range(H.shape[1]) if pattern[a, b]]
    worst = 0.0
    for c in constants:
        dL, dU = moved(**c)
        t_k = resolvent @ np.asarray(dL @ F + dU @ x, np.float64)
        miss = resolvent @ np.asarray(dL @ (F - xs) + dU @ (x - xs), np.float64)
        size = float(np.linalg.norm(WS @ t_k))
        if size > 0:
            worst = max(worst, float(np.linalg.norm(WS @ miss)) / size)
    return worst


@functools.lru_cache(maxsize=4096)
def observe(case: Case) -> dict:
    """One step of *case* and the four scores of what it reported."""
    cell = CELLS[case.cell]
    topo = cell.topo
    values = values_of(case)
    with precision(cell.dtype == "float64"):
        built = _built(case.cell)
        (step,) = ct.run(built, values, 1)
        d = dict(step.reports[0])
        floor = _reported_floor(built.gm, topo.group_key(0), step.metas[0])
    model = ct.LinearModel(topo, values, dtype=cell.dtype, group_cfgs=cell.cfgs)
    out = dict(bound=0.0, radius=0.0, gradient=0.0, floor=0.0,
               spectral_usable=bool(d["spectral_usable"]),
               gradient_usable=bool(d["gradient_bound_usable"]),
               floor_reported=floor > 0 and math.isfinite(floor),
               report={k: d[k] for k in ("iterations", "converged", "residual", "rho_spectral",
                                         "spectral_error_bound", "spectral_usable",
                                         "gradient_relative_error_bound",
                                         "gradient_bound_usable", "precision_limited")})
    finite = all(np.all(np.isfinite(s["x"])) for s in step.state.values())
    if not finite or not math.isfinite(d["residual"]):
        return out
    residual = float(d["residual"])
    cancels = _cancellation(model, step.pre, step.state)
    # The floor a node that cancels inside itself is outside the promise by.
    allowed = ((residual + cancels * floor) / (residual + floor)
               if out["floor_reported"] else 1.0)
    out["report"].update(floor=floor, cancellation=cancels)

    if out["floor_reported"]:
        _dn, _b, detail = model.group_report_consistency(0, step.pre, step.state, residual)
        above = detail["residual_true"] - residual * (1.0 + 2.0 ** 8 * case.eps)
        out["floor"] = max(0.0, above) / (cancels * floor)
        out["report"]["residual_true"] = detail["residual_true"]

    J = _pass_jacobian(model)
    wv = _state_weights(model, step.state)
    A = (wv[:, None] * J) / wv[None, :]
    rho = _radius(J)
    rho_reported = float(d["rho_spectral"])
    out["report"]["rho_true"] = rho
    # The analysis runs in the group's dtype (at least float32).
    eps_analysis = case.eps
    if math.isfinite(rho_reported):
        if out["spectral_usable"]:
            out["radius"] = abs(rho_reported - rho) / max(
                _radius_allowance(A, rho, eps_analysis, case.seed), 1e-300)
        norm_A = float(np.linalg.norm(A, 2))
        if norm_A > 0 and np.linalg.norm(A @ A.T - A.T @ A, 2) <= 1e-9 * norm_A ** 2:
            # Normal in the norm's weights: an estimate "from below".
            out["radius"] = max(out["radius"], (rho_reported - rho) / max(
                _radius_allowance(A, rho, eps_analysis, case.seed), 1e-300))

    if out["spectral_usable"]:
        dist = model.returned_weight_distance(0, step.pre, step.state)
        bound = float(d["spectral_error_bound"]) * allowed
        out["bound"] = dist / bound if bound > 0 else (math.inf if dist > 0 else 0.0)
        out["report"]["distance"] = dist

    if out["gradient_usable"]:
        true = _gradient_error(model, step.pre, step.state)
        bound = float(d["gradient_relative_error_bound"]) * allowed
        out["gradient"] = true / bound if bound > 0 else (math.inf if true > 0 else 0.0)
        out["report"]["gradient_error"] = true
    return out


# ---------------------------------------------------------------------------
# The strategy
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Domain:
    """How wide the generator draws, in the directions past defects lay."""

    #: The smallest field beside what drives it.
    small: float = 1e-2
    #: Loop gains up to ``1 - 1e-7`` and slow modes within 1e-7 to 1e-1 of
    #: each other (otherwise 0.05 to 0.98 and the drawn gains).
    degenerate: bool = False
    #: The largest ratio of a field to the difference a mapping row takes of it.
    cancel: float = 1.0


#: What the claims are made on and this tree holds.
CLAIMED = Domain()
#: The directions the known defects lie in (the foot of the module).
SMALL_FIELD = Domain(small=1e-8)
DEGENERATE = Domain(degenerate=True)
CANCELLING = Domain(cancel=1e5)
WIDENED = Domain(small=1e-8, degenerate=True, cancel=1e5)


def _decades(low: float, high: float):
    return st.floats(low, high).map(lambda x: 10.0 ** x)


def cases(cells=ALL_CELLS, domain: Domain = CLAIMED):
    """Draw a :class:`Case` on one of *cells* within *domain*."""
    gain = st.floats(0.05, 0.98)
    spread = st.just(0.0)
    if domain.degenerate:
        gain = st.one_of(gain, _decades(-7.0, -2.0).map(lambda g: 1.0 - g))
        spread = st.one_of(st.just(0.0), _decades(-7.0, -1.0))
    small = st.just(1.0) if domain.small >= 1.0 else st.one_of(
        st.just(1.0), _decades(math.log10(domain.small), 0.0))
    cancel = st.just(1.0) if domain.cancel <= 1.0 else st.one_of(
        st.just(1.0), _decades(0.0, math.log10(domain.cancel)))
    return st.builds(
        Case, cell=st.sampled_from(tuple(cells)), seed=st.integers(0, 2 ** 16),
        rho=gain, nonnormal=st.booleans(), small=small, spread=spread, cancel=cancel,
        unit=st.sampled_from((0, 0, -6, -3, -1, 1, 3, 6)),
        offset=st.one_of(st.just(0.0), _decades(-9.0, 0.0)), member=st.integers(0, 7))


#: The largest score each claim allows.  A bound holds exactly in exact
#: arithmetic; the report computes it in the analysis dtype (float32 for a
#: float32 group) from a few dozen operations, so 2**10 of that dtype's
#: ``eps`` is allowed (1.2e-4), for either dtype.  The radius and the floor
#: carry their allowances in the score.
THRESHOLD = {"bound": 1.0 + 2.0 ** 10 * float(np.finfo(np.float32).eps),
             "radius": 1.0,
             "gradient": 1.0 + 2.0 ** 10 * float(np.finfo(np.float32).eps),
             "floor": 1.0}
FLAG = {"bound": "spectral_usable", "radius": "spectral_usable",
        "gradient": "gradient_usable", "floor": "floor_reported"}
#: The least fraction of a slow hunt's examples whose flag must have been
#: set.  A search that only ever sees unusable numbers proves nothing; the
#: generator draws caps of five passes, gains near one and starts at the
#: fixed point, all of which the flags rightly refuse, and measured 0.5
#: to 0.7 usable over the claimed domain, so a fifth leaves room for a
#: hunt that climbs towards the refused corner and fails one that lives there.
USABLE_FLOOR = 0.2


def scorer(name: str):
    def score(case: Case):
        seen = observe(case)
        return seen[name], seen["report"]
    return score


def usable_fraction(name: str, strategy_cases) -> float:
    seen = [observe(c)[FLAG[name]] for c in strategy_cases]
    return sum(seen) / max(len(seen), 1)


def search(name: str, *, cells=PER_PUSH_CELLS, domain: Domain = CLAIMED, profile=PER_PUSH,
           fail: bool = True):
    """Run search *name*; returns ``(report, usable fraction)``."""
    drawn = []

    def score(case: Case):
        drawn.append(case)
        return scorer(name)(case)

    report = targeted_search(cases(cells, domain), score, THRESHOLD[name], profile=profile,
                             label=name, fail=fail)
    return report, usable_fraction(name, drawn)


# ---------------------------------------------------------------------------
# The four searches
# ---------------------------------------------------------------------------

SEARCHES = ("bound", "radius", "gradient", "floor")


def test_a_usable_error_bound_is_never_below_the_distance_per_push():
    _report, usable = search("bound")
    assert usable > 0, "no example had a usable bound"


def test_a_settled_spectral_radius_is_the_radius_per_push():
    _report, usable = search("radius")
    assert usable > 0, "no example had a settled spectrum"


def test_a_usable_gradient_bound_is_never_below_the_error_per_push():
    _report, usable = search("gradient")
    assert usable > 0, "no example had a usable gradient bound"


def test_the_floor_covers_what_the_reported_residual_misses_per_push():
    _report, usable = search("floor")
    assert usable > 0, "no example reported a floor"


_WITNESS = {
    "bound": "test_a_usable_error_bound_is_never_below_the_distance_per_push",
    "radius": "test_a_settled_spectral_radius_is_the_radius_per_push",
    "gradient": "test_a_usable_gradient_bound_is_never_below_the_error_per_push",
    "floor": "test_the_floor_covers_what_the_reported_residual_misses_per_push",
}


# Slow: 800 random examples over every cell, each cell a compile.
# Per push: tests/property/test_coupling_targeted_search.py::test_a_usable_error_bound_is_never_below_the_distance_per_push
@pytest.mark.slow
@pytest.mark.parametrize("name", SEARCHES)
def test_the_hunt_finds_no_number_on_the_wrong_side(name):
    report, usable = search(name, cells=ALL_CELLS, profile=SLOW)
    print(f"{name}: worst {report}; usable fraction {usable:.2f}")
    assert usable >= USABLE_FLOOR, (
        f"{name}: only {usable:.2f} of the examples had the flag set (floor {USABLE_FLOOR})")
