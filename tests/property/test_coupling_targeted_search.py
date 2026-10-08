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
2. *spectral radius* (CPL-087), two statements.  ``"radius"``: for a pass
   whose Jacobian has rank at most eight ``rho_spectral`` is the radius of
   a matrix within the analysis dtype's rounding of the weighted Jacobian
   -- ``|rho_spectral - rho|`` over 1e-4 of the radius plus its movement
   under perturbations of that size (:func:`_radius_allowance`: what a
   backward-stable computation delivers, which is what the claim says of
   a settled spectrum), and past rank eight, where the claim says "an
   estimate", plus the 5% of ``1 - rho`` the flag tests; and, where the
   weighted Jacobian is normal, how far the estimate is *above* the
   radius ("from below for a normal dF/dx"), whatever the flag.
   ``"radius_strict"``: the statement a user can check, with no reference
   to the Jacobian -- where ``spectral_usable`` and no more than eight
   scalars cross the group's edges, ``rho_spectral`` is within
   ``SPECTRAL_SETTLED_FRACTION`` of ``1 - rho_spectral`` of the radius
   (the margin the flag holds the Arnoldi residual, a discarded
   direction and the radius's measured sensitivity to rounding to).  No row promises
   ``rho_spectral`` as an upper estimate, so none is scored;
3. *gradient bound* (CPL-093): the true relative error of the implicit
   derivative taken at the returned iterate against the same dense solve
   at the fixed point, the worst over every scalar gain and every mapping
   weight the graph holds (:func:`gradient_constants`: every entry of a
   matrix held dense, the pattern's entries of a sparse mapping), over
   ``gradient_relative_error_bound``, where ``gradient_bound_usable``;
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

import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import strategies as st

from maddening.core.coupling.acceleration import (
    SPECTRAL_KRYLOV_STEPS,
    SPECTRAL_SETTLED_FRACTION,
)
from tests.property import coupled_graphs as cg
from tests.property import coupled_topologies as ct
from tests.property.sysid_transform_grid import precision
from tests.property.targeted_search import PER_PUSH, SLOW, targeted_search

#: The per-push profile: 150 examples, not the house floor of 20.  An
#: example costs about 20 ms once its cell is compiled (the cells are the
#: seconds), and at 20 examples a search did not find a resolvent factor
#: halved; at 150 it does.
EVERY_PUSH = dataclasses.replace(PER_PUSH, max_examples=150)

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
#: The rows above that :func:`_cells` rotates over, FROZEN at the seven the
#: cells were laid out with.  A row added to :data:`KNOBS` (anywhere) must
#: not move a cell a pin names by its index: a new configuration gets cells
#: of its own, appended after every cell there is
#: (``test_coupling_search_cells_are_pinned.py`` holds each cell's
#: configuration per push).
ROTATED_KNOBS = (0, 1, 2, 3, 4, 5, 6)
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
                knob = ROTATED_KNOBS[(s + 3 * t + 2 * j) % len(ROTATED_KNOBS)]
                out.append(Cell(name, dtype, knob, CAPS[(s + t + j) % 2], kind))
    return tuple(dict.fromkeys(out))


#: The first cells are the per-push ones (:data:`PER_PUSH_CELLS`), chosen
#: for the shapes past defects lived on: the ring of three scalars under
#: Jacobi in float64 (a small field), the pair with a prescribed spectrum
#: stopped early in float64 (near-degenerate modes), the mapped ring under
#: Gauss-Seidel in float32 (a cancelling mapping row at the floor), the
#: hub under the interface norm in float32 (fan-out) and the pair of
#: twelve scalars under Jacobi (a spectrum eight Krylov steps do not
#: resolve: the one per-push cell on which a flag says False -- the
#: gradient's, always, and the spectrum's where the gain is large --
#: without which a flag forced True changes nothing a search sees).
_FIRST = (
    Cell("ring-3", "float64", 1, 120),
    Cell("pair-3", "float64", 0, 5),
    Cell("mapped", "float32", 0, 120),
    Cell("hub", "float32", 6, 5),
    Cell("pair-6", "float32", 1, 5),
)
#: A cell a pinned finding lives on (compiled per push for that pin only).
_PINNED = (Cell("ring-3-multirate", "float32", 6, 5),)
CELLS = _FIRST + _PINNED + tuple(c for c in _cells() if c not in _FIRST + _PINNED)
PER_PUSH_CELLS = tuple(range(len(_FIRST)))
ALL_CELLS = tuple(range(len(CELLS)))
#: The slow hunt takes the cells a block at a time, so that a block's
#: compiled graphs are all held while it is searched and each cell
#: compiles once: every seventh cell, which spreads the structures, the
#: dtypes and the configurations over the blocks.
BLOCKS = tuple(ALL_CELLS[k::7] for k in range(7))


@functools.lru_cache(maxsize=max(len(b) for b in BLOCKS) + 1)
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


def values_of(case: Case, cell: "Cell | None" = None) -> dict:
    """The gains, biases, mappings and start of *case*, rounded to its dtype.

    *cell*: the cell to draw on in place of ``CELLS[case.cell]`` (the
    nonlinear cells of ``test_coupling_nonlinear_search.py`` draw the same
    numbers on cells of their own)."""
    cell = CELLS[case.cell] if cell is None else cell
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


def _reported_floor(gm, key: str, meta: dict, report: dict) -> float:
    """The floor ``coupling_diagnostics()`` used for group *key*.

    The report carries the flag and the bound that rest on the floor, not
    the number, so it is taken twice.  By the documented rule (CPL-100,
    CPL-102): the step's own measurement per evaluation times the
    evaluation count where it took one, the public function of the
    returned state otherwise.  And out of the reported bound, which is
    ``(residual + floor)`` times a factor the slots give: where the
    residual does not swamp it (at most four floors), the floor the bound
    was made with.  The smaller of the two is what the report rests on.
    """
    from maddening.core.coupling.acceleration import (  # noqa: PLC0415
        residual_precision_floor,
        spectral_error_bound,
    )

    evaluations, _declared, internal = gm._committed_floor_inputs[key]  # noqa: SLF001
    measured = float(meta.get("pass_evaluations", np.nan))
    if math.isfinite(measured):
        evaluations = max(evaluations, measured)
    unit = meta.get("reading_floor")
    if unit is not None and np.isfinite(unit):
        floor = float(unit * np.asarray(evaluations, unit.dtype))
    else:
        (group,) = gm._coupling_groups  # noqa: SLF001
        floor = float(residual_precision_floor(
            gm._state, sorted(group.nodes), group.convergence_norm, group.atol,  # noqa: SLF001
            group.rtol, list(internal), evaluations=evaluations,
            mappings=(gm._params or {}).get("mappings")))  # noqa: SLF001
    slots = [meta.get(k) for k in ("rho_spectral", "spectral_residual", "spectral_amplification")]
    bound, residual = float(report["spectral_error_bound"]), float(report["residual"])
    if all(s is not None for s in slots) and math.isfinite(bound) and residual <= 4.0 * floor:
        factor = float(spectral_error_bound(1.0, *slots))
        if math.isfinite(factor) and factor > 0:
            floor = min(floor, max(bound / factor - residual, 0.0) * (1.0 + 1e-3))
    return floor


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
    float32": 1e-4 of it, plus sixteen times the largest movement of the
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
    return 1e-4 * rho + 16.0 * moved


def gradient_constants(topo: ct.Topology, mapping_kind: str) -> list:
    """The scalar constants the gradient score is taken over: what a graph
    of *topo* built under *mapping_kind* holds as parameters of its group.

    Every entry of every gain of every member, and of each mapped edge the
    entries :func:`~tests.property.coupled_topologies.parameter_entries`
    names: every entry of a matrix held dense, the pattern's entries of a
    sparse one.  An entry outside a sparse pattern is structure -- the
    graph has no weight there, no user can ask for a derivative with
    respect to one and ``gradient_relative_error_bound`` says nothing of
    it -- so it is no constant; an entry of the pattern is one whatever
    its drawn weight.  Each as the keyword arguments that name it:
    ``node, port, entry`` or ``edge, entry``.
    """
    constants = []
    for m in topo.groups[0]:
        nd = topo.node(m)
        assert not nd.three_arg, (
            f"{m} reads its gains when the step is traced: the graph holds no parameter there")
        for j in range(nd.ports):
            constants += [dict(node=m, port=j, entry=(a, b))
                          for a in range(nd.n) for b in range(nd.n)]
    for i, e in enumerate(topo.edges):
        if e.mapped:
            held = ct.parameter_entries(topo, i, mapping_kind)
            constants += [dict(edge=i, entry=(int(a), int(b))) for a, b in zip(*np.nonzero(held))]
    return constants


def gradient_errors(model: ct.LinearModel, pre: dict, state: dict, mapping_kind: str) -> list:
    """``(relative error, constant)`` of ``d x* / d c`` taken at the
    returned iterate, for every constant ``c`` of
    :func:`gradient_constants` the fixed point depends on.

    *mapping_kind*: how the graph that *model* describes holds its mapped
    edges (the cell's: a model is the dense one whatever the kind, and
    cannot say).

    With the pass ``F(x) = (I - L)^{-1} (U x + c_g)`` and ``M = L + U``,
    the implicit derivative at an iterate ``x`` is ``(I - M)^{-1} (dL F(x)
    + dU x)`` for a constant that moves ``L`` and ``U`` by ``dL`` and
    ``dU`` (both linear in one gain or weight), and the fixed point's is
    the same at ``x*``.  In the norm the bound documents: the raw fields
    the group's norm reads, each over its magnitude at the returned state.
    """
    topo = model.topo
    assert model.geometry is None, "a geometry-dependent mapping holds no weights"
    for i, e in enumerate(topo.edges):
        if e.mapped:
            # The model is the graph's only where its matrix is zero
            # wherever the graph holds nothing.
            outside = ~ct.parameter_entries(topo, i, mapping_kind)
            assert not np.any(np.asarray(model.values["H"][i])[outside]), (
                f"edge {i}: the model's matrix is not one a {mapping_kind!r} edge holds")
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

    errors = []
    for c in gradient_constants(topo, mapping_kind):
        dL, dU = moved(**c)
        t_k = resolvent @ np.asarray(dL @ F + dU @ x, np.float64)
        miss = resolvent @ np.asarray(dL @ (F - xs) + dU @ (x - xs), np.float64)
        size = float(np.linalg.norm(WS @ t_k))
        if size > 0:
            errors.append((float(np.linalg.norm(WS @ miss)) / size, c))
    return errors


def _gradient_error(model: ct.LinearModel, pre: dict, state: dict, mapping_kind: str) -> float:
    """The worst of :func:`gradient_errors`: over every scalar gain and
    every mapping weight the graph holds."""
    return max((error for error, _c in gradient_errors(model, pre, state, mapping_kind)),
               default=0.0)


def radius_scores(out: dict, J: np.ndarray, wv: np.ndarray, rho_reported: float, eps: float,
                  seed: int) -> None:
    """Write the two radius scores of a reported ``rho_spectral`` into
    *out* (``"radius"`` and ``"radius_strict"``; the module docstring says
    what each is), given the pass's Jacobian *J* at the state the report
    describes and that state's weights *wv*.  *eps*: the analysis dtype's
    (the group's, at least float32)."""
    A = (wv[:, None] * J) / wv[None, :]
    rho = _radius(J)
    if math.isfinite(rho_reported):
        allowance = max(_radius_allowance(A, rho, eps, seed), 1e-300)
        norm_A = float(np.linalg.norm(A, 2))
        off = abs(rho_reported - rho) if out["spectral_usable"] else 0.0
        resolved = np.linalg.matrix_rank(J) <= SPECTRAL_KRYLOV_STEPS
        if not resolved:
            # "An estimate otherwise, which spectral_usable reports":
            # past eight independent scalars nothing is called exact.  A
            # settled estimate is read as good to the margin the flag
            # tests it by, 5% of ``1 - rho``.
            allowance += SPECTRAL_SETTLED_FRACTION * max(1.0 - rho, 0.0)
            out["report"]["rank"] = int(np.linalg.matrix_rank(J))
        if norm_A > 0 and np.linalg.norm(A @ A.T - A.T @ A, 2) <= 1e-9 * norm_A ** 2:
            # Normal in the norm's weights: an estimate "from below",
            # whatever the flag.
            off = max(off, rho_reported - rho)
        # The statement a user can check: within the flag's own margin of
        # the radius, wherever the flag is set and the group has no more
        # scalars than the Krylov steps resolve.
        if out["spectral_usable"] and resolved:
            out["radius_strict"] = abs(rho_reported - rho) / max(
                SPECTRAL_SETTLED_FRACTION * (1.0 - rho_reported), 1e-300)
        out["radius"] = off / allowance
        out["report"]["jacobian_norm"] = norm_A


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
        floor = _reported_floor(built.gm, topo.group_key(0), step.metas[0], d)
    model = ct.LinearModel(topo, values, dtype=cell.dtype, group_cfgs=cell.cfgs)
    out = dict(bound=0.0, radius=0.0, radius_strict=0.0, gradient=0.0, floor=0.0,
               spectral_usable=bool(d["spectral_usable"]),
               gradient_usable=bool(d["gradient_bound_usable"]),
               floor_reported=math.isfinite(floor),
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
               if out["floor_reported"] and residual + floor > 0 else 1.0)
    out["report"].update(floor=floor, cancellation=cancels)

    if out["floor_reported"]:
        _dn, _b, detail = model.group_report_consistency(0, step.pre, step.state, residual)
        above = detail["residual_true"] - residual * (1.0 + 2.0 ** 8 * case.eps)
        out["floor"] = max(0.0, above) / max(cancels * floor, 1e-300)
        out["report"]["residual_true"] = detail["residual_true"]

    J = _pass_jacobian(model)
    wv = _state_weights(model, step.state)
    out["report"]["rho_true"] = _radius(J)
    radius_scores(out, J, wv, float(d["rho_spectral"]), case.eps, case.seed)

    if out["spectral_usable"]:
        dist = model.returned_weight_distance(0, step.pre, step.state)
        bound = float(d["spectral_error_bound"]) * allowed
        # A usable bound that is not a number bounds nothing.
        out["bound"] = (math.inf if math.isnan(bound) else
                        dist / bound if bound > 0 else (math.inf if dist > 0 else 0.0))
        out["report"]["distance"] = dist

    if out["gradient_usable"]:
        true, constant = max(
            gradient_errors(model, step.pre, step.state, cell.mapping_kind),
            key=lambda error: error[0], default=(0.0, None))
        # The bound is on the error of stopping early.  The derivative it
        # is relative to is itself a float solve, good to no better than a
        # few ``eps``: an error below 64 of them (the allowance of the
        # domain check of CPL-093) is not one a gradient in this dtype has.
        bound = float(d["gradient_relative_error_bound"]) * allowed + 64.0 * case.eps
        out["gradient"] = math.inf if math.isnan(bound) else true / bound
        out["report"].update(gradient_error=true, gradient_constant=constant,
                             gradient_allowed=bound)
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
             "radius_strict": 1.0,
             "gradient": 1.0 + 2.0 ** 10 * float(np.finfo(np.float32).eps),
             "floor": 1.0}
FLAG = {"bound": "spectral_usable", "radius": "spectral_usable",
        "radius_strict": "spectral_usable",
        "gradient": "gradient_usable", "floor": "floor_reported"}
#: The least fraction of a hunt's examples whose flag must have been set.
#: A search that only ever sees unusable numbers proves nothing.  Measured
#: on this tree: 1.00 in every block of two hunts over the claimed domain
#: (every member declares its evaluation count and no group has more than
#: eight scalars, so the flags have nothing to refuse there) and 0.85 to
#: 1.00 over the widened one, where a field at its driver's rounding reads
#: a radius above one.  A half is far enough below both that a fix which
#: makes the flags refuse more does not trip it, and a change that makes
#: them refuse most of what is drawn does.
USABLE_FLOOR = 0.5


SEARCHES = ("bound", "radius", "gradient", "floor")


def scorer(name: str):
    def score(case: Case):
        seen = observe(case)
        return seen[name], seen["report"]
    return score


def usable_fraction(name: str, strategy_cases) -> float:
    seen = [observe(c)[FLAG[name]] for c in strategy_cases]
    return sum(seen) / max(len(seen), 1)


def search(name: str, *, cells=PER_PUSH_CELLS, domain: Domain = CLAIMED, profile=None,
           fail: bool = True):
    """Run search *name*; returns ``(report, usable fraction)``.

    The default profile is the per-push one under a seed of the search's
    own: the same draws on every run, and not the same draws for the four
    searches (derandomised, they would all score one set of examples)."""
    if profile is None:
        profile = EVERY_PUSH.seeded(sorted(THRESHOLD).index(name))
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


# Slow: 800 random examples a search over every cell (seven blocks of
# nine or ten cells, 115 examples each),
# each cell a compile; the blocks outermost, so the four searches share a
# block's compiled graphs.
# Per push: tests/property/test_coupling_targeted_search.py::test_a_usable_error_bound_is_never_below_the_distance_per_push
@pytest.mark.slow
@pytest.mark.parametrize("block,name", [(b, n) for b in range(len(BLOCKS)) for n in SEARCHES])
def test_the_hunt_finds_no_number_on_the_wrong_side(block, name):
    profile = dataclasses.replace(SLOW, max_examples=SLOW.max_examples // len(BLOCKS) + 1)
    report, usable = search(name, cells=BLOCKS[block], profile=profile)
    print(f"{name}, block {block}: worst {report}; usable fraction {usable:.2f}")
    assert usable >= USABLE_FLOOR, (
        f"{name}, block {block}: only {usable:.2f} of the examples had the flag set "
        f"(floor {USABLE_FLOOR})")


# ---------------------------------------------------------------------------
# The seeds, and what the search found
# ---------------------------------------------------------------------------

#: The directions the generator was widened in, one explicit example each,
#: inside the claimed domain: every score holds on every one.
SEEDS = {
    "twelve-scalars-the-flags-refuse": Case(4, 9, 0.8, False, 1.0, 0.0, 1.0, 0, 1e-2, 0),
    "a-field-a-hundredth-of-its-driver": Case(0, 3, 0.6, False, 1e-2, 0.0, 1.0, 0, 1e-3, 0),
    "gain-0.98-in-other-units": Case(0, 4, 0.98, True, 1.0, 0.0, 1.0, 3, 1e-6, 1),
    "modes-within-a-thousandth-stopped-early": Case(1, 5, 0.9, False, 1.0, 1e-3, 1.0, 0, 1e-2, 0),
    "a-mapping-row-differencing-at-the-floor": Case(2, 6, 0.5, False, 1.0, 0.0, 30.0, -3, 0.0, 2),
    "a-hub-read-four-times-under-the-interface-norm": Case(3, 7, 0.7, True, 0.1, 0.0, 1.0, 6,
                                                         1e-1, 3),
    "a-start-at-the-fixed-point": Case(3, 8, 0.05, False, 1.0, 0.0, 1.0, -6, 0.0, 0),
}


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_every_score_holds_on_the_seed_shapes(seed):
    seen = observe(SEEDS[seed])
    over = {name: seen[name] for name in SEARCHES if seen[name] > THRESHOLD[name]}
    assert not over, f"{seed}: {over} ({seen['report']})"


def _same_report(a: dict, b: dict) -> bool:
    """Equal key for key, a NaN (a number not computed) equal to itself."""
    def norm(d):
        return {k: ("nan" if isinstance(v, float) and v != v else v) for k, v in d.items()}
    return norm(a) == norm(b)


@pytest.mark.parametrize("seed", sorted(SEEDS))
def test_a_report_on_a_seed_shape_does_not_move_when_the_state_is_written_afterwards(seed):
    """Step, write the state, read the report: it is still the step's.

    The scores above are of the report read right after the step.  The
    floor under the bound is measured on the returned state at report
    time, so a ``set_node_state`` in between used to give the same step
    another bound (to 0.0 with its flag set, where every member was
    written to zero: a field at exactly zero leaves the norm).
    """
    case = SEEDS[seed]
    cell = CELLS[case.cell]
    with precision(cell.dtype == "float64"):
        built = _built(case.cell)
        (step,) = ct.run(built, values_of(case), 1)
        gm, key = built.gm, cell.topo.group_key(0)
        first = dict(gm.coupling_diagnostics()[key])
        assert _same_report(first, step.reports[0])
        members = key.split("+")
        returned = {name: gm.get_node_state(name) for name in members}
        edits = {
            "one member to zero": {members[0]: 0.0},
            "every member to zero": dict.fromkeys(members, 0.0),
            "every member a thousand times larger": dict.fromkeys(members, 1e3),
            "put back": dict.fromkeys(members, 1.0),
        }
        for label, factors in edits.items():
            for name, factor in factors.items():
                gm.set_node_state(name, {f: v * jnp.asarray(factor, v.dtype)
                                         for f, v in returned[name].items()})
            assert _same_report(dict(gm.coupling_diagnostics()[key]), first), (seed, label)


def _known(case: Case, score: str, reason: str):
    return pytest.param(case, score, marks=pytest.mark.xfail(strict=True, reason=reason))


#: What the search reached and a fix closed, each the shrunk example of the
#: hunt that found it: ``(case, score, flag)``.  The score holds, and the
#: flag reads what it should -- set where the number is now right (a fix
#: that withdrew the flag everywhere would pass a score alone), withdrawn
#: where the dtype cannot determine it.
FIXED = {
    # A breakdown test at 1e-5 of the product in every dtype (audit round 7,
    # F2): rho_spectral 0.379 for an exact 0.5, a field 1e-4 of its driver.
    "F2-a-small-field": (Case(0, 0, 0.5, False, 1e-4, 0.0, 1.0, 0, 0.0, 0), "radius", True),
    # The same cause with no small field: non-normal gains, the weighted
    # Jacobian's norm 56 beside a radius of 0.084, read 0.9% low in float64.
    "F2-non-normal-gains": (
        Case(1, 46570, 0.2897030151530067, True, 1.0, 0.0, 1.0, 1, 0.03712724112413696, 2),
        "radius", True),
    # The same on jaxlib 0.11.2's per-push draw: 0.0202 for 0.0156.
    "F2-the-draw-jaxlib-0.11.2-reaches": (
        Case(1, 2200, 0.125, True, 0.1, 0.0, 1.0, 0, 0.0, 0), "radius", True),
    # The float32 reading of F2: a field 1e-4 of its driver on the fan-out
    # hub, 0.574 for 0.092.  Float32 rounding moves that radius by more
    # than the flag's margin: the measured sensitivity withdraws the flag.
    "F2-a-small-field-in-float32-is-not-settled": (
        Case(3, 5, 0.3, False, 1e-4, 0.0, 1.0, 0, 0.0, 0), "radius_strict", False),
    # spectral_error_bound 0.61x the distance: three modes within 1e-5 of
    # each other at a gain of 1 - 1e-6, float64 (audit round 7, F3).
    "F3-near-degenerate-slow-modes": (
        Case(1, 98, 0.999999, True, 1.0, 1e-5, 1.0, 0, 1.0, 0), "bound", True),
    # Twelve scalars, non-normal, float64: 0.941 for 0.735 with an Arnoldi
    # residual under 5% of the gap.  The ninth vector moves the radius.
    "a-non-normal-spectrum-past-eight-is-not-settled": (
        Case(45, 434, 0.7345602069808425, True, 0.1, 0.0, 1.0, 0, 3.4603590782731456e-06, 2),
        "radius", False),
    # Twelve float32 scalars under Gauss-Seidel (rank six), one field 0.014
    # of its driver: 0.576 for 0.340.  The space had been continued from
    # the residual, so a ninth column says nothing: the rounding probe does.
    "a-space-continued-from-the-residual-is-probed-for-rounding": (
        Case(41, 614, 0.5827283898433404, True, 0.014131225734819956, 0.0, 1.0, 0,
             2.944906874682467e-07, 0), "radius_strict", False),
    # A float32 ring of five, one field a fiftieth of its driver: 0.0512
    # for 0.0500, the repeated squaring's own rounding (eigvals: 0.049998).
    "a-float32-ring-of-five-the-squaring-misread": (
        Case(23, 0, 0.05, True, 0.02025875358340762, 0.0, 1.0, 6, 0.0, 0), "radius", True),
    # A float32 Jacobi ring of mapped edges stopped at its float floor:
    # gradient_relative_error_bound 5.1e-6 for a true 3.0e-5 (found by the
    # floor search, twelve examples of one hunt).  The bound took the
    # change of the adjoint's right-hand side along one fixed-seed
    # direction where the residual, at its floor, gives none; it now takes
    # the operator norm over the directions the Jacobian reads
    # (MADD-ANO-226): 1.2e-4, usable.
    # A float32 multi-rate ring whose small field is 1e-6 of its driver
    # read rho_spectral 1e-12 for 1.25e-4.  The sub-cycled member's
    # boundary interpolation ``a + alpha * (b - a)`` rounded the tangent of
    # the new value at one float32 eps of the old one's, so the loop left
    # the Jacobian-vector products.  Its tangent is now ``(1 - alpha)
    # a_dot + alpha b_dot`` (the value is unchanged): usable and right
    # (MADD-ANO-222).
    "MADD-ANO-222-a-loop-below-a-fields-rounding-is-in-the-products": (
        Case(len(_FIRST), 5, 0.05, False, 1e-6, 0.0, 1.0, 0, 0.0, 5), "radius", True),
    # Twelve float32 scalars under Jacobi, non-normal gains: 0.273 for
    # 0.219 with the Arnoldi residual and the ninth vector's movement of
    # the radius both inside the margin.  A space still growing at the cap
    # is not invariant and its Ritz value is within no computed distance
    # of the radius: never settled (MADD-ANO-225).
    "MADD-ANO-225-a-space-still-growing-at-the-cap-is-not-settled": (
        Case(4, 0, 0.21902815820121518, True, 1.0, 0.0, 1.0, -6, 1e-09, 0), "radius", False),
    # Twelve float32 scalars under Gauss-Seidel (rank six), non-normal, a
    # field 0.015 of its driver: 0.250 for 0.206 and 0.107 for 0.056 with
    # eight sampled perturbations moving the radius by a third to a half
    # of the margin.  The certificate over every perturbation of the
    # measured size refuses both (MADD-ANO-229).
    "MADD-ANO-229-a-radius-rounding-can-move-is-not-settled-i": (
        Case(41, 29814, 0.453609867338665, True, 0.01539605614570328, 0.0, 1.0, 1, 0.0, 0),
        "radius_strict", False),
    "MADD-ANO-229-a-radius-rounding-can-move-is-not-settled-ii": (
        Case(41, 51310, 0.23676567122390113, True, 0.016305935878241445, 0.0, 1.0, -1,
             3.188721180654517e-07, 6), "radius_strict", False),
    "MADD-ANO-226-the-gradient-bound-at-the-float-floor": (
        Case(53, 6984, 0.05, False, 1.0, 0.0, 1.0, 6, 1.0, 3), "gradient", True),
}


@pytest.mark.parametrize("name", sorted(FIXED))
def test_a_defect_the_search_reached_stays_fixed(name):
    case, score, flag = FIXED[name]
    seen = observe(case)
    assert seen[FLAG[score]] is flag, f"the flag reads {seen[FLAG[score]]}: {seen['report']}"
    assert all(seen[s] <= THRESHOLD[s] for s in THRESHOLD), (
        f"{ {s: seen[s] for s in THRESHOLD if seen[s] > THRESHOLD[s]} }: {seen['report']}")


#: The known defects the search reached on this tree, each the shrunk
#: example of a hunt over the widened domain that names it.  Strict: the
#: fix turns each green, and its domain then joins :data:`CLAIMED`.
KNOWN = {
    # A mapping row [1, -1] on a field 1e4 times the difference, read in
    # the same Gauss-Seidel pass, stalled in float32: the exact residual is
    # 16 floors and the bound 0.06x the distance, spectral_usable (found
    # by the floor search in 730 examples on one seed of three).
    "MADD-ANO-212-the-floor": _known(
        Case(2, 6985, 0.25, True, 1.0, 0.0, 1e4, 0, 0.0, 0), "floor",
        "MADD-ANO-212: the counted floor does not see a same-pass read that differences"),
    "MADD-ANO-212-the-bound": _known(
        Case(2, 6985, 0.25, True, 1.0, 0.0, 1e4, 0, 0.0, 0), "bound",
        "MADD-ANO-212: the counted floor does not see a same-pass read that differences"),
}


@pytest.mark.parametrize("case,score", list(KNOWN.values()), ids=list(KNOWN))
def test_a_known_defect_the_search_reached_is_fixed(case, score):
    seen = observe(case)
    assert seen[FLAG[score]], f"the flag is no longer set: {seen['report']}"
    assert seen[score] <= THRESHOLD[score], (
        f"{score} is {seen[score]!r}, over {THRESHOLD[score]!r}: {seen['report']}")


# =============================================================================
# THE INTERFACE NORM'S SIDE OF A MAPPED EDGE  (a block of its own: everything
# below is the fifth score and its cells; nothing above reads it)
# =============================================================================
#
# ``convergence_norm="interface"`` reads a mapped internal edge on one side
# of its mapping.  The decision of 2026-10-07 is the *compact* side (a
# target larger than its source is read at the source; a tie and a smaller
# target as delivered); this tree reads what every edge delivers
# (``ct.INTERFACE_SIDE``).  A wrong side is a criterion that is diluted by
# the large field's entry count, or measured on a scale the consumer never
# sees, and a fifth score finds either:
#
# 5. *side* (``"side"``): where the group reports ``converged`` under the
#    interface norm, the true distance to the fixed point in the compact
#    readings over ``K`` tolerances, ``K`` the constant the loop's own
#    operator gives a residual at its threshold
#    (:meth:`~tests.property.coupled_topologies.LinearModel.interface_claim`).
#
# **The generator** draws a pair (or the hub) of
# :func:`~tests.property.coupled_topologies.side_topologies`, whose two
# mapped internal edges have size ratios from 1/300 to 300 and a tie, held
# dense, dense with a local (interpolation) matrix, or sparse; the loop
# gain; and how far from the fixed point the step starts.
#
# **Where it holds today** is where the two rules read the same thing: the
# tie.  Every other cell has an edge onto a larger target and waits for the
# rule (:data:`SIDE_AWAITING`): the per-push search runs the tie -- and one
# cell of ratio 300 from the day ``ct.INTERFACE_SIDE`` says ``"compact"``
# -- and what the hunt reaches on the rest today is pinned below, strict,
# beside the dilution pins of ``test_coupling_interface_side.py``.
#
# The same structures at small sizes join :data:`CELLS` (after every cell
# the searches above index, which keep their numbers), so the four scores
# above are taken on mappings between sizes too: the spectral analysis and
# the floor read the edges by an enumeration of their own, and a bound
# taken on another reading than the criterion's is a wrong bound.

SIDE_STRUCTURES = ct.side_topologies()
STRUCTURES = {**STRUCTURES, **SIDE_STRUCTURES}
#: The interface norm under Jacobi, which no row above has.
KNOBS = KNOBS + (
    dict(acceleration="none", iteration_mode="jacobi", convergence_norm="interface"),
)
_JACOBI_INTERFACE = len(KNOBS) - 1
#: Small cells for the four scores above, with ``diagnostics=True``: no
#: member has more than eight entries, so every gain has at most 64 and
#: the gradient bound is made per entry (``GRADIENT_PROBE_ENTRY_LIMIT``;
#: with a member of twelve the search reads a "gradient" score of 1.9,
#: which is outside CPL-093's conditions and no finding).
SIDE_DIAGNOSED = (
    Cell("side-4-4", "float64", 6, 120, "matrix-local"),
    Cell("side-2-8", "float64", _JACOBI_INTERFACE, 120, "matrix-local"),
    Cell("side-3-6", "float32", 2, 120, "matrix"),
    Cell("side-2-8-r", "float64", 6, 5, "matrix-local"),
    Cell("side-hub", "float32", 6, 120, "matrix-local"),
    Cell("side-2-8-r", "float64", 6, 5, "sparse-local"),
    Cell("side-hub", "float32", 6, 120, "sparse-local"),
)
SIDE_DIAGNOSED_CELLS = tuple(range(len(CELLS), len(CELLS) + len(SIDE_DIAGNOSED)))
#: Float64 cells with a dense gather for the widened draw below: a mapping
#: row that differences two entries of a large source field
#: (``Domain.cancel``).  There the delivered value and the source field are
#: on different scales, so an analysis taken on one reading under a
#: criterion taken on the other reports a wrong bound -- which, on a
#: well-conditioned mapping, it does not (seeded: the "bound" search over
#: :data:`SIDE_DIAGNOSED` alone lets it through).
SIDE_CANCELLING = (
    Cell("side-3-6", "float64", 6, 120, "matrix"),
    Cell("side-2-8", "float64", _JACOBI_INTERFACE, 120, "matrix"),
)
SIDE_CANCELLING_CELLS = tuple(range(len(CELLS) + len(SIDE_DIAGNOSED),
                                    len(CELLS) + len(SIDE_DIAGNOSED) + len(SIDE_CANCELLING)))
SIDE_CANCEL = Domain(cancel=30.0)
CELLS = CELLS + SIDE_DIAGNOSED + SIDE_CANCELLING

AWAITING_THE_SIDE_RULE = ct.INTERFACE_SIDE != "compact"
SIDE_DECISION = ("the interface norm reads a mapped edge on its compact side (decision of "
                 "2026-10-07); this tree reads what a scatter delivers")
SIDE_CAP = 200
_SIDE_ACCELERATIONS = (
    dict(acceleration="none"), dict(acceleration="aitken"),
    dict(acceleration="fixed", relaxation=0.7), dict(acceleration="iqn-ils"))


@dataclasses.dataclass(frozen=True)
class SideCell:
    """One compiled graph of the fifth score (no diagnostics: it reads the verdict)."""

    structure: str
    mapping_kind: str
    dtype: str
    schedule: str
    acceleration: int = 0

    @property
    def topo(self) -> ct.Topology:
        return SIDE_STRUCTURES[self.structure]

    @property
    def knobs(self) -> dict:
        return cg.live_knobs(dict(
            _SIDE_ACCELERATIONS[self.acceleration], iteration_mode=self.schedule,
            convergence_norm="interface", rtol=RTOL, solver="ift", max_iterations=SIDE_CAP))

    @property
    def awaits(self) -> bool:
        """Does a compact-side rule read this cell differently from a delivered one?"""
        topo = self.topo
        return any(topo.edges[i].mapped
                   and topo.node(topo.edges[i].dst).n > topo.node(topo.edges[i].src).n
                   for i in topo.internal_edges(0))


def _side_cells() -> tuple:
    """Every structure under every mapping kind, the dtype, the schedule and
    the acceleration rotated over them."""
    out = []
    kinds = ("matrix", "matrix-local", "sparse-local")
    for s, name in enumerate(sorted(SIDE_STRUCTURES)):
        for k, kind in enumerate(kinds):
            out.append(SideCell(name, kind, ("float64", "float32")[(s + k) % 2],
                                ("gauss-seidel", "jacobi")[(s + 2 * k) % 2],
                                (s + k) % len(_SIDE_ACCELERATIONS)))
    return tuple(out)


#: The per-push cells first: the tie, and the cell of ratio 300 that joins
#: it when the rule lands; then the second cell a pin lives on.
_SIDE_TIE = SideCell("side-4-4", "sparse-local", "float64", "jacobi")
_SIDE_300 = SideCell("side-1-300", "sparse-local", "float32", "jacobi")
_SIDE_300_DENSE = SideCell("side-1-300", "matrix-local", "float64", "gauss-seidel")
_SIDE_FIRST = (_SIDE_TIE, _SIDE_300, _SIDE_300_DENSE)
SIDE_CELLS = _SIDE_FIRST + tuple(c for c in _side_cells() if c not in _SIDE_FIRST)
SIDE_AWAITING = tuple(i for i, c in enumerate(SIDE_CELLS) if c.awaits)
SIDE_CLAIMED = tuple(i for i, c in enumerate(SIDE_CELLS)
                     if not (c.awaits and AWAITING_THE_SIDE_RULE))
SIDE_PER_PUSH = tuple(i for i in (0, 1) if i in SIDE_CLAIMED)


@functools.lru_cache(maxsize=len(SIDE_CELLS))
def _side_built(index: int) -> ct.Built:
    cell = SIDE_CELLS[index]
    with precision(cell.dtype == "float64"):
        return ct.build(cell.topo, cell.knobs, dtype=cell.dtype, mapping_kind=cell.mapping_kind)


@dataclasses.dataclass(frozen=True)
class SideCase:
    cell: int
    seed: int
    rho: float
    #: How far from the biases the step starts, in field magnitudes.
    offset: float


@functools.lru_cache(maxsize=4096)
def observe_side(case: SideCase) -> dict:
    """One step of *case* and the fifth score of its verdict."""
    cell = SIDE_CELLS[case.cell]
    topo, cfgs = cell.topo, ct.group_cfgs_of([cell.knobs])
    values = ct.side_values(topo, np.random.default_rng(case.seed), case.rho, dtype=cell.dtype,
                            group_cfgs=cfgs, mapping_kind=cell.mapping_kind,
                            offset=case.offset)
    with precision(cell.dtype == "float64"):
        (step,) = ct.run(_side_built(case.cell), values, 1)
    d = step.reports[0]
    out = dict(side=0.0, converged=bool(d["converged"]),
               report={k: d[k] for k in ("iterations", "converged", "residual")})
    if not out["converged"] or not all(np.all(np.isfinite(s["x"])) for s in step.state.values()):
        return out
    model = ct.LinearModel(topo, values, dtype=cell.dtype, group_cfgs=cfgs)
    distance, K = model.interface_claim(0, step.pre, step.state, "compact")
    out["side"] = distance / K
    out["report"].update(distance=distance, K=K)
    return out


def side_cases(cells=SIDE_CLAIMED):
    return st.builds(SideCase, cell=st.sampled_from(tuple(cells)), seed=st.integers(0, 2 ** 16),
                     rho=st.floats(0.05, 0.95),
                     offset=st.one_of(st.just(1.0), _decades(-3.0, 1.0)))


#: A distance of ``K`` tolerances holds exactly in exact arithmetic; the
#: allowance is the bound score's.
SIDE_THRESHOLD = THRESHOLD["bound"]


def search_side(cells, profile, *, fail: bool = True):
    """Run the fifth search over *cells*; ``(report, converged fraction)``."""
    drawn = []

    def score(case: SideCase):
        drawn.append(case)
        seen = observe_side(case)
        return seen["side"], seen["report"]

    report = targeted_search(side_cases(cells), score, SIDE_THRESHOLD, profile=profile,
                             label="side", fail=fail)
    return report, sum(observe_side(c)["converged"] for c in drawn) / max(len(drawn), 1)


def test_a_converged_interface_quantity_is_within_K_tolerances_per_push():
    _report, converged = search_side(SIDE_PER_PUSH, EVERY_PUSH.seeded(len(THRESHOLD)))
    assert converged > 0.5, f"only {converged:.2f} of the examples converged"


# Slow: every claimed cell, 800 random examples.
# Per push: tests/property/test_coupling_targeted_search.py::test_a_converged_interface_quantity_is_within_K_tolerances_per_push
@pytest.mark.slow
def test_the_hunt_finds_no_interface_quantity_beyond_K_tolerances():
    report, converged = search_side(SIDE_CLAIMED, SLOW)
    print(f"side, {len(SIDE_CLAIMED)} cells: worst {report}; converged fraction {converged:.2f}")
    assert converged >= USABLE_FLOOR


# Slow: the cells that wait for the rule (some thirty compiles on this tree).  Strict:
# until the rule lands the hunt must *reach* the dilution, which is what
# shows the score can find a wrong side; afterwards these cells are claimed
# and the test above hunts them.
# Per push: tests/property/test_coupling_targeted_search.py::test_a_known_side_defect_the_search_reached_is_fixed
@pytest.mark.slow
@pytest.mark.xfail(AWAITING_THE_SIDE_RULE, strict=True, raises=AssertionError,
                   reason=SIDE_DECISION)
def test_the_hunt_finds_no_interface_quantity_beyond_K_tolerances_on_a_larger_target():
    search_side(SIDE_AWAITING, dataclasses.replace(SLOW, max_examples=SLOW.max_examples // 4))


def _known_side(case: SideCase):
    return pytest.param(case, marks=pytest.mark.xfail(
        AWAITING_THE_SIDE_RULE, strict=True, raises=AssertionError, reason=SIDE_DECISION))


#: What the fifth search reaches on this tree: a converged pair whose one
#: marker value is several ``K`` tolerances from its fixed point, the
#: scatter onto 300 entries pooled into the norm.  Each is over the
#: threshold by a factor of two or more on jaxlib 0.10.2, 0.11.0 and 0.11.2.
KNOWN_SIDE = {
    "a-scatter-onto-300-sparse": _known_side(SideCase(1, 1, 0.6, 1.0)),
    "a-scatter-onto-300-dense-local": _known_side(SideCase(2, 3, 0.5, 1.0)),
}


@pytest.mark.parametrize("case", list(KNOWN_SIDE.values()), ids=list(KNOWN_SIDE))
def test_a_known_side_defect_the_search_reached_is_fixed(case):
    seen = observe_side(case)
    assert seen["converged"], seen["report"]
    assert seen["side"] <= SIDE_THRESHOLD, (
        f"converged at {seen['side']:.3g} K tolerances from the fixed point in the compact "
        f"readings: {seen['report']}")


# Slow: seven more cells compiled with diagnostics, the four searches on them.
# Per push: tests/property/test_coupling_targeted_search.py::test_a_usable_error_bound_is_never_below_the_distance_per_push
@pytest.mark.slow
@pytest.mark.parametrize("name", SEARCHES)
def test_the_reported_numbers_hold_on_cells_with_a_mapping_between_sizes(name):
    profile = dataclasses.replace(SLOW, max_examples=SLOW.max_examples // 4)
    report, usable = search(name, cells=SIDE_DIAGNOSED_CELLS, profile=profile)
    print(f"{name}, side cells: worst {report}; usable fraction {usable:.2f}")
    assert usable > 0, f"{name}: no example had the flag set"


# Slow: two more cells compiled with diagnostics.
# Per push: tests/property/test_coupling_targeted_search.py::test_a_usable_error_bound_is_never_below_the_distance_per_push
@pytest.mark.slow
@pytest.mark.parametrize("name", ("bound", "floor"))
def test_the_reported_numbers_hold_where_a_gather_row_differences_a_large_field(name):
    profile = dataclasses.replace(SLOW, max_examples=SLOW.max_examples // 4)
    report, usable = search(name, cells=SIDE_CANCELLING_CELLS, domain=SIDE_CANCEL,
                            profile=profile)
    print(f"{name}, cancelling side cells: worst {report}; usable fraction {usable:.2f}")
    assert usable > 0, f"{name}: no example had the flag set"


# =============================================================================
# THE GRADIENT SCORE'S CONSTANTS ON A SPARSE EDGE
# =============================================================================
#
# The gradient score is the worst over "every scalar gain and mapping
# weight".  Which entries of a mapped edge's matrix are weights depends on
# how the graph holds it (:func:`gradient_constants`), and a score taken
# over entries the graph does not hold is a wrong score with the flag set.
# The enumeration was every entry of the matrix whatever the edge: on the
# example below it read 2.62 (jaxlib 0.11.0; 2.58 on 0.10.2 and 0.11.2),
# all of it from entries outside a local pattern (0.53 over the gains and
# the pattern's eight weights).  The same matrices held dense score 0.48,
# and rightly over every entry: there each is a weight and
# ``gradient_relative_error_bound`` covers it (5.6e-3, against 1.0e-3 on
# the sparse graph).
#
# Which patterns can show it: the relative error of the derivative with
# respect to a weight is that of the source entry it reads, whatever its
# row (the weight moves the pass along one fixed direction, by that
# entry).  So the entries outside a pattern change the worst only in a
# column that holds none -- a source entry the mapping never reads.  A
# local gather has such columns (the example reads two of eight cells); a
# ragged pattern has a full row and none, so its scores are the same
# numbers under either enumeration.

_LOCAL_SPARSE = Cell("side-2-8-r", "float64", 6, 5, "sparse-local")
#: The example: two markers and eight cells, four of each edge's sixteen
#: entries in its pattern, stopped after five passes at a loop gain of 0.5.
OUTSIDE_THE_PATTERN = Case(CELLS.index(_LOCAL_SPARSE), 1, 0.5, False, 1.0, 0.0, 1.0, 0, 1.0, 0)
#: One cell per sparse kind whose pattern leaves entries out.
_SPARSE_SHAPES = {
    "sparse-local": _LOCAL_SPARSE,
    "sparse-ragged": Cell("mapped", "float64", 0, 5, "sparse-ragged"),
    "sparse-scatter": Cell("mapped", "float64", 0, 5, "sparse-scatter"),
}
#: The mapping kinds under which an edge holds fewer weights than its
#: matrix has entries, on the structures below.
_HOLDS_A_PATTERN = ("sparse-ragged", "sparse-scatter", "sparse-local")


def _named(constant: dict) -> tuple:
    return (constant.get("node"), constant.get("port"), constant.get("edge"), constant["entry"])


def _outside(topo: ct.Topology, mapping_kind: str) -> set:
    """``(edge, entry)`` of every matrix entry outside its edge's pattern."""
    return {(i, (int(a), int(b))) for i, e in enumerate(topo.edges) if e.mapped
            for a, b in zip(*np.nonzero(~ct.mapping_pattern(topo, i, mapping_kind)))}


def _plain_passes(model: ct.LinearModel, values: dict, count: int) -> tuple:
    """``(pre, state)`` after *count* plain passes of group 0 from the drawn
    start: an iterate away from the fixed point, with no graph compiled."""
    topo = model.topo
    assert set(topo.groups[0]) == set(topo.names), "a group with no outside reads"
    members, off, k = model._group_layout(0)  # noqa: SLF001
    pre = {m: {"x": np.asarray(values["nodes"][m]["x0"], np.float64)} for m in members}
    L, U = (np.asarray(a, np.float64) for a in model.group_pass(0))
    c = np.asarray(model.group_constant(0, pre, pre), np.float64)
    x = np.concatenate([pre[m]["x"] for m in members])
    for _ in range(count):
        x = np.linalg.solve(np.eye(k) - L, U @ x + c)
    return pre, {m: {"x": x[off[m]:off[m] + topo.node(m).n]} for m in members}


@pytest.mark.parametrize("kind", ct.MAPPING_KINDS + ct.LOCAL_KINDS)
def test_the_gradient_constants_of_a_mapped_edge_are_the_weights_its_build_holds(kind):
    """Read off the build, not restated: a sparse edge's weights are the
    valid slots ``ct.build`` lays out for it (where ``ct.params_for``
    writes a matrix into the graph), a dense edge's are its whole matrix.
    No graph is compiled."""
    fewer = 0
    for name in ("mapped", "side-2-8-r", "side-hub", "side-4-4"):
        topo = STRUCTURES[name]
        built = ct.build(topo, Cell(name, "float32", 0, 5).knobs, compile=False,
                         mapping_kind=kind)
        constants = gradient_constants(topo, kind)
        assert len({_named(c) for c in constants}) == len(constants), "a constant named twice"
        mapped = [i for i, e in enumerate(topo.edges) if e.mapped]
        assert mapped and {c["edge"] for c in constants if "edge" in c} == set(mapped)
        for i in mapped:
            e = topo.edges[i]
            every = {(a, b) for a in range(topo.node(e.dst).n) for b in range(topo.node(e.src).n)}
            if i in built.slots:
                slots = built.slots[i]
                held = {(int(a), int(b))
                        for a, b in zip(slots.target[slots.valid], slots.source[slots.valid])}
            else:
                held = every
            scored = {c["entry"] for c in constants if c.get("edge") == i}
            assert scored == held, (
                f"{name}, edge {i} under {kind!r}: the score differentiates {sorted(scored - held)} "
                f"which the graph does not hold and leaves out {sorted(held - scored)} which it does")
            fewer += held < every
    # The premise: the kinds that hold a pattern hold fewer weights than
    # entries somewhere, and no other kind does.
    assert (fewer > 0) is (kind in _HOLDS_A_PATTERN), (kind, fewer)


@pytest.mark.parametrize("kind", sorted(_SPARSE_SHAPES))
def test_the_gradient_errors_of_a_sparse_edge_are_the_dense_edges_on_its_pattern(kind):
    """One matrix and one iterate, enumerated as a sparse edge and as a
    dense edge hold them (no graph: the oracle alone).  The sparse errors
    are the dense ones constant for constant, without the entries outside
    the pattern; a pattern entry whose weight is zero is still a constant;
    and a dense *local* matrix, zero outside its pattern, is differentiated
    in every entry.  On the local example the entries a sparse edge does
    not hold carry five times the error of those it does: a bound that is
    tight for the graph's constants reads far over on them."""
    cell = _SPARSE_SHAPES[kind]
    topo = cell.topo
    values = values_of(OUTSIDE_THE_PATTERN, cell)
    model = ct.LinearModel(topo, values, dtype=cell.dtype, group_cfgs=cell.cfgs)
    pre, state = _plain_passes(model, values, cell.cap)
    as_sparse = {_named(c): error for error, c in gradient_errors(model, pre, state, kind)}
    as_dense = {_named(c): error for error, c in gradient_errors(model, pre, state, "matrix")}
    outside = {(None, None, edge, entry) for edge, entry in _outside(topo, kind)}
    assert outside and as_sparse
    left_out = set(as_dense) - set(as_sparse)
    assert left_out and left_out <= outside, "the premise: the fixed point moves with them"
    assert not set(as_sparse) & outside
    assert as_sparse == {name: error for name, error in as_dense.items() if name not in outside}
    # A weight's error is its source entry's: every entry of a column has
    # one error, so only a column no weight reads can raise the worst.
    patterns = {i: ct.mapping_pattern(topo, i, kind) for i, e in enumerate(topo.edges) if e.mapped}
    for i in patterns:
        columns: dict = {}
        for (_node, _port, at, (_row, column)), error in as_dense.items():
            if at == i:
                columns.setdefault(column, []).append(error)
        assert all(max(errors) - min(errors) <= 1e-9 * max(errors) for errors in columns.values())
    unread = {(i, int(b)) for i, pattern in patterns.items()
              for b in np.nonzero(~pattern.any(axis=0))[0]}
    assert bool(unread) is (kind == "sparse-local")
    if not unread:
        # (To rounding: two rows of a column differ in the last place.)
        assert max(as_dense.values()) <= max(as_sparse.values()) * (1.0 + 1e-9)

    # A weight at zero is still a weight: the pattern says so, not the values.
    edge = next(i for i, e in enumerate(topo.edges) if e.mapped)
    entry = tuple(int(v) for v in np.argwhere(ct.mapping_pattern(topo, edge, kind))[0])
    zeroed = {**values, "H": {**values["H"], edge: np.array(values["H"][edge])}}
    zeroed["H"][edge][entry] = 0.0
    at_zero = ct.LinearModel(topo, zeroed, dtype=cell.dtype, group_cfgs=cell.cfgs)
    assert (None, None, edge, entry) in {
        _named(c) for _error, c in gradient_errors(at_zero, *_plain_passes(
            at_zero, zeroed, cell.cap), kind)}

    if kind == "sparse-local":
        # The same matrices held dense: every entry, the zeros included.
        local = {_named(c): error
                 for error, c in gradient_errors(model, pre, state, "matrix-local")}
        assert local == as_dense
        beyond = max(as_dense, key=as_dense.get)
        assert beyond in outside and (beyond[2], beyond[3][1]) in unread
        assert as_dense[beyond] > 4.0 * max(as_sparse.values()), (
            as_dense[beyond], max(as_sparse.values()))


def test_the_gradient_oracle_refuses_a_matrix_a_sparse_edge_cannot_hold():
    """A model whose matrix is non-zero outside the pattern is not the
    model of a sparse graph: scoring it as one would be a third reading."""
    cell = dataclasses.replace(_LOCAL_SPARSE, mapping_kind="matrix")
    values = values_of(OUTSIDE_THE_PATTERN, cell)
    model = ct.LinearModel(cell.topo, values, dtype=cell.dtype, group_cfgs=cell.cfgs)
    pre, state = _plain_passes(model, values, cell.cap)
    assert gradient_errors(model, pre, state, "matrix")
    with pytest.raises(AssertionError, match="is not one a 'sparse-local' edge holds"):
        gradient_errors(model, pre, state, "sparse-local")


def test_the_gradient_constants_refuse_a_member_whose_gains_are_read_at_trace_time():
    """A three-argument member's gains are baked into the compiled step:
    like an entry outside a pattern, nothing the graph holds as a parameter."""
    b = ct.TopologyBuilder()
    b.node("a", 2)
    b.node("b", 2, three_arg=True)
    b.edge("a", "b")
    b.edge("b", "a")
    b.group("a", "b")
    with pytest.raises(AssertionError, match="reads its gains when the step is traced"):
        gradient_constants(b.build(), "matrix")


# Slow: the sparse cell and its dense twin compiled with diagnostics.
# Per push: tests/property/test_coupling_targeted_search.py::test_the_gradient_errors_of_a_sparse_edge_are_the_dense_edges_on_its_pattern
@pytest.mark.slow
def test_the_gradient_score_of_a_sparse_edge_is_taken_over_its_pattern():
    """The step itself, on the example the old enumeration read 2.62 on.

    The score holds and its worst constant is one the graph holds; the
    entries outside the pattern would have carried it past the threshold;
    and the same matrices held dense are scored over every entry, which
    the library's own bound allows for (it is the larger of the two)."""
    case = OUTSIDE_THE_PATTERN
    cell = CELLS[case.cell]
    topo, kind = cell.topo, cell.mapping_kind
    seen = observe(case)
    report = seen["report"]
    assert seen["gradient_usable"], report
    assert seen["gradient"] <= THRESHOLD["gradient"], report
    values = values_of(case)
    with precision(cell.dtype == "float64"):
        (step,) = ct.run(_built(case.cell), values, 1)
    model = ct.LinearModel(topo, values, dtype=cell.dtype, group_cfgs=cell.cfgs)
    outside = _outside(topo, kind)
    worst, constant = max(gradient_errors(model, step.pre, step.state, kind),
                          key=lambda error: error[0])
    assert worst == pytest.approx(report["gradient_error"], rel=1e-9)
    assert constant == report["gradient_constant"]
    assert (constant.get("edge"), constant["entry"]) not in outside
    beyond, where = max(gradient_errors(model, step.pre, step.state, "matrix"),
                        key=lambda error: error[0])
    assert (where.get("edge"), where["entry"]) in outside
    # units: a score; measured 2.58 on jaxlib 0.10.2 and 0.11.2, 2.62 on 0.11.0.
    assert beyond / report["gradient_allowed"] > 1.5 * THRESHOLD["gradient"], (
        beyond, report["gradient_allowed"])

    dense = dataclasses.replace(case, cell=CELLS.index(
        dataclasses.replace(cell, mapping_kind="matrix-local")))
    held_dense = observe(dense)
    assert held_dense["gradient_usable"], held_dense["report"]
    assert held_dense["gradient"] <= THRESHOLD["gradient"], held_dense["report"]
    at = held_dense["report"]["gradient_constant"]
    # Every entry of the dense matrix is scored: the worst is a zero the
    # sparse edge does not hold, at the error the sparse score left out.
    assert (at.get("edge"), at["entry"]) in outside
    assert held_dense["report"]["gradient_error"] == pytest.approx(beyond, rel=1e-6)
    assert (held_dense["report"]["gradient_relative_error_bound"]
            > 2.0 * report["gradient_relative_error_bound"])
