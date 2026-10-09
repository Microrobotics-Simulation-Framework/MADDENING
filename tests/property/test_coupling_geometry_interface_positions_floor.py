"""What stored positions put into the float floor of an interface reading,
at ``compile()`` and at every step (experimental: ``convergence_norm=
"interface"`` over a ``multilinear_grid`` mapping; MAP-050, MAP-048).

Four properties, each found missing by an audit of the feature before it
merged, and one condition of the claim:

* **The report reads the floor at every step.**  ``compile()`` asks once
  whether a dtype resolves the positions a reading rests on.  Markers that
  drift, a state write and a loaded checkpoint are later than that, and a
  float32 pair then read ``converged=True``, ``residual=0.0`` and
  ``precision_limited=False`` 3 to 20 tolerances from its fixed point.
  The report of such a group now carries the floor of the state each step
  returned (``residual_precision_floor``) and ``precision_limited`` by the
  rule of every group's report.  The oracle is the rule restated in NumPy
  from the stored numbers: four roundings per evaluation of every entry a
  part holds, a value at the group's coarsest ``eps``, positions at that
  ``eps`` times their distance from zero in spacings, pooled.
* **The count is an assumption for a delivered value.**  It takes the
  field to vary across a cell by about the size of the value delivered.
  A gather that samples a field near its zero is moved by more, and the
  floor does not see it (MADD-ANO-247: pinned as a strict xfail, with the
  controls that hold beside it).
* **The sides are decided by entry counts.**  With more markers than grid
  entries the gather is the edge read at its source and the scatter the
  one read as delivered.
* **The advisory takes the floor's decisions.**  A delivered value the
  dead band drops, a coordinate on an axis of one lattice point and a
  point clamped to the hull put nothing into the floor and are not
  warned of; the true alarms beside each stay.
* **The claim is first order, and a lattice plane ends it.**  A group
  whose iterate jumps a plane into a cell where its pass contracts slowly
  can report ``converged=True`` far from a fixed point in the cell it
  came from: constructed and pinned with its numbers.

Nothing here imports the library's kernel, plan or norm for an oracle:
the stencil (one axis, clamped), the parts, the pooling and the pass are
restated in NumPy on the numbers the graph stores.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Optional

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling.acceleration import residual_precision_floor
from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.property.sysid_transform_grid import precision

KEY = "grid+markers"
SCATTER, GATHER = "markers.f->grid.deposit", "grid.x->markers.sampled"
#: Evaluations a pass of a Gauss-Seidel pair rounds like (the second
#: member reads the first from the same pass): the structural count.
EVALUATIONS = 2.0
#: ``PRECISION_FLOOR_ULPS``, restated: roundings counted per evaluation.
ULPS = 4.0
ADVISORY = "cannot be resolved to this tolerance"
PART, DELIVERED_AT = "read on edge", "is delivered at"


@pytest.fixture(autouse=True)
def _float64_available():
    """Every graph here names its dtypes; float64 ones need x64."""
    with precision(True):
        yield


# ---------------------------------------------------------------------------
# The pair
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Pair:
    """Markers ``(f, pos)`` and a grid ``(x[, gp])`` joined by a scatter
    (``markers.f -> grid.deposit``) and a gather (``grid.x ->
    markers.sampled``).  ``anchors`` are ``(scatter's, gather's)``: the
    markers hold ``pos`` (the scatter's source, the gather's target) and
    the grid holds ``gp`` (the scatter's target, the gather's source)."""

    n: int = 8000
    m: int = 4
    spacing: tuple = (0.5,)
    shape: Optional[tuple] = None        # the lattice (``(n,)`` when ``None``)
    origin: float = 0.0                  # in spacings, on every axis
    anchors: tuple = ("source", "target")
    dtype: str = "float32"
    pos_dtype: Optional[str] = None
    rtol: float = 1e-5
    atol: float = 0.0
    cap: int = 100
    u0: float = 2.0                      # the first marker, in spacings from zero
    drift: float = 0.0                   # spacings a step, along the first axis
    push: float = 0.4                    # spacings per unit of what a point samples
    #: ``x <- kx x + bx + gx deposit`` and ``f <- kf f + bf + gf sampled``.
    grid_update: tuple = (0.5, 0.5, 0.3)
    markers_update: tuple = (0.2, 0.3, 0.5)
    field: str = "random"                # or "ramp": zero at ``ramp_zero`` spacings
    ramp_zero: float = 0.0
    f0: float = 0.7
    offsets: Optional[tuple] = None      # the markers, in spacings from ``u0``
    across: float = 0.0                  # every marker's second coordinate (lengths)

    @property
    def lattice(self) -> tuple:
        return (self.n,) if self.shape is None else tuple(self.shape)

    @property
    def geometry_dtype(self) -> str:
        return self.pos_dtype or self.dtype

    @property
    def holds_gp(self) -> bool:
        return self.anchors[0] == "target" or self.anchors[1] == "source"

    def start_positions(self) -> np.ndarray:
        """``(m, d)``: what both nodes hold at ``compile()``, as stored."""
        offsets = (tuple(1.37 * k for k in range(self.m)) if self.offsets is None
                   else self.offsets)
        pos = np.full((self.m, len(self.lattice)), self.across)
        pos[:, 0] = [(self.u0 + o) * self.spacing[0] for o in offsets]
        return np.asarray(np.asarray(pos, np.dtype(self.geometry_dtype)), np.float64)

    def start_field(self) -> np.ndarray:
        if self.field == "ramp":
            return np.arange(self.n) - self.ramp_zero
        if self.field == "zero":
            return np.zeros(self.n)
        return 1.0 + np.random.default_rng(0).random(self.n)


class _Grid(SimulationNode):
    def __init__(self, name, timestep, pair: Pair):
        super().__init__(name, timestep)
        self._pair = pair

    def initial_state(self):
        p = self._pair
        state = {"x": jnp.asarray(p.start_field(), p.dtype)}
        if p.holds_gp:
            state["gp"] = jnp.asarray(p.start_positions(), p.geometry_dtype)
        return state

    def boundary_input_spec(self):
        p = self._pair
        return {"deposit": BoundaryInputSpec(shape=(p.n,), dtype=jnp.dtype(p.dtype))}

    def update(self, state, boundary_inputs, dt):
        p = self._pair
        kx, bx, gx = p.grid_update
        deposit = boundary_inputs.get("deposit", jnp.zeros(p.n, p.dtype))
        out = {"x": (kx * state["x"] + bx + gx * deposit).astype(p.dtype)}
        if p.holds_gp:
            # The move is computed in the positions' own dtype: what a
            # position carries of a coarser field is that field's rounding
            # of the push, not of the distance from zero.
            held = state["gp"].dtype
            along = jnp.zeros(len(p.lattice), held).at[0].set(p.spacing[0])
            move = (p.drift + p.push * jnp.mean(deposit).astype(held)) * along
            out["gp"] = (state["gp"] + move).astype(held)
        return out


class _Markers(SimulationNode):
    def __init__(self, name, timestep, pair: Pair):
        super().__init__(name, timestep)
        self._pair = pair

    def initial_state(self):
        p = self._pair
        return {"f": jnp.full(p.m, p.f0, p.dtype),
                "pos": jnp.asarray(p.start_positions(), p.geometry_dtype)}

    def boundary_input_spec(self):
        p = self._pair
        return {"sampled": BoundaryInputSpec(shape=(p.m,), dtype=jnp.dtype(p.dtype))}

    def update(self, state, boundary_inputs, dt):
        p = self._pair
        kf, bf, gf = p.markers_update
        sampled = boundary_inputs.get("sampled", jnp.zeros(p.m, p.dtype))
        held = state["pos"].dtype            # the move in the positions' dtype (see the grid)
        along = jnp.zeros(len(p.lattice), held).at[0].set(p.spacing[0])
        move = (p.drift + p.push * sampled.astype(held))[:, None] * along
        return {"f": (kf * state["f"] + bf + gf * sampled).astype(p.dtype),
                "pos": (state["pos"] + move).astype(held)}


def build(pair: Pair, *, norm: str = "interface"):
    """``(graph, advisories)``: *pair* compiled, and the texts of the
    position advisories ``compile()`` gave."""
    gm = GraphManager()
    gm.add_node(_Grid("grid", 0.01, pair))
    gm.add_node(_Markers("markers", 0.01, pair))
    grid = dict(origin=tuple(pair.origin * h for h in pair.spacing), spacing=pair.spacing,
                shape=pair.lattice, n_points=pair.m)
    gm.add_edge("markers", "grid", "f", "deposit",
                mapping=multilinear_grid_mapping(mode="conservative", **grid),
                geometry=(pair.anchors[0], "pos" if pair.anchors[0] == "source" else "gp"))
    gm.add_edge("grid", "markers", "x", "sampled",
                mapping=multilinear_grid_mapping(mode="consistent", **grid),
                geometry=(pair.anchors[1], "pos" if pair.anchors[1] == "target" else "gp"))
    tolerance = {"tolerance": 1e-6} if norm == "l2" else {"rtol": pair.rtol, "atol": pair.atol}
    gm.add_coupling_group(["grid", "markers"], convergence_norm=norm,
                          iteration_mode="gauss-seidel", max_iterations=pair.cap, **tolerance)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
    return gm, [str(w.message) for w in caught if ADVISORY in str(w.message)]


def stored(gm) -> dict:
    """The members' state as stored, each number exactly, in float64."""
    return {name: {field: np.asarray(value, np.float64)
                   for field, value in gm.get_node_state(name).items()}
            for name in ("grid", "markers")}


def report(gm) -> dict:
    return dict(gm.coupling_diagnostics()[KEY])


def same_reports(a: dict, b: dict) -> bool:
    def norm(v):
        return "nan" if isinstance(v, float) and v != v else v
    return {k: norm(v) for k, v in a.items()} == {k: norm(v) for k, v in b.items()}


def warned(advisories) -> dict:
    """``{edge key: "part" | "delivered"}`` of the advisories given."""
    out = {}
    for text in advisories:
        (key,) = [k for k in (SCATTER, GATHER) if repr(k) in text]
        assert key not in out, advisories
        assert (PART in text) != (DELIVERED_AT in text), text
        out[key] = "part" if PART in text else "delivered"
    return out


def write_far(gm, pair: Pair, by: float) -> None:
    """Move every stored position *by* spacings along the first axis."""
    for name, field in (("markers", "pos"), ("grid", "gp")):
        state = dict(gm.get_node_state(name))
        if field in state:
            far = np.array(state[field], np.float64)
            far[:, 0] += by * pair.spacing[0]
            gm.set_node_state(name, {**state, field: jnp.asarray(far, state[field].dtype)})


# ---------------------------------------------------------------------------
# The rule, restated
# ---------------------------------------------------------------------------


def coarsest_eps(pair: Pair) -> float:
    return max(float(np.finfo(np.dtype(d)).eps) for d in (pair.dtype, pair.geometry_dtype))


def coordinates_read(pair: Pair, positions: np.ndarray) -> np.ndarray:
    """Which coordinates a delivered value depends on: not one on an axis
    of one lattice point, and not one outside the hull by more than
    ``sqrt(eps)`` of its own distance from zero (both in spacings)."""
    eps = float(np.finfo(np.dtype(pair.geometry_dtype)).eps)
    read = np.ones(positions.shape, bool)
    for a, count in enumerate(pair.lattice):
        q = positions[:, a] / pair.spacing[a]
        u = q - pair.origin
        outside = np.maximum(np.maximum(-u, u - (count - 1)), 0.0)
        read[:, a] = (count >= 2) & ~(outside > math.sqrt(eps) * np.abs(q))
    return read


def floor_parts(pair: Pair, pre: dict, post: dict) -> list:
    """``[(entries, resolution), ...]`` of the reading at the state *post*
    a step returned from *pre*: every entry of a part at one resolution."""
    eps = coarsest_eps(pair)
    h = np.asarray(pair.spacing)

    def reach(positions, delivered):
        q = np.abs(positions / h)
        if delivered:
            q = np.where(coordinates_read(pair, positions), q, 0.0)
        return float(np.max(q))

    parts = []
    # The scatter, m points onto n entries.
    if pair.m < pair.n:                 # read at its source
        parts.append((pair.m, eps))
        if pair.anchors[0] == "source":
            parts.append((post["markers"]["pos"].size,
                          eps * reach(post["markers"]["pos"], False)))
    else:                               # read as delivered
        at = post["markers"]["pos"] if pair.anchors[0] == "source" else pre["grid"]["gp"]
        parts.append((pair.n, max(eps, eps * reach(at, True))))
    # The gather, n entries onto m points.
    if pair.m > pair.n:                 # read at its source
        parts.append((pair.n, eps))
        if pair.anchors[1] == "source":
            parts.append((post["grid"]["gp"].size, eps * reach(post["grid"]["gp"], False)))
    else:                               # read as delivered
        at = post["grid"]["gp"] if pair.anchors[1] == "source" else pre["markers"]["pos"]
        parts.append((pair.m, max(eps, eps * reach(at, True))))
    return parts


def floor_of(pair: Pair, pre: dict, post: dict) -> float:
    """The residual's float floor at *post*, in tolerances."""
    parts = floor_parts(pair, pre, post)
    pooled = math.sqrt(sum(n * r * r for n, r in parts) / sum(n for n, _r in parts))
    return ULPS * EVALUATIONS * pooled / pair.rtol


# -- the pass on one axis, in float64 ---------------------------------------


def _stencil(pair: Pair, positions: np.ndarray):
    n, h = pair.lattice[0], pair.spacing[0]
    u = np.clip(positions[:, 0] / h - pair.origin, 0.0, n - 1.0)
    base = np.clip(np.floor(u), 0, n - 2).astype(np.int64)
    return base, u - base


def _gather(pair, x, positions):
    base, w = _stencil(pair, positions)
    return (1.0 - w) * x[base] + w * x[base + 1]


def _scatter(pair, f, positions):
    base, w = _stencil(pair, positions)
    out = np.zeros(pair.lattice[0])
    np.add.at(out, base, (1.0 - w) * f)
    np.add.at(out, base + 1, w * f)
    return out


def one_pass(pair: Pair, x: dict, pre: dict) -> dict:
    """One Gauss-Seidel pass (the grid, then the markers) from the iterate
    *x*; a target-anchored geometry is the target's pre-step state."""
    kx, bx, gx = pair.grid_update
    kf, bf, gf = pair.markers_update
    h = pair.spacing[0]
    at = x["markers"]["pos"] if pair.anchors[0] == "source" else pre["grid"]["gp"]
    deposit = _scatter(pair, x["markers"]["f"], at)
    grid = {"x": kx * pre["grid"]["x"] + bx + gx * deposit}
    if pair.holds_gp:
        grid["gp"] = pre["grid"]["gp"] + (pair.drift + pair.push * np.mean(deposit)) * h
    at = grid["gp"] if pair.anchors[1] == "source" else pre["markers"]["pos"]
    sampled = _gather(pair, grid["x"], at)
    markers = {"f": kf * pre["markers"]["f"] + bf + gf * sampled,
               "pos": pre["markers"]["pos"] + ((pair.drift + pair.push * sampled) * h)[:, None]}
    return {"grid": grid, "markers": markers}


def fixed_point(pair: Pair, pre: dict) -> dict:
    x = pre
    for _ in range(5000):
        nxt = one_pass(pair, x, pre)
        moved = max(float(np.max(np.abs(nxt[n][f] - x[n][f])) / (1e-300 + np.max(np.abs(nxt[n][f]))))
                    for n in nxt for f in nxt[n])
        x = nxt
        if moved < 1e-15:
            return x
    raise AssertionError("the float64 pass did not settle")


def readings(pair: Pair, x: dict, pre: dict) -> list:
    """``[(values, in spacings?)]``: what the norm reads of the iterate
    *x*, for ``m < n`` (the scatter at its source, the gather as
    delivered)."""
    assert pair.m < pair.n
    out = [(x["markers"]["f"], False)]
    if pair.anchors[0] == "source":
        out.append((x["markers"]["pos"] / np.asarray(pair.spacing), True))
    at = x["grid"]["gp"] if pair.anchors[1] == "source" else pre["markers"]["pos"]
    out.append((_gather(pair, x["grid"]["x"], at), False))
    return out


def scaled_change(pair: Pair, new: dict, old: dict, pre: dict, *, reference=None) -> float:
    """The pooled change between two iterates, in tolerances (``atol=0``):
    the residual of *new* against *old*; or, with *reference*, each value
    over the reference's magnitude (a distance from it)."""
    total, count = 0.0, 0
    against = readings(pair, reference if reference is not None else new, pre)
    for (a, lengths), (b, _l), (r, _l2) in zip(readings(pair, new, pre),
                                               readings(pair, old, pre), against):
        scale = 1.0 if lengths else (float(np.max(np.abs(r))) if reference is not None
                                     else float(max(np.max(np.abs(a)), np.max(np.abs(b)))))
        if scale == 0.0:
            continue
        total += float(np.sum((np.abs(a - b) / (pair.rtol * scale)) ** 2))
        count += a.size
    return math.sqrt(total / max(count, 1))


# ---------------------------------------------------------------------------
# A. The report reads the floor at every step
# ---------------------------------------------------------------------------

#: float32 at ``rtol=1e-5`` under Gauss-Seidel: ``compile()`` warns from
#: ``rtol / (4 E eps)`` = 10.5 spacings from zero; the lattice is 8000
#: points long (under the 1/1024-cell warning of a float32 geometry).
DRIFTING = Pair(drift=950.0)
THRESHOLD = DRIFTING.rtol / (ULPS * EVALUATIONS * float(np.finfo(np.float32).eps))
ALL_ANCHORS = (("source", "target"), ("target", "source"), ("source", "source"),
               ("target", "target"))


def _checked(pair: Pair, gm, pre: dict) -> dict:
    """The group's report after a step from *pre*, checked against the
    rule: the floor of the state the step returned, and the flag by the
    rule of every group's report."""
    post, d = stored(gm), report(gm)
    want = floor_of(pair, pre, post)
    assert d["residual_precision_floor"] == pytest.approx(want, rel=2e-5), (d, want)
    assert d["precision_limited"] is bool(
        want > 0 and math.isfinite(d["residual"]) and d["residual"] <= d[
            "residual_precision_floor"]), d
    for name in ("spectral_error_bound", "rho_spectral", "error_estimate"):
        assert math.isnan(d[name]), (name, d)           # the bounds stay unreported
    assert not d["spectral_usable"] and not d["gradient_bound_usable"], d
    reason = d["not_usable_reason"]
    assert "under convergence_norm='interface'" in reason and "is not in 0.4.0" in reason, reason
    assert "'mixed' or 'l2'" in reason and "precision_limited" in reason, reason
    return d


def test_markers_that_drift_past_what_their_dtype_resolves_are_flagged_at_every_step():
    """Compiled 6 spacings from zero (no advisory: the count is 0.6 of the
    tolerance), the markers are then carried 950 spacings a step by their
    own update.  From the second step the positions are 90 threshold
    distances out and the float32 pair stalls where it is, off its fixed
    point by more than a tolerance on some step (the positions cannot be
    held to the tolerance there: one float32 rounding of them is 6 to 46
    tolerances).  The report says so on every one: the floor is that of
    the returned state (117 tolerances and rising), the residual is at or
    below it, ``precision_limited`` is ``True``.  A flag taken from the
    state ``compile()`` saw reads ``False`` throughout (the seeded
    fault)."""
    gm, advisories = build(DRIFTING)
    assert advisories == []
    worst, floors = 0.0, []
    for step in range(4):
        pre = stored(gm)
        gm.step()
        d = _checked(DRIFTING, gm, pre)
        post = stored(gm)
        reach = float(np.max(np.abs(post["markers"]["pos"]))) / DRIFTING.spacing[0]
        assert reach > (0.9 * 950.0) * (step + 1)
        assert d["precision_limited"] is True, (step, d)
        assert d["residual_precision_floor"] > 10.0, (step, d)
        floors.append(d["residual_precision_floor"])
        if step:
            star = fixed_point(DRIFTING, pre)
            worst = max(worst, scaled_change(DRIFTING, post, star, pre, reference=star))
    assert floors == sorted(floors) and floors[-1] > 3.0 * floors[0], floors
    # Premise: the group is really off where it says converged (1 float32
    # rounding of a position 1000 spacings out is 12 tolerances).
    assert worst > 1.0, worst
    # The table and the printed report carry the flag and the floor.
    (row,) = list(gm.coupling_report())
    flags = " | ".join(row["flags"])
    assert "precision_limited=True" in flags and "residual_precision_floor=" in flags, flags


def test_the_same_drift_in_float64_reads_a_floor_far_under_the_tolerance():
    """The control: every field in float64.  The floor the report gives is
    a millionth of a tolerance at every distance, and the pair is within
    a thousandth of a tolerance of its fixed point.  The flag alone does
    not tell the two pairs apart: carried 950 spacings a step, the
    markers deposit far from where they sampled, the pass settles exactly
    and the residual is 0.0 in either dtype, which is at or below any
    floor.  The floor's size against the tolerance is the reading."""
    pair = dataclasses.replace(DRIFTING, dtype="float64")
    gm, advisories = build(pair)
    assert advisories == []
    for step in range(3):
        pre = stored(gm)
        gm.step()
        d = _checked(pair, gm, pre)
        assert d["converged"] is True and 0.0 < d["residual_precision_floor"] < 1e-4, (step, d)
        star = fixed_point(pair, pre)
        assert scaled_change(pair, stored(gm), star, pre, reference=star) < 1e-3


def test_float64_positions_beside_float32_fields_are_counted_at_the_groups_coarsest_dtype():
    """The floor counts every entry of a group at the coarsest floating
    dtype among its fields (CPL-100: a field computed from a coarser
    member's output may carry that member's rounding), positions among
    them.  So float64 positions beside float32 values are counted at
    float32's rounding by the report -- flagged, with a floor of hundreds
    of tolerances -- although these positions, which the markers' update
    increments, settle to a thousandth of a tolerance, and although
    ``compile()``'s advisory, which speaks of the dtype the positions are
    stored in, is silent at the far state too.  The conservative
    direction, stated in the guide; pinned so that a change of either
    rule shows."""
    pair = dataclasses.replace(DRIFTING, pos_dtype="float64")
    gm, advisories = build(pair)
    assert advisories == []
    for _ in range(2):
        pre = stored(gm)
        gm.step()
        d = _checked(pair, gm, pre)
    assert d["precision_limited"] is True and d["residual_precision_floor"] > 50.0, d
    star = fixed_point(pair, pre)
    post = stored(gm)
    off = np.max(np.abs(post["markers"]["pos"] - star["markers"]["pos"])) / pair.spacing[0]
    assert off / pair.rtol < 0.1, off / pair.rtol
    far = dataclasses.replace(pair, u0=float(np.max(post["markers"]["pos"])) / pair.spacing[0])
    assert build(far)[1] == []


def _written_and_loaded(anchors, tmp_path):
    pair = Pair(anchors=anchors)
    gm, advisories = build(pair)
    assert advisories == []                       # 6 spacings out: resolved
    pre = stored(gm)
    gm.step()
    near = _checked(pair, gm, pre)
    # Written 7000 spacings out: no warning of any kind, and the next step
    # reads the floor where the positions now are.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        write_far(gm, pair, 7000.0)
        assert same_reports(report(gm), near)     # the report describes the step that ran
        pre = stored(gm)
        gm.step()
    assert [str(w.message) for w in caught if ADVISORY in str(w.message)] == []
    far = _checked(pair, gm, pre)
    # Every pair here rests on stored positions: its gather is read as
    # delivered, at the markers' positions or at the grid node's.
    assert any(r > 100 * coarsest_eps(pair) for _n, r in floor_parts(pair, pre, stored(gm)))
    assert far["residual_precision_floor"] > 100.0 * max(near["residual_precision_floor"], 0.2)
    assert far["precision_limited"] is True, far
    # A checkpoint of the far state, loaded into a graph compiled near
    # zero: the state, the report and the next step are the ones of the
    # graph that was never reloaded, to the bit.
    path = gm.save_state(tmp_path / "far.npz")
    other, _ = build(pair)
    other.load_state(path)
    assert same_reports(report(other), far)
    pre = stored(gm)
    gm.step()
    other.step()
    again = _checked(pair, gm, pre)
    assert same_reports(report(other), again), (report(other), again)
    for name in ("grid", "markers"):
        for field, value in gm.get_node_state(name).items():
            assert np.asarray(value).tobytes() == np.asarray(
                other.get_node_state(name)[field]).tobytes(), (name, field)
    # Whatever the loop did there (it can run to its cap: one position
    # alternating between two neighbouring float32 numbers is 49
    # tolerances a pass), the report explains it.
    assert again["precision_limited"] is True, again
    assert again["converged"] or again["iterations"] == pair.cap, again
    return far, again


@pytest.mark.parametrize("anchors", ALL_ANCHORS[:2], ids="-".join)
def test_positions_written_or_loaded_far_from_zero_are_flagged_by_the_next_step(anchors, tmp_path):
    """A state write and a loaded checkpoint put the group where
    ``compile()`` would have warned.  The step after says so in its
    report, on both paths the floor reaches it by: recorded by the step
    (a gather anchored at its target, whose pre-step positions the
    returned state does not hold) and measured on the returned state (a
    gather anchored at its source).  The loaded graph does what the
    graph that was never reloaded does, to the bit."""
    _written_and_loaded(anchors, tmp_path)


# Per push: tests/property/test_coupling_geometry_interface_positions_floor.py::test_positions_written_or_loaded_far_from_zero_are_flagged_by_the_next_step
@pytest.mark.slow
@pytest.mark.parametrize("anchors", ALL_ANCHORS[2:], ids="-".join)
def test_positions_written_or_loaded_far_are_flagged_on_the_other_anchors(anchors, tmp_path):
    _written_and_loaded(anchors, tmp_path)


def test_compile_on_the_far_state_gives_the_advisory_the_report_stands_in_for():
    """The same question asked at ``compile()``: the pair started where the
    write puts it is warned of on both edges."""
    assert warned(build(Pair(u0=7002.0))[1]) == {SCATTER: "part", GATHER: "delivered"}


def test_a_written_state_in_float64_is_not_flagged():
    """The control of the state write: the same positions, every field in
    float64."""
    pair = Pair(dtype="float64")
    gm, _ = build(pair)
    write_far(gm, pair, 7000.0)
    pre = stored(gm)
    gm.step()
    d = _checked(pair, gm, pre)
    assert d["converged"] is True and d["precision_limited"] is False, d


def test_the_floor_of_a_report_no_step_measured_is_not_reported_and_says_why():
    """Where the floor could only have been measured by the step (a gather
    at its target's pre-step positions) and the state carries none, the
    entry reports neither the floor nor the flag, and its reason says
    which numbers are missing and why."""
    pair = Pair()
    gm, _ = build(pair)
    gm.step()
    slot = f"coupling_{KEY}_reading_floor"
    meta = dict(gm._state["_meta"])                                           # noqa: SLF001
    meta[slot] = jnp.asarray(jnp.nan, meta[slot].dtype)
    gm._state = {**gm._state, "_meta": meta}                                   # noqa: SLF001
    d = report(gm)
    assert d["precision_limited"] is False and math.isnan(d["residual_precision_floor"]), d
    reason = d["not_usable_reason"]
    assert "under convergence_norm='interface'" in reason, reason
    assert "is not reported either" in reason and "only the step that solved" in reason, reason


@pytest.mark.parametrize("norm", ["mixed", "l2"])
def test_the_other_norms_reports_are_what_they_were(norm):
    """Only a group withheld on account of the interface norm carries the
    floor's key: the same pair under a norm whose diagnostics read the
    geometry reports as every group does."""
    gm, advisories = build(Pair(), norm=norm)
    assert advisories == []
    gm.step()
    d = report(gm)
    assert "residual_precision_floor" not in d and "not_usable_reason" not in d, d


# ---------------------------------------------------------------------------
# B. What the count assumes for a delivered value (MADD-ANO-247)
# ---------------------------------------------------------------------------


def level_set(ratio: float, *, pos_dtype="float32", u0: float = 100.2) -> Pair:
    """A gather that samples a ramp of slope one per cell near its zero:
    the values delivered are ``1 / ratio`` of the field's variation
    across a cell.  The markers are a relay of what they sample and the
    grid takes a twentieth of their deposit; the positions do not move."""
    off = 0.5 / ratio
    return Pair(n=160, m=4, spacing=(0.3,), dtype="float32", pos_dtype=pos_dtype, rtol=1e-4,
                cap=200, u0=u0, push=0.0, grid_update=(1.0, 0.0, 0.05),
                markers_update=(0.0, 0.0, 1.0), field="ramp", ramp_zero=u0, f0=off,
                offsets=(-off, 0.6 * off, off, 1.9 * off))


def _exact_residual_against_the_floor(pair: Pair):
    """``(exact residual of the returned state, reported + floor, report,
    advisories)``: the float64 pass and reading from the stored numbers
    against what the library says a residual can be that is rounding."""
    gm, advisories = build(pair)
    pre = stored(gm)
    gm.step()
    post, d = stored(gm), report(gm)
    exact = scaled_change(pair, one_pass(pair, post, pre), post, pre)
    return exact, d["residual"] + d["residual_precision_floor"], d, advisories


def _count(pair: Pair) -> float:
    eps = float(np.finfo(np.dtype(pair.geometry_dtype)).eps)
    return ULPS * EVALUATIONS * eps * float(
        np.max(np.abs(pair.start_positions()))) / pair.spacing[0] / pair.rtol


@pytest.mark.parametrize("ratio", [250.0, 2500.0])
@pytest.mark.xfail(strict=True, reason=(
    "MADD-ANO-247 (open, 0.5.0), entered through the positions: the floor of a delivered "
    "value counts eps |u| of the value's own magnitude, which assumes a field that varies "
    "across a cell by about the size of the value delivered; a gather that samples a field "
    "near its zero is moved by eps |u| times the field's variation over the value"))
def test_the_floor_covers_a_gather_that_samples_a_field_near_its_zero(ratio):
    """What the floor is for: the exact residual of the returned state is
    no more than the reported one plus the floor.  100 spacings from zero
    (the count is 0.96: no advisory), with the field varying across a
    cell by 250 and 2500 times the value delivered, the exact residual is
    6.3 and 22.5 tolerances beside 0.35 and 0.49 reported and a floor of
    0.78, and the group reports ``converged=True`` 7.3 and 23 tolerances
    from its fixed point (jaxlib 0.11.0, CPU)."""
    exact, allowed, _d, _advisories = _exact_residual_against_the_floor(level_set(ratio))
    assert exact <= allowed, (exact, allowed)


@pytest.mark.parametrize("ratio", [250.0, 2500.0])
def test_a_gather_near_a_fields_zero_is_not_warned_of_and_float64_positions_hold_it(ratio):
    """The controls beside the pin, and the documented reading of it: the
    advisory's count is under the tolerance (0.96) and ``compile()`` is
    silent, the solve reports ``converged=True`` with a residual under
    one, and the same pair with its positions in float64 is inside the
    reported residual plus the floor."""
    narrow = level_set(ratio)
    assert 0.9 < _count(narrow) < 1.0
    exact, allowed, d, advisories = _exact_residual_against_the_floor(narrow)
    assert advisories == [] and d["converged"] is True and d["residual"] < 1.0, d
    assert exact > 2.0 * allowed, (exact, allowed)      # premise of the pin, with a margin
    wide = level_set(ratio, pos_dtype="float64")
    exact, allowed, d, advisories = _exact_residual_against_the_floor(wide)
    assert advisories == [] and d["converged"] is True, d
    assert exact <= allowed, (exact, allowed)


def test_the_count_holds_where_the_field_varies_by_about_the_value_delivered():
    """The count's premise, at a ratio of 1.7: the float32 pair at the
    same distance is inside the reported residual plus the floor."""
    exact, allowed, d, advisories = _exact_residual_against_the_floor(level_set(5.0 / 3.0))
    assert advisories == [] and d["converged"] is True, d
    assert exact <= allowed, (exact, allowed)


def test_the_advisory_states_what_its_count_assumes_and_claims_no_worst_case():
    """The text of the advisory for a delivered value: the assumption, the
    two directions it errs in, the open anomaly, and where the run-time
    reading is.  Not "the worst case", and not a distance that "resolves
    this tolerance"."""
    far = Pair(u0=400.0, n=1000, rtol=1e-4)
    _gm, advisories = build(far)
    by_edge = {key: text for key, text in zip(warned(advisories), advisories)}
    assert warned(advisories) == {SCATTER: "part", GATHER: "delivered"}
    text = by_edge[GATHER]
    for phrase in ("assumes a field that varies across one cell by about the size of the value "
                   "delivered", "the warning is then early", "a field sampled near its zero",
                   "MADD-ANO-247", "asked once, of the state compile() sees",
                   "precision_limited and residual_precision_floor"):
        assert phrase in text, (phrase, text)
    for text in advisories:
        assert "worst case" not in text and "resolves this tolerance" not in text, text
        assert "count is under the tolerance within" in text, text


# ---------------------------------------------------------------------------
# C. The side is decided by entry counts
# ---------------------------------------------------------------------------

#: Forty markers on twelve grid entries, the lattice 400 spacings from
#: zero and the markers inside it.
CROWDED = dict(n=12, m=40, origin=400.0, u0=401.0, rtol=1e-4,
               offsets=tuple(9.0 * k / 39 for k in range(40)), push=0.01)


@pytest.mark.parametrize("anchors", ALL_ANCHORS, ids="-".join)
def test_with_more_markers_than_grid_entries_the_gather_is_read_at_its_source(anchors):
    """The library decides the side an edge is read on by the entry counts
    its mapping declares.  With more markers than grid entries the gather
    is the edge that delivers more than it reads: it is read at the grid
    field (kept as the accepted iterate holds it) and, anchored at its
    source, at the grid node's positions as a part; the scatter is read
    as delivered, and the markers' value and positions are no part.  The
    advisory follows: the positions part is the gather's, the delivered
    value the scatter's, and the float floor is the rule's."""
    from maddening.core.coupling import _interface_plan  # noqa: PLC0415

    pair = Pair(anchors=anchors, **CROWDED)
    gm, advisories = build(pair)
    records = {r.key: r for r in _interface_plan.interface_records(
        gm._committed_floor_inputs[KEY][2], gm._state)}                        # noqa: SLF001
    gather = [(p.what, ".".join(p.field), p.whole) for p in records[GATHER].parts]
    scatter = [(p.what, ".".join(p.field), p.whole) for p in records[SCATTER].parts]
    assert scatter == [("delivered", "markers.f", False)]
    want = [("source", "grid.x", True)]
    if anchors[1] == "source":
        want.append(("geometry", "grid.gp", True))
    assert gather == want
    expected = {SCATTER: "delivered"}
    if anchors[1] == "source":
        expected[GATHER] = "part"
    assert warned(advisories) == expected
    for text in advisories:
        holder = "grid.gp" if (GATHER in text or anchors[0] == "target") else "markers.pos"
        assert f"positions {holder}" in text, text
    pre = stored(gm)
    gm.step()
    _checked(pair, gm, pre)


def test_with_fewer_markers_the_scatter_is_the_edge_read_at_its_source():
    """The usual proportion, as the control: the same pair with four
    markers on a thousand entries is read the other way round."""
    from maddening.core.coupling import _interface_plan  # noqa: PLC0415

    gm, advisories = build(Pair(n=1000, u0=400.0, rtol=1e-4))
    records = {r.key: r for r in _interface_plan.interface_records(
        gm._committed_floor_inputs[KEY][2], gm._state)}                        # noqa: SLF001
    assert [(p.what, ".".join(p.field)) for p in records[SCATTER].parts] == [
        ("source", "markers.f"), ("geometry", "markers.pos")]
    assert [(p.what, ".".join(p.field)) for p in records[GATHER].parts] == [
        ("delivered", "grid.x")]
    assert warned(advisories) == {SCATTER: "part", GATHER: "delivered"}


# ---------------------------------------------------------------------------
# D. The advisory takes the floor's decisions
# ---------------------------------------------------------------------------

#: float32 positions 400 spacings from zero on a lattice that holds them,
#: at ``rtol=1e-4``: 3.8 times the tolerance.
FAR = Pair(n=1000, u0=400.0, rtol=1e-4, anchors=("target", "target"))


def _library_floor(gm, pair: Pair) -> float:
    state = {name: gm.get_node_state(name) for name in ("grid", "markers")}
    return float(residual_precision_floor(
        state, ["grid", "markers"], "interface", pair.atol, pair.rtol,
        list(gm._committed_floor_inputs[KEY][2]), evaluations=EVALUATIONS,     # noqa: SLF001
        pre_step=state))


def test_a_delivered_value_inside_the_dead_band_is_not_warned_of():
    """With ``atol`` above every magnitude the norm does not read the
    delivered value and the floor of the group is 0.0: no advisory quotes
    a floor for it.  At ``atol=0`` the same pair is warned of, with the
    floor the library computes for the state."""
    gm, advisories = build(FAR)
    assert warned(advisories) == {GATHER: "delivered"}
    said = float(advisories[0].split("which is ")[1].split(" times the tolerance")[0])
    assert said == pytest.approx(_library_floor(gm, FAR) * math.sqrt(2.0), rel=5e-3)
    banded = dataclasses.replace(FAR, atol=50.0)
    gm, advisories = build(banded)
    assert advisories == []
    assert _library_floor(gm, banded) == 0.0


def test_a_value_that_is_zero_at_compile_is_still_asked_under_a_dead_band():
    """A field not yet computed has no magnitude to compare with ``atol``:
    the floor counts it from its first non-zero value, so the advisory
    asks it, with a dead band declared or without one."""
    for atol in (0.0, 50.0):
        _gm, advisories = build(dataclasses.replace(FAR, atol=atol, field="zero"))
        assert warned(advisories) == {GATHER: "delivered"}, (atol, advisories)


def test_the_kernel_says_which_coordinates_a_transfer_depends_on():
    """``geometry_coordinates_read``: not a coordinate on an axis of one
    lattice point; not one outside the hull by more than ``sqrt(eps)`` of
    its own distance from zero; every other, a point on a face and one
    within its rounding of it included, and a non-finite one."""
    for dtype in ("float32", "float64"):
        eps = float(np.finfo(dtype).eps)
        mapping = multilinear_grid_mapping((50.0, 0.0), (0.5, 1e-3), (40, 1), n_points=6,
                                           mode="consistent")
        top = 50.0 + 39 * 0.5
        just_out = top + 0.5 * math.sqrt(eps) * (top / 0.5) * 0.5      # half the slack
        well_out = top + 4.0 * math.sqrt(eps) * (top / 0.5) * 0.5      # four times it
        x = np.asarray([55.2, 50.0, top, just_out, well_out, 20.0])
        # On the one lattice point of the second axis, beside it and far from it.
        y = np.asarray([2.0, 0.0, 1e-9, -1e-3, 2.0, 0.0])
        geom = jnp.asarray(np.stack([x, y], axis=1), dtype)
        read = np.asarray(mapping.geometry_coordinates_read(geom))
        assert read.shape == (6, 2) and read.dtype == bool
        assert read[:, 0].tolist() == [True, True, True, True, False, False], (dtype, read)
        assert not read[:, 1].any()
        broken = geom.at[0, 0].set(jnp.nan).at[1, 0].set(jnp.inf)
        assert np.asarray(mapping.geometry_coordinates_read(broken))[:2, 0].all()
        one_axis = multilinear_grid_mapping((0.0,), (0.5,), (40,), n_points=3, mode="conservative")
        flat = jnp.asarray([1.0, -3.0, 25.0], dtype)
        assert np.asarray(one_axis.geometry_coordinates_read(flat)).tolist() == [True, False, False]


def test_a_coordinate_on_an_axis_of_one_lattice_point_puts_nothing_into_a_delivered_value():
    """A lattice of 40 by 1 points: the stencil does not read the second
    coordinate, and its spacing is an arbitrary number.  Markers 2000 of
    those spacings from zero are not warned of for the value gathered at
    them, and the floor of that value does not count them.  The true
    alarm beside it stays (the next test).  Positions that are a part of
    the reading are asked of every coordinate, as the criterion reads
    them: the scatter anchored at its source is warned of, and says which
    axis."""
    flat = dict(n=40, m=3, shape=(40, 1), spacing=(0.5, 1e-3), rtol=1e-4, u0=3.5,
                offsets=(0.0, 1.0, 2.0), across=2.0)
    gathered = Pair(anchors=("target", "target"), **flat)
    gm, advisories = build(gathered)
    assert advisories == []
    assert _library_floor(gm, gathered) < 0.1
    pre = stored(gm)
    gm.step()
    _checked(gathered, gm, pre)
    read = Pair(anchors=("source", "target"), **flat)
    _gm, advisories = build(read)
    assert warned(advisories) == {SCATTER: "part"}
    assert "2000 spacings from zero (axis 1, spacing 0.001)" in advisories[0], advisories[0]


def test_the_true_alarms_beside_a_one_point_axis_and_a_clamped_point_stay():
    """What the two false alarms sit beside: markers 400 spacings out
    along the axis of the lattice that has points, and markers 400
    spacings out on a lattice long enough to hold them, are warned of
    for the value gathered at them."""
    along = Pair(n=1000, m=3, shape=(1000, 1), spacing=(0.5, 1e-3), rtol=1e-4, u0=400.0,
                 offsets=(0.0, 1.0, 2.0), across=2.0, anchors=("target", "target"))
    assert warned(build(along)[1]) == {GATHER: "delivered"}
    inside = Pair(n=1000, m=3, rtol=1e-4, u0=400.5, offsets=(0.0, 1.0, 2.0),
                  anchors=("target", "target"))
    assert warned(build(inside)[1]) == {GATHER: "delivered"}


def test_points_clamped_to_the_hull_put_nothing_into_a_delivered_value():
    """Markers 360 spacings beyond the last lattice point are clamped to
    it: the weights are exactly 0 and 1 whatever the last digits of the
    position, the value gathered there does not depend on them, and
    neither the advisory nor the floor counts them.  The same markers on
    a lattice long enough to hold them are warned of (the true alarm:
    the test above), and so are markers clamped from within their own
    rounding of the face, which a rounding can bring inside."""
    clamped = Pair(n=40, m=3, rtol=1e-4, u0=400.5, offsets=(0.0, 1.0, 2.0),
                   anchors=("target", "target"))
    gm, advisories = build(clamped)
    assert advisories == []
    assert _library_floor(gm, clamped) < 0.1
    pre = stored(gm)
    gm.step()
    _checked(clamped, gm, pre)
    # On the face of a lattice 400 spacings from zero, and a tenth of the
    # slack past it: read, and warned of.
    edge = Pair(n=40, m=3, rtol=1e-4, origin=361.0, u0=400.0,
                offsets=(-2.0, -1.0, 0.004), anchors=("target", "target"))
    assert warned(build(edge)[1]) == {GATHER: "delivered"}


# ---------------------------------------------------------------------------
# The claim is first order: a lattice plane ends it
# ---------------------------------------------------------------------------

_PLANE = 6           # the lattice point the field's slope changes at


class _KinkedGrid(SimulationNode):
    """Holds a field whose slope is *left* per cell up to lattice point 6
    and *right* beyond it, and one probe position ``gp <- c0 + d``."""

    def __init__(self, name, timestep, left, right, c0, start):
        super().__init__(name, timestep)
        i = np.arange(12.0)
        self._x = np.where(i <= _PLANE, 1.0 + left * (i - _PLANE), 1.0 + right * (i - _PLANE))
        self._c0, self._start = c0, start

    def initial_state(self):
        return {"x": jnp.asarray(self._x), "gp": jnp.asarray([[self._start]])}

    def boundary_input_spec(self):
        return {"d": BoundaryInputSpec(shape=(1,), dtype=jnp.float64)}

    def update(self, state, boundary_inputs, dt):
        return {"x": state["x"], "gp": (self._c0 + boundary_inputs["d"])[:, None]}


class _Relay(SimulationNode):
    def __init__(self, name, timestep, start):
        super().__init__(name, timestep)
        self._start = start

    def initial_state(self):
        return {"f": jnp.asarray([self._start])}

    def boundary_input_spec(self):
        return {"sampled": BoundaryInputSpec(shape=(1,), dtype=jnp.float64)}

    def update(self, state, boundary_inputs, dt):
        return {"f": boundary_inputs["sampled"]}


def _across_a_plane(left: float, right: float, c0: float, start: float, rtol: float = 1e-4):
    """``f <- phi(c0 + f)``, ``phi`` the lattice's interpolant of the
    kinked field: one step of the library, and the float64 fixed point.
    Returns the report, the cells of the returned and of the fixed
    position, the distance of the returned value from the fixed point in
    tolerances, and ``K = 1 / (1 - slope)`` in the fixed point's cell."""
    gm = GraphManager()
    grid = _KinkedGrid("grid", 0.01, left, right, c0, c0 + start)
    gm.add_node(grid)
    gm.add_node(_Relay("markers", 0.01, start))
    gm.add_edge("grid", "markers", "x", "sampled", geometry=("source", "gp"),
                mapping=multilinear_grid_mapping((0.0,), (1.0,), (12,), n_points=1,
                                                 mode="consistent"))
    gm.add_edge("markers", "grid", "f", "d")
    gm.add_coupling_group(["grid", "markers"], convergence_norm="interface", rtol=rtol,
                          iteration_mode="gauss-seidel", max_iterations=400)
    gm.compile()
    gm.step()
    x = grid._x                                                                # noqa: SLF001

    def phi(u):
        j = int(min(math.floor(u), 10))
        return (1.0 - (u - j)) * x[j] + (u - j) * x[j + 1]

    star = start
    for _ in range(100_000):
        nxt = phi(c0 + star)
        if abs(nxt - star) < 1e-15:
            break
        star = nxt
    else:
        raise AssertionError("the float64 iteration did not settle")
    f = float(gm.get_node_state("markers")["f"][0])
    gp = float(gm.get_node_state("grid")["gp"][0, 0])
    slope = left if c0 + star <= _PLANE else right
    return dict(report=report(gm), cell=int(gp > _PLANE), star_cell=int(c0 + star > _PLANE),
                distance=abs(f - star) / (rtol * abs(star)), K=1.0 / (1.0 - slope))


@pytest.mark.parametrize("start, tolerances", [(0.5, 250.0), (0.9, 50.0)])
def test_a_jump_across_a_lattice_plane_into_a_slow_cell_reads_converged_far_from_the_fixed_point(
        start, tolerances):
    """The condition of MAP-050's claim, constructed.  One point gathers a
    field whose slope is -0.05 per cell left of a lattice plane and 0.999
    right of it; the point's position is the value it gathers plus a
    constant, so the pass contracts at 0.05 on one side and at 0.999 on
    the other.  The fixed point is 3.8e-5 of a spacing left of the plane
    (``K = 0.95``).  Started half a unit under it, the first pass throws
    the position 0.025 of a spacing past the plane, where the next pass
    moves it by 0.63 tolerances: the loop, which has seen one large step
    and one small one, accepts.  ``converged=True``, residual under one,
    250 tolerances from the fixed point.  Started at 0.9, 50 tolerances.
    The linearisation at the fixed point says nothing of an iterate in
    another lattice cell."""
    out = _across_a_plane(-0.05, 0.999, 4.99996, start)
    assert out["report"]["converged"] is True and out["report"]["residual"] < 1.0, out
    assert (out["cell"], out["star_cell"]) == (1, 0), out
    assert out["K"] == pytest.approx(1.0 / 1.05)
    assert 0.8 * tolerances < out["distance"] < 1.2 * tolerances, out


@pytest.mark.parametrize("left, right, c0", [(-0.05, -0.05, 4.99996), (-0.05, 0.999, 4.9),
                                             (-0.05, 0.5, 4.99996)],
                         ids=["no-kink", "no-crossing", "a-faster-cell"])
def test_within_one_lattice_cell_or_a_cell_of_a_like_rate_the_claim_holds(left, right, c0):
    """The controls: the same pair with one slope on both sides, with the
    first pass landing short of the plane, and with the cell beyond the
    plane contracting at 0.5: ``converged=True`` within a tolerance."""
    out = _across_a_plane(left, right, c0, 0.5)
    assert out["report"]["converged"] is True and out["distance"] < 1.0, out
