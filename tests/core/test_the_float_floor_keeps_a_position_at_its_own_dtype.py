"""A position keeps its own dtype's eps in a group's float floor; values take the group's coarsest.

The float floor of a group of mixed dtypes counts every **value** entry
at the ``eps`` of the group's coarsest floating dtype
(``acceleration._group_coarsest_eps``).  A **position** is exempt: a
geometry field that an internal edge anchors a geometry-dependent
mapping at keeps its own dtype's ``eps`` under ``"l2"`` and ``"mixed"``
(``acceleration._position_fields``), as a position part read in kernel
lengths does under ``"interface"`` (held there by
``tests/property/test_coupling_geometry_interface_norm.py``).  A stored
float64 position has float64 resolution; what a float32 member adds to
it in a pass is a rounding of the increment.

**The corner this leaves**, characterised below: a float64 position
that a member computes within the pass *purely* from float32 data is as
coarse as that data and is counted finer than it is.  On the pair here
(three markers whose positions are set, every pass, from the float32
values they sample; float32 fields; stalled at the float floor) the
bound still holds, by hundreds, because the evaluation count carries
the measured gain of the read through the scatter; the exemption lowers
it by 5%.  That is a measurement on one pair, not a proof for every
group.
"""

from __future__ import annotations

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _group_layout, reason_codes
from maddening.core.coupling.acceleration import (
    PRECISION_FLOOR_ULPS,
    _position_fields,
    residual_precision_floor,
)
from maddening.core.coupling.grid_mapping import multilinear_grid_mapping
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.property import coupling_reference as cr
from tests.sparse_mapping_support import x64

M, N = 3, 30
CELLS = np.array([4, 13, 22]); P0 = CELLS + 0.4
G = np.zeros((M, N)); G[np.arange(M), CELLS] = 0.6; G[np.arange(M), CELLS + 1] = 0.4
_A = np.random.default_rng(5).normal(size=(M, M))
A = _A * (0.9 / np.max(np.abs(np.linalg.eigvals(_A @ G @ G.T))))
B = np.array([1.0, 1.5, 2.0]); C = np.linspace(0.5, 1.0, N); Q0 = np.linspace(1.0, 3.0, N)
SHIFT = 0.02
KEY = "p+q"


class Markers(SimulationNode):
    """x <- b + A u;  pos <- P0 + SHIFT * u, computed in the VALUE dtype and stored in the position dtype."""
    def __init__(self, name, dt, vdt, pdt):
        super().__init__(name, dt); self._v, self._p = jnp.dtype(vdt), jnp.dtype(pdt)
    def initial_state(self):
        return {"x": jnp.asarray(B, self._v), "pos": jnp.asarray(P0.reshape(M, 1), self._p)}
    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(M,), dtype=self._v, default=jnp.zeros(M, self._v))}
    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"]
        x = (jnp.asarray(B, self._v) + jnp.asarray(A, self._v) @ u).astype(self._v)
        pos = (jnp.asarray(P0, self._v) + jnp.asarray(SHIFT, self._v) * u).astype(self._p).reshape(M, 1)
        return {"x": x, "pos": pos}
    def update_evaluations(self):
        return 1


class Grid(SimulationNode):
    def __init__(self, name, dt, vdt):
        super().__init__(name, dt); self._v = jnp.dtype(vdt)
    def initial_state(self):
        return {"x": jnp.asarray(Q0, self._v)}
    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(N,), dtype=self._v, default=jnp.zeros(N, self._v))}
    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"x": (jnp.asarray(C, self._v) + jnp.asarray(0.5, self._v) * state["x"] + boundary_inputs["u"]).astype(self._v)}
    def update_evaluations(self):
        return 1


def build(vdt, pdt, norm, schedule, rtol=None, **knobs):
    gm = GraphManager()
    gm.add_node(Markers("p", 1.0, vdt, pdt)); gm.add_node(Grid("q", 1.0, vdt))
    gm.add_edge("p", "q", "x", "u", mapping=multilinear_grid_mapping([0.0], [1.0], [N], n_points=M, mode="conservative"),
                geometry=("source", "pos"))
    gm.add_edge("q", "p", "x", "u", mapping=matrix_mapping(jnp.asarray(G, vdt)))
    tol = {} if rtol is None else ({"tolerance": rtol * math.sqrt(M + M + N)} if norm == "l2" else {"rtol": rtol})
    gm.add_coupling_group(["p", "q"], convergence_norm=norm, solver="ift", **{"iteration_mode": schedule, **tol, **knobs})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore"); gm.compile()
    return gm


def members(gm):
    return {n: {f: np.asarray(v) for f, v in gm.get_node_state(n).items()} for n in ("p", "q")}


def _edges(vdt="float32"):
    return [EdgeSpec("p", "q", "x", "u", mapping=multilinear_grid_mapping(
                [0.0], [1.0], [N], n_points=M, mode="conservative"), geometry=("source", "pos")),
            EdgeSpec("q", "p", "x", "u", mapping=matrix_mapping(jnp.asarray(G, vdt)))]


def test_the_position_fields_are_the_ones_an_internal_edge_anchors_its_geometry_at():
    assert _position_fields(_edges()) == frozenset({("p", "pos")})
    assert _position_fields([EdgeSpec("q", "p", "x", "u")]) == frozenset()
    assert _position_fields(None) == frozenset() and _position_fields(()) == frozenset()


@pytest.mark.parametrize("norm", ["mixed", "l2"])
def test_a_float64_position_field_beside_float32_values_is_floored_at_its_own_eps(norm):
    """Closed form: 33 float32 value entries at float32's eps and three
    float64 positions at float64's.  Without the edge that anchors its
    geometry there, the same float64 field is a value like any other and
    takes the group's coarsest eps; and float32 positions beside float64
    values put the values at float32's eps (the coarsest rule) and stay
    at their own.  Those float32 positions are up to 22.4 spacings from
    the grid's first point, and what the scatter delivers at them is no
    finer than its kernel's weights: the values are counted at one
    rounding of that lattice coordinate over ``PRECISION_FLOOR_ULPS``
    (5.6 float32 eps; MADD-ANO-261), where float64 positions leave them
    at float32's."""
    eps32, eps64 = (float(np.finfo(t).eps) for t in (np.float32, np.float64))
    with x64(True):
        state = {"p": {"x": jnp.ones(M, jnp.float32), "pos": jnp.asarray(P0.reshape(M, 1))},
                 "q": {"x": jnp.ones(N, jnp.float32)}}
        got = float(residual_precision_floor(state, ["p", "q"], norm, rtol=1e-3,
                                             interface_edges=_edges()))
        plain = float(residual_precision_floor(state, ["p", "q"], norm, rtol=1e-3))
        reverse = {"p": {"x": jnp.ones(M, jnp.float64), "pos": jnp.asarray(P0.reshape(M, 1), jnp.float32)},
                   "q": {"x": jnp.ones(N, jnp.float64)}}
        coarse_positions = float(residual_precision_floor(
            reverse, ["p", "q"], norm, rtol=1e-3, interface_edges=_edges("float64")))
    values, positions = M + N, M

    def pooled(eps_values, eps_positions):
        total = values * eps_values ** 2 + positions * eps_positions ** 2
        return PRECISION_FLOOR_ULPS * (math.sqrt(total) if norm == "l2"
                                       else math.sqrt(total / (values + positions)) / 1e-3)

    assert got == pytest.approx(pooled(eps32, eps64), rel=1e-6)
    assert plain == pytest.approx(pooled(eps32, eps32), rel=1e-6)
    assert coarse_positions == pytest.approx(
        pooled(eps32 * float(P0.max()) / PRECISION_FLOOR_ULPS, eps32), rel=1e-6)
    assert float(P0.max()) > PRECISION_FLOOR_ULPS
    assert got < plain


def _stalled(vdt, pdt, schedule="gauss-seidel", norm="mixed", rtol=1e-7):
    """One step of the pair at a tolerance below float32's floor, and its
    true distance in the report's units against the fixed point of a
    float64 twin's pass (``coupling_reference``)."""
    twin = build("float64", "float64", norm, schedule, **cr.twin_knobs({"iteration_mode": schedule}))
    reference = cr.PassReference.of(twin)
    gm = build(vdt, pdt, norm, schedule, rtol=rtol, max_iterations=4000, diagnostics=True)
    pre = members(gm)
    gm.step()
    report = dict(gm.coupling_diagnostics()[KEY])
    got = members(gm)
    for name, fields in pre.items():
        twin.set_node_state(name, {f: jnp.asarray(v, jnp.float64) for f, v in fields.items()})
    bound = reference.at(twin._state, twin.params)      # noqa: SLF001
    x = bound.flat({n: {f: np.asarray(v, np.float64) for f, v in fs.items()}
                    for n, fs in got.items()})
    fixed = bound.fixed_point(x)
    assert fixed.converged, fixed.history
    terms = []
    for node, field in (("p", "x"), ("p", "pos"), ("q", "x")):
        star = np.asarray(bound.field(fixed.x, node, field))
        have = np.asarray(bound.field(x, node, field))
        terms.append(np.abs(have - star).ravel() / np.max(np.abs(have)))
    report["distance"] = float(np.sqrt(np.mean(np.concatenate(terms) ** 2))) / rtol
    report["graph"] = gm
    return report


# Per push: tests/core/test_the_float_floor_keeps_a_position_at_its_own_dtype.py::test_a_float64_position_field_beside_float32_values_is_floored_at_its_own_eps
@pytest.mark.slow
def test_float64_positions_set_from_float32_data_within_the_pass_still_stand_on_a_bound_that_holds(
        monkeypatch):
    """The corner, characterised (not a proof): positions stored in
    float64 and recomputed every pass from float32 samples carry float32
    rounding, and the floor counts them at float64's eps.  On this pair
    the bound is hundreds of times the distance, and it is the lower of
    the two against the same pair with float32 positions: by a few
    percent while nothing counted the weights the kernel forms from
    those positions, and by the factor that count takes now (float32
    positions up to 22.4 spacings into the grid: 5.6 float32 eps for
    every value field, MADD-ANO-261), between four and six.

    **Its flags.**  The markers' positions are recomputed in the pass,
    so this group solves positions, and in 0.4.0 such a group has no
    usable flag on any step, with that rule's reason (MADD-ANO-252).
    (Before that rule the two reports set their flags, which is how this
    corner was first pinned.)  The edge back, ``q.x -> p.u``, is a dense
    matrix 30 entries wide, which the float-floor guard on a mapped row
    counts at its width (MADD-ANO-257).  It finds no flag to withdraw
    here and names its caveat all the same, after the group's own reason
    and with its code: the float floor this entry reports does not count
    the row.  With its limit out of reach the report is the same but for
    that sentence and that code.  Every number is the one it was."""
    with x64(True):
        fine = _stalled("float32", "float64")
        coarse = _stalled("float32", "float32")
        monkeypatch.setattr(_group_layout, "MAPPED_ROW_FLOOR_LIMIT", 10 ** 9)
        unguarded = [dict(report["graph"].coupling_diagnostics()[KEY]) for report in (fine, coarse)]
        monkeypatch.undo()
    for report, bare in zip((fine, coarse), unguarded):
        assert report["precision_limited"], report
        assert not report["spectral_usable"] and not report["gradient_bound_usable"], report
        assert "the group solves position(s)" in report["not_usable_reason"], report
        own, also, rows = report["not_usable_reason"].partition(" Also: ")
        assert own == bare["not_usable_reason"] and "MADD-ANO-257" not in own, (report, bare)
        assert also and "MADD-ANO-257" in rows, report
        solved = [reason_codes.GEOMETRY_POSITIONS_SOLVED]
        assert bare["reason_codes"]["spectral_usable"] == solved, bare["reason_codes"]
        assert report["reason_codes"]["spectral_usable"] == [
            *solved, reason_codes.LONG_MAPPED_ROW], report["reason_codes"]
        for name in set(bare) - {"not_usable_reason", "reason_codes"}:
            assert np.asarray(report[name]).tobytes() == np.asarray(bare[name]).tobytes(), name
        assert report["spectral_error_bound"] == bare["spectral_error_bound"], (report, bare)
        assert report["spectral_error_bound"] >= 10.0 * report["distance"] > 0.0, report
    assert 4.0 * fine["spectral_error_bound"] < coarse["spectral_error_bound"], (fine, coarse)
    assert 6.0 * fine["spectral_error_bound"] > coarse["spectral_error_bound"], (fine, coarse)
