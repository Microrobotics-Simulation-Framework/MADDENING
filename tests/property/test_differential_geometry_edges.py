"""An edge with a geometry-dependent mapping steps as the same graph with the mapping inside the target node.

``add_edge(S, T, sf, tf, mapping=m, geometry=(anchor, g))`` hands ``m`` a
moving geometry, read from the state of the edge's source or target at a
time level that depends on where the edge is resolved: a plain step's
forward or back edge, a coupling pass under Gauss-Seidel or Jacobi, a
sub-step of a sub-cycled member, the flux hook of the target.  What each
of those *should* read is defined by reduction: move the mapping inside
the target node, feed it the value (and, for a source anchor, the
geometry) through ordinary edges, and let the node read a target-anchored
geometry from the ``state`` its hook is handed
(:func:`tests.property.geometry_graphs.inline_geometry`).  The twin uses
nothing new, so its time levels are the ones ordinary edges and node state
already have, and the two graphs must agree:

* **states** after every step: within 8 ulps of the field's scale where
  no solve's tolerance is in play (no group, or a group run for a fixed
  few passes), and within twice what the group's tolerance allows where
  one converges.  Never bit for bit: the two are different programs;
* **reports** (``coupling_diagnostics()``): ``converged`` equal,
  ``iterations`` equal within one, ``residual`` on the same side of the
  threshold, equal to rounding when the iteration counts are equal and
  within a factor of two otherwise;
* **gradients** of one loss with respect to the initial geometry, to the
  parameter that moves it and to a value-side parameter: relative
  ``1e-4`` in float32, ``1e-8`` in float64.

Both ends of every geometry edge hold a geometry field of the same name
and shape with different values, so reading the wrong end is visible.

The cases cover a forward and a back edge, either end fast on a
multi-rate graph, groups under both schedules with every acceleration and
all three norms, sub-cycled members of either end under linear and
constant interpolation, both anchors, ``test_geom_matrix`` and both modes
of the library's ``multilinear_grid`` kind, an additive port with a
transform, and float64 and mixed dtypes.  Four of them are the only cases
that see one particular fault each, and so run per push:

* ``group, the geometry follows the iterate``: a group member's geometry
  that depends on the iterate (a geometry read once outside the pass, or
  from the pre-step state where the iterate is meant);
* ``sub-cycled target, one pass``: at a converged fixed point the two end
  points of a sub-step interpolation coincide, so "the geometry is not
  interpolated" is invisible everywhere but in a single pass;
* ``flux reads the mapped input``: the target's flux hook receives its
  post-update state, so a target-anchored geometry must be read again for
  it;
* ``interface norm, three passes`` and ``group IQN-ILS, three passes``:
  the interface norm and the quasi-Newton interface set must read a
  source-anchored geometry as they read the value, and three passes from
  convergence the residual and the iterate say so.

The second half of the module is the frozen case: a geometry that never
moves gives the static mapping built from the same points.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.property import geometry_graphs as gg
from tests.property.geometry_graphs import DT, case

#: Ulps of a field's scale two different programs of the same arithmetic
#: may differ by.
ROUNDING_ULPS = 8
#: The mixed and interface criteria are a root mean square over the entries
#: they read, so one entry may hold this many times its share of the
#: tolerance (the square root of the number of entries, generously).
RMS_ENTRY_SLACK = 8

_TIGHT64 = dict(max_iterations=400, tolerance=1e-13)
_TIGHT32 = dict(max_iterations=200, tolerance=1e-5)
_SUB = dict(subcycling=True)

#: The anchors of (``F.x -> P.u``, ``P.x -> F.u``): which body holds each
#: edge's geometry.
P_HOLDS = dict(down="target", up="source")
F_HOLDS = dict(down="source", up="target")
TARGETS = dict(down="target", up="target")
SOURCES = dict(down="source", up="source")

# ---------------------------------------------------------------------------
# The cases
# ---------------------------------------------------------------------------

#: Per push.  The four the module docstring names, and one of every other
#: call site: the first two put a source-anchored and a target-anchored
#: geometry on a forward and on a back edge each.
PER_PUSH = [
    case("forward and back edge, source anchors", adv=0.3, **SOURCES),
    case("back and forward edge, target anchors", order=("P", "F"), adv=0.3, **TARGETS),
    case("flux reads the mapped input", flux="reader", adv=0.3, **TARGETS),
    case("group, the geometry follows the iterate", group=_TIGHT32, adv=0.3, **P_HOLDS),
    case("group Jacobi, two passes", adv=0.3, **TARGETS,
         group=dict(max_iterations=2, iteration_mode="jacobi")),
    case("sub-cycled target, one pass", dt_p=DT / 4, adv=0.3, **SOURCES,
         group=dict(_SUB, max_iterations=1, boundary_interpolation="linear")),
    case("sub-cycled target, one pass, constant, target anchors", dt_p=DT / 4, adv=0.3,
         **TARGETS, group=dict(_SUB, max_iterations=1, boundary_interpolation="constant")),
    case("interface norm, three passes", adv=0.6, dtype="float64", **P_HOLDS,
         group=dict(max_iterations=3, convergence_norm="interface", rtol=1e-6)),
    case("group IQN-ILS, three passes", adv=0.6, dtype="float64", **P_HOLDS,
         group=dict(max_iterations=3, tolerance=1e-13, acceleration="iqn-ils")),
    case("group, a member's flux read in the same pass, two passes", flux="internal",
         order=("P", "F"), adv=0.3, **TARGETS, group=dict(max_iterations=2)),
    case("group, a member's flux seeded from the iterate, two passes", flux="internal",
         adv=0.3, **TARGETS, group=dict(max_iterations=2)),
    case("multilinear, forward and back edge", kind="multilinear", adv=0.3, **P_HOLDS),
]

#: The slow lane: the rest of the product.
SLOW = [
    # plain steps
    case("forward and back edge, P holds", adv=0.3, **P_HOLDS),
    case("back and forward edge, F holds", order=("P", "F"), adv=0.3, **F_HOLDS),
    case("forward edge only, target anchor", up=None, down="target", adv=0.3),
    case("forward edge only, source anchor", up=None, down="source"),
    case("the reverse edge only", down=None, up="source", order=("F", "P"), adv=0.3),
    case("additive port with a transform", extra=True, adv=0.3, **TARGETS),
    case("flux reads the mapped input, F holds", flux="reader", adv=0.3, **F_HOLDS),
    case("float64, forward and back edge", adv=0.3, dtype="float64", **SOURCES),
    case("multilinear, F holds", kind="multilinear", order=("P", "F"), adv=0.3, **F_HOLDS),
    case("multilinear 2-D", kind="multilinear", d=2, adv=0.3, **P_HOLDS),
    case("multilinear, float32 field and float64 geometry", kind="multilinear", adv=0.3,
         geom_dtype="float64", **TARGETS),
    case("multilinear, flux reads the mapped input", kind="multilinear", flux="reader",
         adv=0.3, **P_HOLDS),
    # multi-rate, either end fast
    case("multi-rate, P fast", dt_p=DT / 2, adv=0.3, steps=4, **P_HOLDS),
    case("multi-rate, F fast", dt_f=DT / 2, adv=0.3, steps=4, **F_HOLDS),
    case("multi-rate, P fast, targets", dt_p=DT / 2, adv=0.3, steps=4, order=("P", "F"),
         **TARGETS),
    # groups: both schedules, every acceleration, all three norms, both solvers
    case("group Jacobi", group=dict(_TIGHT32, iteration_mode="jacobi"), adv=0.3, **F_HOLDS),
    case("group float64", group=_TIGHT64, adv=0.3, dtype="float64", **TARGETS),
    case("group Aitken, mixed norm", adv=0.3, dtype="float64", **SOURCES,
         group=dict(max_iterations=400, acceleration="aitken", convergence_norm="mixed",
                    rtol=1e-11)),
    case("group fixed relaxation, Jacobi", adv=0.3, dtype="float64", **P_HOLDS,
         group=dict(_TIGHT64, acceleration="fixed", relaxation=0.7, iteration_mode="jacobi")),
    case("group IQN-ILS, Jacobi", adv=0.3, dtype="float64", **F_HOLDS,
         group=dict(_TIGHT64, acceleration="iqn-ils", iteration_mode="jacobi")),
    case("group IQN-IMVJ, interface norm", adv=0.3, dtype="float64", **SOURCES,
         group=dict(max_iterations=400, acceleration="iqn-imvj", jacobian_reuse=2,
                    convergence_norm="interface", rtol=1e-11)),
    case("group interface norm, Jacobi, three passes", adv=0.6, dtype="float64", **SOURCES,
         group=dict(max_iterations=3, convergence_norm="interface", rtol=1e-6,
                    iteration_mode="jacobi")),
    case("group, fori solver", adv=0.3, dtype="float64", **P_HOLDS,
         group=dict(max_iterations=120, tolerance=1e-13, solver="fori")),
    case("group, two passes", group=dict(max_iterations=2), adv=0.3, **TARGETS),
    case("group, quadratic predictor", group=dict(_TIGHT32, predictor="quadratic"), adv=0.3,
         steps=5, **P_HOLDS),
    case("group IQN-IMVJ, linear predictor, float64", adv=0.3, dtype="float64", steps=5,
         **F_HOLDS, group=dict(_TIGHT64, acceleration="iqn-imvj", jacobian_reuse=2,
                               predictor="linear")),
    case("sub-cycled target, interface norm, three passes", dt_p=DT / 2, adv=0.6,
         dtype="float64", **SOURCES,
         group=dict(_SUB, max_iterations=3, convergence_norm="interface", rtol=1e-6,
                    boundary_interpolation="linear")),
    case("group, a member's flux scattered inside it", group=_TIGHT32, flux="internal",
         adv=0.3, **P_HOLDS),
    case("group, a member's flux read in the same pass", group=_TIGHT32, flux="internal",
         order=("P", "F"), adv=0.3, **TARGETS),
    case("group Jacobi, a member's flux scattered inside it, float64", flux="internal",
         adv=0.3, dtype="float64", **TARGETS, group=dict(_TIGHT64, iteration_mode="jacobi")),
    case("group with an additive port and a transform", group=_TIGHT32, extra=True, adv=0.3,
         **F_HOLDS),
    case("multilinear group", kind="multilinear", group=_TIGHT32, adv=0.3, **P_HOLDS),
    case("multilinear group, F holds, float64", kind="multilinear", group=_TIGHT64, adv=0.3,
         dtype="float64", **F_HOLDS),
    case("multilinear 2-D group, Jacobi", kind="multilinear", d=2, adv=0.3, **TARGETS,
         group=dict(_TIGHT32, iteration_mode="jacobi")),
    # sub-cycled members of either end, linear and constant, converged and one pass
    case("sub-cycled P, linear", dt_p=DT / 2, adv=0.3, **P_HOLDS,
         group=dict(_TIGHT32, **_SUB, boundary_interpolation="linear")),
    case("sub-cycled P, constant, Jacobi", dt_p=DT / 2, adv=0.3, **F_HOLDS,
         group=dict(_TIGHT32, **_SUB, boundary_interpolation="constant",
                    iteration_mode="jacobi")),
    case("sub-cycled F, linear, float64", dt_f=DT / 2, adv=0.3, dtype="float64", **SOURCES,
         group=dict(_TIGHT64, **_SUB, boundary_interpolation="linear")),
    case("sub-cycled F, constant", dt_f=DT / 2, adv=0.3, **TARGETS,
         group=dict(_TIGHT32, **_SUB, boundary_interpolation="constant")),
    case("sub-cycled source, one pass", dt_f=DT / 4, adv=0.3, order=("P", "F"), **SOURCES,
         group=dict(_SUB, max_iterations=1, boundary_interpolation="linear")),
    case("sub-cycled target, one pass, target anchors", dt_p=DT / 4, adv=0.3, **TARGETS,
         group=dict(_SUB, max_iterations=1, boundary_interpolation="linear")),
    case("sub-cycled target, one pass, constant", dt_p=DT / 4, adv=0.3, **SOURCES,
         group=dict(_SUB, max_iterations=1, boundary_interpolation="constant")),
    case("sub-cycled target, one pass, Jacobi", dt_p=DT / 4, adv=0.3, **P_HOLDS,
         group=dict(_SUB, max_iterations=1, boundary_interpolation="linear",
                    iteration_mode="jacobi")),
    case("multilinear, sub-cycled target, one pass", kind="multilinear", dt_p=DT / 4,
         adv=0.3, **F_HOLDS,
         group=dict(_SUB, max_iterations=1, boundary_interpolation="linear")),
]

#: The cases whose gradients are compared per push, and in the slow lane.
GRADIENTS_PER_PUSH = [c for c in PER_PUSH if c.label in (
    "forward and back edge, source anchors", "flux reads the mapped input")]
GRADIENTS_SLOW = [
    case("float64, forward and back edge", adv=0.3, dtype="float64", **SOURCES),
    case("float64, F holds", order=("P", "F"), adv=0.3, dtype="float64", **F_HOLDS),
    case("float64 group", group=_TIGHT64, adv=0.3, dtype="float64", **P_HOLDS),
    case("float64 group, Jacobi, IQN-ILS", adv=0.3, dtype="float64", **F_HOLDS,
         group=dict(_TIGHT64, acceleration="iqn-ils", iteration_mode="jacobi")),
    case("float64 group, one pass", group=dict(max_iterations=1), adv=0.3, dtype="float64",
         **TARGETS),
    case("float64, sub-cycled target", dt_p=DT / 2, adv=0.3, dtype="float64", **SOURCES,
         group=dict(_TIGHT64, **_SUB, boundary_interpolation="linear")),
    case("float64, sub-cycled source, constant", dt_f=DT / 2, adv=0.3, dtype="float64",
         **P_HOLDS, group=dict(_TIGHT64, **_SUB, boundary_interpolation="constant")),
    case("float32 group", group=_TIGHT32, adv=0.3, **P_HOLDS),
    case("multilinear, float64", kind="multilinear", adv=0.3, dtype="float64", **P_HOLDS),
    case("multilinear group, float64", kind="multilinear", group=_TIGHT64, adv=0.3,
         dtype="float64", **F_HOLDS),
]


# ---------------------------------------------------------------------------
# PHASE 1 (see the block of that name in ``geometry_graphs``)
# ---------------------------------------------------------------------------


def GEOMETRY_EDGE_KEYS(c: gg.Case) -> list:
    """The keys of *c*'s geometry edges (both end in the group when it has one)."""
    with gg.x64(c.needs_x64):
        gm = gg.build(gg.two_body(c), compile=False)
    return [e.key for e in gm.edges if e.geometry is not None]


def _runnable(cases) -> list:
    """*cases* without those ``compile()`` refuses in phase 1."""
    def bare(c):        # a case, or a ``pytest.param`` holding one
        return c.values[0] if hasattr(c, "values") else c
    return [c for c in cases if not gg.refused(bare(c))]


#: The cases ``compile()`` refuses in phase 1: the interface norm over a
#: geometry edge.  Each is asserted refused, per push.
REFUSED_IN_PHASE_1 = [c for c in PER_PUSH + SLOW if gg.refused(c)]


@pytest.mark.parametrize("c", REFUSED_IN_PHASE_1, ids=repr)
def test_the_interface_norm_over_a_geometry_edge_is_refused_at_compile(c):
    """The norm reads the geometry of the ``multilinear_grid`` kind only: a
    position is measured in units of the kind's own length scale, which
    the ``test_geom_matrix`` kind of these cases does not declare.  The
    group is refused and the message names the kind and the norms that
    measure the state.  The node-inlined twin, which has plain edges only,
    compiles."""
    with gg.x64(c.needs_x64):
        gg.assert_interface_norm_refused(lambda: gg.build(gg.two_body(c)),
                                         GEOMETRY_EDGE_KEYS(c), gg.refused(c))
        gg.build(gg.inline_geometry(gg.two_body(c)))


def test_the_phase_1_refusals_are_the_interface_norm_cases_and_only_those():
    assert len(REFUSED_IN_PHASE_1) == 4
    assert all(c.group is not None and c.down and c.up for c in REFUSED_IN_PHASE_1)
    assert all(c.kind == "geom_matrix" and gg.refused(c) == "kind" for c in REFUSED_IN_PHASE_1)
    assert any(c in PER_PUSH for c in REFUSED_IN_PHASE_1)


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def _solve_tolerance(c: gg.Case, scale: float) -> float:
    """What a converging group's criterion allows a field of size *scale* to be off by."""
    knobs = c.knobs
    if not knobs or knobs.get("max_iterations", 10) <= 4:
        return 0.0          # a fixed few passes: the same arithmetic in both graphs
    if knobs.get("convergence_norm", "l2") == "l2":
        return float(knobs.get("tolerance", 1e-6))
    return float(knobs.get("rtol", 1e-6)) * scale * RMS_ENTRY_SLACK


def assert_same_states(c: gg.Case, a: dict, b: dict, *, step: int) -> float:
    """Every field of every node, within rounding plus what the solve allows; the worst gap."""
    worst = 0.0
    for name in a:
        for field in a[name]:
            x, y = np.asarray(a[name][field]), np.asarray(b[name][field])
            assert x.dtype == y.dtype and x.shape == y.shape, (name, field, x.dtype, y.dtype)
            scale = float(max(np.max(np.abs(x)), np.max(np.abs(y))))
            # Each of the two solves is within its tolerance of the fixed
            # point: twice the tolerance apart, per step taken.
            allowed = (ROUNDING_ULPS * float(np.finfo(x.dtype).eps) * scale
                       + 2 * step * _solve_tolerance(c, scale))
            gap = float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))))
            assert gap <= allowed, (
                f"{c.label} step {step}: {name}.{field} of the edge-mapped graph is "
                f"{gap:.3e} from the node-inlined graph's ({allowed:.3e} allowed)")
            worst = max(worst, gap)
    return worst


def _residual_floor(c: gg.Case, state: dict) -> float:
    """A residual's own float rounding, in the units it is reported in."""
    knobs = c.knobs
    eps = float(np.finfo(np.dtype(c.dtype)).eps)
    if knobs.get("convergence_norm", "l2") == "l2":
        n = sum(v.size for name in ("F", "P") for v in state[name].values())
        scale = max(float(np.max(np.abs(v))) for name in ("F", "P")
                    for v in state[name].values())
        return eps * scale * float(np.sqrt(n))
    return eps / float(knobs.get("rtol", 1e-6))


def assert_same_reports(c: gg.Case, a: dict, b: dict, state: dict, *, step: int) -> None:
    assert sorted(a) == sorted(b), (c.label, sorted(a), sorted(b))
    knobs = c.knobs
    threshold = (float(knobs.get("tolerance", 1e-6))
                 if knobs.get("convergence_norm", "l2") == "l2" else 1.0)
    for key in a:
        ra, rb = a[key], b[key]
        where = f"{c.label} step {step} group {key}"
        assert bool(ra["converged"]) == bool(rb["converged"]), (where, ra, rb)
        ia, ib = int(ra["iterations"]), int(rb["iterations"])
        assert abs(ia - ib) <= 1, (where, ia, ib)
        res_a, res_b = float(ra["residual"]), float(rb["residual"])
        assert np.isfinite(res_a) and np.isfinite(res_b), (where, res_a, res_b)
        assert (res_a <= threshold) == (res_b <= threshold), (where, res_a, res_b, threshold)
        if ia == ib:
            allowed = 64 * _residual_floor(c, state) + 1e-6 * max(res_a, res_b)
            assert abs(res_a - res_b) <= allowed, (
                f"{where}: after the same {ia} passes the edge-mapped graph reports "
                f"residual {res_a:.6e} and the node-inlined graph {res_b:.6e}")
        else:
            lo, hi = sorted((res_a, res_b))
            assert hi <= 2 * lo + 64 * _residual_floor(c, state), (where, res_a, res_b)
        if gg.withheld(c) is not None:
            # PHASE 1 (see ``geometry_graphs``): beyond the solve's outcome,
            # compared above, a group of another mapping kind and a
            # sub-cycled one report no bound and say which they are.
            gg.assert_not_diagnosed(ra, GEOMETRY_EDGE_KEYS(c), gg.withheld(c))
            continue
        # Everything else a report says: the same keys, the same verdicts,
        # and after the same number of passes the same numbers to within a
        # factor of two (they are estimates; the residual above is not).
        assert sorted(ra) == sorted(rb), (where, sorted(ra), sorted(rb))
        if ia != ib:
            continue
        for name in ra:
            va, vb = ra[name], rb[name]
            if isinstance(va, (bool, np.bool_)) or isinstance(vb, (bool, np.bool_)):
                assert bool(va) == bool(vb), (where, name, va, vb)
                continue
            va, vb = float(va), float(vb)
            assert np.isfinite(va) == np.isfinite(vb), (where, name, va, vb)
            if np.isfinite(va) and name not in ("iterations", "total_iterations", "residual"):
                lo, hi = sorted((abs(va), abs(vb)))
                assert hi <= 2 * lo + 64 * _residual_floor(c, state), (where, name, va, vb)


@functools.lru_cache(maxsize=4)
def _pair(c: gg.Case):
    with gg.x64(c.needs_x64):
        return gg.graphs(c)


def assert_steps_as_its_inlined_twin(c: gg.Case) -> float:
    """States and reports of *c*'s two graphs agree after every step; the worst state gap."""
    edge, inline = _pair(c)
    worst = 0.0
    with gg.x64(c.needs_x64):
        edge.reset_state()
        inline.reset_state()
        before = gg.snapshot(edge)
        for step in range(1, c.steps + 1):
            edge.step()
            inline.step()
            a, b = gg.snapshot(edge), gg.snapshot(inline)
            worst = max(worst, assert_same_states(c, a, b, step=step))
            if c.group is not None:
                assert_same_reports(c, edge.coupling_diagnostics(),
                                    inline.coupling_diagnostics(), b, step=step)
        after = gg.snapshot(edge)
    # The premise: the geometry both edges read has moved, and differs
    # between the two bodies, so a wrong time level or a wrong end is
    # another number.
    for which in ("down", "up"):
        if (c.down if which == "down" else c.up) is None:
            continue
        field = "pos" if c.kind == "multilinear" else ("A" if which == "down" else "B")
        held = gg.holder(c, which)
        assert np.any(after[held][field] != before[held][field]), (c.label, which, "frozen")
        assert np.any(after["F"][field] != after["P"][field]), (c.label, which, "same at both")
    return worst


@pytest.mark.parametrize("c", _runnable(PER_PUSH), ids=repr)
def test_an_edge_mapped_graph_steps_as_its_node_inlined_twin(c):
    """Per push; slow sibling :func:`test_every_drawn_edge_mapped_graph_steps_as_its_node_inlined_twin`."""
    assert_steps_as_its_inlined_twin(c)


# Slow: two graphs compiled per case, forty cases.
# Per push: tests/property/test_differential_geometry_edges.py::test_an_edge_mapped_graph_steps_as_its_node_inlined_twin
@pytest.mark.slow
@pytest.mark.parametrize("c", _runnable(SLOW), ids=repr)
def test_every_drawn_edge_mapped_graph_steps_as_its_node_inlined_twin(c):
    assert_steps_as_its_inlined_twin(c)


def test_the_cases_cover_what_the_module_claims():
    """The product the docstring lists is in the lists above (a case deleted
    to make a run green would otherwise go unnoticed)."""
    cases = PER_PUSH + SLOW
    groups = [c.knobs for c in cases if c.group is not None]
    assert {g.get("acceleration", "none") for g in groups} >= {
        "none", "aitken", "fixed", "iqn-ils", "iqn-imvj"}
    assert {g.get("convergence_norm", "l2") for g in groups} == {"l2", "mixed", "interface"}
    assert {g.get("iteration_mode", "gauss-seidel") for g in groups} == {"gauss-seidel",
                                                                          "jacobi"}
    assert {g.get("solver", "ift") for g in groups} == {"ift", "fori"}
    assert {g.get("boundary_interpolation") for g in groups if g.get("subcycling")} == {
        "linear", "constant"}
    assert {(c.down, c.up) for c in cases} >= {
        ("target", "source"), ("source", "target"), ("target", "target"), ("source", "source")}
    assert {c.kind for c in cases} == {"geom_matrix", "multilinear"}
    assert {c.flux for c in cases} == {None, "reader", "internal"}
    assert any(c.group is None and c.dt_p < c.dt_f for c in cases)
    assert any(c.group is None and c.dt_f < c.dt_p for c in cases)
    one_pass = [c for c in cases if c.group is not None and c.knobs.get("subcycling")
                and c.knobs.get("max_iterations") == 1]
    assert any(c.dt_p < c.dt_f for c in one_pass) and any(c.dt_f < c.dt_p for c in one_pass)
    assert any(c.knobs.get("convergence_norm") == "interface" and c.adv
               for c in PER_PUSH if c.group is not None)
    assert any(c.flux == "reader" and c.group is None and c.down == "target" for c in PER_PUSH)
    assert any(c.knobs.get("acceleration") == "iqn-ils" and c.adv
               for c in PER_PUSH if c.group is not None)
    # A member's flux that reads its mapped input, read by a member swept
    # before the producer (seeded from the iterate) and by one swept after
    # it (computed in the pass).
    internal = {c.order for c in cases if c.flux == "internal" and c.group is not None}
    assert internal == {("F", "P"), ("P", "F")}
    assert {g.get("predictor", "none") for g in groups} >= {"none", "linear", "quadratic"}
    # A forward and a back edge, each with a source-anchored and a
    # target-anchored geometry, per push and outside any group.
    plain = {(c.order, c.down, c.up) for c in PER_PUSH if c.group is None and not c.flux
             and c.kind == "geom_matrix"}
    assert plain >= {(("F", "P"), "source", "source"), (("P", "F"), "target", "target")}


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------


def assert_same_gradients(c: gg.Case) -> None:
    """``jax.grad`` of the output loss with respect to the initial geometry,
    the rate that moves it and a value-side parameter, edge-mapped against
    node-inlined."""
    rel = 1e-8 if c.dtype == "float64" else 1e-4
    with gg.x64(c.needs_x64):
        edge, inline = gg.graphs(c)
        loss_e, theta = gg.loss_of(edge, c)
        loss_i, theta_i = gg.loss_of(inline, c)
        assert sorted(theta) == sorted(theta_i)
        ge = jax.jit(jax.grad(loss_e))(theta)
        gi = jax.jit(jax.grad(loss_i))(theta)
        for key in sorted(theta):
            a, b = np.asarray(ge[key], np.float64), np.asarray(gi[key], np.float64)
            scale = float(np.max(np.abs(b)))
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), (c.label, key)
            if key[0] == "state" or key[2] == "rate":
                # The premise: the loss does depend on the geometry, so a
                # dropped gradient is not a correct zero.
                assert scale > 0, f"{c.label}: the loss does not depend on {key}"
            assert float(np.max(np.abs(a - b))) <= rel * scale, (
                f"{c.label}: d loss / d {key} is {float(np.max(np.abs(a - b))) / scale:.3e} "
                f"(relative) from the node-inlined graph's")


@pytest.mark.parametrize("c", _runnable(GRADIENTS_PER_PUSH), ids=repr)
def test_the_gradients_of_an_edge_mapped_graph_are_its_node_inlined_twin_s(c):
    """Per push; slow sibling :func:`test_the_gradients_of_every_drawn_edge_mapped_graph_are_its_twin_s`."""
    assert_same_gradients(c)


# Slow: a gradient through three steps of two graphs compiled per case.
# Per push: tests/property/test_differential_geometry_edges.py::test_the_gradients_of_an_edge_mapped_graph_are_its_node_inlined_twin_s
@pytest.mark.slow
@pytest.mark.parametrize("c", _runnable(GRADIENTS_SLOW), ids=repr)
def test_the_gradients_of_every_drawn_edge_mapped_graph_are_its_twin_s(c):
    assert_same_gradients(c)


# Slow: two graphs with diagnostics compiled.
# Per push: tests/property/test_differential_geometry_edges.py::test_an_edge_mapped_graph_steps_as_its_node_inlined_twin
@pytest.mark.slow
def test_a_report_s_float_floor_counts_a_source_anchored_geometry():
    """The float floor a report is judged against reads what the group's
    norm reads, a source-anchored geometry included.  It shows where the
    geometry's dtype is not the value's: under the interface norm the floor
    is a root mean square of ``eps / rtol`` over the entries read, so a
    float64 geometry among float32 values lowers it.  A group run to its
    floor then reports the same ``spectral_error_bound`` from both graphs
    (leaving the geometry out of the floor's inputs moves it by 30%)."""
    c = case("floor", kind="multilinear", geom_dtype="float64", adv=0.3, **SOURCES,
             group=dict(max_iterations=80, convergence_norm="interface", rtol=1e-6,
                        diagnostics=True))
    if not gg.INTERFACE_BOUNDS_READ_GEOMETRY:
        # The criterion reads the geometry of this group and its report
        # withholds every bound, the floor's among them (see
        # ``geometry_graphs``); the comparison below is the bounds stage's,
        # and its twin is the wrong one for the rule (the node-inlined twin
        # reads a gather at its raw source).
        with gg.x64(True):
            edge = gg.build(gg.two_body(c))
            edge.step()
            gg.assert_not_diagnosed(edge.coupling_diagnostics()["F+P"],
                                    GEOMETRY_EDGE_KEYS(c), "norm")
        return
    with gg.x64(True):
        edge, inline = gg.graphs(c)
        for step in range(1, 4):
            edge.step()
            inline.step()
            re, ri = edge.coupling_diagnostics()["F+P"], inline.coupling_diagnostics()["F+P"]
            assert int(re["iterations"]) == int(ri["iterations"]), (step, re, ri)
            assert bool(re["precision_limited"]) and bool(ri["precision_limited"]), (
                "premise: a residual at its float floor", step, re, ri)
            a, b = float(re["spectral_error_bound"]), float(ri["spectral_error_bound"])
            assert np.isfinite(a) and np.isfinite(b) and b > 0, (step, a, b)
            assert abs(a - b) <= 0.05 * b, (
                f"step {step}: the edge-mapped graph reports spectral_error_bound {a:.6g} and "
                f"the node-inlined graph {b:.6g}")


# ---------------------------------------------------------------------------
# Batches and restarts
# ---------------------------------------------------------------------------

BATCHED_PER_PUSH = [c for c in PER_PUSH if c.label in (
    "forward and back edge, source anchors", "multilinear, forward and back edge")]
BATCHED_SLOW = [
    case("batched group", group=_TIGHT32, adv=0.3, **P_HOLDS),
    case("batched group, float64, Jacobi", adv=0.3, dtype="float64", **F_HOLDS,
         group=dict(_TIGHT64, iteration_mode="jacobi")),
    case("batched multilinear group", kind="multilinear", group=_TIGHT32, adv=0.3, **TARGETS),
    case("batched sub-cycled group", dt_p=DT / 2, adv=0.3, **SOURCES,
         group=dict(_TIGHT32, **_SUB, boundary_interpolation="linear")),
]


def _batch_of(state0: dict, members: int = 3) -> dict:
    """*members* initial states: every floating field of every node moved a
    little further per member (the geometry with them); ``_meta`` repeated."""
    batch = {}
    for name, fields in state0.items():
        batch[name] = {}
        for field, v in fields.items():
            moved = [v if (name.startswith("_") or not jnp.issubdtype(v.dtype, jnp.floating))
                     else v + jnp.asarray(0.004 * k, v.dtype) for k in range(members)]
            batch[name][field] = jnp.stack(moved)
    return batch


def assert_batched_run_matches(c: gg.Case) -> None:
    """``jax.vmap`` of *c.steps* steps over a batch of initial states (the
    geometry is state, so it is batched with them): the edge-mapped graph
    against its node-inlined twin member by member, and every member
    against its own unbatched run."""
    with gg.x64(c.needs_x64):
        edge, inline = gg.graphs(c)
        outs = []
        for gm in (edge, inline):
            run, state0, params0 = gg.rollout(gm, c.steps)
            batch = _batch_of(state0)
            batched = jax.jit(jax.vmap(run, in_axes=(0, None)))(batch, params0)
            outs.append((run, batch, params0, batched))
        nodes = [n for n in outs[0][3] if not n.startswith("_")]

        def member(tree, k):
            return {n: {f: np.asarray(v[k]) for f, v in tree[n].items()} for n in nodes}

        run, batch, params0, batched = outs[0]
        single = jax.jit(run)
        for k in range(3):
            assert_same_states(c, member(batched, k), member(outs[1][3], k), step=c.steps)
            alone = single(jax.tree.map(lambda v, k=k: v[k], batch), params0)
            assert_same_states(
                c, member(batched, k),
                {n: {f: np.asarray(v) for f, v in alone[n].items()} for n in nodes},
                step=c.steps)
        first, last = member(batched, 0), member(batched, 2)
        assert any(np.any(first[n]["x"] != last[n]["x"]) for n in nodes), (
            "premise: the members of the batch differ")


@pytest.mark.parametrize("c", _runnable(BATCHED_PER_PUSH), ids=repr)
def test_a_batched_run_of_an_edge_mapped_graph_matches_its_twin_and_its_single_runs(c):
    """Per push; slow sibling :func:`test_every_batched_run_matches_its_twin_and_its_single_runs`."""
    assert_batched_run_matches(c)


# Slow: a vmapped scan of two graphs and a single one compiled per case.
# Per push: tests/property/test_differential_geometry_edges.py::test_a_batched_run_of_an_edge_mapped_graph_matches_its_twin_and_its_single_runs
@pytest.mark.slow
@pytest.mark.parametrize("c", _runnable(BATCHED_SLOW), ids=repr)
def test_every_batched_run_matches_its_twin_and_its_single_runs(c):
    assert_batched_run_matches(c)


_RESTART_CASES = [
    PER_PUSH[0],
    pytest.param(case("restarted group", group=_TIGHT32, adv=0.3, **P_HOLDS),
                 marks=pytest.mark.slow),
    pytest.param(case("restarted group, quadratic predictor, multilinear", kind="multilinear",
                      group=dict(_TIGHT32, predictor="quadratic"), adv=0.3, **F_HOLDS),
                 marks=pytest.mark.slow),
]


# Per push: tests/property/test_differential_geometry_edges.py::test_a_run_restarted_from_a_checkpoint_continues_as_the_uninterrupted_run
# (on the plain graph; the slow parameters are the same check on groups)
@pytest.mark.parametrize("c", _runnable(_RESTART_CASES), ids=repr)
def test_a_run_restarted_from_a_checkpoint_continues_as_the_uninterrupted_run(c, tmp_path):
    """A geometry is ordinary node state and a geometry-dependent mapping
    carries nothing between steps: saved after two steps and loaded into
    the reset graph, the run continues bit for bit."""
    with gg.x64(c.needs_x64):
        gm = gg.build(gg.two_body(c))
        straight = gg.run_steps(gm, 5)
        first = gg.run_steps(gm, 2)
        assert all(np.array_equal(first[-1][n][f], straight[1][n][f])
                   for n in first[-1] for f in first[-1][n])
        path = gm.save_state(tmp_path / "checkpoint.npz")
        gm.reset_state()
        gm.load_state(path)
        for k in range(2, 5):
            gm.step()
            got = gg.snapshot(gm)
            for n in got:
                for f in got[n]:
                    assert got[n][f].tobytes() == straight[k][n][f].tobytes(), (
                        f"{c.label}: {n}.{f} after the restart, step {k + 1}")
    assert np.any(straight[-1]["P"]["x"] != straight[1]["P"]["x"])


# ---------------------------------------------------------------------------
# The twin itself
# ---------------------------------------------------------------------------


def test_the_inlined_twin_keeps_names_orders_and_groups_and_uses_plain_edges_only():
    """The reduction changes what the definition says it changes and nothing else."""
    c = case("twin", extra=True, flux="reader", group=dict(max_iterations=5), **F_HOLDS)
    graph = gg.two_body(c)
    twin = gg.inline_geometry(graph)
    assert [nd.name for nd in twin.nodes] == [nd.name for nd in graph.nodes]
    assert twin.groups == graph.groups
    assert all(e.mapping is None and e.geometry is None for e in twin.edges)
    assert [(e.src, e.dst, e.sf, e.tf) for e in twin.edges] == [
        ("E", "P", "x", "u"),
        ("F", "P", "x", "u@value"), ("F", "P", "A", "u@geometry"),     # source anchor
        ("P", "F", "x", "u@value"),                                     # target anchor
        ("P", "R", "q", "q"),
    ]
    for inner, outer in zip(graph.nodes, twin.nodes):
        assert (outer.name, outer.delta_t) == (inner.name, inner.delta_t)
        assert sorted(outer.initial_state()) == sorted(inner.initial_state())
        assert sorted(outer.params_pytree()) == sorted(inner.params_pytree())
        assert outer.update_evaluations() == inner.update_evaluations()
    assert isinstance(twin.node("P"), gg.InlinedFluxMappingNode)
    assert type(twin.node("F")) is gg.InlinedMappingNode
    assert type(twin.node("R")) is gg.Reader and type(twin.node("E")) is gg.Body


def test_two_geometry_edges_into_one_port_are_inlined_on_a_port_each():
    """Two grid-side bodies deliver into ``P.u`` through a mapping each,
    the second additively: the twin gives each edge its own value and
    geometry port.  (They used to share ``"u@value"``, the second value
    replacing the first before either was mapped: the twin's states were
    0.11 to 0.17 from the edge-mapped graph's with nothing refused.  No
    case of this module has that shape.)"""
    rng = np.random.default_rng(5)
    mats = [rng.uniform(-0.6, 0.6, size=(gg.M_POINTS, gg.N_MATRIX)) for _ in range(3)]

    def graph():
        nodes = [gg.Body("F1", DT, n=gg.N_MATRIX, geoms={"A": mats[0]}, seed=21),
                 gg.Body("F2", DT, n=gg.N_MATRIX, geoms={"A": mats[1]}, seed=22),
                 gg.Body("P", DT, n=gg.M_POINTS, geoms={"A": mats[2]}, seed=23)]
        mapping = lambda: gg.geom_matrix_mapping(gg.M_POINTS, gg.N_MATRIX)   # noqa: E731
        return gg.GGraph(nodes, [
            gg.GEdge("F1", "P", "x", "u", mapping=mapping(), geometry=("source", "A")),
            gg.GEdge("F2", "P", "x", "u", mapping=mapping(), geometry=("target", "A"),
                     additive=True)])

    twin = gg.inline_geometry(graph())
    assert [(e.src, e.tf) for e in twin.edges] == [
        ("F1", "u@value"), ("F1", "u@geometry"), ("F2", "u@value@1")]
    edge, inline = gg.build(graph()), gg.build(twin)
    for a, b in zip(gg.run_steps(edge, 3), gg.run_steps(inline, 3)):
        for name in a:
            for field in a[name]:
                np.testing.assert_allclose(a[name][field], b[name][field], rtol=2e-6, atol=1e-7)
    # Both edges reach the port: with either removed the state is another.
    alone = gg.GGraph(graph().nodes, graph().edges[:1])
    last, only = gg.run_steps(edge, 3)[-1], gg.run_steps(gg.build(alone), 3)[-1]
    assert np.max(np.abs(last["P"]["x"] - only["P"]["x"])) > 1e-2


def test_a_geometry_edge_that_is_not_the_last_into_its_port_is_refused_by_the_twin():
    """The additive sum of a port follows the edge order, and the twin adds
    the mapped value last: a geometry edge ahead of a plain one would sum
    in another order in the two graphs, so the generator refuses to draw it."""
    graph = gg.two_body(case("order", extra=True, **TARGETS))
    plain, geometry = graph.edges[0], graph.edges[1]
    assert plain.mapping is None and geometry.geometry is not None
    graph.edges[0], graph.edges[1] = geometry, plain
    with pytest.raises(AssertionError, match="last edge into its port"):
        gg.inline_geometry(graph)


# ---------------------------------------------------------------------------
# A geometry that never moves gives the static mapping built from the same points
# ---------------------------------------------------------------------------
#
# A frozen multilinear map is a fixed linear map, so a graph whose mapped
# edges gather from or scatter onto a grid at points that no update moves
# is, exactly, the graph with the static ``matrix_mapping`` of that gather
# (or its transpose).  Three statements:
#
# * the two-body graph with its geometry frozen, against its static twin
#   (``H`` from the independent reference stencil, at arbitrary points):
#   states to rounding, in float32, float64 and with a float64 geometry
#   under a float32 field, in and out of a group, and the same gradient
#   with respect to a value-side parameter;
# * the named coupled topologies with every mapped edge built as a frozen
#   multilinear edge (``Geometry("frozen", ...)``: both anchors, both
#   modes) reproduce the monolithic reference with ``values["H"]`` the
#   reference stencil's matrix -- every oracle of the static case,
#   unchanged -- and their points are bit for bit where they started;
# * the same graphs in the topology harness's other numeric domains
#   (float64, bfloat16 and float16 fields under a float32 geometry,
#   multi-rate, sub-cycled, ``vmap``, predictors and warm starts,
#   checkpoint and restart), against the static graph of the same
#   structure and values.  The points are multiples of 1/8 of a cell, so
#   the weights are exact in every float dtype.

from tests.property import coupled_topologies as ct  # noqa: E402
from tests.property import test_differential_coupling_topologies as topo_tests  # noqa: E402

NAMED = ct.named_topologies()
_FROZEN = dict(rate=0.0, adv=0.0)
FROZEN_CASES = [
    case("frozen, multilinear", kind="multilinear", **_FROZEN, **P_HOLDS),
    case("frozen, matrix, F holds", order=("P", "F"), **_FROZEN, **F_HOLDS),
]
FROZEN_SLOW = [
    case("frozen group, multilinear, F holds", kind="multilinear", group=_TIGHT32, **_FROZEN,
         **F_HOLDS),
    case("frozen, multilinear 2-D, float64", kind="multilinear", d=2, dtype="float64",
         **_FROZEN, **TARGETS),
    case("frozen, multilinear 3-D", kind="multilinear", d=3, **_FROZEN, **SOURCES),
    case("frozen, multilinear, float64 geometry under a float32 field", kind="multilinear",
         geom_dtype="float64", **_FROZEN, **P_HOLDS),
    case("frozen group, matrix, float64", group=_TIGHT64, dtype="float64", **_FROZEN,
         **TARGETS),
    case("frozen group, multilinear, Jacobi, IQN-ILS", kind="multilinear", dtype="float64",
         **_FROZEN, **SOURCES,
         group=dict(_TIGHT64, acceleration="iqn-ils", iteration_mode="jacobi")),
    case("frozen, sub-cycled, multilinear", kind="multilinear", dt_p=DT / 2, **_FROZEN,
         **P_HOLDS, group=dict(_TIGHT32, **_SUB, boundary_interpolation="linear")),
    case("frozen, multi-rate, multilinear", kind="multilinear", dt_f=DT / 2, steps=4,
         **_FROZEN, **F_HOLDS),
]


def assert_frozen_equals_static(c: gg.Case) -> None:
    assert c.rate == 0.0 and c.adv == 0.0, "premise: nothing moves the geometry"
    with gg.x64(c.needs_x64):
        frozen, static = gg.build(gg.two_body(c)), gg.build(gg.static_twin(c))
        start = gg.snapshot(frozen)
        a, b = gg.run_steps(frozen, c.steps), gg.run_steps(static, c.steps)
        for step, (sa, sb) in enumerate(zip(a, b), start=1):
            assert_same_states(c, sa, sb, step=step)
            for body in ("F", "P"):
                for field in start[body]:
                    if field != "x":
                        assert sa[body][field].tobytes() == start[body][field].tobytes(), (
                            c.label, body, field, "a frozen geometry moved")
        assert np.any(a[-1]["F"]["x"] != start["F"]["x"]), "premise: the values do move"
        # The same derivative with respect to a value-side parameter.
        loss_f, theta = gg.loss_of(frozen, c)
        loss_s, _theta = gg.loss_of(static, c)
        key = ("params", "F", "c")
        gf = np.asarray(jax.grad(lambda v: loss_f({**theta, key: v}))(theta[key]), np.float64)
        gs = np.asarray(jax.grad(lambda v: loss_s({**theta, key: v}))(theta[key]), np.float64)
        rel = 1e-8 if c.dtype == "float64" else 1e-4
        assert np.max(np.abs(gf - gs)) <= rel * np.max(np.abs(gs)), (c.label, gf, gs)


@pytest.mark.parametrize("c", _runnable(FROZEN_CASES), ids=repr)
def test_a_frozen_geometry_steps_as_the_static_mapping_of_the_same_points(c):
    """Per push; slow sibling :func:`test_every_frozen_geometry_steps_as_its_static_mapping`."""
    assert_frozen_equals_static(c)


# Slow: two graphs and two gradients compiled per case.
# Per push: tests/property/test_differential_geometry_edges.py::test_a_frozen_geometry_steps_as_the_static_mapping_of_the_same_points
@pytest.mark.slow
@pytest.mark.parametrize("c", _runnable(FROZEN_SLOW), ids=repr)
def test_every_frozen_geometry_steps_as_its_static_mapping(c):
    assert_frozen_equals_static(c)


#: ``(anchor, mode)`` of a topology's mapped edges, in rotation.  Two places
#: on, both the anchor and the mode are the other one.
_COMBINATIONS = (("source", "consistent"), ("target", "consistent"),
                 ("target", "conservative"), ("source", "conservative"))


def frozen_geometry(topo, variant: int = 0):
    """Every mapped edge of *topo* as a frozen multilinear edge, anchors and
    modes rotating over the four combinations from *variant* on."""
    mapped = [i for i, e in enumerate(topo.edges) if e.mapped]
    picks = [_COMBINATIONS[(k + variant) % 4] for k in range(len(mapped))]
    return ct.Geometry("frozen", tuple((i, p[0]) for i, p in zip(mapped, picks)),
                       tuple((i, p[1]) for i, p in zip(mapped, picks)))


@functools.lru_cache(maxsize=24)
def _frozen_built(domain: str, name: str, variant: int):
    """``(topology, knobs, geometry, Built)`` of a named structure in *domain*,
    its mapped edges frozen multilinear ones."""
    topo = topo_tests._domain_topology(NAMED[name], domain)            # noqa: SLF001
    knobs = topo_tests._domain_knobs(ct.topology_knobs(topo, 0), domain)   # noqa: SLF001
    geometry = frozen_geometry(topo, variant)
    with gg.x64(domain == "f64"):
        built = ct.build(topo, knobs, dtype=topo_tests._domain_dtype(domain),  # noqa: SLF001
                         geometry=geometry)
    return topo, knobs, geometry, built


def _frozen_values(topo, knobs, geometry, domain) -> list:
    with gg.x64(domain == "f64"):
        return [ct.draw_values(topo, np.random.default_rng(seed), rho,
                               dtype=topo_tests._domain_dtype(domain),      # noqa: SLF001
                               group_cfgs=ct.group_cfgs_of(knobs), geometry=geometry)
                for seed, rho in topo_tests._DRAWS]                         # noqa: SLF001


def _assert_points_never_moved(values, traj, where: str) -> None:
    """Every geometry field holds, after every step, the bits it started with."""
    for step in traj:
        for (node, field), g in values["geometry"].items():
            got = np.asarray(step.state[node][field])
            assert got.dtype == g["start"].dtype and got.tobytes() == g["start"].tobytes(), (
                f"{where}: {node}.{field} moved")


def _assert_frozen_premises(topo, geometry, name: str) -> None:
    assert len(geometry.anchor) == sum(e.mapped for e in topo.edges) >= 1, name


def test_the_two_rotations_give_every_mapped_edge_both_anchors_and_both_modes():
    """The premise of the frozen sweeps: over rotations 0 and 2 each mapped
    edge of each named structure is anchored at its source and at its
    target, and is a gather once and a scatter once; and the per-push
    structure alone already has all four combinations."""
    for name, topo in NAMED.items():
        seen: dict = {}
        for variant in (0, 2):
            geometry = frozen_geometry(topo, variant)
            for i, anchor in geometry.anchor.items():
                seen.setdefault(i, set()).add((anchor, geometry.mode(i)))
        assert seen, name
        for i, combos in seen.items():
            assert {a for a, _m in combos} == {"source", "target"}, (name, i, combos)
            assert {m for _a, m in combos} == {"consistent", "conservative"}, (name, i, combos)
    ring = NAMED["chain-into-ring"]
    assert {(a, frozen_geometry(ring, v).mode(i)) for v in (0, 2)
            for i, a in frozen_geometry(ring, v).anchor.items()} == set(_COMBINATIONS)


_NAMED_PARAMS = [n if n == "chain-into-ring" else pytest.param(n, marks=pytest.mark.slow)
                 for n in sorted(NAMED)]


# Per push: tests/property/test_differential_geometry_edges.py::test_a_named_topology_with_frozen_geometry_edges_reproduces_the_monolithic_reference
# (on chain-into-ring, both rotations; the slow structures are the same check)
@pytest.mark.parametrize("variant", [0, 2])
@pytest.mark.parametrize("name", _NAMED_PARAMS)
def test_a_named_topology_with_frozen_geometry_edges_reproduces_the_monolithic_reference(
        name, variant):
    """Every mapped edge a frozen gather or scatter, anchored at either end:
    the graph passes the static case's oracles with ``H`` the reference
    stencil's matrix.  The two rotations give every edge both anchors and
    both modes."""
    topo, knobs, geometry, built = _frozen_built("f32", name, variant)
    _assert_frozen_premises(topo, geometry, name)
    model_kw = dict(node_order=built.node_order, group_cfgs=ct.group_cfgs_of(knobs),
                    geometry=geometry)
    for draw, values in enumerate(_frozen_values(topo, knobs, geometry, "f32")):
        model = ct.LinearModel(built.topo, values, **model_kw)
        traj = ct.run(built, values, 3)
        for k, step in enumerate(traj, start=1):
            where = f"{name} frozen variant {variant} draw {draw} step {k}"
            model.check_step(step.pre, step.state, step.reports,
                             thresholds=ct.thresholds_of(knobs), where=where)
            ct.check_leaves(built.topo, step.state, k, dividers=model.divider, where=where)
        _assert_points_never_moved(values, traj, f"{name} variant {variant}")


def _assert_close_runs(domain: str, knobs, a, b, what: str) -> None:
    """Two runs of the same linear maps, one through a gather / scatter and
    one through the dense matrix: the same to rounding, plus what the
    groups' tolerances allow where they converge."""
    dtype = np.dtype(topo_tests._domain_dtype(domain))                 # noqa: SLF001
    eps = float(jnp.finfo(dtype).eps)
    ulps = 16 if dtype.itemsize == 2 else ROUNDING_ULPS
    tolerance = max((float(g.get("tolerance", 1e-6)) if g.get("convergence_norm", "l2") == "l2"
                     else float(g.get("rtol", 1e-6)) * RMS_ENTRY_SLACK for g in knobs),
                    default=0.0)
    for k, (sa, sb) in enumerate(zip(a, b), start=1):
        scale = max(float(np.max(np.abs(np.asarray(f["x"], np.float64))))
                    for f in list(sa.state.values()) + list(sb.state.values()))
        allowed = k * (ulps * eps * scale + 2 * tolerance * max(scale, 1.0))
        for node in sb.state:
            gap = float(np.max(np.abs(np.asarray(sa.state[node]["x"], np.float64)
                                      - np.asarray(sb.state[node]["x"], np.float64))))
            assert gap <= allowed, (
                f"{what} step {k}: {node}.x through the frozen geometry is {gap:.3e} from "
                f"the static mapping's ({allowed:.3e} allowed)")
        # (The verdicts are not compared: these groups stop at a tolerance
        # near their dtype's floor, where one program reaches exact
        # stationarity and the other a two-state cycle an ulp wide.)
        assert set(sa.reports) == set(sb.reports)


_VMAPPED: dict = {}


def _batched_runs(built, values_list, steps: int) -> list:
    """One trajectory per draw, all through one ``jax.vmap`` of the step, each
    member from its own draw's state and parameters.

    (``set_initial`` rewrites the graph's state dictionary in place, so each
    draw's state is copied out before the next is written: a frozen
    geometry and the static matrix it is compared with must come from the
    same draw.)
    """
    gm = built.gm
    if id(gm) not in _VMAPPED:
        _VMAPPED[id(gm)] = (gm, jax.jit(jax.vmap(gm._raw_step_fn,          # noqa: SLF001
                                                 in_axes=(0, None, 0))))
    step = _VMAPPED[id(gm)][1]
    states, params = [], []
    for v in values_list:
        ct.set_initial(built, v)
        states.append({name: dict(fields) for name, fields in gm._state.items()})  # noqa: SLF001
        params.append(ct.params_for(built, v))
    state = jax.tree.map(lambda *xs: jnp.stack(xs), *states)
    batched_params = jax.tree.map(lambda *xs: jnp.stack(xs), *params)
    ext = gm._default_external_inputs()                                     # noqa: SLF001
    out = [[] for _ in values_list]
    pres = [None] * len(values_list)
    saved = gm._state                                                       # noqa: SLF001
    try:
        for _ in range(steps):
            state = step(state, ext, batched_params)
            for i in range(len(values_list)):
                gm._state = jax.tree.map(lambda x, i=i: x[i], state)        # noqa: SLF001
                snap = ct._snapshot(gm, {})                                 # noqa: SLF001
                diag = gm.coupling_diagnostics()
                reports = {gi: dict(diag[built.topo.group_key(gi)])
                           for gi in range(len(built.topo.groups))
                           if built.topo.group_key(gi) in diag}
                out[i].append(ct.Step(pres[i], snap, reports, {}))
                pres[i] = snap
    finally:
        gm._state = saved                                                   # noqa: SLF001
    return out


def _domain_runs(domain: str, built, values: list) -> list:
    if domain != "vmap":
        return topo_tests._runs_in(domain, built, values)                   # noqa: SLF001
    runs = _batched_runs(built, values, topo_tests._domain_steps(domain))   # noqa: SLF001
    topo_tests._assert_in_domain(domain, built, runs)                       # noqa: SLF001
    return runs


_DOMAIN_PARAMS = [(d, "chain-into-ring", 0) for d in topo_tests.DOMAINS] + [
    pytest.param(d, n, v, marks=pytest.mark.slow)
    for d in topo_tests.DOMAINS for n in sorted(NAMED) for v in (0, 2)
    if (n, v) != ("chain-into-ring", 0)]


# Per push: tests/property/test_differential_geometry_edges.py::test_a_frozen_geometry_equals_the_static_mapping_in_every_domain
# (on chain-into-ring, one rotation, in every domain; the slow parameters are
# the same check on the other structures and the other rotation)
@pytest.mark.parametrize("domain, name, variant", _DOMAIN_PARAMS)
def test_a_frozen_geometry_equals_the_static_mapping_in_every_domain(domain, name, variant):
    """The frozen-geometry graph against the static graph of the same
    structure and values, in *domain*; and in the domains the monolithic
    reference models, against that reference too."""
    topo, knobs, geometry, built = _frozen_built(domain, name, variant)
    _assert_frozen_premises(topo, geometry, name)
    _t, static_knobs, static = topo_tests._built_in(domain, name, 0)        # noqa: SLF001
    assert static_knobs == knobs
    values = _frozen_values(topo, knobs, geometry, domain)
    if len(geometry.anchor) >= 2:
        # (A structure's one mapped edge may gather from, or scatter onto, a
        # grid of a single point, whose matrix is all ones wherever the
        # points are; with two edges or more some matrix depends on them.)
        assert any(np.any((v["H"][i] > 0) & (v["H"][i] < 1)) for v in values
                   for i in geometry.anchor), "premise: a matrix that depends on its points"
    runs = _domain_runs(domain, built, values)
    twins = _domain_runs(domain, static, values)
    for draw, (v, a, b) in enumerate(zip(values, runs, twins)):
        what = f"{domain}/{name} variant {variant} draw {draw}"
        _assert_close_runs(domain, knobs, a, b, what)
        _assert_points_never_moved(v, a, what)
        want = ct.geometry_dtype(topo_tests._domain_dtype(domain))          # noqa: SLF001
        assert all(np.asarray(a[-1].state[n][f]).dtype == want for n, f in v["geometry"])
        if domain in topo_tests._REFERENCE_DOMAINS:                         # noqa: SLF001
            with gg.x64(domain == "f64"):
                model = ct.LinearModel(built.topo, v, node_order=built.node_order,
                                       group_cfgs=ct.group_cfgs_of(knobs), geometry=geometry,
                                       dtype=topo_tests._domain_dtype(domain))  # noqa: SLF001
                for k, step in enumerate(a, start=1):
                    model.check_step(step.pre, step.state, step.reports,
                                     thresholds=ct.thresholds_of(knobs),
                                     where=f"{what} step {k}")


# =============================================================================
# THE FROZEN IDENTITY UNDER THE INTERFACE NORM  (the static half now; the
# comparison waits for the geometry stage)
# =============================================================================
#
# ``compile()`` refuses the interface norm over a geometry edge in 0.4.0
# (phase 1, above), so the frozen identity has never been stated for that
# norm.  Two rules are about to be written for it -- which side of a
# *static* mapping the norm reads (``ct.INTERFACE_SIDE``; the decision of
# 2026-10-07 is the compact side: a target larger than its source is read
# at the source), and later what it reads of a geometry edge (a gather as
# delivered; a scatter at its source value, plus the geometry) -- and a
# frozen geometry edge *is* a static mapping, so the two must be one rule.
#
# * The static half runs today: one coupling pass of the static twin under
#   the interface norm reports the residual of the returned state against
#   the pre-step state, restated here from the two snapshots and the twin's
#   matrices under the tree's rule, and *not* under the other rule (each
#   case has a scatter onto a larger target, so the two differ).
# * The comparison runs for the ``multilinear_grid`` kind: the frozen graph
#   returns the static twin's states and verdict, and its residual is the
#   twin's with the positions a *source-anchored* scatter reads counted in
#   the pool (positions that do not move add entries and no change:
#   :func:`frozen_interface_entries`).  A scatter anchored at its target
#   reads no positions (the pre-step state is a constant of the solve), so
#   its residual is the static twin's itself.  The ``test_geom_matrix``
#   kind is still refused (it declares no length scale).

_ONE_PASS = dict(max_iterations=1, convergence_norm="interface", rtol=1e-6)
FROZEN_INTERFACE = [
    case("frozen group, multilinear, interface norm, one pass", kind="multilinear",
         dtype="float64", group=_ONE_PASS, **_FROZEN, **SOURCES),
    case("frozen group, multilinear, interface norm, one pass, Jacobi, P first",
         kind="multilinear", dtype="float64", order=("P", "F"),
         group=dict(_ONE_PASS, iteration_mode="jacobi"), **_FROZEN, **P_HOLDS),
    case("frozen group, matrix, interface norm, one pass", dtype="float64", group=_ONE_PASS,
         **_FROZEN, **F_HOLDS),
    case("frozen group, multilinear, interface norm, one pass, target anchors",
         kind="multilinear", dtype="float64", group=_ONE_PASS, **_FROZEN, **TARGETS),
    case("frozen group, multilinear 2-D, interface norm, one pass",
         kind="multilinear", d=2, dtype="float64", group=_ONE_PASS, **_FROZEN, **SOURCES),
]


#: How closely a frozen graph's residual is its static twin's: the two
#: sum the same numbers through another program (a kernel against a
#: matrix), so they differ by the rounding of the fields' dtype.
_FROZEN_RESIDUAL_RTOL = {"float64": 1e-9}


def _one_pass_steps(gm, steps: int) -> list:
    """``[(pre, post, report)]`` of *steps* steps from the initial state."""
    gm.reset_state()
    out = []
    for _ in range(steps):
        pre = gg.snapshot(gm)
        gm.step()
        out.append((pre, gg.snapshot(gm), dict(gm.coupling_diagnostics()["F+P"])))
    return out


def static_interface_residual(c: gg.Case, pre: dict, post: dict, rule: str):
    """``(residual, entries)`` of one pass of *c*'s static twin under *rule*.

    The interface norm of the returned state against the pre-step state:
    each internal edge's reading -- ``H x`` of its source where it is read
    as delivered, the source's ``x`` where *rule* is ``"compact"`` and the
    target is larger -- changes by ``rtol`` times its largest magnitude
    over both, pooled into one RMS.
    """
    assert rule in ("delivered", "compact"), rule
    rtol = float(c.knobs["rtol"])
    total, count = 0.0, 0
    for which, src in (("down", "F"), ("up", "P")):
        H = np.asarray(gg.static_matrix(c, which), np.float64)
        new, old = (np.asarray(s[src]["x"], np.float64) for s in (post, pre))
        if not (rule == "compact" and H.shape[0] > H.shape[1]):
            new, old = H @ new, H @ old
        ref = max(float(np.max(np.abs(new))), float(np.max(np.abs(old))))
        if ref > 0:
            total += float(np.sum(((new - old) / (rtol * ref)) ** 2))
            count += new.size
    return float(np.sqrt(total / max(count, 1))), count


def frozen_interface_entries(c: gg.Case, state: dict) -> int:
    """The entries a *frozen* geometry adds to the interface norm's pool.

    The decision for geometry edges: a gather is read as delivered, with
    nothing of its geometry; a scatter at its source value plus the
    positions it reads **from its source** (a source anchor: they move
    with the iterate), which do not move here and so add their entries to
    the count and nothing to the sum.  Positions are measured in grid
    spacings, not against their own magnitude, so they are counted
    wherever they are (the dead band does not apply to them).  A scatter
    anchored at its target reads the pre-step state, a constant of the
    solve: no entries.  The one place the two rules meet.
    """
    extra = 0
    for which in ("down", "up"):
        n_target, n_source = gg.static_matrix(c, which).shape
        if n_target <= n_source or (c.down if which == "down" else c.up) != "source":
            continue
        assert c.kind == "multilinear", c
        extra += int(np.asarray(state[gg.holder(c, which)]["pos"]).size)
    return extra


@pytest.mark.parametrize("c", FROZEN_INTERFACE, ids=repr)
def test_the_static_twin_of_a_frozen_case_reports_the_interface_residual_of_the_trees_rule(c):
    other = "compact" if ct.INTERFACE_SIDE == "delivered" else "delivered"
    with gg.x64(c.needs_x64):
        static = gg.build(gg.static_twin(c))
        steps = _one_pass_steps(static, c.steps)
    assert any(H.shape[0] > H.shape[1] for H in (gg.static_matrix(c, w) for w in ("down", "up")))
    for k, (pre, post, report) in enumerate(steps, start=1):
        assert int(report["iterations"]) == 1, (c.label, k, report)
        reported = float(report["residual"])
        restated, _n = static_interface_residual(c, pre, post, ct.INTERFACE_SIDE)
        assert abs(reported - restated) <= 1e-9 * restated, (
            f"{c.label} step {k}: the static twin reports {reported!r}; the "
            f"{ct.INTERFACE_SIDE} reading of its two states gives {restated!r}")
        wrong, _n = static_interface_residual(c, pre, post, other)
        assert abs(reported - wrong) > 1e-3 * reported, (
            f"{c.label} step {k}: the two rules read this case alike ({reported!r}, "
            f"{wrong!r}): it cannot tell them apart")


@pytest.mark.parametrize("c", [c for c in FROZEN_INTERFACE if gg.refused(c)], ids=repr)
def test_a_frozen_geometry_of_another_kind_is_still_refused_under_the_interface_norm(c):
    with gg.x64(c.needs_x64):
        gg.assert_interface_norm_refused(lambda: gg.build(gg.two_body(c)),
                                         GEOMETRY_EDGE_KEYS(c), "kind")


def test_the_frozen_interface_cases_hold_both_anchors_of_a_scatter():
    """The premise of the comparison below: a scatter anchored at its source
    (its positions are counted) and one anchored at its target (they are
    not), and a grid of two axes."""
    running = [c for c in FROZEN_INTERFACE if not gg.refused(c)]
    assert {c.up for c in running} == {"source", "target"}
    assert {c.d for c in running} == {1, 2} and len(running) == 4


@pytest.mark.parametrize("c", [c for c in FROZEN_INTERFACE if not gg.refused(c)], ids=repr)
def test_a_frozen_geometry_reports_as_its_static_mapping_under_the_interface_norm(c):
    with gg.x64(c.needs_x64):
        frozen = gg.build(gg.two_body(c))
        static = gg.build(gg.static_twin(c))
        a, b = _one_pass_steps(frozen, c.steps), _one_pass_steps(static, c.steps)
    for k, ((_pre_a, post_a, ra), (pre_b, post_b, rb)) in enumerate(zip(a, b), start=1):
        assert_same_states(c, post_a, post_b, step=k)
        assert int(ra["iterations"]) == int(rb["iterations"]) == 1, (c.label, k, ra, rb)
        assert bool(ra["converged"]) == bool(rb["converged"]), (c.label, k, ra, rb)
        _res, entries = static_interface_residual(c, pre_b, post_b, ct.INTERFACE_SIDE)
        extra = frozen_interface_entries(c, post_a)
        assert (extra > 0) == (c.up == "source"), (c.label, extra)
        want = float(rb["residual"]) * np.sqrt(entries / (entries + extra))
        assert abs(float(ra["residual"]) - want) <= _FROZEN_RESIDUAL_RTOL[c.dtype] * want, (
            f"{c.label} step {k}: the frozen graph reports residual {ra['residual']!r}; "
            f"its static twin's {rb['residual']!r} over {entries} entries, with the "
            f"{extra} of the geometry the scatter reads, is {want!r}")
