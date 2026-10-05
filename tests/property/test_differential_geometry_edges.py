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
* ``interface norm, three passes``: the interface norm and the
  quasi-Newton interface set must read a source-anchored geometry as they
  read the value, and three passes from convergence the residual says so.

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
#: A solve that stops on its criterion returns a state within its
#: tolerance of the fixed point by its own *estimate*; this many times the
#: tolerance is allowed for the estimate's error and for entries of a
#: root-mean-square norm.
TOLERANCE_SLACK = 32

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
    case("sub-cycled target, one pass", dt_p=DT / 4, adv=0.3, **SOURCES,
         group=dict(_SUB, max_iterations=1, boundary_interpolation="linear")),
    case("interface norm, three passes", adv=0.6, dtype="float64", **P_HOLDS,
         group=dict(max_iterations=3, convergence_norm="interface", rtol=1e-6)),
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
    case("group IQN-ILS, three passes", adv=0.6, dtype="float64", **P_HOLDS,
         group=dict(max_iterations=3, tolerance=1e-13, acceleration="iqn-ils")),
    case("group IQN-IMVJ, interface norm", adv=0.3, dtype="float64", **SOURCES,
         group=dict(max_iterations=400, acceleration="iqn-imvj", jacobian_reuse=2,
                    convergence_norm="interface", rtol=1e-11)),
    case("group interface norm, Jacobi, three passes", adv=0.6, dtype="float64", **SOURCES,
         group=dict(max_iterations=3, convergence_norm="interface", rtol=1e-6,
                    iteration_mode="jacobi")),
    case("group, fori solver", adv=0.3, dtype="float64", **P_HOLDS,
         group=dict(max_iterations=120, tolerance=1e-13, solver="fori")),
    case("group, two passes", group=dict(max_iterations=2), adv=0.3, **TARGETS),
    case("group, a member's flux scattered inside it", group=_TIGHT32, flux="internal",
         adv=0.3, **P_HOLDS),
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
GRADIENTS_PER_PUSH = [PER_PUSH[0], PER_PUSH[2]]
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
# The comparison
# ---------------------------------------------------------------------------


def _solve_tolerance(c: gg.Case, scale: float) -> float:
    """What a converging group's criterion allows a field of size *scale* to be off by."""
    knobs = c.knobs
    if not knobs or knobs.get("max_iterations", 10) <= 4:
        return 0.0          # a fixed few passes: the same arithmetic in both graphs
    if knobs.get("convergence_norm", "l2") == "l2":
        return float(knobs.get("tolerance", 1e-6))
    return float(knobs.get("rtol", 1e-6)) * scale


def assert_same_states(c: gg.Case, a: dict, b: dict, *, step: int) -> float:
    """Every field of every node, within rounding plus what the solve allows; the worst gap."""
    worst = 0.0
    for name in a:
        for field in a[name]:
            x, y = np.asarray(a[name][field]), np.asarray(b[name][field])
            assert x.dtype == y.dtype and x.shape == y.shape, (name, field, x.dtype, y.dtype)
            scale = float(max(np.max(np.abs(x)), np.max(np.abs(y))))
            allowed = (ROUNDING_ULPS * float(np.finfo(x.dtype).eps) * scale
                       + 2 * TOLERANCE_SLACK * step * _solve_tolerance(c, scale))
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


@pytest.mark.parametrize("c", PER_PUSH, ids=repr)
def test_an_edge_mapped_graph_steps_as_its_node_inlined_twin(c):
    """Per push; slow sibling :func:`test_every_drawn_edge_mapped_graph_steps_as_its_node_inlined_twin`."""
    assert_steps_as_its_inlined_twin(c)


# Slow: two graphs compiled per case, forty cases.
# Per push: tests/property/test_differential_geometry_edges.py::test_an_edge_mapped_graph_steps_as_its_node_inlined_twin
@pytest.mark.slow
@pytest.mark.parametrize("c", SLOW, ids=repr)
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


@pytest.mark.parametrize("c", GRADIENTS_PER_PUSH, ids=repr)
def test_the_gradients_of_an_edge_mapped_graph_are_its_node_inlined_twin_s(c):
    """Per push; slow sibling :func:`test_the_gradients_of_every_drawn_edge_mapped_graph_are_its_twin_s`."""
    assert_same_gradients(c)


# Slow: a gradient through three steps of two graphs compiled per case.
# Per push: tests/property/test_differential_geometry_edges.py::test_the_gradients_of_an_edge_mapped_graph_are_its_node_inlined_twin_s
@pytest.mark.slow
@pytest.mark.parametrize("c", GRADIENTS_SLOW, ids=repr)
def test_the_gradients_of_every_drawn_edge_mapped_graph_are_its_twin_s(c):
    assert_same_gradients(c)


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


@pytest.mark.parametrize("c", FROZEN_CASES, ids=repr)
def test_a_frozen_geometry_steps_as_the_static_mapping_of_the_same_points(c):
    """Per push; slow sibling :func:`test_every_frozen_geometry_steps_as_its_static_mapping`."""
    assert_frozen_equals_static(c)


# Slow: two graphs and two gradients compiled per case.
# Per push: tests/property/test_differential_geometry_edges.py::test_a_frozen_geometry_steps_as_the_static_mapping_of_the_same_points
@pytest.mark.slow
@pytest.mark.parametrize("c", FROZEN_SLOW, ids=repr)
def test_every_frozen_geometry_steps_as_its_static_mapping(c):
    assert_frozen_equals_static(c)


#: ``(anchor, mode)`` of a topology's mapped edges, in rotation.
_COMBINATIONS = (("source", "consistent"), ("target", "conservative"),
                 ("target", "consistent"), ("source", "conservative"))


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


def _assert_points_never_moved(starts: list, traj, where: str) -> None:
    """Every geometry field holds, after every step, the bits one of the
    drawn *starts* gave it.  (One of them, not a particular one: the
    harness's batched runs start every member of the batch from the last
    draw's state.)"""
    for node, field in starts[0]["geometry"]:
        first = np.asarray(traj[0].state[node][field])
        candidates = [v["geometry"][(node, field)]["start"] for v in starts]
        assert any(first.dtype == c.dtype and first.tobytes() == c.tobytes()
                   for c in candidates), f"{where}: {node}.{field} is not where it started"
        for step in traj:
            assert np.asarray(step.state[node][field]).tobytes() == first.tobytes(), (
                f"{where}: {node}.{field} moved")


def _assert_frozen_premises(topo, geometry, name: str) -> None:
    assert len(geometry.anchor) == sum(e.mapped for e in topo.edges) >= 2, name
    assert {geometry.mode(i) for i in geometry.anchor} == {"consistent", "conservative"}
    assert set(geometry.anchor.values()) == {"source", "target"}


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
        _assert_points_never_moved([values], traj, f"{name} variant {variant}")


def _assert_close_runs(domain: str, knobs, a, b, what: str) -> None:
    """Two runs of the same linear maps, one through a gather / scatter and
    one through the dense matrix: the same to rounding, plus what the
    groups' tolerances allow where they converge."""
    dtype = np.dtype(topo_tests._domain_dtype(domain))                 # noqa: SLF001
    eps = float(jnp.finfo(dtype).eps)
    ulps = 16 if dtype.itemsize == 2 else ROUNDING_ULPS
    tolerance = max((float(g.get("tolerance", 1e-6)) if g.get("convergence_norm", "l2") == "l2"
                     else float(g.get("rtol", 1e-6)) for g in knobs), default=0.0)
    for k, (sa, sb) in enumerate(zip(a, b), start=1):
        scale = max(float(np.max(np.abs(np.asarray(f["x"], np.float64))))
                    for f in list(sa.state.values()) + list(sb.state.values()))
        allowed = k * (ulps * eps * scale + 2 * TOLERANCE_SLACK * tolerance * max(scale, 1.0))
        for node in sb.state:
            gap = float(np.max(np.abs(np.asarray(sa.state[node]["x"], np.float64)
                                      - np.asarray(sb.state[node]["x"], np.float64))))
            assert gap <= allowed, (
                f"{what} step {k}: {node}.x through the frozen geometry is {gap:.3e} from "
                f"the static mapping's ({allowed:.3e} allowed)")
        assert set(sa.reports) == set(sb.reports)
        for gi in sa.reports:
            assert bool(sa.reports[gi]["converged"]) == bool(sb.reports[gi]["converged"]) or (
                dtype.itemsize == 2), (what, k, gi, sa.reports[gi], sb.reports[gi])


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
    runs = topo_tests._runs_in(domain, built, values)                       # noqa: SLF001
    twins = topo_tests._runs_in(domain, static, values)                     # noqa: SLF001
    for draw, (v, a, b) in enumerate(zip(values, runs, twins)):
        what = f"{domain}/{name} variant {variant} draw {draw}"
        _assert_close_runs(domain, knobs, a, b, what)
        _assert_points_never_moved(values, a, what)
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
