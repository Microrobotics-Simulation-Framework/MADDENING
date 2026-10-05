"""The time level a geometry-dependent mapping reads its geometry at, against an exact solve.

A graph of linear relays stays one linear system per step when a mapped
edge multiplies by a matrix held in a relay's state
(``test_geom_matrix``: the geometry *is* the matrix) and that matrix moves
by itself, ``M <- M + dM``.  Which matrix the edge applies is then the
whole question, and the answer is a rule per call site:

==========  =====================================  ==============================
anchor      where the edge is resolved             the matrix
==========  =====================================  ==============================
source      forward edge (uncoupled, or from       the source's, after this step
            outside a group)
source      back edge                              the source's, before the step
source      inside a group, at the fixed point     the source's, after this step
target      uncoupled, or a member not sub-cycled  the target's, before the step
target      sub-step ``k`` of a sub-cycled member  the target's after ``k`` of
                                                   its own updates
either      a node that does not fire              whatever the holder holds
            (multi-rate)                           between its firings
==========  =====================================  ==============================

:class:`~tests.property.coupled_topologies.LinearModel` restates that
table (``bind``), repeats the matrices' float additions in the graph's
dtype -- so they are the graph's bit for bit, which is asserted -- and
solves the step in extended precision.  Every node must then sit within
its own float rounding of its update at those matrices, every group within
what its reported residual allows, and the whole state within the
propagated allowance of the exact solve
(:meth:`~tests.property.coupled_topologies.LinearModel.check_step`): the
oracles of the static case, unchanged.

Both ends of every such edge hold a matrix of the same name and shape, so
an anchor read from the wrong end is another matrix.

**During a pass** a sub-cycled member reads more than a fixed point shows:
under linear interpolation the geometry of a source swept before it is
interpolated between the incoming iterate's and this pass's, with the
value's own weight, and the mapping is applied to the interpolated pair.
At a converged fixed point the two end points coincide, so that is checked
on a single pass (``max_iterations=1``) against
:func:`~tests.property.coupled_topologies.single_pass`, an explicit loop
over the members and their sub-steps.

Groups that hold a moving matrix run without acceleration here: a relaxed
iterate's matrix is not the updated one bit for bit.  Accelerated groups
are compared with their node-inlined twins in
``test_differential_geometry_edges.py``.
"""

from __future__ import annotations

import functools
import itertools

import numpy as np
import pytest

from tests.property import coupled_topologies as ct
from tests.property.geometry_graphs import x64

#: ``(seed, group rate)`` of the fixed draws.
_DRAWS = ((3, 0.3), (17, 0.6), (29, 0.5))
_STEPS = 3
_DTYPES = ["float32", pytest.param("float64", marks=pytest.mark.slow)]


def _quiet(**kw) -> dict:
    """A converging group without acceleration (see the module docstring)."""
    return dict(dict(acceleration="none", iteration_mode="gauss-seidel", convergence_norm="l2",
                     tolerance=1e-6, max_iterations=200), **kw)


# ---------------------------------------------------------------------------
# Structures: every edge between relays of different sizes, so every edge is mapped
# ---------------------------------------------------------------------------


def _forward():
    """``a -> b`` and nothing back: one forward edge."""
    b = ct.TopologyBuilder()
    b.node("a", 2, alpha=0.5, beta=1.0)
    b.node("b", 3, alpha=0.25)
    b.edge("a", "b")
    return b.build("forward")


def _ungrouped_cycle():
    """``a <-> b`` in no group: built ``a, b``, ``a -> b`` is read this step and ``b -> a`` late."""
    b = ct.TopologyBuilder()
    b.node("a", 2, alpha=0.5, beta=1.0)
    b.node("b", 3, alpha=0.25, beta=-0.5)
    b.edge("a", "b")
    b.edge("b", "a")
    return b.build("ungrouped-cycle")


def _group_in_a_loop():
    """A two-member group with a driver, and an outside node in its loop.

    Edges 0 ``d -> g0`` (forward, from outside), 1 ``g0 -> g1`` and 2 ``g1
    -> g0`` (inside the group), 3 ``g1 -> o`` (forward, out of the group)
    and 4 ``o -> g0`` (a back edge: ``o`` runs after the group's block).
    """
    b = ct.TopologyBuilder()
    b.node("d", 2, alpha=1.0, beta=0.5)
    b.node("g0", 3, alpha=0.5, beta=1.0)
    b.node("g1", 2, alpha=-0.25)
    b.node("o", 1, alpha=0.5)
    b.edge("d", "g0")
    b.edge("g0", "g1")
    b.edge("g1", "g0")
    b.edge("g1", "o")
    b.edge("o", "g0")
    b.group("g0", "g1")
    return b.build("group-in-a-loop")


def _subcycled(divider: int, fast: str = "g1"):
    """A two-member group one member of which takes *divider* sub-steps, and a driver into it."""
    b = ct.TopologyBuilder()
    b.node("d", 3, alpha=1.0, beta=0.5)
    b.node("g0", 2, alpha=0.5, beta=1.0)
    b.node("g1", 3, alpha=-0.25, beta=0.5)
    b.edge("g0", "g1")
    b.edge("g1", "g0")
    b.edge("d", fast, mapped=True)
    b.group("g0", "g1")
    return b.build(f"sub-cycled-{fast}").with_timesteps({fast: 1.0 / divider})


def _two_rates():
    """``s <-> f`` in no group, ``f`` at half ``s``'s step: ``s`` fires every other base step."""
    b = ct.TopologyBuilder()
    b.node("s", 2, alpha=0.5, beta=1.0)
    b.node("f", 3, alpha=0.25, beta=-0.5)
    b.edge("s", "f")
    b.edge("f", "s")
    return b.build("two-rates").with_timesteps({"f": 0.5})


def _anchors(topo, pattern: str):
    """A ``"moving"`` geometry on every mapped edge; ``pattern`` gives each its
    anchor (``s`` / ``t``), repeated over the edges."""
    mapped = [i for i, e in enumerate(topo.edges) if e.mapped]
    return ct.Geometry("moving", tuple(
        (i, "source" if pattern[k % len(pattern)] == "s" else "target")
        for k, i in enumerate(mapped)))


# ``(structure, node order, group knobs, anchor pattern)``
CASES = {
    "forward, source anchor": (_forward, None, [], "s"),
    "forward, target anchor, reader built first": (_forward, ("b", "a"), [], "t"),
    "ungrouped cycle, source anchors": (_ungrouped_cycle, None, [], "s"),
    "ungrouped cycle, target anchors": (_ungrouped_cycle, None, [], "t"),
    "ungrouped cycle, mixed anchors, built b, a": (_ungrouped_cycle, ("b", "a"), [], "st"),
    "group, source anchors": (_group_in_a_loop, None, [_quiet()], "s"),
    "group, target anchors, Jacobi, mixed norm": (
        _group_in_a_loop, None,
        [_quiet(iteration_mode="jacobi", convergence_norm="mixed", rtol=1e-4)], "t"),
    "group, mixed anchors, interface norm": (
        _group_in_a_loop, None, [_quiet(convergence_norm="interface", rtol=1e-4)], "sst"),
    "group, mixed anchors, fori": (
        _group_in_a_loop, None, [_quiet(solver="fori", max_iterations=100, diagnostics=True)],
        "ts"),
    "sub-cycled target, linear, target anchors": (
        functools.partial(_subcycled, 2), None,
        [_quiet(subcycling=True, boundary_interpolation="linear")], "t"),
    "sub-cycled target, linear, source anchors": (
        functools.partial(_subcycled, 2), None,
        [_quiet(subcycling=True, boundary_interpolation="linear")], "s"),
    "sub-cycled four times, constant, mixed anchors": (
        functools.partial(_subcycled, 4), None,
        [_quiet(subcycling=True, boundary_interpolation="constant")], "ts"),
    "sub-cycled, Jacobi, mixed norm": (
        functools.partial(_subcycled, 2), None,
        [_quiet(subcycling=True, boundary_interpolation="linear", iteration_mode="jacobi",
                convergence_norm="mixed", rtol=1e-4)], "st"),
    "sub-cycled first member, linear": (
        functools.partial(_subcycled, 2, "g0"), None,
        [_quiet(subcycling=True, boundary_interpolation="linear")], "ts"),
    "two rates, source anchors": (_two_rates, None, [], "s"),
    "two rates, target anchors": (_two_rates, None, [], "t"),
    "two rates, mixed anchors, built f, s": (_two_rates, ("f", "s"), [], "st"),
}
#: Per push: one of each call site.  The rest is the same check on the
#: other anchors, schedules and norms.
_PER_PUSH = ("forward, source anchor", "ungrouped cycle, mixed anchors, built b, a",
             "group, mixed anchors, interface norm",
             "sub-cycled target, linear, target anchors", "two rates, source anchors")
_CASE_PARAMS = [name if name in _PER_PUSH else pytest.param(name, marks=pytest.mark.slow)
                for name in CASES]


@functools.lru_cache(maxsize=8)
def _built(name: str, dtype: str):
    make, order, knobs, pattern = CASES[name]
    topo = make()
    geometry = _anchors(topo, pattern)
    with x64(dtype == "float64"):
        built = ct.build(topo, knobs, dtype=dtype, node_order=order, geometry=geometry)
    return built, knobs, geometry


def _values(built, knobs, geometry, dtype):
    with x64(dtype == "float64"):
        return [ct.draw_values(built.topo, np.random.default_rng(seed), rho, dtype=dtype,
                               group_cfgs=ct.group_cfgs_of(knobs), geometry=geometry)
                for seed, rho in _DRAWS]


def _firing(topo, step_index: int):
    """The nodes whose update base step *step_index* applies, on a graph
    without groups: a node fires when the step count is a multiple of its
    timestep over the smallest one."""
    base = min(nd.timestep for nd in topo.nodes)
    return {nd.name for nd in topo.nodes if step_index % round(nd.timestep / base) == 0}


def _assert_same_bits(got, want, what: str) -> None:
    got, want = np.asarray(got), np.asarray(want)
    assert got.dtype == want.dtype and got.shape == want.shape, (what, got.dtype, want.dtype)
    assert got.tobytes() == want.tobytes(), (
        f"{what}: {np.max(np.abs(got.astype(np.float64) - want.astype(np.float64))):.3e} "
        f"from the value the relay's own additions give")


def assert_steps_on_the_time_level_table(name: str, dtype: str, steps: int = _STEPS) -> None:
    built, knobs, geometry = _built(name, dtype)
    topo = built.topo
    multirate = len({nd.timestep for nd in topo.nodes}) > 1 and not topo.groups
    steps = 4 if multirate else steps
    assert bool(built.gm._is_multirate) == multirate, "premise: the graph's rates"  # noqa: SLF001
    for draw, values in enumerate(_values(built, knobs, geometry, dtype)):
        with x64(dtype == "float64"):
            traj = ct.run(built, values, steps)
            model = ct.LinearModel(topo, values, node_order=built.node_order, dtype=dtype,
                                   group_cfgs=ct.group_cfgs_of(knobs), geometry=geometry)
            for k, step in enumerate(traj):
                where = f"{name} ({dtype}) draw {draw} step {k + 1}"
                firing = _firing(topo, k) if multirate else None
                expected = model.bind(step.pre, firing)
                assert expected, "premise: moving matrices"
                for (node, field), want in expected.items():
                    _assert_same_bits(step.state[node][field], want, f"{where}: {node}.{field}")
                for i in geometry.anchor:
                    e = topo.edges[i]
                    assert np.any(step.state[e.src][geometry.field(i)]
                                  != step.state[e.dst][geometry.field(i)]), (
                        "premise: the two ends hold different matrices")
                if not multirate:
                    assert any(np.any(step.state[n][f] != step.pre[n][f])
                               for n, f in expected), "premise: the matrices move"
                model.check_step(step.pre, step.state, step.reports,
                                 thresholds=ct.thresholds_of(knobs), where=where)
                for gi in range(len(topo.groups)):
                    # From the second pass on the iterate's matrices are the
                    # updated ones, which is what the table's group row says.
                    assert int(step.reports[gi]["iterations"]) >= 2, (where, step.reports[gi])


# Per push: tests/property/test_geometry_time_levels.py::test_a_moving_geometry_is_read_at_its_documented_time_level
# (one case per call site, in float32; the slow parameters are the same check
# on the other anchors, schedules and norms, and in float64)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("name", _CASE_PARAMS)
def test_a_moving_geometry_is_read_at_its_documented_time_level(name, dtype):
    """Every row of the module's table, against the exact solve."""
    assert_steps_on_the_time_level_table(name, dtype)


def test_the_cases_cover_every_row_of_the_table():
    """Each row of the module's table is drawn, with both anchors."""
    seen = set()
    for name, (make, order, knobs, pattern) in CASES.items():
        topo = make()
        geometry = _anchors(topo, pattern)
        back = ct.documented_back_edges(topo, order or topo.names)
        rates = len({nd.timestep for nd in topo.nodes}) > 1
        for i, anchor in geometry.anchor.items():
            e = topo.edges[i]
            if topo.internal(e):
                sub = bool(knobs[0].get("subcycling")) and topo.node(e.dst).timestep < 1.0
                site = "sub-cycled target" if sub else "group"
            elif rates and not topo.groups:
                site = "multi-rate"
            else:
                site = "back" if i in back else "forward"
            seen.add((site, anchor))
    assert seen >= set(itertools.product(
        ("forward", "back", "group", "sub-cycled target", "multi-rate"), ("source", "target")))


# ---------------------------------------------------------------------------
# During a pass: the single-pass reference
# ---------------------------------------------------------------------------

#: ``(sub-steps, fast member, schedule, interpolation, anchors)``
SINGLE_PASS = {
    "target sub-cycled, linear, source anchors": (4, "g1", "gauss-seidel", "linear", "s"),
    "target sub-cycled, linear, target anchors": (4, "g1", "gauss-seidel", "linear", "t"),
    "target sub-cycled, constant, source anchors": (4, "g1", "gauss-seidel", "constant", "s"),
    "target sub-cycled, Jacobi, mixed anchors": (2, "g1", "jacobi", "linear", "st"),
    "first member sub-cycled, linear, mixed anchors": (4, "g0", "gauss-seidel", "linear", "ts"),
}
_SINGLE_PARAMS = [name if k < 2 else pytest.param(name, marks=pytest.mark.slow)
                  for k, name in enumerate(SINGLE_PASS)]


@functools.lru_cache(maxsize=4)
def _built_single(name: str, dtype: str):
    divider, fast, mode, interp, pattern = SINGLE_PASS[name]
    topo = _subcycled(divider, fast)
    geometry = _anchors(topo, pattern)
    knobs = [dict(acceleration="none", iteration_mode=mode, subcycling=True,
                  boundary_interpolation=interp, max_iterations=1)]
    with x64(dtype == "float64"):
        built = ct.build(topo, knobs, dtype=dtype, geometry=geometry)
    return built, knobs, geometry


# Per push: tests/property/test_geometry_time_levels.py::test_a_single_pass_reads_the_interpolated_geometry_at_every_sub_step
# (the two linear Gauss-Seidel cases, in float32; the slow parameters are the
# same check under the other schedule and interpolation, and in float64)
@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("name", _SINGLE_PARAMS)
def test_a_single_pass_reads_the_interpolated_geometry_at_every_sub_step(name, dtype):
    """``max_iterations=1``: the returned state is one pass from the pre-step
    state, which :func:`~tests.property.coupled_topologies.single_pass`
    evaluates sub-step by sub-step.  The interpolated *matrix* times the
    interpolated *value* is bilinear in the weight: neither "the geometry
    is not interpolated" nor "the mapping is applied before the
    interpolation" gives the same number."""
    built, knobs, geometry = _built_single(name, dtype)
    topo = built.topo
    eps = float(np.finfo(np.dtype(dtype)).eps)
    for draw, values in enumerate(_values(built, knobs, geometry, dtype)):
        with x64(dtype == "float64"):
            traj = ct.run(built, values, _STEPS)
            model = ct.LinearModel(topo, values, node_order=built.node_order, dtype=dtype,
                                   group_cfgs=ct.group_cfgs_of(knobs), geometry=geometry)
        for k, step in enumerate(traj):
            where = f"{name} ({dtype}) draw {draw} step {k + 1}"
            assert int(step.reports[0]["iterations"]) == 1, (where, step.reports[0])
            want = ct.single_pass(model, 0, step.pre, step.state)
            for m, fields in want.items():
                d = model.divider.get(m, 1)
                allowed = d * (model._flops(m) + 8) * eps * fields["magnitude"]  # noqa: SLF001
                gap = np.abs(np.asarray(
                    np.asarray(step.state[m]["x"], np.float64) - fields["x"], np.float64))
                assert np.all(gap <= allowed), (
                    f"{where}: member {m!r} is {gap} from one pass at the documented "
                    f"sub-step time levels (rounding allows {allowed})")
                for field, _shape, _moves in topo.node(m).geom:
                    _assert_same_bits(step.state[m][field], fields[field],
                                      f"{where}: {m}.{field}")


# ---------------------------------------------------------------------------
# The inspection helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["s", "t"])
def test_resolve_boundary_inputs_reads_the_geometry_of_the_current_state(pattern):
    """``resolve_boundary_inputs`` reads every edge from the current state,
    whatever its direction: a source-anchored geometry from the source's
    state, a target-anchored one from the state an ``update`` of the node
    starting now would receive."""
    topo = _ungrouped_cycle()
    geometry = _anchors(topo, pattern)
    built = ct.build(topo, [], geometry=geometry)
    values = ct.draw_values(built.topo, np.random.default_rng(5), 0.5, geometry=geometry)
    ct.run(built, values, 2)
    gm = built.gm
    state = ct._snapshot(gm, {})  # noqa: SLF001
    for i, e in enumerate(built.topo.edges):
        holder = geometry.holder(built.topo, i)
        want = (np.asarray(state[holder][geometry.field(i)], np.float64)
                @ np.asarray(state[e.src]["x"], np.float64))
        got = np.asarray(gm.resolve_boundary_inputs(e.dst)[f"u{e.port}"], np.float64)
        assert np.allclose(got, want, rtol=0, atol=16 * float(np.finfo(np.float32).eps)
                           * float(np.max(np.abs(want)) + 1.0)), (e, got, want)
