"""``converged=True`` puts every member's state where the group's tolerance says, under every schedule and norm.

A converged coupling group promises that what its norm measures has stopped
moving to within its threshold.  The state a user reads is every member's
every field, and whether *that* is near the group's fixed point depends on
what the norm leaves out: ``"l2"`` and ``"mixed"`` measure the state, and
``"interface"`` what the internal edges deliver.  The existing oracles
(:meth:`~tests.property.coupled_topologies.LinearModel.check_step`) hold a
group to its reported residual *in its own norm*, so a member the interface
norm does not read had no allowance and was skipped -- and a one-way group
under Jacobi returned such a member computed from the pre-step value of
what it reads, converged (MADD-ANO-238).

**The property.**  For a group of linear relays the fixed point is a
float64 solve.  Each member's *defect* at the returned state -- its value
minus its update evaluated at the values the other members were returned
with -- is what the tolerance bounds, member by member:

* a field the norm measures whole moved at most ``tau`` of its magnitude
  between the returned iterate and its successor, and that move is the
  defect of a member the pass computes from the iterate (Jacobi), or the
  defect up to the moves of the members swept before it (Gauss-Seidel);
* a field the interface norm does not measure whole -- read by no
  internal edge (MADD-ANO-238), or only through a mapping, which may
  deliver less than the field (MADD-ANO-239: the ``lossy-ring`` shape) --
  is returned as one plain pass computes it at the accepted iterate, from
  the readings the verdict compared, and the other members read it within
  ``tau`` of those; so its defect is at most its gains times that.

**The return rule itself** (last section): on the exact model, for any
iterate ``x``, the state ``x + P (F(x) - x)`` reads within the residual of
``x`` of what ``x`` reads, and its defect is ``-[(I - L) - (I - M) P]`` of
that residual; and the library returns that state for the iterate it
accepted, beside a report that is the accepted iterate's.

So for every member ``|defect| <= SLACK tau (|x| + sum_e |G_e| |reading_e|)``
with ``tau`` the per-entry change the threshold allows (``tolerance`` under
``"l2"``; ``rtol sqrt(entries the norm reads)`` under the RMS norms), and
the state is within the resolvent ``|(I - M)^-1|`` of those allowances of
the fixed point.  A stale member's defect is its gain times the whole
change of what it reads -- ``1 / tau`` beyond the allowance.

**The shapes** are the ones a group can hide a member on: a one-way pair,
plain and mapped; a chain; a cycle with a one-way tail, the tail added
first or last; a cycle with a head; a ring whose mapped edges deliver
fewer entries than the fields they read.  Values are arguments of the
compiled step, so a cell compiles once: the loop gain from 0.005 (where the
pass before the one that meets the tolerance is up to two hundred
tolerances away) to 0.9, and the start far from the fixed point.

Per push: every (schedule, norm) pair on three shapes under the default
solver.  Slow: every shape, both solvers, and the accelerations.
"""

from __future__ import annotations

import functools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.property import coupled_graphs as cg
from tests.property import coupled_topologies as ct

RTOL = 1e-4
#: Twice the allowance: the scale of each change is the larger of the two
#: iterates' magnitudes, and the oracle reads the returned one.
SLACK = 2.0
#: Rounding of one float32 update, in units of ``eps`` times the terms summed.
ROUNDING_ULPS = 64

SCHEDULES = ("jacobi", "gauss-seidel")
NORMS = ("interface", "mixed", "l2")


def _topologies() -> dict:
    N, E, T = ct.TNode, ct.TEdge, ct.Topology

    def topo(label, nodes, edges):
        return T(nodes=tuple(nodes), edges=tuple(edges),
                 groups=(tuple(nd.name for nd in nodes),), label=label)

    cycle = [E("a", "b", 0), E("b", "a", 0)]
    return {
        "one-way": topo("one-way", [N("a", 2), N("b", 2, ports=1)], [E("a", "b", 0)]),
        "one-way-mapped": topo("one-way-mapped", [N("a", 3), N("b", 2, ports=1)],
                               [E("a", "b", 0, mapped=True)]),
        "chain": topo("chain", [N("a", 2), N("b", 2, ports=1), N("c", 2, ports=1)],
                      [E("a", "b", 0), E("b", "c", 0)]),
        "cycle-tail": topo("cycle-tail",
                           [N("a", 2, ports=1), N("b", 2, ports=1), N("c", 2, ports=1)],
                           cycle + [E("b", "c", 0)]),
        "tail-first": topo("tail-first",
                           [N("c", 2, ports=1), N("a", 2, ports=1), N("b", 2, ports=1)],
                           cycle + [E("b", "c", 0)]),
        "head-cycle": topo("head-cycle",
                           [N("h", 2), N("a", 2, ports=2), N("b", 2, ports=1)],
                           [E("a", "b", 0), E("b", "a", 0), E("h", "a", 1)]),
        # Each field of three is delivered as two entries: one direction of
        # it is computed by every pass and measured by none.
        "lossy-ring": topo("lossy-ring", [N("a", 3, ports=1), N("b", 3, ports=1)],
                           [E("a", "b", 0, mapped=True), E("b", "a", 0, mapped=True)]),
    }


TOPOLOGIES = _topologies()
#: The mapped edges of ``lossy-ring`` deliver two entries of three.
_DELIVERED = {"lossy-ring": 2}
PER_PUSH = ("one-way-mapped", "cycle-tail", "lossy-ring")


def _cells(names):
    return [(name, schedule, norm) for name in names for schedule in SCHEDULES
            for norm in NORMS]


def _knobs(schedule, norm, solver="ift", **extra) -> dict:
    knobs = dict(acceleration="none", iteration_mode=schedule, convergence_norm=norm,
                 max_iterations=400, solver=solver, diagnostics=True, **extra)
    knobs["tolerance" if norm == "l2" else "rtol"] = RTOL
    return knobs


@functools.lru_cache(maxsize=None)
def _built(name: str, schedule: str, norm: str, solver: str = "ift", extra: tuple = ()):
    topo = TOPOLOGIES[name]
    knobs = _knobs(schedule, norm, solver, **dict(extra))
    return topo, knobs, ct.build(topo, knobs)


#: The one-way tail of each cyclic shape.
_TAIL = {"cycle-tail": "c", "tail-first": "c"}


def _values(topo, knobs, seed: int, rho: float, scale: float) -> dict:
    """Drawn values with the loop at rate *rho* and what hangs off it at order one.

    ``draw_values`` scales every internal gain by one factor, so at a weak
    loop a tail, or the part of a field no edge delivers, would be as
    insensitive to the loop as the loop is to itself, and a reading one
    pass old would not show in it.  A tail's gain is therefore redrawn at
    order one, and a lossy member gets an order-one gain onto the
    direction of its field its mapping loses; neither feeds back, so the
    loop's rate is the drawn one.
    """
    rng = np.random.default_rng(seed)
    values = ct.draw_values(topo, rng, rho, bias_scale=scale,
                            group_cfgs=ct.group_cfgs_of([knobs]))
    tail = _TAIL.get(topo.label)
    if tail is not None:
        G = values["nodes"][tail]["G"][0]
        values["nodes"][tail]["G"][0] = rng.normal(size=G.shape).astype(G.dtype)
    rows = _DELIVERED.get(topo.label)
    if rows is not None:
        for i in sorted(values["H"]):
            # A mapping that loses a direction: its last row delivers nothing.
            H = np.array(values["H"][i])
            H[rows:] = 0.0
            values["H"][i] = H
            lost = np.linalg.svd(np.asarray(H, np.float64))[2][-1]
            G = values["nodes"][topo.edges[i].src]["G"][0]
            values["nodes"][topo.edges[i].src]["G"][0] = (
                G + np.outer(lost, rng.normal(size=G.shape[1]))).astype(G.dtype)
    return values


def _delivery(topo, values, i) -> np.ndarray:
    e = topo.edges[i]
    assert e.transform is None and e.field == "x"
    return (np.asarray(values["H"][i], np.float64) if e.mapped
            else np.eye(topo.node(e.src).n))


def member_allowances(topo, knobs, values, state) -> dict:
    """``{member: per-entry bound on its defect}`` that ``converged=True`` promises."""
    norm = knobs["convergence_norm"]
    readings = {i: _delivery(topo, values, i) @ np.asarray(state[topo.edges[i].src]["x"],
                                                           np.float64)
                for i in range(len(topo.edges))}
    if norm == "l2":
        tau = RTOL
    elif norm == "mixed":
        tau = RTOL * np.sqrt(sum(nd.n for nd in topo.nodes))
    else:
        tau = RTOL * np.sqrt(sum(r.size for r in readings.values()))
    eps = float(np.finfo(np.float32).eps)
    out = {}
    for nd in topo.nodes:
        x = np.abs(np.asarray(state[nd.name]["x"], np.float64))
        allow = np.full(nd.n, np.max(x))
        for i, e in enumerate(topo.edges):
            if e.dst != nd.name:
                continue
            G = np.abs(np.asarray(values["nodes"][nd.name]["G"][e.port], np.float64))
            D = np.abs(_delivery(topo, values, i))
            # The larger of what the edge delivered and of what its source
            # field is (the state norms measure the field, the interface
            # norm the delivery).
            scale = max(np.max(np.abs(readings[i])),
                        np.max(np.abs(np.asarray(state[e.src]["x"], np.float64))))
            allow = allow + (G @ (D @ np.ones(D.shape[1]))) * scale
        out[nd.name] = SLACK * tau * allow + ROUNDING_ULPS * eps * allow
    return out


def check_converged_state(topo, knobs, built, values) -> bool:
    """Assert the property on one step from the drawn start; ``False`` if the group did not converge."""
    (step,) = ct.run(built, values, 1)
    report = step.reports[0]
    if not report["converged"]:
        return False
    model = ct.LinearModel(topo, values, node_order=built.node_order,
                           group_cfgs=ct.group_cfgs_of([knobs]))
    defects = model.defects(step.pre, step.state)
    allow = member_allowances(topo, knobs, values, step.state)
    what = (f"{topo.label} {knobs['iteration_mode']}/{knobs['convergence_norm']}/"
            f"{knobs['solver']}/{knobs['acceleration']} iterations={report['iterations']} "
            f"residual={report['residual']:.3e}")
    for nd in topo.nodes:
        d = np.abs(np.asarray(defects[nd.name], np.float64))
        assert np.all(d <= allow[nd.name]), (
            f"{what}: member {nd.name!r} is {d} from its update at the state returned, "
            f"{np.max(d / allow[nd.name]):.3g} times what the tolerance allows "
            f"({allow[nd.name]})")
    reference = model.monolithic(step.pre)
    resolvent = np.abs(model.distance_operator())
    bound = resolvent @ np.concatenate([allow[nd.name] for nd in topo.nodes])
    got = np.concatenate([np.asarray(step.state[nd.name]["x"], np.float64)
                          for nd in topo.nodes])
    want = np.concatenate([np.asarray(reference[nd.name], np.float64)
                           for nd in topo.nodes])
    assert np.all(np.abs(got - want) <= bound), (
        f"{what}: the state is {np.max(np.abs(got - want) / bound):.3g} times further from "
        f"the fixed point than the tolerance allows")
    return True


_DRAWS = st.tuples(st.integers(0, 2**32 - 1),
                   st.sampled_from([0.005, 0.02, 0.1, 0.4, 0.9]),
                   st.sampled_from([1.0, 30.0]))


def _hold(name, schedule, norm, draws, solver="ift", extra=()):
    topo, knobs, built = _built(name, schedule, norm, solver, extra)
    return check_converged_state(topo, knobs, built, _values(topo, knobs, *draws))


@pytest.mark.parametrize("name,schedule,norm", _cells(PER_PUSH))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(draws=_DRAWS)
def test_a_converged_group_is_at_its_fixed_point_per_push(name, schedule, norm, draws):
    """Slow sibling: :func:`test_a_converged_group_is_at_its_fixed_point`."""
    _hold(name, schedule, norm, draws)


@pytest.mark.parametrize("name,schedule,norm", _cells(PER_PUSH))
def test_the_cells_converge_so_the_property_is_asked(name, schedule, norm):
    """A check that returns early on an unconverged group proves nothing:
    at a loop gain of 0.3 every per-push cell converges."""
    for seed in range(4):
        assert _hold(name, schedule, norm, (seed, 0.3, 1.0))


def test_the_oracle_refuses_a_member_computed_from_the_pre_step_readings():
    """The oracle can fail: the state the defect was -- the target of a
    one-way pair at the pre-step source -- is ``1 / tau`` beyond it."""
    topo, knobs, built = _built("one-way-mapped", "jacobi", "interface")
    values = _values(topo, knobs, 3, 0.3, 1.0)
    (step,) = ct.run(built, values, 1)
    model = ct.LinearModel(topo, values, node_order=built.node_order,
                           group_cfgs=ct.group_cfgs_of([knobs]))
    v = values["nodes"]["b"]
    stale = (np.asarray(v["b"], np.float64) + np.asarray(v["G"][0], np.float64)
             @ np.asarray(values["H"][0], np.float64) @ np.asarray(step.pre["a"]["x"], np.float64))
    state = {"a": step.state["a"], "b": {"x": stale}}
    defect = np.abs(np.asarray(model.defects(step.pre, state)["b"], np.float64))
    allow = member_allowances(topo, knobs, values, state)["b"]
    assert np.max(defect / allow) > 100.0, (defect, allow)


_ACCELERATED = (
    (("acceleration", "fixed"), ("relaxation", 0.6)),
    (("acceleration", "aitken"),),
    (("acceleration", "iqn-ils"),),
)


# Slow: 84 compiled cells.
# Per push: tests/property/test_converged_groups_are_at_their_fixed_point.py::test_a_converged_group_is_at_its_fixed_point_per_push
@pytest.mark.slow
@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("name,schedule,norm", _cells(sorted(TOPOLOGIES)))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(draws=_DRAWS)
def test_a_converged_group_is_at_its_fixed_point(name, schedule, norm, solver, draws):
    _hold(name, schedule, norm, draws, solver)


# Slow: 42 compiled cells.
# Per push: tests/property/test_converged_groups_are_at_their_fixed_point.py::test_a_converged_group_is_at_its_fixed_point_per_push
@pytest.mark.slow
@pytest.mark.parametrize("extra", _ACCELERATED, ids=lambda e: dict(e)["acceleration"])
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(draws=_DRAWS)
def test_a_converged_accelerated_group_is_at_its_fixed_point_under_the_interface_norm(
        name, schedule, extra, draws):
    _hold(name, schedule, "interface", draws, extra=extra)


# ---------------------------------------------------------------------------
# The return rule: on the exact model, and the library against it
# ---------------------------------------------------------------------------

#: ``(members recomputed, of which the norm reads)`` under the interface norm.
_RECOMPUTED = {
    "one-way": (("b",), ()),
    "one-way-mapped": (("a", "b"), ("a",)),
    "chain": (("c",), ()),
    "cycle-tail": (("c",), ()),
    "tail-first": (("c",), ()),
    "head-cycle": ((), ()),
    "lossy-ring": (("a", "b"), ("a", "b")),
}


def _stacked(model, state) -> np.ndarray:
    return np.concatenate([np.asarray(state[m]["x"], np.float64) for m in model.topo.groups[0]])


def _interface_model(name, schedule, draws, built=None):
    topo = TOPOLOGIES[name]
    knobs = _knobs(schedule, "interface")
    values = _values(topo, knobs, *draws)
    model = ct.LinearModel(topo, values, group_cfgs=ct.group_cfgs_of([knobs]),
                           **({} if built is None else {"node_order": built.node_order}))
    return topo, knobs, values, model


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("name", sorted(TOPOLOGIES))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(draws=_DRAWS, passes=st.integers(0, 6))
def test_on_the_exact_model_the_returned_state_reads_within_the_residual_of_its_iterate(
        name, schedule, draws, passes):
    """No graph.  For **any** iterate ``x`` of the plain loop -- accepted or
    not -- and ``y = x + P (F(x) - x)``:

    * the members named are the ones no plain internal edge reads;
    * the readings of ``y`` are those of ``x`` moved by the residual's own
      terms on the edges whose source was recomputed: within the residual,
      equal to it where every field read was recomputed, unmoved where none
      was;
    * ``y - Phi(y) = -[(I - L) - (I - M) P] (F(x) - x)``.
    """
    topo, _knobs_, values, model = _interface_model(name, schedule, draws)
    recomputed, read = _RECOMPUTED[name]
    assert set(model.recomputed(0)) == set(recomputed)
    pre = {nd.name: {"x": np.asarray(values["nodes"][nd.name]["x0"])} for nd in topo.nodes}
    k = sum(nd.n for nd in topo.nodes)
    L, U = (np.asarray(a, np.float64) for a in model.group_pass(0))
    c = np.asarray(model.group_constant(0, pre, pre), np.float64)

    def one_pass(v):
        return np.linalg.solve(np.eye(k) - L, U @ v + c)

    x = _stacked(model, pre)
    for _ in range(passes):
        x = one_pass(x)
    F = one_pass(x)
    y = model.returned_from(0, pre, pre, x)
    residual = model.residual_between(0, F, x)
    moved = model.readings_moved(0, pre, pre, x)
    slack = 1e-9 * max(residual, 1.0)
    assert moved <= residual + slack, (moved, residual)
    members = topo.groups[0]
    read_by_the_norm = {topo.edges[i].src for i in topo.internal_edges(0)}
    if read_by_the_norm <= set(recomputed):
        assert abs(moved - residual) <= slack, (moved, residual)
    if not read:
        assert moved == 0.0
    P = model._recomputed_projector(0)      # noqa: SLF001
    defect = y - ((L + U) @ y + c)
    want = -((np.eye(k) - L) - (np.eye(k) - L - U) @ P) @ (F - x)
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(F))), 1.0)
    assert np.max(np.abs(defect - want)) <= 1e-9 * scale, (members, defect, want)


@functools.lru_cache(maxsize=None)
def _built_at_the_accepted_iterate(name: str, schedule: str):
    """The interface cell built and first stepped with the return rule
    switched off (:func:`coupled_graphs.accepted_iterate`)."""
    topo, knobs = TOPOLOGIES[name], _knobs(schedule, "interface")
    with cg.accepted_iterate() as asked:
        built = ct.build(topo, knobs)
        ct.run(built, _values(topo, knobs, 0, 0.3, 1.0), 1)
    assert asked, "the step never asked the return rule: the patch is on the wrong name"
    return built


def check_the_library_returns_the_models_state(name, schedule, draws) -> None:
    topo, knobs, built = _built(name, schedule, "interface")
    values = _values(topo, knobs, *draws)
    model = ct.LinearModel(topo, values, node_order=built.node_order,
                           group_cfgs=ct.group_cfgs_of([knobs]))
    (step,) = ct.run(built, values, 1)
    (loop,) = ct.run(_built_at_the_accepted_iterate(name, schedule), values, 1)
    report, where = step.reports[0], f"{name} {schedule} {draws}"
    # The rule moves no iterate, residual or pass count.
    for key in ("iterations", "converged", "residual"):
        assert report[key] == loop.reports[0][key], (where, key)
    # The report is the accepted iterate's: the plain checks hold of it, the
    # equality of the reported residual included.
    model.check_step(loop.pre, loop.state, loop.reports, thresholds=[1.0], where=where,
                     accepted=True)
    # ... and the state returned is the model's for that iterate: a member
    # an edge delivers whole to the bit, the others one pass on.
    x = _stacked(model, loop.state)
    want = model.returned_from(0, step.pre, step.pre, x)
    got = _stacked(model, step.state)
    eps, at = float(np.finfo(np.float32).eps), 0
    for m in topo.groups[0]:
        n = topo.node(m).n
        if m not in model.recomputed(0):
            assert np.array_equal(step.state[m]["x"], loop.state[m]["x"]), (where, m)
        terms = np.max(np.abs(x)) + np.max(np.abs(want))
        assert np.all(np.abs(got[at:at + n] - want[at:at + n]) <= ROUNDING_ULPS * eps * terms), (
            where, m, got[at:at + n], want[at:at + n])
        at += n
    # The restated checks hold of the state returned.
    model.check_step(step.pre, step.state, step.reports, thresholds=[1.0], where=where)
    moved = model.readings_moved(0, step.pre, step.pre, x)
    assert moved <= float(report["residual"]) * (1.0 + 2e-3) + 16 * eps / RTOL, (
        f"{where}: the returned state's readings are {moved:.4g} from the accepted "
        f"iterate's, beside a reported residual of {report['residual']!r}")


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("name", PER_PUSH)
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(draws=_DRAWS)
def test_the_library_returns_the_models_state_for_the_iterate_it_accepted_per_push(
        name, schedule, draws):
    """Slow sibling: :func:`test_the_library_returns_the_models_state_for_the_iterate_it_accepted`."""
    check_the_library_returns_the_models_state(name, schedule, draws)


# Slow: 8 more compiled cells, each twice.
# Per push: tests/property/test_converged_groups_are_at_their_fixed_point.py::test_the_library_returns_the_models_state_for_the_iterate_it_accepted_per_push
@pytest.mark.slow
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("name", sorted(set(TOPOLOGIES) - set(PER_PUSH)))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None)
@given(draws=_DRAWS)
def test_the_library_returns_the_models_state_for_the_iterate_it_accepted(
        name, schedule, draws):
    check_the_library_returns_the_models_state(name, schedule, draws)
