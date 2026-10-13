"""What a coupling group returns under ``convergence_norm="interface"``, and what its report is of.

The interface norm measures, on each pass, how far what the group's
internal edges deliver moved between an iterate and its successor.  The
members of the iterate it accepts were computed from the readings of the
one *before* it, which the exit compares with nothing.  So every floating
field the norm does not **measure whole** -- the source field of an
internal edge that carries no mapping and no transform -- could be
returned from a pass the settled readings never produced, with
``converged=True``:

* a field **no internal edge reads**: a one-way pair ``A -> B`` under
  Jacobi returned ``B`` computed from the *pre-step* ``A`` at
  ``iterations=1`` and a residual of exactly zero (MADD-ANO-240; 0.1.0 to
  0.3.1);
* a field edges read **only through a mapping or a transform**: the part
  of it they do not deliver was measured by nothing (MADD-ANO-241; 0.1.0
  to 0.3.1).

**The return rule.**  With ``x`` the iterate the loop accepts (or stops on
at its cap) and ``F`` one plain pass of the group's schedule: a field
measured whole is returned as ``x`` holds it, bit for bit; every other
floating field as ``F(x)`` holds it.  One rule for every member, schedule,
acceleration, solver and verdict.  Its consequences, each tested here:

* the report (``iterations``, ``residual``, ``converged``) is of ``x`` and
  is the one the loop measured: no iterate, residual or pass count moves;
* the returned state is not an iterate of the loop.  Its readings differ
  from those of ``x`` only on the edges whose source field was recomputed,
  and there by exactly the term the residual holds for that edge: **the
  readings of the returned state are within the reported residual of those
  of ``x``, in the residual's own weights**;
* ``max_iterations=1`` returns its one pass as it is (one pass has to cost
  one pass).

Every oracle is a float64 closed form of the linear map the graph holds:
the fixed point, and one pass at a given iterate.  The accepted iterate
itself is read from the same library with the rule switched off (``_rule``
patches the name the step reads, and shows the patch was read).
"""

from __future__ import annotations

import contextlib
import itertools
import os
import warnings
from unittest import mock

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _coupled_block, _interface_plan
from maddening.core.coupling._group_layout import _fields_the_interface_norm_misses
from maddening.core.coupling.acceleration import float_fields_of
from maddening.core.coupling.group import CouplingGroup
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core import coupling_domains as cd

RTOL = 1e-4


class _Lin(SimulationNode):
    """``u <- alpha u_pre + c + sum_p G_p @ inp_p``; with *wide*, also
    ``w <- 1 + sum_p K_p @ inp_p``, a field no edge reads.  *calls* counts
    the evaluations of ``update`` the compiled step runs."""

    def __init__(self, name, n, ports, alpha, c, u0, dtype, wide, timestep=1.0, calls=None):
        super().__init__(name, timestep)
        self._n, self._dtype, self._wide = n, dtype, wide
        self._ports = {p: (np.asarray(G, dtype), np.asarray(K, dtype))
                       for p, (G, K) in ports.items()}
        self._alpha = dtype(alpha)
        self._c = np.asarray(c, dtype)
        self._u0 = np.asarray(u0, dtype)
        self._calls = calls

    def initial_state(self):
        s = {"u": jnp.asarray(self._u0)}
        if self._wide:
            s["w"] = jnp.zeros(self._n, self._dtype)
        return s

    def boundary_input_spec(self):
        return {p: BoundaryInputSpec(shape=(G.shape[1],), dtype=self._dtype,
                                     default=jnp.zeros(G.shape[1], self._dtype))
                for p, (G, _K) in self._ports.items()}

    def update(self, state, boundary_inputs, dt):
        if self._calls is not None:
            jax.debug.callback(lambda: self._calls.append(self.name))
        u = self._alpha * state["u"] + jnp.asarray(self._c)
        w = jnp.ones(self._n, self._dtype)
        for p, (G, K) in self._ports.items():
            u = u + jnp.asarray(G) @ boundary_inputs[p]
            w = w + jnp.asarray(K) @ boundary_inputs[p]
        return {"u": u, "w": w} if self._wide else {"u": u}

    def update_evaluations(self):
        return 1.0


def _first(v):
    """A transform that delivers the first entry only."""
    return v[:1]


#: ``(name, n, alpha, c, u0)``; every start is far from where the step ends.
A = ("A", 2, 0.5, [1.0, -2.0], [4.0, 4.0])
B = ("B", 2, 0.0, [0.25, 0.5], [-3.0, 5.0])
C = ("C", 2, 0.0, [-1.0, 0.75], [2.0, -6.0])
#: ``B`` and ``C`` with a memory of their own pre-step value: what a step
#: returns for them is what the next one starts from.
B_KEEPS = ("B", 2, 0.5, [0.25, 0.5], [-3.0, 5.0])
C_KEEPS = ("C", 2, 0.25, [-1.0, 0.75], [2.0, -6.0])
G = np.array([[0.7, 0.3], [-0.4, 0.9]])
#: The gain of the unread field ``w``: order one, so a reading one pass old
#: shows in it at the size of that reading's change.
K = np.array([[2.0, 1.0], [1.0, -2.0]])
H = np.array([[1.0, 0.5], [-0.25, 1.5]])
#: A mapping that delivers the first entry of a field of two.
SELECT = np.array([[1.0, 0.0]])


def _graph(nodes, edges, group, *, dtype=np.float32, wide=(), timesteps=None, calls=None,
           diagnostics=None):
    """*edges*: ``(src, dst, G, how)`` with *how* ``None`` (plain), a matrix
    (the edge's mapping) or a callable (its transform).  *diagnostics*:
    ``None`` asks for them only where the solver reports nothing without."""
    ports = {name: {} for name, *_ in nodes}
    for i, (_src, dst, gain, _how) in enumerate(edges):
        gain = np.asarray(gain, np.float64)
        ports[dst][f"p{i}"] = (gain, K[:, :gain.shape[1]])
    gm = GraphManager()
    for name, n, alpha, c, u0 in nodes:
        gm.add_node(_Lin(name, n, ports[name], alpha, c, u0, dtype, name in wide,
                         timestep=(timesteps or {}).get(name, 1.0), calls=calls))
    for i, (src, dst, _gain, how) in enumerate(edges):
        extra = {}
        if callable(how):
            extra["transform"] = how
        elif how is not None:
            extra["mapping"] = matrix_mapping(np.asarray(how, dtype))
        gm.add_edge(src, dst, "u", f"p{i}", **extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         # the deprecated "fori" says so
        # ``"fori"`` reports only with its diagnostics on.
        gm.add_coupling_group(
            [name for name, *_ in nodes],
            diagnostics=group.get("solver") == "fori" if diagnostics is None else diagnostics,
            **group)
        gm.compile()
    return gm


def _delivers(how, n):
    """The matrix an edge applies to its source field of *n* entries."""
    if how is None:
        return np.eye(n)
    if callable(how):
        return np.eye(n)[:1]            # ``_first``
    return np.asarray(how, np.float64)


def _sizes(nodes):
    return {name: n for name, n, *_ in nodes}


def _fixed_point(nodes, edges, wide=(), pre=None):
    """``{node: {field: value}}`` with every edge reading the new value, in
    float64: the step's fixed point from *pre* (the nodes' own starts)."""
    off, k = {}, 0
    for name, n, *_ in nodes:
        off[name] = k
        k += n
    size = _sizes(nodes)
    M, b = np.zeros((k, k)), np.zeros(k)
    for name, n, alpha, c, u0 in nodes:
        start = u0 if pre is None else pre[name]["u"]
        b[off[name]:off[name] + n] = alpha * np.asarray(start, float) + np.asarray(c, float)
    for src, dst, gain, how in edges:
        M[off[dst]:off[dst] + size[dst], off[src]:off[src] + size[src]] += (
            np.asarray(gain, float) @ _delivers(how, size[src]))
    x = np.linalg.solve(np.eye(k) - M, b)
    out = {name: {"u": x[off[name]:off[name] + size[name]]} for name in size}
    for name in wide:
        w = np.ones(size[name])
        for src, dst, gain, how in edges:
            if dst == name:
                w = w + K[:, :np.asarray(gain).shape[1]] @ _delivers(how, size[src]) @ out[src]["u"]
        out[name]["w"] = w
    return out


def _plain_pass(nodes, edges, x, schedule, wide=(), order=None):
    """``F(x)`` in float64: one plain pass of *schedule* at the iterate *x*,
    from the nodes' own starts.  Under Gauss-Seidel a member reads this
    pass's value of every member before it in the sweep (*order*, the
    graph's own) and *x* for the others; under Jacobi it reads *x*
    throughout."""
    size, new = _sizes(nodes), {}
    by_name = {node[0]: node for node in nodes}
    for name, n, alpha, c, u0 in (by_name[m] for m in order or by_name):
        u = alpha * np.asarray(u0, float) + np.asarray(c, float)
        w = np.ones(n)
        for src, dst, gain, how in edges:
            if dst != name:
                continue
            read = new[src] if schedule == "gauss-seidel" and src in new else x[src]
            delivered = _delivers(how, size[src]) @ np.asarray(read["u"], float)
            u = u + np.asarray(gain, float) @ delivered
            w = w + K[:, :np.asarray(gain).shape[1]] @ delivered
        new[name] = {"u": u, "w": w} if name in wide else {"u": u}
    return new


def _residual_terms(nodes, edges, new, old, rtol):
    """``{edge index: (sum of squares, entries)}`` of the interface norm of
    *new* against *old*: what each internal edge delivers, each entry over
    ``rtol`` times the delivered value's largest magnitude at either; an
    edge that delivers nothing but zeros is left out (the dead band at
    ``atol=0``).  The norm is the root of the summed squares over the
    summed entries."""
    size, terms = _sizes(nodes), {}
    for i, (src, _dst, _gain, how) in enumerate(edges):
        D = _delivers(how, size[src])
        a, b = D @ np.asarray(new[src]["u"], float), D @ np.asarray(old[src]["u"], float)
        ref = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
        if ref > 0:
            terms[i] = (float(np.sum(((a - b) / (rtol * ref)) ** 2)), a.size)
    return terms


def _whole(edges):
    """The fields the norm measures whole: delivered as they are by an edge."""
    return {(src, "u") for src, _dst, _gain, how in edges if how is None}


@contextlib.contextmanager
def _rule(which):
    """The step's return rule replaced: ``"none"`` recomputes nothing (the
    solve returns the iterate it accepted), ``"all"`` every floating field
    (one plain pass at that iterate); ``None`` is the library's own.  Yields
    the list of groups the step asked the rule about: the patch is on the
    name ``_coupled_block`` reads, and a caller asserts it was read."""
    real, seen = _coupled_block._fields_the_interface_norm_misses, []

    def rule(group, interface_edges, schedule, state):
        seen.append(tuple(schedule))
        if which is None or group.convergence_norm != "interface":
            return real(group, interface_edges, schedule, state)
        if which == "none":
            return {}
        floats = float_fields_of(state, list(schedule))
        return {nn: fields for nn, fields in floats.items() if fields}

    with mock.patch.object(_coupled_block, "_fields_the_interface_norm_misses", rule):
        yield seen


def _solve(nodes, edges, group, *, rule=None, x64=False, wide=(), steps=1, timesteps=None,
           calls=None):
    """``(state, report)`` after *steps* steps of a fresh graph: every field
    as the array the graph holds; ``report["sweep"]`` is the members in the
    order a pass runs them."""
    with cd.x64(x64), _rule(rule) as seen:
        gm = _graph(nodes, edges, group, dtype=np.float64 if x64 else np.float32,
                    wide=wide, timesteps=timesteps, calls=calls)
        for _ in range(steps):
            gm.step()
        if calls is not None:
            jax.effects_barrier()
        (report,) = gm.coupling_diagnostics().values()
        report = dict(report, sweep=[m for m in gm.schedule if m in _sizes(nodes)])
        state = {name: {f: np.asarray(v) for f, v in gm.get_node_state(name).items()}
                 for name, *_ in nodes}
    assert seen, "the step never asked the return rule: the patch is on the wrong name"
    return state, report


def _distance(state, want):
    """The worst field's ``max |x - x*| / max |x*|``, and which field it is."""
    worst, where = 0.0, None
    for name, fields in want.items():
        for f, v in fields.items():
            d = float(np.max(np.abs(np.asarray(state[name][f], np.float64) - v))
                      / np.max(np.abs(v)))
            if d > worst:
                worst, where = d, f"{name}.{f}"
    return worst, where


def _step(nodes, edges, group, *, x64=False, wide=()):
    """``(distance to the fixed point, where, report)`` of one step from the start."""
    state, report = _solve(nodes, edges, group, x64=x64, wide=wide)
    distance, where = _distance(state, _fixed_point(nodes, edges, wide))
    return distance, where, report


def _eps(x64):
    return float(np.finfo(np.float64 if x64 else np.float32).eps)


# ---------------------------------------------------------------------------
# The rule: what is returned, and what the report is of
# ---------------------------------------------------------------------------

#: Each member's first entry reads the other's at a loop gain of 0.3; its
#: second entry, which a selection does not deliver, reads the other's
#: first at order one.
G_A = np.array([[0.5], [1.0]])
G_B = np.array([[0.6], [-0.7]])

#: ``name: (nodes, edges, members with the unread field w)``.  Loop gains
#: of about 0.3, so a cap of three passes is reached well outside ``RTOL``.
CASES = {
    # ``u`` delivered whole both ways; ``w`` read by no edge.
    "unread": ([A, B], [("A", "B", G, None), ("B", "A", 0.3 * G.T, None)], ("A", "B")),
    # ``u`` read only through a mapping that loses nothing.
    "mapped": ([A, B], [("A", "B", G, H), ("B", "A", 0.3 * G.T, H)], ()),
    # ... through a mapping that delivers one entry of two.
    "selected": ([A, B], [("A", "B", G_B, SELECT), ("B", "A", G_A, SELECT)], ("B",)),
    # ... through a transform that does.
    "transformed": ([A, B], [("A", "B", G_B, _first), ("B", "A", G_A, _first)], ()),
    # ``A.u`` and ``C.u`` each read whole by ``B`` and through a selection
    # by the other, so measured whole; ``B.u`` and ``C.w`` read by nothing.
    "tail": ([A, B, C], [("A", "B", G, None), ("A", "C", G_B, SELECT),
                            ("C", "B", 0.5 * G, None), ("C", "A", G_A, SELECT)], ("C",)),
}
ACCELERATIONS = {
    "none": {},
    "fixed": dict(acceleration="fixed", relaxation=0.5),
    "aitken": dict(acceleration="aitken"),
    "iqn-ils": dict(acceleration="iqn-ils"),
    "iqn-imvj": dict(acceleration="iqn-imvj", jacobian_reuse=2),
}
#: Passes enough for every row to converge (Jacobi relaxed at 0.5 on the
#: mapped pair takes more than forty).
CONVERGES = 120
#: ``(case, schedule, solver, acceleration, cap)``: ``cap`` ``None`` lets
#: the group converge; 3 stops the plain and the relaxed loops well short,
#: and every per-push row with a cap (an accelerated Gauss-Seidel loop of
#: these linear pairs can converge inside three passes, and is then one
#: more converged row).
GRID = tuple(itertools.product(tuple(CASES), ("jacobi", "gauss-seidel"), ("ift", "fori"),
                               ACCELERATIONS, (None, 3)))
#: The rows run on every push: every case, schedule, solver, acceleration
#: and verdict at least twice, and every case under both solvers.
PER_PUSH = (
    ("unread", "jacobi", "ift", "none", None),
    ("unread", "gauss-seidel", "fori", "iqn-ils", 3),
    ("unread", "gauss-seidel", "ift", "fixed", None),
    ("mapped", "gauss-seidel", "ift", "none", None),
    ("mapped", "jacobi", "fori", "aitken", 3),
    ("mapped", "jacobi", "ift", "iqn-imvj", None),
    ("selected", "jacobi", "ift", "fixed", 3),
    ("selected", "gauss-seidel", "fori", "none", None),
    ("selected", "gauss-seidel", "ift", "iqn-ils", None),
    ("transformed", "jacobi", "fori", "none", 3),
    ("transformed", "gauss-seidel", "ift", "aitken", None),
    ("tail", "jacobi", "ift", "none", None),
    ("tail", "jacobi", "fori", "iqn-imvj", 3),
    ("tail", "gauss-seidel", "ift", "fixed", None),
)
assert set(PER_PUSH) <= set(GRID)


def _row_id(row):
    return "-".join(str(part) for part in row)


def assert_the_return_rule(case, schedule, solver, acceleration, cap, *, x64=False):
    """The returned state is the accepted iterate with its unmeasured fields
    from one plain pass, and the report is of the accepted iterate."""
    nodes, edges, wide = CASES[case]
    group = dict(iteration_mode=schedule, convergence_norm="interface", rtol=RTOL,
                 solver=solver, **ACCELERATIONS[acceleration])
    group["max_iterations"] = CONVERGES if cap is None else cap
    returned, report = _solve(nodes, edges, group, wide=wide, x64=x64)
    accepted, loop = _solve(nodes, edges, group, wide=wide, rule="none", x64=x64)
    where = _row_id((case, schedule, solver, acceleration, cap))

    # 1. The rule moves no iterate, residual or pass count.
    assert (report["iterations"], report["converged"]) == (
        loop["iterations"], loop["converged"]), where
    assert float(report["residual"]) == float(loop["residual"]), where
    if cap is None or (case, schedule, solver, acceleration, cap) in PER_PUSH:
        assert report["converged"] is (cap is None), (where, report["iterations"])
    if not report["converged"]:
        assert report["iterations"] == cap, where

    # 2. A field measured whole is the accepted iterate's, to the bit; every
    #    other is one plain pass at the accepted iterate.
    after = _plain_pass(nodes, edges, accepted, schedule, wide, report["sweep"])
    whole, eps, moved = _whole(edges), _eps(x64), 0.0
    for name, fields in returned.items():
        for f, value in fields.items():
            if (name, f) in whole:
                assert np.array_equal(value, accepted[name][f]), (where, name, f)
                continue
            scale = float(np.max(np.abs(after[name][f])))
            gap = float(np.max(np.abs(value.astype(np.float64) - after[name][f])))
            assert gap <= 32 * eps * scale, (
                f"{where}: {name}.{f} is {gap / scale:.2e} (relative) from one plain pass "
                f"at the accepted iterate")
            moved = max(moved, float(np.max(np.abs(
                value.astype(np.float64) - accepted[name][f]))) / scale)
    if acceleration == "none" or not report["converged"]:
        # (An accelerated solve of a linear group can stop on the fixed
        # point to rounding, where the pass changes nothing.)
        assert moved > 64 * eps, f"{where}: the rule changed nothing; the row tests nothing"

    # 3. The report is of the accepted iterate: its residual is the norm of
    #    the pass at it.  A float32 residual carries the rounding of its
    #    readings over ``rtol``.
    terms = _residual_terms(nodes, edges, after, accepted, RTOL)
    count = sum(n for _sq, n in terms.values())
    restated = float(np.sqrt(sum(sq for sq, _n in terms.values()) / max(count, 1)))
    slack = 2e-3 * restated + 16 * eps / RTOL
    assert abs(float(report["residual"]) - restated) <= slack, (where, report, restated)

    # 4. The readings of the returned state are within the reported residual
    #    of the accepted iterate's, in the residual's own weights: they
    #    moved on the edges whose source was recomputed, by that edge's own
    #    term of the residual, and nowhere else.
    drift = _residual_terms(nodes, edges, returned, accepted, RTOL)
    recomputed = {i for i, (src, *_rest) in enumerate(edges) if (src, "u") not in whole}
    assert set(drift) <= set(terms), where
    for i, (sq, _n) in drift.items():
        if i in recomputed:
            assert abs(np.sqrt(sq) - np.sqrt(terms[i][0])) <= slack * np.sqrt(count), (where, i)
        else:
            assert sq == 0.0, (where, i)
    readings_moved = float(np.sqrt(sum(sq for sq, _n in drift.values()) / max(count, 1)))
    assert readings_moved <= float(report["residual"]) + slack, (
        f"{where}: the returned state's readings are {readings_moved:.4g} from the accepted "
        f"iterate's; the report says residual={report['residual']!r}")
    if recomputed == set(range(len(edges))):
        assert abs(readings_moved - float(report["residual"])) <= slack, where


@pytest.mark.parametrize("row", PER_PUSH, ids=_row_id)
def test_the_solve_returns_the_accepted_iterate_with_its_unmeasured_fields_from_one_pass(row):
    assert_the_return_rule(*row)


@pytest.mark.parametrize("row", [PER_PUSH[3], PER_PUSH[6]], ids=_row_id)
def test_the_return_rule_holds_in_float64(row):
    assert_the_return_rule(*row, x64=True)


# Per push: tests/core/test_coupling_interface_norm_answers_for_the_state_it_returns.py::test_the_solve_returns_the_accepted_iterate_with_its_unmeasured_fields_from_one_pass
@pytest.mark.slow
@pytest.mark.parametrize("row", [r for r in GRID if r not in PER_PUSH], ids=_row_id)
def test_the_return_rule_holds_under_every_schedule_solver_acceleration_and_verdict(row):
    assert_the_return_rule(*row)


@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
def test_a_sub_cycled_member_is_returned_by_the_same_rule(schedule):
    """``B`` at half the group's timestep, sub-stepped twice per pass between
    interpolated readings.  No closed form here: the pass is the library's
    own, read by recomputing every field (``"all"``)."""
    nodes, edges, wide = CASES["selected"]
    group = dict(iteration_mode=schedule, convergence_norm="interface", rtol=RTOL,
                 subcycling=True, boundary_interpolation="linear", max_iterations=CONVERGES)
    kw = dict(wide=wide, timesteps={"A": 1.0, "B": 0.5})
    returned, report = _solve(nodes, edges, group, **kw)
    accepted, loop = _solve(nodes, edges, group, rule="none", **kw)
    after, _ = _solve(nodes, edges, group, rule="all", **kw)
    assert report["converged"] is True
    assert (report["iterations"], float(report["residual"])) == (
        loop["iterations"], float(loop["residual"]))
    differs = False
    for name, fields in returned.items():
        for f, value in fields.items():
            assert np.array_equal(value, after[name][f]), (name, f)     # nothing is whole here
            differs = differs or not np.array_equal(value, accepted[name][f])
    assert differs, "the accepted iterate is already the pass at it: the case tests nothing"


def test_a_cap_of_one_returns_its_one_pass_as_it_is():
    """``max_iterations=1`` is a request for one staggered pass and costs
    one: nothing is recomputed, and the residual is how far that pass moved."""
    nodes, edges, wide = CASES["selected"]
    group = dict(iteration_mode="jacobi", convergence_norm="interface", rtol=RTOL,
                 max_iterations=1)
    calls = []
    returned, report = _solve(nodes, edges, group, wide=wide, calls=calls)
    accepted, _ = _solve(nodes, edges, group, wide=wide, rule="none")
    assert report["iterations"] == 1 and len(calls) == len(nodes)
    for name, fields in returned.items():
        for f, value in fields.items():
            assert np.array_equal(value, accepted[name][f]), (name, f)


#: A pair whose every floating field an edge delivers whole.
CASES["whole"] = ([A, B], [("A", "B", G, None), ("B", "A", 0.3 * G.T, None)], ())


@pytest.mark.parametrize("case,solver,extra", [
    ("whole", "ift", 0), ("whole", "fori", 0), ("unread", "ift", 2), ("selected", "fori", 2),
    ("tail", "ift", 3)])
def test_the_rule_costs_one_evaluation_of_the_pass_and_only_where_it_recomputes(
        case, solver, extra):
    """Counted as calls of the members' ``update`` in the compiled step: one
    more of each member where a field is recomputed, none in a group whose
    every floating field is measured whole."""
    nodes, edges, wide = CASES[case]
    group = dict(iteration_mode="gauss-seidel", convergence_norm="interface", rtol=RTOL,
                 solver=solver, max_iterations=CONVERGES)
    counts = {}
    for rule in (None, "none"):
        calls = []
        _state, report = _solve(nodes, edges, group, wide=wide, rule=rule, calls=calls)
        counts[rule] = len(calls)
    assert report["converged"] is True
    assert counts[None] - counts["none"] == extra, counts


@pytest.mark.parametrize("norm", ["interface", "mixed", "l2"])
def test_an_identity_mapped_group_equals_its_unmapped_twin_within_the_residual(norm):
    """A mapping that delivers its source unchanged cannot be told from one
    that does not, so under the interface norm its source field is not
    measured whole and is returned one plain pass on.  The unmapped twin's
    every field is measured whole: it returns the iterate both loops
    accept.  So the reports are equal to the bit, and the states within the
    reported residual on what the edges deliver -- to the bit under the
    norms that measure the state.

    (The compact-side rule reads a mapping between fields of one size as
    delivered, like a gather: this holds under it as it is.  A mapping onto
    more entries is read at its source, and that field is measured whole
    and kept: ``test_the_interface_norm_reads_a_mapped_edge_on_its_compact_side.py``.)
    """
    plain = [("A", "B", G, None), ("B", "A", 0.3 * G.T, None)]
    mapped = [(src, dst, gain, np.eye(2)) for src, dst, gain, _how in plain]
    group = dict(iteration_mode="jacobi", convergence_norm=norm, max_iterations=CONVERGES,
                 **({"tolerance": RTOL} if norm == "l2" else {"rtol": RTOL}))
    twin, report = _solve([A, B], plain, group)
    state, through = _solve([A, B], mapped, group)
    assert report["converged"] is True
    assert (through["iterations"], through["converged"], float(through["residual"])) == (
        report["iterations"], report["converged"], float(report["residual"]))
    same = all(np.array_equal(state[n]["u"], twin[n]["u"]) for n in ("A", "B"))
    if norm != "interface":
        assert same
        return
    assert not same, "the mapped field was returned as the accepted iterate holds it"
    after = _plain_pass([A, B], plain, twin, "jacobi")
    for n in ("A", "B"):
        np.testing.assert_allclose(state[n]["u"], after[n]["u"], rtol=32 * _eps(False))
    terms = _residual_terms([A, B], plain, state, twin, RTOL)
    moved = float(np.sqrt(sum(sq for sq, _n in terms.values())
                          / sum(n for _sq, n in terms.values())))
    assert moved <= float(report["residual"]) + 2e-3 + 16 * _eps(False) / RTOL, (moved, report)


class _Root(_Lin):
    """``w <- sqrt(inp)``: not finite once an entry it reads is negative."""

    def update(self, state, boundary_inputs, dt):
        out = super().update(state, boundary_inputs, dt)
        (inp,) = boundary_inputs.values()
        return {**out, "w": jnp.sqrt(inp)}


def _root_pair(solver, **group):
    """``A -> B`` under Jacobi: the pre-step ``A`` is ``[4, 4]`` and the step
    takes it to ``[3, -1]``.  The loop accepts its first iterate, whose
    ``B.w`` is the root of the pre-step ``A`` -- finite -- on readings that
    do not move; the pass at that iterate takes the root of ``-1``."""
    gm = GraphManager()
    gm.add_node(_Lin("A", 2, {}, 0.5, [1.0, -3.0], [4.0, 4.0], np.float32, False))
    gm.add_node(_Root("B", 2, {"p0": (G, K)}, 0.0, [0.25, 0.5], [-3.0, 5.0], np.float32, True))
    gm.add_edge("A", "B", "u", "p0")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gm.add_coupling_group(["A", "B"], iteration_mode="jacobi", convergence_norm="interface",
                              rtol=RTOL, solver=solver, diagnostics=solver == "fori", **group)
        gm.compile()
    return gm


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_recomputed_field_that_is_not_finite_is_in_the_verdict(solver):
    """The verdict on a non-finite state covers every floating field of the
    state returned: a recomputed one is not the accepted iterate's, and the
    residual never saw it."""
    gm = _root_pair(solver)
    gm.step()
    (report,) = gm.coupling_diagnostics().values()
    w = np.asarray(gm.get_node_state("B")["w"])
    assert np.isfinite(w[0]) and np.isnan(w[1]), w
    assert np.all(np.isfinite(np.asarray(gm.get_node_state("A")["u"])))
    assert report["converged"] is False and report["residual"] == float("inf"), report


def test_strict_convergence_names_a_recomputed_field_that_is_not_finite():
    gm = _root_pair("ift", strict_convergence=True)
    with pytest.raises(Exception, match="state is non-finite"):
        gm.step()
        jax.block_until_ready(gm.get_node_state("B")["w"])


# ---------------------------------------------------------------------------
# A field no internal edge reads
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("x64", [False, True], ids=["f32", "f64"])
@pytest.mark.parametrize("how", [None, H], ids=["plain", "static-mapped"])
def test_a_one_way_group_under_jacobi_returns_its_target_at_the_source_it_returns(
        how, x64, solver):
    """The report: ``B`` was computed from the pre-step ``A`` and called converged.

    ``A`` reads nothing, so both members are exact after the pass that
    reads the new ``A``: the state is on the closed form to rounding.
    """
    edges = [("A", "B", G, how)]
    distance, where, report = _step(
        [A, B], edges, dict(iteration_mode="jacobi", convergence_norm="interface",
                            rtol=RTOL, solver=solver), x64=x64)
    assert report["converged"] is True
    assert distance <= 16 * _eps(x64), (
        f"{where} is {distance:.3e} (relative) from the step's fixed point, reported "
        f"converged at iterations={report['iterations']} residual={report['residual']}")


@pytest.mark.parametrize("acceleration", [a for a in ACCELERATIONS if a != "none"] + ["over"])
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_no_acceleration_leaves_a_one_way_target_between_passes(solver, acceleration):
    """A relaxation blends every pass so far into a field nothing measures:
    at 0.5 the target was half-way from the stale value to the right one,
    6800 tolerances off."""
    group = (dict(acceleration="fixed", relaxation=1.3) if acceleration == "over"
             else ACCELERATIONS[acceleration])
    distance, where, report = _step(
        [A, B], [("A", "B", G, None)],
        dict(iteration_mode="jacobi", convergence_norm="interface", rtol=RTOL,
             solver=solver, **group))
    assert report["converged"] is True
    assert distance <= 16 * _eps(False), (where, distance, report["iterations"])


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_chain_under_jacobi_returns_its_last_member_at_the_members_before_it(solver):
    """``A -> B -> C``: ``B`` is delivered whole, so the norm waits for it;
    ``C`` is read by nothing and was returned one pass behind ``B``."""
    edges = [("A", "B", G, None), ("B", "C", G.T, None)]
    distance, where, report = _step(
        [A, B, C], edges, dict(iteration_mode="jacobi", convergence_norm="interface",
                               rtol=RTOL, solver=solver))
    assert report["converged"] is True
    assert distance <= 16 * _eps(False), (where, distance, report["iterations"])


#: A pair coupled at 1e-3: each pass gains three digits, so the pass before
#: the one that meets ``RTOL`` is ten tolerances away -- and that is the pass
#: an unmeasured field of the returned iterate was computed from.
WEAK = 1e-3


@pytest.mark.parametrize("norm", ["interface", "mixed"])
@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
def test_a_field_no_edge_reads_is_within_the_tolerance_in_a_weakly_coupled_pair(
        schedule, norm):
    """Under Gauss-Seidel it is ``A``, which reads the back edge, that lags.

    The unread field is ``K`` times the reading, so a reading within the
    tolerance leaves it within ``|K|`` tolerances; a reading one pass old
    left it ten times that.  ``"mixed"`` measures the field and always held.
    """
    edges = [("A", "B", G, None), ("B", "A", WEAK * G.T, None)]
    distance, where, report = _step(
        [A, B], edges, dict(iteration_mode=schedule, convergence_norm=norm, rtol=RTOL),
        wide=("A", "B"))
    assert report["converged"] is True
    assert distance <= RTOL, (
        f"{where} is {distance / RTOL:.1f} tolerances from the fixed point "
        f"(iterations={report['iterations']})")


# ---------------------------------------------------------------------------
# A field edges read only through a mapping or a transform
# ---------------------------------------------------------------------------

#: ``B``'s first entry does not depend on what ``B`` reads; its second does.
G_SECOND = np.array([[0.0, 0.0], [-0.4, 0.9]])
#: The chain whose middle member's second entry no edge delivers.
LOSSY_CHAIN = {how_id: [("A", "B", G_SECOND, None), ("B", "C", G[:, :1], how)]
               for how_id, how in (("mapping", SELECT), ("transform", _first))}


@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("how", sorted(LOSSY_CHAIN))
def test_the_part_of_a_field_its_edge_does_not_deliver_is_not_left_a_pass_behind(how, solver):
    """``A -> B -> C`` with ``B`` delivered through a selection of its first
    entry, which no input moves: what ``B`` delivers is settled from the
    first pass, a residual of exactly zero, while its second entry still
    held the pre-step ``A``.  The same stale member, behind a mapping."""
    distance, where, report = _step(
        [A, B, C], LOSSY_CHAIN[how],
        dict(iteration_mode="jacobi", convergence_norm="interface", rtol=RTOL, solver=solver))
    assert report["converged"] is True
    assert distance <= 16 * _eps(False), (where, distance, report["iterations"])


#: Each member's first entry reads the other's at 0.03 (the loop the norm
#: watches gains a digit and a half per pass); its second entry, which no
#: edge delivers, reads the other's first at order one.
G_A_WEAK = np.array([[0.03], [1.0]])
G_B_WEAK = np.array([[0.03], [-0.7]])


@pytest.mark.parametrize("relaxation", [0.5, 1.3])
def test_a_relaxation_does_not_leave_the_undelivered_part_of_a_field_between_passes(relaxation):
    """The same chain under ``acceleration="fixed"``: ``B``'s second entry
    was a blend of the passes so far, and would have been after a second
    pass within the threshold too."""
    distance, where, report = _step(
        [A, B, C], LOSSY_CHAIN["mapping"],
        dict(iteration_mode="jacobi", convergence_norm="interface", rtol=RTOL,
             acceleration="fixed", relaxation=relaxation))
    assert report["converged"] is True
    assert distance <= RTOL, (where, distance, report["iterations"])


@pytest.mark.parametrize("how", [SELECT, _first], ids=["mapping", "transform"])
@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
def test_a_selected_field_is_within_the_tolerance_in_a_weakly_coupled_pair(schedule, how):
    """Both edges deliver a first entry; each member's second entry is
    computed from the other's first and measured by nothing.  On the pass
    the delivered entries met the tolerance, the second entries held what
    the pass before had delivered, nine tolerances away."""
    edges = [("A", "B", G_B_WEAK, how), ("B", "A", G_A_WEAK, how)]
    distance, where, report = _step(
        [A, B], edges, dict(iteration_mode=schedule, convergence_norm="interface", rtol=RTOL))
    assert report["converged"] is True
    assert distance <= RTOL, (
        f"{where} is {distance / RTOL:.1f} tolerances from the fixed point "
        f"(iterations={report['iterations']})")


# ---------------------------------------------------------------------------
# More than one step: a recomputed field is the next step's start
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case", ["no-edge", "mapping", "transform"])
def test_every_step_of_a_trajectory_ends_at_its_own_fixed_point(case):
    """Three steps.  ``B`` and ``C`` carry half and a quarter of their
    pre-step value into the next step, so a field returned a pass behind
    is a wrong start too: the trajectory left the closed form's and never
    came back."""
    nodes = [A, B_KEEPS, C_KEEPS]
    edges = ([("A", "B", G, None), ("B", "C", G.T, None)] if case == "no-edge"
             else LOSSY_CHAIN[case])
    state, report = _solve(nodes, edges, dict(
        iteration_mode="jacobi", convergence_norm="interface", rtol=RTOL), steps=3)
    want = None
    for _ in range(3):
        want = _fixed_point(nodes, edges, pre=want)
    distance, where = _distance(state, want)
    assert report["converged"] is True
    assert distance <= 64 * _eps(False), (where, distance)


# ---------------------------------------------------------------------------
# Derivatives: a recomputed field carries the fixed point's
# ---------------------------------------------------------------------------

#: ``case: (nodes, edges, the recomputed field differentiated)``.
DERIVATIVES = {
    "no-edge": ([A, B], [("A", "B", G, None)], ("B", "u")),
    "mapping": ([A, B, C], LOSSY_CHAIN["mapping"], ("B", "u")),
    "transform": ([A, B, C], LOSSY_CHAIN["transform"], ("B", "u")),
}


@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("case", sorted(DERIVATIVES))
def test_a_recomputed_field_carries_the_fixed_points_derivative(case, solver):
    """``d field / d A_pre`` through the step, by ``jvp`` and by ``grad``,
    against the closed form's.  The field is recomputed outside the implicit
    rule and differentiated as a pass is; returned a pass behind, it had the
    derivative of the value it held (a one-way target: ``G`` where the fixed
    point's is ``alpha G``)."""
    nodes, edges, (name, f) = DERIVATIVES[case]
    with cd.x64(True):
        gm = _graph(nodes, edges, dict(iteration_mode="jacobi", convergence_norm="interface",
                                       rtol=1e-12, solver=solver, max_iterations=4),
                    dtype=np.float64)
        gm.step()
        assert next(iter(gm.coupling_diagnostics().values()))["converged"] is True
        gm.reset_state()
        compiled, base = gm._compiled_step, gm._state       # noqa: SLF001

        def field(a_pre):
            state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
            state["A"]["u"] = a_pre
            return compiled(state, {})[name][f]

        a0 = jnp.asarray(A[4], jnp.float64)
        tangent, cotangent = jnp.asarray([1.0, -0.5]), jnp.asarray([0.25, 2.0])
        pushed = np.asarray(jax.jvp(field, (a0,), (tangent,))[1])
        pulled = np.asarray(jax.grad(lambda a: field(a) @ cotangent)(a0))

    want = np.zeros((2, 2))
    here = {n_[0]: {"u": np.asarray(n_[4], float)} for n_ in nodes}
    base_point = _fixed_point(nodes, edges, pre=here)[name][f]
    for j in range(2):
        moved = {k: {"u": v["u"].copy()} for k, v in here.items()}
        moved["A"]["u"][j] += 1.0
        want[:, j] = _fixed_point(nodes, edges, pre=moved)[name][f] - base_point
    assert np.max(np.abs(want)) > 0.1, "the field does not depend on A: nothing is tested"
    np.testing.assert_allclose(pushed, want @ np.asarray(tangent), rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(pulled, np.asarray(cotangent) @ want, rtol=1e-9, atol=1e-10)


# ---------------------------------------------------------------------------
# Derivatives at an early exit: every returned field carries the implicit
# rule's derivative of the accepted iterate, so the gradient bound covers it
# ---------------------------------------------------------------------------
#
# ``solver="ift"`` gives the accepted iterate ``x`` the derivative ``g = (I -
# J)^{-1} dF`` of the pass ``F`` linearised at ``x`` (``J = dF/dx``, ``dF``
# the pass's own derivative in the constant), whatever pass ``x`` was
# accepted on; ``gradient_relative_error_bound`` bounds how far ``g`` is from
# the fixed point's (CPL-093).  A recomputed field is ``F(x)``'s, and
# differentiating ``F(x)`` with ``x`` carrying ``g`` gives ``J g + dF = g``:
# **the same derivative, field for field.**  So the state returned has the
# gradient the bound is on, recomputed fields included, and no second bound
# is needed.  Tested with a constant the pass is *not* linear in jointly
# with the iterate -- a mapping weight, which multiplies the field it reads
# -- so that the derivative at an early exit is not the fixed point's.


def _implicit_derivative(nodes, edges, wide, at, schedule, sweep, edge, entry):
    """``{node: {field: d field / d H[entry]}}`` in float64, ``H`` the
    mapping of edge number *edge*: the implicit rule of one plain pass of
    *schedule* linearised at the iterate *at*.

    With ``M`` the group's coupling matrix and each edge read at the value
    the pass reads it at (this pass's, where the sweep ran its source
    first; *at*'s otherwise), ``g = (I - M)^{-1} (gain dH read)``; a field
    no edge reads has ``K (dH read + H g)`` of what it reads.  At the fixed
    point both readings are the fixed point's.
    """
    off, k = {}, 0
    for name, n, *_ in nodes:
        off[name] = k
        k += n
    size = _sizes(nodes)
    after = _plain_pass(nodes, edges, at, schedule, wide, sweep)

    def read(src, dst):
        ahead = schedule == "gauss-seidel" and sweep.index(src) < sweep.index(dst)
        return np.asarray((after if ahead else at)[src]["u"], float)

    M = np.zeros((k, k))
    for src, dst, gain, how in edges:
        M[off[dst]:off[dst] + size[dst], off[src]:off[src] + size[src]] += (
            np.asarray(gain, float) @ _delivers(how, size[src]))
    src, dst, gain, how = edges[edge]
    dH = np.zeros(np.asarray(how).shape)
    dH[entry] = 1.0
    rhs = np.zeros(k)
    rhs[off[dst]:off[dst] + size[dst]] = np.asarray(gain, float) @ dH @ read(src, dst)
    g = np.linalg.solve(np.eye(k) - M, rhs)
    out = {name: {"u": g[off[name]:off[name] + size[name]]} for name in size}
    for name in wide:
        dw = np.zeros(size[name])
        for i, (s_, d_, gain_, how_) in enumerate(edges):
            if d_ != name:
                continue
            through = _delivers(how_, size[s_]) @ out[s_]["u"]
            if i == edge:
                through = through + dH @ read(s_, d_)
            dw = dw + K[:, :np.asarray(gain_).shape[1]] @ through
        out[name]["w"] = dw
    return out


def _weights_of(params, key):
    """``(weights, put)``: the weight matrix of the mapping in slot *key*
    of ``params["mappings"]`` and a function giving the parameter pytree
    with another matrix in its place."""
    slot = params["mappings"][key]
    if not isinstance(slot, dict):
        return slot, lambda q: {**params, "mappings": {**params["mappings"], key: q}}
    ((leaf, weights),) = slot.items()
    return weights, lambda q: {**params, "mappings": {**params["mappings"], key: {leaf: q}}}


#: The cases with a mapping weight to differentiate in: every ``u`` read
#: only through a mapping; a selection, and a field no edge reads; fields
#: measured whole beside a recomputed member.
GRADIENT_CASES = ("mapped", "selected", "tail")
#: Passes short of convergence at ``RTOL`` in every case and schedule.
EARLY = 4


#: ``(case, schedule)``.  Per push: fields measured whole beside a
#: recomputed member and a field no edge reads, under Jacobi; and every
#: field recomputed under a sweep that reads this pass's values.
GRADIENT_ROWS = tuple(itertools.product(GRADIENT_CASES, ("jacobi", "gauss-seidel")))
GRADIENT_PER_PUSH = (("tail", "jacobi"), ("mapped", "gauss-seidel"))


def assert_the_returned_gradient_is_the_accepted_iterates(case, schedule):
    """And so the gradient bound covers the state returned (CPL-093).

    Forward mode over every scalar weight of every mapping, against the
    float64 closed form at the accepted iterate (read with the rule
    switched off).  The premise is shown too: the derivative at this exit
    is not the fixed point's, and it is not the value's own (a field
    returned a pass behind would carry that pass's).  Where the report
    says ``gradient_bound_usable``, the bound is then held against the
    library's own derivative of the returned state: its distance from the
    fixed point's derivative, in the raw fields the norm reads, each over
    its magnitude at the returned state.
    """
    nodes, edges, wide = CASES[case]
    group = dict(iteration_mode=schedule, convergence_norm="interface", rtol=RTOL,
                 solver="ift", max_iterations=EARLY)
    accepted, loop = _solve(nodes, edges, group, wide=wide, rule="none", x64=True)
    assert loop["converged"] is False and loop["iterations"] == EARLY, loop
    mapped = [i for i, (*_e, how) in enumerate(edges) if how is not None and not callable(how)]
    with cd.x64(True):
        gm = _graph(nodes, edges, group, dtype=np.float64, wide=wide, diagnostics=True)
        gm.step()
        (report,) = gm.coupling_diagnostics().values()
        returned = {name: {f: np.asarray(v) for f, v in gm.get_node_state(name).items()}
                    for name, *_ in nodes}
        gm.reset_state()
        step, ext = gm._raw_step_fn, gm._default_external_inputs()      # noqa: SLF001
        params, start = gm.params, gm._state                             # noqa: SLF001
        keys = {i: gm._edges[i].key for i in mapped}                     # noqa: SLF001

        def fields(mappings):
            out = step(start, ext, {**params, "mappings": mappings})
            return {name: {f: out[name][f] for f in returned[name]} for name, *_ in nodes}

        # One compiled tangent map for every weight of every mapping.
        jvp = jax.jit(lambda t: jax.jvp(fields, (params["mappings"],), (t,))[1])
        still = jax.tree.map(jnp.zeros_like, params["mappings"])
        pushed = {}
        for i in mapped:
            weights, put = _weights_of({"mappings": still}, keys[i])
            for entry in np.ndindex(*np.shape(weights)):
                tangent = put(weights.at[entry].set(1.0))["mappings"]
                pushed[(i, entry)] = {name: {f: np.asarray(v) for f, v in fs.items()}
                                      for name, fs in jvp(tangent).items()}
    assert (report["iterations"], report["converged"]) == (EARLY, False), report
    sweep = [m for m in gm.schedule if m in _sizes(nodes)]
    fixed = _fixed_point(nodes, edges, wide)
    whole = _whole(edges)
    #: The raw fields the norm reads, each over its magnitude as returned.
    read = sorted({src for src, *_rest in edges})
    weight = {src: 1.0 / float(np.max(np.abs(returned[src]["u"]))) for src in read}

    def size(g):
        return float(np.sqrt(sum(np.sum((weight[src] * g[src]["u"]) ** 2) for src in read)))

    differs, behind, worst = 0.0, 0.0, 0.0
    for (i, entry), got in pushed.items():
        here = _implicit_derivative(nodes, edges, wide, accepted, schedule, sweep, i, entry)
        there = _implicit_derivative(nodes, edges, wide, fixed, schedule, sweep, i, entry)
        scale = max(float(np.max(np.abs(f))) for fs in here.values() for f in fs.values())
        for name, fs in got.items():
            for f, value in fs.items():
                assert np.max(np.abs(value - here[name][f])) <= 1e-9 * max(scale, 1e-3), (
                    f"{case} {schedule}: d {name}.{f} / d H{i}{list(entry)} of the returned "
                    f"state is {value}; the implicit rule at the accepted iterate gives "
                    f"{here[name][f]} ({'kept' if (name, f) in whole else 'recomputed'})")
                differs = max(differs, float(np.max(np.abs(here[name][f] - there[name][f]))))
        gap = {src: {"u": got[src]["u"] - there[src]["u"]} for src in read}
        if size(got) > 0:
            worst = max(worst, size(gap) / size(got))
        behind = max(behind, scale)
    assert differs > 1e-3 * behind, (
        "at this exit the implicit derivative is the fixed point's: nothing is tested")
    if report["gradient_bound_usable"]:
        bound = float(report["gradient_relative_error_bound"])
        assert worst <= bound * (1.0 + 1e-6) + 64 * _eps(True), (
            f"{case} {schedule}: the derivative of the returned state is {worst:.4g} "
            f"(relative) from the fixed point's; the report bounds it by {bound:.4g}")
    # Every row's report says so (jaxlib 0.10.2, 0.11.0 and 0.11.2): the
    # bound half of the check ran.
    assert report["gradient_bound_usable"], (
        f"{case} {schedule}: the gradient bound is no longer usable here, and the bound "
        f"half of this check ran on nothing ({report})")


@pytest.mark.parametrize("row", GRADIENT_PER_PUSH, ids=_row_id)
def test_at_an_early_exit_every_returned_field_carries_the_accepted_iterates_implicit_derivative(
        row):
    assert_the_returned_gradient_is_the_accepted_iterates(*row)


# Per push: tests/core/test_coupling_interface_norm_answers_for_the_state_it_returns.py::test_at_an_early_exit_every_returned_field_carries_the_accepted_iterates_implicit_derivative
@pytest.mark.slow
@pytest.mark.parametrize("row", [r for r in GRADIENT_ROWS if r not in GRADIENT_PER_PUSH],
                         ids=_row_id)
def test_the_returned_gradient_is_the_accepted_iterates_in_every_case_and_schedule(row):
    assert_the_returned_gradient_is_the_accepted_iterates(*row)


@pytest.mark.parametrize("case", ["tail"])
def test_under_the_unrolled_solver_a_returned_field_has_the_derivative_of_its_value(case):
    """``solver="fori"`` differentiates the passes it ran: at an early exit
    the derivative of every returned field, recomputed or kept, is the
    central difference of that field as returned (the map is linear in
    one weight at a time up to the products the passes compound, so a
    step of ``1e-6`` is good to ``1e-9``)."""
    nodes, edges, wide = CASES[case]
    group = dict(iteration_mode="jacobi", convergence_norm="interface", rtol=RTOL,
                 solver="fori", max_iterations=EARLY)
    i = next(i for i, (*_e, how) in enumerate(edges) if how is not None and not callable(how))
    with cd.x64(True):
        gm = _graph(nodes, edges, group, dtype=np.float64, wide=wide)
        gm.step()
        (report,) = gm.coupling_diagnostics().values()
        assert (report["iterations"], report["converged"]) == (EARLY, False), report
        names = {name: tuple(gm.get_node_state(name)) for name, *_ in nodes}
        gm.reset_state()
        step, ext = gm._raw_step_fn, gm._default_external_inputs()      # noqa: SLF001
        params, start, key = gm.params, gm._state, gm._edges[i].key      # noqa: SLF001
        weights, put = _weights_of(params, key)

        @jax.jit
        def fields(q):
            out = step(start, ext, put(q))
            return {name: {f: out[name][f] for f in fs} for name, fs in names.items()}

        tangent = jnp.zeros_like(weights).at[(0, 0)].set(1.0)
        pushed = jax.jit(lambda t: jax.jvp(fields, (weights,), (t,))[1])(tangent)
        h = 1e-6
        up, down = fields(weights + h * tangent), fields(weights - h * tangent)
    moved = 0.0
    for name, fs in names.items():
        for f in fs:
            central = (np.asarray(up[name][f]) - np.asarray(down[name][f])) / (2.0 * h)
            np.testing.assert_allclose(np.asarray(pushed[name][f]), central, rtol=1e-7,
                                       atol=1e-8, err_msg=f"{case}: {name}.{f}")
            moved = max(moved, float(np.max(np.abs(central))))
    assert moved > 1e-2, "no field depends on the weight: nothing is tested"


# ---------------------------------------------------------------------------
# The static rule
# ---------------------------------------------------------------------------

_STATE = {
    "a": {"u": jnp.zeros(2), "w": jnp.zeros(2), "count": jnp.zeros((), jnp.int32)},
    "b": {"u": jnp.zeros(2), "w": jnp.zeros(2)},
    "c": {"u": jnp.zeros(2)},
}


def _named(edges, *, order=("a", "b", "c"), **group):
    group.setdefault("convergence_norm", "interface")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = CouplingGroup(nodes=frozenset(order), **group)
    return _fields_the_interface_norm_misses(g, edges, list(order), _STATE)


def _edge(src, dst, **kw):
    return EdgeSpec(src, dst, "u", "inp", **kw)


_RING = [_edge("a", "b"), _edge("b", "c"), _edge("c", "a")]
_MAPPED = matrix_mapping(np.eye(2, dtype=np.float32))


def test_only_the_interface_norm_misses_a_field():
    for norm in ("l2", "mixed"):
        assert _named(_RING, convergence_norm=norm, iteration_mode="jacobi") == {}
        assert _named([_edge("a", "b", mapping=_MAPPED)], convergence_norm=norm) == {}


def test_a_field_an_edge_delivers_as_it_is_is_measured_whole_and_never_named():
    """Every ``u`` of the ring is; the unread ``w`` are named, a counter never."""
    assert _named(_RING) == {"a": ("w",), "b": ("w",)}


def test_a_group_whose_every_floating_field_is_measured_whole_names_nothing():
    """Such a group keeps the compiled step it had and pays no pass."""
    state = {"a": {"u": jnp.zeros(2), "count": jnp.zeros((), jnp.int32)}, "b": {"u": jnp.zeros(2)}}
    g = CouplingGroup(nodes=frozenset("ab"), convergence_norm="interface")
    pair = [_edge("a", "b"), _edge("b", "a")]
    assert _fields_the_interface_norm_misses(g, pair, ["a", "b"], state) == {}


@pytest.mark.parametrize("acceleration", sorted(ACCELERATIONS))
@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
def test_every_other_floating_field_of_every_member_is_named_whatever_the_loop(
        schedule, acceleration):
    """One rule: a member no internal edge feeds, and one a Gauss-Seidel
    sweep feeds forward, are named like the rest."""
    kw = dict(iteration_mode=schedule, acceleration=acceleration)
    assert _named([_edge("a", "b")], **kw) == {"a": ("w",), "b": ("u", "w"), "c": ("u",)}
    assert _named([_edge("b", "a")], **kw) == {"a": ("u", "w"), "b": ("w",), "c": ("u",)}


@pytest.mark.parametrize("lossy", [dict(mapping=_MAPPED), dict(transform=_first)],
                         ids=["mapping", "transform"])
def test_a_field_read_only_through_a_mapping_or_a_transform_is_named(lossy):
    """Even an identity mapping: what a mapping delivers cannot be told from
    the field statically."""
    ring = [_edge("a", "b", **lossy), _edge("b", "c"), _edge("c", "a")]
    assert _named(ring) == {"a": ("u", "w"), "b": ("w",)}


@pytest.mark.parametrize("lossy", [dict(mapping=_MAPPED), dict(transform=_first)],
                         ids=["mapping", "transform"])
def test_a_field_one_edge_delivers_whole_is_not_named_for_another_that_maps_it(lossy):
    """Recomputing a field the norm measures would move a reading the
    verdict was taken on."""
    edges = [_edge("a", "b"), _edge("a", "c", **lossy), _edge("b", "a"), _edge("c", "a")]
    assert _named(edges) == {"a": ("w",), "b": ("w",)}


def test_the_set_is_the_same_from_the_groups_plan_as_from_its_bare_edges():
    """The step hands the rule the group's plan (``InterfacePlan``); a
    direct call may hand it the edges.  One answer."""
    edges = [_edge("a", "b"), _edge("b", "c", mapping=_MAPPED), _edge("c", "a", transform=_first)]
    order = ["a", "b", "c"]
    plan = _interface_plan.interface_plan(frozenset(order), edges, order, _STATE, None)
    g = CouplingGroup(nodes=frozenset(order), convergence_norm="interface")
    want = {"a": ("w",), "b": ("u", "w"), "c": ("u",)}
    assert _fields_the_interface_norm_misses(g, plan, order, _STATE) == want
    assert _fields_the_interface_norm_misses(g, edges, order, _STATE) == want


def test_the_set_follows_the_plans_answer_on_which_edge_reads_its_source_as_it_is():
    """Which fields the norm measures whole on an edge is the plan's to
    say (``InterfaceEdge.measured_whole``: the fields the parts of its
    reading hold entry for entry), on whichever side it reads that edge.
    With that answer changed for a mapped edge -- as a rule that reads a
    static mapping at its source changes it, and one that reads a
    geometry-dependent mapping's positions too adds a second field -- the
    field is measured whole and no longer named; the rule holds no test
    of an edge's mapping or transform of its own."""
    edges = [_edge("a", "b", mapping=_MAPPED), _edge("b", "a")]
    assert _named(edges) == {"a": ("u", "w"), "b": ("w",), "c": ("u",)}
    with mock.patch.object(_interface_plan.InterfaceEdge, "measured_whole",
                           property(lambda self: (self.source,))):
        assert _named(edges) == {"a": ("w",), "b": ("w",), "c": ("u",)}
    with mock.patch.object(_interface_plan.InterfaceEdge, "measured_whole",
                           property(lambda self: ())):
        assert _named(edges) == {"a": ("u", "w"), "b": ("u", "w"), "c": ("u",)}
    with mock.patch.object(_interface_plan.InterfaceEdge, "measured_whole",
                           property(lambda self: (self.source, (self.source[0], "w")))):
        assert _named(edges) == {"c": ("u",)}, "a second field of the source, held whole"


def test_which_fields_are_floating_is_decided_on_the_state_handed_not_on_the_plans():
    """The norm's reading decides it on the state it is handed, and the
    step hands the rule the first pass's.  A plan built from a state in
    which a source field was not yet floating must not make the rule
    recompute a field the norm measures whole; and a field that is not
    floating in the state handed is never named (it is recomputed with the
    other non-floating fields)."""
    edges = [_edge("a", "b"), _edge("b", "a")]
    order = ["a", "b"]
    floating = {"a": {"u": jnp.zeros(2)}, "b": {"u": jnp.zeros(2)}}
    counted = {"a": {"u": jnp.zeros(2, jnp.int32)}, "b": {"u": jnp.zeros(2)}}
    g = CouplingGroup(nodes=frozenset(order), convergence_norm="interface")
    before = _interface_plan.interface_plan(frozenset(order), edges, order, counted, None)
    assert before.internal[0].source_kind == _interface_plan.NON_FLOATING
    assert _fields_the_interface_norm_misses(g, before, order, floating) == {}
    after = _interface_plan.interface_plan(frozenset(order), edges, order, floating, None)
    assert _fields_the_interface_norm_misses(g, after, order, counted) == {}

