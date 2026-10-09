"""An edge from a node to itself: which of the node's own values it reads.

``add_edge("a", "a", ...)`` is accepted, and it means one of two things:

* **outside a coupling group** it is a back edge: the node reads its own
  value from the *previous* step, so the term the edge carries is explicit.
  With ``dx/dt = -k x`` delivered through the edge, one step is
  ``x (1 - k dt)``;
* **inside a coupling group** (a group of that one node is enough) it is an
  internal edge and is iterated with the group: at convergence the node
  reads its own *new* value, so the term is implicit, ``x / (1 + k dt)``.

Nothing else decides it: not the node's other edges, not its neighbours'
groups, not the order the graph was built in.  ``compile()`` warns in
neither case, by decision: putting the node in a group is how a term is
made implicit.  ``validate()`` says which of the two it is, in one
``INFO:`` line per such edge, as it names every longer loop; a graph that
draws only such lines is valid.

This module reads the values from the compiled graph, against closed forms
in float64, across what could change them: the schedule, the solver, the
acceleration, an iteration cap below convergence, a multi-rate graph, a
sub-cycled member, a predictor, ``run_adaptive``, gradients, and both
precisions.  What it found beside the two sentences, each held below:

* "at convergence" matters.  A pass reads the previous iterate, and the
  first pass the previous step: ``max_iterations=1`` is the explicit value,
  and a cap of ``n`` the first ``n`` terms of ``1 - k dt + (k dt)^2 - ...``.
  That series converges only for ``|k dt| < 1``; an acceleration carries
  the group past it.
* A **sub-cycled** member reads the group's iterate, its own value at the
  end of the *group's* step, at every one of its sub-steps: the term is
  implicit over the group's step, not over the member's
  (MADD-ANO-027: the sub-step interpolation runs between iterates).
* A self-edge on a **flux** cannot be read outside a group at all
  (MADD-ANO-157, a bare ``KeyError``): the previous step's flux is not
  kept.  Inside a group it is iterated like any other.

The wide products (the 40 graphs around a neighbour, schedule by solver by
acceleration, scheme by precision of the gradient, interpolation by
schedule of a sub-cycled member) run whole in the slow lane; a part of
each that holds every value of every axis at least once runs on every
push, with everything else in this module.
"""

from __future__ import annotations

import itertools
import logging
import re
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.nodes.heat import HeatNode
from tests.core.coupling_domains import x64

K = 1.0         # the rate the self-edge carries: dx/dt = -K x
DT = 0.01       # so K * DT = 0.01: the two schemes differ in the fourth digit
#: The neighbour's gains: ``dx_a/dt += C_BA x_b`` and ``dx_b/dt = C_AB x_a``.
C_BA, C_AB = 0.7, -0.4
X_A, X_B = 1.0, 2.0
#: A converged float64 group, and how close to the closed form it must be.
TIGHT = dict(max_iterations=200, tolerance=1e-13)
CLOSE = 1e-11

EXPLICIT = 1.0 - K * DT             # 0.99
IMPLICIT = 1.0 / (1.0 + K * DT)     # 0.990099...

#: What ``validate()`` says after naming the edge.  Written out here, not
#: read from the library: the two sentences of ``add_edge``'s docstring,
#: and the one for a flux that cannot be read (MADD-ANO-157).  The copies
#: of these lines in the two guides are held to the library by
#: ``tests/compliance/test_validate_lines_in_the_guides.py``.
READS_THE_PREVIOUS_STEP = (
    "Outside a coupling group it is a back edge: the node reads its state of the previous "
    "step, so the term the edge carries is explicit.")
ITERATED_WITH = (
    "With the node in a coupling group ({members}) the edge is iterated with the group and at "
    "convergence the node reads its new value, so the term is implicit.")
A_FLUX_CANNOT_BE_READ = (
    "Outside a coupling group it is a back edge, and its source is a boundary flux, which "
    "cannot be read there: the previous step's flux is not kept, and the step raises instead "
    "of reading it (MADD-ANO-157). Put the node in a group, or carry the quantity in a state "
    "field.")


def _says(source, target="u", node="a", *, group=None, flux=False) -> str:
    """``validate()``'s line for the edge ``node.source -> node.target``.

    *group*: the members of the coupling group the node is in.  *flux*: the
    source is a boundary flux (it matters outside a group only).
    """
    reading = (ITERATED_WITH.format(members=sorted(group)) if group is not None
               else A_FLUX_CANNOT_BE_READ if flux else READS_THE_PREVIOUS_STEP)
    return f"INFO: edge {node}.{source} -> {node}.{target} is from node '{node}' to itself. {reading}"


def _cells(per_push, *axes, slow_only=()):
    """The product of *axes* as parameters: the cells not in *per_push* are slow.

    Every value of every axis is in a per-push cell, except the values
    named in *slow_only* (said where it is used, with why).  The ids are
    the ones stacked ``parametrize`` decorators give (a tuple's items
    joined by a hyphen).
    """
    cells = list(itertools.product(*axes))
    assert set(per_push) <= set(cells), set(per_push) - set(cells)
    for i, axis in enumerate(axes):
        assert {cell[i] for cell in per_push} == set(axis) - set(slow_only), (i, axis)
    return [pytest.param(*cell, marks=() if cell in per_push else pytest.mark.slow,
                         id="-".join(v if isinstance(v, str) else "-".join(v) for v in cell))
            for cell in cells]


class Rate(SimulationNode):
    """``x <- x + dt * g * u``: one explicit Euler step of ``dx/dt = g u``.

    With ``x`` wired back into ``u`` the node integrates ``dx/dt = g x``, and
    whether that term is explicit or implicit is the edge's time level and
    nothing in the node.  ``g`` is a parameter, so a gradient reaches it.
    """

    def __init__(self, name, timestep, dtype, *, g=1.0, x0=1.0):
        super().__init__(name, timestep, g=jnp.asarray(g, dtype))
        self._dtype, self._x0 = dtype, x0

    def initial_state(self):
        return {"x": jnp.asarray(self._x0, self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=self._dtype,
                                       default=jnp.zeros((), self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = jnp.asarray(boundary_inputs.get("u", 0.0)).astype(self._dtype)
        step = jnp.asarray(dt, self._dtype) * p["g"] * u
        return {"x": (state["x"] + step).astype(self._dtype)}


class FluxRate(Rate):
    """A :class:`Rate` that also produces the flux ``q = 2 x``."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt):
        return {"q": 2.0 * state["x"]}


def _loss(temperature):
    """The rod's own term: heat lost in proportion to its temperature."""
    return -K * temperature


def _compile(gm) -> list[str]:
    """Compile *gm*; every warning it raised, as text.

    The ones the suite's configuration ignores are recorded too, except the
    notice that a node has no edges to any other.  (That ``solver="fori"``
    is deprecated is said when the group is made, before this.)
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
    return [str(w.message) for w in caught if "is disconnected" not in str(w.message)]


def _alone(group=None, *, dtype=jnp.float64, k=K, dt=DT, node=Rate, field="x"):
    """One node whose own ``x`` (or flux) feeds its ``u`` at gain ``-k``; compiled."""
    gm = GraphManager()
    gm.add_node(node("a", dt, dtype, g=-k))
    gm.add_edge("a", "a", field, "u")
    if group is not None:
        gm.add_coupling_group(["a"], **group)
    # One line from validate(), which says which value the edge reads, and
    # no warning from compile().
    assert gm.validate() == [_says(field, group=None if group is None else ["a"],
                                   flux=field != "x")]
    assert _compile(gm) == []
    return gm


def _x(gm, name="a") -> float:
    return float(gm.get_node_state(name)["x"])


def _by_the_rule(x, dt, edges, groups, schedule) -> dict:
    """One step of ``x_n <- x_n + dt_n * sum(gain * x_source)`` in float64, each
    edge read at the level the documentation gives it.

    An edge is read at the *new* level when its two ends are members of one
    coupling group, or when its source runs earlier in the step; otherwise
    at the previous step's.  A self-edge's source never runs earlier than
    itself, so only the group decides.  *edges* are ``(source, target,
    gain)``; the new values solve a linear system.
    """
    names = sorted(x)
    at = {n: i for i, n in enumerate(names)}
    place = {n: i for i, n in enumerate(schedule)}
    group_of = {n: i for i, g in enumerate(groups) for n in g}
    A = np.eye(len(names))
    rhs = np.array([x[n] for n in names], np.float64)
    for source, target, gain in edges:
        together = group_of.get(source, -1) == group_of.get(target, -2)
        if together or place[source] < place[target]:
            A[at[target], at[source]] -= dt[target] * gain
        else:
            rhs[at[target]] += dt[target] * gain * x[source]
    return dict(zip(names, np.linalg.solve(A, rhs)))


# ---------------------------------------------------------------------------
# The two sentences, on a shipped node
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_outside_a_group_it_reads_the_previous_step_and_inside_one_its_new_value(precision):
    """A uniform rod losing heat in proportion to its own temperature.

    ``a.temperature -> a.heat_source`` times ``-k``: with no group one step
    is ``1 - k dt`` and three are its cube; in a group of the one rod it is
    ``1 / (1 + k dt)``.  Neither graph draws a warning from ``compile()``;
    ``validate()`` gives the edge's one line and ``format_graph()`` names
    its level, in each.  In float32 and, under x64, in float64.
    """
    with x64(precision == "float64"):
        dtype = jnp.dtype(precision)
        # float32 holds 0.99 to about 6e-8; the schemes are 1e-4 apart.
        close = CLOSE if precision == "float64" else 5e-7

        def rod(group):
            gm = GraphManager()
            gm.add_node(HeatNode("a", DT, n_cells=4, thermal_diffusivity=0.1))
            gm.add_edge("a", "a", "temperature", "heat_source", transform=_loss)
            if group:
                gm.add_coupling_group(["a"], max_iterations=200,
                                      tolerance=1e-13 if precision == "float64" else 1e-6)
            assert gm.validate() == [_says("temperature", "heat_source",
                                           group=["a"] if group else None)]
            assert _compile(gm) == []
            gm.set_node_state("a", {"temperature": jnp.ones(4, dtype)})
            return gm

        for group, one_step, level in (
                (False, EXPLICIT, "back edge: reads the previous step's value"),
                (True, IMPLICIT, "iterated inside its coupling group")):
            gm = rod(group)
            assert re.search(r"a\.temperature -> a\.heat_source\n +state; [^\n]*"
                             + re.escape(level) + r"\n", gm.format_graph())
            T = gm.step()["a"]["temperature"]
            assert T.dtype == dtype
            np.testing.assert_allclose(np.asarray(T, np.float64), one_step, rtol=0, atol=close)
            T3 = rod(group).run_scan(3)["a"]["temperature"]
            np.testing.assert_allclose(np.asarray(T3, np.float64), one_step ** 3,
                                       rtol=0, atol=3 * close)
        assert abs(EXPLICIT - IMPLICIT) > 100 * close       # the test can tell them apart


def test_auto_couple_puts_no_group_around_a_node_whose_only_loop_is_its_own_edge():
    """``auto_couple()`` groups cycles of two or more nodes: the edge stays explicit."""
    with x64(True):
        gm = GraphManager()
        gm.add_node(Rate("a", DT, jnp.float64, g=-K))
        gm.add_edge("a", "a", "x", "u")
        assert gm.auto_couple(**TIGHT) == []
        assert gm.validate() == [_says("x")] and _compile(gm) == []
        gm.step()
        assert _x(gm) == pytest.approx(EXPLICIT, abs=CLOSE)


# ---------------------------------------------------------------------------
# What validate() says, and that nothing takes the graph for an invalid one
# ---------------------------------------------------------------------------
def _built(group=None, *, node=Rate, field="x"):
    """``a`` with ``a.<field> -> a.u``, in a group of itself if *group*; not compiled."""
    gm = GraphManager()
    gm.add_node(node("a", DT, jnp.float32, g=-K))
    gm.add_edge("a", "a", field, "u")
    if group is not None:
        gm.add_coupling_group(["a"], **group)
    return gm


def test_validate_says_which_of_its_own_values_the_node_reads_through_the_edge():
    """The lines, word for word: the two sentences of ``add_edge``'s docstring.

    Outside a group, and in a group of the one node; a group of two names
    both members, sorted, whatever order they were given in.  An edge from
    the node's own flux is iterated inside a group like any other, and is
    told so.
    """
    assert _built().validate() == [
        "INFO: edge a.x -> a.u is from node 'a' to itself. Outside a coupling group it is a "
        "back edge: the node reads its state of the previous step, so the term the edge carries "
        "is explicit."]
    assert _built({}).validate() == [
        "INFO: edge a.x -> a.u is from node 'a' to itself. With the node in a coupling group "
        "(['a']) the edge is iterated with the group and at convergence the node reads its new "
        "value, so the term is implicit."]
    assert _built({}, node=FluxRate, field="q").validate() == [
        "INFO: edge a.q -> a.u is from node 'a' to itself. With the node in a coupling group "
        "(['a']) the edge is iterated with the group and at convergence the node reads its new "
        "value, so the term is implicit."]
    two = _built()
    two.add_node(Rate("b", DT, jnp.float32))
    two.add_edge("a", "b", "x", "u")
    two.add_coupling_group(["b", "a"])
    assert two.validate() == [
        "INFO: edge a.x -> a.u is from node 'a' to itself. With the node in a coupling group "
        "(['a', 'b']) the edge is iterated with the group and at convergence the node reads its "
        "new value, so the term is implicit."]


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
def test_validate_gives_each_edge_to_itself_its_own_line_after_the_lines_for_longer_loops(mode):
    """One line per edge, in the order the edges were added, after the loops' lines.

    ``a`` reads its own flux and its own state, ``b`` its own state, and
    the two read each other.  With no group the loop ``a -> b`` is
    staggered and each edge to itself reads the previous step (the flux
    cannot); with ``b`` alone in a group only ``b``'s line changes; with
    both in one group the loop is iterated in the group's schedule, which
    its line names, and so is every edge to itself.
    """
    def graph(*groups):
        gm = GraphManager()
        gm.add_node(FluxRate("a", DT, jnp.float32))
        gm.add_node(Rate("b", DT, jnp.float32))
        for source, target, field in (("a", "a", "q"), ("b", "a", "x"), ("a", "b", "x"),
                                      ("a", "a", "x"), ("b", "b", "x")):
            gm.add_edge(source, target, field, "u", additive=True)
        for members in groups:
            gm.add_coupling_group(list(members), iteration_mode=mode)
        return gm

    staggered = ("INFO: cycle detected: a -> b. Back-edges will use previous-timestep values "
                 "(staggering).")
    assert graph().validate() == [
        staggered, _says("q", flux=True), _says("x"), _says("x", node="b")]
    assert graph(["b"]).validate() == [
        staggered, _says("q", flux=True), _says("x"), _says("x", node="b", group=["b"])]
    name = {"gauss-seidel": "Gauss-Seidel", "jacobi": "Jacobi"}[mode]
    both = ["b", "a"]
    assert graph(both).validate() == [
        f"INFO: cycle a -> b handled by iterative coupling ({name}).",
        _says("q", group=both), _says("x", group=both), _says("x", node="b", group=both)]


def test_an_edge_to_itself_from_a_field_that_does_not_exist_has_its_error_and_no_line():
    """Nothing is read through it, so nothing is said about what it reads."""
    for group in (None, {}):
        gm = _built(group, field="no_such_field")
        assert gm.validate() == [
            "ERROR: source field 'no_such_field' not in state of node 'a'. Available: ['x']"]


def test_the_line_for_an_edge_outside_a_group_is_logged_as_a_staggered_loop_is(caplog):
    """``logging`` at ``INFO`` on the graph manager's logger, and never a warning.

    Outside a group the edge is a back edge, and its line goes where the
    line for a staggered loop of two goes; inside a group nothing is
    logged, as nothing is for a loop a group iterates.
    """
    logger = "maddening.core.graph_manager"
    for gm, logged in ((_built(), True), (_built(node=FluxRate, field="q"), True),
                       (_built({}), False)):
        caplog.clear()
        with caplog.at_level(logging.INFO, logger=logger), warnings.catch_warnings():
            warnings.simplefilter("error")
            (line,) = gm.validate()
        records = [(r.name, r.levelno, r.getMessage()) for r in caplog.records]
        assert records == ([(logger, logging.INFO, line.removeprefix("INFO: "))] if logged else [])


@pytest.mark.parametrize("group", [None, {}], ids=["no group", "a group of the node"])
def test_a_graph_whose_only_remark_is_that_line_compiles_and_steps_with_warnings_as_errors(group):
    """An ``INFO:`` line is not an error and not a warning.

    ``compile()`` refuses a graph for its ``ERROR`` lines and warns for its
    ``WARNING`` lines; under ``warnings.simplefilter("error")``, which is
    how a project that turns warnings into errors runs, the graph compiles
    and steps.
    """
    gm = _built(group)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert [line[:5] for line in gm.validate()] == ["INFO:"]
        gm.compile()
        gm.step()
    assert _x(gm) == pytest.approx(EXPLICIT if group is None else IMPLICIT, abs=5e-7)


# ---------------------------------------------------------------------------
# What it depends on: the node's own group, and nothing else
# ---------------------------------------------------------------------------
_NEIGHBOUR_EDGES = {"none": (), "b->a": ("ba",), "a->b": ("ab",), "both": ("ba", "ab")}
_GROUPS = {"no group": (), "[a]": (("a",),), "[b]": (("b",),),
           "[a, b]": (("a", "b"),), "[a] and [b]": (("a",), ("b",))}


def _pair(neighbour, groups, order=("a", "b"), self_edge_last=False, group_kw=None):
    """``a`` with its own edge, and a neighbour ``b`` joined to it by *neighbour*.

    Every edge into ``a.u`` is additive, so the input is their sum whatever
    their order.  Each of *groups* is made with *group_kw* (a converged
    float64 group by default).  Returns the graph and its edges as
    ``(source, target, gain)``.
    """
    gm = GraphManager()
    for name in order:
        gm.add_node(Rate(name, DT, jnp.float64, x0=X_A if name == "a" else X_B))
    edges = []

    def own():
        gm.add_edge("a", "a", "x", "u", transform=lambda v: -K * v, additive=True)
        edges.append(("a", "a", -K))

    if not self_edge_last:
        own()
    if "ba" in neighbour:
        gm.add_edge("b", "a", "x", "u", transform=lambda v: C_BA * v, additive=True)
        edges.append(("b", "a", C_BA))
    if "ab" in neighbour:
        gm.add_edge("a", "b", "x", "u", transform=lambda v: C_AB * v, additive=True)
        edges.append(("a", "b", C_AB))
    if self_edge_last:
        own()
    for members in groups:
        gm.add_coupling_group(list(members), **(group_kw or TIGHT))
    return gm, edges


_ORDERS = (("a", "b"), ("b", "a"))
#: Of the 40 graphs, the ones stepped on every push: the eight with no group
#: (an uncoupled step compiles in a few hundredths of a second) and four
#: grouped ones, one for each grouping, among them the group of one inside
#: the loop ``a -> b -> a`` and the group of both.
_LEVEL_PER_PUSH = (
    *((neighbour, "no group", order) for neighbour in _NEIGHBOUR_EDGES for order in _ORDERS),
    ("b->a", "[b]", ("a", "b")),
    ("both", "[a]", ("b", "a")),
    ("both", "[a, b]", ("a", "b")),
    ("a->b", "[a] and [b]", ("a", "b")),
)


# Per push: tests/core/test_an_edge_from_a_node_to_itself.py::test_the_time_level_depends_only_on_whether_the_node_itself_is_in_a_group
# (the 12 graphs of _LEVEL_PER_PUSH; the other 28 are the slow ones)
@pytest.mark.parametrize("neighbour, groups, order", _cells(
    _LEVEL_PER_PUSH, tuple(_NEIGHBOUR_EDGES), tuple(_GROUPS), _ORDERS))
def test_the_time_level_depends_only_on_whether_the_node_itself_is_in_a_group(
        neighbour, groups, order):
    """A neighbour, its edges, its group and the build order change nothing.

    ``a`` reads itself; ``b`` is joined to it by no edge, one edge either
    way, or both; the groups are none, ``[a]``, ``[b]`` alone, both in one,
    and one each; the nodes are added in either order.  Each of the 40
    graphs steps to what the rule gives when the self-edge's level is
    decided by ``a``'s membership of a group alone, and ``a`` is on the
    implicit side of ``b``'s contribution exactly when it is in a group.
    The only word from ``compile()`` is for a group of one node that sits in
    the loop ``a -> b -> a`` and does not iterate it.
    """
    members = _GROUPS[groups]
    with x64(True):
        gm, edges = _pair(_NEIGHBOUR_EDGES[neighbour], members, order)
        said = _compile(gm)
        if neighbour == "both" and members and ("a", "b") not in members:
            assert said and all("is part of a larger feedback loop" in s for s in said), said
        else:
            assert said == []
        x0 = {"a": X_A, "b": X_B}
        want = _by_the_rule(x0, {"a": DT, "b": DT}, edges, members, gm.schedule)
        gm.step()
        got = {name: _x(gm, name) for name in ("a", "b")}
    assert got == pytest.approx(want, abs=CLOSE)
    # ... and the self-edge's own term, isolated: what a's step would be with
    # every other reading as the step made it.
    from_b = sum(g * (got["b"] if gm.schedule.index("b") < gm.schedule.index("a")
                      or ("a", "b") in members else X_B)
                 for s, t, g in edges if (s, t) == ("b", "a"))
    grouped = any("a" in m for m in members)
    own_term = (-K * got["a"]) if grouped else (-K * X_A)
    assert got["a"] == pytest.approx(X_A + DT * (own_term + from_b), abs=CLOSE)


# Per push: tests/core/test_an_edge_from_a_node_to_itself.py::test_an_additive_edge_to_itself_adds_to_what_another_edge_into_the_input_delivers
# (no group in both orders, and each group in one; the other order of each group is slow)
@pytest.mark.parametrize("groups, own_edge", _cells(
    (("no group", "own edge first"), ("no group", "own edge last"),
     ("[a]", "own edge first"), ("[a, b]", "own edge last")),
    ("no group", "[a]", "[a, b]"), ("own edge first", "own edge last")))
def test_an_additive_edge_to_itself_adds_to_what_another_edge_into_the_input_delivers(
        groups, own_edge):
    """``a.u`` is fed by ``a`` itself and by ``b``, both additive, in either order.

    The input is the sum, each term at its own level: outside a group
    ``x_a + dt (-k x_a + c x_b')`` with ``x_b'`` this step's (``b`` runs
    first), and inside one ``(x_a + dt c x_b') / (1 + k dt)``.
    """
    members = _GROUPS[groups]
    with x64(True):
        gm, _ = _pair(("ba",), members, self_edge_last=own_edge == "own edge last")
        assert _compile(gm) == []
        gm.step()
        got = _x(gm)
    forcing = X_A + DT * C_BA * X_B         # b has no input: x_b' = x_b
    want = forcing / (1.0 + K * DT) if members else forcing - DT * K * X_A
    assert got == pytest.approx(want, abs=CLOSE)


# ---------------------------------------------------------------------------
# Inside a group: every way of iterating reaches the same value
# ---------------------------------------------------------------------------
_ACCELERATIONS = ("none", "aitken", "fixed", "iqn-ils", "iqn-imvj")


def _group_kw(solver, mode, acceleration):
    kw = dict(TIGHT, solver=solver, iteration_mode=mode, acceleration=acceleration)
    if acceleration == "fixed":
        kw["relaxation"] = 0.7
    return kw


# Per push: tests/core/test_an_edge_from_a_node_to_itself.py::test_every_schedule_solver_and_acceleration_makes_the_term_implicit[ift-gauss-seidel-none]
# and its [fori-jacobi-aitken] and [fori-gauss-seidel-fixed]: each solver and each schedule.
# The IQN methods are the costliest programs of the module to compile and are slow here.
# IQN-ILS around an edge to itself runs on every push on a group of the one node, in
# tests/core/test_an_edge_from_a_node_to_itself.py::test_past_k_dt_of_one_the_term_is_implicit_only_under_an_acceleration[iqn-ils]
# and IQN-IMVJ, which differs from it in what it carries from one step to the next, is checked
# in the slow lane only.
@pytest.mark.parametrize("solver, mode, acceleration", _cells(
    (("ift", "gauss-seidel", "none"), ("fori", "jacobi", "aitken"),
     ("fori", "gauss-seidel", "fixed")),
    ("ift", "fori"), ("gauss-seidel", "jacobi"), _ACCELERATIONS,
    slow_only=("iqn-ils", "iqn-imvj")))
def test_every_schedule_solver_and_acceleration_makes_the_term_implicit(
        solver, mode, acceleration):
    """Gauss-Seidel and Jacobi, ``"ift"`` and ``"fori"``, with and without an acceleration.

    In the group ``[a, b]`` with both edges between the two and ``a``'s own:
    the converged step is the implicit one in all three terms, the solution
    of ``x_a' = x_a + dt (-k x_a' + c x_b')``, ``x_b' = x_b + dt c' x_a'``.
    And in the group of ``a`` alone, ``x_a / (1 + k dt)``.
    """
    kw = _group_kw(solver, mode, acceleration)
    with x64(True):
        gm, edges = _pair(("ba", "ab"), [("a", "b")], group_kw=kw)
        assert _compile(gm) == []
        want = _by_the_rule({"a": X_A, "b": X_B}, {"a": DT, "b": DT}, edges,
                            [("a", "b")], gm.schedule)
        gm.step()
        got = {name: _x(gm, name) for name in ("a", "b")}
        alone = _alone(kw)
        alone.step()
        got_alone = _x(alone)
    assert got == pytest.approx(want, abs=CLOSE)
    assert got_alone == pytest.approx(IMPLICIT, abs=CLOSE)


@pytest.mark.parametrize("norm", ["mixed", "interface"])
def test_each_convergence_norm_stops_at_the_implicit_value(norm):
    """``"l2"`` is the default above; ``"mixed"`` and ``"interface"`` carry ``rtol``.

    The interface norm measures what the group's internal edges deliver,
    and the node's edge to itself is one of them (here the only one).
    """
    with x64(True):
        gm = _alone(dict(max_iterations=200, convergence_norm=norm, rtol=1e-13))
        gm.step()
        assert _x(gm) == pytest.approx(IMPLICIT, abs=CLOSE)


# Per push: tests/core/test_an_edge_from_a_node_to_itself.py::test_a_group_stopped_before_it_converges_has_made_the_term_only_partly_implicit[ift]
# (the same three caps under the solver that is not deprecated, and strict_convergence)
@pytest.mark.parametrize("solver", ["ift", pytest.param("fori", marks=pytest.mark.slow)])
def test_a_group_stopped_before_it_converges_has_made_the_term_only_partly_implicit(solver):
    """A pass reads the previous iterate, and the first pass the previous step.

    So ``max_iterations=1`` (one staggered pass) returns the *explicit*
    value, and a cap of ``n`` the first ``n`` corrections of the series
    ``1 - k dt + (k dt)^2 - ...`` whose sum is ``1 / (1 + k dt)``.  Nothing
    is said, as for any group that reaches its cap: it is in the group's
    diagnostics, and ``strict_convergence`` raises.
    """
    with x64(True):
        for cap in (1, 2, 3):
            gm = _alone(dict(max_iterations=cap, tolerance=1e-13, solver=solver))
            gm.step()
            partial = sum((-K * DT) ** j for j in range(cap + 1))
            assert _x(gm) == pytest.approx(partial, abs=1e-12), cap     # the caps are 1e-8 apart
        if solver == "ift":
            strict = _alone(dict(max_iterations=2, tolerance=1e-13, strict_convergence=True))
            with pytest.raises(Exception, match="without converging"):
                strict.step()
    assert sum((-K * DT) ** j for j in range(2)) == EXPLICIT


# Per push: tests/core/test_an_edge_from_a_node_to_itself.py::test_past_k_dt_of_one_the_term_is_implicit_only_under_an_acceleration[iqn-ils]
# (the other IQN method, on the same graph; "none" and "aitken" run on every push as well)
@pytest.mark.parametrize("acceleration", [
    "none", "aitken", "iqn-ils", pytest.param("iqn-imvj", marks=pytest.mark.slow)])
def test_past_k_dt_of_one_the_term_is_implicit_only_under_an_acceleration(acceleration):
    """The plain iteration is that series, which converges only for ``|k dt| < 1``.

    At ``k dt = 1.5`` the explicit step is ``-0.5`` (and unstable) and the
    implicit one ``0.4``.  The unaccelerated group does not reach it: at its
    cap of 30 it returns the series' 31 terms, about ``1e5``, with no word
    (a group reports a cap it reached through its diagnostics, or raises
    under ``strict_convergence``).  Aitken, IQN-ILS and IQN-IMVJ each
    converge to ``0.4``.
    """
    k = 1.5 / DT
    with x64(True):
        gm = _alone(dict(max_iterations=30, tolerance=1e-13, acceleration=acceleration), k=k)
        gm.step()
        got = _x(gm)
    implicit = 1.0 / (1.0 + k * DT)
    assert implicit == pytest.approx(0.4)
    if acceleration == "none":
        assert got == pytest.approx(sum((-k * DT) ** j for j in range(31)), rel=1e-12)
        assert abs(got - implicit) > 1e4
    else:
        assert got == pytest.approx(implicit, abs=CLOSE)


def test_a_predictor_only_moves_where_the_iteration_starts():
    """``predictor="quadratic"`` extrapolates the first guess from earlier steps.

    The converged value is the fixed point either way: five steps are
    ``(1 + k dt) ** -5``.
    """
    with x64(True):
        gm = _alone(dict(TIGHT, predictor="quadratic"))
        for n in range(1, 6):
            gm.step()
            assert _x(gm) == pytest.approx(IMPLICIT ** n, abs=CLOSE), n


# ---------------------------------------------------------------------------
# Multi-rate graphs and sub-cycled members
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("grouped", [False, True], ids=["no group", "a group of each"])
def test_on_a_multi_rate_graph_each_node_reads_its_own_value_at_its_own_rate(grouped):
    """A slow node (rate divider 2) and a fast one, each reading itself.

    The slow node fires on every other base step and holds its state
    between; "the previous step" is the state it held.  Outside a group
    its firing is ``x (1 - k 2 dt)``, inside a group of itself
    ``x / (1 + k 2 dt)``; the fast node takes ``1 - k dt`` or
    ``1 / (1 + k dt)`` on every base step.
    """
    with x64(True):
        gm = GraphManager()
        gm.add_node(Rate("slow", 2 * DT, jnp.float64, g=-K))
        gm.add_node(Rate("fast", DT, jnp.float64, g=-K))
        for name in ("slow", "fast"):
            gm.add_edge(name, name, "x", "u")
            if grouped:
                gm.add_coupling_group([name], **TIGHT)
        assert _compile(gm) == []
        assert "rate divider 2" in gm.format_graph()            # a multi-rate graph
        seen = []
        for _ in range(4):
            gm.step()
            seen.append((_x(gm, "slow"), _x(gm, "fast")))
    slow = 1 / (1 + K * 2 * DT) if grouped else 1 - K * 2 * DT
    fast = IMPLICIT if grouped else EXPLICIT
    want = [(slow ** ((n + 1) // 2), fast ** n) for n in range(1, 5)]
    np.testing.assert_allclose(seen, want, rtol=0, atol=CLOSE)


# Per push: tests/core/test_an_edge_from_a_node_to_itself.py::test_a_sub_cycled_member_reads_its_own_value_at_the_end_of_the_groups_step[linear-gauss-seidel]
# and its [constant-jacobi] and [quadratic-jacobi]: each interpolation, both schedules.
@pytest.mark.parametrize("interpolation, mode", _cells(
    (("constant", "jacobi"), ("linear", "gauss-seidel"), ("quadratic", "jacobi")),
    ("constant", "linear", "quadratic"), ("gauss-seidel", "jacobi")))
def test_a_sub_cycled_member_reads_its_own_value_at_the_end_of_the_groups_step(
        interpolation, mode):
    """``subcycling=True``: the member at half the group's timestep takes two sub-steps.

    Both of them read the group's iterate, which holds the member's value
    at the end of the *group's* step (MADD-ANO-027: the interpolation runs
    between iterates, not across the step).  So the member's own term is
    implicit over the group's step, ``x / (1 + k dt)``: not over its own
    sub-step (``(1 + k dt / 2) ** -2``), and not explicit either.  The same
    under each ``boundary_interpolation`` and both schedules, alone and
    beside its neighbour's edges.
    """
    kw = dict(TIGHT, subcycling=True, boundary_interpolation=interpolation, iteration_mode=mode)
    with x64(True):
        got = {}
        for label, neighbour in (("alone", ()), ("with neighbour", ("ba", "ab"))):
            gm = GraphManager()
            gm.add_node(Rate("a", DT / 2, jnp.float64, x0=X_A))     # two sub-steps per pass
            gm.add_node(Rate("b", DT, jnp.float64, x0=X_B))
            gm.add_edge("a", "a", "x", "u", transform=lambda v: -K * v, additive=True)
            edges = [("a", "a", -K)]
            if neighbour:
                gm.add_edge("b", "a", "x", "u", transform=lambda v: C_BA * v, additive=True)
                gm.add_edge("a", "b", "x", "u", transform=lambda v: C_AB * v, additive=True)
                edges += [("b", "a", C_BA), ("a", "b", C_AB)]
            gm.add_coupling_group(["a", "b"], **kw)
            assert _compile(gm) == []
            assert "sub-cycled x2 per coupling pass" in gm.format_graph()
            gm.step()
            # The group's step is DT for both members: two sub-steps of DT / 2.
            want = _by_the_rule({"a": X_A, "b": X_B}, {"a": DT, "b": DT}, edges,
                                [("a", "b")], gm.schedule)
            got[label] = ({n: _x(gm, n) for n in ("a", "b")}, want)
    for label, (have, want) in got.items():
        assert have == pytest.approx(want, abs=CLOSE), label
    alone = got["alone"][0]["a"]
    assert alone == pytest.approx(IMPLICIT, abs=CLOSE)
    assert abs(alone - (1 + K * DT / 2) ** -2) > 1e-5       # not implicit per sub-step
    assert abs(alone - (1 - K * DT / 2) ** 2) > 1e-5        # nor explicit per sub-step


def test_run_adaptive_takes_each_of_its_steps_by_the_same_rule():
    """The adaptive stepper keeps two half steps for each step it accepts.

    Each is taken by the rule: outside a group the product of
    ``(1 - k h / 2) ** 2`` over the accepted steps ``h``, inside a group of
    the node ``(1 + k h / 2) ** -2``.
    """
    with x64(True):
        for group, half_step in ((None, lambda h: 1 - K * h / 2),
                                 (TIGHT, lambda h: 1 / (1 + K * h / 2))):
            gm = _alone(group)
            _, info = gm.run_adaptive(DT, dt_initial=DT, dt_max=DT, dt_min=DT / 64,
                                      rtol=1e-2, atol=1e-6)
            assert info["n_steps"] >= 1 and sum(info["dt_history"]) == pytest.approx(DT)
            want = float(np.prod([half_step(h) ** 2 for h in info["dt_history"]]))
            assert _x(gm) == pytest.approx(want, abs=CLOSE)


# ---------------------------------------------------------------------------
# Gradients
# ---------------------------------------------------------------------------
# Per push: tests/core/test_an_edge_from_a_node_to_itself.py::test_the_gradient_through_the_step_is_the_gradient_of_the_scheme_it_took[ift-float64]
# and its [fori-float32] and both precisions of no group.  The gradient through an accelerated
# group ("ift, aitken") is checked in the slow lane only: its value is the plain group's.
@pytest.mark.parametrize("scheme, precision", _cells(
    (("no group", "float32"), ("no group", "float64"), ("fori", "float32"), ("ift", "float64")),
    ("no group", "ift", "fori", "ift, aitken"), ("float32", "float64"), slow_only=("ift, aitken",)))
def test_the_gradient_through_the_step_is_the_gradient_of_the_scheme_it_took(scheme, precision):
    """``jax.grad`` of one step in the start value and in the rate.

    Outside a group ``x' = x (1 + g dt)`` (``g = -k``): ``dx'/dx = 1 - k dt``
    and ``dx'/dg = dt x``.  Inside one ``x' = x / (1 - g dt)``:
    ``dx'/dx = 1 / (1 + k dt)`` and ``dx'/dg = dt x / (1 + k dt) ** 2``, the
    implicit scheme's, under the implicit-function solver and the unrolled
    one, plain and accelerated.
    """
    group = {"no group": None, "ift": dict(solver="ift"), "fori": dict(solver="fori"),
             "ift, aitken": dict(solver="ift", acceleration="aitken")}[scheme]
    with x64(precision == "float64"):
        dtype = jnp.dtype(precision)
        tight = precision == "float64"
        gm = GraphManager()
        gm.add_node(Rate("a", DT, dtype, g=-K))
        gm.add_edge("a", "a", "x", "u")
        if group is not None:
            gm.add_coupling_group(["a"], max_iterations=200,
                                  tolerance=1e-13 if tight else 1e-7, **group)
        gm.compile()

        def one_step(x0, g):
            gm.set_node_state("a", {"x": x0})
            params = jax.tree.map(lambda leaf: leaf, gm.params)
            params["nodes"]["a"]["g"] = g
            return gm.step(params=params)["a"]["x"]

        d_x, d_g = jax.grad(one_step, argnums=(0, 1))(jnp.asarray(X_A, dtype),
                                                      jnp.asarray(-K, dtype))
        assert d_x.dtype == dtype and d_g.dtype == dtype
        d_x, d_g = float(d_x), float(d_g)
    if group is None:
        want = (1 - K * DT, DT * X_A)
    else:
        want = (1 / (1 + K * DT), DT * X_A / (1 + K * DT) ** 2)
    # The two schemes' derivatives differ by 1e-4 and 2e-4 of themselves.
    rel = 1e-10 if tight else 2e-5
    assert (d_x, d_g) == pytest.approx(want, rel=rel)


# ---------------------------------------------------------------------------
# A flux
# ---------------------------------------------------------------------------
def test_an_edge_from_a_nodes_own_flux_is_iterated_inside_a_group():
    """``a.q -> a.u`` with ``q = 2 x``: in a group of ``a``, ``x / (1 + 2 k dt)``."""
    with x64(True):
        for mode in ("gauss-seidel", "jacobi"):
            gm = _alone(dict(TIGHT, iteration_mode=mode), node=FluxRate, field="q")
            gm.step()
            assert _x(gm) == pytest.approx(1 / (1 + 2 * K * DT), abs=CLOSE), mode


def test_validate_tells_an_edge_from_a_nodes_own_flux_outside_a_group_that_it_cannot_be_read():
    """Not "reads the previous step": the step does not trace (MADD-ANO-157).

    The line says what happens and what to do instead, and it is an
    ``INFO:`` line like the others, so ``compile()`` behaves as it did:
    no error, no warning.  That the step raises is held by the strict
    xfail below: when MADD-ANO-157 is fixed that test fails, and this line
    becomes the one every other edge outside a group has.
    """
    gm = _built(node=FluxRate, field="q")
    assert gm.validate() == [
        "INFO: edge a.q -> a.u is from node 'a' to itself. Outside a coupling group it is a "
        "back edge, and its source is a boundary flux, which cannot be read there: the previous "
        "step's flux is not kept, and the step raises instead of reading it (MADD-ANO-157). Put "
        "the node in a group, or carry the quantity in a state field."]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gm.compile()


@pytest.mark.xfail(strict=True, raises=KeyError, reason=(
    "CPL-186: a flux edge read from the previous step fails to trace with a bare KeyError "
    "naming the flux (MADD-ANO-157); an edge from a node's own flux is always one outside a "
    "group; pending fix"))
def test_an_edge_from_a_nodes_own_flux_reads_the_previous_step_outside_a_group():
    """Outside a group the edge would read the flux of the state the node held.

    That is ``x (1 - 2 k dt)``.  Today the step does not keep a flux from
    one step to the next, and the lookup raises ``KeyError: 'q'``
    (MADD-ANO-157).  The workaround that entry gives, adding the producer
    before its reader, does not exist for a node reading itself: put the
    node in a group, or carry the quantity in a state field.
    """
    with x64(True):
        gm = _alone(node=FluxRate, field="q")
        gm.step()
        assert _x(gm) == pytest.approx(1 - 2 * K * DT, abs=CLOSE)
