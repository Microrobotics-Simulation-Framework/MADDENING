"""Boundary fluxes that read each other inside a coupling group.

A boundary flux is not part of a group's iterate: every pass rebuilds the
fluxes from nothing in two sweeps (``coupling/_flux_loops.py`` has the
argument).  That is the flux of the iterate for a flux computed from the
state, and for fluxes that read each other along a chain the two sweeps
reach; it is not a solution of fluxes that read each other in a loop.  A
group with such a loop used to stop with ``converged=True`` at a state
that does not solve the coupled equations (4e-4 to 0.8 off on the affine
pair below), and ``validate()`` said the loop was "handled by iterative
coupling".  It is refused at ``compile()``.

Every case here is affine, so the coupled equations are one linear
system, solved in float64 (:func:`exact`): each structure the passes are
right for is held to it under both schedules and both solvers, and each
structure they are not right for is refused.  Which is which was measured
on the tree before the refusal (the comments give the numbers).
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _flux_loops
from maddening.core.coupling.helpers import add_robin_coupling, check_conservation
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryFluxSpec, BoundaryInputSpec, SimulationNode

F = jnp.float32
ZERO = jnp.zeros((), F)
# The states agree with the float64 solution to 4e-7 of themselves where a
# pass is right (measured, tolerance 1e-6); the wrong ones were 4e-4 off
# or more.
CLOSE = 5e-6
# Each schedule with each solver: one pair on every push, the other slow.
PER_PUSH = [("gauss-seidel", "ift"), ("jacobi", "fori")]
SLOW = [("gauss-seidel", "fori"), ("jacobi", "ift")]
SCHEDULES = PER_PUSH + [pytest.param(*cell, marks=pytest.mark.slow) for cell in SLOW]


class Side(SimulationNode):
    """``u <- cu + ku * r`` and the flux ``q = cq + kq * s + mq * u``.

    ``r`` is the input the update reads and ``s`` (*reads*) the one the
    flux reads; ``kq=None`` is a flux computed from the state alone, whose
    hook does not touch its inputs.  *strict* indexes the input, the
    ordinary way to read one; otherwise the hook tolerates its absence.
    """

    def __init__(self, name, cu, ku, cq, kq, mq, *, reads="s", strict=False):
        super().__init__(name=name, timestep=1.0)
        self.k = tuple(np.float32(v) for v in (cu, ku, cq, 0.0 if kq is None else kq, mq))
        self.reads = None if kq is None else reads
        self.strict = strict

    def initial_state(self):
        return {"u": ZERO}

    def boundary_input_spec(self):
        return {name: BoundaryInputSpec(shape=(), dtype=F, default=ZERO) for name in "rs"}

    def boundary_flux_spec(self):
        return {"q": BoundaryFluxSpec(shape=(), dtype=F)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"u": self.k[0] + self.k[1] * boundary_inputs.get("r", ZERO)}

    def flux(self, u, boundary_inputs):
        _cu, _ku, cq, kq, mq = self.k
        q = cq + mq * u
        if self.reads is not None:
            s = (boundary_inputs[self.reads] if self.strict
                 else boundary_inputs.get(self.reads, ZERO))
            q = q + kq * s
        return q

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": self.flux(state["u"], boundary_inputs)}


class Carried(Side):
    """A :class:`Side` that writes its flux to the state field ``q``: the way out."""

    compute_boundary_fluxes = SimulationNode.compute_boundary_fluxes

    def initial_state(self):
        return {"u": ZERO, "q": ZERO}

    def boundary_flux_spec(self):
        return {}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = Side.update(self, state, boundary_inputs, dt)["u"]
        return {"u": u, "q": self.flux(u, boundary_inputs)}


def exact(nodes, edges):
    """The solution of the coupled equations, in float64.

    *edges*: ``(source, field, target, input, gain)``; the edges into one
    input are summed.  Unknowns: every ``u`` and every ``q``.
    """
    names = [node.name for node in nodes]
    n = len(names)
    col = {(name, "u"): i for i, name in enumerate(names)}
    col.update({(name, "q"): n + i for i, name in enumerate(names)})
    A, b = np.eye(2 * n), np.zeros(2 * n)
    for i, node in enumerate(nodes):
        cu, ku, cq, kq, mq = (float(v) for v in node.k)
        b[i], b[n + i] = cu, cq
        A[n + i, i] -= mq
        for source, field, target, name, gain in edges:
            if target != node.name:
                continue
            if name == "r":
                A[i, col[(source, field)]] -= ku * gain
            if name == node.reads:
                A[n + i, col[(source, field)]] -= kq * gain
    z = np.linalg.solve(A, b)
    return {name: z[i] for i, name in enumerate(names)}


def built(nodes, edges, group, **knobs):
    gm = GraphManager()
    for node in nodes:
        gm.add_node(node)
    seen = set()
    for source, field, target, name, gain in edges:
        gm.add_edge(source=source, target=target, source_field=field, target_field=name,
                    transform=None if gain == 1.0 else (lambda v, g=gain: np.float32(g) * v),
                    additive=(target, name) in seen)
        seen.add((target, name))
    if group is not None:
        grouped(gm, group, **knobs)
    return gm


def grouped(gm, group, mode="gauss-seidel", solver="ift"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)     # solver="fori"
        gm.add_coupling_group(group, max_iterations=200, tolerance=1e-6,
                              iteration_mode=mode, solver=solver)
    return gm


def off(gm, reference):
    """The largest distance of a stepped state from *reference*, relative."""
    gm.compile()
    gm.step()
    return max(abs(float(gm.get_node_state(name)["u"]) - value) / max(1.0, abs(value))
               for name, value in reference.items())


def coefficients(kq, i):
    return [(1.0, 0.4, 0.3, kq, -0.6), (-0.5, 0.3, 0.2, kq, -0.4), (0.7, -0.2, 0.1, kq, 0.5)][i]


# ---------------------------------------------------------------------------
# The Robin pair of the developer guide: each node is handed
# alpha * value + (1 - alpha) * flux of the other, and reads it in its
# update and (kq != 0) in its flux.
# ---------------------------------------------------------------------------

ALPHA = 0.5
ROBIN = [("node_b", "u", "node_a", "r", ALPHA), ("node_b", "q", "node_a", "r", 1 - ALPHA),
         ("node_a", "u", "node_b", "r", ALPHA), ("node_a", "q", "node_b", "r", 1 - ALPHA)]


def robin_pair(kq, cls=Side, alpha=ALPHA, **knobs):
    nodes = [cls("node_a", *coefficients(kq, 0), reads="r"),
             cls("node_b", *coefficients(kq, 1), reads="r")]
    gm = GraphManager()
    for node in nodes:
        gm.add_node(node)
    add_robin_coupling(gm, "node_a", "node_b", value_field_a="u", flux_field_a="q",
                       value_field_b="u", flux_field_b="q", input_a="r", input_b="r",
                       alpha=alpha)
    return grouped(gm, ["node_a", "node_b"], **knobs), nodes


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_a_flux_computed_from_the_state_alone_reaches_the_coupled_solution[gauss-seidel-ift]
# and its [jacobi-fori]: each schedule and each solver once.
@pytest.mark.parametrize("mode, solver", SCHEDULES)
def test_a_flux_computed_from_the_state_alone_reaches_the_coupled_solution(mode, solver):
    """(a): one evaluation of such a flux is the flux of the iterate.

    The Robin recipe with fluxes that do not read their datum.  Before
    and after the refusal: 1.5e-8 (Gauss-Seidel) and 3.7e-7 (Jacobi).
    """
    gm, nodes = robin_pair(None, mode=mode, solver=solver)
    assert off(gm, exact(nodes, ROBIN)) < CLOSE


# ---------------------------------------------------------------------------
# A chain: a.q is computed from a's state, b.q reads a.q, c.q reads b.q,
# and the updates close a loop of the three nodes, so the group sweeps
# them in add_node order.  CHAIN hands c.q to a's update; SEED_CHAIN
# hands it to c's own.
# ---------------------------------------------------------------------------

CHAIN = [("a", "q", "b", "s", 1.0), ("b", "q", "c", "s", 1.0), ("c", "q", "a", "r", 1.0),
         ("a", "u", "b", "r", 1.0), ("b", "u", "c", "r", 1.0)]
SEED_CHAIN = [("a", "q", "b", "s", 1.0), ("b", "q", "c", "s", 1.0), ("c", "q", "c", "r", 1.0),
              ("c", "u", "a", "r", 1.0), ("a", "u", "b", "r", 1.0)]


def chain_nodes(order, strict=False, cls=Side):
    make = {"a": lambda: cls("a", *coefficients(None, 0)),
            "b": lambda: cls("b", *coefficients(0.8, 1), strict=strict),
            "c": lambda: cls("c", *coefficients(0.6, 2), strict=strict)}
    return [make[name]() for name in order]


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_a_chain_of_fluxes_the_seed_reaches_gives_the_coupled_solution[abc-True-gauss-seidel-ift]
# and its [bac-False-jacobi-fori]: a chain along the sweep and one a step against it,
# under each schedule and each solver once.
@pytest.mark.parametrize("mode, solver", SCHEDULES)
@pytest.mark.parametrize("order, strict", [("abc", True), ("bac", False)])
def test_a_chain_of_fluxes_the_seed_reaches_gives_the_coupled_solution(order, strict, mode, solver):
    """(b), ``S <= 2``: along the sweep order (one sweep, and a hook may
    index its input), and one step against it (two; the hook tolerates
    the first).  Measured 4.7e-8 to 5.8e-8 before and after.
    """
    nodes = chain_nodes(order, strict)
    gm = built(nodes, CHAIN, ["a", "b", "c"], mode=mode, solver=solver)
    assert gm._group_sweep_order(gm._coupling_groups[0]) == list(order)
    assert off(gm, exact(nodes, CHAIN)) < CLOSE


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_gauss_seidel_reads_along_the_sweep_the_flux_it_recomputed[ift]
@pytest.mark.parametrize("solver", ["ift", pytest.param("fori", marks=pytest.mark.slow)])
def test_gauss_seidel_reads_along_the_sweep_the_flux_it_recomputed(solver):
    """Two steps against the sweep order, and the one reader of the last
    flux is swept after its producer: it reads the flux Gauss-Seidel
    recomputed after the producer's update, whose inputs are seeds two
    sweeps reach.  Right before the refusal (1.5e-7), and compiled still;
    the Jacobi pass reads the seed there, and was 6.3e-2 off.
    """
    nodes = chain_nodes("cba")
    gm = built(nodes, CHAIN, ["a", "b", "c"], mode="gauss-seidel", solver=solver)
    assert off(gm, exact(nodes, CHAIN)) < CLOSE


@pytest.mark.parametrize("edges, mode", [(CHAIN, "jacobi"), (SEED_CHAIN, "jacobi"),
                                         (SEED_CHAIN, "gauss-seidel")])
def test_a_chain_whose_seed_the_pass_reads_too_early_is_refused(edges, mode):
    """(b), ``S = 3`` where the seed is read: 6.3e-2 and 4.0e-2 off with
    ``converged=True`` before.  The message gives the chain, the order
    and both ways out, and each way out compiles.
    """
    gm = built(chain_nodes("cba"), edges, ["a", "b", "c"], mode=mode)
    with pytest.raises(ValueError, match="along a chain that runs against the order") as info:
        gm.compile()
    text = str(info.value)
    assert "(['c', 'b', 'a'])" in text and "flux c.q would take 3" in text
    assert ("flux c.q reads input 's', which edge 'b.q->c.s' delivers from flux b.q; "
            "flux b.q reads input 's', which edge 'a.q->b.s' delivers from flux a.q") in text
    assert "write it to a state field" in text and "add_node order" in text
    assert [line for line in gm.validate() if line.startswith("ERROR")] == [f"ERROR: {text}"]
    for way_out in (chain_nodes("abc"), chain_nodes("cba", cls=Carried)):
        built(way_out, edges, ["a", "b", "c"], mode=mode)._refuse_unsolved_group_fluxes()


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_each_way_out_of_a_refused_chain_reaches_the_coupled_solution[carried-jacobi-fori]
@pytest.mark.parametrize("mode, solver", [
    ("jacobi", "fori"), pytest.param("gauss-seidel", "ift", marks=pytest.mark.slow)])
@pytest.mark.parametrize("way", ["carried", pytest.param("added in order", marks=pytest.mark.slow)])
def test_each_way_out_of_a_refused_chain_reaches_the_coupled_solution(way, mode, solver):
    """The flux in a state field (5.8e-8), and the nodes added so the
    chain runs along the sweep (7.3e-8 to 1.2e-7)."""
    nodes = chain_nodes("cba", cls=Carried) if way == "carried" else chain_nodes("abc")
    gm = built(nodes, SEED_CHAIN, ["a", "b", "c"], mode=mode, solver=solver)
    assert off(gm, exact(nodes, SEED_CHAIN)) < CLOSE


# ---------------------------------------------------------------------------
# (c): fluxes that read each other in a loop.
# ---------------------------------------------------------------------------

def _loop_of_one(mode):
    node = Side("a", *coefficients(0.5, 0))
    return built([node], [("a", "q", "a", "s", 1.0)], ["a"], mode=mode), ["a.q"]


def _the_robin_pair(mode):
    return robin_pair(0.5, mode=mode)[0], ["node_a.q", "node_b.q"]


def _fluxes_only(mode, strict):
    nodes = [Side("a", 0.0, 1.0, 1.0, 0.9, 0.0, reads="r", strict=strict),
             Side("b", 0.0, 1.0, 0.0, 1.0, 0.0, reads="r", strict=strict)]
    edges = [("b", "q", "a", "r", 1.0), ("a", "q", "b", "r", 1.0)]
    return built(nodes, edges, ["a", "b"], mode=mode), ["a.q", "b.q"]


def _loop_of_three(mode):
    nodes = [Side(name, *coefficients(0.5, i)) for i, name in enumerate("abc")]
    edges = [("a", "q", "b", "s", 1.0), ("b", "q", "c", "s", 1.0), ("c", "q", "a", "s", 1.0)]
    return built(nodes, edges, ["a", "b", "c"], mode=mode), ["a.q", "b.q", "c.q"]


LOOPS = {"one": _loop_of_one, "the robin pair": _the_robin_pair,
         "fluxes only, indexed": lambda mode: _fluxes_only(mode, True),
         "fluxes only, tolerated": lambda mode: _fluxes_only(mode, False),
         "three": _loop_of_three}


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("loop", sorted(LOOPS))
def test_fluxes_that_read_each_other_in_a_loop_are_refused(loop, mode):
    """One ``ValueError`` from ``compile()`` and from a reader of the graph
    never compiled, the same text as the one ``ERROR`` line of
    ``validate()``, which no longer says the loop is handled.

    Before: the Robin pair 4.3e-4 to 0.77 off with ``converged=True``;
    the pair that exchanges fluxes only stopped after one pass at
    (1.9, 2.71) where the solution is (10, 10), or, with hooks that index
    their input, ``validate()`` itself raised a bare ``KeyError``.
    """
    gm, fluxes = LOOPS[loop](mode)
    members = sorted({flux.split(".")[0] for flux in fluxes})
    with pytest.raises(ValueError, match="depend on each other in a loop") as info:
        gm.compile()
    text = str(info.value)
    assert text.startswith(f"coupling group {members} is not compiled: ")
    for flux in fluxes:
        assert f"flux {flux} reads input" in text and f"delivers from flux {flux}" in text
    for said in ("can stop with converged=True", "write it to a state field",
                 "Taking the nodes out of the group does not help"):
        assert said in text
    assert "assumed" not in text                    # every hook here was traced
    lines = gm.validate()
    assert [line for line in lines if line.startswith("ERROR")] == [f"ERROR: {text}"]
    assert not any("handled by iterative coupling" in line or "iterated with the group" in line
                   for line in lines)
    with pytest.raises(ValueError) as read:
        gm.resolve_boundary_inputs(members[0])
    assert str(read.value) == text
    with pytest.raises(ValueError):
        check_conservation(gm, gm._state, [(members[0], "q", members[-1], "q")])
    with pytest.raises(ValueError):                 # and still, after a refused compile
        gm.step()


def test_dependence_is_structural_and_an_edge_is_not_looked_into():
    """A flux that multiplies its input by zero reads it, and an edge
    that delivers a flux times zero (``alpha=1``) still delivers it:
    both are refused, on the side of refusing, as the criterion says."""
    for gm in (robin_pair(0.0)[0], robin_pair(0.5, alpha=1.0)[0]):
        with pytest.raises(ValueError, match="depend on each other in a loop"):
            gm.compile()


class Untraceable(Side):
    """Its flux hook runs on values and cannot be traced."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": jnp.asarray(float(state["u"]) + float(boundary_inputs.get("s", 0.0)), F)}


def test_a_hook_that_cannot_be_traced_is_taken_to_read_every_input_and_the_message_says_so():
    nodes = [Untraceable("a", *coefficients(None, 0)), Side("b", *coefficients(0.5, 1))]
    gm = built(nodes, [("a", "q", "b", "s", 1.0), ("b", "q", "a", "s", 1.0)], ["a", "b"])
    with pytest.raises(ValueError, match="depend on each other in a loop") as info:
        gm.compile()
    assert "what ['a.q'] read could not be established" in str(info.value)
    assert "that dependence is assumed" in str(info.value)


class TwoFluxes(Side):
    """``held`` is computed from the state and ``read`` from the input ``s``."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"held": 2.0 * state["u"], "read": state["u"] + boundary_inputs.get("s", ZERO)}


class Branching(Side):
    """Its flux ``q`` reads no input by value and is another function of
    the state when ``x`` is handed to it; ``steady`` is not."""

    def boundary_input_spec(self):
        return {name: BoundaryInputSpec(shape=(), dtype=F, default=ZERO) for name in "rsx"}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        q = (3.0 if "x" in boundary_inputs else 5.0) * state["u"]
        return {"q": q, "steady": 2.0 * state["u"]}


def _reads(node, handed, absent=()):
    gm = GraphManager()
    gm.add_node(node)
    return _flux_loops._flux_reads(gm._nodes[node.name], gm._state[node.name], {}, handed, absent)


def test_each_flux_of_a_node_is_asked_what_it_reads():
    """By value, one flux at a time; and by which inputs the first sweep
    leaves out, for a hook that branches on what it is handed."""
    assert _reads(TwoFluxes("n", *coefficients(0.5, 0)), ["r", "s"]) == {
        "held": set(), "read": {"s"}}
    assert _reads(Side("n", *coefficients(0.5, 0)), ["r", "s"], ["s"]) == {"q": {"s"}}
    assert _reads(Side("n", *coefficients(None, 0)), ["r", "s"], ["r", "s"]) == {"q": set()}
    branching = Branching("n", *coefficients(None, 0))
    assert _reads(branching, ["r", "x"]) == {"q": set(), "steady": set()}
    assert _reads(branching, ["r", "x"], ["x"]) == {"q": {"x"}, "steady": set()}
    # An input the node does not declare cannot be traced: nothing is established.
    assert _reads(branching, ["r", "undeclared"]) is None


def test_a_flux_that_changes_with_the_inputs_it_is_handed_is_not_a_seed_of_one_sweep():
    """``b.q`` reads nothing by value, but the first sweep hands ``b`` its
    inputs without ``x`` (its own flux ``steady``), so its first value is
    not its value; ``a.q`` reads it a step against the sweep, and a
    Jacobi pass reads ``a.q`` from the seed: three sweeps, refused.
    Without the second trace this passes as a chain of two.
    """
    nodes = [Side("a", *coefficients(0.5, 0)), Branching("b", *coefficients(None, 1))]
    edges = [("b", "q", "a", "s", 1.0), ("b", "steady", "b", "x", 1.0),
             ("a", "q", "b", "r", 1.0), ("b", "u", "a", "r", 1.0)]
    gm = built(nodes, edges, ["a", "b"], mode="jacobi")
    with pytest.raises(ValueError, match="flux a.q would take 3"):
        gm.compile()
    # Gauss-Seidel hands a.q to b along the sweep, recomputed from the seed of b.q.
    built([Side("a", *coefficients(0.5, 0)), Branching("b", *coefficients(None, 1))],
          edges, ["a", "b"])._refuse_unsolved_group_fluxes()


def test_no_hook_is_traced_for_a_group_its_edges_alone_pass(monkeypatch):
    """A flux edge to a node that produces none, a chain the seed reaches
    whose last flux no producer reads, and a graph without groups: the
    criterion returns on the edges."""
    asked = []
    monkeypatch.setattr(_flux_loops, "_flux_reads", lambda *args: asked.append(args[0]))
    plain = type("Plain", (Side,), {
        "compute_boundary_fluxes": SimulationNode.compute_boundary_fluxes})
    open_chain = [("a", "q", "b", "s", 1.0), ("b", "q", "c", "s", 1.0),
                  ("c", "u", "a", "r", 1.0), ("a", "u", "b", "r", 1.0)]
    for gm in (
            built([Side("a", *coefficients(None, 0)), plain("b", *coefficients(None, 1))],
                  [("a", "q", "b", "r", 1.0), ("b", "u", "a", "r", 1.0)], ["a", "b"]),
            built(chain_nodes("abc"), open_chain, ["a", "b", "c"], mode="jacobi"),
            built(chain_nodes("bac"), open_chain, ["a", "b", "c"], mode="jacobi"),
            built(chain_nodes("cba"), CHAIN, None)):
        gm._refuse_unsolved_group_fluxes()
        assert not any(line.startswith("ERROR") for line in gm.validate())
    assert asked == []
    # The patch is read: a group its edges alone would refuse asks each
    # member, and one that answers nothing is taken to read every input.
    with pytest.raises(ValueError, match="that dependence is assumed"):
        robin_pair(None)[0]._refuse_unsolved_group_fluxes()
    assert len(asked) == 2


# ---------------------------------------------------------------------------
# The way out the message gives, on the pair that was wrong.
# ---------------------------------------------------------------------------

# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_the_flux_carried_in_a_state_field_reaches_the_coupled_solution[1.5-gauss-seidel-ift]
# and its [1.5-jacobi-fori]: the strongest loop (gain 0.56), each schedule and each solver once.
@pytest.mark.parametrize("mode, solver", SCHEDULES)
@pytest.mark.parametrize("kq", [1.5, pytest.param(0.5, marks=pytest.mark.slow),
                                pytest.param(1.0, marks=pytest.mark.slow)])
def test_the_flux_carried_in_a_state_field_reaches_the_coupled_solution(kq, mode, solver):
    """The same pair with ``q`` written by ``update`` and read as a state
    field: the group iterates it.  8.7e-8 to 3.4e-7 off the 6x6 solution
    where the flux edges were 4.3e-4 (kq 0.5) to 0.77 (kq 1.5) off.
    """
    gm, nodes = robin_pair(kq, cls=Carried, mode=mode, solver=solver)
    assert off(gm, exact(nodes, ROBIN)) < CLOSE


def test_outside_a_group_the_loop_cannot_be_read():
    """What the message says of taking the nodes out of the group
    (MADD-ANO-157, open): the step raises on the flux edge it would read
    from the previous step."""
    nodes = [Side("node_a", *coefficients(0.5, 0), reads="r"),
             Side("node_b", *coefficients(0.5, 1), reads="r")]
    gm = built(nodes, ROBIN, None)
    gm.compile()
    with pytest.raises(KeyError, match="q"):
        gm.step()


# ---------------------------------------------------------------------------
# A hook that indexes its inputs.
# ---------------------------------------------------------------------------

def test_a_hook_that_indexes_its_input_is_asked_for_its_fluxes_with_the_declared_inputs():
    """``validate()`` asks each flux hook for its keys with no inputs; one
    that indexes an input raised a bare ``KeyError`` there, for every
    edge from its flux.  Outside a group the edge is read forward."""
    nodes = [Side("a", *coefficients(None, 0)), Side("b", *coefficients(0.8, 1), strict=True),
             Side("c", *coefficients(None, 2))]
    edges = [("a", "u", "b", "s", 1.0), ("b", "q", "c", "r", 1.0)]
    gm = built(nodes, edges, None)
    assert not any(line.startswith(("ERROR", "WARNING")) for line in gm.validate())
    assert off(gm, exact(nodes, edges)) < CLOSE


@pytest.mark.parametrize("mode, solver", PER_PUSH)
def test_a_hook_that_indexes_an_input_the_first_sweep_leaves_out_is_told_so(mode, solver):
    """One step against the sweep order is right for a hook that
    tolerates the first sweep; one that indexes the input used to raise a
    bare ``KeyError`` naming it."""
    gm = built(chain_nodes("bac", strict=True), CHAIN, ["a", "b", "c"], mode=mode, solver=solver)
    gm.compile()
    with pytest.raises(ValueError, match=r"compute_boundary_fluxes of 'b' indexes "
                                         r"boundary_inputs\['s'\]") as info:
        gm.step()
    assert "edge 'a.q->b.s'" in str(info.value) and "boundary_inputs.get('s'" in str(info.value)
