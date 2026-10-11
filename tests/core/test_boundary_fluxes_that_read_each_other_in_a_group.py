"""Boundary fluxes that read each other inside a coupling group.

A boundary flux is not part of a group's iterate: every pass rebuilds the
fluxes from nothing in two sweeps (``coupling/_flux_loops.py`` has the
argument).  That is the flux of the iterate for a flux computed from the
state, and for fluxes that read each other along a chain the two sweeps
reach; it is not a solution of fluxes that read each other in a loop.  A
group with such a loop used to stop with ``converged=True`` at a state
that does not solve the coupled equations, and ``validate()`` said the
loop was "handled by iterative coupling".  It is refused at ``compile()``.

One measure of error throughout (:func:`off`): the largest distance of a
member's state from its value in the coupled solution, as a fraction of
that value.  On the affine pair below the refused loop was 4.3e-4 to 0.77
off in it.

Every case here is affine, so the coupled equations are one linear
system, solved in float64 (:func:`exact`): each structure the passes are
right for is held to it under both schedules and both solvers, and each
structure they are not right for is refused.  Which is which was measured
on the tree before the refusal (the comments give the numbers).
"""

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _flux_loops
from maddening.core.coupling.helpers import add_robin_coupling, check_conservation
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryFluxSpec, BoundaryInputSpec, SimulationNode

F = jnp.float32
ZERO = jnp.zeros((), F)
# Where a pass is right the states are within 1.0e-6 of their coupled
# values, as a fraction of those values (measured at tolerance 1e-6; the
# largest is the carried flux at loop gain 0.56 under Jacobi); the wrong
# ones were 4.3e-4 off or more.
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

    drives = "r"

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
        return {"u": self.k[0] + self.k[1] * boundary_inputs.get(self.drives, ZERO)}

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
            if name == node.drives:
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
    """The largest distance of a stepped state from its value in
    *reference*, as a fraction of that value."""
    gm.compile()
    gm.step()
    return max(abs(float(gm.get_node_state(name)["u"]) - value) / abs(value)
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
    and after the refusal: 1.7e-8 (Gauss-Seidel) and 3.8e-7 (Jacobi).
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


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_a_chain_of_fluxes_the_seed_reaches_gives_the_coupled_solution[abc-gauss-seidel-ift]
# and its [bac-jacobi-fori]: a chain along the sweep and one a step against it,
# under each schedule and each solver once.
@pytest.mark.parametrize("mode, solver", SCHEDULES)
@pytest.mark.parametrize("order", ["abc", "bac"])
def test_a_chain_of_fluxes_the_seed_reaches_gives_the_coupled_solution(order, mode, solver):
    """(b), ``S <= 2``: along the sweep order (one sweep), and one step
    against it (two; the hook is handed its inputs without the flux in
    the first).  Measured 1.8e-7 to 3.2e-7 before and after.
    """
    nodes = chain_nodes(order)
    gm = built(nodes, CHAIN, ["a", "b", "c"], mode=mode, solver=solver)
    assert gm._group_sweep_order(gm._coupling_groups[0]) == list(order)
    assert off(gm, exact(nodes, CHAIN)) < CLOSE
    assert list(gm._schedule) == list(order)        # asked before the compile, and the same


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_gauss_seidel_reads_along_the_sweep_the_flux_it_recomputed[ift]
@pytest.mark.parametrize("solver", ["ift", pytest.param("fori", marks=pytest.mark.slow)])
def test_gauss_seidel_reads_along_the_sweep_the_flux_it_recomputed(solver):
    """Two steps against the sweep order, and the one reader of the last
    flux is swept after its producer: it reads the flux Gauss-Seidel
    recomputed after the producer's update, whose inputs are seeds two
    sweeps reach.  Right before the refusal (3.8e-7), and compiled still;
    the Jacobi pass reads the seed there, and was 0.15 off.
    """
    nodes = chain_nodes("cba")
    gm = built(nodes, CHAIN, ["a", "b", "c"], mode="gauss-seidel", solver=solver)
    assert off(gm, exact(nodes, CHAIN)) < CLOSE


@pytest.mark.parametrize("edges, mode", [(CHAIN, "jacobi"), (SEED_CHAIN, "jacobi"),
                                         (SEED_CHAIN, "gauss-seidel")])
def test_a_chain_whose_seed_the_pass_reads_too_early_is_refused(edges, mode):
    """(b), ``S = 3`` where the seed is read: 0.15 and 6.3e-2 off with
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
    assert "write it to a state field" in text
    assert "in the order of the reads, ['a', 'b', 'c']: a group sweeps" in text
    assert [line for line in gm.validate() if line.startswith("ERROR")] == [f"ERROR: {text}"]
    for way_out in (chain_nodes("abc"), chain_nodes("cba", cls=Carried)):
        built(way_out, edges, ["a", "b", "c"], mode=mode)._refuse_unsolved_group_fluxes()


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_each_way_out_of_a_refused_chain_reaches_the_coupled_solution[carried-jacobi-fori]
@pytest.mark.parametrize("mode, solver", [
    ("jacobi", "fori"), pytest.param("gauss-seidel", "ift", marks=pytest.mark.slow)])
@pytest.mark.parametrize("way", ["carried", pytest.param("added in order", marks=pytest.mark.slow)])
def test_each_way_out_of_a_refused_chain_reaches_the_coupled_solution(way, mode, solver):
    """The flux in a state field (6.7e-8), and the nodes added so the
    chain runs along the sweep (3.7e-7 Gauss-Seidel, 2.3e-7 Jacobi)."""
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
    errors = [line for line in lines if line.startswith("ERROR")]
    # A hook that indexes its input has a line of its own (see below).
    assert [line for line in errors if "raised KeyError" not in line] == [f"ERROR: {text}"]
    assert len(errors) == (1 + len(members) if "indexed" in loop else 1)
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
    text = str(info.value)
    assert "What the fluxes of 'a' read is assumed, not established: for 'a', its " \
        "compute_boundary_fluxes could not be traced on the inputs the pass hands it (" in text
    assert "is never refused" not in text and "once that can be established" in text


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


class Fallback(Side):
    """Its flux reads ``s`` only when ``x`` is not handed to it."""

    def boundary_input_spec(self):
        return {name: BoundaryInputSpec(shape=(), dtype=F, default=ZERO) for name in "rsx"}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        if "x" in boundary_inputs:
            return {"q": 2.0 * state["u"]}
        return {"q": state["u"] + boundary_inputs.get("s", ZERO)}


class Switched(Branching):
    """A :class:`Branching` flux computed inside ``lax.cond``: the two
    traces differ only within the branch."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        gain = 3.0 if "x" in boundary_inputs else 5.0
        return {"q": jax.lax.cond(state["u"] > 0, lambda u: gain * u, lambda u: u, state["u"])}


def _reads(node, handed, absent=()):
    gm = GraphManager()
    gm.add_node(node)
    return _flux_loops._flux_reads(gm._nodes[node.name], gm._state[node.name], {},
                                   {name: ZERO for name in handed}, absent)


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
    # Traced on the inputs the graph hands it, not on every input it declares.
    assert _reads(Fallback("n", *coefficients(None, 0)), ["r", "s"]) == {"q": {"s"}}
    assert _reads(Fallback("n", *coefficients(None, 0)), ["r", "s", "x"]) == {"q": set()}
    # Declared or not: the hook is traced on what it is handed.
    assert _reads(branching, ["r", "undeclared"]) == {"q": set(), "steady": set()}
    # An equation that holds a sub-program (control flow) is the same as no other.
    assert _reads(Switched("n", *coefficients(None, 0)), ["r", "x"], ["x"]) == {"q": {"x"}}


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
    asked, real = [], _flux_loops._flux_reads

    def counting(spec, *args):
        asked.append(spec.node.name)
        return real(spec, *args)

    monkeypatch.setattr(_flux_loops, "_flux_reads", counting)
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
    # member, once in a compile (validate() does not ask again inside it).
    gm = robin_pair(None)[0]
    gm.compile()
    assert asked == ["node_a", "node_b"]
    gm.validate()
    assert asked == ["node_a", "node_b"] * 2


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
    field: the group iterates it.  8.7e-8 (kq 0.5 and 1.0), 7.0e-7 (kq
    1.5, Gauss-Seidel) and 1.0e-6 (kq 1.5, Jacobi) off the 6x6 solution,
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
# A hook that indexes an input.
# ---------------------------------------------------------------------------

class Counting(Side):
    """A :class:`Side` whose flux hook indexes ``s`` and counts its calls."""

    calls = 0

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        type(self).calls += 1
        return {"q": state["u"] + boundary_inputs["s"]}


@pytest.mark.parametrize("group", [None, ["b", "c"]])
@pytest.mark.parametrize("edges", [
    [("b", "q", "c", "r", 1.0), ("c", "u", "b", "r", 1.0)],
    [("a", "u", "b", "s", 1.0), ("b", "q", "c", "r", 1.0), ("b", "q", "c", "s", 1.0),
     ("c", "u", "b", "r", 1.0)]], ids=["input not delivered", "input delivered"])
def test_a_flux_hook_that_indexes_an_input_is_refused_by_name_and_asked_once(edges, group):
    """``validate()`` asks each flux hook for its flux names with no
    inputs, once, however many edges read its fluxes.  A hook that
    indexes an input raised a bare ``KeyError`` there; it is an ``ERROR``
    line that names the node, the hook and the input, and ``compile()``
    raises it.  (Asked a second time with the declared inputs, such a
    hook passed and failed at the first step instead.)
    """
    Counting.calls = 0
    nodes = [Side("a", *coefficients(None, 0)), Counting("b", *coefficients(0.8, 1)),
             Side("c", *coefficients(None, 2))]
    gm = built(nodes, edges, group)
    errors = [line for line in gm.validate() if line.startswith("ERROR")]
    assert Counting.calls == 1
    assert len(errors) == 1 and errors[0].startswith(
        "ERROR: compute_boundary_fluxes of node 'b' raised KeyError('s') when it was asked "
        "for its flux names with no boundary inputs: it indexes boundary_inputs['s'].")
    assert "boundary_inputs.get('s'," in errors[0]
    with pytest.raises(RuntimeError, match="compute_boundary_fluxes of node 'b' raised KeyError"):
        gm.compile()


# ---------------------------------------------------------------------------
# Inputs the node does not declare.  ``boundary_input_spec()`` is empty by
# default and a node may read an input it does not declare, so the hook is
# traced on what the edges deliver, declared or not.
# ---------------------------------------------------------------------------

class Bare(Side):
    """A :class:`Side` that declares no input (the default)."""

    boundary_input_spec = SimulationNode.boundary_input_spec


class Guide(Bare):
    """The node of the developer guide's Robin recipe: it is handed ``robin_in``."""

    drives = "robin_in"


class GuideDeclared(Guide):
    def boundary_input_spec(self):
        return {"robin_in": BoundaryInputSpec(shape=(), dtype=F, default=ZERO)}


GUIDE = [(source, field, target, "robin_in", gain) for source, field, target, _r, gain in ROBIN]


def guide_pair(cls_a, cls_b, kq=None, bias=False):
    """The guide's recipe, verbatim, on two nodes named as it names them."""
    nodes = [cls_a("node_a", *coefficients(kq, 0), reads="robin_in"),
             cls_b("node_b", *coefficients(kq, 1), reads="robin_in")]
    gm = GraphManager()
    for node in nodes:
        gm.add_node(node)
    add_robin_coupling(
        gm, "node_a", "node_b",
        value_field_a="u", flux_field_a="q",
        value_field_b="u", flux_field_b="q",
        input_a="robin_in", input_b="robin_in",
        alpha=0.5,  # mixing: 0=pure Neumann, 1=pure Dirichlet
    )
    edges = list(GUIDE)
    if bias:    # a state edge into an input nobody declares, and a third member
        nodes.append(Carried("src", 0.8, 0.0, 0.0, None, 0.0))
        gm.add_node(nodes[-1])
        gm.add_edge(source="src", target="node_a", source_field="u", target_field="bias")
        gm.add_edge(source="node_a", target="src", source_field="u", target_field="r")
        edges += [("src", "u", "node_a", "bias", 1.0), ("node_a", "u", "src", "r", 1.0)]
    return gm, nodes, edges


VARIANTS = {"V1 neither declares": (Guide, Guide, False),
            "V2 both declare": (GuideDeclared, GuideDeclared, False),
            "V3 one declares": (Guide, GuideDeclared, False),
            "V4 a state edge into an undeclared input": (GuideDeclared, GuideDeclared, True)}


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_a_flux_from_the_state_is_right_whether_or_not_its_node_declares_the_input(variant, mode):
    """The Robin recipe with fluxes computed from the state: 1.7e-8
    (Gauss-Seidel) and 3.8e-7 (Jacobi) off, the same with the input
    declared, undeclared, or beside an undeclared input a state edge
    feeds.  (While dependence on an undeclared input was assumed, V1, V3
    and V4 were refused.)
    """
    cls_a, cls_b, bias = VARIANTS[variant]
    gm, nodes, edges = guide_pair(cls_a, cls_b, bias=bias)
    grouped(gm, [node.name for node in nodes], mode=mode)
    assert off(gm, {k: v for k, v in exact(nodes, edges).items() if k != "src"}) < CLOSE


def test_the_guides_robin_recipe_compiles_as_written():
    """Verbatim, group and all, on nodes that override nothing but what
    the recipe names: the default ``boundary_input_spec()``."""
    gm, nodes, edges = guide_pair(Guide, Guide)
    gm.add_coupling_group(["node_a", "node_b"])
    assert not any(line.startswith(("ERROR", "WARNING")) for line in gm.validate())
    assert off(gm, exact(nodes, edges)) < 1e-4      # the group's default tolerance


def test_an_undeclared_input_a_flux_does_read_in_a_loop_is_refused_as_established():
    gm, _nodes, _edges = guide_pair(Guide, Guide, kq=0.5)
    grouped(gm, ["node_a", "node_b"])
    with pytest.raises(ValueError, match="depend on each other in a loop") as info:
        gm.compile()
    assert "flux node_a.q reads input 'robin_in'" in str(info.value)
    assert "assumed" not in str(info.value) and "is never refused" in str(info.value)


class Wide(GuideDeclared):
    """V5, a control: ``profile`` is declared at the default shape ``()``
    and handed three values; the flux reads ``profile[0]``, which only
    traces at the shape that is handed."""

    def boundary_input_spec(self):
        return {"robin_in": BoundaryInputSpec(shape=(), dtype=F, default=ZERO),
                "profile": BoundaryInputSpec()}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        profile = boundary_inputs.get("profile", jnp.zeros(3, F))
        return {"q": self.flux(state["u"], boundary_inputs) + 0.0 * profile[0]}


class Three(SimulationNode):
    def initial_state(self):
        return {"v": jnp.asarray([0.1, 0.2, 0.3], F)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"v": state["v"]}


def test_a_hook_is_traced_at_the_shape_it_is_handed_not_the_declared_one():
    """The flux refusal has nothing to say (the fluxes read a state
    field's value); what stops the graph is what always stopped it, the
    declared shape against the delivered one."""
    gm, _nodes, _edges = guide_pair(Wide, Wide)
    gm.add_node(Three("vec", 1.0))
    for target in ("node_a", "node_b"):
        gm.add_edge(source="vec", target=target, source_field="v", target_field="profile")
    grouped(gm, ["node_a", "node_b"])
    gm._refuse_unsolved_group_fluxes()
    assert not any("is not compiled" in line for line in gm.validate())
    with pytest.raises(Exception, match="edge validation failed"):
        gm.compile()


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("order", ["abc", "acb", "bac", "bca", "cab", "cba"])
def test_a_chain_with_undeclared_inputs_is_judged_as_one_with_declared_inputs(order, mode):
    """Every sweep order of the chain, nothing declared.  Only two steps
    against the sweep under Jacobi reads a seed the two sweeps do not
    reach (0.15 off before); every other order is right (1.8e-7 to
    3.8e-7) and compiles."""
    nodes = chain_nodes(order, cls=Bare)
    gm = built(nodes, CHAIN, ["a", "b", "c"], mode=mode)
    if (order, mode) == ("cba", "jacobi"):
        with pytest.raises(ValueError, match="flux c.q would take 3") as info:
            gm.compile()
        assert "assumed" not in str(info.value)
    else:
        assert off(gm, exact(nodes, CHAIN)) < CLOSE


# ---------------------------------------------------------------------------
# What the message offers, and what the trace is handed.
# ---------------------------------------------------------------------------

class Several(SimulationNode):
    """Fluxes ``{name: the input it reads, or None}``, each the state plus
    half that input.  Nothing declared."""

    def __init__(self, name, fluxes):
        super().__init__(name=name, timestep=1.0)
        self.fluxes = fluxes

    def initial_state(self):
        return {"u": ZERO}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"u": 0.5 + 0.25 * boundary_inputs.get("r", ZERO)}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {flux: state["u"] + (0.5 * boundary_inputs.get(reads, ZERO) if reads else 0.0)
                for flux, reads in self.fluxes.items()}


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("order", ["AB", "BA"])
def test_a_chain_that_bounces_between_two_members_is_not_offered_an_order(order, mode):
    """``A.f1 <- B.g1 <- A.f2 <- B.g2 <- A.f3``, and ``f1`` read by ``A``'s
    update: refused whichever node is added first, so the message offers
    the state field and no order of the nodes."""
    make = {"A": lambda: Several("A", {"f3": None, "f2": "x2", "f1": "x1"}),
            "B": lambda: Several("B", {"g2": "y2", "g1": "y1"})}
    edges = [("A", "f3", "B", "y2", 1.0), ("B", "g2", "A", "x2", 1.0), ("A", "f2", "B", "y1", 1.0),
             ("B", "g1", "A", "x1", 1.0), ("A", "f1", "A", "r", 1.0)]
    gm = built([make[name]() for name in order], edges, ["A", "B"], mode=mode)
    with pytest.raises(ValueError, match="along a chain that runs against the order") as info:
        gm.compile()
    assert "write it to a state field" in str(info.value)
    assert "add the nodes" not in str(info.value)


class Gated(Side):
    """Its flux reads ``s`` only when it is also handed ``x``."""

    def boundary_input_spec(self):
        return {name: BoundaryInputSpec(shape=(), dtype=F, default=ZERO) for name in "rsx"}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        q = self.k[2] + self.k[4] * state["u"]
        if "x" in boundary_inputs:
            q = q + self.k[3] * boundary_inputs.get("s", ZERO)
        return {"q": q}


@pytest.mark.parametrize("handed", ["nothing", "an external input", "an edge from outside the group"])
def test_the_hook_is_traced_with_every_input_the_pass_hands_it(handed):
    """An external input and an edge from a node outside the group are
    inputs of the hook like any other: with ``x`` handed the two fluxes
    read each other, and the group is refused; without it they do not."""
    nodes = [Gated("a", *coefficients(0.5, 0)), Side("b", *coefficients(0.5, 1))]
    edges = [("a", "q", "b", "s", 1.0), ("b", "q", "a", "s", 1.0),
             ("a", "u", "b", "r", 1.0), ("b", "u", "a", "r", 1.0)]
    if handed == "an edge from outside the group":
        nodes.append(Carried("o", 0.8, 0.0, 0.0, None, 0.0))
        edges.append(("o", "u", "a", "x", 1.0))
    gm = built(nodes, edges, ["a", "b"])
    if handed == "an external input":
        gm.add_external_input("a", "x", shape=(), dtype=F)
    if handed == "nothing":
        gm._refuse_unsolved_group_fluxes()
    else:
        with pytest.raises(ValueError, match="depend on each other in a loop"):
            gm.compile()


def test_every_group_of_the_graph_is_asked():
    """A loop in the second group added is refused like one in the first."""
    nodes = [Side(name, *coefficients(None if name in "ab" else 0.5, i % 3))
             for i, name in enumerate("abcd")]
    edges = [("a", "u", "b", "r", 1.0), ("b", "u", "a", "r", 1.0),
             ("c", "q", "d", "s", 1.0), ("d", "q", "c", "s", 1.0)]
    gm = built(nodes, edges, ["a", "b"])
    grouped(gm, ["c", "d"])
    with pytest.raises(ValueError, match=r"coupling group \['c', 'd'\] is not compiled"):
        gm.compile()
    assert sum(line.startswith("ERROR: coupling group ['c', 'd']") for line in gm.validate()) == 1


# ---------------------------------------------------------------------------
# What a pass hands a hook, composed as the pass composes it.  The Robin
# pair once more, each node handed ``x``: the other's value (a scalar state
# field) and the other's flux, of another shape, by two edges into the same
# input, the second additive, so the pass hands their sum broadcast to (3,).
# The flux hook reads ``x`` by a rule on its shape (a Python branch: shapes
# are static), so whether the two fluxes read each other in a loop is
# decided by the shape the pass hands: read where ``x`` is a vector, a loop
# (refused); read only where it is a scalar, no loop (compiles, and the
# pass is right).
#
# N  the flux is (3,); T  the flux is a scalar, its edge's transform makes
# it (3,); O  a node outside the group first hands ``a.x`` a scalar, and
# both edges of the pair add to it.  Each with the flux named "q" (before
# the value "u" in any sort by field name) and "w" (after it), and the
# value edge or the flux edge declared first.  Before the composition
# followed the pass, the derivation walked the edges sorted by source
# field: "q" was derived as a scalar, accepted where the hook reads a
# vector (6.1e-3 off under Gauss-Seidel, 9.96e-3 under Jacobi, O 1.3e-2
# and 2.3e-2), and refused where it reads a scalar although the pass was
# right (1.4e-8 and 3.7e-7).
# ---------------------------------------------------------------------------

class Shaped(SimulationNode):
    """``u <- cu + ku * mean(x)``; flux ``cq + mq * u``, plus ``mean(x)``
    where ``x`` has *ndim* dimensions, of shape *shape*, named *flux*."""

    def __init__(self, name, i, *, ndim, shape=(), flux="q"):
        super().__init__(name=name, timestep=1.0)
        cu, ku, cq, _kq, mq = coefficients(None, i)
        self.k = tuple(np.float32(v) for v in (cu, ku, cq, mq))
        self.ndim, self.shape, self.flux = ndim, shape, flux

    def initial_state(self):
        return {"u": ZERO}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"u": self.k[0] + self.k[1] * jnp.mean(boundary_inputs.get("x", ZERO))}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        q = self.k[2] + self.k[3] * state["u"]
        x = boundary_inputs.get("x")
        if x is not None and jnp.ndim(x) == self.ndim:
            q = q + jnp.mean(x)
        return {self.flux: q * jnp.ones(self.shape, F)}


class Constant(SimulationNode):
    """A node outside the group whose ``v`` never changes."""

    def __init__(self, name, value):
        super().__init__(name=name, timestep=1.0)
        self.value = value

    def initial_state(self):
        return {"v": jnp.asarray(self.value, F)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"v": state["v"]}


HALF = staticmethod(lambda v: np.float32(0.5) * v)
OUTSIDE = 0.2
CELLS = [("gauss-seidel", "ift"), ("jacobi", "fori"), ("gauss-seidel", "fori"), ("jacobi", "ift")]
COMBOS = [(case, flux, order) for case in "NTO" for flux in ("q", "w")
          for order in ("value first", "flux first")]


def handed_pair(case, ndim, flux, order, mode="gauss-seidel", solver="ift", replaced=False):
    """The pair of the section above; *replaced*: the second edge into
    each input is not additive, so it replaces what the first delivered."""
    shape = () if case == "T" else (3,)
    nodes = [Shaped("a", 0, ndim=ndim, shape=shape, flux=flux),
             Shaped("b", 1, ndim=ndim, shape=shape, flux=flux)]
    gm = GraphManager()
    for node in nodes:
        gm.add_node(node)
    if case == "O":
        gm.add_node(Constant("o", OUTSIDE))
        gm.add_edge(source="o", target="a", source_field="v", target_field="x")
    widen = (lambda v: np.float32(0.5) * v * jnp.ones(3, F)) if case == "T" else HALF.__func__
    for source, target in (("b", "a"), ("a", "b")):
        pair = [("u", HALF.__func__), (flux, widen)]
        if order == "flux first":
            pair.reverse()
        for k, (field, transform) in enumerate(pair):
            gm.add_edge(source=source, target=target, source_field=field, target_field="x",
                        transform=transform,
                        additive=not replaced and (k > 0 or (case == "O" and target == "a")))
    return grouped(gm, ["a", "b"], mode=mode, solver=solver), nodes


def handed_exact(nodes, live, bias=0.0, flux_delivered=True):
    """The coupled solution of :func:`handed_pair`, in float64: ``u`` and
    the flux's value ``q`` of each node (the mean of a constant vector is
    its value).  *live*: the flux reads ``x``."""
    (cua, kua, cqa, mqa), (cub, kub, cqb, mqb) = ([float(v) for v in n.k] for n in nodes)
    kq, g = (1.0 if live else 0.0), (0.5 if flux_delivered else 0.0)
    A = np.array([[1, -0.5 * kua, 0, -g * kua], [-0.5 * kub, 1, -g * kub, 0],
                  [-mqa, -0.5 * kq, 1, -g * kq], [-0.5 * kq, -mqb, -g * kq, 1]])
    z = np.linalg.solve(A, np.array([cua + kua * bias, cub, cqa + kq * bias, cqb]))
    return {"a": z[0], "b": z[1]}


def without_the_refusal(monkeypatch, build):
    """*build()* compiled with the criterion taken out: the program the
    step runs is the same (the criterion changes nothing in it), so this
    is what the tree computed before the refusal."""
    with monkeypatch.context() as patch:
        patch.setattr(_flux_loops, "_refuse_unsolved_fluxes", lambda *args, **kwargs: None)
        gm = build()
        gm.compile()
    gm.step()
    return gm


def numbers(gm, names=("a", "b")):
    """Every state value's bits and every diagnostics entry, exactly."""
    return ({name: np.asarray(gm.get_node_state(name)["u"]).tobytes() for name in names},
            {key: repr(dict(value)) for key, value in gm.coupling_diagnostics().items()})


@pytest.mark.parametrize("mode, solver", CELLS)
@pytest.mark.parametrize("case, flux, order", COMBOS)
def test_a_loop_through_the_shape_the_pass_hands_is_refused(case, flux, order, mode, solver):
    """The hook reads ``x`` where it is a vector, and the pass hands a
    vector: the fluxes read each other in a loop, established."""
    gm, _nodes = handed_pair(case, 1, flux, order, mode, solver)
    with pytest.raises(ValueError, match="depend on each other in a loop") as info:
        gm.compile()
    text = str(info.value)
    assert "assumed" not in text and "is never refused" in text
    assert f"which edge 'b.{flux}->a.x' delivers from flux b.{flux}" in text


def _cells_of_each_combo():
    params = []
    for i, combo in enumerate(COMBOS):
        for j, cell in enumerate(CELLS):
            marks = () if j == i % 2 else (pytest.mark.slow,)
            params.append(pytest.param(*combo, *cell, marks=marks))
    return params


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_a_shape_the_hook_does_not_read_compiles_with_the_numbers_of_the_pass[N-q-value first-gauss-seidel-ift]
# and each other combination once, under one of the two per-push cells in turn.
@pytest.mark.parametrize("case, flux, order, mode, solver", _cells_of_each_combo())
def test_a_shape_the_hook_does_not_read_compiles_with_the_numbers_of_the_pass(
        case, flux, order, mode, solver, monkeypatch):
    """The hook reads ``x`` only where it is a scalar, and the pass hands
    a vector: no flux reads another, the group compiles, and its state
    and diagnostics are those of the pass, to the bit, at the coupled
    solution."""
    gm, nodes = handed_pair(case, 0, flux, order, mode, solver)
    assert off(gm, handed_exact(nodes, False, OUTSIDE if case == "O" else 0.0)) < CLOSE
    before = without_the_refusal(
        monkeypatch, lambda: handed_pair(case, 0, flux, order, mode, solver)[0])
    assert numbers(gm) == numbers(before)


# ---------------------------------------------------------------------------
# A flux whose shape follows an input that a member swept later delivers
# (``a.v``, (3,) once ``b.w`` is known, a scalar in the first sweep), read
# by ``b.g`` by a rule on its shape; ``a.q`` reads ``b.g`` and is read by
# ``a``'s own update, from the seed.  "S": ``a.v`` reads ``z`` by value
# (times zero) and ``b.g`` reads ``y`` where it is a vector: ``a.q`` needs
# three sweeps.  "ones": ``a.v`` takes only its shape from ``z``.  "S2":
# ``b.g`` reads ``y`` only where it is a scalar, which the first sweep
# hands it: its first value is not its settled one, so ``a.q``, which reads
# that value from the seed, needs three sweeps too.  All three were
# accepted with ``converged=True`` and are off the coupled solution (S
# 0.103 under both schedules); refused, each as a chain the seed does not
# reach.
# ---------------------------------------------------------------------------

class Follows(SimulationNode):
    """``a``: ``v`` follows ``z`` (*by*: ``"value"`` or ``"shape"``), ``q``
    reads ``p``; ``u <- 1 + 0.4 mean(r)``."""

    def __init__(self, names, by):
        super().__init__(name="a", timestep=1.0)
        self.n, self.by = names, by

    def initial_state(self):
        return {"u": ZERO}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"u": 1.0 + 0.4 * jnp.mean(boundary_inputs.get("r", ZERO))}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        u, z = state["u"], boundary_inputs.get("z", ZERO)
        v = 0.3 - 0.6 * u + 0.0 * z if self.by == "value" else (0.3 - 0.6 * u) * jnp.ones_like(z)
        return {self.n["v"]: v,
                self.n["q"]: 0.1 + 0.5 * u + 0.7 * jnp.mean(boundary_inputs.get("p", ZERO))}


class Reader(SimulationNode):
    """``b``: ``w`` a constant (3,) vector of its state, ``g`` reads ``y``
    where it has *ndim* dimensions; ``u <- -0.5 + 0.3 mean(r)``."""

    def __init__(self, names, ndim):
        super().__init__(name="b", timestep=1.0)
        self.n, self.ndim = names, ndim

    def initial_state(self):
        return {"u": ZERO}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"u": -0.5 + 0.3 * jnp.mean(boundary_inputs.get("r", ZERO))}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        u, y = state["u"], boundary_inputs.get("y")
        g = 0.2 + 0.3 * u
        if y is not None and jnp.ndim(y) == self.ndim:
            g = g + 0.8 * jnp.mean(y)
        return {self.n["w"]: (0.2 - 0.4 * u) * jnp.ones(3, F), self.n["g"]: g}


FOLLOWS = {"S": ("value", 1), "ones": ("shape", 1), "S2": ("value", 0)}
NAMINGS = [dict(v="v", q="q", w="w", g="g"), dict(v="q", q="v", w="g", g="w")]


def follows_graph(variant, names, reverse, mode="gauss-seidel", solver="ift"):
    by, ndim = FOLLOWS[variant]
    gm = GraphManager()
    gm.add_node(Follows(names, by))
    gm.add_node(Reader(names, ndim))
    edges = [("b", names["w"], "a", "z"), ("a", names["v"], "b", "y"), ("b", names["g"], "a", "p"),
             ("a", names["q"], "a", "r"), ("a", "u", "b", "r")]
    for source, field, target, name in (edges[::-1] if reverse else edges):
        gm.add_edge(source=source, target=target, source_field=field, target_field=name)
    return grouped(gm, ["a", "b"], mode=mode, solver=solver)


def follows_exact(variant):
    """The coupled solution at the shapes the sweeps settle at: ``v`` and
    ``y`` (3,), so ``g`` reads ``y`` in S and "ones" and not in S2."""
    reads = 0.8 if FOLLOWS[variant][1] == 1 else 0.0
    # u_a = 1 + .4 q ; u_b = -.5 + .3 u_a ; q = .1 + .5 u_a + .7 g ;
    # g = .2 + .3 u_b + reads (.3 - .6 u_a)
    A = np.array([[1, 0, -0.4, 0], [-0.3, 1, 0, 0], [-0.5, 0, 1, -0.7], [0.6 * reads, -0.3, 0, 1]])
    z = np.linalg.solve(A, np.array([1.0, -0.5, 0.1, 0.2 + 0.3 * reads]))
    return {"a": z[0], "b": z[1]}


@pytest.mark.parametrize("mode, solver", CELLS)
@pytest.mark.parametrize("reverse", [False, True], ids=["declared", "reversed"])
@pytest.mark.parametrize("naming", [0, 1])
@pytest.mark.parametrize("variant", sorted(FOLLOWS))
def test_a_flux_whose_shape_follows_a_later_flux_is_judged_at_the_shape_the_sweeps_settle_at(
        variant, naming, reverse, mode, solver):
    names = NAMINGS[naming]
    gm = follows_graph(variant, names, reverse, mode, solver)
    with pytest.raises(ValueError, match="along a chain that runs against the order") as info:
        gm.compile()
    text = str(info.value)
    assert f"flux a.{names['q']} would take 3" in text and "assumed" not in text


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_what_the_refused_shape_cases_compute_without_the_refusal[gauss-seidel-ift]
@pytest.mark.parametrize("mode, solver", [CELLS[0], pytest.param(*CELLS[1], marks=pytest.mark.slow)])
def test_what_the_refused_shape_cases_compute_without_the_refusal(mode, solver, monkeypatch):
    """The refusals above are due: with the criterion taken out each
    graph stops with ``converged=True`` off the coupled solution."""
    wrong = []
    for variant in sorted(FOLLOWS):
        gm = without_the_refusal(monkeypatch, lambda v=variant: follows_graph(v, NAMINGS[0], False, mode, solver))
        wrong.append((gm, follows_exact(variant)))
    for case in "NTO":
        gm = without_the_refusal(
            monkeypatch, lambda c=case: handed_pair(c, 1, "q", "value first", mode, solver)[0])
        wrong.append((gm, handed_exact(gm_nodes(gm), True, OUTSIDE if case == "O" else 0.0)))
    for gm, reference in wrong:
        assert all(bool(d["converged"]) for d in gm.coupling_diagnostics().values())
        assert max(abs(float(np.mean(gm.get_node_state(n)["u"])) - v) / abs(v)
                   for n, v in reference.items()) > 1e-3


def gm_nodes(gm):
    return [gm._nodes[name].node for name in ("a", "b")]


# ---------------------------------------------------------------------------
# The derivation itself.
# ---------------------------------------------------------------------------

class Half(SimulationNode):
    """A float16 scalar state field ``h`` and a flux ``q``."""

    def initial_state(self):
        return {"h": jnp.zeros((), jnp.float16)}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return state

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": jnp.ones(3, F)}


def test_an_additive_edge_adds_its_value_to_what_an_earlier_edge_delivered():
    """``_handed`` composes as the pass does: in the order of the
    ``add_edge`` calls, an additive edge adds (``+``: (3,) float32 and a
    float16 scalar make (3,) float32), any other replaces; the external
    inputs replace what the edges delivered; the first sweep leaves out
    an edge whose flux is not computed yet, a later sweep cannot derive
    the input then (the declared shape, where there is one)."""
    from maddening.core.coupling import _interface_plan

    def handed(edges, fluxes, first=False, given=None, declared=None):
        gm = GraphManager()
        gm.add_node(Half(name="b", timestep=1.0))
        gm.add_node(Half(name="a", timestep=1.0))
        for field, additive in edges:
            gm.add_edge(source="b", target="a", source_field=field, target_field="x",
                        additive=additive)
        plan = _interface_plan.interface_plan(["a", "b"], gm._edges, ["a", "b"], gm._state, gm._nodes)
        return _flux_loops._handed(plan.edges_into("a"), given or {}, gm._state, fluxes,
                                   declared or {}, None, first)

    flux = {"b": {"q": jax.ShapeDtypeStruct((3,), F)}}
    for edges in ([("q", False), ("h", True)], [("h", False), ("q", True)]):
        values, unknown, fallback = handed(edges, flux)
        assert (values["x"].shape, values["x"].dtype, unknown, fallback) == ((3,), F, set(), [])
    values, _, _ = handed([("q", False), ("h", False)], flux)
    assert (values["x"].shape, values["x"].dtype) == ((), jnp.float16)
    ext = jax.ShapeDtypeStruct((2,), jnp.float32)
    assert handed([("q", False)], flux, given={"x": ext})[0]["x"] is ext
    values, _, _ = handed([("h", False), ("q", True)], {}, first=True)
    assert (values["x"].shape, values["x"].dtype) == ((), jnp.float16)
    assert handed([("h", False), ("q", True)], {})[:2] == ({}, {"x"})
    values, unknown, fallback = handed([("h", False), ("q", True)], {}, declared={"x": ZERO})
    assert (values["x"].shape, unknown, fallback) == ((), set(), ["x"])


class IndexesLate(Side):
    """Its flux indexes ``s``, which a member swept after it delivers: the
    first sweep cannot call it, the second can."""

    boundary_input_spec = SimulationNode.boundary_input_spec

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": state["u"] + 0.5 * boundary_inputs["s"]}


def test_a_hook_that_cannot_be_evaluated_in_one_sweep_is_evaluated_in_the_next():
    """``b``'s hook cannot be evaluated in the first sweep (it indexes
    ``s``, which ``a``'s flux delivers after it); in the second it can, so
    ``a``'s input is derived and its flux, which reads none, is
    established.  The graph is then refused for what it is, a hook that
    indexes an input (by name, from ``validate()``), and not as a loop
    assumed because ``a``'s input could not be derived."""
    nodes = [IndexesLate("b", *coefficients(0.5, 1)), Bare("a", *coefficients(None, 0))]
    gm = built(nodes, [("a", "q", "b", "s", 1.0), ("b", "q", "a", "s", 1.0),
                       ("a", "u", "b", "r", 1.0), ("b", "u", "a", "r", 1.0)], ["a", "b"],
               mode="jacobi")
    gm._refuse_unsolved_group_fluxes()
    with pytest.raises(RuntimeError, match=r"it indexes boundary_inputs\['s'\]"):
        gm.compile()


class Relay(SimulationNode):
    """Fluxes ``{name: (base, the input whose shape it takes, or None)}``,
    each ``base + 0.1 u`` times a vector of ones of that input's shape;
    ``g`` reads ``y`` where it has *ndim* dimensions, ``q`` reads ``p``;
    ``u <- c + 0.3 mean(r)``."""

    def __init__(self, name, c, fluxes, ndim=1):
        super().__init__(name=name, timestep=1.0)
        self.c, self.fluxes, self.ndim = c, fluxes, ndim

    def initial_state(self):
        return {"u": ZERO}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"u": self.c + 0.3 * jnp.mean(boundary_inputs.get("r", ZERO))}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        u, out = state["u"], {}
        for flux, (base, follows) in self.fluxes.items():
            like = jnp.ones(3, F) if follows == "three" else boundary_inputs.get(follows, ZERO)
            out[flux] = (base + 0.1 * u) * jnp.ones_like(like)
        y = boundary_inputs.get("y")
        if "g" in out and y is not None and jnp.ndim(y) == self.ndim:
            out["g"] = out["g"] + 0.8 * jnp.mean(y)
        if "q" in out:
            out["q"] = out["q"] + 0.7 * jnp.mean(boundary_inputs.get("p", ZERO))
        return out


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
def test_a_flux_shape_that_settles_in_the_third_sweep_is_traced_at_that_shape(mode):
    """``b.k`` is (3,); ``a.t`` takes its shape (the second sweep),
    ``b.w`` takes ``a.t``'s (the second), ``a.v`` takes ``b.w``'s (the
    third), and ``b.g`` reads ``a.v`` where it is a vector.  Taken after
    the two seeding sweeps, ``a.v`` is a scalar, ``b.g`` reads nothing,
    and ``a.q`` (read by ``a``'s update from the seed) looks two sweeps
    away; at the settled shapes it is four."""
    a = Relay("a", 1.0, {"t": (0.1, "kin"), "v": (0.3, "z"), "q": (0.1, None)})
    b = Relay("b", -0.5, {"k": (0.2, "three"), "w": (0.3, "tin"), "g": (0.2, None)})
    edges = [("b", "k", "a", "kin"), ("a", "t", "b", "tin"), ("b", "w", "a", "z"), ("a", "v", "b", "y"),
             ("b", "g", "a", "p"), ("a", "q", "a", "r"), ("a", "u", "b", "r")]
    gm = GraphManager()
    for node in (a, b):
        gm.add_node(node)
    for source, field, target, name in edges:
        gm.add_edge(source=source, target=target, source_field=field, target_field=name)
    grouped(gm, ["a", "b"], mode=mode)
    with pytest.raises(ValueError, match="flux a.q would take 4") as info:
        gm.compile()
    assert "assumed" not in str(info.value)


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
def test_a_shape_the_second_sweep_hands_before_it_settles_is_traced_too(mode):
    """``b.g`` reads ``y`` only where it is a scalar.  ``y`` is ``b``'s own
    flux ``s``, which takes the shape of ``a.t``, which takes the shape of
    ``b.k`` (3,): the first sweep leaves ``y`` out, the settled shape is
    (3,), and neither is read; the second sweep hands ``y`` as the first
    computed ``s``, a scalar, and ``b.g`` reads it there.  So the seed of
    ``b.g``, which ``a``'s update reads, is not its settled value after
    the two sweeps; the message says the read is that sweep's."""
    a = Relay("a", 1.0, {"t": (0.1, "kin")})
    b = Relay("b", -0.5, {"k": (0.2, "three"), "s": (0.3, "tin"), "g": (0.2, None)}, ndim=0)
    gm = GraphManager()
    for node in (a, b):
        gm.add_node(node)
    for source, field, target, name in [("b", "k", "a", "kin"), ("a", "t", "b", "tin"),
                                        ("b", "s", "b", "y"), ("b", "g", "a", "r"),
                                        ("a", "u", "b", "r")]:
        gm.add_edge(source=source, target=target, source_field=field, target_field=name)
    grouped(gm, ["a", "b"], mode=mode)
    with pytest.raises(ValueError, match="flux b.g would take 3") as info:
        gm.compile()
    assert ("flux b.g reads input 'y' in seeding sweep 2, which hands it otherwise than once "
            "the fluxes settle (edge 'b.s->b.y' delivers it from flux b.s)") in str(info.value)


def test_an_input_taken_at_its_declared_shape_is_named_in_the_message():
    """``b``'s input ``s`` carries the flux of ``a``'s hook, which cannot
    be evaluated: ``b`` is traced with ``s`` at the shape it declares, and
    the message says so beside what it establishes."""
    nodes = [Untraceable("a", *coefficients(None, 0)), Side("b", *coefficients(0.5, 1))]
    gm = built(nodes, [("a", "q", "b", "s", 1.0), ("b", "q", "a", "s", 1.0)], ["a", "b"])
    with pytest.raises(ValueError, match="depend on each other in a loop") as info:
        gm.compile()
    assert ("'b' was traced with its input ['s'] at the shape and dtype its "
            "boundary_input_spec() declares, as what the edges deliver there could not be "
            "derived") in str(info.value)


class UntraceableBare(Untraceable):
    """An :class:`Untraceable` that declares no input."""

    boundary_input_spec = SimulationNode.boundary_input_spec


def test_an_input_one_of_whose_edges_cannot_be_evaluated_is_not_derived_from_the_others():
    """``b``'s input ``r`` is ``a``'s value plus ``a``'s flux, whose hook
    cannot be evaluated: the pass would hand their sum, whose shape is
    not known, so ``r`` is not derived from the value alone; ``b``
    declares nothing, and what its flux reads is assumed, with that
    reason, naming the edge; ``a`` is reported as a hook that cannot be
    traced, with the tracer's error, though its own input cannot be
    derived either."""
    nodes = [UntraceableBare("a", *coefficients(None, 0), reads="r"),
             Bare("b", *coefficients(None, 1), reads="r")]
    gm = built(nodes, ROBIN_AB, ["a", "b"])
    with pytest.raises(ValueError, match="depend on each other in a loop") as info:
        gm.compile()
    text = str(info.value)
    assert ("for 'b', no shape could be derived for its input ['r']: edge 'a.q->b.r' carries "
            "a flux of 'a', whose fluxes could not be evaluated on the inputs the pass hands "
            "it; and boundary_input_spec() does not declare it") in text
    assert ("for 'a', its compute_boundary_fluxes could not be traced on the inputs the pass "
            "hands it (ConcretizationTypeError") in text


ROBIN_AB = [(s.replace("node_", ""), f, t.replace("node_", ""), n, g) for s, f, t, n, g in ROBIN]


class Grows(SimulationNode):
    """Its flux ``v`` is one entry longer than the input ``z`` it is
    handed, and reads it: round a loop its shape never settles."""

    def initial_state(self):
        return {"u": ZERO}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"u": 0.5 + 0.1 * jnp.mean(boundary_inputs.get("z", ZERO))}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        z = jnp.atleast_1d(boundary_inputs.get("z", ZERO))
        return {"v": jnp.concatenate([0.5 * z, state["u"][None]])}


def test_flux_shapes_that_never_settle_are_assumed_and_the_message_says_why():
    gm = GraphManager()
    gm.add_node(Grows(name="a", timestep=1.0))
    gm.add_node(Grows(name="b", timestep=1.0))
    gm.add_edge(source="a", target="b", source_field="v", target_field="z")
    gm.add_edge(source="b", target="a", source_field="v", target_field="z")
    grouped(gm, ["a", "b"])
    with pytest.raises(ValueError, match="depend on each other in a loop") as info:
        gm.compile()
    text = str(info.value)
    assert "had not settled after 4 sweeps over the members" in text
    assert "give each flux a shape and dtype that do not follow" in text
    assert "JAX operations" not in text and "declare an input" not in text


# Per push: tests/core/test_boundary_fluxes_that_read_each_other_in_a_group.py::test_an_edge_whose_value_a_later_edge_replaces_is_read_by_nothing[gauss-seidel-ift]
@pytest.mark.parametrize("mode, solver", [CELLS[0], pytest.param(*CELLS[1], marks=pytest.mark.slow)])
def test_an_edge_whose_value_a_later_edge_replaces_is_read_by_nothing(mode, solver, monkeypatch):
    """The flux edge is declared first and the value edge after it, not
    additive: the pass hands the value alone, and the hook, which reads
    ``x`` at every shape here, reads no flux.  It compiles, at the
    solution in which no flux is delivered, with the pass's numbers."""
    gm, nodes = handed_pair("N", 0, "q", "flux first", mode, solver, replaced=True)
    assert off(gm, handed_exact(nodes, True, flux_delivered=False)) < CLOSE
    before = without_the_refusal(
        monkeypatch, lambda: handed_pair("N", 0, "q", "flux first", mode, solver, replaced=True)[0])
    assert numbers(gm) == numbers(before)


class ShapeOnly(Side):
    """Its fluxes ``q`` (3,) and ``w`` read only ``x.shape[0]`` (static):
    computed from the state alone."""

    boundary_input_spec = SimulationNode.boundary_input_spec

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        x = boundary_inputs.get("x")
        q = (self.k[2] + self.k[4] * state["u"]) * (x.shape[0] / 3.0 if x is not None else 1.0)
        return {"q": q * jnp.ones(3, F), "w": q}


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
def test_a_hook_that_reads_only_the_shape_of_an_input_compiles(mode, monkeypatch):
    """``x`` is a scalar flux ``w`` (declared first) plus a (3,) flux
    ``q`` (additive): the pass hands (3,).  Derived as a scalar, the hook
    failed on ``x.shape[0]`` and the group was refused with two assumed
    reasons, both false; its fluxes are of the state alone."""
    def build():
        nodes = [ShapeOnly("a", *coefficients(None, 0), reads="x"),
                 ShapeOnly("b", *coefficients(None, 1), reads="x")]
        gm = GraphManager()
        for node in nodes:
            gm.add_node(node)
        for source, target in (("b", "a"), ("a", "b")):
            gm.add_edge(source=source, target=target, source_field="w", target_field="x",
                        transform=HALF.__func__)
            gm.add_edge(source=source, target=target, source_field="q", target_field="x",
                        transform=HALF.__func__, additive=True)
        return grouped(gm, ["a", "b"], mode=mode)

    gm = build()
    gm.compile()
    gm.step()
    assert numbers(gm) == numbers(without_the_refusal(monkeypatch, build))


class StateKey(Side):
    """Its flux hook indexes a field its state does not hold."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": state["u"] + 0.0 * state["gone"]}


class TableKey(Side):
    """Its flux hook indexes a dict of its own."""

    table: dict = {}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": state["u"] * self.table["k"]}


class CopyKey(Side):
    """Its flux hook indexes a copy of its state."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"q": state["u"] + 0.0 * dict(state)["gone"]}


class JitInputKey(Side):
    """Its flux hook indexes an input inside a ``jax.jit``, which hands it
    copies of its dicts."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return jax.jit(lambda st, bi: {"q": st["u"] + bi["s"]})(state, boundary_inputs)


class JitStateKey(Side):
    """Its flux hook indexes a field its state does not hold, inside a
    ``jax.jit``."""

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return jax.jit(lambda st: {"q": st["u"] + 0.0 * st["gone"]})(state)


NOT_KNOWN = ("The key was not looked up in the boundary_inputs or the state the hook is "
             "handed, so where it was looked up is not known")
GET_ADVICE = "boundary_inputs.get("


@pytest.mark.parametrize("cls, said", [
    (StateKey, "it indexes state['gone'], which is not a field of the node's state (its "
               "fields: ['u'])"),
    (JitStateKey, "it indexes state['gone'], which is not a field of the node's state (its "
                  "fields: ['u'])"),
    (TableKey, NOT_KNOWN),
    (CopyKey, NOT_KNOWN),
    (Counting, "it indexes boundary_inputs['s']"),
    (JitInputKey, "it indexes boundary_inputs['s']")])
def test_a_flux_hook_that_raises_key_error_is_told_which_dict_it_indexed(cls, said):
    """The dict a hook's ``KeyError`` came from is named where it is
    known (the dict handed, or its copy inside a ``jax.jit``), and where
    it is not (a copy made with ``dict(...)``, a table of the hook's
    own) the message says so and gives the advice for both: the
    ``.get`` of an input, and the state's fields."""
    nodes = [cls("a", *coefficients(None, 0)), Side("b", *coefficients(None, 1))]
    gm = built(nodes, [("a", "q", "b", "r", 1.0), ("b", "u", "a", "r", 1.0)], ["a", "b"])
    errors = [line for line in gm.validate() if line.startswith("ERROR")]
    assert len(errors) == 1 and said in errors[0]
    assert ("boundary_inputs[" in errors[0]) == (cls in (Counting, JitInputKey))
    assert (GET_ADVICE in errors[0]) == (cls in (Counting, JitInputKey, TableKey, CopyKey))
    assert ("not from indexing" in errors[0]) is False
