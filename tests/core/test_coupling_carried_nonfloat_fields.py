"""An integer or Boolean field on a coupling group's loop is iterated with the solve.

A coupling group iterates its floating fields and reads convergence from
them.  When an edge inside the group carries an integer or a Boolean field
-- one member computes it from what it reads, another reads it -- that
field is part of the loop all the same, and the solvers used to lose it in
two ways, each ending in ``converged=True``, ``residual=0.0`` far from the
fixed point:

* ``solver="ift"`` evaluated such a field with the group's own pass.  A
  Gauss-Seidel sweep hands a producer the reader's *updated* state, so with
  the reader swept first the field came back one pass ahead of the iterate,
  the floating map no longer depended on the iterate, and the solve stopped
  on its second pass.  One evaluation also settles one field only: a flag
  relayed through a second non-floating field stayed a pass behind.
* ``solver="fori"`` carried the field as the previous pass left it.  Under
  Jacobi a float that reads an integer that reads the float then changed
  only on every other pass, and the pass between measured a floating
  residual of exactly zero.

Both solvers now start every pass they measure from those fields evaluated
at the floating iterate (``maddening.core.coupling._carried_fields`` states
the rule and proves it), so every combination of solver and sweep iterates
one map, and a zero residual is a fixed point of the group's pass.

Which non-floating fields a pass reads from the iterate, and which of them
is computed from which, is read from one abstract trace of the group's own
Jacobi pass, not assumed from the edges.  So a step counter or a PRNG key
held by a member that exchanges boundary fluxes both ways is not carried at
all (no flux reads it) and such a group steps as it always did; and the one
thing ``compile()`` refuses is a group in which the trace shows non-floating
fields computed from one another in a closed loop.

The fixtures are exact in binary floating point (halves and small
integers), so states and pass counts do not depend on rounding, on the
jaxlib or on whether a multiply-add is fused.
"""

import functools
import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _carried_fields, _coupled_block, _interface_plan
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode

#: ``a: q <- C + K p`` and ``b: p <- floor(q)``.  ``q = C + K floor(q)`` needs
#: ``C - (1 - K) p`` in ``[0, 1)``: with ``K = -1/2``, ``C = 20`` only
#: ``p = 13`` qualifies, so the pass has exactly one fixed point.
K, C = -0.5, 20.0
Q_STAR, P_STAR = 13.5, 13

_F32 = jnp.float32


def _zero(dtype):
    return jnp.zeros((), dtype)


class Reader(SimulationNode):
    """``q <- c + k p``: a float that reads a non-floating field ``p``."""

    def __init__(self, name, p_dtype, *, k=K, c=C):
        super().__init__(name, 1.0, k=k, c=c)
        self._pd = p_dtype

    def initial_state(self):
        return {"q": _zero(_F32)}

    def boundary_input_spec(self):
        return {"p": BoundaryInputSpec(shape=(), dtype=self._pd, default=_zero(self._pd))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"q": (p["c"] + p["k"] * boundary_inputs["p"].astype(_F32)).astype(_F32)}


class Floor(SimulationNode):
    """``p <- floor(q)``, stored as ``p_dtype``."""

    def __init__(self, name, p_dtype):
        super().__init__(name, 1.0)
        self._pd = p_dtype

    def initial_state(self):
        return {"p": _zero(self._pd)}

    def boundary_input_spec(self):
        return {"q": BoundaryInputSpec(shape=(), dtype=_F32, default=_zero(_F32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"p": jnp.floor(boundary_inputs["q"]).astype(self._pd)}


class Above(SimulationNode):
    """``p <- (q > threshold)`` or ``(q < threshold)``: a Boolean."""

    def __init__(self, name, threshold, *, below=False):
        super().__init__(name, 1.0)
        self._threshold, self._below = threshold, below

    def initial_state(self):
        return {"p": jnp.asarray(False)}

    def boundary_input_spec(self):
        return {"q": BoundaryInputSpec(shape=(), dtype=_F32, default=_zero(_F32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        q = boundary_inputs["q"]
        return {"p": (q < self._threshold) if self._below else (q > self._threshold)}


class Relay(SimulationNode):
    """``p <- p``: hands a non-floating field on unchanged."""

    def __init__(self, name, p_dtype):
        super().__init__(name, 1.0)
        self._pd = p_dtype

    def initial_state(self):
        return {"p": _zero(self._pd)}

    def boundary_input_spec(self):
        return {"p": BoundaryInputSpec(shape=(), dtype=self._pd, default=_zero(self._pd))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"p": boundary_inputs["p"]}


class FluxGate(SimulationNode):
    """``on <- (u > 1)``; its boundary flux ``f = 2 + 4 on`` reads its own flag."""

    def __init__(self, name):
        super().__init__(name, 1.0)

    def initial_state(self):
        return {"y": _zero(_F32), "on": jnp.asarray(False)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=_F32, default=_zero(_F32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"]
        return {"y": u.astype(_F32), "on": u > 1.0}

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        return {"f": jnp.where(state["on"], 6.0, 2.0).astype(_F32)}


class FluxReader(SimulationNode):
    """``q <- f``: takes a flux as its value."""

    def __init__(self, name):
        super().__init__(name, 1.0)

    def initial_state(self):
        return {"q": _zero(_F32)}

    def boundary_input_spec(self):
        return {"f": BoundaryInputSpec(shape=(), dtype=_F32, default=_zero(_F32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        return {"q": boundary_inputs["f"].astype(_F32)}


def _group(gm, names, **group):
    """The group, reporting: ``"ift"`` always does, ``"fori"`` with ``diagnostics``."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "CouplingGroup solver='fori' is deprecated",
                                DeprecationWarning)
        gm.add_coupling_group(names, **{"max_iterations": 60, "tolerance": 1e-6,
                                        "diagnostics": group.get("solver") == "fori",
                                        **group})
    return gm


def _loop(producer, *, reader_first, relays=0, p_dtype=jnp.int32, k=K, c=C, **group):
    """``a`` reads the field ``b`` computes from ``a.q``, through *relays* relay nodes.

    The field is carried ``b -> r1 -> ... -> a``; with ``reader_first`` the
    members are scheduled reader, producer, relays (each reads the field
    from the iterate under Gauss-Seidel), otherwise in the order the field
    travels.
    """
    nodes = {"a": Reader("a", p_dtype, k=k, c=c), "b": producer}
    relay_names = [f"r{i + 1}" for i in range(relays)]
    for name in relay_names:
        nodes[name] = Relay(name, p_dtype)
    travel = ["b", *relay_names, "a"]
    gm = GraphManager()
    for name in (["a", "b", *relay_names] if reader_first else travel):
        gm.add_node(nodes[name])
    gm.add_edge("a", "b", "q", "q")
    for src, dst in zip(travel, travel[1:]):
        gm.add_edge(src, dst, "p", "p")
    return _group(gm, sorted(nodes), **group)


def _chains(gm):
    """``(what the edges allow, what a trace of the pass shows)`` for *gm*'s one group."""
    (group,) = gm._coupling_groups
    members = [name for name in gm._nodes if name in group.nodes]
    plan = _interface_plan.interface_plan(group.nodes, gm._edges, members, gm._state, gm._nodes)
    bound = _carried_fields.possible_chain(plan, gm._state)
    return bound, gm._traced_carried_chain(group, members, bound)


def _floor_loop(**kw):
    p_dtype = kw.pop("p_dtype", jnp.int32)
    return _loop(Floor("b", p_dtype), p_dtype=p_dtype, **kw)


def _report(gm):
    (report,) = gm.coupling_diagnostics().values()
    return report


@functools.lru_cache(maxsize=None)
def _floor_result(solver, mode, reader_first, relays=0, dtype="int32", **group_items):
    """``(q, p, iterations, residual, converged)`` after one step (plain numbers)."""
    gm = _floor_loop(reader_first=reader_first, relays=relays, p_dtype=jnp.dtype(dtype).type,
                     solver=solver, iteration_mode=mode, **dict(group_items))
    gm.step()
    r = _report(gm)
    return (float(gm.get_node_state("a")["q"]), int(gm.get_node_state("b")["p"]),
            int(r["iterations"]), float(r["residual"]), bool(r["converged"]))


def _passes_to_the_fixed_point():
    """Passes ``x <- C + K floor(x)`` takes from the first pass's ``q = C`` (NumPy)."""
    x, passes = np.float32(C), 1
    while True:
        x_next = np.float32(C) + np.float32(K) * np.floor(x)
        if x_next == x:
            return passes
        x, passes = x_next, passes + 1


ROUTES = [(solver, mode, reader_first)
          for solver in ("ift", "fori")
          for mode in ("gauss-seidel", "jacobi")
          for reader_first in (True, False)]


def test_the_fixture_has_one_fixed_point():
    """The oracle: exactly one ``p`` satisfies ``p == floor(C + K p)``."""
    fixed = [p for p in range(-200, 200) if np.floor(C + K * p) == p]
    assert fixed == [P_STAR] and C + K * P_STAR == Q_STAR
    assert _passes_to_the_fixed_point() == 7


@pytest.mark.parametrize("solver, mode, reader_first", ROUTES)
def test_an_integer_on_the_loop_reaches_the_fixed_point_on_every_route(
        solver, mode, reader_first):
    """Every solver, sweep and schedule returns the one fixed point, in the same passes.

    ``"ift"`` under Gauss-Seidel with the reader swept first used to stop
    after two passes, and ``"fori"`` under Jacobi after one, both with
    ``converged=True`` and ``residual=0.0``.
    """
    q, p, iterations, residual, converged = _floor_result(solver, mode, reader_first)
    assert (q, p) == (Q_STAR, P_STAR)
    assert converged and residual == 0.0
    # One map on every route: the float sees the integer at the iterate.
    assert iterations == _passes_to_the_fixed_point()


@pytest.mark.parametrize("dtype", ["int8", "uint16", "int16"])
@pytest.mark.parametrize("solver, mode, reader_first", [
    ("ift", "gauss-seidel", True), ("fori", "jacobi", False)])
def test_the_width_of_the_integer_does_not_matter(solver, mode, reader_first, dtype):
    assert _floor_result(solver, mode, reader_first, dtype=dtype) == (
        Q_STAR, P_STAR, _passes_to_the_fixed_point(), 0.0, True)


@pytest.mark.parametrize("relays", [1, 2])
@pytest.mark.parametrize("solver, mode, reader_first", ROUTES)
def test_a_field_relayed_through_more_non_floating_fields_is_settled_at_the_iterate(
        solver, mode, reader_first, relays):
    """Two and three non-floating hops: one evaluation per hop of the chain.

    ``"ift"`` took one evaluation whatever the chain, so under Jacobi the
    relayed copy stayed at its first-pass value and the solve returned the
    first pass with ``converged=True``.
    """
    q, p, iterations, residual, converged = _floor_result(
        solver, mode, reader_first, relays=relays)
    assert (q, p) == (Q_STAR, P_STAR)
    assert converged and residual == 0.0
    assert iterations == _passes_to_the_fixed_point()


@pytest.mark.parametrize("relays, depth", [(0, 1), (1, 2), (2, 3)])
def test_the_chain_is_as_long_as_the_fields_that_feed_one_another(relays, depth):
    gm = _floor_loop(reader_first=True, relays=relays)
    bound, traced = _chains(gm)
    assert traced.loop is None and traced.depth == depth and traced.assumed is None
    assert traced.fields == tuple(sorted(
        [("b", "p"), *((f"r{i + 1}", "p") for i in range(relays))]))
    # Every hand-over here is real, so the bound the edges give is the trace's.
    assert bound == traced


_ACCELERATED = [
    (("acceleration", "fixed"), ("relaxation", 0.5)),
    (("acceleration", "aitken"),),
    (("acceleration", "iqn-ils"),),
    (("acceleration", "iqn-imvj"),),
]


@pytest.mark.parametrize("accel", _ACCELERATED, ids=lambda a: str(a[0][1]))
@pytest.mark.parametrize("mode, reader_first", [("gauss-seidel", True), ("jacobi", False)])
def test_the_solvers_agree_under_every_acceleration(mode, reader_first, accel):
    """One map under both solvers: the same passes, the same verdict, the same state.

    And a solve that reads ``converged=True`` has returned the fixed point,
    whatever path the accelerator took to it.
    """
    ift = _floor_result("ift", mode, reader_first, **dict(accel))
    fori = _floor_result("fori", mode, reader_first, **dict(accel))
    assert ift[1:3] == fori[1:3] and ift[4] == fori[4]
    assert ift[0] == pytest.approx(fori[0], abs=1e-5)
    for q, p, _iterations, _residual, converged in (ift, fori):
        if converged:
            assert p == P_STAR and q == pytest.approx(Q_STAR, abs=1e-4)


@pytest.mark.parametrize("accel", [(), *_ACCELERATED], ids=lambda a: str(a[0][1]) if a else "none")
def test_fori_without_diagnostics_iterates_the_same_map(accel):
    """Each acceleration has a second loop body, taken with ``diagnostics=False``.

    Held to eight passes, one more than the plain iteration needs: a body
    that carried the integer a pass behind would need about twice as many,
    so it is told apart by the state it has reached, converged or not
    (Aitken's two-pass exit does not stop on the lagged zero residual, and
    would arrive at the same fixed point later).
    """
    with_report = _floor_result("fori", "jacobi", False, max_iterations=8, **dict(accel))
    gm = _floor_loop(reader_first=False, solver="fori", iteration_mode="jacobi",
                     diagnostics=False, max_iterations=8, **dict(accel))
    gm.step()
    assert float(gm.get_node_state("a")["q"]) == with_report[0]
    assert int(gm.get_node_state("b")["p"]) == with_report[1]
    if not accel:
        assert float(gm.get_node_state("a")["q"]) == Q_STAR


@pytest.mark.parametrize("options", [
    {"predictor": "linear"}, {"predictor": "quadratic"},
    {"convergence_norm": "mixed"}, {"convergence_norm": "interface"}],
    ids=lambda o: "-".join(map(str, o.values())))
@pytest.mark.parametrize("solver, mode, reader_first", [
    ("ift", "gauss-seidel", True), ("fori", "jacobi", False)])
def test_the_fixed_point_is_reached_under_the_groups_other_options(
        solver, mode, reader_first, options):
    """A predictor's start and each norm, over a scan of steps.

    The pass does not depend on the pre-step state, so every step has the
    same fixed point; a predictor only moves the guess, and the interface
    norm reads the one floating edge.
    """
    gm = _floor_loop(reader_first=reader_first, solver=solver, iteration_mode=mode, **options)
    out = gm.run_scan(4)
    assert float(out["a"]["q"]) == Q_STAR and int(out["b"]["p"]) == P_STAR
    assert _report(gm)["converged"]


def _boolean_loop(solver, mode, reader_first, *, below, relays=0):
    """``q <- 2 + 4 p`` with ``p <- (q > 1)``: one fixed point, ``q = 6`` with ``p`` set.

    With ``below`` the flag is ``q < 5``: set, it gives ``q = 6`` which
    clears it; clear, ``q = 2`` which sets it.  No state is consistent.
    """
    gm = _loop(Above("b", 5.0 if below else 1.0, below=below), reader_first=reader_first,
               relays=relays, p_dtype=jnp.bool_, k=4.0, c=2.0, solver=solver,
               iteration_mode=mode, max_iterations=12)
    gm.step()
    return gm


@pytest.mark.parametrize("solver, mode, reader_first", ROUTES)
def test_a_boolean_on_the_loop_reaches_the_fixed_point_on_every_route(
        solver, mode, reader_first):
    gm = _boolean_loop(solver, mode, reader_first, below=False)
    r = _report(gm)
    assert float(gm.get_node_state("a")["q"]) == 6.0
    assert bool(gm.get_node_state("b")["p"]) is True
    assert r["converged"] and r["residual"] == 0.0


@pytest.mark.parametrize("solver, mode, reader_first", ROUTES)
def test_a_flag_that_flips_on_every_pass_reaches_the_cap_unconverged(
        solver, mode, reader_first):
    """No consistent state exists, so no route may read ``converged=True``.

    Both early stops reported it: the frozen map of ``"ift"`` and the lagged
    carry of ``"fori"`` each produced a pass that did not move the float.
    """
    r = _report(_boolean_loop(solver, mode, reader_first, below=True))
    assert not r["converged"]
    # The float goes 2, 6, 2, ...: every pass moves it by 4, which the norm
    # reads against its magnitude.
    assert r["iterations"] == 12 and r["residual"] > 0.5


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_at_the_cap_the_returned_fields_are_the_pass_at_the_returned_floats(solver):
    """An unconverged solve still returns fields that describe one state.

    The flag and its relayed copy are those a pass computes from the
    returned float with the carried fields evaluated at it: the flag of
    that float, and a copy equal to it.  (The loop's last raw pass left a
    copy one pass behind the flag.)
    """
    gm = _boolean_loop(solver, "jacobi", False, below=True, relays=1)
    assert not _report(gm)["converged"]
    q = float(gm.get_node_state("a")["q"])
    assert q in (2.0, 6.0)
    assert bool(gm.get_node_state("b")["p"]) is (q < 5.0)
    assert bool(gm.get_node_state("r1")["p"]) is (q < 5.0)


@pytest.mark.parametrize("solver, mode, reader_first", ROUTES)
def test_a_flag_a_flux_is_computed_from_is_evaluated_at_the_iterate(
        solver, mode, reader_first):
    """A flux producer's own non-floating state is read from the iterate too.

    ``f``'s flux is ``2 + 4 on`` and ``on = (q > 1)``: the one consistent
    state has ``q = 6``.  No edge carries ``on``; the pass reads it when it
    computes the flux.  ``"ift"`` held it at its first-pass value inside the
    loop and, under Jacobi, returned ``q = 2`` with ``converged=True``.
    """
    from maddening.core.coupling.helpers import add_flux_coupling

    gm = GraphManager()
    nodes = [FluxReader("a"), FluxGate("f")]
    for node in (nodes if reader_first else nodes[::-1]):
        gm.add_node(node)
    gm.add_edge("a", "f", "q", "u")
    add_flux_coupling(gm, "f", "a", "f", "f")
    _group(gm, ["a", "f"], solver=solver, iteration_mode=mode, max_iterations=12)
    gm.step()
    r = _report(gm)
    assert float(gm.get_node_state("a")["q"]) == 6.0
    assert bool(gm.get_node_state("f")["on"]) is True
    assert r["converged"] and r["residual"] == 0.0


@pytest.mark.parametrize("solver, mode, reader_first", [
    ("ift", "gauss-seidel", True), ("fori", "jacobi", False)])
def test_the_gradient_holds_the_integer_fixed(solver, mode, reader_first):
    """``q* = c + k p*`` with ``p*`` locally constant: ``dq*/dc = 1``, ``dq*/dk = p*``."""
    gm = _floor_loop(reader_first=reader_first, solver=solver, iteration_mode=mode)
    gm.compile()

    def loss(p):
        return gm.run_scan(1, params=p)["a"]["q"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "the graph held JAX tracers", RuntimeWarning)
        g = jax.grad(loss)(gm.params)["nodes"]["a"]
    assert float(g["c"]) == pytest.approx(1.0, rel=1e-5)
    assert float(g["k"]) == pytest.approx(float(P_STAR), rel=1e-5)


@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
def test_one_staggered_pass_reads_the_field_as_it_is_stored(solver, mode):
    """``max_iterations=1`` iterates nothing, and nothing is evaluated for it.

    Each step is one pass from the stored state: under Jacobi the float
    reads the integer the previous step left, and the integer the float it
    left.
    """
    gm = _floor_loop(reader_first=True, solver=solver, iteration_mode=mode, max_iterations=1)
    q, p = np.float32(0.0), 0
    for _ in range(4):
        gm.step()
        q_new = np.float32(C) + np.float32(K) * np.float32(p)
        p = int(np.floor(q_new if mode == "gauss-seidel" else q))
        q = q_new
        assert float(gm.get_node_state("a")["q"]) == q
        assert int(gm.get_node_state("b")["p"]) == p


# ---------------------------------------------------------------------------
# What is carried is read from a trace of the pass, not assumed from the edges
# ---------------------------------------------------------------------------

class FluxPair(SimulationNode):
    """``x <- 1 + u / 4`` with the boundary flux ``f = 2 x``, and a non-floating leaf.

    ``leaf`` is ``"count"`` (``n <- n + 1`` from the pre-step state),
    ``"key"`` (a typed PRNG key split from the pre-step key) or ``None``.
    ``flux_reads_count`` adds ``n / 8`` to the flux.  ``count_reads_input``
    makes the count ``n + (u > 0)``, and ``flux_reads_input`` adds ``u / 2``
    to the flux: together with ``flux_reads_count`` on the other side of
    the pair they close a loop through the two fluxes.
    """

    def __init__(self, name, leaf=None, *, flux_reads_count=False, count_reads_input=False,
                 flux_reads_input=False):
        super().__init__(name, 1.0)
        self._leaf = leaf
        self._flux_reads_count = flux_reads_count
        self._count_reads_input = count_reads_input
        self._flux_reads_input = flux_reads_input

    def initial_state(self):
        state = {"x": _zero(_F32)}
        if self._leaf == "count":
            state["n"] = _zero(jnp.int32)
        if self._leaf == "key":
            state["key"] = jax.random.key(7)
        return state

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=_F32, default=_zero(_F32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"]
        out = {"x": (1.0 + 0.25 * u).astype(_F32)}
        if self._leaf == "count":
            step = (u > 0).astype(jnp.int32) if self._count_reads_input else jnp.int32(1)
            out["n"] = state["n"] + step
        if self._leaf == "key":
            out["key"] = jax.random.split(state["key"])[0]
        return out

    def compute_boundary_fluxes(self, state, boundary_inputs, dt, *, params=None):
        flux = 2.0 * state["x"]
        if self._flux_reads_count:
            flux = flux + 0.125 * state["n"].astype(_F32)
        if self._flux_reads_input:
            # With a default: ``validate()`` calls a flux hook with no inputs.
            flux = flux + 0.5 * boundary_inputs.get("u", _zero(_F32))
        return {"f": flux.astype(_F32)}


def _flux_pair(a, b, *, solver="ift", mode="gauss-seidel", a_first=True):
    """``a`` and ``b`` exchanging their boundary fluxes, both ways."""
    from maddening.core.coupling.helpers import add_flux_coupling

    gm = GraphManager()
    for node in ((a, b) if a_first else (b, a)):
        gm.add_node(node)
    add_flux_coupling(gm, "a", "b", "f", "u")
    add_flux_coupling(gm, "b", "a", "f", "u")
    return _group(gm, ["a", "b"], solver=solver, iteration_mode=mode, max_iterations=40)


def _trajectory(gm, steps=3):
    """Per step: both floats (exact), the passes, the verdict and the residual."""
    out = []
    for _ in range(steps):
        gm.step()
        r = _report(gm)
        out.append((float(gm.get_node_state("a")["x"]).hex(), float(gm.get_node_state("b")["x"]).hex(),
                    int(r["iterations"]), bool(r["converged"]), float(r["residual"])))
    return out


@pytest.mark.parametrize("solver, mode, a_first", ROUTES)
def test_a_counter_and_a_key_beside_fluxes_both_ways_are_not_carried(solver, mode, a_first):
    """A member with a step counter and one with a PRNG key, exchanging fluxes both ways.

    The edges alone would allow the counter to feed itself round the two
    fluxes (a flux is a function of its producer's state).  The trace of the
    pass shows that no flux reads the counter or the key and that each is
    computed from the pre-step state: nothing is carried, the group compiles,
    and its floats, passes, verdict and residual are those of the same pair
    without the leaves, bit for bit, on every route -- while the counter
    counts the steps and the key is the pre-step key split once a step.
    """
    with_leaves = _flux_pair(FluxPair("a", "count"), FluxPair("b", "key"),
                             solver=solver, mode=mode, a_first=a_first)
    bound, traced = _chains(with_leaves)
    assert bound.loop == (("a", "n"),)                  # what the edges alone would say
    assert traced == _carried_fields.CarriedChain((), 0, None)
    assert with_leaves.validate() == _flux_pair(
        FluxPair("a"), FluxPair("b"), solver=solver, mode=mode, a_first=a_first).validate()
    bare = _flux_pair(FluxPair("a"), FluxPair("b"), solver=solver, mode=mode, a_first=a_first)
    assert _trajectory(with_leaves) == _trajectory(bare)
    assert int(with_leaves.get_node_state("a")["n"]) == 3
    key = jax.random.key(7)
    for _ in range(3):
        key = jax.random.split(key)[0]
    np.testing.assert_array_equal(
        jax.random.key_data(with_leaves.get_node_state("b")["key"]), jax.random.key_data(key))


@pytest.mark.parametrize("solver, mode, a_first", ROUTES)
def test_a_counter_its_flux_reads_is_carried_and_evaluated_at_the_iterate(solver, mode, a_first):
    """The flux is ``2 x + n / 8``: the counter is carried, and is on no loop.

    Its own value is still the pre-step count plus one, so it has depth one
    and every pass reads the flux at the count the step returns.  With
    ``x_a = 1 + x_b / 2`` and ``x_b = 1 + x_a / 2 + n / 32`` the fixed point
    after ``n`` steps is ``x_a = 2 + n / 48``, ``x_b = 2 + n / 24``.
    """
    gm = _flux_pair(FluxPair("a", "count", flux_reads_count=True), FluxPair("b", "key"),
                    solver=solver, mode=mode, a_first=a_first)
    _bound, traced = _chains(gm)
    assert traced == _carried_fields.CarriedChain((("a", "n"),), 1, None)
    for n in (1, 2, 3):
        gm.step()
        assert _report(gm)["converged"]
        assert int(gm.get_node_state("a")["n"]) == n
        assert float(gm.get_node_state("a")["x"]) == pytest.approx(2.0 + n / 48.0, abs=2e-5)
        assert float(gm.get_node_state("b")["x"]) == pytest.approx(2.0 + n / 24.0, abs=2e-5)


@pytest.mark.parametrize("solver, mode", [("ift", "gauss-seidel"), ("fori", "jacobi")])
def test_a_count_computed_from_a_flux_that_reads_it_is_a_loop_and_is_refused(solver, mode):
    """``n <- n + (u > 0)``, ``u`` the other member's flux, which reads this member's.

    ``a``'s flux reads ``n``; ``b``'s flux reads its input, which is
    ``a``'s flux; ``a``'s count reads its input, which is ``b``'s flux.  The
    trace shows ``n`` to be an operand of its own value, and no number of
    evaluations gives it a value at an iterate.  ``validate()`` finds
    nothing else wrong with the graph: the refusal is its one error.
    """
    gm = _flux_pair(
        FluxPair("a", "count", flux_reads_count=True, count_reads_input=True),
        FluxPair("b", flux_reads_input=True), solver=solver, mode=mode)
    (error,) = [issue for issue in gm.validate() if issue.startswith("ERROR")]
    assert "a.n (int32) -> a.n (int32)" in error
    with pytest.raises(ValueError, match=r"a\.n \(int32\) -> a\.n \(int32\)") as refused:
        gm.compile()
    assert "read from one trace of the group's pass" in str(refused.value)
    assert "assumed" not in str(refused.value)


class Flagged(SimulationNode):
    """``x <- u / 2 + 0.2 + flag_in / 4``; ``flag <- (u > 0.3)``, or that ``| flag_in``.

    ``flag_in`` arrives on an input named ``"signal"``.  ``declared`` says
    whether ``boundary_input_spec()`` declares it: an edge may deliver to an
    input the node does not declare, and the pass hands it over all the
    same.  With ``latched`` the flag reads the flag it is handed.
    """

    def __init__(self, name, *, latched, declared=True, flag_dtype=jnp.bool_,
                 in_dtype=jnp.bool_):
        super().__init__(name, 1.0)
        self._latched, self._declared = latched, declared
        self._fd, self._ind = flag_dtype, in_dtype

    def initial_state(self):
        return {"x": _zero(_F32), "flag": _zero(self._fd)}

    def boundary_input_spec(self):
        spec = {"u": BoundaryInputSpec(shape=(), dtype=_F32, default=_zero(_F32))}
        if self._declared:
            spec["signal"] = BoundaryInputSpec(shape=(), dtype=self._ind,
                                               default=_zero(self._ind))
        return spec

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"]
        handed = boundary_inputs.get("signal", _zero(self._ind)) > 0
        flag = (u > 0.3) | handed if self._latched else (u > 0.3)
        x = 0.5 * u + 0.2 + jnp.where(handed, 0.25, 0.0)
        return {"x": x.astype(_F32), "flag": flag.astype(self._fd)}


def _flags_both_ways(*, latched, declared=True, flag_dtypes=(jnp.bool_, jnp.bool_), **group):
    """Two members that hand each other their float and their flag."""
    gm = GraphManager()
    gm.add_node(Flagged("a", latched=latched, declared=declared,
                        flag_dtype=flag_dtypes[0], in_dtype=flag_dtypes[1]))
    gm.add_node(Flagged("b", latched=latched, declared=declared,
                        flag_dtype=flag_dtypes[1], in_dtype=flag_dtypes[0]))
    for src, dst in (("a", "b"), ("b", "a")):
        gm.add_edge(src, dst, "x", "u")
        gm.add_edge(src, dst, "flag", "signal")
    return _group(gm, ["a", "b"], **group)


@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
@pytest.mark.parametrize("solver, mode", [
    ("ift", "gauss-seidel"), ("ift", "jacobi"), ("fori", "gauss-seidel"), ("fori", "jacobi")])
def test_flags_handed_both_ways_but_computed_from_floats_are_solved(solver, mode, declared):
    """Each flag is read by the other member's float, and by no flag: no loop.

    The edges alone would allow a loop (each member reads the other's flag
    and computes one).  The trace shows each flag computed from the float
    input only, so both are carried at depth one, and the group reaches its
    one consistent state: ``x = x / 2 + 0.2 + flag / 4`` with
    ``flag = (x > 0.3)`` holds only at ``x = 0.9`` with the flags set.  The
    same whether or not the nodes declare the input the flag arrives on.
    """
    gm = _flags_both_ways(latched=False, declared=declared, solver=solver, iteration_mode=mode)
    bound, traced = _chains(gm)
    assert bound.loop is not None
    assert traced == _carried_fields.CarriedChain((("a", "flag"), ("b", "flag")), 1, None)
    assert not [issue for issue in gm.validate() if issue.startswith("ERROR")]
    gm.step()
    assert _report(gm)["converged"]
    for name in ("a", "b"):
        assert float(gm.get_node_state(name)["x"]) == pytest.approx(0.9, abs=1e-5)
        assert bool(gm.get_node_state(name)["flag"]) is True


# ---------------------------------------------------------------------------
# A closed loop of non-floating fields is refused
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("declared", [True, False], ids=["declared", "undeclared"])
@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
def test_non_floating_fields_computed_from_one_another_in_a_loop_are_refused(
        solver, mode, declared):
    """Each flag is computed from the flag the other member hands it.

    Such fields have no value at a floating iterate: ``compile()`` says so.
    The message names the fields of the loop with their dtypes, says that
    the loop was read from a trace of the pass, what the iteration would do
    with such fields, and three ways out.  An input the node does not
    declare is read like any other: the pass hands it over.
    """
    gm = _flags_both_ways(latched=True, declared=declared, solver=solver, iteration_mode=mode)
    with pytest.raises(ValueError) as refused:
        gm.compile()
    text = str(refused.value)
    assert "a.flag (bool) -> b.flag (bool) -> a.flag (bool)" in text
    assert "['a', 'b']" in text and "closed loop" in text
    assert "read from one trace of the group's pass" in text and "assumed" not in text
    assert "converged=True while they still change" in text
    assert "floating point" in text and "outside the group" in text
    assert "max_iterations=1" in text


def test_validate_lists_the_refusal_and_every_entry_that_compiles_raises_it():
    """One ``ERROR`` line a group from ``validate()``; ``step`` and ``run_scan`` raise it.

    And a reader of the graph that never compiles it refuses it as
    ``compile()`` does, as for an edge onto a declared external input.
    """
    gm = _flags_both_ways(latched=True)
    errors = [issue for issue in gm.validate() if issue.startswith("ERROR")]
    assert len(errors) == 1 and errors[0].startswith("ERROR: Coupling group ['a', 'b']")
    with pytest.raises(ValueError) as refused:
        gm.compile()
    assert errors[0] == f"ERROR: {refused.value}"
    with pytest.raises(ValueError, match="closed loop"):
        _flags_both_ways(latched=True).step()
    with pytest.raises(ValueError, match="closed loop"):
        _flags_both_ways(latched=True).run_scan(2)
    with pytest.raises(ValueError, match="closed loop"):
        _flags_both_ways(latched=True).resolve_boundary_inputs("a")
    # The same graph with the flags unlatched: nothing listed, the reader reads.
    fine = _flags_both_ways(latched=False)
    assert not [issue for issue in fine.validate() if issue.startswith("ERROR")]
    assert set(fine.resolve_boundary_inputs("a")) == {"u", "signal"}


def test_each_way_out_the_refusal_names_compiles_and_steps():
    """One field of the loop in floating point; or one staggered pass."""
    as_float = _flags_both_ways(latched=True, flag_dtypes=(jnp.bool_, _F32))
    as_float.step()
    assert _report(as_float)["converged"]
    staggered = _flags_both_ways(latched=True, max_iterations=1)
    assert not [issue for issue in staggered.validate() if issue.startswith("ERROR")]
    staggered.step()
    assert int(_report(staggered)["iterations"]) == 1


def test_a_chain_that_does_not_close_is_not_a_loop():
    """Flags handed one way round an open chain are settled, not refused."""
    gm = _floor_loop(reader_first=True, relays=2)
    gm.compile()            # does not raise


class Untraceable(Flagged):
    """:class:`Flagged`, whose ``update`` raises while ``broken`` is set."""

    broken = False

    def update(self, state, boundary_inputs, dt, *, params=None):
        if type(self).broken:
            raise RuntimeError("this hook cannot be traced just now")
        return super().update(state, boundary_inputs, dt, params=params)


def test_where_the_pass_cannot_be_traced_the_refusal_says_the_loop_is_assumed(monkeypatch):
    """The one case in which dependence is assumed, and the message says so.

    The flags here are computed from floats alone (no loop: the group of
    :func:`test_flags_handed_both_ways_but_computed_from_floats_are_solved`).
    While the nodes' ``update`` raises under the trace, what the pass reads
    cannot be established; the bound from the edges stands, and the text
    names the exception and says the loop may not be one.
    """
    gm = GraphManager()
    for name in ("a", "b"):
        gm.add_node(Untraceable(name, latched=False))
    for src, dst in (("a", "b"), ("b", "a")):
        gm.add_edge(src, dst, "x", "u")
        gm.add_edge(src, dst, "flag", "signal")
    _group(gm, ["a", "b"], solver="ift")
    assert not [issue for issue in gm.validate() if issue.startswith("ERROR")]
    monkeypatch.setattr(Untraceable, "broken", True)
    (error,) = [issue for issue in gm.validate() if issue.startswith("ERROR")]
    assert "a.flag (bool) -> b.flag (bool) -> a.flag (bool)" in error
    assert "could not be read from the group's pass" in error
    assert "RuntimeError: this hook cannot be traced just now" in error
    assert "assumed from the edges" in error and "the loop may not be one" in error
    assert "read from one trace" not in error


def test_a_group_whose_edges_allow_no_carried_field_is_not_traced(monkeypatch):
    """Nothing is traced, at ``compile()`` or in the step, where the edges settle it.

    A counter beside a float edge is read by nothing a pass hands on; the
    analysis stops at the edges and the step is the program it was.
    """
    calls = []
    real = _carried_fields.traced_reads
    monkeypatch.setattr(_carried_fields, "traced_reads",
                        lambda *a, **k: calls.append(1) or real(*a, **k))

    class Counting(SimulationNode):
        def __init__(self, name):
            super().__init__(name, 1.0)

        def initial_state(self):
            return {"x": _zero(_F32), "n": _zero(jnp.int32)}

        def boundary_input_spec(self):
            return {"u": BoundaryInputSpec(shape=(), dtype=_F32, default=_zero(_F32))}

        def update(self, state, boundary_inputs, dt, *, params=None):
            return {"x": (1.0 + 0.25 * boundary_inputs["u"]).astype(_F32), "n": state["n"] + 1}

    gm = GraphManager()
    gm.add_node(Counting("a"))
    gm.add_node(Counting("b"))
    gm.add_edge("a", "b", "x", "u")
    gm.add_edge("b", "a", "x", "u")
    _group(gm, ["a", "b"])
    gm.step()
    assert calls == [] and int(gm.get_node_state("a")["n"]) == 1
    # ... and the patch does take effect where a field may be carried.
    _flags_both_ways(latched=False).step()
    assert calls
