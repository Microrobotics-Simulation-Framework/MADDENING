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

from maddening.core.coupling import _carried_fields, _interface_plan
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
    gm.compile()
    plan = _interface_plan.interface_plan(
        gm._coupling_groups[0].nodes, gm._edges, sorted(gm._nodes), gm._state, gm._nodes)
    chain = _carried_fields.carried_nonfloat_chain(plan, gm._state)
    assert chain.loop is None and chain.depth == depth
    assert chain.fields == tuple(sorted(
        [("b", "p"), *((f"r{i + 1}", "p") for i in range(relays))]))


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
    """Each acceleration has a second loop body, taken with ``diagnostics=False``."""
    with_report = _floor_result("fori", "jacobi", False, **dict(accel))
    gm = _floor_loop(reader_first=False, solver="fori", iteration_mode="jacobi",
                     diagnostics=False, **dict(accel))
    gm.step()
    assert float(gm.get_node_state("a")["q"]) == pytest.approx(with_report[0], abs=1e-6)
    assert int(gm.get_node_state("b")["p"]) == with_report[1]
    if not accel:
        assert float(gm.get_node_state("a")["q"]) == Q_STAR


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
# A closed loop of non-floating fields is refused
# ---------------------------------------------------------------------------

class Latch(SimulationNode):
    """``flag <- flag_in or (u > 0)``; ``x <- u``: a flag computed from a flag it reads."""

    def __init__(self, name, flag_dtype=jnp.bool_, in_dtype=jnp.bool_):
        super().__init__(name, 1.0)
        self._fd, self._ind = flag_dtype, in_dtype

    def initial_state(self):
        return {"x": _zero(_F32), "flag": _zero(self._fd)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(), dtype=_F32, default=_zero(_F32)),
                "flag_in": BoundaryInputSpec(shape=(), dtype=self._ind,
                                             default=_zero(self._ind))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"]
        seen = (boundary_inputs["flag_in"] > 0) | (u > 0)
        return {"x": (0.5 * u).astype(_F32), "flag": seen.astype(self._fd)}


def _latched(flag_dtypes=(jnp.bool_, jnp.bool_), **group):
    gm = GraphManager()
    gm.add_node(Latch("a", flag_dtypes[0], flag_dtypes[1]))
    gm.add_node(Latch("b", flag_dtypes[1], flag_dtypes[0]))
    for src, dst in (("a", "b"), ("b", "a")):
        gm.add_edge(src, dst, "x", "u")
        gm.add_edge(src, dst, "flag", "flag_in")
    return _group(gm, ["a", "b"], **group)


@pytest.mark.parametrize("solver", ["ift", "fori"])
@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
def test_non_floating_fields_that_feed_one_another_in_a_loop_are_refused(solver, mode):
    """Such fields have no value at a floating iterate: ``compile()`` says so.

    The message names the fields of the loop with their dtypes, what the
    iteration would do with them, and three ways out.
    """
    gm = _latched(solver=solver, iteration_mode=mode)
    with pytest.raises(ValueError) as refused:
        gm.compile()
    text = str(refused.value)
    assert "a.flag (bool) -> b.flag (bool) -> a.flag (bool)" in text
    assert "['a', 'b']" in text and "closed loop" in text
    assert "converged=True while they still change" in text
    assert "floating point" in text and "outside the group" in text
    assert "max_iterations=1" in text


def test_the_refusal_is_raised_by_every_entry_that_compiles():
    with pytest.raises(ValueError, match="closed loop"):
        _latched().step()
    with pytest.raises(ValueError, match="closed loop"):
        _latched().run_scan(2)


def test_each_way_out_the_refusal_names_compiles_and_steps():
    """One field of the loop in floating point; or one staggered pass."""
    as_float = _latched(flag_dtypes=(jnp.bool_, _F32))
    as_float.step()
    assert _report(as_float)["converged"]
    staggered = _latched(max_iterations=1)
    staggered.step()
    assert int(_report(staggered)["iterations"]) == 1


def test_a_chain_that_does_not_close_is_not_a_loop():
    """Flags handed one way round an open chain are settled, not refused."""
    gm = _floor_loop(reader_first=True, relays=2)
    gm.compile()            # does not raise
