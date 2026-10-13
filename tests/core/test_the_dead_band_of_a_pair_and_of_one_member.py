"""A dead band hides a change on a pair, and on one member, under Gauss-Seidel too.

A field at or below ``atol`` leaves the residual (CPL-011), and the loop
accepts on the first residual it measures, between its first two passes.
A change that has to cross such a field is therefore not in the residual
until it reaches a field the norm keeps.  That takes TWO lagged reads in
series between the kept field and itself.  A Gauss-Seidel pair has one
lagged read each turn of its loop (the first member swept reads the
second's previous iterate), so a loop that passes that read twice has
the two, and so has an edge from a member to itself:

* **a pair of 2-vectors**: one field a member, one plain edge each way,
  the entries crossed::

      A.x = [f(u[0]), u[1]]          u = B.y
      B.y = [G(v[1]), v[0]]          v = A.x

* **a pair with two fields a member** (the loop wound twice)::

      A.x1 <- f(B.y1)   B.y2 <- A.x1   A.x2 <- B.y2   B.y1 <- G(A.x2)

* **a pair with one edge from a member to itself**::

      A.x <- f(B.y)     B.w <- A.x     B.y <- G(B.w)

* **ONE member with two edges to itself**::

      A.p <- f(A.q)     A.q <- G(A.p)

with a force ``f(y) = S (1 + C y)`` of about 1e-8 (``S = 1e-8``, inside
``atol = 1e-6``) and the one kept field ``G(force) = b_n + (K / S)
force``, of order one.  The problem moves from step to step, as any
time-dependent one does: ``b_n = 1 + n / 2`` at step ``n`` (an integer
step counter in the member's state; integers are in no norm).  The fixed
point of the kept field at step ``n`` is closed form, ``(b_n + K) / (1 -
K C)``, with loop gain ``K C = 0.4``.

What happens with the band declared: the first pass of a step moves the
kept field half a turn of the loop towards the new fixed point, the
second pass moves only fields inside the band, the kept field's residual
is exactly zero, and the loop accepts.  This is the dead band's own
behaviour under every norm (MADD-ANO-254, open; the redesign of ``atol``
is 0.5.0's).  What is held here, for each of the four shapes and both
sweep orders of the three pairs:

* ``validate()`` gives the advisory and ``compile()`` warns, once;
* the defect stays visible: a strict xfail of the claim "a converged
  step leaves the kept field within ``PROMISE`` tolerances of its fixed
  point", and its premise stated as what happens today;
* its control passes: the same graph at ``atol = 0`` holds;
* a pair with ONE scalar field a member and no edge to itself held with
  the band declared, in both sweep orders: a measured instance, not a
  rule, and it is advised on all the same.
"""

from __future__ import annotations

import functools
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling import _group_layout
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.sparse_mapping_support import x64

ATOL = 1e-6
#: The size of a force: inside the dead band.
S = 1e-8
#: How a force responds to the kept field it reads.
C = 0.5
#: The kept field's response to a force, in units of the force.
K = 0.8
#: ``rtol`` (``tolerance`` under ``"l2"``) in each dtype.
RTOL = {"float32": 1e-5, "float64": 1e-9}
STEPS = 6
NORMS = ("mixed", "l2", "interface")
#: The norm the per-push tests run (the three give the same numbers here:
#: one field is kept, and its one entry of order one carries the norm).
PER_PUSH_NORM = "mixed"

#: The claim, in tolerances: a converged step leaves the kept field this
#: close to its fixed point.  Measured without the band (jax 0.11.0,
#: float32): 0.18 to 1.6 on the four shapes.
PROMISE = 10.0

#: What a banded group is pinned at: more than this many tolerances off
#: on some converged one-pass step.  Measured (CPU, jax 0.11.0, float32,
#: ``rtol = 1e-5``, the same under the three norms): 4.35e3, 5.56e3 and
#: 7.69e3 on the one-pass steps of every shape swept A then B, and
#: 4.88e3, 6.45e3 and 1.56e5 swept B then A; in float64 at ``rtol =
#: 1e-9``, 6.45e7 to 1.56e9.  1e3 is a loose lower bound on the smallest:
#: a factor 4 of room for another jax or platform, and still 100
#: promises.
PINNED_OFF = 1e3

#: ``(shape, the order the members are added in)``: the four shapes, and
#: both sweep orders where there are two members.
CELLS = (
    ("pair of 2-vectors", "AB"), ("pair of 2-vectors", "BA"),
    ("pair with two fields a member", "AB"), ("pair with two fields a member", "BA"),
    ("pair with one self-edge", "AB"), ("pair with one self-edge", "BA"),
    ("one member with two self-edges", "A"),
)
#: The control that held with the band declared: one scalar field a
#: member, one plain edge each way, no edge from a member to itself.
SCALAR_PAIR = "pair with one scalar field a member"


def _cell_id(cell) -> str:
    shape, order = cell
    return f"{shape}, swept {order}"


class VectorForce(SimulationNode):
    """``x = [S (1 + C u[0]), u[1]]``: a force, and a relay of what it reads."""

    def __init__(self, name: str, dtype: str):
        super().__init__(name, 1.0)
        self._dtype = dtype

    def initial_state(self):
        return {"x": jnp.zeros(2, self._dtype)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(2,), dtype=jnp.dtype(self._dtype),
                                       default=jnp.zeros(2, self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        u = boundary_inputs["u"]
        force = jnp.asarray(S, self._dtype) * (1.0 + jnp.asarray(C, self._dtype) * u[0])
        return {"x": jnp.stack([force, u[1]])}


class VectorKept(SimulationNode):
    """``y = [b_n + (K / S) v[1], v[0]]``: the kept entry, and a relay."""

    def __init__(self, name: str, dtype: str):
        super().__init__(name, 1.0)
        self._dtype = dtype

    def initial_state(self):
        return {"y": jnp.zeros(2, self._dtype), "n": jnp.zeros((), jnp.int32)}

    def boundary_input_spec(self):
        return {"v": BoundaryInputSpec(shape=(2,), dtype=jnp.dtype(self._dtype),
                                       default=jnp.zeros(2, self._dtype))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        v = boundary_inputs["v"]
        n = state["n"] + 1
        kept = (1.0 + 0.5 * n.astype(self._dtype)) + jnp.asarray(K / S, self._dtype) * v[1]
        return {"y": jnp.stack([kept, v[0]]), "n": n}


class Fields(SimulationNode):
    """One field of one entry per rule, each computed from the input of
    its name: ``"force"`` is ``S (1 + C u)``, ``"relay"`` is ``u`` and
    ``"kept"`` is ``b_n + (K / S) u`` with the step counter ``n``."""

    def __init__(self, name: str, dtype: str, rules: dict):
        super().__init__(name, 1.0)
        self._dtype, self._rules = dtype, dict(rules)

    def initial_state(self):
        return {**{field: jnp.zeros(1, self._dtype) for field in self._rules},
                "n": jnp.zeros((), jnp.int32)}

    def boundary_input_spec(self):
        return {"in_" + field: BoundaryInputSpec(shape=(1,), dtype=jnp.dtype(self._dtype),
                                                 default=jnp.zeros(1, self._dtype))
                for field in self._rules}

    def update(self, state, boundary_inputs, dt, *, params=None):
        n = state["n"] + 1
        out = {"n": n}
        for field, rule in self._rules.items():
            u = boundary_inputs["in_" + field]
            if rule == "force":
                out[field] = jnp.asarray(S, self._dtype) * (1.0 + jnp.asarray(C, self._dtype) * u)
            elif rule == "relay":
                out[field] = u
            else:
                out[field] = ((1.0 + 0.5 * n.astype(self._dtype))
                              + jnp.asarray(K / S, self._dtype) * u)
        return out


def graph(shape: str, order: str, norm: str, atol: float, *, acceleration: str = "none",
          schedule: str = "gauss-seidel", dtype: str = "float32"):
    """``(the graph, not compiled; (member, field) of the kept field)``;
    *order* is the order the members are added in (the sweep's)."""
    if shape == "pair of 2-vectors":
        nodes = {"A": VectorForce("A", dtype), "B": VectorKept("B", dtype)}
        edges = [("B", "A", "y", "u"), ("A", "B", "x", "v")]
        kept = ("B", "y")
    elif shape == "pair with two fields a member":
        nodes = {"A": Fields("A", dtype, {"x1": "force", "x2": "relay"}),
                 "B": Fields("B", dtype, {"y2": "relay", "y1": "kept"})}
        edges = [("B", "A", "y1", "in_x1"), ("A", "B", "x1", "in_y2"),
                 ("B", "A", "y2", "in_x2"), ("A", "B", "x2", "in_y1")]
        kept = ("B", "y1")
    elif shape == "pair with one self-edge":
        nodes = {"A": Fields("A", dtype, {"x": "force"}),
                 "B": Fields("B", dtype, {"w": "relay", "y": "kept"})}
        edges = [("B", "A", "y", "in_x"), ("A", "B", "x", "in_w"), ("B", "B", "w", "in_y")]
        kept = ("B", "y")
    elif shape == "one member with two self-edges":
        nodes = {"A": Fields("A", dtype, {"p": "force", "q": "kept"})}
        edges = [("A", "A", "q", "in_p"), ("A", "A", "p", "in_q")]
        kept = ("A", "q")
    elif shape == SCALAR_PAIR:
        nodes = {"A": Fields("A", dtype, {"x": "force"}), "B": Fields("B", dtype, {"y": "kept"})}
        edges = [("B", "A", "y", "in_x"), ("A", "B", "x", "in_y")]
        kept = ("B", "y")
    else:
        raise ValueError(shape)
    assert sorted(order) == sorted(nodes), (shape, order)
    gm = GraphManager()
    for name in order:
        gm.add_node(nodes[name])
    for source, target, source_field, target_field in edges:
        gm.add_edge(source, target, source_field, target_field)
    tolerance = {"tolerance": RTOL[dtype]} if norm == "l2" else {"rtol": RTOL[dtype]}
    gm.add_coupling_group(list(order), convergence_norm=norm, atol=atol,
                          iteration_mode=schedule, max_iterations=200, solver="ift",
                          acceleration=acceleration, **tolerance)
    return gm, kept


def _advisories(gm: GraphManager) -> list:
    return [issue for issue in gm.validate() if _group_layout._DEAD_BAND_ADVISORY in issue]


def _compile(gm: GraphManager, advised: bool) -> None:
    """Compile, expecting the dead-band advisory exactly where it is due."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gm.compile()
    ours = [w for w in caught if _group_layout._DEAD_BAND_ADVISORY in str(w.message)]
    assert len(ours) == (1 if advised else 0), [str(w.message) for w in caught]
    assert all(issubclass(w.category, UserWarning) for w in ours)


def _fixed_point(n: int) -> float:
    """The kept field at step *n*: ``y = b_n + K (1 + C y)``."""
    return (1.0 + 0.5 * n + K) / (1.0 - K * C)


@functools.lru_cache(maxsize=None)
def run(shape: str, order: str, norm: str, atol: float, acceleration: str = "none",
        schedule: str = "gauss-seidel", dtype: str = "float32") -> tuple:
    """``(iterations, converged, distance of the kept field in
    tolerances)`` for each of ``STEPS`` steps of a freshly built graph,
    compiled expecting the advisory exactly where the band is declared.
    The numbers only are kept: no compiled graph outlives the call."""
    with x64(dtype == "float64"):
        gm, (member, field) = graph(shape, order, norm, atol, acceleration=acceleration,
                                    schedule=schedule, dtype=dtype)
        _compile(gm, advised=atol > 0)
        out = []
        for n in range(1, STEPS + 1):
            gm.step()
            (report,) = gm.coupling_diagnostics().values()
            y = float(np.asarray(gm.get_node_state(member)[field], np.float64)[0])
            off = abs(y - _fixed_point(n)) / (RTOL[dtype] * abs(y))
            out.append((int(report["iterations"]), bool(report["converged"]), off))
    return tuple(out)


def _held(steps: tuple) -> bool:
    """Every converged step left the kept field within the promise."""
    return all(off < PROMISE for _iterations, converged, off in steps if converged)


def _pinned(steps: tuple) -> bool:
    """Some converged step accepted after ONE pass far outside the promise."""
    return any(converged and iterations == 1 and off > PINNED_OFF
               for iterations, converged, off in steps)


_XFAIL = pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MADD-ANO-254 (open): under Gauss-Seidel a pair whose loop passes its lagged read twice, "
    "and one member with two edges to itself, accept after one pass 4.3e3 to 1.6e5 "
    "tolerances off"))


# ---------------------------------------------------------------------------
# The advisory, on every shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("norm", NORMS)
@pytest.mark.parametrize("cell", CELLS + ((SCALAR_PAIR, "AB"), (SCALAR_PAIR, "BA")), ids=_cell_id)
def test_validate_advises_on_the_dead_band_of_a_pair_and_of_one_member(cell, norm):
    """One ``WARNING:`` line under Gauss-Seidel for a pair and for one
    member, naming the group, its ``atol``, the member count in English,
    the measured pairs, that no group is excepted and the one way out;
    none at the default ``atol``.  The scalar pair that held is advised
    on like the others: the condition reads nothing from the graph."""
    shape, order = cell
    members = sorted(order)
    (line,) = _advisories(graph(shape, order, norm, ATOL)[0])
    count = "1 member " if len(members) == 1 else "2 members "
    assert line.startswith(f"WARNING: coupling group {members} declares a dead band on {count}(")
    assert "atol=1e-06" in line and "iteration_mode='gauss-seidel'" in line
    assert "under Gauss-Seidel a pair of 2-vectors" in line and "one member with two" in line
    assert "4.3e3 to 7.7e3" in line and "1.6e5 in the other sweep order" in line
    assert "No group is excepted" in line and "is not affected" not in line
    assert "a ring of three" not in line and "under Jacobi" not in line and "Use " not in line
    assert "Set atol=0.0 (the default) on this group" in line and "MADD-ANO-254" in line
    assert _advisories(graph(shape, order, norm, 0.0)[0]) == []


@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_compile_warns_once_for_the_dead_band_of_a_pair_and_of_one_member(cell):
    """``compile()`` emits the advisory as one ``UserWarning`` for each
    shape and sweep order, under the opening that names the member count,
    and none at the default ``atol``."""
    shape, order = cell
    with pytest.warns(UserWarning, match=_group_layout._DEAD_BAND_ADVISORY) as caught:
        graph(shape, order, PER_PUSH_NORM, ATOL)[0].compile()
    ours = [str(w.message) for w in caught if _group_layout._DEAD_BAND_ADVISORY in str(w.message)]
    assert len(ours) == 1
    assert f"{_group_layout._DEAD_BAND_ON_MEMBERS} {len(order)} member" in ours[0]
    assert _group_layout._DEAD_BAND_UNDER_JACOBI not in ours[0]
    _compile(graph(shape, order, PER_PUSH_NORM, 0.0)[0], advised=False)


def test_one_member_under_jacobi_is_advised_on_in_the_singular():
    """The other opening, for the group of one member: it names the
    schedule, and "1 member"."""
    (line,) = _advisories(graph("one member with two self-edges", "A", PER_PUSH_NORM, ATOL,
                                schedule="jacobi")[0])
    assert line.startswith("WARNING: coupling group ['A'] "
                           + _group_layout._DEAD_BAND_UNDER_JACOBI + " (atol=1e-06, 1 member):")


# ---------------------------------------------------------------------------
# The claim that does not hold today, and its premise
# ---------------------------------------------------------------------------

@_XFAIL
@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_a_converged_step_of_a_banded_pair_or_member_holds_its_kept_field(cell):
    """What CPL-011's oracle asks of the kept field, and today does not
    hold on any of the four shapes, in either sweep order."""
    steps = run(*cell, PER_PUSH_NORM, ATOL)
    assert all(converged for _i, converged, _off in steps), steps
    assert _held(steps), steps


@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_the_banded_pair_or_member_accepts_after_one_pass_far_from_its_fixed_point(cell):
    """The premise of the strict xfail, as what happens today (a
    characterisation of MADD-ANO-254).  When this stops being true the
    xfail above turns into a pass and both are to be rewritten."""
    steps = run(*cell, PER_PUSH_NORM, ATOL)
    assert all(converged for _i, converged, _off in steps), steps
    assert _pinned(steps), steps


# Per push: tests/core/test_the_dead_band_of_a_pair_and_of_one_member.py::test_a_converged_step_of_a_banded_pair_or_member_holds_its_kept_field
@pytest.mark.slow
@_XFAIL
@pytest.mark.parametrize("norm", [n for n in NORMS if n != PER_PUSH_NORM])
@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_a_converged_step_of_a_banded_pair_or_member_holds_under_the_other_norms(cell, norm):
    """The same claim under ``"l2"`` and ``"interface"``: neither holds."""
    steps = run(*cell, norm, ATOL)
    assert all(converged for _i, converged, _off in steps), steps
    assert _held(steps), steps


# Per push: tests/core/test_the_dead_band_of_a_pair_and_of_one_member.py::test_the_banded_pair_or_member_accepts_after_one_pass_far_from_its_fixed_point
@pytest.mark.slow
@pytest.mark.parametrize("norm", [n for n in NORMS if n != PER_PUSH_NORM])
@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_the_banded_pair_or_member_accepts_after_one_pass_under_the_other_norms(cell, norm):
    """The premise in every cell of the slow xfail above."""
    steps = run(*cell, norm, ATOL)
    assert all(converged for _i, converged, _off in steps), steps
    assert _pinned(steps), steps


# Per push: tests/core/test_the_dead_band_of_a_pair_and_of_one_member.py::test_the_banded_pair_or_member_accepts_after_one_pass_far_from_its_fixed_point
@pytest.mark.slow
@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_the_banded_pair_or_member_accepts_after_one_pass_in_float64_and_under_iqn_ils(cell):
    """The same acceptance in float64 at ``rtol = 1e-9`` (measured 6.45e7
    to 1.56e9 tolerances off: the distance is the same and the tolerance
    1e4 times smaller), and in float32 with IQN-ILS (which accepts after
    one pass as far off as the plain loop; one cell, two fields a member
    swept B then A, also has a step that IQN-ILS does not converge on,
    so only the acceptance is asked of it)."""
    steps = run(*cell, PER_PUSH_NORM, ATOL, dtype="float64")
    assert all(converged for _i, converged, _off in steps), steps
    assert _pinned(steps) and max(off for _i, _c, off in steps) > 1e4 * PINNED_OFF, steps
    assert _pinned(run(*cell, PER_PUSH_NORM, ATOL, acceleration="iqn-ils"))


# ---------------------------------------------------------------------------
# The controls
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", CELLS, ids=_cell_id)
def test_the_same_pair_or_member_with_no_dead_band_is_held(cell):
    """``atol = 0`` keeps the forces of 1e-8 in the residual (only a field
    that is exactly zero leaves it), and every step takes its passes to
    the fixed point."""
    steps = run(*cell, PER_PUSH_NORM, 0.0)
    assert all(converged for _i, converged, _off in steps), steps
    assert _held(steps) and min(i for i, _c, _off in steps) > 3, steps


@pytest.mark.parametrize("order", ["AB", "BA"], ids=["the force read from the previous pass",
                                                     "the force read in the same sweep"])
def test_a_pair_with_one_scalar_field_a_member_held_with_the_dead_band_declared(order):
    """One measured instance, not a rule: the loop of this pair passes
    its one lagged read once, and both sweep orders held with the band
    declared (0.41 to 0.75 tolerances).  It is advised on all the same
    (``run`` expects the warning): which groups have a single lagged
    read is a criterion nobody has proved."""
    steps = run(SCALAR_PAIR, order, PER_PUSH_NORM, ATOL)
    assert all(converged for _i, converged, _off in steps), steps
    assert _held(steps) and min(i for i, _c, _off in steps) > 3, steps


def test_the_scalar_pair_under_jacobi_fails_as_the_four_shapes_do():
    """The mechanism, on the pair that held: under Jacobi both of its
    reads are lagged, and it accepts after one pass on the same steps and
    at the same distances as the four shapes do under Gauss-Seidel."""
    jacobi = run(SCALAR_PAIR, "AB", PER_PUSH_NORM, ATOL, schedule="jacobi")
    assert _pinned(jacobi), jacobi
    swept = run("pair of 2-vectors", "AB", PER_PUSH_NORM, ATOL)
    assert [(i, c) for i, c, _off in jacobi] == [(i, c) for i, c, _off in swept], (jacobi, swept)
    np.testing.assert_allclose([off for _i, _c, off in jacobi], [off for _i, _c, off in swept],
                               rtol=0.05)


# Per push: tests/core/test_the_dead_band_of_a_pair_and_of_one_member.py::test_the_same_pair_or_member_with_no_dead_band_is_held
@pytest.mark.slow
@pytest.mark.parametrize("norm", [n for n in NORMS if n != PER_PUSH_NORM])
def test_the_controls_hold_under_the_other_norms(norm):
    """No dead band on the four shapes; the scalar pair with the band, in
    both sweep orders."""
    for shape, order in CELLS:
        steps = run(shape, order, norm, 0.0)
        assert all(converged for _i, converged, _off in steps), (shape, order, steps)
        assert _held(steps), (shape, order, steps)
    for order in ("AB", "BA"):
        steps = run(SCALAR_PAIR, order, norm, ATOL)
        assert all(converged for _i, converged, _off in steps), (order, steps)
        assert _held(steps), (order, steps)
