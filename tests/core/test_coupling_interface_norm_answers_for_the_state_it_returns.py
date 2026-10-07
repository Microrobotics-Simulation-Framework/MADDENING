"""``converged=True`` under ``convergence_norm="interface"`` is a statement about the state returned.

The interface norm measures, on each pass, how far what the group's
internal edges deliver moved between an iterate and its successor.  The
members of that iterate were computed from the readings of the one before
it, so a floating field the norm does not measure whole can hold a value
those settled readings never produced:

* a field **no internal edge reads**.  A one-way pair ``A -> B`` under
  Jacobi stopped on its first pass -- ``A`` does not depend on the iterate,
  so what it delivers "stopped moving" at once, with a residual of exactly
  zero -- and returned ``B`` computed from the *pre-step* ``A``, with
  ``converged=True`` and ``iterations=1`` (MADD-ANO-235; 0.1.0 to 0.3.1).
  A weakly coupled pair returned such a field a few hundred tolerances
  off, under Gauss-Seidel too (the member that reads a back edge), and a
  fixed relaxation left it a blend of every pass so far.  The solve now
  returns these fields as one pass computes them at the state it returns
  -- from the readings the verdict was taken on -- and changes nothing
  else: iterates, residuals and pass counts are the ones they were;
* a field edges read **only through a mapping or a transform** that
  delivers less than the field.  It feeds back, so it is not recomputed;
  where its member reads the previous iterate the solve stops only once
  the readings moved within the threshold over the pass that computed the
  state as well, and at the cap reports the larger change.  (Under a
  relaxation that is not enough -- MADD-ANO-236, open, pinned below.)

Every oracle here is a float64 closed form of the linear map the graph
holds; none calls the code under test.  The static rule
(``_fields_the_interface_norm_misses``) is checked on its own, branch by
branch, and the loop's exit against a scripted residual sequence.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.core.coupling._fixed_point import _fixed_point_while
from maddening.core.coupling._group_layout import _fields_the_interface_norm_misses
from maddening.core.coupling.group import CouplingGroup
from maddening.core.coupling.mapping import matrix_mapping
from maddening.core.edge import EdgeSpec
from maddening.core.graph_manager import GraphManager
from maddening.core.node import BoundaryInputSpec, SimulationNode
from tests.core import coupling_domains as cd

RTOL = 1e-4


class _Lin(SimulationNode):
    """``u <- alpha u_pre + c + sum_p G_p @ inp_p``; with *wide*, also
    ``w <- 1 + sum_p K_p @ inp_p``, a field no edge reads."""

    def __init__(self, name, n, ports, alpha, c, u0, dtype, wide):
        super().__init__(name, 1.0)
        self._n, self._dtype, self._wide = n, dtype, wide
        self._ports = {p: (np.asarray(G, dtype), np.asarray(K, dtype))
                       for p, (G, K) in ports.items()}
        self._alpha = dtype(alpha)
        self._c = np.asarray(c, dtype)
        self._u0 = np.asarray(u0, dtype)

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
D = ("D", 2, 0.0, [0.5, 0.5], [1.0, 1.0])
G = np.array([[0.7, 0.3], [-0.4, 0.9]])
#: The gain of the unread field ``w``: order one, so a reading one pass old
#: shows in it at the size of that reading's change.
K = np.array([[2.0, 1.0], [1.0, -2.0]])
H = np.array([[1.0, 0.5], [-0.25, 1.5]])
#: A mapping that delivers the first entry of a field of two.
SELECT = np.array([[1.0, 0.0]])


def _graph(nodes, edges, group, *, dtype=np.float32, wide=(), compile_=True):
    """*edges*: ``(src, dst, G, how)`` with *how* ``None`` (plain), a matrix
    (the edge's mapping) or a callable (its transform)."""
    ports = {name: {} for name, *_ in nodes}
    for i, (_src, dst, gain, _how) in enumerate(edges):
        gain = np.asarray(gain, np.float64)
        ports[dst][f"p{i}"] = (gain, K[:, :gain.shape[1]])
    gm = GraphManager()
    for name, n, alpha, c, u0 in nodes:
        gm.add_node(_Lin(name, n, ports[name], alpha, c, u0, dtype, name in wide))
    for i, (src, dst, _gain, how) in enumerate(edges):
        extra = {}
        if callable(how):
            extra["transform"] = how
        elif how is not None:
            extra["mapping"] = matrix_mapping(np.asarray(how, dtype))
        gm.add_edge(src, dst, "u", f"p{i}", **extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")         # the deprecated "fori" says so
        gm.add_coupling_group([name for name, *_ in nodes], diagnostics=True, **group)
        if compile_:
            gm.compile()
    return gm


def _delivers(how, n):
    """The matrix an edge applies to its source field of *n* entries."""
    if how is None:
        return np.eye(n)
    if callable(how):
        return np.eye(n)[:1]            # ``_first``
    return np.asarray(how, np.float64)


def _fixed_point(nodes, edges, wide=()):
    """``{node: {field: value}}`` with every edge reading the new value, in float64."""
    off, k = {}, 0
    for name, n, *_ in nodes:
        off[name] = k
        k += n
    size = {name: n for name, n, *_ in nodes}
    M, b = np.zeros((k, k)), np.zeros(k)
    for name, n, alpha, c, u0 in nodes:
        b[off[name]:off[name] + n] = alpha * np.asarray(u0, float) + np.asarray(c, float)
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


def _distance(gm, want):
    """The worst field's ``max |x - x*| / max |x*|``, and which field it is."""
    worst, where = 0.0, None
    for name, fields in want.items():
        state = gm.get_node_state(name)
        for f, v in fields.items():
            d = float(np.max(np.abs(np.asarray(state[f], np.float64) - v)) / np.max(np.abs(v)))
            if d > worst:
                worst, where = d, f"{name}.{f}"
    return worst, where


def _step(nodes, edges, group, *, x64=False, wide=()):
    """``(distance to the fixed point, where, report)`` of one step from the start."""
    with cd.x64(x64):
        gm = _graph(nodes, edges, group, dtype=np.float64 if x64 else np.float32, wide=wide)
        gm.step()
        (report,) = gm.coupling_diagnostics().values()
        distance, where = _distance(gm, _fixed_point(nodes, edges, wide))
    return distance, where, report


def _eps(x64):
    return float(np.finfo(np.float64 if x64 else np.float32).eps)


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


@pytest.mark.parametrize("group", [
    dict(acceleration="fixed", relaxation=0.5),
    dict(acceleration="fixed", relaxation=1.3),
    dict(acceleration="aitken"),
    dict(acceleration="iqn-ils"),
    dict(acceleration="iqn-imvj", jacobian_reuse=2),
], ids=lambda g: f"{g['acceleration']}{g.get('relaxation', '')}")
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_no_acceleration_leaves_a_one_way_target_between_passes(solver, group):
    """A relaxation blends every pass so far into a field nothing measures:
    at 0.5 the target was half-way from the stale value to the right one,
    6800 tolerances off, with the two-pass exit satisfied."""
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


def test_the_refreshed_field_carries_the_fixed_points_derivative():
    """``dB/dA_pre`` through the step is the closed form's: the field is
    recomputed outside the implicit rule and differentiated as a pass is."""
    gm = _graph([A, B], [("A", "B", G, None)],
                dict(iteration_mode="jacobi", convergence_norm="interface", rtol=RTOL))
    gm.step()
    gm.reset_state()
    compiled, base = gm._compiled_step, gm._state       # noqa: SLF001

    def target(a_pre):
        state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        state["A"]["u"] = a_pre
        return compiled(state, {})["B"]["u"]

    jac = jax.jacfwd(target)(jnp.asarray(A[4], jnp.float32))
    np.testing.assert_allclose(np.asarray(jac), A[2] * G, rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# A field edges read only through a mapping or a transform
# ---------------------------------------------------------------------------

#: ``B``'s first entry does not depend on what ``B`` reads; its second does.
G_SECOND = np.array([[0.0, 0.0], [-0.4, 0.9]])


@pytest.mark.parametrize("how", [SELECT, _first], ids=["mapping", "transform"])
@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_the_part_of_a_field_its_edge_does_not_deliver_is_not_left_a_pass_behind(
        solver, how):
    """``A -> B -> C`` with ``B`` delivered through a selection of its first
    entry, which no input moves: what ``B`` delivers is settled from the
    first pass, a residual of exactly zero, while its second entry still
    held the pre-step ``A``.  The same stale member, behind a mapping."""
    edges = [("A", "B", G_SECOND, None), ("B", "C", G[:, :1], how)]
    distance, where, report = _step(
        [A, B, C], edges, dict(iteration_mode="jacobi", convergence_norm="interface",
                               rtol=RTOL, solver=solver))
    assert report["converged"] is True
    assert distance <= 16 * _eps(False), (where, distance, report["iterations"])


#: Each member's first entry reads the other's at 0.03 (the loop the norm
#: watches gains a digit and a half per pass); its second entry, which no
#: edge delivers, reads the other's first at order one.
G_A = np.array([[0.03], [1.0]])
G_B = np.array([[0.03], [-0.7]])


@pytest.mark.xfail(strict=True, reason=(
    "MADD-ANO-236 (open): a relaxation blends every pass so far into the part of a field "
    "its edges do not deliver, and the interface norm measures none of it"))
def test_a_relaxation_does_not_leave_the_undelivered_part_of_a_field_between_passes():
    """The same chain under ``acceleration="fixed"`` at 0.5: the readings are
    settled on both passes the exit asks for, and ``B``'s second entry is
    half-way from the pre-step value to the right one."""
    edges = [("A", "B", G_SECOND, None), ("B", "C", G[:, :1], SELECT)]
    distance, where, report = _step(
        [A, B, C], edges, dict(iteration_mode="jacobi", convergence_norm="interface",
                               rtol=RTOL, acceleration="fixed", relaxation=0.5))
    assert report["converged"] is True
    assert distance <= RTOL, (where, distance, report["iterations"])


@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
def test_a_selected_field_is_within_the_tolerance_in_a_weakly_coupled_pair(schedule):
    """Both edges deliver a first entry; each member's second entry is
    computed from the other's first and measured by nothing.  On the pass
    the delivered entries met the tolerance, the second entries held what
    the pass before had delivered, nine tolerances away."""
    edges = [("A", "B", G_B, SELECT), ("B", "A", G_A, SELECT)]
    distance, where, report = _step(
        [A, B], edges, dict(iteration_mode=schedule, convergence_norm="interface", rtol=RTOL))
    assert report["converged"] is True
    assert distance <= RTOL, (
        f"{where} is {distance / RTOL:.1f} tolerances from the fixed point "
        f"(iterations={report['iterations']})")


#: ``A -> B -> C -(first entry)-> D``: at a cap of two passes ``C`` was
#: computed from the ``B`` of the first, whose input was the pre-step ``A``.
_CAPPED = ([A, B, C, D],
           [("A", "B", G, None), ("B", "C", G_SECOND, None), ("C", "D", G[:, :1], SELECT)])


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_a_cap_does_not_call_a_state_converged_that_its_last_readings_did_not_give(solver):
    """At the cap the state returned is the successor, and the one evaluation
    that measures it saw nothing move: ``C``'s delivered entry never does.
    The loop's last pass did see ``B`` move, and that is the pass ``C`` was
    computed in, so the report says so."""
    nodes, edges = _CAPPED
    distance, _where, report = _step(nodes, edges, dict(
        iteration_mode="jacobi", convergence_norm="interface", rtol=RTOL, solver=solver,
        max_iterations=2))
    assert distance > 1e3 * RTOL           # the fixture: the state is far off
    assert report["converged"] is False and report["residual"] > 1.0, report


@pytest.mark.parametrize("solver", ["ift", "fori"])
def test_one_pass_past_that_cap_the_state_is_converged_and_is_the_fixed_point(solver):
    nodes, edges = _CAPPED
    distance, where, report = _step(nodes, edges, dict(
        iteration_mode="jacobi", convergence_norm="interface", rtol=RTOL, solver=solver,
        max_iterations=3))
    assert report["converged"] is True and report["iterations"] == 3, report
    assert distance <= 16 * _eps(False), (where, distance)


def test_strict_convergence_refuses_the_capped_state():
    nodes, edges = _CAPPED
    gm = _graph(nodes, edges, dict(iteration_mode="jacobi", convergence_norm="interface",
                                   rtol=RTOL, max_iterations=2, strict_convergence=True))
    with pytest.raises(Exception, match="without converging"):
        gm.step()


# ---------------------------------------------------------------------------
# The static rule
# ---------------------------------------------------------------------------

_STATE = {
    "a": {"u": jnp.zeros(2), "w": jnp.zeros(2), "count": jnp.zeros((), jnp.int32)},
    "b": {"u": jnp.zeros(2), "w": jnp.zeros(2)},
    "c": {"u": jnp.zeros(2)},
}


def _rule(edges, *, order=("a", "b", "c"), dividers=None, **group):
    group.setdefault("convergence_norm", "interface")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g = CouplingGroup(nodes=frozenset(order), **group)
    return _fields_the_interface_norm_misses(g, edges, list(order), _STATE, dividers or {})


def _edge(src, dst, **kw):
    return EdgeSpec(src, dst, "u", "inp", **kw)


_RING = [_edge("a", "b"), _edge("b", "c"), _edge("c", "a")]
_MAPPED = matrix_mapping(np.eye(2, dtype=np.float32))


def test_only_the_interface_norm_misses_a_field():
    for norm in ("l2", "mixed"):
        assert _rule(_RING, convergence_norm=norm, iteration_mode="jacobi") == ({}, False)


def test_under_jacobi_every_fed_member_reads_the_previous_iterate():
    """Unread floating fields are named (never a counter); a field delivered
    whole is not; a member with no internal input is not."""
    assert _rule(_RING, iteration_mode="jacobi") == ({"a": ("w",), "b": ("w",)}, False)
    assert _rule([_edge("a", "b")], iteration_mode="jacobi") == ({"b": ("u", "w")}, False)


def test_under_gauss_seidel_only_a_back_edges_target_reads_the_previous_iterate():
    assert _rule(_RING, iteration_mode="gauss-seidel") == ({"a": ("w",)}, False)
    assert _rule([_edge("a", "b")], iteration_mode="gauss-seidel") == ({}, False)
    assert _rule([_edge("b", "a")], iteration_mode="gauss-seidel") == ({"a": ("u", "w")}, False)
    # ... and a member that reads itself.
    assert _rule([_edge("b", "b")], iteration_mode="gauss-seidel") == ({"b": ("w",)}, False)


def test_a_sub_cycled_member_that_interpolates_reads_the_previous_iterate():
    forward = [_edge("a", "b")]
    kw = dict(iteration_mode="gauss-seidel", subcycling=True)
    assert _rule(forward, dividers={"a": 1, "b": 4}, **kw) == ({"b": ("u", "w")}, False)
    assert _rule(forward, dividers={"a": 4, "b": 1}, **kw) == ({}, False)
    assert _rule(forward, dividers={"a": 1, "b": 4}, boundary_interpolation="constant",
                 **kw) == ({}, False)


@pytest.mark.parametrize("acceleration", ["fixed", "aitken", "iqn-ils", "iqn-imvj"])
def test_an_acceleration_refreshes_every_fed_members_unread_fields(acceleration):
    """It relaxes or extrapolates what it is handed, in whatever order the
    members ran; a member no internal edge feeds does not depend on the
    iterate and is left alone."""
    refreshed, lagged = _rule([_edge("a", "b")], iteration_mode="gauss-seidel",
                              acceleration=acceleration)
    assert refreshed == {"b": ("u", "w")} and lagged is False


@pytest.mark.parametrize("lossy", [dict(mapping=_MAPPED), dict(transform=_first)],
                         ids=["mapping", "transform"])
def test_a_field_read_only_through_a_mapping_or_a_transform_asks_for_the_second_pass(lossy):
    ring = [_edge("a", "b", **lossy), _edge("b", "c"), _edge("c", "a")]
    # ``a.u`` is the lossy field, and ``a`` reads the back edge under either schedule.
    assert _rule(ring, iteration_mode="jacobi")[1] is True
    assert _rule(ring, iteration_mode="gauss-seidel")[1] is True
    # Delivered whole by a second edge, it is measured.
    assert _rule(ring + [_edge("a", "c")], iteration_mode="jacobi")[1] is False
    # A lossy field of a member that reads the pass (not the previous iterate) is current.
    forward = [_edge("a", "b"), _edge("b", "c", **lossy)]
    assert _rule(forward, iteration_mode="gauss-seidel")[1] is False
    assert _rule(forward, iteration_mode="jacobi")[1] is True
    # ... as is one of a member nothing in the group feeds.
    assert _rule([_edge("a", "b", **lossy)], iteration_mode="jacobi")[1] is False
    # A lossy field is read, so it is never recomputed.
    assert "a" not in _rule(ring, iteration_mode="gauss-seidel")[0] or (
        "u" not in _rule(ring, iteration_mode="gauss-seidel")[0]["a"])


# ---------------------------------------------------------------------------
# The loop's exit, against a scripted residual sequence
# ---------------------------------------------------------------------------

def _scripted(residuals, *, lagged, first_res, max_iter=30, threshold=1.0, acceleration="none"):
    """``(passes, reported residual)`` of ``_fixed_point_while`` on *residuals*.

    ``x[1]`` counts passes, so pass ``k`` reads ``residuals[k]`` whatever
    the acceleration does to ``x[0]``; the first is the residual of the
    iterate the loop starts on, *first_res* that of the pass before it.
    """
    x0 = jnp.asarray([1.0, 0.0])
    schedule = jnp.asarray(residuals, x0.dtype)

    def step_pure(x):
        k = jnp.clip(x[1].astype(jnp.int32), 0, schedule.shape[0] - 1)
        return jnp.stack([x[0] * 0.5, x[1] + 1.0]), schedule[k]

    args = (step_pure, x0, (), (), jnp.asarray(first_res, x0.dtype), threshold, max_iter,
            acceleration, 1.0, 0, (0,))
    _x, n, res, _amp, _vw = (_fixed_point_while(*args, lagged) if lagged is not None
                             else _fixed_point_while(*args))
    return int(n), float(res)


def test_a_lagged_reading_stops_on_the_second_pass_within_the_threshold():
    settled = [0.0, 0.0, 0.0]
    assert _scripted(settled, lagged=False, first_res=50.0) == (1, 0.0)
    assert _scripted(settled, lagged=None, first_res=50.0) == (1, 0.0)     # the default
    assert _scripted(settled, lagged=True, first_res=50.0) == (2, 0.0)
    # The pass before the loop counts: readings that had already settled stop at once.
    assert _scripted(settled, lagged=True, first_res=0.5) == (1, 0.0)
    # The streak is of consecutive passes.
    dip = [0.25, 40.0, 10.0, 0.5, 0.125, 0.0, 0.0]
    assert _scripted(dip, lagged=False, first_res=50.0)[0] == 1
    assert _scripted(dip, lagged=True, first_res=50.0)[0] == 5


def test_at_the_cap_a_lagged_reading_reports_the_larger_change():
    """The successor returned at the cap was computed over the loop's last
    pass: where that pass was above the threshold its residual is the one
    reported, and where it was not the measurement of the successor is."""
    moved = [40.0, 0.0, 0.0]
    assert _scripted(moved, lagged=False, first_res=50.0, max_iter=2) == (2, 0.0)
    assert _scripted(moved, lagged=True, first_res=50.0, max_iter=2) == (2, 40.0)
    settled = [0.5, 0.25, 0.0]
    assert _scripted(settled, lagged=True, first_res=50.0, max_iter=2) == (2, 0.25)
    # Above the threshold either way: the larger of the two.
    assert _scripted([40.0, 60.0, 0.0], lagged=True, first_res=50.0, max_iter=2) == (2, 60.0)
    assert _scripted([40.0, 20.0, 0.0], lagged=True, first_res=50.0, max_iter=2) == (2, 40.0)
