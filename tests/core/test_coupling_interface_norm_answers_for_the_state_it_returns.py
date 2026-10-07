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
  ``converged=True`` and ``iterations=1`` (MADD-ANO-238; 0.1.0 to 0.3.1).
  A weakly coupled pair returned such a field a few hundred tolerances
  off, under Gauss-Seidel too (the member that reads a back edge), and a
  fixed relaxation left it a blend of every pass so far.  The solve now
  returns these fields as one pass computes them at the state it returns
  -- from the readings the verdict was taken on -- and changes nothing
  else: iterates, residuals and pass counts are the ones they were;
* a field edges read **only through a mapping or a transform** that
  delivers less than the field.  It feeds back, so it cannot be recomputed
  without moving the readings the verdict was taken on, and the part of it
  the edges do not deliver is still measured by nothing: MADD-ANO-239,
  open, pinned here by strict xfails.

Every oracle here is a float64 closed form of the linear map the graph
holds; none calls the code under test.  The static rule
(``_fields_the_interface_norm_misses``) is checked on its own, branch by
branch.
"""

from __future__ import annotations

import os
import warnings

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    6800 tolerances off."""
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


_OPEN = pytest.mark.xfail(strict=True, reason=(
    "MADD-ANO-239 (open): the part of a field its edges deliver only through a mapping or "
    "a transform is measured by nothing"))


@_OPEN
@pytest.mark.parametrize("how", [SELECT, _first], ids=["mapping", "transform"])
def test_the_part_of_a_field_its_edge_does_not_deliver_is_not_left_a_pass_behind(how):
    """``A -> B -> C`` with ``B`` delivered through a selection of its first
    entry, which no input moves: what ``B`` delivers is settled from the
    first pass, a residual of exactly zero, while its second entry still
    held the pre-step ``A``.  The same stale member, behind a mapping."""
    edges = [("A", "B", G_SECOND, None), ("B", "C", G[:, :1], how)]
    distance, where, report = _step(
        [A, B, C], edges, dict(iteration_mode="jacobi", convergence_norm="interface",
                               rtol=RTOL))
    assert report["converged"] is True
    assert distance <= 16 * _eps(False), (where, distance, report["iterations"])


#: Each member's first entry reads the other's at 0.03 (the loop the norm
#: watches gains a digit and a half per pass); its second entry, which no
#: edge delivers, reads the other's first at order one.
G_A = np.array([[0.03], [1.0]])
G_B = np.array([[0.03], [-0.7]])


@_OPEN
def test_a_relaxation_does_not_leave_the_undelivered_part_of_a_field_between_passes():
    """The same chain under ``acceleration="fixed"`` at 0.5: ``B``'s second
    entry is a blend of the passes so far, and would be after a second pass
    within the threshold too."""
    edges = [("A", "B", G_SECOND, None), ("B", "C", G[:, :1], SELECT)]
    distance, where, report = _step(
        [A, B, C], edges, dict(iteration_mode="jacobi", convergence_norm="interface",
                               rtol=RTOL, acceleration="fixed", relaxation=0.5))
    assert report["converged"] is True
    assert distance <= RTOL, (where, distance, report["iterations"])


@_OPEN
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
        assert _rule(_RING, convergence_norm=norm, iteration_mode="jacobi") == {}


def test_under_jacobi_every_fed_member_reads_the_previous_iterate():
    """Unread floating fields are named (never a counter); a field delivered
    whole is not; a member with no internal input is not."""
    assert _rule(_RING, iteration_mode="jacobi") == {"a": ("w",), "b": ("w",)}
    assert _rule([_edge("a", "b")], iteration_mode="jacobi") == {"b": ("u", "w")}


def test_under_gauss_seidel_only_a_back_edges_target_reads_the_previous_iterate():
    assert _rule(_RING, iteration_mode="gauss-seidel") == {"a": ("w",)}
    assert _rule([_edge("a", "b")], iteration_mode="gauss-seidel") == {}
    assert _rule([_edge("b", "a")], iteration_mode="gauss-seidel") == {"a": ("u", "w")}
    # ... and a member that reads itself.
    assert _rule([_edge("b", "b")], iteration_mode="gauss-seidel") == {"b": ("w",)}


def test_a_sub_cycled_member_that_interpolates_reads_the_previous_iterate():
    forward = [_edge("a", "b")]
    kw = dict(iteration_mode="gauss-seidel", subcycling=True)
    assert _rule(forward, dividers={"a": 1, "b": 4}, **kw) == {"b": ("u", "w")}
    assert _rule(forward, dividers={"a": 4, "b": 1}, **kw) == {}
    assert _rule(forward, dividers={"a": 1, "b": 4}, boundary_interpolation="constant",
                 **kw) == {}


@pytest.mark.parametrize("acceleration", ["fixed", "aitken", "iqn-ils", "iqn-imvj"])
def test_an_acceleration_refreshes_every_fed_members_unread_fields(acceleration):
    """It relaxes or extrapolates what it is handed, in whatever order the
    members ran; a member no internal edge feeds does not depend on the
    iterate and is left alone."""
    refreshed = _rule([_edge("a", "b")], iteration_mode="gauss-seidel",
                      acceleration=acceleration)
    assert refreshed == {"b": ("u", "w")}


@pytest.mark.parametrize("lossy", [dict(mapping=_MAPPED), dict(transform=_first)],
                         ids=["mapping", "transform"])
@pytest.mark.parametrize("schedule", ["jacobi", "gauss-seidel"])
def test_a_field_read_through_a_mapping_or_a_transform_is_never_recomputed(schedule, lossy):
    """It is read, so recomputing it would move the readings the verdict was
    taken on; the member's other unread fields are still refreshed."""
    ring = [_edge("a", "b", **lossy), _edge("b", "c"), _edge("c", "a")]
    assert _rule(ring, iteration_mode=schedule)["a"] == ("w",)
