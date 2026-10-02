"""A Gauss-Seidel pass rounds along its chain of same-pass reads: the bound's floor counts it.

The float floor ``spectral_error_bound`` adds to the residual is
``PRECISION_FLOOR_ULPS`` units of ``eps * max|field|`` per entry *per
evaluation* a coupling pass rounds like (``_group_evaluations``).  It took
the worst node's count, ``max_n d_n * e_n``.  That is right under Jacobi,
where every node reads the stored previous iterate, and wrong under
Gauss-Seidel: a node reads the output of each member scheduled before it
*from the same pass*, already rounded, so the pass is a composition and the
error at the end of a chain carries every rounding upstream of it.

On a ring of ``N`` scalar relays (loop gain 0.99, every node declaring one
evaluation) stalled at float32 the exact residual is about ``N`` times the
per-pass floor, and the bound -- the case the docs call a bound, every node
declared, the Krylov space exact -- read 0.51x the true distance at
``N = 32`` and 0.30x at ``N = 64`` with ``spectral_usable=True``; the
gradient bound built on the same floor read 0.50x the true gradient error
at ``N = 32`` with ``gradient_bound_usable=True``.  The count is now the
longest chain of same-pass reads.

* the count itself, on the shapes that tell the rules apart (per push, no
  compile);
* both call sites hand it the schedule and the edges (per push, a small
  graph);
* the ``N = 32`` ring end to end, against the exact float64 fixed point and
  the exact gradient (slow).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import maddening.core.graph_manager as gm_mod
from maddening.core.coupling.group import CouplingGroup
from maddening.core.graph_manager import GraphManager, _group_evaluations
from maddening.core.node import BoundaryInputSpec, SimulationNode


class _Relay(SimulationNode):
    """``x <- gain * u + bias`` (scalar), declaring ``evaluations`` per update."""

    def __init__(self, name, *, gain=0.5, bias=0.0, x0=0.0, evaluations=1, dt=1.0):
        super().__init__(name, dt, gain=jnp.float32(gain), bias=jnp.float32(bias))
        self._x0 = float(x0)
        self._evaluations = evaluations

    def initial_state(self):
        return {"x": jnp.asarray([self._x0], jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(1,), dtype=jnp.float32,
                                       default=jnp.zeros(1, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        u = boundary_inputs.get("u", jnp.zeros(1, jnp.float32))
        return {"x": p["gain"] * u + p["bias"]}

    def update_evaluations(self):
        return self._evaluations


def _spec(node):
    return SimpleNamespace(node=node, timestep=node.delta_t)


def _edge(src, dst):
    return SimpleNamespace(source_node=src, target_node=dst)


def _count(names, edges, *, mode="gauss-seidel", schedule=None, evaluations=None,
           timesteps=None, subcycling=False):
    evaluations = evaluations or {}
    timesteps = timesteps or {}
    nodes = {nm: _spec(_Relay(nm, evaluations=evaluations.get(nm, 1),
                              dt=timesteps.get(nm, 1.0))) for nm in names}
    group = CouplingGroup(nodes=frozenset(names), iteration_mode=mode,
                          subcycling=subcycling)
    return _group_evaluations(group, nodes, schedule or list(names),
                              [_edge(s, d) for s, d in edges])


def _ring(n):
    names = [f"n{k}" for k in range(n)]
    return names, [(names[k - 1], names[k]) for k in range(n)]


# ---------------------------------------------------------------------------
# The count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [2, 3, 8, 32])
def test_a_gauss_seidel_ring_rounds_like_its_length(n):
    """Every node reads the one before it from the same pass: a chain of ``n``."""
    names, edges = _ring(n)
    assert _count(names, edges) == (float(n), True)


@pytest.mark.parametrize("n", [2, 3, 32])
def test_a_jacobi_ring_rounds_like_one_node(n):
    """Under Jacobi every read is of the stored previous iterate: the worst node."""
    names, edges = _ring(n)
    assert _count(names, edges, mode="jacobi") == (1.0, True)


def test_a_read_of_a_member_scheduled_later_starts_no_chain():
    """The same ring swept in reverse: only the closing edge is a same-pass read."""
    names, edges = _ring(4)
    assert _count(names, edges, schedule=list(reversed(names))) == (2.0, True)


def test_the_count_is_the_longest_chain_not_the_sum():
    """A hub feeding three leaves rounds like two; a diamond like its longest path."""
    star = ["h", "a", "b", "c"]
    assert _count(star, [("h", "a"), ("h", "b"), ("h", "c"), ("c", "h")]) == (2.0, True)
    diamond = ["a", "b", "c", "d"]
    edges = [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d"), ("d", "a")]
    assert _count(diamond, edges) == (3.0, True)


def test_each_link_carries_its_own_evaluation_count():
    """A node declaring 4 evaluations in a chain of three: 1 + 4 + 1."""
    names, edges = _ring(3)
    assert _count(names, edges, evaluations={"n1": 4}) == (6.0, True)
    assert _count(names, edges, evaluations={"n1": 4}, mode="jacobi") == (4.0, True)


def test_a_sub_cycled_member_counts_its_divider_along_the_chain():
    """``d_n * e_n`` per link: a member sub-cycled 3x in a ring of three is 1 + 3 + 1."""
    names, edges = _ring(3)
    kw = dict(timesteps={"n0": 1.0, "n1": 1.0 / 3.0, "n2": 1.0}, subcycling=True)
    assert _count(names, edges, **kw) == (5.0, True)
    assert _count(names, edges, mode="jacobi", **kw) == (3.0, True)


def test_an_undeclared_member_still_counts_one_and_clears_declared():
    names, edges = _ring(3)
    nodes = {nm: _spec(_Relay(nm, evaluations=None if nm == "n2" else 1)) for nm in names}
    group = CouplingGroup(nodes=frozenset(names))
    assert _group_evaluations(group, nodes, names, [_edge(s, d) for s, d in edges]) == (
        3.0, False)


def test_edges_outside_the_group_do_not_count():
    names, edges = _ring(3)
    edges = edges + [("drv", "n0"), ("n2", "sink"), ("n0", "n0")]
    assert _count(names, edges) == (3.0, True)


# ---------------------------------------------------------------------------
# Both call sites get the schedule and the edges
# ---------------------------------------------------------------------------


def _ring_graph(n, mode, *, diagnostics=False, x0=0.0, g=0.5, c=0.0):
    names = [f"n{k:03d}" for k in range(n)]
    gm = GraphManager()
    for nm in names:
        gm.add_node(_Relay(nm, gain=g, bias=c, x0=x0))
    for k in range(n):
        gm.add_edge(names[k - 1], names[k], "x", "u")
    gm.add_coupling_group(names, max_iterations=400, tolerance=1e-12,
                          diagnostics=diagnostics, iteration_mode=mode)
    gm.compile()
    return gm, "+".join(sorted(names))


@pytest.mark.parametrize("mode, want", [("gauss-seidel", 5.0), ("jacobi", 1.0)])
def test_the_report_and_the_step_count_the_same_chain(monkeypatch, mode, want):
    """The in-graph resolution and ``coupling_diagnostics``' floor see one count.

    Both read ``_group_evaluations``; a call site that went back to the
    worst node's count (or lost the schedule) would answer 1 here under
    Gauss-Seidel.
    """
    seen = []
    real = gm_mod._group_evaluations

    def spy(group, nodes, schedule, edges):
        out = real(group, nodes, schedule, edges)
        seen.append(out[0])
        return out

    monkeypatch.setattr(gm_mod, "_group_evaluations", spy)
    gm, key = _ring_graph(5, mode)
    gm.step()
    assert seen, "the step was built without counting"
    built = list(seen)
    gm.coupling_diagnostics()
    assert built and set(built) == {want}, built
    assert seen[len(built):] == [want], seen


# ---------------------------------------------------------------------------
# The ring end to end
# ---------------------------------------------------------------------------

#: Loop gain once round the ring, and how far short of the fixed point it starts.
_RHO, _HEAD = 0.99, 1e-3


def _stalled_ring(n):
    """The ring of the finding, at its float32 stall, with diagnostics."""
    g = _RHO ** (1.0 / n)
    g32, c32 = float(np.float32(g)), float(np.float32(1.0 - g))
    xstar = c32 / (1.0 - g32)          # exact fixed point of the float32 map

    def build():
        return _ring_graph(n, "gauss-seidel", diagnostics=True,
                           x0=xstar * (1.0 - _HEAD), g=g, c=1.0 - g)
    return build, g32, c32, xstar


def _exact_x0(n, gains, biases):
    """``x_0`` at the fixed point of ``x_k = g_k x_{k-1} + c_k`` round the ring."""
    M = np.eye(n)
    for k in range(n):
        M[k, (k - 1) % n] -= gains[k]
    return float(np.linalg.solve(M, np.asarray(biases, np.float64))[0])


# Slow: a 32-node Gauss-Seidel group with diagnostics=True compiles the
# spectral and gradient-bound machinery over 96 constants, and the gradient
# compiles the step again (~50 s on a 3-core slice).  Per push, the count is
# pinned above without a compile, and both call sites by
# ``test_the_report_and_the_step_count_the_same_chain``.
@pytest.mark.slow
def test_both_bounds_hold_on_a_stalled_gauss_seidel_ring_of_32():
    """``spectral_error_bound`` and the gradient bound over a 32-relay ring's stall.

    Read 0.51x the true distance and 0.50x the true gradient error, both
    flags ``True``, while the floor took one evaluation per pass.  The
    stall is genuine (residual exactly 0.0, ``precision_limited``) and the
    case is the one the docs call a bound: every node declares its count,
    the Jacobian has rank one, so ``rho_spectral`` is the loop gain.
    """
    n = 32
    build, g32, c32, xstar = _stalled_ring(n)
    gm, key = build()
    gm.step()
    d = gm.coupling_diagnostics()[key]
    names = sorted(gm._nodes)
    x = np.array([float(gm.get_node_state(nm)["x"][0]) for nm in names])
    assert d["residual"] == 0.0 and d["precision_limited"] and d["converged"], dict(d)
    assert d["rho_spectral"] == pytest.approx(_RHO, abs=1e-5), dict(d)
    assert d["spectral_usable"] and d["gradient_bound_usable"], dict(d)
    dist = float(np.sqrt(np.sum(((x - xstar) / np.maximum(np.abs(x), abs(xstar))) ** 2)))
    assert dist > 100 * np.finfo(np.float32).eps, "fixture premise: the stall is far off"
    assert d["spectral_error_bound"] >= dist, (
        f"spectral_error_bound {d['spectral_error_bound']:.4e} below the true "
        f"distance {dist:.4e} on a stalled Gauss-Seidel ring of {n}")

    # The gradient of x_0 with respect to node 0's gain, which multiplies
    # the state, so the IFT gradient at the stall is off by about the
    # relative distance; against the exact fixed point's, by a central
    # difference of the float64 linear solve.
    first = names[0]

    def loss(p):
        return jnp.ravel(build()[0].run_scan(1, params=p)[first]["x"])[0]

    got = float(jax.grad(loss)(build()[0].params)["nodes"][first]["gain"])
    h = 1e-6
    gains = [g32] * n
    exact = (_exact_x0(n, [g32 + h] + gains[1:], [c32] * n)
             - _exact_x0(n, [g32 - h] + gains[1:], [c32] * n)) / (2 * h)
    err = abs(got - exact) / abs(got)
    assert err > 0, "fixture premise: the gradient at the stall is not exact"
    assert d["gradient_relative_error_bound"] >= err, (
        f"gradient_relative_error_bound {d['gradient_relative_error_bound']:.3e} below "
        f"the true relative error {err:.3e} on a stalled Gauss-Seidel ring of {n}")
