"""Differential oracle: ``gradient_relative_error_bound`` against exact dense tangents at early exits.

For a linear coupled group ``x = M x + c(theta)`` the derivative of the
fixed point in any scalar constant ``theta_i`` is exactly
``t* = (I - M)^{-1} dc/dtheta_i`` evaluated at ``x*`` -- a float64 dense
solve, whatever the iteration mode.  The IFT gradient a step returns is the
same solve at the iterate the forward *returned*, so a group stopped early
by its cap has a gradient that is off wherever ``dc/dtheta`` moves with
the state (a gain multiplying an input).  Where ``gradient_bound_usable``
is ``True`` the reported bound must cover the relative error of every
scalar constant's gradient, in the group's norm at the returned state
(each field divided by its own ``max|field|`` there):
``||D (g_k - t*)|| <= bound * ||D g_k||``.

Drawn here: gains and biases of a three-relay group (6 coupled DOF, and
12 on the wide cycle -- more than the range basis's eight vectors, so the
full resolvent norm goes through the transposed map) at caps 2, 3 and 5,
under both iteration modes, from dense, rank-one (a pure cycle with one
rank-one gain, so the coupling Jacobian has rank one) and strongly
non-normal draws.  Rank-one and non-normal maps are where the
Arnoldi resolvent, measured on the Krylov space of the start vector and
the residual, falls short of the full resolvent the secant needs: the
bound read 0.19x the true error there before 0.4.0's round-5 fix applied
the resolvent to each secant exactly.  Every scalar entry of every gain
and bias is checked, which is what the bound now probes (entry by entry,
up to ``GRADIENT_PROBE_ENTRY_LIMIT`` entries per constant).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, note, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.property import coupled_graphs as cg

_STRUCTURES = {
    # A cycle with a chord: dense, possibly non-normal coupling.
    "chord": cg._cycle(3, 2, chords=((0, 2),), leaves=(), outside=False, beta=0.0),
    # A pure cycle: ``drawn_values(rank_one=True)`` makes its Jacobian rank one.
    "cycle": cg._cycle(3, 2, leaves=(), outside=False, beta=0.0),
    # The same at four entries a node: 12 coupled DOF, more than the
    # eight-vector range basis, so the full resolvent norm the
    # Kantorovich check takes is computed through the transposed map.
    "wide-cycle": cg._cycle(3, 4, leaves=(), outside=False, beta=0.0),
}


@functools.lru_cache(maxsize=None)
def _compiled(structure, cap, mode):
    gdef = _STRUCTURES[structure]
    gm = cg.build_graph(gdef, dict(max_iterations=cap, tolerance=1e-12, diagnostics=True,
                                   iteration_mode=mode))
    step = gm._compiled_step
    ext = gm._resolve_external_inputs(None)
    base = gm._state
    names = gdef.group_nodes

    def run(params, x0s):
        state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        for nm, x0 in zip(names, x0s):
            state[nm] = {**state[nm], "x": x0}
        return step(state, ext, params)

    def states(params, x0s):
        out = run(params, x0s)
        return jnp.concatenate([out[nm]["x"] for nm in names])

    jac = jax.jit(jax.jacfwd(states))     # d(states) / d(params), every leaf at once
    return gm, jax.jit(run), jac


def _exact_tangents(gdef, values, xs):
    """``{(node, leaf, entry): t*}``: the fixed point's derivative per scalar gain and bias entry."""
    n = gdef.n
    names = list(gdef.group_nodes)
    M = cg.coupling_matrix(gdef, values)
    R = np.linalg.inv(np.eye(M.shape[0]) - M)
    out = {}
    for k, nm in enumerate(names):
        for i in range(n):                      # bias b[i]
            dc = np.zeros(M.shape[0])
            dc[k * n + i] = 1.0
            out[(nm, "b", (i,))] = R @ dc
        for e in gdef.internal_edges:           # gain G_port[i, l] reads the source's x[l]
            if e.dst != nm:
                continue
            u = xs[e.src]
            for i in range(n):
                for col in range(n):
                    dc = np.zeros(M.shape[0])
                    dc[k * n + i] = u[col]
                    out[(nm, f"G{e.port}", (i, col))] = R @ dc
    return out


def assert_the_bound_covers_every_scalar_constant(structure, cap, mode, values):
    gdef = _STRUCTURES[structure]
    gm, run, jac = _compiled(structure, cap, mode)
    names = list(gdef.group_nodes)
    params = cg.params_for(gm, values)
    x0s = tuple(jnp.asarray(values[nm]["x0"], jnp.float32) for nm in names)
    out = run(params, x0s)
    gm._store_state(jax.tree.map(np.asarray, out))
    d = gm.coupling_diagnostics()[gdef.key]
    pre = {nm: {"x": np.asarray(values[nm]["x0"], np.float64)} for nm in names}
    xs = cg.exact_fixed_point(gdef, values, pre, {}, dt=1.0)
    xk = {nm: np.asarray(out[nm]["x"], np.float64) for nm in names}
    w = np.concatenate([np.full(gdef.n, 1.0 / max(np.max(np.abs(xk[nm])), 1e-30)) for nm in names])
    J = jac(params, x0s)
    worst, where = 0.0, None
    for (nm, leaf, entry), t_star in _exact_tangents(gdef, values, xs).items():
        g_k = np.asarray(J["nodes"][nm][leaf], np.float64)[(slice(None),) + entry]
        den = np.linalg.norm(w * g_k)
        if den == 0:
            continue
        rel = float(np.linalg.norm(w * (g_k - t_star)) / den)
        if rel > worst:
            worst, where = rel, (nm, leaf, entry)
    note(f"{structure} cap={cap} {mode}: worst true {worst:.3e} at {where}; {dict(d)}")
    if d["gradient_bound_usable"]:
        assert d["gradient_relative_error_bound"] >= worst, (
            f"{structure} cap={cap} {mode}: bound {d['gradient_relative_error_bound']:.3e} under "
            f"the true relative error {worst:.3e} of the gradient in {where}")


@st.composite
def _draws(draw, structure):
    rank_one = structure in ("cycle", "wide-cycle")
    rho = draw(st.sampled_from([0.5, 0.8, 0.95]))
    seed = draw(st.integers(0, 2 ** 32 - 1))
    nonnormal = draw(st.booleans())
    values = cg.draw_values(np.random.default_rng(seed), _STRUCTURES[structure], rho,
                            nonnormal=nonnormal, rank_one=rank_one)
    return values


# Costly tier: one compile of the step and its Jacobian per case; the
# examples draw values on it.
@pytest.mark.parametrize("structure", sorted(_STRUCTURES))
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_usable_gradient_bound_covers_every_scalar_constant_at_an_early_exit(structure, data):
    """Per push at three passes under Gauss-Seidel; the slow sibling sweeps caps and modes."""
    assert_the_bound_covers_every_scalar_constant(structure, 3, "gauss-seidel",
                                                  data.draw(_draws(structure)))


# Slow: every (cap, mode) pair compiles its own step and Jacobian.
# Per push: tests/property/test_differential_gradient_bound.py::test_a_usable_gradient_bound_covers_every_scalar_constant_at_an_early_exit
@pytest.mark.slow
@pytest.mark.parametrize("structure", sorted(_STRUCTURES))
@pytest.mark.parametrize("cap", [2, 5])
@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_usable_gradient_bound_covers_every_scalar_constant_at_every_cap(structure, cap, mode, data):
    assert_the_bound_covers_every_scalar_constant(structure, cap, mode,
                                                  data.draw(_draws(structure)))
