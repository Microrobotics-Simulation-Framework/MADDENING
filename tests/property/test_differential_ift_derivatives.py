"""Differential oracle: the IFT derivatives at every scale, against dense and fori.

Under ``solver="ift"`` the tangent (``jax.jacfwd``) and the adjoint
(``jax.grad``) of a coupled step are one linear solve each,
``(I - dF/dx) v = b`` (``_ift_linear_solve``).  Its Krylov criterion carried
an absolute ``1e-8``, so with the zero initial guess any ``b`` below about
``1e-8`` "converged" before a single step and the derivative came back
exactly zero, reported successful.  Three paths compute the same
derivative and must agree at every scale of the problem:

* ``linear_solver="gmres"`` -- the default, matrix-free;
* ``linear_solver="dense"`` -- the Jacobian materialised and LU-solved, no
  tolerance at all;
* ``solver="fori"`` -- reverse/forward mode straight through the unrolled
  passes (no acceleration), which agrees with the fixed point's derivative
  once the loop has run long enough for ``rho**k`` to vanish: 400 passes,
  the iteration contracting at a drawn ``rho <= 0.9`` in its own mode
  (``_at_iteration_rate``) -- a sweep that does not contract has an
  unrolled derivative that never settles, which is not a defect.

The group is linear and equivariant (no ``beta * dt``, no leaves), so its
state at scale ``s`` -- every bias and initial state times a power of two
``s`` -- is ``s`` times the unscaled one, and so are both right-hand
sides: the cotangent of the least-squares loss (``grad`` in a bias), and
the tangent of a gain multiplier (``jacfwd``; the tangent in a bias would
not scale).  The derivatives are compared relative to their own size, from
``2**-100`` (about 8e-31, near the bottom of float32's normal range for the
state) to ``2**60`` (about 1e18, below where the loss's squares overflow).

Each path is jitted once per graph through the compiled step itself
(``gm._compiled_step``, a pure function of the state and the parameters),
so a scale sweep reuses one compile.

The same oracle holds the non-finite right-hand side: a tangent seed or an
observation with a NaN or infinite entry made the default solve's
tolerance NaN or ``inf``, and the derivative came back exactly zero,
reported successful, where ``"dense"`` and ``"fori"`` read NaN
(MADD-ANO-154).  At every scale the default path is now NaN in every entry,
so in particular wherever either reference is not finite.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.conftest import EXAMPLES_COSTLY
from tests.property import coupled_graphs as cg

#: ``2**k`` for these ``k``: 8e-31 .. 1e18.
SCALE_EXPONENTS = (-100, -80, -60, -40, -20, 0, 20, 40, 60)

#: The per-push triangle without its leaves, its outside driver and sink,
#: or the ``beta * dt`` term: linear and equivariant.  Two entries a node
#: (6 coupled DOF, where a failed Krylov solve falls back to dense) and 20
#: (60, above the fallback: GMRES alone answers).
_GDEFS = {n: cg._cycle(3, n, chords=((0, 2),), leaves=(), outside=False, beta=0.0)
          for n in (2, 20)}

_PATHS = {
    "gmres": dict(solver="ift", linear_solver="gmres"),
    "dense": dict(solver="ift", linear_solver="dense"),
    "fori": dict(solver="fori"),
}


def _at_iteration_rate(gdef, values, rho, mode):
    """*values* with the internal gains rescaled so the iteration contracts at *rho*.

    ``drawn_values`` fixes the Jacobi rate; a Gauss-Seidel sweep of the
    same gains can contract faster or not at all (a radius of 1.004 was
    drawn at Jacobi rate 0.9), and the unrolled ``fori`` derivative only
    approaches the fixed point's where the sweep contracts.  Under
    Gauss-Seidel the gains are scaled by the factor, found by bisection,
    that puts the sweep's spectral radius at *rho*.
    """
    if mode == "jacobi":
        return values
    internal = gdef.internal_edges

    def scaled(t):
        out = {nm: {**v, "G": list(v["G"])} for nm, v in values.items()}
        for e in internal:
            out[e.dst]["G"][e.port] = np.asarray(
                np.asarray(values[e.dst]["G"][e.port], np.float64) * t, np.float32)
        return out

    def radius(t):
        return float(np.max(np.abs(np.linalg.eigvals(cg.gauss_seidel_matrix(gdef, scaled(t))))))

    lo, hi = 0.0, 1.0
    while radius(hi) < rho:
        hi *= 2.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if radius(mid) < rho else (lo, mid)
    return scaled(lo)


@functools.lru_cache(maxsize=None)
def _derivative_fns(path, n, mode):
    """``(grad_fn, jac_fn, gm)``: jitted derivatives of one step, for one path."""
    _GDEF = _GDEFS[n]
    gm = cg.build_graph(_GDEF, dict(_PATHS[path], tolerance=1e-12, max_iterations=400,
                                    iteration_mode=mode))
    # The state as compiled, before any step: no warm-start history from
    # another scale.
    step = gm._compiled_step
    ext = gm._resolve_external_inputs(None)
    base = jax.tree.map(lambda v: v, gm._state)

    def run(params, x0s):
        state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        for nm, x0 in zip(_GDEF.group_nodes, x0s):
            state[nm] = {**state[nm], "x": x0}
        return step(state, ext, params)

    def loss(b0, params, x0s, obs):
        p = jax.tree.map(lambda v: v, params)
        p["nodes"][_GDEF.group_nodes[0]]["b"] = b0
        out = run(p, x0s)
        return sum(jnp.sum((out[nm]["x"] - o) ** 2) for nm, o in zip(_GDEF.group_nodes, obs))

    def state_of(t, params, x0s):
        # The tangent of a gain multiplier: its rhs is ``G0 @ u``, which
        # scales with the state (the tangent in a bias would not).
        p = jax.tree.map(lambda v: v, params)
        first = _GDEF.group_nodes[0]
        p["nodes"][first]["G0"] = t * p["nodes"][first]["G0"]
        return run(p, x0s)[first]["x"]

    def fields_along(b0, b0_dot, params, x0s):
        # The tangent of every group field along a tangent in the first
        # node's bias: its rhs is the tangent itself, entry by entry.
        def fields(b):
            p = jax.tree.map(lambda v: v, params)
            p["nodes"][_GDEF.group_nodes[0]]["b"] = b
            out = run(p, x0s)
            return jnp.concatenate([out[nm]["x"] for nm in _GDEF.group_nodes])
        return jax.jvp(fields, (b0,), (b0_dot,))[1]

    return (jax.jit(jax.grad(loss)), jax.jit(jax.jacfwd(state_of)), gm,
            jax.jit(fields_along))


def _derivatives(path, n, mode, values, k):
    _GDEF = _GDEFS[n]
    grad_fn, jac_fn, gm, _ = _derivative_fns(path, n, mode)
    s = np.float32(2.0 ** k)
    scaled = {nm: {"G": v["G"], "b": np.asarray(v["b"] * s, np.float32),
                   "x0": np.asarray(v["x0"] * s, np.float32)} for nm, v in values.items()}
    params = cg.params_for(gm, scaled)
    x0s = tuple(jnp.asarray(scaled[nm]["x0"]) for nm in _GDEF.group_nodes)
    pre = {nm: {"x": np.asarray(scaled[nm]["x0"], np.float64)} for nm in _GDEF.group_nodes}
    exact = cg.exact_fixed_point(_GDEF, scaled, pre, {}, dt=1.0)
    obs = tuple(jnp.asarray(1.05 * exact[nm], jnp.float32) for nm in _GDEF.group_nodes)
    b0 = params["nodes"][_GDEF.group_nodes[0]]["b"]
    g = np.asarray(grad_fn(b0, params, x0s, obs), np.float64)
    j = np.asarray(jac_fn(jnp.float32(1.0), params, x0s), np.float64)
    return g, j


def _rel(a, b):
    scale = max(float(np.max(np.abs(b))), np.finfo(np.float64).tiny)
    return float(np.max(np.abs(a - b))) / scale


def assert_derivatives_agree_at_every_scale(n, mode, values):
    for k in SCALE_EXPONENTS:
        dense_g, dense_j = _derivatives("dense", n, mode, values, k)
        assert np.any(dense_g != 0) and np.any(dense_j != 0), (k, "the reference is zero")
        for path in ("gmres", "fori"):
            g, j = _derivatives(path, n, mode, values, k)
            assert _rel(g, dense_g) < 1e-3, (path, k, "grad", g, dense_g)
            assert _rel(j, dense_j) < 1e-3, (path, k, "jacfwd", j, dense_j)


# Slow: six jitted derivative programs per (size, mode), 6-12 s each on CI;
# each example sweeps nine scales on them.
# Per push: tests/core/test_coupling_ift_gradient_in_any_units.py::test_the_tangent_and_the_adjoint_are_the_same_at_every_scale
# Per push: tests/core/test_coupling_ift_gradient_in_any_units.py::test_the_adjoint_above_the_dense_fallback_is_the_same_at_every_scale
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("n", sorted(_GDEFS), ids=lambda n: f"{3 * n}dof")
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_the_ift_derivatives_agree_with_dense_and_fori_at_every_scale(n, mode, data):
    rho = data.draw(st.sampled_from([0.3, 0.6, 0.9]))
    seed = data.draw(st.integers(0, 2 ** 32 - 1))
    # Non-normal gains at 20 entries a node make ``I - dF/dx`` ill-conditioned
    # past float32 (cond 1e3-1e12 measured), where GMRES raises its
    # documented adjoint error at every scale alike: loud, and not this claim.
    nonnormal = n < 20 and data.draw(st.booleans())
    values = cg.draw_values(np.random.default_rng(seed), _GDEFS[n], rho, nonnormal=nonnormal)
    assert_derivatives_agree_at_every_scale(n, mode, _at_iteration_rate(_GDEFS[n], values, rho, mode))


#: ``2**k`` for these ``k``: the bottom, the middle and the top of the sweep.
NON_FINITE_SCALE_EXPONENTS = (-100, 0, 60)


def _scaled_problem(n, mode, values, k):
    """``(params, x0s, b0, obs)`` at scale ``2**k`` for the graph of ``(dense, n, mode)``."""
    _GDEF = _GDEFS[n]
    gm = _derivative_fns("dense", n, mode)[2]
    s = np.float32(2.0 ** k)
    scaled = {nm: {"G": v["G"], "b": np.asarray(v["b"] * s, np.float32),
                   "x0": np.asarray(v["x0"] * s, np.float32)} for nm, v in values.items()}
    params = cg.params_for(gm, scaled)
    x0s = tuple(jnp.asarray(scaled[nm]["x0"]) for nm in _GDEF.group_nodes)
    pre = {nm: {"x": np.asarray(scaled[nm]["x0"], np.float64)} for nm in _GDEF.group_nodes}
    exact = cg.exact_fixed_point(_GDEF, scaled, pre, {}, dt=1.0)
    obs = tuple(jnp.asarray(1.05 * exact[nm], jnp.float32) for nm in _GDEF.group_nodes)
    return params, x0s, params["nodes"][_GDEF.group_nodes[0]]["b"], obs


def assert_a_non_finite_rhs_is_non_finite_on_the_default_path(n, mode, values, bad, entry):
    """At every scale: the default path is NaN in every entry, and the
    references are not finite somewhere (so the case tests something)."""
    for k in NON_FINITE_SCALE_EXPONENTS:
        params, x0s, b0, obs = _scaled_problem(n, mode, values, k)
        s = np.float32(2.0 ** k)
        # A tangent seed in the first node's bias, one entry of it non-finite.
        b0_dot = (jnp.ones_like(b0) * s).at[entry].set(bad)
        # An observation of the first node with one entry non-finite: the
        # cotangent ``2 (x - obs)`` is not finite there.
        bad_obs = (obs[0].at[entry].set(bad),) + obs[1:]
        got = {}
        for path in ("gmres", "dense", "fori"):
            grad_fn, _, _, jvp_fn = _derivative_fns(path, n, mode)
            got[path] = (np.asarray(grad_fn(b0, params, x0s, bad_obs)),
                         np.asarray(jvp_fn(b0, b0_dot, params, x0s)))
        for i, kind in enumerate(("grad", "jvp")):
            default = got["gmres"][i]
            assert np.all(np.isnan(default)), (k, kind, bad, default)
            for ref in ("dense", "fori"):
                r = got[ref][i]
                assert not np.all(np.isfinite(r)), (k, kind, ref, "the reference is finite", r)
                assert np.all(~np.isfinite(default[~np.isfinite(r)])), (k, kind, ref, default, r)


# Slow: the same six jitted programs per (size, mode) as the oracle above,
# plus the tangent program; each example runs three scales on them.
# Per push: tests/core/test_krylov_solves_propagate_a_non_finite_rhs.py::test_a_non_finite_tangent_gives_a_nan_tangent_not_zero
# Per push: tests/core/test_krylov_solves_propagate_a_non_finite_rhs.py::test_a_non_finite_cotangent_gives_a_nan_gradient_not_zero
@pytest.mark.slow
@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("n", sorted(_GDEFS), ids=lambda n: f"{3 * n}dof")
@settings(max_examples=EXAMPLES_COSTLY, deadline=None, derandomize=True)
@given(data=st.data())
def test_a_non_finite_rhs_gives_a_non_finite_derivative_at_every_scale(n, mode, data):
    rho = data.draw(st.sampled_from([0.3, 0.6, 0.9]))
    seed = data.draw(st.integers(0, 2 ** 32 - 1))
    bad = data.draw(st.sampled_from([float("nan"), float("inf"), float("-inf")]))
    entry = data.draw(st.integers(0, n - 1))
    nonnormal = n < 20 and data.draw(st.booleans())
    values = cg.draw_values(np.random.default_rng(seed), _GDEFS[n], rho, nonnormal=nonnormal)
    assert_a_non_finite_rhs_is_non_finite_on_the_default_path(
        n, mode, _at_iteration_rate(_GDEFS[n], values, rho, mode), bad, entry)
