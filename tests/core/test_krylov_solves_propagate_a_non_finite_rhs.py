"""A right-hand side with a NaN or infinite entry gives NaN from every Krylov solve, never zeros.

Every matrix-free solve in the library -- the IFT tangent and adjoint of a
coupling group (``graph_manager._ift_linear_solve``), ``ift_linear_solve``
and ``sharded_cg`` / ``sharded_gmres`` -- poses its Krylov iteration on the
right-hand side framed by ``max|b|``, with a tolerance relative to it.  A
NaN or infinite entry made that tolerance NaN or ``inf``, the zero initial
guess passed lineax's test (and the loop CG's) before a single step, and the
solve returned zeros reported successful.  So a NaN tangent or cotangent
came back as an exactly zero derivative under the default
``linear_solver="gmres"``, where ``"dense"`` and ``solver="fori"`` read NaN
(MADD-ANO-154), and the public solvers returned zeros, ``sharded_cg``
mostly with ``converged=True`` (MADD-ANO-155).

Now each solve answers NaN in every entry (an honest Krylov iteration does
the same: its first basis vector is ``b / ||b||``), its backend is handed
zeros in place of the rhs, and a sharded result reports ``converged=False``
and a NaN residual norm.  The references are ``"dense"`` (an LU solve, no
tolerance at all) and, for the coupled step, ``solver="fori"``: wherever
either is not finite, the Krylov answer is not finite either.

Each program is jitted once with the right-hand side (or the tangent, or
the observation that sets the cotangent) as an input, so the three
non-finite values run on one compile.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import functools
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maddening.cloud.multigpu.iterative_solver import sharded_cg, sharded_gmres
from maddening.core.graph_manager import GraphManager, _ift_linear_solve
from maddening.core.node import BoundaryInputSpec, SimulationNode
from maddening.core.solver_utils import ift_linear_solve

#: The non-finite values a right-hand side entry is set to.
BAD = (float("nan"), float("inf"), float("-inf"))
BAD_IDS = ("nan", "+inf", "-inf")


def _assert_nan_everywhere_and_where_the_reference_is_not_finite(got, *refs, what=""):
    """``got`` is NaN in every entry, and non-finite wherever a reference is.

    The second half follows from the first; it is written out because it
    is the property a user relies on, and the first is only how it is met.
    """
    got = np.asarray(got)
    assert np.all(np.isnan(got)), (what, got)
    for ref in refs:
        ref = np.asarray(ref)
        assert not np.all(np.isfinite(ref)), (what, "the reference is finite: the case tests nothing", ref)
        assert np.all(~np.isfinite(got[~np.isfinite(ref)])), (what, got, ref)


# ---------------------------------------------------------------------------
# The coupled step: tangent (jax.jvp) and adjoint (jax.grad) of the fixed point
# ---------------------------------------------------------------------------

class _Relay(SimulationNode):
    """``x <- b + g u`` on an ``n``-entry field; ``g`` and ``b`` are scalar parameters."""

    def __init__(self, name, g, b, n):
        super().__init__(name, 1.0, g=jnp.float32(g), b=jnp.float32(b))
        self._n = n

    def initial_state(self):
        return {"x": jnp.zeros(self._n, jnp.float32)}

    def boundary_input_spec(self):
        return {"u": BoundaryInputSpec(shape=(self._n,), dtype=jnp.float32,
                                       default=jnp.zeros(self._n, jnp.float32))}

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["b"] + p["g"] * boundary_inputs["u"]}


#: Two coupled DOF (a failed Krylov solve there re-solves densely) and 60
#: (above ``_DENSE_ADJOINT_FALLBACK_MAX_DOF``: GMRES alone answers, and a
#: failed solve raises).
SIZES = {"2dof": 1, "60dof": 30}

#: The paths that compute the same derivative.
PATHS = {
    "gmres": dict(solver="ift"),
    "dense": dict(solver="ift", linear_solver="dense"),
}


@functools.lru_cache(maxsize=None)
def _coupled(path, size, mode):
    """``(jvp_fn, grad_fn, params)`` for the pair ``A <-> B``, jitted once.

    ``jvp_fn(t)`` is the tangent of both fields along ``t`` in ``A``'s bias;
    ``grad_fn(obs)`` the gradient in every parameter of
    ``sum((x_A - obs)**2)``, whose cotangent is ``2 (x_A - obs)``.
    """
    n = SIZES[size]
    gm = GraphManager()
    gm.add_node(_Relay("A", 0.5, 1.0, n))
    gm.add_node(_Relay("B", 0.8, 0.5, n))
    gm.add_edge("A", "B", "x", "u")
    gm.add_edge("B", "A", "x", "u")
    gm.add_coupling_group(["A", "B"], max_iterations=200, tolerance=1e-7,
                          iteration_mode=mode, **PATHS[path])
    gm.compile()
    step, state, ext, params = gm._compiled_step, gm._state, gm._default_external_inputs(), gm.params

    def tangent(t):
        dt = jax.tree.map(jnp.zeros_like, params)
        dt["nodes"]["A"]["b"] = t
        return dt

    @jax.jit
    def jvp_fn(t):
        _, out = jax.jvp(lambda p: step(state, ext, p), (params,), (tangent(t),))
        return jnp.concatenate([out["A"]["x"], out["B"]["x"]])

    @jax.jit
    def grad_fn(obs):
        g = jax.grad(lambda p: jnp.sum((step(state, ext, p)["A"]["x"] - obs) ** 2))(params)
        return jnp.stack([g["nodes"][nm][k] for nm in ("A", "B") for k in ("b", "g")])

    return jvp_fn, grad_fn, params


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("size", list(SIZES))
def test_a_non_finite_tangent_gives_a_nan_tangent_not_zero(size, mode):
    """Before the fix the default solve read 0.0 here at both sizes; dense read NaN."""
    jvp_gmres = _coupled("gmres", size, mode)[0]
    jvp_dense = _coupled("dense", size, mode)[0]
    finite = np.asarray(jvp_gmres(jnp.float32(1.0)))
    np.testing.assert_allclose(finite, np.asarray(jvp_dense(jnp.float32(1.0))), rtol=1e-4)
    for bad, name in zip(BAD, BAD_IDS):
        t = jnp.float32(bad)
        _assert_nan_everywhere_and_where_the_reference_is_not_finite(
            jvp_gmres(t), jvp_dense(t), what=(size, mode, name))


@pytest.mark.parametrize("mode", ["gauss-seidel", "jacobi"])
@pytest.mark.parametrize("size", list(SIZES))
def test_a_non_finite_cotangent_gives_a_nan_gradient_not_zero(size, mode):
    """An observation that is NaN (a missing value) or infinite makes the
    loss's cotangent non-finite in that entry; the gradient in every
    parameter the group reads read 0.0 under the default solve."""
    grad_gmres = _coupled("gmres", size, mode)[1]
    grad_dense = _coupled("dense", size, mode)[1]
    n = SIZES[size]
    obs = jnp.full((n,), 3.0, jnp.float32)
    np.testing.assert_allclose(np.asarray(grad_gmres(obs)), np.asarray(grad_dense(obs)), rtol=1e-4)
    for bad, name in zip(BAD, BAD_IDS):
        o = obs.at[n - 1].set(bad)
        _assert_nan_everywhere_and_where_the_reference_is_not_finite(
            grad_gmres(o), grad_dense(o), what=(size, mode, name))


@jax.custom_jvp
def _no_derivative(c):
    """The identity, whose derivative along any non-zero tangent is NaN."""
    return c


@_no_derivative.defjvp
def _no_derivative_jvp(primals, tangents):
    (c,), (t,) = primals, tangents
    return c, jnp.where(t != 0, jnp.nan, 0.0).astype(t.dtype)


class _OpaqueRelay(_Relay):
    """``x <- b + g u + opaque(c)``: a constant whose derivative cannot be evaluated."""

    def __init__(self, name, g, b, c):
        SimulationNode.__init__(self, name, 1.0, g=jnp.float32(g), b=jnp.float32(b),
                                c=jnp.float32(c))
        self._n = 1

    def update(self, state, boundary_inputs, dt, *, params=None):
        p = self.params if params is None else {**self.params, **params}
        return {"x": p["b"] + p["g"] * boundary_inputs["u"] + _no_derivative(p["c"])}


def _opaque_jacobian(**kw):
    """``jacfwd`` of both fields in ``(A.b, A.c)``: one column per constant."""
    gm = GraphManager()
    gm.add_node(_OpaqueRelay("A", 0.5, 1.0, 1.0))
    gm.add_node(_Relay("B", 0.8, 0.5, 1))
    gm.add_edge("A", "B", "x", "u")
    gm.add_edge("B", "A", "x", "u")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)    # solver="fori"
        gm.add_coupling_group(["A", "B"], max_iterations=60, tolerance=1e-7, **kw)
    gm.compile()
    step, state, ext, params = gm._compiled_step, gm._state, gm._default_external_inputs(), gm.params

    def fields(bc):
        p = jax.tree.map(lambda v: v, params)
        p["nodes"]["A"]["b"], p["nodes"]["A"]["c"] = bc[0], bc[1]
        out = step(state, ext, p)
        return jnp.concatenate([out["A"]["x"], out["B"]["x"]])

    bc = jnp.stack([params["nodes"]["A"]["b"], params["nodes"]["A"]["c"]])
    return np.asarray(jax.jacfwd(fields)(bc))


def test_a_constant_with_no_derivative_gives_a_nan_column_and_leaves_the_others_right():
    """The reported case: a ``custom_jvp`` identity with a NaN derivative in a
    two-relay group.  ``jacfwd`` solves one tangent per constant, so the
    column of the opaque constant is NaN (it read 0.0 under the default
    solve) and the bias's column is still the dense and fori answer."""
    gmres = _opaque_jacobian()
    for ref in (_opaque_jacobian(linear_solver="dense"), _opaque_jacobian(solver="fori")):
        _assert_nan_everywhere_and_where_the_reference_is_not_finite(gmres[:, 1], ref[:, 1],
                                                                     what="the opaque column")
        np.testing.assert_allclose(gmres[:, 0], ref[:, 0], rtol=1e-4)
        assert np.all(np.isfinite(gmres[:, 0]))


# ---------------------------------------------------------------------------
# _ift_linear_solve itself, every backend (BiCGStab is reachable only here)
# ---------------------------------------------------------------------------

def _operator(n):
    """``v -> v - J v`` for a fixed contraction ``J`` (spectral radius ~ 0.5)."""
    rng = np.random.default_rng(n)
    j = rng.standard_normal((n, n)).astype(np.float32)
    j = jnp.asarray(0.5 * j / np.max(np.abs(np.linalg.eigvals(j))))
    return lambda v: v - j @ v


@functools.lru_cache(maxsize=None)
def _private_solve(solver, n):
    """``(primal, tangent, cotangent)`` of ``_ift_linear_solve``, jitted once."""
    mv = _operator(n)
    b0 = jnp.linspace(1.0, 2.0, n, dtype=jnp.float32)

    def solve(b):
        return _ift_linear_solve(mv, b, solver)

    primal = jax.jit(solve)
    tangent = jax.jit(lambda t: jax.jvp(solve, (b0,), (t,))[1])
    cotangent = jax.jit(lambda ct: jax.vjp(solve, b0)[1](ct)[0])
    return primal, tangent, cotangent


@pytest.mark.parametrize("n", [4, 60])
@pytest.mark.parametrize("solver", ["gmres", "bicgstab"])
def test_the_private_solve_answers_a_non_finite_rhs_with_nan_on_every_backend(solver, n):
    """Primal, tangent and cotangent, against ``"dense"``.  At ``n = 60`` a
    "failed" non-finite solve would raise the adjoint error; it must not."""
    dense = _private_solve("dense", n)
    krylov = _private_solve(solver, n)
    for bad, name in zip(BAD, BAD_IDS):
        v = jnp.linspace(1.0, 2.0, n, dtype=jnp.float32).at[n // 2].set(bad)
        for kind, k_fn, d_fn in zip(("primal", "tangent", "cotangent"), krylov, dense):
            _assert_nan_everywhere_and_where_the_reference_is_not_finite(
                k_fn(v), d_fn(v), what=(solver, n, name, kind))


def test_a_zero_and_a_finite_rhs_are_answered_as_before():
    """The neighbours of the new branch: a zero rhs is exact zeros (not NaN),
    and a finite one is the dense answer."""
    n = 4
    primal = {s: _private_solve(s, n)[0] for s in ("gmres", "bicgstab", "dense")}
    for solver in ("gmres", "bicgstab"):
        z = np.asarray(primal[solver](jnp.zeros(n, jnp.float32)))
        np.testing.assert_array_equal(z, np.zeros(n, np.float32))
    b = jnp.linspace(-1.0, 1.0, n, dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(primal["gmres"](b)), np.asarray(primal["dense"](b)),
                               rtol=1e-4)


def test_under_vmap_only_the_non_finite_column_is_nan():
    """``jacfwd`` and ``jacrev`` batch one solve per column: the check is per
    column, so a NaN column leaves its neighbours' answers alone."""
    n = 6
    mv = _operator(n)
    cols = jnp.stack([jnp.linspace(1.0, 2.0, n, dtype=jnp.float32),
                      jnp.ones(n, jnp.float32).at[3].set(jnp.nan),
                      jnp.linspace(-2.0, 1.0, n, dtype=jnp.float32)])
    dense = _private_solve("dense", n)[0]       # the same operator: ``_operator`` is seeded by n
    for solver in ("gmres", "dense"):
        out = np.asarray(jax.jit(jax.vmap(lambda b, s=solver: _ift_linear_solve(mv, b, s)))(cols))
        assert np.all(np.isnan(out[1])), solver
        for i in (0, 2):
            np.testing.assert_allclose(out[i], np.asarray(dense(cols[i])),
                                       rtol=1e-4, err_msg=f"{solver} column {i}")


# ---------------------------------------------------------------------------
# The public solvers: ift_linear_solve and sharded_cg / sharded_gmres
# ---------------------------------------------------------------------------

N = 8
_M = np.random.default_rng(0).standard_normal((N, N)).astype(np.float32)
A_SPD = jnp.asarray(_M @ _M.T + N * np.eye(N, dtype=np.float32))
B0 = jnp.linspace(1.0, 2.0, N, dtype=jnp.float32)


def _mv(v):
    return A_SPD @ v


def _bad_vectors():
    for bad, name in zip(BAD, BAD_IDS):
        yield name, B0.at[N // 2].set(bad)


@functools.lru_cache(maxsize=None)
def _dense_reference():
    """``(primal, tangent, cotangent)`` of the dense solve, the reference."""
    def solve(b):
        return ift_linear_solve(_mv, b, solver="dense")
    return (jax.jit(solve), jax.jit(lambda t: jax.jvp(solve, (B0,), (t,))[1]),
            jax.jit(lambda ct: jax.vjp(solve, B0)[1](ct)[0]))


@pytest.mark.parametrize("atol", [None, 1e-8], ids=["relative", "absolute"])
@pytest.mark.parametrize("solver", ["gmres", "cg"])
def test_ift_linear_solve_answers_a_non_finite_rhs_with_nan(solver, atol):
    """Before the fix: zeros with no error for a relative ``atol`` (both
    solvers) and for CG under an explicit one; lineax's "non-finite output"
    error for GMRES under an explicit one."""
    def solve(b):
        return ift_linear_solve(_mv, b, solver=solver, atol=atol)

    fns = (jax.jit(solve), jax.jit(lambda t: jax.jvp(solve, (B0,), (t,))[1]),
           jax.jit(lambda ct: jax.vjp(solve, B0)[1](ct)[0]))
    for name, v in _bad_vectors():
        for kind, fn, ref in zip(("primal", "tangent", "cotangent"), fns, _dense_reference()):
            _assert_nan_everywhere_and_where_the_reference_is_not_finite(
                fn(v), ref(v), what=(solver, atol, name, kind))


SHARDED = {
    "sharded_cg/lineax": (sharded_cg, "lineax"),
    "sharded_cg/loop": (sharded_cg, "loop"),
    "sharded_gmres/lineax": (sharded_gmres, "lineax"),
    "sharded_gmres/loop": (sharded_gmres, "loop"),
}


@pytest.mark.parametrize("atol", [None, 1e-8], ids=["relative", "absolute"])
@pytest.mark.parametrize("name", list(SHARDED))
def test_a_sharded_solve_of_a_non_finite_rhs_is_nan_and_not_converged(name, atol):
    """``value`` NaN, ``residual_norm`` NaN, ``converged=False``.  Before the
    fix CG returned zeros on both backends (``converged=True`` on lineax for
    NaN and on the loop for ``+-inf``), and lineax GMRES zeros with
    ``converged=True`` under the default relative tolerance."""
    fn, backend = SHARDED[name]

    @jax.jit
    def run(b):
        r = fn(_mv, b, backend=backend, atol=atol)
        return r.value, r.converged, r.residual_norm

    dense = _dense_reference()[0]
    for bad_name, v in _bad_vectors():
        value, converged, residual_norm = run(v)
        _assert_nan_everywhere_and_where_the_reference_is_not_finite(
            value, dense(v), what=(name, atol, bad_name))
        assert not bool(converged), (name, atol, bad_name)
        assert np.isnan(float(residual_norm)), (name, atol, bad_name, residual_norm)


@pytest.mark.parametrize("name", list(SHARDED))
def test_the_sharded_tangent_and_cotangent_of_a_non_finite_rhs_are_nan(name):
    """``differentiable=True``: the tangent and adjoint solves are the same
    framed solve on their own right-hand side."""
    fn, backend = SHARDED[name]

    def solve(b):
        return fn(_mv, b, backend=backend, differentiable=True).value

    tangent = jax.jit(lambda t: jax.jvp(solve, (B0,), (t,))[1])
    cotangent = jax.jit(lambda ct: jax.vjp(solve, B0)[1](ct)[0])
    @jax.jit
    def primal(b):
        r = fn(_mv, b, backend=backend, differentiable=True)
        return r.value, r.converged, r.residual_norm

    _, d_tan, d_cot = _dense_reference()
    for bad_name, v in _bad_vectors():
        _assert_nan_everywhere_and_where_the_reference_is_not_finite(
            tangent(v), d_tan(v), what=(name, bad_name, "tangent"))
        _assert_nan_everywhere_and_where_the_reference_is_not_finite(
            cotangent(v), d_cot(v), what=(name, bad_name, "cotangent"))
        value, converged, residual_norm = primal(v)
        assert np.all(np.isnan(np.asarray(value))) and not bool(converged), (name, bad_name)
        assert np.isnan(float(residual_norm)), (name, bad_name)


def test_a_loop_cg_handed_a_non_finite_rhs_does_not_iterate_from_its_start():
    """The start is replaced with zeros beside the rhs: from a non-zero ``x0``
    the loop CG would otherwise run toward the solution of a zero rhs, on a
    tolerance of zero, to its iteration budget."""
    x0 = jnp.linspace(-1.0, 1.0, N, dtype=jnp.float32)
    for _, v in _bad_vectors():
        r = sharded_cg(_mv, v, x0=x0, backend="loop")
        assert int(r.iters) == 0 and np.all(np.isnan(np.asarray(r.value))), r
    r = sharded_cg(_mv, B0, x0=x0, backend="loop")
    assert bool(r.converged) and int(r.iters) > 0


# ---------------------------------------------------------------------------
# No backend sees the non-finite rhs
# ---------------------------------------------------------------------------

def _refuse_a_non_finite_rhs(real, position):
    """``real`` with an ``equinox.error_if`` on its right-hand side argument."""
    import equinox as eqx

    def spy(*args, **kwargs):
        args = list(args)
        args[position] = eqx.error_if(
            args[position], jnp.logical_not(jnp.all(jnp.isfinite(args[position]))),
            "a non-finite right-hand side reached the Krylov backend")
        return real(*args, **kwargs)

    return spy


def test_no_krylov_backend_is_handed_a_non_finite_rhs(monkeypatch):
    """Each backend gets zeros in place of a non-finite rhs, so none iterates
    on NaN (an explicit ``atol`` ran lineax GMRES to its step budget and
    raised), and no backend's status can route the solve to the dense
    re-solve or the adjoint error.  Spies that raise on a non-finite vector
    stand in for ``lineax.linear_solve`` and the sharded loop backends."""
    import lineax as lx

    from maddening.cloud.multigpu import iterative_solver

    monkeypatch.setattr(lx, "linear_solve", _refuse_a_non_finite_rhs(lx.linear_solve, 1))
    monkeypatch.setattr(iterative_solver, "_cg_loop",
                        _refuse_a_non_finite_rhs(iterative_solver._cg_loop, 1))
    monkeypatch.setattr(iterative_solver, "_gmres_loop",
                        _refuse_a_non_finite_rhs(iterative_solver._gmres_loop, 1))
    mv5 = _operator(5)
    solves = {
        "_ift_linear_solve/gmres": lambda b: _ift_linear_solve(mv5, b, "gmres"),
        "_ift_linear_solve/bicgstab": lambda b: _ift_linear_solve(mv5, b, "bicgstab"),
        "ift_linear_solve/gmres+atol": lambda b: ift_linear_solve(_mv, b, solver="gmres", atol=1e-8),
        "ift_linear_solve/cg": lambda b: ift_linear_solve(_mv, b, solver="cg"),
    }
    for name, (fn, backend) in SHARDED.items():
        solves[name] = lambda b, fn=fn, backend=backend: fn(_mv, b, backend=backend, atol=1e-8).value
    for name, solve in solves.items():
        size = 5 if name.startswith("_ift") else N
        jitted = jax.jit(solve)
        for bad in BAD:
            out = jitted(jnp.ones(size, jnp.float32).at[2].set(bad))
            assert np.all(np.isnan(np.asarray(out))), (name, bad)
